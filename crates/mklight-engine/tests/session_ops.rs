//! Session operations beyond edits: applying solves as a new snapshot,
//! validation through the engine, autosave, file migration and parameter
//! reads.

use serde_json::{Value, json};

use mklight_core::materials::MaterialLibrary;
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, Solve, samples};
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineError};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "mklight-engine-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn apply_solves_commits_a_new_snapshot() {
    let e = Engine::default();
    let mut sys = samples::cooke_triplet();
    let img = sys.surfaces.len() - 1;
    let j = img - 1;
    sys.surfaces[j].thickness += 2.5; // defocus
    sys.solves = vec![Solve::marginal_ray_height(sys.surfaces[j].id, 0.0)];
    let mut s = e.session_from_project(Project::new(sys.clone())).unwrap();
    let h0 = s.snapshot_hash();

    // The dry run reports the value without changing anything.
    let preview = s
        .run_analysis(&AnalysisRequest::new("solves", Value::Null), &JobContext::new())
        .unwrap();
    assert_eq!(s.snapshot_hash(), h0);
    let solved = preview.json["outcomes"][0]["value"].as_f64().unwrap();

    let (h1, report) = s.apply_solves(&Value::Null, &JobContext::new()).unwrap();
    assert_ne!(h1, h0);
    assert_eq!(s.snapshot_hash(), h1);
    assert!(s.is_dirty());
    assert_eq!(report["outcomes"][0]["value"].as_f64().unwrap(), solved);
    assert_eq!(s.system().surfaces[j].thickness, solved);
    // Same as the solver called directly.
    let (direct, _) = mklight_trace::apply_solves(&sys, &MaterialLibrary::with_builtin()).unwrap();
    assert_eq!(direct.surfaces[j].thickness, solved);
    assert!(report["outcomes"][0]["residual"].as_f64().unwrap().abs() < 1e-9);
    // Solving again changes nothing.
    let (h2, _) = s.apply_solves(&Value::Null, &JobContext::new()).unwrap();
    assert_eq!(h2, h1);
    // Bad settings and configuration solves are refused, leaving the session.
    assert!(matches!(
        s.apply_solves(&json!({"tolerance": 2.0}), &JobContext::new()),
        Err(EngineError::InvalidSettings(_))
    ));
    assert!(s.apply_solves(&json!({"configuration": 0}), &JobContext::new()).is_err());
    assert_eq!(s.snapshot_hash(), h1);
    // A cancelled solve is an error and leaves the session unchanged.
    s.apply_edits(&[mklight_engine::ParamEdit {
        path: ParamPath::Thickness {
            surface: s.system().surfaces[j].id,
        },
        value: solved + 1.0,
    }])
    .unwrap();
    let h3 = s.snapshot_hash();
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(s.apply_solves(&Value::Null, &ctx).unwrap_err().is_cancelled());
    assert_eq!(s.snapshot_hash(), h3);
}

#[test]
fn validate_reports_engine_checks() {
    let e = Engine::default();
    let mut sys = samples::achromatic_doublet();
    sys.aperture = mklight_core::ApertureSpec::ImageSpaceFNumber { value: 0.0 };
    let s = e.session_from_project(Project::new(sys)).unwrap();
    let v = s.validate().unwrap();
    assert_eq!(v["ok"], false);
    let codes: Vec<&str> = v["issues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["code"].as_str().unwrap())
        .collect();
    assert!(codes.contains(&"aperture_value") && codes.contains(&"paraxial"), "{codes:?}");
    let ok = e
        .session_from_project(Project::new(samples::achromatic_doublet()))
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(ok["ok"], true, "{ok}");
}

#[test]
fn autosave_writes_a_recoverable_copy_without_touching_the_session() {
    let d = temp_dir("autosave");
    let e = Engine::default();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    assert!(s.autosave(None).is_err(), "untitled needs a path");
    let p = d.join("lens.mklight");
    s.save(Some(&p)).unwrap();
    let saved = std::fs::read(&p).unwrap();
    s.apply_edits(&[mklight_engine::ParamEdit {
        path: ParamPath::ApertureValue,
        value: 12.0,
    }])
    .unwrap();
    let a = s.autosave(None).unwrap();
    assert_eq!(a, d.join("lens.mklight.autosave"));
    assert!(s.is_dirty());
    assert_eq!(s.path(), Some(p.as_path()));
    assert_eq!(std::fs::read(&p).unwrap(), saved, "the project file is untouched");
    let r = e.open_session(&a).unwrap();
    assert_eq!(r.snapshot_hash(), s.snapshot_hash());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn migrate_file_upgrades_with_a_backup() {
    let d = temp_dir("migrate");
    let e = Engine::default();
    let p = d.join("old.mklight");
    // Version 0: a bare optical system.
    let bare = serde_json::to_string(&samples::singlet()).unwrap();
    std::fs::write(&p, &bare).unwrap();
    let dry = e.migrate_file(&p, true, true).unwrap();
    assert!(!dry.migrated && dry.dry_run && dry.from_version == 0);
    assert_eq!(std::fs::read_to_string(&p).unwrap(), bare);
    let m = e.migrate_file(&p, true, false).unwrap();
    assert!(m.migrated);
    assert_eq!((m.from_version, m.to_version), (0, 1));
    let b = m.backup.unwrap();
    assert_eq!(std::fs::read_to_string(&b).unwrap(), bare);
    let again = e.migrate_file(&p, true, false).unwrap();
    assert!(!again.migrated && again.backup.is_none());
    let s = e.open_session(&p).unwrap();
    assert!(!s.migration().migrated());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn parameter_values() {
    let e = Engine::default();
    let s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    let id = s.system().surfaces[1].id;
    let t = s
        .param_value(&ParamPath::Thickness { surface: id })
        .unwrap();
    assert_eq!(t, Some(s.system().surfaces[1].thickness));
    // An automatic semi-diameter has no value, but exists.
    assert_eq!(
        s.param_value(&ParamPath::SemiDiameter { surface: id }).unwrap(),
        None
    );
    // A surface that does not exist is an error.
    assert!(
        s.param_value(&ParamPath::Thickness {
            surface: mklight_core::SurfaceId(999)
        })
        .is_err()
    );
}
