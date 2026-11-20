//! Session operations beyond edits: applying solves as a new snapshot,
//! validation through the engine, autosave, the schema version check and
//! parameter reads.

use serde_json::{Value, json};

use mklight_core::materials::MaterialLibrary;
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, Solve, samples};
use mklight_engine::backend::Backend;
use mklight_engine::project::{CatalogFormat, CatalogRef, Project};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, EngineError};

/// An engine on the CPU reference: these tests do not depend on a device.
fn engine() -> Engine {
    Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    })
}

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
    let e = engine();
    let mut sys = samples::cooke_triplet();
    let img = sys.surfaces.len() - 1;
    let j = img - 1;
    sys.surfaces[j].thickness += 2.5; // defocus
    sys.solves = vec![Solve::marginal_ray_height(sys.surfaces[j].id, 0.0)];
    let mut s = e.session_from_project(Project::new(sys.clone())).unwrap();
    let h0 = s.snapshot_hash();

    // The dry run reports the value without changing anything.
    let preview = s
        .run_analysis(
            &AnalysisRequest::new("solves", Value::Null),
            &JobContext::new(),
        )
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
    assert!(
        s.apply_solves(&json!({"configuration": 0}), &JobContext::new())
            .is_err()
    );
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
    assert!(
        s.apply_solves(&Value::Null, &ctx)
            .unwrap_err()
            .is_cancelled()
    );
    assert_eq!(s.snapshot_hash(), h3);
}

/// Solving without the session (prepare, run, commit) commits only onto
/// the snapshot it started from: an edit in between supersedes it.
#[test]
fn solves_prepared_from_a_stale_snapshot_are_not_committed() {
    let e = engine();
    let mut sys = samples::cooke_triplet();
    let j = sys.surfaces.len() - 2;
    sys.surfaces[j].thickness += 2.5;
    sys.solves = vec![Solve::marginal_ray_height(sys.surfaces[j].id, 0.0)];
    let mut s = e.session_from_project(Project::new(sys)).unwrap();
    let h0 = s.snapshot_hash();
    let p = s.prepare_solves(&Value::Null).unwrap();
    assert_eq!(p.base_hash(), h0);
    let (solved, _) = p.run(&JobContext::new()).unwrap();
    // Meanwhile the user edits another surface.
    let path = ParamPath::Thickness {
        surface: s.system().surfaces[1].id,
    };
    let t1 = s.system().surfaces[1].thickness;
    let edit = |value| mklight_engine::ParamEdit {
        path: path.clone(),
        value,
    };
    s.apply_edits(&[edit(t1 + 1.0)]).unwrap();
    let h1 = s.snapshot_hash();
    let err = s.commit_solves(&p, solved.clone()).unwrap_err();
    assert!(err.is_cancelled(), "{err}");
    assert_eq!(s.snapshot_hash(), h1);
    // Reverting the edit restores the base snapshot: the commit applies.
    s.apply_edits(&[edit(t1)]).unwrap();
    assert_eq!(s.snapshot_hash(), h0);
    let h2 = s.commit_solves(&p, solved).unwrap();
    assert_ne!(h2, h0);
    // Bad settings are refused before anything runs.
    assert!(matches!(
        s.prepare_solves(&json!({"tolerance": 2.0})),
        Err(EngineError::InvalidSettings(_))
    ));
}

#[test]
fn validate_reports_engine_checks() {
    let e = engine();
    let mut sys = samples::achromatic_doublet();
    sys.aperture = mklight_core::ApertureSpec::ImageSpaceFNumber { value: 0.0 };
    let err = e.session_from_project(Project::new(sys)).err().unwrap();
    assert!(err.to_string().contains("invalid_aperture_value"), "{err}");

    // Structurally valid, but afocal: an F/# cannot define its pupil.
    let mut afocal = samples::achromatic_doublet();
    for surface in &mut afocal.surfaces {
        surface.shape = mklight_core::SurfaceShape::Plane;
    }
    afocal.aperture = mklight_core::ApertureSpec::ImageSpaceFNumber { value: 5.0 };
    let s = e.session_from_project(Project::new(afocal)).unwrap();
    let v = s.validate().unwrap();
    assert_eq!(v["ok"], false);
    assert!(
        v["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i["code"] == "paraxial"),
        "{v}"
    );

    // A clipped marginal ray is a warning, so the model remains valid.
    let mut clipped = samples::achromatic_doublet();
    clipped.surfaces[2].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 0.01 };
    let v = e
        .session_from_project(Project::new(clipped))
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(v["ok"], true, "{v}");
    assert!(
        v["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i["code"] == "marginal_ray" && i["severity"] == "warning"),
        "{v}"
    );
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
    let e = engine();
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
    assert_eq!(
        std::fs::read(&p).unwrap(),
        saved,
        "the project file is untouched"
    );
    let r = e.open_session(&a).unwrap();
    assert_eq!(r.snapshot_hash(), s.snapshot_hash());
    std::fs::remove_dir_all(&d).ok();
}

/// A front end's whole project: every section it keeps, and a top-level
/// key the engine does not know.
fn front_end_project(system: mklight_core::OpticalSystem) -> Project {
    let mut doc = Project::new(system).to_value().unwrap();
    doc["metadata"]["author"] = json!("Optics team");
    doc["views"] = json!([{"key": "spot-1", "analysis": "spot_diagram",
        "settings": {"field": 2}, "ui": {"split": 0.4, "tab": "plot"}}]);
    doc["merit"] = json!({"operands": [{"kind": "effl", "target": 100.0, "weight": 1.0}]});
    doc["variables"] = json!([{"param": "curvature", "surface": 1}]);
    doc["tolerances"] = json!({"operands": [{"kind": "tthi", "min": -0.05, "max": 0.05}]});
    doc["layout"] = json!({"window": [10, 20, 1200, 800], "panels": ["lde", "spot-1"]});
    Project::from_value(doc).unwrap()
}

/// The project without the fields every save stamps.
fn unstamped(p: &Project) -> Value {
    let mut v = p.to_value().unwrap();
    v["metadata"]["modified"] = Value::Null;
    v["metadata"]["engine_version"] = Value::Null;
    v
}

#[test]
fn set_project_keeps_every_section_through_autosave_and_recovery() {
    let d = temp_dir("set-project");
    let e = engine();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    let p = d.join("lens.mklight");
    s.save(Some(&p)).unwrap();
    assert!(!s.is_dirty());

    let mut sys = samples::singlet();
    sys.surfaces[1].thickness += 1.0;
    let wanted = front_end_project(sys);
    let h = s.set_project(wanted.clone()).unwrap();
    assert_eq!(h, s.snapshot_hash());
    assert!(s.is_dirty());
    assert_eq!(s.system(), &wanted.system);
    assert_eq!(unstamped(s.project()), unstamped(&wanted));

    let a = s.autosave(None).unwrap();
    let r = e.open_session(&a).unwrap();
    assert_eq!(r.snapshot_hash(), s.snapshot_hash());
    assert_eq!(unstamped(r.project()), unstamped(&wanted));
    assert_eq!(r.project().extra["layout"], wanted.extra["layout"]);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn set_project_is_atomic_and_leaves_catalogs_and_a_clean_session_alone() {
    let e = engine();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    s.set_project(front_end_project(samples::singlet()))
        .unwrap();
    let before = s.project().clone();
    let h = s.snapshot_hash();

    // An invalid system changes nothing, the other sections included.
    let mut bad = front_end_project(samples::singlet());
    bad.merit = json!("other");
    bad.system.surfaces[1].thickness = f64::NAN;
    assert!(s.set_project(bad).is_err());
    assert_eq!(s.project(), &before);
    assert_eq!(s.snapshot_hash(), h);

    // The same document again is no change; its catalogs are ignored.
    let d = temp_dir("set-project-clean");
    s.save(Some(&d.join("x.mklight"))).unwrap();
    let mut same = s.project().clone();
    same.catalogs.push(mklight_engine::project::CatalogRef {
        name: "ELSEWHERE".into(),
        path: None,
        relative_path: None,
        format: Default::default(),
    });
    assert_eq!(s.set_project(same).unwrap(), h);
    assert!(!s.is_dirty());
    assert!(s.project().catalogs.is_empty());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn files_at_another_schema_version_are_refused() {
    let d = temp_dir("schema");
    let e = engine();
    let p = d.join("old.mklight");
    // A bare optical system has no schema_version.
    let bare = serde_json::to_string(&samples::singlet()).unwrap();
    std::fs::write(&p, &bare).unwrap();
    let err = e.open_session(&p).err().expect("refused");
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert_eq!(std::fs::read_to_string(&p).unwrap(), bare, "left untouched");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn parameter_values() {
    let e = engine();
    for system in [
        samples::singlet(),
        samples::toroidal_element(),
        samples::biconic_element(),
    ] {
        let s = e.session_from_project(Project::new(system)).unwrap();
        let id = s.system().surfaces[1].id;
        let t = s
            .param_value(&ParamPath::Thickness { surface: id })
            .unwrap();
        assert_eq!(t, Some(s.system().surfaces[1].thickness));
        let layout = s
            .run_analysis(
                &AnalysisRequest::new("layout", Value::Null),
                &JobContext::new(),
            )
            .unwrap();
        let radius = layout.json["surfaces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["index"] == 1)
            .unwrap()["semi_diameter"]
            .as_f64()
            .unwrap();
        // An automatic semi-diameter reads as the radius layout draws.
        assert_eq!(
            s.param_value(&ParamPath::SemiDiameter { surface: id })
                .unwrap(),
            Some(radius)
        );
        // A surface that does not exist is an error.
        assert!(
            s.param_value(&ParamPath::Thickness {
                surface: mklight_core::SurfaceId(999)
            })
            .is_err()
        );
    }
}

/// A synthetic AGF catalog with two made-up glasses; catalogs are named
/// after the file stem.
const AGF: &str = "NM SYNCROWN 2 517642 1.5168 64.17 0 0 0\r\n\
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\n\
LD 0.3 2.5\r\n\
NM SYNFLINT 2 620364 1.62004 36.37 0 0 0\r\n\
CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r\n\
LD 0.32 2.5\r\n";

fn reference(name: &str, path: Option<&str>, rel: Option<&str>) -> CatalogRef {
    CatalogRef {
        name: name.into(),
        path: path.map(Into::into),
        relative_path: rel.map(Into::into),
        format: CatalogFormat::Agf,
    }
}

fn resolves(s: &mklight_engine::Session, catalog: &str) -> bool {
    s.library().find_glass(Some(catalog), "SYNCROWN").is_ok()
}

fn warns_about(s: &mklight_engine::Session, catalog: &str) -> bool {
    s.open_warnings()
        .iter()
        .any(|w| w.contains(&format!("'{catalog}'")))
}

/// The front end resolved the catalog files itself: the one on an offline
/// share was not found, so only the local one was loaded. Its reference
/// still survives autosave, recovery and save, reported as missing.
#[test]
fn set_project_with_catalogs_keeps_an_unloaded_reference_as_missing() {
    let d = temp_dir("catalog-refs-missing");
    let proj = d.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let local = proj.join("LOCAL.agf");
    std::fs::write(&local, AGF).unwrap();
    let offline = d.join("share").join("NETCAT.agf"); // never exists
    let file = proj.join("lens.mklight");

    let e = engine();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    s.save(Some(&file)).unwrap();
    s.load_catalog_file(&local).unwrap();

    let mut doc = s.project().clone();
    let net = reference(
        "NETCAT",
        Some(offline.to_str().unwrap()),
        Some("../optics/NETCAT.agf"),
    );
    let loc = reference("LOCAL", Some(local.to_str().unwrap()), Some("LOCAL.agf"));
    doc.catalogs = vec![net.clone(), loc.clone()];
    s.set_project_with_catalogs(doc).unwrap();
    assert!(s.is_dirty());
    assert_eq!(s.project().catalogs, vec![net.clone(), loc.clone()]);
    assert!(resolves(&s, "LOCAL"));
    assert!(warns_about(&s, "NETCAT"), "{:?}", s.open_warnings());
    assert!(!warns_about(&s, "LOCAL"));

    // Autosave, recover, save: the reference is still there, as written.
    let a = s.autosave(None).unwrap();
    let mut r = e.open_session(&a).unwrap();
    assert!(warns_about(&r, "NETCAT"), "{:?}", r.open_warnings());
    assert!(resolves(&r, "LOCAL"));
    assert_eq!(r.project().catalogs[0], net);
    r.save(Some(&file)).unwrap();
    let saved = Project::load(&file).unwrap();
    let n = saved.catalogs.iter().find(|c| c.name == "NETCAT").unwrap();
    assert_eq!(n, &net);
    let again = e.open_session(&file).unwrap();
    assert!(warns_about(&again, "NETCAT"));
    assert!(resolves(&again, "LOCAL"));
    std::fs::remove_dir_all(&d).ok();
}

/// A relative path the front end supplies is written as given, so that the
/// project still finds the catalog after its folder moves.
#[test]
fn set_project_with_catalogs_keeps_the_given_relative_path() {
    let d = temp_dir("catalog-refs-relative");
    let proj = d.join("proj");
    let optics = d.join("optics");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::create_dir_all(&optics).unwrap();
    let cat = optics.join("SHARED.agf");
    std::fs::write(&cat, AGF).unwrap();
    let file = proj.join("lens.mklight");

    let e = engine();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    s.save(Some(&file)).unwrap();
    // Loaded by path: the engine records no relative path.
    s.load_catalog_file(&cat).unwrap();
    assert_eq!(s.project().catalogs[0].relative_path, None);

    let given = reference(
        "SHARED",
        Some(cat.to_str().unwrap()),
        Some("../optics/SHARED.agf"),
    );
    let mut doc = s.project().clone();
    doc.catalogs = vec![given.clone()];
    s.set_project_with_catalogs(doc).unwrap();
    assert!(resolves(&s, "SHARED"));
    assert!(s.open_warnings().is_empty());

    // Autosave and save both write it as given (the file lies outside the
    // project folder, where a computed relative path would be none).
    let a = s.autosave(None).unwrap();
    assert_eq!(Project::load(&a).unwrap().catalogs, vec![given.clone()]);
    s.save(None).unwrap();
    assert_eq!(Project::load(&file).unwrap().catalogs, vec![given.clone()]);

    // Move the whole folder: the absolute path fails, the relative one works.
    let moved = d.with_extension("moved");
    std::fs::rename(&d, &moved).unwrap();
    for f in ["lens.mklight", "lens.mklight.autosave"] {
        let r = e.open_session(&moved.join("proj").join(f)).unwrap();
        assert!(r.open_warnings().is_empty(), "{f}: {:?}", r.open_warnings());
        assert!(resolves(&r, "SHARED"), "{f}");
        assert_eq!(
            r.project().catalogs[0].relative_path.as_deref(),
            Some("../optics/SHARED.agf")
        );
    }

    // Saving into another folder recomputes it (the given one would point
    // elsewhere): the file lies outside that folder, so there is none.
    let other = moved.join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    let mut r = e
        .open_session(&moved.join("proj").join("lens.mklight"))
        .unwrap();
    let mut doc = r.project().clone();
    doc.catalogs[0].relative_path = Some("../optics/SHARED.agf".into());
    r.set_project_with_catalogs(doc).unwrap();
    r.save(Some(&other.join("copy.mklight"))).unwrap();
    assert_eq!(r.project().catalogs[0].relative_path, None);
    std::fs::remove_dir_all(&moved).ok();
}

/// A catalog the project records but the given list leaves out is removed
/// as by `remove_catalog`; catalogs added in memory without a reference
/// stay; errors change nothing.
#[test]
fn set_project_with_catalogs_removes_recorded_catalogs_left_out() {
    let d = temp_dir("catalog-refs-dropped");
    let keep = d.join("KEEP.agf");
    let gone = d.join("GONE.agf");
    std::fs::write(&keep, AGF).unwrap();
    std::fs::write(&gone, AGF).unwrap();
    let e = engine();
    let mut s = e
        .session_from_project(Project::new(samples::singlet()))
        .unwrap();
    s.load_catalog_file(&keep).unwrap();
    s.load_catalog_file(&gone).unwrap();
    let mut mem = s.library().catalogs().last().unwrap().as_ref().clone();
    mem.name = "INMEMORY".into();
    s.add_catalog(mem, None);
    assert!(resolves(&s, "GONE") && resolves(&s, "INMEMORY"));

    let kept = s.project().catalogs[0].clone();
    assert_eq!(kept.name, "KEEP");
    let before = s.project().clone();
    let library = s.library_hash().to_string();

    // Duplicate names are refused and leave everything as it was.
    let mut dup = s.project().clone();
    dup.catalogs = vec![kept.clone(), reference("keep", None, None)];
    assert!(s.set_project_with_catalogs(dup).is_err());
    assert_eq!(s.project(), &before);
    assert_eq!(s.library_hash(), library);

    let mut doc = s.project().clone();
    doc.catalogs = vec![kept.clone()];
    s.set_project_with_catalogs(doc).unwrap();
    assert_eq!(s.project().catalogs, vec![kept]);
    assert!(resolves(&s, "KEEP"));
    assert!(!resolves(&s, "GONE"));
    assert!(resolves(&s, "INMEMORY"));
    assert!(s.open_warnings().is_empty());
    s.restore_catalog("GONE").unwrap();
    assert!(resolves(&s, "GONE"));
    assert_eq!(s.project().catalogs.len(), 2);

    // Without catalogs (`set_project`), the references are untouched.
    let mut doc = s.project().clone();
    doc.catalogs.clear();
    s.set_project(doc).unwrap();
    assert_eq!(s.project().catalogs.len(), 2);
    assert!(resolves(&s, "GONE"));
    std::fs::remove_dir_all(&d).ok();
}
