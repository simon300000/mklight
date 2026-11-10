use mklight_core::materials::{MaterialOffset, agf, builtin, lines};
use mklight_core::{JobContext, MaterialRef, samples};
use mklight_engine::backend::Backend;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig};
use serde_json::{Value, json};
use std::sync::Arc;

/// An engine on the CPU reference: these tests do not depend on a device.
fn engine() -> Engine {
    Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    })
}

#[test]
fn agf_and_json_catalogue_projects_keep_reference_offsets() {
    let dir = std::env::temp_dir().join(format!(
        "mklight-offset-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut cat = builtin::catalog();
    cat.name = "OFFSET".into();
    let (text, warnings) = agf::write_agf(&agf::AgfCatalog::from_catalog(cat.clone()));
    assert!(warnings.is_empty());
    let engine = engine();
    for (ext, data) in [
        ("agf", text),
        ("json", serde_json::to_string(&cat).unwrap()),
    ] {
        let cat_path = dir.join(format!("offset.{ext}"));
        std::fs::write(&cat_path, data).unwrap();
        let mut session = engine
            .session_from_project(Project::new(samples::singlet()))
            .unwrap();
        let loaded = session.load_catalog_file(&cat_path).unwrap();
        let mut sys = session.system().clone();
        sys.surfaces[1].material = MaterialRef::Catalog {
            catalog: Some(loaded.name),
            name: "N-BK7".into(),
            offset: MaterialOffset {
                dn_d: 0.002,
                dv_d: -3.0,
            },
        };
        session.set_system(sys).unwrap();
        let before = session.resolved().unwrap();
        let path = dir.join(format!("{ext}.mklight"));
        session.save(Some(&path)).unwrap();
        let reopened = engine.open_session(&path).unwrap();
        assert!(reopened.open_warnings().is_empty());
        assert_eq!(session.snapshot_hash(), reopened.snapshot_hash());
        assert_eq!(session.system(), reopened.system());
        let after = reopened.resolved().unwrap();
        assert_eq!(before.surfaces[1].n_after, after.surfaces[1].n_after);
        assert!((after.n_after_at(1, lines::D) - 1.5188000345005885).abs() < 2e-15);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn glass_map_info_and_substitution_use_the_offset_glass() {
    let engine = engine();
    let mut sys = samples::singlet();
    let offset = json!({"dn_d": 0.002, "dv_d": -3.0});
    sys.surfaces[1].material =
        serde_json::from_value(json!({"type":"catalog", "name":"N-BK7", "offset":offset})).unwrap();
    let session = engine.session_from_project(Project::new(sys)).unwrap();
    let run = |name: &str, settings: Value| {
        session.run_analysis(&AnalysisRequest::new(name, settings), &JobContext::new())
    };
    let map = run("glass_map", Value::Null).unwrap();
    let points = map.json["glasses"].as_array().unwrap();
    let shifted = points.iter().find(|p| p["kind"] == "offset").unwrap();
    let nominal = points
        .iter()
        .find(|p| p["name"] == "N-BK7" && p["kind"] == "catalog")
        .unwrap();
    assert_eq!(shifted["surfaces"], json!([1]));
    assert_eq!(nominal["surfaces"], json!([]));
    assert_eq!(shifted["offset"], offset);
    assert!((shifted["nd"].as_f64().unwrap() - 1.5188000345005885).abs() < 2e-15);
    assert!((shifted["vd"].as_f64().unwrap() - 61.167336237499).abs() < 3e-12);
    assert!((shifted["pgf"].as_f64().unwrap() - nominal["pgf"].as_f64().unwrap()).abs() < 1e-13);

    let info = run(
        "glass_info",
        json!({"name":"N-BK7", "offset":offset, "curve_samples":3}),
    )
    .unwrap();
    assert_eq!(info.json["offset"], offset);
    assert_eq!(info.json["nd"], shifted["nd"]);
    assert_eq!(info.json["vd"], shifted["vd"]);
    assert_eq!(info.json["indices"][1]["n_catalog"], shifted["nd"]);

    let substituted = run(
        "glass_substitution",
        json!({
            "material_variables":[{"surface":1,"candidates":{"nearest":1}}],
            "max_combinations":1,"local_iterations":1,"hold_efl":false,
            "default_merit":{"fields":[0],"wavelengths":[1]},"backend":"cpu"
        }),
    )
    .unwrap();
    let rows = substituted.json["results"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    let replacement = &rows[0]["system"]["surfaces"][1]["material"];
    assert_eq!(replacement["type"], "catalog");
    assert!(replacement.get("offset").is_none());
    let conflict = run(
        "glass_substitution",
        json!({
            "material_variables":[{"surface":1}],
            "variables":[{"param":{"param":"material_dn_d","surface":1}}]
        }),
    );
    assert!(matches!(
        conflict,
        Err(mklight_engine::EngineError::InvalidSettings(_))
    ));
}

fn offset_tracer(environment: mklight_core::model::Environment) -> mklight_trace::SequentialTracer {
    let mut sys = samples::singlet();
    sys.environment = environment;
    sys.surfaces[1].material = MaterialRef::Catalog {
        catalog: None,
        name: "N-BK7".into(),
        offset: MaterialOffset {
            dn_d: 0.002,
            dv_d: -3.0,
        },
    };
    let snapshot = mklight_core::Snapshot::new(sys).unwrap();
    let resolved = mklight_core::ResolvedSystem::new(
        &snapshot,
        Arc::new(mklight_core::materials::MaterialLibrary::with_builtin()),
    )
    .unwrap();
    mklight_trace::SequentialTracer::new(Arc::new(resolved)).unwrap()
}

#[test]
fn offset_paraxial_power_matches_thick_lens_formula() {
    let tracer = offset_tracer(Default::default());
    let packed = mklight_gpu::PackedSystem::pack_all(tracer.system()).unwrap();
    for (w, n) in [1.5246724198828, 1.5188000345005885, 1.516190768295322]
        .into_iter()
        .enumerate()
    {
        let c1 = 1.0 / 60.0;
        let c2 = -1.0 / 350.0;
        let power = (n - 1.0) * (c1 - c2 + (n - 1.0) * 5.0 * c1 * c2 / n);
        assert!((tracer.paraxial().at(w).efl - 1.0 / power).abs() < 1e-11);
        let stored = packed.surfaces[1].n_after[w] as f64 + packed.surfaces[1].n_after_lo[w] as f64;
        assert!((stored - n).abs() < 4e-15);
    }
}

fn gpu_offset_agreement(backend: &dyn mklight_gpu::QualificationBackend) {
    use mklight_gpu::xval::{XvalSettings, cross_validate};
    for env in [
        Default::default(),
        mklight_core::model::Environment {
            temperature_c: 60.0,
            pressure_atm: 0.7,
        },
    ] {
        let tracer = offset_tracer(env);
        let settings = XvalSettings {
            pupil: mklight_trace::PupilSampling::Grid { n: 16 },
            ..Default::default()
        };
        let report = cross_validate(backend, &tracer, &settings, &JobContext::new()).unwrap();
        assert!(report.compared > 100 && report.opd_compared > 100);
        report.check(&settings.budget).unwrap();
        eprintln!(
            "{} offset trace: position {} mm, direction {}, OPD {} waves",
            report.backend, report.max_position_mm, report.max_direction, report.max_opd_waves
        );
    }
}

#[test]
fn offset_indices_reach_gpu_emulation_within_precision_budget() {
    gpu_offset_agreement(&mklight_gpu::CpuF32Backend::new());
}

#[test]
#[cfg(target_os = "macos")]
fn offset_indices_reach_metal_within_precision_budget() {
    match mklight_metal::MetalBackend::new().unwrap() {
        Some(backend) => gpu_offset_agreement(&backend),
        None => eprintln!("SKIP: no Metal device"),
    }
}
