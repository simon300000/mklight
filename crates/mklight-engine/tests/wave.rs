//! Engine wave results against the solver, closed-form diffraction and jobs.
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use mklight_core::materials::MaterialLibrary;
use mklight_core::{
    ApertureSpec, Configuration, Field, FieldKind, FieldSet, JobContext, MaterialRef,
    OpticalSystem, Surface, SurfaceShape, Wavelength, WavelengthSet, samples,
};
use mklight_engine::analysis::schema::{check_buffers, validate};
use mklight_engine::buffer::BufferData;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use mklight_wave::analysis as wa;
use serde_json::{Value, json};

fn engine() -> Engine {
    Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: mklight_engine::backend::Backend::CpuReference,
        ..Default::default()
    })
}
fn session(e: &Engine, sys: OpticalSystem) -> Session {
    e.session_from_project(Project::new(sys)).unwrap()
}
fn run(s: &Session, name: &str, settings: Value) -> AnalysisResult {
    let ctx = JobContext::new();
    let result = s
        .run_analysis(&AnalysisRequest::new(name, settings), &ctx)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(ctx.progress.fraction(), 1.0);
    result
}
fn data<'a>(r: &'a AnalysisResult, name: &str) -> &'a [f64] {
    let b = r.buffers.iter().find(|b| b.name == name).unwrap();
    let BufferData::F64(v) = b.data() else {
        panic!("{name}: expected f64")
    };
    assert_eq!(b.shape.iter().product::<usize>(), v.len());
    v
}
fn flatten(v: &Value, out: &mut Vec<f64>) {
    if let Value::Array(a) = v {
        for e in a {
            flatten(e, out);
        }
    } else {
        out.push(v.as_f64().unwrap_or(f64::NAN));
    }
}

/// Compare every scalar and every buffer value to the direct JSON entry
/// point; offsets/shapes are checked independently against the schemas.
fn compare(actual: &Value, direct: &Value, r: &AnalysisResult, seen: &mut BTreeSet<String>) {
    if let (Value::String(name), Value::Array(_)) = (actual, direct) {
        let mut want = Vec::new();
        flatten(direct, &mut want);
        let got = data(r, name);
        assert_eq!(got.len(), want.len(), "{name}");
        for (&a, &b) in got.iter().zip(&want) {
            assert!(a == b || a.is_nan() && b.is_nan(), "{name}: {a} != {b}");
        }
        assert!(seen.insert(name.clone()), "buffer referenced twice: {name}");
    } else if let Value::Object(m) = direct {
        assert_eq!(actual.as_object().unwrap().len(), m.len());
        for (k, v) in m {
            compare(&actual[k], v, r, seen);
        }
    } else if let Value::Array(a) = direct {
        assert_eq!(actual.as_array().unwrap().len(), a.len());
        for (a, b) in actual.as_array().unwrap().iter().zip(a) {
            compare(a, b, r, seen);
        }
    } else {
        assert_eq!(actual, direct);
    }
}
fn check_schema(e: &Engine, r: &AnalysisResult) {
    let schema = e.analysis(&r.run_info.analysis).unwrap().result_schema();
    validate(&schema, &r.json, true).unwrap();
    let descs = r
        .buffers
        .iter()
        .map(|b| {
            (
                b.name.clone(),
                serde_json::to_value(b.dtype())
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
                b.shape.len(),
            )
        })
        .collect::<Vec<_>>();
    check_buffers(&schema, &descs).unwrap();
}

fn small_settings(name: &str) -> Value {
    let mut s = json!({
        "wavelengths": {"type": "primary"},
        "pupil": {"grid": 24, "rays": 17, "supersample": 2}
    });
    if name != "mtf_vs_field" {
        s["fields"] = json!({"type": "index", "index": 0});
    }
    if matches!(name, "mtf" | "mtf_vs_field" | "through_focus_mtf") {
        s["detector"] = json!({"pitch_mm": [0.005, 0.01], "diffusion_sigma_mm": 0.001});
    }
    match name {
        "psf" | "line_spread" | "diffraction_encircled_energy" => {
            s["grid_size"] = json!(32);
            s["padding"] = json!(2.0);
            if name == "diffraction_encircled_energy" {
                s["samples"] = json!(7);
            }
        }
        "mtf" => {
            s["frequency_samples"] = json!(9);
        }
        "mtf_vs_field" => {
            s["field_samples"] = json!(3);
        }
        "through_focus_mtf" => {
            s["focus_samples"] = json!(3);
        }
        _ => {}
    }
    s
}

#[test]
fn all_wave_analyses_match_schemas_and_direct_calls_on_every_sample() {
    let e = engine();
    let lib = Arc::new(MaterialLibrary::with_builtin());
    for (label, sys) in samples::all() {
        let cfg = mklight_wave::configured_tracer(&sys, lib.clone(), 0).unwrap();
        let s = session(&e, sys);
        for entry in wa::ANALYSES {
            let info = e.analysis(entry.name).unwrap().info();
            let mut expected_defaults = (entry.default_settings)();
            expected_defaults["configuration"] = json!(0);
            assert_eq!(info.default_settings, expected_defaults);
            validate(&info.settings_schema, &info.default_settings, true).unwrap();
            let r = run(&s, entry.name, small_settings(entry.name));
            check_schema(&e, &r);
            assert_eq!(r.run_info.snapshot_hash, s.snapshot().hash().to_hex());
            assert_eq!(r.run_info.settings["configuration"], 0);
            assert_eq!(r.run_info.backend, "cpu_reference");
            let mut settings = r.run_info.settings.clone();
            settings.as_object_mut().unwrap().remove("configuration");
            let mut direct = entry
                .run_configured(&cfg, &settings, &JobContext::new())
                .unwrap_or_else(|e| panic!("{label}/{}: {e}", entry.name));
            for warning in direct["run_info"]["warnings"].as_array().unwrap() {
                assert!(
                    r.run_info
                        .warnings
                        .contains(&warning.as_str().unwrap().to_string())
                );
            }
            direct.as_object_mut().unwrap().remove("run_info");
            let mut seen = BTreeSet::new();
            compare(&r.json, &direct, &r, &mut seen);
            assert_eq!(seen.len(), r.buffers.len());
        }
    }
}

fn ideal(defocus: f64) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 20.0 };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial {
                focal_length: 100.0,
            },
            100.0 + defocus,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0)],
    };
    s.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.55,
            weight: 1.0,
        }],
        primary: 0,
    };
    s
}

#[test]
fn ideal_strehl_and_circular_pupil_mtf_with_detector_nyquist() {
    let e = engine();
    let s = session(&e, ideal(0.0));
    let pupil = json!({"grid": 128, "rays": 65, "amplitude": "uniform"});
    let w = run(&s, "wavefront_map", json!({"pupil": pupil}));
    let map = &w.json["maps"][0];
    assert!((map["strehl_at_reference"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!(map["rms_waves"].as_f64().unwrap() < 1e-8);
    let p = run(&s, "psf", json!({"pupil": pupil, "grid_size": 64}));
    check_schema(&e, &p);
    assert!((p.json["psfs"][0]["strehl"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    let peak = data(&p, p.json["psfs"][0]["psf"]["data"].as_str().unwrap())
        .iter()
        .copied()
        .fold(0.0, f64::max);
    assert!((peak - 1.0).abs() < 1e-9);

    // Use the angular working F/# (finite NA), not the nominal focal length / EPD.
    let sin_max = map["exit_pupil"]["sin_max"].as_f64().unwrap();
    let working_f_number = 1.0 / (2.0 * sin_max);
    let cutoff = 1.0 / (0.55e-3 * working_f_number);
    let r = run(
        &s,
        "mtf",
        json!({
            "pupil": pupil, "method": "fft", "max_frequency": cutoff, "frequency_samples": 21,
            "detector": {"pitch_mm": [0.005, 0.01]}
        }),
    );
    check_schema(&e, &r);
    assert_eq!(r.json["nyquist"], json!([100.0, 50.0]));
    let f = &r.json["fields"][0];
    assert_eq!(f["detector"]["nyquist_tangential"], 50.0);
    assert_eq!(f["detector"]["nyquist_sagittal"], 100.0);
    let frequencies = data(&r, "frequencies");
    for key in ["tangential", "sagittal"] {
        let optical = data(&r, f[key].as_str().unwrap());
        let detector = data(&r, f["detector"][key].as_str().unwrap());
        let system = data(&r, f[format!("system_{key}")].as_str().unwrap());
        for i in 0..frequencies.len() {
            let expected = mklight_wave::diffraction_limited_mtf(frequencies[i] / cutoff, 0.0);
            assert!(
                (optical[i] - expected).abs() < 0.002,
                "{key} at {}: {} != {expected}",
                frequencies[i],
                optical[i]
            );
            assert!((system[i] - optical[i] * detector[i]).abs() < 1e-14);
        }
    }
}

#[test]
fn configuration_hash_settings_and_sampling_warnings_survive() {
    let e = engine();
    let mut sys = ideal(1.0);
    sys.configurations = vec![
        Configuration::default(),
        Configuration {
            name: "focused".into(),
            overrides: vec![(
                mklight_core::param::ParamPath::Thickness {
                    surface: mklight_core::SurfaceId(1),
                },
                100.0,
            )],
        },
    ];
    let s = session(&e, sys);
    let r = run(
        &s,
        "wavefront_map",
        json!({"configuration": 1, "pupil": {"grid": 32, "rays": 33}}),
    );
    assert_eq!(r.run_info.snapshot_hash, s.snapshot().hash().to_hex());
    assert_eq!(r.run_info.settings["configuration"], 1);
    assert!(r.json["maps"][0]["strehl_at_reference"].as_f64().unwrap() > 0.999999);

    // Coarse pupil, excessive image extent and a small defocused PSF window.
    let pupil = json!({"grid": 16, "rays": 33});
    let wrapped = run(
        &s,
        "psf",
        json!({"pupil": pupil, "grid_size": 64, "padding": 2.0}),
    );
    assert!(
        wrapped
            .run_info
            .warnings
            .iter()
            .any(|w| w.contains("wavefront changes"))
    );
    assert!(
        wrapped
            .run_info
            .warnings
            .iter()
            .any(|w| w.contains("wraps around"))
    );
    let truncated = run(
        &s,
        "psf",
        json!({"pupil": pupil, "grid_size": 16, "padding": 4.0}),
    );
    assert!(
        truncated
            .run_info
            .warnings
            .iter()
            .any(|w| w.contains("truncated"))
    );
    for r in [&wrapped, &truncated] {
        check_schema(&e, r);
    }
}

#[test]
fn monotone_progress_and_mid_run_cancellation() {
    let e = engine();
    let s = session(&e, samples::cooke_triplet());
    let ctx = JobContext::new();
    let done = AtomicBool::new(false);
    let (result, seen) = std::thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            let start = Instant::now();
            let mut seen = Vec::new();
            while !done.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(30) {
                let f = ctx.progress.fraction();
                seen.push(f);
                // Pupil tracing occupies 40 % of the solver subrange.
                if f > 0.48 {
                    ctx.cancel.cancel();
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
            seen
        });
        let result = s.run_analysis(
            &AnalysisRequest::new(
                "psf",
                json!({
                    "method": "huygens", "grid_size": 128,
                    "pupil": {"grid": 48, "rays": 24}
                }),
            ),
            &ctx,
        );
        done.store(true, Ordering::Relaxed);
        (result, monitor.join().unwrap())
    });
    assert!(result.unwrap_err().is_cancelled());
    assert!(seen.iter().any(|&f| f > 0.1 && f < 0.95));
    assert!(seen.windows(2).all(|w| w[1] >= w[0]));
    assert!(ctx.progress.fraction() < 1.0);
}
