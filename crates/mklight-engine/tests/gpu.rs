//! GPU backend selection and the GPU paths of the ray-grid analyses,
//! cross-checked against the CPU f64 reference within the CONVENTIONS §7
//! budget. The CPU-f32 emulation backend runs the GPU algorithms on any
//! machine; the Metal tests run when this machine has a qualifying Metal
//! device and are skipped (with a message) otherwise.

use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::job::Precision;
use mklight_core::{JobContext, OpticalSystem, samples};
use mklight_engine::backend::{Backend, BackendProvider};
use mklight_engine::buffer::BufferData;
use mklight_engine::gpu::{GpuDevice, GpuProvider};
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use mklight_gpu::ErrorBudget;
use mklight_gpu::xval::XvalSettings;

/// CONVENTIONS §7.
const POSITION_MM: f64 = 2e-5;
const RMS_REL: f64 = 5e-3;
const RMS_ABS_MM: f64 = 2e-5;

fn engine_with(p: Option<GpuProvider>) -> Engine {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        ..Default::default()
    });
    if let Some(p) = p {
        e.register_backend(Arc::new(p));
    }
    e
}

fn session(e: &Engine, sys: OpticalSystem) -> Session {
    e.session_from_project(Project::new(sys)).unwrap()
}

fn run_on(s: &Session, name: &str, settings: Value, backend: Option<Backend>) -> AnalysisResult {
    let mut req = AnalysisRequest::new(name, settings);
    req.backend = backend;
    s.run_analysis(&req, &JobContext::new())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn f64s(r: &AnalysisResult, name: &str) -> Vec<f64> {
    match &r.buffers.iter().find(|b| b.name == name).unwrap().data {
        BufferData::F64(v) => v.clone(),
        _ => panic!(),
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// Spot statistics and ray grids of `backend` against the CPU reference.
fn cross_check(e: &Engine, backend: &str) {
    let b = Backend::parse(backend);
    for (name, sys) in samples::all() {
        let s = session(e, sys);
        let cpu = run_on(&s, "spot_diagram", Value::Null, None);
        let gpu = run_on(&s, "spot_diagram", Value::Null, Some(b.clone()));
        assert_eq!(gpu.run_info.backend, backend, "{name}: {:?}", gpu.backend);
        assert_eq!(gpu.run_info.precision, Precision::F32Gpu);
        assert_eq!(gpu.backend.used, backend);
        assert!(gpu.backend.fallback_reason.is_none());
        assert_eq!(gpu.backend.qualification.as_ref().unwrap()["passed"], true);
        assert_eq!(cpu.run_info.backend, "cpu_reference");
        let (fc, fg) = (&cpu.json["fields"], &gpu.json["fields"]);
        for k in 0..fc.as_array().unwrap().len() {
            let (c, g) = (&fc[k], &fg[k]);
            assert_eq!(
                c["polychromatic"]["rays_arrived"], g["polychromatic"]["rays_arrived"],
                "{name} field {k}: survival differs"
            );
            for key in ["rms_radius", "geo_radius"] {
                let (x, y) = (num(&c[key]), num(&g[key]));
                let tol = (RMS_REL * x).max(RMS_ABS_MM);
                assert!((x - y).abs() <= tol, "{name} field {k} {key}: {x} vs {y}");
            }
            for i in 0..2 {
                let (x, y) = (num(&c["chief"][i]), num(&g["chief"][i]));
                assert!((x - y).abs() <= POSITION_MM, "{name} chief: {x} vs {y}");
            }
        }
        // Ray grids: every hit within the position budget, same survival.
        let cpu = run_on(&s, "ray_grid", json!({"rings": 4}), None);
        let gpu = run_on(&s, "ray_grid", json!({"rings": 4}), Some(b.clone()));
        assert_eq!(gpu.run_info.backend, backend);
        let (pc, pg) = (f64s(&cpu, "points"), f64s(&gpu, "points"));
        assert_eq!(pc.len(), pg.len());
        let mut worst = 0.0f64;
        for (x, y) in pc.iter().zip(&pg) {
            assert_eq!(x.is_nan(), y.is_nan(), "{name}: survival differs");
            if !x.is_nan() {
                worst = worst.max((x - y).abs());
            }
        }
        assert!(worst <= POSITION_MM, "{name}: position error {worst} mm");
        let (dc, dg) = (f64s(&cpu, "directions"), f64s(&gpu, "directions"));
        for (x, y) in dc.iter().zip(&dg) {
            if !x.is_nan() {
                assert!((x - y).abs() <= 2e-6, "{name}: direction {x} vs {y}");
            }
        }
        // Energy analyses through the same path.
        for a in ["encircled_energy", "geometric_mtf"] {
            let g = run_on(&s, a, json!({"rings": 6}), Some(b.clone()));
            assert_eq!(g.run_info.backend, backend, "{a}");
        }
    }
}

#[test]
fn emulated_gpu_matches_the_reference_within_budget() {
    let e = engine_with(Some(GpuProvider::emulation()));
    cross_check(&e, "gpu:cpu_f32");
}

#[test]
fn metal_matches_the_reference_within_budget() {
    let e = engine_with(None);
    let metal = e.backends().into_iter().find(|b| b.id == "gpu:metal");
    match metal {
        Some(m) if m.available => {
            assert_eq!(m.precision, Precision::F32Gpu);
            assert!(
                m.capabilities
                    .analyses
                    .contains(&"spot_diagram".to_string())
            );
            assert!(m.capabilities.device_name.is_some());
            cross_check(&e, "gpu:metal");
        }
        Some(m) => eprintln!("skipping: Metal unavailable ({:?})", m.reason),
        None => eprintln!("skipping: no Metal backend on this platform"),
    }
}

#[test]
fn auto_picks_a_gpu_only_for_analyses_with_a_gpu_path() {
    let e = engine_with(Some(GpuProvider::emulation()));
    // `auto` takes the first available GPU in registration order: Metal
    // where it qualifies as a device, else the emulation.
    let s = session(&e, samples::singlet());
    let r = run_on(&s, "spot_diagram", json!({"rings": 3}), Some(Backend::Auto));
    assert_eq!(r.backend.requested, "auto");
    assert_ne!(r.run_info.backend, "cpu_reference");
    assert_eq!(r.run_info.precision, Precision::F32Gpu);
    let r = run_on(&s, "seidel", Value::Null, Some(Backend::Auto));
    assert_eq!(r.run_info.backend, "cpu_reference");
    assert!(
        r.backend
            .fallback_reason
            .unwrap()
            .contains("no GPU path for 'seidel'")
    );
    assert!(r.run_info.warnings.is_empty(), "{:?}", r.run_info.warnings);
    // As the engine default.
    e.set_default_backend(Backend::Auto).unwrap();
    let r = run_on(&s, "ray_grid", json!({"rings": 2}), None);
    assert_eq!(r.backend.requested, "default:auto");
    assert_eq!(r.run_info.precision, Precision::F32Gpu);
    // An explicit GPU request for an analysis without a GPU path fails
    // (or falls back when allowed, with a warning).
    let mut req = AnalysisRequest::new("seidel", Value::Null);
    req.backend = Some(Backend::Gpu("cpu_f32".into()));
    assert!(s.run_analysis(&req, &JobContext::new()).is_err());
    req.allow_backend_fallback = true;
    let r = s.run_analysis(&req, &JobContext::new()).unwrap();
    assert_eq!(r.run_info.backend, "cpu_reference");
    assert!(!r.run_info.warnings.is_empty());
}

#[test]
fn failed_qualification_falls_back_to_the_reference_and_says_so() {
    // A budget no f32 trace can meet.
    let mut q = XvalSettings::qualification();
    q.budget = ErrorBudget {
        position_mm: 1e-15,
        ..ErrorBudget::CONVENTIONS
    };
    q.radiometry = mklight_gpu::Radiometry::Off;
    let p = GpuProvider::emulation().with_qualification(q);
    let dev = p.device().unwrap();
    let e = engine_with(Some(p));
    let s = session(&e, samples::cooke_triplet());
    let b = Some(Backend::Gpu("cpu_f32".into()));
    let gpu = run_on(&s, "spot_diagram", Value::Null, b.clone());
    let cpu = run_on(&s, "spot_diagram", Value::Null, None);
    assert_eq!(gpu.run_info.backend, "cpu_reference");
    assert_eq!(gpu.run_info.precision, Precision::F64Reference);
    assert_eq!(gpu.backend.requested, "gpu:cpu_f32");
    let why = gpu.backend.fallback_reason.clone().unwrap();
    assert!(why.contains("qualification"), "{why}");
    assert!(
        gpu.run_info
            .warnings
            .iter()
            .any(|w| w.contains("gpu:cpu_f32") && w.contains("CPU f64 reference")),
        "{:?}",
        gpu.run_info.warnings
    );
    assert_eq!(gpu.backend.qualification.as_ref().unwrap()["passed"], false);
    // The numbers are the reference's, bit for bit.
    assert_eq!(gpu.json, cpu.json);
    assert_eq!(gpu.buffers, cpu.buffers);
    // The verdict is cached per snapshot: a second run does not qualify again.
    let d = dev.downcast_ref::<GpuDevice>().unwrap();
    assert_eq!(d.cached_qualifications(), 1);
    run_on(&s, "ray_grid", Value::Null, b);
    assert_eq!(d.cached_qualifications(), 1);
}

#[test]
fn cancelling_a_gpu_job_cancels_instead_of_falling_back() {
    let e = engine_with(Some(GpuProvider::emulation()));
    let s = session(&e, samples::double_gauss());
    let mut req = AnalysisRequest::new("spot_diagram", Value::Null);
    req.backend = Some(Backend::Gpu("cpu_f32".into()));
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    let err = s.run_analysis(&req, &ctx).unwrap_err();
    assert!(err.is_cancelled(), "{err}");
}
