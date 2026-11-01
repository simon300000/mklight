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
const MTF_ABS: f64 = 2e-3;

fn engine_with(p: Option<GpuProvider>) -> Engine {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: Backend::CpuReference,
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
        // Energy analyses and best focus through the same path, against the
        // reference: MTF ≤ 2e-3 absolute (CONVENTIONS §7), encircled-energy
        // fractions to the same budget (or a few rays' weight, below), radii
        // and merits to the RMS budget, the best defocus to 1e-4 mm.
        let set = json!({"rings": 6});
        let pair = |a: &str, set: &Value| {
            let c = run_on(&s, a, set.clone(), None);
            let g = run_on(&s, a, set.clone(), Some(b.clone()));
            if a == "encircled_energy" && g.run_info.backend == "cpu_reference" {
                assert!(
                    g.backend
                        .fallback_reason
                        .as_deref()
                        .is_some_and(|r| r.contains("completion") || r.contains("qualification"))
                );
            } else {
                assert_eq!(g.run_info.backend, backend, "{a}: {:?}", g.backend);
            }
            assert_eq!(c.run_info.backend, "cpu_reference", "{a}");
            (c.json, g.json)
        };
        let radius_ok = |x: f64, y: f64| {
            (x.is_nan() && y.is_nan()) || (x - y).abs() <= (RMS_REL * x.abs()).max(RMS_ABS_MM)
        };
        let (c, g) = pair("geometric_mtf", &set);
        assert_eq!(c["frequencies"], g["frequencies"], "{name}");
        let mut worst = 0.0f64;
        for (cc, gc) in c["curves"]
            .as_array()
            .unwrap()
            .iter()
            .zip(g["curves"].as_array().unwrap())
        {
            for (x, y) in cc["modulation"]
                .as_array()
                .unwrap()
                .iter()
                .zip(gc["modulation"].as_array().unwrap())
            {
                worst = worst.max((num(x) - num(y)).abs());
            }
        }
        assert!(worst <= MTF_ABS, "{name}: MTF differs by {worst}");
        // Encircled energy is a step function of the radius: a ray within
        // the position budget of a radius sample may cross it, and a ring of
        // symmetric rays crosses together. Allow a few rays' weight.
        let (c, g) = pair("encircled_energy", &json!({"rings": 24}));
        // The radius axis ends at the largest GEO radius, itself within the
        // RMS budget.
        for (x, y) in c["radius"]
            .as_array()
            .unwrap()
            .iter()
            .zip(g["radius"].as_array().unwrap())
        {
            assert!(radius_ok(num(x), num(y)), "{name} radius: {x} vs {y}");
        }
        for (cf, gf) in c["fields"]
            .as_array()
            .unwrap()
            .iter()
            .zip(g["fields"].as_array().unwrap())
        {
            assert_eq!(cf["rays_arrived"], gf["rays_arrived"], "{name}");
            let step = MTF_ABS.max(8.0 / num(&cf["rays_arrived"]));
            for key in ["encircled", "ensquared"] {
                for (x, y) in cf[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(gf[key].as_array().unwrap())
                {
                    let d = (num(x) - num(y)).abs();
                    assert!(d <= step, "{name} {key}: {x} vs {y}");
                }
            }
            for key in ["r50", "r80", "r90", "half_width80"] {
                let (x, y) = (num(&cf[key]), num(&gf[key]));
                assert!(radius_ok(x, y), "{name} {key}: {x} vs {y}");
            }
        }
        let (c, g) = pair("best_focus", &set);
        let (x, y) = (num(&c["defocus"]), num(&g["defocus"]));
        assert!((x - y).abs() <= 1e-4, "{name} best defocus: {x} vs {y}");
        for key in ["merit", "merit_at_zero"] {
            let (x, y) = (num(&c[key]), num(&g[key]));
            assert!(radius_ok(x, y), "{name} {key}: {x} vs {y}");
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
    // Large grids are past every cost threshold.
    let big = json!({"rings": 150});
    let r = run_on(&s, "spot_diagram", big.clone(), Some(Backend::Auto));
    assert_eq!(r.backend.requested, "auto");
    assert!(r.backend.qualification.is_some());
    // An explicit GPU request is honoured for any size.
    let r = run_on(
        &s,
        "spot_diagram",
        json!({"rings": 3}),
        Some(Backend::Gpu("cpu_f32".into())),
    );
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
    let r = run_on(&s, "ray_grid", big, None);
    assert_eq!(r.backend.requested, "default:auto");
    assert_eq!(r.run_info.precision, Precision::F32Gpu);
    // A small grid: on an Apple GPU the cost rule keeps the reference (no
    // warning); the emulation has no cost evidence and qualifies.
    let r = run_on(&s, "ray_grid", json!({"rings": 2}), None);
    let cost = &r.backend.cost_decision.as_ref().unwrap()["decision"];
    assert_eq!(
        r.run_info.backend == "cpu_reference",
        cost == "cpu_reference"
    );
    assert!(r.run_info.warnings.is_empty(), "{:?}", r.run_info.warnings);
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

/// Cancelling while the host generates a large grid's rays (after
/// qualification; generation reports progress per block) stops the job
/// promptly with `Cancelled`, never a CPU fallback result.
#[test]
fn cancelling_a_gpu_job_mid_run_stops_it() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    let e = engine_with(Some(GpuProvider::emulation()));
    let s = session(&e, samples::double_gauss());
    let mut req = AnalysisRequest::new("spot_diagram", json!({"rings": 400}));
    req.backend = Some(Backend::Gpu("cpu_f32".into()));
    let ctx = JobContext::new();
    let (cancel, progress) = (ctx.cancel.clone(), ctx.progress.clone());
    let fired = Arc::new(AtomicBool::new(false));
    let cancelled_at = Arc::new(std::sync::Mutex::new(None::<Instant>));
    let watcher = {
        let (fired, at) = (fired.clone(), cancelled_at.clone());
        std::thread::spawn(move || {
            // Cancel once generation has reported progress twice (it reports
            // per block, so a large grid's generation is observable).
            let t0 = Instant::now();
            let mut first = None;
            while t0.elapsed() < Duration::from_secs(120) {
                let f = progress.fraction();
                if f <= 0.15 || f >= 0.65 {
                    std::thread::sleep(Duration::from_micros(200));
                    continue;
                }
                match first {
                    None => first = Some(f),
                    Some(f0) if f > f0 => {
                        *at.lock().unwrap() = Some(Instant::now());
                        cancel.cancel();
                        fired.store(true, Ordering::SeqCst);
                        return;
                    }
                    _ => {}
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };
    let err = s.run_analysis(&req, &ctx).unwrap_err();
    let done = Instant::now();
    watcher.join().unwrap();
    assert!(err.is_cancelled(), "{err}");
    assert!(
        fired.load(Ordering::SeqCst),
        "the cancel did not land mid-run"
    );
    let latency = done - cancelled_at.lock().unwrap().unwrap();
    // One generation block (8192 rays) is milliseconds; allow for a loaded
    // machine.
    assert!(latency < Duration::from_secs(5), "{latency:?}");
}

/// Merit jobs use the provider selected by the facade and share its
/// snapshot qualification with ray-grid analyses.
#[test]
fn merit_jobs_share_provider_qualification_and_report_actual_backend() {
    let provider = GpuProvider::emulation();
    let handle = provider.device().unwrap();
    let device = handle.downcast_ref::<GpuDevice>().unwrap();
    let engine = engine_with(Some(provider));
    let session = session(&engine, samples::cooke_triplet());
    let backend = Some(Backend::Gpu("cpu_f32".into()));
    run_on(&session, "ray_grid", json!({"rings": 2}), backend.clone());
    assert_eq!(device.cached_qualifications(), 1);
    for (name, settings) in [
        ("optimize", json!({"max_iterations": 2})),
        ("tolerance", json!({"mode": "monte_carlo", "trials": 3})),
    ] {
        let gpu = run_on(&session, name, settings.clone(), backend.clone());
        assert_eq!(gpu.backend.requested, "gpu:cpu_f32");
        assert_eq!(gpu.run_info.backend, "gpu:cpu_f32");
        assert_eq!(gpu.backend.used, gpu.run_info.backend);
        assert_eq!(gpu.run_info.precision, Precision::F32Gpu);
        assert_eq!(gpu.backend.qualification.as_ref().unwrap()["passed"], true);
        if name == "tolerance" {
            assert_eq!(
                gpu.json["backend"]["reported_on"],
                gpu.json["backend"]["evaluated_on"]
            );
        }
        assert!(
            gpu.json["backend"]["gpu_stats"]["gpu_candidates"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(gpu.backend.fallback_reason.is_none());
        let cached = device.cached_qualifications();
        run_on(&session, name, settings.clone(), backend.clone());
        assert_eq!(device.cached_qualifications(), cached);
        let auto = run_on(&session, name, settings, Some(Backend::Auto));
        assert!(auto.backend.qualification.is_some() || auto.backend.fallback_reason.is_some());
    }
}

#[test]
fn merit_jobs_reuse_failed_provider_qualification_and_fall_back() {
    let mut qualification = XvalSettings::qualification();
    qualification.budget.position_mm = 1e-15;
    let provider = GpuProvider::emulation().with_qualification(qualification);
    let handle = provider.device().unwrap();
    let device = handle.downcast_ref::<GpuDevice>().unwrap();
    let engine = engine_with(Some(provider));
    let session = session(&engine, samples::singlet());
    for (name, settings) in [
        ("optimize", json!({"max_iterations": 2})),
        ("tolerance", json!({"mode": "monte_carlo", "trials": 3})),
    ] {
        let cpu = run_on(
            &session,
            name,
            settings.clone(),
            Some(Backend::CpuReference),
        );
        let fallback = run_on(
            &session,
            name,
            settings,
            Some(Backend::Gpu("cpu_f32".into())),
        );
        assert_eq!(fallback.run_info.backend, "cpu_reference");
        assert_eq!(fallback.backend.used, "cpu_reference");
        assert_eq!(fallback.run_info.precision, Precision::F64Reference);
        assert_eq!(
            fallback.backend.qualification.as_ref().unwrap()["passed"],
            false
        );
        assert!(fallback.backend.fallback_reason.is_some());
        assert!(
            fallback
                .run_info
                .warnings
                .iter()
                .any(|w| w.contains("gpu:cpu_f32"))
        );
        assert_eq!(device.cached_qualifications(), 1);
        let key = if name == "optimize" {
            "final"
        } else {
            "monte_carlo"
        };
        let (mut actual, mut expected) = (fallback.json[key].clone(), cpu.json[key].clone());
        for result in [&mut actual, &mut expected] {
            result.as_object_mut().unwrap().remove("seconds_per_trial");
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn cpu_only_merit_reports_cpu_even_with_a_selected_gpu() {
    use mklight_optim::merit::operand::OperandKind;
    use mklight_optim::merit::{MeritFunctionSpec, Operand};
    let engine = engine_with(Some(GpuProvider::emulation()));
    let session = session(&engine, samples::singlet());
    let spec = MeritFunctionSpec {
        operands: vec![
            Operand::new(OperandKind::ParamValue {
                param: mklight_core::param::ParamPath::Curvature {
                    surface: mklight_core::SurfaceId(1),
                },
            })
            .target(0.01),
        ],
        ..Default::default()
    };
    let settings = json!({"merit": spec, "max_iterations": 2, "hold_efl": false});
    let cpu = run_on(
        &session,
        "optimize",
        settings.clone(),
        Some(Backend::CpuReference),
    );
    let gpu = run_on(
        &session,
        "optimize",
        settings,
        Some(Backend::Gpu("cpu_f32".into())),
    );
    assert_eq!(gpu.run_info.backend, "cpu_reference");
    assert_eq!(gpu.run_info.precision, Precision::F64Reference);
    assert_eq!(gpu.json["backend"]["precision"], "f64_reference");
    assert_eq!(gpu.json["final"], cpu.json["final"]);
    assert!(
        gpu.backend
            .fallback_reason
            .unwrap()
            .contains("no candidates")
    );
    assert!(
        !gpu.run_info
            .warnings
            .iter()
            .any(|w| w.starts_with("ray operands evaluated on"))
    );
}
