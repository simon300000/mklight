//! The CUDA cost calibration on a real device: `prepare_gpu` measures it,
//! caches it under `<cache_dir>/cost/`, and the next engine loads it; an
//! engine that never prepares calibrates on its first `auto` request.

use mklight_core::{JobContext, samples};
use mklight_engine::backend::Backend;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, GpuCalibrationState};
use serde_json::{Value, json};

fn decision(engine: &Engine, rings: u32) -> Value {
    let session = engine
        .session_from_project(Project::new(samples::double_gauss()))
        .unwrap();
    let mut req = AnalysisRequest::new("spot_diagram", json!({ "rings": rings }));
    req.backend = Some(Backend::Auto);
    req.fresh = true;
    let r = session.run_analysis(&req, &JobContext::new()).unwrap();
    r.backend.cost_decision.unwrap()
}

fn settled(engine: &Engine) -> mklight_engine::analysis::cost::measured::DeviceCalibration {
    let t = std::time::Instant::now();
    loop {
        match engine.cuda_calibration_state() {
            GpuCalibrationState::Ready { calibration } => return calibration,
            GpuCalibrationState::Failed { message } => panic!("{message}"),
            _ if t.elapsed().as_secs() < 600 => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            s => panic!("calibration did not finish: {s:?}"),
        }
    }
}

/// An engine with an available CUDA device (listing the backends opens it).
fn cuda_engine() -> Option<Engine> {
    let engine = Engine::new(EngineConfig::default());
    engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:cuda" && b.available)
        .then_some(engine)
}

#[test]
#[ignore = "needs a CUDA device and runs a real calibration (seconds; use --release)"]
fn cuda_calibration_measures_caches_and_routes() {
    let Some(engine) = cuda_engine() else {
        return eprintln!("SKIP: no CUDA device");
    };
    let dir = std::env::temp_dir().join(format!("mklight-cuda-cost-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(engine.prepare_gpu(Some(dir.clone())));
    // The CPU while the kernels are prepared, then the prior until the
    // calibration is installed.
    let prior = decision(&engine, 6);
    assert!(
        prior["rule"] == "cuda preparing" || prior["rule"] == "cuda break-even",
        "{prior}"
    );
    assert_eq!(prior["decision"], "cpu_reference");
    let cal = settled(&engine);
    eprintln!(
        "CUDA calibration: {:.2} s, {}",
        cal.seconds,
        serde_json::to_string(&cal.analyses).unwrap()
    );
    assert_eq!(cal.source, "measured");
    assert_eq!(cal.analyses.len(), 7);
    assert!(!cal.driver.is_empty());
    assert_eq!(cal.fingerprint.len(), 32);
    let file = dir.join("cost").join("cuda.json");
    // The settled request keeps its first route.
    let again = decision(&engine, 6);
    assert_eq!(again["rule"], prior["rule"], "{again}");
    assert_eq!(again["repeated"], true, "{again}");
    // New small spot diagrams stay on the CPU, by the calibration.
    let small = decision(&engine, 5);
    assert_eq!(small["rule"], "cuda calibration", "{small}");
    assert_eq!(small["decision"], "cpu_reference", "{small}");
    eprintln!("250 rings: {}", decision(&engine, 250));
    if !cal.not_cached.is_empty() {
        // Other work on the machine disturbed it: installed, not cached.
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(&dir);
        return eprintln!("not cached: {}", cal.not_cached);
    }
    assert!(file.is_file());
    // The next engine loads the cached calibration.
    drop(engine);
    let next = Engine::new(EngineConfig::default());
    assert!(next.prepare_gpu(Some(dir.clone())));
    let cached = settled(&next);
    assert_eq!(cached.source, "cached");
    assert_eq!(cached.analyses, cal.analyses);
    assert_eq!(cached.fingerprint, cal.fingerprint);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without `prepare_gpu`, the first `auto` request on the open device
/// starts the calibration in the background; it is installed, not cached.
#[test]
#[ignore = "needs a CUDA device and runs a real calibration (seconds; use --release)"]
fn cuda_calibration_starts_lazily_without_prepare() {
    let Some(engine) = cuda_engine() else {
        return eprintln!("SKIP: no CUDA device");
    };
    assert_eq!(engine.cuda_calibration_state(), GpuCalibrationState::Idle);
    let first = decision(&engine, 6);
    assert_eq!(first["rule"], "cuda break-even", "{first}");
    assert_ne!(engine.cuda_calibration_state(), GpuCalibrationState::Idle);
    let cal = settled(&engine);
    assert_eq!(cal.source, "measured");
    assert!(!cal.analyses.is_empty());
    eprintln!(
        "lazy calibration: {:.2} s, not cached: {:?}",
        cal.seconds, cal.not_cached
    );
    let next = decision(&engine, 5);
    assert_eq!(next["rule"], "cuda calibration", "{next}");
}
