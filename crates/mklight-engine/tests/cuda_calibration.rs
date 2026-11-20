//! The CUDA cost calibration on a real device: `prepare_gpu` measures it,
//! caches it under `<cache_dir>/cost/`, and the next engine loads it.

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

#[test]
#[ignore = "needs a CUDA device and runs a real calibration (seconds; use --release)"]
fn cuda_calibration_measures_caches_and_routes() {
    let engine = Engine::new(EngineConfig::default());
    if !engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:cuda" && b.available)
    {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("mklight-cuda-cost-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // The prior until the calibration is installed.
    let prior = decision(&engine, 6);
    assert_eq!(prior["rule"], "cuda break-even", "{prior}");
    assert_eq!(prior["decision"], "cpu_reference");
    assert!(engine.prepare_gpu(Some(dir.clone())));
    let cal = settled(&engine);
    eprintln!(
        "CUDA calibration: {:.2} s, {}",
        cal.seconds,
        serde_json::to_string(&cal.analyses).unwrap()
    );
    assert_eq!(cal.source, "measured");
    assert_eq!(cal.analyses.len(), 8);
    assert!(!cal.driver.is_empty());
    assert!(dir.join("cost").join("cuda.json").is_file());
    // Small spot diagrams stay on the CPU.
    let small = decision(&engine, 6);
    assert_eq!(small["rule"], "cuda calibration", "{small}");
    assert_eq!(small["decision"], "cpu_reference", "{small}");
    eprintln!("250 rings: {}", decision(&engine, 250));
    // The next engine loads the cached calibration.
    drop(engine);
    let next = Engine::new(EngineConfig::default());
    assert!(next.prepare_gpu(Some(dir.clone())));
    let cached = settled(&next);
    assert_eq!(cached.source, "cached");
    assert_eq!(cached.analyses, cal.analyses);
    let _ = std::fs::remove_dir_all(&dir);
}
