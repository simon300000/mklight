//! The WebGPU calibration (GPU.md §11.6) natively: it measures both backends,
//! installs itself, and `auto` decisions on WebGPU then come from it.

use mklight_core::{JobContext, samples};
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, WebGpuCalibrationState};
use serde_json::json;
use std::sync::Arc;

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release only: unoptimised, the first WebGPU runs take minutes"
)]
fn calibration_runs_installs_and_routes() {
    let engine = Engine::new(EngineConfig::default());
    // WebGPU as the only GPU, as in the browser.
    engine.register_backend(Arc::new(GpuProvider::new("metal", "off", || {
        Err("off for this test".into())
    })));
    engine.register_backend(Arc::new(GpuProvider::webgpu()));
    if !engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:webgpu" && b.available)
    {
        eprintln!("SKIP: no WebGPU adapter");
        return;
    }
    let session = engine
        .session_from_project(Project::new(samples::double_gauss()))
        .unwrap();
    let decision = |rings: u32| {
        let mut req = AnalysisRequest::new("ray_grid", json!({ "rings": rings }));
        req.backend = Some(Backend::Auto);
        req.fresh = true;
        let r = session.run_analysis(&req, &JobContext::new()).unwrap();
        r.backend.cost_decision.unwrap()
    };
    assert_eq!(decision(8)["rule"], "webgpu break-even");
    assert!(engine.start_webgpu_calibration());
    assert!(
        !engine.start_webgpu_calibration(),
        "one calibration at a time"
    );
    let t = std::time::Instant::now();
    let cal = loop {
        match engine.webgpu_calibration_state() {
            WebGpuCalibrationState::Ready { calibration } => break calibration,
            WebGpuCalibrationState::Failed { message } => panic!("{message}"),
            _ if t.elapsed().as_secs() < 600 => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            s => panic!("calibration did not finish: {s:?}"),
        }
    };
    eprintln!("calibration {}", serde_json::to_string(&cal).unwrap());
    assert_eq!(cal.analyses.len(), 3);
    assert_eq!(cal.source, "measured");
    let d = decision(8);
    assert_eq!(d["rule"], "webgpu calibration", "{d}");
    assert!(d["predicted_ms"]["cpu"].as_f64().unwrap() > 0.);
    // A forced estimate changes the route.
    let mut fast = cal.clone();
    for p in fast.analyses.values_mut() {
        p.webgpu.fixed_ms = 0.;
        p.webgpu.per_mwork_ms = 1e-6;
        p.cpu.fixed_ms = 100.;
    }
    fast.source = "forced".into();
    engine.set_webgpu_calibration(Some(fast));
    assert_eq!(decision(8)["decision"], "qualified_webgpu");
    let mut slow = cal;
    for p in slow.analyses.values_mut() {
        p.webgpu.fixed_ms = 1e6;
    }
    slow.source = "forced".into();
    engine.set_webgpu_calibration(Some(slow));
    assert_eq!(decision(250)["decision"], "cpu_reference");
    engine.set_webgpu_calibration(None);
    assert_eq!(decision(8)["rule"], "webgpu break-even");
}
