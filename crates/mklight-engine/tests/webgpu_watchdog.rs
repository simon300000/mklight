//! A WebGPU device whose trace pipeline never finishes compiling (simulated
//! with the pipeline watchdog's test stall, `mklight_webgpu::watchdog`):
//! `auto` runs on the CPU reference with the reason, an explicit request
//! fails with it, and the calibration fails instead of hanging or
//! recording numbers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mklight_core::{JobContext, samples};
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, EngineError, Session, WebGpuCalibrationState,
};
use mklight_gpu::ComputeBackend;
use mklight_webgpu::watchdog::Watchdog;
use mklight_webgpu::{WebGpuBackend, WgpuDevice};
use serde_json::json;

const LIMIT: Duration = Duration::from_millis(300);

/// An engine whose only GPU is WebGPU with the trace pipelines stalled, or
/// `None` without a WebGPU adapter.
fn stalled_engine() -> Option<Engine> {
    let engine = Engine::new(EngineConfig::default());
    engine.register_backend(Arc::new(GpuProvider::new("metal", "off", || {
        Err("off for this test".into())
    })));
    engine.register_backend(Arc::new(GpuProvider::webgpu_with(|| {
        let device = match WgpuDevice::open_native() {
            Ok(Some(d)) => d,
            Ok(None) => return Err("no WebGPU adapter".into()),
            Err(e) => return Err(e.to_string()),
        };
        device.set_watchdog(Watchdog {
            timeout: LIMIT,
            stalled: vec!["trace".into(), "trace_overlays".into()],
        });
        WebGpuBackend::with_gpu(Arc::new(device))
            .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>)
            .map_err(|e| e.to_string())
    })));
    if !webgpu_available(&engine) {
        eprintln!("SKIP: no WebGPU adapter");
        return None;
    }
    Some(engine)
}

fn webgpu_available(engine: &Engine) -> bool {
    engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:webgpu" && b.available)
}

fn webgpu_reason(engine: &Engine) -> Option<String> {
    engine
        .backends()
        .into_iter()
        .find(|b| b.id == "gpu:webgpu")
        .and_then(|b| b.reason)
}

fn session(engine: &Engine) -> Session {
    engine
        .session_from_project(Project::new(samples::cooke_triplet()))
        .unwrap()
}

/// A ray grid the prior sends to WebGPU (3 fields × 3 wavelengths × 27 937
/// rays at 96 rings: above the 250 000-ray break-even).
fn large_ray_grid(backend: Backend) -> AnalysisRequest {
    let mut req = AnalysisRequest::new("ray_grid", json!({ "rings": 96 }));
    req.backend = Some(backend);
    req.fresh = true;
    req
}

/// The reason (the first trace pipeline of the run is the stalled one).
const REASON: &str = "compute pipeline did not finish compiling within 0.3 s, \
                      so WebGPU is not used on this device";

#[test]
fn auto_falls_back_to_the_cpu_with_the_reason() {
    let Some(engine) = stalled_engine() else {
        return;
    };
    let s = session(&engine);
    let t = Instant::now();
    let r = s
        .run_analysis(&large_ray_grid(Backend::Auto), &JobContext::new())
        .unwrap();
    assert!(t.elapsed() < LIMIT + Duration::from_secs(60));
    assert_eq!(r.backend.used, "cpu_reference");
    let why = r.backend.fallback_reason.clone().unwrap();
    assert!(why.contains("gpu:webgpu: the trace::"), "{why}");
    assert!(why.contains(REASON), "{why}");
    // From now on the backend is unavailable: auto decides before the
    // device, with the same reason.
    assert!(!webgpu_available(&engine));
    assert!(webgpu_reason(&engine).unwrap().contains(REASON));
    let r = s
        .run_analysis(&large_ray_grid(Backend::Auto), &JobContext::new())
        .unwrap();
    assert_eq!(r.backend.used, "cpu_reference");
    let why = r.backend.fallback_reason.unwrap();
    assert!(
        why.contains("gpu:webgpu: the trace::") && why.contains(REASON),
        "{why}"
    );
}

#[test]
fn an_explicit_request_fails_with_the_reason() {
    let Some(engine) = stalled_engine() else {
        return;
    };
    let s = session(&engine);
    let webgpu = Backend::Gpu("webgpu".into());
    for _ in 0..2 {
        // The first request meets the stall, the second the unavailable
        // backend: both fail with the reason, neither hangs.
        let t = Instant::now();
        let e = s
            .run_analysis(&large_ray_grid(webgpu.clone()), &JobContext::new())
            .unwrap_err();
        assert!(t.elapsed() < LIMIT + Duration::from_secs(60));
        match &e {
            EngineError::BackendUnavailable(m) => {
                assert!(
                    m.contains("gpu:webgpu: the trace::") && m.contains(REASON),
                    "{m}"
                )
            }
            e => panic!("{e:?}"),
        }
    }
    // With fallback allowed it runs on the CPU and says why.
    let mut req = large_ray_grid(webgpu);
    req.allow_backend_fallback = true;
    let r = s.run_analysis(&req, &JobContext::new()).unwrap();
    assert_eq!(r.backend.used, "cpu_reference");
    assert!(
        r.backend
            .fallback_reason
            .as_deref()
            .is_some_and(|w| w.contains(REASON)),
        "{:?}",
        r.backend
    );
}

#[test]
fn the_calibration_fails_with_the_reason_instead_of_hanging() {
    let Some(engine) = stalled_engine() else {
        return;
    };
    assert!(engine.start_webgpu_calibration());
    let t = Instant::now();
    let message = loop {
        match engine.webgpu_calibration_state() {
            WebGpuCalibrationState::Failed { message } => break message,
            WebGpuCalibrationState::Ready { calibration } => {
                panic!("calibrated a device that cannot trace: {calibration:?}")
            }
            _ if t.elapsed().as_secs() < 300 => std::thread::sleep(Duration::from_millis(20)),
            s => panic!("the calibration did not finish: {s:?}"),
        }
    };
    assert!(message.contains(REASON), "{message}");
    assert!(!webgpu_available(&engine));
    // `auto` keeps the prior (no calibration installed) and routes to the
    // CPU with the reason.
    let r = session(&engine)
        .run_analysis(&large_ray_grid(Backend::Auto), &JobContext::new())
        .unwrap();
    assert_eq!(r.backend.used, "cpu_reference");
    assert!(
        r.backend
            .fallback_reason
            .as_deref()
            .is_some_and(|w| w.contains("gpu:webgpu: the trace::") && w.contains(REASON)),
        "{:?}",
        r.backend
    );
}
