//! The pipeline watchdog (`mklight_webgpu::watchdog`) with a simulated
//! shader compilation that never finishes: the device fails within the time
//! limit with the reason, instead of waiting forever, and every later
//! pipeline request fails at once.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use mklight_gpu::ComputeBackend;
use mklight_webgpu::watchdog::Watchdog;
use mklight_webgpu::{Gpu, WebGpuBackend, WgpuDevice, proxy};

const LIMIT: Duration = Duration::from_millis(300);

fn stalled_trace() -> Watchdog {
    Watchdog {
        timeout: LIMIT,
        stalled: vec!["trace".into(), "trace_overlays".into()],
    }
}

#[test]
fn a_stalled_pipeline_fails_the_device_within_the_limit() {
    crate::gpu_or_skip!();
    for via_proxy in [false, true] {
        let device = WgpuDevice::open_native().unwrap().unwrap();
        device.set_watchdog(stalled_trace());
        let gpu: Arc<dyn Gpu> = if via_proxy {
            Arc::new(proxy::spawn_native(device).unwrap().0)
        } else {
            Arc::new(device)
        };
        // The numerics probe's pipeline compiles: the backend opens.
        let b = WebGpuBackend::with_gpu(gpu.clone()).unwrap();
        assert_eq!(b.unavailable(), None);

        let trace = mklight_webgpu::ops::trace::pipelines();
        let t = Instant::now();
        let e = gpu.prepare(&trace[..1]).unwrap_err().to_string();
        let waited = t.elapsed();
        assert!(waited >= LIMIT, "{waited:?}");
        assert!(waited < LIMIT + Duration::from_secs(20), "{waited:?}");
        assert!(e.contains("trace::mkl_trace"), "{e}");
        assert!(e.contains("did not finish compiling within 0.3 s"), "{e}");
        let why = b.unavailable().expect("the device is marked unavailable");
        assert!(why.contains("trace::mkl_trace"), "{why}");
        assert_eq!(gpu.unavailable(), Some(why.clone()));

        // Every later pipeline fails at once with the same reason, also
        // through kernel preparation.
        let t = Instant::now();
        let e = b
            .prepare_kernels(None, &|_, _| {}, &AtomicBool::new(false))
            .unwrap_err()
            .to_string();
        assert!(t.elapsed() < LIMIT, "{:?}", t.elapsed());
        assert!(e.contains(&why), "{e}");
    }
}

#[test]
fn pipelines_that_compile_leave_the_device_available() {
    crate::gpu_or_skip!();
    let device = WgpuDevice::open_native().unwrap().unwrap();
    device.set_watchdog(Watchdog {
        timeout: Duration::from_secs(600),
        stalled: vec!["no_such_module".into()],
    });
    let b = WebGpuBackend::with_gpu(Arc::new(device)).unwrap();
    b.prepare_kernels(None, &|_, _| {}, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(b.unavailable(), None);
}
