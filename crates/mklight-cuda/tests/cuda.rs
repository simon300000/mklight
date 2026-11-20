//! CUDA-specific checks: the device, the numerics probe of plain and
//! protected expressions, and kernel compilation.

use mklight_gpu::{ComputeBackend, JobKind};

#[test]
fn probe_and_identity() {
    let b = crate::gpu_or_skip!();
    let p = b.precision().probe;
    eprintln!("adapter {:?}", b.adapter());
    eprintln!("raw probe {:?}", b.raw_probe());
    eprintln!("protected probe {p:?}");
    eprintln!("kernel build: {}", b.precision().kernel_build);
    assert!(p.error_free_transforms(), "{p:?}");
    assert!(b.raw_probe().error_free_transforms(), "{:?}", b.raw_probe());
    assert!(b.supports(JobKind::DfOps) && b.supports(JobKind::Reduce));
    assert_eq!(b.info().backend_id(), format!("cuda:{}", b.adapter().name));
}

/// A device whose context a sticky error made unusable is unavailable with
/// the reason, and fails every later request at once (the engine then
/// routes `auto` to the CPU and refuses explicit requests). Its own
/// device, so the shared test backend stays usable.
#[test]
fn lost_device_is_unavailable() {
    use mklight_core::JobContext;
    use mklight_gpu::{Df32, DfOp, Job, QualificationBackend};
    if crate::gpu().is_none() {
        return;
    }
    let device = std::sync::Arc::new(mklight_cuda::CudaDevice::open().unwrap().unwrap());
    let b = mklight_webgpu::WebGpuBackend::with_gpu(device.clone()).unwrap();
    let job = || Job::DfOps {
        op: DfOp::Add,
        a: vec![Df32::default(); 4],
        b: vec![Df32::default(); 4],
    };
    assert_eq!(b.unavailable(), None);
    b.run(job(), &JobContext::new()).unwrap();
    device.fail_for_test("simulated device loss");
    assert_eq!(b.unavailable().as_deref(), Some("simulated device loss"));
    let e = b.run(job(), &JobContext::new()).unwrap_err().to_string();
    assert!(e.contains("simulated device loss"), "{e}");
}
