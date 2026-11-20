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
