#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p2_transforms.rs"]
mod common;
use common::*;
use mklight_core::JobContext;
use mklight_gpu::{resident::*, transforms::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
fn backend(library: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions {
        library,
        ..Default::default()
    })
    .unwrap()
    .expect("P2 qualification requires a host Metal device");
    eprintln!(
        "P2 device={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    assert!(b.precision().probe.error_free_transforms());
    b
}
#[test]
fn actual_metal_both_libraries() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = backend(origin);
        suite(&b);
        correlation_suite(&b);
        primitives_suite(&b);
        phase_and_views(&b);
        complex_correlation_suite(&b);
        physical_grid_mft(&b);
    }
}
#[test]
fn deterministic_reuse_concurrent_and_cancellation() {
    let b = backend(LibraryChoice::Precompiled);
    let a = input(256 * 256);
    let h = upload(&b, a, vec![256, 256]);
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(h.view());
    let mut prev = i;
    for _ in 0..4 {
        let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![256, 256]));
        p.operations
            .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
                kind: TransformKind::Fft,
                input: p.view(prev).unwrap(),
                output: o,
                dimensions: vec![256, 256],
                output_dimensions: vec![256, 256],
                sign: -1,
                scale: 1. / 256.,
                affine: [Df32::ZERO; 6],
            })));
        prev = o;
    }
    p.outputs = vec![prev];
    let read = |r: &PlanResult| {
        b.readback(r.outputs[0].view(), &JobContext::new())
            .unwrap()
            .data
            .words()
            .to_vec()
    };
    let r = b
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let bits = read(&r);
    drop(r);
    let c1 = JobContext::new();
    let c2 = JobContext::new();
    let a = b.submit_plan(p.clone(), &c1).unwrap();
    let d = b.submit_plan(p.clone(), &c2).unwrap();
    let mut progress = 0.;
    while !a.is_ready() {
        let q = c1.progress.fraction();
        assert!(q >= progress);
        progress = q;
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let a = a.wait().unwrap();
    let d = d.wait().unwrap();
    assert_eq!(read(&a), bits);
    assert_eq!(read(&d), bits);
    assert!(a.stats.reused_allocations > 0);
    let ctx = JobContext::new();
    let q = b.submit_plan(p.clone(), &ctx).unwrap();
    while ctx.progress.fraction() < 0.15 && !q.is_ready() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    q.cancel();
    assert!(matches!(q.wait(), Err(mklight_core::MkError::Cancelled)));
    let q = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(read(&q), bits);
}
