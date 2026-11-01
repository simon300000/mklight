use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_optim::{gpu::GpuContext, linalg::Matrix};
use std::sync::Arc;
fn check(backend: Arc<dyn ComputeBackend>) {
    let g = GpuContext::new(
        backend,
        Snapshot::new(mklight_core::samples::singlet()).unwrap(),
    );
    let ctx = JobContext::new();
    // A tiny difference of large operands cannot survive an f32 subtraction.
    let h = 0.001;
    let x = 0.25;
    let f = |x: f64| 1e5 + x * x;
    let j = g
        .differences(
            &[f(x), f(x + h), f(x - h), f(x + 2.0 * h)],
            4,
            1,
            &[[0, 1, 2], [0, 1, 3], [0, 2, 0]],
            &[[0.5 / h, -0.5 / h], [2.0 / h, -0.5 / h], [-1.0 / h, 0.0]],
            &ctx,
        )
        .unwrap();
    for (a, b) in j.data.iter().zip([0.5, 0.5, 0.499]) {
        assert!((a - b).abs() < 2e-6, "{a} != {b}");
    }
    // Two independent overdetermined systems, known exact coefficients.
    let j = Matrix::from_row_major(3, 2, vec![1., 0., 0., 1., 1., 1.]);
    let result = g
        .least_squares(
            &[j.data.clone(), j.data.clone()].concat(),
            &[-2., 3., 1., -4., -1., -5.],
            &[1.; 3],
            &[1.; 4],
            2,
            3,
            2,
            0.,
            &ctx,
        )
        .unwrap();
    let x = g.values(&result, 0).unwrap();
    for (a, b) in x.iter().zip([2., -3., 4., 1.]) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
    assert_eq!(result.stats.copied_bytes, 0);
}
#[test]
fn emulation_known_answers() {
    check(Arc::new(CpuF32Backend::default()));
}
#[cfg(target_os = "macos")]
#[test]
fn host_metal_known_answers() {
    use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P8 requires host Metal");
        assert!(b.precision().probe.error_free_transforms());
        eprintln!("P8 {:?} {:?}", b.info(), b.precision());
        check(Arc::new(b));
    }
}
