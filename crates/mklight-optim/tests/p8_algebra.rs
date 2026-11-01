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

fn raw_algebra(b: Arc<dyn ComputeBackend>) -> Vec<Vec<u32>> {
    use mklight_gpu::{optim_ops::Request, resident::*};
    let g = GpuContext::new(b, Snapshot::new(mklight_core::samples::singlet()).unwrap());
    let mut p = g.plan();
    let values = g.df(&mut p, [1, 3], &[2., -1., -4.]).unwrap();
    let targets = g.df(&mut p, [3], &[1., 0., 0.]).unwrap();
    let comparison = g
        .fill(&mut p, [3], |v: &mut [u32]| v.copy_from_slice(&[0, 1, 2]))
        .unwrap();
    let residual = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 3]));
    p.operations.push(ResidentOp::Optim(Request::Residuals {
        values: values.clone(),
        targets,
        comparison,
        output: residual,
    }));
    let delta = g
        .df(&mut p, [2, 3], &[1e-7, 2e-7, -3e-7, -1., 4., 8.])
        .unwrap();
    let perturbed = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, 3]));
    p.operations.push(ResidentOp::Optim(Request::Perturb {
        nominal: values.clone(),
        delta,
        output: perturbed,
    }));
    let mut source = values;
    source.desc.shape = vec![3];
    let mask = g.fill(&mut p, [3], |v: &mut [u32]| v.fill(0)).unwrap();
    let indices = g
        .fill(&mut p, [1, 3], |v: &mut [u32]| {
            v.copy_from_slice(&[0, 1, 2])
        })
        .unwrap();
    let basis = g.df(&mut p, [1, 3, 2], &[1., 0., 1., 1., 1., 2.]).unwrap();
    let matrix = p.scratch(basis.desc.clone());
    let rhs = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 3]));
    p.operations.push(ResidentOp::Optim(Request::FitRows {
        values: source,
        mask,
        indices,
        basis,
        matrix,
        rhs,
    }));
    let coefficients = g.df(&mut p, [1, 2], &[2., -3.]).unwrap();
    let error = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, 3]));
    p.operations.push(ResidentOp::Optim(Request::FitError {
        matrix: p.view(matrix).unwrap(),
        rhs: p.view(rhs).unwrap(),
        coefficients,
        output: error,
    }));
    let right = g.df(&mut p, [1, 3, 1], &[2., -1., -4.]).unwrap();
    let product = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 2, 1]));
    p.operations.push(ResidentOp::Optim(Request::Product {
        left: p.view(matrix).unwrap(),
        right,
        transpose_left: true,
        output: product,
    }));
    p.outputs = vec![residual, perturbed, matrix, rhs, error, product];
    let r = g.submit(p, &JobContext::new()).unwrap();
    assert_eq!(g.values(&r, 0).unwrap(), vec![1., 0., -4.]);
    assert_eq!(g.values(&r, 4).unwrap(), vec![0., 0., 0., 1., 1., 1.]);
    assert_eq!(g.values(&r, 5).unwrap(), vec![-3., -9.]);
    r.outputs
        .iter()
        .map(|o| g.backend.read_view(o.view()).unwrap().words().to_vec())
        .collect()
}
#[test]
fn all_algebra_emulation() {
    raw_algebra(Arc::new(CpuF32Backend::new()));
}
#[cfg(target_os = "macos")]
#[test]
fn all_algebra_metal_matches_emulation() {
    use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P8 requires host Metal");
        let cpu = CpuF32Backend::matching(&b.precision().probe);
        assert_eq!(raw_algebra(Arc::new(cpu)), raw_algebra(Arc::new(b)));
    }
}
