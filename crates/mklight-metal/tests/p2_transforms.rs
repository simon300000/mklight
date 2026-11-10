#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p2_transforms.rs"]
mod common;
use common::*;
use mklight_core::JobContext;
use mklight_gpu::{resident::*, transforms::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
fn backend(library: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions { library })
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
fn read(b: &MetalBackend, r: &PlanResult) -> Vec<u32> {
    b.readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
        .words()
        .to_vec()
}
/// Four chained 256² FFTs of `data`.
fn chain(b: &MetalBackend, data: Vec<[f32; 2]>) -> ResidentPlan {
    let h = upload(b, data, vec![256, 256]);
    let mut p = ResidentPlan::new(snapshot());
    let mut prev = p.import(h.view());
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
    p
}
/// A 1024² off-axis optical correlation over 256 frequencies: many bounded commands.
fn long_correlation(b: &MetalBackend) -> ResidentPlan {
    let n = 1024;
    let mut amplitude = vec![0f32; n * n];
    let mut coverage = vec![0f32; n * n];
    let mut opd = vec![Df32::ZERO; n * n];
    for k in 0..n * n {
        let (u, v) = (
            (2 * (k % n) + 1) as f64 / n as f64 - 1.,
            (2 * (k / n) + 1) as f64 / n as f64 - 1.,
        );
        if u * u + v * v < 1. {
            (amplitude[k], coverage[k], opd[k]) = (1., 1., Df32::from_f64(3. * u * v));
        }
    }
    let frequencies: Vec<f32> = (0..256).flat_map(|i| [i as f32, 0.]).collect();
    let mut p = ResidentPlan::new(snapshot());
    let import = |p: &mut ResidentPlan, ty, shape, data| {
        let i = p.import(
            b.upload_buffer(BufferDesc::new(ty, shape), data)
                .unwrap()
                .view(),
        );
        p.view(i).unwrap()
    };
    let amplitude = import(
        &mut p,
        ElementType::F32,
        vec![n, n],
        BufferData::F32(amplitude),
    );
    let coverage = import(
        &mut p,
        ElementType::F32,
        vec![n, n],
        BufferData::F32(coverage),
    );
    let opd = import(
        &mut p,
        ElementType::DoubleFloat,
        vec![n, n],
        BufferData::DoubleFloat(opd),
    );
    let frequencies = import(
        &mut p,
        ElementType::F32,
        vec![256, 2],
        BufferData::F32(frequencies),
    );
    let output = p.scratch(BufferDesc::new(ElementType::Complex32, vec![256]));
    let (s, c) = 0.2f64.sin_cos();
    p.operations
        .push(ResidentOp::Transforms(Request::OpticalCorrelation(
            OpticalCorrelation {
                stencil: None,
                amplitude,
                opd,
                coverage,
                frequencies,
                frame: [[c, 0., -s], [0., 1., 0.], [s, 0., c]].map(|v| v.map(Df32::from_f64)),
                sin_max: Df32::from_f64(0.1),
                wavelength_over_index: Df32::from_f64(0.00055),
                output,
            },
        )));
    p.outputs = vec![output];
    p
}
#[test]
fn deterministic_concurrent_and_cancellation() {
    let b = backend(LibraryChoice::Precompiled);
    let first = chain(&b, input(256 * 256));
    let other = chain(
        &b,
        input(256 * 256)
            .into_iter()
            .map(|z| [0.3 * z[1] + 0.25, -0.8 * z[0]])
            .collect(),
    );
    let run = |p: &ResidentPlan| {
        read(
            &b,
            &b.submit_plan(p.clone(), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap(),
        )
    };
    let (bits, other_bits) = (run(&first), run(&other));
    assert_ne!(bits, other_bits);
    // Concurrent jobs reproduce their isolated bits; progress is monotone.
    let (c1, c2) = (JobContext::new(), JobContext::new());
    let (a, d) = (
        b.submit_plan(first.clone(), &c1).unwrap(),
        b.submit_plan(other, &c2).unwrap(),
    );
    let mut progress = 0.;
    while !a.is_ready() {
        let q = c1.progress.fraction();
        assert!(q >= progress);
        progress = q;
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    let a = a.wait().unwrap();
    assert_eq!(read(&b, &a), bits);
    assert_eq!(read(&b, &d.wait().unwrap()), other_bits);
    assert!(a.stats.reused_allocations > 0);
    // Cancellation between bounded device commands discards the job.
    let long = long_correlation(&b);
    let ctx = JobContext::new();
    let q = b.submit_plan(long.clone(), &ctx).unwrap();
    while ctx.progress.fraction() < 0.05 && !q.is_ready() {
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    q.cancel();
    assert!(matches!(q.wait(), Err(mklight_core::MkError::Cancelled)));
    let r = b
        .submit_plan(long, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert!(r.stats.command_buffers > 8, "{:?}", r.stats);
    assert_eq!(run(&first), bits);
}
