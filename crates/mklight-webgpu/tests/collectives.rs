//! Segmented reductions on the device against the CPU-f32 specification in
//! the device's environment, bit for bit (the WebGPU twin of the segment
//! cases of `mklight-metal/tests/p3_collectives.rs`), and the known answers
//! of `mklight-gpu/tests/p3_cases.rs`.

use mklight_gpu::collectives::*;
use mklight_gpu::resident::*;
use mklight_gpu::{ComputeBackend, CpuF32Backend, Df32, QualificationBackend, ReduceMode};

#[path = "../../mklight-gpu/tests/p3_cases.rs"]
mod p3_cases;

/// Every segment operation of the plan ran on the device.
fn on_device(r: &PlanResult) {
    assert!(
        r.stats
            .stages
            .iter()
            .any(|s| s.name == "webgpu:segment_reduce"),
        "{:?}",
        r.stats.stages
    );
}

fn same(
    g: &dyn QualificationBackend,
    c: &CpuF32Backend,
    build: &dyn Fn(&dyn QualificationBackend) -> ResidentPlan,
) {
    let (a, _) = p3_cases::read(c, build(c));
    let (b, r) = p3_cases::read(g, build(g));
    on_device(&r);
    for (a, b) in a.iter().zip(&b) {
        p3_cases::assert_bits(a, b);
    }
}

#[test]
fn segment_known_answers() {
    let g = crate::gpu_or_skip!();
    p3_cases::segment_answers(g);
}

/// Values of every kind and mode, segments empty, short and spanning
/// several tree groups, f32 and double-float, invalid offsets.
fn random_case(
    b: &dyn QualificationBackend,
    kind: SegmentKind,
    mode: ReduceMode,
    df: bool,
    channels: usize,
) -> ResidentPlan {
    let offsets: Vec<u32> = vec![0, 0, 1, 7, 300, 4096, 4097, 13001, 30001, 30002, 29000];
    let n = 30002;
    let value = |i: usize| {
        let u = ((i as u64 * 2654435761) % 1000003) as f64 / 1000003.;
        (u - 0.3) * [1., 1e4, 1e-3][i % 3] + if i.is_multiple_of(97) { 1e7 } else { 0. }
    };
    let data = if df {
        BufferData::DoubleFloat(
            (0..channels * n)
                .map(|i| Df32::from_f64(value(i)))
                .collect(),
        )
    } else {
        BufferData::F32((0..channels * n).map(|i| value(i) as f32).collect())
    };
    let mut p = p3_cases::segment(b, data, channels, offsets, kind, None, 17001);
    let ResidentOp::Collectives(Request::SegmentReduce(r)) = &mut p.operations[0] else {
        unreachable!()
    };
    r.mode = mode;
    p
}

#[test]
fn segment_reduce_matches_cpu_bitwise() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    for mode in [
        ReduceMode::Plain,
        ReduceMode::Kahan,
        ReduceMode::DoubleFloat,
    ] {
        for df in [false, true] {
            for (kind, channels) in [
                (SegmentKind::Sum, 2),
                (SegmentKind::Summary, 2),
                (SegmentKind::SourceProducts, 3),
                (SegmentKind::Moments2, 3),
            ] {
                eprintln!("{kind:?} {mode:?} df={df}");
                same(g, &c, &|b| random_case(b, kind, mode, df, channels));
            }
        }
    }
    // Moments with a reference, a mask and separate weights (rejected
    // records hold NaN); short segments and one spanning several groups.
    let short: Vec<u32> = (0..=600)
        .map(|s| s * 61 / 2)
        .chain([18_300, 30_001])
        .collect();
    same(g, &c, &|b| {
        p3_cases::moments_case(b, short[..601].to_vec(), 61)
    });
    same(g, &c, &|b| {
        p3_cases::moments_case(b, short[..601].to_vec(), 18_300)
    });
    same(g, &c, &|b| p3_cases::moments_case(b, short.clone(), 30_001));
}
