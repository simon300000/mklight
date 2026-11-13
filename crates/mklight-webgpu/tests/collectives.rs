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

/// Focus terms and characteristic terms feed segment sums in one plan whose
/// inputs are imported: bitwise the CPU-f32 plan's, and no operation reads an
/// intermediate back (each takes one command buffer; only the final outputs
/// are read after the last operation).
fn chained(b: &dyn QualificationBackend) -> ResidentPlan {
    use mklight_gpu::geometry::{Characteristic, FocusTerms, Request as Geometry, view};
    let (len, segs, k) = (37usize, 9usize, 3usize);
    let n = len * segs;
    let value = |i: usize| ((i as u64 * 2654435761) % 1000003) as f64 / 1000003. - 0.4;
    let df = |count: usize, salt: usize| {
        BufferData::DoubleFloat(
            (0..count)
                .map(|i| Df32::from_f64(value(i + salt) * [1., 3., 0.01][i % 3]))
                .collect(),
        )
    };
    let mut p = p3_cases::plan();
    let points = p3_cases::import(b, &mut p, [5, n], df(5 * n, 0));
    let mask = p3_cases::import(
        b,
        &mut p,
        [n],
        BufferData::U32((0..n as u32).map(|i| u32::from(i % 7 != 3)).collect()),
    );
    let position_moments = p3_cases::import(b, &mut p, [segs, 10], df(10 * segs, 11));
    let slope_moments = p3_cases::import(b, &mut p, [segs, 10], df(10 * segs, 29));
    let frequency = p3_cases::import(b, &mut p, [k, segs, 2], df(k * segs * 2, 5));
    let focus = p3_cases::scratch(&mut p, ElementType::DoubleFloat, [3, n]);
    let characteristic = p3_cases::scratch(&mut p, ElementType::DoubleFloat, [2, k * n]);
    p.operations
        .push(ResidentOp::Geometry(Geometry::FocusTerms(FocusTerms {
            points: points.clone(),
            mask: mask.clone(),
            position_moments,
            slope_moments,
            segment_len: len,
            centroid: true,
            output: focus,
        })));
    p.operations
        .push(ResidentOp::Geometry(Geometry::Characteristic(
            Characteristic {
                points,
                mask,
                segment_len: len,
                frequency,
                output: characteristic,
            },
        )));
    let mut outputs = vec![];
    for (id, channels, count) in [(focus, 3, n), (characteristic, 2, k * n)] {
        let offsets: Vec<u32> = (0..=segs).map(|s| (s * count / segs) as u32).collect();
        let offsets = p3_cases::import(b, &mut p, [segs + 1], BufferData::U32(offsets));
        let status = p3_cases::scratch(&mut p, ElementType::U32, [segs]);
        let mut r = SegmentReduce {
            mask: None,
            mask_value: 0,
            weights: None,
            values: view(id, ElementType::DoubleFloat, [channels, count]),
            offsets,
            reference: None,
            max_segment_len: count / segs + 1,
            mode: ReduceMode::DoubleFloat,
            kind: SegmentKind::Sum,
            output: BufferId(0),
            status,
        };
        r.output = p3_cases::scratch(&mut p, ElementType::DoubleFloat, segment_shape(&r));
        outputs.extend([r.output, status]);
        p.operations
            .push(ResidentOp::Collectives(Request::SegmentReduce(r)));
    }
    p.outputs = outputs;
    p
}

#[test]
fn geometry_to_segments_stays_on_device() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let (a, _) = p3_cases::read(&c, chained(&c));
    let (b, r) = p3_cases::read(g, chained(g));
    // Focus sums and statuses are bitwise; the characteristic's cos/sin are
    // within the geometry tolerance, so are its sums.
    for (a, b) in a[..2].iter().zip(&b[..2]) {
        p3_cases::assert_bits(a, b);
    }
    p3_cases::assert_bits(&a[3], &b[3]);
    for (x, y) in p3_cases::df(&a[2]).iter().zip(p3_cases::df(&b[2])) {
        assert!((x - y).abs() <= 1e-4 * (1. + x.abs()), "{x} {y}");
    }
    let stages: Vec<_> = r
        .stats
        .stages
        .iter()
        .filter(|s| s.name != "metadata")
        .map(|s| (s.name.as_str(), s.command_buffers))
        .collect();
    let names: Vec<_> = stages.iter().map(|s| s.0).collect();
    assert_eq!(
        names,
        [
            "webgpu:focus_terms",
            "webgpu:characteristic",
            "webgpu:segment_reduce",
            "webgpu:segment_reduce"
        ]
    );
    // One batch each: nothing was read back between the operations.
    assert!(stages.iter().all(|s| s.1 == 1), "{stages:?}");
}
