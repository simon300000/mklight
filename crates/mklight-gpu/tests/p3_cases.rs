#![allow(dead_code)]
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{collectives::*, resident::*, *};
pub fn plan() -> ResidentPlan {
    ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap())
}
pub fn import(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    shape: impl Into<Vec<usize>>,
    data: BufferData,
) -> PlanView {
    let h = b
        .upload_buffer(BufferDesc::new(data.element(), shape), data)
        .unwrap();
    let id = p.import(h.view());
    p.view(id).unwrap()
}
pub fn scratch(p: &mut ResidentPlan, t: ElementType, s: impl Into<Vec<usize>>) -> BufferId {
    p.scratch(BufferDesc::new(t, s))
}
pub fn read(b: &dyn ComputeBackend, p: ResidentPlan) -> (Vec<BufferData>, PlanResult) {
    let (v, out, _, _) = read_timed(b, p);
    (v, out)
}
pub fn read_timed(
    b: &dyn ComputeBackend,
    mut p: ResidentPlan,
) -> (Vec<BufferData>, PlanResult, f64, u64) {
    p.settings["p3_requests"] = serde_json::json!(
        p.operations
            .iter()
            .filter_map(|op| match op {
                ResidentOp::Collectives(r) => Some(format!("{r:?}")),
                _ => None,
            })
            .collect::<Vec<_>>()
    );
    let out = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(out.stats.readback_bytes, 0);
    assert_eq!(out.stats.fallback_count, 0);
    let mut seconds = 0.0;
    let mut bytes = 0;
    let v = out
        .outputs
        .iter()
        .map(|h| {
            // Empty leases execute and publish. P0 empty DF readback has an
            // alignment bug; the evidence records this separate runtime gap.
            if h.desc().len().unwrap() == 0 {
                return match h.desc().element {
                    ElementType::U32 => BufferData::U32(vec![]),
                    ElementType::F32 => BufferData::F32(vec![]),
                    ElementType::DoubleFloat => BufferData::DoubleFloat(vec![]),
                    ElementType::Complex32 => BufferData::Complex32(vec![]),
                };
            }
            let r = b.readback(h.view(), &JobContext::new()).unwrap();
            seconds += r.wall_seconds;
            bytes += r.copied_bytes;
            r.data
        })
        .collect();
    (v, out, seconds, bytes)
}
pub fn df(v: &BufferData) -> Vec<f64> {
    let BufferData::DoubleFloat(v) = v else {
        panic!("expected DF")
    };
    v.iter().map(|v| v.to_f64()).collect()
}
pub fn uint(v: &BufferData) -> &[u32] {
    let BufferData::U32(v) = v else {
        panic!("expected u32")
    };
    v
}
pub fn assert_bits(a: &BufferData, b: &BufferData) {
    assert_eq!(a.element(), b.element());
    assert_eq!(a.words().len(), b.words().len());
    let integer = a.element() == ElementType::U32;
    for (i, (&a, &b)) in a.words().iter().zip(b.words()).enumerate() {
        let (af, bf) = (f32::from_bits(a), f32::from_bits(b));
        assert!(
            a == b || !integer && af.is_nan() && bf.is_nan(),
            "word {i}: {a:08x} != {b:08x}"
        );
    }
}
pub fn scan_case(b: &dyn ComputeBackend, n: usize, integer: bool, inclusive: bool) -> ResidentPlan {
    let mut p = plan();
    let data = if integer {
        BufferData::U32(
            (0..n)
                .map(|i| if i % 3 == 0 { u32::MAX } else { 2 })
                .collect(),
        )
    } else {
        BufferData::DoubleFloat(
            (0..n)
                .map(|i| Df32::from_f64([1e8, 1., -1e8, 1e-10][i % 4]))
                .collect(),
        )
    };
    let v = import(b, &mut p, [n], data);
    let o = scratch(
        &mut p,
        if integer {
            ElementType::U32
        } else {
            ElementType::DoubleFloat
        },
        [n],
    );
    p.operations
        .push(ResidentOp::Collectives(Request::Scan(Scan {
            values: v,
            inclusive,
            output: o,
        })));
    p.outputs = vec![o];
    p
}
pub fn scan_answers(b: &dyn ComputeBackend) {
    for n in [0, 1, 255, 256, 257, 65539] {
        for inclusive in [true, false] {
            for integer in [true, false] {
                let (v, _) = read(b, scan_case(b, n, integer, inclusive));
                if integer {
                    let mut want = 0u32;
                    for (i, &got) in uint(&v[0]).iter().enumerate() {
                        let val = if i % 3 == 0 { u32::MAX } else { 2 };
                        if inclusive {
                            want = want.wrapping_add(val);
                        }
                        assert_eq!(got, want);
                        if !inclusive {
                            want = want.wrapping_add(val);
                        }
                    }
                } else {
                    for (i, got) in df(&v[0]).iter().enumerate() {
                        let k = i + usize::from(inclusive);
                        let want = (k / 4) as f64 * (1. + 1e-10) + [0., 1e8, 1e8 + 1., 1.][k % 4];
                        assert!(
                            (got - want).abs() < 1e-5,
                            "scan n={n} i={i} got={got} want={want}"
                        );
                    }
                }
            }
        }
    }
}
pub fn sort_case(b: &dyn ComputeBackend, n: usize, key_type: ElementType) -> ResidentPlan {
    let mut p = plan();
    let raw: Vec<_> = (0..n).map(|i| ((i * 7919) % 17) as u32).collect();
    let keys = match key_type {
        ElementType::U32 => BufferData::U32(raw.clone()),
        ElementType::F32 => BufferData::F32(raw.iter().map(|&v| v as f32).collect()),
        _ => BufferData::DoubleFloat(
            raw.iter()
                .map(|&v| Df32::from_f64(v as f64 + v as f64 * 1e-9))
                .collect(),
        ),
    };
    let k = import(b, &mut p, [n], keys);
    let ids = import(
        b,
        &mut p,
        [n],
        BufferData::U32((0..n).map(|i| ((n - i) * 13 % 31) as u32).collect()),
    );
    let v = import(
        b,
        &mut p,
        [n, 2],
        BufferData::U32((0..n).flat_map(|i| [i as u32, !(i as u32)]).collect()),
    );
    let ok = scratch(&mut p, key_type, [n]);
    let oi = scratch(&mut p, ElementType::U32, [n]);
    let ov = scratch(&mut p, ElementType::U32, [n, 2]);
    p.operations
        .push(ResidentOp::Collectives(Request::StableSort(StableSort {
            keys: k,
            ids,
            values: Some(v),
            output_keys: ok,
            output_ids: oi,
            output_values: Some(ov),
        })));
    p.outputs = vec![ok, oi, ov];
    p
}
pub fn sort_answers(b: &dyn ComputeBackend) {
    let mut p = plan();
    let keys = import(
        b,
        &mut p,
        [4],
        BufferData::DoubleFloat(vec![
            Df32::new(f32::NAN, 3.),
            Df32::new(2., f32::NAN),
            Df32::from_f32(1.),
            Df32::new(f32::NAN, -2.),
        ]),
    );
    let ids = import(b, &mut p, [4], BufferData::U32(vec![3, 1, 7, 2]));
    let ok = scratch(&mut p, ElementType::DoubleFloat, [4]);
    let oi = scratch(&mut p, ElementType::U32, [4]);
    p.operations
        .push(ResidentOp::Collectives(Request::StableSort(StableSort {
            keys,
            ids,
            values: None,
            output_keys: ok,
            output_ids: oi,
            output_values: None,
        })));
    p.outputs = vec![oi];
    let (v, _) = read(b, p);
    assert_eq!(uint(&v[0]), &[7, 1, 2, 3]);

    for n in [0, 1, 257, 10003] {
        for t in [ElementType::U32, ElementType::F32, ElementType::DoubleFloat] {
            let (v, _) = read(b, sort_case(b, n, t));
            let mut expected: Vec<_> = (0..n).collect();
            expected.sort_by_key(|&i| ((i * 7919) % 17, (n - i) * 13 % 31, i));
            for (j, &i) in expected.iter().enumerate() {
                assert_eq!(uint(&v[1])[j], ((n - i) * 13 % 31) as u32);
                assert_eq!(&uint(&v[2])[2 * j..2 * j + 2], &[i as u32, !(i as u32)]);
            }
        }
    }
    let mut p = plan();
    let keys = import(
        b,
        &mut p,
        [8],
        BufferData::F32(vec![
            f32::NAN,
            0.,
            -0.,
            f32::INFINITY,
            -2.,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-39,
        ]),
    );
    let ids = import(
        b,
        &mut p,
        [8],
        BufferData::U32(vec![1, 5, 2, 0, 9, 0, 0, 1]),
    );
    let ok = scratch(&mut p, ElementType::F32, [8]);
    let oi = scratch(&mut p, ElementType::U32, [8]);
    p.operations
        .push(ResidentOp::Collectives(Request::StableSort(StableSort {
            keys,
            ids,
            values: None,
            output_keys: ok,
            output_ids: oi,
            output_values: None,
        })));
    p.outputs = vec![oi];
    let (v, _) = read(b, p);
    assert_eq!(uint(&v[0]), &[0, 9, 2, 5, 1, 0, 0, 1]);
}
pub fn compact_answers(b: &dyn ComputeBackend) {
    for n in [0, 1, 257, 10003] {
        for none in [false, true] {
            let mut p = plan();
            let values = import(
                b,
                &mut p,
                [n, 2],
                BufferData::DoubleFloat(
                    (0..n * 2)
                        .map(|i| Df32::from_f64(i as f64 + 1e-8))
                        .collect(),
                ),
            );
            let ids = import(
                b,
                &mut p,
                [n],
                BufferData::U32((0..n).map(|i| (n - i) as u32).collect()),
            );
            let mask = import(
                b,
                &mut p,
                [n],
                BufferData::U32((0..n).map(|i| u32::from(!none && i % 3 != 0) * 7).collect()),
            );
            let out = scratch(&mut p, ElementType::DoubleFloat, [n, 2]);
            let oi = scratch(&mut p, ElementType::U32, [n]);
            let cnt = scratch(&mut p, ElementType::U32, [1]);
            p.operations
                .push(ResidentOp::Collectives(Request::StableCompact(
                    StableCompact {
                        values,
                        ids,
                        mask,
                        output: out,
                        output_ids: oi,
                        count: cnt,
                    },
                )));
            p.outputs = vec![out, oi, cnt];
            let (v, _) = read(b, p);
            let keep: Vec<_> = (0..n).filter(|i| !none && i % 3 != 0).collect();
            assert_eq!(uint(&v[2]), &[keep.len() as u32]);
            let vals = df(&v[0]);
            for (j, &i) in keep.iter().enumerate() {
                assert_eq!(uint(&v[1])[j], (n - i) as u32);
                assert!((vals[2 * j] - (2 * i) as f64 - 1e-8).abs() < 1e-10);
            }
            assert!(uint(&v[1])[keep.len()..].iter().all(|&v| v == u32::MAX));
            assert!(vals[2 * keep.len()..].iter().all(|&v| v == 0.));
        }
    }
}
pub fn segment(
    b: &dyn ComputeBackend,
    values: BufferData,
    c: usize,
    offsets: Vec<u32>,
    kind: SegmentKind,
    reference: Option<BufferData>,
    bound: usize,
) -> ResidentPlan {
    let mut p = plan();
    let n = values.words().len() / values.element().words() / c;
    let s = offsets.len() - 1;
    let values = import(b, &mut p, [c, n], values);
    let offsets = import(b, &mut p, [s + 1], BufferData::U32(offsets));
    let reference = reference.map(|d| import(b, &mut p, [s, 2], d));
    let status = scratch(&mut p, ElementType::U32, [s]);
    let mut r = SegmentReduce {
        mask: None,
        mask_value: 0,
        weights: None,
        values,
        offsets,
        reference,
        max_segment_len: bound,
        mode: ReduceMode::DoubleFloat,
        kind,
        output: BufferId(0),
        status,
    };
    r.output = scratch(&mut p, ElementType::DoubleFloat, segment_shape(&r));
    p.outputs = vec![r.output, status];
    p.operations
        .push(ResidentOp::Collectives(Request::SegmentReduce(r)));
    p
}
pub fn segment_answers(b: &dyn ComputeBackend) {
    let (v, _) = read(
        b,
        segment(
            b,
            BufferData::F32(vec![]),
            1,
            vec![0],
            SegmentKind::Sum,
            None,
            0,
        ),
    );
    assert!(df(&v[0]).is_empty() && uint(&v[1]).is_empty());

    let (v, _) = read(
        b,
        segment(
            b,
            BufferData::F32(vec![0., 1., 0., 1., 1., 1.]),
            3,
            vec![0, 0, 2],
            SegmentKind::Moments2,
            None,
            2,
        ),
    );
    let empty = df(&v[0]);
    assert_eq!(&empty[..2], &[0., 0.]);
    assert!(empty[2].is_nan() && empty[3].is_nan());
    assert_eq!(&empty[8..10], &[0., 0.]);

    let values: Vec<_> = (0..20003).map(|i| [1e8f32, 1., -1e8, 2.][i % 4]).collect();
    let offsets = vec![0, 0, 1, 257, 5003, 20003];
    let (v, _) = read(
        b,
        segment(
            b,
            BufferData::F32(values.clone()),
            1,
            offsets.clone(),
            SegmentKind::Summary,
            None,
            20003,
        ),
    );
    let got = df(&v[0]);
    for (s, pair) in offsets.windows(2).enumerate() {
        let slice = &values[pair[0] as usize..pair[1] as usize];
        let sum: f64 = slice.iter().map(|&v| v as f64).sum();
        assert!((got[s * 4] - sum).abs() < 1e-6);
        assert_eq!(got[s * 4 + 1], slice.len() as f64);
        assert_eq!(
            got[s * 4 + 2],
            slice
                .iter()
                .map(|&v| v as f64)
                .fold(f64::INFINITY, f64::min)
        );
        assert_eq!(
            got[s * 4 + 3],
            slice
                .iter()
                .map(|&v| v as f64)
                .fold(f64::NEG_INFINITY, f64::max)
        );
    }
    assert!(uint(&v[1]).iter().all(|&s| s == 0));
    let src = vec![1., 3., 2., 0.5, 1.5, -1., 2., 4., -1., 2., 3., 1.];
    let (v, _) = read(
        b,
        segment(
            b,
            BufferData::F32(src),
            2,
            vec![0, 2, 3, 6],
            SegmentKind::SourceProducts,
            None,
            3,
        ),
    );
    assert_eq!(
        df(&v[0]),
        vec![
            4., 6., 16., 24., 36., 2., -1., 4., -2., 1., 1., 6., 1., 6., 36.
        ]
    );
    // Independent source units give Qxx=21, Qxy=28, Qyy=73; child squares
    // would give Qxx=17.5 and lose the split-path covariance.
    let rows = df(&v[0]);
    assert_eq!(rows[2] + rows[7] + rows[12], 21.);
    assert_eq!(rows[3] + rows[8] + rows[13], 28.);
    assert_eq!(rows[4] + rows[9] + rows[14], 73.);
    let (v, _) = read(
        b,
        segment(
            b,
            BufferData::F32(vec![1., 2., 3.]),
            1,
            vec![0, 4, 2, 3],
            SegmentKind::Sum,
            None,
            3,
        ),
    );
    assert_eq!(uint(&v[1]), &[1, 1, 0]);
    assert!(df(&v[0])[0].is_nan());
    assert_eq!(df(&v[0])[2], 3.);
    let x = [10000. - 3e-5, 10000. + 1e-5, 10000. + 2e-5];
    let y = [-2000. + 1e-5, -2000. - 2e-5, -2000. + 4e-5];
    let w = [1., 2., 3.];
    let values = BufferData::DoubleFloat(
        x.into_iter()
            .chain(y)
            .chain(w)
            .map(Df32::from_f64)
            .collect(),
    );
    let (v, _) = read(
        b,
        segment(
            b,
            values,
            3,
            vec![0, 3],
            SegmentKind::Moments2,
            Some(BufferData::DoubleFloat(vec![
                Df32::from_f64(10000.),
                Df32::from_f64(-2000.),
            ])),
            3,
        ),
    );
    let g = df(&v[0]);
    let mx = (0..3).map(|i| w[i] * (x[i] - 10000.)).sum::<f64>() / 6.;
    let my = (0..3).map(|i| w[i] * (y[i] + 2000.)).sum::<f64>() / 6.;
    let xx = (0..3)
        .map(|i| w[i] * (x[i] - 10000. - mx).powi(2))
        .sum::<f64>();
    let xy = (0..3)
        .map(|i| w[i] * (x[i] - 10000. - mx) * (y[i] + 2000. - my))
        .sum::<f64>();
    let yy = (0..3)
        .map(|i| w[i] * (y[i] + 2000. - my).powi(2))
        .sum::<f64>();
    assert_eq!(g[0], 3.);
    assert_eq!(g[1], 6.);
    for (a, b) in [g[2], g[3], g[5], g[6], g[7]]
        .into_iter()
        .zip([mx, my, xx, xy, yy])
    {
        assert!((a - b).abs() < 3e-12, "moments {a:e} {b:e}");
    }
}
pub fn qr_plan(
    b: &dyn ComputeBackend,
    m: usize,
    n: usize,
    batch: usize,
    a: Vec<f64>,
    rhs: Vec<f64>,
    tol: f32,
) -> ResidentPlan {
    let mut p = plan();
    let matrix = import(
        b,
        &mut p,
        [batch, m, n],
        BufferData::DoubleFloat(a.into_iter().map(Df32::from_f64).collect()),
    );
    let rhs = import(
        b,
        &mut p,
        [batch, m],
        BufferData::DoubleFloat(rhs.into_iter().map(Df32::from_f64).collect()),
    );
    let factors = scratch(&mut p, ElementType::DoubleFloat, [batch, m + n + 2, n]);
    let solution = scratch(&mut p, ElementType::DoubleFloat, [batch, n]);
    let rank_condition = scratch(&mut p, ElementType::DoubleFloat, [batch, 4]);
    p.operations
        .push(ResidentOp::Collectives(Request::QrFit(QrFit {
            matrix,
            rhs,
            rank_tolerance: tol,
            factors,
            solution,
            rank_condition,
        })));
    p.outputs = vec![factors, solution, rank_condition];
    p
}
// Independent f64 Householder reference, never a normal-equation solve.
pub fn reference_qr(a: &[f64], rhs: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut b = rhs.to_vec();
    for k in 0..n {
        let norm = (k..m)
            .map(|i| a[i * n + k] * a[i * n + k])
            .sum::<f64>()
            .sqrt();
        let alpha = -norm.copysign(a[k * n + k]);
        let mut v: Vec<_> = (k..m).map(|i| a[i * n + k]).collect();
        v[0] -= alpha;
        let vv = v.iter().map(|x| x * x).sum::<f64>();
        for j in k..n {
            let d = 2.
                * v.iter()
                    .enumerate()
                    .map(|(i, v)| v * a[(i + k) * n + j])
                    .sum::<f64>()
                / vv;
            for (i, &v) in v.iter().enumerate() {
                a[(i + k) * n + j] -= d * v;
            }
        }
        let d = 2. * v.iter().enumerate().map(|(i, v)| v * b[i + k]).sum::<f64>() / vv;
        for (i, &v) in v.iter().enumerate() {
            b[i + k] -= d * v;
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        let s = (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    x
}
pub fn qr_check(
    b: &dyn ComputeBackend,
    a: Vec<f64>,
    rhs: Vec<f64>,
    m: usize,
    n: usize,
    tol: f32,
    expected_rank: usize,
) -> f64 {
    let (v, _) = read(b, qr_plan(b, m, n, 1, a.clone(), rhs.clone(), tol));
    let f = df(&v[0]);
    let x = df(&v[1]);
    let diag = df(&v[2]);
    assert_eq!(diag[0], expected_rank as f64);
    if expected_rank < n {
        assert_eq!(diag[3], 1.);
        assert!(diag[1].is_infinite());
        assert!(x.iter().all(|x| x.is_nan()));
        return 0.;
    }
    assert_eq!(diag[3], 0.);
    assert!(diag[1] >= 1.);
    let mut maxerr = 0f64;
    for j in 0..n {
        let pivot = f[(m + n + 1) * n + j] as usize;
        let scale = f[(m + n) * n + pivot];
        for i in 0..m {
            let pred = (0..n)
                .map(|k| f[i * n + k] * f[(m + k) * n + j])
                .sum::<f64>()
                * scale;
            let error = (pred - a[i * n + pivot]).abs();
            let colscale = (0..m)
                .map(|r| a[r * n + pivot].abs())
                .fold(0f64, f64::max)
                .max(1.0);
            assert!(
                error < 1e-9 * colscale,
                "QR scaled reconstruction {error:e}"
            );
            maxerr = maxerr.max(error);
        }
    }

    let reference = reference_qr(&a, &rhs, m, n);
    let pred: Vec<_> = (0..m)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>())
        .collect();
    let res = (0..m)
        .map(|i| (pred[i] - rhs[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((res - diag[2]).abs() < 1e-8);
    for j in 0..n {
        assert!(
            (x[j] - reference[j]).abs() < 1e-8 * (1. + reference[j].abs()) * diag[1].sqrt(),
            "QR solution {} {}",
            x[j],
            reference[j]
        );
    }
    for j in 0..n {
        for k in 0..n {
            let dot = (0..m).map(|i| f[i * n + j] * f[i * n + k]).sum::<f64>();
            assert!((dot - f64::from(j == k)).abs() < 1e-10);
        }
    }
    maxerr
}
pub fn qr_answers(b: &dyn ComputeBackend) {
    let (v, _) = read(
        b,
        qr_plan(
            b,
            3,
            3,
            1,
            vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
            vec![2., -3., 0.5],
            1e-10,
        ),
    );
    assert_eq!(df(&v[1]), vec![2., -3., 0.5]);
    assert_eq!(&df(&v[0])[21..24], &[0., 1., 2.]);
    assert_eq!(df(&v[2]), vec![3., 1., 0., 0.]);

    let m = 41;
    let n = 3;
    let a: Vec<_> = (0..m)
        .flat_map(|i| {
            let t = (i as f64 - 20.) / 20.;
            [1., t, 2. * t * t - 1.]
        })
        .collect();
    let rhs: Vec<_> = (0..m)
        .map(|i| a[i * n] * 0.2 - a[i * n + 1] * 0.35 + a[i * n + 2] * 0.12)
        .collect();
    qr_check(b, a.clone(), rhs.clone(), m, n, 1e-10, 3);
    let mut deficient = a.clone();
    for i in 0..m {
        deficient[i * n + 2] = deficient[i * n] + 1e-12 * deficient[i * n + 1];
    }
    qr_check(b, deficient, rhs, m, n, 1e-9, 2);
    // Known low-order Zernike pupil (piston, x/y tilt, defocus, astigmatism),
    // clipped to x>-0.35 so the basis is no longer orthogonal.
    let mut a = vec![];
    let mut rhs = vec![];
    let coeff = [0.12, -0.03, 0.04, 0.15, -0.08, 0.02];
    for j in 0..25 {
        for i in 0..25 {
            let x = i as f64 / 12. - 1.;
            let y = j as f64 / 12. - 1.;
            if x * x + y * y <= 1. && x > -0.35 {
                let row = [
                    1.,
                    2. * x,
                    2. * y,
                    3f64.sqrt() * (2. * (x * x + y * y) - 1.),
                    6f64.sqrt() * (x * x - y * y),
                    6f64.sqrt() * 2. * x * y,
                ];
                rhs.push(row.iter().zip(coeff).map(|(a, b)| a * b).sum());
                a.extend(row);
            }
        }
    }
    let m = rhs.len();
    let p = qr_plan(b, m, 6, 1, a.clone(), rhs.clone(), 1e-10);
    let (v, _) = read(b, p);
    for (x, want) in df(&v[1]).iter().zip(coeff) {
        assert!((x - want).abs() < 1e-10);
    }
    let max = qr_check(b, a, rhs, m, 6, 1e-10, 6);
    eprintln!("QR f64 reconstruction max={max:e}");
}

pub fn energy_chain(b: &dyn ComputeBackend) {
    let mut p = plan();
    let keys = import(
        b,
        &mut p,
        [6],
        BufferData::F32(vec![4., 1., 1., 0., 9., 4.]),
    );
    let ids = import(b, &mut p, [6], BufferData::U32(vec![2, 5, 1, 7, 0, 0]));
    let values = import(
        b,
        &mut p,
        [6],
        BufferData::F32(vec![1., 2., 3., 4., 5., 6.]),
    );
    let ok = scratch(&mut p, ElementType::F32, [6]);
    let oi = scratch(&mut p, ElementType::U32, [6]);
    let ov = scratch(&mut p, ElementType::F32, [6]);
    p.operations
        .push(ResidentOp::Collectives(Request::StableSort(StableSort {
            keys,
            ids,
            values: Some(values),
            output_keys: ok,
            output_ids: oi,
            output_values: Some(ov),
        })));
    let prefix = scratch(&mut p, ElementType::DoubleFloat, [6]);
    p.operations
        .push(ResidentOp::Collectives(Request::Scan(Scan {
            values: p.view(ov).unwrap(),
            inclusive: true,
            output: prefix,
        })));
    p.outputs = vec![oi, prefix];
    let (v, _) = read(b, p);
    assert_eq!(uint(&v[0]), &[7, 1, 5, 0, 2, 0]);
    assert_eq!(df(&v[1]), vec![4., 7., 9., 15., 16., 21.]);
    assert_eq!(df(&v[1])[5], 21.);
}
pub fn conditioned_qr(b: &dyn ComputeBackend) {
    let (m, n) = (71, 3);
    let a: Vec<_> = (0..m)
        .flat_map(|i| {
            let x = (i as f64 - 35.) / 35.;
            [1e-20, x * 1e20, x * x]
        })
        .collect();
    let rhs: Vec<_> = (0..m)
        .map(|i| {
            let x = (i as f64 - 35.) / 35.;
            0.3 + 0.7 * x - 0.2 * x * x + 0.01 * (13. * x).sin()
        })
        .collect();
    qr_check(b, a, rhs, m, n, 1e-10, 3);
    let mut a = vec![];
    let mut rhs = vec![];
    for j in 0..31 {
        for i in 0..3 {
            let x = 0.7 + (i as f64 - 1.) * 1e-10;
            let y = j as f64 / 30. - 0.5;
            let row = [
                1.,
                2. * x,
                2. * y,
                3f64.sqrt() * (2. * (x * x + y * y) - 1.),
                6f64.sqrt() * (x * x - y * y),
                6f64.sqrt() * 2. * x * y,
            ];
            rhs.push(row.iter().sum());
            a.extend(row);
        }
    }
    let (v, _) = read(b, qr_plan(b, rhs.len(), 6, 1, a, rhs, 1e-8));
    assert_eq!(df(&v[2])[0], 3.);
    assert_eq!(df(&v[2])[3], 1.);
    assert!(df(&v[1]).iter().all(|x| x.is_nan()));
}

pub fn nonfinite_qr(b: &dyn ComputeBackend) {
    let (v, _) = read(
        b,
        qr_plan(
            b,
            3,
            2,
            1,
            vec![1., 0., 1., 1., 1., 2.],
            vec![1., f64::NAN, 2.],
            1e-10,
        ),
    );
    assert_eq!(df(&v[2])[3], 2.);
    assert!(df(&v[1]).iter().all(|x| x.is_nan()));
}
