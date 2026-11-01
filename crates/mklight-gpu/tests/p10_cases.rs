#![allow(dead_code)]
use bytemuck::Zeroable;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, stray_ops::*, *};
pub fn snapshot() -> Snapshot {
    Snapshot::new(testutil::singlet()).unwrap()
}
pub fn source(kind: u32) -> Source {
    let z = Df32::ZERO;
    let one = Df32::from_f32(1.);
    Source {
        kind,
        wavelength_count: 3,
        center: [z; 3],
        axis: [z, z, one],
        tangent: [[one, z, z], [z, one, z]],
        radius: Df32::from_f32(2.),
        inner_radius: Df32::from_f32(0.5),
        cone: Df32::from_f64(0.3),
        distance: Df32::from_f32(5.),
    }
}
pub fn plan(b: &dyn ComputeBackend, kind: u32, n: usize, tile: usize) -> ResidentPlan {
    let mut p = ResidentPlan::new(snapshot());
    let rays = p.scratch(record_desc::<Ray>(n));
    p.operations
        .push(ResidentOp::Stray(Request::Generate(Generate {
            source: source(kind),
            cdf: vec![Df32::from_f64(0.2), Df32::from_f64(0.6), Df32::from_f32(1.)],
            seed: 9127,
            first_ray: u32::MAX as u64 - 7,
            ray_count: n,
            output: rays,
            tile,
        })));
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.].map(Df32::from_f32);
    let prims = [Primitive {
        kind: 1,
        mask: 0,
        origin: [Df32::ZERO; 3],
        rotation: identity,
        lower: [-3., -3., 0.].map(Df32::from_f32),
        upper: [3., 3., 0.46060799].map(Df32::from_f32),
        curvature: Df32::from_f64(0.1),
        conic: Df32::ZERO,
        inner_radius: Df32::ZERO,
        outer_radius: Df32::from_f32(3.),
        aperture: [Df32::ZERO; 2],
    }];
    let nodes = [Node {
        lower: prims[0].lower,
        upper: prims[0].upper,
        escape: 1,
        primitive: 0,
    }];
    let geometry = Geometry::new(b, &prims, &nodes, &JobContext::new()).unwrap();
    let view = p.view(rays).unwrap();
    let hits = geometry.append(&mut p, view, tile);
    p.outputs = vec![rays, hits];
    p
}
pub fn run(b: &dyn ComputeBackend, p: ResidentPlan) -> (Vec<Vec<u32>>, PlanResult) {
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.stats.copied_bytes, 0);
    assert_eq!(r.stats.fallback_count, 0);
    let words = r
        .outputs
        .iter()
        .map(|h| b.read_view(h.view()).unwrap().words().to_vec())
        .collect();
    (words, r)
}
pub fn answers(b: &dyn ComputeBackend) {
    let (w, _) = run(b, plan(b, 0, 1024, 257));
    let rays: &[Ray] = bytemuck::cast_slice(&w[0]);
    let hits: &[Hit] = bytemuck::cast_slice(&w[1]);
    for (i, (r, h)) in rays.iter().zip(hits).enumerate() {
        let id = u32::MAX as u64 - 7 + i as u64;
        assert_eq!(r.id, [id as u32, (id >> 32) as u32]);
        assert_eq!(r.draws, 3);
        assert_eq!(h.primitive, 0);
        let [x, y, z] = h.point.map(Df32::to_f64);
        let zz = (x * x + y * y) / (10. + (100. - x * x - y * y).sqrt());
        assert!((z - zz).abs() < 2e-10, "sag {z} vs {zz}");
        assert!((h.distance.to_f64() - (z + 5.)).abs() < 2e-10);
        let norm = h.normal.map(Df32::to_f64);
        assert!((norm.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 2e-12);
        assert_eq!(r.wavelength, {
            let mut rng = mklight_core::rng::Philox::new(9127, id);
            let u = rng.uniform();
            if u < 0.2 {
                0
            } else if u < 0.6 {
                1
            } else {
                2
            }
        });
    }
}
pub fn close_hits(b: &dyn ComputeBackend) {
    let mut p = ResidentPlan::new(snapshot());
    let mut primitive = Primitive::zeroed();
    primitive.rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.].map(Df32::from_f32);
    primitive.lower = [-2., -2., 0.].map(Df32::from_f32);
    primitive.upper = [2., 2., 0.].map(Df32::from_f32);
    primitive.outer_radius = Df32::from_f32(2.);
    let mut prims = vec![primitive; 3];
    prims[0].origin[2] = Df32::from_f64(100. + 1e-8);
    prims[1].origin[2] = Df32::from_f64(100.);
    prims[2] = prims[1];
    let mut nodes = vec![];
    // Reverse leaf order exercises the primitive-ID tie rule.
    for k in [2, 0, 1] {
        let z = prims[k].origin[2];
        let i = nodes.len();
        nodes.push(Node {
            lower: [Df32::from_f32(-2.), Df32::from_f32(-2.), z],
            upper: [Df32::from_f32(2.), Df32::from_f32(2.), z],
            escape: i as u32 + 1,
            primitive: k as u32,
        });
    }
    let geometry = Geometry::new(b, &prims, &nodes, &JobContext::new()).unwrap();
    let h = fill_records(b, 4, &JobContext::new(), |i| Ray {
        origin: [
            Df32::from_f32(if i == 3 { 3. } else { 0. }),
            Df32::ZERO,
            Df32::from_f64(if i == 1 { 100. + 1e-9 } else { 0. }),
        ],
        direction: [
            Df32::ZERO,
            Df32::ZERO,
            Df32::from_f32(if i == 2 { -1. } else { 1. }),
        ],
        id: [i as u32, 0],
        draws: 0,
        wavelength: 0,
    })
    .unwrap();
    let input = p.import(h.view());
    let v = p.view(input).unwrap();
    let out = geometry.append(&mut p, v, 4);
    p.outputs = vec![out];
    let (w, _) = run(b, p);
    let hits: &[Hit] = bytemuck::cast_slice(&w[0]);
    assert_eq!(
        hits.iter().map(|h| h.primitive).collect::<Vec<_>>(),
        vec![1, 0, u32::MAX, u32::MAX]
    );
    assert!((hits[1].distance.to_f64() - 9e-9).abs() < 2e-12);
}
#[test]
fn cpu_known_answers() {
    let b = CpuF32Backend::new();
    answers(&b);
    close_hits(&b);
}

pub fn split_child_estimator(b: &dyn ComputeBackend) {
    use mklight_gpu::collectives::{self, SegmentKind, SegmentReduce};
    let ctx = JobContext::new();
    let n = 3;
    let cap = 4;
    let mut covered = History::zeroed();
    covered.length = 2;
    covered.events[..2].copy_from_slice(&[2, 1]);
    let coverage = fill_records(b, 1, &ctx, |_| covered).unwrap();
    let mut a = History::zeroed();
    a.length = 2;
    a.events[..2].copy_from_slice(&[(1 << 28) | 2, (1 << 28) | 1]);
    let mut c = a;
    c.length = 4;
    c.events[..4].copy_from_slice(&[(1 << 28) | 2, (1 << 28) | 1, (1 << 28) | 2, (1 << 28) | 1]);
    let hits = fill_records(b, n * cap, &ctx, |i| {
        let (source, child) = (i / cap, i % cap);
        let weights = [[1., 1., 1., 1.], [4., 0.5, 0.5, 4.], [2., 0., 0., 0.]];
        Detection {
            history: if child == 0 {
                a
            } else if child < 3 {
                c
            } else {
                History::zeroed()
            },
            weight: Df32::from_f32(weights[source][child]),
            x: Df32::from_f64(0.13 * (child as f64 - 1.5)),
            y: Df32::from_f64(0.07 * source as f64),
            class: if child < 3 { 1 } else { 4 },
            wavelength: 0,
        }
    })
    .unwrap();
    let ledger = fill_records(b, n, &ctx, |_| Ledger {
        hits: cap as u32,
        ..Ledger::zeroed()
    })
    .unwrap();
    let params = TallyParams {
        half: [Df32::from_f32(0.5); 2],
        nx: 1,
        ny: 1,
        owned: 28,
        covered: 1,
        count: (n * cap) as u32,
        capacity: cap as u32,
        sources: n as u32,
        queries: 4,
    };
    let queries = fill_records(b, 4, &ctx, |i| Query {
        kind: if i == 3 { 1 } else { 0 },
        slot: [5, 6, 8, 5][i],
        ..Default::default()
    })
    .unwrap();
    let mut p = ResidentPlan::new(snapshot());
    let h = p.import(hits.view());
    let l = p.import(ledger.view());
    let cv = p.import(coverage.view());
    let q = p.import(queries.view());
    let tags = p.scratch(BufferDesc::new(ElementType::U32, [n * cap, 4]));
    p.operations
        .push(ResidentOp::Stray(Request::Tally(Tally::Tags {
            hits: p.view(h).unwrap(),
            ledger: p.view(l).unwrap(),
            coverage: p.view(cv).unwrap(),
            params,
            output: tags,
        })));
    let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, 4 * n]));
    p.operations
        .push(ResidentOp::Stray(Request::Tally(Tally::Terms {
            products: false,
            hits: p.view(h).unwrap(),
            ledger: p.view(l).unwrap(),
            tags: p.view(tags).unwrap(),
            queries: p.view(q).unwrap(),
            params,
            output: terms,
        })));
    let offsets = b
        .fill_buffer(
            BufferDesc::new(ElementType::U32, [4 * n + 1]),
            &mut |mut out| {
                for (i, v) in out.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                    *v = i as u32;
                }
                Ok(())
            },
        )
        .unwrap();
    let o = p.import(offsets.view());
    let products = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4 * n, 14]));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [4 * n]));
    p.operations.push(ResidentOp::Collectives(
        collectives::Request::SegmentReduce(SegmentReduce {
            values: p.view(terms).unwrap(),
            offsets: p.view(o).unwrap(),
            mask: None,
            mask_value: 0,
            weights: None,
            max_segment_len: 1,
            reference: None,
            mode: ReduceMode::DoubleFloat,
            kind: SegmentKind::SourceProducts,
            output: products,
            status,
        }),
    ));
    let fused = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [14 * 4, n]));
    p.operations
        .push(ResidentOp::Stray(Request::Tally(Tally::Terms {
            products: true,
            hits: p.view(h).unwrap(),
            ledger: p.view(l).unwrap(),
            tags: p.view(tags).unwrap(),
            queries: p.view(q).unwrap(),
            params,
            output: fused,
        })));
    p.outputs = vec![products, tags, fused];
    let result = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    let view = b.read_view(result.outputs[0].view()).unwrap();
    let v = view.as_slice::<Df32>().unwrap();
    let fv = b.read_view(result.outputs[2].view()).unwrap();
    let fused = fv.as_slice::<Df32>().unwrap();
    for query in 0..4 {
        for source in 0..n {
            for c in 0..14 {
                assert_eq!(
                    v[(query * n + source) * 14 + c],
                    fused[(query * 14 + c) * n + source],
                    "fused P3 source product"
                );
            }
        }
    }
    for (query, answers) in [[3., 5., 0.], [2., 1., 0.], [1., 4., 2.], [3., 5., 0.]]
        .iter()
        .enumerate()
    {
        for (source, w) in answers.iter().enumerate() {
            let row = &v[(query * n + source) * 14..];
            assert_eq!(row[0].to_f64(), *w);
            assert_eq!(row[4].to_f64(), w * w);
        }
    }
    let flux = v[0].to_f64() + v[14].to_f64() + v[28].to_f64();
    let sq = v[4].to_f64() + v[18].to_f64() + v[32].to_f64();
    assert_eq!(flux, 8.);
    assert!((1.5 * (sq - flux * flux / 3.) - 19.).abs() < 1e-12);
}
#[test]
fn cpu_split_child_estimator() {
    split_child_estimator(&CpuF32Backend::new());
}
