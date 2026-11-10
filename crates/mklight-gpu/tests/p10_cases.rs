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
    assert_eq!((rays.len(), hits.len()), (1024, 1024));
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
