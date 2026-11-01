mod common;
use mklight_core::{
    JobContext, Snapshot,
    math::{Frame, Mat3, Vec3},
    model::{ApertureShape, SurfaceShape},
    rng::Philox,
};
use mklight_gpu::{ComputeBackend, CpuF32Backend, Df32, resident::*, stray_ops as ops};
use mklight_stray::{
    SourceSpec, StraySystem, SystemOptions,
    geom::{BruteForce, Cylinder, Mask, Primitive, Ray, RayCaster, SagPatch, Shape},
    gpu,
};
use std::sync::Arc;
fn v(a: [Df32; 3]) -> Vec3 {
    Vec3::new(a[0].to_f64(), a[1].to_f64(), a[2].to_f64())
}
fn snapshot() -> Snapshot {
    Snapshot::new(mklight_core::samples::singlet()).unwrap()
}

pub fn sources(b: &dyn ComputeBackend) {
    let axis = Vec3::new(0.1, 0.2, 1.);
    let center = Vec3::new(300., -100., -40.);
    let cases = [
        SourceSpec::Collimated {
            direction: axis,
            center,
            radius: 12.,
            start_distance: 1000.,
            power: 1.,
        },
        SourceSpec::CollimatedThrough {
            direction: axis,
            center,
            normal: Vec3::Z,
            radius: 12.,
            start_distance: 1000.,
            power: 1.,
        },
        SourceSpec::Point {
            position: center,
            axis,
            half_angle_deg: 0.001,
            intensity: 1.,
        },
        SourceSpec::Point {
            position: center,
            axis,
            half_angle_deg: 179.,
            intensity: 1.,
        },
        SourceSpec::LambertianDisk {
            center,
            normal: axis,
            radius: 12.,
            inner_radius: 4.,
            radiance: 1.,
            max_angle_deg: 85.,
        },
    ];
    let mut max_p = 0f64;
    let mut max_d = 0f64;
    for source in &cases {
        let mut p = ResidentPlan::new(snapshot());
        let out = gpu::generate(
            &mut p,
            source,
            &[1., 2., 1.],
            17,
            u32::MAX as u64 - 10,
            1001,
            127,
        )
        .unwrap();
        p.outputs = vec![out];
        let result = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let view = b.read_view(result.outputs[0].view()).unwrap();
        for (i, r) in ops::read_records::<ops::Ray>(&view)
            .unwrap()
            .iter()
            .enumerate()
        {
            let id = u32::MAX as u64 - 10 + i as u64;
            let mut rng = Philox::new(17, id);
            let wl = rng.uniform();
            assert_eq!(
                r.wavelength,
                if wl < 0.25 {
                    0
                } else if wl < 0.75 {
                    1
                } else {
                    2
                }
            );
            let reference = source.sample(&mut rng);
            max_p = max_p.max(reference.origin.max_abs_diff(v(r.origin)));
            max_d = max_d.max(reference.dir.max_abs_diff(v(r.direction)));
            assert!(max_p < 2e-9, "source error {max_p}");
            assert!(max_d < 2e-11, "source direction {max_d}");
            assert_eq!(
                r.draws,
                if matches!(source, SourceSpec::LambertianDisk { .. }) {
                    5
                } else {
                    3
                }
            );
        }
    }
    eprintln!("P10 source f64 maximum position={max_p:e} direction={max_d:e}");
}
pub fn geometry(b: &dyn ComputeBackend) {
    let mut rng = Philox::new(87, 3);
    let mut prims = vec![];
    for i in 0..60 {
        let frame = Frame {
            origin: Vec3::new(
                25. * (rng.uniform() - 0.5),
                25. * (rng.uniform() - 0.5),
                25. * (rng.uniform() - 0.5),
            ),
            rot: Mat3::rot_x(rng.uniform() * 3.).mul_mat(&Mat3::rot_y(rng.uniform() * 3.)),
        };
        let shape = if i % 4 == 0 {
            Shape::Cylinder(Cylinder::new(2. + rng.uniform() * 2., -4., 4.).unwrap())
        } else {
            Shape::Sag(
                SagPatch::new(
                    if i % 4 == 1 {
                        SurfaceShape::Plane
                    } else {
                        SurfaceShape::sphere(if i % 2 == 0 { 12. } else { -15. })
                    },
                    0.25,
                    4.,
                    match i % 5 {
                        0 => Some(Mask::Inside(ApertureShape::Elliptical {
                            semi_x: 3.9,
                            semi_y: 2.8,
                        })),
                        1 => Some(Mask::Outside(ApertureShape::Circular { radius: 1. })),
                        2 => Some(Mask::InsideRect {
                            half_x: 2.7,
                            half_y: 3.5,
                        }),
                        3 => Some(Mask::Inside(ApertureShape::Annular {
                            inner_radius: 1.5,
                            outer_radius: 3.8,
                        })),
                        _ => None,
                    },
                )
                .unwrap(),
            )
        };
        prims.push(Primitive::new(frame, shape));
    }
    let rays: Vec<_> = (0..8192)
        .map(|_| {
            Ray::new(
                Vec3::new(
                    40. * (rng.uniform() - 0.5),
                    40. * (rng.uniform() - 0.5),
                    40. * (rng.uniform() - 0.5),
                ),
                Vec3::new(
                    rng.uniform() - 0.5,
                    rng.uniform() - 0.5,
                    rng.uniform() - 0.5,
                ),
            )
        })
        .collect();
    compare(b, &prims, &rays);
}
pub fn compare(b: &dyn ComputeBackend, prims: &[Primitive], rays: &[Ray]) {
    let geometry = gpu::prepare_geometry(b, prims, &JobContext::new()).unwrap();
    let upload = ops::fill_records(b, rays.len(), &JobContext::new(), |i| ops::Ray {
        origin: [rays[i].origin.x, rays[i].origin.y, rays[i].origin.z].map(Df32::from_f64),
        direction: [rays[i].dir.x, rays[i].dir.y, rays[i].dir.z].map(Df32::from_f64),
        id: [i as u32, 0],
        draws: 0,
        wavelength: 0,
    })
    .unwrap();
    let mut p = ResidentPlan::new(snapshot());
    let id = p.import(upload.view());
    let input = p.view(id).unwrap();
    let out = geometry.append(&mut p, input, 4096);
    p.outputs = vec![out];
    let result = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let view = b.read_view(result.outputs[0].view()).unwrap();
    let mut max_t = 0f64;
    let mut max_p = 0f64;
    let mut max_n = 0f64;
    let mut count = 0;
    for (r, h) in rays
        .iter()
        .zip(ops::read_records::<ops::Hit>(&view).unwrap())
    {
        let reference = BruteForce(prims).closest_hit(r, f64::INFINITY);
        assert_eq!(
            reference.map_or(u32::MAX, |h| h.prim),
            h.primitive,
            "ray={r:?}"
        );
        if let Some(ref_h) = reference {
            count += 1;
            max_t = max_t.max((ref_h.local.t - h.distance.to_f64()).abs());
            max_p = max_p.max(ref_h.local.p.max_abs_diff(v(h.point)));
            max_n = max_n.max(ref_h.local.n.max_abs_diff(v(h.normal)));
        }
    }
    assert!(
        max_p < 2e-7 && max_t < 2e-7 && max_n < 2e-8,
        "{max_p} {max_t} {max_n}"
    );
    eprintln!(
        "P10 BVH rays={} hits={count} max distance={max_t:e} point={max_p:e} normal={max_n:e}",
        rays.len()
    );
}
pub fn samples(b: &dyn ComputeBackend) {
    let mut supported = 0;
    let mut unsupported = 0;
    for (name, s) in mklight_core::samples::all() {
        let sys =
            StraySystem::new(Arc::new(common::resolve(s)), &SystemOptions::default()).unwrap();
        match gpu::prepare_geometry(b, sys.scene.primitives(), &JobContext::new()) {
            Ok(_) => {
                supported += 1;
            }
            Err(mklight_core::MkError::Unsupported(reason)) => {
                unsupported += 1;
                eprintln!("P10 sample={name} unsupported={reason}");
                continue;
            }
            Err(e) => panic!("{e}"),
        }
        for f in 0..sys.resolved().snapshot.system().fields.fields.len() {
            let source = sys.system_field_source(f, 0.1, 1.).unwrap();
            let mut rays = vec![];
            for i in 0..1000 {
                let mut rng = Philox::new(51, i);
                rays.push(source.spec.sample(&mut rng));
            }
            compare(b, sys.scene.primitives(), &rays);
        }
    }
    eprintln!("P10 sample geometry supported={supported} unsupported={unsupported}");
    assert_eq!(supported + unsupported, 16);
    assert!(supported >= 5);
}
#[test]
fn cpu_sources() {
    sources(&CpuF32Backend::new());
}
#[test]
fn cpu_geometry() {
    geometry(&CpuF32Backend::new());
}
#[test]
fn cpu_samples() {
    samples(&CpuF32Backend::new());
}
