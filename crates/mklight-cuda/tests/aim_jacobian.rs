//! A failed Jacobian refresh keeps the Jacobian: the CUDA twin of
//! `failed_jacobian_refresh_keeps_the_jacobian` in
//! `mklight-metal/tests/p1_aiming.rs` (the same synthetic system), bit for
//! bit against the CPU-f32 specification, for the plan of that test and for
//! its host-seeded form, whose Newton solves run in CUDA's aim kernel.

use mklight_core::*;
use mklight_gpu::resident::PlanResult;
use mklight_gpu::{ComputeBackend, CpuF32Backend, QualificationBackend, testutil};
use mklight_trace::PupilSampling;

use crate::trace::p1_support;

/// Two decentered unit spheres meeting in a tip in front of a refracting
/// surface that makes the map nonlinear: Newton walks onto the tip, where
/// the x column of a refresh succeeds one-sided and both y steps miss.
fn tip_system() -> OpticalSystem {
    let (radius, offset, gap) = (1.0f64, 0.9f64, 0.003);
    let tip = (radius * radius - offset * offset).sqrt();
    // Height on the stop plane, `depth` into glass of index `n` behind a
    // sphere of radius `r`, of a ray parallel to the axis at height `y`.
    let (r, n, depth) = (5.0f64, 1.5f64, 2.0f64);
    let on_stop = |y: f64| {
        let root = (r * r - y * y).sqrt();
        let (sag, normal) = (r - root, [y / r, -root / r]);
        let (eta, cos_i) = (1.0 / n, root / r);
        let k = (1.0 - eta * eta * (1.0 - cos_i * cos_i)).sqrt();
        let t = [
            (eta * cos_i - k) * normal[0],
            eta + (eta * cos_i - k) * normal[1],
        ];
        y + t[0] / t[1] * (depth - sag)
    };
    let placed = |mut s: Surface, x: f64, y: f64| {
        s.placement = Some(Placement {
            decenter_x: x,
            decenter_y: y,
            restore_after: true,
            ..Default::default()
        });
        s
    };
    let sphere = |id, y| {
        let shape = SurfaceShape::Conic {
            curvature: 1.0 / radius,
            conic: 0.0,
        };
        placed(Surface::new(id, shape, 1.0, MaterialRef::Air), 0.0, y)
    };
    let glass = MaterialRef::Model {
        nd: n,
        vd: 60.0,
        dpgf: 0.0,
    };
    let lens = SurfaceShape::Conic {
        curvature: 1.0 / r,
        conic: 0.0,
    };
    let stop = Surface::new(4, SurfaceShape::Plane, 10.0, glass.clone());
    let mut model = OpticalSystem::new_empty();
    model.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        sphere(1, offset),
        sphere(2, -offset),
        Surface::new(3, lens, depth, glass),
        placed(stop, on_stop(tip + gap), 0.0),
        Surface::new(5, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    model.stop = 4;
    model.aperture = ApertureSpec::EntrancePupilDiameter { value: 1.0 };
    model.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0., 0.)],
    };
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    model
}

#[test]
fn failed_jacobian_refresh_keeps_the_jacobian() {
    let gpu = crate::gpu_or_skip!();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let t = testutil::tracer(tip_system());
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 5 }).points());
    for host_seeded in [false, true] {
        let run = |b: &dyn QualificationBackend| {
            let (p, h) = p1_support::make_plan(b, &t, 0, 0, &points, false, 40, false);
            let p = if host_seeded {
                p1_support::host_seeded(b, p, &t, 0, 0)
            } else {
                p
            };
            let r: PlanResult = b
                .submit_plan(p, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let data = p1_support::read_all(b, &r);
            b.release_systems(h).unwrap();
            data
        };
        let (dg, dc) = (run(gpu), run(&cpu));
        if !host_seeded {
            // The centre ray starts at the unconverged chief and stays
            // unresolved.
            assert_ne!(p1_support::uints(&dc[1])[0], 0);
        }
        p1_support::bits(&dg, &dc);
    }
}
