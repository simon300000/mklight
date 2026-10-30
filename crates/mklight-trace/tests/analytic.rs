#![allow(clippy::needless_range_loop)]
//! Known-answer tests against closed-form optics.

mod common;

use common::*;
use mklight_core::{Interaction, MaterialRef, SurfaceShape, Vec3, samples};
use mklight_trace::{
    PupilSampling, Ray, TraceOptions, surface::intersect, trace_ray, trace_ray_recorded,
};

fn paraboloid(epd: f64) -> mklight_trace::SequentialTracer {
    let mut m = surf(
        1,
        SurfaceShape::Conic {
            curvature: -1.0 / 200.0,
            conic: -1.0,
        },
        -100.0,
        MaterialRef::Air,
    );
    m.interaction = Interaction::Reflect;
    tracer(infinite_system(epd, vec![m], 2))
}

#[test]
fn paraboloid_on_axis_is_perfect() {
    // F/1.67 paraboloid, f = 100 mm: every axial ray meets the focus and has
    // the same optical path.
    let t = paraboloid(60.0);
    let f = t.field(0);
    let opts = TraceOptions::default();
    let reference = t.opd_reference(f, 0).unwrap();
    let mut max_spot: f64 = 0.0;
    let mut max_opd: f64 = 0.0;
    for p in (PupilSampling::Hexapolar { rings: 12 }).points() {
        let r = t.trace(f, 0, p[0], p[1], &opts);
        assert!(r.status.is_ok());
        max_spot = max_spot.max(r.ray.pos.x.hypot(r.ray.pos.y));
        max_opd = max_opd.max(reference.opd_waves(&r).unwrap().abs());
    }
    assert!(max_spot < 1e-10, "spot {max_spot:e} mm");
    assert!(max_opd < 1e-9, "OPD {max_opd:e} waves");
}

#[test]
fn newtonian_sample_on_axis_is_perfect() {
    let t = tracer(samples::newtonian_telescope());
    let f = t.field(0);
    let wl = t.system().primary_wavelength;
    let reference = t.opd_reference(f, wl).unwrap();
    for p in (PupilSampling::Hexapolar { rings: 8 }).points() {
        let r = t.trace(f, wl, p[0], p[1], &TraceOptions::default());
        assert!(r.ray.pos.x.hypot(r.ray.pos.y) < 1e-10);
        assert!(reference.opd_waves(&r).unwrap().abs() < 1e-9);
    }
}

/// Fit `OPD(ρ) = a ρ⁴ + b ρ⁶ + c ρ⁸` along +Py and return `a` (waves).
fn fitted_w040(t: &mklight_trace::SequentialTracer) -> f64 {
    let f = t.field(0);
    let reference = t.opd_reference(f, 0).unwrap();
    let rho: Vec<f64> = (1..=20).map(|k| k as f64 / 20.0).collect();
    let opd: Vec<f64> = rho
        .iter()
        .map(|&r| {
            reference
                .opd_waves(&t.trace(f, 0, 0.0, r, &TraceOptions::default()))
                .unwrap()
        })
        .collect();
    poly_fit(&rho, &opd, &[4, 6, 8])[0]
}

#[test]
fn spherical_mirror_w040() {
    // Concave sphere R = −200, collimated input, stop at the mirror, image at
    // paraxial focus: W040 = h⁴ / (4 |R|³) (positive: the sphere is deeper
    // than the perfect paraboloid, so the marginal path is shorter).
    let (r, h) = (-200.0f64, 10.0f64);
    let mut m = surf(1, SurfaceShape::sphere(r), r / 2.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    let t = tracer(infinite_system(2.0 * h, vec![m], 2));
    let w040 = h.powi(4) / (4.0 * r.abs().powi(3)) / (D_LINE * 1e-3);
    let a = fitted_w040(&t);
    assert!((a - w040).abs() < 1e-6 * w040, "fit {a} vs {w040}");
}

#[test]
fn plano_convex_w040_matches_seidel() {
    // Convex side to infinity. Seidel S_I = −Σ A² y Δ(u/n), W040 = S_I / 8,
    // from an independent hand-written paraxial marginal-ray trace.
    let (r1, d, h) = (50.0f64, 5.0f64, 4.0f64);
    let body = vec![
        surf(1, SurfaceShape::sphere(r1), d, MaterialRef::glass("N-BK7")),
        surf(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    let mut sys = infinite_system(2.0 * h, body, 3);
    let n = resolve(sys.clone()).surfaces[1].n_after[0];
    let c1 = 1.0 / r1;
    // Surface 1.
    let y1 = h;
    let u0 = 0.0;
    let a1 = y1 * c1 + u0;
    let u1 = (u0 - y1 * c1 * (n - 1.0)) / n;
    let s1 = -a1 * a1 * y1 * (u1 / n - u0);
    // Surface 2 (plane).
    let y2 = y1 + d * u1;
    let a2 = n * u1;
    let u2 = n * u1;
    let s2 = -a2 * a2 * y2 * (u2 - u1 / n);
    let w040 = (s1 + s2) / 8.0 / (D_LINE * 1e-3);
    sys.surfaces[2].thickness = -y2 / u2; // paraxial focus
    let a = fitted_w040(&tracer(sys));
    assert!(w040 > 0.0);
    assert!((a - w040).abs() < 1e-6 * w040, "fit {a} vs Seidel {w040}");
}

#[test]
fn plane_parallel_plate_lateral_displacement() {
    // d = t sinθ (1 − cosθ / √(n² − sin²θ)), measured perpendicular to the ray.
    let (thick, n) = (10.0, 1.6);
    let body = vec![
        surf(1, SurfaceShape::Plane, thick, MaterialRef::Constant { n }),
        surf(2, SurfaceShape::Plane, 5.0, MaterialRef::Air),
    ];
    let rs = resolve(infinite_system(10.0, body, 3));
    for deg in [1.0f64, 10.0, 30.0, 60.0] {
        let th = deg.to_radians();
        let start = Ray::new(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, th.sin(), th.cos()),
        );
        let r = trace_ray(&rs, &start, 0, &TraceOptions::default(), None);
        assert!(r.status.is_ok());
        // Image plane is at global z = thick + 5 (surface 1 at z = 0).
        let y_straight = (thick + 5.0 + 5.0) * th.tan();
        let lateral = (y_straight - r.ray.pos.y) * th.cos();
        let s = th.sin();
        let expect = thick * s * (1.0 - th.cos() / (n * n - s * s).sqrt());
        assert!(
            (lateral - expect).abs() < 1e-12,
            "{deg}°: {lateral} vs {expect}"
        );
        assert!(r.ray.dir.max_abs_diff(start.dir) < 1e-15);
    }
}

#[test]
fn thick_lens_efl_matches_lensmaker() {
    for (r1, r2, d) in [(50.0, -80.0, 6.0), (-30.0, -15.0, 4.0), (100.0, 1e9, 2.0)] {
        let body = vec![
            surf(
                1,
                SurfaceShape::sphere(r1),
                d,
                MaterialRef::Constant { n: 1.7 },
            ),
            surf(2, SurfaceShape::sphere(r2), 50.0, MaterialRef::Air),
        ];
        let t = tracer(infinite_system(2.0, body, 3));
        let n = 1.7;
        let phi = (n - 1.0) * (1.0 / r1 - 1.0 / r2 + (n - 1.0) * d / (n * r1 * r2));
        let fo = t.paraxial().primary();
        assert!((fo.efl * phi - 1.0).abs() < 1e-13, "{r1} {r2}: {}", fo.efl);
        // Back focal distance: f (1 − (n−1) d / (n R1)).
        let bfl = (1.0 - (n - 1.0) * d / (n * r1)) / phi;
        assert!((fo.bfl - bfl).abs() < 1e-11 * bfl.abs());
        // A real ray at a tiny height converges to the paraxial focal length.
        let h = 1e-5;
        let start = Ray::new(Vec3::new(0.0, h, -1.0), Vec3::Z);
        let (res, rec) = trace_ray_recorded(t.system(), &start, 0, &TraceOptions::default());
        assert!(res.status.is_ok());
        let out = rec[2].dir;
        let f_real = -h / (out.y / out.z);
        assert!(
            (f_real - fo.efl).abs() < 1e-7 * fo.efl.abs(),
            "{f_real} vs {}",
            fo.efl
        );
    }
    // Thin lens limit: 1/f = (n − 1)(1/R1 − 1/R2).
    let body = vec![
        surf(
            1,
            SurfaceShape::sphere(40.0),
            0.0,
            MaterialRef::Constant { n: 1.5 },
        ),
        surf(2, SurfaceShape::sphere(-60.0), 50.0, MaterialRef::Air),
    ];
    let t = tracer(infinite_system(2.0, body, 3));
    let f = 1.0 / (0.5 * (1.0 / 40.0 + 1.0 / 60.0));
    assert!((t.paraxial().primary().efl - f).abs() < 1e-12 * f);
}

#[test]
fn snell_invariant_conserved_at_every_surface() {
    // n₁ |k × N| = n₂ |k' × N| and k, k', N coplanar, for skew rays through
    // the Double Gauss (axial frames, so directions carry over unchanged).
    let t = tracer(samples::double_gauss());
    let sys = t.system();
    for w in 0..t.num_wavelengths() {
        for p in [[0.7, 0.5], [-0.3, 0.9], [0.0, -1.0]] {
            let mut rec = Vec::new();
            let r = t.trace_with(
                t.field(2),
                w,
                p[0],
                p[1],
                &TraceOptions::default(),
                Some(&mut rec),
            );
            assert!(r.status.is_ok());
            for i in 1..sys.image_index() {
                let k = rec[i - 1].dir;
                let kp = rec[i].dir;
                let nrm = rec[i].normal;
                let n1 = sys.n_before(i, w);
                let n2 = sys.surfaces[i].n_after[w];
                let lhs = k.cross(nrm) * n1;
                let rhs = kp.cross(nrm) * n2;
                assert!(
                    lhs.max_abs_diff(rhs) < 2e-15,
                    "surface {i}: {lhs:?} {rhs:?}"
                );
                // Directions are not renormalised, so |k| drifts by about
                // one ulp per surface (10 ulp after nine surfaces is normal).
                assert!(
                    (kp.norm() - 1.0).abs() < 1e-14,
                    "surface {i} w {w} p {p:?}: |k|-1 = {:e}",
                    kp.norm() - 1.0
                );
            }
        }
    }
}

#[test]
fn paraxial_agrees_with_real_rays_at_tiny_aperture() {
    // Cooke triplet: a marginal ray of height ε and a chief ray of angle ε
    // reproduce the paraxial heights/slopes divided by their scale.
    let t = tracer(samples::cooke_triplet());
    let sys = t.system();
    let w = sys.primary_wavelength;
    let fo = t.paraxial().primary();
    let eps = 1e-7;
    // Marginal.
    let scale = eps / fo.marginal.y[1];
    let start = Ray::new(Vec3::new(0.0, eps, 0.0), Vec3::Z);
    let (r, rec) = trace_ray_recorded(sys, &start, w, &TraceOptions::default());
    assert!(r.status.is_ok());
    for j in 1..sys.num_surfaces() {
        let yp = fo.marginal.y[j] * scale;
        assert!(
            (rec[j].point.y - yp).abs() < 1e-6 * eps,
            "marginal y at {j}"
        );
        let up = fo.marginal.u[j.min(sys.num_surfaces() - 2)] * scale;
        if j < sys.num_surfaces() - 1 {
            assert!((rec[j].dir.y / rec[j].dir.z - up).abs() < 1e-6 * eps);
        }
    }
    // Chief: slope ε through the entrance-pupil centre.
    let z_ep = fo.entrance_pupil_position;
    let dir = Vec3::new(0.0, eps, 1.0).normalized();
    let ep = Vec3::new(0.0, 0.0, z_ep);
    let start = Ray::new(ep - dir * ep.dot(dir), dir);
    let (_, rec) = trace_ray_recorded(sys, &start, w, &TraceOptions::default());
    let cscale = eps / fo.chief.u[0];
    for j in 1..sys.num_surfaces() {
        let yp = fo.chief.y[j] * cscale;
        assert!(
            (rec[j].point.y - yp).abs() < 1e-6 * eps * 50.0,
            "chief y at {j}"
        );
    }
    assert!(rec[sys.stop].point.y.abs() < 1e-12);
}

#[test]
fn ideal_lens_is_perfect_for_collimated_beams() {
    let mut body = vec![
        surf(1, SurfaceShape::Plane, 10.0, MaterialRef::Air),
        surf(
            2,
            SurfaceShape::Paraxial { focal_length: 80.0 },
            80.0,
            MaterialRef::Air,
        ),
    ];
    body[0].label = "stop".into();
    let mut sys = infinite_system(20.0, body, 3);
    sys.fields = angle_fields(&[0.0, 7.0]);
    let t = tracer(sys);
    for (fi, f) in t.fields().iter().enumerate() {
        let reference = t.opd_reference(f, 0).unwrap();
        let chief = t.chief(f, 0);
        let expect_y = 80.0 * (fi as f64 * 7.0f64).to_radians().tan();
        assert!((chief.ray.pos.y - expect_y).abs() < 1e-12);
        for p in (PupilSampling::Hexapolar { rings: 6 }).points() {
            let r = t.trace(f, 0, p[0], p[1], &TraceOptions::default());
            assert!((r.ray.pos - chief.ray.pos).norm() < 1e-12);
            assert!(reference.opd_waves(&r).unwrap().abs() < 1e-8);
        }
    }
}

/// Finite-conjugate system of ideal lenses: object `t0` before surface 1,
/// `(focal length, material after, thickness after)` per lens, the stop on
/// surface `stop`, object-height fields; the last thickness is replaced by the
/// paraxial image distance.
fn ideal_relay(
    t0: f64,
    lenses: &[(Option<f64>, MaterialRef, f64)],
    stop: usize,
    na: f64,
    heights: &[f64],
) -> mklight_trace::SequentialTracer {
    let build = |last: Option<f64>| {
        let mut s = mklight_core::OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = mklight_core::ApertureSpec::ObjectSpaceNa { value: na };
        s.fields = mklight_core::FieldSet {
            kind: mklight_core::FieldKind::ObjectHeight,
            fields: heights
                .iter()
                .map(|&y| mklight_core::Field::new(0.0, y))
                .collect(),
        };
        let mut surfaces = vec![surf(0, SurfaceShape::Plane, t0, MaterialRef::Air)];
        for (j, (f, m, t)) in lenses.iter().enumerate() {
            let shape = match f {
                Some(f) => SurfaceShape::Paraxial { focal_length: *f },
                None => SurfaceShape::Plane,
            };
            let t = if j + 1 == lenses.len() {
                last.unwrap_or(*t)
            } else {
                *t
            };
            surfaces.push(surf(j as u64 + 1, shape, t, m.clone()));
        }
        let m_img = lenses.last().unwrap().1.clone();
        surfaces.push(surf(99, SurfaceShape::Plane, 0.0, m_img));
        s.surfaces = surfaces;
        s.stop = stop;
        s
    };
    let t = tracer(build(None));
    let b = t.paraxial().primary().image_distance;
    tracer(build(Some(b)))
}

/// Largest |OPD| (waves) and spot radius (mm) over a hexapolar pupil.
fn worst_opd_and_spot(t: &mklight_trace::SequentialTracer) -> (f64, f64) {
    let (mut opd, mut spot) = (0.0f64, 0.0f64);
    for f in t.fields() {
        let reference = t.opd_reference(f, 0).unwrap();
        let chief = t.chief(f, 0);
        for p in (PupilSampling::Hexapolar { rings: 8 }).points() {
            let r = t.trace(f, 0, p[0], p[1], &TraceOptions::default());
            assert!(r.status.is_ok());
            spot = spot.max((r.ray.pos - chief.ray.pos).norm());
            opd = opd.max(reference.opd_waves(&r).unwrap().abs());
        }
    }
    (opd, spot)
}

#[test]
fn ideal_lens_is_perfect_at_finite_conjugates() {
    let air = || MaterialRef::Air;
    let cases: Vec<(&str, mklight_trace::SequentialTracer)> = vec![
        // 2f–2f at NA 0.2 (previously off by ~150 waves with a plane-wave
        // phase law).
        (
            "2f-2f",
            ideal_relay(200.0, &[(Some(100.0), air(), 200.0)], 1, 0.2, &[0.0, 10.0]),
        ),
        // Object-space telecentric: stop in the back focal plane.
        (
            "telecentric",
            ideal_relay(
                100.0,
                &[(Some(50.0), air(), 50.0), (None, air(), 50.0)],
                2,
                0.05,
                &[0.0, 5.0],
            ),
        ),
        // Second lens sees a virtual (converging) object.
        (
            "virtual object",
            ideal_relay(
                75.0,
                &[(Some(50.0), air(), 100.0), (Some(100.0), air(), 30.0)],
                1,
                0.15,
                &[0.0, 4.0],
            ),
        ),
        // Image space in glass (n₁ ≠ n₂).
        (
            "glass image space",
            ideal_relay(
                150.0,
                &[(Some(80.0), MaterialRef::glass("N-BK7"), 100.0)],
                1,
                0.25,
                &[0.0, 6.0],
            ),
        ),
    ];
    for (name, t) in &cases {
        let (opd, spot) = worst_opd_and_spot(t);
        assert!(spot < 1e-11, "{name}: spot {spot:e} mm");
        assert!(opd < 1e-9, "{name}: OPD {opd:e} waves");
    }
}

#[test]
fn ideal_lens_carries_incoming_aberration_unchanged() {
    // A plano-convex singlet followed by an ideal lens placed in its
    // converging beam (virtual object): perfect imaging preserves the wave
    // aberration of every ray, so the fitted W040 (same entrance-pupil
    // coordinates) must equal the singlet's own. This holds only because the
    // ideal lens's path term is stationary in the incoming direction.
    let singlet = |extra: bool| {
        let mut body = vec![
            surf(
                1,
                SurfaceShape::sphere(50.0),
                4.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(2, SurfaceShape::Plane, 60.0, MaterialRef::Air),
        ];
        if extra {
            body.push(surf(
                3,
                SurfaceShape::Paraxial { focal_length: 40.0 },
                60.0,
                MaterialRef::Air,
            ));
        }
        let sys = infinite_system(8.0, body, 9);
        let t = tracer(sys.clone());
        let mut sys = sys;
        let n = sys.surfaces.len();
        sys.surfaces[n - 2].thickness = t.paraxial().primary().image_distance;
        tracer(sys)
    };
    let w_single = fitted_w040(&singlet(false));
    let w_relay = fitted_w040(&singlet(true));
    assert!(w_single.abs() > 0.05, "{w_single}");
    assert!(
        (w_relay - w_single).abs() < 1e-6 * w_single.abs(),
        "{w_single} vs {w_relay}"
    );
}

#[test]
fn intersection_matches_sag_definition_on_aspheres_near_the_edge() {
    // Strong asphere, rays up to 99.5 % of the conic domain: Newton converges
    // and the hit satisfies the core sag definition.
    let shape = SurfaceShape::EvenAsphere {
        curvature: 1.0 / 20.0,
        conic: 0.0,
        coefficients: vec![-4e-5, 2e-8],
    };
    let settings = mklight_trace::IntersectSettings::default();
    for k in 0..=40 {
        let y = 19.9 * k as f64 / 40.0;
        let d = Vec3::new(0.0, -0.05, 1.0).normalized();
        let p = Vec3::new(0.0, y + 0.05 * 10.0 / d.z * d.z, -10.0);
        let h = intersect(&shape, p, d, &settings).expect("converges");
        let r2 = h.point.x.powi(2) + h.point.y.powi(2);
        assert!(
            (h.point.z - shape.sag(r2).unwrap()).abs() < 1e-12,
            "y = {y}"
        );
    }
}
