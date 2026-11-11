//! Relative illumination, vignetting and footprints against closed forms.

use crate::common;

use common::*;
use mklight_analysis::illumination::*;
use mklight_analysis::util::gauss_legendre_01;
use mklight_core::{MaterialRef, RayAiming, SemiDiameter, SurfaceShape, Vec3, samples};
use mklight_trace::{
    FieldPoint, ObjectSource, PreparedField, SequentialTracer, TraceOptions, TracerSettings,
};

/// Projected solid angle of a disk of radius `a` seen from a point at axial
/// distance `f` and lateral offset `h`, from a surface element parallel to
/// the disk: `π/2 · (1 − (f² + h² − a²)/√((f² + h² + a²)² − 4a²h²))`.
fn disk_psa(a: f64, f: f64, h: f64) -> f64 {
    let s = f * f + h * h;
    0.5 * std::f64::consts::PI
        * (1.0 - (s - a * a) / ((s + a * a).powi(2) - 4.0 * a * a * h * h).sqrt())
}

#[test]
fn ideal_lens_relative_illumination_is_the_disk_view_factor() {
    let (f, epd, th) = (100.0, 10.0, 30.0f64);
    let t = tracer(ideal_lens(f, epd, &[0.0, th]));
    let r = relative_illumination(
        &t,
        &RelativeIlluminationSettings {
            field_samples: 4,
            ..Default::default()
        },
    )
    .unwrap();
    for (i, &h) in r.normalized_field.iter().enumerate() {
        let a = (th * h).to_radians();
        let exact = disk_psa(0.5 * epd, f, f * a.tan());
        let ri = exact / disk_psa(0.5 * epd, f, 0.0);
        assert!(
            (r.projected_solid_angle[i] - exact).abs() < 1e-8 * exact,
            "{} {}",
            r.projected_solid_angle[i],
            exact
        );
        assert!((r.relative_illumination[i] - ri).abs() < 1e-8);
        // Small aperture: the cos⁴ law.
        assert!((r.relative_illumination[i] - a.cos().powi(4)).abs() < 2e-3);
        assert!((r.cos4_object[i] - a.cos().powi(4)).abs() < 1e-12);
        assert!((r.chief_ray_angle_deg[i] - th * h).abs() < 1e-9);
        assert_eq!(r.vignetting_fraction[i], 0.0);
    }
}

/// Area of the intersection of two disks (radii `r1`, `r2`, centres `d` apart).
fn lens_area(r1: f64, r2: f64, d: f64) -> f64 {
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return std::f64::consts::PI * r1.min(r2).powi(2);
    }
    let a1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos();
    let a2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos();
    r1 * r1 * (a1 - a1.sin() * a1.cos()) + r2 * r2 * (a2 - a2.sin() * a2.cos())
}

#[test]
fn vignetting_by_a_field_aperture_matches_disk_overlap() {
    // Ideal lens (stop), then a clipping aperture 40 mm behind it: the beam
    // there is a disk of radius a(1 − d/f) centred at d·tan θ.
    let (f, epd, th, d, ra) = (100.0, 10.0, 6.0f64, 40.0, 4.0);
    let mut clip = surf(2, SurfaceShape::Plane, f - d, MaterialRef::Air);
    clip.semi_diameter = SemiDiameter::Fixed { value: ra };
    let sys = infinite_system(
        epd,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial { focal_length: f },
                d,
                MaterialRef::Air,
            ),
            clip,
        ],
        1,
        &[0.0, th],
    );
    let t = tracer(sys);
    let r = relative_illumination(
        &t,
        &RelativeIlluminationSettings {
            field_samples: 3,
            radial_samples: 32,
            azimuth_samples: 128,
            ..Default::default()
        },
    )
    .unwrap();
    let rb = 0.5 * epd * (1.0 - d / f);
    for (i, &h) in r.normalized_field.iter().enumerate() {
        let e = d * (th * h).to_radians().tan();
        let pass = lens_area(rb, ra, e) / (std::f64::consts::PI * rb * rb);
        assert!(
            ((1.0 - r.vignetting_fraction[i]) - pass).abs() < 0.01,
            "H {h}: {} vs {pass}",
            1.0 - r.vignetting_fraction[i]
        );
    }
    assert_eq!(r.vignetting_fraction[0], 0.0);
    assert!(r.vignetting_fraction[2] > 0.2);
    assert!(r.relative_illumination[2] < 0.8);
}

#[test]
fn footprint_on_an_ideal_lens() {
    let mut sys = ideal_lens(100.0, 10.0, &[0.0, 5.0]);
    let t = tracer(sys.clone());
    let fp = footprint(&t, &FootprintSettings::default()).unwrap();
    assert_eq!(fp.surface, 1);
    assert!((fp.max_radius - 5.0).abs() < 1e-12);
    assert!((fp.semi_diameter - 5.0).abs() < 1e-12);
    for r in &fp.rays {
        assert!(r.clipped.is_empty() && r.lost_before == 0);
        assert_eq!(r.points.len(), r.launched);
    }
    // A smaller fixed aperture clips the rim.
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 4.0 };
    let t = tracer(sys);
    let fp = footprint(&t, &FootprintSettings::default()).unwrap();
    let r = &fp.rays[0];
    assert!(!r.clipped.is_empty());
    assert!(r.points.iter().all(|p| p[0].hypot(p[1]) <= 4.0));
    assert!(r.clipped.iter().all(|p| p[0].hypot(p[1]) > 4.0));
    assert_eq!(r.points.len() + r.clipped.len(), r.launched);
    assert!(fp.max_radius <= 4.0);
}

fn real_aiming(t: &SequentialTracer) -> SequentialTracer {
    SequentialTracer::with_settings(
        t.system_arc(),
        TracerSettings {
            aiming: Some(RayAiming::Real { tolerance: 1e-12 }),
            ..*t.settings()
        },
    )
    .unwrap()
}

fn mono_tracer(mut sys: mklight_core::OpticalSystem) -> SequentialTracer {
    sys.wavelengths = mono();
    tracer(sys)
}

/// Chief-ray image point of normalised field `(hx, hy)`.
fn chief_xy(t: &SequentialTracer, hx: f64, hy: f64) -> [f64; 2] {
    let f = t.prepare_field(FieldPoint::normalized(hx, hy)).unwrap();
    let c = t.chief(&f, 0);
    assert!(c.status.is_ok());
    [c.ray.pos.x, c.ray.pos.y]
}

/// Independent étendue form of the relative illumination of an object at
/// infinity (angle fields, no vignetting): `E ∝ A_⊥ dΩ_obj/dA_img`, with the
/// collimated beam's cross-section `A_⊥` integrated from the generated
/// (really aimed) starting rays on the wavefront plane, `dΩ = dq/(1 + q²)^{3/2}`
/// for direction tangents `q`, and the chief-ray image Jacobian
/// `dA_img = |det ∂X/∂q| dq`.
fn etendue_ri(t: &SequentialTracer, hs: &[f64]) -> Vec<f64> {
    let t = real_aiming(t);
    let th = t.system().system().fields.max_radial().to_radians();
    let (rx, rw) = gauss_legendre_01(16);
    let m = 64;
    let e = |h: f64| {
        let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
        let mklight_trace::ObjectSource::Direction { dir: d } = f.source else {
            panic!()
        };
        let e1 = (Vec3::X - d * d.x).normalized();
        let e2 = d.cross(e1);
        let pos = |px: f64, py: f64| {
            let p = t.generate(&f, 0, px, py).unwrap().ray.pos;
            [p.dot(e1), p.dot(e2)]
        };
        const D: f64 = 1e-4;
        let mut area = 0.0;
        for (r, w) in rx.iter().zip(&rw) {
            for j in 0..m {
                let (s, c) = (2.0 * std::f64::consts::PI * j as f64 / m as f64).sin_cos();
                let (px, py) = (r * c, r * s);
                let (a, b) = (pos(px + D, py), pos(px - D, py));
                let (u, v) = (pos(px, py + D), pos(px, py - D));
                let jx = [(a[0] - b[0]) / (2.0 * D), (a[1] - b[1]) / (2.0 * D)];
                let jy = [(u[0] - v[0]) / (2.0 * D), (u[1] - v[1]) / (2.0 * D)];
                area += w
                    * r
                    * (2.0 * std::f64::consts::PI / m as f64)
                    * (jx[0] * jy[1] - jx[1] * jy[0]).abs();
            }
        }
        // ∂X/∂h by central differences, then q = tan(h θ_max) per axis.
        let dh = 1e-5;
        let (xa, xb) = (chief_xy(&t, dh, h), chief_xy(&t, -dh, h));
        let (ya, yb) = (chief_xy(&t, 0.0, h + dh), chief_xy(&t, 0.0, h - dh));
        let jh = [
            [(xa[0] - xb[0]) / (2.0 * dh), (ya[0] - yb[0]) / (2.0 * dh)],
            [(xa[1] - xb[1]) / (2.0 * dh), (ya[1] - yb[1]) / (2.0 * dh)],
        ];
        let det_h = (jh[0][0] * jh[1][1] - jh[0][1] * jh[1][0]).abs();
        let qy = (h * th).tan();
        let det_q = det_h / (th * th * (1.0 + qy * qy));
        area * (1.0 + qy * qy).powf(-1.5) / det_q
    };
    let e0 = e(0.0);
    hs.iter().map(|&h| e(h) / e0).collect()
}

#[test]
fn real_singlet_with_stop_at_the_lens_follows_cos4_times_distortion_jacobian() {
    // Stop on the (spherical) front surface, whose rim is a plane circle: the
    // beam cross-section is π r² cos θ, so E ∝ cos θ · sin θ/(y dy/dθ) with
    // the real chief heights y(θ), i.e. cos⁴θ times the distortion Jacobian
    // f² tan θ sec²θ/(y y'). The singlet has ~1.3 mm of field curvature at
    // 5°, which must not enter (the image irradiance is fixed by étendue).
    let t = mono_tracer(samples::singlet());
    let r = relative_illumination(
        &t,
        &RelativeIlluminationSettings {
            field_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let th_max = 5f64.to_radians();
    let k = t.paraxial().primary().image_height_per_tan.unwrap();
    let rt = real_aiming(&t);
    for (i, &h) in r.normalized_field.iter().enumerate().skip(1) {
        let th = h * th_max;
        let y = chief_xy(&rt, 0.0, h)[1];
        let dh = 1e-5;
        let dy =
            (chief_xy(&rt, 0.0, h + dh)[1] - chief_xy(&rt, 0.0, h - dh)[1]) / (2.0 * dh * th_max);
        let expect = th.cos() * th.sin() * k * k / (y * dy);
        let cos4_jac = th.cos().powi(4) * (k * th.tan() / y) * (k / (th.cos().powi(2) * dy));
        assert!((expect - cos4_jac).abs() < 1e-12);
        assert!(r.relative_illumination[i] < 1.0, "RI must fall with field");
        assert!(
            (r.relative_illumination[i] - expect).abs() < 1e-3 * expect,
            "H {h}: RI {} vs étendue {expect}",
            r.relative_illumination[i]
        );
        assert_eq!(r.vignetting_fraction[i], 0.0);
    }
}

/// Exact object-side form of the irradiance at the chief image point `X₀` of
/// field `h` (uniform Lambertian object at infinity, no vignetting):
/// `E ∝ ∫ dP A_⊥(q, P) ω(q) / |det ∂X/∂q (q, P)|` at the object point
/// `q(P)` whose ray through stop point `P` lands on `X₀`. Here `A_⊥` is the
/// beam cross-section density on the wavefront plane and
/// `ω = (1 + q²)^{−3/2}` the solid-angle density of the direction tangents.
/// This integrates the delta function `δ(X(q, P) − X₀)` over the object
/// side, while the analysis integrates image-side directions; étendue
/// conservation makes them equal.
fn object_side_ri(t: &SequentialTracer, hs: &[f64]) -> Vec<f64> {
    let t = real_aiming(t);
    let (rx, rw) = gauss_legendre_01(10);
    let m = 40;
    let e = |h: f64| {
        let base = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
        let field = |q: [f64; 2]| PreparedField {
            point: base.point,
            source: ObjectSource::Direction {
                dir: Vec3::new(q[0], q[1], 1.0).normalized(),
            },
            aims: base.aims.clone(),
            error: None,
        };
        let un = TraceOptions::unclipped();
        let img = |q: [f64; 2], p: [f64; 2]| {
            let r = t.trace(&field(q), 0, p[0], p[1], &un);
            assert!(r.status.is_ok());
            [r.ray.pos.x, r.ray.pos.y]
        };
        let xq = |q: [f64; 2], p: [f64; 2]| {
            let hq = 1e-6;
            let a = img([q[0] + hq, q[1]], p);
            let b = img([q[0] - hq, q[1]], p);
            let c = img([q[0], q[1] + hq], p);
            let d = img([q[0], q[1] - hq], p);
            [
                [(a[0] - b[0]) / (2.0 * hq), (c[0] - d[0]) / (2.0 * hq)],
                [(a[1] - b[1]) / (2.0 * hq), (c[1] - d[1]) / (2.0 * hq)],
            ]
        };
        let ObjectSource::Direction { dir } = base.source else {
            panic!()
        };
        let q0 = [dir.x / dir.z, dir.y / dir.z];
        let x0 = img(q0, [0.0, 0.0]);
        let mut sum = 0.0;
        for (r, w) in rx.iter().zip(&rw) {
            for j in 0..m {
                let (s, c) = (2.0 * std::f64::consts::PI * j as f64 / m as f64).sin_cos();
                let p = [r * c, r * s];
                // Full Newton for q(P).
                let mut q = q0;
                for _ in 0..30 {
                    let x = img(q, p);
                    let res = [x[0] - x0[0], x[1] - x0[1]];
                    if res[0].hypot(res[1]) < 1e-11 {
                        break;
                    }
                    let a = xq(q, p);
                    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                    q[0] -= (a[1][1] * res[0] - a[0][1] * res[1]) / det;
                    q[1] -= (a[0][0] * res[1] - a[1][0] * res[0]) / det;
                }
                let x = img(q, p);
                assert!((x[0] - x0[0]).hypot(x[1] - x0[1]) < 1e-9);
                let f = field(q);
                let ObjectSource::Direction { dir: d } = f.source else {
                    panic!()
                };
                let e1 = (Vec3::X - d * d.x).normalized();
                let e2 = d.cross(e1);
                let pos = |px: f64, py: f64| {
                    let v = t.generate(&f, 0, px, py).unwrap().ray.pos;
                    [v.dot(e1), v.dot(e2)]
                };
                const D: f64 = 1e-4;
                let (a, b) = (pos(p[0] + D, p[1]), pos(p[0] - D, p[1]));
                let (u, v) = (pos(p[0], p[1] + D), pos(p[0], p[1] - D));
                let jx = [(a[0] - b[0]) / (2.0 * D), (a[1] - b[1]) / (2.0 * D)];
                let jy = [(u[0] - v[0]) / (2.0 * D), (u[1] - v[1]) / (2.0 * D)];
                let a_perp = (jx[0] * jy[1] - jx[1] * jy[0]).abs();
                let jq = xq(q, p);
                let det_q = (jq[0][0] * jq[1][1] - jq[0][1] * jq[1][0]).abs();
                let omega = (1.0 + q[0] * q[0] + q[1] * q[1]).powf(-1.5);
                sum += w * r * (2.0 * std::f64::consts::PI / m as f64) * a_perp * omega / det_q;
            }
        }
        sum
    };
    let e0 = e(0.0);
    hs.iter().map(|&h| e(h) / e0).collect()
}

#[test]
fn relative_illumination_matches_object_side_and_etendue_forms() {
    // Field-curved lenses, the triplet and double Gauss with the stop inside
    // (pupil aberrations). The exact object-side integral must agree to the
    // finite-difference noise (observed ~1e-10).
    // The étendue form (chief-ray Jacobian for the whole beam) agrees only to
    // first order in the aberrations, so its tolerance is looser (observed
    // 7e-5 on the singlet, 5e-3 on the triplet and 8e-3 on the double Gauss
    // at full field).
    for (name, sys, tol_exact, tol_etendue) in [
        ("singlet", samples::singlet(), 1e-8, 2e-4),
        ("cooke", samples::cooke_triplet(), 1e-8, 1e-2),
        ("double_gauss", samples::double_gauss(), 1e-8, 1.5e-2),
    ] {
        let t = mono_tracer(sys);
        let r = relative_illumination(
            &t,
            &RelativeIlluminationSettings {
                field_samples: 3,
                radial_samples: 10,
                azimuth_samples: 40,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.run_info
                .warnings
                .iter()
                .all(|w| !w.contains("pupil nodes"))
        );
        if r.vignetting_fraction.iter().any(|&v| v > 0.0) {
            panic!("{name}: the references assume no vignetting");
        }
        let exact = object_side_ri(&t, &r.normalized_field);
        let et = etendue_ri(&t, &r.normalized_field);
        for (i, h) in r.normalized_field.iter().enumerate() {
            let ri = r.relative_illumination[i];
            assert!(
                (ri - exact[i]).abs() < tol_exact * exact[i],
                "{name} H {h}: RI {ri} vs object side {}",
                exact[i]
            );
            assert!(
                (ri - et[i]).abs() < tol_etendue * et[i],
                "{name} H {h}: RI {ri} vs étendue {}",
                et[i]
            );
        }
    }
}

#[test]
fn relative_illumination_never_exceeds_one_on_unvignetted_samples() {
    // The beam-focus measure grew with field curvature (RI > 1 at the edge of
    // the achromat and relay); the image-point measure must not.
    for (name, sys) in [
        ("achromat", samples::achromatic_doublet()),
        ("relay", samples::finite_relay()),
    ] {
        let t = mono_tracer(sys);
        let r = relative_illumination(
            &t,
            &RelativeIlluminationSettings {
                field_samples: 5,
                ..Default::default()
            },
        )
        .unwrap();
        for (i, v) in r.relative_illumination.iter().enumerate().skip(1) {
            assert!(
                *v < 1.0 && *v <= r.relative_illumination[i - 1] + 1e-12,
                "{name} {i}: {v}"
            );
        }
    }
}
