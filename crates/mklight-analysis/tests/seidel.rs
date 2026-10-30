//! Seidel sums against closed forms and real rays; longitudinal aberration,
//! chromatic focal shift and lateral colour.

mod common;

use common::*;
use mklight_analysis::seidel::*;
use mklight_core::{ApertureSpec, Interaction, MaterialRef, SurfaceShape, samples};
use mklight_trace::{FieldPoint, ParaxialModel, SequentialTracer, TraceOptions};

fn mirror(r: f64, conic: f64, stop_at_centre: bool, epd: f64, field: f64) -> SequentialTracer {
    let mut m = surf(
        if stop_at_centre { 2 } else { 1 },
        SurfaceShape::Conic {
            curvature: -1.0 / r,
            conic,
        },
        -0.5 * r,
        MaterialRef::Air,
    );
    m.interaction = Interaction::Reflect;
    let (body, stop) = if stop_at_centre {
        (
            vec![surf(1, SurfaceShape::Plane, r, MaterialRef::Air), m],
            1,
        )
    } else {
        (vec![m], 1)
    };
    tracer(infinite_system(epd, body, stop, &[0.0, field]))
}

#[test]
fn spherical_mirror_closed_forms() {
    let (r, epd, th) = (200.0f64, 20.0, 3.0f64);
    let y = 0.5 * epd;
    let ub = th.to_radians().tan();
    let c = -1.0 / r;
    let h = ub * y;
    // Stop at the mirror.
    let s = seidel(&mirror(r, 0.0, false, epd, th), &SeidelSettings::default()).unwrap();
    let t = s.sums;
    let tol = |v: f64| 1e-12 * (1.0 + v.abs());
    assert!((t.s1 - (-2.0 * c.powi(3) * y.powi(4))).abs() < tol(t.s1));
    assert!(t.s1 > 0.0); // concave mirror: under-corrected like a positive lens
    assert!((t.s2 - (-2.0 * c * c * y.powi(3) * ub)).abs() < tol(t.s2));
    assert!((t.s3 - (-2.0 * c * y * y * ub * ub)).abs() < tol(t.s3));
    assert!((t.s4 - 2.0 * h * h * c).abs() < tol(t.s4));
    assert!((t.s3 + t.s4).abs() < 1e-15, "flat sagittal field");
    assert!(t.s5.abs() < 1e-15 && t.cl == 0.0 && t.ct == 0.0);
    assert!((s.lagrange_invariant - h).abs() < 1e-15);
    // W040 = y⁴/(4R³): twice the sphere–paraboloid sag difference.
    assert!((s.wavefront_mm.w040 - y.powi(4) / (4.0 * r.powi(3))).abs() < 1e-15);
    // Petzval radius of a mirror: R/2, concave towards the mirror (+z).
    assert!(
        (s.petzval_radius - 0.5 * r).abs() < 1e-9,
        "{}",
        s.petzval_radius
    );
    // Paraboloid: no spherical aberration; coma etc. unchanged (stop at mirror).
    let p = seidel(&mirror(r, -1.0, false, epd, th), &SeidelSettings::default()).unwrap();
    assert!(p.sums.s1.abs() < 1e-15);
    assert!((p.sums.s2 - t.s2).abs() < 1e-15 && (p.sums.s3 - t.s3).abs() < 1e-15);
    // Stop at the centre of curvature: only spherical and Petzval.
    let q = seidel(&mirror(r, 0.0, true, epd, th), &SeidelSettings::default()).unwrap();
    assert!((q.sums.s1 - t.s1).abs() < tol(t.s1));
    assert!(q.sums.s2.abs() < 1e-14 && q.sums.s3.abs() < 1e-14 && q.sums.s5.abs() < 1e-14);
    assert!((q.sums.s4 - t.s4).abs() < tol(t.s4));
}

#[test]
fn thin_plano_convex_lens_matches_thin_lens_theory() {
    // Convex side to infinity, stop at the lens, 1 µm thick: the thin-lens
    // formulas hold to O(t).
    let r = 51.68;
    let lens = |t: f64| {
        infinite_system(
            10.0,
            vec![
                surf(1, SurfaceShape::sphere(r), t, MaterialRef::glass("N-BK7")),
                surf(2, SurfaceShape::Plane, 100.0, MaterialRef::Air),
            ],
            1,
            &[0.0, 5.0],
        )
    };
    let tr = tracer(lens(1e-3));
    let s = seidel(&tr, &SeidelSettings::default()).unwrap();
    let n = tr.system().surfaces[1].n_after[0];
    let k = 1.0 / tr.paraxial().primary().efl;
    let y: f64 = 5.0;
    let h = s.lagrange_invariant;
    let (b, c) = (1.0, -1.0);
    let s1 = y.powi(4) * k.powi(3) / 4.0
        * ((n / (n - 1.0)).powi(2)
            + (n + 2.0) / (n * (n - 1.0).powi(2))
                * (b + 2.0 * (n * n - 1.0) / (n + 2.0) * c).powi(2)
            - n / (n + 2.0) * c * c);
    let s2 = y * y * k * k * h / 2.0 * ((n + 1.0) / (n * (n - 1.0)) * b + (2.0 * n + 1.0) / n * c);
    let rel = |a: f64, e: f64| (a - e).abs() / e.abs();
    assert!(rel(s.sums.s1, s1) < 1e-4, "{} {}", s.sums.s1, s1);
    assert!(
        rel(s.sums.s2.abs(), s2.abs()) < 1e-4,
        "{} {}",
        s.sums.s2,
        s2
    );
    assert!(rel(s.sums.s3, h * h * k) < 1e-4);
    assert!(rel(s.sums.s4, h * h * k / n) < 1e-4);
    assert!(s.sums.s5.abs() < 1e-4 * s.sums.s3.abs());
    // Petzval sum = Σφ/(n n') = K/n for a thin lens.
    assert!(rel(s.petzval_sum, k / n) < 1e-4);
}

#[test]
fn petzval_sum_matches_paraxial_powers_on_every_sample() {
    for (name, t) in samples() {
        let s = seidel(&t, &SeidelSettings::default()).unwrap();
        let m = ParaxialModel::new(t.system(), t.system().primary_wavelength);
        let n = &m.reduced_index;
        let p: f64 = (1..n.len() - 1)
            .map(|j| m.power[j] / (n[j - 1] * n[j]))
            .sum();
        assert!(
            (s.petzval_sum - p).abs() < 1e-14 * (1.0 + p.abs()),
            "{name}"
        );
        let h2 = s.lagrange_invariant.powi(2);
        assert!(
            (s.sums.s4 - h2 * p).abs() < 1e-12 * (1.0 + s.sums.s4.abs()),
            "{name}"
        );
        // S_V division-free form equals (Ā/A)(S_III + S_IV) wherever A ≠ 0.
        let fo = t.paraxial().primary();
        for row in &s.surfaces {
            let j = row.surface;
            let cv = t.system().surfaces[j].shape.curvature();
            let a = fo.marginal.nu[j - 1] + n[j - 1] * fo.marginal.y[j] * cv;
            let ab = fo.chief.nu[j - 1] + n[j - 1] * fo.chief.y[j] * cv;
            let asph = matches!(
                t.system().surfaces[j].shape,
                SurfaceShape::EvenAsphere { .. }
            ) || t.system().surfaces[j].shape.conic() != 0.0;
            if a.abs() > 1e-6 && !asph && !t.system().surfaces[j].is_passive() {
                let e = ab / a * (row.terms.s3 + row.terms.s4);
                assert!(
                    (row.terms.s5 - e).abs() < 1e-10 * (1.0 + e.abs()),
                    "{name} {j}"
                );
            }
        }
    }
}

/// Real-ray transverse aberration relative to the real chief ray.
fn real_eps(t: &SequentialTracer, h: f64, px: f64, py: f64) -> [f64; 2] {
    let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
    let o = TraceOptions::unclipped();
    let c = t.trace(&f, 0, 0.0, 0.0, &o);
    let r = t.trace(&f, 0, px, py, &o);
    assert!(c.status.is_ok() && r.status.is_ok());
    [r.ray.pos.x - c.ray.pos.x, r.ray.pos.y - c.ray.pos.y]
}

/// A sample at reduced aperture and field, monochromatic, with the image at
/// the paraxial focus.
fn small(
    sys: mklight_core::OpticalSystem,
    aperture_scale: f64,
    max_field: f64,
) -> SequentialTracer {
    let mut s = sys;
    s.wavelengths = mono();
    s.fields = angle_fields(&[0.0, max_field]);
    if let ApertureSpec::EntrancePupilDiameter { value } = s.aperture {
        s.aperture = ApertureSpec::EntrancePupilDiameter {
            value: value * aperture_scale,
        };
    }
    let t0 = tracer(s.clone());
    let n = s.surfaces.len();
    s.surfaces[n - 2].thickness = t0.paraxial().primary().image_distance;
    tracer(s)
}

#[test]
fn seidel_predicts_real_ray_aberrations_at_small_aperture_and_field() {
    // Agreement improves as aperture and field shrink (fifth order ∝ scale²);
    // the Cooke triplet balances third against fifth order, so its residual is
    // larger. Every comparison also checks the sign (coma direction etc.).
    for (name, sys, tol) in [
        ("cooke", samples::cooke_triplet(), 0.08),
        ("singlet", samples::singlet(), 0.02),
    ] {
        let t = small(sys, 0.25, 2.0);
        let s = seidel(&t, &SeidelSettings::default()).unwrap();
        let check = |what: &str, real: f64, pred: f64| {
            assert!(
                (real - pred).abs() <= tol * pred.abs(),
                "{name} {what}: real {real:e} vs Seidel {pred:e}"
            );
        };
        // Spherical: on-axis tangential fan.
        check(
            "spherical",
            real_eps(&t, 0.0, 0.0, 1.0)[1],
            s.transverse_aberration(0.0, 0.0, 1.0, false)[1],
        );
        // Coma (even part) and astigmatism + field curvature (odd part) of
        // the full-field tangential fan.
        let (a, b) = (
            real_eps(&t, 1.0, 0.0, 1.0)[1],
            real_eps(&t, 1.0, 0.0, -1.0)[1],
        );
        let qa = s.transverse_aberration(1.0, 0.0, 1.0, false)[1];
        let qb = s.transverse_aberration(1.0, 0.0, -1.0, false)[1];
        check("tangential coma", 0.5 * (a + b), 0.5 * (qa + qb));
        check("tangential odd", 0.5 * (a - b), 0.5 * (qa - qb));
        // Sagittal fan: εx (spherical + sagittal field), εy (sagittal coma).
        let sx = real_eps(&t, 1.0, 1.0, 0.0);
        let qx = s.transverse_aberration(1.0, 1.0, 0.0, false);
        check("sagittal x", sx[0], qx[0]);
        check("sagittal coma", sx[1], qx[1]);
        // Distortion: real chief vs paraxial chief.
        let f = t.prepare_field(FieldPoint::normalized(0.0, 1.0)).unwrap();
        let yc = t.chief(&f, 0).ray.pos.y;
        let yp = t.paraxial().primary().paraxial_image_height;
        check(
            "distortion",
            yc - yp,
            s.transverse_aberration(1.0, 0.0, 0.0, true)[1],
        );
    }
}

#[test]
fn w040_matches_on_axis_opd() {
    let t = small(samples::singlet(), 0.25, 2.0);
    let s = seidel(&t, &SeidelSettings::default()).unwrap();
    let f = t.prepare_field(FieldPoint::normalized(0.0, 0.0)).unwrap();
    let opd = t.opd(&f, 0, 0.0, 1.0, &TraceOptions::unclipped()).unwrap();
    let w = s.wavefront_waves.w040;
    assert!(
        w > 0.0 && (opd - w).abs() < 0.02 * w,
        "OPD {opd} vs W040 {w}"
    );
}

#[test]
fn longitudinal_aberration_of_mirrors_is_exact() {
    let (r, epd) = (200.0f64, 40.0);
    let set = LongitudinalSettings {
        pupil_samples: 6,
        ..Default::default()
    };
    let sph = longitudinal_aberration(&mirror(r, 0.0, false, epd, 1.0), &set).unwrap();
    let par = longitudinal_aberration(&mirror(r, -1.0, false, epd, 1.0), &set).unwrap();
    for (i, &rho) in sph.pupil.iter().enumerate() {
        let si = rho * 0.5 * epd / r;
        let exact = 0.5 * r * (1.0 / (1.0 - si * si).sqrt() - 1.0);
        let got = sph.curves[0].aberration[i];
        assert!((got - exact).abs() < 1e-9, "{rho}: {got} vs {exact}");
        assert!(par.curves[0].aberration[i].abs() < 1e-9);
    }
    assert!(sph.primary_paraxial_focus.abs() < 1e-12);
}

#[test]
fn colour_terms_match_paraxial_and_real_colour() {
    let t = tracer(samples::singlet());
    let sys = t.system();
    let s = seidel(&t, &SeidelSettings::default()).unwrap();
    let cfs = chromatic_focal_shift(&t, &ChromaticFocalShiftSettings::default()).unwrap();
    // The sweep end points are the system's F and C lines.
    let (ws, wl) = (s.short_wavelength, s.long_wavelength);
    let par = t.paraxial();
    let pf = |w: usize| {
        let m = &par.at(w).marginal;
        let n = m.y.len();
        -m.y[n - 1] / m.u[n - 2]
    };
    let k = cfs.wavelengths_um.len() - 1;
    assert!((cfs.focal_shift[0] - (pf(ws) - pf(1))).abs() < 1e-12);
    assert!((cfs.focal_shift[k] - (pf(wl) - pf(1))).abs() < 1e-12);
    // Thin-lens longitudinal colour f_F − f_C = −f/V.
    let n = |w: usize| sys.surfaces[1].n_after[w];
    let v = (n(1) - 1.0) / (n(0) - n(2));
    let efl = par.primary().efl;
    let dz = cfs.focal_shift[0] - cfs.focal_shift[k];
    assert!(
        (dz + efl / v).abs() < 0.03 * efl / v,
        "{dz} vs {}",
        -efl / v
    );
    // Third order: δz = −n' C_L / ω'², δy = C_T / ω' (at full field).
    let w1 = s.image_marginal_slope;
    let pz = -s.sums.cl / (w1 * w1);
    assert!((dz - pz).abs() < 0.02 * pz.abs(), "{dz} vs {pz}");
    let lc = lateral_colour(&t, &LateralColourSettings::default()).unwrap();
    let dy = lc.lateral_colour[lc.lateral_colour.len() - 1];
    let py = s.sums.ct / w1;
    assert!((dy - py).abs() < 0.02 * py.abs(), "{dy} vs {py}");
    assert_eq!(lc.lateral_colour[0], 0.0);
    // An ideal lens has neither.
    let i = tracer(ideal_lens(100.0, 10.0, &[0.0, 10.0]));
    let c = chromatic_focal_shift(&i, &ChromaticFocalShiftSettings::default()).unwrap();
    assert!(c.range == 0.0);
    let l = lateral_colour(&i, &LateralColourSettings::default()).unwrap();
    assert!(l.lateral_colour.iter().all(|v| v.abs() < 1e-12));
}
