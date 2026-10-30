//! Seidel sums against closed forms and real rays; longitudinal aberration,
//! chromatic focal shift and lateral colour.

mod common;

use common::*;
use mklight_analysis::seidel::*;
use mklight_core::{ApertureSpec, Interaction, MaterialRef, Placement, SurfaceShape, samples};
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
    // Petzval sum of a mirror with signed reduced indices: φ/(n n') with
    // φ = c(n' − n) = 2/R, n n' = −1, so P = −2/R.
    assert!((s.petzval_sum + 2.0 / r).abs() < 1e-15, "{}", s.petzval_sum);
    // Petzval radius −1/(n'_k P) with n'_k = −1: −R/2. Image space runs
    // towards local −z, and the Petzval surface (the sphere of radius R/2
    // about the centre of curvature) bends back towards −z.
    assert!(
        (s.petzval_radius + 0.5 * r).abs() < 1e-9,
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
    // Independent Lagrange invariant: ȳ = 0 at the stop, so H = ū y with the
    // chief ray of the +y field rising (ū = tan 5° > 0).
    let h = 5f64.to_radians().tan() * y;
    assert!((s.lagrange_invariant - h).abs() < 1e-12 * h);
    let (b, c) = (1.0, -1.0);
    let s1 = y.powi(4) * k.powi(3) / 4.0
        * ((n / (n - 1.0)).powi(2)
            + (n + 2.0) / (n * (n - 1.0).powi(2))
                * (b + 2.0 * (n * n - 1.0) / (n + 2.0) * c).powi(2)
            - n / (n + 2.0) * c * c);
    let s2 = y * y * k * k * h / 2.0 * ((n + 1.0) / (n * (n - 1.0)) * b + (2.0 * n + 1.0) / n * c);
    // Sign of coma in this crate's convention, derived surface by surface
    // from S_II = −A Ā y Δ(u/n) with ȳ = 0 (so Ā = ū at both surfaces),
    // A₁ = yc₁, Δ₁(u/n) = −yK/n², A₂ = −yK, Δ₂(u/n) = −yK(1 − 1/n²):
    // S_II = −H y² K² (n² − n − 1)/(n(n − 1)), which is the thin-lens form
    // above with shape factor B = +1 and conjugate factor C = −1.
    let s2_signed = -h * y * y * k * k * (n * n - n - 1.0) / (n * (n - 1.0));
    let rel = |a: f64, e: f64| (a - e).abs() / e.abs();
    assert!(rel(s2, s2_signed) < 1e-12);
    assert!(rel(s.sums.s1, s1) < 1e-4, "{} {}", s.sums.s1, s1);
    // n ≈ 1.517 < φ (golden ratio): n² − n − 1 < 0, so S_II > 0 here.
    assert!(s.sums.s2 > 0.0);
    assert!(rel(s.sums.s2, s2_signed) < 1e-4, "{} {}", s.sums.s2, s2);
    assert!(rel(s.sums.s3, h * h * k) < 1e-4);
    assert!(rel(s.sums.s4, h * h * k / n) < 1e-4);
    assert!(s.sums.s5.abs() < 1e-4 * s.sums.s3.abs());
    // Petzval sum = Σφ/(n n') = K/n for a thin lens.
    assert!(rel(s.petzval_sum, k / n) < 1e-4);
}

#[test]
fn petzval_sum_of_separated_lenses_is_sum_of_k_over_n() {
    // Two thick lenses of different glasses, separated: P = Σ K_i/n_i
    // exactly (each lens contributes (n−1)(c₁ − c₂)/n whatever its
    // thickness), independent of spacing, stop position and bending.
    let (r1, r2, r3, r4) = (40.0, -120.0, -35.0, 80.0);
    let sys = infinite_system(
        10.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(r1),
                4.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(2, SurfaceShape::sphere(r2), 7.0, MaterialRef::Air),
            surf(3, SurfaceShape::sphere(r3), 2.0, MaterialRef::glass("F2")),
            surf(4, SurfaceShape::sphere(r4), 60.0, MaterialRef::Air),
        ],
        3,
        &[0.0, 4.0],
    );
    let t = tracer(sys);
    let s = seidel(&t, &SeidelSettings::default()).unwrap();
    let n1 = t.system().surfaces[1].n_after[0];
    let n2 = t.system().surfaces[3].n_after[0];
    assert!(n1 < 1.6 && n2 > 1.6);
    let k1 = (n1 - 1.0) * (1.0 / r1 - 1.0 / r2);
    let k2 = (n2 - 1.0) * (1.0 / r3 - 1.0 / r4);
    let p = k1 / n1 + k2 / n2;
    assert!((s.petzval_sum - p).abs() < 1e-14, "{} {p}", s.petzval_sum);
    // Refractive image space (n'_k = +1): R_p = −1/P.
    assert!((s.petzval_radius + 1.0 / p).abs() < 1e-9 / p.abs());
}

#[test]
fn petzval_radius_is_the_focal_surface_of_a_concentric_mirror() {
    // Stop at the centre of curvature: S_III = 0, so the Petzval surface is
    // the focal surface. Its sag h²/(2R_p) at the paraxial image height must
    // equal the Coddington and close-ray axial focus shifts at a small field.
    use mklight_analysis::curvature::{FieldCurvatureSettings, field_curvature};
    let (r, th) = (200.0, 1.0f64);
    let t = tracer(concentric_mirror(r, 2.0, &[0.0, th]));
    let s = seidel(&t, &SeidelSettings::default()).unwrap();
    assert!(s.sums.s3.abs() < 1e-14);
    assert!(s.petzval_radius < 0.0);
    let fc = field_curvature(
        &t,
        &FieldCurvatureSettings {
            field_samples: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let c = &fc.curves[0];
    let h = c.paraxial_height[1];
    let sag = h * h / (2.0 * s.petzval_radius);
    for dz in [
        c.tangential[1],
        c.sagittal[1],
        c.tangential_close_rays[1],
        c.sagittal_close_rays[1],
    ] {
        // Third order: relative error O(θ²).
        assert!((dz - sag).abs() < 1e-3 * sag.abs(), "{dz} vs {sag}");
    }
    // The report table agrees.
    let rep = mklight_analysis::system_data::system_data(&t, &Default::default()).unwrap();
    let p = rep.tables[0].get("petzval_radius").unwrap();
    assert!((p - s.petzval_radius).abs() < 1e-9 * p.abs());
}

#[test]
fn petzval_radius_matches_third_order_foci_of_a_refractive_lens() {
    // Third order: Δz_s ∝ S_III + S_IV, Δz_t ∝ 3S_III + S_IV, so the Petzval
    // shift is (3Δz_s − Δz_t)/2 = h²/(2R_p).
    use mklight_analysis::curvature::{FieldCurvatureSettings, field_curvature};
    let t = small(samples::singlet(), 0.25, 1.0);
    let s = seidel(&t, &SeidelSettings::default()).unwrap();
    let fc = field_curvature(
        &t,
        &FieldCurvatureSettings {
            field_samples: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let c = &fc.curves[0];
    let h = c.paraxial_height[1];
    let dzp = 0.5 * (3.0 * c.sagittal[1] - c.tangential[1]);
    let sag = h * h / (2.0 * s.petzval_radius);
    assert!(
        sag < 0.0,
        "positive lens: Petzval surface curves towards the lens"
    );
    assert!((dzp - sag).abs() < 2e-3 * sag.abs(), "{dzp} vs {sag}");
}

#[test]
fn s4_and_s5_are_consistent_on_every_sample() {
    for (name, t) in samples() {
        let s = seidel(&t, &SeidelSettings::default()).unwrap();
        if s.status == SeidelStatus::NotApplicable {
            continue; // folded (non-rotational) samples
        }
        let m = ParaxialModel::new(t.system(), t.system().primary_wavelength);
        let n = &m.reduced_index;
        // S_IV = H² P (P checked against closed forms elsewhere).
        let h2 = s.lagrange_invariant.powi(2);
        assert!(
            (s.sums.s4 - h2 * s.petzval_sum).abs() < 1e-12 * (1.0 + s.sums.s4.abs()),
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

#[test]
fn folded_newtonian_matches_the_unfolded_mirror() {
    let folded = seidel(
        &tracer(samples::newtonian_telescope()),
        &SeidelSettings::default(),
    )
    .unwrap();
    assert_eq!(folded.status, SeidelStatus::Applicable);
    let mut sys = samples::newtonian_telescope();
    sys.surfaces.drain(2..5);
    sys.surfaces[1].thickness = -500.0;
    let flat = seidel(&tracer(sys), &SeidelSettings::default()).unwrap();
    assert_eq!(flat.status, SeidelStatus::Applicable);
    let (a, b) = (folded.sums, flat.sums);
    for (x, y) in [
        (a.s1, b.s1),
        (a.s2, b.s2),
        (a.s3, b.s3),
        (a.s4, b.s4),
        (a.s5, b.s5),
        (a.cl, b.cl),
        (a.ct, b.ct),
        (folded.petzval_sum, flat.petzval_sum),
        (folded.lagrange_invariant, flat.lagrange_invariant),
    ] {
        assert!(x.is_finite() && (x - y).abs() < 1e-12, "{a:?} {b:?}");
    }
}

#[test]
fn decentred_lens_is_not_applicable() {
    let mut sys = samples::singlet();
    sys.surfaces[1].placement = Some(Placement {
        decenter_y: 1.0,
        ..Default::default()
    });
    let s = seidel(&tracer(sys), &SeidelSettings::default()).unwrap();
    assert_eq!(s.status, SeidelStatus::NotApplicable);
    assert!(s.sums.s1.is_nan());
}
