//! Cross-validation of the `Primary` and `Ghost` classes against the
//! sequential reference tracer with every scatter mechanism disabled. Each
//! comparison prints its z-score `(MC − reference)/SE_MC`, where `SE_MC` is
//! the estimator's own standard error. Every case uses its own seed, so the
//! z-scores are independent.
//!
//! The reference shares the source sampling with the Monte Carlo, so it is
//! itself checked independently of the source: its flux does not depend on
//! the source margin, and its geometric spot matches the sequential
//! tracer's own pupil-grid spot.

mod common;

use common::*;
use mklight_core::JobContext;
use mklight_core::materials::coating::fresnel;
use mklight_core::model::*;
use mklight_stray::{
    Beam, PathClass, PrimaryReference, StraySettings, StraySystem, SurfaceOp, primary_reference,
};

const Z_MAX: f64 = 3.0;

fn transport(rays: u64, seed: u64) -> StraySettings {
    transport_with(rays, seed, mklight_stray::SplitMode::default())
}

fn transport_with(rays: u64, seed: u64, mode: mklight_stray::SplitMode) -> StraySettings {
    StraySettings {
        rays,
        seed,
        scatter_reflection: false,
        scatter_transmission: false,
        mode,
        ..Default::default()
    }
}

/// Checks MC `Primary` flux, centroid and RMS radius against the
/// sequential reference for one source. Returns the reference and the MC
/// flux with its standard error.
fn check_primary(
    name: &str,
    sys: &StraySystem,
    beam: &Beam,
    rays: u64,
    seed: u64,
) -> (PrimaryReference, f64, f64) {
    let ctx = JobContext::new();
    let r = mklight_stray::run(&sys.scene, &beam.spec, &transport(rays, seed), &ctx).unwrap();
    assert!(r.energy.is_balanced(), "{}", r.energy.summary());
    let e = &r.energy;
    assert_eq!(e.detected_in(PathClass::Scatter), 0.0);
    assert_eq!(e.detected_in(PathClass::Mechanical), 0.0);
    let reference = primary_reference(sys, &beam.spec, 360, &ctx).unwrap();
    let coarse = primary_reference(sys, &beam.spec, 250, &ctx).unwrap();

    let mc = e.detected_in(PathClass::Primary);
    let se = e.std_error_in(PathClass::Primary);
    let m = r.detector.moments_of(PathClass::Primary);
    let zf = z(mc, reference.flux, se);
    let zx = z(
        m.centroid[0],
        reference.centroid[0],
        m.centroid_std_error[0],
    );
    let zy = z(
        m.centroid[1],
        reference.centroid[1],
        m.centroid_std_error[1],
    );
    let zr = z(m.rms_radius, reference.rms_radius, m.rms_radius_std_error);
    println!(
        "{name}: transmitted {:.6} vs {:.6} of the emitted flux (z = {zf:+.2}), centroid ({:.6}, {:.6}) vs ({:.6}, {:.6}) mm (z = {zx:+.2}, {zy:+.2}), RMS {:.6} vs {:.6} mm (z = {zr:+.2}); geometric RMS {:.6} mm",
        mc / e.emitted,
        reference.flux / reference.emitted,
        m.centroid[0],
        m.centroid[1],
        reference.centroid[0],
        reference.centroid[1],
        m.rms_radius,
        reference.rms_radius,
        reference.geometric_rms_radius,
    );
    // The quadrature error is far below the Monte-Carlo error.
    assert!(
        (reference.flux - coarse.flux).abs() < 0.2 * se,
        "{name}: quadrature {} vs {} (SE {se})",
        reference.flux,
        coarse.flux
    );
    assert!((reference.rms_radius - coarse.rms_radius).abs() < 0.2 * m.rms_radius_std_error);
    for (what, zz) in [("flux", zf), ("x", zx), ("y", zy), ("rms", zr)] {
        assert!(zz.abs() < Z_MAX, "{name} {what}: z = {zz}");
    }
    // The error bars are tight enough for the comparison to mean something.
    assert!(se / mc < 0.01, "{name}: relative SE {}", se / mc);
    assert!(m.rms_radius_std_error < 0.03 * m.rms_radius.max(1e-3));
    // Something is lost to reflections and the margin, and the beam is
    // limited by the stop rather than the source.
    assert!(reference.flux < 0.95 * beam.reference_flux);
    assert!(reference.flux > 0.5 * beam.reference_flux);
    (reference, mc, se)
}

/// `check_primary` for a system without obstruction: every designed path
/// is unobstructed, so the reference equals the purely sequential model.
fn check_unobstructed(name: &str, sys: &StraySystem, beam: &Beam, rays: u64, seed: u64) {
    let (reference, _, _) = check_primary(name, sys, beam, rays, seed);
    assert_eq!(reference.obstructed, 0, "{name}");
    assert_eq!(reference.flux, reference.sequential_flux, "{name}");
}

/// The Double Gauss on axis and at 10°. (At its 14° edge field the
/// automatic clear apertures of L1 and L6 are so large that the two
/// surfaces of each element intersect inside them: see
/// `full_field_double_gauss_is_not_physical`.)
#[test]
fn double_gauss_primary_matches_sequential() {
    let sys = stray_system(mklight_core::samples::double_gauss(), &no_scatter_options());
    for (k, seed) in [(0usize, 11u64), (1, 12)] {
        let beam = sys.system_field_source(k, 0.1, 1.0).unwrap();
        check_unobstructed(
            &format!("double Gauss field {k}"),
            &sys,
            &beam,
            60_000,
            seed,
        );
    }
}

#[test]
fn cooke_triplet_primary_matches_sequential() {
    let sys = stray_system(
        mklight_core::samples::cooke_triplet(),
        &no_scatter_options(),
    );
    for (k, seed) in [(0usize, 31u64), (2, 32)] {
        let beam = sys.system_field_source(k, 0.1, 1.0).unwrap();
        check_unobstructed(
            &format!("Cooke triplet field {k}"),
            &sys,
            &beam,
            60_000,
            seed,
        );
    }
    // A field between the system fields.
    let beam = sys
        .field_source(mklight_trace::FieldPoint::normalized(0.0, -0.55), 0.2, 1.0)
        .unwrap();
    check_unobstructed("Cooke triplet field −0.55", &sys, &beam, 60_000, 33);
}

/// The Double Gauss sample without its vignetting factors: the unvignetted
/// 14° beam sizes clear apertures at which L1 and L6 have negative edge
/// thickness (the sample's vignetting avoids this, `samples::double_gauss`).
fn unvignetted_double_gauss() -> OpticalSystem {
    let mut s = mklight_core::samples::double_gauss();
    for f in &mut s.fields.fields {
        f.vignetting = Default::default();
    }
    s
}

/// The Double Gauss sample is physical over its clear apertures: no
/// crossing surfaces and no obstruction of the designed beam at any field.
#[test]
fn double_gauss_sample_is_physical() {
    let sys = stray_system(mklight_core::samples::double_gauss(), &no_scatter_options());
    assert!(
        !sys.scene
            .warnings
            .iter()
            .any(|w| w.contains("intersect inside their clear apertures")),
        "{:?}",
        sys.scene.warnings
    );
    for (k, o) in sys.obstruction.iter().enumerate() {
        assert!(o.unwrap() < 1e-9, "field {k}: {o:?}");
    }
}

/// Surfaces that intersect inside their clear apertures are reported. The
/// Monte Carlo then (correctly) departs from the purely sequential model,
/// which ignores the crossing; the reference's path check against the scene
/// sees the crossing as an obstruction of the designed path and reproduces
/// the Monte Carlo.
#[test]
fn crossing_surfaces_are_reported_and_obstruct() {
    let sys = stray_system(unvignetted_double_gauss(), &no_scatter_options());
    for pair in ["surfaces 1 and 2", "surfaces 10 and 11"] {
        assert!(
            sys.scene
                .warnings
                .iter()
                .any(|w| w.contains(pair) && w.contains("intersect inside their clear apertures")),
            "{:?}",
            sys.scene.warnings
        );
    }
    assert!(
        sys.warnings
            .iter()
            .any(|w| w.starts_with("field 2:") && w.contains("obstructed")),
        "{:?}",
        sys.warnings
    );
    assert!(sys.obstruction[0].unwrap() < 1e-9);
    let beam = sys.system_field_source(2, 0.1, 1.0).unwrap();
    let r = mklight_stray::run(
        &sys.scene,
        &beam.spec,
        &transport(30_000, 3),
        &JobContext::new(),
    )
    .unwrap();
    let reference = primary_reference(&sys, &beam.spec, 250, &JobContext::new()).unwrap();
    let mc = r.energy.detected_in(PathClass::Primary);
    let se = r.energy.std_error_in(PathClass::Primary);
    let zs = z(mc, reference.sequential_flux, se);
    let zf = z(mc, reference.flux, se);
    println!(
        "double Gauss 14°: MC {mc:.4} W, sequential {:.4} W (z = {zs:+.2}), obstruction-checked {:.4} W (z = {zf:+.2}), obstruction {:.2} %",
        reference.sequential_flux,
        reference.flux,
        100.0 * reference.obstruction()
    );
    assert!(zs < -Z_MAX, "z = {zs}");
    assert!(zf.abs() < Z_MAX, "z = {zf}");
}

#[test]
fn finite_relay_primary_matches_sequential() {
    let sys = stray_system(mklight_core::samples::finite_relay(), &no_scatter_options());
    let n_fields = sys.resolved().system().fields.fields.len();
    for (k, seed) in [(0usize, 21u64), (n_fields - 1, 22)] {
        let beam = sys.system_field_source(k, 0.1, 1.0).unwrap();
        assert!(matches!(beam.spec, mklight_stray::SourceSpec::Point { .. }));
        check_unobstructed(&format!("relay field {k}"), &sys, &beam, 60_000, seed);
    }
}

/// Plane-parallel plate (n = 1.5, 5 mm) with automatic semi-diameters, a
/// 10 mm entrance pupil on its front face and a single on-axis field.
fn plate() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Plane,
            5.0,
            MaterialRef::Constant { n: 1.5 },
        ),
        Surface::new(2, SurfaceShape::Plane, 20.0, MaterialRef::Air),
        Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

/// Known answer of the sequential reference itself: an uncoated plate at
/// normal incidence transmits `E·π r²·T²` through its stop.
#[test]
fn reference_transmittance_of_a_plate() {
    let sys = stray_system(plate(), &no_scatter_options());
    // The ghost solver's stop radius (paraxial radius × (1 + 10⁻¹²)), shared
    // so that both solvers clip at the same aperture.
    let r_stop = sys.clear_semi_diameters[1].unwrap();
    assert!((r_stop / 5.0 - 1.0).abs() <= 2e-12, "stop radius {r_stop}");
    let beam = sys.system_field_source(0, 0.2, 1.0).unwrap();
    let t = fresnel(1.0, 1.5, 1.0).t_unpolarized();
    let r = primary_reference(&sys, &beam.spec, 400, &JobContext::new()).unwrap();
    assert!((beam.reference_flux - std::f64::consts::PI * 25.0).abs() < 1e-12);
    assert!(
        (r.flux / (beam.reference_flux * t * t) - 1.0).abs() < 2e-3,
        "{} vs {}",
        r.flux,
        beam.reference_flux * t * t
    );
    assert!(r.rms_radius > 0.0 && (r.rms_radius - 5.0 / 2f64.sqrt()).abs() < 0.01);
}

/// The ghost of a single uncoated plate: the double reflection (back face,
/// then front face) carries `T²R²` of the flux through the stop, and all
/// ghost orders together `T²R²/(1 − R²)`.
#[test]
fn plate_ghost_matches_closed_form() {
    let sys = stray_system(plate(), &no_scatter_options());
    let beam = sys.system_field_source(0, 0.1, 1.0).unwrap();
    let f = fresnel(1.0, 1.5, 1.0);
    let (rr, tt) = (f.r_unpolarized(), f.t_unpolarized());
    let incident = beam.reference_flux;
    // Bare reflections (4 %) are above the default rare-branch floor; a
    // higher floor forces them so the ghost is well resolved.
    let set = transport_with(
        40_000,
        7,
        mklight_stray::SplitMode::Weighted { rare_floor: 0.25 },
    );
    let r = mklight_stray::run(&sys.scene, &beam.spec, &set, &JobContext::new()).unwrap();
    assert!(r.energy.is_balanced());
    let e = &r.energy;
    let ghost = e.detected_in(PathClass::Ghost);
    let se = e.std_error_in(PathClass::Ghost);
    let all_orders = incident * tt * tt * rr * rr / (1.0 - rr * rr);
    let second = incident * tt * tt * rr * rr;
    let z_all = z(ghost, all_orders, se);
    // Orders ≥ 4 are R² ≈ 0.2 % of the ghost: within the error bar.
    let z_two = z(e.ghost_by_order[2], second, se);
    println!(
        "plate ghost: {:.6e} vs {:.6e} W (z = {z_all:+.2}); second order {:.6e} vs {:.6e} W (z = {z_two:+.2}); relative SE {:.3}",
        ghost,
        all_orders,
        e.ghost_by_order[2],
        second,
        se / ghost
    );
    assert!(z_all.abs() < Z_MAX && z_two.abs() < Z_MAX);
    assert!(se / ghost < 0.05);
    let zp = z(
        e.detected_in(PathClass::Primary),
        incident * tt * tt,
        e.std_error_in(PathClass::Primary),
    );
    assert!(zp.abs() < Z_MAX, "primary z = {zp}");
    // The strongest ghost signature is "back face, then front face", whose
    // full ordered signature is T1 R2 R1 T2.
    let g = r
        .signatures
        .iter()
        .find(|s| s.class == PathClass::Ghost)
        .unwrap();
    assert_eq!(g.label, "R2 R1");
    let full = g
        .signature
        .expand_ghost(&sys.scene.sequence, |_| false)
        .unwrap();
    assert_eq!(
        full,
        [
            (1, SurfaceOp::Transmit),
            (2, SurfaceOp::Reflect),
            (1, SurfaceOp::Reflect),
            (2, SurfaceOp::Transmit)
        ]
    );
    assert!((g.flux - e.ghost_by_order[2]).abs() < 1e-12 * g.flux);
    // Primary is the empty ("designed") signature.
    let p = r
        .signatures
        .iter()
        .find(|s| s.class == PathClass::Primary)
        .unwrap();
    assert_eq!(p.label, "designed");
    assert_eq!(p.flux, e.detected_in(PathClass::Primary));
}

/// A coated, absorbing triplet: a lossy single-layer AR coating on every
/// lens surface (exercises `ResolvedCoating::rt` with `A > 0`) and bulk
/// absorption in the glass (exercises the internal transmittance).
#[test]
fn coated_absorbing_triplet_primary_matches_sequential() {
    use std::sync::Arc;
    let mut s = mklight_core::samples::cooke_triplet();
    s.coatings.insert(
        "lossy_ar".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.38 },
                thickness_um: 0.0,
                quarter_waves: Some(1.0),
                extinction: 0.003,
            }],
            mirror_substrate: None,
        },
    );
    let n = s.surfaces.len();
    for surf in &mut s.surfaces[1..n - 1] {
        surf.coating = Some("lossy_ar".into());
    }
    let mut rs = resolve(s);
    for surf in &mut rs.surfaces {
        if surf.n_after.iter().all(|n| *n > 1.2) {
            surf.alpha_after = vec![0.02; surf.alpha_after.len()];
        }
    }
    let sys = mklight_stray::StraySystem::new(Arc::new(rs), &no_scatter_options()).unwrap();
    let beam = sys.system_field_source(1, 0.1, 1.0).unwrap();
    check_unobstructed("coated absorbing triplet field 1", &sys, &beam, 60_000, 41);
    // The coating and the absorption change the answer: the comparison is
    // sensitive to them (the AR coating gains more than its loss and the
    // bulk absorption cost).
    let bare = stray_system(
        mklight_core::samples::cooke_triplet(),
        &no_scatter_options(),
    );
    let ctx = JobContext::new();
    let a = primary_reference(&sys, &beam.spec, 200, &ctx).unwrap();
    let b = primary_reference(&bare, &beam.spec, 200, &ctx).unwrap();
    assert!(
        (a.flux / b.flux - 1.0).abs() > 0.05,
        "{} vs {}",
        a.flux,
        b.flux
    );
}

/// A mirror system whose designed beam is obstructed: the back of the
/// Newtonian's fold flat and its mount sit in the incoming beam. The
/// Monte-Carlo `Primary` matches the obstruction-checked reference, while
/// the purely sequential flux (which cannot see the flat's back) is far off;
/// the system warns about it per field.
#[test]
fn newtonian_primary_matches_with_obstruction() {
    let sys = stray_system(
        mklight_core::samples::newtonian_telescope(),
        &no_scatter_options(),
    );
    for (k, o) in sys.obstruction.iter().enumerate() {
        let o = o.unwrap();
        assert!(o > 0.05 && o < 0.3, "field {k}: obstruction {o}");
        assert!(
            sys.warnings
                .iter()
                .any(|w| w.starts_with(&format!("field {k}:")) && w.contains("obstructed")),
            "{:?}",
            sys.warnings
        );
    }
    for (k, seed) in [(1usize, 51u64), (2, 52)] {
        let beam = sys.system_field_source(k, 0.1, 1.0).unwrap();
        let (reference, mc, se) =
            check_primary(&format!("Newtonian field {k}"), &sys, &beam, 60_000, seed);
        let zs = z(mc, reference.sequential_flux, se);
        println!(
            "Newtonian field {k}: obstruction {:.3} %, MC vs purely sequential z = {zs:+.1}",
            100.0 * reference.obstruction()
        );
        assert!(reference.obstructed > 0);
        assert!(zs < -10.0, "z = {zs}");
    }
}

/// Collimated-field systems and fields used by the source-independence
/// checks. With `real_aiming` the systems aim real rays at the stop, so the
/// traced stop height equals the stop radius.
fn collimated_cases(real_aiming: bool) -> Vec<(&'static str, StraySystem, Vec<usize>)> {
    use mklight_core::samples::*;
    let opts = no_scatter_options();
    let sys = |mut s: OpticalSystem| {
        if real_aiming {
            s.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        }
        stray_system(s, &opts)
    };
    vec![
        ("singlet", sys(singlet()), vec![0, 2]),
        ("achromat", sys(achromatic_doublet()), vec![0, 2]),
        ("Cooke triplet", sys(cooke_triplet()), vec![0, 2]),
        // Vignetting factors compress the sequential pupil, while the scene
        // clips the full beam at the apertures; the two agree only
        // approximately, so the spot comparison uses the unvignetted lens.
        ("double Gauss", sys(unvignetted_double_gauss()), vec![0, 1]),
        ("Petzval", sys(petzval_asphere()), vec![0, 2]),
    ]
}

/// The beam is limited by the stop, not by the source: the reference flux
/// in W (fixed irradiance) is the same for margins 0.1 and 0.5, within the
/// quadrature error. (Without a margin the pupil-plane disk underfills the
/// stop off axis and loses flux, which this check would see.)
#[test]
fn reference_flux_does_not_depend_on_the_margin() {
    let ctx = JobContext::new();
    for (name, sys, fields) in collimated_cases(false) {
        for k in fields {
            let flux = |margin: f64| {
                let beam = sys.system_field_source(k, margin, 1.0).unwrap();
                primary_reference(&sys, &beam.spec, 300, &ctx).unwrap().flux
            };
            let (a, b, c) = (flux(0.1), flux(0.5), flux(0.0));
            println!(
                "{name} field {k}: margin 0.1 {a:.6} W, 0.5 {b:.6} W (rel {:+.1e}), 0 {c:.6} W (rel {:+.1e})",
                b / a - 1.0,
                c / a - 1.0
            );
            assert!((b / a - 1.0).abs() < 2e-3, "{name} field {k}: {a} vs {b}");
        }
    }
}

/// The reference's geometric spot (equal weight per ray, uniform over the
/// overfilled pupil-plane disk, clipped by the scene's apertures) matches
/// the sequential tracer's own spot over a dense real-aimed pupil grid,
/// which shares no source code with it. A mis-aimed or undersized source
/// shifts the centroid or shrinks the spot.
///
/// Real aiming makes the stop, not the paraxial entrance pupil, define the
/// sequential beam, as the stop plate does in the scene. (With paraxial
/// aiming the off-axis beams cross the stop slightly outside its paraxial
/// radius; the scene's stop is frozen at the largest traced height so it
/// does not vignette them, and on axis it then passes a slightly wider beam
/// than the paraxial pupil: `docs/STRAY.md`.)
#[test]
fn reference_spot_matches_the_sequential_tracer() {
    let ctx = JobContext::new();
    for (name, sys, fields) in collimated_cases(true) {
        let t = sys.tracer();
        let opts = t.options();
        let nw = t.num_wavelengths();
        let pts = mklight_trace::PupilSampling::Grid { n: 121 }.points();
        for k in fields {
            let f = mklight_trace::FieldPoint::system(sys.resolved().system(), k);
            let pf = t
                .prepare_field(mklight_trace::FieldPoint::normalized(f.hx, f.hy))
                .unwrap();
            let (mut n, mut sx, mut sy, mut sr) = (0usize, 0.0, 0.0, 0.0);
            for wl in 0..nw {
                for p in &pts {
                    let r = t.trace(&pf, wl, p[0], p[1], &opts);
                    assert_eq!(r.status, mklight_trace::TraceStatus::Ok, "{name} {k}");
                    let (x, y) = (r.ray.pos.x, r.ray.pos.y);
                    n += 1;
                    sx += x;
                    sy += y;
                    sr += x * x + y * y;
                }
            }
            let (cx, cy) = (sx / n as f64, sy / n as f64);
            let rms = (sr / n as f64 - cx * cx - cy * cy).max(0.0).sqrt();
            let beam = sys.system_field_source(k, 0.1, 1.0).unwrap();
            let r = primary_reference(&sys, &beam.spec, 300, &ctx).unwrap();
            let [gx, gy] = r.geometric_centroid;
            let g = r.geometric_rms_radius;
            let dc = (gx - cx).hypot(gy - cy);
            println!(
                "{name} field {k}: centroid ({gx:.6}, {gy:.6}) vs ({cx:.6}, {cy:.6}) mm (Δ = {:.2e} RMS), RMS {:.3} vs {:.3} µm (rel {:+.2e})",
                dc / rms,
                1e3 * g,
                1e3 * rms,
                g / rms - 1.0
            );
            // The remaining differences are pupil aberration: the beam is
            // uniform on the entrance-pupil plane, the tracer's grid uniform
            // on the stop (Cooke triplet: 0.8 % on axis, 1.5 % at 20°).
            assert!(dc < 0.02 * rms, "{name} field {k}: centroid off by {dc} mm");
            assert!(
                (g / rms - 1.0).abs() < 0.02,
                "{name} field {k}: RMS {g} vs {rms}"
            );
        }
    }
}
