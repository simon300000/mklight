//! Every sample traces cleanly and is well focused; batches are deterministic
//! and cancellable; OPD is consistent with transverse aberration.

use crate::common;

use std::time::Instant;

use common::*;
use mklight_core::{JobContext, MkError, OpticalSystem, RayAiming, samples};
use mklight_trace::{
    Grid, PupilSampling, TraceOptions, TracerSettings, opd_grid, opd_grid_job, trace_grid,
    trace_grid_job,
};

fn polychromatic_rms(sys: &OpticalSystem) -> f64 {
    let t = tracer(sys.clone());
    let pts = PupilSampling::Hexapolar { rings: 6 }.points();
    let mut total = 0.0;
    for f in t.fields() {
        let mut xy = Vec::new();
        for w in 0..t.num_wavelengths() {
            for p in &pts {
                let r = t.trace(f, w, p[0], p[1], &TraceOptions::default());
                assert!(r.status.is_ok());
                xy.push((r.ray.pos.x, r.ray.pos.y));
            }
        }
        let n = xy.len() as f64;
        let cx = xy.iter().map(|p| p.0).sum::<f64>() / n;
        let cy = xy.iter().map(|p| p.1).sum::<f64>() / n;
        total += (xy
            .iter()
            .map(|p| (p.0 - cx).powi(2) + (p.1 - cy).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
    }
    total / t.fields().len() as f64
}

#[test]
fn all_samples_trace_every_field_and_wavelength() {
    for (name, sys) in samples::all() {
        for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            let t = tracer_with(
                sys.clone(),
                TracerSettings {
                    aiming: Some(aiming),
                    ..Default::default()
                },
            );
            assert!(
                t.warnings().iter().all(|w| !w.contains("did not converge")),
                "{name}: {:?}",
                t.warnings()
            );
            let pts = PupilSampling::Hexapolar { rings: 6 }.points();
            for (fi, f) in t.fields().iter().enumerate() {
                for w in 0..t.num_wavelengths() {
                    for p in &pts {
                        let r = t.trace(f, w, p[0], p[1], &TraceOptions::default());
                        assert!(
                            r.status.is_ok(),
                            "{name} {aiming:?} field {fi} wl {w} pupil {p:?}: {:?}",
                            r.status
                        );
                    }
                    assert!(t.opd(f, w, 0.0, 1.0, &TraceOptions::default()).is_some());
                }
            }
        }
    }
}

#[test]
fn samples_are_well_focused() {
    // Either the image sits at the paraxial focus, or it is at a local minimum
    // of the polychromatic RMS spot (within 1 % against ±0.5 % focus shifts).
    for (name, sys) in samples::all() {
        let t = tracer(sys.clone());
        let last = sys.surfaces.len() - 2;
        let current = sys.surfaces[last].thickness;
        let paraxial = t.paraxial().primary().image_distance;
        if (current - paraxial).abs() < 1e-6 {
            continue;
        }
        let rms0 = polychromatic_rms(&sys);
        for k in [-1.0, 1.0] {
            let mut s = sys.clone();
            s.surfaces[last].thickness = current + k * 0.005 * current.abs();
            let r = polychromatic_rms(&s);
            assert!(rms0 <= 1.01 * r, "{name}: rms {rms0} vs shifted {r}");
        }
    }
}

/// Smallest axial separation `t_i + sag_{i+1}(r) − sag_i(r)` between
/// surfaces `i` and `i + 1` for `r` in `[0, r_max]`, each surface continued
/// by a flat beyond its clear radius `ca` (how lens elements are closed).
fn min_separation(sys: &OpticalSystem, i: usize, ca: [f64; 2], r_max: f64) -> f64 {
    let (a, b) = (&sys.surfaces[i], &sys.surfaces[i + 1]);
    (0..=1000)
        .map(|k| {
            let r = r_max * k as f64 / 1000.0;
            let (ra, rb) = (r.min(ca[0]), r.min(ca[1]));
            let sag = |s: &mklight_core::Surface, r: f64| s.shape.sag(r * r).unwrap_or(f64::NAN);
            a.thickness + sag(b, rb) - sag(a, ra)
        })
        .fold(f64::INFINITY, |m, g| if g >= m { m } else { g })
}

/// Edge thickness of the element between surfaces `i` and `i + 1` at their
/// mechanical rims, as the stray-light scene builds them: the explicit
/// mechanical semi-diameter, else the clear one `ca` plus the scene's
/// default 1 mm margin (`SceneOptions::default_edge_margin`).
fn mechanical_edge(sys: &OpticalSystem, i: usize, ca: [f64; 2]) -> f64 {
    let rim = |k: usize| {
        let s = &sys.surfaces[i + k];
        let m = s.edge.mechanical_semi_diameter.unwrap_or(ca[k] + 1.0);
        s.shape.sag(m * m).unwrap_or(f64::NAN)
    };
    sys.surfaces[i].thickness + rim(1) - rim(0)
}

#[test]
fn sample_elements_are_physical_over_their_clear_apertures() {
    // At the automatic clear semi-diameters (the full field, with the
    // fields' vignetting factors): every glass element has an edge
    // thickness ≥ 0.5 mm out to the larger of its two clear apertures and
    // > 0.2 mm at its mechanical rims, and no two surfaces cross inside the
    // smaller clear aperture of an air space. Mirrors, coordinate breaks
    // and tilted surfaces are skipped.
    use mklight_core::{Interaction, MaterialRef};
    use mklight_trace::SemiDiameterSettings;
    for (name, sys) in samples::all() {
        let t = tracer(sys.clone());
        let sd = t.semi_diameters(&SemiDiameterSettings::default());
        let ca = |j: usize| sd.values[j].max(sd.traced[j]);
        let n = sys.surfaces.len();
        let plain = |j: usize| {
            let s = &sys.surfaces[j];
            s.interaction == Interaction::Refract
                && !s.is_coordinate_break()
                && s.placement.is_none_or(|p| p.is_identity())
        };
        let mut checked = 0;
        for i in 1..n - 2 {
            if !(plain(i) && plain(i + 1)) {
                continue;
            }
            let glass = sys.surfaces[i].material != MaterialRef::Air;
            let (c0, c1) = (ca(i), ca(i + 1));
            let (r_max, min) = if glass {
                (c0.max(c1), 0.5)
            } else {
                (c0.min(c1), 0.0)
            };
            let gap = min_separation(&sys, i, [c0, c1], r_max);
            assert!(
                gap > min,
                "{name}: surfaces {i}–{} ({}) separate by {gap:.4} mm within r = {r_max:.3} mm",
                i + 1,
                if glass { "glass" } else { "air" }
            );
            if glass {
                let edge = mechanical_edge(&sys, i, [c0, c1]);
                assert!(
                    edge > 0.2,
                    "{name}: element {i}–{} has a {edge:.4} mm edge at its mechanical rim",
                    i + 1
                );
            }
            checked += 1;
        }
        assert!(checked > 0 || name == "newtonian", "{name}");
    }
    // The mechanical check can fail: the Cooke triplet's L3 at the default
    // margin instead of its mount.
    let mut ck = samples::cooke_triplet();
    for j in [5, 6] {
        ck.surfaces[j].edge.mechanical_semi_diameter = None;
    }
    let sd = tracer(ck.clone()).semi_diameters(&SemiDiameterSettings::default());
    assert!(mechanical_edge(&ck, 5, [sd.values[5], sd.values[6]]) < 0.0);
    // The check can fail: the Double Gauss without its vignetting factors
    // needs clear apertures where L1 and L6 have negative edge thickness.
    let mut dg = samples::double_gauss();
    for f in &mut dg.fields.fields {
        f.vignetting = Default::default();
    }
    let sd = tracer(dg.clone()).semi_diameters(&SemiDiameterSettings::default());
    for i in [1, 10] {
        let ca = [sd.values[i], sd.values[i + 1]];
        assert!(
            min_separation(&dg, i, ca, ca[0].max(ca[1])) < -0.3,
            "L at {i}"
        );
    }
}

#[test]
fn sample_first_order_values() {
    let fo = |s: OpticalSystem| tracer(s).paraxial().primary().clone();
    let a = fo(samples::achromatic_doublet());
    assert!((a.efl - 100.0).abs() < 0.01 && (a.image_f_number - 5.0).abs() < 1e-12);
    let c = fo(samples::cooke_triplet());
    assert!((c.efl - 50.0).abs() < 0.1);
    let d = fo(samples::double_gauss());
    assert!((d.efl - 50.0).abs() < 0.1 && (d.image_f_number - 2.0).abs() < 1e-12);
    let n = fo(samples::newtonian_telescope());
    assert!((n.efl - 500.0).abs() < 1e-9 && (n.bfl - 100.0).abs() < 1e-9);
    let r = fo(samples::finite_relay());
    assert!((r.magnification + 1.0).abs() < 1e-6 && (r.object_na - 0.08).abs() < 1e-14);
}

#[test]
fn opd_is_consistent_with_transverse_aberration() {
    // ε_y = −(R / n) ∂W/∂Y on the reference sphere, with Y ≈ Py·r_XP and W in
    // mm. Checked on the axial singlet (pure spherical aberration).
    let t = tracer(samples::singlet());
    let w = t.system().primary_wavelength;
    let f = t.field(0);
    let reference = t.opd_reference(f, w).unwrap();
    let fo = t.paraxial().primary();
    let r = reference.s_exit_pupil.abs();
    let r_xp = 0.5 * fo.exit_pupil_diameter;
    let lam = fo.wavelength_um * 1e-3;
    for py in [0.2, 0.5, 0.8] {
        let h = 1e-4;
        let wp = t.opd(f, w, 0.0, py + h, &TraceOptions::default()).unwrap() * lam;
        let wm = t.opd(f, w, 0.0, py - h, &TraceOptions::default()).unwrap() * lam;
        let dw_dy = (wp - wm) / (2.0 * h * r_xp);
        let eps = t.trace(f, w, 0.0, py, &TraceOptions::default()).ray.pos.y;
        let predicted = -r * dw_dy;
        assert!(eps < 0.0, "undercorrected marginal rays cross the axis");
        assert!(
            (eps - predicted).abs() < 0.03 * eps.abs(),
            "Py {py}: ε {eps} vs −R ∂W/∂Y {predicted}"
        );
    }
}

#[test]
fn batch_is_deterministic_and_matches_single_traces() {
    let t = tracer(samples::cooke_triplet());
    let pupil = PupilSampling::Grid { n: 9 }.points();
    let wls = [2usize, 0];
    let grid = Grid {
        fields: t.fields(),
        wavelengths: &wls,
        pupil: &pupil,
    };
    let opts = TraceOptions::default();
    let a = trace_grid(&t, &grid, &opts);
    let b = trace_grid_job(&t, &grid, &opts, &JobContext::new()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 3 * 2 * pupil.len());
    for (i, s) in a.iter().enumerate() {
        let (f, wi, p) = grid.split(i);
        assert_eq!(
            (s.field as usize, s.wavelength as usize, s.pupil),
            (f, wls[wi], pupil[p])
        );
        let r = t.trace(t.field(f), wls[wi], pupil[p][0], pupil[p][1], &opts);
        assert_eq!(r.ray.pos, s.point);
        assert_eq!(r.opl, s.opl);
    }
    let o1 = opd_grid(&t, &grid, &opts);
    let o2 = opd_grid_job(&t, &grid, &opts, &JobContext::new()).unwrap();
    assert_eq!(o1, o2);
    let (f, wi, p) = grid.split(17);
    let single = t.opd(t.field(f), wls[wi], pupil[p][0], pupil[p][1], &opts);
    assert_eq!(o1[17], single);
}

#[test]
fn batch_job_cancels_and_reports_progress() {
    let t = tracer(samples::double_gauss());
    let pupil = PupilSampling::Grid { n: 64 }.points();
    let wls = [0usize, 1, 2];
    let grid = Grid {
        fields: t.fields(),
        wavelengths: &wls,
        pupil: &pupil,
    };
    let ctx = JobContext::new();
    trace_grid_job(&t, &grid, &TraceOptions::default(), &ctx).unwrap();
    assert!((ctx.progress.fraction() - 1.0).abs() < 1e-12);
    // An empty grid still finishes at 1.
    let empty: [[f64; 2]; 0] = [];
    let none = Grid {
        fields: t.fields(),
        wavelengths: &wls,
        pupil: &empty,
    };
    let ctx = JobContext::new();
    assert!(
        trace_grid_job(&t, &none, &TraceOptions::default(), &ctx)
            .unwrap()
            .is_empty()
    );
    assert_eq!(ctx.progress.fraction(), 1.0);
    let ctx = JobContext::new();
    assert!(
        opd_grid_job(&t, &none, &TraceOptions::default(), &ctx)
            .unwrap()
            .is_empty()
    );
    assert_eq!(ctx.progress.fraction(), 1.0);
    // OPD jobs reach 1 and cancel before any reference is computed.
    let ctx = JobContext::new();
    opd_grid_job(&t, &grid, &TraceOptions::default(), &ctx).unwrap();
    assert!((ctx.progress.fraction() - 1.0).abs() < 1e-12);
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        opd_grid_job(&t, &grid, &TraceOptions::default(), &ctx).unwrap_err(),
        MkError::Cancelled
    );
    assert_eq!(ctx.progress.fraction(), 0.0);
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        trace_grid_job(&t, &grid, &TraceOptions::default(), &ctx).unwrap_err(),
        MkError::Cancelled
    );
}

#[test]
#[ignore = "benchmark: cargo test -p mklight-trace --release -- --ignored --nocapture"]
fn throughput_benchmark() {
    let t = tracer(samples::double_gauss());
    let pupil = PupilSampling::Grid { n: 300 }.points();
    let wls = [0usize, 1, 2];
    let grid = Grid {
        fields: t.fields(),
        wavelengths: &wls,
        pupil: &pupil,
    };
    let opts = TraceOptions::default();
    let _ = trace_grid(&t, &grid, &opts); // warm-up
    let t0 = Instant::now();
    let out = trace_grid(&t, &grid, &opts);
    let dt = t0.elapsed().as_secs_f64();
    let surfaces = t.system().num_surfaces() - 1;
    let n = out.len() as f64;
    println!(
        "{} rays × {} surfaces in {:.3} s: {:.3e} rays/s, {:.3e} ray-surfaces/s (parallel, aim off)",
        out.len(),
        surfaces,
        dt,
        n / dt,
        n * surfaces as f64 / dt
    );
    // Single-thread core loop on pre-generated rays.
    let rays: Vec<_> = pupil
        .iter()
        .map(|p| t.generate(t.field(1), 1, p[0], p[1]).unwrap().ray)
        .collect();
    let t0 = Instant::now();
    let mut ok = 0usize;
    for r in &rays {
        ok += mklight_trace::trace_ray(t.system(), r, 1, &opts, None)
            .status
            .is_ok() as usize;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "single thread: {:.3e} ray-surfaces/s ({ok} ok)",
        rays.len() as f64 * surfaces as f64 / dt
    );
}
