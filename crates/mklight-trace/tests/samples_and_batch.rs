//! Every sample traces cleanly and is well focused; batches are deterministic
//! and cancellable; OPD is consistent with transverse aberration.

mod common;

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
