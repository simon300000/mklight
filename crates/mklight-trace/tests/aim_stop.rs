//! The conditioned aim stop (CONVENTIONS §7): the OPD tangent behind it
//! against central differences, the derived tolerances' range, and the
//! propagated error of the looser stop against the requested-tolerance
//! reference on every sample with real aiming.

mod common;

use common::*;
use mklight_core::{RayAiming, samples};
use mklight_trace::{
    AimBudget, AimStop, PupilSampling, TraceOptions, TraceStatus, TracerSettings, trace_ray,
};

fn real(mut sys: mklight_core::OpticalSystem) -> mklight_core::OpticalSystem {
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    sys
}

#[test]
fn opd_tangent_matches_central_differences() {
    for sys in [samples::cooke_triplet(), samples::finite_relay()] {
        let t = tracer(real(sys));
        let plan = t.tangent_plan();
        let image = t.system().image_index();
        for (fi, field) in t.fields().iter().enumerate() {
            for wl in 0..t.num_wavelengths() {
                let reference = t.opd_reference(field, wl).unwrap();
                let g = t.generate(field, wl, 0.4, -0.55).unwrap();
                let (ray, tangent) = t.start_tangent(&field.source, g.aim).unwrap();
                let mut rec = Vec::new();
                let r = trace_ray(
                    t.system(),
                    &ray,
                    wl,
                    &TraceOptions::unclipped(),
                    Some(&mut rec),
                );
                assert!(r.status.is_ok());
                let ti = plan
                    .differentiate(t.system(), wl, &tangent, &rec, image)
                    .unwrap();
                let analytic = reference.opd_tangent(&r, &ti).unwrap();
                let opd = |a: [f64; 2]| {
                    let ray = t.start_ray(&field.source, a).unwrap();
                    let r = trace_ray(t.system(), &ray, wl, &TraceOptions::unclipped(), None);
                    reference.opd_waves(&r).unwrap()
                };
                // Step sweep: the central differences descend as h² and
                // rise again below the roundoff floor of the path sums.
                let scale = t.pupil().r_ep.max(1e-3);
                for k in 0..2 {
                    let sweep: Vec<f64> = (2..=6)
                        .map(|e| {
                            let h = 10f64.powi(-e) * scale;
                            let (mut ap, mut am) = (g.aim, g.aim);
                            ap[k] += h;
                            am[k] -= h;
                            (opd(ap) - opd(am)) / (2.0 * h)
                        })
                        .collect();
                    let best = sweep
                        .iter()
                        .map(|fd| (fd - analytic[k]).abs())
                        .fold(f64::INFINITY, f64::min);
                    // Richardson pair in the truncation-dominated regime
                    // (h, h/2 at 1e-2 of the pupil radius): the h² term
                    // cancels and the roundoff of the path sums (~1e-9
                    // waves) is divided by a large step.
                    let fd = |h: f64| {
                        let (mut ap, mut am) = (g.aim, g.aim);
                        ap[k] += h;
                        am[k] -= h;
                        (opd(ap) - opd(am)) / (2.0 * h)
                    };
                    let h = 1e-2 * scale;
                    let richardson = (4.0 * fd(0.5 * h) - fd(h)) / 3.0;
                    let size = analytic[k].abs().max(1e-3);
                    assert!(
                        best < 1e-4 * size && (richardson - analytic[k]).abs() < 1e-5 * size,
                        "field {fi} wl {wl} k {k}: analytic {:e}, best sweep error {best:e}, \
                         Richardson {richardson:e}",
                        analytic[k]
                    );
                }
            }
        }
    }
}

#[test]
fn derived_stops_lie_between_the_request_and_the_cap_and_keep_the_budget() {
    let budget = AimBudget::DESIGN;
    let mut derived_count = 0;
    for (name, sys) in samples::all() {
        let sys = real(sys);
        let strict = tracer(sys.clone());
        let loose = tracer_with(
            sys,
            TracerSettings {
                aim_stop: AimStop::Conditioned(budget),
                ..Default::default()
            },
        );
        let requested = strict.requested_tolerance();
        for (fi, (fs, fl)) in strict.fields().iter().zip(loose.fields()).enumerate() {
            for wl in 0..strict.num_wavelengths() {
                // Chief rays are aimed at the requested tolerance on both.
                assert_eq!(
                    fs.aims[wl].chief, fl.aims[wl].chief,
                    "{name} field {fi} wl {wl}"
                );
                let tol = loose.aim_tolerance_of(fl, wl);
                assert!(
                    tol >= requested && tol <= budget.max_relative,
                    "{name}: {tol:e}"
                );
                if fl.aims[wl].stop_tolerance.is_some() {
                    derived_count += 1;
                }
            }
        }
        // Every ray of a hexapolar grid: identical survival, positions on
        // every surface and direction within the allocation, OPD within
        // its allocation.
        let pupil = PupilSampling::Hexapolar { rings: 8 }.points();
        let opts = strict.options();
        let (mut worst_pos, mut worst_dir, mut worst_opd) = (0.0f64, 0.0f64, 0.0f64);
        for (fi, (fs, fl)) in strict.fields().iter().zip(loose.fields()).enumerate() {
            for wl in 0..strict.num_wavelengths() {
                let rs = strict.opd_reference(fs, wl).unwrap();
                let rl = loose.opd_reference(fl, wl).unwrap();
                for p in &pupil {
                    let (mut recs, mut recl) = (Vec::new(), Vec::new());
                    let a = strict.trace_with(fs, wl, p[0], p[1], &opts, Some(&mut recs));
                    let b = loose.trace_with(fl, wl, p[0], p[1], &opts, Some(&mut recl));
                    assert_eq!(
                        a.status.is_ok(),
                        b.status.is_ok(),
                        "{name} field {fi} wl {wl} pupil {p:?}: {:?} vs {:?}",
                        a.status,
                        b.status
                    );
                    if let (
                        TraceStatus::Failed { surface: sa, .. },
                        TraceStatus::Failed { surface: sb, .. },
                    ) = (a.status, b.status)
                    {
                        assert_eq!(sa, sb, "{name} field {fi} wl {wl} pupil {p:?}");
                    }
                    for (x, y) in recs.iter().zip(&recl) {
                        worst_pos = worst_pos.max((x.point - y.point).norm());
                        worst_dir = worst_dir.max((x.dir - y.dir).norm());
                    }
                    if let (Some(oa), Some(ob)) = (rs.opd_waves(&a), rl.opd_waves(&b)) {
                        worst_opd = worst_opd.max((oa - ob).abs());
                    }
                }
            }
        }
        assert!(
            worst_pos <= budget.position_mm
                && worst_dir <= budget.direction
                && worst_opd <= budget.opd_waves,
            "{name}: position {worst_pos:e} direction {worst_dir:e} opd {worst_opd:e}"
        );
    }
    assert!(derived_count > 0, "no field derived a stop");
}

#[test]
fn requested_policy_is_bitwise_the_default() {
    let sys = real(samples::double_gauss());
    let a = tracer(sys.clone());
    let b = tracer_with(
        sys,
        TracerSettings {
            aim_stop: AimStop::Requested,
            ..Default::default()
        },
    );
    assert_eq!(a.device_aim_tolerance(), 1e-10);
    let f = (a.field(2), b.field(2));
    for p in (PupilSampling::Hexapolar { rings: 4 }).points() {
        let x = a.generate(f.0, 0, p[0], p[1]).unwrap();
        let y = b.generate(f.1, 0, p[0], p[1]).unwrap();
        assert_eq!(format!("{x:?}"), format!("{y:?}"));
    }
}

#[test]
fn an_explicit_tolerance_is_honoured_exactly_by_the_conditioned_policy() {
    // Any tolerance but the model default is an explicit request: the
    // conditioned policy derives nothing and traces bitwise as requested.
    let mut sys = samples::double_gauss();
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-12 };
    assert_eq!(sys.ray_aiming.explicit_tolerance(), Some(1e-12));
    let conditioned = tracer_with(
        sys.clone(),
        TracerSettings {
            aim_stop: AimStop::Conditioned(AimBudget::DESIGN),
            ..Default::default()
        },
    );
    let requested = tracer(sys);
    assert_eq!(conditioned.device_aim_tolerance(), 1e-12);
    let f = (conditioned.field(2), requested.field(2));
    for w in 0..conditioned.num_wavelengths() {
        assert_eq!(conditioned.aim_tolerance_of(f.0, w), 1e-12);
    }
    for p in (PupilSampling::Hexapolar { rings: 4 }).points() {
        let x = conditioned.generate(f.0, 0, p[0], p[1]).unwrap();
        let y = requested.generate(f.1, 0, p[0], p[1]).unwrap();
        assert_eq!(format!("{x:?}"), format!("{y:?}"));
    }
    // The model default stays derivable.
    let derived = tracer_with(
        real(samples::double_gauss()),
        TracerSettings {
            aim_stop: AimStop::Conditioned(AimBudget::DESIGN),
            ..Default::default()
        },
    );
    assert!(derived.device_aim_tolerance() > RayAiming::DEFAULT_TOLERANCE);
}
