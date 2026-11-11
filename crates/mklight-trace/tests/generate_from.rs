//! `SequentialTracer::generate_many_from` (started real-aim solves for
//! hybrid aiming): without starts it is bitwise `generate_many`; from good
//! starts it converges to the same rays within the stop tolerance in fewer
//! iterations; from bad starts it falls back to the reference schedule and
//! fails exactly where the reference fails.

use crate::common;

use common::*;
use mklight_core::{RayAiming, samples};
use mklight_trace::{AimStats, PupilSampling, RayRequest};

fn same<T: std::fmt::Debug>(a: &T, b: &T, what: &str) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}");
}

fn points() -> Vec<[f64; 2]> {
    let mut p = PupilSampling::Hexapolar { rings: 4 }.points();
    p.extend([[1.0, 0.0], [0.0, -1.0], [0.9, 0.9], [-1.4, 0.2], [0.0, 2.5]]);
    p
}

#[test]
fn started_solves_match_the_reference_within_tolerance() {
    for (name, mut sys) in samples::all() {
        sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = tracer(sys);
        let r_stop = t.pupil().r_stop;
        for f in t.fields() {
            for wl in 0..t.num_wavelengths() {
                let reqs: Vec<RayRequest> = points().into_iter().map(|p| (f, wl, p)).collect();
                let reference = t.generate_many(&reqs);
                // No starts: bitwise the reference.
                let none = vec![None; reqs.len()];
                let plain = t.generate_many_from(&reqs, &none, &mut AimStats::default());
                same(&reference, &plain, &format!("{name} no starts"));
                // Starts at the reference's own aims (perturbed), with its
                // stop Jacobian: converged rays agree within the tolerance.
                let starts: Vec<_> = reference
                    .iter()
                    .zip(&reqs)
                    .map(|(g, r)| {
                        let g = g.as_ref().ok()?;
                        let j = f.aims[r.1].jacobian?;
                        Some(([g.aim[0] * (1.0 + 1e-6), g.aim[1] + 1e-6], j))
                    })
                    .collect();
                let mut s_ref = AimStats::default();
                let _ = t.generate_many_stats(&reqs, &mut s_ref);
                let from = t.generate_many_from(&reqs, &starts, &mut AimStats::default());
                for ((a, b), r) in reference.iter().zip(&from).zip(&reqs) {
                    match (a, b) {
                        (Ok(a), Ok(b)) => {
                            let d = (a.ray.pos - b.ray.pos).norm();
                            assert!(d < 1e-6 * r_stop.max(1.0), "{name} {r:?}: {d}");
                            assert!(b.report.converged);
                        }
                        (Err(_), Err(_)) => {}
                        _ => panic!(
                            "{name} field {:?} pupil {:?}: outcome differs",
                            f.point, r.2
                        ),
                    }
                }
                // Starts at the converged aims themselves need no iteration
                // the reference did not need.
                let exact: Vec<_> = reference
                    .iter()
                    .zip(&reqs)
                    .map(|(g, r)| Some((g.as_ref().ok()?.aim, f.aims[r.1].jacobian?)))
                    .collect();
                let mut s_exact = AimStats::default();
                let _ = t.generate_many_from(&reqs, &exact, &mut s_exact);
                assert!(
                    s_exact.iterations <= s_ref.iterations,
                    "{name}: {} > {}",
                    s_exact.iterations,
                    s_ref.iterations
                );
                // Starts far off: the reference schedule takes over.
                let bad: Vec<_> = starts
                    .iter()
                    .map(|s| s.map(|(a, j)| ([a[0] + 50.0 * r_stop, a[1] - 50.0 * r_stop], j)))
                    .collect();
                let fallback = t.generate_many_from(&reqs, &bad, &mut AimStats::default());
                for ((a, b), r) in reference.iter().zip(&fallback).zip(&reqs) {
                    assert_eq!(a.is_ok(), b.is_ok(), "{name} {r:?}");
                    if let (Ok(a), Ok(b)) = (a, b) {
                        assert!((a.ray.pos - b.ray.pos).norm() < 1e-6 * r_stop.max(1.0));
                    }
                }
            }
        }
    }
}
