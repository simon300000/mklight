//! Global and derivative-free methods.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use mklight_core::JobContext;
use mklight_optim::global::cmaes::{CmaEsSettings, cma_es};
use mklight_optim::global::de::{DeSettings, DeStrategy, differential_evolution};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::global::nelder_mead::{NelderMeadSettings, nelder_mead};
use mklight_optim::{Constraint, EvalStatus, FnProblem, Monitor, Termination, Variable};

#[test]
fn de_rastrigin_reaches_global_minimum() {
    for (dim, strategy) in [
        (2, DeStrategy::Rand1Bin),
        (4, DeStrategy::Rand1Bin),
        (4, DeStrategy::CurrentToBest1Bin),
    ] {
        let p = rastrigin(dim, 5.12);
        let s = DeSettings {
            strategy,
            seed: 11,
            max_generations: 3000,
            target_merit: 1e-12,
            ..Default::default()
        };
        let r = differential_evolution(&p, None, &s, &Monitor::default()).unwrap();
        assert!(
            r.merit < 1e-10,
            "dim {dim} {strategy:?}: merit {} ({:?})",
            r.merit,
            r.termination
        );
        assert!(r.x.iter().all(|v| v.abs() < 1e-5), "{:?}", r.x);
    }
}

#[test]
fn de_ackley_reaches_global_minimum() {
    let p = ackley(3, 32.768);
    let s = DeSettings {
        seed: 3,
        max_generations: 3000,
        target_merit: 1e-16,
        ..Default::default()
    };
    let r = differential_evolution(&p, None, &s, &Monitor::default()).unwrap();
    assert!(r.merit < 1e-12, "merit {}", r.merit);
    assert!(r.x.iter().all(|v| v.abs() < 1e-6), "{:?}", r.x);
}

#[test]
fn de_is_deterministic_per_seed() {
    let p = rastrigin(3, 5.12);
    let s = DeSettings {
        seed: 42,
        max_generations: 40,
        ..Default::default()
    };
    let a = differential_evolution(&p, None, &s, &Monitor::default()).unwrap();
    let b = differential_evolution(&p, None, &s, &Monitor::default()).unwrap();
    assert_eq!(a.x, b.x);
    assert_eq!(a.history, b.history);
    let c = differential_evolution(&p, None, &DeSettings { seed: 43, ..s }, &Monitor::default())
        .unwrap();
    assert_ne!(a.x, c.x);
}

#[test]
fn de_respects_constraints_and_failures() {
    // min (x−2)² + (y−2)² s.t. x + y ≤ 2 → (1, 1); evaluations with x < −1 fail.
    let v = vec![
        Variable::bounded("x", -3.0, 3.0),
        Variable::bounded("y", -3.0, 3.0),
    ];
    let p = FnProblem::new(v, 2, |x, r, c| {
        r[0] = x[0] - 2.0;
        r[1] = x[1] - 2.0;
        c[0] = x[0] + x[1];
        if x[0] < -1.0 {
            EvalStatus::Failed
        } else {
            EvalStatus::Ok
        }
    })
    .with_constraints(vec![Constraint::at_most("sum", 2.0)]);
    let s = DeSettings {
        seed: 5,
        max_generations: 400,
        ..Default::default()
    };
    let r = differential_evolution(&p, None, &s, &Monitor::default()).unwrap();
    assert!(r.violation <= 1e-8, "{r:?}");
    assert!(
        (r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3,
        "{:?}",
        r.x
    );
}

#[test]
fn nelder_mead_rosenbrock_and_bounds() {
    let p = rosenbrock();
    for speculative in [false, true] {
        let s = NelderMeadSettings {
            speculative,
            ..Default::default()
        };
        let r = nelder_mead(&p, &[-1.2, 1.0], &s, &Monitor::default()).unwrap();
        assert!(
            r.merit < 1e-14,
            "spec {speculative}: {} {:?}",
            r.merit,
            r.termination
        );
        assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
    }
    // Active bound x₀ ≤ 0.5 → (0.5, 0.25).
    let mut p = rosenbrock();
    p.variables_mut()[0] = Variable::new("x0").with_upper(0.5);
    let r = nelder_mead(
        &p,
        &[-1.2, 1.0],
        &NelderMeadSettings::default(),
        &Monitor::default(),
    )
    .unwrap();
    assert!((r.x[0] - 0.5).abs() < 1e-7, "{:?}", r.x);
    assert!((r.x[1] - 0.25).abs() < 1e-6, "{:?}", r.x);
    assert!((r.merit - 0.25).abs() < 1e-10);
}

#[test]
fn cmaes_rosenbrock_and_rotated_ellipsoid() {
    let p = rosenbrock();
    let s = CmaEsSettings {
        seed: 1,
        sigma0: 0.5,
        target_merit: 1e-20,
        ..Default::default()
    };
    let r = cma_es(&p, Some(&[-1.2, 1.0]), &s, &Monitor::default()).unwrap();
    assert!(r.merit < 1e-18, "{} {:?}", r.merit, r.termination);
    assert!((r.x[0] - 1.0).abs() < 1e-8);

    // Rotated, ill-conditioned (1e6) ellipsoid in 6-D: Σ 10^(6i/(n−1)) (Q x)ᵢ².
    let n = 6;
    let p = FnProblem::residuals(vars(n), n, move |x, r| {
        for i in 0..n {
            // A fixed orthogonal-ish coupling: Householder reflection about (1,…,1).
            let s: f64 = x.iter().sum::<f64>() * 2.0 / n as f64;
            let qi = x[i] - s;
            r[i] = 10f64.powf(3.0 * i as f64 / (n - 1) as f64) * (qi - 1.0);
        }
    });
    let s = CmaEsSettings {
        seed: 2,
        target_merit: 1e-16,
        ..Default::default()
    };
    let r = cma_es(&p, Some(&[0.0; 6]), &s, &Monitor::default()).unwrap();
    assert!(r.merit < 1e-14, "{} {:?}", r.merit, r.termination);
}

#[test]
fn cmaes_rastrigin_with_ipop_restarts() {
    let p = rastrigin(3, 5.12);
    let s = CmaEsSettings {
        seed: 9,
        sigma0: 2.0,
        restarts: 6,
        target_merit: 1e-12,
        ..Default::default()
    };
    let r = cma_es(&p, None, &s, &Monitor::default()).unwrap();
    assert!(r.merit < 1e-10, "{} {:?}", r.merit, r.termination);
}

#[test]
fn hammer_escapes_freudenstein_roth_local_minimum() {
    // Plain LM from the standard start stops at the local minimum (≈ 48.98);
    // the hammer finds the global zero at (5, 4).
    let mut p = freudenstein_roth();
    for v in p.variables_mut().iter_mut() {
        v.lower = Some(-20.0);
        v.upper = Some(20.0);
    }
    let plain =
        mklight_optim::least_squares(&p, &[0.5, -2.0], &Default::default(), &Monitor::default())
            .unwrap();
    assert!((plain.merit - FR_LOCAL_MIN).abs() < 1e-8, "{}", plain.merit);
    let s = HammerSettings {
        seed: 7,
        target_merit: 1e-20,
        ..Default::default()
    };
    let r = hammer(&p, &[0.5, -2.0], &s, &Monitor::default()).unwrap();
    assert_eq!(r.termination, Termination::TargetReached);
    assert!(r.merit < 1e-20);
    assert!(
        (r.x[0] - 5.0).abs() < 1e-8 && (r.x[1] - 4.0).abs() < 1e-8,
        "{:?}",
        r.x
    );
    // Same seed → same result.
    let r2 = hammer(&p, &[0.5, -2.0], &s, &Monitor::default()).unwrap();
    assert_eq!(r.x, r2.x);
    assert_eq!(r.evaluations, r2.evaluations);
}

#[test]
fn global_methods_honour_cancellation() {
    let p = rastrigin(4, 5.12);
    let make = || {
        let ctx = JobContext::new();
        let token = ctx.cancel.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let mon = Monitor::new(ctx).with_callback(move |_| {
            if c2.fetch_add(1, Ordering::SeqCst) + 1 == 5 {
                token.cancel();
            }
        });
        (mon, calls)
    };
    let (mon, calls) = make();
    let r = differential_evolution(&p, None, &DeSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert!(mon.best.get().is_some() && r.merit.is_finite());

    let (mon, calls) = make();
    let r = cma_es(&p, None, &CmaEsSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 5);

    let (mon, calls) = make();
    let r = nelder_mead(&p, &[1.0; 4], &NelderMeadSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 5);

    let (mon, calls) = make();
    let r = hammer(&p, &[1.0; 4], &HammerSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 5);
}
