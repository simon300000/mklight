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

/// Basin hopping runs each parallel local solve on the problem's instance
/// for one solve (`Problem::for_solve`) and everything else on the problem
/// itself; the result is that of the same function without such instances.
#[test]
fn hammer_local_solves_run_on_their_own_problem_instances() {
    use mklight_core::MkResult;
    use mklight_optim::Problem;
    use mklight_optim::problem::{BatchEvaluation, CandidateBatch};
    /// A problem counting the batches it evaluates itself and the
    /// instances it hands out (when it does).
    #[derive(Clone, Copy)]
    struct Counted<'a> {
        inner: &'a FnProblem,
        hands_out: bool,
        instance: bool,
        batches: &'a AtomicUsize,
        instances: &'a AtomicUsize,
    }
    impl Problem for Counted<'_> {
        fn variables(&self) -> &[Variable] {
            self.inner.variables()
        }
        fn num_residuals(&self) -> usize {
            self.inner.num_residuals()
        }
        fn weights(&self) -> &[f64] {
            self.inner.weights()
        }
        fn evaluate_batch(
            &self,
            x: &CandidateBatch,
            ctx: &JobContext,
        ) -> MkResult<BatchEvaluation> {
            if !self.instance {
                self.batches.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.evaluate_batch(x, ctx)
        }
        fn for_solve(&self) -> Option<Box<dyn Problem + '_>> {
            self.hands_out.then(|| {
                self.instances.fetch_add(1, Ordering::SeqCst);
                Box::new(Counted {
                    instance: true,
                    ..*self
                }) as Box<dyn Problem>
            })
        }
    }
    let mut inner = freudenstein_roth();
    for v in inner.variables_mut().iter_mut() {
        v.lower = Some(-20.0);
        v.upper = Some(20.0);
    }
    let s = HammerSettings {
        seed: 7,
        rounds: 3,
        local_per_round: 2,
        target_merit: -1.0,
        ..Default::default()
    };
    let plain = hammer(&inner, &[0.5, -2.0], &s, &Monitor::default()).unwrap();
    let run = |hands_out: bool| {
        let (batches, instances) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let p = Counted {
            inner: &inner,
            hands_out,
            instance: false,
            batches: &batches,
            instances: &instances,
        };
        let r = hammer(&p, &[0.5, -2.0], &s, &Monitor::default()).unwrap();
        assert_eq!(r.x, plain.x);
        assert_eq!(r.merit, plain.merit);
        assert_eq!(r.evaluations, plain.evaluations);
        (batches.into_inner(), instances.into_inner())
    };
    let (own_batches, instances) = run(true);
    // Two local solves in each of the three rounds, each on an instance.
    assert_eq!(instances, 6);
    // The problem itself evaluated the start solve and the screens only.
    let (all_batches, none) = run(false);
    assert_eq!(none, 0);
    assert!(own_batches < all_batches, "{own_batches} {all_batches}");
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

#[test]
fn hammer_rejects_a_step_that_is_not_positive() {
    let p = rastrigin(2, 5.12);
    for step in [-0.5, 0.0, f64::NAN] {
        let s = HammerSettings {
            step,
            ..Default::default()
        };
        let r = hammer(&p, &[1.0; 2], &s, &Monitor::default());
        assert!(
            matches!(r, Err(mklight_core::MkError::InvalidParam(_))),
            "{step}: {r:?}"
        );
    }
}

/// A target reached by the last allowed iteration is reported as reached.
#[test]
fn target_reached_in_the_last_iteration() {
    let p = rastrigin(3, 5.12);
    let unreachable = f64::NEG_INFINITY;
    // The first iteration count whose last iteration improves the best merit,
    // and a target between the merits before and after it.
    let pick = |run: &dyn Fn(usize, f64) -> mklight_optim::OptimResult| {
        (1..50)
            .find_map(|k| {
                let (before, after) = (run(k - 1, unreachable).merit, run(k, unreachable).merit);
                (after < before).then_some((k, 0.5 * (before + after)))
            })
            .unwrap()
    };
    let de = |generations: usize, target: f64| {
        let s = DeSettings {
            seed: 5,
            max_generations: generations,
            target_merit: target,
            ..Default::default()
        };
        differential_evolution(&p, None, &s, &Monitor::default()).unwrap()
    };
    let (k, target) = pick(&de);
    let r = de(k, target);
    assert_eq!(
        (r.termination, r.iterations),
        (Termination::TargetReached, k)
    );

    let nm = |iterations: usize, target: f64| {
        let s = NelderMeadSettings {
            max_iterations: iterations,
            target_merit: target,
            ..Default::default()
        };
        nelder_mead(&p, &[1.3, -0.7, 2.1], &s, &Monitor::default()).unwrap()
    };
    let (k, target) = pick(&nm);
    let r = nm(k, target);
    assert_eq!(
        (r.termination, r.iterations),
        (Termination::TargetReached, k)
    );
}
