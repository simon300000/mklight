//! Local least squares on standard problems.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use mklight_core::JobContext;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::{
    Constraint, EvalStatus, FdScheme, FnProblem, JacobianSettings, Monitor, Termination, Variable,
};

fn methods() -> Vec<(&'static str, LsqSettings)> {
    let geo = LsqSettings {
        geodesic_acceleration: true,
        ..Default::default()
    };
    let central = LsqSettings {
        jacobian: JacobianSettings {
            scheme: FdScheme::Central,
            use_analytic: false,
            relative_step: None,
        },
        ..Default::default()
    };
    vec![
        ("lm", LsqSettings::default()),
        ("lm+geodesic", geo),
        ("lm-central-fd", central),
        ("dogleg", LsqSettings::dogleg()),
    ]
}

#[test]
fn rosenbrock_all_methods() {
    let p = rosenbrock();
    for (name, s) in methods() {
        let r = least_squares(&p, &[-1.2, 1.0], &s, &Monitor::default()).unwrap();
        assert!(r.termination.is_converged(), "{name}: {:?}", r.termination);
        assert!(r.merit < 1e-20, "{name}: merit {}", r.merit);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-9 && (r.x[1] - 1.0).abs() < 1e-9,
            "{name}"
        );
        assert!(r.iterations < 100, "{name}: {} iterations", r.iterations);
        // History is monotone in merit for an unconstrained trust-region method.
        for w in r.history.windows(2) {
            assert!(w[1].merit <= w[0].merit, "{name}");
        }
    }
}

#[test]
fn powell_singular_converges() {
    let p = powell_singular();
    for (name, mut s) in methods() {
        s.max_iterations = 300;
        let r = least_squares(&p, &[3.0, -1.0, 0.0, 1.0], &s, &Monitor::default()).unwrap();
        assert!(
            r.merit < 1e-30,
            "{name}: merit {} ({:?})",
            r.merit,
            r.termination
        );
        assert!(r.x.iter().all(|v| v.abs() < 1e-6), "{name}: {:?}", r.x);
    }
}

#[test]
fn freudenstein_roth_minima() {
    let p = freudenstein_roth();
    // Standard start: converges to the known local minimum (or the global one).
    let r = least_squares(
        &p,
        &[0.5, -2.0],
        &LsqSettings::default(),
        &Monitor::default(),
    )
    .unwrap();
    assert!(
        (r.merit - FR_LOCAL_MIN).abs() < 1e-8 || r.merit < 1e-20,
        "merit {}",
        r.merit
    );
    // Near the global basin: the exact zero-residual solution (5, 4).
    let r = least_squares(
        &p,
        &[6.0, 3.0],
        &LsqSettings::default(),
        &Monitor::default(),
    )
    .unwrap();
    assert!(r.merit < 1e-20, "merit {}", r.merit);
    assert!((r.x[0] - 5.0).abs() < 1e-9 && (r.x[1] - 4.0).abs() < 1e-9);
}

#[test]
fn bounded_rosenbrock_active_bound() {
    // x₀ ≤ 0.5: minimum on the bound at (0.5, 0.25) with merit 0.25.
    let mut p = rosenbrock();
    p.variables_mut()[0] = Variable::new("x0").with_upper(0.5);
    for (name, s) in methods() {
        let r = least_squares(&p, &[-1.2, 1.0], &s, &Monitor::default()).unwrap();
        assert!(r.termination.is_converged(), "{name}: {:?}", r.termination);
        assert_eq!(r.x[0], 0.5, "{name}");
        assert!((r.x[1] - 0.25).abs() < 1e-9, "{name}: {:?}", r.x);
        assert!((r.merit - 0.25).abs() < 1e-12, "{name}");
    }
}

#[test]
fn bounded_linear_problem_two_active_bounds() {
    // r = (x₀ − 3, x₁ + 1, x₂ − 0.2), 0 ≤ x ≤ 1 → (1, 0, 0.2).
    let v = (0..3)
        .map(|i| Variable::bounded(format!("x{i}"), 0.0, 1.0))
        .collect();
    let p = FnProblem::residuals(v, 3, |x, r| {
        r[0] = x[0] - 3.0;
        r[1] = x[1] + 1.0;
        r[2] = x[2] - 0.2;
    });
    // Evaluations are never outside the bounds.
    for (name, s) in methods() {
        let r = least_squares(&p, &[0.5, 0.5, 0.5], &s, &Monitor::default()).unwrap();
        assert_eq!(r.x[0], 1.0, "{name}");
        assert_eq!(r.x[1], 0.0, "{name}");
        assert!((r.x[2] - 0.2).abs() < 1e-12, "{name}");
        assert!((r.merit - 5.0).abs() < 1e-12);
    }
}

#[test]
fn evaluations_stay_inside_bounds() {
    let v = vec![
        Variable::bounded("a", 0.1, 2.0),
        Variable::bounded("b", -1.0, 1.0),
    ];
    let p = FnProblem::new(v, 2, |x, r, _| {
        assert!((0.1..=2.0).contains(&x[0]) && (-1.0..=1.0).contains(&x[1]));
        r[0] = 10.0 * (x[1] - x[0].ln());
        r[1] = x[0] - 3.0;
        mklight_optim::EvalStatus::Ok
    });
    for (_, s) in methods() {
        let r = least_squares(&p, &[1.0, 0.0], &s, &Monitor::default()).unwrap();
        assert_eq!(r.x[0], 2.0);
        assert!((r.x[1] - 1.0).abs() < 1e-12 || (r.x[1] - 2f64.ln()).abs() < 1e-8);
    }
}

#[test]
fn constrained_kkt_point() {
    let p = constrained_kkt();
    let xs = kkt_solution();
    for (name, s) in methods() {
        let r = least_squares(&p, &[2.0, 2.0], &s, &Monitor::default()).unwrap();
        assert!(r.termination.is_converged(), "{name}: {:?}", r.termination);
        assert!((r.x[0] - xs[0]).abs() < 1e-7, "{name}: {:?} vs {xs:?}", r.x);
        assert!((r.x[1] - xs[1]).abs() < 1e-7, "{name}");
        assert!(r.violation < 1e-8, "{name}");
        // KKT: ∇Φ + λ₁∇c₁ + λ₂∇c₂ = 0 with λ₂ ≥ 0 (inequality active).
        let l = r.multipliers.clone().unwrap();
        let (x0, x1) = (r.x[0], r.x[1]);
        let gphi = [2.0 * (x0 - 2.0), 2.0 * (x1 - 1.0)];
        let g1 = [1.0, -2.0];
        let g2 = [0.5 * x0, 2.0 * x1];
        for i in 0..2 {
            let res = gphi[i] + l[0] * g1[i] + l[1] * g2[i];
            assert!(res.abs() < 1e-6, "{name}: KKT residual {res}, λ = {l:?}");
        }
        assert!(l[1] > 0.0, "{name}: λ = {l:?}");
        // Analytic multipliers.
        let a = [[g1[0], g2[0]], [g1[1], g2[1]]];
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let l1 = (-gphi[0] * a[1][1] + gphi[1] * a[0][1]) / det;
        let l2 = (-gphi[1] * a[0][0] + gphi[0] * a[1][0]) / det;
        assert!(
            (l[0] - l1).abs() < 1e-5 && (l[1] - l2).abs() < 1e-5,
            "{name}"
        );
    }
}

#[test]
fn failed_evaluations_are_rejected_steps() {
    // Rosenbrock that "fails" (like a ray missing a surface) for x₀ > 1.05.
    let p = FnProblem::new(vars(2), 2, |x, r, _| {
        r[0] = 10.0 * (x[1] - x[0] * x[0]);
        r[1] = 1.0 - x[0];
        if x[0] > 1.05 {
            mklight_optim::EvalStatus::Failed
        } else {
            mklight_optim::EvalStatus::Ok
        }
    });
    let r = least_squares(
        &p,
        &[-1.2, 1.0],
        &LsqSettings::default(),
        &Monitor::default(),
    )
    .unwrap();
    assert!(r.merit < 1e-20, "{:?} {}", r.termination, r.merit);
}

#[test]
fn cancellation_mid_run() {
    let p = powell_singular();
    let ctx = JobContext::new();
    let token = ctx.cancel.clone();
    let seen = Arc::new(AtomicUsize::new(0));
    let s2 = seen.clone();
    let mon = Monitor::new(ctx).with_callback(move |rec| {
        if s2.fetch_add(1, Ordering::SeqCst) + 1 == 3 {
            token.cancel();
        }
        assert!(rec.merit.is_finite());
    });
    let s = LsqSettings {
        max_iterations: 1000,
        ..Default::default()
    };
    let r = least_squares(&p, &[3.0, -1.0, 0.0, 1.0], &s, &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert_eq!(seen.load(Ordering::SeqCst), 3);
    // Best-so-far is available and consistent with the result.
    let best = mon.best.get().unwrap();
    assert!(best.merit <= r.merit);
    assert!(best.merit < 215.0);
}

#[test]
fn methods_agree_on_lm_variants() {
    // A non-zero-residual problem with a unique minimiser: every variant
    // (LM, LM + geodesic, central FD, dogleg) must reach the same point and
    // merit, and each run must be bit-for-bit repeatable.
    // r = (x₀ − 1, x₁ − 2, x₀x₁ − 1, 0.5(x₀ + x₁) − 3).
    let p = FnProblem::residuals(vars(2), 4, |x, r| {
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        r[2] = x[0] * x[1] - 1.0;
        r[3] = 0.5 * (x[0] + x[1]) - 3.0;
    });
    let mut sols = Vec::new();
    for (name, s) in methods() {
        let a = least_squares(&p, &[3.0, -1.0], &s, &Monitor::default()).unwrap();
        let b = least_squares(&p, &[3.0, -1.0], &s, &Monitor::default()).unwrap();
        assert_eq!(a, b, "{name}: deterministic");
        assert!(a.termination.is_converged(), "{name}: {:?}", a.termination);
        assert!(a.merit > 0.1, "{name}: non-zero residual problem");
        sols.push((name, a));
    }
    let (_, r0) = &sols[0];
    for (name, r) in &sols[1..] {
        assert!(
            (r.merit - r0.merit).abs() <= 1e-12 * r0.merit,
            "{name}: {} vs {}",
            r.merit,
            r0.merit
        );
        for k in 0..2 {
            assert!(
                (r.x[k] - r0.x[k]).abs() < 1e-6,
                "{name}: {:?} vs {:?}",
                r.x,
                r0.x
            );
        }
    }
    // Stationarity: Jᵀr = 0 at the common solution.
    let (a, b) = (r0.x[0], r0.x[1]);
    let r = [a - 1.0, b - 2.0, a * b - 1.0, 0.5 * (a + b) - 3.0];
    let g0 = r[0] + b * r[2] + 0.5 * r[3];
    let g1 = r[1] + a * r[2] + 0.5 * r[3];
    assert!(g0.abs() < 1e-7 && g1.abs() < 1e-7, "{g0} {g1}");
}

#[test]
fn cancellation_inside_evaluation_returns_best_so_far() {
    // The token is cancelled from inside the evaluator (as another thread
    // would, asynchronously); the solver must still return Ok(Cancelled).
    for s in [LsqSettings::default(), LsqSettings::dogleg()] {
        let ctx = JobContext::new();
        let token = ctx.cancel.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let p = FnProblem::residuals(vars(4), 4, move |x, r| {
            if c2.fetch_add(1, Ordering::SeqCst) == 40 {
                token.cancel();
            }
            r[0] = x[0] + 10.0 * x[1];
            r[1] = 5f64.sqrt() * (x[2] - x[3]);
            r[2] = (x[1] - 2.0 * x[2]).powi(2);
            r[3] = 10f64.sqrt() * (x[0] - x[3]).powi(2);
        });
        let mon = Monitor::new(ctx);
        let r = least_squares(&p, &[3.0, -1.0, 0.0, 1.0], &s, &mon).unwrap();
        assert_eq!(r.termination, Termination::Cancelled);
        assert!(r.merit < 215.0 && r.merit.is_finite());
        assert!(mon.best.get().unwrap().merit <= r.merit);
        assert!(calls.load(Ordering::SeqCst) <= 45);
    }
}

/// Powell's singular function (linear convergence, so an inner solve takes
/// many iterations) with the equality `x₁ + x₂ + x₃ + x₄ = 1`.
fn slow_constrained() -> FnProblem {
    FnProblem::new(vars(4), 4, |x, r, c| {
        r[0] = x[0] + 10.0 * x[1];
        r[1] = 5f64.sqrt() * (x[2] - x[3]);
        r[2] = (x[1] - 2.0 * x[2]).powi(2);
        r[3] = 10f64.sqrt() * (x[0] - x[3]).powi(2);
        c[0] = x[0] + x[1] + x[2] + x[3];
        EvalStatus::Ok
    })
    .with_constraints(vec![Constraint::equal("sum", 1.0)])
}

/// `max_inner_iterations` bounds every augmented-Lagrangian round taken at
/// an infeasible point, so a run that ends on its iteration budget has
/// updated its multipliers and is close to feasible; without the cap the
/// first round (at μ₀) uses the whole budget. Converged runs reach the same
/// point either way.
#[test]
fn augmented_lagrangian_inner_cap() {
    let p = slow_constrained();
    let x0 = [3.0, -1.0, 0.0, 1.0];
    let run = |cap: Option<usize>, iters: usize| {
        let mut s = LsqSettings {
            max_iterations: iters,
            ..Default::default()
        };
        s.constraints.max_inner_iterations = cap;
        least_squares(&p, &x0, &s, &Monitor::default()).unwrap()
    };
    let (free, capped) = (run(None, 40), run(Some(5), 40));
    // Compare the AL endpoints before the separate feasibility restoration.
    let before_restoration = |r: &mklight_optim::OptimResult| {
        r.history
            .iter()
            .rfind(|h| h.stage != "feasibility restoration")
            .unwrap()
            .violation
    };
    assert!(before_restoration(&capped) < 1e-3 * before_restoration(&free));
    assert!(free.violation < 1e-8 && capped.violation < 1e-8);
    // While infeasible no round exceeds the cap.
    let mut rows = std::collections::BTreeMap::<String, usize>::new();
    for h in capped.history.iter().filter(|h| h.violation > 1e-9) {
        *rows.entry(h.stage.clone()).or_default() += 1;
    }
    assert!(rows.len() > 1 && rows.values().all(|&n| n <= 5), "{rows:?}");
    let (a, b) = (run(None, 2000), run(Some(5), 2000));
    assert!(a.termination.is_converged() && b.termination.is_converged());
    assert!(a.violation < 1e-8 && b.violation < 1e-8);
    for (u, v) in a.x.iter().zip(&b.x) {
        assert!((u - v).abs() < 1e-5, "{:?} vs {:?}", a.x, b.x);
    }
}

/// A tangent LM step improves the merit but leaves a second-order equality
/// defect just above the ranking tolerance, as with a focal-length hold.
#[test]
fn restore_tiny_nonlinear_constraint_defect_before_ranking() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let p = FnProblem::new(
        vec![
            Variable::bounded("x", 0.0, 2.0),
            Variable::bounded("y", -1e-3, 0.0),
        ],
        1,
        move |x, r, c| {
            counter.fetch_add(1, Ordering::SeqCst);
            assert!((0.0..=2.0).contains(&x[0]) && (-1e-3..=0.0).contains(&x[1]));
            r[0] = x[0] - 1.0;
            c[0] = x[1] + 2.7e-6 * x[0] * x[0];
            EvalStatus::Ok
        },
    )
    .with_constraints(vec![Constraint::equal("hold", 0.0)]);
    for method in [LsqSettings::default(), LsqSettings::dogleg()] {
        let s = LsqSettings {
            max_iterations: 1,
            ..method
        };
        let mut mon = Monitor::default();
        mon.feasibility_tol = 1e-6;
        let before = calls.load(Ordering::SeqCst);
        let r = least_squares(&p, &[0.0, 0.0], &s, &mon).unwrap();
        assert_eq!(r.evaluations, calls.load(Ordering::SeqCst) - before);
        assert!(r.violation <= mon.feasibility_tol, "{r:?}");
        assert!(r.merit < 0.01, "{r:?}");
        assert!(mon.best.get().unwrap().merit < 0.01);
        assert_eq!(r.iterations, 1);
    }
    for budget in [4, 6, 20] {
        let s = LsqSettings {
            max_iterations: 1,
            max_evaluations: budget,
            ..Default::default()
        };
        let before = calls.load(Ordering::SeqCst);
        let r = least_squares(&p, &[0.0, 0.0], &s, &Monitor::default()).unwrap();
        assert_eq!(r.evaluations, calls.load(Ordering::SeqCst) - before);
        assert!(r.evaluations <= budget);
    }
    let ctx = JobContext::new();
    let token = ctx.cancel.clone();
    let mon = Monitor::new(ctx).with_callback(move |r| {
        if r.stage == "feasibility restoration" {
            token.cancel();
        }
    });
    let s = LsqSettings {
        max_iterations: 1,
        ..Default::default()
    };
    let r = least_squares(&p, &[0.0, 0.0], &s, &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert!(mon.best.get().unwrap().merit < 0.01);
}

#[test]
fn settings_the_solver_cannot_run_with_are_rejected() {
    let p = rosenbrock();
    // A zero trust factor gives a zero trust radius.
    for factor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let s = LsqSettings {
            initial_trust_factor: factor,
            ..Default::default()
        };
        let r = least_squares(&p, &[-1.2, 1.0], &s, &Monitor::default());
        assert!(
            matches!(r, Err(mklight_core::MkError::InvalidParam(_))),
            "{factor}: {r:?}"
        );
    }
}

/// A point that reaches the target within the monitor's feasibility
/// tolerance ends a constrained run, also when that tolerance is looser than
/// the augmented-Lagrangian one.
#[test]
fn target_reached_ends_a_constrained_run() {
    let p = FnProblem::new(vars(1), 1, |x, r, c| {
        r[0] = x[0];
        c[0] = x[0];
        EvalStatus::Ok
    })
    .with_constraints(vec![Constraint::equal("c", 1.0 + 1e-6)]);
    let s = LsqSettings {
        target_merit: 10.0,
        ..Default::default()
    };
    let mut mon = Monitor::default();
    mon.feasibility_tol = 1e-3;
    let r = least_squares(&p, &[1.0], &s, &mon).unwrap();
    assert_eq!(r.termination, Termination::TargetReached);
    assert_eq!((r.x.as_slice(), r.evaluations), (&[1.0][..], 1));
}

/// Speculative probes are finite-difference work: with an analytic Jacobian
/// a run neither evaluates them nor differences them.
#[test]
fn speculation_leaves_an_analytic_jacobian_alone() {
    let p = rosenbrock();
    let plain = LsqSettings::default();
    let speculative = LsqSettings {
        speculative_jacobian: true,
        ..plain
    };
    let a = least_squares(&p, &[-1.2, 1.0], &plain, &Monitor::default()).unwrap();
    let b = least_squares(&p, &[-1.2, 1.0], &speculative, &Monitor::default()).unwrap();
    assert_eq!(a, b);
}
