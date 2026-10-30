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
    eprintln!(
        "40 iterations: violation {:.3e} uncapped, {:.3e} capped",
        free.violation, capped.violation
    );
    assert!(capped.violation < 1e-3 * free.violation);
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
