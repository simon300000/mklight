//! Bounded Nelder–Mead simplex search.
//!
//! The classic reflection / expansion / contraction / shrink iteration
//! (Lagarias et al. 1998), with the dimension-adaptive coefficients of Gao &
//! Han (2012) — `α = 1`, `β = 1 + 2/n`, `γ = 3/4 − 1/(2n)`, `δ = 1 − 1/n` —
//! which keep the method effective beyond a handful of variables.
//!
//! **Bounds:** every trial point is projected onto the box. Projection keeps
//! the evaluator inside its domain but can flatten the simplex against a
//! bound, so after convergence the search is restarted around the best vertex
//! with a fresh, full-size simplex (`restarts`); a restart that brings no
//! improvement ends the run. Constraints and failed evaluations are ranked by
//! Deb's feasibility rule.
//!
//! **Batching:** the initial simplex and shrink steps are single batches.
//! With `speculative = true` the reflection, expansion and both contraction
//! points are evaluated together in one batch of 4 per iteration, which wastes
//! evaluations on a CPU but halves the number of round trips to a GPU.

use serde::{Deserialize, Serialize};

use mklight_core::{MkError, MkResult};

use super::{Evaluator, require_domain, spread};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{Problem, ScalarEval, project};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NelderMeadSettings {
    /// Initial simplex edge, in variable scales.
    pub initial_step: f64,
    /// Gao–Han coefficients (otherwise the standard 1, 2, ½, ½).
    pub adaptive: bool,
    pub speculative: bool,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    /// Simplex diameter (in scales) regarded as converged.
    pub xtol: f64,
    /// Merit range regarded as converged, relative to `1 + |best|`.
    pub ftol: f64,
    pub restarts: usize,
    pub target_merit: f64,
}

impl Default for NelderMeadSettings {
    fn default() -> Self {
        NelderMeadSettings {
            initial_step: 0.5,
            adaptive: true,
            speculative: false,
            max_iterations: 20_000,
            max_evaluations: 100_000,
            xtol: 1e-10,
            ftol: 1e-14,
            restarts: 3,
            target_merit: 0.0,
        }
    }
}

/// Minimise from `x0` with the bounded Nelder–Mead method.
pub fn nelder_mead<P: Problem + ?Sized>(
    p: &P,
    x0: &[f64],
    s: &NelderMeadSettings,
    mon: &Monitor,
) -> MkResult<OptimResult> {
    let vars = p.variables();
    require_domain(vars, Some(x0))?;
    let n = vars.len();
    let nf = n as f64;
    let (alpha, beta, gamma, delta) = if s.adaptive && n > 1 {
        (1.0, 1.0 + 2.0 / nf, 0.75 - 0.5 / nf, 1.0 - 1.0 / nf)
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };
    let mut ev = Evaluator::new(p, mon, s.max_evaluations);
    let mut start = x0.to_vec();
    project(vars, &mut start);

    let build_simplex = |c: &[f64]| -> Vec<Vec<f64>> {
        let mut sim = vec![c.to_vec()];
        for j in 0..n {
            let v = &vars[j];
            let mut h = s.initial_step * v.safe_scale();
            if v.is_bounded() {
                h = h.min(0.5 * (v.hi() - v.lo()));
            }
            let mut x = c.to_vec();
            x[j] = if c[j] + h <= v.hi() {
                c[j] + h
            } else {
                c[j] - h
            };
            x[j] = v.clamp(x[j]);
            sim.push(x);
        }
        sim
    };
    let cancelled = |ev: Evaluator<P>, x: &[f64], f: ScalarEval, it| {
        Ok(ev.finish((x, f), Termination::Cancelled, it))
    };

    let mut sim = build_simplex(&start);
    let mut f = match ev.eval(&sim) {
        Ok(f) => f,
        Err(MkError::Cancelled) => return cancelled(ev, &start, ScalarEval::FAILED, 0),
        Err(e) => return Err(e),
    };
    let mut restarts_left = s.restarts;
    let mut best_at_restart = None::<ScalarEval>;
    let mut termination = Termination::MaxIterations;
    let mut it = 0;
    while it < s.max_iterations {
        // Sort vertices best → worst (stable, so ties keep their order).
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| ev.cmp(&f[a], &f[b]));
        sim = idx.iter().map(|&i| sim[i].clone()).collect();
        f = idx.iter().map(|&i| f[i]).collect();

        if mon.cancelled() {
            termination = Termination::Cancelled;
            break;
        }
        if f[0].merit <= s.target_merit && f[0].violation <= ev.tol() {
            termination = Termination::TargetReached;
            break;
        }
        let sp = spread(&sim, vars);
        let frange = if f[n].merit.is_finite() {
            f[n].merit - f[0].merit
        } else {
            f64::INFINITY
        };
        if sp <= s.xtol || (frange <= s.ftol * (1.0 + f[0].merit.abs()) && sp <= s.xtol.sqrt()) {
            let improved = best_at_restart.is_none_or(|b| f[0].better_than(&b, ev.tol()));
            if restarts_left == 0 || !improved {
                termination = Termination::Converged;
                break;
            }
            restarts_left -= 1;
            best_at_restart = Some(f[0]);
            let b = sim[0].clone();
            let fb = f[0];
            sim = build_simplex(&b);
            let rest: Vec<Vec<f64>> = sim[1..].to_vec();
            if !ev.budget_left(rest.len()) {
                termination = Termination::MaxEvaluations;
                break;
            }
            let fr = match ev.eval(&rest) {
                Ok(v) => v,
                Err(MkError::Cancelled) => {
                    termination = Termination::Cancelled;
                    break;
                }
                Err(e) => return Err(e),
            };
            f = std::iter::once(fb).chain(fr).collect();
            continue;
        }
        if !ev.budget_left(if s.speculative { 4 } else { 2 }) {
            termination = Termination::MaxEvaluations;
            break;
        }
        it += 1;

        let mut c = vec![0.0; n];
        for v in &sim[..n] {
            for (cj, vj) in c.iter_mut().zip(v) {
                *cj += vj / nf;
            }
        }
        let worst = sim[n].clone();
        let pt = |t: f64| -> Vec<f64> {
            // c + t (c − x_worst), projected.
            let mut x: Vec<f64> = c.iter().zip(&worst).map(|(c, w)| c + t * (c - w)).collect();
            project(vars, &mut x);
            x
        };
        let xr = pt(alpha);
        let xe = pt(alpha * beta);
        let xoc = pt(alpha * gamma);
        let xic = pt(-gamma);

        let res: MkResult<()> = (|| {
            let spec = if s.speculative {
                Some(ev.eval(&[xr.clone(), xe.clone(), xoc.clone(), xic.clone()])?)
            } else {
                None
            };
            let get = |k: usize, x: &Vec<f64>, ev: &mut Evaluator<P>| -> MkResult<ScalarEval> {
                match &spec {
                    Some(v) => Ok(v[k]),
                    None => Ok(ev.eval(std::slice::from_ref(x))?[0]),
                }
            };
            let t = ev.tol();
            let fr = get(0, &xr, &mut ev)?;
            if !fr.better_than(&f[0], t) && fr.better_than(&f[n - 1], t) {
                sim[n] = xr;
                f[n] = fr;
            } else if fr.better_than(&f[0], t) {
                let fe = get(1, &xe, &mut ev)?;
                if fe.better_than(&fr, t) {
                    sim[n] = xe;
                    f[n] = fe;
                } else {
                    sim[n] = xr;
                    f[n] = fr;
                }
            } else {
                let (xc, fc, ok) = if fr.better_than(&f[n], t) {
                    let fc = get(2, &xoc, &mut ev)?;
                    let ok = !fr.better_than(&fc, t);
                    (xoc, fc, ok)
                } else {
                    let fc = get(3, &xic, &mut ev)?;
                    let ok = fc.better_than(&f[n], t);
                    (xic, fc, ok)
                };
                if ok {
                    sim[n] = xc;
                    f[n] = fc;
                } else {
                    // Shrink towards the best vertex.
                    let b = sim[0].clone();
                    let pts: Vec<Vec<f64>> = sim[1..]
                        .iter()
                        .map(|v| {
                            let mut x: Vec<f64> =
                                b.iter().zip(v).map(|(b, v)| b + delta * (v - b)).collect();
                            project(vars, &mut x);
                            x
                        })
                        .collect();
                    let fs = ev.eval(&pts)?;
                    for (k, (x, fx)) in pts.into_iter().zip(fs).enumerate() {
                        sim[k + 1] = x;
                        f[k + 1] = fx;
                    }
                }
            }
            Ok(())
        })();
        match res {
            Ok(()) => {}
            Err(MkError::Cancelled) => {
                termination = Termination::Cancelled;
                break;
            }
            Err(e) => return Err(e),
        }
        let bi = (0..=n).min_by(|&a, &b| ev.cmp(&f[a], &f[b])).unwrap_or(0);
        let mut rec = IterationRecord::new("nelder_mead", it, ev.evaluations);
        rec.merit = f[bi].merit;
        rec.violation = f[bi].violation;
        rec.best_merit = f[bi].merit;
        rec.spread = Some(sp);
        ev.record(rec);
        mon.progress(it as f64 / s.max_iterations as f64, "nelder-mead");
    }
    let bi = (0..=n).min_by(|&a, &b| ev.cmp(&f[a], &f[b])).unwrap_or(0);
    let (bx, bf) = (sim[bi].clone(), f[bi]);
    Ok(ev.finish((&bx, bf), termination, it))
}
