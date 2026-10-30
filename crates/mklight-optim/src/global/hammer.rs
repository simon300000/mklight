//! "Hammer": basin hopping with batched screening and local least-squares
//! refinement (Wales & Doye 1997 style), the workhorse global search for lens
//! design where a good local optimiser exists.
//!
//! Each round:
//! 1. draw `candidates_per_round` start points — Gaussian perturbations of the
//!    current point (`step` in variable scales) and, with probability
//!    `uniform_fraction`, uniform samples of the box — and evaluate them as
//!    **one batch**;
//! 2. run the local solver ([`crate::lsq`], LM by default) from the best
//!    `local_per_round` of them, in parallel;
//! 3. accept the best refined point if it improves on the current one, or
//!    with Metropolis probability `exp(−ΔΦ / (T·Φ_current))` otherwise (a
//!    relative temperature, so `T` does not depend on merit units);
//! 4. adapt `step` towards a 50 % acceptance rate.
//!
//! `max_evaluations` is a hard cap covering the screening batches *and* the
//! local solves: each local solve gets an equal share of the remaining
//! budget as its own `max_evaluations`, and fewer local solves are started
//! when the shares would be too small for one Jacobian and one step.
//!
//! The run stops after `rounds`, after `stall_rounds` rounds without a new
//! global best, on reaching `target_merit`, or on cancellation; the best
//! point found is always returned.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::{MkError, MkResult};

use super::{Evaluator, require_domain};
use crate::lsq::{LsqSettings, least_squares};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{EvalStatus, Problem, ScalarEval, project};
use crate::rand::{Rng, streams};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HammerSettings {
    pub rounds: usize,
    pub candidates_per_round: usize,
    pub local_per_round: usize,
    /// Initial perturbation, in variable scales.
    pub step: f64,
    pub adapt_step: bool,
    /// Probability that a candidate is drawn uniformly over the box (only
    /// when every variable is bounded).
    pub uniform_fraction: f64,
    /// Relative Metropolis temperature; 0 accepts improvements only.
    pub temperature: f64,
    pub stall_rounds: usize,
    pub target_merit: f64,
    pub seed: u64,
    pub max_evaluations: usize,
    pub local: LsqSettings,
}

impl Default for HammerSettings {
    fn default() -> Self {
        let local = LsqSettings {
            max_iterations: 100,
            ..LsqSettings::default()
        };
        HammerSettings {
            rounds: 50,
            candidates_per_round: 16,
            local_per_round: 2,
            step: 0.5,
            adapt_step: true,
            uniform_fraction: 0.25,
            temperature: 0.05,
            stall_rounds: 20,
            target_merit: 0.0,
            seed: 0,
            max_evaluations: 1_000_000,
            local,
        }
    }
}

fn as_scalar(r: &OptimResult) -> ScalarEval {
    if r.merit.is_finite() {
        ScalarEval {
            merit: r.merit,
            violation: r.violation,
            status: EvalStatus::Ok,
        }
    } else {
        ScalarEval::FAILED
    }
}

/// Global search from `x0` by basin hopping with local least squares.
pub fn hammer<P: Problem + ?Sized>(
    p: &P,
    x0: &[f64],
    s: &HammerSettings,
    mon: &Monitor,
) -> MkResult<OptimResult> {
    let vars = p.variables();
    require_domain(vars, Some(x0))?;
    let n = vars.len();
    let all_bounded = vars.iter().all(|v| v.is_bounded());
    let mut ev = Evaluator::new(p, mon, s.max_evaluations);
    let rounds = s.rounds.max(1);
    let frac = |r: usize| r as f64 / (rounds + 1) as f64;

    // Smallest useful local budget: one start point, one FD Jacobian, one step.
    let min_local = n + 2;
    let local_with = |budget: usize| LsqSettings {
        max_evaluations: s.local.max_evaluations.min(budget),
        ..s.local
    };
    if ev.remaining() < min_local {
        return Ok(ev.finish((x0, ScalarEval::FAILED), Termination::MaxEvaluations, 0));
    }
    // Initial local solve.
    let first = least_squares(
        p,
        x0,
        &local_with(ev.remaining()),
        &mon.child(0.0, frac(1), "hammer start"),
    )?;
    ev.evaluations += first.evaluations;
    let mut cur = (first.x.clone(), as_scalar(&first));
    let mut best = cur.clone();
    if first.termination == Termination::Cancelled {
        return Ok(ev.finish((&best.0, best.1), Termination::Cancelled, 0));
    }
    let mut step = s.step;
    let mut stall = 0;
    let mut termination = Termination::MaxIterations;
    let mut done_rounds = 0;
    for round in 0..rounds {
        if mon.cancelled() {
            termination = Termination::Cancelled;
            break;
        }
        if best.1.merit <= s.target_merit && best.1.violation <= ev.tol() {
            termination = Termination::TargetReached;
            break;
        }
        if !ev.budget_left(s.candidates_per_round.max(1)) {
            termination = Termination::MaxEvaluations;
            break;
        }
        // 1. Screen a batch of candidate starts.
        let mut rng = Rng::new(s.seed, streams::HAMMER | round as u64);
        let cands: Vec<Vec<f64>> = (0..s.candidates_per_round.max(1))
            .map(|_| {
                let uniform = all_bounded && rng.uniform() < s.uniform_fraction;
                let mut x: Vec<f64> = (0..n)
                    .map(|j| {
                        let v = &vars[j];
                        if uniform {
                            rng.range(v.lo(), v.hi())
                        } else {
                            cur.0[j] + step * v.safe_scale() * rng.normal()
                        }
                    })
                    .collect();
                project(vars, &mut x);
                x
            })
            .collect();
        let fit = match ev.eval(&cands) {
            Ok(f) => f,
            Err(MkError::Cancelled) => {
                termination = Termination::Cancelled;
                break;
            }
            Err(e) => return Err(e),
        };
        let mut order: Vec<usize> = (0..cands.len())
            .filter(|&k| fit[k].status.is_usable())
            .collect();
        order.sort_by(|&a, &b| ev.cmp(&fit[a], &fit[b]).then(a.cmp(&b)));
        // Split the remaining budget over the local solves.
        let n_local = s
            .local_per_round
            .max(1)
            .min(ev.remaining() / min_local)
            .min(order.len());
        if n_local == 0 && !order.is_empty() {
            termination = Termination::MaxEvaluations;
            break;
        }
        order.truncate(n_local);
        let local = local_with(ev.remaining() / n_local.max(1));

        // 2. Local refinement, in parallel.
        let (f0, f1) = (frac(round + 1), frac(round + 2));
        let locals: Vec<MkResult<OptimResult>> = order
            .par_iter()
            .map(|&k| {
                least_squares(
                    p,
                    &cands[k],
                    &local,
                    &mon.child(f0, f1, &format!("hammer round {round}")),
                )
            })
            .collect();
        let mut round_best: Option<(Vec<f64>, ScalarEval)> = None;
        let mut cancelled = false;
        for l in locals {
            let l = l?;
            ev.evaluations += l.evaluations;
            cancelled |= l.termination == Termination::Cancelled;
            let sc = as_scalar(&l);
            if round_best
                .as_ref()
                .is_none_or(|(_, b)| sc.better_than(b, ev.tol()))
            {
                round_best = Some((l.x, sc));
            }
        }
        done_rounds = round + 1;
        // 3. Acceptance.
        let mut accepted = false;
        if let Some((x, f)) = round_best {
            if f.better_than(&best.1, ev.tol()) {
                best = (x.clone(), f);
                stall = 0;
            } else {
                stall += 1;
            }
            if f.better_than(&cur.1, ev.tol()) {
                accepted = true;
            } else if s.temperature > 0.0
                && f.violation <= ev.tol()
                && cur.1.violation <= ev.tol()
                && f.merit.is_finite()
            {
                let rel = (f.merit - cur.1.merit) / cur.1.merit.abs().max(f64::MIN_POSITIVE);
                accepted = rng.uniform() < (-rel / s.temperature).exp();
            }
            if accepted {
                cur = (x, f);
            }
        } else {
            stall += 1;
        }
        // 4. Step adaptation towards 50 % acceptance.
        if s.adapt_step {
            step = if accepted { step / 0.9 } else { step * 0.9 };
            step = step.clamp(1e-3 * s.step, 10.0 * s.step);
        }
        let mut rec = IterationRecord::new("hammer", done_rounds, ev.evaluations);
        rec.merit = cur.1.merit;
        rec.violation = cur.1.violation;
        rec.best_merit = best.1.merit;
        rec.trust_radius = Some(step);
        ev.record(rec);
        mon.progress(f1, "hammer");
        if cancelled {
            termination = Termination::Cancelled;
            break;
        }
        if stall >= s.stall_rounds.max(1) {
            termination = Termination::Stalled;
            break;
        }
    }
    if termination == Termination::MaxIterations
        && best.1.merit <= s.target_merit
        && best.1.violation <= ev.tol()
    {
        termination = Termination::TargetReached;
    }
    Ok(ev.finish((&best.0, best.1), termination, done_rounds))
}
