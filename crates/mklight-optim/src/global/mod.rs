//! Global and derivative-free optimisers.
//!
//! * [`de`] — differential evolution; the whole generation is one batch.
//! * [`nelder_mead`] — bounded Nelder–Mead with adaptive coefficients.
//! * [`cmaes`] — CMA-ES (chosen over particle swarm, see its module docs).
//! * [`hammer`] — basin hopping: batched random restarts refined by the local
//!   least-squares solver, with Metropolis acceptance.
//!
//! They rank candidates with Deb's feasibility rule
//! ([`ScalarEval::better_than`](crate::problem::ScalarEval::better_than)), so
//! explicit constraints and failed evaluations (`Φ = ∞`, infeasible) are
//! handled uniformly. Every evaluated point is inside the variable bounds.

pub mod cmaes;
pub mod de;
pub mod hammer;
pub mod nelder_mead;

use std::cmp::Ordering;

use mklight_core::{MkError, MkResult};

use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{CandidateBatch, Problem, ScalarEval, Variable, evaluate_scalar};
use crate::rand::Rng;

/// Batched scalar evaluation with evaluation counting and best-so-far
/// bookkeeping.
pub(crate) struct Evaluator<'a, P: Problem + ?Sized> {
    pub p: &'a P,
    pub mon: &'a Monitor,
    pub evaluations: usize,
    pub max_evaluations: usize,
    pub history: Vec<IterationRecord>,
}

impl<'a, P: Problem + ?Sized> Evaluator<'a, P> {
    /// Starts a run: clears the best-so-far slot of a top-level monitor.
    pub fn new(p: &'a P, mon: &'a Monitor, max_evaluations: usize) -> Self {
        mon.begin_run();
        Evaluator {
            p,
            mon,
            evaluations: 0,
            max_evaluations,
            history: Vec::new(),
        }
    }
    pub fn eval(&mut self, pts: &[Vec<f64>]) -> MkResult<Vec<ScalarEval>> {
        let n = self.p.num_variables();
        let batch = CandidateBatch::from_points(n, pts.iter().map(|v| v.as_slice()));
        let out = evaluate_scalar(self.p, &batch, &self.mon.job)?;
        self.evaluations += pts.len();
        for (x, e) in pts.iter().zip(&out) {
            self.mon.offer(x, *e);
        }
        Ok(out)
    }
    pub fn budget_left(&self, next: usize) -> bool {
        self.evaluations + next <= self.max_evaluations
    }
    /// Evaluations still allowed.
    pub fn remaining(&self) -> usize {
        self.max_evaluations.saturating_sub(self.evaluations)
    }
    pub fn tol(&self) -> f64 {
        self.mon.feasibility_tol
    }
    /// Total order consistent with Deb's rule (for sorting).
    pub fn cmp(&self, a: &ScalarEval, b: &ScalarEval) -> Ordering {
        a.compare(b, self.tol())
    }
    pub fn record(&mut self, rec: IterationRecord) {
        self.mon.record(&mut self.history, rec);
    }
    /// Final result from the monitor's best point (or `fallback`). A best
    /// point of the wrong dimension (only possible if the slot was shared
    /// with an unrelated run) is ignored.
    pub fn finish(
        self,
        fallback: (&[f64], ScalarEval),
        mut termination: Termination,
        iterations: usize,
    ) -> MkResult<OptimResult> {
        let (x, mut merit, mut violation) = match self.mon.best.get() {
            Some(b)
                if b.x.len() == fallback.0.len()
                    && !fallback.1.better_than(
                        &ScalarEval {
                            merit: b.merit,
                            violation: b.violation,
                            status: crate::problem::EvalStatus::Ok,
                        },
                        self.mon.feasibility_tol,
                    ) =>
            {
                (b.x, b.merit, b.violation)
            }
            _ => (fallback.0.to_vec(), fallback.1.merit, fallback.1.violation),
        };
        if termination != Termination::Cancelled
            && merit.is_finite()
            && let Some(result) = self.p.reference_evaluation(&x, &self.mon.job)
        {
            match result {
                Ok(e) => {
                    merit = crate::problem::merit(self.p.weights(), &e.residuals, e.status);
                    violation = crate::problem::total_violation(
                        self.p.constraints(),
                        &e.constraints,
                        e.status,
                    );
                }
                Err(MkError::Cancelled) => termination = Termination::Cancelled,
                Err(e) => return Err(e),
            }
        }
        Ok(OptimResult {
            x,
            merit,
            violation,
            termination,
            iterations,
            evaluations: self.evaluations,
            multipliers: None,
            history: self.history,
            failed_variables: Vec::new(),
        })
    }
}

/// Sample one coordinate: uniform over finite bounds when there is no
/// centre, otherwise `centre + spread·scale·N(0,1)` clipped to the bounds.
pub(crate) fn sample_coord(v: &Variable, centre: Option<f64>, spread: f64, rng: &mut Rng) -> f64 {
    match centre {
        None if v.is_bounded() => rng.range(v.lo(), v.hi()),
        None => spread * v.safe_scale() * rng.normal(),
        Some(c) => v.clamp(c + spread * v.safe_scale() * rng.normal()),
    }
}

/// Population/simplex extent: max over variables of `(max − min) / scale`.
pub(crate) fn spread(points: &[Vec<f64>], vars: &[Variable]) -> f64 {
    let mut s = 0.0_f64;
    for (j, v) in vars.iter().enumerate() {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in points {
            lo = lo.min(p[j]);
            hi = hi.max(p[j]);
        }
        s = s.max((hi - lo) / v.safe_scale());
    }
    s
}

pub(crate) fn require_domain(vars: &[Variable], x0: Option<&[f64]>) -> MkResult<()> {
    if let Some(x) = x0 {
        if x.len() != vars.len() {
            return Err(MkError::InvalidParam(format!(
                "start point has {} values for {} variables",
                x.len(),
                vars.len()
            )));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(MkError::InvalidParam("start point is not finite".into()));
        }
    } else if let Some(v) = vars.iter().find(|v| !v.is_bounded()) {
        return Err(MkError::InvalidParam(format!(
            "variable '{}' is unbounded and no start point was given",
            v.name
        )));
    }
    if vars.is_empty() {
        return Err(MkError::InvalidParam("problem has no variables".into()));
    }
    if let Some(v) = vars.iter().find(|v| v.lo() > v.hi()) {
        return Err(MkError::InvalidParam(format!(
            "variable '{}': lower bound above upper bound",
            v.name
        )));
    }
    Ok(())
}
