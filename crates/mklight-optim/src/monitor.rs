//! Run control shared by every optimiser: cancellation and progress (through
//! `JobContext`), an iteration-history callback for the UI, and a
//! best-so-far point that another thread can read at any time.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use mklight_core::JobContext;

use crate::problem::{EvalStatus, ScalarEval};

/// One entry of an optimiser's history. Method-specific diagnostics are
/// optional so the UI can plot any of them against `iteration`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationRecord {
    /// Optimiser name (`"lm"`, `"dogleg"`, `"de"`, …).
    pub method: String,
    /// Free-form stage label (outer augmented-Lagrangian loop, hammer round…).
    #[serde(default)]
    pub stage: String,
    pub iteration: usize,
    /// Evaluations of single candidates so far (a batch of N counts N).
    pub evaluations: usize,
    /// Merit `Σ w r²` of the current iterate (population best for population
    /// methods).
    pub merit: f64,
    /// Total normalised constraint violation of the current iterate.
    pub violation: f64,
    /// Merit of the best point seen so far in this run.
    pub best_merit: f64,
    #[serde(default)]
    pub trust_radius: Option<f64>,
    #[serde(default)]
    pub damping: Option<f64>,
    #[serde(default)]
    pub gain_ratio: Option<f64>,
    #[serde(default)]
    pub step_norm: Option<f64>,
    #[serde(default)]
    pub gradient_norm: Option<f64>,
    /// Population / simplex spread in scaled variable units.
    #[serde(default)]
    pub spread: Option<f64>,
}

impl IterationRecord {
    pub fn new(method: &str, iteration: usize, evaluations: usize) -> Self {
        IterationRecord {
            method: method.to_string(),
            stage: String::new(),
            iteration,
            evaluations,
            merit: f64::NAN,
            violation: 0.0,
            best_merit: f64::NAN,
            trust_radius: None,
            damping: None,
            gain_ratio: None,
            step_norm: None,
            gradient_norm: None,
            spread: None,
        }
    }
}

/// The best point found so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BestPoint {
    pub x: Vec<f64>,
    pub merit: f64,
    pub violation: f64,
}

/// Thread-safe best-so-far holder. Cloning shares the same slot.
#[derive(Clone, Debug, Default)]
pub struct BestSoFar(Arc<Mutex<Option<BestPoint>>>);

impl BestSoFar {
    pub fn get(&self) -> Option<BestPoint> {
        self.0.lock().ok().and_then(|g| g.clone())
    }
    /// Offer a point; it replaces the current best under the feasibility rule
    /// of [`ScalarEval::better_than`]. Returns whether it was taken.
    pub fn offer(&self, x: &[f64], e: ScalarEval, feas_tol: f64) -> bool {
        if !e.merit.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let Ok(mut g) = self.0.lock() else {
            return false;
        };
        let take = match g.as_ref() {
            None => true,
            Some(b) => e.better_than(
                &ScalarEval {
                    merit: b.merit,
                    violation: b.violation,
                    status: EvalStatus::Ok,
                },
                feas_tol,
            ),
        };
        if take {
            *g = Some(BestPoint {
                x: x.to_vec(),
                merit: e.merit,
                violation: e.violation,
            });
        }
        take
    }
    pub fn clear(&self) {
        if let Ok(mut g) = self.0.lock() {
            *g = None;
        }
    }
}

type Callback = dyn Fn(&IterationRecord) + Send + Sync;

/// Everything an optimiser needs besides the problem and its settings.
#[derive(Clone)]
pub struct Monitor {
    pub job: JobContext,
    callback: Option<Arc<Callback>>,
    pub best: BestSoFar,
    /// Constraint violation regarded as feasible when ranking points.
    pub feasibility_tol: f64,
    /// Stage label copied into each record.
    pub stage: String,
    /// Keep the full history in the result (the callback is always called).
    pub keep_history: bool,
}

impl std::fmt::Debug for Monitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Monitor")
            .field("stage", &self.stage)
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl Default for Monitor {
    fn default() -> Self {
        Monitor::new(JobContext::new())
    }
}

impl Monitor {
    pub fn new(job: JobContext) -> Self {
        Monitor {
            job,
            callback: None,
            best: BestSoFar::default(),
            feasibility_tol: 1e-8,
            stage: String::new(),
            keep_history: true,
        }
    }
    /// Stream every iteration record to `f` (e.g. to update a merit plot).
    pub fn with_callback(mut self, f: impl Fn(&IterationRecord) + Send + Sync + 'static) -> Self {
        self.callback = Some(Arc::new(f));
        self
    }
    /// A monitor for a nested run: same cancellation token and best-so-far
    /// slot, progress mapped into `[from, to]` of this one, no callback (the
    /// outer method reports its own records).
    pub fn child(&self, from: f64, to: f64, stage: &str) -> Monitor {
        Monitor {
            job: self.job.subrange(from, to),
            callback: None,
            best: self.best.clone(),
            feasibility_tol: self.feasibility_tol,
            stage: stage.to_string(),
            keep_history: false,
        }
    }
    #[inline]
    pub fn cancelled(&self) -> bool {
        self.job.cancel.is_cancelled()
    }
    pub fn progress(&self, f: f64, stage: &str) {
        self.job.report(f, stage);
    }
    /// Emit a record: call the callback and append to `history`.
    pub fn record(&self, history: &mut Vec<IterationRecord>, mut rec: IterationRecord) {
        if rec.stage.is_empty() {
            rec.stage.clone_from(&self.stage);
        }
        if let Some(cb) = &self.callback {
            cb(&rec);
        }
        if self.keep_history {
            history.push(rec);
        }
    }
    pub fn offer(&self, x: &[f64], e: ScalarEval) -> bool {
        self.best.offer(x, e, self.feasibility_tol)
    }
}

/// Why an optimiser stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    /// Relative merit reduction (actual and predicted) below `ftol`.
    SmallMeritChange,
    /// Step / trust region below `xtol`.
    SmallStep,
    /// Scaled projected gradient below `gtol` (first-order stationarity).
    SmallGradient,
    /// Merit reached the requested target.
    TargetReached,
    /// Population or simplex collapsed below its tolerance.
    Converged,
    MaxIterations,
    MaxEvaluations,
    /// Cancelled through the `JobContext`; the result holds the best point.
    Cancelled,
    /// No usable evaluation could be made (all candidates failed).
    EvaluationFailure,
    /// The constraints could not be satisfied within the penalty limits.
    Infeasible,
    /// No improvement for the configured number of iterations.
    Stalled,
}

impl Termination {
    /// Did the method stop because it met a convergence criterion?
    pub fn is_converged(self) -> bool {
        matches!(
            self,
            Termination::SmallMeritChange
                | Termination::SmallStep
                | Termination::SmallGradient
                | Termination::TargetReached
                | Termination::Converged
        )
    }
}

/// Result of an optimisation run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimResult {
    pub x: Vec<f64>,
    pub merit: f64,
    /// Total normalised constraint violation at `x`.
    pub violation: f64,
    pub termination: Termination,
    pub iterations: usize,
    pub evaluations: usize,
    /// Lagrange multipliers per problem constraint, so that at a KKT point
    /// `∇Φ + Σ_k λ_k ∇c_k = 0` (constrained least squares only).
    #[serde(default)]
    pub multipliers: Option<Vec<f64>>,
    #[serde(default)]
    pub history: Vec<IterationRecord>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_so_far_prefers_feasible() {
        let b = BestSoFar::default();
        let ok = |m, v| ScalarEval {
            merit: m,
            violation: v,
            status: EvalStatus::Ok,
        };
        assert!(b.offer(&[1.0], ok(5.0, 1.0), 1e-9));
        assert!(b.offer(&[2.0], ok(9.0, 0.0), 1e-9));
        assert!(!b.offer(&[3.0], ok(1.0, 0.5), 1e-9));
        assert!(b.offer(&[4.0], ok(8.0, 0.0), 1e-9));
        assert_eq!(b.get().unwrap().x, vec![4.0]);
        let shared = b.clone();
        assert!(shared.offer(&[5.0], ok(1.0, 0.0), 1e-9));
        assert_eq!(b.get().unwrap().x, vec![5.0]);
    }
}
