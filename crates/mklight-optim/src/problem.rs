//! The problem abstraction every optimiser in this crate works on.
//!
//! A problem has `n` **variables** (with optional bounds and a typical scale),
//! `M` weighted **residuals** `r_i(x)` and `K` explicit **constraint functions**
//! `c_k(x)` with optional lower/upper limits. The merit function is
//!
//! ```text
//! Φ(x) = Σ_i w_i · r_i(x)²          (w_i ≥ 0)
//! ```
//!
//! subject to `lo_k ≤ c_k(x) ≤ hi_k` and the variable bounds.
//!
//! Evaluation is **batched**: a backend receives `N` candidate points as one
//! contiguous `N × n` row-major block and returns `N × M` residuals and
//! `N × K` constraint values in one call. Finite-difference Jacobians, DE /
//! CMA-ES populations and multi-start seeds all go through this single entry
//! point, which is where a GPU backend amortises its launch overhead.
//!
//! Evaluation failures are data, not panics: each candidate carries an
//! [`EvalStatus`]. `Degraded` means some residuals were replaced by the
//! evaluator's penalty values (e.g. a few vignetted rays); `Failed` means the
//! candidate has no meaningful merit (e.g. the chief ray missed a surface) and
//! every optimiser treats it as infinitely bad / infeasible.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::{JobContext, MkError, MkResult};

use crate::linalg::Matrix;

/// One continuous optimisation variable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub lower: Option<f64>,
    #[serde(default)]
    pub upper: Option<f64>,
    /// Typical magnitude of *changes* in this variable (> 0). Sets
    /// finite-difference steps, initial simplex / population spreads and the
    /// initial trust region. For a curvature this might be `1e-3 mm⁻¹`, for an
    /// air space `1 mm`.
    #[serde(default = "one")]
    pub scale: f64,
    /// Explicit finite-difference step (absolute). `None` = automatic.
    #[serde(default)]
    pub fd_step: Option<f64>,
}

fn one() -> f64 {
    1.0
}

impl Variable {
    pub fn new(name: impl Into<String>) -> Self {
        Variable {
            name: name.into(),
            lower: None,
            upper: None,
            scale: 1.0,
            fd_step: None,
        }
    }
    pub fn bounded(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Variable {
            lower: Some(lower),
            upper: Some(upper),
            ..Variable::new(name)
        }
    }
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }
    pub fn with_lower(mut self, lo: f64) -> Self {
        self.lower = Some(lo);
        self
    }
    pub fn with_upper(mut self, hi: f64) -> Self {
        self.upper = Some(hi);
        self
    }
    #[inline]
    pub fn lo(&self) -> f64 {
        self.lower.unwrap_or(f64::NEG_INFINITY)
    }
    #[inline]
    pub fn hi(&self) -> f64 {
        self.upper.unwrap_or(f64::INFINITY)
    }
    #[inline]
    pub fn clamp(&self, x: f64) -> f64 {
        x.max(self.lo()).min(self.hi())
    }
    pub fn is_bounded(&self) -> bool {
        self.lower.is_some() && self.upper.is_some()
    }
    /// A positive, finite scale (falls back to 1).
    #[inline]
    pub fn safe_scale(&self) -> f64 {
        if self.scale.is_finite() && self.scale > 0.0 {
            self.scale
        } else {
            1.0
        }
    }
}

/// An explicit constraint `lower ≤ c_k(x) ≤ upper` on a scalar function the
/// problem evaluates. Equality when `lower == upper`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub lower: Option<f64>,
    #[serde(default)]
    pub upper: Option<f64>,
    /// Typical magnitude of `c_k`, used to normalise violations so that
    /// constraints in mm and in degrees are comparable.
    #[serde(default = "one")]
    pub scale: f64,
}

impl Constraint {
    pub fn equal(name: impl Into<String>, target: f64) -> Self {
        Constraint {
            name: name.into(),
            lower: Some(target),
            upper: Some(target),
            scale: 1.0,
        }
    }
    pub fn at_most(name: impl Into<String>, upper: f64) -> Self {
        Constraint {
            name: name.into(),
            lower: None,
            upper: Some(upper),
            scale: 1.0,
        }
    }
    pub fn at_least(name: impl Into<String>, lower: f64) -> Self {
        Constraint {
            name: name.into(),
            lower: Some(lower),
            upper: None,
            scale: 1.0,
        }
    }
    pub fn between(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Constraint {
            name: name.into(),
            lower: Some(lower),
            upper: Some(upper),
            scale: 1.0,
        }
    }
    pub fn is_equality(&self) -> bool {
        matches!((self.lower, self.upper), (Some(a), Some(b)) if (a - b).abs() <= 1e-15 * (1.0 + a.abs()))
    }
    /// Signed-free violation `max(lo − c, c − hi, 0)`, in units of `scale`.
    pub fn violation(&self, c: f64) -> f64 {
        if !c.is_finite() {
            return f64::INFINITY;
        }
        let s = if self.scale > 0.0 && self.scale.is_finite() {
            self.scale
        } else {
            1.0
        };
        let lo = self.lower.map_or(0.0, |l| (l - c).max(0.0));
        let hi = self.upper.map_or(0.0, |h| (c - h).max(0.0));
        lo.max(hi) / s
    }
}

/// Outcome of evaluating one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EvalStatus {
    #[default]
    Ok,
    /// Usable, but `failures` residuals carry evaluator penalty values (for
    /// example vignetted or failed rays). Optimisers use the values as given.
    Degraded { failures: u32 },
    /// No meaningful merit. Treated as `Φ = +∞` and infeasible.
    Failed,
}

impl EvalStatus {
    pub fn is_usable(self) -> bool {
        !matches!(self, EvalStatus::Failed)
    }
}

/// `N` candidate points, row-major `N × n`. Contiguous so a GPU backend can
/// upload it in one copy.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CandidateBatch {
    pub n_vars: usize,
    pub data: Vec<f64>,
}

impl CandidateBatch {
    pub fn new(n_vars: usize) -> Self {
        CandidateBatch {
            n_vars,
            data: Vec::new(),
        }
    }
    pub fn with_capacity(n_vars: usize, n: usize) -> Self {
        CandidateBatch {
            n_vars,
            data: Vec::with_capacity(n * n_vars),
        }
    }
    pub fn from_points<'a>(n_vars: usize, pts: impl IntoIterator<Item = &'a [f64]>) -> Self {
        let mut b = CandidateBatch::new(n_vars);
        for p in pts {
            b.push(p);
        }
        b
    }
    pub fn push(&mut self, x: &[f64]) {
        assert_eq!(x.len(), self.n_vars, "candidate length mismatch");
        self.data.extend_from_slice(x);
    }
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.n_vars).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    #[inline]
    pub fn point(&self, k: usize) -> &[f64] {
        &self.data[k * self.n_vars..(k + 1) * self.n_vars]
    }
}

/// Results for a [`CandidateBatch`]: `N × M` residuals, `N × K` constraint
/// values and one status per candidate, all row-major.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BatchEvaluation {
    pub n_residuals: usize,
    pub n_constraints: usize,
    pub residuals: Vec<f64>,
    pub constraints: Vec<f64>,
    pub status: Vec<EvalStatus>,
}

impl BatchEvaluation {
    /// A batch of `n` candidates initialised to zero / `Ok`.
    pub fn zeros(n: usize, n_residuals: usize, n_constraints: usize) -> Self {
        BatchEvaluation {
            n_residuals,
            n_constraints,
            residuals: vec![0.0; n * n_residuals],
            constraints: vec![0.0; n * n_constraints],
            status: vec![EvalStatus::Ok; n],
        }
    }
    pub fn len(&self) -> usize {
        self.status.len()
    }
    pub fn is_empty(&self) -> bool {
        self.status.is_empty()
    }
    #[inline]
    pub fn residuals(&self, k: usize) -> &[f64] {
        &self.residuals[k * self.n_residuals..(k + 1) * self.n_residuals]
    }
    #[inline]
    pub fn constraints(&self, k: usize) -> &[f64] {
        &self.constraints[k * self.n_constraints..(k + 1) * self.n_constraints]
    }
    /// The single evaluation of candidate `k`.
    pub fn get(&self, k: usize) -> Evaluation {
        Evaluation {
            residuals: self.residuals(k).to_vec(),
            constraints: self.constraints(k).to_vec(),
            status: self.status[k],
        }
    }
    /// Check the shape against the problem and the batch size.
    pub fn validate(&self, n: usize, m: usize, k: usize) -> MkResult<()> {
        if self.status.len() != n
            || self.n_residuals != m
            || self.n_constraints != k
            || self.residuals.len() != n * m
            || self.constraints.len() != n * k
        {
            return Err(MkError::Numerical(format!(
                "batch evaluation shape mismatch: expected {n}×({m} residuals, {k} constraints)"
            )));
        }
        Ok(())
    }
}

/// One evaluated candidate.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Evaluation {
    pub residuals: Vec<f64>,
    pub constraints: Vec<f64>,
    pub status: EvalStatus,
}

/// A least-squares problem with explicit constraints and batched evaluation.
///
/// Implementors must be `Sync`; optimisers may call `evaluate_batch` from a
/// worker thread. Variable bounds are enforced by the optimisers (candidates
/// passed in are always inside the bounds), so evaluators never see
/// out-of-range values.
pub trait Problem: Sync {
    fn variables(&self) -> &[Variable];
    fn num_residuals(&self) -> usize;
    /// Residual weights `w_i ≥ 0` (length `num_residuals`).
    fn weights(&self) -> &[f64];
    fn constraints(&self) -> &[Constraint] {
        &[]
    }
    /// Evaluate every candidate of `x`. Returns `Err` only for cancellation or
    /// infrastructure errors; per-candidate failures go in `status`.
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation>;

    /// Analytic Jacobian hook. Return `Some((J_r, J_c))` with the `M × n`
    /// residual Jacobian (unweighted) and the `K × n` constraint Jacobian at
    /// `x`. The default `None` makes the optimisers use finite differences.
    fn jacobian(&self, _x: &[f64], _ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        None
    }

    /// Relative finite-difference step `η` for `scheme` when
    /// `JacobianSettings::relative_step` is not set: problems whose values
    /// carry more than relative rounding noise (absolute OPD noise, f32 GPU
    /// traces) say what they need. `None` = `√ε` / `∛ε` (see `deriv`).
    fn fd_relative_step(&self, _scheme: crate::deriv::FdScheme) -> Option<f64> {
        None
    }

    fn num_variables(&self) -> usize {
        self.variables().len()
    }
    fn num_constraints(&self) -> usize {
        self.constraints().len()
    }
}

/// Convenience: evaluate one point.
pub fn evaluate_one<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    ctx: &JobContext,
) -> MkResult<Evaluation> {
    let b = CandidateBatch::from_points(x.len(), [x]);
    let e = p.evaluate_batch(&b, ctx)?;
    e.validate(1, p.num_residuals(), p.num_constraints())?;
    Ok(e.get(0))
}

/// `Φ = Σ w_i r_i²` with compensated summation; `+∞` for failed or
/// non-finite evaluations.
pub fn merit(weights: &[f64], residuals: &[f64], status: EvalStatus) -> f64 {
    if !status.is_usable() {
        return f64::INFINITY;
    }
    let s: mklight_core::math::KahanSum = weights
        .iter()
        .zip(residuals)
        .map(|(w, r)| w * r * r)
        .collect();
    let v = s.value();
    if v.is_finite() { v } else { f64::INFINITY }
}

/// Total normalised constraint violation `Σ_k violation_k` (0 when feasible,
/// `+∞` for failed evaluations).
pub fn total_violation(constraints: &[Constraint], values: &[f64], status: EvalStatus) -> f64 {
    if !status.is_usable() {
        return f64::INFINITY;
    }
    constraints
        .iter()
        .zip(values)
        .map(|(c, v)| c.violation(*v))
        .sum()
}

/// Clamp a point into the variable bounds (in place).
pub fn project(vars: &[Variable], x: &mut [f64]) {
    for (xi, v) in x.iter_mut().zip(vars) {
        *xi = v.clamp(*xi);
    }
}

/// Validate a starting point and the problem's static description.
pub fn check_problem<P: Problem + ?Sized>(p: &P, x0: &[f64]) -> MkResult<()> {
    let vars = p.variables();
    if x0.len() != vars.len() {
        return Err(MkError::InvalidParam(format!(
            "start point has {} values for {} variables",
            x0.len(),
            vars.len()
        )));
    }
    if vars.is_empty() {
        return Err(MkError::InvalidParam("problem has no variables".into()));
    }
    for v in vars {
        if v.lo() > v.hi() {
            return Err(MkError::InvalidParam(format!(
                "variable '{}': lower bound above upper bound",
                v.name
            )));
        }
    }
    if p.weights().len() != p.num_residuals() {
        return Err(MkError::InvalidParam(
            "weights length differs from residual count".into(),
        ));
    }
    if p.weights().iter().any(|w| !(*w >= 0.0 && w.is_finite())) {
        return Err(MkError::InvalidParam(
            "weights must be finite and ≥ 0".into(),
        ));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(MkError::InvalidParam("start point is not finite".into()));
    }
    Ok(())
}

type EvalFn = dyn Fn(&[f64], &mut [f64], &mut [f64]) -> EvalStatus + Send + Sync;

/// A [`Problem`] built from a closure `f(x, residuals_out, constraints_out)
/// -> status`. Batches are evaluated in parallel on the rayon pool, with the
/// cancellation token checked between candidates.
pub struct FnProblem {
    vars: Vec<Variable>,
    weights: Vec<f64>,
    constraints: Vec<Constraint>,
    f: Box<EvalFn>,
    #[allow(clippy::type_complexity)]
    jac: Option<Box<dyn Fn(&[f64]) -> (Matrix, Matrix) + Send + Sync>>,
}

impl FnProblem {
    /// Unit weights.
    pub fn new(
        vars: Vec<Variable>,
        n_residuals: usize,
        f: impl Fn(&[f64], &mut [f64], &mut [f64]) -> EvalStatus + Send + Sync + 'static,
    ) -> Self {
        FnProblem {
            vars,
            weights: vec![1.0; n_residuals],
            constraints: Vec::new(),
            f: Box::new(f),
            jac: None,
        }
    }
    /// Residual-only problem from `f(x, r)`, all weights 1, always `Ok`.
    pub fn residuals(
        vars: Vec<Variable>,
        n_residuals: usize,
        f: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> Self {
        FnProblem::new(vars, n_residuals, move |x, r, _c| {
            f(x, r);
            EvalStatus::Ok
        })
    }
    pub fn with_weights(mut self, w: Vec<f64>) -> Self {
        assert_eq!(w.len(), self.weights.len());
        self.weights = w;
        self
    }
    pub fn with_constraints(mut self, c: Vec<Constraint>) -> Self {
        self.constraints = c;
        self
    }
    /// Attach an analytic Jacobian `x → (J_r, J_c)`.
    pub fn with_jacobian(
        mut self,
        j: impl Fn(&[f64]) -> (Matrix, Matrix) + Send + Sync + 'static,
    ) -> Self {
        self.jac = Some(Box::new(j));
        self
    }
    pub fn variables_mut(&mut self) -> &mut Vec<Variable> {
        &mut self.vars
    }
}

impl Problem for FnProblem {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_residuals(&self) -> usize {
        self.weights.len()
    }
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        let (n, m, k) = (x.len(), self.num_residuals(), self.constraints.len());
        let mut out = BatchEvaluation::zeros(n, m, k);
        type Row = MkResult<(Vec<f64>, Vec<f64>, EvalStatus)>;
        let results: Vec<Row> = (0..n)
            .into_par_iter()
            .map(|i| {
                ctx.check()?;
                let mut r = vec![0.0; m];
                let mut c = vec![0.0; k];
                let st = (self.f)(x.point(i), &mut r, &mut c);
                Ok((r, c, st))
            })
            .collect();
        for (i, res) in results.into_iter().enumerate() {
            let (r, c, st) = res?;
            out.residuals[i * m..(i + 1) * m].copy_from_slice(&r);
            out.constraints[i * k..(i + 1) * k].copy_from_slice(&c);
            out.status[i] = st;
        }
        Ok(out)
    }
    fn jacobian(&self, x: &[f64], _ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        self.jac.as_ref().map(|j| Ok(j(x)))
    }
}

/// Scalar view of a problem used by the derivative-free and global methods:
/// `(merit, total violation, status)` per candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarEval {
    pub merit: f64,
    pub violation: f64,
    pub status: EvalStatus,
}

impl ScalarEval {
    pub const FAILED: ScalarEval = ScalarEval {
        merit: f64::INFINITY,
        violation: f64::INFINITY,
        status: EvalStatus::Failed,
    };
    /// Deb's feasibility rule: feasible beats infeasible, two feasible
    /// points compare by merit, two infeasible ones by violation. `tol` is
    /// the violation regarded as feasible.
    pub fn better_than(&self, o: &ScalarEval, tol: f64) -> bool {
        let fa = self.violation <= tol;
        let fb = o.violation <= tol;
        match (fa, fb) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => self.merit < o.merit,
            (false, false) => {
                self.violation < o.violation
                    || (self.violation == o.violation && self.merit < o.merit)
            }
        }
    }
    /// Total order consistent with [`Self::better_than`] (best first), for sorting.
    pub fn compare(&self, o: &ScalarEval, tol: f64) -> std::cmp::Ordering {
        if self.better_than(o, tol) {
            std::cmp::Ordering::Less
        } else if o.better_than(self, tol) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

/// Evaluate a batch and reduce each candidate to a [`ScalarEval`].
pub fn evaluate_scalar<P: Problem + ?Sized>(
    p: &P,
    x: &CandidateBatch,
    ctx: &JobContext,
) -> MkResult<Vec<ScalarEval>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let e = p.evaluate_batch(x, ctx)?;
    e.validate(x.len(), p.num_residuals(), p.num_constraints())?;
    let w = p.weights();
    let cons = p.constraints();
    Ok((0..x.len())
        .map(|k| {
            let st = e.status[k];
            let merit = merit(w, e.residuals(k), st);
            let violation = total_violation(cons, e.constraints(k), st);
            if !merit.is_finite() {
                ScalarEval::FAILED
            } else {
                ScalarEval {
                    merit,
                    violation,
                    status: st,
                }
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_layout_and_merit() {
        let p = FnProblem::new(
            vec![Variable::new("a"), Variable::new("b")],
            2,
            |x, r, _| {
                r[0] = x[0];
                r[1] = 2.0 * x[1];
                if x[0] < 0.0 {
                    EvalStatus::Failed
                } else {
                    EvalStatus::Ok
                }
            },
        )
        .with_weights(vec![1.0, 0.5]);
        let b = CandidateBatch::from_points(2, [&[1.0, 1.0][..], &[-1.0, 0.0][..]]);
        let s = evaluate_scalar(&p, &b, &JobContext::new()).unwrap();
        assert_eq!(s[0].merit, 1.0 + 0.5 * 4.0);
        assert_eq!(s[1].status, EvalStatus::Failed);
        assert!(s[1].merit.is_infinite());
    }

    #[test]
    fn constraint_violation() {
        let c = Constraint::between("c", -1.0, 2.0);
        assert_eq!(c.violation(0.0), 0.0);
        assert_eq!(c.violation(3.0), 1.0);
        assert_eq!(c.violation(-4.0), 3.0);
        assert!(Constraint::equal("e", 1.0).is_equality());
        assert!(!c.is_equality());
    }

    #[test]
    fn feasibility_rule() {
        let feas = ScalarEval {
            merit: 10.0,
            violation: 0.0,
            status: EvalStatus::Ok,
        };
        let infeas = ScalarEval {
            merit: 1.0,
            violation: 0.1,
            status: EvalStatus::Ok,
        };
        assert!(feas.better_than(&infeas, 1e-9));
        assert!(!infeas.better_than(&feas, 1e-9));
        assert!(feas.better_than(&ScalarEval::FAILED, 1e-9));
    }
}
