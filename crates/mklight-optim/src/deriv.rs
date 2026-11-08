//! Jacobians: batched finite differences, the analytic hook, and a
//! validation utility that compares the two.
//!
//! Step selection. For variable `j` the base step is
//! `h_j = η · max(|x_j|, scale_j)` with `η = √ε ≈ 1.5·10⁻⁸` (forward) or
//! `η = ∛ε ≈ 6·10⁻⁶` (central), which balances truncation `O(h)` / `O(h²)`
//! against rounding `O(ε/h)` / `O(ε/h²)` for functions of unit scale (Dennis &
//! Schnabel §5.4). Values with more than relative rounding noise need a
//! larger `η`: unless `JacobianSettings::relative_step` is set, the problem
//! chooses it ([`Problem::fd_relative_step`]; an optical problem asks its
//! evaluators — `1e-6` / `1e-4` for f64 wavefront operands, larger for f32
//! GPU evaluation, see `docs/OPTIMIZATION.md`). The step is then rounded so
//! that `x + h` is exactly
//! representable (`h ← (x + h) − x`), removing one rounding error from the
//! quotient. Steps never leave the variable bounds: a forward step that would
//! cross the upper bound is taken backwards, and a central difference that
//! does not fit is replaced by the second-order one-sided formula
//! `f'(x) ≈ (−3f(x) + 4f(x+h) − f(x+2h)) / 2h`.
//!
//! All perturbed points of one Jacobian go to the backend as **one batch**.
//! Variables pinned by equal bounds are not probed (zero column). A column
//! whose probe fails (ray failure, non-finite value) is differenced again
//! towards the other side in a second batch; only columns failing on both
//! sides are reported as failed.

use serde::{Deserialize, Serialize};

use mklight_core::{JobContext, MkError, MkResult};

use crate::linalg::Matrix;
use crate::problem::{BatchEvaluation, CandidateBatch, Evaluation, Problem, Variable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FdScheme {
    /// `n` extra evaluations, `O(h)` truncation error.
    #[default]
    Forward,
    /// `2n` extra evaluations, `O(h²)` truncation error.
    Central,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JacobianSettings {
    pub scheme: FdScheme,
    /// Relative step `η`; `None` = `√ε` (forward) / `∛ε` (central).
    #[serde(default)]
    pub relative_step: Option<f64>,
    /// Use the problem's analytic Jacobian when it provides one.
    #[serde(default = "yes")]
    pub use_analytic: bool,
}

fn yes() -> bool {
    true
}

impl Default for JacobianSettings {
    fn default() -> Self {
        JacobianSettings {
            scheme: FdScheme::Forward,
            relative_step: None,
            use_analytic: true,
        }
    }
}

/// A Jacobian of the raw residuals (`M × n`, unweighted) and constraints
/// (`K × n`).
#[derive(Clone, Debug, PartialEq)]
pub struct Jacobian {
    pub residuals: Matrix,
    pub constraints: Matrix,
    /// Candidate evaluations spent.
    pub evaluations: usize,
    /// Columns whose perturbed evaluations failed (set to zero).
    pub failed_columns: Vec<usize>,
    pub analytic: bool,
}

/// How column `j` is differenced: the offsets (multiples of `h`) evaluated
/// and the coefficients applied to `f(x + o·h)` (the `o = 0` term uses `f0`).
#[derive(Clone, Copy, Debug)]
enum Stencil {
    Forward,
    Backward,
    Central,
    /// Second-order one-sided, `sign = ±1`.
    OneSided2(f64),
}

impl Stencil {
    fn offsets(self) -> &'static [f64] {
        match self {
            Stencil::Forward => &[1.0],
            Stencil::Backward => &[-1.0],
            Stencil::Central => &[1.0, -1.0],
            Stencil::OneSided2(s) if s > 0.0 => &[1.0, 2.0],
            Stencil::OneSided2(_) => &[-1.0, -2.0],
        }
    }
}

/// The relative step `η` of a finite-difference Jacobian: the settings'
/// `relative_step`, else the problem's own requirement
/// ([`Problem::fd_relative_step`], e.g. wavefront operands or f32 GPU
/// evaluation), else `√ε` (forward) / `∛ε` (central).
pub fn relative_step<P: Problem + ?Sized>(p: &P, settings: &JacobianSettings) -> f64 {
    settings
        .relative_step
        .or_else(|| p.fd_relative_step(settings.scheme))
        .unwrap_or(match settings.scheme {
            FdScheme::Forward => f64::EPSILON.sqrt(),
            FdScheme::Central => f64::EPSILON.cbrt(),
        })
}

/// Base step for variable `v` at `x` with relative step `eta`.
pub fn fd_step(v: &Variable, x: f64, eta: f64) -> f64 {
    if let Some(h) = v.fd_step
        && h > 0.0
        && h.is_finite()
    {
        return h;
    }
    eta * x.abs().max(v.safe_scale())
}

fn representable(x: f64, h: f64) -> f64 {
    let t = x + h;
    let h2 = t - x;
    if h2 != 0.0 { h2 } else { h }
}

/// Choose stencil and (representable) step for one variable; `None` for a
/// variable pinned by equal bounds (its column is left at zero).
fn plan_column(v: &Variable, x: f64, scheme: FdScheme, eta: f64) -> Option<(Stencil, f64)> {
    let (lo, hi) = (v.lo(), v.hi());
    if !(lo < hi) {
        return None;
    }
    let h = fd_step(v, x, eta);
    let plan = match scheme {
        FdScheme::Forward => {
            if x + h <= hi {
                one_sided(x, 1.0, h)
            } else if x - h >= lo {
                one_sided(x, -1.0, h)
            } else {
                // Bounds narrower than the step: use the wider side.
                let (room_up, room_dn) = (hi - x, x - lo);
                if room_up >= room_dn {
                    one_sided(x, 1.0, room_up)
                } else {
                    one_sided(x, -1.0, room_dn)
                }
            }
        }
        FdScheme::Central => {
            if x + h <= hi && x - h >= lo {
                (Stencil::Central, h)
            } else if x + 2.0 * h <= hi {
                (Stencil::OneSided2(1.0), h)
            } else if x - 2.0 * h >= lo {
                (Stencil::OneSided2(-1.0), h)
            } else {
                let s = if hi - x >= x - lo { 1.0 } else { -1.0 };
                (Stencil::OneSided2(s), 0.5 * (hi - x).max(x - lo))
            }
        }
    };
    (plan.1 > 0.0 && plan.1.is_finite()).then_some(plan)
}

/// First-order one-sided stencil in direction `dir` with a representable step.
fn one_sided(x: f64, dir: f64, h: f64) -> (Stencil, f64) {
    if dir > 0.0 {
        (Stencil::Forward, representable(x, h))
    } else {
        (Stencil::Backward, -representable(x, -h))
    }
}

/// Retry plan for a column whose probes failed on the side(s) flagged in
/// `failed = (positive side, negative side)`: difference towards the other
/// side (second order when the scheme is central and there is room for
/// `2h`). `None` when both sides failed or there is no room.
fn retry_plan(
    v: &Variable,
    x: f64,
    h: f64,
    scheme: FdScheme,
    failed: (bool, bool),
) -> Option<(Stencil, f64)> {
    let dir = match failed {
        (true, false) => -1.0,
        (false, true) => 1.0,
        _ => return None,
    };
    let room = if dir > 0.0 { v.hi() - x } else { x - v.lo() };
    let plan = if scheme == FdScheme::Central && room >= 2.0 * h {
        (Stencil::OneSided2(dir), h)
    } else if room >= h {
        one_sided(x, dir, h)
    } else if room > 0.0 {
        one_sided(x, dir, room)
    } else {
        return None;
    };
    (plan.1 > 0.0 && plan.1.is_finite()).then_some(plan)
}

impl Stencil {
    /// Derivative from `f(x)` and the probe values in `offsets()` order.
    fn combine(self, h: f64, f0: f64, vals: &[f64]) -> f64 {
        match self {
            Stencil::Forward => (vals[0] - f0) / h,
            Stencil::Backward => (f0 - vals[0]) / h,
            Stencil::Central => (vals[0] - vals[1]) / (2.0 * h),
            Stencil::OneSided2(s) => s * (-3.0 * f0 + 4.0 * vals[0] - vals[1]) / (2.0 * h),
        }
    }
}

/// Finite-difference Jacobian at `x`, given the evaluation `f0` at `x`.
///
/// Variables pinned by equal bounds get a zero column and cost no
/// evaluations. When a probe fails (failed status or non-finite values, e.g.
/// a ray crossing an aperture or TIR edge), the column is differenced again
/// towards the other side in a second batch; only columns that fail on both
/// sides end up in [`Jacobian::failed_columns`] (left at zero).
pub fn fd_jacobian<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    f0: &Evaluation,
    settings: &JacobianSettings,
    ctx: &JobContext,
) -> MkResult<Jacobian> {
    Ok(fd_jacobian_budget(p, x, f0, settings, ctx, usize::MAX)?.expect("unlimited budget"))
}

/// [`fd_jacobian`] spending at most `budget` evaluations: `Ok(None)` when the
/// next batch would exceed it (nothing beyond the budget is evaluated).
pub(crate) fn fd_jacobian_budget<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    f0: &Evaluation,
    settings: &JacobianSettings,
    ctx: &JobContext,
    budget: usize,
) -> MkResult<Option<Jacobian>> {
    fd_jacobian_prefetched(p, x, f0, settings, ctx, budget, None)
}

/// The planned stencils of every column at `x` (pinned columns omitted).
fn plan_columns<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    settings: &JacobianSettings,
) -> Vec<(usize, Stencil, f64)> {
    let vars = p.variables();
    let eta = relative_step(p, settings);
    (0..vars.len())
        .filter_map(|j| plan_column(&vars[j], x[j], settings.scheme, eta).map(|(s, h)| (j, s, h)))
        .collect()
}

/// The batch of one pass: the base point first when `anchored`, then each
/// column's probes in stencil order; anchored at `x`.
fn pass_batch<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    pending: &[(usize, Stencil, f64)],
    anchored: bool,
) -> CandidateBatch {
    let vars = p.variables();
    let mut batch = CandidateBatch::with_capacity(vars.len(), 2 * pending.len() + 1);
    batch.anchor = Some(x.to_vec());
    if anchored {
        batch.push(x);
    }
    let mut xp = x.to_vec();
    for &(j, st, h) in pending {
        for &o in st.offsets() {
            xp[j] = vars[j].clamp(x[j] + o * h);
            batch.push(&xp);
        }
        xp[j] = x[j];
    }
    batch
}

/// The first batch [`fd_jacobian`] evaluates at `x` (its base point first
/// when the problem's differences are anchored), or `None` when no column
/// needs a probe. A caller that evaluates it ahead of time (a speculative
/// trial point) hands the result to [`fd_jacobian_prefetched`].
pub(crate) fn fd_first_batch<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    settings: &JacobianSettings,
) -> Option<CandidateBatch> {
    let pending = plan_columns(p, x, settings);
    (!pending.is_empty()).then(|| pass_batch(p, x, &pending, p.anchored_differences()))
}

/// [`fd_jacobian_budget`] whose first pass may already be evaluated:
/// `prefetched` holds [`fd_first_batch`]`(x)` and its evaluation, which is
/// used (and not counted again) when it is exactly the planned batch.
pub(crate) fn fd_jacobian_prefetched<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    f0: &Evaluation,
    settings: &JacobianSettings,
    ctx: &JobContext,
    budget: usize,
    prefetched: Option<(CandidateBatch, BatchEvaluation)>,
) -> MkResult<Option<Jacobian>> {
    let vars = p.variables();
    let n = vars.len();
    let (m, k) = (p.num_residuals(), p.num_constraints());
    let mut jr = Matrix::zeros(m, n);
    let mut jc = Matrix::zeros(k, n);
    let mut evaluations = 0;
    let mut pending = plan_columns(p, x, settings);
    let mut failed = Vec::new();
    let mut prefetched = prefetched;
    // Evaluators that solve iteratively from the anchor (or that may change
    // precision between batches) re-evaluate `x` in the same batch, and the
    // probes are differenced against that row: every value of one
    // derivative then comes from the same evaluator, precision and solves.
    let anchored = p.anchored_differences();
    // Pass 0: the planned stencils; pass 1: retries towards the other side.
    for pass in 0..2 {
        if pending.is_empty() {
            break;
        }
        let batch = pass_batch(p, x, &pending, anchored);
        let e = match prefetched.take() {
            Some((b, e)) if pass == 0 && b.n_vars == batch.n_vars && b.data == batch.data => e,
            _ => {
                if evaluations + batch.len() > budget {
                    return Ok(None);
                }
                let e = p.evaluate_batch(&batch, ctx)?;
                evaluations += batch.len();
                e
            }
        };
        e.validate(batch.len(), m, k)?;
        let probe_ok = |q: usize| {
            e.status[q].is_usable()
                && e.residuals(q).iter().all(|v| v.is_finite())
                && e.constraints(q).iter().all(|v| v.is_finite())
        };
        let (base_r, base_c) = if anchored && probe_ok(0) {
            (e.residuals(0), e.constraints(0))
        } else {
            (&f0.residuals[..], &f0.constraints[..])
        };
        let mut retry = Vec::new();
        let mut row = anchored as usize;
        for &(j, st, h) in &pending {
            let offs = st.offsets();
            let cnt = offs.len();
            let bad: Vec<f64> = (0..cnt)
                .filter(|&q| !probe_ok(row + q))
                .map(|q| offs[q])
                .collect();
            if bad.is_empty() {
                for i in 0..m {
                    let vals: Vec<f64> = (0..cnt).map(|q| e.residuals(row + q)[i]).collect();
                    jr[(i, j)] = st.combine(h, base_r[i], &vals);
                }
                for i in 0..k {
                    let vals: Vec<f64> = (0..cnt).map(|q| e.constraints(row + q)[i]).collect();
                    jc[(i, j)] = st.combine(h, base_c[i], &vals);
                }
            } else {
                let side = (bad.iter().any(|&o| o > 0.0), bad.iter().any(|&o| o < 0.0));
                match retry_plan(&vars[j], x[j], h, settings.scheme, side) {
                    Some((st2, h2)) if pass == 0 => retry.push((j, st2, h2)),
                    _ => failed.push(j),
                }
            }
            row += cnt;
        }
        pending = retry;
    }
    failed.sort_unstable();
    Ok(Some(Jacobian {
        residuals: jr,
        constraints: jc,
        evaluations,
        failed_columns: failed,
        analytic: false,
    }))
}

/// Jacobian at `x`: analytic when available and allowed, otherwise FD.
pub fn jacobian<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    f0: &Evaluation,
    settings: &JacobianSettings,
    ctx: &JobContext,
) -> MkResult<Jacobian> {
    Ok(jacobian_budget(p, x, f0, settings, ctx, usize::MAX)?.expect("unlimited budget"))
}

/// [`jacobian`] with an evaluation budget for the finite-difference path.
/// Non-finite analytic columns are reported as failed.
pub(crate) fn jacobian_budget<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    f0: &Evaluation,
    settings: &JacobianSettings,
    ctx: &JobContext,
    budget: usize,
) -> MkResult<Option<Jacobian>> {
    if settings.use_analytic
        && let Some(res) = p.jacobian(x, ctx)
    {
        let (jr, jc) = res?;
        let n = p.num_variables();
        if jr.rows != p.num_residuals()
            || jr.cols != n
            || jc.rows != p.num_constraints()
            || (jc.rows > 0 && jc.cols != n)
        {
            return Err(MkError::Numerical(
                "analytic Jacobian has the wrong shape".into(),
            ));
        }
        let jc = if jc.rows == 0 {
            Matrix::zeros(0, n)
        } else {
            jc
        };
        let failed_columns = (0..n)
            .filter(|&j| {
                (0..jr.rows).any(|i| !jr[(i, j)].is_finite())
                    || (0..jc.rows).any(|i| !jc[(i, j)].is_finite())
            })
            .collect();
        return Ok(Some(Jacobian {
            residuals: jr,
            constraints: jc,
            evaluations: 0,
            failed_columns,
            analytic: true,
        }));
    }
    fd_jacobian_budget(p, x, f0, settings, ctx, budget)
}

/// One compared Jacobian entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntryCheck {
    /// `"residual"` or `"constraint"`.
    pub block: String,
    pub row: usize,
    pub col: usize,
    pub analytic: f64,
    pub finite_difference: f64,
    pub abs_error: f64,
    /// `|A − F| / max(|A|, |F|, abs_floor)`.
    pub rel_error: f64,
}

/// Report of [`check_jacobian`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JacobianCheck {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    /// Entries exceeding the tolerance, worst first (capped at 50).
    pub mismatches: Vec<EntryCheck>,
    /// Worst relative error per variable column.
    pub column_max_rel_error: Vec<f64>,
    pub passed: bool,
    pub evaluations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckSettings {
    /// Relative tolerance for an entry to pass.
    pub rel_tol: f64,
    /// Magnitude below which entries are compared absolutely.
    pub abs_floor: f64,
    /// Relative FD step (central differences with Richardson extrapolation).
    pub relative_step: f64,
}

impl Default for CheckSettings {
    fn default() -> Self {
        CheckSettings {
            rel_tol: 1e-6,
            abs_floor: 1e-8,
            relative_step: 1e-3,
        }
    }
}

/// Stencil of the derivative check: `(0, h)` central; `(±1, h)` one-sided
/// towards `±`, with every probe (up to `2h`) inside the bounds — when the box
/// is narrower than that, the wider side is used with `h = room/2`. `(0, 0)`
/// for a variable pinned by equal bounds.
fn check_plan(v: &Variable, x: f64, h: f64) -> (f64, f64) {
    let (lo, hi) = (v.lo(), v.hi());
    if !(lo < hi) {
        return (0.0, 0.0);
    }
    if x + h <= hi && x - h >= lo {
        (0.0, h)
    } else if x + 2.0 * h <= hi {
        (1.0, h)
    } else if x - 2.0 * h >= lo {
        (-1.0, h)
    } else if hi - x >= x - lo {
        (1.0, 0.5 * (hi - x))
    } else {
        (-1.0, 0.5 * (x - lo))
    }
}

/// Compare the analytic Jacobian with a high-accuracy finite-difference one.
///
/// The reference uses central differences at steps `h` and `h/2` combined by
/// Richardson extrapolation, `D = (4·D(h/2) − D(h)) / 3`, which is `O(h⁴)`
/// accurate, so a relatively large `h` (default `10⁻³ · scale`) keeps
/// rounding error negligible. Variables pinned against a bound are differenced
/// with the one-sided second-order formula instead.
pub fn check_jacobian<P: Problem + ?Sized>(
    p: &P,
    x: &[f64],
    settings: &CheckSettings,
    ctx: &JobContext,
) -> MkResult<JacobianCheck> {
    let Some(res) = p.jacobian(x, ctx) else {
        return Err(MkError::Unsupported(
            "problem has no analytic Jacobian to check".into(),
        ));
    };
    let (ja, jca) = res?;
    let vars = p.variables();
    let n = vars.len();
    let (m, k) = (p.num_residuals(), p.num_constraints());

    // Points: per column 4 central points (±h, ±h/2), or fallbacks.
    let mut batch = CandidateBatch::with_capacity(n, 4 * n + 1);
    batch.push(x);
    let mut steps = Vec::with_capacity(n);
    let mut xp = x.to_vec();
    for (j, v) in vars.iter().enumerate() {
        let (sign, h) = check_plan(v, x[j], fd_step(v, x[j], settings.relative_step));
        let offs: [f64; 4] = if sign == 0.0 {
            [1.0, -1.0, 0.5, -0.5]
        } else {
            [sign, 2.0 * sign, 0.5 * sign, sign]
        };
        for o in offs {
            xp[j] = v.clamp(x[j] + o * h);
            batch.push(&xp);
        }
        xp[j] = x[j];
        steps.push((h, sign));
    }
    let e = p.evaluate_batch(&batch, ctx)?;
    e.validate(batch.len(), m, k)?;
    if !e.status[0].is_usable() {
        return Err(MkError::Numerical(
            "evaluation failed at the check point".into(),
        ));
    }
    let mut entries = Vec::new();
    let mut col_max = vec![0.0_f64; n];
    let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
    let mut compare = |block: &str, row: usize, col: usize, a: f64, f: f64| {
        let abs = (a - f).abs();
        let rel = abs / a.abs().max(f.abs()).max(settings.abs_floor);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        col_max[col] = col_max[col].max(rel);
        if rel > settings.rel_tol || !rel.is_finite() {
            entries.push(EntryCheck {
                block: block.to_string(),
                row,
                col,
                analytic: a,
                finite_difference: f,
                abs_error: abs,
                rel_error: rel,
            });
        }
    };
    for (j, &(h, sign)) in steps.iter().enumerate() {
        let base = 1 + 4 * j;
        if !(h > 0.0) {
            continue; // pinned variable: nothing to compare
        }
        if !(base..base + 4).all(|q| e.status[q].is_usable()) {
            return Err(MkError::Numerical(format!(
                "evaluation failed while differencing variable {j}"
            )));
        }
        let deriv = |vals: [f64; 4], f0: f64| -> f64 {
            if sign == 0.0 {
                let d1 = (vals[0] - vals[1]) / (2.0 * h);
                let d2 = (vals[2] - vals[3]) / h;
                (4.0 * d2 - d1) / 3.0
            } else {
                // One-sided second order at h/2 and h, then Richardson (O(h³)).
                let (f_h, f_2h, f_half) = (vals[0], vals[1], vals[2]);
                let d_h = sign * (-3.0 * f0 + 4.0 * f_h - f_2h) / (2.0 * h);
                let d_half = sign * (-3.0 * f0 + 4.0 * f_half - f_h) / h;
                (4.0 * d_half - d_h) / 3.0
            }
        };
        for i in 0..m {
            let v = [0, 1, 2, 3].map(|q| e.residuals(base + q)[i]);
            let fd = deriv(v, e.residuals(0)[i]);
            compare("residual", i, j, ja[(i, j)], fd);
        }
        for i in 0..k {
            let v = [0, 1, 2, 3].map(|q| e.constraints(base + q)[i]);
            let fd = deriv(v, e.constraints(0)[i]);
            compare("constraint", i, j, jca[(i, j)], fd);
        }
    }
    entries.sort_by(|a, b| b.rel_error.total_cmp(&a.rel_error));
    let passed = entries.is_empty();
    entries.truncate(50);
    Ok(JacobianCheck {
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        mismatches: entries,
        column_max_rel_error: col_max,
        passed,
        evaluations: batch.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{BatchEvaluation, FnProblem, evaluate_one};

    fn test_problem() -> FnProblem {
        // r0 = exp(a)·sin(b), r1 = a³ − 2ab, constraint c = a² + b.
        FnProblem::new(
            vec![Variable::new("a"), Variable::bounded("b", -1.0, 0.7)],
            2,
            |x, r, c| {
                r[0] = x[0].exp() * x[1].sin();
                r[1] = x[0].powi(3) - 2.0 * x[0] * x[1];
                c[0] = x[0] * x[0] + x[1];
                crate::problem::EvalStatus::Ok
            },
        )
        .with_constraints(vec![crate::problem::Constraint::at_most("c", 1.0)])
    }

    fn exact(x: &[f64]) -> [[f64; 2]; 3] {
        let (a, b) = (x[0], x[1]);
        [
            [a.exp() * b.sin(), a.exp() * b.cos()],
            [3.0 * a * a - 2.0 * b, -2.0 * a],
            [2.0 * a, 1.0],
        ]
    }

    #[test]
    fn fd_accuracy_forward_and_central() {
        let p = test_problem();
        let ctx = JobContext::new();
        let x = [0.3, 0.4];
        let f0 = evaluate_one(&p, &x, &ctx).unwrap();
        let ex = exact(&x);
        for (scheme, tol) in [(FdScheme::Forward, 1e-7), (FdScheme::Central, 1e-10)] {
            let s = JacobianSettings {
                scheme,
                ..Default::default()
            };
            let j = fd_jacobian(&p, &x, &f0, &s, &ctx).unwrap();
            assert_eq!(
                j.evaluations,
                if scheme == FdScheme::Forward { 2 } else { 4 }
            );
            for c in 0..2 {
                for r in 0..2 {
                    assert!((j.residuals[(r, c)] - ex[r][c]).abs() < tol, "{scheme:?}");
                }
                assert!((j.constraints[(0, c)] - ex[2][c]).abs() < tol);
            }
        }
    }

    /// Values that depend on the batch they are evaluated in (an offset per
    /// batch, standing in for a device/reference precision change between
    /// batches), optionally anchored.
    struct BatchOffset(FnProblem, bool);

    impl Problem for BatchOffset {
        fn variables(&self) -> &[Variable] {
            self.0.variables()
        }
        fn num_residuals(&self) -> usize {
            self.0.num_residuals()
        }
        fn weights(&self) -> &[f64] {
            self.0.weights()
        }
        fn constraints(&self) -> &[crate::problem::Constraint] {
            self.0.constraints()
        }
        fn evaluate_batch(
            &self,
            x: &CandidateBatch,
            ctx: &JobContext,
        ) -> MkResult<BatchEvaluation> {
            let mut e = self.0.evaluate_batch(x, ctx)?;
            let offset = 1e-9 * x.len() as f64;
            e.residuals.iter_mut().for_each(|r| *r += offset);
            e.constraints.iter_mut().for_each(|c| *c += offset);
            Ok(e)
        }
        fn anchored_differences(&self) -> bool {
            self.1
        }
    }

    #[test]
    fn anchored_differences_share_the_batch_of_their_probes() {
        let ctx = JobContext::new();
        let x = [0.3, 0.4];
        let ex = exact(&x);
        let s = JacobianSettings::default();
        for anchored in [false, true] {
            let p = BatchOffset(test_problem(), anchored);
            // The base value comes from a batch of one, the probes from one
            // of two (three with the anchored base row).
            let f0 = evaluate_one(&p, &x, &ctx).unwrap();
            let j = fd_jacobian(&p, &x, &f0, &s, &ctx).unwrap();
            assert_eq!(j.evaluations, 2 + anchored as usize);
            let err = (0..2)
                .flat_map(|c| (0..2).map(move |r| (r, c)))
                .map(|(r, c)| (j.residuals[(r, c)] - ex[r][c]).abs())
                .fold(0.0, f64::max);
            if anchored {
                assert!(err < 1e-7, "anchored error {err}");
            } else {
                // The batch offset, divided by the ~1.5e-8 step.
                assert!(err > 1e-2, "unanchored error {err}");
            }
        }
    }

    #[test]
    fn fd_respects_bounds() {
        // b sits on its upper bound: forward → backward, central → one-sided.
        let p = test_problem();
        let ctx = JobContext::new();
        let x = [0.3, 0.7];
        let f0 = evaluate_one(&p, &x, &ctx).unwrap();
        let ex = exact(&x);
        for (scheme, tol) in [(FdScheme::Forward, 1e-7), (FdScheme::Central, 1e-9)] {
            let s = JacobianSettings {
                scheme,
                ..Default::default()
            };
            let j = fd_jacobian(&p, &x, &f0, &s, &ctx).unwrap();
            assert!((j.residuals[(0, 1)] - ex[0][1]).abs() < tol, "{scheme:?}");
        }
    }

    #[test]
    fn check_finds_wrong_derivative() {
        let good = test_problem().with_jacobian(|x| {
            let e = exact(x);
            (
                Matrix::from_row_major(2, 2, vec![e[0][0], e[0][1], e[1][0], e[1][1]]),
                Matrix::from_row_major(1, 2, vec![e[2][0], e[2][1]]),
            )
        });
        let ctx = JobContext::new();
        let rep = check_jacobian(&good, &[0.3, 0.4], &CheckSettings::default(), &ctx).unwrap();
        assert!(rep.passed, "{rep:?}");
        assert!(rep.max_rel_error < 1e-8);
        // Also on the bound (one-sided branch).
        let rep = check_jacobian(&good, &[0.3, 0.7], &CheckSettings::default(), &ctx).unwrap();
        assert!(rep.passed, "{rep:?}");

        let bad = test_problem().with_jacobian(|x| {
            let e = exact(x);
            (
                // d r1 / d b has the wrong sign.
                Matrix::from_row_major(2, 2, vec![e[0][0], e[0][1], e[1][0], -e[1][1]]),
                Matrix::from_row_major(1, 2, vec![e[2][0], e[2][1]]),
            )
        });
        let rep = check_jacobian(&bad, &[0.3, 0.4], &CheckSettings::default(), &ctx).unwrap();
        assert!(!rep.passed);
        assert_eq!(rep.mismatches.len(), 1);
        assert_eq!((rep.mismatches[0].row, rep.mismatches[0].col), (1, 1));
        assert!(rep.column_max_rel_error[0] < 1e-8);
    }
}
