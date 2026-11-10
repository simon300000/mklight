//! Local bound- and constraint-aware nonlinear least squares:
//! Levenberg–Marquardt and Powell's dogleg, both as trust-region methods in
//! the scaled norm `‖D p‖` of Moré (1978).
//!
//! **Model.** At `x` with weighted residual vector `r` and Jacobian `J`, the
//! step minimises `‖r + J p‖²` subject to `‖D p‖ ≤ Δ`, where
//! `D = diag(d_j)`, `d_j = max over iterations of ‖J_{:,j}‖` (Moré scaling,
//! which makes the method invariant to variable units).
//!
//! * **LM** solves the trust-region subproblem exactly: `p(λ)` from
//!   `min ‖[J; √λ D] p + [r; 0]‖` by Householder QR (never via the normal
//!   equations), with `λ` found by safeguarded Newton iteration on the secular
//!   equation `1/‖D p(λ)‖ = 1/Δ` (Moré's algorithm, as in MINPACK `lmpar`).
//!   Optional geodesic acceleration (Transtrum & Sethna 2012) adds the
//!   second-order correction `a = −(JᵀJ + λD²)⁻¹ Jᵀ r''_v`, with `r''_v` from
//!   one extra finite-difference evaluation along the step, accepted when
//!   `2‖Da‖ ≤ 0.75 ‖Dv‖`.
//! * **Dogleg** combines the Gauss–Newton step and the scaled Cauchy point on
//!   the dogleg path, with no `λ` search (cheaper per step, less robust far
//!   from a solution).
//!
//! `Δ` follows MINPACK: shrink to `¼ min(Δ, ‖Dp‖)` when the gain ratio
//! `ρ = actual / predicted reduction < ¼`, set to `2‖Dp‖` when `ρ ≥ ¾` (or the
//! Gauss–Newton step was taken); steps with `ρ ≤ 10⁻⁴` are rejected.
//!
//! **Bounds: projected active-set.** A variable sitting on a bound whose
//! gradient pushes it outward is frozen for the iteration; the subproblem is
//! solved over the free variables and the trial point is projected onto the
//! box. The predicted reduction is computed for the *projected* step, so the
//! gain ratio stays honest. This keeps every evaluated point inside the
//! bounds (lens evaluators must never see e.g. negative thicknesses), keeps
//! the variables in their natural units (no `sin`/`tanh` transforms that
//! flatten gradients at the bounds), and reduces to plain LM in the interior.
//!
//! **Fixed and failing variables.** A variable with equal bounds is pinned:
//! it gets a zero Jacobian column and never enters the subproblem. A variable
//! whose finite-difference probes fail on *both* sides (after the one-sided
//! retry of [`crate::deriv::fd_jacobian`]) is frozen for that iteration; the
//! predicted reduction is formed over the free columns only, so a bad column
//! can never poison the step. A run that would stop on a convergence test
//! while such columns exist reports [`Termination::DerivativeFailure`]
//! instead (stationarity is not certified for those variables), listing them
//! in `OptimResult::failed_variables`.
//!
//! **Budget.** `max_evaluations` is a hard cap: a Jacobian batch is only
//! evaluated when it and one trial point still fit.
//!
//! **Constraints: augmented Lagrangian (PHR).** For each normalised constraint
//! row (`h = (c − t)/s = 0`, or `g ≤ 0` built from either side of a range)
//! the residual vector is extended with
//! `√μ · (h + y/2μ)` and `√μ · max(0, g + y/2μ)`, so the inner problem is
//! still a least-squares problem minimising
//! `Φ + Σ (y h + μ h²) + Σ (1/4μ)(max(0, y + 2μ g)² − y²)`. After each inner
//! solve the multipliers are updated (`y ← y + 2μh`, `y ← max(0, y + 2μg)`) if
//! the violation fell below the current threshold `η`, otherwise `μ` grows
//! (Conn, Gould & Toint; Nocedal & Wright Alg. 17.4). Unlike a pure penalty,
//! `μ` stays bounded, so the inner problems stay well conditioned, and the
//! converged multipliers are returned as `OptimResult::multipliers`.
//!
//! **End-point restoration.** A constrained run that ends just outside the
//! feasibility tolerance takes up to three bounded constraint-only
//! Gauss–Newton corrections (`docs/OPTIMIZATION.md`).

use serde::{Deserialize, Serialize};

use mklight_core::{JobContext, MkError, MkResult};

use crate::deriv::{
    Jacobian, JacobianSettings, analytic_jacobian, fd_first_batch, fd_jacobian_prefetched,
};
use crate::linalg::{Matrix, Qr, dot, norm2, solve_upper, solve_upper_transpose};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{
    BatchEvaluation, CandidateBatch, Evaluation, Problem, ScalarEval, check_problem, merit,
    project, total_violation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LsqMethod {
    #[default]
    LevenbergMarquardt,
    Dogleg,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AugLagSettings {
    /// Initial penalty `μ₀` (on constraints normalised by their `scale`).
    pub mu0: f64,
    pub mu_growth: f64,
    pub mu_max: f64,
    pub max_outer: usize,
    /// Max normalised violation regarded as feasible.
    pub tol: f64,
    /// Iterations an inner solve may take while its point is infeasible
    /// before the multipliers / penalty are updated (`None`: until the inner
    /// solve converges). Inner solves in narrow valleys can converge slowly;
    /// without a cap the first round may use the whole iteration budget at
    /// `μ₀` and end infeasible. Rounds at a feasible point always run to
    /// convergence.
    #[serde(default)]
    pub max_inner_iterations: Option<usize>,
}

impl Default for AugLagSettings {
    fn default() -> Self {
        AugLagSettings {
            mu0: 10.0,
            mu_growth: 10.0,
            mu_max: 1e12,
            max_outer: 30,
            tol: 1e-9,
            max_inner_iterations: None,
        }
    }
}

impl AugLagSettings {
    /// Reject penalty settings the augmented-Lagrangian loop cannot run with.
    pub fn validate(&self) -> MkResult<()> {
        let invalid = |what: &str| Err(MkError::InvalidParam(format!("constraints: {what}")));
        if !(self.mu0 > 0.0 && self.mu0.is_finite()) {
            return invalid("mu0 must be finite and > 0");
        }
        if !(self.mu_growth > 1.0 && self.mu_growth.is_finite()) {
            return invalid("mu_growth must be finite and > 1");
        }
        if !(self.mu_max > 0.0) {
            return invalid("mu_max must be > 0");
        }
        if !(self.tol >= 0.0) {
            return invalid("tol must be ≥ 0");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LsqSettings {
    pub method: LsqMethod,
    /// LM only.
    pub geodesic_acceleration: bool,
    /// Jacobian evaluations (outer trust-region iterations), summed over all
    /// augmented-Lagrangian rounds.
    pub max_iterations: usize,
    pub max_evaluations: usize,
    /// Stop when actual and predicted relative merit reductions are ≤ ftol.
    pub ftol: f64,
    /// Stop when `Δ ≤ xtol · ‖D x‖`.
    pub xtol: f64,
    /// Stop when `max_j |J_jᵀ r| / (‖J_j‖ ‖r‖) ≤ gtol` over free variables.
    pub gtol: f64,
    /// Stop when the (feasible) merit is ≤ this value.
    pub target_merit: f64,
    /// `Δ₀ = factor · ‖D x₀‖` (or `factor` when that is zero).
    pub initial_trust_factor: f64,
    pub jacobian: JacobianSettings,
    pub constraints: AugLagSettings,
    /// Evaluate a trial point in one batch with the first-pass probes of the
    /// finite-difference Jacobian at it (`deriv::fd_first_batch`) when the
    /// previous trial point was accepted: an accepted step then has its next
    /// Jacobian already, so an iteration costs one batch instead of two; a
    /// rejected step wastes the probes (at most one batch per accepted
    /// step). Set by callers (the engine, for runs whose device evaluator
    /// aims on the device), not by settings.
    #[serde(skip)]
    pub speculative_jacobian: bool,
}

impl Default for LsqSettings {
    fn default() -> Self {
        LsqSettings {
            method: LsqMethod::LevenbergMarquardt,
            geodesic_acceleration: false,
            max_iterations: 500,
            max_evaluations: 100_000,
            ftol: 1e-14,
            xtol: 1e-12,
            gtol: 1e-12,
            target_merit: 0.0,
            initial_trust_factor: 100.0,
            jacobian: JacobianSettings::default(),
            constraints: AugLagSettings::default(),
            speculative_jacobian: false,
        }
    }
}

impl LsqSettings {
    pub fn dogleg() -> Self {
        LsqSettings {
            method: LsqMethod::Dogleg,
            ..Default::default()
        }
    }
    fn name(&self) -> &'static str {
        match self.method {
            LsqMethod::LevenbergMarquardt => "lm",
            LsqMethod::Dogleg => "dogleg",
        }
    }
    /// Reject values the solver cannot run with: a trust factor that is not
    /// finite and positive, negative (or NaN) tolerances, and the penalty
    /// settings [`AugLagSettings::validate`] rejects.
    pub fn validate(&self) -> MkResult<()> {
        if !(self.initial_trust_factor > 0.0 && self.initial_trust_factor.is_finite()) {
            return Err(MkError::InvalidParam(
                "initial_trust_factor must be finite and > 0".into(),
            ));
        }
        for (name, tol) in [
            ("ftol", self.ftol),
            ("xtol", self.xtol),
            ("gtol", self.gtol),
        ] {
            if !(tol >= 0.0) {
                return Err(MkError::InvalidParam(format!("{name} must be ≥ 0")));
            }
        }
        self.constraints.validate()
    }
}

#[derive(Clone, Copy, Debug)]
enum RowKind {
    Eq,
    Upper,
    Lower,
}

#[derive(Clone, Copy, Debug)]
struct ConsRow {
    k: usize,
    kind: RowKind,
    bound: f64,
    inv_scale: f64,
}

/// Augmented-Lagrangian residual map: raw evaluation → least-squares rows.
#[derive(Clone, Debug)]
struct Aug {
    sqrt_w: Vec<f64>,
    rows: Vec<ConsRow>,
    y: Vec<f64>,
    mu: f64,
}

impl Aug {
    fn new<P: Problem + ?Sized>(p: &P, mu: f64) -> Aug {
        let mut rows = Vec::new();
        for (k, c) in p.constraints().iter().enumerate() {
            let inv_scale = 1.0 / c.safe_scale();
            if c.is_equality() {
                rows.push(ConsRow {
                    k,
                    kind: RowKind::Eq,
                    bound: c.lower.unwrap_or(0.0),
                    inv_scale,
                });
                continue;
            }
            if let Some(hi) = c.upper {
                rows.push(ConsRow {
                    k,
                    kind: RowKind::Upper,
                    bound: hi,
                    inv_scale,
                });
            }
            if let Some(lo) = c.lower {
                rows.push(ConsRow {
                    k,
                    kind: RowKind::Lower,
                    bound: lo,
                    inv_scale,
                });
            }
        }
        Aug {
            sqrt_w: p.weights().iter().map(|w| w.sqrt()).collect(),
            y: vec![0.0; rows.len()],
            rows,
            mu,
        }
    }
    fn n_rows(&self) -> usize {
        self.sqrt_w.len() + self.rows.len()
    }
    /// Normalised constraint value (`= 0` or `≤ 0` wanted).
    fn g(row: &ConsRow, c: &[f64]) -> f64 {
        match row.kind {
            RowKind::Eq | RowKind::Upper => (c[row.k] - row.bound) * row.inv_scale,
            RowKind::Lower => (row.bound - c[row.k]) * row.inv_scale,
        }
    }
    fn shifted(&self, i: usize, c: &[f64]) -> f64 {
        Aug::g(&self.rows[i], c) + self.y[i] / (2.0 * self.mu)
    }
    fn residual(&self, e: &Evaluation) -> Option<Vec<f64>> {
        if !e.status.is_usable() {
            return None;
        }
        let sm = self.mu.sqrt();
        let mut r: Vec<f64> = self
            .sqrt_w
            .iter()
            .zip(&e.residuals)
            .map(|(s, v)| s * v)
            .collect();
        for (i, row) in self.rows.iter().enumerate() {
            let v = self.shifted(i, &e.constraints);
            r.push(match row.kind {
                RowKind::Eq => sm * v,
                _ => sm * v.max(0.0),
            });
        }
        if r.iter().all(|v| v.is_finite()) {
            Some(r)
        } else {
            None
        }
    }
    fn jacobian(&self, e: &Evaluation, jr: &Matrix, jc: &Matrix) -> Matrix {
        let n = jr.cols;
        let m = self.sqrt_w.len();
        let mut j = Matrix::zeros(self.n_rows(), n);
        for i in 0..m {
            let s = self.sqrt_w[i];
            for (o, v) in j.row_mut(i).iter_mut().zip(jr.row(i)) {
                *o = s * v;
            }
        }
        let sm = self.mu.sqrt();
        for (i, row) in self.rows.iter().enumerate() {
            let active = match row.kind {
                RowKind::Eq => true,
                _ => self.shifted(i, &e.constraints) > 0.0,
            };
            if !active {
                continue;
            }
            let f = sm
                * row.inv_scale
                * match row.kind {
                    RowKind::Lower => -1.0,
                    _ => 1.0,
                };
            for (o, v) in j.row_mut(m + i).iter_mut().zip(jc.row(row.k)) {
                *o = f * v;
            }
        }
        j
    }
    fn update_multipliers(&mut self, c: &[f64]) {
        for i in 0..self.rows.len() {
            let g = Aug::g(&self.rows[i], c);
            let y = self.y[i] + 2.0 * self.mu * g;
            self.y[i] = match self.rows[i].kind {
                RowKind::Eq => y,
                _ => y.max(0.0),
            };
        }
    }
    /// Multipliers per problem constraint for `∇Φ + Σ λ_k ∇c_k = 0`.
    fn multipliers(&self, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; k];
        for (row, y) in self.rows.iter().zip(&self.y) {
            out[row.k] += y
                * row.inv_scale
                * match row.kind {
                    RowKind::Lower => -1.0,
                    _ => 1.0,
                };
        }
        out
    }
}

fn scalar<P: Problem + ?Sized>(p: &P, e: &Evaluation) -> ScalarEval {
    let m = merit(p.weights(), &e.residuals, e.status);
    if !m.is_finite() {
        return ScalarEval::FAILED;
    }
    ScalarEval {
        merit: m,
        violation: total_violation(p.constraints(), &e.constraints, e.status),
        status: e.status,
    }
}

fn max_violation<P: Problem + ?Sized>(p: &P, c: &[f64]) -> f64 {
    p.constraints()
        .iter()
        .zip(c)
        .map(|(k, v)| k.violation(*v))
        .fold(0.0, f64::max)
}

#[derive(Default)]
struct Counters {
    evaluations: usize,
    iterations: usize,
}

struct Ctx<'a, P: Problem + ?Sized> {
    p: &'a P,
    s: &'a LsqSettings,
    mon: &'a Monitor,
    ctr: Counters,
    history: Vec<IterationRecord>,
    /// Moré scaling, persistent across iterations and AL rounds.
    d: Vec<f64>,
    /// Warm-start LM parameter.
    lambda: f64,
    /// Failed Jacobian columns at the current iterate.
    failed: Vec<usize>,
    /// A speculative trial point's first Jacobian pass: the point, the
    /// batch and its evaluation.
    speculated: Option<(Vec<f64>, CandidateBatch, BatchEvaluation)>,
    /// Was the last trial point accepted? Speculation follows acceptance:
    /// while steps are being rejected, each rejected speculative trial would
    /// waste a whole probe batch.
    last_accepted: bool,
    /// The problem supplies its Jacobian, so a trial point needs no
    /// finite-difference probes.
    analytic: bool,
}

impl<P: Problem + ?Sized> Ctx<'_, P> {
    fn eval(&mut self, x: &[f64]) -> MkResult<Evaluation> {
        let b = CandidateBatch::from_points(x.len(), [x]);
        let e = self.p.evaluate_batch(&b, &self.mon.job)?;
        e.validate(1, self.p.num_residuals(), self.p.num_constraints())?;
        self.ctr.evaluations += 1;
        let ev = e.get(0);
        self.mon.offer(x, scalar(self.p, &ev));
        Ok(ev)
    }
    /// A trial point; with `speculative_jacobian`, evaluated in one batch
    /// with the first pass of the Jacobian at it, which is kept for
    /// [`Self::jacobian`]. Every evaluation performed counts.
    fn eval_trial(&mut self, x: &[f64]) -> MkResult<Evaluation> {
        self.speculated = None;
        if !self.s.speculative_jacobian || !self.last_accepted || self.analytic {
            return self.eval(x);
        }
        let Some(first) = fd_first_batch(self.p, x, &self.s.jacobian) else {
            return self.eval(x);
        };
        // The trial point is the anchored pass's base row, or goes first.
        let anchored = self.p.anchored_differences();
        let mut batch = CandidateBatch::with_capacity(x.len(), first.len() + 1);
        batch.anchor = first.anchor.clone();
        if !anchored {
            batch.push(x);
        }
        for k in 0..first.len() {
            batch.push(first.point(k));
        }
        if self.ctr.evaluations + batch.len() > self.s.max_evaluations {
            return self.eval(x);
        }
        let (m, k) = (self.p.num_residuals(), self.p.num_constraints());
        let e = self.p.evaluate_batch(&batch, &self.mon.job)?;
        e.validate(batch.len(), m, k)?;
        self.ctr.evaluations += batch.len();
        let ev = e.get(0);
        self.mon.offer(x, scalar(self.p, &ev));
        let probes = if anchored {
            e
        } else {
            let mut rest = BatchEvaluation::zeros(first.len(), m, k);
            rest.residuals.copy_from_slice(&e.residuals[m..]);
            rest.constraints.copy_from_slice(&e.constraints[k..]);
            rest.status.copy_from_slice(&e.status[1..]);
            rest
        };
        self.speculated = Some((x.to_vec(), first, probes));
        Ok(ev)
    }
    /// The Jacobian at `x` (`e` its evaluation): analytic when the problem
    /// provides one and the settings allow it, otherwise finite differences
    /// within the evaluation budget, from a speculative first pass when one
    /// was taken at `x`.
    fn jacobian(&mut self, x: &[f64], e: &Evaluation) -> MkResult<Option<Jacobian>> {
        let prefetched = self
            .speculated
            .take()
            .filter(|(at, _, _)| at.as_slice() == x)
            .map(|(_, b, e)| (b, e));
        if let Some(j) = analytic_jacobian(self.p, x, &self.s.jacobian, &self.mon.job)? {
            self.analytic = true;
            return Ok(Some(j));
        }
        let budget = self
            .s
            .max_evaluations
            .saturating_sub(self.ctr.evaluations + 1);
        fd_jacobian_prefetched(
            self.p,
            x,
            e,
            &self.s.jacobian,
            &self.mon.job,
            budget,
            prefetched,
        )
    }
    fn budget_left(&self) -> bool {
        self.ctr.evaluations < self.s.max_evaluations
    }
    /// A convergence verdict is downgraded while derivative columns fail.
    fn certify(&self, t: Termination) -> Termination {
        if t.is_converged() && !self.failed.is_empty() {
            Termination::DerivativeFailure
        } else {
            t
        }
    }
}

/// `J p` over the free columns only (`p_free[k]` belongs to `free[k]`).
fn mul_free(j: &Matrix, free: &[usize], p_free: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; j.rows];
    for (&c, &pc) in free.iter().zip(p_free) {
        if pc != 0.0 {
            for (i, o) in out.iter_mut().enumerate() {
                *o += j[(i, c)] * pc;
            }
        }
    }
    out
}

/// Jacobian size `m·n` from which the resident tall QR beats the host
/// Householder QR (measured on an Apple M2 Pro, GPU.md §9.2 "QR policy":
/// host faster up to 400×100, device from 1000×100, 4000×32 and 2000×60).
const RESIDENT_QR_MIN_ELEMENTS: usize = 1 << 16;

/// Compressed free-variable subproblem: `min ‖A₀ p + b₀‖` with
/// `A₀ᵀA₀ = J_Fᵀ J_F`, `A₀ᵀ b₀ = J_Fᵀ r`.
struct Sub {
    resident_q: Option<crate::gpu::ResidentQr>,
    a0: Matrix,
    b0: Vec<f64>,
    qr: Option<Qr>,
    d: Vec<f64>,
    g: Vec<f64>,
    gn: Option<Vec<f64>>,
}

impl Sub {
    fn new(jf: Matrix, r: &[f64], d: Vec<f64>) -> Sub {
        let g = jf.tr_mul_vec(r);
        let nf = jf.cols;
        if jf.rows >= nf {
            let qr = Qr::new(jf);
            let mut qtr = r.to_vec();
            qr.apply_qt(&mut qtr);
            qtr.truncate(nf);
            let gn = if qr.is_full_rank() {
                let a = qr.r();
                let neg: Vec<f64> = qtr.iter().map(|v| -v).collect();
                Some(solve_upper(&a, &neg))
            } else {
                None
            };
            Sub {
                resident_q: None,
                a0: qr.r(),
                b0: qtr,
                qr: Some(qr),
                d,
                g,
                gn,
            }
        } else {
            Sub {
                resident_q: None,
                a0: jf,
                b0: r.to_vec(),
                qr: None,
                d,
                g,
                gn: None,
            }
        }
    }
    fn prepare(
        j: Matrix,
        r: &[f64],
        d: Vec<f64>,
        gpu: Option<&crate::gpu::GpuContext>,
        ctx: &JobContext,
    ) -> MkResult<Sub> {
        if let Some(gpu) =
            gpu.filter(|_| j.rows >= j.cols && j.rows * j.cols >= RESIDENT_QR_MIN_ELEMENTS)
            && let Some((resident_q, reduced, mut qtr)) = gpu.reduce_qr(&j, r, ctx)?
        {
            let g = reduced.tr_mul_vec(&qtr);
            let qr = Qr::new(reduced);
            qr.apply_qt(&mut qtr);
            let a0 = qr.r();
            let gn = qr
                .is_full_rank()
                .then(|| solve_upper(&a0, &qtr.iter().map(|v| -v).collect::<Vec<_>>()));
            return Ok(Sub {
                resident_q: Some(resident_q),
                a0,
                b0: qtr,
                qr: Some(qr),
                d,
                g,
                gn,
            });
        }
        Ok(Self::new(j, r, d))
    }
    fn transform(&self, v: &[f64], ctx: &JobContext) -> MkResult<Vec<f64>> {
        let projected = self
            .resident_q
            .as_ref()
            .map(|q| q.project(v, ctx))
            .transpose()?;
        let v = projected.as_deref().unwrap_or(v);
        match &self.qr {
            Some(q) => {
                let mut t = v.to_vec();
                q.apply_qt(&mut t);
                t.truncate(self.a0.cols);
                Ok(t)
            }
            None => Ok(v.to_vec()),
        }
    }
    fn dnorm(&self, p: &[f64]) -> f64 {
        let v: Vec<f64> = p.iter().zip(&self.d).map(|(a, b)| a * b).collect();
        norm2(&v)
    }
    /// `p(λ)` for `λ > 0` and the triangular factor of `[A₀; √λ D]`.
    fn solve_lambda(&self, lambda: f64, rhs: &[f64]) -> (Vec<f64>, Matrix) {
        let nf = self.a0.cols;
        let ra = self.a0.rows;
        let mut a = Matrix::zeros(ra + nf, nf);
        a.data[..ra * nf].copy_from_slice(&self.a0.data);
        let sl = lambda.sqrt();
        for j in 0..nf {
            a[(ra + j, j)] = sl * self.d[j];
        }
        let mut b = vec![0.0; ra + nf];
        for (o, v) in b.iter_mut().zip(rhs) {
            *o = -v;
        }
        let qr = Qr::new(a);
        qr.apply_qt(&mut b);
        let r = qr.r();
        (solve_upper(&r, &b[..nf]), r)
    }
    /// Moré's LM step for trust radius `delta`; returns `(p, λ)`.
    fn lm_step(&self, delta: f64, lambda0: f64) -> (Vec<f64>, f64) {
        let nf = self.a0.cols;
        let gs: Vec<f64> = self.g.iter().zip(&self.d).map(|(g, d)| g / d).collect();
        let gnorm = norm2(&gs);
        let mut lo = 0.0;
        if let Some(gn) = &self.gn {
            let dn = self.dnorm(gn);
            if dn <= 1.1 * delta {
                return (gn.clone(), 0.0);
            }
            // Newton lower bound from λ = 0 (convexity of 1/‖Dp(λ)‖).
            let w: Vec<f64> = (0..nf).map(|j| self.d[j] * self.d[j] * gn[j]).collect();
            let wt = solve_upper_transpose(&self.a0, &w);
            let ww = dot(&wt, &wt);
            if ww > 0.0 && ww.is_finite() {
                lo = ((dn - delta) / delta) * dn * dn / ww;
            }
        }
        let mut hi = gnorm / delta;
        if !(hi > 0.0) || !hi.is_finite() {
            hi = f64::MAX.sqrt();
        }
        // Not `clamp`: the bounds can cross (or be NaN) for a degenerate radius.
        let mut lambda = lambda0.max(lo).min(hi);
        if lambda == 0.0 {
            lambda = gnorm / delta.max(f64::MIN_POSITIVE);
        }
        let mut best = None;
        for _ in 0..12 {
            if lambda <= lo || lambda >= hi {
                lambda = (1e-3 * hi).max((lo * hi).sqrt());
            }
            if !(lambda > 0.0) {
                lambda = f64::MIN_POSITIVE.sqrt();
            }
            let (p, r) = self.solve_lambda(lambda, &self.b0);
            let dp = self.dnorm(&p);
            let phi = dp - delta;
            let done = phi.abs() <= 0.1 * delta;
            best = Some((p.clone(), lambda));
            if done || dp == 0.0 {
                break;
            }
            if phi > 0.0 {
                lo = lo.max(lambda);
            } else {
                hi = hi.min(lambda);
            }
            let w: Vec<f64> = (0..nf).map(|j| self.d[j] * self.d[j] * p[j] / dp).collect();
            let wt = solve_upper_transpose(&r, &w);
            let ww = dot(&wt, &wt);
            if !(ww > 0.0) || !ww.is_finite() {
                break;
            }
            lambda = lo.max(lambda + (phi / delta) / ww);
        }
        best.unwrap_or((vec![0.0; nf], lambda))
    }
    /// Powell dogleg step, and whether it is the full Gauss–Newton step
    /// (which lies inside the trust region).
    fn dogleg_step(&self, delta: f64) -> (Vec<f64>, bool) {
        let nf = self.a0.cols;
        let gn = match &self.gn {
            Some(g) => g.clone(),
            None => {
                let dmax = self.d.iter().fold(0.0_f64, |m, v| m.max(*v));
                self.solve_lambda((1e-10 * dmax * dmax).max(f64::MIN_POSITIVE), &self.b0)
                    .0
            }
        };
        if self.dnorm(&gn) <= delta {
            return (gn, true);
        }
        let s: Vec<f64> = (0..nf)
            .map(|j| -self.g[j] / (self.d[j] * self.d[j]))
            .collect();
        let js = self.a0.mul_vec(&s);
        let gsd = -dot(&self.g, &s); // ‖D⁻¹g‖²
        let jj = dot(&js, &js);
        let ds = self.dnorm(&s);
        if !(jj > 0.0) || ds == 0.0 {
            let p = s
                .iter()
                .map(|v| v * delta / ds.max(f64::MIN_POSITIVE))
                .collect();
            return (p, false);
        }
        let alpha = gsd / jj;
        if alpha * ds >= delta {
            return (s.iter().map(|v| v * delta / ds).collect(), false);
        }
        let a: Vec<f64> = s.iter().map(|v| alpha * v).collect();
        let b: Vec<f64> = gn.iter().zip(&a).map(|(g, a)| g - a).collect();
        let da: Vec<f64> = a.iter().zip(&self.d).map(|(x, d)| x * d).collect();
        let db: Vec<f64> = b.iter().zip(&self.d).map(|(x, d)| x * d).collect();
        let qa = dot(&db, &db);
        let qb = 2.0 * dot(&da, &db);
        let qc = dot(&da, &da) - delta * delta;
        let disc = (qb * qb - 4.0 * qa * qc).max(0.0).sqrt();
        // Stable root of qa τ² + qb τ + qc = 0 with qc < 0 (τ ∈ (0, 1]).
        let tau = if qb > 0.0 {
            -2.0 * qc / (qb + disc)
        } else {
            (-qb + disc) / (2.0 * qa)
        };
        let tau = tau.clamp(0.0, 1.0);
        (a.iter().zip(&b).map(|(a, b)| a + tau * b).collect(), false)
    }
}

struct InnerOut {
    x: Vec<f64>,
    e: Evaluation,
    termination: Termination,
    /// Ended by `AugLagSettings::max_inner_iterations` (not converged, but
    /// the run goes on with updated multipliers).
    capped: bool,
}

/// One trust-region solve of the current augmented least-squares problem.
fn inner<P: Problem + ?Sized>(
    c: &mut Ctx<'_, P>,
    aug: &Aug,
    mut x: Vec<f64>,
    mut e: Evaluation,
    stage: &str,
    cap: Option<usize>,
) -> MkResult<InnerOut> {
    let vars = c.p.variables();
    let start_iteration = c.ctr.iterations;
    let n = vars.len();
    let s = c.s;
    let mut r = match aug.residual(&e) {
        Some(r) => r,
        None => {
            return Ok(InnerOut {
                x,
                e,
                termination: Termination::EvaluationFailure,
                capped: false,
            });
        }
    };
    let mut phi = dot(&r, &r);
    let mut delta = 0.0;
    let mut first = true;
    fn out(x: Vec<f64>, e: Evaluation, termination: Termination) -> MkResult<InnerOut> {
        Ok(InnerOut {
            x,
            e,
            termination,
            capped: false,
        })
    }
    loop {
        if c.mon.cancelled() {
            return out(x, e, Termination::Cancelled);
        }
        if c.ctr.iterations >= s.max_iterations {
            return out(x, e, Termination::MaxIterations);
        }
        if !c.budget_left() {
            return out(x, e, Termination::MaxEvaluations);
        }
        if let Some(cap) = cap
            && c.ctr.iterations - start_iteration >= cap
            && max_violation(c.p, &e.constraints) > s.constraints.tol
        {
            return Ok(InnerOut {
                x,
                e,
                termination: Termination::MaxIterations,
                capped: true,
            });
        }
        let sc = scalar(c.p, &e);
        if sc.merit <= s.target_merit && sc.violation <= c.mon.feasibility_tol {
            return out(x, e, Termination::TargetReached);
        }
        // Jacobian (keeping one evaluation for the trial point).
        let jac = match c.jacobian(&x, &e) {
            Ok(Some(j)) => j,
            Ok(None) => return out(x, e, Termination::MaxEvaluations),
            Err(MkError::Cancelled) => return out(x, e, Termination::Cancelled),
            Err(err) => return Err(err),
        };
        c.ctr.evaluations += jac.evaluations;
        c.ctr.iterations += 1;
        let mut jm = aug.jacobian(&e, &jac.residuals, &jac.constraints);
        // Failed (or non-finite) columns are zeroed and frozen.
        let mut failed = jac.failed_columns.clone();
        for j in 0..n {
            if !failed.contains(&j) && (0..jm.rows).any(|i| !jm[(i, j)].is_finite()) {
                failed.push(j);
            }
        }
        failed.sort_unstable();
        for &j in &failed {
            for i in 0..jm.rows {
                jm[(i, j)] = 0.0;
            }
        }
        c.failed = failed;
        let cn = jm.column_norms();
        // Host-owned J and r: a device product costs one fill of J plus a
        // command, more than the product itself (measured, OPTIMIZATION.md).
        let g = jm.tr_mul_vec(&r);
        for j in 0..n {
            if cn[j] > c.d[j] {
                c.d[j] = cn[j];
            }
            if !(c.d[j] > 0.0) {
                c.d[j] = 1.0 / vars[j].safe_scale();
            }
        }
        // Free set: not pinned at a bound by an outward gradient, not failed.
        let free: Vec<usize> = (0..n)
            .filter(|&j| {
                let v = &vars[j];
                !c.failed.contains(&j)
                    && !(x[j] <= v.lo() && g[j] > 0.0)
                    && !(x[j] >= v.hi() && g[j] < 0.0)
                    && !(v.lo() == v.hi())
            })
            .collect();
        let rnorm = phi.sqrt();
        let gcos = free
            .iter()
            .map(|&j| {
                if cn[j] > 0.0 && rnorm > 0.0 {
                    g[j].abs() / (cn[j] * rnorm)
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max);
        let dx: Vec<f64> = (0..n).map(|j| c.d[j] * x[j]).collect();
        let xnorm = norm2(&dx);
        if first {
            delta = s.initial_trust_factor * if xnorm > 0.0 { xnorm } else { 1.0 };
        }
        if phi == 0.0 {
            return out(x, e, Termination::SmallGradient);
        }
        if free.is_empty() || gcos <= s.gtol {
            return out(x, e, c.certify(Termination::SmallGradient));
        }
        let df: Vec<f64> = free.iter().map(|&j| c.d[j]).collect();
        let sub = match Sub::prepare(
            jm.select_columns(&free),
            &r,
            df,
            c.p.gpu_context(),
            &c.mon.job,
        ) {
            Ok(sub) => sub,
            Err(MkError::Cancelled) => return out(x, e, Termination::Cancelled),
            Err(e) => return Err(e),
        };

        // Inner loop: find an acceptable step with this Jacobian.
        loop {
            if c.mon.cancelled() {
                return out(x, e, Termination::Cancelled);
            }
            if !c.budget_left() {
                return out(x, e, Termination::MaxEvaluations);
            }
            // `gauss_newton`: the full Gauss–Newton step was taken.
            let (mut pf, lambda, gauss_newton) = match s.method {
                LsqMethod::LevenbergMarquardt => {
                    let (p, lambda) = sub.lm_step(delta, c.lambda);
                    (p, lambda, lambda == 0.0)
                }
                LsqMethod::Dogleg => {
                    let (p, gauss_newton) = sub.dogleg_step(delta);
                    (p, 0.0, gauss_newton)
                }
            };
            // Optional geodesic acceleration.
            if s.geodesic_acceleration
                && s.method == LsqMethod::LevenbergMarquardt
                && c.ctr.evaluations + 2 <= s.max_evaluations
            {
                let h = 0.1;
                let mut xv = x.clone();
                for (k, &j) in free.iter().enumerate() {
                    xv[j] += h * pf[k];
                }
                let inside = xv.iter().zip(vars).all(|(v, var)| *v == var.clamp(*v));
                if inside {
                    let ev = match c.eval(&xv) {
                        Ok(v) => v,
                        Err(MkError::Cancelled) => return out(x, e, Termination::Cancelled),
                        Err(err) => return Err(err),
                    };
                    if let Some(rv) = aug.residual(&ev) {
                        let jv = mul_free(&jm, &free, &pf);
                        let rvv: Vec<f64> = (0..rv.len())
                            .map(|i| (2.0 / h) * ((rv[i] - r[i]) / h - jv[i]))
                            .collect();
                        let rhs = sub.transform(&rvv, &c.mon.job)?;
                        let acc = if lambda > 0.0 {
                            sub.solve_lambda(lambda, &rhs).0
                        } else {
                            let neg: Vec<f64> = rhs.iter().map(|v| -v).collect();
                            solve_upper(&sub.a0, &neg)
                        };
                        if acc.iter().all(|v| v.is_finite())
                            && 2.0 * sub.dnorm(&acc) <= 0.75 * sub.dnorm(&pf)
                        {
                            for (p, a) in pf.iter_mut().zip(&acc) {
                                *p += 0.5 * a;
                            }
                        }
                    }
                }
            }
            // Projected trial point and the step actually taken.
            let mut xt = x.clone();
            for (k, &j) in free.iter().enumerate() {
                xt[j] += pf[k];
            }
            project(vars, &mut xt);
            let p: Vec<f64> = xt.iter().zip(&x).map(|(a, b)| a - b).collect();
            let dp: Vec<f64> = p.iter().zip(&c.d).map(|(a, b)| a * b).collect();
            let pnorm = norm2(&dp);
            if first {
                delta = delta.min(pnorm.max(f64::MIN_POSITIVE));
            }
            // Only free columns move (p_j = 0 elsewhere); never touch the others.
            let p_free: Vec<f64> = free.iter().map(|&j| p[j]).collect();
            let jp = mul_free(&jm, &free, &p_free);
            // pred = ‖r‖² − ‖r + Jp‖² = −(2 rᵀJp + ‖Jp‖²)
            let pred = -(2.0 * dot(&r, &jp) + dot(&jp, &jp));
            let (et, rt) = if pnorm > 0.0 {
                let et = match c.eval_trial(&xt) {
                    Ok(v) => v,
                    Err(MkError::Cancelled) => return out(x, e, Termination::Cancelled),
                    Err(err) => return Err(err),
                };
                let rt = aug.residual(&et);
                (Some(et), rt)
            } else {
                (None, None)
            };
            let phit = rt.as_ref().map_or(f64::INFINITY, |v| dot(v, v));
            let act = phi - phit;
            let rho = if pred > 0.0 && act.is_finite() {
                act / pred
            } else {
                -1.0
            };
            // Trust-region update (MINPACK).
            if rho < 0.25 {
                delta = 0.25 * delta.min(pnorm.max(f64::MIN_POSITIVE));
                c.lambda = lambda * 2.0;
            } else if rho >= 0.75 || gauss_newton {
                delta = 2.0 * pnorm;
                c.lambda = 0.5 * lambda;
            } else {
                c.lambda = lambda;
            }
            first = false;
            let accepted = rho > 1e-4;
            c.last_accepted = accepted;
            if accepted {
                let phi_old = phi;
                x = xt;
                e = et.expect("evaluated");
                r = rt.expect("usable");
                phi = phit;
                let sc = scalar(c.p, &e);
                let mut rec = IterationRecord::new(s.name(), c.ctr.iterations, c.ctr.evaluations);
                rec.stage = stage.to_string();
                rec.merit = sc.merit;
                rec.violation = sc.violation;
                rec.best_merit = c.mon.best.get().map_or(sc.merit, |b| b.merit);
                rec.trust_radius = Some(delta);
                rec.damping = Some(lambda);
                rec.gain_ratio = Some(rho);
                rec.step_norm = Some(pnorm);
                rec.gradient_norm = Some(gcos);
                rec.failed_variables = c.failed.clone();
                c.mon.record(&mut c.history, rec);
                c.mon.progress(
                    c.ctr.iterations as f64 / s.max_iterations.max(1) as f64,
                    "least squares",
                );
                let rel_act = act.abs() / phi_old.max(f64::MIN_POSITIVE);
                let rel_pred = pred / phi_old.max(f64::MIN_POSITIVE);
                if rel_act <= s.ftol && rel_pred <= s.ftol && rho <= 2.0 {
                    return out(x, e, c.certify(Termination::SmallMeritChange));
                }
                let dx: Vec<f64> = (0..n).map(|j| c.d[j] * x[j]).collect();
                let xn = norm2(&dx);
                if delta <= s.xtol * xn || pnorm <= s.xtol * xn {
                    return out(x, e, c.certify(Termination::SmallStep));
                }
                break;
            }
            let xn = norm2(&dx);
            if delta <= s.xtol * xn || !(delta > 0.0) || (pnorm == 0.0 && delta < 1e-300) {
                return out(x, e, c.certify(Termination::SmallStep));
            }
        }
    }
}

/// Minimise `Σ w r²` from `x0` subject to the problem's bounds and
/// constraints. Cancellation returns `Ok` with `Termination::Cancelled` and
/// the last accepted iterate; `monitor.best` holds the best point seen.
pub fn least_squares<P: Problem + ?Sized>(
    p: &P,
    x0: &[f64],
    settings: &LsqSettings,
    monitor: &Monitor,
) -> MkResult<OptimResult> {
    check_problem(p, x0)?;
    settings.validate()?;
    monitor.begin_run();
    let vars = p.variables();
    let n = vars.len();
    let mut c = Ctx {
        p,
        s: settings,
        mon: monitor,
        ctr: Counters::default(),
        history: Vec::new(),
        d: vec![0.0; n],
        lambda: 0.0,
        failed: Vec::new(),
        speculated: None,
        last_accepted: true,
        analytic: false,
    };
    let mut x = x0.to_vec();
    project(vars, &mut x);
    let mut e = match c.eval(&x) {
        Ok(e) => e,
        Err(MkError::Cancelled) => {
            return finish(c, x, None, Termination::Cancelled, None);
        }
        Err(err) => return Err(err),
    };
    if !e.status.is_usable() {
        return finish(c, x, Some(e), Termination::EvaluationFailure, None);
    }
    let al = settings.constraints;
    let mut aug = Aug::new(p, al.mu0);
    if aug.rows.is_empty() {
        let o = inner(&mut c, &aug, x, e, "", None)?;
        return finish(c, o.x, Some(o.e), o.termination, None);
    }
    let k = p.num_constraints();
    let mut eta = 1.0 / al.mu0.powf(0.1);
    let mut termination = Termination::MaxIterations;
    // Rounds ended by the inner-iteration cap do not count against
    // `max_outer` (each takes at least one iteration of the finite budget).
    let mut outer = 0;
    let mut round = 0;
    while outer < al.max_outer {
        let o = inner(
            &mut c,
            &aug,
            x,
            e,
            &format!("al {round}"),
            al.max_inner_iterations.map(|m| m.max(1)),
        )?;
        round += 1;
        x = o.x;
        e = o.e;
        if !o.capped
            && matches!(
                o.termination,
                Termination::Cancelled
                    | Termination::MaxEvaluations
                    | Termination::MaxIterations
                    | Termination::EvaluationFailure
                    | Termination::DerivativeFailure
            )
        {
            termination = o.termination;
            break;
        }
        let viol = max_violation(p, &e.constraints);
        // A point that reached the target is feasible by the monitor's
        // tolerance, which may be looser than `tol`: the run ends there.
        if viol <= al.tol || o.termination == Termination::TargetReached {
            aug.update_multipliers(&e.constraints);
            if o.termination.is_converged() {
                termination = o.termination;
                break;
            }
            eta = (eta / aug.mu.powf(0.9)).max(al.tol);
        } else if viol <= eta {
            aug.update_multipliers(&e.constraints);
            eta = (eta / aug.mu.powf(0.9)).max(al.tol);
        } else {
            aug.mu *= al.mu_growth;
            if aug.mu > al.mu_max {
                termination = Termination::Infeasible;
                break;
            }
            eta = (1.0 / aug.mu.powf(0.1)).max(al.tol);
        }
        if !o.capped {
            outer += 1;
        }
        if outer == al.max_outer {
            termination = if viol <= al.tol {
                Termination::MaxIterations
            } else {
                Termination::Infeasible
            };
        }
    }
    if termination != Termination::Cancelled {
        match restore_feasibility(&mut c, &aug.rows, &mut x, &mut e) {
            Ok(()) => {}
            Err(MkError::Cancelled) => termination = Termination::Cancelled,
            Err(err) => return Err(err),
        }
    }
    let mult = aug.multipliers(k);
    finish(c, x, Some(e), termination, Some(mult))
}

/// End-point restoration (`docs/OPTIMIZATION.md`): while the end point is
/// infeasible by the monitor's tolerance, up to three Gauss–Newton
/// corrections of the violated constraint `rows` alone, each the
/// ridge-regularised minimum-norm step in variable scales, capped at 1e-3
/// scales and halved until the total violation decreases.
fn restore_feasibility<P: Problem + ?Sized>(
    c: &mut Ctx<'_, P>,
    rows: &[ConsRow],
    x: &mut Vec<f64>,
    e: &mut Evaluation,
) -> MkResult<()> {
    let vars = c.p.variables();
    let n = vars.len();
    'corrections: for _ in 0..3 {
        c.mon.job.check()?;
        let violation = scalar(c.p, e).violation;
        if !e.status.is_usable() || violation <= c.mon.feasibility_tol || !c.budget_left() {
            break;
        }
        let Some(jac) = c.jacobian(x, e)? else {
            break;
        };
        c.ctr.evaluations += jac.evaluations;
        // Active rows: every equality, violated inequalities.
        let mut r = vec![0.0; rows.len()];
        let mut j = Matrix::zeros(rows.len(), n);
        for (i, row) in rows.iter().enumerate() {
            let g = Aug::g(row, &e.constraints);
            if matches!(row.kind, RowKind::Eq) || g > 0.0 {
                r[i] = g;
                let f = match row.kind {
                    RowKind::Lower => -row.inv_scale,
                    _ => row.inv_scale,
                };
                for k in 0..n {
                    j[(i, k)] = f * jac.constraints[(row.k, k)] * vars[k].safe_scale();
                }
            }
        }
        // Freeze failed columns and variables a descent step would push out
        // of their bounds.
        let grad = j.tr_mul_vec(&r);
        for k in 0..n {
            if jac.failed_columns.contains(&k)
                || (x[k] <= vars[k].lo() && grad[k] > 0.0)
                || (x[k] >= vars[k].hi() && grad[k] < 0.0)
            {
                for i in 0..j.rows {
                    j[(i, k)] = 0.0;
                }
            }
        }
        // Unit Frobenius norm makes the ridge independent of units.
        let scale = norm2(&j.data);
        if !(scale > 0.0 && scale.is_finite()) {
            break;
        }
        j.data.iter_mut().chain(&mut r).for_each(|v| *v /= scale);
        let sub = Sub::new(j, &r, vec![1.0; n]);
        let step = sub.solve_lambda(f64::EPSILON.sqrt(), &sub.b0).0;
        let len = norm2(&step);
        if !(len > 0.0) {
            break;
        }
        let mut alpha = (1e-3 / len).min(1.0);
        for _ in 0..8 {
            if !c.budget_left() {
                break;
            }
            let trial: Vec<f64> = (0..n)
                .map(|k| vars[k].clamp(x[k] + alpha * vars[k].safe_scale() * step[k]))
                .collect();
            let value = c.eval(&trial)?;
            let sc = scalar(c.p, &value);
            if sc.violation < violation {
                *x = trial;
                *e = value;
                let mut rec = IterationRecord::new(c.s.name(), c.ctr.iterations, c.ctr.evaluations);
                rec.stage = "feasibility restoration".into();
                rec.merit = sc.merit;
                rec.violation = sc.violation;
                rec.best_merit = c.mon.best.get().map_or(sc.merit, |b| b.merit);
                c.mon.record(&mut c.history, rec);
                continue 'corrections;
            }
            alpha *= 0.5;
        }
        break;
    }
    Ok(())
}

fn finish<P: Problem + ?Sized>(
    c: Ctx<'_, P>,
    x: Vec<f64>,
    e: Option<Evaluation>,
    mut termination: Termination,
    multipliers: Option<Vec<f64>>,
) -> MkResult<OptimResult> {
    // Cancellation never starts a replacement calculation; Stop scores the
    // best completed state with the independent reference.
    let e = if termination != Termination::Cancelled {
        match c.p.reference_evaluation(&x, &c.mon.job) {
            Some(Ok(reference)) => Some(reference),
            Some(Err(MkError::Cancelled)) => {
                termination = Termination::Cancelled;
                e
            }
            Some(Err(error)) => return Err(error),
            None => e,
        }
    } else {
        e
    };
    let sc = e.as_ref().map_or(ScalarEval::FAILED, |e| scalar(c.p, e));
    Ok(OptimResult {
        x,
        merit: sc.merit,
        violation: sc.violation,
        termination,
        iterations: c.ctr.iterations,
        evaluations: c.ctr.evaluations,
        multipliers,
        history: c.history,
        failed_variables: c.failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `min ‖p + (1, 1)‖` in unit scales: the Gauss–Newton step is `(−1, −1)`.
    fn unit_sub() -> Sub {
        Sub::new(Matrix::identity(2), &[1.0, 1.0], vec![1.0, 1.0])
    }

    #[test]
    fn lm_step_does_not_panic_when_the_lambda_bounds_cross() {
        // Δ = 0: the Newton lower bound is +∞, above the upper bound.
        let (_, lambda) = unit_sub().lm_step(0.0, 0.0);
        assert!(lambda > 0.0);
    }

    #[test]
    fn dogleg_reports_the_full_gauss_newton_step() {
        let sub = unit_sub();
        let (p, full) = sub.dogleg_step(10.0);
        assert!(full);
        assert!(p.iter().all(|v| (v + 1.0).abs() < 1e-15), "{p:?}");
        // A radius inside the Gauss–Newton step cuts the path short.
        let (p, full) = sub.dogleg_step(0.5);
        assert!(!full);
        assert!((sub.dnorm(&p) - 0.5).abs() < 1e-15, "{p:?}");
    }

    #[test]
    fn settings_are_validated() {
        let ok = LsqSettings::default();
        assert!(ok.validate().is_ok());
        let bad = [
            LsqSettings {
                initial_trust_factor: 0.0,
                ..ok
            },
            LsqSettings {
                initial_trust_factor: f64::NAN,
                ..ok
            },
            LsqSettings { ftol: -1.0, ..ok },
            LsqSettings {
                gtol: f64::NAN,
                ..ok
            },
            LsqSettings {
                constraints: AugLagSettings {
                    mu0: 0.0,
                    ..ok.constraints
                },
                ..ok
            },
            LsqSettings {
                constraints: AugLagSettings {
                    mu_growth: 1.0,
                    ..ok.constraints
                },
                ..ok
            },
        ];
        for s in bad {
            assert!(
                matches!(s.validate(), Err(MkError::InvalidParam(_))),
                "{s:?}"
            );
        }
    }
}
