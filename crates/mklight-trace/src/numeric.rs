//! Error-free transformations, compensated accumulation and the small dense
//! solver used by ray aiming. Every routine here has a one-to-one `float`
//! counterpart in the GPU kernels (CONVENTIONS §7.4–7.5).

use serde::{Deserialize, Serialize};

/// Knuth's TwoSum: `a + b = s + e` exactly, with `s = fl(a + b)`.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// TwoProd via fused multiply-add: `a · b = p + e` exactly, with `p = fl(a·b)`.
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Accurate `b² − a·c` (Kahan's discriminant). The rounding error of `a·c`
/// is recovered with an `fma` and added back, so the result is accurate to a
/// few ulps even when `b² ≈ a·c` (grazing rays, near-tangent intersections).
#[inline]
pub fn discriminant(a: f64, b: f64, c: f64) -> f64 {
    let w = a * c;
    let e = (-a).mul_add(c, w); // w − a·c exactly
    let f = b.mul_add(b, -w);
    f + e
}

/// An unevaluated sum `hi + lo` that accumulates with error-free
/// transformations ("double-double light"). The optical path length of every
/// ray is carried in this form so that OPD (a difference of two nearly equal
/// path lengths) is formed before rounding to one `f64` (CONVENTIONS §7.4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Compensated {
    pub hi: f64,
    pub lo: f64,
}

impl Compensated {
    pub const ZERO: Compensated = Compensated { hi: 0.0, lo: 0.0 };

    #[inline]
    pub fn new(x: f64) -> Self {
        Compensated { hi: x, lo: 0.0 }
    }
    /// Add one `f64` term.
    #[inline]
    pub fn add(&mut self, x: f64) {
        let (s, e) = two_sum(self.hi, x);
        self.hi = s;
        self.lo += e;
    }
    /// Add the exact product `a · b` (index × geometric length).
    #[inline]
    pub fn add_product(&mut self, a: f64, b: f64) {
        let (p, e) = two_prod(a, b);
        self.add(p);
        self.lo += e;
    }
    /// The sum rounded to one `f64`.
    #[inline]
    pub fn value(&self) -> f64 {
        self.hi + self.lo
    }
    /// `self − other`, formed in compensated arithmetic before rounding.
    #[inline]
    pub fn diff(&self, other: &Compensated) -> f64 {
        let (d, e) = two_sum(self.hi, -other.hi);
        d + (e + (self.lo - other.lo))
    }
}

/// 2×2 matrix, row-major: `m[row][col]`.
pub type Mat2 = [[f64; 2]; 2];

/// Solve `m · x = r` by Cramer's rule. `None` when `m` is singular or not finite.
#[inline]
pub fn solve2(m: Mat2, r: [f64; 2]) -> Option<[f64; 2]> {
    let det = m[0][0].mul_add(m[1][1], -(m[0][1] * m[1][0]));
    let scale = m[0][0]
        .abs()
        .max(m[0][1].abs())
        .max(m[1][0].abs())
        .max(m[1][1].abs());
    if !(det.is_finite() && det.abs() > 1e-300 && det.abs() > 1e-14 * scale * scale) {
        return None;
    }
    let x0 = r[0].mul_add(m[1][1], -(m[0][1] * r[1])) / det;
    let x1 = m[0][0].mul_add(r[1], -(r[0] * m[1][0])) / det;
    (x0.is_finite() && x1.is_finite()).then_some([x0, x1])
}

#[inline]
pub(crate) fn norm2d(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

/// How [`newton2`] obtains a *fresh* Jacobian (initially without `j0`, and
/// when a step is singular, slow or rejected). Between fresh Jacobians it
/// always uses Broyden's rank-one update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobianPolicy {
    /// Central differences (four evaluations).
    FiniteDifference,
    /// The objective's analytic Jacobian (one evaluation with `jacobian`
    /// requested); central differences where it is undefined.
    AnalyticRefresh,
    /// The analytic Jacobian at every evaluated point, `x0` included
    /// (Newton's method; `j0` and Broyden updates only where it is
    /// undefined, central differences where `j0` is absent too).
    AnalyticNewton,
}

impl JacobianPolicy {
    #[inline]
    fn analytic(self) -> bool {
        self != JacobianPolicy::FiniteDifference
    }
    #[inline]
    fn newton(self) -> bool {
        self == JacobianPolicy::AnalyticNewton
    }
}

/// The value of an [`Objective`] at a point: `f(x)` and, when requested and
/// the objective can form it, `∂f/∂x`.
pub(crate) type Eval = Option<([f64; 2], Option<Mat2>)>;

/// An objective for [`newton2`]: `f(x, jacobian)` returns `f(x)` and its
/// analytic Jacobian when `jacobian` is requested (`None` where the
/// objective cannot be evaluated, e.g. a ray that misses; a `None` Jacobian
/// where the derivative cannot be formed).
pub(crate) type Objective<'a> = &'a dyn Fn([f64; 2], bool) -> Eval;

/// Outcome of [`newton2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Solve2 {
    pub x: [f64; 2],
    pub residual: f64,
    pub iterations: u32,
    pub converged: bool,
    pub jacobian: Option<Mat2>,
    /// Evaluations of the objective: plain, with a Jacobian requested, and
    /// finite-difference fallbacks (four plain evaluations each) after an
    /// undefined analytic Jacobian.
    pub evaluations: u32,
    pub jacobian_evaluations: u32,
    pub fd_fallbacks: u32,
}

/// Central-difference Jacobian of `f` at `x`, falling back to one-sided
/// differences where a perturbed evaluation fails.
pub(crate) fn fd_jacobian(f: Objective, x: [f64; 2], fx: [f64; 2], h: f64) -> Option<Mat2> {
    let v: [Option<[f64; 2]>; 4] = std::array::from_fn(|q| {
        let mut xq = x;
        xq[q / 2] += if q % 2 == 0 { h } else { -h };
        f(xq, false).map(|(r, _)| r)
    });
    fd_from_stencil(&v, fx, h)
}

/// [`fd_jacobian`] from the four stencil values `f(x ± h e_k)`.
fn fd_from_stencil(v: &[Option<[f64; 2]>], fx: [f64; 2], h: f64) -> Option<Mat2> {
    let mut j = [[0.0; 2]; 2];
    for k in 0..2 {
        let col = match (v[2 * k], v[2 * k + 1]) {
            (Some(a), Some(b)) => [(a[0] - b[0]) / (2.0 * h), (a[1] - b[1]) / (2.0 * h)],
            (Some(a), None) => [(a[0] - fx[0]) / h, (a[1] - fx[1]) / h],
            (None, Some(b)) => [(fx[0] - b[0]) / h, (fx[1] - b[1]) / h],
            (None, None) => return None,
        };
        j[0][k] = col[0];
        j[1][k] = col[1];
    }
    Some(j)
}

/// Broyden's update `J ← J + ((Δr − J·Δx) Δxᵀ) / (Δx·Δx)` (no-op for `Δx = 0`).
#[inline]
fn broyden(j: &mut Mat2, x: [f64; 2], xn: [f64; 2], r: [f64; 2], rnew: [f64; 2]) {
    let s = [xn[0] - x[0], xn[1] - x[1]];
    let ss = s[0] * s[0] + s[1] * s[1];
    if ss > 0.0 {
        let y = [rnew[0] - r[0], rnew[1] - r[1]];
        let js = [
            j[0][0] * s[0] + j[0][1] * s[1],
            j[1][0] * s[0] + j[1][1] * s[1],
        ];
        for row in 0..2 {
            for col in 0..2 {
                j[row][col] += (y[row] - js[row]) * s[col] / ss;
            }
        }
    }
}

/// Solve `f(x) = 0` for `x ∈ ℝ²`: Newton steps with a backtracking line
/// search on `|f|`. The Jacobian starts from `j0` (or a fresh one at `x0`),
/// is updated with Broyden's rank-one formula, and is refreshed whenever it
/// is singular, a step reduces `|f|` by less than half, or the line search
/// fails; `policy` says how fresh Jacobians are formed ([`JacobianPolicy`]).
/// `f` returns `None` where it cannot be evaluated (a ray that misses); the
/// line search treats that as "too far". Converged when `|f| ≤ tol`.
pub(crate) fn newton2(
    f: Objective,
    x0: [f64; 2],
    j0: Option<Mat2>,
    h: f64,
    tol: f64,
    max_iter: u32,
    policy: JacobianPolicy,
) -> Solve2 {
    let mut out = Solve2 {
        x: x0,
        residual: f64::INFINITY,
        iterations: 0,
        converged: false,
        jacobian: j0,
        evaluations: 0,
        jacobian_evaluations: 0,
        fd_fallbacks: 0,
    };
    // A fresh Jacobian at `x` where `f(x) = r` is known (policy above).
    let fresh_jacobian = |x: [f64; 2], r: [f64; 2], out: &mut Solve2| -> Option<Mat2> {
        if policy.analytic() {
            out.jacobian_evaluations += 1;
            if let Some((_, Some(j))) = f(x, true) {
                return Some(j);
            }
            out.fd_fallbacks += 1;
        }
        out.evaluations += 4;
        fd_jacobian(f, x, r, h)
    };
    let want0 = policy.newton() || (policy.analytic() && j0.is_none());
    if want0 {
        out.jacobian_evaluations += 1;
    } else {
        out.evaluations += 1;
    }
    let Some((mut r, ja)) = f(x0, want0) else {
        return out;
    };
    let mut x = x0;
    let mut rn = norm2d(r);
    let (mut j, mut fresh) = match (ja, j0) {
        (Some(j), _) => (j, true),
        (None, Some(j)) => (j, false),
        (None, None) => {
            if want0 {
                out.fd_fallbacks += 1;
            }
            out.evaluations += 4;
            match fd_jacobian(f, x, r, h) {
                Some(j) => (j, true),
                None => {
                    out.residual = rn;
                    return out;
                }
            }
        }
    };
    let mut it = 0;
    while it < max_iter {
        if rn <= tol {
            break;
        }
        it += 1;
        let Some(dx) = solve2(j, [-r[0], -r[1]]) else {
            if fresh {
                break;
            }
            match fresh_jacobian(x, r, &mut out) {
                Some(jn) => {
                    j = jn;
                    fresh = true;
                    continue;
                }
                None => break,
            }
        };
        let mut lam = 1.0;
        let mut accepted = None;
        for _ in 0..16 {
            let xn = [x[0] + lam * dx[0], x[1] + lam * dx[1]];
            if policy.newton() {
                out.jacobian_evaluations += 1;
            } else {
                out.evaluations += 1;
            }
            if let Some((rnew, jn)) = f(xn, policy.newton()) {
                let nn = norm2d(rnew);
                if nn.is_finite() && (nn < rn || nn <= tol) {
                    accepted = Some((xn, rnew, nn, jn));
                    break;
                }
            }
            lam *= 0.5;
        }
        match accepted {
            Some((xn, rnew, nn, jn)) => {
                match jn {
                    Some(jn) => {
                        j = jn;
                        fresh = true;
                    }
                    None => {
                        broyden(&mut j, x, xn, r, rnew);
                        fresh = false;
                    }
                }
                let slow = nn > 0.5 * rn;
                x = xn;
                r = rnew;
                rn = nn;
                if !fresh
                    && slow
                    && rn > tol
                    && let Some(jn) = fresh_jacobian(x, r, &mut out)
                {
                    j = jn;
                    fresh = true;
                }
            }
            None => {
                if fresh {
                    break;
                }
                match fresh_jacobian(x, r, &mut out) {
                    Some(jn) => {
                        j = jn;
                        fresh = true;
                    }
                    None => break,
                }
            }
        }
    }
    out.x = x;
    out.residual = rn;
    out.iterations = it;
    out.converged = rn <= tol;
    out.jacobian = Some(j);
    out
}

/// [`newton2`] as a resumable state machine, so that the function
/// evaluations of many independent solves can be batched (packet traces of
/// many rays' aims). It performs exactly the same arithmetic in the same
/// order as [`newton2`] and returns the same [`Solve2`] bit for bit; the
/// test `newton2_state_machine_matches_newton2` checks this, failure paths
/// and every policy included.
///
/// Use: [`Newton2::pending`] lists the points at which `f` is needed next
/// (1, or 4 for a finite-difference Jacobian; empty once finished) and
/// [`Newton2::wants_jacobian`] whether the analytic Jacobian is requested
/// there; [`Newton2::resume`] takes `f` at exactly those points.
#[derive(Clone, Debug)]
pub(crate) struct Newton2 {
    h: f64,
    tol: f64,
    max_iter: u32,
    policy: JacobianPolicy,
    x0: [f64; 2],
    j0: Option<Mat2>,
    x: [f64; 2],
    r: [f64; 2],
    rn: f64,
    j: Mat2,
    fresh: bool,
    it: u32,
    dx: [f64; 2],
    lam: f64,
    tries: u32,
    phase: Phase,
    pending: [[f64; 2]; 4],
    n_pending: usize,
    want: bool,
    counts: [u32; 3],
    out: Option<Solve2>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Waiting for `f(x0)`.
    First,
    /// Waiting for the initial finite-difference stencil.
    InitialJacobian,
    /// Waiting for `f` at the current line-search point.
    LineSearch,
    /// Waiting for the analytic Jacobian at `x`, requested for the reason.
    Analytic(Refresh),
    /// Waiting for a finite-difference stencil, requested for the reason.
    Stencil(Refresh),
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Refresh {
    /// `J` was singular.
    Singular,
    /// The accepted step reduced `|f|` by less than half.
    Slow,
    /// The line search found no acceptable step.
    Rejected,
}

impl Newton2 {
    pub fn new(
        x0: [f64; 2],
        j0: Option<Mat2>,
        h: f64,
        tol: f64,
        max_iter: u32,
        policy: JacobianPolicy,
    ) -> Self {
        let want = policy.newton() || (policy.analytic() && j0.is_none());
        Newton2 {
            h,
            tol,
            max_iter,
            policy,
            x0,
            j0,
            x: x0,
            r: [0.0; 2],
            rn: f64::INFINITY,
            j: [[0.0; 2]; 2],
            fresh: false,
            it: 0,
            dx: [0.0; 2],
            lam: 1.0,
            tries: 0,
            phase: Phase::First,
            pending: [x0, [0.0; 2], [0.0; 2], [0.0; 2]],
            n_pending: 1,
            want,
            counts: [u32::from(!want), u32::from(want), 0],
            out: None,
        }
    }

    /// Points at which `f` is needed next; empty when finished.
    pub fn pending(&self) -> &[[f64; 2]] {
        &self.pending[..self.n_pending]
    }

    /// Whether the analytic Jacobian is requested at the pending point.
    pub fn wants_jacobian(&self) -> bool {
        self.want
    }

    /// The outcome, once [`Self::pending`] is empty.
    pub fn result(&self) -> Option<Solve2> {
        self.out
    }

    fn request(&mut self, x: [f64; 2], want: bool) {
        self.pending[0] = x;
        self.n_pending = 1;
        self.want = want;
        self.counts[usize::from(want)] += 1;
    }

    /// Request the central-difference stencil of [`fd_jacobian`] at `self.x`.
    fn request_stencil(&mut self, phase: Phase) {
        for k in 0..2 {
            let mut xp = self.x;
            let mut xm = self.x;
            xp[k] += self.h;
            xm[k] -= self.h;
            self.pending[2 * k] = xp;
            self.pending[2 * k + 1] = xm;
        }
        self.n_pending = 4;
        self.want = false;
        self.counts[0] += 4;
        self.phase = phase;
    }

    /// A fresh Jacobian at `self.x` (`fresh_jacobian` of [`newton2`]): the
    /// analytic one first under an analytic policy, else the stencil.
    fn request_fresh(&mut self, why: Refresh) {
        if self.policy.analytic() {
            self.phase = Phase::Analytic(why);
            self.request(self.x, true);
        } else {
            self.request_stencil(Phase::Stencil(why));
        }
    }

    fn finish(&mut self, mut out: Solve2) {
        out.evaluations = self.counts[0];
        out.jacobian_evaluations = self.counts[1];
        out.fd_fallbacks = self.counts[2];
        self.out = Some(out);
        self.n_pending = 0;
        self.phase = Phase::Done;
    }

    /// The normal end of [`newton2`]'s loop.
    fn conclude(&mut self) {
        let out = Solve2 {
            x: self.x,
            residual: self.rn,
            iterations: self.it,
            converged: self.rn <= self.tol,
            jacobian: Some(self.j),
            evaluations: 0,
            jacobian_evaluations: 0,
            fd_fallbacks: 0,
        };
        self.finish(out);
    }

    /// The early exits of [`newton2`] before its loop.
    fn abandon(&mut self, residual: f64) {
        let out = Solve2 {
            x: self.x0,
            residual,
            iterations: 0,
            converged: false,
            jacobian: self.j0,
            evaluations: 0,
            jacobian_evaluations: 0,
            fd_fallbacks: 0,
        };
        self.finish(out);
    }

    /// Top of [`newton2`]'s `while` loop.
    fn iterate(&mut self) {
        if !(self.it < self.max_iter) || self.rn <= self.tol {
            return self.conclude();
        }
        self.it += 1;
        let Some(dx) = solve2(self.j, [-self.r[0], -self.r[1]]) else {
            if self.fresh {
                return self.conclude();
            }
            return self.request_fresh(Refresh::Singular);
        };
        self.dx = dx;
        self.lam = 1.0;
        self.tries = 0;
        self.phase = Phase::LineSearch;
        let want = self.policy.newton();
        self.request(
            [self.x[0] + self.lam * dx[0], self.x[1] + self.lam * dx[1]],
            want,
        );
    }

    /// Continue with `f` evaluated at [`Self::pending`] (same order).
    pub fn resume(&mut self, v: &[Eval]) {
        debug_assert_eq!(v.len(), self.n_pending);
        match self.phase {
            Phase::First => {
                let Some((r, ja)) = v[0] else {
                    return self.abandon(f64::INFINITY);
                };
                self.r = r;
                self.rn = norm2d(r);
                match (ja, self.j0) {
                    (Some(j), _) => {
                        self.j = j;
                        self.fresh = true;
                        self.iterate()
                    }
                    (None, Some(j)) => {
                        self.j = j;
                        self.fresh = false;
                        self.iterate()
                    }
                    (None, None) => {
                        if self.want {
                            self.counts[2] += 1;
                        }
                        self.request_stencil(Phase::InitialJacobian)
                    }
                }
            }
            Phase::InitialJacobian => match self.stencil(v) {
                Some(j) => {
                    self.j = j;
                    self.fresh = true;
                    self.iterate()
                }
                None => self.abandon(self.rn),
            },
            Phase::LineSearch => {
                let xn = self.pending[0];
                if let Some((rnew, jn)) = v[0] {
                    let nn = norm2d(rnew);
                    if nn.is_finite() && (nn < self.rn || nn <= self.tol) {
                        return self.accept(xn, rnew, nn, jn);
                    }
                }
                self.lam *= 0.5;
                self.tries += 1;
                if self.tries < 16 {
                    let (x, dx, lam) = (self.x, self.dx, self.lam);
                    let want = self.policy.newton();
                    return self.request([x[0] + lam * dx[0], x[1] + lam * dx[1]], want);
                }
                if self.fresh {
                    return self.conclude();
                }
                self.request_fresh(Refresh::Rejected)
            }
            Phase::Analytic(why) => match v[0] {
                Some((_, Some(j))) => self.refreshed(j, why),
                _ => {
                    self.counts[2] += 1;
                    self.request_stencil(Phase::Stencil(why))
                }
            },
            Phase::Stencil(why) => match (self.stencil(v), why) {
                (Some(j), _) => self.refreshed(j, why),
                (None, Refresh::Slow) => self.iterate(),
                (None, Refresh::Singular | Refresh::Rejected) => self.conclude(),
            },
            Phase::Done => {}
        }
    }

    /// [`fd_jacobian`] from `f` at the requested stencil.
    fn stencil(&self, v: &[Eval]) -> Option<Mat2> {
        let v: [Option<[f64; 2]>; 4] = std::array::from_fn(|q| v[q].map(|(r, _)| r));
        fd_from_stencil(&v, self.r, self.h)
    }

    /// A fresh Jacobian arrived (`fresh_jacobian` returned `Some`).
    fn refreshed(&mut self, j: Mat2, _why: Refresh) {
        self.j = j;
        self.fresh = true;
        self.iterate()
    }

    /// An accepted line-search step: the analytic Jacobian or Broyden's
    /// update, then a fresh Jacobian if progress was slow.
    fn accept(&mut self, xn: [f64; 2], rnew: [f64; 2], nn: f64, jn: Option<Mat2>) {
        match jn {
            Some(jn) => {
                self.j = jn;
                self.fresh = true;
            }
            None => {
                broyden(&mut self.j, self.x, xn, self.r, rnew);
                self.fresh = false;
            }
        }
        let slow = nn > 0.5 * self.rn;
        self.x = xn;
        self.r = rnew;
        self.rn = nn;
        if !self.fresh && slow && self.rn > self.tol {
            return self.request_fresh(Refresh::Slow);
        }
        self.iterate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: [JacobianPolicy; 3] = [
        JacobianPolicy::FiniteDifference,
        JacobianPolicy::AnalyticRefresh,
        JacobianPolicy::AnalyticNewton,
    ];

    #[test]
    fn two_sum_is_exact() {
        let (s, e) = two_sum(1e16, 1.0);
        assert_eq!(s, 1e16);
        assert_eq!(e, 1.0);
        let (p, e) = two_prod(1.0 + f64::EPSILON, 1.0 + f64::EPSILON);
        assert_eq!(p, 1.0 + 2.0 * f64::EPSILON);
        assert_eq!(e, f64::EPSILON * f64::EPSILON);
    }

    #[test]
    fn compensated_difference_keeps_low_bits() {
        let mut a = Compensated::ZERO;
        let mut b = Compensated::ZERO;
        for _ in 0..1000 {
            a.add_product(1.5168, 0.1);
            b.add_product(1.5168, 0.1);
        }
        b.add(1e-13);
        assert!((b.diff(&a) - 1e-13).abs() < 1e-27);
    }

    #[test]
    fn discriminant_accurate_near_double_root() {
        // (x − 1)(x − (1 + 2⁻²⁶)) has b² − ac = 2⁻⁵⁴ with a=1, b=−(1+2⁻²⁷), c = 1+2⁻²⁶.
        let d = (2.0f64).powi(-26);
        let b = -(1.0 + 0.5 * d);
        let c = 1.0 + d;
        let exact = 0.25 * d * d;
        assert!((discriminant(1.0, b, c) - exact).abs() <= 1e-3 * exact);
    }

    /// `f = (x² + y − 3, x − y² + 3)` with its exact Jacobian.
    fn smooth(x: [f64; 2], jac: bool) -> Eval {
        let j = jac.then(|| [[2.0 * x[0], 1.0], [1.0, -2.0 * x[1]]]);
        Some(([x[0] * x[0] + x[1] - 3.0, x[0] - x[1] * x[1] + 3.0], j))
    }

    #[test]
    fn newton2_solves_nonlinear_system_under_every_policy() {
        let mut iterations = Vec::new();
        for policy in POLICIES {
            let s = newton2(&smooth, [0.5, 0.5], None, 1e-6, 1e-13, 50, policy);
            assert!(s.converged, "{policy:?}: {s:?}");
            assert!((s.x[0] - 1.0).abs() < 1e-12 && (s.x[1] - 2.0).abs() < 1e-12);
            iterations.push(s.iterations);
        }
        // Newton with exact Jacobians converges in fewer iterations than
        // Broyden, and never spends a finite-difference stencil.
        assert!(iterations[2] < iterations[0], "{iterations:?}");
        let s = newton2(
            &smooth,
            [0.5, 0.5],
            None,
            1e-6,
            1e-13,
            50,
            JacobianPolicy::AnalyticNewton,
        );
        assert_eq!((s.evaluations, s.fd_fallbacks), (0, 0));
        // One Jacobian trace per accepted point, plus rejected trials.
        assert!(s.jacobian_evaluations > s.iterations);
    }

    #[test]
    fn analytic_policies_fall_back_to_differences_where_undefined() {
        // The Jacobian is only available in the right half-plane.
        let f = |x: [f64; 2], jac: bool| -> Eval {
            let (r, j) = smooth(x, jac).unwrap();
            Some((r, j.filter(|_| x[0] > 0.9)))
        };
        for policy in [
            JacobianPolicy::AnalyticRefresh,
            JacobianPolicy::AnalyticNewton,
        ] {
            let s = newton2(&f, [0.5, 0.5], None, 1e-6, 1e-13, 50, policy);
            assert!(s.converged, "{policy:?}: {s:?}");
            assert!(
                s.fd_fallbacks >= 1 && s.evaluations >= 4,
                "{policy:?}: {s:?}"
            );
        }
    }

    /// Drive [`Newton2`] with `f` evaluated one point at a time.
    fn drive(
        f: Objective,
        x0: [f64; 2],
        j0: Option<Mat2>,
        h: f64,
        tol: f64,
        max_iter: u32,
        policy: JacobianPolicy,
    ) -> (Solve2, usize) {
        let mut n = Newton2::new(x0, j0, h, tol, max_iter, policy);
        let mut evaluations = 0;
        while !n.pending().is_empty() {
            let want = n.wants_jacobian();
            let v: Vec<Eval> = n.pending().iter().map(|&x| f(x, want)).collect();
            evaluations += v.len();
            n.resume(&v);
        }
        (n.result().unwrap(), evaluations)
    }

    fn bits(s: &Solve2) -> String {
        format!("{s:?}")
    }

    #[test]
    fn newton2_state_machine_matches_newton2() {
        // Smooth, singular, partially undefined and unreachable problems,
        // with and without a starting Jacobian, with analytic Jacobians
        // that are exact, wrong, undefined in places, or absent.
        type F = Box<dyn Fn([f64; 2], bool) -> Eval>;
        let problems: Vec<(F, f64)> = vec![
            (Box::new(smooth), 1e-13),
            // Undefined outside a disk: line searches and one-sided columns.
            (
                Box::new(|x: [f64; 2], jac: bool| {
                    (x[0] * x[0] + x[1] * x[1] < 4.0).then(|| {
                        let j = jac.then(|| [[x[0].cos(), 0.3], [-0.2, 3.0 * x[1] * x[1]]]);
                        (
                            [x[0].sin() + 0.3 * x[1] - 0.5, x[1].powi(3) - 0.2 * x[0]],
                            j,
                        )
                    })
                }),
                1e-12,
            ),
            // No root: |f| ≥ 1 everywhere; the solve stalls and refreshes J.
            (
                Box::new(|x: [f64; 2], jac: bool| {
                    let j = jac.then(|| [[2.0 * x[0], 0.0], [0.0, 2.0 * x[1]]]);
                    Some(([x[0] * x[0] + 1.0, x[1] * x[1] + 1.0], j))
                }),
                1e-12,
            ),
            // Singular Jacobian along x[1] = 0; analytic one undefined there.
            (
                Box::new(|x: [f64; 2], jac: bool| {
                    let j = (jac && x[1] != 0.0).then(|| [[1.0, 0.0], [0.0, 3.0 * x[1] * x[1]]]);
                    Some(([x[0] - 1.0, x[1] * x[1] * x[1]], j))
                }),
                1e-14,
            ),
            // Defined only on a small patch: initial Jacobian may fail.
            (
                Box::new(|x: [f64; 2], _: bool| {
                    (x[0].abs() < 1e-7).then_some(([x[0], x[1] - 1.0], None))
                }),
                1e-12,
            ),
            // A wrong analytic Jacobian: slow steps, refreshes, rejections.
            (
                Box::new(|x: [f64; 2], jac: bool| {
                    let (r, _) = smooth(x, false).unwrap();
                    Some((r, jac.then_some([[0.5, 0.0], [0.0, 0.5]])))
                }),
                1e-13,
            ),
        ];
        let starts = [[0.5, 0.5], [1.9, -0.3], [0.0, 0.0], [-1.2, 1.1], [3.0, 3.0]];
        let jacobians = [
            None,
            Some([[1.0, 0.2], [0.1, 1.5]]),
            Some([[0.0, 0.0], [0.0, 0.0]]),
        ];
        let mut cases = 0;
        for (f, tol) in &problems {
            for &x0 in &starts {
                for &j0 in &jacobians {
                    for max_iter in [0, 3, 40] {
                        for policy in POLICIES {
                            let a = newton2(&**f, x0, j0, 1e-6, *tol, max_iter, policy);
                            let (b, n) = drive(&**f, x0, j0, 1e-6, *tol, max_iter, policy);
                            assert_eq!(
                                bits(&a),
                                bits(&b),
                                "x0 {x0:?} j0 {j0:?} max {max_iter} {policy:?}"
                            );
                            assert_eq!(n as u32, a.evaluations + a.jacobian_evaluations, "{a:?}");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 810);
    }
}
