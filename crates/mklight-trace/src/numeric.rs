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

/// Outcome of [`newton2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Solve2 {
    pub x: [f64; 2],
    pub residual: f64,
    pub iterations: u32,
    pub converged: bool,
    pub jacobian: Option<Mat2>,
}

/// Central-difference Jacobian of `f` at `x`, falling back to one-sided
/// differences where a perturbed evaluation fails.
pub(crate) fn fd_jacobian(
    f: &dyn Fn([f64; 2]) -> Option<[f64; 2]>,
    x: [f64; 2],
    fx: [f64; 2],
    h: f64,
) -> Option<Mat2> {
    let mut j = [[0.0; 2]; 2];
    for k in 0..2 {
        let mut xp = x;
        let mut xm = x;
        xp[k] += h;
        xm[k] -= h;
        let col = match (f(xp), f(xm)) {
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

/// Solve `f(x) = 0` for `x ∈ ℝ²`: Newton steps with a backtracking line
/// search on `|f|`, a Jacobian that starts from `j0` (or central differences
/// with step `h`) and is updated with Broyden's rank-one formula, and a fresh
/// finite-difference Jacobian whenever progress stalls. `f` returns `None`
/// where it cannot be evaluated (a ray that misses); the line search treats
/// that as "too far". Converged when `|f| ≤ tol`.
pub(crate) fn newton2(
    f: &dyn Fn([f64; 2]) -> Option<[f64; 2]>,
    x0: [f64; 2],
    j0: Option<Mat2>,
    h: f64,
    tol: f64,
    max_iter: u32,
) -> Solve2 {
    let mut out = Solve2 {
        x: x0,
        residual: f64::INFINITY,
        iterations: 0,
        converged: false,
        jacobian: j0,
    };
    let Some(mut r) = f(x0) else {
        return out;
    };
    let mut x = x0;
    let mut rn = norm2d(r);
    let (mut j, mut fresh) = match j0 {
        Some(j) => (j, false),
        None => match fd_jacobian(f, x, r, h) {
            Some(j) => (j, true),
            None => {
                out.residual = rn;
                return out;
            }
        },
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
            match fd_jacobian(f, x, r, h) {
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
            if let Some(rnew) = f(xn) {
                let nn = norm2d(rnew);
                if nn.is_finite() && (nn < rn || nn <= tol) {
                    accepted = Some((xn, rnew, nn));
                    break;
                }
            }
            lam *= 0.5;
        }
        match accepted {
            Some((xn, rnew, nn)) => {
                // Broyden: J ← J + ((Δr − J·Δx) Δxᵀ) / (Δx·Δx).
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
                let slow = nn > 0.5 * rn;
                x = xn;
                r = rnew;
                rn = nn;
                fresh = false;
                if slow
                    && rn > tol
                    && let Some(jn) = fd_jacobian(f, x, r, h)
                {
                    j = jn;
                    fresh = true;
                }
            }
            None => {
                if fresh {
                    break;
                }
                match fd_jacobian(f, x, r, h) {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn newton2_solves_nonlinear_system() {
        let f = |x: [f64; 2]| Some([x[0] * x[0] + x[1] - 3.0, x[0] - x[1] * x[1] + 3.0]);
        let s = newton2(&f, [0.5, 0.5], None, 1e-6, 1e-13, 50);
        assert!(s.converged, "{s:?}");
        assert!((s.x[0] - 1.0).abs() < 1e-12 && (s.x[1] - 2.0).abs() < 1e-12);
    }
}
