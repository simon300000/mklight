//! Small dense linear algebra for the optimisers.
//!
//! Problems here are "tall and thin" (many residuals, at most a few hundred
//! variables), so everything is plain row-major `f64` with Householder QR as
//! the workhorse. Normal equations `JᵀJ` are never formed for solving: they
//! square the condition number of `J` (lens Jacobians are notoriously badly
//! conditioned). The symmetric eigen solver (cyclic Jacobi) is used by CMA-ES.

/// Dense row-major matrix.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }
    /// Build from row-major data. Panics if the length does not match.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data length mismatch");
        Matrix { rows, cols, data }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t
    }
    /// `A·v`.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.cols);
        (0..self.rows).map(|i| dot(self.row(i), v)).collect()
    }
    /// `Aᵀ·v`.
    pub fn tr_mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.rows);
        let mut out = vec![0.0; self.cols];
        for (i, &vi) in v.iter().enumerate() {
            if vi != 0.0 {
                for (o, a) in out.iter_mut().zip(self.row(i)) {
                    *o = a.mul_add(vi, *o);
                }
            }
        }
        out
    }
    pub fn mul_mat(&self, o: &Matrix) -> Matrix {
        assert_eq!(self.cols, o.rows);
        let mut r = Matrix::zeros(self.rows, o.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a == 0.0 {
                    continue;
                }
                let orow = o.row(k);
                for (x, b) in r.row_mut(i).iter_mut().zip(orow) {
                    *x = a.mul_add(*b, *x);
                }
            }
        }
        r
    }
    /// Euclidean norm of each column (scale-safe).
    pub fn column_norms(&self) -> Vec<f64> {
        (0..self.cols).map(|j| norm2(&self.col(j))).collect()
    }
    /// Copy of the listed columns, in order.
    pub fn select_columns(&self, cols: &[usize]) -> Matrix {
        let mut m = Matrix::zeros(self.rows, cols.len());
        for i in 0..self.rows {
            for (k, &j) in cols.iter().enumerate() {
                m[(i, k)] = self[(i, j)];
            }
        }
        m
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |s, (x, y)| x.mul_add(*y, s))
}

/// Overflow/underflow-safe Euclidean norm (scaled, as in LAPACK `dnrm2`).
pub fn norm2(v: &[f64]) -> f64 {
    let m = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let s: f64 = v.iter().map(|x| (x / m) * (x / m)).sum();
    m * s.sqrt()
}

/// Householder QR factorisation `A = Q·R` of an `m × n` matrix with `m ≥ n`.
/// `R` is stored in the upper triangle, the Householder vectors below it.
#[derive(Clone, Debug)]
pub struct Qr {
    a: Matrix,
    beta: Vec<f64>,
}

impl Qr {
    pub fn new(mut a: Matrix) -> Qr {
        let (m, n) = (a.rows, a.cols);
        assert!(m >= n, "QR needs rows ≥ cols");
        let mut beta = vec![0.0; n];
        for k in 0..n {
            // Householder vector v = x + sign(x0)‖x‖e0 (no cancellation).
            let col: Vec<f64> = (k..m).map(|i| a[(i, k)]).collect();
            let nrm = norm2(&col);
            if nrm == 0.0 {
                beta[k] = 0.0;
                continue;
            }
            let alpha = if col[0] >= 0.0 { -nrm } else { nrm };
            let v0 = col[0] - alpha;
            // Normalise v so that v[0] = 1; H = I − β v vᵀ, β = −v0/alpha.
            for i in k + 1..m {
                a[(i, k)] /= v0;
            }
            beta[k] = -v0 / alpha;
            a[(k, k)] = alpha;
            for j in k + 1..n {
                let mut s = a[(k, j)];
                for i in k + 1..m {
                    s = a[(i, k)].mul_add(a[(i, j)], s);
                }
                s *= beta[k];
                a[(k, j)] -= s;
                for i in k + 1..m {
                    let vik = a[(i, k)];
                    a[(i, j)] = (-s).mul_add(vik, a[(i, j)]);
                }
            }
        }
        Qr { a, beta }
    }

    /// `b ← Qᵀ b`.
    pub fn apply_qt(&self, b: &mut [f64]) {
        let (m, n) = (self.a.rows, self.a.cols);
        for k in 0..n {
            if self.beta[k] == 0.0 {
                continue;
            }
            let mut s = b[k];
            for i in k + 1..m {
                s = self.a[(i, k)].mul_add(b[i], s);
            }
            s *= self.beta[k];
            b[k] -= s;
            for i in k + 1..m {
                b[i] = (-s).mul_add(self.a[(i, k)], b[i]);
            }
        }
    }

    /// The `n × n` upper-triangular factor.
    pub fn r(&self) -> Matrix {
        let n = self.a.cols;
        let mut r = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                r[(i, j)] = self.a[(i, j)];
            }
        }
        r
    }

    /// Largest and smallest `|R_kk|`, a cheap rank / conditioning indicator.
    pub fn diag_range(&self) -> (f64, f64) {
        let n = self.a.cols;
        let mut hi = 0.0_f64;
        let mut lo = f64::INFINITY;
        for k in 0..n {
            let d = self.a[(k, k)].abs();
            hi = hi.max(d);
            lo = lo.min(d);
        }
        (hi, if n == 0 { 0.0 } else { lo })
    }

    /// Is `R` numerically nonsingular (relative tolerance on its diagonal)?
    pub fn is_full_rank(&self) -> bool {
        let (hi, lo) = self.diag_range();
        let n = self.a.cols.max(1) as f64;
        hi > 0.0 && lo > hi * n * 1e3 * f64::EPSILON
    }

    /// Least-squares solution of `min ‖A x − b‖`. `None` when `R` is singular.
    pub fn solve_ls(&self, b: &[f64]) -> Option<Vec<f64>> {
        if !self.is_full_rank() {
            return None;
        }
        let mut qb = b.to_vec();
        self.apply_qt(&mut qb);
        Some(solve_upper(&self.a, &qb[..self.a.cols]))
    }
}

/// Back substitution with the upper triangle of `r` (only the leading
/// `n × n` block, `n = rhs.len()`, is read).
pub fn solve_upper(r: &Matrix, rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut x = rhs.to_vec();
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s = (-r[(i, j)]).mul_add(x[j], s);
        }
        x[i] = s / r[(i, i)];
    }
    x
}

/// Solve `Rᵀ x = rhs` (forward substitution) with the upper triangle of `r`.
pub fn solve_upper_transpose(r: &Matrix, rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut x = rhs.to_vec();
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s = (-r[(j, i)]).mul_add(x[j], s);
        }
        x[i] = s / r[(i, i)];
    }
    x
}

/// Eigen-decomposition of a symmetric matrix by the cyclic Jacobi method.
/// Returns `(eigenvalues, eigenvectors as columns)`. Jacobi is slower than
/// tridiagonal QR but unconditionally stable and accurate to high relative
/// precision, which matters for the badly scaled covariances CMA-ES builds.
pub fn sym_eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    let n = a.rows;
    assert_eq!(n, a.cols);
    let mut m = a.clone();
    // Symmetrise defensively.
    for i in 0..n {
        for j in i + 1..n {
            let s = 0.5 * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = s;
            m[(j, i)] = s;
        }
    }
    let mut v = Matrix::identity(n);
    for _sweep in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..n {
            diag += m[(i, i)] * m[(i, i)];
            for j in i + 1..n {
                off += m[(i, j)] * m[(i, j)];
            }
        }
        if off <= f64::EPSILON * f64::EPSILON * diag || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let app = m[(p, p)];
                let aqq = m[(q, q)];
                // Stable rotation (Golub & Van Loan Alg. 8.4.1).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[(k, p)];
                    let mkq = m[(k, q)];
                    m[(k, p)] = c * mkp - s * mkq;
                    m[(k, q)] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[(p, k)];
                    let mqk = m[(q, k)];
                    m[(p, k)] = c * mpk - s * mqk;
                    m[(q, k)] = s * mpk + c * mqk;
                }
                m[(p, q)] = 0.0;
                m[(q, p)] = 0.0;
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[(i, i)]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_least_squares_matches_exact_fit() {
        // Fit y = 1 + 2t − 0.5t² exactly through 6 points.
        let ts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let mut a = Matrix::zeros(6, 3);
        let mut b = vec![0.0; 6];
        for (i, &t) in ts.iter().enumerate() {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = t;
            a[(i, 2)] = t * t;
            b[i] = 1.0 + 2.0 * t - 0.5 * t * t;
        }
        let qr = Qr::new(a.clone());
        let x = qr.solve_ls(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-13);
        assert!((x[1] - 2.0).abs() < 1e-13);
        assert!((x[2] + 0.5).abs() < 1e-13);
        // RᵀR = AᵀA.
        let r = qr.r();
        let rtr = r.transpose().mul_mat(&r);
        let ata = a.transpose().mul_mat(&a);
        for (u, v) in rtr.data.iter().zip(&ata.data) {
            assert!((u - v).abs() < 1e-12 * (1.0 + v.abs()));
        }
    }

    #[test]
    fn qr_detects_rank_deficiency() {
        let a = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        assert!(Qr::new(a).solve_ls(&[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn triangular_solves() {
        let r = Matrix::from_row_major(3, 3, vec![2.0, 1.0, -1.0, 0.0, 3.0, 0.5, 0.0, 0.0, 4.0]);
        let x = [1.0, -2.0, 0.25];
        let b = r.mul_vec(&x);
        let y = solve_upper(&r, &b);
        let bt = r.transpose().mul_vec(&x);
        let z = solve_upper_transpose(&r, &bt);
        for k in 0..3 {
            assert!((y[k] - x[k]).abs() < 1e-14);
            assert!((z[k] - x[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn jacobi_eigen_reconstructs() {
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, -0.2, 0.5, -0.2, 1.0]);
        let (w, v) = sym_eigen(&a);
        // A v_k = w_k v_k and V orthonormal.
        for k in 0..3 {
            let vk = v.col(k);
            let av = a.mul_vec(&vk);
            for i in 0..3 {
                assert!((av[i] - w[k] * vk[i]).abs() < 1e-13);
            }
        }
        let vtv = v.transpose().mul_mat(&v);
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((vtv[(i, j)] - e).abs() < 1e-14);
            }
        }
        // Trace is preserved.
        assert!((w.iter().sum::<f64>() - 8.0).abs() < 1e-13);
    }

    #[test]
    fn norm_is_overflow_safe() {
        assert!((norm2(&[3e200, 4e200]) / 5e200 - 1.0).abs() < 1e-15);
        assert!((norm2(&[3e-200, 4e-200]) - 5e-200).abs() < 1e-213);
    }
}
