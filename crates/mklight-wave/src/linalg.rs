//! Small dense least-squares solver: column-equilibrated Householder QR with
//! a singular-value condition estimate (one-sided Jacobi on R).

use mklight_core::{JobContext, MkError, MkResult};

/// Solution of `min ‖A·x − b‖₂`.
#[derive(Clone, Debug)]
pub(crate) struct Lstsq {
    pub x: Vec<f64>,
    /// 2-norm condition number of the column-equilibrated `A`.
    pub condition: f64,
    /// `‖A·x − b‖₂`.
    pub residual_norm: f64,
}

/// Least squares with `A` given column-major (`cols[j][i] = A[i][j]`).
/// Columns are scaled to unit norm first, so the reported condition number
/// does not depend on the basis normalisation. Fails if `A` is rank
/// deficient (a zero column, fewer rows than columns, or `σ_min/σ_max` below
/// `1e-13`).
pub(crate) fn lstsq(mut cols: Vec<Vec<f64>>, mut b: Vec<f64>, ctx: &JobContext) -> MkResult<Lstsq> {
    let k = cols.len();
    let n = b.len();
    if k == 0 {
        return Ok(Lstsq {
            x: vec![],
            condition: 1.0,
            residual_norm: norm(&b),
        });
    }
    if n < k {
        return Err(MkError::Numerical(format!(
            "least squares needs at least {k} samples, got {n}"
        )));
    }
    let mut scale = vec![0.0; k];
    for (j, c) in cols.iter_mut().enumerate() {
        let s = norm(c);
        if s == 0.0 || !s.is_finite() {
            return Err(MkError::Numerical(format!(
                "least squares: basis function {j} vanishes on all samples"
            )));
        }
        c.iter_mut().for_each(|v| *v /= s);
        scale[j] = s;
    }
    // Householder QR, applying each reflector to the trailing columns and b.
    let mut rdiag = vec![0.0; k];
    for j in 0..k {
        ctx.check()?;
        let (head, tail) = cols.split_at_mut(j + 1);
        let cj = &mut head[j];
        let alpha_norm = norm(&cj[j..]);
        let alpha = if cj[j] > 0.0 { -alpha_norm } else { alpha_norm };
        // v = x − α e₁, stored in cj[j..].
        cj[j] -= alpha;
        let vnorm2: f64 = cj[j..].iter().map(|v| v * v).sum();
        rdiag[j] = alpha;
        if vnorm2 > 0.0 {
            for c in tail.iter_mut() {
                let d: f64 = cj[j..].iter().zip(&c[j..]).map(|(v, x)| v * x).sum();
                let f = 2.0 * d / vnorm2;
                c[j..]
                    .iter_mut()
                    .zip(&cj[j..])
                    .for_each(|(x, v)| *x -= f * v);
            }
            let d: f64 = cj[j..].iter().zip(&b[j..]).map(|(v, x)| v * x).sum();
            let f = 2.0 * d / vnorm2;
            b[j..]
                .iter_mut()
                .zip(&cj[j..])
                .for_each(|(x, v)| *x -= f * v);
        }
    }
    // R (upper triangular): R[i][j] = cols[j][i] for i < j, rdiag on the diagonal.
    let r = |i: usize, j: usize| -> f64 {
        if i == j {
            rdiag[j]
        } else if i < j {
            cols[j][i]
        } else {
            0.0
        }
    };
    let mut rm: Vec<Vec<f64>> = (0..k).map(|j| (0..k).map(|i| r(i, j)).collect()).collect();
    let sv = singular_values(&mut rm);
    let smax = sv.iter().cloned().fold(0.0, f64::max);
    let smin = sv.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(smin > 1e-13 * smax) {
        return Err(MkError::Numerical(format!(
            "least squares: rank-deficient system (σmin/σmax = {:.2e})",
            smin / smax
        )));
    }
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = b[i];
        for (jj, xj) in x.iter().enumerate().skip(i + 1) {
            s -= r(i, jj) * xj;
        }
        x[i] = s / rdiag[i];
    }
    for (xi, s) in x.iter_mut().zip(&scale) {
        *xi /= s;
    }
    Ok(Lstsq {
        x,
        condition: smax / smin,
        residual_norm: norm(&b[k..]),
    })
}

fn norm(v: &[f64]) -> f64 {
    let m = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * v.iter().map(|x| (x / m) * (x / m)).sum::<f64>().sqrt()
}

/// Singular values of a square matrix given as columns (one-sided Jacobi,
/// Hestenes). The columns are overwritten.
pub(crate) fn singular_values(cols: &mut [Vec<f64>]) -> Vec<f64> {
    let k = cols.len();
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let (a, b, g) = {
                    let (cp, cq) = (&cols[p], &cols[q]);
                    let a: f64 = cp.iter().map(|x| x * x).sum();
                    let b: f64 = cq.iter().map(|x| x * x).sum();
                    let g: f64 = cp.iter().zip(cq).map(|(x, y)| x * y).sum();
                    (a, b, g)
                };
                if g.abs() <= 1e-15 * (a * b).sqrt() || g == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (b - a) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                let (lo, hi) = cols.split_at_mut(q);
                for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                    let (xp, yq) = (*x, *y);
                    *x = c * xp - s * yq;
                    *y = s * xp + c * yq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    cols.iter().map(|c| norm(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_overdetermined_exactly() {
        // y = 1 + 2t − 3t² sampled exactly.
        let ts: Vec<f64> = (0..20).map(|i| i as f64 / 7.0 - 1.0).collect();
        let cols = vec![
            ts.iter().map(|_| 1.0).collect(),
            ts.clone(),
            ts.iter().map(|t| t * t).collect(),
        ];
        let b: Vec<f64> = ts.iter().map(|t| 1.0 + 2.0 * t - 3.0 * t * t).collect();
        let s = lstsq(cols, b, &JobContext::new()).unwrap();
        assert!((s.x[0] - 1.0).abs() < 1e-13);
        assert!((s.x[1] - 2.0).abs() < 1e-13);
        assert!((s.x[2] + 3.0).abs() < 1e-13);
        assert!(s.residual_norm < 1e-12);
    }

    #[test]
    fn singular_values_of_diagonal_and_rank_deficiency() {
        let mut c = vec![vec![3.0, 0.0], vec![0.0, 0.5]];
        let mut sv = singular_values(&mut c);
        sv.sort_by(f64::total_cmp);
        assert!((sv[0] - 0.5).abs() < 1e-15 && (sv[1] - 3.0).abs() < 1e-15);
        // Two identical columns: rank deficient.
        let col: Vec<f64> = (0..5).map(|i| i as f64).collect();
        assert!(lstsq(vec![col.clone(), col], vec![1.0; 5], &JobContext::new()).is_err());
    }
}
