//! 2-D discrete Fourier transforms on square grids: row-parallel FFTs via
//! `rustfft`, and an exact matrix Fourier transform (MFT) for arbitrary
//! output sampling.
//!
//! Grids are row-major `data[j * m + i]`, `i` along x (columns), `j` along y
//! (rows). Every row transform is independent, so the results do not depend
//! on the thread count.

use mklight_core::{JobContext, MkResult};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Direction of a transform. `Inverse` is the unnormalised
/// `Σ x_j e^{+2πi jk/M}`; `Forward` uses `e^{−2πi jk/M}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dir {
    Forward,
    Inverse,
}

fn plan(m: usize, dir: Dir) -> Arc<dyn Fft<f64>> {
    let mut p = FftPlanner::new();
    match dir {
        Dir::Forward => p.plan_fft_forward(m),
        Dir::Inverse => p.plan_fft_inverse(m),
    }
}

/// `true` if `n` has no prime factor above 7 (fast mixed-radix size).
pub(crate) fn is_fast_len(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for p in [2, 3, 5, 7] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

/// Smallest fast length ≥ `n`.
pub(crate) fn next_fast_len(n: usize) -> usize {
    let mut k = n.max(1);
    while !is_fast_len(k) {
        k += 1;
    }
    k
}

/// The fast length ≥ `min` closest to the real-valued `target`.
pub(crate) fn nearest_fast_len(target: f64, min: usize) -> usize {
    let t = target.max(min as f64);
    let up = next_fast_len(t.ceil() as usize);
    let mut down = t.floor() as usize;
    while down >= min.max(1) && !is_fast_len(down) {
        down -= 1;
    }
    if down >= min.max(1) && (t - down as f64) <= (up as f64 - t) {
        down
    } else {
        up
    }
}

fn rows_fft(
    data: &mut [C64],
    m: usize,
    rows: usize,
    fft: &Arc<dyn Fft<f64>>,
    ctx: &JobContext,
) -> MkResult<()> {
    let scratch_len = fft.get_inplace_scratch_len();
    data[..rows * m].par_chunks_mut(m).try_for_each_init(
        || vec![C64::default(); scratch_len],
        |scratch, row| {
            ctx.check()?;
            fft.process_with_scratch(row, scratch);
            Ok(())
        },
    )
}

/// Out-of-place transpose of an `m × m` grid, tiled for cache locality.
pub(crate) fn transpose(src: &[C64], dst: &mut [C64], m: usize) {
    const B: usize = 32;
    dst.par_chunks_mut(m * B)
        .enumerate()
        .for_each(|(blk, out)| {
            let i0 = blk * B;
            let nb = out.len() / m;
            for j0 in (0..m).step_by(B) {
                for di in 0..nb {
                    let i = i0 + di;
                    for j in j0..(j0 + B).min(m) {
                        out[di * m + j] = src[j * m + i];
                    }
                }
            }
        });
}

/// In-place 2-D DFT of an `m × m` grid. Only the first `nonzero_rows` rows
/// may hold non-zero data (the rest are skipped in the first pass).
pub(crate) fn fft2(
    data: &mut [C64],
    m: usize,
    dir: Dir,
    nonzero_rows: usize,
    ctx: &JobContext,
) -> MkResult<()> {
    debug_assert_eq!(data.len(), m * m);
    let fft = plan(m, dir);
    rows_fft(data, m, nonzero_rows.min(m), &fft, ctx)?;
    ctx.report(0.4, "");
    let mut tmp = vec![C64::default(); m * m];
    transpose(data, &mut tmp, m);
    rows_fft(&mut tmp, m, m, &fft, ctx)?;
    transpose(&tmp, data, m);
    ctx.report(1.0, "");
    Ok(())
}

/// Signed frequency index of DFT bin `k` of length `m`: `0, 1, …, −1`.
#[inline]
pub(crate) fn signed_index(k: usize, m: usize) -> i64 {
    if 2 * k < m {
        k as i64
    } else {
        k as i64 - m as i64
    }
}

/// Exact matrix Fourier transform
/// `U(x_k, y_l) = Σ_{i,j} f[j·n + i] · exp(+2πi (νx_i·x_k + νy_j·y_l))`
/// for arbitrary output coordinates. Cost `O(n²·K + n·K·L)`. Returns an
/// `L × K` row-major grid (`ys.len()` rows of `xs.len()`).
pub(crate) fn mft(
    f: &[C64],
    n: usize,
    nu_x: &[f64],
    nu_y: &[f64],
    xs: &[f64],
    ys: &[f64],
    ctx: &JobContext,
) -> MkResult<Vec<C64>> {
    let (kx, ly) = (xs.len(), ys.len());
    let tau = 2.0 * std::f64::consts::PI;
    // Kernel along x: ex[i * kx + k].
    let ex: Vec<C64> = (0..n * kx)
        .map(|q| {
            let (i, k) = (q / kx, q % kx);
            C64::from_polar(1.0, tau * nu_x[i] * xs[k])
        })
        .collect();
    // Pass 1: t[j][k] = Σ_i f[j][i] ex[i][k], skipping empty rows.
    let mut t = vec![C64::default(); n * kx];
    t.par_chunks_mut(kx)
        .enumerate()
        .try_for_each(|(j, trow)| -> MkResult<()> {
            ctx.check()?;
            let frow = &f[j * n..(j + 1) * n];
            for (i, &fv) in frow.iter().enumerate() {
                if fv.re == 0.0 && fv.im == 0.0 {
                    continue;
                }
                let e = &ex[i * kx..(i + 1) * kx];
                for (o, &ek) in trow.iter_mut().zip(e) {
                    *o += fv * ek;
                }
            }
            Ok(())
        })?;
    let active: Vec<usize> = (0..n)
        .filter(|&j| t[j * kx..(j + 1) * kx].iter().any(|z| z.norm_sqr() > 0.0))
        .collect();
    // Pass 2: u[l][k] = Σ_j ey[l][j] t[j][k].
    let mut u = vec![C64::default(); ly * kx];
    u.par_chunks_mut(kx)
        .enumerate()
        .try_for_each(|(l, urow)| -> MkResult<()> {
            ctx.check()?;
            for &j in &active {
                let e = C64::from_polar(1.0, tau * nu_y[j] * ys[l]);
                let trow = &t[j * kx..(j + 1) * kx];
                for (o, &tv) in urow.iter_mut().zip(trow) {
                    *o += e * tv;
                }
            }
            Ok(())
        })?;
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft2(f: &[C64], m: usize, sign: f64) -> Vec<C64> {
        let tau = 2.0 * std::f64::consts::PI;
        let mut out = vec![C64::default(); m * m];
        for l in 0..m {
            for k in 0..m {
                let mut s = C64::default();
                for j in 0..m {
                    for i in 0..m {
                        let ph = sign * tau * ((i * k + j * l) as f64) / m as f64;
                        s += f[j * m + i] * C64::from_polar(1.0, ph);
                    }
                }
                out[l * m + k] = s;
            }
        }
        out
    }

    #[test]
    fn fft2_matches_naive_dft() {
        let m = 12;
        let f: Vec<C64> = (0..m * m)
            .map(|q| C64::new((q as f64 * 0.37).sin(), (q as f64 * 0.11).cos()))
            .collect();
        let ctx = JobContext::new();
        for (dir, sign) in [(Dir::Forward, -1.0), (Dir::Inverse, 1.0)] {
            let mut g = f.clone();
            fft2(&mut g, m, dir, m, &ctx).unwrap();
            let r = naive_dft2(&f, m, sign);
            for (a, b) in g.iter().zip(&r) {
                assert!((a - b).norm() < 1e-11);
            }
        }
    }

    #[test]
    fn mft_matches_fft_on_dft_grid() {
        let (n, m) = (6, 16);
        let f: Vec<C64> = (0..n * n)
            .map(|q| C64::new((q as f64).sin(), 0.3 * q as f64))
            .collect();
        let mut g = vec![C64::default(); m * m];
        for j in 0..n {
            for i in 0..n {
                g[j * m + i] = f[j * n + i];
            }
        }
        let ctx = JobContext::new();
        fft2(&mut g, m, Dir::Inverse, n, &ctx).unwrap();
        // ν_i = i, x_k = k/m reproduces the DFT kernel.
        let nu: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let xs: Vec<f64> = (0..m).map(|k| k as f64 / m as f64).collect();
        let u = mft(&f, n, &nu, &nu, &xs, &xs, &ctx).unwrap();
        for (a, b) in u.iter().zip(&g) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn fast_lengths() {
        assert!(is_fast_len(2 * 3 * 5 * 7 * 64));
        assert!(!is_fast_len(11));
        assert_eq!(next_fast_len(11), 12);
        assert_eq!(nearest_fast_len(130.2, 64), 128);
        assert_eq!(nearest_fast_len(10.0, 64), 64);
    }
}
