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
use std::sync::{Arc, Mutex, OnceLock};

/// Direction of a transform. `Inverse` is the unnormalised
/// `Σ x_j e^{+2πi jk/M}`; `Forward` uses `e^{−2πi jk/M}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dir {
    Forward,
    Inverse,
}

fn plan(m: usize, dir: Dir) -> Arc<dyn Fft<f64>> {
    static PLANNER: OnceLock<Mutex<FftPlanner<f64>>> = OnceLock::new();
    let mut p = PLANNER
        .get_or_init(|| Mutex::new(FftPlanner::new()))
        .lock()
        .unwrap();
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
    if n >= 32 && kx >= 32 && ly >= 32 {
        return chirp_mft(f, n, nu_x, nu_y, xs, ys, ctx);
    }
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

/// Exact affine image-grid transform. The chief-frame projection B is
/// symmetric, so the 2-D chirp identity turns its mixed xy term into one
/// convolution, without interpolation on an oversized normal-plane FFT.
pub(crate) fn affine_mft(f: &[C64], n: usize, nu: &[f64], xs: &[f64], ys: &[f64], b: [[f64;2];2], ctx: &JobContext) -> MkResult<Vec<C64>> {
    let size = xs.len();
    let m = next_fast_len(n+size-1);
    let tau = 2.0*std::f64::consts::PI;
    let dn = (nu[n-1]-nu[0])/(n-1) as f64;
    let dx = (xs[size-1]-xs[0])/(size-1) as f64;
    let quadratic = |x: f64,y: f64| 0.5*tau*dn*dx*(b[0][0]*x*x+2.0*b[0][1]*x*y+b[1][1]*y*y);
    let mut kernel = vec![C64::default();m*m];
    kernel.par_chunks_mut(m).enumerate().for_each(|(j,row)| {
        let y = if j<size {Some(j as f64)} else if j>m-n {Some(j as f64-m as f64)} else {None};
        if let Some(y)=y {
            for (i,z) in row.iter_mut().enumerate() {
                let x = if i<size {Some(i as f64)} else if i>m-n {Some(i as f64-m as f64)} else {None};
                if let Some(x)=x { *z=C64::from_polar(1.0,-quadratic(x,y)); }
            }
        }
    });
    fft2(&mut kernel,m,Dir::Forward,m,&ctx.subrange(0.0,0.3))?;
    let mut buf = vec![C64::default();m*m];
    buf[..n*m].par_chunks_mut(m).enumerate().for_each(|(j,row)| {
        let y=j as f64;
        for i in 0..n {
            let x=i as f64;
            let phase=tau*dn*(x*(b[0][0]*xs[0]+b[0][1]*ys[0])+y*(b[1][0]*xs[0]+b[1][1]*ys[0]))+quadratic(x,y);
            row[i]=f[j*n+i]*C64::from_polar(1.0,phase);
        }
    });
    fft2(&mut buf,m,Dir::Forward,n,&ctx.subrange(0.3,0.6))?;
    buf.par_iter_mut().zip(&kernel).for_each(|(z,k)| *z*=k);
    drop(kernel);
    fft2(&mut buf,m,Dir::Inverse,m,&ctx.subrange(0.6,0.95))?;
    let mut out=vec![C64::default();size*size];
    out.par_chunks_mut(size).enumerate().for_each(|(j,row)| {
        for (i,z) in row.iter_mut().enumerate() {
            let phase=tau*nu[0]*((b[0][0]+b[1][0])*xs[i]+(b[0][1]+b[1][1])*ys[j])+quadratic(i as f64,j as f64);
            *z=buf[j*m+i]*C64::from_polar(1.0/(m*m) as f64,phase);
        }
    });
    Ok(out)
}

/// Uniform-frequency, uniform-image transform via Bluestein convolution.
/// exp(i θ i k) = exp(i θ i²/2) exp(i θ k²/2) exp(−i θ (k−i)²/2).
/// This evaluates the same finite sum as the matrix transform, without
/// interpolation or a padded image-plane approximation.
struct ChirpTransform {
    len: usize,
    input: Vec<C64>,
    output: Vec<C64>,
    kernel: Vec<C64>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl ChirpTransform {
    fn new(nu: &[f64], x: &[f64]) -> Self {
        let n = nu.len();
        let len = next_fast_len(n + x.len() - 1);
        let tau = 2.0 * std::f64::consts::PI;
        let dn = (nu[n - 1] - nu[0]) / (n - 1) as f64;
        let theta = tau * dn * (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64;
        let input = (0..n)
            .map(|i| {
                let i = i as f64;
                C64::from_polar(1.0, tau * dn * x[0] * i + 0.5 * theta * i * i)
            })
            .collect();
        let output = x
            .iter()
            .enumerate()
            .map(|(k, &x)| {
                let k = k as f64;
                C64::from_polar(1.0 / len as f64, tau * nu[0] * x + 0.5 * theta * k * k)
            })
            .collect();
        let mut kernel = vec![C64::default(); len];
        for k in 0..x.len() {
            kernel[k] = C64::from_polar(1.0, -0.5 * theta * (k as f64).powi(2));
        }
        for k in 1..n {
            kernel[len - k] = C64::from_polar(1.0, -0.5 * theta * (k as f64).powi(2));
        }
        let forward = plan(len, Dir::Forward);
        let inverse = plan(len, Dir::Inverse);
        forward.process(&mut kernel);
        Self {
            len,
            input,
            output,
            kernel,
            forward,
            inverse,
        }
    }

    fn rows(&self, src: &[C64], dst: &mut [C64], ctx: &JobContext) -> MkResult<()> {
        let scratch = self
            .forward
            .get_inplace_scratch_len()
            .max(self.inverse.get_inplace_scratch_len());
        src.par_chunks(self.input.len())
            .zip(dst.par_chunks_mut(self.output.len()))
            .try_for_each_init(
                || {
                    (
                        vec![C64::default(); self.len],
                        vec![C64::default(); scratch],
                    )
                },
                |(buf, scratch), (row, out)| {
                    ctx.check()?;
                    buf.fill(C64::default());
                    for ((b, v), ph) in buf.iter_mut().zip(row).zip(&self.input) {
                        *b = v * ph;
                    }
                    self.forward.process_with_scratch(buf, scratch);
                    for (b, k) in buf.iter_mut().zip(&self.kernel) {
                        *b *= k;
                    }
                    self.inverse.process_with_scratch(buf, scratch);
                    for ((o, b), ph) in out.iter_mut().zip(buf.iter()).zip(&self.output) {
                        *o = b * ph;
                    }
                    Ok(())
                },
            )
    }
}

fn chirp_mft(
    f: &[C64],
    n: usize,
    nu_x: &[f64],
    nu_y: &[f64],
    xs: &[f64],
    ys: &[f64],
    ctx: &JobContext,
) -> MkResult<Vec<C64>> {
    let (kx, ly) = (xs.len(), ys.len());
    let mut t = vec![C64::default(); n * kx];
    ChirpTransform::new(nu_x, xs).rows(f, &mut t, ctx)?;
    let mut columns = vec![C64::default(); n * kx];
    columns.par_chunks_mut(n).enumerate().for_each(|(k, col)| {
        for (j, z) in col.iter_mut().enumerate() {
            *z = t[j * kx + k];
        }
    });
    let mut u = vec![C64::default(); ly * kx];
    ChirpTransform::new(nu_y, ys).rows(&columns, &mut u, ctx)?;
    let mut out = vec![C64::default(); ly * kx];
    out.par_chunks_mut(kx).enumerate().for_each(|(j, row)| {
        for (k, z) in row.iter_mut().enumerate() {
            *z = u[k * ly + j];
        }
    });
    Ok(out)
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
    fn chirp_transform_matches_finite_sum_on_arbitrary_uniform_grids() {
        for (n, m, dn, dx) in [(37, 53, 0.013, 0.21), (64, 47, -0.04, 0.37), (513, 769, 0.017, 0.09)] {
            let nu: Vec<_> = (0..n).map(|i| (i as f64 - 0.5*n as f64)*dn).collect();
            let xs: Vec<_> = (0..m).map(|i| -0.37 + i as f64*dx).collect();
            let f: Vec<_> = (0..n).map(|i| C64::new((i as f64*0.3).sin(), (i as f64*0.7).cos())).collect();
            let c = ChirpTransform::new(&nu, &xs);
            let mut got = vec![C64::default(); m];
            c.rows(&f, &mut got, &JobContext::new()).unwrap();
            for (k, &x) in xs.iter().enumerate() {
                let want: C64 = f.iter().zip(&nu).map(|(z,v)| z*C64::from_polar(1.0, 2.0*std::f64::consts::PI*v*x)).sum();
                assert!((got[k]-want).norm() < 2e-9, "n={n} k={k}: {}", (got[k]-want).norm());
            }
        }
    }

    #[test]
    fn affine_transform_matches_direct_skew_sum() {
        let (n,m)=(13,19);
        let nu: Vec<_>=(0..n).map(|i| (i as f64-6.0)*0.31).collect();
        let xs: Vec<_>=(0..m).map(|i| (i as f64-9.0)*0.037-0.013).collect();
        let ys: Vec<_>=xs.iter().map(|x| x+0.021).collect();
        let f: Vec<_>=(0..n*n).map(|i| C64::new((i as f64*0.3).sin(),(i as f64*0.7).cos())).collect();
        let b=[[0.83,-0.17],[-0.17,0.76]];
        let got=affine_mft(&f,n,&nu,&xs,&ys,b,&JobContext::new()).unwrap();
        for j in 0..m { for i in 0..m {
            let mut want=C64::default();
            for y in 0..n { for x in 0..n {
                let phase=2.0*std::f64::consts::PI*(nu[x]*(b[0][0]*xs[i]+b[0][1]*ys[j])+nu[y]*(b[1][0]*xs[i]+b[1][1]*ys[j]));
                want+=f[y*n+x]*C64::from_polar(1.0,phase);
            }}
            assert!((got[j*m+i]-want).norm()<1e-11);
        }}
    }

    #[test]
    fn fast_lengths() {
        assert!(is_fast_len(2 * 3 * 5 * 7 * 64));
        assert!(!is_fast_len(11));
        assert_eq!(next_fast_len(11), 12);
        assert_eq!(next_fast_len(97), 98);
        assert_eq!(next_fast_len(128), 128);
    }
}
