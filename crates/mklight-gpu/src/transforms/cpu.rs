//! CPU-f32 specification of the recipe steps, following the Metal kernels.
use super::{recipe::*, *};
use crate::consts::*;
use crate::resident::immutable::CacheCategory::TransformPlans as TRANSFORM_PLANS;
use std::sync::Arc;
type C = [f32; 2];
pub(super) fn cis(mut a: Df32) -> C {
    a = a.add_f32(-a.hi.round());
    a = a.add_f32(-a.hi.round());
    let t = (a.hi + a.lo) * std::f32::consts::TAU;
    let (s, c) = t.sin_cos();
    [c, s]
}
fn mul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn add(a: C, b: C) -> C {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: C, b: C) -> C {
    [a[0] - b[0], a[1] - b[1]]
}
/// v·(s·i)
fn rot(v: C, s: f32) -> C {
    [-s * v[1], s * v[0]]
}
fn dft4(u: &mut [C], i: [usize; 4], s: f32) {
    let (a0, a1) = (add(u[i[0]], u[i[2]]), sub(u[i[0]], u[i[2]]));
    let (a2, a3) = (add(u[i[1]], u[i[3]]), rot(sub(u[i[1]], u[i[3]]), s));
    (u[i[0]], u[i[2]], u[i[1]], u[i[3]]) = (add(a0, a2), sub(a0, a2), add(a1, a3), sub(a1, a3));
}
/// Unnormalised DFT of R values, exp(s·2πi·m·t/R).
fn dft(u: &mut [C], s: f32) {
    match u.len() {
        2 => (u[0], u[1]) = (add(u[0], u[1]), sub(u[0], u[1])),
        4 => dft4(u, [0, 1, 2, 3], s),
        8 => {
            let c = std::f32::consts::FRAC_1_SQRT_2;
            dft4(u, [0, 2, 4, 6], s);
            dft4(u, [1, 3, 5, 7], s);
            let o = [
                u[1],
                [c * (u[3][0] - s * u[3][1]), c * (u[3][1] + s * u[3][0])],
                rot(u[5], s),
                [c * (-u[7][0] - s * u[7][1]), c * (s * u[7][0] - u[7][1])],
            ];
            let e = [u[0], u[2], u[4], u[6]];
            for m in 0..4 {
                u[m] = add(e[m], o[m]);
                u[m + 4] = sub(e[m], o[m]);
            }
        }
        r => {
            let v = u.to_vec();
            for (m, out) in u.iter_mut().enumerate() {
                let mut acc = v[0];
                for (t, x) in v.iter().enumerate().skip(1) {
                    let (sn, cs) =
                        (std::f64::consts::TAU * ((m * t) % r) as f64 / r as f64).sin_cos();
                    acc = add(acc, mul(*x, [cs as f32, s * sn as f32]));
                }
                *out = acc;
            }
        }
    }
}
/// In-place Stockham FFT of one row with the shared f64-derived twiddles.
fn fft_row(row: &mut [C], p: &Params, tw: &[C], sign: f32) {
    let n = row.len();
    let mut out = vec![[0.; 2]; n];
    let mut span = 1;
    for stage in 0..p.stages as usize {
        let r = p.radix(stage);
        let m = n / r;
        for b in 0..m {
            let k = b % span;
            let mut v: Vec<C> = (0..r).map(|t| row[b + t * m]).collect();
            for t in 1..r {
                let c = tw[span - 1 + k * (r - 1) + t - 1];
                v[t] = mul(v[t], [c[0], -sign * c[1]]);
            }
            dft(&mut v, sign);
            for (t, v) in v.into_iter().enumerate() {
                out[(b - k) * r + k + t * span] = v;
            }
        }
        row.copy_from_slice(&out);
        span *= r;
    }
}
fn quadratic(a: &[Df32; 6], x: f32, y: f32) -> Df32 {
    (a[0].mul_f32(x).mul_f32(x) + a[1].mul_f32(2.0 * x).mul_f32(y) + a[4].mul_f32(y).mul_f32(y))
        .mul_f32(0.5)
}
fn pre(a: &[Df32; 6], x: f32, y: f32) -> Df32 {
    quadratic(a, x, y) + a[2].mul_f32(x) + a[5].mul_f32(y)
}
fn eval(p: &Params, a: &[C], b: &[C], ctl: &JobControl) -> MkResult<Vec<C>> {
    let affine = p.affine.map(|v| Df32::new(v[0], v[1]));
    let (w, h) = (p.width as usize, p.height as usize);
    let (nx, ny, ox, oy) = (p.nx as usize, p.ny as usize, p.ox as usize, p.oy as usize);
    let (sign, scale) = (p.sign(), p.scale());
    if matches!(p.mode, MKL_TRANSFORM_FFT_ROWS | MKL_TRANSFORM_CHIRP_ROWS) {
        let tw = twiddles(w);
        let chirp = (p.mode == MKL_TRANSFORM_CHIRP_ROWS).then(|| chirp_factors(p));
        let mut out = vec![];
        for y in 0..h {
            ctl.check()?;
            let Some(c) = &chirp else {
                let mut row = a[y * w..(y + 1) * w].to_vec();
                fft_row(&mut row, p, &tw, sign);
                out.extend(row.into_iter().map(|v| v.map(|v| v * scale)));
                continue;
            };
            let mut row: Vec<C> = (0..w)
                .map(|x| {
                    if x < nx {
                        mul(a[y * nx + x], c[x])
                    } else {
                        [0.; 2]
                    }
                })
                .collect();
            fft_row(&mut row, p, &tw, -1.);
            row.iter_mut().zip(b).for_each(|(v, k)| *v = mul(*v, *k));
            fft_row(&mut row, p, &tw, 1.);
            out.extend((0..ox).map(|x| mul(row[x], c[nx + x]).map(|v| v * scale)));
        }
        return Ok(out);
    }
    let mut out = vec![[0.; 2]; p.count as usize];
    for (id, z) in out.iter_mut().enumerate() {
        if id % 4096 == 0 {
            ctl.check()?;
        }
        *z = match p.mode {
            MKL_TRANSFORM_TRANSPOSE => a[(id % h) * w + id / h].map(|v| v * scale),
            MKL_TRANSFORM_KERNEL => {
                let lag = |v: usize, pad: usize, len: usize, o: usize| {
                    if v < o {
                        Some(v as f32)
                    } else if v > pad - len {
                        Some(v as f32 - pad as f32)
                    } else {
                        None
                    }
                };
                match (lag(id % w, w, nx, ox), lag(id / w, h, ny, oy)) {
                    (Some(x), Some(y)) => cis(quadratic(&affine, x, y).mul_f32(-sign)),
                    _ => [0.; 2],
                }
            }
            MKL_TRANSFORM_PRE => {
                let (x, y) = (id % w, id / w);
                if x < nx && y < ny {
                    mul(
                        a[y * nx + x],
                        cis(pre(&affine, x as f32, y as f32).mul_f32(sign)),
                    )
                } else {
                    [0.; 2]
                }
            }
            MKL_TRANSFORM_POST => {
                let (x, y) = (id % ox, id / ox);
                mul(
                    a[y * w + x],
                    cis(quadratic(&affine, x as f32, y as f32).mul_f32(sign)),
                )
                .map(|v| v * scale)
            }
            MKL_TRANSFORM_MULTIPLY => {
                let mut v = b[if p.flags & BROADCAST != 0 { id % w } else { id }];
                if p.flags & CONJUGATE != 0 {
                    v[1] = -v[1];
                }
                mul(a[id], v).map(|v| v * scale)
            }
            MKL_TRANSFORM_PHASE => {
                let (x, y) = ((id % w) as f32, (id / w) as f32);
                let c = affine;
                let ph = c[0].mul_f32(x).mul_f32(x)
                    + c[1].mul_f32(x).mul_f32(y)
                    + c[2].mul_f32(y).mul_f32(y)
                    + c[3].mul_f32(x)
                    + c[4].mul_f32(y)
                    + c[5];
                mul(a[id], cis(ph))
            }
            _ => unreachable!(),
        };
    }
    Ok(out)
}
pub(super) fn execute(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    if matches!(
        r,
        Request::Correlation(_)
            | Request::OpticalCorrelation(_)
            | Request::PrepareOpticalStencil(_)
    ) {
        return super::correlation::cpu(r, x, ctl);
    }
    let a = accesses(r);
    let recipe = request(r)?;
    let mut slots: Vec<Option<Arc<Vec<C>>>> = vec![None; recipe.steps.len() + 2];
    for (i, v) in a.reads.iter().enumerate() {
        let data: &[C] = bytemuck::cast_slice(x.words(v.buffer));
        slots[i] = Some(Arc::new(data[v.offset..v.offset + v.desc.len()?].to_vec()));
    }
    let keys = recipe.cache_keys();
    let mut last_use = vec![usize::MAX; slots.len()];
    for (i, s) in recipe.steps.iter().enumerate() {
        dependencies(s).into_iter().for_each(|d| last_use[d] = i);
    }
    for (i, s) in recipe.steps.iter().enumerate() {
        ctl.check()?;
        // Chirp spectra depend on their step parameters only (`Recipe::
        // cache_keys`), so they are content-addressed as on Metal.
        let key = x.content_key(TRANSFORM_PLANS, keys[i].clone());
        let cached = if s.cache {
            x.cached::<Vec<C>>(&key)?
        } else {
            None
        };
        let result = match cached {
            Some(v) => v,
            None => {
                let input = |slot: usize| slots[slot].as_deref().map(Vec::as_slice).unwrap_or(&[]);
                let t = std::time::Instant::now();
                let v = Arc::new(eval(&s.p, input(s.a), input(s.b), ctl)?);
                if s.cache {
                    x.cache(key, v.len() * 8, t.elapsed().as_secs_f64(), v.clone());
                }
                v
            }
        };
        slots[i + 2] = Some(result);
        for d in dependencies(s).into_iter().filter(|&d| last_use[d] == i) {
            slots[d] = None;
        }
        ctl.report((i + 1) as f64 / recipe.steps.len() as f64, "transform");
    }
    x.words_mut(a.writes[0])
        .copy_from_slice(bytemuck::cast_slice(
            slots[recipe.output].as_ref().unwrap().as_slice(),
        ));
    Ok(())
}
