use super::{recipe::*, *};
use crate::consts::*;
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
fn quadratic(a: &[Df32; 6], x: f32, y: f32) -> Df32 {
    (a[0].mul_f32(x).mul_f32(x) + a[1].mul_f32(2.0 * x).mul_f32(y) + a[4].mul_f32(y).mul_f32(y))
        .mul_f32(0.5)
}
fn eval(s: &Step, a: &[C], b: &[C], ctl: &JobControl) -> MkResult<Vec<C>> {
    let p = s.p;
    let affine = p.affine.map(|v| Df32::new(v[0], v[1]));
    let (w, h) = (p.width as usize, p.height as usize);
    let (n, m) = (p.nx as usize, p.ny as usize);
    let (ow, oh) = (p.ox as usize, p.oy as usize);
    let sign = p.sign_bits as i32 as f32;
    let scale = f32::from_bits(p.scale_bits);
    let mut out = vec![[0.; 2]; p.count as usize];
    for (id, z) in out.iter_mut().enumerate() {
        if id % 4096 == 0 {
            ctl.check()?;
        }
        *z = match p.mode {
            MKL_TRANSFORM_COPY => a[id].map(|v| v * scale),
            MKL_TRANSFORM_TRANSPOSE => a[(id % h) * w + id / h],
            MKL_TRANSFORM_TWIDDLE => cis(Df32::from_f64(sign as f64 * id as f64 / w as f64)),
            MKL_TRANSFORM_FFT => {
                let k = id % w;
                let row = id / w;
                let r = p.radix as usize;
                let span = p.span as usize;
                let i = k % span;
                let j = (k / span) % r;
                let g = k / (span * r);
                let (mut re, mut im) = (Df32::ZERO, Df32::ZERO);
                for t in 0..r {
                    let v = mul(
                        a[row * w + g * span + i + t * w / r],
                        b[(t * (i + j * span) * (w / (r * span))) % w],
                    );
                    re = re.add_f32(v[0]);
                    im = im.add_f32(v[1]);
                }
                [re.hi + re.lo, im.hi + im.lo]
            }
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
                if let (Some(x), Some(y)) = (lag(id % w, w, n, ow), lag(id / w, h, m, oh)) {
                    cis(quadratic(&affine, x, y).mul_f32(-sign))
                } else {
                    [0.; 2]
                }
            }
            MKL_TRANSFORM_PRE => {
                let (x, y) = (id % w, id / w);
                if x < n && y < m {
                    let ph = quadratic(&affine, x as f32, y as f32)
                        + affine[2].mul_f32(x as f32)
                        + affine[5].mul_f32(y as f32);
                    mul(a[y * n + x], cis(ph.mul_f32(sign)))
                } else {
                    [0.; 2]
                }
            }
            MKL_TRANSFORM_POST => {
                let (x, y) = (id % ow, id / ow);
                mul(
                    a[y * w + x],
                    cis(quadratic(&affine, x as f32, y as f32).mul_f32(sign)),
                )
                .map(|v| v * scale)
            }
            MKL_TRANSFORM_MULTIPLY => mul(a[id], {
                let v = b[if p.reserved0 != 0 { id % w } else { id }];
                if p.conjugate != 0 { [v[0], -v[1]] } else { v }
            })
            .map(|v| v * scale),
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
    let mut uses = vec![0usize; slots.len()];
    for s in &recipe.steps {
        for d in dependencies(s) {
            uses[d] += 1;
        }
    }
    let keys = recipe.cache_keys();
    for (i, s) in recipe.steps.iter().enumerate() {
        ctl.check()?;
        let key = x.cache_key(keys[i].clone());
        let cached = if s.cache {
            x.cached::<Vec<C>>(&key)?
        } else {
            None
        };
        let result = if let Some(v) = cached {
            v
        } else {
            let v = Arc::new(eval(
                s,
                slots[s.a].as_deref().map(Vec::as_slice).unwrap_or(&[]),
                slots[s.b].as_deref().map(Vec::as_slice).unwrap_or(&[]),
                ctl,
            )?);
            if s.cache {
                x.cache(key, v.len() * 8, v.clone());
            }
            v
        };
        slots[i + 2] = Some(result);
        for d in dependencies(s) {
            uses[d] -= 1;
            if d >= 2 && uses[d] == 0 && d != recipe.output {
                slots[d] = None;
            }
        }
        ctl.report((i + 1) as f64 / recipe.steps.len() as f64, "transform");
    }
    x.words_mut(a.writes[0])
        .copy_from_slice(bytemuck::cast_slice(
            slots[recipe.output].as_ref().unwrap().as_slice(),
        ));
    Ok(())
}
