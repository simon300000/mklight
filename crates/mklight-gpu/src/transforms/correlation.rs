//! Correlation numeric records and CPU-f32 specification. Geometry stencils
//! are explicit resident outputs, so callers can reuse them across focus
//! planes without relying on a cache key for mutable frequency data.
use super::*;
use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub n: u32,
    pub height: u32,
    /// Frequencies in the current tile, starting at `first`.
    pub frequencies: u32,
    pub first: u32,
    pub optical: u32,
    pub covered: u32,
    pub prepared: u32,
    pub reserved: u32,
    pub mapping: [[f32; 2]; 6],
    pub frame: [[[f32; 2]; 3]; 3],
    pub sin_max: [f32; 2],
    pub wavelength: [f32; 2],
    pub reserved_tail: [u32; 2],
}
fn pack(v: Df32) -> [f32; 2] {
    [v.hi, v.lo]
}
fn df(v: [f32; 2]) -> Df32 {
    Df32::new(v[0], v[1])
}
pub fn params(r: &Request) -> Params {
    match r {
        Request::Correlation(c) => {
            let (n, h) = recipe::shape(&c.pupil.desc.shape);
            Params {
                n: n as u32,
                height: h as u32,
                mapping: c.mapping.map(pack),
                covered: c.coverage.is_some() as u32,
                ..Params::zeroed()
            }
        }
        Request::OpticalCorrelation(c) | Request::PrepareOpticalStencil(c) => Params {
            n: c.amplitude.desc.shape[0] as u32,
            height: c.amplitude.desc.shape[0] as u32,
            optical: 1,
            covered: 1,
            prepared: c.stencil.is_some() as u32,
            frame: c.frame.map(|v| v.map(pack)),
            sin_max: pack(c.sin_max),
            wavelength: pack(c.wavelength_over_index),
            ..Params::zeroed()
        },
        _ => unreachable!(),
    }
}
fn dot(a: [Df32; 3], b: [Df32; 3]) -> Df32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// [shifted x, shifted y, |shifted qz|, gamma, |source qz|].
pub fn geometry(p: &Params, k: usize, nu: [f32; 2]) -> [f32; 5] {
    let n = p.n as usize;
    let d = |v: f32| Df32::from_f32(v);
    let a = df(p.sin_max).mul_f32((2 * (k % n) + 1) as f32 / n as f32 - 1.0);
    let b = df(p.sin_max).mul_f32((2 * (k / n) + 1) as f32 / n as f32 - 1.0);
    let gamma = (d(1.) - a * a - b * b).sqrt();
    let q: [Df32; 3] = std::array::from_fn(|i| {
        df(p.frame[0][i]) * a + df(p.frame[1][i]) * b - df(p.frame[2][i]) * gamma
    });
    let sx = q[0] + df(p.wavelength).mul_f32(nu[0]);
    let sy = q[1] + df(p.wavelength).mul_f32(nu[1]);
    let z2 = d(1.) - sx * sx - sy * sy;
    if z2.hi <= 0. || q[2].hi >= 0. || !gamma.is_finite() {
        return [f32::NAN, f32::NAN, 0., gamma.hi, -q[2].hi];
    }
    let sz = -z2.sqrt();
    let shifted = [sx, sy, sz];
    if dot(shifted, p.frame[2].map(df)).hi >= 0. {
        return [f32::NAN, f32::NAN, 0., gamma.hi, -q[2].hi];
    }
    let scale = d(0.5 * n as f32) / df(p.sin_max);
    let offset = 0.5 * (n - 1) as f32;
    let ix = (dot(shifted, p.frame[0].map(df)) * scale).add_f32(offset);
    let iy = (dot(shifted, p.frame[1].map(df)) * scale).add_f32(offset);
    [ix.hi + ix.lo, iy.hi + iy.lo, -sz.hi, gamma.hi, -q[2].hi]
}
fn values<T: Pod + Copy>(x: &crate::resident::cpu::Execution, v: &PlanView) -> Vec<T> {
    let a: &[T] = bytemuck::cast_slice(x.words(v.buffer));
    a[v.offset..v.offset + v.desc.len().unwrap()].to_vec()
}
pub(super) fn cpu(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    let p = params(r);
    let n = p.n as usize;
    let h = p.height as usize;
    let count = n * h;
    let (freq, output) = match r {
        Request::Correlation(c) => (&c.frequencies, c.output),
        Request::OpticalCorrelation(c) | Request::PrepareOpticalStencil(c) => {
            (&c.frequencies, c.output)
        }
        _ => unreachable!(),
    };
    let fs = values::<f32>(x, freq);
    let mut prepared = vec![];
    if let Request::PrepareOpticalStencil(_) = r {
        for nu in fs.as_chunks::<2>().0 {
            for k in 0..count {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                prepared.extend(geometry(&p, k, [nu[0], nu[1]]));
            }
            ctl.report(
                prepared.len() as f64 / (fs.len() / 2 * count * 5) as f64,
                "correlation stencil",
            );
        }
        x.words_mut(output)
            .copy_from_slice(bytemuck::cast_slice(&prepared));
        return Ok(());
    }
    let (field, amp, opd, cov, stencil) = match r {
        Request::Correlation(c) => (
            values::<[f32; 2]>(x, &c.pupil),
            vec![],
            vec![],
            c.coverage
                .as_ref()
                .map(|v| values::<f32>(x, v))
                .unwrap_or(vec![1.; count]),
            vec![],
        ),
        Request::OpticalCorrelation(c) => (
            vec![],
            values::<f32>(x, &c.amplitude),
            values::<Df32>(x, &c.opd),
            values::<f32>(x, &c.coverage),
            c.stencil
                .as_ref()
                .map(|v| values::<f32>(x, v))
                .unwrap_or_default(),
        ),
        _ => unreachable!(),
    };
    let mut out = vec![];
    for (f, nu) in fs.as_chunks::<2>().0.iter().enumerate() {
        let mut channels = vec![0f32; 3 * count];
        for k in 0..count {
            if k % 4096 == 0 {
                ctl.check()?;
            }
            if cov[k] <= 0. {
                continue;
            }
            let (ix, iy, qz, gamma, srcz) = if p.optical != 0 {
                let g = if stencil.is_empty() {
                    geometry(&p, k, [nu[0], nu[1]])
                } else {
                    stencil[(f * count + k) * 5..(f * count + k + 1) * 5]
                        .try_into()
                        .unwrap()
                };
                (g[0], g[1], g[2], g[3], g[4])
            } else {
                let sx = df(p.mapping[0]).mul_f32(nu[0])
                    + df(p.mapping[1]).mul_f32(nu[1])
                    + df(p.mapping[2]);
                let sy = df(p.mapping[3]).mul_f32(nu[0])
                    + df(p.mapping[4]).mul_f32(nu[1])
                    + df(p.mapping[5]);
                (
                    (k % n) as f32 + sx.hi + sx.lo,
                    (k / n) as f32 + sy.hi + sy.lo,
                    1.,
                    1.,
                    1.,
                )
            };
            let power = if p.optical != 0 {
                amp[k] * amp[k]
            } else {
                field[k][0] * field[k][0] + field[k][1] * field[k][1]
            };
            let energy = cov[k] * power / (gamma * srcz);
            channels[2 * count + k] = energy;
            if !ix.is_finite() || !iy.is_finite() {
                continue;
            }
            let (fx, fy) = (ix.floor(), iy.floor());
            let (tx, ty) = (ix - fx, iy - fy);
            let (mut ar, mut ai, mut coverage, mut phase) = (0., 0., 0., Df32::ZERO);
            for (dx, dy, wt) in [
                (0, 0, (1. - tx) * (1. - ty)),
                (1, 0, tx * (1. - ty)),
                (0, 1, (1. - tx) * ty),
                (1, 1, tx * ty),
            ] {
                let (xi, yi) = (fx as i64 + dx, fy as i64 + dy);
                if xi < 0 || yi < 0 || xi >= n as i64 || yi >= h as i64 {
                    continue;
                }
                let j = yi as usize * n + xi as usize;
                let w = wt * cov[j];
                if w == 0. {
                    continue;
                }
                if p.optical != 0 {
                    ar += w * amp[j];
                    coverage += w;
                    phase = phase + opd[j].mul_f32(w);
                } else {
                    ar += w * field[j][0];
                    ai += w * field[j][1];
                }
            }
            let deficit = (1. - cov[k])
                * energy
                * (1. - (ix - (k % n) as f32).abs()).max(0.)
                * (1. - (iy - (k / n) as f32).abs()).max(0.);
            let v = if p.optical != 0 {
                if coverage <= 0. {
                    [0., 0.]
                } else {
                    let z = super::cpu::cis(phase / Df32::from_f32(coverage) - opd[k]);
                    let m = cov[k] * amp[k] / gamma * ar / qz;
                    [m * z[0], m * z[1]]
                }
            } else {
                [
                    cov[k] * (ar * field[k][0] + ai * field[k][1]),
                    cov[k] * (ai * field[k][0] - ar * field[k][1]),
                ]
            };
            channels[k] = v[0] + deficit;
            channels[count + k] = v[1];
        }
        let sums: Vec<_> = channels
            .chunks(count)
            .map(|v| {
                crate::reduce::pass2_in::<crate::fenv::Ftz>(
                    crate::ReduceMode::DoubleFloat,
                    &crate::reduce::pass1_in::<crate::fenv::Ftz>(
                        crate::ReduceMode::DoubleFloat,
                        v,
                        0..crate::reduce::num_groups(count),
                    ),
                )
            })
            .collect();
        let a = sums[0] / sums[2];
        let b = sums[1] / sums[2];
        out.push([a.hi + a.lo, b.hi + b.lo]);
        ctl.report((f + 1) as f64 / (fs.len() / 2) as f64, "correlation");
    }
    x.words_mut(output)
        .copy_from_slice(bytemuck::cast_slice(&out));
    Ok(())
}
