//! Stable radial Fourier bins; one Bessel evaluation is shared by every image.
use super::*;
#[derive(Clone, Debug)]
pub enum Shells {
    Keys {
        size: usize,
        keys: BufferId,
        ids: BufferId,
    },
    Boundaries {
        keys: PlanView,
        mask: BufferId,
        positions: BufferId,
    },
    Offsets {
        positions: PlanView,
        count: PlanView,
        output: BufferId,
        summary: BufferId,
    },
    Gather {
        spectrum: PlanView,
        order: PlanView,
        center: [Df32; 2],
        normalization: Df32,
        output: BufferId,
    },
    Query {
        values: Vec<PlanView>,
        keys: PlanView,
        radii: PlanView,
        frequency_step: Df32,
        output: BufferId,
    },
}
pub fn validate(r: &Shells, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Shells::Keys { size, keys, ids } => {
            if !(1..=8192).contains(size) {
                return Err(bad("radial bin size"));
            }
            out(p, *keys, U32, &[size * size])?;
            out(p, *ids, U32, &[size * size])
        }
        Shells::Boundaries {
            keys,
            mask,
            positions,
        } => {
            let n = keys.desc.len()?;
            view(keys, U32, &[n])?;
            out(p, *mask, U32, &[n])?;
            out(p, *positions, U32, &[n])
        }
        Shells::Offsets {
            positions,
            count,
            output,
            summary,
        } => {
            let n = positions.desc.len()?;
            view(positions, U32, &[n])?;
            view(count, U32, &[1])?;
            out(p, *output, U32, &[n + 1])?;
            out(p, *summary, U32, &[2])
        }
        Shells::Gather {
            spectrum,
            order,
            center,
            normalization,
            output,
        } => {
            let n = square(spectrum, Complex32)?;
            view(order, U32, &[n * n])?;
            if center.iter().any(|v| !v.is_finite())
                || !normalization.is_finite()
                || normalization.hi <= 0.
            {
                return Err(bad("energy centre/normalization"));
            }
            out(p, *output, DoubleFloat, &[1, n * n])
        }
        Shells::Query {
            values,
            keys,
            radii,
            frequency_step,
            output,
        } => {
            let n = keys.desc.len()?;
            let q = radii.desc.len()?;
            view(keys, U32, &[n])?;
            view(radii, DoubleFloat, &[q])?;
            if values.is_empty()
                || values.len() > 16
                || !frequency_step.is_finite()
                || frequency_step.hi <= 0.
            {
                return Err(bad("radial spectrum query"));
            }
            for v in values {
                view(v, DoubleFloat, &[n])?;
            }
            out(p, *output, DoubleFloat, &[values.len(), q])
        }
    }
}
pub fn access(r: &Shells) -> Access {
    let (reads, writes) = match r {
        Shells::Keys { keys, ids, .. } => (vec![], vec![*keys, *ids]),
        Shells::Boundaries {
            keys,
            mask,
            positions,
        } => (vec![keys.clone()], vec![*mask, *positions]),
        Shells::Offsets {
            positions,
            count,
            output,
            summary,
        } => (
            vec![positions.clone(), count.clone()],
            vec![*output, *summary],
        ),
        Shells::Gather {
            spectrum,
            order,
            output,
            ..
        } => (vec![spectrum.clone(), order.clone()], vec![*output]),
        Shells::Query {
            values,
            keys,
            radii,
            output,
            ..
        } => {
            let mut v = values.clone();
            v.extend([keys.clone(), radii.clone()]);
            (v, vec![*output])
        }
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub size: u32,
    pub length: u32,
    pub mode: u32,
    pub first: u32,
    pub count: u32,
    pub fields: u32,
    pub queries: u32,
    pub query_first: u32,
    pub query_count: u32,
    pub finish: u32,
    pub reserved: [u32; 2],
    pub center: [Df32; 2],
    pub normalization: Df32,
    pub frequency_step: Df32,
}
pub fn params(r: &Shells) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    match r {
        Shells::Keys { size, .. } => {
            p.size = *size as u32;
            p.length = (size * size) as u32;
        }
        Shells::Boundaries { keys, .. } => {
            p.mode = 1;
            p.length = keys.desc.len().unwrap() as u32;
        }
        Shells::Offsets { positions, .. } => {
            p.mode = 2;
            p.length = positions.desc.len().unwrap() as u32;
        }
        Shells::Gather {
            spectrum,
            center,
            normalization,
            ..
        } => {
            p.mode = 3;
            p.size = spectrum.desc.shape[0] as u32;
            p.length = p.size * p.size;
            p.center = *center;
            p.normalization = *normalization;
        }
        Shells::Query {
            values,
            keys,
            radii,
            frequency_step,
            ..
        } => {
            p.mode = 4;
            p.fields = values.len() as u32;
            p.length = keys.desc.len().unwrap() as u32;
            p.queries = radii.desc.len().unwrap() as u32;
            p.frequency_step = *frequency_step;
        }
    }
    p
}
pub fn cpu(r: &Shells, x: &mut crate::resident::cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    match r {
        Shells::Keys { size, keys, ids } => {
            let signed = |i: usize| {
                if 2 * i < *size {
                    i as i32
                } else {
                    i as i32 - *size as i32
                }
            };
            let mut a = vec![0u32; size * size];
            let mut order = a.clone();
            for (k, v) in a.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let i = signed(k % size);
                let j = signed(k / size);
                *v = (i * i + j * j) as u32;
                order[k] = k as u32;
            }
            x.words_mut(*keys).copy_from_slice(&a);
            x.words_mut(*ids).copy_from_slice(&order);
        }
        Shells::Boundaries {
            keys,
            mask,
            positions,
        } => {
            let a = vals::<u32>(x, keys);
            let m: Vec<u32> = a
                .iter()
                .enumerate()
                .map(|(k, v)| u32::from(k == 0 || *v != a[k - 1]))
                .collect();
            let ids: Vec<u32> = (0..a.len() as u32).collect();
            x.words_mut(*mask).copy_from_slice(&m);
            x.words_mut(*positions).copy_from_slice(&ids);
        }
        Shells::Offsets {
            positions,
            count,
            output,
            summary,
        } => {
            let a = vals::<u32>(x, positions);
            let n = vals::<u32>(x, count)[0] as usize;
            let mut out = vec![a.len() as u32; a.len() + 1];
            out[..n].copy_from_slice(&a[..n]);
            let max = (0..n).map(|i| out[i + 1] - out[i]).max().unwrap_or(0);
            x.words_mut(*output).copy_from_slice(&out);
            x.words_mut(*summary).copy_from_slice(&[n as u32, max]);
        }
        Shells::Gather {
            spectrum,
            order,
            center,
            normalization,
            output,
        } => {
            let n = spectrum.desc.shape[0];
            let a = vals::<[f32; 2]>(x, spectrum);
            let ids = vals::<u32>(x, order);
            let mut out = vec![Df32::ZERO; ids.len()];
            for (k, (&i, v)) in ids.iter().zip(&mut out).enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let i = i as usize;
                let signed = |q| {
                    if 2 * q < n {
                        q as i32
                    } else {
                        q as i32 - n as i32
                    }
                };
                let phase = (center[0].mul_f32(signed(i % n) as f32)
                    + center[1].mul_f32(signed(i / n) as f32))
                    / Df32::from_f32(n as f32);
                let z = spec::cis(phase);
                *v = Df32::from_f32(a[i][0] * z[0] - a[i][1] * z[1]) / *normalization;
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Shells::Query {
            values,
            keys,
            radii,
            frequency_step,
            output,
        } => {
            let a: Vec<&[Df32]> = values.iter().map(|v| vals(x, v)).collect();
            let keys = vals::<u32>(x, keys);
            let radii = vals::<Df32>(x, radii);
            let mut out = vec![Df32::ZERO; a.len() * radii.len()];
            for (q, &r) in radii.iter().enumerate() {
                let r = spec::scalar(r);
                if r <= 0. {
                    continue;
                }
                for (k, &key) in keys.iter().enumerate() {
                    if k % 512 == 0 {
                        ctl.check()?;
                    }
                    let rho = spec::scalar(Df32::from_f32(key as f32).sqrt() * *frequency_step);
                    let weight = if rho * r < 1e-12 {
                        std::f32::consts::PI * r * r
                    } else {
                        r * super::image_integrals::j1(std::f32::consts::TAU * rho * r) / rho
                    };
                    for (i, a) in a.iter().enumerate() {
                        out[i * radii.len() + q] = out[i * radii.len() + q] + a[k].mul_f32(weight);
                    }
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
    }
    Ok(())
}
