//! Sensor angular-spectrum sampling and the reference FFT-OTF interpolation.
use super::*;
#[derive(Clone, Debug)]
pub enum FftMtf {
    Domain {
        coverage: PlanView,
        sin_max: Df32,
        normal: [Df32; 3],
        status: BufferId,
    },
    Sensor {
        pupil: Pupil,
        sin_max: Df32,
        frame: [[Df32; 3]; 3],
        field: BufferId,
        moments: BufferId,
    },
    Normalize {
        input: PlanView,
        sums: PlanView,
        output: BufferId,
    },
    Sample {
        data: PlanView,
        reference: PlanView,
        frequencies: PlanView,
        spacing: Df32,
        mapping: [[Df32; 2]; 2],
        output: BufferId,
    },
}
pub fn validate(r: &FftMtf, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        FftMtf::Domain {
            coverage,
            sin_max,
            normal,
            status,
        } => {
            square(coverage, F32)?;
            if !sin_max.is_finite()
                || sin_max.hi <= 0.
                || sin_max.hi >= 1.
                || normal.iter().any(|v| !v.is_finite())
            {
                return Err(bad("sensor domain"));
            }
            out(p, *status, U32, &[1])
        }
        FftMtf::Sensor {
            pupil,
            sin_max,
            frame,
            field,
            moments,
        } => {
            let n = square(&pupil.amplitude, F32)?;
            view(&pupil.coverage, F32, &[n, n])?;
            view(&pupil.opd, DoubleFloat, &[n, n])?;
            let d = p.desc(*field)?;
            if d.element != Complex32
                || d.shape.len() != 2
                || d.shape[0] != d.shape[1]
                || d.shape[0] < n
                || d.shape[0] > 8192
                || !sin_max.is_finite()
                || sin_max.hi <= 0.
                || sin_max.hi >= 1.
                || frame.iter().flatten().any(|v| !v.is_finite())
            {
                return Err(bad("sensor spectrum geometry"));
            }
            out(p, *moments, F32, &[3, n * n])
        }
        FftMtf::Normalize {
            input,
            sums,
            output,
        } => {
            let n = square(input, Complex32)?;
            view(sums, DoubleFloat, &[3])?;
            out(p, *output, Complex32, &[n, n])
        }
        FftMtf::Sample {
            data,
            reference,
            frequencies,
            spacing,
            mapping,
            output,
        } => {
            let n = square(data, Complex32)?;
            view(reference, Complex32, &[n, n])?;
            let count = frequencies.desc.shape[0];
            view(frequencies, F32, &[count, 2])?;
            if n < 4
                || !spacing.is_finite()
                || spacing.hi <= 0.
                || mapping.iter().flatten().any(|v| !v.is_finite())
            {
                return Err(bad("OTF interpolation mapping"));
            }
            out(p, *output, Complex32, &[count])
        }
    }
}
pub fn access(r: &FftMtf) -> Access {
    let (reads, writes) = match r {
        FftMtf::Domain {
            coverage, status, ..
        } => (vec![coverage.clone()], vec![*status]),
        FftMtf::Sensor {
            pupil,
            field,
            moments,
            ..
        } => (
            vec![
                pupil.amplitude.clone(),
                pupil.coverage.clone(),
                pupil.opd.clone(),
            ],
            vec![*field, *moments],
        ),
        FftMtf::Normalize {
            input,
            sums,
            output,
        } => (vec![input.clone(), sums.clone()], vec![*output]),
        FftMtf::Sample {
            data,
            reference,
            frequencies,
            output,
            ..
        } => (
            vec![data.clone(), reference.clone(), frequencies.clone()],
            vec![*output],
        ),
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
    pub n: u32,
    pub side: u32,
    pub count: u32,
    pub kind: u32,
    pub sine: Df32,
    pub spacing: Df32,
    pub frame: [[Df32; 3]; 3],
    pub mapping: [[Df32; 2]; 2],
}
pub fn params(r: &FftMtf) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    match r {
        FftMtf::Domain {
            coverage,
            sin_max,
            normal,
            ..
        } => {
            p.n = coverage.desc.shape[0] as u32;
            p.count = p.n * p.n;
            p.kind = 3;
            p.sine = *sin_max;
            p.frame[0] = *normal;
        }
        FftMtf::Sensor {
            pupil,
            sin_max,
            frame,
            ..
        } => {
            p.n = pupil.amplitude.desc.shape[0] as u32;
            p.sine = *sin_max;
            p.frame = *frame;
        }
        FftMtf::Normalize { input, .. } => {
            p.n = input.desc.shape[0] as u32;
            p.kind = 1;
        }
        FftMtf::Sample {
            data,
            frequencies,
            spacing,
            mapping,
            ..
        } => {
            p.n = data.desc.shape[0] as u32;
            p.kind = 2;
            p.count = frequencies.desc.shape[0] as u32;
            p.spacing = *spacing;
            p.mapping = *mapping;
        }
    }
    p
}
pub fn sensor(p: &Params, k: usize, a: &[f32], c: &[f32], w: &[Df32]) -> ([f32; 2], f32) {
    let n = p.n as usize;
    let one = Df32::from_f32(1.);
    let coord = |i| Df32::from_f32((2 * i + 1) as f32) / Df32::from_f32(n as f32) - one;
    let al = p.sine * coord(k % n);
    let be = p.sine * coord(k / n);
    let g = (one - p.sine * p.sine).sqrt();
    let x = p.frame[0][0] * al + p.frame[1][0] * be - p.frame[2][0] * g;
    let y = p.frame[0][1] * al + p.frame[1][1] * be - p.frame[2][1] * g;
    let z2 = one - x * x - y * y;
    if z2.hi <= 0. {
        return ([0.; 2], 0.);
    }
    let q = [x, y, -z2.sqrt()];
    if spec::dot(q, p.frame[2]).hi >= 0. {
        return ([0.; 2], 0.);
    }
    let scale = Df32::from_f32(n as f32 * 0.5) / p.sine;
    let ix = spec::scalar((spec::dot(q, p.frame[0]) * scale).add_f32((n - 1) as f32 * 0.5));
    let iy = spec::scalar((spec::dot(q, p.frame[1]) * scale).add_f32((n - 1) as f32 * 0.5));
    let fx = ix.floor();
    let fy = iy.floor();
    let tx = ix - fx;
    let ty = iy - fy;
    let (mut amp, mut cov, mut opd) = (0., 0., Df32::ZERO);
    for (dx, dy, wt) in [
        (0, 0, (1. - tx) * (1. - ty)),
        (1, 0, tx * (1. - ty)),
        (0, 1, (1. - tx) * ty),
        (1, 1, tx * ty),
    ] {
        let (x, y) = (fx as i64 + dx, fy as i64 + dy);
        if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
            continue;
        }
        let j = y as usize * n + x as usize;
        let weight = wt * c[j];
        amp += weight * a[j];
        cov += weight;
        opd = opd + w[j].mul_f32(weight);
    }
    if cov <= 0. {
        return ([0.; 2], 0.);
    }
    let qz = -spec::scalar(q[2]);
    let z = spec::cis(opd / Df32::from_f32(cov));
    (
        [amp / qz * z[0], amp / qz * z[1]],
        amp * amp / (cov * qz * qz),
    )
}
fn mul(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn keys(x: f32) -> f32 {
    let x = x.abs();
    if x <= 1. {
        (1.5 * x - 2.5) * x * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
pub fn sample(p: &Params, data: &[[f32; 2]], dl: &[[f32; 2]], nu: [f32; 2]) -> [f32; 2] {
    let n = p.n as usize;
    let center = n / 2;
    let d = spec::scalar(p.spacing);
    let centroid = |z: [f32; 2]| {
        if z[0].hypot(z[1]) > 1e-9 {
            -z[1].atan2(z[0]) / (std::f32::consts::TAU * d)
        } else {
            0.
        }
    };
    let cx = centroid(data[center * n + center + 1]);
    let cy = centroid(data[(center + 1) * n + center]);
    let nx = spec::scalar(p.mapping[0][0].mul_f32(nu[0]) + p.mapping[0][1].mul_f32(nu[1]));
    let ny = spec::scalar(p.mapping[1][0].mul_f32(nu[0]) + p.mapping[1][1].mul_f32(nu[1]));
    let x = nx / d + center as f32;
    let y = ny / d + center as f32;
    let ix = x.floor();
    let iy = y.floor();
    let tx = x - ix;
    let ty = y - iy;
    let mut sum = [0.; 2];
    let at = |v: &[[f32; 2]], i: i64, j: i64| {
        if i < 0 || j < 0 || i >= n as i64 || j >= n as i64 {
            [0.; 2]
        } else {
            v[j as usize * n + i as usize]
        }
    };
    for by in -1..=2 {
        for bx in -1..=2 {
            let i = ix as i64 + bx;
            let j = iy as i64 + by;
            let phase = Df32::from_f32(d)
                * (Df32::from_f32((i - center as i64) as f32).mul_f32(cx)
                    + Df32::from_f32((j - center as i64) as f32).mul_f32(cy));
            let z = mul(at(data, i, j), spec::cis(phase));
            let r = at(dl, i, j);
            let wt = keys(tx - bx as f32) * keys(ty - by as f32);
            sum[0] += (z[0] - r[0]) * wt;
            sum[1] += (z[1] - r[1]) * wt;
        }
    }
    for (dx, dy, wt) in [
        (0, 0, (1. - tx) * (1. - ty)),
        (1, 0, tx * (1. - ty)),
        (0, 1, (1. - tx) * ty),
        (1, 1, tx * ty),
    ] {
        let z = at(dl, ix as i64 + dx, iy as i64 + dy);
        sum[0] += z[0] * wt;
        sum[1] += z[1] * wt;
    }
    mul(
        sum,
        spec::cis(-(Df32::from_f32(nx).mul_f32(cx) + Df32::from_f32(ny).mul_f32(cy))),
    )
}
pub fn cpu(r: &FftMtf, x: &mut crate::resident::cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    let p = params(r);
    let n = p.n as usize;
    match r {
        FftMtf::Domain {
            coverage,
            sin_max,
            normal,
            status,
        } => {
            let c = values::<f32>(x, coverage);
            let mut count = 0u32;
            for (k, &c) in c.iter().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if c <= 0. {
                    continue;
                }
                let coord = |i| {
                    Df32::from_f32((2 * i + 1) as f32) / Df32::from_f32(n as f32)
                        - Df32::from_f32(1.)
                };
                let a = *sin_max * coord(k % n);
                let b = *sin_max * coord(k / n);
                let gamma = (Df32::from_f32(1.) - a * a - b * b).sqrt();
                let z = normal[0] * a + normal[1] * b - normal[2] * gamma;
                if !z.is_finite() || z.hi >= 0. {
                    count += 1;
                }
            }
            x.words_mut(*status)[0] = count;
            if count > 0 {
                return Err(mklight_core::MkError::InvalidParam(format!(
                    "pupil crosses image-surface horizon at {count} covered cells"
                )));
            }
        }

        FftMtf::Sensor {
            pupil,
            field,
            moments,
            ..
        } => {
            let side = (x.words(*field).len() / 2).isqrt();
            let a = values::<f32>(x, &pupil.amplitude);
            let c = values::<f32>(x, &pupil.coverage);
            let w = values::<Df32>(x, &pupil.opd);
            let mut f = vec![[0.; 2]; side * side];
            let mut m = vec![0.; 3 * n * n];
            for k in 0..n * n {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let (z, e) = sensor(&p, k, a, c, w);
                f[k / n * side + k % n] = z;
                m[n * n + k] = e;
            }
            x.words_mut(*field)
                .copy_from_slice(bytemuck::cast_slice(&f));
            x.words_mut(*moments)
                .copy_from_slice(bytemuck::cast_slice(&m));
        }
        FftMtf::Normalize {
            input,
            sums,
            output,
        } => {
            let a = values::<[f32; 2]>(x, input);
            let e = values::<Df32>(x, sums)[1].mul_f32((n * n) as f32);
            let mut out = vec![[0.; 2]; n * n];
            for (k, z) in out.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let x = (k % n + n - n / 2) % n;
                let y = (k / n + n - n / 2) % n;
                *z = a[y * n + x].map(|v| spec::scalar(Df32::from_f32(v) / e));
            }
            out[n / 2 * n + n / 2] = [1., 0.];
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        FftMtf::Sample {
            data,
            reference,
            frequencies,
            output,
            ..
        } => {
            let a = values(x, data);
            let dl = values(x, reference);
            let fs = values::<f32>(x, frequencies).as_chunks::<2>().0;
            let mut out = vec![[0.; 2]; fs.len()];
            for (k, (z, f)) in out.iter_mut().zip(fs).enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                *z = sample(&p, a, dl, *f);
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
    }
    Ok(())
}
