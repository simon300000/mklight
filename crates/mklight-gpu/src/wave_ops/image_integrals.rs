//! Fourier-domain image integrals; finite-window normalization is explicit.
use super::*;
#[derive(Clone, Debug)]
pub enum ImageIntegral {
    Metrics {
        input: PlanView,
        output: BufferId,
    },
    Complex {
        input: PlanView,
        squared: bool,
        output: BufferId,
    },
    /// kind 0: PSF DFT at (fx,fy); 1/2: circular/square spectral integral
    /// at radius query.x. Center is in pixels relative to the low-corner sample.
    Query {
        input: PlanView,
        queries: PlanView,
        kind: u32,
        spacing: Df32,
        center: [Df32; 2],
        normalization: Df32,
        output: BufferId,
    },
}
pub fn validate(r: &ImageIntegral, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        ImageIntegral::Metrics { input, output } => {
            square(input, F32)?;
            out(p, *output, DoubleFloat, &[5])
        }
        ImageIntegral::Complex {
            input,
            squared,
            output,
        } => {
            let n = square(input, if *squared { Complex32 } else { F32 })?;
            out(p, *output, Complex32, &[n, n])
        }
        ImageIntegral::Query {
            input,
            queries,
            kind,
            spacing,
            center,
            normalization,
            output,
        } => {
            let n = square(input, if *kind == 0 { F32 } else { Complex32 })?;
            let _ = n;
            let count = queries.desc.shape[0];
            view(queries, DoubleFloat, &[count, 2])?;
            if *kind > 2
                || !spacing.is_finite()
                || spacing.hi <= 0.
                || !normalization.is_finite()
                || normalization.hi <= 0.
                || center.iter().any(|v| !v.is_finite())
            {
                return Err(bad("image integral settings"));
            }
            out(p, *output, Complex32, &[count])
        }
    }
}
pub fn access(r: &ImageIntegral) -> Access {
    let (reads, output) = match r {
        ImageIntegral::Metrics { input, output } => (vec![input.clone()], *output),
        ImageIntegral::Complex { input, output, .. } => (vec![input.clone()], *output),
        ImageIntegral::Query {
            input,
            queries,
            output,
            ..
        } => (vec![input.clone(), queries.clone()], *output),
    };
    Access {
        reads,
        writes: vec![output],
        systems: vec![],
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub n: u32,
    pub queries: u32,
    pub kind: u32,
    pub first: u32,
    pub count: u32,
    pub finish: u32,
    pub reserved: [u32; 2],
    pub spacing: Df32,
    pub center: [Df32; 2],
    pub normalization: Df32,
}
pub fn params(r: &ImageIntegral) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    match r {
        ImageIntegral::Metrics { input, .. } => {
            p.n = input.desc.shape[0] as u32;
            p.kind = 5;
        }
        ImageIntegral::Complex { input, squared, .. } => {
            p.n = input.desc.shape[0] as u32;
            p.kind = 3 + u32::from(*squared);
        }
        ImageIntegral::Query {
            input,
            queries,
            kind,
            spacing,
            center,
            normalization,
            ..
        } => {
            p.n = input.desc.shape[0] as u32;
            p.queries = queries.desc.shape[0] as u32;
            p.kind = *kind;
            p.spacing = *spacing;
            p.center = *center;
            p.normalization = *normalization;
        }
    }
    p
}
pub fn j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax <= 3. {
        let t = (x / 3.) * (x / 3.);
        x * (0.5
            + t * (-0.562_499_85
                + t * (0.210_935_73
                    + t * (-0.039_542_89
                        + t * (0.004_433_19 + t * (-0.000_317_61 + t * 0.000_011_09))))))
    } else {
        let y = 3. / ax;
        let f = 0.797_884_56
            + y * (0.000_001_56
                + y * (0.016_596_67
                    + y * (0.000_171_05
                        + y * (-0.002_495_11 + y * (0.001_136_53 - y * 0.000_200_33)))));
        let th = ax - 2.356_194_49
            + y * (0.124_996_12
                + y * (0.000_056_50
                    + y * (-0.006_378_79
                        + y * (0.000_743_48 + y * (0.000_798_24 - y * 0.000_291_66)))));
        let r = f * th.cos() / ax.sqrt();
        if x < 0. { -r } else { r }
    }
}
pub fn term(p: &Params, k: usize, q: [Df32; 2], v: [f32; 2]) -> [f32; 2] {
    let n = p.n as usize;
    let one = Df32::from_f32(1.);
    let x = Df32::from_f32((k % n) as f32);
    let y = Df32::from_f32((k / n) as f32);
    if p.kind == 0 {
        let phase = -((x - p.center[0]) * q[0] + (y - p.center[1]) * q[1]) * p.spacing;
        let z = spec::cis(phase);
        return [v[0] * z[0], v[0] * z[1]];
    }
    let signed = |i: usize| {
        if i <= (n - 1) / 2 {
            i as i32
        } else {
            i as i32 - n as i32
        }
    };
    let ix = signed(k % n);
    let iy = signed(k / n);
    let inv = one / (Df32::from_f32(n as f32) * p.spacing);
    let fx = inv.mul_f32(ix as f32);
    let fy = inv.mul_f32(iy as f32);
    let phase = (p.center[0].mul_f32(ix as f32) + p.center[1].mul_f32(iy as f32))
        / Df32::from_f32(n as f32);
    let z = spec::cis(phase);
    let real = v[0] * z[0] - v[1] * z[1];
    let radius = spec::scalar(q[0]);
    let kernel = if radius <= 0. {
        0.
    } else if p.kind == 1 {
        let rho = spec::scalar((fx * fx + fy * fy).sqrt());
        if rho * radius < 1e-12 {
            std::f32::consts::PI * radius * radius
        } else {
            radius * j1(std::f32::consts::TAU * rho * radius) / rho
        }
    } else {
        let sinc = |f: Df32| {
            let f = spec::scalar(f);
            if f == 0. {
                2. * radius
            } else {
                (std::f32::consts::TAU * f * radius).sin() / (std::f32::consts::PI * f)
            }
        };
        sinc(fx) * sinc(fy)
    };
    [real * kernel, 0.]
}
pub fn cpu(
    r: &ImageIntegral,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    match r {
        ImageIntegral::Metrics { input, output } => {
            let a = vals::<f32>(x, input);
            let n = input.desc.shape[0];
            let mut partial = vec![[Df32::ZERO; 5]; a.len().div_ceil(256)];
            for (k, &v) in a.iter().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let row = &mut partial[k / 256];
                row[0] = row[0].add_f32(v);
                row[1] = row[1] + Df32::from_f32(v).mul_f32((k % n) as f32);
                row[2] = row[2] + Df32::from_f32(v).mul_f32((k / n) as f32);
                if k % 256 == 0 || v > row[3].hi {
                    row[3] = Df32::from_f32(v);
                    row[4] = Df32::from_f32(k as f32);
                }
            }
            let mut sum = partial[0];
            for row in &partial[1..] {
                for j in 0..3 {
                    sum[j] = sum[j] + row[j];
                }
                if row[3].hi > sum[3].hi {
                    sum[3] = row[3];
                    sum[4] = row[4];
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&sum));
        }
        ImageIntegral::Complex {
            input,
            squared,
            output,
        } => {
            let n = input.desc.len()?;
            let mut out = vec![[0.; 2]; n];
            if *squared {
                let a = vals::<[f32; 2]>(x, input);
                for (k, (v, z)) in out.iter_mut().zip(a).enumerate() {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    v[0] = z[0] * z[0] + z[1] * z[1];
                }
            } else {
                let a = vals::<f32>(x, input);
                for (k, (v, z)) in out.iter_mut().zip(a).enumerate() {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    v[0] = *z;
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        ImageIntegral::Query {
            input,
            queries,
            output,
            ..
        } => {
            let p = params(r);
            let q = vals::<Df32>(x, queries).as_chunks::<2>().0;
            let a = vals::<f32>(x, input);
            let complex = if p.kind == 0 {
                None
            } else {
                Some(vals::<[f32; 2]>(x, input))
            };
            let mut out = vec![[0.; 2]; q.len()];
            for (i, q) in q.iter().enumerate() {
                ctl.check()?;
                let sum = spec::blocked_sum((p.n * p.n) as usize, |k| {
                    term(&p, k, *q, complex.map_or([a[k], 0.], |a| a[k]))
                });
                out[i] = if p.kind == 0 && q[0].hi == 0. && q[1].hi == 0. {
                    [1., 0.]
                } else {
                    sum.map(|v| spec::scalar(v / p.normalization))
                };
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
    }
    Ok(())
}
