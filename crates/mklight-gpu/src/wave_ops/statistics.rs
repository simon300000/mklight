//! Wavefront statistics with Debye weights and a P3 QR tilt fit.
use super::*;
#[derive(Clone, Debug)]
pub enum Statistics {
    Prepare {
        pupil: Pupil,
        geometry: Geometry,
        samples: BufferId,
        sums: BufferId,
    },
    Residual {
        samples: PlanView,
        sums: PlanView,
        solution: PlanView,
        output: BufferId,
    },
}
pub fn validate(r: &Statistics, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Statistics::Prepare {
            pupil: v,
            geometry,
            samples,
            sums,
        } => {
            let n = pupil(v, *geometry)?;
            out(p, *samples, DoubleFloat, &[n * n, 4])?;
            out(p, *sums, DoubleFloat, &[4])
        }
        Statistics::Residual {
            samples,
            sums,
            solution,
            output,
        } => {
            let n = samples.desc.shape.first().copied().unwrap_or(0);
            view(samples, DoubleFloat, &[n, 4])?;
            view(sums, DoubleFloat, &[4])?;
            view(solution, DoubleFloat, &[1, 3])?;
            out(p, *output, DoubleFloat, &[5])
        }
    }
}
pub fn access(r: &Statistics) -> Access {
    let (reads, writes) = match r {
        Statistics::Prepare {
            pupil,
            samples,
            sums,
            ..
        } => (
            vec![
                pupil.amplitude.clone(),
                pupil.coverage.clone(),
                pupil.opd.clone(),
            ],
            vec![*samples, *sums],
        ),
        Statistics::Residual {
            samples,
            sums,
            solution,
            output,
        } => (
            vec![samples.clone(), sums.clone(), solution.clone()],
            vec![*output],
        ),
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
pub fn cpu(
    r: &Statistics,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    fn values<'a, T: bytemuck::Pod>(
        x: &'a crate::resident::cpu::Execution,
        v: &PlanView,
    ) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    match r {
        Statistics::Prepare {
            pupil,
            geometry,
            samples,
            sums,
        } => {
            let a = values::<f32>(x, &pupil.amplitude);
            let c = values::<f32>(x, &pupil.coverage);
            let w = values::<Df32>(x, &pupil.opd);
            let n = pupil.amplitude.desc.shape[0];
            let nn = n * n;
            let mut rows = vec![[Df32::ZERO; 4]; nn];
            let mut partial = vec![[Df32::ZERO; 4]; nn.div_ceil(256)];
            let coord = |i| {
                Df32::from_f32((2 * i + 1) as f32) / Df32::from_f32(n as f32) - Df32::from_f32(1.)
            };
            for k in 0..nn {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let u = coord(k % n);
                let v = coord(k / n);
                let gamma = (Df32::from_f32(1.)
                    - geometry.sin_max * geometry.sin_max * (u * u + v * v))
                    .sqrt();
                let wt = if c[k] > 0. {
                    Df32::from_f32(c[k]) * Df32::from_f32(a[k]) / gamma
                } else {
                    Df32::ZERO
                };
                let phase = if c[k] > 0. { w[k] } else { Df32::ZERO };
                rows[k] = [u, v, phase, wt.sqrt()];
                let z = spec::cis(phase);
                let vals = [wt, wt * phase, wt.mul_f32(z[0]), wt.mul_f32(z[1])];
                for j in 0..4 {
                    partial[k / 256][j] = partial[k / 256][j] + vals[j];
                }
            }
            let mut total = [Df32::ZERO; 4];
            for row in partial {
                for j in 0..4 {
                    total[j] = total[j] + row[j];
                }
            }
            x.words_mut(*samples)
                .copy_from_slice(bytemuck::cast_slice(&rows));
            x.words_mut(*sums)
                .copy_from_slice(bytemuck::cast_slice(&total));
        }
        Statistics::Residual {
            samples,
            sums,
            solution,
            output,
        } => {
            let a = values::<Df32>(x, samples).as_chunks::<4>().0;
            let s = values::<Df32>(x, sums);
            let q = values::<Df32>(x, solution);
            let mean = s[1] / s[0];
            let mut part = vec![[Df32::ZERO; 6]; a.len().div_ceil(256)];
            for row in &mut part {
                row[2] = Df32::from_f32(f32::INFINITY);
                row[3] = -row[2];
                row[4] = row[2];
                row[5] = row[3];
            }
            for (k, a) in a.iter().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if a[3].hi <= 0. {
                    continue;
                }
                let weight = a[3] * a[3];
                let r0 = a[2] - mean;
                let r1 = a[2] - q[0] - q[1] * a[0] - q[2] * a[1];
                let v = &mut part[k / 256];
                v[0] = v[0] + weight * r0 * r0;
                v[1] = v[1] + weight * r1 * r1;
                for (j, r) in [(2, r0), (4, r1)] {
                    if r.to_f64() < v[j].to_f64() {
                        v[j] = r;
                    }
                    if r.to_f64() > v[j + 1].to_f64() {
                        v[j + 1] = r;
                    }
                }
            }
            let mut v = part[0];
            for b in &part[1..] {
                v[0] = v[0] + b[0];
                v[1] = v[1] + b[1];
                for j in [2, 4] {
                    if b[j].to_f64() < v[j].to_f64() {
                        v[j] = b[j];
                    }
                    if b[j + 1].to_f64() > v[j + 1].to_f64() {
                        v[j + 1] = b[j + 1];
                    }
                }
            }
            let result = [
                (v[0] / s[0]).sqrt(),
                v[3] - v[2],
                (v[1] / s[0]).sqrt(),
                v[5] - v[4],
                (s[2] * s[2] + s[3] * s[3]) / (s[0] * s[0]),
            ];
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&result));
        }
    }
    Ok(())
}
