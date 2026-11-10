use super::{spec::*, *};
pub fn execute(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    // Parameters, grid side `n` and `n²` of the requests evaluated here
    // (the grouped requests have their own).
    let grid_params = || {
        let p = params(r);
        let n = p.n as usize;
        (p, n, n * n)
    };
    match r {
        Request::Resample(r) => return resample::cpu(r, x, ctl),
        Request::Diagnostics(r) => return diagnostics::cpu(r, x, ctl),
        Request::Lattice(r) => return lattice::cpu(r, x, ctl),
        Request::Placement(r) => return placement::cpu(r, x, ctl),
        Request::Spectral(r) => return spectral::cpu(r, x, ctl),
        Request::ImageIntegral(r) => return image_integrals::cpu(r, x, ctl),
        Request::FftMtf(r) => return fft_mtf::cpu(r, x, ctl),
        Request::Simulation(r) => return simulation::cpu(r, x, ctl),
        Request::Shells(r) => return shells::cpu(r, x, ctl),
        Request::FocusCorrelation {
            pupil,
            stencil,
            shifts,
            output,
            phase_steps,
            ..
        } => {
            let (p, n, count) = grid_params();
            let a = x.rows::<f32>(&pupil.amplitude);
            let c = x.rows::<f32>(&pupil.coverage);
            let w = x.rows::<Df32>(&pupil.opd);
            let g = x.rows::<f32>(stencil);
            let shifts = shifts
                .as_ref()
                .map(|v| x.rows::<Df32>(v))
                .unwrap_or(&[Df32::ZERO]);
            let fs = p.field_side as usize;
            let mut out = vec![[0.; 2]; fs * shifts.len()];
            let mut steps = vec![0f32; shifts.len()];
            for (plane, &dz) in shifts.iter().enumerate() {
                let mut focused = vec![Df32::ZERO; count];
                for k in 0..count {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    focused[k] = if c[k] > 0. {
                        w[k] + focus(&p, k, dz)
                    } else {
                        w[k]
                    };
                }
                steps[plane] = phase_step(n, c, &focused);
                for frequency in 0..fs {
                    ctl.check()?;
                    let sums = blocked_sum(count, |k| {
                        let q = (frequency * count + k) * 5;
                        correlation_term(&p, k, a, c, &focused, &g[q..q + 5], Df32::ZERO)
                    });
                    out[plane * fs + frequency] =
                        [scalar(sums[0] / sums[2]), scalar(sums[1] / sums[2])];
                    ctl.report(
                        (plane * fs + frequency + 1) as f64 / (fs * shifts.len()) as f64,
                        "focus correlation",
                    );
                }
            }
            if let Some(id) = phase_steps {
                x.words_mut(*id)
                    .copy_from_slice(bytemuck::cast_slice(&steps));
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }

        Request::Prepare {
            pupil,
            field,
            moments,
            ..
        } => {
            let (p, n, count) = grid_params();
            let side = (x.words(*field).len() / 2).isqrt();
            let a = x.rows::<f32>(&pupil.amplitude);
            let c = x.rows::<f32>(&pupil.coverage);
            let w = x.rows::<Df32>(&pupil.opd);
            let mut f = vec![[0.; 2]; side * side];
            let mut m = vec![0.; 3 * count];
            for k in 0..count {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let (z, v) = prepare(&p, k, a[k], c[k], w[k]);
                f[k / n * side + k % n] = z;
                for j in 0..3 {
                    m[j * count + k] = v[j];
                }
            }
            x.words_mut(*field)
                .copy_from_slice(bytemuck::cast_slice(&f));
            x.words_mut(*moments)
                .copy_from_slice(bytemuck::cast_slice(&m));
        }
        Request::Intensity {
            field,
            sums,
            output,
            ..
        } => {
            let (p, ..) = grid_params();
            let f = x.rows::<[f32; 2]>(field);
            let s = x.rows::<Df32>(sums);
            let m = p.size as usize;
            let side = p.field_side as usize;
            let out: Vec<_> = (0..m * m)
                .map(|k| {
                    let j = if p.shifted == 0 {
                        k
                    } else {
                        let ix = ((k % m) as i64 - (m / 2) as i64).rem_euclid(side as i64) as usize;
                        let iy = ((k / m) as i64 - (m / 2) as i64).rem_euclid(side as i64) as usize;
                        iy * side + ix
                    };
                    intensity(&p, f[j], s)
                })
                .collect();
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::Huygens {
            pupil,
            sums,
            output,
            ..
        } => {
            let (p, _, count) = grid_params();
            let a = x.rows::<f32>(&pupil.amplitude);
            let c = x.rows::<f32>(&pupil.coverage);
            let w = x.rows::<Df32>(&pupil.opd);
            let s = x.rows::<Df32>(sums);
            let mut out = vec![0.; (p.size * p.size) as usize];
            for (i, v) in out.iter_mut().enumerate() {
                ctl.check()?;
                let z = blocked_sum(count, |k| huygens_term(&p, i, k, a[k], c[k], w[k]));
                *v = huygens_intensity(&p, z, s);
                ctl.report((i + 1) as f64 / (p.size * p.size) as f64, "Huygens");
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::Focus {
            pupil,
            shifts,
            output,
            phase_steps,
            ..
        } => {
            let (p, n, count) = grid_params();
            let w = x.rows::<Df32>(&pupil.opd);
            let coverage = x.rows::<f32>(&pupil.coverage);
            let shifts = x.rows::<Df32>(shifts);
            let mut out = Vec::with_capacity(count * shifts.len());
            for &dz in shifts {
                for (k, &v) in w.iter().enumerate() {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    out.push(if coverage[k] > 0. {
                        v + focus(&p, k, dz)
                    } else {
                        v
                    });
                }
            }
            if let Some(id) = phase_steps {
                let steps: Vec<f32> = out
                    .chunks(count)
                    .map(|a| phase_step(n, coverage, a))
                    .collect();
                x.words_mut(*id)
                    .copy_from_slice(bytemuck::cast_slice(&steps));
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::LineSpread { image, output, .. } => {
            let (p, n, _) = grid_params();
            let a = x.rows::<f32>(image);
            let mut out = vec![0.; 2 * n];
            for k in 0..2 * n {
                ctl.check()?;
                let mut s = Df32::ZERO;
                for j in 0..n {
                    s = s.add_f32(a[if k < n { j * n + k } else { (k - n) * n + j }]);
                }
                out[k] = scalar(s * df(p.spacing));
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
    }
    ctl.check()?;
    ctl.report(1., "wave");
    Ok(())
}

fn phase_step(n: usize, coverage: &[f32], opd: &[Df32]) -> f32 {
    let mut step = 0f32;
    for k in 0..n * n {
        if coverage[k] <= 0. {
            continue;
        }
        for j in [
            if k % n + 1 < n { k + 1 } else { k },
            if k / n + 1 < n { k + n } else { k },
        ] {
            if coverage[j] > 0. {
                step = step.max(scalar(opd[j] - opd[k]).abs());
            }
        }
    }
    step
}
