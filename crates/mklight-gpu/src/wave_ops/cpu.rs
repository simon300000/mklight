use super::{spec::*, *};
use bytemuck::Pod;
fn values<'a, T: Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
    let a = bytemuck::cast_slice(x.words(v.buffer));
    &a[v.offset..v.offset + v.desc.len().unwrap()]
}
pub fn execute(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    if let Request::Resample(r) = r {
        return resample::cpu(r, x, ctl);
    }
    if let Request::Diagnostics(r) = r {
        return diagnostics::cpu(r, x, ctl);
    }
    if let Request::Statistics(r) = r {
        return statistics::cpu(r, x, ctl);
    }
    if let Request::Lattice(r) = r {
        return lattice::cpu(r, x, ctl);
    }
    if let Request::Placement(r) = r {
        return placement::cpu(r, x, ctl);
    }
    if let Request::Spectral(r) = r {
        return spectral::cpu(r, x, ctl);
    }
    if let Request::Blur(r) = r {
        return sampling::cpu(r, x, ctl);
    }
    if let Request::ImageIntegral(r) = r {
        return image_integrals::cpu(r, x, ctl);
    }
    let p = params(r);
    let n = p.n as usize;
    let count = n * n;
    match r {
        Request::Zernike {
            samples,
            terms,
            standard,
            matrix,
            rhs,
        } => {
            let a = values::<Df32>(x, samples).as_chunks::<4>().0;
            let t = values::<u32>(x, terms).as_chunks::<2>().0;
            let mut design = vec![Df32::ZERO; a.len() * t.len()];
            let mut b = vec![Df32::ZERO; a.len()];
            for (i, row) in a.iter().enumerate() {
                if i % 256 == 0 {
                    ctl.check()?;
                }
                if row[3].hi <= 0. {
                    continue;
                }
                b[i] = row[2] * row[3];
                for (j, term) in t.iter().enumerate() {
                    design[i * t.len() + j] =
                        row[3] * zernike(row[0], row[1], term[0], term[1] as i32, *standard);
                }
            }
            x.words_mut(*matrix)
                .copy_from_slice(bytemuck::cast_slice(&design));
            x.words_mut(*rhs).copy_from_slice(bytemuck::cast_slice(&b));
        }

        Request::ImageIntegral(_)
        | Request::Blur(_)
        | Request::Spectral(_)
        | Request::Placement(_)
        | Request::Lattice(_)
        | Request::Statistics(_)
        | Request::Diagnostics(_)
        | Request::Resample(_) => unreachable!(),
        Request::FocusCorrelation {
            pupil,
            stencil,
            shifts,
            output,
            ..
        } => {
            let a = values::<f32>(x, &pupil.amplitude);
            let c = values::<f32>(x, &pupil.coverage);
            let w = values::<Df32>(x, &pupil.opd);
            let g = values::<f32>(x, stencil);
            let shifts = shifts
                .as_ref()
                .map(|v| values::<Df32>(x, v))
                .unwrap_or(&[Df32::ZERO]);
            let fs = p.field_side as usize;
            let mut out = vec![[0.; 2]; fs * shifts.len()];
            for (i, v) in out.iter_mut().enumerate() {
                let mut sums = [Df32::ZERO; 3];
                for k in 0..count {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    let q = ((i % fs) * count + k) * 5;
                    let t = correlation_term(&p, k, a, c, w, &g[q..q + 5], shifts[i / fs]);
                    for j in 0..3 {
                        sums[j] = sums[j].add_f32(t[j]);
                    }
                }
                *v = [scalar(sums[0] / sums[2]), scalar(sums[1] / sums[2])];
                ctl.report(
                    (i + 1) as f64 / (fs * shifts.len()) as f64,
                    "focus correlation",
                );
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
            let side = (x.words(*field).len() / 2).isqrt();
            let a = values::<f32>(x, &pupil.amplitude);
            let c = values::<f32>(x, &pupil.coverage);
            let w = values::<Df32>(x, &pupil.opd);
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
            let f = values::<[f32; 2]>(x, field);
            let s = values::<Df32>(x, sums);
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
            let a = values::<f32>(x, &pupil.amplitude);
            let c = values::<f32>(x, &pupil.coverage);
            let w = values::<Df32>(x, &pupil.opd);
            let s = values::<Df32>(x, sums);
            let mut out = vec![0.; (p.size * p.size) as usize];
            for (i, v) in out.iter_mut().enumerate() {
                let mut z = [Df32::ZERO; 2];
                for k in 0..count {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    let t = huygens_term(&p, i, k, a[k], c[k], w[k]);
                    for j in 0..2 {
                        z[j] = z[j].add_f32(t[j]);
                    }
                }
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
            ..
        } => {
            let w = values::<Df32>(x, &pupil.opd);
            let shifts = values::<Df32>(x, shifts);
            let mut out = Vec::with_capacity(count * shifts.len());
            for &dz in shifts {
                for (k, &v) in w.iter().enumerate() {
                    if k % 4096 == 0 {
                        ctl.check()?;
                    }
                    out.push(v + focus(&p, k, dz));
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::Detector {
            frequencies,
            output,
            ..
        } => {
            let fs = values::<f32>(x, frequencies).as_chunks::<2>().0;
            let out: Vec<_> = fs.iter().map(|v| detector(&p, *v)).collect();
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::LineSpread { image, output, .. } => {
            let a = values::<f32>(x, image);
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
