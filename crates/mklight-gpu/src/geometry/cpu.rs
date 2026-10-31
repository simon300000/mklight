use super::*;
use crate::{
    FpEnv,
    fenv::{Ftz, Ieee},
    fx::D,
    pending::JobControl,
    resident::cpu::Execution,
};
fn df(w: &[u32], i: usize) -> Df32 {
    Df32::new(f32::from_bits(w[2 * i]), f32::from_bits(w[2 * i + 1]))
}
fn words<'a>(x: &'a Execution, v: &PlanView) -> &'a [u32] {
    let s = v.desc.element.words();
    &x.words(v.buffer)[v.offset * s..(v.offset + v.desc.len().unwrap()) * s]
}
fn hit<E: FpEnv>(w: &[u32], n: usize, i: usize) -> [Df32; 4] {
    let f = |k| f32::from_bits(w[k * n + i]);
    let dz = D::<E>::new(f(5), f(14));
    [
        Df32::new(f(0), f(9)),
        Df32::new(f(1), f(10)),
        (D::new(f(3), f(12)) / dz).0,
        (D::new(f(4), f(13)) / dz).0,
    ]
}
pub fn execute(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    if x.flush_to_zero {
        run::<Ftz>(r, x, ctl)
    } else {
        run::<Ieee>(r, x, ctl)
    }
}
fn put(x: &mut Execution, id: BufferId, v: &[Df32]) {
    for (w, d) in x.words_mut(id).as_chunks_mut::<2>().0.iter_mut().zip(v) {
        *w = [d.hi.to_bits(), d.lo.to_bits()];
    }
}
fn run<E: FpEnv>(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    let n = match r {
        Request::Coordinates(r) => r.hits.rays.count,
        Request::Radial(r) => r.mask.desc.len()?,
        Request::Characteristic(r) => r.mask.desc.len()?,
        Request::SurfaceSamples(r) => r.samples.desc.shape[0],
        Request::Distribution(r) => r.radii.desc.len()? + 4,
    };
    match r {
        Request::Coordinates(r) => {
            let n = r.hits.rays.count;
            let nc = r.chiefs.rays.count;
            let mut out = vec![Df32::ZERO; 5 * n];
            let mut mask = vec![0; n];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n.max(1) as f64, "image coordinates");
                }
                let ci = words(x, &r.chief_index)[i] as usize;
                if x.words(r.hits.status)[i] != 0 {
                    continue;
                }
                mask[i] = 1;
                let c = if ci < nc && x.words(r.chiefs.status)[ci] == 0 {
                    hit::<E>(x.words(r.chiefs.rays.floats), nc, ci)
                } else {
                    [Df32::ZERO; 4]
                };
                let q = spec::coordinate::<E>(
                    hit::<E>(x.words(r.hits.rays.floats), n, i),
                    c,
                    r.defocus,
                );
                for k in 0..4 {
                    out[k * n + i] = q[k];
                }
                out[4 * n + i] =
                    Df32::from_f32(f32::from_bits(x.words(r.hits.rays.floats)[8 * n + i]));
            }
            put(x, r.output, &out);
            x.words_mut(r.mask).copy_from_slice(&mask);
        }
        Request::Radial(r) => {
            let p = words(x, &r.points);
            let m = words(x, &r.moments);
            let mask = words(x, &r.mask);
            let mut radius = vec![Df32::from_f32(f32::INFINITY); n];
            let mut square = radius.clone();
            let mut weight = vec![Df32::ZERO; n];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n.max(1) as f64, "geometric radii");
                }
                if mask[i] == 0 {
                    continue;
                }
                let c = if r.centroid {
                    [
                        df(m, (i / r.segment_len) * 10 + 2),
                        df(m, (i / r.segment_len) * 10 + 3),
                    ]
                } else {
                    [Df32::ZERO; 2]
                };
                let v = spec::radial::<E>([df(p, i), df(p, n + i)], c);
                radius[i] = v[0];
                square[i] = v[1];
                weight[i] = df(p, 4 * n + i);
            }
            put(x, r.radius, &radius);
            put(x, r.square, &square);
            put(x, r.weight, &weight);
            for (i, v) in x.words_mut(r.ids).iter_mut().enumerate() {
                *v = i as u32;
            }
        }
        Request::Characteristic(r) => {
            let p = words(x, &r.points);
            let mask = words(x, &r.mask);
            let f = words(x, &r.frequency);
            let mut out = vec![Df32::ZERO; 2 * n];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n.max(1) as f64, "geometric OTF");
                }
                if mask[i] != 0 {
                    let j = 2 * (i / r.segment_len);
                    let q = spec::characteristic::<E>(
                        [df(p, i), df(p, n + i)],
                        [df(f, j), df(f, j + 1)],
                        df(p, 4 * n + i),
                    );
                    out[i] = q[0];
                    out[n + i] = q[1];
                }
            }
            put(x, r.output, &out);
        }
        Request::Distribution(r) => {
            let keys = words(x, &r.keys);
            let cum = words(x, &r.cumulative);
            let radii = words(x, &r.radii);
            let len = r.keys.desc.len()?;
            let k = r.radii.desc.len()?;
            let total = if len > 0 {
                D::<E>::from_df(df(cum, len - 1))
            } else {
                D::ZERO
            };
            let mut out = vec![Df32::ZERO; k + 4];
            for (i, v) in out.iter_mut().enumerate() {
                ctl.check()?;
                if i == k + 3 {
                    *v = total.0;
                    continue;
                }
                if i < k {
                    let radius = D::<E>::from_df(df(radii, i));
                    let (mut a, mut b) = (0, len);
                    while a < b {
                        let mid = (a + b) / 2;
                        if (D::<E>::from_df(df(keys, mid)) - radius).hi() <= 0.0 {
                            a = mid + 1;
                        } else {
                            b = mid;
                        }
                    }
                    *v = if a > 0 && total.hi() > 0.0 {
                        (D::<E>::from_df(df(cum, a - 1)) / total).0
                    } else {
                        Df32::ZERO
                    };
                } else {
                    let target =
                        total * D::from_df(Df32::from_f64([0.5, 0.8, 0.9][i - k] * (1.0 - 1e-15)));
                    let (mut a, mut b) = (0, len);
                    while a < b {
                        let mid = (a + b) / 2;
                        if (D::<E>::from_df(df(cum, mid)) - target).hi() < 0.0 {
                            a = mid + 1;
                        } else {
                            b = mid;
                        }
                    }
                    *v = if a < len && total.hi() > 0.0 {
                        df(keys, a)
                    } else {
                        Df32::from_f32(f32::NAN)
                    };
                }
            }
            put(x, r.output, &out);
        }
        Request::SurfaceSamples(r) => {
            let owned;
            let (headers, surfaces, pool) = match &r.systems {
                SystemSet::Uploaded(h) => {
                    owned = x.system(*h)?;
                    (
                        &owned.headers[..],
                        &owned.surfaces[..],
                        &owned.coefficients[..],
                    )
                }
                SystemSet::Resident {
                    headers,
                    surfaces,
                    coefficients,
                    ..
                } => (
                    bytemuck::cast_slice(x.words(*headers)),
                    bytemuck::cast_slice(x.words(*surfaces)),
                    bytemuck::cast_slice(x.words(*coefficients)),
                ),
            };
            let samples: &[SurfaceSample] = bytemuck::cast_slice(words(x, &r.samples));
            let mut out = vec![Df32::from_f32(f32::NAN); 3 * n];
            let mut status = vec![0; n];
            for (i, s) in samples.iter().enumerate() {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n.max(1) as f64, "surface samples");
                }
                let Some(h) = headers.get(s.system as usize) else {
                    status[i] = 2;
                    continue;
                };
                if s.surface >= h.num_surfaces {
                    status[i] = 2;
                    continue;
                }
                let surface = &surfaces[(h.surface_offset + s.surface) as usize];
                let xx = D::<E>::from_df(s.xy[0]);
                let yy = D::<E>::from_df(s.xy[1]);
                let v = if surface.shape <= crate::consts::MKL_SHAPE_COORD_BREAK
                    && surface.flags & crate::consts::MKL_SURF_HAS_SAG_OVERLAY == 0
                {
                    // The shared intersection also supplies legacy sag and normal.
                    let params = crate::aiming::kernel::newton_params();
                    crate::trace_kernel::intersect(
                        surface,
                        pool,
                        [xx, yy, D::ZERO],
                        [D::ZERO, D::ZERO, D::new(1.0, 0.0)],
                        &params,
                    )
                    .ok()
                    .map(|h| [h.x[2], -h.g[0] / h.g[2], -h.g[1] / h.g[2]])
                } else {
                    crate::surface_math::evaluate(surface, pool, xx, yy).map(|v| [v.v, v.x, v.y])
                };
                match v {
                    Some(v) => {
                        for k in 0..3 {
                            out[3 * i + k] = v[k].0;
                        }
                    }
                    None => status[i] = 1,
                }
            }
            put(x, r.output, &out);
            x.words_mut(r.status).copy_from_slice(&status);
        }
    }
    ctl.check()?;
    ctl.report(1.0, "geometry");
    Ok(())
}
