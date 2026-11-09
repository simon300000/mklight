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
/// The packed systems of an illumination request (resident sets copied).
fn ri_system(
    x: &Execution,
    systems: &SystemSet,
) -> MkResult<std::sync::Arc<crate::layout::PackedSystemSet>> {
    match systems {
        SystemSet::Uploaded(h) => x.system(*h),
        SystemSet::Resident {
            headers,
            surfaces,
            coefficients,
            count,
            max_surfaces,
            ..
        } => Ok(std::sync::Arc::new(crate::layout::PackedSystemSet {
            headers: bytemuck::cast_slice(
                &x.words(*headers)[..count * size_of::<crate::layout::PackedHeader>() / 4],
            )
            .to_vec(),
            surfaces: bytemuck::cast_slice(
                &x.words(*surfaces)
                    [..count * max_surfaces * size_of::<crate::layout::PackedSurface>() / 4],
            )
            .to_vec(),
            coefficients: bytemuck::cast_slice(x.words(*coefficients)).to_vec(),
            hashes: vec![],
        })),
    }
}
/// Header and surfaces of a field row's system, as the kernels check them.
fn ri_header<'a>(
    set: &'a crate::layout::PackedSystemSet,
    f: &super::ri::RiField,
    slots: usize,
) -> Option<(
    &'a crate::layout::PackedHeader,
    &'a [crate::layout::PackedSurface],
)> {
    let h = set.headers.get(f.aim.system as usize)?;
    if f.aim.wavelength >= h.num_wavelengths || h.num_surfaces as usize != slots {
        return None;
    }
    Some((
        h,
        set.surfaces
            .get(h.surface_offset as usize..)?
            .get(..slots)?,
    ))
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
        Request::Characteristic(r) => r.mask.desc.len()? * r.frequency.desc.shape[0],
        Request::SurfaceSamples(r) => r.samples.desc.shape[0],
        Request::Distribution(r) => r.radii.desc.len()? + 4,
        Request::FieldData(r) => r.hits.rays.count / if r.kind == 1 { 5 } else { 1 },
        Request::Records(r) => r.slots * r.hits.rays.count,
        Request::Coddington(r) => r.hits.rays.count,
        Request::CloseFocus(r) => r.hits.rays.count / 5,
        Request::IlluminationRays(r) => r.tasks.desc.shape[0],
        Request::Opd(r) => r.hits.rays.count,
        Request::FocusTerms(r) => r.mask.desc.len()?,
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
            let nr = r.mask.desc.len()?;
            let ns = nr / r.segment_len;
            let p = words(x, &r.points);
            let mask = words(x, &r.mask);
            let f = words(x, &r.frequency);
            let mut out = vec![Df32::ZERO; 2 * n];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n.max(1) as f64, "geometric OTF");
                }
                let ray = i % nr;
                let frequency = i / nr;
                if mask[ray] != 0 {
                    let j = 2 * (frequency * ns + ray / r.segment_len);
                    let q = spec::characteristic::<E>(
                        [df(p, ray), df(p, nr + ray)],
                        [df(f, j), df(f, j + 1)],
                        df(p, 4 * nr + ray),
                    );
                    out[i] = q[0];
                    out[n + i] = q[1];
                }
            }
            put(x, r.output, &out);
        }
        Request::FocusTerms(r) => {
            let p = words(x, &r.points);
            let mask = words(x, &r.mask);
            let qm = words(x, &r.position_moments);
            let tm = words(x, &r.slope_moments);
            let mut out = vec![Df32::ZERO; 3 * n];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                if mask[i] == 0 {
                    continue;
                }
                let row = (i / r.segment_len) * 10;
                let q: [D<E>; 2] = std::array::from_fn(|k| {
                    D::from_df(df(p, k * n + i))
                        - if r.centroid {
                            D::from_df(df(qm, row + 2 + k))
                        } else {
                            D::ZERO
                        }
                });
                let t: [D<E>; 2] = std::array::from_fn(|k| {
                    D::from_df(df(p, (k + 2) * n + i))
                        - if r.centroid {
                            D::from_df(df(tm, row + 2 + k))
                        } else {
                            D::ZERO
                        }
                });
                let w = D::from_df(df(p, 4 * n + i));
                out[i] = ((q[0] * q[0] + q[1] * q[1]) * w).0;
                out[n + i] = ((q[0] * t[0] + q[1] * t[1]) * w).0;
                out[2 * n + i] = ((t[0] * t[0] + t[1] * t[1]) * w).0;
            }
            put(x, r.output, &out);
        }
        Request::Opd(r) => {
            let n = r.hits.rays.count;
            let nc = r.chiefs.rays.count;
            let f = x.words(r.hits.rays.floats);
            let cf = x.words(r.chiefs.rays.floats);
            let meta = words(x, &r.reference);
            let index = words(x, &r.chief_index);
            let mut out = vec![Df32::from_f32(f32::NAN); n];
            let mut status = vec![1; n];
            let get = |w: &[u32], n: usize, i: usize, k: usize| {
                Df32::new(
                    f32::from_bits(w[k * n + i]),
                    f32::from_bits(w[(k + 9) * n + i]),
                )
            };
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let ci = index[i] as usize;
                if ci >= nc || x.words(r.hits.status)[i] != 0 || x.words(r.chiefs.status)[ci] != 0 {
                    continue;
                }
                if r.axial {
                    let p: [D<E>; 3] = std::array::from_fn(|k| D::from_df(get(f, n, i, k)));
                    let d: [D<E>; 3] = std::array::from_fn(|k| D::from_df(get(f, n, i, k + 3)));
                    let cp: [D<E>; 3] = std::array::from_fn(|k| D::from_df(get(cf, nc, ci, k)));
                    let cd: [D<E>; 3] = std::array::from_fn(|k| D::from_df(get(cf, nc, ci, k + 3)));
                    let dc = crate::trace_kernel::ddot(d, cd);
                    let qp = std::array::from_fn(|k| cp[k] - p[k]);
                    let distance = crate::trace_kernel::ddot(qp, cd) / dc;
                    let q = std::array::from_fn(|k| p[k] + d[k] * distance - cp[k]);
                    let t = std::array::from_fn(|k| d[k] / dc - cd[k]);
                    let t2 = crate::trace_kernel::ddot(t, t);
                    let s = if t2.hi() > 0.0 {
                        -crate::trace_kernel::ddot(q, t) / t2
                    } else {
                        D::new(f32::INFINITY, 0.0)
                    };
                    out[i] = if t2.hi() > 0.0 {
                        (cp[2] + s * cd[2]).0
                    } else {
                        Df32::from_f32(if cd[2].hi() == 0.0 {
                            f32::NAN
                        } else {
                            f32::INFINITY.copysign(cd[2].hi().v())
                        })
                    };
                    status[i] = 0;
                    continue;
                }
                let chief: [Df32; 7] = std::array::from_fn(|k| {
                    if k < 6 {
                        get(cf, nc, ci, k)
                    } else {
                        Df32::new(
                            f32::from_bits(cf[6 * nc + ci]),
                            f32::from_bits(cf[7 * nc + ci]),
                        )
                    }
                });
                let centre = &chief[..3];
                let zxp = df(meta, ci * 3);
                let tele = !zxp.hi.is_finite();
                let se = (D::<E>::from_df(zxp) - D::from_df(chief[2])) / D::from_df(chief[5]);
                let dummy = view(BufferId(0), ElementType::DoubleFloat, [0]);
                let reference = crate::aiming::PupilReference {
                    chief: dummy.clone(),
                    centre: dummy,
                    radius: if se.hi() < 0.0 { (-se).0 } else { se.0 },
                    image_index: df(meta, ci * 3 + 1),
                    wavelength_um: df(meta, ci * 3 + 2),
                    sphere_sheet: if se.hi() < 0.0 { -1 } else { 1 },
                    telecentric: tele,
                };
                let ray = crate::trace_kernel::KRayIn {
                    pos: std::array::from_fn(|k| get(f, n, i, k).hi),
                    pos_lo: std::array::from_fn(|k| get(f, n, i, k).lo),
                    dir: std::array::from_fn(|k| get(f, n, i, k + 3).hi),
                    dir_lo: std::array::from_fn(|k| get(f, n, i, k + 3).lo),
                    opl: Df32::new(f32::from_bits(f[6 * n + i]), f32::from_bits(f[7 * n + i])),
                    ..Default::default()
                };
                if let Some(v) =
                    crate::aiming::kernel::project::<E>(&ray, &chief, centre, &reference)
                {
                    out[i] = v[0];
                    status[i] = 0;
                }
            }
            put(x, r.output, &out);
            x.words_mut(r.status).copy_from_slice(&status);
        }
        Request::IlluminationRays(r) => {
            use rayon::prelude::*;
            let set = ri_system(x, &r.systems)?;
            let fields: &[super::ri::RiField] = bytemuck::cast_slice(words(x, &r.fields));
            let tasks: &[super::ri::RiTask] = bytemuck::cast_slice(words(x, &r.tasks));
            let out: Vec<super::ri::RiRay> = tasks
                .par_iter()
                .map(|task| {
                    let f = fields.get(task.field as usize);
                    match f.and_then(|f| ri_header(&set, f, r.slots)) {
                        Some((h, s)) => {
                            let steps: Vec<u32> = (1..h.num_surfaces).collect();
                            let sys = super::ri::System {
                                h,
                                s,
                                pool: &set.coefficients,
                                steps: &steps,
                            };
                            super::ri::run_ray::<E>(&sys, fields, task, &|| ctl.check().is_ok())
                        }
                        None => super::ri::RiRay {
                            status: crate::consts::MKL_AIM_INVALID,
                            ..Default::default()
                        },
                    }
                })
                .collect();
            ctl.check()?;
            x.words_mut(r.output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Request::CloseFocus(r) => {
            let f = bytemuck::cast_slice(x.words(r.hits.rays.floats));
            let st = x.words(r.hits.status);
            let mut out = vec![Df32::from_f32(f32::NAN); n * 3];
            let mut status = vec![1; n];
            for i in 0..n {
                ctl.check()?;
                if let Some(v) = super::curvature::close::<E>(f, st, r.hits.rays.count, i) {
                    out[i * 3..i * 3 + 3].copy_from_slice(&v);
                    status[i] = 0;
                }
            }
            put(x, r.output, &out);
            x.words_mut(r.status).copy_from_slice(&status);
        }
        Request::Coddington(r) => {
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
                    count,
                    max_surfaces,
                    ..
                } => (
                    bytemuck::cast_slice(
                        &x.words(*headers)[..count * size_of::<crate::layout::PackedHeader>() / 4],
                    ),
                    bytemuck::cast_slice(
                        &x.words(*surfaces)[..count
                            * max_surfaces
                            * size_of::<crate::layout::PackedSurface>()
                            / 4],
                    ),
                    bytemuck::cast_slice(x.words(*coefficients)),
                ),
            };
            let hi = bytemuck::cast_slice(x.words(r.hits.records.unwrap()));
            let lo = bytemuck::cast_slice(x.words(r.hits.record_low.unwrap()));
            let indices = x.words(r.hits.rays.indices);
            let mut out = vec![Df32::from_f32(f32::NAN); n * 3];
            let mut status = vec![1; n];
            for i in 0..n {
                ctl.check()?;
                if x.words(r.hits.status)[i] != 0 {
                    continue;
                }
                let Some(h) = headers.get(indices[i] as usize) else {
                    continue;
                };
                let wl = indices[n + i] as usize;
                if wl >= h.num_wavelengths as usize || h.num_surfaces as usize != r.slots {
                    continue;
                }
                if let Some(v) = super::curvature::focus::<E>(
                    h,
                    &surfaces[h.surface_offset as usize..][..r.slots],
                    pool,
                    hi,
                    lo,
                    n,
                    i,
                    wl,
                ) {
                    out[i * 3..i * 3 + 3].copy_from_slice(&v);
                    status[i] = 0;
                }
            }
            put(x, r.output, &out);
            x.words_mut(r.status).copy_from_slice(&status);
        }
        Request::Records(r) => {
            let nr = r.hits.rays.count;
            let hi = x.words(r.hits.records.unwrap());
            let lo = x.words(r.hits.record_low.unwrap());
            let frames = words(x, &r.frames);
            let mut out = vec![Df32::from_f32(f32::NAN); 3 * n];
            let mut radii = vec![Df32::ZERO; n];
            let mut mask = vec![0; n];
            let mut failures = vec![Df32::ZERO; nr];
            let mut launch = vec![Df32::ZERO; nr * 3];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n as f64, "global records");
                }
                if i < nr {
                    for k in 0..3 {
                        launch[3 * i + k] = Df32::new(
                            f32::from_bits(hi[RECORD_FLOATS * i + 3 + k]),
                            f32::from_bits(lo[RECORD_FLOATS * i + 3 + k]),
                        );
                    }
                    failures[i] = Df32::from_f32(if x.words(r.hits.status)[i] == 0 {
                        0.0
                    } else {
                        1.0
                    });
                }
                let a = i * RECORD_FLOATS;
                let ray = i % nr;
                let code = x.words(r.hits.status)[ray] & crate::consts::MKL_RAY_CODE_MASK;
                let terminal = r.terminal
                    && x.words(r.hits.status)[nr + ray] as usize == i / nr
                    && (code == crate::consts::MKL_RAY_VIGNETTED
                        || code == crate::consts::MKL_RAY_TIR);
                if hi[a + 12] != 0 && !terminal {
                    continue;
                }
                mask[i] = if terminal { 2 } else { 1 };
                let floats = x.words(r.hits.rays.floats);
                let p: [D<E>; 3] = std::array::from_fn(|k| {
                    if terminal {
                        D::new(
                            f32::from_bits(floats[k * nr + ray]),
                            f32::from_bits(floats[(k + 9) * nr + ray]),
                        )
                    } else {
                        D::new(f32::from_bits(hi[a + k]), f32::from_bits(lo[a + k]))
                    }
                });
                let f = (i / nr) * 12;
                radii[i] = (p[0] * p[0] + p[1] * p[1]).sqrt().0;
                for k in 0..3 {
                    let rot = std::array::from_fn(|j| D::<E>::from_df(df(frames, f + 3 * k + j)));
                    out[3 * i + k] = ((D::<E>::from_df(df(frames, f + 9 + k))
                        - D::from_df(r.origin[k]))
                        + crate::trace_kernel::ddot(rot, p))
                    .0;
                }
            }
            put(x, r.launch, &launch);
            put(x, r.output, &out);
            put(x, r.radii, &radii);
            put(x, r.failures, &failures);
            x.words_mut(r.mask).copy_from_slice(&mask);
        }
        Request::FieldData(r) => {
            let nr = r.hits.rays.count;
            let floats = x.words(r.hits.rays.floats);
            let status = x.words(r.hits.status);
            let get = |i: usize, k: usize| {
                D::<E>::new(
                    f32::from_bits(floats[k * nr + i]),
                    f32::from_bits(floats[(k + 9) * nr + i]),
                )
            };
            let mut out = vec![Df32::from_f32(f32::NAN); 5 * n];
            for i in 0..n {
                ctl.check()?;
                if r.kind == 2 {
                    let hi = x.words(r.hits.records.unwrap());
                    let lo = x.words(r.hits.record_low.unwrap());
                    let a = i * RECORD_FLOATS;
                    if status[i] == 0 {
                        out[5 * i] = get(i, 0).0;
                        out[5 * i + 1] = get(i, 1).0;
                    }
                    if hi[a + 12] != 0 {
                        continue;
                    }
                    let v = |k| D::<E>::new(f32::from_bits(hi[a + k]), f32::from_bits(lo[a + k]));
                    for k in 0..2 {
                        out[5 * i + 2 + k] = if r.pupil_step.hi == 0.0 {
                            (v(3 + k) / v(5)).0
                        } else {
                            v(k).0
                        };
                    }
                    out[5 * i + 4] = Df32::ZERO;
                } else if r.kind == 0 {
                    if status[i] != 0 {
                        continue;
                    }
                    let hi = x.words(r.hits.records.unwrap());
                    let lo = x.words(r.hits.record_low.unwrap());
                    let slot = r.record_slots - 1;
                    let a = (slot * nr + i) * RECORD_FLOATS;
                    let normal: [D<E>; 3] = std::array::from_fn(|k| {
                        D::new(f32::from_bits(hi[a + 8 + k]), f32::from_bits(lo[a + 8 + k]))
                    });
                    let d = [get(i, 3), get(i, 4), get(i, 5)];
                    let dot = crate::trace_kernel::ddot(d, normal);
                    let cross = [
                        d[1] * normal[2] - d[2] * normal[1],
                        d[2] * normal[0] - d[0] * normal[2],
                        d[0] * normal[1] - d[1] * normal[0],
                    ];
                    let sn = crate::trace_kernel::ddot(cross, cross).sqrt();
                    out[i * 5] = get(i, 0).0;
                    out[i * 5 + 1] = get(i, 1).0;
                    out[i * 5 + 2] =
                        Df32::from_f32(sn.hi().v().atan2(dot.hi().v().abs()).to_degrees());
                    out[i * 5 + 3] =
                        Df32::from_f32(d[0].hi().v().atan2(d[2].hi().v().abs()).to_degrees());
                    out[i * 5 + 4] =
                        Df32::from_f32(d[1].hi().v().atan2(d[2].hi().v().abs()).to_degrees());
                } else {
                    let a = i * 5;
                    if status[a..a + 5].iter().any(|&v| v != 0) {
                        continue;
                    }
                    out[i * 5] = get(a, 0).0;
                    out[i * 5 + 1] = get(a, 1).0;
                    for k in 0..2 {
                        let u = a + 1 + 2 * k;
                        let dx = get(u, 0) - get(u + 1, 0);
                        let dy = get(u, 1) - get(u + 1, 1);
                        out[i * 5 + 2 + k] = if r.pupil_step.hi.is_infinite() {
                            Df32::ZERO
                        } else {
                            ((dx * dx + dy * dy).sqrt() / D::from_df(r.pupil_step)).0
                        };
                    }
                    out[i * 5 + 4] = Df32::ZERO;
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
            let mut boundary = vec![Df32::ZERO; k];
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
                    let bound = |radius: D<E>, inclusive: bool| {
                        let (mut a, mut b) = (0, len);
                        while a < b {
                            let mid = (a + b) / 2;
                            let d = (D::<E>::from_df(df(keys, mid)) - radius).hi();
                            if if inclusive { d <= 0.0 } else { d < 0.0 } {
                                a = mid + 1;
                            } else {
                                b = mid;
                            }
                        }
                        if a > 0 {
                            D::<E>::from_df(df(cum, a - 1))
                        } else {
                            D::ZERO
                        }
                    };
                    if total.hi() > 0.0 {
                        let e = D::from_df(r.uncertainty);
                        boundary[i] =
                            ((bound(radius + e, true) - bound(radius - e, false)) / total).0;
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
            put(x, r.boundary, &boundary);
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
                    count,
                    max_surfaces,
                    ..
                } => (
                    bytemuck::cast_slice(
                        &x.words(*headers)[..count * size_of::<crate::layout::PackedHeader>() / 4],
                    ),
                    bytemuck::cast_slice(
                        &x.words(*surfaces)[..count
                            * max_surfaces
                            * size_of::<crate::layout::PackedSurface>()
                            / 4],
                    ),
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
                let v = super::sag::evaluate(surface, pool, xx, yy).map(|v| [v[0], v[1], v[2]]);
                match v {
                    Some(v) => {
                        for k in 0..3 {
                            out[3 * i + k] = v[k].0;
                        }
                    }
                    None => status[i] = 1,
                }
            }
            if let Some(projection) = &r.projection {
                let xy = words(x, &projection.xy);
                let frames = words(x, &projection.frames);
                let rows = projection.frames.desc.shape[0];
                for (i, s) in samples.iter().enumerate() {
                    if status[i] != 0 {
                        continue;
                    }
                    let h = &headers[s.system as usize];
                    let frame = (h.surface_offset + s.surface) as usize;
                    if frame >= rows {
                        status[i] = 2;
                        out[3 * i..3 * i + 3].fill(Df32::from_f32(f32::NAN));
                        continue;
                    }
                    let point = [
                        D::<E>::from_df(df(xy, 2 * i)),
                        D::from_df(df(xy, 2 * i + 1)),
                        D::from_df(out[3 * i]),
                    ];
                    for k in 0..3 {
                        let rot = std::array::from_fn(|j| {
                            D::<E>::from_df(df(frames, 12 * frame + 3 * k + j))
                        });
                        out[3 * i + k] = ((D::<E>::from_df(df(frames, 12 * frame + 9 + k))
                            - D::from_df(projection.origin[k]))
                            + crate::trace_kernel::ddot(rot, point))
                        .0;
                    }
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
