use super::*;
use crate::resident::immutable::CacheCategory;
use crate::{
    FpEnv, Ftz, Ieee,
    layout::TraceParams,
    pending::JobControl,
    resident::cpu::Execution,
    trace_kernel::{self as tr, KRayIn, ResidentState},
};
use std::{sync::Arc, time::Instant};
fn words<'a>(x: &'a Execution, v: &PlanView) -> &'a [u32] {
    let start = v.offset * v.desc.element.words();
    &x.words(v.buffer)[start..start + v.desc.len().unwrap() * v.desc.element.words()]
}
pub(super) fn read_ray(f: &[f32], u: &[u32], n: usize, i: usize) -> KRayIn {
    KRayIn {
        pos: std::array::from_fn(|k| f[k * n + i]),
        dir: std::array::from_fn(|k| f[(3 + k) * n + i]),
        opl: Df32 {
            hi: f[6 * n + i],
            lo: f[7 * n + i],
        },
        weight: f[8 * n + i],
        pos_lo: std::array::from_fn(|k| f[(9 + k) * n + i]),
        dir_lo: std::array::from_fn(|k| f[(12 + k) * n + i]),
        wavelength: u[n + i],
    }
}
fn write_ray(
    x: &mut Execution,
    r: &ResidentRays,
    i: usize,
    ray: &KRayIn,
    system: u32,
    id: [u32; 2],
    report: [u32; 4],
) {
    let n = r.count;
    let f: &mut [f32] = bytemuck::cast_slice_mut(x.words_mut(r.floats));
    let planes = [
        ray.pos[0],
        ray.pos[1],
        ray.pos[2],
        ray.dir[0],
        ray.dir[1],
        ray.dir[2],
        ray.opl.hi,
        ray.opl.lo,
        ray.weight,
        ray.pos_lo[0],
        ray.pos_lo[1],
        ray.pos_lo[2],
        ray.dir_lo[0],
        ray.dir_lo[1],
        ray.dir_lo[2],
    ];
    for (k, v) in planes.into_iter().enumerate() {
        f[k * n + i] = v;
    }
    x.words_mut(r.indices)[i] = system;
    x.words_mut(r.indices)[n + i] = ray.wavelength;
    for (k, v) in id.into_iter().enumerate() {
        x.words_mut(r.logical_ids)[k * n + i] = v;
    }
    for (k, v) in report.into_iter().enumerate() {
        x.words_mut(r.report)[k * n + i] = v;
    }
}
pub(super) fn record_planes(s: ResidentState) -> ([f32; 13], [f32; 13]) {
    let r = s.ray;
    let hi = [
        r.pos[0],
        r.pos[1],
        r.pos[2],
        r.dir[0],
        r.dir[1],
        r.dir[2],
        r.opl.hi,
        r.weight,
        s.normal[0].hi,
        s.normal[1].hi,
        s.normal[2].hi,
        s.index.hi,
        f32::from_bits(s.status),
    ];
    let lo = [
        r.pos_lo[0],
        r.pos_lo[1],
        r.pos_lo[2],
        r.dir_lo[0],
        r.dir_lo[1],
        r.dir_lo[2],
        r.opl.lo,
        0.0,
        s.normal[0].lo,
        s.normal[1].lo,
        s.normal[2].lo,
        s.index.lo,
        0.0,
    ];
    (hi, lo)
}
pub fn execute(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    // The frozen execution seam exposes build identity but not the FP mode.
    // CPU backend build identities distinguish IEEE from flush-to-zero.
    if x.kernel_build().ends_with("flush-to-zero") {
        execute_in::<Ftz>(r, x, ctl)
    } else {
        execute_in::<Ieee>(r, x, ctl)
    }
}
fn execute_in<E: FpEnv>(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    match r {
        Request::GenerateAim(g) => {
            let set = x.system(g.systems)?;
            let fields: Vec<FieldSpec> =
                bytemuck::cast_slice(words(x, &g.field_wavelength)).to_vec();
            let samples: Vec<Sample> = bytemuck::cast_slice(words(x, &g.samples)).to_vec();
            // Content is checked at this upload/operation boundary, not inside Newton.
            for f in &fields {
                if f.system as usize >= set.headers.len()
                    || f.wavelength >= set.headers[f.system as usize].num_wavelengths
                {
                    return Err(bad("field system/wavelength"));
                }
            }
            if samples
                .iter()
                .any(|s| (s.field & !MKL_AIM_SKIP) as usize >= fields.len())
            {
                return Err(bad("sample field index"));
            }
            // CPU storage is complete at every operation, so in-plan rows have
            // content too.
            let identity = aim_identity(
                g,
                set.digest(),
                Some(words(x, &g.field_wavelength)),
                g.seeds.as_ref().map(|v| words(x, v)),
            );
            let key = |category, what: &str| {
                identity
                    .as_ref()
                    .map(|id| x.content_key(category, format!("{what}:{id}")))
            };
            let seed_key = key(CacheCategory::ChiefSources, "aim chiefs");
            let table_key = key(CacheCategory::SeedTables, "aim table");
            let check = || ctl.check().is_ok();
            let make = |f| {
                let f: &FieldSpec = f;
                let h = &set.headers[f.system as usize];
                kernel::FieldKernel::<E> {
                    h,
                    s: &set.surfaces[h.surface_offset as usize..][..h.num_surfaces as usize],
                    pool: &set.coefficients,
                    field: f,
                    tolerance: g.tolerance,
                    cap: g.max_iterations,
                    schedule_cap: g.options.schedule_cap,
                    check: &check,
                    env: Default::default(),
                }
            };
            // Host rows take precedence: nothing is solved or cached for them.
            let cached = match (&g.seeds, &seed_key) {
                (None, Some(k)) => x.cached::<Vec<ChiefSeed>>(k)?,
                _ => None,
            };
            let seeds = match (cached, &g.seeds) {
                (Some(s), _) => s,
                (None, Some(v)) => Arc::new(bytemuck::cast_slice(words(x, v)).to_vec()),
                (None, None) => {
                    let start = Instant::now();
                    let mut seeds = vec![];
                    for (i, f) in fields.iter().enumerate() {
                        ctl.check()?;
                        seeds.push(make(f).prepare());
                        ctl.report(0.05 * (i + 1) as f64 / fields.len() as f64, "chief aiming");
                    }
                    let seeds = Arc::new(seeds);
                    if let Some(k) = seed_key {
                        let cost = start.elapsed().as_secs_f64();
                        x.cache(k, seeds.len() * size_of::<ChiefSeed>(), cost, seeds.clone());
                    }
                    seeds
                }
            };
            let side = g.options.table_side as usize;
            let nodes = table_nodes(g.options.table_side);
            // A host table takes precedence: nothing is solved or cached.
            let table = if let Some(v) = &g.options.table {
                Some(Arc::new(bytemuck::cast_slice(words(x, v)).to_vec()))
            } else if uses_table(g) {
                let cached = match &table_key {
                    Some(k) => x.cached::<Vec<SeedNode>>(k)?,
                    None => None,
                };
                Some(match cached {
                    Some(t) => t,
                    None => {
                        let start = Instant::now();
                        let mut table = Vec::with_capacity(fields.len() * nodes);
                        for (row, f) in fields.iter().enumerate() {
                            for k in 0..nodes {
                                ctl.check()?;
                                table.push(make(f).node(&seeds[row], k, side));
                            }
                            ctl.report(
                                0.05 + 0.05 * (row + 1) as f64 / fields.len() as f64,
                                "seed table",
                            );
                        }
                        let table = Arc::new(table);
                        if let Some(k) = table_key {
                            let cost = start.elapsed().as_secs_f64();
                            x.cache(k, table.len() * size_of::<SeedNode>(), cost, table.clone());
                        }
                        table
                    }
                })
            } else {
                None
            };
            for (i, &sample) in samples.iter().enumerate().take(g.output.count) {
                ctl.check()?;
                let row = (sample.field & !MKL_AIM_SKIP) as usize;
                let f = &fields[row];
                let nodes = table.as_ref().map(|t| (&t[row * nodes..][..nodes], side));
                let (ray, report) =
                    make(f).generate(&seeds[row], nodes, &sample, g.options.cull_outside);
                if report[0] != 0 && report[0] != MKL_AIM_OUTSIDE {
                    x.stats.fallback_count += 1;
                }
                write_ray(x, &g.output, i, &ray, f.system, sample.id, report);
                if i % 32 == 0 {
                    ctl.report(
                        0.1 + 0.9 * (i + 1) as f64 / g.output.count as f64,
                        "pupil aiming",
                    );
                }
            }
            let n = g.output.count;
            x.stats.aiming.add(&resident::AimStats::from_report(
                x.words(g.output.report),
                n,
            ));
        }
        Request::TraceResident(t) => {
            let set = x.system(t.systems)?;
            if t.radiometry != 0 && set.surfaces.iter().any(|s| s.coating == MKL_COATING_TABLE) {
                return resident::unsupported("resident table-coating radiometry");
            }
            let n = t.input.count;
            let steps = words(x, &t.steps).to_vec();
            let params = TraceParams {
                num_steps: steps.len() as u32,
                start_surface: t.start_surface,
                max_iterations: t.newton.max_iterations,
                tolerance_abs: t.newton.tolerance_abs,
                tolerance_rel: t.newton.tolerance_rel,
                flags: if t.apply_apertures { MKL_TRACE_CLIP } else { 0 }
                    | if t.radiometry != 0 {
                        MKL_TRACE_RADIOMETRY
                    } else {
                        0
                    },
                ..Default::default()
            };
            let f: Vec<f32> = bytemuck::cast_slice(x.words(t.input.floats)).to_vec();
            let u = x.words(t.input.indices).to_vec();
            let ids = x.words(t.input.logical_ids).to_vec();
            let reports = x.words(t.input.report).to_vec();
            // Resident values are validated before they can index a system.
            for i in 0..n {
                if u[i] as usize >= set.headers.len()
                    || u[n + i] >= set.headers[u[i] as usize].num_wavelengths
                {
                    return Err(bad("resident ray system/wavelength"));
                }
            }
            for h in &set.headers {
                let ss = &set.surfaces[h.surface_offset as usize..][..h.num_surfaces as usize];
                let program: Vec<_> = steps
                    .iter()
                    .map(|&s| crate::PathStep::decode(s).ok_or_else(|| bad("invalid path word")))
                    .collect::<MkResult<_>>()?;
                crate::PathStep::validate_program(&program, t.start_surface, ss)?;
            }
            for i in 0..n {
                ctl.check()?;
                let h = &set.headers[u[i] as usize];
                let ray = read_ray(&f, &u, n, i);
                let report = std::array::from_fn(|k| reports[k * n + i]);
                let record = |slot, s| {
                    if let (Some(hi), Some(lo)) = (t.output.records, t.output.record_low) {
                        let (hv, lv) = record_planes(s);
                        let offset = (slot * n + i) * 13;
                        x.words_mut(hi)[offset..offset + 13]
                            .copy_from_slice(bytemuck::cast_slice(&hv));
                        x.words_mut(lo)[offset..offset + 13]
                            .copy_from_slice(bytemuck::cast_slice(&lv));
                    }
                };
                let out = if report[0] == 0 {
                    tr::trace_resident::<E>(
                        h,
                        &set.surfaces[h.surface_offset as usize..][..h.num_surfaces as usize],
                        &set.coefficients,
                        &steps,
                        &params,
                        &ray,
                        record,
                    )
                } else {
                    let out = ResidentState {
                        ray,
                        status: report[0],
                        last_surface: report[0] >> MKL_RAY_SURFACE_SHIFT,
                        ..Default::default()
                    };
                    let mut record = record;
                    for k in 0..=steps.len() {
                        record(k, out);
                    }
                    out
                };
                write_ray(
                    x,
                    &t.output.rays,
                    i,
                    &out.ray,
                    u[i],
                    [ids[i], ids[n + i]],
                    report,
                );
                for (k, value) in [out.status, out.last_surface, out.steps]
                    .into_iter()
                    .enumerate()
                {
                    x.words_mut(t.output.status)[k * n + i] = value;
                }
                if i % 32 == 0 {
                    ctl.report((i + 1) as f64 / n as f64, "resident trace");
                }
            }
        }
        Request::ProjectPupil(p) => {
            let n = p.hits.rays.count;
            let chief: Vec<Df32> = bytemuck::cast_slice(words(x, &p.reference.chief)).to_vec();
            let centre: Vec<Df32> = bytemuck::cast_slice(words(x, &p.reference.centre)).to_vec();
            for i in 0..n {
                ctl.check()?;
                let ray = read_ray(
                    bytemuck::cast_slice(x.words(p.hits.rays.floats)),
                    x.words(p.hits.rays.indices),
                    n,
                    i,
                );
                let status = x.words(p.hits.status)[i];
                let value = if status == 0 {
                    kernel::project::<E>(&ray, &chief, &centre, &p.reference)
                } else {
                    None
                };
                x.words_mut(p.status)[i] = if status != 0 {
                    status
                } else if value.is_none() {
                    MKL_PUPIL_MISS
                } else {
                    0
                };
                let values = value.unwrap_or(
                    [Df32 {
                        hi: f32::NAN,
                        lo: 0.0,
                    }; 4],
                );
                for (k, v) in values.into_iter().enumerate() {
                    x.words_mut(p.output)[(k * n + i) * 2..(k * n + i) * 2 + 2]
                        .copy_from_slice(bytemuck::cast_slice(&[v]));
                }
                if i % 128 == 0 {
                    ctl.report((i + 1) as f64 / n as f64, "pupil projection");
                }
            }
        }
    }
    ctl.check()
}
