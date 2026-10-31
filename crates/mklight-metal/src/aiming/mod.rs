//! Encoders for P1. Only a four-byte failure count crosses to host during
//! generation; optical outputs and cached chief seeds remain device-resident.
use crate::{
    SharedBuffer,
    backend::Uploaded,
    context::{Encoder, PipelineKey},
    resident::{Execution, Lease},
};
use bytemuck::{Pod, Zeroable};
use mklight_core::MkResult;
use mklight_gpu::{aiming::*, consts::*, layout::TraceParams, pending::JobControl, resident::*};
use std::{sync::Arc, time::Instant};
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AimParams {
    count: u32,
    offset: u32,
    stride: u32,
    fields: u32,
    systems: u32,
    cap: u32,
    tolerance: f32,
    reserved: u32,
    pattern: u32,
    side: u32,
    extent: [f32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResidentTraceParams {
    trace: TraceParams,
    offset: u32,
    systems: u32,
    reserved: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProjectParams {
    radius: [f32; 2],
    index: [f32; 2],
    wavelength: [f32; 2],
    sheet: i32,
    telecentric: u32,
    count: u32,
    offset: u32,
    stride: u32,
    reserved: u32,
}
fn view(id: BufferId, element: ElementType) -> PlanView {
    PlanView {
        buffer: id,
        offset: 0,
        desc: BufferDesc::new(element, [0]),
        stride: 1,
    }
}
fn bind(e: &Encoder<'_>, index: usize, x: &Execution<'_>, v: &PlanView) {
    let (b, o) = x.buffer(v);
    e.set_buffer(index, b, o);
}
enum Systems {
    Uploaded(Arc<Uploaded>),
    Resident {
        headers: BufferId,
        surfaces: BufferId,
        coefficients: BufferId,
        count: usize,
    },
}
impl Systems {
    fn new(x: &Execution<'_>, s: &SystemSet) -> MkResult<Self> {
        Ok(match s {
            SystemSet::Uploaded(h) => Self::Uploaded(x.system(*h)?),
            SystemSet::Resident {
                headers,
                surfaces,
                coefficients,
                count,
                ..
            } => Self::Resident {
                headers: *headers,
                surfaces: *surfaces,
                coefficients: *coefficients,
                count: *count,
            },
        })
    }
    fn count(&self) -> u32 {
        match self {
            Self::Uploaded(u) => u.set.headers.len() as u32,
            Self::Resident { count, .. } => *count as u32,
        }
    }
    fn bind(&self, e: &Encoder<'_>, x: &Execution<'_>) {
        match self {
            Self::Uploaded(u) => {
                e.set_buffer(0, &u.headers, 0);
                e.set_buffer(1, &u.surfaces, 0);
                e.set_buffer(2, &u.coefficients, 0);
            }
            Self::Resident {
                headers,
                surfaces,
                coefficients,
                ..
            } => {
                bind(e, 0, x, &view(*headers, ElementType::U32));
                bind(e, 1, x, &view(*surfaces, ElementType::U32));
                bind(e, 2, x, &view(*coefficients, ElementType::DoubleFloat));
            }
        }
    }
}
fn take_rays(x: &mut Execution<'_>, r: &ResidentRays) -> [Lease; 4] {
    [
        x.take_output(r.floats),
        x.take_output(r.indices),
        x.take_output(r.logical_ids),
        x.take_output(r.report),
    ]
}
fn put_rays(x: &mut Execution<'_>, r: &ResidentRays, b: [Lease; 4]) {
    for (id, b) in [r.floats, r.indices, r.logical_ids, r.report]
        .into_iter()
        .zip(b)
    {
        x.put_output(id, b);
    }
}
enum ChiefSeeds {
    Cached(Arc<SharedBuffer<u32>>),
    Temporary(Lease),
}
impl ChiefSeeds {
    fn buffer(&self) -> &SharedBuffer<u32> {
        match self {
            Self::Cached(b) => b,
            Self::Temporary(b) => b,
        }
    }
    fn buffer_mut(&mut self) -> &mut SharedBuffer<u32> {
        match self {
            Self::Cached(b) => Arc::get_mut(b).expect("unpublished chief buffer"),
            Self::Temporary(b) => b,
        }
    }
}
fn account(
    x: &mut Execution<'_>,
    stage: &mut StageStats,
    cs: crate::CommandStats,
    start: Instant,
    bytes: usize,
) {
    stage.device_seconds += cs.gpu_seconds;
    stage.command_buffers += 1;
    stage.max_command_seconds = stage.max_command_seconds.max(start.elapsed().as_secs_f64());
    x.stats.parameter_bytes += bytes as u64;
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        ..Default::default()
    };
    match r {
        Request::GenerateAim(g) => {
            stage.name = "generate_aim".into();
            let systems = Systems::new(x, &g.systems)?;
            let nf = g.field_wavelength.desc.shape[0];
            let n = g.output.count;
            let key = g.field_key.as_ref().map(|k| {
                x.cache_key(format!(
                    "aim:{:?}:{k}:{}:{}",
                    g.systems,
                    g.tolerance.to_bits(),
                    g.max_iterations
                ))
            });
            let cached = if let Some(k) = &key {
                x.cached::<SharedBuffer<u32>>(k)?
            } else {
                None
            };
            let seeds = if let Some(seeds) = cached {
                ChiefSeeds::Cached(seeds)
            } else {
                let mut seeds = if key.is_some() {
                    x.stats.allocations += 1;
                    ChiefSeeds::Cached(Arc::new(
                        x.gpu
                            .buffer_filled(nf * MKL_AIM_CACHE_WORDS as usize, |_| {})?,
                    ))
                } else {
                    ChiefSeeds::Temporary(x.allocate(nf * MKL_AIM_CACHE_WORDS as usize)?)
                };
                let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_aim_chiefs"))?;
                for offset in (0..nf).step_by(16) {
                    ctl.check()?;
                    let count = (nf - offset).min(16);
                    let p = AimParams {
                        count: count as u32,
                        offset: offset as u32,
                        stride: n as u32,
                        fields: nf as u32,
                        systems: systems.count(),
                        cap: g.max_iterations,
                        tolerance: g.tolerance,
                        reserved: 0,
                        pattern: g.sampling.parameters().0,
                        side: g.sampling.parameters().1,
                        extent: [g.sampling.parameters().2.hi, g.sampling.parameters().2.lo],
                    };
                    let wall = Instant::now();
                    // SAFETY: field rows and destination extent were validated;
                    // the kernel checks field system/wavelength indices.
                    let cs = unsafe {
                        x.gpu.run("aim chiefs", |e| {
                            systems.bind(e, x);
                            bind(e, 3, x, &g.field_wavelength);
                            e.set_buffer_mut(4, seeds.buffer_mut(), 0);
                            e.set_bytes(5, &p);
                            e.dispatch_1d(&pso, count);
                            Ok(())
                        })?
                    };
                    account(x, &mut stage, cs, wall, size_of::<AimParams>());
                    ctl.report(0.1 * (offset + count) as f64 / nf as f64, "chief aiming");
                }
                if let (Some(k), ChiefSeeds::Cached(b)) = (key, &seeds) {
                    x.cache(k, nf * MKL_AIM_CACHE_WORDS as usize * 4, b.clone());
                }
                seeds
            };
            let mut out = take_rays(x, &g.output);
            let mut failures = x.allocate(1)?;
            failures.as_mut_slice()[0] = 0;
            x.stats.uploaded_bytes += 4;
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_generate_aim"))?;
            // Bounded Newton commands. Independent rays make chunking exact.
            for offset in (0..n).step_by(4096) {
                ctl.check()?;
                let count = (n - offset).min(4096);
                let p = AimParams {
                    count: count as u32,
                    offset: offset as u32,
                    stride: n as u32,
                    fields: nf as u32,
                    systems: systems.count(),
                    cap: g.max_iterations,
                    tolerance: g.tolerance,
                    reserved: 0,
                    pattern: g.sampling.parameters().0,
                    side: g.sampling.parameters().1,
                    extent: [g.sampling.parameters().2.hi, g.sampling.parameters().2.lo],
                };
                let wall = Instant::now();
                let [ff, uu, ids, reports] = &mut out;
                // SAFETY: all bindings are disjoint validated plan leases;
                // sample field indices are checked in the kernel.
                let cs = unsafe {
                    x.gpu.run("generate aim", |e| {
                        systems.bind(e, x);
                        bind(e, 3, x, &g.field_wavelength);
                        e.set_buffer(4, seeds.buffer(), 0);
                        e.set_bytes(5, &p);
                        bind(e, 6, x, &g.samples);
                        e.set_buffer_mut(7, ff, 0);
                        e.set_buffer_mut(8, uu, 0);
                        e.set_buffer_mut(9, ids, 0);
                        e.set_buffer_mut(10, reports, 0);
                        e.set_buffer_mut(11, &mut failures, 0);
                        e.dispatch_1d(&pso, count);
                        Ok(())
                    })?
                };
                account(x, &mut stage, cs, wall, size_of::<AimParams>());
                ctl.check()?;
                ctl.report(
                    0.1 + 0.9 * (offset + count) as f64 / n as f64,
                    "pupil aiming",
                );
            }
            if let ChiefSeeds::Temporary(b) = seeds {
                x.recycle(b);
            }
            x.stats.fallback_count += failures.as_slice()[0] as u64;
            x.stats.readback_bytes += 4;
            x.recycle(failures);
            put_rays(x, &g.output, out);
            if let Some(id) = g.unresolved {
                compact(x, g.output.report, id, n, ctl, &mut stage)?;
            }
        }
        Request::TraceResident(t) => {
            stage.name = "trace_resident".into();
            let systems = Systems::new(x, &t.systems)?;
            if t.radiometry != 0 {
                match &systems {
                    Systems::Uploaded(up)
                        if up
                            .set
                            .surfaces
                            .iter()
                            .all(|s| s.coating != MKL_COATING_TABLE) => {}
                    _ => return unsupported("resident table-coating radiometry"),
                }
            }
            let n = t.input.count;
            let mut out = take_rays(x, &t.output.rays);
            let mut status = x.take_output(t.output.status);
            let mut hi = if let Some(id) = t.output.records {
                x.take_output(id)
            } else {
                x.allocate(0)?
            };
            let mut lo = if let Some(id) = t.output.record_low {
                x.take_output(id)
            } else {
                x.allocate(0)?
            };
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_trace_resident"))?;
            for offset in (0..n).step_by(65536) {
                ctl.check()?;
                let count = (n - offset).min(65536);
                let p = ResidentTraceParams {
                    trace: TraceParams {
                        num_rays: count as u32,
                        stride: n as u32,
                        num_steps: t.steps.desc.len()? as u32,
                        start_surface: t.start_surface,
                        max_iterations: t.newton.max_iterations,
                        tolerance_abs: t.newton.tolerance_abs,
                        tolerance_rel: t.newton.tolerance_rel,
                        flags: if t.apply_apertures { MKL_TRACE_CLIP } else { 0 }
                            | if t.radiometry != 0 {
                                MKL_TRACE_RADIOMETRY
                            } else {
                                0
                            }
                            | if t.output.records.is_some() {
                                MKL_TRACE_RECORD
                            } else {
                                0
                            },
                        ..Default::default()
                    },
                    offset: offset as u32,
                    systems: systems.count(),
                    reserved: [0; 2],
                };
                let [ff, uu, ids, reports] = &mut out;
                let wall = Instant::now();
                // SAFETY: extents are plan-validated; dynamic system/path and
                // wavelength indices are checked before the shared trace core.
                let cs = unsafe {
                    x.gpu.run("resident trace", |e| {
                        systems.bind(e, x);
                        bind(e, 3, x, &t.steps);
                        e.set_bytes(4, &p);
                        bind(e, 5, x, &view(t.input.floats, ElementType::F32));
                        bind(e, 6, x, &view(t.input.indices, ElementType::U32));
                        bind(e, 7, x, &view(t.input.logical_ids, ElementType::U32));
                        bind(e, 8, x, &view(t.input.report, ElementType::U32));
                        e.set_buffer_mut(9, ff, 0);
                        e.set_buffer_mut(10, uu, 0);
                        e.set_buffer_mut(11, ids, 0);
                        e.set_buffer_mut(12, reports, 0);
                        e.set_buffer_mut(13, &mut status, 0);
                        e.set_buffer_mut(14, &mut hi, 0);
                        e.set_buffer_mut(15, &mut lo, 0);
                        e.dispatch_1d(&pso, count);
                        Ok(())
                    })?
                };
                account(x, &mut stage, cs, wall, size_of::<ResidentTraceParams>());
                ctl.check()?;
                ctl.report((offset + count) as f64 / n as f64, "resident trace");
            }
            put_rays(x, &t.output.rays, out);
            x.put_output(t.output.status, status);
            if let Some(id) = t.output.records {
                x.put_output(id, hi);
            } else {
                x.recycle(hi);
            }
            if let Some(id) = t.output.record_low {
                x.put_output(id, lo);
            } else {
                x.recycle(lo);
            }
        }
        Request::ProjectPupil(q) => {
            stage.name = "project_pupil".into();
            let n = q.hits.rays.count;
            let mut out = x.take_output(q.output);
            let mut status = x.take_output(q.status);
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_project_pupil"))?;
            for offset in (0..n).step_by(65536) {
                ctl.check()?;
                let count = (n - offset).min(65536);
                let r = &q.reference;
                let p = ProjectParams {
                    radius: [r.radius.hi, r.radius.lo],
                    index: [r.image_index.hi, r.image_index.lo],
                    wavelength: [r.wavelength_um.hi, r.wavelength_um.lo],
                    sheet: r.sphere_sheet,
                    telecentric: u32::from(r.telecentric),
                    count: count as u32,
                    offset: offset as u32,
                    stride: n as u32,
                    reserved: 0,
                };
                let wall = Instant::now();
                // SAFETY: plan validation fixed all plane and reference sizes.
                let cs = unsafe {
                    x.gpu.run("project pupil", |e| {
                        bind(e, 0, x, &view(q.hits.rays.floats, ElementType::F32));
                        bind(e, 1, x, &view(q.hits.status, ElementType::U32));
                        bind(e, 2, x, &r.chief);
                        bind(e, 3, x, &r.centre);
                        e.set_bytes(4, &p);
                        e.set_buffer_mut(5, &mut out, 0);
                        e.set_buffer_mut(6, &mut status, 0);
                        e.dispatch_1d(&pso, count);
                        Ok(())
                    })?
                };
                account(x, &mut stage, cs, wall, size_of::<ProjectParams>());
                ctl.check()?;
                ctl.report((offset + count) as f64 / n as f64, "pupil projection");
            }
            x.put_output(q.output, out);
            x.put_output(q.status, status);
        }
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stage.device_seconds;
    x.stats.command_buffers += stage.command_buffers;
    x.stats.stages.push(stage);
    ctl.check()
}

fn compact(
    x: &mut Execution<'_>,
    report: BufferId,
    output: BufferId,
    n: usize,
    ctl: &JobControl,
    stage: &mut StageStats,
) -> MkResult<()> {
    let groups = n.div_ceil(256);
    let mut counts = x.allocate(groups)?;
    let mut sparse = x.take_output(output);
    for pass in 0..3 {
        ctl.check()?;
        let kernel = ["mkl_aim_count", "mkl_aim_prefix", "mkl_aim_scatter"][pass];
        let pso = x.gpu.pipeline(&PipelineKey::plain(kernel))?;
        let wall = Instant::now();
        // SAFETY: plan validation fixes sparse size, and each group owns a
        // disjoint source range and prefix-selected destination range.
        let cs = unsafe {
            x.gpu.run(kernel, |e| {
                if pass == 1 {
                    e.set_buffer_mut(0, &mut counts, 0);
                    e.set_buffer_mut(1, &mut sparse, 0);
                    e.set_bytes(2, &(groups as u32));
                    e.dispatch_1d(&pso, 1);
                } else {
                    bind(e, 0, x, &view(report, ElementType::U32));
                    e.set_buffer_mut(1, &mut counts, 0);
                    e.set_buffer_mut(2, &mut sparse, 0);
                    e.set_bytes(3, &(n as u32));
                    e.dispatch_1d(&pso, groups.max(1));
                }
                Ok(())
            })?
        };
        account(x, stage, cs, wall, 4);
    }
    x.recycle(counts);
    x.put_output(output, sparse);
    Ok(())
}
