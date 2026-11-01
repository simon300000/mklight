//! Encoders for P1. Generation copies only a four-byte failure count to the
//! host; optical outputs, chief seeds and seed tables stay device-resident.
use crate::{
    SharedBuffer,
    backend::Uploaded,
    context::{CommandStats, Encoder, PipelineKey},
    resident::{Execution, Lease},
};
use bytemuck::{Pod, Zeroable};
use mklight_core::MkResult;
use mklight_gpu::{
    aiming::*, consts::*, layout::TraceParams, pending::JobControl,
    resident::immutable::OperationCacheKey, resident::*,
};
use std::{ops::Deref, sync::Arc, time::Instant};
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
    table: u32,
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
    /// The kernel variant: the overlay code only for systems that have sag
    /// overlays, or device-built systems whose surfaces the host cannot see.
    fn kernel(&self, plain: &'static str, overlays: &'static str) -> PipelineKey {
        let any = match self {
            Self::Uploaded(u) => u
                .set
                .surfaces
                .iter()
                .any(|s| s.flags & MKL_SURF_HAS_SAG_OVERLAY != 0),
            Self::Resident { .. } => true,
        };
        PipelineKey::plain(if any { overlays } else { plain })
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
/// Per-field-row data (chief seeds, seed tables): cached across plans when
/// the request names its immutable fields, otherwise plan-local.
enum Rows {
    Cached(Arc<SharedBuffer<u32>>),
    Local(Lease),
}
impl Deref for Rows {
    type Target = SharedBuffer<u32>;
    fn deref(&self) -> &SharedBuffer<u32> {
        match self {
            Self::Cached(b) => b,
            Self::Local(b) => b,
        }
    }
}
fn rows(
    x: &mut Execution<'_>,
    key: Option<OperationCacheKey>,
    words: usize,
    build: impl FnOnce(&mut Execution<'_>, &mut SharedBuffer<u32>) -> MkResult<()>,
) -> MkResult<Rows> {
    if let Some(k) = &key
        && let Some(b) = x.cached::<SharedBuffer<u32>>(k)?
    {
        return Ok(Rows::Cached(b));
    }
    Ok(match key {
        Some(k) => {
            x.stats.allocations += 1;
            let mut b = x.gpu.buffer_filled(words, |_| {})?;
            build(x, &mut b)?;
            let b = Arc::new(b);
            x.cache(k, words * 4, b.clone());
            Rows::Cached(b)
        }
        None => {
            let mut b = x.allocate(words)?;
            build(x, &mut b)?;
            Rows::Local(b)
        }
    })
}
/// Device seconds aimed for per Newton command: commands stay well inside
/// the cancellation budget while amortising submission latency.
const AIM_COMMAND_SECONDS: f64 = 0.015;
/// Rays per trace/projection command (bounded work per ray).
const TRACE_CHUNK: usize = 1 << 18;
const PROJECT_CHUNK: usize = 1 << 20;

/// Newton work per thread varies by orders of magnitude between systems and
/// pupil regions, so commands follow the measured device time: they start
/// small, then grow at most eightfold towards [`AIM_COMMAND_SECONDS`].
/// Threads are independent, so the split never changes a result.
fn newton_commands(
    x: &mut Execution<'_>,
    stage: &mut StageStats,
    ctl: &JobControl,
    n: usize,
    progress: (f64, f64, &str),
    mut encode: impl FnMut(&Encoder<'_>, &Execution<'_>, AimParams),
) -> MkResult<()> {
    const MIN: usize = 4096;
    const MAX: usize = 1 << 18;
    let (mut offset, mut next) = (0, MIN);
    while offset < n {
        ctl.check()?;
        let count = (n - offset).min(next);
        let p = AimParams {
            count: count as u32,
            offset: offset as u32,
            ..Zeroable::zeroed()
        };
        let wall = Instant::now();
        // SAFETY: callers bind plan-validated extents; kernels check every
        // device-provided field, system and wavelength index.
        let mut cs = unsafe {
            x.run(progress.2, |e| {
                encode(e, x, p);
                Ok(())
            })?
        };
        // Measure the first Newton tile only for large batches; small plans
        // keep chief/table/pupil work together. Subsequent tiles use that rate.
        if n > 65536 || (offset == 0 && n > MIN) { cs.gpu_seconds = x.flush()?.gpu_seconds; }
        account(x, stage, CommandStats::default(), wall, size_of::<AimParams>());
        if cs.gpu_seconds > 0. {
            let fit = (count as f64 * AIM_COMMAND_SECONDS / cs.gpu_seconds) as usize;
            next = fit.clamp(MIN, (8 * count).clamp(MIN, MAX));
        }
        offset += count;
        ctl.check()?;
        ctl.report(
            progress.0 + progress.1 * offset as f64 / n as f64,
            progress.2,
        );
    }
    Ok(())
}
fn account(
    x: &mut Execution<'_>,
    stage: &mut StageStats,
    cs: CommandStats,
    start: Instant,
    bytes: usize,
) {
    stage.device_seconds += cs.gpu_seconds;
    stage.command_buffers += cs.command_buffers;
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
            let (pattern, side, extent) = g.sampling.parameters();
            let params = |p: AimParams| AimParams {
                stride: n as u32,
                fields: nf as u32,
                systems: systems.count(),
                cap: g.max_iterations,
                tolerance: g.tolerance,
                pattern,
                side,
                extent: [extent.hi, extent.lo],
                ..p
            };
            let newton = kernel::newton_params();
            let key = |x: &Execution<'_>, what: &str| {
                g.field_key.as_ref().map(|k| {
                    x.cache_key(format!(
                        "{what}:{:?}:{k}:{}:{}",
                        g.systems,
                        g.tolerance.to_bits(),
                        g.max_iterations
                    ))
                })
            };
            let seed_key = key(x, "aim");
            let seeds = rows(
                x,
                seed_key,
                nf * MKL_AIM_CACHE_WORDS as usize,
                |x, seeds| {
                    let pso = x
                        .gpu
                        .pipeline(&systems.kernel("mkl_aim_chiefs", "mkl_aim_chiefs_overlays"))?;
                    newton_commands(
                        x,
                        &mut stage,
                        ctl,
                        nf,
                        (0.0, 0.05, "chief aiming"),
                        |e, x, p| {
                            systems.bind(e, x);
                            bind(e, 3, x, &g.field_wavelength);
                            e.set_buffer_mut(4, seeds, 0);
                            e.set_bytes(5, &params(p));
                            e.set_bytes(6, &newton);
                            e.dispatch_1d(&pso, p.count as usize);
                        },
                    )
                },
            )?;
            let table = if uses_table(g) {
                let table_key = key(x, "aim table");
                let words = nf * TABLE_NODES * MKL_AIM_NODE_WORDS as usize;
                Some(rows(x, table_key, words, |x, table| {
                    let pso = x
                        .gpu
                        .pipeline(&systems.kernel("mkl_aim_table", "mkl_aim_table_overlays"))?;
                    newton_commands(
                        x,
                        &mut stage,
                        ctl,
                        nf * TABLE_NODES,
                        (0.05, 0.05, "seed table"),
                        |e, x, p| {
                            systems.bind(e, x);
                            bind(e, 3, x, &g.field_wavelength);
                            e.set_buffer(4, &seeds, 0);
                            e.set_bytes(5, &params(p));
                            e.set_bytes(6, &newton);
                            e.set_buffer_mut(7, table, 0);
                            e.dispatch_1d(&pso, p.count as usize);
                        },
                    )
                })?)
            } else {
                None
            };
            let mut out = take_rays(x, &g.output);
            let mut failures = x.allocate(1)?;
            failures.as_mut_slice()[0] = 0;
            x.stats.uploaded_bytes += 4;
            let pso = x
                .gpu
                .pipeline(&systems.kernel("mkl_generate_aim", "mkl_generate_aim_overlays"))?;
            newton_commands(
                x,
                &mut stage,
                ctl,
                n,
                (0.1, 0.9, "pupil aiming"),
                |e, x, p| {
                    let [ff, uu, ids, reports] = &mut out;
                    systems.bind(e, x);
                    bind(e, 3, x, &g.field_wavelength);
                    e.set_buffer(4, &seeds, 0);
                    e.set_bytes(
                        5,
                        &AimParams {
                            table: u32::from(table.is_some()),
                            ..params(p)
                        },
                    );
                    bind(e, 6, x, &g.samples);
                    e.set_buffer_mut(7, ff, 0);
                    e.set_buffer_mut(8, uu, 0);
                    e.set_buffer_mut(9, ids, 0);
                    e.set_buffer_mut(10, reports, 0);
                    e.set_buffer_mut(11, &mut failures, 0);
                    e.set_bytes(12, &newton);
                    // Without a table the binding is never read.
                    e.set_buffer(13, table.as_deref().unwrap_or(&seeds), 0);
                    e.dispatch_1d(&pso, p.count as usize);
                },
            )?;
            for b in [Some(seeds), table].into_iter().flatten() {
                if let Rows::Local(b) = b {
                    x.recycle(b);
                }
            }
            x.count_fallbacks(failures);
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
            let pso = x.gpu.pipeline(&if t.output.records.is_some() {
                systems.kernel(
                    "mkl_trace_resident_records",
                    "mkl_trace_resident_overlays_records",
                )
            } else {
                systems.kernel("mkl_trace_resident", "mkl_trace_resident_overlays")
            })?;
            for offset in (0..n).step_by(TRACE_CHUNK) {
                ctl.check()?;
                let count = (n - offset).min(TRACE_CHUNK);
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
                    x.run("resident trace", |e| {
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
            for offset in (0..n).step_by(PROJECT_CHUNK) {
                ctl.check()?;
                let count = (n - offset).min(PROJECT_CHUNK);
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
                    x.run("project pupil", |e| {
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

/// Stable unresolved-index compaction: count, prefix and scatter as three
/// serial dispatches of one command.
fn compact(
    x: &mut Execution<'_>,
    report: BufferId,
    output: BufferId,
    n: usize,
    ctl: &JobControl,
    stage: &mut StageStats,
) -> MkResult<()> {
    ctl.check()?;
    let groups = n.div_ceil(256);
    let mut counts = x.allocate(groups)?;
    let mut sparse = x.take_output(output);
    let [count, prefix, scatter] = ["mkl_aim_count", "mkl_aim_prefix", "mkl_aim_scatter"]
        .map(|k| x.gpu.pipeline(&PipelineKey::plain(k)));
    let (count, prefix, scatter) = (count?, prefix?, scatter?);
    let wall = Instant::now();
    // SAFETY: plan validation fixes the sparse size; each group owns a
    // disjoint source range and prefix-selected destination range, and the
    // serial dispatches of one encoder complete in order.
    let cs = unsafe {
        x.run("aim compaction", |e| {
            bind(e, 0, x, &view(report, ElementType::U32));
            e.set_buffer_mut(1, &mut counts, 0);
            e.set_buffer_mut(2, &mut sparse, 0);
            e.set_bytes(3, &(n as u32));
            e.dispatch_1d(&count, groups.max(1));
            e.set_buffer_mut(0, &mut counts, 0);
            e.set_buffer_mut(1, &mut sparse, 0);
            e.set_bytes(2, &(groups as u32));
            e.dispatch_1d(&prefix, 1);
            bind(e, 0, x, &view(report, ElementType::U32));
            e.set_buffer_mut(1, &mut counts, 0);
            e.set_buffer_mut(2, &mut sparse, 0);
            e.set_bytes(3, &(n as u32));
            e.dispatch_1d(&scatter, groups.max(1));
            Ok(())
        })?
    };
    account(x, stage, cs, wall, 8);
    x.recycle(counts);
    x.put_output(output, sparse);
    Ok(())
}
