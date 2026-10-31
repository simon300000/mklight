//! `MetalBackend`: `mklight_gpu::ComputeBackend` on Metal.
//!
//! Jobs run on a worker thread per submission. Every job is split into
//! command buffers of at most `chunk_items` work items (values, for
//! reductions); the submission's cancel token and the caller's context are
//! checked and progress is reported between them, and progress reaches 1 on
//! success. Each command buffer is awaited and its status checked, so GPU
//! errors surface as `MkError::Backend` and GPU time is summed into
//! `JobStats`. A worker panic becomes an error naming the job.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mklight_core::job::Precision;
use mklight_core::rng::Philox;
use mklight_core::{JobContext, MkError, MkResult};
use mklight_gpu::backend::next_backend_instance;
use mklight_gpu::consts::*;
use mklight_gpu::layout::{PhiloxParams, ReduceParams, SagParams};
use mklight_gpu::pending::{JobControl, pending, spawn_job};
use mklight_gpu::{
    BackendKind, ComputeBackend, DeviceInfo, Df32, ErrorBudget, Job, JobKind, JobOutput, JobResult,
    JobStats, NumericsProbe, PackedHeader, PackedSurface, PackedSystem, PackedSystemSet, Pending,
    PrecisionDescriptor, SystemHandle, reduce,
};
use mklight_gpu::{PreparedTrace, TraceRequest, TraceResult};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLGPUFamily};

use crate::buffer::SharedBuffer;
use crate::context::{
    CommandStats, LibraryChoice, LibraryOrigin, MetalContext, MetalOptions, PipelineKey,
};

/// Default work items per command buffer.
const DEFAULT_CHUNK: usize = 1 << 20;

pub(crate) struct Uploaded {
    pub(crate) headers: SharedBuffer<PackedHeader>,
    pub(crate) surfaces: SharedBuffer<PackedSurface>,
    pub(crate) coefficients: SharedBuffer<Df32>,
    pub(crate) set: PackedSystemSet,
    pub(crate) sag_overlays: bool,
}

struct Inner {
    ctx: MetalContext,
    info: DeviceInfo,
    precision: PrecisionDescriptor,
    instance: u64,
    next_id: AtomicU64,
    chunk_items: AtomicUsize,
    systems: Mutex<HashMap<u64, Arc<Uploaded>>>,
    resident: crate::resident::Runtime,
    cache: Mutex<mklight_gpu::resident::cache::SystemCache<Uploaded>>,
}

#[derive(Clone)]
pub struct MetalBackend {
    inner: Arc<Inner>,
}

/// Deterministic inputs for the numerics probe: random magnitudes and
/// signs, triples with `c = −fl(a·b)` (where fused and unfused results
/// differ), and operands whose products land in the subnormal range.
pub fn probe_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Philox::new(0x0005_EED0_FA11, 0);
    let mut rnd = |emin: i32, emax: i32| {
        let m = 1.0 + r.uniform();
        let s = if r.uniform() < 0.5 { -1.0 } else { 1.0 };
        let e = emin + ((emax - emin) as f64 * r.uniform()) as i32;
        (s * m * 2f64.powi(e)) as f32
    };
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..4096 {
        let x = rnd(-20, 20);
        let y = rnd(-20, 20);
        let z = match i % 3 {
            0 => -(x * y),
            1 => rnd(-20, 20),
            _ => x * y * rnd(-3, 0),
        };
        // Every 8th triple: |a| in [2^-9, 2^13) so (a·2^-70)·2^-70 is subnormal.
        let x = if i % 8 == 7 { rnd(-9, 12) } else { x };
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

/// The IEEE binary32 results the probe kernel must reproduce.
fn probe_expected(x: f32, y: f32, z: f32) -> [f32; MKL_PROBE_WORDS as usize] {
    let p70 = 2f32.powi(-70);
    [
        x * y + z,
        x.mul_add(y, z),
        x / y,
        x.abs().sqrt(),
        x + y,
        x * y,
        (x * p70) * p70,
        (x + y) - y,
    ]
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn run_probe(ctx: &MetalContext) -> MkResult<NumericsProbe> {
    let (a, b, c) = probe_inputs();
    let n = a.len();
    let words = MKL_PROBE_WORDS as usize;
    let (ba, bb, bc) = (ctx.buffer(&a)?, ctx.buffer(&b)?, ctx.buffer(&c)?);
    let mut out = ctx.buffer_zeroed::<f32>(n * words)?;
    let pso = ctx.pipeline(&PipelineKey::plain("mkl_numerics_probe"))?;
    // SAFETY: thread i < n reads a/b/c[i] (n elements each) and writes
    // out[i*W .. i*W + W] (n*W elements); `out` is bound mutably.
    unsafe {
        ctx.run("mklight numerics probe", |e| {
            e.set_buffer(0, &ba, 0);
            e.set_buffer(1, &bb, 0);
            e.set_buffer(2, &bc, 0);
            e.set_buffer_mut(3, &mut out, 0);
            e.set_bytes(4, &(n as u32));
            e.dispatch_1d(&pso, n);
            Ok(())
        })?;
    }
    let got = out.as_slice();
    let mut ok = [true; MKL_PROBE_WORDS as usize];
    for i in 0..n {
        let want = probe_expected(a[i], b[i], c[i]);
        for w in 0..words {
            ok[w] &= same(got[i * words + w], want[w]);
        }
    }
    Ok(NumericsProbe {
        no_contraction: ok[0],
        true_fma: ok[1],
        correctly_rounded_div: ok[2],
        correctly_rounded_sqrt: ok[3],
        ieee_add_mul: ok[4] && ok[5],
        subnormals: ok[6],
        no_reassociation: ok[7],
    })
}

impl MetalBackend {
    /// The default device, or `Ok(None)` if this machine has no Metal device.
    pub fn new() -> MkResult<Option<Self>> {
        Self::with_options(&MetalOptions::default())
    }

    /// With `LibraryChoice::Auto`, a precompiled library that fails the
    /// numerics probe (e.g. built with weaker fallback flags) is replaced by
    /// the runtime-compiled source when that one passes.
    pub fn with_options(opts: &MetalOptions) -> MkResult<Option<Self>> {
        let Some(mut ctx) = MetalContext::new(opts)? else {
            return Ok(None);
        };
        let mut probe = run_probe(&ctx)?;
        if opts.library == LibraryChoice::Auto
            && ctx.library_origin() == LibraryOrigin::Precompiled
            && !probe.error_free_transforms()
        {
            let rt_opts = MetalOptions {
                library: LibraryChoice::RuntimeSource,
                ..opts.clone()
            };
            if let Ok(Some(rt)) = MetalContext::new(&rt_opts) {
                let rt_probe = run_probe(&rt)?;
                if rt_probe.error_free_transforms() {
                    (ctx, probe) = (rt, rt_probe);
                }
            }
        }
        let pso = ctx.pipeline(&PipelineKey::plain("mkl_philox_blocks"))?;
        let d = ctx.device();
        let name = d.name().to_string();
        let vendor = ["Apple", "AMD", "Intel"]
            .into_iter()
            .find(|v| name.contains(v))
            .unwrap_or("unknown")
            .to_string();
        let info = DeviceInfo {
            kind: BackendKind::Metal,
            name,
            vendor,
            max_buffer_bytes: d.maxBufferLength() as u64,
            recommended_working_set_bytes: Some(d.recommendedMaxWorkingSetSize()),
            unified_memory: d.hasUnifiedMemory(),
            ray_tracing_api: d.supportsRaytracing(),
            hardware_ray_tracing: d.supportsRaytracing() && d.supportsFamily(MTLGPUFamily::Apple9),
            native_fp64: false,
            simd_width: pso.threadExecutionWidth() as u32,
            max_threads_per_group: d.maxThreadsPerThreadgroup().width as u32,
        };
        let precision = PrecisionDescriptor {
            model: Precision::F32Gpu,
            fast_math_disabled: probe.no_reassociation && probe.no_contraction,
            opl_double_float: true,
            compensated_reductions: true,
            probe,
            budget: ErrorBudget::CONVENTIONS,
            kernel_build: format!(
                "mklight-metal {}: {}",
                env!("CARGO_PKG_VERSION"),
                ctx.library_build()
            ),
        };
        let instance = next_backend_instance();
        Ok(Some(MetalBackend {
            inner: Arc::new(Inner {
                ctx,
                info,
                precision,
                instance,
                resident: crate::resident::Runtime::new(instance),
                cache: Mutex::new(mklight_gpu::resident::cache::SystemCache::new(64 << 20)),
                next_id: AtomicU64::new(1),
                chunk_items: AtomicUsize::new(DEFAULT_CHUNK),
                systems: Mutex::new(HashMap::new()),
            }),
        }))
    }

    /// Device, library and pipeline information (safe accessors only;
    /// kernels are dispatched through `ComputeBackend`).
    pub fn context(&self) -> &MetalContext {
        &self.inner.ctx
    }

    /// Work items per command buffer (cancellation/progress granularity).
    pub fn set_chunk_items(&self, n: usize) {
        self.inner.chunk_items.store(n.max(1), Ordering::Relaxed);
    }

    /// Sizes, alignments and offsets of the shared structs as compiled by
    /// Metal: `[sizeof header, sizeof surface, alignof header, alignof
    /// surface, header offsets…, surface offsets…]` in the order of
    /// `mklight_gpu::layout::{header_offsets, surface_offsets}`.
    pub fn layout_probe(&self) -> MkResult<Vec<u32>> {
        let ctx = &self.inner.ctx;
        let n = 4
            + mklight_gpu::layout::header_offsets().len()
            + mklight_gpu::layout::surface_offsets().len();
        let mut out = ctx.buffer_zeroed::<u32>(n)?;
        let pso = ctx.pipeline(&PipelineKey::plain("mkl_layout_probe"))?;
        // SAFETY: only thread 0 writes, `n` words (the count of entries the
        // kernel writes is the length of the two offset lists plus 4).
        unsafe {
            ctx.run("mklight layout probe", |e| {
                e.set_buffer_mut(0, &mut out, 0);
                e.dispatch_1d(&pso, 1);
                Ok(())
            })?;
        }
        Ok(out.to_vec())
    }
}

impl Inner {
    fn upload(&self, set: PackedSystemSet) -> MkResult<Uploaded> {
        Ok(Uploaded {
            headers: self.ctx.buffer(&set.headers)?,
            surfaces: self.ctx.buffer(&set.surfaces)?,
            coefficients: self.ctx.buffer(&set.coefficients)?,
            sag_overlays: (set.surfaces.iter()).any(|s| s.flags & MKL_SURF_HAS_SAG_OVERLAY != 0),
            set,
        })
    }
    fn register(&self, up: Arc<Uploaded>) -> SystemHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let count = up.set.len() as u32;
        (self.systems.lock().unwrap_or_else(|e| e.into_inner())).insert(id, up);
        SystemHandle::new(self.instance, id, count)
    }
    fn lookup(&self, h: SystemHandle) -> MkResult<Arc<Uploaded>> {
        if h.backend_instance() != self.instance {
            return Err(MkError::Backend(
                "system handle belongs to another backend".into(),
            ));
        }
        self.systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&h.id())
            .cloned()
            .ok_or_else(|| MkError::Backend("unknown or released system handle".into()))
    }

    fn supports(&self, kind: JobKind) -> bool {
        match kind {
            JobKind::PhiloxStream | JobKind::PhiloxBlocks | JobKind::SagProbe => true,
            JobKind::DfOps | JobKind::Reduce | JobKind::Trace => {
                self.precision.probe.error_free_transforms()
            }
        }
    }

    fn chunk(&self) -> usize {
        self.chunk_items.load(Ordering::Relaxed)
    }

    /// Split `total` items into command buffers, checking cancellation and
    /// reporting progress between them.
    fn chunked(
        &self,
        total: usize,
        ctl: &JobControl,
        stage: &str,
        stats: &mut JobStats,
        mut f: impl FnMut(usize, usize) -> MkResult<CommandStats>,
    ) -> MkResult<()> {
        let chunk = self.chunk();
        let mut start = 0;
        while start < total {
            ctl.check()?;
            let len = chunk.min(total - start);
            let cs = f(start, len)?;
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            start += len;
            ctl.report(start as f64 / total as f64, stage);
        }
        ctl.check()
    }

    fn execute(&self, job: Job, ctl: &JobControl) -> MkResult<JobResult> {
        let t0 = Instant::now();
        let mut stats = JobStats::default();
        let gpu = &self.ctx;
        let output = match job {
            Job::PhiloxStream {
                seed,
                stream,
                first_block,
                num_blocks,
            } => {
                let n = num_blocks as usize;
                let mut raw = gpu.buffer_zeroed::<[u32; 4]>(n)?;
                let mut uni = gpu.buffer_zeroed::<[f32; 4]>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_philox_stream"))?;
                self.chunked(n, ctl, "philox", &mut stats, |start, len| {
                    let b = first_block.wrapping_add(start as u64);
                    let p = PhiloxParams {
                        key: [seed as u32, (seed >> 32) as u32],
                        stream: [stream as u32, (stream >> 32) as u32],
                        first_block: [b as u32, (b >> 32) as u32],
                        num_blocks: len as u32,
                        reserved: 0,
                    };
                    // SAFETY: thread t < len writes raw/uni[start + t], and
                    // start + len <= n = both buffers' lengths.
                    unsafe {
                        gpu.run("mkl_philox_stream", |e| {
                            e.set_bytes(0, &p);
                            e.set_buffer_mut(1, &mut raw, start);
                            e.set_buffer_mut(2, &mut uni, start);
                            e.dispatch_1d(&pso, len);
                            Ok(())
                        })
                    }
                })?;
                philox_output(&raw, &uni)
            }
            Job::PhiloxBlocks { counters, keys } => {
                let n = counters.len();
                let (bc, bk) = (gpu.buffer(&counters)?, gpu.buffer(&keys)?);
                let mut raw = gpu.buffer_zeroed::<[u32; 4]>(n)?;
                let mut uni = gpu.buffer_zeroed::<[f32; 4]>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_philox_blocks"))?;
                self.chunked(n, ctl, "philox", &mut stats, |start, len| {
                    // SAFETY: thread t < len accesses element start + t of
                    // four buffers of length n (validated: keys.len() == n).
                    unsafe {
                        gpu.run("mkl_philox_blocks", |e| {
                            e.set_buffer(0, &bc, start);
                            e.set_buffer(1, &bk, start);
                            e.set_buffer_mut(2, &mut raw, start);
                            e.set_buffer_mut(3, &mut uni, start);
                            e.set_bytes(4, &(len as u32));
                            e.dispatch_1d(&pso, len);
                            Ok(())
                        })
                    }
                })?;
                philox_output(&raw, &uni)
            }
            Job::DfOps { op, a, b } => {
                let n = a.len();
                let (ba, bb) = (gpu.buffer(&a)?, gpu.buffer(&b)?);
                let mut out = gpu.buffer_zeroed::<Df32>(n)?;
                let pso = gpu.pipeline(&PipelineKey::with_u32("mkl_df_ops", 0, op as u32))?;
                self.chunked(n, ctl, "double-float", &mut stats, |start, len| {
                    // SAFETY: thread i < len accesses element start + i of
                    // three buffers of length n (validated: b.len() == n).
                    unsafe {
                        gpu.run("mkl_df_ops", |e| {
                            e.set_buffer(0, &ba, start);
                            e.set_buffer(1, &bb, start);
                            e.set_buffer_mut(2, &mut out, start);
                            e.set_bytes(3, &(len as u32));
                            e.dispatch_1d(&pso, len);
                            Ok(())
                        })
                    }
                })?;
                JobOutput::Df(out.to_vec())
            }
            Job::Reduce {
                mode,
                count,
                values,
            } => JobOutput::Sums(self.reduce(mode, count, &values, ctl, &mut stats)?),
            Job::SagProbe {
                systems,
                system,
                surface,
                r2,
            } => {
                let up = self.lookup(systems)?;
                // Index the stored set, never trust the handle's count.
                let h = *up.set.headers.get(system as usize).ok_or_else(|| {
                    MkError::Backend("sag probe: system index out of range".into())
                })?;
                if surface >= h.num_surfaces {
                    return Err(MkError::Backend("sag probe: surface out of range".into()));
                }
                let n = r2.len();
                let br = gpu.buffer(&r2)?;
                let mut out = gpu.buffer_zeroed::<f32>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_sag_probe"))?;
                let abs_surface = h.surface_offset + surface;
                debug_assert!((abs_surface as usize) < up.surfaces.len());
                self.chunked(n, ctl, "sag", &mut stats, |start, len| {
                    let p = SagParams {
                        surface: abs_surface,
                        count: len as u32,
                        reserved: [0; 2],
                    };
                    // SAFETY: the kernel reads surfaces[abs_surface] (inside
                    // the validated set: surface_offset + surface <
                    // surface_offset + num_surfaces <= len) and, for i < len,
                    // r2/out[start + i] of buffers of length n. The record
                    // passed `PackedSystem::validate` (num_asphere <= MAX).
                    unsafe {
                        gpu.run("mkl_sag_probe", |e| {
                            e.set_buffer(0, &up.surfaces, 0);
                            e.set_bytes(1, &p);
                            e.set_buffer(2, &br, start);
                            e.set_buffer(4, &up.coefficients, 0);
                            e.set_buffer_mut(3, &mut out, start);
                            e.dispatch_1d(&pso, len);
                            Ok(())
                        })
                    }
                })?;
                JobOutput::F32(out.to_vec())
            }
            Job::Trace(req) => {
                let up = self.lookup(req.systems)?;
                JobOutput::Trace(Box::new(self.trace(&req, &up, ctl, &mut stats)?))
            }
        };
        ctl.report(1.0, "");
        stats.wall_seconds = t0.elapsed().as_secs_f64();
        Ok(JobResult { output, stats })
    }

    /// The sequential / path trace kernel (`mkl_trace`) over the request's
    /// rays, in command buffers of `chunk_items` rays.
    fn trace(
        &self,
        req: &TraceRequest,
        up: &Uploaded,
        ctl: &JobControl,
        stats: &mut JobStats,
    ) -> MkResult<TraceResult> {
        let gpu = &self.ctx;
        let prep = PreparedTrace::new(req, &up.set)?;
        let n = req.rays.len();
        let mut bu = gpu.buffer_zeroed::<u32>(n * MKL_TRACE_IN_UINTS as usize)?;
        let bf = gpu.buffer_filled::<f32>(n * MKL_TRACE_IN_FLOATS as usize, |f| {
            PreparedTrace::write_input_planes(&req.rays, f, bu.as_mut_slice())
        })?;
        let steps = gpu.buffer(&prep.steps)?;
        // The kernel writes every output and record element of every ray,
        // so these skip the explicit zeroing pass.
        let mut out_f = gpu.buffer_filled::<f32>(n * MKL_TRACE_OUT_FLOATS as usize, |_| {})?;
        let mut out_u = gpu.buffer_filled::<u32>(n * MKL_TRACE_OUT_UINTS as usize, |_| {})?;
        let rec_floats = MKL_TRACE_RECORD_FLOATS as usize;
        let mut rec_f = gpu.buffer_filled::<f32>(prep.record_slots * n * rec_floats, |_| {})?;
        let mut rec_u = gpu.buffer_filled::<u32>(prep.record_slots * n, |_| {})?;
        // Overlay code costs every trace a few percent: only where needed.
        let kernel = if up.sag_overlays {
            "mkl_trace_overlays"
        } else {
            "mkl_trace"
        };
        let pso = gpu.pipeline(&PipelineKey::plain(kernel))?;
        let recording = prep.record_slots > 0;
        self.chunked(n, ctl, "trace", stats, |start, len| {
            let p = mklight_gpu::layout::TraceParams {
                num_rays: len as u32,
                stride: n as u32,
                ..prep.params
            };
            // SAFETY: thread i < len reads in planes k·n + start + i (k <
            // MKL_TRACE_IN_FLOATS / _UINTS: inside buffers of that many
            // planes of n) and writes the out planes likewise; with records
            // it writes rec_f[(slot·n + start + i)·9 + j] and
            // rec_u[slot·n + start + i] for slot <= num_steps < record_slots.
            // It reads headers[system] (every ray's system index was checked
            // against the stored set) and, for that system, surfaces
            // surface_offset + j for the start surface and every program
            // step (validated < num_surfaces, so inside the set's surface
            // array), the step words (num_steps of them), and per-wavelength
            // arrays at the ray's wavelength (validated < num_wavelengths <=
            // MKL_MAX_WAVELENGTHS). Without records the kernel never touches
            // rec_f / rec_u (bound as 16-byte placeholders).
            unsafe {
                gpu.run(kernel, |e| {
                    e.set_buffer(0, &up.headers, 0);
                    e.set_buffer(1, &up.surfaces, 0);
                    e.set_buffer(2, &steps, 0);
                    e.set_buffer(10, &up.coefficients, 0);
                    e.set_bytes(3, &p);
                    e.set_buffer(4, &bf, start);
                    e.set_buffer(5, &bu, start);
                    e.set_buffer_mut(6, &mut out_f, start);
                    e.set_buffer_mut(7, &mut out_u, start);
                    if recording {
                        e.set_buffer_mut(8, &mut rec_f, start * rec_floats);
                        e.set_buffer_mut(9, &mut rec_u, start);
                    } else {
                        e.set_buffer_mut(8, &mut rec_f, 0);
                        e.set_buffer_mut(9, &mut rec_u, 0);
                    }
                    e.dispatch_1d(&pso, len);
                    Ok(())
                })
            }
        })?;
        Ok(TraceResult::from_planes(
            out_f.as_slice(),
            out_u.as_slice(),
            n,
            n,
            rec_f.to_vec(),
            rec_u.to_vec(),
        ))
    }

    /// Two-pass reduction in command buffers of about `chunk_items` values.
    /// Pass 1 runs over rectangles of (channel range) × (group range): whole
    /// channels when several fit in a chunk, otherwise group ranges of one
    /// channel (`first_group`). Pass 2 runs over channel ranges. The result is
    /// bit-identical to a single launch (`mklight_gpu::reduce`).
    fn reduce(
        &self,
        mode: mklight_gpu::ReduceMode,
        count: usize,
        values: &[f32],
        ctl: &JobControl,
        stats: &mut JobStats,
    ) -> MkResult<Vec<Df32>> {
        let gpu = &self.ctx;
        let chunk = self.chunk();
        let channels = values.len() / count;
        let groups = reduce::num_groups(count);
        let threads = reduce::THREADS;
        let p1 = gpu.pipeline(&PipelineKey::with_u32("mkl_reduce_pass1", 1, mode as u32))?;
        let p2 = gpu.pipeline(&PipelineKey::with_u32("mkl_reduce_pass2", 1, mode as u32))?;
        for p in [&p1, &p2] {
            if p.maxTotalThreadsPerThreadgroup() < threads {
                return Err(MkError::Unsupported(format!(
                    "reduction needs {threads} threads per group"
                )));
            }
        }
        let x = gpu.buffer(values)?;
        let mut partials = gpu.buffer_zeroed::<Df32>(groups * channels)?;
        let mut sums = gpu.buffer_zeroed::<Df32>(channels)?;
        let params = |first_group: usize| ReduceParams {
            count: count as u32,
            channel_stride: count as u32,
            num_groups: groups as u32,
            first_group: first_group as u32,
        };

        // Pass-1 launches: (first channel, channels, first group, groups).
        let per_group = count.div_ceil(groups);
        let group_step = (chunk / per_group).clamp(1, groups);
        let mut launches = Vec::new();
        if group_step == groups {
            let ch_step = (chunk / count).max(1);
            for c0 in (0..channels).step_by(ch_step) {
                launches.push((c0, ch_step.min(channels - c0), 0, groups));
            }
        } else {
            for c in 0..channels {
                for g0 in (0..groups).step_by(group_step) {
                    launches.push((c, 1, g0, group_step.min(groups - g0)));
                }
            }
        }
        // Pass 2 handles `groups` partials per channel.
        let ch2_step = (chunk / (groups * threads).max(1)).max(1);
        let total = (channels * groups + channels) as f64;
        let mut done = 0usize;
        let mut account = |cs: CommandStats, units: usize, stats: &mut JobStats| {
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            done += units;
            ctl.report(done as f64 / total, "reduce");
        };

        for (c0, nc, g0, ng) in launches {
            ctl.check()?;
            let p = params(g0);
            // SAFETY: channel c0 + y (y < nc <= channels − c0) reads
            // x[(c0 + y)·count + i] for i < count and writes
            // partials[(c0 + y)·groups + g] for g < g0 + ng <= groups: inside
            // buffers of channels·count and channels·groups elements.
            let cs = unsafe {
                gpu.run("mkl_reduce_pass1", |e| {
                    e.set_pipeline(&p1);
                    e.set_buffer(0, &x, c0 * count);
                    e.set_buffer_mut(1, &mut partials, c0 * groups);
                    e.set_bytes(2, &p);
                    e.dispatch_groups([ng, nc, 1], threads);
                    Ok(())
                })?
            };
            account(cs, nc * ng, stats);
        }
        for c0 in (0..channels).step_by(ch2_step) {
            ctl.check()?;
            let nc = ch2_step.min(channels - c0);
            let p = params(0);
            // SAFETY: channel c0 + y reads partials[(c0 + y)·groups + j] for
            // j < groups and writes sums[c0 + y], y < nc <= channels − c0.
            let cs = unsafe {
                gpu.run("mkl_reduce_pass2", |e| {
                    e.set_pipeline(&p2);
                    e.set_buffer(0, &partials, c0 * groups);
                    e.set_buffer_mut(1, &mut sums, c0);
                    e.set_bytes(2, &p);
                    e.dispatch_groups([1, nc, 1], threads);
                    Ok(())
                })?
            };
            account(cs, nc, stats);
        }
        ctl.check()?;
        Ok(sums.to_vec())
    }
}

fn philox_output(raw: &SharedBuffer<[u32; 4]>, uni: &SharedBuffer<[f32; 4]>) -> JobOutput {
    JobOutput::Philox {
        raw: raw.as_slice().iter().flatten().copied().collect(),
        uniform: uni.as_slice().iter().flatten().copied().collect(),
    }
}

impl ComputeBackend for MetalBackend {
    fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }
    fn precision(&self) -> &PrecisionDescriptor {
        &self.inner.precision
    }
    fn supports(&self, kind: JobKind) -> bool {
        self.inner.supports(kind)
    }

    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        // `PackedSystemSet::new` validates every record (kernels rely on it).
        let up = self.inner.upload(PackedSystemSet::new(systems)?)?;
        Ok(self.inner.register(Arc::new(up)))
    }

    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.lookup(handle)?;
        // Jobs in flight keep their own `Arc`, so the buffers outlive them.
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id());
        Ok(())
    }

    fn upload_buffer(
        &self,
        desc: mklight_gpu::resident::BufferDesc,
        data: mklight_gpu::resident::BufferData,
    ) -> MkResult<mklight_gpu::resident::BufferHandle> {
        if desc.byte_len()? as u64 > self.info().max_buffer_bytes {
            return Err(mklight_gpu::resident::bad("upload exceeds device limit"));
        }
        let ctx = &self.inner.ctx;
        self.inner
            .resident
            .upload(desc, data, |n| ctx.buffer_filled(n, |_| {}))
    }
    fn fill_buffer(
        &self,
        desc: mklight_gpu::resident::BufferDesc,
        ctx: &JobContext,
        fill: &mut dyn FnMut(mklight_gpu::resident::BufferFill<'_>) -> MkResult<()>,
    ) -> MkResult<mklight_gpu::resident::BufferHandle> {
        self.inner
            .resident
            .fill(desc, ctx, fill, |n| self.inner.ctx.buffer_filled(n, |_| {}))
    }
    fn import_buffer(
        &self,
        desc: mklight_gpu::resident::BufferDesc,
        data: mklight_gpu::resident::HostAllocation,
    ) -> MkResult<mklight_gpu::resident::BufferHandle> {
        let storage = SharedBuffer::from_host(self.inner.ctx.device(), data)?;
        self.inner.resident.adopt(desc, storage)
    }
    fn read_view(
        &self,
        view: mklight_gpu::resident::BufferView,
        ctx: &JobContext,
    ) -> MkResult<mklight_gpu::resident::ReadView> {
        self.inner.resident.read_view(view, ctx)
    }
    fn readback(
        &self,
        view: mklight_gpu::resident::BufferView,
        ctx: &JobContext,
    ) -> MkResult<mklight_gpu::resident::Readback> {
        self.inner.resident.readback(view, ctx)
    }
    fn resident_cache_stats(&self) -> mklight_gpu::resident::ResidentCacheStats {
        let s = self
            .inner
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats();
        self.inner.resident.cache_stats(s)
    }
    fn upload_systems_cached(
        &self,
        key: &mklight_gpu::resident::SystemCacheKey,
        systems: &[PackedSystem],
    ) -> MkResult<SystemHandle> {
        let hash = mklight_gpu::resident::cache::fingerprint(
            key,
            systems,
            &self.precision().kernel_build,
        )?;
        let mut cache = self.inner.cache.lock().unwrap_or_else(|e| e.into_inner());
        let up = match cache.get(key, hash) {
            Some(up) => up,
            None => {
                let up = Arc::new(self.inner.upload(PackedSystemSet::new(systems)?)?);
                cache.insert(key.clone(), hash, up.set.byte_len(), up.clone());
                up
            }
        };
        drop(cache);
        Ok(self.inner.register(up))
    }
    fn submit_plan(
        &self,
        plan: mklight_gpu::resident::ResidentPlan,
        ctx: &JobContext,
    ) -> MkResult<Pending<mklight_gpu::resident::PlanResult>> {
        use mklight_gpu::resident::ResidentOp;
        ctx.check()?;
        plan.validate(self.inner.instance, self.info().max_buffer_bytes)?;
        if !self.precision().probe.error_free_transforms() {
            return Err(MkError::Unsupported(
                "resident plan requires strict numerics".into(),
            ));
        }
        let bytes: u64 = plan
            .buffers
            .iter()
            .map(|b| b.desc().byte_len().map(|n| n as u64))
            .sum::<MkResult<u64>>()?;
        if self
            .info()
            .recommended_working_set_bytes
            .is_some_and(|limit| bytes > limit)
        {
            return Err(MkError::Unsupported(
                "resident plan exceeds device working set; tile the request".into(),
            ));
        }
        let mut pinned = HashMap::new();
        for h in plan.system_handles() {
            pinned.insert(h, self.inner.lookup(h)?);
        }
        let prepared = plan
            .operations
            .iter()
            .map(|op| match op {
                ResidentOp::TraceUpload(t) => {
                    PreparedTrace::new(t.request(), &pinned[&t.request().systems].set).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<MkResult<Vec<_>>>()?;
        let (p, c, ctl) = pending(ctx);
        let b = self.clone();
        let chunk = self.inner.chunk();
        spawn_job(
            "mklight-resident-metal",
            "Metal resident plan".into(),
            c,
            move || {
                let i = &b.inner;
                crate::resident::execute(
                    &i.resident,
                    &b,
                    &i.ctx,
                    &plan,
                    pinned,
                    &prepared,
                    &ctl,
                    chunk,
                )
            },
        )?;
        Ok(p)
    }
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        job.validate()?;
        if !self.inner.supports(job.kind()) {
            return Err(MkError::Unsupported(format!(
                "metal backend cannot run {:?} on '{}' (numerics probe: {:?})",
                job.kind(),
                self.inner.info.name,
                self.inner.precision.probe
            )));
        }
        match &job {
            Job::SagProbe { systems, .. } => {
                self.inner.lookup(*systems)?;
            }
            Job::Trace(req) => {
                PreparedTrace::new(req, &self.inner.lookup(req.systems)?.set)?;
            }
            _ => {}
        }
        let (p, c, ctl) = pending(ctx);
        let inner = self.inner.clone();
        let what = format!("metal {:?} job", job.kind());
        spawn_job("mklight-metal", what, c, move || inner.execute(job, &ctl))?;
        Ok(p)
    }
}
