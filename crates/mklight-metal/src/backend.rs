//! `MetalBackend`: `mklight_gpu::ComputeBackend` on Metal.
//!
//! Jobs run on a worker thread per submission. Large jobs are split into
//! command buffers of at most `chunk_items` work items; the job's cancel
//! token is checked and progress reported between them. Each command buffer
//! is awaited and its status checked, so GPU errors surface as
//! `MkError::Backend` and GPU time is summed into `JobStats`.

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
use mklight_gpu::pending::pending;
use mklight_gpu::{
    BackendKind, ComputeBackend, DeviceInfo, Df32, ErrorBudget, Job, JobKind, JobOutput, JobResult,
    JobStats, NumericsProbe, PackedHeader, PackedSurface, PackedSystem, PackedSystemSet, Pending,
    PrecisionDescriptor, SystemHandle, reduce,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLGPUFamily};

use crate::buffer::SharedBuffer;
use crate::context::{CommandStats, MetalContext, MetalOptions, PipelineKey};

/// Default work items per command buffer.
const DEFAULT_CHUNK: usize = 1 << 20;

struct Uploaded {
    #[allow(dead_code)] // bound by trace kernels (next phase)
    headers: SharedBuffer<PackedHeader>,
    surfaces: SharedBuffer<PackedSurface>,
    set: PackedSystemSet,
}

struct Inner {
    ctx: MetalContext,
    info: DeviceInfo,
    precision: PrecisionDescriptor,
    instance: u64,
    next_id: AtomicU64,
    chunk_items: AtomicUsize,
    systems: Mutex<HashMap<u64, Arc<Uploaded>>>,
}

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
    let out = ctx.buffer_zeroed::<f32>(n * words)?;
    let pso = ctx.pipeline(&PipelineKey::plain("mkl_numerics_probe"))?;
    ctx.run("mklight numerics probe", |e| {
        e.set_buffer(0, &ba, 0);
        e.set_buffer(1, &bb, 0);
        e.set_buffer(2, &bc, 0);
        e.set_buffer(3, &out, 0);
        e.set_bytes(4, &(n as u32));
        e.dispatch_1d(&pso, n);
        Ok(())
    })?;
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

    pub fn with_options(opts: &MetalOptions) -> MkResult<Option<Self>> {
        let Some(ctx) = MetalContext::new(opts)? else {
            return Ok(None);
        };
        let probe = run_probe(&ctx)?;
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
        };
        Ok(Some(MetalBackend {
            inner: Arc::new(Inner {
                ctx,
                info,
                precision,
                instance: next_backend_instance(),
                next_id: AtomicU64::new(1),
                chunk_items: AtomicUsize::new(DEFAULT_CHUNK),
                systems: Mutex::new(HashMap::new()),
            }),
        }))
    }

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
        let out = ctx.buffer_zeroed::<u32>(n)?;
        let pso = ctx.pipeline(&PipelineKey::plain("mkl_layout_probe"))?;
        ctx.run("mklight layout probe", |e| {
            e.set_buffer(0, &out, 0);
            e.dispatch_1d(&pso, 1);
            Ok(())
        })?;
        Ok(out.to_vec())
    }
}

impl Inner {
    fn lookup(&self, h: SystemHandle) -> MkResult<Arc<Uploaded>> {
        if h.backend_instance != self.instance {
            return Err(MkError::Backend(
                "system handle belongs to another backend".into(),
            ));
        }
        self.systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&h.id)
            .cloned()
            .ok_or_else(|| MkError::Backend("unknown or released system handle".into()))
    }

    fn supports(&self, kind: JobKind) -> bool {
        match kind {
            JobKind::PhiloxStream | JobKind::PhiloxBlocks | JobKind::SagProbe => true,
            JobKind::DfOps | JobKind::Reduce => self.precision.probe.error_free_transforms(),
            JobKind::Trace => false,
        }
    }

    /// Split `total` items into command buffers, checking cancellation and
    /// reporting progress between them.
    fn chunked(
        &self,
        total: usize,
        ctx: &JobContext,
        stage: &str,
        stats: &mut JobStats,
        mut f: impl FnMut(usize, usize) -> MkResult<CommandStats>,
    ) -> MkResult<()> {
        let chunk = self.chunk_items.load(Ordering::Relaxed);
        let mut start = 0;
        while start < total {
            ctx.check()?;
            let len = chunk.min(total - start);
            let cs = f(start, len)?;
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            start += len;
            ctx.report(start as f64 / total as f64, stage);
        }
        ctx.check()
    }

    fn execute(&self, job: Job, ctx: &JobContext) -> MkResult<JobResult> {
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
                let raw = gpu.buffer_zeroed::<[u32; 4]>(n)?;
                let uni = gpu.buffer_zeroed::<[f32; 4]>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_philox_stream"))?;
                self.chunked(n, ctx, "philox", &mut stats, |start, len| {
                    let b = first_block.wrapping_add(start as u64);
                    let p = PhiloxParams {
                        key: [seed as u32, (seed >> 32) as u32],
                        stream: [stream as u32, (stream >> 32) as u32],
                        first_block: [b as u32, (b >> 32) as u32],
                        num_blocks: len as u32,
                        reserved: 0,
                    };
                    gpu.run("mkl_philox_stream", |e| {
                        e.set_bytes(0, &p);
                        e.set_buffer(1, &raw, start);
                        e.set_buffer(2, &uni, start);
                        e.dispatch_1d(&pso, len);
                        Ok(())
                    })
                })?;
                philox_output(&raw, &uni)
            }
            Job::PhiloxBlocks { counters, keys } => {
                let n = counters.len();
                let (bc, bk) = (gpu.buffer(&counters)?, gpu.buffer(&keys)?);
                let raw = gpu.buffer_zeroed::<[u32; 4]>(n)?;
                let uni = gpu.buffer_zeroed::<[f32; 4]>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_philox_blocks"))?;
                self.chunked(n, ctx, "philox", &mut stats, |start, len| {
                    gpu.run("mkl_philox_blocks", |e| {
                        e.set_buffer(0, &bc, start);
                        e.set_buffer(1, &bk, start);
                        e.set_buffer(2, &raw, start);
                        e.set_buffer(3, &uni, start);
                        e.set_bytes(4, &(len as u32));
                        e.dispatch_1d(&pso, len);
                        Ok(())
                    })
                })?;
                philox_output(&raw, &uni)
            }
            Job::DfOps { op, a, b } => {
                let n = a.len();
                let (ba, bb) = (gpu.buffer(&a)?, gpu.buffer(&b)?);
                let out = gpu.buffer_zeroed::<Df32>(n)?;
                let pso = gpu.pipeline(&PipelineKey::with_u32("mkl_df_ops", 0, op as u32))?;
                self.chunked(n, ctx, "double-float", &mut stats, |start, len| {
                    gpu.run("mkl_df_ops", |e| {
                        e.set_buffer(0, &ba, start);
                        e.set_buffer(1, &bb, start);
                        e.set_buffer(2, &out, start);
                        e.set_bytes(3, &(len as u32));
                        e.dispatch_1d(&pso, len);
                        Ok(())
                    })
                })?;
                JobOutput::Df(out.to_vec())
            }
            Job::Reduce {
                mode,
                count,
                values,
            } => {
                let channels = values.len() / count;
                let groups = reduce::num_groups(count);
                let threads = reduce::THREADS;
                let p1 =
                    gpu.pipeline(&PipelineKey::with_u32("mkl_reduce_pass1", 1, mode as u32))?;
                let p2 =
                    gpu.pipeline(&PipelineKey::with_u32("mkl_reduce_pass2", 1, mode as u32))?;
                for p in [&p1, &p2] {
                    if p.maxTotalThreadsPerThreadgroup() < threads {
                        return Err(MkError::Unsupported(format!(
                            "reduction needs {threads} threads per group"
                        )));
                    }
                }
                let x = gpu.buffer(&values)?;
                let partials = gpu.buffer_zeroed::<Df32>(groups * channels)?;
                let sums = gpu.buffer_zeroed::<Df32>(channels)?;
                let params = ReduceParams {
                    count: count as u32,
                    channel_stride: count as u32,
                    num_groups: groups as u32,
                    reserved: 0,
                };
                ctx.check()?;
                let cs = gpu.run("mkl_reduce", |e| {
                    e.set_pipeline(&p1);
                    e.set_buffer(0, &x, 0);
                    e.set_buffer(1, &partials, 0);
                    e.set_bytes(2, &params);
                    e.dispatch_groups([groups, channels, 1], threads);
                    // Serial compute encoder: pass 2 starts after pass 1.
                    e.set_pipeline(&p2);
                    e.set_buffer(0, &partials, 0);
                    e.set_buffer(1, &sums, 0);
                    e.dispatch_groups([1, channels, 1], threads);
                    Ok(())
                })?;
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                ctx.report(1.0, "reduce");
                JobOutput::Sums(sums.to_vec())
            }
            Job::SagProbe {
                systems,
                system,
                surface,
                r2,
            } => {
                let up = self.lookup(systems)?;
                let h = up.set.headers[system as usize];
                if surface >= h.num_surfaces {
                    return Err(MkError::Backend("sag probe: surface out of range".into()));
                }
                let n = r2.len();
                let br = gpu.buffer(&r2)?;
                let out = gpu.buffer_zeroed::<f32>(n)?;
                let pso = gpu.pipeline(&PipelineKey::plain("mkl_sag_probe"))?;
                let abs_surface = h.surface_offset + surface;
                self.chunked(n, ctx, "sag", &mut stats, |start, len| {
                    let p = SagParams {
                        surface: abs_surface,
                        count: len as u32,
                        reserved: [0; 2],
                    };
                    gpu.run("mkl_sag_probe", |e| {
                        e.set_buffer(0, &up.surfaces, 0);
                        e.set_bytes(1, &p);
                        e.set_buffer(2, &br, start);
                        e.set_buffer(3, &out, start);
                        e.dispatch_1d(&pso, len);
                        Ok(())
                    })
                })?;
                JobOutput::F32(out.to_vec())
            }
            Job::Trace(_) => {
                return Err(MkError::Unsupported(
                    "metal: sequential trace kernel not implemented yet".into(),
                ));
            }
        };
        stats.wall_seconds = t0.elapsed().as_secs_f64();
        Ok(JobResult { output, stats })
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
        let set = PackedSystemSet::new(systems)?;
        let up = Uploaded {
            headers: self.inner.ctx.buffer(&set.headers)?,
            surfaces: self.inner.ctx.buffer(&set.surfaces)?,
            set,
        };
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let count = up.set.len() as u32;
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, Arc::new(up));
        Ok(SystemHandle {
            backend_instance: self.inner.instance,
            id,
            count,
        })
    }

    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.lookup(handle)?;
        // Jobs in flight keep their own `Arc`, so the buffers outlive them.
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id);
        Ok(())
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
        if let Job::SagProbe { systems, .. } = &job {
            self.inner.lookup(*systems)?;
        }
        let (p, c) = pending(ctx.cancel.clone());
        let inner = self.inner.clone();
        let ctx = ctx.clone();
        std::thread::Builder::new()
            .name("mklight-metal".into())
            .spawn(move || c.complete(inner.execute(job, &ctx)))
            .map_err(|e| MkError::Backend(e.to_string()))?;
        Ok(p)
    }
}
