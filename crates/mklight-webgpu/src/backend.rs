//! `WebGpuBackend`: `mklight_gpu::ComputeBackend` on WebGPU (GPU.md §11).
//!
//! WebGPU memory is not host-visible: buffers are written with
//! `queue.writeBuffer` and read through mapped staging copies. Plan storage
//! therefore lives on the host, in the CPU-f32 backend's resident runtime
//! (`HostAllocation`s, so `fill_buffer` and `read_view` stay zero-copy for
//! the solvers), and the plan runs on its executor
//! (`CpuF32Backend::submit_plan_hosted`): validation, system pinning,
//! operation caches, cancellation, progress and publication are the shared
//! `mklight-gpu` code. The operations this backend has WGSL kernels for
//! ([`crate::ops`]) run on the device: their inputs are uploaded from the
//! plan's buffers, their outputs read back into them. Every other operation
//! runs its CPU-f32 specification in the device's f32 environment
//! (flush-to-zero when the probe says so), which is the arithmetic the
//! kernels specify; `PlanStats` names each stage's placement
//! (`webgpu:<operation>` or the specification's name).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use mklight_core::job::Precision;
use mklight_core::{JobContext, MkError, MkResult};
use mklight_gpu::pending::{JobControl, pending, spawn_job};
use mklight_gpu::resident::{
    BufferData, BufferDesc, BufferFill, BufferHandle, BufferView, HostAllocation, PlanResult,
    ReadView, Readback, ResidentCacheStats, ResidentPlan, SystemCacheKey,
};
use mklight_gpu::{
    BackendKind, ComputeBackend, CpuF32Backend, DeviceInfo, Df32, ErrorBudget, Job, JobKind,
    JobOutput, JobResult, JobStats, KernelPreparation, NumericsProbe, PackedSystem,
    PackedSystemSet, Pending, PrecisionDescriptor, QualificationBackend, SystemHandle,
};

use crate::device::*;
use crate::kernels;

/// Device copies of packed systems kept across plans (by content digest).
const SYSTEM_CACHE_BYTES: usize = 64 << 20;

/// A packed system set in device memory: headers, surfaces and coefficient
/// pool as three read-only storage buffers. Released when dropped.
pub struct DeviceSystems {
    gpu: Arc<dyn Gpu>,
    pub headers: BufferKey,
    pub surfaces: BufferKey,
    pub coefficients: BufferKey,
    pub count: u32,
    bytes: usize,
}

impl Drop for DeviceSystems {
    fn drop(&mut self) {
        self.gpu.release(self.headers);
        self.gpu.release(self.surfaces);
        self.gpu.release(self.coefficients);
    }
}

pub(crate) struct Inner {
    pub(crate) gpu: Arc<dyn Gpu>,
    pub(crate) host: CpuF32Backend,
    info: DeviceInfo,
    precision: PrecisionDescriptor,
    /// The compiler's own treatment of unprotected expressions.
    raw_probe: NumericsProbe,
    systems: Mutex<Vec<([u8; 32], Arc<DeviceSystems>)>>,
}

#[derive(Clone)]
pub struct WebGpuBackend {
    pub(crate) inner: Arc<Inner>,
}

/// Run the numerics probe kernel: (raw, protected) verdicts.
fn run_probe(gpu: &dyn Gpu) -> MkResult<(NumericsProbe, NumericsProbe)> {
    let (a, b, c) = mklight_gpu::probe::inputs();
    let n = a.len();
    let w = mklight_gpu::probe::WORDS;
    let mut out = vec![0u32; 2 * n * w];
    let mut batch = Batch::default();
    let ra = batch.upload(bytemuck::cast_slice(&a));
    let rb = batch.upload(bytemuck::cast_slice(&b));
    let rc = batch.upload(bytemuck::cast_slice(&c));
    let ro = batch.scratch(2 * n * w);
    batch.passes.push(Pass {
        kernel: Kernel::new("numerics", "mkl_numerics_probe"),
        uniform: Some(uniform_words(&[n as u32])),
        storage: vec![
            Binding::read(ra),
            Binding::read(rb),
            Binding::read(rc),
            Binding::write(ro),
        ],
        workgroups: [(n as u32).div_ceil(64), 1, 1],
    });
    batch.read(ro, 0, &mut out);
    gpu.run(batch)?;
    let f: &[f32] = bytemuck::cast_slice(&out);
    Ok((
        mklight_gpu::probe::evaluate(&a, &b, &c, &f[..n * w]),
        mklight_gpu::probe::evaluate(&a, &b, &c, &f[n * w..]),
    ))
}

impl WebGpuBackend {
    /// The default native adapter through wgpu, or `Ok(None)` when there is
    /// none.
    pub fn new() -> MkResult<Option<Self>> {
        match crate::WgpuDevice::open_native()? {
            Some(d) => Self::with_gpu(Arc::new(d)).map(Some),
            None => Ok(None),
        }
    }

    /// A backend on `gpu` (natively a `WgpuDevice`, in the browser the GPU
    /// thread's proxy). Runs the numerics probe.
    pub fn with_gpu(gpu: Arc<dyn Gpu>) -> MkResult<Self> {
        let (raw, probe) = run_probe(&*gpu)?;
        let s = gpu.summary().clone();
        let host = CpuF32Backend::matching(&probe);
        let info = DeviceInfo {
            kind: BackendKind::WebGpu,
            name: s.name.clone(),
            vendor: if s.backend == "Metal" && s.name.starts_with("Apple") {
                "Apple".into()
            } else {
                s.vendor.clone()
            },
            // Plan storage is host memory (module docs).
            max_buffer_bytes: host.info().max_buffer_bytes,
            recommended_working_set_bytes: None,
            // The API copies through staging buffers on every device.
            unified_memory: false,
            ray_tracing_api: false,
            hardware_ray_tracing: false,
            native_fp64: false,
            simd_width: 1,
            max_threads_per_group: s.max_compute_invocations_per_workgroup,
        };
        let yn = |b: bool| if b { "yes" } else { "no" };
        let precision = PrecisionDescriptor {
            model: Precision::F32Gpu,
            fast_math_disabled: raw.no_reassociation && raw.no_contraction,
            opl_double_float: true,
            compensated_reductions: true,
            probe,
            budget: ErrorBudget::CONVENTIONS,
            kernel_build: format!(
                "WGSL {} via {} ({}); opaque-barrier double-float; compiler without barriers: \
                 contraction {}, reassociation {}, correctly rounded div {} / sqrt {}",
                kernels::digest(),
                s.backend,
                s.driver,
                yn(!raw.no_contraction),
                yn(!raw.no_reassociation),
                yn(raw.correctly_rounded_div),
                yn(raw.correctly_rounded_sqrt),
            ),
        };
        Ok(Self {
            inner: Arc::new(Inner {
                gpu,
                host,
                info,
                precision,
                raw_probe: raw,
                systems: Mutex::new(Vec::new()),
            }),
        })
    }

    /// The device seam.
    pub fn gpu(&self) -> &Arc<dyn Gpu> {
        &self.inner.gpu
    }

    /// The adapter and its limits.
    pub fn adapter(&self) -> &AdapterSummary {
        self.inner.gpu.summary()
    }

    /// The probe of plain (unprotected) WGSL expressions on this compiler.
    pub fn raw_probe(&self) -> &NumericsProbe {
        &self.inner.raw_probe
    }

    /// The CPU-f32 host holding this backend's storage and systems.
    pub fn host(&self) -> &CpuF32Backend {
        &self.inner.host
    }

    fn double_float(&self) -> bool {
        self.inner.precision.probe.error_free_transforms()
    }

    /// The device copy of a pinned system set (uploaded on first use, then
    /// shared by content).
    pub fn device_systems(&self, set: &PackedSystemSet) -> MkResult<Arc<DeviceSystems>> {
        let digest: [u8; 32] = *set.digest().as_bytes();
        let mut cache = self.inner.systems.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = cache.iter().position(|(d, _)| *d == digest) {
            let e = cache.remove(i);
            let s = e.1.clone();
            cache.push(e);
            return Ok(s);
        }
        let gpu = &self.inner.gpu;
        let headers = gpu.persist(bytemuck::cast_slice(&set.headers))?;
        let surfaces = gpu.persist(bytemuck::cast_slice(&set.surfaces))?;
        let coefficients = if set.coefficients.is_empty() {
            gpu.persist(&[0; 4])?
        } else {
            gpu.persist(bytemuck::cast_slice(&set.coefficients))?
        };
        let s = Arc::new(DeviceSystems {
            gpu: gpu.clone(),
            headers,
            surfaces,
            coefficients,
            count: set.headers.len() as u32,
            bytes: set.byte_len(),
        });
        cache.push((digest, s.clone()));
        let mut total: usize = cache.iter().map(|(_, s)| s.bytes).sum();
        while total > SYSTEM_CACHE_BYTES && cache.len() > 1 {
            total -= cache.remove(0).1.bytes;
        }
        Ok(s)
    }

    /// Device double-float operations (qualification).
    fn df_ops(&self, op: mklight_gpu::DfOp, a: &[Df32], b: &[Df32]) -> MkResult<Vec<Df32>> {
        let n = a.len();
        let mut out = vec![Df32::default(); n];
        let max = (self.adapter().max_storage_buffer_binding_size / 8) as usize;
        let groups = self.adapter().max_compute_workgroups_per_dimension as usize * 64;
        let step = max.min(groups).max(1);
        for (k, o) in out.chunks_mut(step).enumerate() {
            let r = k * step..k * step + o.len();
            let mut batch = Batch::default();
            let ra = batch.upload(bytemuck::cast_slice(&a[r.clone()]));
            let rb = batch.upload(bytemuck::cast_slice(&b[r.clone()]));
            let ro = batch.scratch(2 * o.len());
            batch.passes.push(Pass {
                kernel: Kernel::new("numerics", "mkl_df_ops"),
                uniform: Some(uniform_words(&[o.len() as u32, op as u32])),
                storage: vec![Binding::read(ra), Binding::read(rb), Binding::write(ro)],
                workgroups: [(o.len() as u32).div_ceil(64), 1, 1],
            });
            batch.read(ro, 0, bytemuck::cast_slice_mut(o));
            self.inner.gpu.run(batch)?;
        }
        Ok(out)
    }

    fn execute_job(&self, job: Job, ctl: &JobControl) -> MkResult<JobResult> {
        let t0 = Instant::now();
        let mut stats = JobStats::default();
        let output = match job {
            Job::DfOps { op, a, b } => {
                let r = self.df_ops(op, &a, &b)?;
                stats.command_buffers += 1;
                JobOutput::Df(r)
            }
            Job::Reduce {
                mode,
                count,
                values,
            } => {
                let channels = values.len() / count;
                let mut sums = vec![Df32::default(); channels];
                crate::ops::sum(
                    &*self.inner.gpu,
                    mode,
                    channels,
                    count,
                    bytemuck::cast_slice(&values),
                    bytemuck::cast_slice_mut(&mut sums),
                    ctl,
                    &mut stats,
                )?;
                JobOutput::Sums(sums)
            }
            Job::Trace(req) => {
                let set = self.inner.host.system_set(req.systems)?;
                JobOutput::Trace(Box::new(crate::ops::trace_job(
                    self, &req, &set, ctl, &mut stats,
                )?))
            }
            other => {
                return Err(MkError::Unsupported(format!(
                    "webgpu does not run {:?} jobs",
                    other.kind()
                )));
            }
        };
        ctl.report(1.0, "");
        stats.wall_seconds = t0.elapsed().as_secs_f64();
        Ok(JobResult { output, stats })
    }
}

impl ComputeBackend for WebGpuBackend {
    fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }
    fn precision(&self) -> &PrecisionDescriptor {
        &self.inner.precision
    }
    fn supports(&self, kind: JobKind) -> bool {
        match kind {
            JobKind::DfOps | JobKind::Reduce => self.double_float(),
            JobKind::Trace => self.double_float() && crate::ops::TRACE_PORTED,
            _ => false,
        }
    }
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        self.inner.host.upload_systems(systems)
    }
    fn upload_systems_cached(
        &self,
        key: &SystemCacheKey,
        systems: &[PackedSystem],
    ) -> MkResult<SystemHandle> {
        self.inner.host.upload_systems_cached(key, systems)
    }
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.host.release_systems(handle)
    }
    fn fill_buffer(
        &self,
        desc: BufferDesc,
        fill: &mut dyn FnMut(BufferFill<'_>) -> MkResult<()>,
    ) -> MkResult<BufferHandle> {
        self.inner.host.fill_buffer(desc, fill)
    }
    fn read_view(&self, view: BufferView) -> MkResult<ReadView> {
        self.inner.host.read_view(view)
    }
    fn submit_plan(&self, plan: ResidentPlan, ctx: &JobContext) -> MkResult<Pending<PlanResult>> {
        if !self.double_float() {
            return Err(MkError::Unsupported(format!(
                "{} cannot run double-float kernels (numerics probe: {:?})",
                self.info().backend_id(),
                self.precision().probe
            )));
        }
        let me = self.clone();
        self.inner
            .host
            .submit_plan_hosted(Arc::new(self.clone()), plan, ctx, move |pinned| {
                Ok(Box::new(crate::ops::WebGpuOps::new(me, pinned)))
            })
    }
    fn resident_cache_stats(&self) -> ResidentCacheStats {
        self.inner.host.resident_cache_stats()
    }
    fn prepare_kernels(
        &self,
        _cache_dir: Option<&std::path::Path>,
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &std::sync::atomic::AtomicBool,
    ) -> MkResult<KernelPreparation> {
        let t = Instant::now();
        let all = crate::ops::pipelines();
        for (i, k) in all.iter().enumerate() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            self.inner.gpu.prepare(std::slice::from_ref(k))?;
            progress(i + 1, all.len());
        }
        Ok(KernelPreparation {
            kernels: all.len(),
            seconds: t.elapsed().as_secs_f64(),
            ..Default::default()
        })
    }
}

impl QualificationBackend for WebGpuBackend {
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        job.validate()?;
        if !self.supports(job.kind()) {
            return Err(MkError::Unsupported(format!(
                "webgpu does not run {:?} jobs on '{}' (numerics probe: {:?})",
                job.kind(),
                self.info().name,
                self.precision().probe
            )));
        }
        if let Job::Trace(req) = &job {
            mklight_gpu::PreparedTrace::new(req, &*self.inner.host.system_set(req.systems)?)?;
        }
        let (p, c, ctl) = pending(ctx);
        let b = self.clone();
        let what = format!("webgpu {:?} job", job.kind());
        spawn_job("mklight-webgpu", what, c, move || b.execute_job(job, &ctl))?;
        Ok(p)
    }
    fn upload_buffer(&self, desc: BufferDesc, data: BufferData) -> MkResult<BufferHandle> {
        self.inner.host.upload_buffer(desc, data)
    }
    fn import_buffer(&self, desc: BufferDesc, data: HostAllocation) -> MkResult<BufferHandle> {
        self.inner.host.import_buffer(desc, data)
    }
    fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback> {
        self.inner.host.readback(view, ctx)
    }
}
