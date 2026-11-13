//! CPU-f32 emulation backend.
//!
//! Runs the GPU algorithms on the CPU with the same f32 operations in the
//! same order and the same f32 environment ([`crate::fenv`]: IEEE, or
//! flush-to-zero like Apple GPUs). Its results are bit-identical to a GPU
//! whose numerics probe reports error-free transforms when it is built with
//! [`CpuF32Backend::matching`] for that GPU's probe; division, square root and
//! the sag additionally need the probe's correctly rounded `/` and `sqrt`.
//! Comparing *reference f64 ↔ CPU-f32* isolates precision error; comparing
//! *CPU-f32 ↔ GPU* isolates backend/compiler error.
//!
//! Kernels: Philox (stream and raw blocks), double-float operations,
//! compensated reduction, the packed-sag probe and the sequential / path
//! trace kernel ([`crate::trace_kernel`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mklight_core::job::Precision;
use mklight_core::rng::philox4x32_10;
use mklight_core::{JobContext, MkError, MkResult};
use rayon::prelude::*;

use crate::backend::*;
use crate::batch::{PreparedTrace, TraceRequest, TraceResult};
use crate::consts::*;
use crate::df::Df32;
use crate::fenv::{FpEnv, Ftz, Ieee};
use crate::fmath::{packed_sag_with_pool_in, u32_to_unit_f32};
use crate::layout::{PackedSystem, PackedSystemSet};
use crate::pending::{JobControl, Pending, pending, spawn_job};
use crate::reduce;
use crate::trace_kernel::{KRayIn, KRayOut, KState, trace_ray};

/// Work items between cancellation checks / progress reports.
pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone)]
pub struct CpuF32Backend {
    inner: Arc<Inner>,
}

struct Inner {
    info: DeviceInfo,
    precision: PrecisionDescriptor,
    instance: u64,
    next_id: AtomicU64,
    systems: Mutex<HashMap<u64, Arc<PackedSystemSet>>>,
    resident: crate::resident::runtime::Runtime<crate::resident::HostAllocation>,
    cache: Mutex<crate::resident::cache::SystemCache<PackedSystemSet>>,
}

impl Default for CpuF32Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuF32Backend {
    /// IEEE binary32 with subnormals.
    pub fn new() -> Self {
        Self::with_subnormals(true)
    }

    /// The environment of a device with this numerics probe: flush-to-zero
    /// when `probe.subnormals` is false.
    pub fn matching(probe: &NumericsProbe) -> Self {
        Self::with_subnormals(probe.subnormals)
    }

    /// IEEE (`true`) or flush-to-zero (`false`) f32 environment.
    pub fn with_subnormals(subnormals: bool) -> Self {
        let env = if subnormals {
            "IEEE binary32, subnormals kept"
        } else {
            "flush-to-zero"
        };
        let instance = next_backend_instance();
        let inner = Inner {
            info: DeviceInfo {
                kind: BackendKind::CpuF32,
                name: format!("CPU f32 emulation ({env})"),
                vendor: std::env::consts::ARCH.into(),
                max_buffer_bytes: isize::MAX as u64,
                recommended_working_set_bytes: None,
                unified_memory: true,
                ray_tracing_api: false,
                hardware_ray_tracing: false,
                native_fp64: true,
                simd_width: 1,
                max_threads_per_group: reduce::THREADS as u32,
            },
            precision: PrecisionDescriptor {
                model: Precision::F32Gpu,
                fast_math_disabled: true,
                opl_double_float: true,
                compensated_reductions: true,
                probe: NumericsProbe {
                    subnormals,
                    ..NumericsProbe::IEEE
                },
                budget: ErrorBudget::CONVENTIONS,
                kernel_build: format!(
                    "mklight-gpu {} Rust f32 emulation, {env}",
                    env!("CARGO_PKG_VERSION")
                ),
            },
            instance,
            resident: crate::resident::runtime::Runtime::new(instance),
            cache: Mutex::new(crate::resident::cache::SystemCache::new()),
            next_id: AtomicU64::new(1),
            systems: Mutex::new(HashMap::new()),
        };
        CpuF32Backend {
            inner: Arc::new(inner),
        }
    }
}

impl Inner {
    fn register(&self, set: Arc<PackedSystemSet>) -> SystemHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let count = set.len() as u32;
        (self.systems.lock().unwrap_or_else(|e| e.into_inner())).insert(id, set);
        SystemHandle::new(self.instance, id, count)
    }
    fn lookup(&self, h: SystemHandle) -> MkResult<Arc<PackedSystemSet>> {
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
}

/// Run `f` over `0..total` in chunks, checking cancellation and reporting
/// progress between chunks (the CPU analogue of command buffers).
fn chunked<T: Send>(
    total: usize,
    ctl: &JobControl,
    stage: &str,
    stats: &mut JobStats,
    f: impl Fn(usize) -> T + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    while start < total {
        ctl.check()?;
        let end = (start + CHUNK).min(total);
        out.par_extend((start..end).into_par_iter().map(&f));
        stats.command_buffers += 1;
        start = end;
        ctl.report(end as f64 / total as f64, stage);
    }
    ctl.check()?;
    Ok(out)
}

fn philox_output(blocks: Vec<[u32; 4]>) -> JobOutput {
    let raw: Vec<u32> = blocks.into_iter().flatten().collect();
    let uniform = raw.iter().map(|&w| u32_to_unit_f32(w)).collect();
    JobOutput::Philox { raw, uniform }
}

impl Inner {
    fn execute<E: FpEnv>(&self, job: Job, ctl: &JobControl) -> MkResult<JobResult> {
        let t0 = Instant::now();
        let mut stats = JobStats::default();
        let output = match job {
            Job::PhiloxStream {
                seed,
                stream,
                first_block,
                num_blocks,
            } => {
                let key = [seed as u32, (seed >> 32) as u32];
                let blocks = chunked(num_blocks as usize, ctl, "philox", &mut stats, |t| {
                    let b = first_block.wrapping_add(t as u64);
                    let ctr = [
                        b as u32,
                        (b >> 32) as u32,
                        stream as u32,
                        (stream >> 32) as u32,
                    ];
                    philox4x32_10(ctr, key)
                })?;
                philox_output(blocks)
            }
            Job::PhiloxBlocks { counters, keys } => {
                let blocks = chunked(counters.len(), ctl, "philox", &mut stats, |i| {
                    philox4x32_10(counters[i], keys[i])
                })?;
                philox_output(blocks)
            }
            Job::DfOps { op, a, b } => {
                JobOutput::Df(chunked(a.len(), ctl, "df", &mut stats, |i| {
                    op.apply_in::<E>(a[i], b[i])
                })?)
            }
            Job::Reduce {
                mode,
                count,
                values,
            } => {
                let channels = values.len() / count;
                let total = channels * reduce::num_groups(count);
                let mut sums = Vec::with_capacity(channels);
                let mut done = 0;
                for c in 0..channels {
                    let x = &values[c * count..(c + 1) * count];
                    let launched = |groups| {
                        stats.command_buffers += 1;
                        done += groups;
                        ctl.report(done as f64 / total as f64, "reduce");
                    };
                    sums.push(reduce::reduce_channel_ranges_in::<E>(
                        mode, x, CHUNK, ctl, launched,
                    )?);
                }
                ctl.check()?;
                JobOutput::Sums(sums)
            }
            Job::SagProbe {
                systems,
                system,
                surface,
                r2,
            } => {
                let set = self.lookup(systems)?;
                let h = set.headers.get(system as usize).ok_or_else(|| {
                    MkError::Backend("sag probe: system index out of range".into())
                })?;
                if surface >= h.num_surfaces {
                    return Err(MkError::Backend("sag probe: surface out of range".into()));
                }
                let s = set.surfaces[(h.surface_offset + surface) as usize];
                JobOutput::F32(chunked(r2.len(), ctl, "sag", &mut stats, |i| {
                    packed_sag_with_pool_in::<E>(&s, &set.coefficients, r2[i])
                })?)
            }
            Job::Trace(req) => {
                let set = self.lookup(req.systems)?;
                JobOutput::Trace(Box::new(trace_job::<E>(&req, &set, ctl, &mut stats)?))
            }
        };
        ctl.report(1.0, "");
        stats.wall_seconds = t0.elapsed().as_secs_f64();
        Ok(JobResult { output, stats })
    }
}

impl ComputeBackend for CpuF32Backend {
    fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }
    fn precision(&self) -> &PrecisionDescriptor {
        &self.inner.precision
    }
    fn supports(&self, _kind: JobKind) -> bool {
        true
    }
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        Ok(self
            .inner
            .register(Arc::new(PackedSystemSet::new(systems)?)))
    }
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.lookup(handle)?;
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id());
        Ok(())
    }
    fn fill_buffer(
        &self,
        desc: crate::resident::BufferDesc,
        fill: &mut dyn FnMut(crate::resident::BufferFill<'_>) -> MkResult<()>,
    ) -> MkResult<crate::resident::BufferHandle> {
        self.inner
            .resident
            .fill(desc, fill, crate::resident::HostAllocation::new)
    }
    fn read_view(&self, view: crate::resident::BufferView) -> MkResult<crate::resident::ReadView> {
        self.inner.resident.read_view(view)
    }
    fn resident_cache_stats(&self) -> crate::resident::ResidentCacheStats {
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
        key: &crate::resident::SystemCacheKey,
        systems: &[PackedSystem],
    ) -> MkResult<SystemHandle> {
        let hash =
            crate::resident::cache::fingerprint(key, systems, &self.precision().kernel_build)?;
        let mut cache = self.inner.cache.lock().unwrap_or_else(|e| e.into_inner());
        let up = match cache.get(key, hash) {
            Some(up) => up,
            None => {
                let up = Arc::new(PackedSystemSet::new(systems)?);
                cache.insert(key.clone(), hash, up.byte_len(), up.clone());
                up
            }
        };
        drop(cache);
        Ok(self.inner.register(up))
    }
    fn submit_plan(
        &self,
        plan: crate::resident::ResidentPlan,
        ctx: &JobContext,
    ) -> MkResult<Pending<crate::resident::PlanResult>> {
        self.submit_plan_hosted(Arc::new(self.clone()), plan, ctx, |_| {
            Ok(Box::new(crate::resident::cpu::Specification))
        })
    }
}

/// Device operations a hosted plan runs ([`CpuF32Backend::submit_plan_hosted`]).
pub type HostedDevice = Box<dyn crate::resident::cpu::DeviceOps + Send>;

impl CpuF32Backend {
    /// Run `plan` on this backend's storage and CPU-f32 executor on behalf
    /// of `backend` (whose identity the result's `RunInfo` records), with
    /// the device operations `device` creates for the pinned systems: a
    /// backend without host-visible memory (WebGPU) runs the operations it
    /// has kernels for and leaves the others to their CPU-f32
    /// specification. Validation, system pinning, cancellation and
    /// publication are this backend's.
    pub fn submit_plan_hosted(
        &self,
        backend: Arc<dyn ComputeBackend>,
        plan: crate::resident::ResidentPlan,
        ctx: &JobContext,
        device: impl FnOnce(&HashMap<SystemHandle, Arc<PackedSystemSet>>) -> MkResult<HostedDevice>
        + Send
        + 'static,
    ) -> MkResult<Pending<crate::resident::PlanResult>> {
        use crate::resident::ResidentOp;
        ctx.check()?;
        plan.validate(self.inner.instance, self.info().max_buffer_bytes)?;
        let mut pinned = HashMap::new();
        for h in plan.system_handles() {
            pinned.insert(h, self.inner.lookup(h)?);
        }
        let prepared = plan
            .operations
            .iter()
            .map(|op| match op {
                ResidentOp::TraceUpload(t) => {
                    PreparedTrace::new(t.request(), &pinned[&t.request().systems]).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<MkResult<Vec<_>>>()?;
        #[allow(clippy::too_many_arguments)]
        fn run<E: FpEnv>(
            b: &CpuF32Backend,
            backend: &dyn ComputeBackend,
            plan: &crate::resident::ResidentPlan,
            pinned: HashMap<SystemHandle, Arc<PackedSystemSet>>,
            prepared: &[Option<PreparedTrace>],
            ctl: &JobControl,
            device: &mut dyn crate::resident::cpu::DeviceOps,
        ) -> MkResult<crate::resident::PlanResult> {
            let sets = pinned.clone();
            crate::resident::cpu::execute_with::<E>(
                &b.inner.resident,
                backend,
                plan,
                pinned,
                ctl,
                |i, sub, stats| {
                    let crate::resident::ResidentOp::TraceUpload(t) = &plan.operations[i] else {
                        unreachable!()
                    };
                    let req = t.request();
                    trace_job_prepared::<E>(
                        req,
                        &sets[&req.systems],
                        prepared[i].as_ref().unwrap(),
                        sub,
                        stats,
                    )
                },
                device,
            )
        }
        let (p, c, ctl) = pending(ctx);
        let b = self.clone();
        spawn_job(
            "mklight-resident-cpu",
            "CPU resident plan".into(),
            c,
            move || {
                let mut device = device(&pinned)?;
                if b.precision().probe.subnormals {
                    run::<Ieee>(&b, &*backend, &plan, pinned, &prepared, &ctl, &mut *device)
                } else {
                    run::<Ftz>(&b, &*backend, &plan, pinned, &prepared, &ctl, &mut *device)
                }
            },
        )?;
        Ok(p)
    }

    /// The system set behind a handle this backend issued.
    pub fn system_set(&self, handle: SystemHandle) -> MkResult<Arc<PackedSystemSet>> {
        self.inner.lookup(handle)
    }

    /// This backend's instance (the owner of its handles and buffers).
    pub fn instance(&self) -> u64 {
        self.inner.instance
    }

    /// The CPU-f32 emulation of a [`Job`], synchronously on this thread.
    pub fn execute_job(&self, job: Job, ctl: &JobControl) -> MkResult<JobResult> {
        if self.inner.precision.probe.subnormals {
            self.inner.execute::<Ieee>(job, ctl)
        } else {
            self.inner.execute::<Ftz>(job, ctl)
        }
    }
}

impl QualificationBackend for CpuF32Backend {
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        job.validate()?;
        // Resolve handles now so a released handle fails at submission.
        match &job {
            Job::SagProbe { systems, .. } => {
                self.inner.lookup(*systems)?;
            }
            Job::Trace(req) => {
                PreparedTrace::new(req, &*self.inner.lookup(req.systems)?)?;
            }
            _ => {}
        }
        let (p, c, ctl) = pending(ctx);
        // The emulation is CPU-bound and parallel inside; run it on its own
        // thread so `submit` returns immediately, like a GPU queue.
        let inner = self.inner.clone();
        let what = format!("cpu-f32 {:?} job", job.kind());
        let ftz = !self.inner.precision.probe.subnormals;
        spawn_job("mklight-cpu-f32", what, c, move || {
            if ftz {
                inner.execute::<Ftz>(job, &ctl)
            } else {
                inner.execute::<Ieee>(job, &ctl)
            }
        })?;
        Ok(p)
    }
    fn upload_buffer(
        &self,
        desc: crate::resident::BufferDesc,
        data: crate::resident::BufferData,
    ) -> MkResult<crate::resident::BufferHandle> {
        self.inner
            .resident
            .upload(desc, data, crate::resident::HostAllocation::new)
    }
    fn import_buffer(
        &self,
        desc: crate::resident::BufferDesc,
        data: crate::resident::HostAllocation,
    ) -> MkResult<crate::resident::BufferHandle> {
        self.inner.resident.adopt(desc, data)
    }
    fn readback(
        &self,
        view: crate::resident::BufferView,
        ctx: &JobContext,
    ) -> MkResult<crate::resident::Readback> {
        self.inner.resident.readback(view, ctx)
    }
}

/// Run the trace kernel specification over every ray of `req`.
fn trace_job<E: FpEnv>(
    req: &TraceRequest,
    set: &PackedSystemSet,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<TraceResult> {
    let prep = PreparedTrace::new(req, set)?;
    trace_job_prepared::<E>(req, set, &prep, ctl, stats)
}
fn trace_job_prepared<E: FpEnv>(
    req: &TraceRequest,
    set: &PackedSystemSet,
    prep: &PreparedTrace,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<TraceResult> {
    let rays = &req.rays;
    let n = rays.len();
    let slots = prep.record_slots;
    let outs: Vec<(KRayOut, Vec<KState>)> = chunked(n, ctl, "trace", stats, |i| {
        let lo = |v: &Vec<f32>| v.get(i).copied().unwrap_or(0.0);
        let ray = KRayIn {
            pos: [rays.x[i], rays.y[i], rays.z[i]],
            pos_lo: [lo(&rays.x_lo), lo(&rays.y_lo), lo(&rays.z_lo)],
            dir: [rays.l[i], rays.m[i], rays.n[i]],
            dir_lo: [lo(&rays.l_lo), lo(&rays.m_lo), lo(&rays.n_lo)],
            opl: Df32::new(rays.opl_hi[i], rays.opl_lo[i]),
            weight: rays.weight.get(i).copied().unwrap_or(1.0),
            wavelength: rays.wavelength[i],
        };
        let h = &set.headers[rays.system.get(i).copied().unwrap_or(0) as usize];
        let surf = &set.surfaces[h.surface_offset as usize..][..h.num_surfaces as usize];
        let mut rec = vec![KState::default(); slots];
        let r = (slots > 0).then_some(rec.as_mut_slice());
        (
            trace_ray::<E>(
                h,
                surf,
                &set.coefficients,
                &prep.steps,
                &prep.params,
                &ray,
                r,
            ),
            rec,
        )
    })?;
    let mut res = TraceResult::default();
    for (o, _) in &outs {
        let s = &o.state;
        res.x.push(s.pos[0]);
        res.y.push(s.pos[1]);
        res.z.push(s.pos[2]);
        res.l.push(s.dir[0]);
        res.m.push(s.dir[1]);
        res.n.push(s.dir[2]);
        res.opl_hi.push(s.opl.hi);
        res.opl_lo.push(s.opl.lo);
        res.weight.push(s.weight);
        res.transmittance.push(o.transmittance);
        res.status.push(s.status);
        res.last_surface.push(o.last_surface);
        res.steps.push(o.steps);
    }
    if slots > 0 {
        let k = MKL_TRACE_RECORD_FLOATS as usize;
        res.recorded = vec![0.0; slots * n * k];
        res.recorded_status = vec![0; slots * n];
        for (i, (_, rec)) in outs.iter().enumerate() {
            for (slot, st) in rec.iter().enumerate() {
                let base = (slot * n + i) * k;
                res.recorded[base..base + k].copy_from_slice(&[
                    st.pos[0], st.pos[1], st.pos[2], st.dir[0], st.dir[1], st.dir[2], st.opl.hi,
                    st.opl.lo, st.weight,
                ]);
                res.recorded_status[slot * n + i] = st.status;
            }
        }
    }
    Ok(res)
}

/// Convenience: run a job and return its output, for tests and tools.
pub fn run_output(b: &dyn QualificationBackend, job: Job) -> MkResult<JobOutput> {
    Ok(b.run(job, &JobContext::new())?.output)
}

/// Convenience: the double-float sums of a reduction job.
pub fn reduce_sums(
    b: &dyn QualificationBackend,
    mode: ReduceMode,
    count: usize,
    values: Vec<f32>,
) -> MkResult<Vec<Df32>> {
    match run_output(
        b,
        Job::Reduce {
            mode,
            count,
            values,
        },
    )? {
        JobOutput::Sums(s) => Ok(s),
        other => Err(MkError::Backend(format!("unexpected output {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::rng::Philox;

    #[test]
    fn philox_stream_matches_core() {
        let b = CpuF32Backend::new();
        let JobOutput::Philox { raw, uniform } = run_output(
            &b,
            Job::PhiloxStream {
                seed: 0x1234_5678_9abc_def0,
                stream: 77,
                first_block: 3,
                num_blocks: 1000,
            },
        )
        .unwrap() else {
            panic!()
        };
        let mut r = Philox::new(0x1234_5678_9abc_def0, 77);
        r.seek(12); // block 3
        for (i, w) in raw.iter().enumerate() {
            assert_eq!(*w, r.next_u32(), "draw {i}");
            assert_eq!(uniform[i], u32_to_unit_f32(*w));
        }
    }

    #[test]
    fn cancellation_between_chunks() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        let r = b.run(
            Job::PhiloxStream {
                seed: 1,
                stream: 0,
                first_block: 0,
                num_blocks: 1 << 20,
            },
            &ctx,
        );
        assert_eq!(r, Err(MkError::Cancelled));
    }

    #[test]
    fn progress_reaches_one() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let r = b
            .run(
                Job::DfOps {
                    op: DfOp::Add,
                    a: vec![Df32::from_f32(1.0); 200_000],
                    b: vec![Df32::from_f32(2.0); 200_000],
                },
                &ctx,
            )
            .unwrap();
        assert_eq!(ctx.progress.fraction(), 1.0);
        assert!(r.stats.command_buffers >= 4);
        let JobOutput::Df(v) = r.output else { panic!() };
        assert!(v.iter().all(|z| *z == Df32::from_f32(3.0)));
    }

    #[test]
    fn handles_are_checked() {
        let a = CpuF32Backend::new();
        let b = CpuF32Backend::new();
        let rs = crate::testutil::singlet_resolved();
        let p = PackedSystem::pack_all(&rs).unwrap();
        let h = a.upload_system(&p).unwrap();
        let probe = |h| Job::SagProbe {
            systems: h,
            system: 0,
            surface: 1,
            r2: vec![0.0, 100.0],
        };
        assert!(b.submit(probe(h), &JobContext::new()).is_err());
        let JobOutput::F32(s) = run_output(&a, probe(h)).unwrap() else {
            panic!()
        };
        assert_eq!(s[0], 0.0);
        assert!((s[1] as f64 - rs.surfaces[1].shape.sag(100.0).unwrap()).abs() < 1e-6);
        a.release_systems(h).unwrap();
        assert!(a.submit(probe(h), &JobContext::new()).is_err());
        assert!(
            a.submit(
                Job::Trace(Box::new(TraceRequest::new(h, Default::default(), 0, 3))),
                &JobContext::new()
            )
            .is_err()
        );
    }

    /// A handle whose count was forged cannot make the worker index past the
    /// stored set: the error names the problem instead of a worker panic.
    #[test]
    fn forged_handle_count_is_rejected() {
        let b = CpuF32Backend::new();
        let p = PackedSystem::pack_all(&crate::testutil::singlet_resolved()).unwrap();
        let h = b.upload_system(&p).unwrap();
        let forged = SystemHandle::new(h.backend_instance(), h.id(), 5);
        let r = b.run(
            Job::SagProbe {
                systems: forged,
                system: 3,
                surface: 1,
                r2: vec![1.0],
            },
            &JobContext::new(),
        );
        assert_eq!(
            r,
            Err(MkError::Backend(
                "sag probe: system index out of range".into()
            ))
        );
    }

    /// Cancelling one submission does not cancel siblings sharing the context.
    #[test]
    fn pending_cancel_is_per_job() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let job = || Job::PhiloxStream {
            seed: 1,
            stream: 0,
            first_block: 0,
            num_blocks: 1 << 22,
        };
        let p1 = b.submit(job(), &ctx).unwrap();
        let p2 = b.submit(job(), &ctx).unwrap();
        p1.cancel();
        // p1 either finished first or was cancelled; p2 always completes.
        if let Err(e) = p1.wait() {
            assert_eq!(e, MkError::Cancelled);
        }
        assert!(p2.wait().is_ok());
        assert!(!ctx.cancel.is_cancelled());
    }

    #[test]
    fn empty_jobs_complete_with_full_progress() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let r = b
            .run(
                Job::DfOps {
                    op: DfOp::Add,
                    a: vec![],
                    b: vec![],
                },
                &ctx,
            )
            .unwrap();
        assert_eq!(r.output, JobOutput::Df(vec![]));
        assert_eq!(ctx.progress.fraction(), 1.0);
    }

    /// The flush-to-zero emulation reproduces the cases measured on Apple
    /// GPUs (see the Metal tests for the device comparison).
    #[test]
    fn ftz_environment() {
        let ieee = CpuF32Backend::new();
        let ftz = CpuF32Backend::with_subnormals(false);
        assert!(!ftz.precision().probe.subnormals);
        assert!(ftz.precision().probe.error_free_transforms());
        let job = || Job::DfOps {
            op: DfOp::TwoProd,
            a: vec![Df32::from_f32(1e-20)],
            b: vec![Df32::from_f32(1e-20)],
        };
        assert_eq!(
            run_output(&ftz, job()).unwrap(),
            JobOutput::Df(vec![Df32::ZERO])
        );
        let JobOutput::Df(v) = run_output(&ieee, job()).unwrap() else {
            panic!()
        };
        assert_eq!(v[0].hi, 1e-20f32 * 1e-20f32);
        let s = reduce_sums(&ftz, ReduceMode::DoubleFloat, 4096, vec![1e-39; 4096]).unwrap();
        assert_eq!(s, vec![Df32::ZERO]);
    }

    /// Chunked reduction (several pass-1 launches per channel) is
    /// bit-identical to the one-shot specification, and progress reaches 1.
    #[test]
    fn reduce_chunks_are_exact() {
        let b = CpuF32Backend::new();
        let n = 1_000_003;
        let v: Vec<f32> = (0..2 * n)
            .map(|i| ((i * 7919) % 1013) as f32 * 0.37)
            .collect();
        let ctx = JobContext::new();
        let r = b
            .run(
                Job::Reduce {
                    mode: ReduceMode::DoubleFloat,
                    count: n,
                    values: v.clone(),
                },
                &ctx,
            )
            .unwrap();
        assert!(r.stats.command_buffers > 2);
        assert_eq!(ctx.progress.fraction(), 1.0);
        let JobOutput::Sums(s) = r.output else {
            panic!()
        };
        for c in 0..2 {
            let want = reduce::reduce_channel(ReduceMode::DoubleFloat, &v[c * n..(c + 1) * n]);
            assert_eq!(s[c], want);
        }
    }

    #[test]
    fn reduce_via_backend() {
        let b = CpuF32Backend::new();
        let n = 10_000;
        let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        v.extend((0..n).map(|i| -(i as f32) * 0.5));
        let s = reduce_sums(&b, ReduceMode::DoubleFloat, n, v).unwrap();
        let want = (n * (n - 1) / 2) as f64;
        assert_eq!(s[0].to_f64(), want);
        assert_eq!(s[1].to_f64(), -want * 0.5);
    }
}
