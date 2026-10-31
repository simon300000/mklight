//! Resident Metal execution. Mutable buffers stay worker-local; immutable
//! leases and system Arcs pin resources through every committed completion.
use crate::{
    backend::Uploaded,
    buffer::SharedBuffer,
    context::{CommandStats, MetalContext, PipelineKey},
};
use mklight_core::{JobContext, MkResult, job::RunTimer};
use mklight_gpu::resident::immutable as super_cache;
use mklight_gpu::resident::{
    pool::{Allocation, Pool},
    *,
};
use mklight_gpu::{
    ComputeBackend, PreparedTrace,
    consts::*,
    layout::{ReduceParams, TraceParams},
    pending::JobControl,
    reduce,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

pub(crate) type Lease = Allocation<SharedBuffer<u32>>;
pub(crate) struct Runtime {
    owner: u64,
    cache: Arc<Mutex<super_cache::ImmutableCache>>,
    next: AtomicU64,
    pool: Arc<Mutex<Pool<SharedBuffer<u32>>>>,
}
impl Runtime {
    pub(crate) fn new(owner: u64) -> Self {
        Self {
            owner,
            cache: Arc::new(Mutex::new(super_cache::ImmutableCache::new(64 << 20))),
            next: AtomicU64::new(1),
            pool: Arc::new(Mutex::new(Pool::new(256 << 20))),
        }
    }
    pub(crate) fn operation_cache_bytes(&self) -> u64 {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes() as u64
    }

    pub(crate) fn pooled_bytes(&self) -> u64 {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).bytes() as u64
    }
    fn take(&self, ctx: &MetalContext, words: usize, stats: &mut PlanStats) -> MkResult<Lease> {
        let (b, reuse) = Pool::take(&self.pool, words, |n| ctx.buffer_filled(n, |_| {}))?;
        if reuse {
            stats.reused_allocations += 1;
        } else {
            stats.allocations += 1;
        }
        Ok(b)
    }
    fn publish(&self, desc: BufferDesc, b: Lease) -> BufferHandle {
        BufferHandle::from_lease(
            self.owner,
            self.next.fetch_add(1, Ordering::Relaxed),
            desc,
            Arc::new(b),
        )
    }
    pub(crate) fn upload(
        &self,
        ctx: &MetalContext,
        desc: BufferDesc,
        data: BufferData,
    ) -> MkResult<BufferHandle> {
        if desc.element != data.element() || desc.byte_len()? != data.words().len() * 4 {
            return Err(bad("upload type or length"));
        }
        if desc.byte_len()? as u64 > ctx.device().maxBufferLength() as u64 {
            return Err(bad("upload exceeds device limit"));
        }
        let mut b = self.take(ctx, data.words().len(), &mut PlanStats::default())?;
        b.as_mut_slice().copy_from_slice(data.words());
        Ok(self.publish(desc, b))
    }
    pub(crate) fn validate_import(&self, view: &BufferView) -> MkResult<()> {
        view.buffer.lease::<Lease>(self.owner)?;
        if view.stride != 1 {
            return mklight_gpu::resident::unsupported("strided plan import");
        }
        Ok(())
    }
    pub(crate) fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback> {
        ctx.check()?;
        view.validate()?;
        let t = Instant::now();
        let b = view.buffer.lease::<Lease>(self.owner)?;
        let width = view.desc.element.words();
        let mut words = Vec::with_capacity(view.desc.byte_len()? / 4);
        for i in 0..view.desc.len()? {
            if i % 65536 == 0 {
                ctx.check()?;
            }
            let a = (view.offset + i * view.stride) * width;
            words.extend_from_slice(&b.as_slice()[a..a + width]);
        }
        ctx.check()?;
        ctx.report(1., "readback");
        Ok(Readback {
            copied_bytes: (words.len() * 4) as u64,
            data: BufferData::from_words(view.desc.element, &words),
            wall_seconds: t.elapsed().as_secs_f64(),
        })
    }
    pub(crate) fn execute(
        &self,
        backend: &dyn ComputeBackend,
        gpu: &MetalContext,
        plan: ResidentPlan,
        pinned: HashMap<mklight_gpu::SystemHandle, Arc<Uploaded>>,
        prepared: Vec<Option<PreparedTrace>>,
        ctl: &JobControl,
        chunk: usize,
    ) -> MkResult<PlanResult> {
        let timer = RunTimer::start();
        let start = Instant::now();
        ctl.check()?;
        let mut x = Execution {
            gpu,
            runtime: self,
            buffers: vec![],
            stats: PlanStats::default(),
            live_bytes: 0,
            chunk: chunk.min(1 << 18),
            sum_scratch: None,
            systems: pinned.clone(),
            cache: self.cache.clone(),
            cache_identity: mklight_gpu::resident::SystemCacheKey {
                snapshot: plan.snapshot.hash().to_hex(),
                library: plan.identity.library_hash.clone(),
                configuration: plan.identity.configuration.clone(),
                spectrum: plan.identity.spectrum.clone(),
                layout: MKL_LAYOUT_VERSION,
                kernel_build: backend.precision().kernel_build.clone(),
                operation_settings: plan.settings.to_string(),
            },
        };
        let mut imported = std::collections::HashSet::new();
        for b in &plan.buffers {
            ctl.check()?;
            if let PlanBuffer::Import(v) = b {
                if imported.insert(v.buffer.generation()) {
                    x.live_bytes += v.buffer.desc().byte_len()?.max(16) as u64;
                    x.stats.peak_bytes = x.stats.peak_bytes.max(x.live_bytes);
                }
            }
            let slot = match b {
                PlanBuffer::Scratch(d) => Slot::Owned(x.allocate(d.byte_len()? / 4)?),
                PlanBuffer::Import(v) => Slot::Import(
                    v.buffer.lease::<Lease>(self.owner)?,
                    v.offset * v.desc.element.words(),
                ),
            };
            x.buffers.push(Some(slot));
        }
        let sum_words = plan
            .operations
            .iter()
            .filter_map(|op| match op {
                ResidentOp::Sum(s) => {
                    Some(reduce::num_groups(s.input.desc.shape[1]) * s.input.desc.shape[0] * 2)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if sum_words > 0 {
            x.sum_scratch = Some(x.allocate(sum_words)?);
        }
        for (i, op) in plan.operations.iter().enumerate() {
            ctl.check()?;
            let sub = ctl.subrange(
                i as f64 / plan.operations.len() as f64,
                (i + 1) as f64 / plan.operations.len() as f64,
            );
            let t = Instant::now();
            let mut stage = StageStats {
                operation: i,
                ..Default::default()
            };
            match op {
                ResidentOp::TraceUpload(t) => {
                    stage.name = "trace".into();
                    let up = x.system(t.request().systems)?;
                    x.trace(t, &up, prepared[i].as_ref().unwrap(), &sub, &mut stage)?;
                }
                ResidentOp::Sum(s) => {
                    stage.name = "sum".into();
                    x.sum(s, &sub, &mut stage)?;
                }
                _ => dispatch(op, &mut x, &sub)?,
            }
            ctl.check()?;
            ctl.report(
                (i + 1) as f64 / plan.operations.len() as f64,
                "resident plan",
            );
            stage.wall_seconds = t.elapsed().as_secs_f64();
            x.stats.device_seconds += stage.device_seconds;
            x.stats.command_buffers += stage.command_buffers;
            x.stats.stages.push(stage);
        }
        ctl.check()?;
        let mut outputs = vec![];
        let mut published: HashMap<BufferId, BufferHandle> = HashMap::new();
        for id in &plan.outputs {
            let h = if let Some(h) = published.get(id) {
                h.clone()
            } else {
                let h = match x.buffers[id.0].take().unwrap() {
                    Slot::Owned(b) => self.publish(plan.desc(*id)?.clone(), b),
                    _ => unreachable!(),
                };
                published.insert(*id, h.clone());
                h
            };
            outputs.push(h);
        }
        let pack = Instant::now();
        let mut run_info = stamp(&plan, backend, timer, &x.stats);
        x.stats.stages.push(StageStats {
            operation: plan.operations.len(),
            name: "metadata".into(),
            wall_seconds: pack.elapsed().as_secs_f64(),
            ..Default::default()
        });
        x.stats.wall_seconds = start.elapsed().as_secs_f64();
        run_info.settings["plan_stats"] = serde_json::to_value(&x.stats).unwrap();
        ctl.check()?;
        ctl.report(1., "resident plan");
        Ok(PlanResult {
            outputs,
            stats: x.stats,
            run_info,
        })
    }
}
enum Slot {
    Owned(Lease),
    Import(Arc<Lease>, usize),
}
impl Slot {
    fn buffer(&self) -> &SharedBuffer<u32> {
        match self {
            Self::Owned(b) => b,
            Self::Import(b, _) => b,
        }
    }
    fn offset(&self) -> usize {
        match self {
            Self::Import(_, a) => *a,
            _ => 0,
        }
    }
    fn owned(self) -> Lease {
        match self {
            Self::Owned(b) => b,
            _ => unreachable!(),
        }
    }
}
pub(crate) struct Execution<'a> {
    cache: Arc<Mutex<super_cache::ImmutableCache>>,
    cache_identity: mklight_gpu::resident::SystemCacheKey,
    pub(crate) gpu: &'a MetalContext,
    runtime: &'a Runtime,
    buffers: Vec<Option<Slot>>,
    pub(crate) stats: PlanStats,
    live_bytes: u64,
    chunk: usize,
    sum_scratch: Option<Lease>,
    systems: HashMap<mklight_gpu::SystemHandle, Arc<Uploaded>>,
}
impl Execution<'_> {
    pub(crate) fn cached<T: std::any::Any + Send + Sync>(
        &self,
        key: &super_cache::OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }
    pub(crate) fn cache<T: std::any::Any + Send + Sync>(
        &self,
        key: super_cache::OperationCacheKey,
        bytes: usize,
        value: Arc<T>,
    ) {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, bytes, value);
    }

    pub(crate) fn system(&self, handle: mklight_gpu::SystemHandle) -> MkResult<Arc<Uploaded>> {
        self.systems
            .get(&handle)
            .cloned()
            .ok_or_else(|| bad("system not declared in operation accesses"))
    }

    pub(crate) fn buffer(&self, view: &PlanView) -> (&SharedBuffer<u32>, usize) {
        let slot = self.buffers[view.buffer.0].as_ref().unwrap();
        (
            slot.buffer(),
            slot.offset() + view.offset * view.desc.element.words(),
        )
    }
    pub(crate) fn take_output(&mut self, id: BufferId) -> Lease {
        self.buffers[id.0].take().unwrap().owned()
    }
    pub(crate) fn put_output(&mut self, id: BufferId, value: Lease) {
        self.buffers[id.0] = Some(Slot::Owned(value));
    }

    pub(crate) fn allocate(&mut self, n: usize) -> MkResult<Lease> {
        let b = self.runtime.take(self.gpu, n, &mut self.stats)?;
        self.live_bytes += n.max(4) as u64 * 4;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.live_bytes);
        Ok(b)
    }
    pub(crate) fn recycle(&mut self, b: Lease) {
        self.live_bytes -= b.words().max(4) as u64 * 4;
        drop(b);
    }
    fn account(stage: &mut StageStats, cs: CommandStats, wall: f64) {
        stage.device_seconds += cs.gpu_seconds;
        stage.command_buffers += 1;
        stage.max_command_seconds = stage.max_command_seconds.max(wall);
    }
    fn trace(
        &mut self,
        t: &TraceUpload,
        up: &Uploaded,
        prep: &PreparedTrace,
        ctl: &JobControl,
        stage: &mut StageStats,
    ) -> MkResult<()> {
        let n = t.request().rays.len();
        let mut fu = self.allocate(n * MKL_TRACE_IN_UINTS as usize)?;
        let mut ff = self.allocate(n * MKL_TRACE_IN_FLOATS as usize)?;
        PreparedTrace::write_input_planes(
            &t.request().rays,
            bytemuck::cast_slice_mut(ff.as_mut_slice()),
            fu.as_mut_slice(),
        );
        let key = super_cache::OperationCacheKey {
            system: self.cache_identity.clone(),
            operation: format!("trace_steps:{:?}", prep.steps),
            plan_version: PLAN_VERSION,
        };
        let steps = if let Some(steps) = self.cached::<SharedBuffer<u32>>(&key)? {
            steps
        } else {
            let steps = Arc::new(self.gpu.buffer(&prep.steps)?);
            self.stats.allocations += 1;
            self.stats.uploaded_bytes += (prep.steps.len() * 4) as u64;
            self.cache(key, (prep.steps.len() * 4).max(16), steps.clone());
            steps
        };
        self.stats.uploaded_bytes += ((fu.len() + ff.len()) * 4) as u64;
        let mut of = self.take_output(t.output.floats);
        let mut ou = self.take_output(t.output.indices);
        let mut rf = self.take_output(t.output.records);
        let mut ru = self.take_output(t.output.record_status);
        let kernel = if up.sag_overlays {
            "mkl_trace_overlays"
        } else {
            "mkl_trace"
        };
        let pso = self.gpu.pipeline(&PipelineKey::plain(kernel))?;
        for start in (0..n).step_by(self.chunk) {
            ctl.check()?;
            let len = self.chunk.min(n - start);
            let p = TraceParams {
                num_rays: len as u32,
                stride: n as u32,
                ..prep.params
            };
            self.stats.parameter_bytes += size_of::<TraceParams>() as u64;
            let wall = Instant::now();
            // SAFETY: PreparedTrace validated every system, wavelength and path
            // step. Plan validation fixed the output plane/record extents.
            // All ray pointers are advanced by start, preserving stride n.
            let cs = unsafe {
                self.gpu.run(kernel, |e| {
                    e.set_buffer(0, &up.headers, 0);
                    e.set_buffer(1, &up.surfaces, 0);
                    e.set_buffer(2, &steps, 0);
                    e.set_bytes(3, &p);
                    e.set_buffer(4, &ff, start);
                    e.set_buffer(5, &fu, start);
                    e.set_buffer_mut(6, &mut of, start);
                    e.set_buffer_mut(7, &mut ou, start);
                    e.set_buffer_mut(
                        8,
                        &mut rf,
                        if prep.record_slots > 0 {
                            start * MKL_TRACE_RECORD_FLOATS as usize
                        } else {
                            0
                        },
                    );
                    e.set_buffer_mut(9, &mut ru, if prep.record_slots > 0 { start } else { 0 });
                    e.set_buffer(10, &up.coefficients, 0);
                    e.dispatch_1d(&pso, len);
                    Ok(())
                })?
            };
            Self::account(stage, cs, wall.elapsed().as_secs_f64());
            ctl.check()?;
            ctl.report((start + len) as f64 / n as f64, "resident trace");
        }
        for (id, b) in [
            (t.output.floats, of),
            (t.output.indices, ou),
            (t.output.records, rf),
            (t.output.record_status, ru),
        ] {
            self.put_output(id, b);
        }
        self.recycle(ff);
        self.recycle(fu);
        Ok(())
    }
    fn sum(&mut self, s: &Sum, ctl: &JobControl, stage: &mut StageStats) -> MkResult<()> {
        let count = s.input.desc.shape[1];
        let channels = s.input.desc.shape[0];
        let groups = reduce::num_groups(count);
        let threads = reduce::THREADS;
        let p1 = self
            .gpu
            .pipeline(&PipelineKey::with_u32("mkl_reduce_pass1", 1, s.mode as u32))?;
        let p2 = self
            .gpu
            .pipeline(&PipelineKey::with_u32("mkl_reduce_pass2", 1, s.mode as u32))?;
        if p1.maxTotalThreadsPerThreadgroup() < threads
            || p2.maxTotalThreadsPerThreadgroup() < threads
        {
            return unsupported("reduction threadgroup size");
        }
        let mut partials = self.sum_scratch.take().unwrap();
        let mut out = self.take_output(s.output);
        let (input, input_offset) = self.buffer(&s.input);
        // Chunk boundaries partition the fixed reduction tree, never re-form it.
        let group_step = (self.chunk / count.div_ceil(groups)).clamp(1, groups);
        let total = channels * (groups + 1);
        let mut done = 0;
        let mut launches = vec![];
        if group_step == groups {
            let ch_step = (self.chunk / count).max(1);
            for c in (0..channels).step_by(ch_step) {
                launches.push((c, ch_step.min(channels - c), 0, groups));
            }
        } else {
            for c in 0..channels {
                for g in (0..groups).step_by(group_step) {
                    launches.push((c, 1, g, group_step.min(groups - g)));
                }
            }
        }
        for (c, nc, g, ng) in launches {
            ctl.check()?;
            let p = ReduceParams {
                count: count as u32,
                channel_stride: count as u32,
                num_groups: groups as u32,
                first_group: g as u32,
            };
            let wall = Instant::now();
            // SAFETY: nc channels of count f32 values and ng fixed-tree groups
            // lie within the validated input and preallocated partial planes.
            let cs = unsafe {
                self.gpu.run("resident sum pass1", |e| {
                    e.set_pipeline(&p1);
                    e.set_buffer(0, input, input_offset + c * count);
                    e.set_buffer_mut(1, &mut partials, c * groups * 2);
                    e.set_bytes(2, &p);
                    e.dispatch_groups([ng, nc, 1], threads);
                    Ok(())
                })?
            };
            Self::account(stage, cs, wall.elapsed().as_secs_f64());
            ctl.check()?;
            done += nc * ng;
            ctl.report(done as f64 / total as f64, "resident sum");
        }
        let ch_step = (self.chunk / (groups * threads)).max(1);
        for c in (0..channels).step_by(ch_step) {
            ctl.check()?;
            let nc = ch_step.min(channels - c);
            let p = ReduceParams {
                count: count as u32,
                channel_stride: count as u32,
                num_groups: groups as u32,
                first_group: 0,
            };
            let wall = Instant::now();
            // SAFETY: nc channels each read groups float2s and write one float2.
            let cs = unsafe {
                self.gpu.run("resident sum pass2", |e| {
                    e.set_pipeline(&p2);
                    e.set_buffer(0, &partials, c * groups * 2);
                    e.set_buffer_mut(1, &mut out, c * 2);
                    e.set_bytes(2, &p);
                    e.dispatch_groups([1, nc, 1], threads);
                    Ok(())
                })?
            };
            Self::account(stage, cs, wall.elapsed().as_secs_f64());
            ctl.check()?;
            done += nc;
            ctl.report(done as f64 / total as f64, "resident sum");
        }
        self.stats.parameter_bytes +=
            stage.command_buffers as u64 * size_of::<ReduceParams>() as u64;
        self.put_output(s.output, out);
        self.sum_scratch = Some(partials);
        Ok(())
    }
}
fn dispatch(op: &ResidentOp, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    match op {
        ResidentOp::Aiming(r) => crate::aiming::execute(r, x, ctl),
        ResidentOp::Transforms(r) => crate::transforms::execute(r, x, ctl),
        ResidentOp::Collectives(r) => crate::collectives::execute(r, x, ctl),
        ResidentOp::Systems(r) => crate::systems::execute(r, x, ctl),
        ResidentOp::Geometry(r) => crate::geometry::execute(r, x, ctl),
        ResidentOp::Radiometry(r) => crate::radiometry::execute(r, x, ctl),
        ResidentOp::Wave(r) => crate::wave_ops::execute(r, x, ctl),
        ResidentOp::Ghost(r) => crate::ghost_ops::execute(r, x, ctl),
        ResidentOp::Stray(r) => crate::stray_ops::execute(r, x, ctl),
        ResidentOp::Optim(r) => crate::optim_ops::execute(r, x, ctl),
        _ => Ok(()),
    }
}
