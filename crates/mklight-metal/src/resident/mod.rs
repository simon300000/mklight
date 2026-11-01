//! Resident Metal execution. Mutable buffers stay worker-local; immutable
//! leases and system Arcs pin resources through every committed completion
//! and no host view or cache publication precedes device completion.
use crate::{
    backend::Uploaded,
    buffer::SharedBuffer,
    context::{CommandBatch, CommandStats, Encoder, MetalContext, PipelineKey},
};
use mklight_core::{MkResult, job::RunTimer};
use mklight_gpu::resident::{
    immutable::OperationCacheKey,
    pool::Allocation,
    runtime::{self, OperationCache},
    *,
};
use mklight_gpu::{
    ComputeBackend, PreparedTrace, SystemHandle,
    consts::*,
    layout::{ReduceParams, TraceParams},
    pending::JobControl,
    reduce,
};
use objc2_metal::MTLComputePipelineState;
use std::{
    any::Any,
    cell::RefCell,
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

pub(crate) type Lease = Allocation<SharedBuffer<u32>>;
pub(crate) type Runtime = runtime::Runtime<SharedBuffer<u32>>;
impl runtime::Storage for SharedBuffer<u32> {
    fn available(&self) -> bool {
        self.available()
    }
    fn words(&self) -> &[u32] {
        self.as_slice()
    }
    fn words_mut(&mut self) -> &mut [u32] {
        self.as_mut_slice()
    }
}
/// Resident trace commands are capped for cancellation latency.
const MAX_TRACE_CHUNK: usize = 1 << 18;

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute(
    rt: &Runtime,
    backend: &dyn ComputeBackend,
    gpu: &MetalContext,
    plan: &ResidentPlan,
    systems: HashMap<SystemHandle, Arc<Uploaded>>,
    prepared: &[Option<PreparedTrace>],
    ctl: &JobControl,
    chunk: usize,
) -> MkResult<PlanResult> {
    let copied_before = crate::buffer::copied_bytes();
    let timer = RunTimer::start();
    let start = Instant::now();
    let mut x = Execution {
        gpu,
        rt,
        buffers: vec![],
        stats: PlanStats::default(),
        live_bytes: 0,
        chunk,
        sum_scratch: None,
        systems,
        commands: RefCell::new(CommandBatch::new(plan.command_work_budget)),
        retired: vec![],
        checks: vec![],
        cache_pending: RefCell::new(vec![]),
        cache: rt.operation_cache(plan, &backend.precision().kernel_build),
    };
    let mut imported = HashSet::new();
    for b in &plan.buffers {
        let slot = match b {
            PlanBuffer::Scratch(d) => Slot::Owned(x.allocate(d.byte_len()? / 4)?),
            PlanBuffer::Import(v) => {
                if imported.insert(v.buffer.generation()) {
                    x.live_bytes += v.buffer.desc().byte_len()?.max(16) as u64;
                    x.stats.peak_bytes = x.stats.peak_bytes.max(x.live_bytes);
                }
                Slot::Import(rt.lease(&v.buffer)?, v.offset * v.desc.element.words())
            }
        };
        x.buffers.push(Some(slot));
    }
    // One partial plane, sized for the largest sum, serves every sum in order.
    let sum_words = plan
        .operations
        .iter()
        .filter_map(|op| match op {
            ResidentOp::Sum(s) => {
                Some(reduce::num_groups(s.input.desc.shape[1]) * s.input.desc.shape[0] * 2)
            }
            _ => None,
        })
        .max();
    if let Some(n) = sum_words {
        x.sum_scratch = Some(x.allocate(n)?);
    }
    let ops = plan.operations.len() as f64;
    for (i, op) in plan.operations.iter().enumerate() {
        ctl.check()?;
        let sub = ctl.subrange(0.99 * i as f64 / ops, 0.99 * (i + 1) as f64 / ops);
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
            _ => {
                stage.name = "domain".into();
                dispatch(op, &mut x, &sub)?;
            }
        }
        ctl.check()?;
        ctl.report(0.99 * (i + 1) as f64 / ops, "resident plan");
        stage.wall_seconds = t.elapsed().as_secs_f64();
        x.stats.device_seconds += stage.device_seconds;
        x.stats.command_buffers += stage.command_buffers;
        x.stats.stages.push(stage);
    }
    ctl.check()?;
    x.flush()?;
    ctl.check()?;
    let completed = x.commands.borrow().completed;
    x.stats.command_buffers += completed.command_buffers;
    x.stats.device_seconds += completed.gpu_seconds;
    x.stats.stages.push(StageStats {
        name: "fused completion".into(),
        command_buffers: completed.command_buffers,
        device_seconds: completed.gpu_seconds,
        max_command_seconds: x.commands.borrow().max_seconds,
        ..Default::default()
    });
    x.stats.copied_bytes = crate::buffer::copied_bytes() - copied_before;
    let stats = std::mem::take(&mut x.stats);
    rt.finish(plan, backend, timer, start, stats, ctl, |id| {
        x.take_output(id)
    })
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
/// Execution state handed to Metal package encoders.
pub(crate) struct Execution<'a> {
    pub(crate) gpu: &'a MetalContext,
    rt: &'a Runtime,
    cache: OperationCache,
    buffers: Vec<Option<Slot>>,
    pub(crate) stats: PlanStats,
    live_bytes: u64,
    chunk: usize,
    sum_scratch: Option<Lease>,
    systems: HashMap<SystemHandle, Arc<Uploaded>>,
    commands: RefCell<CommandBatch>,
    retired: Vec<Lease>,
    checks: Vec<(
        Lease,
        fn(&SharedBuffer<u32>, &mut PlanStats) -> MkResult<()>,
    )>,
    cache_pending: RefCell<Vec<CachePublication>>,
}
struct CachePublication {
    key: OperationCacheKey,
    value: Arc<dyn Any + Send + Sync>,
    publish: Box<dyn FnOnce(&OperationCache)>,
}
impl Execution<'_> {
    /// Deferred resident encoding. No host storage access until `flush`.
    /// Safety: the kernel bounds/access contract is identical to MetalContext::run.
    pub(crate) unsafe fn run(
        &self,
        label: &str,
        encode: impl FnOnce(&Encoder<'_>) -> MkResult<()>,
    ) -> MkResult<CommandStats> {
        // Stats are collected once at plan completion, rather than at encoding.
        unsafe {
            self.commands.borrow_mut().encode(self.gpu, label, encode)?;
        }
        Ok(CommandStats::default())
    }
    pub(crate) fn flush(&mut self) -> MkResult<CommandStats> {
        let stats = self.commands.borrow_mut().flush()?;
        for (buffer, check) in std::mem::take(&mut self.checks) {
            check(&buffer, &mut self.stats)?;
            self.live_bytes -= buffer.words().max(4) as u64 * 4;
        }
        for entry in self.cache_pending.borrow_mut().drain(..) {
            (entry.publish)(&self.cache);
        }
        self.live_bytes -= self
            .retired
            .iter()
            .map(|b| b.words().max(4) as u64 * 4)
            .sum::<u64>();
        self.retired.clear();
        Ok(stats)
    }
    pub(crate) fn check_buffer(
        &mut self,
        buffer: Lease,
        check: fn(&SharedBuffer<u32>, &mut PlanStats) -> MkResult<()>,
    ) {
        self.checks.push((buffer, check));
    }
    pub(crate) fn count_fallbacks(&mut self, counter: Lease) {
        self.stats.readback_bytes += 4;
        self.check_buffer(counter, |buffer, stats| {
            stats.fallback_count += buffer.as_slice()[0] as u64;
            Ok(())
        });
    }

    /// Operation-cache key including the plan's full system/build/settings identity.
    pub(crate) fn cache_key(&self, operation: impl Into<String>) -> OperationCacheKey {
        self.cache.key(operation)
    }
    pub(crate) fn cached<T: Any + Send + Sync>(
        &self,
        key: &OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        if let Some(entry) = self
            .cache_pending
            .borrow()
            .iter()
            .rev()
            .find(|e| &e.key == key)
        {
            return entry
                .value
                .clone()
                .downcast::<T>()
                .map(Some)
                .map_err(|_| bad("operation cache type collision"));
        }
        self.cache.get(key)
    }
    pub(crate) fn cache<T: Any + Send + Sync>(
        &self,
        key: OperationCacheKey,
        bytes: usize,
        value: Arc<T>,
    ) {
        let (k, v) = (key.clone(), value.clone());
        self.cache_pending.borrow_mut().push(CachePublication {
            key,
            value,
            publish: Box::new(move |cache| cache.insert(k, bytes, v)),
        });
    }
    /// A system pinned at submission (declared in the operation's accesses).
    pub(crate) fn system(&self, handle: SystemHandle) -> MkResult<Arc<Uploaded>> {
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
        let gpu = self.gpu;
        let b = self
            .rt
            .take(n, &mut self.stats, |n| gpu.buffer_filled(n, |_| {}))?;
        self.live_bytes += n.max(4) as u64 * 4;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.live_bytes);
        Ok(b)
    }
    pub(crate) fn recycle(&mut self, b: Lease) {
        self.retired.push(b);
    }
    fn account(stage: &mut StageStats, cs: CommandStats, wall: f64) {
        stage.device_seconds += cs.gpu_seconds;
        stage.command_buffers += cs.command_buffers;
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
        let key = self.cache_key(format!("trace_steps:{:?}", prep.steps));
        let steps = if let Some(steps) = self.cached::<SharedBuffer<u32>>(&key)? {
            steps
        } else {
            // A few words of cached metadata, like a parameter block.
            let steps = Arc::new(
                self.gpu
                    .buffer_filled(prep.steps.len(), |out| out.copy_from_slice(&prep.steps))?,
            );
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
        let chunk = self.chunk.min(MAX_TRACE_CHUNK);
        for start in (0..n).step_by(chunk) {
            ctl.check()?;
            let len = chunk.min(n - start);
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
                self.run(kernel, |e| {
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
        let [channels, count] = s.input.desc.shape[..] else {
            unreachable!("validated sum shape")
        };
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
        let mut parameter_blocks = 0;
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
            parameter_blocks += 1;
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
                self.run("resident sum pass1", |e| {
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
            parameter_blocks += 1;
            let p = ReduceParams {
                count: count as u32,
                channel_stride: count as u32,
                num_groups: groups as u32,
                first_group: 0,
            };
            let wall = Instant::now();
            // SAFETY: nc channels each read groups float2s and write one float2.
            let cs = unsafe {
                self.run("resident sum pass2", |e| {
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
        self.stats.parameter_bytes += parameter_blocks * size_of::<ReduceParams>() as u64;
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
        ResidentOp::Stray(r) => {
            x.flush()?;
            crate::stray_ops::execute(r, x, ctl)
        }
        ResidentOp::Optim(r) => crate::optim_ops::execute(r, x, ctl),
        ResidentOp::TraceUpload(_) | ResidentOp::Sum(_) => unreachable!(),
    }
}
