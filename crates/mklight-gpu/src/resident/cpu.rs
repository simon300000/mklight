//! CPU-f32 specification for resident ordered plans; no CPU speed path.
use super::immutable::{BuildCost, CacheCategory, CacheStats, OperationCacheKey};
use super::pool::Allocation;
use super::runtime::{OperationCache, Runtime};
use super::*;
use crate::{
    FpEnv, JobStats, PackedSystemSet, SystemHandle, TraceResult, pending::JobControl, reduce,
};
use mklight_core::job::RunTimer;
use mklight_core::platform::Instant;
use std::{any::Any, cell::RefCell, collections::HashMap};

type Lease = Allocation<HostAllocation>;

/// A device that runs some operations of a plan in place of their CPU-f32
/// specification, on the same host storage (`HostAllocation` buffers): a
/// backend whose memory is not the host's, such as WebGPU, uploads an
/// operation's inputs from and reads its outputs back into the
/// [`Execution`]'s buffers.
pub trait DeviceOps {
    /// Run operation `i` on the device and return its stage name, or
    /// `Ok(None)` to run its CPU-f32 specification.
    fn run(
        &mut self,
        i: usize,
        op: &ResidentOp,
        x: &mut Execution,
        ctl: &JobControl,
        stats: &mut JobStats,
    ) -> MkResult<Option<&'static str>>;

    /// Called once every operation has run, before the plan's outputs are
    /// published: a device that kept results in its own memory writes the
    /// plan's final outputs (`plan.outputs`) into their host buffers.
    fn finish(
        &mut self,
        _plan: &ResidentPlan,
        _x: &mut Execution,
        _ctl: &JobControl,
    ) -> MkResult<()> {
        Ok(())
    }
}

/// No device: every operation runs its specification.
pub struct Specification;
impl DeviceOps for Specification {
    fn run(
        &mut self,
        _: usize,
        _: &ResidentOp,
        _: &mut Execution,
        _: &JobControl,
        _: &mut JobStats,
    ) -> MkResult<Option<&'static str>> {
        Ok(None)
    }
}

pub fn execute<E: FpEnv>(
    rt: &Runtime<HostAllocation>,
    backend: &dyn ComputeBackend,
    plan: &ResidentPlan,
    systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
    ctl: &JobControl,
    trace: impl FnMut(usize, &JobControl, &mut JobStats) -> MkResult<TraceResult>,
) -> MkResult<PlanResult> {
    execute_with::<E>(rt, backend, plan, systems, ctl, trace, &mut Specification)
}

/// [`execute`] with `device` running the operations it claims; `trace`
/// computes the trace uploads it leaves to the host.
pub fn execute_with<E: FpEnv>(
    rt: &Runtime<HostAllocation>,
    backend: &dyn ComputeBackend,
    plan: &ResidentPlan,
    systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
    ctl: &JobControl,
    mut trace: impl FnMut(usize, &JobControl, &mut JobStats) -> MkResult<TraceResult>,
    device: &mut dyn DeviceOps,
) -> MkResult<PlanResult> {
    let timer = RunTimer::start();
    let start = Instant::now();
    let mut x = Execution {
        buffers: vec![],
        systems,
        cache: rt.operation_cache(plan, &backend.precision().kernel_build),
        pending: RefCell::new(vec![]),
        cache_stats: RefCell::new(CacheStats::new()),
        stats: PlanStats::default(),
        flush_to_zero: E::FTZ,
    };
    for b in &plan.buffers {
        let slot = match b {
            PlanBuffer::Scratch(d) => {
                Slot::Owned(rt.take(d.byte_len()? / 4, &mut x.stats, HostAllocation::new)?)
            }
            PlanBuffer::Import(v) => {
                Slot::Import(rt.lease(&v.buffer)?, v.offset * v.desc.element.words())
            }
        };
        x.stats.peak_bytes += b.desc().byte_len()? as u64;
        x.buffers.push(Some(slot));
    }
    let ops = plan.operations.len() as f64;
    for (i, op) in plan.operations.iter().enumerate() {
        ctl.check()?;
        let sub = ctl.subrange(i as f64 / ops, (i + 1) as f64 / ops);
        let t = Instant::now();
        let mut js = JobStats::default();
        let name = match device.run(i, op, &mut x, &sub, &mut js)? {
            Some(name) => name,
            None => match op {
                ResidentOp::TraceUpload(t) => {
                    let r = trace(i, &sub, &mut js)?;
                    let n = r.len();
                    let planes: [&[f32]; 10] = [
                        &r.x,
                        &r.y,
                        &r.z,
                        &r.l,
                        &r.m,
                        &r.n,
                        &r.opl_hi,
                        &r.opl_lo,
                        &r.weight,
                        &r.transmittance,
                    ];
                    let out = x.words_mut(t.output.floats);
                    for (k, p) in planes.iter().enumerate() {
                        out[k * n..(k + 1) * n].copy_from_slice(bytemuck::cast_slice(p));
                    }
                    let out = x.words_mut(t.output.indices);
                    for (k, p) in [&r.status, &r.last_surface, &r.steps].iter().enumerate() {
                        out[k * n..(k + 1) * n].copy_from_slice(p);
                    }
                    x.words_mut(t.output.records)
                        .copy_from_slice(bytemuck::cast_slice(&r.recorded));
                    x.words_mut(t.output.record_status)
                        .copy_from_slice(&r.recorded_status);
                    x.stats.uploaded_bytes += (n
                        * (crate::consts::MKL_TRACE_IN_FLOATS + crate::consts::MKL_TRACE_IN_UINTS)
                            as usize
                        * 4) as u64;
                    "trace"
                }
                ResidentOp::Sum(s) => {
                    let [channels, count] = s.input.desc.shape[..] else {
                        unreachable!("validated sum shape")
                    };
                    let mut out = x.buffers[s.output.0].take().unwrap();
                    let values: &[f32] = x.rows(&s.input);
                    for c in 0..channels {
                        let sum = reduce::reduce_channel_ranges_in::<E>(
                            s.mode,
                            &values[c * count..(c + 1) * count],
                            crate::cpu::CHUNK,
                            ctl,
                            |_| js.command_buffers += 1,
                        )?;
                        out.words_mut()[c * 2..c * 2 + 2]
                            .copy_from_slice(bytemuck::cast_slice(&[sum]));
                        sub.report((c + 1) as f64 / channels as f64, "sum");
                    }
                    x.buffers[s.output.0] = Some(out);
                    "sum"
                }
                _ => {
                    dispatch(op, &mut x, &sub)?;
                    "domain"
                }
            },
        };
        ctl.check()?;
        ctl.report((i + 1) as f64 / ops, "resident plan");
        x.stats.command_buffers += js.command_buffers;
        x.stats.device_seconds += js.device_seconds;
        x.stats.stages.push(StageStats {
            operation: i,
            name: name.into(),
            wall_seconds: t.elapsed().as_secs_f64(),
            device_seconds: js.device_seconds,
            command_buffers: js.command_buffers,
            ..Default::default()
        });
    }
    device.finish(plan, &mut x, ctl)?;
    // Entries publish only once every operation completed uncancelled, as on
    // a device; a cancelled build never reaches the shared cache.
    ctl.check()?;
    for p in x.pending.take() {
        x.cache.insert(
            p.key,
            p.bytes,
            BuildCost::Seconds(p.cost),
            p.value,
            x.cache_stats.get_mut(),
        );
    }
    let mut stats = std::mem::take(&mut x.stats);
    stats.cache = x.cache_stats.take();
    rt.finish(plan, backend, timer, start, stats, ctl, |id| {
        match x.buffers[id.0].take().unwrap() {
            Slot::Owned(b) => b,
            Slot::Import(..) => unreachable!("validated plan output"),
        }
    })
}
enum Slot {
    Owned(Lease),
    Import(Arc<Lease>, usize),
}
impl Slot {
    fn words(&self) -> &[u32] {
        match self {
            Self::Owned(b) => b,
            Self::Import(b, a) => &b[*a..],
        }
    }
    fn words_mut(&mut self) -> &mut [u32] {
        match self {
            Self::Owned(b) => b,
            Self::Import(..) => unreachable!("validated destination"),
        }
    }
}
struct Pending {
    key: OperationCacheKey,
    bytes: usize,
    cost: f64,
    value: Arc<dyn Any + Send + Sync>,
}
/// Execution state handed to CPU package handlers.
pub struct Execution {
    cache: OperationCache,
    pending: RefCell<Vec<Pending>>,
    cache_stats: RefCell<CacheStats>,
    systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
    buffers: Vec<Option<Slot>>,
    pub stats: PlanStats,
    /// The emulated f32 environment: handlers run `execute_in::<Ftz>` when
    /// set, `execute_in::<Ieee>` otherwise.
    pub flush_to_zero: bool,
}
impl Execution {
    /// Operation-cache key including the plan's full system/build/settings identity.
    pub fn cache_key(
        &self,
        category: CacheCategory,
        operation: impl Into<String>,
    ) -> OperationCacheKey {
        self.cache.key(category, operation)
    }
    /// Content-addressed key; `operation` names digests of all its inputs.
    pub fn content_key(
        &self,
        category: CacheCategory,
        operation: impl Into<String>,
    ) -> OperationCacheKey {
        self.cache.content_key(category, operation)
    }
    pub fn cached<T: Any + Send + Sync>(
        &self,
        key: &OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        if let Some(p) = self.pending.borrow().iter().rev().find(|p| &p.key == key) {
            return p
                .value
                .clone()
                .downcast::<T>()
                .map(Some)
                .map_err(|_| bad("operation cache type collision"));
        }
        self.cache.get(key, &mut self.cache_stats.borrow_mut())
    }
    /// Publish `value` when the plan completes; same-plan consumers see it now.
    pub fn cache<T: Any + Send + Sync>(
        &self,
        key: OperationCacheKey,
        bytes: usize,
        cost: f64,
        value: Arc<T>,
    ) {
        self.pending.borrow_mut().push(Pending {
            key,
            bytes,
            cost,
            value,
        });
    }
    /// A system pinned at submission (declared in the operation's accesses).
    pub fn system(&self, handle: SystemHandle) -> MkResult<Arc<PackedSystemSet>> {
        self.systems
            .get(&handle)
            .cloned()
            .ok_or_else(|| bad("system not declared in operation accesses"))
    }
    /// Run `f` with the plan-owned outputs `ids` writable while every other
    /// buffer stays readable through `&Execution` (a device reads an
    /// operation's inputs and writes its outputs in one call). The ids must
    /// be distinct destinations of the operation.
    pub fn with_outputs<R>(
        &mut self,
        ids: &[BufferId],
        f: impl FnOnce(&Execution, &mut [&mut [u32]]) -> R,
    ) -> R {
        let mut taken: Vec<Slot> = ids
            .iter()
            .map(|id| {
                self.buffers[id.0]
                    .take()
                    .expect("distinct operation outputs")
            })
            .collect();
        let r = {
            let mut words: Vec<&mut [u32]> = taken.iter_mut().map(|s| s.words_mut()).collect();
            f(self, &mut words)
        };
        for (id, s) in ids.iter().zip(taken) {
            self.buffers[id.0] = Some(s);
        }
        r
    }
    /// Whether `id` is an imported (read-only, published) buffer rather
    /// than plan-owned scratch.
    pub fn is_import(&self, id: BufferId) -> bool {
        matches!(self.buffers[id.0], Some(Slot::Import(..)))
    }
    pub fn words(&self, id: BufferId) -> &[u32] {
        self.buffers[id.0].as_ref().unwrap().words()
    }
    pub fn words_mut(&mut self, id: BufferId) -> &mut [u32] {
        self.buffers[id.0].as_mut().unwrap().words_mut()
    }
    /// The words of a validated input view.
    pub fn view(&self, v: &PlanView) -> &[u32] {
        let w = v.desc.element.words();
        &self.words(v.buffer)[v.offset * w..(v.offset + v.desc.len().unwrap()) * w]
    }
    /// A validated input view as its elements, or as the records of a
    /// [`check_records`] view.
    pub fn rows<T: bytemuck::Pod>(&self, v: &PlanView) -> &[T] {
        bytemuck::cast_slice(self.view(v))
    }
}
fn dispatch(op: &ResidentOp, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    match op {
        ResidentOp::Aiming(r) => crate::aiming::execute(r, x, ctl),
        ResidentOp::Transforms(r) => crate::transforms::execute(r, x, ctl),
        ResidentOp::Collectives(r) => crate::collectives::execute(r, x, ctl),
        ResidentOp::Geometry(r) => crate::geometry::execute(r, x, ctl),
        ResidentOp::Radiometry(r) => crate::radiometry::execute(r, x, ctl),
        ResidentOp::Wave(r) => crate::wave_ops::execute(r, x, ctl),
        ResidentOp::Ghost(r) => crate::ghost_ops::execute(r, x, ctl),
        ResidentOp::Stray(r) => crate::stray_ops::execute(r, x, ctl),
        ResidentOp::Optim(r) => crate::optim_ops::execute(r, x, ctl),
        ResidentOp::TraceUpload(_) | ResidentOp::Sum(_) => unreachable!(),
    }
}
