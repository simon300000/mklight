//! CPU-f32 specification for resident ordered plans; no CPU speed path.
use super::immutable::OperationCacheKey;
use super::pool::Allocation;
use super::runtime::{OperationCache, Runtime};
use super::*;
use crate::{
    FpEnv, JobStats, PackedSystemSet, SystemHandle, TraceResult, pending::JobControl, reduce,
};
use mklight_core::job::RunTimer;
use std::{collections::HashMap, time::Instant};

type Lease = Allocation<Vec<u32>>;

pub fn execute<E: FpEnv>(
    rt: &Runtime<Vec<u32>>,
    backend: &dyn ComputeBackend,
    plan: &ResidentPlan,
    systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
    ctl: &JobControl,
    mut trace: impl FnMut(usize, &JobControl, &mut JobStats) -> MkResult<TraceResult>,
) -> MkResult<PlanResult> {
    let timer = RunTimer::start();
    let start = Instant::now();
    let mut x = Execution {
        buffers: vec![],
        systems,
        cache: rt.operation_cache(plan, &backend.precision().kernel_build),
        stats: PlanStats::default(),
    };
    for b in &plan.buffers {
        let slot = match b {
            PlanBuffer::Scratch(d) => {
                Slot::Owned(rt.take(d.byte_len()? / 4, &mut x.stats, |n| Ok(vec![0; n]))?)
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
        let name = match op {
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
                let values: &[f32] = bytemuck::cast_slice(x.words(s.input.buffer));
                let groups = reduce::num_groups(count);
                for c in 0..channels {
                    let a = s.input.offset + c * count;
                    let mut partials = Vec::with_capacity(groups);
                    for g in 0..groups {
                        ctl.check()?;
                        partials.extend(reduce::pass1_in::<E>(
                            s.mode,
                            &values[a..a + count],
                            g..g + 1,
                        ));
                    }
                    let sum = reduce::pass2_in::<E>(s.mode, &partials);
                    out.words_mut()[c * 2..c * 2 + 2].copy_from_slice(bytemuck::cast_slice(&[sum]));
                    sub.report((c + 1) as f64 / channels as f64, "sum");
                }
                x.buffers[s.output.0] = Some(out);
                js.command_buffers = 2;
                "sum"
            }
            _ => {
                dispatch(op, &mut x, &sub)?;
                "domain"
            }
        };
        ctl.check()?;
        ctl.report((i + 1) as f64 / ops, "resident plan");
        x.stats.command_buffers += js.command_buffers;
        x.stats.stages.push(StageStats {
            operation: i,
            name: name.into(),
            wall_seconds: t.elapsed().as_secs_f64(),
            command_buffers: js.command_buffers,
            ..Default::default()
        });
    }
    let stats = std::mem::take(&mut x.stats);
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
/// Execution state handed to CPU package handlers.
pub struct Execution {
    cache: OperationCache,
    systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
    buffers: Vec<Option<Slot>>,
    pub stats: PlanStats,
}
impl Execution {
    /// Operation-cache key including the plan's full system/build/settings identity.
    pub fn cache_key(&self, operation: impl Into<String>) -> OperationCacheKey {
        self.cache.key(operation)
    }
    pub fn cached<T: Any + Send + Sync>(
        &self,
        key: &OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        self.cache.get(key)
    }
    pub fn cache<T: Any + Send + Sync>(&self, key: OperationCacheKey, bytes: usize, value: Arc<T>) {
        self.cache.insert(key, bytes, value)
    }
    /// A system pinned at submission (declared in the operation's accesses).
    pub fn system(&self, handle: SystemHandle) -> MkResult<Arc<PackedSystemSet>> {
        self.systems
            .get(&handle)
            .cloned()
            .ok_or_else(|| bad("system not declared in operation accesses"))
    }
    pub fn words(&self, id: BufferId) -> &[u32] {
        self.buffers[id.0].as_ref().unwrap().words()
    }
    pub fn words_mut(&mut self, id: BufferId) -> &mut [u32] {
        self.buffers[id.0].as_mut().unwrap().words_mut()
    }
}
fn dispatch(op: &ResidentOp, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
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
        ResidentOp::TraceUpload(_) | ResidentOp::Sum(_) => unreachable!(),
    }
}
