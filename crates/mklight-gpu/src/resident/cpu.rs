//! CPU-f32 specification for resident ordered plans; no CPU speed path.
use super::immutable as super_cache;
use super::pool::{Allocation, Pool};
use super::*;
use crate::{FpEnv, JobStats, TraceRequest, TraceResult, pending::JobControl, reduce};
use mklight_core::{MkResult, job::RunTimer};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

type Lease = Allocation<Vec<u32>>;
pub struct Runtime {
    pub owner: u64,
    cache: Arc<Mutex<super_cache::ImmutableCache>>,
    next: AtomicU64,
    pool: Arc<Mutex<Pool<Vec<u32>>>>,
}
impl Runtime {
    pub fn new(owner: u64) -> Self {
        Self {
            owner,
            cache: Arc::new(Mutex::new(super_cache::ImmutableCache::new(64 << 20))),
            next: AtomicU64::new(1),
            pool: Arc::new(Mutex::new(Pool::new(256 << 20))),
        }
    }
    fn allocate(&self, desc: &BufferDesc, stats: &mut PlanStats) -> MkResult<Lease> {
        let (b, reuse) = Pool::take(&self.pool, desc.byte_len()? / 4, |n| Ok(vec![0; n]))?;
        if reuse {
            stats.reused_allocations += 1;
        } else {
            stats.allocations += 1;
        }
        Ok(b)
    }
    pub fn upload(&self, desc: BufferDesc, data: BufferData) -> MkResult<BufferHandle> {
        if data.element() != desc.element || data.words().len() * 4 != desc.byte_len()? {
            return Err(bad("upload type or length"));
        }
        let mut b = self.allocate(&desc, &mut PlanStats::default())?;
        b.copy_from_slice(data.words());
        Ok(self.publish(desc, b))
    }
    fn publish(&self, desc: BufferDesc, b: Lease) -> BufferHandle {
        BufferHandle::from_lease(
            self.owner,
            self.next.fetch_add(1, Ordering::Relaxed),
            desc,
            Arc::new(b),
        )
    }
    pub fn operation_cache_bytes(&self) -> u64 {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes() as u64
    }

    pub fn pooled_bytes(&self) -> u64 {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).bytes() as u64
    }
    pub fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback> {
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
            words.extend_from_slice(&b[a..a + width]);
        }
        ctx.check()?;
        ctx.report(1., "readback");
        Ok(Readback {
            copied_bytes: (words.len() * 4) as u64,
            data: BufferData::from_words(view.desc.element, &words),
            wall_seconds: t.elapsed().as_secs_f64(),
        })
    }
    pub fn execute<E: FpEnv>(
        &self,
        backend: &dyn ComputeBackend,
        plan: ResidentPlan,
        systems: std::collections::HashMap<crate::SystemHandle, Arc<crate::PackedSystemSet>>,
        ctl: &JobControl,
        mut trace: impl FnMut(usize, &TraceRequest, &JobControl, &mut JobStats) -> MkResult<TraceResult>,
    ) -> MkResult<PlanResult> {
        let timer = RunTimer::start();
        let start = Instant::now();
        let mut x = Execution {
            buffers: vec![],
            systems,
            cache: self.cache.clone(),
            stats: PlanStats::default(),
        };
        for b in &plan.buffers {
            x.buffers.push(Some(match b {
                PlanBuffer::Scratch(d) => Slot::Owned(self.allocate(d, &mut x.stats)?),
                PlanBuffer::Import(v) => Slot::Import(
                    v.buffer.lease::<Lease>(self.owner)?,
                    v.offset * v.desc.element.words(),
                ),
            }));
        }
        x.stats.peak_bytes = plan
            .buffers
            .iter()
            .map(|b| b.desc().byte_len().unwrap() as u64)
            .sum();
        for (i, op) in plan.operations.iter().enumerate() {
            ctl.check()?;
            let sub = ctl.subrange(
                i as f64 / plan.operations.len() as f64,
                (i + 1) as f64 / plan.operations.len() as f64,
            );
            let t = Instant::now();
            let mut js = JobStats::default();
            match op {
                ResidentOp::TraceUpload(t) => {
                    let r = trace(i, t.request(), &sub, &mut js)?;
                    let n = r.len();
                    let f: [&[f32]; 10] = [
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
                    for (k, p) in f.iter().enumerate() {
                        x.words_mut(t.output.floats)[k * n..(k + 1) * n]
                            .copy_from_slice(bytemuck::cast_slice(p));
                    }
                    for (k, p) in [&r.status, &r.last_surface, &r.steps].iter().enumerate() {
                        x.words_mut(t.output.indices)[k * n..(k + 1) * n].copy_from_slice(p);
                    }
                    x.words_mut(t.output.records)
                        .copy_from_slice(bytemuck::cast_slice(&r.recorded));
                    x.words_mut(t.output.record_status)
                        .copy_from_slice(&r.recorded_status);
                    x.stats.uploaded_bytes += (n
                        * (crate::consts::MKL_TRACE_IN_FLOATS + crate::consts::MKL_TRACE_IN_UINTS)
                            as usize
                        * 4) as u64;
                }
                ResidentOp::Sum(s) => {
                    let count = s.input.desc.shape[1];
                    let channels = s.input.desc.shape[0];
                    let mut out = x.buffers[s.output.0].take().unwrap();
                    let values: &[f32] = bytemuck::cast_slice(x.words(s.input.buffer));
                    for c in 0..channels {
                        ctl.check()?;
                        let a = s.input.offset + c * count;
                        let groups = reduce::num_groups(count);
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
                        out.words_mut()[c * 2..c * 2 + 2]
                            .copy_from_slice(bytemuck::cast_slice(&[sum]));
                        ctl.report(
                            (i as f64 + (c + 1) as f64 / channels as f64)
                                / plan.operations.len() as f64,
                            "sum",
                        );
                    }
                    x.buffers[s.output.0] = Some(out);
                    js.command_buffers = 2;
                }
                _ => dispatch(op, &mut x, ctl)?,
            }
            ctl.check()?;
            ctl.report(
                (i + 1) as f64 / plan.operations.len() as f64,
                "resident plan",
            );
            x.stats.command_buffers += js.command_buffers;
            x.stats.stages.push(StageStats {
                operation: i,
                name: match op {
                    ResidentOp::TraceUpload(_) => "trace",
                    ResidentOp::Sum(_) => "sum",
                    _ => "domain",
                }
                .into(),
                wall_seconds: t.elapsed().as_secs_f64(),
                command_buffers: js.command_buffers,
                ..Default::default()
            });
        }
        ctl.check()?;
        let mut outputs = vec![];
        // Repeated final IDs share the same immutable lease.
        let mut published: std::collections::HashMap<BufferId, BufferHandle> =
            std::collections::HashMap::new();
        for id in &plan.outputs {
            let h = if let Some(h) = published.get(id) {
                h.clone()
            } else {
                let h = match x.buffers[id.0].take().unwrap() {
                    Slot::Owned(b) => self.publish(plan.desc(*id)?.clone(), b),
                    Slot::Import(_, _) => {
                        return Err(bad("publish a view through explicit readback"));
                    }
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
    fn words(&self) -> &[u32] {
        match self {
            Self::Owned(b) => b,
            Self::Import(b, a) => &b[*a..],
        }
    }
    fn words_mut(&mut self) -> &mut [u32] {
        match self {
            Self::Owned(b) => b,
            _ => unreachable!(),
        }
    }
}
pub struct Execution {
    cache: Arc<Mutex<super_cache::ImmutableCache>>,
    systems: std::collections::HashMap<crate::SystemHandle, Arc<crate::PackedSystemSet>>,
    buffers: Vec<Option<Slot>>,
    pub stats: PlanStats,
}
impl Execution {
    pub fn cached<T: std::any::Any + Send + Sync>(
        &self,
        key: &super_cache::OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }
    pub fn cache<T: std::any::Any + Send + Sync>(
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

    pub fn system(&self, handle: crate::SystemHandle) -> MkResult<Arc<crate::PackedSystemSet>> {
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
        _ => Ok(()),
    }
}
