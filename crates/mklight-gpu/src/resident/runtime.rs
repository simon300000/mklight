//! Backend-independent half of a resident runtime: pooled leases, publication,
//! explicit readback, the operation cache and result packing.
use super::immutable::{ImmutableCache, OperationCacheKey};
use super::pool::{Allocation, Pool};
use super::*;
use mklight_core::job::RunTimer;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::{collections::HashMap, time::Instant};

/// Word storage of one backend allocation.
pub trait Storage: Send + Sync + 'static {
    fn words(&self) -> &[u32];
    fn words_mut(&mut self) -> &mut [u32];
}
impl Storage for Vec<u32> {
    fn words(&self) -> &[u32] {
        self
    }
    fn words_mut(&mut self) -> &mut [u32] {
        self
    }
}

pub struct Runtime<T> {
    pub owner: u64,
    cache: Arc<Mutex<ImmutableCache>>,
    next: AtomicU64,
    pool: Arc<Mutex<Pool<T>>>,
}
impl<T: Storage> Runtime<T> {
    pub fn new(owner: u64) -> Self {
        Self {
            owner,
            cache: Arc::new(Mutex::new(ImmutableCache::new(64 << 20))),
            next: AtomicU64::new(1),
            pool: Arc::new(Mutex::new(Pool::new(256 << 20))),
        }
    }
    /// An exclusive allocation of `words`; contents are unspecified.
    pub fn take(
        &self,
        words: usize,
        stats: &mut PlanStats,
        allocate: impl FnOnce(usize) -> MkResult<T>,
    ) -> MkResult<Allocation<T>> {
        let (b, reused) = Pool::take(&self.pool, words, allocate)?;
        if reused {
            stats.reused_allocations += 1;
        } else {
            stats.allocations += 1;
        }
        Ok(b)
    }
    pub fn publish(&self, desc: BufferDesc, b: Allocation<T>) -> BufferHandle {
        BufferHandle::from_lease(
            self.owner,
            self.next.fetch_add(1, Ordering::Relaxed),
            desc,
            Arc::new(b),
        )
    }
    pub fn lease(&self, h: &BufferHandle) -> MkResult<Arc<Allocation<T>>> {
        h.lease(self.owner)
    }
    pub fn upload(
        &self,
        desc: BufferDesc,
        data: BufferData,
        allocate: impl FnOnce(usize) -> MkResult<T>,
    ) -> MkResult<BufferHandle> {
        if data.element() != desc.element || data.words().len() * 4 != desc.byte_len()? {
            return Err(bad("upload type or length"));
        }
        let mut b = self.take(data.words().len(), &mut PlanStats::default(), allocate)?;
        b.words_mut().copy_from_slice(data.words());
        Ok(self.publish(desc, b))
    }
    pub fn upload_with(
        &self,
        desc: BufferDesc,
        fill: &mut dyn FnMut(&mut [u32]),
        allocate: impl FnOnce(usize) -> MkResult<T>,
    ) -> MkResult<BufferHandle> {
        let mut b = self.take(desc.byte_len()? / 4, &mut PlanStats::default(), allocate)?;
        fill(b.words_mut());
        Ok(self.publish(desc, b))
    }
    pub fn read_in_place(
        &self,
        view: BufferView,
        ctx: &JobContext,
        read: &mut dyn FnMut(&[u32]),
    ) -> MkResult<()> {
        ctx.check()?;
        view.validate()?;
        if view.stride != 1 {
            return Err(bad("in-place readback needs a contiguous view"));
        }
        let b = self.lease(&view.buffer)?;
        let width = view.desc.element.words();
        read(&T::words(&b)[view.offset * width..(view.offset + view.desc.len()?) * width]);
        Ok(())
    }
    pub fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback> {
        ctx.check()?;
        view.validate()?;
        let t = Instant::now();
        let b = self.lease(&view.buffer)?;
        let src = T::words(&b);
        let width = view.desc.element.words();
        let n = view.desc.len()?;
        let words = if view.stride == 1 {
            src[view.offset * width..(view.offset + n) * width].to_vec()
        } else {
            let mut words = Vec::with_capacity(n * width);
            for i in 0..n {
                let a = (view.offset + i * view.stride) * width;
                words.extend_from_slice(&src[a..a + width]);
            }
            words
        };
        ctx.check()?;
        ctx.report(1., "readback");
        Ok(Readback {
            copied_bytes: (words.len() * 4) as u64,
            data: BufferData::from_words(view.desc.element, &words),
            wall_seconds: t.elapsed().as_secs_f64(),
        })
    }
    pub fn cache_stats(&self, mut s: ResidentCacheStats) -> ResidentCacheStats {
        s.pooled_bytes = lock(&self.pool).bytes() as u64;
        s.operation_cache_bytes = lock(&self.cache).bytes() as u64;
        s
    }
    pub fn operation_cache(&self, plan: &ResidentPlan, kernel_build: &str) -> OperationCache {
        OperationCache {
            cache: self.cache.clone(),
            identity: SystemCacheKey {
                snapshot: plan.snapshot.hash().to_hex(),
                library: plan.identity.library_hash.clone(),
                configuration: plan.identity.configuration.clone(),
                spectrum: plan.identity.spectrum.clone(),
                layout: crate::consts::MKL_LAYOUT_VERSION,
                kernel_build: kernel_build.into(),
                operation_settings: plan.settings.to_string(),
            },
        }
    }
    /// Publish the final outputs (repeated IDs share one lease) and stamp RunInfo.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        plan: &ResidentPlan,
        backend: &dyn ComputeBackend,
        timer: RunTimer,
        start: Instant,
        mut stats: PlanStats,
        ctl: &crate::pending::JobControl,
        mut take: impl FnMut(BufferId) -> Allocation<T>,
    ) -> MkResult<PlanResult> {
        ctl.check()?;
        let mut published: HashMap<BufferId, BufferHandle> = HashMap::new();
        let mut outputs = vec![];
        for &id in &plan.outputs {
            let h = published
                .entry(id)
                .or_insert_with(|| self.publish(plan.buffers[id.0].desc().clone(), take(id)));
            outputs.push(h.clone());
        }
        let pack = Instant::now();
        let mut run_info = stamp(plan, backend, timer);
        stats.stages.push(StageStats {
            operation: plan.operations.len(),
            name: "metadata".into(),
            wall_seconds: pack.elapsed().as_secs_f64(),
            ..Default::default()
        });
        stats.wall_seconds = start.elapsed().as_secs_f64();
        run_info.settings["plan_stats"] = serde_json::to_value(&stats).unwrap();
        ctl.report(1., "resident plan");
        Ok(PlanResult {
            outputs,
            stats,
            run_info,
        })
    }
}

/// Immutable per-operation state shared by plans with the same identity.
pub struct OperationCache {
    cache: Arc<Mutex<ImmutableCache>>,
    /// The submitting plan's snapshot/library/configuration/spectrum, layout,
    /// build and settings identity.
    pub identity: SystemCacheKey,
}
impl OperationCache {
    pub fn key(&self, operation: impl Into<String>) -> OperationCacheKey {
        OperationCacheKey {
            system: self.identity.clone(),
            operation: operation.into(),
            plan_version: PLAN_VERSION,
        }
    }
    pub fn get<V: Any + Send + Sync>(&self, key: &OperationCacheKey) -> MkResult<Option<Arc<V>>> {
        lock(&self.cache).get(key)
    }
    pub fn insert<V: Any + Send + Sync>(
        &self,
        key: OperationCacheKey,
        bytes: usize,
        value: Arc<V>,
    ) {
        lock(&self.cache).insert(key, bytes, value)
    }
}

fn lock<V>(m: &Mutex<V>) -> std::sync::MutexGuard<'_, V> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
