//! Backend-independent half of a resident runtime: pooled leases, publication,
//! explicit readback, the operation cache and result packing.
use super::immutable::{
    BuildCost, CacheCategory, CacheCategoryStats, CacheStats, ImmutableCache, OperationCacheKey,
};
use super::pool::{Allocation, Pool};
use super::*;
use mklight_core::job::RunTimer;
use mklight_core::platform::Instant;
use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Word storage of one backend allocation.
pub trait Storage: Send + Sync + 'static {
    fn words(&self) -> &[u32];
    fn words_mut(&mut self) -> &mut [u32];
    /// Deferred device encodings may retain an allocation returned on error.
    fn available(&self) -> bool {
        true
    }
}
impl Storage for HostAllocation {
    fn words(&self) -> &[u32] {
        self
    }
    fn words_mut(&mut self) -> &mut [u32] {
        self
    }
}

impl<T: Storage> ReadStorage for Allocation<T> {
    fn words(&self) -> &[u32] {
        T::words(self)
    }
}

pub struct Runtime<T> {
    pub owner: u64,
    cache: Arc<Mutex<ImmutableCache>>,
    next: AtomicU64,
    copied_bytes: AtomicU64,
    pool: Arc<Mutex<Pool<T>>>,
}
impl<T: Storage> Runtime<T> {
    pub fn new(owner: u64) -> Self {
        Self {
            owner,
            cache: Arc::new(Mutex::new(ImmutableCache::new())),
            next: AtomicU64::new(1),
            copied_bytes: AtomicU64::new(0),
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
        let (b, reused) = Pool::take(&self.pool, words, allocate, T::available)?;
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
        self.copied_bytes
            .fetch_add(desc.byte_len()? as u64, Ordering::Relaxed);
        Ok(self.publish(desc, b))
    }
    pub fn fill(
        &self,
        desc: BufferDesc,
        fill: &mut dyn FnMut(BufferFill<'_>) -> MkResult<()>,
        allocate: impl FnOnce(usize) -> MkResult<T>,
    ) -> MkResult<BufferHandle> {
        let mut b = self.take(desc.byte_len()? / 4, &mut PlanStats::default(), allocate)?;
        fill(BufferFill {
            element: desc.element,
            words: b.words_mut(),
        })?;
        Ok(self.publish(desc, b))
    }
    /// Publish caller-owned storage; it is released, never pooled.
    pub fn adopt(&self, desc: BufferDesc, storage: T) -> MkResult<BufferHandle> {
        let words = desc.byte_len()? / 4;
        if storage.words().len() != words {
            return Err(bad("import length"));
        }
        Ok(self.publish(desc, Allocation::unpooled(storage, words)))
    }
    pub fn read_view(&self, view: BufferView) -> MkResult<ReadView> {
        view.validate()?;
        if view.stride != 1 {
            return unsupported("strided read view; use readback");
        }
        let width = view.desc.element.words();
        Ok(ReadView {
            storage: self.lease(&view.buffer)?,
            offset: view.offset * width,
            len: view.desc.len()? * width,
            element: view.desc.element,
        })
    }
    /// One copy into owned data (a memcpy when contiguous).
    pub fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback> {
        ctx.check()?;
        view.validate()?;
        let t = Instant::now();
        let b = self.lease(&view.buffer)?;
        let src = T::words(&b);
        let data = match view.desc.element {
            ElementType::F32 => BufferData::F32(gather(src, &view)?),
            ElementType::U32 => BufferData::U32(gather(src, &view)?),
            ElementType::DoubleFloat => BufferData::DoubleFloat(gather(src, &view)?),
            ElementType::Complex32 => BufferData::Complex32(gather(src, &view)?),
        };
        let copied_bytes = data.words().len() as u64 * 4;
        self.copied_bytes.fetch_add(copied_bytes, Ordering::Relaxed);
        ctx.check()?;
        ctx.report(1., "readback");
        Ok(Readback {
            copied_bytes,
            data,
            wall_seconds: t.elapsed().as_secs_f64(),
        })
    }
    pub fn cache_stats(&self, mut s: ResidentCacheStats) -> ResidentCacheStats {
        s.copied_bytes = self.copied_bytes.load(Ordering::Relaxed);
        s.pooled_bytes = lock(&self.pool).bytes() as u64;
        let cache = lock(&self.cache);
        s.operation_cache_bytes = cache.bytes() as u64;
        s.categories.extend(cache.stats());
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
        stats.host.add(&plan.host);
        stats.memory.host_peak_rss_bytes = MemoryStats::host_peak_rss();
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
    /// A key scoped to the plan's whole identity (snapshot, settings, ...).
    pub fn key(&self, category: CacheCategory, operation: impl Into<String>) -> OperationCacheKey {
        OperationCacheKey {
            system: self.identity.clone(),
            category,
            operation: operation.into(),
            plan_version: PLAN_VERSION,
        }
    }
    /// A content-addressed key: only layout and kernel build come from the
    /// plan, so `operation` must name digests of every input it depends on.
    /// Plans of different analyses or snapshots share such entries whenever
    /// their inputs are bitwise identical.
    pub fn content_key(
        &self,
        category: CacheCategory,
        operation: impl Into<String>,
    ) -> OperationCacheKey {
        OperationCacheKey {
            system: SystemCacheKey {
                snapshot: String::new(),
                library: String::new(),
                configuration: String::new(),
                spectrum: String::new(),
                layout: self.identity.layout,
                kernel_build: self.identity.kernel_build.clone(),
                operation_settings: "content-addressed".into(),
            },
            category,
            operation: operation.into(),
            plan_version: PLAN_VERSION,
        }
    }
    /// The entry, counting a hit (with its saved build cost) or a miss in `plan`.
    pub fn get<V: Any + Send + Sync>(
        &self,
        key: &OperationCacheKey,
        plan: &mut CacheStats,
    ) -> MkResult<Option<Arc<V>>> {
        let got = lock(&self.cache).get::<V>(key)?;
        let s = plan.entry(key.category).or_default();
        Ok(match got {
            Some((v, cost)) => {
                s.hits += 1;
                s.saved_seconds += cost;
                Some(v)
            }
            None => {
                s.misses += 1;
                None
            }
        })
    }
    pub fn insert(
        &self,
        key: OperationCacheKey,
        bytes: usize,
        cost: BuildCost,
        value: Arc<dyn Any + Send + Sync>,
        plan: &mut CacheStats,
    ) {
        let category = key.category;
        let delta: CacheCategoryStats = lock(&self.cache).insert(key, bytes, cost.seconds(), value);
        plan.entry(category).or_default().add(&delta);
    }
}

fn gather<V: BufferElement>(src: &[u32], view: &BufferView) -> MkResult<Vec<V>> {
    let w = V::ELEMENT.words();
    let n = view.desc.len()?;
    let at = |i: usize| (view.offset + i * view.stride) * w;
    if view.stride == 1 {
        return Ok(typed::<V>(view.desc.element, &src[at(0)..at(n)])?.to_vec());
    }
    (0..n)
        .map(|i| Ok(typed::<V>(view.desc.element, &src[at(i)..at(i) + w])?[0]))
        .collect()
}

fn lock<V>(m: &Mutex<V>) -> std::sync::MutexGuard<'_, V> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
