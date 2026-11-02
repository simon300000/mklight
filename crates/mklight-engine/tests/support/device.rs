//! Count legacy trace submissions separately from resident solver plans.
use mklight_core::{JobContext, MkResult};
use mklight_gpu::{resident::*, *};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Device {
    inner: CpuF32Backend,
    pub traces: AtomicUsize,
}
impl Device {
    pub fn new() -> Self {
        Self {
            inner: CpuF32Backend::new(),
            traces: AtomicUsize::new(0),
        }
    }
}
impl ComputeBackend for Device {
    fn info(&self) -> &DeviceInfo {
        self.inner.info()
    }
    fn precision(&self) -> &PrecisionDescriptor {
        self.inner.precision()
    }
    fn supports(&self, kind: JobKind) -> bool {
        self.inner.supports(kind)
    }
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        self.inner.upload_systems(systems)
    }
    fn upload_systems_cached(
        &self,
        key: &SystemCacheKey,
        systems: &[PackedSystem],
    ) -> MkResult<SystemHandle> {
        self.inner.upload_systems_cached(key, systems)
    }
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.release_systems(handle)
    }
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        if matches!(job, Job::Trace(_)) {
            self.traces.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.submit(job, ctx)
    }
    fn fill_buffer(
        &self,
        desc: BufferDesc,
        fill: &mut dyn FnMut(BufferFill<'_>) -> MkResult<()>,
    ) -> MkResult<BufferHandle> {
        self.inner.fill_buffer(desc, fill)
    }
    fn read_view(&self, view: BufferView) -> MkResult<ReadView> {
        self.inner.read_view(view)
    }
    fn submit_plan(&self, plan: ResidentPlan, ctx: &JobContext) -> MkResult<Pending<PlanResult>> {
        self.inner.submit_plan(plan, ctx)
    }
}
