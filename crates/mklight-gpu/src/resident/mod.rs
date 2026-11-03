//! Ordered resident plans. Buffers become immutable leases on publication;
//! host views pin those leases until their last borrow ends. Plan-local storage has
//! one writer and cannot alias imported buffers or another live submission.
use crate::{ComputeBackend, Df32, ReduceMode, TraceRequest};
use mklight_core::job::{RunInfo, RunTimer};
use mklight_core::{JobContext, MkError, MkResult, Snapshot};
use serde::{Deserialize, Serialize};
use std::{any::Any, fmt, sync::Arc};

pub const PLAN_VERSION: u32 = crate::consts::MKL_RESIDENT_PLAN_VERSION;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    F32,
    U32,
    DoubleFloat,
    Complex32,
}
impl ElementType {
    pub fn words(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 1,
            _ => 2,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferDesc {
    pub element: ElementType,
    pub shape: Vec<usize>,
}
impl BufferDesc {
    pub fn new(element: ElementType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            element,
            shape: shape.into(),
        }
    }
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> MkResult<usize> {
        if self.shape.is_empty() {
            return Err(bad("buffer needs a shape"));
        }
        self.shape.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b).ok_or_else(|| bad("buffer shape overflow"))
        })
    }
    pub fn byte_len(&self) -> MkResult<usize> {
        self.len()?
            .checked_mul(self.element.words() * 4)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| bad("buffer byte size overflow"))
    }
}
/// Backend-owned, immutable allocation. Payloads never expose Metal objects to
/// solvers. Generation is unique even when the pool reuses the allocation.
#[derive(Clone)]
pub struct BufferHandle {
    owner: u64,
    generation: u64,
    desc: BufferDesc,
    payload: Arc<dyn Any + Send + Sync>,
}
impl fmt::Debug for BufferHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferHandle")
            .field("owner", &self.owner)
            .field("generation", &self.generation)
            .field("desc", &self.desc)
            .finish()
    }
}
impl BufferHandle {
    /// Backend implementors only: attach a lease that returns storage on drop.
    pub fn from_lease<T: Any + Send + Sync>(
        owner: u64,
        generation: u64,
        desc: BufferDesc,
        lease: Arc<T>,
    ) -> Self {
        Self {
            owner,
            generation,
            desc,
            payload: lease,
        }
    }
    pub fn owner(&self) -> u64 {
        self.owner
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }
    pub fn lease<T: Any + Send + Sync>(&self, owner: u64) -> MkResult<Arc<T>> {
        if self.owner != owner {
            return Err(bad("foreign buffer handle"));
        }
        self.payload
            .clone()
            .downcast::<T>()
            .map_err(|_| bad("stale or incompatible buffer handle"))
    }
    pub fn view(&self) -> BufferView {
        BufferView {
            buffer: self.clone(),
            offset: 0,
            desc: self.desc.clone(),
            stride: 1,
        }
    }
}
/// Offsets and strides are in typed elements, never bytes.
#[derive(Clone, Debug)]
pub struct BufferView {
    pub buffer: BufferHandle,
    pub offset: usize,
    pub desc: BufferDesc,
    pub stride: usize,
}
impl BufferView {
    pub fn validate(&self) -> MkResult<()> {
        let n = self.desc.len()?;
        let end = span_end(self.offset, n, self.stride)?;
        if self.desc.element != self.buffer.desc.element || end > self.buffer.desc.len()? {
            return Err(bad("buffer view type or extent"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum BufferData {
    F32(Vec<f32>),
    U32(Vec<u32>),
    DoubleFloat(Vec<Df32>),
    Complex32(Vec<[f32; 2]>),
}
impl BufferData {
    pub fn element(&self) -> ElementType {
        match self {
            Self::F32(_) => ElementType::F32,
            Self::U32(_) => ElementType::U32,
            Self::DoubleFloat(_) => ElementType::DoubleFloat,
            Self::Complex32(_) => ElementType::Complex32,
        }
    }
    pub fn words(&self) -> &[u32] {
        match self {
            Self::F32(x) => bytemuck::cast_slice(x),
            Self::U32(x) => x,
            Self::DoubleFloat(x) => bytemuck::cast_slice(x),
            Self::Complex32(x) => bytemuck::cast_slice(x),
        }
    }
    pub fn from_words(t: ElementType, words: &[u32]) -> Self {
        match t {
            ElementType::F32 => Self::F32(bytemuck::cast_slice(words).to_vec()),
            ElementType::U32 => Self::U32(words.to_vec()),
            // Df32 is 8-byte aligned; a u32 slice (e.g. an empty one) need not be.
            ElementType::DoubleFloat => Self::DoubleFloat(
                (words.as_chunks().0.iter())
                    .map(|&[hi, lo]| crate::Df32::new(f32::from_bits(hi), f32::from_bits(lo)))
                    .collect(),
            ),
            ElementType::Complex32 => Self::Complex32(bytemuck::cast_slice(words).to_vec()),
        }
    }
}
/// Element types of typed host views (sealed: layout matches `ElementType`).
pub trait BufferElement: bytemuck::Pod + element::Sealed {
    const ELEMENT: ElementType;
}
mod element {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for crate::Df32 {}
    impl Sealed for [f32; 2] {}
}
impl BufferElement for f32 {
    const ELEMENT: ElementType = ElementType::F32;
}
impl BufferElement for u32 {
    const ELEMENT: ElementType = ElementType::U32;
}
impl BufferElement for Df32 {
    const ELEMENT: ElementType = ElementType::DoubleFloat;
}
impl BufferElement for [f32; 2] {
    const ELEMENT: ElementType = ElementType::Complex32;
}
/// Words of `T`; storage is page-aligned and views start on element boundaries.
fn typed<T: BufferElement>(element: ElementType, words: &[u32]) -> MkResult<&[T]> {
    if element != T::ELEMENT {
        return Err(bad("host view element type"));
    }
    Ok(bytemuck::cast_slice(words))
}

/// The unpublished allocation of `fill_buffer`, exclusive for the callback.
/// Its slices cannot escape the callback:
/// ```compile_fail,E0521
/// # use mklight_gpu::{resident::*, *};
/// let b = CpuF32Backend::new();
/// let mut escaped: &mut [u32] = &mut [];
/// b.fill_buffer(BufferDesc::new(ElementType::U32, [1]), &mut |mut f| {
///     escaped = f.as_mut_slice::<u32>()?;
///     Ok(())
/// });
/// ```
pub struct BufferFill<'a> {
    pub(crate) element: ElementType,
    pub(crate) words: &'a mut [u32],
}
impl BufferFill<'_> {
    pub fn as_mut_slice<T: BufferElement>(&mut self) -> MkResult<&mut [T]> {
        if self.element != T::ELEMENT {
            return Err(bad("host view element type"));
        }
        Ok(bytemuck::cast_slice_mut(self.words))
    }
}
/// Type-erased published storage pinned by a `ReadView`.
pub trait ReadStorage: Send + Sync {
    fn words(&self) -> &[u32];
}
/// A contiguous view of published (hence device-complete, immutable) storage.
/// It pins the allocation: the pool cannot reuse it, and dropping handles or
/// the backend does not invalidate it. Slices borrow the guard:
/// ```compile_fail,E0716
/// # use mklight_gpu::{resident::*, *};
/// let b = CpuF32Backend::new();
/// let h = b.fill_buffer(BufferDesc::new(ElementType::U32, [1]), &mut |_| Ok(())).unwrap();
/// let words = b.read_view(h.view()).unwrap().words();
/// assert_eq!(words.len(), 1);
/// ```
pub struct ReadView {
    pub(crate) storage: Arc<dyn ReadStorage>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) element: ElementType,
}
impl ReadView {
    pub fn words(&self) -> &[u32] {
        &self.storage.words()[self.offset..self.offset + self.len]
    }
    pub fn as_slice<T: BufferElement>(&self) -> MkResult<&[T]> {
        typed(self.element, self.words())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);
#[derive(Clone, Debug)]
pub struct PlanView {
    pub buffer: BufferId,
    pub offset: usize,
    pub desc: BufferDesc,
    pub stride: usize,
}
#[derive(Clone, Debug)]
pub enum PlanBuffer {
    Import(BufferView),
    Scratch(BufferDesc),
}
impl PlanBuffer {
    pub fn desc(&self) -> &BufferDesc {
        match self {
            Self::Import(v) => &v.desc,
            Self::Scratch(d) => d,
        }
    }
}
#[derive(Clone, Debug)]
pub struct TraceBuffers {
    pub floats: BufferId,
    pub indices: BufferId,
    pub records: BufferId,
    pub record_status: BufferId,
}
/// Compatibility adapter for the existing host trace. Its outputs stay resident.
/// Device generation and aiming use aiming::TraceResident (P1).
#[derive(Clone, Debug)]
pub struct TraceUpload {
    request: Arc<TraceRequest>,
    pub output: TraceBuffers,
}
impl TraceUpload {
    pub fn new(request: TraceRequest, output: TraceBuffers) -> Self {
        Self {
            request: Arc::new(request),
            output,
        }
    }
    pub fn request(&self) -> &TraceRequest {
        &self.request
    }
}
/// Output planes of a trace, in `MKL_TRACE_*` order: f32 `[10,rays]`, u32
/// `[3,rays]`, f32 `[slots,rays,9]` and u32 `[slots,rays]`.
fn trace_descs(request: &TraceRequest) -> [BufferDesc; 4] {
    use crate::consts::*;
    let n = request.rays.len();
    let slots = if request.record == crate::RecordMode::AllSurfaces {
        request.program().len() + 1
    } else {
        0
    };
    [
        BufferDesc::new(ElementType::F32, [MKL_TRACE_OUT_FLOATS as usize, n]),
        BufferDesc::new(ElementType::U32, [MKL_TRACE_OUT_UINTS as usize, n]),
        BufferDesc::new(
            ElementType::F32,
            [slots, n, MKL_TRACE_RECORD_FLOATS as usize],
        ),
        BufferDesc::new(ElementType::U32, [slots, n]),
    ]
}
#[derive(Clone, Debug)]
pub struct Sum {
    pub input: PlanView,
    pub mode: ReduceMode,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub enum ResidentOp {
    TraceUpload(Box<TraceUpload>),
    Sum(Sum),
    Aiming(crate::aiming::Request),
    Transforms(crate::transforms::Request),
    Collectives(crate::collectives::Request),
    Systems(crate::systems::Request),
    Geometry(crate::geometry::Request),
    Radiometry(crate::radiometry::Request),
    Wave(crate::wave_ops::Request),
    Ghost(crate::ghost_ops::Request),
    Stray(crate::stray_ops::Request),
    Optim(crate::optim_ops::Request),
}
#[derive(Clone, Debug)]
pub struct ResidentPlan {
    pub version: u32,
    /// Deferred command work limit. Zero completes each encoder call separately;
    /// this changes scheduling only, never the operation or reduction order.
    pub command_work_budget: usize,
    pub buffers: Vec<PlanBuffer>,
    pub operations: Vec<ResidentOp>,
    pub outputs: Vec<BufferId>,
    pub snapshot: Snapshot,
    pub identity: PlanIdentity,
    pub settings: serde_json::Value,
    pub seed: Option<u64>,
    /// Host preparation already done for this plan (reported, not identity).
    pub host: HostPreparation,
}
impl ResidentPlan {
    pub fn new(snapshot: Snapshot) -> Self {
        Self {
            version: PLAN_VERSION,
            command_work_budget: 1 << 19,
            buffers: vec![],
            operations: vec![],
            outputs: vec![],
            snapshot,
            identity: PlanIdentity::default(),
            settings: serde_json::json!({}),
            seed: None,
            host: HostPreparation::default(),
        }
    }
    pub fn scratch(&mut self, desc: BufferDesc) -> BufferId {
        let id = BufferId(self.buffers.len());
        self.buffers.push(PlanBuffer::Scratch(desc));
        id
    }
    pub fn import(&mut self, view: BufferView) -> BufferId {
        let id = BufferId(self.buffers.len());
        self.buffers.push(PlanBuffer::Import(view));
        id
    }
    pub fn desc(&self, id: BufferId) -> MkResult<&BufferDesc> {
        self.buffers
            .get(id.0)
            .map(PlanBuffer::desc)
            .ok_or_else(|| bad("unknown plan buffer"))
    }
    pub fn view(&self, id: BufferId) -> MkResult<PlanView> {
        Ok(PlanView {
            buffer: id,
            offset: 0,
            desc: self.desc(id)?.clone(),
            stride: 1,
        })
    }
    /// Add the legacy trace adapter and allocate its precisely sized planes.
    pub fn trace(&mut self, request: TraceRequest) -> TraceBuffers {
        let [f, u, r, rs] = trace_descs(&request);
        let out = TraceBuffers {
            floats: self.scratch(f),
            indices: self.scratch(u),
            records: self.scratch(r),
            record_status: self.scratch(rs),
        };
        self.operations
            .push(ResidentOp::TraceUpload(Box::new(TraceUpload::new(
                request,
                out.clone(),
            ))));
        out
    }
    pub fn sum(&mut self, input: PlanView, mode: ReduceMode) -> BufferId {
        let out = self.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            vec![input.desc.shape.first().copied().unwrap_or(0)],
        ));
        self.operations.push(ResidentOp::Sum(Sum {
            input,
            mode,
            output: out,
        }));
        out
    }
    /// Uploads referenced by either compatibility or domain operations.
    pub fn system_handles(&self) -> Vec<crate::SystemHandle> {
        self.operations
            .iter()
            .flat_map(|op| match op {
                ResidentOp::TraceUpload(t) => vec![t.request.systems],
                _ => accesses(op).systems,
            })
            .collect()
    }
    /// Single-writer ordered dependency validation, once at submission.
    pub fn validate(&self, owner: u64, max_bytes: u64) -> MkResult<()> {
        if self.version != PLAN_VERSION {
            return Err(bad("unsupported resident plan version"));
        }
        let mut ready = vec![false; self.buffers.len()];
        for (i, b) in self.buffers.iter().enumerate() {
            if b.desc().byte_len()? as u64 > max_bytes {
                return Err(bad("resident buffer exceeds device limit"));
            }
            if let PlanBuffer::Import(v) = b {
                v.validate()?;
                if v.stride != 1 {
                    return unsupported("strided plan import");
                }
                if v.buffer.owner() != owner {
                    return Err(bad("foreign buffer handle"));
                }
                ready[i] = true;
            }
        }
        for op in &self.operations {
            let writes = match op {
                ResidentOp::TraceUpload(t) => {
                    let o = &t.output;
                    let ids = vec![o.floats, o.indices, o.records, o.record_status];
                    for (&id, desc) in ids.iter().zip(trace_descs(&t.request)) {
                        if self.desc(id)? != &desc {
                            return Err(bad("trace output layout"));
                        }
                    }
                    ids
                }
                ResidentOp::Sum(s) => {
                    let d = self.desc(s.input.buffer)?;
                    if !ready[s.input.buffer.0] {
                        return Err(bad("read before producer"));
                    }
                    let shape = &s.input.desc.shape;
                    if s.input.desc.element != ElementType::F32
                        || d.element != ElementType::F32
                        || shape.len() != 2
                        || shape.contains(&0)
                        || s.input.stride != 1
                        || span_end(s.input.offset, s.input.desc.len()?, 1)? > d.len()?
                        || s.input.desc.len()? > u32::MAX as usize
                    {
                        return Err(bad("sum requires contiguous channel-major f32 view"));
                    }
                    if self.desc(s.output)?
                        != &BufferDesc::new(ElementType::DoubleFloat, vec![shape[0]])
                    {
                        return Err(bad("sum output layout"));
                    }
                    vec![s.output]
                }
                _ => {
                    route(op, self)?;
                    let a = accesses(op);
                    for v in a.reads {
                        let d = self.desc(v.buffer)?;
                        if !ready[v.buffer.0]
                            || v.desc.element != d.element
                            || span_end(v.offset, v.desc.len()?, v.stride)? > d.len()?
                        {
                            return Err(bad("domain input dependency or view"));
                        }
                    }
                    for &id in &a.writes {
                        self.desc(id)?;
                    }
                    a.writes
                }
            };
            for id in writes {
                if ready[id.0] {
                    return Err(bad("multiple writers or imported destination"));
                }
                ready[id.0] = true;
            }
        }
        for &id in &self.outputs {
            self.desc(id)?;
            if matches!(self.buffers[id.0], PlanBuffer::Import(_)) {
                return Err(bad("plan outputs must be produced buffers"));
            }
            if !ready[id.0] {
                return Err(bad("uninitialised plan output"));
            }
        }
        Ok(())
    }
}
/// Routers stay P0-owned. Later packages implement their own handlers.
pub fn route(op: &ResidentOp, plan: &ResidentPlan) -> MkResult<()> {
    match op {
        ResidentOp::Aiming(x) => crate::aiming::validate(x, plan),
        ResidentOp::Transforms(x) => crate::transforms::validate(x, plan),
        ResidentOp::Collectives(x) => crate::collectives::validate(x, plan),
        ResidentOp::Systems(x) => crate::systems::validate(x, plan),
        ResidentOp::Geometry(x) => crate::geometry::validate(x, plan),
        ResidentOp::Radiometry(x) => crate::radiometry::validate(x, plan),
        ResidentOp::Wave(x) => crate::wave_ops::validate(x, plan),
        ResidentOp::Ghost(x) => crate::ghost_ops::validate(x, plan),
        ResidentOp::Stray(x) => crate::stray_ops::validate(x, plan),
        ResidentOp::Optim(x) => crate::optim_ops::validate(x, plan),
        _ => Ok(()),
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageStats {
    pub operation: usize,
    pub name: String,
    pub wall_seconds: f64,
    pub device_seconds: f64,
    pub command_buffers: u32,
    pub max_command_seconds: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanStats {
    /// Actual bulk staging copies; excludes direct fills and small setBytes parameters.
    pub copied_bytes: u64,
    pub stages: Vec<StageStats>,
    pub wall_seconds: f64,
    pub device_seconds: f64,
    pub command_buffers: u32,
    pub allocations: u64,
    pub reused_allocations: u64,
    pub uploaded_bytes: u64,
    pub parameter_bytes: u64,
    pub readback_bytes: u64,
    pub peak_bytes: u64,
    pub fallback_count: u64,
    pub polished_rays: u64,
    /// This plan's operation-cache activity by category.
    #[serde(default)]
    pub cache: immutable::CacheStats,
    /// Pupil-aiming convergence of this plan's generations.
    #[serde(default)]
    pub aiming: AimStats,
    /// Host f64 preparation handed to the plan (chief seeds).
    #[serde(default)]
    pub host: HostPreparation,
    /// Memory observed when the plan completed.
    #[serde(default)]
    pub memory: MemoryStats,
}
impl PlanStats {
    /// Merge another plan's cache, aiming, host and memory records (the
    /// scalar counters and stages stay with the caller's own accumulation).
    pub fn add_activity(&mut self, o: &PlanStats) {
        immutable::merge_cache_stats(&mut self.cache, &o.cache);
        self.aiming.add(&o.aiming);
        self.host.add(&o.host);
        self.memory.max(&o.memory);
    }
}
/// Iteration statistics of generated (pupil) aims. Bins: 0, 1, 2, 3–4, 5–8,
/// 9–16, 17–60, more than 60 Newton iterations.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AimStats {
    pub generations: u64,
    pub rays: u64,
    /// Rays reported unresolved (any nonzero aim status).
    pub failed: u64,
    pub iterations: u64,
    pub max_iterations: u32,
    pub histogram: [u64; 8],
}
impl AimStats {
    pub const BINS: [u32; 7] = [0, 1, 2, 4, 8, 16, 60];
    pub fn bin(iterations: u32) -> usize {
        Self::BINS
            .iter()
            .position(|&b| iterations <= b)
            .unwrap_or(7)
    }
    /// Statistics of a `ResidentRays` report (u32 [4,n]); excluded samples
    /// (`MKL_AIM_OUTSIDE`) were never aimed and are not counted.
    pub fn from_report(report: &[u32], n: usize) -> Self {
        let mut s = Self {
            generations: 1,
            ..Default::default()
        };
        for i in 0..n {
            if report[i] == crate::consts::MKL_AIM_OUTSIDE {
                continue;
            }
            let it = report[n + i];
            s.rays += 1;
            s.failed += u64::from(report[i] != 0);
            s.iterations += it as u64;
            s.max_iterations = s.max_iterations.max(it);
            s.histogram[Self::bin(it)] += 1;
        }
        s
    }
    pub fn add(&mut self, o: &AimStats) {
        self.generations += o.generations;
        self.rays += o.rays;
        self.failed += o.failed;
        self.iterations += o.iterations;
        self.max_iterations = self.max_iterations.max(o.max_iterations);
        for (a, b) in self.histogram.iter_mut().zip(o.histogram) {
            *a += b;
        }
    }
}
/// Host f64 preparation reused by a plan: chief seed rows built from the
/// reference tracer's prepared fields (`reused_fields`) or fields solved for
/// this plan (`solved_fields`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HostPreparation {
    pub seed_rows: u64,
    pub reused_fields: u64,
    pub solved_fields: u64,
    pub seconds: f64,
}
impl HostPreparation {
    pub fn add(&mut self, o: &HostPreparation) {
        self.seed_rows += o.seed_rows;
        self.reused_fields += o.reused_fields;
        self.solved_fields += o.solved_fields;
        self.seconds += o.seconds;
    }
}
/// Memory high-water marks. `peak_bytes` of [`PlanStats`] is the plan's own
/// live storage; these are backend/process figures at plan completion.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Backend device allocations (Metal `currentAllocatedSize`), 0 if unknown.
    pub device_allocated_bytes: u64,
    /// Process resident-set high-water mark, 0 if unknown.
    pub host_peak_rss_bytes: u64,
}
impl MemoryStats {
    pub fn max(&mut self, o: &MemoryStats) {
        self.device_allocated_bytes = self.device_allocated_bytes.max(o.device_allocated_bytes);
        self.host_peak_rss_bytes = self.host_peak_rss_bytes.max(o.host_peak_rss_bytes);
    }
    /// Process resident-set high-water mark (getrusage), 0 where unavailable.
    pub fn host_peak_rss() -> u64 {
        #[cfg(unix)]
        {
            #[repr(C)]
            struct Timeval {
                sec: i64,
                usec: i64,
            }
            #[repr(C)]
            struct Rusage {
                utime: Timeval,
                stime: Timeval,
                maxrss: i64,
                rest: [i64; 13],
            }
            unsafe extern "C" {
                fn getrusage(who: i32, usage: *mut Rusage) -> i32;
            }
            let mut u = std::mem::MaybeUninit::<Rusage>::zeroed();
            // SAFETY: RUSAGE_SELF (0) fills the caller-provided struct, whose
            // layout matches the 64-bit Darwin/Linux `struct rusage`.
            if unsafe { getrusage(0, u.as_mut_ptr()) } != 0 {
                return 0;
            }
            let rss = unsafe { u.assume_init() }.maxrss.max(0) as u64;
            // Darwin reports bytes, Linux kilobytes.
            if cfg!(target_os = "macos") {
                rss
            } else {
                rss * 1024
            }
        }
        #[cfg(not(unix))]
        {
            0
        }
    }
}
#[derive(Clone, Debug)]
pub struct PlanResult {
    pub outputs: Vec<BufferHandle>,
    pub stats: PlanStats,
    pub run_info: RunInfo,
}
#[derive(Clone, Debug)]
pub struct Readback {
    pub data: BufferData,
    pub copied_bytes: u64,
    pub wall_seconds: f64,
}
/// Cache identity includes all immutable resolution and optical sampling inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SystemCacheKey {
    pub snapshot: String,
    pub library: String,
    pub configuration: String,
    pub spectrum: String,
    pub layout: u32,
    pub kernel_build: String,
    pub operation_settings: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResidentCacheStats {
    /// Cumulative explicit upload/readback copies outside plan execution.
    pub copied_bytes: u64,
    pub system_uploads: u64,
    pub system_hits: u64,
    pub uploaded_bytes: u64,
    pub system_bytes: u64,
    pub pooled_bytes: u64,
    pub operation_cache_bytes: u64,
    /// Every cache category, prepared systems included: cumulative activity,
    /// occupancy and budget.
    #[serde(default)]
    pub categories: immutable::CacheStats,
}
pub fn stamp(plan: &ResidentPlan, backend: &dyn ComputeBackend, timer: RunTimer) -> RunInfo {
    let operations: Vec<_> = plan.operations.iter().map(|op| match op {
        ResidentOp::TraceUpload(t) => trace_manifest(&t.request),
        ResidentOp::Sum(s) => serde_json::json!({"kind":"sum","input":s.input.buffer.0,"offset":s.input.offset,"shape":s.input.desc.shape,"mode":s.mode,"output":s.output.0}),
        _ => serde_json::json!({"kind":"domain","settings":plan.settings}),
    }).collect();
    let settings = serde_json::json!({"plan_version":plan.version,"command_work_budget":plan.command_work_budget,"identity":plan.identity,"requested_backend":plan.identity.requested_backend.as_deref().unwrap_or(&backend.info().backend_id()),"used_backend":backend.info().backend_id(),"cost_decision":"explicit_backend","operation_settings":plan.settings,"operations":operations,"buffers":plan.buffers.iter().map(PlanBuffer::desc).collect::<Vec<_>>(),"device":backend.info(),"precision":backend.precision()});
    timer.finish(
        &plan.snapshot,
        "resident_plan",
        &settings,
        &backend.info().backend_id(),
        backend.precision().model,
        plan.seed,
        vec![],
    )
}
pub fn span_end(offset: usize, len: usize, stride: usize) -> MkResult<usize> {
    if stride == 0 {
        return Err(bad("zero buffer stride"));
    }
    if len == 0 {
        return Ok(offset);
    }
    (len - 1)
        .checked_mul(stride)
        .and_then(|x| offset.checked_add(x))
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| bad("buffer view overflow"))
}
pub fn bad(s: &str) -> MkError {
    MkError::Backend(format!("resident: {s}"))
}
pub fn unsupported<T>(name: &str) -> MkResult<T> {
    Err(MkError::Unsupported(format!(
        "resident {name} is not implemented"
    )))
}
pub mod host;
pub use host::HostAllocation;
pub mod cache;
pub mod cpu;
pub mod immutable;
pub mod pool;
pub mod runtime;

#[derive(Default)]
pub struct Access {
    pub reads: Vec<PlanView>,
    pub writes: Vec<BufferId>,
    pub systems: Vec<crate::SystemHandle>,
}
pub fn accesses(op: &ResidentOp) -> Access {
    match op {
        ResidentOp::TraceUpload(t) => Access {
            writes: vec![
                t.output.floats,
                t.output.indices,
                t.output.records,
                t.output.record_status,
            ],
            ..Default::default()
        },
        ResidentOp::Sum(s) => Access {
            reads: vec![s.input.clone()],
            writes: vec![s.output],
            ..Default::default()
        },
        ResidentOp::Aiming(r) => crate::aiming::accesses(r),
        ResidentOp::Transforms(r) => crate::transforms::accesses(r),
        ResidentOp::Collectives(r) => crate::collectives::accesses(r),
        ResidentOp::Systems(r) => crate::systems::accesses(r),
        ResidentOp::Geometry(r) => crate::geometry::accesses(r),
        ResidentOp::Radiometry(r) => crate::radiometry::accesses(r),
        ResidentOp::Wave(r) => crate::wave_ops::accesses(r),
        ResidentOp::Ghost(r) => crate::ghost_ops::accesses(r),
        ResidentOp::Stray(r) => crate::stray_ops::accesses(r),
        ResidentOp::Optim(r) => crate::optim_ops::accesses(r),
    }
}

/// A system set may come from immutable upload or a P4 numeric sidecar.
#[derive(Clone, Debug)]
pub enum SystemSet {
    Uploaded(crate::SystemHandle),
    Resident {
        headers: BufferId,
        surfaces: BufferId,
        coefficients: BufferId,
        count: usize,
        max_surfaces: usize,
        max_wavelengths: usize,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanIdentity {
    pub library_hash: String,
    pub derived_snapshot_hash: Option<String>,
    pub configuration: String,
    pub spectrum: String,
    pub requested_backend: Option<String>,
}

fn trace_manifest(r: &TraceRequest) -> serde_json::Value {
    serde_json::json!({"kind":"trace_upload","rays":r.rays.len(),"start_surface":r.start_surface,"end_surface":r.end_surface,"path":r.program().iter().map(|p|p.encode()).collect::<Vec<_>>(),"clip":r.apply_apertures,"radiometry":r.radiometry,"record":r.record,"newton":r.newton})
}
