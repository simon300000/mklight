//! `ComputeBackend`: the only GPU interface solvers see (ARCHITECTURE,
//! "Platform backends").
//!
//! The interface is deliberately small and data-oriented:
//!
//! 1. `upload_systems` copies one or more [`PackedSystem`]s to the device and
//!    returns a [`SystemHandle`] (systems are immutable on the device, like the
//!    snapshots they come from).
//! 2. `submit` takes a [`Job`]: a batched, backend-neutral description of work
//!    over host SoA arrays. It returns a [`Pending`] immediately. The backend
//!    splits large jobs into command buffers (or launches), checks the job's
//!    `CancelToken` between them and reports progress through the
//!    `JobContext`.
//! 3. Each result carries [`JobStats`] (device time, launches), and the
//!    backend describes itself with [`DeviceInfo`] and a
//!    [`PrecisionDescriptor`], which go into `RunInfo` for reproducibility.
//!
//! A backend that cannot run a job kind reports `supports(kind) == false` and
//! returns `MkError::Unsupported`; the engine then uses the CPU reference
//! (CONVENTIONS §7: never silently degraded numbers).

use std::sync::atomic::{AtomicU64, Ordering};

use mklight_core::job::Precision;
use mklight_core::{JobContext, MkResult};
use serde::{Deserialize, Serialize};

use crate::batch::{TraceRequest, TraceResult};
use crate::df::Df32;
use crate::layout::PackedSystem;
use crate::pending::Pending;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Bit-faithful CPU emulation of the f32 GPU numerics.
    CpuF32,
    Metal,
    Vulkan,
    Cuda,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::CpuF32 => "cpu-f32",
            BackendKind::Metal => "metal",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Cuda => "cuda",
        }
    }
}

/// What the device is and what it can do.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: BackendKind,
    pub name: String,
    pub vendor: String,
    /// Largest single buffer the device accepts (bytes).
    pub max_buffer_bytes: u64,
    /// Working-set size the driver recommends staying under (bytes).
    pub recommended_working_set_bytes: Option<u64>,
    /// Host and device share memory (Apple Silicon); uploads are memcpys.
    pub unified_memory: bool,
    /// The API for ray-tracing acceleration structures is available (possibly
    /// implemented in software/compute), reserved for the non-sequential
    /// solver's spatial searches.
    pub ray_tracing_api: bool,
    /// Dedicated ray-tracing hardware is present.
    pub hardware_ray_tracing: bool,
    /// Native FP64 arithmetic. Kernels never require it.
    pub native_fp64: bool,
    /// SIMD-group / warp / subgroup width.
    pub simd_width: u32,
    pub max_threads_per_group: u32,
}

impl DeviceInfo {
    /// `"metal:Apple M2 Pro"`-style identifier for `RunInfo::backend`.
    pub fn backend_id(&self) -> String {
        format!("{}:{}", self.kind.as_str(), self.name)
    }
}

/// Error budget for GPU results against the f64 reference (CONVENTIONS §7);
/// cross-validation tests enforce these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorBudget {
    pub position_mm: f64,
    pub direction_cosine: f64,
    pub opd_waves: f64,
    pub rms_spot_rel: f64,
    pub rms_spot_abs_mm: f64,
    pub mtf_abs: f64,
    pub aperture_edge_mm: f64,
}

impl ErrorBudget {
    /// The CONVENTIONS §7 table, for systems up to 1000 mm and 30 surfaces.
    pub const CONVENTIONS: ErrorBudget = ErrorBudget {
        position_mm: 2e-5,
        direction_cosine: 2e-6,
        opd_waves: 5e-3,
        rms_spot_rel: 5e-3,
        rms_spot_abs_mm: 2e-5,
        mtf_abs: 2e-3,
        aperture_edge_mm: 1e-5,
    };
}

/// Measured floating-point behaviour of a device. The double-float and
/// compensated-sum code requires the first four to be true.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericsProbe {
    /// `a + b` and `a · b` round to nearest-even, bit-identical to IEEE.
    pub ieee_add_mul: bool,
    /// `fma(a, b, c)` is a true fused multiply-add (single rounding).
    pub true_fma: bool,
    /// The compiler does not contract `a·b + c` into an fma.
    pub no_contraction: bool,
    /// The compiler does not reassociate (`(a + b) − b` is not folded to `a`).
    pub no_reassociation: bool,
    /// `a / b` is correctly rounded (needed only for bitwise df division).
    pub correctly_rounded_div: bool,
    /// `sqrt(a)` is correctly rounded (needed only for bitwise df sqrt).
    pub correctly_rounded_sqrt: bool,
    /// f32 subnormals are preserved (otherwise flushed to zero).
    pub subnormals: bool,
}

impl NumericsProbe {
    /// Behaviour of IEEE-754 binary32 on the CPU (the emulation backend).
    pub const IEEE: NumericsProbe = NumericsProbe {
        ieee_add_mul: true,
        true_fma: true,
        no_contraction: true,
        no_reassociation: true,
        correctly_rounded_div: true,
        correctly_rounded_sqrt: true,
        subnormals: true,
    };
    /// Error-free transforms (two-sum, two-prod) are valid on this device.
    pub fn error_free_transforms(&self) -> bool {
        self.ieee_add_mul && self.true_fma && self.no_contraction && self.no_reassociation
    }
}

/// How a backend computes, recorded with every result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionDescriptor {
    /// Core precision class (`F32Gpu` for every GPU backend).
    pub model: Precision,
    /// Fast-math (reassociation, finite-math assumptions) is off.
    pub fast_math_disabled: bool,
    /// Optical path accumulation: double-float.
    pub opl_double_float: bool,
    /// Reductions: compensated / double-float tree (deterministic order).
    pub compensated_reductions: bool,
    pub probe: NumericsProbe,
    pub budget: ErrorBudget,
}

/// Opaque handle to systems uploaded with `upload_systems`. Only valid on the
/// backend instance that issued it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SystemHandle {
    pub backend_instance: u64,
    pub id: u64,
    /// Number of systems in the uploaded set.
    pub count: u32,
}

/// Unique id per backend instance, so handles cannot cross backends.
pub fn next_backend_instance() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Double-float self-test operations (`MKL_DFOP_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum DfOp {
    TwoSum = 0,
    FastTwoSum = 1,
    TwoProd = 2,
    Add = 3,
    Mul = 4,
    Div = 5,
    Sqrt = 6,
    AddF32 = 7,
    MulF32 = 8,
    Hypot2 = 9,
}

impl DfOp {
    pub const ALL: [DfOp; 10] = [
        DfOp::TwoSum,
        DfOp::FastTwoSum,
        DfOp::TwoProd,
        DfOp::Add,
        DfOp::Mul,
        DfOp::Div,
        DfOp::Sqrt,
        DfOp::AddF32,
        DfOp::MulF32,
        DfOp::Hypot2,
    ];
    /// Reference implementation (the CPU-f32 emulation).
    pub fn apply(self, a: Df32, b: Df32) -> Df32 {
        use crate::df::{fast_two_sum, two_prod, two_sum};
        let pair = |(h, l): (f32, f32)| Df32::new(h, l);
        match self {
            DfOp::TwoSum => pair(two_sum(a.hi, b.hi)),
            DfOp::FastTwoSum => pair(fast_two_sum(a.hi, b.hi)),
            DfOp::TwoProd => pair(two_prod(a.hi, b.hi)),
            DfOp::Add => a + b,
            DfOp::Mul => a * b,
            DfOp::Div => a / b,
            DfOp::Sqrt => a.sqrt(),
            DfOp::AddF32 => a.add_f32(b.hi),
            DfOp::MulF32 => a.mul_f32(b.hi),
            DfOp::Hypot2 => Df32::from_f32(crate::fmath::hypot2(a.hi, b.hi)),
        }
    }
    /// Whether bitwise CPU/GPU agreement additionally needs a correctly
    /// rounded hardware division or square root.
    pub fn needs_correct_div_sqrt(self) -> bool {
        matches!(self, DfOp::Div | DfOp::Sqrt | DfOp::Hypot2)
    }
}

/// Reduction accumulator (`MKL_REDUCE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum ReduceMode {
    /// Plain f32 (diagnostics/comparison only).
    Plain = 0,
    /// Per-thread Neumaier f32, combined in double-float.
    Kahan = 1,
    /// Double-float throughout.
    DoubleFloat = 2,
}

/// A unit of backend work. Inputs are host arrays; outputs come back in
/// [`JobOutput`].
#[derive(Clone, Debug)]
pub enum Job {
    /// Draws `[4·first_block, 4·(first_block + num_blocks))` of the Philox
    /// stream `(seed, stream)`, identical to `mklight_core::rng::Philox`.
    /// Output: [`JobOutput::Philox`].
    PhiloxStream {
        seed: u64,
        stream: u64,
        first_block: u64,
        num_blocks: u32,
    },
    /// Raw Philox4x32-10 blocks for arbitrary counters and keys.
    /// Output: [`JobOutput::Philox`] (4 words per block).
    PhiloxBlocks {
        counters: Vec<[u32; 4]>,
        keys: Vec<[u32; 2]>,
    },
    /// Elementwise double-float operation. Output: [`JobOutput::Df`].
    DfOps {
        op: DfOp,
        a: Vec<Df32>,
        b: Vec<Df32>,
    },
    /// Deterministic compensated sum of each channel of `values`
    /// (channel-major, `count` values per channel). Output: [`JobOutput::Sums`].
    Reduce {
        mode: ReduceMode,
        count: usize,
        values: Vec<f32>,
    },
    /// Sag of one packed surface at the given `r²` (mm²). Output: [`JobOutput::F32`].
    SagProbe {
        systems: SystemHandle,
        system: u32,
        surface: u32,
        r2: Vec<f32>,
    },
    /// Sequential trace. Output: [`JobOutput::Trace`].
    Trace(Box<TraceRequest>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    PhiloxStream,
    PhiloxBlocks,
    DfOps,
    Reduce,
    SagProbe,
    Trace,
}

impl Job {
    pub fn kind(&self) -> JobKind {
        match self {
            Job::PhiloxStream { .. } => JobKind::PhiloxStream,
            Job::PhiloxBlocks { .. } => JobKind::PhiloxBlocks,
            Job::DfOps { .. } => JobKind::DfOps,
            Job::Reduce { .. } => JobKind::Reduce,
            Job::SagProbe { .. } => JobKind::SagProbe,
            Job::Trace(_) => JobKind::Trace,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum JobOutput {
    /// Raw Philox words and their f32 uniforms (`fmath::u32_to_unit_f32`).
    Philox {
        raw: Vec<u32>,
        uniform: Vec<f32>,
    },
    Df(Vec<Df32>),
    /// One double-float sum per channel.
    Sums(Vec<Df32>),
    F32(Vec<f32>),
    Trace(TraceResult),
}

/// Execution statistics for one job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JobStats {
    /// Device execution time summed over command buffers (s); 0 if unknown.
    pub device_seconds: f64,
    /// Host wall time from submission to completion (s).
    pub wall_seconds: f64,
    /// Command buffers (or launches) used.
    pub command_buffers: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobResult {
    pub output: JobOutput,
    pub stats: JobStats,
}

impl Job {
    /// Backend-independent input checks, run by every backend before work
    /// starts.
    pub fn validate(&self) -> MkResult<()> {
        use mklight_core::MkError;
        let bad = |m: &str| Err(MkError::Backend(m.to_string()));
        match self {
            Job::PhiloxStream { .. } => Ok(()),
            Job::PhiloxBlocks { counters, keys } => {
                if counters.len() != keys.len() {
                    return bad("philox: counters and keys differ in length");
                }
                Ok(())
            }
            Job::DfOps { a, b, .. } => {
                if a.len() != b.len() {
                    return bad("df ops: operand lengths differ");
                }
                Ok(())
            }
            Job::Reduce { count, values, .. } => {
                if *count == 0 || values.is_empty() || values.len() % count != 0 {
                    return bad("reduce: values.len() must be a positive multiple of count");
                }
                if values.len() > u32::MAX as usize {
                    return bad("reduce: more than 2^32 values");
                }
                Ok(())
            }
            Job::SagProbe {
                systems, system, ..
            } => {
                if *system >= systems.count {
                    return bad("sag probe: system index out of range");
                }
                Ok(())
            }
            Job::Trace(t) => t.rays.validate(),
        }
    }
}

/// The backend interface. Object-safe: the engine holds `Arc<dyn ComputeBackend>`.
pub trait ComputeBackend: Send + Sync {
    fn info(&self) -> &DeviceInfo;
    fn precision(&self) -> &PrecisionDescriptor;
    fn supports(&self, kind: JobKind) -> bool;

    /// Copy systems to the device. The handle stays valid until released.
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle>;
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()>;

    /// Start a job asynchronously. Cancellation (`ctx.cancel`) is checked
    /// between command buffers; progress is reported through `ctx`.
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>>;

    /// Upload a single system.
    fn upload_system(&self, system: &PackedSystem) -> MkResult<SystemHandle> {
        self.upload_systems(std::slice::from_ref(system))
    }
    /// Submit and block until done.
    fn run(&self, job: Job, ctx: &JobContext) -> MkResult<JobResult> {
        self.submit(job, ctx)?.wait()
    }
}
