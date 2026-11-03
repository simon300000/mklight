//! Metal device, queue, shader library and pipeline cache, plus synchronous
//! command-buffer execution with error and timing reporting.
//!
//! The public surface is safe: device and library information, buffer
//! allocation and pipeline compilation. Encoding and dispatching kernels
//! ([`MetalContext::run`], [`Encoder`]) is crate-private and `unsafe`,
//! because a kernel indexes buffers by the parameters it is given, which the
//! type system cannot relate to the bound buffers' lengths. Solvers reach the
//! GPU only through `ComputeBackend`.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bytemuck::Pod;
use dispatch2::DispatchData;
use mklight_core::{MkError, MkResult};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues,
    MTLLanguageVersion, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize,
};

use crate::buffer::SharedBuffer;
use crate::shaders;

/// Which GPU to use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DeviceSelector {
    /// The system default device.
    #[default]
    Default,
    /// The first device whose name contains this text (case-insensitive).
    NameContains(String),
    /// A device by its stable registry id.
    RegistryId(u64),
}

/// Where the shader library comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LibraryChoice {
    /// The embedded offline-compiled metallib, falling back to runtime
    /// compilation of the embedded source if it is missing or fails to load.
    #[default]
    Auto,
    /// Only the embedded metallib.
    Precompiled,
    /// Always compile the embedded source at runtime (fast-math off).
    RuntimeSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryOrigin {
    Precompiled,
    RuntimeSource,
}

/// Kernel arithmetic. `PureF32` (the `MKLIGHT_PURE_F32` metallib, docs/GPU.md
/// §3.6) is an opt-in measurement mode, not qualified for OPD, wave or aiming
/// results; default construction (including engine Auto) is DoubleFloat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetalPrecision {
    #[default]
    DoubleFloat,
    PureF32,
}

impl MetalPrecision {
    pub fn name(self) -> &'static str {
        match self {
            Self::DoubleFloat => "double_float",
            Self::PureF32 => "pure_f32",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalOptions {
    pub device: DeviceSelector,
    pub library: LibraryChoice,
    pub precision: MetalPrecision,
}

impl MetalOptions {
    /// Defaults, with `MKLIGHT_METAL_PRECISION=f32` selecting `PureF32` for
    /// benchmarks. Only explicit callers read it; `default()` never does.
    pub fn from_env() -> Self {
        let f32 = std::env::var("MKLIGHT_METAL_PRECISION").is_ok_and(|v| v == "f32");
        Self {
            precision: if f32 {
                MetalPrecision::PureF32
            } else {
                MetalPrecision::DoubleFloat
            },
            ..Self::default()
        }
    }
}

/// Completed command-buffer GPU time in this process, nanoseconds
/// (see [`device_seconds`]).
static DEVICE_NANOS: AtomicU64 = AtomicU64::new(0);
/// Diagnostic (proposal study): `MKLIGHT_METAL_TRACE_DISPATCHES=1` completes
/// every deferred encode as its own command buffer and prints its device time.
fn trace_dispatches() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MKLIGHT_METAL_TRACE_DISPATCHES").is_ok_and(|v| v == "1"))
}

fn add_device_time(seconds: f64) {
    DEVICE_NANOS.fetch_add((seconds * 1e9) as u64, Ordering::Relaxed);
}

/// Sum of completed command-buffer GPU timestamps over all Metal contexts in
/// this process. Differences around a job measure its device time, including
/// fused and resident batches (benchmarks only).
pub fn device_seconds() -> f64 {
    DEVICE_NANOS.load(Ordering::Relaxed) as f64 * 1e-9
}

/// A specialisation constant (`[[function_constant(index)]]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionConstant {
    U32 { index: u16, value: u32 },
    Bool { index: u16, value: bool },
}

/// Pipeline cache key: kernel name plus its function-constant values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub function: &'static str,
    pub constants: Vec<FunctionConstant>,
}

impl PipelineKey {
    pub fn plain(function: &'static str) -> Self {
        PipelineKey {
            function,
            constants: Vec::new(),
        }
    }
    pub fn with_u32(function: &'static str, index: u16, value: u32) -> Self {
        PipelineKey {
            function,
            constants: vec![FunctionConstant::U32 { index, value }],
        }
    }
}

/// Timing and identity of one completed command buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CommandStats {
    /// GPU execution time (`GPUEndTime − GPUStartTime`), seconds.
    pub gpu_seconds: f64,
    /// Completed submissions (zero for deferred encoding).
    pub command_buffers: u32,
}

type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    origin: LibraryOrigin,
    library_build: String,
    pipelines: Mutex<HashMap<PipelineKey, Pso>>,
}

fn ns_error(what: &str, e: &NSError) -> MkError {
    MkError::Backend(format!("{what}: {}", e.localizedDescription()))
}

fn select_device(sel: &DeviceSelector) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    match sel {
        DeviceSelector::Default => MTLCreateSystemDefaultDevice(),
        DeviceSelector::NameContains(s) => {
            let s = s.to_lowercase();
            MTLCopyAllDevices()
                .iter()
                .find(|d| d.name().to_string().to_lowercase().contains(&s))
        }
        DeviceSelector::RegistryId(id) => {
            MTLCopyAllDevices().iter().find(|d| d.registryID() == *id)
        }
    }
}

/// Runtime compile options: the same numerics contract as the offline build
/// (fast-math off, precise fp32 functions). `setMathMode:` and
/// `setMathFloatingPointFunctions:` exist from macOS 15; older systems get the
/// equivalent `fastMathEnabled = NO`.
///
/// There is no compile option for FP contraction: even in the safe math mode
/// the runtime compiler fuses `a*b + c`. MSL 3.2 adds
/// `#pragma METAL fp contract(off)` (used by `mklight_common.h`), so the
/// language version is 3.2 where available. On older systems the library
/// still compiles, the numerics probe reports the contraction, and the
/// backend refuses double-float work instead of returning different bits.
///
/// Returns the options and a description of them for `RunInfo`.
fn compile_options(version: MTLLanguageVersion) -> (Retained<MTLCompileOptions>, String) {
    let o = MTLCompileOptions::new();
    let mut desc = vec!["fastMathEnabled=NO"];
    #[allow(deprecated)]
    o.setFastMathEnabled(false);
    if o.respondsToSelector(sel!(setMathMode:)) {
        o.setMathMode(MTLMathMode::Safe);
        desc.push("mathMode=safe");
    }
    if o.respondsToSelector(sel!(setMathFloatingPointFunctions:)) {
        o.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        desc.push("mathFloatingPointFunctions=precise");
    }
    o.setLanguageVersion(version);
    (o, desc.join(" "))
}

type Library = (
    Retained<ProtocolObject<dyn MTLLibrary>>,
    LibraryOrigin,
    String,
);

/// Loads the shader library. `Auto` prefers the precompiled metallib here;
/// `MetalBackend` additionally falls back to runtime source when the
/// precompiled library fails the numerics probe.
fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
    choice: LibraryChoice,
    precision: MetalPrecision,
) -> MkResult<Library> {
    let precompiled = || -> MkResult<Library> {
        let (metallib, flags) = match precision {
            MetalPrecision::DoubleFloat => (shaders::METALLIB, shaders::OFFLINE_FLAGS),
            MetalPrecision::PureF32 => (shaders::METALLIB_F32, shaders::OFFLINE_FLAGS_F32),
        };
        let bytes = metallib.ok_or_else(|| {
            MkError::Unsupported("no precompiled metallib was embedded at build time".into())
        })?;
        let data = DispatchData::from_static_bytes(bytes);
        let lib = device
            .newLibraryWithData_error(&data)
            .map_err(|e| ns_error("loading embedded metallib", &e))?;
        let desc = format!("{} precompiled metallib, flags: {flags}", precision.name());
        Ok((lib, LibraryOrigin::Precompiled, desc))
    };
    let runtime = || -> MkResult<Library> {
        let source = if precision == MetalPrecision::PureF32 {
            format!("#define MKLIGHT_PURE_F32 1\n{}", shaders::SOURCE)
        } else {
            shaders::SOURCE.to_string()
        };
        let src = NSString::from_str(&source);
        let compile = |v, name: &str| {
            let (opts, desc) = compile_options(v);
            device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map(|l| {
                    let d = format!(
                        "{} runtime-compiled source, MSL {name}, {desc}",
                        precision.name()
                    );
                    (l, LibraryOrigin::RuntimeSource, d)
                })
        };
        compile(MTLLanguageVersion::Version3_2, "3.2")
            .or_else(|_| compile(MTLLanguageVersion::Version3_0, "3.0"))
            .map_err(|e| ns_error("runtime shader compilation", &e))
    };
    match choice {
        LibraryChoice::Precompiled => precompiled(),
        LibraryChoice::RuntimeSource => runtime(),
        LibraryChoice::Auto => precompiled().or_else(|_| runtime()),
    }
}

impl MetalContext {
    /// `Ok(None)` when no matching Metal device exists.
    pub fn new(opts: &MetalOptions) -> MkResult<Option<Self>> {
        autoreleasepool(|_| {
            let Some(device) = select_device(&opts.device) else {
                return Ok(None);
            };
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| MkError::Backend("cannot create a command queue".into()))?;
            queue.setLabel(Some(&NSString::from_str("mklight")));
            let (library, origin, library_build) =
                load_library(&device, opts.library, opts.precision)?;
            Ok(Some(MetalContext {
                device,
                queue,
                library,
                origin,
                library_build,
                pipelines: Mutex::new(HashMap::new()),
            }))
        })
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
    pub fn library_origin(&self) -> LibraryOrigin {
        self.origin
    }
    /// How the shader library was built (origin, MSL version, flags or
    /// compile options), recorded in `PrecisionDescriptor::kernel_build`.
    pub fn library_build(&self) -> &str {
        &self.library_build
    }

    pub fn buffer<T: Pod>(&self, data: &[T]) -> MkResult<SharedBuffer<T>> {
        SharedBuffer::from_slice(&self.device, data)
    }
    pub fn buffer_zeroed<T: Pod>(&self, len: usize) -> MkResult<SharedBuffer<T>> {
        SharedBuffer::zeroed(&self.device, len)
    }
    /// See [`SharedBuffer::filled`].
    pub fn buffer_filled<T: Pod>(
        &self,
        len: usize,
        fill: impl FnOnce(&mut [T]),
    ) -> MkResult<SharedBuffer<T>> {
        SharedBuffer::filled(&self.device, len, fill)
    }

    /// The compute pipeline for `key`, compiled once and cached.
    pub fn pipeline(&self, key: &PipelineKey) -> MkResult<Pso> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(key) {
            return Ok(p.clone());
        }
        let pso = autoreleasepool(|_| -> MkResult<Pso> {
            let name = NSString::from_str(key.function);
            let function = if key.constants.is_empty() {
                self.library.newFunctionWithName(&name).ok_or_else(|| {
                    MkError::Backend(format!("kernel '{}' not in library", key.function))
                })?
            } else {
                let values = MTLFunctionConstantValues::new();
                for c in &key.constants {
                    // SAFETY: each pointer refers to a live local of the
                    // matching type for the duration of the call; Metal
                    // copies the value.
                    unsafe {
                        match *c {
                            FunctionConstant::U32 { index, value } => values
                                .setConstantValue_type_atIndex(
                                    NonNull::from(&value).cast::<c_void>(),
                                    MTLDataType::UInt,
                                    index as usize,
                                ),
                            FunctionConstant::Bool { index, value } => values
                                .setConstantValue_type_atIndex(
                                    NonNull::from(&value).cast::<c_void>(),
                                    MTLDataType::Bool,
                                    index as usize,
                                ),
                        }
                    }
                }
                self.library
                    .newFunctionWithName_constantValues_error(&name, &values)
                    .map_err(|e| ns_error(&format!("specialising '{}'", key.function), &e))?
            };
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| ns_error(&format!("pipeline '{}'", key.function), &e))
        })?;
        // Jobs retain their own pipeline objects across cache eviction.
        if cache.len() == 128 {
            cache.clear();
        }
        cache.insert(key.clone(), pso.clone());
        Ok(pso)
    }

    /// Encode one compute command buffer with `encode`, commit it and wait for
    /// completion. Buffers the closure borrows stay borrowed until the GPU has
    /// finished; buffers a kernel writes are bound with
    /// [`Encoder::set_buffer_mut`], which needs `&mut`, so no host reference
    /// to their contents can exist while the GPU writes them.
    ///
    /// # Safety
    ///
    /// For every dispatch encoded by `encode`, the kernel's accesses implied
    /// by its parameters (counts, offsets, grid size) must stay inside the
    /// bound buffers (measured from the bound element offset), and the kernel
    /// may write only buffers bound with `set_buffer_mut`.
    pub(crate) unsafe fn run(
        &self,
        label: &str,
        encode: impl FnOnce(&Encoder<'_>) -> MkResult<()>,
    ) -> MkResult<CommandStats> {
        autoreleasepool(|_| {
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| MkError::Backend("cannot create a command buffer".into()))?;
            cb.setLabel(Some(&NSString::from_str(label)));
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MkError::Backend("cannot create a compute encoder".into()))?;
            let r = encode(&Encoder {
                enc: &enc,
                work: &Cell::new(0),
                deferred: None,
            });
            enc.endEncoding();
            r?;
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into());
                return Err(MkError::Backend(format!("command buffer '{label}': {msg}")));
            }
            let (t0, t1) = (cb.GPUStartTime(), cb.GPUEndTime());
            let gpu_seconds = (t1 - t0).max(0.0);
            add_device_time(gpu_seconds);
            Ok(CommandStats {
                command_buffers: 1,
                gpu_seconds,
            })
        })
    }
}

/// A serial compute encoder borrowed while a synchronous or deferred batch is encoded.
pub(crate) struct Encoder<'a> {
    enc: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    work: &'a Cell<usize>,
    deferred: Option<&'a RefCell<Vec<crate::buffer::DeviceAccess>>>,
}

impl Encoder<'_> {
    pub(crate) fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.enc.setComputePipelineState(pso);
    }
    fn bind<T: Pod>(&self, index: usize, buf: &SharedBuffer<T>, offset: usize) {
        assert!(offset <= buf.len(), "buffer offset out of range");
        // SAFETY: the buffer is alive for the whole command buffer (borrowed
        // by the `run` closure, and retained by the command buffer), and the
        // offset is within it. Access bounds are `run`'s safety contract.
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(buf.raw()), offset * size_of::<T>(), index)
        };
    }
    /// Bind `buf`, which the kernel only reads, at element `offset`.
    /// A synchronous command commits before any pending deferred batch, so it
    /// must not read storage that batch writes (it would see stale data).
    pub(crate) fn set_buffer<T: Pod>(&self, index: usize, buf: &SharedBuffer<T>, offset: usize) {
        match self.deferred {
            Some(accesses) => accesses.borrow_mut().push(buf.device_access(false)),
            None => buf.assert_settled(false),
        }
        self.bind(index, buf, offset);
    }
    /// Bind `buf`, which the kernel writes, at element `offset`. The `&mut`
    /// borrow excludes host access while encoding. Deferred bindings additionally
    /// register a device-access guard until completion or abandonment.
    pub(crate) fn set_buffer_mut<T: Pod>(
        &self,
        index: usize,
        buf: &mut SharedBuffer<T>,
        offset: usize,
    ) {
        match self.deferred {
            Some(accesses) => accesses.borrow_mut().push(buf.device_access(true)),
            None => buf.assert_settled(true),
        }
        self.bind(index, buf, offset);
    }
    /// Copy a small parameter block into the argument table.
    pub(crate) fn set_bytes<T: Pod>(&self, index: usize, value: &T) {
        // SAFETY: `value` is a live Pod of the given size; Metal copies it.
        unsafe {
            self.enc.setBytes_length_atIndex(
                NonNull::from(value).cast::<c_void>(),
                size_of::<T>(),
                index,
            )
        };
    }
    /// `groups` threadgroups of `threads` threads each.
    pub(crate) fn dispatch_groups(&self, groups: [usize; 3], threads: usize) {
        self.work.set(
            self.work.get().saturating_add(
                groups
                    .into_iter()
                    .product::<usize>()
                    .saturating_mul(threads)
                    .max(1 << 15),
            ),
        );
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups[0],
                height: groups[1],
                depth: groups[2],
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        if self.deferred.is_some() {
            self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }
    /// At least `n` threads in 1-D, in uniform threadgroups (kernels check
    /// bounds), sized for the pipeline.
    pub(crate) fn dispatch_1d(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>, n: usize) {
        let w = pso.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        self.set_pipeline(pso);
        self.dispatch_groups([n.div_ceil(w).max(1), 1, 1], w);
    }
}

/// One worker's uncommitted resident commands. No command is left in flight:
/// flush commits and waits; dropping an unfinished batch abandons its encoding.
/// The executor retains recycled leases until flush and publishes only afterwards.
pub(crate) struct CommandBatch {
    pending: Option<BatchEncoding>,
    work_budget: usize,
    pub(crate) completed: CommandStats,
    /// Device seconds of a submission completed by the latest `encode` call
    /// or a later `flush` (timing feedback for calibrated tiles).
    pub(crate) observed: Option<f64>,
    pub(crate) max_seconds: f64,
}
struct BatchEncoding {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    accesses: RefCell<Vec<crate::buffer::DeviceAccess>>,
    work: Cell<usize>,
    start: Instant,
}
impl CommandBatch {
    pub(crate) fn new(work_budget: usize) -> Self {
        Self {
            pending: None,
            work_budget,
            completed: CommandStats::default(),
            observed: None,
            max_seconds: 0.,
        }
    }
    /// Same kernel bounds contract as `run`. Host slices require `flush` first.
    pub(crate) unsafe fn encode(
        &mut self,
        gpu: &MetalContext,
        label: &str,
        encode: impl FnOnce(&Encoder<'_>) -> MkResult<()>,
    ) -> MkResult<CommandStats> {
        self.observed = None;
        if self.pending.is_none() {
            let cb = gpu
                .queue
                .commandBuffer()
                .ok_or_else(|| MkError::Backend("cannot create a command buffer".into()))?;
            cb.setLabel(Some(&NSString::from_str(label)));
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MkError::Backend("cannot create a compute encoder".into()))?;
            self.pending = Some(BatchEncoding {
                cb,
                enc,
                accesses: RefCell::new(vec![]),
                work: Cell::new(0),
                start: Instant::now(),
            });
        }
        let b = self.pending.as_ref().unwrap();
        encode(&Encoder {
            enc: &b.enc,
            work: &b.work,
            deferred: Some(&b.accesses),
        })?;
        // Work bound supplements the existing package-specific weighted tiles.
        // Each dispatch costs at least 2^15 units, bounding tiny expensive
        // dispatch chains to 32 per batch. Encoding time is a second bound.
        if trace_dispatches() {
            let label = label.to_string();
            let stats = self.flush()?;
            eprintln!("MKL_DISPATCH {label} {:.6} ms", stats.gpu_seconds * 1e3);
            return Ok(stats);
        }
        if b.work.get() >= self.work_budget || b.start.elapsed().as_secs_f64() >= 0.010 {
            self.flush()
        } else {
            Ok(CommandStats::default())
        }
    }
    pub(crate) fn flush(&mut self) -> MkResult<CommandStats> {
        let Some(b) = self.pending.take() else {
            return Ok(CommandStats::default());
        };
        b.enc.endEncoding();
        let start = Instant::now();
        b.cb.commit();
        b.cb.waitUntilCompleted();
        self.max_seconds = self.max_seconds.max(start.elapsed().as_secs_f64());
        if b.cb.status() == MTLCommandBufferStatus::Error {
            let msg =
                b.cb.error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into());
            return Err(MkError::Backend(format!("fused command buffer: {msg}")));
        }
        let (a, z) = (b.cb.GPUStartTime(), b.cb.GPUEndTime());
        let stats = CommandStats {
            gpu_seconds: (z - a).max(0.),
            command_buffers: 1,
        };
        add_device_time(stats.gpu_seconds);
        self.completed.gpu_seconds += stats.gpu_seconds;
        self.completed.command_buffers += 1;
        self.observed = Some(stats.gpu_seconds);
        Ok(stats)
    }
}

impl Drop for CommandBatch {
    fn drop(&mut self) {
        if let Some(b) = self.pending.take() {
            b.enc.endEncoding();
        }
    }
}
