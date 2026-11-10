//! Metal device, queue, shader library and pipeline cache, plus synchronous
//! command-buffer execution and the resident executor's bounded asynchronous
//! command batch, with error and timing reporting.
//!
//! The public surface is safe: device and library information, buffer
//! allocation and pipeline compilation. Encoding and dispatching kernels
//! ([`MetalContext::run`], [`Encoder`]) is crate-private and `unsafe`,
//! because a kernel indexes buffers by the parameters it is given, which the
//! type system cannot relate to the bound buffers' lengths. Solvers reach the
//! GPU only through `ComputeBackend`.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use bytemuck::Pod;
use dispatch2::DispatchData;
use mklight_core::{MkError, MkResult};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::{NSArray, NSError, NSProcessInfo, NSString, NSURL};
use objc2_metal::{
    MTLBarrierScope, MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues,
    MTLLanguageVersion, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode, MTLPipelineOption,
    MTLSize,
};

use crate::buffer::SharedBuffer;
use crate::shaders;

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
}

/// Pipeline cache key: kernel name plus its function-constant values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub function: Cow<'static, str>,
    pub constants: Vec<FunctionConstant>,
}

impl PipelineKey {
    pub fn plain(function: impl Into<Cow<'static, str>>) -> Self {
        PipelineKey {
            function: function.into(),
            constants: Vec::new(),
        }
    }
    pub fn with_u32(function: impl Into<Cow<'static, str>>, index: u16, value: u32) -> Self {
        PipelineKey {
            function: function.into(),
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
type Archive = Retained<ProtocolObject<dyn MTLBinaryArchive>>;
/// One cached pipeline, created under its own lock: concurrent requests for
/// a key share one creation while requests for other keys proceed.
type PipelineSlot = Arc<Mutex<Option<Pso>>>;

/// The saved compiled pipelines of this device and shader library (a Metal
/// binary archive), so a later process creates them without compiling.
struct PipelineArchive {
    archive: Archive,
    path: PathBuf,
    /// Pipelines added since it was loaded or last saved.
    added: AtomicUsize,
}

// SAFETY: Metal binary archives are thread-safe ("MTLBinaryArchive", Apple
// Metal documentation: adding functions and creating pipelines from an
// archive may happen concurrently); the other fields are Send + Sync.
unsafe impl Send for PipelineArchive {}
unsafe impl Sync for PipelineArchive {}

/// Outcome of [`MetalContext::prepare_pipelines`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PipelinePreparation {
    /// Pipelines prepared (every kernel and its common specialisations).
    pub(crate) pipelines: usize,
    /// Of them, created from the saved archive without compiling.
    pub(crate) from_archive: usize,
    /// Wall time, seconds.
    pub(crate) seconds: f64,
}

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    origin: LibraryOrigin,
    library_build: String,
    pipelines: Mutex<HashMap<PipelineKey, PipelineSlot>>,
    archive: Mutex<Option<Arc<PipelineArchive>>>,
}

/// Run the calling thread at user-initiated quality of service: Metal's
/// compiler service schedules a request at its sender's class, and the
/// preparation is work the user waits for.
fn raise_qos() {
    // <pthread/qos.h>: QOS_CLASS_USER_INITIATED.
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: changes only the calling thread's scheduling class.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

/// File-name prefix of pipeline archives and their temporary files
/// ([`MetalContext::attach_pipeline_archive`]).
const ARCHIVE_PREFIX: &str = "pipelines-";
/// Time since an archive was last loaded or written beyond which a process
/// using another archive removes it.
const ARCHIVE_UNUSED: Duration = Duration::from_secs(30 * 24 * 3600);
/// Temporary archive files written by this process (part of their names).
static ARCHIVE_SAVES: AtomicUsize = AtomicUsize::new(0);

/// File name of the archive of one GPU, system version and shader library:
/// the GPU's name and a hash of the rest, so a new build, a system update or
/// another GPU never reads an archive made for something else.
fn archive_file_name(gpu: &str, system: &str, library_build: &str, library: &[u8]) -> String {
    let gpu: String = gpu
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    // 0xff never occurs in UTF-8 text: it separates the three parts.
    let parts = [system.as_bytes(), library_build.as_bytes(), library];
    for b in parts.iter().flat_map(|p| p.iter().copied().chain([0xff])) {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    format!("{ARCHIVE_PREFIX}{gpu}-{h:016x}.metalarchive")
}

/// Remove the archives and temporary files in `dir`, other than `keep`,
/// that nothing has loaded or written for [`ARCHIVE_UNUSED`]. Newer ones
/// belong to builds, libraries or devices that may still be in use (a debug
/// and a release build share the directory) or to a process saving now.
fn remove_unused_archives(dir: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let unused = (e.metadata().and_then(|m| m.modified()).ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > ARCHIVE_UNUSED);
        if name.starts_with(ARCHIVE_PREFIX) && name != keep && unused {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// The specialisations [`MetalContext::prepare_pipelines`] creates besides
/// every unspecialised kernel: (function, constant index, value). Rarer
/// combinations compile on first use (macOS's shader cache keeps them).
const PREPARED_PIPELINES: &[(&str, u16, u32)] = {
    use mklight_gpu::consts::*;
    const GHOST: u16 = MKL_GHOST_FEATURES_CONSTANT as u16;
    const STRAY: u16 = MKL_STRAY_FEATURES_CONSTANT as u16;
    &[
        // DfOp: the numerics probe and diagnostics.
        ("mkl_df_ops", 0, 0),
        ("mkl_df_ops", 0, 1),
        ("mkl_df_ops", 0, 2),
        ("mkl_df_ops", 0, 3),
        ("mkl_df_ops", 0, 4),
        ("mkl_df_ops", 0, 5),
        ("mkl_df_ops", 0, 6),
        ("mkl_df_ops", 0, 7),
        ("mkl_df_ops", 0, 8),
        ("mkl_df_ops", 0, 9),
        // ReduceMode.
        ("mkl_reduce_pass1", 1, MKL_REDUCE_PLAIN),
        ("mkl_reduce_pass1", 1, MKL_REDUCE_KAHAN),
        ("mkl_reduce_pass1", 1, MKL_REDUCE_DF),
        ("mkl_reduce_pass2", 1, MKL_REDUCE_PLAIN),
        ("mkl_reduce_pass2", 1, MKL_REDUCE_KAHAN),
        ("mkl_reduce_pass2", 1, MKL_REDUCE_DF),
        // Ghost trace: geometry, coatings, thin films.
        ("mkl_ghost_trace", GHOST, 0),
        ("mkl_ghost_trace", GHOST, MKL_GHOST_FEATURE_COATINGS),
        (
            "mkl_ghost_trace",
            GHOST,
            MKL_GHOST_FEATURE_COATINGS | MKL_GHOST_FEATURE_FILMS,
        ),
        // Stray transport, weighted mode: plain, Lambertian and fitted
        // scatter, each with and without aspheres.
        ("mkl_stray_wave_cast", STRAY, 1),
        ("mkl_stray_wave_cast", STRAY, 1 | 32),
        ("mkl_stray_wave_cast", STRAY, 1 | 32 | 64),
        ("mkl_stray_wave_cast", STRAY, 1 | 8),
        ("mkl_stray_wave_cast", STRAY, 1 | 32 | 8),
        ("mkl_stray_wave_cast", STRAY, 1 | 32 | 64 | 8),
        ("mkl_stray_wave_interact", STRAY, 1),
        ("mkl_stray_wave_interact", STRAY, 1 | 32),
        ("mkl_stray_wave_interact", STRAY, 1 | 32 | 64),
        ("mkl_stray_wave_interact", STRAY, 1 | 8),
        ("mkl_stray_wave_interact", STRAY, 1 | 32 | 8),
        ("mkl_stray_wave_interact", STRAY, 1 | 32 | 64 | 8),
    ]
};

/// Pipelines kept in memory; compiled ones also live in the archive and the
/// system shader cache, so eviction costs a lookup, not a compile.
const PIPELINE_CACHE: usize = 512;

fn ns_error(what: &str, e: &NSError) -> MkError {
    MkError::Backend(format!("{what}: {}", e.localizedDescription()))
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
    /// The system default device; `Ok(None)` when this machine has none.
    pub fn new(opts: &MetalOptions) -> MkResult<Option<Self>> {
        autoreleasepool(|_| {
            let Some(device) = MTLCreateSystemDefaultDevice() else {
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
                archive: Mutex::new(None),
            }))
        })
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
    /// Bytes the device currently has allocated (all of this process's
    /// buffers, pools and caches on it).
    pub fn allocated_bytes(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }
    /// Names of every function in the shader library.
    pub fn function_names(&self) -> Vec<String> {
        self.library
            .functionNames()
            .iter()
            .map(|n| n.to_string())
            .collect()
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

    /// The compute pipeline for `key`, created once and cached. With a
    /// pipeline archive attached it comes from the archive when present, and
    /// is added to it otherwise. Concurrent requests for one key wait for a
    /// single creation. The cache lock is not held while compiling, so a job
    /// does not wait behind another kernel's compilation.
    pub fn pipeline(&self, key: &PipelineKey) -> MkResult<Pso> {
        Ok(self.cached_pipeline(key, false)?.0)
    }

    /// [`Self::pipeline`], and whether this call created it from the
    /// archive. With `renew` a cached pipeline is created again, so that one
    /// made before the archive was attached comes from it or is added to it.
    fn cached_pipeline(&self, key: &PipelineKey, renew: bool) -> MkResult<(Pso, bool)> {
        let slot = {
            let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            // Jobs retain their own pipeline objects across cache eviction.
            if cache.len() >= PIPELINE_CACHE && !cache.contains_key(key) {
                cache.clear();
            }
            cache.entry(key.clone()).or_default().clone()
        };
        // Only this key's lock is held from here on. A failed creation
        // leaves the slot empty, so the next request tries again.
        let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pso) = &*slot
            && !renew
        {
            return Ok((pso.clone(), false));
        }
        let (pso, from_archive) = self.create_pipeline(key)?;
        *slot = Some(pso.clone());
        Ok((pso, from_archive))
    }

    /// The specialised function of `key`.
    fn function(&self, key: &PipelineKey) -> MkResult<Retained<ProtocolObject<dyn MTLFunction>>> {
        let name = NSString::from_str(&key.function);
        if key.constants.is_empty() {
            return self.library.newFunctionWithName(&name).ok_or_else(|| {
                MkError::Backend(format!("kernel '{}' not in library", key.function))
            });
        }
        let values = MTLFunctionConstantValues::new();
        for c in &key.constants {
            let FunctionConstant::U32 { index, value } = *c;
            // SAFETY: the pointer refers to a live local of the stated type
            // for the duration of the call; Metal copies the value.
            unsafe {
                values.setConstantValue_type_atIndex(
                    NonNull::from(&value).cast::<c_void>(),
                    MTLDataType::UInt,
                    index as usize,
                )
            }
        }
        self.library
            .newFunctionWithName_constantValues_error(&name, &values)
            .map_err(|e| ns_error(&format!("specialising '{}'", key.function), &e))
    }

    /// Create the pipeline of `key`: from the archive when it holds it
    /// (`true`), otherwise compiled and added to the archive.
    fn create_pipeline(&self, key: &PipelineKey) -> MkResult<(Pso, bool)> {
        let archive = self
            .archive
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        autoreleasepool(|_| {
            let function = self.function(key)?;
            let Some(archive) = archive else {
                let pso = self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| ns_error(&format!("pipeline '{}'", key.function), &e))?;
                return Ok((pso, false));
            };
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&function));
            desc.setBinaryArchives(Some(&NSArray::from_retained_slice(std::slice::from_ref(
                &archive.archive,
            ))));
            let create = |options| {
                self.device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &desc, options, None,
                    )
            };
            if let Ok(pso) = create(MTLPipelineOption::FailOnBinaryArchiveMiss) {
                return Ok((pso, true));
            }
            let pso = create(MTLPipelineOption::None)
                .map_err(|e| ns_error(&format!("pipeline '{}'", key.function), &e))?;
            // A failed addition only costs a compile in a later process.
            if archive
                .archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .is_ok()
            {
                archive.added.fetch_add(1, Ordering::Relaxed);
            }
            Ok((pso, false))
        })
    }

    /// Attach the pipeline archive of this device, system version and
    /// shader library in `dir` (created if missing), loading it when one was
    /// saved. Other archives in `dir` are removed once unused for
    /// [`ARCHIVE_UNUSED`] ([`remove_unused_archives`]). Pipelines created
    /// from now on come from the archive or are added to it;
    /// [`save_pipeline_archive`] writes it.
    ///
    /// [`save_pipeline_archive`]: Self::save_pipeline_archive
    pub(crate) fn attach_pipeline_archive(&self, dir: &Path) -> MkResult<()> {
        std::fs::create_dir_all(dir)
            .map_err(|e| MkError::Backend(format!("pipeline cache {}: {e}", dir.display())))?;
        let name = self.archive_name();
        let path = dir.join(&name);
        remove_unused_archives(dir, &name);
        let open = |url: Option<&NSURL>| {
            let desc = MTLBinaryArchiveDescriptor::new();
            desc.setUrl(url);
            self.device.newBinaryArchiveWithDescriptor_error(&desc)
        };
        let (archive, loaded) = autoreleasepool(|_| {
            let url = path
                .exists()
                .then(|| NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy())));
            // An unreadable archive is replaced by an empty one.
            match url.as_deref().map(|u| open(Some(u))) {
                Some(Ok(a)) => Ok((a, true)),
                _ => open(None).map(|a| (a, false)),
            }
        })
        .map_err(|e| ns_error("creating the pipeline archive", &e))?;
        if loaded {
            // Loading counts as use for other processes' clean-up; a failure
            // only makes them remove the archive earlier.
            let _ = std::fs::File::options()
                .append(true)
                .open(&path)
                .and_then(|f| f.set_modified(SystemTime::now()));
        }
        *self.archive.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(PipelineArchive {
            archive,
            path,
            added: AtomicUsize::new(0),
        }));
        Ok(())
    }

    /// Write the attached pipeline archive if pipelines were added to it
    /// (atomically: a temporary file of this process renamed over the old
    /// one). Returns whether it wrote. After a failure the additions stay
    /// counted, so a later call writes them.
    pub(crate) fn save_pipeline_archive(&self) -> MkResult<bool> {
        let Some(a) = self
            .archive
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        else {
            return Ok(false);
        };
        let added = a.added.load(Ordering::Relaxed);
        if added == 0 {
            return Ok(false);
        }
        let mut tmp = a.path.clone().into_os_string();
        tmp.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            ARCHIVE_SAVES.fetch_add(1, Ordering::Relaxed)
        ));
        let tmp = PathBuf::from(tmp);
        let written = autoreleasepool(|_| {
            let url = NSURL::fileURLWithPath(&NSString::from_str(&tmp.to_string_lossy()));
            a.archive
                .serializeToURL_error(&url)
                .map_err(|e| ns_error("saving the pipeline archive", &e))
        })
        .and_then(|()| {
            std::fs::rename(&tmp, &a.path)
                .map_err(|e| MkError::Backend(format!("pipeline cache {}: {e}", a.path.display())))
        });
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        // Pipelines added while it was written stay counted for the next save.
        a.added.fetch_sub(added, Ordering::Relaxed);
        Ok(true)
    }

    /// File name of this context's archive ([`archive_file_name`]).
    fn archive_name(&self) -> String {
        archive_file_name(
            &self.device.name().to_string(),
            &NSProcessInfo::processInfo()
                .operatingSystemVersionString()
                .to_string(),
            &self.library_build,
            self.library_identity(),
        )
    }

    /// Bytes identifying the shader library: the embedded metallib, or the
    /// source a runtime library was compiled from.
    fn library_identity(&self) -> &'static [u8] {
        match self.origin {
            LibraryOrigin::Precompiled => if self.library_build.starts_with("pure_f32") {
                shaders::METALLIB_F32
            } else {
                shaders::METALLIB
            }
            .unwrap_or_default(),
            LibraryOrigin::RuntimeSource => shaders::SOURCE.as_bytes(),
        }
    }

    /// Create every pipeline the backend uses ([`PREPARED_PIPELINES`]): all
    /// kernels and the common specialisations of the specialised ones, on
    /// `threads` threads, reporting `(done, total)` after each. With an
    /// archive attached, the pipelines come from it or are compiled into it;
    /// the caller saves it. Stops early, without error, when `cancel` is set.
    /// A pipeline that cannot be created does not stop the others; the first
    /// such error is returned once all have been tried.
    pub(crate) fn prepare_pipelines(
        &self,
        threads: usize,
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &AtomicBool,
    ) -> MkResult<PipelinePreparation> {
        let start = Instant::now();
        // Every pipeline goes through an attached archive, also those jobs
        // have already created, so the saved archive holds them all.
        let archived = (self.archive.lock())
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        let keys = self.prepared_keys();
        let total = keys.len();
        let next = AtomicUsize::new(0);
        let done = Mutex::new(0usize);
        let from_archive = AtomicUsize::new(0);
        let error: Mutex<Option<MkError>> = Mutex::new(None);
        progress(0, total);
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    raise_qos();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= total || cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        match self.cached_pipeline(&keys[i], archived) {
                            Ok((_, hit)) => {
                                from_archive.fetch_add(hit as usize, Ordering::Relaxed);
                            }
                            Err(e) => {
                                error
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .get_or_insert(e);
                            }
                        }
                        // Counted and reported under one lock, so the
                        // reported count never steps backwards.
                        let mut done = done.lock().unwrap_or_else(|e| e.into_inner());
                        *done += 1;
                        progress(*done, total);
                    }
                });
            }
        });
        if let Some(e) = error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        Ok(PipelinePreparation {
            pipelines: done.into_inner().unwrap_or_else(|e| e.into_inner()),
            from_archive: from_archive.into_inner(),
            seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// Every library function without specialisation constants, plus the
    /// listed specialisations.
    fn prepared_keys(&self) -> Vec<PipelineKey> {
        let specialised: Vec<&str> = PREPARED_PIPELINES.iter().map(|(f, _, _)| *f).collect();
        let mut names = self.function_names();
        names.sort();
        let mut keys: Vec<PipelineKey> = names
            .into_iter()
            .filter(|n| !specialised.contains(&n.as_str()))
            .map(PipelineKey::plain)
            .collect();
        keys.extend(
            PREPARED_PIPELINES
                .iter()
                .map(|&(f, index, value)| PipelineKey::with_u32(f, index, value)),
        );
        keys
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
        let w = Self::width(pso);
        self.set_pipeline(pso);
        self.dispatch_groups([n.div_ceil(w).max(1), 1, 1], w);
    }
    /// Threads per threadgroup of [`Self::dispatch_1d`] and
    /// [`Self::dispatch_indirect`] for `pso`.
    pub(crate) fn width(pso: &ProtocolObject<dyn MTLComputePipelineState>) -> usize {
        pso.maxTotalThreadsPerThreadgroup().clamp(1, 256)
    }
    /// 1-D dispatch whose threadgroup count a device kernel wrote to `args`
    /// (three u32 words at element `offset`), in threadgroups of
    /// [`Self::width`] threads; zero threadgroups dispatch nothing.
    pub(crate) fn dispatch_indirect(
        &self,
        pso: &ProtocolObject<dyn MTLComputePipelineState>,
        args: &SharedBuffer<u32>,
        offset: usize,
    ) {
        assert!(offset + 3 <= args.len(), "indirect arguments out of range");
        let w = Self::width(pso);
        self.set_pipeline(pso);
        // The argument buffer is read by the command, not bound to a slot.
        match self.deferred {
            Some(accesses) => accesses.borrow_mut().push(args.device_access(false)),
            None => args.assert_settled(false),
        }
        self.work.set(self.work.get().saturating_add(1 << 15));
        // SAFETY: `args` holds three words at `offset` (asserted above) and
        // its access guard spans the command; the kernel checks bounds.
        unsafe {
            self.enc
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    args.raw(),
                    offset * 4,
                    MTLSize {
                        width: w,
                        height: 1,
                        depth: 1,
                    },
                )
        };
        if self.deferred.is_some() {
            self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }
}

/// Committed, unfinished command buffers per worker beyond which committing
/// another waits for the oldest. The device always has the next command
/// queued while the host encodes, and a cancelled or failed plan waits for at
/// most this many bounded commands before its storage is released.
/// `MKLIGHT_METAL_IN_FLIGHT=0` completes every command before the next one is
/// encoded (the synchronous executor, for diagnostics and A/B runs).
const IN_FLIGHT: usize = 2;
/// Committed, unfinished command buffers of all workers beyond which a
/// worker keeps only its newest one in flight. Several plans side by side
/// (candidate solves, concurrent wave chains) then queue about as deep as
/// the synchronous executor did, so the device still yields between
/// commands to the window server.
const IN_FLIGHT_TOTAL: usize = 3;
static IN_FLIGHT_NOW: AtomicUsize = AtomicUsize::new(0);
/// Device seconds beyond which a command buffer risks the driver's
/// interactivity watchdog (counted in `PlanStats::long_commands`).
const LONG_COMMAND: f64 = 0.050;
fn in_flight_window() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("MKLIGHT_METAL_IN_FLIGHT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(IN_FLIGHT)
            .min(8)
    })
}

/// One worker's resident commands: the command buffer being encoded and the
/// committed ones still running (at most [`in_flight_window`]). Committing
/// never waits for its own command; the host waits only where it needs
/// device results (`complete`, `flush`), when the window is full, and when
/// the batch is dropped. Every committed command keeps the device-access
/// guards of its bindings until it has completed, so no host slice, pool
/// reuse or publication can observe storage it may still access. Dropping
/// the batch abandons the encoding and waits for the committed commands.
pub(crate) struct CommandBatch {
    pending: Option<BatchEncoding>,
    in_flight: VecDeque<Submitted>,
    window: usize,
    work_budget: usize,
    /// Tickets issued: the n-th committed command buffer has ticket n.
    submitted: u64,
    /// Device seconds of the latest completed command buffers, by ticket.
    recent: VecDeque<(u64, f64)>,
    pub(crate) completed: CommandStats,
    /// Device seconds of the newest submission once it has completed while
    /// nothing newer was encoded (timing feedback for calibrated tiles).
    pub(crate) observed: Option<f64>,
    /// Longest device time of one command buffer (the interactivity bound).
    pub(crate) max_seconds: f64,
    /// Command buffers that ran longer than [`LONG_COMMAND`] on the device.
    pub(crate) long_commands: u32,
    /// Host waits for device completion, and their wall seconds.
    pub(crate) waits: u32,
    pub(crate) wait_seconds: f64,
}
struct BatchEncoding {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    accesses: RefCell<Vec<crate::buffer::DeviceAccess>>,
    work: Cell<usize>,
    start: Instant,
}
struct Submitted {
    _counted: InFlight,
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Released only after completion.
    accesses: Vec<crate::buffer::DeviceAccess>,
    ticket: u64,
}
/// One process-wide in-flight command buffer (released when its
/// `Submitted` drops, after completion).
struct InFlight;
impl InFlight {
    fn new() -> Self {
        IN_FLIGHT_NOW.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT_NOW.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Submitted {
    fn finished(&self) -> bool {
        matches!(
            self.cb.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    }
}
impl CommandBatch {
    pub(crate) fn new(work_budget: usize) -> Self {
        Self {
            pending: None,
            in_flight: VecDeque::new(),
            window: in_flight_window(),
            work_budget,
            submitted: 0,
            recent: VecDeque::new(),
            completed: CommandStats::default(),
            observed: None,
            max_seconds: 0.,
            long_commands: 0,
            waits: 0,
            wait_seconds: 0.,
        }
    }
    /// Same kernel bounds contract as `run`. Host slices of storage bound
    /// here require its command's completion (`complete` or `flush`) first.
    pub(crate) unsafe fn encode(
        &mut self,
        gpu: &MetalContext,
        label: &str,
        encode: impl FnOnce(&Encoder<'_>) -> MkResult<()>,
    ) -> MkResult<()> {
        self.observed = None;
        if self.pending.is_none() {
            self.poll()?;
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
        if b.work.get() >= self.work_budget || b.start.elapsed().as_secs_f64() >= 0.010 {
            self.submit()?;
        }
        Ok(())
    }
    /// The ticket of the command buffer holding the latest encoder call
    /// (the one being encoded, or the newest committed one).
    pub(crate) fn ticket(&self) -> u64 {
        self.submitted + u64::from(self.pending.is_some())
    }
    /// Commit the encoded commands without waiting for them; returns the
    /// ticket of the newest committed command buffer (0: none yet).
    pub(crate) fn submit(&mut self) -> MkResult<u64> {
        if let Some(b) = self.pending.take() {
            b.enc.endEncoding();
            b.cb.commit();
            self.submitted += 1;
            self.in_flight.push_back(Submitted {
                _counted: InFlight::new(),
                cb: b.cb,
                accesses: b.accesses.into_inner(),
                ticket: self.submitted,
            });
            while self.in_flight.len() > self.window
                || (self.in_flight.len() > 1
                    && IN_FLIGHT_NOW.load(Ordering::Acquire) > IN_FLIGHT_TOTAL)
            {
                self.retire(true)?;
            }
        }
        Ok(self.submitted)
    }
    /// Retire the oldest committed command once it has completed (waiting
    /// for it with `block`); false when there is none or it is still running.
    fn retire(&mut self, block: bool) -> MkResult<bool> {
        let Some(s) = self.in_flight.front() else {
            return Ok(false);
        };
        if !s.finished() {
            if !block {
                return Ok(false);
            }
            let start = Instant::now();
            s.cb.waitUntilCompleted();
            self.waits += 1;
            self.wait_seconds += start.elapsed().as_secs_f64();
        }
        let s = self.in_flight.pop_front().unwrap();
        if s.cb.status() == MTLCommandBufferStatus::Error {
            let msg =
                s.cb.error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into());
            return Err(MkError::Backend(format!("fused command buffer: {msg}")));
        }
        let (a, z) = (s.cb.GPUStartTime(), s.cb.GPUEndTime());
        let seconds = (z - a).max(0.);
        add_device_time(seconds);
        self.completed.gpu_seconds += seconds;
        self.completed.command_buffers += 1;
        self.max_seconds = self.max_seconds.max(seconds);
        self.long_commands += u32::from(seconds > LONG_COMMAND);
        if s.ticket == self.submitted && self.pending.is_none() {
            self.observed = Some(seconds);
        }
        if self.recent.len() == 8 {
            self.recent.pop_front();
        }
        self.recent.push_back((s.ticket, seconds));
        drop(s.accesses);
        Ok(true)
    }
    /// Retire the commands that have already completed, without waiting.
    pub(crate) fn poll(&mut self) -> MkResult<()> {
        while self.retire(false)? {}
        Ok(())
    }
    /// Wait until the command buffer with `ticket` (and every earlier one)
    /// has completed; returns its device seconds when still on record.
    pub(crate) fn complete(&mut self, ticket: u64) -> MkResult<Option<f64>> {
        while self.in_flight.front().is_some_and(|s| s.ticket <= ticket) {
            self.retire(true)?;
        }
        Ok(self
            .recent
            .iter()
            .find(|(t, _)| *t == ticket)
            .map(|(_, s)| *s))
    }
    /// Commit any encoded commands and wait for every committed one. Returns
    /// the device seconds of the newest command buffer, if one ran.
    pub(crate) fn flush(&mut self) -> MkResult<CommandStats> {
        let ticket = self.submit()?;
        let before = self.completed.command_buffers;
        while self.retire(true)? {}
        let ran = self.completed.command_buffers - before;
        Ok(CommandStats {
            gpu_seconds: if ran > 0 {
                self.recent
                    .iter()
                    .find(|(t, _)| *t == ticket)
                    .map_or(0., |(_, s)| *s)
            } else {
                0.
            },
            command_buffers: ran,
        })
    }
}

impl Drop for CommandBatch {
    fn drop(&mut self) {
        if let Some(b) = self.pending.take() {
            b.enc.endEncoding();
        }
        // Committed work cannot be withdrawn: its storage stays guarded
        // (unavailable to the host and the pool) until it has completed.
        for s in self.in_flight.drain(..) {
            s.cb.waitUntilCompleted();
            if s.cb.status() != MTLCommandBufferStatus::Error {
                add_device_time((s.cb.GPUEndTime() - s.cb.GPUStartTime()).max(0.));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> MetalContext {
        MetalContext::new(&MetalOptions::default())
            .unwrap()
            .expect("host Metal required")
    }

    /// A fresh, empty directory for one test.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mklight-metal-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn archive_name_follows_gpu_system_and_library() {
        let name = archive_file_name("Apple M2 Pro", "Version 15.1", "flags", b"library");
        assert!(name.starts_with("pipelines-Apple-M2-Pro-"), "{name}");
        assert!(name.ends_with(".metalarchive"), "{name}");
        assert_eq!(
            name,
            archive_file_name("Apple M2 Pro", "Version 15.1", "flags", b"library")
        );
        for other in [
            archive_file_name("Apple M3", "Version 15.1", "flags", b"library"),
            archive_file_name("Apple M2 Pro", "Version 15.2", "flags", b"library"),
            archive_file_name("Apple M2 Pro", "Version 15.1", "other", b"library"),
            archive_file_name("Apple M2 Pro", "Version 15.1", "flags", b"changed"),
            // The parts are hashed separately, not as one concatenation.
            archive_file_name("Apple M2 Pro", "Version 15.1f", "lags", b"library"),
        ] {
            assert_ne!(name, other);
        }
    }

    /// Only archives and temporary files unused for longer than
    /// `ARCHIVE_UNUSED` are removed, and never the archive in use.
    #[test]
    fn only_long_unused_archives_are_removed() {
        let dir = test_dir("unused");
        let old = SystemTime::now() - ARCHIVE_UNUSED - Duration::from_secs(3600);
        let recent = SystemTime::now() - ARCHIVE_UNUSED + Duration::from_secs(3600);
        for (name, modified) in [
            ("pipelines-mine.metalarchive", old),
            ("pipelines-old.metalarchive", old),
            ("pipelines-old.metalarchive.7.0.tmp", old),
            ("pipelines-recent.metalarchive", recent),
            ("pipelines-saving.metalarchive.7.0.tmp", SystemTime::now()),
            ("notes.txt", old),
        ] {
            let f = std::fs::File::create(dir.join(name)).unwrap();
            f.set_modified(modified).unwrap();
        }
        remove_unused_archives(&dir, "pipelines-mine.metalarchive");
        assert_eq!(
            file_names(&dir),
            [
                "notes.txt",
                "pipelines-mine.metalarchive",
                "pipelines-recent.metalarchive",
                "pipelines-saving.metalarchive.7.0.tmp",
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A saved archive serves another context of the same device and
    /// library without compiling; saving leaves no temporary file, writes
    /// only when pipelines were added, and loading marks the archive used.
    #[test]
    fn archive_is_saved_and_loaded() {
        let dir = test_dir("archive");
        let key = PipelineKey::plain("mkl_philox_stream");
        let first = context();
        first.attach_pipeline_archive(&dir).unwrap();
        assert!(!first.save_pipeline_archive().unwrap(), "nothing added");
        assert!(!first.cached_pipeline(&key, false).unwrap().1, "compiled");
        assert!(first.save_pipeline_archive().unwrap());
        assert_eq!(file_names(&dir), [first.archive_name()]);
        assert!(!first.save_pipeline_archive().unwrap(), "nothing new");

        let path = dir.join(first.archive_name());
        let old = SystemTime::now() - ARCHIVE_UNUSED - Duration::from_secs(3600);
        let file = std::fs::File::options().append(true).open(&path).unwrap();
        file.set_modified(old).unwrap();
        let second = context();
        second.attach_pipeline_archive(&dir).unwrap();
        let age = path.metadata().unwrap().modified().unwrap().elapsed();
        assert!(age.unwrap() < Duration::from_secs(3600), "marked used");
        assert!(
            second.cached_pipeline(&key, false).unwrap().1,
            "from the archive"
        );
        // A pipeline cached before the archive was attached is renewed
        // through it, and found in it once added.
        let early = PipelineKey::plain("mkl_sag_probe");
        let third = context();
        third.pipeline(&early).unwrap();
        third.attach_pipeline_archive(&dir).unwrap();
        assert!(!third.cached_pipeline(&early, false).unwrap().1, "cached");
        assert!(!third.cached_pipeline(&early, true).unwrap().1, "added");
        assert!(third.cached_pipeline(&early, true).unwrap().1, "found");
        assert!(third.save_pipeline_archive().unwrap());
        assert!(!second.save_pipeline_archive().unwrap(), "nothing added");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A save that fails keeps the additions counted: the next save writes
    /// them, and no temporary file is left behind.
    #[test]
    fn failed_archive_save_is_retried() {
        let dir = test_dir("retry");
        let ctx = context();
        ctx.attach_pipeline_archive(&dir).unwrap();
        ctx.pipeline(&PipelineKey::plain("mkl_philox_stream"))
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(ctx.save_pipeline_archive().is_err());
        std::fs::create_dir_all(&dir).unwrap();
        assert!(ctx.save_pipeline_archive().unwrap());
        assert_eq!(file_names(&dir), [ctx.archive_name()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Several threads requesting one uncached pipeline share a single
    /// creation: every caller gets the same pipeline object.
    #[test]
    fn concurrent_requests_create_a_pipeline_once() {
        let ctx = context();
        let key = PipelineKey::plain("mkl_sag_probe");
        let start = std::sync::Barrier::new(8);
        let got: Vec<Pso> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        start.wait();
                        ctx.pipeline(&key).unwrap()
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        for p in &got {
            assert_eq!(Retained::as_ptr(p), Retained::as_ptr(&got[0]));
        }
        assert_eq!(
            Retained::as_ptr(&ctx.pipeline(&key).unwrap()),
            Retained::as_ptr(&got[0])
        );
    }
}
