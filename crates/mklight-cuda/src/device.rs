//! [`Gpu`] on a CUDA device through the driver API (`nvcuda`, loaded at run
//! time), with kernels compiled by NVRTC (loaded at run time too) and cached
//! on disk ([`crate::kernels`]).
//!
//! A batch runs on one stream taken from a small pool, so batches of
//! different threads overlap: uploads, zeroed scratch and device-side slices
//! are stream-ordered allocations from the device's memory pool (kept
//! allocated between batches), every pass is one launch, copies and
//! readbacks follow, and the call returns when the stream has completed.
//! Host transfers go through page-locked staging buffers (one per stream),
//! so they run at full PCIe speed and asynchronously on the stream.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{result as cu, sys};
use mklight_core::{MkError, MkResult};
use mklight_gpu::{BackendKind, NumericsProbe};
use mklight_webgpu::device::*;
use mklight_webgpu::watchdog::Health;

use crate::kernels;

fn err(what: &str, e: impl std::fmt::Debug) -> MkError {
    MkError::Backend(format!("cuda: {what}: {e:?}"))
}

fn check(what: &str, r: sys::CUresult) -> MkResult<()> {
    r.result().map_err(|e| err(what, e))
}

/// A loaded compilation unit.
struct Module {
    module: sys::CUmodule,
    /// Read from the disk cache rather than compiled.
    from_cache: bool,
}

/// A compilation unit, loaded once.
type Unit = Arc<std::sync::OnceLock<Result<Arc<Module>, String>>>;

// CUDA handles are usable from any thread with the context current.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// Page-locked host memory of at least `bytes` bytes.
struct Pinned {
    ptr: *mut u8,
    bytes: usize,
}

impl Pinned {
    fn new(bytes: usize) -> MkResult<Self> {
        let bytes = bytes.next_power_of_two().max(1 << 20);
        let ptr = unsafe { cu::malloc_host(bytes, 0) }.map_err(|e| err("cuMemHostAlloc", e))?;
        Ok(Self {
            ptr: ptr as *mut u8,
            bytes,
        })
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::free_host(self.ptr as *mut c_void);
        }
    }
}

/// A stream with its staging buffer and timing events.
struct Lane {
    stream: sys::CUstream,
    staging: Option<Pinned>,
    start: sys::CUevent,
    end: sys::CUevent,
}

unsafe impl Send for Lane {}

pub struct CudaDevice {
    ctx: sys::CUcontext,
    summary: AdapterSummary,
    arch: (i32, i32),
    nvrtc: Option<(i32, i32)>,
    driver_version: i32,
    /// Where compiled units are kept (`None`: nowhere).
    cache_dir: Mutex<Option<PathBuf>>,
    /// Why the cache could not be written, if it could not.
    cache_warning: Mutex<Option<String>>,
    modules: Mutex<HashMap<String, Unit>>,
    blocks: Mutex<HashMap<Kernel, (sys::CUfunction, [u32; 3])>>,
    persistent: Mutex<HashMap<u64, (sys::CUdeviceptr, usize)>>,
    next_key: AtomicU64,
    lanes: Mutex<Vec<Lane>>,
    /// The stream persistent buffers are allocated and freed on (from the
    /// memory pool: plan mirrors come and go with every plan, and
    /// `cuMemFree` would synchronise the device).
    admin: Mutex<sys::CUstream>,
    /// Set once a sticky CUDA error has made the context unusable
    /// (`Gpu::unavailable`).
    health: Health,
}

/// Module statistics of this process: (compiled, loaded from the cache,
/// seconds spent compiling or reading).
static COMPILE_STATS: Mutex<(usize, usize, f64)> = Mutex::new((0, 0, 0.0));

/// (compiled, loaded from cache, seconds) of the kernel modules this process
/// has built so far (start-up measurements).
pub fn compile_stats() -> (usize, usize, f64) {
    *COMPILE_STATS.lock().unwrap()
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

/// Device time of completed batches in this process, nanoseconds.
static DEVICE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Sum of the device time of every completed batch in this process (from the
/// first transfer to the last readback on its stream; benchmarks only).
pub fn device_seconds() -> f64 {
    DEVICE_NANOS.load(Ordering::Relaxed) as f64 * 1e-9
}

static PROFILE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("MKLIGHT_CUDA_PROFILE").is_ok_and(|v| v == "1"));

/// Whether the CUDA driver library can be loaded.
pub fn driver_present() -> bool {
    unsafe { sys::is_culib_present() }
}

/// The kernel cache directory: `MKLIGHT_CUDA_CACHE`, else the platform's
/// local cache directory (`%LOCALAPPDATA%\mklight\cuda`,
/// `$XDG_CACHE_HOME/mklight/cuda` or `~/.cache/mklight/cuda`).
pub fn default_cache_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("MKLIGHT_CUDA_CACHE") {
        return (!d.is_empty() && d != "0").then(|| PathBuf::from(d));
    }
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("mklight").join("cuda"))
}

impl CudaDevice {
    /// Device 0, or `Ok(None)` without a driver or a device.
    pub fn open() -> MkResult<Option<Self>> {
        if !driver_present() {
            return Ok(None);
        }
        cu::init().map_err(|e| err("cuInit", e))?;
        if cu::device::get_count().map_err(|e| err("cuDeviceGetCount", e))? < 1 {
            return Ok(None);
        }
        let dev = cu::device::get(0).map_err(|e| err("cuDeviceGet", e))?;
        let ctx = unsafe { cu::primary_ctx::retain(dev) }.map_err(|e| err("primary context", e))?;
        unsafe { cu::ctx::set_current(ctx) }.map_err(|e| err("cuCtxSetCurrent", e))?;
        let attr =
            |a| unsafe { cu::device::get_attribute(dev, a) }.map_err(|e| err("attribute", e));
        use sys::CUdevice_attribute_enum as A;
        let arch = (
            attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        );
        let name = cu::device::get_name(dev).map_err(|e| err("cuDeviceGetName", e))?;
        let mut driver_version = 0;
        check("cuDriverGetVersion", unsafe {
            sys::cuDriverGetVersion(&mut driver_version)
        })?;
        // Keep freed stream-ordered allocations in the pool (batches allocate
        // their uploads and scratch there).
        unsafe {
            let pool = cu::device::get_default_mem_pool(dev).map_err(|e| err("mem pool", e))?;
            let mut threshold: u64 = u64::MAX;
            check(
                "cuMemPoolSetAttribute",
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut c_void,
                ),
            )?;
        }
        let admin = cu::stream::create(cu::stream::StreamKind::NonBlocking)
            .map_err(|e| err("cuStreamCreate", e))?;
        let nvrtc = crate::nvrtc::version();
        let l = mklight_webgpu::wgpu_device::design_limits();
        let summary = AdapterSummary {
            name,
            vendor: "NVIDIA".into(),
            backend: "CUDA".into(),
            driver: format!(
                "CUDA driver {}.{}, sm_{}{}",
                driver_version / 1000,
                driver_version % 1000 / 10,
                arch.0,
                arch.1
            ),
            integrated: false,
            // The WebGPU host operations chunk work by these limits; the
            // WebGPU defaults keep their chunking (and the kernels' index
            // arithmetic) unchanged.
            max_storage_buffer_binding_size: l.max_storage_buffer_binding_size as u64,
            max_buffer_size: l.max_buffer_size,
            max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage,
            max_compute_workgroup_size_x: l.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
            min_storage_buffer_offset_alignment: l.min_storage_buffer_offset_alignment,
        };
        Ok(Some(Self {
            ctx,
            summary,
            arch,
            nvrtc,
            driver_version,
            cache_dir: Mutex::new(default_cache_dir()),
            cache_warning: Mutex::new(None),
            modules: Mutex::new(HashMap::new()),
            blocks: Mutex::new(HashMap::new()),
            persistent: Mutex::new(HashMap::new()),
            next_key: AtomicU64::new(1),
            lanes: Mutex::new(Vec::new()),
            admin: Mutex::new(admin),
            health: Health::default(),
        }))
    }

    /// Compute capability (8, 9).
    pub fn arch(&self) -> (i32, i32) {
        self.arch
    }

    /// Use `dir` for the kernel cache (`None`: no cache). Units already
    /// loaded stay loaded.
    pub fn set_cache_dir(&self, dir: Option<PathBuf>) {
        *self.cache_dir.lock().unwrap() = dir;
    }

    fn bind(&self) -> MkResult<()> {
        unsafe { cu::ctx::set_current(self.ctx) }.map_err(|e| err("cuCtxSetCurrent", e))
    }

    /// The image of unit `name`: embedded at build time, else from the disk
    /// cache, else compiled by NVRTC now (and saved to the cache). The flag
    /// says whether it was not compiled now.
    fn image(&self, name: &str, source: &str) -> MkResult<(Vec<u8>, bool)> {
        let t = Instant::now();
        if let Some((target, image)) = kernels::embedded(name, self.arch) {
            let image = miniz_oxide::inflate::decompress_to_vec(image)
                .map_err(|e| MkError::Backend(format!("cuda: built-in {name}.{target}: {e:?}")))?;
            let mut s = COMPILE_STATS.lock().unwrap();
            s.1 += 1;
            s.2 += t.elapsed().as_secs_f64();
            return Ok((image, true));
        }
        let Some(nvrtc) = self.nvrtc else {
            return Err(MkError::Unsupported(format!(
                "cuda: no kernels for sm_{}{} were built in, and NVRTC (nvrtc64_*.dll / \
                 libnvrtc.so) is not available to compile them",
                self.arch.0, self.arch.1
            )));
        };
        let target = format!("sm_{}{}", self.arch.0, self.arch.1);
        let mut h = blake3::Hasher::new();
        h.update(kernels::digest().as_bytes());
        h.update(name.as_bytes());
        h.update(source.as_bytes());
        h.update(target.as_bytes());
        h.update(format!("{nvrtc:?}").as_bytes());
        let key = h.finalize().to_hex()[..20].to_string();
        let path = self
            .cache_dir
            .lock()
            .unwrap()
            .as_ref()
            .map(|d| d.join(format!("{name}-{key}.cubin")));
        if let Some(p) = &path
            && let Ok(b) = std::fs::read(p)
        {
            let mut s = COMPILE_STATS.lock().unwrap();
            s.1 += 1;
            s.2 += t.elapsed().as_secs_f64();
            return Ok((b, true));
        }
        let (cubin, log) = crate::nvrtc::compile(name, source, &target)
            .map_err(|e| MkError::Backend(format!("cuda: {e}")))?;
        if std::env::var_os("MKLIGHT_CUDA_LOG").is_some() {
            eprintln!(
                "cuda: compiled {name} in {:.2} s ({} KiB){}",
                t.elapsed().as_secs_f64(),
                cubin.len() / 1024,
                if log.trim().is_empty() {
                    String::new()
                } else {
                    format!("\n{log}")
                }
            );
        }
        if let Some(p) = &path {
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            let saved = std::fs::create_dir_all(p.parent().unwrap())
                .and_then(|()| std::fs::write(&tmp, &cubin))
                .and_then(|()| std::fs::rename(&tmp, p));
            if let Err(e) = saved {
                let _ = std::fs::remove_file(&tmp);
                *self.cache_warning.lock().unwrap() = Some(format!(
                    "the CUDA kernel cache {} could not be written: {e}",
                    p.display()
                ));
            }
        }
        let mut s = COMPILE_STATS.lock().unwrap();
        s.0 += 1;
        s.2 += t.elapsed().as_secs_f64();
        Ok((cubin, false))
    }

    /// The loaded unit holding `k` (compiled or read from the cache on first
    /// use; concurrent first uses wait for one compilation).
    fn unit(&self, k: Kernel) -> MkResult<Arc<Module>> {
        let (name, source) = kernels::unit(k.module, k.entry).ok_or_else(|| {
            MkError::Unsupported(format!("cuda: no kernel module '{}'", k.module))
        })?;
        let cell = self
            .modules
            .lock()
            .unwrap()
            .entry(name.clone())
            .or_default()
            .clone();
        cell.get_or_init(|| {
            let (cubin, from_cache) = self.image(&name, &source).map_err(|e| e.to_string())?;
            self.bind().map_err(|e| e.to_string())?;
            let module = unsafe { cu::module::load_data(cubin.as_ptr() as *const c_void) }
                .map_err(|e| format!("cuda: loading module {name}: {e:?}"))?;
            Ok(Arc::new(Module { module, from_cache }))
        })
        .clone()
        .map_err(MkError::Backend)
    }

    fn function(&self, k: Kernel) -> MkResult<(sys::CUfunction, [u32; 3])> {
        if let Some(f) = self.blocks.lock().unwrap().get(&k) {
            return Ok(*f);
        }
        let m = self.unit(k)?;
        let entry = CString::new(k.entry).unwrap();
        let f = unsafe { cu::module::get_function(m.module, entry) };
        let f = f.map_err(|e| err(&format!("kernel {}::{}", k.module, k.entry), e))?;
        let block = kernels::block_size(k.module, k.entry).ok_or_else(|| {
            MkError::Backend(format!(
                "cuda: no @workgroup_size for {}::{}",
                k.module, k.entry
            ))
        })?;
        self.blocks.lock().unwrap().insert(k, (f, block));
        Ok((f, block))
    }

    /// `Ok` while the device can run work; once a sticky error made its
    /// context unusable, that reason (`Gpu::unavailable`).
    fn usable(&self) -> MkResult<()> {
        match self.health.failure() {
            Some(reason) => Err(MkError::Unsupported(format!("cuda: {reason}"))),
            None => Ok(()),
        }
    }

    /// After a failed call: whether the context is still usable. A sticky
    /// error (a kernel fault, a launch the operating system's watchdog
    /// stopped, an uncorrectable memory error, a lost device) is returned by
    /// every later call, so the device is marked unavailable for the rest of
    /// the process; anything else (out of memory, an invalid argument)
    /// failed only this request. Returns `e`, or the unavailability.
    fn diagnose(&self, e: MkError) -> MkError {
        if self.health.failure().is_none() {
            let _ = self.bind();
            if let Some(code) = sticky(unsafe { sys::cuCtxSynchronize() }) {
                self.health.fail(format!(
                    "the CUDA device stopped working ({code:?} after: {e}); \
                     the GPU needs a restart of the application"
                ));
            }
        }
        match self.health.failure() {
            Some(reason) => MkError::Unsupported(format!("cuda: {reason}")),
            None => e,
        }
    }

    /// Mark the device unusable (tests of the unavailable path).
    #[doc(hidden)]
    pub fn fail_for_test(&self, reason: &str) {
        self.health.fail(reason);
    }

    fn allocate_now(&self, words: usize) -> MkResult<BufferKey> {
        self.bind()?;
        let p = {
            let admin = self.admin.lock().unwrap();
            let p = unsafe { cu::malloc_async(*admin, words.max(4) * 4) }
                .map_err(|e| err("cuMemAllocAsync", e))?;
            // Usable on every stream once the allocation has completed.
            unsafe { cu::stream::synchronize(*admin) }
                .map_err(|e| err("cuStreamSynchronize", e))?;
            p
        };
        let key = self.next_key.fetch_add(1, Ordering::Relaxed);
        self.persistent.lock().unwrap().insert(key, (p, words));
        Ok(BufferKey(key))
    }

    fn lane(&self) -> MkResult<Lane> {
        if let Some(l) = self.lanes.lock().unwrap().pop() {
            return Ok(l);
        }
        let stream = cu::stream::create(cu::stream::StreamKind::NonBlocking)
            .map_err(|e| err("cuStreamCreate", e))?;
        let ev = || {
            cu::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
                .map_err(|e| err("cuEventCreate", e))
        };
        Ok(Lane {
            stream,
            staging: None,
            start: ev()?,
            end: ev()?,
        })
    }

    fn ptr(&self, key: BufferKey) -> MkResult<(sys::CUdeviceptr, usize)> {
        self.persistent
            .lock()
            .unwrap()
            .get(&key.0)
            .copied()
            .ok_or_else(|| MkError::Backend(format!("cuda: unknown buffer {}", key.0)))
    }

    fn run_on(&self, lane: &mut Lane, batch: Batch<'_>) -> MkResult<BatchStats> {
        let s = lane.stream;
        let t_run = Instant::now();
        let mut stats = BatchStats::default();
        // Inputs: every upload and uniform block at an aligned offset of one
        // staging block, copied to one device allocation; scratch and slices
        // share a second allocation (scratch first, zeroed by one memset).
        // Few driver calls per batch: small plans run many batches.
        const ALIGN: usize = 256;
        let up = |bytes: usize| bytes.div_ceil(ALIGN) * ALIGN;
        let mut input_at = vec![usize::MAX; batch.resources.len()];
        let mut work_at = vec![usize::MAX; batch.resources.len()];
        let mut input_bytes = 0usize;
        let mut scratch_bytes = 0usize;
        for (i, r) in batch.resources.iter().enumerate() {
            match r {
                Resource::Upload(w) => {
                    input_at[i] = input_bytes;
                    input_bytes += up((w.len() * 4).max(16));
                }
                Resource::Scratch(n) => {
                    work_at[i] = scratch_bytes;
                    scratch_bytes += up((*n).max(4) * 4);
                }
                _ => {}
            }
        }
        let mut work_bytes = scratch_bytes;
        for (i, r) in batch.resources.iter().enumerate() {
            if let Resource::Slice { len, .. } = r {
                work_at[i] = work_bytes;
                work_bytes += up((*len).max(4) * 4);
            }
        }
        let mut uniform_at = Vec::with_capacity(batch.passes.len());
        for p in &batch.passes {
            uniform_at.push(p.uniform.as_ref().map(|u| {
                let at = input_bytes;
                input_bytes += up(u.len() * 4);
                at
            }));
        }
        let read_bytes: usize = batch.reads.iter().map(|r| r.into.len() * 4).sum();
        let staging_bytes = input_bytes.max(read_bytes).max(ALIGN);
        if lane
            .staging
            .as_ref()
            .is_none_or(|p| p.bytes < staging_bytes)
        {
            lane.staging = None;
            lane.staging = Some(Pinned::new(staging_bytes)?);
        }
        let staging = lane.staging.as_ref().unwrap();
        let stage = unsafe { std::slice::from_raw_parts_mut(staging.ptr, staging.bytes) };
        // Stage the inputs first (large ones in parallel), so the device
        // clock below starts with the transfer.
        {
            let mut parts: Vec<(&mut [u8], &[u8])> = Vec::new();
            let mut rest: &mut [u8] = &mut stage[..input_bytes];
            let mut at = 0usize;
            let sources = batch
                .resources
                .iter()
                .enumerate()
                .filter_map(|(i, r)| match r {
                    Resource::Upload(w) => Some((input_at[i], bytemuck::cast_slice::<u32, u8>(w))),
                    _ => None,
                })
                .chain(
                    batch.passes.iter().zip(&uniform_at).filter_map(|(p, a)| {
                        Some(((*a)?, bytemuck::cast_slice(p.uniform.as_ref()?)))
                    }),
                );
            for (offset, bytes) in sources {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(offset - at);
                let (head, tail) = tail.split_at_mut(bytes.len());
                parts.push((head, bytes));
                rest = tail;
                at = offset + bytes.len();
            }
            par_copy(parts);
        }
        stats.uploaded_bytes = batch
            .resources
            .iter()
            .map(|r| match r {
                Resource::Upload(w) => (w.len() * 4) as u64,
                _ => 0,
            })
            .sum();

        unsafe { cu::event::record(lane.start, s) }.map_err(|e| err("cuEventRecord", e))?;
        // MKLIGHT_CUDA_PROFILE=1: an event after the inputs, each pass and
        // the copies; one line per batch on stderr.
        let profile = *PROFILE;
        let mut marks: Vec<(String, sys::CUevent)> = Vec::new();
        let mut mark = |label: String| -> MkResult<()> {
            if profile {
                let e = cu::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
                    .map_err(|e| err("cuEventCreate", e))?;
                unsafe { cu::event::record(e, s) }.map_err(|e| err("cuEventRecord", e))?;
                marks.push((label, e));
            }
            Ok(())
        };
        let mut owned: Vec<sys::CUdeviceptr> = Vec::new();
        let mut ptrs: Vec<sys::CUdeviceptr> = Vec::with_capacity(batch.resources.len());
        let result = (|| -> MkResult<()> {
            let alloc =
                |words: usize, owned: &mut Vec<sys::CUdeviceptr>| -> MkResult<sys::CUdeviceptr> {
                    let p = unsafe { cu::malloc_async(s, words.max(4) * 4) }
                        .map_err(|e| err("cuMemAllocAsync", e))?;
                    owned.push(p);
                    Ok(p)
                };
            let inputs = if input_bytes > 0 {
                let p = alloc(input_bytes / 4, &mut owned)?;
                check("cuMemcpyHtoDAsync", unsafe {
                    sys::cuMemcpyHtoDAsync_v2(p, staging.ptr as *const c_void, input_bytes, s)
                })?;
                p
            } else {
                0
            };
            let work = if work_bytes > 0 {
                let p = alloc(work_bytes / 4, &mut owned)?;
                if scratch_bytes > 0 {
                    check("cuMemsetD32Async", unsafe {
                        sys::cuMemsetD32Async(p, 0, scratch_bytes / 4, s)
                    })?;
                }
                p
            } else {
                0
            };
            for (i, r) in batch.resources.iter().enumerate() {
                let p = match r {
                    Resource::Persistent(k) => self.ptr(*k)?.0,
                    Resource::Upload(_) => inputs + input_at[i] as u64,
                    Resource::Scratch(_) => work + work_at[i] as u64,
                    Resource::Slice { key, offset, len } => {
                        let p = work + work_at[i] as u64;
                        if *len > 0 {
                            let (src, _) = self.ptr(*key)?;
                            check("cuMemcpyDtoDAsync", unsafe {
                                sys::cuMemcpyDtoDAsync_v2(
                                    p,
                                    src + (*offset * 4) as u64,
                                    *len * 4,
                                    s,
                                )
                            })?;
                        }
                        p
                    }
                };
                ptrs.push(p);
            }
            let uniforms = inputs;
            mark("inputs".into())?;
            for (p, u) in batch.passes.iter().zip(&uniform_at) {
                let (f, block) = self.function(p.kernel)?;
                let mut args: Vec<u64> = Vec::with_capacity(1 + p.storage.len());
                args.push(u.map_or(0, |o| uniforms + o as u64));
                for b in &p.storage {
                    let off = b.range.map_or(0, |r| r.0);
                    args.push(ptrs[b.resource] + (off * 4) as u64);
                }
                let mut params: Vec<*mut c_void> = args
                    .iter_mut()
                    .map(|a| a as *mut u64 as *mut c_void)
                    .collect();
                let [x, y, z] = p.workgroups;
                if x == 0 || y == 0 || z == 0 {
                    continue;
                }
                check(
                    &format!("launching {}::{}", p.kernel.module, p.kernel.entry),
                    unsafe {
                        sys::cuLaunchKernel(
                            f,
                            x,
                            y,
                            z,
                            block[0],
                            block[1],
                            block[2],
                            0,
                            s,
                            params.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    },
                )?;
                stats.passes += 1;
                mark(format!("{}[{}x{}x{}]", p.kernel.entry, x, y, z))?;
            }
            for c in &batch.copies {
                if c.len == 0 {
                    continue;
                }
                let (dst, _) = self.ptr(c.to)?;
                check("cuMemcpyDtoDAsync", unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        dst + (c.to_offset * 4) as u64,
                        ptrs[c.from] + (c.from_offset * 4) as u64,
                        c.len * 4,
                        s,
                    )
                })?;
            }
            Ok(())
        })();
        // Readbacks into staging, then into the destinations after the sync.
        let _ = mark("copies".into());
        let mut reads = batch.reads;
        let mut ok = result.and_then(|()| {
            let mut cursor = 0usize;
            for r in &reads {
                let bytes = r.into.len() * 4;
                if bytes == 0 {
                    continue;
                }
                let src = ptrs[r.resource];
                check("cuMemcpyDtoHAsync", unsafe {
                    sys::cuMemcpyDtoHAsync_v2(
                        staging.ptr.add(cursor) as *mut c_void,
                        src + (r.offset * 4) as u64,
                        bytes,
                        s,
                    )
                })?;
                cursor += bytes;
            }
            Ok(())
        });
        if ok.is_ok() {
            ok = unsafe { cu::event::record(lane.end, s) }.map_err(|e| err("cuEventRecord", e));
        }
        for p in owned {
            unsafe {
                let _ = cu::free_async(p, s);
            }
        }
        let t_sync = Instant::now();
        let sync = unsafe { cu::stream::synchronize(s) }.map_err(|e| err("cuStreamSynchronize", e));
        ok?;
        sync?;
        let mut parts: Vec<(&mut [u8], &[u8])> = Vec::new();
        let mut rest: &[u8] = stage;
        for r in &mut reads {
            let bytes = r.into.len() * 4;
            let (head, tail) = rest.split_at(bytes);
            parts.push((bytemuck::cast_slice_mut(r.into), head));
            rest = tail;
            stats.readback_bytes += bytes as u64;
        }
        par_copy(parts);
        stats.device_seconds = unsafe { cu::event::elapsed(lane.start, lane.end) }
            .map_err(|e| err("cuEventElapsedTime", e))? as f64
            * 1e-3;
        DEVICE_NANOS.fetch_add((stats.device_seconds * 1e9) as u64, Ordering::Relaxed);
        if profile {
            let mut line = format!(
                "cuda batch {:.3} ms, host {:.3} ms enqueue + {:.3} ms wait/unstage (up {} KiB, down {} KiB):",
                stats.device_seconds * 1e3,
                (t_sync - t_run).as_secs_f64() * 1e3,
                t_sync.elapsed().as_secs_f64() * 1e3,
                stats.uploaded_bytes / 1024,
                stats.readback_bytes / 1024
            );
            let mut prev = lane.start;
            for (label, e) in &marks {
                let ms = unsafe { cu::event::elapsed(prev, *e) }.unwrap_or(f32::NAN);
                line += &format!(" {label} {ms:.3}");
                prev = *e;
            }
            let ms = unsafe { cu::event::elapsed(prev, lane.end) }.unwrap_or(f32::NAN);
            line += &format!(" readback {ms:.3}");
            eprintln!("{line}");
            for (_, e) in marks {
                unsafe {
                    let _ = cu::event::destroy(e);
                }
            }
        }
        Ok(stats)
    }

    /// NVRTC version and driver version, for `kernel_build`.
    pub fn versions(&self) -> (Option<(i32, i32)>, i32) {
        (self.nvrtc, self.driver_version)
    }
}

impl Gpu for CudaDevice {
    fn summary(&self) -> &AdapterSummary {
        &self.summary
    }

    fn persist(&self, words: &[u32]) -> MkResult<BufferKey> {
        let key = self.allocate(words.len())?;
        if !words.is_empty() {
            let copied = self.bind().and_then(|()| {
                let (p, _) = self.ptr(key)?;
                unsafe { cu::memcpy_htod_sync(p, words) }.map_err(|e| err("cuMemcpyHtoD", e))
            });
            if let Err(e) = copied {
                self.release(key);
                return Err(self.diagnose(e));
            }
        }
        Ok(key)
    }

    fn allocate(&self, words: usize) -> MkResult<BufferKey> {
        self.usable()?;
        self.allocate_now(words).map_err(|e| self.diagnose(e))
    }

    fn release(&self, key: BufferKey) {
        if self.health.failure().is_some() {
            // The context is unusable; its memory goes with the process.
            self.persistent.lock().unwrap().remove(&key.0);
            return;
        }
        if let Some((p, _)) = self.persistent.lock().unwrap().remove(&key.0) {
            // Batches complete before `run` returns, so nothing uses `p`.
            let _ = self.bind();
            let admin = self.admin.lock().unwrap();
            unsafe {
                let _ = cu::free_async(p, *admin);
            }
        }
    }

    fn prepare(&self, kernels: &[(Kernel, PassShape)]) -> MkResult<()> {
        self.usable()?;
        let none = std::sync::atomic::AtomicBool::new(false);
        self.prepare_parallel(kernels, &|_, _| {}, &none)
            .map(|_| ())
    }

    fn prepare_all(
        &self,
        kernels: &[(Kernel, PassShape)],
        cache_dir: Option<&std::path::Path>,
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &std::sync::atomic::AtomicBool,
    ) -> MkResult<(usize, usize, Option<String>)> {
        self.usable()?;
        if let Some(d) = cache_dir {
            self.set_cache_dir(Some(d.join("cuda")));
        }
        // The host's kernels and those it uses only on CUDA (FftMtf).
        let mut all = kernels.to_vec();
        all.extend(kernels::CUDA_ENTRIES.iter().map(|(m, e, _)| {
            let shape = PassShape {
                uniform: false,
                storage: Vec::new(),
            };
            (Kernel::new(m, e), shape)
        }));
        let from_cache = self.prepare_parallel(&all, progress, cancel)?;
        Ok((
            all.len(),
            from_cache,
            self.cache_warning.lock().unwrap().clone(),
        ))
    }

    fn run(&self, batch: Batch<'_>) -> MkResult<BatchStats> {
        self.usable()?;
        self.bind()?;
        let mut lane = self.lane()?;
        match self.run_on(&mut lane, batch) {
            Ok(stats) => {
                self.lanes.lock().unwrap().push(lane);
                Ok(stats)
            }
            // The lane's stream may hold the failure: it is not reused.
            Err(e) => Err(self.diagnose(e)),
        }
    }

    fn unavailable(&self) -> Option<String> {
        self.health.failure()
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn kernel_build(&self, raw: &NumericsProbe) -> Option<String> {
        let yn = |b: bool| if b { "yes" } else { "no" };
        Some(format!(
            "CUDA {} ({}, sm_{}{}, driver {}; {}); plain expressions: contraction {}, \
             reassociation {}, correctly rounded div {} / sqrt {}",
            kernels::digest(),
            match (kernels::embedded_targets(), self.nvrtc) {
                (t, _) if !t.is_empty() => format!("built in: {}", t.join(" ")),
                (_, Some((a, b))) => format!("NVRTC {a}.{b}"),
                _ => "no compiler".into(),
            },
            self.arch.0,
            self.arch.1,
            self.driver_version,
            kernels::OPTIONS.join(" "),
            yn(!raw.no_contraction),
            yn(!raw.no_reassociation),
            yn(raw.correctly_rounded_div),
            yn(raw.correctly_rounded_sqrt),
        ))
    }

    fn ports(&self, family: &str) -> bool {
        kernels::ported(family)
    }
}

/// Copy each `(dst, src)` pair, large ones in parallel pieces: one thread
/// copies about 10 GB/s, so a 32 MiB plane alone took 3 ms.
fn par_copy(parts: Vec<(&mut [u8], &[u8])>) {
    use rayon::prelude::*;
    const PIECE: usize = 1 << 20;
    let total: usize = parts.iter().map(|(_, s)| s.len()).sum();
    if total < 4 * PIECE {
        for (d, s) in parts {
            d.copy_from_slice(s);
        }
        return;
    }
    parts
        .into_par_iter()
        .flat_map(|(d, s)| d.par_chunks_mut(PIECE).zip(s.par_chunks(PIECE)))
        .for_each(|(d, s)| d.copy_from_slice(s));
}

impl CudaDevice {
    /// Prepare `kernels` on every core (NVRTC is thread-safe); returns how
    /// many came from the cache.
    fn prepare_parallel(
        &self,
        kernels: &[(Kernel, PassShape)],
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &std::sync::atomic::AtomicBool,
    ) -> MkResult<usize> {
        let mut ks: Vec<Kernel> = kernels
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| kernels::source(k.module).is_some())
            .collect();
        // The units that inline the trace take seconds each: start them first.
        ks.sort_by_key(|k| {
            !kernels::source(k.module).is_some_and(|s| s.contains("trace_bindings.h"))
        });
        let total = kernels.len();
        // Kernels without a CUDA module have nothing to prepare.
        let done = std::sync::atomic::AtomicUsize::new(total - ks.len());
        let cached = std::sync::atomic::AtomicUsize::new(0);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(ks.len().max(1));
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| -> MkResult<()> {
                        while !cancel.load(Ordering::Relaxed) {
                            let Some(k) = ks.get(next.fetch_add(1, Ordering::Relaxed)) else {
                                break;
                            };
                            self.function(*k)?;
                            if self.unit(*k)?.from_cache {
                                cached.fetch_add(1, Ordering::Relaxed);
                            }
                            progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                        }
                        Ok(())
                    })
                })
                .collect();
            workers.into_iter().try_for_each(|w| {
                w.join().unwrap_or_else(|_| {
                    Err(MkError::Backend("cuda: compile thread panicked".into()))
                })
            })
        })?;
        Ok(cached.into_inner())
    }
}

/// The code of a sticky error: one that leaves the context unusable, so
/// every later call returns it until the process ends.
fn sticky(r: sys::CUresult) -> Option<sys::CUresult> {
    use sys::CUresult::*;
    matches!(
        r,
        CUDA_ERROR_ILLEGAL_ADDRESS
            | CUDA_ERROR_LAUNCH_FAILED
            | CUDA_ERROR_LAUNCH_TIMEOUT
            | CUDA_ERROR_HARDWARE_STACK_ERROR
            | CUDA_ERROR_ILLEGAL_INSTRUCTION
            | CUDA_ERROR_MISALIGNED_ADDRESS
            | CUDA_ERROR_INVALID_ADDRESS_SPACE
            | CUDA_ERROR_INVALID_PC
            | CUDA_ERROR_ECC_UNCORRECTABLE
            | CUDA_ERROR_CONTEXT_IS_DESTROYED
            | CUDA_ERROR_DEVICE_UNAVAILABLE
            | CUDA_ERROR_DEINITIALIZED
            | CUDA_ERROR_UNKNOWN
    )
    .then_some(r)
}
