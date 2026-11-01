//! GPU backends for the engine: a `BackendProvider` over any
//! `mklight_gpu::ComputeBackend` (Metal on macOS, the CPU-f32 emulation for
//! tests and diagnostics), and the device handle GPU analyses receive.
//!
//! * The device is opened on first use (listing backends or selecting one),
//!   not when the engine starts. A device that does not exist, fails to
//!   open, or fails its start-up numerics probe (so it refuses traces) makes
//!   the backend unavailable, with the reason.
//! * `MKLIGHT_DISABLE_GPU=1` in the environment makes every GPU provider
//!   unavailable (reproducible CPU-only runs on a machine with a GPU).
//! * Precision qualification (docs/GPU.md §5.2) is per system: a GPU
//!   analysis calls [`GpuDevice::qualify`] with the snapshot and library
//!   hashes, which runs `mklight_gpu::xval::qualify` once per key and caches
//!   the verdict. A failed qualification means the CPU reference computes
//!   the result (CONVENTIONS §12).

use std::sync::{Arc, Mutex, OnceLock};

use mklight_core::job::Precision;
use mklight_core::{JobContext, MkResult};
use mklight_gpu::xval::{Qualification, XvalSettings, qualify};
use mklight_gpu::{ComputeBackend, CpuF32Backend, JobKind, Radiometry};
use mklight_trace::SequentialTracer;

use crate::analysis::gpu::GPU_ANALYSES;
use crate::backend::{Backend, BackendCapabilities, BackendInfo, BackendProvider, DeviceHandle};
use crate::lru::LruCache;

/// Qualifications kept per device (one per recently analysed system).
const QUALIFICATION_CACHE: usize = 32;

/// An open GPU backend as GPU analyses see it (`AnalysisInput::device`
/// downcasts to this type).
pub struct GpuDevice {
    /// Engine backend id (`gpu:metal`), recorded as `RunInfo.backend`.
    pub id: String,
    pub backend: Arc<dyn ComputeBackend>,
    /// The cross-validation sample of the qualification. Defaults to
    /// `XvalSettings::qualification()` with radiometry off: the GPU
    /// analyses trace geometry only, so a coating the GPU cannot weight
    /// does not disqualify it.
    pub qualification: XvalSettings,
    cache: Mutex<LruCache<String, Arc<Qualification>>>,
    chains: Mutex<LruCache<String, serde_json::Value>>,
}

impl GpuDevice {
    pub fn new(id: impl Into<String>, backend: Arc<dyn ComputeBackend>) -> Self {
        let mut q = XvalSettings::qualification();
        q.radiometry = Radiometry::Off;
        GpuDevice {
            id: id.into(),
            backend,
            qualification: q,
            cache: Mutex::new(LruCache::new(QUALIFICATION_CACHE)),
            chains: Mutex::new(LruCache::new(QUALIFICATION_CACHE)),
        }
    }

    /// Replace the qualification sample (tests use an impossible budget to
    /// exercise the fallback).
    pub fn with_qualification(mut self, settings: XvalSettings) -> Self {
        self.qualification = settings;
        self
    }

    /// Device name (`"Apple M2 Pro"`).
    pub fn device_name(&self) -> String {
        self.backend.info().name.clone()
    }

    /// The qualification of this device for `tracer`'s system, cached under
    /// `key` (snapshot hash + library hash). Fails only when cancelled.
    pub fn qualify(
        &self,
        key: &str,
        tracer: &SequentialTracer,
        ctx: &JobContext,
    ) -> MkResult<Arc<Qualification>> {
        let key = format!(
            "{key}:{}:{}",
            serde_json::to_string(&self.qualification).unwrap(),
            serde_json::to_string(self.backend.precision()).unwrap()
        );
        let lock = || self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(q) = lock().get(&key.to_string()) {
            return Ok(q);
        }
        // Outside the lock: concurrent jobs may qualify the same system
        // twice, with identical verdicts.
        let q = Arc::new(qualify(
            self.backend.as_ref(),
            tracer,
            Some(&self.qualification),
            ctx,
        )?);
        lock().insert(key.to_string(), q.clone());
        Ok(q)
    }

    pub(crate) fn chain_qualification(&self, key: &str) -> Option<serde_json::Value> {
        self.chains
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key.to_string())
    }
    pub(crate) fn cache_chain_qualification(&self, key: String, report: serde_json::Value) {
        self.chains
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, report);
    }

    /// Qualifications currently cached (diagnostics, tests).
    pub fn cached_qualifications(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

type Opener = Box<dyn Fn() -> Result<Arc<dyn ComputeBackend>, String> + Send + Sync>;

/// A `BackendProvider` that opens a `ComputeBackend` on first use.
pub struct GpuProvider {
    backend: Backend,
    title: String,
    open: Opener,
    qualification: Option<XvalSettings>,
    device: OnceLock<Result<Arc<GpuDevice>, String>>,
}

impl GpuProvider {
    /// A provider for `gpu:<name>` whose device `open` creates.
    pub fn new(
        name: &str,
        title: &str,
        open: impl Fn() -> Result<Arc<dyn ComputeBackend>, String> + Send + Sync + 'static,
    ) -> Self {
        GpuProvider {
            backend: Backend::Gpu(name.into()),
            title: title.into(),
            open: Box::new(open),
            qualification: None,
            device: OnceLock::new(),
        }
    }

    /// The system default Metal device (`gpu:metal`).
    pub fn metal() -> Self {
        Self::new("metal", "Metal GPU (f32)", open_metal)
    }

    /// The bit-faithful CPU emulation of the GPU numerics (`gpu:cpu_f32`).
    /// Not registered by default: it runs the GPU algorithms on the CPU,
    /// slower than the reference, to separate algorithm error from precision
    /// error and to exercise the GPU paths on machines without a GPU.
    pub fn emulation() -> Self {
        Self::new("cpu_f32", "CPU f32 emulation of the GPU kernels", || {
            Ok(Arc::new(CpuF32Backend::new()) as Arc<dyn ComputeBackend>)
        })
    }

    /// Use `settings` for the precision qualification.
    pub fn with_qualification(mut self, settings: XvalSettings) -> Self {
        self.qualification = Some(settings);
        self
    }

    fn get(&self) -> &Result<Arc<GpuDevice>, String> {
        self.device.get_or_init(|| {
            if std::env::var("MKLIGHT_DISABLE_GPU").is_ok_and(|v| v == "1") {
                return Err("disabled by MKLIGHT_DISABLE_GPU=1".into());
            }
            let b = (self.open)()?;
            if !b.supports(JobKind::Trace) {
                return Err(format!(
                    "{} does not run the trace kernel (numerics probe: {:?})",
                    b.info().backend_id(),
                    b.precision().probe
                ));
            }
            let mut d = GpuDevice::new(self.backend.id(), b);
            if let Some(q) = &self.qualification {
                d = d.with_qualification(q.clone());
            }
            Ok(Arc::new(d))
        })
    }
}

fn open_metal() -> Result<Arc<dyn ComputeBackend>, String> {
    #[cfg(target_os = "macos")]
    {
        match mklight_metal::MetalBackend::new() {
            Ok(Some(b)) => Ok(Arc::new(b)),
            Ok(None) => Err("this machine has no Metal device".into()),
            Err(e) => Err(format!("the Metal device could not be opened: {e}")),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Metal is only available on macOS".into())
    }
}

impl BackendProvider for GpuProvider {
    fn id(&self) -> String {
        self.backend.id()
    }

    fn info(&self) -> BackendInfo {
        let mut info = BackendInfo {
            id: self.backend.id(),
            backend: self.backend.clone(),
            title: self.title.clone(),
            available: false,
            reason: None,
            precision: Precision::F32Gpu,
            capabilities: BackendCapabilities::default(),
        };
        match self.get() {
            Ok(d) => {
                let di = d.backend.info();
                info.available = true;
                info.capabilities = BackendCapabilities {
                    fp64: false,
                    reference: false,
                    analyses: GPU_ANALYSES.iter().map(|s| s.to_string()).collect(),
                    device_name: Some(di.name.clone()),
                    unified_memory: Some(di.unified_memory),
                    max_rays_per_dispatch: None,
                };
            }
            Err(reason) => info.reason = Some(reason.clone()),
        }
        info
    }

    fn device(&self) -> Option<DeviceHandle> {
        self.get().as_ref().ok().map(|d| d.clone() as DeviceHandle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emulation_provider_is_available_and_lists_gpu_analyses() {
        let p = GpuProvider::emulation();
        assert_eq!(p.id(), "gpu:cpu_f32");
        let i = p.info();
        assert!(i.available, "{:?}", i.reason);
        assert_eq!(i.precision, Precision::F32Gpu);
        assert!(i.supports("spot_diagram") && !i.supports("validate"));
        let d = p.device().unwrap();
        let g = d.downcast_ref::<GpuDevice>().unwrap();
        assert_eq!(g.id, "gpu:cpu_f32");
        assert_eq!(g.cached_qualifications(), 0);
    }

    #[test]
    fn failing_open_makes_the_backend_unavailable() {
        let p = GpuProvider::new("broken", "broken", || Err("no device here".into()));
        let i = p.info();
        assert!(!i.available);
        assert_eq!(i.reason.as_deref(), Some("no device here"));
        assert!(p.device().is_none());
    }
}
