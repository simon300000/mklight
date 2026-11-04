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
//! * Accuracy is established by offline cross-validation and the engine's
//!   static per-analysis capability policy.

use mklight_core::job::Precision;
use mklight_gpu::{ComputeBackend, CpuF32Backend, JobKind};
use std::sync::{Arc, Mutex, OnceLock};

use crate::analysis::gpu::GPU_ANALYSES;
use crate::backend::{
    Backend, BackendCapabilities, BackendInfo, BackendProvider, DeviceHandle, ProviderHint,
};
use crate::lru::LruCache;

/// Native requests remembered as failing per device.
const KNOWN_FAILURES: usize = 256;

/// A native request that failed deterministically on a device
/// (`GpuDevice::known_failure`).
#[derive(Clone, Debug, PartialEq)]
pub struct KnownFailure {
    pub reason: String,
    /// Time the failed attempt took.
    pub elapsed_ms: f64,
}

/// An open GPU backend as GPU analyses see it (`AnalysisInput::device`
/// downcasts to this type).
pub struct GpuDevice {
    /// Engine backend id (`gpu:metal`), recorded as `RunInfo.backend`.
    pub id: String,
    pub backend: Arc<dyn ComputeBackend>,
    /// Deterministic native failures by request identity (snapshot, glass
    /// library, analysis and settings; see `analysis::routing`), so the
    /// same request is not attempted again on this device. An edit changes
    /// the identity; bounded LRU.
    known_failures: Mutex<LruCache<[u8; 32], KnownFailure>>,
}

impl GpuDevice {
    pub fn new(id: impl Into<String>, backend: Arc<dyn ComputeBackend>) -> Self {
        Self {
            id: id.into(),
            backend,
            known_failures: Mutex::new(LruCache::new(KNOWN_FAILURES)),
        }
    }

    /// The recorded failure of the request with identity `key`.
    pub fn known_failure(&self, key: &[u8; 32]) -> Option<KnownFailure> {
        self.known_failures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }

    pub fn remember_failure(&self, key: [u8; 32], failure: KnownFailure) {
        self.known_failures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, failure);
    }

    /// Device name (`"Apple M2 Pro"`).
    pub fn device_name(&self) -> String {
        self.backend.info().name.clone()
    }
}

type Opener = Box<dyn Fn() -> Result<Arc<dyn ComputeBackend>, String> + Send + Sync>;

/// A `BackendProvider` that opens a `ComputeBackend` on first use.
pub struct GpuProvider {
    backend: Backend,
    title: String,
    open: Opener,
    device: OnceLock<Result<Arc<GpuDevice>, String>>,
    /// The device family before opening: every Metal device on Apple
    /// silicon is the Apple GPU.
    apple_gpu: Option<bool>,
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
            device: OnceLock::new(),
            apple_gpu: (name == "metal" && cfg!(all(target_os = "macos", target_arch = "aarch64")))
                .then_some(true),
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
            let d = GpuDevice::new(self.backend.id(), b);
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

    fn hint(&self) -> Option<ProviderHint> {
        let opened = match self.device.get() {
            Some(Err(reason)) => Some(Err(reason.clone())),
            Some(Ok(d)) => Some(Ok(crate::analysis::is_apple_gpu(d.backend.info()))),
            None if std::env::var("MKLIGHT_DISABLE_GPU").is_ok_and(|v| v == "1") => {
                Some(Err("disabled by MKLIGHT_DISABLE_GPU=1".into()))
            }
            None => None,
        };
        Some(match opened {
            Some(Err(reason)) => ProviderHint {
                unavailable: Some(reason),
                ..Default::default()
            },
            known => ProviderHint {
                analyses: GPU_ANALYSES.iter().map(|s| s.to_string()).collect(),
                apple_gpu: known.as_ref().map_or(self.apple_gpu, |k| k.clone().ok()),
                open: known.is_some(),
                ..Default::default()
            },
        })
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
