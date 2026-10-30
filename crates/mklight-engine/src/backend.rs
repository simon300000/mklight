//! Compute-backend selection.
//!
//! The CPU f64 reference is always present. GPU backends (`mklight-gpu`
//! implementations such as Metal) register a `BackendProvider` once they are
//! wired in; until then a Metal entry is listed as unavailable on macOS so
//! front ends can show the choice.
//!
//! Selection is per analysis: a GPU backend is used only for the analyses it
//! lists in `capabilities.analyses`. Requesting a backend that is
//! unavailable, or that does not implement the analysis, is an error unless
//! the caller allows fallback, in which case the job runs on the CPU
//! reference and its `RunInfo` says so (never silently degrade). The chosen
//! provider's `device()` handle travels to the analysis in `AnalysisInput`.

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::Arc;

use mklight_core::job::Precision;

use crate::error::{EngineError, EngineResult};

/// A requested backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum Backend {
    /// CPU double-precision reference (`mklight-trace`).
    #[default]
    CpuReference,
    /// A GPU backend by short name (`"metal"`, later `"vulkan"`, `"cuda"`).
    Gpu(String),
}

impl Backend {
    /// Stable string id: `cpu_reference` or `gpu:<name>`. Recorded in
    /// `RunInfo.backend`.
    pub fn id(&self) -> String {
        match self {
            Backend::CpuReference => "cpu_reference".into(),
            Backend::Gpu(n) => format!("gpu:{n}"),
        }
    }
    /// Inverse of `id` (also accepts `cpu` and a bare GPU name like `metal`).
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "" | "cpu" | "cpu_reference" => Backend::CpuReference,
            other => Backend::Gpu(other.strip_prefix("gpu:").unwrap_or(other).to_string()),
        }
    }
}

/// What a backend can do. Extended as GPU backends land.
#[derive(Clone, Debug, PartialEq, Default, Serialize)]
pub struct BackendCapabilities {
    /// Native double precision.
    pub fp64: bool,
    /// This is the reference every other backend is validated against.
    pub reference: bool,
    /// Analyses the backend accelerates (empty: none beyond the reference).
    pub analyses: Vec<String>,
    pub device_name: Option<String>,
    pub unified_memory: Option<bool>,
    /// Largest ray batch per dispatch, if limited.
    pub max_rays_per_dispatch: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BackendInfo {
    pub backend: Backend,
    pub id: String,
    pub title: String,
    pub available: bool,
    /// Why the backend is unavailable.
    pub reason: Option<String>,
    pub precision: Precision,
    pub capabilities: BackendCapabilities,
}

impl BackendInfo {
    /// Can this backend run `analysis`? The reference runs everything; any
    /// other backend only what it lists.
    pub fn supports(&self, analysis: &str) -> bool {
        self.capabilities.reference || self.capabilities.analyses.iter().any(|a| a == analysis)
    }

    pub fn cpu_reference() -> Self {
        BackendInfo {
            backend: Backend::CpuReference,
            id: Backend::CpuReference.id(),
            title: "CPU reference (f64)".into(),
            available: true,
            reason: None,
            precision: Precision::F64Reference,
            capabilities: BackendCapabilities {
                fp64: true,
                reference: true,
                ..Default::default()
            },
        }
    }
}

/// An opaque handle to a compute device. Analyses downcast it to the
/// interface they need (`mklight-gpu`'s compute-backend trait object once
/// that crate is wired in).
pub type DeviceHandle = Arc<dyn Any + Send + Sync>;

/// Implemented by GPU backend crates to describe and provide a device.
/// Integration point: `mklight-gpu` / `mklight-metal`.
pub trait BackendProvider: Send + Sync {
    fn info(&self) -> BackendInfo;
    /// The device analyses compute on; `None` for the CPU reference.
    fn device(&self) -> Option<DeviceHandle> {
        None
    }
}

/// The outcome of backend selection for one analysis run.
#[derive(Clone)]
pub struct SelectedBackend {
    pub info: BackendInfo,
    pub device: Option<DeviceHandle>,
    /// Set when the run fell back to the CPU reference; goes into `RunInfo`.
    pub warning: Option<String>,
}

impl std::fmt::Debug for SelectedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedBackend")
            .field("info", &self.info)
            .field("device", &self.device.is_some())
            .field("warning", &self.warning)
            .finish()
    }
}

impl SelectedBackend {
    pub fn cpu_reference() -> Self {
        SelectedBackend {
            info: BackendInfo::cpu_reference(),
            device: None,
            warning: None,
        }
    }
}

struct Static(BackendInfo);
impl BackendProvider for Static {
    fn info(&self) -> BackendInfo {
        self.0.clone()
    }
}

#[derive(Clone)]
pub struct BackendRegistry {
    providers: Vec<Arc<dyn BackendProvider>>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl BackendRegistry {
    /// The CPU reference, plus (on macOS) a placeholder Metal entry that is
    /// listed as unavailable until `mklight-metal` is wired in.
    pub fn with_defaults() -> Self {
        let mut r = BackendRegistry {
            providers: vec![Arc::new(Static(BackendInfo::cpu_reference()))],
        };
        if cfg!(target_os = "macos") {
            // PLACEHOLDER: replaced by the real Metal provider.
            let b = Backend::Gpu("metal".into());
            r.register(Arc::new(Static(BackendInfo {
                id: b.id(),
                backend: b,
                title: "Metal GPU (f32)".into(),
                available: false,
                reason: Some("the Metal backend is not wired into the engine yet".into()),
                precision: Precision::F32Gpu,
                capabilities: BackendCapabilities::default(),
            })));
        }
        r
    }

    /// Add a provider, replacing one with the same id.
    pub fn register(&mut self, p: Arc<dyn BackendProvider>) {
        let id = p.info().id;
        self.providers.retain(|q| q.info().id != id);
        self.providers.push(p);
    }

    pub fn list(&self) -> Vec<BackendInfo> {
        self.providers.iter().map(|p| p.info()).collect()
    }

    pub fn info(&self, b: &Backend) -> Option<BackendInfo> {
        self.provider(b).map(|(_, i)| i)
    }

    fn provider(&self, b: &Backend) -> Option<(&Arc<dyn BackendProvider>, BackendInfo)> {
        let id = b.id();
        self.providers
            .iter()
            .map(|p| (p, p.info()))
            .find(|(_, i)| i.id == id)
    }

    /// Resolve a request for running `analysis` (`None`: only check that
    /// the backend is available). A backend that is unavailable or does not
    /// implement the analysis is an error, or, with `allow_fallback`, the CPU
    /// reference plus a warning for `RunInfo`.
    pub fn select(
        &self,
        requested: &Backend,
        analysis: Option<&str>,
        allow_fallback: bool,
    ) -> EngineResult<SelectedBackend> {
        let why = match self.provider(requested) {
            None => "unknown backend".to_string(),
            Some((_, i)) if !i.available => i.reason.unwrap_or_else(|| "unavailable".into()),
            Some((p, i)) => match analysis {
                Some(a) if !i.supports(a) => format!("it does not implement '{a}'"),
                _ => {
                    return Ok(SelectedBackend {
                        device: p.device(),
                        info: i,
                        warning: None,
                    });
                }
            },
        };
        if allow_fallback && *requested != Backend::CpuReference {
            let mut s = SelectedBackend::cpu_reference();
            s.warning = Some(format!(
                "backend '{}' not used ({why}); ran on the CPU reference",
                requested.id()
            ));
            Ok(s)
        } else {
            Err(EngineError::BackendUnavailable(format!(
                "{}: {why}",
                requested.id()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip() {
        for b in [Backend::CpuReference, Backend::Gpu("metal".into())] {
            assert_eq!(Backend::parse(&b.id()), b);
        }
        assert_eq!(Backend::parse("metal"), Backend::Gpu("metal".into()));
        assert_eq!(
            serde_json::to_value(Backend::Gpu("metal".into())).unwrap(),
            serde_json::json!({"kind": "gpu", "name": "metal"})
        );
    }

    #[test]
    fn selection_and_fallback() {
        let r = BackendRegistry::with_defaults();
        let s = r
            .select(&Backend::CpuReference, Some("validate"), false)
            .unwrap();
        assert!(s.info.available && s.info.capabilities.fp64 && s.warning.is_none());
        assert!(s.device.is_none());
        let gpu = Backend::Gpu("nonexistent".into());
        assert!(matches!(
            r.select(&gpu, None, false),
            Err(EngineError::BackendUnavailable(_))
        ));
        let s = r.select(&gpu, Some("validate"), true).unwrap();
        assert_eq!(s.info.backend, Backend::CpuReference);
        assert!(s.warning.unwrap().contains("gpu:nonexistent"));
    }

    /// A fake device, standing in for a GPU compute backend.
    struct FakeDevice(&'static str);

    struct Emu;
    impl BackendProvider for Emu {
        fn info(&self) -> BackendInfo {
            let b = Backend::Gpu("emu".into());
            BackendInfo {
                id: b.id(),
                backend: b,
                title: "emulated".into(),
                available: true,
                reason: None,
                precision: Precision::F32Gpu,
                capabilities: BackendCapabilities {
                    analyses: vec!["spot_diagram".into()],
                    ..Default::default()
                },
            }
        }
        fn device(&self) -> Option<DeviceHandle> {
            Some(Arc::new(FakeDevice("emu-0")))
        }
    }

    #[test]
    fn providers_supply_devices_per_analysis() {
        let mut r = BackendRegistry::with_defaults();
        r.register(Arc::new(Emu));
        let b = Backend::Gpu("emu".into());
        // Availability only.
        assert!(r.select(&b, None, false).is_ok());
        // An analysis it implements: the GPU, with its device.
        let s = r.select(&b, Some("spot_diagram"), false).unwrap();
        assert_eq!(s.info.precision, Precision::F32Gpu);
        assert!(s.warning.is_none());
        let dev = s.device.unwrap();
        assert_eq!(dev.downcast_ref::<FakeDevice>().unwrap().0, "emu-0");
        // One it does not: an error, or a CPU fallback that says why.
        let e = r.select(&b, Some("mtf"), false).unwrap_err();
        assert!(e.to_string().contains("does not implement 'mtf'"), "{e}");
        let s = r.select(&b, Some("mtf"), true).unwrap();
        assert_eq!(s.info.backend, Backend::CpuReference);
        assert!(s.device.is_none());
        let w = s.warning.unwrap();
        assert!(w.contains("gpu:emu") && w.contains("'mtf'"), "{w}");
        // The reference supports everything.
        assert!(BackendInfo::cpu_reference().supports("anything"));
    }
}
