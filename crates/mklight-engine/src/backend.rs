//! Compute-backend selection.
//!
//! The CPU f64 reference is always present. GPU backends (`mklight-gpu`
//! implementations) register a `BackendProvider`; on macOS the Metal
//! backend (`crate::gpu::GpuProvider::metal`) is registered by default and
//! is available when the machine has a Metal device that passes the
//! numerics probe.
//!
//! A request names a backend (`{"kind": "cpu_reference"}`,
//! `{"kind": "gpu", "name": "metal"}`) or asks for `{"kind": "auto"}`.
//!
//! * **Explicit.** A GPU backend is used only for the analyses it lists in
//!   `capabilities.analyses`. Requesting a backend that is unavailable, or
//!   that does not implement the analysis, is an error unless the caller
//!   allows fallback, in which case the job runs on the CPU reference and
//!   its `RunInfo` says so (never silently degrade).
//! * **Auto.** The first available GPU backend that implements the analysis,
//!   else the CPU reference (the reason is in the result's
//!   `BackendReport::fallback_reason`; no warning, since this is the
//!   expected outcome for analyses without a GPU path).
//!
//! Either way a selected GPU analysis first qualifies the backend's
//! precision for the snapshot (docs/GPU.md §5.2, CONVENTIONS §12) and falls
//! back to the CPU reference, with a `RunInfo` warning, when it fails. The
//! chosen provider's `device()` handle travels to the analysis in
//! `AnalysisInput`.

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
    /// The first available GPU backend implementing the analysis, else the
    /// CPU reference.
    Auto,
}

impl Backend {
    /// Stable string id: `cpu_reference`, `gpu:<name>` or `auto`. The id
    /// of the backend that computed a result is `RunInfo.backend`.
    pub fn id(&self) -> String {
        match self {
            Backend::CpuReference => "cpu_reference".into(),
            Backend::Gpu(n) => format!("gpu:{n}"),
            Backend::Auto => "auto".into(),
        }
    }
    /// Inverse of `id` (also accepts `cpu` and a bare GPU name like `metal`).
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "" | "cpu" | "cpu_reference" => Backend::CpuReference,
            "auto" => Backend::Auto,
            other => Backend::Gpu(other.strip_prefix("gpu:").unwrap_or(other).to_string()),
        }
    }
    /// Whether a recorded request id (`SelectedBackend::requested`, which
    /// carries a `default:` prefix when the engine default applied) asked
    /// for `auto`.
    pub fn is_auto_request(requested: &str) -> bool {
        requested.strip_prefix("default:").unwrap_or(requested) == Backend::Auto.id()
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

/// What a provider can tell without opening its device
/// (`BackendProvider::hint`), so that `auto` can settle CPU routes before any
/// device initialisation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderHint {
    /// The provider is a reference (the CPU f64 backend).
    pub reference: bool,
    /// Analyses the provider implements once its device is open.
    pub analyses: Vec<String>,
    /// Why the device is unavailable, once known.
    pub unavailable: Option<String>,
    /// The device is an Apple GPU (the family of the measured cost rules);
    /// `None` until known.
    pub apple_gpu: Option<bool>,
    /// The device is already open (no setup to pay).
    pub open: bool,
}

/// An opaque handle to a compute device. Analyses downcast it to the
/// interface they need (`mklight-gpu`'s compute-backend trait object once
/// that crate is wired in).
pub type DeviceHandle = Arc<dyn Any + Send + Sync>;

/// Implemented by GPU backend crates to describe and provide a device.
/// Integration point: `mklight-gpu` / `mklight-metal` (`crate::gpu`).
pub trait BackendProvider: Send + Sync {
    /// `BackendInfo::id`, without probing the device (providers that open
    /// devices lazily override it).
    fn id(&self) -> String {
        self.info().id
    }
    fn info(&self) -> BackendInfo;
    /// The device analyses compute on; `None` for the CPU reference.
    fn device(&self) -> Option<DeviceHandle> {
        None
    }
    /// What `info` would report, without opening a device; `None` when the
    /// provider cannot tell without opening it.
    fn hint(&self) -> Option<ProviderHint> {
        None
    }
}

/// `auto`'s first candidate for an analysis, from provider hints.
#[derive(Clone, Debug, PartialEq)]
pub enum AutoCandidate {
    /// No registered GPU implements it: the reference, and why.
    NoGpu(String),
    /// The first GPU provider `auto` would select, with its device family
    /// when known (`ProviderHint::apple_gpu`) and whether it is open.
    Gpu { apple_gpu: Option<bool>, open: bool },
}

/// The outcome of backend selection for one analysis run.
#[derive(Clone)]
pub struct SelectedBackend {
    pub info: BackendInfo,
    pub device: Option<DeviceHandle>,
    /// Set when the run fell back to the CPU reference; goes into `RunInfo`.
    pub warning: Option<String>,
    /// What was requested (`BackendReport::requested`).
    pub requested: String,
    /// Why the preferred backend was not selected, without a warning
    /// (`auto` and engine-default preferences): `BackendReport::fallback_reason`.
    pub note: Option<String>,
    /// The cost decision when `auto` settled the route before opening a
    /// device (`BackendReport::cost_decision`).
    pub cost_decision: Option<serde_json::Value>,
}

impl std::fmt::Debug for SelectedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedBackend")
            .field("info", &self.info)
            .field("device", &self.device.is_some())
            .field("warning", &self.warning)
            .field("requested", &self.requested)
            .field("note", &self.note)
            .field("cost_decision", &self.cost_decision)
            .finish()
    }
}

impl SelectedBackend {
    pub fn cpu_reference() -> Self {
        SelectedBackend {
            info: BackendInfo::cpu_reference(),
            device: None,
            warning: None,
            requested: Backend::CpuReference.id(),
            note: None,
            cost_decision: None,
        }
    }
}

struct Static(BackendInfo);
impl BackendProvider for Static {
    fn info(&self) -> BackendInfo {
        self.0.clone()
    }
    fn hint(&self) -> Option<ProviderHint> {
        let i = &self.0;
        Some(ProviderHint {
            reference: i.capabilities.reference,
            analyses: i.capabilities.analyses.clone(),
            unavailable: (!i.available).then(|| i.reason.clone().unwrap_or_default()),
            apple_gpu: None,
            open: false,
        })
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
    /// The CPU reference, plus on macOS the Metal backend (opened on first
    /// use; unavailable without a qualifying device).
    pub fn with_defaults() -> Self {
        let mut r = Self::cpu_only();
        if cfg!(target_os = "macos") {
            r.register(Arc::new(crate::gpu::GpuProvider::metal()));
        }
        r
    }

    /// Only the CPU reference.
    pub fn cpu_only() -> Self {
        BackendRegistry {
            providers: vec![Arc::new(Static(BackendInfo::cpu_reference()))],
        }
    }

    /// Add a provider, replacing one with the same id.
    pub fn register(&mut self, p: Arc<dyn BackendProvider>) {
        let id = p.id();
        self.providers.retain(|q| q.id() != id);
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
            .find(|p| p.id() == id)
            .map(|p| (p, p.info()))
    }

    /// `auto`: the first available GPU backend implementing `analysis`
    /// (registration order), else the CPU reference with the reason as a
    /// note.
    pub(crate) fn select_auto(&self, analysis: Option<&str>) -> SelectedBackend {
        let mut why = Vec::new();
        for p in &self.providers {
            let i = p.info();
            if i.capabilities.reference {
                continue;
            }
            if !i.available {
                why.push(format!(
                    "{}: {}",
                    i.id,
                    i.reason.as_deref().unwrap_or("unavailable")
                ));
                continue;
            }
            match analysis {
                Some(a) if !i.supports(a) => why.push(format!("{}: no GPU path for '{a}'", i.id)),
                _ => {
                    return SelectedBackend {
                        device: p.device(),
                        info: i,
                        warning: None,
                        requested: Backend::Auto.id(),
                        note: None,
                        cost_decision: None,
                    };
                }
            }
        }
        let mut s = SelectedBackend::cpu_reference();
        s.requested = Backend::Auto.id();
        s.note = Some(if why.is_empty() {
            "no GPU backend is registered".into()
        } else {
            why.join("; ")
        });
        s
    }

    /// `select_auto` from provider hints, without opening any device. The
    /// reasons match `select_auto`'s note. `None` when a provider can only
    /// tell by opening its device.
    pub fn auto_candidate(&self, analysis: &str) -> Option<AutoCandidate> {
        let mut why = Vec::new();
        for p in &self.providers {
            let h = p.hint()?;
            if h.reference {
                continue;
            }
            if let Some(reason) = h.unavailable {
                let reason = if reason.is_empty() {
                    "unavailable".into()
                } else {
                    reason
                };
                why.push(format!("{}: {reason}", p.id()));
            } else if !h.analyses.iter().any(|a| a == analysis) {
                why.push(format!("{}: no GPU path for '{analysis}'", p.id()));
            } else {
                return Some(AutoCandidate::Gpu {
                    apple_gpu: h.apple_gpu,
                    open: h.open,
                });
            }
        }
        Some(AutoCandidate::NoGpu(if why.is_empty() {
            "no GPU backend is registered".into()
        } else {
            why.join("; ")
        }))
    }

    /// Resolve a request for running `analysis` (`None`: only check that
    /// the backend is available). A backend that is unavailable or does not
    /// implement the analysis is an error, or, with `allow_fallback`, the CPU
    /// reference plus a warning for `RunInfo`. `auto` never fails.
    pub fn select(
        &self,
        requested: &Backend,
        analysis: Option<&str>,
        allow_fallback: bool,
    ) -> EngineResult<SelectedBackend> {
        if *requested == Backend::Auto {
            return Ok(self.select_auto(analysis));
        }
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
                        requested: requested.id(),
                        note: None,
                        cost_decision: None,
                    });
                }
            },
        };
        if allow_fallback && *requested != Backend::CpuReference {
            let mut s = SelectedBackend::cpu_reference();
            s.requested = requested.id();
            s.note = Some(why.clone());
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
    fn auto_prefers_a_gpu_that_implements_the_analysis() {
        let mut r = BackendRegistry::cpu_only();
        let s = r
            .select(&Backend::Auto, Some("spot_diagram"), false)
            .unwrap();
        assert_eq!(s.info.backend, Backend::CpuReference);
        assert_eq!(s.requested, "auto");
        assert!(s.warning.is_none() && s.note.is_some());
        r.register(Arc::new(Emu));
        let s = r
            .select(&Backend::Auto, Some("spot_diagram"), false)
            .unwrap();
        assert_eq!(s.info.id, "gpu:emu");
        assert!(s.device.is_some() && s.note.is_none());
        let s = r.select(&Backend::Auto, Some("validate"), false).unwrap();
        assert_eq!(s.info.backend, Backend::CpuReference);
        assert!(s.note.unwrap().contains("no GPU path for 'validate'"));
        assert_eq!(Backend::parse("auto"), Backend::Auto);
        assert_eq!(
            serde_json::to_value(Backend::Auto).unwrap(),
            serde_json::json!({"kind": "auto"})
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
