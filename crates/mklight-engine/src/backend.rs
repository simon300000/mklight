//! Compute-backend selection.
//!
//! The CPU f64 reference is always present. GPU backends (`mklight-gpu`
//! implementations such as Metal) register a `BackendProvider` once they are
//! wired in; until then a Metal entry is listed as unavailable on macOS so
//! front ends can show the choice. Requesting an unavailable backend is an
//! error unless the caller allows fallback, in which case the job runs on the
//! CPU reference and its `RunInfo` says so (CONVENTIONS §7: never silently
//! degrade).

use serde::{Deserialize, Serialize};
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

/// Implemented by GPU backend crates to describe (and later provide) a
/// device. Integration point: `mklight-gpu` / `mklight-metal`.
pub trait BackendProvider: Send + Sync {
    fn info(&self) -> BackendInfo;
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
        let id = b.id();
        self.providers.iter().map(|p| p.info()).find(|i| i.id == id)
    }

    /// Resolve a request to a usable backend. With `allow_fallback`, an
    /// unavailable backend falls back to the CPU reference and a warning
    /// for `RunInfo` is returned.
    pub fn select(
        &self,
        requested: &Backend,
        allow_fallback: bool,
    ) -> EngineResult<(BackendInfo, Option<String>)> {
        match self.info(requested) {
            Some(i) if i.available => Ok((i, None)),
            other => {
                let why = other
                    .and_then(|i| i.reason)
                    .unwrap_or_else(|| "unknown backend".into());
                if allow_fallback && *requested != Backend::CpuReference {
                    Ok((
                        BackendInfo::cpu_reference(),
                        Some(format!(
                            "backend '{}' unavailable ({why}); ran on the CPU reference",
                            requested.id()
                        )),
                    ))
                } else {
                    Err(EngineError::BackendUnavailable(format!(
                        "{}: {why}",
                        requested.id()
                    )))
                }
            }
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
        let (i, w) = r.select(&Backend::CpuReference, false).unwrap();
        assert!(i.available && i.capabilities.fp64 && w.is_none());
        let gpu = Backend::Gpu("nonexistent".into());
        assert!(matches!(
            r.select(&gpu, false),
            Err(EngineError::BackendUnavailable(_))
        ));
        let (i, w) = r.select(&gpu, true).unwrap();
        assert_eq!(i.backend, Backend::CpuReference);
        assert!(w.unwrap().contains("gpu:nonexistent"));
    }

    #[test]
    fn providers_can_be_registered() {
        let mut r = BackendRegistry::with_defaults();
        let b = Backend::Gpu("emu".into());
        r.register(Arc::new(Static(BackendInfo {
            id: b.id(),
            backend: b.clone(),
            title: "emulated".into(),
            available: true,
            reason: None,
            precision: Precision::F32Gpu,
            capabilities: BackendCapabilities::default(),
        })));
        assert_eq!(r.select(&b, false).unwrap().0.precision, Precision::F32Gpu);
    }
}
