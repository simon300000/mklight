//! Pluggable analyses.
//!
//! An `Analysis` turns an immutable snapshot (and, when it needs one, the
//! `ResolvedSystem` built from it) plus a settings object into a JSON result
//! and optional typed buffers. Analyses are looked up by name in an
//! `AnalysisRegistry`; solver crates add theirs with `register` (see
//! `builtin::PLANNED` for the names reserved for them).
//!
//! `run_analysis` is the single execution path used both synchronously
//! (CLI, Python, tests) and by the job scheduler: it normalises settings
//! against the schema, runs the analysis and stamps the result with a
//! `RunInfo`.

pub mod builtin;
pub mod optim;

use serde::Serialize;
use serde_json::{Value, json};
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use mklight_core::job::RunTimer;
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkResult, ResolvedSystem, RunInfo, Snapshot};

use crate::backend::{BackendInfo, DeviceHandle};
use crate::buffer::{Buffer, BufferDesc};
use crate::error::{EngineError, EngineResult};
use crate::settings;

/// Static description of an analysis, for menus and forms.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalysisInfo {
    pub name: String,
    pub title: String,
    /// Grouping for menus: `model`, `materials`, `geometric`, `wave`,
    /// `ghost`, `stray`, `diagnostic`.
    pub category: String,
    pub description: String,
    /// False for reserved names whose solver is not wired in yet.
    pub available: bool,
    pub unavailable_reason: Option<String>,
    /// JSON-Schema subset (see `settings`).
    pub settings_schema: Value,
    /// Settings with every default filled in (null if some are required).
    pub default_settings: Value,
}

/// What an analysis returns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnalysisOutput {
    pub json: Value,
    pub buffers: Vec<Buffer>,
    /// Non-fatal problems, copied into `RunInfo.warnings`.
    pub warnings: Vec<String>,
}

/// A finished analysis: result plus reproducibility metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisResult {
    pub run_info: RunInfo,
    /// Content hash of the glass library the run resolved materials with
    /// (see `library_hash`). Together with `run_info.snapshot_hash` it
    /// identifies the inputs: a result is current only while both match the
    /// session's. (Moves into `RunInfo` once mklight-core carries it.)
    pub library_hash: String,
    pub json: Value,
    pub buffers: Vec<Buffer>,
}

impl AnalysisResult {
    /// The JSON envelope handed to front ends:
    /// `{"run_info": …, "library_hash": …, "result": …,
    /// "buffers": [{index, name, dtype, shape}]}`.
    pub fn envelope(&self) -> Value {
        let descs: Vec<BufferDesc> = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| b.desc(i))
            .collect();
        json!({
            "run_info": self.run_info,
            "library_hash": self.library_hash,
            "result": self.json,
            "buffers": descs,
        })
    }
}

type Resolver<'a> = Box<dyn Fn() -> MkResult<Arc<ResolvedSystem>> + 'a>;

/// BLAKE3 content hash (64 hex digits) of a material library: every
/// catalog's name and canonical JSON, length-prefixed, in load order (the
/// order matters because unqualified glass names resolve to the first
/// catalog that has them). Equal hashes mean every lookup gives the same
/// glass data.
pub fn library_hash(library: &MaterialLibrary) -> String {
    let mut h = blake3::Hasher::new();
    for c in library.catalogs() {
        let bytes = serde_json::to_vec(c.as_ref()).unwrap_or_default();
        for part in [c.name.as_bytes(), &bytes] {
            h.update(&(part.len() as u64).to_le_bytes());
            h.update(part);
        }
    }
    h.finalize().to_hex().to_string()
}

/// Everything an analysis may read. The resolved system is built lazily (and
/// normally comes from the session's cache), so analyses that do not need it
/// never pay for it, and analyses that report resolution errors (such as
/// `validate`) can still run.
pub struct AnalysisInput<'a> {
    pub snapshot: Snapshot,
    pub library: Arc<MaterialLibrary>,
    pub backend: BackendInfo,
    /// The selected backend's device (`None` on the CPU reference). GPU
    /// analyses downcast it to the compute interface they use.
    pub device: Option<DeviceHandle>,
    resolver: Resolver<'a>,
    resolved: OnceCell<MkResult<Arc<ResolvedSystem>>>,
    library_hash: OnceCell<String>,
}

impl<'a> AnalysisInput<'a> {
    /// Input that resolves the snapshot directly (no cache).
    pub fn new(snapshot: Snapshot, library: Arc<MaterialLibrary>, backend: BackendInfo) -> Self {
        let (s, l) = (snapshot.clone(), library.clone());
        Self::with_resolver(
            snapshot,
            library,
            backend,
            Box::new(move || ResolvedSystem::new(&s, l.clone()).map(Arc::new)),
        )
    }

    pub fn with_resolver(
        snapshot: Snapshot,
        library: Arc<MaterialLibrary>,
        backend: BackendInfo,
        resolver: Resolver<'a>,
    ) -> Self {
        AnalysisInput {
            snapshot,
            library,
            backend,
            device: None,
            resolver,
            resolved: OnceCell::new(),
            library_hash: OnceCell::new(),
        }
    }

    pub fn with_device(mut self, device: Option<DeviceHandle>) -> Self {
        self.device = device;
        self
    }

    /// Supply the library hash when the caller already knows it (sessions
    /// keep it), instead of hashing the library on first use.
    pub fn with_library_hash(self, hash: String) -> Self {
        let _ = self.library_hash.set(hash);
        self
    }

    /// See `library_hash`.
    pub fn library_hash(&self) -> &str {
        self.library_hash
            .get_or_init(|| library_hash(&self.library))
    }

    /// The resolved system (built on first use).
    pub fn resolved(&self) -> MkResult<Arc<ResolvedSystem>> {
        self.resolved.get_or_init(|| (self.resolver)()).clone()
    }

    fn resolved_warnings(&self) -> Vec<String> {
        match self.resolved.get() {
            Some(Ok(r)) => r.warnings.clone(),
            _ => Vec::new(),
        }
    }
}

pub trait Analysis: Send + Sync {
    /// Registry key (snake_case, stable: persisted in projects and RunInfo).
    fn name(&self) -> &str;
    fn title(&self) -> &str;
    fn category(&self) -> &str;
    fn description(&self) -> &str {
        ""
    }
    /// Settings schema (see `settings`). The default accepts no settings.
    fn settings_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    /// `Err(reason)` when the analysis cannot run in this build.
    fn availability(&self) -> Result<(), String> {
        Ok(())
    }
    /// The random seed used, for stochastic analyses (recorded in `RunInfo`).
    fn seed(&self, settings: &Value) -> Option<u64> {
        settings.get("seed").and_then(Value::as_u64)
    }
    /// Run with already-normalised settings. Long loops must call
    /// `ctx.check()` at bounded intervals and report progress.
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput>;

    fn info(&self) -> AnalysisInfo {
        let schema = self.settings_schema();
        let reason = self.availability().err();
        AnalysisInfo {
            name: self.name().to_string(),
            title: self.title().to_string(),
            category: self.category().to_string(),
            description: self.description().to_string(),
            available: reason.is_none(),
            unavailable_reason: reason,
            default_settings: settings::defaults(&schema).unwrap_or(Value::Null),
            settings_schema: schema,
        }
    }
}

/// Validate settings and check availability, without running anything.
/// Returns the normalised settings.
pub fn prepare(analysis: &dyn Analysis, settings: &Value) -> EngineResult<Value> {
    if let Err(reason) = analysis.availability() {
        return Err(EngineError::AnalysisUnavailable {
            name: analysis.name().to_string(),
            reason,
        });
    }
    settings::normalize(&analysis.settings_schema(), settings)
}

/// Run `analysis` and stamp the result. `extra_warnings` (for example a
/// backend fallback notice) are added to the `RunInfo`.
pub fn run_analysis(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    extra_warnings: Vec<String>,
    ctx: &JobContext,
) -> EngineResult<AnalysisResult> {
    let settings = prepare(analysis, settings)?;
    let timer = RunTimer::start();
    ctx.check()?;
    let out = analysis.run(input, &settings, ctx)?;
    ctx.check()?;
    let mut warnings = extra_warnings;
    for w in input.resolved_warnings().into_iter().chain(out.warnings) {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }
    let run_info = timer.finish(
        &input.snapshot,
        analysis.name(),
        &settings,
        &input.backend.id,
        input.backend.precision,
        analysis.seed(&settings),
        warnings,
    );
    ctx.report(1.0, "");
    Ok(AnalysisResult {
        run_info,
        library_hash: input.library_hash().to_string(),
        json: out.json,
        buffers: out.buffers,
    })
}

/// Analyses by name.
#[derive(Clone, Default)]
pub struct AnalysisRegistry {
    map: BTreeMap<String, Arc<dyn Analysis>>,
}

impl AnalysisRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Built-in analyses plus placeholders for the planned solver analyses.
    pub fn with_builtin() -> Self {
        let mut r = Self::new();
        builtin::register_all(&mut r);
        optim::register(&mut r);
        r
    }

    /// Add an analysis, replacing (and returning) one with the same name.
    /// This is how solver crates replace the placeholders.
    pub fn register(&mut self, a: Arc<dyn Analysis>) -> Option<Arc<dyn Analysis>> {
        self.map.insert(a.name().to_string(), a)
    }

    pub fn get(&self, name: &str) -> EngineResult<Arc<dyn Analysis>> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| EngineError::UnknownAnalysis(name.to_string()))
    }

    pub fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    /// Descriptions of all analyses, sorted by name.
    pub fn list(&self) -> Vec<AnalysisInfo> {
        self.map.values().map(|a| a.info()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::OpticalSystem;

    struct Echo;
    impl Analysis for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn title(&self) -> &str {
            "Echo"
        }
        fn category(&self) -> &str {
            "diagnostic"
        }
        fn settings_schema(&self) -> Value {
            json!({"type": "object", "properties": {
                "x": {"type": "number", "default": 2.0},
                "seed": {"type": "integer", "default": 9}}})
        }
        fn run(
            &self,
            input: &AnalysisInput<'_>,
            s: &Value,
            _: &JobContext,
        ) -> EngineResult<AnalysisOutput> {
            let n = input.resolved()?.num_surfaces();
            Ok(AnalysisOutput {
                json: json!({"x": s["x"], "n": n}),
                buffers: vec![Buffer::f64_1d("v", vec![1.0, 2.0])],
                warnings: vec!["w".into()],
            })
        }
    }

    #[test]
    fn run_stamps_run_info() {
        let mut reg = AnalysisRegistry::new();
        assert!(reg.register(Arc::new(Echo)).is_none());
        let a = reg.get("echo").unwrap();
        let snap = Snapshot::new(OpticalSystem::new_empty()).unwrap();
        let input = AnalysisInput::new(
            snap.clone(),
            Arc::new(MaterialLibrary::with_builtin()),
            BackendInfo::cpu_reference(),
        );
        let ctx = JobContext::new();
        let r = run_analysis(
            a.as_ref(),
            &input,
            &json!({"x": 5}),
            vec!["pre".into()],
            &ctx,
        )
        .unwrap();
        assert_eq!(r.json, json!({"x": 5, "n": 3}));
        assert_eq!(r.run_info.analysis, "echo");
        assert_eq!(r.run_info.snapshot_hash, snap.hash().to_hex());
        assert_eq!(r.run_info.settings, json!({"x": 5, "seed": 9}));
        assert_eq!(r.run_info.seed, Some(9));
        assert_eq!(r.run_info.backend, "cpu_reference");
        assert_eq!(r.run_info.warnings, vec!["pre", "w"]);
        let lib_hash = library_hash(&MaterialLibrary::with_builtin());
        assert_eq!(r.library_hash, lib_hash);
        let env = r.envelope();
        assert_eq!(env["library_hash"], lib_hash.as_str());
        assert_eq!(env["buffers"][0]["name"], "v");
        assert_eq!(env["buffers"][0]["dtype"], "f64");
        assert_eq!(ctx.progress.fraction(), 1.0);
        assert!(matches!(
            run_analysis(a.as_ref(), &input, &json!({"y": 1}), vec![], &ctx),
            Err(EngineError::InvalidSettings(_))
        ));
        assert!(matches!(
            reg.get("nope"),
            Err(EngineError::UnknownAnalysis(_))
        ));
    }

    #[test]
    fn library_hash_tracks_content_and_order() {
        use mklight_core::materials::GlassCatalog;
        let base = MaterialLibrary::with_builtin();
        let h0 = library_hash(&base);
        assert_eq!(h0.len(), 64);
        assert_eq!(h0, library_hash(&MaterialLibrary::with_builtin()));
        let builtin = base.catalogs()[0].as_ref().clone();
        let mut a = builtin.clone();
        a.name = "A".into();
        let mut lib = base.clone();
        lib.add_catalog(Arc::new(a.clone()));
        let h1 = library_hash(&lib);
        assert_ne!(h0, h1);
        // Same content again: same hash.
        lib.add_catalog(Arc::new(a.clone()));
        assert_eq!(library_hash(&lib), h1);
        // One coefficient changed: different hash.
        let mut a2 = a.clone();
        let g = a2.glasses.values_mut().next().unwrap();
        g.nd += 1e-12;
        lib.add_catalog(Arc::new(a2));
        assert_ne!(library_hash(&lib), h1);
        // Same catalogs, other order: different hash.
        let b = GlassCatalog {
            name: "B".into(),
            glasses: a.glasses.clone(),
        };
        let mut ab = base.clone();
        ab.add_catalog(Arc::new(a.clone()));
        ab.add_catalog(Arc::new(b.clone()));
        let mut ba = base.clone();
        ba.add_catalog(Arc::new(b));
        ba.add_catalog(Arc::new(a));
        assert_ne!(library_hash(&ab), library_hash(&ba));
    }

    #[test]
    fn cancelled_context_does_not_run() {
        let snap = Snapshot::new(OpticalSystem::new_empty()).unwrap();
        let input = AnalysisInput::new(
            snap,
            Arc::new(MaterialLibrary::with_builtin()),
            BackendInfo::cpu_reference(),
        );
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        let e = run_analysis(&Echo, &input, &Value::Null, vec![], &ctx).unwrap_err();
        assert!(e.is_cancelled());
    }
}
