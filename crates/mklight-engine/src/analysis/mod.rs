//! Pluggable analyses.
//!
//! An `Analysis` turns an immutable snapshot (and, when it needs one, the
//! `ResolvedSystem` built from it) plus a settings object into a JSON result
//! and optional typed buffers. Analyses are looked up by name in an
//! `AnalysisRegistry`. `register_all` registers every solver's analyses:
//!
//! | module | analyses |
//! |---|---|
//! | `builtin` | `validate`, `system_summary`, `glass_info`, `rng_stream` |
//! | `geometric` | everything in `mklight-analysis` (first-order data, layout, spots, fans, field curvature, Seidel, illumination, energy, …) and `ray_grid` |
//! | `ghost` | `ghost_paths`, `ghost_irradiance` (`mklight-ghost`) |
//! | `stray` | `stray_light` (`mklight-stray`, partitioned against the ghost solver) |
//! | `radiometry` | `system_transmission`, `polarization_pupil_map` (`mklight-trace`) |
//! | `solves` | `solves` (dry run of `Session::apply_solves`) |
//!
//! Each analysis publishes a settings schema with defaults and a result
//! schema (`schema`), documented in `docs/RESULTS.md`.
//!
//! `run_analysis` is the single execution path used both synchronously
//! (CLI, Python, tests) and by the job scheduler: it normalises settings
//! against the schema, runs the analysis and stamps the result with a
//! `RunInfo` naming the backend that actually computed it.

pub mod builtin;
pub mod cost;
pub mod extended;
pub mod geometric;
pub mod ghost;
pub mod gpu;
pub mod merit_evaluation;
pub mod optim;
pub mod radiometry;
mod routing;
pub mod schema;
pub mod solves;
pub mod spectrum;
pub mod stray;
pub mod wave;

pub(crate) use cost::is_apple as is_apple_gpu;
pub(crate) use routing::{AutoRequest, auto_route};

/// Graph key of the reference tracer of a snapshot resolved with a
/// library under a run policy (`AnalysisInput::tracer`): the policy sets
/// the tracer's aim stop.
pub(crate) fn tracer_key(snapshot: &[u8; 32], library_hash: &str, policy: &RunPolicy) -> NodeKey {
    let mut h = KeyHasher::new("mklight-engine/tracer/1");
    h.bytes(snapshot).str(library_hash).json(policy);
    h.finish()
}

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::{OnceCell, RefCell};
use std::collections::BTreeMap;
use std::sync::Arc;

use mklight_core::job::{AimStopKind, BackendAttempt, CancelToken, Precision, RunTimer, Tier};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkResult, ResolvedSystem, RunInfo, Snapshot};

use crate::backend::{Backend, BackendInfo, DeviceHandle};
use crate::buffer::{Buffer, BufferDesc};
use crate::error::{EngineError, EngineResult};
use crate::graph::{AnalysisGraph, ScopedMemo};
use crate::settings;
use mklight_core::memo::{KeyHasher, NodeKey};
use mklight_trace::{AimBudget, AimStop, SequentialTracer, TracerSettings};
use mklight_wave::traced::ConfiguredTracer;

/// The accuracy policies of one run (CONVENTIONS §7, GPU §9.4): the tier
/// and the aim-stop rule. Both are stamped into `RunInfo` and are part of
/// every graph key, so a result never silently mixes policies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunPolicy {
    pub tier: Tier,
    pub aim_stop: AimStopKind,
}

impl RunPolicy {
    /// The reference tracer's settings under this policy: the derived aim
    /// stop uses the tier's error allocation.
    pub fn tracer_settings(&self) -> TracerSettings {
        let aim_stop = match (self.aim_stop, self.tier) {
            (AimStopKind::Requested, _) => AimStop::Requested,
            (AimStopKind::Conditioned, Tier::Design) => AimStop::Conditioned(AimBudget::DESIGN),
            (AimStopKind::Conditioned, Tier::Preview) => AimStop::Conditioned(AimBudget::PREVIEW),
        };
        TracerSettings {
            aim_stop,
            ..Default::default()
        }
    }
}

/// Static description of an analysis, for menus and forms.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalysisInfo {
    pub name: String,
    pub title: String,
    /// Grouping for menus: `model`, `materials`, `geometric`, `wave`,
    /// `ghost`, `stray`, `radiometry`, `polarization`, `optimization`,
    /// `diagnostic`.
    pub category: String,
    pub description: String,
    /// False for reserved names whose solver is not wired in yet.
    pub available: bool,
    pub unavailable_reason: Option<String>,
    /// JSON-Schema subset (see `settings`).
    pub settings_schema: Value,
    /// Settings with every default filled in (null if some are required).
    pub default_settings: Value,
    /// JSON Schema of `result`, with the buffers under `x-buffers`
    /// (`schema`, `docs/RESULTS.md`).
    pub result_schema: Value,
    /// Version of the result shape (bumped on incompatible changes).
    pub result_version: u32,
    /// The analysis can run on a GPU backend (`auto` selects one).
    pub gpu_path: bool,
}

/// What an analysis returns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnalysisOutput {
    pub json: Value,
    pub buffers: Vec<Buffer>,
    /// Non-fatal problems, copied into `RunInfo.warnings`.
    pub warnings: Vec<String>,
}

/// Which backend computed a result, and why (`AnalysisResult::backend`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BackendReport {
    /// What was asked for: a backend id, `auto`, or `default:<id>` when the
    /// request named none and the engine default applied.
    pub requested: String,
    /// The backend id that computed the result (= `run_info.backend`).
    pub used: String,
    pub precision: Precision,
    /// GPU device name, when a GPU computed the result.
    pub device: Option<String>,
    /// Why the requested (or automatically preferred) backend was not used:
    /// unavailable, no GPU path, a static capability restriction, a cost
    /// decision of `auto`, or a native run that failed. In
    /// `run_info.warnings` too, except for the deliberate routes of an
    /// `auto` or default preference (no GPU path, static capability, cost
    /// decision).
    pub fallback_reason: Option<String>,
    /// Native execution evidence: device/build/probe, plan versions and stages.
    pub execution: Vec<Value>,
    pub cost_decision: Option<Value>,
}

/// What an analysis tells the engine about the backend it actually used
/// (`AnalysisInput::report_backend`).
#[derive(Clone, Debug, PartialEq)]
pub struct BackendUse {
    pub id: String,
    pub precision: Precision,
    pub device: Option<String>,
    /// Set when the selected GPU was not used; becomes a `RunInfo` warning
    /// unless `deliberate`.
    pub fallback_reason: Option<String>,
    /// The fallback is a policy choice of `auto` (for example a grid too
    /// small to gain from the GPU), not a problem: it is reported in
    /// `BackendReport::fallback_reason` only, not as a warning.
    pub deliberate: bool,
}

/// A finished analysis: result plus reproducibility metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisResult {
    pub run_info: RunInfo,
    /// Requested and used backend, fallback reason and device execution evidence.
    pub backend: BackendReport,
    /// Version of the result shape (`Analysis::result_version`).
    pub result_version: u32,
    /// Content hash of the glass library the run resolved materials with
    /// (see `library_hash`). Together with `run_info.snapshot_hash` it
    /// identifies the inputs: a result is current only while both match the
    /// session's. (Moves into `RunInfo` once mklight-core carries it.)
    pub library_hash: String,
    /// Run-local spectrum snapshot; the submitted snapshot stays in RunInfo.
    pub derived_snapshot_hash: Option<String>,
    pub json: Value,
    pub buffers: Vec<Buffer>,
}

impl AnalysisResult {
    /// The JSON envelope handed to front ends:
    /// `{"run_info": …, "library_hash": …, "backend": …, "result_version": …,
    /// "result": …, "buffers": [{index, name, dtype, shape}]}`.
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
            "derived_snapshot_hash": self.derived_snapshot_hash,
            "backend": self.backend,
            "result_version": self.result_version,
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
    /// What the request asked for (`BackendReport::requested`).
    pub requested_backend: String,
    /// Why the preferred backend was not selected (`BackendReport`).
    pub selection_note: Option<String>,
    /// Job-level stop that retains the best result (optimisation/tolerancing).
    pub stop: CancelToken,
    resolver: Resolver<'a>,
    backend_used: RefCell<Option<BackendUse>>,
    executions: RefCell<Vec<Value>>,
    cost_decision: RefCell<Option<Value>>,
    attempts: RefCell<Vec<BackendAttempt>>,
    resolved: OnceCell<MkResult<Arc<ResolvedSystem>>>,
    library_hash: OnceCell<String>,
    /// The engine's analysis graph (`crate::graph`); `None` builds every
    /// intermediate.
    graph: Option<Arc<AnalysisGraph>>,
    /// The engine's cost model, told about every completed run.
    cost_model: Option<Arc<cost::CostModel>>,
    /// Tier and aim-stop policy of this run ([`RunPolicy`]).
    pub policy: RunPolicy,
}

/// Rough bytes of a tracer node (fields × wavelengths of aim data).
fn tracer_bytes(t: &SequentialTracer) -> usize {
    4096 + t.fields().len() * (256 + 96 * t.num_wavelengths())
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
            requested_backend: Backend::CpuReference.id(),
            selection_note: None,
            stop: CancelToken::new(),
            resolver,
            backend_used: RefCell::new(None),
            executions: RefCell::new(Vec::new()),
            cost_decision: RefCell::new(None),
            attempts: RefCell::new(Vec::new()),
            resolved: OnceCell::new(),
            library_hash: OnceCell::new(),
            graph: None,
            cost_model: None,
            policy: RunPolicy::default(),
        }
    }

    /// An input of the same run for another snapshot or backend: the
    /// library and its hash, the run policy, the analysis graph and the
    /// job-level stop carry over. Device, selection and cost decision are
    /// the caller's to set.
    pub(crate) fn derive<'b>(
        &self,
        snapshot: Snapshot,
        backend: BackendInfo,
        resolver: Resolver<'b>,
    ) -> AnalysisInput<'b> {
        AnalysisInput::with_resolver(snapshot, self.library.clone(), backend, resolver)
            .with_library_hash(self.library_hash().to_string())
            .with_policy(self.policy)
            .with_graph(self.graph.clone())
            .with_stop_token(self.stop.clone())
    }

    /// Report completed runs to the engine's cost model (`cost`).
    pub(crate) fn with_cost_model(mut self, model: Option<Arc<cost::CostModel>>) -> Self {
        self.cost_model = model;
        self
    }

    /// Run under `policy` (tier and aim stop); the default is design grade
    /// at the requested aiming tolerance.
    pub fn with_policy(mut self, policy: RunPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Share intermediates through the engine's analysis graph.
    pub fn with_graph(mut self, graph: Option<Arc<AnalysisGraph>>) -> Self {
        self.graph = graph;
        self
    }

    pub(crate) fn graph(&self) -> Option<&Arc<AnalysisGraph>> {
        self.graph.as_ref()
    }

    /// Key of the snapshot resolved with the library (material
    /// resolution): the root of every node of this input.
    fn content_key(&self, domain: &str) -> KeyHasher {
        let mut h = KeyHasher::new(domain);
        h.bytes(&self.snapshot.hash().0)
            .str(self.library_hash())
            .json(&self.policy);
        h
    }

    /// The reference tracer of the resolved snapshot (first order and
    /// prepared system fields), shared through the graph.
    pub(crate) fn tracer(&self, ctx: &JobContext) -> EngineResult<Arc<SequentialTracer>> {
        let build =
            || SequentialTracer::with_settings(self.resolved()?, self.policy.tracer_settings());
        let Some(g) = &self.graph else {
            return Ok(Arc::new(build()?));
        };
        let key = tracer_key(&self.snapshot.hash().0, self.library_hash(), &self.policy);
        Ok(g.node("tracer", key, ctx, tracer_bytes, build)?)
    }

    /// [`Self::tracer`] with the default axis fields of a geometric field
    /// selection (`mklight_analysis::util::default_fields`).
    pub(crate) fn field_tracer(
        &self,
        fields: &Value,
        ctx: &JobContext,
    ) -> EngineResult<Arc<SequentialTracer>> {
        let tracer = self.tracer(ctx)?;
        let selection: mklight_analysis::util::FieldSelection = geometric::parse(fields)?;
        // `None`: the selection keeps the system fields (the base tracer).
        let build = || {
            Ok::<_, mklight_core::MkError>(
                match mklight_analysis::util::default_fields(&tracer, &selection)? {
                    std::borrow::Cow::Borrowed(_) => None,
                    std::borrow::Cow::Owned(t) => Some(Arc::new(t)),
                },
            )
        };
        let extended = match &self.graph {
            None => Arc::new(build()?),
            Some(g) => {
                let key = self
                    .content_key("mklight-engine/field_tracer/1")
                    .json(fields)
                    .finish();
                g.node(
                    "field_tracer",
                    key,
                    ctx,
                    |t: &Option<Arc<SequentialTracer>>| t.as_deref().map_or(64, tracer_bytes),
                    build,
                )?
            }
        };
        Ok(match &*extended {
            None => tracer,
            Some(t) => t.clone(),
        })
    }

    /// Configuration `k` with its solves applied and its reference tracer
    /// (`mklight_wave::traced::configured_tracer_job`), shared through the
    /// graph.
    pub(crate) fn configured(
        &self,
        k: usize,
        ctx: &JobContext,
    ) -> EngineResult<Arc<ConfiguredTracer>> {
        let build = || {
            mklight_wave::traced::configured_tracer_job_with(
                self.snapshot.system(),
                self.library.clone(),
                k,
                self.policy.tracer_settings(),
                ctx,
            )
        };
        let Some(g) = &self.graph else {
            return Ok(Arc::new(build()?));
        };
        let key = self
            .content_key("mklight-engine/configured/1")
            .usize(k)
            .json(&mklight_trace::SolveSettings::default())
            .finish();
        Ok(g.node(
            "configured",
            key,
            ctx,
            |c: &ConfiguredTracer| tracer_bytes(&c.tracer),
            build,
        )?)
    }

    /// A memo for solver-side nodes (`mklight_wave::shared`), scoped by the
    /// material library the solver cannot see.
    pub(crate) fn memo(&self) -> Option<ScopedMemo> {
        let g = self.graph.as_ref()?;
        let scope: NodeKey = KeyHasher::new("mklight-engine/library/1")
            .str(self.library_hash())
            .finish();
        Some(g.scoped(scope))
    }

    pub fn with_stop_token(mut self, stop: CancelToken) -> Self {
        self.stop = stop;
        self
    }

    pub fn with_device(mut self, device: Option<DeviceHandle>) -> Self {
        self.device = device;
        self
    }

    /// Record what was requested and why the preference was not selected
    /// (see `SelectedBackend`).
    pub fn with_selection(mut self, requested: impl Into<String>, note: Option<String>) -> Self {
        self.requested_backend = requested.into();
        self.selection_note = note;
        self
    }

    /// The cost decision when `auto` chose the backend before opening a
    /// device (`SelectedBackend::cost_decision`).
    pub fn with_cost_decision(self, decision: Option<Value>) -> Self {
        *self.cost_decision.borrow_mut() = decision;
        self
    }

    /// Record a backend attempt that did not produce the result
    /// (`RunInfo::failed_attempts`).
    pub(crate) fn record_attempt(&self, attempt: BackendAttempt) {
        self.attempts.borrow_mut().push(attempt);
    }

    /// Record the backend that actually computed the result, when it is not
    /// the selected one (a GPU analysis that fell back to the CPU reference,
    /// or one that ran on the GPU with device details). The last report
    /// wins.
    pub fn report_backend(&self, used: BackendUse) {
        *self.backend_used.borrow_mut() = Some(used);
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
    /// JSON Schema of the result (`schema`); the default documents nothing
    /// beyond "an object".
    fn result_schema(&self) -> Value {
        json!({"type": "object", "additionalProperties": true, "x-buffers": []})
    }
    /// Version of the result shape: bump it when a field changes meaning
    /// or disappears (adding fields keeps the version).
    fn result_version(&self) -> u32 {
        1
    }
    /// True when the analysis has a GPU implementation (it then reads
    /// `AnalysisInput::device` and reports what it used): its name is in
    /// the catalogue the GPU providers advertise (`gpu::GPU_ANALYSES`).
    fn gpu_path(&self) -> bool {
        gpu::GPU_ANALYSES.contains(&self.name())
    }
    /// The result is a deterministic function of the snapshot, the
    /// material library, the normalised settings and the backend selection
    /// (no files, clocks or job-level stops), so the engine may answer an
    /// identical request from its analysis graph (`crate::graph`). Opt-in.
    fn memoizable(&self) -> bool {
        false
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
        let schema =
            spectrum::settings_schema(self.settings_schema(), self.category(), self.name());
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
            result_schema: self.result_schema(),
            result_version: self.result_version(),
            gpu_path: self.gpu_path(),
        }
    }
}

/// Validate settings and check availability, without running anything.
/// Returns the normalised settings.
pub fn prepare(analysis: &dyn Analysis, settings: &Value) -> EngineResult<Value> {
    prepare_tier(analysis, settings, Tier::Design)
}

/// [`prepare`] at a tier: the preview tier takes an analysis' `x-preview`
/// defaults (cheaper sampling) for every setting the request leaves out, so
/// `RunInfo.settings` always states the sampling actually used.
pub fn prepare_tier(analysis: &dyn Analysis, settings: &Value, tier: Tier) -> EngineResult<Value> {
    if let Err(reason) = analysis.availability() {
        return Err(EngineError::AnalysisUnavailable {
            name: analysis.name().to_string(),
            reason,
        });
    }
    let schema = spectrum::settings_schema(
        analysis.settings_schema(),
        analysis.category(),
        analysis.name(),
    );
    settings::normalize_tier(&schema, settings, tier == Tier::Preview)
}

/// Run `analysis` and stamp the result. `extra_warnings` (for example a
/// backend fallback notice) are added to the `RunInfo`.
///
/// With an analysis graph, a [`Analysis::memoizable`] request identical to
/// an earlier one (same snapshot content, library, normalised settings and
/// policy; for an explicit backend also the backend selection) returns the
/// stored result, restamped with this call's start time and duration;
/// identical concurrent requests run once. Results without failed backend
/// attempts are stored: an `auto` result whichever backend computed it, an
/// explicit request's only from the CPU reference.
pub fn run_analysis(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    extra_warnings: Vec<String>,
    ctx: &JobContext,
) -> EngineResult<AnalysisResult> {
    let settings = prepare(analysis, settings)?;
    ctx.check()?;
    let Some(graph) = input.graph().filter(|_| analysis.memoizable()) else {
        return run_prepared(analysis, input, &settings, extra_warnings, ctx);
    };
    let key = result_key(analysis, input, &settings, &extra_warnings);
    let auto = Backend::is_auto_request(&input.requested_backend);
    let timer = RunTimer::start();
    let started = std::time::Instant::now();
    let mut built = false;
    let r = graph.single("result", key, ctx, || {
        built = true;
        let r = run_prepared(analysis, input, &settings, extra_warnings, ctx)?;
        // An `auto` result is stored whichever backend computed it (its
        // key does not depend on the route); explicit native results are
        // not.
        let storable = r.run_info.failed_attempts.is_empty()
            && (auto || r.backend.used == Backend::CpuReference.id());
        let bytes = if storable {
            result_bytes(&r)
        } else {
            crate::graph::UNSTORED
        };
        Ok::<_, EngineError>((r, bytes))
    })?;
    let mut r = Arc::try_unwrap(r).unwrap_or_else(|r| (*r).clone());
    if !built {
        // This call's start time (`RunTimer` keeps it private).
        let fresh = timer.finish(
            &input.snapshot,
            "",
            &Value::Null,
            "",
            r.run_info.precision,
            None,
            vec![],
        );
        r.run_info.started_unix_ms = fresh.started_unix_ms;
        r.run_info.duration_ms = started.elapsed().as_secs_f64() * 1e3;
        ctx.report(1.0, "");
    }
    Ok(r)
}

/// Graph key of a result. An `auto` request's key is its content alone
/// (snapshot, library, analysis, settings): a stored answer is returned
/// before routing, whichever backend computed it, so a repeated request
/// costs nothing on either backend. Explicit requests also key the backend
/// selection, its precision and reasons.
pub(crate) fn result_key(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    extra_warnings: &[String],
) -> NodeKey {
    let mut h = KeyHasher::new("mklight-engine/result/1");
    h.str(mklight_core::ENGINE_VERSION)
        .bytes(&input.snapshot.hash().0)
        .str(input.library_hash())
        .str(analysis.name())
        .u64(analysis.result_version() as u64)
        .json(settings)
        .json(&input.policy)
        // The device divisions in effect (process-wide route options).
        .json(&mklight_core::split::selection())
        .str(&input.requested_backend);
    if !Backend::is_auto_request(&input.requested_backend) {
        h.str(&input.backend.id)
            .json(&input.backend.precision)
            .json(&input.selection_note)
            .json(&*input.cost_decision.borrow());
    }
    h.json(&extra_warnings.to_vec());
    h.finish()
}

/// Bytes held by a stored result (buffers plus an estimate of the JSON).
fn result_bytes(r: &AnalysisResult) -> usize {
    fn json(v: &Value) -> usize {
        32 + match v {
            Value::String(s) => s.len(),
            Value::Array(a) => a.iter().map(json).sum(),
            Value::Object(o) => o.iter().map(|(k, v)| k.len() + json(v)).sum(),
            _ => 0,
        }
    }
    let buffers: usize = r
        .buffers
        .iter()
        .map(|b| b.data.len() * b.data.dtype().size_bytes())
        .sum();
    1024 + buffers + json(&r.json)
}

fn run_prepared(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    extra_warnings: Vec<String>,
    ctx: &JobContext,
) -> EngineResult<AnalysisResult> {
    let settings = settings.clone();
    let timer = RunTimer::start();
    ctx.check()?;
    // A run-local spectrum replaces the design's wavelengths; the solver
    // sees the derived snapshot, RunInfo records the full settings.
    let mut solver_settings = settings.clone();
    let derived = match solver_settings
        .as_object_mut()
        .and_then(|o| o.remove("spectrum"))
        .filter(|v| !v.is_null())
    {
        Some(v) => Some(spectrum::derived(input, v)?),
        None => None,
    };
    let submitted = input;
    let submitted_snapshot = &input.snapshot;
    let input = derived.as_ref().unwrap_or(input);
    let out = routing::run(analysis, input, &solver_settings, ctx)?;
    ctx.check()?;
    let mut backend = BackendReport {
        requested: input.requested_backend.clone(),
        used: input.backend.id.clone(),
        precision: input.backend.precision,
        device: input.backend.capabilities.device_name.clone(),
        fallback_reason: input.selection_note.clone(),
        execution: input.executions.borrow_mut().drain(..).collect(),
        cost_decision: input.cost_decision.borrow_mut().take(),
    };
    let mut warnings = extra_warnings;
    if let Some(u) = input.backend_used.borrow_mut().take() {
        backend.used = u.id;
        backend.precision = u.precision;
        backend.device = u.device;
        if let Some(r) = u.fallback_reason {
            if !u.deliberate {
                warnings.push(format!(
                    "backend '{}' not used ({r}); computed with the CPU f64 reference",
                    input.backend.id
                ));
            }
            backend.fallback_reason = Some(r);
        }
    }
    if backend.used == Backend::CpuReference.id() {
        backend.device = None;
    }
    let attempts = input.attempts.take();
    for a in attempts.iter().filter(|a| a.outcome == "downgraded") {
        warnings.push(format!(
            "backend '{}' downgraded to the CPU f64 reference ({})",
            a.backend, a.reason
        ));
    }
    for w in input.resolved_warnings().into_iter().chain(out.warnings) {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }
    let mut run_info = timer.finish(
        submitted_snapshot,
        analysis.name(),
        &settings,
        &backend.used,
        backend.precision,
        analysis.seed(&settings),
        warnings,
    );
    run_info.failed_attempts = attempts;
    run_info.tier = input.policy.tier;
    run_info.aim_stop = input.policy.aim_stop;
    ctx.report(1.0, "");
    let result = AnalysisResult {
        run_info,
        backend,
        result_version: analysis.result_version(),
        library_hash: input.library_hash().to_string(),
        derived_snapshot_hash: derived.as_ref().map(|i| i.snapshot.hash().to_hex()),
        json: out.json,
        buffers: out.buffers,
    };
    if let Some(m) = &submitted.cost_model {
        m.observe(
            analysis.name(),
            &settings,
            submitted_snapshot.system(),
            &submitted_snapshot.hash().0,
            submitted.library_hash(),
            &(&result).into(),
        );
    }
    Ok(result)
}

/// Register every analysis of this build.
pub fn register_all(r: &mut AnalysisRegistry) {
    builtin::register_all(r);
    geometric::register_all(r);
    wave::register_all(r);
    extended::register_all(r);
    ghost::register_all(r);
    stray::register_all(r);
    radiometry::register_all(r);
    solves::register_all(r);
    optim::register_all(r);
    merit_evaluation::register_all(r);
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

    /// Every analysis of this build (`register_all`).
    pub fn with_builtin() -> Self {
        let mut r = Self::new();
        register_all(&mut r);
        r
    }

    /// Add an analysis, replacing (and returning) one with the same name.
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

    /// The preview tier samples no denser than the design tier: a preview
    /// default never exceeds the setting's design default.
    #[test]
    fn preview_defaults_do_not_exceed_design_defaults() {
        let registry = AnalysisRegistry::with_builtin();
        for info in registry.list() {
            let properties = info.settings_schema["properties"].as_object().unwrap();
            for (key, p) in properties {
                if let (Some(preview), Some(design)) =
                    (p["x-preview"].as_f64(), p["default"].as_f64())
                {
                    assert!(preview <= design, "{}.{key}", info.name);
                }
            }
        }
        let rings = |name: &str, tier| {
            let a = registry.get(name).unwrap();
            prepare_tier(a.as_ref(), &Value::Null, tier).unwrap()["rings"].clone()
        };
        // Sparse defaults stay; the dense square default halves.
        for name in ["footprint", "best_focus", "ray_grid"] {
            assert_eq!(rings(name, Tier::Preview), rings(name, Tier::Design));
        }
        assert_eq!(rings("encircled_energy", Tier::Design), 32);
        assert_eq!(rings("encircled_energy", Tier::Preview), 16);
    }

    /// The reference tracer carries the policy's aim stop, so the graph
    /// holds one per policy: runs of one snapshot under different tiers or
    /// aim stops never share a tracer.
    #[test]
    fn the_graph_keeps_one_tracer_per_policy() {
        let graph = Arc::new(AnalysisGraph::new(1 << 24));
        let snap = Snapshot::new(mklight_core::samples::cooke_triplet()).unwrap();
        let library = Arc::new(MaterialLibrary::with_builtin());
        let ctx = JobContext::new();
        let tracer = |policy: RunPolicy| {
            AnalysisInput::new(snap.clone(), library.clone(), BackendInfo::cpu_reference())
                .with_graph(Some(graph.clone()))
                .with_policy(policy)
                .tracer(&ctx)
                .unwrap()
        };
        let design = RunPolicy {
            tier: Tier::Design,
            aim_stop: AimStopKind::Conditioned,
        };
        let preview = RunPolicy {
            tier: Tier::Preview,
            ..design
        };
        let first = tracer(design);
        assert!(Arc::ptr_eq(&first, &tracer(design)));
        assert_eq!(
            first.settings().aim_stop,
            AimStop::Conditioned(AimBudget::DESIGN)
        );
        for (other, stop) in [
            (preview, AimStop::Conditioned(AimBudget::PREVIEW)),
            (RunPolicy::default(), AimStop::Requested),
        ] {
            let t = tracer(other);
            assert!(!Arc::ptr_eq(&first, &t), "{other:?}");
            assert_eq!(t.settings().aim_stop, stop);
            let key =
                |p: &RunPolicy| tracer_key(&snap.hash().0, library_hash(&library).as_str(), p);
            assert_ne!(key(&design), key(&other));
            assert!(graph.contains(&key(&other)));
        }
    }
}
