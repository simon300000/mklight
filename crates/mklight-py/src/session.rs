//! `mklight.Engine` and `mklight.Session`: thin wrappers over the engine
//! facade (`mklight-engine`), so Python, the C ABI and the CLI share one
//! code path for documents, edits, catalogs and analyses.
//!
//! Locking: a `Session` is a `Mutex<mklight_engine::Session>`. Python
//! objects are converted before the lock is taken and after it is released,
//! and slow operations (file I/O, validation) run with the GIL released, so
//! the session lock and the GIL are never waited for while the other is
//! held.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyDict;
use serde_json::{Value, json};

use mklight_core::{OpticalSystem, samples};
use mklight_engine::backend::Backend;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, EngineError, ParamEdit, Session};
use mklight_trace::SequentialTracer;

use crate::convert::{from_py, ser_to_py, to_py};
use crate::errors::{ErrorKind, engine_err, new_err};
use crate::job::{PyAnalysisResult, PyJob};

/// Look up a sample system by its identifier (see `mklight.samples()`).
pub fn sample_system(name: &str) -> Result<OpticalSystem, String> {
    samples::all()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s)
        .ok_or_else(|| {
            let names: Vec<_> = samples::all().into_iter().map(|(n, _)| n).collect();
            format!("unknown sample '{name}' (known: {})", names.join(", "))
        })
}

/// Parse the `backend` argument: `None` → engine default, `"auto"` → automatic selection,
/// `"cpu"`/`"cpu_reference"` → CPU reference, `"metal"`/`"gpu:metal"` → a
/// GPU backend, or the JSON form `{"kind": "gpu", "name": "metal"}`.
pub fn parse_backend(v: &Value) -> Result<Option<Backend>, String> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(Backend::parse(s))),
        Value::Object(_) => serde_json::from_value(v.clone())
            .map(Some)
            .map_err(|e| format!("invalid backend {v}: {e}")),
        other => Err(format!("invalid backend {other}")),
    }
}

/// Merge `seed` into `settings`. The analysis must declare a `seed`
/// setting; a conflicting explicit `settings["seed"]` is an error.
pub fn apply_seed(schema: &Value, settings: Value, seed: Option<u64>) -> Result<Value, String> {
    let Some(seed) = seed else {
        return Ok(settings);
    };
    if schema.pointer("/properties/seed").is_none() {
        return Err("this analysis takes no seed (it is deterministic)".into());
    }
    let mut obj = match settings {
        Value::Null => serde_json::Map::new(),
        Value::Object(o) => o,
        other => return Err(format!("settings must be a dict, got {other}")),
    };
    if let Some(s) = obj.get("seed")
        && s.as_u64() != Some(seed)
    {
        return Err(format!("seed={seed} conflicts with settings['seed']={s}"));
    }
    obj.insert("seed".into(), json!(seed));
    Ok(Value::Object(obj))
}

fn opt_value(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    obj.map_or(Ok(Value::Null), from_py)
}

fn invalid(msg: impl Into<String>) -> PyErr {
    new_err(ErrorKind::InvalidArgument, msg)
}

/// Build the engine configuration from the merged `config` dict and
/// keyword arguments.
///
/// * Keys must be `EngineConfig` fields (the C ABI's configuration keys);
///   a misspelt key is an error rather than silently ignored.
/// * `default_backend` takes the same forms as the `backend` argument of
///   `run`/`submit` (`"cpu"`, `"metal"`, `{"kind": "gpu", ...}`; `"auto"`
///   or `None` keep the engine default).
/// * `max_retained_jobs` defaults to unlimited. The engine's bounded
///   retention exists for front ends that may never collect a result; here
///   every job is owned by a `Job` object (whose result is collected, or
///   which releases the job when garbage-collected) or by a blocking
///   `run`, so a bound would only discard results a script still holds a
///   `Job` for (a sweep submitting more jobs than the bound before
///   collecting any). An explicit value is honoured; jobs discarded by it
///   report the state `evicted`.
pub fn engine_config(mut cfg: serde_json::Map<String, Value>) -> Result<EngineConfig, String> {
    let known = match serde_json::to_value(EngineConfig::default()) {
        Ok(Value::Object(o)) => o,
        _ => return Err("engine configuration is not a JSON object".into()),
    };
    let mut unknown: Vec<&str> = cfg
        .keys()
        .filter(|k| !known.contains_key(*k))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        let mut names: Vec<&str> = known.keys().map(String::as_str).collect();
        names.sort_unstable();
        return Err(format!(
            "unknown engine configuration key(s) {} (known: {})",
            unknown.join(", "),
            names.join(", ")
        ));
    }
    if let Some(b) = cfg.remove("default_backend")
        && let Some(backend) = parse_backend(&b)?
    {
        let v = serde_json::to_value(backend).map_err(|e| e.to_string())?;
        cfg.insert("default_backend".into(), v);
    }
    if !cfg.contains_key("max_retained_jobs") {
        cfg.insert("max_retained_jobs".into(), json!(usize::MAX));
    }
    serde_json::from_value(Value::Object(cfg)).map_err(|e| format!("invalid engine config: {e}"))
}

/// Parse an integer argument that must fit `0..2^64` (seeds, indices),
/// so negative or huge values raise `kind` instead of `OverflowError`.
pub fn u64_arg(obj: &Bound<'_, PyAny>, what: &str, kind: ErrorKind) -> PyResult<u64> {
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(new_err(
            kind,
            format!("{what} must be an integer, got a bool"),
        ));
    }
    obj.extract::<u64>().map_err(|_| {
        let repr = obj
            .repr()
            .map_or_else(|_| "?".to_string(), |r| r.to_string());
        new_err(
            kind,
            format!("{what} must be an integer in 0..2**64, got {repr}"),
        )
    })
}

/// The MKLight engine: worker pool, analysis registry and backends.
/// Cheap to share; sessions keep it alive.
///
/// `Engine(config=None, **kwargs)`: configuration keys are those of the C
/// ABI (`worker_threads`, `max_retained_jobs`, `resolved_cache_capacity`,
/// `progress_interval_ms`, `app_name`, `app_version`, `default_backend`);
/// see [`engine_config`] for validation and the retention default.
#[pyclass(frozen, module = "mklight", name = "Engine")]
pub struct PyEngine {
    pub(crate) engine: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (config=None, **kwargs))]
    fn py_new(
        config: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut cfg = match opt_value(config.as_ref())? {
            Value::Null => serde_json::Map::new(),
            Value::Object(o) => o,
            other => return Err(invalid(format!("config must be a dict, got {other}"))),
        };
        if let Some(kw) = kwargs
            && let Value::Object(o) = from_py(kw.as_any())?
        {
            cfg.extend(o);
        }
        let config = engine_config(cfg).map_err(invalid)?;
        Ok(PyEngine {
            engine: Engine::new(config),
        })
    }

    /// A new session. `sample` names a built-in sample system (see
    /// `mklight.samples()`); `system` is an `OpticalSystem` dict. With
    /// neither, the minimal empty system.
    #[pyo3(name = "new", signature = (sample=None, system=None))]
    fn new_session(
        &self,
        py: Python<'_>,
        sample: Option<&str>,
        system: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PySession> {
        let sys = match (sample, system) {
            (Some(_), Some(_)) => return Err(invalid("give either sample or system, not both")),
            (Some(name), None) => {
                Some(sample_system(name).map_err(|m| new_err(ErrorKind::NotFound, m))?)
            }
            (None, Some(d)) => Some(system_from_py(&d)?),
            (None, None) => None,
        };
        let engine = self.engine.clone();
        let session = py.detach(move || match sys {
            Some(s) => engine.session_from_project(Project::new(s)),
            None => Ok(engine.new_session()),
        });
        Ok(PySession::wrap(session.map_err(engine_err)?))
    }

    /// Open a `.mklight` project file (older schema versions are migrated;
    /// see `Session.info["migration"]`).
    fn open(&self, py: Python<'_>, path: PathBuf) -> PyResult<PySession> {
        let engine = self.engine.clone();
        let s = py
            .detach(move || engine.open_session(&path))
            .map_err(engine_err)?;
        Ok(PySession::wrap(s))
    }

    /// Registered analyses (name, title, category, availability, settings
    /// schema, defaults).
    fn analyses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ser_to_py(py, &self.engine.analyses())
    }

    /// Compute backends and their availability.
    fn backends<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ser_to_py(py, &self.engine.backends())
    }

    /// Status dicts of the jobs the engine still knows.
    fn jobs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ser_to_py(py, &self.engine.job_statuses())
    }

    /// Cancel every queued and running job.
    fn cancel_all(&self) {
        self.engine.cancel_all_jobs()
    }

    /// The engine configuration (dict).
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ser_to_py(py, self.engine.config())
    }

    #[getter]
    fn default_backend<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ser_to_py(py, &self.engine.default_backend())
    }

    fn __repr__(&self) -> String {
        format!(
            "<mklight.Engine {} backend={}>",
            mklight_core::ENGINE_VERSION,
            self.engine.default_backend().id()
        )
    }
}

/// Parse an `OpticalSystem` dict.
pub fn system_from_py(d: &Bound<'_, PyAny>) -> PyResult<OpticalSystem> {
    let v = from_py(d)?;
    serde_json::from_value(v).map_err(|e| new_err(ErrorKind::Parse, format!("invalid system: {e}")))
}

/// The low-level tracer of one (snapshot, library) state.
struct TracerCache {
    key: (String, String),
    tracer: Arc<SequentialTracer>,
}

/// One open document: the optical system (as immutable, content-hashed
/// snapshots), its glass catalogs, and analyses run against it.
#[pyclass(frozen, module = "mklight", name = "Session")]
pub struct PySession {
    engine: Engine,
    inner: Mutex<Session>,
    tracer: Mutex<Option<TracerCache>>,
}

impl PySession {
    pub fn wrap(s: Session) -> Self {
        PySession {
            engine: s.engine().clone(),
            inner: Mutex::new(s),
            tracer: Mutex::new(None),
        }
    }

    /// Lock without the GIL (inside `py.detach`, or from Rust).
    pub fn lock(&self) -> MutexGuard<'_, Session> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lock while attached: the GIL is released while waiting, so a thread
    /// holding the session lock can never be blocked by this one.
    pub fn lock_py(&self, py: Python<'_>) -> MutexGuard<'_, Session> {
        self.inner
            .lock_py_attached(py)
            .unwrap_or_else(|e| e.into_inner())
    }

    /// The reference tracer of the current state (built once per snapshot
    /// and library, GIL released).
    pub fn tracer(&self, py: Python<'_>) -> PyResult<Arc<SequentialTracer>> {
        py.detach(|| -> Result<_, EngineError> {
            let s = self.lock();
            let key = (s.snapshot_hash().to_hex(), s.library_hash().to_string());
            let mut cache = self.tracer.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(c) = cache.as_ref()
                && c.key == key
            {
                return Ok(c.tracer.clone());
            }
            let resolved = s.resolved()?;
            drop(s);
            let tracer = Arc::new(SequentialTracer::new(resolved)?);
            *cache = Some(TracerCache {
                key,
                tracer: tracer.clone(),
            });
            Ok(tracer)
        })
        .map_err(engine_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn request(
        &self,
        analysis: &str,
        settings: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
        backend: Option<&Bound<'_, PyAny>>,
        allow_backend_fallback: bool,
        view_key: Option<String>,
    ) -> PyResult<AnalysisRequest> {
        let seed = seed
            .filter(|s| !s.is_none())
            .map(|s| u64_arg(s, "seed", ErrorKind::InvalidSettings))
            .transpose()?;
        let settings = opt_value(settings)?;
        let backend = parse_backend(&opt_value(backend)?).map_err(invalid)?;
        let schema = self
            .engine
            .analysis(analysis)
            .map_err(engine_err)?
            .settings_schema();
        let settings = apply_seed(&schema, settings, seed)
            .map_err(|m| new_err(ErrorKind::InvalidSettings, m))?;
        Ok(AnalysisRequest {
            analysis: analysis.to_string(),
            settings,
            view_key,
            backend,
            allow_backend_fallback,
            tier: Default::default(),
            aim_stop: None,
        })
    }

    fn submit_request(&self, py: Python<'_>, req: &AnalysisRequest) -> PyResult<PyJob> {
        let id = py
            .detach(|| {
                let spec = self.lock().analysis_job(req)?;
                self.engine.submit_job(spec)
            })
            .map_err(engine_err)?;
        Ok(PyJob::new(self.engine.clone(), id, req.analysis.clone()))
    }
}

#[pymethods]
impl PySession {
    /// The optical system (the `OpticalSystem` JSON model as a dict; a
    /// fresh copy on every access). Assigning validates it (see
    /// `set_system`).
    #[getter]
    fn system<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v =
            serde_json::to_value(self.lock_py(py).system()).map_err(|e| engine_err(e.into()))?;
        to_py(py, &v)
    }

    #[setter(system)]
    fn set_system_attr(&self, py: Python<'_>, system: Bound<'_, PyAny>) -> PyResult<()> {
        self.set_system(py, system).map(|_| ())
    }

    /// Replace the system. An invalid system raises `InvalidModelError`
    /// and leaves the session unchanged. Returns the new snapshot hash.
    fn set_system(&self, py: Python<'_>, system: Bound<'_, PyAny>) -> PyResult<String> {
        let sys = system_from_py(&system)?;
        py.detach(|| self.lock().set_system(sys))
            .map(|h| h.to_hex())
            .map_err(engine_err)
    }

    /// Set one parameter: `path` is a `ParamPath` dict such as
    /// `{"param": "thickness", "surface": 3}` (`surface` is the stable
    /// surface id), `value` a number or `"inf"`. Returns the new snapshot
    /// hash.
    fn set_param(
        &self,
        py: Python<'_>,
        path: Bound<'_, PyAny>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<String> {
        let edit = json!({"path": from_py(&path)?, "value": from_py(&value)?});
        self.apply_edits_value(py, Value::Array(vec![edit]))
    }

    /// Apply several `{"path": ..., "value": ...}` edits atomically (all or
    /// none). Returns the new snapshot hash.
    fn set_params(&self, py: Python<'_>, edits: Bound<'_, PyAny>) -> PyResult<String> {
        self.apply_edits_value(py, from_py(&edits)?)
    }

    /// Content hash (64 hex digits) of the current snapshot.
    #[getter]
    fn snapshot_hash(&self, py: Python<'_>) -> String {
        self.lock_py(py).snapshot_hash().to_hex()
    }

    /// Content hash of the session's glass library.
    #[getter]
    fn library_hash(&self, py: Python<'_>) -> String {
        self.lock_py(py).library_hash().to_string()
    }

    /// Session id (unique per engine).
    #[getter]
    fn id(&self, py: Python<'_>) -> u64 {
        self.lock_py(py).id()
    }

    /// Project file path, or None for an unsaved session.
    #[getter]
    fn path(&self, py: Python<'_>) -> Option<PathBuf> {
        self.lock_py(py).path().map(PathBuf::from)
    }

    /// True when there are unsaved changes.
    #[getter]
    fn dirty(&self, py: Python<'_>) -> bool {
        self.lock_py(py).is_dirty()
    }

    /// Session state: path, dirty flag, hashes, name, open warnings,
    /// migration report, catalogs.
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.lock_py(py).info();
        to_py(py, &v)
    }

    /// Save atomically to `path`, or to the session's path when omitted.
    #[pyo3(signature = (path=None))]
    fn save(&self, py: Python<'_>, path: Option<PathBuf>) -> PyResult<()> {
        py.detach(|| self.lock().save(path.as_deref()))
            .map_err(engine_err)
    }

    /// Load a glass catalog (AGF text, or a `GlassCatalog` JSON file when
    /// the extension is `.json`) into the session's library and record it
    /// in the project. Returns `{name, source, path, glass_count}`.
    fn load_catalog<'py>(&self, py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyAny>> {
        let summary = py
            .detach(|| self.lock().load_catalog_file(&path))
            .map_err(engine_err)?;
        ser_to_py(py, &summary)
    }

    /// Replace the system with one imported from a `.zmx` or `.seq` lens
    /// prescription file. Glasses resolve against this session's catalogs
    /// (load them first). Content that does not map is not an error: it is
    /// listed in the result's `warnings`. Returns `{format, encoding,
    /// system, warnings}`.
    fn import_prescription<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
    ) -> PyResult<Bound<'py, PyAny>> {
        let imported = py
            .detach(|| self.lock().import_prescription(&path))
            .map_err(engine_err)?;
        ser_to_py(py, &imported)
    }

    /// Catalogs in the library (built-in first).
    fn catalogs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.lock_py(py).catalogs();
        ser_to_py(py, &c)
    }

    /// Glasses of one catalog (case-insensitive), or of all catalogs.
    #[pyo3(signature = (catalog=None))]
    fn glasses<'py>(&self, py: Python<'py>, catalog: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.lock_py(py).glasses(catalog).map_err(engine_err)?;
        ser_to_py(py, &g)
    }

    /// Replace a non-system project section (`views`, `merit`,
    /// `variables`, `tolerances`, `metadata`).
    fn set_project_section(
        &self,
        py: Python<'_>,
        section: &str,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let v = from_py(&value)?;
        self.lock_py(py)
            .set_project_section(section, v)
            .map_err(engine_err)
    }

    /// Run an analysis on the current snapshot and wait for it (GIL
    /// released). See `submit` for the arguments; `progress(fraction,
    /// stage)` is called with monotone fractions from the calling thread.
    /// Ctrl-C (or an exception raised by `progress`) cancels the job.
    #[pyo3(signature = (analysis, settings=None, seed=None, backend=None, progress=None, allow_backend_fallback=false, timeout=None))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        analysis: &str,
        settings: Option<Bound<'_, PyAny>>,
        seed: Option<Bound<'_, PyAny>>,
        backend: Option<Bound<'_, PyAny>>,
        progress: Option<Bound<'_, PyAny>>,
        allow_backend_fallback: bool,
        timeout: Option<f64>,
    ) -> PyResult<PyAnalysisResult> {
        let req = self.request(
            analysis,
            settings.as_ref(),
            seed.as_ref(),
            backend.as_ref(),
            allow_backend_fallback,
            None,
        )?;
        // On any error (including a timeout) the job is dropped here,
        // which cancels and forgets it.
        let job = self.submit_request(py, &req)?;
        job.wait_result(py, timeout, progress.as_ref())
    }

    /// Queue an analysis and return a `Job` at once.
    ///
    /// * `settings`: partial settings dict (missing keys take defaults).
    /// * `seed`: sets the analysis' `seed` setting (stochastic analyses).
    /// * `backend`: `"auto"` (automatic selection), `"cpu"`, `"metal"`, or
    ///   `{"kind": "gpu", "name": "metal"}`.
    /// * `view_key`: a newer job with the same key supersedes this one.
    #[pyo3(signature = (analysis, settings=None, seed=None, backend=None, allow_backend_fallback=false, view_key=None))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        py: Python<'_>,
        analysis: &str,
        settings: Option<Bound<'_, PyAny>>,
        seed: Option<Bound<'_, PyAny>>,
        backend: Option<Bound<'_, PyAny>>,
        allow_backend_fallback: bool,
        view_key: Option<String>,
    ) -> PyResult<PyJob> {
        let req = self.request(
            analysis,
            settings.as_ref(),
            seed.as_ref(),
            backend.as_ref(),
            allow_backend_fallback,
            view_key,
        )?;
        self.submit_request(py, &req)
    }

    /// The `Engine` this session belongs to.
    #[getter]
    fn engine(&self) -> PyEngine {
        PyEngine {
            engine: self.engine.clone(),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let s = self.lock_py(py);
        format!(
            "<mklight.Session {} '{}' {}>",
            s.id(),
            s.system().name,
            s.snapshot_hash().short()
        )
    }
}

impl PySession {
    fn apply_edits_value(&self, py: Python<'_>, v: Value) -> PyResult<String> {
        let edits: Vec<ParamEdit> = match v {
            Value::Array(_) => serde_json::from_value(v),
            other => serde_json::from_value(Value::Array(vec![other])),
        }
        .map_err(|e| invalid(format!("invalid parameter edit: {e}")))?;
        py.detach(|| self.lock().apply_edits(&edits))
            .map(|h| h.to_hex())
            .map_err(engine_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_argument_forms() {
        assert_eq!(parse_backend(&Value::Null), Ok(None));
        assert_eq!(parse_backend(&json!("auto")), Ok(Some(Backend::Auto)));
        assert_eq!(
            parse_backend(&json!("cpu")),
            Ok(Some(Backend::CpuReference))
        );
        assert_eq!(
            parse_backend(&json!("gpu:metal")),
            Ok(Some(Backend::Gpu("metal".into())))
        );
        assert_eq!(
            parse_backend(&json!({"kind": "gpu", "name": "metal"})),
            Ok(Some(Backend::Gpu("metal".into())))
        );
        assert!(parse_backend(&json!(3)).is_err());
    }

    #[test]
    fn seed_is_merged_only_into_seeded_analyses() {
        let seeded = json!({"type": "object", "properties": {"seed": {"type": "integer"}}});
        let plain = json!({"type": "object", "properties": {}});
        assert_eq!(apply_seed(&plain, Value::Null, None), Ok(Value::Null));
        assert!(apply_seed(&plain, Value::Null, Some(1)).is_err());
        assert_eq!(
            apply_seed(&seeded, json!({"count": 3}), Some(9)),
            Ok(json!({"count": 3, "seed": 9}))
        );
        assert_eq!(
            apply_seed(&seeded, json!({"seed": 9}), Some(9)),
            Ok(json!({"seed": 9}))
        );
        assert!(apply_seed(&seeded, json!({"seed": 8}), Some(9)).is_err());
    }

    #[test]
    fn engine_config_validates_keys_backend_and_retention() {
        let cfg = |v: Value| engine_config(v.as_object().unwrap().clone());
        let c = cfg(json!({})).unwrap();
        assert_eq!(c.max_retained_jobs, usize::MAX);
        assert_eq!(c.default_backend, Backend::Auto);
        let c = cfg(json!({"max_retained_jobs": 4, "default_backend": "gpu:metal"})).unwrap();
        assert_eq!(c.max_retained_jobs, 4);
        assert_eq!(c.default_backend, Backend::Gpu("metal".into()));
        assert_eq!(
            cfg(json!({"default_backend": "auto"}))
                .unwrap()
                .default_backend,
            Backend::Auto
        );
        let e = cfg(json!({"worker_thread": 2})).unwrap_err();
        assert!(
            e.contains("worker_thread") && e.contains("worker_threads"),
            "{e}"
        );
        assert!(cfg(json!({"default_backend": 3})).is_err());
        assert!(cfg(json!({"worker_threads": "many"})).is_err());
    }

    #[test]
    fn samples_are_found_by_name() {
        let s = sample_system("achromat").unwrap();
        assert_eq!(s.surfaces.len(), 5);
        let e = sample_system("nope").unwrap_err();
        assert!(e.contains("achromat"), "{e}");
    }

    #[test]
    fn requests_through_the_engine_match_direct_runs() {
        // The same request through a session job and a synchronous run gives
        // the same result (one code path).
        let engine = Engine::default();
        let session = engine
            .session_from_project(Project::new(sample_system("cooke_triplet").unwrap()))
            .unwrap();
        let req = AnalysisRequest::new("rng_stream", json!({"seed": 3, "count": 100}));
        let job = session.submit_analysis(&req).unwrap();
        engine.wait_job(job, None).unwrap();
        let a = engine.take_result(job).unwrap();
        let b = session
            .run_analysis(&req, &mklight_core::JobContext::new())
            .unwrap();
        assert_eq!(a.json, b.json);
        assert_eq!(a.buffers, b.buffers);
        assert_eq!(a.run_info.seed, Some(3));
    }
}
