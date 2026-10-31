//! `Session`: one open document. It holds the project, the material library
//! (built-in plus the user catalogs the project references) and the current
//! immutable `Snapshot`.
//!
//! Every edit builds a new snapshot (validated, content-hashed) and swaps it
//! in only if it is valid, so the session never holds an invalid system and
//! running jobs keep the snapshot they started with. The library has a
//! content hash too (`library_hash`); a result is current while both its
//! snapshot hash and its library hash equal the session's. `ResolvedSystem`s are
//! cached in a small LRU keyed by `(snapshot hash, library generation)`:
//! undo/redo between recent states, and repeated analyses of one state, do
//! not re-resolve materials.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mklight_core::materials::builtin::BUILTIN_CATALOG_NAME;
use mklight_core::materials::{GlassCatalog, MaterialLibrary, agf};
use mklight_core::param::ParamPath;
use mklight_core::snapshot::SnapshotHash;
use mklight_core::{JobContext, MkError, OpticalSystem, ResolvedSystem, Snapshot};

use crate::analysis::{AnalysisInput, AnalysisResult, library_hash, run_analysis};
use crate::backend::Backend;
use crate::engine::Engine;
use crate::error::{EngineError, EngineResult};
use crate::lru::LruCache;
use crate::project::{
    AnalysisView, CatalogFormat, CatalogRef, MigrationReport, Project, ProjectMetadata,
};
use crate::scheduler::{JobId, JobSpec};
use crate::time::now_rfc3339;

/// One parameter edit (`ParamPath` + new value). `value` accepts `"inf"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamEdit {
    pub path: ParamPath,
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub value: f64,
}

/// A request to run an analysis on the session's current snapshot.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisRequest {
    pub analysis: String,
    /// Partial settings; missing keys take the schema defaults.
    pub settings: Value,
    /// Analysis window key: a newer job with the same key (in the same
    /// session) supersedes and cancels the older one.
    pub view_key: Option<String>,
    /// `None` uses the engine default.
    pub backend: Option<Backend>,
    /// Run on the CPU reference (with a `RunInfo` warning) if the requested
    /// backend is unavailable, instead of failing.
    pub allow_backend_fallback: bool,
}

impl AnalysisRequest {
    pub fn new(analysis: impl Into<String>, settings: Value) -> Self {
        AnalysisRequest {
            analysis: analysis.into(),
            settings,
            ..Default::default()
        }
    }
    pub fn view_key(mut self, k: impl Into<String>) -> Self {
        self.view_key = Some(k.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CatalogSummary {
    pub name: String,
    /// `builtin` or `user`.
    pub source: &'static str,
    pub path: Option<String>,
    pub glass_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GlassSummary {
    pub catalog: String,
    pub name: String,
    pub nd: f64,
    pub vd: f64,
    pub min_um: f64,
    pub max_um: f64,
}

type ResolvedCache = Arc<Mutex<LruCache<(SnapshotHash, u64), Arc<ResolvedSystem>>>>;

pub struct Session {
    id: u64,
    engine: Engine,
    project: Project,
    current: Snapshot,
    library: Arc<MaterialLibrary>,
    /// Bumped whenever the library changes, so cached resolutions made with
    /// an older library are never reused.
    library_generation: u64,
    /// `analysis::library_hash` of `library`.
    library_hash: String,
    resolved: ResolvedCache,
    path: Option<PathBuf>,
    dirty: bool,
    open_warnings: Vec<String>,
    migration: MigrationReport,
}

/// Read a catalog file. AGF files are named by their file stem.
pub fn read_catalog_file(path: &Path, format: CatalogFormat) -> EngineResult<GlassCatalog> {
    let bytes = std::fs::read(path).map_err(|e| {
        EngineError::Core(MkError::Io(format!("cannot read {}: {e}", path.display())))
    })?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut cat = match format {
        // Vendor AGF files are often UTF-16; `parse_agf_bytes` detects the encoding.
        CatalogFormat::Agf => {
            let agf = agf::parse_agf_bytes(&stem, &bytes, &agf::AgfOptions::default())?;
            if agf.catalog.glasses.is_empty() {
                let detail: Vec<String> = agf.warnings.iter().map(|w| w.to_string()).collect();
                return Err(EngineError::Core(MkError::Parse(format!(
                    "{}: no usable glasses ({})",
                    path.display(),
                    detail.join("; ")
                ))));
            }
            agf.catalog
        }
        CatalogFormat::Json => serde_json::from_slice::<GlassCatalog>(&bytes)?,
    };
    if cat.name.is_empty() {
        cat.name = stem;
    }
    if cat.name.is_empty() {
        return Err(EngineError::Core(MkError::Parse(
            "catalog has no name".into(),
        )));
    }
    if cat.name.eq_ignore_ascii_case(BUILTIN_CATALOG_NAME) {
        return Err(EngineError::Core(MkError::Parse(format!(
            "a user catalog cannot be named '{BUILTIN_CATALOG_NAME}'"
        ))));
    }
    Ok(cat)
}

fn resolve_cached(
    cache: &ResolvedCache,
    snapshot: &Snapshot,
    library: &Arc<MaterialLibrary>,
    generation: u64,
) -> mklight_core::MkResult<Arc<ResolvedSystem>> {
    let key = (snapshot.hash(), generation);
    if let Some(r) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(r);
    }
    // Resolve outside the lock; two racing jobs may both resolve, which is
    // harmless (identical results).
    let r = Arc::new(ResolvedSystem::new(snapshot, library.clone())?);
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, r.clone());
    Ok(r)
}

impl Session {
    pub(crate) fn from_project(
        engine: &Engine,
        mut project: Project,
        path: Option<PathBuf>,
    ) -> EngineResult<Session> {
        let current = Snapshot::new(project.system.clone())?;
        project.system = current.system().clone();
        let mut s = Session {
            id: engine.next_session_id(),
            engine: engine.clone(),
            project,
            current,
            library: Arc::new(MaterialLibrary::with_builtin()),
            library_generation: 0,
            library_hash: String::new(),
            resolved: Arc::new(Mutex::new(LruCache::new(
                engine.config().resolved_cache_capacity,
            ))),
            path,
            dirty: false,
            open_warnings: Vec::new(),
            migration: MigrationReport::default(),
        };
        s.migration.from_version = s.project.schema_version;
        s.migration.to_version = s.project.schema_version;
        s.reload_catalogs();
        Ok(s)
    }

    pub(crate) fn open(engine: &Engine, path: &Path) -> EngineResult<Session> {
        let (project, report) = Project::load(path)?;
        let mut s = Self::from_project(engine, project, Some(path.to_path_buf()))?;
        s.open_warnings.extend(report.notes.iter().cloned());
        s.migration = report;
        Ok(s)
    }

    /// Load every catalog the project references. Missing or broken files
    /// become `open_warnings`; the project keeps the references. A catalog
    /// found through its relative path (the project folder was moved) gets
    /// its absolute `path` updated to where it was found, so the next save
    /// records both correctly.
    fn reload_catalogs(&mut self) {
        let dir = self
            .path
            .as_deref()
            .and_then(Path::parent)
            .map(|d| {
                if d.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    d
                }
            })
            .map(Path::to_path_buf);
        let mut lib = MaterialLibrary::with_builtin();
        for r in &mut self.project.catalogs {
            let mut last_err = None;
            let mut loaded = false;
            for cand in r.candidates(dir.as_deref()) {
                match read_catalog_file(&cand, r.format) {
                    Ok(mut c) => {
                        c.name = r.name.clone();
                        lib.add_catalog(Arc::new(c));
                        let abs = std::fs::canonicalize(&cand).unwrap_or(cand);
                        r.path = Some(abs.display().to_string());
                        loaded = true;
                        break;
                    }
                    Err(e) => last_err = Some(format!("{}: {e}", cand.display())),
                }
            }
            if !loaded {
                self.open_warnings.push(format!(
                    "glass catalog '{}' could not be loaded ({})",
                    r.name,
                    last_err.unwrap_or_else(|| "no path recorded".into())
                ));
            }
        }
        self.set_library(lib);
    }

    fn set_library(&mut self, lib: MaterialLibrary) {
        self.library_hash = library_hash(&lib);
        self.library = Arc::new(lib);
        self.library_generation += 1;
        self.resolved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    // --- accessors

    /// Unique per engine; namespaces view keys.
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn snapshot(&self) -> &Snapshot {
        &self.current
    }
    pub fn snapshot_hash(&self) -> SnapshotHash {
        self.current.hash()
    }
    pub fn system(&self) -> &OpticalSystem {
        self.current.system()
    }
    /// The project, with `system` equal to the current snapshot.
    pub fn project(&self) -> &Project {
        &self.project
    }
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
    /// Unsaved changes?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
    /// Problems found while opening (missing catalogs, migration notes).
    pub fn open_warnings(&self) -> &[String] {
        &self.open_warnings
    }
    pub fn migration(&self) -> &MigrationReport {
        &self.migration
    }
    pub fn library(&self) -> &Arc<MaterialLibrary> {
        &self.library
    }
    /// Content hash of the material library (64 hex digits); changes when
    /// a catalog is added, replaced or reloaded with different data.
    pub fn library_hash(&self) -> &str {
        &self.library_hash
    }

    /// Session state for front ends.
    pub fn info(&self) -> Value {
        serde_json::json!({
            "session_id": self.id,
            "path": self.path.as_ref().map(|p| p.display().to_string()),
            "dirty": self.dirty,
            "snapshot_hash": self.current.hash(),
            "library_hash": self.library_hash,
            "name": self.current.system().name,
            "open_warnings": self.open_warnings,
            "migration": self.migration,
            "catalogs": self.catalogs(),
        })
    }

    // --- edits

    fn commit(&mut self, snap: Snapshot) -> SnapshotHash {
        if snap.hash() != self.current.hash() {
            self.project.system = snap.system().clone();
            self.current = snap;
            self.dirty = true;
        }
        self.current.hash()
    }

    /// Replace the whole system. Invalid systems are rejected and leave the
    /// session unchanged.
    pub fn set_system(&mut self, system: OpticalSystem) -> EngineResult<SnapshotHash> {
        let snap = Snapshot::new(system)?;
        Ok(self.commit(snap))
    }

    /// Apply parameter edits atomically: all succeed and validate, or none
    /// is applied.
    pub fn apply_edits(&mut self, edits: &[ParamEdit]) -> EngineResult<SnapshotHash> {
        let mut sys = self.current.system().clone();
        for e in edits {
            e.path.set(&mut sys, e.value)?;
        }
        self.set_system(sys)
    }

    /// Replace a non-system project section: `views`, `merit`, `variables`,
    /// `tolerances` or `metadata`.
    pub fn set_project_section(&mut self, section: &str, value: Value) -> EngineResult<()> {
        match section {
            "views" => self.project.views = serde_json::from_value::<Vec<AnalysisView>>(value)?,
            "merit" => self.project.merit = value,
            "variables" => self.project.variables = value,
            "tolerances" => self.project.tolerances = value,
            "metadata" => self.project.metadata = serde_json::from_value::<ProjectMetadata>(value)?,
            other => {
                return Err(EngineError::Core(MkError::InvalidParam(format!(
                    "unknown or read-only project section '{other}' \
                     (use set_system / load_catalog for those)"
                ))));
            }
        }
        self.dirty = true;
        Ok(())
    }

    // --- files

    /// Save atomically to `path`, or to the current path when `None`.
    /// Stamps `modified`, the app and engine versions, and records catalog
    /// paths relative to the project directory where possible: from the
    /// catalog's (resolved) absolute path when it lies under the directory;
    /// otherwise the previous relative path is kept only while the absolute
    /// path is unusable (the catalog file was not found), since it is then
    /// the only location information left.
    pub fn save(&mut self, path: Option<&Path>) -> EngineResult<()> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => self.path.clone().ok_or_else(|| {
                EngineError::Core(MkError::InvalidParam(
                    "the session has no file path yet: save with an explicit path".into(),
                ))
            })?,
        };
        let mut p = self.project.clone();
        let cfg = self.engine.config();
        p.metadata.modified = now_rfc3339();
        if p.metadata.created.is_empty() {
            p.metadata.created = p.metadata.modified.clone();
        }
        if !cfg.app_name.is_empty() {
            p.metadata.app_name = cfg.app_name.clone();
        }
        if !cfg.app_version.is_empty() {
            p.metadata.app_version = cfg.app_version.clone();
        }
        p.metadata.engine_version = mklight_core::ENGINE_VERSION.to_string();
        // Catalog paths are canonical, so compare against the canonical
        // project directory (symlinked temp dirs, `..` components).
        if let Some(dir) = path
            .parent()
            .map(|d| {
                if d.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    d
                }
            })
            .and_then(|d| std::fs::canonicalize(d).ok())
        {
            for c in &mut p.catalogs {
                let abs = c.path.as_deref().map(Path::new);
                let rel = abs
                    .and_then(|a| a.strip_prefix(&dir).ok())
                    .map(|r| r.to_string_lossy().into_owned());
                if rel.is_some() || abs.is_some_and(Path::is_file) {
                    c.relative_path = rel;
                }
            }
        }
        p.save(&path)?;
        self.project = p;
        self.path = Some(path);
        self.dirty = false;
        Ok(())
    }

    // --- materials

    /// Load a user glass catalog (AGF, or JSON by `.json` extension), add it
    /// to the library and record it in the project. Replaces a catalog of
    /// the same name.
    pub fn load_catalog_file(&mut self, path: &Path) -> EngineResult<CatalogSummary> {
        let format = CatalogFormat::from_path(path);
        let cat = read_catalog_file(path, format)?;
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let summary = CatalogSummary {
            name: cat.name.clone(),
            source: "user",
            path: Some(abs.display().to_string()),
            glass_count: cat.glasses.len(),
        };
        let r = CatalogRef {
            name: cat.name.clone(),
            path: summary.path.clone(),
            relative_path: None,
            format,
        };
        self.add_catalog(cat, Some(r));
        Ok(summary)
    }

    /// Add an in-memory catalog. With a `CatalogRef` it is recorded in the
    /// project so it reloads on open.
    pub fn add_catalog(&mut self, cat: GlassCatalog, reference: Option<CatalogRef>) {
        let mut lib = (*self.library).clone();
        lib.add_catalog(Arc::new(cat));
        self.set_library(lib);
        if let Some(r) = reference {
            self.project
                .catalogs
                .retain(|c| !c.name.eq_ignore_ascii_case(&r.name));
            self.project.catalogs.push(r);
            self.dirty = true;
        }
    }

    pub fn catalogs(&self) -> Vec<CatalogSummary> {
        self.library
            .catalogs()
            .iter()
            .map(|c| {
                let builtin = c.name == BUILTIN_CATALOG_NAME;
                CatalogSummary {
                    name: c.name.clone(),
                    source: if builtin { "builtin" } else { "user" },
                    path: self
                        .project
                        .catalogs
                        .iter()
                        .find(|r| r.name.eq_ignore_ascii_case(&c.name))
                        .and_then(|r| r.path.clone()),
                    glass_count: c.glasses.len(),
                }
            })
            .collect()
    }

    /// Glasses of one catalog (case-insensitive name), or of all catalogs.
    pub fn glasses(&self, catalog: Option<&str>) -> EngineResult<Vec<GlassSummary>> {
        let cats: Vec<_> = self
            .library
            .catalogs()
            .iter()
            .filter(|c| catalog.is_none_or(|n| c.name.eq_ignore_ascii_case(n)))
            .collect();
        if let Some(n) = catalog
            && cats.is_empty()
        {
            return Err(EngineError::Core(MkError::UnknownMaterial(format!(
                "no catalog named '{n}'"
            ))));
        }
        Ok(cats
            .iter()
            .flat_map(|c| {
                c.glasses.values().map(|g| GlassSummary {
                    catalog: c.name.clone(),
                    name: g.name.clone(),
                    nd: g.nd,
                    vd: g.vd,
                    min_um: g.min_um,
                    max_um: g.max_um,
                })
            })
            .collect())
    }

    /// The current snapshot resolved against the library (cached).
    pub fn resolved(&self) -> EngineResult<Arc<ResolvedSystem>> {
        Ok(resolve_cached(
            &self.resolved,
            &self.current,
            &self.library,
            self.library_generation,
        )?)
    }

    /// Number of cached resolved systems (for tests and diagnostics).
    pub fn resolved_cache_len(&self) -> usize {
        self.resolved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    // --- solves, validation, autosave

    /// Apply the system's solves (docs/SOLVES.md) and commit the solved
    /// system as a new snapshot. `settings` follows the `solves` analysis'
    /// schema (`configuration` must be null: a solved configuration has no
    /// configuration list, so it cannot be stored back). Returns the new
    /// snapshot hash (unchanged when every solve already held) and the
    /// solve report (the `solves` result shape). On failure the session is
    /// unchanged.
    pub fn apply_solves(
        &mut self,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<(SnapshotHash, Value)> {
        let p = self.prepare_solves(settings)?;
        let (sys, report) = p.run(ctx)?;
        let hash = self.commit_solves(&p, sys)?;
        Ok((hash, report))
    }

    /// `apply_solves` in three steps, so a caller that shares the session
    /// behind a lock can solve without holding it: `prepare_solves` (under
    /// the lock: checks the settings and captures the current system and
    /// library), `PendingSolves::run` (no lock), `commit_solves` (under the
    /// lock again).
    pub fn prepare_solves(&self, settings: &Value) -> EngineResult<PendingSolves> {
        let schema = crate::analysis::solves::solve_settings_schema();
        let s = crate::settings::normalize(&schema, settings)?;
        let o: crate::analysis::solves::SolveOptions =
            serde_json::from_value(s).map_err(|e| EngineError::InvalidSettings(e.to_string()))?;
        if o.configuration.is_some() {
            return Err(EngineError::InvalidSettings(
                "apply_solves solves the base system; run the `solves` analysis to preview a \
                 configuration"
                    .into(),
            ));
        }
        Ok(PendingSolves {
            base: self.current.clone(),
            library: self.library.clone(),
            library_generation: self.library_generation,
            solver: o.solver,
        })
    }

    /// Commit a solved system computed from `pending`. If the session
    /// changed in between (an edit, undo, a new library), nothing is
    /// committed and the call fails with `MkError::Cancelled` (superseded,
    /// like a job with the same view key): solve again.
    pub fn commit_solves(
        &mut self,
        pending: &PendingSolves,
        solved: OpticalSystem,
    ) -> EngineResult<SnapshotHash> {
        if self.current.hash() != pending.base.hash()
            || self.library_generation != pending.library_generation
        {
            return Err(MkError::Cancelled.into());
        }
        self.set_system(solved)
    }

    /// The engine's `validate` report of the current snapshot (computed in
    /// the calling thread): `{ok, errors, warnings, issues}`.
    pub fn validate(&self) -> EngineResult<Value> {
        let r = self.run_analysis(
            &AnalysisRequest::new("validate", Value::Null),
            &JobContext::new(),
        )?;
        Ok(r.json)
    }

    /// Current parameter value; automatic semi-diameters use the layout solve.
    pub fn param_value(&self, path: &ParamPath) -> EngineResult<Option<f64>> {
        if let ParamPath::SemiDiameter { surface } = path
            && let Some(i) = self.system().surface_index(*surface)
            && matches!(
                self.system().surfaces[i].semi_diameter,
                mklight_core::model::SemiDiameter::Auto
            )
        {
            let tracer = mklight_trace::SequentialTracer::new(self.resolved()?)?;
            let tracer = mklight_analysis::util::default_fields(&tracer, &Default::default())?;
            let sd = tracer.semi_diameters(&Default::default());
            return Ok(Some(
                mklight_analysis::layout::resolved_semi_diameters(&tracer, &sd)[i],
            ));
        }
        match path.get(self.system()) {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                // Distinguish "no value" from "no such parameter": a path
                // that cannot be set does not exist.
                let mut probe = self.system().clone();
                path.set(&mut probe, 1.0).map_err(|_| e)?;
                Ok(None)
            }
        }
    }

    /// Where `autosave(None)` writes: `<project file>.autosave` next to the
    /// project, `None` for an untitled session.
    pub fn autosave_path(&self) -> Option<PathBuf> {
        self.path.as_ref().map(|p| {
            let mut s = p.clone().into_os_string();
            s.push(".autosave");
            PathBuf::from(s)
        })
    }

    /// Write the current project atomically to `path` (or `autosave_path`)
    /// without changing the session: its path, dirty flag and saved file
    /// stay as they are. The file is a normal project (open it to recover);
    /// catalog paths are recorded as absolute paths. Returns the path
    /// written.
    pub fn autosave(&self, path: Option<&Path>) -> EngineResult<PathBuf> {
        let target = match path {
            Some(p) => p.to_path_buf(),
            None => self.autosave_path().ok_or_else(|| {
                EngineError::Core(MkError::InvalidParam(
                    "an untitled session needs an explicit autosave path".into(),
                ))
            })?,
        };
        let mut p = self.project.clone();
        p.metadata.modified = now_rfc3339();
        p.metadata.engine_version = mklight_core::ENGINE_VERSION.to_string();
        p.save(&target)?;
        Ok(target)
    }

    // --- analyses

    /// Run an analysis on the current snapshot in the calling thread.
    pub fn run_analysis(
        &self,
        req: &AnalysisRequest,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisResult> {
        let analysis = self.engine.analysis(&req.analysis)?;
        let backend = self.engine.select_backend(
            req.backend.as_ref(),
            &req.analysis,
            req.allow_backend_fallback,
        )?;
        let (cache, snap, lib, generation) = (
            &self.resolved,
            &self.current,
            &self.library,
            self.library_generation,
        );
        let input = AnalysisInput::with_resolver(
            snap.clone(),
            lib.clone(),
            backend.info,
            Box::new(move || resolve_cached(cache, snap, lib, generation)),
        )
        .with_device(backend.device)
        .with_selection(backend.requested, backend.note)
        .with_library_hash(self.library_hash.clone());
        run_analysis(
            analysis.as_ref(),
            &input,
            &req.settings,
            backend.warning.into_iter().collect(),
            ctx,
        )
    }

    /// Build the job for an analysis of the current snapshot without queueing
    /// it. Settings, availability and backend are checked now, so those
    /// errors surface here rather than as failed jobs. The job owns
    /// everything it needs, so callers may release any lock on the session
    /// before handing it to `Engine::submit_job`.
    pub fn analysis_job(&self, req: &AnalysisRequest) -> EngineResult<JobSpec<AnalysisResult>> {
        let analysis = self.engine.analysis(&req.analysis)?;
        let settings = crate::analysis::prepare(analysis.as_ref(), &req.settings)?;
        let backend = self.engine.select_backend(
            req.backend.as_ref(),
            &req.analysis,
            req.allow_backend_fallback,
        )?;
        let snap = self.current.clone();
        let lib = self.library.clone();
        let lib_hash = self.library_hash.clone();
        let cache = self.resolved.clone();
        let generation = self.library_generation;
        let label = format!("{} @ {}", req.analysis, snap.hash().short());
        let stop = mklight_core::job::CancelToken::new();
        let job_stop = stop.clone();
        let mut spec = JobSpec::new(label, move |ctx: &JobContext| {
            let input = AnalysisInput::with_resolver(
                snap.clone(),
                lib.clone(),
                backend.info,
                Box::new(|| resolve_cached(&cache, &snap, &lib, generation)),
            )
            .with_stop_token(job_stop)
            .with_device(backend.device)
            .with_selection(backend.requested, backend.note)
            .with_library_hash(lib_hash);
            run_analysis(
                analysis.as_ref(),
                &input,
                &settings,
                backend.warning.into_iter().collect(),
                ctx,
            )
        });
        if matches!(
            req.analysis.as_str(),
            "optimize" | "tolerance" | "glass_substitution"
        ) {
            spec = spec.with_stop_token(stop);
        }
        if let Some(k) = &req.view_key {
            spec = spec.with_view_key(format!("session-{}/{k}", self.id));
        }
        Ok(spec)
    }

    /// Queue an analysis on the current snapshot (see `analysis_job`).
    pub fn submit_analysis(&self, req: &AnalysisRequest) -> EngineResult<JobId> {
        self.engine.submit_job(self.analysis_job(req)?)
    }
}

/// Solves captured by `Session::prepare_solves`, run without the session.
pub struct PendingSolves {
    base: Snapshot,
    library: Arc<MaterialLibrary>,
    library_generation: u64,
    solver: mklight_trace::SolveSettings,
}

impl PendingSolves {
    /// The snapshot the solves start from.
    pub fn base_hash(&self) -> SnapshotHash {
        self.base.hash()
    }

    /// Solve (`mklight_trace::apply_solves_with`): the solved system and
    /// the solve report (the `solves` result shape).
    pub fn run(&self, ctx: &JobContext) -> EngineResult<(OpticalSystem, Value)> {
        let (sys, report) =
            mklight_trace::apply_solves_with(self.base.system(), &self.library, &self.solver, ctx)?;
        let (json, _) = crate::analysis::geometric::crate_json(&report)?;
        Ok((sys, json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::fsutil::test_util::TempDir;
    use crate::scheduler::JobState;
    use mklight_core::materials::{Dispersion, Glass};
    use mklight_core::{MaterialRef, SurfaceId};
    use serde_json::json;
    use std::time::Duration;

    const LONG: Option<Duration> = Some(Duration::from_secs(30));

    fn engine() -> Engine {
        Engine::new(EngineConfig {
            worker_threads: 2,
            resolved_cache_capacity: 2,
            app_name: "test".into(),
            app_version: "0.0.1".into(),
            ..Default::default()
        })
    }

    fn thickness(id: u64) -> ParamPath {
        ParamPath::Thickness {
            surface: SurfaceId(id),
        }
    }

    #[test]
    fn edits_create_new_snapshots_atomically() {
        let e = engine();
        let mut s = e.new_session();
        let h0 = s.snapshot_hash();
        assert!(!s.is_dirty());
        let h1 = s
            .apply_edits(&[ParamEdit {
                path: thickness(1),
                value: 12.0,
            }])
            .unwrap();
        assert_ne!(h0, h1);
        assert!(s.is_dirty());
        assert_eq!(s.system().surfaces[1].thickness, 12.0);
        assert_eq!(s.project().system.surfaces[1].thickness, 12.0);
        // Setting the same value again keeps the hash.
        let again = s
            .apply_edits(&[ParamEdit {
                path: thickness(1),
                value: 12.0,
            }])
            .unwrap();
        assert_eq!(again, h1);
        // A batch with one bad edit is rejected entirely.
        let err = s.apply_edits(&[
            ParamEdit {
                path: thickness(1),
                value: 3.0,
            },
            ParamEdit {
                path: thickness(99),
                value: 1.0,
            },
        ]);
        assert!(err.is_err());
        assert_eq!(s.snapshot_hash(), h1);
        // An edit that makes the system invalid is rejected too.
        let mut bad = s.system().clone();
        bad.stop = 0;
        assert!(s.set_system(bad).is_err());
        assert_eq!(s.snapshot_hash(), h1);
        // Edits round-trip through JSON, including "inf".
        let pe: ParamEdit = serde_json::from_value(
            json!({"path": {"param": "thickness", "surface": 2}, "value": "inf"}),
        )
        .unwrap();
        assert!(pe.value.is_infinite());
    }

    #[test]
    fn resolved_cache_is_bounded_and_keyed_by_hash() {
        let e = engine();
        let mut s = e.new_session();
        let r0 = s.resolved().unwrap();
        assert!(Arc::ptr_eq(&r0, &s.resolved().unwrap()));
        let h0 = s.system().clone();
        for t in [11.0, 12.0, 13.0] {
            s.apply_edits(&[ParamEdit {
                path: thickness(1),
                value: t,
            }])
            .unwrap();
            s.resolved().unwrap();
        }
        assert_eq!(s.resolved_cache_len(), 2);
        // Going back to an evicted state re-resolves to an equal system.
        s.set_system(h0).unwrap();
        let r = s.resolved().unwrap();
        assert!(!Arc::ptr_eq(&r0, &r));
        assert_eq!(r.snapshot.hash(), r0.snapshot.hash());
    }

    fn user_catalog() -> GlassCatalog {
        let mut g = GlassCatalog {
            name: "MYCAT".into(),
            ..Default::default()
        };
        g.glasses.insert(
            "FAKE1".into(),
            Glass {
                name: "FAKE1".into(),
                dispersion: Dispersion::Cauchy {
                    a: 1.6,
                    b: 0.0,
                    c: 0.0,
                },
                nd: 1.6,
                vd: 1e9,
                min_um: 0.3,
                max_um: 2.0,
                thermal: None,
                cte: None,
                density: None,
                transmittance: vec![],
                relative_cost: None,
                substitution: Default::default(),
                comment: String::new(),
            },
        );
        g
    }

    #[test]
    fn catalogs_load_save_and_reopen() {
        let dir = TempDir::new("session");
        let cat_path = dir.path().join("mycat.json");
        std::fs::write(&cat_path, serde_json::to_string(&user_catalog()).unwrap()).unwrap();
        let e = engine();
        let mut s = e.new_session();
        let sum = s.load_catalog_file(&cat_path).unwrap();
        assert_eq!((sum.name.as_str(), sum.glass_count), ("MYCAT", 1));
        assert_eq!(s.catalogs().len(), 2);
        assert_eq!(s.glasses(Some("mycat")).unwrap()[0].name, "FAKE1");
        assert!(s.glasses(Some("nope")).is_err());
        assert!(s.glasses(None).unwrap().len() > 1);

        let mut sys = s.system().clone();
        sys.surfaces[1].material = MaterialRef::Catalog {
            catalog: Some("MYCAT".into()),
            name: "FAKE1".into(),
        };
        s.set_system(sys).unwrap();
        let n = s.resolved().unwrap().surfaces[1].n_after[0];
        assert!((n - 1.6).abs() < 1e-15);

        let proj = dir.path().join("p.mklight");
        s.save(Some(&proj)).unwrap();
        assert!(!s.is_dirty());
        let saved = &s.project().catalogs[0];
        assert_eq!(saved.relative_path.as_deref(), Some("mycat.json"));
        assert_eq!(s.project().metadata.app_name, "test");

        let s2 = e.open_session(&proj).unwrap();
        assert!(s2.open_warnings().is_empty(), "{:?}", s2.open_warnings());
        assert_eq!(s2.snapshot_hash(), s.snapshot_hash());
        assert!(s2.resolved().is_ok());

        // A missing catalog is a warning, not an error.
        std::fs::remove_file(&cat_path).unwrap();
        let s3 = e.open_session(&proj).unwrap();
        assert_eq!(s3.open_warnings().len(), 1);
        assert!(s3.resolved().is_err());
        assert_eq!(s3.project().catalogs.len(), 1);
    }

    #[test]
    fn moved_project_keeps_relative_catalog_paths() {
        let root = TempDir::new("move");
        let a = root.path().join("A");
        std::fs::create_dir_all(a.join("glass")).unwrap();
        std::fs::write(
            a.join("glass/mycat.json"),
            serde_json::to_string(&user_catalog()).unwrap(),
        )
        .unwrap();
        let e = engine();
        let mut s = e.new_session();
        s.load_catalog_file(&a.join("glass/mycat.json")).unwrap();
        s.save(Some(&a.join("p.mklight"))).unwrap();
        assert_eq!(
            s.project().catalogs[0].relative_path.as_deref(),
            Some("glass/mycat.json")
        );
        // Move the folder twice, opening and re-saving in between.
        let mut here = a;
        for name in ["B", "C"] {
            let next = root.path().join(name);
            std::fs::rename(&here, &next).unwrap();
            here = next;
            let mut s = e.open_session(&here.join("p.mklight")).unwrap();
            assert!(s.open_warnings().is_empty(), "{:?}", s.open_warnings());
            let canon_dir = std::fs::canonicalize(&here).unwrap();
            let c = &s.project().catalogs[0];
            assert!(
                Path::new(c.path.as_deref().unwrap()).starts_with(&canon_dir),
                "absolute path not updated: {c:?}"
            );
            s.save(None).unwrap();
            let c = &s.project().catalogs[0];
            assert_eq!(c.relative_path.as_deref(), Some("glass/mycat.json"));
        }
        // Remove the catalog: the reference keeps its relative path on save
        // (the only location left), and loads again once the file is back.
        let cat = here.join("glass/mycat.json");
        let bytes = std::fs::read(&cat).unwrap();
        std::fs::remove_file(&cat).unwrap();
        let mut s = e.open_session(&here.join("p.mklight")).unwrap();
        assert_eq!(s.open_warnings().len(), 1);
        s.save(None).unwrap();
        assert_eq!(
            s.project().catalogs[0].relative_path.as_deref(),
            Some("glass/mycat.json")
        );
        std::fs::write(&cat, bytes).unwrap();
        let s = e.open_session(&here.join("p.mklight")).unwrap();
        assert!(s.open_warnings().is_empty(), "{:?}", s.open_warnings());
        // A catalog outside the project folder gets no relative path.
        let other = TempDir::new("move-out");
        let mut s = e.new_session();
        let outside = other.path().join("OUT.json");
        let mut cat = user_catalog();
        cat.name = "OUT".into();
        std::fs::write(&outside, serde_json::to_string(&cat).unwrap()).unwrap();
        s.load_catalog_file(&outside).unwrap();
        s.save(Some(&here.join("q.mklight"))).unwrap();
        assert_eq!(s.project().catalogs[0].relative_path, None);
    }

    #[test]
    fn library_changes_are_visible_in_hashes_and_results() {
        let dir = TempDir::new("libhash");
        let p = dir.path().join("mycat.json");
        let mut cat = user_catalog();
        std::fs::write(&p, serde_json::to_string(&cat).unwrap()).unwrap();
        let e = engine();
        let mut s = e.new_session();
        let h0 = s.library_hash().to_string();
        assert_eq!(s.info()["library_hash"], h0.as_str());
        s.load_catalog_file(&p).unwrap();
        let h1 = s.library_hash().to_string();
        assert_ne!(h0, h1);
        let snap = s.snapshot_hash();
        let req = AnalysisRequest::new("validate", Value::Null);
        let r1 = s.run_analysis(&req, &JobContext::new()).unwrap();
        assert_eq!(r1.library_hash, h1);
        // Reloading identical data keeps the hash.
        s.load_catalog_file(&p).unwrap();
        assert_eq!(s.library_hash(), h1);
        // Edited catalog file, same name, same system: the snapshot hash is
        // unchanged but the library hash (and so result identity) is not.
        cat.glasses.get_mut("FAKE1").unwrap().dispersion = Dispersion::Cauchy {
            a: 1.7,
            b: 0.0,
            c: 0.0,
        };
        std::fs::write(&p, serde_json::to_string(&cat).unwrap()).unwrap();
        s.load_catalog_file(&p).unwrap();
        assert_eq!(s.snapshot_hash(), snap);
        assert_ne!(s.library_hash(), h1);
        let id = s.submit_analysis(&req).unwrap();
        e.wait_job(id, LONG).unwrap();
        let r2 = e.take_result(id).unwrap();
        assert_eq!(r2.run_info.snapshot_hash, r1.run_info.snapshot_hash);
        assert_eq!(r2.library_hash, s.library_hash());
        assert_ne!(r2.library_hash, r1.library_hash);
        assert_eq!(r2.envelope()["library_hash"], s.library_hash());
    }

    #[test]
    fn agf_loading_reports_parser_errors() {
        let dir = TempDir::new("agf");
        let p = dir.path().join("X.agf");
        std::fs::write(&p, "NM nonsense").unwrap();
        let e = engine();
        let mut s = e.new_session();
        // The AGF reader lives in mklight-core; whatever it returns for bad
        // input must surface as an error and leave the library unchanged.
        assert!(s.load_catalog_file(&p).is_err());
        assert_eq!(s.catalogs().len(), 1);
        assert!(s.project().catalogs.is_empty());
    }

    #[test]
    fn agf_loading_accepts_utf16_catalogs() {
        // Re-export a built-in glass as UTF-16 LE AGF, the common vendor encoding.
        let builtin = mklight_core::materials::builtin::catalog();
        let mut cat = GlassCatalog {
            name: "USER".into(),
            glasses: Default::default(),
        };
        cat.glasses
            .insert("N-BK7".into(), builtin.glasses["N-BK7"].clone());
        let (text, _) = agf::write_agf(&agf::AgfCatalog::from_catalog(cat));
        let bytes = agf::encode_text(&text, agf::TextEncoding::Utf16Le);
        let dir = TempDir::new("agf16");
        let p = dir.path().join("USER.agf");
        std::fs::write(&p, bytes).unwrap();
        let e = engine();
        let mut s = e.new_session();
        let summary = s.load_catalog_file(&p).unwrap();
        assert_eq!(summary.glass_count, 1);
        assert_eq!(s.catalogs().len(), 2);
    }

    #[test]
    fn submit_runs_against_submission_snapshot() {
        let e = engine();
        let mut s = e.new_session();
        let h0 = s.snapshot_hash();
        let id = s
            .submit_analysis(&AnalysisRequest::new("system_summary", Value::Null))
            .unwrap();
        // Editing after submission does not affect the job.
        s.apply_edits(&[ParamEdit {
            path: thickness(1),
            value: 33.0,
        }])
        .unwrap();
        assert_eq!(e.wait_job(id, LONG).unwrap(), JobState::Done);
        let r = e.take_result(id).unwrap();
        assert_eq!(r.run_info.snapshot_hash, h0.to_hex());
        assert_eq!(r.json["surfaces"][1]["thickness"], json!(10.0));
    }

    #[test]
    fn submit_errors_surface_immediately() {
        let e = engine();
        // Install an unavailable fixture now that the wave solver replaces it.
        let mut planned = crate::analysis::AnalysisRegistry::new();
        crate::analysis::builtin::register_all(&mut planned);
        e.register_analysis(planned.get("psf").unwrap());
        let s = e.new_session();
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("nope", Value::Null)),
            Err(EngineError::UnknownAnalysis(_))
        ));
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("psf", Value::Null)),
            Err(EngineError::AnalysisUnavailable { .. })
        ));
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("rng_stream", json!({"count": -1}))),
            Err(EngineError::InvalidSettings(_))
        ));
        let mut req = AnalysisRequest::new("validate", Value::Null);
        req.backend = Some(Backend::Gpu("nonexistent".into()));
        assert!(matches!(
            s.submit_analysis(&req),
            Err(EngineError::BackendUnavailable(_))
        ));
        req.allow_backend_fallback = true;
        let id = s.submit_analysis(&req).unwrap();
        e.wait_job(id, LONG).unwrap();
        let r = e.take_result(id).unwrap();
        assert_eq!(r.run_info.backend, "cpu_reference");
        assert!(r.run_info.warnings[0].contains("gpu:nonexistent"));
    }

    #[test]
    fn default_gpu_backend_is_used_per_analysis() {
        use crate::backend::{BackendCapabilities, BackendInfo, BackendProvider, DeviceHandle};
        use mklight_core::job::Precision;
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
                        analyses: vec!["system_summary".into()],
                        ..Default::default()
                    },
                }
            }
            fn device(&self) -> Option<DeviceHandle> {
                Some(Arc::new(7u32))
            }
        }
        let e = engine();
        e.register_backend(Arc::new(Emu));
        e.set_default_backend(Backend::Gpu("emu".into())).unwrap();
        let s = e.new_session();
        let run = |name: &str| {
            let id = s
                .submit_analysis(&AnalysisRequest::new(name, Value::Null))
                .unwrap();
            assert_eq!(e.wait_job(id, LONG).unwrap(), JobState::Done);
            e.take_result(id).unwrap().run_info
        };
        let ri = run("system_summary");
        assert_eq!(
            (ri.backend.as_str(), ri.precision),
            ("gpu:emu", Precision::F32Gpu)
        );
        assert!(ri.warnings.is_empty(), "{:?}", ri.warnings);
        // Not implemented by the default backend: the reference, with a note.
        let ri = run("validate");
        assert_eq!(ri.backend, "cpu_reference");
        assert!(
            ri.warnings
                .iter()
                .any(|w| w.contains("does not implement 'validate'"))
        );
        // Explicitly requested without fallback: an error at submission.
        let mut req = AnalysisRequest::new("validate", Value::Null);
        req.backend = Some(Backend::Gpu("emu".into()));
        assert!(matches!(
            s.submit_analysis(&req),
            Err(EngineError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn view_keys_supersede_within_a_session_only() {
        let e = engine();
        let a = e.new_session();
        let b = e.new_session();
        let slow = json!({"count": 1000000, "chunk": 1000, "delay_ms": 2.0});
        let a1 = a
            .submit_analysis(&AnalysisRequest::new("rng_stream", slow.clone()).view_key("v"))
            .unwrap();
        let b1 = b
            .submit_analysis(&AnalysisRequest::new("rng_stream", slow.clone()).view_key("v"))
            .unwrap();
        let a2 = a
            .submit_analysis(
                &AnalysisRequest::new("rng_stream", json!({"count": 10})).view_key("v"),
            )
            .unwrap();
        assert_eq!(e.wait_job(a1, LONG).unwrap(), JobState::Cancelled);
        assert_eq!(e.job_status(a1).unwrap().superseded_by, Some(a2));
        assert_eq!(e.wait_job(a2, LONG).unwrap(), JobState::Done);
        assert_ne!(e.job_status(b1).unwrap().state, JobState::Cancelled);
        e.cancel_job(b1).unwrap();
        assert_eq!(e.wait_job(b1, LONG).unwrap(), JobState::Cancelled);
    }

    #[test]
    fn sync_run_matches_job_run() {
        let e = engine();
        let s = e.new_session();
        let req = AnalysisRequest::new("rng_stream", json!({"seed": 5, "count": 1000}));
        let sync = s.run_analysis(&req, &JobContext::new()).unwrap();
        let id = s.submit_analysis(&req).unwrap();
        e.wait_job(id, LONG).unwrap();
        let job = e.take_result(id).unwrap();
        assert_eq!(sync.json, job.json);
        assert_eq!(sync.buffers, job.buffers);
        assert_eq!(sync.run_info.settings, job.run_info.settings);
    }

    #[test]
    fn project_sections() {
        let e = engine();
        let mut s = e.new_session();
        s.set_project_section("views", json!([{"key": "k", "analysis": "validate"}]))
            .unwrap();
        s.set_project_section("merit", json!({"x": 1})).unwrap();
        assert_eq!(s.project().views[0].key, "k");
        assert!(s.set_project_section("system", json!({})).is_err());
        assert!(s.set_project_section("views", json!(3)).is_err());
        assert!(s.save(None).is_err()); // never saved: needs a path
    }
}
