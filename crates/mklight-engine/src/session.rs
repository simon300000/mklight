//! `Session`: one open document. It holds the project, the material library
//! (built-in plus the user catalogs the project references) and the current
//! immutable `Snapshot`.
//!
//! Every edit builds a new snapshot (validated, content-hashed) and swaps it
//! in only if it is valid, so the session never holds an invalid system and
//! running jobs keep the snapshot they started with. `ResolvedSystem`s are
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

use crate::analysis::{AnalysisInput, AnalysisResult, run_analysis};
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
    resolved: ResolvedCache,
    path: Option<PathBuf>,
    dirty: bool,
    open_warnings: Vec<String>,
    migration: MigrationReport,
}

/// Read a catalog file. AGF files are named by their file stem.
pub fn read_catalog_file(path: &Path, format: CatalogFormat) -> EngineResult<GlassCatalog> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        EngineError::Core(MkError::Io(format!("cannot read {}: {e}", path.display())))
    })?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut cat = match format {
        CatalogFormat::Agf => agf::parse_agf(&stem, &text)?,
        CatalogFormat::Json => serde_json::from_str::<GlassCatalog>(&text)?,
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
    /// become `open_warnings`; the project keeps the references.
    fn reload_catalogs(&mut self) {
        let dir = self
            .path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let mut lib = MaterialLibrary::with_builtin();
        for r in &self.project.catalogs {
            let mut last_err = None;
            let mut loaded = false;
            for cand in r.candidates(dir.as_deref()) {
                match read_catalog_file(&cand, r.format) {
                    Ok(mut c) => {
                        c.name = r.name.clone();
                        lib.add_catalog(Arc::new(c));
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

    /// Session state for front ends.
    pub fn info(&self) -> Value {
        serde_json::json!({
            "session_id": self.id,
            "path": self.path.as_ref().map(|p| p.display().to_string()),
            "dirty": self.dirty,
            "snapshot_hash": self.current.hash(),
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
    /// paths relative to the project directory where possible.
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
                c.relative_path = c
                    .path
                    .as_deref()
                    .and_then(|a| Path::new(a).strip_prefix(&dir).ok())
                    .map(|r| r.to_string_lossy().into_owned());
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

    // --- analyses

    /// Run an analysis on the current snapshot in the calling thread.
    pub fn run_analysis(
        &self,
        req: &AnalysisRequest,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisResult> {
        let analysis = self.engine.analysis(&req.analysis)?;
        let (backend, fallback) = self
            .engine
            .select_backend(req.backend.as_ref(), req.allow_backend_fallback)?;
        let (cache, snap, lib, generation) = (
            &self.resolved,
            &self.current,
            &self.library,
            self.library_generation,
        );
        let input = AnalysisInput::with_resolver(
            snap.clone(),
            lib.clone(),
            backend,
            Box::new(move || resolve_cached(cache, snap, lib, generation)),
        );
        run_analysis(
            analysis.as_ref(),
            &input,
            &req.settings,
            fallback.into_iter().collect(),
            ctx,
        )
    }

    /// Queue an analysis on the current snapshot. Settings, availability and
    /// backend are checked now, so those errors surface here rather than as
    /// failed jobs.
    pub fn submit_analysis(&self, req: &AnalysisRequest) -> EngineResult<JobId> {
        let analysis = self.engine.analysis(&req.analysis)?;
        let settings = crate::analysis::prepare(analysis.as_ref(), &req.settings)?;
        let (backend, fallback) = self
            .engine
            .select_backend(req.backend.as_ref(), req.allow_backend_fallback)?;
        let snap = self.current.clone();
        let lib = self.library.clone();
        let cache = self.resolved.clone();
        let generation = self.library_generation;
        let label = format!("{} @ {}", req.analysis, snap.hash().short());
        let mut spec = JobSpec::new(label, move |ctx: &JobContext| {
            let input = AnalysisInput::with_resolver(
                snap.clone(),
                lib.clone(),
                backend,
                Box::new(|| resolve_cached(&cache, &snap, &lib, generation)),
            );
            run_analysis(
                analysis.as_ref(),
                &input,
                &settings,
                fallback.into_iter().collect(),
                ctx,
            )
        });
        if let Some(k) = &req.view_key {
            spec = spec.with_view_key(format!("session-{}/{k}", self.id));
        }
        self.engine.submit(spec)
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
        let s = e.new_session();
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("nope", Value::Null)),
            Err(EngineError::UnknownAnalysis(_))
        ));
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("spot_diagram", Value::Null)),
            Err(EngineError::AnalysisUnavailable { .. })
        ));
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("rng_stream", json!({"count": -1}))),
            Err(EngineError::InvalidSettings(_))
        ));
        let mut req = AnalysisRequest::new("validate", Value::Null);
        req.backend = Some(Backend::Gpu("metal".into()));
        assert!(matches!(
            s.submit_analysis(&req),
            Err(EngineError::BackendUnavailable(_))
        ));
        req.allow_backend_fallback = true;
        let id = s.submit_analysis(&req).unwrap();
        e.wait_job(id, LONG).unwrap();
        let r = e.take_result(id).unwrap();
        assert_eq!(r.run_info.backend, "cpu_reference");
        assert!(r.run_info.warnings[0].contains("gpu:metal"));
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
