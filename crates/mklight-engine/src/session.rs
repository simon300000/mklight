//! `Session`: one open document. It holds the project, the material library
//! (built-in, the front end's app-wide library catalogs, and the user
//! catalogs the project references, with the user's availability overrides
//! applied) and the current immutable `Snapshot`.
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

use mklight_core::import::{PrescriptionImport, read_prescription_file};
use mklight_core::materials::builtin::BUILTIN_CATALOG_NAME;
use mklight_core::materials::manage::{
    CatalogChanges, GlassDetails, GlassOverride, GlassRow, apply_overrides, diff_catalogs,
    glass_details, glass_row,
};
use mklight_core::materials::{GlassCatalog, MaterialLibrary, agf};
use mklight_core::param::ParamPath;
use mklight_core::snapshot::SnapshotHash;
use mklight_core::{JobContext, MkError, OpticalSystem, ResolvedSystem, Snapshot};

use crate::analysis::{AnalysisInput, AnalysisResult, library_hash, run_analysis};
use crate::backend::Backend;
use crate::engine::Engine;
use crate::error::{EngineError, EngineResult};
use crate::lru::LruCache;
use crate::project::{AnalysisView, CatalogFormat, CatalogRef, Project, ProjectMetadata};
use crate::scheduler::{JobId, JobSpec};
use crate::time::now_rfc3339;

/// One parameter edit (`ParamPath` + new value). `value` accepts `"inf"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamEdit {
    pub path: ParamPath,
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub value: f64,
}

/// A request to run an analysis on the session's current snapshot. Every
/// field but `analysis` has a default; a key that is not a field is an
/// error.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    /// Accuracy tier (`design` by default; `preview` is opt-in and labelled
    /// in `RunInfo`).
    pub tier: mklight_core::Tier,
    /// Aim-stop policy; `None` uses the engine default
    /// (`EngineConfig::aim_stop`).
    pub aim_stop: Option<mklight_core::AimStopKind>,
    /// Route options (`crate::route`); `None` uses the engine default
    /// (`EngineConfig::route`).
    pub route: Option<crate::route::RouteOptions>,
    /// Recompute: read no stored analysis-graph node (the result and its
    /// intermediates) nor adapter cache, and replace the stored ones with
    /// what this run builds, so later identical requests get the new value
    /// (a user's explicit "Run"/"Try Again"). Material resolution may
    /// still come from the session's cache.
    pub fresh: bool,
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
    /// Loaded from the front end's app-wide library.
    pub library: bool,
    /// Recorded in the project (`Project::catalogs`). A catalog can be both;
    /// the project's copy is the one in effect then.
    pub in_project: bool,
}

/// A glass table row (`mklight_core::materials::manage::GlassRow`).
pub type GlassSummary = GlassRow;

/// A catalog of the app-wide library: loaded into the session by the front
/// end, not recorded in the project until the design uses it (see
/// `Session::library_catalogs_in_use`).
#[derive(Clone, Debug)]
struct LibraryCatalog {
    catalog: Arc<GlassCatalog>,
    /// Canonical path, for catalogs read from a file.
    path: Option<String>,
    /// File name, for catalogs given as bytes.
    file_name: Option<String>,
    format: CatalogFormat,
    /// Modification time and size when read, so an unchanged file is not
    /// read again.
    stamp: Option<(std::time::SystemTime, u64)>,
}

/// A library file `set_library_catalog_files` could not load.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LibraryError {
    pub path: String,
    pub message: String,
}

/// The outcome of `set_library_catalog_files`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LibrarySync {
    /// The library catalogs now loaded, in order.
    pub catalogs: Vec<CatalogSummary>,
    pub errors: Vec<LibraryError>,
    /// Whether the material library changed.
    pub changed: bool,
}

fn file_stamp(path: &Path) -> Option<(std::time::SystemTime, u64)> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
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
    /// App-wide library catalogs, in the front end's order.
    library_catalogs: Vec<LibraryCatalog>,
    /// Catalogs loaded for the project, or added in memory, in load order.
    project_catalogs: Vec<Arc<GlassCatalog>>,
    /// Project catalog references that could not be loaded, with why.
    missing_catalogs: Vec<(String, String)>,
    /// The user's availability overrides, applied to the library.
    overrides: Vec<GlassOverride>,
    /// Catalogs removed from the project, for `restore_catalog`.
    removed: Vec<(CatalogRef, Arc<GlassCatalog>)>,
}

/// A project section's value as its type; a value of another shape is an
/// invalid argument.
fn section_value<T: serde::de::DeserializeOwned>(section: &str, value: Value) -> EngineResult<T> {
    serde_json::from_value(value)
        .map_err(|e| EngineError::InvalidArgument(format!("project section '{section}': {e}")))
}

/// Read a catalog file. AGF files are named by their file stem.
pub fn read_catalog_file(path: &Path, format: CatalogFormat) -> EngineResult<GlassCatalog> {
    let bytes = std::fs::read(path).map_err(|e| {
        EngineError::Core(MkError::Io(format!("cannot read {}: {e}", path.display())))
    })?;
    parse_catalog_bytes(path, &bytes, format)
}

/// Parse the contents of a catalog file named `path` (only its name is
/// used: the file stem names an AGF catalog, and messages show it).
pub fn parse_catalog_bytes(
    path: &Path,
    bytes: &[u8],
    format: CatalogFormat,
) -> EngineResult<GlassCatalog> {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut cat = match format {
        // Vendor AGF files are often UTF-16; `parse_agf_bytes` detects the encoding.
        CatalogFormat::Agf => {
            let agf = agf::parse_agf_bytes(&stem, bytes, &agf::AgfOptions::default())?;
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
        CatalogFormat::Json => serde_json::from_slice::<GlassCatalog>(bytes)?,
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

/// `(catalog, name)` of every catalog material in a system's JSON form.
fn collect_catalog_materials(v: &Value, out: &mut Vec<(Option<String>, String)>) {
    match v {
        Value::Object(m) => {
            if m.get("type").and_then(Value::as_str) == Some("catalog")
                && let Some(name) = m.get("name").and_then(Value::as_str)
            {
                let catalog = m.get("catalog").and_then(Value::as_str).map(str::to_string);
                out.push((catalog, name.to_string()));
            }
            for x in m.values() {
                collect_catalog_materials(x, out);
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_catalog_materials(x, out);
            }
        }
        _ => {}
    }
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
            library_catalogs: Vec::new(),
            project_catalogs: Vec::new(),
            missing_catalogs: Vec::new(),
            overrides: Vec::new(),
            removed: Vec::new(),
        };
        s.reload_catalogs();
        Ok(s)
    }

    pub(crate) fn open(engine: &Engine, path: &Path) -> EngineResult<Session> {
        let project = Project::load(path)?;
        Self::from_project(engine, project, Some(path.to_path_buf()))
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
        self.project_catalogs.clear();
        self.missing_catalogs.clear();
        for r in &mut self.project.catalogs {
            let mut last_err = None;
            let mut loaded = false;
            for cand in r.candidates(dir.as_deref()) {
                match read_catalog_file(&cand, r.format) {
                    Ok(mut c) => {
                        c.name = r.name.clone();
                        self.project_catalogs.push(Arc::new(c));
                        let abs = std::fs::canonicalize(&cand).unwrap_or(cand);
                        r.path = Some(abs.display().to_string());
                        loaded = true;
                        break;
                    }
                    Err(e) => last_err = Some(format!("{}: {e}", cand.display())),
                }
            }
            if !loaded {
                self.missing_catalogs.push((
                    r.name.clone(),
                    last_err.unwrap_or_else(|| "no path recorded".into()),
                ));
            }
        }
        self.rebuild_library();
    }

    /// The catalogs in effect, before overrides, in library order:
    /// built-in, library, project; a later catalog replaces an earlier one
    /// of the same name (case-insensitive).
    fn effective_catalogs(&self) -> Vec<Arc<GlassCatalog>> {
        let mut lib = MaterialLibrary::with_builtin();
        for c in &self.library_catalogs {
            lib.add_catalog(c.catalog.clone());
        }
        for c in &self.project_catalogs {
            lib.add_catalog(c.clone());
        }
        lib.catalogs().to_vec()
    }

    /// The catalog named `name` as loaded, before overrides.
    fn original_catalog(&self, name: &str) -> Option<Arc<GlassCatalog>> {
        self.effective_catalogs()
            .into_iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Composes the material library from its parts and applies the
    /// overrides. A project catalog that could not be loaded but that the
    /// library provides is not reported as missing.
    fn rebuild_library(&mut self) {
        let mut lib = MaterialLibrary::new();
        for c in self.effective_catalogs() {
            match apply_overrides(&c, &self.overrides) {
                Some(o) => lib.add_catalog(Arc::new(o)),
                None => lib.add_catalog(c),
            }
        }
        self.open_warnings = self
            .missing_catalogs
            .iter()
            .filter(|(n, _)| {
                !lib.catalogs()
                    .iter()
                    .any(|c| c.name.eq_ignore_ascii_case(n))
            })
            .map(|(n, why)| format!("glass catalog '{n}' could not be loaded ({why})"))
            .collect();
        self.set_library(lib);
    }

    fn set_library(&mut self, lib: MaterialLibrary) {
        let hash = library_hash(&lib);
        if hash == self.library_hash && self.library_generation > 0 {
            // Same content: keep the library and the resolved cache.
            return;
        }
        self.library_hash = hash;
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
    /// Problems found while opening (missing or unreadable catalogs).
    pub fn open_warnings(&self) -> &[String] {
        &self.open_warnings
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
            "views" => self.project.views = section_value::<Vec<AnalysisView>>(section, value)?,
            "merit" => self.project.merit = value,
            "variables" => self.project.variables = value,
            "tolerances" => self.project.tolerances = value,
            "metadata" => self.project.metadata = section_value::<ProjectMetadata>(section, value)?,
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

    /// The project with its save metadata (modified time, app and engine
    /// versions) brought up to date.
    fn stamped_project(&self) -> Project {
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
        // The library catalogs the design uses go into the project, so it
        // opens where the library is different.
        for r in self.library_catalogs_in_use() {
            if !p
                .catalogs
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&r.name))
            {
                p.catalogs.push(r);
            }
        }
        p
    }

    /// The project as `.mklight` text, as `save` would write it, without a
    /// file (the web build's save): catalog references are kept as they
    /// are. Marks the session clean.
    pub fn save_to_string(&mut self) -> EngineResult<String> {
        let p = self.stamped_project();
        let text = p.to_json_string()?;
        self.project = p;
        self.dirty = false;
        Ok(text)
    }

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
        let mut p = self.stamped_project();
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

    // --- lens prescription files

    /// Replace the system with one imported from a `.zmx` or `.seq` file
    /// (`docs/IMPORT.md`). Glasses resolve against this session's library,
    /// so load user catalogs first. Content that does not map is listed in
    /// the returned warnings; only an unreadable file or one without
    /// surfaces is an error, and leaves the session unchanged.
    pub fn import_prescription(&mut self, path: &Path) -> EngineResult<PrescriptionImport> {
        let imported = read_prescription_file(path, &self.library)?;
        self.set_system(imported.system.clone())?;
        Ok(imported)
    }

    // --- materials

    /// Load a user glass catalog from the contents of a file named
    /// `file_name` (AGF, or JSON by `.json` extension; no file is read: the
    /// web build's catalog load). It is recorded in the project by that
    /// name, relative to the project's folder.
    pub fn load_catalog_bytes(
        &mut self,
        file_name: &str,
        bytes: &[u8],
    ) -> EngineResult<CatalogSummary> {
        let path = Path::new(file_name);
        let format = CatalogFormat::from_path(path);
        let cat = parse_catalog_bytes(path, bytes, format)?;
        let r = CatalogRef {
            name: cat.name.clone(),
            path: None,
            relative_path: Some(file_name.to_string()),
            format,
        };
        let name = cat.name.clone();
        self.add_catalog(cat, Some(r));
        Ok(self.summary(&name))
    }

    /// Load a user glass catalog (AGF, or JSON by `.json` extension), add it
    /// to the library and record it in the project. Replaces a catalog of
    /// the same name.
    pub fn load_catalog_file(&mut self, path: &Path) -> EngineResult<CatalogSummary> {
        let format = CatalogFormat::from_path(path);
        let cat = read_catalog_file(path, format)?;
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let r = CatalogRef {
            name: cat.name.clone(),
            path: Some(abs.display().to_string()),
            relative_path: None,
            format,
        };
        let name = cat.name.clone();
        self.add_catalog(cat, Some(r));
        Ok(self.summary(&name))
    }

    /// Add an in-memory catalog. With a `CatalogRef` it is recorded in the
    /// project so it reloads on open.
    pub fn add_catalog(&mut self, cat: GlassCatalog, reference: Option<CatalogRef>) {
        self.project_catalogs
            .retain(|c| !c.name.eq_ignore_ascii_case(&cat.name));
        self.missing_catalogs
            .retain(|(n, _)| !n.eq_ignore_ascii_case(&cat.name));
        self.project_catalogs.push(Arc::new(cat));
        if let Some(r) = reference {
            self.project
                .catalogs
                .retain(|c| !c.name.eq_ignore_ascii_case(&r.name));
            self.project.catalogs.push(r);
            self.dirty = true;
        }
        self.rebuild_library();
    }

    /// Remove a catalog from the project (its reference and, unless the
    /// app-wide library has a catalog of that name, its glasses). Returns
    /// the removed reference; `restore_catalog` undoes the removal. The
    /// built-in catalog and library catalogs the project does not record
    /// cannot be removed here (`remove_library_catalog` removes those).
    pub fn remove_catalog(&mut self, name: &str) -> EngineResult<CatalogRef> {
        let reference = self
            .project
            .catalogs
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .cloned();
        let loaded = self
            .project_catalogs
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .cloned();
        if reference.is_none() && loaded.is_none() {
            return Err(EngineError::Core(MkError::UnknownMaterial(format!(
                "no project catalog named '{name}'"
            ))));
        }
        self.project
            .catalogs
            .retain(|c| !c.name.eq_ignore_ascii_case(name));
        self.project_catalogs
            .retain(|c| !c.name.eq_ignore_ascii_case(name));
        self.missing_catalogs
            .retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        let reference = reference.unwrap_or_else(|| CatalogRef {
            name: loaded.as_ref().map_or(name, |c| &c.name).to_string(),
            path: None,
            relative_path: None,
            format: CatalogFormat::Json,
        });
        if let Some(c) = loaded {
            self.removed
                .retain(|(r, _)| !r.name.eq_ignore_ascii_case(&reference.name));
            self.removed.push((reference.clone(), c));
        }
        self.dirty = true;
        self.rebuild_library();
        Ok(reference)
    }

    /// Put back a catalog `remove_catalog` removed, with the data it had
    /// (no file is read).
    pub fn restore_catalog(&mut self, name: &str) -> EngineResult<CatalogSummary> {
        let i = self
            .removed
            .iter()
            .rposition(|(r, _)| r.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                EngineError::Core(MkError::UnknownMaterial(format!(
                    "no removed catalog named '{name}'"
                )))
            })?;
        let (r, c) = self.removed.remove(i);
        let recorded = r.path.is_some() || r.relative_path.is_some();
        let name = c.name.clone();
        self.add_catalog((*c).clone(), recorded.then_some(r));
        Ok(self.summary(&name))
    }

    /// Read a catalog again from its file and report what changed. The
    /// project's copy is reloaded when the project records the catalog
    /// (paths as on open), else the library's.
    pub fn reload_catalog(&mut self, name: &str) -> EngineResult<CatalogChanges> {
        let dir = self.project_dir();
        let reference = self
            .project
            .catalogs
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .cloned();
        if let Some(r) = reference {
            let mut last_err = None;
            for cand in r.candidates(dir.as_deref()) {
                match read_catalog_file(&cand, r.format) {
                    Ok(c) => {
                        // A library catalog of the same name is read again
                        // too, so removing the project's copy later does
                        // not bring back stale glasses.
                        if let Ok(i) = self.library_index(&r.name)
                            && let Some(p) = self.library_catalogs[i].path.clone()
                            && let Ok(lc) =
                                read_catalog_file(Path::new(&p), self.library_catalogs[i].format)
                        {
                            let stamp = file_stamp(Path::new(&p));
                            let entry = &mut self.library_catalogs[i];
                            let mut lc = lc;
                            lc.name = entry.catalog.name.clone();
                            entry.catalog = Arc::new(lc);
                            entry.stamp = stamp;
                        }
                        return Ok(self.replace_project_catalog(&r.name, c));
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            return Err(last_err.unwrap_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "catalog '{}' has no file to reload (load it again)",
                    r.name
                ))
            }));
        }
        let i = self.library_index(name)?;
        let lc = &self.library_catalogs[i];
        let Some(path) = lc.path.clone() else {
            return Err(EngineError::InvalidArgument(format!(
                "catalog '{}' was not read from a file (load it again)",
                lc.catalog.name
            )));
        };
        let c = read_catalog_file(Path::new(&path), lc.format)?;
        Ok(self.replace_library_catalog(i, c, file_stamp(Path::new(&path))))
    }

    /// `reload_catalog` with the file's new contents (the web build): the
    /// catalog keeps its name and format.
    pub fn reload_catalog_bytes(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> EngineResult<CatalogChanges> {
        let reference = self
            .project
            .catalogs
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .cloned();
        if let Some(r) = reference {
            let file = r
                .relative_path
                .clone()
                .or(r.path.clone())
                .unwrap_or_default();
            let c = parse_catalog_bytes(Path::new(&file), bytes, r.format)?;
            return Ok(self.replace_project_catalog(&r.name, c));
        }
        if self
            .project_catalogs
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(name))
        {
            return Err(EngineError::InvalidArgument(format!(
                "catalog '{name}' is not recorded with a format (load it again)"
            )));
        }
        let i = self.library_index(name)?;
        let lc = &self.library_catalogs[i];
        let file = lc.file_name.clone().or(lc.path.clone()).unwrap_or_default();
        let c = parse_catalog_bytes(Path::new(&file), bytes, lc.format)?;
        let stamp = lc.stamp;
        Ok(self.replace_library_catalog(i, c, stamp))
    }

    fn project_dir(&self) -> Option<PathBuf> {
        self.path
            .as_deref()
            .and_then(Path::parent)
            .map(|d| {
                if d.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    d
                }
            })
            .map(Path::to_path_buf)
    }

    fn library_index(&self, name: &str) -> EngineResult<usize> {
        self.library_catalogs
            .iter()
            .position(|c| c.catalog.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                EngineError::Core(MkError::UnknownMaterial(format!(
                    "no user catalog named '{name}'"
                )))
            })
    }

    fn replace_project_catalog(&mut self, name: &str, mut c: GlassCatalog) -> CatalogChanges {
        c.name = name.to_string();
        let before = self
            .project_catalogs
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
            .cloned()
            .unwrap_or_else(|| {
                Arc::new(GlassCatalog {
                    name: name.to_string(),
                    ..Default::default()
                })
            });
        let changes = diff_catalogs(&before, &c);
        self.project_catalogs
            .retain(|p| !p.name.eq_ignore_ascii_case(name));
        self.missing_catalogs
            .retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.project_catalogs.push(Arc::new(c));
        self.rebuild_library();
        changes
    }

    fn replace_library_catalog(
        &mut self,
        i: usize,
        mut c: GlassCatalog,
        stamp: Option<(std::time::SystemTime, u64)>,
    ) -> CatalogChanges {
        let lc = &mut self.library_catalogs[i];
        c.name = lc.catalog.name.clone();
        let changes = diff_catalogs(&lc.catalog, &c);
        lc.catalog = Arc::new(c);
        lc.stamp = stamp;
        self.rebuild_library();
        changes
    }

    // --- app-wide library

    /// Add (or replace, by catalog name) a catalog of the front end's
    /// app-wide library from a file. Library catalogs are not recorded in
    /// the project, except those the design uses, which `save` records.
    pub fn add_library_catalog_file(&mut self, path: &Path) -> EngineResult<CatalogSummary> {
        let format = CatalogFormat::from_path(path);
        let cat = read_catalog_file(path, format)?;
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let name = cat.name.clone();
        let stamp = file_stamp(&abs);
        self.put_library_catalog(LibraryCatalog {
            catalog: Arc::new(cat),
            path: Some(abs.display().to_string()),
            file_name: None,
            format,
            stamp,
        });
        self.rebuild_library();
        Ok(self.summary(&name))
    }

    /// `add_library_catalog_file` from the contents of a file named
    /// `file_name` (the web build).
    pub fn add_library_catalog_bytes(
        &mut self,
        file_name: &str,
        bytes: &[u8],
    ) -> EngineResult<CatalogSummary> {
        let path = Path::new(file_name);
        let format = CatalogFormat::from_path(path);
        let cat = parse_catalog_bytes(path, bytes, format)?;
        let name = cat.name.clone();
        self.put_library_catalog(LibraryCatalog {
            catalog: Arc::new(cat),
            path: None,
            file_name: Some(file_name.to_string()),
            format,
            stamp: None,
        });
        self.rebuild_library();
        Ok(self.summary(&name))
    }

    fn put_library_catalog(&mut self, c: LibraryCatalog) {
        match self
            .library_catalogs
            .iter()
            .position(|x| x.catalog.name.eq_ignore_ascii_case(&c.catalog.name))
        {
            Some(i) => self.library_catalogs[i] = c,
            None => self.library_catalogs.push(c),
        }
    }

    /// Remove a library catalog from this session. Returns whether there was one.
    pub fn remove_library_catalog(&mut self, name: &str) -> bool {
        let n = self.library_catalogs.len();
        self.library_catalogs
            .retain(|c| !c.catalog.name.eq_ignore_ascii_case(name));
        let removed = self.library_catalogs.len() != n;
        if removed {
            self.rebuild_library();
        }
        removed
    }

    /// Make the library catalogs exactly those of `paths`, in that order.
    /// A file already loaded and unchanged on disk (same modification time
    /// and size) is not read again; files that cannot be read, and a
    /// second catalog of a name already loaded, are reported and left out.
    /// Library catalogs given as bytes are kept.
    pub fn set_library_catalog_files(&mut self, paths: &[PathBuf]) -> LibrarySync {
        let mut next: Vec<LibraryCatalog> = Vec::new();
        let mut errors = Vec::new();
        for p in paths {
            let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
            let key = abs.display().to_string();
            let stamp = file_stamp(&abs);
            let old = self
                .library_catalogs
                .iter()
                .find(|c| c.path.as_deref() == Some(key.as_str()));
            let entry = match old {
                Some(old) if stamp.is_some() && old.stamp == stamp => old.clone(),
                _ => {
                    let format = CatalogFormat::from_path(&abs);
                    match read_catalog_file(&abs, format) {
                        Ok(c) => LibraryCatalog {
                            catalog: Arc::new(c),
                            path: Some(key.clone()),
                            file_name: None,
                            format,
                            stamp,
                        },
                        Err(e) => {
                            errors.push(LibraryError {
                                path: key,
                                message: e.to_string(),
                            });
                            continue;
                        }
                    }
                }
            };
            if let Some(dup) = next
                .iter()
                .find(|x| x.catalog.name.eq_ignore_ascii_case(&entry.catalog.name))
            {
                errors.push(LibraryError {
                    path: key,
                    message: format!(
                        "a catalog named '{}' is already loaded from {}",
                        entry.catalog.name,
                        dup.path.as_deref().unwrap_or("memory")
                    ),
                });
                continue;
            }
            next.push(entry);
        }
        for c in &self.library_catalogs {
            if c.path.is_none()
                && !next
                    .iter()
                    .any(|x| x.catalog.name.eq_ignore_ascii_case(&c.catalog.name))
            {
                next.push(c.clone());
            }
        }
        let before = self.library_hash.clone();
        self.library_catalogs = next;
        self.rebuild_library();
        let names: Vec<String> = self
            .library_catalogs
            .iter()
            .map(|c| c.catalog.name.clone())
            .collect();
        LibrarySync {
            catalogs: names.iter().map(|n| self.summary(n)).collect(),
            errors,
            changed: self.library_hash != before,
        }
    }

    /// The user's availability overrides (status, exclusion from
    /// substitution) applied to the library. Front-end preferences: not
    /// recorded in the project and not a design change.
    pub fn set_glass_overrides(&mut self, overrides: Vec<GlassOverride>) {
        if overrides != self.overrides {
            self.overrides = overrides;
            self.rebuild_library();
        }
    }

    pub fn glass_overrides(&self) -> &[GlassOverride] {
        &self.overrides
    }

    /// References to the library catalogs the design uses that the project
    /// does not record yet: catalogs a catalog material of the system
    /// (surfaces, coatings, configurations) resolves to. `save` and
    /// `save_to_string` add them to the project.
    pub fn library_catalogs_in_use(&self) -> Vec<CatalogRef> {
        let mut out: Vec<CatalogRef> = Vec::new();
        let mut refs = Vec::new();
        if let Ok(v) = serde_json::to_value(self.current.system()) {
            collect_catalog_materials(&v, &mut refs);
        }
        for (catalog, name) in refs {
            let Some((cat, _)) = self.library.locate_glass(catalog.as_deref(), &name) else {
                continue;
            };
            let known = |c: &CatalogRef| c.name.eq_ignore_ascii_case(&cat.name);
            if self.project.catalogs.iter().any(known) || out.iter().any(known) {
                continue;
            }
            if let Some(lc) = self
                .library_catalogs
                .iter()
                .find(|c| c.catalog.name.eq_ignore_ascii_case(&cat.name))
            {
                out.push(CatalogRef {
                    name: lc.catalog.name.clone(),
                    path: lc.path.clone(),
                    relative_path: lc.file_name.clone(),
                    format: lc.format,
                });
            }
        }
        out
    }

    fn summary(&self, name: &str) -> CatalogSummary {
        let found = self
            .library
            .catalogs()
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name));
        let library = self
            .library_catalogs
            .iter()
            .find(|c| c.catalog.name.eq_ignore_ascii_case(name));
        let reference = self
            .project
            .catalogs
            .iter()
            .find(|r| r.name.eq_ignore_ascii_case(name));
        let builtin = name == BUILTIN_CATALOG_NAME;
        CatalogSummary {
            name: found.map_or(name, |c| &c.name).to_string(),
            source: if builtin { "builtin" } else { "user" },
            path: reference
                .and_then(|r| r.path.clone())
                .or_else(|| library.and_then(|l| l.path.clone())),
            glass_count: found.map_or(0, |c| c.glasses.len()),
            library: library.is_some(),
            in_project: reference.is_some(),
        }
    }

    pub fn catalogs(&self) -> Vec<CatalogSummary> {
        self.library
            .catalogs()
            .iter()
            .map(|c| self.summary(&c.name))
            .collect()
    }

    /// Glasses of one catalog (case-insensitive name), or of all catalogs,
    /// as table rows with the overrides applied.
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
        let originals = self.effective_catalogs();
        Ok(cats
            .iter()
            .flat_map(|c| {
                let original = originals
                    .iter()
                    .find(|o| o.name.eq_ignore_ascii_case(&c.name));
                c.glasses.iter().map(move |(k, g)| {
                    glass_row(&c.name, g, original.and_then(|o| o.glasses.get(k)))
                })
            })
            .collect())
    }

    /// Everything about one glass (see `GlassDetails`), resolved as a
    /// material would be: the named catalog, or the first that has it.
    pub fn glass_details(&self, catalog: Option<&str>, name: &str) -> EngineResult<GlassDetails> {
        let (cat, g) = self.library.locate_glass(catalog, name).ok_or_else(|| {
            EngineError::Core(MkError::UnknownMaterial(match catalog {
                Some(c) => format!("{c}:{name}"),
                None => name.to_string(),
            }))
        })?;
        let original = self
            .original_catalog(&cat.name)
            .and_then(|o| o.glasses.get(&g.name).cloned())
            .unwrap_or_else(|| g.clone());
        Ok(glass_details(&cat.name, g, &original))
    }

    /// One catalog as loaded (before overrides), e.g. to export it.
    pub fn catalog(&self, name: &str) -> EngineResult<Arc<GlassCatalog>> {
        self.original_catalog(name).ok_or_else(|| {
            EngineError::Core(MkError::UnknownMaterial(format!(
                "no catalog named '{name}'"
            )))
        })
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

    /// Number of cached resolved systems.
    #[cfg(test)]
    fn resolved_cache_len(&self) -> usize {
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
    fn autosave_path(&self) -> Option<PathBuf> {
        self.path.as_ref().map(|p| {
            let mut s = p.clone().into_os_string();
            s.push(".autosave");
            PathBuf::from(s)
        })
    }

    /// Write the current project atomically to `path` (or, with `None`, to
    /// `<project file>.autosave` next to the project) without changing the
    /// session: its path, dirty flag and saved file stay as they are. The
    /// file is a normal project (open it to recover); catalog paths are
    /// recorded as absolute paths. Returns the path written.
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

    /// The backend for `req`. An `auto` request whose result the analysis
    /// graph already holds is answered from it and needs no route (nor any
    /// device): a stored answer costs nothing on either backend. A fresh
    /// request recomputes, so it is always routed.
    fn route(
        &self,
        req: &AnalysisRequest,
        analysis: &dyn crate::analysis::Analysis,
        settings: &Value,
        policy: &crate::analysis::RunPolicy,
        route: crate::route::RouteOptions,
    ) -> EngineResult<crate::backend::SelectedBackend> {
        if let (false, Some(id), Some(graph)) = (
            req.fresh,
            self.engine.auto_request_id(req.backend.as_ref()),
            self.engine
                .analysis_graph()
                .filter(|_| analysis.memoizable()),
        ) {
            let probe = AnalysisInput::new(
                self.current.clone(),
                self.library.clone(),
                crate::backend::BackendInfo::cpu_reference(),
            )
            .with_selection(id.clone(), None)
            .with_library_hash(self.library_hash.clone())
            .with_policy(*policy)
            .with_route(route);
            let key = crate::analysis::result_key(analysis, &probe, settings, &[]);
            if graph.contains(&key) {
                let mut s = crate::backend::SelectedBackend::cpu_reference();
                s.requested = id;
                return Ok(s);
            }
        }
        self.engine.route_backend(
            req.backend.as_ref(),
            &req.analysis,
            settings,
            &self.current,
            &self.library_hash,
            policy,
            &route,
            req.allow_backend_fallback,
        )
    }

    /// The run policy of a request: its tier, and its aim stop or the
    /// engine default. An explicit aiming tolerance in the system (any value
    /// but the model default) is honoured exactly, so the run is `requested`
    /// whatever the request or the engine default says (CONVENTIONS §7).
    fn policy(&self, req: &AnalysisRequest) -> crate::analysis::RunPolicy {
        let aim_stop = match self.current.system().ray_aiming.explicit_tolerance() {
            Some(_) => mklight_core::AimStopKind::Requested,
            None => req.aim_stop.unwrap_or(self.engine.config().aim_stop),
        };
        crate::analysis::RunPolicy {
            tier: req.tier,
            aim_stop,
        }
    }

    /// The route options of a request: its own, or the engine default;
    /// checked here, so an invalid value fails the request before it runs.
    fn route_options(&self, req: &AnalysisRequest) -> EngineResult<crate::route::RouteOptions> {
        let route = req.route.unwrap_or(self.engine.config().route);
        route.validate()?;
        Ok(route)
    }

    /// Run an analysis on the current snapshot in the calling thread.
    pub fn run_analysis(
        &self,
        req: &AnalysisRequest,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisResult> {
        let analysis = self.engine.analysis(&req.analysis)?;
        let policy = self.policy(req);
        let route = self.route_options(req)?;
        let settings = crate::analysis::prepare_tier(analysis.as_ref(), &req.settings, req.tier)?;
        let backend = self.route(req, analysis.as_ref(), &settings, &policy, route)?;
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
        .with_cost_decision(backend.cost_decision)
        .with_library_hash(self.library_hash.clone())
        .with_graph(self.engine.analysis_graph().cloned())
        .with_cost_model(Some(self.engine.cost_model().clone()))
        .with_policy(policy)
        .with_route(route)
        .with_fresh(req.fresh);
        run_analysis(
            analysis.as_ref(),
            &input,
            &settings,
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
        let policy = self.policy(req);
        let route = self.route_options(req)?;
        let settings = crate::analysis::prepare_tier(analysis.as_ref(), &req.settings, req.tier)?;
        let backend = self.route(req, analysis.as_ref(), &settings, &policy, route)?;
        let snap = self.current.clone();
        let lib = self.library.clone();
        let lib_hash = self.library_hash.clone();
        let cache = self.resolved.clone();
        let generation = self.library_generation;
        let graph = self.engine.analysis_graph().cloned();
        let cost_model = self.engine.cost_model().clone();
        let label = format!("{} @ {}", req.analysis, snap.hash().short());
        let supports_stop = analysis.supports_stop();
        let fresh = req.fresh;
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
            .with_cost_decision(backend.cost_decision)
            .with_library_hash(lib_hash)
            .with_graph(graph)
            .with_cost_model(Some(cost_model))
            .with_policy(policy)
            .with_route(route)
            .with_fresh(fresh);
            run_analysis(
                analysis.as_ref(),
                &input,
                &settings,
                backend.warning.into_iter().collect(),
                ctx,
            )
        });
        if supports_stop {
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
                agf: None,
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
            offset: Default::default(),
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

    /// A synthetic AGF catalog with two made-up glasses.
    const LIB_AGF: &str = "NM LIBCROWN 2 517642 1.5168 64.17 0 0 0\r\n\
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\n\
LD 0.3 2.5\r\n\
NM LIBFLINT 2 620364 1.62004 36.37 0 0 0\r\n\
CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r\n\
LD 0.32 2.5\r\n";

    fn use_glass(s: &mut Session, catalog: Option<&str>, name: &str) {
        let mut sys = s.system().clone();
        sys.surfaces[1].material = MaterialRef::Catalog {
            catalog: catalog.map(str::to_string),
            name: name.into(),
            offset: Default::default(),
        };
        s.set_system(sys).unwrap();
    }

    #[test]
    fn library_catalogs_load_without_being_recorded_until_used() {
        let dir = TempDir::new("library");
        let lib_path = dir.path().join("LIBCAT.agf");
        std::fs::write(&lib_path, LIB_AGF).unwrap();
        let e = engine();
        let mut s = e.new_session();
        let sync = s.set_library_catalog_files(std::slice::from_ref(&lib_path));
        assert!(sync.errors.is_empty() && sync.changed, "{sync:?}");
        assert_eq!(sync.catalogs[0].name, "LIBCAT");
        assert!(sync.catalogs[0].library && !sync.catalogs[0].in_project);
        assert!(s.project().catalogs.is_empty());
        assert!(!s.is_dirty());
        // Unchanged file: nothing reread, nothing changed.
        let again = s.set_library_catalog_files(std::slice::from_ref(&lib_path));
        assert!(!again.changed);
        assert!(s.library_catalogs_in_use().is_empty());

        // Used by the design: recorded on save, with a path relative to the
        // project when it lies in the project's folder.
        use_glass(&mut s, None, "LIBFLINT");
        let used = s.library_catalogs_in_use();
        assert_eq!(used.len(), 1);
        assert_eq!(used[0].name, "LIBCAT");
        let proj = dir.path().join("p.mklight");
        s.save(Some(&proj)).unwrap();
        let r = &s.project().catalogs[0];
        assert_eq!(r.name, "LIBCAT");
        assert_eq!(r.relative_path.as_deref(), Some("LIBCAT.agf"));
        assert!(s.catalogs().iter().any(|c| c.library && c.in_project));

        // Opened without the library, the project still finds the catalog.
        let s2 = e.open_session(&proj).unwrap();
        assert!(s2.open_warnings().is_empty(), "{:?}", s2.open_warnings());
        assert!(s2.resolved().is_ok());

        // An unreadable file and a duplicate name are reported, not fatal.
        let bad = dir.path().join("BAD.agf");
        std::fs::write(&bad, "nonsense").unwrap();
        let copy = dir.path().join("copy.json");
        let mut cat = read_catalog_file(&lib_path, CatalogFormat::Agf).unwrap();
        cat.name = "LIBCAT".into();
        std::fs::write(&copy, serde_json::to_string(&cat).unwrap()).unwrap();
        let mut s3 = e.new_session();
        let sync = s3.set_library_catalog_files(&[lib_path.clone(), bad, copy]);
        assert_eq!(sync.catalogs.len(), 1);
        assert_eq!(sync.errors.len(), 2, "{:?}", sync.errors);
        assert!(s3.remove_library_catalog("libcat"));
        assert_eq!(s3.catalogs().len(), 1);
    }

    #[test]
    fn a_missing_project_catalog_is_satisfied_by_the_library() {
        let dir = TempDir::new("missing");
        let lib_path = dir.path().join("LIBCAT.agf");
        std::fs::write(&lib_path, LIB_AGF).unwrap();
        let e = engine();
        let mut s = e.new_session();
        s.load_catalog_file(&lib_path).unwrap();
        use_glass(&mut s, Some("LIBCAT"), "LIBCROWN");
        let proj = dir.path().join("sub/p.mklight");
        std::fs::create_dir_all(proj.parent().unwrap()).unwrap();
        s.save(Some(&proj)).unwrap();
        let moved = dir.path().join("elsewhere.agf");
        std::fs::rename(&lib_path, &moved).unwrap();
        let mut s2 = e.open_session(&proj).unwrap();
        assert_eq!(s2.open_warnings().len(), 1);
        let mut renamed = dir.path().join("LIB2");
        std::fs::create_dir_all(&renamed).unwrap();
        renamed.push("LIBCAT.agf");
        std::fs::rename(&moved, &renamed).unwrap();
        s2.set_library_catalog_files(&[renamed]);
        assert!(s2.open_warnings().is_empty(), "{:?}", s2.open_warnings());
        assert!(s2.resolved().is_ok());
    }

    #[test]
    fn remove_restore_and_reload_report_changes() {
        let dir = TempDir::new("reload");
        let p = dir.path().join("LIBCAT.agf");
        std::fs::write(&p, LIB_AGF).unwrap();
        let e = engine();
        let mut s = e.new_session();
        s.load_catalog_file(&p).unwrap();
        let h1 = s.library_hash().to_string();
        let r = s.remove_catalog("libcat").unwrap();
        assert_eq!(r.name, "LIBCAT");
        assert!(s.project().catalogs.is_empty());
        assert_eq!(s.catalogs().len(), 1);
        assert!(s.remove_catalog("LIBCAT").is_err());
        assert!(s.remove_catalog(BUILTIN_CATALOG_NAME).is_err());
        let back = s.restore_catalog("LIBCAT").unwrap();
        assert!(back.in_project);
        assert_eq!(s.library_hash(), h1);
        assert_eq!(s.project().catalogs[0], r);

        // Edit the file: one glass changed, one removed, one added.
        let edited = LIB_AGF
            .replace("1.62004 36.37", "1.62010 36.40")
            .replace("LIBCROWN", "LIBNEW");
        std::fs::write(&p, edited).unwrap();
        let changes = s.reload_catalog("LIBCAT").unwrap();
        assert_eq!(changes.added, ["LIBNEW"]);
        assert_eq!(changes.removed, ["LIBCROWN"]);
        assert_eq!(changes.changed.len(), 1);
        let c = &changes.changed[0];
        assert_eq!(
            (c.name.as_str(), c.nd_before, c.nd_after),
            ("LIBFLINT", 1.62004, 1.6201)
        );
        assert_eq!((c.vd_before, c.vd_after), (36.37, 36.4));
        assert_ne!(s.library_hash(), h1);
        // Reloading again changes nothing.
        let none = s.reload_catalog("LIBCAT").unwrap();
        assert!(none.is_empty() && none.unchanged == 2);
        // The web form.
        let changes = s
            .reload_catalog_bytes("LIBCAT", LIB_AGF.as_bytes())
            .unwrap();
        assert_eq!(changes.added, ["LIBCROWN"]);
        assert!(s.reload_catalog("NOPE").is_err());

        // A library catalog reloads from its file too.
        std::fs::write(&p, LIB_AGF).unwrap();
        let mut s2 = e.new_session();
        s2.add_library_catalog_file(&p).unwrap();
        std::fs::write(&p, LIB_AGF.replace("LIBFLINT", "LIBOTHER")).unwrap();
        let changes = s2.reload_catalog("LIBCAT").unwrap();
        assert_eq!((changes.added.len(), changes.removed.len()), (1, 1));
        // Bytes-loaded library catalogs reload from bytes only.
        let mut s3 = e.new_session();
        s3.add_library_catalog_bytes("WEB.agf", LIB_AGF.as_bytes())
            .unwrap();
        assert!(s3.reload_catalog("WEB").is_err());
        assert!(s3.reload_catalog_bytes("WEB", LIB_AGF.as_bytes()).is_ok());
    }

    #[test]
    fn reloading_a_project_catalog_also_refreshes_the_library_copy() {
        let dir = TempDir::new("reload-both");
        let p = dir.path().join("LIBCAT.agf");
        std::fs::write(&p, LIB_AGF).unwrap();
        let e = engine();
        let mut s = e.new_session();
        s.add_library_catalog_file(&p).unwrap();
        s.load_catalog_file(&p).unwrap();
        std::fs::write(&p, LIB_AGF.replace("LIBCROWN", "LIBNEW")).unwrap();
        s.reload_catalog("LIBCAT").unwrap();
        s.remove_catalog("LIBCAT").unwrap();
        let names: Vec<String> = s
            .glasses(Some("LIBCAT"))
            .unwrap()
            .into_iter()
            .map(|g| g.name)
            .collect();
        assert_eq!(names, ["LIBFLINT", "LIBNEW"]);
    }

    #[test]
    fn overrides_reach_the_glass_table_glass_map_and_substitution() {
        use mklight_core::materials::agf::GlassStatus;
        let e = engine();
        let mut s = e.new_session();
        s.add_library_catalog_bytes("LIBCAT.agf", LIB_AGF.as_bytes())
            .unwrap();
        let h0 = s.library_hash().to_string();
        let o = GlassOverride {
            catalog: "LIBCAT".into(),
            name: "LIBFLINT".into(),
            status: Some(GlassStatus::Obsolete),
            exclude_substitution: Some(true),
        };
        s.set_glass_overrides(vec![o.clone()]);
        assert_ne!(s.library_hash(), h0);
        assert!(!s.is_dirty());
        let rows = s.glasses(Some("LIBCAT")).unwrap();
        let flint = rows.iter().find(|r| r.name == "LIBFLINT").unwrap();
        assert_eq!(flint.status, GlassStatus::Obsolete);
        assert!(flint.exclude_substitution && flint.overridden);
        let d = s.glass_details(None, "libflint").unwrap();
        assert_eq!(d.catalog_status, GlassStatus::Standard);
        assert_eq!(d.glass.substitution.status, GlassStatus::Standard);
        // The glass map reports the overridden availability by default.
        let r = s
            .run_analysis(
                &AnalysisRequest::new("glass_map", Value::Null),
                &JobContext::new(),
            )
            .unwrap();
        let v = r.json.clone();
        let g = v["glasses"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["name"] == "LIBFLINT")
            .unwrap();
        assert_eq!(g["status"], "obsolete");
        assert_eq!(g["exclude_substitution"], true);
        // Glass substitution's candidates leave the excluded glass out.
        use_glass(&mut s, Some("LIBCAT"), "LIBCROWN");
        let cands = mklight_optim::glass::CandidateSet {
            nearest: None,
            ..Default::default()
        };
        let sys = s.system().clone();
        let names: Vec<String> = cands
            .resolve(&sys, sys.surfaces[1].id, s.library())
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["LIBCROWN"]);
        s.set_glass_overrides(Vec::new());
        assert_eq!(s.library_hash(), h0);
        assert_eq!(s.glass_overrides(), &[] as &[GlassOverride]);
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
        /// An analysis that reports itself unavailable.
        struct Unavailable;
        impl crate::analysis::Analysis for Unavailable {
            fn name(&self) -> &str {
                "unavailable"
            }
            fn title(&self) -> &str {
                "Unavailable"
            }
            fn category(&self) -> &str {
                "test"
            }
            fn availability(&self) -> Result<(), String> {
                Err("not wired in".into())
            }
            fn run(
                &self,
                _: &AnalysisInput<'_>,
                _: &Value,
                _: &JobContext,
            ) -> EngineResult<crate::analysis::AnalysisOutput> {
                unreachable!("an unavailable analysis never runs")
            }
        }
        let e = engine();
        e.register_analysis(Arc::new(Unavailable));
        assert!(
            e.analyses()
                .iter()
                .any(|i| i.name == "unavailable" && !i.available && i.unavailable_reason.is_some())
        );
        let s = e.new_session();
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("nope", Value::Null)),
            Err(EngineError::UnknownAnalysis(_))
        ));
        assert!(matches!(
            s.submit_analysis(&AnalysisRequest::new("unavailable", Value::Null)),
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
        // A value of the wrong shape is an invalid argument naming the section.
        let e = s.set_project_section("views", json!(3)).unwrap_err();
        assert!(
            matches!(&e, EngineError::InvalidArgument(m) if m.contains("'views'")),
            "{e}"
        );
        assert!(s.save(None).is_err()); // never saved: needs a path
    }

    #[test]
    fn requests_reject_unknown_keys() {
        let full = AnalysisRequest {
            analysis: "validate".into(),
            settings: json!({}),
            view_key: Some("k".into()),
            backend: Some(Backend::Gpu("metal".into())),
            allow_backend_fallback: true,
            tier: mklight_core::Tier::Preview,
            aim_stop: Some(mklight_core::AimStopKind::Requested),
            route: Some(Default::default()),
            fresh: true,
        };
        // What the engine writes, it reads.
        let text = serde_json::to_string(&full).unwrap();
        assert_eq!(
            serde_json::from_str::<AnalysisRequest>(&text).unwrap(),
            full
        );
        assert_eq!(
            serde_json::from_str::<AnalysisRequest>(r#"{"analysis": "validate"}"#).unwrap(),
            AnalysisRequest::new("validate", Value::Null)
        );
        for (bad, key) in [
            (r#"{"analysis": "validate", "setings": {}}"#, "setings"),
            (
                r#"{"analysis": "validate", "backened": {"kind": "auto"}}"#,
                "backened",
            ),
            (
                r#"{"analysis": "validate", "route": {"fieldmap": "device-chief"}}"#,
                "fieldmap",
            ),
        ] {
            let e = serde_json::from_str::<AnalysisRequest>(bad)
                .unwrap_err()
                .to_string();
            assert!(e.contains(&format!("`{key}`")), "{bad}: {e}");
        }
    }

    #[test]
    fn only_analyses_that_support_stop_get_a_stop_handler() {
        let e = engine();
        let s = e.new_session();
        for info in e.analyses() {
            let analysis = e.analysis(&info.name).unwrap();
            assert_eq!(info.stoppable, analysis.supports_stop(), "{}", info.name);
        }
        let stoppable: Vec<String> = e
            .analyses()
            .into_iter()
            .filter(|a| a.stoppable)
            .map(|a| a.name)
            .collect();
        assert_eq!(stoppable, ["glass_substitution", "optimize", "tolerance"]);
        // A job of an analysis without stop support refuses `stop_job`.
        let job = s
            .submit_analysis(&AnalysisRequest::new(
                "rng_stream",
                json!({"count": 4, "delay_ms": 200}),
            ))
            .unwrap();
        assert!(matches!(
            e.stop_job(job),
            Err(EngineError::Core(MkError::Unsupported(_)))
        ));
        e.release_job(job).unwrap();
    }
}
