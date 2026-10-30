//! The `.mklight` project file: UTF-8 JSON holding the optical system plus
//! everything a document needs around it (analysis views, user catalog
//! references, optimisation and tolerancing setup, metadata).
//!
//! Files carry a `schema_version`. Older files are upgraded on load by a chain
//! of migrations, one per version step (`MIGRATIONS[k]` turns version `k` into
//! `k + 1`), so each step stays small and testable. Files written by a newer
//! engine are refused rather than silently misread. Saves are atomic
//! (`fsutil::atomic_write`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

use mklight_core::{MkError, OpticalSystem};

use crate::error::{EngineError, EngineResult};
use crate::fsutil::atomic_write;
use crate::time::now_rfc3339;

/// Schema version written by this engine.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;
/// File-name extension of project files.
pub const PROJECT_FILE_EXTENSION: &str = "mklight";
/// Value of the top-level `format` key, so tools can recognise the file.
pub const PROJECT_FORMAT_ID: &str = "mklight-project";

fn format_id() -> String {
    PROJECT_FORMAT_ID.to_string()
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// RFC 3339 UTC timestamps.
    #[serde(default)]
    pub created: String,
    #[serde(default)]
    pub modified: String,
    /// Name and version of the application that last saved the file.
    #[serde(default)]
    pub app_name: String,
    #[serde(default)]
    pub app_version: String,
    /// `mklight_core::ENGINE_VERSION` at the last save.
    #[serde(default)]
    pub engine_version: String,
    #[serde(default)]
    pub author: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogFormat {
    /// AGF-format glass catalog text.
    #[default]
    Agf,
    /// `mklight_core::materials::GlassCatalog` serialised as JSON.
    Json,
}

impl CatalogFormat {
    /// Guess from a file extension (`.json` → Json, anything else → Agf).
    pub fn from_path(p: &Path) -> Self {
        match p.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("json") => CatalogFormat::Json,
            _ => CatalogFormat::Agf,
        }
    }
}

/// A user glass catalog the project depends on. Vendor catalogs are never
/// embedded (DECISIONS: glass data), only referenced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogRef {
    /// Catalog name as used in `MaterialRef::Catalog { catalog, .. }`.
    pub name: String,
    /// Absolute path when the catalog was loaded.
    #[serde(default)]
    pub path: Option<String>,
    /// Path relative to the project file's directory, tried first on load so
    /// that a project folder can be moved as a whole.
    #[serde(default)]
    pub relative_path: Option<String>,
    #[serde(default)]
    pub format: CatalogFormat,
}

impl CatalogRef {
    /// Candidate file locations, in the order they should be tried.
    pub fn candidates(&self, project_dir: Option<&Path>) -> Vec<PathBuf> {
        let mut v = Vec::new();
        if let (Some(dir), Some(rel)) = (project_dir, &self.relative_path) {
            v.push(dir.join(rel));
        }
        if let Some(p) = &self.path {
            v.push(PathBuf::from(p));
        }
        v
    }
}

/// A persisted analysis window: which analysis, with which settings. `key` is
/// the view key used for job supersession. `ui` is front-end state the engine
/// stores but never interprets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisView {
    pub key: String,
    pub analysis: String,
    #[serde(default)]
    pub settings: Value,
    #[serde(default)]
    pub ui: Value,
}

/// Contents of a `.mklight` file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "format_id")]
    pub format: String,
    pub schema_version: u32,
    #[serde(default)]
    pub metadata: ProjectMetadata,
    pub system: OpticalSystem,
    #[serde(default)]
    pub catalogs: Vec<CatalogRef>,
    #[serde(default)]
    pub views: Vec<AnalysisView>,
    /// Merit function. Opaque until `mklight-optim` defines its type.
    #[serde(default)]
    pub merit: Value,
    /// Optimisation variables. Opaque until `mklight-optim` defines its type.
    #[serde(default)]
    pub variables: Value,
    /// Tolerance set. Opaque until tolerancing defines its type.
    #[serde(default)]
    pub tolerances: Value,
    /// Unknown top-level keys, kept so that a round trip does not lose data
    /// added by a newer application at the same schema version.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// What happened while loading a file.
#[derive(Clone, Debug, PartialEq, Default, Serialize)]
pub struct MigrationReport {
    pub from_version: u32,
    pub to_version: u32,
    pub notes: Vec<String>,
}

impl MigrationReport {
    pub fn migrated(&self) -> bool {
        self.from_version != self.to_version
    }
}

/// One migration step on the raw JSON document. It may append notes for the
/// user (for example "converted a bare system file").
pub type Migration = fn(Value, &mut Vec<String>) -> Result<Value, String>;

/// `MIGRATIONS[k]` upgrades schema version `k` to `k + 1`. Its length is
/// therefore `PROJECT_SCHEMA_VERSION`.
pub const MIGRATIONS: &[Migration] = &[migrate_v0_to_v1];

/// Version 0 means files without `schema_version`: early drafts that were
/// either a bare `OpticalSystem` or `{ "system": … }` with optional extras.
fn migrate_v0_to_v1(doc: Value, notes: &mut Vec<String>) -> Result<Value, String> {
    let Value::Object(mut obj) = doc else {
        return Err("project document must be a JSON object".into());
    };
    if !obj.contains_key("system") {
        if obj.contains_key("surfaces") {
            notes.push("converted a bare optical-system file into a project".into());
            obj = Map::from_iter([("system".to_string(), Value::Object(obj))]);
        } else {
            return Err("document has neither 'system' nor 'surfaces'".into());
        }
    }
    obj.entry("format")
        .or_insert_with(|| json!(PROJECT_FORMAT_ID));
    obj.entry("metadata").or_insert_with(|| json!({}));
    Ok(Value::Object(obj))
}

fn read_version(doc: &Value) -> Result<u32, String> {
    match doc.get("schema_version") {
        None | Some(Value::Null) => Ok(0),
        Some(v) => v
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("schema_version must be a non-negative integer, got {v}")),
    }
}

/// Upgrade `doc` through `steps` to version `steps.len()`.
pub fn migrate_with(doc: Value, steps: &[Migration]) -> EngineResult<(Value, MigrationReport)> {
    let target = steps.len() as u32;
    let from = read_version(&doc).map_err(EngineError::Project)?;
    if from > target {
        return Err(EngineError::Project(format!(
            "file schema version {from} is newer than this engine supports ({target}); \
             please update MKLight"
        )));
    }
    let mut report = MigrationReport {
        from_version: from,
        to_version: target,
        notes: Vec::new(),
    };
    let mut doc = doc;
    for (k, step) in steps.iter().enumerate().skip(from as usize) {
        doc = step(doc, &mut report.notes)
            .map_err(|e| EngineError::Project(format!("migration v{k} → v{}: {e}", k + 1)))?;
        match &mut doc {
            Value::Object(o) => {
                o.insert("schema_version".into(), json!(k + 1));
            }
            _ => {
                return Err(EngineError::Project(format!(
                    "migration v{k} → v{} produced a non-object",
                    k + 1
                )));
            }
        }
    }
    Ok((doc, report))
}

/// Upgrade a raw document to `PROJECT_SCHEMA_VERSION`.
pub fn migrate(doc: Value) -> EngineResult<(Value, MigrationReport)> {
    migrate_with(doc, MIGRATIONS)
}

impl Project {
    /// A new project around `system`, stamped with the current time.
    pub fn new(system: OpticalSystem) -> Self {
        let now = now_rfc3339();
        Project {
            format: format_id(),
            schema_version: PROJECT_SCHEMA_VERSION,
            metadata: ProjectMetadata {
                created: now.clone(),
                modified: now,
                engine_version: mklight_core::ENGINE_VERSION.to_string(),
                ..Default::default()
            },
            system,
            catalogs: Vec::new(),
            views: Vec::new(),
            merit: Value::Null,
            variables: Value::Null,
            tolerances: Value::Null,
            extra: Map::new(),
        }
    }

    /// Parse (and migrate) a project from JSON text.
    pub fn from_json_str(s: &str) -> EngineResult<(Project, MigrationReport)> {
        let doc: Value = serde_json::from_str(s)?;
        Self::from_value(doc)
    }

    /// Parse (and migrate) a project from a JSON value.
    pub fn from_value(doc: Value) -> EngineResult<(Project, MigrationReport)> {
        let (doc, report) = migrate(doc)?;
        let mut p: Project = serde_json::from_value(doc)?;
        if p.format != PROJECT_FORMAT_ID {
            return Err(EngineError::Project(format!(
                "not an MKLight project (format '{}')",
                p.format
            )));
        }
        p.schema_version = PROJECT_SCHEMA_VERSION;
        Ok((p, report))
    }

    pub fn to_value(&self) -> EngineResult<Value> {
        Ok(serde_json::to_value(self)?)
    }

    /// Pretty-printed JSON with a trailing newline (diff-friendly).
    pub fn to_json_string(&self) -> EngineResult<String> {
        let mut s = serde_json::to_string_pretty(self)?;
        s.push('\n');
        Ok(s)
    }

    pub fn load(path: &Path) -> EngineResult<(Project, MigrationReport)> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            EngineError::Core(MkError::Io(format!("cannot read {}: {e}", path.display())))
        })?;
        Self::from_json_str(&text)
    }

    /// Write atomically. `metadata.modified` is left to the caller (the
    /// session stamps it).
    pub fn save(&self, path: &Path) -> EngineResult<()> {
        let text = self.to_json_string()?;
        atomic_write(path, text.as_bytes()).map_err(|e| {
            EngineError::Core(MkError::Io(format!("cannot write {}: {e}", path.display())))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil::test_util::TempDir;

    #[test]
    fn roundtrip_preserves_everything() {
        let mut p = Project::new(OpticalSystem::new_empty());
        p.views.push(AnalysisView {
            key: "summary-1".into(),
            analysis: "system_summary".into(),
            settings: json!({}),
            ui: json!({"pane": "left"}),
        });
        p.merit = json!({"operands": [1, 2, 3]});
        p.extra.insert("future_key".into(), json!({"x": 1}));
        let s = p.to_json_string().unwrap();
        let (q, rep) = Project::from_json_str(&s).unwrap();
        assert_eq!(p, q);
        assert!(!rep.migrated());
        // Infinite object distance survives as "inf".
        assert!(s.contains("\"inf\""));
    }

    #[test]
    fn v0_bare_system_is_migrated() {
        let bare = serde_json::to_value(OpticalSystem::new_empty()).unwrap();
        let (p, rep) = Project::from_value(bare).unwrap();
        assert_eq!(rep.from_version, 0);
        assert_eq!(rep.to_version, PROJECT_SCHEMA_VERSION);
        assert_eq!(rep.notes.len(), 1);
        assert_eq!(p.system, OpticalSystem::new_empty());
        assert_eq!(p.format, PROJECT_FORMAT_ID);
        assert_eq!(p.schema_version, 1);
    }

    #[test]
    fn v0_wrapped_system_is_migrated() {
        let doc = json!({"system": OpticalSystem::new_empty(), "merit": [1]});
        let (p, rep) = Project::from_value(doc).unwrap();
        assert!(rep.migrated() && rep.notes.is_empty());
        assert_eq!(p.merit, json!([1]));
    }

    #[test]
    fn newer_files_are_refused() {
        let mut v = Project::new(OpticalSystem::new_empty()).to_value().unwrap();
        v["schema_version"] = json!(PROJECT_SCHEMA_VERSION + 1);
        let e = Project::from_value(v).unwrap_err();
        assert!(e.to_string().contains("newer"), "{e}");
    }

    #[test]
    fn wrong_format_and_garbage_refused() {
        let mut v = Project::new(OpticalSystem::new_empty()).to_value().unwrap();
        v["format"] = json!("something-else");
        assert!(Project::from_value(v).is_err());
        assert!(Project::from_value(json!({"hello": 1})).is_err());
        assert!(Project::from_value(json!([1, 2])).is_err());
        assert!(Project::from_value(json!({"schema_version": -1, "system": {}})).is_err());
    }

    /// The framework runs steps in order from the file's version and stamps
    /// each intermediate version.
    #[test]
    fn migration_chain_order() {
        fn s0(mut d: Value, n: &mut Vec<String>) -> Result<Value, String> {
            d["trail"] = json!("0");
            n.push("s0".into());
            Ok(d)
        }
        fn s1(mut d: Value, n: &mut Vec<String>) -> Result<Value, String> {
            assert_eq!(d["schema_version"], json!(1));
            let t = d["trail"].as_str().unwrap_or("").to_string();
            d["trail"] = json!(t + "1");
            n.push("s1".into());
            Ok(d)
        }
        fn s2(mut d: Value, n: &mut Vec<String>) -> Result<Value, String> {
            let t = d["trail"].as_str().unwrap_or("").to_string();
            d["trail"] = json!(t + "2");
            n.push("s2".into());
            Ok(d)
        }
        fn fail(_: Value, _: &mut Vec<String>) -> Result<Value, String> {
            Err("boom".into())
        }
        let steps: &[Migration] = &[s0, s1, s2];
        let (d, r) = migrate_with(json!({}), steps).unwrap();
        assert_eq!(d["trail"], json!("012"));
        assert_eq!(d["schema_version"], json!(3));
        assert_eq!(r.notes, vec!["s0", "s1", "s2"]);
        let (d, r) = migrate_with(json!({"schema_version": 1, "trail": "x"}), steps).unwrap();
        assert_eq!(d["trail"], json!("x12"));
        assert_eq!(r.from_version, 1);
        let e = migrate_with(json!({"schema_version": 1}), &[s0, fail]).unwrap_err();
        assert!(e.to_string().contains("v1 → v2: boom"), "{e}");
    }

    #[test]
    fn save_and_load_file() {
        let dir = TempDir::new("project");
        let path = dir.path().join("lens.mklight");
        let p = Project::new(OpticalSystem::new_empty());
        p.save(&path).unwrap();
        let (q, _) = Project::load(&path).unwrap();
        assert_eq!(p, q);
        assert_eq!(dir.entries(), vec!["lens.mklight".to_string()]);
    }

    #[test]
    fn catalog_candidates_prefer_relative() {
        let r = CatalogRef {
            name: "MY".into(),
            path: Some("/abs/my.agf".into()),
            relative_path: Some("glass/my.agf".into()),
            format: CatalogFormat::Agf,
        };
        let c = r.candidates(Some(Path::new("/proj")));
        assert_eq!(
            c,
            vec![
                PathBuf::from("/proj/glass/my.agf"),
                PathBuf::from("/abs/my.agf")
            ]
        );
        assert_eq!(
            CatalogFormat::from_path(Path::new("x.JSON")),
            CatalogFormat::Json
        );
    }
}
