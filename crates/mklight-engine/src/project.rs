//! The `.mklight` project file: UTF-8 JSON holding the optical system plus
//! everything a document needs around it (analysis views, user catalog
//! references, optimisation and tolerancing setup, metadata).
//!
//! Files carry a `schema_version`. Only `PROJECT_SCHEMA_VERSION` is read; a
//! file at any other version is refused rather than silently misread. Saves
//! are atomic (`fsutil::atomic_write`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use mklight_core::{MkError, OpticalSystem};

use crate::error::{EngineError, EngineResult};
use crate::fsutil::atomic_write;
use crate::time::now_rfc3339;

/// Schema version written by this engine.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;
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
    /// Merit-function setup of the front end; stored, never interpreted.
    #[serde(default)]
    pub merit: Value,
    /// Optimisation-variable setup of the front end; stored, never
    /// interpreted.
    #[serde(default)]
    pub variables: Value,
    /// Tolerance setup of the front end; stored, never interpreted.
    #[serde(default)]
    pub tolerances: Value,
    /// Unknown top-level keys, kept so that a round trip does not lose data
    /// added by a newer application at the same schema version.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Check the `schema_version` of a raw document before it is deserialised,
/// so a file at another version gets a clear error instead of a field error.
fn check_schema_version(doc: &Value) -> EngineResult<()> {
    let Value::Object(obj) = doc else {
        return Err(EngineError::Project(
            "project document must be a JSON object".into(),
        ));
    };
    let v = match obj.get("schema_version") {
        None | Some(Value::Null) => {
            return Err(EngineError::Project(
                "not an MKLight project (no schema_version)".into(),
            ));
        }
        Some(v) => v
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                EngineError::Project(format!(
                    "schema_version must be a non-negative integer, got {v}"
                ))
            })?,
    };
    if v > PROJECT_SCHEMA_VERSION {
        return Err(EngineError::Project(format!(
            "file schema version {v} is newer than this engine supports \
             ({PROJECT_SCHEMA_VERSION}); please update MKLight"
        )));
    }
    if v != PROJECT_SCHEMA_VERSION {
        return Err(EngineError::Project(format!(
            "file schema version {v} is not supported (this engine reads version \
             {PROJECT_SCHEMA_VERSION})"
        )));
    }
    Ok(())
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

    /// Parse a project from JSON text.
    pub fn from_json_str(s: &str) -> EngineResult<Project> {
        let doc: Value = serde_json::from_str(s)?;
        Self::from_value(doc)
    }

    /// Parse a project from a JSON value at `PROJECT_SCHEMA_VERSION`.
    pub fn from_value(doc: Value) -> EngineResult<Project> {
        check_schema_version(&doc)?;
        let p: Project = serde_json::from_value(doc)?;
        if p.format != PROJECT_FORMAT_ID {
            return Err(EngineError::Project(format!(
                "not an MKLight project (format '{}')",
                p.format
            )));
        }
        Ok(p)
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

    pub fn load(path: &Path) -> EngineResult<Project> {
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
    use serde_json::json;

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
        let q = Project::from_json_str(&s).unwrap();
        assert_eq!(p, q);
        // Infinite object distance survives as "inf".
        assert!(s.contains("\"inf\""));
    }

    #[test]
    fn files_without_a_schema_version_are_refused() {
        let bare = serde_json::to_value(OpticalSystem::new_empty()).unwrap();
        let e = Project::from_value(bare).unwrap_err();
        assert!(e.to_string().contains("no schema_version"), "{e}");
        let wrapped = json!({"system": OpticalSystem::new_empty()});
        assert!(Project::from_value(wrapped).is_err());
    }

    #[test]
    fn newer_files_are_refused() {
        let mut v = Project::new(OpticalSystem::new_empty()).to_value().unwrap();
        v["schema_version"] = json!(PROJECT_SCHEMA_VERSION + 1);
        let e = Project::from_value(v.clone()).unwrap_err();
        assert!(e.to_string().contains("newer"), "{e}");
        v["schema_version"] = json!(0);
        let e = Project::from_value(v).unwrap_err();
        assert!(e.to_string().contains("not supported"), "{e}");
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

    #[test]
    fn save_and_load_file() {
        let dir = TempDir::new("project");
        let path = dir.path().join("lens.mklight");
        let p = Project::new(OpticalSystem::new_empty());
        p.save(&path).unwrap();
        let q = Project::load(&path).unwrap();
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
