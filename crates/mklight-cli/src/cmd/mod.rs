//! Command implementations. Each command returns a `Report`: a JSON document
//! (printed with `--json`) and its human-readable rendering, plus the exit
//! code. Every command goes through the `mklight-engine` facade (sessions,
//! projects, analysis registry, job scheduler), the same code path as the
//! app and the Python bindings; `trace` and `paraxial` additionally call the
//! `mklight-trace` reference directly on the session's resolved system.

pub mod analyses;
pub mod catalogs;
pub mod info;
pub mod project;
pub mod trace;

use serde_json::Value;
use std::path::Path;

use mklight_core::JobContext;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, Session};

use crate::cli::GlobalOpts;
use crate::error::{CliError, CliResult, Exit, open_error};

/// What a command produced.
pub struct Report {
    pub json: Value,
    pub text: String,
    pub exit: Exit,
    /// Print `text` on stderr instead of stdout (partial status after a
    /// cancellation, so stdout carries only results).
    pub text_to_stderr: bool,
}

impl Report {
    pub fn ok(json: Value, text: String) -> Self {
        Report {
            json,
            text,
            exit: Exit::Ok,
            text_to_stderr: false,
        }
    }
    pub fn with_exit(mut self, exit: Exit) -> Self {
        self.exit = exit;
        self
    }
}

/// Shared state of one invocation.
pub struct Ctx {
    pub global: GlobalOpts,
    pub engine: Engine,
}

impl Ctx {
    pub fn new(global: GlobalOpts) -> Self {
        let engine = Engine::new(EngineConfig {
            app_name: "mklight-cli".into(),
            app_version: env!("CARGO_PKG_VERSION").into(),
            ..Default::default()
        });
        Ctx { global, engine }
    }

    /// Open a project file (migrating older schema versions in memory).
    pub fn open(&self, path: &Path) -> CliResult<Session> {
        if !path.exists() {
            return Err(CliError::io(format!("{}: no such file", path.display())));
        }
        self.engine
            .open_session(path)
            .map_err(|e| open_error(path, e))
    }
}

/// Save a session (to its own path, or `to`), mapping failures to exit 4.
pub fn save(session: &mut Session, to: Option<&Path>) -> CliResult<()> {
    session.save(to).map_err(|e| {
        let target = to.or(session.path()).map(|p| p.display().to_string());
        CliError::io(format!("cannot save {}: {e}", target.unwrap_or_default()))
    })
}

/// Run an analysis synchronously in this thread (same `run_analysis` path
/// the scheduler uses) and return its result JSON.
pub fn run_sync(session: &Session, analysis: &str) -> CliResult<Value> {
    let r = session.run_analysis(
        &AnalysisRequest::new(analysis, Value::Null),
        &JobContext::new(),
    )?;
    Ok(r.json)
}

/// Lines for a warning list (`"warning: …"`), empty when there are none.
pub fn warning_lines(warnings: &[String]) -> String {
    warnings.iter().map(|w| format!("warning: {w}\n")).collect()
}

/// Session warnings raised while opening (missing catalogs, migration notes).
pub fn open_warnings(session: &Session) -> Vec<String> {
    session.open_warnings().to_vec()
}
