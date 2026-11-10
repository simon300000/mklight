//! Exit codes and the mapping from engine errors onto them (the same table
//! as `docs/CLI.md`).
//!
//! | code | meaning |
//! |---|---|
//! | 0 | success |
//! | 1 | usage: bad command line; unknown analysis, sample, catalog or parameter path (a path that does not address a parameter of the system); invalid settings JSON or values; index out of range |
//! | 2 | invalid model: the project cannot be parsed (including non-UTF-8 content and another `schema_version`); it fails validation; materials (also a glass named in analysis settings) or coatings do not resolve; an edit value the parameter cannot take, or an edit that would make the system invalid |
//! | 3 | analysis failed: the job failed, the analysis or backend is unavailable, a trace or numerical failure |
//! | 4 | I/O: a file could not be read or written |
//! | 130 | cancelled (Ctrl-C or `--timeout`) |

use std::fmt;

use mklight_core::MkError;
use mklight_engine::{EngineError, ErrorKind};

/// Process exit codes (see the module table and `docs/CLI.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Ok = 0,
    Usage = 1,
    InvalidModel = 2,
    AnalysisFailed = 3,
    Io = 4,
    Cancelled = 130,
}

impl Exit {
    pub fn code(self) -> i32 {
        self as i32
    }
    /// Stable snake_case name, used in `--json` error documents.
    pub fn kind(self) -> &'static str {
        match self {
            Exit::Ok => "ok",
            Exit::Usage => "usage",
            Exit::InvalidModel => "invalid_model",
            Exit::AnalysisFailed => "analysis_failed",
            Exit::Io => "io",
            Exit::Cancelled => "cancelled",
        }
    }
}

/// An error that ends the command with a message and an exit code.
#[derive(Debug, Clone, PartialEq)]
pub struct CliError {
    pub exit: Exit,
    pub message: String,
}

pub type CliResult<T> = Result<T, CliError>;

impl CliError {
    pub fn new(exit: Exit, message: impl Into<String>) -> Self {
        CliError {
            exit,
            message: message.into(),
        }
    }
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(Exit::Usage, message)
    }
    pub fn model(message: impl Into<String>) -> Self {
        Self::new(Exit::InvalidModel, message)
    }
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(Exit::Io, message)
    }
    /// Prefix the message with context (`"<ctx>: <message>"`).
    pub fn context(mut self, ctx: impl fmt::Display) -> Self {
        self.message = format!("{ctx}: {}", self.message);
        self
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Exit code of each error kind (the module table). The grouping is the
/// CLI's own: what the user must correct decides the code, so a glass or a
/// project file that does not resolve is "invalid model" and an unknown
/// analysis is "usage".
pub fn exit_of_kind(kind: ErrorKind) -> Exit {
    match kind {
        ErrorKind::Cancelled => Exit::Cancelled,
        ErrorKind::InvalidModel
        | ErrorKind::UnknownMaterial
        | ErrorKind::WavelengthOutOfRange
        | ErrorKind::ProjectFile => Exit::InvalidModel,
        ErrorKind::InvalidArgument
        | ErrorKind::InvalidSettings
        | ErrorKind::Parse
        | ErrorKind::UnknownAnalysis => Exit::Usage,
        ErrorKind::Io => Exit::Io,
        ErrorKind::Unsupported
        | ErrorKind::Compute
        | ErrorKind::UnknownJob
        | ErrorKind::NotFinished
        | ErrorKind::Timeout
        | ErrorKind::JobFailed
        | ErrorKind::ShutDown => Exit::AnalysisFailed,
    }
}

/// Exit code of an engine error.
pub fn exit_of(e: &EngineError) -> Exit {
    exit_of_kind(e.kind())
}

impl From<EngineError> for CliError {
    fn from(e: EngineError) -> Self {
        CliError::new(exit_of(&e), e.to_string())
    }
}

impl From<MkError> for CliError {
    fn from(e: MkError) -> Self {
        CliError::new(exit_of_kind(ErrorKind::of(&e)), e.to_string())
    }
}

/// Errors from opening a project file: a parse error there means the file
/// is not a valid project (exit 2), not a command-line mistake.
pub fn open_error(path: &std::path::Path, e: EngineError) -> CliError {
    let exit = match e.kind() {
        ErrorKind::Parse => Exit::InvalidModel,
        kind => exit_of_kind(kind),
    };
    CliError::new(exit, format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable() {
        assert_eq!(Exit::Ok.code(), 0);
        assert_eq!(Exit::Usage.code(), 1);
        assert_eq!(Exit::InvalidModel.code(), 2);
        assert_eq!(Exit::AnalysisFailed.code(), 3);
        assert_eq!(Exit::Io.code(), 4);
        assert_eq!(Exit::Cancelled.code(), 130);
    }

    #[test]
    fn engine_errors_map_to_documented_codes() {
        let c = |e: EngineError| exit_of(&e);
        assert_eq!(c(EngineError::Core(MkError::Cancelled)), Exit::Cancelled);
        assert_eq!(
            c(EngineError::Core(MkError::UnknownMaterial("X".into()))),
            Exit::InvalidModel
        );
        assert_eq!(c(EngineError::UnknownAnalysis("x".into())), Exit::Usage);
        assert_eq!(c(EngineError::InvalidSettings("x".into())), Exit::Usage);
        assert_eq!(c(EngineError::JobFailed("x".into())), Exit::AnalysisFailed);
        assert_eq!(c(EngineError::Project("x".into())), Exit::InvalidModel);
        assert_eq!(c(EngineError::Core(MkError::Io("x".into()))), Exit::Io);
        // The groups the C ABI and Python draw differently (docs/CLI.md).
        assert_eq!(
            c(EngineError::Core(MkError::UnknownCoating("X".into()))),
            Exit::InvalidModel
        );
        assert_eq!(
            c(EngineError::Core(MkError::WavelengthOutOfRange {
                material: "X".into(),
                wavelength_um: 3.0,
                min_um: 0.3,
                max_um: 2.5,
            })),
            Exit::InvalidModel
        );
        assert_eq!(
            c(EngineError::Core(MkError::InvalidParam("x".into()))),
            Exit::Usage
        );
        assert_eq!(c(EngineError::InvalidArgument("x".into())), Exit::Usage);
        assert_eq!(
            c(EngineError::Core(MkError::Parse("x".into()))),
            Exit::Usage
        );
        for e in [
            EngineError::Core(MkError::Unsupported("x".into())),
            EngineError::Core(MkError::Numerical("x".into())),
            EngineError::BackendUnavailable("gpu:x".into()),
            EngineError::AnalysisUnavailable {
                name: "x".into(),
                reason: "y".into(),
            },
            EngineError::UnknownJob(1),
            EngineError::Timeout(1),
        ] {
            assert_eq!(c(e.clone()), Exit::AnalysisFailed, "{e}");
        }
        assert_eq!(
            CliError::from(MkError::UnknownMaterial("X".into())).exit,
            Exit::InvalidModel
        );
        let p = std::path::Path::new("a.mklight");
        let e = open_error(p, EngineError::Core(MkError::Parse("bad".into())));
        assert_eq!(e.exit, Exit::InvalidModel);
        assert!(e.message.starts_with("a.mklight: "));
    }
}
