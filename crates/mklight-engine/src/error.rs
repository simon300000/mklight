use mklight_core::MkError;
use thiserror::Error;

/// Errors raised by the engine facade. Model and physics errors from
/// `mklight-core` are wrapped unchanged in `Core`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EngineError {
    #[error(transparent)]
    Core(#[from] MkError),
    #[error("unknown analysis '{0}'")]
    UnknownAnalysis(String),
    #[error("analysis '{name}' is not available yet: {reason}")]
    AnalysisUnavailable { name: String, reason: String },
    #[error("invalid settings: {0}")]
    InvalidSettings(String),
    #[error("unknown job {0}")]
    UnknownJob(u64),
    #[error("job {0} has not finished")]
    NotFinished(u64),
    #[error("timed out waiting for job {0}")]
    Timeout(u64),
    #[error("job failed: {0}")]
    JobFailed(String),
    #[error("project file: {0}")]
    Project(String),
    #[error("backend '{0}' is not available")]
    BackendUnavailable(String),
    #[error("engine is shutting down")]
    ShutDown,
}

impl EngineError {
    /// True when the error means "cancelled".
    pub fn is_cancelled(&self) -> bool {
        matches!(self, EngineError::Core(MkError::Cancelled))
    }
}

impl From<serde_json::Error> for EngineError {
    fn from(e: serde_json::Error) -> Self {
        EngineError::Core(MkError::Parse(e.to_string()))
    }
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Core(MkError::Io(e.to_string()))
    }
}

pub type EngineResult<T> = Result<T, EngineError>;
