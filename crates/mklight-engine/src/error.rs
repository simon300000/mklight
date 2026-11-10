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
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("engine is shutting down")]
    ShutDown,
}

/// What an error means to a caller, whatever its message. Every error has
/// exactly one kind; the front ends (C ABI status codes, Python exception
/// classes, CLI exit codes) map kinds, never error variants, so they agree
/// on the classification and differ only in how they group the kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The job was cancelled or superseded.
    Cancelled,
    /// The optical system fails validation.
    InvalidModel,
    /// A material or coating named by the model or by settings does not
    /// resolve.
    UnknownMaterial,
    /// A wavelength outside the range a material's data covers.
    WavelengthOutOfRange,
    /// An argument the call cannot take: a parameter path that addresses no
    /// parameter, an unknown backend name, input of the wrong shape.
    InvalidArgument,
    /// Analysis settings or route options the analysis does not accept.
    InvalidSettings,
    /// Malformed text: JSON, a glass catalog, a lens prescription.
    Parse,
    /// A file that is not a project of the current schema version.
    ProjectFile,
    /// A file could not be read or written.
    Io,
    /// No analysis of that name.
    UnknownAnalysis,
    /// No job of that id (never submitted, taken, released or dropped).
    UnknownJob,
    /// The analysis, the backend or a feature is not available.
    Unsupported,
    /// A ray-trace, numerical or compute-backend failure.
    Compute,
    /// The job has not finished.
    NotFinished,
    /// Waiting for a job timed out.
    Timeout,
    /// The job panicked.
    JobFailed,
    /// The engine is shutting down.
    ShutDown,
}

impl ErrorKind {
    /// The kind of a core error.
    pub fn of(e: &MkError) -> Self {
        match e {
            MkError::Cancelled => ErrorKind::Cancelled,
            MkError::InvalidModel(_) => ErrorKind::InvalidModel,
            MkError::UnknownMaterial(_) | MkError::UnknownCoating(_) => ErrorKind::UnknownMaterial,
            MkError::WavelengthOutOfRange { .. } => ErrorKind::WavelengthOutOfRange,
            MkError::InvalidParam(_) => ErrorKind::InvalidArgument,
            MkError::Parse(_) => ErrorKind::Parse,
            MkError::Trace(_) | MkError::Numerical(_) | MkError::Backend(_) => ErrorKind::Compute,
            MkError::Unsupported(_) => ErrorKind::Unsupported,
            MkError::Io(_) => ErrorKind::Io,
        }
    }
}

impl EngineError {
    /// True when the error means "cancelled".
    pub fn is_cancelled(&self) -> bool {
        matches!(self, EngineError::Core(MkError::Cancelled))
    }

    /// The error's classification (see [`ErrorKind`]).
    pub fn kind(&self) -> ErrorKind {
        match self {
            EngineError::Core(e) => ErrorKind::of(e),
            EngineError::UnknownAnalysis(_) => ErrorKind::UnknownAnalysis,
            EngineError::AnalysisUnavailable { .. } | EngineError::BackendUnavailable(_) => {
                ErrorKind::Unsupported
            }
            EngineError::InvalidSettings(_) => ErrorKind::InvalidSettings,
            EngineError::UnknownJob(_) => ErrorKind::UnknownJob,
            EngineError::NotFinished(_) => ErrorKind::NotFinished,
            EngineError::Timeout(_) => ErrorKind::Timeout,
            EngineError::JobFailed(_) => ErrorKind::JobFailed,
            EngineError::Project(_) => ErrorKind::ProjectFile,
            EngineError::InvalidArgument(_) => ErrorKind::InvalidArgument,
            EngineError::ShutDown => ErrorKind::ShutDown,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_has_one_kind() {
        let core = |e: MkError| EngineError::Core(e).kind();
        assert_eq!(core(MkError::Cancelled), ErrorKind::Cancelled);
        assert_eq!(
            core(MkError::InvalidModel("x".into())),
            ErrorKind::InvalidModel
        );
        assert_eq!(
            core(MkError::UnknownMaterial("x".into())),
            ErrorKind::UnknownMaterial
        );
        assert_eq!(
            core(MkError::UnknownCoating("x".into())),
            ErrorKind::UnknownMaterial
        );
        assert_eq!(
            core(MkError::WavelengthOutOfRange {
                material: "x".into(),
                wavelength_um: 3.0,
                min_um: 0.3,
                max_um: 2.5,
            }),
            ErrorKind::WavelengthOutOfRange
        );
        assert_eq!(
            core(MkError::InvalidParam("x".into())),
            ErrorKind::InvalidArgument
        );
        assert_eq!(core(MkError::Parse("x".into())), ErrorKind::Parse);
        for e in [
            MkError::Trace("x".into()),
            MkError::Numerical("x".into()),
            MkError::Backend("x".into()),
        ] {
            assert_eq!(core(e), ErrorKind::Compute);
        }
        assert_eq!(
            core(MkError::Unsupported("x".into())),
            ErrorKind::Unsupported
        );
        assert_eq!(core(MkError::Io("x".into())), ErrorKind::Io);

        let cases = [
            (
                EngineError::UnknownAnalysis("x".into()),
                ErrorKind::UnknownAnalysis,
            ),
            (
                EngineError::AnalysisUnavailable {
                    name: "x".into(),
                    reason: "y".into(),
                },
                ErrorKind::Unsupported,
            ),
            (
                EngineError::BackendUnavailable("x".into()),
                ErrorKind::Unsupported,
            ),
            (
                EngineError::InvalidSettings("x".into()),
                ErrorKind::InvalidSettings,
            ),
            (EngineError::UnknownJob(1), ErrorKind::UnknownJob),
            (EngineError::NotFinished(1), ErrorKind::NotFinished),
            (EngineError::Timeout(1), ErrorKind::Timeout),
            (EngineError::JobFailed("x".into()), ErrorKind::JobFailed),
            (EngineError::Project("x".into()), ErrorKind::ProjectFile),
            (
                EngineError::InvalidArgument("x".into()),
                ErrorKind::InvalidArgument,
            ),
            (EngineError::ShutDown, ErrorKind::ShutDown),
        ];
        for (e, k) in cases {
            assert_eq!(e.kind(), k, "{e}");
        }
        // Conversions keep the kind of what they wrap.
        let parse: EngineError = serde_json::from_str::<u32>("x").unwrap_err().into();
        assert_eq!(parse.kind(), ErrorKind::Parse);
        let io: EngineError = std::io::Error::other("x").into();
        assert_eq!(io.kind(), ErrorKind::Io);
    }
}
