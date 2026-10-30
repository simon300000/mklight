use thiserror::Error;

/// Errors shared across the engine. Solver crates wrap or reuse these.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MkError {
    #[error("job cancelled")]
    Cancelled,
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("unknown material '{0}'")]
    UnknownMaterial(String),
    #[error("unknown coating '{0}'")]
    UnknownCoating(String),
    #[error(
        "wavelength {wavelength_um} µm outside valid range of '{material}' ({min_um}–{max_um} µm)"
    )]
    WavelengthOutOfRange {
        material: String,
        wavelength_um: f64,
        min_um: f64,
        max_um: f64,
    },
    #[error("invalid parameter path '{0}'")]
    InvalidParam(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("ray trace failure: {0}")]
    Trace(String),
    #[error("numerical failure: {0}")]
    Numerical(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("i/o error: {0}")]
    Io(String),
}

pub type MkResult<T> = Result<T, MkError>;

impl From<serde_json::Error> for MkError {
    fn from(e: serde_json::Error) -> Self {
        MkError::Parse(e.to_string())
    }
}

impl From<std::io::Error> for MkError {
    fn from(e: std::io::Error) -> Self {
        MkError::Io(e.to_string())
    }
}
