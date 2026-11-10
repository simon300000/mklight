//! The Python exception hierarchy and the mapping from engine errors.
//!
//! ```text
//! MKLightError(Exception)
//! ├── InvalidArgumentError(MKLightError, ValueError)
//! │   └── InvalidSettingsError
//! ├── ParseError(MKLightError, ValueError)
//! ├── InvalidModelError(MKLightError, ValueError)
//! ├── NotFoundError(MKLightError, LookupError)
//! ├── FileError(MKLightError, OSError)
//! ├── UnsupportedError(MKLightError, NotImplementedError)
//! ├── CancelledError(MKLightError)
//! ├── JobTimeoutError(MKLightError, TimeoutError)
//! ├── JobFailedError(MKLightError)
//! ├── ComputeError(MKLightError)
//! └── InternalError(MKLightError)
//! ```
//!
//! The classes are created once (with multiple inheritance, so `except
//! ValueError` keeps working) and stored in a `PyOnceLock`. An engine error
//! is raised as the class of its `mklight_engine::ErrorKind`, the
//! classification the C ABI and the CLI map too.

use pyo3::exceptions::{
    PyException, PyLookupError, PyNotImplementedError, PyOSError, PyTimeoutError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

use mklight_core::MkError;
use mklight_engine::EngineError;

/// Error classes, one per exception type of the hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    InvalidArgument,
    InvalidSettings,
    Parse,
    InvalidModel,
    NotFound,
    File,
    Unsupported,
    Cancelled,
    Timeout,
    JobFailed,
    Compute,
    Internal,
}

/// Every kind, in creation order (parents before children).
pub const ALL_KINDS: [ErrorKind; 12] = [
    ErrorKind::InvalidArgument,
    ErrorKind::InvalidSettings,
    ErrorKind::Parse,
    ErrorKind::InvalidModel,
    ErrorKind::NotFound,
    ErrorKind::File,
    ErrorKind::Unsupported,
    ErrorKind::Cancelled,
    ErrorKind::Timeout,
    ErrorKind::JobFailed,
    ErrorKind::Compute,
    ErrorKind::Internal,
];

impl ErrorKind {
    pub fn class_name(self) -> &'static str {
        match self {
            ErrorKind::InvalidArgument => "InvalidArgumentError",
            ErrorKind::InvalidSettings => "InvalidSettingsError",
            ErrorKind::Parse => "ParseError",
            ErrorKind::InvalidModel => "InvalidModelError",
            ErrorKind::NotFound => "NotFoundError",
            ErrorKind::File => "FileError",
            ErrorKind::Unsupported => "UnsupportedError",
            ErrorKind::Cancelled => "CancelledError",
            ErrorKind::Timeout => "JobTimeoutError",
            ErrorKind::JobFailed => "JobFailedError",
            ErrorKind::Compute => "ComputeError",
            ErrorKind::Internal => "InternalError",
        }
    }

    fn doc(self) -> &'static str {
        match self {
            ErrorKind::InvalidArgument => "An argument, parameter edit or setting is invalid.",
            ErrorKind::InvalidSettings => "Analysis settings fail the analysis' settings schema.",
            ErrorKind::Parse => "Malformed JSON, project file or glass catalog.",
            ErrorKind::InvalidModel => "The optical system fails validation.",
            ErrorKind::NotFound => "Unknown analysis, job, glass, coating or catalog.",
            ErrorKind::File => "A file-system error.",
            ErrorKind::Unsupported => "Analysis or backend not available in this build.",
            ErrorKind::Cancelled => "The job was cancelled, superseded or interrupted.",
            ErrorKind::Timeout => "Waiting for a job timed out (the job keeps running).",
            ErrorKind::JobFailed => "A job ended with an error.",
            ErrorKind::Compute => "Ray trace, numerical or compute-backend failure.",
            ErrorKind::Internal => "An internal engine error.",
        }
    }

    fn parent(self) -> Option<ErrorKind> {
        match self {
            ErrorKind::InvalidSettings => Some(ErrorKind::InvalidArgument),
            _ => None,
        }
    }

    fn builtin_base(self, py: Python<'_>) -> Option<Bound<'_, PyType>> {
        Some(match self {
            ErrorKind::InvalidArgument | ErrorKind::Parse | ErrorKind::InvalidModel => {
                py.get_type::<PyValueError>()
            }
            ErrorKind::NotFound => py.get_type::<PyLookupError>(),
            ErrorKind::File => py.get_type::<PyOSError>(),
            ErrorKind::Unsupported => py.get_type::<PyNotImplementedError>(),
            ErrorKind::Timeout => py.get_type::<PyTimeoutError>(),
            _ => return None,
        })
    }
}

/// The exception class of each engine error kind (the hierarchy of
/// `docs/PYTHON.md`).
pub fn class_kind(kind: mklight_engine::ErrorKind) -> ErrorKind {
    use mklight_engine::ErrorKind as Engine;
    match kind {
        Engine::Cancelled => ErrorKind::Cancelled,
        Engine::InvalidModel => ErrorKind::InvalidModel,
        Engine::UnknownMaterial | Engine::UnknownAnalysis | Engine::UnknownJob => {
            ErrorKind::NotFound
        }
        Engine::WavelengthOutOfRange | Engine::InvalidArgument => ErrorKind::InvalidArgument,
        Engine::InvalidSettings => ErrorKind::InvalidSettings,
        Engine::Parse | Engine::ProjectFile => ErrorKind::Parse,
        Engine::Io => ErrorKind::File,
        Engine::Unsupported => ErrorKind::Unsupported,
        Engine::Compute => ErrorKind::Compute,
        Engine::Timeout => ErrorKind::Timeout,
        Engine::JobFailed => ErrorKind::JobFailed,
        // `Job.result()` waits, so no caller sees an unfinished job.
        Engine::NotFinished | Engine::ShutDown => ErrorKind::Internal,
    }
}

struct Classes {
    base: Py<PyType>,
    kinds: Vec<(ErrorKind, Py<PyType>)>,
}

static CLASSES: PyOnceLock<Classes> = PyOnceLock::new();

fn make_class<'py>(
    py: Python<'py>,
    name: &str,
    doc: &str,
    bases: Vec<Bound<'py, PyType>>,
) -> PyResult<Py<PyType>> {
    let ns = PyDict::new(py);
    ns.set_item("__module__", "mklight")?;
    ns.set_item("__doc__", doc)?;
    let bases = PyTuple::new(py, bases)?;
    let cls = py.get_type::<PyType>().call1((name, bases, ns))?;
    Ok(cls.cast_into::<PyType>()?.unbind())
}

fn classes(py: Python<'_>) -> &Classes {
    CLASSES.get_or_init(py, || {
        let base = make_class(
            py,
            "MKLightError",
            "Base class of every MKLight error.",
            vec![py.get_type::<PyException>()],
        )
        .expect("creating MKLightError");
        let mut kinds: Vec<(ErrorKind, Py<PyType>)> = Vec::new();
        for k in ALL_KINDS {
            let first = match k.parent() {
                Some(p) => kinds
                    .iter()
                    .find(|(q, _)| *q == p)
                    .map(|(_, c)| c.bind(py).clone())
                    .expect("parent kinds come first"),
                None => base.bind(py).clone(),
            };
            let mut bases = vec![first];
            if k.parent().is_none()
                && let Some(b) = k.builtin_base(py)
            {
                bases.push(b);
            }
            let cls = make_class(py, k.class_name(), k.doc(), bases).expect("creating error");
            kinds.push((k, cls));
        }
        Classes { base, kinds }
    })
}

/// The `MKLightError` base class.
pub fn base_class(py: Python<'_>) -> Bound<'_, PyType> {
    classes(py).base.bind(py).clone()
}

/// The class of one kind.
pub fn class_of(py: Python<'_>, kind: ErrorKind) -> Bound<'_, PyType> {
    classes(py)
        .kinds
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, c)| c.bind(py).clone())
        .expect("every kind has a class")
}

/// A Python exception of `kind` with `msg`.
pub fn new_err(kind: ErrorKind, msg: impl Into<String>) -> PyErr {
    let msg = msg.into();
    Python::attach(|py| match class_of(py, kind).call1((msg.clone(),)) {
        Ok(v) => PyErr::from_value(v),
        Err(e) => e,
    })
}

/// Convert an engine error into the matching Python exception.
pub fn engine_err(e: EngineError) -> PyErr {
    new_err(class_kind(e.kind()), e.to_string())
}

/// Convert a core error into the matching Python exception.
pub fn core_err(e: MkError) -> PyErr {
    new_err(class_kind(mklight_engine::ErrorKind::of(&e)), e.to_string())
}

/// Add the hierarchy to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("MKLightError", base_class(py))?;
    for k in ALL_KINDS {
        m.add(k.class_name(), class_of(py, k))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_the_documented_classes() {
        let engine_kind = |e: &EngineError| class_kind(e.kind());
        let cases = [
            (EngineError::Core(MkError::Cancelled), ErrorKind::Cancelled),
            (
                EngineError::Core(MkError::InvalidModel("x".into())),
                ErrorKind::InvalidModel,
            ),
            (
                EngineError::Core(MkError::UnknownMaterial("x".into())),
                ErrorKind::NotFound,
            ),
            (EngineError::Core(MkError::Io("x".into())), ErrorKind::File),
            (
                EngineError::Core(MkError::Trace("x".into())),
                ErrorKind::Compute,
            ),
            (
                EngineError::UnknownAnalysis("x".into()),
                ErrorKind::NotFound,
            ),
            (
                EngineError::AnalysisUnavailable {
                    name: "mtf".into(),
                    reason: "later".into(),
                },
                ErrorKind::Unsupported,
            ),
            (
                EngineError::InvalidSettings("x".into()),
                ErrorKind::InvalidSettings,
            ),
            (EngineError::Timeout(3), ErrorKind::Timeout),
            (EngineError::JobFailed("x".into()), ErrorKind::JobFailed),
            (EngineError::Project("x".into()), ErrorKind::Parse),
            (
                EngineError::InvalidArgument("x".into()),
                ErrorKind::InvalidArgument,
            ),
            (EngineError::UnknownJob(3), ErrorKind::NotFound),
            (
                EngineError::BackendUnavailable("gpu:x".into()),
                ErrorKind::Unsupported,
            ),
            (EngineError::NotFinished(3), ErrorKind::Internal),
            (EngineError::ShutDown, ErrorKind::Internal),
            (
                EngineError::Core(MkError::InvalidParam("x".into())),
                ErrorKind::InvalidArgument,
            ),
            (
                EngineError::Core(MkError::Parse("x".into())),
                ErrorKind::Parse,
            ),
            (
                EngineError::Core(MkError::Unsupported("x".into())),
                ErrorKind::Unsupported,
            ),
        ];
        for (e, k) in cases {
            assert_eq!(engine_kind(&e), k, "{e}");
        }
    }

    #[test]
    fn class_names_are_unique() {
        let mut names: Vec<_> = ALL_KINDS.iter().map(|k| k.class_name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL_KINDS.len());
        assert!(names.iter().all(|n| n.ends_with("Error")));
    }
}
