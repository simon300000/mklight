//! MKLight Python bindings (`import mklight`).
//!
//! A pyo3 extension module (`mklight._mklight`, re-exported by the pure
//! Python package in `python/mklight`) over the engine facade
//! `mklight-engine`, so Python, the C ABI and the CLI share one code path:
//! sessions, snapshots, parameter edits, catalogs, the analysis registry,
//! the job scheduler and backend selection. Low-level ray tracing goes
//! straight to the CPU f64 reference tracer `mklight-trace`.
//!
//! | module | contents |
//! |---|---|
//! | [`session`] | `Engine`, `Session` |
//! | [`job`] | `Job` (queued analysis), `Result` (finished analysis) |
//! | [`array`] | `Array`: read-only buffer-protocol arrays (zero-copy numpy) |
//! | [`lowlevel`] | `trace`, `trace_grid`, `paraxial` |
//! | [`wait`] | GIL-released waiting with Ctrl-C cancellation and progress |
//! | [`convert`] | JSON ↔ Python objects |
//! | [`errors`] | the exception hierarchy |
//!
//! Every computation releases the GIL. See `docs/PYTHON.md` for the API,
//! threading model and build instructions.

pub mod array;
pub mod convert;
pub mod errors;
pub mod job;
pub mod lowlevel;
pub mod session;
pub mod wait;

use pyo3::prelude::*;

use crate::convert::to_py;
use crate::errors::{ErrorKind, new_err};

/// Identifiers of the built-in sample systems (for `Engine.new(sample=...)`).
#[pyfunction]
fn samples() -> Vec<&'static str> {
    mklight_core::samples::all()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

/// The `OpticalSystem` dict of a built-in sample.
#[pyfunction]
fn sample<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let sys = session::sample_system(name).map_err(|m| new_err(ErrorKind::NotFound, m))?;
    let v = serde_json::to_value(&sys).map_err(|e| errors::core_err(e.into()))?;
    to_py(py, &v)
}

/// Return numpy arrays (default, when numpy is importable) or nested lists.
#[pyfunction]
fn set_numpy_enabled(enabled: bool) {
    array::set_numpy_enabled(enabled)
}

/// True when results are returned as numpy arrays.
#[pyfunction]
fn numpy_enabled(py: Python<'_>) -> bool {
    array::numpy(py).is_some()
}

#[pymodule]
fn _mklight(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", mklight_core::ENGINE_VERSION)?;
    m.add("ENGINE_VERSION", mklight_core::ENGINE_VERSION)?;
    m.add("TRACE_STATUS", lowlevel::TRACE_STATUS.to_vec())?;
    errors::register(m)?;
    m.add_class::<session::PyEngine>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<job::PyJob>()?;
    m.add_class::<job::PyAnalysisResult>()?;
    m.add_class::<array::Array>()?;
    m.add_function(wrap_pyfunction!(samples, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(set_numpy_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(lowlevel::trace, m)?)?;
    m.add_function(wrap_pyfunction!(lowlevel::trace_grid, m)?)?;
    m.add_function(wrap_pyfunction!(lowlevel::paraxial, m)?)?;
    Ok(())
}
