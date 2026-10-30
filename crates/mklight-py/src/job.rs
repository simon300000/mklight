//! `mklight.Job` (an analysis queued on the engine's worker pool) and
//! `mklight.Result` (a finished analysis).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use mklight_core::RunInfo;
use mklight_engine::{AnalysisResult, Buffer, Engine, EngineError, JobId, JobState};

use crate::array;
use crate::convert::{ser_to_py, to_py};
use crate::errors::{ErrorKind, engine_err, new_err};
use crate::wait::wait_job;

/// The Rust side of a finished analysis; buffers are shared with the
/// `Array`s (and numpy arrays) handed to Python.
pub struct ResultData {
    pub run_info: RunInfo,
    pub library_hash: String,
    pub json: Value,
    pub buffers: Vec<Arc<Buffer>>,
}

impl From<AnalysisResult> for ResultData {
    fn from(r: AnalysisResult) -> Self {
        ResultData {
            run_info: r.run_info,
            library_hash: r.library_hash,
            json: r.json,
            buffers: r.buffers.into_iter().map(Arc::new).collect(),
        }
    }
}

/// The result of an analysis run: `data` (the analysis' JSON result as a
/// dict), `run_info` (reproducibility metadata) and `arrays` (bulk data:
/// numpy arrays sharing the engine's memory, or lists without numpy).
#[pyclass(frozen, module = "mklight", name = "Result")]
pub struct PyAnalysisResult {
    data: Arc<ResultData>,
    data_obj: PyOnceLock<Py<PyAny>>,
    run_info_obj: PyOnceLock<Py<PyAny>>,
    arrays_obj: PyOnceLock<Py<PyAny>>,
}

impl PyAnalysisResult {
    pub fn new(data: Arc<ResultData>) -> Self {
        PyAnalysisResult {
            data,
            data_obj: PyOnceLock::new(),
            run_info_obj: PyOnceLock::new(),
            arrays_obj: PyOnceLock::new(),
        }
    }

    fn cached<'py>(
        py: Python<'py>,
        cell: &PyOnceLock<Py<PyAny>>,
        make: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cell.get_or_try_init(py, || make().map(Bound::unbind))
            .map(|o| o.bind(py).clone())
    }
}

#[pymethods]
impl PyAnalysisResult {
    /// The analysis-specific result (dict). Cached: the same dict is
    /// returned on every access.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::cached(py, &self.data_obj, || to_py(py, &self.data.json))
    }

    /// `RunInfo`: engine version, snapshot hash, analysis, normalised
    /// settings, backend, precision, seed, timing and warnings.
    #[getter]
    fn run_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::cached(py, &self.run_info_obj, || {
            ser_to_py(py, &self.data.run_info)
        })
    }

    /// Bulk arrays by name: numpy arrays (read-only, zero copy) when numpy
    /// is available, otherwise nested lists.
    #[getter]
    fn arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::cached(py, &self.arrays_obj, || {
            let d = PyDict::new(py);
            for b in &self.data.buffers {
                d.set_item(&b.name, array::export(py, b.clone())?)?;
            }
            Ok(d.into_any())
        })
    }

    /// The raw buffers as `mklight.Array` objects (buffer protocol).
    fn buffers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self
            .data
            .buffers
            .iter()
            .map(|b| Bound::new(py, array::Array::new(b.clone())))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    #[getter]
    fn analysis(&self) -> &str {
        &self.data.run_info.analysis
    }

    #[getter]
    fn snapshot_hash(&self) -> &str {
        &self.data.run_info.snapshot_hash
    }

    /// Hash of the glass library the run used (see `docs/FFI.md`).
    #[getter]
    fn library_hash(&self) -> &str {
        &self.data.library_hash
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.data.run_info.seed
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.data.run_info.warnings.clone()
    }

    /// The JSON envelope the C ABI returns (`run_info`, `library_hash`,
    /// `result`, `buffers` descriptors).
    fn envelope<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let descs: Vec<_> = self
            .data
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| b.desc(i))
            .collect();
        ser_to_py(
            py,
            &serde_json::json!({
                "run_info": self.data.run_info,
                "library_hash": self.data.library_hash,
                "result": self.data.json,
                "buffers": descs,
            }),
        )
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self.data(py)?.get_item(key)
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.data.buffers.iter().map(|b| b.name.as_str()).collect();
        format!(
            "<mklight.Result {} @ {} arrays={:?}>",
            self.data.run_info.analysis,
            &self.data.run_info.snapshot_hash[..self.data.run_info.snapshot_hash.len().min(12)],
            names
        )
    }
}

enum Slot {
    Pending,
    Taken {
        state: JobState,
        outcome: Result<Arc<ResultData>, EngineError>,
    },
}

/// An analysis queued on the engine's worker pool. Dropping a `Job` whose
/// result was never retrieved cancels and forgets it.
#[pyclass(frozen, module = "mklight", name = "Job")]
pub struct PyJob {
    engine: Engine,
    id: JobId,
    analysis: String,
    slot: Mutex<Slot>,
}

impl PyJob {
    pub fn new(engine: Engine, id: JobId, analysis: String) -> Self {
        PyJob {
            engine,
            id,
            analysis,
            slot: Mutex::new(Slot::Pending),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take the engine's outcome once (the job must be terminal).
    fn take(&self, state: JobState) -> Result<Arc<ResultData>, EngineError> {
        let mut g = self.lock();
        if let Slot::Pending = *g {
            let outcome = self
                .engine
                .take_result(self.id)
                .map(|r| Arc::new(ResultData::from(r)));
            *g = Slot::Taken { state, outcome };
        }
        match &*g {
            Slot::Taken { outcome, .. } => outcome.clone(),
            Slot::Pending => unreachable!(),
        }
    }

    /// Wait (interruptibly) and return the result.
    pub fn wait_result(
        &self,
        py: Python<'_>,
        timeout: Option<f64>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyAnalysisResult> {
        let taken = matches!(*self.lock(), Slot::Taken { .. });
        let state = if taken {
            JobState::Done
        } else {
            let deadline = match timeout {
                Some(t) if !(t >= 0.0 && t.is_finite()) => {
                    return Err(new_err(
                        ErrorKind::InvalidArgument,
                        "timeout must be a finite number of seconds >= 0",
                    ));
                }
                Some(t) => Instant::now().checked_add(Duration::from_secs_f64(t)),
                None => None,
            };
            wait_job(py, &self.engine, self.id, deadline, progress)?
        };
        match self.take(state) {
            Ok(d) => Ok(PyAnalysisResult::new(d)),
            Err(e) if e.is_cancelled() => Err(new_err(
                ErrorKind::Cancelled,
                format!("job {} ({}) was cancelled", self.id, self.analysis),
            )),
            Err(e) => Err(engine_err(e)),
        }
    }

    fn state_value(&self) -> JobState {
        match &*self.lock() {
            Slot::Taken { state, .. } => *state,
            Slot::Pending => self
                .engine
                .job_status(self.id)
                .map(|s| s.state)
                .unwrap_or(JobState::Cancelled),
        }
    }
}

fn state_name(s: JobState) -> String {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[pymethods]
impl PyJob {
    /// Engine job id.
    #[getter]
    fn id(&self) -> JobId {
        self.id
    }

    #[getter]
    fn analysis(&self) -> &str {
        &self.analysis
    }

    /// `queued`, `running`, `done`, `failed` or `cancelled`.
    #[getter]
    fn state(&self) -> String {
        state_name(self.state_value())
    }

    /// Progress fraction in `[0, 1]` (monotone).
    #[getter]
    fn progress(&self) -> f64 {
        match &*self.lock() {
            Slot::Taken { outcome, .. } => {
                if outcome.is_ok() {
                    1.0
                } else {
                    0.0
                }
            }
            Slot::Pending => self.engine.job_status(self.id).map_or(0.0, |s| s.progress),
        }
    }

    /// Current stage text.
    #[getter]
    fn stage(&self) -> String {
        self.engine
            .job_status(self.id)
            .map(|s| s.stage)
            .unwrap_or_default()
    }

    /// Full status dict (`id`, `state`, `progress`, `stage`, `label`,
    /// `view_key`, `error`, `superseded_by`, `elapsed_ms`), or `None` once
    /// the result has been retrieved.
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.engine.job_status(self.id) {
            Ok(s) => ser_to_py(py, &s).map(Some),
            Err(_) => Ok(None),
        }
    }

    /// True once the job has finished (done, failed or cancelled).
    fn done(&self) -> bool {
        self.state_value().is_terminal()
    }

    /// Request cancellation. A queued job is cancelled at once; a running
    /// one stops at its next check. `result()` then raises `CancelledError`.
    fn cancel(&self) -> PyResult<()> {
        if matches!(*self.lock(), Slot::Taken { .. }) {
            return Ok(());
        }
        match self.engine.cancel_job(self.id) {
            Ok(()) | Err(EngineError::UnknownJob(_)) => Ok(()),
            Err(e) => Err(engine_err(e)),
        }
    }

    /// Wait for the job; True if it finished within `timeout` seconds.
    #[pyo3(signature = (timeout=None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        if self.done() {
            return Ok(true);
        }
        let deadline = timeout
            .and_then(|t| Instant::now().checked_add(Duration::from_secs_f64(t.clamp(0.0, 1e9))));
        match wait_job(py, &self.engine, self.id, deadline, None) {
            Ok(_) => Ok(true),
            Err(e) if e.is_instance(py, &crate::errors::class_of(py, ErrorKind::Timeout)) => {
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Wait for the job (GIL released) and return its `Result`.
    /// Raises `CancelledError`, `JobFailedError` (or the specific error the
    /// analysis raised), or `JobTimeoutError` after `timeout` seconds (the
    /// job keeps running). Ctrl-C while waiting cancels the job.
    /// `progress(fraction, stage)` is called while waiting.
    #[pyo3(signature = (timeout=None, progress=None))]
    fn result(
        &self,
        py: Python<'_>,
        timeout: Option<f64>,
        progress: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyAnalysisResult> {
        self.wait_result(py, timeout, progress.as_ref())
    }

    fn __repr__(&self) -> String {
        format!(
            "<mklight.Job {} {} {}>",
            self.id,
            self.analysis,
            state_name(self.state_value())
        )
    }
}

impl Drop for PyJob {
    fn drop(&mut self) {
        if matches!(*self.lock(), Slot::Pending) {
            let _ = self.engine.release_job(self.id);
        }
    }
}
