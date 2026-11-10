//! `mklight.Job` (an analysis queued on the engine's worker pool) and
//! `mklight.Result` (a finished analysis).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use mklight_core::RunInfo;
use mklight_engine::{AnalysisResult, BackendReport, Buffer, Engine, EngineError, JobId, JobState};

use crate::array;
use crate::convert::{ser_to_py, to_py};
use crate::errors::{ErrorKind, engine_err, new_err};
use crate::wait::wait_job;

/// The Rust side of a finished analysis; buffers are shared with the
/// `Array`s (and numpy arrays) handed to Python.
pub struct ResultData {
    pub run_info: RunInfo,
    pub backend: BackendReport,
    pub result_version: u32,
    pub library_hash: String,
    pub derived_snapshot_hash: Option<String>,
    pub json: Value,
    pub buffers: Vec<Arc<Buffer>>,
}

impl From<AnalysisResult> for ResultData {
    fn from(r: AnalysisResult) -> Self {
        ResultData {
            run_info: r.run_info,
            backend: r.backend,
            result_version: r.result_version,
            library_hash: r.library_hash,
            derived_snapshot_hash: r.derived_snapshot_hash,
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

    /// The C ABI envelope: RunInfo, library hash, backend report, result
    /// version, result JSON and buffer descriptors.
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
                "derived_snapshot_hash": self.data.derived_snapshot_hash,
                "backend": self.data.backend,
                "result_version": self.data.result_version,
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
        /// Last progress seen before the outcome was taken (1 on success),
        /// so `Job.progress` stays monotone.
        progress: f64,
    },
    /// The engine dropped the job before its outcome was collected (only
    /// possible with an explicit `Engine(max_retained_jobs=...)`).
    Evicted {
        progress: f64,
    },
}

/// Python-visible job state: the engine's, plus `evicted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyJobState {
    Engine(JobState),
    Evicted,
}

impl PyJobState {
    fn name(self) -> String {
        match self {
            PyJobState::Engine(s) => state_name(s).into(),
            PyJobState::Evicted => "evicted".into(),
        }
    }
    fn is_terminal(self) -> bool {
        match self {
            PyJobState::Engine(s) => s.is_terminal(),
            PyJobState::Evicted => true,
        }
    }
}

/// A wait deadline from a Python `timeout` in seconds: `None` waits
/// forever, and so does a timeout too large to represent (including
/// `inf`), as the scheduler's own `wait` does. NaN and negative values are
/// rejected.
pub fn deadline(timeout: Option<f64>) -> PyResult<Option<Instant>> {
    match timeout {
        None => Ok(None),
        Some(t) if t.is_nan() || t < 0.0 => Err(new_err(
            ErrorKind::InvalidArgument,
            format!("timeout must be a number of seconds >= 0 (or None), got {t}"),
        )),
        Some(t) => Ok(Duration::try_from_secs_f64(t)
            .ok()
            .and_then(|d| Instant::now().checked_add(d))),
    }
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

    fn evicted_err(&self) -> PyErr {
        new_err(
            ErrorKind::NotFound,
            format!(
                "the outcome of job {} ({}) was discarded by the engine before it was \
                 collected: more than max_retained_jobs={} finished jobs were waiting \
                 (leave max_retained_jobs unset to keep every result until collected)",
                self.id,
                self.analysis,
                self.engine.config().max_retained_jobs
            ),
        )
    }

    /// Mark the job evicted if the engine no longer knows it while the
    /// slot is still pending. Returns whether the job is (now) evicted.
    fn check_evicted(&self) -> bool {
        let mut g = self.lock();
        match &*g {
            Slot::Evicted { .. } => true,
            Slot::Taken { .. } => false,
            Slot::Pending => {
                if let Err(EngineError::UnknownJob(_)) = self.engine.job_status(self.id) {
                    *g = Slot::Evicted { progress: 0.0 };
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Take the engine's outcome once (the job must be terminal). `None`
    /// when the engine evicted the job first.
    fn take(&self, state: JobState) -> Option<Result<Arc<ResultData>, EngineError>> {
        let mut g = self.lock();
        if let Slot::Pending = *g {
            let progress = self.engine.job_status(self.id).map_or(0.0, |s| s.progress);
            *g = match self.engine.take_result(self.id) {
                Err(EngineError::UnknownJob(_)) => Slot::Evicted { progress },
                outcome => {
                    let outcome = outcome.map(|r| Arc::new(ResultData::from(r)));
                    let progress = if outcome.is_ok() { 1.0 } else { progress };
                    Slot::Taken {
                        state,
                        outcome,
                        progress,
                    }
                }
            };
        }
        match &*g {
            Slot::Taken { outcome, .. } => Some(outcome.clone()),
            Slot::Evicted { .. } => None,
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
        let state = match &*self.lock() {
            Slot::Taken { state, .. } => Some(*state),
            Slot::Evicted { .. } => return Err(self.evicted_err()),
            Slot::Pending => None,
        };
        let state = match state {
            Some(s) => s,
            None => {
                let deadline = deadline(timeout)?;
                match wait_job(py, &self.engine, self.id, deadline, progress) {
                    Ok(s) => s,
                    Err(e) if self.check_evicted() => {
                        drop(e);
                        return Err(self.evicted_err());
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        match self.take(state) {
            None => Err(self.evicted_err()),
            Some(Ok(d)) => Ok(PyAnalysisResult::new(d)),
            Some(Err(e)) if e.is_cancelled() => Err(new_err(
                ErrorKind::Cancelled,
                format!("job {} ({}) was cancelled", self.id, self.analysis),
            )),
            Some(Err(e)) => Err(engine_err(e)),
        }
    }

    fn state_value(&self) -> PyJobState {
        if let Slot::Taken { state, .. } = &*self.lock() {
            return PyJobState::Engine(*state);
        }
        match self.engine.job_status(self.id) {
            Ok(s) => PyJobState::Engine(s.state),
            Err(_) if self.check_evicted() => PyJobState::Evicted,
            // Taken by another thread in between.
            Err(_) => match &*self.lock() {
                Slot::Taken { state, .. } => PyJobState::Engine(*state),
                _ => PyJobState::Evicted,
            },
        }
    }
}

/// The state's name, as the engine serialises it.
fn state_name(s: JobState) -> &'static str {
    match s {
        JobState::Queued => "queued",
        JobState::Running => "running",
        JobState::Done => "done",
        JobState::Failed => "failed",
        JobState::Cancelled => "cancelled",
    }
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

    /// `queued`, `running`, `done`, `failed` or `cancelled`; `evicted` if
    /// the engine discarded the outcome before it was collected (only with
    /// an explicit `Engine(max_retained_jobs=...)`).
    #[getter]
    fn state(&self) -> String {
        self.state_value().name()
    }

    /// Progress fraction in `[0, 1]` (monotone; 1 once a result was
    /// retrieved, the last value seen for a failed or cancelled job).
    #[getter]
    fn progress(&self) -> f64 {
        let g = self.lock();
        match &*g {
            Slot::Taken { progress, .. } | Slot::Evicted { progress } => *progress,
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

    /// Stop optimize/tolerance, retaining the best-so-far result (stopped=True).
    /// Other active jobs raise UnsupportedError; finished jobs are unchanged.
    fn stop(&self) -> PyResult<()> {
        if !matches!(*self.lock(), Slot::Pending) {
            return Ok(());
        }
        self.engine.stop_job(self.id).map_err(engine_err)
    }

    /// Request cancellation. A queued job is cancelled at once; a running
    /// one stops at its next check. `result()` then raises `CancelledError`.
    fn cancel(&self) -> PyResult<()> {
        if !matches!(*self.lock(), Slot::Pending) {
            return Ok(());
        }
        match self.engine.cancel_job(self.id) {
            Ok(()) | Err(EngineError::UnknownJob(_)) => Ok(()),
            Err(e) => Err(engine_err(e)),
        }
    }

    /// Wait for the job; True if it finished within `timeout` seconds
    /// (`None` or `inf`: wait forever; NaN and negative values raise
    /// `InvalidArgumentError`).
    #[pyo3(signature = (timeout=None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let deadline = deadline(timeout)?;
        if self.done() {
            return Ok(true);
        }
        match wait_job(py, &self.engine, self.id, deadline, None) {
            Ok(_) => Ok(true),
            Err(e) if e.is_instance(py, &crate::errors::class_of(py, ErrorKind::Timeout)) => {
                Ok(false)
            }
            Err(_) if self.check_evicted() => Ok(true),
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
            self.state_value().name()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeouts_map_to_deadlines() {
        pyo3::Python::initialize();
        Python::attach(|_| {
            assert_eq!(deadline(None).unwrap(), None);
            // Too large to represent (or infinite): wait forever, no panic.
            assert_eq!(deadline(Some(1e20)).unwrap(), None);
            assert_eq!(deadline(Some(f64::INFINITY)).unwrap(), None);
            let d = deadline(Some(0.5)).unwrap().unwrap();
            assert!(d > Instant::now() && d <= Instant::now() + Duration::from_secs(1));
            assert!(deadline(Some(f64::NAN)).is_err());
            assert!(deadline(Some(-1.0)).is_err());
        });
    }
}
