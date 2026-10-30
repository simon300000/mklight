//! Running and waiting with the GIL released while staying responsive to
//! Ctrl-C and progress callbacks.
//!
//! The calling thread never blocks with the GIL held: it waits in slices of
//! [`POLL`] with the GIL released, then re-attaches briefly to run
//! `PyErr_CheckSignals` (so `KeyboardInterrupt` is raised promptly) and to
//! call the user's progress callback. Any Python exception raised there
//! (Ctrl-C, or an exception from the callback) cancels the computation
//! before it propagates.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;

use mklight_core::{JobContext, MkResult};
use mklight_engine::{Engine, EngineError, JobId, JobState};

use crate::errors::{ErrorKind, engine_err, new_err};

/// Wait slice between signal checks and progress updates.
pub const POLL: Duration = Duration::from_millis(20);

/// Calls a Python `progress(fraction, stage)` callback with monotone
/// fractions, only when the fraction or the stage changed.
pub struct ProgressReporter<'a, 'py> {
    cb: Option<&'a Bound<'py, PyAny>>,
    last: f64,
    stage: String,
    called: bool,
}

impl<'a, 'py> ProgressReporter<'a, 'py> {
    pub fn new(cb: Option<&'a Bound<'py, PyAny>>) -> Self {
        ProgressReporter {
            cb,
            last: 0.0,
            stage: String::new(),
            called: false,
        }
    }

    /// Report `f` (clamped to `[last, 1]`). Exceptions from the callback
    /// propagate.
    pub fn report(&mut self, f: f64, stage: &str) -> PyResult<()> {
        let Some(cb) = self.cb else { return Ok(()) };
        let f = monotone(self.last, f);
        if self.called && f == self.last && stage == self.stage {
            return Ok(());
        }
        self.last = f;
        self.called = true;
        stage.clone_into(&mut self.stage);
        cb.call1((f, stage))?;
        Ok(())
    }

    /// Final report of 1.0 (sent unless 1.0 was already reported).
    pub fn finish(&mut self) -> PyResult<()> {
        let stage = self.stage.clone();
        self.report(1.0, &stage)
    }
}

/// `max(last, f)` clamped to `[0, 1]`; non-finite input keeps `last`.
pub fn monotone(last: f64, f: f64) -> f64 {
    if f.is_finite() {
        f.clamp(0.0, 1.0).max(last)
    } else {
        last
    }
}

/// Run `work` on a helper thread with the GIL released. `KeyboardInterrupt`
/// (or an exception from `progress`) cancels `ctx` and is re-raised once the
/// work has stopped.
pub fn run_interruptible<T, F>(
    py: Python<'_>,
    ctx: &JobContext,
    progress: Option<&Bound<'_, PyAny>>,
    work: F,
) -> PyResult<T>
where
    T: Send,
    F: FnOnce(&JobContext) -> MkResult<T> + Send,
{
    let mut reporter = ProgressReporter::new(progress);
    let done = Done::default();
    std::thread::scope(|s| {
        let worker_ctx = ctx.clone();
        let done_ref = &done;
        let handle = s.spawn(move || {
            // Set the flag even if `work` panics.
            let _set = SetOnDrop(done_ref);
            work(&worker_ctx)
        });
        let mut interrupted: Option<PyErr> = None;
        while !py.detach(|| done.wait(Some(POLL))) {
            let r = py
                .check_signals()
                .and_then(|()| reporter.report(ctx.progress.fraction(), &ctx.progress.stage()));
            if let Err(e) = r {
                ctx.cancel.cancel();
                interrupted = Some(e);
                break;
            }
        }
        // Join with the GIL released (after a cancel the worker stops at
        // its next check).
        let result = py.detach(|| handle.join());
        if let Some(e) = interrupted {
            return Err(e);
        }
        match result {
            Ok(Ok(v)) => {
                reporter.finish()?;
                Ok(v)
            }
            Ok(Err(e)) => Err(crate::errors::core_err(e)),
            Err(_) => Err(new_err(ErrorKind::Internal, "the computation panicked")),
        }
    })
}

/// A one-shot completion flag.
#[derive(Default)]
struct Done {
    flag: Mutex<bool>,
    cv: Condvar,
}

struct SetOnDrop<'a>(&'a Done);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.set();
    }
}

impl Done {
    fn set(&self) {
        *self.flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.cv.notify_all();
    }
    /// Wait up to `timeout` (`None`: forever); returns the flag.
    fn wait(&self, timeout: Option<Duration>) -> bool {
        let g = self.flag.lock().unwrap_or_else(|e| e.into_inner());
        match timeout {
            None => *self
                .cv
                .wait_while(g, |f| !*f)
                .unwrap_or_else(|e| e.into_inner()),
            Some(t) => {
                *self
                    .cv
                    .wait_timeout_while(g, t, |f| !*f)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
        }
    }
}

/// Wait for an engine job, GIL released, until it finishes or `deadline`.
/// Returns the terminal state. `KeyboardInterrupt` (or an exception from
/// `progress`) cancels the job and propagates; a deadline raises
/// `JobTimeoutError` and leaves the job running.
pub fn wait_job(
    py: Python<'_>,
    engine: &Engine,
    id: JobId,
    deadline: Option<Instant>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<JobState> {
    let mut reporter = ProgressReporter::new(progress);
    loop {
        let slice = match deadline {
            Some(d) => d.saturating_duration_since(Instant::now()).min(POLL),
            None => POLL,
        };
        let waited = py.detach(|| engine.wait_job(id, Some(slice)));
        let state = match waited {
            Ok(s) => Some(s),
            Err(EngineError::Timeout(_)) => None,
            Err(e) => return Err(engine_err(e)),
        };
        let status = engine.job_status(id).ok();
        let checked = py.check_signals().and_then(|()| match &status {
            Some(st) if st.state != JobState::Queued => reporter.report(st.progress, &st.stage),
            _ => Ok(()),
        });
        if let Err(e) = checked {
            let _ = engine.cancel_job(id);
            return Err(e);
        }
        if let Some(s) = state {
            if s == JobState::Done {
                reporter.finish()?;
            }
            return Ok(s);
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(new_err(
                ErrorKind::Timeout,
                format!("timed out waiting for job {id}"),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_never_goes_back() {
        let mut last = 0.0;
        for (f, want) in [
            (0.2, 0.2),
            (0.1, 0.2),
            (f64::NAN, 0.2),
            (0.7, 0.7),
            (3.0, 1.0),
            (-1.0, 1.0),
        ] {
            last = monotone(last, f);
            assert_eq!(last, want);
        }
    }
}
