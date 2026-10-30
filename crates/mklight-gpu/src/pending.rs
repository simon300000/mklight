//! One-shot completion handle for asynchronous backend work.
//!
//! A backend returns a [`Pending`] from `submit` and keeps the matching
//! [`Completer`] and [`JobControl`]. The result can be awaited (it implements
//! `Future`), waited on from a blocking thread, or polled. Dropping a
//! `Completer` without completing it resolves the `Pending` with a backend
//! error, so a waiter can never hang on a crashed worker; [`spawn_job`] also
//! turns a worker panic into an error that names the job.
//!
//! Cancellation is per submission. Each submission has its own token, and the
//! worker stops when either that token or the caller's `JobContext` token is
//! cancelled. [`Pending::cancel`] (and dropping an unfinished `Pending`)
//! cancels only that submission, so sibling jobs that share a `JobContext`
//! keep running, while cancelling the context still stops all of them.

use std::any::Any;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use mklight_core::{CancelToken, JobContext, MkError, MkResult};

struct State<T> {
    result: Option<MkResult<T>>,
    waker: Option<Waker>,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    cv: Condvar,
}

/// The receiving side: the eventual result of a submitted job. Dropping it
/// before the job finishes cancels the job (a superseded request stops
/// occupying the device).
pub struct Pending<T> {
    shared: Arc<Shared<T>>,
    cancel: CancelToken,
}

/// The sending side, owned by the backend's worker.
pub struct Completer<T> {
    shared: Option<Arc<Shared<T>>>,
}

/// What a backend worker consults between command buffers: the submission's
/// own cancel token, the caller's context (cancellation and progress).
#[derive(Clone, Debug)]
pub struct JobControl {
    own: CancelToken,
    ctx: JobContext,
}

impl JobControl {
    /// `Err(Cancelled)` once this submission or the caller's context has been
    /// cancelled.
    #[inline]
    pub fn check(&self) -> MkResult<()> {
        self.own.check()?;
        self.ctx.check()
    }
    pub fn is_cancelled(&self) -> bool {
        self.own.is_cancelled() || self.ctx.cancel.is_cancelled()
    }
    /// Progress `f` in 0…1 within the caller's context range.
    pub fn report(&self, f: f64, stage: &str) {
        self.ctx.report(f, stage);
    }
    pub fn context(&self) -> &JobContext {
        &self.ctx
    }
}

/// A connected `(Pending, Completer, JobControl)` for one submission under
/// the caller's `ctx`.
pub fn pending<T>(ctx: &JobContext) -> (Pending<T>, Completer<T>, JobControl) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            result: None,
            waker: None,
        }),
        cv: Condvar::new(),
    });
    let own = CancelToken::new();
    (
        Pending {
            shared: shared.clone(),
            cancel: own.clone(),
        },
        Completer {
            shared: Some(shared),
        },
        JobControl {
            own,
            ctx: ctx.clone(),
        },
    )
}

fn panic_message(p: &(dyn Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

/// Run `work` on a new named thread and complete `completer` with its result.
/// A panic becomes `MkError::Backend` naming `what` and the panic message.
pub fn spawn_job<T: Send + 'static>(
    thread_name: &str,
    what: String,
    completer: Completer<T>,
    work: impl FnOnce() -> MkResult<T> + Send + 'static,
) -> MkResult<()> {
    let err_what = what.clone();
    std::thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            let r = catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|p| {
                Err(MkError::Backend(format!(
                    "{what}: backend worker panicked: {}",
                    panic_message(p.as_ref())
                )))
            });
            completer.complete(r);
        })
        .map(|_| ())
        .map_err(|e| MkError::Backend(format!("{err_what}: cannot start a worker thread: {e}")))
}

impl<T> Completer<T> {
    pub fn complete(mut self, r: MkResult<T>) {
        self.finish(r);
    }
    fn finish(&mut self, r: MkResult<T>) {
        if let Some(sh) = self.shared.take() {
            let waker = {
                let mut st = sh.state.lock().unwrap_or_else(|e| e.into_inner());
                st.result = Some(r);
                st.waker.take()
            };
            sh.cv.notify_all();
            if let Some(w) = waker {
                w.wake();
            }
        }
    }
}

impl<T> Drop for Completer<T> {
    fn drop(&mut self) {
        self.finish(Err(MkError::Backend(
            "backend worker ended without a result".into(),
        )));
    }
}

impl<T> Pending<T> {
    /// A `Pending` that is already resolved (for synchronous backends).
    pub fn ready(r: MkResult<T>) -> Self {
        let (p, c, _) = pending(&JobContext::new());
        c.complete(r);
        p
    }
    /// Cancel this submission only. The worker stops at its next check
    /// (between command buffers) and the result becomes `Err(Cancelled)`;
    /// other jobs submitted under the same `JobContext` are unaffected.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
    pub fn is_ready(&self) -> bool {
        self.lock().result.is_some()
    }
    /// Block until the result is available.
    pub fn wait(self) -> MkResult<T> {
        let mut st = self.lock();
        loop {
            if let Some(r) = st.result.take() {
                return r;
            }
            st = self.shared.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
    /// Block for at most `timeout`; `None` if the job is still running.
    pub fn wait_timeout(&mut self, timeout: Duration) -> Option<MkResult<T>> {
        let st = self.lock();
        let (mut st, _) = self
            .shared
            .cv
            .wait_timeout_while(st, timeout, |s| s.result.is_none())
            .unwrap_or_else(|e| e.into_inner());
        st.result.take()
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        // Nobody can receive the result any more. After completion this is
        // a no-op for the worker.
        self.cancel.cancel();
    }
}

impl<T> Future for Pending<T> {
    type Output = MkResult<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut st = self.lock();
        match st.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                st.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Wake;

    #[test]
    fn completes_across_threads() {
        let (p, c, _) = pending::<u32>(&JobContext::new());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            c.complete(Ok(7));
        });
        assert_eq!(p.wait(), Ok(7));
    }

    #[test]
    fn dropped_completer_is_an_error() {
        let (p, c, _) = pending::<u32>(&JobContext::new());
        drop(c);
        assert!(matches!(p.wait(), Err(MkError::Backend(_))));
    }

    #[test]
    fn timeout_and_future() {
        let (mut p, c, _) = pending::<u32>(&JobContext::new());
        assert!(p.wait_timeout(Duration::from_millis(1)).is_none());
        struct Flag(std::sync::atomic::AtomicBool);
        impl Wake for Flag {
            fn wake(self: Arc<Self>) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let flag = Arc::new(Flag(false.into()));
        let waker = Waker::from(flag.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut p).poll(&mut cx).is_pending());
        c.complete(Ok(3));
        assert!(flag.0.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(Pin::new(&mut p).poll(&mut cx), Poll::Ready(Ok(3)));
    }

    /// Cancelling one submission leaves its siblings and the shared context
    /// alone; cancelling the context reaches every submission.
    #[test]
    fn cancellation_is_per_submission() {
        let ctx = JobContext::new();
        let (p1, _c1, w1) = pending::<u32>(&ctx);
        let (_p2, _c2, w2) = pending::<u32>(&ctx);
        p1.cancel();
        assert_eq!(w1.check(), Err(MkError::Cancelled));
        assert_eq!(w2.check(), Ok(()));
        assert!(!ctx.cancel.is_cancelled());
        ctx.cancel.cancel();
        assert_eq!(w2.check(), Err(MkError::Cancelled));
    }

    #[test]
    fn dropping_pending_cancels_its_job() {
        let ctx = JobContext::new();
        let (p, _c, w) = pending::<u32>(&ctx);
        assert!(!w.is_cancelled());
        drop(p);
        assert!(w.is_cancelled());
        assert!(!ctx.cancel.is_cancelled());
    }

    #[test]
    fn worker_panic_names_the_job() {
        let (p, c, _) = pending::<u32>(&JobContext::new());
        spawn_job("test-worker", "SagProbe job".into(), c, || {
            let v: Vec<u32> = Vec::new();
            Ok(v[3])
        })
        .unwrap();
        let Err(MkError::Backend(m)) = p.wait() else {
            panic!()
        };
        assert!(
            m.starts_with("SagProbe job: backend worker panicked: index out of bounds"),
            "{m}"
        );
    }
}
