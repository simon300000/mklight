//! One-shot completion handle for asynchronous backend work.
//!
//! A backend returns a [`Pending`] from `submit` and keeps the matching
//! [`Completer`]. The result can be awaited (it implements `Future`), waited
//! on from a blocking thread, or polled. Dropping a `Completer` without
//! completing it resolves the `Pending` with a backend error, so a waiter can
//! never hang on a crashed worker.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use mklight_core::{CancelToken, MkError, MkResult};

struct State<T> {
    result: Option<MkResult<T>>,
    waker: Option<Waker>,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    cv: Condvar,
}

/// The receiving side: the eventual result of a submitted job.
pub struct Pending<T> {
    shared: Arc<Shared<T>>,
    cancel: CancelToken,
}

/// The sending side, owned by the backend's worker.
pub struct Completer<T> {
    shared: Option<Arc<Shared<T>>>,
}

/// A connected `(Pending, Completer)` pair. `cancel` is the job's token, so
/// `Pending::cancel` reaches the worker's cancellation checks.
pub fn pending<T>(cancel: CancelToken) -> (Pending<T>, Completer<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            result: None,
            waker: None,
        }),
        cv: Condvar::new(),
    });
    (
        Pending {
            shared: shared.clone(),
            cancel,
        },
        Completer {
            shared: Some(shared),
        },
    )
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
        let (p, c) = pending(CancelToken::new());
        c.complete(r);
        p
    }
    /// Request cancellation. The worker stops at its next check (between
    /// command buffers) and the result becomes `Err(Cancelled)`.
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
        let (p, c) = pending::<u32>(CancelToken::new());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            c.complete(Ok(7));
        });
        assert_eq!(p.wait(), Ok(7));
    }

    #[test]
    fn dropped_completer_is_an_error() {
        let (p, c) = pending::<u32>(CancelToken::new());
        drop(c);
        assert!(matches!(p.wait(), Err(MkError::Backend(_))));
    }

    #[test]
    fn timeout_and_future() {
        let (mut p, c) = pending::<u32>(CancelToken::new());
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
}
