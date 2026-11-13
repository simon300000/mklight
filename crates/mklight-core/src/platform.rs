//! Platform plumbing shared by every crate, so the engine's algorithms are
//! one codebase on native targets and on `wasm32-unknown-unknown` (the web
//! build, docs/WEB.md). Only plumbing lives here: clocks, threads and host
//! statistics. On native targets every item is the `std` one (or a direct
//! system call), so native behaviour is unchanged.
//!
//! * [`Instant`], [`SystemTime`], [`UNIX_EPOCH`]: `std::time` natively; the
//!   `web-time` crate on wasm32, where the `std` clocks panic.
//! * [`spawn`] / [`JoinHandle`]: a named OS thread natively. On wasm32 a
//!   thread is a Web Worker sharing the module's memory: [`spawn`] queues
//!   the closure for a *host thread* (a worker the web glue starts, which
//!   runs [`run_thread_host`]), because a worker cannot start a new worker
//!   while it blocks, and engine threads block. [`wanted_host_threads`]
//!   tells the glue how many more host workers to start.
//! * [`process_cpu_seconds`], [`load_per_core`], [`available_parallelism`]:
//!   host statistics with neutral fallbacks on wasm32.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(target_arch = "wasm32")]
pub use web_time::{Instant, SystemTime, UNIX_EPOCH};

use std::io;

/// Logical cores available to this process (`std::thread::
/// available_parallelism` natively; on wasm32 the rayon pool's size as the
/// web glue set it with [`set_available_parallelism`], else 1).
pub fn available_parallelism() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm::PARALLELISM
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1)
    }
}

/// Record the web build's core count (`navigator.hardwareConcurrency`, or
/// the rayon pool size). No effect natively.
pub fn set_available_parallelism(_n: usize) {
    #[cfg(target_arch = "wasm32")]
    wasm::PARALLELISM.store(_n, std::sync::atomic::Ordering::Relaxed);
}

/// Process CPU time (user + system, all threads) in seconds; 0 where the
/// platform has no such counter (wasm32).
pub fn process_cpu_seconds() -> f64 {
    #[cfg(unix)]
    {
        let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage writes one rusage record into valid storage.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) };
        // SAFETY: zero-initialised and written by getrusage.
        let u = unsafe { u.assume_init() };
        let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
        t(u.ru_utime) + t(u.ru_stime)
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}

/// One-minute load average per logical core; 0 when unknown (wasm32).
pub fn load_per_core() -> f64 {
    #[cfg(unix)]
    {
        let mut l = [0f64; 3];
        // SAFETY: `getloadavg` writes at most `l.len()` samples into `l`.
        let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
        let cores = available_parallelism();
        if n < 1 { 0. } else { l[0] / cores as f64 }
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Threads

/// Handle of a thread started with [`spawn`] (`std::thread::JoinHandle`
/// natively).
pub struct JoinHandle<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: std::thread::JoinHandle<T>,
    #[cfg(target_arch = "wasm32")]
    inner: wasm::Handle<T>,
}

impl<T> JoinHandle<T> {
    /// Wait for the thread; `Err` carries its panic payload.
    pub fn join(self) -> std::thread::Result<T> {
        self.inner.join()
    }

    /// Whether the thread has finished.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// The thread's id; `None` only on wasm32 while the closure waits for a
    /// host thread (so it is not the calling thread either).
    pub fn thread_id(&self) -> Option<std::thread::ThreadId> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Some(self.inner.thread().id())
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.thread_id()
        }
    }
}

/// Start a named thread running `f` (`std::thread::Builder::spawn`
/// natively). On wasm32 the closure waits for a free host thread (see the
/// module documentation); it fails only natively, when the OS refuses.
pub fn spawn<F, T>(name: impl Into<String>, f: F) -> io::Result<JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::Builder::new()
            .name(name.into())
            .spawn(f)
            .map(|inner| JoinHandle { inner })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        Ok(JoinHandle {
            inner: wasm::spawn(f),
        })
    }
}

/// Host threads the web glue should still start: queued closures that no
/// idle host thread will take. Always 0 natively.
pub fn wanted_host_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        wasm::wanted()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

/// Body of a web host thread: run queued [`spawn`] closures, one at a
/// time, forever. Called by the web glue in each host worker; never
/// returns. Natively it is not used and returns at once.
pub fn run_thread_host() {
    #[cfg(target_arch = "wasm32")]
    wasm::host_loop();
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::collections::VecDeque;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Condvar, Mutex, OnceLock};
    use std::thread::ThreadId;

    pub(super) static PARALLELISM: AtomicUsize = AtomicUsize::new(1);

    type Task = Box<dyn FnOnce() + Send + 'static>;

    struct Queue {
        tasks: VecDeque<Task>,
        /// Host threads waiting for a task.
        idle: usize,
        /// Host threads the glue was asked for but have not started.
        starting: usize,
    }

    static QUEUE: Mutex<Queue> = Mutex::new(Queue {
        tasks: VecDeque::new(),
        idle: 0,
        starting: 0,
    });
    static WORK: Condvar = Condvar::new();

    fn lock() -> std::sync::MutexGuard<'static, Queue> {
        QUEUE.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn wanted() -> usize {
        let mut q = lock();
        let n = q.tasks.len().saturating_sub(q.idle + q.starting);
        q.starting += n;
        n
    }

    pub(super) fn host_loop() {
        let mut q = lock();
        q.starting = q.starting.saturating_sub(1);
        loop {
            while q.tasks.is_empty() {
                q.idle += 1;
                q = WORK.wait(q).unwrap_or_else(|e| e.into_inner());
                q.idle -= 1;
            }
            let task = q.tasks.pop_front().expect("non-empty");
            drop(q);
            // A task's panic is caught in its own wrapper (`spawn`).
            task();
            q = lock();
        }
    }

    struct Slot<T> {
        result: Mutex<Option<std::thread::Result<T>>>,
        done: Condvar,
        thread: OnceLock<ThreadId>,
    }

    pub(super) struct Handle<T> {
        slot: Arc<Slot<T>>,
    }

    impl<T> Handle<T> {
        pub(super) fn join(self) -> std::thread::Result<T> {
            let s = &*self.slot;
            let mut g = s.result.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(r) = g.take() {
                    return r;
                }
                g = s.done.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }

        pub(super) fn is_finished(&self) -> bool {
            self.slot
                .result
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
        }

        pub(super) fn thread_id(&self) -> Option<ThreadId> {
            self.slot.thread.get().copied()
        }
    }

    pub(super) fn spawn<F, T>(f: F) -> Handle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let slot = Arc::new(Slot {
            result: Mutex::new(None),
            done: Condvar::new(),
            thread: OnceLock::new(),
        });
        let out = slot.clone();
        let task: Task = Box::new(move || {
            let _ = out.thread.set(std::thread::current().id());
            let r = catch_unwind(AssertUnwindSafe(f));
            *out.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            out.done.notify_all();
        });
        let mut q = lock();
        q.tasks.push_back(task);
        drop(q);
        WORK.notify_one();
        Handle { slot }
    }
}
