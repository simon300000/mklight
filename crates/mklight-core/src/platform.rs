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

// ---------------------------------------------------------------------------
// Fused multiply-add without the instruction

/// `a·b + c` with a single rounding (round to nearest, ties to even): the
/// same value as `f64::mul_add` (IEEE 754 fusedMultiplyAdd), bit for bit,
/// for every input. For targets without a fused multiply-add instruction:
/// wasm32 has none, and `mul_add` there is a call to a slow
/// integer-arithmetic `fma`; the web build links this one in its place
/// (`mklight-wasm`), so the tracer's `mul_add`s keep their exact results
/// at a fraction of the cost. Natively `mul_add` is one instruction and
/// this function is unused, except by the tests that check it against the
/// hardware.
///
/// Method (Boldo & Melquiond, "Emulation of FMA and correctly rounded sums:
/// proved algorithms using rounding to odd", IEEE TC 57(4), 2008): the
/// product exactly as `ph + pl` (Veltkamp split, Dekker product), `c + ph`
/// exactly as `sh + sl` (TwoSum), `v = RO(sl + pl)` rounded to odd, and the
/// result `RN(sh + v)`. That is exact when nothing overflows or underflows;
/// every input outside the range where that is guaranteed (non-finite
/// operands, nonzero operands with exponents beyond ±500, a nonzero product
/// below 2⁻⁹⁰⁰, an addend above 2¹⁰⁰⁰ or equal to zero, a tiny nonzero
/// intermediate sum) takes `fma_exact`, the correctly rounded integer
/// algorithm, instead.
#[inline]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    let (ea, eb, ec) = (biased_exp(a), biased_exp(b), biased_exp(c));
    // Infinities, NaN and operands too large to split or add without
    // overflow. (Biased exponents: 0 for zero and subnormals, 2047 for
    // infinities and NaN.)
    if (ea > 1523) | (eb > 1523) | (ec > 2023) {
        return fma_exact(a, b, c);
    }
    // The general path, without data-dependent branches: the tracer's
    // inputs (zero products, exact cancellations of symmetric rays) would
    // make branches on them unpredictable. A zero operand gives ph = pl = 0
    // and the result c; an exact cancellation c = -ph gives sh = sl = 0
    // and the result pl (or +0).
    let (ph, pl) = two_prod(a, b);
    let (sh, sl) = two_sum(c, ph);
    let (v, e) = two_sum(sl, pl);
    let r = sh + round_to_odd(v, e);
    // Outside the proved range (one rarely taken branch): a nonzero product
    // of tiny operands or below 2^-900, where the split or the product's
    // error could underflow; a tiny nonzero intermediate sum, where the
    // result could be subnormal; and c = 0, where IEEE's sign of zero
    // rules need the exact algorithm.
    let product = (a != 0.0) & (b != 0.0);
    let risky = (product & ((ea < 523) | (eb < 523) | (ea + eb < 1146)))
        | ((sh != 0.0) & (biased_exp(sh) < 123))
        | (c == 0.0);
    if risky {
        return fma_exact(a, b, c);
    }
    r
}

#[inline(always)]
fn biased_exp(x: f64) -> u32 {
    ((x.to_bits() >> 52) & 0x7ff) as u32
}

/// `a + b = s + e` exactly (Knuth's TwoSum; no precondition on order).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    let av = s - bv;
    (s, (a - av) + (b - bv))
}

/// Veltkamp split: `x = hi + lo` with both halves 26 bits wide.
#[inline(always)]
fn split(x: f64) -> (f64, f64) {
    let t = 134_217_729.0 * x; // 2^27 + 1
    let hi = t - (t - x);
    (hi, x - hi)
}

/// `a·b = p + e` exactly (Dekker), given no overflow or underflow.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// `RO(v + e)` for `v = RN(v + e)` exactly from TwoSum: `v` itself when the
/// sum is exact or `v`'s last bit is odd, else the neighbour of `v` towards
/// `e` (whose last bit is odd). Branch-free; `v` is nonzero whenever `e` is.
#[inline(always)]
fn round_to_odd(v: f64, e: f64) -> f64 {
    let bits = v.to_bits();
    let adjust = ((e != 0.0) as u64) & !bits & 1;
    // 0 when e has v's sign (away from zero: +1), 1 when not (towards: -1).
    let towards_zero = (e.to_bits() ^ bits) >> 63;
    f64::from_bits(
        bits.wrapping_add(adjust)
            .wrapping_sub(2 * adjust * towards_zero),
    )
}

/// The correctly rounded fused multiply-add for every input, by integer
/// arithmetic (`libm::fma`, the algorithm `mul_add` uses where there is no
/// instruction). Slow; `fma`'s fallback.
pub fn fma_exact(a: f64, b: f64, c: f64) -> f64 {
    libm::fma(a, b, c)
}

/// On wasm32 every `f64::mul_add` is a call to the C symbol `fma`, which
/// the compiler's builtins define weakly with the slow integer algorithm;
/// this strong definition replaces it with [`fma`], bit for bit the same
/// function (docs/WEB.md "Speed"). `--cfg mklight_libm_fma` keeps the
/// builtin, for measuring the difference.
#[cfg(all(target_arch = "wasm32", not(mklight_libm_fma)))]
#[unsafe(export_name = "fma")]
pub extern "C" fn fma_symbol(a: f64, b: f64, c: f64) -> f64 {
    fma(a, b, c)
}
