//! Admission of resident plans to the device. At most `plans` plans execute
//! at once, holding at most `bytes` of declared plan storage between them (a
//! larger plan runs alone). Consumers may therefore submit independent plans
//! ahead of their results without bounding device memory or the commands in
//! flight themselves; a plan cancelled while it waits (its consumer became
//! obsolete) never reaches the device.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use mklight_core::MkResult;
use mklight_gpu::pending::JobControl;

/// Concurrently executing plans: enough to keep the device fed while other
/// plans' hosts prepare, wait or read back. `MKLIGHT_METAL_ADMIT_PLANS`.
const PLANS: usize = 4;
/// Declared plan storage of the admitted plans. `MKLIGHT_METAL_ADMIT_MIB`.
const BYTES: u64 = 1 << 30;

#[derive(Default)]
struct State {
    plans: usize,
    bytes: u64,
    peak_bytes: u64,
}

pub(crate) struct Admission {
    plans: usize,
    bytes: u64,
    state: Mutex<State>,
    cv: Condvar,
}

/// An admitted plan; dropping it releases its share.
pub(crate) struct Permit<'a> {
    admission: &'a Admission,
    bytes: u64,
    /// Seconds the plan waited for admission.
    pub(crate) waited: f64,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut s = self.admission.lock();
        s.plans -= 1;
        s.bytes -= self.bytes;
        drop(s);
        self.admission.cv.notify_all();
    }
}

fn env(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

impl Admission {
    pub(crate) fn new() -> Self {
        Self::with_limits(
            env("MKLIGHT_METAL_ADMIT_PLANS").map_or(PLANS, |v| v as usize),
            env("MKLIGHT_METAL_ADMIT_MIB").map_or(BYTES, |v| v << 20),
        )
    }
    pub(crate) fn with_limits(plans: usize, bytes: u64) -> Self {
        Self {
            plans: plans.max(1),
            bytes,
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
        }
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Wait until a plan of `bytes` fits, checking `ctl` while waiting.
    pub(crate) fn admit(&self, bytes: u64, ctl: &JobControl) -> MkResult<Permit<'_>> {
        let start = Instant::now();
        let mut s = self.lock();
        loop {
            ctl.check()?;
            if s.plans == 0 || (s.plans < self.plans && s.bytes + bytes <= self.bytes) {
                s.plans += 1;
                s.bytes += bytes;
                s.peak_bytes = s.peak_bytes.max(s.bytes);
                return Ok(Permit {
                    admission: self,
                    bytes,
                    waited: start.elapsed().as_secs_f64(),
                });
            }
            s = self
                .cv
                .wait_timeout(s, Duration::from_millis(2))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
    /// Largest declared storage admitted at once so far.
    #[cfg(test)]
    pub(crate) fn peak_bytes(&self) -> u64 {
        self.lock().peak_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{JobContext, MkError};
    use mklight_gpu::pending::pending;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn plans_and_bytes_are_bounded_and_waiters_cancel() {
        let a = Admission::with_limits(2, 100);
        let ctx = JobContext::new();
        let (_p, _c, ctl) = pending::<()>(&ctx);
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let _permit = a.admit(40, &ctl).unwrap();
                    let n = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(5));
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert!(a.peak_bytes() <= 100);
        // An oversized plan runs alone.
        let big = a.admit(500, &ctl).unwrap();
        assert_eq!(a.peak_bytes(), 500);
        // A waiter whose consumer is cancelled leaves without admission.
        let (p, _c, waiting) = pending::<()>(&ctx);
        let t = Instant::now();
        std::thread::scope(|s| {
            let h = s.spawn(|| a.admit(10, &waiting).map(|_| ()));
            std::thread::sleep(Duration::from_millis(10));
            p.cancel();
            assert!(matches!(h.join().unwrap(), Err(MkError::Cancelled)));
        });
        assert!(t.elapsed() < Duration::from_millis(200));
        drop(big);
        assert!(a.admit(10, &ctl).is_ok());
    }
}
