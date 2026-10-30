//! Ctrl-C (SIGINT) handling for long jobs.
//!
//! While a job runs, the first SIGINT only raises a flag; the wait loop sees
//! it, cancels the job through the engine and reports the partial status, so
//! the job stops at its next cancellation check and the process exits with
//! code 130. A second SIGINT ends the process immediately (also with 130).
//! SIGTERM is treated like SIGINT. On platforms without POSIX signals the
//! default behaviour (immediate termination) is kept.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicU32 = AtomicU32::new(0);

/// Has an interrupt arrived since `install`?
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

#[cfg(unix)]
extern "C" fn on_signal(_: libc::c_int) {
    // Only async-signal-safe operations: atomics and `_exit`.
    INTERRUPTED.store(true, Ordering::SeqCst);
    if COUNT.fetch_add(1, Ordering::SeqCst) >= 1 {
        unsafe { libc::_exit(130) };
    }
}

/// Install the handlers (idempotent).
pub fn install() {
    #[cfg(unix)]
    unsafe {
        let h = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGINT, h);
        libc::signal(libc::SIGTERM, h);
    }
    let _ = &COUNT;
}
