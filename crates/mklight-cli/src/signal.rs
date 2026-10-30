//! Ctrl-C (SIGINT) handling for long jobs.
//!
//! The wait loop translates the first signal into stop (when supported),
//! and the second into cancellation. Signal handlers only count requests.

use std::sync::atomic::{AtomicU32, Ordering};

static COUNT: AtomicU32 = AtomicU32::new(0);

/// Number of interrupt requests since installation.
pub fn count() -> u32 {
    COUNT.load(Ordering::SeqCst)
}

#[cfg(unix)]
extern "C" fn on_signal(_: libc::c_int) {
    COUNT.fetch_add(1, Ordering::SeqCst);
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
