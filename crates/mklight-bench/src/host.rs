//! The measuring host: its load averages and the process's CPU time.

/// The 1, 5 and 15 minute load averages (zeros when unknown).
pub fn load_averages() -> [f64; 3] {
    let mut l = [0f64; 3];
    // SAFETY: getloadavg writes at most the 3 doubles it is given.
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    l
}

/// The one-minute load average (0 when unknown).
pub fn load() -> f64 {
    load_averages()[0]
}

/// Process CPU time so far (user + system, all threads) in ms.
pub fn cpu_ms() -> f64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes one rusage record.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) };
    // SAFETY: zero-initialised and written by getrusage.
    let u = unsafe { u.assume_init() };
    let t = |v: libc::timeval| v.tv_sec as f64 * 1e3 + v.tv_usec as f64 * 1e-3;
    t(u.ru_utime) + t(u.ru_stime)
}
