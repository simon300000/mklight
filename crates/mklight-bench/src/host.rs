//! The measuring host: its load averages and the process's CPU time.

/// The 1, 5 and 15 minute load averages (zeros when unknown, as off unix).
pub fn load_averages() -> [f64; 3] {
    let mut l = [0f64; 3];
    // SAFETY: getloadavg writes at most the 3 doubles it is given.
    #[cfg(unix)]
    unsafe {
        libc::getloadavg(l.as_mut_ptr(), 3)
    };
    l
}

/// The one-minute load average (0 when unknown).
pub fn load() -> f64 {
    load_averages()[0]
}

/// Process CPU time so far (user + system, all threads) in ms (0 off unix,
/// e.g. the wasm32-wasip1 build of the benches).
pub fn cpu_ms() -> f64 {
    mklight_core::platform::process_cpu_seconds() * 1e3
}
