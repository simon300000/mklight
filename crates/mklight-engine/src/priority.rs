//! Feature `windows-worker-priority`: the global rayon pool's workers run
//! at `THREAD_PRIORITY_ABOVE_NORMAL` on Windows.
//!
//! The CPU analyses are fork–join: each block of rays waits for its slowest
//! chunk. When other processes keep every core busy at normal priority, a
//! preempted worker holds its block for a scheduler quantum, and the
//! analysis slows by far more than its share of the CPU (i9-12900KS, 24
//! competing busy threads: spot diagram 128 rings 36 → 1000 ms, PSF
//! 30 → 250 ms). One priority step above normal keeps the workers
//! scheduled (36 and 30 ms). Results do not change.

/// Builds the global rayon pool with raised worker priority, once. Without
/// effect when a global pool already exists (the embedder built it or
/// rayon was used before the first engine), and off Windows.
pub(crate) fn raise_workers() {
    #[cfg(windows)]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            unsafe extern "system" {
                fn GetCurrentThread() -> isize;
                fn SetThreadPriority(thread: isize, priority: i32) -> i32;
            }
            const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
            let _ = rayon::ThreadPoolBuilder::new()
                // SAFETY: Win32 calls on the current thread's pseudo-handle.
                .start_handler(|_| unsafe {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
                })
                .build_global();
        });
    }
}
