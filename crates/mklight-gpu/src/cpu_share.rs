//! CPU thread budget of the CPU share inside hybrid CPU/GPU designs.
//!
//! A hybrid runs CPU work (host aims, image-point solves, reference rows)
//! while its plans run on the GPU. By default that work uses the global
//! rayon pool (every core). The work-split study key [`KEY`] gives it a pool
//! of its own of `N` threads instead (`MKLIGHT_SPLIT_FIELDMAPS_THREADS=N`),
//! leaving the rest of the cores to the threads that feed the GPU. Results
//! never depend on the budget: every parallel stage collects in a fixed order.
//!
//! Measurement scaffolding of the `fieldmaps` study (`mklight_core::split`).

use std::sync::{Arc, Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Work-split key of the CPU share's thread budget (a positive integer).
pub const KEY: &str = "fieldmaps_threads";

/// The selected budget, `None` for the global pool (also for a value that
/// is not a positive integer).
pub fn threads() -> Option<usize> {
    mklight_core::split::variant(KEY)
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
}

/// Runs `f` in the CPU share's pool: the global pool, or a cached pool of
/// [`threads`] threads.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match threads() {
        Some(n) => pool(n).install(f),
        None => f(),
    }
}

fn pool(n: usize) -> Arc<ThreadPool> {
    static POOLS: Mutex<Vec<(usize, Arc<ThreadPool>)>> = Mutex::new(Vec::new());
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = pools.iter().find(|(k, _)| *k == n) {
        return p.clone();
    }
    let p = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("mklight-cpu-share-{i}"))
            .build()
            .expect("CPU share thread pool"),
    );
    pools.push((n, p.clone()));
    p
}

/// The budget for the run's evidence: the thread count, or `"global"`.
pub fn evidence() -> serde_json::Value {
    match threads() {
        Some(n) => serde_json::json!(n),
        None => serde_json::json!("global"),
    }
}
