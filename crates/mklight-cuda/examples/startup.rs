//! Start-up cost of the CUDA backend: device open (driver, context, the
//! numerics probe including its module) and every other kernel module
//! (`prepare_kernels`), cold or from the CUBIN cache.
//!
//! ```text
//! cargo run --release -p mklight-cuda --example startup -- [CACHE_DIR]
//! ```
//!
//! With `CACHE_DIR` the cache is that directory (`MKLIGHT_CUDA_CACHE`): run
//! twice on an empty directory for the cold and the cached start. Prints
//! one JSON line.


#[cfg(not(any(windows, target_os = "linux")))]
fn main() {
    eprintln!("CUDA runs on Windows and Linux only");
}

#[cfg(any(windows, target_os = "linux"))]
fn main() {
    use mklight_gpu::ComputeBackend;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    if let Some(dir) = std::env::args().nth(1) {
        // SAFETY: single-threaded before any other code reads it.
        unsafe { std::env::set_var("MKLIGHT_CUDA_CACHE", dir) };
    }
    let t0 = Instant::now();
    let Some(b) = mklight_cuda::open().expect("CUDA open") else {
        eprintln!("no CUDA device");
        std::process::exit(2);
    };
    let open = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let p = b
        .prepare_kernels(None, &|_, _| {}, &AtomicBool::new(false))
        .expect("prepare");
    let prepare = t1.elapsed().as_secs_f64();
    let (compiled, cached, module_s) = mklight_cuda::device::compile_stats();
    println!(
        "{{\"device\":{:?},\"open_s\":{open:.3},\"prepare_s\":{prepare:.3},\"kernels\":{},\"modules_compiled\":{compiled},\"modules_cached\":{cached},\"module_s\":{module_s:.3}}}",
        b.info().name,
        p.kernels
    );
}
