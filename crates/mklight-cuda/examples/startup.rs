//! Start-up cost of the CUDA backend: device open (driver, context and the
//! numerics probe with its kernel module) and `prepare_kernels` (every other
//! kernel, compiled into `CACHE_DIR/cuda` or loaded from it), as a front end
//! runs them through `Engine::prepare_gpu`.
//!
//! ```text
//! cargo run --release -p mklight-cuda --example startup -- CACHE_DIR
//! ```
//!
//! Run twice on an empty directory for the cold and the cached start. Prints
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

    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("CACHE_DIR"));
    let t0 = Instant::now();
    let Some(b) = mklight_cuda::open().expect("CUDA open") else {
        eprintln!("no CUDA device");
        std::process::exit(2);
    };
    let open = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let p = b
        .prepare_kernels(Some(&dir), &|_, _| {}, &AtomicBool::new(false))
        .expect("prepare");
    let prepare = t1.elapsed().as_secs_f64();
    println!(
        "{{\"device\":{:?},\"open_s\":{open:.3},\"prepare_s\":{prepare:.3},\"kernels\":{},\"from_cache\":{},\"warning\":{:?}}}",
        b.info().name,
        p.kernels,
        p.from_cache,
        p.warning
    );
}
