//! The integration tests of `mklight-cuda` as one test program. The test
//! files of `mklight-webgpu` run unchanged against the CUDA backend
//! (`build.rs` renames their `webgpu` placements to `cuda`); `cuda.rs` holds
//! the CUDA-specific ones. Every test skips with a message when no CUDA
//! device is available.
//! The backend is empty on macOS, and the copied WebGPU test files compare
//! with Metal there, so the program is built off macOS only.
#![cfg(not(target_os = "macos"))]

use std::sync::OnceLock;

use mklight_webgpu::WebGpuBackend;

pub fn gpu() -> Option<&'static WebGpuBackend> {
    static B: OnceLock<Option<WebGpuBackend>> = OnceLock::new();
    let b = B
        .get_or_init(|| mklight_cuda::open().expect("CUDA backend initialisation failed"))
        .as_ref();
    if b.is_none() {
        eprintln!("SKIP: no CUDA device");
    }
    b
}

#[macro_export]
macro_rules! gpu_or_skip {
    () => {
        match $crate::gpu() {
            Some(b) => b,
            None => return,
        }
    };
}

mod aim_jacobian;
mod cuda;
mod fft_mtf;

macro_rules! webgpu_tests {
    ($($m:ident),*) => {$(
        mod $m {
            include!(concat!(env!("OUT_DIR"), "/webgpu_tests/", stringify!($m), ".rs"));
        }
    )*};
}
webgpu_tests!(
    numerics,
    transforms,
    geometry,
    collectives,
    wave,
    trace,
    aiming
);
