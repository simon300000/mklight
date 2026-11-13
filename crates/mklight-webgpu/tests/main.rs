//! The integration tests of `mklight-webgpu` as one test program: every file
//! of this directory is a module of it (`cargo test -p mklight-webgpu --test
//! tests <file>::`). Every test skips with a message when no WebGPU adapter
//! is available.

use std::sync::OnceLock;

use mklight_webgpu::WebGpuBackend;

pub fn gpu() -> Option<&'static WebGpuBackend> {
    static B: OnceLock<Option<WebGpuBackend>> = OnceLock::new();
    let b = B
        .get_or_init(|| WebGpuBackend::new().expect("WebGPU backend initialisation failed"))
        .as_ref();
    if b.is_none() {
        eprintln!("SKIP: no WebGPU adapter");
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

mod numerics;
mod proxy;
mod trace;
mod transforms;
