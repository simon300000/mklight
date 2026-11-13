//! MKLight's WebGPU compute backend (GPU.md §11): `mklight_gpu::ComputeBackend`
//! on wgpu with WGSL kernels over the packed f32 layout every backend shares.

pub mod backend;
pub mod clock;
pub mod device;
pub mod kernels;
pub mod ops;
pub mod proxy;
#[cfg(target_arch = "wasm32")]
pub mod web;
pub mod wgpu_device;

pub use backend::WebGpuBackend;
pub use device::Gpu;
pub use wgpu_device::WgpuDevice;
