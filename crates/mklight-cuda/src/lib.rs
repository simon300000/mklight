//! MKLight's CUDA compute backend: `mklight_gpu::ComputeBackend` on
//! NVIDIA GPUs.
//!
//! The backend is the WebGPU backend's host (`mklight_webgpu::WebGpuBackend`:
//! plans on the shared hosted executor, device operations with CPU-f32
//! specifications for the rest, device-resident mirrors between operations)
//! over a CUDA implementation of its device seam (`mklight_webgpu::Gpu`):
//! [`CudaDevice`] runs the same batches with CUDA kernels that implement the
//! WGSL kernels' interface and include the Metal kernels' arithmetic.
//!
//! The driver (`nvcuda.dll`, `libcuda.so`) and NVRTC are loaded at run time,
//! so the crate builds without a CUDA toolkit; without a driver, a device or
//! (for an uncached kernel) NVRTC the backend is unavailable. On other
//! platforms the crate is empty.

#[cfg(any(windows, target_os = "linux"))]
pub mod device;
pub mod kernels;

#[cfg(any(windows, target_os = "linux"))]
pub use device::CudaDevice;

use mklight_core::MkResult;

/// Device seconds of every CUDA batch completed in this process (0 without
/// CUDA; benchmarks only).
pub fn device_seconds() -> f64 {
    #[cfg(any(windows, target_os = "linux"))]
    {
        device::device_seconds()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        0.0
    }
}
use mklight_webgpu::WebGpuBackend;

/// The CUDA backend: the WebGPU host over a [`CudaDevice`] (its
/// `info().kind` is `BackendKind::Cuda`, its id `cuda:<device>`).
pub type CudaBackend = WebGpuBackend;

/// Open the CUDA backend on device 0 (runs the numerics probe), or
/// `Ok(None)` when there is no CUDA driver or device.
pub fn open() -> MkResult<Option<CudaBackend>> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        match CudaDevice::open()? {
            Some(d) => WebGpuBackend::with_gpu(std::sync::Arc::new(d)).map(Some),
            None => Ok(None),
        }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Ok(None)
    }
}
