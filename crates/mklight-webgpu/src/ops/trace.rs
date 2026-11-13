//! The sequential / path trace kernel on the device: `Job::Trace`,
//! `ResidentOp::TraceUpload` and `aiming::Request::TraceResident`
//! (`kernels/trace.wgsl`).

use mklight_core::{MkError, MkResult};
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::ResidentOp;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::{JobStats, PackedSystemSet, TraceRequest, TraceResult};

use super::WebGpuOps;
use crate::backend::WebGpuBackend;
use crate::device::{Kernel, PassShape};

/// Whether the kernel is ported.
pub const PORTED: bool = false;

/// The trace pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    vec![]
}

/// Run a trace operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    _ops: &mut WebGpuOps,
    _i: usize,
    _op: &ResidentOp,
    _x: &mut Execution,
    _ctl: &JobControl,
    _stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    Ok(None)
}

/// `Job::Trace` on the device.
pub fn job(
    _backend: &WebGpuBackend,
    _req: &TraceRequest,
    _set: &PackedSystemSet,
    _ctl: &JobControl,
    _stats: &mut JobStats,
) -> MkResult<TraceResult> {
    Err(MkError::Unsupported(
        "webgpu: the trace kernel is not ported yet".into(),
    ))
}
