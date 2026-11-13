//! Regular transforms on the device: `ResidentOp::Transforms`
//! (`kernels/transforms.wgsl`).

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::ResidentOp;
use mklight_gpu::resident::cpu::Execution;

use super::WebGpuOps;
use crate::device::{Kernel, PassShape};

/// The transform pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    vec![]
}

/// Run a transform operation of a plan on the device, or `Ok(None)` for its
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
