//! The plan operations and jobs that run on the WebGPU device, and the
//! device side of each (GPU.md §11, "Ported operations"). Everything else in
//! a plan runs its CPU-f32 specification on the host (`backend` module
//! docs).

use std::collections::HashMap;
use std::sync::Arc;

use mklight_core::MkResult;
use mklight_gpu::pending::JobControl;
use mklight_gpu::reduce;
use mklight_gpu::resident::ResidentOp;
use mklight_gpu::resident::cpu::{DeviceOps, Execution};
use mklight_gpu::{JobStats, PackedSystemSet, ReduceMode, SystemHandle, TraceRequest, TraceResult};

use crate::backend::WebGpuBackend;
use crate::device::*;

pub mod trace;
pub mod transforms;

/// Whether the trace kernel is ported (the engine opens a device only when
/// it runs `JobKind::Trace`).
pub const TRACE_PORTED: bool = trace::PORTED;

/// Every pipeline the backend creates, for `prepare_kernels`.
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    use Access::*;
    let shape = |storage: &[Access]| PassShape {
        uniform: true,
        storage: storage.to_vec(),
    };
    vec![
        (
            Kernel::new("numerics", "mkl_numerics_probe"),
            shape(&[Read, Read, Read, ReadWrite]),
        ),
        (
            Kernel::new("numerics", "mkl_df_ops"),
            shape(&[Read, Read, ReadWrite]),
        ),
        (REDUCE_PASS1, shape(&REDUCE_SHAPE)),
        (REDUCE_PASS2, shape(&REDUCE_SHAPE)),
    ]
    .into_iter()
    .chain(trace::pipelines())
    .chain(transforms::pipelines())
    .collect()
}

const REDUCE_PASS1: Kernel = Kernel::new("reduce", "mkl_reduce_pass1");
const REDUCE_PASS2: Kernel = Kernel::new("reduce", "mkl_reduce_pass2");
const REDUCE_SHAPE: [Access; 3] = [Access::Read, Access::ReadWrite, Access::ReadWrite];

/// The device operations of one plan.
pub struct WebGpuOps {
    pub backend: WebGpuBackend,
    /// The plan's pinned systems (their device copies:
    /// `WebGpuBackend::device_systems`).
    pub systems: HashMap<SystemHandle, Arc<PackedSystemSet>>,
}

impl WebGpuOps {
    pub fn new(
        backend: WebGpuBackend,
        pinned: &HashMap<SystemHandle, Arc<PackedSystemSet>>,
    ) -> Self {
        Self {
            backend,
            systems: pinned.clone(),
        }
    }
}

impl DeviceOps for WebGpuOps {
    fn run(
        &mut self,
        i: usize,
        op: &ResidentOp,
        x: &mut Execution,
        ctl: &JobControl,
        stats: &mut JobStats,
    ) -> MkResult<Option<&'static str>> {
        let gpu = self.backend.gpu().clone();
        match op {
            ResidentOp::Sum(s) => {
                let [channels, count] = s.input.desc.shape[..] else {
                    unreachable!("validated sum shape")
                };
                if !sum_fits(&*gpu, count) {
                    return Ok(None);
                }
                x.with_outputs(&[s.output], |x, out| {
                    sum(
                        &*gpu,
                        s.mode,
                        channels,
                        count,
                        x.view(&s.input),
                        out[0],
                        ctl,
                        stats,
                    )
                })?;
                Ok(Some("webgpu:sum"))
            }
            ResidentOp::TraceUpload(_)
            | ResidentOp::Aiming(mklight_gpu::aiming::Request::TraceResident(_)) => {
                trace::run(self, i, op, x, ctl, stats)
            }
            ResidentOp::Transforms(_) => transforms::run(self, i, op, x, ctl, stats),
            _ => Ok(None),
        }
    }
}

/// Whether one channel of `count` values fits a storage binding.
pub fn sum_fits(gpu: &dyn Gpu, count: usize) -> bool {
    (count * 4) as u64 <= gpu.summary().max_storage_buffer_binding_size
}

/// The fixed-tree sum of `channels` channels of `count` f32 values
/// (`values`, channel-major words) into `out` (`channels` double-floats):
/// bitwise `mklight_gpu::reduce` in the device's environment.
#[allow(clippy::too_many_arguments)]
pub fn sum(
    gpu: &dyn Gpu,
    mode: ReduceMode,
    channels: usize,
    count: usize,
    values: &[u32],
    out: &mut [u32],
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<()> {
    if !sum_fits(gpu, count) {
        return Err(mklight_core::MkError::Unsupported(
            "webgpu: a reduction channel exceeds maxStorageBufferBindingSize".into(),
        ));
    }
    let groups = reduce::num_groups(count);
    let s = gpu.summary();
    let per_binding = (s.max_storage_buffer_binding_size as usize / (count * 4).max(1)).max(1);
    let step = per_binding
        .min(s.max_compute_workgroups_per_dimension as usize)
        .min((1 << 22) / count.max(1) + 1);
    for c0 in (0..channels).step_by(step) {
        ctl.check()?;
        let nc = step.min(channels - c0);
        let params = |first_group: u32| {
            uniform_words(&[
                count as u32,
                count as u32,
                groups as u32,
                first_group,
                mode as u32,
                0,
                0,
                0,
            ])
        };
        let mut batch = Batch::default();
        let x = batch.upload(&values[c0 * count..(c0 + nc) * count]);
        let partials = batch.scratch(nc * groups * 2);
        let sums = batch.scratch(nc * 2);
        let storage = vec![
            Binding::read(x),
            Binding::write(partials),
            Binding::write(sums),
        ];
        batch.passes.push(Pass {
            kernel: REDUCE_PASS1,
            uniform: Some(params(0)),
            storage: storage.clone(),
            workgroups: [groups as u32, nc as u32, 1],
        });
        batch.passes.push(Pass {
            kernel: REDUCE_PASS2,
            uniform: Some(params(0)),
            storage,
            workgroups: [1, nc as u32, 1],
        });
        batch.read(sums, 0, &mut out[c0 * 2..(c0 + nc) * 2]);
        let b = gpu.run(batch)?;
        stats.device_seconds += b.device_seconds;
        stats.command_buffers += 1;
        ctl.report((c0 + nc) as f64 / channels as f64, "webgpu sum");
    }
    Ok(())
}

/// `Job::Trace` on the device.
pub fn trace_job(
    backend: &WebGpuBackend,
    req: &TraceRequest,
    set: &PackedSystemSet,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<TraceResult> {
    trace::job(backend, req, set, ctl, stats)
}
