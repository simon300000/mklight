//! The plan operations and jobs that run on the WebGPU device, and the
//! device side of each (GPU.md §11, "Ported operations"). Everything else in
//! a plan runs its CPU-f32 specification on the host (`backend` module
//! docs).

use std::collections::HashMap;
use std::sync::Arc;

use mklight_core::MkResult;
use mklight_gpu::pending::JobControl;
use mklight_gpu::reduce;
use mklight_gpu::resident::cpu::{DeviceOps, Execution};
use mklight_gpu::resident::{PlanView, ResidentOp, ResidentPlan};
use mklight_gpu::{JobStats, PackedSystemSet, ReduceMode, SystemHandle, TraceRequest, TraceResult};

use crate::backend::WebGpuBackend;
use crate::device::*;

pub mod aiming;
pub mod collectives;
pub mod geometry;
pub mod resident;
pub mod trace;
pub mod transforms;
pub mod wave;

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
    .chain(wave::pipelines())
    .chain(aiming::pipelines())
    .chain(geometry::pipelines())
    .chain(collectives::pipelines())
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
    /// Plan buffers kept on the device between device operations.
    pub mirrors: resident::Mirrors,
    /// `MKLIGHT_WEBGPU_TRACE_PLANS=1`: each operation's name, placement and
    /// wall time (from its offer to the next offer), printed on drop.
    trace: Option<Trace>,
}

struct Trace {
    plan: crate::clock::Instant,
    last: Option<(String, crate::clock::Instant)>,
    lines: Vec<(String, f64)>,
}

impl Trace {
    fn close(&mut self) {
        if let Some((name, t)) = self.last.take() {
            self.lines.push((name, t.elapsed().as_secs_f64() * 1e3));
        }
    }
}

impl Drop for WebGpuOps {
    fn drop(&mut self) {
        if let Some(t) = &mut self.trace {
            t.close();
            let ops: Vec<String> = t
                .lines
                .iter()
                .map(|(n, ms)| format!("{n} {ms:.2}"))
                .collect();
            eprintln!(
                "webgpu plan {:.2} ms: {}",
                t.plan.elapsed().as_secs_f64() * 1e3,
                ops.join(", ")
            );
        }
    }
}

/// The variant path of an operation (`Wave::Intensity`).
fn op_name(op: &ResidentOp) -> String {
    let d = format!("{op:?}");
    let mut parts = d
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|p| !p.is_empty());
    let a = parts.next().unwrap_or("?");
    match parts.next() {
        Some(b) if b.chars().next().is_some_and(char::is_uppercase) => format!("{a}::{b}"),
        _ => a.to_string(),
    }
}

impl WebGpuOps {
    pub fn new(
        backend: WebGpuBackend,
        pinned: &HashMap<SystemHandle, Arc<PackedSystemSet>>,
    ) -> Self {
        let trace = std::env::var("MKLIGHT_WEBGPU_TRACE_PLANS")
            .is_ok_and(|v| v == "1")
            .then(|| Trace {
                plan: crate::clock::Instant::now(),
                last: None,
                lines: Vec::new(),
            });
        Self {
            mirrors: resident::Mirrors::new(backend.gpu().clone()),
            backend,
            systems: pinned.clone(),
            trace,
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
        // An operation's time runs from its offer to the next one's: the
        // device run inside `offer`, or the specification after it.
        let start = crate::clock::Instant::now();
        if let Some(t) = &mut self.trace {
            t.close();
        }
        // Operations whose device path does not load its inputs through the
        // mirrors (and every specification) read the host copies.
        if !resident_inputs(op) {
            self.mirrors.sync_reads(op, x, stats)?;
        }
        let r = self.offer(i, op, x, ctl, stats);
        match &r {
            Ok(None) => {
                self.mirrors.sync_reads(op, x, stats)?;
                self.mirrors.after(op);
            }
            Ok(Some(_)) => self.mirrors.after(op),
            Err(_) => {}
        }
        if let Some(t) = &mut self.trace {
            let place = match &r {
                Ok(Some(_)) => "device",
                Ok(None) => "spec",
                Err(_) => "error",
            };
            t.last = Some((format!("{}[{place}]", op_name(op)), start));
        }
        r
    }

    fn finish(
        &mut self,
        plan: &ResidentPlan,
        x: &mut Execution,
        _ctl: &JobControl,
    ) -> MkResult<()> {
        let mut stats = JobStats::default();
        let r = self.mirrors.finish(plan, x, &mut stats);
        if let Some(t) = &mut self.trace {
            t.close();
            t.lines.push((
                format!(
                    "final readback (resident saved {:.1} MB)",
                    self.mirrors.saved_bytes as f64 / 1e6
                ),
                stats.device_seconds * 1e3,
            ));
        }
        r
    }
}

/// Whether `op`'s device path takes its inputs through [`resident::Mirrors`]
/// (so device-resident inputs need no readback before it).
fn resident_inputs(op: &ResidentOp) -> bool {
    // One arm per converted family; `matches!` keeps them as a list.
    matches!(
        op,
        ResidentOp::Sum(_)
            | ResidentOp::Geometry(_)
            | ResidentOp::Collectives(_)
            | ResidentOp::TraceUpload(_)
            | ResidentOp::Aiming(_)
    )
}

impl WebGpuOps {
    fn offer(
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
                let step = sum_step(&*gpu, count);
                for c0 in (0..channels).step_by(step) {
                    ctl.check()?;
                    let nc = step.min(channels - c0);
                    let part = PlanView {
                        offset: s.input.offset + c0 * count,
                        desc: mklight_gpu::resident::BufferDesc::new(
                            mklight_gpu::resident::ElementType::F32,
                            [nc, count],
                        ),
                        ..s.input.clone()
                    };
                    let mut batch = Batch::default();
                    let input = self.mirrors.load(&mut batch, x, &part);
                    let sums = sum_passes(&mut batch, s.mode, nc, count, input);
                    self.mirrors
                        .store(&mut batch, x, sums, 0, s.output, c0 * 2, nc * 2)?;
                    let b = gpu.run(batch)?;
                    stats.device_seconds += b.device_seconds;
                    stats.command_buffers += 1;
                    ctl.report((c0 + nc) as f64 / channels as f64, "webgpu sum");
                }
                Ok(Some("webgpu:sum"))
            }
            ResidentOp::TraceUpload(_)
            | ResidentOp::Aiming(mklight_gpu::aiming::Request::TraceResident(_)) => {
                trace::run(self, i, op, x, ctl, stats)
            }
            ResidentOp::Transforms(_) => transforms::run(self, i, op, x, ctl, stats),
            ResidentOp::Wave(_) => wave::run(self, i, op, x, ctl, stats),
            ResidentOp::Aiming(_) => aiming::run(self, i, op, x, ctl, stats),
            ResidentOp::Geometry(_) => geometry::run(self, i, op, x, ctl, stats),
            ResidentOp::Collectives(_) => collectives::run(self, i, op, x, ctl, stats),
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
    let step = sum_step(gpu, count);
    for c0 in (0..channels).step_by(step) {
        ctl.check()?;
        let nc = step.min(channels - c0);
        let mut batch = Batch::default();
        let x = batch.upload(&values[c0 * count..(c0 + nc) * count]);
        let sums = sum_passes(&mut batch, mode, nc, count, x);
        batch.read(sums, 0, &mut out[c0 * 2..(c0 + nc) * 2]);
        let b = gpu.run(batch)?;
        stats.device_seconds += b.device_seconds;
        stats.command_buffers += 1;
        ctl.report((c0 + nc) as f64 / channels as f64, "webgpu sum");
    }
    Ok(())
}

/// Channels per sum batch: within one storage binding, the workgroup grid
/// and about 4 M values.
fn sum_step(gpu: &dyn Gpu, count: usize) -> usize {
    let s = gpu.summary();
    let per_binding = (s.max_storage_buffer_binding_size as usize / (count * 4).max(1)).max(1);
    per_binding
        .min(s.max_compute_workgroups_per_dimension as usize)
        .min((1 << 22) / count.max(1) + 1)
}

/// The two fixed-tree passes over `nc` channels of `count` values in batch
/// resource `x`; returns the resource holding the `nc` double-float sums.
fn sum_passes(batch: &mut Batch<'_>, mode: ReduceMode, nc: usize, count: usize, x: usize) -> usize {
    let groups = reduce::num_groups(count);
    let params = uniform_words(&[
        count as u32,
        count as u32,
        groups as u32,
        0,
        mode as u32,
        0,
        0,
        0,
    ]);
    let partials = batch.scratch(nc * groups * 2);
    let sums = batch.scratch(nc * 2);
    let storage = vec![
        Binding::read(x),
        Binding::write(partials),
        Binding::write(sums),
    ];
    batch.passes.push(Pass {
        kernel: REDUCE_PASS1,
        uniform: Some(params.clone()),
        storage: storage.clone(),
        workgroups: [groups as u32, nc as u32, 1],
    });
    batch.passes.push(Pass {
        kernel: REDUCE_PASS2,
        uniform: Some(params),
        storage,
        workgroups: [1, nc as u32, 1],
    });
    sums
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
