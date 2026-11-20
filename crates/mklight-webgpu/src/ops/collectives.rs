//! Resident collectives on the device: `collectives::Request::SegmentReduce`
//! of every kind (`kernels/collectives.wgsl`).
//!
//! One batch per operation: the inputs are loaded through the mirrors, every
//! term of every segment is reduced by the fixed tree of
//! `mklight_gpu::reduce::tree_in` in two passes per phase (pass 1: one
//! workgroup per pass-1 group, term and segment into a partials scratch;
//! pass 2: one workgroup per segment; tiles of segments bounded by the
//! dispatch and binding limits), and the output and status are stored into
//! their mirrors (no readback). Results are bitwise the CPU-f32 specification's. Scans,
//! compaction, sorts and QR fits (no geometric analysis plans them) and
//! inputs beyond one storage binding run the specification.

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::collectives::{Request, SegmentKind, SegmentReduce, segment_shape};
use mklight_gpu::pending::JobControl;
use mklight_gpu::reduce;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::{PlanView, ResidentOp};

use super::WebGpuOps;
use crate::device::*;

const PASS1: Kernel = Kernel::new("collectives", "mkl_segment_pass1");
const PASS2: Kernel = Kernel::new("collectives", "mkl_segment_pass2");

fn shape() -> PassShape {
    use Access::*;
    PassShape {
        uniform: true,
        storage: vec![
            Read, Read, Read, Read, Read, ReadWrite, ReadWrite, ReadWrite,
        ],
    }
}

/// The collective pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    vec![(PASS1, shape()), (PASS2, shape())]
}

/// Run a collective of a plan on the device, or `Ok(None)` for its CPU-f32
/// specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    match op {
        ResidentOp::Collectives(Request::SegmentReduce(r)) => segments(ops, r, x, ctl, stats),
        _ => Ok(None),
    }
}

/// Whether `words` 32-bit words fit one storage binding.
pub(crate) fn fits(gpu: &dyn Gpu, words: usize) -> bool {
    (words as u64) * 4 <= gpu.summary().max_storage_buffer_binding_size
}

fn segments(
    ops: &mut WebGpuOps,
    r: &SegmentReduce,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let gpu = ops.backend.gpu().clone();
    let [c, n] = r.values.desc.shape[..] else {
        unreachable!("validated segment values")
    };
    let s = r.offsets.desc.shape[0] - 1;
    let out_len: usize = segment_shape(r).iter().product();
    let inputs = [
        Some(&r.values),
        Some(&r.offsets),
        r.mask.as_ref(),
        r.weights.as_ref(),
        r.reference.as_ref(),
    ];
    let words = |v: &PlanView| v.desc.len().unwrap_or(usize::MAX) * v.desc.element.words();
    if !fits(&*gpu, 2 * out_len)
        || inputs.iter().flatten().any(|v| !fits(&*gpu, words(v)))
        || inputs
            .iter()
            .flatten()
            .any(|v| v.buffer == r.output || v.buffer == r.status)
    {
        return Ok(None);
    }
    let terms = |phase: usize| match r.kind {
        SegmentKind::Moments2 => [4, 6][phase],
        SegmentKind::Summary => 4 * c,
        _ => c,
    };
    let phases = if r.kind == SegmentKind::Moments2 {
        2
    } else {
        1
    };
    let max_terms = (0..phases).map(terms).max().unwrap_or(1);
    let maxg = reduce::num_groups(r.max_segment_len);
    let max_dim = gpu.summary().max_compute_workgroups_per_dimension as usize;
    // Segments per tile: one grid dimension, partials within one binding.
    let binding_words = (gpu.summary().max_storage_buffer_binding_size / 4) as usize;
    let tile = max_dim.min(binding_words / (2 * max_terms * maxg)).max(1);
    if max_terms > max_dim || !fits(&*gpu, 2 * max_terms * maxg) {
        return Ok(None);
    }
    let w = |v: Option<&PlanView>| v.map_or(0, |v| v.desc.element.words() as u32);
    let params = |first: usize, phase: usize| {
        uniform_words(&[
            n as u32,
            c as u32,
            s as u32,
            first as u32,
            r.kind as u32,
            r.mode as u32,
            r.max_segment_len as u32,
            u32::from(r.mask.is_some()),
            r.mask_value,
            w(Some(&r.values)),
            w(r.weights.as_ref()),
            w(r.reference.as_ref()),
            phase as u32,
            maxg as u32,
            terms(phase) as u32,
        ])
    };
    ctl.check()?;
    if s == 0 {
        return Ok(Some("webgpu:segment_reduce"));
    }
    let x: &Execution = x;
    let placeholder = [0u32; 4];
    let mut batch = Batch::default();
    let bound: Vec<usize> = inputs
        .iter()
        .map(|v| match v {
            Some(v) => ops.mirrors.load(&mut batch, x, v),
            None => batch.upload(&placeholder),
        })
        .collect();
    let o = batch.scratch(2 * out_len.max(1));
    let st = batch.scratch(s.max(1));
    let partials = batch.scratch(2 * tile.min(s) * max_terms * maxg);
    let storage: Vec<Binding> = bound
        .iter()
        .map(|&b| Binding::read(b))
        .chain([o, st, partials].map(Binding::write))
        .collect();
    for first in (0..s).step_by(tile) {
        let segs = (s - first).min(tile) as u32;
        for phase in 0..phases {
            for (kernel, workgroups) in [
                (PASS1, [maxg as u32, terms(phase) as u32, segs]),
                (PASS2, [segs, 1, 1]),
            ] {
                batch.passes.push(Pass {
                    kernel,
                    uniform: Some(params(first, phase)),
                    storage: storage.clone(),
                    workgroups,
                });
            }
        }
    }
    ops.mirrors
        .store(&mut batch, x, o, 0, r.output, 0, 2 * out_len)?;
    ops.mirrors.store(&mut batch, x, st, 0, r.status, 0, s)?;
    let b = gpu.run(batch)?;
    stats.device_seconds += b.device_seconds;
    stats.command_buffers += 1;
    ctl.check()?;
    ctl.report(1.0, "collectives");
    Ok(Some("webgpu:segment_reduce"))
}
