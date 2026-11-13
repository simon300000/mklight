//! Resident geometric algebra on the device: `geometry::Request::{
//! Coordinates, FocusTerms, Characteristic}` (`kernels/geometry.wgsl`), the
//! per-ray steps of the spot, ray-grid, best-focus and geometric-MTF plans.
//!
//! One batch per operation: the input planes are loaded through the
//! mirrors (device copies when current), passes of at most
//! `maxComputeWorkgroupsPerDimension` workgroups write the output planes and
//! the batch stores them into the outputs' mirrors (no readback). Coordinates and focus terms are bitwise the
//! CPU-f32 specification's; the characteristic function's cos/sin are
//! double-float evaluations rounded once (the specification calls libm;
//! both meet its tolerance). Other geometry requests, and operations whose
//! buffers exceed one storage binding, run the specification.

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::consts::*;
use mklight_gpu::geometry::Request;
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::{BufferDesc, BufferId, ElementType, PlanView, ResidentOp};

use super::WebGpuOps;
use super::collectives::fits;
use crate::device::*;

const GROUP: usize = 256;
const COORDINATES: Kernel = Kernel::new("geometry", "mkl_geometry_coordinates");
const FOCUS_TERMS: Kernel = Kernel::new("geometry", "mkl_geometry_focus_terms");
const CHARACTERISTIC: Kernel = Kernel::new("geometry", "mkl_geometry_characteristic");

fn shape() -> PassShape {
    use Access::*;
    PassShape {
        uniform: true,
        storage: vec![Read, Read, Read, Read, Read, ReadWrite, ReadWrite],
    }
}

/// The geometry pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    [COORDINATES, FOCUS_TERMS, CHARACTERISTIC]
        .into_iter()
        .map(|k| (k, shape()))
        .collect()
}

/// One element-parallel operation: its kernel, the parameter words after
/// `count` and `offset` (`MklWgslGeometryParams`), up to five input views
/// and two outputs (buffer, words written from word 0).
struct Elementwise {
    kernel: Kernel,
    elements: usize,
    params: [u32; 8],
    inputs: Vec<PlanView>,
    outputs: [Option<(BufferId, usize)>; 2],
}

/// The first `words` words of `buffer` as a plain word view.
fn prefix(buffer: BufferId, words: usize) -> PlanView {
    PlanView {
        buffer,
        offset: 0,
        desc: BufferDesc::new(ElementType::U32, [words]),
        stride: 1,
    }
}

fn launch(
    ops: &mut WebGpuOps,
    x: &Execution,
    e: Elementwise,
    stats: &mut JobStats,
) -> MkResult<()> {
    let gpu = ops.backend.gpu().clone();
    let placeholder = [0u32; 4];
    let mut batch = Batch::default();
    let mut storage: Vec<Binding> = Vec::new();
    for v in &e.inputs {
        let r = if v.desc.len().unwrap_or(0) == 0 {
            batch.upload(&placeholder)
        } else {
            ops.mirrors.load(&mut batch, x, v)
        };
        storage.push(Binding::read(r));
    }
    while storage.len() < 5 {
        storage.push(Binding::read(batch.upload(&placeholder)));
    }
    let o: Vec<usize> = e
        .outputs
        .iter()
        .map(|w| batch.scratch(w.map_or(0, |w| w.1).max(4)))
        .collect();
    storage.extend(o.iter().map(|&o| Binding::write(o)));
    let per_pass = gpu.summary().max_compute_workgroups_per_dimension as usize * GROUP;
    for offset in (0..e.elements).step_by(per_pass) {
        let count = per_pass.min(e.elements - offset);
        let mut words = vec![count as u32, offset as u32];
        words.extend(e.params);
        batch.passes.push(Pass {
            kernel: e.kernel,
            uniform: Some(uniform_words(&words)),
            storage: storage.clone(),
            workgroups: [count.div_ceil(GROUP) as u32, 1, 1],
        });
    }
    for (&o, out) in o.iter().zip(&e.outputs) {
        if let Some((id, words)) = *out
            && words > 0
        {
            ops.mirrors.store(&mut batch, x, o, 0, id, 0, words)?;
        }
    }
    let b = gpu.run(batch)?;
    stats.device_seconds += b.device_seconds;
    stats.command_buffers += 1;
    Ok(())
}

/// Run a geometry operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let ResidentOp::Geometry(r) = op else {
        return Ok(None);
    };
    let gpu = ops.backend.gpu().clone();
    let gpu = &*gpu;
    let floats = MKL_TRACE_IN_FLOATS as usize;
    // Inputs read whole (validated layouts) and the outputs.
    let (inputs, outputs, name): (Vec<BufferId>, Vec<BufferId>, _) = match r {
        Request::Coordinates(r) => (
            vec![
                r.hits.rays.floats,
                r.hits.status,
                r.chiefs.rays.floats,
                r.chiefs.status,
                r.chief_index.buffer,
            ],
            vec![r.output, r.mask],
            "webgpu:coordinates",
        ),
        Request::FocusTerms(r) => (
            vec![
                r.points.buffer,
                r.mask.buffer,
                r.position_moments.buffer,
                r.slope_moments.buffer,
            ],
            vec![r.output],
            "webgpu:focus_terms",
        ),
        Request::Characteristic(r) => (
            vec![r.points.buffer, r.mask.buffer, r.frequency.buffer],
            vec![r.output],
            "webgpu:characteristic",
        ),
        _ => return Ok(None),
    };
    if inputs.iter().any(|i| outputs.contains(i))
        || inputs.iter().any(|&i| !fits(gpu, x.words(i).len()))
        || outputs.iter().any(|&o| !fits(gpu, x.words(o).len()))
    {
        return Ok(None);
    }
    ctl.check()?;
    let x: &Execution = x;
    match r {
        Request::Coordinates(r) => {
            let n = r.hits.rays.count;
            let nc = r.chiefs.rays.count;
            let d = r.defocus;
            launch(
                ops,
                x,
                Elementwise {
                    kernel: COORDINATES,
                    elements: n,
                    params: [
                        n as u32,
                        nc as u32,
                        0,
                        0,
                        0,
                        0,
                        d.hi.to_bits(),
                        d.lo.to_bits(),
                    ],
                    inputs: vec![
                        prefix(r.hits.rays.floats, floats * n),
                        prefix(r.hits.status, n),
                        prefix(r.chiefs.rays.floats, floats * nc),
                        prefix(r.chiefs.status, nc),
                        r.chief_index.clone(),
                    ],
                    outputs: [Some((r.output, 10 * n)), Some((r.mask, n))],
                },
                stats,
            )?;
        }
        Request::FocusTerms(r) => {
            let n = r.mask.desc.len()?;
            launch(
                ops,
                x,
                Elementwise {
                    kernel: FOCUS_TERMS,
                    elements: n,
                    params: [
                        n as u32,
                        0,
                        r.segment_len as u32,
                        u32::from(r.centroid),
                        0,
                        0,
                        0,
                        0,
                    ],
                    inputs: vec![
                        r.points.clone(),
                        r.mask.clone(),
                        r.position_moments.clone(),
                        r.slope_moments.clone(),
                    ],
                    outputs: [Some((r.output, 6 * n)), None],
                },
                stats,
            )?;
        }
        Request::Characteristic(r) => {
            let nr = r.mask.desc.len()?;
            let total = nr * r.frequency.desc.shape[0];
            if total > u32::MAX as usize / 4 {
                return Ok(None);
            }
            launch(
                ops,
                x,
                Elementwise {
                    kernel: CHARACTERISTIC,
                    elements: total,
                    params: [nr as u32, total as u32, r.segment_len as u32, 0, 0, 0, 0, 0],
                    inputs: vec![r.points.clone(), r.mask.clone(), r.frequency.clone()],
                    outputs: [Some((r.output, 4 * total)), None],
                },
                stats,
            )?;
        }
        _ => unreachable!(),
    }
    ctl.check()?;
    ctl.report(1.0, "geometry");
    Ok(Some(name))
}
