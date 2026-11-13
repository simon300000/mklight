//! Pupil aiming on the device: `aiming::Request::GenerateAim` from host
//! chief seeds (`GenerateAim::seeds`, the default host-aiming division),
//! with a host seed table or none (`kernels/aiming.wgsl`, the trace module
//! followed by the aim solver).
//!
//! One thread per sample runs the whole aiming schedule of the
//! specification (`mklight_gpu::aiming::kernel`: table seed, chief
//! prediction, continuation); the trace of every Newton evaluation is the
//! trace kernel's core. Generated rays and reports are bitwise the CPU-f32
//! specification's. Samples go out in launches of at most [`AIM_LAUNCH`]
//! (one batch each, cancellation checked between them); the host validates
//! the field rows as the specification does, marks culled samples with
//! `MKL_AIM_SKIP` (`kernel::excluded`), and accounts the aim statistics and
//! fallbacks from the read-back reports (`AimStats::from_report`).
//!
//! Generations without host seeds (the device chief solve) or with real
//! aims that would build a device seed table (`uses_table` without a host
//! table) run the specification, which solves and caches those rows.

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::aiming::{FieldSpec, GenerateAim, Request, Sample, kernel, uses_table};
use mklight_gpu::consts::*;
use mklight_gpu::layout::TraceParams;
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::{AimStats, ResidentOp};

use super::WebGpuOps;
use super::collectives::fits;
use super::trace::{has_overlays, shape, uniform};
use crate::device::*;

/// Samples per launch: a launch runs every sample's whole schedule, so it
/// stays short enough for cancellation checks and device watchdogs.
pub const AIM_LAUNCH: usize = 1 << 14;
const GROUP: usize = 64;
/// Header words of the packed input (`AIM_*` in `aiming.wgsl`).
const HEADER: usize = 16;
/// Output planes of a launch: the `ResidentRays` layout.
const PLANES: usize = 23;

fn module(overlays: bool) -> &'static str {
    if overlays {
        "aiming_overlays"
    } else {
        "aiming"
    }
}

/// The aiming pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    [false, true]
        .into_iter()
        .map(|o| (Kernel::new(module(o), "mkl_aim_generate"), shape()))
        .collect()
}

/// Run an aiming operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    match op {
        ResidentOp::Aiming(Request::GenerateAim(g)) => generate(ops, g, x, ctl, stats),
        _ => Ok(None),
    }
}

fn generate(
    ops: &WebGpuOps,
    g: &GenerateAim,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let Some(seeds) = &g.seeds else {
        return Ok(None);
    };
    let set = x.system(g.systems)?;
    let fields: &[FieldSpec] = x.rows(&g.field_wavelength);
    // A device seed table serves real aims only (paraxial launches never
    // read it): build it with the specification.
    let real = fields.iter().any(|f| f.flags & MKL_AIM_REAL != 0);
    if real && g.options.table.is_none() && uses_table(g) {
        return Ok(None);
    }
    let samples: Vec<Sample> = x.rows(&g.samples).to_vec();
    // Invalid rows: the specification reports them.
    if fields.iter().any(|f| {
        f.system as usize >= set.headers.len()
            || f.wavelength >= set.headers[f.system as usize].num_wavelengths
    }) || samples
        .iter()
        .any(|s| (s.field & !MKL_AIM_SKIP) as usize >= fields.len())
    {
        return Ok(None);
    }
    let n = g.output.count;
    let side = g.options.table_side;
    // Packed input: header, field rows, seeds, table, then each launch's
    // samples (culled samples carry MKL_AIM_SKIP).
    let mut head = vec![0u32; HEADER];
    head.extend_from_slice(x.view(&g.field_wavelength));
    let seeds_at = head.len();
    head.extend_from_slice(x.view(seeds));
    let table_at = head.len();
    if let Some(t) = &g.options.table {
        head.extend_from_slice(x.view(t));
    }
    let samples_at = head.len();
    head[1] = g.max_iterations;
    head[2] = g.tolerance.to_bits();
    head[3] = u32::from(g.options.table.is_some());
    head[4] = side;
    head[5] = g.options.schedule_cap;
    head[6] = HEADER as u32;
    head[7] = seeds_at as u32;
    head[8] = table_at as u32;
    head[9] = samples_at as u32;
    let gpu = ops.backend.gpu().clone();
    let gpu = &*gpu;
    let words = MKL_AIM_SAMPLE_WORDS as usize;
    let launch = AIM_LAUNCH
        .min((gpu.summary().max_compute_workgroups_per_dimension as usize) * GROUP)
        .min(
            (gpu.summary().max_storage_buffer_binding_size as usize / 4).saturating_sub(head.len())
                / words,
        )
        .min(gpu.summary().max_storage_buffer_binding_size as usize / 4 / PLANES);
    if launch == 0 || !fits(gpu, head.len() + words) {
        return Ok(None);
    }
    let dev = ops.backend.device_systems(&set)?;
    let kernel = Kernel::new(module(has_overlays(&set)), "mkl_aim_generate");
    let max_surfaces = set
        .headers
        .iter()
        .map(|h| h.num_surfaces)
        .max()
        .unwrap_or(1);
    let steps: Vec<u32> = (1..max_surfaces.max(2)).collect();
    let params = uniform(
        &TraceParams {
            num_steps: steps.len() as u32,
            ..kernel::newton_params()
        },
        dev.count,
    );
    let cull = g.options.cull_outside;
    let o = &g.output;
    x.with_outputs(&[o.floats, o.indices, o.logical_ids, o.report], |_, out| {
        // Output planes in kernel order.
        let mut planes: Vec<&mut [u32]> = Vec::with_capacity(PLANES);
        for (b, k) in out.iter_mut().zip([15, 2, 2, 4]) {
            planes.extend(b[..k * n].chunks_mut(n.max(1)));
        }
        let mut input = head.clone();
        let mut start = 0;
        while start < n {
            ctl.check()?;
            let len = launch.min(n - start);
            input.truncate(samples_at);
            input[0] = len as u32;
            for s in &samples[start..start + len] {
                let mut s = *s;
                if kernel::excluded(&s, cull) {
                    s.field |= MKL_AIM_SKIP;
                }
                input.extend_from_slice(bytemuck::cast_slice(std::slice::from_ref(&s)));
            }
            let mut batch = Batch::default();
            let mut storage = vec![
                Binding::read(batch.persistent(dev.headers)),
                Binding::read(batch.persistent(dev.surfaces)),
                Binding::read(batch.persistent(dev.coefficients)),
                Binding::read(batch.upload(&steps)),
                Binding::read(batch.upload(&input)),
            ];
            let result = batch.scratch(PLANES * len);
            let unused = batch.scratch(4);
            storage.extend([Binding::write(result), Binding::write(unused)]);
            batch.passes.push(Pass {
                kernel,
                uniform: Some(params.clone()),
                storage,
                workgroups: [len.div_ceil(GROUP) as u32, 1, 1],
            });
            for (k, plane) in planes.iter_mut().enumerate() {
                batch.read(result, k * len, &mut plane[start..start + len]);
            }
            let b = gpu.run(batch)?;
            stats.device_seconds += b.device_seconds;
            stats.command_buffers += 1;
            start += len;
            ctl.report(start as f64 / n as f64, "pupil aiming");
        }
        ctl.check()
    })?;
    let report = x.words(o.report);
    let fallbacks = (0..n)
        .filter(|&i| report[i] != 0 && report[i] != MKL_AIM_OUTSIDE)
        .count();
    let aim = AimStats::from_report(report, n);
    x.stats.fallback_count += fallbacks as u64;
    x.stats.aiming.add(&aim);
    Ok(Some("webgpu:generate_aim"))
}
