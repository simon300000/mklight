//! GPU paths of the ray-grid analyses.
//!
//! The analyses in [`GPU_ANALYSES`] trace a `fields × wavelengths × pupil`
//! grid plus one chief ray per (field, wavelength) and reduce the image
//! hits. On a GPU backend the host generates and aims the rays with the f64
//! reference (`SequentialTracer::generate`, in parallel blocks with
//! cancellation and progress between them), the device traces them with the
//! sequential trace kernel (clipped; chief rays unclipped, as
//! `SequentialTracer::chief`), and the hits are reduced by exactly the code
//! the CPU path uses (`mklight_analysis::spot::FieldRays::from_image_rays`
//! and the `*_from_rays` reductions). Only the trace differs, so the
//! results agree within the CONVENTIONS §7 budget (tested).
//!
//! Before tracing, the device's precision is qualified for the system
//! (`crate::gpu::GpuDevice::qualify`). A failed qualification, a system the
//! packer cannot represent or a device error falls back to the CPU
//! reference and is reported (`AnalysisInput::report_backend`, which puts
//! the reason into `RunInfo.warnings`). Cancellation is never a fallback.

use std::sync::Arc;

use mklight_analysis::spot::trace_field_rays;
use mklight_analysis::spot::{FieldRays, ImageHit, ImageRay, RayBatch as SpotBatch};
use mklight_analysis::util::BackendInfo as AnalysisBackend;
use mklight_core::job::Precision;
use mklight_core::{JobContext, MkError, MkResult, Vec3};
use mklight_gpu::layout::MAX_WAVELENGTHS;
use mklight_gpu::{
    ComputeBackend, Df32, Job, JobOutput, PackOptions, PackedSystem, Radiometry, RayBatch,
    RecordMode, TraceRequest, TraceResult,
};
use mklight_trace::{SemiDiameterSettings, SequentialTracer};
use rayon::prelude::*;

use super::{AnalysisInput, BackendUse};
use crate::error::EngineResult;
use crate::gpu::GpuDevice;

/// Analyses with a GPU path (listed in a GPU backend's capabilities).
pub const GPU_ANALYSES: &[&str] = &[
    "spot_diagram",
    "encircled_energy",
    "geometric_mtf",
    "best_focus",
    "ray_grid",
    "optimize",
    "tolerance",
];

/// Smallest ray grid (`fields × wavelengths × pupil`) that `auto` traces on
/// a GPU; smaller grids run on the CPU reference (a deliberate fallback,
/// reported in `BackendReport::fallback_reason` without a warning). An
/// explicitly requested GPU backend is used for any size.
///
/// The GPU path moves only the trace to the device: the host still
/// generates and aims every ray with the f64 reference (in parallel), and
/// packing, upload and two dispatches cost about 1–2 ms. Measured on
/// Apple-silicon Metal with double_gauss (`ray_grid`), the CPU reference is
/// faster at 11 k rays and the GPU at 99 k (docs/RESULTS.md §5), so below
/// this size `auto` would gain nothing by switching to f32 results.
pub const AUTO_MIN_GPU_RAYS: usize = 1 << 16;

/// Image hits of a ray grid, from any backend.
#[derive(Clone, Debug, PartialEq)]
pub struct GridHits {
    /// `fields × wavelengths × pupil`, row-major; `None` for a ray that was
    /// not generated, vignetted, missed or failed.
    pub rays: Vec<Option<ImageHit>>,
    /// Chief ray (unclipped) per `(field, wavelength)` of `chief_wavelengths`.
    pub chiefs: Vec<Option<ImageHit>>,
    /// The wavelengths of `chiefs`: the grid's wavelengths plus the primary.
    pub chief_wavelengths: Vec<usize>,
}

impl GridHits {
    fn chief(&self, field_pos: usize, wl: usize) -> Option<ImageHit> {
        let nw = self.chief_wavelengths.len();
        let k = self.chief_wavelengths.iter().position(|&w| w == wl)?;
        self.chiefs[field_pos * nw + k]
    }
}

fn chief_wavelengths(tracer: &SequentialTracer, wls: &[usize]) -> Vec<usize> {
    let mut all = wls.to_vec();
    all.push(tracer.system().primary_wavelength);
    all.sort_unstable();
    all.dedup();
    all
}

fn run_trace(
    backend: &dyn ComputeBackend,
    req: TraceRequest,
    ctx: &JobContext,
) -> MkResult<TraceResult> {
    match backend.run(Job::Trace(Box::new(req)), ctx)?.output {
        JobOutput::Trace(t) => Ok(*t),
        other => Err(MkError::Backend(format!("trace job returned {other:?}"))),
    }
}

fn hit(r: &TraceResult, i: usize) -> Option<ImageHit> {
    r.ray_status(i).is_ok().then(|| ImageHit {
        point: [r.x[i] as f64, r.y[i] as f64],
        dir: Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64),
    })
}

/// Rays per parallel generation block (cancellation and progress between
/// blocks).
const GEN_BLOCK: usize = 1 << 16;

/// Generate (and aim) the grid rays of one wavelength chunk with the f64
/// reference, in parallel blocks, straight into the batch's double-float
/// planes (as `RayBatch::push_f64`). Batch ray `i` is grid item `i`, `(field,
/// grid wavelength, pupil)` in row-major order, so the batch, and hence the
/// result, does not depend on the thread count. A ray that cannot be
/// generated is traced as a placeholder (the axial ray) and flagged `false`
/// in the returned mask; its result is ignored.
fn generate_grid(
    tracer: &SequentialTracer,
    fields: &[usize],
    grid_wls: &[(u32, usize, usize)],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<(RayBatch, Vec<bool>)> {
    let (ng, np) = (grid_wls.len(), pupil.len());
    let n = fields.len() * ng * np;
    let mut b = RayBatch::with_capacity(n);
    let mut ok = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        ctx.check()?;
        let end = (start + GEN_BLOCK).min(n);
        // Per ray: [x, y, z, l, m, n] as (hi, lo) pairs, and the packed
        // wavelength.
        let recs: Vec<(Option<[Df32; 6]>, u32)> = (start..end)
            .into_par_iter()
            .map(|i| {
                let (fi, gi, pi) = (i / (ng * np), (i / np) % ng, i % np);
                let (pw, wl, _) = grid_wls[gi];
                let pp = pupil[pi];
                let r = tracer
                    .generate(tracer.field(fields[fi]), wl, pp[0], pp[1])
                    .ok()
                    .map(|g| {
                        let (p, d) = (g.ray.pos, g.ray.dir);
                        [p.x, p.y, p.z, d.x, d.y, d.z].map(Df32::from_f64)
                    });
                (r, pw)
            })
            .collect();
        let axial = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0].map(Df32::from_f64);
        let planes: [(&mut Vec<f32>, &mut Vec<f32>); 6] = [
            (&mut b.x, &mut b.x_lo),
            (&mut b.y, &mut b.y_lo),
            (&mut b.z, &mut b.z_lo),
            (&mut b.l, &mut b.l_lo),
            (&mut b.m, &mut b.m_lo),
            (&mut b.n, &mut b.n_lo),
        ];
        for (k, (hi, lo)) in planes.into_iter().enumerate() {
            let v = |r: &(Option<[Df32; 6]>, u32)| r.0.as_ref().unwrap_or(&axial)[k];
            hi.par_extend(recs.par_iter().map(|r| v(r).hi));
            lo.par_extend(recs.par_iter().map(|r| v(r).lo));
        }
        b.wavelength.par_extend(recs.par_iter().map(|r| r.1));
        ok.par_extend(recs.par_iter().map(|r| r.0.is_some()));
        start = end;
        ctx.report(start as f64 / n as f64, "generating rays");
    }
    b.opl_hi = vec![0.0; n];
    b.opl_lo = vec![0.0; n];
    ctx.check()?;
    Ok((b, ok))
}

/// Trace a ray grid on `backend` (module docs). Wavelengths are packed in
/// chunks of at most `MAX_WAVELENGTHS`, one upload each.
pub fn trace_grid_gpu(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<GridHits> {
    let sys = tracer.system();
    let image = sys.image_index() as u32;
    let cw = chief_wavelengths(tracer, wls);
    let (nf, nw, np, nc) = (fields.len(), wls.len(), pupil.len(), cw.len());
    let mut out = GridHits {
        rays: vec![None; nf * nw * np],
        chiefs: vec![None; nf * nc],
        chief_wavelengths: cw.clone(),
    };
    let hints = tracer
        .semi_diameters(&SemiDiameterSettings::default())
        .values;
    let chunks: Vec<&[usize]> = cw.chunks(MAX_WAVELENGTHS).collect();
    let k = chunks.len() as f64;
    for (ci, chunk) in chunks.iter().enumerate() {
        let sub = ctx.subrange(ci as f64 / k, (ci + 1) as f64 / k);
        sub.check()?;
        sub.report(0.0, "packing the system");
        let packed = PackedSystem::pack(
            sys,
            &PackOptions {
                wavelengths: Some(chunk.to_vec()),
                radius_hints: Some(hints.clone()),
            },
        )?;
        // Rays (only the grid's wavelengths) and chiefs (all of the chunk),
        // with their slot in `out`.
        let mut chiefs = RayBatch::with_capacity(nf * chunk.len());
        let mut chief_slot = Vec::new();
        sub.report(0.0, "generating rays");
        for (fi, &f) in fields.iter().enumerate() {
            let field = tracer.field(f);
            for (pw, &wl) in chunk.iter().enumerate() {
                if let Ok(g) = tracer.generate(field, wl, 0.0, 0.0) {
                    let (p, d) = (g.ray.pos, g.ray.dir);
                    chiefs.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], pw as u32);
                    let c = cw.iter().position(|&w| w == wl).unwrap_or(0);
                    chief_slot.push(fi * nc + c);
                }
            }
        }
        // The grid's wavelengths in this chunk: (packed index, system index,
        // position in `wls`).
        let grid_wls: Vec<(u32, usize, usize)> = chunk
            .iter()
            .enumerate()
            .filter_map(|(pw, &wl)| {
                let wi = wls.iter().position(|&w| w == wl)?;
                Some((pw as u32, wl, wi))
            })
            .collect();
        let (rays, generated) =
            generate_grid(tracer, fields, &grid_wls, pupil, &sub.subrange(0.0, 0.3))?;
        let handle = backend.upload_system(&packed)?;
        let request = |b: RayBatch, clip: bool| {
            let mut r = TraceRequest::new(handle, b, 0, image);
            r.apply_apertures = clip;
            r.record = RecordMode::Final;
            r.radiometry = Radiometry::Off;
            r
        };
        let traced = (|| {
            let g = if rays.is_empty() {
                None
            } else {
                Some(run_trace(
                    backend,
                    request(rays, true),
                    &sub.subrange(0.3, 0.9),
                )?)
            };
            let c = if chiefs.is_empty() {
                None
            } else {
                Some(run_trace(
                    backend,
                    request(chiefs, false),
                    &sub.subrange(0.9, 1.0),
                )?)
            };
            Ok::<_, MkError>((g, c))
        })();
        let _ = backend.release_systems(handle);
        let (g, c) = traced?;
        if let Some(g) = g {
            // Batch ray i is item (field, grid wavelength, pupil) of the
            // chunk's grid; its slot in `out` uses the position in `wls`.
            let ng = grid_wls.len();
            for (i, _) in generated.iter().enumerate().filter(|(_, ok)| **ok) {
                let (fi, gi, pi) = (i / (ng * np), (i / np) % ng, i % np);
                out.rays[(fi * nw + grid_wls[gi].2) * np + pi] = hit(&g, i);
            }
        }
        if let Some(c) = c {
            for (i, &slot) in chief_slot.iter().enumerate() {
                out.chiefs[slot] = hit(&c, i);
            }
        }
        sub.report(1.0, "GPU trace");
    }
    ctx.check()?;
    Ok(out)
}

/// Reduce grid hits to the spot batch every ray-grid statistic uses
/// (the same reduction as `mklight_analysis::spot::trace_field_rays`).
pub fn batch_from_hits(
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    np: usize,
    hits: &GridHits,
    warnings: &mut Vec<String>,
) -> MkResult<SpotBatch> {
    let sys = tracer.system();
    let primary = sys.primary_wavelength;
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();
    let mut out = Vec::with_capacity(fields.len());
    for (fi, &f) in fields.iter().enumerate() {
        let pc = hits.chief(fi, primary);
        let wl_chiefs: Vec<_> = wls.iter().map(|&w| hits.chief(fi, w)).collect();
        let nw = wls.len();
        let block: Vec<ImageRay> = (0..nw * np)
            .map(|j| ImageRay {
                wavelength: wls[j / np],
                hit: hits.rays[fi * nw * np + j],
            })
            .collect();
        out.push(FieldRays::from_image_rays(
            f, wls, &weights, pc, &wl_chiefs, &block, warnings,
        )?);
    }
    Ok(SpotBatch {
        wavelengths: wls.to_vec(),
        fields: out,
    })
}

/// The device of the selected backend, if it is a GPU.
pub fn device(input: &AnalysisInput<'_>) -> Option<Arc<GpuDevice>> {
    input
        .device
        .clone()
        .and_then(|d| d.downcast::<GpuDevice>().ok())
}

/// The backend label the `mklight-analysis` reductions record (their
/// `RunInfo` is replaced by the engine's).
pub fn analysis_backend(used: &BackendUse) -> AnalysisBackend {
    AnalysisBackend {
        name: used.id.clone(),
        precision: used.precision,
    }
}

/// Where a ray grid was traced.
pub enum Traced {
    /// On the GPU (`BackendUse` already reported).
    Gpu(GridHits),
    /// On the CPU reference.
    Cpu,
}

/// Trace a ray grid on the selected GPU when there is one and it qualifies
/// for the system; otherwise report the fallback and return `Traced::Cpu`
/// (the caller then runs the CPU path). Progress: qualification takes the
/// first 10 % of `ctx`.
pub fn try_gpu_grid(
    input: &AnalysisInput<'_>,
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
) -> EngineResult<Traced> {
    let Some(dev) = device(input) else {
        return Ok(Traced::Cpu);
    };
    let cpu = |reason: String, q: Option<serde_json::Value>| BackendUse {
        id: crate::backend::Backend::CpuReference.id(),
        precision: Precision::F64Reference,
        device: None,
        fallback_reason: Some(reason),
        qualification: q,
        deliberate: false,
    };
    let rays = fields.len() * wls.len() * pupil.len();
    let auto = crate::backend::Backend::is_auto_request(&input.requested_backend);
    if auto && rays < AUTO_MIN_GPU_RAYS {
        input.report_backend(BackendUse {
            deliberate: true,
            ..cpu(
                format!(
                    "auto: a grid of {rays} rays is below the GPU threshold of \
                     {AUTO_MIN_GPU_RAYS}; the CPU reference is as fast there"
                ),
                None,
            )
        });
        return Ok(Traced::Cpu);
    }
    let key = format!(
        "{}:{}",
        input.snapshot.hash().to_hex(),
        input.library_hash()
    );
    ctx.report(0.0, "GPU precision qualification");
    let q = dev.qualify(&key, tracer, &ctx.subrange(0.0, 0.1))?;
    let qv = serde_json::to_value(q.as_ref()).ok();
    if !q.passed {
        let why = q
            .reason
            .clone()
            .unwrap_or_else(|| "failed precision qualification".into());
        input.report_backend(cpu(why, qv));
        return Ok(Traced::Cpu);
    }
    match trace_grid_gpu(
        dev.backend.as_ref(),
        tracer,
        fields,
        wls,
        pupil,
        &ctx.subrange(0.1, 1.0),
    ) {
        Ok(h) => {
            input.report_backend(BackendUse {
                id: dev.id.clone(),
                precision: Precision::F32Gpu,
                device: Some(dev.device_name()),
                fallback_reason: None,
                qualification: qv,
                deliberate: false,
            });
            Ok(Traced::Gpu(h))
        }
        Err(MkError::Cancelled) => Err(MkError::Cancelled.into()),
        Err(e) => {
            input.report_backend(cpu(format!("GPU trace failed: {e}"), qv));
            Ok(Traced::Cpu)
        }
    }
}

/// The spot batch of a ray grid, traced on the selected GPU when possible
/// (module docs), else with the CPU reference. Returns the batch and the
/// backend label for the reductions.
pub fn field_rays(
    input: &AnalysisInput<'_>,
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
    warnings: &mut Vec<String>,
) -> EngineResult<(SpotBatch, AnalysisBackend)> {
    match try_gpu_grid(input, tracer, fields, wls, pupil, &ctx.subrange(0.0, 0.9))? {
        Traced::Gpu(h) => {
            let b = batch_from_hits(tracer, fields, wls, pupil.len(), &h, warnings)?;
            let label = AnalysisBackend {
                name: device(input).map_or_else(String::new, |d| d.id.clone()),
                precision: Precision::F32Gpu,
            };
            Ok((b, label))
        }
        Traced::Cpu => {
            let b = trace_field_rays(
                tracer,
                fields,
                wls,
                pupil,
                &ctx.subrange(0.0, 0.9),
                warnings,
            )?;
            Ok((b, AnalysisBackend::cpu_reference()))
        }
    }
}
