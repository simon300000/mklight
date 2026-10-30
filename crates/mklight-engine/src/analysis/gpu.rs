//! GPU paths of the ray-grid analyses.
//!
//! The analyses in [`GPU_ANALYSES`] trace a `fields × wavelengths × pupil`
//! grid plus one chief ray per (field, wavelength) and reduce the image
//! hits. On a GPU backend the host generates and aims the rays with the f64
//! reference (`SequentialTracer::generate`), the device traces them with the
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
    ComputeBackend, Job, JobOutput, PackOptions, PackedSystem, Radiometry, RayBatch, RecordMode,
    TraceRequest, TraceResult,
};
use mklight_trace::{SemiDiameterSettings, SequentialTracer};

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
];

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
        let mut rays = RayBatch::with_capacity(nf * chunk.len() * np);
        let mut ray_slot = Vec::new();
        let mut chiefs = RayBatch::with_capacity(nf * chunk.len());
        let mut chief_slot = Vec::new();
        sub.report(0.05, "generating rays");
        for (fi, &f) in fields.iter().enumerate() {
            sub.check()?;
            let field = tracer.field(f);
            for (pw, &wl) in chunk.iter().enumerate() {
                if let Ok(g) = tracer.generate(field, wl, 0.0, 0.0) {
                    let (p, d) = (g.ray.pos, g.ray.dir);
                    chiefs.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], pw as u32);
                    let c = cw.iter().position(|&w| w == wl).unwrap_or(0);
                    chief_slot.push(fi * nc + c);
                }
                let Some(wi) = wls.iter().position(|&w| w == wl) else {
                    continue;
                };
                for (pi, pp) in pupil.iter().enumerate() {
                    if let Ok(g) = tracer.generate(field, wl, pp[0], pp[1]) {
                        let (p, d) = (g.ray.pos, g.ray.dir);
                        rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], pw as u32);
                        ray_slot.push((fi * nw + wi) * np + pi);
                    }
                }
            }
        }
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
            for (i, &slot) in ray_slot.iter().enumerate() {
                out.rays[slot] = hit(&g, i);
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
    };
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
