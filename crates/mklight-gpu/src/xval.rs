//! Cross-validation of a backend's trace kernel against the CPU f64
//! reference (`mklight_trace`), and the precision qualification built on it.
//!
//! [`cross_validate`] generates rays with the reference tracer (field and
//! pupil mapping, ray aiming in f64), rounds them once to f32
//! ([`RayBatch::push_f64`]), traces them on the backend and with
//! `mklight_trace::trace_ray`, and compares, per ray that both traces
//! complete:
//!
//! * position (Euclidean distance, mm) and direction cosines (Euclidean
//!   distance of the unit vectors) on every surface, from per-surface
//!   records (coordinate breaks, which only re-express the previous hit,
//!   are skipped), and at the image;
//! * OPD in waves: the backend's OPD is formed on the CPU from the returned
//!   OPLs with the same reference sphere as `mklight_trace::opd` (centre and
//!   OPL from the backend's own chief ray, exit pupil from the paraxial
//!   data), and compared with the reference OPD;
//! * bulk transmittance;
//! * survival: both traces must end with the same status word. A difference
//!   is excused only for rays whose f64 trace passes within
//!   `budget.aperture_edge_mm` of an aperture edge, or within
//!   [`XvalSettings::tir_band`] of the TIR condition `1 − μ² sin² i = 0`;
//!   such rays are counted separately.
//!
//! [`qualify`] runs a small cross-validation for one system and returns a
//! [`Qualification`] (pass/fail, measured errors, reason); the sampled
//! maxima must stay within a margin (a third) of the budget, because rays
//! outside the sample can be worse. A caller that gets a failure runs the
//! CPU reference instead and records the decision
//! with [`Qualification::record`], which writes the backend, precision and
//! fallback reason into the result's `RunInfo` (CONVENTIONS §7: never
//! silently degraded numbers).

use mklight_core::job::{Precision, RunInfo};
use mklight_core::model::ApertureShape;
use mklight_core::resolved::ResolvedSurface;
use mklight_core::{JobContext, MkError, MkResult, Vec3};
use mklight_trace::surface::surface_normal;
use mklight_trace::{
    Compensated, FailReason, OpdReference, PupilSampling, Ray, SemiDiameterSettings,
    SequentialTracer, TraceOptions, TraceStatus, trace_ray_recorded,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backend::{ComputeBackend, ErrorBudget, Job, JobOutput};
use crate::batch::{
    NewtonSettings, RECORD_FLOATS, Radiometry, RayBatch, RayStatus, RecordMode, TraceRequest,
    TraceResult,
};
use crate::consts::*;
use crate::layout::{MAX_WAVELENGTHS, PackOptions, PackedSystem};

/// What to cross-validate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct XvalSettings {
    pub pupil: PupilSampling,
    /// Further pupil samples traced in addition to `pupil` (per field and
    /// wavelength). The qualification adds dense tangential and sagittal
    /// fans: they cross narrow pupil zones (near-critical refraction, where
    /// f32 errors peak) that a coarse 2-D pattern can step over.
    #[serde(default)]
    pub pupil_extra: Vec<PupilSampling>,
    /// Indices into the tracer's prepared fields; `None` = all.
    pub fields: Option<Vec<usize>>,
    /// System wavelength indices (validated in chunks of
    /// `MKL_MAX_WAVELENGTHS`); `None` = all.
    pub wavelengths: Option<Vec<usize>>,
    /// Apply apertures (vignetting) to the pupil rays. Chief rays are always
    /// traced unclipped, as for the reference OPD.
    pub clip: bool,
    pub radiometry: Radiometry,
    pub newton: NewtonSettings,
    pub budget: ErrorBudget,
    /// Survival may differ for rays with `|1 − μ² sin² i| <` this at a
    /// refracting surface.
    pub tir_band: f64,
    /// Fraction of `budget` (position, direction, OPD) the measured maxima
    /// must stay within for [`qualify`] to pass: 1 checks the budget
    /// itself; the qualification default, 1/3, leaves room for rays the
    /// small sample does not contain (a 64 × 64 grid has measured up to
    /// ~2× the maxima of the 61-point sample).
    #[serde(default = "one")]
    pub margin: f64,
}

fn one() -> f64 {
    1.0
}

impl Default for XvalSettings {
    /// The full suite: a 64 × 64 pupil grid, every field and wavelength.
    fn default() -> Self {
        XvalSettings {
            pupil: PupilSampling::Grid { n: 64 },
            pupil_extra: Vec::new(),
            fields: None,
            wavelengths: None,
            clip: true,
            radiometry: Radiometry::Unpolarized,
            newton: NewtonSettings::default(),
            budget: ErrorBudget::CONVENTIONS,
            tir_band: 1e-5,
            margin: 1.0,
        }
    }
}

impl XvalSettings {
    /// The small sample [`qualify`] uses, per field and wavelength: 61
    /// hexapolar pupil points (the rim ring included) plus 65-point
    /// tangential and sagittal fans, checked against a third of the budget.
    pub fn qualification() -> Self {
        XvalSettings {
            pupil: PupilSampling::Hexapolar { rings: 4 },
            pupil_extra: vec![PupilSampling::FanY { n: 65 }, PupilSampling::FanX { n: 65 }],
            margin: QUALIFICATION_MARGIN,
            ..Default::default()
        }
    }

    /// `budget` with its position, direction and OPD limits scaled by
    /// `margin` (the survival bands are unchanged).
    pub fn effective_budget(&self) -> ErrorBudget {
        let m = self.margin;
        ErrorBudget {
            position_mm: self.budget.position_mm * m,
            direction_cosine: self.budget.direction_cosine * m,
            opd_waves: self.budget.opd_waves * m,
            ..self.budget
        }
    }
}

/// Default [`XvalSettings::margin`] of [`qualify`].
pub const QUALIFICATION_MARGIN: f64 = 1.0 / 3.0;

/// Measured differences between a backend and the f64 reference.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XvalReport {
    /// `DeviceInfo::backend_id` of the backend.
    pub backend: String,
    /// Rays generated and traced by both.
    pub rays: usize,
    /// Rays that both traces complete (compared quantitatively).
    pub compared: usize,
    /// Of those, rays with an OPD on both sides.
    pub opd_compared: usize,
    /// Largest position difference (mm) on any surface (per-surface
    /// records, image included).
    pub max_position_mm: f64,
    /// Largest direction difference on any surface.
    pub max_direction: f64,
    pub max_opd_waves: f64,
    pub max_transmittance_abs: f64,
    /// Rays with a different outcome outside the excused bands.
    pub survival_mismatches: usize,
    /// Rays with a different outcome inside an aperture-edge or TIR band.
    pub band_mismatches: usize,
    /// (field, wavelength) pairs whose chief ray failed on one side only.
    pub chief_mismatches: usize,
    /// Descriptions of the first few failures, for diagnostics.
    pub notes: Vec<String>,
}

impl XvalReport {
    /// The largest of the measured position, direction and OPD errors as a
    /// fraction of `budget` (1 = at the limit).
    pub fn budget_fraction(&self, budget: &ErrorBudget) -> f64 {
        (self.max_position_mm / budget.position_mm)
            .max(self.max_direction / budget.direction_cosine)
            .max(self.max_opd_waves / budget.opd_waves)
    }

    /// Combine the reports of two disjoint samples (wavelength chunks).
    fn merge(&mut self, o: XvalReport) {
        self.rays += o.rays;
        self.compared += o.compared;
        self.opd_compared += o.opd_compared;
        self.max_position_mm = self.max_position_mm.max(o.max_position_mm);
        self.max_direction = self.max_direction.max(o.max_direction);
        self.max_opd_waves = self.max_opd_waves.max(o.max_opd_waves);
        self.max_transmittance_abs = self.max_transmittance_abs.max(o.max_transmittance_abs);
        self.survival_mismatches += o.survival_mismatches;
        self.band_mismatches += o.band_mismatches;
        self.chief_mismatches += o.chief_mismatches;
        self.notes.extend(o.notes);
        self.notes.truncate(8);
    }

    /// `Err(reason)` when any measured quantity exceeds `budget` or any ray
    /// changes its outcome outside the excused bands.
    pub fn check(&self, budget: &ErrorBudget) -> Result<(), String> {
        let mut why = Vec::new();
        if self.max_position_mm > budget.position_mm {
            why.push(format!(
                "position error {:.3e} mm > {:.1e} mm",
                self.max_position_mm, budget.position_mm
            ));
        }
        if self.max_direction > budget.direction_cosine {
            why.push(format!(
                "direction error {:.3e} > {:.1e}",
                self.max_direction, budget.direction_cosine
            ));
        }
        if self.max_opd_waves > budget.opd_waves {
            why.push(format!(
                "OPD error {:.3e} waves > {:.1e} waves",
                self.max_opd_waves, budget.opd_waves
            ));
        }
        if self.survival_mismatches > 0 || self.chief_mismatches > 0 {
            why.push(format!(
                "{} rays and {} chief rays change survival or status outside the tolerance bands",
                self.survival_mismatches, self.chief_mismatches
            ));
        }
        if self.compared == 0 {
            why.push("no ray completed on both sides".into());
        }
        if why.is_empty() {
            Ok(())
        } else {
            Err(why.join("; "))
        }
    }
}

/// The `MKL_RAY_*` status word of a reference result.
pub fn reference_status_word(st: &TraceStatus) -> u32 {
    let (code, s) = match *st {
        TraceStatus::Ok => (MKL_RAY_OK, 0),
        TraceStatus::Miss { surface } => (MKL_RAY_MISS, surface),
        TraceStatus::Clipped { surface } => (MKL_RAY_VIGNETTED, surface),
        TraceStatus::TotalInternalReflection { surface } => (MKL_RAY_TIR, surface),
        TraceStatus::Failed { surface, reason } => (
            match reason {
                FailReason::NoConvergence => MKL_RAY_NOT_CONVERGED,
                FailReason::OutsideDomain => MKL_RAY_OUTSIDE_DOMAIN,
                FailReason::Degenerate => MKL_RAY_DEGENERATE,
                FailReason::EvanescentOrder => MKL_RAY_EVANESCENT,
                _ => MKL_RAY_NONFINITE,
            },
            surface,
        ),
    };
    code | ((s as u32) << MKL_RAY_SURFACE_SHIFT)
}

/// Distance (mm) of a local point from the clipping edge of a surface,
/// `INFINITY` when it has no aperture.
pub fn aperture_edge_distance(s: &ResolvedSurface, x: f64, y: f64) -> f64 {
    let r = x.hypot(y);
    match (&s.aperture, s.fixed_semi_diameter) {
        (Some(a), _) => match *a {
            ApertureShape::Circular { radius } => (r - radius).abs(),
            ApertureShape::Annular {
                inner_radius,
                outer_radius,
            } => (r - inner_radius).abs().min((r - outer_radius).abs()),
            ApertureShape::Rectangular { half_x, half_y } => {
                let (dx, dy) = (x.abs() - half_x, y.abs() - half_y);
                if dx <= 0.0 && dy <= 0.0 {
                    (-dx).min(-dy)
                } else {
                    dx.max(0.0).hypot(dy.max(0.0))
                }
            }
            ApertureShape::Elliptical { semi_x, semi_y } => {
                ((x / semi_x).hypot(y / semi_y) - 1.0).abs() * semi_x.min(semi_y)
            }
        },
        (None, Some(rr)) => (r - rr).abs(),
        (None, None) => f64::INFINITY,
    }
}

/// One generated ray of the validation sample.
struct Sample {
    field: usize,
    wl: usize,
    /// Packed wavelength index.
    pw: usize,
    ray: Ray,
}

/// `true` when the f64 trace of `ray` passes close to a decision boundary
/// (an aperture edge, or the TIR condition at a refracting surface).
fn in_band(
    tracer: &SequentialTracer,
    s: &Sample,
    opts: &TraceOptions,
    edge: f64,
    tir_band: f64,
) -> bool {
    let sys = tracer.system();
    let (res, rec) = trace_ray_recorded(sys, &s.ray, s.wl, opts);
    let near_edge = |j: usize, p: Vec3| {
        let surf = &sys.surfaces[j];
        !matches!(surf.shape, mklight_core::SurfaceShape::CoordinateBreak)
            && aperture_edge_distance(surf, p.x, p.y) < edge
    };
    let tir_margin = |j: usize, cos_i: f64| {
        let surf = &sys.surfaces[j];
        if surf.is_passive() || surf.is_mirror() || j == sys.image_index() {
            return f64::INFINITY;
        }
        let mu = sys.n_before(j, s.wl) / surf.n_after[s.wl];
        (1.0 - mu * mu * (1.0 - cos_i) * (1.0 + cos_i)).abs()
    };
    for (j, r) in rec.iter().enumerate().skip(1) {
        if near_edge(j, r.point) || tir_margin(j, r.cos_incidence) < tir_band {
            return true;
        }
    }
    if let Some(j) = res.status.surface()
        && j > 0
    {
        if near_edge(j, res.ray.pos) {
            return true;
        }
        if let Some(n) = surface_normal(&sys.surfaces[j].shape, res.ray.pos) {
            let c = res.ray.dir.dot(n).abs().min(1.0);
            if tir_margin(j, c) < tir_band {
                return true;
            }
        }
    }
    false
}

/// The backend's result for ray `i` as a reference-style result.
fn as_reference(r: &TraceResult, i: usize) -> mklight_trace::TraceResult {
    let p = Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64);
    let d = Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64);
    mklight_trace::TraceResult {
        status: if r.ray_status(i).is_ok() {
            TraceStatus::Ok
        } else {
            TraceStatus::Failed {
                surface: r.last_surface[i] as usize,
                reason: FailReason::NonFinite,
            }
        },
        ray: Ray::new(p, d),
        surface: r.last_surface[i] as usize,
        opl: Compensated {
            hi: r.opl_hi[i] as f64,
            lo: r.opl_lo[i] as f64,
        },
        transmittance: r.transmittance[i] as f64,
    }
}

/// The reference sphere of a backend chief ray (`SequentialTracer::opd_reference`
/// with the backend's chief point, direction and OPL).
fn backend_reference(
    tracer: &SequentialTracer,
    chief: &mklight_trace::TraceResult,
    wl: usize,
) -> Option<OpdReference> {
    let sys = tracer.system();
    let img = sys.image_index();
    let z_xp = tracer.paraxial().at(wl).exit_pupil_position;
    let s_e = if z_xp.is_finite() {
        if chief.ray.dir.z == 0.0 {
            return None;
        }
        (z_xp - chief.ray.pos.z) / chief.ray.dir.z
    } else {
        f64::INFINITY
    };
    Some(OpdReference {
        chief_opl: chief.opl,
        center: chief.ray.pos,
        chief_dir: chief.ray.dir,
        s_exit_pupil: s_e,
        n_image: sys.n_before(img, wl),
        wavelength_um: sys.wavelengths_um[wl],
    })
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

/// Cross-validate `backend` against the reference for `tracer`'s system
/// (module docs). Systems with more than `MKL_MAX_WAVELENGTHS` wavelengths
/// are validated in packed chunks (as the packer splits them) and the
/// reports merged. Fails with the packer's or backend's error when the
/// system cannot run on the backend at all, and with `Cancelled` when `ctx`
/// is cancelled.
pub fn cross_validate(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    settings: &XvalSettings,
    ctx: &JobContext,
) -> MkResult<XvalReport> {
    let sys = tracer.system();
    let wls: Vec<usize> = settings
        .wavelengths
        .clone()
        .unwrap_or_else(|| (0..sys.wavelengths_um.len()).collect());
    let chunks: Vec<&[usize]> = wls.chunks(MAX_WAVELENGTHS).collect();
    let k = chunks.len().max(1) as f64;
    let mut report = XvalReport {
        backend: backend.info().backend_id(),
        ..Default::default()
    };
    for (i, c) in chunks.iter().enumerate() {
        let sub = ctx.subrange(i as f64 / k, (i + 1) as f64 / k);
        report.merge(cross_validate_chunk(backend, tracer, settings, c, &sub)?);
    }
    ctx.report(1.0, "cross-validation");
    Ok(report)
}

/// One packed chunk of a cross-validation with its sample and chief rays.
struct PreparedValidation {
    packed: PackedSystem,
    samples: Vec<Sample>,
    chiefs: Vec<Sample>,
    image: u32,
}
fn prepare_validation(
    tracer: &SequentialTracer,
    settings: &XvalSettings,
    wls: &[usize],
    ctx: &JobContext,
) -> MkResult<PreparedValidation> {
    let sys = tracer.system();
    let wls = wls.to_vec();
    let fields: Vec<usize> = settings
        .fields
        .clone()
        .unwrap_or_else(|| (0..tracer.fields().len()).collect());
    let mut pupil = settings.pupil.points();
    for p in &settings.pupil_extra {
        pupil.extend(p.points());
    }
    ctx.report(0.0, "cross-validation: packing");
    let hints = tracer
        .semi_diameters(&SemiDiameterSettings::default())
        .values;
    let packed = PackedSystem::pack(
        sys,
        &PackOptions {
            wavelengths: Some(wls.clone()),
            radius_hints: Some(hints),
        },
    )?;
    let image = sys.image_index() as u32;

    // Rays: pupil samples and one chief ray per (field, wavelength).
    let mut samples = Vec::new();
    let mut chiefs = Vec::new();
    for &f in &fields {
        let field = tracer.field(f);
        for (pw, &wl) in wls.iter().enumerate() {
            for p in &pupil {
                if let Ok(g) = tracer.generate(field, wl, p[0], p[1]) {
                    samples.push(Sample {
                        field: f,
                        wl,
                        pw,
                        ray: g.ray,
                    });
                }
            }
            if let Ok(g) = tracer.generate(field, wl, 0.0, 0.0) {
                chiefs.push(Sample {
                    field: f,
                    wl,
                    pw,
                    ray: g.ray,
                });
            }
        }
    }
    Ok(PreparedValidation {
        packed,
        samples,
        chiefs,
        image,
    })
}

/// [`cross_validate`] for at most `MKL_MAX_WAVELENGTHS` wavelengths (one
/// packed system).
fn cross_validate_chunk(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    settings: &XvalSettings,
    wls: &[usize],
    ctx: &JobContext,
) -> MkResult<XvalReport> {
    let PreparedValidation {
        packed,
        samples,
        chiefs,
        image,
    } = prepare_validation(tracer, settings, wls, ctx)?;
    let handle = backend.upload_system(&packed)?;
    let batch = |v: &[Sample]| {
        let mut b = RayBatch::with_capacity(v.len());
        for s in v {
            let (p, d) = (s.ray.pos, s.ray.dir);
            b.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], s.pw as u32);
        }
        b
    };
    let request = |rays: RayBatch, clip: bool, record: RecordMode| {
        let mut r = TraceRequest::new(handle, rays, 0, image);
        r.apply_apertures = clip;
        r.record = record;
        r.radiometry = settings.radiometry;
        r.newton = settings.newton;
        r
    };
    ctx.report(0.05, "cross-validation: backend trace");
    let traced = (|| {
        let g = run_trace(
            backend,
            request(batch(&samples), settings.clip, RecordMode::AllSurfaces),
            &ctx.subrange(0.05, 0.5),
        )?;
        let c = run_trace(
            backend,
            request(batch(&chiefs), false, RecordMode::Final),
            &ctx.subrange(0.5, 0.55),
        )?;
        Ok::<_, MkError>((g, c))
    })();
    let _ = backend.release_systems(handle);
    let (gpu, gpu_chief) = traced?;
    compare_validation(
        backend, tracer, settings, &samples, &chiefs, &gpu, &gpu_chief, 0, 0, ctx,
    )
}

#[allow(clippy::too_many_arguments)]
fn compare_validation(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    settings: &XvalSettings,
    samples: &[Sample],
    chiefs: &[Sample],
    gpu: &TraceResult,
    gpu_chief: &TraceResult,
    first: usize,
    chief_first: usize,
    ctx: &JobContext,
) -> MkResult<XvalReport> {
    let sys = tracer.system();
    ctx.check()?;
    ctx.report(0.55, "cross-validation: reference trace");

    // Reference sphere pairs per (field, wavelength): (reference, backend).
    let mut refs = std::collections::HashMap::new();
    let mut report = XvalReport {
        backend: backend.info().backend_id(),
        rays: samples.len(),
        ..Default::default()
    };
    for (k, c) in chiefs.iter().enumerate() {
        let r = tracer.opd_reference(tracer.field(c.field), c.wl).ok();
        let g = if gpu_chief.ray_status(chief_first + k).is_ok() {
            backend_reference(tracer, &as_reference(gpu_chief, chief_first + k), c.wl)
        } else {
            None
        };
        if r.is_some() != g.is_some() {
            report.chief_mismatches += 1;
            report.notes.push(format!(
                "field {} wavelength {}: chief ray reference {:?}, backend {:?}",
                c.field,
                c.wl,
                r.is_some(),
                gpu_chief.ray_status(chief_first + k)
            ));
        }
        refs.insert((c.field, c.wl), (r, g));
    }

    let opts = TraceOptions {
        clip: settings.clip,
        ..tracer.options()
    };
    let budget = settings.budget;
    #[derive(Default)]
    struct Acc {
        compared: usize,
        opd: usize,
        pos: f64,
        dir: f64,
        opdw: f64,
        trans: f64,
        surv: usize,
        band: usize,
        notes: Vec<String>,
    }
    let acc = samples
        .par_iter()
        .enumerate()
        .fold(Acc::default, |mut a, (i, s)| {
            let mut rec = Vec::new();
            let r = mklight_trace::trace_ray(sys, &s.ray, s.wl, &opts, Some(&mut rec));
            let want = reference_status_word(&r.status);
            let got = gpu.status[first + i];
            if want != got {
                if in_band(tracer, s, &opts, budget.aperture_edge_mm, settings.tir_band) {
                    a.band += 1;
                } else {
                    a.surv += 1;
                    if a.notes.len() < 4 {
                        a.notes.push(format!(
                            "field {} wl {} ray {:?}: reference {:?}, backend {:?}",
                            s.field,
                            s.wl,
                            s.ray,
                            r.status,
                            RayStatus::decode(got)
                        ));
                    }
                }
                return a;
            }
            if !r.status.is_ok() {
                return a;
            }
            let g = as_reference(gpu, first + i);
            a.compared += 1;
            let dp = (g.ray.pos - r.ray.pos).norm();
            let dd = (g.ray.dir - r.ray.dir).norm();
            a.pos = a.pos.max(dp);
            a.dir = a.dir.max(dd);
            // Every surface on the way (CONVENTIONS §7: position on any
            // surface), from the per-surface records. A coordinate break
            // records the previous hit in its own frame, which can be far
            // from its vertex (400 mm in the Newtonian), where the f32
            // record alone rounds by more than the budget: skipped.
            let n = gpu.len();
            for (j, want) in rec.iter().enumerate().skip(1) {
                if matches!(
                    sys.surfaces[j].shape,
                    mklight_core::SurfaceShape::CoordinateBreak
                ) {
                    continue;
                }
                let k = (j * n + first + i) * RECORD_FLOATS;
                let v = |o: usize| {
                    Vec3::new(
                        gpu.recorded[k + o] as f64,
                        gpu.recorded[k + o + 1] as f64,
                        gpu.recorded[k + o + 2] as f64,
                    )
                };
                a.pos = a.pos.max((v(0) - want.point).norm());
                a.dir = a.dir.max((v(3) - want.dir).norm());
            }
            a.trans = a.trans.max((g.transmittance - r.transmittance).abs());
            if let Some((Some(rr), Some(gr))) = refs.get(&(s.field, s.wl))
                && let (Some(o_ref), Some(o_gpu)) = (rr.opd_waves(&r), gr.opd_waves(&g))
            {
                a.opd += 1;
                a.opdw = a.opdw.max((o_gpu - o_ref).abs());
            }
            a
        })
        .reduce(Acc::default, |mut a, b| {
            a.compared += b.compared;
            a.opd += b.opd;
            a.pos = a.pos.max(b.pos);
            a.dir = a.dir.max(b.dir);
            a.opdw = a.opdw.max(b.opdw);
            a.trans = a.trans.max(b.trans);
            a.surv += b.surv;
            a.band += b.band;
            a.notes.extend(b.notes);
            a
        });
    ctx.check()?;
    report.compared = acc.compared;
    report.opd_compared = acc.opd;
    report.max_position_mm = acc.pos;
    report.max_direction = acc.dir;
    report.max_opd_waves = acc.opdw;
    report.max_transmittance_abs = acc.trans;
    report.survival_mismatches = acc.surv;
    report.band_mismatches = acc.band;
    report.notes.extend(acc.notes.into_iter().take(4));
    ctx.report(1.0, "cross-validation: chunk");
    Ok(report)
}

/// Outcome of [`qualify`]: whether a backend may run traces of one system,
/// with the measured evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qualification {
    /// `DeviceInfo::backend_id` of the candidate backend.
    pub backend: String,
    /// The candidate's `PrecisionDescriptor::kernel_build`.
    pub kernel_build: String,
    pub passed: bool,
    /// The full budget (CONVENTIONS §7) the results must meet.
    pub budget: ErrorBudget,
    /// Fraction of `budget` the sampled maxima had to stay within.
    pub margin: f64,
    /// The largest measured error as a fraction of `budget` (position,
    /// direction or OPD; `None` when nothing was measured). Passing needs
    /// at most `margin`.
    pub budget_fraction: Option<f64>,
    /// The measurements (absent when the trace could not run at all).
    pub report: Option<XvalReport>,
    /// Why the backend is not used (`None` when it passed).
    pub reason: Option<String>,
}

impl Qualification {
    /// Record the decision in a result's `RunInfo`: on success the backend
    /// and `Precision::F32Gpu`; on failure the CPU reference
    /// (`"cpu-f64"`, `Precision::F64Reference`) and a warning naming the
    /// rejected backend and the reason. The qualification itself goes into
    /// `settings["backend_qualification"]`.
    pub fn record(&self, info: &mut RunInfo) {
        if self.passed {
            info.backend = self.backend.clone();
            info.precision = Precision::F32Gpu;
        } else {
            info.backend = "cpu-f64".into();
            info.precision = Precision::F64Reference;
            info.warnings.push(format!(
                "GPU backend {} not used ({}); computed with the CPU f64 reference",
                self.backend,
                self.reason.as_deref().unwrap_or("failed qualification")
            ));
        }
        let q = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        match &mut info.settings {
            serde_json::Value::Object(m) => {
                m.insert("backend_qualification".into(), q);
            }
            other => {
                let old = std::mem::take(other);
                *other = serde_json::json!({ "settings": old, "backend_qualification": q });
            }
        }
    }
}

/// Precision qualification of `backend` for `tracer`'s system: a small
/// cross-validation ([`XvalSettings::qualification`] unless given) whose
/// maxima must stay within `margin` × the budget (a third by default), so
/// that rays outside the sample, which can show up to about twice the
/// sampled error, still meet the full budget. Systems with more than 16
/// wavelengths are qualified chunk by chunk.
///
/// A backend that cannot run the system (unsupported job, unpackable
/// model, backend error) fails with that reason. The only error is
/// `MkError::Cancelled`: a cancelled qualification is not a verdict, and
/// the caller should stop rather than fall back.
pub fn qualify(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    settings: Option<&XvalSettings>,
    ctx: &JobContext,
) -> MkResult<Qualification> {
    let default = XvalSettings::qualification();
    let settings = settings.unwrap_or(&default);
    ctx.check()?;
    verdict(
        backend,
        settings,
        cross_validate(backend, tracer, settings, ctx),
    )
}

fn verdict(
    backend: &dyn ComputeBackend,
    settings: &XvalSettings,
    result: MkResult<XvalReport>,
) -> MkResult<Qualification> {
    let mut q = Qualification {
        backend: backend.info().backend_id(),
        kernel_build: backend.precision().kernel_build.clone(),
        passed: false,
        budget: settings.budget,
        margin: settings.margin,
        budget_fraction: None,
        report: None,
        reason: None,
    };
    match result {
        Ok(r) => {
            q.budget_fraction = Some(r.budget_fraction(&settings.budget));
            match r.check(&settings.effective_budget()) {
                Ok(()) => q.passed = true,
                Err(why) => {
                    q.reason = Some(format!(
                        "precision qualification failed ({} of the budget required): {why}",
                        fraction_label(settings.margin)
                    ))
                }
            }
            q.report = Some(r);
        }
        Err(MkError::Cancelled) => return Err(MkError::Cancelled),
        Err(e) => q.reason = Some(format!("cannot run on this backend: {e}")),
    }
    Ok(q)
}

fn fraction_label(m: f64) -> String {
    if (m - 1.0 / 3.0).abs() < 1e-12 {
        "1/3".into()
    } else {
        format!("{m}")
    }
}

/// Qualify changed candidates in stable input order, sharing system registration
/// and trace submissions. Every candidate retains its own f64 geometry/OPD
/// comparison; this establishes no parameter envelope. Groups are bounded by
/// 32 systems and common record geometry, keeping the all-surface scratch small.
pub fn qualify_many(
    backend: &dyn ComputeBackend,
    tracers: &[&SequentialTracer],
    settings: &XvalSettings,
    ctx: &JobContext,
) -> MkResult<Vec<Qualification>> {
    let mut result: Vec<Option<Qualification>> = vec![None; tracers.len()];
    let mut groups = std::collections::BTreeMap::<usize, Vec<(usize, PreparedValidation)>>::new();
    for (index, tracer) in tracers.iter().enumerate() {
        ctx.check()?;
        let wavelengths = settings
            .wavelengths
            .clone()
            .unwrap_or_else(|| (0..tracer.num_wavelengths()).collect());
        if wavelengths.len() > MAX_WAVELENGTHS {
            result[index] = Some(qualify(
                backend,
                tracer,
                Some(settings),
                &JobContext::with(ctx.cancel.clone(), Default::default()),
            )?);
            continue;
        }
        match prepare_validation(
            tracer,
            settings,
            &wavelengths,
            &JobContext::with(ctx.cancel.clone(), Default::default()),
        ) {
            Ok(p) => groups.entry(p.image as usize).or_default().push((index, p)),
            Err(MkError::Cancelled) => return Err(MkError::Cancelled),
            Err(e) => result[index] = Some(verdict(backend, settings, Err(e))?),
        }
    }
    let mut done = result.iter().filter(|v| v.is_some()).count();
    for group in groups.values() {
        for rows in group.chunks(32) {
            ctx.check()?;
            let systems: Vec<_> = rows.iter().map(|(_, p)| p.packed.clone()).collect();
            let handle = backend.upload_systems(&systems);
            let traces = match handle {
                Err(e) => Err(e),
                Ok(handle) => {
                    let trace = |chief: bool| {
                        let mut rays = RayBatch::default();
                        for (system, (_, p)) in rows.iter().enumerate() {
                            for sample in if chief { &p.chiefs } else { &p.samples } {
                                let (p, d) = (sample.ray.pos, sample.ray.dir);
                                rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], sample.pw as u32);
                                rays.system.push(system as u32);
                            }
                        }
                        let mut request = TraceRequest::new(handle, rays, 0, rows[0].1.image);
                        request.record = if chief {
                            RecordMode::Final
                        } else {
                            RecordMode::AllSurfaces
                        };
                        request.apply_apertures = !chief && settings.clip;
                        request.radiometry = settings.radiometry;
                        request.newton = settings.newton;
                        run_trace(
                            backend,
                            request,
                            &JobContext::with(ctx.cancel.clone(), Default::default()),
                        )
                    };
                    let traces = (|| Ok((trace(false)?, trace(true)?)))();
                    let _ = backend.release_systems(handle);
                    traces
                }
            };
            let (mut first, mut chief_first) = (0, 0);
            for (index, p) in rows {
                let report = match &traces {
                    Ok((gpu, chiefs)) => compare_validation(
                        backend,
                        tracers[*index],
                        settings,
                        &p.samples,
                        &p.chiefs,
                        gpu,
                        chiefs,
                        first,
                        chief_first,
                        &JobContext::with(ctx.cancel.clone(), Default::default()),
                    ),
                    Err(e) => Err(e.clone()),
                };
                result[*index] = Some(verdict(backend, settings, report)?);
                first += p.samples.len();
                chief_first += p.chiefs.len();
                done += 1;
            }
            ctx.report(
                done as f64 / tracers.len().max(1) as f64,
                "candidate geometry qualification",
            );
        }
    }
    ctx.check()?;
    ctx.report(1., "candidate qualification complete");
    Ok(result.into_iter().map(Option::unwrap).collect())
}
