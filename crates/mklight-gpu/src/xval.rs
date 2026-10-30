//! Cross-validation of a backend's trace kernel against the CPU f64
//! reference (`mklight_trace`), and the precision qualification built on it.
//!
//! [`cross_validate`] generates rays with the reference tracer (field and
//! pupil mapping, ray aiming in f64), rounds them once to f32
//! ([`RayBatch::push_f64`]), traces them on the backend and with
//! `mklight_trace::trace_ray`, and compares, per ray that both traces
//! complete:
//!
//! * image-surface position (Euclidean distance, mm) and direction cosines
//!   (Euclidean distance of the unit vectors);
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
//! [`Qualification`] (pass/fail, measured errors, reason). A caller that
//! gets a failure runs the CPU reference instead and records the decision
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
use crate::batch::{NewtonSettings, Radiometry, RayBatch, RayStatus, TraceRequest, TraceResult};
use crate::consts::*;
use crate::layout::{MAX_WAVELENGTHS, PackOptions, PackedSystem};

/// What to cross-validate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct XvalSettings {
    pub pupil: PupilSampling,
    /// Indices into the tracer's prepared fields; `None` = all.
    pub fields: Option<Vec<usize>>,
    /// System wavelength indices (at most 16); `None` = all.
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
}

impl Default for XvalSettings {
    /// The full suite: a 64 × 64 pupil grid, every field and wavelength.
    fn default() -> Self {
        XvalSettings {
            pupil: PupilSampling::Grid { n: 64 },
            fields: None,
            wavelengths: None,
            clip: true,
            radiometry: Radiometry::Unpolarized,
            newton: NewtonSettings::default(),
            budget: ErrorBudget::CONVENTIONS,
            tir_band: 1e-5,
        }
    }
}

impl XvalSettings {
    /// The small sample [`qualify`] uses: 61 hexapolar pupil points per
    /// field and wavelength.
    pub fn qualification() -> Self {
        XvalSettings {
            pupil: PupilSampling::Hexapolar { rings: 4 },
            ..Default::default()
        }
    }
}

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
    pub max_position_mm: f64,
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
        other => Err(MkError::Backend(format!(
            "trace job returned {other:?}"
        ))),
    }
}

/// Cross-validate `backend` against the reference for `tracer`'s system
/// (module docs). Fails with the packer's or backend's error when the system
/// cannot run on the backend at all.
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
    if wls.len() > MAX_WAVELENGTHS {
        return Err(MkError::Unsupported(format!(
            "cross-validation covers at most {MAX_WAVELENGTHS} wavelengths"
        )));
    }
    let fields: Vec<usize> = settings
        .fields
        .clone()
        .unwrap_or_else(|| (0..tracer.fields().len()).collect());
    let pupil = settings.pupil.points();
    ctx.report(0.0, "cross-validation: packing");
    let hints = tracer.semi_diameters(&SemiDiameterSettings::default()).values;
    let packed = PackedSystem::pack(
        sys,
        &PackOptions {
            wavelengths: Some(wls.clone()),
            radius_hints: Some(hints),
        },
    )?;
    let handle = backend.upload_system(&packed)?;
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
    let batch = |v: &[Sample]| {
        let mut b = RayBatch::with_capacity(v.len());
        for s in v {
            let (p, d) = (s.ray.pos, s.ray.dir);
            b.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], s.pw as u32);
        }
        b
    };
    let request = |rays: RayBatch, clip: bool| {
        let mut r = TraceRequest::new(handle, rays, 0, image);
        r.apply_apertures = clip;
        r.radiometry = settings.radiometry;
        r.newton = settings.newton;
        r
    };
    ctx.report(0.05, "cross-validation: backend trace");
    let traced = (|| {
        let g = run_trace(
            backend,
            request(batch(&samples), settings.clip),
            &ctx.subrange(0.05, 0.5),
        )?;
        let c = run_trace(backend, request(batch(&chiefs), false), &ctx.subrange(0.5, 0.55))?;
        Ok::<_, MkError>((g, c))
    })();
    let _ = backend.release_systems(handle);
    let (gpu, gpu_chief) = traced?;
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
        let g = if gpu_chief.ray_status(k).is_ok() {
            backend_reference(tracer, &as_reference(&gpu_chief, k), c.wl)
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
                gpu_chief.ray_status(k)
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
            let r = mklight_trace::trace_ray(sys, &s.ray, s.wl, &opts, None);
            let want = reference_status_word(&r.status);
            let got = gpu.status[i];
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
            let g = as_reference(&gpu, i);
            a.compared += 1;
            let dp = (g.ray.pos - r.ray.pos).norm();
            let dd = (g.ray.dir - r.ray.dir).norm();
            a.pos = a.pos.max(dp);
            a.dir = a.dir.max(dd);
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
    ctx.report(1.0, "cross-validation");
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
    pub budget: ErrorBudget,
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
/// cross-validation ([`XvalSettings::qualification`] unless given) checked
/// against its budget. Never an error: a backend that cannot run the system
/// (unsupported job, unpackable model, backend error) fails with that
/// reason. Cancellation is reported as a failed qualification as well.
pub fn qualify(
    backend: &dyn ComputeBackend,
    tracer: &SequentialTracer,
    settings: Option<&XvalSettings>,
    ctx: &JobContext,
) -> Qualification {
    let default = XvalSettings::qualification();
    let settings = settings.unwrap_or(&default);
    let mut q = Qualification {
        backend: backend.info().backend_id(),
        kernel_build: backend.precision().kernel_build.clone(),
        passed: false,
        budget: settings.budget,
        report: None,
        reason: None,
    };
    if !backend.supports(crate::backend::JobKind::Trace) {
        q.reason = Some("backend does not support the trace kernel".into());
        return q;
    }
    match cross_validate(backend, tracer, settings, ctx) {
        Ok(r) => {
            match r.check(&settings.budget) {
                Ok(()) => q.passed = true,
                Err(why) => q.reason = Some(format!("precision qualification failed: {why}")),
            }
            q.report = Some(r);
        }
        Err(e) => q.reason = Some(format!("cannot run on this backend: {e}")),
    }
    q
}
