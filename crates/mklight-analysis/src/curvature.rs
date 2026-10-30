//! Field curvature (astigmatic foci along the real chief ray) and distortion.
//!
//! # Astigmatic foci: Coddington along the real chief ray
//!
//! The chief ray of a field is traced (unclipped) with per-surface records.
//! Along it, a thin tangential and a thin sagittal pencil are followed with
//! **vergences** `V = n/t`, where `t` is the signed distance along the ray
//! from the current point to the pencil's focus (`t > 0`: converging to a
//! point ahead). `V = 0` is a collimated pencil, so objects at infinity need
//! no special case; a finite object starts with `V = −n/d` at surface 1 (the
//! pencil diverges from the object point a distance `d` behind).
//!
//! * transfer over a distance `d` along the ray: `V ← V / (1 − d V / n)`;
//! * refraction (`n → n'`, angles `I`, `I'` of the chief ray with the normal):
//!   `n' cos²I'/t' = n cos²I/t + (n' cos I' − n cos I) κ_t` (tangential) and
//!   `n'/s' = n/s + (n' cos I' − n cos I) κ_s` (sagittal);
//! * reflection: `V_t' = V_t − 2 n κ_t / cos I`, `V_s' = V_s − 2 n κ_s cos I`;
//! * ideal lens (`Paraxial` shape, which images planes onto planes without
//!   aberration): `V' = cos U' (V / cos U + n'/f)` in both sections, `U`, `U'`
//!   being the ray angles with the lens axis;
//! * dummies and coordinate breaks: nothing.
//!
//! `κ_t`, `κ_s` are the normal curvatures of the surface at the hit point in
//! the plane of incidence and perpendicular to it, signed positive when the
//! centre of curvature lies ahead along the incident ray. For a surface of
//! revolution the principal curvatures at radius `r` are the meridional
//! `κ_m = f''/(1 + f'²)^{3/2}` and sagittal `κ_s = f'/(r √(1 + f'²))`
//! ([`principal_curvatures`], evaluated without dividing by `r` or by the
//! conic square root); along a tangent direction at angle `α` from the
//! meridional direction the normal curvature is `κ_m cos²α + κ_s sin²α`.
//! These equations are exact for the chief ray when its plane of incidence is
//! a principal plane at every surface (every meridional chief ray of a
//! centred system, including folds about x). For skew chief rays the twist
//! terms of the generalised equations are neglected, and the close-ray
//! values below are the reference.
//!
//! At the image the foci lie at `t_t = n/V_t` and `t_s = n/V_s` along the
//! chief ray from its image point. The reported **field curvature** is their
//! projection on the image surface's local `z` axis, `Δz = t · d_z`, i.e. the
//! axial focus shift from the image surface (positive along local `+z`).
//!
//! # Close-ray cross-check
//!
//! Two pairs of rays at pupil offsets `(0, ±h)` and `(±h, 0)` from the chief
//! are traced to the image. Each ray is moved onto the plane through the
//! chief image point `I` normal to the chief direction `c`, giving a point
//! `Q` and a transverse slope `T = d/(d·c) − c`; the pair's focus is the
//! distance `s = −ΔQ·ΔT/|ΔT|²` along `c` that minimises the pair's separation
//! (central differences: error `O(h²)`). Both values are reported.
//!
//! # Distortion
//!
//! The real chief-ray height `y` (image local frame) is compared with the
//! paraxial chief height `y_p` of the same object source at the same
//! wavelength: `k·tan θ` for an object at infinity (`k` = paraxial image
//! height per unit `tan θ` on the actual image surface) and `m·h` for a
//! finite object. `distortion = 100 (y − y_p)/y_p` %. For objects at
//! infinity, the `f·tan θ` and `f·θ` distortions use `f = n₀·EFL`, the image
//! height per unit `tan θ` at the paraxial focal plane. All are 0 on axis
//! (their limit) and NaN where undefined.
//!
//! The grid distortion map traces the chief rays of a square grid of
//! normalised fields `(Hx, Hy) ∈ [−a, a]²` (`a = 1/√2` by default, so the
//! corners are at the maximum radial field) and reports real and paraxial
//! image points and the radial distortion `100 (|r| − |r_p|)/|r_p|`.

use mklight_core::job::RunTimer;
use mklight_core::{Interaction, JobContext, MkResult, RunInfo, SurfaceShape, Vec3};
use mklight_trace::trace::transfer_into;
use mklight_trace::{FieldPoint, ObjectSource, PreparedField, Ray, SequentialTracer, TraceOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{
    WavelengthSelection, check_count, check_finite, field_scale, field_unit, linspace, nf,
    paraxial_image_point, run_info, wavelength_or_primary,
};

/// Principal curvatures `(κ_m, κ_s)` (meridional, sagittal) of a surface of
/// revolution at radius `r ≥ 0`, positive when the centre of curvature lies
/// at local `+z`. `None` outside the sag domain. Planes, ideal lenses and
/// coordinate breaks have zero curvature.
///
/// With `w = √(1 − (1+k)c²r²)` the conic has `f'/r = c/w` and `f'' = c/w³`,
/// so `κ_m = c/D^{3/2}` and `κ_s = c/√D` with `D = 1 − k c² r²`. An even
/// asphere adds `Σ aᵢ(2i+4) r^{2i+2}` to `f'/r` and `Σ aᵢ(2i+4)(2i+3) r^{2i+2}`
/// to `f''`.
pub fn principal_curvatures(shape: &SurfaceShape, r: f64) -> Option<(f64, f64)> {
    let r2 = r * r;
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
            Some((0.0, 0.0))
        }
        SurfaceShape::Conic {
            curvature: c,
            conic: k,
        } => {
            let w2 = 1.0 - (1.0 + k) * c * c * r2;
            if w2 < 0.0 {
                return None;
            }
            let d = (-k * c * c).mul_add(r2, 1.0);
            if d <= 0.0 {
                return None;
            }
            Some((c / (d * d.sqrt()), c / d.sqrt()))
        }
        SurfaceShape::EvenAsphere {
            curvature: c,
            conic: k,
            coefficients,
        } => {
            let w2 = 1.0 - (1.0 + k) * c * c * r2;
            if w2 < 0.0 {
                return None;
            }
            let w = w2.sqrt();
            let (mut g, mut f2) = if *c == 0.0 {
                (0.0, 0.0)
            } else {
                (c / w, c / (w * w2))
            };
            // Σ aᵢ (2i+4) r^{2i+2} and Σ aᵢ (2i+4)(2i+3) r^{2i+2}.
            let mut p = r2; // r^{2i+2}
            for (i, a) in coefficients.iter().enumerate() {
                let e = (2 * i + 4) as f64;
                g = (a * e).mul_add(p, g);
                f2 = (a * e * (e - 1.0)).mul_add(p, f2);
                p *= r2;
            }
            let q = (r2 * g).mul_add(g, 1.0); // 1 + f'²
            let sq = q.sqrt();
            Some((f2 / (q * sq), g / sq))
        }
    }
}

/// Tangential and sagittal foci of the pencil around a chief ray: signed
/// distances (mm) along the image-space chief ray from its image point
/// (`±∞` for a collimated pencil), and the chief direction's local `z`
/// component on the image surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AstigmaticFocus {
    #[serde(with = "nf")]
    pub tangential: f64,
    #[serde(with = "nf")]
    pub sagittal: f64,
    pub dir_z: f64,
}

impl AstigmaticFocus {
    /// Axial (image local `z`) focus shifts `(Δz_t, Δz_s)` from the image surface.
    pub fn axial(&self) -> (f64, f64) {
        (self.tangential * self.dir_z, self.sagittal * self.dir_z)
    }
}

/// Normal curvatures `(κ_t, κ_s)` along the in-surface tangent `e_t` (in the
/// plane of incidence) and the perpendicular tangent, at local hit `p` with
/// unit normal `n`.
fn section_curvatures(shape: &SurfaceShape, p: Vec3, n: Vec3, e_t: Vec3) -> Option<(f64, f64)> {
    let r = p.x.hypot(p.y);
    let (km, ks) = principal_curvatures(shape, r)?;
    if km == ks || r == 0.0 {
        return Some((km, ks));
    }
    // Meridional principal tangent: the radial direction projected onto the
    // tangent plane.
    let er = Vec3::new(p.x / r, p.y / r, 0.0);
    let m = (er - n * er.dot(n)).normalized();
    let ca = e_t.dot(m);
    let c2 = (ca * ca).min(1.0);
    let s2 = 1.0 - c2;
    Some((km.mul_add(c2, ks * s2), km.mul_add(s2, ks * c2)))
}

/// Coddington tangential/sagittal foci along the real chief ray of `field`
/// at wavelength `wl` (module docs). `None` if the chief ray fails or a
/// surface curvature is undefined at its hit.
pub fn coddington_focus(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
) -> Option<AstigmaticFocus> {
    let sys = tracer.system();
    let img = sys.image_index();
    let mut rec = Vec::with_capacity(sys.num_surfaces());
    let r = tracer.trace_with(
        field,
        wl,
        0.0,
        0.0,
        &TraceOptions::unclipped(),
        Some(&mut rec),
    );
    if !r.status.is_ok() || rec.len() != img + 1 {
        return None;
    }
    let (mut vt, mut vs) = (0.0f64, 0.0f64);
    let mut dir_z = f64::NAN;
    for i in 1..=img {
        let inc = transfer_into(sys, i, Ray::new(rec[i - 1].point, rec[i - 1].dir));
        let k = inc.dir;
        let d = (rec[i].point - inc.pos).dot(k);
        let n = sys.n_before(i, wl);
        if i == 1 && !sys.object_at_infinity {
            vt = -n / d;
            vs = vt;
        } else {
            vt /= (-d / n).mul_add(vt, 1.0);
            vs /= (-d / n).mul_add(vs, 1.0);
        }
        if i == img {
            dir_z = k.z;
            break;
        }
        let s = &sys.surfaces[i];
        if s.is_passive() {
            continue;
        }
        let nrm = rec[i].normal;
        let kout = rec[i].dir;
        let kn = k.dot(nrm);
        let cos_i = kn.abs().min(1.0);
        let cos_o = kout.dot(nrm).abs().min(1.0);
        if let SurfaceShape::Paraxial { focal_length } = s.shape {
            let n2 = if s.interaction == Interaction::Reflect {
                n
            } else {
                s.n_after[wl]
            };
            let (cu, cu2) = (k.z.abs(), kout.z.abs());
            vt = cu2 * (vt / cu + n2 / focal_length);
            vs = cu2 * (vs / cu + n2 / focal_length);
            continue;
        }
        // Tangent in the plane of incidence (any tangent at normal incidence).
        let mut e_t = k - nrm * kn;
        if e_t.norm() < 1e-12 {
            e_t = Vec3::Y - nrm * nrm.y;
            if e_t.norm() < 1e-12 {
                e_t = Vec3::X - nrm * nrm.x;
            }
        }
        let e_t = e_t.normalized();
        let (kt, ks) = section_curvatures(&s.shape, rec[i].point, nrm, e_t)?;
        // Signed along the incident propagation direction.
        let sg = if kn < 0.0 { -1.0 } else { 1.0 };
        let (kt, ks) = (sg * kt, sg * ks);
        match s.interaction {
            Interaction::Reflect => {
                vt -= 2.0 * n * kt / cos_i;
                vs -= 2.0 * n * ks * cos_i;
            }
            _ => {
                let n2 = s.n_after[wl];
                let p = n2.mul_add(cos_o, -n * cos_i);
                vt = (vt * cos_i * cos_i + p * kt) / (cos_o * cos_o);
                vs = p.mul_add(ks, vs);
            }
        }
    }
    let n = sys.n_before(img, wl);
    Some(AstigmaticFocus {
        tangential: n / vt,
        sagittal: n / vs,
        dir_z,
    })
}

/// Close-ray (finite-difference) foci of `field` at `wl` with pupil step `h`
/// (module docs). `None` if any of the five rays fails.
pub fn close_ray_focus(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    h: f64,
) -> Option<AstigmaticFocus> {
    let opts = TraceOptions::unclipped();
    let c = tracer.trace(field, wl, 0.0, 0.0, &opts);
    if !c.status.is_ok() {
        return None;
    }
    let ip = c.ray.pos;
    let cd = c.ray.dir;
    let proj = |px: f64, py: f64| -> Option<(Vec3, Vec3)> {
        let r = tracer.trace(field, wl, px, py, &opts);
        if !r.status.is_ok() {
            return None;
        }
        let d = r.ray.dir;
        let dc = d.dot(cd);
        if dc <= 0.0 {
            return None;
        }
        let q = r.ray.pos + d * ((ip - r.ray.pos).dot(cd) / dc);
        Some((q, d / dc - cd))
    };
    let focus = |a: (Vec3, Vec3), b: (Vec3, Vec3)| {
        let dq = a.0 - b.0;
        let dt = a.1 - b.1;
        let t2 = dt.norm2();
        if t2 > 0.0 {
            -dq.dot(dt) / t2
        } else if dq.norm2() == 0.0 {
            f64::NAN
        } else {
            f64::INFINITY
        }
    };
    let tan = focus(proj(0.0, h)?, proj(0.0, -h)?);
    let sag = focus(proj(h, 0.0)?, proj(-h, 0.0)?);
    Some(AstigmaticFocus {
        tangential: tan,
        sagittal: sag,
        dir_z: cd.z,
    })
}

// ----- Field curvature and distortion vs field ----------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldCurvatureSettings {
    /// Field samples from the axis to the maximum field along `+y`.
    pub field_samples: usize,
    pub wavelengths: WavelengthSelection,
    /// Normalised pupil step of the close-ray cross-check.
    pub close_ray_step: f64,
}

impl Default for FieldCurvatureSettings {
    fn default() -> Self {
        FieldCurvatureSettings {
            field_samples: 21,
            wavelengths: WavelengthSelection::All,
            close_ray_step: 1e-3,
        }
    }
}

/// Field curvature and distortion of one wavelength; one value per field
/// sample, NaN (`null`) where undefined.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldCurvatureCurve {
    pub wavelength: usize,
    pub wavelength_um: f64,
    /// Tangential axial focus shift from the image surface (mm, Coddington).
    #[serde(with = "nf::vec")]
    pub tangential: Vec<f64>,
    /// Sagittal axial focus shift (mm, Coddington).
    #[serde(with = "nf::vec")]
    pub sagittal: Vec<f64>,
    /// Close-ray cross-checks of `tangential` / `sagittal`.
    #[serde(with = "nf::vec")]
    pub tangential_close_rays: Vec<f64>,
    #[serde(with = "nf::vec")]
    pub sagittal_close_rays: Vec<f64>,
    /// Real chief-ray height on the image surface (mm).
    #[serde(with = "nf::vec")]
    pub real_height: Vec<f64>,
    /// Paraxial chief height of the same source (mm).
    #[serde(with = "nf::vec")]
    pub paraxial_height: Vec<f64>,
    /// `100 (y − y_p)/y_p` (%).
    #[serde(with = "nf::vec")]
    pub distortion_percent: Vec<f64>,
    /// Against `f·tan θ` (objects at infinity; NaN otherwise).
    #[serde(with = "nf::vec")]
    pub f_tan_theta_distortion_percent: Vec<f64>,
    /// Against `f·θ` (objects at infinity; NaN otherwise).
    #[serde(with = "nf::vec")]
    pub f_theta_distortion_percent: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldCurvatureResult {
    /// Field values (in `field_unit`) from the axis to the maximum field.
    pub field_values: Vec<f64>,
    /// The same samples as normalised `Hy`.
    pub normalized_field: Vec<f64>,
    pub field_unit: String,
    pub curves: Vec<FieldCurvatureCurve>,
    pub wavelengths_um: Vec<f64>,
    pub run_info: RunInfo,
}

impl FieldCurvatureResult {
    pub fn curve(&self, wavelength: usize) -> Option<&FieldCurvatureCurve> {
        self.curves.iter().find(|c| c.wavelength == wavelength)
    }
}

fn percent(real: f64, reference: f64) -> f64 {
    100.0 * (real - reference) / reference
}

/// Field curvature and distortion vs field (module docs).
pub fn field_curvature(
    tracer: &SequentialTracer,
    settings: &FieldCurvatureSettings,
) -> MkResult<FieldCurvatureResult> {
    field_curvature_job(tracer, settings, &JobContext::new())
}

/// [`field_curvature`] with cancellation and progress.
pub fn field_curvature_job(
    tracer: &SequentialTracer,
    settings: &FieldCurvatureSettings,
    ctx: &JobContext,
) -> MkResult<FieldCurvatureResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("field samples", settings.field_samples, 2, 100_001)?;
    check_finite("close-ray step", settings.close_ray_step)?;
    if !(settings.close_ray_step > 0.0 && settings.close_ray_step < 0.5) {
        return Err(mklight_core::MkError::InvalidModel(
            "close-ray step must be in (0, 0.5)".into(),
        ));
    }
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let hs = linspace(0.0, 1.0, settings.field_samples);
    let mut warnings = Vec::new();
    ctx.report(0.0, "field curvature");
    // Prepare every field sample once (real image heights solve here).
    let fields: Vec<Option<PreparedField>> = hs
        .par_iter()
        .map(|&h| tracer.prepare_field(FieldPoint::normalized(0.0, h)).ok())
        .collect();
    for (h, f) in hs.iter().zip(&fields) {
        if f.is_none() {
            warnings.push(format!("field Hy = {h}: could not be resolved"));
        }
    }
    ctx.check()?;
    let n0 = |w: usize| sys.n_object[w];
    let mut curves = Vec::with_capacity(wls.len());
    for (k, &w) in wls.iter().enumerate() {
        let fo = tracer.paraxial().at(w);
        let f_eff = n0(w) * fo.efl;
        let rows: Vec<[f64; 10]> = fields
            .par_iter()
            .zip(&hs)
            .map(|(f, &h)| {
                let mut row = [f64::NAN; 10];
                let Some(f) = f else { return row };
                if let Some(a) = coddington_focus(tracer, f, w) {
                    (row[0], row[1]) = a.axial();
                }
                if let Some(a) = close_ray_focus(tracer, f, w, settings.close_ray_step) {
                    (row[2], row[3]) = a.axial();
                }
                let c = tracer.chief(f, w);
                if c.status.is_ok() {
                    row[4] = c.ray.pos.y;
                }
                if let Some(p) = paraxial_image_point(tracer, f, w) {
                    row[5] = p[1];
                }
                if h == 0.0 {
                    row[6] = 0.0;
                    row[7] = if sys.object_at_infinity {
                        0.0
                    } else {
                        f64::NAN
                    };
                    row[8] = row[7];
                } else {
                    row[6] = percent(row[4], row[5]);
                    if let ObjectSource::Direction { dir } = f.source
                        && sys.object_at_infinity
                    {
                        let t = dir.y / dir.z;
                        row[7] = percent(row[4], f_eff * t);
                        row[8] = percent(row[4], f_eff * t.atan());
                    }
                }
                row
            })
            .collect();
        let col = |j: usize| rows.iter().map(|r| r[j]).collect::<Vec<f64>>();
        curves.push(FieldCurvatureCurve {
            wavelength: w,
            wavelength_um: sys.wavelengths_um[w],
            tangential: col(0),
            sagittal: col(1),
            tangential_close_rays: col(2),
            sagittal_close_rays: col(3),
            real_height: col(4),
            paraxial_height: col(5),
            distortion_percent: col(6),
            f_tan_theta_distortion_percent: col(7),
            f_theta_distortion_percent: col(8),
        });
        ctx.check()?;
        ctx.report((k + 1) as f64 / wls.len() as f64, "field curvature");
    }
    let scale = field_scale(tracer);
    Ok(FieldCurvatureResult {
        field_values: hs.iter().map(|h| h * scale).collect(),
        normalized_field: hs,
        field_unit: field_unit(sys.system().fields.kind).to_string(),
        curves,
        wavelengths_um: sys.wavelengths_um.clone(),
        run_info: run_info(
            timer,
            tracer,
            "field_curvature_distortion",
            settings,
            None,
            warnings,
        ),
    })
}

// ----- Grid distortion ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GridDistortionSettings {
    /// Grid points per side (≥ 2).
    pub grid_size: usize,
    /// Half-width `a` of the normalised field square.
    pub half_width: f64,
    /// System wavelength index; the primary when `None`.
    pub wavelength: Option<usize>,
}

impl Default for GridDistortionSettings {
    fn default() -> Self {
        GridDistortionSettings {
            grid_size: 11,
            half_width: std::f64::consts::FRAC_1_SQRT_2,
            wavelength: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridPoint {
    /// Normalised field `(Hx, Hy)`.
    pub field: [f64; 2],
    /// Real chief-ray image point (image local frame, mm).
    #[serde(with = "nf::arr2")]
    pub real: [f64; 2],
    /// Paraxial image point (mm).
    #[serde(with = "nf::arr2")]
    pub paraxial: [f64; 2],
    /// Radial distortion (%); 0 at the centre.
    #[serde(with = "nf")]
    pub distortion_percent: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridDistortionResult {
    pub grid_size: usize,
    pub wavelength: usize,
    /// Row-major, `Hy` outer and ascending, `Hx` inner and ascending.
    pub points: Vec<GridPoint>,
    /// Largest `|distortion|` over the grid (%).
    #[serde(with = "nf")]
    pub max_abs_distortion_percent: f64,
    pub run_info: RunInfo,
}

/// Grid distortion map (module docs).
pub fn grid_distortion(
    tracer: &SequentialTracer,
    settings: &GridDistortionSettings,
) -> MkResult<GridDistortionResult> {
    grid_distortion_job(tracer, settings, &JobContext::new())
}

/// [`grid_distortion`] with cancellation and progress (one block per grid row).
pub fn grid_distortion_job(
    tracer: &SequentialTracer,
    settings: &GridDistortionSettings,
    ctx: &JobContext,
) -> MkResult<GridDistortionResult> {
    let timer = RunTimer::start();
    check_count("grid size", settings.grid_size, 2, 1001)?;
    check_finite("grid half-width", settings.half_width)?;
    if !(settings.half_width > 0.0) {
        return Err(mklight_core::MkError::InvalidModel(
            "grid half-width must be positive".into(),
        ));
    }
    let w = wavelength_or_primary(tracer, settings.wavelength)?;
    let n = settings.grid_size;
    let t = linspace(-settings.half_width, settings.half_width, n);
    let mut points = Vec::with_capacity(n * n);
    let mut failed = 0usize;
    for (j, &hy) in t.iter().enumerate() {
        ctx.check()?;
        let row: Vec<GridPoint> = t
            .par_iter()
            .map(|&hx| {
                let mut g = GridPoint {
                    field: [hx, hy],
                    real: [f64::NAN; 2],
                    paraxial: [f64::NAN; 2],
                    distortion_percent: f64::NAN,
                };
                let Ok(f) = tracer.prepare_field(FieldPoint::normalized(hx, hy)) else {
                    return g;
                };
                let c = tracer.chief(&f, w);
                if c.status.is_ok() {
                    g.real = [c.ray.pos.x, c.ray.pos.y];
                }
                if let Some(p) = paraxial_image_point(tracer, &f, w) {
                    g.paraxial = p;
                }
                let rp = g.paraxial[0].hypot(g.paraxial[1]);
                g.distortion_percent = if hx == 0.0 && hy == 0.0 {
                    0.0
                } else {
                    percent(g.real[0].hypot(g.real[1]), rp)
                };
                g
            })
            .collect();
        failed += row.iter().filter(|g| g.real[0].is_nan()).count();
        points.extend(row);
        ctx.report((j + 1) as f64 / n as f64, "grid distortion");
    }
    let mut warnings = Vec::new();
    if failed > 0 {
        warnings.push(format!("{failed} grid chief rays failed"));
    }
    let max = points
        .iter()
        .map(|g| g.distortion_percent.abs())
        .filter(|v| v.is_finite())
        .fold(f64::NAN, f64::max);
    Ok(GridDistortionResult {
        grid_size: n,
        wavelength: w,
        points,
        max_abs_distortion_percent: max,
        run_info: run_info(timer, tracer, "grid_distortion", settings, None, warnings),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_curvatures_of_known_surfaces() {
        // Sphere: both curvatures equal c everywhere.
        let s = SurfaceShape::sphere(40.0);
        let (km, ks) = principal_curvatures(&s, 25.0).unwrap();
        assert!((km - 0.025).abs() < 1e-15 && (ks - 0.025).abs() < 1e-15);
        // Paraboloid z = r²/(2R): κ_m = c/(1 + c²r²)^{3/2}, κ_s = c/√(1 + c²r²).
        let c = 0.01;
        let p = SurfaceShape::Conic {
            curvature: c,
            conic: -1.0,
        };
        let r: f64 = 30.0;
        let q = 1.0 + c * c * r * r;
        let (km, ks) = principal_curvatures(&p, r).unwrap();
        assert!((km - c / q.powf(1.5)).abs() < 1e-16);
        assert!((ks - c / q.sqrt()).abs() < 1e-16);
        // The same paraboloid as an even asphere with c = 0 and A4 = 0 must
        // differ; with c given as conic −1 plus zero terms it must agree.
        let a = SurfaceShape::EvenAsphere {
            curvature: c,
            conic: -1.0,
            coefficients: vec![0.0, 0.0],
        };
        let (am, as_) = principal_curvatures(&a, r).unwrap();
        assert!((am - km).abs() < 1e-16 && (as_ - ks).abs() < 1e-16);
        // Pure quartic z = A r⁴: f' = 4Ar³, f'' = 12Ar².
        let a4 = 1e-5;
        let q = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![a4],
        };
        let r: f64 = 7.0;
        let fp = 4.0 * a4 * r.powi(3);
        let fpp = 12.0 * a4 * r * r;
        let (qm, qs) = principal_curvatures(&q, r).unwrap();
        assert!((qm - fpp / (1.0 + fp * fp).powf(1.5)).abs() < 1e-15);
        assert!((qs - fp / (r * (1.0 + fp * fp).sqrt())).abs() < 1e-15);
        // Outside the sphere's domain.
        assert!(principal_curvatures(&s, 41.0).is_none());
    }
}
