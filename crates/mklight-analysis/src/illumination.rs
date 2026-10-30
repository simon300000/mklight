//! Geometric relative illumination, vignetting vs field, and surface
//! footprints.
//!
//! # Relative illumination
//!
//! For a Lambertian object of uniform radiance `L`, radiance conservation
//! gives the image irradiance `E = L (n'/n)² Ω_p`, where `Ω_p` is the
//! **projected solid angle** of the beam converging on the image point,
//! i.e. the area of the beam in direction-cosine space `(L, M)` of the image
//! surface (`dΩ cos θ = dL dM`). The relative illumination is
//! `RI(H) = Ω_p(H)/Ω_p(0)`; `cos⁴` fall-off, pupil distortion and coma, and
//! vignetting are all included because nothing is assumed about the pupil.
//!
//! The beam is generated with **real ray aiming** (a copy of the tracer with
//! `RayAiming::Real` when the system does not already use it), so normalised
//! pupil coordinates map linearly onto the stop's clear aperture: the beam
//! of every field fills exactly the real stop, however the paraxial
//! entrance pupil is distorted. Then
//!
//! `Ω_p = ∫∫_{unit disk} s(P) |∂(L, M)/∂(P_x, P_y)| dP_x dP_y`,
//!
//! where `s(P) ∈ {0, 1}` is the survival of the clipped ray (fixed apertures
//! elsewhere vignette the beam) and the Jacobian is formed by central
//! differences of unclipped rays (step `10⁻⁴`). The integral uses polar
//! coordinates with Gauss–Legendre nodes in `ρ` and the trapezoidal rule in
//! `φ` (spectrally accurate for the periodic integrand), so an unvignetted
//! beam is integrated to near machine precision; a vignetting boundary
//! limits the accuracy to `O(1/N)`. The **vignetting fraction** is the
//! clipped share of the stop area, `1 − ∫∫ s dP / π`. Direction cosines are
//! taken in the image surface's local frame (exact for plane image surfaces).
//!
//! The geometry assumes the beam converges to (near) one image point; for a
//! strongly aberrated beam, `Ω_p` is the angular extent of the rays, the
//! usual geometric definition.
//!
//! # Footprints
//!
//! A pupil pattern is traced (clipping on) up to a chosen surface for each
//! selected field and wavelength; the hit points are reported in that
//! surface's local frame, with the rays clipped *at* that surface flagged.
//! Rays lost before it are counted but not shown.

use std::f64::consts::PI;

use mklight_core::job::RunTimer;
use mklight_core::{ApertureShape, JobContext, MkError, MkResult, RayAiming, RunInfo};
use mklight_trace::{
    FieldPoint, Grid, PreparedField, SemiDiameterSettings, SequentialTracer, TraceOptions,
    TraceStatus, TracerSettings, trace_grid_job,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{
    FieldSelection, PupilPattern, WavelengthSelection, check_count, field_scale, field_unit,
    gauss_legendre_01, linspace, nf, pupil_points, run_info, wavelength_or_primary,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RelativeIlluminationSettings {
    /// Field samples from the axis to the maximum field along `+y`.
    pub field_samples: usize,
    /// System wavelength index; the primary when `None`.
    pub wavelength: Option<usize>,
    /// Gauss–Legendre nodes in the pupil radius.
    pub radial_samples: usize,
    /// Trapezoidal nodes in the pupil azimuth.
    pub azimuth_samples: usize,
}

impl Default for RelativeIlluminationSettings {
    fn default() -> Self {
        RelativeIlluminationSettings {
            field_samples: 11,
            wavelength: None,
            radial_samples: 12,
            azimuth_samples: 48,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelativeIlluminationResult {
    pub field_values: Vec<f64>,
    pub normalized_field: Vec<f64>,
    pub field_unit: String,
    pub wavelength: usize,
    /// `Ω_p(H)/Ω_p(0)`.
    #[serde(with = "nf::vec")]
    pub relative_illumination: Vec<f64>,
    /// Projected solid angle of the beam at the image point (sr).
    #[serde(with = "nf::vec")]
    pub projected_solid_angle: Vec<f64>,
    /// Clipped share of the stop area.
    #[serde(with = "nf::vec")]
    pub vignetting_fraction: Vec<f64>,
    /// Image-space chief-ray angle with the image surface's local `z` (degrees).
    #[serde(with = "nf::vec")]
    pub chief_ray_angle_deg: Vec<f64>,
    /// `cos⁴` of the object-space chief-ray angle (for comparison).
    #[serde(with = "nf::vec")]
    pub cos4_object: Vec<f64>,
    pub run_info: RunInfo,
}

/// One field's integrals: `(Ω_p, surviving share of the pupil area, chief
/// angle, cos⁴)`.
fn integrate_field(
    rt: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    nodes: &[(f64, f64, f64)],
) -> [f64; 4] {
    const D: f64 = 1e-4;
    let un = TraceOptions::unclipped();
    let cl = rt.options();
    let lm = |px: f64, py: f64| -> Option<[f64; 2]> {
        let r = rt.trace(field, wl, px, py, &un);
        r.status.is_ok().then_some([r.ray.dir.x, r.ray.dir.y])
    };
    let parts: Vec<(f64, f64)> = nodes
        .par_iter()
        .map(|&(px, py, w)| {
            let alive = rt.trace(field, wl, px, py, &cl).status.is_ok();
            if !alive {
                return (0.0, 0.0);
            }
            let det = (|| {
                let (a, b) = (lm(px + D, py)?, lm(px - D, py)?);
                let (c, d) = (lm(px, py + D)?, lm(px, py - D)?);
                let (lx, mx) = ((a[0] - b[0]) / (2.0 * D), (a[1] - b[1]) / (2.0 * D));
                let (ly, my) = ((c[0] - d[0]) / (2.0 * D), (c[1] - d[1]) / (2.0 * D));
                Some(lx.mul_add(my, -(ly * mx)).abs())
            })();
            match det {
                Some(j) => (w * j, w),
                None => (0.0, w),
            }
        })
        .collect();
    let omega: mklight_core::math::KahanSum = parts.iter().map(|p| p.0).collect();
    let area: mklight_core::math::KahanSum = parts.iter().map(|p| p.1).collect();
    let total: mklight_core::math::KahanSum = nodes.iter().map(|n| n.2).collect();
    let chief = rt.chief(field, wl);
    let cra = if chief.status.is_ok() {
        chief.ray.dir.z.abs().min(1.0).acos().to_degrees()
    } else {
        f64::NAN
    };
    let cos4 = match rt.generate(field, wl, 0.0, 0.0) {
        Ok(g) => {
            let mut d = g.ray.dir;
            if !rt.system().object_at_infinity {
                d = rt.system().surfaces[0].frame.to_global_dir(d);
            }
            d.z.abs().powi(4)
        }
        Err(_) => f64::NAN,
    };
    [omega.value(), area.value() / total.value(), cra, cos4]
}

/// Relative illumination and vignetting vs field (module docs).
pub fn relative_illumination(
    tracer: &SequentialTracer,
    settings: &RelativeIlluminationSettings,
) -> MkResult<RelativeIlluminationResult> {
    relative_illumination_job(tracer, settings, &JobContext::new())
}

/// [`relative_illumination`] with cancellation and progress (per field).
pub fn relative_illumination_job(
    tracer: &SequentialTracer,
    settings: &RelativeIlluminationSettings,
    ctx: &JobContext,
) -> MkResult<RelativeIlluminationResult> {
    let timer = RunTimer::start();
    check_count("field samples", settings.field_samples, 2, 10_001)?;
    check_count("radial samples", settings.radial_samples, 1, 512)?;
    check_count("azimuth samples", settings.azimuth_samples, 4, 4096)?;
    let wl = wavelength_or_primary(tracer, settings.wavelength)?;
    let owned;
    let rt = if matches!(tracer.aiming(), RayAiming::Real { .. }) {
        tracer
    } else {
        owned = SequentialTracer::with_settings(
            tracer.system_arc(),
            TracerSettings {
                aiming: Some(RayAiming::Real { tolerance: 1e-12 }),
                ..*tracer.settings()
            },
        )?;
        &owned
    };
    let (rx, rw) = gauss_legendre_01(settings.radial_samples);
    let m = settings.azimuth_samples;
    let dphi = 2.0 * PI / m as f64;
    let mut nodes = Vec::with_capacity(rx.len() * m);
    for (r, w) in rx.iter().zip(&rw) {
        for j in 0..m {
            let (s, c) = (j as f64 * dphi).sin_cos();
            nodes.push((r * c, r * s, w * r * dphi));
        }
    }
    let hs = linspace(0.0, 1.0, settings.field_samples);
    let mut rows = Vec::with_capacity(hs.len());
    let mut warnings = Vec::new();
    ctx.report(0.0, "relative illumination");
    for (k, &h) in hs.iter().enumerate() {
        ctx.check()?;
        let row = match rt.prepare_field(FieldPoint::normalized(0.0, h)) {
            Ok(f) if f.aims[wl].converged => integrate_field(rt, &f, wl, &nodes),
            _ => {
                warnings.push(format!("field Hy = {h}: chief ray could not be aimed"));
                [f64::NAN; 4]
            }
        };
        rows.push(row);
        ctx.report((k + 1) as f64 / hs.len() as f64, "relative illumination");
    }
    let omega0 = rows[0][0];
    if !(omega0 > 0.0) {
        return Err(MkError::Numerical(
            "on-axis beam has no projected solid angle".into(),
        ));
    }
    let scale = field_scale(tracer);
    Ok(RelativeIlluminationResult {
        field_values: hs.iter().map(|h| h * scale).collect(),
        field_unit: field_unit(tracer.system().system().fields.kind).to_string(),
        normalized_field: hs,
        wavelength: wl,
        relative_illumination: rows.iter().map(|r| r[0] / omega0).collect(),
        projected_solid_angle: rows.iter().map(|r| r[0]).collect(),
        vignetting_fraction: rows.iter().map(|r| 1.0 - r[1]).collect(),
        chief_ray_angle_deg: rows.iter().map(|r| r[2]).collect(),
        cos4_object: rows.iter().map(|r| r[3]).collect(),
        run_info: run_info(
            timer,
            tracer,
            "relative_illumination",
            settings,
            None,
            warnings,
        ),
    })
}

// ----- Footprint ----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FootprintSettings {
    /// Surface index (1 … image).
    pub surface: usize,
    pub pattern: PupilPattern,
    pub rings: usize,
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub seed: u64,
}

impl Default for FootprintSettings {
    fn default() -> Self {
        FootprintSettings {
            surface: 1,
            pattern: PupilPattern::Square,
            rings: 8,
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            seed: 1,
        }
    }
}

/// The footprint of one (field, wavelength).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootprintRays {
    pub field: usize,
    pub wavelength: usize,
    /// Hit points `[x, y]` (surface local frame, mm) of rays that pass.
    #[serde(with = "nf::vec2")]
    pub points: Vec<[f64; 2]>,
    /// Hit points of rays clipped by this surface's aperture.
    #[serde(with = "nf::vec2")]
    pub clipped: Vec<[f64; 2]>,
    /// Rays launched and rays lost before reaching the surface.
    pub launched: usize,
    pub lost_before: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootprintResult {
    pub surface: usize,
    /// Clear semi-diameter of the surface (fixed, stop or traced, mm).
    pub semi_diameter: f64,
    pub aperture: Option<ApertureShape>,
    pub rays: Vec<FootprintRays>,
    /// `[x_min, x_max, y_min, y_max]` over all passing hits (NaN if none).
    #[serde(with = "nf::vec")]
    pub extent: Vec<f64>,
    /// Largest radial hit height (mm).
    #[serde(with = "nf")]
    pub max_radius: f64,
    pub run_info: RunInfo,
}

/// Ray footprint on a surface (module docs).
pub fn footprint(
    tracer: &SequentialTracer,
    settings: &FootprintSettings,
) -> MkResult<FootprintResult> {
    footprint_job(tracer, settings, &JobContext::new())
}

/// [`footprint`] with cancellation and progress.
pub fn footprint_job(
    tracer: &SequentialTracer,
    settings: &FootprintSettings,
    ctx: &JobContext,
) -> MkResult<FootprintResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    let s = settings.surface;
    if s == 0 || s > sys.image_index() {
        return Err(MkError::InvalidModel(format!(
            "footprint surface {s} must be in 1…{}",
            sys.image_index()
        )));
    }
    check_count("rings", settings.rings, 1, 512)?;
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let pupil = pupil_points(settings.pattern, settings.rings, settings.seed);
    let prepared: Vec<_> = fields.iter().map(|&f| tracer.field(f).clone()).collect();
    let grid = Grid {
        fields: &prepared,
        wavelengths: &wls,
        pupil: &pupil,
    };
    let opts = TraceOptions {
        last_surface: Some(s),
        ..tracer.options()
    };
    let rays = trace_grid_job(tracer, &grid, &opts, &ctx.subrange(0.0, 0.9))?;
    let np = pupil.len();
    let mut out = Vec::with_capacity(fields.len() * wls.len());
    let mut ext = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut max_r = f64::NAN;
    for (fi, &f) in fields.iter().enumerate() {
        for (wi, &w) in wls.iter().enumerate() {
            let block = &rays[(fi * wls.len() + wi) * np..(fi * wls.len() + wi + 1) * np];
            let mut fr = FootprintRays {
                field: f,
                wavelength: w,
                points: Vec::new(),
                clipped: Vec::new(),
                launched: np,
                lost_before: 0,
            };
            for r in block {
                let p = [r.point.x, r.point.y];
                match r.status {
                    TraceStatus::Ok => {
                        ext = [
                            ext[0].min(p[0]),
                            ext[1].max(p[0]),
                            ext[2].min(p[1]),
                            ext[3].max(p[1]),
                        ];
                        max_r = max_r.max(p[0].hypot(p[1]));
                        fr.points.push(p);
                    }
                    TraceStatus::Clipped { surface } if surface == s => fr.clipped.push(p),
                    _ => fr.lost_before += 1,
                }
            }
            out.push(fr);
        }
    }
    let extent = if ext[0].is_finite() {
        ext.to_vec()
    } else {
        vec![f64::NAN; 4]
    };
    let sd = tracer.semi_diameters(&SemiDiameterSettings::default());
    ctx.report(1.0, "footprint");
    let seed = matches!(
        settings.pattern,
        PupilPattern::Random | PupilPattern::Dithered
    )
    .then_some(settings.seed);
    Ok(FootprintResult {
        surface: s,
        semi_diameter: sd.values[s],
        aperture: sys.surfaces[s].aperture.clone(),
        rays: out,
        extent,
        max_radius: max_r,
        run_info: run_info(timer, tracer, "footprint", settings, seed, Vec::new()),
    })
}
