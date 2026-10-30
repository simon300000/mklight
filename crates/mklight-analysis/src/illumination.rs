//! Geometric relative illumination, vignetting vs field, and surface
//! footprints.
//!
//! # Relative illumination
//!
//! For a Lambertian object of uniform radiance `L`, radiance conservation
//! gives the image irradiance `E = L (n'/n)² Ω_p` (transmission losses
//! aside), where `Ω_p` is the **projected solid angle of the real exit pupil
//! seen from the image point**: the area, in the direction-cosine
//! coordinates `(L, M)` of the image surface's tangent plane at that point
//! (`dΩ cos θ = dL dM`), of the set of directions in which light arrives
//! there. The relative illumination is `RI(H) = Ω_p(H)/Ω_p(0)`; `cos⁴`
//! fall-off, pupil distortion and coma, field curvature and vignetting are
//! all included because nothing is assumed about the pupil.
//!
//! The image point `X₀` of field `H` is where its real chief ray meets the
//! image surface. The rays arriving at `X₀` do **not** all come from the
//! object point of `H`: with defocus, field curvature or any aberration the
//! beam of one object point is spread around `X₀`, and the rays through
//! `X₀` come from neighbouring object points. They are found by real-ray
//! aiming. With a copy of the tracer that uses `RayAiming::Real` (unless the
//! system already does), normalised pupil coordinates `P` map linearly onto
//! the stop's clear aperture for every object point. So `P` labels the
//! stop point of a ray and `q` its object point: direction tangents
//! `(d_x/d_z, d_y/d_z)` at infinity, object-surface `(x, y)` otherwise. For
//! each pupil node `P`, the object point `q(P)` whose ray through stop point
//! `P` lands on `X₀` solves `X(q, P) = X₀`. It is found by a chord-Newton
//! iteration from the chief ray's `q₀`, with the chief Jacobian `∂X/∂q`
//! refreshed when the iteration stalls. Then
//!
//! `Ω_p = ∫∫_{unit disk} s(P) |det (d(L, M)/dP)|_{X = X₀} dP_x dP_y`,
//!
//! `d(L, M)/dP|_X = ∂(L, M)/∂P − ∂(L, M)/∂q · (∂X/∂q)⁻¹ · ∂X/∂P`,
//!
//! the implicit derivative along the solution `q(P)`. All partial
//! derivatives are central differences of unclipped rays (pupil step
//! `10⁻⁴`, object step `10⁻⁵` of the maximum field's `|q|`), and
//! `s(P) ∈ {0, 1}` is the survival of the clipped ray (fixed apertures
//! elsewhere vignette the beam). An ideal, aberration-free system has
//! `q(P) = q₀` and `∂X/∂P = 0`, so the second term vanishes. The integral
//! uses polar coordinates, with Gauss–Legendre nodes in `ρ` and the
//! trapezoidal rule in `φ` (spectrally accurate for the periodic
//! integrand). An unvignetted beam is therefore integrated to near machine
//! precision, while a vignetting boundary limits the accuracy to `O(1/N)`.
//! The **vignetting fraction** is the clipped share of the stop area,
//! `1 − ∫∫ s dP / π`.
//!
//! This is exact (geometrically) for a uniform Lambertian object of
//! unlimited extent. It agrees with the étendue form
//! `E ∝ A_⊥ dΩ_obj/dA_img` (beam cross-section times the chief-ray mapping
//! Jacobian) to first order in the aberrations. The two forms differ only
//! by how the aberrations vary across the blur, and the tests check them
//! against each other on real lenses. Measuring the beam of *one* object
//! point instead would give the solid angle at that beam's own astigmatic
//! foci. Such a measure scales with field curvature and is not the image
//! irradiance. Cost: about 15 traced rays per pupil node and field.
//!
//! # Footprints
//!
//! A pupil pattern is traced (clipping on) up to a chosen surface for each
//! selected field and wavelength; the hit points are reported in that
//! surface's local frame, with the rays clipped *at* that surface flagged.
//! Rays lost before it are counted but not shown.

use std::f64::consts::PI;

use mklight_core::job::RunTimer;
use mklight_core::math::KahanSum;
use mklight_core::{ApertureShape, JobContext, MkError, MkResult, RayAiming, RunInfo, Vec3};
use mklight_trace::{
    FieldPoint, Grid, ObjectSource, PreparedField, SemiDiameterSettings, SequentialTracer,
    TraceOptions, TraceStatus, TracerSettings, trace_grid_job,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{
    FieldSelection, PupilPattern, WavelengthSelection, check_count, field_scale, field_unit,
    gauss_legendre_01, m2, nf, pupil_points, run_info, wavelength_or_primary,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RelativeIlluminationSettings {
    /// Samples per selected field axis (signed where symmetry requires it).
    pub field_samples: usize,
    pub field_axes: crate::util::FieldAxes,
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
            field_axes: Default::default(),
            wavelength: None,
            radial_samples: 12,
            azimuth_samples: 48,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelativeIlluminationResult {
    pub field_coordinates: Vec<[f64; 2]>,
    pub field_values: Vec<f64>,
    pub normalized_field: Vec<f64>,
    pub field_unit: String,
    pub wavelength: usize,
    /// `Ω_p(H)/Ω_p(0)`.
    #[serde(with = "nf::vec")]
    pub relative_illumination: Vec<f64>,
    /// Projected solid angle of the exit pupil seen from the image point
    /// (sr, about the image surface normal).
    #[serde(with = "nf::vec")]
    pub projected_solid_angle: Vec<f64>,
    /// Clipped share of the stop area.
    #[serde(with = "nf::vec")]
    pub vignetting_fraction: Vec<f64>,
    /// Image-space chief-ray angle with the image surface normal (degrees).
    #[serde(with = "nf::vec")]
    pub chief_ray_angle_deg: Vec<f64>,
    /// `cos⁴` of the object-space chief-ray angle (for comparison).
    #[serde(with = "nf::vec")]
    pub cos4_object: Vec<f64>,
    pub run_info: RunInfo,
}

/// Object-source parameters `q` of a source (module docs).
fn source_params(src: &ObjectSource) -> [f64; 2] {
    match *src {
        ObjectSource::Direction { dir } => [dir.x / dir.z, dir.y / dir.z],
        ObjectSource::Point { point } => [point.x, point.y],
    }
}

/// The rays of one field's image point `X₀` (module docs).
struct Probe<'a> {
    rt: &'a SequentialTracer,
    base: &'a PreparedField,
    wl: usize,
    /// Image point (image local `x, y`).
    x0: [f64; 2],
    /// Tangent basis of the image surface at `X₀` (direction cosines).
    t1: Vec3,
    t2: Vec3,
    /// Object step of the `q` differences.
    hq: f64,
}

/// Outcome of one pupil node.
enum Node {
    /// Projected-solid-angle density `|det d(L, M)/dP|`.
    Ok(f64),
    Vignetted,
    /// No ray through this stop point reaches `X₀`.
    Failed,
}

impl Probe<'_> {
    fn field(&self, q: [f64; 2]) -> Option<PreparedField> {
        let source = if self.rt.system().object_at_infinity {
            ObjectSource::Direction {
                dir: Vec3::new(q[0], q[1], 1.0).normalized(),
            }
        } else {
            let z = mklight_core::surface_math::evaluate(
                &self.rt.system().surfaces[0].shape,
                q[0],
                q[1],
            )?
            .sag;
            ObjectSource::Point {
                point: Vec3::new(q[0], q[1], z),
            }
        };
        Some(PreparedField {
            point: self.base.point,
            source,
            aims: self.base.aims.clone(),
            error: None,
        })
    }

    /// Image point and direction cosines of the ray `(q, P)`.
    fn ray(&self, q: [f64; 2], p: [f64; 2], opts: &TraceOptions) -> Option<[f64; 4]> {
        let f = self.field(q)?;
        let r = self.rt.trace(&f, self.wl, p[0], p[1], opts);
        let d = r.ray.dir;
        r.status
            .is_ok()
            .then_some([r.ray.pos.x, r.ray.pos.y, d.dot(self.t1), d.dot(self.t2)])
    }

    /// Central differences of `(X, L, M)` in `q` (columns 0, 1) and `P`
    /// (columns 2, 3).
    fn jacobian(&self, q: [f64; 2], p: [f64; 2]) -> Option<[[f64; 4]; 4]> {
        const D: f64 = 1e-4;
        let un = TraceOptions::unclipped();
        let mut j = [[0.0; 4]; 4];
        for k in 0..4 {
            let (h, mut qa, mut qb, mut pa, mut pb) = (if k < 2 { self.hq } else { D }, q, q, p, p);
            if k < 2 {
                qa[k] += h;
                qb[k] -= h;
            } else {
                pa[k - 2] += h;
                pb[k - 2] -= h;
            }
            let (a, b) = (self.ray(qa, pa, &un)?, self.ray(qb, pb, &un)?);
            for i in 0..4 {
                j[i][k] = (a[i] - b[i]) / (2.0 * h);
            }
        }
        Some(j)
    }

    /// `(∂X/∂q)⁻¹` at `(q, P)`.
    fn x_q_inverse(&self, q: [f64; 2], p: [f64; 2]) -> Option<m2::M> {
        let j = self.jacobian_xq(q, p)?;
        m2::inv(&j)
    }

    fn jacobian_xq(&self, q: [f64; 2], p: [f64; 2]) -> Option<m2::M> {
        let un = TraceOptions::unclipped();
        let mut j = [[0.0; 2]; 2];
        for k in 0..2 {
            let (mut qa, mut qb) = (q, q);
            qa[k] += self.hq;
            qb[k] -= self.hq;
            let (a, b) = (self.ray(qa, p, &un)?, self.ray(qb, p, &un)?);
            for i in 0..2 {
                j[i][k] = (a[i] - b[i]) / (2.0 * self.hq);
            }
        }
        Some(j)
    }

    /// Solve `X(q, P) = X₀` from `q₀` (chord Newton with refresh), then
    /// the projected-solid-angle density at the solution.
    fn node(&self, q0: [f64; 2], j0inv: &m2::M, p: [f64; 2]) -> Node {
        let un = TraceOptions::unclipped();
        let tol = 1e-10 * (1.0 + self.x0[0].hypot(self.x0[1]));
        let (mut q, mut jinv, mut prev) = (q0, *j0inv, f64::INFINITY);
        let mut converged = false;
        for _ in 0..40 {
            let Some(x) = self.ray(q, p, &un) else {
                return Node::Failed;
            };
            let r = [x[0] - self.x0[0], x[1] - self.x0[1]];
            let rn = r[0].hypot(r[1]);
            if rn <= tol {
                converged = true;
                break;
            }
            if rn > 0.5 * prev {
                match self.x_q_inverse(q, p) {
                    Some(j) => jinv = j,
                    None => return Node::Failed,
                }
            }
            prev = rn;
            let dq = m2::apply(&jinv, r);
            q = [q[0] - dq[0], q[1] - dq[1]];
        }
        if !converged {
            return Node::Failed;
        }
        if self.ray(q, p, &self.rt.options()).is_none() {
            return Node::Vignetted;
        }
        let Some(j) = self.jacobian(q, p) else {
            return Node::Failed;
        };
        let xq = [[j[0][0], j[0][1]], [j[1][0], j[1][1]]];
        let xp = [[j[0][2], j[0][3]], [j[1][2], j[1][3]]];
        let lq = [[j[2][0], j[2][1]], [j[3][0], j[3][1]]];
        let lp = [[j[2][2], j[2][3]], [j[3][2], j[3][3]]];
        let Some(xqi) = m2::inv(&xq) else {
            return Node::Failed;
        };
        let corr = m2::mul(&lq, &m2::mul(&xqi, &xp));
        let d = m2::add(&lp, &m2::scale(&corr, -1.0));
        Node::Ok(m2::det(&d).abs())
    }
}

/// One field's integrals: `(Ω_p, surviving share of the pupil area, chief
/// angle, cos⁴)` and the number of failed pupil nodes. `None` when the
/// chief ray fails.
fn integrate_field(
    rt: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    nodes: &[(f64, f64, f64)],
    q_scale: f64,
    ctx: &JobContext,
) -> MkResult<Option<([f64; 4], usize)>> {
    let sys = rt.system();
    let img = sys.image_index();
    let mut rec = Vec::with_capacity(sys.num_surfaces());
    let chief = rt.trace_with(
        field,
        wl,
        0.0,
        0.0,
        &TraceOptions::unclipped(),
        Some(&mut rec),
    );
    if !chief.status.is_ok() || rec.len() != img + 1 {
        return Ok(None);
    }
    let nrm = rec[img].normal;
    let mut t1 = Vec3::X - nrm * nrm.x;
    if t1.norm() < 1e-6 {
        t1 = Vec3::Y - nrm * nrm.y;
    }
    let t1 = t1.normalized();
    let probe = Probe {
        rt,
        base: field,
        wl,
        x0: [chief.ray.pos.x, chief.ray.pos.y],
        t1,
        t2: nrm.cross(t1),
        hq: 1e-5 * q_scale,
    };
    let q0 = source_params(&field.source);
    let Some(j0inv) = probe.x_q_inverse(q0, [0.0, 0.0]) else {
        return Ok(None);
    };
    let parts: Vec<(f64, f64, bool)> = nodes
        .par_iter()
        .map(|&(px, py, w)| {
            ctx.check()?;
            Ok(match probe.node(q0, &j0inv, [px, py]) {
                Node::Ok(j) => (w * j, w, false),
                Node::Vignetted => (0.0, 0.0, false),
                Node::Failed => (0.0, 0.0, true),
            })
        })
        .collect::<MkResult<_>>()?;
    let omega: KahanSum = parts.iter().map(|p| p.0).collect();
    let area: KahanSum = parts.iter().map(|p| p.1).collect();
    let total: KahanSum = nodes.iter().map(|n| n.2).collect();
    let failed = parts.iter().filter(|p| p.2).count();
    let cra = chief.ray.dir.dot(nrm).abs().min(1.0).acos().to_degrees();
    let cos4 = match rt.generate(field, wl, 0.0, 0.0) {
        Ok(g) => {
            let mut d = g.ray.dir;
            if !sys.object_at_infinity {
                d = sys.surfaces[0].frame.to_global_dir(d);
            }
            d.z.abs().powi(4)
        }
        Err(_) => f64::NAN,
    };
    Ok(Some((
        [omega.value(), area.value() / total.value(), cra, cos4],
        failed,
    )))
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
    ctx.check()?;
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
    let sweep = crate::util::field_sweep(tracer, settings.field_samples, settings.field_axes);
    let hs = &sweep.heights;
    let mut rows = Vec::with_capacity(hs.len());
    let mut warnings = Vec::new();
    ctx.report(0.0, "relative illumination");
    // Scale of the object parameters `q` for the difference steps.
    let q_scale = rt
        .prepare_field(FieldPoint::normalized(0.0, 1.0))
        .map(|f| {
            let q = source_params(&f.source);
            q[0].hypot(q[1])
        })
        .unwrap_or(0.0)
        .max(1e-3);
    for k in 0..hs.len() {
        ctx.check()?;
        let [hx, hy] = sweep.coordinates[k];
        let row = match rt.prepare_field(FieldPoint::normalized(hx, hy)) {
            Ok(f) if f.aims[wl].converged => {
                match integrate_field(rt, &f, wl, &nodes, q_scale, ctx)? {
                    Some((row, failed)) => {
                        if failed > 0 {
                            warnings.push(format!(
                                "field ({hx}, {hy}): {failed} of {} pupil nodes have no ray \
                                 through the image point",
                                nodes.len()
                            ));
                        }
                        row
                    }
                    None => {
                        warnings.push(format!(
                            "field ({hx}, {hy}): chief ray or its object Jacobian failed"
                        ));
                        [f64::NAN; 4]
                    }
                }
            }
            _ => {
                warnings.push(format!("field ({hx}, {hy}): chief ray could not be aimed"));
                [f64::NAN; 4]
            }
        };
        rows.push(row);
        ctx.report((k + 1) as f64 / hs.len() as f64, "relative illumination");
    }
    let omega0 = if let Some(k) = sweep.coordinates.iter().position(|p| *p == [0.0; 2]) {
        rows[k][0]
    } else {
        let on_axis = rt.prepare_field(FieldPoint::normalized(0.0, 0.0))?;
        integrate_field(rt, &on_axis, wl, &nodes, q_scale, ctx)?.map_or(f64::NAN, |(row, _)| row[0])
    };
    if !(omega0 > 0.0) {
        return Err(MkError::Numerical(
            "on-axis beam has no projected solid angle".into(),
        ));
    }
    let scale = field_scale(tracer);
    Ok(RelativeIlluminationResult {
        field_values: hs.iter().map(|h| h * scale).collect(),
        field_unit: field_unit(tracer.system().system().fields.kind).to_string(),
        normalized_field: hs.clone(),
        field_coordinates: sweep.coordinates,
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
    pub field_values: Vec<[f64; 2]>,
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
    let fields_tracer = crate::util::default_fields(tracer, &settings.fields)?;
    let tracer = fields_tracer.as_ref();
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
        field_values: (0..tracer.fields().len())
            .map(|i| crate::util::field_value(tracer, i))
            .collect(),
        surface: s,
        semi_diameter: sd.values[s],
        aperture: sys.surfaces[s].aperture.clone(),
        rays: out,
        extent,
        max_radius: max_r,
        run_info: run_info(timer, tracer, "footprint", settings, seed, Vec::new()),
    })
}
