//! Ray generation from normalised field and pupil coordinates, with ray
//! aiming (CONVENTIONS §3), on top of the per-ray core in [`crate::trace`].
//!
//! # Fields
//!
//! A [`FieldPoint`] holds normalised field coordinates `(Hx, Hy)` and
//! vignetting factors. The physical field value is `(Hx, Hy)·max_radial`
//! in the system's [`FieldKind`] units, and is resolved once into an
//! [`ObjectSource`]:
//!
//! * `ObjectAngle` `(θx, θy)`: chief direction `∝ (tan θx, tan θy, 1)` in the
//!   entrance (global) frame. For a finite object, the object point is where
//!   the chief line through the paraxial entrance-pupil centre meets the
//!   object surface.
//! * `ObjectHeight` `(x, y)`: the object-surface point `(x, y, sag)`.
//! * `ParaxialImageHeight` `(x', y')`: the field whose paraxial chief ray lands
//!   at `(x', y')` on the image surface (`tan θ = h'/(h' per unit tan θ)`, or
//!   object height `= h'/(h' per unit object height)` for finite objects).
//! * `RealImageHeight` `(x', y')`: the field whose **real** chief ray (primary
//!   wavelength, current aiming mode) lands at `(x', y')`, solved by 2-D
//!   Newton iteration from the paraxial guess.
//!
//! # Pupils and aiming
//!
//! Pupil coordinates `(Px, Py)` are first mapped through the field's
//! vignetting factors, `P' = vd + (1 − vc)·P`. Rays are then parameterised by
//! an *aim point* `a = (a_x, a_y)`:
//!
//! * normally the point where the object-space ray crosses the paraxial
//!   entrance-pupil plane `z = z_EP` (global frame, primary wavelength); for
//!   an object at infinity the ray has the field direction `d` and starts at
//!   `a − d (a·d)`, i.e. on the plane through the global origin normal to `d`,
//!   which is the common plane wavefront of the field;
//! * for a finite, object-space telecentric system, the direction tangents of
//!   the object-space ray.
//!
//! With **aiming off**, `a = P'·r_EP` (or `P'·u_marginal`). With **real
//! aiming**, `a` solves `stop_hit(a) = P'·r_stop` in the stop's local frame.
//! `r_stop` is the stop's own size when the aperture floats by it. For any
//! other aperture it is the paraxial stop semi-diameter when the system's
//! ray aiming is off (a tracer that aims such a system anyway keeps it),
//! and when it is real the stop height of the real on-axis marginal ray
//! (the mean of the four rays through the rim of the paraxial entrance
//! pupil, primary wavelength), so that the on-axis beam is the aperture
//! specification's with aiming on and off. The solve is a Newton iteration
//! ([`crate::numeric`]) seeded by a
//! linear prediction from the cached chief-ray aim point and Jacobian of the
//! (field, wavelength). The Jacobian `∂(stop hit)/∂a` is **analytic**: the
//! tangent of the starting ray with respect to `a` is carried through the
//! surfaces of the stop trace ([`crate::diff`]), so a fresh Jacobian costs
//! one trace instead of four, and every accepted step uses the exact
//! Jacobian at its point ([`AIM_POLICY`]). Where the tangent cannot be
//! formed (grazing incidence, near-TIR, an undefined surface curvature) the
//! solver falls back to central differences; [`AimStats`] counts this. If
//! the direct solve fails, it falls back to continuation in the pupil
//! radius (25 %, 50 %, 75 %, 100 %). Aiming ignores clipping. At infinity,
//! chief-ray recovery blends unit directions from the axis and keeps the
//! entrance hit between steps. Convergence is reported per ray.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use mklight_core::{
    ApertureSpec, FieldKind, MkError, MkResult, OpticalSystem, RayAiming, ResolvedSystem, Vec3,
    Vignetting,
};
use serde::{Deserialize, Serialize};

use crate::diff::{RayTangent, TangentPlan};
use crate::numeric::{
    Eval, JacobianPolicy, Mat2, Newton2, Solve2, fd_jacobian, newton2_seeded, solve2,
};
use crate::paraxial::{FirstOrder, Paraxial};
use crate::surface::{IntersectSettings, intersect};
use crate::trace::{
    FailReason, Ray, SurfaceRecord, TraceOptions, TraceResult, TraceStatus, start_frame, trace_ray,
    trace_rays, trace_rays_recorded,
};

/// How real-aim solves form their Jacobians (module docs).
pub const AIM_POLICY: JacobianPolicy = JacobianPolicy::AnalyticRefresh;

/// Per-ray error allocation of a derived aim stop ([`AimStop::Conditioned`]):
/// the propagated error of an accepted stop residual must stay within these
/// absolute bounds on every surface of the ray (CONVENTIONS §7).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AimBudget {
    /// Ray position on any surface (mm).
    pub position_mm: f64,
    /// Direction cosines on any surface.
    pub direction: f64,
    /// OPD against the field's reference sphere (waves).
    pub opd_waves: f64,
    /// The derived stop is never looser than this fraction of the stop
    /// radius, whatever the conditioning says.
    pub max_relative: f64,
    /// Safety factor on the sensitivities sampled at the chief and four
    /// marginal probe rays (pupil variation beyond the samples).
    pub margin: f64,
}

impl AimBudget {
    /// Design tier: one tenth of one third of the §7 budgets (position
    /// 2e-5 mm, direction 2e-6, OPD 5e-3 waves), capped at 1e-6 of the stop
    /// radius.
    pub const DESIGN: AimBudget = AimBudget {
        position_mm: 2e-5 / 30.0,
        direction: 2e-6 / 30.0,
        opd_waves: 5e-3 / 30.0,
        max_relative: 1e-6,
        margin: 2.0,
    };
    /// Preview tier: one tenth of the preview budgets (position 2e-4 mm,
    /// direction 2e-5, OPD RMS 2e-2 waves), capped at 1e-5 of the stop radius.
    pub const PREVIEW: AimBudget = AimBudget {
        position_mm: 2e-4 / 10.0,
        direction: 2e-5 / 10.0,
        opd_waves: 2e-2 / 10.0,
        max_relative: 1e-5,
        margin: 2.0,
    };
}

impl Default for AimBudget {
    fn default() -> Self {
        AimBudget::DESIGN
    }
}

/// When a real-aim Newton solve stops ([`TracerSettings::aim_stop`]).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AimStop {
    /// Stop residual = the system's requested `RayAiming::Real { tolerance }`
    /// (a fraction of the stop radius) for every ray.
    #[default]
    Requested,
    /// Derive the stop residual of every (field, wavelength) from the
    /// propagated error of a stop residual: the sensitivities of the ray
    /// position and direction on every surface and of the OPD to a stop
    /// displacement, from the analytic tangents ([`crate::diff`]) of the
    /// chief ray and four marginal probe rays, give the largest residual
    /// whose propagated error stays within the budget. The result is clamped
    /// to `[requested, max_relative]`, so it is never tighter than the
    /// request (an ill-conditioned field keeps the request and is warned
    /// about) and never looser than the cap. Chief rays are always aimed at
    /// the requested tolerance. Only the model-default tolerance is derived:
    /// an explicit tolerance ([`RayAiming::explicit_tolerance`]) is honoured
    /// exactly for every ray, as under [`AimStop::Requested`].
    Conditioned(AimBudget),
}

/// Sensitivities of one (field, wavelength) to a stop residual: the largest
/// propagated error per unit residual (mm per mm, direction per mm, waves
/// per mm) over the surfaces of the chief and marginal probe rays
/// ([`AimStop::Conditioned`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimConditioning {
    pub position: f64,
    pub direction: f64,
    pub opd: f64,
    /// Probe rays that contributed (chief plus up to four marginal rays).
    pub probes: u32,
}

/// Largest singular value of the `3 × 2` matrix `[a, b]` (columns).
fn spectral_norm(a: Vec3, b: Vec3) -> f64 {
    let (aa, bb, ab) = (a.dot(a), b.dot(b), a.dot(b));
    let tr = aa + bb;
    let det = aa.mul_add(bb, -(ab * ab)).max(0.0);
    (0.5 * (tr + (tr * tr - 4.0 * det).max(0.0).sqrt())).sqrt()
}

/// Work counters of real-aim solves ([`SequentialTracer::generate_many_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AimStats {
    /// Rays that needed a real-aim solve.
    pub rays: u64,
    /// Rays whose direct solve failed and that continued in the pupil radius.
    pub continued: u64,
    /// Rays that could not be aimed.
    pub failed: u64,
    /// Newton iterations (direct and continuation solves).
    pub iterations: u64,
    /// Stop traces without a Jacobian.
    pub traces: u64,
    /// Stop traces with the analytic Jacobian (records and differentiation).
    pub jacobian_traces: u64,
    /// Finite-difference Jacobians formed because the analytic one was
    /// undefined (four of `traces` each).
    pub fd_fallbacks: u64,
}

impl AimStats {
    fn add(&mut self, s: &Solve2) {
        self.iterations += u64::from(s.iterations);
        self.traces += u64::from(s.evaluations);
        self.jacobian_traces += u64::from(s.jacobian_evaluations);
        self.fd_fallbacks += u64::from(s.fd_fallbacks);
    }
    /// Mean Newton iterations per aimed ray.
    pub fn mean_iterations(&self) -> f64 {
        self.iterations as f64 / self.rays.max(1) as f64
    }
    /// Mean stop traces (plain and with Jacobian) per aimed ray.
    pub fn mean_traces(&self) -> f64 {
        (self.traces + self.jacobian_traces) as f64 / self.rays.max(1) as f64
    }
}

/// A field point in normalised coordinates (CONVENTIONS §3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldPoint {
    pub hx: f64,
    pub hy: f64,
    #[serde(default)]
    pub vignetting: Vignetting,
}

impl FieldPoint {
    /// An arbitrary normalised field point without vignetting factors.
    pub fn normalized(hx: f64, hy: f64) -> Self {
        FieldPoint {
            hx,
            hy,
            vignetting: Vignetting::default(),
        }
    }
    /// The system's field `index`, with its vignetting factors.
    pub fn system(sys: &OpticalSystem, index: usize) -> Self {
        let m = sys.fields.max_radial();
        let f = &sys.fields.fields[index];
        FieldPoint {
            hx: f.x / m,
            hy: f.y / m,
            vignetting: f.vignetting,
        }
    }
}

/// Where the rays of a field come from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectSource {
    /// Object at infinity: unit direction in the entrance (global) frame.
    Direction { dir: Vec3 },
    /// Finite object: point in the object surface's local frame.
    Point { point: Vec3 },
}

/// Cached aiming data of one (field, wavelength).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimData {
    /// Aim point of the chief ray.
    pub chief: [f64; 2],
    /// `∂(stop hit)/∂(aim point)` at the chief ray (real aiming only).
    pub jacobian: Option<Mat2>,
    pub converged: bool,
    /// Chief-ray stop residual as a fraction of the stop radius.
    pub residual: f64,
    /// Stop residual of this (field, wavelength)'s pupil aims as a fraction
    /// of the stop radius when derived by [`AimStop::Conditioned`]; `None`
    /// uses the requested tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_tolerance: Option<f64>,
    /// The sensitivities behind `stop_tolerance`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditioning: Option<AimConditioning>,
}

/// A field resolved for tracing: its source and per-wavelength aim cache.
///
/// A system field that cannot be resolved (e.g. a real image height that no
/// chief ray reaches) is still prepared, with `error` set: every ray traced
/// from it is reported as `Failed { surface: 0, reason: FieldUnreachable }`,
/// so the other fields stay usable (optimiser trial steps, partial results).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedField {
    pub point: FieldPoint,
    /// Placeholder (on-axis) when `error` is set.
    pub source: ObjectSource,
    /// Indexed by system wavelength.
    pub aims: Vec<AimData>,
    /// Why the field could not be resolved; `None` for a usable field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl PreparedField {
    /// `true` unless the field could not be resolved.
    pub fn is_valid(&self) -> bool {
        self.error.is_none()
    }
}

/// Convergence report of one generated ray.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimReport {
    pub iterations: u32,
    /// Stop residual as a fraction of the stop radius (0 with aiming off).
    pub residual: f64,
    pub converged: bool,
}

/// A generated starting ray (in the starting frame, see [`crate::trace`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratedRay {
    pub ray: Ray,
    /// Aim parameters (module docs).
    pub aim: [f64; 2],
    pub report: AimReport,
}

/// One ray of a batch: prepared field, system wavelength index and
/// (pre-vignetting) pupil coordinates `[px, py]`.
pub type RayRequest<'a> = (&'a PreparedField, usize, [f64; 2]);

/// [`SequentialTracer::generate`] up to the real-aim solve.
enum AimStart {
    Ready(GeneratedRay),
    Solve {
        target: [f64; 2],
        x0: [f64; 2],
        j0: Option<Mat2>,
        /// Rounding floor when `x0` is an anchor seed (`newton2_seeded`).
        seed: Option<f64>,
        key: Option<AimKey>,
    },
}

/// Identity of one aimed ray: field point (with vignetting), wavelength and
/// pupil point, as bits.
type AimKey = [u64; 9];

/// A recorded aim and its aim Jacobian.
type RaySeed = ([f64; 2], Option<Mat2>);

fn aim_key(field: &PreparedField, wl: usize, px: f64, py: f64) -> AimKey {
    let (p, v) = (field.point, field.point.vignetting);
    [
        p.hx.to_bits(),
        p.hy.to_bits(),
        v.vdx.to_bits(),
        v.vdy.to_bits(),
        v.vcx.to_bits(),
        v.vcy.to_bits(),
        wl as u64,
        px.to_bits(),
        py.to_bits(),
    ]
}

/// Converged solves of an anchor tracer (prepared fields and pupil-ray
/// aims with their Jacobians), seeding the same solves of a nearby system:
/// the finite-difference neighbours of a Jacobian or the trials of a
/// tolerance run. A seeded solve starts at the anchor's solution with the
/// anchor's Jacobian (the aim's stop Jacobian, recorded with it) and takes
/// at least one Newton step, so it corrects for the change even when the
/// seed already meets the tolerance; it must then resolve the change to
/// 1e-6 of itself down to rounding, never looser than the aiming tolerance
/// (`newton2_seeded`), and falls back to the unseeded solve otherwise.
/// Results agree with the unseeded tracer to within the aiming tolerance
/// (not bit for bit) and do not depend on the order of tracing. Recording
/// costs one Jacobian trace per anchor ray.
#[derive(Debug, Default)]
pub struct AimSeeds {
    fields: RwLock<Vec<PreparedField>>,
    rays: RwLock<HashMap<AimKey, RaySeed>>,
}

impl AimSeeds {
    /// Recorded pupil-ray aims.
    pub fn rays(&self) -> usize {
        self.rays.read().unwrap().len()
    }
    fn field(&self, point: &FieldPoint, nw: usize) -> Option<PreparedField> {
        self.fields
            .read()
            .unwrap()
            .iter()
            .find(|f| f.point == *point && f.is_valid() && f.aims.len() == nw)
            .cloned()
    }
    fn ray(&self, key: &AimKey) -> Option<RaySeed> {
        self.rays.read().unwrap().get(key).copied()
    }
}

/// A caller's start of a real-aim solve: the aim point and the stop Jacobian
/// there ([`SequentialTracer::generate_many_from`]).
type AimStartPoint = ([f64; 2], Mat2);

/// How a tracer uses [`AimSeeds`].
#[derive(Clone, Debug)]
enum Seeds {
    /// The anchor: converged solves are written.
    Record(Arc<AimSeeds>),
    /// A nearby system: solves start from the anchor's.
    Read(Arc<AimSeeds>),
}

/// Settings of a [`SequentialTracer`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TracerSettings {
    /// Overrides the system's ray-aiming mode when set.
    pub aiming: Option<RayAiming>,
    pub max_aim_iterations: u32,
    pub intersect: IntersectSettings,
    /// When the real-aim solves stop (requested tolerance, or derived from
    /// the propagated error).
    #[serde(default)]
    pub aim_stop: AimStop,
}

impl Default for TracerSettings {
    fn default() -> Self {
        TracerSettings {
            aiming: None,
            max_aim_iterations: 40,
            intersect: IntersectSettings::default(),
            aim_stop: AimStop::Requested,
        }
    }
}

/// Primary-wavelength pupil geometry used for ray generation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilGeometry {
    /// Entrance-pupil plane `z` in the entrance (global) frame.
    pub z_ep: f64,
    pub r_ep: f64,
    /// Object-space telecentric (finite object): aim parameters are tangents.
    pub telecentric: bool,
    /// Marginal object-space slope (telecentric aim scale).
    pub u_marginal: f64,
    /// The stop radius that pupil coordinates refer to under real aiming:
    /// the paraxial stop semi-diameter of the aperture specification, or,
    /// when the system's ray aiming is real and the aperture is not the
    /// stop size itself, the stop height of the real on-axis marginal ray.
    pub r_stop: f64,
}

/// The reference sequential tracer for one resolved system: paraxial data,
/// pupil geometry and prepared system fields. Immutable after construction
/// and `Sync`, so batches share it across threads.
#[derive(Clone, Debug)]
pub struct SequentialTracer {
    sys: Arc<ResolvedSystem>,
    paraxial: Paraxial,
    pupil: PupilGeometry,
    aiming: RayAiming,
    settings: TracerSettings,
    fields: Vec<PreparedField>,
    warnings: Vec<String>,
    tangents: TangentPlan,
    seeds: Option<Seeds>,
}

impl SequentialTracer {
    pub fn new(sys: Arc<ResolvedSystem>) -> MkResult<Self> {
        Self::with_settings(sys, TracerSettings::default())
    }

    pub fn with_settings(sys: Arc<ResolvedSystem>, settings: TracerSettings) -> MkResult<Self> {
        Self::build(sys, settings, None)
    }

    /// [`Self::with_settings`] for a system near the anchor of `seeds`:
    /// field sources, chief aims and pupil-ray aims start from the anchor's
    /// solutions ([`AimSeeds`]).
    pub fn with_seeds(
        sys: Arc<ResolvedSystem>,
        settings: TracerSettings,
        seeds: Arc<AimSeeds>,
    ) -> MkResult<Self> {
        Self::build(sys, settings, Some(Seeds::Read(seeds)))
    }

    /// Make this tracer an anchor: its prepared fields, and the aims of the
    /// pupil rays it generates from now on, are recorded into the returned
    /// seeds for [`Self::with_seeds`].
    pub fn record_seeds(&mut self) -> Arc<AimSeeds> {
        let seeds = Arc::new(AimSeeds {
            fields: RwLock::new(self.fields.clone()),
            rays: Default::default(),
        });
        self.seeds = Some(Seeds::Record(seeds.clone()));
        seeds
    }

    fn build(
        sys: Arc<ResolvedSystem>,
        settings: TracerSettings,
        seeds: Option<Seeds>,
    ) -> MkResult<Self> {
        let paraxial = Paraxial::new(&sys)?;
        let p = paraxial.primary();
        let pupil = PupilGeometry {
            z_ep: p.entrance_pupil_position,
            r_ep: p.entrance_pupil_radius(),
            telecentric: p.object_space_telecentric(),
            u_marginal: p.marginal_object_slope,
            r_stop: p.stop_semi_diameter,
        };
        let aiming = settings.aiming.unwrap_or(sys.system().ray_aiming);
        let mut warnings = sys.warnings.clone();
        warnings.extend(paraxial.warnings.iter().cloned());
        let tangents = TangentPlan::new(&sys);
        let mut t = SequentialTracer {
            sys,
            paraxial,
            pupil,
            aiming,
            settings,
            fields: Vec::new(),
            warnings,
            tangents,
            seeds,
        };
        // The stop radius belongs to the system: its own aiming mode sets
        // it, not a tracer's override of that mode.
        if matches!(t.sys.system().ray_aiming, RayAiming::Real { .. })
            && t.sys.system().aperture != ApertureSpec::FloatByStopSize
        {
            match t.marginal_stop_radius() {
                Some(r) => t.pupil.r_stop = r,
                None => t.warnings.push(
                    "the on-axis marginal rays do not reach the stop; the paraxial stop radius \
                     is used"
                        .into(),
                ),
            }
        }
        // Fields are independent solves; prepare them on the rayon pool
        // (in order, so results and warnings do not depend on threads).
        let n = t.sys.system().fields.fields.len();
        let prepared: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = FieldPoint::system(t.sys.system(), i);
                (point, t.prepare_field(point))
            })
            .collect();
        let mut fields = Vec::with_capacity(n);
        for (i, (point, prepared)) in prepared.into_iter().enumerate() {
            let f = match prepared {
                Ok(f) => {
                    if f.aims.iter().any(|a| !a.converged) {
                        t.warnings
                            .push(format!("field {i}: chief-ray aiming did not converge"));
                    }
                    f
                }
                Err(e) => {
                    t.warnings.push(format!("field {i} unusable: {e}"));
                    t.failed_field(point, e.to_string())
                }
            };
            fields.push(f);
        }
        t.fields = fields;
        Ok(t)
    }

    pub fn system(&self) -> &ResolvedSystem {
        &self.sys
    }
    pub fn system_arc(&self) -> Arc<ResolvedSystem> {
        self.sys.clone()
    }
    pub fn paraxial(&self) -> &Paraxial {
        &self.paraxial
    }
    pub fn pupil(&self) -> &PupilGeometry {
        &self.pupil
    }
    pub fn aiming(&self) -> RayAiming {
        self.aiming
    }
    pub fn settings(&self) -> &TracerSettings {
        &self.settings
    }
    /// Prepared system fields, in system order.
    pub fn fields(&self) -> &[PreparedField] {
        &self.fields
    }
    pub fn field(&self, i: usize) -> &PreparedField {
        &self.fields[i]
    }
    /// The per-surface plan of the analytic differential tracer.
    pub fn tangent_plan(&self) -> &TangentPlan {
        &self.tangents
    }
    /// Append analysis-local fields without changing the physical snapshot.
    /// Existing field indices keep their meaning; new ones follow them.
    pub fn with_additional_fields(&self, points: &[FieldPoint]) -> MkResult<Self> {
        // Independent solves, prepared on the rayon pool and taken in order
        // (the first failing point is the error).
        let prepared: Vec<_> = points
            .par_iter()
            .map(|&point| self.prepare_field(point))
            .collect();
        let mut tracer = self.clone();
        for field in prepared {
            tracer.fields.push(field?);
        }
        Ok(tracer)
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
    pub fn num_wavelengths(&self) -> usize {
        self.sys.wavelengths_um.len()
    }
    /// Default trace options (clipping on, this tracer's intersection settings).
    pub fn options(&self) -> TraceOptions {
        TraceOptions {
            intersect: self.settings.intersect,
            ..Default::default()
        }
    }

    // ----- Ray construction --------------------------------------------------

    /// The starting ray for aim parameters `a` (module docs).
    pub fn start_ray(&self, src: &ObjectSource, a: [f64; 2]) -> Option<Ray> {
        match *src {
            ObjectSource::Direction { dir } => {
                let ap = Vec3::new(a[0], a[1], self.pupil.z_ep);
                Some(Ray::new(ap - dir * ap.dot(dir), dir))
            }
            ObjectSource::Point { point } => {
                let f0 = start_frame(&self.sys);
                if self.pupil.telecentric {
                    let dg = Vec3::new(a[0], a[1], 1.0).normalized();
                    return Some(Ray::new(point, f0.to_local_dir(dg)));
                }
                let al = f0.to_local_point(Vec3::new(a[0], a[1], self.pupil.z_ep));
                let mut d = (al - point).normalized();
                let gz = f0.to_global_dir(d).z;
                if gz == 0.0 || !gz.is_finite() || !d.is_finite() {
                    return None;
                }
                if gz < 0.0 {
                    // The aim point is behind the object (virtual pupil).
                    d = -d;
                }
                Some(Ray::new(point, d))
            }
        }
    }

    fn aim_scale(&self) -> f64 {
        if self.pupil.telecentric {
            self.pupil.u_marginal.abs()
        } else {
            self.pupil.r_ep
        }
    }

    /// Paraxial (aim-off) aim parameters of pupil point `P'`.
    fn paraxial_aim(&self, p: [f64; 2]) -> [f64; 2] {
        let s = self.aim_scale();
        [p[0] * s, p[1] * s]
    }

    /// The starting ray of aim `a` ([`Self::start_ray`], bitwise) and its
    /// tangent `∂ray/∂a` in the starting frame ([`crate::diff`]).
    pub fn start_tangent(&self, src: &ObjectSource, a: [f64; 2]) -> Option<(Ray, RayTangent)> {
        let ray = self.start_ray(src, a)?;
        let tangent = match *src {
            // `pos = a' − d (a'·d)`, `a' = (a_x, a_y, z_EP)`.
            ObjectSource::Direction { dir } => RayTangent {
                dpos: [Vec3::X - dir * dir.x, Vec3::Y - dir * dir.y],
                ddir: [Vec3::ZERO; 2],
            },
            ObjectSource::Point { point } => {
                let f0 = start_frame(&self.sys);
                let ddir = if self.pupil.telecentric {
                    // `d = R₀ᵀ g/|g|`, `g = (a_x, a_y, 1)`.
                    let g = Vec3::new(a[0], a[1], 1.0);
                    let (gn, dg) = (g.norm(), g.normalized());
                    [Vec3::X, Vec3::Y].map(|e| f0.to_local_dir((e - dg * dg.dot(e)) / gn))
                } else {
                    // `d = ±u/|u|`, `u = R₀ᵀ(a' − o₀) − point`.
                    let al = f0.to_local_point(Vec3::new(a[0], a[1], self.pupil.z_ep));
                    let u = al - point;
                    let un = u.norm();
                    let sign = if ray.dir.dot(u) < 0.0 { -1.0 } else { 1.0 };
                    [Vec3::X, Vec3::Y].map(|e| {
                        let du = f0.to_local_dir(e);
                        (du - u * (u.dot(du) / (un * un))) * (sign / un)
                    })
                };
                RayTangent {
                    dpos: [Vec3::ZERO; 2],
                    ddir,
                }
            }
        };
        Some((ray, tangent))
    }

    /// The stop radius of real aiming when the aperture is given in object
    /// or image space (module docs): the mean stop height of the four
    /// on-axis rays through the rim of the paraxial entrance pupil (primary
    /// wavelength, unclipped). `None` when one of them does not reach the
    /// stop.
    fn marginal_stop_radius(&self) -> Option<f64> {
        let src = if self.sys.object_at_infinity {
            ObjectSource::Direction { dir: Vec3::Z }
        } else {
            self.object_point(0.0, 0.0).ok()?
        };
        let wl = self.sys.system().wavelengths.primary;
        let opts = self.stop_options();
        let mut sum = 0.0_f64;
        for p in [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]] {
            let ray = self.start_ray(&src, self.paraxial_aim(p))?;
            let r = trace_ray(&self.sys, &ray, wl, &opts, None);
            if !r.status.is_ok() {
                return None;
            }
            sum += r.ray.pos.x.hypot(r.ray.pos.y);
        }
        let radius = 0.25 * sum;
        (radius.is_finite() && radius > 0.0).then_some(radius)
    }

    fn stop_options(&self) -> TraceOptions {
        TraceOptions {
            clip: false,
            last_surface: Some(self.sys.stop),
            intersect: self.settings.intersect,
        }
    }

    /// The real-aim objective: stop hit `(x, y)` (stop local frame) of the
    /// ray with aim `a` minus `target`, and when `jac` its analytic
    /// Jacobian (`None` where the tangent cannot be formed, module docs).
    /// `scratch` holds the records of the Jacobian trace.
    fn stop_eval(
        &self,
        src: &ObjectSource,
        a: [f64; 2],
        wl: usize,
        target: [f64; 2],
        jac: bool,
        scratch: &mut Vec<SurfaceRecord>,
    ) -> Eval {
        let opts = self.stop_options();
        let residual = |r: &TraceResult| [r.ray.pos.x - target[0], r.ray.pos.y - target[1]];
        if !jac {
            let ray = self.start_ray(src, a)?;
            let r = trace_ray(&self.sys, &ray, wl, &opts, None);
            return r.status.is_ok().then(|| (residual(&r), None));
        }
        let (ray, tangent) = self.start_tangent(src, a)?;
        let r = trace_ray(&self.sys, &ray, wl, &opts, Some(scratch));
        if !r.status.is_ok() {
            return None;
        }
        let j = self
            .tangents
            .differentiate(&self.sys, wl, &tangent, scratch, r.surface)
            .ok()
            .map(|t| t.xy_jacobian());
        Some((residual(&r), j))
    }

    /// `∂(stop hit)/∂a` at `a`: analytic, else central differences, `None`
    /// where the ray misses the stop.
    fn stop_jacobian(&self, src: &ObjectSource, a: [f64; 2], wl: usize) -> Option<Mat2> {
        let scratch = RefCell::new(Vec::new());
        let f = |x: [f64; 2], jac: bool| {
            self.stop_eval(src, x, wl, [0.0, 0.0], jac, &mut scratch.borrow_mut())
        };
        match f(a, true)? {
            (_, Some(j)) => Some(j),
            (fx, None) => fd_jacobian(&f, a, fx, self.fd_step()),
        }
    }

    /// The system's requested aiming tolerance as a fraction of the stop
    /// radius (the model default with aiming off).
    pub fn requested_tolerance(&self) -> f64 {
        match self.aiming {
            RayAiming::Real { tolerance } => tolerance,
            RayAiming::Off => RayAiming::DEFAULT_TOLERANCE,
        }
    }

    /// The stop radius (mm) that relative stop residuals refer to; never
    /// zero.
    fn stop_radius(&self) -> f64 {
        self.pupil.r_stop.max(f64::MIN_POSITIVE)
    }

    /// Absolute stop residual (mm) of relative tolerance `rel`.
    fn stop_mm(&self, rel: f64) -> f64 {
        rel * self.stop_radius()
    }

    /// Absolute stop residual (mm) of the requested tolerance: chief-ray
    /// aims, and every aim under [`AimStop::Requested`].
    fn aim_tolerance(&self) -> f64 {
        self.stop_mm(self.requested_tolerance())
    }

    /// The stop tolerance of a prepared field's pupil aims at wavelength
    /// `wl`, as a fraction of the stop radius: the derived stop
    /// ([`AimStop::Conditioned`]) or the requested tolerance. Rays at the
    /// edge of a clipping stop are aimed at the requested tolerance
    /// (CONVENTIONS §7).
    pub fn aim_tolerance_of(&self, field: &PreparedField, wl: usize) -> f64 {
        field.aims[wl]
            .stop_tolerance
            .unwrap_or_else(|| self.requested_tolerance())
    }

    /// Absolute stop tolerance (mm) of a pupil aim at stop point `target`:
    /// the field's derived stop, or the requested tolerance. Where the stop
    /// clips, a target within the derived stop of its edge is aimed at the
    /// requested tolerance, which the edge tolerance of apertures covers
    /// (CONVENTIONS §2): the ray is not clipped by its own aiming residual.
    fn field_tolerance(&self, field: &PreparedField, wl: usize, target: [f64; 2]) -> f64 {
        let requested = self.aim_tolerance();
        let Some(rel) = field.aims[wl].stop_tolerance else {
            return requested;
        };
        let derived = self.stop_mm(rel);
        match self.stop_edge() {
            Some(r) if target[0].hypot(target[1]) + derived > r => requested,
            _ => derived,
        }
    }

    /// The radius (mm) at which the stop surface itself clips rays: its
    /// aperture's bounding radius or its fixed semi-diameter.
    fn stop_edge(&self) -> Option<f64> {
        let stop = &self.sys.surfaces[self.sys.stop];
        stop.aperture
            .as_ref()
            .map(|a| a.bounding_radius())
            .or(stop.fixed_semi_diameter)
    }

    /// One stop tolerance for every pupil aim of the prepared fields (a
    /// fraction of the stop radius): the tightest derived stop, or the
    /// requested tolerance. Device aiming takes one tolerance per request,
    /// so where the stop clips it is the requested tolerance, which keeps
    /// rays aimed at the stop's edge (CONVENTIONS §7).
    pub fn device_aim_tolerance(&self) -> f64 {
        if self.stop_edge().is_some() {
            return self.requested_tolerance();
        }
        let derived = self
            .fields
            .iter()
            .flat_map(|f| f.aims.iter().filter_map(|a| a.stop_tolerance))
            .fold(f64::INFINITY, f64::min);
        if derived.is_finite() {
            derived
        } else {
            self.requested_tolerance()
        }
    }

    /// Rounding floor of a seeded aim solve (stop coordinates), from the
    /// requested tolerance.
    fn aim_floor(&self) -> f64 {
        (1e-4 * self.aim_tolerance()).max(16.0 * f64::EPSILON * self.pupil.r_stop)
    }

    fn fd_step(&self) -> f64 {
        1e-6 * self.aim_scale().max(1e-6)
    }

    /// Solve the real aim for stop target `target`, seeded at `x0`.
    fn solve_aim(
        &self,
        src: &ObjectSource,
        wl: usize,
        target: [f64; 2],
        x0: [f64; 2],
        j0: Option<Mat2>,
        tol: f64,
    ) -> Solve2 {
        self.solve_aim_seeded(src, wl, target, x0, j0, None, tol)
    }

    /// [`Self::solve_aim`], seeded as `newton2_seeded` when `seed` is set.
    #[allow(clippy::too_many_arguments)]
    fn solve_aim_seeded(
        &self,
        src: &ObjectSource,
        wl: usize,
        target: [f64; 2],
        x0: [f64; 2],
        j0: Option<Mat2>,
        seed: Option<f64>,
        tol: f64,
    ) -> Solve2 {
        let scratch = RefCell::new(Vec::new());
        let f = |a: [f64; 2], jac: bool| {
            self.stop_eval(src, a, wl, target, jac, &mut scratch.borrow_mut())
        };
        newton2_seeded(
            &f,
            x0,
            j0,
            self.fd_step(),
            tol,
            self.settings.max_aim_iterations,
            AIM_POLICY,
            seed,
        )
    }

    /// The seeds this tracer reads, if any.
    fn read_seeds(&self) -> Option<&AimSeeds> {
        match &self.seeds {
            Some(Seeds::Read(s)) => Some(s),
            _ => None,
        }
    }

    fn chief_aim(&self, src: &ObjectSource, wl: usize, seed: Option<&AimData>) -> AimData {
        self.chief_aim_with(src, wl, true, seed)
    }

    /// [`Self::chief_aim`]; without the final aim Jacobian when `jacobian`
    /// is false, for callers that only use the aim. A converged `seed` (the
    /// anchor's aim, [`AimSeeds`]) starts the solve.
    fn chief_aim_with(
        &self,
        src: &ObjectSource,
        wl: usize,
        jacobian: bool,
        seed: Option<&AimData>,
    ) -> AimData {
        match self.aiming {
            RayAiming::Off => AimData {
                chief: [0.0, 0.0],
                jacobian: None,
                converged: true,
                residual: 0.0,
                stop_tolerance: None,
                conditioning: None,
            },
            RayAiming::Real { .. } => {
                let tol = self.aim_tolerance();
                let seeded = seed
                    .filter(|a| a.converged)
                    .map(|a| {
                        let floor = Some(self.aim_floor());
                        self.solve_aim_seeded(src, wl, [0.0, 0.0], a.chief, a.jacobian, floor, tol)
                    })
                    .filter(|s| s.converged);
                let mut s = match seeded {
                    Some(s) => s,
                    None => self.solve_aim(src, wl, [0.0, 0.0], [0.0, 0.0], None, tol),
                };
                if !s.converged {
                    // A wide-field paraxial seed can miss the front surfaces:
                    // continue the source from the axis, ending on `src`
                    // itself. Directions are blended as unit vectors so a
                    // grazing field advances evenly, and each step re-aims
                    // through the previous entrance hit. Any failed step
                    // keeps the first solve.
                    let p = self.source_params(src);
                    let at_infinity = matches!(src, ObjectSource::Direction { .. });
                    let n = if at_infinity { 64 } else { 16 };
                    let steps = (0..n)
                        .map(|k| {
                            let f = k as f64 / n as f64;
                            match *src {
                                ObjectSource::Direction { dir } => Some(ObjectSource::Direction {
                                    dir: (Vec3::Z * (1.0 - f) + dir * f).normalized(),
                                }),
                                ObjectSource::Point { .. } => {
                                    self.source_from_params([f * p[0], f * p[1]])
                                }
                            }
                        })
                        .chain([Some(*src)]);
                    let (mut x, mut j) = ([0.0, 0.0], None);
                    let mut entrance: Option<Vec3> = None;
                    for (k, step) in steps.enumerate() {
                        let Some(step) = step else { break };
                        if let (Some(e), ObjectSource::Direction { dir }) = (entrance, step) {
                            let t = (self.pupil.z_ep - e.z) / dir.z;
                            x = [e.x + t * dir.x, e.y + t * dir.y];
                        }
                        let sk = self.solve_aim(&step, wl, [0.0, 0.0], x, j, tol);
                        if !sk.converged {
                            break;
                        }
                        (x, j) = (sk.x, sk.jacobian);
                        if k == n {
                            s = sk;
                        } else if at_infinity {
                            let ray = self.start_ray(&step, x).unwrap();
                            let options = TraceOptions {
                                clip: false,
                                last_surface: Some(1),
                                intersect: self.settings.intersect,
                            };
                            let hit = trace_ray(&self.sys, &ray, wl, &options, None);
                            entrance =
                                Some(self.sys.surfaces[1].frame.to_global_point(hit.ray.pos));
                        }
                    }
                }
                let jac = if jacobian {
                    self.stop_jacobian(src, s.x, wl).or(s.jacobian)
                } else {
                    s.jacobian
                };
                AimData {
                    chief: s.x,
                    jacobian: jac,
                    converged: s.converged,
                    residual: s.residual / self.stop_radius(),
                    stop_tolerance: None,
                    conditioning: None,
                }
            }
        }
    }

    /// Resolve a field point into its source and per-wavelength aim cache.
    pub fn prepare_field(&self, point: FieldPoint) -> MkResult<PreparedField> {
        let nw = self.num_wavelengths();
        let seed = self.read_seeds().and_then(|s| s.field(&point, nw));
        let source = match seed.as_ref().and_then(|f| self.seeded_source(&point, f)) {
            Some(source) => source,
            None => self.field_source(&point)?,
        };
        let aims = (0..nw)
            .into_par_iter()
            .map(|w| self.chief_aim(&source, w, seed.as_ref().map(|f| &f.aims[w])))
            .collect();
        let mut field = PreparedField {
            point,
            source,
            aims,
            error: None,
        };
        if let (AimStop::Conditioned(budget), RayAiming::Real { .. }) =
            (self.settings.aim_stop, self.aiming)
            && self.aiming.explicit_tolerance().is_none()
        {
            // Each wavelength's conditioning reads only that wavelength's
            // chief aim.
            let conditioning: Vec<_> = (0..nw)
                .into_par_iter()
                .map(|wl| self.conditioning(&field, wl))
                .collect();
            for (aim, c) in field.aims.iter_mut().zip(conditioning) {
                aim.conditioning = c;
                aim.stop_tolerance = c.map(|c| self.derived_tolerance(&budget, &c));
            }
        }
        if let Some(Seeds::Record(s)) = &self.seeds {
            let mut fields = s.fields.write().unwrap();
            if !fields.iter().any(|f| f.point == point) {
                fields.push(field.clone());
            }
        }
        Ok(field)
    }

    /// The derived stop of [`AimStop::Conditioned`] (module docs): the
    /// largest residual whose propagated error fits `budget`, clamped to
    /// `[requested, max_relative]`.
    fn derived_tolerance(&self, budget: &AimBudget, c: &AimConditioning) -> f64 {
        let requested = self.requested_tolerance();
        let r_stop = self.stop_radius();
        let mm = (budget.position_mm / c.position)
            .min(budget.direction / c.direction)
            .min(budget.opd_waves / c.opd)
            / budget.margin;
        let rel = mm / r_stop;
        if !rel.is_finite() {
            return requested;
        }
        rel.min(budget.max_relative).max(requested)
    }

    /// Sensitivities of a prepared field's rays at wavelength `wl` to a
    /// stop residual ([`AimConditioning`]): for the chief ray and four
    /// marginal probes (the chief Jacobian's prediction of the pupil edge,
    /// unaimed) the tangent with respect to the aim point is carried
    /// through an unclipped image trace, and `T_k J_stop⁻¹` (`T_k` the
    /// position or direction tangent at surface `k`, `J_stop` the stop-hit
    /// Jacobian) bounds the error at surface `k` per unit stop residual;
    /// the OPD sensitivity comes from the tangent at the image through the
    /// reference sphere ([`crate::opd::OpdReference::opd_tangent`]).
    /// `None` when the chief ray gives no usable tangent or Jacobian.
    fn conditioning(&self, field: &PreparedField, wl: usize) -> Option<AimConditioning> {
        let ad = &field.aims[wl];
        if !ad.converged {
            return None;
        }
        let r_stop = self.pupil.r_stop;
        let edge = |p: [f64; 2]| {
            let t = [p[0] * r_stop, p[1] * r_stop];
            ad.jacobian
                .and_then(|j| solve2(j, t))
                .map(|d| [ad.chief[0] + d[0], ad.chief[1] + d[1]])
                .unwrap_or_else(|| {
                    let q = self.paraxial_aim(p);
                    [ad.chief[0] + q[0], ad.chief[1] + q[1]]
                })
        };
        let probes = [
            ad.chief,
            edge([1.0, 0.0]),
            edge([-1.0, 0.0]),
            edge([0.0, 1.0]),
            edge([0.0, -1.0]),
        ];
        let reference = self.opd_reference(field, wl).ok();
        let opts = TraceOptions::unclipped();
        let image = self.sys.image_index();
        let stop = self.sys.stop;
        let mut records = Vec::with_capacity(self.sys.surfaces.len());
        let mut chain: Vec<RayTangent> = Vec::with_capacity(image);
        let mut out = AimConditioning {
            position: 0.0,
            direction: 0.0,
            opd: 0.0,
            probes: 0,
        };
        for (k, a) in probes.into_iter().enumerate() {
            let chief = k == 0;
            let Some((ray, tangent)) = self.start_tangent(&field.source, a) else {
                continue;
            };
            let r = trace_ray(&self.sys, &ray, wl, &opts, Some(&mut records));
            if !r.status.is_ok() || r.surface != image {
                if chief {
                    return None;
                }
                continue;
            }
            // Tangents at every surface; the stop Jacobian is among them.
            chain.clear();
            let ok = self
                .tangents
                .differentiate_each(&self.sys, wl, &tangent, &records, image, |_, t| {
                    chain.push(*t)
                })
                .is_ok();
            // M = J_stop⁻¹, column by column.
            let m = ok
                .then(|| chain[stop - 1].xy_jacobian())
                .and_then(|j| Some((solve2(j, [1.0, 0.0])?, solve2(j, [0.0, 1.0])?)));
            let Some((m0, m1)) = m else {
                if chief {
                    return None;
                }
                continue;
            };
            let m = [[m0[0], m1[0]], [m0[1], m1[1]]];
            let apply = |v: &[Vec3; 2]| -> (Vec3, Vec3) {
                (
                    v[0] * m[0][0] + v[1] * m[1][0],
                    v[0] * m[0][1] + v[1] * m[1][1],
                )
            };
            for t in &chain {
                let (a, b) = apply(&t.dpos);
                out.position = out.position.max(spectral_norm(a, b));
                let (a, b) = apply(&t.ddir);
                out.direction = out.direction.max(spectral_norm(a, b));
            }
            if let Some(reference) = &reference
                && let Some(g) = reference.opd_tangent(&r, &chain[image - 1])
            {
                let gm = [
                    g[0] * m[0][0] + g[1] * m[1][0],
                    g[0] * m[0][1] + g[1] * m[1][1],
                ];
                out.opd = out.opd.max(gm[0].hypot(gm[1]));
            }
            out.probes += 1;
        }
        (out.probes > 0
            && out.position.is_finite()
            && out.direction.is_finite()
            && out.opd.is_finite())
        .then_some(out)
    }

    /// The real-image-height source of `point`, solved from the anchor's
    /// `seed` field; `None` for other field kinds, the axis, or when the
    /// seeded solve fails (the caller then solves unseeded).
    fn seeded_source(&self, point: &FieldPoint, seed: &PreparedField) -> Option<ObjectSource> {
        let osys = self.sys.system();
        if !matches!(osys.fields.kind, FieldKind::RealImageHeight) {
            return None;
        }
        let fmax = osys.fields.max_radial();
        let target = [point.hx * fmax, point.hy * fmax];
        if target == [0.0, 0.0] {
            return None;
        }
        let aim = &seed.aims[self.sys.primary_wavelength];
        self.solve_real_image_height_from(self.source_params(&seed.source), target, Some(aim))
            .ok()
    }

    /// A field that could not be resolved (see [`PreparedField`]).
    fn failed_field(&self, point: FieldPoint, error: String) -> PreparedField {
        let source = if self.sys.object_at_infinity {
            ObjectSource::Direction { dir: Vec3::Z }
        } else {
            ObjectSource::Point { point: Vec3::ZERO }
        };
        let aim = AimData {
            chief: [0.0, 0.0],
            jacobian: None,
            converged: false,
            residual: f64::INFINITY,
            stop_tolerance: None,
            conditioning: None,
        };
        PreparedField {
            point,
            source,
            aims: vec![aim; self.num_wavelengths()],
            error: Some(error),
        }
    }

    /// The object source of a normalised field point (module docs).
    pub fn field_source(&self, point: &FieldPoint) -> MkResult<ObjectSource> {
        let osys = self.sys.system();
        let fmax = osys.fields.max_radial();
        let (fx, fy) = (point.hx * fmax, point.hy * fmax);
        let fo = self.paraxial.primary();
        let at_inf = self.sys.object_at_infinity;
        match osys.fields.kind {
            FieldKind::ObjectAngle => {
                let dir = Vec3::new(fx.to_radians().tan(), fy.to_radians().tan(), 1.0).normalized();
                if at_inf {
                    Ok(ObjectSource::Direction { dir })
                } else {
                    self.object_point_on_chief_line(dir)
                }
            }
            FieldKind::ObjectHeight => self.object_point(fx, fy),
            FieldKind::ParaxialImageHeight => self.paraxial_image_source(fo, fx, fy),
            FieldKind::RealImageHeight => {
                let guess = self.paraxial_image_source(fo, fx, fy)?;
                self.solve_real_image_height(guess, [fx, fy])
            }
        }
    }

    fn object_point(&self, x: f64, y: f64) -> MkResult<ObjectSource> {
        let shape = &self.sys.surfaces[0].shape;
        let z = shape.sag(x * x + y * y).ok_or_else(|| {
            MkError::Trace(format!(
                "object point ({x}, {y}) outside the object surface"
            ))
        })?;
        Ok(ObjectSource::Point {
            point: Vec3::new(x, y, z),
        })
    }

    fn object_point_on_chief_line(&self, dir_global: Vec3) -> MkResult<ObjectSource> {
        if self.pupil.telecentric {
            return Err(MkError::Trace(
                "angle fields need a finite entrance pupil".into(),
            ));
        }
        let f0 = start_frame(&self.sys);
        let p = f0.to_local_point(Vec3::new(0.0, 0.0, self.pupil.z_ep));
        let d = f0.to_local_dir(dir_global);
        let h = intersect(&self.sys.surfaces[0].shape, p, d, &self.settings.intersect)
            .map_err(|e| MkError::Trace(format!("chief ray does not meet the object: {e:?}")))?;
        Ok(ObjectSource::Point { point: h.point })
    }

    fn paraxial_image_source(&self, fo: &FirstOrder, fx: f64, fy: f64) -> MkResult<ObjectSource> {
        let err = || MkError::Trace("paraxial chief ray does not reach the image".into());
        if self.sys.object_at_infinity {
            let h = fo
                .image_height_per_tan
                .filter(|h| *h != 0.0)
                .ok_or_else(err)?;
            Ok(ObjectSource::Direction {
                dir: Vec3::new(fx / h, fy / h, 1.0).normalized(),
            })
        } else {
            let h = fo
                .image_height_per_object_height
                .filter(|h| *h != 0.0)
                .ok_or_else(err)?;
            self.object_point(fx / h, fy / h)
        }
    }

    /// The field value `(x, y)` in units of `kind` that resolves to `src`:
    /// the inverse of [`Self::field_source`] for every kind, so a field
    /// converted to another kind keeps its chief ray. The image heights are
    /// those of the paraxial chief ray and of the real chief ray (primary
    /// wavelength, current aiming). An object height needs a finite object,
    /// an angle a finite entrance pupil.
    pub fn field_value(&self, src: &ObjectSource, kind: FieldKind) -> MkResult<[f64; 2]> {
        let fail = |m: &str| MkError::Trace(m.into());
        let fo = self.paraxial.primary();
        match (kind, *src) {
            (FieldKind::ObjectAngle, ObjectSource::Direction { dir }) => Ok([
                dir.x.atan2(dir.z).to_degrees(),
                dir.y.atan2(dir.z).to_degrees(),
            ]),
            (FieldKind::ObjectAngle, ObjectSource::Point { point }) => {
                if self.pupil.telecentric {
                    return Err(fail("angle fields need a finite entrance pupil"));
                }
                // The chief line from the object point through the paraxial
                // entrance-pupil centre (`object_point_on_chief_line`).
                let g = start_frame(&self.sys).to_global_point(point);
                let d = Vec3::new(-g.x, -g.y, self.pupil.z_ep - g.z);
                if d.z == 0.0 {
                    return Err(fail("the object lies in the entrance-pupil plane"));
                }
                Ok([
                    (d.x / d.z).atan().to_degrees(),
                    (d.y / d.z).atan().to_degrees(),
                ])
            }
            (FieldKind::ObjectHeight, ObjectSource::Point { point }) => Ok([point.x, point.y]),
            (FieldKind::ObjectHeight, ObjectSource::Direction { .. }) => {
                Err(fail("object height needs a finite object"))
            }
            (FieldKind::ParaxialImageHeight, src) => {
                let p = self.source_params(&src);
                let h = match src {
                    ObjectSource::Direction { .. } => fo.image_height_per_tan,
                    ObjectSource::Point { .. } => fo.image_height_per_object_height,
                }
                .filter(|h| h.is_finite())
                .ok_or_else(|| fail("paraxial chief ray does not reach the image"))?;
                Ok([p[0] * h, p[1] * h])
            }
            (FieldKind::RealImageHeight, src) => self
                .chief_image_eval(&src, false, None)
                .map(|(h, _)| h)
                .ok_or_else(|| fail("the real chief ray does not reach the image")),
        }
    }

    fn source_params(&self, src: &ObjectSource) -> [f64; 2] {
        match *src {
            ObjectSource::Direction { dir } => [dir.x / dir.z, dir.y / dir.z],
            ObjectSource::Point { point } => [point.x, point.y],
        }
    }

    fn source_from_params(&self, p: [f64; 2]) -> Option<ObjectSource> {
        if self.sys.object_at_infinity {
            Some(ObjectSource::Direction {
                dir: Vec3::new(p[0], p[1], 1.0).normalized(),
            })
        } else {
            self.object_point(p[0], p[1]).ok()
        }
    }

    /// The tangent of the starting ray of `src` with respect to the source
    /// parameters `p` ([`Self::source_params`]) at a fixed aim `a`: the
    /// field direction `(p_x, p_y, 1)/|·|` at infinity, else the object
    /// point `(p_x, p_y, sag)`.
    fn source_tangent(&self, src: &ObjectSource, a: [f64; 2], ray: &Ray) -> Option<RayTangent> {
        match *src {
            ObjectSource::Direction { dir } => {
                let g = Vec3::new(dir.x / dir.z, dir.y / dir.z, 1.0);
                let gn = g.norm();
                let ap = Vec3::new(a[0], a[1], self.pupil.z_ep);
                let ddir = [Vec3::X, Vec3::Y].map(|e| (e - dir * dir.dot(e)) / gn);
                // `pos = a' − d (a'·d)`.
                let dpos = ddir.map(|dd| -(dd * ap.dot(dir) + dir * ap.dot(dd)));
                Some(RayTangent { dpos, ddir })
            }
            ObjectSource::Point { point } => {
                let r2 = point.x.mul_add(point.x, point.y * point.y);
                let (_, slope) = crate::surface::sag_and_slope(&self.sys.surfaces[0].shape, r2)?;
                if !slope.is_finite() {
                    return None;
                }
                let dpoint = [
                    Vec3::new(1.0, 0.0, 2.0 * point.x * slope),
                    Vec3::new(0.0, 1.0, 2.0 * point.y * slope),
                ];
                let ddir = if self.pupil.telecentric {
                    [Vec3::ZERO; 2]
                } else {
                    // `d = ±u/|u|`, `u = a_local − point`.
                    let f0 = start_frame(&self.sys);
                    let al = f0.to_local_point(Vec3::new(a[0], a[1], self.pupil.z_ep));
                    let u = al - point;
                    let un = u.norm();
                    let sign = if ray.dir.dot(u) < 0.0 { -1.0 } else { 1.0 };
                    dpoint.map(|dp| {
                        let du = -dp;
                        (du - u * (u.dot(du) / (un * un))) * (sign / un)
                    })
                };
                Some(RayTangent { dpos: dpoint, ddir })
            }
        }
    }

    /// Image hit of the chief ray of `src` and, when `jac`, its Jacobian
    /// with respect to the source parameters through the aim solve: `d(img)/dp = ∂img/∂p − ∂img/∂a J_stop⁻¹ ∂stop/∂p`
    /// (implicit function theorem on `stop(a, p) = 0`), from two tangent
    /// passes over one recorded trace. The Jacobian is `None` where a
    /// tangent cannot be formed or `J_stop` is singular.
    fn chief_image_eval(
        &self,
        src: &ObjectSource,
        jac: bool,
        seed: Option<&AimData>,
    ) -> Option<([f64; 2], Option<Mat2>)> {
        let wl = self.sys.primary_wavelength;
        // Only the aim is used: its own Jacobian trace is not needed.
        let aim = self.chief_aim_with(src, wl, false, seed);
        if !aim.converged {
            return None;
        }
        let (ray, ta) = self.start_tangent(src, aim.chief)?;
        let want = jac;
        let mut rec = Vec::with_capacity(if want { self.sys.surfaces.len() } else { 0 });
        let r = trace_ray(
            &self.sys,
            &ray,
            wl,
            &TraceOptions::unclipped(),
            want.then_some(&mut rec),
        );
        if !r.status.is_ok() {
            return None;
        }
        let hit = [r.ray.pos.x, r.ray.pos.y];
        if !want {
            return Some((hit, None));
        }
        let jac = (|| {
            let ts = self.source_tangent(src, aim.chief, &ray)?;
            let plan = &self.tangents;
            let image = r.surface;
            let js_img = plan
                .differentiate(&self.sys, wl, &ts, &rec, image)
                .ok()?
                .xy_jacobian();
            if matches!(self.aiming, RayAiming::Off) {
                return Some(js_img);
            }
            let stop = self.sys.stop;
            let ja_stop = plan
                .differentiate(&self.sys, wl, &ta, &rec, stop)
                .ok()?
                .xy_jacobian();
            let ja_img = plan
                .differentiate(&self.sys, wl, &ta, &rec, image)
                .ok()?
                .xy_jacobian();
            let js_stop = plan
                .differentiate(&self.sys, wl, &ts, &rec, stop)
                .ok()?
                .xy_jacobian();
            // M = J_stop⁻¹ ∂stop/∂p, column by column.
            let m0 = solve2(ja_stop, [js_stop[0][0], js_stop[1][0]])?;
            let m1 = solve2(ja_stop, [js_stop[0][1], js_stop[1][1]])?;
            let m = [[m0[0], m1[0]], [m0[1], m1[1]]];
            Some(std::array::from_fn(|row| {
                std::array::from_fn(|col| {
                    js_img[row][col] - (ja_img[row][0] * m[0][col] + ja_img[row][1] * m[1][col])
                })
            }))
        })();
        Some((hit, jac))
    }

    fn solve_real_image_height(
        &self,
        guess: ObjectSource,
        target: [f64; 2],
    ) -> MkResult<ObjectSource> {
        if target == [0.0, 0.0] {
            return Ok(guess);
        }
        self.solve_real_image_height_from(self.source_params(&guess), target, None)
    }

    /// The real-image-height solve from source parameters `x0`; with an
    /// anchor's chief `aim`, `x0` is the anchor's source and the solve is
    /// seeded (`newton2_seeded`).
    fn solve_real_image_height_from(
        &self,
        x0: [f64; 2],
        target: [f64; 2],
        aim: Option<&AimData>,
    ) -> MkResult<ObjectSource> {
        let f = |p: [f64; 2], jac: bool| {
            let src = self.source_from_params(p)?;
            self.chief_image_eval(&src, jac, aim)
                .map(|(h, j)| ([h[0] - target[0], h[1] - target[1]], j))
        };
        let scale = x0[0].hypot(x0[1]).max(1e-6);
        let height = target[0].hypot(target[1]);
        let tol = 1e-12 * height.max(1e-3);
        let seed = aim.map(|_| (1e-4 * tol).max(16.0 * f64::EPSILON * height));
        let s = newton2_seeded(
            &f,
            x0,
            None,
            1e-6 * scale,
            tol,
            60,
            JacobianPolicy::AnalyticNewton,
            seed,
        );
        if !s.converged {
            return Err(MkError::Trace(format!(
                "real image height ({}, {}) not reached by any chief ray (residual {:.3e} mm)",
                target[0], target[1], s.residual
            )));
        }
        self.source_from_params(s.x)
            .ok_or_else(|| MkError::Trace("real image height solve left the object".into()))
    }

    // ----- Public ray generation and tracing -------------------------------------

    /// Generate the starting ray of pupil point `(px, py)` for a prepared
    /// field at wavelength `wl` (vignetting factors applied here).
    pub fn generate(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
    ) -> Result<GeneratedRay, AimReport> {
        match self.aim_start(field, wl, px, py)? {
            AimStart::Ready(g) => Ok(g),
            AimStart::Solve {
                target,
                x0,
                j0,
                seed,
                key,
            } => {
                let tol = self.field_tolerance(field, wl, target);
                let s = self.solve_aim_seeded(&field.source, wl, target, x0, j0, seed, tol);
                let stats = &mut AimStats::default();
                self.finish_seeded(field, wl, target, s, stats, seed.is_some(), key)
            }
        }
    }

    /// [`Self::finish_aim`] of a direct solve; a seeded solve that did not
    /// converge is first repeated unseeded. Records the aim on an anchor.
    #[allow(clippy::too_many_arguments)]
    fn finish_seeded(
        &self,
        field: &PreparedField,
        wl: usize,
        target: [f64; 2],
        mut s: Solve2,
        stats: &mut AimStats,
        seeded: bool,
        key: Option<AimKey>,
    ) -> Result<GeneratedRay, AimReport> {
        if seeded && !s.converged {
            stats.add(&s);
            let x0 = self.linear_aim(field, wl, target);
            let tol = self.field_tolerance(field, wl, target);
            s = self.solve_aim(&field.source, wl, target, x0, field.aims[wl].jacobian, tol);
        }
        let g = self.finish_aim(field, wl, target, s, stats);
        if let (Some(Seeds::Record(seeds)), Some(key), Ok(g)) = (&self.seeds, key, &g) {
            // The stop Jacobian at the solution (the solve's own may be a
            // Broyden estimate): a seeded solve then resolves its small
            // change in one step.
            let j = self.stop_jacobian(&field.source, g.aim, wl);
            seeds.rays.write().unwrap().insert(key, (g.aim, j));
        }
        g
    }

    /// The linear prediction of the aim for stop target `target` from the
    /// chief ray's aim Jacobian (or the paraxial aim).
    fn linear_aim(&self, field: &PreparedField, wl: usize, target: [f64; 2]) -> [f64; 2] {
        let ad = &field.aims[wl];
        ad.jacobian
            .and_then(|j| solve2(j, target))
            .map(|d| [ad.chief[0] + d[0], ad.chief[1] + d[1]])
            .unwrap_or_else(|| {
                let r = self.stop_radius();
                let p = self.paraxial_aim([target[0] / r, target[1] / r]);
                [ad.chief[0] + p[0], ad.chief[1] + p[1]]
            })
    }

    /// Everything of [`Self::generate`] before the real-aim solve.
    fn aim_start(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
    ) -> Result<AimStart, AimReport> {
        // Seeds are keyed by the requested pupil point; the chief ray is
        // the prepared field's own aim and never seeded.
        let key =
            (self.seeds.is_some() && (px, py) != (0.0, 0.0)).then(|| aim_key(field, wl, px, py));
        let (px, py) = field.point.vignetting.apply(px, py);
        let fail = AimReport {
            iterations: 0,
            residual: f64::INFINITY,
            converged: false,
        };
        if !field.is_valid() {
            return Err(fail);
        }
        match self.aiming {
            RayAiming::Off => {
                let a = self.paraxial_aim([px, py]);
                let ray = self.start_ray(&field.source, a).ok_or(fail)?;
                Ok(AimStart::Ready(GeneratedRay {
                    ray,
                    aim: a,
                    report: AimReport {
                        iterations: 0,
                        residual: 0.0,
                        converged: true,
                    },
                }))
            }
            RayAiming::Real { .. } => {
                let ad = &field.aims[wl];
                let r_stop = self.pupil.r_stop;
                let target = [px * r_stop, py * r_stop];
                // Linear prediction from the chief-ray Jacobian.
                let x0 = ad
                    .jacobian
                    .and_then(|j| solve2(j, target))
                    .map(|d| [ad.chief[0] + d[0], ad.chief[1] + d[1]])
                    .unwrap_or_else(|| {
                        let p = self.paraxial_aim([px, py]);
                        [ad.chief[0] + p[0], ad.chief[1] + p[1]]
                    });
                let seed = key
                    .as_ref()
                    .and_then(|k| self.read_seeds().and_then(|s| s.ray(k)));
                Ok(match seed {
                    Some((aim, j)) => AimStart::Solve {
                        target,
                        x0: aim,
                        j0: j.or(ad.jacobian),
                        seed: Some(self.aim_floor()),
                        key,
                    },
                    None => AimStart::Solve {
                        target,
                        x0,
                        j0: ad.jacobian,
                        seed: None,
                        key,
                    },
                })
            }
        }
    }

    /// Everything of [`Self::generate`] after the direct real-aim solve `s`:
    /// continuation in the pupil radius if it did not converge, then the
    /// starting ray. `stats` accumulates the solves' work.
    fn finish_aim(
        &self,
        field: &PreparedField,
        wl: usize,
        target: [f64; 2],
        mut s: Solve2,
        stats: &mut AimStats,
    ) -> Result<GeneratedRay, AimReport> {
        let ad = &field.aims[wl];
        let mut iters = s.iterations;
        stats.rays += 1;
        stats.add(&s);
        if !s.converged {
            // Continuation in the pupil radius.
            stats.continued += 1;
            let tol = self.field_tolerance(field, wl, target);
            let mut x = ad.chief;
            let mut j = ad.jacobian;
            let mut prev = [0.0, 0.0];
            for frac in [0.25, 0.5, 0.75, 1.0] {
                let t = [target[0] * frac, target[1] * frac];
                let seed = j
                    .and_then(|j| solve2(j, [t[0] - prev[0], t[1] - prev[1]]))
                    .map_or(x, |d| [x[0] + d[0], x[1] + d[1]]);
                s = self.solve_aim(&field.source, wl, t, seed, j, tol);
                iters += s.iterations;
                stats.add(&s);
                if !s.converged {
                    break;
                }
                x = s.x;
                j = s.jacobian;
                prev = t;
            }
        }
        let report = AimReport {
            iterations: iters,
            residual: s.residual / self.stop_radius(),
            converged: s.converged,
        };
        if !s.converged {
            stats.failed += 1;
            return Err(report);
        }
        let ray = self.start_ray(&field.source, s.x).ok_or(report)?;
        Ok(GeneratedRay {
            ray,
            aim: s.x,
            report,
        })
    }

    /// [`Self::generate`] for many pupil points at once, bitwise identical
    /// to generating them one by one. The direct real-aim solves of all rays
    /// advance together ([`Newton2`]) and each round's stop traces run as
    /// packets ([`trace_rays`], or [`trace_rays_recorded`] plus
    /// [`TangentPlan::differentiate`] where the Jacobian is requested); the
    /// rare rays that need continuation in the pupil radius finish on their
    /// own.
    pub fn generate_many(&self, rays: &[RayRequest]) -> Vec<Result<GeneratedRay, AimReport>> {
        self.generate_many_stats(rays, &mut AimStats::default())
    }

    /// [`Self::generate_many`], accumulating the solves' work into `stats`.
    pub fn generate_many_stats(
        &self,
        rays: &[RayRequest],
        stats: &mut AimStats,
    ) -> Vec<Result<GeneratedRay, AimReport>> {
        self.generate_batch(rays, None, stats)
    }

    /// [`Self::generate_many_stats`] with caller start points, for hybrid
    /// aiming (a host seed table's interpolated aim and stop Jacobian, for
    /// example): the direct real-aim solve of ray `k` starts from
    /// `starts[k] = (aim, stop Jacobian)` when it is `Some`, instead of the
    /// chief ray's linear prediction. A started solve that does not converge
    /// is repeated by the reference schedule ([`Self::generate`]: the chief
    /// prediction, then continuation in the pupil radius), so a ray fails
    /// only where the reference fails too. Started rays meet the same stop
    /// tolerance as the reference's and differ from them only within it;
    /// rays without a start (and seeded or non-real aims) are bitwise those
    /// of [`Self::generate_many`]. The stop traces of all solves run as
    /// packets, as there.
    pub fn generate_many_from(
        &self,
        rays: &[RayRequest],
        starts: &[Option<([f64; 2], crate::numeric::Mat2)>],
        stats: &mut AimStats,
    ) -> Vec<Result<GeneratedRay, AimReport>> {
        assert_eq!(rays.len(), starts.len(), "one start per ray");
        self.generate_batch(rays, Some(starts), stats)
    }

    /// The batched generator of [`Self::generate_many_stats`] (no `starts`)
    /// and [`Self::generate_many_from`].
    fn generate_batch(
        &self,
        rays: &[RayRequest],
        starts: Option<&[Option<AimStartPoint>]>,
        stats: &mut AimStats,
    ) -> Vec<Result<GeneratedRay, AimReport>> {
        let mut out = Vec::with_capacity(rays.len());
        // Direct solves: (ray index, stop target, solver, seeded, key, started).
        let mut solves = Vec::new();
        for (k, &(field, wl, [px, py])) in rays.iter().enumerate() {
            match self.aim_start(field, wl, px, py) {
                Ok(AimStart::Solve {
                    target,
                    x0,
                    j0,
                    seed,
                    key,
                }) => {
                    let (x0, j0, started) = match starts.and_then(|s| s[k]) {
                        Some((x, j)) if seed.is_none() => (x, Some(j), true),
                        _ => (x0, j0, false),
                    };
                    let n = Newton2::new(
                        x0,
                        j0,
                        self.fd_step(),
                        self.field_tolerance(field, wl, target),
                        self.settings.max_aim_iterations,
                        AIM_POLICY,
                    )
                    .seeded(seed);
                    solves.push((k, target, n, seed.is_some(), key, started));
                    // Placeholder until the solve finishes.
                    out.push(Err(AimReport {
                        iterations: 0,
                        residual: f64::INFINITY,
                        converged: false,
                    }));
                }
                Ok(AimStart::Ready(g)) => out.push(Ok(g)),
                Err(e) => out.push(Err(e)),
            }
        }
        let opts = self.stop_options();
        // Plain traces, and Jacobian traces with their start tangents.
        let (mut plain, mut wls) = (Vec::new(), Vec::new());
        let (mut jstarts, mut jwls, mut jtangents) = (Vec::new(), Vec::new(), Vec::new());
        let mut records: Vec<Vec<SurfaceRecord>> = Vec::new();
        // Per pending point: no starting ray (`f` undefined there, as in
        // `stop_eval`), a plain trace, or a Jacobian trace.
        let mut kind: Vec<u8> = Vec::new();
        let mut values: Vec<Eval> = Vec::new();
        loop {
            plain.clear();
            wls.clear();
            jstarts.clear();
            jwls.clear();
            jtangents.clear();
            kind.clear();
            for (k, _, n, ..) in &solves {
                let (field, wl, _) = rays[*k];
                let want = n.wants_jacobian();
                for &a in n.pending() {
                    if want {
                        match self.start_tangent(&field.source, a) {
                            Some((ray, t)) => {
                                jstarts.push(ray);
                                jwls.push(wl);
                                jtangents.push(t);
                                kind.push(2);
                            }
                            None => kind.push(0),
                        }
                    } else {
                        match self.start_ray(&field.source, a) {
                            Some(ray) => {
                                plain.push(ray);
                                wls.push(wl);
                                kind.push(1);
                            }
                            None => kind.push(0),
                        }
                    }
                }
            }
            if kind.is_empty() {
                break;
            }
            let hits = trace_rays(&self.sys, &plain, &wls, &opts);
            if records.len() < jstarts.len() {
                records.resize_with(jstarts.len(), || Vec::with_capacity(self.sys.stop + 1));
            }
            let jhits = trace_rays_recorded(
                &self.sys,
                &jstarts,
                &jwls,
                &opts,
                &mut records[..jstarts.len()],
            );
            let mut hit = hits.iter();
            let mut jhit = jhits.iter().zip(&records).zip(&jtangents);
            let mut point = kind.iter();
            for (k, target, n, ..) in &mut solves {
                let wl = rays[*k].1;
                values.clear();
                for _ in 0..n.pending().len() {
                    let residual =
                        |r: &TraceResult| [r.ray.pos.x - target[0], r.ray.pos.y - target[1]];
                    values.push(match *point.next().expect("one kind per pending point") {
                        0 => None,
                        1 => {
                            let r = hit.next().expect("one trace per starting ray");
                            r.status.is_ok().then(|| (residual(r), None))
                        }
                        _ => {
                            let ((r, rec), t) = jhit.next().expect("one trace per Jacobian ray");
                            r.status.is_ok().then(|| {
                                let j = self
                                    .tangents
                                    .differentiate(&self.sys, wl, t, rec, r.surface)
                                    .ok()
                                    .map(|t| t.xy_jacobian());
                                (residual(r), j)
                            })
                        }
                    });
                }
                if !values.is_empty() {
                    n.resume(&values);
                }
            }
        }
        for (k, target, n, seeded, key, started) in solves {
            let (field, wl, [px, py]) = rays[k];
            let s = n.result().expect("a finished solve has a result");
            out[k] = if started && !s.converged {
                // The reference schedule from the chief prediction.
                stats.add(&s);
                match self.aim_start(field, wl, px, py) {
                    Ok(AimStart::Solve {
                        target,
                        x0,
                        j0,
                        seed,
                        key,
                    }) => {
                        let tol = self.field_tolerance(field, wl, target);
                        let s = self.solve_aim_seeded(&field.source, wl, target, x0, j0, seed, tol);
                        self.finish_seeded(field, wl, target, s, stats, seed.is_some(), key)
                    }
                    Ok(AimStart::Ready(g)) => Ok(g),
                    Err(e) => Err(e),
                }
            } else {
                self.finish_seeded(field, wl, target, s, stats, seeded, key)
            };
        }
        out
    }

    /// Generate and trace pupil point `(px, py)`. A ray that cannot be aimed
    /// is reported as `Failed { surface: stop, reason: AimNotConverged }`,
    /// and every ray of an unresolved field as
    /// `Failed { surface: 0, reason: FieldUnreachable }`.
    pub fn trace(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
    ) -> TraceResult {
        self.trace_with(field, wl, px, py, opts, None)
    }

    /// As [`Self::trace`], filling `record` with per-surface data. A ray that
    /// cannot be generated leaves `record` empty.
    pub fn trace_with(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
        record: Option<&mut Vec<SurfaceRecord>>,
    ) -> TraceResult {
        match self.generate(field, wl, px, py) {
            Ok(g) => trace_ray(&self.sys, &g.ray, wl, opts, record),
            Err(_) => {
                if let Some(r) = record {
                    r.clear();
                }
                self.ungenerated(field)
            }
        }
    }

    /// [`Self::trace`] for many pupil points at once, bitwise identical to
    /// tracing them one by one: aims via [`Self::generate_many`], image
    /// traces via [`trace_rays`].
    pub fn trace_many(&self, rays: &[RayRequest], opts: &TraceOptions) -> Vec<TraceResult> {
        let generated = self.generate_many(rays);
        let (starts, wls): (Vec<Ray>, Vec<usize>) = generated
            .iter()
            .zip(rays)
            .filter_map(|(g, r)| g.as_ref().ok().map(|g| (g.ray, r.1)))
            .unzip();
        let mut traced = trace_rays(&self.sys, &starts, &wls, opts).into_iter();
        generated
            .iter()
            .zip(rays)
            .map(|(g, &(field, _, _))| match g {
                Ok(_) => traced.next().expect("one trace per generated ray"),
                Err(_) => self.ungenerated(field),
            })
            .collect()
    }

    /// The result of a ray that could not be generated (see [`Self::trace`]).
    fn ungenerated(&self, field: &PreparedField) -> TraceResult {
        let (surface, reason) = if field.is_valid() {
            (self.sys.stop, FailReason::AimNotConverged)
        } else {
            (0, FailReason::FieldUnreachable)
        };
        TraceResult {
            status: TraceStatus::Failed { surface, reason },
            ray: Ray::new(Vec3::ZERO, Vec3::Z),
            surface,
            opl: Default::default(),
            transmittance: 0.0,
        }
    }

    /// Trace the chief ray `(0, 0)` of a field (unclipped).
    pub fn chief(&self, field: &PreparedField, wl: usize) -> TraceResult {
        self.trace(field, wl, 0.0, 0.0, &TraceOptions::unclipped())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::{
        ApertureSpec, Field, FieldSet, MaterialRef, Snapshot, Surface, SurfaceShape, samples,
    };

    fn lens(aiming: RayAiming) -> SequentialTracer {
        let mut s = OpticalSystem::new_empty();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
        s.fields = FieldSet {
            kind: FieldKind::ObjectAngle,
            fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 10.0)],
        };
        s.ray_aiming = aiming;
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(1, SurfaceShape::Plane, 10.0, MaterialRef::Air),
            Surface::new(
                2,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            Surface::new(3, SurfaceShape::sphere(-50.0), 46.0, MaterialRef::Air),
            Surface::new(4, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        let snap = Snapshot::new(s).unwrap();
        let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        SequentialTracer::new(Arc::new(r)).unwrap()
    }

    #[test]
    fn aim_off_hits_stop_exactly_when_stop_is_first() {
        // The stop is in object space, so the paraxial EP is the stop itself.
        let t = lens(RayAiming::Off);
        let f = t.field(1).clone();
        let g = t.generate(&f, 1, 0.3, -0.8).unwrap();
        let r = trace_ray(
            t.system(),
            &g.ray,
            1,
            &TraceOptions::unclipped().to_surface(1),
            None,
        );
        assert!((r.ray.pos.x - 1.5).abs() < 1e-12 && (r.ray.pos.y + 4.0).abs() < 1e-12);
    }

    #[test]
    fn real_aim_reaches_target() {
        let t = lens(RayAiming::Real { tolerance: 1e-12 });
        let f = t.field(1).clone();
        let g = t.generate(&f, 0, -0.6, 0.7).unwrap();
        assert!(g.report.converged);
        let r = trace_ray(
            t.system(),
            &g.ray,
            0,
            &TraceOptions::unclipped().to_surface(1),
            None,
        );
        assert!((r.ray.pos.x + 3.0).abs() < 1e-10 && (r.ray.pos.y - 3.5).abs() < 1e-10);
    }

    /// Central differences of `f` over a step sweep: the smallest error
    /// against `j` and the error at `h = 10⁻⁴·scale`.
    fn sweep(
        f: &dyn Fn([f64; 2]) -> Option<[f64; 2]>,
        x: [f64; 2],
        j: Mat2,
        scale: f64,
    ) -> (f64, f64) {
        let (mut best, mut at4) = (f64::INFINITY, f64::INFINITY);
        for e in 2..=9 {
            let h = 10f64.powi(-e) * scale;
            let mut err = 0.0f64;
            for k in 0..2 {
                let (mut xp, mut xm) = (x, x);
                xp[k] += h;
                xm[k] -= h;
                let (a, b) = (f(xp).unwrap(), f(xm).unwrap());
                for r in 0..2 {
                    err = err.max(((a[r] - b[r]) / (2.0 * h) - j[r][k]).abs());
                }
            }
            best = best.min(err);
            if e == 4 {
                at4 = err;
            }
        }
        (best, at4)
    }

    #[test]
    fn stop_jacobian_matches_step_sweep() {
        let t = lens(RayAiming::Real { tolerance: 1e-12 });
        let f = t.field(1).clone();
        let a = [0.7, -1.9];
        let (_, Some(j)) = t
            .stop_eval(&f.source, a, 0, [0.0; 2], true, &mut Vec::new())
            .unwrap()
        else {
            panic!("analytic Jacobian expected");
        };
        let g = |x: [f64; 2]| {
            t.stop_eval(&f.source, x, 0, [0.0; 2], false, &mut Vec::new())
                .map(|(v, _)| v)
        };
        let (best, at4) = sweep(&g, a, j, 1.0);
        assert!(best < 1e-8 && at4 < 1e-6, "best {best:e} at 1e-4 {at4:e}");
    }

    #[test]
    fn real_image_height_jacobian_matches_step_sweep() {
        for sys in [samples::cooke_triplet(), samples::finite_relay()] {
            let mut sys = sys;
            sys.ray_aiming = RayAiming::Real { tolerance: 1e-12 };
            let snap = Snapshot::new(sys).unwrap();
            let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
            let t = SequentialTracer::new(Arc::new(r)).unwrap();
            let src = t.field(t.fields().len() - 1).source;
            let p = t.source_params(&src);
            let (_, Some(j)) = t.chief_image_eval(&src, true, None).unwrap() else {
                panic!("analytic Jacobian expected");
            };
            let g = |x: [f64; 2]| {
                t.chief_image_eval(&t.source_from_params(x)?, false, None)
                    .map(|(h, _)| h)
            };
            let scale = p[0].hypot(p[1]).max(1e-3);
            let (best, at4) = sweep(&g, p, j, scale);
            let jn = j.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                best < 1e-7 * jn && at4 < 1e-5 * jn,
                "best {best:e} at 1e-4 {at4:e} (|J| {jn:e})"
            );
        }
    }
}
