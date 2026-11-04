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
//! aiming**, `a` solves `stop_hit(a) = P'·r_stop` in the stop's local frame
//! (`r_stop` is the paraxial stop semi-diameter from the aperture
//! specification), by Newton–Broyden iteration ([`crate::numeric`]) seeded by a
//! linear prediction from the cached chief-ray aim point and Jacobian of the
//! (field, wavelength). If the direct solve fails, it falls back to
//! continuation in the pupil radius (25 %, 50 %, 75 %, 100 %). Aiming ignores
//! clipping. At infinity, chief-ray recovery blends unit directions from the
//! axis and keeps the entrance hit between steps. Convergence is reported per
//! ray.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use mklight_core::{
    FieldKind, MkError, MkResult, OpticalSystem, RayAiming, ResolvedSystem, Vec3, Vignetting,
};
use serde::{Deserialize, Serialize};

use crate::numeric::{Mat2, Newton2, fd_jacobian, newton2_seeded, solve2};
use crate::paraxial::{FirstOrder, Paraxial};
use crate::surface::{IntersectSettings, intersect};
use crate::trace::{
    FailReason, Ray, SurfaceRecord, TraceOptions, TraceResult, TraceStatus, start_frame, trace_ray,
    trace_rays,
};

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
/// anchor's Jacobian (central differences, recorded with the aim) and takes
/// at least one Newton step, so it corrects for the change even when the
/// seed already meets the tolerance; it must then resolve the change to
/// 1e-6 of itself down to rounding, never looser than the aiming tolerance
/// (`newton2_seeded`), and falls back to the unseeded solve otherwise.
/// Results agree with the unseeded tracer to within the aiming tolerance
/// (not bit for bit) and do not depend on the order of tracing. Recording
/// costs a five-trace Jacobian per anchor ray.
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
}

impl Default for TracerSettings {
    fn default() -> Self {
        TracerSettings {
            aiming: None,
            max_aim_iterations: 40,
            intersect: IntersectSettings::default(),
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
    /// Paraxial stop semi-diameter from the aperture specification.
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
        let mut t = SequentialTracer {
            sys,
            paraxial,
            pupil,
            aiming,
            settings,
            fields: Vec::new(),
            warnings,
            seeds,
        };
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
    /// Append analysis-local fields without changing the physical snapshot.
    /// Existing field indices keep their meaning; new ones follow them.
    pub fn with_additional_fields(&self, points: &[FieldPoint]) -> MkResult<Self> {
        let mut tracer = self.clone();
        for &point in points {
            tracer.fields.push(self.prepare_field(point)?);
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

    /// Stop-surface hit `(x, y)` (stop local frame) of the ray with aim `a`.
    fn stop_hit(&self, src: &ObjectSource, a: [f64; 2], wl: usize) -> Option<[f64; 2]> {
        let ray = self.start_ray(src, a)?;
        let opts = TraceOptions {
            clip: false,
            last_surface: Some(self.sys.stop),
            intersect: self.settings.intersect,
        };
        let r = trace_ray(&self.sys, &ray, wl, &opts, None);
        r.status.is_ok().then_some([r.ray.pos.x, r.ray.pos.y])
    }

    fn aim_tolerance(&self) -> f64 {
        let rel = match self.aiming {
            RayAiming::Real { tolerance } => tolerance,
            RayAiming::Off => 1e-10,
        };
        rel * self.pupil.r_stop.max(f64::MIN_POSITIVE)
    }

    /// Rounding floor of a seeded aim solve (stop coordinates).
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
    ) -> (crate::numeric::Solve2, u32) {
        self.solve_aim_seeded(src, wl, target, x0, j0, None)
    }

    /// [`Self::solve_aim`], seeded as `newton2_seeded` when `seed` is set.
    fn solve_aim_seeded(
        &self,
        src: &ObjectSource,
        wl: usize,
        target: [f64; 2],
        x0: [f64; 2],
        j0: Option<Mat2>,
        seed: Option<f64>,
    ) -> (crate::numeric::Solve2, u32) {
        let f = |a: [f64; 2]| {
            self.stop_hit(src, a, wl)
                .map(|s| [s[0] - target[0], s[1] - target[1]])
        };
        let s = newton2_seeded(
            &f,
            x0,
            j0,
            self.fd_step(),
            self.aim_tolerance(),
            self.settings.max_aim_iterations,
            seed,
        );
        let it = s.iterations;
        (s, it)
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

    /// [`Self::chief_aim`]; without the final aim Jacobian (four stop
    /// traces) when `jacobian` is false, for callers that only use the aim.
    /// A converged `seed` (the anchor's aim, [`AimSeeds`]) starts the solve.
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
            },
            RayAiming::Real { .. } => {
                let seeded = seed
                    .filter(|a| a.converged)
                    .map(|a| {
                        self.solve_aim_seeded(
                            src,
                            wl,
                            [0.0, 0.0],
                            a.chief,
                            a.jacobian,
                            Some(self.aim_floor()),
                        )
                        .0
                    })
                    .filter(|s| s.converged);
                let mut s = match seeded {
                    Some(s) => s,
                    None => self.solve_aim(src, wl, [0.0, 0.0], [0.0, 0.0], None).0,
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
                        let (sk, _) = self.solve_aim(&step, wl, [0.0, 0.0], x, j);
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
                let f = |a: [f64; 2]| self.stop_hit(src, a, wl);
                let jac = if jacobian {
                    f(s.x)
                        .and_then(|fx| fd_jacobian(&f, s.x, fx, self.fd_step()))
                        .or(s.jacobian)
                } else {
                    s.jacobian
                };
                AimData {
                    chief: s.x,
                    jacobian: jac,
                    converged: s.converged,
                    residual: s.residual / self.pupil.r_stop,
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
        let field = PreparedField {
            point,
            source,
            aims,
            error: None,
        };
        if let Some(Seeds::Record(s)) = &self.seeds {
            let mut fields = s.fields.write().unwrap();
            if !fields.iter().any(|f| f.point == point) {
                fields.push(field.clone());
            }
        }
        Ok(field)
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

    /// Image-surface hit of the chief ray of `src` (primary wavelength).
    fn chief_image_hit(&self, src: &ObjectSource, seed: Option<&AimData>) -> Option<[f64; 2]> {
        let wl = self.sys.primary_wavelength;
        let aim = self.chief_aim_with(src, wl, false, seed);
        if !aim.converged {
            return None;
        }
        let ray = self.start_ray(src, aim.chief)?;
        let r = trace_ray(&self.sys, &ray, wl, &TraceOptions::unclipped(), None);
        r.status.is_ok().then_some([r.ray.pos.x, r.ray.pos.y])
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
        let f = |p: [f64; 2]| {
            let src = self.source_from_params(p)?;
            self.chief_image_hit(&src, aim)
                .map(|h| [h[0] - target[0], h[1] - target[1]])
        };
        let scale = x0[0].hypot(x0[1]).max(1e-6);
        let height = target[0].hypot(target[1]);
        let tol = 1e-12 * height.max(1e-3);
        let seed = aim.map(|_| (1e-4 * tol).max(16.0 * f64::EPSILON * height));
        let s = newton2_seeded(&f, x0, None, 1e-6 * scale, tol, 60, seed);
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
                let (s, iters) = self.solve_aim_seeded(&field.source, wl, target, x0, j0, seed);
                self.finish_seeded(field, wl, target, s, iters, seed.is_some(), key)
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
        mut s: crate::numeric::Solve2,
        mut iters: u32,
        seeded: bool,
        key: Option<AimKey>,
    ) -> Result<GeneratedRay, AimReport> {
        if seeded && !s.converged {
            let x0 = self.linear_aim(field, wl, target);
            let (cold, it) = self.solve_aim(&field.source, wl, target, x0, field.aims[wl].jacobian);
            (s, iters) = (cold, iters + it);
        }
        let (g, jacobian) = self.finish_aim(field, wl, target, s, iters);
        if let (Some(Seeds::Record(seeds)), Some(key), Ok(g)) = (&self.seeds, key, &g) {
            // A central-difference Jacobian at the solution (the solve's own
            // is a Broyden estimate): a seeded solve then resolves its small
            // change in one step.
            let f = |a: [f64; 2]| self.stop_hit(&field.source, a, wl);
            let exact = f(g.aim).and_then(|fx| fd_jacobian(&f, g.aim, fx, self.fd_step()));
            seeds
                .rays
                .write()
                .unwrap()
                .insert(key, (g.aim, exact.or(jacobian)));
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
                let r = self.pupil.r_stop;
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

    /// Everything of [`Self::generate`] after the direct real-aim solve `s`
    /// (`iters` Newton iterations): continuation in the pupil radius if it
    /// did not converge, then the starting ray, and the final aim Jacobian.
    fn finish_aim(
        &self,
        field: &PreparedField,
        wl: usize,
        target: [f64; 2],
        mut s: crate::numeric::Solve2,
        mut iters: u32,
    ) -> (Result<GeneratedRay, AimReport>, Option<Mat2>) {
        let ad = &field.aims[wl];
        let r_stop = self.pupil.r_stop;
        if !s.converged {
            // Continuation in the pupil radius.
            let mut x = ad.chief;
            let mut j = ad.jacobian;
            let mut prev = [0.0, 0.0];
            for frac in [0.25, 0.5, 0.75, 1.0] {
                let t = [target[0] * frac, target[1] * frac];
                let seed = j
                    .and_then(|j| solve2(j, [t[0] - prev[0], t[1] - prev[1]]))
                    .map_or(x, |d| [x[0] + d[0], x[1] + d[1]]);
                let (sn, it) = self.solve_aim(&field.source, wl, t, seed, j);
                iters += it;
                s = sn;
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
            residual: s.residual / r_stop,
            converged: s.converged,
        };
        if !s.converged {
            return (Err(report), s.jacobian);
        }
        let g = self
            .start_ray(&field.source, s.x)
            .ok_or(report)
            .map(|ray| GeneratedRay {
                ray,
                aim: s.x,
                report,
            });
        (g, s.jacobian)
    }

    /// [`Self::generate`] for many pupil points at once, bitwise identical
    /// to generating them one by one. The direct real-aim solves of all rays
    /// advance together ([`Newton2`]) and each round's stop traces run as
    /// packets ([`trace_rays`]); the rare rays that need continuation in the
    /// pupil radius finish on their own.
    pub fn generate_many(&self, rays: &[RayRequest]) -> Vec<Result<GeneratedRay, AimReport>> {
        let mut out = Vec::with_capacity(rays.len());
        // Direct solves: (ray index, stop target, solver).
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
                    let n = Newton2::new(
                        x0,
                        j0,
                        self.fd_step(),
                        self.aim_tolerance(),
                        self.settings.max_aim_iterations,
                    )
                    .seeded(seed);
                    solves.push((k, target, n, seed.is_some(), key));
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
        let opts = TraceOptions {
            clip: false,
            last_surface: Some(self.sys.stop),
            intersect: self.settings.intersect,
        };
        let (mut starts, mut wls) = (Vec::new(), Vec::new());
        // Per pending point: whether it has a starting ray (else `f` is
        // undefined there, as in `stop_hit`).
        let mut has_ray = Vec::new();
        let mut values = Vec::new();
        loop {
            starts.clear();
            wls.clear();
            has_ray.clear();
            for (k, _, n, ..) in &solves {
                let (field, wl, _) = rays[*k];
                for &a in n.pending() {
                    let ray = self.start_ray(&field.source, a);
                    has_ray.push(ray.is_some());
                    if let Some(ray) = ray {
                        starts.push(ray);
                        wls.push(wl);
                    }
                }
            }
            if has_ray.is_empty() {
                break;
            }
            let hits = trace_rays(&self.sys, &starts, &wls, &opts);
            let (mut hit, mut point) = (hits.iter(), has_ray.iter());
            for (_, target, n, ..) in &mut solves {
                values.clear();
                for _ in 0..n.pending().len() {
                    let traced = *point.next().expect("one flag per pending point");
                    values.push(if traced {
                        let r = hit.next().expect("one trace per starting ray");
                        r.status
                            .is_ok()
                            .then(|| [r.ray.pos.x - target[0], r.ray.pos.y - target[1]])
                    } else {
                        None
                    });
                }
                if !values.is_empty() {
                    n.resume(&values);
                }
            }
        }
        for (k, target, n, seeded, key) in solves {
            let (field, wl, _) = rays[k];
            let s = n.result().expect("a finished solve has a result");
            out[k] = self.finish_seeded(field, wl, target, s, s.iterations, seeded, key);
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
        ApertureSpec, Field, FieldSet, MaterialRef, Snapshot, Surface, SurfaceShape,
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
}
