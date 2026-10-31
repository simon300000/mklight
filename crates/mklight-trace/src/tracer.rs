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
//! clipping. Convergence is reported per ray.

use std::sync::Arc;

use mklight_core::{
    FieldKind, MkError, MkResult, OpticalSystem, RayAiming, ResolvedSystem, Vec3, Vignetting,
};
use serde::{Deserialize, Serialize};

use crate::numeric::{Mat2, fd_jacobian, newton2, solve2};
use crate::paraxial::{FirstOrder, Paraxial};
use crate::surface::{IntersectSettings, intersect};
use crate::trace::{
    FailReason, Ray, SurfaceRecord, TraceOptions, TraceResult, TraceStatus, start_frame, trace_ray,
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
}

impl SequentialTracer {
    pub fn new(sys: Arc<ResolvedSystem>) -> MkResult<Self> {
        Self::with_settings(sys, TracerSettings::default())
    }

    pub fn with_settings(sys: Arc<ResolvedSystem>, settings: TracerSettings) -> MkResult<Self> {
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
        };
        let n = t.sys.system().fields.fields.len();
        let mut fields = Vec::with_capacity(n);
        for i in 0..n {
            let point = FieldPoint::system(t.sys.system(), i);
            let f = match t.prepare_field(point) {
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
        let f = |a: [f64; 2]| {
            self.stop_hit(src, a, wl)
                .map(|s| [s[0] - target[0], s[1] - target[1]])
        };
        let s = newton2(
            &f,
            x0,
            j0,
            self.fd_step(),
            self.aim_tolerance(),
            self.settings.max_aim_iterations,
        );
        let it = s.iterations;
        (s, it)
    }

    fn chief_aim(&self, src: &ObjectSource, wl: usize) -> AimData {
        match self.aiming {
            RayAiming::Off => AimData {
                chief: [0.0, 0.0],
                jacobian: None,
                converged: true,
                residual: 0.0,
            },
            RayAiming::Real { .. } => {
                let (mut s, _) = self.solve_aim(src, wl, [0.0, 0.0], [0.0, 0.0], None);
                if !s.converged {
                    // A wide-field paraxial seed can miss the front surfaces.
                    // Follow the chief ray from the axis to enter its basin.
                    let p = self.source_params(src);
                    let mut seed = [0.0, 0.0];
                    let mut jac = None;
                    for k in 0..=16 {
                        let fraction = k as f64 / 16.0;
                        let Some(source) =
                            self.source_from_params([fraction * p[0], fraction * p[1]])
                        else {
                            break;
                        };
                        let (next, _) = self.solve_aim(&source, wl, [0.0, 0.0], seed, jac);
                        if !next.converged {
                            break;
                        }
                        seed = next.x;
                        jac = next.jacobian;
                        if k == 16 {
                            s = next;
                        }
                    }
                }
                let f = |a: [f64; 2]| self.stop_hit(src, a, wl);
                let jac = f(s.x)
                    .and_then(|fx| fd_jacobian(&f, s.x, fx, self.fd_step()))
                    .or(s.jacobian);
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
        let source = self.field_source(&point)?;
        let aims = (0..self.num_wavelengths())
            .map(|w| self.chief_aim(&source, w))
            .collect();
        Ok(PreparedField {
            point,
            source,
            aims,
            error: None,
        })
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
    fn chief_image_hit(&self, src: &ObjectSource) -> Option<[f64; 2]> {
        let wl = self.sys.primary_wavelength;
        let aim = self.chief_aim(src, wl);
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
        let x0 = self.source_params(&guess);
        let f = |p: [f64; 2]| {
            let src = self.source_from_params(p)?;
            self.chief_image_hit(&src)
                .map(|h| [h[0] - target[0], h[1] - target[1]])
        };
        let scale = x0[0].hypot(x0[1]).max(1e-6);
        let tol = 1e-12 * target[0].hypot(target[1]).max(1e-3);
        let s = newton2(&f, x0, None, 1e-6 * scale, tol, 60);
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
                Ok(GeneratedRay {
                    ray,
                    aim: a,
                    report: AimReport {
                        iterations: 0,
                        residual: 0.0,
                        converged: true,
                    },
                })
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
                let (mut s, mut iters) = self.solve_aim(&field.source, wl, target, x0, ad.jacobian);
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
                    return Err(report);
                }
                let ray = self.start_ray(&field.source, s.x).ok_or(report)?;
                Ok(GeneratedRay {
                    ray,
                    aim: s.x,
                    report,
                })
            }
        }
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
