//! The per-ray reference trace: a pure function over a [`ResolvedSystem`].
//!
//! # Frames
//!
//! A ray is always expressed in one surface's local frame. The trace moves it
//! from surface `i−1` to surface `i` with the relative transform
//! `ResolvedSurface::to_next` of surface `i−1` (CONVENTIONS §7.1): with
//! `T = to_next = (o, R)` (surface `i`'s frame expressed in `i−1`'s),
//! `p_i = Rᵀ(p_{i−1} − o)` and `d_i = Rᵀ d_{i−1}`. When `to_next_is_axial`, the
//! transform is the exact translation `p_i = p_{i−1} − (0, 0, t_{i−1})`, using
//! the stored thickness rather than any difference of global positions.
//!
//! The **starting frame** (where a trace's input ray lives) is
//! * the object surface's local frame for a finite object, and
//! * the *entrance frame* for an object at infinity: surface 1's
//!   pre-placement frame, i.e. the global frame. Surface 1's own frame maps
//!   it into surface 1.
//!
//! # Per surface
//!
//! 1. transfer into the surface frame (above);
//! 2. intersect ([`crate::surface::intersect`]); the signed distance times the
//!    index before the surface is added to the optical path with exact
//!    products and compensated sums; internal absorption `exp(−α·t)` of the
//!    medium before the surface is accumulated as a sum of exponents;
//! 3. clip against the surface's fixed aperture (`ResolvedSurface::passes_aperture`),
//!    unless [`TraceOptions::clip`] is false. Coordinate breaks never clip;
//! 4. interact: refract (stable vector Snell, TIR detection), diffract, reflect, ideal
//!    lens, or nothing (dummy of any shape, coordinate break, and the image
//!    surface, where the direction is reported as incident). An ideal lens
//!    takes its object vergence from
//!    [`crate::paraxial::ideal_lens_object_vergence`] (an `O(i)` scalar y–nu
//!    sweep, done only when a ray reaches an ideal lens).
//!
//! The object surface (index 0) never interacts. A trace of an object at
//! infinity starts its optical path at the input ray's point; ray generation
//! places all rays of one field on a common plane normal to their direction,
//! so their path lengths share a plane-wavefront origin.
//!
//! Nothing in the no-record path allocates.

use mklight_core::{Interaction, ResolvedSystem, SurfaceShape, Vec3};
use serde::{Deserialize, Serialize};

use crate::interact::{diffract, ideal_lens, reflect, refract};
use crate::numeric::Compensated;
use crate::paraxial::ideal_lens_object_vergence;
use crate::surface::{IntersectError, IntersectSettings, intersect};

/// A ray: point and unit direction in one frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ray {
    pub pos: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(pos: Vec3, dir: Vec3) -> Self {
        Ray { pos, dir }
    }
}

/// Why a ray failed numerically or could not be generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailReason {
    /// Selected diffraction order has no real outgoing normal momentum.
    EvanescentOrder,
    /// Iterative asphere intersection did not converge.
    NoConvergence,
    /// The ray reached a region where the surface sag is undefined.
    OutsideDomain,
    /// A non-finite value appeared (including a non-finite input ray, which
    /// is reported at surface 0).
    NonFinite,
    /// The ray is parallel to an ideal-lens plane.
    Degenerate,
    /// Real ray aiming did not converge, so the requested pupil point could not
    /// be generated.
    AimNotConverged,
    /// The field point could not be resolved (e.g. a real image height that
    /// no chief ray reaches); no ray of the field is generated.
    FieldUnreachable,
}

/// Outcome of a trace. `surface` is the index where the ray stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TraceStatus {
    Ok,
    Miss { surface: usize },
    Clipped { surface: usize },
    TotalInternalReflection { surface: usize },
    Failed { surface: usize, reason: FailReason },
}

impl TraceStatus {
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, TraceStatus::Ok)
    }
    /// The surface where the ray stopped, `None` for `Ok`.
    pub fn surface(&self) -> Option<usize> {
        match *self {
            TraceStatus::Ok => None,
            TraceStatus::Miss { surface }
            | TraceStatus::Clipped { surface }
            | TraceStatus::TotalInternalReflection { surface }
            | TraceStatus::Failed { surface, .. } => Some(surface),
        }
    }
}

/// Options of a single trace.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceOptions {
    /// Apply fixed semi-diameters and explicit apertures.
    pub clip: bool,
    /// Stop after this surface (inclusive). `None` traces to the image.
    pub last_surface: Option<usize>,
    pub intersect: IntersectSettings,
}

impl Default for TraceOptions {
    fn default() -> Self {
        TraceOptions {
            clip: true,
            last_surface: None,
            intersect: IntersectSettings::default(),
        }
    }
}

impl TraceOptions {
    pub fn unclipped() -> Self {
        TraceOptions {
            clip: false,
            ..Default::default()
        }
    }
    pub fn to_surface(mut self, last: usize) -> Self {
        self.last_surface = Some(last);
        self
    }
}

/// Ray data at one surface, in that surface's local frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceRecord {
    /// Intersection point.
    pub point: Vec3,
    /// Direction after the interaction.
    pub dir: Vec3,
    /// Unit surface normal at the hit (towards local +z on the vertex sheet).
    pub normal: Vec3,
    /// Cosine of the angle of incidence, in [0, 1].
    pub cos_incidence: f64,
    /// Cumulative optical path length from the start of the trace (mm).
    pub opl: f64,
    /// Cumulative internal (bulk absorption) transmittance; Fresnel and
    /// coating losses are not included.
    pub transmittance: f64,
}

/// Result of [`trace_ray`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceResult {
    pub status: TraceStatus,
    /// Final ray, expressed in surface `surface`'s frame. On success it is the
    /// image-surface intersection and incident direction. On failure it is the
    /// last valid state (after transfer into the failing surface's frame,
    /// before or at its intersection).
    pub ray: Ray,
    pub surface: usize,
    /// Optical path from the start of the trace, compensated (`hi + lo`).
    pub opl: Compensated,
    /// Internal transmittance (bulk absorption only).
    pub transmittance: f64,
}

/// Transfer a ray from surface `i − 1`'s frame (or the starting frame when
/// `i == 1`) into surface `i`'s frame.
#[inline]
pub fn transfer_into(sys: &ResolvedSystem, i: usize, ray: Ray) -> Ray {
    debug_assert!(i >= 1);
    if i == 1 && sys.object_at_infinity {
        let f = &sys.surfaces[1].frame;
        if f.rot.is_identity() && f.origin == Vec3::ZERO {
            return ray;
        }
        return Ray::new(f.to_local_point(ray.pos), f.to_local_dir(ray.dir));
    }
    let prev = &sys.surfaces[i - 1];
    if prev.to_next_is_axial {
        let p = ray.pos;
        return Ray::new(Vec3::new(p.x, p.y, p.z - prev.thickness), ray.dir);
    }
    match &prev.to_next {
        Some(f) => Ray::new(f.to_local_point(ray.pos), f.to_local_dir(ray.dir)),
        None => ray,
    }
}

/// The starting frame expressed in global coordinates: the object surface's
/// frame for a finite object, the identity (entrance frame) at infinity.
/// Only ray *generation* uses it; the trace itself never does.
#[inline]
pub fn start_frame(sys: &ResolvedSystem) -> mklight_core::math::Frame {
    if sys.object_at_infinity {
        mklight_core::math::Frame::IDENTITY
    } else {
        sys.surfaces[0].frame
    }
}

/// Trace one ray from the starting frame (module docs) through the system.
///
/// `wl` indexes `sys.wavelengths_um`. When `record` is given it is cleared
/// and receives one [`SurfaceRecord`] per surface reached, starting with the
/// input ray at index 0 (so `record[i]` is surface `i` on success).
pub fn trace_ray(
    sys: &ResolvedSystem,
    start: &Ray,
    wl: usize,
    opts: &TraceOptions,
    mut record: Option<&mut Vec<SurfaceRecord>>,
) -> TraceResult {
    let n_surf = sys.surfaces.len();
    let image = n_surf - 1;
    let last = opts.last_surface.unwrap_or(image).min(image);
    let mut ray = *start;
    let mut opl = Compensated::ZERO;
    let mut absorb = 0.0f64; // Σ α·t
    if let Some(r) = record.as_deref_mut() {
        r.clear();
        r.push(SurfaceRecord {
            point: start.pos,
            dir: start.dir,
            normal: Vec3::Z,
            cos_incidence: start.dir.z.abs().min(1.0),
            opl: 0.0,
            transmittance: 1.0,
        });
    }
    let alpha_object = if sys.object_at_infinity {
        0.0
    } else {
        sys.object_medium.absorption_per_mm(sys.wavelengths_um[wl])
    };
    let done =
        |status: TraceStatus, ray: Ray, surface: usize, opl: Compensated, a: f64| TraceResult {
            status,
            ray,
            surface,
            opl,
            transmittance: (-a).exp(),
        };
    // A non-finite input is a numerical failure, not a geometric miss.
    if !(start.pos.is_finite() && start.dir.is_finite()) {
        return done(
            TraceStatus::Failed {
                surface: 0,
                reason: FailReason::NonFinite,
            },
            ray,
            0,
            opl,
            absorb,
        );
    }

    for i in 1..=last {
        ray = transfer_into(sys, i, ray);
        let s = &sys.surfaces[i];
        let n1 = sys.n_before(i, wl);
        let hit = match intersect(&s.shape, ray.pos, ray.dir, &opts.intersect) {
            Ok(h) => h,
            Err(e) => {
                let status = match e {
                    IntersectError::Miss => TraceStatus::Miss { surface: i },
                    IntersectError::NoConvergence => TraceStatus::Failed {
                        surface: i,
                        reason: FailReason::NoConvergence,
                    },
                    IntersectError::OutsideDomain => TraceStatus::Failed {
                        surface: i,
                        reason: FailReason::OutsideDomain,
                    },
                };
                return done(status, ray, i, opl, absorb);
            }
        };
        opl.add_product(n1, hit.t_transfer);
        opl.add_product(n1, hit.t_surface);
        let alpha = if i == 1 {
            alpha_object
        } else {
            sys.surfaces[i - 1].alpha_after[wl]
        };
        if alpha != 0.0 {
            absorb = alpha.mul_add(hit.t_transfer + hit.t_surface, absorb);
        }
        ray.pos = hit.point;
        if !ray.pos.is_finite() {
            return done(
                TraceStatus::Failed {
                    surface: i,
                    reason: FailReason::NonFinite,
                },
                ray,
                i,
                opl,
                absorb,
            );
        }
        let is_cb = matches!(s.shape, SurfaceShape::CoordinateBreak);
        if opts.clip && !is_cb && !s.passes_aperture(hit.point.x, hit.point.y) {
            return done(TraceStatus::Clipped { surface: i }, ray, i, opl, absorb);
        }

        let mut cos_i = ray.dir.dot(hit.normal).abs().min(1.0);
        if i < image && !is_cb {
            match (&s.shape, s.interaction) {
                // Dummy is a pass-through for every shape, ideal lenses included.
                (_, Interaction::Dummy) => {}
                (SurfaceShape::Diffractive { phase, .. }, inter) if phase.trace_order != 0 => {
                    let v = mklight_core::surface_math::phase(phase, hit.point.x, hit.point.y)
                        .expect("intersection checked phase domain");
                    let lambda = sys.wavelengths_um[wl] * 1e-3;
                    let reflected = inter == Interaction::Reflect;
                    let n2 = if reflected { n1 } else { s.n_after[wl] };
                    match diffract(
                        ray.dir,
                        hit.normal,
                        n1,
                        n2,
                        v.gradient,
                        lambda,
                        phase.trace_order,
                        reflected,
                    ) {
                        Some(d) => {
                            ray.dir = d;
                            opl.add_product(
                                phase.trace_order as f64 * lambda / std::f64::consts::TAU,
                                v.sag,
                            );
                        }
                        None => {
                            return done(
                                TraceStatus::Failed {
                                    surface: i,
                                    reason: FailReason::EvanescentOrder,
                                },
                                ray,
                                i,
                                opl,
                                absorb,
                            );
                        }
                    }
                }
                (SurfaceShape::Paraxial { focal_length }, inter) => {
                    let mut k = ray.dir;
                    let n2 = if inter == Interaction::Reflect {
                        k = reflect(k, hit.normal).0;
                        n1
                    } else {
                        s.n_after[wl]
                    };
                    let vergence = ideal_lens_object_vergence(sys, i, wl);
                    match ideal_lens(hit.point, k, *focal_length, n1, n2, vergence) {
                        Some((d, delta)) => {
                            ray.dir = d;
                            opl.add(delta);
                        }
                        None => {
                            return done(
                                TraceStatus::Failed {
                                    surface: i,
                                    reason: FailReason::Degenerate,
                                },
                                ray,
                                i,
                                opl,
                                absorb,
                            );
                        }
                    }
                }
                (_, Interaction::Refract) => {
                    match refract(ray.dir, hit.normal, n1, s.n_after[wl]) {
                        Some((d, c)) => {
                            ray.dir = d;
                            cos_i = c;
                        }
                        None => {
                            return done(
                                TraceStatus::TotalInternalReflection { surface: i },
                                ray,
                                i,
                                opl,
                                absorb,
                            );
                        }
                    }
                }
                (_, Interaction::Reflect) => {
                    let (d, c) = reflect(ray.dir, hit.normal);
                    ray.dir = d;
                    cos_i = c;
                }
            }
        }
        if let Some(r) = record.as_deref_mut() {
            r.push(SurfaceRecord {
                point: ray.pos,
                dir: ray.dir,
                normal: hit.normal,
                cos_incidence: cos_i,
                opl: opl.value(),
                transmittance: (-absorb).exp(),
            });
        }
    }
    done(TraceStatus::Ok, ray, last, opl, absorb)
}

/// Trace with records into a freshly allocated vector (convenience for
/// layouts, tests and debugging).
pub fn trace_ray_recorded(
    sys: &ResolvedSystem,
    start: &Ray,
    wl: usize,
    opts: &TraceOptions,
) -> (TraceResult, Vec<SurfaceRecord>) {
    let mut rec = Vec::with_capacity(sys.surfaces.len());
    let r = trace_ray(sys, start, wl, opts, Some(&mut rec));
    (r, rec)
}

/// Map a ray given in surface `surface`'s frame (surface 0: the starting
/// frame) into the global frame, for layouts and tests. Not used by traces.
pub fn to_global(sys: &ResolvedSystem, surface: usize, ray: &Ray) -> Ray {
    let f = if surface == 0 {
        start_frame(sys)
    } else {
        sys.surfaces[surface].frame
    };
    Ray::new(f.to_global_point(ray.pos), f.to_global_dir(ray.dir))
}
