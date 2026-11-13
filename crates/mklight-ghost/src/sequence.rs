//! The general sequence tracer: one ray along an arbitrary list of
//! [`Step`]s, forward and backward through the sequential surfaces, with
//! per-polarisation radiometry.
//!
//! # Frames
//!
//! A ray is always in the local frame of the surface it last met (or the
//! starting frame of [`mklight_trace::trace`] before surface 1). A step to
//! surface `i` arriving
//!
//! * **forward** (from `i − 1`) uses the reference transfer
//!   [`transfer_into`]: `p_i = Rᵀ(p_{i−1} − o)` with `to_next(i−1) = (o, R)`,
//!   or the exact `−t` translation on axial gaps;
//! * **backward** (from `i + 1`) uses the inverse of `to_next(i)`, written in
//!   the same subtract-then-rotate form, `p_i = R(p_{i+1} − q)`,
//!   `d_i = R d_{i+1}`, with `q = −Rᵀo` (vertex `i` in frame `i + 1`)
//!   precomputed per surface, or the exact `+t_i` translation on axial
//!   gaps. Composing a backward step with the forward one is the identity
//!   up to rounding (tested).
//!
//! Intersections ([`intersect_within`], with the surface's fixed extent
//! choosing between two crossings as in the sequential trace), refraction
//! ([`refract`]), reflection ([`reflect`]) and the ideal lens
//! ([`ideal_lens`]) are the reference tracer's functions, called with the
//! ray in the surface's local frame.
//!
//! # Media
//!
//! Arriving forward at `i`, the incident medium is the one before `i`; the
//! medium on the far side is the one after `i`. Arriving backward, the two
//! swap. A pass refracts into the far medium; a reflection (designed mirror,
//! or ghost reflection at a refracting surface) stays in the incident
//! medium. The segment between `i` and `i + 1` lies in the medium after `i`
//! whichever way it is crossed, and absorbs with `alpha_after[i]`.
//!
//! An ideal lens crossed backward has the image-space focal length
//! `f·n_before/n_after` (in its incident medium), which makes the backward
//! crossing the exact inverse of the forward one; the ghost OPL does not
//! include the ideal lens's phase term (irrelevant for irradiance).
//!
//! # Virtual segments
//!
//! A sequential design may contain virtual segments (a dummy surface or a
//! stop plane placed "behind" the ray), but physical light cannot travel
//! backwards between two interactions. With
//! [`GhostSettings::reject_virtual`](crate::GhostSettings) the signed
//! distance travelled since the last *physical* interaction (any surface
//! that is not [`InterfaceKind::Passive`], the image included) is summed
//! over the intermediate passive surfaces, and a ray whose sum is negative at
//! a physical surface is lost (`Virtual`). Passive surfaces can therefore
//! sit on either side of the ray. For an object at infinity nothing is
//! counted before the first physical interaction (the light comes from
//! `z = −∞`; the start plane and any passive planes before the first
//! element are reference planes, not positions).
//!
//! The rule applies to the primary path as well: a design whose real rays
//! cross a physically impossible element (negative edge thickness, overlapping
//! elements) loses those rays here although the reference tracer passes
//! them. [`ghost_irradiance`](crate::ghost_irradiance) reports such primary
//! losses as a `RunInfo` warning.
//!
//! # Radiometry
//!
//! At each interaction the coating code ([`ResolvedCoating::rt`], bare
//! Fresnel, thin film, ideal or tabulated) is evaluated once at the actual
//! angle of incidence and wavelength; a pass multiplies the s and p weights
//! by `T_s`, `T_p`, a reflection by `R_s`, `R_p` (a designed mirror uses the
//! mirror form), a sensor reflection by the sensor reflectance `R_sensor`,
//! and detection by `1 − R_sensor` (the absorbed fraction; 1 without a
//! sensor reflectance). s and p are carried as
//! two incoherent channels and averaged at the end, `w = ½(w_s + w_p)·τ`,
//! with the bulk transmittance `τ = exp(−Σ α·t)` over all segments. This is
//! exact for meridional rays of rotationally symmetric systems and neglects
//! the rotation of the plane of incidence between surfaces for skew rays.
//! Because the primary pass and the ghost reflection use the same `rt` call,
//! `T + R + A = 1` holds at every interface (tested).
//!
//! [`ResolvedCoating::rt`]: mklight_core::materials::coating::ResolvedCoating::rt

use mklight_core::materials::coating::InterfaceRT;
use mklight_core::{Interaction, SurfaceShape, Vec3};
use mklight_trace::interact::{diffract, ideal_lens, reflect, refract};
use mklight_trace::surface::{Extent, IntersectError, intersect_within};
use mklight_trace::trace::transfer_into;
use mklight_trace::{Compensated, Ray};
use serde::{Deserialize, Serialize};

use crate::model::GhostModel;
use crate::path::{Action, Direction, Step};
use crate::signature::InterfaceKind;

/// A signed distance between physical interactions below `−VIRTUAL_TOL`
/// (mm) is virtual.
const VIRTUAL_TOL: f64 = 1e-9;

/// Why a sequence trace failed numerically or was given an impossible step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceFail {
    NoConvergence,
    OutsideDomain,
    NonFinite,
    /// Ray parallel to an ideal-lens plane.
    Degenerate,
    /// A step that is not adjacent to the previous surface, or a reflection
    /// at a surface that cannot reflect.
    InvalidStep,
}

/// Outcome of a sequence trace. `step` indexes the step list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SequenceStatus {
    Ok,
    Miss {
        step: u32,
        surface: u32,
    },
    Clipped {
        step: u32,
        surface: u32,
    },
    TotalInternalReflection {
        step: u32,
        surface: u32,
    },
    /// The intersection lies behind the ray (see `GhostSettings::reject_virtual`).
    Virtual {
        step: u32,
        surface: u32,
    },
    Failed {
        step: u32,
        surface: u32,
        reason: SequenceFail,
    },
}

impl SequenceStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, SequenceStatus::Ok)
    }
}

/// Ray data at one step, in that surface's local frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepRecord {
    pub step: Step,
    pub point: Vec3,
    /// Incident direction.
    pub dir_in: Vec3,
    /// Direction after the interaction.
    pub dir_out: Vec3,
    pub normal: Vec3,
    pub cos_incidence: f64,
    /// Signed geometric length of the segment that reached this surface.
    pub distance: f64,
    /// Incident medium index.
    pub n_in: f64,
    /// Index of the medium the ray continues in.
    pub n_out: f64,
    /// Full power split at this interaction (`TRANSPARENT` where nothing
    /// happens).
    pub split: InterfaceRT,
    /// Factors applied to the s and p weights here.
    pub applied: [f64; 2],
    /// Cumulative s and p weights after this step (interfaces only).
    pub weight: [f64; 2],
    /// Cumulative bulk transmittance.
    pub internal: f64,
    /// Cumulative optical path from the start.
    pub opl: f64,
}

/// Result of a sequence trace.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequenceResult {
    pub status: SequenceStatus,
    /// Final ray in surface `surface`'s frame (on success: the last step's
    /// hit point and outgoing direction; at the image, the incident one).
    pub ray: Ray,
    pub surface: usize,
    /// Steps completed.
    pub steps_done: usize,
    /// Interface (s, p) weights, without bulk absorption.
    pub weight_s: f64,
    pub weight_p: f64,
    /// Bulk transmittance `exp(−Σ α t)`.
    pub internal: f64,
    pub opl: Compensated,
}

impl SequenceResult {
    /// Unpolarised flux weight relative to the input: `½(w_s + w_p)·τ`.
    pub fn weight(&self) -> f64 {
        0.5 * (self.weight_s + self.weight_p) * self.internal
    }
}

impl GhostModel {
    /// Transfer a ray from surface `surface + 1`'s frame into `surface`'s
    /// (module docs).
    #[inline]
    pub fn transfer_back(&self, surface: usize, ray: Ray) -> Ray {
        let s = &self.system().surfaces[surface];
        if s.to_next_is_axial {
            let p = ray.pos;
            return Ray::new(Vec3::new(p.x, p.y, p.z + s.thickness), ray.dir);
        }
        match self.back_frame(surface) {
            Some(f) => Ray::new(f.to_local_point(ray.pos), f.to_local_dir(ray.dir)),
            None => ray,
        }
    }

    /// Trace a ray given in the starting frame along `steps`.
    pub fn trace_steps(
        &self,
        start: &Ray,
        wl: usize,
        steps: &[Step],
        record: Option<&mut Vec<StepRecord>>,
    ) -> SequenceResult {
        self.trace_from(0, start, wl, steps, record)
    }

    /// Trace a ray given in surface `from`'s frame (0: the starting frame)
    /// along `steps`; the first step must be adjacent to `from`.
    pub fn trace_from(
        &self,
        from: usize,
        start: &Ray,
        wl: usize,
        steps: &[Step],
        mut record: Option<&mut Vec<StepRecord>>,
    ) -> SequenceResult {
        if let Some(r) = record.as_deref_mut() {
            r.clear();
        }
        let mut state = match self.begin(from, start) {
            Ok(s) => s,
            Err(r) => return r,
        };
        for st in steps {
            if let Some(r) = self.advance(&mut state, wl, st, record.as_deref_mut()) {
                return r;
            }
        }
        state.finish()
    }

    /// The state before the first step of [`Self::trace_from`], or its
    /// result for a non-finite start.
    pub(crate) fn begin(&self, from: usize, start: &Ray) -> Result<TraceState, SequenceResult> {
        let state = TraceState {
            ray: *start,
            at: from,
            k: 0,
            ws: 1.0,
            wp: 1.0,
            absorb: 0.0,
            opl: Compensated::ZERO,
            // Signed distance travelled since the last physical interaction;
            // not counted before the first one for an object at infinity.
            since_physical: 0.0,
            counting: !(from == 0 && self.system().object_at_infinity),
        };
        if !(start.pos.is_finite() && start.dir.is_finite()) {
            let st = SequenceStatus::Failed {
                step: 0,
                surface: from as u32,
                reason: SequenceFail::NonFinite,
            };
            return Err(state.result(st, from));
        }
        Ok(state)
    }

    /// One step of [`Self::trace_from`]: `None` when it completed (the state
    /// then holds the next step's input), else the path's final result.
    /// A copy of a state continues bit-identically, so paths that share a
    /// step prefix can share its trace.
    pub(crate) fn advance(
        &self,
        state: &mut TraceState,
        wl: usize,
        st: &Step,
        record: Option<&mut Vec<StepRecord>>,
    ) -> Option<SequenceResult> {
        let sys = self.system();
        let image = sys.image_index();
        let lambda = sys.wavelengths_um[wl];
        let TraceState {
            mut ray,
            mut at,
            k,
            mut ws,
            mut wp,
            mut absorb,
            mut opl,
            mut since_physical,
            mut counting,
        } = *state;
        let done = |status, ray, surface, k, ws, wp, absorb: f64, opl| {
            Some(SequenceResult {
                status,
                ray,
                surface,
                steps_done: k,
                weight_s: ws,
                weight_p: wp,
                internal: (-absorb).exp(),
                opl,
            })
        };
        {
            let s = st.surface as usize;
            let (kk, ss) = (k as u32, st.surface);
            let fail = |reason| SequenceStatus::Failed {
                step: kk,
                surface: ss,
                reason,
            };
            // 1. Frame transfer.
            let adjacent = match st.direction {
                Direction::Forward => s == at + 1 && s <= image,
                Direction::Backward => s + 1 == at && s >= 1,
            };
            if !adjacent {
                return done(
                    fail(SequenceFail::InvalidStep),
                    ray,
                    at,
                    k,
                    ws,
                    wp,
                    absorb,
                    opl,
                );
            }
            ray = match st.direction {
                Direction::Forward => transfer_into(sys, s, ray),
                Direction::Backward => self.transfer_back(s, ray),
            };
            at = s;
            let surf = &sys.surfaces[s];
            let (n_in, n_far) = self.media(s, st.direction, wl);
            let alpha = match st.direction {
                Direction::Forward if s == 1 => {
                    if sys.object_at_infinity {
                        0.0
                    } else {
                        sys.object_medium.absorption_per_mm(lambda)
                    }
                }
                Direction::Forward => sys.surfaces[s - 1].alpha_after[wl],
                Direction::Backward => surf.alpha_after[wl],
            };
            // 2. Intersection.
            let outline = surf.root_extent();
            let extent = outline.as_ref().map(|outline| Extent {
                outline,
                // Only the start arrives forward at surface 1.
                from_infinity: s == 1
                    && sys.object_at_infinity
                    && matches!(st.direction, Direction::Forward),
            });
            let hit = match intersect_within(
                &surf.shape,
                extent,
                ray.pos,
                ray.dir,
                &self.settings().intersect,
            ) {
                Ok(h) => h,
                Err(e) => {
                    let status = match e {
                        IntersectError::Miss => SequenceStatus::Miss {
                            step: kk,
                            surface: ss,
                        },
                        IntersectError::NoConvergence => fail(SequenceFail::NoConvergence),
                        IntersectError::OutsideDomain => fail(SequenceFail::OutsideDomain),
                    };
                    return done(status, ray, s, k, ws, wp, absorb, opl);
                }
            };
            let t = hit.t_transfer + hit.t_surface;
            // Virtual-segment rule (module docs).
            if counting {
                since_physical += t;
            }
            if self.interface_kinds()[s] != InterfaceKind::Passive {
                if counting && self.settings().reject_virtual && since_physical < -VIRTUAL_TOL {
                    let status = SequenceStatus::Virtual {
                        step: kk,
                        surface: ss,
                    };
                    return done(status, ray, s, k, ws, wp, absorb, opl);
                }
                since_physical = 0.0;
                counting = true;
            }
            opl.add_product(n_in, hit.t_transfer);
            opl.add_product(n_in, hit.t_surface);
            if alpha != 0.0 {
                absorb = alpha.mul_add(t, absorb);
            }
            ray.pos = hit.point;
            if !ray.pos.is_finite() {
                return done(
                    fail(SequenceFail::NonFinite),
                    ray,
                    s,
                    k,
                    ws,
                    wp,
                    absorb,
                    opl,
                );
            }
            // 3. Clipping.
            if !self.passes(s, hit.point.x, hit.point.y) {
                let status = SequenceStatus::Clipped {
                    step: kk,
                    surface: ss,
                };
                return done(status, ray, s, k, ws, wp, absorb, opl);
            }
            // 4. Interaction.
            let dir_in = ray.dir;
            let cos_i = dir_in.dot(hit.normal).abs().min(1.0);
            let mut split = InterfaceRT::TRANSPARENT;
            let mut applied = [1.0, 1.0];
            let mut n_out = n_in;
            let is_cb = matches!(surf.shape, SurfaceShape::CoordinateBreak);
            if s == image {
                if st.action == Action::Reflect {
                    let Some(r) = self.settings().sensor_reflectance else {
                        return done(
                            fail(SequenceFail::InvalidStep),
                            ray,
                            s,
                            k,
                            ws,
                            wp,
                            absorb,
                            opl,
                        );
                    };
                    if let Some(sensor) = &sys.system().sensor
                        && (hit.point.x.abs() > sensor.half_width()
                            || hit.point.y.abs() > sensor.half_height())
                    {
                        let status = SequenceStatus::Clipped {
                            step: kk,
                            surface: ss,
                        };
                        return done(status, ray, s, k, ws, wp, absorb, opl);
                    }
                    split = InterfaceRT {
                        rs: r,
                        rp: r,
                        ts: 1.0 - r,
                        tp: 1.0 - r,
                    };
                    applied = [r, r];
                    ray.dir = reflect(dir_in, hit.normal).0;
                } else if let Some(r) = self.settings().sensor_reflectance {
                    // Detection: the sensor absorbs what it does not reflect.
                    split = InterfaceRT {
                        rs: r,
                        rp: r,
                        ts: 1.0 - r,
                        tp: 1.0 - r,
                    };
                    applied = [1.0 - r, 1.0 - r];
                }
                // Without a sensor reflectance detection absorbs everything.
            } else if is_cb || surf.interaction == Interaction::Dummy {
                if st.action == Action::Reflect {
                    return done(
                        fail(SequenceFail::InvalidStep),
                        ray,
                        s,
                        k,
                        ws,
                        wp,
                        absorb,
                        opl,
                    );
                }
            } else if surf.interaction == Interaction::Reflect {
                // Designed mirror.
                if st.action == Action::Reflect {
                    return done(
                        fail(SequenceFail::InvalidStep),
                        ray,
                        s,
                        k,
                        ws,
                        wp,
                        absorb,
                        opl,
                    );
                }
                split = surf
                    .coating
                    .rt(lambda, n_in, n_in, cos_i, true, st.direction.arrival());
                applied = [split.rs, split.rp];
                let k_ref = reflect(dir_in, hit.normal).0;
                ray.dir = match surf.shape {
                    SurfaceShape::Paraxial { focal_length } => {
                        match ideal_lens(hit.point, k_ref, focal_length, n_in, n_in, 0.0) {
                            Some((d, _)) => d,
                            None => {
                                return done(
                                    fail(SequenceFail::Degenerate),
                                    ray,
                                    s,
                                    k,
                                    ws,
                                    wp,
                                    absorb,
                                    opl,
                                );
                            }
                        }
                    }
                    _ => k_ref,
                };
            } else {
                // Refracting surface.
                split = surf
                    .coating
                    .rt(lambda, n_in, n_far, cos_i, false, st.direction.arrival());
                match (st.action, &surf.shape) {
                    (Action::Pass, SurfaceShape::Paraxial { focal_length }) => {
                        let f = match st.direction {
                            Direction::Forward => *focal_length,
                            Direction::Backward => *focal_length * n_far / n_in,
                        };
                        match ideal_lens(hit.point, dir_in, f, n_in, n_far, 0.0) {
                            Some((d, _)) => ray.dir = d,
                            None => {
                                return done(
                                    fail(SequenceFail::Degenerate),
                                    ray,
                                    s,
                                    k,
                                    ws,
                                    wp,
                                    absorb,
                                    opl,
                                );
                            }
                        }
                        applied = [split.ts, split.tp];
                        n_out = n_far;
                    }
                    (Action::Pass, SurfaceShape::Diffractive { .. }) => {
                        applied = [split.ts, split.tp];
                        n_out = n_far;
                    }
                    (Action::Pass, _) => match refract(dir_in, hit.normal, n_in, n_far) {
                        Some((d, _)) => {
                            ray.dir = d;
                            applied = [split.ts, split.tp];
                            n_out = n_far;
                        }
                        None => {
                            let status = SequenceStatus::TotalInternalReflection {
                                step: kk,
                                surface: ss,
                            };
                            return done(status, ray, s, k, ws, wp, absorb, opl);
                        }
                    },
                    (Action::Reflect, SurfaceShape::Paraxial { .. }) => {
                        return done(
                            fail(SequenceFail::InvalidStep),
                            ray,
                            s,
                            k,
                            ws,
                            wp,
                            absorb,
                            opl,
                        );
                    }
                    (Action::Reflect, _) => {
                        ray.dir = reflect(dir_in, hit.normal).0;
                        applied = [split.rs, split.rp];
                    }
                }
            }
            if s != image
                && !surf.is_passive()
                && let SurfaceShape::Diffractive { phase, .. } = &surf.shape
            {
                // The reference tracer's grating step, in the step's order.
                let order = st.order.unwrap_or(phase.trace_order);
                let reflected = surf.is_mirror() || st.action == Action::Reflect;
                let v = mklight_core::surface_math::phase(phase, hit.point.x, hit.point.y)
                    .expect("intersection checked the phase domain");
                let scale = order as f64 * lambda * 1e-3 / std::f64::consts::TAU;
                let g = v.gradient.map(|d| d * scale);
                let Some(d) = diffract(dir_in, hit.normal, n_in, n_out, g, reflected) else {
                    return done(
                        fail(SequenceFail::Degenerate),
                        ray,
                        s,
                        k,
                        ws,
                        wp,
                        absorb,
                        opl,
                    );
                };
                ray.dir = d;
                opl.add_product(scale, v.sag);
                let eta = mklight_core::materials::diffraction::order_efficiency(
                    sys,
                    s,
                    wl,
                    st.order,
                    reflected,
                    st.direction == Direction::Backward,
                );
                applied = applied.map(|a| a * eta);
            }
            ws *= applied[0];
            wp *= applied[1];
            if let Some(r) = record {
                r.push(StepRecord {
                    step: *st,
                    point: ray.pos,
                    dir_in,
                    dir_out: ray.dir,
                    normal: hit.normal,
                    cos_incidence: cos_i,
                    distance: t,
                    n_in,
                    n_out,
                    split,
                    applied,
                    weight: [ws, wp],
                    internal: (-absorb).exp(),
                    opl: opl.value(),
                });
            }
        }
        *state = TraceState {
            ray,
            at,
            k: k + 1,
            ws,
            wp,
            absorb,
            opl,
            since_physical,
            counting,
        };
        None
    }
}

/// The running state of a sequence trace between steps (see
/// [`GhostModel::advance`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceState {
    ray: Ray,
    at: usize,
    /// Steps completed.
    k: usize,
    ws: f64,
    wp: f64,
    absorb: f64,
    opl: Compensated,
    since_physical: f64,
    counting: bool,
}

impl TraceState {
    fn result(&self, status: SequenceStatus, surface: usize) -> SequenceResult {
        SequenceResult {
            status,
            ray: self.ray,
            surface,
            steps_done: self.k,
            weight_s: self.ws,
            weight_p: self.wp,
            internal: (-self.absorb).exp(),
            opl: self.opl,
        }
    }
    /// The result after the last step.
    pub(crate) fn finish(&self) -> SequenceResult {
        self.result(SequenceStatus::Ok, self.at)
    }
}
