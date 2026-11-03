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
use crate::surface::{IntersectError, IntersectSettings, SHEET_TOL, intersect};

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

/// State of one ray between surfaces: the ray in the current surface frame,
/// its compensated optical path and `Σ α·t`.
#[derive(Clone, Copy, Debug)]
struct Walk {
    ray: Ray,
    opl: Compensated,
    absorb: f64,
}

impl Walk {
    fn new(ray: Ray) -> Self {
        Walk {
            ray,
            opl: Compensated::ZERO,
            absorb: 0.0,
        }
    }
    fn result(&self, status: TraceStatus, surface: usize) -> TraceResult {
        TraceResult {
            status,
            ray: self.ray,
            surface,
            opl: self.opl,
            transmittance: (-self.absorb).exp(),
        }
    }
}

/// Absorption coefficient of object space (0 at infinity).
#[inline]
fn object_alpha(sys: &ResolvedSystem, wl: usize) -> f64 {
    if sys.object_at_infinity {
        0.0
    } else {
        sys.object_medium.absorption_per_mm(sys.wavelengths_um[wl])
    }
}

#[inline]
fn first_record(start: &Ray) -> SurfaceRecord {
    SurfaceRecord {
        point: start.pos,
        dir: start.dir,
        normal: Vec3::Z,
        cos_incidence: start.dir.z.abs().min(1.0),
        opl: 0.0,
        transmittance: 1.0,
    }
}

/// One surface of a trace (module docs, "Per surface"): transfer into
/// surface `i`, intersect, clip and interact. On failure `w` holds the last
/// valid state and the status is returned. The scalar [`trace_ray`] and the
/// interleaved [`trace_rays`] both run exactly this code, so they agree
/// bitwise.
#[inline(always)]
fn step(
    sys: &ResolvedSystem,
    i: usize,
    wl: usize,
    alpha_object: f64,
    opts: &TraceOptions,
    w: &mut Walk,
    record: Option<&mut Vec<SurfaceRecord>>,
) -> Result<(), TraceStatus> {
    let image = sys.surfaces.len() - 1;
    w.ray = transfer_into(sys, i, w.ray);
    let s = &sys.surfaces[i];
    let n1 = sys.n_before(i, wl);
    let hit = match intersect(&s.shape, w.ray.pos, w.ray.dir, &opts.intersect) {
        Ok(h) => h,
        Err(e) => {
            return Err(match e {
                IntersectError::Miss => TraceStatus::Miss { surface: i },
                IntersectError::NoConvergence => TraceStatus::Failed {
                    surface: i,
                    reason: FailReason::NoConvergence,
                },
                IntersectError::OutsideDomain => TraceStatus::Failed {
                    surface: i,
                    reason: FailReason::OutsideDomain,
                },
            });
        }
    };
    w.opl.add_product(n1, hit.t_transfer);
    w.opl.add_product(n1, hit.t_surface);
    let alpha = if i == 1 {
        alpha_object
    } else {
        sys.surfaces[i - 1].alpha_after[wl]
    };
    if alpha != 0.0 {
        w.absorb = alpha.mul_add(hit.t_transfer + hit.t_surface, w.absorb);
    }
    w.ray.pos = hit.point;
    if !w.ray.pos.is_finite() {
        return Err(TraceStatus::Failed {
            surface: i,
            reason: FailReason::NonFinite,
        });
    }
    let is_cb = matches!(s.shape, SurfaceShape::CoordinateBreak);
    if opts.clip && !is_cb && !s.passes_aperture(hit.point.x, hit.point.y) {
        return Err(TraceStatus::Clipped { surface: i });
    }

    let mut cos_i = w.ray.dir.dot(hit.normal).abs().min(1.0);
    if i < image && !is_cb {
        match (&s.shape, s.interaction) {
            // Dummy is a pass-through for every shape, ideal lenses included.
            (_, Interaction::Dummy) => {}
            (SurfaceShape::Diffractive { phase, .. }, inter) if phase.trace_order != 0 => {
                let v = mklight_core::surface_math::phase(phase, hit.point.x, hit.point.y)
                    .expect("intersection checked the phase domain");
                // mλ/(2π), λ in mm: scales ∇Φ to momentum and Φ to OPL.
                let scale = phase.trace_order as f64 * sys.wavelengths_um[wl] * 1e-3
                    / std::f64::consts::TAU;
                let reflected = inter == Interaction::Reflect;
                let n2 = if reflected { n1 } else { s.n_after[wl] };
                let g = v.gradient.map(|d| d * scale);
                let Some(d) = diffract(w.ray.dir, hit.normal, n1, n2, g, reflected) else {
                    return Err(TraceStatus::Failed {
                        surface: i,
                        reason: FailReason::EvanescentOrder,
                    });
                };
                w.ray.dir = d;
                w.opl.add_product(scale, v.sag);
            }
            (SurfaceShape::Paraxial { focal_length }, inter) => {
                let mut k = w.ray.dir;
                let n2 = if inter == Interaction::Reflect {
                    k = reflect(k, hit.normal).0;
                    n1
                } else {
                    s.n_after[wl]
                };
                let vergence = ideal_lens_object_vergence(sys, i, wl);
                let Some((d, delta)) = ideal_lens(hit.point, k, *focal_length, n1, n2, vergence)
                else {
                    return Err(TraceStatus::Failed {
                        surface: i,
                        reason: FailReason::Degenerate,
                    });
                };
                w.ray.dir = d;
                w.opl.add(delta);
            }
            (_, Interaction::Refract) => {
                let Some((d, c)) = refract(w.ray.dir, hit.normal, n1, s.n_after[wl]) else {
                    return Err(TraceStatus::TotalInternalReflection { surface: i });
                };
                w.ray.dir = d;
                cos_i = c;
            }
            (_, Interaction::Reflect) => {
                let (d, c) = reflect(w.ray.dir, hit.normal);
                w.ray.dir = d;
                cos_i = c;
            }
        }
    }
    if let Some(r) = record {
        r.push(SurfaceRecord {
            point: w.ray.pos,
            dir: w.ray.dir,
            normal: hit.normal,
            cos_incidence: cos_i,
            opl: w.opl.value(),
            transmittance: (-w.absorb).exp(),
        });
    }
    Ok(())
}

/// The last surface a trace with `opts` reaches.
#[inline]
fn last_surface(sys: &ResolvedSystem, opts: &TraceOptions) -> usize {
    let image = sys.surfaces.len() - 1;
    opts.last_surface.unwrap_or(image).min(image)
}

/// A non-finite input is a numerical failure, not a geometric miss.
#[inline]
fn non_finite_start(start: &Ray) -> Option<TraceResult> {
    (!(start.pos.is_finite() && start.dir.is_finite())).then(|| {
        Walk::new(*start).result(
            TraceStatus::Failed {
                surface: 0,
                reason: FailReason::NonFinite,
            },
            0,
        )
    })
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
    let last = last_surface(sys, opts);
    if let Some(r) = record.as_deref_mut() {
        r.clear();
        r.push(first_record(start));
    }
    if let Some(failed) = non_finite_start(start) {
        return failed;
    }
    let alpha_object = object_alpha(sys, wl);
    let mut w = Walk::new(*start);
    for i in 1..=last {
        if let Err(status) = step(
            sys,
            i,
            wl,
            alpha_object,
            opts,
            &mut w,
            record.as_deref_mut(),
        ) {
            return w.result(status, i);
        }
    }
    w.result(TraceStatus::Ok, last)
}

/// Rays traced together by [`trace_rays`].
pub const PACKET: usize = 8;

type Lane = [f64; PACKET];

/// Structure-of-arrays state of one packet: point, direction, compensated
/// optical path and `Σ α·t` of each ray.
#[derive(Clone, Copy)]
struct Lanes {
    p: [Lane; 3],
    d: [Lane; 3],
    hi: Lane,
    lo: Lane,
    absorb: Lane,
}

impl Lanes {
    fn walk(&self, l: usize) -> Walk {
        Walk {
            ray: Ray::new(
                Vec3::new(self.p[0][l], self.p[1][l], self.p[2][l]),
                Vec3::new(self.d[0][l], self.d[1][l], self.d[2][l]),
            ),
            opl: Compensated {
                hi: self.hi[l],
                lo: self.lo[l],
            },
            absorb: self.absorb[l],
        }
    }
    fn set(&mut self, l: usize, w: &Walk) {
        let (p, d) = (w.ray.pos, w.ray.dir);
        (self.p[0][l], self.p[1][l], self.p[2][l]) = (p.x, p.y, p.z);
        (self.d[0][l], self.d[1][l], self.d[2][l]) = (d.x, d.y, d.z);
        (self.hi[l], self.lo[l], self.absorb[l]) = (w.opl.hi, w.opl.lo, w.absorb);
    }
}

/// A surface the lane kernel handles: axial (or no) transfer, a plane or
/// a conic, refraction or no interaction. Everything else, and every ray
/// that leaves the regular path of [`step`] at such a surface (miss,
/// tangent root, clipping, TIR, non-finite values), goes through [`step`].
#[derive(Clone, Copy)]
struct LaneSurface {
    /// Thickness subtracted from `z` by the transfer (`None`: no transfer).
    shift: Option<f64>,
    /// `Some((c, k))` for a conic with `c ≠ 0`, `None` for a plane.
    conic: Option<(f64, f64)>,
    refract: bool,
}

fn lane_surface(sys: &ResolvedSystem, i: usize) -> Option<LaneSurface> {
    let image = sys.surfaces.len() - 1;
    let shift = if i == 1 && sys.object_at_infinity {
        let f = &sys.surfaces[1].frame;
        (f.rot.is_identity() && f.origin == Vec3::ZERO).then_some(None)?
    } else {
        let prev = &sys.surfaces[i - 1];
        if prev.to_next_is_axial {
            Some(prev.thickness)
        } else if prev.to_next.is_none() {
            None
        } else {
            return None;
        }
    };
    let s = &sys.surfaces[i];
    let conic = match s.shape {
        SurfaceShape::Plane => None,
        SurfaceShape::Conic { curvature, conic } if curvature != 0.0 => Some((curvature, conic)),
        _ => return None,
    };
    let refract = i < image
        && match s.interaction {
            Interaction::Refract => true,
            Interaction::Dummy => false,
            _ => return None,
        };
    Some(LaneSurface {
        shift,
        conic,
        refract,
    })
}

/// [`step`] at a [`LaneSurface`] for every lane, without records: the same
/// operations as [`step`] (transfer, [`crate::surface::intersect_conic`] or
/// `intersect_plane`, `conic_normal`, [`Compensated::add_product`],
/// absorption, [`refract`]) written lane by lane with selects in place of
/// branches, so it vectorises. `regular[l]` is false where [`step`] would
/// take another branch; those lanes must be redone with [`step`] from `st`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lane_step(
    ls: &LaneSurface,
    n1: &Lane,
    n2: &Lane,
    alpha: &Lane,
    st: &Lanes,
    out: &mut Lanes,
    regular: &mut [bool; PACKET],
    incidence: &mut ([Lane; 3], Lane),
) {
    const N: usize = PACKET;
    let [px, py, mut pz] = st.p;
    if let Some(t) = ls.shift {
        for z in &mut pz {
            *z -= t;
        }
    }
    let [dx, dy, dz] = st.d;
    let (mut tt, mut ts) = ([0.0; N], [0.0; N]);
    let (mut hx, mut hy, mut hz) = ([0.0; N], [0.0; N], [0.0; N]);
    let (mut nx, mut ny, mut nz) = ([0.0; N], [0.0; N], [1.0; N]);
    let mut ok = [true; N];
    match ls.conic {
        None => {
            for l in 0..N {
                let t = -pz[l] / dz[l];
                hx[l] = t.mul_add(dx[l], px[l]);
                hy[l] = t.mul_add(dy[l], py[l]);
                tt[l] = t;
                ok[l] = dz[l] != 0.0 && hx[l].is_finite() && hy[l].is_finite();
            }
        }
        Some((c, k)) => {
            let kap = 1.0 + k;
            let (mut qx, mut qy, mut qz) = ([0.0; N], [0.0; N], [0.0; N]);
            let (mut a, mut b, mut cc) = ([0.0; N], [0.0; N], [0.0; N]);
            for l in 0..N {
                let t0 = -px[l].mul_add(dx[l], py[l].mul_add(dy[l], pz[l] * dz[l]));
                tt[l] = t0;
                qx[l] = t0.mul_add(dx[l], px[l]);
                qy[l] = t0.mul_add(dy[l], py[l]);
                qz[l] = t0.mul_add(dz[l], pz[l]);
                a[l] = c * (dx[l].mul_add(dx[l], dy[l] * dy[l]) + kap * dz[l] * dz[l]);
                b[l] = c * (qx[l].mul_add(dx[l], qy[l] * dy[l]) + kap * qz[l] * dz[l]) - dz[l];
                cc[l] =
                    c * (qx[l].mul_add(qx[l], qy[l] * qy[l]) + kap * qz[l] * qz[l]) - 2.0 * qz[l];
            }
            for l in 0..N {
                let disc = crate::numeric::discriminant(a[l], b[l], cc[l]);
                let sq = disc.sqrt();
                let qq = -(b[l] + if b[l] >= 0.0 { sq } else { -sq });
                let t1 = cc[l] / qq;
                let t2 = qq / a[l];
                let (x1, y1, z1) = (
                    t1.mul_add(dx[l], qx[l]),
                    t1.mul_add(dy[l], qy[l]),
                    t1.mul_add(dz[l], qz[l]),
                );
                let (x2, y2, z2) = (
                    t2.mul_add(dx[l], qx[l]),
                    t2.mul_add(dy[l], qy[l]),
                    t2.mul_add(dz[l], qz[l]),
                );
                let sheet1 = 1.0 - kap * c * z1 >= -SHEET_TOL;
                let sheet2 = 1.0 - kap * c * z2 >= -SHEET_TOL;
                let v1 =
                    t1.is_finite() && x1.is_finite() && y1.is_finite() && z1.is_finite() && sheet1;
                let v2 =
                    t2.is_finite() && x2.is_finite() && y2.is_finite() && z2.is_finite() && sheet2;
                // Both on the vertex sheet: the one nearer the vertex, ties
                // to t₁ (`conic_from`).
                let first = v1 && (!v2 || x1.mul_add(x1, y1 * y1) <= x2.mul_add(x2, y2 * y2));
                ts[l] = if first { t1 } else { t2 };
                hx[l] = if first { x1 } else { x2 };
                hy[l] = if first { y1 } else { y2 };
                hz[l] = if first { z1 } else { z2 };
                ok[l] = !disc.is_nan() && disc >= 0.0 && qq != 0.0 && a[l] != 0.0 && (v1 || v2);
            }
            // `conic_normal`: (−cx, −cy, 1 − κcz) normalised as `Vec3::norm`.
            for l in 0..N {
                let vx = -c * hx[l];
                let vy = -c * hy[l];
                let vz = (-(1.0 + k) * c).mul_add(hz[l], 1.0);
                let m = vx.abs().max(vy.abs()).max(vz.abs());
                let (x, y, z) = (vx / m, vy / m, vz / m);
                let n = m * (x * x + y * y + z * z).sqrt();
                nx[l] = vx / n;
                ny[l] = vy / n;
                nz[l] = vz / n;
                ok[l] &= m.is_finite() && m != 0.0 && n != 0.0;
            }
        }
    }
    let (mut hi, mut lo, mut ab) = (st.hi, st.lo, st.absorb);
    for l in 0..N {
        let n = n1[l];
        for t in [tt[l], ts[l]] {
            // `Compensated::add_product` (TwoProd, then TwoSum into hi).
            let p = n * t;
            let e = n.mul_add(t, -p);
            let s = hi[l] + p;
            let bb = s - hi[l];
            let e2 = (hi[l] - (s - bb)) + (p - bb);
            hi[l] = s;
            lo[l] += e2;
            lo[l] += e;
        }
        let absorbed = alpha[l].mul_add(tt[l] + ts[l], ab[l]);
        ab[l] = if alpha[l] != 0.0 { absorbed } else { ab[l] };
    }
    let (mut ox, mut oy, mut oz) = (dx, dy, dz);
    // `cos_incidence` of the record: |d·n| clamped, replaced by refraction's.
    let mut cosines = [0.0; N];
    for l in 0..N {
        cosines[l] = dx[l]
            .mul_add(nx[l], dy[l].mul_add(ny[l], dz[l] * nz[l]))
            .abs()
            .min(1.0);
    }
    if ls.refract {
        for l in 0..N {
            let cos = dx[l].mul_add(nx[l], dy[l].mul_add(ny[l], dz[l] * nz[l]));
            let flip = cos < 0.0;
            let (cos, mx, my, mz) = if flip {
                (-cos, -nx[l], -ny[l], -nz[l])
            } else {
                (cos, nx[l], ny[l], nz[l])
            };
            let cos = cos.min(1.0);
            let (a, b) = (n1[l], n2[l]);
            let mu = a / b;
            let sin2 = (1.0 - cos) * (1.0 + cos);
            let disc = (-mu * mu).mul_add(sin2, 1.0);
            let root = disc.sqrt();
            let denom = mu.mul_add(cos, root);
            let gamma = if denom > 0.0 {
                ((b - a) * (b + a)) / (b * b) / denom
            } else {
                root - mu * cos
            };
            ox[l] = mu.mul_add(dx[l], gamma * mx);
            oy[l] = mu.mul_add(dy[l], gamma * my);
            oz[l] = mu.mul_add(dz[l], gamma * mz);
            cosines[l] = cos;
            // `refract` rejects `disc < 0` only (a NaN passes through).
            ok[l] &= disc >= 0.0 || disc.is_nan();
        }
    }
    *incidence = ([nx, ny, nz], cosines);
    *out = Lanes {
        p: [hx, hy, hz],
        d: [ox, oy, oz],
        hi,
        lo,
        absorb: ab,
    };
    *regular = ok;
}

/// Trace `starts[k]` at wavelength `wls[k]` for every `k`, without records.
/// Bitwise identical to calling [`trace_ray`] on each ray. The rays of a
/// packet of [`PACKET`] advance surface by surface: through the
/// vectorised [`lane_step`] at planes and conics with axial transfers,
/// otherwise through [`step`] one ray after another, interleaved so the
/// core overlaps their independent dependency chains (square roots and
/// divisions dominate one ray's latency). `wls` has one entry per ray, or
/// a single entry for all rays.
pub fn trace_rays(
    sys: &ResolvedSystem,
    starts: &[Ray],
    wls: &[usize],
    opts: &TraceOptions,
) -> Vec<TraceResult> {
    trace_packets(sys, starts, wls, opts, None)
}

/// [`trace_rays`] with per-surface records: `records[k]` is cleared and
/// filled exactly as [`trace_ray`] fills its `record` for ray `k`.
pub fn trace_rays_recorded(
    sys: &ResolvedSystem,
    starts: &[Ray],
    wls: &[usize],
    opts: &TraceOptions,
    records: &mut [Vec<SurfaceRecord>],
) -> Vec<TraceResult> {
    assert_eq!(records.len(), starts.len());
    trace_packets(sys, starts, wls, opts, Some(records))
}

fn trace_packets(
    sys: &ResolvedSystem,
    starts: &[Ray],
    wls: &[usize],
    opts: &TraceOptions,
    mut records: Option<&mut [Vec<SurfaceRecord>]>,
) -> Vec<TraceResult> {
    assert!(wls.len() == starts.len() || wls.len() == 1);
    let wl_of = |k: usize| if wls.len() == 1 { wls[0] } else { wls[k] };
    let last = last_surface(sys, opts);
    let fast: Vec<Option<LaneSurface>> = (0..=last)
        .map(|i| (i >= 1).then(|| lane_surface(sys, i)).flatten())
        .collect();
    let mut out = Vec::with_capacity(starts.len());
    for (c, chunk) in starts.chunks(PACKET).enumerate() {
        let base = c * PACKET;
        let n = chunk.len();
        let mut lanes = Lanes {
            p: [[0.0; PACKET]; 3],
            d: [[0.0; PACKET]; 3],
            hi: [0.0; PACKET],
            lo: [0.0; PACKET],
            absorb: [0.0; PACKET],
        };
        let mut done: [Option<TraceResult>; PACKET] = [None; PACKET];
        let mut lane_wl = [0usize; PACKET];
        let mut alpha0 = [0.0f64; PACKET];
        let mut live = 0usize;
        for (l, start) in chunk.iter().enumerate() {
            if let Some(r) = records.as_deref_mut() {
                r[base + l].clear();
                r[base + l].push(first_record(start));
            }
            lanes.set(l, &Walk::new(*start));
            lane_wl[l] = wl_of(base + l);
            alpha0[l] = object_alpha(sys, lane_wl[l]);
            done[l] = non_finite_start(start);
            live += done[l].is_none() as usize;
        }
        let mut i = 1;
        while live > 0 && i <= last {
            let s = &sys.surfaces[i];
            if let Some(ls) = &fast[i] {
                let (mut n1, mut n2, mut alpha) = ([1.0; PACKET], [1.0; PACKET], [0.0; PACKET]);
                for l in 0..n {
                    let w = lane_wl[l];
                    n1[l] = sys.n_before(i, w);
                    n2[l] = s.n_after[w];
                    alpha[l] = if i == 1 {
                        alpha0[l]
                    } else {
                        sys.surfaces[i - 1].alpha_after[w]
                    };
                }
                let mut next = lanes;
                let mut regular = [false; PACKET];
                let mut incidence = ([[0.0; PACKET]; 3], [0.0; PACKET]);
                lane_step(
                    ls,
                    &n1,
                    &n2,
                    &alpha,
                    &lanes,
                    &mut next,
                    &mut regular,
                    &mut incidence,
                );
                for l in 0..n {
                    if done[l].is_some() {
                        continue;
                    }
                    if regular[l] && (!opts.clip || s.passes_aperture(next.p[0][l], next.p[1][l])) {
                        let w = next.walk(l);
                        if let Some(r) = records.as_deref_mut() {
                            let ([nx, ny, nz], cos) = &incidence;
                            r[base + l].push(SurfaceRecord {
                                point: w.ray.pos,
                                dir: w.ray.dir,
                                normal: Vec3::new(nx[l], ny[l], nz[l]),
                                cos_incidence: cos[l],
                                opl: w.opl.value(),
                                transmittance: (-w.absorb).exp(),
                            });
                        }
                        lanes.set(l, &w);
                        continue;
                    }
                    let mut w = lanes.walk(l);
                    let rec = records.as_deref_mut().map(|r| &mut r[base + l]);
                    match step(sys, i, lane_wl[l], alpha0[l], opts, &mut w, rec) {
                        Ok(()) => lanes.set(l, &w),
                        Err(status) => {
                            done[l] = Some(w.result(status, i));
                            live -= 1;
                        }
                    }
                }
            } else {
                for l in 0..n {
                    if done[l].is_none() {
                        let mut w = lanes.walk(l);
                        let rec = records.as_deref_mut().map(|r| &mut r[base + l]);
                        match step(sys, i, lane_wl[l], alpha0[l], opts, &mut w, rec) {
                            Ok(()) => lanes.set(l, &w),
                            Err(status) => {
                                done[l] = Some(w.result(status, i));
                                live -= 1;
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        out.extend(
            (0..n).map(|l| done[l].unwrap_or_else(|| lanes.walk(l).result(TraceStatus::Ok, last))),
        );
    }
    out
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
