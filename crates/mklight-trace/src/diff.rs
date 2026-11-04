//! Analytic differential ray tracing (forward mode): the derivative of a
//! traced ray with respect to two launch parameters, propagated surface by
//! surface along the per-surface records of a trace.
//!
//! A trace is a composition of per-surface maps, each differentiable away
//! from a few singular configurations. Given a trace's [`SurfaceRecord`]s
//! (hit point, unit normal and outgoing direction at every surface), the
//! tangent `(∂p/∂a, ∂d/∂a)` of the ray with respect to parameters `a ∈ ℝ²`
//! is carried through the same steps, so nothing is differenced:
//!
//! * **transfer** between frames is linear: the rotation part of the relative
//!   transform applies to every tangent vector ([`transfer_vector`]);
//! * **intersection** `F(p + t d) = 0`: by the implicit function theorem,
//!   with `v = δp + t δd` and the unit normal `N ∝ ∇F`,
//!   `δh = v − d (N·v)/(N·d)` for every surface shape, whether the hit was
//!   found in closed form or by Newton iteration (the converged root is
//!   differentiated, not the iteration);
//! * **normal** `N = w/|w|`: `δN = (δw − N (N·δw))/|w|`, with `w` the
//!   unnormalised gradient: `(−cx, −cy, 1 − κcz)` for a conic (no second
//!   derivative needed) and `(−z_x, −z_y, 1)` with the sag Hessian
//!   (`surface_math::evaluate`, or the closed form for even aspheres) for
//!   every other sag surface;
//! * **refraction** `d' = μ d + γ N`, `γ = √(1 − μ²(1 − cos²i)) − μ cos i`:
//!   `δcos i = δd·N + d·δN`, `δγ = δcos i (μ² cos i/√… − μ)`,
//!   `δd' = μ δd + δγ N + γ δN`; **reflection**
//!   `δd' = δd − 2((δd·N + d·δN) N + (d·N) δN)`; **diffraction** and the
//!   **ideal lens** are differentiated from their closed forms in the same
//!   way (phase Hessian from `surface_math::phase`).
//!
//! The derivative is **not formed** where the map is ill-conditioned; the
//! ray itself is fine and callers fall back to finite differences
//! ([`TangentError`]): grazing incidence `|N·d| < 10⁻⁹`, near total internal
//! reflection or an evanescent order (`√… < 10⁻⁶`, where `δγ` diverges), an
//! undefined sag Hessian (the vertex of a cone, a vertical rim) and a
//! degenerate ideal-lens ray. Clipping, misses and TIR stop the ray before
//! any tangent is needed, so they need no special handling here.
//!
//! Uses: the Jacobian of the stop hit with respect to the aim point in real
//! ray aiming ([`crate::tracer`]), and the thin pencil around a chief ray
//! (astigmatic foci) in place of close rays.

use mklight_core::{Interaction, ResolvedSystem, SurfaceShape, Vec3};

use crate::numeric::Mat2;
use crate::trace::{
    Ray, SurfaceRecord, TraceOptions, TraceResult, trace_ray, transfer_into, transfer_vector,
};

/// `|N·d|` below which the intersection map is treated as ill-conditioned.
pub const GRAZING_COS: f64 = 1e-9;
/// `√(1 − μ² sin²i)` (or the diffraction root) below which refraction is
/// treated as ill-conditioned (near TIR / evanescence).
pub const NEAR_TIR_ROOT: f64 = 1e-6;

/// Derivative of a ray's point and direction with respect to two parameters
/// (one vector per parameter), in the ray's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTangent {
    pub dpos: [Vec3; 2],
    pub ddir: [Vec3; 2],
}

impl RayTangent {
    pub const ZERO: RayTangent = RayTangent {
        dpos: [Vec3::ZERO; 2],
        ddir: [Vec3::ZERO; 2],
    };

    /// Jacobian of the transverse point `(x, y)` with respect to the
    /// parameters: `[[∂x/∂a₀, ∂x/∂a₁], [∂y/∂a₀, ∂y/∂a₁]]`.
    #[inline]
    pub fn xy_jacobian(&self) -> Mat2 {
        [
            [self.dpos[0].x, self.dpos[1].x],
            [self.dpos[0].y, self.dpos[1].y],
        ]
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.dpos.iter().chain(&self.ddir).all(|v| v.is_finite())
    }
}

/// Why a tangent could not be formed (module docs). The traced ray is valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentError {
    /// `|N·d| < GRAZING_COS` at the surface.
    Grazing { surface: usize },
    /// Refraction or diffraction root `< NEAR_TIR_ROOT` at the surface.
    NearTir { surface: usize },
    /// The surface's curvature (Hessian) or an ideal-lens map is undefined
    /// at the hit, or the tangent became non-finite.
    Undefined { surface: usize },
    /// The records do not reach the requested surface (the ray failed).
    Incomplete,
}

/// `(δw − N (N·δw))/|w|`: the derivative of `w/|w|` at `N = w/|w|`.
#[inline]
fn normalize_tangent(n: Vec3, w_norm: f64, dw: Vec3) -> Vec3 {
    (dw - n * n.dot(dw)) / w_norm
}

/// Sag Hessian `[z_xx, z_xy, z_yy]` of an even asphere at `(x, y)`, from
/// `s(ρ) = cρ/(1 + w) + P(ρ)`, `ρ = r²`, `w = √(1 − κc²ρ)`:
/// `s' = c/(2w) + P'`, `s'' = κc³/(4w³) + P''`, `z_xx = 2s' + 4x²s''`,
/// `z_xy = 4xys''`, `z_yy = 2s' + 4y²s''`.
fn even_asphere_hessian(c: f64, k: f64, coefficients: &[f64], x: f64, y: f64) -> Option<[f64; 3]> {
    let kap = 1.0 + k;
    let r2 = x.mul_add(x, y * y);
    let arg = (-kap * c * c).mul_add(r2, 1.0);
    if arg.is_nan() || arg <= 0.0 {
        return None;
    }
    let w = arg.sqrt();
    let (mut dp, mut ddp) = (0.0f64, 0.0f64);
    for (i, a) in coefficients.iter().enumerate().rev() {
        let m = i as f64 + 2.0;
        dp = dp.mul_add(r2, a * m);
        ddp = ddp.mul_add(r2, a * m * (m - 1.0));
    }
    let s1 = (0.5 * c / w) + dp * r2;
    let s2 = kap * c * c * c / (4.0 * w * w * w) + ddp;
    Some([
        (4.0 * x * x).mul_add(s2, 2.0 * s1),
        4.0 * x * y * s2,
        (4.0 * y * y).mul_add(s2, 2.0 * s1),
    ])
}

/// Derivative of the unit normal of `shape` at hit `h` (unit normal `n`)
/// for the hit displacements `dh`. `None` where the curvature is undefined.
fn normal_tangent(shape: &SurfaceShape, h: Vec3, n: Vec3, dh: &[Vec3; 2]) -> Option<[Vec3; 2]> {
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
            Some([Vec3::ZERO; 2])
        }
        SurfaceShape::Conic { curvature, conic } => {
            let (c, kap) = (*curvature, 1.0 + conic);
            let w = Vec3::new(-c * h.x, -c * h.y, (-kap * c).mul_add(h.z, 1.0));
            let wn = w.norm();
            if wn.is_nan() || wn <= 0.0 || !wn.is_finite() {
                return None;
            }
            let dw = |d: Vec3| Vec3::new(-c * d.x, -c * d.y, -kap * c * d.z);
            Some([
                normalize_tangent(n, wn, dw(dh[0])),
                normalize_tangent(n, wn, dw(dh[1])),
            ])
        }
        SurfaceShape::Diffractive { base, .. }
        | SurfaceShape::SagOverlay {
            base,
            astigmatism: 0.0,
            spherical: 0.0,
            ..
        } => normal_tangent(base, h, n, dh),
        _ => {
            let ([zx, zy], hess) = match shape {
                SurfaceShape::EvenAsphere {
                    curvature,
                    conic,
                    coefficients,
                } => {
                    let hess = even_asphere_hessian(*curvature, *conic, coefficients, h.x, h.y)?;
                    let r2 = h.x.mul_add(h.x, h.y * h.y);
                    let (_, slope) = crate::surface::sag_and_slope(shape, r2)?;
                    ([2.0 * h.x * slope, 2.0 * h.y * slope], hess)
                }
                _ => {
                    let v = mklight_core::surface_math::evaluate(shape, h.x, h.y)?;
                    (v.gradient, v.hessian)
                }
            };
            if !(zx.is_finite() && zy.is_finite() && hess.iter().all(|v| v.is_finite())) {
                return None;
            }
            let w = Vec3::new(-zx, -zy, 1.0);
            let wn = w.norm();
            let dw = |d: Vec3| {
                Vec3::new(
                    -(hess[0] * d.x + hess[1] * d.y),
                    -(hess[1] * d.x + hess[2] * d.y),
                    0.0,
                )
            };
            Some([
                normalize_tangent(n, wn, dw(dh[0])),
                normalize_tangent(n, wn, dw(dh[1])),
            ])
        }
    }
}

/// `n₂ k₂∥ = n₁ k₁∥ + g∥` ([`crate::interact::diffract`]) differentiated.
#[allow(clippy::too_many_arguments)]
fn diffract_tangent(
    k: Vec3,
    n: Vec3,
    n1: f64,
    n2: f64,
    g: Vec3,
    reflected: bool,
    dk: &[Vec3; 2],
    dn: &[Vec3; 2],
    dg: &[Vec3; 2],
) -> Option<[Vec3; 2]> {
    let kn = k.dot(n);
    let gn = g.dot(n);
    let tangent = ((k - n * kn) * n1 + g - n * gn) * (1.0 / n2);
    let disc = 1.0 - tangent.dot(tangent);
    if disc.is_nan() || disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    if root < NEAR_TIR_ROOT {
        return None;
    }
    let sign = if (kn < 0.0) != reflected { -1.0 } else { 1.0 };
    let one = |dk: Vec3, dn: Vec3, dg: Vec3| {
        let dkn = dk.dot(n) + k.dot(dn);
        let dgn = dg.dot(n) + g.dot(dn);
        let dt = ((dk - dn * kn - n * dkn) * n1 + dg - dn * gn - n * dgn) * (1.0 / n2);
        let ddisc = -2.0 * tangent.dot(dt);
        dt + dn * (sign * root) + n * (sign * ddisc / (2.0 * root))
    };
    Some([one(dk[0], dn[0], dg[0]), one(dk[1], dn[1], dg[1])])
}

/// [`crate::interact::ideal_lens`]'s direction map differentiated:
/// `τ = (k_x, k_y)/|k_z|`, `τ' = r τ − (x, y)/f`, `d' = (τ', sgn k_z)/|·|`.
fn ideal_lens_tangent(
    h: Vec3,
    k: Vec3,
    f: f64,
    r: f64,
    dh: &[Vec3; 2],
    dk: &[Vec3; 2],
) -> Option<[Vec3; 2]> {
    if k.z == 0.0 || f == 0.0 || !f.is_finite() {
        return None;
    }
    let sz = k.z.signum();
    let az = k.z.abs();
    let t = Vec3::new(k.x / az, k.y / az, sz);
    let m = Vec3::new(r.mul_add(t.x, -h.x / f), r.mul_add(t.y, -h.y / f), sz);
    let mn = m.norm();
    let out = m / mn;
    let one = |dh: Vec3, dk: Vec3| {
        let daz = sz * dk.z;
        let dt = Vec3::new((dk.x - t.x * daz) / az, (dk.y - t.y * daz) / az, 0.0);
        let dm = Vec3::new(r * dt.x - dh.x / f, r * dt.y - dh.y / f, 0.0);
        normalize_tangent(out, mn, dm)
    };
    Some([one(dh[0], dk[0]), one(dh[1], dk[1])])
}

/// Carry a tangent through surface `i` (module docs): `prev` and `rec` are
/// the records at `i − 1` and `i`, `(dp, dd)` the tangent in frame `i − 1`.
/// Returns the tangent of the hit point and the outgoing direction in
/// frame `i`. This is the general routine for every shape, transform and
/// interaction; [`TangentPlan`] runs a cheaper equivalent at axial planes
/// and conics.
fn surface_tangent_general(
    sys: &ResolvedSystem,
    i: usize,
    wl: usize,
    prev: &SurfaceRecord,
    rec: &SurfaceRecord,
    dp: [Vec3; 2],
    dd: [Vec3; 2],
) -> Result<([Vec3; 2], [Vec3; 2]), TangentError> {
    let image = sys.surfaces.len() - 1;
    let s = &sys.surfaces[i];
    let dp = dp.map(|v| transfer_vector(sys, i, v));
    let dd = dd.map(|v| transfer_vector(sys, i, v));
    if matches!(s.shape, SurfaceShape::CoordinateBreak) {
        return Ok((dp, dd));
    }
    let inc = transfer_into(sys, i, Ray::new(prev.point, prev.dir));
    let (p, k) = (inc.pos, inc.dir);
    let (h, n) = (rec.point, rec.normal);
    // Intersection: the implicit function theorem on F(p + t d) = 0.
    let nk = n.dot(k);
    if nk.is_nan() || nk.abs() < GRAZING_COS {
        return Err(TangentError::Grazing { surface: i });
    }
    let t = (h - p).dot(k);
    let dh = std::array::from_fn(|j| {
        let v = dp[j] + dd[j] * t;
        v - k * (n.dot(v) / nk)
    });
    let undefined = TangentError::Undefined { surface: i };
    if i == image || s.interaction == Interaction::Dummy {
        return Ok((dh, dd));
    }
    let dn = normal_tangent(&s.shape, h, n, &dh).ok_or(undefined)?;
    let n1 = sys.n_before(i, wl);
    let dk = match (&s.shape, s.interaction) {
        (SurfaceShape::Diffractive { phase, .. }, inter) if phase.trace_order != 0 => {
            let v = mklight_core::surface_math::phase(phase, h.x, h.y).ok_or(undefined)?;
            let scale =
                phase.trace_order as f64 * sys.wavelengths_um[wl] * 1e-3 / std::f64::consts::TAU;
            let reflected = inter == Interaction::Reflect;
            let n2 = if reflected { n1 } else { s.n_after[wl] };
            let g = Vec3::new(v.gradient[0] * scale, v.gradient[1] * scale, 0.0);
            let hs = v.hessian;
            let dg = dh.map(|d| {
                Vec3::new(
                    (hs[0] * d.x + hs[1] * d.y) * scale,
                    (hs[1] * d.x + hs[2] * d.y) * scale,
                    0.0,
                )
            });
            if !(g.is_finite() && dg.iter().all(|v| v.is_finite())) {
                return Err(undefined);
            }
            diffract_tangent(k, n, n1, n2, g, reflected, &dd, &dn, &dg)
                .ok_or(TangentError::NearTir { surface: i })?
        }
        (SurfaceShape::Paraxial { focal_length }, inter) => {
            let (k, dd, n2) = if inter == Interaction::Reflect {
                (
                    Vec3::new(k.x, k.y, -k.z),
                    dd.map(|d| Vec3::new(d.x, d.y, -d.z)),
                    n1,
                )
            } else {
                (k, dd, s.n_after[wl])
            };
            ideal_lens_tangent(h, k, *focal_length, n1 / n2, &dh, &dd).ok_or(undefined)?
        }
        (_, Interaction::Refract) => {
            let mut cos = k.dot(n);
            let (n, dn) = if cos < 0.0 {
                cos = -cos;
                (-n, dn.map(|v| -v))
            } else {
                (n, dn)
            };
            let clamped = cos > 1.0;
            let cos = cos.min(1.0);
            let n2 = s.n_after[wl];
            let mu = n1 / n2;
            let sin2 = (1.0 - cos) * (1.0 + cos);
            let disc = (-mu * mu).mul_add(sin2, 1.0);
            if disc.is_nan() || disc < 0.0 {
                return Err(undefined);
            }
            let root = disc.sqrt();
            if root < NEAR_TIR_ROOT {
                return Err(TangentError::NearTir { surface: i });
            }
            let denom = mu.mul_add(cos, root);
            let gamma = if denom > 0.0 {
                ((n2 - n1) * (n2 + n1)) / (n2 * n2) / denom
            } else {
                root - mu * cos
            };
            let dgamma_dcos = mu * (mu * cos / root - 1.0);
            std::array::from_fn(|j| {
                let dcos = if clamped {
                    0.0
                } else {
                    dd[j].dot(n) + k.dot(dn[j])
                };
                dd[j] * mu + n * (dgamma_dcos * dcos) + dn[j] * gamma
            })
        }
        (_, Interaction::Reflect) => {
            let kn = k.dot(n);
            std::array::from_fn(|j| {
                let dkn = dd[j].dot(n) + k.dot(dn[j]);
                dd[j] - (n * dkn + dn[j] * kn) * 2.0
            })
        }
        (_, Interaction::Dummy) => dd,
    };
    Ok((dh, dk))
}

/// Geometry of a surface as [`TangentPlan`] sees it.
#[derive(Clone, Copy, Debug)]
enum Geometry {
    /// Plane or ideal lens: the normal is constant.
    Flat,
    /// Conic with `c ≠ 0`: `w = (−cx, −cy, 1 − κcz)`.
    Conic { c: f64, kap: f64 },
    /// Everything else (general sag, coordinate break): the general routine.
    General,
}

/// Interaction of a surface as [`TangentPlan`] sees it.
#[derive(Clone, Copy, Debug)]
enum Act {
    /// Dummy, or the image surface: the direction passes through.
    Pass,
    Refract,
    Reflect,
    /// Diffraction, ideal lens: the general routine.
    General,
}

#[derive(Clone, Copy, Debug)]
struct Desc {
    /// Axial transfer: `z` shift (0 for none). `None` for a rotation.
    shift: Option<f64>,
    geometry: Geometry,
    act: Act,
}

/// Per-surface data of one system for [`TangentPlan::differentiate`]: the
/// transfer, geometry and interaction class of every surface and the index
/// ratio `μ = n₁/n₂` per (surface, wavelength), so the per-ray pass does no
/// shape dispatch and no square roots. Built once per tracer.
#[derive(Clone, Debug)]
pub struct TangentPlan {
    surfaces: Vec<Desc>,
    mu: Vec<f64>,
    nwl: usize,
}

impl TangentPlan {
    pub fn new(sys: &ResolvedSystem) -> Self {
        let image = sys.surfaces.len() - 1;
        let nwl = sys.wavelengths_um.len();
        let mut mu = vec![1.0; sys.surfaces.len() * nwl];
        let surfaces = (0..sys.surfaces.len())
            .map(|i| {
                let s = &sys.surfaces[i];
                if i == 0 {
                    return Desc {
                        shift: None,
                        geometry: Geometry::General,
                        act: Act::General,
                    };
                }
                let shift = if i == 1 && sys.object_at_infinity {
                    let f = &sys.surfaces[1].frame;
                    (f.rot.is_identity() && f.origin == Vec3::ZERO).then_some(0.0)
                } else {
                    let prev = &sys.surfaces[i - 1];
                    if prev.to_next_is_axial {
                        Some(prev.thickness)
                    } else {
                        prev.to_next.is_none().then_some(0.0)
                    }
                };
                let geometry = match s.shape {
                    SurfaceShape::Plane | SurfaceShape::Paraxial { .. } => Geometry::Flat,
                    SurfaceShape::Conic { curvature, conic } if curvature != 0.0 => {
                        Geometry::Conic {
                            c: curvature,
                            kap: 1.0 + conic,
                        }
                    }
                    SurfaceShape::Conic { .. } => Geometry::Flat,
                    _ => Geometry::General,
                };
                let act = if i == image {
                    Act::Pass
                } else {
                    match (&s.shape, s.interaction) {
                        (_, Interaction::Dummy) => Act::Pass,
                        (SurfaceShape::Paraxial { .. }, _) => Act::General,
                        (SurfaceShape::Diffractive { phase, .. }, _) if phase.trace_order != 0 => {
                            Act::General
                        }
                        (_, Interaction::Refract) => {
                            for w in 0..nwl {
                                mu[i * nwl + w] = sys.n_before(i, w) / s.n_after[w];
                            }
                            Act::Refract
                        }
                        (_, Interaction::Reflect) => Act::Reflect,
                    }
                };
                Desc {
                    shift,
                    geometry,
                    act,
                }
            })
            .collect();
        TangentPlan { surfaces, mu, nwl }
    }

    /// [`surface_tangent_general`] without shape dispatch, square roots or
    /// frame transforms at an axial plane or conic (`c ≠ 0`) that passes,
    /// refracts or reflects; the general routine otherwise.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn surface_tangent(
        &self,
        sys: &ResolvedSystem,
        i: usize,
        wl: usize,
        prev: &SurfaceRecord,
        rec: &SurfaceRecord,
        dp: [Vec3; 2],
        dd: [Vec3; 2],
    ) -> Result<([Vec3; 2], [Vec3; 2]), TangentError> {
        let d = &self.surfaces[i];
        let (
            Some(shift),
            Geometry::Flat | Geometry::Conic { .. },
            Act::Pass | Act::Refract | Act::Reflect,
        ) = (d.shift, d.geometry, d.act)
        else {
            return surface_tangent_general(sys, i, wl, prev, rec, dp, dd);
        };
        let (k, h, n) = (prev.dir, rec.point, rec.normal);
        let nk = n.dot(k);
        if nk.is_nan() || nk.abs() < GRAZING_COS {
            return Err(TangentError::Grazing { surface: i });
        }
        // `t = (h − p)·k` with `p = prev.point − (0, 0, shift)`.
        let t = (h.x - prev.point.x) * k.x
            + (h.y - prev.point.y) * k.y
            + (h.z - prev.point.z + shift) * k.z;
        let inv_nk = 1.0 / nk;
        let dh: [Vec3; 2] = std::array::from_fn(|j| {
            let v = dp[j] + dd[j] * t;
            v - k * (n.dot(v) * inv_nk)
        });
        if matches!(d.act, Act::Pass) {
            return Ok((dh, dd));
        }
        let dn: [Vec3; 2] = match d.geometry {
            Geometry::Flat => [Vec3::ZERO; 2],
            Geometry::Conic { c, kap } => {
                // |w| = w_z / n_z, unless the normal is nearly transverse.
                let wz = (-kap * c).mul_add(h.z, 1.0);
                let inv_w = if n.z.abs() > 1e-3 {
                    n.z / wz
                } else {
                    1.0 / Vec3::new(-c * h.x, -c * h.y, wz).norm()
                };
                if !inv_w.is_finite() {
                    return Err(TangentError::Undefined { surface: i });
                }
                std::array::from_fn(|j| {
                    let dw = Vec3::new(-c * dh[j].x, -c * dh[j].y, -kap * c * dh[j].z);
                    (dw - n * n.dot(dw)) * inv_w
                })
            }
            Geometry::General => unreachable!("general geometry takes the general routine"),
        };
        let dk = match d.act {
            Act::Refract => {
                let mu = self.mu[i * self.nwl + wl];
                let (cos, nf, dnf) = if nk < 0.0 {
                    (-nk, -n, dn.map(|v| -v))
                } else {
                    (nk, n, dn)
                };
                let clamped = cos > 1.0;
                let cos = cos.min(1.0);
                // `√(1 − μ² sin²i) = k'·N` for the outgoing `k' = μk + γN`.
                let root = rec.dir.dot(nf);
                if root.is_nan() || root < NEAR_TIR_ROOT {
                    return Err(TangentError::NearTir { surface: i });
                }
                let gamma = root - mu * cos;
                let dgamma_dcos = mu * (mu * cos - root) / root;
                std::array::from_fn(|j| {
                    let dcos = if clamped {
                        0.0
                    } else {
                        dd[j].dot(nf) + k.dot(dnf[j])
                    };
                    dd[j] * mu + nf * (dgamma_dcos * dcos) + dnf[j] * gamma
                })
            }
            Act::Reflect => std::array::from_fn(|j| {
                let dkn = dd[j].dot(n) + k.dot(dn[j]);
                dd[j] - (n * dkn + dn[j] * nk) * 2.0
            }),
            Act::Pass | Act::General => unreachable!("handled above"),
        };
        Ok((dh, dk))
    }

    /// Carry the tangent `start` of a traced ray (`records[0]` is the input
    /// ray in the starting frame, `records[i]` surface `i`) to surface
    /// `last` inclusive: the derivative of the hit point and of the outgoing
    /// direction there, in surface `last`'s frame.
    pub fn differentiate(
        &self,
        sys: &ResolvedSystem,
        wl: usize,
        start: &RayTangent,
        records: &[SurfaceRecord],
        last: usize,
    ) -> Result<RayTangent, TangentError> {
        let mut t = *start;
        self.differentiate_each(sys, wl, start, records, last, |_, tangent| t = *tangent)?;
        Ok(t)
    }

    /// [`Self::differentiate`] visiting the tangent at every surface
    /// `1..=last` on the way (`visit(surface, tangent)`), for callers that
    /// need the whole chain (the propagated-error conditioning of the aim
    /// stop) without differentiating once per surface.
    pub fn differentiate_each(
        &self,
        sys: &ResolvedSystem,
        wl: usize,
        start: &RayTangent,
        records: &[SurfaceRecord],
        last: usize,
        mut visit: impl FnMut(usize, &RayTangent),
    ) -> Result<(), TangentError> {
        if records.len() <= last || self.surfaces.len() != sys.surfaces.len() {
            return Err(TangentError::Incomplete);
        }
        let mut t = *start;
        for i in 1..=last {
            let (dh, dk) =
                self.surface_tangent(sys, i, wl, &records[i - 1], &records[i], t.dpos, t.ddir)?;
            t = RayTangent { dpos: dh, ddir: dk };
            if !t.is_finite() {
                return Err(TangentError::Undefined { surface: i });
            }
            visit(i, &t);
        }
        Ok(())
    }
}

/// [`TangentPlan::differentiate`] with a plan built for the call (tests and
/// one-off uses; the tracer keeps a plan).
pub fn differentiate(
    sys: &ResolvedSystem,
    wl: usize,
    start: &RayTangent,
    records: &[SurfaceRecord],
    last: usize,
) -> Result<RayTangent, TangentError> {
    TangentPlan::new(sys).differentiate(sys, wl, start, records, last)
}

/// [`trace_ray`] with records into `scratch`, then [`differentiate`] to the
/// trace's last surface. The trace result is bitwise that of `trace_ray`.
pub fn trace_ray_tangent(
    sys: &ResolvedSystem,
    start: &Ray,
    tangent: &RayTangent,
    wl: usize,
    opts: &TraceOptions,
    scratch: &mut Vec<SurfaceRecord>,
) -> (TraceResult, Result<RayTangent, TangentError>) {
    let r = trace_ray(sys, start, wl, opts, Some(scratch));
    let t = if r.status.is_ok() {
        differentiate(sys, wl, tangent, scratch, r.surface)
    } else {
        Err(TangentError::Incomplete)
    };
    (r, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::{ResolvedSystem, Snapshot, samples};
    use std::sync::Arc;

    fn resolved(sys: mklight_core::OpticalSystem) -> ResolvedSystem {
        let snap = Snapshot::new(sys).unwrap();
        ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap()
    }

    /// Start ray `a ↦ (a_x, a_y, z₀) − d (a·d)` with direction `d`, its
    /// tangent, and central differences of the full trace for the sweep.
    fn sweep(sys: &ResolvedSystem, dir: Vec3, z0: f64, wl: usize) -> (f64, f64) {
        let opts = TraceOptions::unclipped();
        let start = |a: [f64; 2]| {
            let ap = Vec3::new(a[0], a[1], z0);
            Ray::new(ap - dir * ap.dot(dir), dir)
        };
        let a0 = [0.7, -1.9];
        let tangent = RayTangent {
            dpos: [Vec3::X - dir * dir.x, Vec3::Y - dir * dir.y],
            ddir: [Vec3::ZERO; 2],
        };
        let mut scratch = Vec::new();
        let (r, t) = trace_ray_tangent(sys, &start(a0), &tangent, wl, &opts, &mut scratch);
        assert!(r.status.is_ok(), "{:?}", r.status);
        let t = t.unwrap();
        // Error plateau of central differences: the best step is the
        // minimum over a sweep, which must agree with the analytic value.
        let mut best = f64::INFINITY;
        let mut at_1e4 = f64::INFINITY;
        for e in 2..=9 {
            let h = 10f64.powi(-e);
            let mut err = 0.0f64;
            for j in 0..2 {
                let (mut ap, mut am) = (a0, a0);
                ap[j] += h;
                am[j] -= h;
                let rp = trace_ray(sys, &start(ap), wl, &opts, None);
                let rm = trace_ray(sys, &start(am), wl, &opts, None);
                assert!(rp.status.is_ok() && rm.status.is_ok());
                let fp = (rp.ray.pos - rm.ray.pos) / (2.0 * h);
                let fd = (rp.ray.dir - rm.ray.dir) / (2.0 * h);
                err = err
                    .max(fp.max_abs_diff(t.dpos[j]))
                    .max(fd.max_abs_diff(t.ddir[j]));
            }
            best = best.min(err);
            if e == 4 {
                at_1e4 = err;
            }
        }
        (best, at_1e4)
    }

    #[test]
    fn tangent_matches_central_differences_on_every_sample() {
        for (name, sys) in samples::all() {
            if !sys.surfaces.is_empty()
                && !matches!(sys.surfaces[0].thickness, t if t.is_infinite())
            {
                continue; // finite objects are covered through the tracer
            }
            let sys = resolved(sys);
            let dir = Vec3::new(0.01, -0.03, 1.0).normalized();
            let (best, at_1e4) = sweep(&sys, dir, 0.0, sys.primary_wavelength);
            assert!(
                best < 1e-7 && at_1e4 < 1e-5,
                "{name}: best {best:e}, at 1e-4 {at_1e4:e}"
            );
        }
    }

    /// The plan's fast path (axial planes and conics) agrees with the
    /// general routine to rounding on every sample.
    #[test]
    fn plan_fast_path_matches_general_routine() {
        for (name, sys) in samples::all() {
            let sys = resolved(sys);
            let plan = TangentPlan::new(&sys);
            let wl = sys.primary_wavelength;
            let dir = Vec3::new(0.02, -0.05, 1.0).normalized();
            let start = if sys.object_at_infinity {
                let ap = Vec3::new(1.3, -2.1, 0.0);
                Ray::new(ap - dir * ap.dot(dir), dir)
            } else {
                Ray::new(Vec3::new(0.3, -0.4, 0.0), dir)
            };
            let tangent = RayTangent {
                dpos: [Vec3::X, Vec3::Y],
                ddir: [Vec3::new(0.0, 0.01, 0.0), Vec3::new(-0.01, 0.0, 0.0)],
            };
            let mut rec = Vec::new();
            let r = trace_ray(&sys, &start, wl, &TraceOptions::unclipped(), Some(&mut rec));
            if !r.status.is_ok() {
                continue;
            }
            let (mut a, mut b) = (tangent, tangent);
            for i in 1..=r.surface {
                let fa = plan.surface_tangent(&sys, i, wl, &rec[i - 1], &rec[i], a.dpos, a.ddir);
                let fb = surface_tangent_general(&sys, i, wl, &rec[i - 1], &rec[i], b.dpos, b.ddir);
                match (fa, fb) {
                    (Ok((ha, ka)), Ok((hb, kb))) => {
                        for j in 0..2 {
                            let scale = 1.0 + ha[j].norm().max(ka[j].norm());
                            assert!(
                                ha[j].max_abs_diff(hb[j]) < 1e-11 * scale
                                    && ka[j].max_abs_diff(kb[j]) < 1e-11 * scale,
                                "{name} surface {i}: {ha:?} {ka:?} vs {hb:?} {kb:?}"
                            );
                        }
                        a = RayTangent { dpos: ha, ddir: ka };
                        b = RayTangent { dpos: hb, ddir: kb };
                    }
                    (Err(ea), Err(eb)) => {
                        assert_eq!(ea, eb, "{name} surface {i}");
                        break;
                    }
                    (fa, fb) => panic!("{name} surface {i}: {fa:?} vs {fb:?}"),
                }
            }
        }
    }

    #[test]
    fn tangent_is_not_formed_at_grazing_incidence() {
        let sys = resolved(samples::singlet());
        // A ray along the first surface's tangent plane at the vertex.
        let start = Ray::new(Vec3::new(0.0, -1.0, -0.0), Vec3::Y);
        let mut scratch = Vec::new();
        let (r, t) = trace_ray_tangent(
            &sys,
            &start,
            &RayTangent::ZERO,
            0,
            &TraceOptions::unclipped(),
            &mut scratch,
        );
        assert!(!r.status.is_ok() || matches!(t, Err(TangentError::Grazing { .. })));
    }

    #[test]
    fn even_asphere_hessian_matches_differences() {
        let (c, k, a) = (1.0 / 30.0, -0.6, [2e-5, -3e-8, 1e-11]);
        let shape = SurfaceShape::EvenAsphere {
            curvature: c,
            conic: k,
            coefficients: a.to_vec(),
        };
        let (x, y) = (3.0, -5.0);
        let h = even_asphere_hessian(c, k, &a, x, y).unwrap();
        let v = mklight_core::surface_math::evaluate(&shape, x, y).unwrap();
        for j in 0..3 {
            assert!(
                (h[j] - v.hessian[j]).abs() < 1e-12,
                "{h:?} vs {:?}",
                v.hessian
            );
        }
    }
}
