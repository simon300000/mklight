//! Ray–surface intersection and surface normals in the surface's **local**
//! frame (vertex at the origin, sag `z = f(x, y)`, CONVENTIONS §2).
//!
//! The ray is `P + t·D` with `|D| = 1`. Distances `t` are signed: a sequential
//! trace may propagate backwards (virtual segments), so a negative `t` is
//! legal and is accumulated into the optical path with its sign.
//!
//! # Conic (`c`, `k`)
//!
//! Implicit form `F = c(x² + y² + (1+k) z²) − 2z = 0`, with the vertex sheet
//! `z = c r² / (1 + √(1 − (1+k)c²r²))`.
//!
//! 1. **Transfer to the point of closest approach to the vertex**:
//!    `t₀ = −P·D` and `Q = P + t₀ D` (so `Q·D = 0` and `|Q| ≤ |P|` for every
//!    direction). This keeps the quadratic's coefficients small
//!    (CONVENTIONS §7.2): for near-axial rays `Q` is close to the vertex
//!    tangent plane, and for grazing rays it stays bounded, whereas a transfer
//!    to the tangent plane (`t₀ = −P_z/D_z`) would move the ray up to
//!    `|P_z/D_z|` away and lose all accuracy as `D_z → 0`.
//! 2. With `κ = 1 + k`, substituting `Q + t D` gives `A t² + 2B t + C = 0`:
//!    `A = c(D_x² + D_y² + κ D_z²)`, `B = c(Q·D)_κ − D_z`,
//!    `C = c(Q_x² + Q_y² + κ Q_z²) − 2 Q_z`, where `(Q·D)_κ = Q_xD_x + Q_yD_y + κQ_zD_z`.
//! 3. The discriminant `Δ = B² − AC` is formed with Kahan's fma trick
//!    ([`crate::numeric::discriminant`]); `Δ < 0` is a miss.
//! 4. **Stable roots**: `q = −(B + sgn(B)·√Δ)` (no cancellation, since both
//!    terms have the same sign), then `t₁ = C / q` and `t₂ = q / A`. `t₁` is
//!    the root that tends to the plane solution `−C/(2B)` as `c → 0`; it stays
//!    finite when `A → 0` (paraboloid on axis, flat surface) where the
//!    textbook `(−B ± √Δ)/A` form would divide `0/0`.
//! 5. **Sheet selection**: a point lies on the vertex sheet iff
//!    `1 − κ c z ≥ 0` (on that sheet `1 − κcz = √(1 − κc²r²)`). Roots on the
//!    other sheet of a sphere, ellipsoid or hyperboloid are rejected. If both
//!    roots lie on the vertex sheet (a ray crossing a deep surface twice),
//!    the one nearer the vertex (smaller `r²`) is chosen; ties go to `t₁`.
//!    No root on the vertex sheet is a miss.
//!
//! The unit normal is `∇F`-based: `N ∝ (−c x, −c y, 1 − κ c z)`, which needs
//! no square root of the sag argument and stays finite at the rim. It points
//! towards local `+z` on the vertex sheet.
//!
//! # Even asphere
//!
//! The base conic intersection from `Q` (the plane `z = 0` when `c = 0`)
//! seeds Newton's method on the sag residual `g(t) = z(t) − sag(r²(t))` with
//! `g'(t) = D_z − 2 sag'(r²)(x D_x + y D_y)`. Iteration stops when
//! `|g| ≤ tol` (then one last Newton correction is applied, which squares the
//! error); otherwise it gives up after `max_iter` Newton steps.
//! `sag'(r²) = c / (2w) + Σ aᵢ (i+2) r^(2i+2)`, `w = √(1 − κc²r²)`; the normal
//! `N ∝ (−x(c + 2wP'), −y(c + 2wP'), w)` (multiplied through by `w`, so no
//! division by `w`).
//!
//! Failures are classified so that every backend agrees on ray survival:
//!
//! * [`IntersectError::Miss`]: the iteration broke down (`g' = 0`,
//!   non-finite step), or stalled at the iteration cap without converging,
//!   **and** `g` never changed sign over the iterates, i.e. nothing indicates
//!   that the ray crosses the surface. Leaving the sag domain before any sign
//!   change, when the conic seed already missed, is also a miss.
//! * [`IntersectError::NoConvergence`]: `g` changed sign (a root is
//!   bracketed) but Newton did not converge, or the cap was reached while
//!   `|g|` was still decreasing by at least a factor two per step.
//! * [`IntersectError::OutsideDomain`]: an iterate left the domain
//!   `1 − κc²r² ≥ 0` in any other case.
//!
//! # Plane, ideal lens, coordinate break
//!
//! Plane and ideal (paraxial) lens: `t = −P_z / D_z`, normal `+z`. A
//! coordinate break has no geometry: the ray point is carried unchanged
//! (`t = 0`), which keeps 90° breaks well defined.

use mklight_core::{SurfaceShape, Vec3};
use serde::{Deserialize, Serialize};

use crate::numeric::discriminant;

/// Tolerance of the vertex-sheet test `1 − κcz ≥ −tol`.
pub(crate) const SHEET_TOL: f64 = 1e-12;

/// Settings of the iterative (asphere) intersection.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntersectSettings {
    /// Convergence tolerance on the sag residual `|z − sag(r²)|` (mm).
    pub tolerance: f64,
    /// Maximum Newton iterations before the ray is reported as failed.
    pub max_iterations: u32,
}

impl Default for IntersectSettings {
    fn default() -> Self {
        IntersectSettings {
            tolerance: 1e-12,
            max_iterations: 40,
        }
    }
}

/// Why a ray found no valid intersection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntersectError {
    /// The ray does not meet the surface's vertex sheet.
    Miss,
    /// Newton iteration on an asphere did not converge.
    NoConvergence,
    /// The ray left the domain where the sag is defined (`1 − κc²r² < 0`).
    OutsideDomain,
}

/// A valid intersection. The signed distance from the incoming ray point to
/// the hit is `t_transfer + t_surface`; it is kept in two parts so the
/// optical path can be accumulated without rounding the sum first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Distance to the point of closest approach to the vertex (step 1
    /// above; the plane distance for planes and ideal lenses).
    pub t_transfer: f64,
    /// Remaining distance from that point to the surface.
    pub t_surface: f64,
    /// Local intersection point.
    pub point: Vec3,
    /// Unit surface normal, pointing towards local `+z` on the vertex sheet.
    pub normal: Vec3,
}

/// Intersect the ray `p + t·d` (local frame, `|d| = 1`) with `shape`.
pub fn intersect(
    shape: &SurfaceShape,
    p: Vec3,
    d: Vec3,
    settings: &IntersectSettings,
) -> Result<Hit, IntersectError> {
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } => intersect_plane(p, d),
        SurfaceShape::Conic { curvature, conic } => intersect_conic(*curvature, *conic, p, d),
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => intersect_even_asphere(*curvature, *conic, coefficients, p, d, settings),
        SurfaceShape::Diffractive { base, .. } => {
            let hit = intersect(base, p, d, settings)?;
            if !mklight_core::surface_math::in_domain(shape, hit.point.x, hit.point.y) {
                return Err(IntersectError::OutsideDomain);
            }
            Ok(hit)
        }
        // A zero overlay traces exactly as its base.
        SurfaceShape::SagOverlay {
            base,
            astigmatism: 0.0,
            spherical: 0.0,
            ..
        } => intersect(base, p, d, settings),
        SurfaceShape::CoordinateBreak => Ok(Hit {
            t_transfer: 0.0,
            t_surface: 0.0,
            point: p,
            normal: Vec3::Z,
        }),
        _ => intersect_sag(shape, p, d, settings),
    }
}

/// Intersection with the plane `z = 0`.
#[inline]
pub fn intersect_plane(p: Vec3, d: Vec3) -> Result<Hit, IntersectError> {
    if d.z == 0.0 {
        return Err(IntersectError::Miss);
    }
    let t = -p.z / d.z;
    let point = Vec3::new(t.mul_add(d.x, p.x), t.mul_add(d.y, p.y), 0.0);
    if !point.is_finite() {
        return Err(IntersectError::Miss);
    }
    Ok(Hit {
        t_transfer: t,
        t_surface: 0.0,
        point,
        normal: Vec3::Z,
    })
}

/// Step 1: transfer to the point of closest approach to the vertex.
/// Returns `(t₀, Q)` with `t₀ = −P·D`.
#[inline]
fn to_closest_approach(p: Vec3, d: Vec3) -> (f64, Vec3) {
    let t0 = -p.dot(d);
    (
        t0,
        Vec3::new(
            t0.mul_add(d.x, p.x),
            t0.mul_add(d.y, p.y),
            t0.mul_add(d.z, p.z),
        ),
    )
}

/// Conic intersection (module docs, steps 1–5).
pub fn intersect_conic(c: f64, k: f64, p: Vec3, d: Vec3) -> Result<Hit, IntersectError> {
    if c == 0.0 {
        return intersect_plane(p, d);
    }
    let (t0, q) = to_closest_approach(p, d);
    let (t1, point) = conic_from(c, k, q, d)?;
    Ok(Hit {
        t_transfer: t0,
        t_surface: t1,
        point,
        normal: conic_normal(c, k, point),
    })
}

/// Steps 2–5 from a point `q` (normally the closest approach to the vertex).
/// Returns the distance from `q` and the hit point. Also valid for `c = 0`
/// (the plane `z = 0`).
#[inline]
fn conic_from(c: f64, k: f64, q: Vec3, d: Vec3) -> Result<(f64, Vec3), IntersectError> {
    let kap = 1.0 + k;
    let a = c * (d.x.mul_add(d.x, d.y * d.y) + kap * d.z * d.z);
    let b = c * (q.x.mul_add(d.x, q.y * d.y) + kap * q.z * d.z) - d.z;
    let cc = c * (q.x.mul_add(q.x, q.y * q.y) + kap * q.z * q.z) - 2.0 * q.z;
    let disc = discriminant(a, b, cc);
    // A NaN discriminant (non-finite input) is a miss as well.
    if disc.is_nan() || disc < 0.0 {
        return Err(IntersectError::Miss);
    }
    let sq = disc.sqrt();
    let qq = -(b + if b >= 0.0 { sq } else { -sq });
    let at = |t: f64| {
        Vec3::new(
            t.mul_add(d.x, q.x),
            t.mul_add(d.y, q.y),
            t.mul_add(d.z, q.z),
        )
    };
    let on_vertex_sheet = |pt: Vec3| 1.0 - kap * c * pt.z >= -SHEET_TOL;
    let mut best: Option<(f64, Vec3)> = None;
    // t₁ = C/q (continuous with the plane solution).
    if qq != 0.0 {
        let t = cc / qq;
        let pt = at(t);
        if t.is_finite() && pt.is_finite() && on_vertex_sheet(pt) {
            best = Some((t, pt));
        }
    } else if cc == 0.0 {
        // Double root at the starting point (tangent contact).
        best = Some((0.0, q));
    }
    // t₂ = q/A (the far root).
    if a != 0.0 && qq != 0.0 {
        let t = qq / a;
        let pt = at(t);
        if t.is_finite() && pt.is_finite() && on_vertex_sheet(pt) {
            let r2 = pt.x.mul_add(pt.x, pt.y * pt.y);
            match best {
                Some((_, b1)) if b1.x.mul_add(b1.x, b1.y * b1.y) <= r2 => {}
                _ => best = Some((t, pt)),
            }
        }
    }
    best.ok_or(IntersectError::Miss)
}

/// Unit normal of a conic at a point on it: `(−cx, −cy, 1 − κcz)/‖·‖`.
#[inline]
pub fn conic_normal(c: f64, k: f64, p: Vec3) -> Vec3 {
    Vec3::new(-c * p.x, -c * p.y, (-(1.0 + k) * c).mul_add(p.z, 1.0)).normalized()
}

/// Even-asphere polynomial `P(r²) = Σ aᵢ r^(2i+4)` and `dP/d(r²)`, by Horner.
#[inline]
pub fn asphere_poly(coefficients: &[f64], r2: f64) -> (f64, f64) {
    let mut p = 0.0f64;
    let mut dp = 0.0f64;
    for (i, a) in coefficients.iter().enumerate().rev() {
        p = p.mul_add(r2, *a);
        dp = dp.mul_add(r2, a * (i as f64 + 2.0));
    }
    (p * r2 * r2, dp * r2)
}

/// Sag `z(r²)` and its derivative `dz/d(r²)` for any shape. `None` outside
/// the conic's domain. `dz/d(r²)` is infinite at the rim of a closed conic.
pub fn sag_and_slope(shape: &SurfaceShape, r2: f64) -> Option<(f64, f64)> {
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
            Some((0.0, 0.0))
        }
        SurfaceShape::Conic { curvature, conic } => {
            let (z, w) = conic_sag_w(*curvature, *conic, r2)?;
            Some((z, 0.5 * curvature / w))
        }
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => {
            let (z, w) = conic_sag_w(*curvature, *conic, r2)?;
            let (p, dp) = asphere_poly(coefficients, r2);
            Some((z + p, 0.5 * curvature / w + dp))
        }
        _ => {
            let r = r2.sqrt();
            let v = mklight_core::surface_math::evaluate(shape, 0.0, r)?;
            Some((
                v.sag,
                if r == 0.0 {
                    0.5 * mklight_core::surface_math::vertex_curvatures(shape)[2]
                } else {
                    v.gradient[1] / (2.0 * r)
                },
            ))
        }
    }
}

/// Conic sag and `w = √(1 − κc²r²)`. `None` outside the domain.
#[inline]
fn conic_sag_w(c: f64, k: f64, r2: f64) -> Option<(f64, f64)> {
    let arg = (-(1.0 + k) * c * c).mul_add(r2, 1.0);
    if arg.is_nan() || arg < 0.0 {
        return None;
    }
    let w = arg.sqrt();
    Some((c * r2 / (1.0 + w), w))
}

/// Unit normal of `shape` at local point `p` (assumed on the surface).
pub fn surface_normal(shape: &SurfaceShape, p: Vec3) -> Option<Vec3> {
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
            Some(Vec3::Z)
        }
        SurfaceShape::Conic { curvature, conic } => Some(conic_normal(*curvature, *conic, p)),
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => asphere_normal(*curvature, *conic, coefficients, p),
        SurfaceShape::Diffractive { base, .. }
        | SurfaceShape::SagOverlay {
            base,
            astigmatism: 0.0,
            spherical: 0.0,
            ..
        } => surface_normal(base, p),
        _ => mklight_core::surface_math::normal(shape, p.x, p.y),
    }
}

#[inline]
fn asphere_normal(c: f64, k: f64, coefficients: &[f64], p: Vec3) -> Option<Vec3> {
    let r2 = p.x.mul_add(p.x, p.y * p.y);
    let (_, w) = conic_sag_w(c, k, r2)?;
    let (_, dp) = asphere_poly(coefficients, r2);
    let g = (2.0 * w).mul_add(dp, c);
    Some(Vec3::new(-p.x * g, -p.y * g, w).normalized())
}

/// Even asphere: conic seed, then Newton on the sag residual (module docs).
pub fn intersect_even_asphere(
    c: f64,
    k: f64,
    coefficients: &[f64],
    p: Vec3,
    d: Vec3,
    settings: &IntersectSettings,
) -> Result<Hit, IntersectError> {
    if coefficients.iter().all(|a| *a == 0.0) {
        return intersect_conic(c, k, p, d);
    }
    let (t0, q) = to_closest_approach(p, d);
    let seed = conic_from(c, k, q, d);
    let seed_hit = seed.is_ok();
    let mut t = seed.map_or(0.0, |(t, _)| t);
    let kap = 1.0 + k;
    // Sign of the first residual and whether any later one differs.
    let mut g_first = 0.0f64;
    let mut sign_changed = false;
    let mut prev_abs_g = f64::INFINITY;
    // Breakdown without evidence of a crossing is a miss (module docs).
    let breakdown = |sign_changed: bool| {
        if sign_changed {
            IntersectError::NoConvergence
        } else {
            IntersectError::Miss
        }
    };
    for it in 0..=settings.max_iterations {
        let x = t.mul_add(d.x, q.x);
        let y = t.mul_add(d.y, q.y);
        let z = t.mul_add(d.z, q.z);
        let r2 = x.mul_add(x, y * y);
        let arg = (-kap * c * c).mul_add(r2, 1.0);
        if arg.is_nan() || arg < 0.0 {
            return Err(if !seed_hit && !sign_changed {
                IntersectError::Miss
            } else {
                IntersectError::OutsideDomain
            });
        }
        let w = arg.sqrt();
        let (poly, dpoly) = asphere_poly(coefficients, r2);
        let sag = c * r2 / (1.0 + w) + poly;
        let g = z - sag;
        if !g.is_finite() {
            return Err(breakdown(sign_changed));
        }
        if it == 0 {
            g_first = g;
        } else if g != 0.0 && (g > 0.0) != (g_first > 0.0) {
            sign_changed = true;
        }
        // g'(t) · w, kept free of 1/w so it is finite at the rim.
        let slope_w = (2.0 * w).mul_add(dpoly, c);
        let gp = d.z - slope_w / w.max(f64::MIN_POSITIVE) * x.mul_add(d.x, y * d.y);
        let gp_ok = gp.is_finite() && gp != 0.0;
        if g.abs() <= settings.tolerance {
            // Final correction: the residual is now O(g²).
            if gp_ok {
                t -= g / gp;
            }
            let point = Vec3::new(
                t.mul_add(d.x, q.x),
                t.mul_add(d.y, q.y),
                t.mul_add(d.z, q.z),
            );
            let normal =
                asphere_normal(c, k, coefficients, point).ok_or(IntersectError::OutsideDomain)?;
            return Ok(Hit {
                t_transfer: t0,
                t_surface: t,
                point,
                normal,
            });
        }
        if it == settings.max_iterations {
            // Cap reached: still converging, or a bracketed root, is a
            // non-convergence; a stalled search with no crossing is a miss.
            let converging = g.abs() <= 0.5 * prev_abs_g;
            return Err(if converging {
                IntersectError::NoConvergence
            } else {
                breakdown(sign_changed)
            });
        }
        if !gp_ok {
            return Err(breakdown(sign_changed));
        }
        t -= g / gp;
        if !t.is_finite() {
            return Err(breakdown(sign_changed));
        }
        prev_abs_g = g.abs();
    }
    unreachable!("the loop returns at it == max_iterations")
}

/// General sag intersection: conic seed, safeguarded Newton in the closest
/// approach frame. A bracket restricts Newton to its interval; backtracking
/// keeps unbracketed steps in-domain and decreases the sag residual. Failure
/// classes follow the even asphere's (module docs).
fn intersect_sag(
    shape: &SurfaceShape,
    p: Vec3,
    d: Vec3,
    settings: &IntersectSettings,
) -> Result<Hit, IntersectError> {
    use mklight_core::surface_math::{evaluate, exact_conic};
    // Coincident conics keep the closed-form solution exactly.
    if let Some((c, k)) = exact_conic(shape) {
        let hit = intersect_conic(c, k, p, d)?;
        return evaluate(shape, hit.point.x, hit.point.y)
            .map(|_| hit)
            .ok_or(IntersectError::OutsideDomain);
    }
    let (t0, q) = to_closest_approach(p, d);
    let seed = conic_from(shape.curvature(), shape.conic(), q, d);
    let seed_hit = seed.is_ok();
    let mut t = seed.map_or(0.0, |(t, _)| t);
    let at = |t: f64| {
        Vec3::new(
            t.mul_add(d.x, q.x),
            t.mul_add(d.y, q.y),
            t.mul_add(d.z, q.z),
        )
    };
    let (mut positive, mut negative) = (None, None);
    let mut previous = f64::INFINITY;
    for it in 0..=settings.max_iterations {
        let bracketed = positive.is_some() && negative.is_some();
        // Leaving the domain with no evidence of a crossing after a missed
        // seed is a miss.
        let left_domain = if !seed_hit && !bracketed {
            IntersectError::Miss
        } else {
            IntersectError::OutsideDomain
        };
        let pt = at(t);
        let Some(v) = evaluate(shape, pt.x, pt.y) else {
            return Err(left_domain);
        };
        let g = pt.z - v.sag;
        let gp = d.z - v.gradient[0] * d.x - v.gradient[1] * d.y;
        if g > 0.0 {
            positive = Some(t);
        } else if g < 0.0 {
            negative = Some(t);
        }
        let bracket = positive
            .zip(negative)
            .map(|(a, b): (f64, f64)| (a.min(b), a.max(b)));
        let failure = if bracket.is_some() {
            IntersectError::NoConvergence
        } else {
            IntersectError::Miss
        };
        if g.abs() <= settings.tolerance {
            // One final Newton step, provided it stays on this real sheet.
            let corrected = at(t - g / gp);
            if evaluate(shape, corrected.x, corrected.y).is_some() {
                t -= g / gp;
            }
            let point = at(t);
            return Ok(Hit {
                t_transfer: t0,
                t_surface: t,
                point,
                normal: surface_normal(shape, point).ok_or(IntersectError::OutsideDomain)?,
            });
        }
        if it == settings.max_iterations {
            return Err(if g.abs() <= 0.5 * previous {
                IntersectError::NoConvergence
            } else {
                failure
            });
        }
        if !g.is_finite() || gp == 0.0 || !gp.is_finite() {
            return Err(failure);
        }
        let mut next = t - g / gp;
        if let Some((lo, hi)) = bracket
            && !(next > lo && next < hi)
        {
            next = lo + 0.5 * (hi - lo);
        }
        let mut outside = false;
        let mut accepted = false;
        for _ in 0..20 {
            let p = at(next);
            match evaluate(shape, p.x, p.y) {
                Some(v) if bracket.is_some() || (p.z - v.sag).abs() < g.abs() => {
                    accepted = true;
                    break;
                }
                Some(_) => {}
                None => outside = true,
            }
            next = t + 0.5 * (next - t);
        }
        if !accepted {
            return Err(if outside && bracket.is_none() {
                left_domain
            } else {
                failure
            });
        }
        previous = g.abs();
        t = next;
    }
    unreachable!("the loop returns at it == max_iterations")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z).normalized()
    }

    #[test]
    fn sphere_hit_lies_on_surface() {
        let shape = SurfaceShape::sphere(40.0);
        let p = Vec3::new(3.0, -2.0, -25.0);
        let d = unit(0.05, 0.1, 1.0);
        let h = intersect(&shape, p, d, &IntersectSettings::default()).unwrap();
        let r2 = h.point.x * h.point.x + h.point.y * h.point.y;
        assert!((h.point.z - shape.sag(r2).unwrap()).abs() < 1e-14);
        // The hit is p + t d.
        let t = h.t_transfer + h.t_surface;
        assert!((p + d * t).max_abs_diff(h.point) < 1e-13);
        // Normal: unit, and parallel to (centre − point) for a sphere.
        let centre = Vec3::new(0.0, 0.0, 40.0);
        let radial = (centre - h.point).normalized();
        assert!((h.normal.dot(radial) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn double_crossing_takes_the_root_nearer_the_vertex() {
        // c = −0.05 bulges toward the ray, which crosses it twice heading +z
        // at ~87° to the axis: first at y ≈ −5.5 (t ≈ 24.51), then at
        // y ≈ +3.6 (t ≈ 33.61). The surface's extent is unknown here, so
        // the crossing nearer the vertex wins (docs/SURFACES.md §8).
        let shape = SurfaceShape::sphere(-20.0);
        let d = unit(0.0, 0.999, 0.050);
        let p = Vec3::new(0.0, -30.0, -2.0);
        let m = p - Vec3::new(0.0, 0.0, -20.0);
        let (b, c) = (m.dot(d), m.dot(m) - 400.0);
        let t2 = -b + (b * b - c).sqrt();
        assert!((t2 - 33.61).abs() < 0.01);
        let h = intersect(&shape, p, d, &IntersectSettings::default()).unwrap();
        assert!((h.t_transfer + h.t_surface - t2).abs() < 1e-12, "{h:?}");
        assert!(h.point.y > 3.0, "{h:?}");
    }

    #[test]
    fn backward_propagation_allowed() {
        // Ray starting beyond the vertex, heading +z, meets the sphere behind it.
        let shape = SurfaceShape::sphere(-30.0);
        let p = Vec3::new(0.0, 5.0, 3.0);
        let h = intersect(&shape, p, Vec3::Z, &IntersectSettings::default()).unwrap();
        assert!(h.t_transfer + h.t_surface < 0.0);
        let exact = -30.0 + (900.0f64 - 25.0).sqrt();
        assert!((h.point.z - exact).abs() < 1e-14);
    }

    #[test]
    fn rejects_far_sheet_and_misses() {
        // Ray beside a small sphere misses.
        let shape = SurfaceShape::sphere(5.0);
        assert_eq!(
            intersect(
                &shape,
                Vec3::new(0.0, 6.0, -1.0),
                Vec3::Z,
                &Default::default()
            ),
            Err(IntersectError::Miss)
        );
        // Hyperboloid: an axial ray from far behind must hit the vertex sheet.
        let hyp = SurfaceShape::Conic {
            curvature: 0.1,
            conic: -4.0,
        };
        let h = intersect(
            &hyp,
            Vec3::new(0.0, 1.0, -100.0),
            Vec3::Z,
            &Default::default(),
        )
        .unwrap();
        assert!(h.point.z > 0.0 && h.point.z < 0.1);
        assert!(1.0 + 3.0 * 0.1 * h.point.z > 0.0);
    }

    #[test]
    fn oblate_and_prolate_on_surface() {
        for k in [2.5, 0.0, -0.5, -1.0, -3.0] {
            let shape = SurfaceShape::Conic {
                curvature: -0.02,
                conic: k,
            };
            let d = unit(0.0, -0.2, 1.0);
            let h = intersect(&shape, Vec3::new(1.0, 4.0, -10.0), d, &Default::default()).unwrap();
            let r2 = h.point.x.powi(2) + h.point.y.powi(2);
            assert!((h.point.z - shape.sag(r2).unwrap()).abs() < 1e-14, "k={k}");
        }
    }

    #[test]
    fn grazing_ray_on_flat_like_surface() {
        // A ray at 89.9999° to the axis still finds the near-vertex root.
        let shape = SurfaceShape::sphere(1e4);
        let d = unit(1.0, 0.0, 1e-6);
        let h = intersect(&shape, Vec3::new(-5.0, 0.0, -1e-6), d, &Default::default());
        let h = h.unwrap();
        let r2 = h.point.x.powi(2);
        assert!((h.point.z - shape.sag(r2).unwrap()).abs() < 1e-12);
    }

    /// Sag residual of a hit on `shape`.
    fn residual(shape: &SurfaceShape, h: &Hit) -> f64 {
        let r2 = h.point.x.mul_add(h.point.x, h.point.y * h.point.y);
        h.point.z - shape.sag(r2).unwrap()
    }

    #[test]
    fn grazing_rays_stay_on_the_surface() {
        // Rays almost parallel to the vertex tangent plane, D_z down to 0:
        // the closest-approach transfer keeps the hit exact.
        let cases = [
            (0.01, 0.0, Vec3::new(0.0, -5.0, 0.1)),
            (0.02, 0.0, Vec3::new(0.0, -30.0, 2.0)),
            (-0.02, -0.7, Vec3::new(3.0, -30.0, -2.0)),
            (0.05, -1.0, Vec3::new(0.0, -20.0, 1.0)),
        ];
        for (c, k, p) in cases {
            let shape = SurfaceShape::Conic {
                curvature: c,
                conic: k,
            };
            let asph = SurfaceShape::EvenAsphere {
                curvature: c,
                conic: k,
                coefficients: vec![1e-6, -2e-9],
            };
            for e in 2..=12 {
                for dz in [10f64.powi(-e), -(10f64.powi(-e))] {
                    let d = unit(0.0, 1.0, dz);
                    for s in [&shape, &asph] {
                        let h = intersect(s, p, d, &Default::default())
                            .unwrap_or_else(|err| panic!("c={c} k={k} dz={dz}: {err:?}"));
                        assert!(
                            residual(s, &h).abs() < 1e-12,
                            "c={c} k={k} dz={dz}: residual {}",
                            residual(s, &h)
                        );
                        let t = h.t_transfer + h.t_surface;
                        assert!((p + d * t).max_abs_diff(h.point) < 1e-12);
                    }
                }
            }
            // Exactly parallel to the tangent plane.
            let h = intersect(&shape, p, Vec3::Y, &Default::default()).unwrap();
            assert!(residual(&shape, &h).abs() < 1e-12);
        }
        // The near-vertex root of the reviewer's case: y = −√(R² − (R − z)²).
        let shape = SurfaceShape::Conic {
            curvature: 0.01,
            conic: 0.0,
        };
        let h = intersect(
            &shape,
            Vec3::new(0.0, -5.0, 0.1),
            unit(0.0, 1.0, 2e-9),
            &Default::default(),
        )
        .unwrap();
        let y = (1e4f64 - 99.9f64.powi(2)).sqrt();
        assert!((h.point.y.abs() - y).abs() < 1e-6, "{:?}", h.point);
    }

    #[test]
    fn asphere_without_a_crossing_is_a_miss() {
        // z − 1e-3 r⁴ < 0 along the whole ray: no root, so the ray misses
        // (not a numerical failure).
        let shape = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![1e-3],
        };
        let p = Vec3::new(0.0, 9.0, -1.0);
        let d = unit(0.0, 0.5, 1.0);
        for s in 0..=2500 {
            let t = -50.0 + 0.1 * s as f64;
            let q = p + d * t;
            assert!(q.z - 1e-3 * q.y.powi(4) < 0.0);
        }
        assert_eq!(
            intersect(&shape, p, d, &Default::default()),
            Err(IntersectError::Miss)
        );
        // A ray that does cross converges.
        let h = intersect(
            &shape,
            Vec3::new(0.0, 2.0, -1.0),
            Vec3::Z,
            &Default::default(),
        )
        .unwrap();
        assert!((h.point.z - 1.6e-2).abs() < 1e-14);
    }

    #[test]
    fn asphere_failures_are_reported() {
        let shape = SurfaceShape::EvenAsphere {
            curvature: 1.0 / 30.0,
            conic: 0.0,
            coefficients: vec![2e-5],
        };
        let p = Vec3::new(0.0, 8.0, -12.0);
        let one_step = IntersectSettings {
            tolerance: 1e-12,
            max_iterations: 1,
        };
        assert_eq!(
            intersect(&shape, p, unit(0.0, 0.25, 1.0), &one_step),
            Err(IntersectError::NoConvergence)
        );
        // Beyond the sphere's rim (r > 30) the sag is undefined.
        let r = intersect(
            &shape,
            Vec3::new(0.0, 31.0, -1.0),
            Vec3::Z,
            &Default::default(),
        );
        assert!(matches!(
            r,
            Err(IntersectError::OutsideDomain | IntersectError::Miss)
        ));
    }

    #[test]
    fn asphere_newton_converges() {
        let shape = SurfaceShape::EvenAsphere {
            curvature: 1.0 / 30.0,
            conic: -0.6,
            coefficients: vec![2e-5, -3e-8, 1e-11],
        };
        let d = unit(0.0, 0.25, 1.0);
        let h = intersect(&shape, Vec3::new(0.0, 8.0, -12.0), d, &Default::default()).unwrap();
        let r2 = h.point.x.powi(2) + h.point.y.powi(2);
        assert!((h.point.z - shape.sag(r2).unwrap()).abs() < 1e-13);
        // Normal agrees with the analytic sag slope.
        let (_, s) = sag_and_slope(&shape, r2).unwrap();
        let n = Vec3::new(-2.0 * h.point.x * s, -2.0 * h.point.y * s, 1.0).normalized();
        assert!(n.max_abs_diff(h.normal) < 1e-14);
    }

    #[test]
    fn asphere_poly_derivative() {
        let a = [1e-4, -2e-6, 3e-9];
        let r2: f64 = 7.3;
        let (p, dp) = asphere_poly(&a, r2);
        let e = 1e-6;
        let fd = (asphere_poly(&a, r2 + e).0 - asphere_poly(&a, r2 - e).0) / (2.0 * e);
        assert!((p - (a[0] * r2.powi(2) + a[1] * r2.powi(3) + a[2] * r2.powi(4))).abs() < 1e-16);
        assert!((dp - fd).abs() < 1e-10);
    }
}
