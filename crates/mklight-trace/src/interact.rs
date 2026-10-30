//! Ray–surface interactions: refraction, reflection and the ideal thin lens.
//! All vectors are in the surface's local frame; directions are unit vectors
//! and refractive indices are positive (CONVENTIONS §2).

use mklight_core::Vec3;

/// Stable vector Snell law (CONVENTIONS §7.3).
///
/// With `μ = n₁/n₂`, `cos i = |k·n|` and the normal `n` flipped so that
/// `k·n ≥ 0`, the refracted direction is
/// `k' = μ k + γ n`, `γ = √(1 − μ²(1 − cos²i)) − μ cos i`.
///
/// `1 − cos²i` is evaluated as `(1 − cos i)(1 + cos i)`. `γ` itself is
/// computed in the cancellation-free form
/// `γ = (1 − μ²) / (√(…) + μ cos i) = (n₂ − n₁)(n₂ + n₁) / n₂² / (√(…) + μ cos i)`,
/// which is exact algebra (`(√…)² − μ²cos²i = 1 − μ²`) and avoids the
/// subtraction of two nearly equal numbers at near-normal incidence on
/// low-contrast interfaces (e.g. cemented doublets).
///
/// Returns `None` on total internal reflection (`1 − μ² sin²i < 0`).
/// The second value is `cos i ∈ [0, 1]`.
#[inline]
pub fn refract(k: Vec3, normal: Vec3, n1: f64, n2: f64) -> Option<(Vec3, f64)> {
    let mut cos_i = k.dot(normal);
    let n = if cos_i < 0.0 {
        cos_i = -cos_i;
        -normal
    } else {
        normal
    };
    let cos_i = cos_i.min(1.0);
    let mu = n1 / n2;
    let sin2 = (1.0 - cos_i) * (1.0 + cos_i);
    let disc = (-mu * mu).mul_add(sin2, 1.0);
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let denom = mu.mul_add(cos_i, root);
    let gamma = if denom > 0.0 {
        ((n2 - n1) * (n2 + n1)) / (n2 * n2) / denom
    } else {
        root - mu * cos_i
    };
    let out = Vec3::new(
        mu.mul_add(k.x, gamma * n.x),
        mu.mul_add(k.y, gamma * n.y),
        mu.mul_add(k.z, gamma * n.z),
    );
    Some((out, cos_i))
}

/// Fixed-period diffraction: n₂ k₂∥ = n₁ k₁∥ + m λ ∇ₛΦ/(2π).
/// `phase_gradient` is with respect to the base's Cartesian x/y coordinates,
/// in rad/mm; λ is in mm. The graph's surface gradient is its tangential
/// projection. `reflected` selects the outgoing normal hemisphere.
/// `None` is an evanescent order, distinct from ordinary TIR.
pub fn diffract(
    k: Vec3,
    normal: Vec3,
    n1: f64,
    n2: f64,
    phase_gradient: [f64; 2],
    wavelength_mm: f64,
    order: i32,
    reflected: bool,
) -> Option<Vec3> {
    let kn = k.dot(normal);
    let grad = Vec3::new(phase_gradient[0], phase_gradient[1], 0.0);
    let tangent = (k - normal * kn) * (n1 / n2)
        + (grad - normal * grad.dot(normal))
            * (order as f64 * wavelength_mm / (std::f64::consts::TAU * n2));
    let disc = 1.0 - tangent.dot(tangent);
    if disc < 0.0 {
        return None;
    }
    let sign = if (kn < 0.0) != reflected { -1.0 } else { 1.0 };
    Some(tangent + normal * (sign * disc.sqrt()))
}

/// Specular reflection `k' = k − 2(k·n)n`. Returns `(k', cos i)`.
#[inline]
pub fn reflect(k: Vec3, normal: Vec3) -> (Vec3, f64) {
    let kn = k.dot(normal);
    let s = -2.0 * kn;
    (
        Vec3::new(
            s.mul_add(normal.x, k.x),
            s.mul_add(normal.y, k.y),
            s.mul_add(normal.z, k.z),
        ),
        kn.abs().min(1.0),
    )
}

/// Ideal thin lens of focal length `f` (image-space, in the medium after the
/// lens; `f > 0` converges) at the local plane `z = 0`.
///
/// # Ray geometry
///
/// With direction tangents taken along the propagation direction,
/// `t = (k_x, k_y, k_z)/|k_z|` (so `t_z = ±1`), and the hit point
/// `X = (x, y, 0)`, the outgoing lateral tangents are
/// `t'_⊥ = r·t_⊥ − X/f`, `r = n₁/n₂`. Every ray from an object point at
/// distance `a` in front of the lens (along the incoming propagation) then
/// passes through its Gaussian image at distance `b` behind it, where
/// `n₂/b + n₁/a = n₂/f`, for every `a` (plane waves: `b = f`). The element
/// is free of ray aberrations for real rays, not only paraxially.
///
/// # Optical path
///
/// A thin element can only be perfect in phase for one conjugate pair. The
/// caller passes the paraxial **object vergence** `ρ_a = 1/a` of the axial
/// object (see [`crate::paraxial::ideal_lens_object_vergence`]); the image
/// vergence is `ρ_b = 1/b = 1/f − r ρ_a`. For each ray, its object point is
/// put on the incoming line, `O = X − a t`, and its image point on the
/// outgoing line, `O' = X + b t'`. All rays of a pencil from one object point
/// share `O` and `O'` (above), and the path added at the lens is
///
/// `Δ = n₁ sgn(a)(|O| − |X − O|) + n₂ sgn(b)(|O'| − |O' − X|)`,
///
/// which makes the optical path `O → X → O'` equal to the one through the
/// vertex (`Δ(0) = 0`), with the signs selecting diverging (real) or
/// converging (virtual) spheres. `Δ` is also stationary with respect to the
/// incoming direction (the `O` and `O'` terms change by `∓n₁ X·δt`), so
/// aberrations of the incoming wavefront are carried through unchanged to
/// first order, as by a physical thin lens. Each term is evaluated by
/// [`sphere_term`] without cancellation, and reduces to the plane-wave law
/// `−n₁ X·k` (object) or `n₂ X·k'` (image) at zero vergence.
///
/// Returns `None` when the ray is parallel to the lens plane.
#[inline]
pub fn ideal_lens(
    point: Vec3,
    k: Vec3,
    f: f64,
    n1: f64,
    n2: f64,
    object_vergence: f64,
) -> Option<(Vec3, f64)> {
    if k.z == 0.0 || f == 0.0 || !f.is_finite() {
        return None;
    }
    let sz = k.z.signum();
    let az = k.z.abs();
    let r = n1 / n2;
    let t = Vec3::new(k.x / az, k.y / az, sz);
    let tx2 = r.mul_add(t.x, -point.x / f);
    let ty2 = r.mul_add(t.y, -point.y / f);
    let inv = 1.0 / (tx2.mul_add(tx2, ty2.mul_add(ty2, 1.0))).sqrt();
    let out = Vec3::new(tx2 * inv, ty2 * inv, sz * inv);
    let t_out = Vec3::new(tx2, ty2, sz);
    let x = Vec3::new(point.x, point.y, 0.0);
    let rho_b = (-r).mul_add(object_vergence, 1.0 / f);
    let delta = n1.mul_add(
        sphere_term(x, t, object_vergence),
        -n2 * sphere_term(x, t_out, -rho_b),
    );
    Some((out, delta))
}

/// `sgn(a)(|O| − |X − O|)` with `O = X − a t`, from the vergence `ρ = 1/a`
/// (`X·ẑ = 0`, `|t_z| = 1`). With `D = |O|² − |X − O|² = X·X − 2a t·X`:
///
/// * far point (`|a||t| ≥ |X|`, including `ρ = 0`): divide `D` and the sum
///   of lengths by `|a|`, giving `(ρ X·X − 2 t·X) / (|t − ρX| + |t|)`
///   (`−t·X/|t| = −k·X` for a plane wave);
/// * near point: `sgn(a)(2 O·X − X·X)/(|O| + |a||t|)` (`a = 0`: `±|X|`).
///
/// Neither form subtracts two nearly equal lengths. The image-side term is
/// `sgn(b)(|O'| − |O' − X|) = −sphere_term(X, t', −1/b)`.
#[inline]
pub fn sphere_term(x: Vec3, t: Vec3, rho: f64) -> f64 {
    let tn = t.norm();
    let xx = x.dot(x);
    let tx = t.dot(x);
    if rho.abs() * xx.sqrt() <= tn {
        let den = (t - x * rho).norm() + tn;
        rho.mul_add(xx, -2.0 * tx) / den
    } else {
        let a = 1.0 / rho;
        let o = x - t * a;
        a.signum() * 2.0f64.mul_add(o.dot(x), -xx) / (o.norm() + a.abs() * tn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snell_law_holds() {
        let n = Vec3::new(0.1, -0.2, 1.0).normalized();
        let k = Vec3::new(0.3, 0.4, 1.0).normalized();
        let (n1, n2) = (1.0, 1.7);
        let (kp, ci) = refract(k, n, n1, n2).unwrap();
        assert!((kp.norm() - 1.0).abs() < 1e-15);
        // n1 (k × n) = n2 (k' × n): tangential components conserved.
        let a = k.cross(n) * n1;
        let b = kp.cross(n) * n2;
        assert!(a.max_abs_diff(b) < 1e-15);
        assert!((ci - k.dot(n)).abs() < 1e-16);
        // Coplanar with k and n.
        assert!(kp.dot(k.cross(n)).abs() < 1e-15);
    }

    #[test]
    fn normal_orientation_irrelevant_and_tir() {
        let n = Vec3::new(0.0, 0.3, 1.0).normalized();
        let k = Vec3::new(0.0, -0.1, 1.0).normalized();
        let (a, _) = refract(k, n, 1.5, 1.0).unwrap();
        let (b, _) = refract(k, -n, 1.5, 1.0).unwrap();
        assert!(a.max_abs_diff(b) < 1e-16);
        // 45° inside glass n=1.5 → air is beyond the 41.8° critical angle.
        let k45 = Vec3::new(
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        assert!(refract(k45, Vec3::Z, 1.5, 1.0).is_none());
        assert!(refract(k45, Vec3::Z, 1.5, 1.2).is_some());
    }

    #[test]
    fn reflection_preserves_tangential() {
        let n = Vec3::new(0.2, 0.0, 1.0).normalized();
        let k = Vec3::new(0.0, 0.1, 1.0).normalized();
        let (r, _) = reflect(k, n);
        assert!((r.dot(n) + k.dot(n)).abs() < 1e-15);
        assert!(r.cross(n).max_abs_diff(k.cross(n)) < 1e-15);
    }

    #[test]
    fn ideal_lens_focuses_plane_wave_with_equal_path() {
        let f = 80.0;
        let k = Vec3::new(0.05, -0.1, 1.0).normalized();
        let focus = Vec3::new(f * k.x / k.z, f * k.y / k.z, f);
        let mut opls = Vec::new();
        for &(x, y) in &[(0.0, 0.0), (10.0, 0.0), (-7.0, 9.0), (3.0, -12.0)] {
            let p = Vec3::new(x, y, 0.0);
            let (kp, delta) = ideal_lens(p, k, f, 1.0, 1.0, 0.0).unwrap();
            let t = (focus.z - p.z) / kp.z;
            assert!((p + kp * t).max_abs_diff(focus) < 1e-12);
            // Path from the incoming plane wavefront through the origin.
            opls.push(p.dot(k) + delta + t);
        }
        for o in &opls {
            assert!((o - opls[0]).abs() < 1e-12, "{opls:?}");
        }
    }

    #[test]
    fn ideal_lens_is_perfect_at_finite_conjugates() {
        // Object point O at axial distance a (real a > 0 in front, virtual
        // a < 0 behind): all rays through the lens meet at one image point
        // O', and the optical path O → X → O' (signed for virtual points)
        // equals the vertex ray's.
        for (n1, n2) in [(1.0f64, 1.0f64), (1.0, 1.5), (1.6, 1.2)] {
            for f in [80.0f64, -60.0] {
                for a in [300.0f64, 120.0, 37.0, 0.5, -45.0, -400.0, 1e9] {
                    let r = n1 / n2;
                    let rho_b = 1.0 / f - r / a;
                    if rho_b.abs() < 1e-6 {
                        continue;
                    }
                    let b = 1.0 / rho_b;
                    let o = Vec3::new(3.0, -7.0, -a);
                    let sa = a.signum();
                    let sb = b.signum();
                    let mut image = None;
                    let mut opls = Vec::new();
                    for &(x, y) in &[(0.0, 0.0), (10.0, 0.0), (-7.0, 9.0), (3.0, -12.0)] {
                        let p = Vec3::new(x, y, 0.0);
                        let k = ((p - o) * sa).normalized();
                        let (kp, delta) = ideal_lens(p, k, f, n1, n2, 1.0 / a).unwrap();
                        // Image point on the outgoing line at axial distance b.
                        let q = p + kp * (b / kp.z);
                        match image {
                            None => image = Some(q),
                            Some(i) => assert!(
                                q.max_abs_diff(i) < 1e-9 * b.abs().max(1.0),
                                "n=({n1},{n2}) f={f} a={a}: {q:?} vs {i:?}"
                            ),
                        }
                        opls.push(sa * n1 * (p - o).norm() + delta + sb * n2 * (q - p).norm());
                    }
                    let scale = a.abs().max(b.abs()).max(1.0);
                    for v in &opls {
                        assert!(
                            (v - opls[0]).abs() < 1e-13 * scale,
                            "n=({n1},{n2}) f={f} a={a}: {opls:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sphere_term_forms_agree() {
        // The far and near forms are the same function: compare against the
        // direct definition where it is well conditioned.
        let x = Vec3::new(4.0, -3.0, 0.0);
        let t = Vec3::new(0.1, 0.2, 1.0);
        for a in [1000.0f64, 50.0, 5.0, 2.0, 0.5, -0.5, -3.0, -80.0] {
            let o = x - t * a;
            let direct = a.signum() * (o.norm() - (x - o).norm());
            assert!((sphere_term(x, t, 1.0 / a) - direct).abs() < 1e-12, "a={a}");
        }
        assert!((sphere_term(x, t, 0.0) + t.dot(x) / t.norm()).abs() < 1e-15);
        assert!((sphere_term(x, t, f64::INFINITY) - 5.0).abs() < 1e-15);
        assert_eq!(sphere_term(Vec3::ZERO, t, 0.3), 0.0);
    }
}
