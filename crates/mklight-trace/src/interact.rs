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
/// With direction tangents taken along the propagation direction,
/// `t = (k_x, k_y)/|k_z|`, and the hit point `(x, y)`, the outgoing tangents
/// are `t' = (n₁/n₂)·t − (x, y)/f`. Every ray of an incoming plane wave with
/// tangent `t` then passes through the focal point `F = (f(n₁/n₂)t, ±f)`, and
/// (for `n₁ = n₂`) every ray from an object point at distance `a` in front
/// passes through its Gaussian image at `b` behind (`1/a + 1/b = 1/f`): the
/// element is free of ray aberrations for real rays, not only paraxially.
///
/// Optical path: so that the element is also perfect in phase, the
/// optical path of the ray is increased by
/// `Δ = sgn(f)·n₂(|F| − |F − X|) − n₁ (X·k)`, `X = (x, y, 0)`.
/// This makes every ray of an incoming plane wave arrive at `F` with the same
/// optical path as the vertex ray (`Δ(0) = 0`). `|F| − |F − X|` is evaluated
/// as `(2F·X − X·X)/(|F| + |F − X|)`, free of cancellation. The path
/// correction is exact for collimated input (the standard use of an ideal
/// lens: focusing afocal systems or idealised eyepieces); for finite
/// conjugates the ray geometry is still perfect but OPD contains the residual
/// of this plane-wave phase law.
///
/// Returns `None` when the ray is parallel to the lens plane.
#[inline]
pub fn ideal_lens(point: Vec3, k: Vec3, f: f64, n1: f64, n2: f64) -> Option<(Vec3, f64)> {
    if k.z == 0.0 || f == 0.0 || !f.is_finite() {
        return None;
    }
    let sz = k.z.signum();
    let az = k.z.abs();
    let r = n1 / n2;
    let tx = k.x / az;
    let ty = k.y / az;
    let tx2 = r.mul_add(tx, -point.x / f);
    let ty2 = r.mul_add(ty, -point.y / f);
    let inv = 1.0 / (tx2.mul_add(tx2, ty2.mul_add(ty2, 1.0))).sqrt();
    let out = Vec3::new(tx2 * inv, ty2 * inv, sz * inv);
    // Path correction.
    let fx = f * r * tx;
    let fy = f * r * ty;
    let fz = f; // |F| uses f²; the z sign does not matter for lengths.
    let f_len = (fx.mul_add(fx, fy.mul_add(fy, fz * fz))).sqrt();
    let dx = fx - point.x;
    let dy = fy - point.y;
    let fx_len = (dx.mul_add(dx, dy.mul_add(dy, fz * fz))).sqrt();
    let fdotx = fx.mul_add(point.x, fy * point.y);
    let xx = point.x.mul_add(point.x, point.y * point.y);
    let diff = (2.0f64.mul_add(fdotx, -xx)) / (f_len + fx_len);
    let delta = f.signum() * n2 * diff - n1 * point.x.mul_add(k.x, point.y * k.y);
    Some((out, delta))
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
            let (kp, delta) = ideal_lens(p, k, f, 1.0, 1.0).unwrap();
            let t = (focus.z - p.z) / kp.z;
            assert!((p + kp * t).max_abs_diff(focus) < 1e-12);
            // Path from the incoming plane wavefront through the origin.
            opls.push(p.dot(k) + delta + t);
        }
        for o in &opls {
            assert!((o - opls[0]).abs() < 1e-12, "{opls:?}");
        }
    }
}
