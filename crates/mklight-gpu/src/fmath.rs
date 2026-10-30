//! f32 utility math shared bit-for-bit with the GPU (`mklight_common.h`):
//! fma-based dot products, overflow-safe hypot, `1 − x²` in factored form,
//! the cancellation-free conic sag and the packed asphere evaluation
//! (CONVENTIONS §7). Each function matches its MSL twin operation for
//! operation; change both together.

use crate::consts::*;
use crate::layout::PackedSurface;

/// `a·b` with nested fma, in the same order as `mklight_core::Vec3::dot`:
/// `fma(a0, b0, fma(a1, b1, a2·b2))`.
#[inline]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

/// `1 − x²` as `(1 − x)(1 + x)`: exact factors near |x| = 1, where the direct
/// form cancels (cosines of near-normal rays, CONVENTIONS §7.3).
#[inline]
pub fn one_minus_sq(x: f32) -> f32 {
    (1.0 - x) * (1.0 + x)
}

/// `2^e` for normal exponents, built from the bit pattern (exact).
const fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}
const HYPOT_BIG: f32 = pow2(60);
const HYPOT_SMALL: f32 = pow2(-60);
const SCALE_DOWN: f32 = pow2(-80);
const SCALE_UP: f32 = pow2(80);

/// Power-of-two scale factors that bring `m` into [2^-60, 2^60], where the
/// squares of the scaled components cannot overflow or underflow. Scaling by
/// a power of two is exact.
#[inline]
fn hypot_scale(m: f32) -> (f32, f32) {
    if m > HYPOT_BIG {
        (SCALE_DOWN, SCALE_UP)
    } else if m < HYPOT_SMALL {
        (SCALE_UP, SCALE_DOWN)
    } else {
        (1.0, 1.0)
    }
}

/// `sqrt(x² + y²)` without spurious overflow/underflow; `fma` for the sum.
#[inline]
pub fn hypot2(x: f32, y: f32) -> f32 {
    let m = x.abs().max(y.abs());
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (s, inv) = hypot_scale(m);
    let (xs, ys) = (x * s, y * s);
    xs.mul_add(xs, ys * ys).sqrt() * inv
}

/// `|v|` for a 3-vector, overflow-safe like [`hypot2`].
#[inline]
pub fn hypot3(x: f32, y: f32, z: f32) -> f32 {
    let m = x.abs().max(y.abs()).max(z.abs());
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (s, inv) = hypot_scale(m);
    let v = [x * s, y * s, z * s];
    dot3(v, v).sqrt() * inv
}

/// Cancellation-free conic sag `c r² / (1 + √(1 − (1+k) c² r²))` in f32.
/// Returns NaN outside the conic's domain (the f64 definition returns `None`).
#[inline]
pub fn conic_sag(c: f32, k: f32, r2: f32) -> f32 {
    let cr2 = c * r2;
    let ck = (1.0 + k) * c;
    let arg = (-ck).mul_add(cr2, 1.0);
    if arg.is_nan() || arg < 0.0 {
        return f32::NAN;
    }
    cr2 / (1.0 + arg.sqrt())
}

/// Sag of a packed surface at `r²` (mm), the f32 evaluation of the layout
/// documented in `mklight_gpu_types.h`: conic part plus the normalised even
/// polynomial `ρ⁴ Σ a_j ρ^(2j)` by Horner in `ρ² = r²/R²`.
#[inline]
pub fn packed_sag(s: &PackedSurface, r2: f32) -> f32 {
    match s.shape {
        MKL_SHAPE_CONIC => conic_sag(s.curvature, s.conic, r2),
        MKL_SHAPE_EVEN_ASPHERE => {
            let base = conic_sag(s.curvature, s.conic, r2);
            let n = s.num_asphere as usize;
            if n == 0 {
                return base;
            }
            let rho2 = r2 * s.asphere_inv_norm2;
            let mut acc = s.asphere[n - 1];
            for j in (0..n - 1).rev() {
                acc = acc.mul_add(rho2, s.asphere[j]);
            }
            let rho4 = rho2 * rho2;
            acc.mul_add(rho4, base)
        }
        _ => 0.0,
    }
}

/// f32 uniform in the open interval (0, 1) from one Philox output word; the
/// GPU computes exactly this (CONVENTIONS §6).
///
/// `mklight_core::rng::u32_to_unit_f64` maps the top 24 bits `k` to
/// `(k + ½)·2⁻²⁴`. For `k ≥ 2²³` that value needs 25 significand bits and is
/// not an f32 (rounding to nearest could even produce 1.0). This mapping is
/// that value **rounded toward zero to f32**: identical for `k < 2²³`, and
/// `2⁻²⁵` (half an f32 ulp) below it otherwise. Every operation is exact, so
/// the result does not depend on rounding modes or hardware.
#[inline]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    let k = x >> 8;
    let half = if k < 0x80_0000 { pow2(-25) } else { 0.0 };
    (k as f32).mul_add(pow2(-24), half) // k·2^-24 + half, exact
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::SurfaceShape;
    use mklight_core::rng::u32_to_unit_f64;

    #[test]
    fn constants_are_exact_powers_of_two() {
        assert_eq!(HYPOT_BIG, 2f32.powi(60));
        assert_eq!(HYPOT_SMALL, 2f32.powi(-60));
        assert_eq!(SCALE_DOWN, 2f32.powi(-80));
        assert_eq!(SCALE_UP, 2f32.powi(80));
        assert_eq!(pow2(-24), 1.0 / 16_777_216.0);
    }

    #[test]
    fn hypot_extremes() {
        assert_eq!(hypot2(3.0, 4.0), 5.0);
        assert_eq!(hypot2(3e30, 4e30), 5e30);
        let t = hypot2(3e-30, 4e-30);
        assert!((t / 5e-30 - 1.0).abs() < 1e-6);
        assert_eq!(hypot3(2.0, 3.0, 6.0), 7.0);
        assert!((hypot3(2e35, 3e35, 6e35) / 7e35 - 1.0).abs() < 1e-6);
        assert_eq!(hypot2(0.0, -0.0), 0.0);
    }

    #[test]
    fn one_minus_sq_near_one() {
        let x = 1.0f32 - 2f32.powi(-20);
        let exact = 1.0 - (x as f64) * (x as f64);
        assert!(((one_minus_sq(x) as f64 - exact) / exact).abs() < 1e-7);
    }

    /// f32 conic sag against the f64 definition in the core, with relative
    /// error bounded by a few ulp of the sag over a full aperture.
    #[test]
    fn conic_sag_matches_core() {
        for &(r, k) in &[(50.0, 0.0), (-25.0, -1.0), (12.0, 0.5), (200.0, -2.3)] {
            let shape = SurfaceShape::Conic {
                curvature: 1.0 / r,
                conic: k,
            };
            let h_max = 0.9 * f64::abs(r) / f64::max(1.0 + k, 1.0).sqrt();
            for i in 0..=100 {
                let h = h_max * i as f64 / 100.0;
                let r2 = h * h;
                let exact = shape.sag(r2).unwrap();
                let got = conic_sag((1.0 / r) as f32, k as f32, r2 as f32) as f64;
                assert!(
                    (got - exact).abs() <= 6e-7 * exact.abs() + 1e-12,
                    "{r} {k} {h}"
                );
            }
        }
        assert!(conic_sag(0.1, 0.0, 200.0).is_nan());
    }

    #[test]
    fn unit_mapping_vs_core() {
        let half_ulp = 2f64.powi(-25);
        for x in [
            0u32,
            1,
            255,
            256,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FF00,
            u32::MAX,
        ]
        .into_iter()
        .chain((0..100_000u32).map(|i| i.wrapping_mul(0x9E37_79B9)))
        {
            let f = u32_to_unit_f32(x);
            assert!(f > 0.0 && f < 1.0, "{x:#x} -> {f}");
            let d = u32_to_unit_f64(x);
            if (x >> 8) < 0x80_0000 {
                assert_eq!(f as f64, d);
            } else {
                assert_eq!(f as f64, d - half_ulp);
            }
        }
    }
}
