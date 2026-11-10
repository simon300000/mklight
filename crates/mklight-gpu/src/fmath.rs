//! f32 utility math shared bit-for-bit with the GPU (`mklight_common.h`):
//! fma-based dot products, overflow-safe hypot, `1 − x²` in factored form,
//! the cancellation-free conic sag and the packed asphere evaluation
//! (CONVENTIONS §7). Each function matches its MSL twin operation for
//! operation; change both together.

use crate::consts::*;
use crate::fenv::{FpEnv, Ieee};
use crate::layout::PackedSurface;

pub use mklight_core::rng::u32_to_unit_f32;

/// `a·b` with nested fma, in the same order as `mklight_core::Vec3::dot`:
/// `fma(a0, b0, fma(a1, b1, a2·b2))`, in environment `E`.
#[inline]
pub fn dot3_in<E: FpEnv>(a: [f32; 3], b: [f32; 3]) -> f32 {
    E::fma(a[0], b[0], E::fma(a[1], b[1], E::mul(a[2], b[2])))
}

/// `1 − x²` as `(1 − x)(1 + x)`: exact factors near |x| = 1, where the direct
/// form cancels (cosines of near-normal rays, CONVENTIONS §7.3).
#[cfg(test)]
#[inline]
pub fn one_minus_sq(x: f32) -> f32 {
    one_minus_sq_in::<Ieee>(x)
}

/// `1 − x²` as `(1 − x)(1 + x)` in environment `E`.
#[inline]
pub fn one_minus_sq_in<E: FpEnv>(x: f32) -> f32 {
    E::mul(E::sub(1.0, x), E::add(1.0, x))
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
fn hypot_scale<E: FpEnv>(m: f32) -> (f32, f32) {
    if E::gt(m, HYPOT_BIG) {
        (SCALE_DOWN, SCALE_UP)
    } else if E::lt(m, HYPOT_SMALL) {
        (SCALE_UP, SCALE_DOWN)
    } else {
        (1.0, 1.0)
    }
}

/// Special values of hypot, IEEE 754 §9.2: any infinite component gives
/// `+inf` (even with a NaN), otherwise any NaN gives NaN, and all zeros give
/// `+0`. `m` is the largest magnitude (`max` drops NaN, hence the explicit
/// test). Results are constants, never a raw input, so a flush-to-zero device
/// cannot pass a subnormal through.
#[inline]
fn hypot_special<E: FpEnv>(m: f32, any_nan: bool) -> Option<f32> {
    if m == f32::INFINITY {
        Some(f32::INFINITY)
    } else if any_nan {
        Some(f32::NAN)
    } else if E::eq(m, 0.0) {
        Some(0.0)
    } else {
        None
    }
}

/// `sqrt(x² + y²)` without spurious overflow/underflow; `fma` for the sum.
#[inline]
pub fn hypot2(x: f32, y: f32) -> f32 {
    hypot2_in::<Ieee>(x, y)
}

/// [`hypot2`] in environment `E`.
#[inline]
pub fn hypot2_in<E: FpEnv>(x: f32, y: f32) -> f32 {
    let m = x.abs().max(y.abs());
    if let Some(v) = hypot_special::<E>(m, x.is_nan() || y.is_nan()) {
        return v;
    }
    let (s, inv) = hypot_scale::<E>(m);
    let (xs, ys) = (E::mul(x, s), E::mul(y, s));
    E::mul(E::sqrt(E::fma(xs, xs, E::mul(ys, ys))), inv)
}

/// [`hypot3_in`] in the IEEE environment.
#[cfg(test)]
#[inline]
pub fn hypot3(x: f32, y: f32, z: f32) -> f32 {
    hypot3_in::<Ieee>(x, y, z)
}

/// `|v|` for a 3-vector, overflow-safe like [`hypot2`], in environment `E`.
#[inline]
pub fn hypot3_in<E: FpEnv>(x: f32, y: f32, z: f32) -> f32 {
    let m = x.abs().max(y.abs()).max(z.abs());
    if let Some(v) = hypot_special::<E>(m, x.is_nan() || y.is_nan() || z.is_nan()) {
        return v;
    }
    let (s, inv) = hypot_scale::<E>(m);
    let v = [E::mul(x, s), E::mul(y, s), E::mul(z, s)];
    E::mul(E::sqrt(dot3_in::<E>(v, v)), inv)
}

/// `log2(e)`, `ln 2` split for Cody–Waite reduction (`LN2_HI` has 8 trailing
/// zero bits, so `k·LN2_HI` is exact for |k| < 256).
const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
const LN2_HI: f32 = f32::from_bits(0x3F31_7200);
const LN2_LO: f32 = f32::from_bits(0x35BF_BE8E);
/// `fl32(1/n!)`, n = 7 … 3, then ½, for the Taylor polynomial of `e^r` on
/// |r| ≤ ln2/2 (truncation error < 6e-9 relative, below the f32 rounding).
/// Written as bit patterns so the MSL twin uses identical constants.
const EXP_C: [f32; 6] = [
    f32::from_bits(0x3950_0D01),
    f32::from_bits(0x3AB6_0B61),
    f32::from_bits(0x3C08_8889),
    f32::from_bits(0x3D2A_AAAB),
    f32::from_bits(0x3E2A_AAAB),
    0.5,
];

/// `e^x` in f32 (a few ulp), specified here so that every backend and the
/// emulation agree bit for bit (platform `exp` implementations differ):
/// `k = rint(x·log2 e)`, `r = x − k ln2` (Cody–Waite with fma), a degree-7
/// Taylor polynomial by Horner, then `·2^(k/2)·2^(k − k/2)` (two exact
/// power-of-two factors, so no intermediate overflows). `+inf` above
/// 88.72284, `0` below −103.97208, NaN propagates.
#[inline]
pub fn exp(x: f32) -> f32 {
    exp_in::<Ieee>(x)
}

/// [`exp`] in environment `E`.
#[inline]
pub fn exp_in<E: FpEnv>(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if E::gt(x, 88.722_84) {
        return f32::INFINITY;
    }
    if E::lt(x, -103.972_08) {
        return 0.0;
    }
    let kf = E::mul(x, LOG2E).round_ties_even();
    let k = kf as i32;
    let r = E::fma(-kf, LN2_HI, x);
    let r = E::fma(-kf, LN2_LO, r);
    let mut p = EXP_C[0];
    for c in &EXP_C[1..] {
        p = E::fma(p, r, *c);
    }
    p = E::fma(p, r, 1.0);
    p = E::fma(p, r, 1.0);
    let k1 = k / 2;
    E::mul(E::mul(p, pow2(k1)), pow2(k - k1))
}

/// [`conic_sag_in`] in the IEEE environment.
#[cfg(test)]
#[inline]
pub fn conic_sag(c: f32, k: f32, r2: f32) -> f32 {
    conic_sag_in::<Ieee>(c, k, r2)
}

/// Cancellation-free conic sag `c r² / (1 + √(1 − (1+k) c² r²))` in f32, in
/// environment `E`. Returns NaN outside the conic's domain (the f64
/// definition returns `None`).
#[inline]
pub fn conic_sag_in<E: FpEnv>(c: f32, k: f32, r2: f32) -> f32 {
    let cr2 = E::mul(c, r2);
    let ck = E::mul(E::add(1.0, k), c);
    let arg = E::fma(-ck, cr2, 1.0);
    if arg.is_nan() || E::lt(arg, 0.0) {
        return f32::NAN;
    }
    E::div(cr2, E::add(1.0, E::sqrt(arg)))
}

/// [`packed_sag_in`] in the IEEE environment.
#[cfg(test)]
#[inline]
pub fn packed_sag(s: &PackedSurface, r2: f32) -> f32 {
    packed_sag_in::<Ieee>(s, r2)
}

/// Sag of a packed surface at `r²` (mm) in environment `E`, the f32
/// evaluation of the layout documented in `mklight_gpu_types.h`: conic part
/// plus the normalised even polynomial `ρ⁴ Σ a_j ρ^(2j)` by Horner in
/// `ρ² = r²/R²`. The surface must have passed `PackedSystem::validate`
/// (`num_asphere ≤ MAX_ASPHERE`).
#[inline]
pub fn packed_sag_in<E: FpEnv>(s: &PackedSurface, r2: f32) -> f32 {
    match s.shape {
        MKL_SHAPE_CONIC => conic_sag_in::<E>(s.curvature, s.conic, r2),
        MKL_SHAPE_EVEN_ASPHERE => {
            let base = conic_sag_in::<E>(s.curvature, s.conic, r2);
            let n = (s.num_asphere as usize).min(s.asphere.len());
            if n == 0 {
                return base;
            }
            let rho2 = E::mul(r2, s.asphere_inv_norm2);
            let mut acc = s.asphere[n - 1];
            for j in (0..n - 1).rev() {
                acc = E::fma(acc, rho2, s.asphere[j]);
            }
            let rho4 = E::mul(rho2, rho2);
            E::fma(acc, rho4, base)
        }
        _ => 0.0,
    }
}

/// Positive-y meridian sag including extended shapes (SURFACES §1).
/// `pool` is the validated system set's coefficient pool.
pub fn packed_sag_with_pool_in<E: FpEnv>(s: &PackedSurface, pool: &[crate::Df32], r2: f32) -> f32 {
    use crate::{fx::F, surface_math};
    let overlay = s.flags & MKL_SURF_HAS_SAG_OVERLAY != 0;
    if s.shape <= MKL_SHAPE_COORD_BREAK && !overlay {
        return packed_sag_in::<E>(s, r2);
    }
    let y = F::<E>::new(r2).sqrt();
    if s.shape == MKL_SHAPE_DIFFRACTIVE {
        if !surface_math::domain(s, pool, F::<E>::ZERO, y) {
            return f32::NAN;
        }
        let mut base = *s;
        base.shape = surface_math::data(s, pool)[MKL_COEFF_KIND as usize].hi as u32;
        if !overlay {
            return packed_sag_in::<E>(&base, r2);
        }
    }
    surface_math::evaluate(s, pool, F::<E>::ZERO, y).map_or(f32::NAN, |v| v.v.v())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::SurfaceShape;

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
        // NaN is never masked (IEEE: NaN unless a component is infinite).
        assert!(hypot2(f32::NAN, 0.0).is_nan());
        assert!(hypot2(0.0, f32::NAN).is_nan());
        assert!(hypot2(f32::NAN, 1.0).is_nan());
        assert!(hypot3(f32::NAN, 0.0, 0.0).is_nan());
        assert!(hypot3(0.0, 0.0, f32::NAN).is_nan());
        assert_eq!(hypot2(f32::NAN, f32::NEG_INFINITY), f32::INFINITY);
        assert_eq!(hypot3(0.0, f32::INFINITY, f32::NAN), f32::INFINITY);
        // Subnormal components: exact scaling under IEEE, zero under FTZ.
        let t = 2f32.powi(-140);
        assert_eq!(hypot2(3.0 * t, 4.0 * t), 5.0 * t);
        assert_eq!(hypot2_in::<crate::fenv::Ftz>(3.0 * t, 4.0 * t), 0.0);
        assert_eq!(hypot2_in::<crate::fenv::Ftz>(1e-39, 1e-39).to_bits(), 0);
    }

    #[test]
    fn exp_is_accurate() {
        let mut worst = 0.0f64;
        for i in 0..=200_000 {
            let x = -103.0 + 191.7 * i as f64 / 200_000.0;
            let got = exp(x as f32) as f64;
            let want = (x as f32 as f64).exp();
            if want < f32::MIN_POSITIVE as f64 {
                continue;
            }
            let ulp = (want as f32).abs() as f64 * 2f64.powi(-23);
            worst = worst.max((got - want).abs() / ulp);
        }
        assert!(worst < 4.0, "worst {worst} ulp");
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-200.0), 0.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(exp_in::<crate::fenv::Ftz>(-100.0), 0.0);
        assert!(exp(-100.0) > 0.0);
        assert_eq!(LOG2E, std::f32::consts::LOG2_E);
        for (c, n) in EXP_C.iter().zip([5040.0f32, 720.0, 120.0, 24.0, 6.0, 2.0]) {
            assert_eq!(*c, 1.0 / n);
        }
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

    /// The GPU mapping is the core's canonical f32 mapping.
    #[test]
    fn unit_mapping_is_cores() {
        for x in [0u32, 255, 256, 0x7FFF_FFFF, 0x8000_0000, u32::MAX] {
            assert_eq!(u32_to_unit_f32(x), mklight_core::rng::u32_to_unit_f32(x));
            let k = x >> 8;
            let half = if k < 0x80_0000 { pow2(-25) } else { 0.0 };
            assert_eq!(u32_to_unit_f32(x), (k as f32).mul_add(pow2(-24), half));
        }
    }
}
