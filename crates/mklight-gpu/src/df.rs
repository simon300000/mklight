//! Double-float ("double-word") arithmetic in f32: a value is the unevaluated
//! sum `hi + lo` with `|lo| ≤ ulp(hi)/2`, giving ≈ 48 significand bits on
//! hardware without FP64 (CONVENTIONS §7.4).
//!
//! This module is the **specification and CPU emulation** of the GPU
//! primitives in `crates/mklight-metal/shaders/mklight_common.h`. Every
//! function is written operation for operation like its MSL twin, so with
//! IEEE round-to-nearest `+ − ×`, a true fused `fma` and no compiler
//! reassociation or contraction, CPU and GPU produce bit-identical results
//! (tested on the device). Division and square root use one hardware `/` or
//! `sqrt` as a first approximation and correct it with exact residuals, so
//! their *accuracy* does not depend on the hardware rounding those correctly;
//! bitwise agreement for them additionally requires a correctly rounded `/`
//! and `sqrt` (reported by the backend's numerics probe).
//!
//! Algorithms and error bounds (u = 2⁻²⁴) are from Joldes, Muller & Popescu,
//! "Tight and rigorous error bounds for basic building blocks of double-word
//! arithmetic", ACM TOMS 44(2), 2017:
//!
//! | operation | algorithm | relative error bound |
//! |---|---|---|
//! | `two_sum`, `fast_two_sum`, `two_prod` | error-free transforms | exact |
//! | `add_f32` | DWPlusFP (Alg. 4) | 2u² |
//! | `+` (`Add`) | AccurateDWPlusDW (Alg. 6) | 3u² + 13u³ |
//! | `mul_f32` | DWTimesFP3 (Alg. 9) | 2u² |
//! | `*` (`Mul`) | DWTimesDW3 (Alg. 12) | 4u² |
//! | `/` (`Div`) | DWDivDW2 (Alg. 17) | 15u² + 56u³ |
//! | `sqrt` | one Newton/Karp correction of `sqrt(hi)` | ≈ 4u² (tested) |
//!
//! Error-free transforms are destroyed by fast-math (reassociation turns
//! `(s − a) − b` into `0`) and by implicit FMA contraction; the Metal backend
//! compiles with both disabled and proves it at start-up.
//!
//! Every algorithm also exists as a `*_in::<E: FpEnv>` function, so the
//! emulation reproduces devices that flush subnormals to zero
//! ([`crate::fenv::Ftz`]); the plain functions and operators are the IEEE
//! instantiation. Under flush-to-zero the transforms stay exact as long as no
//! result (including the error terms) falls below `f32::MIN_POSITIVE`, so
//! double-float values keep their full ≈ 2⁻⁴⁸ relative accuracy for
//! `|hi| ≥ 2⁻⁷⁸` (optical quantities are many decades above that).

use std::ops::{Add, Div, Mul, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::fenv::{FpEnv, Ieee};

/// A double-float number `hi + lo`. Layout-compatible with MSL `float2`
/// (`x = hi`, `y = lo`; 8-byte size and alignment).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct Df32 {
    pub hi: f32,
    pub lo: f32,
}

/// `s + t = a + b` exactly, for any finite `a`, `b` (Knuth's 2Sum, 6 flops).
#[inline]
pub fn two_sum(a: f32, b: f32) -> (f32, f32) {
    two_sum_in::<Ieee>(a, b)
}

/// `s + t = a + b` exactly, provided `|a| ≥ |b|` or `a == 0` (Dekker, 3 flops).
#[inline]
pub fn fast_two_sum(a: f32, b: f32) -> (f32, f32) {
    fast_two_sum_in::<Ieee>(a, b)
}

/// `p + e = a·b` exactly (barring underflow), using one fused multiply-add.
#[inline]
pub fn two_prod(a: f32, b: f32) -> (f32, f32) {
    two_prod_in::<Ieee>(a, b)
}

/// [`two_sum`] in environment `E` (exact unless a result underflows under FTZ).
#[inline]
pub fn two_sum_in<E: FpEnv>(a: f32, b: f32) -> (f32, f32) {
    let s = E::add(a, b);
    let a1 = E::sub(s, b);
    let b1 = E::sub(s, a1);
    let da = E::sub(a, a1);
    let db = E::sub(b, b1);
    (s, E::add(da, db))
}

/// [`fast_two_sum`] in environment `E`.
#[inline]
pub fn fast_two_sum_in<E: FpEnv>(a: f32, b: f32) -> (f32, f32) {
    let s = E::add(a, b);
    let z = E::sub(s, a);
    (s, E::sub(b, z))
}

/// [`two_prod`] in environment `E`.
#[inline]
pub fn two_prod_in<E: FpEnv>(a: f32, b: f32) -> (f32, f32) {
    let p = E::mul(a, b);
    (p, E::fma(a, b, -p))
}

impl Df32 {
    pub const ZERO: Df32 = Df32 { hi: 0.0, lo: 0.0 };

    #[inline]
    pub const fn new(hi: f32, lo: f32) -> Self {
        Df32 { hi, lo }
    }
    #[inline]
    pub const fn from_f32(x: f32) -> Self {
        Df32 { hi: x, lo: 0.0 }
    }
    /// Nearest double-float to an f64: `hi = fl32(x)`, `lo = fl32(x − hi)`.
    /// `x − hi` is exact in f64, so only `lo` is rounded (≈ 2⁻⁴⁸ relative).
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        let hi = x as f32;
        Df32 {
            hi,
            lo: (x - hi as f64) as f32,
        }
    }
    /// Exact value as f64 (both parts are exactly representable).
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
    #[inline]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    /// DWPlusFP: `self + y`.
    #[inline]
    pub fn add_f32(self, y: f32) -> Df32 {
        self.add_f32_in::<Ieee>(y)
    }
    /// DWTimesFP3: `self · y`.
    #[inline]
    pub fn mul_f32(self, y: f32) -> Df32 {
        self.mul_f32_in::<Ieee>(y)
    }
    /// Square root: `s = sqrt(hi)`, residual `x − s²` formed exactly with
    /// `two_prod` and Sterbenz, one Newton correction `e = r / 2s`. Returns
    /// NaN for negative input and 0 for 0.
    #[inline]
    pub fn sqrt(self) -> Df32 {
        self.sqrt_in::<Ieee>()
    }

    /// [`Df32::add_f32`] in environment `E`.
    #[inline]
    pub fn add_f32_in<E: FpEnv>(self, y: f32) -> Df32 {
        let (sh, sl) = two_sum_in::<E>(self.hi, y);
        let v = E::add(self.lo, sl);
        let (zh, zl) = fast_two_sum_in::<E>(sh, v);
        Df32::new(zh, zl)
    }

    /// [`Df32::mul_f32`] in environment `E`.
    #[inline]
    pub fn mul_f32_in<E: FpEnv>(self, y: f32) -> Df32 {
        let (ch, cl1) = two_prod_in::<E>(self.hi, y);
        let cl3 = E::fma(self.lo, y, cl1);
        let (zh, zl) = fast_two_sum_in::<E>(ch, cl3);
        Df32::new(zh, zl)
    }

    /// [`Df32::sqrt`] in environment `E`.
    #[inline]
    pub fn sqrt_in<E: FpEnv>(self) -> Df32 {
        if E::le(self.hi, 0.0) {
            return if E::eq(self.hi, 0.0) {
                Df32::ZERO
            } else {
                Df32::new(f32::NAN, f32::NAN)
            };
        }
        let s = E::sqrt(self.hi);
        let (ph, pl) = two_prod_in::<E>(s, s);
        let r = E::add(E::sub(E::sub(self.hi, ph), pl), self.lo);
        let e = E::div(r, E::mul(2.0, s));
        let (zh, zl) = fast_two_sum_in::<E>(s, e);
        Df32::new(zh, zl)
    }

    /// AccurateDWPlusDW (`+`) in environment `E`.
    #[inline]
    pub fn add_in<E: FpEnv>(self, y: Df32) -> Df32 {
        let (sh, sl) = two_sum_in::<E>(self.hi, y.hi);
        let (th, tl) = two_sum_in::<E>(self.lo, y.lo);
        let c = E::add(sl, th);
        let (vh, vl) = fast_two_sum_in::<E>(sh, c);
        let w = E::add(tl, vl);
        let (zh, zl) = fast_two_sum_in::<E>(vh, w);
        Df32::new(zh, zl)
    }

    /// DWTimesDW3 (`*`) in environment `E`.
    #[inline]
    pub fn mul_in<E: FpEnv>(self, y: Df32) -> Df32 {
        let (ch, cl1) = two_prod_in::<E>(self.hi, y.hi);
        let tl0 = E::mul(self.lo, y.lo);
        let tl1 = E::fma(self.hi, y.lo, tl0);
        let cl2 = E::fma(self.lo, y.hi, tl1);
        let cl3 = E::add(cl1, cl2);
        let (zh, zl) = fast_two_sum_in::<E>(ch, cl3);
        Df32::new(zh, zl)
    }

    /// DWDivDW2 (`/`) in environment `E`. `th = x.hi / y.hi` only needs to be
    /// within a few ulp; `x.hi − r.hi` is exact by Sterbenz's lemma.
    #[inline]
    pub fn div_in<E: FpEnv>(self, y: Df32) -> Df32 {
        let th = E::div(self.hi, y.hi);
        let r = y.mul_f32_in::<E>(th);
        let pi_h = E::sub(self.hi, r.hi);
        let dl = E::sub(self.lo, r.lo);
        let d = E::add(pi_h, dl);
        let tl = E::div(d, y.hi);
        let (zh, zl) = fast_two_sum_in::<E>(th, tl);
        Df32::new(zh, zl)
    }
}

/// AccurateDWPlusDW: `x + y`.
impl Add for Df32 {
    type Output = Df32;
    #[inline]
    fn add(self, y: Df32) -> Df32 {
        self.add_in::<Ieee>(y)
    }
}

impl Neg for Df32 {
    type Output = Df32;
    #[inline]
    fn neg(self) -> Df32 {
        Df32::new(-self.hi, -self.lo)
    }
}

/// `x + (−y)` (negation is exact).
impl Sub for Df32 {
    type Output = Df32;
    #[inline]
    fn sub(self, y: Df32) -> Df32 {
        self + (-y)
    }
}

/// DWTimesDW3: `x · y`.
impl Mul for Df32 {
    type Output = Df32;
    #[inline]
    fn mul(self, y: Df32) -> Df32 {
        self.mul_in::<Ieee>(y)
    }
}

/// DWDivDW2: `x / y`.
impl Div for Df32 {
    type Output = Df32;
    #[inline]
    fn div(self, y: Df32) -> Df32 {
        self.div_in::<Ieee>(y)
    }
}

/// Neumaier's improved Kahan summation in f32 (CONVENTIONS §7.5); twin of
/// `mkl_neumaier_add` in MSL. `value()` returns `sum + c` as a double-float,
/// which keeps the compensation's extra bits instead of rounding them away.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeumaierF32 {
    pub sum: f32,
    pub c: f32,
}

impl NeumaierF32 {
    #[inline]
    pub fn add(&mut self, x: f32) {
        self.add_in::<Ieee>(x)
    }
    #[inline]
    pub fn value(&self) -> Df32 {
        self.value_in::<Ieee>()
    }
    #[inline]
    pub fn add_in<E: FpEnv>(&mut self, x: f32) {
        let t = E::add(self.sum, x);
        if E::ge(self.sum.abs(), x.abs()) {
            self.c = E::add(self.c, E::add(E::sub(self.sum, t), x));
        } else {
            self.c = E::add(self.c, E::add(E::sub(x, t), self.sum));
        }
        self.sum = t;
    }
    #[inline]
    pub fn value_in<E: FpEnv>(&self) -> Df32 {
        let (h, l) = two_sum_in::<E>(self.sum, self.c);
        Df32::new(h, l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::rng::Philox;

    const U: f64 = 1.0 / 16_777_216.0; // 2^-24

    /// Random f32 with a spread of magnitudes and signs.
    fn rnd(r: &mut Philox, emin: i32, emax: i32) -> f32 {
        let m = r.uniform() * 2.0 - 1.0;
        let e = emin + ((emax - emin) as f64 * r.uniform()) as i32;
        (m * 2f64.powi(e)) as f32
    }
    fn rnd_df(r: &mut Philox, emin: i32, emax: i32) -> Df32 {
        let x = rnd(r, emin, emax) as f64;
        // A random low part within half an ulp of hi.
        Df32::from_f64(x * (1.0 + (r.uniform() - 0.5) * 2f64.powi(-23)))
    }
    fn is_normalized(z: Df32) -> bool {
        z.lo == 0.0 || (z.lo.abs() as f64) <= 0.5 * (z.hi.abs() as f64) * 2f64.powi(-23) * 1.0001
    }

    #[test]
    fn error_free_transforms_are_exact() {
        let mut r = Philox::new(1, 0);
        for _ in 0..200_000 {
            let a = rnd(&mut r, -30, 30);
            let b = rnd(&mut r, -30, 30);
            let (s, t) = two_sum(a, b);
            assert_eq!(s as f64 + t as f64, a as f64 + b as f64);
            assert_eq!(s, a + b);
            let (big, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
            assert_eq!(fast_two_sum(big, small), (s, t));
            let (p, e) = two_prod(a, b);
            // f32·f32 is exact in f64 (48-bit product).
            assert_eq!(p as f64 + e as f64, a as f64 * b as f64);
        }
    }

    #[test]
    fn from_to_f64_roundtrip() {
        let x = std::f64::consts::PI * 1234.5;
        let d = Df32::from_f64(x);
        assert!(((d.to_f64() - x) / x).abs() < 2f64.powi(-48));
        assert!(is_normalized(d));
    }

    /// Each operation against the f64 result, at its published bound
    /// (with the inputs' own exact f64 values as reference).
    #[test]
    fn operations_meet_error_bounds() {
        let mut r = Philox::new(2, 0);
        let (mut worst_add, mut worst_mul, mut worst_div, mut worst_sqrt) =
            (0f64, 0f64, 0f64, 0f64);
        let (mut worst_addf, mut worst_mulf) = (0f64, 0f64);
        for _ in 0..200_000 {
            let x = rnd_df(&mut r, -20, 20);
            let y = rnd_df(&mut r, -20, 20);
            let f = rnd(&mut r, -20, 20);
            let (xf, yf) = (x.to_f64(), y.to_f64());
            let rel = |z: Df32, exact: f64| {
                if exact == 0.0 {
                    z.to_f64().abs()
                } else {
                    ((z.to_f64() - exact) / exact).abs()
                }
            };
            // The f64 reference itself has error ≤ 2^-53 relative; the bounds
            // below are ≥ 3u² ≈ 1.8e-14, far above that.
            let s = x + y;
            // Addition error is relative to |x|+|y| when there is cancellation.
            let ea = (s.to_f64() - (xf + yf)).abs() / (xf.abs() + yf.abs());
            worst_add = worst_add.max(ea);
            assert!(is_normalized(s));
            let ef = (x.add_f32(f).to_f64() - (xf + f as f64)).abs() / (xf.abs() + f.abs() as f64);
            worst_addf = worst_addf.max(ef);
            worst_mul = worst_mul.max(rel(x * y, xf * yf));
            worst_mulf = worst_mulf.max(rel(x.mul_f32(f), xf * f as f64));
            worst_div = worst_div.max(rel(x / y, xf / yf));
            let xa = Df32::new(x.hi.abs(), x.lo * x.hi.signum());
            worst_sqrt = worst_sqrt.max(rel(xa.sqrt(), xa.to_f64().sqrt()));
        }
        let u2 = U * U;
        assert!(worst_add <= 3.0 * u2, "add {}", worst_add / u2);
        assert!(worst_addf <= 2.0 * u2, "add_f32 {}", worst_addf / u2);
        assert!(worst_mul <= 4.0 * u2, "mul {}", worst_mul / u2);
        assert!(worst_mulf <= 2.0 * u2, "mul_f32 {}", worst_mulf / u2);
        assert!(worst_div <= 15.0 * u2, "div {}", worst_div / u2);
        assert!(worst_sqrt <= 4.0 * u2, "sqrt {}", worst_sqrt / u2);
    }

    #[test]
    fn sqrt_edge_cases() {
        assert_eq!(Df32::ZERO.sqrt(), Df32::ZERO);
        assert!(Df32::from_f32(-1.0).sqrt().hi.is_nan());
        let two = Df32::from_f32(2.0).sqrt();
        assert!((two.to_f64() - std::f64::consts::SQRT_2).abs() < 1e-14);
    }

    /// The classic fast-math casualty: summing 1e6 values that plain f32
    /// cannot accumulate (each `1e-1` added to a sum near 1e5 loses bits).
    #[test]
    fn df_summation_beats_f32() {
        let n = 1_000_000;
        let x = 0.1f32;
        let exact = n as f64 * x as f64;
        let mut plain = 0f32;
        let mut df = Df32::ZERO;
        let mut kahan = NeumaierF32::default();
        for _ in 0..n {
            plain += x;
            df = df.add_f32(x);
            kahan.add(x);
        }
        let rel = |v: f64| ((v - exact) / exact).abs();
        assert!(rel(plain as f64) > 1e-3);
        assert!(rel(df.to_f64()) < 1e-14);
        // Single-level Neumaier in f32 is ~100x better than plain here but
        // far from double-float: its compensation `c` accumulates the
        // (systematic) rounding errors in f32 itself. The GPU reduction uses
        // it only for short per-thread runs and combines in double-float.
        let k = rel(kahan.value().to_f64());
        assert!(k < rel(plain as f64) / 100.0 && k > 1e-10, "{k}");
    }
}
