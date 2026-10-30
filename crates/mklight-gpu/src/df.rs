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

use std::ops::{Add, Div, Mul, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

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
    let s = a + b;
    let a1 = s - b;
    let b1 = s - a1;
    let da = a - a1;
    let db = b - b1;
    (s, da + db)
}

/// `s + t = a + b` exactly, provided `|a| ≥ |b|` or `a == 0` (Dekker, 3 flops).
#[inline]
pub fn fast_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let z = s - a;
    (s, b - z)
}

/// `p + e = a·b` exactly (barring underflow), using one fused multiply-add.
#[inline]
pub fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p))
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
        let (sh, sl) = two_sum(self.hi, y);
        let v = self.lo + sl;
        let (zh, zl) = fast_two_sum(sh, v);
        Df32::new(zh, zl)
    }

    /// DWTimesFP3: `self · y`.
    #[inline]
    pub fn mul_f32(self, y: f32) -> Df32 {
        let (ch, cl1) = two_prod(self.hi, y);
        let cl3 = self.lo.mul_add(y, cl1);
        let (zh, zl) = fast_two_sum(ch, cl3);
        Df32::new(zh, zl)
    }

    /// Square root: `s = sqrt(hi)`, residual `x − s²` formed exactly with
    /// `two_prod` and Sterbenz, one Newton correction `e = r / 2s`. Returns
    /// NaN for negative input and 0 for 0.
    #[inline]
    pub fn sqrt(self) -> Df32 {
        if self.hi <= 0.0 {
            return if self.hi == 0.0 {
                Df32::ZERO
            } else {
                Df32::new(f32::NAN, f32::NAN)
            };
        }
        let s = self.hi.sqrt();
        let (ph, pl) = two_prod(s, s);
        let r = ((self.hi - ph) - pl) + self.lo;
        let e = r / (2.0 * s);
        let (zh, zl) = fast_two_sum(s, e);
        Df32::new(zh, zl)
    }
}

/// AccurateDWPlusDW: `x + y`.
impl Add for Df32 {
    type Output = Df32;
    #[inline]
    fn add(self, y: Df32) -> Df32 {
        let (sh, sl) = two_sum(self.hi, y.hi);
        let (th, tl) = two_sum(self.lo, y.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        let (zh, zl) = fast_two_sum(vh, w);
        Df32::new(zh, zl)
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
        let (ch, cl1) = two_prod(self.hi, y.hi);
        let tl0 = self.lo * y.lo;
        let tl1 = self.hi.mul_add(y.lo, tl0);
        let cl2 = self.lo.mul_add(y.hi, tl1);
        let cl3 = cl1 + cl2;
        let (zh, zl) = fast_two_sum(ch, cl3);
        Df32::new(zh, zl)
    }
}

/// DWDivDW2: `x / y`. `th = x.hi / y.hi` only needs to be within a few ulp;
/// `x.hi − r.hi` is exact by Sterbenz's lemma.
impl Div for Df32 {
    type Output = Df32;
    #[inline]
    fn div(self, y: Df32) -> Df32 {
        let th = self.hi / y.hi;
        let r = y.mul_f32(th);
        let pi_h = self.hi - r.hi;
        let dl = self.lo - r.lo;
        let d = pi_h + dl;
        let tl = d / y.hi;
        let (zh, zl) = fast_two_sum(th, tl);
        Df32::new(zh, zl)
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
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }
    #[inline]
    pub fn value(&self) -> Df32 {
        let (h, l) = two_sum(self.sum, self.c);
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
