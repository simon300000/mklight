//! Environment-typed f32 scalars for writing kernel specifications that read
//! like their MSL twins.
//!
//! [`F<E>`] is an `f32` whose arithmetic and comparisons go through the
//! [`FpEnv`] `E` (IEEE or flush-to-zero), so an expression such as
//! `a * b + c` performs exactly the operations `E::mul` then `E::add`, in the
//! order written, like MSL compiled without contraction. Fusion is explicit
//! ([`F::fma`]). Bit operations (`abs`, negation, `min`/`max`, `copysign`,
//! finiteness tests) move bits unchanged, as on the GPU.
//!
//! [`D<E>`] is a double-float (`hi + lo`, [`Df32`]) with the
//! `mklight_common.h` algorithms: `D + D` is AccurateDWPlusDW, `D + F`
//! DWPlusFP, `D * F` DWTimesFP3, `D * D` DWTimesDW3, `D / D` DWDivDW2.

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::df::{Df32, two_prod_in, two_sum_in};
use crate::fenv::FpEnv;

/// An f32 computed in environment `E`.
#[repr(transparent)]
pub struct F<E: FpEnv>(pub f32, PhantomData<E>);

impl<E: FpEnv> Clone for F<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: FpEnv> Copy for F<E> {}

impl<E: FpEnv> fmt::Debug for F<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// `F::<E>::new(x)` without the type noise.
#[inline(always)]
pub fn f<E: FpEnv>(x: f32) -> F<E> {
    F(x, PhantomData)
}

impl<E: FpEnv> F<E> {
    pub const ZERO: Self = F(0.0, PhantomData);
    pub const ONE: Self = F(1.0, PhantomData);

    #[inline(always)]
    pub fn new(x: f32) -> Self {
        F(x, PhantomData)
    }
    /// The raw bits as an f32.
    #[inline(always)]
    pub fn v(self) -> f32 {
        self.0
    }
    /// `self · b + c` with one rounding (MSL `fma`).
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        f(E::fma(self.0, b.0, c.0))
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        f(E::sqrt(self.0))
    }
    /// Bit operation (MSL `fabs`).
    #[inline(always)]
    pub fn abs(self) -> Self {
        f(self.0.abs())
    }
    /// Bit operation with Rust/MSL `fmin` semantics (a NaN operand is ignored).
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        f(self.0.min(o.0))
    }
    /// See [`F::min`].
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        f(self.0.max(o.0))
    }
    /// Magnitude of `self` with the sign of `s` (bit operation).
    #[inline(always)]
    pub fn copysign(self, s: Self) -> Self {
        f(self.0.copysign(s.0))
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.0.is_nan()
    }
}

macro_rules! binop {
    ($tr:ident, $m:ident, $e:ident) => {
        impl<E: FpEnv> $tr for F<E> {
            type Output = F<E>;
            #[inline(always)]
            fn $m(self, o: F<E>) -> F<E> {
                f(E::$e(self.0, o.0))
            }
        }
        impl<E: FpEnv> $tr<f32> for F<E> {
            type Output = F<E>;
            #[inline(always)]
            fn $m(self, o: f32) -> F<E> {
                f(E::$e(self.0, o))
            }
        }
        impl<E: FpEnv> $tr<F<E>> for f32 {
            type Output = F<E>;
            #[inline(always)]
            fn $m(self, o: F<E>) -> F<E> {
                f(E::$e(self, o.0))
            }
        }
    };
}
binop!(Add, add, add);
binop!(Sub, sub, sub);
binop!(Mul, mul, mul);
binop!(Div, div, div);

impl<E: FpEnv> Neg for F<E> {
    type Output = F<E>;
    /// Sign flip (bit operation).
    #[inline(always)]
    fn neg(self) -> F<E> {
        f(-self.0)
    }
}

impl<E: FpEnv> PartialEq for F<E> {
    #[inline(always)]
    fn eq(&self, o: &Self) -> bool {
        E::eq(self.0, o.0)
    }
}
impl<E: FpEnv> PartialEq<f32> for F<E> {
    #[inline(always)]
    fn eq(&self, o: &f32) -> bool {
        E::eq(self.0, *o)
    }
}
impl<E: FpEnv> PartialOrd for F<E> {
    #[inline(always)]
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        E::d(self.0).partial_cmp(&E::d(o.0))
    }
    #[inline(always)]
    fn lt(&self, o: &Self) -> bool {
        E::lt(self.0, o.0)
    }
    #[inline(always)]
    fn le(&self, o: &Self) -> bool {
        E::le(self.0, o.0)
    }
    #[inline(always)]
    fn gt(&self, o: &Self) -> bool {
        E::gt(self.0, o.0)
    }
    #[inline(always)]
    fn ge(&self, o: &Self) -> bool {
        E::ge(self.0, o.0)
    }
}
impl<E: FpEnv> PartialOrd<f32> for F<E> {
    #[inline(always)]
    fn partial_cmp(&self, o: &f32) -> Option<Ordering> {
        E::d(self.0).partial_cmp(&E::d(*o))
    }
    #[inline(always)]
    fn lt(&self, o: &f32) -> bool {
        E::lt(self.0, *o)
    }
    #[inline(always)]
    fn le(&self, o: &f32) -> bool {
        E::le(self.0, *o)
    }
    #[inline(always)]
    fn gt(&self, o: &f32) -> bool {
        E::gt(self.0, *o)
    }
    #[inline(always)]
    fn ge(&self, o: &f32) -> bool {
        E::ge(self.0, *o)
    }
}

/// A double-float `hi + lo` computed in environment `E` (MSL `float2`).
#[repr(transparent)]
pub struct D<E: FpEnv>(pub Df32, PhantomData<E>);

impl<E: FpEnv> Clone for D<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: FpEnv> Copy for D<E> {}

impl<E: FpEnv> fmt::Debug for D<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<E: FpEnv> D<E> {
    pub const ZERO: Self = D(Df32::ZERO, PhantomData);

    #[inline(always)]
    pub fn new(hi: f32, lo: f32) -> Self {
        D(Df32::new(hi, lo), PhantomData)
    }
    #[inline(always)]
    pub fn from_df(x: Df32) -> Self {
        D(x, PhantomData)
    }
    /// `(x, 0)`.
    #[inline(always)]
    pub fn from_f(x: F<E>) -> Self {
        D::new(x.0, 0.0)
    }
    #[inline(always)]
    pub fn hi(self) -> F<E> {
        f(self.0.hi)
    }
    #[inline(always)]
    pub fn lo(self) -> F<E> {
        f(self.0.lo)
    }
    /// The exact product `a · b` (`mkl_two_prod`).
    #[inline(always)]
    pub fn two_prod(a: F<E>, b: F<E>) -> Self {
        let (p, e) = two_prod_in::<E>(a.0, b.0);
        D::new(p, e)
    }
    /// Double-float square root (`mkl_df_sqrt`).
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        D::from_df(self.0.sqrt_in::<E>())
    }
    /// The exact sum `a + b` (`mkl_two_sum`).
    #[inline(always)]
    pub fn two_sum(a: F<E>, b: F<E>) -> Self {
        let (s, e) = two_sum_in::<E>(a.0, b.0);
        D::new(s, e)
    }
}

impl<E: FpEnv> Add for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn add(self, o: D<E>) -> D<E> {
        D::from_df(self.0.add_in::<E>(o.0))
    }
}
impl<E: FpEnv> Add<F<E>> for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn add(self, o: F<E>) -> D<E> {
        D::from_df(self.0.add_f32_in::<E>(o.0))
    }
}
impl<E: FpEnv> Neg for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn neg(self) -> D<E> {
        D::new(-self.0.hi, -self.0.lo)
    }
}
impl<E: FpEnv> Sub for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn sub(self, o: D<E>) -> D<E> {
        self + (-o)
    }
}
impl<E: FpEnv> Sub<F<E>> for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn sub(self, o: F<E>) -> D<E> {
        self + (-o)
    }
}
impl<E: FpEnv> Mul<F<E>> for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn mul(self, o: F<E>) -> D<E> {
        D::from_df(self.0.mul_f32_in::<E>(o.0))
    }
}
impl<E: FpEnv> Mul<f32> for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn mul(self, o: f32) -> D<E> {
        D::from_df(self.0.mul_f32_in::<E>(o))
    }
}
impl<E: FpEnv> Mul for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn mul(self, o: D<E>) -> D<E> {
        D::from_df(self.0.mul_in::<E>(o.0))
    }
}
impl<E: FpEnv> Div for D<E> {
    type Output = D<E>;
    #[inline(always)]
    fn div(self, o: D<E>) -> D<E> {
        D::from_df(self.0.div_in::<E>(o.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fenv::{Ftz, Ieee};

    #[test]
    fn operators_follow_the_environment() {
        let a = f::<Ieee>(1e-20);
        assert_eq!((a * a).v(), 1e-20f32 * 1e-20f32);
        let b = f::<Ftz>(1e-20);
        assert_eq!((b * b).v().to_bits(), 0);
        // No contraction: a·b + c rounds twice.
        // (1 + 2⁻¹²)² = 1 + 2⁻¹¹ + 2⁻²⁴ rounds (tie to even) to 1 + 2⁻¹¹.
        let x = f::<Ieee>(1.0 + 2f32.powi(-12));
        let c = f::<Ieee>(-(1.0 + 2f32.powi(-11)));
        assert_eq!((x * x + c).v(), 0.0);
        assert_eq!(x.fma(x, c).v(), 2f32.powi(-24));
        // Comparisons read subnormals as zero under FTZ.
        assert!(f::<Ftz>(1e-39) == 0.0);
        assert!(f::<Ieee>(1e-39) > 0.0);
        let s = D::<Ieee>::two_sum(f(1.0), f(2f32.powi(-30)));
        assert_eq!((s.hi().v(), s.lo().v()), (1.0, 2f32.powi(-30)));
        assert_eq!((s - s).hi().v(), 0.0);
    }
}
