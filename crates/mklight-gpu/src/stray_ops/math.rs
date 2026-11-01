//! FTZ-aware double-float operations, mirrored in stray_ops.metal.
use crate::{Df32, FpEnv};
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};
#[derive(Clone, Copy)]
pub struct D<E: FpEnv>(pub Df32, PhantomData<E>);
impl<E: FpEnv> D<E> {
    pub fn from(x: Df32) -> Self {
        Self(x, PhantomData)
    }
    pub fn f(x: f32) -> Self {
        Self::from(Df32::from_f32(x))
    }
    pub fn v(self) -> f32 {
        E::add(self.0.hi, self.0.lo)
    }
    pub fn sqrt(self) -> Self {
        Self::from(self.0.sqrt_in::<E>())
    }
}
impl<E: FpEnv> Add for D<E> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::from(self.0.add_in::<E>(b.0))
    }
}
impl<E: FpEnv> Sub for D<E> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::from(self.0.add_in::<E>(-b.0))
    }
}
impl<E: FpEnv> Mul for D<E> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::from(self.0.mul_in::<E>(b.0))
    }
}
impl<E: FpEnv> Div for D<E> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self::from(self.0.div_in::<E>(b.0))
    }
}
impl<E: FpEnv> Neg for D<E> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from(-self.0)
    }
}
pub type V<E> = [D<E>; 3];
pub fn vec<E: FpEnv>(a: [Df32; 3]) -> V<E> {
    a.map(D::from)
}
pub fn add<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    std::array::from_fn(|i| a[i] + b[i])
}
pub fn sub<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    std::array::from_fn(|i| a[i] - b[i])
}
pub fn scale<E: FpEnv>(a: V<E>, b: D<E>) -> V<E> {
    a.map(|x| x * b)
}
pub fn dot<E: FpEnv>(a: V<E>, b: V<E>) -> D<E> {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn unit<E: FpEnv>(a: V<E>) -> V<E> {
    scale(a, D::f(1.) / dot(a, a).sqrt())
}
pub fn rotate<E: FpEnv>(m: [Df32; 9], a: V<E>) -> V<E> {
    std::array::from_fn(|i| {
        D::from(m[i]) * a[0] + D::from(m[3 + i]) * a[1] + D::from(m[6 + i]) * a[2]
    })
}
pub fn sincos<E: FpEnv>(x: D<E>) -> (D<E>, D<E>) {
    let tau = D::from(Df32::new(std::f32::consts::TAU, -1.7484555e-7));
    let r = x - tau * D::f((x / tau).v().round());
    let rr = -(r * r);
    let (mut st, mut ct) = (r, D::f(1.));
    let (mut s, mut c) = (st, ct);
    for i in 1..=13 {
        st = st * rr / D::f((2 * i * (2 * i + 1)) as f32);
        s = s + st;
        ct = ct * rr / D::f(((2 * i - 1) * 2 * i) as f32);
        c = c + ct;
    }
    (s, c)
}
