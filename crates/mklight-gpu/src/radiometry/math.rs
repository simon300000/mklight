//! Double-float complex algebra shared with radiometry.metal. Transcendental
//! arguments are reduced before evaluation; no hardware f64 is required.
use crate::Df32 as D;
use std::ops::{Add, Div, Mul, Neg, Sub};
pub fn d(x: f32) -> D {
    D::from_f32(x)
}
pub fn value(x: D) -> f32 {
    x.hi + x.lo
}
pub fn positive(x: D) -> D {
    if value(x) > 0.0 { x } else { D::ZERO }
}
pub fn min(a: D, b: D) -> D {
    if value(a - b) < 0.0 { a } else { b }
}
pub fn exp(x: D) -> D {
    let ln2 = D::new(std::f32::consts::LN_2, -1.9046542e-9);
    let k = (value(x / ln2)).round();
    let r = x - ln2 * d(k);
    let mut t = d(1.0);
    let mut s = t;
    for i in 1..=14 {
        t = t * r / d(i as f32);
        s = s + t;
    }
    s * d(2.0f32.powi(k as i32))
}
pub fn sincos(x: D) -> (D, D) {
    let tau = D::new(std::f32::consts::TAU, -1.7484555e-7);
    let r = x - tau * d(value(x / tau).round());
    let rr = -(r * r);
    let mut st = r;
    let mut ct = d(1.0);
    let mut s = st;
    let mut c = ct;
    for i in 1..=13 {
        st = st * rr / d((2 * i * (2 * i + 1)) as f32);
        s = s + st;
        ct = ct * rr / d(((2 * i - 1) * 2 * i) as f32);
        c = c + ct;
    }
    (s, c)
}
#[derive(Clone, Copy, Debug, Default)]
pub struct C {
    pub re: D,
    pub im: D,
}
impl C {
    pub fn new(re: D, im: D) -> Self {
        Self { re, im }
    }
    pub fn real(x: D) -> Self {
        Self::new(x, D::ZERO)
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn norm2(self) -> D {
        self.re * self.re + self.im * self.im
    }
    pub fn norm(self) -> D {
        self.norm2().sqrt()
    }
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        if value(r) == 0.0 {
            return Self::default();
        }
        let t = ((r + if value(self.re) >= 0.0 {
            self.re
        } else {
            -self.re
        }) * d(0.5))
        .sqrt();
        if value(self.re) >= 0.0 {
            Self::new(t, self.im / (d(2.0) * t))
        } else {
            let v = if value(self.im) < 0.0 { -t } else { t };
            Self::new(self.im / (d(2.0) * v), v)
        }
    }
    pub fn sincos(self) -> (Self, Self) {
        let (s, c) = sincos(self.re);
        let e = exp(self.im);
        let inv = d(1.0) / e;
        let ch = (e + inv) * d(0.5);
        let sh = (e - inv) * d(0.5);
        (Self::new(s * ch, c * sh), Self::new(c * ch, -s * sh))
    }
}
impl Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Neg for C {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Div for C {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let q = b.norm2();
        let z = self * b.conj();
        Self::new(z.re / q, z.im / q)
    }
}
pub fn nc(n: C, invariant: D) -> C {
    let mut z = (n * n - C::real(invariant * invariant)).sqrt();
    if value(z.im) > 0.0 || (value(z.im) == 0.0 && value(z.re) < 0.0) {
        z = -z;
    }
    z
}
pub type Matrix = [C; 4];
pub fn identity() -> Matrix {
    [C::real(d(1.0)), C::default(), C::default(), C::real(d(1.0))]
}
pub fn product(a: Matrix, b: Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}
