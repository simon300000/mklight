//! Packed sag and gradient specification; MSL twin: `surface_math.h`.
//! First-order Cartesian jets differentiate the reference's Horner, Jacobi
//! and Forbes recurrences. The same expression runs in f32 for decisions
//! and double-float for refinement and normals.
use crate::{
    consts::*,
    df::Df32,
    fenv::FpEnv,
    fx::{D, F},
    layout::PackedSurface,
};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub(crate) trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn parts(hi: f32, lo: f32) -> Self;
    fn sqrt(self) -> Self;
    fn le(self, rhs: f32) -> bool;
    fn zero(self) -> bool;
}
impl<E: FpEnv> Scalar for F<E> {
    fn parts(hi: f32, _: f32) -> Self {
        F::new(hi)
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn le(self, rhs: f32) -> bool {
        self <= rhs
    }
    fn zero(self) -> bool {
        self == 0.0
    }
}
impl<E: FpEnv> Scalar for D<E> {
    fn parts(hi: f32, lo: f32) -> Self {
        D::new(hi, lo)
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn le(self, rhs: f32) -> bool {
        (self - F::new(rhs)).hi() <= 0.0
    }
    fn zero(self) -> bool {
        self.hi() == 0.0
    }
}
fn c<R: Scalar>(v: f32) -> R {
    R::parts(v, 0.0)
}

#[derive(Clone, Copy)]
pub(crate) struct Jet<R> {
    pub v: R,
    pub x: R,
    pub y: R,
}
impl<R: Scalar> Jet<R> {
    fn constant(v: R) -> Self {
        Self {
            v,
            x: c(0.0),
            y: c(0.0),
        }
    }
    fn variable(v: R, x: bool) -> Self {
        Self {
            v,
            x: c(if x { 1.0 } else { 0.0 }),
            y: c(if x { 0.0 } else { 1.0 }),
        }
    }
    fn chain(self, v: R, d: R) -> Self {
        Self {
            v,
            x: d * self.x,
            y: d * self.y,
        }
    }
    fn scale(self, v: R) -> Self {
        Self {
            v: self.v * v,
            x: self.x * v,
            y: self.y * v,
        }
    }
    fn add(self, v: R) -> Self {
        Self {
            v: self.v + v,
            ..self
        }
    }
    fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        self.chain(v, c::<R>(0.5) / v)
    }
    fn recip(self) -> Self {
        let r = c::<R>(1.0) / self.v;
        self.chain(r, -r * r)
    }
}
impl<R: Scalar> Add for Jet<R> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            x: self.x + b.x,
            y: self.y + b.y,
        }
    }
}
impl<R: Scalar> Neg for Jet<R> {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(c(-1.0))
    }
}
impl<R: Scalar> Sub for Jet<R> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl<R: Scalar> Mul for Jet<R> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            x: self.x * b.v + self.v * b.x,
            y: self.y * b.v + self.v * b.y,
        }
    }
}
impl<R: Scalar> Div for Jet<R> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: Self) -> Self {
        self * b.recip()
    }
}
fn load<R: Scalar>(pool: &[Df32], i: usize) -> R {
    R::parts(pool[i].hi, pool[i].lo)
}
fn poly<R: Scalar>(pool: &[Df32], start: usize, n: usize, t: Jet<R>) -> Jet<R> {
    let mut p = Jet::constant(c(0.0));
    for j in (0..n).rev() {
        p = (p * t).add(load(pool, start + j));
    }
    p
}
fn even<R: Scalar>(pool: &[Df32], start: usize, n: usize, t: Jet<R>) -> Jet<R> {
    poly(pool, start, n, t) * t * t
}
fn conic<R: Scalar>(curv: R, k: R, t: Jet<R>) -> Option<Jet<R>> {
    let arg = c::<R>(1.0) - (c::<R>(1.0) + k) * curv * curv * t.v;
    if !c::<R>(0.0).le_value(arg) {
        return None;
    }
    let w = arg.sqrt();
    Some(t.chain(curv * t.v / (c::<R>(1.0) + w), curv / (c::<R>(2.0) * w)))
}
// Comparison expressed via subtraction works for both scalar precisions and
// sends f32 comparisons through FpEnv (including FTZ).
trait Compare: Scalar {
    fn le_value(self, rhs: Self) -> bool {
        (self - rhs).le(0.0)
    }
}
impl<R: Scalar> Compare for R {}

fn normalized<R: Scalar>(x: Jet<R>, y: Jet<R>, radius: R) -> Option<(Jet<R>, Jet<R>, Jet<R>)> {
    let inv = c::<R>(1.0) / radius;
    let (u, v) = (x.scale(inv), y.scale(inv));
    let s = u * u + v * v;
    let margin = R::parts(1.0, 2e-12);
    if !s.v.le_value(margin) {
        return None;
    }
    Some((u, v, s))
}
fn jacobi<R: Scalar>(n: usize, b: R, t: Jet<R>) -> Jet<R> {
    let mut p0 = Jet::constant(c(1.0));
    if n == 0 {
        return p0;
    }
    let mut p1 = t.add(c(-1.0)).scale((b + c(2.0)) / c(2.0)).add(c(1.0));
    for k in 1..n {
        let k = c::<R>(k as f32);
        let s = c::<R>(2.0) * k + b;
        let a1 = c::<R>(2.0) * (k + c(1.0)) * (k + b + c(1.0)) * s;
        let a2 = t.scale((s + c(2.0)) * s).add(-(b * b)).scale(s + c(1.0));
        let a3 = c::<R>(2.0) * k * (k + b) * (s + c(2.0));
        let next = (a2 * p1 - p0.scale(a3)).scale(c::<R>(1.0) / a1);
        p0 = p1;
        p1 = next;
    }
    p1
}
fn qcon_ab<R: Scalar>(k: usize, s: Jet<R>) -> (Jet<R>, R) {
    let k = c::<R>(k as f32);
    let t = c::<R>(2.0) * k + c(4.0);
    let den = c::<R>(2.0) * (k + c(1.0)) * (k + c(5.0)) * t;
    (
        s.scale(c(2.0))
            .add(c(-1.0))
            .scale((t + c(2.0)) * t)
            .add(c(-16.0))
            .scale((t + c(1.0)) / den),
        c::<R>(2.0) * k * (k + c(4.0)) * (t + c(2.0)) / den,
    )
}
fn qcon<R: Scalar>(pool: &[Df32], n: usize, s: Jet<R>) -> Jet<R> {
    let (mut b1, mut b2) = (Jet::constant(c(0.0)), Jet::constant(c(0.0)));
    for k in (0..n).rev() {
        let b = (qcon_ab(k, s).0 * b1 - b2.scale(qcon_ab(k + 1, s).1)).add(load(pool, k));
        b2 = b1;
        b1 = b;
    }
    b1 * s * s
}
fn qbfs<R: Scalar>(pool: &[Df32], n: usize, s: Jet<R>) -> Jet<R> {
    let (mut a1, mut a2) = (Jet::constant(c(0.0)), Jet::constant(c(0.0)));
    for m in (0..n).rev() {
        let a = (s.scale(c(-4.0)).add(c(2.0)) * a1 - a2).add(load(pool, m));
        a2 = a1;
        a1 = a;
    }
    (a1 + a2).scale(c(2.0))
}
fn zernike<R: Scalar>(
    pool: &[Df32],
    n: usize,
    standard: bool,
    x: Jet<R>,
    y: Jet<R>,
    s: Jet<R>,
) -> Jet<R> {
    let mut z = Jet::constant(c(0.0));
    for j in 1..=n {
        let a: R = load(pool, j - 1);
        if a.zero() {
            continue;
        }
        let (degree, m) = if standard {
            let mut degree = 0;
            while (degree + 1) * (degree + 2) / 2 < j {
                degree += 1;
            }
            let k = j - degree * (degree + 1) / 2 - 1;
            let am = if degree % 2 == 0 {
                2 * k.div_ceil(2)
            } else {
                2 * (k / 2) + 1
            } as i32;
            (degree, if j % 2 == 0 { am } else { -am })
        } else if j == 37 {
            (12, 0)
        } else {
            let mut k = 0;
            while (k + 1) * (k + 1) <= j - 1 {
                k += 1;
            }
            let off = j - 1 - k * k;
            let am = (k - off / 2) as i32;
            (2 * k - am as usize, if off % 2 == 1 { -am } else { am })
        };
        let am = m.unsigned_abs() as usize;
        let (mut re, mut im) = (Jet::constant(c(1.0)), Jet::constant(c(0.0)));
        for _ in 0..am {
            let r = re * x - im * y;
            im = re * y + im * x;
            re = r;
        }
        let scale = if standard {
            c::<R>(((degree + 1) * if m == 0 { 1 } else { 2 }) as f32).sqrt()
        } else {
            c(1.0)
        };
        z = z
            + (jacobi(
                (degree - am) / 2,
                c(am as f32),
                s.scale(c(2.0)).add(c(-1.0)),
            ) * if m >= 0 { re } else { im })
            .scale(a * scale);
    }
    z
}

pub(crate) fn data<'a>(s: &PackedSurface, pool: &'a [Df32]) -> &'a [Df32] {
    &pool[s.coefficient_offset as usize..][..s.coefficient_count as usize]
}
pub(crate) fn domain<R: Scalar>(s: &PackedSurface, pool: &[Df32], x: R, y: R) -> bool {
    let d = data(s, pool);
    normalized(Jet::variable(x, true), Jet::variable(y, false), load(d, 0)).is_some()
}
pub(crate) fn phase<R: Scalar>(s: &PackedSurface, pool: &[Df32], x: R, y: R) -> Jet<R> {
    let d = data(s, pool);
    let inv = c::<R>(1.0) / load::<R>(d, 0);
    let (x, y) = (
        Jet::variable(x, true).scale(inv),
        Jet::variable(y, false).scale(inv),
    );
    let t = x * x + y * y;
    poly(
        d,
        (MKL_COEFF_PREFIX + MKL_MAX_WAVELENGTHS) as usize,
        d[MKL_COEFF_NA as usize].hi as usize,
        t,
    ) * t
}
pub(crate) fn evaluate<R: Scalar>(s: &PackedSurface, pool: &[Df32], x: R, y: R) -> Option<Jet<R>> {
    let data = data(s, pool);
    let radius = load::<R>(data, 0);
    let na = data[MKL_COEFF_NA as usize].hi as usize;
    let nb = data[MKL_COEFF_NB as usize].hi as usize;
    let coeff = &data[MKL_COEFF_PREFIX as usize..];
    let (x, y) = (Jet::variable(x, true), Jet::variable(y, false));
    let r2 = x * x + y * y;
    let curv = R::parts(s.curvature, s.curvature_lo);
    let k = R::parts(s.conic, s.conic_lo);
    Some(match s.shape {
        MKL_SHAPE_ODD_ASPHERE => {
            let base = conic(curv, k, r2)?;
            if x.v.zero() && y.v.zero() {
                base
            } else {
                // A nonzero cone radius can have an underflowing square.
                // Recover r and its unit radial gradient with scaled hypot.
                let radial = if r2.v.zero() {
                    let ax = if x.v.le(0.0) { -x.v } else { x.v };
                    let ay = if y.v.le(0.0) { -y.v } else { y.v };
                    let scale = if ax.le_value(ay) { ay } else { ax };
                    let u = x.v / scale;
                    let v = y.v / scale;
                    let w = (u * u + v * v).sqrt();
                    Jet {
                        v: scale * w,
                        x: u / w,
                        y: v / w,
                    }
                } else {
                    r2.sqrt()
                };
                let r = radial.scale(c::<R>(1.0) / radius);
                base + poly(coeff, 0, na, r) * r
            }
        }
        MKL_SHAPE_TOROID | MKL_SHAPE_BICONIC => {
            let cx = load::<R>(data, MKL_COEFF_CX as usize);
            let inv = c::<R>(1.0) / radius;
            let xx = x.scale(inv);
            let yy = y.scale(inv);
            if s.shape == MKL_SHAPE_TOROID {
                let zy = conic(curv, k, y * y)? + even(coeff, na, nb, yy * yy);
                let a = (-zy).scale(cx).add(c(1.0));
                let q = a * a - (x * x).scale(cx * cx);
                if a.v.le(0.0) || !c::<R>(0.0).le_value(q.v) {
                    return None;
                }
                zy + (x * x).scale(cx) / (a + q.sqrt())
            } else {
                let kx = load::<R>(data, MKL_COEFF_KX as usize);
                let arg = ((x * x).scale(-(c::<R>(1.0) + kx) * cx * cx)
                    + (y * y).scale(-(c::<R>(1.0) + k) * curv * curv))
                .add(c(1.0));
                if !c::<R>(0.0).le_value(arg.v) {
                    return None;
                }
                ((x * x).scale(cx) + (y * y).scale(curv)) / (arg.sqrt().add(c(1.0)))
                    + even(coeff, 0, na, xx * xx)
                    + even(coeff, na, nb, yy * yy)
            }
        }
        _ => {
            let (u, v, t) = normalized(x, y, radius)?;
            let base = conic(curv, k, r2)?;
            match s.shape {
                MKL_SHAPE_Q_CON => base + qcon(coeff, na, t),
                MKL_SHAPE_Q_BFS => {
                    if coeff.iter().all(|a| a.hi == 0.0) {
                        base
                    } else {
                        let w = r2.scale(-curv * curv).add(c(1.0)).sqrt();
                        if w.v.zero() {
                            return None;
                        }
                        base + t * (-t).add(c(1.0)) * qbfs(coeff, na, t) / w
                    }
                }
                MKL_SHAPE_XY => {
                    let (mut start, mut degree) = (0, 1);
                    let mut z = base;
                    while start < na {
                        let (mut row, mut vn) = (Jet::constant(c(0.0)), Jet::constant(c(1.0)));
                        for n in 0..=degree {
                            let a = if start + n < na {
                                load(coeff, start + n)
                            } else {
                                c(0.0)
                            };
                            row = row * u + vn.scale(a);
                            vn = vn * v;
                        }
                        z = z + row;
                        start += degree + 1;
                        degree += 1;
                    }
                    z
                }
                MKL_SHAPE_ZERNIKE => {
                    base + zernike(coeff, na, data[MKL_COEFF_KIND as usize].hi != 0.0, u, v, t)
                }
                _ => unreachable!("extended sag shape"),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ftz, Ieee, PackedSystem, testutil};

    fn check<E: FpEnv>() {
        for (name, shape) in testutil::extended_shapes() {
            let p =
                PackedSystem::pack_all(&testutil::resolve(testutil::one_surface(shape.clone())))
                    .unwrap();
            if p.surfaces[1].shape == MKL_SHAPE_DIFFRACTIVE {
                continue;
            }
            for i in 0..65 {
                let angle = i as f64 * 2.399963229728653;
                let r = 11.9 * i as f64 / 64.0;
                let x = r * angle.cos();
                let y = r * angle.sin();
                let want = mklight_core::surface_math::evaluate(&shape, x, y).unwrap();
                let v = evaluate(
                    &p.surfaces[1],
                    &p.coefficients,
                    D::<E>::from_df(Df32::from_f64(x)),
                    D::<E>::from_df(Df32::from_f64(y)),
                )
                .unwrap();
                for (got, want) in [
                    (v.v, want.sag),
                    (v.x, want.gradient[0]),
                    (v.y, want.gradient[1]),
                ] {
                    assert!(
                        (got.0.to_f64() - want).abs() < 2e-10,
                        "{name} r={r}: {:?} vs {want}",
                        got.0
                    );
                }
            }
        }
    }
    #[test]
    fn full_capacity_sags_and_gradients_match_f64() {
        check::<Ftz>();
        check::<Ieee>();
    }
}
