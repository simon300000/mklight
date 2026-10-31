//! Differentiate the shared Cartesian sag/phase gradient with a dual scalar.
//! This retains mixed derivatives without finite differences or radial assumptions.
use crate::{Df32, FpEnv, consts::*, fx::D, layout::PackedSurface, surface_math::Scalar};
use std::ops::{Add, Div, Mul, Neg, Sub};
pub struct Dual<E: FpEnv> {
    pub v: D<E>,
    pub g: [D<E>; 2],
}
impl<E: FpEnv> Copy for Dual<E> {}
impl<E: FpEnv> Clone for Dual<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: FpEnv> Dual<E> {
    fn constant(v: D<E>) -> Self {
        Self { v, g: [D::ZERO; 2] }
    }
    fn variable(v: D<E>, axis: usize) -> Self {
        let mut a = Self::constant(v);
        a.g[axis] = c(1.0);
        a
    }
}
fn c<E: FpEnv>(v: f32) -> D<E> {
    D::new(v, 0.0)
}
impl<E: FpEnv> Add for Dual<E> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            g: std::array::from_fn(|k| self.g[k] + b.g[k]),
        }
    }
}
impl<E: FpEnv> Neg for Dual<E> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            g: self.g.map(|v| -v),
        }
    }
}
impl<E: FpEnv> Sub for Dual<E> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl<E: FpEnv> Mul for Dual<E> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            g: std::array::from_fn(|k| self.g[k] * b.v + self.v * b.g[k]),
        }
    }
}
impl<E: FpEnv> Div for Dual<E> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let r = c::<E>(1.0) / b.v;
        self * Self {
            v: r,
            g: b.g.map(|g| -r * r * g),
        }
    }
}
impl<E: FpEnv> Scalar for Dual<E> {
    fn parts(hi: f32, lo: f32) -> Self {
        Self::constant(D::new(hi, lo))
    }
    fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        let d = c(0.5) / v;
        Self {
            v,
            g: self.g.map(|g| d * g),
        }
    }
    fn le(self, b: f32) -> bool {
        (self.v - c(b)).hi() <= 0.0
    }
    fn zero(self) -> bool {
        self.v.hi() == 0.0
    }
}
/// Sag, gradient, and Hessian xx/xy/yy. All values are double-float.
pub fn evaluate<E: FpEnv>(s: &PackedSurface, pool: &[Df32], x: D<E>, y: D<E>) -> Option<[D<E>; 6]> {
    let xd = Dual::variable(x, 0);
    let yd = Dual::variable(y, 1);
    let (z, gx, gy) = if s.shape <= MKL_SHAPE_COORD_BREAK && s.flags & MKL_SURF_HAS_SAG_OVERLAY == 0
    {
        legacy(s, xd, yd)?
    } else if s.shape == MKL_SHAPE_DIFFRACTIVE && s.flags & MKL_SURF_HAS_SAG_OVERLAY == 0 {
        legacy(s, xd, yd)?
    } else {
        let v = crate::surface_math::evaluate(s, pool, xd, yd)?;
        (v.v, v.x, v.y)
    };
    let mut out = [z.v, gx.v, gy.v, gx.g[0], gx.g[1], gy.g[1]];
    if s.shape == MKL_SHAPE_ODD_ASPHERE && x.hi() == 0.0 && y.hi() == 0.0 {
        let d = crate::surface_math::data(s, pool);
        let na = d[MKL_COEFF_NA as usize].hi as usize;
        if na > 0 && d[MKL_COEFF_PREFIX as usize].hi != 0.0 {
            out[3..].fill(c(f32::NAN));
        } else if na > 1 {
            let r = D::from_df(d[0]);
            let v = c(2.0) * D::from_df(d[MKL_COEFF_PREFIX as usize + 1]) / (r * r);
            out[3] = out[3] + v;
            out[5] = out[5] + v;
        }
    }
    Some(out)
}
fn legacy<E: FpEnv>(
    s: &PackedSurface,
    x: Dual<E>,
    y: Dual<E>,
) -> Option<(Dual<E>, Dual<E>, Dual<E>)> {
    let v = |x| Dual::<E>::constant(x);
    let curvature = v(D::new(s.curvature, s.curvature_lo));
    let conic = v(D::new(s.conic, s.conic_lo));
    let one = v(c(1.0));
    let two = v(c(2.0));
    let r2 = x * x + y * y;
    let arg = one - (one + conic) * curvature * curvature * r2;
    if !arg.v.hi().is_finite() || arg.v.hi() < 0.0 {
        return None;
    }
    let w = arg.sqrt();
    let base = curvature * r2 / (one + w);
    let ds = curvature / (two * w);
    let inv = v(c(s.asphere_inv_norm2));
    let t = r2 * inv;
    let tx = (x + x) * inv;
    let ty = (y + y) * inv;
    let mut p = v(c(0.0));
    let mut px = p;
    let mut py = p;
    for j in (0..s.num_asphere as usize).rev() {
        px = px * t + p * tx;
        py = py * t + p * ty;
        p = p * t + v(D::new(s.asphere[j], s.asphere_lo[j]));
    }
    let pt = p * t;
    let ptx = px * t + p * tx;
    let pty = py * t + p * ty;
    let sag = base + pt * t;
    let gx = ds * (x + x) + (ptx * t + pt * tx);
    let gy = ds * (y + y) + (pty * t + pt * ty);
    Some((sag, gx, gy))
}
pub fn phase<E: FpEnv>(s: &PackedSurface, pool: &[Df32], x: D<E>, y: D<E>) -> [D<E>; 5] {
    let v = crate::surface_math::phase(s, pool, Dual::variable(x, 0), Dual::variable(y, 1));
    [v.x.v, v.y.v, v.x.g[0], v.x.g[1], v.y.g[1]]
}
