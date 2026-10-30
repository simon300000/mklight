//! Exact f64 surface definitions, first and second Cartesian derivatives.
//! See `docs/SURFACES.md` for bases, domains and vertex conventions.

use crate::model::surface::{DiffractivePhase, ZernikeOrdering};
use crate::{OpticalSystem, SurfaceShape};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Polynomial domains include a relative 10⁻¹² rounding margin at the rim.
pub const DOMAIN_MARGIN: f64 = 1.0 + 1e-12;
pub const MAX_Q_TERMS: usize = 20;
pub const MAX_PHASE_TERMS: usize = 10;
pub const MAX_XY_TERMS: usize = 65; // degrees 1 through 10
pub const MAX_ODD_TERMS: usize = 20;
pub const MAX_AXIS_TERMS: usize = 10;

/// Sag in mm, dimensionless gradient, Hessian in mm⁻¹ (`xx, xy, yy`).
/// At a conical vertex the normal is axial and the Hessian is undefined
/// (NaN). At a vertical tangent derivatives may be infinite/undefined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SagDerivatives {
    pub sag: f64,
    pub gradient: [f64; 2],
    pub hessian: [f64; 3],
}

// A second-order Cartesian jet: applying the product/chain rules here also
// differentiates Horner, Jacobi and Clenshaw recurrences, without differencing.
#[derive(Clone, Copy, Debug)]
struct Jet {
    v: f64,
    d: [f64; 2],
    h: [f64; 3],
}
impl Jet {
    const ZERO: Self = Self::constant(0.0);
    const fn constant(v: f64) -> Self {
        Self {
            v,
            d: [0.0; 2],
            h: [0.0; 3],
        }
    }
    fn variable(v: f64, axis: usize) -> Self {
        let mut j = Self::constant(v);
        j.d[axis] = 1.0;
        j
    }
    fn chain(self, v: f64, d: f64, dd: f64) -> Self {
        Self {
            v,
            d: self.d.map(|a| d * a),
            h: [
                d * self.h[0] + dd * self.d[0] * self.d[0],
                d * self.h[1] + dd * self.d[0] * self.d[1],
                d * self.h[2] + dd * self.d[1] * self.d[1],
            ],
        }
    }
    fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        self.chain(v, 0.5 / v, -0.25 / (v * self.v))
    }
    fn recip(self) -> Self {
        let r = self.v.recip();
        self.chain(r, -r * r, 2.0 * r * r * r)
    }
    fn result(self) -> SagDerivatives {
        SagDerivatives {
            sag: self.v,
            gradient: self.d,
            hessian: self.h,
        }
    }
}
impl Add for Jet {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
            h: std::array::from_fn(|i| self.h[i] + b.h[i]),
        }
    }
}
impl Neg for Jet {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}
impl Sub for Jet {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul<f64> for Jet {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self {
            v: self.v * b,
            d: self.d.map(|v| v * b),
            h: self.h.map(|v| v * b),
        }
    }
}
impl Add<f64> for Jet {
    type Output = Self;
    fn add(mut self, b: f64) -> Self {
        self.v += b;
        self
    }
}
impl Sub<f64> for Jet {
    type Output = Self;
    fn sub(self, b: f64) -> Self {
        self + -b
    }
}
impl Mul for Jet {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
            h: [
                self.h[0] * b.v + 2.0 * self.d[0] * b.d[0] + self.v * b.h[0],
                self.h[1] * b.v + self.d[0] * b.d[1] + self.d[1] * b.d[0] + self.v * b.h[1],
                self.h[2] * b.v + 2.0 * self.d[1] * b.d[1] + self.v * b.h[2],
            ],
        }
    }
}
impl Div for Jet {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: Self) -> Self {
        self * b.recip()
    }
}
fn polynomial(c: &[f64], t: Jet) -> Jet {
    c.iter().rev().fold(Jet::ZERO, |v, a| v * t + *a)
}
fn even(c: &[f64], t: Jet) -> Jet {
    polynomial(c, t) * t * t
}
fn conic(c: f64, k: f64, t: Jet) -> Option<Jet> {
    let arg = 1.0 - (1.0 + k) * c * c * t.v;
    if arg < 0.0 {
        return None;
    }
    let w = arg.sqrt();
    Some(t.chain(
        c * t.v / (1.0 + w),
        c / (2.0 * w),
        (1.0 + k) * c * c * c / (4.0 * w * w * w),
    ))
}
fn normalized(x: Jet, y: Jet, r: f64) -> Option<(Jet, Jet, Jet)> {
    let u = x * r.recip();
    let v = y * r.recip();
    let s = u * u + v * v;
    (s.v <= DOMAIN_MARGIN * DOMAIN_MARGIN).then_some((u, v, s))
}

/// Three-term Jacobi recurrence P_n^(0,b)(t). Derivatives are propagated
/// through the recurrence (Forbes 2010, OE 18, 13851).
fn jacobi(n: usize, b: f64, t: Jet) -> Jet {
    if n == 0 {
        return Jet::constant(1.0);
    }
    let mut p0 = Jet::constant(1.0);
    let mut p1 = (t - 1.0) * ((b + 2.0) / 2.0) + 1.0;
    for k in 1..n {
        let k = k as f64;
        let s = 2.0 * k + b;
        let a1 = 2.0 * (k + 1.0) * (k + b + 1.0) * s;
        let a2 = (t * ((s + 2.0) * s) - b * b) * (s + 1.0);
        let a3 = 2.0 * k * (k + b) * (s + 2.0);
        (p0, p1) = (p1, (a2 * p1 - p0 * a3) * (1.0 / a1));
    }
    p1
}

/// Qcon coefficients multiply Q_m(s)=P_m^(0,4)(2s−1), starting at m=0.
/// Clenshaw with differentiated recurrence, including the leading s².
fn qcon(c: &[f64], s: Jet) -> Jet {
    // Q_(k+1) = A_k Q_k - B_k Q_(k-1); k = 0 gives A_0 = 6s−5, B_0 = 0.
    let ab = |k: usize| {
        let k = k as f64;
        let t = 2.0 * k + 4.0;
        let den = 2.0 * (k + 1.0) * (k + 5.0) * t;
        (
            ((s * 2.0 - 1.0) * ((t + 2.0) * t) - 16.0) * ((t + 1.0) / den),
            2.0 * k * (k + 4.0) * (t + 2.0) / den,
        )
    };
    let (mut b1, mut b2) = (Jet::ZERO, Jet::ZERO);
    for k in (0..c.len()).rev() {
        let b = ab(k).0 * b1 - b2 * ab(k + 1).1 + c[k];
        (b2, b1) = (b1, b);
    }
    b1 * s * s
}

/// Qbfs three-term recurrence tables `(f, g, h)`, Forbes 2010 OE 18, 19700,
/// Eqs. A.14–16 and 3.5–12.
pub fn qbfs_recurrence() -> ([f64; MAX_Q_TERMS], [f64; MAX_Q_TERMS], [f64; MAX_Q_TERMS]) {
    let mut f = [0.0; MAX_Q_TERMS];
    let mut g = f;
    let mut h = f;
    f[0] = 2.0;
    f[1] = 19.0_f64.sqrt() / 2.0;
    g[0] = -0.5;
    for m in 2..MAX_Q_TERMS {
        h[m - 2] = -((m * (m - 1)) as f64) / (2.0 * f[m - 2]);
        g[m - 1] = -(1.0 + g[m - 2] * h[m - 2]) / f[m - 1];
        f[m] = ((m * (m + 1) + 3) as f64 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2]).sqrt();
    }
    (f, g, h)
}

// Fixed stack storage; the model boundary limits the coefficient count to
// MAX_Q_TERMS.
fn qbfs_sum(c: &[f64], s: Jet) -> Jet {
    if c.is_empty() {
        return Jet::ZERO;
    }
    let (f, g, h) = qbfs_recurrence();
    let mut b = [0.0; MAX_Q_TERMS + 2];
    for m in (0..c.len()).rev() {
        b[m] = (c[m] - g[m] * b[m + 1] - h[m] * b[m + 2]) / f[m];
    }
    let (mut a1, mut a2) = (Jet::ZERO, Jet::ZERO);
    for m in (0..c.len()).rev() {
        let a = (s * -4.0 + 2.0) * a1 - a2 + b[m];
        (a2, a1) = (a1, a);
    }
    (a1 + a2) * 2.0
}

/// Zero-based XY coefficient index: d(d+1)/2 − 1 + n, d=m+n ≥ 1.
pub fn xy_index(m: usize, n: usize) -> usize {
    let d = m + n;
    d * (d + 1) / 2 - 1 + n
}
pub fn xy_powers(index: usize) -> (usize, usize) {
    let mut d = 1;
    while (d + 1) * (d + 2) / 2 - 1 <= index {
        d += 1;
    }
    let n = index - (d * (d + 1) / 2 - 1);
    (d - n, n)
}

/// One-based Fringe/Noll term (n,m); positive m is cosine, negative sine.
pub fn zernike_term(ordering: ZernikeOrdering, j: usize) -> Option<(usize, i32)> {
    if j == 0 || j > ordering.max_terms() {
        return None;
    }
    match ordering {
        ZernikeOrdering::Fringe => {
            if j == 37 {
                return Some((12, 0));
            }
            let k = ((j - 1) as f64).sqrt() as usize;
            let off = j - 1 - k * k;
            let m = (k - off / 2) as i32;
            Some((2 * k - m as usize, if off % 2 == 1 { -m } else { m }))
        }
        ZernikeOrdering::Standard => {
            let mut n = 0;
            while (n + 1) * (n + 2) / 2 < j {
                n += 1;
            }
            let k = j - n * (n + 1) / 2 - 1;
            let m = if n % 2 == 0 {
                2 * k.div_ceil(2)
            } else {
                2 * (k / 2) + 1
            } as i32;
            Some((n, if j.is_multiple_of(2) { m } else { -m }))
        }
    }
}
fn zernike(c: &[f64], ordering: ZernikeOrdering, x: Jet, y: Jet, s: Jet) -> Jet {
    let mut z = Jet::ZERO;
    for (j, a) in c.iter().enumerate() {
        if *a == 0.0 {
            continue;
        }
        let (n, m) = zernike_term(ordering, j + 1).expect("validated Zernike count");
        let am = m.unsigned_abs() as usize;
        let (mut re, mut im) = (Jet::constant(1.0), Jet::ZERO);
        for _ in 0..am {
            (re, im) = (re * x - im * y, re * y + im * x);
        }
        let scale = match ordering {
            ZernikeOrdering::Fringe => 1.0,
            ZernikeOrdering::Standard => ((n + 1) as f64 * if m == 0 { 1.0 } else { 2.0 }).sqrt(),
        };
        z = z + jacobi((n - am) / 2, am as f64, s * 2.0 - 1.0)
            * if m >= 0 { re } else { im }
            * (*a * scale);
    }
    z
}

fn sag_jet(shape: &SurfaceShape, x: Jet, y: Jet) -> Option<Jet> {
    use SurfaceShape::*;
    let r2 = x * x + y * y;
    Some(match shape {
        Plane | Paraxial { .. } | CoordinateBreak => Jet::ZERO,
        Conic {
            curvature,
            conic: k,
        } => conic(*curvature, *k, r2)?,
        EvenAsphere {
            curvature,
            conic: k,
            coefficients,
        } => conic(*curvature, *k, r2)? + even(coefficients, r2),
        OddAsphere {
            curvature,
            conic: k,
            coefficients,
        } => {
            let base = conic(*curvature, *k, r2)?;
            if r2.v == 0.0 {
                let mut v = base + r2 * coefficients.get(1).copied().unwrap_or(0.0);
                if coefficients.first().is_some_and(|a| *a != 0.0) {
                    v.h = [f64::NAN; 3];
                }
                v
            } else {
                let r = r2.sqrt();
                base + polynomial(coefficients, r) * r
            }
        }
        QCon {
            curvature,
            conic: k,
            norm_radius,
            coefficients,
        } => {
            let (_, _, s) = normalized(x, y, *norm_radius)?;
            conic(*curvature, *k, r2)? + qcon(coefficients, s)
        }
        QBfs {
            curvature,
            norm_radius,
            coefficients,
        } => {
            let (_, _, s) = normalized(x, y, *norm_radius)?;
            let base = conic(*curvature, 0.0, r2)?;
            if coefficients.iter().all(|a| *a == 0.0) {
                base
            } else {
                let w = (r2 * (-curvature * curvature) + 1.0).sqrt();
                if w.v == 0.0 {
                    return None;
                }
                base + s * (-s + 1.0) * qbfs_sum(coefficients, s) / w
            }
        }
        Diffractive { base, phase } => {
            normalized(x, y, phase.norm_radius)?;
            sag_jet(base, x, y)?
        }
        Toroid {
            curvature_x: cx,
            curvature_y: cy,
            conic_y: ky,
            y_asphere,
        } => {
            // Rationalised: exact at c_x = 0 without dividing by c_x.
            let zy = conic(*cy, *ky, y * y)? + even(y_asphere, y * y);
            let a = -zy * *cx + 1.0;
            let q = a * a - x * x * (*cx * *cx);
            if a.v <= 0.0 || q.v < 0.0 {
                return None;
            }
            zy + x * x * *cx / (a + q.sqrt())
        }
        Biconic {
            curvature_x: cx,
            curvature_y: cy,
            conic_x: kx,
            conic_y: ky,
            x_asphere,
            y_asphere,
        } => {
            let arg = x * x * (-(1.0 + kx) * cx * cx) + y * y * (-(1.0 + ky) * cy * cy) + 1.0;
            if arg.v < 0.0 {
                return None;
            }
            (x * x * *cx + y * y * *cy) / (arg.sqrt() + 1.0)
                + even(x_asphere, x * x)
                + even(y_asphere, y * y)
        }
        XYPolynomial {
            curvature,
            conic: k,
            norm_radius,
            coefficients,
        } => {
            let (u, v, _) = normalized(x, y, *norm_radius)?;
            let mut z = conic(*curvature, *k, r2)?;
            // Horner in u within each total degree d: Σ_n C u^(d−n) v^n.
            let (mut start, mut d) = (0, 1);
            while start < coefficients.len() {
                let (mut row, mut vn) = (Jet::ZERO, Jet::constant(1.0));
                for n in 0..=d {
                    row = row * u + vn * coefficients.get(start + n).copied().unwrap_or(0.0);
                    vn = vn * v;
                }
                z = z + row;
                start += d + 1;
                d += 1;
            }
            z
        }
        ZernikeSag {
            curvature,
            conic: k,
            norm_radius,
            ordering,
            coefficients,
        } => {
            let (u, v, s) = normalized(x, y, *norm_radius)?;
            conic(*curvature, *k, r2)? + zernike(coefficients, *ordering, u, v, s)
        }
    })
}

/// Evaluate a validated shape. `None` means outside its real sag domain.
pub fn evaluate(shape: &SurfaceShape, x: f64, y: f64) -> Option<SagDerivatives> {
    sag_jet(shape, Jet::variable(x, 0), Jet::variable(y, 1)).map(Jet::result)
}
pub fn in_domain(shape: &SurfaceShape, x: f64, y: f64) -> bool {
    evaluate(shape, x, y).is_some()
}

/// Φ, Cartesian gradient and Hessian, with Φ in radians and distances in mm.
pub fn phase(phase: &DiffractivePhase, x: f64, y: f64) -> Option<SagDerivatives> {
    let (_, _, s) = normalized(Jet::variable(x, 0), Jet::variable(y, 1), phase.norm_radius)?;
    Some((polynomial(&phase.coefficients, s) * s).result())
}

/// Unit normal `normalize(−z_x, −z_y, 1)`; `None` where the gradient is
/// not finite (outside the domain or on a vertical rim).
pub fn normal(shape: &SurfaceShape, x: f64, y: f64) -> Option<crate::Vec3> {
    let [zx, zy] = evaluate(shape, x, y)?.gradient;
    (zx.is_finite() && zy.is_finite()).then(|| crate::Vec3::new(-zx, -zy, 1.0).normalized())
}

/// `(c, k)` when an extended shape coincides exactly with a conic: zero
/// departures, and equal axes for toroids and biconics.
pub fn exact_conic(shape: &SurfaceShape) -> Option<(f64, f64)> {
    use SurfaceShape::*;
    let zero = |c: &[f64]| c.iter().all(|a| *a == 0.0);
    match shape {
        OddAsphere {
            curvature,
            conic,
            coefficients,
        }
        | QCon {
            curvature,
            conic,
            coefficients,
            ..
        }
        | XYPolynomial {
            curvature,
            conic,
            coefficients,
            ..
        }
        | ZernikeSag {
            curvature,
            conic,
            coefficients,
            ..
        } => zero(coefficients).then_some((*curvature, *conic)),
        QBfs {
            curvature,
            coefficients,
            ..
        } => zero(coefficients).then_some((*curvature, 0.0)),
        Toroid {
            curvature_x,
            curvature_y,
            conic_y,
            y_asphere,
        } => (curvature_x == curvature_y && *conic_y == 0.0 && zero(y_asphere))
            .then_some((*curvature_y, 0.0)),
        Biconic {
            curvature_x,
            curvature_y,
            conic_x,
            conic_y,
            x_asphere,
            y_asphere,
        } => {
            (curvature_x == curvature_y && conic_x == conic_y && zero(x_asphere) && zero(y_asphere))
                .then_some((*curvature_y, *conic_y))
        }
        _ => None,
    }
}

/// Vertex quadratic coefficients for first order. The cone term is omitted
/// by definition; this is a formal first-order model when β₁ ≠ 0.
pub fn vertex_curvatures(shape: &SurfaceShape) -> [f64; 3] {
    if let SurfaceShape::OddAsphere {
        curvature,
        coefficients,
        ..
    } = shape
    {
        let c = curvature + 2.0 * coefficients.get(1).copied().unwrap_or(0.0);
        [c, 0.0, c]
    } else {
        evaluate(shape, 0.0, 0.0)
            .expect("validated vertex domain")
            .hessian
    }
}

/// Geometrical symmetry; the meridional operation is x → −x.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Symmetry {
    Rotational,
    Meridional,
    None,
}

fn xy_is_rotational(c: &[f64]) -> bool {
    let coefficient = |m, n| c.get(xy_index(m, n)).copied().unwrap_or(0.0);
    for d in 1usize..=10 {
        let leading = coefficient(d, 0);
        let mut binomial = 1.0;
        for n in 0..=d {
            let expected = if d.is_multiple_of(2) && n.is_multiple_of(2) {
                let value = leading * binomial;
                binomial *= ((d - n) / 2) as f64 / (n / 2 + 1) as f64;
                value
            } else {
                0.0
            };
            if coefficient(d - n, n) != expected {
                return false;
            }
        }
    }
    true
}

pub fn shape_symmetry(shape: &SurfaceShape) -> Symmetry {
    use SurfaceShape::*;
    match shape {
        Toroid { .. } | Biconic { .. } if exact_conic(shape).is_none() => Symmetry::Meridional,
        XYPolynomial { coefficients, .. } => {
            if xy_is_rotational(coefficients) {
                Symmetry::Rotational
            } else if coefficients
                .iter()
                .enumerate()
                .all(|(j, a)| *a == 0.0 || xy_powers(j).0.is_multiple_of(2))
            {
                Symmetry::Meridional
            } else {
                Symmetry::None
            }
        }
        ZernikeSag {
            coefficients,
            ordering,
            ..
        } => {
            let mut symmetry = Symmetry::Rotational;
            for (j, _) in coefficients.iter().enumerate().filter(|(_, a)| **a != 0.0) {
                let (_, m) = zernike_term(*ordering, j + 1).expect("validated Zernike count");
                if m != 0 {
                    symmetry = symmetry.max(if (m > 0 && m % 2 == 0) || (m < 0 && m % 2 != 0) {
                        Symmetry::Meridional
                    } else {
                        Symmetry::None
                    });
                }
            }
            symmetry
        }
        Diffractive { base, .. } => shape_symmetry(base),
        _ => Symmetry::Rotational,
    }
}

/// Conservative system classification including placements and apertures.
/// Fields are sampling requests and do not change the system's symmetry.
pub fn system_symmetry(system: &OpticalSystem) -> Symmetry {
    use crate::model::ApertureShape;
    system
        .surfaces
        .iter()
        .fold(Symmetry::Rotational, |mut symmetry, s| {
            symmetry = symmetry.max(shape_symmetry(&s.shape));
            if let Some(p) = s.placement {
                if p.decenter_x != 0.0 || p.tilt_y != 0.0 || p.tilt_z != 0.0 {
                    return Symmetry::None;
                }
                if !p.is_identity() {
                    symmetry = symmetry.max(Symmetry::Meridional);
                }
            }
            if matches!(
                s.aperture,
                Some(ApertureShape::Rectangular { .. } | ApertureShape::Elliptical { .. })
            ) {
                symmetry = symmetry.max(Symmetry::Meridional);
            }
            symmetry
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::surface::DiffractivePhase;
    fn shapes() -> Vec<SurfaceShape> {
        vec![
            SurfaceShape::OddAsphere {
                curvature: 0.04,
                conic: -0.4,
                coefficients: vec![0.002, 0.001, -0.0002, 0.00001],
            },
            SurfaceShape::QCon {
                curvature: 0.04,
                conic: -0.4,
                norm_radius: 5.0,
                coefficients: (0..20).map(|i| 0.001 / (i + 1) as f64).collect(),
            },
            SurfaceShape::QBfs {
                curvature: 0.04,
                norm_radius: 5.0,
                coefficients: (0..20).map(|i| 0.001 / (i + 1) as f64).collect(),
            },
            SurfaceShape::Toroid {
                curvature_x: -0.06,
                curvature_y: 0.03,
                conic_y: -0.5,
                y_asphere: vec![1e-5, -1e-7],
            },
            SurfaceShape::Biconic {
                curvature_x: 0.06,
                curvature_y: 0.03,
                conic_x: 0.2,
                conic_y: -0.5,
                x_asphere: vec![2e-5],
                y_asphere: vec![-1e-5],
            },
            SurfaceShape::XYPolynomial {
                curvature: 0.02,
                conic: 0.3,
                norm_radius: 5.0,
                coefficients: (0..65).map(|j| 0.005 * ((j % 3) as f64 - 1.0)).collect(),
            },
            SurfaceShape::ZernikeSag {
                curvature: 0.02,
                conic: 0.3,
                norm_radius: 5.0,
                ordering: ZernikeOrdering::Fringe,
                coefficients: (0..37).map(|j| 1e-4 * ((j % 3) as f64 - 1.0)).collect(),
            },
            SurfaceShape::ZernikeSag {
                curvature: 0.02,
                conic: 0.3,
                norm_radius: 5.0,
                ordering: ZernikeOrdering::Standard,
                coefficients: (0..45).map(|j| 1e-4 * ((j % 3) as f64 - 1.0)).collect(),
            },
            SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::sphere(20.0)),
                phase: DiffractivePhase {
                    norm_radius: 5.0,
                    coefficients: vec![-3.0, 1.0],
                    design_order: 1,
                    trace_order: 1,
                    design_wavelength_um: 0.55,
                },
            },
        ]
    }
    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} != {b} (tol {tol})");
    }
    #[test]
    fn cartesian_derivatives_cross_check_finite_differences() {
        for shape in shapes() {
            for i in 0..30 {
                let angle = i as f64 * 2.399963229728653;
                let radius = 0.1 + 4.5 * i as f64 / 30.0;
                let (x, y) = (radius * angle.cos(), radius * angle.sin());
                let h = 2e-5;
                let v = evaluate(&shape, x, y).unwrap();
                let xp = evaluate(&shape, x + h, y).unwrap();
                let xm = evaluate(&shape, x - h, y).unwrap();
                let yp = evaluate(&shape, x, y + h).unwrap();
                let ym = evaluate(&shape, x, y - h).unwrap();
                close(v.gradient[0], (xp.sag - xm.sag) / (2.0 * h), 2e-8);
                close(v.gradient[1], (yp.sag - ym.sag) / (2.0 * h), 2e-8);
                close(
                    v.hessian[0],
                    (xp.gradient[0] - xm.gradient[0]) / (2.0 * h),
                    2e-7,
                );
                close(
                    v.hessian[1],
                    (yp.gradient[0] - ym.gradient[0]) / (2.0 * h),
                    2e-7,
                );
                close(
                    v.hessian[1],
                    (xp.gradient[1] - xm.gradient[1]) / (2.0 * h),
                    2e-7,
                );
                close(
                    v.hessian[2],
                    (yp.gradient[1] - ym.gradient[1]) / (2.0 * h),
                    2e-7,
                );
            }
        }
    }
    #[test]
    fn forbes_low_orders_and_derivatives() {
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let s = Jet::variable(x, 0);
            // Forbes (2007) explicit forms.
            let q = [
                1.0,
                6.0 * x - 5.0,
                28.0 * x * x - 42.0 * x + 15.0,
                -(35.0 - 12.0 * x * (14.0 - x * (21.0 - 10.0 * x))),
            ];
            let bfs = [
                1.0,
                (13.0 - 16.0 * x) / 19.0_f64.sqrt(),
                (29.0 - 100.0 * x + 76.0 * x * x) / (190.0_f64.sqrt() / 2.0),
                (207.0 - 4.0 * x * (315.0 - x * (577.0 - 320.0 * x))) * (2.0 / 2545.0_f64).sqrt(),
            ];
            let bfs4 = (7737.0
                - 16.0 * x * (4653.0 - 2.0 * x * (7381.0 - 8.0 * x * (1168.0 - 509.0 * x))))
                / (3.0 * 131831.0_f64.sqrt());
            close(qbfs_sum(&[0.0, 0.0, 0.0, 0.0, 1.0], s).v, bfs4, 3e-14);
            for j in 0..4 {
                let mut c = vec![0.0; j + 1];
                c[j] = 1.0;
                close(qcon(&c, s).v, x * x * q[j], 3e-14);
                close(qbfs_sum(&c, s).v, bfs[j], 3e-14);
                let h = 1e-5;
                for eval in [qcon as fn(&[f64], Jet) -> Jet, qbfs_sum] {
                    let v = eval(&c, s);
                    let a = eval(&c, Jet::variable(x + h, 0));
                    let b = eval(&c, Jet::variable(x - h, 0));
                    let (d, dd) = ((a.v - b.v) / (2.0 * h), (a.d[0] - b.d[0]) / (2.0 * h));
                    close(v.d[0], d, 5e-8 * (1.0 + d.abs()));
                    close(v.h[0], dd, 5e-8 * (1.0 + dd.abs()));
                }
            }
        }
    }
    /// Every Q^bfs term to 20 is orthonormal in Forbes' slope norm:
    /// (2/π)∫₀¹ S_m′S_n′ du/√(1−u²) = δ_mn, S_m = u²(1−u²)Q_m(u²).
    /// Gauss–Chebyshev with 64 nodes is exact for these degrees.
    #[test]
    fn qbfs_terms_are_slope_orthonormal() {
        let nodes = 64;
        let mut gram = [[0.0; MAX_Q_TERMS]; MAX_Q_TERMS];
        for i in 0..nodes {
            let u = ((2 * i + 1) as f64 * std::f64::consts::PI / (2 * nodes) as f64).cos();
            let slopes: Vec<f64> = (0..MAX_Q_TERMS)
                .map(|m| {
                    let mut c = vec![0.0; m + 1];
                    c[m] = 1.0;
                    let q = qbfs_sum(&c, Jet::variable(u * u, 0));
                    let t = u * u;
                    (2.0 * u - 4.0 * u * t) * q.v + t * (1.0 - t) * q.d[0] * 2.0 * u
                })
                .collect();
            for m in 0..MAX_Q_TERMS {
                for n in 0..MAX_Q_TERMS {
                    // The integrand is even in u: (2/π)·½·(π/nodes) per node.
                    gram[m][n] += slopes[m] * slopes[n] / nodes as f64;
                }
            }
        }
        for (m, row) in gram.iter().enumerate() {
            for (n, g) in row.iter().enumerate() {
                close(*g, if m == n { 1.0 } else { 0.0 }, 1e-12);
            }
        }
    }
    #[test]
    fn cylinder_limits_and_vertex_are_regular() {
        for c in [-0.1, 0.0, 1e-20, 0.1] {
            let t = SurfaceShape::Toroid {
                curvature_x: c,
                curvature_y: 0.0,
                conic_y: 0.0,
                y_asphere: vec![],
            };
            for x in [0.0, 1e-12, 2.0] {
                let v = evaluate(&t, x, 3.0).unwrap();
                close(
                    v.sag,
                    crate::model::surface::conic_sag(c, 0.0, x * x).unwrap(),
                    1e-15,
                );
                close(v.gradient[1], 0.0, 1e-15);
                close(v.hessian[2], 0.0, 1e-15);
            }
        }
        let cone = SurfaceShape::OddAsphere {
            curvature: 0.02,
            conic: 0.0,
            coefficients: vec![0.1, 0.03],
        };
        let v = evaluate(&cone, 0.0, 0.0).unwrap();
        assert_eq!(v.gradient, [0.0; 2]);
        assert!(v.hessian[0].is_nan());
        assert_eq!(vertex_curvatures(&cone), [0.08, 0.0, 0.08]);
    }
    #[test]
    fn zernike_ordering_known_terms_and_xy_indices() {
        let expected = [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 2),
            (2, -2),
            (3, 1),
            (3, -1),
            (4, 0),
        ];
        for (j, t) in expected.iter().enumerate() {
            assert_eq!(zernike_term(ZernikeOrdering::Fringe, j + 1), Some(*t));
        }
        assert_eq!(zernike_term(ZernikeOrdering::Fringe, 37), Some((12, 0)));
        assert_eq!(zernike_term(ZernikeOrdering::Standard, 5), Some((2, -2)));
        let x = Jet::variable(0.3, 0);
        let y = Jet::variable(-0.2, 1);
        let s = x * x + y * y;
        close(
            zernike(&[0.0, 0.0, 0.0, 1.0], ZernikeOrdering::Fringe, x, y, s).v,
            2.0 * s.v - 1.0,
            1e-15,
        );
        // SURFACES §5 table: X, Y, X², XY, Y², X³, X²Y, XY², Y³, X⁴.
        let table = [
            (1, 0),
            (0, 1),
            (2, 0),
            (1, 1),
            (0, 2),
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
            (4, 0),
        ];
        for (j, (m, n)) in table.into_iter().enumerate() {
            let mut coefficients = vec![0.0; j + 1];
            coefficients[j] = 1.0;
            let shape = SurfaceShape::XYPolynomial {
                curvature: 0.0,
                conic: 0.0,
                norm_radius: 2.0,
                coefficients,
            };
            let expected = 0.35_f64.powi(m) * (-0.65_f64).powi(n);
            close(evaluate(&shape, 0.7, -1.3).unwrap().sag, expected, 1e-15);
            assert_eq!(xy_powers(j), (m as usize, n as usize));
            assert_eq!(xy_index(m as usize, n as usize), j);
        }
    }
    #[test]
    fn domains_and_model_roundtrip() {
        for shape in shapes() {
            let encoded = serde_json::to_string(&shape).unwrap();
            assert!(encoded.contains(&format!("\"type\":\"{}\"", shape.type_name())));
            assert_eq!(
                serde_json::from_str::<SurfaceShape>(&encoded).unwrap(),
                shape
            );
            if matches!(
                shape,
                SurfaceShape::QCon { .. }
                    | SurfaceShape::QBfs { .. }
                    | SurfaceShape::XYPolynomial { .. }
                    | SurfaceShape::ZernikeSag { .. }
                    | SurfaceShape::Diffractive { .. }
            ) {
                assert!(in_domain(&shape, 5.0, 0.0));
                assert!(!in_domain(&shape, 5.01, 0.0));
            }
        }
        assert!(!in_domain(&SurfaceShape::sphere(2.0), 2.01, 0.0));
    }
    #[test]
    fn all_zernike_terms_match_independent_factorial_definition() {
        let factorial = |n: usize| (1..=n).fold(1.0, |a, k| a * k as f64);
        for ordering in [ZernikeOrdering::Fringe, ZernikeOrdering::Standard] {
            let count = if ordering == ZernikeOrdering::Fringe {
                37
            } else {
                45
            };
            for j in 1..=count {
                let (n, m) = zernike_term(ordering, j).unwrap();
                let am = m.unsigned_abs() as usize;
                for (rho, theta) in [(0.0, 0.0), (0.31, 0.43), (0.81, 1.72), (1.0, -0.37)] {
                    let radial = (0..=(n - am) / 2)
                        .map(|k| {
                            (-1.0_f64).powi(k as i32) * factorial(n - k)
                                / (factorial(k)
                                    * factorial((n + am) / 2 - k)
                                    * factorial((n - am) / 2 - k))
                                * f64::powi(rho, (n - 2 * k) as i32)
                        })
                        .sum::<f64>();
                    let angle = am as f64 * theta;
                    let scale = if ordering == ZernikeOrdering::Fringe {
                        1.0
                    } else {
                        ((n + 1) as f64 * if m == 0 { 1.0 } else { 2.0 }).sqrt()
                    };
                    let expected = radial * if m >= 0 { angle.cos() } else { angle.sin() } * scale;
                    let x = Jet::variable(rho * theta.cos(), 0);
                    let y = Jet::variable(rho * theta.sin(), 1);
                    let mut coeff = vec![0.0; j];
                    coeff[j - 1] = 1.0;
                    close(
                        zernike(&coeff, ordering, x, y, x * x + y * y).v,
                        expected,
                        3e-12,
                    );
                }
            }
        }
    }

    #[test]
    fn symmetry_recognizes_radial_xy_and_meridional_zernikes() {
        let mut s = crate::samples::xy_freeform();
        if let SurfaceShape::XYPolynomial { coefficients, .. } = &mut s.surfaces[1].shape {
            *coefficients = vec![0.0; 14];
            coefficients[9] = 0.001;
            coefficients[11] = 0.002;
            coefficients[13] = 0.001;
        }
        assert_eq!(system_symmetry(&s), Symmetry::Rotational);
        s.surfaces[1].placement = Some(crate::Placement {
            decenter_y: 0.1,
            ..Default::default()
        });
        assert_eq!(system_symmetry(&s), Symmetry::Meridional);
        s.surfaces[1].placement.as_mut().unwrap().decenter_x = 0.1;
        assert_eq!(system_symmetry(&s), Symmetry::None);
        let mut z = crate::samples::zernike_sag();
        if let SurfaceShape::ZernikeSag { coefficients, .. } = &mut z.surfaces[1].shape {
            *coefficients = vec![0.0; 8];
            coefficients[7] = 0.001; // y coma, even in x
        }
        assert_eq!(system_symmetry(&z), Symmetry::Meridional);
    }
}
