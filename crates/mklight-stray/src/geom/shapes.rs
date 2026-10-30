//! Local shapes and their robust ray intersections.
//!
//! [`SagPatch`] is a sag surface `z = S(x² + y²)` (the definition in
//! `SurfaceShape::sag`), bounded by `r_inner ≤ r ≤ r_outer` and an optional
//! [`Mask`]. Rays may come from any direction and either side, and the
//! nearest intersection with `t > t_min` is returned:
//!
//! * **Plane**: `t = −o_z/d_z`.
//! * **Conic**: closed form on the full quadric
//!   `c(x² + y² + (1+k)z²) − 2z = 0` with the stable root pair
//!   `q = −(B + sgn B·√(B² − AC))`, `t = q/A`, `t = C/q` (CONVENTIONS §7.2).
//!   The ray origin is first moved to the entry of the patch's bounding box,
//!   so distant origins lose no precision. Roots on the second sheet are
//!   rejected by the vertex-branch test `(1+k)·c·z ≤ 1`.
//! * **Even asphere** (general): the ray is clipped to the patch's bounding
//!   cylinder and z-slab. `f(t) = z(t) − S(r²(t))` is scanned over cells. A
//!   sign change brackets a root. Equal signs at both ends with opposite
//!   slopes mark an interior extremum, which is located by bisection on `f′`
//!   so that a pair of close roots (near-grazing rays) is still bracketed.
//!   Brackets are refined by Newton steps safeguarded by bisection.
//!
//! Normals are unit vectors pointing to the local **+z side** of a sag patch
//! (the gradient of `z − S`) and **outward** on a cylinder. For conics the
//! quadric gradient `(−cx, −cy, 1 − c(1+k)z)` is used, which stays finite at
//! the edge of the conic's domain (for example a hemisphere's equator).

use mklight_core::error::{MkError, MkResult};
use mklight_core::math::Vec3;
use mklight_core::model::{ApertureShape, SurfaceShape};

use super::Aabb;
use mklight_core::surface_math;

/// Extra in-plane clipping of a sag patch.
#[derive(Clone, Debug, PartialEq)]
pub enum Mask {
    /// Keep hits with `|x| ≤ half_x` and `|y| ≤ half_y` (rectangular sensor).
    InsideRect { half_x: f64, half_y: f64 },
    /// Keep hits that pass the aperture.
    Inside(ApertureShape),
    /// Keep hits that the aperture blocks (aperture-stop plates).
    Outside(ApertureShape),
}

impl Mask {
    #[inline]
    pub fn keeps(&self, x: f64, y: f64) -> bool {
        match self {
            Mask::InsideRect { half_x, half_y } => x.abs() <= *half_x && y.abs() <= *half_y,
            Mask::Inside(a) => a.passes(x, y),
            Mask::Outside(a) => !a.passes(x, y),
        }
    }
}

/// Intersection in a primitive's local frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalHit {
    pub t: f64,
    pub p: Vec3,
    /// Unit geometric normal (+z side of a sag patch, outward on a cylinder).
    pub n: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Plane,
    Conic { c: f64, k: f64 },
    General { c: f64, k: f64 },
}

/// Radius where a conic's sag stops being defined (`∞` when it never does).
pub fn conic_domain_radius(c: f64, k: f64) -> f64 {
    let q = (1.0 + k) * c * c;
    if q > 0.0 {
        1.0 / q.sqrt()
    } else {
        f64::INFINITY
    }
}

/// A bounded sag surface in its local frame.
#[derive(Clone, Debug, PartialEq)]
pub struct SagPatch {
    shape: SurfaceShape,
    r_inner: f64,
    r_outer: f64,
    mask: Option<Mask>,
    kind: Kind,
    z_lo: f64,
    z_hi: f64,
}

impl SagPatch {
    /// `r_outer` must lie inside the conic's domain (for an even asphere,
    /// strictly inside, because its slope diverges at the domain edge).
    pub fn new(
        shape: SurfaceShape,
        r_inner: f64,
        r_outer: f64,
        mask: Option<Mask>,
    ) -> MkResult<SagPatch> {
        if !(r_outer.is_finite() && r_outer > 0.0 && r_inner >= 0.0 && r_inner < r_outer) {
            return Err(MkError::InvalidModel(format!(
                "sag patch needs 0 ≤ r_inner < r_outer < ∞ (got {r_inner}, {r_outer})"
            )));
        }
        // A phase screen has the geometry of its substrate. In particular,
        // a plane DOE keeps the exact plane intersection and finite bounds.
        let shape = match shape {
            SurfaceShape::Diffractive { base, .. } => *base,
            shape => shape,
        };
        let kind = match &shape {
            SurfaceShape::Plane | SurfaceShape::Paraxial { .. } => Kind::Plane,
            SurfaceShape::CoordinateBreak => {
                return Err(MkError::InvalidModel(
                    "a coordinate break has no geometry".into(),
                ));
            }
            SurfaceShape::Conic { curvature, conic } => {
                if *curvature == 0.0 {
                    Kind::Plane
                } else {
                    Kind::Conic {
                        c: *curvature,
                        k: *conic,
                    }
                }
            }
            SurfaceShape::EvenAsphere {
                curvature,
                conic,
                coefficients,
            } => {
                if coefficients.iter().all(|a| *a == 0.0) {
                    if *curvature == 0.0 {
                        Kind::Plane
                    } else {
                        Kind::Conic {
                            c: *curvature,
                            k: *conic,
                        }
                    }
                } else {
                    Kind::General {
                        c: *curvature,
                        k: *conic,
                    }
                }
            }
            _ => Kind::General {
                c: shape.curvature(),
                k: shape.conic(),
            },
        };
        let mut r_outer = r_outer;
        match kind {
            Kind::Conic { c, k } => {
                let rd = conic_domain_radius(c, k);
                if r_outer > rd * (1.0 + 1e-12) {
                    return Err(MkError::InvalidModel(format!(
                        "patch radius {r_outer} exceeds the conic's domain {rd}"
                    )));
                }
                r_outer = r_outer.min(rd);
            }
            Kind::General { c, k } => {
                let rd = conic_domain_radius(c, k);
                if r_outer >= rd * (1.0 - 1e-9) {
                    return Err(MkError::InvalidModel(format!(
                        "asphere patch radius {r_outer} must be inside the conic's domain {rd}"
                    )));
                }
            }
            Kind::Plane => {}
        }
        let mut p = SagPatch {
            shape,
            r_inner,
            r_outer,
            mask,
            kind,
            z_lo: 0.0,
            z_hi: 0.0,
        };
        p.compute_z_bounds();
        Ok(p)
    }

    pub fn shape(&self) -> &SurfaceShape {
        &self.shape
    }
    pub fn r_inner(&self) -> f64 {
        self.r_inner
    }
    pub fn r_outer(&self) -> f64 {
        self.r_outer
    }
    pub fn mask(&self) -> Option<&Mask> {
        self.mask.as_ref()
    }

    /// Sag `S(r²)`, clamped to the domain edge.
    #[inline]
    pub fn sag(&self, r2: f64) -> f64 {
        match self.kind {
            Kind::Plane => 0.0,
            Kind::Conic { c, k } => {
                let arg = (1.0 - (1.0 + k) * c * c * r2).max(0.0);
                c * r2 / (1.0 + arg.sqrt())
            }
            Kind::General { .. } => self.shape.sag(r2).unwrap_or(f64::NAN),
        }
    }

    fn compute_z_bounds(&mut self) {
        let (u0, u1) = (self.r_inner * self.r_inner, self.r_outer * self.r_outer);
        match self.kind {
            Kind::Plane => {}
            Kind::Conic { .. } => {
                // The conic sag is monotone in r².
                let (a, b) = (self.sag(u0), self.sag(u1));
                self.z_lo = a.min(b);
                self.z_hi = a.max(b);
            }
            Kind::General { .. } => {
                let n = 4096;
                let (mut lo, mut hi, mut step) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
                let mut prev = self.sag(u0);
                for i in 0..=n {
                    let u = u0 + (u1 - u0) * i as f64 / n as f64;
                    let z = self.sag(u);
                    lo = lo.min(z);
                    hi = hi.max(z);
                    step = step.max((z - prev).abs());
                    prev = z;
                }
                // Between samples the sag can move by at most about one step.
                self.z_lo = lo - 2.0 * step;
                self.z_hi = hi + 2.0 * step;
                // Freeforms need both coordinates. Gradient padding covers a
                // sampling cell rather than only the positive-y meridian.
                let h = self.r_outer / 64.0;
                for ix in -64..=64 {
                    for iy in -64..=64 {
                        let (x, y) = (ix as f64 * h, iy as f64 * h);
                        if x * x + y * y > u1 {
                            continue;
                        }
                        if let Some(v) = surface_math::evaluate(&self.shape, x, y) {
                            let pad = 2.0 * h * (v.gradient[0].abs() + v.gradient[1].abs());
                            self.z_lo = self.z_lo.min(v.sag - pad);
                            self.z_hi = self.z_hi.max(v.sag + pad);
                        }
                    }
                }
            }
        }
    }

    /// Local bounding box.
    pub fn local_bounds(&self) -> Aabb {
        let r = self.r_outer;
        Aabb {
            min: Vec3::new(-r, -r, self.z_lo),
            max: Vec3::new(r, r, self.z_hi),
        }
    }

    #[inline]
    fn accepts(&self, p: Vec3) -> bool {
        let r2 = p.x * p.x + p.y * p.y;
        r2 <= self.r_outer * self.r_outer
            && r2 >= self.r_inner * self.r_inner
            && self.mask.as_ref().is_none_or(|m| m.keeps(p.x, p.y))
    }

    /// Unit normal toward the local +z side at a point on the surface.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        match (&self.kind, &self.shape) {
            (Kind::Plane, _) => Vec3::Z,
            (Kind::Conic { c, k }, _) => {
                Vec3::new(-c * p.x, -c * p.y, 1.0 - c * (1.0 + k) * p.z).normalized()
            }
            (Kind::General { c, k }, SurfaceShape::EvenAsphere { coefficients, .. }) => {
                let u = p.x * p.x + p.y * p.y;
                let s = (1.0 - (1.0 + k) * c * c * u).max(0.0).sqrt();
                let g = c + 2.0 * s * poly_du(coefficients, u);
                Vec3::new(-p.x * g, -p.y * g, s).normalized()
            }
            _ => surface_math::normal(&self.shape, p.x, p.y).unwrap_or(Vec3::new(
                f64::NAN,
                f64::NAN,
                f64::NAN,
            )),
        }
    }

    /// Clips the ray to the padded bounding cylinder and z-slab.
    fn clip(&self, o: Vec3, d: Vec3, t_min: f64, t_max: f64) -> Option<(f64, f64)> {
        let pad = 1e-9 * (1.0 + self.r_outer + self.z_lo.abs().max(self.z_hi.abs()));
        let (mut ta, mut tb) = (t_min, t_max);
        let (zl, zh) = (self.z_lo - pad, self.z_hi + pad);
        if d.z != 0.0 {
            let (a, b) = ((zl - o.z) / d.z, (zh - o.z) / d.z);
            ta = ta.max(a.min(b));
            tb = tb.min(a.max(b));
        } else if o.z < zl || o.z > zh {
            return None;
        }
        let r = self.r_outer + pad;
        let a = d.x * d.x + d.y * d.y;
        let b = o.x * d.x + o.y * d.y;
        let c = o.x * o.x + o.y * o.y - r * r;
        if a > 0.0 {
            let disc = b * b - a * c;
            if disc < 0.0 {
                return None;
            }
            let q = -(b + disc.sqrt().copysign(b));
            let (t1, t2) = if q == 0.0 { (0.0, 0.0) } else { (q / a, c / q) };
            ta = ta.max(t1.min(t2));
            tb = tb.min(t1.max(t2));
        } else if c > 0.0 {
            return None;
        }
        (ta <= tb).then_some((ta, tb))
    }

    /// Nearest intersection with `t_min < t ≤ t_max`.
    pub fn intersect(&self, o: Vec3, d: Vec3, t_min: f64, t_max: f64) -> Option<LocalHit> {
        match self.kind {
            Kind::Plane => {
                if d.z == 0.0 {
                    return None;
                }
                let t = -o.z / d.z;
                if !(t > t_min && t <= t_max) {
                    return None;
                }
                let p = Vec3::new(o.x + t * d.x, o.y + t * d.y, 0.0);
                self.accepts(p).then_some(LocalHit { t, p, n: Vec3::Z })
            }
            Kind::Conic { c, k } => self.intersect_conic(c, k, o, d, t_min, t_max),
            Kind::General { .. } => self.intersect_general(o, d, t_min, t_max),
        }
    }

    fn intersect_conic(
        &self,
        c: f64,
        k: f64,
        o: Vec3,
        d: Vec3,
        t_min: f64,
        t_max: f64,
    ) -> Option<LocalHit> {
        let (ta, _) = self.clip(o, d, t_min, t_max)?;
        // Shift the origin to the box entry: no precision loss for far origins.
        let o2 = o + d * ta;
        let kk = 1.0 + k;
        let a = c * (d.x * d.x + d.y * d.y + kk * d.z * d.z);
        let b = c * (o2.x * d.x + o2.y * d.y + kk * o2.z * d.z) - d.z;
        let cc = c * (o2.x * o2.x + o2.y * o2.y + kk * o2.z * o2.z) - 2.0 * o2.z;
        let mut roots = [f64::NAN; 2];
        if a.abs() <= 1e-15 * b.abs() {
            if b != 0.0 {
                roots[0] = -cc / (2.0 * b);
            }
        } else {
            let mut disc = b * b - a * cc;
            if disc < 0.0 {
                // Rounding at an exact tangent: accept a double root.
                if disc > -4.0 * f64::EPSILON * (b * b).max((a * cc).abs()) {
                    disc = 0.0;
                } else {
                    return None;
                }
            }
            let q = -(b + disc.sqrt().copysign(b));
            if q == 0.0 {
                roots[0] = -b / a;
            } else {
                let (t1, t2) = (q / a, cc / q);
                roots = [t1.min(t2), t1.max(t2)];
            }
        }
        for tr in roots {
            if tr.is_nan() {
                continue;
            }
            let t = ta + tr;
            if !(t > t_min && t <= t_max) {
                continue;
            }
            let p = o + d * t;
            // Vertex branch only: 1 − (1+k)c z = √(1 − (1+k)c²r²) ≥ 0.
            if kk * c * p.z > 1.0 + 1e-9 {
                continue;
            }
            if self.accepts(p) {
                return Some(LocalHit {
                    t,
                    p,
                    n: self.normal(p),
                });
            }
        }
        None
    }

    /// General bracketing solver. Public so tests can cross-check it against
    /// the closed form on conics.
    pub fn intersect_general(&self, o: Vec3, d: Vec3, t_min: f64, t_max: f64) -> Option<LocalHit> {
        let (ta, tb) = self.clip(o, d, t_min, t_max)?;
        let o2 = o + d * ta;
        let len = tb - ta;
        let f = |t: f64| {
            let (x, y) = (o2.x + t * d.x, o2.y + t * d.y);
            o2.z + t * d.z - surface_math::evaluate(&self.shape, x, y).map_or(f64::NAN, |v| v.sag)
        };
        let df = |t: f64| {
            let (x, y) = (o2.x + t * d.x, o2.y + t * d.y);
            surface_math::evaluate(&self.shape, x, y).map_or(f64::NAN, |v| {
                d.z - v.gradient[0] * d.x - v.gradient[1] * d.y
            })
        };
        let n = ((len * 64.0 / self.r_outer).ceil() as usize).clamp(8, 512);
        let h = len / n as f64;
        let tol = 1e-15 * (1.0 + len + o2.norm());
        let mut a = 0.0;
        let mut fa = f(a);
        for i in 1..=n {
            let b = if i == n { len } else { h * i as f64 };
            let fb = f(b);
            let mut cands = [f64::NAN; 2];
            if fa == 0.0 {
                cands[0] = a;
            } else if fa * fb < 0.0 {
                cands[0] = refine(&f, &df, a, b, fa, tol);
            } else if fb != 0.0 {
                let (da, db) = (df(a), df(b));
                if da * db < 0.0 {
                    // One interior extremum: find it on f′ by bisection.
                    let (mut lo, mut hi) = (a, b);
                    for _ in 0..200 {
                        let m = 0.5 * (lo + hi);
                        if df(m) * da > 0.0 {
                            lo = m;
                        } else {
                            hi = m;
                        }
                        if hi - lo <= tol {
                            break;
                        }
                    }
                    let te = 0.5 * (lo + hi);
                    let fe = f(te);
                    if fe == 0.0 {
                        cands[0] = te;
                    } else if fe * fa < 0.0 {
                        cands[0] = refine(&f, &df, a, te, fa, tol);
                        cands[1] = refine(&f, &df, te, b, fe, tol);
                    }
                }
            }
            for tr in cands {
                if tr.is_nan() {
                    continue;
                }
                let t = ta + tr;
                if t > t_min && t <= t_max {
                    let p = o + d * t;
                    if self.accepts(p) {
                        return Some(LocalHit {
                            t,
                            p,
                            n: self.normal(p),
                        });
                    }
                }
            }
            a = b;
            fa = fb;
        }
        None
    }
}

/// `d/du Σ a_i u^(i+2)` for the even-asphere polynomial in `u = r²`.
fn poly_du(coefficients: &[f64], u: f64) -> f64 {
    let mut acc = 0.0;
    for (i, a) in coefficients.iter().enumerate().rev() {
        acc = acc * u + a * (i + 2) as f64;
    }
    acc * u
}

/// Safeguarded Newton on a bracket `[a, b]` with `f(a) = fa`, `f(a)·f(b) < 0`.
fn refine(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    fa: f64,
    tol: f64,
) -> f64 {
    let (mut lo, mut hi) = (a, b);
    let mut x = 0.5 * (a + b);
    for _ in 0..200 {
        let fx = f(x);
        if fx == 0.0 {
            return x;
        }
        if fx * fa < 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if hi - lo <= tol {
            break;
        }
        let step = fx / df(x);
        let xn = x - step;
        if xn > lo && xn < hi {
            x = xn;
            if step.abs() <= tol {
                return x;
            }
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    0.5 * (lo + hi)
}

/// Open cylinder `x² + y² = R²`, `z0 ≤ z ≤ z1` (lens edges, barrels).
#[derive(Clone, Debug, PartialEq)]
pub struct Cylinder {
    pub radius: f64,
    pub z0: f64,
    pub z1: f64,
}

impl Cylinder {
    pub fn new(radius: f64, za: f64, zb: f64) -> MkResult<Cylinder> {
        if !(radius.is_finite() && radius > 0.0 && za.is_finite() && zb.is_finite()) {
            return Err(MkError::InvalidModel(
                "cylinder needs finite radius > 0 and finite z".into(),
            ));
        }
        Ok(Cylinder {
            radius,
            z0: za.min(zb),
            z1: za.max(zb),
        })
    }
    pub fn local_bounds(&self) -> Aabb {
        let r = self.radius;
        Aabb {
            min: Vec3::new(-r, -r, self.z0),
            max: Vec3::new(r, r, self.z1),
        }
    }
    pub fn intersect(&self, o: Vec3, d: Vec3, t_min: f64, t_max: f64) -> Option<LocalHit> {
        let a = d.x * d.x + d.y * d.y;
        if a <= 0.0 {
            return None;
        }
        let b = o.x * d.x + o.y * d.y;
        let c = o.x * o.x + o.y * o.y - self.radius * self.radius;
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let q = -(b + disc.sqrt().copysign(b));
        let (t1, t2) = if q == 0.0 { (0.0, 0.0) } else { (q / a, c / q) };
        for t in [t1.min(t2), t1.max(t2)] {
            if !(t > t_min && t <= t_max) {
                continue;
            }
            let p = o + d * t;
            if p.z >= self.z0 && p.z <= self.z1 {
                let rr = p.x.hypot(p.y);
                return Some(LocalHit {
                    t,
                    p,
                    n: Vec3::new(p.x / rr, p.y / rr, 0.0),
                });
            }
        }
        None
    }
}

/// A local shape.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sag(SagPatch),
    Cylinder(Cylinder),
}

impl Shape {
    pub fn local_bounds(&self) -> Aabb {
        match self {
            Shape::Sag(s) => s.local_bounds(),
            Shape::Cylinder(c) => c.local_bounds(),
        }
    }
    #[inline]
    pub fn intersect(&self, o: Vec3, d: Vec3, t_min: f64, t_max: f64) -> Option<LocalHit> {
        match self {
            Shape::Sag(s) => s.intersect(o, d, t_min, t_max),
            Shape::Cylinder(c) => c.intersect(o, d, t_min, t_max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::rng::Philox;

    /// Brute-force reference: dense scan of `f(t)` for the first sign change
    /// inside the patch, then bisection.
    fn brute(p: &SagPatch, o: Vec3, d: Vec3, t_max: f64) -> Option<f64> {
        let f = |t: f64| {
            let q = o + d * t;
            let r2 = q.x * q.x + q.y * q.y;
            if r2 > p.r_outer * p.r_outer {
                return None;
            }
            let s = p.shape.sag(r2)?;
            Some(q.z - s)
        };
        let n = 100_000;
        let mut prev: Option<(f64, f64)> = None;
        for i in 1..=n {
            let t = t_max * i as f64 / n as f64;
            let v = f(t);
            if let (Some((tp, fp)), Some(fv)) = (prev, v)
                && fp * fv <= 0.0
            {
                let (mut lo, mut hi) = (tp, t);
                for _ in 0..200 {
                    let m = 0.5 * (lo + hi);
                    if f(m).unwrap() * fp > 0.0 {
                        lo = m
                    } else {
                        hi = m
                    }
                }
                let tt = 0.5 * (lo + hi);
                if p.accepts(o + d * tt) {
                    return Some(tt);
                }
            }
            prev = v.map(|fv| (t, fv));
        }
        None
    }

    fn random_dir(rng: &mut Philox) -> Vec3 {
        let z = 2.0 * rng.uniform() - 1.0;
        let ph = 2.0 * std::f64::consts::PI * rng.uniform();
        let s = (1.0 - z * z).sqrt();
        Vec3::new(s * ph.cos(), s * ph.sin(), z)
    }

    fn check_random(p: &SagPatch, seed: u64, n: usize, box_r: f64, general: bool) -> usize {
        let mut rng = Philox::new(seed, 0);
        let mut hits = 0;
        for _ in 0..n {
            let o = Vec3::new(
                box_r * (2.0 * rng.uniform() - 1.0),
                box_r * (2.0 * rng.uniform() - 1.0),
                box_r * (2.0 * rng.uniform() - 1.0),
            );
            // Aim roughly at a random surface point so that most rays hit.
            let rr = p.r_outer * rng.uniform().sqrt();
            let ph = std::f64::consts::TAU * rng.uniform();
            let target = Vec3::new(rr * ph.cos(), rr * ph.sin(), p.sag(rr * rr));
            let d = if rng.uniform() < 0.8 {
                (target - o).normalized()
            } else {
                random_dir(&mut rng)
            };
            let t_max = 4.0 * box_r;
            let got = if general {
                p.intersect_general(o, d, 0.0, t_max)
            } else {
                p.intersect(o, d, 0.0, t_max)
            };
            let want = brute(p, o, d, t_max);
            match (got, want) {
                (Some(h), Some(w)) => {
                    hits += 1;
                    assert!(
                        (h.t - w).abs() < 1e-7 * (1.0 + w),
                        "t {} vs {w}, o={o:?} d={d:?}",
                        h.t
                    );
                    let r2 = h.p.x * h.p.x + h.p.y * h.p.y;
                    let res = h.p.z - p.shape.sag(r2).unwrap();
                    assert!(res.abs() < 1e-10 * (1.0 + box_r), "sag residual {res}");
                    // Normal ⟂ surface: compare with a finite-difference tangent
                    // (skipped where the surface is too steep for differences).
                    if h.n.z.abs() < 0.05 {
                        continue;
                    }
                    let e = 1e-6;
                    let rx = h.p.x + e;
                    let tz = (p.sag(rx * rx + h.p.y * h.p.y) - p.sag(r2)) / e;
                    let tan = Vec3::new(1.0, 0.0, tz).normalized();
                    assert!(h.n.dot(tan).abs() < 1e-4, "normal {:?}", h.n);
                }
                (None, None) => {}
                (g, w) => {
                    // A disagreement is only allowed for a near-tangent graze,
                    // where the dense scan itself can miss a root pair.
                    let t = g.map(|h| h.t).or(w).unwrap();
                    let q = o + d * t;
                    let n = p.normal(q);
                    assert!(
                        n.dot(d).abs() < 1e-3,
                        "solver {g:?} vs brute {w:?} (o={o:?} d={d:?})"
                    );
                }
            }
        }
        hits
    }

    #[test]
    fn plane_and_annulus() {
        let p = SagPatch::new(SurfaceShape::Plane, 1.0, 5.0, None).unwrap();
        let h = p
            .intersect(Vec3::new(2.0, 0.0, -3.0), Vec3::Z, 0.0, 10.0)
            .unwrap();
        assert_eq!(h.t, 3.0);
        assert!(
            p.intersect(Vec3::new(0.5, 0.0, -3.0), Vec3::Z, 0.0, 10.0)
                .is_none()
        );
        // From the +z side.
        let h = p
            .intersect(Vec3::new(2.0, 0.0, 3.0), -Vec3::Z, 0.0, 10.0)
            .unwrap();
        assert!(h.n.dot(-Vec3::Z) < 0.0);
    }

    #[test]
    fn sphere_closed_form_and_branches() {
        // R = 10 sphere, hemisphere patch r ≤ 10 (domain edge).
        let p = SagPatch::new(SurfaceShape::sphere(10.0), 0.0, 10.0, None).unwrap();
        // Axial ray: hits the vertex, not the far sheet at z = 20.
        let h = p
            .intersect(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 0.0, 100.0)
            .unwrap();
        assert!((h.t - 5.0).abs() < 1e-14 && h.n.max_abs_diff(Vec3::Z) < 1e-15);
        // Starting inside the sphere (z = 5) heading −z also hits the vertex cap.
        let h = p
            .intersect(Vec3::new(1.0, 0.0, 5.0), -Vec3::Z, 0.0, 100.0)
            .unwrap();
        let z = 10.0 - (100.0f64 - 1.0).sqrt();
        assert!((h.t - (5.0 - z)).abs() < 1e-13);
        // Starting inside heading +z leaves through the open far side: miss.
        assert!(
            p.intersect(Vec3::new(1.0, 0.0, 5.0), Vec3::Z, 0.0, 100.0)
                .is_none()
        );
        // Horizontal ray in the equator plane: hits the rim (z = R) with a
        // horizontal inward normal, which stays finite at the domain edge.
        let h = p
            .intersect(Vec3::new(-20.0, 0.0, 10.0), Vec3::X, 0.0, 100.0)
            .unwrap();
        assert!((h.t - 10.0).abs() < 1e-9, "{h:?}");
        assert!(h.n.max_abs_diff(Vec3::X) < 1e-6, "{h:?}");
    }

    #[test]
    fn grazing_rays_on_a_sphere() {
        let r = 50.0;
        let p = SagPatch::new(SurfaceShape::sphere(r), 0.0, 20.0, None).unwrap();
        // Tangent line at the point at height y0 in the y–z plane.
        let y0: f64 = 12.0;
        let z0 = p.sag(y0 * y0);
        let n = p.normal(Vec3::new(0.0, y0, z0));
        let tan = Vec3::new(0.0, n.z, -n.y);
        for (off, expect) in [(1e-7, true), (-1e-7, false), (1e-3, true), (-1e-3, false)] {
            // +n points into the concave side, where a chord exists.
            let o = Vec3::new(0.0, y0, z0) + n * off - tan * 5.0;
            let h = p.intersect(o, tan, 0.0, 20.0);
            assert_eq!(h.is_some(), expect, "offset {off}");
            if let Some(h) = h {
                // The nearest of the two close roots.
                let half_chord = (2.0 * r * off).sqrt();
                assert!(
                    (h.t - (5.0 - half_chord)).abs() < 1e-6 * (1.0 + 1.0 / half_chord),
                    "{off}: {}",
                    h.t
                );
                let general = p.intersect_general(o, tan, 0.0, 20.0).unwrap();
                assert!(
                    (general.t - h.t).abs() < 1e-8,
                    "general {} closed {}",
                    general.t,
                    h.t
                );
            }
        }
    }

    #[test]
    fn steep_conics_against_brute_force() {
        let cases = [
            (
                SurfaceShape::Conic {
                    curvature: 0.5,
                    conic: -4.0,
                },
                5.0,
            ), // hyperboloid
            (
                SurfaceShape::Conic {
                    curvature: -0.2,
                    conic: -1.0,
                },
                8.0,
            ), // paraboloid
            (
                SurfaceShape::Conic {
                    curvature: 0.1,
                    conic: 2.0,
                },
                conic_domain_radius(0.1, 2.0),
            ), // oblate, to the domain edge
            (
                SurfaceShape::Conic {
                    curvature: 0.25,
                    conic: -0.5,
                },
                3.99,
            ), // prolate, steep
        ];
        for (i, (s, r)) in cases.into_iter().enumerate() {
            let p = SagPatch::new(s, 0.0, r, None).unwrap();
            let hits = check_random(&p, 10 + i as u64, 100, 2.0 * r, false);
            assert!(hits > 30, "case {i}: only {hits} hits");
        }
    }

    #[test]
    fn general_solver_matches_closed_form() {
        let p = SagPatch::new(
            SurfaceShape::Conic {
                curvature: 0.3,
                conic: -2.0,
            },
            0.0,
            4.0,
            None,
        )
        .unwrap();
        let mut rng = Philox::new(77, 0);
        for _ in 0..500 {
            let o = Vec3::new(
                8.0 * rng.uniform() - 4.0,
                8.0 * rng.uniform() - 4.0,
                8.0 * rng.uniform() - 4.0,
            );
            let d = random_dir(&mut rng);
            let a = p.intersect(o, d, 0.0, 50.0);
            let b = p.intersect_general(o, d, 0.0, 50.0);
            match (a, b) {
                (Some(a), Some(b)) => assert!((a.t - b.t).abs() < 1e-9, "{} {}", a.t, b.t),
                (None, None) => {}
                (a, b) => {
                    let t = a.or(b).unwrap().t;
                    assert!(p.normal(o + d * t).dot(d).abs() < 1e-6, "{a:?} {b:?}");
                }
            }
        }
    }

    #[test]
    fn even_asphere_against_brute_force() {
        let s = SurfaceShape::EvenAsphere {
            curvature: 0.05,
            conic: -1.5,
            coefficients: vec![2e-4, -3e-6, 1e-8],
        };
        let p = SagPatch::new(s, 0.0, 12.0, None).unwrap();
        let hits = check_random(&p, 5, 100, 15.0, false);
        assert!(hits > 40);
        let hits = check_random(&p, 6, 60, 15.0, true);
        assert!(hits > 20);
    }

    #[test]
    fn ray_from_inside_glass() {
        // Biconvex lens: front R = 30 at z = 0, back R = −30 at z = 6.
        let front = SagPatch::new(SurfaceShape::sphere(30.0), 0.0, 10.0, None).unwrap();
        let back = SagPatch::new(SurfaceShape::sphere(-30.0), 0.0, 10.0, None).unwrap();
        // Inside the glass at z = 3, travelling backwards and sideways.
        let o = Vec3::new(1.0, -2.0, 3.0);
        let d = Vec3::new(0.2, 0.1, -1.0).normalized();
        let h = front.intersect(o, d, 0.0, 100.0).unwrap();
        // The ray travels against the +z normal: it arrives from the +z (glass) side.
        assert!(h.n.dot(d) < 0.0);
        // The back surface in its own frame (vertex at local z = 0, i.e. z = 6).
        let ob = o - Vec3::new(0.0, 0.0, 6.0);
        let d2 = Vec3::new(-0.1, 0.3, 1.0).normalized();
        let h2 = back.intersect(ob, d2, 0.0, 100.0).unwrap();
        assert!(h2.n.dot(d2) > 0.0, "arrives from the −z (glass) side");
        let r2 = h2.p.x * h2.p.x + h2.p.y * h2.p.y;
        assert!((h2.p.z - back.sag(r2)).abs() < 1e-13);
    }

    #[test]
    fn cylinder_inside_and_outside() {
        let c = Cylinder::new(5.0, 0.0, 10.0).unwrap();
        let h = c
            .intersect(Vec3::new(0.0, 0.0, 5.0), Vec3::X, 0.0, 100.0)
            .unwrap();
        assert!((h.t - 5.0).abs() < 1e-14 && h.n.max_abs_diff(Vec3::X) < 1e-15);
        let h = c
            .intersect(Vec3::new(-10.0, 0.0, 5.0), Vec3::X, 0.0, 100.0)
            .unwrap();
        assert!((h.t - 5.0).abs() < 1e-14 && h.n.max_abs_diff(-Vec3::X) < 1e-15);
        assert!(
            c.intersect(Vec3::new(-10.0, 0.0, 11.0), Vec3::X, 0.0, 100.0)
                .is_none()
        );
        assert!(
            c.intersect(Vec3::new(0.0, 0.0, 5.0), Vec3::Z, 0.0, 100.0)
                .is_none()
        );
    }
    #[test]
    fn extended_surfaces_hit_known_points_from_both_sides() {
        for (name, sys) in mklight_core::samples::all().into_iter().skip(7) {
            for surf in sys.surfaces.iter().filter(|s| !s.shape.is_legacy()) {
                let patch = SagPatch::new(surf.shape.clone(), 0.0, 1.5, None).unwrap();
                for (x, y) in [(0.0, 0.0), (0.7, -0.4), (-0.8, 0.6)] {
                    let v = surface_math::evaluate(&surf.shape, x, y).unwrap();
                    let p = Vec3::new(x, y, v.sag);
                    let n = surface_math::normal(&surf.shape, x, y).unwrap();
                    for sign in [-1.0, 1.0] {
                        let d = Vec3::new(0.1, -0.07, sign).normalized();
                        let h = patch.intersect(p - d * 2.0, d, 0.0, 4.0).unwrap();
                        assert!(h.p.max_abs_diff(p) < 1e-11, "{name}: {h:?} vs {p:?}");
                        assert!(h.n.max_abs_diff(n) < 1e-11);
                        let eps = 1e-6;
                        let vx = surface_math::evaluate(&surf.shape, x + eps, y).unwrap().sag;
                        let vy = surface_math::evaluate(&surf.shape, x, y + eps).unwrap().sag;
                        // An independent finite-difference tangent check at smooth points.
                        if x != 0.0 {
                            assert!(h.n.dot(Vec3::new(1.0, 0.0, (vx - v.sag) / eps)).abs() < 1e-5);
                            assert!(h.n.dot(Vec3::new(0.0, 1.0, (vy - v.sag) / eps)).abs() < 1e-5);
                        }
                    }
                }
            }
        }
    }
}
