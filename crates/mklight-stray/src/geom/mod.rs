//! Non-sequential geometry: rays, bounding boxes, local shapes, placed
//! primitives and the acceleration structure.
//!
//! Every primitive lives in its own rigid local frame (`Frame`, `global =
//! origin + rot·local`) taken from `ResolvedSurface::frame`. Rays are
//! transformed into that frame for intersection. Ray parameters `t` are equal
//! in both frames because the transform is rigid and directions are unit
//! vectors.

pub mod bvh;
pub mod shapes;

pub use bvh::{BruteForce, Bvh, Hit, Primitive, RayCaster};
pub use shapes::{Cylinder, LocalHit, Mask, SagPatch, Shape};

use mklight_core::math::{Frame, Vec3};

/// A ray with unit direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        Ray {
            origin,
            dir: dir.normalized(),
        }
    }
    #[inline]
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    };
    pub fn grow(&mut self, p: Vec3) {
        self.min = Vec3::new(
            self.min.x.min(p.x),
            self.min.y.min(p.y),
            self.min.z.min(p.z),
        );
        self.max = Vec3::new(
            self.max.x.max(p.x),
            self.max.y.max(p.y),
            self.max.z.max(p.z),
        );
    }
    pub fn union(&self, o: &Aabb) -> Aabb {
        let mut r = *self;
        r.grow(o.min);
        r.grow(o.max);
        r
    }
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }
    /// Box of the eight transformed corners of a local box, padded by a
    /// relative epsilon so rounding never culls a true hit.
    pub fn transformed(&self, f: &Frame) -> Aabb {
        let mut r = Aabb::EMPTY;
        for i in 0..8 {
            let p = Vec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            );
            r.grow(f.to_global_point(p));
        }
        r.padded()
    }
    pub fn padded(&self) -> Aabb {
        let e = self.extent();
        let m = self.min.x.abs().max(self.min.y.abs()).max(self.min.z.abs());
        let m = m.max(self.max.x.abs().max(self.max.y.abs()).max(self.max.z.abs()));
        let eps = 1e-9 * (1.0 + m + e.x.max(e.y).max(e.z));
        let d = Vec3::new(eps, eps, eps);
        Aabb {
            min: self.min - d,
            max: self.max + d,
        }
    }
    /// Slab test. Returns the entry parameter if the ray overlaps the box
    /// within `[0, t_max]`. `inv` is the componentwise reciprocal direction.
    #[inline]
    pub fn hit(&self, o: Vec3, inv: Vec3, t_max: f64) -> Option<f64> {
        let (mut t0, mut t1) = (0.0f64, t_max);
        for (lo, hi, oo, iv) in [
            (self.min.x, self.max.x, o.x, inv.x),
            (self.min.y, self.max.y, o.y, inv.y),
            (self.min.z, self.max.z, o.z, inv.z),
        ] {
            let a = (lo - oo) * iv;
            let b = (hi - oo) * iv;
            if a.is_nan() || b.is_nan() {
                // 0·∞: the ray is parallel to and lies on this slab's plane,
                // so this axis does not constrain it.
                continue;
            }
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        (t0 <= t1).then_some(t0)
    }
}

/// Orthonormal tangent basis `(t₁, t₂)` with `t₁ × t₂ = n` for a unit `n`
/// (construction of Duff et al., 2017). `n.z = −0` takes the `+1` branch, as
/// in the resident twins: signed zeros do not survive double-float
/// arithmetic, and coaxial cylinders (lens edges, barrels) have normals with
/// an exactly zero `z`, so a sign-bit branch would mirror their scatter
/// frames between backends.
#[inline]
pub fn onb(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// Specular reflection of direction `d` about unit normal `n` (either sign).
#[inline]
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    (d - n * (2.0 * d.dot(n))).normalized()
}

/// Refraction by the stable vector Snell law (CONVENTIONS §7.3) with
/// `μ = n_in/n_out` and `n` oriented along the propagation (`d·n > 0`).
/// Returns `None` on total internal reflection.
#[inline]
pub fn refract(d: Vec3, n: Vec3, mu: f64) -> Option<Vec3> {
    let c = d.dot(n);
    let k = 1.0 - mu * mu * ((1.0 - c) * (1.0 + c));
    if k < 0.0 {
        return None;
    }
    Some((d * mu + n * (k.sqrt() - mu * c)).normalized())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onb_is_orthonormal() {
        for n in [
            Vec3::Z,
            -Vec3::Z,
            Vec3::X,
            Vec3::new(0.3, -0.4, 0.2).normalized(),
            Vec3::new(1e-9, 0.0, -1.0).normalized(),
        ] {
            let (a, b) = onb(n);
            assert!((a.norm() - 1.0).abs() < 1e-14 && (b.norm() - 1.0).abs() < 1e-14);
            assert!(a.dot(b).abs() < 1e-14 && a.dot(n).abs() < 1e-14);
            assert!(a.cross(b).max_abs_diff(n) < 1e-14);
        }
    }

    #[test]
    fn snell_and_tir() {
        let d = Vec3::new(30f64.to_radians().sin(), 0.0, 30f64.to_radians().cos());
        let t = refract(d, Vec3::Z, 1.0 / 1.5).unwrap();
        assert!((t.x - 30f64.to_radians().sin() / 1.5).abs() < 1e-15);
        let d = Vec3::new(50f64.to_radians().sin(), 0.0, 50f64.to_radians().cos());
        assert!(refract(d, Vec3::Z, 1.5).is_none());
        let r = reflect(d, -Vec3::Z);
        assert!((r.z + d.z).abs() < 1e-15 && (r.x - d.x).abs() < 1e-15);
    }

    #[test]
    fn aabb_slab() {
        let b = Aabb {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let inv = |d: Vec3| Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        assert_eq!(
            b.hit(Vec3::new(0.0, 0.0, -5.0), inv(Vec3::Z), 100.0),
            Some(4.0)
        );
        assert_eq!(b.hit(Vec3::new(2.0, 0.0, -5.0), inv(Vec3::Z), 100.0), None);
        // Origin on a slab plane with a zero direction component.
        assert!(
            b.hit(Vec3::new(1.0, 0.0, -5.0), inv(Vec3::Z), 100.0)
                .is_some()
        );
        assert_eq!(b.hit(Vec3::new(0.0, 0.0, -5.0), inv(Vec3::Z), 3.0), None);
    }
}
