//! Minimal f64 vector/matrix math. Kept dependency-free and deliberately simple
//! so the same formulas can be mirrored one-to-one in GPU kernels.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const X: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x.mul_add(o.x, self.y.mul_add(o.y, self.z * o.z))
    }
    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> f64 {
        // hypot-style scaling avoids overflow/underflow for extreme inputs.
        // `max` drops NaN, so NaN is tested explicitly (IEEE hypot: any
        // infinite component gives +inf, otherwise any NaN gives NaN).
        let m = self.x.abs().max(self.y.abs()).max(self.z.abs());
        if m == f64::INFINITY {
            return m;
        }
        if self.x.is_nan() || self.y.is_nan() || self.z.is_nan() {
            return f64::NAN;
        }
        if m == 0.0 {
            return m;
        }
        let (x, y, z) = (self.x / m, self.y / m, self.z / m);
        m * (x * x + y * y + z * z).sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 { self } else { self / n }
    }
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline]
    pub fn max_abs_diff(self, o: Vec3) -> f64 {
        (self.x - o.x)
            .abs()
            .max((self.y - o.y).abs())
            .max((self.z - o.z).abs())
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

/// Row-major 3×3 matrix. Used as a rotation (orthonormal) in `Frame`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// Right-handed rotation about +x by `a` radians.
    pub fn rot_x(a: f64) -> Mat3 {
        let (s, c) = a.sin_cos();
        Mat3 {
            m: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }
    /// Right-handed rotation about +y by `a` radians.
    pub fn rot_y(a: f64) -> Mat3 {
        let (s, c) = a.sin_cos();
        Mat3 {
            m: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }
    /// Right-handed rotation about +z by `a` radians.
    pub fn rot_z(a: f64) -> Mat3 {
        let (s, c) = a.sin_cos();
        Mat3 {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }
    pub fn transpose(&self) -> Mat3 {
        let m = &self.m;
        Mat3 {
            m: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }
    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }
    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        Mat3 { m: r }
    }
    pub fn is_identity(&self) -> bool {
        *self == Mat3::IDENTITY
    }
}

/// A rigid frame: `global = origin + rot · local`.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Frame {
    pub origin: Vec3,
    pub rot: Mat3,
}

impl Frame {
    pub const IDENTITY: Frame = Frame {
        origin: Vec3::ZERO,
        rot: Mat3::IDENTITY,
    };

    pub fn to_global_point(&self, p: Vec3) -> Vec3 {
        self.origin + self.rot.mul_vec(p)
    }
    pub fn to_global_dir(&self, d: Vec3) -> Vec3 {
        self.rot.mul_vec(d)
    }
    pub fn to_local_point(&self, p: Vec3) -> Vec3 {
        self.rot.transpose().mul_vec(p - self.origin)
    }
    pub fn to_local_dir(&self, d: Vec3) -> Vec3 {
        self.rot.transpose().mul_vec(d)
    }
    /// Compose: a child frame expressed in `self`'s local coordinates.
    pub fn compose(&self, child: &Frame) -> Frame {
        Frame {
            origin: self.to_global_point(child.origin),
            rot: self.rot.mul_mat(&child.rot),
        }
    }
    /// The transform that maps points in `self`'s local frame to `other`'s local frame.
    pub fn relative_to(&self, other: &Frame) -> Frame {
        let inv_other_rot = other.rot.transpose();
        Frame {
            origin: inv_other_rot.mul_vec(self.origin - other.origin),
            rot: inv_other_rot.mul_mat(&self.rot),
        }
    }
}

/// Neumaier compensated summation (CONVENTIONS §7.5).
#[derive(Clone, Copy, Debug, Default)]
pub struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }
    pub fn value(&self) -> f64 {
        self.sum + self.c
    }
}

impl FromIterator<f64> for KahanSum {
    fn from_iter<I: IntoIterator<Item = f64>>(it: I) -> Self {
        let mut s = KahanSum::default();
        for x in it {
            s.add(x);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn frame_roundtrip() {
        let f = Frame {
            origin: Vec3::new(1.0, -2.0, 3.0),
            rot: Mat3::rot_x(0.3).mul_mat(&Mat3::rot_y(-0.7)),
        };
        let p = Vec3::new(0.2, 5.0, -1.0);
        let q = f.to_local_point(f.to_global_point(p));
        assert!(p.max_abs_diff(q) < 1e-14);
    }

    #[test]
    fn rotations_right_handed() {
        let v = Mat3::rot_x(std::f64::consts::FRAC_PI_2).mul_vec(Vec3::Y);
        assert_abs_diff_eq!(v.z, 1.0, epsilon = 1e-15);
        let v = Mat3::rot_y(std::f64::consts::FRAC_PI_2).mul_vec(Vec3::Z);
        assert_abs_diff_eq!(v.x, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn norm_extremes_and_nan() {
        assert_eq!(Vec3::new(2.0, 3.0, 6.0).norm(), 7.0);
        assert_eq!(Vec3::new(3e300, 4e300, 0.0).norm(), 5e300);
        assert_eq!(Vec3::ZERO.norm(), 0.0);
        assert!(Vec3::new(f64::NAN, 0.0, 0.0).norm().is_nan());
        assert!(Vec3::new(0.0, 0.0, f64::NAN).norm().is_nan());
        assert!(Vec3::new(1.0, f64::NAN, 2.0).norm().is_nan());
        assert_eq!(
            Vec3::new(f64::NAN, f64::NEG_INFINITY, 0.0).norm(),
            f64::INFINITY
        );
    }

    #[test]
    fn kahan() {
        let s: KahanSum = std::iter::once(1e16)
            .chain(std::iter::repeat_n(1.0, 1000))
            .chain(std::iter::once(-1e16))
            .collect();
        assert_eq!(s.value(), 1000.0);
    }
}
