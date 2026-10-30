//! Source models. All positions and directions are global (object space looks
//! along +z). Every source emits a known total power, and the tracer gives each
//! of its `N` rays the weight `power/N`.
//!
//! * [`SourceSpec::Collimated`] — a uniform beam of the given power on a disk
//!   perpendicular to its direction, centred on the line through `center`.
//!   [`SourceSpec::collimated_filling_entrance`] aims it at the first
//!   interface with the radius of that interface's mechanical semi-diameter
//!   (times an overfill factor), so the whole front aperture and its edge are
//!   lit. Field sources aimed at the paraxial entrance pupil, and
//!   out-of-field sources that overfill the whole front aperture, come from
//!   `crate::system::StraySystem`.
//! * [`SourceSpec::CollimatedThrough`] — a uniform collimated beam that
//!   fills a disk lying in a plane that need not be perpendicular to the
//!   beam (the entrance-pupil plane `z = z_EP` for an oblique field). Points
//!   are uniform on that disk, which is uniform in flux because the
//!   projection onto the beam's cross-section has a constant Jacobian
//!   `|d·n|`. `crate::system` builds field and out-of-field sources aimed
//!   at the entrance pupil with it.
//! * [`SourceSpec::Point`] — isotropic point source of intensity `I` (W/sr)
//!   that emits only into a cone (half-angle `α`) toward a target. Power is
//!   `2π I (1 − cos α)`. Directions are uniform in solid angle within the cone.
//! * [`SourceSpec::LambertianDisk`] — extended Lambertian emitter of radiance
//!   `L` (W·mm⁻²·sr⁻¹) on an annulus (the inner radius models the black spot
//!   of a veiling-glare measurement), emitting within `θ_max` of its normal.
//!   Power is `L · π(r² − r_in²) · π sin²θ_max`. Directions are
//!   cosine-weighted (uniform in direction cosines within `sin θ_max`), which
//!   is exact for a Lambertian emitter.

use std::f64::consts::PI;

use mklight_core::error::{MkError, MkResult};
use mklight_core::math::Vec3;
use mklight_core::rng::Philox;
use serde::{Deserialize, Serialize};

use crate::geom::{Ray, onb};
use crate::scene::Scene;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceSpec {
    Collimated {
        direction: Vec3,
        center: Vec3,
        radius: f64,
        /// Rays start this far before `center` along `−direction`.
        start_distance: f64,
        power: f64,
    },
    CollimatedThrough {
        direction: Vec3,
        /// Centre of the filled disk.
        center: Vec3,
        /// Normal of the disk's plane.
        normal: Vec3,
        radius: f64,
        /// Rays start this far before the disk along `−direction`.
        start_distance: f64,
        power: f64,
    },
    Point {
        position: Vec3,
        /// Cone axis direction.
        axis: Vec3,
        /// Cone half-angle (degrees, up to 180 for a full sphere).
        half_angle_deg: f64,
        /// W/sr.
        intensity: f64,
    },
    LambertianDisk {
        center: Vec3,
        /// Emission side.
        normal: Vec3,
        radius: f64,
        #[serde(default)]
        inner_radius: f64,
        /// W·mm⁻²·sr⁻¹.
        radiance: f64,
        /// Largest emission angle from the normal (degrees, ≤ 90).
        max_angle_deg: f64,
    },
}

/// The first interface's frame origin, axis and mechanical semi-diameter.
fn entrance(scene: &Scene) -> MkResult<(Vec3, f64)> {
    let first = scene
        .sequence
        .first()
        .and_then(|&s| scene.surfaces.get(s).cloned().flatten())
        .ok_or_else(|| MkError::InvalidModel("scene has no entrance surface".into()))?;
    Ok((first.frame.origin, first.mechanical_semi_diameter))
}

impl SourceSpec {
    /// Collimated beam at field angles `(θx, θy)` (degrees, CONVENTIONS §2:
    /// direction ∝ `(tan θx, tan θy, 1)`) that covers the first interface's
    /// mechanical aperture times `overfill`, with total power `power`.
    pub fn collimated_filling_entrance(
        scene: &Scene,
        field_x_deg: f64,
        field_y_deg: f64,
        overfill: f64,
        power: f64,
    ) -> MkResult<SourceSpec> {
        let (c, r) = entrance(scene)?;
        let d = Vec3::new(
            field_x_deg.to_radians().tan(),
            field_y_deg.to_radians().tan(),
            1.0,
        )
        .normalized();
        let (sc, sr) = scene.bounding_sphere();
        let s = SourceSpec::Collimated {
            direction: d,
            center: c,
            radius: r * overfill,
            start_distance: (c - sc).norm() + sr + r * overfill + 1.0,
            power,
        };
        s.validate()?;
        Ok(s)
    }

    /// Point source at `position` with intensity `intensity` (W/sr), emitting
    /// into the cone that just covers a sphere of the first interface's
    /// mechanical semi-diameter around its vertex.
    pub fn point_toward_entrance(
        scene: &Scene,
        position: Vec3,
        intensity: f64,
    ) -> MkResult<SourceSpec> {
        let (c, r) = entrance(scene)?;
        let v = c - position;
        let dist = v.norm();
        let half = if dist > r {
            (r / dist).asin().to_degrees()
        } else {
            180.0
        };
        let s = SourceSpec::Point {
            position,
            axis: v.normalized(),
            half_angle_deg: half,
            intensity,
        };
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> MkResult<()> {
        let bad = |m: &str| Err(MkError::InvalidModel(format!("source: {m}")));
        match *self {
            SourceSpec::Collimated {
                direction,
                radius,
                power,
                start_distance,
                ..
            } => {
                if !(direction.norm() > 0.0
                    && radius > 0.0
                    && power >= 0.0
                    && start_distance.is_finite())
                {
                    return bad("collimated source needs a direction, radius > 0 and power ≥ 0");
                }
            }
            SourceSpec::CollimatedThrough {
                direction,
                normal,
                radius,
                start_distance,
                power,
                ..
            } => {
                let (d, n) = (direction.norm(), normal.norm());
                if !(d > 0.0
                    && n > 0.0
                    && (direction.dot(normal) / (d * n)).abs() > 1e-9
                    && radius > 0.0
                    && power >= 0.0
                    && start_distance.is_finite())
                {
                    return bad(
                        "collimated beam needs a direction not parallel to the disk, radius > 0 and power ≥ 0",
                    );
                }
            }
            SourceSpec::Point {
                axis,
                half_angle_deg,
                intensity,
                ..
            } => {
                if !(axis.norm() > 0.0
                    && half_angle_deg > 0.0
                    && half_angle_deg <= 180.0
                    && intensity >= 0.0)
                {
                    return bad(
                        "point source needs an axis, 0 < half-angle ≤ 180° and intensity ≥ 0",
                    );
                }
            }
            SourceSpec::LambertianDisk {
                normal,
                radius,
                inner_radius,
                radiance,
                max_angle_deg,
                ..
            } => {
                if !(normal.norm() > 0.0
                    && radius > inner_radius
                    && inner_radius >= 0.0
                    && radiance >= 0.0
                    && max_angle_deg > 0.0
                    && max_angle_deg <= 90.0)
                {
                    return bad(
                        "Lambertian disk needs radius > inner radius ≥ 0 and 0 < θmax ≤ 90°",
                    );
                }
            }
        }
        if !self.power().is_finite() {
            return bad("emitted power must be finite");
        }
        Ok(())
    }

    /// Total emitted power (W).
    pub fn power(&self) -> f64 {
        match *self {
            SourceSpec::Collimated { power, .. } | SourceSpec::CollimatedThrough { power, .. } => {
                power
            }
            SourceSpec::Point {
                half_angle_deg,
                intensity,
                ..
            } => {
                let a = half_angle_deg.to_radians();
                // 1 − cos α = 2 sin²(α/2), without cancellation for small cones.
                2.0 * PI * intensity * 2.0 * (0.5 * a).sin().powi(2)
            }
            SourceSpec::LambertianDisk {
                radius,
                inner_radius,
                radiance,
                max_angle_deg,
                ..
            } => {
                let s = max_angle_deg.to_radians().sin();
                radiance * PI * (radius * radius - inner_radius * inner_radius) * PI * s * s
            }
        }
    }

    /// Draws one ray.
    pub fn sample(&self, rng: &mut Philox) -> Ray {
        match *self {
            SourceSpec::Collimated {
                direction,
                center,
                radius,
                start_distance,
                ..
            } => {
                let d = direction.normalized();
                let (t1, t2) = onb(d);
                let r = radius * rng.uniform().sqrt();
                let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                Ray {
                    origin: center + t1 * (r * c) + t2 * (r * s) - d * start_distance,
                    dir: d,
                }
            }
            SourceSpec::CollimatedThrough {
                direction,
                center,
                normal,
                radius,
                start_distance,
                ..
            } => {
                let d = direction.normalized();
                let (t1, t2) = onb(normal.normalized());
                let r = radius * rng.uniform().sqrt();
                let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                Ray {
                    origin: center + t1 * (r * c) + t2 * (r * s) - d * start_distance,
                    dir: d,
                }
            }
            SourceSpec::Point {
                position,
                axis,
                half_angle_deg,
                ..
            } => {
                let a = axis.normalized();
                let (t1, t2) = onb(a);
                let h = 2.0 * (0.5 * half_angle_deg.to_radians()).sin().powi(2); // 1 − cos α
                let one_minus_cos = rng.uniform() * h;
                let cos_t = 1.0 - one_minus_cos;
                let sin_t = (one_minus_cos * (2.0 - one_minus_cos)).max(0.0).sqrt();
                let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                Ray {
                    origin: position,
                    dir: (t1 * (sin_t * c) + t2 * (sin_t * s) + a * cos_t).normalized(),
                }
            }
            SourceSpec::LambertianDisk {
                center,
                normal,
                radius,
                inner_radius,
                max_angle_deg,
                ..
            } => {
                let n = normal.normalized();
                let (t1, t2) = onb(n);
                let r2 = inner_radius * inner_radius
                    + rng.uniform() * (radius * radius - inner_radius * inner_radius);
                let r = r2.sqrt();
                let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                let origin = center + t1 * (r * c) + t2 * (r * s);
                let b = max_angle_deg.to_radians().sin() * rng.uniform().sqrt();
                let (s2, c2) = (2.0 * PI * rng.uniform()).sin_cos();
                let (bx, by) = (b * c2, b * s2);
                let bz = ((1.0 - b) * (1.0 + b)).max(0.0).sqrt();
                Ray {
                    origin,
                    dir: (t1 * bx + t2 * by + n * bz).normalized(),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_source_cone_and_power() {
        let s = SourceSpec::Point {
            position: Vec3::ZERO,
            axis: Vec3::Z,
            half_angle_deg: 10.0,
            intensity: 2.0,
        };
        let a = 10f64.to_radians();
        assert!((s.power() - 2.0 * PI * 2.0 * (1.0 - a.cos())).abs() < 1e-14);
        let full = SourceSpec::Point {
            position: Vec3::ZERO,
            axis: Vec3::Z,
            half_angle_deg: 180.0,
            intensity: 1.0,
        };
        assert!((full.power() - 4.0 * PI).abs() < 1e-14);
        let mut rng = Philox::new(1, 0);
        let mut mean_cos = 0.0;
        let n = 100_000;
        for _ in 0..n {
            let r = s.sample(&mut rng);
            assert!(r.dir.z >= a.cos() - 1e-15);
            mean_cos += r.dir.z;
        }
        // Uniform in solid angle: E[cos θ] = (1 + cos α)/2.
        let want = 0.5 * (1.0 + a.cos());
        let sigma = (1.0 - a.cos()) / 12f64.sqrt();
        assert!((mean_cos / n as f64 - want).abs() < 5.0 * sigma / (n as f64).sqrt());
    }

    #[test]
    fn lambertian_disk_power_and_cosine_law() {
        let s = SourceSpec::LambertianDisk {
            center: Vec3::ZERO,
            normal: Vec3::Z,
            radius: 2.0,
            inner_radius: 1.0,
            radiance: 1.0,
            max_angle_deg: 90.0,
        };
        assert!((s.power() - PI * PI * 3.0).abs() < 1e-12);
        let mut rng = Philox::new(2, 0);
        let n = 100_000;
        let mut m = 0.0;
        for _ in 0..n {
            let r = s.sample(&mut rng);
            let rr = r.origin.x.hypot(r.origin.y);
            assert!((1.0..=2.0).contains(&rr));
            m += r.dir.z;
        }
        // Cosine-weighted hemisphere: E[cos θ] = 2/3, σ ≈ 0.236.
        assert!((m / n as f64 - 2.0 / 3.0).abs() < 5.0 * 0.236 / (n as f64).sqrt());
    }
}
