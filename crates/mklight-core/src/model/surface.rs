use serde::{Deserialize, Serialize};

use crate::model::material::MaterialRef;
use crate::model::mechanics::ScatterModel;
use crate::serde_util::f64_inf;

/// Stable identity of a surface, independent of its index. The UI uses it to
/// keep selection and view state across inserts and deletes. It is not part of
/// the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SurfaceId(pub u64);

/// Surface shape in its local frame (CONVENTIONS §2). Curvature `c = 1/R`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SurfaceShape {
    /// Flat surface `z = 0`.
    Plane,
    /// Sphere or conic: `z = c r² / (1 + √(1 − (1+k) c² r²))`.
    Conic {
        curvature: f64,
        #[serde(default)]
        conic: f64,
    },
    /// Conic plus even polynomial: `+ Σ_i a_i r^(2i+4)`, with `coefficients[0] = A4`,
    /// `coefficients[1] = A6`, and so on.
    EvenAsphere {
        curvature: f64,
        #[serde(default)]
        conic: f64,
        coefficients: Vec<f64>,
    },
    /// Ideal thin lens with the given focal length in mm: a perfect paraxial
    /// element that images without aberration (used for afocal output and testing).
    Paraxial { focal_length: f64 },
    /// Changes the frame only and does not interact with light. Uses `placement`.
    CoordinateBreak,
}

impl SurfaceShape {
    pub fn curvature(&self) -> f64 {
        match *self {
            SurfaceShape::Conic { curvature, .. } | SurfaceShape::EvenAsphere { curvature, .. } => {
                curvature
            }
            _ => 0.0,
        }
    }
    pub fn conic(&self) -> f64 {
        match *self {
            SurfaceShape::Conic { conic, .. } | SurfaceShape::EvenAsphere { conic, .. } => conic,
            _ => 0.0,
        }
    }
    pub fn sphere(radius: f64) -> Self {
        SurfaceShape::Conic {
            curvature: if radius.is_infinite() || radius == 0.0 {
                0.0
            } else {
                1.0 / radius
            },
            conic: 0.0,
        }
    }

    /// Sag `z(r²)`, computed in f64. Returns `None` outside the conic's domain.
    /// This is the definition. Tracers may evaluate it differently, but they
    /// must agree with it.
    pub fn sag(&self, r2: f64) -> Option<f64> {
        match self {
            SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
                Some(0.0)
            }
            SurfaceShape::Conic {
                curvature: c,
                conic: k,
            } => conic_sag(*c, *k, r2),
            SurfaceShape::EvenAsphere {
                curvature: c,
                conic: k,
                coefficients,
            } => {
                let base = conic_sag(*c, *k, r2)?;
                // Horner in r²: Σ a_i r^(2i+4) = r⁴ (a0 + r² (a1 + r² (...)))
                let mut acc = 0.0;
                for a in coefficients.iter().rev() {
                    acc = acc * r2 + a;
                }
                Some(base + acc * r2 * r2)
            }
        }
    }
}

/// Cancellation-free conic sag: `c r² / (1 + √(1 − (1+k) c² r²))`.
#[inline]
pub fn conic_sag(c: f64, k: f64, r2: f64) -> Option<f64> {
    let arg = 1.0 - (1.0 + k) * c * c * r2;
    if arg < 0.0 {
        return None;
    }
    Some(c * r2 / (1.0 + arg.sqrt()))
}

/// What the surface does to light that hits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interaction {
    /// Refract into `material` (the medium after the surface).
    #[default]
    Refract,
    /// Specular mirror. The medium after the surface is the medium before it;
    /// `material` is ignored.
    Reflect,
    /// No interaction: the ray passes through with no change of direction
    /// (dummy or reference surface).
    Dummy,
}

/// Clear-aperture size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SemiDiameter {
    /// Computed by the tracer: the largest ray height over all fields and
    /// wavelengths at the current aperture. It does not clip rays.
    Auto,
    /// A user-fixed value that clips rays at this radius (mm).
    Fixed { value: f64 },
}

impl Default for SemiDiameter {
    fn default() -> Self {
        SemiDiameter::Auto
    }
}

/// Optional explicit aperture that clips rays. It overrides the circular
/// `SemiDiameter` when present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApertureShape {
    Circular {
        radius: f64,
    },
    Annular {
        inner_radius: f64,
        outer_radius: f64,
    },
    Rectangular {
        half_x: f64,
        half_y: f64,
    },
    Elliptical {
        semi_x: f64,
        semi_y: f64,
    },
}

impl ApertureShape {
    /// Does the local point `(x, y)` pass the aperture?
    pub fn passes(&self, x: f64, y: f64) -> bool {
        match *self {
            ApertureShape::Circular { radius } => x * x + y * y <= radius * radius,
            ApertureShape::Annular {
                inner_radius,
                outer_radius,
            } => {
                let r2 = x * x + y * y;
                r2 >= inner_radius * inner_radius && r2 <= outer_radius * outer_radius
            }
            ApertureShape::Rectangular { half_x, half_y } => x.abs() <= half_x && y.abs() <= half_y,
            ApertureShape::Elliptical { semi_x, semi_y } => {
                (x / semi_x).powi(2) + (y / semi_y).powi(2) <= 1.0
            }
        }
    }
    /// Radius of the smallest circle that contains the aperture.
    pub fn bounding_radius(&self) -> f64 {
        match *self {
            ApertureShape::Circular { radius } => radius,
            ApertureShape::Annular { outer_radius, .. } => outer_radius,
            ApertureShape::Rectangular { half_x, half_y } => half_x.hypot(half_y),
            ApertureShape::Elliptical { semi_x, semi_y } => semi_x.max(semi_y),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformOrder {
    #[default]
    DecenterThenTilt,
    TiltThenDecenter,
}

/// Rigid placement change applied *before* the surface (CONVENTIONS §2).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Placement {
    #[serde(default)]
    pub decenter_x: f64,
    #[serde(default)]
    pub decenter_y: f64,
    /// Degrees.
    #[serde(default)]
    pub tilt_x: f64,
    #[serde(default)]
    pub tilt_y: f64,
    #[serde(default)]
    pub tilt_z: f64,
    #[serde(default)]
    pub order: TransformOrder,
    /// If true, the frame returns to its pre-placement state after this
    /// surface, before the thickness is applied. Use this to tilt one element
    /// without moving the rest of the system.
    #[serde(default)]
    pub restore_after: bool,
}

impl Placement {
    pub fn is_identity(&self) -> bool {
        self.decenter_x == 0.0
            && self.decenter_y == 0.0
            && self.tilt_x == 0.0
            && self.tilt_y == 0.0
            && self.tilt_z == 0.0
    }
}

/// Mechanical description of a lens surface's edge. The stray-light solver
/// uses it as non-optical geometry. Sizes are in mm.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct EdgeSpec {
    /// Mechanical semi-diameter (≥ the clear aperture). `None` means the clear
    /// aperture plus `default_edge_margin`.
    #[serde(default)]
    pub mechanical_semi_diameter: Option<f64>,
    /// Finish of the edge or flat (ground, blackened and so on).
    #[serde(default)]
    pub edge_scatter: Option<ScatterModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Surface {
    pub id: SurfaceId,
    #[serde(default)]
    pub label: String,
    pub shape: SurfaceShape,
    /// Distance to the next vertex (CONVENTIONS §2). May be `±∞` for the object.
    #[serde(with = "f64_inf")]
    pub thickness: f64,
    /// Medium after this surface.
    pub material: MaterialRef,
    #[serde(default)]
    pub interaction: Interaction,
    #[serde(default)]
    pub semi_diameter: SemiDiameter,
    #[serde(default)]
    pub aperture: Option<ApertureShape>,
    #[serde(default)]
    pub placement: Option<Placement>,
    /// Coating name in `OpticalSystem::coatings`. `None` means an uncoated
    /// (bare Fresnel) interface.
    #[serde(default)]
    pub coating: Option<String>,
    /// Surface roughness or contamination scatter, used by the stray solver.
    #[serde(default)]
    pub scatter: Option<ScatterModel>,
    #[serde(default)]
    pub edge: EdgeSpec,
}

impl Surface {
    pub fn new(id: u64, shape: SurfaceShape, thickness: f64, material: MaterialRef) -> Self {
        Surface {
            id: SurfaceId(id),
            label: String::new(),
            shape,
            thickness,
            material,
            interaction: Interaction::Refract,
            semi_diameter: SemiDiameter::Auto,
            aperture: None,
            placement: None,
            coating: None,
            scatter: None,
            edge: EdgeSpec::default(),
        }
    }
    pub fn is_coordinate_break(&self) -> bool {
        matches!(self.shape, SurfaceShape::CoordinateBreak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sag_matches_geometry() {
        let r: f64 = 50.0;
        let s = SurfaceShape::sphere(r);
        let h: f64 = 10.0;
        let exact = r - (r * r - h * h).sqrt();
        assert!((s.sag(h * h).unwrap() - exact).abs() < 1e-13);
    }

    #[test]
    fn paraboloid_sag() {
        let s = SurfaceShape::Conic {
            curvature: 0.01,
            conic: -1.0,
        };
        assert!((s.sag(400.0).unwrap() - 0.01 * 400.0 / 2.0).abs() < 1e-14);
    }

    #[test]
    fn asphere_terms() {
        let s = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![1e-4, 2e-6],
        };
        let r2: f64 = 9.0;
        assert!((s.sag(r2).unwrap() - (1e-4 * r2.powi(2) + 2e-6 * r2.powi(3))).abs() < 1e-15);
    }
}
