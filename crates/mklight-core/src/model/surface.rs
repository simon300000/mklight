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
#[serde(
    remote = "Self",
    tag = "type",
    rename_all = "snake_case",
    deny_unknown_fields
)]
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
    /// Conic plus β₁r + β₂r² + … (coefficients in mm^(1-i)).
    OddAsphere {
        curvature: f64,
        #[serde(default)]
        conic: f64,
        coefficients: Vec<f64>,
    },
    /// Forbes conic-based departure, coefficients in mm (SURFACES §2).
    QCon {
        curvature: f64,
        #[serde(default)]
        conic: f64,
        norm_radius: f64,
        coefficients: Vec<f64>,
    },
    /// Forbes best-fit-sphere departure; curvature is that of the sphere.
    QBfs {
        curvature: f64,
        norm_radius: f64,
        coefficients: Vec<f64>,
    },
    /// Fixed-period phase structure on a plane, conic or even-asphere base.
    Diffractive {
        base: Box<SurfaceShape>,
        phase: DiffractivePhase,
    },
    /// Rotation of the y conic/even-asphere profile about a parallel y axis.
    Toroid {
        curvature_x: f64,
        curvature_y: f64,
        #[serde(default)]
        conic_y: f64,
        #[serde(default)]
        y_asphere: Vec<f64>,
    },
    Biconic {
        curvature_x: f64,
        curvature_y: f64,
        #[serde(default)]
        conic_x: f64,
        #[serde(default)]
        conic_y: f64,
        #[serde(default)]
        x_asphere: Vec<f64>,
        #[serde(default)]
        y_asphere: Vec<f64>,
    },
    /// Total degree ascending, then x power descending; starts with x, y.
    #[serde(rename = "xy_polynomial")]
    XYPolynomial {
        curvature: f64,
        #[serde(default)]
        conic: f64,
        norm_radius: f64,
        coefficients: Vec<f64>,
    },
    /// Sag coefficients in mm, including piston; see SURFACES §6.
    ZernikeSag {
        curvature: f64,
        #[serde(default)]
        conic: f64,
        norm_radius: f64,
        #[serde(default)]
        ordering: ZernikeOrdering,
        coefficients: Vec<f64>,
    },
    /// Ideal thin lens with the given focal length in mm: a perfect paraxial
    /// element that images without aberration (used for afocal output and testing).
    Paraxial { focal_length: f64 },
    /// Changes the frame only and does not interact with light. Uses `placement`.
    CoordinateBreak,
}

crate::serde_util::strict_unit_variants!(SurfaceShape, "type", ["plane", "coordinate_break"]);

impl SurfaceShape {
    pub fn curvature(&self) -> f64 {
        match *self {
            SurfaceShape::Conic { curvature, .. }
            | SurfaceShape::EvenAsphere { curvature, .. }
            | SurfaceShape::OddAsphere { curvature, .. }
            | SurfaceShape::QCon { curvature, .. }
            | SurfaceShape::QBfs { curvature, .. }
            | SurfaceShape::XYPolynomial { curvature, .. }
            | SurfaceShape::ZernikeSag { curvature, .. } => curvature,
            SurfaceShape::Diffractive { ref base, .. } => base.curvature(),
            SurfaceShape::Toroid { curvature_y, .. }
            | SurfaceShape::Biconic { curvature_y, .. } => curvature_y,
            _ => 0.0,
        }
    }
    pub fn conic(&self) -> f64 {
        match *self {
            SurfaceShape::Conic { conic, .. }
            | SurfaceShape::EvenAsphere { conic, .. }
            | SurfaceShape::OddAsphere { conic, .. }
            | SurfaceShape::QCon { conic, .. }
            | SurfaceShape::XYPolynomial { conic, .. }
            | SurfaceShape::ZernikeSag { conic, .. } => conic,
            SurfaceShape::Diffractive { ref base, .. } => base.conic(),
            SurfaceShape::Toroid { conic_y, .. } | SurfaceShape::Biconic { conic_y, .. } => conic_y,
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

    /// The serde `type` tag.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Plane => "plane",
            Self::Conic { .. } => "conic",
            Self::EvenAsphere { .. } => "even_asphere",
            Self::OddAsphere { .. } => "odd_asphere",
            Self::QCon { .. } => "q_con",
            Self::QBfs { .. } => "q_bfs",
            Self::Diffractive { .. } => "diffractive",
            Self::Toroid { .. } => "toroid",
            Self::Biconic { .. } => "biconic",
            Self::XYPolynomial { .. } => "xy_polynomial",
            Self::ZernikeSag { .. } => "zernike_sag",
            Self::Paraxial { .. } => "paraxial",
            Self::CoordinateBreak => "coordinate_break",
        }
    }

    /// Whether the shape belongs to the original solver subset. Later solvers
    /// must reject other shapes until their mathematics is implemented.
    pub fn is_legacy(&self) -> bool {
        matches!(
            self,
            Self::Plane
                | Self::Conic { .. }
                | Self::EvenAsphere { .. }
                | Self::Paraxial { .. }
                | Self::CoordinateBreak
        )
    }

    /// Sag on the positive y meridian, `z(0, √r²)`. For a freeform use
    /// [`crate::surface_math::evaluate`] to specify both coordinates.
    /// Returns `None` outside the surface domain.
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
            _ => crate::surface_math::evaluate(self, 0.0, r2.sqrt()).map(|s| s.sag),
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

/// Phase lag Φ at order +1, in radians. Wavelength is in µm; geometry in mm.
/// Design metadata does not rescale the fixed groove period during tracing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffractivePhase {
    pub norm_radius: f64,
    /// A₁ρ² + A₂ρ⁴ + …, at most ten terms.
    pub coefficients: Vec<f64>,
    pub design_order: i32,
    pub design_wavelength_um: f64,
    pub trace_order: i32,
}

/// Matches the wave crate's ordering and normalisation, without depending on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZernikeOrdering {
    #[default]
    Fringe,
    #[serde(alias = "noll")]
    Standard,
}

impl ZernikeOrdering {
    /// Supported terms: Fringe Z1–Z37, Standard (Noll) Z1–Z45 (n ≤ 8).
    pub const fn max_terms(self) -> usize {
        match self {
            Self::Fringe => 37,
            Self::Standard => 45,
        }
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "mode",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum SemiDiameter {
    /// Computed by the tracer: the largest ray height over all fields and
    /// wavelengths at the current aperture. It does not clip rays.
    #[default]
    Auto,
    /// A user-fixed value that clips rays at this radius (mm).
    Fixed { value: f64 },
}

crate::serde_util::strict_unit_variants!(SemiDiameter, "mode", ["auto"]);

/// Optional explicit aperture that clips rays. It overrides the circular
/// `SemiDiameter` when present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
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

/// Relative tolerance of an aperture edge (CONVENTIONS §2): the default
/// real-aiming accuracy, so a ray aimed at the rim is not clipped by its own
/// aiming residual.
pub const APERTURE_EDGE_TOLERANCE: f64 = 1e-10;

impl ApertureShape {
    /// The shape whose edges [`passes`](Self::passes) tests: outer sizes
    /// grown and an annulus's inner radius shrunk by
    /// [`APERTURE_EDGE_TOLERANCE`].
    pub fn with_edge_tolerance(&self) -> Self {
        let outer = 1.0 + APERTURE_EDGE_TOLERANCE;
        match *self {
            ApertureShape::Circular { radius } => ApertureShape::Circular {
                radius: radius * outer,
            },
            ApertureShape::Annular {
                inner_radius,
                outer_radius,
            } => ApertureShape::Annular {
                inner_radius: inner_radius * (1.0 - APERTURE_EDGE_TOLERANCE),
                outer_radius: outer_radius * outer,
            },
            ApertureShape::Rectangular { half_x, half_y } => ApertureShape::Rectangular {
                half_x: half_x * outer,
                half_y: half_y * outer,
            },
            ApertureShape::Elliptical { semi_x, semi_y } => ApertureShape::Elliptical {
                semi_x: semi_x * outer,
                semi_y: semi_y * outer,
            },
        }
    }

    /// Does the local point `(x, y)` pass the aperture, edge tolerance
    /// included?
    pub fn passes(&self, x: f64, y: f64) -> bool {
        match self.with_edge_tolerance() {
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Scalar order efficiency of a diffractive surface (SURFACES §3).
    #[serde(default)]
    pub diffraction_efficiency: crate::materials::diffraction::DiffractionEfficiency,
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
            diffraction_efficiency: Default::default(),
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
