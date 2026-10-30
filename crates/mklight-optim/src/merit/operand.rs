//! Operand kinds — *what* an operand measures — and the registry that
//! describes them to the UI and routes them to evaluators.
//!
//! Units follow CONVENTIONS §1: lengths in mm, angles in degrees, wavefront in
//! waves at the operand's wavelength, spatial frequency in cycles/mm.
//! Pupil coordinates `(px, py)` are normalised entrance-pupil coordinates
//! (CONVENTIONS §3). Field and wavelength are chosen by the operand's
//! selectors, not by the kind.

use serde::{Deserialize, Serialize};

use mklight_core::SurfaceId;
use mklight_core::param::ParamPath;

use super::quadrature::PupilSampling;

/// Cartesian component on the image surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    X,
    #[default]
    Y,
}

/// Reference point for spot and transverse-aberration operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpotReference {
    /// The polychromatic/monochromatic centroid of the spot.
    #[default]
    Centroid,
    /// The chief ray's image point (CONVENTIONS §4).
    Chief,
}

/// Reference for wavefront operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WavefrontReference {
    /// Piston and tilt removed (best-fit reference sphere centred on the
    /// centroid): the usual optimisation target.
    #[default]
    Centroid,
    /// The CONVENTIONS §4 reference sphere centred on the chief ray.
    Chief,
}

/// Which wavelengths the reference of a [`RayGroup`] is computed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CentroidScope {
    /// The operand's own wavelength only.
    #[default]
    Monochromatic,
    /// Every system wavelength with positive weight, each node weighted by
    /// `w_node · w_λ` (the centroid of the white-light spot; penalises
    /// lateral colour).
    Polychromatic,
}

/// The ray set that defines the reference of a per-ray operand
/// ([`OperandKind::RayAberration`] / [`OperandKind::WavefrontAberration`])
/// with a `Centroid` reference: the spot centroid, or the piston and tilt
/// removed from the OPD.
///
/// Contract for evaluators: trace **every** node of `sampling` at the
/// operand's field and configuration — at the operand's wavelength, or at
/// every positive-weight system wavelength for `Polychromatic` — and form the
/// reference with [`spot_centroid`](super::reference::spot_centroid) /
/// [`piston_tilt`](super::reference::piston_tilt). The reference therefore
/// does not depend on which operands happen to be in the spec, and with
/// [`PupilSymmetry::Bilateral`](super::quadrature::PupilSymmetry) the half
/// pupil stands for the mirrored full pupil (x centroid and x tilt are zero by
/// symmetry, not the mean over `px ≥ 0`). Rays that fail are left out of the
/// reference.
///
/// A group may also be attached to a `Chief`-referenced per-ray operand; it
/// then only declares the sampling the operand belongs to.
///
/// **Half pupil on an asymmetric candidate.** The half pupil is exact only
/// when the candidate is mirror-symmetric about the meridional plane (and the
/// field is on the y axis). Otherwise an evaluator must trace the mirrored
/// nodes `(−px, py)` as well (each node keeping half its weight), form the
/// reference over that full set without symmetry assumptions, and report a
/// per-ray operand at node `p` as the *pair value*
/// `v = s·√((v(p)² + v(p̄)²)/2)`, where `v(p̄)` is the same operand at the
/// mirrored node and `s` the sign of the part even under the mirror
/// (`v(p) − v(p̄)` for x aberrations, `v(p) + v(p̄)` for y aberrations and
/// OPD). Then `Σ w v²` over the half pupil equals the full-pupil sum exactly,
/// and `v` reduces to the plain value as the candidate becomes symmetric.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayGroup {
    pub sampling: PupilSampling,
    #[serde(default)]
    pub scope: CentroidScope,
}

impl RayGroup {
    pub fn new(sampling: PupilSampling) -> Self {
        RayGroup {
            sampling,
            scope: CentroidScope::Monochromatic,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FNumberKind {
    /// Paraxial image-space `f'/EPD`.
    #[default]
    ParaxialImageSpace,
    /// Paraxial working F/# `1/(2 n' sin θ')`.
    ParaxialWorking,
    /// From the real marginal ray.
    RealWorking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtfOrientation {
    #[default]
    Tangential,
    Sagittal,
    /// Mean of tangential and sagittal.
    Average,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtfMethod {
    /// Diffraction MTF (FFT or pupil autocorrelation; CONVENTIONS §4).
    #[default]
    Diffraction,
    /// Geometric MTF from the ray distribution (fast, for early design).
    Geometric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistortionMapping {
    /// Relative to `y = f tan θ` (rectilinear), in percent.
    #[default]
    FTanTheta,
    /// Relative to `y = f θ` (fisheye / scanning), in percent.
    FTheta,
}

/// Which paraxial ray of the first-order model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParaxialRayKind {
    /// Axial object point to the edge of the pupil, scaled by the aperture
    /// specification.
    #[default]
    Marginal,
    /// Through the stop centre, scaled to the maximum radial field.
    Chief,
}

/// Quantity of a paraxial ray at a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParaxialQuantity {
    /// Height `y` at the surface (mm).
    #[default]
    Height,
    /// Slope `u = dy/dz` after the surface (dimensionless).
    Slope,
    /// `atan u` after the surface (degrees).
    Angle,
}

/// Quantity of a single real ray at a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayQuantity {
    X,
    Y,
    Z,
    /// Direction cosines.
    L,
    M,
    N,
    /// Angle of the ray to the local z axis in the x–z / y–z plane (degrees).
    AngleX,
    AngleY,
    /// Angle of incidence on the surface (degrees).
    IncidenceAngle,
    /// Optical path length from the object (mm).
    OpticalPath,
}

/// Which explicit ghost path (pair of reflecting surfaces, in order of
/// incidence: light reflects at `second` first, travelling backwards, then at
/// `first`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GhostPair {
    pub first: SurfaceId,
    pub second: SurfaceId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GhostMetric {
    /// Peak irradiance on the image (relative to the primary image peak).
    #[default]
    Peak,
    /// Mean irradiance over the sensor.
    Mean,
    /// Total ghost power reaching the sensor (relative to primary).
    Power,
}

/// Path classes of ARCHITECTURE §Solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrayPathClass {
    Ghost,
    Scatter,
    Mechanical,
    /// Everything except the primary path.
    #[default]
    AllNonPrimary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThicknessMeasure {
    /// On axis (vertex to vertex).
    #[default]
    Centre,
    /// At the larger of the two clear semi-diameters (or `height`).
    Edge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediumFilter {
    /// Gaps filled with anything but air/vacuum.
    #[default]
    Glass,
    Air,
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CombineOp {
    /// Σ args.
    Sum,
    /// `args[0] − args[1]`.
    Difference,
    /// Π args.
    Product,
    /// `args[0] / args[1]`.
    Quotient,
    /// `|args[0]|`.
    Abs,
    Min,
    Max,
}

/// What an operand measures. Serialised with a `"type"` tag; new kinds are
/// added as variants (bumping `MERIT_SCHEMA_VERSION` only when an existing
/// layout changes) or, for plug-ins, through [`OperandKind::Custom`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OperandKind {
    // ---- first order (paraxial) ----
    /// Effective focal length (mm).
    Efl,
    /// Back focal length: last surface vertex to paraxial focus (mm).
    Bfl,
    FNumber {
        #[serde(default)]
        kind: FNumberKind,
    },
    /// Paraxial magnification (finite conjugates).
    Magnification,
    /// Entrance-pupil diameter (mm).
    EntrancePupilDiameter,
    /// Exit-pupil position relative to the image surface (mm).
    ExitPupilPosition,
    /// Entrance-pupil position relative to the first surface, in object
    /// space (mm).
    EntrancePupilPosition,
    /// A paraxial marginal or chief ray quantity after `surface` (default:
    /// the image surface).
    ParaxialRay {
        #[serde(default)]
        ray: ParaxialRayKind,
        #[serde(default)]
        quantity: ParaxialQuantity,
        #[serde(default)]
        surface: Option<SurfaceId>,
    },
    /// Petzval sum `P = Σ c (N' − N)/(N N')` over the surfaces (mm⁻¹); the
    /// Petzval surface has curvature `−N'_image · P`.
    PetzvalSum,

    // ---- geometry of the model ----
    /// Sum of thicknesses from `from` to the vertex of `to` (default:
    /// first optical surface to the image surface), mm.
    TotalTrack {
        #[serde(default)]
        from: Option<SurfaceId>,
        #[serde(default)]
        to: Option<SurfaceId>,
    },
    /// Physical on-axis thickness of the gap from `surface` to `to`
    /// (default: the next surface), summing the thicknesses in between (mm;
    /// positive along the propagation direction, see
    /// [`geometric`](super::geometric)).
    CentreThickness {
        surface: SurfaceId,
        #[serde(default)]
        to: Option<SurfaceId>,
    },
    /// Physical thickness of the gap from `surface` to `to` (default: the
    /// next surface) at `height` (default: the larger fixed semi-diameter of
    /// the two bounding surfaces), mm.
    EdgeThickness {
        surface: SurfaceId,
        #[serde(default)]
        to: Option<SurfaceId>,
        #[serde(default)]
        height: Option<f64>,
    },
    /// Minimum (centre or edge) thickness over the optical gaps
    /// ([`geometric::optical_gaps`](super::geometric::optical_gaps)) filled
    /// with the given medium and starting at a surface in `[from, to]`;
    /// without `to`, every gap except the back-focus gap to the image. A
    /// single aggregated boundary operand.
    MinThickness {
        #[serde(default)]
        measure: ThicknessMeasure,
        #[serde(default)]
        medium: MediumFilter,
        #[serde(default)]
        from: Option<SurfaceId>,
        #[serde(default)]
        to: Option<SurfaceId>,
    },
    /// Maximum thickness over the same gap selection as `MinThickness`.
    MaxThickness {
        #[serde(default)]
        measure: ThicknessMeasure,
        #[serde(default)]
        medium: MediumFilter,
        #[serde(default)]
        from: Option<SurfaceId>,
        #[serde(default)]
        to: Option<SurfaceId>,
    },
    /// The current value of any model parameter.
    ParamValue { param: ParamPath },

    // ---- real rays ----
    /// A real-ray quantity at `surface` (default: image) for pupil point
    /// `(px, py)`, in the surface's local frame unless `global`.
    RealRay {
        quantity: RayQuantity,
        #[serde(default)]
        surface: Option<SurfaceId>,
        #[serde(default)]
        px: f64,
        #[serde(default)]
        py: f64,
        #[serde(default)]
        global: bool,
    },
    /// Transverse ray aberration `ε` on the image for pupil point `(px, py)`
    /// (mm), relative to the chief ray or to the centroid of `group` (which
    /// a `Centroid` reference requires). The default spot merit is built from
    /// these.
    RayAberration {
        axis: Axis,
        px: f64,
        py: f64,
        #[serde(default)]
        reference: SpotReference,
        #[serde(default)]
        group: Option<RayGroup>,
    },
    /// OPD at pupil point `(px, py)` (waves), with piston and tilt of
    /// `group` removed for a `Centroid` reference (which requires `group`).
    /// The default wavefront merit is built from these.
    WavefrontAberration {
        px: f64,
        py: f64,
        #[serde(default)]
        reference: WavefrontReference,
        #[serde(default)]
        group: Option<RayGroup>,
    },
    /// Chief-ray angle of incidence on `surface` (default: image), degrees,
    /// unsigned. At the image this is the chief-ray angle (CRA); 0 means
    /// image-space telecentric.
    ChiefRayAngle {
        #[serde(default)]
        surface: Option<SurfaceId>,
    },

    // ---- image quality ----
    /// RMS spot radius (or one component) over the pupil (mm).
    RmsSpot {
        #[serde(default)]
        reference: SpotReference,
        /// `None` = radial RMS; `Some(axis)` = RMS of that component.
        #[serde(default)]
        component: Option<Axis>,
        #[serde(default)]
        sampling: PupilSampling,
    },
    /// Spot centroid coordinate relative to the chief ray (mm).
    SpotCentroid { axis: Axis },
    /// RMS wavefront error over the pupil (waves).
    WavefrontRms {
        #[serde(default)]
        reference: WavefrontReference,
        #[serde(default)]
        sampling: PupilSampling,
    },
    /// MTF at one spatial frequency (cycles/mm).
    Mtf {
        frequency: f64,
        #[serde(default)]
        orientation: MtfOrientation,
        #[serde(default)]
        method: MtfMethod,
    },
    /// Distortion of the real chief ray at the selected field and
    /// wavelength against the paraxial image height scaled by the mapping,
    /// `100 (h_real − h_ref)/h_ref` along the field direction (percent).
    Distortion {
        #[serde(default)]
        mapping: DistortionMapping,
    },
    /// Lateral colour: chief-ray image height at `short` minus at `long`
    /// (indices into the system wavelengths), measured along the field
    /// direction (`+y` on axis), mm.
    LateralColour { short: usize, long: usize },
    /// Axial colour (chromatic focal shift): paraxial image distance at
    /// `short` minus at `long`, mm (negative when the short wavelength
    /// focuses closer, as in an uncorrected positive lens).
    AxialColour { short: usize, long: usize },
    /// Relative illumination at the selected field (0…1, vs on-axis): the
    /// ratio of the image-space projected solid angles of the ray bundles
    /// (geometric, uniform-radiance object, no coating losses).
    RelativeIllumination,

    // ---- ghosts and stray light ----
    /// Distance of a two-reflection ghost's focus from the image surface
    /// (mm, absolute value). Large is good.
    GhostFocus { pair: GhostPair },
    /// Ghost irradiance on the sensor; `pair = None` sums all two-reflection
    /// ghosts.
    GhostIrradiance {
        #[serde(default)]
        pair: Option<GhostPair>,
        #[serde(default)]
        metric: GhostMetric,
    },
    /// Stray-light power reaching the sensor in the given path class,
    /// relative to the primary-path power.
    StrayLightPower {
        #[serde(default)]
        class: StrayPathClass,
        /// Monte-Carlo rays per evaluation.
        #[serde(default = "default_stray_rays")]
        rays: u64,
        #[serde(default)]
        seed: u64,
    },

    // ---- manufacturing ----
    /// Sensitivity of the `criterion` operand to a perturbation `±delta` of
    /// `param`: `√((c(+δ) − c₀)² + (c(−δ) − c₀)²) / √2`, in the
    /// criterion's units.
    ToleranceSensitivity {
        param: ParamPath,
        delta: f64,
        criterion: Box<OperandKind>,
    },

    // ---- composition and extension ----
    /// Arithmetic on other operands of the same spec, by index.
    Combine { op: CombineOp, args: Vec<usize> },
    /// A plug-in operand. `name` must be registered in the
    /// [`OperandRegistry`]; `params` is passed to its evaluator verbatim.
    Custom {
        name: String,
        #[serde(default)]
        params: serde_json::Value,
    },
}

fn default_stray_rays() -> u64 {
    100_000
}

/// Which engine capability an operand needs; the engine uses it to pick or
/// compose evaluators (and the UI to grey out unavailable operands).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Model data only (no tracing).
    Model,
    Paraxial,
    RealRay,
    Wavefront,
    Mtf,
    Ghost,
    StrayLight,
    Tolerance,
    /// Resolved from other operands of the spec.
    Composite,
    Plugin,
}

impl OperandKind {
    /// Stable tag name (the serde `type`).
    pub fn name(&self) -> &str {
        use OperandKind::*;
        match self {
            Efl => "efl",
            Bfl => "bfl",
            FNumber { .. } => "f_number",
            Magnification => "magnification",
            EntrancePupilDiameter => "entrance_pupil_diameter",
            ExitPupilPosition => "exit_pupil_position",
            EntrancePupilPosition => "entrance_pupil_position",
            ParaxialRay { .. } => "paraxial_ray",
            PetzvalSum => "petzval_sum",
            TotalTrack { .. } => "total_track",
            CentreThickness { .. } => "centre_thickness",
            EdgeThickness { .. } => "edge_thickness",
            MinThickness { .. } => "min_thickness",
            MaxThickness { .. } => "max_thickness",
            ParamValue { .. } => "param_value",
            RealRay { .. } => "real_ray",
            RayAberration { .. } => "ray_aberration",
            WavefrontAberration { .. } => "wavefront_aberration",
            ChiefRayAngle { .. } => "chief_ray_angle",
            RmsSpot { .. } => "rms_spot",
            SpotCentroid { .. } => "spot_centroid",
            WavefrontRms { .. } => "wavefront_rms",
            Mtf { .. } => "mtf",
            Distortion { .. } => "distortion",
            LateralColour { .. } => "lateral_colour",
            AxialColour { .. } => "axial_colour",
            RelativeIllumination => "relative_illumination",
            GhostFocus { .. } => "ghost_focus",
            GhostIrradiance { .. } => "ghost_irradiance",
            StrayLightPower { .. } => "stray_light_power",
            ToleranceSensitivity { .. } => "tolerance_sensitivity",
            Combine { .. } => "combine",
            Custom { name, .. } => name,
        }
    }

    pub fn capability(&self) -> Capability {
        use OperandKind::*;
        match self {
            Efl
            | Bfl
            | FNumber {
                kind: FNumberKind::ParaxialImageSpace | FNumberKind::ParaxialWorking,
            }
            | Magnification
            | EntrancePupilDiameter
            | ExitPupilPosition
            | EntrancePupilPosition
            | ParaxialRay { .. }
            | PetzvalSum
            | AxialColour { .. } => Capability::Paraxial,
            TotalTrack { .. }
            | CentreThickness { .. }
            | EdgeThickness { .. }
            | MinThickness { .. }
            | MaxThickness { .. }
            | ParamValue { .. } => Capability::Model,
            FNumber { .. }
            | RealRay { .. }
            | RayAberration { .. }
            | ChiefRayAngle { .. }
            | RmsSpot { .. }
            | SpotCentroid { .. }
            | Distortion { .. }
            | LateralColour { .. }
            | RelativeIllumination => Capability::RealRay,
            WavefrontAberration { .. } | WavefrontRms { .. } => Capability::Wavefront,
            Mtf { .. } => Capability::Mtf,
            GhostFocus { .. } | GhostIrradiance { .. } => Capability::Ghost,
            StrayLightPower { .. } => Capability::StrayLight,
            ToleranceSensitivity { .. } => Capability::Tolerance,
            Combine { .. } => Capability::Composite,
            Custom { .. } => Capability::Plugin,
        }
    }

    /// Does the value depend on the field / wavelength selectors?
    pub fn uses_field(&self) -> bool {
        use OperandKind::*;
        if let ToleranceSensitivity { criterion, .. } = self {
            return criterion.uses_field();
        }
        matches!(
            self,
            RealRay { .. }
                | RayAberration { .. }
                | WavefrontAberration { .. }
                | ChiefRayAngle { .. }
                | RmsSpot { .. }
                | SpotCentroid { .. }
                | WavefrontRms { .. }
                | Mtf { .. }
                | Distortion { .. }
                | LateralColour { .. }
                | RelativeIllumination
                | GhostIrradiance { .. }
                | StrayLightPower { .. }
                | Custom { .. }
        )
    }
    /// Kinds defined at one field point: `FieldSelector::All` is invalid.
    pub fn needs_single_field(&self) -> bool {
        use OperandKind::*;
        match self {
            ToleranceSensitivity { criterion, .. } => criterion.needs_single_field(),
            RealRay { .. }
            | RayAberration { .. }
            | WavefrontAberration { .. }
            | ChiefRayAngle { .. }
            | SpotCentroid { .. }
            | Distortion { .. }
            | LateralColour { .. }
            | RelativeIllumination => true,
            _ => false,
        }
    }
    /// Kinds defined at one wavelength: `WavelengthSelector::All` is invalid.
    pub fn needs_single_wavelength(&self) -> bool {
        use OperandKind::*;
        match self {
            ToleranceSensitivity { criterion, .. } => criterion.needs_single_wavelength(),
            RealRay { .. }
            | RayAberration { .. }
            | WavefrontAberration { .. }
            | ChiefRayAngle { .. }
            | Distortion { .. } => true,
            _ => false,
        }
    }
    /// The [`RayGroup`] of a per-ray operand, if any.
    pub fn ray_group(&self) -> Option<&RayGroup> {
        match self {
            OperandKind::RayAberration { group, .. }
            | OperandKind::WavefrontAberration { group, .. } => group.as_ref(),
            _ => None,
        }
    }
    /// Does a `Centroid` reference lack its [`RayGroup`]?
    pub fn missing_ray_group(&self) -> bool {
        match self {
            OperandKind::RayAberration {
                reference: SpotReference::Centroid,
                group: None,
                ..
            }
            | OperandKind::WavefrontAberration {
                reference: WavefrontReference::Centroid,
                group: None,
                ..
            } => true,
            OperandKind::ToleranceSensitivity { criterion, .. } => criterion.missing_ray_group(),
            _ => false,
        }
    }
    pub fn uses_wavelength(&self) -> bool {
        use OperandKind::*;
        !matches!(
            self,
            TotalTrack { .. }
                | CentreThickness { .. }
                | EdgeThickness { .. }
                | MinThickness { .. }
                | MaxThickness { .. }
                | ParamValue { .. }
                | LateralColour { .. }
                | AxialColour { .. }
                | Combine { .. }
        )
    }
}

/// UI/engine description of an operand kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperandDescriptor {
    pub name: String,
    pub description: String,
    pub unit: String,
    pub capability: Capability,
    pub uses_field: bool,
    pub uses_wavelength: bool,
    /// Example kind with default parameters (a template for the UI).
    pub template: OperandKind,
}

/// Registry of operand kinds: the built-ins plus registered plug-ins.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OperandRegistry {
    pub kinds: Vec<OperandDescriptor>,
}

impl OperandRegistry {
    /// Every built-in kind, with a template instance.
    pub fn builtin() -> Self {
        use OperandKind::*;
        let s = |id: u64| SurfaceId(id);
        let t: Vec<(OperandKind, &str, &str)> = vec![
            (Efl, "Effective focal length", "mm"),
            (Bfl, "Back focal length", "mm"),
            (
                FNumber {
                    kind: FNumberKind::default(),
                },
                "F-number",
                "",
            ),
            (Magnification, "Paraxial magnification", ""),
            (EntrancePupilDiameter, "Entrance pupil diameter", "mm"),
            (ExitPupilPosition, "Exit pupil position from image", "mm"),
            (
                EntrancePupilPosition,
                "Entrance pupil position from first surface",
                "mm",
            ),
            (
                ParaxialRay {
                    ray: ParaxialRayKind::Marginal,
                    quantity: ParaxialQuantity::Height,
                    surface: None,
                },
                "Paraxial ray height / slope / angle",
                "mm / - / deg",
            ),
            (PetzvalSum, "Petzval sum", "1/mm"),
            (
                TotalTrack {
                    from: None,
                    to: None,
                },
                "Total track",
                "mm",
            ),
            (
                CentreThickness {
                    surface: s(1),
                    to: None,
                },
                "Centre thickness",
                "mm",
            ),
            (
                EdgeThickness {
                    surface: s(1),
                    to: None,
                    height: None,
                },
                "Edge thickness",
                "mm",
            ),
            (
                MinThickness {
                    measure: ThicknessMeasure::Centre,
                    medium: MediumFilter::Glass,
                    from: None,
                    to: None,
                },
                "Minimum thickness over gaps",
                "mm",
            ),
            (
                MaxThickness {
                    measure: ThicknessMeasure::Centre,
                    medium: MediumFilter::Glass,
                    from: None,
                    to: None,
                },
                "Maximum thickness over gaps",
                "mm",
            ),
            (
                ParamValue {
                    param: ParamPath::TemperatureC,
                },
                "Model parameter value",
                "",
            ),
            (
                RealRay {
                    quantity: RayQuantity::Y,
                    surface: None,
                    px: 0.0,
                    py: 1.0,
                    global: false,
                },
                "Real ray quantity",
                "mm / deg",
            ),
            (
                RayAberration {
                    axis: Axis::Y,
                    px: 0.0,
                    py: 1.0,
                    reference: SpotReference::Centroid,
                    group: Some(RayGroup::new(PupilSampling::default())),
                },
                "Transverse ray aberration",
                "mm",
            ),
            (
                WavefrontAberration {
                    px: 0.0,
                    py: 1.0,
                    reference: WavefrontReference::Centroid,
                    group: Some(RayGroup::new(PupilSampling::default())),
                },
                "Optical path difference",
                "waves",
            ),
            (ChiefRayAngle { surface: None }, "Chief ray angle", "deg"),
            (
                RmsSpot {
                    reference: SpotReference::Centroid,
                    component: None,
                    sampling: PupilSampling::default(),
                },
                "RMS spot radius",
                "mm",
            ),
            (
                SpotCentroid { axis: Axis::Y },
                "Spot centroid vs chief ray",
                "mm",
            ),
            (
                WavefrontRms {
                    reference: WavefrontReference::Centroid,
                    sampling: PupilSampling::default(),
                },
                "RMS wavefront error",
                "waves",
            ),
            (
                Mtf {
                    frequency: 50.0,
                    orientation: MtfOrientation::Tangential,
                    method: MtfMethod::Diffraction,
                },
                "MTF at frequency",
                "",
            ),
            (
                Distortion {
                    mapping: DistortionMapping::FTanTheta,
                },
                "Distortion",
                "%",
            ),
            (LateralColour { short: 0, long: 2 }, "Lateral colour", "mm"),
            (AxialColour { short: 0, long: 2 }, "Axial colour", "mm"),
            (RelativeIllumination, "Relative illumination", ""),
            (
                GhostFocus {
                    pair: GhostPair {
                        first: s(1),
                        second: s(2),
                    },
                },
                "Ghost focus distance from image",
                "mm",
            ),
            (
                GhostIrradiance {
                    pair: None,
                    metric: GhostMetric::Peak,
                },
                "Ghost irradiance",
                "relative",
            ),
            (
                StrayLightPower {
                    class: StrayPathClass::AllNonPrimary,
                    rays: default_stray_rays(),
                    seed: 0,
                },
                "Stray-light power",
                "relative",
            ),
            (
                ToleranceSensitivity {
                    param: ParamPath::Thickness { surface: s(1) },
                    delta: 0.01,
                    criterion: Box::new(RmsSpot {
                        reference: SpotReference::Centroid,
                        component: None,
                        sampling: PupilSampling::default(),
                    }),
                },
                "Tolerance sensitivity",
                "criterion units",
            ),
            (
                Combine {
                    op: CombineOp::Difference,
                    args: vec![0, 1],
                },
                "Arithmetic on other operands",
                "",
            ),
        ];
        OperandRegistry {
            kinds: t
                .into_iter()
                .map(|(k, d, u)| OperandDescriptor {
                    name: k.name().to_string(),
                    description: d.to_string(),
                    unit: u.to_string(),
                    capability: k.capability(),
                    uses_field: k.uses_field(),
                    uses_wavelength: k.uses_wavelength(),
                    template: k,
                })
                .collect(),
        }
    }

    /// Register a plug-in operand kind (`OperandKind::Custom { name, .. }`).
    pub fn register_custom(
        &mut self,
        name: &str,
        description: &str,
        unit: &str,
        uses_field: bool,
        uses_wavelength: bool,
        template_params: serde_json::Value,
    ) -> Result<(), String> {
        if self.get(name).is_some() {
            return Err(format!("operand kind '{name}' already registered"));
        }
        self.kinds.push(OperandDescriptor {
            name: name.to_string(),
            description: description.to_string(),
            unit: unit.to_string(),
            capability: Capability::Plugin,
            uses_field,
            uses_wavelength,
            template: OperandKind::Custom {
                name: name.to_string(),
                params: template_params,
            },
        });
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&OperandDescriptor> {
        self.kinds.iter().find(|d| d.name == name)
    }

    /// Is this kind known? Built-ins by name; `Custom` kinds only when the
    /// name was registered as a plug-in (a `Custom` named like a built-in,
    /// e.g. `"efl"`, is rejected).
    pub fn knows(&self, kind: &OperandKind) -> bool {
        match (kind, self.get(kind.name())) {
            (OperandKind::ToleranceSensitivity { criterion, .. }, Some(d)) => {
                d.capability == Capability::Tolerance && self.knows(criterion)
            }
            (OperandKind::Custom { .. }, Some(d)) => d.capability == Capability::Plugin,
            (OperandKind::Custom { .. }, None) => false,
            (_, d) => d.is_some_and(|d| d.capability != Capability::Plugin),
        }
    }
}
