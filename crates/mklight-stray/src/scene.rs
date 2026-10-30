//! The non-sequential scene: placed primitives, what each one does to light,
//! the media between them and the detector.
//!
//! [`Scene::from_system`] builds a scene from an `OpticalSystem` and its
//! `ResolvedSystem`:
//!
//! * **Optical surfaces** become [`Interface`]s: a sag patch covering the
//!   surface's *clear aperture* as geometry, placed by
//!   `ResolvedSurface::frame`. An `Annular` aperture gives a patch with a
//!   central hole, and `Rectangular` / `Elliptical` apertures are masks on
//!   the patch. Surfaces that do nothing optically (dummies, coordinate
//!   breaks, and bare air-to-air planes) get no interface.
//! * **Non-clear zones** (see *Semi-diameters*): the ring between the clear
//!   aperture and the mechanical semi-diameter is an opaque *mount zone*
//!   (ground, bevelled or held by the mount) with the edge finish. The
//!   central hole of an annular aperture is a real hole in a mirror (a
//!   Cassegrain primary) and an opaque obscuration on a refracting surface.
//! * **Repeated surfaces**: a surface that the sequential model visits more
//!   than once (double-pass Mangin mirrors, folded catadioptric systems) is
//!   detected by coincident frame and shape and becomes **one** primitive
//!   with several [`Pass`]es, so the return pass is recognised as designed.
//!   Coincident lens edges and flats are built once.
//! * **Lens edges**: between consecutive interfaces whose gap medium differs
//!   from the ambient (object-space) medium, an edge cylinder at
//!   `max(MSD_front, MSD_back)` joins the two rims. A flat annulus closes the
//!   step on the side with the smaller mechanical semi-diameter, so each
//!   element is a closed solid.
//! * **Aperture stop and field apertures**: a passive surface with a
//!   clipping aperture becomes an opaque plate covering everything outside
//!   the aperture. The aperture stop (`system.stop`) always blocks: its
//!   radius is the aperture shape, the fixed semi-diameter or, for an
//!   automatic one, the supplied clear semi-diameter (the sequential model
//!   aims rays to fill the stop, so nothing designed passes outside it). A
//!   stop on a refracting surface gets an opaque mount zone plus a flat
//!   *stop ring* from its mechanical semi-diameter outward, so light cannot
//!   pass around the element. A stop on a mirror is the mirror's own edge.
//!   A stop whose radius cannot be determined is reported as a warning.
//! * **Mechanics**: `Barrel` (cylinder between two vertex planes, along the
//!   `from` surface's axis) and `AnnularBaffle` from `OpticalSystem::mechanics`.
//! * **Detector**: the image surface, bounded by the sensor rectangle or by
//!   the image surface's clear semi-diameter.
//!
//! # Sides and media
//!
//! Every sag patch has a **Minus** side (local −z) and a **Plus** side
//! (local +z, where its normal points). An interface stores the medium on
//! each side. Designed (sequential) light arrives on the Minus side of a
//! surface after an even number of mirrors and on the Plus side after an odd
//! number (CONVENTIONS §2: light travels toward −z after an odd number of
//! mirrors). The tracer reads the incident medium from the side a ray
//! arrives on, never from state carried by the ray, so a ray that slips
//! through a seam cannot keep the wrong index.
//!
//! # Semi-diameters
//!
//! The clear semi-diameter is the fixed semi-diameter, the aperture shape's
//! bounding radius, or `SceneOptions::clear_semi_diameters` (the sequential
//! tracer's automatic values, filled in by `crate::system::StraySystem`).
//! Two surfaces of one element that intersect inside their clear apertures
//! are reported (rays near the rim then meet the second surface first,
//! which the sequential model ignores). The mechanical
//! semi-diameter is `EdgeSpec::mechanical_semi_diameter`, or the clear value
//! plus `default_edge_margin`. A mirror with a rectangular or elliptical
//! aperture and no explicit mechanical semi-diameter ends at its aperture
//! (no mount zone). By default the zone between the clear aperture and the
//! mechanical semi-diameter is opaque ([`OpaqueKind::Mount`]), so an
//! overfilled entrance does not leak unmounted, unclipped light into the
//! image. `SceneOptions::polished_outer_zone` makes it part of the polished
//! interface instead (hits there are flagged `OUTSIDE_CLIP` and classed
//! `Sneak`); the stop surface always blocks outside its aperture.
//!
//! # Scatter on optical surfaces
//!
//! `Surface::scatter` is applied **relative to each channel**: the scattered
//! fraction of the reflected flux is `R·TIS(β₀_R)` and of the transmitted
//! flux `T·TIS(β₀_T)` (the usual convention in which the TIS of a polished
//! surface is quoted relative to its specular reflectance or transmittance).
//! A BRDF measured in absolute terms on a coated lens surface must be divided
//! by `R` before it is used here. The same lobe is used for both channels.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use mklight_core::error::{MkError, MkResult};
use mklight_core::materials::coating::ResolvedCoating;
use mklight_core::math::{Frame, Mat3, Vec3};
use mklight_core::model::{ApertureShape, Interaction, MechanicalPart, ScatterModel, SurfaceShape};
use mklight_core::resolved::ResolvedSystem;
use mklight_core::snapshot::Snapshot;

use crate::bsdf::Bsdf;
use crate::geom::shapes::conic_domain_radius;
use crate::geom::{Aabb, Bvh, Cylinder, Mask, Primitive, SagPatch, Shape};

/// Index into `Scene::media`.
pub type MediumId = u16;

/// An optical medium at the system wavelengths.
#[derive(Clone, Debug, PartialEq)]
pub struct Medium {
    /// Refractive index per system wavelength.
    pub n: Vec<f64>,
    /// Internal absorption coefficient α (1/mm) per wavelength (Beer–Lambert
    /// `τ = exp(−α L)`, CONVENTIONS §5).
    pub alpha: Vec<f64>,
}

/// Side of a sag patch (local −z or +z).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Minus,
    Plus,
}

impl Side {
    pub fn other(self) -> Side {
        match self {
            Side::Minus => Side::Plus,
            Side::Plus => Side::Minus,
        }
    }
    /// The side a ray with direction `d` arrives from, given the patch normal
    /// `n` (which points to Plus).
    #[inline]
    pub fn arrival(d: Vec3, n: Vec3) -> Side {
        if d.dot(n) < 0.0 {
            Side::Plus
        } else {
            Side::Minus
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterfaceKind {
    /// Refracting surface (R/T/A from the coating).
    Refract,
    /// Mirror: reflective on the designed incident side, opaque on the back.
    Mirror,
    /// Ideal thin lens (`SurfaceShape::Paraxial`): after refraction, the
    /// direction tangents change by `−(x, y)/f`, which images perfectly.
    IdealLens { focal_length: f64 },
}

/// One visit of the designed (sequential) path to an interface.
#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    /// Sequential surface index.
    pub surface: usize,
    /// Side on which designed light arrives on this pass.
    pub incident_side: Side,
    /// The sequential model's clipping aperture on this pass. Hits outside it
    /// are flagged (`flags::OUTSIDE_CLIP`) because the sequential and ghost
    /// solvers discard such rays.
    pub clip: Option<ApertureShape>,
}

/// Scalar relief spectrum precomputed from the resolved media.
#[derive(Clone, Debug)]
pub struct DiffractiveInterface {
    pub phase: mklight_core::model::surface::DiffractivePhase,
    pub efficiency: mklight_core::materials::diffraction::DiffractionEfficiency,
    pub transmitted_blaze: Vec<f64>,
    pub reflected_blaze_minus: Vec<f64>,
    pub reflected_blaze_plus: Vec<f64>,
}

/// An optical surface.
#[derive(Clone, Debug)]
pub struct Interface {
    pub diffraction: Option<Arc<DiffractiveInterface>>,
    /// The sequential passes through this surface, in sequence order (one for
    /// an ordinary surface; two or more for a surface the sequential model
    /// repeats, such as the front face of a Mangin mirror).
    pub passes: Vec<Pass>,
    pub kind: InterfaceKind,
    pub medium_minus: MediumId,
    pub medium_plus: MediumId,
    pub coating: Arc<ResolvedCoating>,
    /// Surface roughness or contamination scatter (`Surface::scatter`),
    /// relative to the R and T channels (module docs).
    pub scatter: Option<Arc<Bsdf>>,
}

impl Interface {
    /// A single-pass interface.
    pub fn single(
        surface: usize,
        kind: InterfaceKind,
        medium_minus: MediumId,
        medium_plus: MediumId,
        incident_side: Side,
        coating: Arc<ResolvedCoating>,
    ) -> Interface {
        Interface {
            diffraction: None,
            passes: vec![Pass {
                surface,
                incident_side,
                clip: None,
            }],
            kind,
            medium_minus,
            medium_plus,
            coating,
            scatter: None,
        }
    }
    /// Sequential index of the first pass.
    pub fn surface(&self) -> usize {
        self.passes[0].surface
    }
    pub fn medium(&self, side: Side) -> MediumId {
        match side {
            Side::Minus => self.medium_minus,
            Side::Plus => self.medium_plus,
        }
    }
    /// The pass a hit arriving on `side` belongs to, and whether `side` is a
    /// designed incident side. `expected` is the next designed surface of
    /// the path: the pass for it wins when the side matches, then any pass
    /// incident on `side`, then the first pass (arrival from a non-designed
    /// side).
    pub fn pass_for(&self, side: Side, expected: Option<usize>) -> (&Pass, bool) {
        if let Some(p) = self
            .passes
            .iter()
            .find(|p| p.incident_side == side && Some(p.surface) == expected)
        {
            return (p, true);
        }
        match self.passes.iter().find(|p| p.incident_side == side) {
            Some(p) => (p, true),
            None => (&self.passes[0], false),
        }
    }
}

/// Kind of non-optical geometry (used to categorise absorbed flux).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaqueKind {
    LensEdge,
    LensFlat,
    /// Non-clear zone of an optical surface: the ring between its clear
    /// aperture and its mechanical semi-diameter, or the central obscuration
    /// of an annular aperture on a refracting surface.
    Mount,
    /// Aperture-stop plate, ring or mount zone, and field-aperture plates.
    StopPlate,
    Barrel,
    Baffle,
    Generic,
}

/// Opaque geometry: a fraction `TIS` of the incident flux is scattered back
/// into the incident hemisphere, `min(specular_reflectance, 1 − TIS)` is
/// reflected specularly, and the rest is absorbed. Both sides behave alike.
#[derive(Clone, Debug)]
pub struct Opaque {
    pub kind: OpaqueKind,
    pub surface: Option<usize>,
    pub finish: Option<Arc<Bsdf>>,
    pub specular_reflectance: f64,
}

/// Which sides of the detector count light.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accept {
    Minus,
    Plus,
    Both,
}

impl Accept {
    pub fn accepts(self, side: Side) -> bool {
        matches!(
            (self, side),
            (Accept::Both, _) | (Accept::Minus, Side::Minus) | (Accept::Plus, Side::Plus)
        )
    }
}

/// The detector's tally grid in the detector primitive's local `(x, y)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorSpec {
    /// Sequential index of the image surface (or `usize::MAX` for synthetic scenes).
    pub surface: usize,
    pub accept: Accept,
    pub half_x: f64,
    pub half_y: f64,
    pub nx: u32,
    pub ny: u32,
}

impl DetectorSpec {
    /// Pixel index of a local point, or `None` outside the grid.
    pub fn pixel(&self, x: f64, y: f64) -> Option<usize> {
        let fx = (x + self.half_x) / (2.0 * self.half_x);
        let fy = (y + self.half_y) / (2.0 * self.half_y);
        if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
            return None;
        }
        let ix = ((fx * self.nx as f64) as u32).min(self.nx - 1);
        let iy = ((fy * self.ny as f64) as u32).min(self.ny - 1);
        Some((iy * self.nx + ix) as usize)
    }
    pub fn pixel_area(&self) -> f64 {
        4.0 * self.half_x * self.half_y / (self.nx as f64 * self.ny as f64)
    }
}

/// What a primitive does to light.
#[derive(Clone, Debug)]
pub enum Behavior {
    Interface(Interface),
    Opaque(Opaque),
    /// Part of the detector (`Scene::detector`).
    Detector,
}

/// Geometry recorded per sequential surface, for sources and reports.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGeom {
    pub frame: Frame,
    pub clear_semi_diameter: f64,
    pub mechanical_semi_diameter: f64,
}

/// A finished scene.
#[derive(Clone, Debug)]
pub struct Scene {
    pub wavelengths_um: Vec<f64>,
    pub wavelength_weights: Vec<f64>,
    pub media: Vec<Medium>,
    /// Medium of the space outside all solids (sources emit into it).
    pub ambient: MediumId,
    behaviors: Vec<Behavior>,
    labels: Vec<String>,
    bvh: Bvh,
    pub detector: DetectorSpec,
    /// Surface indices of the interfaces in designed order, ending with the
    /// detector's surface (empty for synthetic scenes).
    pub sequence: Vec<usize>,
    /// Per sequential surface (index), when it has geometry.
    pub surfaces: Vec<Option<SurfaceGeom>>,
    pub snapshot: Option<Snapshot>,
    /// The options the scene was built with (`None` for synthetic scenes).
    /// Recorded in every result's `RunInfo`.
    pub options: Option<SceneOptions>,
    pub warnings: Vec<String>,
    bounds: Aabb,
}

impl Scene {
    pub fn primitives(&self) -> &[Primitive] {
        self.bvh.primitives()
    }
    pub fn behavior(&self, prim: usize) -> &Behavior {
        &self.behaviors[prim]
    }
    pub fn label(&self, prim: usize) -> &str {
        &self.labels[prim]
    }
    /// CPU acceleration structure (the default `RayCaster`).
    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }
    pub fn num_wavelengths(&self) -> usize {
        self.wavelengths_um.len()
    }
    /// Global bounding box of all geometry.
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }
    /// Centre and radius of a sphere that contains all geometry.
    pub fn bounding_sphere(&self) -> (Vec3, f64) {
        let c = self.bounds.centroid();
        (c, 0.5 * self.bounds.extent().norm())
    }
    /// Position of `surface` in `sequence`, if it is an interface or the detector.
    pub fn sequence_position(&self, surface: usize) -> Option<usize> {
        self.sequence.iter().position(|&s| s == surface)
    }
    /// BLAKE3 hash (hex) of everything the tracer reads: media, primitives,
    /// behaviours, detector and sequence. It changes with any edit made to
    /// the scene after it was built, so `RunInfo` identifies the exact scene.
    pub fn fingerprint(&self) -> String {
        let text = format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.wavelengths_um,
            self.wavelength_weights,
            self.media,
            self.ambient,
            self.bvh.primitives(),
            self.behaviors,
            self.detector,
            self.sequence
        );
        blake3::hash(text.as_bytes()).to_hex().to_string()
    }
}

/// Low-level scene construction, used by [`Scene::from_system`] and for
/// synthetic test scenes.
#[derive(Clone, Debug)]
pub struct SceneBuilder {
    wavelengths_um: Vec<f64>,
    wavelength_weights: Vec<f64>,
    media: Vec<Medium>,
    ambient: MediumId,
    prims: Vec<Primitive>,
    behaviors: Vec<Behavior>,
    labels: Vec<String>,
    detector: Option<DetectorSpec>,
    sequence: Vec<usize>,
    surfaces: Vec<Option<SurfaceGeom>>,
    snapshot: Option<Snapshot>,
    warnings: Vec<String>,
}

impl SceneBuilder {
    pub fn new(wavelengths_um: Vec<f64>, wavelength_weights: Vec<f64>) -> Self {
        SceneBuilder {
            wavelengths_um,
            wavelength_weights,
            media: Vec::new(),
            ambient: 0,
            prims: Vec::new(),
            behaviors: Vec::new(),
            labels: Vec::new(),
            detector: None,
            sequence: Vec::new(),
            surfaces: Vec::new(),
            snapshot: None,
            warnings: Vec::new(),
        }
    }
    pub fn add_medium(&mut self, n: Vec<f64>, alpha: Vec<f64>) -> MediumId {
        self.media.push(Medium { n, alpha });
        (self.media.len() - 1) as MediumId
    }
    pub fn set_ambient(&mut self, m: MediumId) {
        self.ambient = m;
    }
    pub fn add(
        &mut self,
        frame: Frame,
        shape: Shape,
        behavior: Behavior,
        label: impl Into<String>,
    ) -> usize {
        self.prims.push(Primitive::new(frame, shape));
        self.behaviors.push(behavior);
        self.labels.push(label.into());
        self.prims.len() - 1
    }
    pub fn set_detector(&mut self, spec: DetectorSpec) {
        self.detector = Some(spec);
    }
    pub fn set_sequence(&mut self, seq: Vec<usize>) {
        self.sequence = seq;
    }
    pub fn warn(&mut self, w: impl Into<String>) {
        self.warnings.push(w.into());
    }

    pub fn build(self) -> MkResult<Scene> {
        let nw = self.wavelengths_um.len();
        if nw == 0 || self.wavelength_weights.len() != nw {
            return Err(MkError::InvalidModel(
                "scene needs wavelengths with one weight each".into(),
            ));
        }
        if self
            .wavelength_weights
            .iter()
            .any(|w| w.is_nan() || *w < 0.0)
            || self.wavelength_weights.iter().sum::<f64>() <= 0.0
        {
            return Err(MkError::InvalidModel(
                "wavelength weights must be ≥ 0 with a positive sum".into(),
            ));
        }
        for m in &self.media {
            if m.n.len() != nw || m.alpha.len() != nw {
                return Err(MkError::InvalidModel(
                    "every medium needs one index and one α per wavelength".into(),
                ));
            }
            if m.n.iter().any(|n| !(n.is_finite() && *n > 0.0))
                || m.alpha.iter().any(|a| a.is_nan() || *a < 0.0)
            {
                return Err(MkError::InvalidModel("non-physical medium".into()));
            }
        }
        if self.ambient as usize >= self.media.len() {
            return Err(MkError::InvalidModel("ambient medium missing".into()));
        }
        for b in &self.behaviors {
            if let Behavior::Interface(i) = b {
                if i.passes.is_empty() {
                    return Err(MkError::InvalidModel(
                        "interface without a sequential pass".into(),
                    ));
                }
                if i.medium_minus as usize >= self.media.len()
                    || i.medium_plus as usize >= self.media.len()
                {
                    return Err(MkError::InvalidModel(format!(
                        "interface {} refers to a missing medium",
                        i.surface()
                    )));
                }
            }
        }
        let detector = self
            .detector
            .ok_or_else(|| MkError::InvalidModel("scene has no detector".into()))?;
        if !(detector.half_x > 0.0 && detector.half_y > 0.0 && detector.nx > 0 && detector.ny > 0) {
            return Err(MkError::InvalidModel(
                "detector grid must be non-empty".into(),
            ));
        }
        if !self
            .behaviors
            .iter()
            .any(|b| matches!(b, Behavior::Detector))
        {
            return Err(MkError::InvalidModel(
                "scene has no detector geometry".into(),
            ));
        }
        let bounds = self
            .prims
            .iter()
            .fold(Aabb::EMPTY, |a, p| a.union(&p.aabb()));
        Ok(Scene {
            wavelengths_um: self.wavelengths_um,
            wavelength_weights: self.wavelength_weights,
            media: self.media,
            ambient: self.ambient,
            behaviors: self.behaviors,
            labels: self.labels,
            bvh: Bvh::build(self.prims),
            detector,
            sequence: self.sequence,
            surfaces: self.surfaces,
            snapshot: self.snapshot,
            options: None,
            warnings: self.warnings,
            bounds,
        })
    }
}

/// Options for [`Scene::from_system`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SceneOptions {
    /// Clear semi-diameters per surface index for surfaces with automatic
    /// semi-diameters (normally the sequential tracer's values). `None`
    /// entries, or a short list, mean "not supplied".
    pub clear_semi_diameters: Vec<Option<f64>>,
    /// Mechanical margin added to the clear semi-diameter when
    /// `EdgeSpec::mechanical_semi_diameter` is not set (mm).
    pub default_edge_margin: f64,
    /// Finish of lens edges, flats and stop plates without an explicit
    /// `EdgeSpec::edge_scatter`.
    pub default_edge_finish: ScatterModel,
    /// Specular reflectance of all opaque geometry (default 0: matte or black).
    pub mechanical_specular_reflectance: f64,
    /// Detector tally grid.
    pub detector_pixels: (u32, u32),
    /// Include `OpticalSystem::mechanics`.
    pub include_mechanics: bool,
    /// Treat the zone between an optical surface's clear aperture and its
    /// mechanical semi-diameter as the same polished interface (unmounted,
    /// unclipped glass or mirror) instead of an opaque mount zone. The stop
    /// surface ignores this and always blocks outside its aperture.
    pub polished_outer_zone: bool,
}

impl Default for SceneOptions {
    fn default() -> Self {
        SceneOptions {
            clear_semi_diameters: Vec::new(),
            default_edge_margin: 1.0,
            default_edge_finish: ScatterModel::Lambertian { tis: 0.05 },
            mechanical_specular_reflectance: 0.0,
            detector_pixels: (128, 128),
            include_mechanics: true,
            polished_outer_zone: false,
        }
    }
}

fn translate(frame: &Frame, z: f64) -> Frame {
    frame.compose(&Frame {
        origin: Vec3::new(0.0, 0.0, z),
        rot: Mat3::IDENTITY,
    })
}

/// Largest radius a sag patch of `shape` may use.
fn shape_radius_limit(shape: &SurfaceShape) -> f64 {
    match shape {
        SurfaceShape::Conic { curvature, conic } => conic_domain_radius(*curvature, *conic),
        SurfaceShape::EvenAsphere {
            curvature, conic, ..
        }
        | SurfaceShape::OddAsphere {
            curvature, conic, ..
        } => conic_domain_radius(*curvature, *conic) * (1.0 - 1e-6),
        SurfaceShape::QCon {
            curvature,
            conic,
            norm_radius,
            ..
        }
        | SurfaceShape::XYPolynomial {
            curvature,
            conic,
            norm_radius,
            ..
        }
        | SurfaceShape::ZernikeSag {
            curvature,
            conic,
            norm_radius,
            ..
        } => norm_radius.min(conic_domain_radius(*curvature, *conic) * (1.0 - 1e-6)),
        SurfaceShape::QBfs {
            curvature,
            norm_radius,
            ..
        } => norm_radius.min(conic_domain_radius(*curvature, 0.0) * (1.0 - 1e-6)),
        SurfaceShape::Diffractive { base, phase } => {
            phase.norm_radius.min(shape_radius_limit(base))
        }
        // At y = 0 the x section is a circle of curvature c_x; this bound is
        // exact while the y profile sags no further than that circle.
        SurfaceShape::Toroid {
            curvature_x,
            curvature_y,
            conic_y,
            ..
        } => {
            conic_domain_radius(*curvature_y, *conic_y).min(curvature_x.abs().recip())
                * (1.0 - 1e-6)
        }
        SurfaceShape::Biconic {
            curvature_x,
            curvature_y,
            conic_x,
            conic_y,
            ..
        } => {
            conic_domain_radius(*curvature_x, *conic_x)
                .min(conic_domain_radius(*curvature_y, *conic_y))
                * (1.0 - 1e-6)
        }
        _ => f64::INFINITY,
    }
}

/// Shares one `Bsdf` (and its TIS table) between identical models.
#[derive(Default)]
struct BsdfCache(BTreeMap<String, Arc<Bsdf>>);

impl BsdfCache {
    fn get(&mut self, m: &ScatterModel, warnings: &mut Vec<String>) -> MkResult<Arc<Bsdf>> {
        let key = serde_json::to_string(m)?;
        if let Some(b) = self.0.get(&key) {
            return Ok(b.clone());
        }
        let b = Arc::new(Bsdf::new(m)?);
        if b.was_normalised() {
            warnings.push(format!(
                "scatter model {key} exceeded TIS = 1 and was normalised (scale {:.6})",
                b.scale()
            ));
        }
        self.0.insert(key, b.clone());
        Ok(b)
    }
}

/// Two frames that coincide to rounding.
fn same_frame(a: &Frame, b: &Frame) -> bool {
    let tol = 1e-9 * (1.0 + a.origin.norm().max(b.origin.norm()));
    a.origin.max_abs_diff(b.origin) <= tol
        && a.rot
            .m
            .iter()
            .flatten()
            .zip(b.rot.m.iter().flatten())
            .all(|(x, y)| (x - y).abs() <= 1e-12)
}

/// Global keys of the lens edges and flats already built, so that a solid
/// visited twice by the sequential model (a double-pass element) gets its
/// edge geometry once.
#[derive(Default)]
struct EdgeKeys(Vec<(Vec3, Vec3, f64, f64)>);

impl EdgeKeys {
    /// Inserts a key `(p, q, a, b)`, with the points compared unordered.
    /// Returns false when an equal key already exists.
    fn insert(&mut self, p: Vec3, q: Vec3, a: f64, b: f64) -> bool {
        let tol = 1e-9 * (1.0 + p.norm().max(q.norm()) + a.abs().max(b.abs()));
        let near = |u: Vec3, v: Vec3| u.max_abs_diff(v) <= tol;
        let dup = self.0.iter().any(|&(p2, q2, a2, b2)| {
            ((near(p, p2) && near(q, q2)) || (near(p, q2) && near(q, p2)))
                && (a - a2).abs() <= tol
                && (b - b2).abs() <= tol
        });
        if !dup {
            self.0.push((p, q, a, b));
        }
        !dup
    }
}

/// An interface primitive under construction.
struct IfaceGeom {
    prim: usize,
    surface: usize,
    frame: Frame,
    shape: SurfaceShape,
    kind: InterfaceKind,
    media: (usize, usize),
    msd: f64,
    clear: f64,
}

impl Scene {
    /// Builds the non-sequential scene for a resolved system (see the module
    /// documentation).
    pub fn from_system(rs: &ResolvedSystem, opts: &SceneOptions) -> MkResult<Scene> {
        let sys = rs.system();
        let n = rs.num_surfaces();
        let image = n - 1;
        let stop = rs.stop;
        let mut b = SceneBuilder::new(rs.wavelengths_um.clone(), rs.wavelength_weights.clone());
        b.snapshot = Some(rs.snapshot.clone());
        b.surfaces = vec![None; n];
        let mut cache = BsdfCache::default();
        let mut warnings = Vec::new();

        // Medium id i = the space after surface i (0 = object space).
        for s in &rs.surfaces {
            b.add_medium(s.n_after.clone(), s.alpha_after.clone());
        }
        b.set_ambient(0);
        let same_medium = |i: usize, j: usize| {
            let (a, c) = (&rs.surfaces[i], &rs.surfaces[j]);
            a.n_after
                .iter()
                .zip(&c.n_after)
                .all(|(x, y)| (x - y).abs() <= 1e-12)
                && a.alpha_after
                    .iter()
                    .zip(&c.alpha_after)
                    .all(|(x, y)| (x - y).abs() <= 1e-15)
        };

        let supplied = |i: usize| {
            opts.clear_semi_diameters
                .get(i)
                .copied()
                .flatten()
                .filter(|r| r.is_finite() && *r > 0.0)
        };
        let clear_sd = |i: usize| -> Option<f64> {
            let s = &rs.surfaces[i];
            s.fixed_semi_diameter
                .or_else(|| s.aperture.as_ref().map(|a| a.bounding_radius()))
                .or_else(|| supplied(i))
        };
        // The sequential clipping aperture. An automatic semi-diameter does
        // not clip, except at the stop, which the sequential model fills
        // exactly (so its supplied clear semi-diameter is the stop radius).
        let clip_of = |i: usize| -> Option<ApertureShape> {
            let s = &rs.surfaces[i];
            s.aperture.clone().or_else(|| {
                s.fixed_semi_diameter
                    .or_else(|| supplied(i).filter(|_| i == stop))
                    .map(|radius| ApertureShape::Circular { radius })
            })
        };
        let edge_finish = |i: usize, cache: &mut BsdfCache, w: &mut Vec<String>| {
            let m = sys.surfaces[i]
                .edge
                .edge_scatter
                .clone()
                .unwrap_or_else(|| opts.default_edge_finish.clone());
            cache.get(&m, w)
        };
        let opaque = |kind, surface, finish| {
            Behavior::Opaque(Opaque {
                kind,
                surface: Some(surface),
                finish: Some(finish),
                specular_reflectance: opts.mechanical_specular_reflectance,
            })
        };

        // Classify surfaces and lay out interfaces.
        let mut mirrors = 0usize;
        struct Rim {
            surface: usize,
            msd: f64,
            clear: f64,
            frame: Frame,
            shape: SurfaceShape,
        }
        let mut rims: Vec<Rim> = Vec::new();
        let mut ifaces: Vec<IfaceGeom> = Vec::new();
        let mut plates: Vec<usize> = Vec::new();
        // (surface, frame, shape, msd) of a stop on a refracting surface.
        let mut stop_ring: Option<(usize, Frame, SurfaceShape, f64)> = None;
        let mut seq = Vec::new();
        let mut image_side = Side::Minus;
        for i in 1..n {
            let rsurf = &rs.surfaces[i];
            let incident = if mirrors.is_multiple_of(2) {
                Side::Minus
            } else {
                Side::Plus
            };
            if i == image {
                image_side = incident;
                break;
            }
            let is_paraxial = matches!(rsurf.shape, SurfaceShape::Paraxial { .. });
            let passive = rsurf.is_passive()
                || (rsurf.interaction == Interaction::Refract
                    && !is_paraxial
                    && !matches!(rsurf.shape, SurfaceShape::Diffractive { .. })
                    && same_medium(i - 1, i)
                    && matches!(rsurf.coating, ResolvedCoating::Bare));
            if passive {
                if matches!(rsurf.shape, SurfaceShape::CoordinateBreak) {
                    if i == stop {
                        warnings.push(format!(
                            "the aperture stop (surface {i}) is a coordinate break and blocks nothing"
                        ));
                    }
                } else if clip_of(i).is_some() {
                    plates.push(i);
                } else if i == stop {
                    warnings.push(format!(
                        "the aperture stop (surface {i}) has an automatic semi-diameter and none was supplied in SceneOptions::clear_semi_diameters; no stop plate was built, so the stop blocks nothing"
                    ));
                }
                continue;
            }
            let ca0 = clear_sd(i).ok_or_else(|| {
                MkError::InvalidModel(format!(
                    "surface {i} has an automatic semi-diameter; supply SceneOptions::clear_semi_diameters"
                ))
            })?;
            let mirror = rsurf.is_mirror();
            let kind = match rsurf.shape {
                SurfaceShape::Paraxial { focal_length } => {
                    InterfaceKind::IdealLens { focal_length }
                }
                _ if mirror => InterfaceKind::Mirror,
                _ => InterfaceKind::Refract,
            };
            let media = match incident {
                Side::Minus => (i - 1, i),
                Side::Plus => (i, i - 1),
            };
            let pass = Pass {
                surface: i,
                incident_side: incident,
                clip: clip_of(i),
            };
            seq.push(i);
            if mirror {
                mirrors += 1;
            }

            // A surface the sequence visits again: merge with the first visit.
            if let Some(g) = ifaces
                .iter()
                .find(|g| same_frame(&g.frame, &rsurf.frame) && g.shape == rsurf.shape)
            {
                let consistent = g.kind == kind
                    && same_medium(g.media.0, media.0)
                    && same_medium(g.media.1, media.1);
                if consistent {
                    let (first, prim, msd, clear, frame) =
                        (g.surface, g.prim, g.msd, g.clear, g.frame);
                    if let Behavior::Interface(f) = &mut b.behaviors[prim] {
                        f.passes.push(pass);
                    }
                    let (sa, sb) = (&sys.surfaces[first], &sys.surfaces[i]);
                    if sa.coating != sb.coating || sa.scatter != sb.scatter {
                        warnings.push(format!(
                            "surface {i} repeats surface {first} with a different coating or scatter model; surface {first}'s is used for both passes"
                        ));
                    }
                    if ca0 > clear * (1.0 + 1e-12) {
                        warnings.push(format!(
                            "surface {i} repeats surface {first} with a larger clear aperture; surface {first}'s geometry is used"
                        ));
                    }
                    b.surfaces[i] = Some(SurfaceGeom {
                        frame,
                        clear_semi_diameter: clear,
                        mechanical_semi_diameter: msd,
                    });
                    rims.push(Rim {
                        surface: i,
                        msd,
                        clear,
                        frame,
                        shape: rsurf.shape.clone(),
                    });
                    continue;
                }
                warnings.push(format!(
                    "surfaces {} and {i} coincide but differ in interaction or media; they are built as two primitives and the first one is hit",
                    g.surface
                ));
            }

            let aperture = rsurf.aperture.clone();
            let masked = matches!(
                aperture,
                Some(ApertureShape::Rectangular { .. } | ApertureShape::Elliptical { .. })
            );
            let explicit_msd = sys.surfaces[i].edge.mechanical_semi_diameter;
            // A rectangular or elliptical mirror ends at its aperture unless
            // it is given a circular substrate.
            let outer_zone = !(mirror && masked && explicit_msd.is_none());
            let default_msd = if outer_zone {
                ca0 + opts.default_edge_margin
            } else {
                ca0
            };
            let mut msd = explicit_msd.unwrap_or(default_msd).max(ca0);
            let lim = shape_radius_limit(&rsurf.shape);
            if msd > lim {
                warnings.push(format!(
                    "surface {i}: mechanical semi-diameter {msd} clamped to the sag domain {lim}"
                ));
                msd = lim;
            }
            let ca = ca0.min(msd);
            let scatter = match &sys.surfaces[i].scatter {
                Some(m) => Some(cache.get(m, &mut warnings)?),
                None => None,
            };

            // Clear zone (the interface) and non-clear zones.
            let is_stop = i == stop;
            let polished = opts.polished_outer_zone && !is_stop;
            let (r_in, clear_mask) = match &aperture {
                Some(ApertureShape::Annular { inner_radius, .. }) => (inner_radius.max(0.0), None),
                Some(a) if masked => (0.0, Some(Mask::Inside(a.clone()))),
                _ => (0.0, None),
            };
            let (patch_r, patch_mask) = if polished && outer_zone {
                (msd, None)
            } else {
                (ca, clear_mask)
            };
            let prim = b.add(
                rsurf.frame,
                Shape::Sag(SagPatch::new(
                    rsurf.shape.clone(),
                    r_in,
                    patch_r,
                    patch_mask,
                )?),
                Behavior::Interface(Interface {
                    diffraction: if let SurfaceShape::Diffractive { phase, .. } = &rsurf.shape {
                        let blaze = |reflected, backward| {
                            rs.wavelengths_um
                                .iter()
                                .map(|&l| {
                                    mklight_core::materials::diffraction::blaze(
                                        rs, i, l, reflected, backward,
                                    )
                                })
                                .collect()
                        };
                        Some(Arc::new(DiffractiveInterface {
                            phase: phase.clone(),
                            efficiency: sys.surfaces[i].diffraction_efficiency,
                            transmitted_blaze: blaze(false, false),
                            reflected_blaze_minus: blaze(true, incident != Side::Minus),
                            reflected_blaze_plus: blaze(true, incident != Side::Plus),
                        }))
                    } else {
                        None
                    },
                    passes: vec![pass],
                    kind,
                    medium_minus: media.0 as MediumId,
                    medium_plus: media.1 as MediumId,
                    coating: Arc::new(rsurf.coating.clone()),
                    scatter,
                }),
                format!("surface {i}"),
            );
            let zone_kind = if is_stop {
                OpaqueKind::StopPlate
            } else {
                OpaqueKind::Mount
            };
            if outer_zone && !polished {
                let zone = if masked {
                    Some(SagPatch::new(
                        rsurf.shape.clone(),
                        0.0,
                        msd,
                        aperture.clone().map(Mask::Outside),
                    )?)
                } else if msd > ca * (1.0 + 1e-12) {
                    Some(SagPatch::new(rsurf.shape.clone(), ca, msd, None)?)
                } else {
                    None
                };
                if let Some(z) = zone {
                    let fin = edge_finish(i, &mut cache, &mut warnings)?;
                    b.add(
                        rsurf.frame,
                        Shape::Sag(z),
                        opaque(zone_kind, i, fin),
                        format!("mount zone {i}"),
                    );
                }
            }
            if r_in > 0.0 && !mirror {
                let fin = edge_finish(i, &mut cache, &mut warnings)?;
                b.add(
                    rsurf.frame,
                    Shape::Sag(SagPatch::new(rsurf.shape.clone(), 0.0, r_in, None)?),
                    opaque(zone_kind, i, fin),
                    format!("obscuration {i}"),
                );
            }
            if is_stop && !mirror {
                stop_ring = Some((i, rsurf.frame, rsurf.shape.clone(), msd));
            }
            b.surfaces[i] = Some(SurfaceGeom {
                frame: rsurf.frame,
                clear_semi_diameter: ca,
                mechanical_semi_diameter: msd,
            });
            ifaces.push(IfaceGeom {
                prim,
                surface: i,
                frame: rsurf.frame,
                shape: rsurf.shape.clone(),
                kind,
                media,
                msd,
                clear: ca,
            });
            rims.push(Rim {
                surface: i,
                msd,
                clear: ca,
                frame: rsurf.frame,
                shape: rsurf.shape.clone(),
            });
        }
        seq.push(image);

        // Lens edges and flats between consecutive interfaces in a solid gap.
        let mut keys = EdgeKeys::default();
        for w in rims.windows(2) {
            let (a, c) = (&w[0], &w[1]);
            if same_medium(a.surface, 0) {
                continue;
            }
            let rel = c.frame.relative_to(&a.frame);
            let axial =
                rel.rot.is_identity() && rel.origin.x.abs() < 1e-12 && rel.origin.y.abs() < 1e-12;
            if !axial {
                warnings.push(format!(
                    "edge between surfaces {} and {} is tilted or decentred; its edge cylinder follows surface {}'s axis (approximation)",
                    a.surface, c.surface, a.surface
                ));
            }
            let za = a.shape.sag(a.msd * a.msd).unwrap_or(0.0);
            let zc = rel.origin.z + c.shape.sag(c.msd * c.msd).unwrap_or(0.0);
            // Thickness at the smaller clear radius: a sign change means the
            // two surfaces intersect inside their clear apertures, so rays
            // near the rim meet the second surface first (the element is
            // not physical at these semi-diameters, and non-sequential
            // results there differ from the sequential model).
            let rc = a.clear.min(c.clear);
            let ta = a.shape.sag(rc * rc).unwrap_or(0.0);
            let tc = rel.origin.z + c.shape.sag(rc * rc).unwrap_or(0.0);
            if (tc - ta) * rel.origin.z < 0.0 {
                warnings.push(format!(
                    "surfaces {} and {} intersect inside their clear apertures (edge thickness {:.4} mm at r = {rc:.4} mm); rays near the rim leave the designed path",
                    a.surface,
                    c.surface,
                    (tc - ta) * rel.origin.z.signum()
                ));
            } else if (zc - za) * rel.origin.z < 0.0 {
                warnings.push(format!(
                    "negative edge thickness between surfaces {} and {}",
                    a.surface, c.surface
                ));
            }
            let re = a.msd.max(c.msd);
            let on_axis = |z: f64| a.frame.to_global_point(Vec3::new(0.0, 0.0, z));
            if (zc - za).abs() > 1e-9 && keys.insert(on_axis(za), on_axis(zc), re, re) {
                let fin = edge_finish(a.surface, &mut cache, &mut warnings)?;
                b.add(
                    a.frame,
                    Shape::Cylinder(Cylinder::new(re, za, zc)?),
                    opaque(OpaqueKind::LensEdge, a.surface, fin),
                    format!("edge {}–{}", a.surface, c.surface),
                );
            }
            for (r, z, s) in [(a.msd, za, a.surface), (c.msd, zc, c.surface)] {
                let centre = on_axis(z);
                if r < re * (1.0 - 1e-12) && keys.insert(centre, centre, r, re) {
                    let fin = edge_finish(s, &mut cache, &mut warnings)?;
                    b.add(
                        translate(&a.frame, z),
                        Shape::Sag(SagPatch::new(SurfaceShape::Plane, r, re, None)?),
                        opaque(OpaqueKind::LensFlat, s, fin),
                        format!("flat {s}"),
                    );
                }
            }
        }

        // Mechanics.
        let mut max_radius = rims.iter().map(|r| r.msd).fold(0.0, f64::max);
        if opts.include_mechanics {
            for (k, part) in sys.mechanics.iter().enumerate() {
                let find = |id| {
                    sys.surface_index(id).ok_or_else(|| {
                        MkError::InvalidModel(format!(
                            "mechanical part {k}: unknown surface id {}",
                            id.0
                        ))
                    })
                };
                match part {
                    MechanicalPart::Barrel {
                        from,
                        to,
                        inner_radius,
                        finish,
                    } => {
                        let (i, j) = (find(*from)?, find(*to)?);
                        let fi = rs.surfaces[i].frame;
                        let zt = rs.surfaces[j].frame.relative_to(&fi).origin.z;
                        let fin = cache.get(finish, &mut warnings)?;
                        b.add(
                            fi,
                            Shape::Cylinder(Cylinder::new(*inner_radius, 0.0, zt)?),
                            opaque(OpaqueKind::Barrel, i, fin),
                            format!("barrel {k}"),
                        );
                        max_radius = max_radius.max(*inner_radius);
                    }
                    MechanicalPart::AnnularBaffle {
                        surface,
                        z,
                        inner_radius,
                        outer_radius,
                        finish,
                    } => {
                        let i = find(*surface)?;
                        let fin = cache.get(finish, &mut warnings)?;
                        b.add(
                            translate(&rs.surfaces[i].frame, *z),
                            Shape::Sag(SagPatch::new(
                                SurfaceShape::Plane,
                                *inner_radius,
                                *outer_radius,
                                None,
                            )?),
                            opaque(OpaqueKind::Baffle, i, fin),
                            format!("baffle {k}"),
                        );
                    }
                }
            }
        }

        // Aperture-stop and field-aperture plates on passive surfaces.
        for i in plates {
            let rsurf = &rs.surfaces[i];
            let clip = clip_of(i).expect("plate has a clip");
            let r_clip = clip.bounding_radius();
            let mut r_out = sys.surfaces[i].edge.mechanical_semi_diameter.unwrap_or(
                (r_clip + opts.default_edge_margin).max(max_radius + opts.default_edge_margin),
            );
            r_out = r_out.min(shape_radius_limit(&rsurf.shape));
            if r_out <= r_clip {
                warnings.push(format!("stop plate at surface {i} has no area"));
                continue;
            }
            let fin = edge_finish(i, &mut cache, &mut warnings)?;
            b.add(
                rsurf.frame,
                Shape::Sag(SagPatch::new(
                    rsurf.shape.clone(),
                    0.0,
                    r_out,
                    Some(Mask::Outside(clip)),
                )?),
                opaque(OpaqueKind::StopPlate, i, fin),
                format!("stop plate {i}"),
            );
            b.surfaces[i] = Some(SurfaceGeom {
                frame: rsurf.frame,
                clear_semi_diameter: r_clip,
                mechanical_semi_diameter: r_out,
            });
        }

        // A stop on a refracting surface: a flat ring from the element's rim
        // outward, so light cannot pass around the element.
        if let Some((i, frame, shape, msd)) = stop_ring {
            let r_out = (max_radius + opts.default_edge_margin).max(msd + opts.default_edge_margin);
            let z = shape.sag(msd * msd).unwrap_or(0.0);
            let fin = edge_finish(i, &mut cache, &mut warnings)?;
            b.add(
                translate(&frame, z),
                Shape::Sag(SagPatch::new(SurfaceShape::Plane, msd, r_out, None)?),
                opaque(OpaqueKind::StopPlate, i, fin),
                format!("stop ring {i}"),
            );
        }

        // Detector at the image surface.
        let isurf = &rs.surfaces[image];
        let (hx, hy, mask, r) = match &sys.sensor {
            Some(s) => {
                let (hx, hy) = (s.half_width(), s.half_height());
                (
                    hx,
                    hy,
                    Some(Mask::InsideRect {
                        half_x: hx,
                        half_y: hy,
                    }),
                    hx.hypot(hy),
                )
            }
            None => {
                let r = clear_sd(image).ok_or_else(|| {
                    MkError::InvalidModel(
                        "image surface needs a sensor, a fixed semi-diameter or a supplied clear semi-diameter".into(),
                    )
                })?;
                (r, r, None, r)
            }
        };
        let lim = shape_radius_limit(&isurf.shape);
        if r > lim {
            return Err(MkError::InvalidModel(
                "detector extends beyond the image surface's sag domain".into(),
            ));
        }
        let det_shape = match isurf.shape {
            SurfaceShape::CoordinateBreak | SurfaceShape::Paraxial { .. } => SurfaceShape::Plane,
            ref s => s.clone(),
        };
        b.add(
            isurf.frame,
            Shape::Sag(SagPatch::new(det_shape, 0.0, r, mask)?),
            Behavior::Detector,
            "detector",
        );
        b.surfaces[image] = Some(SurfaceGeom {
            frame: isurf.frame,
            clear_semi_diameter: r,
            mechanical_semi_diameter: r,
        });
        b.set_detector(DetectorSpec {
            surface: image,
            accept: match image_side {
                Side::Minus => Accept::Minus,
                Side::Plus => Accept::Plus,
            },
            half_x: hx,
            half_y: hy,
            nx: opts.detector_pixels.0,
            ny: opts.detector_pixels.1,
        });
        b.set_sequence(seq);
        b.warnings.extend(rs.warnings.iter().cloned());
        b.warnings.extend(warnings);
        let mut scene = b.build()?;
        scene.options = Some(opts.clone());
        Ok(scene)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_patch_radii_stay_in_the_sag_domain() {
        // A toroid whose x circle (radius 4) is tighter than its y conic.
        let toroid = SurfaceShape::Toroid {
            curvature_x: 0.25,
            curvature_y: 0.1,
            conic_y: 0.0,
            y_asphere: vec![],
        };
        assert!(shape_radius_limit(&toroid) <= 4.0);
        let samples = mklight_core::samples::all().into_iter();
        let shapes = samples.flat_map(|(_, s)| s.surfaces.into_iter().map(|s| s.shape));
        for shape in shapes.filter(|s| !s.is_legacy()).chain([toroid]) {
            let r = shape_radius_limit(&shape).min(1e3);
            for k in 0..3600 {
                let a = k as f64 * std::f64::consts::TAU / 3600.0;
                assert!(
                    mklight_core::surface_math::in_domain(&shape, r * a.cos(), r * a.sin()),
                    "{shape:?} at r = {r}"
                );
            }
        }
    }
}
