//! The single physical model (`OpticalSystem`) shared by every solver.

pub mod coating;
pub mod material;
pub mod mechanics;
pub mod solve;
pub mod spec;
pub mod surface;

pub use coating::{CoatingDef, Layer};
pub use material::MaterialRef;
pub use mechanics::{MechanicalPart, ScatterModel, Sensor};
pub use solve::{Solve, SolveKind};
pub use spec::{
    ApertureSpec, Environment, Field, FieldKind, FieldSet, RayAiming, Vignetting, Wavelength,
    WavelengthSet,
};
pub use surface::{
    ApertureShape, EdgeSpec, Interaction, Placement, SemiDiameter, Surface, SurfaceId,
    SurfaceShape, TransformOrder,
};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{MkError, MkResult};
use crate::param::ParamPath;

/// One configuration of a multi-configuration system (zoom positions, focus
/// distances, alternative filters). Each one is a set of parameter overrides
/// on top of the base system.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Configuration {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub overrides: Vec<(ParamPath, f64)>,
}

/// The complete physical description of an optical system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpticalSystem {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub notes: String,
    pub aperture: ApertureSpec,
    pub fields: FieldSet,
    pub wavelengths: WavelengthSet,
    /// `surfaces[0]` is the object, `surfaces[N-1]` is the image.
    pub surfaces: Vec<Surface>,
    /// Index of the aperture stop.
    pub stop: usize,
    #[serde(default)]
    pub ray_aiming: RayAiming,
    #[serde(default)]
    pub environment: Environment,
    #[serde(default)]
    pub coatings: BTreeMap<String, CoatingDef>,
    #[serde(default)]
    pub mechanics: Vec<MechanicalPart>,
    #[serde(default)]
    pub sensor: Option<Sensor>,
    #[serde(default)]
    pub configurations: Vec<Configuration>,
    /// Solves and pickups (`docs/SOLVES.md`). They are applied by
    /// `mklight_trace::solves::apply_solves`, per configuration after its
    /// overrides; the model itself stores whatever values were last solved.
    /// Omitted from JSON when empty, so systems without solves serialise (and
    /// hash) exactly as before solves existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub solves: Vec<Solve>,
}

impl OpticalSystem {
    /// A minimal valid system: object at infinity, one stop, and the image.
    pub fn new_empty() -> Self {
        let mut stop = Surface::new(1, SurfaceShape::Plane, 10.0, MaterialRef::Air);
        stop.label = "STOP".into();
        OpticalSystem {
            name: "Untitled".into(),
            notes: String::new(),
            aperture: ApertureSpec::default(),
            fields: FieldSet::default(),
            wavelengths: WavelengthSet::default(),
            surfaces: vec![
                Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
                stop,
                Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
            ],
            stop: 1,
            ray_aiming: RayAiming::Off,
            environment: Environment::default(),
            coatings: BTreeMap::new(),
            mechanics: Vec::new(),
            sensor: None,
            configurations: Vec::new(),
            solves: Vec::new(),
        }
    }

    pub fn num_surfaces(&self) -> usize {
        self.surfaces.len()
    }
    pub fn image_index(&self) -> usize {
        self.surfaces.len() - 1
    }
    pub fn object_at_infinity(&self) -> bool {
        self.surfaces[0].thickness.is_infinite()
    }
    pub fn surface_index(&self, id: SurfaceId) -> Option<usize> {
        self.surfaces.iter().position(|s| s.id == id)
    }
    /// A fresh `SurfaceId` that no surface uses yet.
    pub fn next_surface_id(&self) -> SurfaceId {
        SurfaceId(
            self.surfaces
                .iter()
                .map(|s| s.id.0)
                .max()
                .map_or(0, |m| m + 1),
        )
    }

    /// Structural validation. It does not trace. Solvers can assume a
    /// validated system.
    pub fn validate(&self) -> MkResult<()> {
        let n = self.surfaces.len();
        if n < 3 {
            return Err(MkError::InvalidModel(
                "a system needs object, at least one surface, and image".into(),
            ));
        }
        if self.stop == 0 || self.stop >= n - 1 {
            return Err(MkError::InvalidModel(format!(
                "stop index {} must be between 1 and {}",
                self.stop,
                n - 2
            )));
        }
        if self.wavelengths.wavelengths.is_empty()
            || self.wavelengths.primary >= self.wavelengths.wavelengths.len()
        {
            return Err(MkError::InvalidModel(
                "wavelength set empty or primary out of range".into(),
            ));
        }
        if self
            .wavelengths
            .wavelengths
            .iter()
            .any(|w| !(w.um > 0.0 && w.um.is_finite()) || w.weight < 0.0)
        {
            return Err(MkError::InvalidModel(
                "wavelengths must be positive and finite with weights ≥ 0".into(),
            ));
        }
        if self.fields.fields.is_empty() {
            return Err(MkError::InvalidModel(
                "at least one field is required".into(),
            ));
        }
        if self.object_at_infinity() && matches!(self.fields.kind, FieldKind::ObjectHeight) {
            return Err(MkError::InvalidModel(
                "object-height fields require a finite object distance".into(),
            ));
        }
        if self.object_at_infinity() && matches!(self.aperture, ApertureSpec::ObjectSpaceNa { .. })
        {
            return Err(MkError::InvalidModel(
                "object-space NA requires a finite object distance".into(),
            ));
        }
        if self.object_at_infinity()
            && let FieldKind::ObjectAngle = self.fields.kind
            && self.fields.fields.iter().any(|f| f.x.hypot(f.y) >= 90.0)
        {
            return Err(MkError::InvalidModel(
                "field angles must be below 90°".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        for (i, s) in self.surfaces.iter().enumerate() {
            if !ids.insert(s.id) {
                return Err(MkError::InvalidModel(format!(
                    "duplicate surface id {}",
                    s.id.0
                )));
            }
            if i > 0 && !s.thickness.is_finite() && i != n - 1 {
                return Err(MkError::InvalidModel(format!(
                    "surface {i}: only the object may have infinite thickness"
                )));
            }
            if let Some(c) = &s.coating
                && !self.coatings.contains_key(c)
            {
                return Err(MkError::UnknownCoating(c.clone()));
            }
            if let SemiDiameter::Fixed { value } = s.semi_diameter
                && !(value > 0.0)
            {
                return Err(MkError::InvalidModel(format!(
                    "surface {i}: fixed semi-diameter must be > 0"
                )));
            }
            if (i == 0 || i == n - 1) && s.is_coordinate_break() {
                return Err(MkError::InvalidModel(
                    "object/image cannot be coordinate breaks".into(),
                ));
            }
        }
        if self.surfaces[n - 1].interaction == Interaction::Reflect {
            return Err(MkError::InvalidModel(
                "the image surface cannot be a mirror".into(),
            ));
        }
        for (k, c) in self.configurations.iter().enumerate() {
            for (p, _) in &c.overrides {
                p.get(self)
                    .map_err(|e| MkError::InvalidModel(format!("configuration {k}: {e}")))?;
            }
        }
        solve::validate_solves(self)
    }

    /// Apply configuration `k`'s overrides. Configuration `0` and an empty
    /// configuration list both mean "base system".
    pub fn with_configuration(&self, k: usize) -> MkResult<OpticalSystem> {
        let mut s = self.clone();
        if let Some(cfg) = self.configurations.get(k) {
            for (p, v) in &cfg.overrides {
                p.set(&mut s, *v)?;
            }
        } else if k != 0 {
            return Err(MkError::InvalidModel(format!(
                "configuration {k} does not exist"
            )));
        }
        Ok(s)
    }
}
