//! The single physical model (`OpticalSystem`) shared by every solver.

pub mod coating;
pub mod material;
pub mod mechanics;
pub mod solve;
pub mod spec;
pub mod surface;
pub mod validate;

pub use coating::{CoatingDef, CoatingSubstrate, Layer};
pub use material::MaterialRef;
pub use mechanics::{MechanicalPart, ScatterModel, Sensor};
pub use solve::{Solve, SolveKind};
pub use spec::{
    ApertureSpec, Environment, Field, FieldKind, FieldSet, RayAiming, Vignetting, Wavelength,
    WavelengthSet,
};
pub use surface::{
    APERTURE_EDGE_TOLERANCE, ApertureShape, EdgeSpec, Interaction, Placement, SemiDiameter,
    Surface, SurfaceId, SurfaceShape, TransformOrder,
};
pub use validate::ValidationIssue;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{MkError, MkResult};
use crate::param::ParamPath;

/// One configuration of a multi-configuration system (zoom positions, focus
/// distances, alternative filters). Each one is a set of parameter overrides
/// on top of the base system.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub overrides: Vec<(ParamPath, f64)>,
}

/// The complete physical description of an optical system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

    /// Apply configuration `k`'s overrides. A system without configurations
    /// has the single configuration `0`, the system itself; any other `k`
    /// that does not exist is an error.
    ///
    /// Solves are not applied: solves run *after* overrides
    /// (`mklight_trace::solves::apply_solves_configuration`). Never re-apply a
    /// configuration to an already solved system: its overrides would
    /// overwrite solved values, or change what they depend on, without
    /// solving again (`docs/SOLVES.md` §5).
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
