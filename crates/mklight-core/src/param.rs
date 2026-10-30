//! Addressing of individual numeric parameters of an `OpticalSystem`.
//!
//! Optimisation variables, tolerance operands, multi-configuration overrides,
//! UI edits and the Python API all go through `ParamPath`. Surfaces are
//! addressed by `SurfaceId`, so paths survive surface inserts and deletes.

use serde::{Deserialize, Serialize};

use crate::error::{MkError, MkResult};
use crate::model::{ApertureSpec, MaterialRef, OpticalSystem, Placement, SurfaceId, SurfaceShape};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "param",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ParamPath {
    Curvature {
        surface: SurfaceId,
    },
    Thickness {
        surface: SurfaceId,
    },
    Conic {
        surface: SurfaceId,
    },
    /// Even-asphere coefficient `index` (0 → A4, 1 → A6, …).
    Asphere {
        surface: SurfaceId,
        index: usize,
    },
    ParaxialFocalLength {
        surface: SurfaceId,
    },
    ModelNd {
        surface: SurfaceId,
    },
    ModelVd {
        surface: SurfaceId,
    },
    ModelDpgf {
        surface: SurfaceId,
    },
    DecenterX {
        surface: SurfaceId,
    },
    DecenterY {
        surface: SurfaceId,
    },
    TiltX {
        surface: SurfaceId,
    },
    TiltY {
        surface: SurfaceId,
    },
    TiltZ {
        surface: SurfaceId,
    },
    SemiDiameter {
        surface: SurfaceId,
    },
    /// The numeric value of the system aperture specification.
    ApertureValue,
    FieldX {
        index: usize,
    },
    FieldY {
        index: usize,
    },
    FieldWeight {
        index: usize,
    },
    WavelengthUm {
        index: usize,
    },
    WavelengthWeight {
        index: usize,
    },
    TemperatureC,
    PressureAtm,
}

crate::serde_util::strict_unit_variants!(
    ParamPath,
    "param",
    ["aperture_value", "temperature_c", "pressure_atm"]
);

fn surf(s: &OpticalSystem, id: SurfaceId) -> MkResult<&crate::model::Surface> {
    s.surfaces
        .iter()
        .find(|x| x.id == id)
        .ok_or_else(|| MkError::InvalidParam(format!("no surface with id {}", id.0)))
}

fn surf_mut(s: &mut OpticalSystem, id: SurfaceId) -> MkResult<&mut crate::model::Surface> {
    s.surfaces
        .iter_mut()
        .find(|x| x.id == id)
        .ok_or_else(|| MkError::InvalidParam(format!("no surface with id {}", id.0)))
}

fn bad(p: &ParamPath, why: &str) -> MkError {
    MkError::InvalidParam(format!("{p:?}: {why}"))
}

impl ParamPath {
    pub fn surface(&self) -> Option<SurfaceId> {
        use ParamPath::*;
        match self {
            Curvature { surface }
            | Thickness { surface }
            | Conic { surface }
            | Asphere { surface, .. }
            | ParaxialFocalLength { surface }
            | ModelNd { surface }
            | ModelVd { surface }
            | ModelDpgf { surface }
            | DecenterX { surface }
            | DecenterY { surface }
            | TiltX { surface }
            | TiltY { surface }
            | TiltZ { surface }
            | SemiDiameter { surface } => Some(*surface),
            _ => None,
        }
    }

    pub fn get(&self, s: &OpticalSystem) -> MkResult<f64> {
        use ParamPath::*;
        Ok(match self {
            Curvature { surface } => match surf(s, *surface)?.shape {
                SurfaceShape::Conic { curvature, .. }
                | SurfaceShape::EvenAsphere { curvature, .. } => curvature,
                SurfaceShape::Plane => 0.0,
                _ => return Err(bad(self, "surface has no curvature")),
            },
            Thickness { surface } => surf(s, *surface)?.thickness,
            Conic { surface } => match surf(s, *surface)?.shape {
                SurfaceShape::Conic { conic, .. } | SurfaceShape::EvenAsphere { conic, .. } => {
                    conic
                }
                _ => return Err(bad(self, "surface has no conic")),
            },
            Asphere { surface, index } => match &surf(s, *surface)?.shape {
                SurfaceShape::EvenAsphere { coefficients, .. } => {
                    coefficients.get(*index).copied().unwrap_or(0.0)
                }
                _ => return Err(bad(self, "surface is not an even asphere")),
            },
            ParaxialFocalLength { surface } => match surf(s, *surface)?.shape {
                SurfaceShape::Paraxial { focal_length } => focal_length,
                _ => return Err(bad(self, "surface is not paraxial")),
            },
            ModelNd { surface } | ModelVd { surface } | ModelDpgf { surface } => {
                match surf(s, *surface)?.material {
                    MaterialRef::Model { nd, vd, dpgf } => match self {
                        ModelNd { .. } => nd,
                        ModelVd { .. } => vd,
                        _ => dpgf,
                    },
                    _ => return Err(bad(self, "material is not a model glass")),
                }
            }
            DecenterX { surface }
            | DecenterY { surface }
            | TiltX { surface }
            | TiltY { surface }
            | TiltZ { surface } => {
                let p = surf(s, *surface)?.placement.unwrap_or_default();
                match self {
                    DecenterX { .. } => p.decenter_x,
                    DecenterY { .. } => p.decenter_y,
                    TiltX { .. } => p.tilt_x,
                    TiltY { .. } => p.tilt_y,
                    _ => p.tilt_z,
                }
            }
            SemiDiameter { surface } => match surf(s, *surface)?.semi_diameter {
                crate::model::SemiDiameter::Fixed { value } => value,
                crate::model::SemiDiameter::Auto => {
                    return Err(bad(self, "semi-diameter is automatic"));
                }
            },
            ApertureValue => match s.aperture {
                ApertureSpec::EntrancePupilDiameter { value }
                | ApertureSpec::ImageSpaceFNumber { value }
                | ApertureSpec::ObjectSpaceNa { value }
                | ApertureSpec::ParaxialWorkingFNumber { value } => value,
                ApertureSpec::FloatByStopSize => {
                    return Err(bad(self, "aperture floats by stop size"));
                }
            },
            FieldX { index } => {
                s.fields
                    .fields
                    .get(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .x
            }
            FieldY { index } => {
                s.fields
                    .fields
                    .get(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .y
            }
            FieldWeight { index } => {
                s.fields
                    .fields
                    .get(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .weight
            }
            WavelengthUm { index } => {
                s.wavelengths
                    .wavelengths
                    .get(*index)
                    .ok_or_else(|| bad(self, "no such wavelength"))?
                    .um
            }
            WavelengthWeight { index } => {
                s.wavelengths
                    .wavelengths
                    .get(*index)
                    .ok_or_else(|| bad(self, "no such wavelength"))?
                    .weight
            }
            TemperatureC => s.environment.temperature_c,
            PressureAtm => s.environment.pressure_atm,
        })
    }

    pub fn set(&self, s: &mut OpticalSystem, v: f64) -> MkResult<()> {
        use ParamPath::*;
        if !v.is_finite() && !matches!(self, Thickness { .. }) {
            return Err(bad(self, "value must be finite"));
        }
        match self {
            Curvature { surface } => {
                let sf = surf_mut(s, *surface)?;
                match &mut sf.shape {
                    SurfaceShape::Conic { curvature, .. }
                    | SurfaceShape::EvenAsphere { curvature, .. } => *curvature = v,
                    SurfaceShape::Plane => {
                        sf.shape = SurfaceShape::Conic {
                            curvature: v,
                            conic: 0.0,
                        }
                    }
                    _ => return Err(bad(self, "surface has no curvature")),
                }
            }
            Thickness { surface } => surf_mut(s, *surface)?.thickness = v,
            Conic { surface } => match &mut surf_mut(s, *surface)?.shape {
                SurfaceShape::Conic { conic, .. } | SurfaceShape::EvenAsphere { conic, .. } => {
                    *conic = v
                }
                _ => return Err(bad(self, "surface has no conic")),
            },
            Asphere { surface, index } => match &mut surf_mut(s, *surface)?.shape {
                SurfaceShape::EvenAsphere { coefficients, .. } => {
                    if coefficients.len() <= *index {
                        coefficients.resize(*index + 1, 0.0);
                    }
                    coefficients[*index] = v;
                }
                _ => return Err(bad(self, "surface is not an even asphere")),
            },
            ParaxialFocalLength { surface } => match &mut surf_mut(s, *surface)?.shape {
                SurfaceShape::Paraxial { focal_length } => *focal_length = v,
                _ => return Err(bad(self, "surface is not paraxial")),
            },
            ModelNd { surface } | ModelVd { surface } | ModelDpgf { surface } => {
                match &mut surf_mut(s, *surface)?.material {
                    MaterialRef::Model { nd, vd, dpgf } => match self {
                        ModelNd { .. } => *nd = v,
                        ModelVd { .. } => *vd = v,
                        _ => *dpgf = v,
                    },
                    _ => return Err(bad(self, "material is not a model glass")),
                }
            }
            DecenterX { surface }
            | DecenterY { surface }
            | TiltX { surface }
            | TiltY { surface }
            | TiltZ { surface } => {
                let sf = surf_mut(s, *surface)?;
                let p: &mut Placement = sf.placement.get_or_insert_with(Placement::default);
                match self {
                    DecenterX { .. } => p.decenter_x = v,
                    DecenterY { .. } => p.decenter_y = v,
                    TiltX { .. } => p.tilt_x = v,
                    TiltY { .. } => p.tilt_y = v,
                    _ => p.tilt_z = v,
                }
            }
            SemiDiameter { surface } => {
                if !(v.is_finite() && v > 0.0) {
                    return Err(bad(self, "semi-diameter must be finite and > 0"));
                }
                surf_mut(s, *surface)?.semi_diameter =
                    crate::model::SemiDiameter::Fixed { value: v }
            }
            ApertureValue => match &mut s.aperture {
                ApertureSpec::EntrancePupilDiameter { value }
                | ApertureSpec::ImageSpaceFNumber { value }
                | ApertureSpec::ObjectSpaceNa { value }
                | ApertureSpec::ParaxialWorkingFNumber { value } => *value = v,
                ApertureSpec::FloatByStopSize => {
                    return Err(bad(self, "aperture floats by stop size"));
                }
            },
            FieldX { index } => {
                s.fields
                    .fields
                    .get_mut(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .x = v
            }
            FieldY { index } => {
                s.fields
                    .fields
                    .get_mut(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .y = v
            }
            FieldWeight { index } => {
                s.fields
                    .fields
                    .get_mut(*index)
                    .ok_or_else(|| bad(self, "no such field"))?
                    .weight = v
            }
            WavelengthUm { index } => {
                s.wavelengths
                    .wavelengths
                    .get_mut(*index)
                    .ok_or_else(|| bad(self, "no such wavelength"))?
                    .um = v
            }
            WavelengthWeight { index } => {
                s.wavelengths
                    .wavelengths
                    .get_mut(*index)
                    .ok_or_else(|| bad(self, "no such wavelength"))?
                    .weight = v
            }
            TemperatureC => s.environment.temperature_c = v,
            PressureAtm => s.environment.pressure_atm = v,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set_roundtrip() {
        let mut s = OpticalSystem::new_empty();
        let id = s.surfaces[1].id;
        let p = ParamPath::Curvature { surface: id };
        p.set(&mut s, 0.02).unwrap();
        assert_eq!(p.get(&s).unwrap(), 0.02);
        let t = ParamPath::Thickness { surface: id };
        t.set(&mut s, 7.5).unwrap();
        assert_eq!(t.get(&s).unwrap(), 7.5);
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<ParamPath>(&json).unwrap(), p);
    }
}
