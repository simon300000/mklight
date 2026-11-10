//! Addressing of individual numeric parameters of an `OpticalSystem`.
//!
//! Optimisation variables, tolerance operands, multi-configuration overrides,
//! UI edits and the Python API all go through `ParamPath`. Surfaces are
//! addressed by `SurfaceId`, so paths survive surface inserts and deletes.

use serde::{Deserialize, Serialize};

use crate::error::{MkError, MkResult};
use crate::model::{ApertureSpec, MaterialRef, OpticalSystem, Placement, SurfaceId, SurfaceShape};
use crate::surface_math::MAX_EVEN_TERMS;

/// Coefficient families; indices are zero-based (SURFACES.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoefficientSet {
    Odd,
    Q,
    Phase,
    XAsphere,
    YAsphere,
    Xy,
    Zernike,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeParameter {
    CurvatureX,
    ConicX,
    CurvatureY,
    ConicY,
    NormRadius,
    AstigmatismAngleDeg,
    DesignOrder,
    DesignWavelengthUm,
    TraceOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementComponent {
    DecenterX,
    DecenterY,
    TiltX,
    TiltY,
    TiltZ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrregularityMode {
    Astigmatism,
    Spherical,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "param",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ParamPath {
    /// Reflected test fringes over the overlay's clear radius.
    Irregularity {
        surface: SurfaceId,
        mode: IrregularityMode,
    },
    /// Rigid decentre (mm) or tilt (°) of the range `surface..=last` about
    /// the `pivot` vertex (default `surface`).
    ElementPlacement {
        surface: SurfaceId,
        last: SurfaceId,
        #[serde(default)]
        pivot: Option<SurfaceId>,
        component: PlacementComponent,
    },
    ShapeCoefficient {
        surface: SurfaceId,
        set: CoefficientSet,
        index: usize,
    },
    /// `parameter` avoids collision with this enum's existing JSON tag `param`.
    ShapeParam {
        surface: SurfaceId,
        parameter: ShapeParameter,
    },
    Curvature {
        surface: SurfaceId,
    },
    Thickness {
        surface: SurfaceId,
    },
    Conic {
        surface: SurfaceId,
    },
    /// Even-asphere coefficient `index` (0 → A4, 1 → A6, …), below
    /// [`MAX_EVEN_TERMS`].
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
    /// `Δn_d` of a catalogue glass (`MaterialOffset`).
    MaterialDnD {
        surface: SurfaceId,
    },
    /// `ΔV_d` of a catalogue glass (`MaterialOffset`).
    MaterialDvD {
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
            Irregularity { surface, .. }
            | ElementPlacement { surface, .. }
            | Curvature { surface }
            | Thickness { surface }
            | Conic { surface }
            | Asphere { surface, .. }
            | ShapeCoefficient { surface, .. }
            | ShapeParam { surface, .. }
            | ParaxialFocalLength { surface }
            | ModelNd { surface }
            | ModelVd { surface }
            | ModelDpgf { surface }
            | MaterialDnD { surface }
            | MaterialDvD { surface }
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
            Irregularity { surface, mode } => match surf(s, *surface)?.shape.sag_shape() {
                SurfaceShape::SagOverlay {
                    astigmatism,
                    spherical,
                    ..
                } => match mode {
                    IrregularityMode::Astigmatism => *astigmatism,
                    IrregularityMode::Spherical => *spherical,
                },
                _ => 0.0,
            },
            ElementPlacement {
                surface,
                last,
                pivot,
                component,
            } => {
                let first = s.surface_index(*surface);
                let end = s.surface_index(*last);
                let interior = match (first, end) {
                    (Some(first), Some(end)) => {
                        first > 0 && first <= end && end + 1 < s.surfaces.len()
                    }
                    _ => false,
                };
                if !interior || pivot.is_some_and(|id| s.surface_index(id).is_none()) {
                    return Err(bad(
                        self,
                        "expected an interior, ordered range and existing pivot",
                    ));
                }
                match &s.surfaces[first.unwrap()].element_placement {
                    Some(e) if e.last == *last && e.pivot == *pivot => component.get(&e.motion),
                    Some(_) => return Err(bad(self, "another range or pivot starts here")),
                    None => 0.0,
                }
            }
            ShapeCoefficient {
                surface,
                set,
                index,
            } => {
                let (c, max) = shape_coefficients(&surf(s, *surface)?.shape, *set)
                    .ok_or_else(|| bad(self, "coefficient set does not belong to shape"))?;
                if *index >= max {
                    return Err(bad(self, "[coefficient_count] index exceeds shape limit"));
                }
                c.get(*index).copied().unwrap_or(0.0)
            }
            ShapeParam { surface, parameter } => {
                match shape_parameter(&surf(s, *surface)?.shape, *parameter)
                    .ok_or_else(|| bad(self, "scalar does not belong to shape"))?
                {
                    Slot::Real(x) => *x,
                    Slot::Order(o) => *o as f64,
                }
            }
            Curvature { surface } => match geometry(&surf(s, *surface)?.shape) {
                SurfaceShape::Conic { curvature, .. }
                | SurfaceShape::EvenAsphere { curvature, .. }
                | SurfaceShape::OddAsphere { curvature, .. }
                | SurfaceShape::QCon { curvature, .. }
                | SurfaceShape::QBfs { curvature, .. }
                | SurfaceShape::XYPolynomial { curvature, .. }
                | SurfaceShape::ZernikeSag { curvature, .. }
                | SurfaceShape::Toroid {
                    curvature_y: curvature,
                    ..
                }
                | SurfaceShape::Biconic {
                    curvature_y: curvature,
                    ..
                } => *curvature,
                SurfaceShape::Plane => 0.0,
                _ => return Err(bad(self, "surface has no curvature")),
            },
            Thickness { surface } => surf(s, *surface)?.thickness,
            Conic { surface } => match geometry(&surf(s, *surface)?.shape) {
                SurfaceShape::Conic { conic, .. }
                | SurfaceShape::EvenAsphere { conic, .. }
                | SurfaceShape::OddAsphere { conic, .. }
                | SurfaceShape::QCon { conic, .. }
                | SurfaceShape::XYPolynomial { conic, .. }
                | SurfaceShape::ZernikeSag { conic, .. } => *conic,
                _ => return Err(bad(self, "surface has no conic")),
            },
            Asphere { surface, index } => match geometry(&surf(s, *surface)?.shape) {
                SurfaceShape::EvenAsphere { coefficients, .. } => {
                    if *index >= MAX_EVEN_TERMS {
                        return Err(bad(self, "[coefficient_count] index exceeds shape limit"));
                    }
                    coefficients.get(*index).copied().unwrap_or(0.0)
                }
                _ => return Err(bad(self, "surface is not an even asphere")),
            },
            ParaxialFocalLength { surface } => match surf(s, *surface)?.shape {
                SurfaceShape::Paraxial { focal_length } => focal_length,
                _ => return Err(bad(self, "surface is not paraxial")),
            },
            MaterialDnD { surface } | MaterialDvD { surface } => {
                match surf(s, *surface)?.material {
                    MaterialRef::Catalog { offset, .. } => match self {
                        MaterialDnD { .. } => offset.dn_d,
                        _ => offset.dv_d,
                    },
                    _ => return Err(bad(self, "material is not a catalogue glass")),
                }
            }
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
        // The only non-finite value a path may write is ±∞ on a thickness
        // (an object at infinity); which surfaces may hold it is structural
        // validation.
        if v.is_nan() || (v.is_infinite() && !matches!(self, Thickness { .. })) {
            return Err(bad(self, "value must be finite"));
        }
        match self {
            Irregularity { surface, mode } => {
                let wavelength = s.wavelengths.primary_um();
                let sf = surf_mut(s, *surface)?;
                let radius = match (&sf.aperture, sf.semi_diameter) {
                    (Some(a), _) => Some(a.bounding_radius()),
                    (None, crate::model::SemiDiameter::Fixed { value }) => Some(value),
                    _ => None,
                };
                let shape = sf.shape.sag_shape_mut();
                if matches!(
                    shape,
                    SurfaceShape::CoordinateBreak | SurfaceShape::Paraxial { .. }
                ) {
                    return Err(bad(self, "surface has no manufacturing sag"));
                }
                if !matches!(shape, SurfaceShape::SagOverlay { .. }) {
                    let base = Box::new(std::mem::replace(shape, SurfaceShape::Plane));
                    *shape = SurfaceShape::SagOverlay {
                        base,
                        norm_radius: radius
                            .ok_or_else(|| bad(self, "irregularity needs a fixed clear radius"))?,
                        test_wavelength_um: wavelength,
                        astigmatism: 0.0,
                        astigmatism_angle_deg: 0.0,
                        spherical: 0.0,
                    };
                }
                if let SurfaceShape::SagOverlay {
                    astigmatism,
                    spherical,
                    ..
                } = shape
                {
                    *match mode {
                        IrregularityMode::Astigmatism => astigmatism,
                        IrregularityMode::Spherical => spherical,
                    } = v;
                }
            }
            ElementPlacement {
                surface,
                last,
                pivot,
                component,
            } => {
                self.get(s)?;
                let e = surf_mut(s, *surface)?.element_placement.get_or_insert(
                    crate::model::surface::ElementPlacement {
                        last: *last,
                        pivot: *pivot,
                        motion: Placement::default(),
                    },
                );
                component.set(&mut e.motion, v);
            }
            ShapeCoefficient {
                surface,
                set,
                index,
            } => {
                let (c, max) = shape_coefficients_mut(&mut surf_mut(s, *surface)?.shape, *set)
                    .ok_or_else(|| bad(self, "coefficient set does not belong to shape"))?;
                if *index >= max {
                    return Err(bad(self, "[coefficient_count] index exceeds shape limit"));
                }
                c.resize(c.len().max(*index + 1), 0.0);
                c[*index] = v;
            }
            ShapeParam { surface, parameter } => {
                // Range rules are structural validation; only the integer
                // representation of orders is checked here.
                let slot = shape_parameter_mut(&mut surf_mut(s, *surface)?.shape, *parameter)
                    .ok_or_else(|| bad(self, "scalar does not belong to shape"))?;
                match slot {
                    Slot::Real(x) => *x = v,
                    Slot::Order(o) => {
                        if v.fract() != 0.0 || v.abs() > i32::MAX as f64 {
                            return Err(bad(self, "[invalid_diffraction_order] expected an i32"));
                        }
                        *o = v as i32;
                    }
                }
            }
            Curvature { surface } => {
                let sf = surf_mut(s, *surface)?;
                match geometry_mut(&mut sf.shape) {
                    SurfaceShape::Conic { curvature, .. }
                    | SurfaceShape::EvenAsphere { curvature, .. }
                    | SurfaceShape::OddAsphere { curvature, .. }
                    | SurfaceShape::QCon { curvature, .. }
                    | SurfaceShape::QBfs { curvature, .. }
                    | SurfaceShape::XYPolynomial { curvature, .. }
                    | SurfaceShape::ZernikeSag { curvature, .. }
                    | SurfaceShape::Toroid {
                        curvature_y: curvature,
                        ..
                    }
                    | SurfaceShape::Biconic {
                        curvature_y: curvature,
                        ..
                    } => *curvature = v,
                    shape @ SurfaceShape::Plane => {
                        *shape = SurfaceShape::Conic {
                            curvature: v,
                            conic: 0.0,
                        }
                    }
                    _ => return Err(bad(self, "surface has no curvature")),
                }
            }
            Thickness { surface } => surf_mut(s, *surface)?.thickness = v,
            Conic { surface } => match geometry_mut(&mut surf_mut(s, *surface)?.shape) {
                SurfaceShape::Conic { conic, .. }
                | SurfaceShape::EvenAsphere { conic, .. }
                | SurfaceShape::OddAsphere { conic, .. }
                | SurfaceShape::QCon { conic, .. }
                | SurfaceShape::XYPolynomial { conic, .. }
                | SurfaceShape::ZernikeSag { conic, .. } => *conic = v,
                _ => return Err(bad(self, "surface has no conic")),
            },
            Asphere { surface, index } => match geometry_mut(&mut surf_mut(s, *surface)?.shape) {
                SurfaceShape::EvenAsphere { coefficients, .. } => {
                    if *index >= MAX_EVEN_TERMS {
                        return Err(bad(self, "[coefficient_count] index exceeds shape limit"));
                    }
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
            MaterialDnD { surface } | MaterialDvD { surface } => {
                match &mut surf_mut(s, *surface)?.material {
                    MaterialRef::Catalog { offset, .. } => match self {
                        MaterialDnD { .. } => offset.dn_d = v,
                        _ => offset.dv_d = v,
                    },
                    _ => return Err(bad(self, "material is not a catalogue glass")),
                }
            }
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
                let p: &mut Placement = sf.placement.get_or_insert_with(|| Placement {
                    restore_after: true,
                    ..Placement::default()
                });
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

impl PlacementComponent {
    pub fn get(self, p: &Placement) -> f64 {
        match self {
            Self::DecenterX => p.decenter_x,
            Self::DecenterY => p.decenter_y,
            Self::TiltX => p.tilt_x,
            Self::TiltY => p.tilt_y,
            Self::TiltZ => p.tilt_z,
        }
    }
    pub fn set(self, p: &mut Placement, v: f64) {
        match self {
            Self::DecenterX => p.decenter_x = v,
            Self::DecenterY => p.decenter_y = v,
            Self::TiltX => p.tilt_x = v,
            Self::TiltY => p.tilt_y = v,
            Self::TiltZ => p.tilt_z = v,
        }
    }
}

fn geometry(shape: &SurfaceShape) -> &SurfaceShape {
    if let SurfaceShape::Diffractive { base, .. } | SurfaceShape::SagOverlay { base, .. } = shape {
        geometry(base)
    } else {
        shape
    }
}
fn geometry_mut(shape: &mut SurfaceShape) -> &mut SurfaceShape {
    if let SurfaceShape::Diffractive { base, .. } | SurfaceShape::SagOverlay { base, .. } = shape {
        geometry_mut(base)
    } else {
        shape
    }
}
// One body serves shared and mutable access: match ergonomics bind the
// fields as `&` or `&mut` according to the scrutinee.
macro_rules! coefficient_slot {
    ($shape:expr, $set:expr) => {{
        use crate::model::surface::DiffractivePhase;
        use crate::surface_math::*;
        use CoefficientSet::*;
        use SurfaceShape::*;
        Some(match ($shape, $set) {
            (OddAsphere { coefficients, .. }, Odd) => (coefficients, MAX_ODD_TERMS),
            (QCon { coefficients, .. } | QBfs { coefficients, .. }, Q) => {
                (coefficients, MAX_Q_TERMS)
            }
            (
                Diffractive {
                    phase: DiffractivePhase { coefficients, .. },
                    ..
                },
                Phase,
            ) => (coefficients, MAX_PHASE_TERMS),
            (Biconic { x_asphere, .. }, XAsphere) => (x_asphere, MAX_AXIS_TERMS),
            (Biconic { y_asphere, .. } | Toroid { y_asphere, .. }, YAsphere) => {
                (y_asphere, MAX_AXIS_TERMS)
            }
            (XYPolynomial { coefficients, .. }, Xy) => (coefficients, MAX_XY_TERMS),
            (
                ZernikeSag {
                    coefficients,
                    ordering,
                    ..
                },
                Zernike,
            ) => (coefficients, ordering.max_terms()),
            _ => return None,
        })
    }};
}
fn shape_coefficients(shape: &SurfaceShape, set: CoefficientSet) -> Option<(&Vec<f64>, usize)> {
    if let SurfaceShape::SagOverlay { base, .. } = shape {
        return shape_coefficients(base, set);
    }
    coefficient_slot!(shape, set)
}
fn shape_coefficients_mut(
    shape: &mut SurfaceShape,
    set: CoefficientSet,
) -> Option<(&mut Vec<f64>, usize)> {
    if let SurfaceShape::SagOverlay { base, .. } = shape {
        return shape_coefficients_mut(base, set);
    }
    coefficient_slot!(shape, set)
}

/// A scalar shape parameter: a real value or a diffraction order.
enum Slot<F, I> {
    Real(F),
    Order(I),
}
macro_rules! parameter_slot {
    ($shape:expr, $param:expr) => {{
        use crate::model::surface::DiffractivePhase as P;
        use ShapeParameter::*;
        use Slot::*;
        use SurfaceShape::*;
        Some(match ($shape, $param) {
            (Toroid { curvature_x, .. } | Biconic { curvature_x, .. }, CurvatureX) => {
                Real(curvature_x)
            }
            (Toroid { curvature_y, .. } | Biconic { curvature_y, .. }, CurvatureY) => {
                Real(curvature_y)
            }
            (Biconic { conic_x, .. }, ConicX) => Real(conic_x),
            (Toroid { conic_y, .. } | Biconic { conic_y, .. }, ConicY) => Real(conic_y),
            (
                QCon { norm_radius, .. }
                | QBfs { norm_radius, .. }
                | XYPolynomial { norm_radius, .. }
                | ZernikeSag { norm_radius, .. }
                | Diffractive {
                    phase: P { norm_radius, .. },
                    ..
                },
                NormRadius,
            ) => Real(norm_radius),
            (
                Diffractive {
                    phase:
                        P {
                            design_wavelength_um,
                            ..
                        },
                    ..
                },
                DesignWavelengthUm,
            ) => Real(design_wavelength_um),
            (
                Diffractive {
                    phase: P { design_order, .. },
                    ..
                },
                DesignOrder,
            ) => Order(design_order),
            (
                Diffractive {
                    phase: P { trace_order, .. },
                    ..
                },
                TraceOrder,
            ) => Order(trace_order),
            _ => return None,
        })
    }};
}
fn shape_parameter(shape: &SurfaceShape, param: ShapeParameter) -> Option<Slot<&f64, &i32>> {
    if param == ShapeParameter::AstigmatismAngleDeg {
        return match shape.sag_shape() {
            SurfaceShape::SagOverlay {
                astigmatism_angle_deg,
                ..
            } => Some(Slot::Real(astigmatism_angle_deg)),
            _ => None,
        };
    }
    if let SurfaceShape::SagOverlay { base, .. } = shape {
        return shape_parameter(base, param);
    }
    parameter_slot!(shape, param)
}
fn shape_parameter_mut(
    shape: &mut SurfaceShape,
    param: ShapeParameter,
) -> Option<Slot<&mut f64, &mut i32>> {
    if param == ShapeParameter::AstigmatismAngleDeg {
        return match shape.sag_shape_mut() {
            SurfaceShape::SagOverlay {
                astigmatism_angle_deg,
                ..
            } => Some(Slot::Real(astigmatism_angle_deg)),
            _ => None,
        };
    }
    if let SurfaceShape::SagOverlay { base, .. } = shape {
        return shape_parameter_mut(base, param);
    }
    parameter_slot!(shape, param)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_curvature_edits_y_section_only() {
        for mut system in [
            crate::samples::toroidal_element(),
            crate::samples::biconic_element(),
        ] {
            let surface = system.surfaces[1].id;
            let y = ParamPath::ShapeParam {
                surface,
                parameter: ShapeParameter::CurvatureY,
            };
            let x = ParamPath::ShapeParam {
                surface,
                parameter: ShapeParameter::CurvatureX,
            };
            let original_x = x.get(&system).unwrap();
            let generic = ParamPath::Curvature { surface };
            assert_eq!(generic.get(&system).unwrap(), y.get(&system).unwrap());
            generic.set(&mut system, 0.031).unwrap();
            assert_eq!(y.get(&system).unwrap(), 0.031);
            assert_eq!(x.get(&system).unwrap(), original_x);
        }
    }

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
    #[test]
    fn extended_paths_edit_the_geometry_and_phase_without_changing_metadata() {
        use crate::samples;
        let mut s = samples::hybrid_diffractive_achromat();
        let id = s.surfaces[2].id;
        let base = ParamPath::Curvature { surface: id };
        base.set(&mut s, 0.01).unwrap();
        assert_eq!(base.get(&s).unwrap(), 0.01);
        let p = ParamPath::ShapeCoefficient {
            surface: id,
            set: CoefficientSet::Phase,
            index: 3,
        };
        p.set(&mut s, -0.2).unwrap();
        assert_eq!(p.get(&s).unwrap(), -0.2);
        let order = ParamPath::ShapeParam {
            surface: id,
            parameter: ShapeParameter::TraceOrder,
        };
        assert!(
            order
                .set(&mut s, 0.5)
                .unwrap_err()
                .to_string()
                .contains("invalid_diffraction_order")
        );
        order.set(&mut s, -2.0).unwrap();
        assert_eq!(order.get(&s).unwrap(), -2.0);
        let design = ParamPath::ShapeParam {
            surface: id,
            parameter: ShapeParameter::DesignOrder,
        };
        assert_eq!(design.get(&s).unwrap(), 1.0);
        // Order zero is representable; structural validation rejects it.
        design.set(&mut s, 0.0).unwrap();
        let e = s.validate().unwrap_err().to_string();
        assert!(e.contains("invalid_diffraction_order"), "{e}");
        design.set(&mut s, 1.0).unwrap();
        for path in [p, order, design] {
            let json = serde_json::to_string(&path).unwrap();
            assert_eq!(serde_json::from_str::<ParamPath>(&json).unwrap(), path);
        }
        s.validate().unwrap();
    }

    #[test]
    fn even_asphere_index_is_bounded() {
        let mut s = crate::samples::petzval_asphere();
        let k = s
            .surfaces
            .iter()
            .position(|sf| matches!(sf.shape, SurfaceShape::EvenAsphere { .. }))
            .unwrap();
        let id = s.surfaces[k].id;
        let last = ParamPath::Asphere {
            surface: id,
            index: MAX_EVEN_TERMS - 1,
        };
        last.set(&mut s, 1e-40).unwrap();
        assert_eq!(last.get(&s).unwrap(), 1e-40);
        s.validate().unwrap();
        for index in [MAX_EVEN_TERMS, 1_000_000_000, usize::MAX] {
            let p = ParamPath::Asphere { surface: id, index };
            let e = p.set(&mut s, 1.0).unwrap_err().to_string();
            assert!(e.contains("coefficient_count"), "{e}");
            assert!(p.get(&s).is_err());
        }
        if let SurfaceShape::EvenAsphere { coefficients, .. } = &mut s.surfaces[k].shape {
            assert_eq!(coefficients.len(), MAX_EVEN_TERMS);
            coefficients.push(0.0);
        }
        let e = s.validate().unwrap_err().to_string();
        assert!(e.contains("coefficient_count"), "{e}");
    }

    #[test]
    fn only_a_thickness_takes_an_infinity_and_nothing_takes_nan() {
        let mut s = OpticalSystem::new_empty();
        let surface = s.surfaces[0].id;
        let t = ParamPath::Thickness { surface };
        for v in [f64::NEG_INFINITY, f64::INFINITY] {
            t.set(&mut s, v).unwrap();
            assert_eq!(t.get(&s).unwrap(), v);
        }
        assert!(t.set(&mut s, f64::NAN).is_err());
        assert_eq!(t.get(&s).unwrap(), f64::INFINITY);
        let c = ParamPath::Curvature {
            surface: s.surfaces[1].id,
        };
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(c.set(&mut s, v).is_err());
        }
    }

    #[test]
    fn all_new_coefficient_families_enforce_limits_before_resizing() {
        use crate::samples;
        use CoefficientSet::*;
        for (mut s, set, max) in [
            (samples::odd_asphere_element(), Odd, 20),
            (samples::q_asphere_lens(), Q, 20),
            (samples::biconic_element(), XAsphere, 10),
            (samples::toroidal_element(), YAsphere, 10),
            (samples::xy_freeform(), Xy, 65),
            (samples::zernike_sag(), Zernike, 37),
        ] {
            let id = s.surfaces[1].id;
            let p = ParamPath::ShapeCoefficient {
                surface: id,
                set,
                index: max - 1,
            };
            p.set(&mut s, 1e-10).unwrap();
            assert_eq!(p.get(&s).unwrap(), 1e-10);
            s.validate().unwrap();
            let p = ParamPath::ShapeCoefficient {
                surface: id,
                set,
                index: usize::MAX,
            };
            assert!(
                p.set(&mut s, 1.0)
                    .unwrap_err()
                    .to_string()
                    .contains("coefficient_count")
            );
            assert!(p.get(&s).is_err());
        }
        let mut s = samples::biconic_element();
        let id = s.surfaces[1].id;
        for parameter in [
            ShapeParameter::CurvatureX,
            ShapeParameter::CurvatureY,
            ShapeParameter::ConicX,
            ShapeParameter::ConicY,
        ] {
            let p = ParamPath::ShapeParam {
                surface: id,
                parameter,
            };
            p.set(&mut s, 0.125).unwrap();
            assert_eq!(p.get(&s).unwrap(), 0.125);
        }
    }
}
