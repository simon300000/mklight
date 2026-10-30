//! [`ModelEvaluator`]: operands computable from the model alone, with no ray
//! tracing — total track, centre/edge thicknesses, glass/air thickness
//! boundaries and raw parameter values.
//!
//! Edge thickness at height `h` after surface `i` is
//! `E = tᵢ − zᵢ(h) + zᵢ₊₁(h)` from the reference sag definition
//! (`SurfaceShape::sag`). It is only defined for coaxial surfaces (no
//! placement, no coordinate breaks); otherwise the operand is `Invalid`.

use mklight_core::model::{Interaction, MaterialRef, SemiDiameter, SurfaceShape};
use mklight_core::{JobContext, MkResult, OpticalSystem, SurfaceId};

use super::operand::{MediumFilter, OperandKind, ThicknessMeasure};
use super::{Operand, OperandError, OperandEvaluator, OperandValue};

/// Evaluator for model-only operands.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelEvaluator;

fn index(s: &OpticalSystem, id: SurfaceId) -> Result<usize, OperandError> {
    s.surface_index(id)
        .ok_or_else(|| OperandError::Invalid(format!("no surface with id {}", id.0)))
}

/// Medium after each surface (mirrors, dummies and coordinate breaks keep
/// the medium before them).
fn media(s: &OpticalSystem) -> Vec<MaterialRef> {
    let mut m = s.surfaces[0].material.clone();
    s.surfaces
        .iter()
        .enumerate()
        .map(|(i, sf)| {
            if i == 0 || (sf.interaction == Interaction::Refract && !sf.is_coordinate_break()) {
                m = sf.material.clone();
            }
            m.clone()
        })
        .collect()
}

fn fixed_sd(sd: SemiDiameter) -> Option<f64> {
    match sd {
        SemiDiameter::Fixed { value } => Some(value),
        SemiDiameter::Auto => None,
    }
}

/// Thickness after surface index `i`, at the centre or the edge.
pub fn gap_thickness(
    s: &OpticalSystem,
    i: usize,
    measure: ThicknessMeasure,
    height: Option<f64>,
) -> OperandValue {
    let n = s.surfaces.len();
    if i + 1 >= n {
        return Err(OperandError::Invalid("no gap after the image".into()));
    }
    let a = &s.surfaces[i];
    let t = a.thickness;
    if !t.is_finite() {
        return Err(OperandError::Invalid("infinite thickness".into()));
    }
    if measure == ThicknessMeasure::Centre {
        return Ok(t);
    }
    let b = &s.surfaces[i + 1];
    let coaxial = |sf: &mklight_core::model::Surface| {
        !sf.is_coordinate_break() && sf.placement.is_none_or(|p| p.is_identity())
    };
    if !coaxial(a) || !coaxial(b) {
        return Err(OperandError::Invalid(
            "edge thickness needs coaxial surfaces".into(),
        ));
    }
    let h = match height {
        Some(h) => h,
        None => match (fixed_sd(a.semi_diameter), fixed_sd(b.semi_diameter)) {
            (Some(x), Some(y)) => x.max(y),
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => {
                return Err(OperandError::Invalid(
                    "edge thickness needs a fixed semi-diameter or explicit height".into(),
                ));
            }
        },
    };
    let sag = |shape: &SurfaceShape| {
        shape
            .sag(h * h)
            .ok_or_else(|| OperandError::Invalid("height outside the surface's domain".into()))
    };
    Ok(t - sag(&a.shape)? + sag(&b.shape)?)
}

fn eval_one(s: &OpticalSystem, op: &Operand) -> OperandValue {
    use OperandKind::*;
    let n = s.surfaces.len();
    match &op.kind {
        TotalTrack { from, to } => {
            let i0 = from.map_or(Ok(1), |id| index(s, id))?;
            let i1 = to.map_or(Ok(n - 1), |id| index(s, id))?;
            if i1 < i0 {
                return Err(OperandError::Invalid("total track range reversed".into()));
            }
            let mut sum = mklight_core::math::KahanSum::default();
            for sf in &s.surfaces[i0..i1] {
                if !sf.thickness.is_finite() {
                    return Err(OperandError::Invalid("infinite thickness in range".into()));
                }
                sum.add(sf.thickness);
            }
            Ok(sum.value())
        }
        CentreThickness { surface } => {
            gap_thickness(s, index(s, *surface)?, ThicknessMeasure::Centre, None)
        }
        EdgeThickness { surface, height } => {
            gap_thickness(s, index(s, *surface)?, ThicknessMeasure::Edge, *height)
        }
        MinThickness {
            measure,
            medium,
            from,
            to,
        }
        | MaxThickness {
            measure,
            medium,
            from,
            to,
        } => {
            let i0 = from.map_or(Ok(1), |id| index(s, id))?;
            let i1 = to.map_or(Ok(n.saturating_sub(2)), |id| index(s, id))?;
            let med = media(s);
            let mut acc: Option<f64> = None;
            let is_min = matches!(op.kind, MinThickness { .. });
            for (i, m) in med.iter().enumerate().take(i1 + 1).skip(i0) {
                if s.surfaces[i].is_coordinate_break() {
                    continue;
                }
                let air = matches!(m, MaterialRef::Air | MaterialRef::Vacuum);
                let take = match medium {
                    MediumFilter::Glass => !air,
                    MediumFilter::Air => air,
                    MediumFilter::Any => true,
                };
                if !take {
                    continue;
                }
                let v = gap_thickness(s, i, *measure, None)?;
                acc = Some(match acc {
                    None => v,
                    Some(a) if is_min => a.min(v),
                    Some(a) => a.max(v),
                });
            }
            acc.ok_or_else(|| OperandError::Invalid("no gap matches the selection".into()))
        }
        ParamValue { param } => param
            .get(s)
            .map_err(|e| OperandError::Invalid(e.to_string())),
        _ => Err(OperandError::Unsupported),
    }
}

impl OperandEvaluator for ModelEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        use OperandKind::*;
        matches!(
            kind,
            TotalTrack { .. }
                | CentreThickness { .. }
                | EdgeThickness { .. }
                | MinThickness { .. }
                | MaxThickness { .. }
                | ParamValue { .. }
        )
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        systems
            .iter()
            .map(|s| {
                ctx.check()?;
                Ok(operands.iter().map(|op| eval_one(s, op)).collect())
            })
            .collect()
    }
}
