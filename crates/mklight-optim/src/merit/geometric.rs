//! [`ModelEvaluator`]: operands computable from the model alone, with no ray
//! tracing — total track, centre/edge thicknesses, glass/air thickness
//! boundaries and raw parameter values.
//!
//! **Gaps.** Manufacturable thicknesses are measured between consecutive
//! *optically active* surfaces ([`optical_gaps`]). Dummy surfaces, coordinate
//! breaks and planes that "refract" into the medium they are already in (an
//! aperture stop or reference plane in air) are not physical boundaries, so a
//! gap runs across them: the air between two lenses is one gap even when the
//! stop sits inside it, and a stop placed at a lens vertex adds no zero-width
//! "air gap".
//!
//! **Physical sign.** Thicknesses are reported as physical gaps along the
//! propagation direction: the signed model thickness (CONVENTIONS §2) times
//! `(−1)^(number of mirrors before the gap)`, so a real gap is positive
//! after a fold mirror too.
//!
//! Edge thickness at an explicit height `h` of the gap from surface `a` to
//! surface `b` is `E = s · (Σ t − z_a(h) + z_b(h))` from the reference sag
//! definition (`SurfaceShape::sag`). Without a height it is measured at the
//! rim of the larger clear aperture, each surface's sag taken at its **own**
//! semi-diameter: `E = s · (Σ t − z_a(h_a) + z_b(h_b))` — the smaller surface
//! continues as a flat out to the larger rim, as a lens is edged. (With only
//! one semi-diameter known, both sags are taken at it.) A steep surface is
//! thus never evaluated beyond its own aperture, where its sag may not exist.
//! Edge thickness needs coaxial surfaces (no decenters or tilts on `a..=b`);
//! otherwise the operand is `Invalid`.

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
pub fn media(s: &OpticalSystem) -> Vec<MaterialRef> {
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

/// Is surface `i` (not the object or image) an optically active boundary?
/// Inactive: dummies, coordinate breaks, and planes refracting into the same
/// medium as before (stops and reference planes).
pub fn is_optically_active(s: &OpticalSystem, i: usize, media: &[MaterialRef]) -> bool {
    let sf = &s.surfaces[i];
    match sf.interaction {
        Interaction::Dummy => false,
        Interaction::Reflect => !sf.is_coordinate_break(),
        Interaction::Refract => {
            !sf.is_coordinate_break()
                && !(matches!(sf.shape, SurfaceShape::Plane) && i > 0 && media[i - 1] == media[i])
        }
    }
}

/// One physical gap between consecutive optically active surfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct Gap {
    /// Surface index where the gap starts (an active surface).
    pub from: usize,
    /// Surface index where it ends (the next active surface, or the image).
    pub to: usize,
    pub medium: MaterialRef,
}

impl Gap {
    pub fn is_air(&self) -> bool {
        matches!(self.medium, MaterialRef::Air | MaterialRef::Vacuum)
    }
}

/// Every gap from the first active surface to the image, in order. The last
/// one ends at the image (the back focal distance, not a manufactured part).
pub fn optical_gaps(s: &OpticalSystem) -> Vec<Gap> {
    let n = s.surfaces.len();
    if n < 2 {
        return Vec::new();
    }
    let med = media(s);
    let mut ends: Vec<usize> = (1..n - 1)
        .filter(|&i| is_optically_active(s, i, &med))
        .collect();
    ends.push(n - 1);
    ends.windows(2)
        .map(|w| Gap {
            from: w[0],
            to: w[1],
            medium: med[w[0]].clone(),
        })
        .collect()
}

fn fixed_sd(sd: SemiDiameter) -> Option<f64> {
    match sd {
        SemiDiameter::Fixed { value } => Some(value),
        SemiDiameter::Auto => None,
    }
}

/// `+1` / `−1`: propagation along `+z` / `−z` after surface `i`.
fn direction_after(s: &OpticalSystem, i: usize) -> f64 {
    let mirrors = s.surfaces[..=i]
        .iter()
        .filter(|sf| sf.interaction == Interaction::Reflect && !sf.is_coordinate_break())
        .count();
    if mirrors % 2 == 0 { 1.0 } else { -1.0 }
}

/// Physical thickness of the gap from surface index `a` to `b > a` (default
/// `b = a + 1`), at the centre or the edge.
pub fn gap_thickness(
    s: &OpticalSystem,
    a: usize,
    b: Option<usize>,
    measure: ThicknessMeasure,
    height: Option<f64>,
) -> OperandValue {
    let n = s.surfaces.len();
    let b = b.unwrap_or(a + 1);
    if b <= a || b >= n {
        return Err(OperandError::Invalid(
            "gap must end at a later surface (at most the image)".into(),
        ));
    }
    let mut sum = mklight_core::math::KahanSum::default();
    for sf in &s.surfaces[a..b] {
        if !sf.thickness.is_finite() {
            return Err(OperandError::Invalid("infinite thickness".into()));
        }
        sum.add(sf.thickness);
    }
    let dir = direction_after(s, a);
    let t = sum.value();
    if measure == ThicknessMeasure::Centre {
        return Ok(dir * t);
    }
    let coaxial = s.surfaces[a..=b]
        .iter()
        .all(|sf| sf.placement.is_none_or(|p| p.is_identity()));
    let (sa, sb) = (&s.surfaces[a], &s.surfaces[b]);
    if !coaxial || sa.is_coordinate_break() || sb.is_coordinate_break() {
        return Err(OperandError::Invalid(
            "edge thickness needs coaxial surfaces".into(),
        ));
    }
    // Height at which each surface's sag is taken.
    let (ha, hb) = match height {
        Some(h) => (h, h),
        None => match (fixed_sd(sa.semi_diameter), fixed_sd(sb.semi_diameter)) {
            // The rim of the larger aperture; the smaller surface continues
            // as a flat beyond its own rim.
            (Some(x), Some(y)) => (x, y),
            (Some(x), None) | (None, Some(x)) => (x, x),
            (None, None) => {
                return Err(OperandError::Invalid(
                    "edge thickness needs a fixed semi-diameter or explicit height".into(),
                ));
            }
        },
    };
    let sag = |shape: &SurfaceShape, h: f64| {
        shape
            .sag(h * h)
            .ok_or_else(|| OperandError::Invalid("height outside the surface's domain".into()))
    };
    Ok(dir * (t - sag(&sa.shape, ha)? + sag(&sb.shape, hb)?))
}

/// Value of one model-only operand (`Unsupported` for any other kind).
pub fn model_value(s: &OpticalSystem, op: &Operand) -> OperandValue {
    use OperandKind::*;
    if super::surface::is_kind(&op.kind) {
        return super::surface::value(s, &op.kind);
    }
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
        CentreThickness { surface, to } => {
            let b = to.map(|id| index(s, id)).transpose()?;
            gap_thickness(s, index(s, *surface)?, b, ThicknessMeasure::Centre, None)
        }
        EdgeThickness {
            surface,
            to,
            height,
        } => {
            let b = to.map(|id| index(s, id)).transpose()?;
            gap_thickness(s, index(s, *surface)?, b, ThicknessMeasure::Edge, *height)
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
            // Gaps starting in [from, to]; without `to`, every gap except the
            // one ending at the image (back focus).
            let i0 = from.map_or(Ok(0), |id| index(s, id))?;
            let i1 = to.map(|id| index(s, id)).transpose()?;
            let is_min = matches!(op.kind, MinThickness { .. });
            let mut acc: Option<f64> = None;
            for g in optical_gaps(s) {
                let in_range = g.from >= i0
                    && match i1 {
                        Some(i1) => g.from <= i1,
                        None => g.to < n - 1,
                    };
                let take = match medium {
                    MediumFilter::Glass => !g.is_air(),
                    MediumFilter::Air => g.is_air(),
                    MediumFilter::Any => true,
                };
                if !in_range || !take {
                    continue;
                }
                let v = gap_thickness(s, g.from, Some(g.to), *measure, None)?;
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

/// Does [`model_value`] compute this kind?
pub fn is_model_kind(kind: &OperandKind) -> bool {
    use OperandKind::*;
    super::surface::is_kind(kind)
        || matches!(
            kind,
            TotalTrack { .. }
                | CentreThickness { .. }
                | EdgeThickness { .. }
                | MinThickness { .. }
                | MaxThickness { .. }
                | ParamValue { .. }
        )
}

/// Does this model operand measure an edge thickness at a height taken from
/// the semi-diameters (so automatic semi-diameters need the tracer)?
pub fn needs_semi_diameters(kind: &OperandKind) -> bool {
    if super::surface::is_kind(kind) {
        return true;
    }
    match kind {
        OperandKind::EdgeThickness { height: None, .. } => true,
        OperandKind::MinThickness { measure, .. } | OperandKind::MaxThickness { measure, .. } => {
            *measure == ThicknessMeasure::Edge
        }
        _ => false,
    }
}

impl OperandEvaluator for ModelEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        is_model_kind(kind)
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
                Ok(operands.iter().map(|op| model_value(s, op)).collect())
            })
            .collect()
    }
}
