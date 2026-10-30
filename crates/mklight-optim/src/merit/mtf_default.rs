//! Polychromatic MTF default criteria and exact worst-field combinations.
use super::default::DefaultCriterion;
use super::operand::{CombineOp, MtfOrientation, OperandKind};
use super::{FieldSelector, Operand, WavelengthSelector};
use mklight_core::{MkError, MkResult};

pub(super) fn operands(
    criterion: &DefaultCriterion,
    field_points: &[(FieldSelector, f64)],
    wls: &[usize],
    configs: &[usize],
    weight: f64,
) -> MkResult<Vec<Operand>> {
    let DefaultCriterion::Mtf {
        frequencies,
        target,
        method,
        worst_field,
    } = criterion
    else {
        unreachable!()
    };
    let fw: f64 = field_points.iter().map(|(_, w)| w).sum();
    let mut ops = Vec::new();
    if frequencies.is_empty()
        || frequencies.iter().any(|f| !f.is_finite() || *f < 0.0)
        || !target.is_finite()
        || !(0.0..=1.0).contains(target)
    {
        return Err(MkError::InvalidParam(
            "default MTF needs frequencies >= 0 and a target in [0, 1]".into(),
        ));
    }
    for &c in configs {
        for &frequency in frequencies {
            for orientation in [MtfOrientation::Tangential, MtfOrientation::Sagittal] {
                let mut args = Vec::new();
                for &(field, field_weight) in field_points.iter().filter(|(_, w)| *w > 0.0) {
                    let mut op = Operand::new(OperandKind::Mtf {
                        frequency,
                        orientation,
                        method: *method,
                        defocus_mm: 0.0,
                        detector: None,
                        wavelengths: wls.to_vec(),
                    })
                    .greater_than(*target)
                    .in_config(c);
                    op.field = field;
                    op.wavelength = WavelengthSelector::All;
                    op.weight = if *worst_field {
                        0.0
                    } else {
                        weight * field_weight
                            / (fw * configs.len() as f64 * frequencies.len() as f64 * 2.0)
                    };
                    args.push(ops.len());
                    ops.push(op);
                }
                if *worst_field {
                    ops.push(
                        Operand::new(OperandKind::Combine {
                            op: CombineOp::Min,
                            args,
                        })
                        .greater_than(*target)
                        .weight(weight / (configs.len() as f64 * frequencies.len() as f64 * 2.0)),
                    );
                }
            }
        }
    }
    Ok(ops)
}
