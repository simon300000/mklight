//! Polychromatic MTF default criterion and its worst-field combination.
use super::operand::{CombineOp, MtfMethod, MtfOrientation, OperandKind};
use super::{FieldSelector, Operand, WavelengthSelector};
use mklight_core::{MkError, MkResult};
use serde::{Deserialize, Serialize};

/// Polychromatic tangential and sagittal lower bounds at fixed frequencies
/// (cycles/mm) over the default merit's fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfCriterion {
    pub frequencies: Vec<f64>,
    pub target: f64,
    #[serde(default)]
    pub method: MtfMethod,
    /// One minimum over every field and both directions per frequency,
    /// instead of the weighted per-field bounds.
    #[serde(default)]
    pub worst_field: bool,
}

impl MtfCriterion {
    /// With `worst_field`, the per-field values get zero weight and one
    /// `combine:min` over every field and both directions carries the
    /// residual of each frequency and configuration.
    pub(super) fn operands(
        &self,
        field_points: &[(FieldSelector, f64)],
        wls: &[usize],
        configs: &[usize],
        weight: f64,
    ) -> MkResult<Vec<Operand>> {
        let &MtfCriterion {
            ref frequencies,
            target,
            method,
            worst_field,
        } = self;
        if frequencies.is_empty() || frequencies.iter().any(|f| !(*f >= 0.0)) {
            return Err(MkError::InvalidParam(
                "default MTF needs frequencies >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&target) {
            return Err(MkError::InvalidParam(
                "default MTF target must be in [0, 1]".into(),
            ));
        }
        let fields: Vec<_> = field_points.iter().filter(|(_, w)| *w > 0.0).collect();
        let fw: f64 = fields.iter().map(|(_, w)| w).sum();
        let share = weight / (configs.len() * frequencies.len()) as f64;
        let mut ops = Vec::new();
        for &c in configs {
            for &frequency in frequencies {
                let first = ops.len();
                for orientation in [MtfOrientation::Tangential, MtfOrientation::Sagittal] {
                    for &&(field, field_weight) in &fields {
                        let mut op = Operand::new(OperandKind::Mtf {
                            frequency,
                            orientation,
                            method,
                            defocus_mm: 0.0,
                            detector: None,
                            wavelengths: wls.to_vec(),
                        })
                        .greater_than(target)
                        .in_config(c)
                        .weight(if worst_field {
                            0.0
                        } else {
                            share * field_weight / (2.0 * fw)
                        });
                        op.field = field;
                        op.wavelength = WavelengthSelector::All;
                        ops.push(op);
                    }
                }
                if worst_field {
                    let args = (first..ops.len()).collect();
                    ops.push(
                        Operand::new(OperandKind::Combine {
                            op: CombineOp::Min,
                            args,
                        })
                        .greater_than(target)
                        .weight(share),
                    );
                }
            }
        }
        Ok(ops)
    }
}
