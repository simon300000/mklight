use super::evaluator::ResidentEvaluator;
use crate::merit::{FieldSelector, Operand, OperandError, OperandValue, WavelengthSelector};
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
impl ResidentEvaluator {
    pub(super) fn illumination(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let mut values = Vec::new();
        for system in systems {
            let tr = self
                .reference
                .trace
                .tracer(system)
                .map_err(|e| MkError::Unsupported(format!("RI candidate: {e:?}")))?;
            let mut row = Vec::new();
            for op in ops {
                let field = match op.field {
                    FieldSelector::Index { index } => tr.fields().get(index).map(|f| f.point),
                    FieldSelector::Normalized { hx, hy } => {
                        Some(mklight_trace::FieldPoint::normalized(hx, hy))
                    }
                    FieldSelector::All => None,
                };
                let Some(field) = field else {
                    row.push(Err(OperandError::Invalid("RI needs one field".into())));
                    continue;
                };
                if field.vignetting != Default::default() {
                    return Err(MkError::Unsupported(
                        "RI coordinate adapter does not carry field vignetting".into(),
                    ));
                }
                let wls: Vec<_> = match op.wavelength {
                    WavelengthSelector::Primary => vec![tr.system().primary_wavelength],
                    WavelengthSelector::Index { index } => vec![index],
                    WavelengthSelector::All => (0..tr.num_wavelengths())
                        .filter(|&w| tr.system().wavelength_weights[w] > 0.)
                        .collect(),
                };
                let mut sum = 0.;
                let mut total = 0.;
                for &wl in &wls {
                    let (rows, axis, _, info) = mklight_analysis::gpu::illumination_rows(
                        self.context.backend.as_ref(),
                        &tr,
                        &[[field.hx, field.hy]],
                        wl,
                        self.reference.trace.settings().illumination_rings,
                        self.reference.trace.settings().illumination_arms,
                        ctx,
                    )?;
                    self.context.observe_run(&info);
                    let weight = if wls.len() == 1 {
                        1.
                    } else {
                        tr.system().wavelength_weights[wl]
                    };
                    sum += weight * rows[0][0] / axis;
                    total += weight;
                }
                row.push(Ok(sum / total));
            }
            values.push(row);
        }
        Ok(values)
    }
}
