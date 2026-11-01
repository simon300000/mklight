use super::evaluator::ResidentEvaluator;
use crate::merit::{Operand, OperandError, OperandKind, OperandValue, WavelengthSelector};
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use mklight_gpu::{
    Df32,
    systems::{Candidate, UploadedSystem},
};
impl ResidentEvaluator {
    pub(super) fn first_order(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        template: Option<(&UploadedSystem, &[Candidate])>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let Some((template, rows)) = template else {
            return Err(MkError::Unsupported(
                "batched first order requires a parametric system template".into(),
            ));
        };
        let mut p = self.context.plan();
        let prep = template.append(self.context.backend.as_ref(), &mut p, rows, ctx)?;
        let shape = p.desc(prep.first_order)?.shape.clone();
        p.outputs = vec![prep.first_order, prep.map.status];
        let r = self.context.submit(p, ctx)?;
        let fo = self.context.backend.read_view(r.outputs[0].view())?;
        let fo = fo.as_slice::<Df32>()?;
        let status = self.context.backend.read_view(r.outputs[1].view())?;
        let status = status.as_slice::<u32>()?;
        Ok(systems
            .iter()
            .enumerate()
            .map(|(i, system)| {
                ops.iter()
                    .map(|op| {
                        if status[2 * i] != 0 {
                            return Err(OperandError::Invalid(
                                "first-order candidate status".into(),
                            ));
                        }
                        let (axis, column) = match op.kind {
                            OperandKind::Efl | OperandKind::EflY => (1, 0),
                            OperandKind::Bfl | OperandKind::BflY => (1, 1),
                            OperandKind::EflX => (0, 0),
                            OperandKind::BflX => (0, 1),
                            _ => unreachable!(),
                        };
                        let wls: Vec<_> = match op.wavelength {
                            WavelengthSelector::Primary => vec![system.wavelengths.primary],
                            WavelengthSelector::Index { index } => vec![index],
                            WavelengthSelector::All => (0..system.wavelengths.wavelengths.len())
                                .filter(|&w| system.wavelengths.wavelengths[w].weight > 0.)
                                .collect(),
                        };
                        let mut sum = 0.;
                        let mut weight = 0.;
                        for w in wls {
                            let ww = if matches!(op.wavelength, WavelengthSelector::All) {
                                system.wavelengths.wavelengths[w].weight
                            } else {
                                1.
                            };
                            sum += ww
                                * fo[((i * shape[1] + w) * 2 + axis) * shape[3] + column].to_f64();
                            weight += ww;
                        }
                        Ok(sum / weight)
                    })
                    .collect()
            })
            .collect())
    }
}
