use super::*;
use mklight_gpu::collectives::{self as col, SegmentKind, SegmentReduce};
pub(crate) fn reduce(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    values: PlanView,
    mask: Option<PlanView>,
    weights: Option<PlanView>,
    kind: SegmentKind,
) -> MkResult<BufferId> {
    let n = values.desc.shape[1];
    let offsets = b.fill_buffer(BufferDesc::new(ElementType::U32, [2]), &mut |mut a| {
        a.as_mut_slice::<u32>()?.copy_from_slice(&[0, n as u32]);
        Ok(())
    })?;
    let offsets = p.import(offsets.view());
    let output = p.scratch(BufferDesc::new(
        ElementType::DoubleFloat,
        if kind == SegmentKind::Moments2 {
            vec![1, 10]
        } else {
            vec![values.desc.shape[0], 1, 4]
        },
    ));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
    p.operations
        .push(ResidentOp::Collectives(col::Request::SegmentReduce(
            SegmentReduce {
                values,
                offsets: p.view(offsets)?,
                mask,
                mask_value: 1,
                weights,
                max_segment_len: n,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind,
                output,
                status,
            },
        )));
    Ok(output)
}
