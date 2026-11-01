use super::*;
use mklight_gpu::collectives::{self as col, SegmentKind, SegmentReduce};
#[derive(Clone, Copy, Debug)]
pub struct BlurEstimate {
    pub extent: f64,
    pub rms: f64,
}
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
impl ResidentPupil {
    pub(crate) fn append_blur(
        &self,
        p: &mut ResidentPlan,
        step: usize,
    ) -> MkResult<(BufferId, BufferId, usize)> {
        let count = (self.n - 1).div_ceil(step).pow(2);
        let points = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, count]));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [count]));
        let pupil = self.import(p);
        let b = self.geometry.surface_to_normal();
        let det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
        let inverse = [
            [b[1][1] / det, -b[0][1] / det],
            [-b[1][0] / det, b[0][0] / det],
        ]
        .map(|v| v.map(Df32::from_f64));
        p.operations
            .push(ResidentOp::Wave(w::Request::Blur(w::sampling::Blur {
                pupil,
                step,
                scale: Df32::from_f64(-self.lambda_f() * self.n as f64),
                inverse,
                offset: self.image_offset_mm.map(Df32::from_f64),
                points,
                mask,
            })));
        Ok((points, mask, count))
    }
    pub fn blur(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<BlurEstimate>> {
        let mut p = ResidentPlan::new(snapshot);
        let (points, mask, n) = self.append_blur(&mut p, 1)?;
        let summary = reduce(
            b,
            &mut p,
            p_view(points, &BufferDesc::new(ElementType::DoubleFloat, [5, n])),
            Some(p_view(mask, &BufferDesc::new(ElementType::U32, [n]))),
            None,
            SegmentKind::Summary,
        )?;
        let mut weights = p.view(points)?;
        weights.offset = 4 * n;
        weights.desc.shape = vec![n];
        let moments = reduce(
            b,
            &mut p,
            p_view(points, &BufferDesc::new(ElementType::DoubleFloat, [2, n])),
            Some(p_view(mask, &BufferDesc::new(ElementType::U32, [n]))),
            Some(weights),
            SegmentKind::Moments2,
        )?;
        p.outputs = vec![summary, moments];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let a = b.read_view(r.outputs[0].view())?;
        let a = a.as_slice::<Df32>()?;
        let m = b.read_view(r.outputs[1].view())?;
        let m = m.as_slice::<Df32>()?;
        let value = BlurEstimate {
            extent: a[11].to_f64().max(a[15].to_f64()).max(0.),
            rms: if m[1].hi > 0. {
                ((m[5].to_f64() + m[7].to_f64()) / m[1].to_f64()).sqrt()
            } else {
                0.
            },
        };
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
