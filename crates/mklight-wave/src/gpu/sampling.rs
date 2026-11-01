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
/// Existing phase-slope contrast rule, with slopes and characteristic functions
/// evaluated on device. The short frequency search remains host control.
pub fn useful_frequency(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    pupils: &[(&ResidentPupil, f64)],
    ctx: &JobContext,
) -> MkResult<WaveResult<f64>> {
    use super::traced::accumulate;
    let cutoff = pupils
        .iter()
        .map(|(p, _)| 1. / p.lambda_f())
        .fold(0f64, f64::max);
    let ceiling = 0.8 * cutoff;
    let mut candidates = vec![40f64.min(0.1 * cutoff)];
    while *candidates.last().unwrap() < ceiling {
        candidates.push(candidates.last().unwrap() * 1.25);
    }
    let fs: Vec<_> = candidates
        .iter()
        .flat_map(|&f| [[f, 0.], [0., f]])
        .collect();
    let freq = b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [fs.len(), 1, 2]),
        &mut |mut a| {
            for (row, f) in a.as_mut_slice::<Df32>()?.chunks_mut(2).zip(&fs) {
                row.copy_from_slice(&f.map(Df32::from_f64));
            }
            Ok(())
        },
    )?;
    let mut stats = PlanStats::default();
    let mut curves = vec![];
    let mut info = None;
    let wsum = pupils.iter().map(|(_, w)| w).sum::<f64>();
    for (index, (pupil, weight)) in pupils.iter().enumerate() {
        let mut p = ResidentPlan::new(snapshot.clone());
        let (points, mask, n) = pupil.append_blur(&mut p, (pupil.n / 64).max(1))?;
        let summary = reduce(
            b,
            &mut p,
            p_view(points, &BufferDesc::new(ElementType::DoubleFloat, [5, n])),
            Some(p_view(mask, &BufferDesc::new(ElementType::U32, [n]))),
            None,
            SegmentKind::Summary,
        )?;
        let f = p.import(freq.view());
        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, fs.len() * n]));
        p.operations.push(ResidentOp::Geometry(
            mklight_gpu::geometry::Request::Characteristic(mklight_gpu::geometry::Characteristic {
                points: p.view(points)?,
                mask: p.view(mask)?,
                segment_len: n,
                frequency: p.view(f)?,
                output: terms,
            }),
        ));
        let offsets = b.fill_buffer(
            BufferDesc::new(ElementType::U32, [fs.len() + 1]),
            &mut |mut a| {
                for (i, v) in a.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                    *v = (i * n) as u32;
                }
                Ok(())
            },
        )?;
        let offsets = p.import(offsets.view());
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, fs.len()]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [fs.len()]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::SegmentReduce(
                SegmentReduce {
                    values: p.view(terms)?,
                    offsets: p.view(offsets)?,
                    mask: None,
                    mask_value: 0,
                    weights: None,
                    max_segment_len: n,
                    reference: None,
                    mode: ReduceMode::DoubleFloat,
                    kind: SegmentKind::Sum,
                    output,
                    status,
                },
            )));
        p.outputs = vec![summary, output];
        let r = b
            .submit_plan(
                p,
                &ctx.subrange(
                    index as f64 / pupils.len() as f64,
                    (index + 1) as f64 / pupils.len() as f64,
                ),
            )?
            .wait()?;
        accumulate(&mut stats, &r.stats);
        let sw = b.read_view(r.outputs[0].view())?.as_slice::<Df32>()?[16].to_f64();
        if sw > 0. {
            let a = b.read_view(r.outputs[1].view())?;
            let a = a.as_slice::<Df32>()?;
            curves.push((
                (0..fs.len())
                    .map(|j| [a[j].to_f64(), a[fs.len() + j].to_f64()])
                    .collect::<Vec<_>>(),
                weight / (wsum * sw),
                pupil.lambda_f(),
            ));
        }
        info = Some(r.run_info);
    }
    let mut selected = *candidates.last().unwrap();
    for (i, &f) in candidates.iter().enumerate() {
        let contrast = |axis: usize| {
            let mut z = [0f64; 2];
            for (curve, w, lf) in &curves {
                let dl = w * crate::mtf::diffraction_limited_mtf(f * lf, 0.);
                z[0] += dl * curve[2 * i + axis][0];
                z[1] += dl * curve[2 * i + axis][1];
            }
            z[0].hypot(z[1])
        };
        if f >= ceiling || (contrast(0) <= 0.1 && contrast(1) <= 0.1) {
            selected = f;
            break;
        }
    }
    let step = 10f64.powf(selected.log10().floor() - 1.);
    let value = ((selected / step).ceil() * step).min(ceiling);
    Ok(WaveResult {
        value,
        run_info: info
            .ok_or_else(|| MkError::InvalidParam("frequency estimate needs pupils".into()))?,
        stats,
    })
}
