//! Trial summaries: centred/scaled moments and stable criterion order statistics.
use super::*;
use crate::tolerance::{Statistics, TrialOutcome};
use mklight_core::MkError;
use mklight_gpu::collectives as c;
impl GpuContext {
    pub(crate) fn tolerance_statistics(
        &self,
        ok: &[&TrialOutcome],
        percentiles: &[f64],
        ctx: &JobContext,
    ) -> MkResult<Statistics> {
        let n = ok.len();
        let channels = 1 + ok[0].compensators.len();
        let value = |row: usize, col: usize| {
            if col == 0 {
                Some(ok[row].criterion)
            } else {
                ok[row].compensators.get(col - 1).copied()
            }
        };
        let mut origin = Vec::with_capacity(channels);
        let mut scale = Vec::with_capacity(channels);
        let normal = |v: f64| {
            v == 0.
                || (v.is_finite()
                    && v.abs() >= f32::MIN_POSITIVE as f64
                    && v.abs() <= f32::MAX as f64)
        };
        for col in 0..channels {
            ctx.check()?;
            let base = value(0, col).unwrap();
            let mut size = 0_f64;
            for row in 0..n {
                if row % 4096 == 0 {
                    ctx.check()?;
                }
                if let Some(v) = value(row, col) {
                    if !normal(v) {
                        return Err(MkError::Unsupported(
                            "tolerance statistics numeric range".into(),
                        ));
                    }
                    size = size.max((v - base).abs());
                }
            }
            origin.push(base);
            scale.push(if size == 0. { 1. } else { size });
        }
        let mut means = Vec::new();
        let mut deviations = Vec::new();
        let mut quantiles = Vec::new();
        let (mut min, mut max) = (0., 0.);
        // Independent channels can tile without changing a segment's fixed tree.
        let tile = (64 * 1024 * 1024 / (16 * n)).clamp(1, 32);
        for first in (0..channels).step_by(tile) {
            ctx.check()?;
            let width = (channels - first).min(tile);
            let count = width * n;
            let mut p = self.plan();
            let points = self.backend.fill_buffer(
                BufferDesc::new(ElementType::DoubleFloat, [2, count]),
                &mut |mut f| {
                    let out = f.as_mut_slice::<Df32>()?;
                    for (i, v) in out.iter_mut().enumerate() {
                        if i % 4096 == 0 {
                            ctx.check()?;
                        }
                        *v = if i < count {
                            let col = first + i / n;
                            Df32::from_f64(
                                value(i % n, col).map_or(0., |v| (v - origin[col]) / scale[col]),
                            )
                        } else {
                            Df32::ZERO
                        };
                    }
                    Ok(())
                },
            )?;
            let points = p.import(points.view());
            let mask = self.fill(&mut p, [count], |out: &mut [u32]| {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = u32::from(value(i % n, first + i / n).is_some());
                }
            })?;
            let offsets = self.fill(&mut p, [width + 1], |out: &mut [u32]| {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = (i * n) as u32;
                }
            })?;
            let moments = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [width, 10]));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [width]));
            p.operations
                .push(ResidentOp::Collectives(c::Request::SegmentReduce(
                    c::SegmentReduce {
                        values: p.view(points)?,
                        offsets,
                        mask: Some(mask),
                        mask_value: 1,
                        weights: None,
                        max_segment_len: n,
                        reference: None,
                        mode: mklight_gpu::ReduceMode::DoubleFloat,
                        kind: c::SegmentKind::Moments2,
                        output: moments,
                        status,
                    },
                )));
            p.outputs = vec![moments];
            if first == 0 {
                let largest = ok.iter().map(|o| o.criterion.abs()).fold(0., f64::max);
                let largest = if largest == 0. { 1. } else { largest };
                let keys = self.backend.fill_buffer(
                    BufferDesc::new(ElementType::DoubleFloat, [n]),
                    &mut |mut f| {
                        for (i, v) in f.as_mut_slice::<Df32>()?.iter_mut().enumerate() {
                            if i % 4096 == 0 {
                                ctx.check()?;
                            }
                            let key = ok[i].criterion / largest;
                            if (ok[i].criterion != 0. && key == 0.) || !normal(key) {
                                return Err(MkError::Unsupported(
                                    "tolerance percentile exponent range".into(),
                                ));
                            }
                            *v = Df32::from_f64(key);
                        }
                        Ok(())
                    },
                )?;
                let keys = p.import(keys.view());
                let ids = self.fill(&mut p, [n], |out: &mut [u32]| {
                    for (i, v) in out.iter_mut().enumerate() {
                        *v = i as u32;
                    }
                })?;
                let sorted = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
                let sorted_ids = p.scratch(BufferDesc::new(ElementType::U32, [n]));
                p.operations
                    .push(ResidentOp::Collectives(c::Request::StableSort(
                        c::StableSort {
                            keys: p.view(keys)?,
                            ids,
                            values: None,
                            output_keys: sorted,
                            output_ids: sorted_ids,
                            output_values: None,
                        },
                    )));
                p.outputs.push(sorted_ids);
            }
            let result = self.submit(p, ctx)?;
            let m = self.values(&result, 0)?;
            for (i, row) in m.as_chunks::<10>().0.iter().enumerate() {
                let col = first + i;
                means.push(origin[col] + scale[col] * row[2]);
                deviations.push((row[5] / (row[0] - 1.).max(1.)).sqrt() * scale[col]);
            }
            if first == 0 {
                let view = self.backend.read_view(result.outputs[1].view())?;
                let order = view.as_slice::<u32>()?;
                let at = |i: usize| ok[order[i] as usize].criterion;
                min = at(0);
                max = at(n - 1);
                quantiles = percentiles
                    .iter()
                    .map(|&q| {
                        let rank = q / 100. * (n - 1) as f64;
                        let i = rank.floor() as usize;
                        let a = at(i);
                        (
                            q,
                            if i + 1 < n {
                                a + (rank - i as f64) * (at(i + 1) - a)
                            } else {
                                a
                            },
                        )
                    })
                    .collect();
            }
            ctx.report(
                (first + width) as f64 / channels as f64,
                "tolerance statistics",
            );
        }
        Ok(Statistics {
            mean: means[0],
            std_dev: deviations[0],
            min,
            max,
            percentiles: quantiles,
            compensator_mean: means[1..].to_vec(),
            compensator_std: deviations[1..].to_vec(),
        })
    }
}
