use super::traced::accumulate;
use super::*;
use mklight_gpu::{
    collectives::{self as col, SegmentKind, SegmentReduce, StableCompact, StableSort},
    wave_ops::{image_integrals::ImageIntegral, shells::Shells},
};
/// Reusable stable Fourier-radius topology. No optical values cross the host.
pub struct RadialBins {
    size: usize,
    count: usize,
    max_segment: usize,
    order: BufferHandle,
    keys: BufferHandle,
    offsets: BufferHandle,
}
impl RadialBins {
    pub fn new(
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        size: usize,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Self>> {
        let n = size * size;
        let mut p = ResidentPlan::new(snapshot);
        let keys = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let ids = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Shells(Shells::Keys {
                size,
                keys,
                ids,
            })));
        let sorted = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let order = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::StableSort(
                StableSort {
                    keys: p.view(keys)?,
                    ids: p.view(ids)?,
                    values: None,
                    output_keys: sorted,
                    output_ids: order,
                    output_values: None,
                },
            )));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let positions = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Shells(Shells::Boundaries {
                keys: p.view(sorted)?,
                mask,
                positions,
            })));
        let compact = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let starts = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let count = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::StableCompact(
                StableCompact {
                    values: p.view(sorted)?,
                    ids: p.view(positions)?,
                    mask: p.view(mask)?,
                    output: compact,
                    output_ids: starts,
                    count,
                },
            )));
        let offsets = p.scratch(BufferDesc::new(ElementType::U32, [n + 1]));
        let summary = p.scratch(BufferDesc::new(ElementType::U32, [2]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Shells(Shells::Offsets {
                positions: p.view(starts)?,
                count: p.view(count)?,
                output: offsets,
                summary,
            })));
        p.outputs = vec![order, compact, offsets, summary];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let summary = b.read_view(r.outputs[3].view())?;
        let summary = summary.as_slice::<u32>()?;
        Ok(WaveResult {
            value: Self {
                size,
                count: summary[0] as usize,
                max_segment: summary[1] as usize,
                order: r.outputs[0].clone(),
                keys: r.outputs[1].clone(),
                offsets: r.outputs[2].clone(),
            },
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
impl ResidentPsf {
    pub(super) fn resolve_center(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        center: crate::EnergyCenter,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<[f64; 2]> {
        match center {
            crate::EnergyCenter::Reference => Ok(self.center_mm),
            crate::EnergyCenter::Point { x, y } => Ok([x, y]),
            other => {
                let r = self.metrics(b, snapshot, ctx)?;
                accumulate(stats, &r.stats);
                Ok(if matches!(other, crate::EnergyCenter::Centroid) {
                    r.value.centroid_mm
                } else {
                    [
                        self.center_mm[0]
                            + ((r.value.peak_index % self.size) as f64 - (self.size / 2) as f64)
                                * self.spacing_mm,
                        self.center_mm[1]
                            + ((r.value.peak_index / self.size) as f64 - (self.size / 2) as f64)
                                * self.spacing_mm,
                    ]
                })
            }
        }
    }
}
/// Batch common-grid PSFs and diffraction references. Fourier bins with equal
/// integer radius are reduced once, and their Bessel kernels are shared.
pub fn encircled_energy_batch(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    psfs: &[(&ResidentPsf, crate::EnergyCenter)],
    radii: &[f64],
    ctx: &JobContext,
) -> MkResult<WaveResult<Vec<Vec<f64>>>> {
    let (first, _) = psfs
        .first()
        .ok_or_else(|| MkError::InvalidParam("energy needs PSFs".into()))?;
    let size = first.size;
    let spacing = first.spacing_mm;
    if psfs
        .iter()
        .any(|(p, _)| p.size != size || p.spacing_mm != spacing)
    {
        return Err(MkError::InvalidParam(
            "energy PSFs must share their grid".into(),
        ));
    }
    let geometry = RadialBins::new(b, snapshot.clone(), size, &ctx.subrange(0., 0.1))?;
    let mut stats = geometry.stats;
    let geometry = geometry.value;
    let queries = b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [radii.len()]),
        &mut |mut a| {
            for (v, r) in a.as_mut_slice::<Df32>()?.iter_mut().zip(radii) {
                if !r.is_finite() {
                    return Err(MkError::InvalidParam("energy radius".into()));
                }
                *v = Df32::from_f64(*r);
            }
            Ok(())
        },
    )?;
    let mut out = vec![];
    let mut info = None;
    let chunks = psfs.len().div_ceil(16);
    for (chunk, images) in psfs.chunks(16).enumerate() {
        let sub = ctx.subrange(
            0.1 + 0.9 * chunk as f64 / chunks as f64,
            0.1 + 0.9 * (chunk + 1) as f64 / chunks as f64,
        );
        let mut centres = vec![];
        for (psf, c) in images {
            centres.push(psf.resolve_center(
                b,
                snapshot.clone(),
                *c,
                &sub.subrange(0., 0.05),
                &mut stats,
            )?);
        }
        let mut p = ResidentPlan::new(snapshot.clone());
        let order = p.import(geometry.order.view());
        let mut offsets = geometry.offsets.view();
        offsets.desc.shape = vec![geometry.count + 1];
        let offsets = p.import(offsets);
        let mut keys = geometry.keys.view();
        keys.desc.shape = vec![geometry.count];
        let keys = p.import(keys);
        let radii = p.import(queries.view());
        let mut values = vec![];
        for ((psf, _), centre) in images.iter().zip(centres) {
            let input = p.import(psf.image.view());
            let complex = p.scratch(BufferDesc::new(ElementType::Complex32, [size, size]));
            p.operations
                .push(ResidentOp::Wave(w::Request::ImageIntegral(
                    ImageIntegral::Complex {
                        input: p.view(input)?,
                        squared: false,
                        output: complex,
                    },
                )));
            let fft = p.scratch(BufferDesc::new(ElementType::Complex32, [size, size]));
            p.operations
                .push(ResidentOp::Transforms(tr::Request::Transform(
                    tr::TransformPlan {
                        kind: tr::TransformKind::Fft,
                        input: p.view(complex)?,
                        output: fft,
                        dimensions: vec![size, size],
                        output_dimensions: vec![size, size],
                        sign: -1,
                        scale: 1.,
                        affine: [Df32::ZERO; 6],
                    },
                )));
            let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, size * size]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Shells(Shells::Gather {
                    spectrum: p.view(fft)?,
                    order: p.view(order)?,
                    center: std::array::from_fn(|j| {
                        Df32::from_f64((centre[j] - psf.center_mm[j]) / spacing + (size / 2) as f64)
                    }),
                    normalization: Df32::from_f64((size * size) as f64 * psf.total_energy),
                    output: terms,
                })));
            let shells = p.scratch(BufferDesc::new(
                ElementType::DoubleFloat,
                [1, geometry.count],
            ));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [geometry.count]));
            p.operations
                .push(ResidentOp::Collectives(col::Request::SegmentReduce(
                    SegmentReduce {
                        values: p.view(terms)?,
                        offsets: p.view(offsets)?,
                        mask: None,
                        mask_value: 0,
                        weights: None,
                        max_segment_len: geometry.max_segment,
                        reference: None,
                        mode: ReduceMode::DoubleFloat,
                        kind: SegmentKind::Sum,
                        output: shells,
                        status,
                    },
                )));
            let mut v = p.view(shells)?;
            v.desc.shape = vec![geometry.count];
            values.push(v);
        }
        let output = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [images.len(), queries.desc().len()?],
        ));
        p.operations
            .push(ResidentOp::Wave(w::Request::Shells(Shells::Query {
                values,
                keys: p.view(keys)?,
                radii: p.view(radii)?,
                frequency_step: Df32::from_f64(1. / (geometry.size as f64 * spacing)),
                output,
            })));
        p.outputs = vec![output];
        p.settings = serde_json::json!({"p6":"radial_energy_batch","grid":size,"radial_bins":geometry.count,"max_bin_length":geometry.max_segment});
        let r = b.submit_plan(p, &sub.subrange(0.05, 1.))?.wait()?;
        accumulate(&mut stats, &r.stats);
        let a = b.read_view(r.outputs[0].view())?;
        let values = a.as_slice::<Df32>()?;
        let width = queries.desc().len()?;
        for i in 0..images.len() {
            out.push(
                values[i * width..(i + 1) * width]
                    .iter()
                    .map(|v| v.to_f64())
                    .collect(),
            );
        }
        info = Some(r.run_info);
    }
    Ok(WaveResult {
        value: out,
        run_info: info.unwrap(),
        stats,
    })
}
