use super::*;
use mklight_gpu::wave_ops::image_integrals::ImageIntegral;
#[derive(Clone, Debug)]
pub struct PsfMetrics {
    pub energy_in_grid: f64,
    pub centroid_mm: [f64; 2],
    pub peak_index: usize,
}
impl ResidentPsf {
    pub fn metrics(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<PsfMetrics>> {
        let mut p = ResidentPlan::new(snapshot);
        let input = p.import(self.image.view());
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                ImageIntegral::Metrics {
                    input: p.view(input)?,
                    output,
                },
            )));
        p.outputs = vec![output];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let a = b.read_view(r.outputs[0].view())?;
        let a = a.as_slice::<Df32>()?;
        let weight = a[0].to_f64();
        let value = PsfMetrics {
            energy_in_grid: weight * self.spacing_mm * self.spacing_mm,
            centroid_mm: std::array::from_fn(|j| {
                self.center_mm[j]
                    + (a[j + 1].to_f64() / weight - (self.size / 2) as f64) * self.spacing_mm
            }),
            peak_index: a[4].hi as usize,
        };
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    /// The matched direct DFT route used by Huygens MTF, including exact DC=1.
    pub fn otf(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        frequencies: &[[f64; 2]],
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
        let queries = query_buffer(b, frequencies, ctx)?;
        let mut p = ResidentPlan::new(snapshot);
        let input = p.import(self.image.view());
        let q = p.import(queries.view());
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [frequencies.len()]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                ImageIntegral::Query {
                    input: p.view(input)?,
                    queries: p.view(q)?,
                    kind: 0,
                    spacing: Df32::from_f64(self.spacing_mm),
                    center: [Df32::from_f64((self.size / 2) as f64); 2],
                    normalization: Df32::from_f64(
                        self.total_energy / (self.spacing_mm * self.spacing_mm),
                    ),
                    output,
                },
            )));
        p.outputs = vec![output];
        p.settings = serde_json::json!({"operation":"psf_otf","frequencies":frequencies});
        let r = b.submit_plan(p, ctx)?.wait()?;
        let value = b
            .read_view(r.outputs[0].view())?
            .as_slice::<[f32; 2]>()?
            .iter()
            .map(|v| v.map(f64::from))
            .collect();
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    /// Band-limited spectral ensquared-energy integration over squares of
    /// half-widths `radii` centred on `center_mm`; this is not pixel binning.
    pub fn ensquared(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        radii: &[f64],
        center_mm: [f64; 2],
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Vec<f64>>> {
        let queries: Vec<_> = radii.iter().map(|r| [*r, 0.]).collect();
        let queries = query_buffer(b, &queries, ctx)?;
        let mut p = ResidentPlan::new(snapshot);
        let image = p.import(self.image.view());
        let q = p.import(queries.view());
        let complex = p.scratch(BufferDesc::new(
            ElementType::Complex32,
            [self.size, self.size],
        ));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                ImageIntegral::Complex {
                    input: p.view(image)?,
                    squared: false,
                    output: complex,
                },
            )));
        let spectrum = p.scratch(BufferDesc::new(
            ElementType::Complex32,
            [self.size, self.size],
        ));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::Transform(
                tr::TransformPlan {
                    kind: tr::TransformKind::Fft,
                    input: p.view(complex)?,
                    output: spectrum,
                    dimensions: vec![self.size, self.size],
                    output_dimensions: vec![self.size, self.size],
                    sign: -1,
                    scale: 1.,
                    affine: [Df32::ZERO; 6],
                },
            )));
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [radii.len()]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                ImageIntegral::Query {
                    input: p.view(spectrum)?,
                    queries: p.view(q)?,
                    kind: 2,
                    spacing: Df32::from_f64(self.spacing_mm),
                    center: std::array::from_fn(|j| {
                        Df32::from_f64(
                            (center_mm[j] - self.center_mm[j]) / self.spacing_mm
                                + (self.size / 2) as f64,
                        )
                    }),
                    normalization: Df32::from_f64(
                        (self.size * self.size) as f64 * self.total_energy,
                    ),
                    output,
                },
            )));
        p.outputs = vec![output];
        p.settings = serde_json::json!({"operation":"ensquared_energy","radii_mm":radii,"center_mm":center_mm});
        let r = b.submit_plan(p, ctx)?.wait()?;
        let value = b
            .read_view(r.outputs[0].view())?
            .as_slice::<[f32; 2]>()?
            .iter()
            .map(|v| v[0] as f64)
            .collect();
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    pub fn line_spread(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<[Vec<f64>; 2]>> {
        let mut p = ResidentPlan::new(snapshot);
        let image = p.import(self.image.view());
        let output = p.scratch(BufferDesc::new(ElementType::F32, [2, self.size]));
        p.operations.push(ResidentOp::Wave(w::Request::LineSpread {
            image: p.view(image)?,
            spacing: (self.spacing_mm / self.total_energy) as f32,
            output,
        }));
        p.outputs = vec![output];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let a = b.read_view(r.outputs[0].view())?;
        let a = a.as_slice::<f32>()?;
        let value = std::array::from_fn(|i| {
            a[i * self.size..(i + 1) * self.size]
                .iter()
                .map(|v| *v as f64)
                .collect()
        });
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
fn query_buffer(
    b: &dyn ComputeBackend,
    values: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<BufferHandle> {
    b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [values.len(), 2]),
        &mut |mut a| {
            for (i, (dst, src)) in a
                .as_mut_slice::<Df32>()?
                .chunks_mut(2)
                .zip(values)
                .enumerate()
            {
                if i % 4096 == 0 {
                    ctx.check()?;
                }
                if src
                    .iter()
                    .any(|v| !v.is_finite() || !Df32::from_f64(*v).is_finite())
                {
                    return Err(MkError::InvalidParam("image integral query".into()));
                }
                dst.copy_from_slice(&src.map(Df32::from_f64));
            }
            Ok(())
        },
    )
}
