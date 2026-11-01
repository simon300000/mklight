use super::*;
use mklight_gpu::wave_ops::{fft_mtf::FftMtf, image_integrals::ImageIntegral, spectral::Spectral};
impl ResidentPupil {
    pub(super) fn append_sensor_guard(&self, p: &mut ResidentPlan) -> MkResult<()> {
        // The complete square is safely below the horizon in this cone. Other
        // cases inspect actual coverage, so a clipped high-NA pupil still works.
        if std::f64::consts::SQRT_2 * self.geometry.sin_max() < self.geometry.chief_ray[2] {
            return Ok(());
        }
        let coverage = p.import(self.coverage.view());
        let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        let normal = self.geometry.chief_frame().map(|v| Df32::from_f64(v.z));
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Domain {
                coverage: p.view(coverage)?,
                sin_max: self.wave_geometry().sin_max,
                normal,
                status,
            })));
        Ok(())
    }

    pub fn without_aberrations(&self, b: &dyn ComputeBackend, ctx: &JobContext) -> MkResult<Self> {
        let mut p = self.clone();
        p.opd = b
            .fill_buffer(
                BufferDesc::new(ElementType::DoubleFloat, [self.n, self.n]),
                &mut |mut a| {
                    for block in a.as_mut_slice::<Df32>()?.chunks_mut(4096) {
                        ctx.check()?;
                        block.fill(Df32::ZERO);
                    }
                    Ok(())
                },
            )?
            .view();
        p.warnings
            .retain(|w| !matches!(w, WaveWarning::PupilUndersampled { .. }));
        p.input_hash = format!("{}:zero_opd", self.input_hash);
        Ok(p)
    }
    pub(super) fn fft_lattice(&self, p: &mut ResidentPlan) -> MkResult<BufferId> {
        self.append_sensor_guard(p)?;
        let m = crate::fft::next_fast_len(2 * self.n);
        let v = self.import(p);
        let (field, sums) = if self.geometry.is_on_axis() {
            self.prepare(p, v, m, [0., 0., 1.])?
        } else {
            let field = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
            let moments = p.scratch(BufferDesc::new(ElementType::F32, [3, self.n * self.n]));
            let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3]));
            p.operations
                .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sensor {
                    pupil: v,
                    sin_max: self.wave_geometry().sin_max,
                    frame: self
                        .geometry
                        .chief_frame()
                        .map(|v| [v.x, v.y, v.z].map(Df32::from_f64)),
                    field,
                    moments,
                })));
            p.operations.push(ResidentOp::Sum(Sum {
                input: p.view(moments)?,
                mode: ReduceMode::DoubleFloat,
                output: sums,
            }));
            (field, sums)
        };
        let fft = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::Transform(
                tr::TransformPlan {
                    kind: tr::TransformKind::Fft,
                    input: p.view(field)?,
                    output: fft,
                    dimensions: vec![m, m],
                    output_dimensions: vec![m, m],
                    sign: 1,
                    scale: 1.,
                    affine: [Df32::ZERO; 6],
                },
            )));
        let power = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                ImageIntegral::Complex {
                    input: p.view(fft)?,
                    squared: true,
                    output: power,
                },
            )));
        let raw = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::Transform(
                tr::TransformPlan {
                    kind: tr::TransformKind::Fft,
                    input: p.view(power)?,
                    output: raw,
                    dimensions: vec![m, m],
                    output_dimensions: vec![m, m],
                    sign: -1,
                    scale: 1.,
                    affine: [Df32::ZERO; 6],
                },
            )));
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Normalize {
                input: p.view(raw)?,
                sums: p.view(sums)?,
                output,
            })));
        Ok(output)
    }
    pub fn append_fft_otf(
        &self,
        p: &mut ResidentPlan,
        ideal: &ResidentPupil,
        frequencies: PlanView,
    ) -> MkResult<BufferId> {
        let data = self.fft_lattice(p)?;
        let reference = ideal.fft_lattice(p)?;
        let output = p.scratch(BufferDesc::new(
            ElementType::Complex32,
            [frequencies.desc.shape[0]],
        ));
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sample {
                data: p.view(data)?,
                reference: p.view(reference)?,
                frequencies,
                spacing: Df32::from_f64(1. / (self.n as f64 * self.lambda_f())),
                mapping: self
                    .geometry
                    .surface_to_normal_frequency()
                    .map(|v| v.map(Df32::from_f64)),
                output,
            })));
        Ok(output)
    }
    pub fn fft_otf(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        frequencies: &[[f64; 2]],
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
        let ideal = self.without_aberrations(b, ctx)?;
        let f = frequency_buffer(b, frequencies, ctx)?;
        let mut p = ResidentPlan::new(snapshot);
        let f = p.import(f.view());
        let value = self.append_fft_otf(
            &mut p,
            &ideal,
            p_view(
                f,
                &BufferDesc::new(ElementType::F32, [frequencies.len(), 2]),
            ),
        )?;
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [frequencies.len()]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Spectral(Spectral {
                inputs: vec![p.view(value)?],
                weights: vec![Df32::from_f32(1.)],
                frequencies: Some(p.view(f)?),
                offsets: vec![self.image_offset_mm.map(Df32::from_f64)],
                output,
            })));
        p.outputs = vec![output];
        p.settings = serde_json::json!({"p6":"fft_otf","frequencies":frequencies,"pupil_input_hash":self.input_hash});
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
}
