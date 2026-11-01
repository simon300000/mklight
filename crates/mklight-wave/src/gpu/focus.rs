use super::traced::accumulate;
use super::*;
use mklight_gpu::wave_ops::{fft_mtf::FftMtf, image_integrals::ImageIntegral, spectral::Spectral};
impl ResidentPupil {
    pub fn append_focus_correlation(
        &self,
        p: &mut ResidentPlan,
        frequencies: PlanView,
        shifts: PlanView,
    ) -> MkResult<BufferId> {
        self.append_sensor_guard(p)?;
        let count = frequencies.desc.shape[0];
        let planes = shifts.desc.shape[0];
        let pupil = self.import(p);
        let stencil = p.scratch(BufferDesc::new(
            ElementType::F32,
            [count, self.n * self.n, 5],
        ));
        let frame = self
            .geometry
            .chief_frame()
            .map(|v| [v.x, v.y, v.z].map(Df32::from_f64));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::PrepareOpticalStencil(
                tr::OpticalCorrelation {
                    stencil: None,
                    amplitude: pupil.amplitude.clone(),
                    coverage: pupil.coverage.clone(),
                    opd: pupil.opd.clone(),
                    frequencies,
                    frame,
                    sin_max: self.wave_geometry().sin_max,
                    wavelength_over_index: self.wave_geometry().wavelength,
                    output: stencil,
                },
            )));
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [planes, count]));
        p.operations
            .push(ResidentOp::Wave(w::Request::FocusCorrelation {
                on_axis: self.geometry.is_on_axis(),
                pupil,
                geometry: self.wave_geometry(),
                stencil: p.view(stencil)?,
                shifts: Some(shifts),
                output,
            }));
        Ok(output)
    }
    pub fn focus_batch(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        shifts: &[f64],
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Vec<ResidentPupil>>> {
        let z = shift_buffer(b, shifts, ctx)?;
        let mut p = ResidentPlan::new(snapshot);
        let z = p.import(z.view());
        let pupil = self.import(&mut p);
        let output = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [shifts.len(), self.n, self.n],
        ));
        p.operations.push(ResidentOp::Wave(w::Request::Focus {
            pupil,
            geometry: self.wave_geometry(),
            shifts: p.view(z)?,
            output,
        }));
        p.outputs = vec![output];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let value = shifts
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let mut p = self.clone();
                p.opd = r.outputs[0].view();
                p.opd.offset = i * self.n * self.n;
                p.opd.desc.shape = vec![self.n, self.n];
                p.input_hash = format!("{}:focus:{z}", self.input_hash);
                p
            })
            .collect();
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    pub fn phase_step(
        &self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<f64>> {
        let zeros = b.fill_buffer(
            BufferDesc::new(ElementType::F32, [3, self.n * self.n]),
            &mut |mut a| {
                a.as_mut_slice::<f32>()?.fill(0.);
                Ok(())
            },
        )?;
        let mut p = ResidentPlan::new(snapshot);
        let v = self.import(&mut p);
        let d = p.import(zeros.view());
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [10]));
        p.operations.push(ResidentOp::Wave(w::Request::Diagnostics(
            w::diagnostics::Diagnostics {
                coverage: v.coverage,
                opd: v.opd,
                extrapolation: p.view(d)?,
                output,
            },
        )));
        p.outputs = vec![output];
        let r = b.submit_plan(p, ctx)?.wait()?;
        let value = b.read_view(r.outputs[0].view())?.as_slice::<Df32>()?[2].to_f64();
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
fn shift_buffer(
    b: &dyn ComputeBackend,
    shifts: &[f64],
    ctx: &JobContext,
) -> MkResult<BufferHandle> {
    b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [shifts.len()]),
        &mut |mut a| {
            for (i, (v, z)) in a.as_mut_slice::<Df32>()?.iter_mut().zip(shifts).enumerate() {
                if i % 4096 == 0 {
                    ctx.check()?;
                }
                if !z.is_finite() {
                    return Err(MkError::InvalidParam("focus shift".into()));
                }
                *v = Df32::from_f64(*z);
            }
            Ok(())
        },
    )
}
/// All focus planes share the direct correlation stencil. FFT/Huygens work is
/// grouped in bounded plane batches; no per-plane plans or image readbacks.
pub fn focus_otf(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    pupils: &[(&ResidentPupil, f64)],
    focus: &[f64],
    frequencies: &[[f64; 2]],
    method: crate::analysis::MtfAnalysisMethod,
    huygens: crate::analysis::HuygensGrid,
    ctx: &JobContext,
) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
    let sum = pupils.iter().map(|(_, w)| w).sum::<f64>();
    if pupils.is_empty()
        || !sum.is_finite()
        || sum <= 0.
        || pupils.iter().any(|(_, w)| !w.is_finite() || *w < 0.)
    {
        return Err(MkError::InvalidParam("focus pupil weights".into()));
    }
    if method == crate::analysis::MtfAnalysisMethod::Huygens
        && pupils.iter().map(|(p, _)| p.covered_samples).sum::<usize>() as f64
            * (huygens.grid_size as f64).powi(2)
            > crate::analysis::MAX_HUYGENS_WORK
    {
        return Err(MkError::InvalidParam(
            "Huygens work exceeds MAX_HUYGENS_WORK".into(),
        ));
    }
    let f = frequency_buffer(b, frequencies, ctx)?;
    let mut stats = PlanStats::default();
    if method == crate::analysis::MtfAnalysisMethod::Direct {
        let repeated = b.fill_buffer(
            BufferDesc::new(ElementType::F32, [focus.len() * frequencies.len(), 2]),
            &mut |mut a| {
                for (i, row) in a.as_mut_slice::<f32>()?.chunks_mut(2).enumerate() {
                    row.copy_from_slice(&frequencies[i % frequencies.len()].map(|v| v as f32));
                }
                Ok(())
            },
        )?;
        let mut previous: Option<BufferHandle> = None;
        let mut info = None;
        let chunks = pupils.len().div_ceil(15);
        for (chunk, group) in pupils.chunks(15).enumerate() {
            let mut p = ResidentPlan::new(snapshot.clone());
            let fi = p.import(f.view());
            let mut inputs = vec![];
            let mut weights = vec![];
            let mut offsets = vec![];
            if let Some(h) = &previous {
                let id = p.import(h.view());
                inputs.push(p.view(id)?);
                weights.push(Df32::from_f32(1.));
                offsets.push([Df32::ZERO; 2]);
            }
            for (pupil, weight) in group {
                let z: Vec<_> = focus
                    .iter()
                    .map(|z| z / pupil.geometry.chief_ray[2])
                    .collect();
                let z = shift_buffer(b, &z, ctx)?;
                let z = p.import(z.view());
                let output = pupil.append_focus_correlation(
                    &mut p,
                    p_view(fi, f.desc()),
                    p_view(z, &BufferDesc::new(ElementType::DoubleFloat, [focus.len()])),
                )?;
                let mut view = p.view(output)?;
                view.desc.shape = vec![focus.len() * frequencies.len()];
                inputs.push(view);
                weights.push(Df32::from_f64(weight / sum));
                offsets.push(pupil.image_offset_mm.map(Df32::from_f64));
            }
            let ff = p.import(repeated.view());
            let output = p.scratch(BufferDesc::new(
                ElementType::Complex32,
                [focus.len() * frequencies.len()],
            ));
            p.operations
                .push(ResidentOp::Wave(w::Request::Spectral(Spectral {
                    inputs,
                    weights,
                    frequencies: Some(p.view(ff)?),
                    offsets,
                    output,
                })));
            p.outputs = vec![output];
            let r = b
                .submit_plan(
                    p,
                    &ctx.subrange(
                        chunk as f64 / chunks as f64,
                        (chunk + 1) as f64 / chunks as f64,
                    ),
                )?
                .wait()?;
            accumulate(&mut stats, &r.stats);
            previous = Some(r.outputs[0].clone());
            info = Some(r.run_info);
        }
        let value = b
            .read_view(previous.unwrap().view())?
            .as_slice::<[f32; 2]>()?
            .iter()
            .map(|v| v.map(f64::from))
            .collect();
        return Ok(WaveResult {
            value,
            run_info: info.unwrap(),
            stats,
        });
    }
    let mut value = vec![];
    let mut info = None;
    let max_side = pupils.iter().map(|(p, _)| p.n).max().unwrap();
    let batch = (16 * 1024 * 1024 / (max_side * max_side * pupils.len())).clamp(1, 4);
    let chunks = focus.len().div_ceil(batch);
    let ideal: Vec<_> = pupils
        .iter()
        .map(|(p, _)| p.without_aberrations(b, ctx))
        .collect::<MkResult<_>>()?;
    for (chunk, planes) in focus.chunks(batch).enumerate() {
        let sub = ctx.subrange(
            chunk as f64 / chunks as f64,
            (chunk + 1) as f64 / chunks as f64,
        );
        let mut shifted = vec![];
        for (pupil, _) in pupils {
            let z: Vec<_> = planes
                .iter()
                .map(|z| z / pupil.geometry.chief_ray[2])
                .collect();
            let r = pupil.focus_batch(b, snapshot.clone(), &z, &sub.subrange(0., 0.1))?;
            accumulate(&mut stats, &r.stats);
            shifted.push(r.value);
        }
        let mut p = ResidentPlan::new(snapshot.clone());
        let fi = p.import(f.view());
        let mut references = vec![];
        if method == crate::analysis::MtfAnalysisMethod::Fft {
            for clean in &ideal {
                references.push(clean.fft_lattice(&mut p)?);
            }
        }
        let query = if method == crate::analysis::MtfAnalysisMethod::Huygens {
            Some(b.fill_buffer(
                BufferDesc::new(ElementType::DoubleFloat, [frequencies.len(), 2]),
                &mut |mut a| {
                    for (row, f) in a.as_mut_slice::<Df32>()?.chunks_mut(2).zip(frequencies) {
                        row.copy_from_slice(&f.map(Df32::from_f64));
                    }
                    Ok(())
                },
            )?)
        } else {
            None
        };
        let query = query.as_ref().map(|q| p.import(q.view()));
        let spacing = pupils
            .iter()
            .map(|(p, _)| p.lambda_f() / huygens.padding)
            .fold(f64::INFINITY, f64::min);
        for plane in 0..planes.len() {
            let mut inputs = vec![];
            for (wi, (original, _)) in pupils.iter().enumerate() {
                let pupil = &shifted[wi][plane];
                let output =
                    p.scratch(BufferDesc::new(ElementType::Complex32, [frequencies.len()]));
                if method == crate::analysis::MtfAnalysisMethod::Fft {
                    let data = pupil.fft_lattice(&mut p)?;
                    p.operations
                        .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sample {
                            data: p.view(data)?,
                            reference: p.view(references[wi])?,
                            frequencies: p.view(fi)?,
                            spacing: Df32::from_f64(1. / (pupil.n as f64 * pupil.lambda_f())),
                            mapping: pupil
                                .geometry
                                .surface_to_normal_frequency()
                                .map(|v| v.map(Df32::from_f64)),
                            output,
                        })));
                } else {
                    let mut image_plane = crate::ImagePlane::image_surface(&pupil.geometry);
                    let off = original.image_offset_mm;
                    image_plane.origin_mm =
                        image_plane.x_axis * (-off[0]) + image_plane.y_axis * (-off[1]);
                    let psf = pupil.append_huygens(
                        &mut p,
                        &HuygensSettings {
                            grid_size: huygens.grid_size,
                            spacing_mm: spacing,
                            normalization: PsfNormalization::UnitEnergy,
                            plane: image_plane,
                        },
                    )?;
                    p.operations
                        .push(ResidentOp::Wave(w::Request::ImageIntegral(
                            ImageIntegral::Query {
                                input: p.view(psf.image)?,
                                queries: p.view(query.unwrap())?,
                                kind: 0,
                                spacing: Df32::from_f64(spacing),
                                center: [Df32::from_f64((huygens.grid_size / 2) as f64); 2],
                                normalization: Df32::from_f64(1. / (spacing * spacing)),
                                output,
                            },
                        )));
                }
                inputs.push(p.view(output)?);
            }
            let mut combined = None;
            for (batch, group) in inputs.chunks(15).enumerate() {
                let mut views = vec![];
                let mut weights = vec![];
                let mut offsets = vec![];
                if let Some(id) = combined {
                    views.push(p.view(id)?);
                    weights.push(Df32::from_f32(1.));
                    offsets.push([Df32::ZERO; 2]);
                }
                for (j, input) in group.iter().enumerate() {
                    let (pupil, w) = pupils[batch * 15 + j];
                    views.push(input.clone());
                    weights.push(Df32::from_f64(w / sum));
                    offsets.push(pupil.image_offset_mm.map(Df32::from_f64));
                }
                let output =
                    p.scratch(BufferDesc::new(ElementType::Complex32, [frequencies.len()]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Spectral(Spectral {
                        inputs: views,
                        weights,
                        frequencies: if method == crate::analysis::MtfAnalysisMethod::Fft {
                            Some(p.view(fi)?)
                        } else {
                            None
                        },
                        offsets,
                        output,
                    })));
                combined = Some(output);
            }
            p.outputs.push(combined.unwrap());
        }
        let r = b.submit_plan(p, &sub.subrange(0.1, 1.))?.wait()?;
        accumulate(&mut stats, &r.stats);
        for output in &r.outputs {
            value.extend(
                b.read_view(output.view())?
                    .as_slice::<[f32; 2]>()?
                    .iter()
                    .map(|v| v.map(f64::from)),
            );
        }
        info = Some(r.run_info);
    }
    Ok(WaveResult {
        value,
        run_info: info.ok_or_else(|| MkError::InvalidParam("no focus planes".into()))?,
        stats,
    })
}
