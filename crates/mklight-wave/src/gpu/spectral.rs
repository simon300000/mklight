use super::*;
use crate::{ImagePlane, PolyPsfSettings};
use mklight_gpu::wave_ops::spectral::Spectral;
/// A complete PSF remains resident for energy, MTF and image convolution.
#[derive(Clone, Debug)]
pub struct ResidentPsf {
    pub image: BufferHandle,
    pub size: usize,
    pub spacing_mm: f64,
    pub center_mm: [f64; 2],
    pub normalization: PsfNormalization,
    pub total_energy: f64,
    pub diffraction_peak: f64,
    pub wavelength_um: Option<f64>,
    pub warnings: Vec<WaveWarning>,
}
impl ResidentPsf {
    pub fn finish(&self, b: &dyn ComputeBackend, ctx: &JobContext) -> MkResult<Psf> {
        let guard = b.read_view(self.image.view())?;
        let mut data = Vec::with_capacity(self.size * self.size);
        for block in guard.as_slice::<f32>()?.chunks(4096) {
            ctx.check()?;
            data.extend(block.iter().map(|v| *v as f64));
        }
        Ok(Psf {
            size: self.size,
            spacing_mm: self.spacing_mm,
            center_mm: self.center_mm,
            data,
            normalization: self.normalization,
            total_energy: self.total_energy,
            diffraction_peak: self.diffraction_peak,
            wavelength_um: self.wavelength_um,
            warnings: self.warnings.clone(),
        })
    }
}
fn weights(pupils: &[(&ResidentPupil, f64)]) -> MkResult<f64> {
    let sum = pupils.iter().map(|(_, w)| w).sum::<f64>();
    if pupils.is_empty()
        || !sum.is_finite()
        || sum <= 0.
        || pupils.iter().any(|(_, w)| !w.is_finite() || *w < 0.)
    {
        return Err(MkError::InvalidParam(
            "spectral pupils need positive weights".into(),
        ));
    }
    Ok(sum)
}
/// Common-grid FFT/MFT or Huygens PSF with spectral accumulation on device.
pub fn polychromatic_psf(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    pupils: &[(&ResidentPupil, f64)],
    settings: &PolyPsfSettings,
    huygens: bool,
    ctx: &JobContext,
) -> MkResult<WaveResult<ResidentPsf>> {
    let sum = weights(pupils)?;
    if huygens
        && (pupils.iter().map(|(p, _)| p.covered_samples).sum::<usize>() as f64)
            * (settings.grid_size as f64).powi(2)
            > crate::analysis::MAX_HUYGENS_WORK
    {
        return Err(MkError::InvalidParam(
            "Huygens work exceeds MAX_HUYGENS_WORK".into(),
        ));
    }
    let mut p = ResidentPlan::new(snapshot.clone());
    p.settings =
        serde_json::json!({"operation":"polychromatic_psf","settings":settings,"huygens":huygens});
    let mut outputs = vec![];
    let mut warnings = vec![];
    for (pupil, _) in pupils {
        let off = pupil.image_offset_mm;
        let o = if huygens {
            let mut plane = ImagePlane::image_surface(&pupil.geometry);
            plane.origin_mm = plane.x_axis * (-off[0]) + plane.y_axis * (-off[1]);
            pupil.append_huygens(
                &mut p,
                &HuygensSettings {
                    grid_size: settings.grid_size,
                    spacing_mm: settings.spacing_mm,
                    normalization: PsfNormalization::UnitEnergy,
                    plane,
                },
            )?
        } else {
            pupil.append_psf(
                &mut p,
                &PsfSettings {
                    grid_size: settings.grid_size,
                    sampling: ImageSampling::Spacing {
                        mm: settings.spacing_mm,
                    },
                    normalization: PsfNormalization::UnitEnergy,
                    transform: PsfTransform::Matrix,
                },
                [-off[0], -off[1]],
            )?
        };
        for w in &o.warnings {
            crate::warnings::push_unique(&mut warnings, w.clone());
        }
        p.outputs.extend([o.image, o.sums]);
        outputs.push(o);
    }
    let r = b.submit_plan(p, &ctx.subrange(0., 0.9))?.wait()?;
    let mut stats = r.stats;
    let mut peak = 0.;
    for (i, (pupil, w)) in pupils.iter().enumerate() {
        let s = b.read_view(r.outputs[2 * i + 1].view())?;
        let s = s.as_slice::<Df32>()?;
        let l = pupil.n as f64 * pupil.lambda_f();
        let energy = l
            * l
            * if huygens {
                s[2].to_f64()
            } else {
                s[1].to_f64() / pupil.geometry.projection_jacobian()
            };
        peak += w / sum * s[0].to_f64().powi(2) / energy;
    }
    if !(peak > 0. && peak.is_finite()) {
        return Err(MkError::Numerical(
            "pupil has no diffraction normalization".into(),
        ));
    }
    let mut combined: Option<BufferHandle> = None;
    let chunks = pupils.len().div_ceil(15);
    let mut run_info = r.run_info;
    for (chunk, group) in pupils.chunks(15).enumerate() {
        let mut p = ResidentPlan::new(snapshot.clone());
        let mut inputs = vec![];
        let mut weights = vec![];
        if let Some(h) = &combined {
            let id = p.import(h.view());
            inputs.push(p.view(id)?);
            weights.push(Df32::from_f32(1.));
        }
        for (i, (_, w)) in group.iter().enumerate() {
            let id = p.import(r.outputs[2 * (chunk * 15 + i)].view());
            inputs.push(p.view(id)?);
            weights.push(Df32::from_f64(
                w / sum
                    / if settings.normalization == PsfNormalization::Strehl {
                        peak
                    } else {
                        1.
                    },
            ));
        }
        let output = p.scratch(BufferDesc::new(
            ElementType::F32,
            [settings.grid_size, settings.grid_size],
        ));
        let count = inputs.len();
        p.operations
            .push(ResidentOp::Wave(w::Request::Spectral(Spectral {
                inputs,
                weights,
                frequencies: None,
                offsets: vec![[Df32::ZERO; 2]; count],
                output,
            })));
        p.outputs = vec![output];
        if chunk + 1 == chunks {
            let metrics = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5]));
            p.operations
                .push(ResidentOp::Wave(w::Request::ImageIntegral(
                    w::image_integrals::ImageIntegral::Metrics {
                        input: p.view(output)?,
                        output: metrics,
                    },
                )));
            p.outputs.push(metrics);
        }

        let r = b
            .submit_plan(
                p,
                &ctx.subrange(
                    0.9 + 0.1 * chunk as f64 / chunks as f64,
                    0.9 + 0.1 * (chunk + 1) as f64 / chunks as f64,
                ),
            )?
            .wait()?;
        accumulate(&mut stats, &r.stats);
        if chunk + 1 == chunks {
            let sums = b.read_view(r.outputs[1].view())?;
            let energy = sums.as_slice::<Df32>()?[0].to_f64()
                * settings.spacing_mm.powi(2)
                * if settings.normalization == PsfNormalization::Strehl {
                    peak
                } else {
                    1.
                };
            if energy < crate::analysis::MIN_ENERGY_IN_GRID {
                crate::warnings::push_unique(
                    &mut warnings,
                    WaveWarning::PsfTruncated {
                        energy_in_grid: energy,
                    },
                );
            }
        }
        combined = Some(r.outputs[0].clone());
    }
    run_info
        .warnings
        .extend(warnings.iter().map(ToString::to_string));
    run_info.settings = serde_json::json!({"operation":"polychromatic_psf","settings":settings,"huygens":huygens,"resident_stats":stats});
    Ok(WaveResult {
        value: ResidentPsf {
            image: combined.unwrap(),
            size: settings.grid_size,
            spacing_mm: settings.spacing_mm,
            center_mm: [0.; 2],
            normalization: settings.normalization,
            total_energy: if settings.normalization == PsfNormalization::Strehl {
                1. / peak
            } else {
                1.
            },
            diffraction_peak: if settings.normalization == PsfNormalization::Strehl {
                1.
            } else {
                peak
            },
            wavelength_um: None,
            warnings,
        },
        run_info,
        stats,
    })
}
/// Complex polychromatic direct OTF, with lateral colour phases summed before magnitude.
pub fn polychromatic_otf(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    pupils: &[(&ResidentPupil, f64)],
    frequencies: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
    polychromatic_otf_method(b, snapshot, pupils, frequencies, false, ctx)
}

pub fn polychromatic_otf_method(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    pupils: &[(&ResidentPupil, f64)],
    frequencies: &[[f64; 2]],
    fft: bool,
    ctx: &JobContext,
) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
    let sum = weights(pupils)?;
    let f = frequency_buffer(b, frequencies, ctx)?;
    let mut combined = None;
    let mut stats = PlanStats::default();
    let mut info = None;
    let chunks = pupils.len().div_ceil(15);
    for (chunk, group) in pupils.chunks(15).enumerate() {
        let mut p = ResidentPlan::new(snapshot.clone());
        let fi = p.import(f.view());
        let mut inputs = vec![];
        let mut weights = vec![];
        let mut offsets = vec![];
        if let Some(h) = &combined {
            let h: &BufferHandle = h;
            let id = p.import(h.view());
            inputs.push(p.view(id)?);
            weights.push(Df32::from_f32(1.));
            offsets.push([Df32::ZERO; 2]);
        }
        for (pupil, w) in group {
            let out = if fft {
                let ideal = pupil.without_aberrations(b, ctx)?;
                pupil.append_fft_otf(&mut p, &ideal, p_view(fi, f.desc()))?
            } else {
                pupil.append_correlation(&mut p, p_view(fi, f.desc()))?
            };
            let mut v = p.view(out)?;
            v.desc.shape = vec![frequencies.len()];
            inputs.push(v);
            weights.push(Df32::from_f64(w / sum));
            offsets.push(pupil.image_offset_mm.map(Df32::from_f64));
        }
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [frequencies.len()]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Spectral(Spectral {
                inputs,
                weights,
                frequencies: Some(p.view(fi)?),
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
        combined = Some(r.outputs[0].clone());
        info = Some(r.run_info);
    }
    let guard = b.read_view(combined.unwrap().view())?;
    let value = guard
        .as_slice::<[f32; 2]>()?
        .iter()
        .map(|v| v.map(f64::from))
        .collect();
    let mut run_info = info.unwrap();
    run_info.settings = serde_json::json!({"operation":"polychromatic_otf","frequencies":frequencies,"resident_stats":stats});
    Ok(WaveResult {
        value,
        run_info,
        stats,
    })
}
use super::traced::accumulate;
