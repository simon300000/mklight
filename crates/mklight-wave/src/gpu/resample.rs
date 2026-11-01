use super::*;
use crate::{Aperture, PupilMask, StructuredSamples};
use mklight_gpu::wave_ops::resample::{Mask, Resample};
/// Resample a traced logical lattice on the device. Small polygon loop topology
/// is supplied by the caller; triangle construction, coverage and interpolation
/// are resident. Returns reusable channels, not an intermediate host pupil copy.
#[allow(clippy::too_many_arguments)]
pub fn from_structured(
    backend: &dyn ComputeBackend,
    snapshot: Snapshot,
    rays: &StructuredSamples,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    mask: Option<&PupilMask>,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<WaveResult<ResidentPupil>> {
    if rays.nx < 2
        || rays.ny < 2
        || rays.samples.len() != rays.nx * rays.ny
        || !wavelength_um.is_finite()
        || wavelength_um <= 0.
    {
        return Err(MkError::InvalidParam("pupil lattice/wavelength".into()));
    }
    geometry.validate()?;
    if let Some(mask) = mask {
        mask.validate()?;
    }
    let mut hash = blake3::Hasher::new();
    hash.update(&serde_json::to_vec(&(n, supersample, wavelength_um, geometry, mask)).unwrap());
    let nodes = backend.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [rays.ny, rays.nx, 4]),
        &mut |mut b| {
            let out = b.as_mut_slice::<Df32>()?;
            for (i, row) in out.chunks_mut(4).enumerate() {
                if i % 4096 == 0 {
                    ctx.check()?;
                }
                hash.update(&[u8::from(rays.samples[i].is_some())]);
                let a = match rays.samples[i] {
                    Some(r) => [r.u, r.v, r.opd_waves, r.amplitude],
                    None => [0.; 4],
                };
                for (v, a) in row.iter_mut().zip(a) {
                    if !a.is_finite() || !Df32::from_f64(a).is_finite() {
                        return Err(MkError::InvalidParam("non-finite pupil sample".into()));
                    }
                    hash.update(&a.to_le_bytes());
                    *v = Df32::from_f64(a);
                }
            }
            Ok(())
        },
    )?;
    let valid = backend.fill_buffer(
        BufferDesc::new(ElementType::U32, [rays.ny, rays.nx]),
        &mut |mut b| {
            for (v, s) in b.as_mut_slice::<u32>()?.iter_mut().zip(&rays.samples) {
                *v = s.is_some() as u32;
            }
            Ok(())
        },
    )?;
    from_resident_lattice(
        backend,
        snapshot,
        nodes.view(),
        valid.view(),
        hash.finalize().to_hex().to_string(),
        n,
        wavelength_um,
        geometry,
        mask,
        supersample,
        ctx,
    )
}

/// Resample already resident [ny,nx,4] lattice nodes (u,v,OPD,amplitude).
#[allow(clippy::too_many_arguments)]
pub fn from_resident_lattice(
    backend: &dyn ComputeBackend,
    snapshot: Snapshot,
    nodes: BufferView,
    valid: BufferView,
    input_hash: String,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    mask: Option<&PupilMask>,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<WaveResult<ResidentPupil>> {
    geometry.validate()?;
    if let Some(mask) = mask {
        mask.validate()?;
    }
    let edge_count = match mask {
        Some(PupilMask::Polygons { loops }) => loops.iter().map(Vec::len).sum(),
        _ => 0,
    };
    let edges = backend.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [edge_count, 4]),
        &mut |mut b| {
            if let Some(PupilMask::Polygons { loops }) = mask {
                let mut out = b.as_mut_slice::<Df32>()?.chunks_mut(4);
                for l in loops {
                    for i in 0..l.len() {
                        let a = l[i];
                        let z = l[(i + 1) % l.len()];
                        out.next()
                            .unwrap()
                            .copy_from_slice(&[a[0], a[1], z[0], z[1]].map(Df32::from_f64));
                    }
                }
            }
            Ok(())
        },
    )?;
    let mask_kind = match mask {
        None => Mask::Triangles,
        Some(PupilMask::Analytic { aperture }) => match *aperture {
            Aperture::Circular => Mask::Circular,
            Aperture::Annular { obscuration } => Mask::Annular(obscuration as f32),
            Aperture::Rectangular { half_u, half_v } => {
                Mask::Rectangle([half_u as f32, half_v as f32])
            }
        },
        Some(PupilMask::Polygons { .. }) => Mask::Polygons,
    };
    let mut p = ResidentPlan::new(snapshot);
    p.settings = serde_json::json!({"p6":"structured_pupil","grid":n,"supersample":supersample,"wavelength_um":wavelength_um});
    let nodes = p.import(nodes);
    let valid = p.import(valid);
    let edges = p.import(edges.view());
    let amplitude = p.scratch(BufferDesc::new(ElementType::F32, [n, n]));
    let coverage = p.scratch(BufferDesc::new(ElementType::F32, [n, n]));
    let opd = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n, n]));
    let diagnostics = p.scratch(BufferDesc::new(ElementType::F32, [3, n * n]));
    p.operations
        .push(ResidentOp::Wave(w::Request::Resample(Resample {
            nodes: p.view(nodes)?,
            valid: p.view(valid)?,
            edges: p.view(edges)?,
            mask: mask_kind,
            size: n,
            supersample,
            amplitude,
            coverage,
            opd,
            diagnostics,
        })));
    p.settings["pupil_input_hash"] = input_hash.clone().into();
    let summary = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [10]));
    p.operations.push(ResidentOp::Wave(w::Request::Diagnostics(
        w::diagnostics::Diagnostics {
            coverage: p.view(coverage)?,
            opd: p.view(opd)?,
            extrapolation: p.view(diagnostics)?,
            output: summary,
        },
    )));
    p.outputs = vec![amplitude, coverage, opd, summary];
    let mut r = backend.submit_plan(p, ctx)?.wait()?;
    let guard = backend.read_view(r.outputs[3].view())?;
    let d = guard.as_slice::<Df32>()?;
    if d[1].hi > 0. {
        return Err(MkError::Unsupported("pupil extrapolation exceeds 128 subpoints or has no triangle source; use the reference resampler".into()));
    }
    let (gap, pitch) = (d[3].to_f64(), d[4].to_f64());
    let mut warnings = if gap > 2. * pitch {
        vec![WaveWarning::SparseRays {
            max_gap: gap,
            typical_spacing: pitch,
        }]
    } else {
        vec![]
    };
    let step = d[2].to_f64();
    let across = if d[0].hi == 0. {
        0
    } else {
        ((d[7].hi - d[5].hi + 1.).max(d[8].hi - d[6].hi + 1.)) as usize
    };
    if across < crate::pupil::MIN_PUPIL_SAMPLES {
        warnings.push(WaveWarning::FewPupilSamples {
            samples: across,
            recommended: crate::pupil::MIN_PUPIL_SAMPLES,
        });
    }
    if step > 0.5 {
        warnings.push(WaveWarning::PupilUndersampled {
            max_step_waves: step,
        });
    }
    r.run_info
        .warnings
        .extend(warnings.iter().map(ToString::to_string));
    let covered_samples = d[0].hi as usize;
    let value = ResidentPupil {
        input_hash,
        covered_samples,
        max_phase_step: step,
        clear_area: d[9].to_f64() * 4. / (n * n) as f64,
        amplitude: r.outputs[0].clone(),
        coverage: r.outputs[1].clone(),
        opd: r.outputs[2].view(),
        n,
        wavelength_um,
        geometry,
        image_offset_mm: [0.; 2],
        warnings,
    };
    Ok(WaveResult {
        value,
        run_info: r.run_info,
        stats: r.stats,
    })
}
