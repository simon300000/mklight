//! Native result adapters. Engine routing and qualification policy remain in P11.
use super::traced::accumulate;
use super::*;
use crate::analysis::{self as a, PupilSampling};
use crate::{PupilSettings, ReferencePoint, ZernikeBasis};
use mklight_core::job::RunTimer;
use mklight_trace::SequentialTracer;
use serde::Serialize;
pub(super) struct Field {
    pub(super) index: usize,
    pub(super) value: [f64; 2],
    pub(super) reference: [f64; 2],
    pub(super) pupils: Vec<(usize, f64, ResidentTracedPupil)>,
}
impl Field {
    pub(super) fn weighted(&self) -> Vec<(&ResidentPupil, f64)> {
        self.pupils.iter().map(|(_, w, p)| (&p.pupil, *w)).collect()
    }
    pub(super) fn sampling(&self) -> Vec<PupilSampling> {
        self.pupils
            .iter()
            .map(|(wl, _, p)| PupilSampling {
                wavelength: *wl,
                grid: p.pupil.n,
            })
            .collect()
    }
}
pub(super) fn info(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    timer: RunTimer,
    name: &str,
    settings: &impl Serialize,
    stats: &PlanStats,
    warnings: Vec<String>,
) -> RunInfo {
    let mut r = timer.finish(
        &t.system().snapshot,
        name,
        settings,
        &b.info().backend_id(),
        b.precision().model,
        None,
        warnings,
    );
    r.settings["resident"] = serde_json::json!({"version":1,"device":b.info(),"precision":b.precision(),"plan_stats":stats});
    r
}
#[allow(clippy::too_many_arguments)]
fn fields(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    indices: &[usize],
    wls: &[(usize, f64)],
    settings: &PupilSettings,
    ctx: &JobContext,
    stats: &mut PlanStats,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<Field>> {
    let service = PupilTracer::new(b, t, ctx)?;
    let requests: Vec<_> = indices
        .iter()
        .map(|&index| {
            let f = &t.system().system().fields.fields[index];
            (index, [f.x, f.y], t.field(index))
        })
        .collect();
    let traced = trace_fields(&service, t, &requests, wls, settings, ctx, stats, warnings)?;
    let mut out = vec![];
    let mut first_error = None;
    for (&index, f) in indices.iter().zip(traced) {
        match f {
            Ok(f) => out.push(f),
            Err(e) => {
                warnings.push(format!("field {index} skipped: {e}"));
                first_error.get_or_insert(e);
            }
        }
    }
    if out.is_empty()
        && let Some(e) = first_error
    {
        return Err(e);
    }
    Ok(out)
}
/// Trace every requested field at `wls` in shared batches. A field whose
/// pupil fails with a Trace/Numerical error (first in wavelength order) is
/// returned as that error; any other error fails the call.
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_fields(
    service: &PupilTracer<'_>,
    t: &SequentialTracer,
    requests: &[(usize, [f64; 2], &mklight_trace::PreparedField)],
    wls: &[(usize, f64)],
    settings: &PupilSettings,
    ctx: &JobContext,
    stats: &mut PlanStats,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<MkResult<Field>>> {
    let pupils: Vec<_> = requests
        .iter()
        .flat_map(|&(_, _, f)| wls.iter().map(move |&(wl, _)| (f, wl)))
        .collect();
    let (traced, shared) = service.trace_many(&pupils, settings, ctx)?;
    accumulate(stats, &shared);
    let mut traced = traced.into_iter();
    let mut out = vec![];
    for &(index, value, _) in requests {
        let mut pupils = vec![];
        let mut error = None;
        for (&(wl, weight), r) in wls.iter().zip(traced.by_ref().take(wls.len())) {
            match r {
                Ok(r) => {
                    accumulate(stats, &r.stats);
                    warnings.extend(r.run_info.warnings);
                    pupils.push((wl, weight, r.value));
                }
                Err(e) => {
                    error.get_or_insert(e);
                }
            }
        }
        out.push(match error {
            Some(e) => Err(e),
            None => Ok(Field {
                index,
                value,
                reference: [0.; 2],
                pupils,
            }),
        });
    }
    // The chief reference is the primary wavelength's chief ray; when the
    // primary is not traced (or its chief fails) the first pupil's reference.
    let primary = t.system().primary_wavelength;
    let missing: Vec<_> = requests
        .iter()
        .zip(&out)
        .filter(|(_, f)| {
            settings.reference == ReferencePoint::Chief
                && f.as_ref()
                    .is_ok_and(|f| f.pupils.iter().all(|(wl, _, _)| *wl != primary))
        })
        .map(|(r, _)| r.2)
        .collect();
    let mut chiefs = if missing.is_empty() {
        vec![]
    } else {
        service.chief_points(&missing, &ctx.subrange(1., 1.), stats)?
    }
    .into_iter();
    for f in out.iter_mut().flatten() {
        let first = f.pupils[0].2.info.reference_point_mm;
        f.reference = match settings.reference {
            ReferencePoint::Chief => match f.pupils.iter().find(|(wl, _, _)| *wl == primary) {
                Some((_, _, p)) => [p.info.chief_point_mm[0], p.info.chief_point_mm[1]],
                None => chiefs.next().unwrap().unwrap_or([first[0], first[1]]),
            },
            ReferencePoint::Centroid => {
                let mut v = [0.; 2];
                for (_, w, p) in &f.pupils {
                    for (j, x) in v.iter_mut().enumerate() {
                        *x += w * p.info.reference_point_mm[j];
                    }
                }
                v
            }
        };
        for (_, _, p) in &mut f.pupils {
            p.pupil.image_offset_mm = [
                p.info.reference_point_mm[0] - f.reference[0],
                p.info.reference_point_mm[1] - f.reference[1],
            ];
        }
    }
    Ok(out)
}

/// Complete GPU-traced wavefront maps, including the original lattice QR fit.
pub fn wavefront_map_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::WavefrontSettings,
    ctx: &JobContext,
) -> MkResult<a::WavefrontResult> {
    let timer = RunTimer::start();
    let basis = ZernikeBasis::new(s.zernike_ordering, s.zernike_terms)?;
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let fields = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.7),
        &mut stats,
        &mut warnings,
    )?;
    let total = fields.iter().map(|f| f.pupils.len()).sum::<usize>().max(1);
    let mut maps = vec![];
    let mut first_error = None;
    for f in fields {
        for (wl, _, p) in f.pupils {
            let sub = ctx.subrange(
                0.7 + 0.3 * maps.len() as f64 / total as f64,
                0.7 + 0.3 * (maps.len() + 1) as f64 / total as f64,
            );
            let fit = match fit_resident_zernike(
                b,
                t.system().snapshot.clone(),
                p.nodes.view(),
                &basis,
                &sub.subrange(0., 0.5),
            ) {
                Ok(v) => v,
                Err(e @ MkError::Numerical(_)) => {
                    warnings.push(format!("field {}, wavelength {wl} skipped: {e}", f.index));
                    first_error.get_or_insert(e);
                    continue;
                }
                Err(e) => return Err(e),
            };
            accumulate(&mut stats, &fit.stats);
            warnings.extend(
                fit.run_info
                    .warnings
                    .iter()
                    .map(|w| format!("field {}, wavelength {wl}: {w}", f.index)),
            );
            let st = p
                .pupil
                .statistics(b, t.system().snapshot.clone(), &sub.subrange(0.5, 0.9))?;
            accumulate(&mut stats, &st.stats);
            let v = st.value;
            let rp = &p.pupil;
            let opd = b.read_view(rp.opd.clone())?;
            let opd = opd.as_slice::<Df32>()?;
            let c = b.read_view(rp.coverage.view())?;
            let c = c.as_slice::<f32>()?;
            let amp = b.read_view(rp.amplitude.view())?;
            let amp = amp.as_slice::<f32>()?;
            let mut waves = Vec::with_capacity(rp.n * rp.n);
            let mut amplitudes = Vec::with_capacity(rp.n * rp.n);
            for k in 0..rp.n * rp.n {
                if k % 4096 == 0 {
                    ctx.check()?;
                }
                waves.push(if c[k] > 0. { opd[k].to_f64() } else { f64::NAN });
                amplitudes.push(if c[k] > 0. { amp[k] as f64 } else { 0. });
            }
            let rim = match p.mask {
                Some(crate::PupilMask::Polygons { loops }) => loops,
                _ => vec![],
            };
            let fit = fit.value;
            maps.push(a::WavefrontMap {
                field: f.index,
                field_value: f.value,
                wavelength: wl,
                wavelength_um: rp.wavelength_um,
                exit_pupil: p.info,
                grid: rp.n,
                coordinates: (0..rp.n)
                    .map(|i| (2 * i + 1) as f64 / rp.n as f64 - 1.)
                    .collect(),
                opd_waves: waves,
                amplitude: amplitudes,
                rms_waves: v.rms,
                pv_waves: v.pv,
                rms_waves_tilt_removed: v.rms_tilt_removed,
                pv_waves_tilt_removed: v.pv_tilt_removed,
                strehl_at_reference: v.strehl,
                strehl_marechal: crate::energy::strehl_marechal(v.rms_tilt_removed),
                seidel: fit.set.seidel()?,
                zernike: fit.set,
                zernike_residual_rms: fit.residual_rms,
                zernike_condition: fit.condition_inf,
                rim,
            });
            sub.report(1., "wavefront assembly");
        }
    }
    warnings.push("Zernike condition is the equilibrated R infinity-norm estimate".into());
    if maps.is_empty()
        && let Some(e) = first_error
    {
        return Err(e);
    }
    ctx.report(1., "wavefront map");
    Ok(a::WavefrontResult {
        maps,
        run_info: info(b, t, timer, "wavefront_map", s, &stats, warnings),
    })
}
/// Automatic-grid native FFT/MFT or Huygens PSFs with resident spectral sums.
pub fn psf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::PsfAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<a::PsfResult> {
    let timer = RunTimer::start();
    if s.grid_size > 4096 {
        return Err(MkError::InvalidParam("PSF grid size".into()));
    }
    if s.method == a::PsfMethod::Huygens && s.grid_size != 0 {
        a::psf::check_huygens_grid(s.grid_size)?;
    }
    a::check_pos("padding", s.padding)?;
    if let Some(dx) = s.spacing_mm {
        a::check_pos("PSF spacing", dx)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let per = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.4),
        &mut stats,
        &mut warnings,
    )?;
    let total = per.len().max(1);
    let mut psfs = vec![];
    for (index, mut f) in per.into_iter().enumerate() {
        let sub = ctx.subrange(
            0.4 + 0.6 * index as f64 / total as f64,
            0.4 + 0.6 * (index + 1) as f64 / total as f64,
        );
        let spacing = s.spacing_mm.unwrap_or_else(|| {
            f.pupils
                .iter()
                .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
                .fold(f64::INFINITY, f64::min)
        });
        let grid = if s.grid_size != 0 {
            s.grid_size
        } else {
            let mut side = 0f64;
            for (_, _, p) in &f.pupils {
                let r = p
                    .pupil
                    .blur(b, t.system().snapshot.clone(), &sub.subrange(0., 0.1))?;
                accumulate(&mut stats, &r.stats);
                side = side.max(2. * (r.value.extent + 16. * p.pupil.lambda_f()));
            }
            ((side / spacing).ceil() as usize).clamp(
                64,
                if s.method == a::PsfMethod::Huygens {
                    1024
                } else {
                    2048
                },
            )
        };
        if s.pupil.grid == 0 {
            for (_, _, p) in &mut f.pupils {
                let n = ((grid as f64 * spacing / p.pupil.lambda_f())
                    .max(p.pupil.n as f64)
                    .min(1024.)
                    .ceil() as usize)
                    .next_power_of_two();
                if n != p.pupil.n {
                    let r = p.resample(
                        b,
                        t.system().snapshot.clone(),
                        n,
                        s.pupil.supersample,
                        &sub.subrange(0.1, 0.2),
                    )?;
                    accumulate(&mut stats, &r.stats);
                    warnings.extend(r.run_info.warnings);
                }
            }
        }
        let groups: Vec<_> = if s.polychromatic {
            vec![((f.pupils.len() == 1).then(|| f.pupils[0].0), f.weighted())]
        } else {
            f.pupils
                .iter()
                .map(|(wl, _, p)| (Some(*wl), vec![(&p.pupil, 1.)]))
                .collect()
        };
        for (j, (wl, group)) in groups.iter().enumerate() {
            let r = polychromatic_psf(
                b,
                t.system().snapshot.clone(),
                group,
                &crate::PolyPsfSettings {
                    grid_size: grid,
                    spacing_mm: spacing,
                    normalization: s.normalization,
                    transform: PsfTransform::Matrix,
                },
                s.method == a::PsfMethod::Huygens,
                &sub.subrange(
                    0.2 + 0.7 * j as f64 / groups.len() as f64,
                    0.2 + 0.7 * (j + 1) as f64 / groups.len() as f64,
                ),
            )?;
            accumulate(&mut stats, &r.stats);
            let mut psf = r.value.finish(b, &sub.subrange(0.9, 1.))?;
            if group.len() == 1 {
                psf.wavelength_um = Some(group[0].0.wavelength_um);
            }
            warnings.extend(psf.warnings.iter().map(ToString::to_string));
            let (_, i, j) = psf.peak();
            psfs.push(a::FieldPsf {
                field: f.index,
                field_value: f.value,
                wavelength: *wl,
                pupil_sampling: f
                    .sampling()
                    .into_iter()
                    .filter(|s| wl.is_none_or(|w| s.wavelength == w))
                    .collect(),
                reference_mm: f.reference,
                strehl: psf.strehl(),
                centroid_mm: psf.centroid(),
                peak_mm: [psf.x(i), psf.y(j)],
                psf,
            });
        }
    }
    ctx.report(1., "PSF");
    Ok(a::PsfResult {
        psfs,
        run_info: info(b, t, timer, "psf", s, &stats, warnings),
    })
}

#[allow(clippy::too_many_arguments)]
fn curve(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    pupils: &[(&ResidentPupil, f64)],
    freqs: &[f64],
    orientation: crate::MtfOrientation,
    method: a::MtfAnalysisMethod,
    huygens: a::HuygensGrid,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<crate::MtfCurve> {
    let vectors: Vec<_> = [orientation.tangential, orientation.sagittal()]
        .iter()
        .flat_map(|v| freqs.iter().map(move |f| [v[0] * f, v[1] * f]))
        .collect();
    let mut curve_warnings: Vec<_> = pupils
        .iter()
        .flat_map(|(p, _)| p.warnings.clone())
        .collect();
    let optical = if method == a::MtfAnalysisMethod::Huygens {
        let dx = pupils
            .iter()
            .map(|(p, _)| p.lambda_f() / huygens.padding)
            .fold(f64::INFINITY, f64::min);
        let r = polychromatic_psf(
            b,
            t.system().snapshot.clone(),
            pupils,
            &crate::PolyPsfSettings {
                grid_size: huygens.grid_size,
                spacing_mm: dx,
                normalization: PsfNormalization::UnitEnergy,
                transform: PsfTransform::Matrix,
            },
            true,
            &ctx.subrange(0., 0.7),
        )?;
        accumulate(stats, &r.stats);
        curve_warnings = r.value.warnings.clone();
        let nyquist = 0.5 / dx;
        if let Some(&f) = freqs.iter().find(|&&f| f > nyquist * (1. + 1e-12)) {
            curve_warnings.push(WaveWarning::FrequencyBeyondSampling {
                frequency: f,
                limit: nyquist,
            });
        }
        r.value.otf(
            b,
            t.system().snapshot.clone(),
            &vectors,
            &ctx.subrange(0.7, 0.8),
        )?
    } else {
        polychromatic_otf_method(
            b,
            t.system().snapshot.clone(),
            pupils,
            &vectors,
            method == a::MtfAnalysisMethod::Fft,
            &ctx.subrange(0., 0.8),
        )?
    };
    accumulate(stats, &optical.stats);
    let ideal: Vec<_> = pupils
        .iter()
        .map(|(p, w)| {
            let mut ideal = p.without_aberrations(b, ctx)?;
            ideal.image_offset_mm = [0.; 2];
            Ok((ideal, *w))
        })
        .collect::<MkResult<_>>()?;
    let ideal_refs: Vec<_> = ideal.iter().map(|(p, w)| (p, *w)).collect();
    let dl = polychromatic_otf_method(
        b,
        t.system().snapshot.clone(),
        &ideal_refs,
        &vectors,
        method != a::MtfAnalysisMethod::Direct,
        &ctx.subrange(0.8, 1.),
    )?;
    accumulate(stats, &dl.stats);
    let n = freqs.len();
    let magnitude = |v: &[f64; 2]| v[0].hypot(v[1]);
    Ok(crate::MtfCurve {
        frequencies: freqs.to_vec(),
        tangential: optical.value[..n].iter().map(magnitude).collect(),
        sagittal: optical.value[n..].iter().map(magnitude).collect(),
        diffraction_limit: dl.value[..n].iter().map(magnitude).collect(),
        cutoff: pupils
            .iter()
            .map(|(p, _)| 1. / p.lambda_f())
            .fold(0., f64::max),
        warnings: curve_warnings,
    })
}
/// Complete native direct/FFT/Huygens MTF with the unchanged automatic frequency rule.
pub fn mtf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfSettings,
    ctx: &JobContext,
) -> MkResult<a::WaveMtfResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if !(2..=100001).contains(&s.frequency_samples) {
        return Err(MkError::InvalidParam("MTF frequency count".into()));
    }
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if let Some(f) = s.max_frequency {
        a::check_pos("maximum frequency", f)?;
    }
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let per = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
        &mut stats,
        &mut warnings,
    )?;
    let maximum = if let Some(f) = s.max_frequency {
        f
    } else {
        let mut maximum = f64::INFINITY;
        for f in &per {
            let r = useful_frequency(
                b,
                t.system().snapshot.clone(),
                &f.weighted(),
                &ctx.subrange(0.5, 0.55),
            )?;
            accumulate(&mut stats, &r.stats);
            maximum = maximum.min(r.value);
        }
        maximum
    };
    let frequencies = a::linspace(0., maximum, s.frequency_samples);
    let mut out = vec![];
    for (i, f) in per.iter().enumerate() {
        let sub = ctx.subrange(
            0.55 + 0.45 * i as f64 / per.len() as f64,
            0.55 + 0.45 * (i + 1) as f64 / per.len() as f64,
        );
        let point = t.field(f.index).point;
        let o = crate::MtfOrientation::from_field(point.hx, point.hy);
        let c = curve(
            b,
            t,
            &f.weighted(),
            &frequencies,
            o,
            s.method,
            s.huygens,
            &sub.subrange(0., 0.95),
            &mut stats,
        )?;
        warnings.extend(c.warnings.iter().map(ToString::to_string));
        let detector = if let Some(d) = s.detector {
            let vectors: Vec<_> = [o.tangential, o.sagittal()]
                .iter()
                .flat_map(|v| frequencies.iter().map(move |f| [v[0] * f, v[1] * f]))
                .collect();
            let r = detector_mtf(
                b,
                t.system().snapshot.clone(),
                &d,
                &vectors,
                &sub.subrange(0.95, 1.),
            )?;
            accumulate(&mut stats, &r.stats);
            Some(crate::detector::DetectorCurve {
                tangential: r.value[..frequencies.len()].to_vec(),
                sagittal: r.value[frequencies.len()..].to_vec(),
                nyquist_tangential: d.nyquist_along(o.tangential),
                nyquist_sagittal: d.nyquist_along(o.sagittal()),
            })
        } else {
            None
        };
        let system_tangential = detector.as_ref().map(|d| {
            d.tangential
                .iter()
                .zip(&c.tangential)
                .map(|(a, b)| a * b)
                .collect()
        });
        let system_sagittal = detector.as_ref().map(|d| {
            d.sagittal
                .iter()
                .zip(&c.sagittal)
                .map(|(a, b)| a * b)
                .collect()
        });
        out.push(a::FieldMtf {
            field: f.index,
            field_value: f.value,
            pupil_sampling: f.sampling(),
            tangential_direction: o.tangential,
            tangential: c.tangential,
            sagittal: c.sagittal,
            diffraction_limit: c.diffraction_limit,
            cutoff: c.cutoff,
            detector,
            system_tangential,
            system_sagittal,
        });
    }
    ctx.report(1., "MTF");
    Ok(a::WaveMtfResult {
        frequencies,
        fields: out,
        nyquist: s.detector.map(|d| d.nyquist()),
        run_info: info(b, t, timer, "mtf", s, &stats, warnings),
    })
}
fn fixed_frequencies<'a>(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    fields: impl Iterator<Item = &'a Field>,
    list: &[f64],
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<Vec<f64>> {
    if !list.is_empty() {
        if list.iter().any(|f| !f.is_finite() || *f < 0.) {
            return Err(MkError::InvalidParam(
                "frequencies must be finite and nonnegative".into(),
            ));
        }
        return Ok(list.to_vec());
    }
    let mut limit = f64::INFINITY;
    for field in fields {
        let r = useful_frequency(b, t.system().snapshot.clone(), &field.weighted(), ctx)?;
        accumulate(stats, &r.stats);
        limit = limit.min(r.value);
    }
    let base = crate::sampling::round_frequency(limit / 4.);
    Ok(vec![base, 2. * base, 4. * base])
}
/// Native field sweep; unusable samples retain their positions as NaN.
pub fn mtf_vs_field_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfVsFieldSettings,
    ctx: &JobContext,
) -> MkResult<a::MtfVsFieldResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&s.field_samples) {
        return Err(MkError::InvalidParam("field sample count".into()));
    }
    let len = s.direction[0].hypot(s.direction[1]);
    a::check_pos("field direction", len)?;
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let o = crate::MtfOrientation {
        tangential: s.direction.map(|v| v / len),
    };
    let hs = a::linspace(0., 1., s.field_samples);
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let service = PupilTracer::new(b, t, ctx)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let mut values = vec![];
    let mut prepared = vec![];
    for &h in &hs {
        let point = mklight_trace::FieldPoint::normalized(h * o.tangential[0], h * o.tangential[1]);
        values.push(a::field_value(t, &point));
        prepared.push(t.prepare_field(point));
    }
    let requests: Vec<_> = prepared
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.as_ref().ok().map(|f| (i, values[i], f)))
        .collect();
    let mut traced = trace_fields(
        &service,
        t,
        &requests,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.6),
        &mut stats,
        &mut warnings,
    )?
    .into_iter();
    let mut per = vec![];
    let mut error = None;
    for (&h, f) in hs.iter().zip(prepared) {
        match f.and_then(|_| traced.next().unwrap()) {
            Ok(f) => per.push(Some(f)),
            Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                warnings.push(format!("field {h:.3} skipped: {e}"));
                error.get_or_insert(e);
                per.push(None);
            }
            Err(e) => return Err(e),
        }
    }
    if per.iter().all(Option::is_none) {
        return Err(error.unwrap_or_else(|| MkError::Trace("no usable field".into())));
    }
    let freqs = fixed_frequencies(
        b,
        t,
        per.iter().flatten(),
        &s.frequencies,
        &ctx.subrange(0.6, 0.65),
        &mut stats,
    )?;
    let mut curves: Vec<_> = freqs
        .iter()
        .map(|&frequency| a::MtfFieldCurve {
            frequency,
            tangential: vec![f64::NAN; hs.len()],
            sagittal: vec![f64::NAN; hs.len()],
        })
        .collect();
    let detector = s.detector.map(|d| d.curve(&freqs, &o));
    for (i, f) in per.iter().enumerate() {
        if let Some(f) = f {
            let c = curve(
                b,
                t,
                &f.weighted(),
                &freqs,
                o,
                s.method,
                s.huygens,
                &ctx.subrange(
                    0.65 + 0.35 * i as f64 / hs.len() as f64,
                    0.65 + 0.35 * (i + 1) as f64 / hs.len() as f64,
                ),
                &mut stats,
            )?;
            for (j, row) in curves.iter_mut().enumerate() {
                row.tangential[i] =
                    c.tangential[j] * detector.as_ref().map_or(1., |d| d.tangential[j]);
                row.sagittal[i] = c.sagittal[j] * detector.as_ref().map_or(1., |d| d.sagittal[j]);
            }
            warnings.extend(c.warnings.iter().map(ToString::to_string));
        }
    }
    ctx.report(1., "MTF vs field");
    Ok(a::MtfVsFieldResult {
        pupil_sampling: per
            .iter()
            .map(|f| f.as_ref().map_or_else(Vec::new, Field::sampling))
            .collect(),
        normalized_field: hs,
        field_values: values,
        frequencies: freqs,
        curves,
        run_info: info(b, t, timer, "mtf_vs_field", s, &stats, warnings),
    })
}
#[allow(clippy::too_many_arguments)]
pub(super) fn image_grid(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    field: &Field,
    requested: usize,
    spacing: f64,
    method: a::PsfMethod,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<usize> {
    if requested != 0 {
        return Ok(requested);
    }
    let mut side = 0f64;
    for (_, _, p) in &field.pupils {
        let r = p.pupil.blur(b, t.system().snapshot.clone(), ctx)?;
        accumulate(stats, &r.stats);
        side = side.max(2. * (r.value.extent + 16. * p.pupil.lambda_f()));
    }
    Ok(((side / spacing).ceil() as usize).clamp(
        64,
        if method == a::PsfMethod::Huygens {
            1024
        } else {
            2048
        },
    ))
}
pub(super) fn ensure_window(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    field: &mut Field,
    window: f64,
    s: &PupilSettings,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<()> {
    if s.grid == 0 {
        for (_, _, p) in &mut field.pupils {
            let n = ((window / p.pupil.lambda_f())
                .max(p.pupil.n as f64)
                .min(1024.)
                .ceil() as usize)
                .next_power_of_two();
            if n != p.pupil.n {
                let r = p.resample(b, t.system().snapshot.clone(), n, s.supersample, ctx)?;
                accumulate(stats, &r.stats);
            }
        }
    }
    Ok(())
}
/// Native line-spread profiles from the resident polychromatic PSF.
pub fn line_spread_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::LineSpreadSettings,
    ctx: &JobContext,
) -> MkResult<a::LineSpreadResult> {
    let timer = RunTimer::start();
    a::energy::check_grid(s.grid_size, s.padding, s.method)?;
    a::check_pos("padding", s.padding)?;
    if s.grid_size > 4096 {
        return Err(MkError::InvalidParam("PSF grid".into()));
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let per = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
        &mut stats,
        &mut warnings,
    )?;
    let total = per.len();
    let mut out = vec![];
    for (i, mut f) in per.into_iter().enumerate() {
        let sub = ctx.subrange(
            0.5 + 0.5 * i as f64 / total as f64,
            0.5 + 0.5 * (i + 1) as f64 / total as f64,
        );
        let dx = f
            .pupils
            .iter()
            .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
            .fold(f64::INFINITY, f64::min);
        let grid = image_grid(
            b,
            t,
            &f,
            s.grid_size,
            dx,
            s.method,
            &sub.subrange(0., 0.1),
            &mut stats,
        )?;
        ensure_window(
            b,
            t,
            &mut f,
            grid as f64 * dx,
            &s.pupil,
            &sub.subrange(0.1, 0.2),
            &mut stats,
        )?;
        let r = polychromatic_psf(
            b,
            t.system().snapshot.clone(),
            &f.weighted(),
            &crate::PolyPsfSettings {
                grid_size: grid,
                spacing_mm: dx,
                normalization: PsfNormalization::UnitEnergy,
                transform: PsfTransform::Matrix,
            },
            s.method == a::PsfMethod::Huygens,
            &sub.subrange(0.2, 0.9),
        )?;
        accumulate(&mut stats, &r.stats);
        warnings.extend(r.value.warnings.iter().map(ToString::to_string));
        let r = r
            .value
            .line_spread(b, t.system().snapshot.clone(), &sub.subrange(0.9, 1.))?;
        accumulate(&mut stats, &r.stats);
        let [x, y] = r.value;
        out.push(a::FieldLineSpread {
            field: f.index,
            field_value: f.value,
            pupil_sampling: f.sampling(),
            positions_mm: (0..grid)
                .map(|i| (i as f64 - (grid / 2) as f64) * dx)
                .collect(),
            x,
            y,
        });
    }
    ctx.report(1., "line spread");
    Ok(a::LineSpreadResult {
        fields: out,
        run_info: info(b, t, timer, "line_spread", s, &stats, warnings),
    })
}
/// Native band-limited diffraction energy, preserving dense quantile sampling.
pub fn diffraction_encircled_energy_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::EncircledEnergySettings,
    ctx: &JobContext,
) -> MkResult<a::WaveEncircledEnergyResult> {
    let timer = RunTimer::start();
    a::energy::check_grid(s.grid_size, s.padding, s.method)?;
    a::check_pos("padding", s.padding)?;
    if !(2..=10001).contains(&s.samples) || s.grid_size > 4096 {
        return Err(MkError::InvalidParam("energy sampling".into()));
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let mut per = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut stats,
        &mut warnings,
    )?;
    let spacing = per
        .iter()
        .flat_map(|f| &f.pupils)
        .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
        .fold(f64::INFINITY, f64::min);
    let mut grid = s.grid_size;
    for f in &per {
        grid = grid.max(image_grid(
            b,
            t,
            f,
            s.grid_size,
            spacing,
            s.method,
            &ctx.subrange(0.3, 0.35),
            &mut stats,
        )?);
    }
    let max = s.max_radius_mm.unwrap_or(0.5 * grid as f64 * spacing);
    a::check_pos("maximum radius", max)?;
    let radius = a::linspace(0., max, s.samples);
    let dense = a::linspace(0., max, 8 * s.samples);
    let mut all = radius.clone();
    all.extend_from_slice(&dense);
    let total = per.len();
    let mut pending = vec![];
    for (i, f) in per.iter_mut().enumerate() {
        let sub = ctx.subrange(
            0.35 + 0.45 * i as f64 / total as f64,
            0.35 + 0.45 * (i + 1) as f64 / total as f64,
        );
        ensure_window(
            b,
            t,
            f,
            grid as f64 * spacing,
            &s.pupil,
            &sub.subrange(0., 0.05),
            &mut stats,
        )?;
        let settings = crate::PolyPsfSettings {
            grid_size: grid,
            spacing_mm: spacing,
            normalization: PsfNormalization::UnitEnergy,
            transform: PsfTransform::Matrix,
        };
        let r = polychromatic_psf(
            b,
            t.system().snapshot.clone(),
            &f.weighted(),
            &settings,
            s.method == a::PsfMethod::Huygens,
            &sub.subrange(0.05, 0.3),
        )?;
        accumulate(&mut stats, &r.stats);
        let psf = r.value;
        let metrics = psf.metrics(b, t.system().snapshot.clone(), &sub.subrange(0.3, 0.32))?;
        accumulate(&mut stats, &metrics.stats);
        let center_mm = match s.center {
            crate::EnergyCenter::Reference => [0.; 2],
            crate::EnergyCenter::Centroid => metrics.value.centroid_mm,
            crate::EnergyCenter::Peak => [
                ((metrics.value.peak_index % grid) as f64 - (grid / 2) as f64) * spacing,
                ((metrics.value.peak_index / grid) as f64 - (grid / 2) as f64) * spacing,
            ],
            crate::EnergyCenter::Point { x, y } => [x, y],
        };
        let center = crate::EnergyCenter::Point {
            x: center_mm[0],
            y: center_mm[1],
        };
        let sq = psf.energy(
            b,
            t.system().snapshot.clone(),
            &radius,
            center,
            true,
            &sub.subrange(0.32, 0.42),
        )?;
        accumulate(&mut stats, &sq.stats);
        let clean: Vec<_> = f
            .pupils
            .iter()
            .map(|(_, w, p)| Ok((p.pupil.without_aberrations(b, ctx)?, *w)))
            .collect::<MkResult<_>>()?;
        let clean: Vec<_> = clean.iter().map(|(p, w)| (p, *w)).collect();
        let dl = polychromatic_psf(
            b,
            t.system().snapshot.clone(),
            &clean,
            &settings,
            false,
            &sub.subrange(0.7, 0.85),
        )?;
        accumulate(&mut stats, &dl.stats);
        if metrics.value.energy_in_grid < a::MIN_ENERGY_IN_GRID {
            warnings.push(
                WaveWarning::PsfTruncated {
                    energy_in_grid: metrics.value.energy_in_grid,
                }
                .to_string(),
            );
        }
        warnings.extend(psf.warnings.iter().map(ToString::to_string));
        pending.push((
            f.index,
            f.value,
            f.sampling(),
            center_mm,
            sq.value,
            psf,
            dl.value,
        ));
    }
    let psfs: Vec<_> = pending
        .iter()
        .flat_map(|(_, _, _, center, _, psf, dl)| {
            [
                (
                    psf,
                    crate::EnergyCenter::Point {
                        x: center[0],
                        y: center[1],
                    },
                ),
                (dl, crate::EnergyCenter::Centroid),
            ]
        })
        .collect();
    let curves = encircled_energy_batch(
        b,
        t.system().snapshot.clone(),
        &psfs,
        &all,
        &ctx.subrange(0.8, 1.),
    )?;
    accumulate(&mut stats, &curves.stats);
    let out = pending
        .into_iter()
        .zip(curves.value.chunks(2))
        .map(
            |((field, field_value, pupil_sampling, center_mm, ensquared, _, _), energy)| {
                let (enc, fine) = energy[0].split_at(radius.len());
                a::FieldEnergy {
                    field,
                    field_value,
                    pupil_sampling,
                    grid_size: grid,
                    spacing_mm: spacing,
                    center_mm,
                    encircled: enc.to_vec(),
                    ensquared,
                    diffraction_limit: energy[1][..radius.len()].to_vec(),
                    r50_mm: a::energy::quantile(&dense, fine, 0.5),
                    r80_mm: a::energy::quantile(&dense, fine, 0.8),
                    r90_mm: a::energy::quantile(&dense, fine, 0.9),
                }
            },
        )
        .collect();
    ctx.report(1., "encircled energy");
    Ok(a::WaveEncircledEnergyResult {
        radius_mm: radius,
        fields: out,
        run_info: info(
            b,
            t,
            timer,
            "diffraction_encircled_energy",
            s,
            &stats,
            warnings,
        ),
    })
}
/// Native focus sweeps with shared correlation stencils and bounded FFT/Huygens batches.
pub fn through_focus_mtf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::ThroughFocusSettings,
    ctx: &JobContext,
) -> MkResult<a::ThroughFocusResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&s.focus_samples) {
        return Err(MkError::InvalidParam("focus samples".into()));
    }
    if let Some(r) = s.focus_range_mm {
        a::check_pos("focus range", r)?;
    }
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let per = fields(
        b,
        t,
        &idx,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut stats,
        &mut warnings,
    )?;
    let frequencies = fixed_frequencies(
        b,
        t,
        per.iter(),
        &s.frequencies,
        &ctx.subrange(0.3, 0.32),
        &mut stats,
    )?;
    let half = if let Some(r) = s.focus_range_mm {
        r
    } else {
        let mut half = 0f64;
        for f in &per {
            for (_, _, p) in &f.pupils {
                let r = p
                    .pupil
                    .blur(b, t.system().snapshot.clone(), &ctx.subrange(0.32, 0.35))?;
                accumulate(&mut stats, &r.stats);
                let n = p.pupil.geometry.working_f_number();
                half =
                    half.max((8. * p.pupil.wavelength_um * 1e-3 * n * n).max(4. * n * r.value.rms));
            }
        }
        half
    };
    let focus = if s.focus_samples == 1 {
        vec![0.]
    } else {
        a::linspace(-half, half, s.focus_samples)
    };
    let count = per.len();
    let mut out = vec![];
    for (i, mut field) in per.into_iter().enumerate() {
        let sub = ctx.subrange(
            0.35 + 0.65 * i as f64 / count as f64,
            0.35 + 0.65 * (i + 1) as f64 / count as f64,
        );
        if s.pupil.grid == 0 {
            for (_, _, p) in &mut field.pupils {
                let shifts = [
                    -half / p.pupil.geometry.chief_ray[2],
                    half / p.pupil.geometry.chief_ray[2],
                ];
                let r = p.pupil.focus_batch(
                    b,
                    t.system().snapshot.clone(),
                    &shifts,
                    &sub.subrange(0., 0.05),
                )?;
                accumulate(&mut stats, &r.stats);
                let step = r
                    .value
                    .iter()
                    .map(|p| p.max_phase_step)
                    .fold(0f64, f64::max);
                let n = ((p.pupil.n as f64 * step / 0.35)
                    .max(p.pupil.n as f64)
                    .min(1024.)
                    .ceil() as usize)
                    .next_power_of_two();
                if n != p.pupil.n {
                    let r = p.resample(
                        b,
                        t.system().snapshot.clone(),
                        n,
                        s.pupil.supersample,
                        &sub.subrange(0.1, 0.2),
                    )?;
                    accumulate(&mut stats, &r.stats);
                    warnings.extend(r.run_info.warnings);
                }
            }
        }
        let point = t.field(field.index).point;
        let o = crate::MtfOrientation::from_field(point.hx, point.hy);
        let vectors: Vec<_> = [o.tangential, o.sagittal()]
            .iter()
            .flat_map(|v| frequencies.iter().map(move |f| [v[0] * f, v[1] * f]))
            .collect();
        let r = focus_otf(
            b,
            t.system().snapshot.clone(),
            &field.weighted(),
            &focus,
            &vectors,
            s.method,
            s.huygens,
            &sub.subrange(0.2, 0.98),
        )?;
        accumulate(&mut stats, &r.stats);
        warnings.extend(r.run_info.warnings);
        let detector = s.detector.map(|d| d.curve(&frequencies, &o));
        let nf = frequencies.len();
        let curves = frequencies
            .iter()
            .enumerate()
            .map(|(q, &frequency)| a::ThroughFocusCurve {
                frequency,
                tangential: (0..focus.len())
                    .map(|z| {
                        let v = r.value[z * 2 * nf + q];
                        v[0].hypot(v[1]) * detector.as_ref().map_or(1., |d| d.tangential[q])
                    })
                    .collect(),
                sagittal: (0..focus.len())
                    .map(|z| {
                        let v = r.value[z * 2 * nf + nf + q];
                        v[0].hypot(v[1]) * detector.as_ref().map_or(1., |d| d.sagittal[q])
                    })
                    .collect(),
            })
            .collect();
        out.push(a::ThroughFocusField {
            field: field.index,
            field_value: field.value,
            pupil_sampling: field.sampling(),
            curves,
        });
        sub.report(1., "focus result");
    }
    ctx.report(1., "through-focus MTF");
    Ok(a::ThroughFocusResult {
        focus_mm: focus,
        frequencies,
        fields: out,
        run_info: info(b, t, timer, "through_focus_mtf", s, &stats, warnings),
    })
}
