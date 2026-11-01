//! Native result adapters. Engine routing and qualification policy remain in P11.
use super::traced::accumulate;
use super::*;
use crate::analysis::{self as a, PupilSampling};
use crate::{PupilSettings, ReferencePoint, ZernikeBasis};
use mklight_core::job::RunTimer;
use mklight_trace::SequentialTracer;
use serde::Serialize;
struct Field {
    index: usize,
    value: [f64; 2],
    reference: [f64; 2],
    pupils: Vec<(usize, f64, ResidentTracedPupil)>,
}
impl Field {
    fn weighted(&self) -> Vec<(&ResidentPupil, f64)> {
        self.pupils.iter().map(|(_, w, p)| (&p.pupil, *w)).collect()
    }
    fn sampling(&self) -> Vec<PupilSampling> {
        self.pupils
            .iter()
            .map(|(wl, _, p)| PupilSampling {
                wavelength: *wl,
                grid: p.pupil.n,
            })
            .collect()
    }
}
fn info(
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
    let mut fields = vec![];
    let mut first_error = None;
    for (i, &index) in indices.iter().enumerate() {
        let field = t.field(index);
        let mut pupils = vec![];
        let mut failed = false;
        for (j, &(wl, weight)) in wls.iter().enumerate() {
            let part = ctx.subrange(
                (i * wls.len() + j) as f64 / (indices.len() * wls.len()) as f64,
                (i * wls.len() + j + 1) as f64 / (indices.len() * wls.len()) as f64,
            );
            match service.trace(field, wl, settings, &part) {
                Ok(r) => {
                    accumulate(stats, &r.stats);
                    warnings.extend(r.run_info.warnings);
                    pupils.push((wl, weight, r.value));
                }
                Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                    warnings.push(format!("field {index} skipped: {e}"));
                    first_error.get_or_insert(e);
                    failed = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        if failed {
            continue;
        }
        let reference = match settings.reference {
            ReferencePoint::Chief => {
                if let Some((_, _, p)) = pupils
                    .iter()
                    .find(|(wl, _, _)| *wl == t.system().primary_wavelength)
                {
                    [p.info.chief_point_mm[0], p.info.chief_point_mm[1]]
                } else {
                    let r = service.trace(
                        field,
                        t.system().primary_wavelength,
                        settings,
                        &ctx.subrange(1., 1.),
                    )?;
                    accumulate(stats, &r.stats);
                    [
                        r.value.info.chief_point_mm[0],
                        r.value.info.chief_point_mm[1],
                    ]
                }
            }
            ReferencePoint::Centroid => {
                let mut v = [0.; 2];
                for (_, w, p) in &pupils {
                    for (j, x) in v.iter_mut().enumerate() {
                        *x += w * p.info.reference_point_mm[j];
                    }
                }
                v
            }
        };
        for (_, _, p) in &mut pupils {
            p.pupil.image_offset_mm = [
                p.info.reference_point_mm[0] - reference[0],
                p.info.reference_point_mm[1] - reference[1],
            ];
        }
        let f = &t.system().system().fields.fields[index];
        fields.push(Field {
            index,
            value: [f.x, f.y],
            reference,
            pupils,
        });
    }
    if fields.is_empty() {
        if let Some(e) = first_error {
            return Err(e);
        }
    }
    Ok(fields)
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
    for f in fields {
        for (wl, _, p) in f.pupils {
            let sub = ctx.subrange(
                0.7 + 0.3 * maps.len() as f64 / total as f64,
                0.7 + 0.3 * (maps.len() + 1) as f64 / total as f64,
            );
            let fit = fit_resident_zernike(
                b,
                t.system().snapshot.clone(),
                p.nodes.view(),
                &basis,
                &sub.subrange(0., 0.5),
            )?;
            accumulate(&mut stats, &fit.stats);
            let st = p
                .pupil
                .statistics(b, t.system().snapshot.clone(), &sub.subrange(0.5, 0.9))?;
            accumulate(&mut stats, &st.stats);
            let v = st.value;
            let rp = &p.pupil;
            let opd = b.read_view(rp.opd.view())?;
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
            let energy = psf.energy_in_grid() / psf.total_energy;
            if energy < a::MIN_ENERGY_IN_GRID {
                crate::warnings::push_unique(
                    &mut psf.warnings,
                    WaveWarning::PsfTruncated {
                        energy_in_grid: energy,
                    },
                );
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
