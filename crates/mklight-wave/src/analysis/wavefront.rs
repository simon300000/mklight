//! Wavefront map: the traced OPD on the exit-pupil reference sphere,
//! resampled on the pupil grid, with RMS/PV statistics, Strehl estimates
//! and a Zernike fit.
//!
//! The Zernike fit is a weighted least-squares fit to the **lattice rays**
//! themselves (their pupil positions and OPDs, weights `A²`), not to the
//! resampled grid, so it carries no resampling error. Coefficients are in
//! waves over the normalised pupil `(u, v)` of [`crate::traced`]: the unit
//! circle is the pupil scale `sinθ'max` (the full aperture of an
//! unvignetted beam), `u` along image `x`, `v` along image `y`.

use mklight_analysis::util::{FieldSelection, WavelengthSelection, nf};
use mklight_core::job::RunTimer;
use mklight_core::memo::Memo;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{fields, run_info, system_field_pupils, wavelengths};
use crate::energy::strehl_marechal;
use crate::traced::{ExitPupilInfo, PupilSettings, TracedPupil};
use crate::zernike::{
    SeidelTerms, ZernikeBasis, ZernikeFitSettings, ZernikeOrdering, ZernikeSet, fit_zernike,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WavefrontSettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub zernike_ordering: ZernikeOrdering,
    /// Number of Zernike terms (at most 37 for Fringe and
    /// [`MAX_ZERNIKE_TERMS`] for Standard).
    pub zernike_terms: usize,
}

/// Largest number of fitted Zernike terms: the Standard (Noll) terms through
/// radial order 20. The fit holds `rays × terms` values and costs
/// `rays × terms²` operations (about 2·10⁸ at the default lattice).
pub const MAX_ZERNIKE_TERMS: usize = 231;

/// The fit basis of `settings`.
pub(crate) fn fit_basis(settings: &WavefrontSettings) -> MkResult<ZernikeBasis> {
    if settings.zernike_terms > MAX_ZERNIKE_TERMS {
        return Err(MkError::InvalidParam(format!(
            "{} Zernike terms requested; at most {MAX_ZERNIKE_TERMS} are fitted",
            settings.zernike_terms
        )));
    }
    ZernikeBasis::new(settings.zernike_ordering, settings.zernike_terms)
}

impl Default for WavefrontSettings {
    fn default() -> Self {
        WavefrontSettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::Primary,
            pupil: PupilSettings::default(),
            zernike_ordering: ZernikeOrdering::Fringe,
            zernike_terms: 37,
        }
    }
}

/// The wavefront of one (field, wavelength).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WavefrontMap {
    pub field: usize,
    /// Field value in the system's field units.
    pub field_value: [f64; 2],
    pub wavelength: usize,
    pub wavelength_um: f64,
    pub exit_pupil: ExitPupilInfo,
    /// Grid side `n`.
    pub grid: usize,
    /// Sample coordinates (both axes) in normalised pupil units.
    pub coordinates: Vec<f64>,
    /// OPD (waves), row-major `j·n + i` (`i` along `u`), `null` outside the
    /// clear aperture. The chief ray (pupil centre) has OPD 0.
    #[serde(with = "nf::vec")]
    pub opd_waves: Vec<f64>,
    /// Relative amplitude on the reference sphere (0 outside).
    pub amplitude: Vec<f64>,
    /// Weighted RMS and PV about the mean (piston removed).
    pub rms_waves: f64,
    pub pv_waves: f64,
    /// RMS and PV with the best-fit tilt removed too.
    pub rms_waves_tilt_removed: f64,
    pub pv_waves_tilt_removed: f64,
    /// Strehl ratio at the reference point (exact for the sampled pupil).
    pub strehl_at_reference: f64,
    /// `exp(−(2πσ)²)` from the tilt-removed RMS (peak Strehl estimate).
    pub strehl_marechal: f64,
    pub zernike: ZernikeSet,
    /// RMS of the fit residual over the rays (waves).
    pub zernike_residual_rms: f64,
    pub zernike_condition: f64,
    /// Seidel-equivalent terms from Fringe Z1–Z9 (waves at the pupil edge).
    pub seidel: SeidelTerms,
    /// Clear-aperture rim loops in normalised pupil coordinates (empty
    /// without rim tracing).
    pub rim: Vec<Vec<[f64; 2]>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WavefrontResult {
    pub maps: Vec<WavefrontMap>,
    /// Units of the image-space quantities ([`super::WaveUnits`]).
    pub units: super::WaveUnits,
    pub run_info: RunInfo,
}

impl WavefrontResult {
    pub fn map(&self, field: usize, wavelength: usize) -> Option<&WavefrontMap> {
        self.maps
            .iter()
            .find(|m| m.field == field && m.wavelength == wavelength)
    }
}

pub fn wavefront_map(
    tracer: &SequentialTracer,
    settings: &WavefrontSettings,
) -> MkResult<WavefrontResult> {
    wavefront_map_job(tracer, settings, &JobContext::new())
}

/// The Zernike fit of a traced pupil's lattice rays (module docs).
pub fn fit_traced(
    tp: &TracedPupil,
    basis: &ZernikeBasis,
    ctx: &JobContext,
) -> MkResult<crate::zernike::ZernikeFit> {
    fit_samples(&tp.samples, basis, ctx)
}

/// The Zernike fit of lattice rays (weights `A²`), as [`fit_traced`].
pub(crate) fn fit_samples(
    samples: &crate::StructuredSamples,
    basis: &ZernikeBasis,
    ctx: &JobContext,
) -> MkResult<crate::zernike::ZernikeFit> {
    let rays: Vec<_> = samples
        .samples
        .iter()
        .flatten()
        .filter(|r| r.amplitude > 0.0)
        .collect();
    let pts: Vec<(f64, f64)> = rays.iter().map(|r| (r.u, r.v)).collect();
    let vals: Vec<f64> = rays.iter().map(|r| r.opd_waves).collect();
    let w: Vec<f64> = rays.iter().map(|r| r.amplitude * r.amplitude).collect();
    fit_zernike(
        &pts,
        &vals,
        Some(&w),
        basis,
        &ZernikeFitSettings::default(),
        ctx,
    )
}

/// One pupil's outcome in [`wavefront_map_shared`].
enum Fitted {
    /// The map and its warnings.
    Map(Vec<String>, Box<WavefrontMap>),
    /// Too few rays for the fit: the warning and the error.
    Skipped(String, MkError),
}

pub fn wavefront_map_job(
    tracer: &SequentialTracer,
    settings: &WavefrontSettings,
    ctx: &JobContext,
) -> MkResult<WavefrontResult> {
    wavefront_map_shared(tracer, settings, None, ctx)
}

/// [`wavefront_map_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn wavefront_map_shared(
    tracer: &SequentialTracer,
    settings: &WavefrontSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<WavefrontResult> {
    let timer = RunTimer::start();
    let basis = fit_basis(settings)?;
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) = system_field_pupils(
        tracer,
        &idx,
        &wls,
        &settings.pupil,
        memo,
        &ctx.subrange(0.0, 0.85),
    )?;
    let mut warnings = skipped;
    let mut maps = Vec::new();
    let mut first_error = None;
    let fit_ctx = ctx.subrange(0.85, 1.0);
    let total = per
        .iter()
        .map(|(_, _, fp)| fp.pupils.len())
        .sum::<usize>()
        .max(1);
    let done = AtomicUsize::new(0);
    // Every pupil is fitted side by side; the outcomes merge in field and
    // wavelength order, so maps, warnings and the first error are those of
    // a serial pass.
    let fitted: Vec<(Vec<String>, Vec<MkResult<Fitted>>)> = per
        .into_par_iter()
        .map(|(f, value, fp)| {
            let outcomes = fp
                .pupils
                .into_par_iter()
                .map(|(wl, _, tp)| -> MkResult<Fitted> {
                    fit_ctx.check()?;
                    // A field with too few surviving rays for the fit is
                    // skipped like an unusable field (`super` module docs).
                    let fit = match fit_traced(&tp, &basis, &fit_ctx) {
                        Ok(fit) => fit,
                        Err(e @ MkError::Numerical(_)) => {
                            let w = format!("field {f}, wavelength {wl} skipped: {e}");
                            return Ok(Fitted::Skipped(w, e));
                        }
                        Err(e) => return Err(e),
                    };
                    let mut warnings: Vec<String> = fit
                        .warnings
                        .iter()
                        .map(|w| format!("field {f}, wavelength {wl}: {w}"))
                        .collect();
                    let p = &tp.pupil;
                    let n = p.n();
                    let opd = p
                        .opd_waves()
                        .iter()
                        .zip(p.coverage())
                        .map(|(&w, &c)| if c > 0.0 { w } else { f64::NAN })
                        .collect();
                    let s0 = p.opd_stats(false);
                    let s1 = p.opd_stats(true);
                    for w in p.sampling_warnings() {
                        warnings.push(format!("field {f}, wavelength {wl}: {w}"));
                    }
                    let rim = match &tp.mask {
                        Some(crate::pupil::PupilMask::Polygons { loops }) => loops.clone(),
                        _ => Vec::new(),
                    };
                    let map = WavefrontMap {
                        field: f,
                        field_value: value,
                        wavelength: wl,
                        wavelength_um: p.wavelength_um(),
                        exit_pupil: tp.info.clone(),
                        grid: n,
                        coordinates: (0..n).map(|i| p.coord(i)).collect(),
                        opd_waves: opd,
                        amplitude: p
                            .amplitude()
                            .iter()
                            .zip(p.coverage())
                            .map(|(a, c)| if *c > 0.0 { *a } else { 0.0 })
                            .collect(),
                        rms_waves: s0.rms,
                        pv_waves: s0.pv,
                        rms_waves_tilt_removed: s1.rms,
                        pv_waves_tilt_removed: s1.pv,
                        strehl_at_reference: p.strehl_at_reference(),
                        strehl_marechal: strehl_marechal(s1.rms),
                        seidel: fit.set.seidel()?,
                        zernike: fit.set,
                        zernike_residual_rms: fit.residual_rms,
                        zernike_condition: fit.condition,
                        rim,
                    };
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    fit_ctx.report(k as f64 / total as f64, "Zernike fit");
                    Ok(Fitted::Map(warnings, Box::new(map)))
                })
                .collect();
            (fp.warnings, outcomes)
        })
        .collect();
    for (field_warnings, outcomes) in fitted {
        warnings.extend(field_warnings);
        for outcome in outcomes {
            match outcome? {
                Fitted::Map(w, map) => {
                    warnings.extend(w);
                    maps.push(*map);
                }
                Fitted::Skipped(w, e) => {
                    warnings.push(w);
                    first_error.get_or_insert(e);
                }
            }
        }
    }
    if maps.is_empty()
        && let Some(e) = first_error
    {
        return Err(e);
    }
    ctx.report(1.0, "wavefront map");
    Ok(WavefrontResult {
        units: super::WaveUnits::of(tracer),
        maps,
        run_info: run_info(timer, tracer, "wavefront_map", settings, warnings),
    })
}
