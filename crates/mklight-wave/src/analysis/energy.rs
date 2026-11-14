//! Diffraction encircled/ensquared energy and line-spread functions, from
//! the (polychromatic) PSF of [`super::psf`].
//!
//! Energies are integrated analytically through the band-limited
//! interpolant of the PSF samples ([`crate::energy`]) and are fractions of
//! the energy in the whole image plane (light outside the grid counts as
//! outside every circle). The diffraction-limited reference is the same
//! computation for the same pupils without aberrations. Line spreads are
//! `LSF_x(x) = ∫PSF dy` and `LSF_y(y) = ∫PSF dx` on the image axes (for
//! fields on the `y` axis, `LSF_y` is the tangential and `LSF_x` the
//! sagittal profile).

use mklight_analysis::util::{FieldSelection, WavelengthSelection, nf};
use mklight_core::job::RunTimer;
use mklight_core::memo::Memo;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::psf::{PsfMethod, check_huygens_grid, compute_psf, image_grid};
use super::{
    ItemProgress, check_pos, fields, linspace, run_info, system_field_pupils, wavelengths,
};
use crate::energy::{
    EnergyCenter, LsfAxis, encircled_energy_batch, ensquared_energy, line_spread_function,
};
use crate::psf::PsfNormalization;
use crate::pupil::PupilFunction;
use crate::traced::PupilSettings;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EncircledEnergySettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: PsfMethod,
    pub grid_size: usize,
    /// PSF spacing `λ_min·F#/padding`.
    pub padding: f64,
    pub center: EnergyCenter,
    /// Largest radius (half-width) in mm; half the PSF grid when `None`.
    pub max_radius_mm: Option<f64>,
    pub samples: usize,
}

impl Default for EncircledEnergySettings {
    fn default() -> Self {
        EncircledEnergySettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: PsfMethod::Fft,
            grid_size: 0,
            padding: 2.5,
            center: EnergyCenter::Centroid,
            max_radius_mm: None,
            samples: 51,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldEnergy {
    pub grid_size: usize,
    pub spacing_mm: f64,
    pub pupil_sampling: Vec<super::PupilSampling>,
    pub field: usize,
    pub field_value: [f64; 2],
    /// Centre of the integration relative to the field's reference point
    /// (mm).
    pub center_mm: [f64; 2],
    pub encircled: Vec<f64>,
    pub ensquared: Vec<f64>,
    /// Encircled energy of the same pupils without aberrations.
    pub diffraction_limit: Vec<f64>,
    /// Radii holding 50/80/90 % of the energy (`null` beyond the range).
    #[serde(with = "nf")]
    pub r50_mm: f64,
    #[serde(with = "nf")]
    pub r80_mm: f64,
    #[serde(with = "nf")]
    pub r90_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaveEncircledEnergyResult {
    /// Radii (encircled) and half-widths (ensquared), mm.
    pub radius_mm: Vec<f64>,
    pub fields: Vec<FieldEnergy>,
    /// Units of the image-space quantities ([`super::WaveUnits`]).
    pub units: super::WaveUnits,
    pub run_info: RunInfo,
}

pub fn diffraction_encircled_energy(
    tracer: &SequentialTracer,
    settings: &EncircledEnergySettings,
) -> MkResult<WaveEncircledEnergyResult> {
    diffraction_encircled_energy_job(tracer, settings, &JobContext::new())
}

/// First radius at which the increasing curve `e(r)` reaches `level`
/// (linear interpolation), NaN if never.
pub(crate) fn quantile(r: &[f64], e: &[f64], level: f64) -> f64 {
    for k in 1..r.len() {
        if e[k] >= level {
            let (a, b) = (e[k - 1], e[k]);
            let t = if b > a { (level - a) / (b - a) } else { 0.0 };
            return r[k - 1] + t.clamp(0.0, 1.0) * (r[k] - r[k - 1]);
        }
    }
    f64::NAN
}

pub(crate) fn check_grid(grid: usize, padding: f64, method: PsfMethod) -> MkResult<()> {
    if grid != 0 && !(8..=4096).contains(&grid) {
        return Err(MkError::InvalidParam(format!("PSF grid size {grid}")));
    }
    if method == PsfMethod::Huygens && grid != 0 {
        check_huygens_grid(grid)?;
    }
    check_pos("padding", padding)
}

pub fn diffraction_encircled_energy_job(
    tracer: &SequentialTracer,
    settings: &EncircledEnergySettings,
    ctx: &JobContext,
) -> MkResult<WaveEncircledEnergyResult> {
    diffraction_encircled_energy_shared(tracer, settings, None, ctx)
}

/// [`diffraction_encircled_energy_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn diffraction_encircled_energy_shared(
    tracer: &SequentialTracer,
    settings: &EncircledEnergySettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<WaveEncircledEnergyResult> {
    let timer = RunTimer::start();
    check_grid(settings.grid_size, settings.padding, settings.method)?;
    if !(2..=10_001).contains(&settings.samples) {
        return Err(MkError::InvalidParam(format!(
            "radius samples {}",
            settings.samples
        )));
    }
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) = system_field_pupils(
        tracer,
        &idx,
        &wls,
        &settings.pupil,
        memo,
        &ctx.subrange(0.0, 0.3),
    )?;
    // One radius list for every field: the finest PSF scale.
    let lf = per
        .iter()
        .map(|(_, _, fp)| fp.min_lambda_f())
        .fold(f64::INFINITY, f64::min);
    let spacing = lf / settings.padding;
    // One window too: the largest automatic one over the fields.
    let grid = per
        .par_iter()
        .map(|(_, _, fp)| image_grid(settings.grid_size, settings.method, &fp.weighted(), spacing))
        .max()
        .unwrap_or(settings.grid_size);
    let rmax = match settings.max_radius_mm {
        Some(r) => {
            check_pos("maximum radius", r)?;
            r
        }
        None => 0.5 * grid as f64 * spacing,
    };
    let radius = linspace(0.0, rmax, settings.samples);
    let dense = linspace(0.0, rmax, 8 * settings.samples);
    let mut warnings = skipped;
    let progress = ItemProgress::new(ctx.subrange(0.3, 0.8), per.len());
    // Fields are independent: they run side by side and merge in field
    // order (results and warnings do not depend on the scheduling).
    let computed: Vec<MkResult<_>> = per
        .into_par_iter()
        .map(|(f, value, mut fp)| {
            let mut warnings = fp.warnings.clone();
            let sub = progress.item();
            warnings.extend(fp.ensure_window(
                grid as f64 * spacing,
                &settings.pupil,
                memo,
                &sub.subrange(0.0, 0.0),
            )?);
            let pupils = fp.weighted();
            let clean: Vec<PupilFunction> = pupils
                .iter()
                .map(|(p, _)| p.without_aberrations())
                .collect();
            let clean_w: Vec<(&PupilFunction, f64)> = clean
                .iter()
                .zip(&pupils)
                .map(|(p, (_, w))| (p, *w))
                .collect();
            // The aberrated and diffraction-limited PSFs side by side.
            let (psf, dl_psf) = rayon::join(
                || {
                    compute_psf(
                        &pupils,
                        settings.method,
                        grid,
                        spacing,
                        PsfNormalization::UnitEnergy,
                        &sub.subrange(0.0, 0.5),
                    )
                },
                || {
                    compute_psf(
                        &clean_w,
                        PsfMethod::Fft,
                        grid,
                        spacing,
                        PsfNormalization::UnitEnergy,
                        &sub.subrange(0.5, 0.7),
                    )
                },
            );
            let psf = psf?;
            for w in &psf.warnings {
                warnings.push(format!("field {f}: {w}"));
            }
            let dl_psf = dl_psf?;
            let c = settings.center;
            let sq = ensquared_energy(&psf, &radius, c, &sub.subrange(0.7, 1.0))?;
            let center_mm = match c {
                EnergyCenter::Reference => psf.center_mm,
                EnergyCenter::Centroid => psf.centroid(),
                EnergyCenter::Peak => {
                    let (_, i, j) = psf.peak();
                    [psf.x(i), psf.y(j)]
                }
                EnergyCenter::Point { x, y } => [x, y],
            };
            progress.finish();
            Ok((
                warnings,
                (f, value, fp.sampling(), center_mm, psf, dl_psf, sq),
            ))
        })
        .collect();
    let mut pending = Vec::new();
    for c in computed {
        let (w, p) = c?;
        warnings.extend(w);
        pending.push(p);
    }
    let mut all_radii = radius.clone();
    all_radii.extend_from_slice(&dense);
    let psfs: Vec<_> = pending
        .iter()
        .flat_map(|(_, _, _, _, p, dl, _)| [(p, settings.center), (dl, EnergyCenter::Centroid)])
        .collect();
    let energies = encircled_energy_batch(&psfs, &all_radii, &ctx.subrange(0.8, 1.0))?;
    let out = pending
        .into_iter()
        .zip(energies.chunks(2))
        .map(|((f, value, sampling, center_mm, _, _, sq), energy)| {
            let (enc, fine) = energy[0].split_at(radius.len());
            FieldEnergy {
                field: f,
                field_value: value,
                center_mm,
                grid_size: grid,
                spacing_mm: spacing,
                pupil_sampling: sampling,
                encircled: enc.to_vec(),
                ensquared: sq,
                diffraction_limit: energy[1][..radius.len()].to_vec(),
                r50_mm: quantile(&dense, fine, 0.5),
                r80_mm: quantile(&dense, fine, 0.8),
                r90_mm: quantile(&dense, fine, 0.9),
            }
        })
        .collect();
    ctx.report(1.0, "encircled energy");
    Ok(WaveEncircledEnergyResult {
        units: super::WaveUnits::of(tracer),
        radius_mm: radius,
        fields: out,
        run_info: run_info(
            timer,
            tracer,
            "diffraction_encircled_energy",
            settings,
            warnings,
        ),
    })
}

// ----- Line spread -----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LineSpreadSettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: PsfMethod,
    pub grid_size: usize,
    pub padding: f64,
}

impl Default for LineSpreadSettings {
    fn default() -> Self {
        LineSpreadSettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: PsfMethod::Fft,
            grid_size: 0,
            padding: 2.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldLineSpread {
    pub pupil_sampling: Vec<super::PupilSampling>,
    pub field: usize,
    pub field_value: [f64; 2],
    /// Sample positions relative to the field's reference point (mm),
    /// along x for `x` and along y for `y`.
    pub positions_mm: Vec<f64>,
    /// `∫PSF dy` (fraction of the energy per mm) along x.
    pub x: Vec<f64>,
    /// `∫PSF dx` along y.
    pub y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineSpreadResult {
    pub fields: Vec<FieldLineSpread>,
    /// Units of the image-space quantities ([`super::WaveUnits`]).
    pub units: super::WaveUnits,
    pub run_info: RunInfo,
}

pub fn line_spread(
    tracer: &SequentialTracer,
    settings: &LineSpreadSettings,
) -> MkResult<LineSpreadResult> {
    line_spread_job(tracer, settings, &JobContext::new())
}

pub fn line_spread_job(
    tracer: &SequentialTracer,
    settings: &LineSpreadSettings,
    ctx: &JobContext,
) -> MkResult<LineSpreadResult> {
    line_spread_shared(tracer, settings, None, ctx)
}

/// [`line_spread_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn line_spread_shared(
    tracer: &SequentialTracer,
    settings: &LineSpreadSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<LineSpreadResult> {
    let timer = RunTimer::start();
    check_grid(settings.grid_size, settings.padding, settings.method)?;
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) = system_field_pupils(
        tracer,
        &idx,
        &wls,
        &settings.pupil,
        memo,
        &ctx.subrange(0.0, 0.4),
    )?;
    let mut warnings = skipped;
    let mut out = Vec::new();
    let progress = ItemProgress::new(ctx.subrange(0.4, 1.0), per.len());
    // Fields are independent: they run side by side and merge in field
    // order (results and warnings do not depend on the scheduling).
    let computed: Vec<MkResult<_>> = per
        .into_par_iter()
        .map(|(f, value, mut fp)| {
            let mut warnings = fp.warnings.clone();
            let sub = progress.item();
            let spacing = fp.min_lambda_f() / settings.padding;
            let grid = image_grid(settings.grid_size, settings.method, &fp.weighted(), spacing);
            warnings.extend(fp.ensure_window(
                grid as f64 * spacing,
                &settings.pupil,
                memo,
                &sub.subrange(0.0, 0.0),
            )?);
            let psf = compute_psf(
                &fp.weighted(),
                settings.method,
                grid,
                spacing,
                PsfNormalization::UnitEnergy,
                &sub,
            )?;
            for w in &psf.warnings {
                warnings.push(format!("field {f}: {w}"));
            }
            let (lx, ly) = rayon::join(
                || line_spread_function(&psf, LsfAxis::X),
                || line_spread_function(&psf, LsfAxis::Y),
            );
            progress.finish();
            Ok((
                warnings,
                FieldLineSpread {
                    field: f,
                    field_value: value,
                    positions_mm: lx.positions_mm,
                    pupil_sampling: fp.sampling(),
                    x: lx.values,
                    y: ly.values,
                },
            ))
        })
        .collect();
    for c in computed {
        let (w, field) = c?;
        warnings.extend(w);
        out.push(field);
    }
    ctx.report(1.0, "line spread");
    Ok(LineSpreadResult {
        units: super::WaveUnits::of(tracer),
        fields: out,
        run_info: run_info(timer, tracer, "line_spread", settings, warnings),
    })
}
