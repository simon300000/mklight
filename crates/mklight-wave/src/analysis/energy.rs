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
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use serde::{Deserialize, Serialize};

use super::psf::{PsfMethod, check_huygens_grid, compute_psf, image_grid};
use super::{check_pos, fields, linspace, run_info, system_field_pupils, wavelengths};
use crate::energy::{
    EnergyCenter, LsfAxis, encircled_energy, ensquared_energy, line_spread_function,
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
fn quantile(r: &[f64], e: &[f64], level: f64) -> f64 {
    for k in 1..r.len() {
        if e[k] >= level {
            let (a, b) = (e[k - 1], e[k]);
            let t = if b > a { (level - a) / (b - a) } else { 0.0 };
            return r[k - 1] + t.clamp(0.0, 1.0) * (r[k] - r[k - 1]);
        }
    }
    f64::NAN
}

fn check_grid(grid: usize, padding: f64, method: PsfMethod) -> MkResult<()> {
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
    let (per, skipped) =
        system_field_pupils(tracer, &idx, &wls, &settings.pupil, &ctx.subrange(0.0, 0.3))?;
    // One radius list for every field: the finest PSF scale.
    let lf = per
        .iter()
        .map(|(_, _, fp)| fp.min_lambda_f())
        .fold(f64::INFINITY, f64::min);
    let spacing = lf / settings.padding;
    // One window too: the largest automatic one over the fields.
    let grid = per
        .iter()
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
    let mut out = Vec::new();
    let nf = per.len().max(1) as f64;
    for (k, (f, value, mut fp)) in per.into_iter().enumerate() {
        warnings.extend(fp.warnings.iter().cloned());
        let sub = ctx.subrange(0.3 + 0.7 * k as f64 / nf, 0.3 + 0.7 * (k + 1) as f64 / nf);
        warnings.extend(fp.ensure_window(
            grid as f64 * spacing,
            &settings.pupil,
            &sub.subrange(0.0, 0.0),
        )?);
        let pupils = fp.weighted();
        let psf = compute_psf(
            &pupils,
            settings.method,
            grid,
            spacing,
            PsfNormalization::UnitEnergy,
            &sub.subrange(0.0, 0.5),
        )?;
        for w in &psf.warnings {
            warnings.push(format!("field {f}: {w}"));
        }
        let clean: Vec<PupilFunction> = pupils
            .iter()
            .map(|(p, _)| p.without_aberrations())
            .collect();
        let clean_w: Vec<(&PupilFunction, f64)> = clean
            .iter()
            .zip(&pupils)
            .map(|(p, (_, w))| (p, *w))
            .collect();
        let dl_psf = compute_psf(
            &clean_w,
            PsfMethod::Fft,
            grid,
            spacing,
            PsfNormalization::UnitEnergy,
            &sub.subrange(0.5, 0.7),
        )?;
        let c = settings.center;
        let enc = encircled_energy(&psf, &radius, c, &sub.subrange(0.7, 0.8))?;
        let sq = ensquared_energy(&psf, &radius, c, &sub.subrange(0.8, 0.85))?;
        let dl = encircled_energy(
            &dl_psf,
            &radius,
            EnergyCenter::Centroid,
            &sub.subrange(0.85, 0.9),
        )?;
        let fine = encircled_energy(&psf, &dense, c, &sub.subrange(0.9, 1.0))?;
        let center_mm = match c {
            EnergyCenter::Reference => psf.center_mm,
            EnergyCenter::Centroid => psf.centroid(),
            EnergyCenter::Peak => {
                let (_, i, j) = psf.peak();
                [psf.x(i), psf.y(j)]
            }
            EnergyCenter::Point { x, y } => [x, y],
        };
        out.push(FieldEnergy {
            field: f,
            field_value: value,
            center_mm,
            encircled: enc,
            ensquared: sq,
            diffraction_limit: dl,
            r50_mm: quantile(&dense, &fine, 0.5),
            r80_mm: quantile(&dense, &fine, 0.8),
            r90_mm: quantile(&dense, &fine, 0.9),
        });
    }
    ctx.report(1.0, "encircled energy");
    Ok(WaveEncircledEnergyResult {
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
    let timer = RunTimer::start();
    check_grid(settings.grid_size, settings.padding, settings.method)?;
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) =
        system_field_pupils(tracer, &idx, &wls, &settings.pupil, &ctx.subrange(0.0, 0.4))?;
    let mut warnings = skipped;
    let mut out = Vec::new();
    let nf = per.len().max(1) as f64;
    for (k, (f, value, mut fp)) in per.into_iter().enumerate() {
        warnings.extend(fp.warnings.iter().cloned());
        let sub = ctx.subrange(0.4 + 0.6 * k as f64 / nf, 0.4 + 0.6 * (k + 1) as f64 / nf);
        let spacing = fp.min_lambda_f() / settings.padding;
        let grid = image_grid(settings.grid_size, settings.method, &fp.weighted(), spacing);
        warnings.extend(fp.ensure_window(
            grid as f64 * spacing,
            &settings.pupil,
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
        let lx = line_spread_function(&psf, LsfAxis::X);
        let ly = line_spread_function(&psf, LsfAxis::Y);
        out.push(FieldLineSpread {
            field: f,
            field_value: value,
            positions_mm: lx.positions_mm,
            x: lx.values,
            y: ly.values,
        });
    }
    ctx.report(1.0, "line spread");
    Ok(LineSpreadResult {
        fields: out,
        run_info: run_info(timer, tracer, "line_spread", settings, warnings),
    })
}
