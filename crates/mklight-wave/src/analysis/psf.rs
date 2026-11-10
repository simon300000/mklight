//! PSF analysis: FFT (Debye, exact matrix transform for off-axis and
//! polychromatic grids) or direct Huygens integration on the image
//! surface, monochromatic or polychromatic, with the Strehl ratio.
//!
//! PSFs are sampled on the image surface around the field's common
//! reference point ([`super`] module docs); `center_mm` of the returned
//! [`Psf`] is relative to it and `reference_mm` gives it in the image
//! surface's local frame. The default spacing is `λ_min·F#/padding` with
//! the pupil scale's F# ([`crate::traced`]).

use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::job::RunTimer;
use mklight_core::memo::Memo;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    FieldPupils, ItemProgress, check_pos, fields, run_info, system_field_pupils, wavelengths,
};
use crate::psf::{
    HuygensSettings, ImagePlane, PolyPsfSettings, Psf, PsfNormalization, huygens_psf,
    polychromatic_psf,
};
use crate::pupil::PupilFunction;
use crate::traced::PupilSettings;
use crate::warnings::{WaveWarning, push_unique};

/// How the PSF is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsfMethod {
    /// Fourier transform of the pupil (Debye integral).
    #[default]
    Fft,
    /// Direct Huygens–Fresnel integration from the reference sphere to the
    /// image surface (no Debye or projection approximation).
    Huygens,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PsfAnalysisSettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: PsfMethod,
    /// Image samples per side.
    pub grid_size: usize,
    /// Image spacing (mm); `λ_min·F#/padding` when `None`.
    pub spacing_mm: Option<f64>,
    pub padding: f64,
    pub normalization: PsfNormalization,
    /// One PSF per field from all selected wavelengths; otherwise one per
    /// (field, wavelength).
    pub polychromatic: bool,
}

impl Default for PsfAnalysisSettings {
    fn default() -> Self {
        PsfAnalysisSettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: PsfMethod::Fft,
            grid_size: 0,
            spacing_mm: None,
            padding: 2.5,
            normalization: PsfNormalization::Strehl,
            polychromatic: true,
        }
    }
}

/// The PSF of one field (polychromatic) or one (field, wavelength).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldPsf {
    pub pupil_sampling: Vec<super::PupilSampling>,
    pub field: usize,
    pub field_value: [f64; 2],
    /// `None` for a polychromatic PSF.
    pub wavelength: Option<usize>,
    /// The field's common reference point on the image surface (mm, image
    /// local frame); the PSF's `center_mm` is relative to it.
    pub reference_mm: [f64; 2],
    pub psf: Psf,
    /// Peak over the diffraction-limited peak of the same pupils.
    pub strehl: f64,
    /// Intensity centroid and brightest sample relative to the reference
    /// point (mm).
    pub centroid_mm: [f64; 2],
    pub peak_mm: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsfResult {
    pub psfs: Vec<FieldPsf>,
    pub run_info: RunInfo,
}

impl PsfResult {
    pub fn get(&self, field: usize, wavelength: Option<usize>) -> Option<&FieldPsf> {
        self.psfs
            .iter()
            .find(|p| p.field == field && p.wavelength == wavelength)
    }
}

/// Largest image grid side of the Huygens route in the analyses.
pub const MAX_HUYGENS_GRID: usize = 1024;

/// Largest Huygens cost of one analysis PSF: image points × covered pupil
/// samples, summed over the wavelengths (each term is one complex
/// exponential; `2·10¹⁰` is tens of seconds on a desktop).
pub const MAX_HUYGENS_WORK: f64 = 2e10;

/// Check an analysis Huygens image grid side against [`MAX_HUYGENS_GRID`].
pub(crate) fn check_huygens_grid(grid: usize) -> MkResult<()> {
    if (8..=MAX_HUYGENS_GRID).contains(&grid) {
        Ok(())
    } else {
        Err(MkError::InvalidParam(format!(
            "Huygens image grid {grid} out of range 8..={MAX_HUYGENS_GRID}"
        )))
    }
}

/// Check the cost `grid² × Σ covered pupil samples` of a Huygens PSF
/// against [`MAX_HUYGENS_WORK`].
fn check_huygens_work(pupils: &[(&PupilFunction, f64)], grid: usize) -> MkResult<()> {
    let samples: usize = pupils
        .iter()
        .map(|(p, _)| p.coverage().iter().filter(|c| **c > 0.0).count())
        .sum();
    let work = (grid * grid) as f64 * samples as f64;
    if work > MAX_HUYGENS_WORK {
        return Err(MkError::InvalidParam(format!(
            "Huygens PSF too expensive: {grid}² image points × {samples} pupil samples = {work:.2e} > limit {MAX_HUYGENS_WORK:.0e}; reduce the image or pupil grid"
        )));
    }
    Ok(())
}

/// Huygens PSF on the image surface of `pupil`, on a grid centred on the
/// field's common reference point (the pupil's `image_offset_mm` places
/// its reference point relative to it).
pub(crate) fn huygens_on_surface(
    pupil: &PupilFunction,
    grid: usize,
    spacing: f64,
    normalization: PsfNormalization,
    ctx: &JobContext,
) -> MkResult<Psf> {
    let mut plane = ImagePlane::image_surface(pupil.geometry());
    let off = pupil.image_offset_mm();
    plane.origin_mm = plane.x_axis * (-off[0]) + plane.y_axis * (-off[1]);
    let mut psf = huygens_psf(
        pupil,
        &HuygensSettings {
            grid_size: grid,
            spacing_mm: spacing,
            plane,
            normalization,
        },
        ctx,
    )?;
    psf.center_mm = [0.0, 0.0];
    Ok(psf)
}

/// Fraction of the whole-plane PSF energy that must fall inside the image
/// grid; below it a [`WaveWarning::PsfTruncated`] is attached. (An Airy
/// pattern keeps ≈ 97.6 % on a 16 λF# square, 98.8 % on 32 λF#.)
pub const MIN_ENERGY_IN_GRID: f64 = 0.95;

/// The (weighted) PSF of `pupils` on a `grid²` image grid of `spacing`,
/// centred on the common reference point. Warns when less than
/// [`MIN_ENERGY_IN_GRID`] of its energy lies inside the grid.
pub(crate) fn compute_psf(
    pupils: &[(&PupilFunction, f64)],
    method: PsfMethod,
    grid: usize,
    spacing: f64,
    normalization: PsfNormalization,
    ctx: &JobContext,
) -> MkResult<Psf> {
    let mut psf = compute_psf_raw(pupils, method, grid, spacing, normalization, ctx)?;
    let e = psf.energy_in_grid() / psf.total_energy;
    if e.is_finite() && e < MIN_ENERGY_IN_GRID {
        push_unique(
            &mut psf.warnings,
            WaveWarning::PsfTruncated { energy_in_grid: e },
        );
    }
    Ok(psf)
}

/// `grid`, or when zero the automatic window of `pupils` (Huygens-capped).
pub(crate) fn image_grid(
    grid: usize,
    method: PsfMethod,
    pupils: &[(&PupilFunction, f64)],
    spacing: f64,
) -> usize {
    match (grid, method) {
        (0, PsfMethod::Huygens) => {
            crate::sampling::image_grid(pupils, spacing).min(MAX_HUYGENS_GRID)
        }
        (0, PsfMethod::Fft) => crate::sampling::image_grid(pupils, spacing),
        _ => grid,
    }
}

fn compute_psf_raw(
    pupils: &[(&PupilFunction, f64)],
    method: PsfMethod,
    grid: usize,
    spacing: f64,
    normalization: PsfNormalization,
    ctx: &JobContext,
) -> MkResult<Psf> {
    if pupils.is_empty() {
        return Err(MkError::InvalidParam("no pupil for the PSF".into()));
    }
    match method {
        PsfMethod::Fft => {
            // The exact matrix transform evaluates every wavelength on the
            // requested grid around the common reference (no FFT-length
            // rounding), so all PSFs of an analysis share one grid.
            let mut psf = polychromatic_psf(
                pupils,
                &PolyPsfSettings {
                    grid_size: grid,
                    spacing_mm: spacing,
                    normalization,
                },
                ctx,
            )?;
            if pupils.len() == 1 {
                psf.wavelength_um = Some(pupils[0].0.wavelength_um());
            }
            Ok(psf)
        }
        PsfMethod::Huygens => {
            check_huygens_grid(grid)?;
            check_huygens_work(pupils, grid)?;
            if pupils.len() == 1 {
                return huygens_on_surface(pupils[0].0, grid, spacing, normalization, ctx);
            }
            let wsum: f64 = pupils.iter().map(|(_, w)| *w).sum();
            if !(wsum > 0.0) {
                return Err(MkError::InvalidParam(
                    "polychromatic PSF needs positive weights".into(),
                ));
            }
            let mut acc = vec![0.0; grid * grid];
            let mut dl = 0.0;
            let mut warnings = Vec::new();
            let np = pupils.len() as f64;
            for (q, (p, w)) in pupils.iter().enumerate() {
                let sub = ctx.subrange(q as f64 / np, (q + 1) as f64 / np);
                let one = huygens_on_surface(p, grid, spacing, PsfNormalization::UnitEnergy, &sub)?;
                let wn = w / wsum;
                for (a, v) in acc.iter_mut().zip(&one.data) {
                    *a += wn * v;
                }
                dl += wn * one.diffraction_peak;
                for x in one.warnings {
                    push_unique(&mut warnings, x);
                }
            }
            let (data, total_energy, diffraction_peak) = match normalization {
                PsfNormalization::UnitEnergy => (acc, 1.0, dl),
                PsfNormalization::Strehl => (acc.iter().map(|v| v / dl).collect(), 1.0 / dl, 1.0),
            };
            Ok(Psf {
                size: grid,
                spacing_mm: spacing,
                center_mm: [0.0, 0.0],
                data,
                normalization,
                total_energy,
                diffraction_peak,
                wavelength_um: None,
                warnings,
            })
        }
    }
}

/// Default image spacing `λ_min·F#/padding` of a field's pupils.
pub(crate) fn default_spacing(
    fp: &FieldPupils,
    spacing: Option<f64>,
    padding: f64,
) -> MkResult<f64> {
    match spacing {
        Some(s) => {
            check_pos("PSF spacing", s)?;
            Ok(s)
        }
        None => {
            check_pos("padding", padding)?;
            Ok(fp.min_lambda_f() / padding)
        }
    }
}

pub fn psf_analysis(
    tracer: &SequentialTracer,
    settings: &PsfAnalysisSettings,
) -> MkResult<PsfResult> {
    psf_analysis_job(tracer, settings, &JobContext::new())
}

pub fn psf_analysis_job(
    tracer: &SequentialTracer,
    settings: &PsfAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<PsfResult> {
    psf_analysis_shared(tracer, settings, None, ctx)
}

/// [`psf_analysis_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn psf_analysis_shared(
    tracer: &SequentialTracer,
    settings: &PsfAnalysisSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<PsfResult> {
    let timer = RunTimer::start();
    if settings.grid_size > 4096 {
        return Err(MkError::InvalidParam(format!(
            "PSF grid size {}",
            settings.grid_size
        )));
    }
    if settings.method == PsfMethod::Huygens && settings.grid_size != 0 {
        check_huygens_grid(settings.grid_size)?;
    }
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
    let mut psfs = Vec::new();
    let jobs = per
        .iter()
        .map(|(_, _, fp)| {
            if settings.polychromatic {
                1
            } else {
                fp.pupils.len()
            }
        })
        .sum();
    let progress = ItemProgress::new(ctx.subrange(0.4, 1.0), jobs);
    // Fields (and a field's PSFs) are independent: they run side by side
    // and merge in field order, so results and warnings do not depend on
    // the scheduling.
    type FieldOut = (Vec<String>, Vec<FieldPsf>);
    let fields: Vec<MkResult<FieldOut>> = per
        .into_par_iter()
        .map(|(f, value, mut fp)| -> MkResult<FieldOut> {
            let mut warnings = fp.warnings.clone();
            let spacing = default_spacing(&fp, settings.spacing_mm, settings.padding)?;
            let grid = image_grid(settings.grid_size, settings.method, &fp.weighted(), spacing);
            warnings.extend(fp.ensure_window(
                grid as f64 * spacing,
                &settings.pupil,
                memo,
                &ctx.subrange(0.4, 0.4),
            )?);
            // (wavelength, weighted pupils) of each PSF of this field.
            type Group<'a> = (Option<usize>, Vec<(&'a PupilFunction, f64)>);
            let groups: Vec<Group> = if settings.polychromatic {
                vec![(
                    (fp.pupils.len() == 1).then(|| fp.pupils[0].0),
                    fp.weighted(),
                )]
            } else {
                fp.pupils
                    .iter()
                    .map(|(wl, _, p)| (Some(*wl), vec![(&p.pupil, 1.0)]))
                    .collect()
            };
            let computed: Vec<MkResult<Psf>> = groups
                .par_iter()
                .map(|(_, group)| {
                    let psf = compute_psf(
                        group,
                        settings.method,
                        grid,
                        spacing,
                        settings.normalization,
                        &progress.item(),
                    );
                    progress.finished(psf)
                })
                .collect();
            let mut psfs = Vec::with_capacity(groups.len());
            for ((wl, _), psf) in groups.iter().zip(computed) {
                let psf = psf?;
                let wl = *wl;
                for w in &psf.warnings {
                    warnings.push(format!("field {f}: {w}"));
                }
                let (_, i, j) = psf.peak();
                psfs.push(FieldPsf {
                    field: f,
                    field_value: value,
                    wavelength: wl,
                    pupil_sampling: fp
                        .sampling()
                        .into_iter()
                        .filter(|s| wl.is_none_or(|wl| s.wavelength == wl))
                        .collect(),
                    reference_mm: fp.reference_mm,
                    strehl: psf.strehl(),
                    centroid_mm: psf.centroid(),
                    peak_mm: [psf.x(i), psf.y(j)],
                    psf,
                });
            }
            Ok((warnings, psfs))
        })
        .collect();
    for field in fields {
        let (w, p) = field?;
        warnings.extend(w);
        psfs.extend(p);
    }
    ctx.report(1.0, "PSF");
    Ok(PsfResult {
        psfs,
        run_info: run_info(timer, tracer, "psf", settings, warnings),
    })
}
