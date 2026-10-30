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
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use serde::{Deserialize, Serialize};

use super::{FieldPupils, check_pos, fields, run_info, system_field_pupils, wavelengths};
use crate::psf::{
    HuygensSettings, ImagePlane, PolyPsfSettings, Psf, PsfNormalization, PsfTransform, huygens_psf,
    polychromatic_psf,
};
use crate::pupil::PupilFunction;
use crate::traced::PupilSettings;
use crate::warnings::push_unique;

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
            grid_size: 128,
            spacing_mm: None,
            padding: 4.0,
            normalization: PsfNormalization::Strehl,
            polychromatic: true,
        }
    }
}

/// The PSF of one field (polychromatic) or one (field, wavelength).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldPsf {
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

/// The (weighted) PSF of `pupils` on a `grid²` image grid of `spacing`,
/// centred on the common reference point.
pub(crate) fn compute_psf(
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
                    transform: PsfTransform::Matrix,
                },
                ctx,
            )?;
            if pupils.len() == 1 {
                psf.wavelength_um = Some(pupils[0].0.wavelength_um());
            }
            Ok(psf)
        }
        PsfMethod::Huygens => {
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
    let timer = RunTimer::start();
    if !(1..=4096).contains(&settings.grid_size) {
        return Err(MkError::InvalidParam(format!(
            "PSF grid size {}",
            settings.grid_size
        )));
    }
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let per = system_field_pupils(tracer, &idx, &wls, &settings.pupil, &ctx.subrange(0.0, 0.4))?;
    let mut warnings = Vec::new();
    let mut psfs = Vec::new();
    let jobs: usize = per
        .iter()
        .map(|(_, _, fp)| {
            if settings.polychromatic {
                1
            } else {
                fp.pupils.len()
            }
        })
        .sum::<usize>()
        .max(1);
    let mut done = 0usize;
    for (f, value, fp) in &per {
        warnings.extend(fp.warnings.iter().cloned());
        let spacing = default_spacing(fp, settings.spacing_mm, settings.padding)?;
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
        for (wl, group) in groups {
            let a = 0.4 + 0.6 * done as f64 / jobs as f64;
            let b = 0.4 + 0.6 * (done + 1) as f64 / jobs as f64;
            let psf = compute_psf(
                &group,
                settings.method,
                settings.grid_size,
                spacing,
                settings.normalization,
                &ctx.subrange(a, b),
            )?;
            done += 1;
            for w in &psf.warnings {
                warnings.push(format!("field {f}: {w}"));
            }
            let (_, i, j) = psf.peak();
            psfs.push(FieldPsf {
                field: *f,
                field_value: *value,
                wavelength: wl,
                reference_mm: fp.reference_mm,
                strehl: psf.strehl(),
                centroid_mm: psf.centroid(),
                peak_mm: [psf.x(i), psf.y(j)],
                psf,
            });
        }
    }
    ctx.report(1.0, "PSF");
    Ok(PsfResult {
        psfs,
        run_info: run_info(timer, tracer, "psf", settings, warnings),
    })
}
