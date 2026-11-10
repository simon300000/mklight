//! MTF analyses: vs frequency, vs field and through focus.
//!
//! Frequencies are on the image surface (cycles/mm); *tangential* runs
//! along the field's meridional direction `(H_x, H_y)/|H|` (image `+y` on
//! axis), *sagittal* perpendicular to it (CONVENTIONS §4), as in the
//! geometric MTF of `mklight-analysis`. Methods:
//!
//! * `direct` (default): pupil autocorrelation at each requested frequency
//!   (`O(n²)` per frequency, no frequency lattice);
//! * `fft`: pupil → PSF → OTF by FFT (exact on the frequency lattice
//!   `Δν = cutoff/n`, interpolated between, [`crate::otf_fft`]). For a
//!   strongly aberrated pupil the interpolation within the first few
//!   lattice spacings can be far off: singlet sample at full field, `n = 64`
//!   (`Δν` = 5.4 cycles/mm), 0.092 at 5 cycles/mm against 0.037 converged
//!   (`direct` gives 0.037). Use it with fine pupil grids or for whole
//!   lattices;
//! * `huygens`: direct DFT of the Huygens PSF on the image surface
//!   ([`crate::mtf_from_psf`]), an independent route through image space.
//!
//! The default frequency range estimates 10% polychromatic contrast from
//! geometric phase slopes and diffraction, with a cutoff-relative floor. With a detector the pixel MTF
//! ([`crate::detector`]) is reported and multiplied in (`system_*`), and the
//! Nyquist frequency is marked.
//!
//! Through-focus moves the image plane by `δz` along its normal (toward the
//! propagation for `δz > 0`); for each pupil this is a reference-point
//! shift of `δz/cosθ_c` along the chief ray ([`PupilFunction::with_focus_shift`],
//! exact on the reference sphere).

use mklight_analysis::util::{FieldSelection, WavelengthSelection, nf};
use mklight_core::job::RunTimer;
use mklight_core::memo::Memo;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, SequentialTracer};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::psf::{PsfMethod, check_huygens_grid, compute_psf};
use super::{
    FieldPupils, ItemProgress, check_pos, field_pupils, field_value, fields, linspace, run_info,
    system_field_pupils, wavelengths,
};
use crate::detector::{DetectorCurve, DetectorSettings};
use crate::mtf::{MtfCurve, MtfMethod, MtfOrientation, mtf_from_psf, polychromatic_mtf};
use crate::psf::PsfNormalization;
use crate::pupil::PupilFunction;
use crate::traced::PupilSettings;

/// How an MTF is computed (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtfAnalysisMethod {
    Fft,
    #[default]
    Direct,
    Huygens,
}

/// Image grid of the Huygens route.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HuygensGrid {
    pub grid_size: usize,
    /// Spacing `λ_min·F#/padding`.
    pub padding: f64,
}

impl Default for HuygensGrid {
    fn default() -> Self {
        HuygensGrid {
            grid_size: 128,
            padding: 4.0,
        }
    }
}

impl HuygensGrid {
    /// Grid side within `8..=`[`super::psf::MAX_HUYGENS_GRID`], positive
    /// padding.
    pub fn validate(&self) -> MkResult<()> {
        check_huygens_grid(self.grid_size)?;
        check_pos("Huygens padding", self.padding)
    }
}

/// MTF of weighted pupils along the field's directions.
pub(crate) fn field_mtf(
    o: &MtfOrientation,
    pupils: &[(&PupilFunction, f64)],
    freqs: &[f64],
    method: MtfAnalysisMethod,
    huygens: &HuygensGrid,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    let o = *o;
    match method {
        MtfAnalysisMethod::Fft => polychromatic_mtf(pupils, freqs, &o, MtfMethod::Fft, ctx),
        MtfAnalysisMethod::Direct => polychromatic_mtf(pupils, freqs, &o, MtfMethod::Direct, ctx),
        MtfAnalysisMethod::Huygens => {
            check_pos("padding", huygens.padding)?;
            let lf = pupils
                .iter()
                .map(|(p, _)| p.lambda_f_mm())
                .fold(f64::INFINITY, f64::min);
            let psf = compute_psf(
                pupils,
                PsfMethod::Huygens,
                huygens.grid_size,
                lf / huygens.padding,
                PsfNormalization::UnitEnergy,
                &ctx.subrange(0.0, 0.8),
            )?;
            let mut c = mtf_from_psf(&psf, freqs, &o, &ctx.subrange(0.8, 0.9))?;
            let dl = polychromatic_mtf(pupils, freqs, &o, MtfMethod::Fft, &ctx.subrange(0.9, 1.0))?;
            c.diffraction_limit = dl.diffraction_limit;
            c.cutoff = dl.cutoff;
            Ok(c)
        }
    }
}

/// Smallest system-derived useful frequency over the fields' pupils.
fn useful_frequency<'a>(fields: impl Iterator<Item = &'a FieldPupils>) -> f64 {
    let fields: Vec<&FieldPupils> = fields.collect();
    // Per field side by side; the minimum folds in field order.
    let each: Vec<f64> = fields
        .par_iter()
        .map(|fp| crate::sampling::useful_frequency(&fp.weighted()))
        .collect();
    each.into_iter().fold(f64::INFINITY, f64::min)
}

fn frequencies(
    max: Option<f64>,
    samples: usize,
    useful: impl FnOnce() -> f64,
) -> MkResult<Vec<f64>> {
    if !(2..=100_001).contains(&samples) {
        return Err(MkError::InvalidParam(format!(
            "frequency samples {samples}"
        )));
    }
    let fmax = max.unwrap_or_else(useful);
    check_pos("maximum frequency", fmax)?;
    Ok(linspace(0.0, fmax, samples))
}

// ----- MTF vs frequency ----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MtfSettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: MtfAnalysisMethod,
    /// Highest frequency (cycles/mm); estimated 10% polychromatic contrast when `None`.
    pub max_frequency: Option<f64>,
    pub frequency_samples: usize,
    pub detector: Option<DetectorSettings>,
    pub huygens: HuygensGrid,
}

impl Default for MtfSettings {
    fn default() -> Self {
        MtfSettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: MtfAnalysisMethod::Direct,
            max_frequency: None,
            frequency_samples: 41,
            detector: None,
            huygens: HuygensGrid::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldMtf {
    pub pupil_sampling: Vec<super::PupilSampling>,
    pub field: usize,
    pub field_value: [f64; 2],
    /// Unit tangential direction on the image surface.
    pub tangential_direction: [f64; 2],
    pub tangential: Vec<f64>,
    pub sagittal: Vec<f64>,
    /// Aberration-free MTF of the same pupils (tangential).
    pub diffraction_limit: Vec<f64>,
    /// Largest incoherent cutoff `1/(λF#)` of the pupils (normal plane).
    pub cutoff: f64,
    pub detector: Option<DetectorCurve>,
    /// Optical × detector MTF (with a detector).
    pub system_tangential: Option<Vec<f64>>,
    pub system_sagittal: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaveMtfResult {
    pub frequencies: Vec<f64>,
    pub fields: Vec<FieldMtf>,
    /// Detector Nyquist frequencies `(1/(2p_x), 1/(2p_y))` (with a detector).
    pub nyquist: Option<[f64; 2]>,
    pub run_info: RunInfo,
}

impl WaveMtfResult {
    pub fn field(&self, field: usize) -> Option<&FieldMtf> {
        self.fields.iter().find(|f| f.field == field)
    }
}

pub fn mtf_analysis(tracer: &SequentialTracer, settings: &MtfSettings) -> MkResult<WaveMtfResult> {
    mtf_analysis_job(tracer, settings, &JobContext::new())
}

fn with_detector(
    det: Option<&DetectorSettings>,
    freqs: &[f64],
    o: &MtfOrientation,
    c: &MtfCurve,
) -> (Option<DetectorCurve>, Option<Vec<f64>>, Option<Vec<f64>>) {
    match det {
        None => (None, None, None),
        Some(d) => {
            let dc = d.curve(freqs, o);
            let st = c
                .tangential
                .iter()
                .zip(&dc.tangential)
                .map(|(a, b)| a * b)
                .collect();
            let ss = c
                .sagittal
                .iter()
                .zip(&dc.sagittal)
                .map(|(a, b)| a * b)
                .collect();
            (Some(dc), Some(st), Some(ss))
        }
    }
}

pub fn mtf_analysis_job(
    tracer: &SequentialTracer,
    settings: &MtfSettings,
    ctx: &JobContext,
) -> MkResult<WaveMtfResult> {
    mtf_analysis_shared(tracer, settings, None, ctx)
}

/// [`mtf_analysis_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn mtf_analysis_shared(
    tracer: &SequentialTracer,
    settings: &MtfSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<WaveMtfResult> {
    let timer = RunTimer::start();
    settings.huygens.validate()?;
    if let Some(d) = &settings.detector {
        d.validate()?;
    }
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) = system_field_pupils(
        tracer,
        &idx,
        &wls,
        &settings.pupil,
        memo,
        &ctx.subrange(0.0, 0.5),
    )?;
    let freqs = frequencies(settings.max_frequency, settings.frequency_samples, || {
        useful_frequency(per.iter().map(|(_, _, fp)| fp))
    })?;
    let mut warnings = skipped;
    let mut out = Vec::new();
    let progress = ItemProgress::new(ctx.subrange(0.5, 1.0), per.len());
    let computed: Vec<_> = per
        .par_iter()
        .map(|(f, value, fp)| -> MkResult<_> {
            let mut field_warnings = fp.warnings.clone();
            let o = fp.orientation();
            let c = field_mtf(
                &o,
                &fp.weighted(),
                &freqs,
                settings.method,
                &settings.huygens,
                &progress.item(),
            )?;
            progress.finish();
            for w in &c.warnings {
                field_warnings.push(format!("field {f}: {w}"));
            }
            let (det, st, ss) = with_detector(settings.detector.as_ref(), &freqs, &o, &c);
            Ok((
                FieldMtf {
                    field: *f,
                    field_value: *value,
                    tangential_direction: fp.orientation().tangential,
                    pupil_sampling: fp.sampling(),
                    tangential: c.tangential,
                    sagittal: c.sagittal,
                    diffraction_limit: c.diffraction_limit,
                    cutoff: c.cutoff,
                    detector: det,
                    system_tangential: st,
                    system_sagittal: ss,
                },
                field_warnings,
            ))
        })
        .collect();
    for result in computed {
        let (field, field_warnings) = result?;
        warnings.extend(field_warnings);
        out.push(field);
    }
    ctx.report(1.0, "MTF");
    Ok(WaveMtfResult {
        frequencies: freqs,
        fields: out,
        nyquist: settings.detector.map(|d| d.nyquist()),
        run_info: run_info(timer, tracer, "mtf", settings, warnings),
    })
}

/// Explicit frequencies or a round base and its 2x/4x multiples.
fn fixed_frequencies(list: &[f64], useful: impl FnOnce() -> f64) -> MkResult<Vec<f64>> {
    if list.is_empty() {
        let base = crate::sampling::round_frequency(useful() / 4.0);
        return Ok([1.0, 2.0, 4.0].map(|x| x * base).to_vec());
    }
    if list.iter().any(|f| !(f.is_finite() && *f >= 0.0)) {
        return Err(MkError::InvalidParam(
            "frequencies must be finite and ≥ 0".into(),
        ));
    }
    Ok(list.to_vec())
}

// ----- MTF vs field ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MtfVsFieldSettings {
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: MtfAnalysisMethod,
    /// Frequencies (cycles/mm); round values spanning the useful range when empty.
    pub frequencies: Vec<f64>,
    /// Field points from the axis to the edge of the field (inclusive).
    pub field_samples: usize,
    /// Direction of the sweep in normalised field coordinates.
    pub direction: [f64; 2],
    pub detector: Option<DetectorSettings>,
    pub huygens: HuygensGrid,
}

impl Default for MtfVsFieldSettings {
    fn default() -> Self {
        MtfVsFieldSettings {
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: MtfAnalysisMethod::Direct,
            frequencies: Vec::new(),
            field_samples: 11,
            direction: [0.0, 1.0],
            detector: None,
            huygens: HuygensGrid::default(),
        }
    }
}

/// MTF at one frequency across the field (NaN, `null` in JSON, at
/// samples whose pupil could not be traced).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfFieldCurve {
    pub frequency: f64,
    #[serde(with = "nf::vec")]
    pub tangential: Vec<f64>,
    #[serde(with = "nf::vec")]
    pub sagittal: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfVsFieldResult {
    /// One selected-wavelength list per field sample; empty for skipped fields.
    pub pupil_sampling: Vec<Vec<super::PupilSampling>>,
    /// Normalised radial field of each sample.
    pub normalized_field: Vec<f64>,
    /// Field values in the system's field units.
    pub field_values: Vec<[f64; 2]>,
    pub frequencies: Vec<f64>,
    /// One curve per frequency (system MTF when a detector is given).
    pub curves: Vec<MtfFieldCurve>,
    pub run_info: RunInfo,
}

pub fn mtf_vs_field(
    tracer: &SequentialTracer,
    settings: &MtfVsFieldSettings,
) -> MkResult<MtfVsFieldResult> {
    mtf_vs_field_job(tracer, settings, &JobContext::new())
}

pub fn mtf_vs_field_job(
    tracer: &SequentialTracer,
    settings: &MtfVsFieldSettings,
    ctx: &JobContext,
) -> MkResult<MtfVsFieldResult> {
    mtf_vs_field_shared(tracer, settings, None, ctx)
}

/// [`mtf_vs_field_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn mtf_vs_field_shared(
    tracer: &SequentialTracer,
    settings: &MtfVsFieldSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<MtfVsFieldResult> {
    let timer = RunTimer::start();
    settings.huygens.validate()?;
    if let Some(d) = &settings.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&settings.field_samples) {
        return Err(MkError::InvalidParam(format!(
            "field samples {}",
            settings.field_samples
        )));
    }
    let [dx, dy] = settings.direction;
    let len = dx.hypot(dy);
    check_pos("field direction length", len)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    // The sweep direction is the tangential direction at every sample,
    // the axial one included.
    let o = MtfOrientation {
        tangential: [dx / len, dy / len],
    };
    let hs = linspace(0.0, 1.0, settings.field_samples);
    let mut warnings = Vec::new();
    let mut values = Vec::new();
    // `None` for a sample whose pupil cannot be built (NaN in the curves).
    let mut pupils = Vec::new();
    let mut first_error = None;
    let progress = ItemProgress::new(ctx.subrange(0.0, 0.6), hs.len());
    let traced: Vec<_> = hs
        .par_iter()
        .map(|&h| {
            let sub = progress.item();
            let point = FieldPoint::normalized(h * dx / len, h * dy / len);
            let label = format!("field {h:.3}");
            let fp = progress.finished(tracer.prepare_field(point).and_then(|prepared| {
                field_pupils(tracer, &prepared, &label, &wls, &settings.pupil, memo, &sub)
            }));
            (field_value(tracer, &point), label, fp)
        })
        .collect();
    for (value, label, fp) in traced {
        values.push(value);
        match fp {
            Ok(fp) => {
                warnings.extend(fp.warnings.iter().cloned());
                pupils.push(Some(fp));
            }
            Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                warnings.push(format!("{label} skipped: {e}"));
                first_error.get_or_insert(e);
                pupils.push(None);
            }
            Err(e) => return Err(e),
        }
    }
    if pupils.iter().all(Option::is_none) {
        return Err(first_error.unwrap_or_else(|| MkError::InvalidParam("no field sample".into())));
    }
    let freqs = fixed_frequencies(&settings.frequencies, || {
        useful_frequency(pupils.iter().flatten())
    })?;
    let mut per_field = Vec::new();
    let progress = ItemProgress::new(ctx.subrange(0.6, 1.0), hs.len());
    let computed: Vec<_> = pupils
        .par_iter()
        .map(|fp| -> MkResult<_> {
            let Some(fp) = fp else {
                progress.finish();
                return Ok(None);
            };
            let c = field_mtf(
                &o,
                &fp.weighted(),
                &freqs,
                settings.method,
                &settings.huygens,
                &progress.item(),
            )?;
            progress.finish();
            Ok(Some(c))
        })
        .collect();
    for (k, result) in computed.into_iter().enumerate() {
        let c = result?;
        if let Some(c) = &c {
            warnings.extend(
                c.warnings
                    .iter()
                    .map(|w| format!("field {:.3}: {w}", hs[k])),
            );
        }
        per_field.push(c);
    }
    let curves = freqs
        .iter()
        .enumerate()
        .map(|(q, &nu)| {
            let (mut t, mut s) = (Vec::new(), Vec::new());
            for c in &per_field {
                let Some(c) = c else {
                    t.push(f64::NAN);
                    s.push(f64::NAN);
                    continue;
                };
                let (_, st, ss) = with_detector(settings.detector.as_ref(), &freqs, &o, c);
                t.push(st.map_or(c.tangential[q], |v| v[q]));
                s.push(ss.map_or(c.sagittal[q], |v| v[q]));
            }
            MtfFieldCurve {
                frequency: nu,
                tangential: t,
                sagittal: s,
            }
        })
        .collect();
    ctx.report(1.0, "MTF vs field");
    Ok(MtfVsFieldResult {
        pupil_sampling: pupils
            .iter()
            .map(|fp| fp.as_ref().map_or_else(Vec::new, |fp| fp.sampling()))
            .collect(),
        normalized_field: hs,
        field_values: values,
        frequencies: freqs,
        curves,
        run_info: run_info(timer, tracer, "mtf_vs_field", settings, warnings),
    })
}

// ----- Through focus ------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThroughFocusSettings {
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub pupil: PupilSettings,
    pub method: MtfAnalysisMethod,
    /// Frequencies (cycles/mm); round values spanning the useful range when empty.
    pub frequencies: Vec<f64>,
    /// Half-range of image-plane shift (mm); max(8 λN², 4 N RMS blur)
    /// over the selected pupils when `None`.
    pub focus_range_mm: Option<f64>,
    pub focus_samples: usize,
    pub detector: Option<DetectorSettings>,
    pub huygens: HuygensGrid,
}

impl Default for ThroughFocusSettings {
    fn default() -> Self {
        ThroughFocusSettings {
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            pupil: PupilSettings::default(),
            method: MtfAnalysisMethod::Direct,
            frequencies: Vec::new(),
            focus_range_mm: None,
            focus_samples: 21,
            detector: None,
            huygens: HuygensGrid::default(),
        }
    }
}

/// MTF at one frequency vs focus.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughFocusCurve {
    pub frequency: f64,
    pub tangential: Vec<f64>,
    pub sagittal: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughFocusField {
    pub pupil_sampling: Vec<super::PupilSampling>,
    pub field: usize,
    pub field_value: [f64; 2],
    pub curves: Vec<ThroughFocusCurve>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughFocusResult {
    /// Image-plane shifts (mm, along the propagation for `> 0`).
    pub focus_mm: Vec<f64>,
    pub frequencies: Vec<f64>,
    pub fields: Vec<ThroughFocusField>,
    pub run_info: RunInfo,
}

pub fn through_focus_mtf_analysis(
    tracer: &SequentialTracer,
    settings: &ThroughFocusSettings,
) -> MkResult<ThroughFocusResult> {
    through_focus_mtf_analysis_job(tracer, settings, &JobContext::new())
}

pub fn through_focus_mtf_analysis_job(
    tracer: &SequentialTracer,
    settings: &ThroughFocusSettings,
    ctx: &JobContext,
) -> MkResult<ThroughFocusResult> {
    through_focus_mtf_analysis_shared(tracer, settings, None, ctx)
}

/// [`through_focus_mtf_analysis_job`] sharing traced pupils with other analyses of the same
/// system through `memo` (`crate::shared`); results are bit-identical.
pub fn through_focus_mtf_analysis_shared(
    tracer: &SequentialTracer,
    settings: &ThroughFocusSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<ThroughFocusResult> {
    let timer = RunTimer::start();
    settings.huygens.validate()?;
    if let Some(d) = &settings.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&settings.focus_samples) {
        return Err(MkError::InvalidParam(format!(
            "focus samples {}",
            settings.focus_samples
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
    let freqs = fixed_frequencies(&settings.frequencies, || {
        useful_frequency(per.iter().map(|(_, _, fp)| fp))
    })?;
    let half = match settings.focus_range_mm {
        Some(r) => {
            check_pos("focus range", r)?;
            r
        }
        None => {
            let pupils: Vec<_> = per.iter().flat_map(|(_, _, fp)| fp.pupils.iter()).collect();
            // Per pupil side by side; the maximum folds in pupil order.
            let each: Vec<f64> = pupils
                .par_iter()
                .map(|(_, _, tp)| {
                    let p = &tp.pupil;
                    let n = p.geometry().working_f_number();
                    (8.0 * p.wavelength_um() * 1e-3 * n * n)
                        .max(4.0 * n * crate::sampling::blur(p).1)
                })
                .collect();
            each.into_iter().fold(0.0, f64::max)
        }
    };
    let focus = if settings.focus_samples == 1 {
        vec![0.0]
    } else {
        linspace(-half, half, settings.focus_samples)
    };
    let mut warnings = skipped;
    let mut out = Vec::new();
    let progress = ItemProgress::new(ctx.subrange(0.3, 1.0), per.len());
    type FocusField = ((Vec<String>, Vec<String>), ThroughFocusField);
    // Fields are independent: they run side by side and merge in field
    // order (results and warnings do not depend on the scheduling).
    let computed: Vec<MkResult<FocusField>> = per
        .into_par_iter()
        .map(|(f, value, mut fp)| {
            let mut warnings: Vec<String> = Vec::new();
            let sub = progress.item();
            warnings.extend(fp.warnings.iter().cloned());
            if settings.pupil.grid == 0 {
                // Automatic pupils also resolve the phase at both focus ends.
                let refined: Vec<Vec<String>> = fp
                    .pupils
                    .par_iter_mut()
                    .zip(&fp.keys)
                    .map(|((_, _, tp), key)| {
                        let p = &tp.pupil;
                        let step = [-half, half]
                            .map(|dz| {
                                p.with_focus_shift(dz / p.geometry().chief_ray[2])
                                    .max_phase_step()
                            })
                            .into_iter()
                            .fold(0.0, f64::max);
                        let n = ((p.n() as f64 * step / 0.35)
                            .max(p.n() as f64)
                            .min(crate::sampling::MAX_AUTO_PUPIL as f64)
                            .ceil() as usize)
                            .next_power_of_two();
                        crate::shared::resample(
                            tp,
                            key,
                            n,
                            settings.pupil.supersample,
                            memo,
                            &sub.subrange(0.0, 0.0),
                        )
                    })
                    .collect::<MkResult<_>>()?;
                warnings.extend(refined.into_iter().flatten());
            }
            let o = fp.orientation();
            let curves = if settings.method == MtfAnalysisMethod::Direct {
                crate::mtf::polychromatic_focus_mtf(&fp.weighted(), &focus, &freqs, &o, &sub)?
            } else {
                let planes = ItemProgress::new(sub.clone(), focus.len());
                // Planes side by side; the first error in plane order wins.
                focus
                    .par_iter()
                    .map(|&dz| {
                        let shifted: Vec<PupilFunction> = fp
                            .pupils
                            .iter()
                            .map(|(_, _, p)| {
                                let cos = p.pupil.geometry().chief_ray[2];
                                p.pupil.with_focus_shift(dz / cos)
                            })
                            .collect();
                        let weighted: Vec<(&PupilFunction, f64)> = shifted
                            .iter()
                            .zip(&fp.pupils)
                            .map(|(p, (_, w, _))| (p, *w))
                            .collect();
                        let c = field_mtf(
                            &o,
                            &weighted,
                            &freqs,
                            settings.method,
                            &settings.huygens,
                            &planes.item(),
                        );
                        planes.finished(c)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<MkResult<Vec<_>>>()?
            };
            let mut t = vec![vec![0.0; focus.len()]; freqs.len()];
            let mut s = t.clone();
            let mut tagged = Vec::new();
            for (z, c) in curves.iter().enumerate() {
                tagged.extend(c.warnings.iter().map(|w| format!("field {f}: {w}")));
                let (_, st, ss) = with_detector(settings.detector.as_ref(), &freqs, &o, c);
                for q in 0..freqs.len() {
                    t[q][z] = st.as_ref().map_or(c.tangential[q], |v| v[q]);
                    s[q][z] = ss.as_ref().map_or(c.sagittal[q], |v| v[q]);
                }
            }
            progress.finish();
            Ok((
                (warnings, tagged),
                ThroughFocusField {
                    pupil_sampling: fp.sampling(),
                    field: f,
                    field_value: value,
                    curves: freqs
                        .iter()
                        .enumerate()
                        .map(|(q, &nu)| ThroughFocusCurve {
                            frequency: nu,
                            tangential: t[q].clone(),
                            sagittal: s[q].clone(),
                        })
                        .collect(),
                },
            ))
        })
        .collect();
    for c in computed {
        // Field and pupil warnings append as they come; per-plane curve
        // warnings are deduplicated against everything so far.
        let ((plain, tagged), field) = c?;
        warnings.extend(plain);
        for m in tagged {
            if !warnings.contains(&m) {
                warnings.push(m);
            }
        }
        out.push(field);
    }
    ctx.report(1.0, "through-focus MTF");
    Ok(ThroughFocusResult {
        focus_mm: focus,
        frequencies: freqs,
        fields: out,
        run_info: run_info(timer, tracer, "through_focus_mtf", settings, warnings),
    })
}
