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
//!   (`Δν` = 5.4 cycles/mm), 0.088 at 5 cycles/mm against 0.041 converged
//!   (`direct` gives 0.038). Use it with fine pupil grids or for whole
//!   lattices;
//! * `huygens`: direct DFT of the Huygens PSF on the image surface
//!   ([`crate::mtf_from_psf`]), an independent route through image space.
//!
//! The default frequency range runs to the largest incoherent cutoff
//! `1/(λ_min·F#)` of the field's pupils. With a detector the pixel MTF
//! ([`crate::detector`]) is reported and multiplied in (`system_*`), and the
//! Nyquist frequency is marked.
//!
//! Through-focus moves the image plane by `δz` along its normal (toward the
//! propagation for `δz > 0`); for each pupil this is a reference-point
//! shift of `δz/cosθ_c` along the chief ray ([`PupilFunction::with_focus_shift`],
//! exact on the reference sphere).

use mklight_analysis::util::{FieldSelection, WavelengthSelection, nf};
use mklight_core::job::RunTimer;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, SequentialTracer};
use serde::{Deserialize, Serialize};

use super::psf::{PsfMethod, check_huygens_grid, compute_psf};
use super::{
    check_pos, field_pupils, field_value, fields, linspace, run_info, system_field_pupils,
    wavelengths,
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

fn frequencies(max: Option<f64>, samples: usize, cutoff: f64) -> MkResult<Vec<f64>> {
    if !(2..=100_001).contains(&samples) {
        return Err(MkError::InvalidParam(format!(
            "frequency samples {samples}"
        )));
    }
    let fmax = max.unwrap_or(cutoff);
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
    /// Highest frequency (cycles/mm); the largest cutoff when `None`.
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
    let timer = RunTimer::start();
    settings.huygens.validate()?;
    if let Some(d) = &settings.detector {
        d.validate()?;
    }
    let idx = fields(tracer, &settings.fields)?;
    let wls = wavelengths(tracer, &settings.wavelengths)?;
    let (per, skipped) =
        system_field_pupils(tracer, &idx, &wls, &settings.pupil, &ctx.subrange(0.0, 0.5))?;
    let cutoff = per
        .iter()
        .map(|(_, _, fp)| fp.max_cutoff())
        .fold(0.0, f64::max);
    let freqs = frequencies(settings.max_frequency, settings.frequency_samples, cutoff)?;
    let mut warnings = skipped;
    let mut out = Vec::new();
    let nf = per.len().max(1) as f64;
    for (k, (f, value, fp)) in per.iter().enumerate() {
        warnings.extend(fp.warnings.iter().cloned());
        let sub = ctx.subrange(0.5 + 0.5 * k as f64 / nf, 0.5 + 0.5 * (k + 1) as f64 / nf);
        let o = fp.orientation();
        let c = field_mtf(
            &o,
            &fp.weighted(),
            &freqs,
            settings.method,
            &settings.huygens,
            &sub,
        )?;
        for w in &c.warnings {
            warnings.push(format!("field {f}: {w}"));
        }
        let (det, st, ss) = with_detector(settings.detector.as_ref(), &freqs, &o, &c);
        out.push(FieldMtf {
            field: *f,
            field_value: *value,
            tangential_direction: fp.orientation().tangential,
            tangential: c.tangential,
            sagittal: c.sagittal,
            diffraction_limit: c.diffraction_limit,
            cutoff: c.cutoff,
            detector: det,
            system_tangential: st,
            system_sagittal: ss,
        });
    }
    ctx.report(1.0, "MTF");
    Ok(WaveMtfResult {
        frequencies: freqs,
        fields: out,
        nyquist: settings.detector.map(|d| d.nyquist()),
        run_info: run_info(timer, tracer, "mtf", settings, warnings),
    })
}

/// Fixed frequencies: the given list, or `cutoff × (1/8, 1/4, 1/2)`.
fn fixed_frequencies(list: &[f64], cutoff: f64) -> MkResult<Vec<f64>> {
    if list.is_empty() {
        return Ok(vec![0.125 * cutoff, 0.25 * cutoff, 0.5 * cutoff]);
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
    /// Frequencies (cycles/mm); on-axis `cutoff × (1/8, 1/4, 1/2)` when empty.
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
    let ns = hs.len() as f64;
    let mut warnings = Vec::new();
    let mut values = Vec::new();
    // `None` for a sample whose pupil cannot be built (NaN in the curves).
    let mut per_field: Vec<Option<MtfCurve>> = Vec::new();
    let mut freqs: Option<Vec<f64>> = None;
    let mut first_error = None;
    for (k, &h) in hs.iter().enumerate() {
        let sub = ctx.subrange(k as f64 / ns, (k + 1) as f64 / ns);
        let point = FieldPoint::normalized(h * dx / len, h * dy / len);
        values.push(field_value(tracer, &point));
        let label = format!("field {h:.3}");
        let fp = tracer.prepare_field(point).and_then(|prepared| {
            field_pupils(
                tracer,
                &prepared,
                &label,
                &wls,
                &settings.pupil,
                &sub.subrange(0.0, 0.6),
            )
        });
        let fp = match fp {
            Ok(fp) => fp,
            Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                warnings.push(format!("{label} skipped: {e}"));
                first_error.get_or_insert(e);
                per_field.push(None);
                continue;
            }
            Err(e) => return Err(e),
        };
        warnings.extend(fp.warnings.iter().cloned());
        let f = match &freqs {
            Some(f) => f.clone(),
            None => {
                let f = fixed_frequencies(&settings.frequencies, fp.max_cutoff())?;
                freqs = Some(f.clone());
                f
            }
        };
        let c = field_mtf(
            &o,
            &fp.weighted(),
            &f,
            settings.method,
            &settings.huygens,
            &sub.subrange(0.6, 1.0),
        )?;
        for w in &c.warnings {
            warnings.push(format!("{label}: {w}"));
        }
        per_field.push(Some(c));
    }
    let Some(freqs) = freqs else {
        return Err(first_error.unwrap_or_else(|| MkError::InvalidParam("no field sample".into())));
    };
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
    /// Frequencies (cycles/mm); `cutoff × (1/8, 1/4, 1/2)` when empty.
    pub frequencies: Vec<f64>,
    /// Half-range of the image-plane shift (mm); `4·λ·F#²` (about four
    /// Rayleigh depths of focus) when `None`.
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
    let (per, skipped) =
        system_field_pupils(tracer, &idx, &wls, &settings.pupil, &ctx.subrange(0.0, 0.3))?;
    let first = &per
        .first()
        .ok_or_else(|| MkError::InvalidParam("no field".into()))?
        .2;
    let freqs = fixed_frequencies(&settings.frequencies, first.max_cutoff())?;
    let half = match settings.focus_range_mm {
        Some(r) => {
            check_pos("focus range", r)?;
            r
        }
        None => {
            let p = &first.pupils[0].2;
            let n = p.pupil.geometry().working_f_number();
            4.0 * p.pupil.wavelength_um() * 1e-3 * n * n
        }
    };
    let focus = if settings.focus_samples == 1 {
        vec![0.0]
    } else {
        linspace(-half, half, settings.focus_samples)
    };
    let mut warnings = skipped;
    let mut out = Vec::new();
    let total = (per.len() * focus.len()).max(1) as f64;
    let mut done = 0.0;
    for (f, value, fp) in &per {
        warnings.extend(fp.warnings.iter().cloned());
        let mut t = vec![vec![0.0; focus.len()]; freqs.len()];
        let mut s = t.clone();
        for (z, &dz) in focus.iter().enumerate() {
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
            let sub = ctx.subrange(0.3 + 0.7 * done / total, 0.3 + 0.7 * (done + 1.0) / total);
            let c = field_mtf(
                &fp.orientation(),
                &weighted,
                &freqs,
                settings.method,
                &settings.huygens,
                &sub,
            )?;
            done += 1.0;
            for w in &c.warnings {
                let m = format!("field {f}: {w}");
                if !warnings.contains(&m) {
                    warnings.push(m);
                }
            }
            let (_, st, ss) =
                with_detector(settings.detector.as_ref(), &freqs, &fp.orientation(), &c);
            for q in 0..freqs.len() {
                t[q][z] = st.as_ref().map_or(c.tangential[q], |v| v[q]);
                s[q][z] = ss.as_ref().map_or(c.sagittal[q], |v| v[q]);
            }
        }
        out.push(ThroughFocusField {
            field: *f,
            field_value: *value,
            curves: freqs
                .iter()
                .enumerate()
                .map(|(q, &nu)| ThroughFocusCurve {
                    frequency: nu,
                    tangential: t[q].clone(),
                    sagittal: s[q].clone(),
                })
                .collect(),
        });
    }
    ctx.report(1.0, "through-focus MTF");
    Ok(ThroughFocusResult {
        focus_mm: focus,
        frequencies: freqs,
        fields: out,
        run_info: run_info(timer, tracer, "through_focus_mtf", settings, warnings),
    })
}
