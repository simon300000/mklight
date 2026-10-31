//! Wave-optics analyses on the reference tracer.
//!
//! Every analysis has the shape of `mklight-analysis` (`docs/ANALYSIS.md`):
//! a pure function `analysis(&SequentialTracer, &Settings)` and a job
//! variant `analysis_job(&SequentialTracer, &Settings, &JobContext)` with
//! cancellation and monotone progress. Settings are serde structs with
//! `Default` and `#[serde(default)]` (a partial JSON object is valid);
//! results are serde structs carrying a `RunInfo` (backend `cpu-f64`,
//! precision `f64_reference`, the tracer's and the analysis' warnings).
//! Field and wavelength selections are `mklight-analysis`'s, with the same
//! JSON.
//!
//! [`ANALYSES`] lists them with a uniform JSON entry point
//! `fn(&SequentialTracer, &serde_json::Value, &JobContext) ->
//! MkResult<serde_json::Value>`, so a registry adds each with one line.
//! A tracer is one configuration and its `RunInfo` names the snapshot the
//! tracer was built from; to analyse configuration `k` of a session
//! system, build a [`ConfiguredTracer`] ([`crate::traced::configured_tracer`])
//! and use [`WaveAnalysisEntry::run_configured`] (or
//! [`ConfiguredTracer::stamp`] on a typed result) so the `RunInfo` carries
//! the session snapshot's hash, the configuration and the solve warnings.
//!
//! Fields whose pupil cannot be built (chief ray failed, fully vignetted)
//! are left out of per-field results with a `RunInfo` warning (`NaN`,
//! JSON `null`, in `mtf_vs_field`); an analysis fails only when no
//! selected field is usable.
//!
//! **Polychromatic weighting.** Spectral weights are the system's
//! wavelength weights (relative energies at the image). Each wavelength's
//! PSF is normalised to unit energy before weighting
//! ([`crate::polychromatic_psf`]), OTFs are weighted the same way
//! ([`crate::polychromatic_mtf`]), and each wavelength is placed at its own
//! reference point relative to the field's common reference (lateral
//! colour): the primary wavelength's chief-ray point, or the weighted mean
//! of the per-wavelength centroids with [`ReferencePoint::Centroid`].

pub mod energy;
pub mod mtf;
pub mod psf;
pub mod wavefront;

use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::job::{Precision, RunTimer};
use mklight_core::{JobContext, MkError, MkResult, RunInfo, Vec3};
use mklight_trace::{FieldPoint, PreparedField, SequentialTracer};
use rayon::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::mtf::MtfOrientation;
use crate::pupil::PupilFunction;
use crate::traced::{ConfiguredTracer, PupilSettings, ReferencePoint, TracedPupil, trace_pupil};

pub use energy::{
    EncircledEnergySettings, FieldEnergy, FieldLineSpread, LineSpreadResult, LineSpreadSettings,
    WaveEncircledEnergyResult, diffraction_encircled_energy, diffraction_encircled_energy_job,
    line_spread, line_spread_job,
};
pub use mtf::{
    FieldMtf, HuygensGrid, MtfAnalysisMethod, MtfFieldCurve, MtfSettings, MtfVsFieldResult,
    MtfVsFieldSettings, ThroughFocusCurve, ThroughFocusField, ThroughFocusResult,
    ThroughFocusSettings, WaveMtfResult, mtf_analysis, mtf_analysis_job, mtf_vs_field,
    mtf_vs_field_job, through_focus_mtf_analysis, through_focus_mtf_analysis_job,
};
pub use psf::{
    FieldPsf, MAX_HUYGENS_GRID, MAX_HUYGENS_WORK, MIN_ENERGY_IN_GRID, PsfAnalysisSettings,
    PsfMethod, PsfResult, psf_analysis, psf_analysis_job,
};
pub use wavefront::{
    WavefrontMap, WavefrontResult, WavefrontSettings, wavefront_map, wavefront_map_job,
};

/// Backend recorded in `RunInfo` (the CPU f64 reference tracer).
pub const BACKEND: &str = "cpu-f64";

/// One wave analysis for registries: name, menu text and a JSON entry
/// point (settings JSON in, result JSON out; missing settings take their
/// defaults).
#[derive(Clone, Copy)]
pub struct WaveAnalysisEntry {
    pub name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    /// The default settings as JSON.
    pub default_settings: fn() -> Value,
    pub run: fn(&SequentialTracer, &Value, &JobContext) -> MkResult<Value>,
}

impl WaveAnalysisEntry {
    /// Run on one configuration of a system: [`Self::run`] on its tracer,
    /// with the `run_info` stamped by [`ConfiguredTracer::stamp`] (base
    /// snapshot hash, `"configuration"` in the settings, solve warnings).
    pub fn run_configured(
        &self,
        cfg: &ConfiguredTracer,
        settings: &Value,
        ctx: &JobContext,
    ) -> MkResult<Value> {
        let mut v = (self.run)(&cfg.tracer, settings, ctx)?;
        cfg.stamp_json(&mut v)?;
        Ok(v)
    }
}

impl std::fmt::Debug for WaveAnalysisEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveAnalysisEntry")
            .field("name", &self.name)
            .finish()
    }
}

fn parse<S: DeserializeOwned>(v: &Value) -> MkResult<S> {
    let v = if v.is_null() {
        Value::Object(Default::default())
    } else {
        v.clone()
    };
    serde_json::from_value(v).map_err(|e| MkError::InvalidParam(format!("settings: {e}")))
}

fn to_json<R: Serialize>(r: &R) -> MkResult<Value> {
    serde_json::to_value(r).map_err(|e| MkError::Numerical(format!("result serialisation: {e}")))
}

macro_rules! entry {
    ($name:literal, $title:literal, $desc:literal, $settings:ty, $job:path) => {{
        fn defaults() -> Value {
            serde_json::to_value(<$settings>::default()).unwrap_or(Value::Null)
        }
        fn run(t: &SequentialTracer, v: &Value, ctx: &JobContext) -> MkResult<Value> {
            let s: $settings = parse(v)?;
            to_json(&$job(t, &s, ctx)?)
        }
        WaveAnalysisEntry {
            name: $name,
            title: $title,
            description: $desc,
            default_settings: defaults,
            run,
        }
    }};
}

/// Every wave analysis, in menu order.
pub const ANALYSES: &[WaveAnalysisEntry] = &[
    entry!(
        "wavefront_map",
        "Wavefront map",
        "OPD on the exit-pupil reference sphere from the reference tracer, with RMS/PV, \
         Strehl estimate and Zernike fit.",
        WavefrontSettings,
        wavefront_map_job
    ),
    entry!(
        "psf",
        "Point spread function",
        "Diffraction PSF by FFT or Huygens integration, monochromatic or polychromatic, \
         with the Strehl ratio.",
        PsfAnalysisSettings,
        psf_analysis_job
    ),
    entry!(
        "mtf",
        "MTF",
        "Diffraction MTF (FFT, pupil autocorrelation or from the Huygens PSF) vs \
         frequency, tangential and sagittal, with the diffraction limit and an optional \
         detector (pixel) MTF.",
        MtfSettings,
        mtf_analysis_job
    ),
    entry!(
        "mtf_vs_field",
        "MTF vs field",
        "Diffraction MTF at fixed frequencies across the field.",
        MtfVsFieldSettings,
        mtf_vs_field_job
    ),
    entry!(
        "through_focus_mtf",
        "Through-focus MTF",
        "Diffraction MTF at fixed frequencies vs image-plane defocus.",
        ThroughFocusSettings,
        through_focus_mtf_analysis_job
    ),
    entry!(
        "diffraction_encircled_energy",
        "Diffraction encircled energy",
        "Encircled and ensquared energy of the diffraction PSF, with the \
         diffraction-limited reference.",
        EncircledEnergySettings,
        diffraction_encircled_energy_job
    ),
    entry!(
        "line_spread",
        "Line spread function",
        "Diffraction line-spread functions along the image axes.",
        LineSpreadSettings,
        line_spread_job
    ),
];

/// The entry named `name`.
pub fn find(name: &str) -> Option<&'static WaveAnalysisEntry> {
    ANALYSES.iter().find(|a| a.name == name)
}

// ---------------------------------------------------------------------------
// Shared pieces

pub(crate) fn run_info(
    timer: RunTimer,
    tracer: &SequentialTracer,
    analysis: &str,
    settings: &impl Serialize,
    warnings: Vec<String>,
) -> RunInfo {
    let mut w = tracer.warnings().to_vec();
    for x in warnings {
        if !w.contains(&x) {
            w.push(x);
        }
    }
    timer.finish(
        &tracer.system().snapshot,
        analysis,
        settings,
        BACKEND,
        Precision::F64Reference,
        None,
        w,
    )
}

/// The selected system fields.
pub(crate) fn fields(tracer: &SequentialTracer, sel: &FieldSelection) -> MkResult<Vec<usize>> {
    sel.resolve(tracer.fields().len())
}

/// The selected wavelengths with their normalised spectral weights.
pub(crate) fn wavelengths(
    tracer: &SequentialTracer,
    sel: &WavelengthSelection,
) -> MkResult<Vec<(usize, f64)>> {
    let sys = tracer.system();
    let wls = sel.resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let w: Vec<f64> = wls.iter().map(|&i| sys.wavelength_weights[i]).collect();
    let sum: f64 = w.iter().sum();
    if wls.len() == 1 {
        return Ok(vec![(wls[0], 1.0)]);
    }
    if !(sum > 0.0) || w.iter().any(|x| *x < 0.0) {
        return Err(MkError::InvalidModel(
            "the selected wavelengths have no positive weight".into(),
        ));
    }
    Ok(wls.iter().zip(&w).map(|(&i, &x)| (i, x / sum)).collect())
}

/// Field value `(x, y)` in the system's field units of a normalised point.
pub(crate) fn field_value(tracer: &SequentialTracer, p: &FieldPoint) -> [f64; 2] {
    let m = tracer.system().system().fields.max_radial();
    [p.hx * m, p.hy * m]
}

/// Actual pupil sampling in selected-wavelength order.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct PupilSampling {
    pub wavelength: usize,
    pub grid: usize,
}

/// The pupils of one field at the selected wavelengths, placed relative to
/// a common reference point.
#[derive(Clone, Debug)]
pub(crate) struct FieldPupils {
    pub point: FieldPoint,
    /// Common reference point (image-surface `x, y`, mm): PSFs are centred
    /// on it and each pupil's `image_offset_mm` is relative to it.
    pub reference_mm: [f64; 2],
    /// `(wavelength index, normalised weight, pupil)`.
    pub pupils: Vec<(usize, f64, TracedPupil)>,
    pub warnings: Vec<String>,
}

impl FieldPupils {
    pub fn weighted(&self) -> Vec<(&PupilFunction, f64)> {
        self.pupils.iter().map(|(_, w, p)| (&p.pupil, *w)).collect()
    }
    pub fn sampling(&self) -> Vec<PupilSampling> {
        self.pupils
            .iter()
            .map(|(wl, _, p)| PupilSampling {
                wavelength: *wl,
                grid: p.pupil.n(),
            })
            .collect()
    }
    pub fn orientation(&self) -> MtfOrientation {
        MtfOrientation::from_field(self.point.hx, self.point.hy)
    }
    /// Smallest `λ·F#` over the pupils (mm): the finest natural PSF scale.
    pub fn min_lambda_f(&self) -> f64 {
        self.pupils
            .iter()
            .map(|(_, _, p)| p.pupil.lambda_f_mm())
            .fold(f64::INFINITY, f64::min)
    }
    /// Refine automatically sampled pupils until their alias-free field
    /// `n λN` holds an image window of `side` mm; explicit grids are kept.
    pub(crate) fn ensure_window(
        &mut self,
        side: f64,
        settings: &PupilSettings,
        ctx: &JobContext,
    ) -> MkResult<Vec<String>> {
        if settings.grid != 0 {
            return Ok(Vec::new());
        }
        let count = self.pupils.len() as f64;
        let warnings: Vec<Vec<String>> = self
            .pupils
            .par_iter_mut()
            .enumerate()
            .map(|(i, (_, _, tp))| {
                let n = ((side / tp.pupil.lambda_f_mm())
                    .max(tp.pupil.n() as f64)
                    .min(crate::sampling::MAX_AUTO_PUPIL as f64)
                    .ceil() as usize)
                    .next_power_of_two();
                tp.resample(
                    n,
                    settings.supersample,
                    &ctx.subrange(i as f64 / count, (i + 1) as f64 / count),
                )
            })
            .collect::<MkResult<_>>()?;
        Ok(warnings.into_iter().flatten().collect())
    }
}

/// `e` with `label` prefixed to its message (cancellation and the other
/// variants pass through unchanged).
fn in_context(e: MkError, label: &str) -> MkError {
    match e {
        MkError::Trace(m) => MkError::Trace(format!("{label}: {m}")),
        MkError::Numerical(m) => MkError::Numerical(format!("{label}: {m}")),
        MkError::InvalidModel(m) => MkError::InvalidModel(format!("{label}: {m}")),
        other => other,
    }
}

/// Trace the pupils of a prepared field at `wls` and place them (module
/// docs, polychromatic weighting).
pub(crate) fn field_pupils(
    tracer: &SequentialTracer,
    field: &PreparedField,
    label: &str,
    wls: &[(usize, f64)],
    settings: &PupilSettings,
    ctx: &JobContext,
) -> MkResult<FieldPupils> {
    let nw = wls.len() as f64;
    let traced: Vec<_> = wls
        .par_iter()
        .enumerate()
        .map(|(k, &(wl, w))| {
            let sub = ctx.subrange(k as f64 / nw, (k + 1) as f64 / nw);
            trace_pupil(tracer, field, wl, settings, &sub)
                .map(|p| (wl, w, p))
                .map_err(|e| in_context(e, &format!("{label}, wavelength {wl}")))
        })
        .collect();
    let mut pupils: Vec<_> = traced.into_iter().collect::<MkResult<_>>()?;
    let mut warnings = Vec::new();
    for (wl, _, tp) in &pupils {
        warnings.extend(
            tp.warnings
                .iter()
                .map(|x| format!("{label}, wavelength {wl}: {x}")),
        );
    }
    let point_of = |p: &TracedPupil| p.info.reference_point_mm;
    let reference_mm = match settings.reference {
        ReferencePoint::Chief => {
            let sys = tracer.system();
            let c = tracer.chief(field, sys.primary_wavelength);
            if c.status.is_ok() {
                [c.ray.pos.x, c.ray.pos.y]
            } else {
                let r = point_of(&pupils[0].2);
                [r[0], r[1]]
            }
        }
        ReferencePoint::Centroid => {
            let mut c = Vec3::ZERO;
            for (_, w, p) in &pupils {
                let r = point_of(p);
                c += Vec3::new(r[0], r[1], 0.0) * *w;
            }
            [c.x, c.y]
        }
    };
    for (_, _, p) in pupils.iter_mut() {
        let r = point_of(p);
        p.pupil
            .set_image_offset_mm([r[0] - reference_mm[0], r[1] - reference_mm[1]]);
    }
    Ok(FieldPupils {
        point: field.point,
        reference_mm,
        pupils,
        warnings,
    })
}

/// A system field's index, field value and pupils.
pub(crate) type SystemFieldPupils = (usize, [f64; 2], FieldPupils);

/// The system fields `idx` traced at `wls`, each with its label and field
/// value, over consecutive progress subranges of `ctx`.
///
/// A field whose pupil cannot be built (chief ray failed, fully vignetted:
/// `Trace` or `Numerical` errors) is left out with a warning, as the
/// geometric analyses skip fields whose chief ray fails; the call fails
/// only when no selected field is usable. Returns the usable fields and
/// those warnings.
pub(crate) fn system_field_pupils(
    tracer: &SequentialTracer,
    idx: &[usize],
    wls: &[(usize, f64)],
    settings: &PupilSettings,
    ctx: &JobContext,
) -> MkResult<(Vec<SystemFieldPupils>, Vec<String>)> {
    let nf = idx.len().max(1) as f64;
    let mut out = Vec::with_capacity(idx.len());
    let mut skipped = Vec::new();
    let mut first_error = None;
    let traced: Vec<_> = idx
        .par_iter()
        .enumerate()
        .map(|(k, &f)| {
            let sub = ctx.subrange(k as f64 / nf, (k + 1) as f64 / nf);
            field_pupils(
                tracer,
                tracer.field(f),
                &format!("field {f}"),
                wls,
                settings,
                &sub,
            )
        })
        .collect();
    for (&f, result) in idx.iter().zip(traced) {
        match result {
            Ok(fp) => {
                let sf = &tracer.system().system().fields.fields[f];
                out.push((f, [sf.x, sf.y], fp));
            }
            Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                skipped.push(format!("field {f} skipped: {e}"));
                first_error.get_or_insert(e);
            }
            Err(e) => return Err(e),
        }
    }
    match first_error {
        Some(e) if out.is_empty() => Err(e),
        _ => Ok((out, skipped)),
    }
}

/// Positive and finite, or an error naming `what`.
pub(crate) fn check_pos(what: &str, v: f64) -> MkResult<()> {
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(MkError::InvalidParam(format!(
            "{what} must be positive: {v}"
        )))
    }
}

/// `n` evenly spaced values from `a` to `b` inclusive.
pub(crate) fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![a],
        _ => (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}
