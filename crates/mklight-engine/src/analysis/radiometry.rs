//! Primary-path radiometry and polarisation (`mklight-trace`,
//! docs/POLARIZATION.md).
//!
//! * `system_transmission`: pupil-averaged throughput (coatings, Fresnel,
//!   bulk absorption, vignetting) versus field and wavelength, unpolarised
//!   or with polarisation ray tracing for a given input state.
//! * `polarization_pupil_map`: Jones pupil of one field and wavelength
//!   with diattenuation, retardance and transmitted intensity per pupil
//!   sample; the per-sample data travel as typed buffers.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::JobContext;
use mklight_trace::{
    FieldPoint, PolarizationMapSettings, PolarizationState, TransmissionSettings,
    polarization_pupil_map, system_transmission,
};

use super::geometric::{crate_json, output, parse, plain};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};

pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(SystemTransmission));
    r.register(Arc::new(PolarizationPupilMap));
}

fn state_annotation() -> Value {
    json!({"description": "Input polarisation in the object-space double-pole basis: {\"type\": \"unpolarized\"}, {\"type\": \"jones\", \"e\": …} or {\"type\": \"stokes\", \"s\": {\"s0\", \"s1\", \"s2\", \"s3\"}} (CONVENTIONS §9)."})
}

// ---------------------------------------------------------------------------
// system_transmission

pub struct SystemTransmission;

/// `TransmissionSettings` with `#[serde(default)]` semantics for schemas.
#[derive(Serialize, Default)]
#[serde(transparent)]
struct TransmissionDefaults(TransmissionSettings);

impl Analysis for SystemTransmission {
    fn memoizable(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "system_transmission"
    }
    fn title(&self) -> &str {
        "System transmission"
    }
    fn category(&self) -> &str {
        "radiometry"
    }
    fn description(&self) -> &str {
        "Pupil-averaged primary-path transmission (selected DOE efficiency, coatings, Fresnel losses, bulk \
         absorption, vignetting) per field and wavelength, and the spectrally weighted \
         polychromatic transmission per field; unpolarised or by polarisation ray tracing."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<TransmissionDefaults>(json!({
            "fields": {"description": "Normalised field points [{\"hx\", \"hy\"}]; empty = the system fields (with their vignetting factors)."},
            "wavelengths": {"items": {"type": "integer", "minimum": 0}, "description": "System wavelength indices; empty = all."},
            "pupil": {"description": "Pupil sampling: {\"type\": \"grid\", \"n\": n} (n×n grid inside the unit disk, n = 1 or ≥ 3) or {\"type\": \"random\", \"count\": n, \"seed\": s}."},
            "model": {"description": "{\"type\": \"unpolarized\"} (product of s/p-averaged interface factors) or {\"type\": \"polarized\", \"state\": …} (polarisation ray tracing)."},
            "clip": {"description": "Apply apertures (vignetted rays transmit nothing)."}
        }))
    }
    fn seed(&self, settings: &Value) -> Option<u64> {
        (settings["pupil"]["type"] == "random")
            .then(|| settings["pupil"]["seed"].as_u64())
            .flatten()
    }
    fn result_schema(&self) -> Value {
        let point = obj(&[
            ("field", int("Index into `fields`.")),
            ("wavelength", int("System wavelength index.")),
            ("wavelength_um", num("µm", "")),
            (
                "transmission",
                num(
                    "",
                    "Mean throughput over all usable samples (blocked rays count 0).",
                ),
            ),
            (
                "mean_unvignetted",
                num("", "Mean throughput of the rays that reach the image."),
            ),
            ("unvignetted_fraction", num("", "")),
            ("samples", int("Usable samples.")),
            ("failed", int("Samples excluded after a numerical failure.")),
        ]);
        result(
            &[
                (
                    "fields",
                    arr(opaque("Normalised field point {hx, hy, vignetting}."), ""),
                ),
                ("wavelengths", arr(int(""), "")),
                ("points", arr(point, "Field-major: entry f·W + w.")),
                (
                    "polychromatic",
                    arr(
                        num("", ""),
                        "Per field: spectral-weight average of transmission.",
                    ),
                ),
            ],
            vec![],
        )
    }
    fn gpu_path(&self) -> bool {
        true
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let tracer = input.tracer(ctx)?;
        let s = parse(settings)?;
        let r = super::gpu::native_job!(
            input,
            |b| mklight_gpu::radiometry::system_transmission(b, &tracer, &s, ctx),
            system_transmission(&tracer, &s, ctx)
        )?;
        let (v, w) = crate_json(&r)?;
        Ok(output(v, vec![], w))
    }
}

// ---------------------------------------------------------------------------
// polarization_pupil_map

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PolarizationMapOptions {
    /// System field index (its vignetting factors apply).
    pub field: usize,
    /// A normalised field point `[hx, hy]` instead of the system field.
    pub point: Option<[f64; 2]>,
    /// System wavelength index; null = primary.
    pub wavelength: Option<usize>,
    /// Pupil grid size (`n × n` inside the unit disk; 1 or at least 3).
    pub grid: usize,
    pub state: PolarizationState,
    pub clip: bool,
}

impl Default for PolarizationMapOptions {
    fn default() -> Self {
        PolarizationMapOptions {
            field: 0,
            point: None,
            wavelength: None,
            grid: 33,
            state: PolarizationState::Unpolarized,
            clip: true,
        }
    }
}

pub struct PolarizationPupilMap;

impl Analysis for PolarizationPupilMap {
    fn memoizable(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "polarization_pupil_map"
    }
    fn title(&self) -> &str {
        "Polarisation pupil map"
    }
    fn category(&self) -> &str {
        "polarization"
    }
    fn description(&self) -> &str {
        "Polarisation ray tracing over the pupil of one field and wavelength: amplitude \
         Jones matrix (double-pole bases), transmitted intensity for an input state, \
         diattenuation and retardance per sample, with pupil statistics."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<PolarizationMapOptions>(json!({
            "field": {"minimum": 0, "description": "System field index (with its vignetting factors)."},
            "point": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Normalised field point [hx, hy] used instead of `field`."},
            "wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "System wavelength index; null = the primary."},
            "grid": {"minimum": 1, "maximum": 2049, "description": "Pupil grid n (n×n inside the unit disk; 1 or ≥ 3)."},
            "state": state_annotation(),
            "clip": {"description": "Apply apertures."}
        }))
    }
    fn result_schema(&self) -> Value {
        result(
            &[
                (
                    "field",
                    opaque("The normalised field point {hx, hy, vignetting}."),
                ),
                ("wavelength", int("System wavelength index.")),
                ("wavelength_um", num("µm", "")),
                ("grid", int("")),
                (
                    "sample_count",
                    int("Samples N (grid points inside the unit disk)."),
                ),
                (
                    "mean_intensity",
                    num(
                        "",
                        "Pupil-mean intensity over the samples reaching the image.",
                    ),
                ),
                ("max_diattenuation", num("", "")),
                (
                    "max_retardance",
                    num("rad", "Over samples with a defined retarder."),
                ),
                ("rms_retardance", num("rad", "")),
                (
                    "undefined_retarder",
                    int(
                        "Samples reaching the image with a singular Jones matrix (excluded from the retardance statistics).",
                    ),
                ),
            ],
            vec![
                buffer(
                    "pupil",
                    "f64",
                    &["N", "2"],
                    "",
                    "Normalised pupil coordinates (Px, Py).",
                ),
                buffer(
                    "reached",
                    "u8",
                    &["N"],
                    "",
                    "1 when the ray reached the image (else every value is zero).",
                ),
                buffer(
                    "jones",
                    "f64",
                    &["N", "2", "2", "2"],
                    "",
                    "Amplitude Jones matrix √τ·J, [row][col][re, im], double-pole bases.",
                ),
                buffer(
                    "intensity",
                    "f64",
                    &["N"],
                    "",
                    "Transmitted power fraction for `state` (normalised by S₀).",
                ),
                buffer(
                    "diattenuation",
                    "f64",
                    &["N"],
                    "",
                    "(T_max − T_min)/(T_max + T_min).",
                ),
                buffer(
                    "diattenuation_axis",
                    "f64",
                    &["N", "3"],
                    "",
                    "Unit Stokes vector of the best-transmitted state.",
                ),
                buffer(
                    "retardance",
                    "f64",
                    &["N"],
                    "rad",
                    "Proper retardance δ ∈ [0, π] (0 when undefined).",
                ),
                buffer(
                    "fast_axis",
                    "f64",
                    &["N", "3"],
                    "",
                    "Unit Stokes vector of the fast eigenpolarisation.",
                ),
                buffer("retarder_defined", "u8", &["N"], "", ""),
                buffer(
                    "retardance_jones",
                    "f64",
                    &["N"],
                    "rad",
                    "Retardance of the Jones matrix itself (includes the geometric rotation and mirror parity).",
                ),
                buffer("t_max", "f64", &["N"], "", ""),
                buffer("t_min", "f64", &["N"], "", ""),
            ],
        )
    }
    fn gpu_path(&self) -> bool {
        true
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let o: PolarizationMapOptions = parse(settings)?;
        let tracer = input.tracer(ctx)?;
        let sys = tracer.system();
        let field = match o.point {
            Some([hx, hy]) => FieldPoint::normalized(hx, hy),
            None => {
                if o.field >= sys.system().fields.fields.len() {
                    return Err(EngineError::InvalidSettings(format!(
                        "field {} out of range",
                        o.field
                    )));
                }
                FieldPoint::system(sys.system(), o.field)
            }
        };
        let wl = o.wavelength.unwrap_or(sys.primary_wavelength);
        let s = PolarizationMapSettings {
            field,
            wavelength: wl,
            grid: o.grid,
            state: o.state,
            clip: o.clip,
        };
        let r = super::gpu::native_job!(
            input,
            |b| mklight_gpu::radiometry::polarization_pupil_map(b, &tracer, &s, ctx),
            polarization_pupil_map(&tracer, &s, ctx)
        )?;
        // Samples go straight into the typed buffers; only the summary is
        // serialised.
        let mut r = r;
        let samples = std::mem::take(&mut r.samples);
        let (mut v, w) = crate_json(&r)?;
        v.as_object_mut().map(|m| m.remove("samples"));
        let n = samples.len();
        let mut pupil = Vec::with_capacity(2 * n);
        let mut reached = Vec::with_capacity(n);
        let mut jones = Vec::with_capacity(8 * n);
        let column = || Vec::with_capacity(n);
        let (mut inten, mut dia, mut ret, mut retj, mut tmax, mut tmin) =
            (column(), column(), column(), column(), column(), column());
        let (mut dax, mut fax, mut defined) = (
            Vec::with_capacity(3 * n),
            Vec::with_capacity(3 * n),
            Vec::with_capacity(n),
        );
        for p in &samples {
            pupil.extend_from_slice(&p.pupil);
            reached.push(p.status.is_ok() as u8);
            // The Jones matrix serialises as plain [re, im] numbers.
            for c in p.jones.m.iter().flatten() {
                jones.extend([plain(c.re), plain(c.im)]);
            }
            let pr = &p.properties;
            inten.push(p.intensity);
            dia.push(pr.diattenuation);
            dax.extend_from_slice(&pr.diattenuation_axis);
            ret.push(pr.retardance);
            fax.extend_from_slice(&pr.fast_axis);
            defined.push(pr.retarder_defined as u8);
            retj.push(p.retardance_jones);
            tmax.push(pr.t_max);
            tmin.push(pr.t_min);
        }
        if let Some(m) = v.as_object_mut() {
            m.insert("field".into(), serde_json::to_value(field)?);
            m.insert("wavelength".into(), json!(wl));
            m.insert("wavelength_um".into(), json!(sys.wavelengths_um[wl]));
            m.insert("grid".into(), json!(o.grid));
            m.insert("sample_count".into(), json!(n));
        }
        let b1 = |name: &str, d: Vec<f64>| Buffer::new(name, vec![n], BufferData::F64(d));
        let b3 = |name: &str, d: Vec<f64>| Buffer::new(name, vec![n, 3], BufferData::F64(d));
        let buffers = vec![
            Buffer::new("pupil", vec![n, 2], BufferData::F64(pupil)),
            Buffer::new("reached", vec![n], BufferData::U8(reached)),
            Buffer::new("jones", vec![n, 2, 2, 2], BufferData::F64(jones)),
            b1("intensity", inten),
            b1("diattenuation", dia),
            b3("diattenuation_axis", dax),
            b1("retardance", ret),
            b3("fast_axis", fax),
            Buffer::new("retarder_defined", vec![n], BufferData::U8(defined)),
            b1("retardance_jones", retj),
            b1("t_max", tmax),
            b1("t_min", tmin),
        ];
        Ok(output(v, buffers, w))
    }
}
