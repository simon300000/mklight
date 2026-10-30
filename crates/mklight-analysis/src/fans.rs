//! Ray fans (transverse aberration) and OPD fans.
//!
//! * **Tangential fan**: pupil points `(0, Py)`, `Py ∈ [−1, 1]`; the plotted
//!   value is `εy = y_ray − y_ref` (CONVENTIONS §4), and `cross` holds `εx`.
//! * **Sagittal fan**: `(Px, 0)`; the value is `εx`, `cross` holds `εy`
//!   (sagittal coma shows up there).
//!
//! Image positions are in the image surface's local frame. The transverse
//! reference is the chief ray of the **primary** wavelength by default, so
//! lateral colour appears as an offset of the other wavelengths' curves at
//! `P = 0`; [`FanReference::SameWavelength`] uses each wavelength's own chief.
//! Rays are traced with clipping on; a vignetted (or failed) sample is NaN
//! (JSON `null`).
//!
//! OPD fans use the exit-pupil reference sphere of each (field, wavelength)
//! from `mklight-trace` (CONVENTIONS §4) and are reported in waves at that
//! wavelength.

use mklight_core::job::RunTimer;
use mklight_core::{JobContext, MkResult, RunInfo};
use mklight_trace::{Grid, SequentialTracer, opd_grid_job, trace_grid_job};
use serde::{Deserialize, Serialize};

use crate::util::{
    FieldSelection, WavelengthSelection, check_count, field_value, linspace, nf, run_info,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FanReference {
    /// The primary-wavelength chief ray.
    #[default]
    PrimaryChief,
    /// Each wavelength's own chief ray.
    SameWavelength,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FanSettings {
    /// Pupil samples across each fan, from −1 to 1.
    pub samples: usize,
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    /// Transverse reference (ray fans only).
    pub reference: FanReference,
}

impl Default for FanSettings {
    fn default() -> Self {
        FanSettings {
            samples: 41,
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            reference: FanReference::PrimaryChief,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FanQuantity {
    /// `εy` (tangential) / `εx` (sagittal) in mm.
    TransverseAberration,
    /// OPD in waves at each ray's wavelength.
    Opd,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FanCurve {
    pub wavelength: usize,
    /// Normalised pupil coordinate `Py` (tangential) or `Px` (sagittal).
    pub pupil: Vec<f64>,
    /// Value per pupil sample; NaN (`null`) where the ray did not arrive.
    #[serde(with = "nf::vec")]
    pub values: Vec<f64>,
    /// The other transverse component (ray fans only).
    #[serde(with = "nf::vec", default, skip_serializing_if = "Vec::is_empty")]
    pub cross: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldFan {
    pub field: usize,
    #[serde(with = "nf::arr2")]
    pub field_value: [f64; 2],
    pub tangential: Vec<FanCurve>,
    pub sagittal: Vec<FanCurve>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FanResult {
    pub quantity: FanQuantity,
    pub fields: Vec<FieldFan>,
    pub wavelengths_um: Vec<f64>,
    pub run_info: RunInfo,
}

impl FanResult {
    pub fn field(&self, field: usize) -> Option<&FieldFan> {
        self.fields.iter().find(|f| f.field == field)
    }
}

impl FieldFan {
    pub fn tangential(&self, w: usize) -> Option<&FanCurve> {
        self.tangential.iter().find(|c| c.wavelength == w)
    }
    pub fn sagittal(&self, w: usize) -> Option<&FanCurve> {
        self.sagittal.iter().find(|c| c.wavelength == w)
    }
}

fn fan_points(n: usize) -> (Vec<f64>, Vec<[f64; 2]>) {
    let t = linspace(-1.0, 1.0, n);
    let mut pts: Vec<[f64; 2]> = t.iter().map(|&p| [0.0, p]).collect();
    pts.extend(t.iter().map(|&p| [p, 0.0]));
    (t, pts)
}

/// Transverse ray fans (module docs).
pub fn ray_fan(tracer: &SequentialTracer, settings: &FanSettings) -> MkResult<FanResult> {
    ray_fan_job(tracer, settings, &JobContext::new())
}

/// [`ray_fan`] with cancellation and progress.
pub fn ray_fan_job(
    tracer: &SequentialTracer,
    settings: &FanSettings,
    ctx: &JobContext,
) -> MkResult<FanResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("fan samples", settings.samples, 2, 100_001)?;
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let (t, pts) = fan_points(settings.samples);
    let prepared: Vec<_> = fields.iter().map(|&f| tracer.field(f).clone()).collect();
    let grid = Grid {
        fields: &prepared,
        wavelengths: &wls,
        pupil: &pts,
    };
    let rays = trace_grid_job(tracer, &grid, &tracer.options(), ctx)?;
    let n = t.len();
    let mut out = Vec::new();
    for (fi, &f) in fields.iter().enumerate() {
        let pf = &prepared[fi];
        let chief_at = |w: usize| {
            let c = tracer.chief(pf, w);
            if c.status.is_ok() {
                [c.ray.pos.x, c.ray.pos.y]
            } else {
                [f64::NAN; 2]
            }
        };
        let primary_chief = chief_at(sys.primary_wavelength);
        let (mut tan, mut sag) = (Vec::new(), Vec::new());
        for (wi, &w) in wls.iter().enumerate() {
            let c = match settings.reference {
                FanReference::PrimaryChief => primary_chief,
                FanReference::SameWavelength => chief_at(w),
            };
            let block = &rays[(fi * wls.len() + wi) * 2 * n..(fi * wls.len() + wi + 1) * 2 * n];
            let eps = |k: usize| {
                let r = &block[k];
                if r.status.is_ok() {
                    [r.point.x - c[0], r.point.y - c[1]]
                } else {
                    [f64::NAN; 2]
                }
            };
            let te: Vec<[f64; 2]> = (0..n).map(eps).collect();
            let se: Vec<[f64; 2]> = (n..2 * n).map(eps).collect();
            tan.push(FanCurve {
                wavelength: w,
                pupil: t.clone(),
                values: te.iter().map(|e| e[1]).collect(),
                cross: te.iter().map(|e| e[0]).collect(),
            });
            sag.push(FanCurve {
                wavelength: w,
                pupil: t.clone(),
                values: se.iter().map(|e| e[0]).collect(),
                cross: se.iter().map(|e| e[1]).collect(),
            });
        }
        out.push(FieldFan {
            field: f,
            field_value: field_value(tracer, f),
            tangential: tan,
            sagittal: sag,
        });
    }
    Ok(FanResult {
        quantity: FanQuantity::TransverseAberration,
        fields: out,
        wavelengths_um: sys.wavelengths_um.clone(),
        run_info: run_info(timer, tracer, "ray_fan", settings, None, Vec::new()),
    })
}

/// OPD fans (module docs). `settings.reference` is ignored: each wavelength
/// has its own reference sphere.
pub fn opd_fan(tracer: &SequentialTracer, settings: &FanSettings) -> MkResult<FanResult> {
    opd_fan_job(tracer, settings, &JobContext::new())
}

/// [`opd_fan`] with cancellation and progress.
pub fn opd_fan_job(
    tracer: &SequentialTracer,
    settings: &FanSettings,
    ctx: &JobContext,
) -> MkResult<FanResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("fan samples", settings.samples, 2, 100_001)?;
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let (t, pts) = fan_points(settings.samples);
    let prepared: Vec<_> = fields.iter().map(|&f| tracer.field(f).clone()).collect();
    let grid = Grid {
        fields: &prepared,
        wavelengths: &wls,
        pupil: &pts,
    };
    let opd = opd_grid_job(tracer, &grid, &tracer.options(), ctx)?;
    let n = t.len();
    let mut warnings = Vec::new();
    let mut out = Vec::new();
    for (fi, &f) in fields.iter().enumerate() {
        let (mut tan, mut sag) = (Vec::new(), Vec::new());
        for (wi, &w) in wls.iter().enumerate() {
            let block = &opd[(fi * wls.len() + wi) * 2 * n..(fi * wls.len() + wi + 1) * 2 * n];
            let v: Vec<f64> = block.iter().map(|o| o.unwrap_or(f64::NAN)).collect();
            if v.iter().all(|x| x.is_nan()) {
                warnings.push(format!(
                    "field {f}, wavelength {w}: no OPD (chief ray or reference sphere failed)"
                ));
            }
            tan.push(FanCurve {
                wavelength: w,
                pupil: t.clone(),
                values: v[..n].to_vec(),
                cross: Vec::new(),
            });
            sag.push(FanCurve {
                wavelength: w,
                pupil: t.clone(),
                values: v[n..].to_vec(),
                cross: Vec::new(),
            });
        }
        out.push(FieldFan {
            field: f,
            field_value: field_value(tracer, f),
            tangential: tan,
            sagittal: sag,
        });
    }
    Ok(FanResult {
        quantity: FanQuantity::Opd,
        fields: out,
        wavelengths_um: sys.wavelengths_um.clone(),
        run_info: run_info(timer, tracer, "opd_fan", settings, None, warnings),
    })
}
