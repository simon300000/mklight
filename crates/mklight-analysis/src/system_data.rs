//! System data: the first-order report table from `mklight-trace`'s
//! paraxial data (definitions in `mklight_trace::paraxial`), with stable
//! keys, labels and units, per selected wavelength. Positions are signed in
//! the conventions of CONVENTIONS §2 (pupil positions: entrance from surface
//! 1, exit from the image surface). Non-finite values (afocal EFL,
//! telecentric pupils) serialise as `"inf"`/`"-inf"`, undefined ones as `null`.

use mklight_core::job::RunTimer;
use mklight_core::{FieldKind, MkResult, RayAiming, RunInfo};
use mklight_trace::{FirstOrder, ParaxialModel, SequentialTracer};
use serde::{Deserialize, Serialize};

use crate::util::{WavelengthSelection, field_unit, nf, run_info};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemDataSettings {
    pub wavelength: WavelengthSelection,
}

impl Default for SystemDataSettings {
    fn default() -> Self {
        SystemDataSettings {
            wavelength: WavelengthSelection::Primary,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReportEntry {
    /// Stable key, e.g. `efl`.
    pub key: String,
    pub label: String,
    #[serde(with = "nf")]
    pub value: f64,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstOrderTable {
    pub wavelength: usize,
    pub wavelength_um: f64,
    pub entries: Vec<ReportEntry>,
}

impl FirstOrderTable {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.value)
    }
}

/// Wavelength-independent facts about the system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemSummary {
    pub name: String,
    pub surfaces: usize,
    pub stop: usize,
    pub object_at_infinity: bool,
    pub field_kind: FieldKind,
    pub field_unit: String,
    pub max_field: f64,
    pub wavelengths_um: Vec<f64>,
    pub primary_wavelength: usize,
    pub ray_aiming: RayAiming,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemDataResult {
    /// The first selected wavelength's table (flattened for simple clients).
    pub wavelength_um: f64,
    pub entries: Vec<ReportEntry>,
    /// One table per selected wavelength.
    pub tables: Vec<FirstOrderTable>,
    pub system: SystemSummary,
    pub run_info: RunInfo,
}

fn entry(key: &str, label: &str, value: f64, unit: &str) -> ReportEntry {
    ReportEntry {
        key: key.into(),
        label: label.into(),
        value,
        unit: unit.into(),
    }
}

fn table(tracer: &SequentialTracer, w: usize) -> FirstOrderTable {
    let sys = tracer.system();
    let fo: &FirstOrder = tracer.paraxial().at(w);
    let osys = sys.system();
    let fu = field_unit(osys.fields.kind);
    let m = ParaxialModel::new(sys, w);
    let n = &m.reduced_index;
    let petzval: f64 = (1..n.len() - 1)
        .map(|j| m.power[j] / (n[j - 1] * n[j]))
        .sum();
    let n_img = n[n.len() - 2].abs();
    let opt = |v: Option<f64>| v.unwrap_or(f64::NAN);
    let entries = vec![
        entry("efl", "Effective focal length", fo.efl, "mm"),
        entry("bfl", "Back focal length", fo.bfl, "mm"),
        entry("ffl", "Front focal point from surface 1", fo.ffl, "mm"),
        entry(
            "image_distance",
            "Paraxial image distance",
            fo.image_distance,
            "mm",
        ),
        entry(
            "object_distance",
            "Object distance",
            sys.surfaces[0].thickness,
            "mm",
        ),
        entry(
            "entrance_pupil_diameter",
            "Entrance pupil diameter",
            fo.entrance_pupil_diameter,
            "mm",
        ),
        entry(
            "entrance_pupil_position",
            "Entrance pupil position from surface 1",
            fo.entrance_pupil_position,
            "mm",
        ),
        entry(
            "exit_pupil_diameter",
            "Exit pupil diameter",
            fo.exit_pupil_diameter,
            "mm",
        ),
        entry(
            "exit_pupil_position",
            "Exit pupil position from image",
            fo.exit_pupil_position,
            "mm",
        ),
        entry("image_f_number", "Image-space F/#", fo.image_f_number, ""),
        entry(
            "working_f_number",
            "Paraxial working F/#",
            fo.working_f_number,
            "",
        ),
        entry("object_na", "Object-space NA", fo.object_na, ""),
        entry("image_na", "Image-space NA", fo.image_na, ""),
        entry(
            "magnification",
            "Paraxial magnification",
            fo.magnification,
            "",
        ),
        entry(
            "lagrange_invariant",
            "Lagrange invariant",
            fo.lagrange_invariant,
            "mm",
        ),
        entry(
            "total_track",
            "Total track (surface 1 to image)",
            fo.total_track,
            "mm",
        ),
        entry(
            "stop_semi_diameter",
            "Stop semi-diameter",
            fo.stop_semi_diameter,
            "mm",
        ),
        entry("max_field", "Maximum field", osys.fields.max_radial(), fu),
        entry(
            "paraxial_image_height",
            "Paraxial image height",
            fo.paraxial_image_height,
            "mm",
        ),
        entry(
            "image_height_per_tan",
            "Image height per unit tan θ",
            opt(fo.image_height_per_tan),
            "mm",
        ),
        entry(
            "petzval_radius",
            "Petzval radius",
            if petzval != 0.0 {
                -1.0 / (n_img * petzval)
            } else {
                f64::INFINITY
            },
            "mm",
        ),
        entry(
            "airy_radius",
            "Airy radius",
            1.22 * fo.wavelength_um * 1e-3 * fo.working_f_number,
            "mm",
        ),
    ];
    FirstOrderTable {
        wavelength: w,
        wavelength_um: fo.wavelength_um,
        entries,
    }
}

/// First-order report (module docs).
pub fn system_data(
    tracer: &SequentialTracer,
    settings: &SystemDataSettings,
) -> MkResult<SystemDataResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    let wls = settings
        .wavelength
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let tables: Vec<FirstOrderTable> = wls.iter().map(|&w| table(tracer, w)).collect();
    let osys = sys.system();
    Ok(SystemDataResult {
        wavelength_um: tables[0].wavelength_um,
        entries: tables[0].entries.clone(),
        system: SystemSummary {
            name: osys.name.clone(),
            surfaces: sys.num_surfaces(),
            stop: sys.stop,
            object_at_infinity: sys.object_at_infinity,
            field_kind: osys.fields.kind,
            field_unit: field_unit(osys.fields.kind).into(),
            max_field: osys.fields.max_radial(),
            wavelengths_um: sys.wavelengths_um.clone(),
            primary_wavelength: sys.primary_wavelength,
            ray_aiming: tracer.aiming(),
        },
        tables,
        run_info: run_info(timer, tracer, "first_order", settings, None, Vec::new()),
    })
}
