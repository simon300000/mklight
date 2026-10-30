//! Wave analyses on a solved configuration of the CPU f64 reference tracer.
//! The solver owns the physics and job progress; this adapter publishes
//! schemas and moves its bulk arrays into the engine's typed buffers.

use std::sync::Arc;

use mklight_core::JobContext;
use mklight_wave::analysis::{self as wa, WaveAnalysisEntry};
use mklight_wave::traced::{PupilSettings, configured_tracer_job};
use serde::Serialize;
use serde_json::{Value, json};

use super::geometric::{num_of, output, take_run_info, xy};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::EngineResult;

struct Wave(&'static WaveAnalysisEntry);

impl Analysis for Wave {
    fn name(&self) -> &str {
        self.0.name
    }
    fn title(&self) -> &str {
        self.0.title
    }
    fn category(&self) -> &str {
        "wave"
    }
    fn description(&self) -> &str {
        self.0.description
    }
    fn settings_schema(&self) -> Value {
        settings_schema(self.0.name)
    }
    fn result_schema(&self) -> Value {
        result_schema(self.0.name)
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        // The solver's serde settings ignore the engine's `configuration` key.
        let cfg = configured_tracer_job(
            input.snapshot.system(),
            input.library.clone(),
            settings["configuration"].as_u64().unwrap() as usize,
            &ctx.subrange(0.0, 0.05),
        )?;
        let mut json = self
            .0
            .run_configured(&cfg, settings, &ctx.subrange(0.05, 1.0))?;
        let warnings = take_run_info(&mut json);
        let buffers = extract_buffers(self.0.name, &mut json);
        Ok(output(json, buffers, warnings))
    }
}

pub fn register_all(r: &mut AnalysisRegistry) {
    for entry in wa::ANALYSES {
        r.register(Arc::new(Wave(entry)));
    }
}

// Settings defaults come from the solver, including nested pupil settings.
#[derive(Serialize, Default)]
struct Configured<S: Default> {
    #[serde(flatten)]
    settings: S,
    configuration: usize,
}

fn settings_schema(name: &str) -> Value {
    let mut annotations = json!({
        "pupil": pupil_schema(),
        "wavelengths": {"description": "System wavelength selection: all, primary, index or list (tagged by type)."}
    });
    if name != "mtf_vs_field" {
        annotations["fields"] =
            json!({"description": "System field selection: all, index or list (tagged by type)."});
    }
    match name {
        "wavefront_map" => {
            annotations["zernike_ordering"] = json!({"enum": ["fringe", "standard"], "description": "Fringe Z1 onward by default; standard uses Noll normalisation."});
            annotations["zernike_terms"] = json!({"minimum": 1, "description": "Number of fitted terms; Fringe supports at most 37."});
        }
        "psf" | "diffraction_encircled_energy" | "line_spread" => {
            annotations["method"] = json!({"enum": ["fft", "huygens"]});
            annotations["grid_size"] = json!({"minimum": 1, "description": "Image samples per axis; Huygens is limited to 1024."});
            annotations["padding"] = json!({"exclusiveMinimum": 0.0, "description": "Positive factor: image spacing = smallest wavelength × working F/# / padding."});
            if name == "psf" {
                annotations["spacing_mm"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "mm", "description": "Positive explicit spacing; null uses padding."});
                annotations["normalization"] = json!({"enum": ["strehl", "unit_energy"], "description": "strehl: ideal peak = 1; unit_energy: integral = 1, in mm^-2."});
            } else if name == "diffraction_encircled_energy" {
                annotations["max_radius_mm"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "mm", "description": "Positive maximum radius; null uses the PSF window."});
                annotations["samples"] = json!({"minimum": 2});
                annotations["center"] = json!({"description": "Tagged energy centre: {type: reference}, {type: centroid}, {type: peak}, or {type: point, x, y} in mm."});
            }
        }
        _ => {
            annotations["method"] = json!({"enum": ["direct", "fft", "huygens"]});
            annotations["detector"] = detector_schema();
            annotations["huygens"] = settings_from_default::<wa::HuygensGrid>(json!({
                "grid_size": {"minimum": 1, "maximum": wa::MAX_HUYGENS_GRID},
                "padding": {"exclusiveMinimum": 0.0, "description": "Positive image sampling factor."}
            }));
            if name == "mtf" {
                annotations["max_frequency"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "cycles/mm", "description": "Positive maximum image-space frequency; null uses the largest optical cutoff."});
                annotations["frequency_samples"] = json!({"minimum": 2});
            } else {
                annotations["frequencies"] = json!({"items": {"type": "number", "minimum": 0.0}, "unit": "cycles/mm", "description": "Image-space frequencies; empty selects cutoff × (1/8, 1/4, 1/2)."});
                if name == "mtf_vs_field" {
                    annotations["field_samples"] = json!({"minimum": 1, "maximum": 1001});
                    annotations["direction"] = tuple(
                        num("", ""),
                        2,
                        "Nonzero direction in normalised field coordinates.",
                    );
                } else {
                    annotations["focus_range_mm"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "mm", "description": "Positive half-range; null uses 4 wavelength × F/# squared."});
                    annotations["focus_samples"] = json!({"minimum": 1, "maximum": 1001});
                }
            }
        }
    }
    annotations["configuration"] = json!({"minimum": 0, "description": "System configuration index; 0 also selects the base system when there are no configurations."});
    match name {
        "wavefront_map" => settings_from_default::<Configured<wa::WavefrontSettings>>(annotations),
        "psf" => settings_from_default::<Configured<wa::PsfAnalysisSettings>>(annotations),
        "mtf" => settings_from_default::<Configured<wa::MtfSettings>>(annotations),
        "mtf_vs_field" => settings_from_default::<Configured<wa::MtfVsFieldSettings>>(annotations),
        "through_focus_mtf" => {
            settings_from_default::<Configured<wa::ThroughFocusSettings>>(annotations)
        }
        "diffraction_encircled_energy" => {
            settings_from_default::<Configured<wa::EncircledEnergySettings>>(annotations)
        }
        "line_spread" => settings_from_default::<Configured<wa::LineSpreadSettings>>(annotations),
        _ => unreachable!(),
    }
}

fn pupil_schema() -> Value {
    settings_from_default::<PupilSettings>(json!({
        "grid": {"minimum": 8, "maximum": 4096, "description": "Cell-centred exit-pupil samples per axis over [-1, 1]."},
        "rays": {"minimum": 8, "maximum": 2048, "description": "Traced entrance-pupil lattice rays per axis."},
        "supersample": {"minimum": 1, "maximum": 32},
        "reference": {"enum": ["chief", "centroid"]},
        "amplitude": {"enum": ["uniform", "transmission", "radiometric"], "description": "Relative field amplitude; radiometric includes ray density and obliquity."},
        "rim_tolerance": {"minimum": 0.0, "maximum": 0.01, "description": "Rim bisection tolerance, strictly between the bounds."}
    }))
}

fn detector_schema() -> Value {
    let mut s = settings_from_default::<mklight_wave::DetectorSettings>(json!({
        "pitch_mm": {"unit": "mm", "description": "Positive pixel pitch [x, y]."},
        "fill_factor": {"exclusiveMinimum": 0.0, "maximum": 1.0, "description": "Positive active-area fraction."},
        "aperture_mm": {"type": ["array", "null"], "unit": "mm", "description": "Explicit active widths [x, y]; null uses pitch × sqrt(fill_factor)."},
        "diffusion_sigma_mm": {"minimum": 0.0, "unit": "mm"}
    }));
    s["type"] = json!(["object", "null"]);
    s["description"] =
        json!("Optional detector pixel-aperture and diffusion MTF; null gives optical MTF only.");
    s
}

// Result schemas: arrays moved to buffers are replaced by their buffer names.
fn vector(unit: &str, description: &str) -> Value {
    let mut s = string(description);
    s["unit"] = json!(unit);
    s
}
fn field_props() -> Vec<(&'static str, Value)> {
    vec![
        ("field", int("System field index.")),
        (
            "field_value",
            xy("Field [x, y] in the system's units (degrees or mm)."),
        ),
    ]
}
fn curve_schema() -> Value {
    obj(&[
        ("frequency", num("cycles/mm", "Image-space frequency.")),
        (
            "tangential",
            vector("1", "Buffer name; MTF along the local field direction."),
        ),
        (
            "sagittal",
            vector(
                "1",
                "Buffer name; MTF perpendicular to the local field direction.",
            ),
        ),
    ])
}
fn curve_buffers(axis: &str) -> Vec<Value> {
    ["tangential_*", "sagittal_*"]
        .iter()
        .map(|name| {
            buffer(
                name,
                "f64",
                &[axis],
                "1",
                "MTF, normalised to one at zero frequency; NaN for unusable field samples.",
            )
        })
        .collect()
}

fn result_schema(name: &str) -> Value {
    match name {
        "wavefront_map" => wavefront_schema(),
        "psf" => psf_schema(),
        "mtf" => mtf_schema(),
        "mtf_vs_field" => result(
            &[
                (
                    "normalized_field",
                    vector(
                        "1",
                        "Buffer name; normalised radial field from axis to edge.",
                    ),
                ),
                (
                    "field_values",
                    vector(
                        "degrees or mm",
                        "Buffer name; specified [x, y] of each field sample.",
                    ),
                ),
                (
                    "frequencies",
                    vector("cycles/mm", "Buffer name; image-space frequencies."),
                ),
                (
                    "curves",
                    arr(
                        curve_schema(),
                        "One curve per frequency; includes the detector when supplied.",
                    ),
                ),
            ],
            {
                let mut b = curve_buffers("field_samples");
                b.extend([
                    buffer(
                        "normalized_field",
                        "f64",
                        &["field_samples"],
                        "1",
                        "Normalised radial field.",
                    ),
                    buffer(
                        "field_values",
                        "f64",
                        &["field_samples", "2"],
                        "degrees or mm",
                        "Specified field [x, y].",
                    ),
                    buffer(
                        "frequencies",
                        "f64",
                        &["frequencies"],
                        "cycles/mm",
                        "Image-space frequencies.",
                    ),
                ]);
                b
            },
        ),
        "through_focus_mtf" => {
            let mut f = field_props();
            f.push((
                "curves",
                arr(
                    curve_schema(),
                    "One curve per frequency; includes the detector when supplied.",
                ),
            ));
            let mut b = curve_buffers("focus_samples");
            b.extend([
                buffer(
                    "focus_mm",
                    "f64",
                    &["focus_samples"],
                    "mm",
                    "Image-plane shift; positive along propagation.",
                ),
                buffer(
                    "frequencies",
                    "f64",
                    &["frequencies"],
                    "cycles/mm",
                    "Image-space frequencies.",
                ),
            ]);
            result(
                &[
                    (
                        "focus_mm",
                        vector(
                            "mm",
                            "Buffer name; image-plane shift, positive along propagation.",
                        ),
                    ),
                    (
                        "frequencies",
                        vector("cycles/mm", "Buffer name; image-space frequencies."),
                    ),
                    ("fields", arr(obj(&f), "Usable selected fields.")),
                ],
                b,
            )
        }
        "diffraction_encircled_energy" => {
            let mut f = field_props();
            f.extend([
                (
                    "center_mm",
                    xy("Energy centre relative to the common reference, mm."),
                ),
                (
                    "encircled",
                    vector(
                        "1",
                        "Buffer name; energy inside a circle of radius radius_mm.",
                    ),
                ),
                (
                    "ensquared",
                    vector(
                        "1",
                        "Buffer name; energy inside a square of half-width radius_mm.",
                    ),
                ),
                (
                    "diffraction_limit",
                    vector("1", "Buffer name; aberration-free encircled energy."),
                ),
                (
                    "r50_mm",
                    nf_u("mm", "Radius enclosing 50 % of total energy."),
                ),
                (
                    "r80_mm",
                    nf_u("mm", "Radius enclosing 80 % of total energy."),
                ),
                (
                    "r90_mm",
                    nf_u("mm", "Radius enclosing 90 % of total energy."),
                ),
            ]);
            result(
                &[
                    (
                        "radius_mm",
                        vector("mm", "Buffer name; radii / square half-widths."),
                    ),
                    ("fields", arr(obj(&f), "Usable selected fields.")),
                ],
                vec![
                    buffer(
                        "radius_mm",
                        "f64",
                        &["radii"],
                        "mm",
                        "Circle radii / square half-widths.",
                    ),
                    buffer(
                        "encircled_*",
                        "f64",
                        &["radii"],
                        "1",
                        "Encircled fraction of total PSF energy.",
                    ),
                    buffer(
                        "ensquared_*",
                        "f64",
                        &["radii"],
                        "1",
                        "Ensquared fraction of total PSF energy.",
                    ),
                    buffer(
                        "diffraction_limit_*",
                        "f64",
                        &["radii"],
                        "1",
                        "Aberration-free encircled energy.",
                    ),
                ],
            )
        }
        "line_spread" => {
            let mut f = field_props();
            f.extend([
                (
                    "positions_mm",
                    vector(
                        "mm",
                        "Buffer name; positions relative to the common reference.",
                    ),
                ),
                (
                    "x",
                    vector(
                        "mm^-1",
                        "Buffer name; integral of the unit-energy PSF over y.",
                    ),
                ),
                (
                    "y",
                    vector(
                        "mm^-1",
                        "Buffer name; integral of the unit-energy PSF over x.",
                    ),
                ),
            ]);
            result(
                &[("fields", arr(obj(&f), "Usable selected fields."))],
                vec![
                    buffer(
                        "positions_mm_*",
                        "f64",
                        &["positions"],
                        "mm",
                        "Coordinates relative to the common reference.",
                    ),
                    buffer(
                        "x_*",
                        "f64",
                        &["positions"],
                        "mm^-1",
                        "Integral of PSF over y.",
                    ),
                    buffer(
                        "y_*",
                        "f64",
                        &["positions"],
                        "mm^-1",
                        "Integral of PSF over x.",
                    ),
                ],
            )
        }
        _ => unreachable!(),
    }
}

fn exit_pupil_schema() -> Value {
    obj(&[
        ("wavelength_um", num("µm", "")),
        (
            "paraxial_position_mm",
            nf_u("mm", "Paraxial exit-pupil position from the image."),
        ),
        (
            "paraxial_radius_mm",
            nf_u("mm", "Paraxial exit-pupil radius."),
        ),
        ("telecentric", boolean("Exit pupil at infinity.")),
        (
            "chief_point_mm",
            tuple(nf(), 3, "Chief image point, image-local mm."),
        ),
        (
            "chief_direction",
            tuple(num("1", ""), 3, "Chief direction, image-local."),
        ),
        (
            "exit_pupil_point_mm",
            nullable(tuple(
                nf(),
                3,
                "Exit-pupil point, image-local mm; null for telecentric beams.",
            )),
        ),
        ("reference", enumeration(&["chief", "centroid"], "")),
        (
            "reference_point_mm",
            tuple(nf(), 3, "Reference-sphere centre, image-local mm."),
        ),
        (
            "reference_radius_mm",
            nf_u(
                "mm",
                "Reference-sphere radius; infinity for telecentric beams.",
            ),
        ),
        (
            "propagation_sign",
            num("1", "Sign of propagation along image-local z."),
        ),
        ("image_index", num("1", "Image-space refractive index.")),
        ("sin_max", num("1", "Pupil angular scale.")),
        ("numerical_aperture", num("1", "Image index × sin_max.")),
        (
            "clear_area",
            num("1", "Clear area in normalised pupil coordinates."),
        ),
        (
            "transmitted_fraction",
            num("1", "Surviving entrance-pupil area fraction."),
        ),
        (
            "effective_f_number",
            num("1", "Working F/# including the clear-area scale."),
        ),
        ("lattice_rays", int("")),
        ("surviving_rays", int("")),
        ("rim_points", int("")),
    ])
}

fn wavefront_schema() -> Value {
    let mut f = field_props();
    f.extend([
        ("wavelength", int("System wavelength index.")),
        ("wavelength_um", num("µm", "")),
        ("exit_pupil", exit_pupil_schema()),
        ("grid", int("Samples per pupil axis.")),
        ("coordinates", vector("1", "Buffer name; cell centres over [-1, 1], shared by u and v.")),
        ("opd_waves", vector("waves", "Buffer name; OPL_chief minus OPL_ray at the reference sphere; NaN outside the pupil.")),
        ("amplitude", vector("1", "Buffer name; relative field amplitude, zero outside the pupil.")),
        ("rms_waves", num("waves", "Weighted RMS with piston removed.")),
        ("pv_waves", num("waves", "Peak-to-valley OPD.")),
        ("rms_waves_tilt_removed", num("waves", "Weighted RMS with piston and best-fit tilt removed.")),
        ("pv_waves_tilt_removed", num("waves", "Peak-to-valley with best-fit tilt removed.")),
        ("strehl_at_reference", num("1", "Sampled-pupil Strehl at the reference point.")),
        ("strehl_marechal", num("1", "exp(-(2π RMS_tilt_removed)^2).")),
        ("zernike", obj(&[
            ("ordering", enumeration(&["fringe", "standard"], "Fringe by default.")),
            ("coefficients", vector("waves", "Buffer name; Z1 onward, fitted to lattice rays with amplitude-squared weights.")),
        ])),
        ("zernike_residual_rms", num("waves", "Weighted fit residual RMS.")),
        ("zernike_condition", num("1", "Fit condition estimate.")),
        ("seidel", obj(&[
            ("piston", num("waves", "")), ("tilt", num("waves", "")),
            ("tilt_angle_deg", num("degrees", "From +x toward +y.")),
            ("focus", num("waves", "")), ("astigmatism", num("waves", "")),
            ("astigmatism_angle_deg", num("degrees", "From +x toward +y.")),
            ("coma", num("waves", "")), ("coma_angle_deg", num("degrees", "From +x toward +y.")),
            ("spherical", num("waves", "Pupil-edge Seidel-equivalent term.")),
        ])),
        ("rim", arr(string("Buffer name of a rim loop [points, 2] in pupil units."), "Clear-aperture loops; empty when rim tracing is disabled.")),
    ]);
    result(
        &[(
            "maps",
            arr(obj(&f), "One map per usable field and wavelength."),
        )],
        vec![
            buffer(
                "coordinates_*",
                "f64",
                &["grid"],
                "1",
                "Cell-centred normalised pupil coordinates.",
            ),
            buffer(
                "opd_waves_*",
                "f64",
                &["grid_v", "grid_u"],
                "waves",
                "OPL_chief minus OPL_ray; NaN outside the clear pupil. u along image x, v along image y.",
            ),
            buffer(
                "amplitude_*",
                "f64",
                &["grid_v", "grid_u"],
                "1",
                "Relative field amplitude; zero outside the clear pupil.",
            ),
            buffer(
                "coefficients_*",
                "f64",
                &["zernike_terms"],
                "waves",
                "Z1 onward in the reported ordering; Fringe by default.",
            ),
            buffer(
                "rim_*",
                "f64",
                &["points", "2"],
                "1",
                "Clear-aperture loop vertices [u, v].",
            ),
        ],
    )
}

fn psf_schema() -> Value {
    let mut f = field_props();
    f.extend([
        ("wavelength", nullable(int("System wavelength index; null for a polychromatic PSF."))),
        ("reference_mm", xy("Common image-local reference point, mm.")),
        ("strehl", num("1", "Peak / aberration-free peak for the same pupil and spectral weights.")),
        ("centroid_mm", xy("Centroid relative to reference_mm, mm.")),
        ("peak_mm", xy("Peak relative to reference_mm, mm.")),
        ("psf", obj(&[
            ("size", int("Image samples per axis.")),
            ("spacing_mm", num("mm", "Image-plane sample spacing.")),
            ("center_mm", xy("Grid centre relative to reference_mm, mm; coordinate i is center + (i - floor(size/2)) × spacing.")),
            ("data", vector("1 or mm^-2", "Buffer name; Strehl-normalised intensity (ideal peak 1), or unit-energy density.")),
            ("normalization", enumeration(&["strehl", "unit_energy"], "")),
            ("total_energy", num("mm^2 or 1", "Integral over the infinite image plane in the selected normalisation.")),
            ("diffraction_peak", num("1 or mm^-2", "Aberration-free peak in the selected normalisation.")),
            ("wavelength_um", nullable(num("µm", "Null for polychromatic results."))),
            ("warnings", arr(any("Tagged sampling diagnostic (mklight_wave::WaveWarning)."), "Also carried as text in RunInfo warnings.")),
        ])),
    ]);
    result(
        &[(
            "psfs",
            arr(
                obj(&f),
                "One PSF per usable field (and wavelength when monochromatic).",
            ),
        )],
        vec![buffer(
            "data_*",
            "f64",
            &["image_y", "image_x"],
            "1 or mm^-2",
            "Strehl-normalised intensity (ideal peak 1), or unit-energy density; x fastest.",
        )],
    )
}

fn mtf_schema() -> Value {
    let mut f = field_props();
    f.extend([
        (
            "tangential_direction",
            xy("Tangential unit vector in image x/y; sagittal is perpendicular."),
        ),
        (
            "tangential",
            vector("1", "Buffer name; optical tangential MTF."),
        ),
        (
            "sagittal",
            vector("1", "Buffer name; optical sagittal MTF."),
        ),
        (
            "diffraction_limit",
            vector("1", "Buffer name; aberration-free MTF of the same pupil."),
        ),
        (
            "cutoff",
            num(
                "cycles/mm",
                "Largest image-space cutoff over the selected wavelengths.",
            ),
        ),
        (
            "detector",
            nullable(obj(&[
                (
                    "tangential",
                    vector("1", "Buffer name; detector tangential MTF."),
                ),
                (
                    "sagittal",
                    vector("1", "Buffer name; detector sagittal MTF."),
                ),
                (
                    "nyquist_tangential",
                    nf_u(
                        "cycles/mm",
                        "Detector Nyquist marker along the tangential direction.",
                    ),
                ),
                (
                    "nyquist_sagittal",
                    nf_u(
                        "cycles/mm",
                        "Detector Nyquist marker along the sagittal direction.",
                    ),
                ),
            ])),
        ),
        (
            "system_tangential",
            nullable(vector("1", "Buffer name; optical × detector MTF.")),
        ),
        (
            "system_sagittal",
            nullable(vector("1", "Buffer name; optical × detector MTF.")),
        ),
    ]);
    let mut b = curve_buffers("frequency_samples");
    for name in [
        "diffraction_limit_*",
        "detector_tangential_*",
        "detector_sagittal_*",
        "system_tangential_*",
        "system_sagittal_*",
    ] {
        b.push(buffer(name, "f64", &["frequency_samples"], "1", "MTF along the frequencies axis; detector/system buffers only when a detector is supplied."));
    }
    b.push(buffer(
        "frequencies",
        "f64",
        &["frequency_samples"],
        "cycles/mm",
        "Image-space spatial frequencies.",
    ));
    result(
        &[
            (
                "frequencies",
                vector("cycles/mm", "Buffer name; image-space spatial frequencies."),
            ),
            ("fields", arr(obj(&f), "Usable selected fields.")),
            (
                "nyquist",
                nullable(xy(
                    "Detector Nyquist [x, y], cycles/mm; null without a detector. A marker, not a truncation of the optical MTF.",
                )),
            ),
        ],
        b,
    )
}

// The JSON entry points serialise the solver's known result types. Buffer
// names replace arrays in place, preserving field/wavelength/curve identity.
// Arrays are flat, or of [x, y] pairs (field values, rim loops).
fn extract(v: &mut Value, name: String, shape: Vec<usize>, buffers: &mut Vec<Buffer>) {
    let values = v.take();
    let a = values.as_array().unwrap();
    let data = if a.first().is_some_and(Value::is_array) {
        a.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(num_of))
            .collect()
    } else {
        a.iter().map(num_of).collect()
    };
    buffers.push(Buffer::new(name.clone(), shape, BufferData::F64(data)));
    *v = json!(name);
}

fn extract_vector(v: &mut Value, name: String, buffers: &mut Vec<Buffer>) {
    if !v.is_null() {
        let n = v.as_array().unwrap().len();
        extract(v, name, vec![n], buffers);
    }
}

fn extract_curves(curves: &mut Value, prefix: &str, buffers: &mut Vec<Buffer>) {
    for (i, c) in curves.as_array_mut().unwrap().iter_mut().enumerate() {
        for key in ["tangential", "sagittal"] {
            extract_vector(&mut c[key], format!("{key}_{prefix}{i}"), buffers);
        }
    }
}

fn extract_buffers(name: &str, json: &mut Value) -> Vec<Buffer> {
    let mut buffers = Vec::new();
    for key in ["frequencies", "focus_mm", "radius_mm", "normalized_field"] {
        if let Some(v) = json.get_mut(key) {
            extract_vector(v, key.into(), &mut buffers);
        }
    }
    if name == "mtf_vs_field" {
        let n = json["field_values"].as_array().unwrap().len();
        extract(
            &mut json["field_values"],
            "field_values".into(),
            vec![n, 2],
            &mut buffers,
        );
        extract_curves(&mut json["curves"], "", &mut buffers);
    } else {
        let group = match name {
            "wavefront_map" => "maps",
            "psf" => "psfs",
            _ => "fields",
        };
        for (i, f) in json[group].as_array_mut().unwrap().iter_mut().enumerate() {
            match name {
                "wavefront_map" => {
                    let n = f["grid"].as_u64().unwrap() as usize;
                    for key in ["opd_waves", "amplitude"] {
                        extract(&mut f[key], format!("{key}_{i}"), vec![n, n], &mut buffers);
                    }
                    extract_vector(
                        &mut f["coordinates"],
                        format!("coordinates_{i}"),
                        &mut buffers,
                    );
                    extract_vector(
                        &mut f["zernike"]["coefficients"],
                        format!("coefficients_{i}"),
                        &mut buffers,
                    );
                    for (j, rim) in f["rim"].as_array_mut().unwrap().iter_mut().enumerate() {
                        let n = rim.as_array().unwrap().len();
                        extract(rim, format!("rim_{i}_{j}"), vec![n, 2], &mut buffers);
                    }
                }
                "psf" => {
                    let n = f["psf"]["size"].as_u64().unwrap() as usize;
                    extract(
                        &mut f["psf"]["data"],
                        format!("data_{i}"),
                        vec![n, n],
                        &mut buffers,
                    );
                }
                "mtf" => {
                    for key in [
                        "tangential",
                        "sagittal",
                        "diffraction_limit",
                        "system_tangential",
                        "system_sagittal",
                    ] {
                        extract_vector(&mut f[key], format!("{key}_{i}"), &mut buffers);
                    }
                    if !f["detector"].is_null() {
                        for key in ["tangential", "sagittal"] {
                            extract_vector(
                                &mut f["detector"][key],
                                format!("detector_{key}_{i}"),
                                &mut buffers,
                            );
                        }
                    }
                }
                "through_focus_mtf" => {
                    extract_curves(&mut f["curves"], &format!("{i}_"), &mut buffers)
                }
                "diffraction_encircled_energy" => {
                    for key in ["encircled", "ensquared", "diffraction_limit"] {
                        extract_vector(&mut f[key], format!("{key}_{i}"), &mut buffers);
                    }
                }
                "line_spread" => {
                    for key in ["positions_mm", "x", "y"] {
                        extract_vector(&mut f[key], format!("{key}_{i}"), &mut buffers);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    buffers
}
