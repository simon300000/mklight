//! Wave analyses on a solved configuration of the CPU f64 reference tracer.
//! The solver owns the physics and job progress; this adapter publishes
//! schemas and moves its bulk arrays into the engine's typed buffers.

use std::sync::Arc;

use mklight_core::{JobContext, MkError, MkResult};
use mklight_wave::analysis::{self as wa, WaveAnalysisEntry};
use mklight_wave::traced::{ConfiguredTracer, PupilSettings, configured_tracer_job};
use serde::Serialize;
use serde_json::{Value, json};

use super::geometric::{output, take_run_info, xy};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::EngineResult;

struct Wave(&'static WaveAnalysisEntry);

/// `$run!(job)` with the solver job of the wave analysis `$name`; the CPU
/// reference ([`wa`]) and Metal (`mklight_wave::gpu::analysis`) jobs share
/// their names and result types.
macro_rules! dispatch {
    ($name:expr, $run:ident) => {
        match $name {
            "wavefront_map" => $run!(wavefront_map_job),
            "psf" => $run!(psf_analysis_job),
            "mtf" => $run!(mtf_analysis_job),
            "mtf_vs_field" => $run!(mtf_vs_field_job),
            "through_focus_mtf" => $run!(through_focus_mtf_analysis_job),
            "diffraction_encircled_energy" => $run!(diffraction_encircled_energy_job),
            "line_spread" => $run!(line_spread_job),
            _ => unreachable!("registered wave operation"),
        }
    };
}

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
    fn gpu_path(&self) -> bool {
        true
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
        let c = &ctx.subrange(0.05, 1.0);
        let Some(d) = super::gpu::device(input) else {
            return run_cpu(self.0.name, &cfg, settings, c);
        };
        let b = d.backend.as_ref();
        macro_rules! gpu {
            ($f:ident) => {{
                let mut r = mklight_wave::gpu::analysis::$f(
                    b,
                    &cfg.tracer,
                    &super::geometric::parse(settings)?,
                    c,
                )?;
                super::gpu::record_native(input, &r.run_info);
                cfg.stamp(&mut r.run_info);
                r.pack()
            }};
        }
        dispatch!(self.0.name, gpu)
    }
}

/// The CPU f64 reference job on a solved configuration, with the settings
/// parsing and `RunInfo` stamping of [`WaveAnalysisEntry::run_configured`].
fn run_cpu(
    name: &str,
    cfg: &ConfiguredTracer,
    settings: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    macro_rules! cpu {
        ($f:ident) => {{
            let mut r = wa::$f(&cfg.tracer, &cpu_settings(settings)?, ctx)?;
            cfg.stamp(&mut r.run_info);
            r.pack()
        }};
    }
    dispatch!(name, cpu)
}

/// Solver settings as the JSON entry points parse them (`null` = defaults).
fn cpu_settings<S: serde::de::DeserializeOwned>(v: &Value) -> MkResult<S> {
    let v = if v.is_null() {
        Value::Object(Default::default())
    } else {
        v.clone()
    };
    serde_json::from_value(v).map_err(|e| MkError::InvalidParam(format!("settings: {e}")))
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
            annotations["grid_size"] = json!({"minimum": 0, "description": "Image samples per axis; 0 selects from traced blur and diffraction (cap 2048); Huygens is limited to 1024."});
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
                annotations["max_frequency"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "cycles/mm", "description": "Positive maximum image-space frequency; null estimates 10% polychromatic contrast including diffraction and lateral colour."});
                annotations["frequency_samples"] = json!({"minimum": 2});
            } else {
                annotations["frequencies"] = json!({"items": {"type": "number", "minimum": 0.0}, "unit": "cycles/mm", "description": "Image-space frequencies; empty selects a round 1-2-5 base and its 2x/4x multiples within the automatic contrast range."});
                if name == "mtf_vs_field" {
                    annotations["field_samples"] = json!({"minimum": 1, "maximum": 1001});
                    annotations["direction"] = tuple(
                        num("", ""),
                        2,
                        "Nonzero direction in normalised field coordinates.",
                    );
                } else {
                    annotations["focus_range_mm"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0.0, "unit": "mm", "description": "Positive half-range; null uses max(8 wavelength × F/# squared, 4 F/# × geometric RMS blur)."});
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
        "grid": {"minimum": 0, "maximum": 4096, "description": "Cell-centred exit-pupil samples per axis over [-1, 1]; 0 selects from traced phase slope (64..1024)."},
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
                    "pupil_sampling",
                    arr(
                        pupil_sampling_schema(),
                        "One pupil sampling list per field sample; empty for skipped fields.",
                    ),
                ),
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
            f.push(("pupil_sampling", pupil_sampling_schema()));
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
                ("pupil_sampling", pupil_sampling_schema()),
                ("grid_size", int("Actual PSF image samples per axis.")),
                ("spacing_mm", num("mm", "Actual PSF image sample spacing.")),
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
                ("pupil_sampling", pupil_sampling_schema()),
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

pub(super) fn pupil_sampling_schema() -> Value {
    arr(
        obj(&[
            ("wavelength", int("System wavelength index.")),
            (
                "grid",
                int("Actual pupil samples per axis, after automatic refinements."),
            ),
        ]),
        "Actual pupil grids in selected-wavelength order.",
    )
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
        ("pupil_sampling", pupil_sampling_schema()),
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
        ("pupil_sampling", pupil_sampling_schema()),
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

// Typed packing: each solver result's bulk arrays move into buffers named
// as the result schema documents, and each array's JSON slot holds its
// buffer name, preserving field/wavelength/curve identity. Values equal a
// round trip through JSON: plain numbers serialise non-finite values as
// null (NaN), `nf` numbers keep ±inf; NaN is the canonical `f64::NAN`.

/// A plain serde `f64` read back from JSON.
fn plain(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::NAN }
}

/// An `nf` (non-finite-safe) `f64` read back from JSON.
fn nonfinite(x: f64) -> f64 {
    if x.is_nan() { f64::NAN } else { x }
}

/// Buffers taken from a typed result, with the JSON pointer of each slot.
#[derive(Default)]
struct Packer {
    buffers: Vec<Buffer>,
    slots: Vec<String>,
}

impl Packer {
    fn push(&mut self, slot: String, name: String, shape: Vec<usize>, data: Vec<f64>) {
        self.slots.push(slot);
        self.buffers
            .push(Buffer::new(name, shape, BufferData::F64(data)));
    }

    /// Move `v` out (left empty), mapping each value with `f` in place.
    fn take(v: &mut Vec<f64>, f: fn(f64) -> f64) -> Vec<f64> {
        let mut v = std::mem::take(v);
        v.iter_mut().for_each(|x| *x = f(*x));
        v
    }

    /// A plain `Vec<f64>` of shape `[len]`.
    fn vector(&mut self, slot: String, name: String, v: &mut Vec<f64>) {
        let v = Self::take(v, plain);
        self.push(slot, name, vec![v.len()], v);
    }

    /// An `nf::vec` `Vec<f64>` of shape `[len]`.
    fn nf_vector(&mut self, slot: String, name: String, v: &mut Vec<f64>) {
        let v = Self::take(v, nonfinite);
        self.push(slot, name, vec![v.len()], v);
    }

    /// Plain `[x, y]` pairs of shape `[len, 2]`.
    fn pairs(&mut self, slot: String, name: String, v: &mut Vec<[f64; 2]>) {
        let n = v.len();
        let data = std::mem::take(v).into_iter().flatten().map(plain).collect();
        self.push(slot, name, vec![n, 2], data);
    }

    /// The remaining (small) result as JSON without its `RunInfo`, whose
    /// warnings are returned, and the buffer names in their slots.
    fn finish(self, r: &impl Serialize) -> EngineResult<AnalysisOutput> {
        let mut json = serde_json::to_value(r)?;
        let warnings = take_run_info(&mut json);
        for (slot, b) in self.slots.iter().zip(&self.buffers) {
            *json.pointer_mut(slot).expect("packed array slot") = Value::String(b.name.clone());
        }
        Ok(output(json, self.buffers, warnings))
    }
}

/// A wave result whose bulk arrays become typed buffers.
trait Pack: Serialize + Sized {
    /// Move the bulk arrays into `p`, in the schema's buffer order.
    fn take_buffers(&mut self, p: &mut Packer);

    fn pack(mut self) -> EngineResult<AnalysisOutput> {
        let mut p = Packer::default();
        self.take_buffers(&mut p);
        p.finish(&self)
    }
}

impl Pack for wa::WavefrontResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        for (i, m) in self.maps.iter_mut().enumerate() {
            let n = m.grid;
            let opd = Packer::take(&mut m.opd_waves, nonfinite);
            p.push(
                format!("/maps/{i}/opd_waves"),
                format!("opd_waves_{i}"),
                vec![n, n],
                opd,
            );
            let amplitude = Packer::take(&mut m.amplitude, plain);
            p.push(
                format!("/maps/{i}/amplitude"),
                format!("amplitude_{i}"),
                vec![n, n],
                amplitude,
            );
            p.vector(
                format!("/maps/{i}/coordinates"),
                format!("coordinates_{i}"),
                &mut m.coordinates,
            );
            // A few fitted terms, private to the set: copied, not moved.
            let c: Vec<f64> = m
                .zernike
                .coefficients()
                .iter()
                .copied()
                .map(plain)
                .collect();
            p.push(
                format!("/maps/{i}/zernike/coefficients"),
                format!("coefficients_{i}"),
                vec![c.len()],
                c,
            );
            for (j, rim) in m.rim.iter_mut().enumerate() {
                p.pairs(format!("/maps/{i}/rim/{j}"), format!("rim_{i}_{j}"), rim);
            }
        }
    }
}

impl Pack for wa::PsfResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        for (i, f) in self.psfs.iter_mut().enumerate() {
            let n = f.psf.size;
            let data = Packer::take(&mut f.psf.data, plain);
            p.push(
                format!("/psfs/{i}/psf/data"),
                format!("data_{i}"),
                vec![n, n],
                data,
            );
        }
    }
}

impl Pack for wa::WaveMtfResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        p.vector(
            "/frequencies".into(),
            "frequencies".into(),
            &mut self.frequencies,
        );
        for (i, f) in self.fields.iter_mut().enumerate() {
            for (key, v) in [
                ("tangential", Some(&mut f.tangential)),
                ("sagittal", Some(&mut f.sagittal)),
                ("diffraction_limit", Some(&mut f.diffraction_limit)),
                ("system_tangential", f.system_tangential.as_mut()),
                ("system_sagittal", f.system_sagittal.as_mut()),
            ] {
                if let Some(v) = v {
                    p.vector(format!("/fields/{i}/{key}"), format!("{key}_{i}"), v);
                }
            }
            if let Some(d) = &mut f.detector {
                for (key, v) in [
                    ("tangential", &mut d.tangential),
                    ("sagittal", &mut d.sagittal),
                ] {
                    p.vector(
                        format!("/fields/{i}/detector/{key}"),
                        format!("detector_{key}_{i}"),
                        v,
                    );
                }
            }
        }
    }
}

impl Pack for wa::MtfVsFieldResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        p.vector(
            "/frequencies".into(),
            "frequencies".into(),
            &mut self.frequencies,
        );
        p.vector(
            "/normalized_field".into(),
            "normalized_field".into(),
            &mut self.normalized_field,
        );
        p.pairs(
            "/field_values".into(),
            "field_values".into(),
            &mut self.field_values,
        );
        for (j, c) in self.curves.iter_mut().enumerate() {
            p.nf_vector(
                format!("/curves/{j}/tangential"),
                format!("tangential_{j}"),
                &mut c.tangential,
            );
            p.nf_vector(
                format!("/curves/{j}/sagittal"),
                format!("sagittal_{j}"),
                &mut c.sagittal,
            );
        }
    }
}

impl Pack for wa::ThroughFocusResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        p.vector(
            "/frequencies".into(),
            "frequencies".into(),
            &mut self.frequencies,
        );
        p.vector("/focus_mm".into(), "focus_mm".into(), &mut self.focus_mm);
        for (i, f) in self.fields.iter_mut().enumerate() {
            for (j, c) in f.curves.iter_mut().enumerate() {
                let at = format!("/fields/{i}/curves/{j}");
                p.vector(
                    format!("{at}/tangential"),
                    format!("tangential_{i}_{j}"),
                    &mut c.tangential,
                );
                p.vector(
                    format!("{at}/sagittal"),
                    format!("sagittal_{i}_{j}"),
                    &mut c.sagittal,
                );
            }
        }
    }
}

impl Pack for wa::WaveEncircledEnergyResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        p.vector("/radius_mm".into(), "radius_mm".into(), &mut self.radius_mm);
        for (i, f) in self.fields.iter_mut().enumerate() {
            for (key, v) in [
                ("encircled", &mut f.encircled),
                ("ensquared", &mut f.ensquared),
                ("diffraction_limit", &mut f.diffraction_limit),
            ] {
                p.vector(format!("/fields/{i}/{key}"), format!("{key}_{i}"), v);
            }
        }
    }
}

impl Pack for wa::LineSpreadResult {
    fn take_buffers(&mut self, p: &mut Packer) {
        for (i, f) in self.fields.iter_mut().enumerate() {
            for (key, v) in [
                ("positions_mm", &mut f.positions_mm),
                ("x", &mut f.x),
                ("y", &mut f.y),
            ] {
                p.vector(format!("/fields/{i}/{key}"), format!("{key}_{i}"), v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::samples;
    use mklight_wave::traced::configured_tracer;

    use super::super::geometric::num_of;
    use super::*;

    // The former route, kept as the reference: the whole result through a
    // `serde_json::Value` tree, then the arrays walked back into buffers.
    fn json_route(name: &str, mut json: Value) -> AnalysisOutput {
        let warnings = take_run_info(&mut json);
        let buffers = extract_buffers(name, &mut json);
        output(json, buffers, warnings)
    }

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

    fn assert_same(what: &str, new: &AnalysisOutput, old: &AnalysisOutput) {
        assert_eq!(new.json, old.json, "{what}: json");
        assert_eq!(new.warnings, old.warnings, "{what}: warnings");
        assert_eq!(new.buffers.len(), old.buffers.len(), "{what}: buffer count");
        for (a, b) in new.buffers.iter().zip(&old.buffers) {
            assert_eq!((&a.name, &a.shape), (&b.name, &b.shape), "{what}");
            let (BufferData::F64(x), BufferData::F64(y)) = (&a.data, &b.data) else {
                panic!("{what}: {} is not f64", a.name);
            };
            let diff = x
                .iter()
                .zip(y)
                .position(|(p, q)| p.to_bits() != q.to_bits());
            assert_eq!(diff, None, "{what}: {} differs at this element", a.name);
        }
    }

    fn small_settings(name: &str) -> Value {
        let mut s = json!({
            "configuration": 0,
            "wavelengths": {"type": "primary"},
            "pupil": {"grid": 16, "rays": 13, "supersample": 2}
        });
        if matches!(name, "mtf" | "mtf_vs_field" | "through_focus_mtf") {
            s["detector"] = json!({"pitch_mm": [0.005, 0.01], "diffusion_sigma_mm": 0.001});
        }
        match name {
            "psf" | "line_spread" | "diffraction_encircled_energy" => {
                s["grid_size"] = json!(32);
                s["padding"] = json!(2.0);
                if name == "diffraction_encircled_energy" {
                    s["samples"] = json!(7);
                }
            }
            "mtf" => s["frequency_samples"] = json!(9),
            "mtf_vs_field" => s["field_samples"] = json!(3),
            "through_focus_mtf" => s["focus_samples"] = json!(3),
            _ => {}
        }
        s
    }

    fn tracers() -> Vec<(&'static str, ConfiguredTracer)> {
        let lib = Arc::new(MaterialLibrary::with_builtin());
        [
            ("cooke_triplet", samples::cooke_triplet()),
            ("newtonian_telescope", samples::newtonian_telescope()),
        ]
        .into_iter()
        .map(|(label, sys)| (label, configured_tracer(&sys, lib.clone(), 0).unwrap()))
        .collect()
    }

    #[test]
    fn typed_packing_matches_the_json_route_on_cpu() {
        let ctx = JobContext::new();
        for (label, cfg) in tracers() {
            for entry in wa::ANALYSES {
                let s = small_settings(entry.name);
                let old = json_route(entry.name, entry.run_configured(&cfg, &s, &ctx).unwrap());
                let new = run_cpu(entry.name, &cfg, &s, &ctx).unwrap();
                assert!(!new.buffers.is_empty(), "{label} {}", entry.name);
                assert_same(&format!("{label} {}", entry.name), &new, &old);
            }
        }
    }

    /// Non-canonical NaN, ±inf and -0 at the front of a bulk array.
    fn poison(v: &mut [f64]) {
        let special = [
            f64::from_bits(0x7ff8_0000_dead_beef),
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            -f64::NAN,
        ];
        for (x, s) in v.iter_mut().zip(special) {
            *x = s;
        }
    }

    fn check_poisoned<R: Pack + Clone>(name: &str, mut r: R, poison_all: impl Fn(&mut R)) {
        poison_all(&mut r);
        let old = json_route(name, serde_json::to_value(r.clone()).unwrap());
        assert_same(name, &r.pack().unwrap(), &old);
    }

    #[test]
    fn typed_packing_matches_the_json_route_for_non_finite_values() {
        let ctx = JobContext::new();
        let (_, cfg) = tracers().swap_remove(0);
        let t = &cfg.tracer;
        macro_rules! job {
            ($f:ident, $name:literal) => {
                wa::$f(t, &cpu_settings(&small_settings($name)).unwrap(), &ctx).unwrap()
            };
        }
        check_poisoned(
            "wavefront_map",
            job!(wavefront_map_job, "wavefront_map"),
            |r| {
                for m in &mut r.maps {
                    poison(&mut m.opd_waves);
                    poison(&mut m.amplitude);
                    poison(&mut m.coordinates);
                    for rim in &mut m.rim {
                        poison(rim.as_flattened_mut());
                    }
                }
            },
        );
        check_poisoned("psf", job!(psf_analysis_job, "psf"), |r| {
            r.psfs.iter_mut().for_each(|f| poison(&mut f.psf.data))
        });
        check_poisoned("mtf", job!(mtf_analysis_job, "mtf"), |r| {
            poison(&mut r.frequencies);
            for f in &mut r.fields {
                poison(&mut f.tangential);
                poison(&mut f.sagittal);
                poison(&mut f.diffraction_limit);
                f.system_tangential = Some(vec![f64::INFINITY, 0.5]);
                f.system_sagittal = None;
                if let Some(d) = &mut f.detector {
                    poison(&mut d.tangential);
                    poison(&mut d.sagittal);
                }
            }
        });
        check_poisoned(
            "mtf_vs_field",
            job!(mtf_vs_field_job, "mtf_vs_field"),
            |r| {
                poison(&mut r.frequencies);
                poison(&mut r.normalized_field);
                poison(r.field_values.as_flattened_mut());
                for c in &mut r.curves {
                    poison(&mut c.tangential);
                    poison(&mut c.sagittal);
                }
            },
        );
        check_poisoned(
            "through_focus_mtf",
            job!(through_focus_mtf_analysis_job, "through_focus_mtf"),
            |r| {
                poison(&mut r.frequencies);
                poison(&mut r.focus_mm);
                for c in r.fields.iter_mut().flat_map(|f| &mut f.curves) {
                    poison(&mut c.tangential);
                    poison(&mut c.sagittal);
                }
            },
        );
        check_poisoned(
            "diffraction_encircled_energy",
            job!(
                diffraction_encircled_energy_job,
                "diffraction_encircled_energy"
            ),
            |r| {
                poison(&mut r.radius_mm);
                for f in &mut r.fields {
                    poison(&mut f.encircled);
                    poison(&mut f.ensquared);
                    poison(&mut f.diffraction_limit);
                }
            },
        );
        check_poisoned("line_spread", job!(line_spread_job, "line_spread"), |r| {
            for f in &mut r.fields {
                poison(&mut f.positions_mm);
                poison(&mut f.x);
                poison(&mut f.y);
            }
        });
    }

    /// Release timing of the result packing alone (old JSON route vs typed),
    /// on a dense PSF and wavefront map of the double Gauss:
    /// cargo test --release -p mklight-engine --lib packing_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn packing_cost() {
        fn time<R: Pack + Clone>(what: &str, name: &str, r: R) {
            let median = |mut t: Vec<f64>| {
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            };
            let (mut old, mut new) = (vec![], vec![]);
            for _ in 0..7 {
                let c = r.clone();
                let t = std::time::Instant::now();
                std::hint::black_box(json_route(name, serde_json::to_value(c).unwrap()));
                old.push(t.elapsed().as_secs_f64() * 1e3);
                let c = r.clone();
                let t = std::time::Instant::now();
                std::hint::black_box(c.pack().unwrap());
                new.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "PACKING_COST {what}: json route {:.2} ms, typed {:.3} ms (medians of 7)",
                median(old),
                median(new)
            );
        }
        let lib = Arc::new(MaterialLibrary::with_builtin());
        let cfg = configured_tracer(&samples::double_gauss(), lib, 0).unwrap();
        let ctx = JobContext::new();
        let s = json!({"grid_size": 1024, "pupil": {"grid": 256, "rays": 65}});
        let psf = wa::psf_analysis_job(&cfg.tracer, &cpu_settings(&s).unwrap(), &ctx).unwrap();
        time("psf 3 fields x 1024^2", "psf", psf);
        let s = json!({"pupil": {"grid": 512, "rays": 129}});
        let wf = wa::wavefront_map_job(&cfg.tracer, &cpu_settings(&s).unwrap(), &ctx).unwrap();
        time("wavefront 3 fields x 512^2", "wavefront_map", wf);
    }
}
