//! The geometric analyses of `mklight-analysis` (plus `ray_grid`) behind the
//! engine's `Analysis` interface.
//!
//! Each analysis is a [`Geo`] record: settings schema (derived from the
//! crate's settings `Default`, so defaults cannot drift), result schema,
//! and a run function that parses the normalised settings into the crate's
//! settings struct, builds the reference tracer from the cached resolved
//! system, calls the crate and reshapes its serde result into the engine's
//! result shape (`docs/RESULTS.md`):
//!
//! * the crate's embedded `run_info` is removed (the envelope carries the
//!   engine's `RunInfo`; its warnings are kept);
//! * ray clouds, point grids and scene geometry move into typed buffers;
//!   curves sampled along one axis (fans, field curves, energy, MTF) stay in
//!   the JSON.
//!
//! The ray-grid analyses (`spot_diagram`, `encircled_energy`,
//! `geometric_mtf`, `best_focus`, `ray_grid`) trace through
//! complete resident adapters, preserving the public result packing.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_analysis::{
    curvature, energy, fans, footprint_overlay, illumination, layout, seidel, spot, system_data,
    util,
};
use mklight_core::JobContext;
use mklight_trace::{Grid, SequentialTracer, TraceOptions, trace_grid_job};

use super::gpu;
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};

type RunFn =
    fn(&AnalysisInput<'_>, &SequentialTracer, &Value, &JobContext) -> EngineResult<AnalysisOutput>;

/// One geometric analysis (module docs).
pub struct Geo {
    pub name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub settings: fn() -> Value,
    pub result: fn() -> Value,
    pub run: RunFn,
    pub seed: fn(&Value) -> Option<u64>,
}

impl Analysis for Geo {
    fn name(&self) -> &str {
        self.name
    }
    fn title(&self) -> &str {
        self.title
    }
    fn category(&self) -> &str {
        "geometric"
    }
    fn description(&self) -> &str {
        self.description
    }
    fn settings_schema(&self) -> Value {
        (self.settings)()
    }
    fn result_schema(&self) -> Value {
        (self.result)()
    }
    fn result_version(&self) -> u32 {
        if matches!(
            self.name,
            "seidel"
                | "field_curvature_distortion"
                | "relative_illumination"
                | "lateral_colour"
                | "longitudinal_aberration"
        ) {
            2
        } else {
            1
        }
    }
    fn gpu_path(&self) -> bool {
        gpu::GPU_ANALYSES.contains(&self.name)
    }
    fn seed(&self, settings: &Value) -> Option<u64> {
        (self.seed)(settings)
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let tracer = SequentialTracer::new(input.resolved()?)?;
        // Analyses with a field selection get the default axis fields here,
        // so the GPU and ray-grid paths see the same fields as the crate.
        match settings.get("fields") {
            Some(fields) => {
                let tracer = util::default_fields(&tracer, &parse(fields)?)?;
                (self.run)(input, &tracer, settings, ctx)
            }
            None => (self.run)(input, &tracer, settings, ctx),
        }
    }
}

/// Register every geometric analysis.
pub fn register_all(r: &mut AnalysisRegistry) {
    for a in all() {
        r.register(Arc::new(a));
    }
}

// ---------------------------------------------------------------------------
// Helpers

pub(crate) fn parse<T: DeserializeOwned>(s: &Value) -> EngineResult<T> {
    serde_json::from_value(s.clone()).map_err(|e| EngineError::InvalidSettings(e.to_string()))
}

/// A solver result as JSON without its embedded `RunInfo` (key `run_info`
/// or `run`), whose warnings are returned for the engine's `RunInfo`.
pub(crate) fn crate_json<R: Serialize>(r: &R) -> EngineResult<(Value, Vec<String>)> {
    let mut v = serde_json::to_value(r)?;
    let warnings = take_run_info(&mut v);
    Ok((v, warnings))
}

/// Remove the embedded `RunInfo` of a solver result already in JSON
/// ([`crate_json`]), returning its warnings.
pub(crate) fn take_run_info(v: &mut Value) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Value::Object(m) = v {
        for key in ["run_info", "run"] {
            if let Some(ri) = m.remove(key)
                && let Some(w) = ri.get("warnings").and_then(Value::as_array)
            {
                warnings.extend(w.iter().filter_map(Value::as_str).map(String::from));
            }
        }
    }
    warnings
}

pub(crate) fn output(json: Value, buffers: Vec<Buffer>, warnings: Vec<String>) -> AnalysisOutput {
    AnalysisOutput {
        json,
        buffers,
        warnings,
    }
}

/// `[x, y]` pairs from JSON (`null`/strings become NaN).
fn pairs(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| {
            a.iter()
                .flat_map(|p| [num_of(&p[0]), num_of(&p[1])])
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn num_of(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        _ => f64::NAN,
    }
}

fn no_seed(_: &Value) -> Option<u64> {
    None
}

/// The seed of a random or dithered pupil pattern (`None` otherwise).
fn pattern_seed(s: &Value) -> Option<u64> {
    matches!(s["pattern"].as_str(), Some("random" | "dithered"))
        .then(|| s["seed"].as_u64())
        .flatten()
}

// --- settings annotations shared by several analyses

fn a_fields() -> Value {
    json!({"description": "Field selection: {\"type\": \"all\"}, {\"type\": \"index\", \"index\": i} or {\"type\": \"list\", \"indices\": [i, …]} (system field indices; all also adds missing axis fields for non-rotational systems, with analysis-local indices and reported coordinates)."})
}
fn a_wavelengths() -> Value {
    json!({"description": "Wavelength selection: {\"type\": \"all\"}, {\"type\": \"primary\"}, {\"type\": \"index\", \"index\": i} or {\"type\": \"list\", \"indices\": [i, …]} (system wavelength indices)."})
}
fn a_pattern() -> Value {
    json!({"enum": ["hexapolar", "square", "random", "dithered"],
           "description": "Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1)."})
}
fn a_rings() -> Value {
    json!({"minimum": 1, "maximum": 512, "description": "Density of the pupil pattern (rings, or grid half-count)."})
}
fn a_seed() -> Value {
    json!({"minimum": 0, "description": "Philox seed of the random and dithered patterns."})
}
fn a_opt_wl() -> Value {
    json!({"type": ["integer", "null"], "minimum": 0,
           "description": "System wavelength index; null = the primary wavelength."})
}
fn a_reference() -> Value {
    json!({"enum": ["chief", "centroid"],
           "description": "Reference point: the primary-wavelength chief ray or the polychromatic centroid."})
}
fn a_count(lo: u64, hi: u64, description: &str) -> Value {
    json!({"minimum": lo, "maximum": hi, "description": description})
}

fn sampling_annotations(extra: Value) -> Value {
    let mut a = json!({
        "pattern": a_pattern(), "rings": a_rings(), "fields": a_fields(),
        "wavelengths": a_wavelengths(), "seed": a_seed()
    });
    if let (Value::Object(m), Value::Object(e)) = (&mut a, extra) {
        m.extend(e);
    }
    a
}

// --- result fragments

pub(crate) fn xy(description: &str) -> Value {
    tuple(nf(), 2, description)
}
fn field_value() -> Value {
    xy("Field value as specified (degrees or mm, x and y).")
}
fn nf_vec(unit: &str, description: &str) -> Value {
    let mut v = arr(nf(), description);
    if !unit.is_empty() {
        v["unit"] = json!(unit);
    }
    v
}
fn wl_list() -> Value {
    arr(int(""), "Selected system wavelength indices.")
}
fn wl_um() -> Value {
    arr(
        num("µm", ""),
        "All system wavelengths (µm), indexed by system wavelength index.",
    )
}

// ---------------------------------------------------------------------------
// The analyses

fn all() -> Vec<Geo> {
    vec![
        Geo {
            name: "first_order",
            title: "First-order properties",
            description: "Paraxial first-order data (focal lengths, pupils, magnifications, \
                          F-numbers, …) as a keyed report table per selected wavelength.",
            settings: || {
                settings_from_default::<system_data::SystemDataSettings>(
                    json!({"wavelength": a_wavelengths()}),
                )
            },
            result: first_order_schema,
            run: |_, t, s, _| {
                let r = system_data::system_data(t, &parse(s)?)?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "layout",
            title: "Layout",
            description: "The viewport scene in global coordinates: surface profiles, rings \
                          and rims, element outlines, stop, image, axis and ray paths as \
                          polylines in typed buffers.",
            settings: || {
                settings_from_default::<layout::LayoutSettings>(json!({
                    "rays_per_field": a_count(1, 10_000, "Rays per fan, evenly spaced across the pupil (1: the chief ray)."),
                    "fields": a_fields(), "wavelengths": a_wavelengths(),
                    "profile_samples": a_count(2, 100_000, "Points per surface profile."),
                    "section": {"enum": ["yz", "xz"], "description": "Profile plane; yz by default. Section flags use its global plane."},
                    "mesh_rings": a_count(0, 100, "Radial subdivisions of the optional 3-D surface mesh; 0 disables it. Angular subdivisions use ring_samples."),
                    "include_3d": {"description": "Also produce out-of-section geometry (sagittal profiles, rings, rims, sagittal fans)."},
                    "rings": a_count(1, 1000, "Concentric rings per surface in 3-D (the last one is the rim)."),
                    "ring_samples": a_count(3, 100_000, "Points per ring."),
                    "edge_margin": {"minimum": 0.0, "unit": "mm", "description": "Mechanical margin beyond the clear semi-diameter where a surface has no mechanical semi-diameter."},
                    "lead_in": {"type": ["number", "null"], "minimum": 0.0, "unit": "mm", "description": "Ray length drawn before the first surface for an object at infinity; null = a quarter of the vertex track."}
                }))
            },
            result: layout_schema,
            run: run_layout,
            seed: no_seed,
        },
        Geo {
            name: "spot_diagram",
            title: "Spot diagram",
            description: "Ray intercepts on the image surface per field and wavelength: \
                          centroid, RMS and GEO radii about centroid and chief ray, \
                          polychromatic statistics, vignetted fraction, Airy radius, optional \
                          through-focus. Ray points in typed buffers. GPU path.",
            settings: || {
                settings_from_default::<spot::SpotDiagramSettings>(json!({
                    "pattern": a_pattern(), "rings": a_rings(), "fields": a_fields(),
                    "wavelengths": a_wavelengths(), "reference": a_reference(), "seed": a_seed(),
                    "defocus": {"unit": "mm", "description": "Defocus from the image surface along its local +z."},
                    "through_focus": {"type": ["object", "null"], "description": "{\"min\": mm, \"max\": mm, \"steps\": n} or null."},
                    "include_points": {"description": "Return the individual ray points (buffers `points`, `point_field`, `point_wavelength`)."}
                }))
            },
            result: spot_schema,
            run: run_spot,
            seed: pattern_seed,
        },
        Geo {
            name: "ray_fan",
            title: "Ray aberration fans",
            description: "Transverse ray aberration εy (tangential fan, P_y) and εx \
                          (sagittal fan, P_x) on the image surface versus normalised pupil \
                          coordinate, per field and wavelength (mm).",
            settings: fan_settings,
            result: || fan_schema(true),
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::ray_fan_job(b, t, &settings, c),
                    fans::ray_fan_job(t, &settings, c)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "opd_fan",
            title: "OPD fans",
            description: "Optical path difference (waves) against the exit-pupil reference \
                          sphere versus normalised pupil coordinate, tangential and sagittal.",
            settings: fan_settings,
            result: || fan_schema(false),
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::opd_fan_job(b, t, &settings, c),
                    fans::opd_fan_job(t, &settings, c)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "field_curvature_distortion",
            title: "Field curvature and distortion",
            description: "Tangential and sagittal focus shifts along the real chief ray \
                          (generalised Coddington, cross-checked by close rays) and distortion \
                          (paraxial, f·tanθ, f·θ) versus field.",
            settings: || {
                settings_from_default::<curvature::FieldCurvatureSettings>(json!({
                    "field_axes": {"enum": ["auto", "x", "y", "both"], "description": "Auto: y for rotational systems; both axes otherwise. Only rotational symmetry permits positive-y sampling alone."},
                    "field_samples": a_count(2, 10_000, "Samples per selected axis, signed where symmetry requires it."),
                    "wavelengths": a_wavelengths(),
                    "close_ray_step": {"minimum": 1e-9, "maximum": 0.5, "description": "Normalised pupil step of the close-ray cross-check."}
                }))
            },
            result: field_curvature_schema,
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::field_curvature_job(b, t, &settings, c),
                    curvature::field_curvature_job(t, &settings, c)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "grid_distortion",
            title: "Grid distortion",
            description: "Chief rays of a square grid of normalised field points against \
                          their paraxial image points. Grid in typed buffers.",
            settings: || {
                settings_from_default::<curvature::GridDistortionSettings>(json!({
                    "grid_size": a_count(2, 1001, "Grid points per side."),
                    "half_width": {"minimum": 0.0, "maximum": 1.0, "description": "Half-width of the normalised field square."},
                    "field_center": {"description": "Normalised field [hx,hy] at the grid centre."},
                    "wavelength": a_opt_wl()
                }))
            },
            result: grid_distortion_schema,
            run: run_grid_distortion,
            seed: no_seed,
        },
        Geo {
            name: "seidel",
            title: "Seidel aberration sums",
            description: "Third-order (Seidel) coefficients S_I…S_V and chromatic C_L, C_T per \
                          surface and summed, Petzval sum and radius, wavefront coefficients.",
            settings: || {
                settings_from_default::<seidel::SeidelSettings>(json!({
                    "wavelength": a_opt_wl(),
                    "short_wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "Short wavelength of δn; null = the shortest system wavelength."},
                    "long_wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "Long wavelength of δn; null = the longest system wavelength."}
                }))
            },
            result: seidel_schema,
            run: |_, t, s, _| {
                let r = seidel::seidel(t, &parse(s)?)?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "longitudinal_aberration",
            title: "Longitudinal aberration",
            description: "On-axis zonal rays: where each crosses the axis, relative to the \
                          primary paraxial focus, versus pupil zone, per wavelength.",
            settings: || {
                settings_from_default::<seidel::LongitudinalSettings>(json!({
                    "axes": {"enum": ["auto", "x", "y", "both"], "description": "Auto traces y in rotational systems, both axes otherwise."},
                    "pupil_samples": a_count(2, 10_000, "Pupil zones: 0 to 1 for rotational systems, -1 to 1 otherwise; selected x/y axes."),
                    "wavelengths": a_wavelengths()
                }))
            },
            result: longitudinal_schema,
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::longitudinal_aberration_job(b, t, &settings, c),
                    seidel::longitudinal_aberration(t, &settings)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "chromatic_focal_shift",
            title: "Chromatic focal shift",
            description: "Paraxial focus versus wavelength over a sweep, relative to the \
                          primary wavelength's focus.",
            settings: || {
                settings_from_default::<seidel::ChromaticFocalShiftSettings>(json!({
                    "samples": a_count(2, 100_000, "Wavelength samples of the sweep."),
                    "min_um": {"type": ["number", "null"], "minimum": 0.1, "unit": "µm", "description": "Sweep start; null = the shortest system wavelength."},
                    "max_um": {"type": ["number", "null"], "minimum": 0.1, "unit": "µm", "description": "Sweep end; null = the longest system wavelength."}
                }))
            },
            result: chromatic_focal_shift_schema,
            run: |_, t, s, _| {
                let r = seidel::chromatic_focal_shift(t, &parse(s)?)?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "lateral_colour",
            title: "Lateral colour",
            description: "Real chief-ray image heights per wavelength versus field, relative \
                          to the primary wavelength, and the short−long difference.",
            settings: || {
                settings_from_default::<seidel::LateralColourSettings>(json!({
                    "field_axes": {"enum": ["auto", "x", "y", "both"], "description": "Auto: y for rotational systems; both axes otherwise. Only rotational symmetry permits positive-y sampling alone."},
                    "field_samples": a_count(2, 10_000, "Samples per selected axis, signed where symmetry requires it."),
                    "short_wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "null = the shortest system wavelength."},
                    "long_wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "null = the longest system wavelength."}
                }))
            },
            result: lateral_colour_schema,
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::lateral_colour_job(b, t, &settings, c),
                    seidel::lateral_colour_job(t, &settings, c)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "relative_illumination",
            title: "Relative illumination",
            description: "Image irradiance of a uniform Lambertian object versus field, from \
                          the projected solid angle of the real exit pupil seen from each \
                          image point (real ray aiming), with the vignetted fraction.",
            settings: || {
                settings_from_default::<illumination::RelativeIlluminationSettings>(json!({
                    "field_axes": {"enum": ["auto", "x", "y", "both"], "description": "Auto: y for rotational systems; both axes otherwise. Only rotational symmetry permits positive-y sampling alone."},
                    "field_samples": a_count(2, 10_000, "Samples per selected axis, signed where symmetry requires it."),
                    "wavelength": a_opt_wl(),
                    "radial_samples": a_count(1, 1000, "Gauss–Legendre nodes in the pupil radius."),
                    "azimuth_samples": a_count(3, 10_000, "Trapezoidal nodes in the pupil azimuth.")
                }))
            },
            result: relative_illumination_schema,
            run: |input, t, s, c| {
                let settings = parse(s)?;
                let r = gpu::native_job!(
                    input,
                    |b| mklight_analysis::gpu::relative_illumination_job(b, t, &settings, c),
                    illumination::relative_illumination_job(t, &settings, c)
                )?;
                let (v, w) = crate_json(&r)?;
                Ok(output(v, vec![], w))
            },
            seed: no_seed,
        },
        Geo {
            name: "footprint",
            title: "Surface footprint",
            description: "Hit points of a pupil pattern on one surface (local frame), with the \
                          points the surface's aperture clips, plus global hit points and convex \
                          outlines on every surface for a 3-D overlay. Points in typed buffers.",
            settings: || {
                settings_from_default::<illumination::FootprintSettings>(sampling_annotations(
                    json!({
                        "surface": a_count(1, 100_000, "Surface index (1 … image).")
                    }),
                ))
            },
            result: footprint_schema,
            run: run_footprint,
            seed: pattern_seed,
        },
        Geo {
            name: "encircled_energy",
            title: "Encircled energy",
            description: "Geometric encircled and ensquared energy versus radius per field, \
                          with r50/r80/r90. GPU path.",
            settings: || {
                settings_from_default::<energy::EncircledEnergySettings>(sampling_annotations(
                    json!({
                        "reference": a_reference(),
                        "defocus": {"unit": "mm", "description": "Defocus from the image surface."},
                        "samples": a_count(2, 100_001, "Radius samples from 0 to max_radius."),
                        "max_radius": {"type": ["number", "null"], "minimum": 0.0, "unit": "mm", "description": "Largest radius; null = the largest GEO radius over the fields."}
                    }),
                ))
            },
            result: encircled_energy_schema,
            run: run_encircled_energy,
            seed: pattern_seed,
        },
        Geo {
            name: "geometric_mtf",
            title: "Geometric MTF",
            description: "Geometric (ray-based) MTF, tangential and sagittal, per field. No \
                          diffraction: compare with the wave-optics MTF. GPU path.",
            settings: || {
                settings_from_default::<energy::GeometricMtfSettings>(sampling_annotations(json!({
                    "defocus": {"unit": "mm", "description": "Defocus from the image surface."},
                    "max_frequency": {"type": ["number", "null"], "minimum": 0.0, "unit": "cycles/mm", "description": "Highest frequency; null = the diffraction cutoff 1/(λN_w) at the primary wavelength."},
                    "frequency_samples": a_count(2, 100_000, "Frequency samples from 0 to max_frequency.")
                })))
            },
            result: geometric_mtf_schema,
            run: run_geometric_mtf,
            seed: pattern_seed,
        },
        Geo {
            name: "best_focus",
            title: "Best focus",
            description: "Defocus minimising the field-weighted mean RMS spot radius (Brent), \
                          with per-field RMS at best and current focus. GPU path.",
            settings: || {
                settings_from_default::<energy::BestFocusSettings>(sampling_annotations(json!({
                    "reference": a_reference(),
                    "range": {"type": ["array", "null"], "items": {"type": "number"}, "unit": "mm", "description": "Search range [min, max] of the defocus; null = automatic."},
                    "tolerance": {"minimum": 0.0, "unit": "mm", "description": "Absolute tolerance on the defocus."},
                    "max_evaluations": a_count(3, 100_000, "Merit evaluations allowed."),
                    "through_focus_samples": a_count(0, 100_000, "Samples of the through-focus merit curve (0: none).")
                })))
            },
            result: best_focus_schema,
            run: run_best_focus,
            seed: pattern_seed,
        },
        Geo {
            name: "ray_grid",
            title: "Ray grid",
            description: "Image-surface hits (local frame) and directions of a fields × \
                          wavelengths × pupil ray grid, plus the chief rays, as typed buffers \
                          (NaN for rays that did not arrive). GPU path.",
            settings: || settings_from_default::<RayGridSettings>(sampling_annotations(json!({}))),
            result: ray_grid_schema,
            run: run_ray_grid,
            seed: pattern_seed,
        },
    ]
}

// ---------------------------------------------------------------------------
// first_order

pub(super) fn paraxial_xy_schema() -> Value {
    obj(&[
        (
            "symmetry",
            enumeration(
                &["rotational", "meridional", "none"],
                "Geometry, placements and apertures; fields are sampling requests.",
            ),
        ),
        ("primary_wavelength", int("System wavelength index.")),
        (
            "squeeze_definition",
            string(
                "Signed EFL_y / EFL_x; unit-index referenced. Mixed xy power gives diagonal projections, not principal focal lengths.",
            ),
        ),
        (
            "warnings",
            arr(
                string(""),
                "Limitations of the nominal-vertex unfolded sweeps.",
            ),
        ),
        (
            "wavelengths",
            arr(
                obj(&[
                    ("wavelength", int("System wavelength index.")),
                    ("wavelength_um", num("µm", "")),
                    ("efl_x", nf_u("mm", "X-axis effective focal length.")),
                    (
                        "efl_y",
                        nf_u("mm", "Y-axis effective focal length (legacy EFL)."),
                    ),
                    (
                        "bfl_x",
                        nf_u("mm", "X-axis back focus from the last optical surface."),
                    ),
                    (
                        "bfl_y",
                        nf_u("mm", "Y-axis back focus from the last optical surface."),
                    ),
                    ("squeeze", nf_u("1", "Signed EFL_y / EFL_x.")),
                    (
                        "doe_power",
                        arr(
                            obj(&[
                                ("surface", int("")),
                                (
                                    "power",
                                    num(
                                        "1/mm",
                                        "Selected-order phase power, -m λ A1/(π r_norm²); λ in mm. Base refractive power excluded.",
                                    ),
                                ),
                            ]),
                            "One entry per DOE; zero for trace order zero.",
                        ),
                    ),
                ]),
                "Both first-order sweeps at every system wavelength.",
            ),
        ),
    ])
}

fn first_order_schema() -> Value {
    let entry = obj(&[
        ("key", string("Stable key (e.g. `efl`, `bfl`, `epd`).")),
        ("label", string("Display label.")),
        ("value", nf()),
        ("unit", string("Unit of `value` (`mm`, `°`, or empty).")),
    ]);
    result(
        &[
            ("paraxial_xy", paraxial_xy_schema()),
            (
                "entries",
                arr(
                    entry.clone(),
                    "The report at the first selected wavelength.",
                ),
            ),
            ("wavelength_um", num("µm", "Wavelength of `entries`.")),
            (
                "tables",
                arr(
                    obj(&[
                        ("wavelength", int("System wavelength index.")),
                        ("wavelength_um", num("µm", "")),
                        ("entries", arr(entry, "")),
                    ]),
                    "One report per selected wavelength.",
                ),
            ),
            (
                "system",
                opaque(
                    "System summary: name, surfaces, stop, wavelengths_um, primary_wavelength, field_kind, field_unit, max_field, object_at_infinity, ray_aiming.",
                ),
            ),
        ],
        vec![],
    )
}

// ---------------------------------------------------------------------------
// layout

fn layout_schema() -> Value {
    let surface = obj(&[
        ("index", int("Surface index.")),
        ("id", int("Stable surface id.")),
        ("label", string("")),
        ("vertex", tuple(nf(), 3, "Global vertex position (mm).")),
        (
            "axis",
            tuple(nf(), 3, "Global direction of the local +z axis."),
        ),
        ("semi_diameter", nf_u("mm", "Clear semi-diameter drawn.")),
        ("semi_diameter_is_auto", boolean("")),
        ("mechanical_semi_diameter", nf_u("mm", "")),
        ("is_mirror", boolean("")),
        ("is_stop", boolean("")),
        (
            "is_diffractive",
            boolean("Base surface carries a DOE phase."),
        ),
    ]);
    let element = obj(&[
        ("first_surface", int("")),
        ("last_surface", int("")),
        ("mechanical_semi_diameter", nf_u("mm", "")),
        (
            "outline_polyline",
            int("Index of the element's closed outline in the polyline buffers (kind 7)."),
        ),
    ]);
    result(
        &[
            (
                "field_values",
                arr(
                    field_value(),
                    "Field coordinates in system units, including analysis-local default fields.",
                ),
            ),
            (
                "section",
                enumeration(&["yz", "xz"], "Selected 2-D section."),
            ),
            (
                "mesh_triangle_count",
                int("Triangle count T; zero when mesh_rings=0 or include_3d=false."),
            ),
            (
                "origin",
                tuple(
                    num("mm", ""),
                    3,
                    "Global point subtracted from the f32 `vertices` (the bounds centre).",
                ),
            ),
            (
                "bounds",
                obj(&[("min", tuple(nf(), 3, "")), ("max", tuple(nf(), 3, ""))]),
            ),
            ("surfaces", arr(surface, "Per drawn surface.")),
            (
                "elements",
                arr(
                    element,
                    "Lens elements (consecutive refracting surfaces with glass between them).",
                ),
            ),
            (
                "polyline_count",
                int("Polylines P (rays, profiles, … then element outlines)."),
            ),
            ("vertex_count", int("Vertices V.")),
            (
                "kind_names",
                arr(
                    string(""),
                    "Name of each kind code: 0 surface_profile, 1 surface_rim, 2 element_edge, 3 stop, 4 image_plane, 5 ray, 6 axis, 7 element_outline.",
                ),
            ),
            ("wavelengths_um", wl_um()),
            ("field_count", int("Number of system fields.")),
        ],
        vec![
            buffer(
                "surface_mesh",
                "f32",
                &["T", "3", "3"],
                "mm",
                "Triangle vertices x,y,z relative to origin, counterclockwise toward the positive local normal. Domain holes omitted.",
            ),
            buffer(
                "mesh_surfaces",
                "u32",
                &["T"],
                "",
                "Surface index per mesh triangle.",
            ),
            buffer(
                "vertices",
                "f32",
                &["V", "3"],
                "mm",
                "x, y, z relative to `origin` (global frame).",
            ),
            buffer(
                "vertices_f64",
                "f64",
                &["V", "3"],
                "mm",
                "Absolute global x, y, z.",
            ),
            buffer(
                "offsets",
                "u32",
                &["P+1"],
                "",
                "Polyline i is vertices offsets[i] .. offsets[i+1].",
            ),
            buffer("kinds", "u8", &["P"], "", "Kind code (see `kind_names`)."),
            buffer("surfaces", "i32", &["P"], "", "Surface index, −1 for none."),
            buffer(
                "fields",
                "i32",
                &["P"],
                "",
                "Field index of a ray, −1 for none.",
            ),
            buffer(
                "wavelengths",
                "i32",
                &["P"],
                "",
                "Wavelength index of a ray, −1 for none.",
            ),
            buffer(
                "flags",
                "u8",
                &["P"],
                "",
                "Bit 0: in the selected global section (a slab of 1% of the largest mechanical diameter); bit 1: vignetted ray; bit 2: DOE surface.",
            ),
        ],
    )
}

fn run_layout(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let settings = parse(s)?;
    let scene = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::layout_job(b, t, &settings, ctx),
        layout::layout_job(t, &settings, ctx)
    )?;
    let b = scene.to_buffers();
    let (v, w) = crate_json(&scene)?;
    let np = b.polyline_count();
    let nv = b.vertices_f64.len() / 3;
    let n_lines = scene.polylines.len();
    let elements: Vec<Value> = v["elements"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(k, e)| {
            json!({
                "first_surface": e["first_surface"],
                "last_surface": e["last_surface"],
                "mechanical_semi_diameter": e["mechanical_semi_diameter"],
                "outline_polyline": n_lines + k,
            })
        })
        .collect();
    let json = json!({
        "origin": b.origin,
        "section": scene.section,
        "mesh_triangle_count": scene.meshes.iter().map(|m| m.triangles.len()).sum::<usize>(),
        "bounds": v["bounds"],
        "surfaces": v["surfaces"],
        "elements": elements,
        "polyline_count": np,
        "vertex_count": nv,
        "kind_names": mklight_analysis::layout::PolylineKind::CODE_NAMES,
        "wavelengths_um": v["wavelengths_um"],
        "field_count": v["field_count"],
        "field_values": v["field_values"],
    });
    let mut mesh_vertices = Vec::new();
    let mut mesh_surfaces = Vec::new();
    for mesh in &scene.meshes {
        for triangle in &mesh.triangles {
            mesh_surfaces.push(mesh.surface as u32);
            for point in triangle {
                mesh_vertices.extend((0..3).map(|k| (point[k] - b.origin[k]) as f32));
            }
        }
    }
    let nt = mesh_surfaces.len();
    let buffers = vec![
        Buffer::new(
            "surface_mesh",
            vec![nt, 3, 3],
            BufferData::F32(mesh_vertices),
        ),
        Buffer::new("mesh_surfaces", vec![nt], BufferData::U32(mesh_surfaces)),
        Buffer::new("vertices", vec![nv, 3], BufferData::F32(b.vertices)),
        Buffer::new("vertices_f64", vec![nv, 3], BufferData::F64(b.vertices_f64)),
        Buffer::new("offsets", vec![np + 1], BufferData::U32(b.offsets)),
        Buffer::new("kinds", vec![np], BufferData::U8(b.kinds)),
        Buffer::new("surfaces", vec![np], BufferData::I32(b.surfaces)),
        Buffer::new("fields", vec![np], BufferData::I32(b.fields)),
        Buffer::new("wavelengths", vec![np], BufferData::I32(b.wavelengths)),
        Buffer::new("flags", vec![np], BufferData::U8(b.flags)),
    ];
    Ok(output(json, buffers, w))
}

// ---------------------------------------------------------------------------
// spot_diagram

fn spot_stats() -> Value {
    obj(&[
        ("rays_launched", int("")),
        ("rays_arrived", int("")),
        (
            "vignetted_fraction",
            nf_u(
                "",
                "Weighted fraction of launched rays that did not arrive.",
            ),
        ),
        ("centroid", xy("Centroid relative to the chief ray (mm).")),
        ("rms_radius_centroid", nf_u("mm", "")),
        ("rms_radius_chief", nf_u("mm", "")),
        ("geo_radius_centroid", nf_u("mm", "")),
        ("geo_radius_chief", nf_u("mm", "")),
        ("rms_x", nf_u("mm", "RMS of x about the centroid.")),
        ("rms_y", nf_u("mm", "RMS of y about the centroid.")),
    ])
}

fn spot_schema() -> Value {
    let per_wl = obj(&[
        ("wavelength", int("System wavelength index.")),
        ("wavelength_um", num("µm", "")),
        ("weight", num("", "Spectral weight.")),
        (
            "chief",
            xy("This wavelength's chief ray on the image surface (local mm)."),
        ),
        ("stats", spot_stats()),
    ]);
    let field = obj_opt(
        &[
            (
                "field",
                int("Field index (system fields, then analysis-local defaults)."),
            ),
            ("field_value", field_value()),
            (
                "point_range",
                tuple(
                    int(""),
                    2,
                    "[offset, count] of this field's rows in the point buffers.",
                ),
            ),
            (
                "reference",
                xy("Reference point of the plotted points (image local mm)."),
            ),
            (
                "chief",
                xy("Primary-wavelength chief ray (image local mm)."),
            ),
            ("centroid", xy("Polychromatic centroid (image local mm).")),
            (
                "rms_radius",
                nf_u("mm", "Polychromatic RMS radius about the reference."),
            ),
            (
                "geo_radius",
                nf_u("mm", "Largest distance from the reference."),
            ),
            ("polychromatic", spot_stats()),
            ("per_wavelength", arr(per_wl, "")),
            (
                "through_focus",
                arr(
                    obj(&[
                        ("defocus", num("mm", "")),
                        ("rms_radius_centroid", nf_u("mm", "")),
                        ("rms_radius_chief", nf_u("mm", "")),
                    ]),
                    "Present when through_focus was requested.",
                ),
            ),
        ],
        &["through_focus"],
    );
    result(
        &[
            ("fields", arr(field, "One entry per selected field.")),
            ("wavelengths_um", wl_um()),
            ("wavelengths", wl_list()),
            (
                "airy_radius",
                nf_u(
                    "mm",
                    "1.22 λ N_w at the primary wavelength; \"inf\" when afocal.",
                ),
            ),
            ("defocus", num("mm", "")),
        ],
        vec![
            buffer(
                "points",
                "f64",
                &["N", "2"],
                "mm",
                "Ray points (x, y) relative to their field's `reference`, image local frame; empty unless include_points.",
            ),
            buffer(
                "point_field",
                "u32",
                &["N"],
                "",
                "Position of each point's field in `fields`.",
            ),
            buffer(
                "point_wavelength",
                "u32",
                &["N"],
                "",
                "System wavelength index of each point.",
            ),
        ],
    )
}

/// Move the per-field `points` of a spot diagram into buffers.
#[cfg(test)]
fn spot_points_to_buffers(v: &mut Value) -> Vec<Buffer> {
    let mut pts = Vec::new();
    let mut fidx = Vec::new();
    let mut widx = Vec::new();
    if let Some(fields) = v["fields"].as_array_mut() {
        for (k, f) in fields.iter_mut().enumerate() {
            let start = fidx.len();
            if let Some(Value::Array(p)) = f.as_object_mut().and_then(|o| o.remove("points")) {
                for q in &p {
                    pts.push(num_of(&q["x"]));
                    pts.push(num_of(&q["y"]));
                    fidx.push(k as u32);
                    widx.push(q["wavelength"].as_u64().unwrap_or(0) as u32);
                }
            }
            f["point_range"] = json!([start, fidx.len() - start]);
        }
    }
    let n = fidx.len();
    vec![
        Buffer::new("points", vec![n, 2], BufferData::F64(pts)),
        Buffer::new("point_field", vec![n], BufferData::U32(fidx)),
        Buffer::new("point_wavelength", vec![n], BufferData::U32(widx)),
    ]
}

fn spot_output(mut r: spot::SpotDiagram, ctx: &JobContext) -> EngineResult<AnalysisOutput> {
    let n = r.fields.iter().map(|f| f.points.len()).sum::<usize>();
    let mut pts = Vec::with_capacity(2 * n);
    let mut fidx = Vec::with_capacity(n);
    let mut widx = Vec::with_capacity(n);
    let mut ranges = Vec::with_capacity(r.fields.len());
    for (k, f) in r.fields.iter_mut().enumerate() {
        let start = fidx.len();
        for chunk in f.points.chunks(16384) {
            ctx.check()?;
            for q in chunk {
                pts.extend([q.x, q.y]);
                fidx.push(k as u32);
                widx.push(q.wavelength as u32);
            }
        }
        ranges.push([start, fidx.len() - start]);
        f.points = Vec::new();
    }
    let (mut v, w) = crate_json(&r)?;
    for (f, range) in v["fields"].as_array_mut().unwrap().iter_mut().zip(ranges) {
        f.as_object_mut().unwrap().remove("points");
        f["point_range"] = json!(range);
    }
    Ok(output(
        v,
        vec![
            Buffer::new("points", vec![n, 2], BufferData::F64(pts)),
            Buffer::new("point_field", vec![n], BufferData::U32(fidx)),
            Buffer::new("point_wavelength", vec![n], BufferData::U32(widx)),
        ],
        w,
    ))
}

#[test]
fn typed_spot_packing_matches_legacy_envelope() {
    let t = mklight_analysis::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    for include_points in [false, true] {
        let s = parse(&json!({
            "fields":{"type":"list","indices":[2,0]},
            "wavelengths":{"type":"list","indices":[2,0]},
            "defocus":0.2,"reference":"centroid","include_points":include_points
        }))
        .unwrap();
        let r = spot::spot_diagram_job(&t, &s, &ctx).unwrap();
        let (mut json, warnings) = crate_json(&r).unwrap();
        let buffers = spot_points_to_buffers(&mut json);
        let got = spot_output(r, &ctx).unwrap();
        assert_eq!(got.json, json);
        assert_eq!(got.buffers, buffers);
        assert_eq!(got.warnings, warnings);
    }
}

fn selection(
    t: &SequentialTracer,
    f: &util::FieldSelection,
    w: &util::WavelengthSelection,
) -> EngineResult<(Vec<usize>, Vec<usize>)> {
    let sys = t.system();
    Ok((
        f.resolve(t.fields().len())?,
        w.resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?,
    ))
}

fn run_spot(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let st: spot::SpotDiagramSettings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::spot_diagram_job(b, t, &st, ctx),
        spot::spot_diagram_job(t, &st, ctx)
    )?;
    spot_output(r, ctx)
}

// ---------------------------------------------------------------------------
// fans

fn fan_settings() -> Value {
    settings_from_default::<fans::FanSettings>(json!({
        "samples": a_count(2, 100_001, "Pupil samples across each fan, from −1 to 1."),
        "fields": a_fields(), "wavelengths": a_wavelengths(),
        "reference": {"enum": ["primary_chief", "same_wavelength"], "description": "Chief ray the aberrations are measured from: the primary wavelength's, or each wavelength's own."}
    }))
}

fn fan_schema(ray: bool) -> Value {
    let (unit, what) = if ray {
        (
            "mm",
            "εy (tangential) / εx (sagittal) on the image surface, mm",
        )
    } else {
        ("waves", "OPD in waves at the curve's wavelength")
    };
    let mut curve_props = vec![
        ("wavelength", int("System wavelength index.")),
        (
            "pupil",
            arr(
                num("", ""),
                "Normalised pupil coordinate P_y (tangential) or P_x (sagittal), −1 … 1.",
            ),
        ),
        ("values", nf_vec(unit, what)),
    ];
    if ray {
        curve_props.push((
            "cross",
            nf_vec("mm", "The other transverse component (εx on the tangential fan, εy on the sagittal fan)."),
        ));
    }
    let curve = obj_opt(&curve_props, &["cross"]);
    result(
        &[
            (
                "quantity",
                enumeration(&["transverse_aberration", "opd"], ""),
            ),
            (
                "fields",
                arr(
                    obj(&[
                        (
                            "field",
                            int("Field index (system fields, then analysis-local defaults)."),
                        ),
                        ("field_value", field_value()),
                        (
                            "tangential",
                            arr(curve.clone(), "One curve per selected wavelength."),
                        ),
                        ("sagittal", arr(curve, "One curve per selected wavelength.")),
                    ]),
                    "",
                ),
            ),
            ("wavelengths_um", wl_um()),
        ],
        vec![],
    )
}

// ---------------------------------------------------------------------------
// field curvature, distortion

fn field_curvature_schema() -> Value {
    let curve = obj(&[
        ("wavelength", int("System wavelength index.")),
        ("wavelength_um", num("µm", "")),
        (
            "tangential",
            nf_vec(
                "mm",
                "Tangential focus shift from the image surface along z (Coddington).",
            ),
        ),
        (
            "sagittal",
            nf_vec("mm", "Sagittal focus shift (Coddington)."),
        ),
        (
            "tangential_close_rays",
            nf_vec("mm", "Close-ray cross-check of `tangential`."),
        ),
        (
            "sagittal_close_rays",
            nf_vec("mm", "Close-ray cross-check of `sagittal`."),
        ),
        (
            "distortion_percent",
            nf_vec(
                "%",
                "100 (h_real − h_paraxial)/h_paraxial along the sampled field axis.",
            ),
        ),
        (
            "f_tan_theta_distortion_percent",
            nf_vec("%", "Against f·tanθ."),
        ),
        ("f_theta_distortion_percent", nf_vec("%", "Against f·θ.")),
        ("real_height", nf_vec("mm", "Real chief-ray image height.")),
        (
            "paraxial_height",
            nf_vec("mm", "Paraxial chief-ray image height."),
        ),
    ]);
    result(
        &[
            (
                "field_coordinates",
                arr(
                    xy("Normalised [Hx, Hy]; paired with each scalar field sample."),
                    "Axis sweeps, with signed fields where symmetry requires them.",
                ),
            ),
            ("field_values", nf_vec("", "Field samples in `field_unit`.")),
            (
                "normalized_field",
                nf_vec("", "Field samples normalised to the maximum field."),
            ),
            ("field_unit", string("`°` for angles, `mm` for heights.")),
            ("curves", arr(curve, "One per selected wavelength.")),
            ("wavelengths_um", wl_um()),
        ],
        vec![],
    )
}

fn grid_distortion_schema() -> Value {
    result(
        &[
            ("grid_size", int("Points per side n.")),
            ("wavelength", int("System wavelength index.")),
            ("max_abs_distortion_percent", nf_u("%", "")),
        ],
        vec![
            buffer(
                "field",
                "f64",
                &["n*n", "2"],
                "",
                "Normalised field (Hx, Hy), x fastest.",
            ),
            buffer(
                "paraxial",
                "f64",
                &["n*n", "2"],
                "mm",
                "Paraxial image point.",
            ),
            buffer(
                "real",
                "f64",
                &["n*n", "2"],
                "mm",
                "Real chief-ray image point (NaN if it failed).",
            ),
            buffer(
                "distortion_percent",
                "f64",
                &["n*n"],
                "%",
                "Radial distortion of each point.",
            ),
        ],
    )
}

fn run_grid_distortion(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let settings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::grid_distortion_job(b, t, &settings, ctx),
        curvature::grid_distortion_job(t, &settings, ctx)
    )?;
    let (mut v, w) = crate_json(&r)?;
    let pts = v
        .as_object_mut()
        .and_then(|o| o.remove("points"))
        .unwrap_or(json!([]));
    let p = pts.as_array().cloned().unwrap_or_default();
    let n = p.len();
    let col = |k: &str| pairs(&Value::Array(p.iter().map(|q| q[k].clone()).collect()));
    let dist: Vec<f64> = p.iter().map(|q| num_of(&q["distortion_percent"])).collect();
    let buffers = vec![
        Buffer::new("field", vec![n, 2], BufferData::F64(col("field"))),
        Buffer::new("paraxial", vec![n, 2], BufferData::F64(col("paraxial"))),
        Buffer::new("real", vec![n, 2], BufferData::F64(col("real"))),
        Buffer::new("distortion_percent", vec![n], BufferData::F64(dist)),
    ];
    Ok(output(v, buffers, w))
}

// ---------------------------------------------------------------------------
// Seidel family

fn seidel_terms() -> Value {
    obj(&[
        ("s1", nf_u("mm", "Spherical S_I.")),
        ("s2", nf_u("mm", "Coma S_II.")),
        ("s3", nf_u("mm", "Astigmatism S_III.")),
        ("s4", nf_u("mm", "Petzval S_IV.")),
        ("s5", nf_u("mm", "Distortion S_V.")),
        ("cl", nf_u("mm", "Axial colour C_L.")),
        ("ct", nf_u("mm", "Lateral colour C_T.")),
    ])
}

fn seidel_schema() -> Value {
    let mut surf = seidel_terms();
    surf["properties"]["surface"] = int("Surface index.");
    surf["required"]
        .as_array_mut()
        .unwrap()
        .push(json!("surface"));
    let w = |unit: &str| {
        let keys = [
            "w020", "w040", "w111", "w131", "w220", "w220p", "w222", "w311",
        ];
        let props: Vec<(&str, Value)> = keys.iter().map(|k| (*k, nf_u(unit, ""))).collect();
        obj(&props)
    };
    result(
        &[
            (
                "status",
                enumeration(
                    &["applicable", "not_applicable"],
                    "Seidel requires rotational system symmetry.",
                ),
            ),
            (
                "reason",
                nullable(string(
                    "Explains non-applicability and suggests full real-ray analyses; coefficients are null and surfaces empty.",
                )),
            ),
            (
                "wavelength",
                int("System wavelength index of the paraxial rays."),
            ),
            ("wavelength_um", num("µm", "")),
            ("short_wavelength", int("Short wavelength of δn.")),
            ("long_wavelength", int("Long wavelength of δn.")),
            (
                "surfaces",
                arr(
                    surf,
                    "Per-surface contributions (surfaces that contribute).",
                ),
            ),
            ("sums", seidel_terms()),
            ("petzval_sum", nf_u("1/mm", "Σ φ/(n n').")),
            (
                "petzval_radius",
                nf_u("mm", "−1/(n'_k P), signed image-space reduced index."),
            ),
            ("lagrange_invariant", nf_u("mm", "")),
            (
                "image_marginal_slope",
                nf_u("", "Image-space reduced marginal slope ω'."),
            ),
            ("wavefront_mm", w("mm")),
            ("wavefront_waves", w("waves")),
        ],
        vec![],
    )
}

fn longitudinal_schema() -> Value {
    result(
        &[
            ("pupil", arr(num("", ""), "Pupil zones P_y from 0 to 1.")),
            (
                "primary_paraxial_focus",
                nf_u("mm", "Primary paraxial focus from the image surface."),
            ),
            (
                "curves",
                arr(
                    obj(&[
                        (
                            "axis",
                            enumeration(&["x", "y"], "Pupil axis of the zonal fan."),
                        ),
                        ("wavelength", int("")),
                        ("wavelength_um", num("µm", "")),
                        ("paraxial_focus", nf_u("mm", "")),
                        (
                            "focus",
                            nf_vec(
                                "mm",
                                "Axis crossing of each zonal ray, from the image surface.",
                            ),
                        ),
                        (
                            "aberration",
                            nf_vec("mm", "Crossing relative to the primary paraxial focus."),
                        ),
                    ]),
                    "",
                ),
            ),
        ],
        vec![],
    )
}

fn chromatic_focal_shift_schema() -> Value {
    result(
        &[
            (
                "focal_shift_x",
                nf_vec(
                    "mm",
                    "X-axis paraxial focus relative to the primary wavelength's x focus.",
                ),
            ),
            (
                "paraxial_focus_x",
                nf_vec(
                    "mm",
                    "X-axis paraxial focus from the image surface; legacy arrays are the y sweep.",
                ),
            ),
            ("wavelengths_um", arr(num("µm", ""), "Sweep wavelengths.")),
            (
                "paraxial_focus",
                nf_vec("mm", "Paraxial focus from the image surface."),
            ),
            (
                "focal_shift",
                nf_vec("mm", "Focus relative to the primary wavelength's."),
            ),
            ("primary_wavelength_um", num("µm", "")),
            ("range", nf_u("mm", "max − min of `focal_shift`.")),
        ],
        vec![],
    )
}

fn lateral_colour_schema() -> Value {
    result(
        &[
            (
                "field_coordinates",
                arr(
                    xy("Normalised [Hx, Hy]; paired with each scalar field sample."),
                    "Axis sweeps, with signed fields where symmetry requires them.",
                ),
            ),
            ("field_values", nf_vec("", "Field samples in `field_unit`.")),
            ("normalized_field", nf_vec("", "")),
            ("field_unit", string("")),
            (
                "curves",
                arr(
                    obj(&[
                        ("wavelength", int("")),
                        ("wavelength_um", num("µm", "")),
                        (
                            "offset",
                            nf_vec("mm", "Chief-ray height minus the primary wavelength's."),
                        ),
                    ]),
                    "",
                ),
            ),
            (
                "lateral_colour",
                nf_vec("mm", "Short minus long chief-ray height."),
            ),
            ("short_wavelength", int("")),
            ("long_wavelength", int("")),
            ("airy_radius", nf_u("mm", "")),
        ],
        vec![],
    )
}

fn relative_illumination_schema() -> Value {
    result(
        &[
            (
                "field_coordinates",
                arr(
                    xy("Normalised [Hx, Hy]; paired with each scalar field sample."),
                    "Axis sweeps, with signed fields where symmetry requires them.",
                ),
            ),
            ("field_values", nf_vec("", "Field samples in `field_unit`.")),
            ("normalized_field", nf_vec("", "")),
            ("field_unit", string("")),
            ("wavelength", int("")),
            ("relative_illumination", nf_vec("", "Ω_p(H)/Ω_p(0).")),
            (
                "projected_solid_angle",
                nf_vec(
                    "sr",
                    "Projected solid angle of the real exit pupil seen from the image point.",
                ),
            ),
            (
                "vignetting_fraction",
                nf_vec("", "Clipped share of the stop area."),
            ),
            (
                "chief_ray_angle_deg",
                nf_vec("°", "Chief-ray angle of incidence on the image surface."),
            ),
            (
                "cos4_object",
                nf_vec("", "cos⁴ of the object-space chief angle, for comparison."),
            ),
        ],
        vec![],
    )
}

// ---------------------------------------------------------------------------
// footprint

fn footprint_schema() -> Value {
    result(
        &[
            (
                "field_values",
                arr(
                    field_value(),
                    "Field coordinates in system units, including analysis-local default fields.",
                ),
            ),
            ("surface", int("")),
            (
                "semi_diameter",
                nf_u("mm", "Clear semi-diameter (fixed, stop or traced)."),
            ),
            (
                "aperture",
                any("The surface's aperture shape (model JSON) or null."),
            ),
            (
                "extent",
                nf_vec("mm", "[x_min, x_max, y_min, y_max] over all passing hits."),
            ),
            ("max_radius", nf_u("mm", "")),
            (
                "sets",
                arr(
                    obj(&[
                        ("field", int("")),
                        ("wavelength", int("")),
                        ("launched", int("")),
                        ("lost_before", int("Rays lost before reaching the surface.")),
                        ("points", tuple(int(""), 2, "[offset, count] in `points`.")),
                        (
                            "clipped",
                            tuple(int(""), 2, "[offset, count] in `clipped`."),
                        ),
                    ]),
                    "One per (field, wavelength).",
                ),
            ),
            (
                "overlays",
                arr(
                    obj(&[
                        ("surface", int("")),
                        ("field", int("")),
                        ("wavelength", int("")),
                        (
                            "points",
                            tuple(int(""), 2, "[offset, count] in `global_points`."),
                        ),
                        (
                            "outline",
                            tuple(int(""), 2, "[offset, count] in `global_outlines`."),
                        ),
                    ]),
                    "One per (surface after the object, field, wavelength): the hits of rays \
                     that pass that surface, whether or not a later surface clips them.",
                ),
            ),
        ],
        vec![
            buffer(
                "points",
                "f64",
                &["N", "2"],
                "mm",
                "Hit points (local frame) of rays that pass.",
            ),
            buffer(
                "clipped",
                "f64",
                &["M", "2"],
                "mm",
                "Hit points of rays the aperture clips.",
            ),
            buffer(
                "global_points",
                "f64",
                &["P", "3"],
                "mm",
                "Global hit points (surface sag and placement included).",
            ),
            buffer(
                "global_outlines",
                "f64",
                &["Q", "3"],
                "mm",
                "Convex hull of each overlay's hits in the surface's local x/y, counterclockwise \
                 and unclosed, as global points; not a vignetting boundary of nonconvex beams.",
            ),
        ],
    )
}

fn run_footprint(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let settings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::footprint_job(b, t, &settings, &ctx.subrange(0.0, 0.3)),
        illumination::footprint_job(t, &settings, &ctx.subrange(0.0, 0.3))
    )?;
    let (mut v, w) = crate_json(&r)?;
    let rays = v
        .as_object_mut()
        .and_then(|o| o.remove("rays"))
        .unwrap_or(json!([]));
    let (mut pts, mut clipped, mut sets) = (Vec::new(), Vec::new(), Vec::new());
    for r in rays.as_array().into_iter().flatten() {
        let (p0, c0) = (pts.len() / 2, clipped.len() / 2);
        pts.extend(pairs(&r["points"]));
        clipped.extend(pairs(&r["clipped"]));
        sets.push(json!({
            "field": r["field"], "wavelength": r["wavelength"],
            "launched": r["launched"], "lost_before": r["lost_before"],
            "points": [p0, pts.len() / 2 - p0],
            "clipped": [c0, clipped.len() / 2 - c0],
        }));
    }
    v["sets"] = Value::Array(sets);
    let o = match gpu::device(input) {
        Some(d) => mklight_analysis::gpu::footprint_overlay_job(
            d.backend.as_ref(),
            t,
            &settings,
            &ctx.subrange(0.3, 1.0),
        )?,
        None => footprint_overlay::footprint_overlay_job(t, &settings, &ctx.subrange(0.3, 1.0))?,
    };
    v["overlays"] = serde_json::to_value(o.sets)?;
    let (n, m) = (pts.len() / 2, clipped.len() / 2);
    let buffers = vec![
        Buffer::new("points", vec![n, 2], BufferData::F64(pts)),
        Buffer::new("clipped", vec![m, 2], BufferData::F64(clipped)),
        Buffer::new(
            "global_points",
            vec![o.points.len(), 3],
            BufferData::F64(o.points.into_iter().flatten().collect()),
        ),
        Buffer::new(
            "global_outlines",
            vec![o.outlines.len(), 3],
            BufferData::F64(o.outlines.into_iter().flatten().collect()),
        ),
    ];
    Ok(output(v, buffers, w))
}

// ---------------------------------------------------------------------------
// energy family (GPU paths)

fn encircled_energy_schema() -> Value {
    result(
        &[
            (
                "radius",
                arr(num("mm", ""), "Radii / half-widths shared by every field."),
            ),
            (
                "fields",
                arr(
                    obj(&[
                        ("field", int("")),
                        ("field_value", field_value()),
                        (
                            "encircled",
                            arr(
                                num("", ""),
                                "Fraction of the arrived energy inside radius[i].",
                            ),
                        ),
                        (
                            "ensquared",
                            arr(
                                num("", ""),
                                "Fraction inside the square of half-width radius[i].",
                            ),
                        ),
                        ("r50", nf_u("mm", "")),
                        ("r80", nf_u("mm", "")),
                        ("r90", nf_u("mm", "")),
                        (
                            "half_width80",
                            nf_u("mm", "Half-width of the square enclosing 80 %."),
                        ),
                        ("rays_arrived", int("")),
                    ]),
                    "",
                ),
            ),
            ("wavelengths", wl_list()),
        ],
        vec![],
    )
}

fn run_encircled_energy(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let st: energy::EncircledEnergySettings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::encircled_energy_job(b, t, &st, ctx),
        energy::encircled_energy_job(t, &st, ctx)
    )?;
    let (v, w) = crate_json(&r)?;
    Ok(output(v, vec![], w))
}

fn geometric_mtf_schema() -> Value {
    result(
        &[
            ("frequencies", arr(num("cycles/mm", ""), "")),
            (
                "curves",
                arr(
                    obj(&[
                        (
                            "field",
                            int("Field index (system fields, then analysis-local defaults)."),
                        ),
                        ("orientation", enumeration(&["tangential", "sagittal"], "")),
                        (
                            "modulation",
                            nf_vec("", "Modulation per frequency (1 at 0)."),
                        ),
                    ]),
                    "",
                ),
            ),
            (
                "diffraction_limit",
                any("Always null for the geometric MTF."),
            ),
            (
                "cutoff_frequency",
                nf_u(
                    "cycles/mm",
                    "Incoherent diffraction cutoff 1/(λN_w), for reference.",
                ),
            ),
            (
                "field_values",
                arr(field_value(), "Field value of each computed field."),
            ),
        ],
        vec![],
    )
}

fn run_geometric_mtf(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let st: energy::GeometricMtfSettings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::geometric_mtf_job(b, t, &st, ctx),
        energy::geometric_mtf_job(t, &st, ctx)
    )?;
    let (v, w) = crate_json(&r)?;
    Ok(output(v, vec![], w))
}

fn best_focus_schema() -> Value {
    result(
        &[
            (
                "field_values",
                arr(
                    field_value(),
                    "Field coordinates in system units, including analysis-local default fields.",
                ),
            ),
            (
                "defocus",
                num(
                    "mm",
                    "Best defocus from the image surface (image local +z).",
                ),
            ),
            (
                "merit",
                num("mm", "Field-weighted mean RMS radius at the best focus."),
            ),
            (
                "merit_at_zero",
                num("mm", "The merit at the current image surface."),
            ),
            ("fields", arr(int(""), "Fields used.")),
            (
                "field_rms",
                nf_vec("mm", "Per-field RMS radius at the best focus."),
            ),
            (
                "field_rms_at_zero",
                nf_vec("mm", "Per-field RMS radius at zero defocus."),
            ),
            (
                "mean_square_defocus",
                num(
                    "mm",
                    "Closed-form minimiser of the mean square RMS (cross-check).",
                ),
            ),
            ("range", tuple(num("mm", ""), 2, "Search range.")),
            ("evaluations", int("")),
            ("converged", boolean("")),
            (
                "through_focus_defocus",
                arr(num("mm", ""), "Empty unless requested."),
            ),
            ("through_focus_merit", arr(num("mm", ""), "")),
        ],
        vec![],
    )
}

fn run_best_focus(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let st: energy::BestFocusSettings = parse(s)?;
    let r = gpu::native_job!(
        input,
        |b| mklight_analysis::gpu::best_focus_job(b, t, &st, ctx),
        energy::best_focus_job(t, &st, ctx)
    )?;
    let (v, w) = crate_json(&r)?;
    Ok(output(v, vec![], w))
}

// ---------------------------------------------------------------------------
// ray_grid

/// Settings of `ray_grid` (same sampling as the energy analyses, hexapolar
/// by default).
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RayGridSettings {
    #[serde(flatten)]
    pub sampling: energy::RaySampling,
}

impl Default for RayGridSettings {
    fn default() -> Self {
        RayGridSettings {
            sampling: energy::RaySampling {
                pattern: util::PupilPattern::Hexapolar,
                rings: 6,
                ..Default::default()
            },
        }
    }
}

fn ray_grid_schema() -> Value {
    result(
        &[
            (
                "field_values",
                arr(
                    field_value(),
                    "Field coordinates in system units, including analysis-local default fields.",
                ),
            ),
            ("fields", arr(int(""), "Selected system fields (axis F).")),
            ("wavelengths", wl_list()),
            (
                "wavelengths_um",
                arr(num("µm", ""), "Wavelengths of axis W."),
            ),
            ("pupil_count", int("Pupil samples P.")),
            ("arrived", int("Rays that reached the image surface.")),
            ("launched", int("F × W × P.")),
        ],
        vec![
            buffer(
                "pupil",
                "f64",
                &["P", "2"],
                "",
                "Normalised pupil coordinates (Px, Py).",
            ),
            buffer(
                "points",
                "f64",
                &["F", "W", "P", "2"],
                "mm",
                "Hit (x, y) on the image surface, local frame; NaN where the ray did not arrive.",
            ),
            buffer(
                "directions",
                "f64",
                &["F", "W", "P", "3"],
                "",
                "Unit direction (image local frame); NaN where the ray did not arrive.",
            ),
            buffer(
                "chief",
                "f64",
                &["F", "W", "2"],
                "mm",
                "Unclipped chief-ray hit (x, y); NaN if it failed.",
            ),
        ],
    )
}

fn run_ray_grid(
    input: &AnalysisInput<'_>,
    t: &SequentialTracer,
    s: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let st: RayGridSettings = parse(s)?;
    let sm = &st.sampling;
    if !(1..=512).contains(&sm.rings) {
        return Err(EngineError::InvalidSettings(
            "rings must be in 1 … 512".into(),
        ));
    }
    let (fields, wls) = selection(t, &sm.fields, &sm.wavelengths)?;
    let pupil = util::pupil_points(sm.pattern, sm.rings, sm.seed);
    let (nf, nw, np) = (fields.len(), wls.len(), pupil.len());
    let mut points = vec![f64::NAN; nf * nw * np * 2];
    let mut dirs = vec![f64::NAN; nf * nw * np * 3];
    let mut chief = vec![f64::NAN; nf * nw * 2];
    let mut put = |i: usize, p: [f64; 2], d: [f64; 3]| {
        points[2 * i..2 * i + 2].copy_from_slice(&p);
        dirs[3 * i..3 * i + 3].copy_from_slice(&d);
    };
    if let Some(d) = gpu::device(input) {
        let h = mklight_analysis::gpu::ray_grid_job(
            d.backend.as_ref(),
            t,
            sm,
            &ctx.subrange(0.0, 0.95),
        )?;
        gpu::record_native(input, &h.run_info);
        for (i, r) in h.rays.iter().enumerate() {
            if let Some(r) = r {
                put(i, r.point, [r.dir.x, r.dir.y, r.dir.z]);
            }
        }
    } else {
        let prepared: Vec<_> = fields.iter().map(|&f| t.field(f).clone()).collect();
        let grid = Grid {
            fields: &prepared,
            wavelengths: &wls,
            pupil: &pupil,
        };
        let rays = trace_grid_job(t, &grid, &t.options(), &ctx.subrange(0.0, 0.95))?;
        for (i, r) in rays.iter().enumerate() {
            if r.status.is_ok() {
                put(i, [r.point.x, r.point.y], [r.dir.x, r.dir.y, r.dir.z]);
            }
        }
    }
    for (fi, &field) in fields.iter().enumerate() {
        for (wi, &wl) in wls.iter().enumerate() {
            let c = t.trace(t.field(field), wl, 0.0, 0.0, &TraceOptions::unclipped());
            if c.status.is_ok() {
                let k = 2 * (fi * nw + wi);
                chief[k] = c.ray.pos.x;
                chief[k + 1] = c.ray.pos.y;
            }
        }
    }
    let arrived = points.chunks(2).filter(|p| !p[0].is_nan()).count();
    let sys = t.system();
    let json = json!({
        "field_values": fields.iter().map(|&f| util::field_value(t, f)).collect::<Vec<_>>(),
        "fields": fields,
        "wavelengths": wls,
        "wavelengths_um": wls.iter().map(|&w| sys.wavelengths_um[w]).collect::<Vec<_>>(),
        "pupil_count": np,
        "arrived": arrived,
        "launched": nf * nw * np,
    });
    let pupil_flat: Vec<f64> = pupil.iter().flat_map(|p| [p[0], p[1]]).collect();
    let buffers = vec![
        Buffer::new("pupil", vec![np, 2], BufferData::F64(pupil_flat)),
        Buffer::new("points", vec![nf, nw, np, 2], BufferData::F64(points)),
        Buffer::new("directions", vec![nf, nw, np, 3], BufferData::F64(dirs)),
        Buffer::new("chief", vec![nf, nw, 2], BufferData::F64(chief)),
    ];
    Ok(output(json, buffers, t.warnings().to_vec()))
}
