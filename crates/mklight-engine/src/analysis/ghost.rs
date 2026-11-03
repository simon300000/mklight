//! Explicit ghost analyses (`mklight-ghost`).
//!
//! * `ghost_paths`: first-order (paraxial) ghost ranking along the unfolded
//!   ghost sequences: focus, blur, image height, relative irradiance.
//! * `ghost_irradiance`: real-ray ghost irradiance maps on the sensor and
//!   ghost-to-signal ratios; maps travel as typed buffers.
//!
//! Both share the ghost model options (`GhostModelOptions`: sensor
//! reflectance, clipping, virtual-segment rejection) and report the run's
//! `coverage` (interface kinds, clipping apertures, the exact path set),
//! which `stray_light` uses to partition paths between the two solvers
//! (CONVENTIONS §10).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::JobContext;
use mklight_ghost::{
    ClipMode, EnumerationSettings, GhostModel, GhostPath, GhostSettings, IrradianceMap,
    IrradianceSettings, ParaxialGhostSettings, ghost_irradiance, paraxial_ghosts,
};
use mklight_trace::FieldPoint;

use super::geometric::{crate_json, output, parse, plain};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};

pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(GhostPaths));
    r.register(Arc::new(GhostIrradiance));
}

/// The ghost-model options every ghost (and partitioned stray) run shares.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GhostModelOptions {
    /// Unpolarised reflectance of the sensor; null: the sensor does not
    /// reflect.
    pub sensor_reflectance: Option<f64>,
    pub clip: ClipMode,
    /// Treat an intersection behind the ray as a lost ray.
    pub reject_virtual: bool,
}

impl Default for GhostModelOptions {
    fn default() -> Self {
        let d = GhostSettings::default();
        GhostModelOptions {
            sensor_reflectance: d.sensor_reflectance,
            clip: d.clip,
            reject_virtual: d.reject_virtual,
        }
    }
}

impl GhostModelOptions {
    pub fn model(&self, input: &AnalysisInput<'_>) -> EngineResult<GhostModel> {
        let settings = GhostSettings {
            sensor_reflectance: self.sensor_reflectance,
            clip: self.clip,
            reject_virtual: self.reject_virtual,
            ..GhostSettings::default()
        };
        Ok(GhostModel::from_system(input.resolved()?, settings)?)
    }

    /// Settings annotations of the three keys.
    pub fn annotations() -> serde_json::Map<String, Value> {
        let v = json!({
            "sensor_reflectance": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0,
                "description": "Unpolarised, angle-independent reflectance of the sensor; null: the sensor does not reflect."},
            "clip": {"description": "Clipping apertures: {\"mode\": \"clear_apertures\", \"margin_mm\": m} (clear semi-diameters + margin; the stop at its size), {\"mode\": \"fixed_apertures\"} or {\"mode\": \"none\"}."},
            "reject_virtual": {"description": "Treat an intersection behind the ray as a lost ray."}
        });
        match v {
            Value::Object(m) => m,
            _ => unreachable!(),
        }
    }
}

fn enumeration_annotation() -> Value {
    json!({"description": "Path enumeration: {\"max_reflections\": 2|4|6|8, \"max_higher_order_paths\": n, \"min_reflectance_product\": r, \"surface_range\": [first, last] | null, \"exclude_surfaces\": [i, …], \"diffraction_orders\": [m, …], \"max_diffraction_paths\": n}."})
}

fn path_schema() -> Value {
    json!({"oneOf": [
        {"type": "array", "items": {"type": "integer", "minimum": 0}},
        {"type": "object", "required": ["reflections", "diffraction"], "properties": {
            "reflections": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "diffraction": {"type": "array", "items": {"type": "object", "required": ["surface", "visit", "order"], "properties": {
                "surface": {"type": "integer", "minimum": 1}, "visit": {"type": "integer", "minimum": 0}, "order": {"type": "integer"}
            }}}
        }}
    ]})
}
fn paths_annotation() -> Value {
    json!({"type": ["array", "null"], "items": path_schema(),
           "description": "Explicit paths: reflection arrays, or {reflections, diffraction: [{surface, visit, order}]} with non-selected DOE orders, sorted by (surface, visit)."})
}

fn merge(mut a: serde_json::Map<String, Value>, b: Value) -> Value {
    if let Value::Object(m) = b {
        a.extend(m);
    }
    Value::Object(a)
}

fn coverage_schema() -> Value {
    obj(&[
        (
            "interfaces",
            arr(
                string(""),
                "Interface kind per surface: object, passive, refractive, ideal_lens, mirror, image.",
            ),
        ),
        (
            "apertures",
            arr(
                opaque(
                    "{\"type\": \"none\"} | {\"type\": \"surface\"} | {\"type\": \"radius\", \"radius\": mm}",
                ),
                "Clipping of every surface in this run.",
            ),
        ),
        (
            "paths",
            arr(
                path_schema(),
                "The exact set of ghost paths this run counted.",
            ),
        ),
    ])
}

// ---------------------------------------------------------------------------
// ghost_paths

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GhostPathsSettings {
    #[serde(flatten)]
    pub model: GhostModelOptions,
    pub enumeration: EnumerationSettings,
    pub paths: Option<Vec<GhostPath>>,
    /// System wavelength index; null = primary.
    pub wavelength: Option<usize>,
    pub signal_diameter_mm: Option<f64>,
    pub near_focus_factor: f64,
}

impl Default for GhostPathsSettings {
    fn default() -> Self {
        let d = ParaxialGhostSettings::default();
        GhostPathsSettings {
            model: GhostModelOptions::default(),
            enumeration: d.enumeration,
            paths: d.paths,
            wavelength: d.wavelength,
            signal_diameter_mm: d.signal_diameter_mm,
            near_focus_factor: d.near_focus_factor,
        }
    }
}

pub struct GhostPaths;

impl Analysis for GhostPaths {
    fn memoizable(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "ghost_paths"
    }
    fn title(&self) -> &str {
        "Ghost paths"
    }
    fn category(&self) -> &str {
        "ghost"
    }
    fn description(&self) -> &str {
        "First-order ghost ranking: every enumerated two-reflection (or higher) ghost with \
         its reflectance product, transmittance, focus from the sensor, blur diameter, \
         image height and estimated irradiance relative to the signal."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<GhostPathsSettings>(merge(
            GhostModelOptions::annotations(),
            json!({
                "enumeration": enumeration_annotation(),
                "paths": paths_annotation(),
                "wavelength": {"type": ["integer", "null"], "minimum": 0, "description": "System wavelength index; null = the primary."},
                "signal_diameter_mm": {"type": ["number", "null"], "minimum": 0.0, "unit": "mm", "description": "Signal spot diameter; null = automatic."},
                "near_focus_factor": {"minimum": 0.0, "description": "Ghosts whose spot is at most this many signal diameters are flagged near_focus."}
            }),
        ))
    }
    fn result_schema(&self) -> Value {
        let ghost = obj(&[
            ("path", path_schema()),
            ("label", string("e.g. `R11·R9`.")),
            (
                "reflectance_product",
                num("", "Π R, normal incidence, unpolarised."),
            ),
            (
                "transmittance",
                num("", "Π T of the passes times axial bulk transmittance."),
            ),
            (
                "focus_from_sensor",
                nf_u(
                    "mm",
                    "Signed distance of the ghost focus from the sensor (image local z).",
                ),
            ),
            (
                "spot_diameter",
                nf_u(
                    "mm",
                    "Ghost blur diameter on the sensor (full aperture, on axis).",
                ),
            ),
            (
                "image_height",
                nf_u(
                    "mm",
                    "Ghost centre height on the sensor at the maximum field.",
                ),
            ),
            (
                "vignetting",
                nf_u("", "Paraxial vignetting estimate in (0, 1]."),
            ),
            (
                "relative_irradiance",
                nf_u("", "Estimated ghost irradiance relative to the signal."),
            ),
            ("near_focus", boolean("")),
        ]);
        result(
            &[
                ("wavelength", int("System wavelength index.")),
                ("signal_diameter", nf_u("mm", "")),
                (
                    "primary_transmittance",
                    num("", "Normal incidence, axial absorption."),
                ),
                ("primary_focus_from_sensor", nf_u("mm", "")),
                ("primary_image_height", nf_u("mm", "")),
                (
                    "ghosts",
                    arr(ghost, "Ranked by relative_irradiance, strongest first."),
                ),
                (
                    "truncated",
                    boolean("Higher-order paths were dropped by the budget."),
                ),
                ("coverage", coverage_schema()),
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
        let s: GhostPathsSettings = parse(settings)?;
        let model = s.model.model(input)?;
        let ps = ParaxialGhostSettings {
            enumeration: s.enumeration,
            paths: s.paths,
            wavelength: s.wavelength,
            signal_diameter_mm: s.signal_diameter_mm,
            near_focus_factor: s.near_focus_factor,
        };
        let r = if let Some(d) = super::gpu::device(input) {
            let r = mklight_ghost::gpu::paraxial_ghosts(d.backend.as_ref(), &model, &ps, ctx)?;
            super::gpu::record_native(input, &r.run);
            r
        } else {
            paraxial_ghosts(&model, &ps)?
        };
        let (v, w) = crate_json(&r)?;
        Ok(output(v, vec![], w))
    }
}

// ---------------------------------------------------------------------------
// ghost_irradiance

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GhostIrradianceSettings {
    #[serde(flatten)]
    pub model: GhostModelOptions,
    pub enumeration: EnumerationSettings,
    pub paths: Option<Vec<GhostPath>>,
    pub fields: Vec<usize>,
    pub points: Vec<FieldPoint>,
    pub wavelengths: Vec<usize>,
    pub pupil_samples: usize,
    pub overfill: f64,
    pub map_bins: [usize; 2],
    pub map_half_size: Option<[f64; 2]>,
    pub keep_path_maps: bool,
}

/// Engine default pupil grid: coarser than the crate's 128 for interactive
/// use (an engine choice, documented in docs/RESULTS.md).
pub const DEFAULT_GHOST_PUPIL_SAMPLES: usize = 48;

impl Default for GhostIrradianceSettings {
    fn default() -> Self {
        let d = IrradianceSettings::default();
        GhostIrradianceSettings {
            model: GhostModelOptions::default(),
            enumeration: d.enumeration,
            paths: d.paths,
            fields: d.fields,
            points: d.points,
            wavelengths: d.wavelengths,
            pupil_samples: DEFAULT_GHOST_PUPIL_SAMPLES,
            overfill: d.overfill,
            map_bins: d.map_bins,
            map_half_size: d.map_half_size,
            keep_path_maps: d.keep_path_maps,
        }
    }
}

impl GhostIrradianceSettings {
    pub fn irradiance(&self) -> IrradianceSettings {
        IrradianceSettings {
            enumeration: self.enumeration.clone(),
            paths: self.paths.clone(),
            fields: self.fields.clone(),
            points: self.points.clone(),
            wavelengths: self.wavelengths.clone(),
            pupil_samples: self.pupil_samples,
            overfill: self.overfill,
            map_bins: self.map_bins,
            map_half_size: self.map_half_size,
            keep_path_maps: self.keep_path_maps,
        }
    }
}

pub struct GhostIrradiance;

/// Whether a map has every `nx × ny` value.
fn map_complete(m: &IrradianceMap) -> bool {
    m.values.len() == m.bins[0] * m.bins[1]
}

/// The values of a complete map as a `[ny, nx]` buffer, as they read back
/// from the result JSON (plain numbers: non-finite values become NaN).
fn map_buffer(name: &str, m: IrradianceMap) -> Option<Buffer> {
    let [nx, ny] = m.bins;
    map_complete(&m).then(|| {
        let v = m.values.into_iter().map(plain).collect();
        Buffer::new(name, vec![ny, nx], BufferData::F64(v))
    })
}

impl Analysis for GhostIrradiance {
    fn memoizable(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "ghost_irradiance"
    }
    fn title(&self) -> &str {
        "Ghost irradiance"
    }
    fn category(&self) -> &str {
        "ghost"
    }
    fn description(&self) -> &str {
        "Real-ray ghost irradiance on the sensor: per-path flux and peak ratios against the \
         primary image, per-field statistics, the total ghost map and the primary map \
         (typed buffers)."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<GhostIrradianceSettings>(merge(
            GhostModelOptions::annotations(),
            json!({
                "enumeration": enumeration_annotation(),
                "paths": paths_annotation(),
                "fields": {"items": {"type": "integer", "minimum": 0}, "description": "System field indices; empty = all fields (unless points are given)."},
                "points": {"description": "Extra normalised field points [{\"hx\", \"hy\"}], weight 1 each, analysed after the system fields."},
                "wavelengths": {"items": {"type": "integer", "minimum": 0}, "description": "System wavelength indices; empty = all."},
                "pupil_samples": {"minimum": 2, "maximum": 4096, "description": "Grid cells across the overfilled pupil square."},
                "overfill": {"minimum": 1.0, "description": "Sampled pupil radius in entrance-pupil radii."},
                "map_bins": {"items": {"type": "integer", "minimum": 1}, "description": "Map bins [nx, ny]."},
                "map_half_size": {"type": ["array", "null"], "items": {"type": "number"}, "unit": "mm", "description": "Map half-size [x, y]; null = the sensor rectangle, else the image clear aperture."},
                "keep_path_maps": {"description": "Also return every path's map (buffer path_maps)."}
            }),
        ))
    }
    fn result_schema(&self) -> Value {
        let field = obj(&[
            (
                "field",
                opaque(
                    "{\"type\": \"system\", \"index\": i} or {\"type\": \"point\", \"point\": {hx, hy}}",
                ),
            ),
            (
                "flux",
                num(
                    "",
                    "Detected flux (on or off the map), relative to the entrance-pupil input.",
                ),
            ),
            ("flux_on_map", num("", "")),
            ("peak", num("1/mm²", "Peak irradiance of this field's map.")),
            ("centroid", tuple(nf(), 2, "mm")),
            ("rms_radius", nf_u("mm", "")),
            ("rays_traced", int("")),
            ("rays_detected", int("")),
        ]);
        let path = obj(&[
            ("path", path_schema()),
            ("label", string("")),
            ("fields", arr(field, "")),
            (
                "flux_ratio",
                num("", "Flux on the map / primary flux on the map."),
            ),
            (
                "peak_ratio",
                num("", "Max over fields of peak / primary peak."),
            ),
            ("flux", num("", "Flux on the map, all fields.")),
            (
                "map_index",
                nullable(int("Index into path_maps when keep_path_maps.")),
            ),
        ]);
        result(
            &[
                (
                    "map",
                    obj(&[
                        ("bins", tuple(int(""), 2, "[nx, ny]")),
                        (
                            "half_size",
                            tuple(
                                num("mm", ""),
                                2,
                                "[x, y]; the maps span −half_size … +half_size.",
                            ),
                        ),
                    ]),
                ),
                ("primary", path.clone()),
                (
                    "ghosts",
                    arr(path, "Ranked by peak_ratio, strongest first."),
                ),
                (
                    "total_peak_ratio",
                    num("", "Total ghost peak / primary peak."),
                ),
                (
                    "total_flux_ratio",
                    num("", "Total ghost flux / primary flux."),
                ),
                ("truncated", boolean("")),
                ("coverage", coverage_schema()),
            ],
            vec![
                buffer(
                    "primary_map",
                    "f64",
                    &["ny", "nx"],
                    "1/mm²",
                    "Primary irradiance (flux per mm², relative to the entrance-pupil input); row 0 at y = −half_y, x fastest.",
                ),
                buffer(
                    "total_ghost_map",
                    "f64",
                    &["ny", "nx"],
                    "1/mm²",
                    "Sum of every ghost's irradiance.",
                ),
                buffer(
                    "path_maps",
                    "f64",
                    &["G", "ny", "nx"],
                    "1/mm²",
                    "Per-ghost maps in `ghosts` order (keep_path_maps only).",
                ),
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
        let s: GhostIrradianceSettings = parse(settings)?;
        if s.map_bins[0] == 0 || s.map_bins[1] == 0 {
            return Err(EngineError::InvalidSettings("map_bins must be ≥ 1".into()));
        }
        let model = s.model.model(input)?;
        let r = if let Some(d) = super::gpu::device(input) {
            let r = mklight_ghost::gpu::ghost_irradiance(
                d.backend.as_ref(),
                &model,
                &s.irradiance(),
                ctx,
            )?;
            super::gpu::record_native(input, &r.run);
            r
        } else {
            ghost_irradiance(&model, &s.irradiance(), ctx)?
        };
        // Maps move into typed buffers; only the summary is serialised.
        let mut r = r;
        let total = std::mem::take(&mut r.total_ghost_map.values);
        let primary_map = r.primary.map.take();
        let (nx, ny) = (s.map_bins[0], s.map_bins[1]);
        let mut path_maps = Vec::new();
        let mut count = 0usize;
        let mut indices = Vec::with_capacity(r.ghosts.len());
        for g in &mut r.ghosts {
            ctx.check()?;
            let m = g.map.take().filter(map_complete);
            indices.push(m.map(|m| {
                if s.keep_path_maps {
                    path_maps.extend(m.values.into_iter().map(plain));
                }
                count += 1;
                count - 1
            }));
        }
        let (mut v, w) = crate_json(&r)?;
        let mut buffers = Vec::new();
        let IrradianceMap {
            bins, half_size, ..
        } = r.total_ghost_map;
        buffers.extend(map_buffer(
            "total_ghost_map",
            IrradianceMap {
                bins,
                half_size,
                values: total,
            },
        ));
        buffers.extend(primary_map.and_then(|m| map_buffer("primary_map", m)));
        if let Some(gs) = v["ghosts"].as_array_mut() {
            for (g, idx) in gs.iter_mut().zip(indices) {
                g.as_object_mut().map(|o| o.remove("map"));
                g["map_index"] = json!(idx);
            }
        }
        if let Some(p) = v["primary"].as_object_mut() {
            p.remove("map");
            p.insert("map_index".into(), Value::Null);
        }
        if s.keep_path_maps {
            buffers.push(Buffer::new(
                "path_maps",
                vec![count, ny, nx],
                BufferData::F64(path_maps),
            ));
        }
        let geometry = json!({"bins": bins, "half_size": half_size});
        if let Some(o) = v.as_object_mut() {
            o.remove("total_ghost_map");
            o.insert("map".into(), geometry);
        }
        Ok(output(v, buffers, w))
    }
}
