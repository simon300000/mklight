//! Stray light (`mklight-stray`), partitioned against the explicit ghost
//! solver.
//!
//! `stray_light` runs the non-sequential Monte Carlo
//! (`mklight_stray::analysis::analyze`). Maps travel as typed buffers.
//!
//! **Partition** (CONVENTIONS §10, docs/STRAY.md "Partition"). When the
//! request sets `ghost_partition` (the explicit ghost solver's result is
//! also used, with the same ghost-model options as its `ghost_paths` /
//! `ghost_irradiance` run), the engine
//!
//! 1. builds the ghost model and its coverage (the exact path set the
//!    ghost run counts, with its clipping apertures);
//! 2. hands the ghost run's clipping radii to the stray scene as its clear
//!    semi-diameters, so both solvers clip the optical surfaces identically;
//! 3. sets `exclude_ghost_class`, so `Ghost` paths leave the stray total and
//!    maps (detected = Primary + Ghost (ghost solver) + stray total);
//! 4. after the run, rebuilds the stray scene's clipping per surface and
//!    checks it with `GhostCoverage::same_apertures`; a mismatch (for
//!    example a ghost run without clear-aperture clipping, which the stray
//!    scene cannot reproduce) is an error, never a silently inconsistent
//!    partition.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::{JobContext, MkError, ResolvedSystem, SurfaceShape};
use mklight_ghost::{Clip, ClipMode, EnumerationSettings, GhostPath, InterfaceKind, enumerate};
use mklight_stray::PathClass;
use mklight_stray::analysis::{StrayAnalysisSettings, analyze};

use super::geometric::{crate_json, output, parse};
use super::ghost::GhostModelOptions;
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};

pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(StrayLight));
}

/// Engine default for source rays (the crate's is 100 000): interactive
/// first, raise for converged maps.
pub const DEFAULT_STRAY_RAYS: u64 = 20_000;

/// `StrayAnalysisSettings` with the engine's defaults.
#[derive(Serialize)]
#[serde(transparent)]
struct EngineStraySettings(StrayAnalysisSettings);

impl Default for EngineStraySettings {
    fn default() -> Self {
        EngineStraySettings(StrayAnalysisSettings {
            rays: DEFAULT_STRAY_RAYS,
            ..Default::default()
        })
    }
}

/// The ghost run the stray result is combined with (`ghost_partition`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GhostPartition {
    #[serde(flatten)]
    pub model: GhostModelOptions,
    pub enumeration: EnumerationSettings,
    pub paths: Option<Vec<GhostPath>>,
}

/// The clipping the stray scene applies at each surface, in the ghost
/// solver's terms (module docs): the object, coordinate breaks, the image
/// (the detector) and passive surfaces other than the stop do not clip; a
/// surface with its own fixed aperture clips there (`Surface`); every other
/// optical surface, and an automatic stop, clips at the scene's clear
/// semi-diameter (beyond it lies an opaque mount or stop plate).
pub fn stray_clips(
    rs: &ResolvedSystem,
    kinds: &[InterfaceKind],
    clear: &[Option<f64>],
) -> Vec<Clip> {
    let n = rs.num_surfaces();
    (0..n)
        .map(|j| {
            let s = &rs.surfaces[j];
            if j == 0 || matches!(s.shape, SurfaceShape::CoordinateBreak) {
                return Clip::None;
            }
            if s.aperture.is_some() || s.fixed_semi_diameter.is_some() {
                return Clip::Surface;
            }
            let radius = clear.get(j).copied().flatten();
            if j == rs.stop {
                return radius.map_or(Clip::None, |radius| Clip::Radius { radius });
            }
            if j == n - 1 || kinds.get(j) == Some(&InterfaceKind::Passive) {
                return Clip::None;
            }
            radius.map_or(Clip::None, |radius| Clip::Radius { radius })
        })
        .collect()
}

pub struct StrayLight;

impl Analysis for StrayLight {
    fn name(&self) -> &str {
        "stray_light"
    }
    fn title(&self) -> &str {
        "Stray light"
    }
    fn category(&self) -> &str {
        "stray"
    }
    fn description(&self) -> &str {
        "Non-sequential Monte-Carlo stray light: detector irradiance per path class \
         (primary, ghost, scatter, mechanical, sneak) with standard errors, the stray total, \
         the strongest path signatures and an energy balance. With `ghost_partition` the \
         ghost class is left to the explicit ghost solver, with identical apertures."
    }
    fn settings_schema(&self) -> Value {
        let mut s = settings_from_default::<EngineStraySettings>(json!({
            "source": {"description": "Source: {\"type\": \"system_field\", \"index\", \"margin\", \"strength\"}, {\"type\": \"field\", \"hx\", \"hy\", \"margin\", \"strength\"}, {\"type\": \"out_of_field\", \"angle_x_deg\", \"angle_y_deg\", \"margin\", \"irradiance\"} or {\"type\": \"custom\", \"spec\", \"reference_flux\"}."},
            "rays": {"minimum": 1, "maximum": 1_000_000_000u64, "description": "Source rays."},
            "seed": {"minimum": 0, "description": "Philox seed; ray i uses stream (seed, i)."},
            "classes": {"minimum": 0, "maximum": 31, "description": "Classes in the stray total and maps, a bit mask: 1 primary, 2 ghost, 4 scatter, 8 mechanical, 16 sneak (default 30: all but primary)."},
            "exclude_ghost_class": {"description": "Remove the ghost class from the total (set automatically by ghost_partition)."},
            "detector_pixels": {"items": {"type": "integer", "minimum": 1}, "description": "Detector map [nx, ny]."},
            "top_signatures": {"minimum": 0, "description": "Number of path signatures reported."},
            "scene": {"description": "Scene construction (mklight_stray::SceneOptions); clear_semi_diameters is replaced under ghost_partition."},
            "semi_diameters": {"description": "Sizing traces of the automatic clear semi-diameters."},
            "detector_margin": {"minimum": 0.0, "description": "Relative detector margin over the traced image height (automatic image surface without a sensor)."},
            "transport": {"description": "Monte-Carlo transport controls (mode, max_events, roulette_threshold, split_ceiling, scatter_reflection, scatter_transmission, chunk_rays)."}
        }));
        s["properties"]["ghost_partition"] = json!({
            "type": ["object", "null"], "default": null,
            "description": "Partition against the explicit ghost solver: null (stray alone), or the ghost run's model options {sensor_reflectance, clip, reject_virtual, enumeration, paths} (the same as its ghost_paths / ghost_irradiance settings)."
        });
        s
    }
    fn seed(&self, settings: &Value) -> Option<u64> {
        settings["seed"].as_u64()
    }
    fn result_schema(&self) -> Value {
        let class_total = obj(&[
            (
                "class",
                enumeration(&["primary", "ghost", "scatter", "mechanical", "sneak"], ""),
            ),
            ("flux", num("W", "")),
            ("std_error", num("W", "Monte-Carlo standard error.")),
            ("fraction", nullable(num("", "flux / reference_flux."))),
            ("included", boolean("Part of the stray total.")),
            ("hits", int("Detector hits.")),
            (
                "moments",
                opaque(
                    "Image moments: flux, centroid, centroid_std_error, rms_radius, rms_radius_std_error (detector local mm).",
                ),
            ),
        ]);
        let partition = obj(&[
            ("excluded_class", string("Always `ghost`.")),
            (
                "same_apertures",
                boolean(
                    "GhostCoverage::same_apertures held (always true: a mismatch is an error).",
                ),
            ),
            (
                "apertures",
                arr(opaque(""), "The shared clipping per surface."),
            ),
            ("ghost_paths", int("Paths in the ghost run's coverage.")),
            (
                "ghost_class_flux_mc",
                num(
                    "W",
                    "Monte-Carlo estimate of the excluded ghost class (comparison only).",
                ),
            ),
            ("ghost_class_std_error", num("W", "")),
        ]);
        result(
            &[
                (
                    "source",
                    opaque("Resolved source beam: spec, reference_flux, description."),
                ),
                ("emitted", num("W", "Emitted power.")),
                ("classes", int("Class mask of the stray total and maps.")),
                (
                    "detector",
                    opaque("Detector: surface, nx, ny, half_x, half_y (mm), accept."),
                ),
                (
                    "map_classes",
                    arr(
                        string(""),
                        "Class of each row of class_irradiance, in path-class order.",
                    ),
                ),
                ("stray_flux", num("W", "Stray total.")),
                ("stray_std_error", num("W", "")),
                (
                    "stray_fraction",
                    nullable(num("", "stray_flux / reference_flux.")),
                ),
                ("primary_obstruction", nullable(nf())),
                ("totals", arr(class_total, "Every class, included or not.")),
                (
                    "top_signatures",
                    arr(
                        opaque("{class, events, overflow, label, flux, std_error, hits}"),
                        "Strongest (class, signature) pairs among the included classes.",
                    ),
                ),
                (
                    "energy",
                    opaque(
                        "Energy report: emitted, detected, absorbed, escaped, balance_residual, … (docs/STRAY.md).",
                    ),
                ),
                (
                    "clear_semi_diameters",
                    arr(
                        nullable(num("mm", "")),
                        "Clear semi-diameters the scene used per surface (null: the surface's own).",
                    ),
                ),
                ("warnings", arr(string(""), "")),
                ("partition", nullable(partition)),
            ],
            vec![
                buffer(
                    "class_irradiance",
                    "f64",
                    &["C", "ny", "nx"],
                    "W/mm²",
                    "Irradiance per included class (rows of map_classes); row 0 at y = −half_y, x fastest.",
                ),
                buffer(
                    "class_irradiance_std_error",
                    "f64",
                    &["C", "ny", "nx"],
                    "W/mm²",
                    "Standard error per pixel.",
                ),
                buffer(
                    "total_irradiance",
                    "f64",
                    &["ny", "nx"],
                    "W/mm²",
                    "Stray total.",
                ),
                buffer(
                    "total_irradiance_std_error",
                    "f64",
                    &["ny", "nx"],
                    "W/mm²",
                    "",
                ),
            ],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let mut rest = settings.clone();
        let gp = rest
            .as_object_mut()
            .and_then(|o| o.remove("ghost_partition"))
            .unwrap_or(Value::Null);
        let mut ss: StrayAnalysisSettings = parse(&rest)?;
        let mut warnings = Vec::new();
        let rs = input.resolved()?;
        let partition = if gp.is_null() {
            None
        } else {
            let g: GhostPartition = parse(&gp)?;
            if !matches!(g.model.clip, ClipMode::ClearApertures { .. }) {
                return Err(EngineError::InvalidSettings(
                    "ghost_partition: the stray scene always clips at clear apertures, so the \
                     ghost run must use {\"mode\": \"clear_apertures\"} clipping for a consistent \
                     partition"
                        .into(),
                ));
            }
            let model = g.model.model(input)?;
            let coverage = match &g.paths {
                Some(p) => model.coverage(p.iter().cloned()),
                None => enumerate(&model, &g.enumeration)?.coverage,
            };
            if !ss.scene.clear_semi_diameters.is_empty() {
                warnings.push(
                    "ghost_partition: scene.clear_semi_diameters replaced by the ghost run's \
                     clipping apertures"
                        .into(),
                );
            }
            ss.scene.clear_semi_diameters = model
                .clips()
                .iter()
                .map(|c| match c {
                    Clip::Radius { radius } => Some(*radius),
                    _ => None,
                })
                .collect();
            ss.exclude_ghost_class = true;
            Some((model.interface_kinds().to_vec(), coverage))
        };
        let r = analyze(rs.clone(), &ss, ctx)?;
        let partition_json = match partition {
            None => Value::Null,
            Some((kinds, coverage)) => {
                let clips = stray_clips(&rs, &kinds, &r.clear_semi_diameters);
                if !coverage.same_apertures(&clips) {
                    let differ: Vec<String> = coverage
                        .apertures
                        .iter()
                        .zip(&clips)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .map(|(j, (a, b))| format!("{j} (ghost {a:?}, stray {b:?})"))
                        .collect();
                    return Err(EngineError::Core(MkError::InvalidModel(format!(
                        "ghost and stray solvers clip differently at surfaces {}; the path \
                         partition would count rays twice or not at all",
                        differ.join(", ")
                    ))));
                }
                let g = r.total_of(PathClass::Ghost);
                json!({
                    "excluded_class": "ghost",
                    "same_apertures": true,
                    "apertures": coverage.apertures,
                    "ghost_paths": coverage.paths.len(),
                    "ghost_class_flux_mc": g.flux,
                    "ghost_class_std_error": g.std_error,
                })
            }
        };
        let (mut v, w) = crate_json(&r)?;
        warnings.extend(w);
        let nx = r.detector.nx as usize;
        let ny = r.detector.ny as usize;
        let mut irr = Vec::new();
        let mut err = Vec::new();
        let mut names = Vec::new();
        for m in &r.maps {
            irr.extend_from_slice(&m.irradiance);
            err.extend_from_slice(&m.irradiance_std_error);
            names.push(serde_json::to_value(m.class)?);
        }
        let c = r.maps.len();
        if let Some(o) = v.as_object_mut() {
            for k in ["maps", "total_irradiance", "total_irradiance_std_error"] {
                o.remove(k);
            }
            o.insert("map_classes".into(), Value::Array(names));
            o.insert("partition".into(), partition_json);
        }
        let buffers = vec![
            Buffer::new("class_irradiance", vec![c, ny, nx], BufferData::F64(irr)),
            Buffer::new(
                "class_irradiance_std_error",
                vec![c, ny, nx],
                BufferData::F64(err),
            ),
            Buffer::new(
                "total_irradiance",
                vec![ny, nx],
                BufferData::F64(r.total_irradiance.clone()),
            ),
            Buffer::new(
                "total_irradiance_std_error",
                vec![ny, nx],
                BufferData::F64(r.total_irradiance_std_error.clone()),
            ),
        ];
        Ok(output(v, buffers, warnings))
    }
}
