//! Model analyses (`validate`, `system_summary`, `glass_info`), the
//! `rng_stream` diagnostic, and placeholders that reserve the names of the
//! wave-optics analyses still to be wired in.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::materials::coating::ResolvedCoating;
use mklight_core::materials::{Glass, MaterialLibrary, ResolvedMaterial, lines};
use mklight_core::math::KahanSum;
use mklight_core::rng::Philox;
use mklight_core::{ApertureSpec, Interaction, JobContext, MkError, SemiDiameter, SurfaceShape};

use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};
use crate::validation::{self, Issue};

/// Register every built-in analysis and placeholder (registered first, so
/// the real analyses registered after them replace placeholders).
pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(Validate));
    r.register(Arc::new(SystemSummary));
    r.register(Arc::new(GlassInfo));
    r.register(Arc::new(RngStream));
    for p in PLANNED {
        r.register(Arc::new(Planned(p)));
    }
}

/// `±∞` as the strings `"inf"`/`"-inf"` (CONVENTIONS §2), NaN as null.
fn num(x: f64) -> Value {
    if x.is_nan() {
        Value::Null
    } else if x.is_infinite() {
        json!(if x > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(x)
    }
}

/// Abbe number `(n_d − 1)/(n_F − n_C)` of a medium, `None` when it does not
/// disperse (air, constant index).
fn abbe(n: impl Fn(f64) -> f64) -> Option<f64> {
    let dfc = n(lines::F) - n(lines::C);
    (dfc.abs() > 1e-12).then(|| (n(lines::D) - 1.0) / dfc)
}

fn parse<T: for<'de> Deserialize<'de>>(settings: &Value) -> EngineResult<T> {
    serde_json::from_value(settings.clone())
        .map_err(|e| EngineError::InvalidSettings(e.to_string()))
}

// ---------------------------------------------------------------------------
// validate

/// Model checks beyond structural validation: material and coating
/// resolution per surface, data-range warnings, aperture definition,
/// thickness signs after mirrors and zero weights, plus the
/// engine-level tracer setup and reachability checks of `crate::validation`.
/// Structural checks have already run when the snapshot was created.
pub struct Validate;

impl Analysis for Validate {
    fn name(&self) -> &str {
        "validate"
    }
    fn title(&self) -> &str {
        "Validate system"
    }
    fn category(&self) -> &str {
        "model"
    }
    fn description(&self) -> &str {
        "Checks that the system is complete and consistent: every number is finite, \
         materials and coatings resolve, wavelengths lie inside glass data ranges, the \
         aperture is defined and positive, weights are not negative, mechanical parts \
         reference existing surfaces, thickness signs follow mirrors, configurations apply, \
         the reference tracer can set the system up, and the chief and marginal rays reach \
         the image."
    }
    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                ("ok", boolean("No error-severity issue.")),
                ("errors", int("")),
                ("warnings", int("")),
                (
                    "issues",
                    arr(
                        obj(&[
                            ("severity", enumeration(&["error", "warning", "info"], "")),
                            ("code", string("Issue code (docs/RESULTS.md).")),
                            ("surface", nullable(int("Surface index."))),
                            ("message", string("")),
                        ]),
                        "",
                    ),
                ),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        _: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let sys = input.snapshot.system();
        let env = sys.environment;
        let lib = &input.library;
        let n = sys.surfaces.len();
        let mut issues = Vec::new();
        let mut push = |severity, code, surface, message: String| {
            issues.push(Issue::new(severity, code, surface, message))
        };

        // Materials, per surface, so every unresolved one is reported.
        let mut material_errors = 0;
        for (i, s) in sys.surfaces.iter().enumerate() {
            let used =
                i == 0 || (s.interaction == Interaction::Refract && !s.is_coordinate_break());
            if used && let Err(e) = lib.resolve(&s.material, &env) {
                material_errors += 1;
                push("error", "material", Some(i), e.to_string());
            }
        }
        for (name, def) in &sys.coatings {
            if let Err(e) = ResolvedCoating::resolve(Some(def), lib, &env) {
                push("error", "coating", None, format!("coating '{name}': {e}"));
            }
            if !sys
                .surfaces
                .iter()
                .any(|s| s.coating.as_deref() == Some(name))
            {
                push(
                    "info",
                    "unused_coating",
                    None,
                    format!("coating '{name}' is not used"),
                );
            }
        }
        ctx.check()?;
        let mut resolved = None;
        if material_errors == 0 {
            match input.resolved() {
                Ok(r) => {
                    for w in &r.warnings {
                        push("warning", "material_range", None, w.clone());
                    }
                    resolved = Some(r);
                }
                Err(e) => push("error", "resolve", None, e.to_string()),
            }
        }

        let stop = &sys.surfaces[sys.stop.min(n - 1)];
        if matches!(sys.aperture, ApertureSpec::FloatByStopSize)
            && stop.semi_diameter == SemiDiameter::Auto
            && stop.aperture.is_none()
        {
            push(
                "error",
                "stop_size_undefined",
                Some(sys.stop),
                "the aperture floats by stop size, but the stop has no fixed semi-diameter".into(),
            );
        }
        if stop.is_coordinate_break() {
            push(
                "warning",
                "stop_on_coordinate_break",
                Some(sys.stop),
                "the stop is a coordinate break, which has no physical aperture".into(),
            );
        }

        // After an odd number of mirrors light travels toward −z, so gaps
        // must be negative (CONVENTIONS §2). Tilted frames make the direction
        // ambiguous, so the check is skipped for them.
        let tilted = sys.surfaces.iter().any(|s| {
            s.placement
                .is_some_and(|p| p.tilt_x != 0.0 || p.tilt_y != 0.0)
        });
        if !tilted {
            let mut dir = 1.0;
            for (i, s) in sys.surfaces.iter().enumerate().take(n - 1).skip(1) {
                if s.interaction == Interaction::Reflect {
                    dir = -dir;
                }
                if s.thickness * dir < 0.0 {
                    push(
                        "warning",
                        "thickness_sign",
                        Some(i),
                        format!(
                            "thickness {} has the opposite sign to the propagation direction \
                             ({} mirror(s) before it)",
                            s.thickness,
                            sys.surfaces[1..=i]
                                .iter()
                                .filter(|s| s.interaction == Interaction::Reflect)
                                .count()
                        ),
                    );
                }
            }
        }

        if sys.wavelengths.wavelengths.iter().all(|w| w.weight == 0.0) {
            push(
                "warning",
                "zero_weights",
                None,
                "all wavelength weights are zero".into(),
            );
        }
        if sys.fields.fields.iter().all(|f| f.weight == 0.0) {
            push(
                "warning",
                "zero_weights",
                None,
                "all field weights are zero".into(),
            );
        }
        ctx.check()?;
        if let Some(r) = resolved {
            issues.extend(validation::tracer_checks(r));
        }

        let errors = issues.iter().filter(|i| i.severity == "error").count();
        let warnings = issues.iter().filter(|i| i.severity == "warning").count();
        Ok(AnalysisOutput {
            json: json!({
                "ok": errors == 0,
                "errors": errors,
                "warnings": warnings,
                "issues": issues,
            }),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// system_summary

/// The surface table with resolved media: index after each surface at every
/// system wavelength, `n_d`/`V_d`, and every surface's global frame.
pub struct SystemSummary;

fn shape_name(s: &SurfaceShape) -> &'static str {
    match s {
        SurfaceShape::Conic { conic, .. } if *conic == 0.0 => "sphere",
        s => s.type_name(),
    }
}

impl Analysis for SystemSummary {
    fn name(&self) -> &str {
        "system_summary"
    }
    fn title(&self) -> &str {
        "System summary"
    }
    fn category(&self) -> &str {
        "model"
    }
    fn description(&self) -> &str {
        "Surface table with resolved refractive indices at every system wavelength and \
         the global frame (vertex position and orientation) of every surface. Buffers: \
         `n_after` [surfaces × wavelengths], `vertex_global` [surfaces × 3] (mm)."
    }
    fn result_schema(&self) -> Value {
        use super::schema::*;
        let row = obj(&[
            ("index", int("Surface index.")),
            ("id", int("Stable surface id.")),
            ("label", string("")),
            (
                "type",
                enumeration(
                    &[
                        "plane",
                        "sphere",
                        "conic",
                        "even_asphere",
                        "odd_asphere",
                        "q_con",
                        "q_bfs",
                        "diffractive",
                        "toroid",
                        "biconic",
                        "xy_polynomial",
                        "zernike_sag",
                        "paraxial",
                        "coordinate_break",
                    ],
                    "",
                ),
            ),
            ("is_stop", boolean("")),
            ("curvature", num("1/mm", "")),
            ("radius", nf_u("mm", "1/curvature; \"inf\" for a plane.")),
            ("conic", num("", "")),
            ("shape", opaque("The model's SurfaceShape.")),
            (
                "thickness",
                nf_u(
                    "mm",
                    "To the next vertex (\"inf\" for an object at infinity).",
                ),
            ),
            (
                "material",
                nullable(string("Display name of the medium after the surface.")),
            ),
            ("material_ref", any("The model's MaterialRef.")),
            ("interaction", string("refract or reflect.")),
            ("semi_diameter", any("The model's SemiDiameter.")),
            ("aperture", any("Resolved aperture shape or null.")),
            ("coating", nullable(string(""))),
            (
                "n_after",
                arr(
                    num("", ""),
                    "Index after the surface at every system wavelength (relative to air).",
                ),
            ),
            ("nd_after", nf_u("", "")),
            ("vd_after", nullable(nf())),
            (
                "alpha_after_per_mm",
                arr(
                    num("1/mm", ""),
                    "Absorption coefficient per system wavelength.",
                ),
            ),
            (
                "frame",
                nullable(obj(&[
                    ("origin", tuple(num("mm", ""), 3, "Global vertex.")),
                    (
                        "rot",
                        any(
                            "3×3 rotation, rows; columns are the local axes in global coordinates.",
                        ),
                    ),
                ])),
            ),
            ("to_next_is_axial", boolean("")),
        ]);
        result(
            &[
                (
                    "symmetry",
                    enumeration(
                        &["rotational", "meridional", "none"],
                        "System geometry symmetry.",
                    ),
                ),
                ("paraxial_xy", super::geometric::paraxial_xy_schema()),
                ("name", string("")),
                ("num_surfaces", int("")),
                ("stop", int("")),
                ("object_at_infinity", boolean("")),
                ("wavelengths_um", arr(num("µm", ""), "")),
                ("wavelength_weights", arr(num("", ""), "")),
                ("primary_wavelength", int("")),
                ("aperture", opaque("The model's ApertureSpec.")),
                ("fields", opaque("The model's FieldSet.")),
                ("ray_aiming", opaque("")),
                (
                    "environment",
                    opaque("Temperature (°C) and pressure (atm)."),
                ),
                (
                    "n_object",
                    arr(num("", ""), "Object-space index per wavelength."),
                ),
                ("surfaces", arr(row, "")),
                (
                    "first_to_image_distance",
                    num("mm", "Distance from surface 1 to the image vertex."),
                ),
                ("num_configurations", int("")),
                ("warnings", arr(string(""), "")),
            ],
            vec![
                buffer(
                    "n_after",
                    "f64",
                    &["surfaces", "wavelengths"],
                    "",
                    "Index after each surface.",
                ),
                buffer(
                    "vertex_global",
                    "f64",
                    &["surfaces", "3"],
                    "mm",
                    "Global vertex (NaN for an object at infinity).",
                ),
            ],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        _: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let r = input.resolved()?;
        let sys = r.system();
        let nw = r.wavelengths_um.len();
        let mut rows = Vec::with_capacity(r.surfaces.len());
        let mut n_after = Vec::with_capacity(r.surfaces.len() * nw);
        let mut vertices = Vec::with_capacity(r.surfaces.len() * 3);
        for (i, rs) in r.surfaces.iter().enumerate() {
            ctx.check()?;
            let s = &sys.surfaces[i];
            let c = rs.shape.curvature();
            let m = &rs.medium_after;
            let refracting_medium = !matches!(m, ResolvedMaterial::Air);
            // The object frame is meaningless for an object at infinity.
            let frame = if i == 0 && r.object_at_infinity {
                Value::Null
            } else {
                json!({
                    "origin": [rs.frame.origin.x, rs.frame.origin.y, rs.frame.origin.z],
                    "rot": rs.frame.rot.m,
                })
            };
            let v = if i == 0 && r.object_at_infinity {
                [f64::NAN; 3]
            } else {
                [rs.frame.origin.x, rs.frame.origin.y, rs.frame.origin.z]
            };
            vertices.extend_from_slice(&v);
            n_after.extend_from_slice(&rs.n_after);
            rows.push(json!({
                "index": i,
                "id": rs.id,
                "label": s.label,
                "type": shape_name(&rs.shape),
                "is_stop": i == r.stop,
                "curvature": c,
                "radius": num(if c == 0.0 { f64::INFINITY } else { 1.0 / c }),
                "conic": rs.shape.conic(),
                "shape": rs.shape,
                "thickness": num(rs.thickness),
                "material": s.material.display_name(),
                "material_ref": s.material,
                "interaction": rs.interaction,
                "semi_diameter": s.semi_diameter,
                "aperture": rs.aperture,
                "coating": s.coating,
                "n_after": rs.n_after,
                "nd_after": if refracting_medium { num(m.index(lines::D)) } else { json!(1.0) },
                "vd_after": abbe(|l| m.index(l)).map_or(Value::Null, num),
                "alpha_after_per_mm": rs.alpha_after,
                "frame": frame,
                "to_next_is_axial": rs.to_next_is_axial,
            }));
            ctx.report((i + 1) as f64 / r.surfaces.len() as f64, "surface table");
        }
        let xy = mklight_analysis::system_data::paraxial_xy_report(&r)?;
        let warnings = xy.warnings.clone();
        let first = &r.surfaces[1].frame.origin;
        let img = &r.surfaces[r.image_index()].frame.origin;
        let track =
            ((img.x - first.x).powi(2) + (img.y - first.y).powi(2) + (img.z - first.z).powi(2))
                .sqrt();
        let ns = r.surfaces.len();
        Ok(AnalysisOutput {
            json: json!({
                "symmetry": xy.symmetry,
                "paraxial_xy": xy,
                "name": sys.name,
                "num_surfaces": ns,
                "stop": r.stop,
                "object_at_infinity": r.object_at_infinity,
                "wavelengths_um": r.wavelengths_um,
                "wavelength_weights": r.wavelength_weights,
                "primary_wavelength": r.primary_wavelength,
                "aperture": sys.aperture,
                "fields": sys.fields,
                "ray_aiming": sys.ray_aiming,
                "environment": sys.environment,
                "n_object": r.n_object,
                "surfaces": rows,
                "first_to_image_distance": track,
                "num_configurations": sys.configurations.len(),
                "warnings": r.warnings,
            }),
            buffers: vec![
                Buffer::new("n_after", vec![ns, nw], BufferData::F64(n_after)),
                Buffer::new("vertex_global", vec![ns, 3], BufferData::F64(vertices)),
            ],
            warnings,
        })
    }
}

// ---------------------------------------------------------------------------
// glass_info

/// Catalog data and computed indices of one glass.
pub struct GlassInfo;

#[derive(Deserialize)]
struct GlassInfoSettings {
    name: String,
    catalog: Option<String>,
    wavelengths_um: Option<Vec<f64>>,
    temperature_c: Option<f64>,
    pressure_atm: Option<f64>,
    curve_samples: usize,
}

/// Same lookup rule as `MaterialLibrary::find_glass` (catalogs in load order,
/// exact name first, then case-insensitive), but also reports the catalog.
pub fn find_glass_with_catalog<'a>(
    lib: &'a MaterialLibrary,
    catalog: Option<&str>,
    name: &str,
) -> Option<(&'a str, &'a Glass)> {
    lib.catalogs()
        .iter()
        .filter(|c| catalog.is_none_or(|n| c.name.eq_ignore_ascii_case(n)))
        .find_map(|c| {
            c.glasses
                .get(name)
                .or_else(|| {
                    c.glasses
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(name))
                        .map(|(_, g)| g)
                })
                .map(|g| (c.name.as_str(), g))
        })
}

impl Analysis for GlassInfo {
    fn name(&self) -> &str {
        "glass_info"
    }
    fn title(&self) -> &str {
        "Glass data"
    }
    fn category(&self) -> &str {
        "materials"
    }
    fn description(&self) -> &str {
        "Catalog data of one glass and its index relative to air at chosen wavelengths and \
         environment. Buffer `dispersion_curve` [samples × 2] holds (λ µm, n) over the \
         glass's valid range."
    }
    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                ("catalog", string("Catalog the glass was found in.")),
                ("name", string("")),
                ("catalog_nd", num("", "Catalog n_d.")),
                ("catalog_vd", num("", "Catalog V_d.")),
                (
                    "nd",
                    num("", "Computed at the d line (catalog conditions)."),
                ),
                ("ne", num("", "")),
                ("vd", nf()),
                ("pgf", nf_u("", "Partial dispersion P_g,F.")),
                ("dispersion", opaque("Dispersion formula and coefficients.")),
                ("min_um", num("µm", "")),
                ("max_um", num("µm", "")),
                ("has_thermal_data", boolean("")),
                ("cte", any("Thermal expansion (1/K) or null.")),
                ("density", any("g/cm³ or null.")),
                ("relative_cost", any("")),
                ("comment", any("")),
                ("environment", opaque("Temperature and pressure used.")),
                (
                    "indices",
                    arr(
                        obj(&[
                            ("wavelength_um", num("µm", "")),
                            ("n", nf_u("", "Index relative to air at the environment.")),
                            ("n_catalog", nf_u("", "At catalog conditions.")),
                            ("absorption_per_mm", num("1/mm", "")),
                            ("in_range", boolean("")),
                        ]),
                        "",
                    ),
                ),
            ],
            vec![buffer(
                "dispersion_curve",
                "f64",
                &["samples", "2"],
                "",
                "(λ µm, n) over the glass's range.",
            )],
        )
    }
    fn settings_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "title": "Glass"},
                "catalog": {"type": ["string", "null"], "default": null,
                            "description": "Catalog to search; null searches all in load order."},
                "wavelengths_um": {"type": ["array", "null"], "default": null,
                                   "items": {"type": "number", "minimum": 0.1, "maximum": 30.0},
                                   "unit": "µm",
                                   "description": "null uses the system wavelengths."},
                "temperature_c": {"type": ["number", "null"], "default": null, "unit": "°C",
                                  "description": "null uses the system environment."},
                "pressure_atm": {"type": ["number", "null"], "default": null, "minimum": 0.0,
                                 "unit": "atm"},
                "curve_samples": {"type": "integer", "default": 0, "minimum": 0,
                                  "maximum": 100000}
            },
            "required": ["name"]
        })
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let s: GlassInfoSettings = parse(settings)?;
        let sys = input.snapshot.system();
        let (cat, glass) = find_glass_with_catalog(&input.library, s.catalog.as_deref(), &s.name)
            .ok_or_else(|| {
            MkError::UnknownMaterial(match &s.catalog {
                Some(c) => format!("{c}:{}", s.name),
                None => s.name.clone(),
            })
        })?;
        let mut env = sys.environment;
        if let Some(t) = s.temperature_c {
            env.temperature_c = t;
        }
        if let Some(p) = s.pressure_atm {
            env.pressure_atm = p;
        }
        let m = ResolvedMaterial::Glass {
            glass: Arc::new(glass.clone()),
            env,
        };
        let wls = s
            .wavelengths_um
            .unwrap_or_else(|| sys.wavelengths.wavelengths.iter().map(|w| w.um).collect());
        let mut warnings = Vec::new();
        let indices: Vec<Value> = wls
            .iter()
            .map(|&w| {
                let in_range = w >= glass.min_um - 1e-9 && w <= glass.max_um + 1e-9;
                if !in_range {
                    warnings.push(format!(
                        "{}: {w} µm is outside the data range {}–{} µm",
                        glass.name, glass.min_um, glass.max_um
                    ));
                }
                json!({
                    "wavelength_um": w,
                    "n": num(m.index(w)),
                    "n_catalog": num(glass.catalog_index(w)),
                    "absorption_per_mm": glass.absorption_per_mm(w),
                    "in_range": in_range,
                })
            })
            .collect();
        let d = |l: f64| glass.catalog_index(l);
        let dfc = d(lines::F) - d(lines::C);
        let mut buffers = Vec::new();
        if s.curve_samples > 0 {
            let k = s.curve_samples;
            let mut curve = Vec::with_capacity(2 * k);
            for j in 0..k {
                if j % 4096 == 0 {
                    ctx.check()?;
                    ctx.report(j as f64 / k as f64, "dispersion curve");
                }
                let t = if k == 1 {
                    0.5
                } else {
                    j as f64 / (k - 1) as f64
                };
                let l = glass.min_um + t * (glass.max_um - glass.min_um);
                curve.push(l);
                curve.push(m.index(l));
            }
            buffers.push(Buffer::new(
                "dispersion_curve",
                vec![k, 2],
                BufferData::F64(curve),
            ));
        }
        Ok(AnalysisOutput {
            json: json!({
                "catalog": cat,
                "name": glass.name,
                "catalog_nd": glass.nd,
                "catalog_vd": glass.vd,
                "nd": d(lines::D),
                "ne": d(lines::E),
                "vd": num((d(lines::D) - 1.0) / dfc),
                "pgf": num((d(lines::G) - d(lines::F)) / dfc),
                "dispersion": glass.dispersion,
                "min_um": glass.min_um,
                "max_um": glass.max_um,
                "has_thermal_data": glass.thermal.is_some(),
                "cte": glass.cte,
                "density": glass.density,
                "relative_cost": glass.relative_cost,
                "comment": glass.comment,
                "environment": env,
                "indices": indices,
            }),
            buffers,
            warnings,
        })
    }
}

// ---------------------------------------------------------------------------
// rng_stream (diagnostic)

/// Draws a Philox uniform stream. Used to check that random streams are
/// reproducible across runs, front ends and (later) backends, and to
/// exercise job progress and cancellation end to end.
pub struct RngStream;

#[derive(Deserialize)]
struct RngSettings {
    seed: u64,
    stream: u64,
    count: usize,
    chunk: usize,
    delay_ms: f64,
}

impl Analysis for RngStream {
    fn name(&self) -> &str {
        "rng_stream"
    }
    fn title(&self) -> &str {
        "Random stream check"
    }
    fn category(&self) -> &str {
        "diagnostic"
    }
    fn description(&self) -> &str {
        "Draws `count` Philox4x32-10 uniforms for (seed, stream) and reports their mean \
         and variance (expected 1/2 and 1/12). Buffer `uniform` [count]. `delay_ms` \
         sleeps after each chunk, to test progress and cancellation."
    }
    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                ("count", int("")),
                ("mean", num("", "")),
                ("variance", num("", "")),
                ("min", num("", "")),
                ("max", num("", "")),
                ("first", arr(num("", ""), "The first (up to) four draws.")),
            ],
            vec![buffer(
                "uniform",
                "f64",
                &["count"],
                "",
                "The draws, in order.",
            )],
        )
    }
    fn settings_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "seed": {"type": "integer", "default": 0, "minimum": 0},
                "stream": {"type": "integer", "default": 0, "minimum": 0},
                "count": {"type": "integer", "default": 100000, "minimum": 1,
                          "maximum": 50000000},
                "chunk": {"type": "integer", "default": 65536, "minimum": 1},
                "delay_ms": {"type": "number", "default": 0.0, "minimum": 0.0,
                             "maximum": 1000.0}
            }
        })
    }
    fn run(
        &self,
        _: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let s: RngSettings = parse(settings)?;
        let mut rng = Philox::new(s.seed, s.stream);
        let mut v = Vec::with_capacity(s.count.min(1 << 20));
        let mut sum = KahanSum::default();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        while v.len() < s.count {
            ctx.check()?;
            let end = (v.len() + s.chunk).min(s.count);
            while v.len() < end {
                let u = rng.uniform();
                sum.add(u);
                lo = lo.min(u);
                hi = hi.max(u);
                v.push(u);
            }
            ctx.report(v.len() as f64 / s.count as f64, "drawing");
            if s.delay_ms > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(s.delay_ms * 1e-3));
            }
        }
        ctx.check()?;
        let n = v.len() as f64;
        let mean = sum.value() / n;
        // Two-pass variance with compensated summation (CONVENTIONS §7.5).
        let var = v
            .iter()
            .map(|u| (u - mean) * (u - mean))
            .collect::<KahanSum>()
            .value()
            / n;
        Ok(AnalysisOutput {
            json: json!({
                "count": v.len(),
                "mean": mean,
                "variance": var,
                "min": lo,
                "max": hi,
                "first": &v[..v.len().min(4)],
            }),
            buffers: vec![Buffer::f64_1d("uniform", v)],
            warnings: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Placeholders

/// A reserved analysis name whose solver is not wired in yet.
pub struct PlannedAnalysis {
    pub name: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    /// Crate that will provide it.
    pub provider: &'static str,
}

/// PLACEHOLDERS: the wave-optics analyses `mklight-wave` registers under
/// these names (each replacing its placeholder with one `register` line in
/// `analysis::register_all`). They are listed with `available: false`, and
/// submitting one fails with `AnalysisUnavailable`.
pub const PLANNED: &[PlannedAnalysis] = &[
    PlannedAnalysis {
        name: "wavefront_map",
        title: "Wavefront map",
        category: "wave",
        provider: "mklight-wave",
    },
    PlannedAnalysis {
        name: "psf",
        title: "Point spread function",
        category: "wave",
        provider: "mklight-wave",
    },
    PlannedAnalysis {
        name: "mtf",
        title: "MTF",
        category: "wave",
        provider: "mklight-wave",
    },
];

struct Planned(&'static PlannedAnalysis);

impl Analysis for Planned {
    fn name(&self) -> &str {
        self.0.name
    }
    fn title(&self) -> &str {
        self.0.title
    }
    fn category(&self) -> &str {
        self.0.category
    }
    fn availability(&self) -> Result<(), String> {
        Err(format!(
            "provided by {}, which is not wired into the engine yet",
            self.0.provider
        ))
    }
    fn run(
        &self,
        _: &AnalysisInput<'_>,
        _: &Value,
        _: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        Err(EngineError::AnalysisUnavailable {
            name: self.0.name.into(),
            reason: self.availability().unwrap_err(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::run_analysis;
    use crate::backend::BackendInfo;
    use mklight_core::{MaterialRef, OpticalSystem, Snapshot, Surface};

    fn singlet() -> OpticalSystem {
        let mut s = OpticalSystem::new_empty();
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            Surface::new(2, SurfaceShape::sphere(-50.0), 45.0, MaterialRef::Air),
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        s
    }

    fn run(
        name: &str,
        sys: OpticalSystem,
        settings: Value,
    ) -> EngineResult<super::super::AnalysisResult> {
        // The placeholders again, over the solvers that replace them.
        let mut reg = AnalysisRegistry::with_builtin();
        register_all(&mut reg);
        let a = reg.get(name)?;
        let input = AnalysisInput::new(
            Snapshot::new(sys).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
            BackendInfo::cpu_reference(),
        );
        run_analysis(a.as_ref(), &input, &settings, vec![], &JobContext::new())
    }

    fn codes(r: &Value) -> Vec<String> {
        r["issues"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["code"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn validate_clean_singlet() {
        let r = run("validate", singlet(), Value::Null).unwrap();
        assert_eq!(r.json["ok"], json!(true));
        assert!(codes(&r.json).is_empty(), "{}", r.json);
    }

    #[test]
    fn validate_reports_every_unknown_glass() {
        let mut s = singlet();
        s.surfaces[1].material = MaterialRef::glass("NOPE-1");
        s.surfaces[2].material = MaterialRef::glass("NOPE-2");
        let r = run("validate", s, Value::Null).unwrap();
        assert_eq!(r.json["ok"], json!(false));
        assert_eq!(r.json["errors"], json!(2));
        assert_eq!(r.json["issues"][0]["surface"], json!(1));
        assert_eq!(r.json["issues"][1]["surface"], json!(2));
    }

    #[test]
    fn validate_mirror_thickness_sign() {
        let mut s = singlet();
        s.surfaces[2].interaction = Interaction::Reflect;
        // Positive gap after a mirror is wrong.
        let r = run("validate", s.clone(), Value::Null).unwrap();
        assert_eq!(codes(&r.json), vec!["thickness_sign"]);
        assert_eq!(r.json["issues"][0]["surface"], json!(2));
        s.surfaces[2].thickness = -45.0;
        let r = run("validate", s, Value::Null).unwrap();
        assert!(codes(&r.json).is_empty(), "{}", r.json);
    }

    #[test]
    fn validate_float_by_stop_needs_size() {
        let mut s = singlet();
        s.aperture = ApertureSpec::FloatByStopSize;
        let r = run("validate", s.clone(), Value::Null).unwrap();
        // The reference tracer cannot size the pupil either.
        assert_eq!(codes(&r.json), vec!["stop_size_undefined", "paraxial"]);
        s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 5.0 };
        assert_eq!(
            run("validate", s, Value::Null).unwrap().json["ok"],
            json!(true)
        );
    }

    #[test]
    fn validate_range_warning() {
        let mut s = singlet();
        s.wavelengths.wavelengths[0].um = 3.0; // beyond N-BK7's 2.5 µm
        let r = run("validate", s, Value::Null).unwrap();
        assert_eq!(r.json["ok"], json!(true));
        assert!(codes(&r.json).contains(&"material_range".to_string()));
        assert!(!r.run_info.warnings.is_empty());
    }

    #[test]
    fn summary_indices_and_frames() {
        let r = run("system_summary", singlet(), Value::Null).unwrap();
        let s = &r.json["surfaces"];
        assert_eq!(s[0]["frame"], Value::Null);
        assert_eq!(s[0]["thickness"], json!("inf"));
        assert_eq!(s[1]["type"], json!("sphere"));
        assert!((s[1]["radius"].as_f64().unwrap() - 50.0).abs() < 1e-12);
        assert_eq!(s[3]["radius"], json!("inf"));
        // N-BK7 at the d line (primary, index 1): catalog n_d = 1.5168.
        let nd = s[1]["n_after"][1].as_f64().unwrap();
        assert!((nd - 1.5168).abs() < 2e-5, "{nd}");
        let vd = s[1]["vd_after"].as_f64().unwrap();
        assert!((vd - 64.17).abs() < 0.05, "{vd}");
        assert_eq!(s[2]["vd_after"], Value::Null);
        assert_eq!(s[3]["frame"]["origin"], json!([0.0, 0.0, 50.0]));
        assert_eq!(r.json["first_to_image_distance"], json!(50.0));
        assert_eq!(s[1]["is_stop"], json!(true));
        let b = &r.buffers[0];
        assert_eq!((b.name.as_str(), b.shape.clone()), ("n_after", vec![4, 3]));
        match &b.data {
            BufferData::F64(v) => assert_eq!(v[3 + 1], nd),
            _ => panic!(),
        }
        match &r.buffers[1].data {
            BufferData::F64(v) => assert_eq!(&v[9..12], &[0.0, 0.0, 50.0]),
            _ => panic!(),
        }
    }

    #[test]
    fn summary_fails_on_unknown_glass() {
        let mut s = singlet();
        s.surfaces[1].material = MaterialRef::glass("NOPE");
        assert!(matches!(
            run("system_summary", s, Value::Null),
            Err(EngineError::Core(MkError::UnknownMaterial(_)))
        ));
    }

    #[test]
    fn glass_info_nbk7() {
        let r = run(
            "glass_info",
            singlet(),
            json!({"name": "n-bk7", "wavelengths_um": [0.5875618, 3.0], "curve_samples": 11}),
        )
        .unwrap();
        let j = &r.json;
        assert_eq!(j["catalog"], json!("BUILTIN"));
        assert_eq!(j["name"], json!("N-BK7"));
        assert!((j["nd"].as_f64().unwrap() - 1.5168).abs() < 2e-5);
        assert!((j["vd"].as_f64().unwrap() - 64.17).abs() < 0.05);
        assert_eq!(j["indices"][0]["in_range"], json!(true));
        assert_eq!(j["indices"][1]["in_range"], json!(false));
        assert_eq!(r.run_info.warnings.len(), 1);
        let b = &r.buffers[0];
        assert_eq!(b.shape, vec![11, 2]);
        match &b.data {
            BufferData::F64(v) => {
                assert_eq!(v[0], 0.3);
                assert!((v[20] - 2.5).abs() < 1e-12);
                // Normal dispersion: index falls with wavelength.
                assert!(
                    v.chunks(2)
                        .collect::<Vec<_>>()
                        .windows(2)
                        .all(|w| w[1][1] < w[0][1])
                );
            }
            _ => panic!(),
        }
        assert!(run("glass_info", singlet(), json!({"name": "NOPE"})).is_err());
        assert!(
            run(
                "glass_info",
                singlet(),
                json!({"name": "N-BK7", "catalog": "OTHER"})
            )
            .is_err()
        );
        assert!(matches!(
            run("glass_info", singlet(), Value::Null),
            Err(EngineError::InvalidSettings(_))
        ));
    }

    #[test]
    fn rng_stream_known_answers() {
        let n = 200_000usize;
        let r = run(
            "rng_stream",
            singlet(),
            json!({"seed": 7, "stream": 3, "count": n, "chunk": 1000}),
        )
        .unwrap();
        let mut p = Philox::new(7, 3);
        let first: Vec<f64> = (0..4).map(|_| p.uniform()).collect();
        assert_eq!(r.json["first"], json!(first));
        let mean = r.json["mean"].as_f64().unwrap();
        let var = r.json["variance"].as_f64().unwrap();
        // 5 σ bounds for the sample mean and variance of U(0,1).
        assert!(
            (mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n as f64).sqrt(),
            "{mean}"
        );
        assert!(
            (var - 1.0 / 12.0).abs() < 5.0 * (1.0 / 180.0 / n as f64).sqrt(),
            "{var}"
        );
        assert_eq!(r.run_info.seed, Some(7));
        // Reproducible.
        let r2 = run(
            "rng_stream",
            singlet(),
            json!({"seed": 7, "stream": 3, "count": n}),
        )
        .unwrap();
        assert_eq!(r.buffers, r2.buffers);
    }

    #[test]
    fn placeholders_are_unavailable() {
        let mut reg = AnalysisRegistry::with_builtin();
        register_all(&mut reg);
        let info = reg.list();
        let psf = info.iter().find(|i| i.name == "psf").unwrap();
        assert!(!psf.available && psf.unavailable_reason.is_some());
        let spot = info.iter().find(|i| i.name == "spot_diagram").unwrap();
        assert!(spot.available && spot.gpu_path);
        assert!(
            info.iter()
                .find(|i| i.name == "validate")
                .unwrap()
                .available
        );
        assert!(matches!(
            run("mtf", singlet(), Value::Null),
            Err(EngineError::AnalysisUnavailable { .. })
        ));
        // glass_info needs a name, so it has no complete defaults.
        assert_eq!(
            info.iter()
                .find(|i| i.name == "glass_info")
                .unwrap()
                .default_settings,
            Value::Null
        );
    }
}
