//! Project commands: `new`, `validate`, `show`, `migrate`, `set`.

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::Path;

use mklight_core::param::ParamPath;
use mklight_core::{
    ApertureSpec, Interaction, MkError, OpticalSystem, SemiDiameter, SurfaceShape, samples,
};
use mklight_engine::{EngineError, ParamEdit, Project, Session};
use mklight_trace::{SemiDiameterSettings, SemiDiameterSource, SequentialTracer};

use super::{Ctx, Report, open_warnings, run_sync, save, warning_lines};
use crate::cli::{MigrateArgs, NewArgs, SetArgs, ShowArgs};
use crate::error::{CliError, CliResult, Exit, open_error};
use crate::output::{Align, Table, cell, fmt_f, fmt_g, key_values, num};

fn path_str(p: &Path) -> String {
    p.display().to_string()
}

// ---------------------------------------------------------------------------
// new

pub fn new(ctx: &Ctx, a: &NewArgs) -> CliResult<Report> {
    let system = match &a.sample {
        None => OpticalSystem::new_empty(),
        Some(name) => {
            let all = samples::all();
            let names: Vec<&str> = all.iter().map(|s| s.0).collect();
            all.iter()
                .find(|s| s.0.eq_ignore_ascii_case(name))
                .map(|s| s.1.clone())
                .ok_or_else(|| {
                    CliError::usage(format!(
                        "unknown sample '{name}' (available: {})",
                        names.join(", ")
                    ))
                })?
        }
    };
    if a.file.exists() && !a.force {
        return Err(CliError::usage(format!(
            "{} already exists (use --force to overwrite)",
            a.file.display()
        )));
    }
    let mut session = ctx.engine.session_from_project(Project::new(system))?;
    save(&mut session, Some(&a.file))?;
    let sys = session.system();
    let json = json!({
        "path": path_str(&a.file),
        "sample": a.sample,
        "name": sys.name,
        "num_surfaces": sys.num_surfaces(),
        "schema_version": session.project().schema_version,
        "snapshot_hash": session.snapshot_hash(),
    });
    let text = format!(
        "created {} ({}, {} surfaces)\n",
        a.file.display(),
        sys.name,
        sys.num_surfaces()
    );
    Ok(Report::ok(json, text))
}

// ---------------------------------------------------------------------------
// validate

/// The engine's `validate` report: `ok`, `errors`, `warnings` and
/// `issues`. It includes the checks that make a project reported valid
/// traceable by `paraxial`, `trace` and the ray-based analyses (aperture
/// value, weights, non-finite numbers, tracer set-up, chief and marginal
/// rays; `mklight_engine::validation`).
pub fn validation(session: &Session) -> CliResult<Value> {
    run_sync(session, "validate")
}

pub fn validate(ctx: &Ctx, path: &Path) -> CliResult<Report> {
    let session = ctx.open(path)?;
    let v = validation(&session)?;
    let ok = v["ok"].as_bool().unwrap_or(false);
    let json = json!({
        "path": path_str(path),
        "name": session.system().name,
        "ok": ok,
        "errors": v["errors"],
        "warnings": v["warnings"],
        "issues": v["issues"],
        "open_warnings": open_warnings(&session),
        "migration": session.migration(),
        "snapshot_hash": session.snapshot_hash(),
        "library_hash": session.library_hash(),
    });
    let mut text = warning_lines(session.open_warnings());
    let issues = v["issues"].as_array().cloned().unwrap_or_default();
    if !issues.is_empty() {
        let mut t = Table::new(&[
            ("severity", Align::Left),
            ("code", Align::Left),
            ("surface", Align::Right),
            ("message", Align::Left),
        ]);
        for i in &issues {
            t.row(vec![
                cell(&i["severity"]),
                cell(&i["code"]),
                cell(&i["surface"]),
                cell(&i["message"]),
            ]);
        }
        text.push_str(&t.render());
    }
    text.push_str(&format!(
        "{}: {} ({} error(s), {} warning(s))\n",
        path.display(),
        if ok { "valid" } else { "INVALID" },
        cell(&v["errors"]),
        cell(&v["warnings"]),
    ));
    let exit = if ok { Exit::Ok } else { Exit::InvalidModel };
    Ok(Report::ok(json, text).with_exit(exit))
}

// ---------------------------------------------------------------------------
// show

pub(crate) fn shape_type(s: &SurfaceShape) -> &'static str {
    match s {
        SurfaceShape::Plane => "plane",
        SurfaceShape::Conic { conic, .. } if *conic == 0.0 => "sphere",
        SurfaceShape::Conic { .. } => "conic",
        SurfaceShape::EvenAsphere { .. } => "even_asphere",
        SurfaceShape::Paraxial { .. } => "paraxial",
        SurfaceShape::CoordinateBreak => "coordinate_break",
    }
}

fn aperture_text(a: &ApertureSpec) -> String {
    match a {
        ApertureSpec::EntrancePupilDiameter { value } => {
            format!("entrance pupil diameter {} mm", fmt_g(*value, 6))
        }
        ApertureSpec::ImageSpaceFNumber { value } => {
            format!("image-space F/# {}", fmt_g(*value, 6))
        }
        ApertureSpec::ObjectSpaceNa { value } => format!("object-space NA {}", fmt_g(*value, 6)),
        ApertureSpec::ParaxialWorkingFNumber { value } => {
            format!("paraxial working F/# {}", fmt_g(*value, 6))
        }
        ApertureSpec::FloatByStopSize => "float by stop size".into(),
    }
}

/// Lens data plus system settings. Semi-diameters of `Auto` surfaces come
/// from the reference tracer (`SequentialTracer::semi_diameters`); when the
/// system cannot be resolved or traced they are reported as `null` with a
/// warning.
pub fn show(ctx: &Ctx, a: &ShowArgs) -> CliResult<Report> {
    let session = ctx.open(&a.project)?;
    let sys = session.system().clone();
    let mut warnings = open_warnings(&session);

    // Resolved indices (system_summary) and traced semi-diameters, if the
    // system resolves.
    let summary = match run_sync(&session, "system_summary") {
        Ok(v) => Some(v),
        Err(e) => {
            warnings.push(format!("system does not resolve: {}", e.message));
            None
        }
    };
    let semi = if a.no_semi_diameters || summary.is_none() {
        None
    } else {
        match session
            .resolved()
            .map_err(CliError::from)
            .and_then(|r| SequentialTracer::new(r).map_err(CliError::from))
        {
            Ok(t) => {
                for w in t.warnings() {
                    if !warnings.contains(w) {
                        warnings.push(w.clone());
                    }
                }
                Some(t.semi_diameters(&SemiDiameterSettings::default()))
            }
            Err(e) => {
                warnings.push(format!("semi-diameters not traced: {}", e.message));
                None
            }
        }
    };

    let mut rows = Vec::new();
    let mut t = Table::new(&[
        ("#", Align::Right),
        ("label", Align::Left),
        ("type", Align::Left),
        ("radius", Align::Right),
        ("thickness", Align::Right),
        ("material", Align::Left),
        ("nd", Align::Right),
        ("semi-dia", Align::Right),
        ("sd", Align::Left),
        ("conic", Align::Right),
        ("notes", Align::Left),
    ]);
    for (i, s) in sys.surfaces.iter().enumerate() {
        let c = s.shape.curvature();
        let radius = if c == 0.0 { f64::INFINITY } else { 1.0 / c };
        let (sd_value, sd_mode) = match (&s.aperture, s.semi_diameter, &semi) {
            (Some(ap), _, _) => (Some(ap.bounding_radius()), "aperture"),
            (None, SemiDiameter::Fixed { value }, _) => (Some(value), "fixed"),
            (None, SemiDiameter::Auto, Some(sd)) => match sd.sources[i] {
                SemiDiameterSource::Stop => (Some(sd.values[i]), "stop"),
                SemiDiameterSource::None => (None, "none"),
                _ => (Some(sd.values[i]), "auto"),
            },
            (None, SemiDiameter::Auto, None) => (None, "auto"),
        };
        let nd = summary
            .as_ref()
            .map_or(Value::Null, |v| v["surfaces"][i]["nd_after"].clone());
        let mut notes = Vec::new();
        if i == sys.stop {
            notes.push("STOP".to_string());
        }
        match s.interaction {
            Interaction::Reflect => notes.push("mirror".into()),
            Interaction::Dummy => notes.push("dummy".into()),
            Interaction::Refract => {}
        }
        if let SurfaceShape::Paraxial { focal_length } = s.shape {
            notes.push(format!("f={}", fmt_g(focal_length, 6)));
        }
        if let SurfaceShape::EvenAsphere { coefficients, .. } = &s.shape {
            notes.push(format!("{} asphere coeff.", coefficients.len()));
        }
        if s.placement.is_some_and(|p| !p.is_identity()) {
            notes.push("decenter/tilt".into());
        }
        if let Some(c) = &s.coating {
            notes.push(format!("coating {c}"));
        }
        let label = if i == 0 {
            "OBJ".to_string()
        } else if i == sys.image_index() {
            "IMA".to_string()
        } else {
            s.label.clone()
        };
        t.row(vec![
            i.to_string(),
            label,
            shape_type(&s.shape).into(),
            fmt_f(radius, 4),
            fmt_f(s.thickness, 4),
            s.material.display_name(),
            if nd.is_null() || nd == json!(1.0) {
                String::new()
            } else {
                cell(&nd)
            },
            sd_value.map_or("-".into(), |v| fmt_f(v, 4)),
            sd_mode.into(),
            fmt_g(s.shape.conic(), 6),
            notes.join(", "),
        ]);
        rows.push(json!({
            "index": i,
            "id": s.id,
            "label": s.label,
            "type": shape_type(&s.shape),
            "is_stop": i == sys.stop,
            "curvature": c,
            "radius": num(radius),
            "conic": s.shape.conic(),
            "shape": s.shape,
            "thickness": num(s.thickness),
            "material": s.material.display_name(),
            "material_ref": s.material,
            "nd": nd,
            "interaction": s.interaction,
            "semi_diameter": sd_value.map_or(Value::Null, num),
            "semi_diameter_mode": sd_mode,
            "coating": s.coating,
            "placement": s.placement,
        }));
    }

    let wl = &sys.wavelengths;
    let fields_json = &sys.fields;
    let settings = json!({
        "aperture": sys.aperture,
        "stop": sys.stop,
        "object_at_infinity": sys.object_at_infinity(),
        "fields": fields_json,
        "wavelengths": wl,
        "ray_aiming": sys.ray_aiming,
        "environment": sys.environment,
        "num_configurations": sys.configurations.len(),
    });
    let json = json!({
        "path": path_str(&a.project),
        "name": sys.name,
        "notes": sys.notes,
        "system": settings,
        "surfaces": rows,
        "catalogs": session.catalogs(),
        "snapshot_hash": session.snapshot_hash(),
        "library_hash": session.library_hash(),
        "warnings": warnings,
    });

    let field_kind = serde_json::to_value(sys.fields.kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let field_list: Vec<String> = sys
        .fields
        .fields
        .iter()
        .map(|f| {
            if f.x == 0.0 {
                fmt_g(f.y, 6)
            } else {
                format!("({}, {})", fmt_g(f.x, 6), fmt_g(f.y, 6))
            }
        })
        .collect();
    let wl_list: Vec<String> = wl
        .wavelengths
        .iter()
        .enumerate()
        .map(|(k, w)| {
            let star = if k == wl.primary { "*" } else { "" };
            format!("{}{star}", fmt_f(w.um, 7))
        })
        .collect();
    let aiming = match sys.ray_aiming {
        mklight_core::RayAiming::Off => "off (paraxial)".to_string(),
        mklight_core::RayAiming::Real { tolerance } => format!("real (tol {tolerance:e})"),
    };
    let mut text = key_values(&[
        ("name", sys.name.clone()),
        ("aperture", aperture_text(&sys.aperture)),
        ("stop", format!("surface {}", sys.stop)),
        ("fields", format!("{field_kind}: {}", field_list.join(", "))),
        (
            "wavelengths",
            format!("{} µm (* primary)", wl_list.join(", ")),
        ),
        ("ray aiming", aiming),
        (
            "environment",
            format!(
                "{} °C, {} atm",
                fmt_g(sys.environment.temperature_c, 6),
                fmt_g(sys.environment.pressure_atm, 6)
            ),
        ),
    ]);
    text.push('\n');
    text.push_str(&t.render());
    text.push_str(&warning_lines(&warnings));
    Ok(Report::ok(json, text))
}

// ---------------------------------------------------------------------------
// migrate

/// Migrate in place with the engine (`Engine::migrate_file`: the backup is
/// `<file>.v<from>.bak`, or `….bak.<k>` for the first free `k` when taken;
/// the file is opened through a session before anything is written).
pub fn migrate(ctx: &Ctx, a: &MigrateArgs) -> CliResult<Report> {
    let path = &a.project;
    let out = ctx
        .engine
        .migrate_file(path, !a.no_backup, a.dry_run)
        .map_err(|e| open_error(path, e))?;
    let json = serde_json::to_value(&out)
        .map_err(|e| CliError::new(Exit::AnalysisFailed, e.to_string()))?;
    if out.from_version == out.to_version {
        let text = format!(
            "{} is already at schema version {}; nothing to do\n",
            path.display(),
            out.to_version
        );
        return Ok(Report::ok(json, text));
    }
    let mut text: String = out.notes.iter().map(|n| format!("note: {n}\n")).collect();
    if !out.migrated {
        text.push_str(&format!(
            "{} would be migrated from schema version {} to {} (dry run, nothing written)\n",
            path.display(),
            out.from_version,
            out.to_version
        ));
        return Ok(Report::ok(json, text));
    }
    text.push_str(&format!(
        "migrated {} from schema version {} to {}\n",
        path.display(),
        out.from_version,
        out.to_version
    ));
    if let Some(b) = &out.backup {
        text.push_str(&format!("backup: {b}\n"));
    }
    Ok(Report::ok(json, text))
}

// ---------------------------------------------------------------------------
// set

/// A parameter value: a number, or `inf` / `-inf`.
pub fn parse_value(s: &str) -> CliResult<f64> {
    let t = s.trim();
    match t {
        "inf" | "+inf" | "Infinity" => return Ok(f64::INFINITY),
        "-inf" | "-Infinity" => return Ok(f64::NEG_INFINITY),
        _ => {}
    }
    t.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| CliError::usage(format!("invalid value '{s}' (a number, inf or -inf)")))
}

/// Issues of [`validation`] with a given severity, as comparable keys
/// (`code[@surface]: message`). A `paraxial` failure is keyed by its code
/// alone: its message changes with the system, and an edit to a system that
/// already fails there must not count as a new error.
fn issue_keys(v: &Value, severity: &str) -> BTreeSet<String> {
    v["issues"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["severity"] == severity)
        .map(|i| {
            if i["code"] == "paraxial" {
                return "paraxial".to_string();
            }
            let at = i["surface"]
                .as_u64()
                .map_or(String::new(), |s| format!("@{s}"));
            format!("{}{at}: {}", cell(&i["code"]), cell(&i["message"]))
        })
        .collect()
}

/// The reason of a parameter error without the error kind's prefix.
fn param_reason(e: &MkError) -> String {
    match e {
        MkError::InvalidParam(m) | MkError::InvalidModel(m) => m.clone(),
        other => other.to_string(),
    }
}

/// Apply one `ParamPath` edit. The edit is applied through the session
/// (structural validation, new snapshot) and checked with [`validation`]
/// (the engine's `validate` analysis); it is rejected
/// (exit 2, nothing written) if it introduces an error that the project did
/// not already have, for example a system the tracer can no longer set up.
/// A path that does not address a parameter of this system is a usage error
/// (1); a value the parameter cannot take is an invalid model (2). New
/// warnings are reported but do not block the edit. Saving is atomic.
pub fn set(ctx: &Ctx, a: &SetArgs) -> CliResult<Report> {
    let param: ParamPath = serde_json::from_str(&a.path)
        .map_err(|e| CliError::usage(format!("invalid parameter path '{}': {e}", a.path)))?;
    let value = parse_value(&a.value)?;
    let mut session = ctx.open(&a.project)?;
    // No current value for some paths (an automatic semi-diameter).
    let old = param.get(session.system()).ok();
    // Does the path address a parameter of this system? Probe it on a copy
    // with a value it accepts (its own, or 1 for an automatic
    // semi-diameter), so a rejected value is not mistaken for a bad path.
    param
        .set(&mut session.system().clone(), old.unwrap_or(1.0))
        .map_err(|e| {
            CliError::usage(format!(
                "invalid parameter path '{}' for this system: {}",
                a.path,
                param_reason(&e)
            ))
        })?;
    let before_hash = session.snapshot_hash();
    let before = validation(&session)?;
    session
        .apply_edits(&[ParamEdit {
            path: param.clone(),
            value,
        }])
        .map_err(|e| {
            let why = match &e {
                EngineError::Core(c) => param_reason(c),
                other => other.to_string(),
            };
            CliError::model(format!(
                "edit rejected, invalid value {} for {}: {why}",
                a.value.trim(),
                a.path
            ))
        })?;
    let after = validation(&session)?;
    let new_errors: Vec<String> = issue_keys(&after, "error")
        .difference(&issue_keys(&before, "error"))
        .cloned()
        .collect();
    if !new_errors.is_empty() {
        return Err(CliError::model(format!(
            "edit rejected, it makes the system invalid: {}",
            new_errors.join("; ")
        )));
    }
    let new_warnings: Vec<String> = issue_keys(&after, "warning")
        .difference(&issue_keys(&before, "warning"))
        .cloned()
        .collect();
    let target = a.out.clone().unwrap_or_else(|| a.project.clone());
    let changed = session.snapshot_hash() != before_hash;
    let saved = !a.dry_run;
    if saved {
        save(&mut session, Some(&target))?;
    }
    let json = json!({
        "path": path_str(&a.project),
        "param": param,
        "old": old.map_or(Value::Null, num),
        "new": num(value),
        "changed": changed,
        "saved": saved,
        "saved_to": if saved { json!(path_str(&target)) } else { Value::Null },
        "snapshot_hash_before": before_hash,
        "snapshot_hash": session.snapshot_hash(),
        "new_warnings": new_warnings,
    });
    let mut text = format!(
        "{}: {} -> {}\n",
        a.path,
        old.map_or("(automatic)".into(), |v| fmt_g(v, 10)),
        fmt_g(value, 10)
    );
    text.push_str(&if saved {
        format!("saved {}\n", target.display())
    } else {
        "dry run: nothing written\n".into()
    });
    text.push_str(&warning_lines(&new_warnings));
    Ok(Report::ok(json, text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_parse_with_infinities() {
        assert_eq!(parse_value("12.5").unwrap(), 12.5);
        assert_eq!(parse_value("-3").unwrap(), -3.0);
        assert_eq!(parse_value("inf").unwrap(), f64::INFINITY);
        assert_eq!(parse_value("-inf").unwrap(), f64::NEG_INFINITY);
        assert_eq!(parse_value("x").unwrap_err().exit, Exit::Usage);
        assert_eq!(parse_value("NaN").unwrap_err().exit, Exit::Usage);
    }
}
