//! Project commands: `new`, `validate`, `show`, `migrate`, `set`.

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use mklight_core::param::ParamPath;
use mklight_core::{ApertureSpec, Interaction, OpticalSystem, SemiDiameter, SurfaceShape, samples};
use mklight_engine::fsutil::atomic_write;
use mklight_engine::{ParamEdit, Project, Session};
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

pub fn validate(ctx: &Ctx, path: &Path) -> CliResult<Report> {
    let session = ctx.open(path)?;
    let v = run_sync(&session, "validate")?;
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

fn shape_type(s: &SurfaceShape) -> &'static str {
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

/// `<file>.v<from>.bak`, or `….bak.<k>` for the first free `k` when taken.
fn backup_path(path: &Path, from: u32) -> PathBuf {
    let base = format!("{}.v{from}.bak", path.display());
    let first = PathBuf::from(&base);
    if !first.exists() {
        return first;
    }
    (1..)
        .map(|k| PathBuf::from(format!("{base}.{k}")))
        .find(|p| !p.exists())
        .expect("unbounded search")
}

pub fn migrate(ctx: &Ctx, a: &MigrateArgs) -> CliResult<Report> {
    let path = &a.project;
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::io(format!("cannot read {}: {e}", path.display())))?;
    let text_in = String::from_utf8(bytes.clone())
        .map_err(|e| CliError::model(format!("{}: not UTF-8 text: {e}", path.display())))?;
    let (_, report) = Project::from_json_str(&text_in).map_err(|e| open_error(path, e))?;
    let mut json = json!({
        "path": path_str(path),
        "from_version": report.from_version,
        "to_version": report.to_version,
        "notes": report.notes,
        "migrated": false,
        "backup": null,
        "dry_run": a.dry_run,
    });
    if !report.migrated() {
        let text = format!(
            "{} is already at schema version {}; nothing to do\n",
            path.display(),
            report.to_version
        );
        return Ok(Report::ok(json, text));
    }
    let mut text: String = report
        .notes
        .iter()
        .map(|n| format!("note: {n}\n"))
        .collect();
    if a.dry_run {
        text.push_str(&format!(
            "{} would be migrated from schema version {} to {} (dry run, nothing written)\n",
            path.display(),
            report.from_version,
            report.to_version
        ));
        return Ok(Report::ok(json, text));
    }
    // Open through the session (catalog references are resolved relative to
    // the project) before writing anything, so a file that cannot be opened
    // is left untouched.
    let mut session = ctx.open(path)?;
    let backup = if a.no_backup {
        None
    } else {
        let b = backup_path(path, report.from_version);
        atomic_write(&b, &bytes)
            .map_err(|e| CliError::io(format!("cannot write backup {}: {e}", b.display())))?;
        Some(b)
    };
    save(&mut session, None)?;
    json["migrated"] = json!(true);
    json["backup"] = json!(backup.as_ref().map(|b| path_str(b)));
    text.push_str(&format!(
        "migrated {} from schema version {} to {}\n",
        path.display(),
        report.from_version,
        report.to_version
    ));
    if let Some(b) = &backup {
        text.push_str(&format!("backup: {}\n", b.display()));
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

/// Error issues of the `validate` analysis, as comparable keys.
fn error_issues(session: &Session) -> CliResult<BTreeSet<String>> {
    let v = run_sync(session, "validate")?;
    Ok(v["issues"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["severity"] == "error")
        .map(|i| format!("{}: {}", cell(&i["code"]), cell(&i["message"])))
        .collect())
}

/// Apply one `ParamPath` edit. The edit is applied through the session
/// (structural validation, new snapshot) and checked with the `validate`
/// analysis; it is rejected (exit 2, nothing written) if it introduces a
/// validation error that the project did not already have. Saving is atomic.
pub fn set(ctx: &Ctx, a: &SetArgs) -> CliResult<Report> {
    let param: ParamPath = serde_json::from_str(&a.path)
        .map_err(|e| CliError::usage(format!("invalid parameter path '{}': {e}", a.path)))?;
    let value = parse_value(&a.value)?;
    let mut session = ctx.open(&a.project)?;
    let old = param.get(session.system()).map_err(CliError::from)?;
    let before_hash = session.snapshot_hash();
    let before = error_issues(&session)?;
    session
        .apply_edits(&[ParamEdit {
            path: param.clone(),
            value,
        }])
        .map_err(|e| CliError::from(e).context("edit rejected"))?;
    let after = error_issues(&session)?;
    let new_errors: Vec<&String> = after.difference(&before).collect();
    if !new_errors.is_empty() {
        let list: Vec<&str> = new_errors.iter().map(|s| s.as_str()).collect();
        return Err(CliError::model(format!(
            "edit rejected, it makes the system invalid: {}",
            list.join("; ")
        )));
    }
    let target = a.out.clone().unwrap_or_else(|| a.project.clone());
    let changed = session.snapshot_hash() != before_hash;
    let saved = !a.dry_run;
    if saved {
        save(&mut session, Some(&target))?;
    }
    let json = json!({
        "path": path_str(&a.project),
        "param": param,
        "old": num(old),
        "new": num(value),
        "changed": changed,
        "saved": saved,
        "saved_to": if saved { json!(path_str(&target)) } else { Value::Null },
        "snapshot_hash_before": before_hash,
        "snapshot_hash": session.snapshot_hash(),
    });
    let mut text = format!("{}: {} -> {}\n", a.path, fmt_g(old, 10), fmt_g(value, 10));
    text.push_str(&if saved {
        format!("saved {}\n", target.display())
    } else {
        "dry run: nothing written\n".into()
    });
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
