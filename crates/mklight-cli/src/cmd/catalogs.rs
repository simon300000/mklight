//! `mklight catalogs list|show|import`.
//!
//! Catalogs live in the session's material library: the built-in catalog
//! plus the user catalogs a project references. `import` loads a user's own
//! AGF (or GlassCatalog JSON) file through the session and records a
//! reference to it (absolute and project-relative path) in the project; the
//! catalog data itself is never copied into the project, and no vendor
//! catalog ships with MKLight.

use serde_json::json;

use mklight_core::MkError;
use mklight_engine::EngineError;

use super::{Ctx, Report, save, warning_lines};
use crate::cli::{CatalogImportArgs, CatalogScope, CatalogShowArgs};
use crate::error::{CliError, CliResult};
use crate::output::{Align, Table, fmt_f, num};

fn session_for(ctx: &Ctx, project: Option<&std::path::Path>) -> CliResult<mklight_engine::Session> {
    match project {
        Some(p) => ctx.open(p),
        None => Ok(ctx.engine.new_session()),
    }
}

pub fn list(ctx: &Ctx, a: &CatalogScope) -> CliResult<Report> {
    let session = session_for(ctx, a.project.as_deref())?;
    let cats = session.catalogs();
    let mut t = Table::new(&[
        ("name", Align::Left),
        ("source", Align::Left),
        ("glasses", Align::Right),
        ("path", Align::Left),
    ]);
    for c in &cats {
        t.row(vec![
            c.name.clone(),
            c.source.into(),
            c.glass_count.to_string(),
            c.path.clone().unwrap_or_else(|| "-".into()),
        ]);
    }
    let mut text = t.render();
    text.push_str(&warning_lines(session.open_warnings()));
    let json = json!({
        "catalogs": cats,
        "library_hash": session.library_hash(),
        "open_warnings": session.open_warnings(),
    });
    Ok(Report::ok(json, text))
}

pub fn show(ctx: &Ctx, a: &CatalogShowArgs) -> CliResult<Report> {
    let session = session_for(ctx, a.project.as_deref())?;
    let mut glasses = session.glasses(Some(&a.name)).map_err(|e| match e {
        // An unknown catalog is a command-line mistake, not a model error.
        EngineError::Core(MkError::UnknownMaterial(m)) => CliError::usage(m),
        other => other.into(),
    })?;
    glasses.sort_by(|x, y| x.name.cmp(&y.name));
    let catalog = glasses
        .first()
        .map_or_else(|| a.name.clone(), |g| g.catalog.clone());
    let mut t = Table::new(&[
        ("glass", Align::Left),
        ("nd", Align::Right),
        ("vd", Align::Right),
        ("min µm", Align::Right),
        ("max µm", Align::Right),
    ]);
    let mut rows = Vec::new();
    for g in &glasses {
        // Computed from the dispersion; none when the range leaves out the d line.
        let (nd, vd) = (g.nd.unwrap_or(f64::NAN), g.vd.unwrap_or(f64::NAN));
        t.row(vec![
            g.name.clone(),
            fmt_f(nd, 6),
            fmt_f(vd, 2),
            fmt_f(g.min_um, 3),
            fmt_f(g.max_um, 3),
        ]);
        rows.push(json!({
            "name": g.name, "nd": num(nd), "vd": num(vd),
            "min_um": num(g.min_um), "max_um": num(g.max_um),
        }));
    }
    let text = format!(
        "catalog {catalog}: {} glasses\n{}",
        glasses.len(),
        t.render()
    );
    Ok(Report::ok(
        json!({"catalog": catalog, "glasses": rows}),
        text,
    ))
}

pub fn import(ctx: &Ctx, a: &CatalogImportArgs) -> CliResult<Report> {
    if !a.file.is_file() {
        return Err(CliError::io(format!("{}: no such file", a.file.display())));
    }
    let mut session = ctx.open(&a.project)?;
    let summary = session
        .load_catalog_file(&a.file)
        .map_err(|e| CliError::from(e).context(a.file.display()))?;
    // Reader warnings (skipped glasses, index mismatches) for the user.
    let warnings: Vec<String> = summary.warnings.iter().map(|w| w.to_string()).collect();
    save(&mut session, None)?;
    let mut text = format!(
        "imported catalog {} ({} glasses) into {}\n",
        summary.name,
        summary.glass_count,
        a.project.display()
    );
    text.push_str(&warning_lines(&warnings));
    let json = json!({
        "project": a.project.display().to_string(),
        "catalog": summary,
        "warnings": warnings,
        "library_hash": session.library_hash(),
    });
    Ok(Report::ok(json, text))
}
