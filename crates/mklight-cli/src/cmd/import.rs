//! `import`: a `.zmx` or `.seq` lens prescription file into a new project.

use serde_json::json;

use mklight_core::MkError;
use mklight_core::import::PrescriptionFormat;
use mklight_engine::EngineError;

use super::{Ctx, Report, save, warning_lines};
use crate::cli::ImportArgs;
use crate::error::{CliError, CliResult, Exit};

pub fn import(ctx: &Ctx, a: &ImportArgs) -> CliResult<Report> {
    if !a.file.is_file() {
        return Err(CliError::io(format!("{}: no such file", a.file.display())));
    }
    if PrescriptionFormat::from_path(&a.file).is_none() {
        return Err(CliError::usage(format!(
            "{}: not a lens prescription file (expected .zmx or .seq)",
            a.file.display()
        )));
    }
    if a.project.exists() && !a.force {
        return Err(CliError::usage(format!(
            "{} already exists (use --force to overwrite)",
            a.project.display()
        )));
    }
    let mut session = ctx.engine.new_session();
    // Catalogs first, so the prescription's glasses resolve against them;
    // they are recorded in the project.
    for c in &a.catalog {
        session
            .load_catalog_file(c)
            .map_err(|e| CliError::from(e).context(c.display()))?;
    }
    // A file that is not a prescription is invalid input (2), as for projects.
    let imported = session.import_prescription(&a.file).map_err(|e| match e {
        EngineError::Core(MkError::Parse(m)) => {
            CliError::new(Exit::InvalidModel, format!("{}: {m}", a.file.display()))
        }
        e => CliError::from(e).context(a.file.display()),
    })?;
    save(&mut session, Some(&a.project))?;
    let sys = session.system();
    let lines: Vec<String> = imported.warnings.iter().map(|w| w.to_string()).collect();
    let mut text = format!(
        "imported {} into {} ({}, {} surfaces, {} warning{})\n",
        a.file.display(),
        a.project.display(),
        sys.name,
        sys.num_surfaces(),
        lines.len(),
        if lines.len() == 1 { "" } else { "s" }
    );
    text.push_str(&warning_lines(&lines));
    let json = json!({
        "file": a.file.display().to_string(),
        "project": a.project.display().to_string(),
        "format": imported.format,
        "encoding": imported.encoding,
        "name": sys.name,
        "num_surfaces": sys.num_surfaces(),
        "warnings": imported.warnings,
        "snapshot_hash": session.snapshot_hash(),
    });
    Ok(Report::ok(json, text))
}
