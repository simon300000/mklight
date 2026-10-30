//! `mklight info`.

use serde_json::json;

use mklight_engine::PROJECT_SCHEMA_VERSION;
use mklight_ffi::{MK_ABI_VERSION, MK_ABI_VERSION_MAJOR, MK_ABI_VERSION_MINOR};

use super::{Ctx, Report};
use crate::error::CliResult;
use crate::output::{Align, Table, key_values};

pub fn info(ctx: &Ctx) -> CliResult<Report> {
    let backends = ctx.engine.backends();
    let default = ctx.engine.default_backend().id();
    let analyses = ctx.engine.analyses();
    let available = analyses.iter().filter(|a| a.available).count();
    let json = json!({
        "engine_version": mklight_core::ENGINE_VERSION,
        "cli_version": env!("CARGO_PKG_VERSION"),
        "abi_version": {
            "major": MK_ABI_VERSION_MAJOR,
            "minor": MK_ABI_VERSION_MINOR,
            "packed": MK_ABI_VERSION,
        },
        "project_schema_version": PROJECT_SCHEMA_VERSION,
        "platform": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        "default_backend": default,
        "backends": backends,
        "analyses": {"registered": analyses.len(), "available": available},
    });

    let mut text = key_values(&[
        ("engine version", mklight_core::ENGINE_VERSION.to_string()),
        (
            "ABI version",
            format!("{MK_ABI_VERSION_MAJOR}.{MK_ABI_VERSION_MINOR}"),
        ),
        ("project schema", PROJECT_SCHEMA_VERSION.to_string()),
        (
            "platform",
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        ),
        ("default backend", default),
        (
            "analyses",
            format!("{available} available of {} registered", analyses.len()),
        ),
    ]);
    text.push('\n');
    let mut t = Table::new(&[
        ("backend", Align::Left),
        ("available", Align::Left),
        ("precision", Align::Left),
        ("device", Align::Left),
        ("note", Align::Left),
    ]);
    for b in &backends {
        let precision = serde_json::to_value(b.precision)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        t.row(vec![
            b.id.clone(),
            if b.available { "yes" } else { "no" }.into(),
            precision,
            b.capabilities
                .device_name
                .clone()
                .unwrap_or_else(|| "-".into()),
            b.reason.clone().unwrap_or_else(|| b.title.clone()),
        ]);
    }
    text.push_str(&t.render());
    Ok(Report::ok(json, text))
}
