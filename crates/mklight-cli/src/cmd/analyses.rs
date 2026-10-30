//! `mklight analyses` and `mklight run`: generic over the engine's analysis
//! registry, so analyses registered by solver crates appear here without
//! any change to the CLI.

use serde_json::{Map, Value, json};
use std::time::{Duration, Instant};

use mklight_engine::backend::Backend;
use mklight_engine::buffer::BufferData;
use mklight_engine::fsutil::atomic_write;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineError, JobState, JobStatus};

use super::{Ctx, Report, warning_lines};
use crate::cli::{AnalysesArgs, BackendChoice, RunArgs};
use crate::error::{CliError, CliResult, Exit, exit_of};
use crate::output::{self, Align, Table, cell, fmt_f, key_values, num};
use crate::signal;

// ---------------------------------------------------------------------------
// analyses

/// `"key=default"` for every setting with a default; `key (required)` else.
fn settings_brief(schema: &Value) -> String {
    let required: Vec<&str> = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    schema["properties"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, p)| match p.get("default") {
            Some(d) => format!("{k}={}", compact(d)),
            None if required.contains(&k.as_str()) => format!("{k} (required)"),
            None => k.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn compact(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn list(ctx: &Ctx, a: &AnalysesArgs) -> CliResult<Report> {
    if let Some(name) = &a.name {
        let info = ctx.engine.analysis(name)?.info();
        let json = serde_json::to_value(&info).map_err(|e| CliError::usage(e.to_string()))?;
        let mut text = key_values(&[
            ("name", info.name.clone()),
            ("title", info.title.clone()),
            ("category", info.category.clone()),
            (
                "available",
                match &info.unavailable_reason {
                    None => "yes".into(),
                    Some(r) => format!("no ({r})"),
                },
            ),
        ]);
        if !info.description.is_empty() {
            text.push_str(&format!("\n{}\n", info.description));
        }
        let props = info.settings_schema["properties"].as_object();
        if props.is_some_and(|p| !p.is_empty()) {
            let mut t = Table::new(&[
                ("setting", Align::Left),
                ("type", Align::Left),
                ("default", Align::Left),
                ("range", Align::Left),
                ("unit", Align::Left),
                ("description", Align::Left),
            ]);
            for (k, p) in props.into_iter().flatten() {
                let range = match (p.get("minimum"), p.get("maximum"), p.get("enum")) {
                    (_, _, Some(e)) => e.to_string(),
                    (None, None, _) => String::new(),
                    (lo, hi, _) => format!(
                        "[{}, {}]",
                        lo.map_or("-".into(), compact),
                        hi.map_or("-".into(), compact)
                    ),
                };
                t.row(vec![
                    k.clone(),
                    p.get("type").map_or(String::new(), compact),
                    p.get("default").map_or("-".into(), compact),
                    range,
                    p.get("unit").map_or(String::new(), compact),
                    p.get("description")
                        .or(p.get("title"))
                        .map_or(String::new(), compact),
                ]);
            }
            text.push('\n');
            text.push_str(&t.render());
        } else {
            text.push_str("\nno settings\n");
        }
        return Ok(Report::ok(json, text));
    }

    let infos = ctx.engine.analyses();
    let mut t = Table::new(&[
        ("name", Align::Left),
        ("category", Align::Left),
        ("available", Align::Left),
        ("title", Align::Left),
        ("settings", Align::Left),
    ]);
    for i in &infos {
        t.row(vec![
            i.name.clone(),
            i.category.clone(),
            if i.available { "yes" } else { "no" }.into(),
            i.title.clone(),
            settings_brief(&i.settings_schema),
        ]);
    }
    let json = json!({ "analyses": infos });
    Ok(Report::ok(json, t.render()))
}

// ---------------------------------------------------------------------------
// run

/// Parse `--settings`: JSON text, or `@path` for a file.
pub fn parse_settings(arg: Option<&str>) -> CliResult<Value> {
    let Some(s) = arg else {
        return Ok(Value::Null);
    };
    let (text, origin) = match s.strip_prefix('@') {
        Some(path) => (
            std::fs::read_to_string(path)
                .map_err(|e| CliError::io(format!("cannot read settings file {path}: {e}")))?,
            path.to_string(),
        ),
        None => (s.to_string(), "--settings".to_string()),
    };
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| CliError::usage(format!("{origin}: invalid JSON: {e}")))?;
    if !(v.is_object() || v.is_null()) {
        return Err(CliError::usage(format!(
            "{origin}: settings must be a JSON object"
        )));
    }
    Ok(v)
}

/// Resolve the `--backend` choice. `auto` is the engine's `auto`: the
/// first available GPU backend that implements the analysis and qualifies
/// for the system, else the CPU reference (the envelope's `backend` says
/// which, and why); `metal` is an explicit request (an error if unavailable
/// or not implemented for the analysis).
pub fn choose_backend(_engine: &Engine, _analysis: &str, choice: BackendChoice) -> Backend {
    match choice {
        BackendChoice::Cpu => Backend::CpuReference,
        BackendChoice::Metal => Backend::Gpu("metal".into()),
        BackendChoice::Auto => Backend::Auto,
    }
}

fn buffer_json(data: &BufferData) -> Value {
    match data {
        BufferData::F64(v) => v.iter().map(|x| num(*x)).collect(),
        BufferData::F32(v) => v.iter().map(|x| num(f64::from(*x))).collect(),
        BufferData::I32(v) => json!(v),
        BufferData::U32(v) => json!(v),
        BufferData::I64(v) => json!(v),
        BufferData::U64(v) => json!(v),
        BufferData::U8(v) => json!(v),
    }
}

/// The result envelope (`docs/FFI.md`), with buffer data inlined on request.
fn envelope(r: &AnalysisResult, with_buffers: bool) -> Value {
    let mut env = r.envelope();
    if with_buffers && let Some(descs) = env["buffers"].as_array_mut() {
        for (d, b) in descs.iter_mut().zip(&r.buffers) {
            d["data"] = buffer_json(&b.data);
        }
    }
    env
}

fn progress_line(st: &JobStatus, json_mode: bool) -> String {
    if json_mode {
        let doc =
            json!({"job": st.id, "state": st.state, "progress": st.progress, "stage": st.stage});
        serde_json::to_string(&doc).unwrap_or_default()
    } else {
        let stage = if st.stage.is_empty() {
            String::new()
        } else {
            format!("  {}", st.stage)
        };
        format!("progress {:>5}%{stage}", fmt_f(st.progress * 100.0, 1))
    }
}

/// Why a job was cancelled from the CLI side.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CancelReason {
    Interrupt,
    Timeout,
}

impl CancelReason {
    fn name(self) -> &'static str {
        match self {
            CancelReason::Interrupt => "interrupt",
            CancelReason::Timeout => "timeout",
        }
    }
}

pub fn run(ctx: &Ctx, a: &RunArgs) -> CliResult<Report> {
    let engine = &ctx.engine;
    let analysis = engine.analysis(&a.analysis)?;
    let mut settings = parse_settings(a.settings.as_deref())?;
    if let Some(seed) = a.seed {
        let schema = analysis.settings_schema();
        if schema["properties"].get("seed").is_none() {
            return Err(CliError::usage(format!(
                "analysis '{}' takes no seed",
                a.analysis
            )));
        }
        if settings.is_null() {
            settings = Value::Object(Map::new());
        }
        settings["seed"] = json!(seed);
    }
    if let Some(t) = a.timeout
        && !(t.is_finite() && t >= 0.0)
    {
        return Err(CliError::usage(
            "--timeout must be a non-negative number of seconds",
        ));
    }
    let session = ctx.open(&a.project)?;
    let req = AnalysisRequest {
        analysis: a.analysis.clone(),
        settings,
        view_key: Some(format!("cli/{}", a.analysis)),
        backend: Some(choose_backend(engine, &a.analysis, a.backend)),
        allow_backend_fallback: false,
    };

    signal::install();
    let job = session.submit_analysis(&req)?;
    let started = Instant::now();
    let timeout = a.timeout.map(Duration::from_secs_f64);
    let mut cancelled_by: Option<CancelReason> = None;
    let mut stopping = false;
    let mut last_shown: Option<(i64, String)> = None;
    let json_mode = ctx.global.json;
    loop {
        if cancelled_by.is_none() {
            let interrupts = signal::count();
            if interrupts >= 1 && !stopping {
                match engine.stop_job(job) {
                    Ok(()) => stopping = true,
                    Err(EngineError::Core(mklight_core::MkError::Unsupported(_))) => {
                        cancelled_by = Some(CancelReason::Interrupt);
                        engine.cancel_job(job)?;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            let reason = if interrupts >= 2 {
                Some(CancelReason::Interrupt)
            } else if timeout.is_some_and(|t| started.elapsed() >= t) {
                Some(CancelReason::Timeout)
            } else {
                None
            };
            if let Some(r) = reason {
                cancelled_by = Some(r);
                engine.cancel_job(job)?;
            }
        }
        match engine.wait_job(job, Some(Duration::from_millis(25))) {
            Ok(_) => break,
            Err(EngineError::Timeout(_)) => {}
            Err(e) => return Err(e.into()),
        }
        if a.progress {
            let st = engine.job_status(job)?;
            let key = ((st.progress * 1000.0).round() as i64, st.stage.clone());
            if st.state == JobState::Running && last_shown.as_ref() != Some(&key) {
                eprintln!("{}", progress_line(&st, json_mode));
                last_shown = Some(key);
            }
        }
    }
    let status = engine.job_status(job)?;
    match status.state {
        JobState::Done => {
            let result = engine.take_result(job)?;
            finish_done(ctx, a, &result)
        }
        JobState::Cancelled => {
            let _ = engine.release_job(job);
            let reason = cancelled_by.map_or("cancelled", CancelReason::name);
            let json = json!({
                "cancelled": true,
                "reason": reason,
                "analysis": a.analysis,
                "status": status,
            });
            let stage = if status.stage.is_empty() {
                String::new()
            } else {
                format!(" during '{}'", status.stage)
            };
            let text = format!(
                "mklight: {} cancelled ({reason}) at {}%{stage} after {} s\n",
                a.analysis,
                fmt_f(status.progress * 100.0, 1),
                fmt_f(status.elapsed_ms * 1e-3, 2),
            );
            let mut r = Report::ok(json, text).with_exit(Exit::Cancelled);
            r.text_to_stderr = true;
            Ok(r)
        }
        _ => {
            // Failed: the job's own error decides the exit code; a model
            // problem found while running stays "invalid model".
            let err = engine.take_result(job).err().unwrap_or_else(|| {
                EngineError::JobFailed(status.error.clone().unwrap_or_default())
            });
            let exit = match exit_of(&err) {
                Exit::Usage | Exit::Ok => Exit::AnalysisFailed,
                e => e,
            };
            Err(CliError::new(
                exit,
                format!("analysis '{}' failed: {err}", a.analysis),
            ))
        }
    }
}

fn finish_done(ctx: &Ctx, a: &RunArgs, result: &AnalysisResult) -> CliResult<Report> {
    let mut env = envelope(result, a.buffers);
    if ctx.global.no_timing {
        output::strip_timing(&mut env);
    }
    let ri = &result.run_info;
    let mut kv = vec![
        ("analysis", ri.analysis.clone()),
        ("snapshot", ri.snapshot_hash.clone()),
        ("backend", ri.backend.clone()),
        (
            "precision",
            serde_json::to_value(ri.precision)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
        ),
        ("seed", ri.seed.map_or("-".into(), |s| s.to_string())),
        ("settings", ri.settings.to_string()),
    ];
    if !ctx.global.no_timing {
        kv.push(("duration", format!("{} ms", fmt_f(ri.duration_ms, 3))));
    }
    let mut text = key_values(&kv);
    text.push_str(&warning_lines(&ri.warnings));

    let json = if let Some(out) = &a.out {
        atomic_write(out, output::to_pretty(&env).as_bytes())
            .map_err(|e| CliError::io(format!("cannot write {}: {e}", out.display())))?;
        text.push_str(&format!("result written to {}\n", out.display()));
        json!({"out": out.display().to_string(), "run_info": env["run_info"]})
    } else {
        text.push_str("\nresult:\n");
        text.push_str(&output::to_pretty(&result.json));
        if !result.buffers.is_empty() {
            let mut t = Table::new(&[
                ("buffer", Align::Left),
                ("dtype", Align::Left),
                ("shape", Align::Left),
            ]);
            for d in env["buffers"].as_array().into_iter().flatten() {
                t.row(vec![
                    cell(&d["name"]),
                    cell(&d["dtype"]),
                    d["shape"].to_string(),
                ]);
            }
            text.push('\n');
            text.push_str(&t.render());
        }
        env
    };
    Ok(Report::ok(json, text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_parse_from_text_and_reject_non_objects() {
        assert_eq!(parse_settings(None).unwrap(), Value::Null);
        assert_eq!(
            parse_settings(Some(r#"{"seed": 3}"#)).unwrap(),
            json!({"seed": 3})
        );
        assert_eq!(parse_settings(Some("[1]")).unwrap_err().exit, Exit::Usage);
        assert_eq!(parse_settings(Some("{")).unwrap_err().exit, Exit::Usage);
        assert_eq!(
            parse_settings(Some("@/nonexistent/settings.json"))
                .unwrap_err()
                .exit,
            Exit::Io
        );
    }

    #[test]
    fn auto_backend_is_the_engines_auto() {
        let e = Engine::default();
        assert_eq!(
            choose_backend(&e, "system_summary", BackendChoice::Auto),
            Backend::Auto
        );
        assert_eq!(
            choose_backend(&e, "x", BackendChoice::Cpu),
            Backend::CpuReference
        );
        assert_eq!(
            choose_backend(&e, "x", BackendChoice::Metal),
            Backend::Gpu("metal".into())
        );
    }

    #[test]
    fn settings_brief_lists_defaults_and_required() {
        let s = json!({"type": "object", "properties": {
            "a": {"type": "integer", "default": 3},
            "name": {"type": "string"}}, "required": ["name"]});
        assert_eq!(settings_brief(&s), "a=3 name (required)");
    }
}
