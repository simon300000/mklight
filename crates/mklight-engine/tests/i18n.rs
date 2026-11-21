//! The engine's display text and the translation catalogs under
//! `locales/<id>/engine.json` (docs/I18N.md).
//!
//! The test collects every English string the engine hands the apps for
//! display (analysis titles, descriptions and unavailable reasons; settings
//! titles, descriptions and enum values; first-order labels; field map column
//! definitions) and every diagnostic code (CONVENTIONS §13 structural codes,
//! the `validate` analysis' codes). It fails when a catalog lacks one of them
//! or still holds one the engine no longer emits, and prints the missing
//! entries as JSON to paste into the catalog. `scripts/i18n-check.sh` runs it
//! and also writes the full English list, with where each string comes from,
//! to `target/i18n/engine-en.json`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use mklight_core::JobContext;
use mklight_core::samples;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig};
use serde_json::{Map, Value, json};

/// Strings that stay English in every language: symbols and names a
/// translation would only obscure.
const KEEP_ENGLISH: &[&str] = &[];

/// Enum values shown as they are (axes, planes, abbreviations).
const KEEP_ENUM: &[&str] = &["x", "y", "xz", "yz"];

/// Placeholders a diagnostic template may use: fields every diagnostic
/// carries (`surface` on `validate` issues, `path` on structural issues).
const PLACEHOLDERS: &[&str] = &["surface", "path"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// English text → where it comes from (for translators).
type Sources = BTreeMap<String, BTreeSet<String>>;

#[derive(Default)]
struct EngineText {
    strings: Sources,
    enums: Sources,
    diagnostics: Sources,
}

fn add(map: &mut Sources, text: &str, source: String) {
    if !text.trim().is_empty() {
        map.entry(text.to_string()).or_default().insert(source);
    }
}

/// Titles, descriptions and enum values of a settings schema. Defaults and
/// constants are data, not display text.
fn walk_schema(v: &Value, at: &str, out: &mut EngineText) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                match (k.as_str(), x) {
                    ("default" | "const" | "examples" | "unit", _) => {}
                    ("title" | "description", Value::String(s)) => {
                        add(&mut out.strings, s, format!("{at}.{k}"))
                    }
                    ("enum", Value::Array(vals)) => {
                        for s in vals.iter().filter_map(Value::as_str) {
                            add(&mut out.enums, s, at.to_string());
                        }
                    }
                    ("properties", Value::Object(props)) => {
                        for (p, s) in props {
                            walk_schema(s, &format!("{at}.{p}"), out);
                        }
                    }
                    _ => walk_schema(x, at, out),
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| walk_schema(x, at, out)),
        _ => {}
    }
}

/// The first string literal argument of each call to `f` in `src`.
fn first_literal_args(src: &str, f: &str) -> Vec<String> {
    let mut out = Vec::new();
    let pat = format!("{f}(");
    let mut rest = src;
    while let Some(i) = rest.find(&pat) {
        let before = rest[..i].chars().last();
        rest = &rest[i + pat.len()..];
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let arg = rest.trim_start();
        if let Some(a) = arg.strip_prefix('"')
            && let Some(end) = a.find('"')
        {
            out.push(a[..end].to_string());
        }
    }
    out
}

/// A snake-case word (`material`, `chief_ray`).
fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c == '_') && !s.starts_with('_')
}

/// A structural code: always more than one word (`non_finite`), which
/// tells it from a path literal (`surfaces`, `stop`).
fn is_code(s: &str) -> bool {
    is_word(s) && s.contains('_')
}

/// Diagnostic codes, read from the code that emits them: the structural
/// checks of `mklight-core` (`Issues::push`, `positive`, `non_negative`,
/// `check_increasing`, `solve_issue`) and the engine's `validate` analysis
/// (`Issue::new(severity, code, …)` and its `push(severity, code, …)`).
fn diagnostic_codes(out: &mut EngineText) {
    let read = |p: &str| std::fs::read_to_string(root().join(p)).unwrap();
    let core = read("crates/mklight-core/src/model/validate.rs");
    let core = core.split("#[cfg(test)]").next().unwrap();
    let src = "crates/mklight-core/src/model/validate.rs".to_string();
    for f in ["push", "positive", "non_negative"] {
        for c in first_literal_args(core, f)
            .into_iter()
            .filter(|c| is_code(c))
        {
            add(&mut out.diagnostics, &c, src.clone());
        }
    }
    // check_increasing(out, "code", …)
    for part in core.split("check_increasing(").skip(1) {
        let lit = part.split('"').nth(1).unwrap_or("");
        if is_code(lit) {
            add(&mut out.diagnostics, lit, src.clone());
        }
    }
    let solve = core.split("fn solve_issue").nth(1).unwrap();
    let solve = solve.split("\n}\n").next().unwrap();
    for lit in solve.split('"').skip(1).step_by(2).filter(|c| is_code(c)) {
        add(&mut out.diagnostics, lit, src.clone());
    }
    for p in [
        "crates/mklight-engine/src/validation.rs",
        "crates/mklight-engine/src/analysis/builtin.rs",
    ] {
        let s = read(p);
        let s = s.split("#[cfg(test)]").next().unwrap();
        for sev in ["\"error\",", "\"warning\",", "\"info\","] {
            for part in s.split(sev).skip(1) {
                let lit = part.trim_start();
                if let Some(a) = lit.strip_prefix('"')
                    && let Some(end) = a.find('"')
                    && is_word(&a[..end])
                    && !["error", "warning", "info"].contains(&&a[..end])
                {
                    add(&mut out.diagnostics, &a[..end], p.to_string());
                }
            }
        }
    }
}

fn collect() -> EngineText {
    let mut out = EngineText::default();
    let engine = Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: mklight_engine::backend::Backend::CpuReference,
        ..Default::default()
    });
    for a in engine.analyses() {
        add(&mut out.strings, &a.title, format!("{}.title", a.name));
        add(
            &mut out.strings,
            &a.description,
            format!("{}.description", a.name),
        );
        if let Some(r) = &a.unavailable_reason {
            add(
                &mut out.strings,
                r,
                format!("{}.unavailable_reason", a.name),
            );
        }
        walk_schema(
            &a.settings_schema,
            &format!("{}.settings", a.name),
            &mut out,
        );
    }

    // Labels and definitions inside results.
    let session = engine
        .session_from_project(Project::new(samples::double_gauss()))
        .unwrap();
    let run = |name: &str| {
        session
            .run_analysis(&AnalysisRequest::new(name, Value::Null), &JobContext::new())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .json
    };
    for t in run("first_order")["entries"].as_array().unwrap() {
        add(
            &mut out.strings,
            t["label"].as_str().unwrap(),
            format!("first_order.entries.{}.label", t["key"].as_str().unwrap()),
        );
    }
    for name in [
        "pupil_aberration",
        "relative_illumination_map",
        "chief_ray_angle",
    ] {
        for c in run(name)["columns"].as_array().unwrap() {
            add(
                &mut out.strings,
                c["definition"].as_str().unwrap(),
                format!("{name}.columns.{}.definition", c["name"].as_str().unwrap()),
            );
        }
    }
    diagnostic_codes(&mut out);
    out
}

/// Placeholders `{name}` of a template; `None` if braces do not pair.
fn placeholders(t: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut rest = t;
    while let Some(i) = rest.find(['{', '}']) {
        if rest[i..].starts_with('}') {
            return None;
        }
        let end = rest[i..].find('}')? + i;
        out.push(rest[i + 1..end].to_string());
        rest = &rest[end + 1..];
    }
    Some(out)
}

fn sources_json(s: &Sources) -> Value {
    Value::Object(
        s.iter()
            .map(|(k, v)| (k.clone(), json!(v.iter().collect::<Vec<_>>())))
            .collect(),
    )
}

/// Missing entries as a catalog fragment with empty values, the sources
/// beside them under `_sources`.
fn missing_fragment(section: &str, missing: &[&String], sources: &Sources) -> String {
    let entries: Map<String, Value> = missing.iter().map(|k| ((*k).clone(), json!(""))).collect();
    let src: Map<String, Value> = missing
        .iter()
        .map(|k| ((*k).clone(), json!(sources[*k].iter().collect::<Vec<_>>())))
        .collect();
    serde_json::to_string_pretty(&json!({ section: entries, "_sources": src })).unwrap()
}

#[test]
fn catalogs_cover_the_engine_text() {
    let text = collect();

    // The English list for translators (scripts/i18n-check.sh copies it).
    let dump = Path::new(env!("CARGO_TARGET_TMPDIR")).join("engine-en.json");
    let all = json!({
        "strings": sources_json(&text.strings),
        "enums": sources_json(&text.enums),
        "diagnostics": sources_json(&text.diagnostics),
    });
    std::fs::write(&dump, serde_json::to_string_pretty(&all).unwrap() + "\n").unwrap();
    println!(
        "engine text: {} strings, {} enum values, {} diagnostic codes in {}",
        text.strings.len(),
        text.enums.len(),
        text.diagnostics.len(),
        dump.display()
    );
    assert!(text.strings.len() > 200 && text.diagnostics.len() > 40);

    let mut locales: Vec<PathBuf> = std::fs::read_dir(root().join("locales"))
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path().join("engine.json");
            p.exists().then_some(p)
        })
        .collect();
    locales.sort();
    assert!(!locales.is_empty(), "no locales/<id>/engine.json");

    let mut failures = Vec::new();
    for path in locales {
        let id = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy();
        let cat: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for (section, sources, keep) in [
            ("strings", &text.strings, KEEP_ENGLISH),
            ("enums", &text.enums, KEEP_ENUM),
            ("diagnostics", &text.diagnostics, &[][..]),
        ] {
            let empty = Map::new();
            let entries = cat[section].as_object().unwrap_or(&empty);
            let missing: Vec<&String> = sources
                .keys()
                .filter(|k| !entries.contains_key(*k) && !keep.contains(&k.as_str()))
                .collect();
            let stale: Vec<&String> = entries
                .keys()
                .filter(|k| !sources.contains_key(*k) || keep.contains(&k.as_str()))
                .collect();
            let bad: Vec<&String> = entries
                .iter()
                .filter(|(_, v)| v.as_str().is_none_or(|s| s.trim().is_empty()))
                .map(|(k, _)| k)
                .collect();
            if !missing.is_empty() {
                failures.push(format!(
                    "{id}: {} {section} missing; add:\n{}",
                    missing.len(),
                    missing_fragment(section, &missing, sources)
                ));
            }
            if !stale.is_empty() {
                failures.push(format!(
                    "{id}: {section} the engine no longer emits (remove): {stale:#?}"
                ));
            }
            if !bad.is_empty() {
                failures.push(format!("{id}: {section} without a translation: {bad:#?}"));
            }
        }
        if let Some(d) = cat["diagnostics"].as_object() {
            for (code, t) in d {
                let t = t.as_str().unwrap_or("");
                match placeholders(t) {
                    Some(ps) if ps.iter().all(|p| PLACEHOLDERS.contains(&p.as_str())) => {}
                    _ => failures.push(format!(
                        "{id}: diagnostic {code}: placeholders must be one of {PLACEHOLDERS:?}: {t}"
                    )),
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}
