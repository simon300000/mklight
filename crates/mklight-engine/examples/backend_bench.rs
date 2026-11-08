//! Backend timing matrix for CPU/GPU work-split studies.
//!
//! ```text
//! cargo run --release -p mklight-engine --example backend_bench -- SUITE.json \
//!     [--filter TEXT] [--backends cpu,metal,auto] [--reps N] [--out FILE.jsonl] \
//!     [--label TEXT] [--no-compare]
//! ```
//!
//! `SUITE.json` names the projects and the cases:
//!
//! ```json
//! {"lenses": {"dg": "/path/double_gauss.mklight"},
//!  "cases": [{"lens": "dg", "analysis": "spot_diagram", "settings": {"rings": 100},
//!             "tag": "dense", "reps": 5, "backends": ["cpu", "metal", "auto"],
//!             "setup": "curvature_variables"}]}
//! ```
//!
//! `settings`, `tag`, `reps` (default 5), `backends` (default all three) and
//! `setup` are optional. `setup: "curvature_variables"` makes every curved
//! surface's curvature a variable (for `optimize`, `tolerance` and the
//! other solvers). `--filter` keeps the cases whose `lens/analysis/tag`
//! contains the text (comma-separated alternatives).
//!
//! Every timed call follows an edit, a tiny unique change of the first
//! curved surface, so nothing is reused from an earlier call; backends
//! alternate within each repetition so drift affects them alike. Before the
//! timed calls, one untimed call per backend on one shared state warms
//! shaders and host state and compares each backend's output with the CPU
//! reference: for every group of numbers (the JSON path without array
//! indices, or a buffer) the largest difference relative to the group's
//! largest CPU magnitude; the worst groups are reported.
//!
//! Each case and backend appends one JSON line to `--out` and prints a row.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use mklight_core::{OpticalSystem, param::ParamPath};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, Session, analysis::AnalysisResult, backend::Backend,
    buffer::BufferData,
};
use serde_json::{Value, json};

struct Args {
    suite: String,
    filter: Option<Vec<String>>,
    backends: Option<Vec<String>>,
    reps: Option<usize>,
    out: Option<String>,
    label: String,
    compare: bool,
}

fn args() -> Args {
    let mut a = Args {
        suite: String::new(),
        filter: None,
        backends: None,
        reps: None,
        out: None,
        label: String::new(),
        compare: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{x} needs a value"));
        match x.as_str() {
            "--filter" => a.filter = Some(value().split(',').map(String::from).collect()),
            "--backends" => a.backends = Some(value().split(',').map(String::from).collect()),
            "--reps" => a.reps = Some(value().parse().expect("--reps N")),
            "--out" => a.out = Some(value()),
            "--label" => a.label = value(),
            "--no-compare" => a.compare = false,
            _ if a.suite.is_empty() => a.suite = x,
            _ => panic!("unexpected argument {x}"),
        }
    }
    assert!(
        !a.suite.is_empty(),
        "usage: backend_bench SUITE.json [options]"
    );
    a
}

fn load() -> f64 {
    let mut l = [0f64; 3];
    // SAFETY: getloadavg writes at most the 3 doubles it is given.
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    l[0]
}

fn commit() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let head = git(&["rev-parse", "--short", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    format!("{head}{}", if dirty { "+dirty" } else { "" })
}

fn backend(name: &str) -> Backend {
    match name {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        gpu => Backend::Gpu(gpu.into()),
    }
}

/// A unique, tiny change of the first curved surface (relative 1e-9 scale).
fn edit(s: &mut Session, base: &OpticalSystem, serial: u64) {
    let mut system = base.clone();
    let surface = system
        .surfaces
        .iter()
        .skip(1)
        .find(|s| s.shape.curvature() != 0.)
        .expect("a curved surface");
    let param = ParamPath::Curvature {
        surface: surface.id,
    };
    let c = surface.shape.curvature();
    let sign = if serial % 2 == 0 { 1. } else { -1. };
    param
        .set(
            &mut system,
            c * (1. + sign * (1. + serial as f64 * 1e-3) * 1e-9),
        )
        .unwrap();
    s.set_system(system).unwrap();
}

fn run(
    s: &Session,
    analysis: &str,
    settings: &Value,
    be: &str,
) -> Result<(f64, AnalysisResult), String> {
    let mut req = AnalysisRequest::new(analysis, settings.clone());
    req.backend = Some(backend(be));
    let engine = s.engine().clone();
    let t = Instant::now();
    let id = s.submit_analysis(&req).map_err(|e| e.to_string())?;
    let r = engine
        .wait_job(id, None)
        .and_then(|_| engine.take_result(id))
        .map_err(|e| e.to_string())?;
    Ok((t.elapsed().as_secs_f64() * 1e3, r))
}

fn skipped(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    [
        "time",
        "_ms",
        "seconds",
        "duration",
        "elapsed",
        "iteration",
        "evaluation",
        "stats",
        "seed",
    ]
    .iter()
    .any(|p| k.contains(p))
}

fn leaves(v: &Value, path: &str, out: &mut BTreeMap<String, Vec<f64>>) {
    match v {
        Value::Number(n) => out
            .entry(path.to_string())
            .or_default()
            .push(n.as_f64().unwrap()),
        Value::Bool(b) => out
            .entry(path.to_string())
            .or_default()
            .push(*b as u8 as f64),
        Value::Array(a) => a.iter().for_each(|x| leaves(x, &format!("{path}[]"), out)),
        Value::Object(m) => m
            .iter()
            .filter(|(k, _)| !skipped(k))
            .for_each(|(k, x)| leaves(x, &format!("{path}.{k}"), out)),
        _ => {}
    }
}

fn groups(r: &AnalysisResult) -> BTreeMap<String, Vec<f64>> {
    let mut out = BTreeMap::new();
    leaves(&r.json, "", &mut out);
    for b in &r.buffers {
        let v: Vec<f64> = match &b.data {
            BufferData::F64(x) => x.clone(),
            BufferData::F32(x) => x.iter().map(|&y| y as f64).collect(),
            BufferData::I32(x) => x.iter().map(|&y| y as f64).collect(),
            BufferData::U32(x) => x.iter().map(|&y| y as f64).collect(),
            BufferData::I64(x) => x.iter().map(|&y| y as f64).collect(),
            BufferData::U64(x) => x.iter().map(|&y| y as f64).collect(),
            #[allow(unreachable_patterns)]
            _ => continue,
        };
        out.insert(format!("buffer:{}", b.name), v);
    }
    out
}

/// The worst groups of `r` against the reference, as (group, relative difference).
fn compare(reference: &AnalysisResult, r: &AnalysisResult) -> Value {
    let a = groups(reference);
    let b = groups(r);
    let mut rows: Vec<(String, f64)> = Vec::new();
    let mut mismatched = Vec::new();
    for (k, x) in &a {
        let Some(y) = b.get(k) else {
            mismatched.push(format!("{k}: missing"));
            continue;
        };
        if x.len() != y.len() {
            mismatched.push(format!("{k}: {} vs {} values", x.len(), y.len()));
            continue;
        }
        let scale = x
            .iter()
            .fold(0f64, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
        let diff = x.iter().zip(y).fold(0f64, |m, (p, q)| {
            if p.is_nan() && q.is_nan() || p == q {
                m
            } else if p.is_finite() && q.is_finite() {
                m.max((p - q).abs())
            } else {
                f64::INFINITY
            }
        });
        let rel = if diff == 0. {
            0.
        } else {
            diff / scale.max(1e-300)
        };
        rows.push((k.clone(), rel));
    }
    rows.sort_by(|p, q| q.1.total_cmp(&p.1));
    rows.truncate(3);
    json!({
        "worst": rows.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        "mismatched": mismatched.into_iter().take(5).collect::<Vec<_>>(),
    })
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        f64::NAN
    } else if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    }
}

fn curvature_variables(s: &mut Session) {
    let vars: Vec<Value> = s.system().surfaces[1..s.system().surfaces.len() - 1]
        .iter()
        .filter(|x| x.shape.curvature() != 0.)
        .map(|x| json!({"param": {"param": "curvature", "surface": x.id}, "lower": null, "upper": null}))
        .collect();
    s.set_project_section("variables", Value::Array(vars))
        .unwrap();
}

fn main() {
    let a = args();
    let suite: Value = serde_json::from_str(&std::fs::read_to_string(&a.suite).unwrap()).unwrap();
    let engine = Engine::new(EngineConfig::default());
    let commit = commit();
    let mut sessions: BTreeMap<String, (Session, OpticalSystem)> = BTreeMap::new();
    let mut out = a.out.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap()
    });
    let mut serial = (std::process::id() as u64) << 20;
    println!(
        "# {} commit {commit}, load {:.1}; median ms after an edit",
        a.label,
        load()
    );
    println!(
        "{:<44} {:>9} {:>9} {:>9}  {:<6} {}",
        "case", "cpu", "metal", "auto", "auto→", "metal vs cpu (worst group)"
    );
    for case in suite["cases"].as_array().unwrap() {
        let lens = case["lens"].as_str().unwrap();
        let analysis = case["analysis"].as_str().unwrap();
        let tag = case["tag"].as_str().unwrap_or("");
        let settings = case.get("settings").cloned().unwrap_or(json!({}));
        let key = format!("{lens}/{analysis}/{tag}");
        if let Some(f) = &a.filter
            && !f.iter().any(|f| key.contains(f.as_str()))
        {
            continue;
        }
        let reps = a
            .reps
            .unwrap_or(case["reps"].as_u64().unwrap_or(5) as usize)
            .max(1);
        let backends: Vec<String> = a.backends.clone().unwrap_or_else(|| {
            case["backends"].as_array().map_or_else(
                || vec!["cpu".into(), "metal".into(), "auto".into()],
                |b| b.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            )
        });
        if !sessions.contains_key(lens) {
            let path = suite["lenses"][lens]
                .as_str()
                .unwrap_or_else(|| panic!("no lens {lens}"));
            let s = engine.open_session(Path::new(path)).unwrap();
            let base = s.system().clone();
            sessions.insert(lens.to_string(), (s, base));
        }
        let (s, base) = sessions.get_mut(lens).unwrap();
        if case["setup"].as_str() == Some("curvature_variables") {
            curvature_variables(s);
        }
        // Warm-up on one shared state; outputs compared with the CPU reference.
        serial += 1;
        edit(s, base, serial);
        let mut warm: BTreeMap<String, Result<(f64, AnalysisResult), String>> = BTreeMap::new();
        for be in &backends {
            warm.insert(be.clone(), run(s, analysis, &settings, be));
        }
        if a.compare && !warm.contains_key("cpu") && warm.values().any(|r| r.is_ok()) {
            warm.insert("cpu".into(), run(s, analysis, &settings, "cpu"));
        }
        let load_before = load();
        let mut times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut device: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut reported: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut used: BTreeMap<String, String> = BTreeMap::new();
        let mut errors: BTreeMap<String, String> = BTreeMap::new();
        for _ in 0..reps {
            for be in &backends {
                if matches!(warm.get(be), Some(Err(_))) {
                    continue;
                }
                serial += 1;
                edit(s, base, serial);
                let d0 = mklight_metal::device_seconds();
                match run(s, analysis, &settings, be) {
                    Ok((ms, r)) => {
                        times.entry(be.clone()).or_default().push(ms);
                        device
                            .entry(be.clone())
                            .or_default()
                            .push((mklight_metal::device_seconds() - d0) * 1e3);
                        reported
                            .entry(be.clone())
                            .or_default()
                            .push(r.run_info.duration_ms);
                        used.insert(be.clone(), r.run_info.backend.clone());
                    }
                    Err(e) => {
                        errors.insert(be.clone(), e);
                    }
                }
            }
        }
        let load_after = load();
        let mut cells = Vec::new();
        for be in ["cpu", "metal", "auto"] {
            cells.push(match times.get(be) {
                Some(t) => format!("{:9.1}", median(t)),
                None if backends.iter().any(|b| b == be) => format!("{:>9}", "err"),
                None => format!("{:>9}", "-"),
            });
        }
        let cpu_ref = warm.get("cpu").and_then(|r| r.as_ref().ok());
        let mut agreement = String::new();
        for be in &backends {
            let comparison = match (cpu_ref, warm.get(be)) {
                (Some((_, c)), Some(Ok((_, r)))) if be != "cpu" && a.compare => Some(compare(c, r)),
                _ => None,
            };
            if be == "metal"
                && let Some(c) = &comparison
            {
                agreement = c["worst"]
                    .get(0)
                    .map(|w| format!("{:.1e} {}", w[1].as_f64().unwrap(), w[0].as_str().unwrap()))
                    .unwrap_or_default();
            }
            let warm_err = warm.get(be).and_then(|r| r.as_ref().err()).cloned();
            let record = json!({
                "label": a.label, "commit": commit, "lens": lens, "analysis": analysis, "tag": tag,
                "settings": settings, "backend": be, "used": used.get(be).or_else(|| {
                    warm.get(be).and_then(|r| r.as_ref().ok()).map(|(_, r)| &r.run_info.backend)
                }),
                "ms": times.get(be), "median_ms": times.get(be).map(|t| median(t)),
                "min_ms": times.get(be).map(|t| t.iter().cloned().fold(f64::INFINITY, f64::min)),
                "device_ms": device.get(be).map(|t| median(t)),
                "reported_ms": reported.get(be).map(|t| median(t)),
                "first_ms": warm.get(be).and_then(|r| r.as_ref().ok()).map(|(ms, _)| *ms),
                "load": [load_before, load_after],
                "error": errors.get(be).cloned().or(warm_err),
                "compare": comparison,
            });
            if let Some(f) = out.as_mut() {
                use std::io::Write;
                writeln!(f, "{record}").unwrap();
            }
        }
        let auto_used = used
            .get("auto")
            .map(|u| if u.starts_with("gpu") { "Metal" } else { "CPU" })
            .unwrap_or("");
        let note = match used.get("metal") {
            Some(u) if !u.starts_with("gpu") => format!("  (metal ran on {u})"),
            _ => String::new(),
        };
        let errs =
            if errors.is_empty() && !warm.values().any(|r| r.is_err()) {
                String::new()
            } else {
                let e: Vec<String> =
                    warm.iter()
                        .filter_map(|(b, r)| {
                            r.as_ref()
                                .err()
                                .map(|e| format!("{b}: {}", e.chars().take(80).collect::<String>()))
                        })
                        .chain(errors.iter().map(|(b, e)| {
                            format!("{b}: {}", e.chars().take(80).collect::<String>())
                        }))
                        .collect();
                format!("  ERR {}", e.join("; "))
            };
        println!(
            "{:<44} {} {} {}  {:<6} {agreement}{note}{errs}",
            format!("{lens} {analysis} {tag}")
                .chars()
                .take(44)
                .collect::<String>(),
            cells[0],
            cells[1],
            cells[2],
            auto_used
        );
    }
    println!("# load at end {:.1}", load());
}
