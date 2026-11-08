//! Backend timing matrix for CPU/GPU work-split studies.
//!
//! ```text
//! cargo run --release -p mklight-engine --example backend_bench -- SUITE.json \
//!     [--filter TEXT] [--backends cpu,metal,auto] [--reps N] [--out FILE.jsonl] \
//!     [--label TEXT] [--no-compare] [--budget SECONDS]
//! ```
//!
//! `SUITE.json` names the projects and the cases:
//!
//! ```json
//! {"lenses": {"dg": "/path/double_gauss.mklight"},
//!  "cases": [{"lens": "dg", "analysis": "spot_diagram", "settings": {"rings": 100},
//!             "tag": "dense", "reps": 5, "backends": ["cpu", "metal", "auto"],
//!             "variants": [{"name": "cpuaim", "set": {"geometric": "cpu-aim"}}],
//!             "setup": "curvature_variables"}]}
//! ```
//!
//! `settings`, `tag`, `reps` (default 5), `backends` (default all three),
//! `variants` and `setup` are optional. Each variant adds a column
//! `metal/<name>` (or `<variant_backend>/<name>` when the case sets
//! `variant_backend`) that runs with the work-split selections of `set`
//! (`mklight_core::split`); the plain `metal` column runs the default
//! designs. `setup: "curvature_variables"` makes every curved surface's
//! curvature a variable (for `optimize`, `tolerance` and the other solvers).
//! `--filter` keeps the cases whose `lens/analysis/tag` contains the text
//! (comma-separated alternatives); `--backends` keeps the columns named, a
//! backend name also keeping its variant columns.
//!
//! Every timed call follows an edit, a tiny unique change of the first
//! curved surface, so nothing is reused from an earlier call; columns
//! alternate within each repetition so drift affects them alike. Before the
//! timed calls, each column gets one untimed call that warms shaders and
//! host state; on its own fresh state the CPU reference runs first, and the
//! column's output is compared with it: for every group of numbers (the
//! JSON path without array indices, or a buffer) the largest difference
//! relative to the group's largest CPU magnitude; the worst groups are
//! reported.
//!
//! `--budget SECONDS` caps each column's timed calls: after the first timed
//! call a column runs only as many repetitions as fit the budget (at least
//! one, at most `reps`), so very slow columns are timed once.
//!
//! Each case and column appends one JSON line to `--out` and each case
//! prints one row.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use mklight_core::{OpticalSystem, param::ParamPath, split};
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
    budget: Option<f64>,
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
        budget: None,
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
            "--budget" => a.budget = Some(value().parse().expect("--budget SECONDS")),
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

/// One column of a case: a backend, optionally with work-split selections.
#[derive(Clone)]
struct Column {
    label: String,
    backend: String,
    set: BTreeMap<String, String>,
}

impl Column {
    fn plain(backend: &str) -> Self {
        Column {
            label: backend.to_string(),
            backend: backend.to_string(),
            set: BTreeMap::new(),
        }
    }
    fn select(&self) {
        split::reset();
        for (k, v) in &self.set {
            split::set(k, Some(v));
        }
    }
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
    column: &Column,
) -> Result<(f64, AnalysisResult), String> {
    column.select();
    let mut req = AnalysisRequest::new(analysis, settings.clone());
    req.backend = Some(backend(&column.backend));
    let engine = s.engine().clone();
    let t = Instant::now();
    let result = s
        .submit_analysis(&req)
        .map_err(|e| e.to_string())
        .and_then(|id| {
            engine
                .wait_job(id, None)
                .and_then(|_| engine.take_result(id))
                .map_err(|e| e.to_string())
        });
    let ms = t.elapsed().as_secs_f64() * 1e3;
    split::reset();
    result.map(|r| (ms, r))
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

fn columns(case: &Value, keep: &Option<Vec<String>>) -> Vec<Column> {
    let backends: Vec<String> = case["backends"].as_array().map_or_else(
        || vec!["cpu".into(), "metal".into(), "auto".into()],
        |b| b.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
    );
    let variant_backend = case["variant_backend"].as_str().unwrap_or("metal");
    let mut out: Vec<Column> = Vec::new();
    for b in &backends {
        out.push(Column::plain(b));
        if b == variant_backend {
            for v in case["variants"].as_array().into_iter().flatten() {
                let name = v["name"].as_str().expect("variant name");
                let set = v["set"]
                    .as_object()
                    .expect("variant set")
                    .iter()
                    .map(|(k, x)| (k.clone(), x.as_str().expect("variant value").to_string()))
                    .collect();
                out.push(Column {
                    label: format!("{b}/{name}"),
                    backend: b.clone(),
                    set,
                });
            }
        }
    }
    if let Some(keep) = keep {
        out.retain(|c| keep.iter().any(|k| *k == c.label || *k == c.backend));
    }
    out
}

fn short(e: &str) -> String {
    e.chars().take(80).collect()
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
        "# {} commit {commit}, load {:.1}; median ms after an edit; (x) worst relative \
         difference from the CPU reference",
        a.label,
        load()
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
        let cols = columns(case, &a.backends);
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
        let cpu = Column::plain("cpu");
        // Warm-up: each column on a fresh state, after the CPU reference on it.
        let mut warm: Vec<Result<(f64, AnalysisResult), String>> = Vec::new();
        let mut comparisons: Vec<Option<Value>> = Vec::new();
        for c in &cols {
            serial += 1;
            edit(s, base, serial);
            let reference = (a.compare && c.label != "cpu")
                .then(|| run(s, analysis, &settings, &cpu).ok())
                .flatten();
            let r = run(s, analysis, &settings, c);
            comparisons.push(match (&reference, &r) {
                (Some((_, x)), Ok((_, y))) => Some(compare(x, y)),
                _ => None,
            });
            warm.push(r);
        }
        let load_before = load();
        let n = cols.len();
        let mut times = vec![Vec::new(); n];
        let mut device = vec![Vec::new(); n];
        let mut reported = vec![Vec::new(); n];
        let mut used: Vec<Option<String>> = vec![None; n];
        let mut errors: Vec<Option<String>> =
            warm.iter().map(|r| r.as_ref().err().cloned()).collect();
        let mut want = vec![reps; n];
        for _ in 0..reps {
            for (i, c) in cols.iter().enumerate() {
                if warm[i].is_err() || times[i].len() >= want[i] {
                    continue;
                }
                serial += 1;
                edit(s, base, serial);
                let d0 = mklight_metal::device_seconds();
                match run(s, analysis, &settings, c) {
                    Ok((ms, r)) => {
                        times[i].push(ms);
                        if times[i].len() == 1
                            && let Some(b) = a.budget
                        {
                            want[i] = ((b * 1e3 / ms).floor() as usize).clamp(1, reps);
                        }
                        device[i].push((mklight_metal::device_seconds() - d0) * 1e3);
                        reported[i].push(r.run_info.duration_ms);
                        used[i] = Some(r.run_info.backend.clone());
                    }
                    Err(e) => errors[i] = Some(e),
                }
            }
        }
        let load_after = load();
        let mut cells = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            let used_i = used[i].clone().or_else(|| {
                warm[i]
                    .as_ref()
                    .ok()
                    .map(|(_, r)| r.run_info.backend.clone())
            });
            let record = json!({
                "label": a.label, "commit": commit, "lens": lens, "analysis": analysis, "tag": tag,
                "settings": settings, "backend": c.label, "base_backend": c.backend,
                "variant": c.set, "used": used_i,
                "ms": (!times[i].is_empty()).then(|| times[i].clone()),
                "median_ms": (!times[i].is_empty()).then(|| median(&times[i])),
                "min_ms": times[i].iter().cloned().reduce(f64::min),
                "device_ms": (!device[i].is_empty()).then(|| median(&device[i])),
                "reported_ms": (!reported[i].is_empty()).then(|| median(&reported[i])),
                "first_ms": warm[i].as_ref().ok().map(|(ms, _)| *ms),
                "load": [load_before, load_after],
                "error": errors[i],
                "compare": comparisons[i],
            });
            if let Some(f) = out.as_mut() {
                use std::io::Write;
                writeln!(f, "{record}").unwrap();
            }
            let time = if times[i].is_empty() {
                "err".to_string()
            } else {
                format!("{:.1}", median(&times[i]))
            };
            let ran = match (&used_i, c.backend.as_str()) {
                (Some(u), "auto") => {
                    if u.starts_with("gpu") {
                        "→Metal".to_string()
                    } else {
                        "→CPU".to_string()
                    }
                }
                (Some(u), b) if b != "cpu" && !u.starts_with("gpu") => format!(" (ran on {u})"),
                _ => String::new(),
            };
            let agreement = comparisons[i]
                .as_ref()
                .and_then(|c| c["worst"].get(0).cloned())
                .map(|w| format!(" ({:.0e})", w[1].as_f64().unwrap_or(f64::INFINITY)))
                .unwrap_or_default();
            let err = errors[i]
                .as_ref()
                .map(|e| format!(" ERR {}", short(e)))
                .unwrap_or_default();
            cells.push(format!("{} {time}{ran}{agreement}{err}", c.label));
        }
        println!(
            "{:<40} {}",
            format!("{lens} {analysis} {tag}"),
            cells.join(" | ")
        );
    }
    println!("# load at end {:.1}", load());
}
