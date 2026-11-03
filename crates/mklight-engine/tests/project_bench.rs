//! Local-project release measurements; no project or catalogue is embedded.
//! MKLIGHT_BENCH_PROJECT=/path/project.mklight cargo test --release -p
//! mklight-engine --test project_bench -- --ignored --nocapture --test-threads=1
//! Optional: MKLIGHT_BENCH_AUTOSD_PROJECT, MKLIGHT_BENCH_FILTER (case names),
//! MKLIGHT_BENCH_RUNS (>=10), MKLIGHT_BENCH_COLD_RUNS (>=3),
//! MKLIGHT_BENCH_QUIET_SECONDS (default 30). Cold calls use this test binary
//! in fresh processes; the private child test does nothing without its marker.
//! Select `edit_loop_queued` to measure the six graphs on the worker queue.
//! Focused runs: MKLIGHT_BENCH_MODES (subset of cpu,df,f32,auto,f32_relaxed;
//! cpu is always kept as the accuracy reference), MKLIGHT_BENCH_PHASES
//! (warm,edit), MKLIGHT_BENCH_SKIP_COLD=1, and MKLIGHT_BENCH_ALLOW_BUSY=1 to
//! measure next to other builds (every gate record is then provisional).
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mklight_core::{JobContext, OpticalSystem, model::RayAiming, param::ParamPath};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, Session,
    analysis::{AnalysisOutput, AnalysisResult},
    backend::Backend,
    buffer::BufferData,
    gpu::GpuProvider,
};
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions, MetalPrecision};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;

const MODES: [&str; 5] = ["cpu", "df", "f32", "auto", "f32_relaxed"];
const SOLVERS: [&str; 4] = [
    "merit_evaluation",
    "optimize",
    "tolerance",
    "glass_substitution",
];
const EDIT_GRAPHS: [&str; 6] = [
    "layout",
    "spot_diagram",
    "ray_fan",
    "mtf",
    "field_curvature_distortion",
    "psf",
];

#[derive(Clone)]
struct Case {
    name: String,
    analysis: String,
    settings: Value,
    geometric: bool,
}

fn emit(kind: &str, value: Value) {
    println!("PROJECT_BENCH {}", json!({"kind":kind,"data":value}));
}

fn command(program: &str, args: &[&str]) -> String {
    let out = Command::new(program).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn load() -> Vec<f64> {
    command("sysctl", &["-n", "vm.loadavg"])
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

// Never compete with a compiler/test invocation. A busy interactive host is
// allowed after the bounded load gate, and remains labelled provisional.
fn quiet(label: &str) {
    let limit = env_usize("MKLIGHT_BENCH_QUIET_SECONDS", 30);
    let cores = std::thread::available_parallelism().unwrap().get();
    let mut ancestors = vec![std::process::id()];
    while let Some(parent) = command(
        "ps",
        &["-o", "ppid=", "-p", &ancestors.last().unwrap().to_string()],
    )
    .trim()
    .parse::<u32>()
    .ok()
    .filter(|p| *p > 0)
    {
        ancestors.push(parent);
    }
    let start = Instant::now();
    let mut observations = Vec::new();
    loop {
        let loads = load();
        let processes = command("ps", &["-axo", "pid=,pcpu=,comm="]);
        let competing: Vec<_> = processes
            .lines()
            .filter(|line| {
                let pid = line
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u32>()
                    .unwrap();
                if ancestors.contains(&pid) {
                    return false;
                }
                let name = line
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap();
                matches!(name, "cargo" | "rustc" | "swift" | "swiftc" | "xcodebuild")
                    || line.contains("/deps/")
            })
            .map(str::to_string)
            .collect();
        let quiet = loads[0] < cores as f64 * 0.5 && competing.is_empty();
        observations.push(json!({"load":loads,"compilers":competing}));
        if quiet || start.elapsed().as_secs() >= limit as u64 {
            emit(
                "gate",
                json!({"block":label,"quiet":quiet,"provisional":!quiet,
                "wait_seconds":start.elapsed().as_secs_f64(),"observations":observations}),
            );
            assert!(
                competing.is_empty() || std::env::var("MKLIGHT_BENCH_ALLOW_BUSY").is_ok(),
                "another build is running; retry after it finishes"
            );
            return;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|s| s.parse().unwrap())
        .unwrap_or(default)
}

fn engine(mode: &str) -> Engine {
    let e = Engine::new(EngineConfig::default());
    // Fixed options avoid process-global environment mutation while alternating
    // precisions. Fresh f32 children additionally set MKLIGHT_METAL_PRECISION.
    let precision = if mode.starts_with("f32") {
        MetalPrecision::PureF32
    } else {
        MetalPrecision::DoubleFloat
    };
    e.register_backend(Arc::new(GpuProvider::new(
        "metal",
        "project measurement",
        move || {
            MetalBackend::with_options(&MetalOptions {
                precision,
                library: LibraryChoice::Precompiled,
                ..Default::default()
            })
            .map_err(|e| e.to_string())?
            .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>)
            .ok_or_else(|| "no Metal device".into())
        },
    )));
    e
}

fn open(mode: &str, path: &Path) -> Session {
    let e = engine(mode);
    let mut s = e.open_session(path).unwrap();
    assert!(s.open_warnings().is_empty(), "{:?}", s.open_warnings());
    if mode == "f32_relaxed" {
        let mut system = s.system().clone();
        system.ray_aiming = RayAiming::Real { tolerance: 1e-6 };
        s.set_system(system).unwrap();
    }
    s
}

fn request(case: &Case, mode: &str) -> AnalysisRequest {
    let mut r = AnalysisRequest::new(&case.analysis, case.settings.clone());
    r.backend = Some(match mode {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        _ => Backend::Gpu("metal".into()),
    });
    // Explicit requests still report unsupported/static CPU routes.
    r.allow_backend_fallback = true;
    r
}

fn variables(system: &OpticalSystem) -> Vec<Value> {
    let mut vars = Vec::new();
    for s in system
        .surfaces
        .iter()
        .skip(1)
        .take(system.surfaces.len() - 2)
    {
        if s.shape.curvature() != 0. {
            vars.push(json!({"param":ParamPath::Curvature{surface:s.id}}));
        }
        if s.thickness.is_finite() && s.thickness > 0. {
            vars.push(json!({"param":ParamPath::Thickness{surface:s.id},
                "lower":s.thickness * 0.8,"upper":s.thickness * 1.2}));
        }
    }
    // Solves control their targets before merit generation.
    vars.retain(|v| !system.solves.iter().any(|s| json!(s.target) == v["param"]));
    vars
}

fn cases(s: &Session) -> Vec<Case> {
    let vars = variables(s.system());
    let glass = s
        .system()
        .surfaces
        .iter()
        .skip(1)
        .find(|s| matches!(s.material, mklight_core::MaterialRef::Catalog { .. }));
    let mut cases = Vec::new();
    for info in s.engine().analyses() {
        let settings = match info.name.as_str() {
            "glass_info" => match glass.map(|g| &g.material) {
                Some(mklight_core::MaterialRef::Catalog { name, catalog, .. }) => {
                    json!({"name":name,"catalog":catalog})
                }
                _ => json!({"name":"N-BK7"}),
            },
            "optimize" => {
                json!({"variables":vars,"max_iterations":3,"options":{"ftol":0.,"xtol":0.,"gtol":0.},"seed":71})
            }
            "tolerance" => json!({"trials":16,"seed":71,"record_trials":16}),
            "glass_substitution" => json!({"variables":vars,"material_variables":
                glass.map(|g| vec![json!({"surface":g.id,"candidates":{"nearest":3}})]).unwrap_or_default(),
                "max_combinations":3,"local_iterations":2,"top_n":3,"seed":71}),
            _ => Value::Null,
        };
        cases.push(Case {
            name: info.name.clone(),
            analysis: info.name,
            settings,
            geometric: info.category == "geometric" || info.category == "layout",
        });
    }
    for name in [
        "spot_diagram",
        "geometric_mtf",
        "ray_grid",
        "best_focus",
        "encircled_energy",
    ] {
        cases.push(Case {
            name: format!("{name}_dense"),
            analysis: name.into(),
            settings: json!({"rings":if matches!(name,"spot_diagram" | "geometric_mtf") {512} else {64}}),
            geometric: true,
        });
    }
    for name in ["ray_fan", "opd_fan"] {
        cases.push(Case {
            name: format!("{name}_dense"),
            analysis: name.into(),
            settings: json!({"samples":1025}),
            geometric: true,
        });
    }
    for name in [
        "psf",
        "mtf",
        "through_focus_mtf",
        "mtf_vs_field",
        "line_spread",
        "diffraction_encircled_energy",
        "wavefront_map",
    ] {
        let mut settings = json!({"pupil":{"grid":512,"rays":129}});
        if matches!(name, "psf" | "line_spread" | "diffraction_encircled_energy") {
            settings["grid_size"] = json!(1024);
            let resolved = s.resolved().unwrap();
            let first =
                mklight_trace::paraxial::first_order(&resolved, resolved.primary_wavelength)
                    .unwrap();
            let shortest = resolved
                .wavelengths_um
                .iter()
                .copied()
                .reduce(f64::min)
                .unwrap();
            if name == "psf" {
                settings["spacing_mm"] =
                    json!(shortest * 1e-3 * first.working_f_number.abs() / 2.5);
            }
        }
        cases.push(Case {
            name: format!("{name}_dense"),
            analysis: name.into(),
            settings,
            geometric: false,
        });
    }
    cases
}

// Every edit has a distinct content hash, including across warm/cold blocks.
// Alternating just two hashes would still hit the immutable operation caches.
fn edit(s: &mut Session, base: &OpticalSystem, serial: usize) {
    let mut system = base.clone();
    let surface = system
        .surfaces
        .iter()
        .skip(1)
        .find(|s| s.shape.curvature() != 0.)
        .unwrap();
    let param = ParamPath::Curvature {
        surface: surface.id,
    };
    let curvature = surface.shape.curvature();
    let relative = (if serial.is_multiple_of(2) { 1. } else { -1. })
        * (1. + serial as f64 / 1_000_000.)
        * 1e-6;
    param.set(&mut system, curvature / (1. + relative)).unwrap();
    s.set_system(system).unwrap();
}

fn output(r: &AnalysisResult) -> AnalysisOutput {
    AnalysisOutput {
        json: r.json.clone(),
        buffers: r.buffers.clone(),
        ..Default::default()
    }
}

// Per-quantity detail uses the former chain comparator's scalar budgets.
// Images additionally report the full §9.4 L1 and energy fractions. The full
// comparator below remains authoritative for masks, discrete values and fits.
fn budget(path: &str, a: f64, energy: Option<f64>, interval: Option<f64>) -> f64 {
    let b = if let Some(flux) = energy.filter(|_| path.starts_with("result.energy.")) {
        1e-5 * flux + 1e-10
    } else if path.contains("direction") {
        2e-6
    } else if ["opd", "wavefront", "zernike"]
        .iter()
        .any(|s| path.contains(s))
    {
        5e-3
    } else if [
        "modulation",
        "mtf",
        "strehl",
        "encircled",
        "ensquared",
        "buffer.tangential",
        "buffer.sagittal",
        "diffraction_limit",
    ]
    .iter()
    .any(|s| path.contains(s))
    {
        2e-3
    } else if path.contains("angle") || path.contains("incidence") {
        2e-4
    } else if path.contains("magnification") || path.contains("variation") {
        (1e-3 * a.abs()).max(1e-5)
    } else if ["rms", "radius", "merit"].iter().any(|s| path.contains(s)) {
        (0.005 * a.abs()).max(2e-5)
    } else if path.contains("retardance") {
        1e-3
    } else if ["transmission", "flux", "weight"]
        .iter()
        .any(|s| path.contains(s))
    {
        1e-5 * a.abs() + 1e-10
    } else {
        2e-5
    };
    if ["r50", "r80", "r90", "half_width80"]
        .iter()
        .any(|s| path.contains(s))
    {
        interval.map_or(b, |step| b.min(step))
    } else {
        b
    }
}

#[derive(Default, serde::Serialize)]
struct Quantity {
    max_abs: f64,
    max_budget_fraction: f64,
    mismatches: usize,
    count: usize,
}

fn accuracy(a: &AnalysisResult, b: &AnalysisResult, name: &str) -> Value {
    let (mut a, mut b) = (output(a), output(b));
    // Solver execution telemetry and returned designs legitimately differ.
    // Compare their physical score reports, not timing/backend identity.
    if SOLVERS.contains(&name) {
        fn scores(name: &str, v: &Value) -> Value {
            match name {
                "merit_evaluation" => {
                    json!({"merit":v["merit"],"violation":v["violation"],"status":v["status"],"operands":v["operands"]})
                }
                "optimize" => json!({"start":v["start"],"final":v["final"]}),
                "tolerance" => {
                    let mut mc = v["monte_carlo"].clone();
                    if let Some(object) = mc.as_object_mut() {
                        object.remove("seconds_per_trial");
                    }
                    json!({"sensitivity":v["sensitivity"],"monte_carlo":mc})
                }
                _ => {
                    json!({"finalists":v["results"].as_array().map(|rs| rs.iter().map(|r| json!({"merit":r["merit"],"violation":r["violation"],"feasible":r["feasible"],"glasses":r["glasses"]})).collect::<Vec<_>>())})
                }
            }
        }
        a.json = scores(name, &a.json);
        b.json = scores(name, &b.json);
        a.buffers.clear();
        b.buffers.clear();
    }
    let c = comparison::compare(&a, &b, name, &JobContext::new()).unwrap();
    let mut quantities = BTreeMap::<String, Quantity>::new();
    let energy = a.json["energy"]["emitted"].as_f64();
    let radius: Vec<f64> = a
        .buffers
        .iter()
        .find(|b| b.name == "radius_mm")
        .and_then(|b| match &b.data {
            BufferData::F64(v) => Some(v.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            a.json["radius"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        });
    let interval = radius
        .windows(2)
        .map(|v| (v[1] - v[0]).abs())
        .filter(|v| *v > 0.)
        .reduce(f64::min);
    fn number(
        m: &mut BTreeMap<String, Quantity>,
        path: &str,
        a: f64,
        b: f64,
        energy: Option<f64>,
        interval: Option<f64>,
    ) {
        if !m.contains_key(path) {
            m.insert(path.into(), Quantity::default());
        }
        let q = m.get_mut(path).unwrap();
        q.count += 1;
        if a == b || a.is_nan() && b.is_nan() {
            return;
        }
        if !a.is_finite() || !b.is_finite() {
            q.mismatches += 1;
            return;
        }
        let error = (a - b).abs();
        q.max_abs = q.max_abs.max(error);
        q.max_budget_fraction = q
            .max_budget_fraction
            .max(error / budget(path, a, energy, interval));
    }
    fn walk(
        m: &mut BTreeMap<String, Quantity>,
        a: &Value,
        b: &Value,
        path: &str,
        energy: Option<f64>,
        interval: Option<f64>,
    ) {
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for (k, v) in a {
                    if ![
                        "run_info",
                        "run",
                        "warnings",
                        "evaluations",
                        "iterations",
                        "duration_ms",
                    ]
                    .contains(&k.as_str())
                    {
                        walk(
                            m,
                            v,
                            b.get(k).unwrap_or(&Value::Null),
                            &format!("{path}.{k}"),
                            energy,
                            interval,
                        );
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
                for (a, b) in a.iter().zip(b) {
                    walk(m, a, b, &format!("{path}[]"), energy, interval);
                }
            }
            (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => number(
                m,
                path,
                a.as_f64().unwrap(),
                b.as_f64().unwrap(),
                energy,
                interval,
            ),
            _ if a == b => (),
            _ => m.entry(path.into()).or_default().mismatches += 1,
        }
    }
    walk(
        &mut quantities,
        &a.json,
        &b.json,
        "result",
        energy,
        interval,
    );
    for x in &a.buffers {
        let Some(y) = b
            .buffers
            .iter()
            .find(|y| x.name == y.name && x.shape == y.shape)
        else {
            quantities
                .entry(format!("buffer.{}", x.name))
                .or_default()
                .mismatches += 1;
            continue;
        };
        let floats = |d: &BufferData| match d {
            BufferData::F64(v) => Some(v.clone()),
            BufferData::F32(v) => Some(v.iter().map(|x| *x as f64).collect::<Vec<_>>()),
            _ => None,
        };
        if let (Some(xs), Some(ys)) = (floats(&x.data), floats(&y.data)) {
            let path = format!("buffer.{}", x.name);
            let image = (name == "psf" && path.contains("data_"))
                || (name == "line_spread"
                    && (path.starts_with("buffer.x_") || path.starts_with("buffer.y_")))
                || ["intensity", "irradiance", "line_spread"]
                    .iter()
                    .any(|v| path.contains(v))
                || path == "buffer.image";
            let mut image_norm = 0.;
            let mut image_delta = 0.;
            let mut image_energy = 0.;
            let columns = (x.name == "values")
                .then(|| a.json["columns"].as_array())
                .flatten()
                .map(|cs| {
                    cs.iter()
                        .map(|c| format!("buffer.values.{}", c["name"].as_str().unwrap()))
                        .collect::<Vec<_>>()
                });
            for (i, (&xv, &yv)) in xs.iter().zip(&ys).enumerate() {
                let quantity = columns
                    .as_ref()
                    .map_or(path.as_str(), |cs| cs[i % cs.len()].as_str());
                number(&mut quantities, quantity, xv, yv, energy, interval);
                if image && xv.is_finite() && yv.is_finite() {
                    image_norm += xv.abs();
                    image_delta += (xv - yv).abs();
                    image_energy += xv - yv;
                }
            }
            if image && image_norm > 0. {
                let q = quantities.get_mut(&path).unwrap();
                q.max_budget_fraction = (image_delta / (image_norm * 2e-3))
                    .max(image_energy.abs() / (image_norm * 1e-3));
            }
        } else if x.data != y.data {
            quantities
                .entry(format!("buffer.{}", x.name))
                .or_default()
                .mismatches += 1;
        }
    }
    json!({"pass_sampled_margin":c.reason.is_none(),"chain_failure":c.reason,"compared":c.compared,
        "max_abs":c.maximum,"max_path":c.maximum_path,"image_l1_relative":c.image_l1_relative,
        "image_l1_budget_fraction":c.image_l1_relative/2e-3,"image_energy_relative":c.energy_relative,
        "image_energy_budget_fraction":c.energy_relative/1e-3,"quantities":quantities})
}

fn measured(
    s: &Session,
    case: &Case,
    mode: &str,
    phase: &str,
    round: usize,
    detail: bool,
) -> Option<AnalysisResult> {
    let before = mklight_metal::device_seconds();
    let start = Instant::now();
    let result = s.run_analysis(&request(case, mode), &JobContext::new());
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let device_ms = (mklight_metal::device_seconds() - before) * 1e3;
    let mut record = json!({"case":case.name,"mode":mode,"phase":phase,"round":round,
        "wall_ms":wall_ms,"device_ms":device_ms,"load":load(),"snapshot":s.snapshot_hash().to_hex()});
    match result {
        Ok(r) => {
            record["used"] = json!(r.backend.used);
            record["fallback_reason"] = json!(r.backend.fallback_reason);
            record["cost_decision"] = json!(r.backend.cost_decision);
            record["solver_backend"] = r.json.get("backend").cloned().unwrap_or(Value::Null);
            let envelope = r.envelope();
            let mut stats = Vec::new();
            plan_stats(&envelope, &mut stats);
            record["plan_stats"] = json!(
                stats
                    .iter()
                    .map(|s| json!({"device_seconds":s["device_seconds"],
                        "cache":s.get("cache"),"aiming":s.get("aiming"),
                        "host":s.get("host"),"peak_bytes":s["peak_bytes"],
                        "memory":s.get("memory")}))
                    .collect::<Vec<_>>()
            );
            if detail {
                record["envelope"] = envelope;
            }
            emit("run", record);
            Some(r)
        }
        Err(e) => {
            record["error"] = json!(e.to_string());
            emit("run", record);
            None
        }
    }
}

fn mode_selected(mode: usize) -> bool {
    mode == 0
        || std::env::var("MKLIGHT_BENCH_MODES")
            .map_or(true, |s| s.split(',').any(|m| m == MODES[mode]))
}

fn phase_selected(phase: &str) -> bool {
    std::env::var("MKLIGHT_BENCH_PHASES").map_or(true, |s| s.split(',').any(|p| p == phase))
}

/// Resident plan statistics of every native execution in an envelope.
fn plan_stats(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                if k == "plan_stats" {
                    out.push(x.clone());
                } else {
                    plan_stats(x, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| plan_stats(x, out)),
        _ => {}
    }
}

fn selected(name: &str) -> bool {
    std::env::var("MKLIGHT_BENCH_FILTER").map_or(true, |s| s.split(',').any(|n| n == name))
}

fn cold(case: &Case, path: &Path, round: usize, mode: &str) {
    let start = Instant::now();
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "project_bench_cold",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MKLIGHT_BENCH_CHILD", "1")
        .env("MKLIGHT_BENCH_PROJECT", path)
        .env("MKLIGHT_BENCH_CASE", &case.name)
        .env("MKLIGHT_BENCH_ANALYSIS", &case.analysis)
        .env("MKLIGHT_BENCH_SETTINGS", case.settings.to_string())
        .env("MKLIGHT_BENCH_MODE", mode)
        .env(
            "MKLIGHT_METAL_PRECISION",
            if mode.starts_with("f32") { "f32" } else { "df" },
        )
        .output()
        .unwrap();
    emit(
        "cold_process",
        json!({"case":case.name,"mode":mode,"round":round,
        "process_wall_ms":start.elapsed().as_secs_f64()*1e3,"success":out.status.success(),
        "stdout":String::from_utf8_lossy(&out.stdout),"stderr":String::from_utf8_lossy(&out.stderr)}),
    );
    assert!(out.status.success(), "cold child failed");
}

#[test]
#[ignore = "local-project release measurements on a quiet Metal host"]
fn project_bench() {
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").expect("set MKLIGHT_BENCH_PROJECT");
    let runs = env_usize("MKLIGHT_BENCH_RUNS", 10).max(10);
    let cold_runs = env_usize("MKLIGHT_BENCH_COLD_RUNS", 3).max(3);
    quiet("start");
    emit(
        "host",
        json!({"unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "os":command("sw_vers",&[]),"kernel":command("uname",&["-a"]),
        "hardware":command("sysctl",&["-n","machdep.cpu.brand_string","hw.memsize","hw.ncpu"]),
        "rust":command("rustc",&["-Vv"]),"git":command("git",&["rev-parse","HEAD"]),
        "dirty":command("git",&["status","--short"]),"rayon_workers":rayon::current_num_threads(),
        "engine_workers":std::thread::available_parallelism().unwrap().get().min(4),
        "env":std::env::vars().filter(|(k,_)|k.starts_with("MKLIGHT_")||k=="RAYON_NUM_THREADS").collect::<BTreeMap<_,_>>(),
        "processes":command("ps",&["-axo","pid,pcpu,comm"]),"runs":runs,"cold_runs":cold_runs}),
    );
    for (variant, path) in std::iter::once(("fixed".to_string(), path)).chain(
        std::env::var("MKLIGHT_BENCH_AUTOSD_PROJECT")
            .ok()
            .map(|p| ("autosd".into(), p)),
    ) {
        emit("variant", json!({"name":variant}));
        let path = Path::new(&path);
        let mut sessions: Vec<_> = MODES.iter().map(|m| open(m, path)).collect();
        let bases: Vec<_> = sessions.iter().map(|s| s.system().clone()).collect();
        emit(
            "project",
            json!({"variant":variant,"hash":sessions[0].snapshot_hash().to_hex(),
            "surfaces":bases[0].surfaces.len(),"fields":bases[0].fields.fields.len(),
            "wavelengths":bases[0].wavelengths.wavelengths.len(),"aiming":bases[0].ray_aiming,
            "catalogue_hash":sessions[0].library_hash(),"analyses":sessions[0].engine().analyses()}),
        );
        let all = cases(&sessions[0]);
        for (index, case) in all
            .iter()
            .enumerate()
            .filter(|(_, c)| selected(&c.name) && (variant == "fixed" || c.geometric))
        {
            emit(
                "case",
                json!({"variant":variant,"name":case.name,"analysis":case.analysis,"settings":case.settings}),
            );
            for phase in ["warm", "edit"].into_iter().filter(|p| phase_selected(p)) {
                quiet(&format!("{variant}/{}/{phase}", case.name));
                let mut first: Vec<Option<AnalysisResult>> =
                    (0..MODES.len()).map(|_| None).collect();
                let mut successful = [0; MODES.len()];
                for round in 0..=runs {
                    let mut order: Vec<_> = (0..MODES.len()).collect();
                    order.rotate_left(round % MODES.len());
                    if round % 2 == 1 {
                        order.reverse();
                    }
                    for mode in order.into_iter().filter(|&m| mode_selected(m)) {
                        if phase == "edit" {
                            edit(
                                &mut sessions[mode],
                                &bases[mode],
                                10000 + index * (runs + 1) + round,
                            );
                        } else if round == 0 {
                            sessions[mode].set_system(bases[mode].clone()).unwrap();
                        }
                        let result =
                            measured(&sessions[mode], case, MODES[mode], phase, round, round == 1);
                        if round > 0
                            && let Some(r) = result
                        {
                            successful[mode] += 1;
                            if round == 1 {
                                first[mode] = Some(r);
                            }
                        }
                    }
                }
                // Comparison and CPU relaxed reference are outside measured calls.
                let relaxed = first[4].as_ref().map(|_| {
                    let mut cpu = open("cpu", path);
                    cpu.set_system(sessions[4].system().clone()).unwrap();
                    // Use the first measured edit, not the final one.
                    if phase == "edit" {
                        let mut base = bases[0].clone();
                        base.ray_aiming = RayAiming::Real { tolerance: 1e-6 };
                        edit(&mut cpu, &base, 10000 + index * (runs + 1) + 1);
                    }
                    cpu.run_analysis(&request(case, "cpu"), &JobContext::new())
                });
                for mode in (0..MODES.len()).filter(|m| mode_selected(*m)) {
                    let reference = if mode == 4 {
                        relaxed.as_ref().and_then(|r| r.as_ref().ok())
                    } else {
                        first[0].as_ref()
                    };
                    if mode > 0
                        && let (Some(a), Some(b)) = (reference, first[mode].as_ref())
                    {
                        emit(
                            "accuracy",
                            json!({"case":case.name,"mode":MODES[mode],"phase":phase,
                            "reference_aiming":if mode==4 {"1e-6"} else {"project"},
                            "report":accuracy(a,b,&case.analysis)}),
                        );
                    }
                    emit(
                        "block_complete",
                        json!({"case":case.name,"mode":MODES[mode],"phase":phase,
                        "successful_runs":successful[mode]}),
                    );
                }
            }
            if std::env::var("MKLIGHT_BENCH_SKIP_COLD").is_ok() {
                continue;
            }
            quiet(&format!("{variant}/{} / cold", case.name));
            for round in 0..cold_runs {
                for mode in (0..MODES.len())
                    .map(|offset| (round + offset) % MODES.len())
                    .filter(|&m| mode_selected(m))
                {
                    cold(case, path, round, MODES[mode]);
                }
            }
        }
        if selected("edit_loop") {
            quiet(&format!("{variant}/edit_loop"));
            for round in 0..=runs {
                for offset in 0..MODES.len() {
                    let mode = (round + offset) % MODES.len();
                    if !mode_selected(mode) {
                        continue;
                    }
                    let s = &mut sessions[mode];
                    let start = Instant::now();
                    edit(s, &bases[mode], 900000 + round);
                    let mut results = Vec::new();
                    let mut errors = Vec::new();
                    for name in EDIT_GRAPHS {
                        let c = all.iter().find(|c| c.name == name).unwrap();
                        match s.run_analysis(&request(c, MODES[mode]), &JobContext::new()) {
                            Ok(r) => results.push(
                                json!({"analysis":name,"run_info":r.run_info,"backend":r.backend}),
                            ),
                            Err(e) => errors.push(json!({"analysis":name,"error":e.to_string()})),
                        }
                    }
                    emit(
                        "edit_loop",
                        json!({"mode":MODES[mode],"round":round,
                        "wall_ms":start.elapsed().as_secs_f64()*1e3,"results":results,"errors":errors,"load":load()}),
                    );
                    // The same six graphs again on the unchanged snapshot.
                    let start = Instant::now();
                    let mut used = Vec::new();
                    for name in EDIT_GRAPHS {
                        let c = all.iter().find(|c| c.name == name).unwrap();
                        let r = s.run_analysis(&request(c, MODES[mode]), &JobContext::new());
                        used.push(json!({"analysis":name,
                            "used":r.as_ref().map(|r| r.backend.used.clone()).map_err(|e| e.to_string())}));
                    }
                    emit(
                        "edit_loop_warm",
                        json!({"mode":MODES[mode],"round":round,
                        "wall_ms":start.elapsed().as_secs_f64()*1e3,"results":used,"load":load()}),
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "private fresh-process entry for project_bench"]
fn project_bench_cold() {
    if std::env::var("MKLIGHT_BENCH_CHILD").as_deref() != Ok("1") {
        return;
    }
    let start = Instant::now();
    let mode = std::env::var("MKLIGHT_BENCH_MODE").unwrap();
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").unwrap();
    let s = open(&mode, Path::new(&path));
    let name = std::env::var("MKLIGHT_BENCH_CASE").unwrap();
    let case = Case {
        name,
        analysis: std::env::var("MKLIGHT_BENCH_ANALYSIS").unwrap(),
        settings: serde_json::from_str(&std::env::var("MKLIGHT_BENCH_SETTINGS").unwrap()).unwrap(),
        geometric: false,
    };
    let setup_ms = start.elapsed().as_secs_f64() * 1e3;
    let call = Instant::now();
    let before = mklight_metal::device_seconds();
    let result = s.run_analysis(&request(&case, &mode), &JobContext::new());
    let wall_ms = call.elapsed().as_secs_f64() * 1e3;
    let first_call_ms = start.elapsed().as_secs_f64() * 1e3;
    let record = match result {
        Ok(r) => json!({"envelope":r.envelope(),"used":r.backend.used}),
        Err(e) => json!({"error":e.to_string()}),
    };
    emit(
        "cold_total",
        json!({"setup_ms":setup_ms,"first_call_ms":first_call_ms,
        "wall_ms":wall_ms,"device_ms":(mklight_metal::device_seconds()-before)*1e3,"result":record}),
    );
}

/// The same graph set through the engine's worker queue, matching concurrent
/// pane refreshes. Result formatting stays outside the total wall timer.
#[test]
#[ignore = "local-project queued app edit-loop measurements"]
fn project_bench_queued_edit_loop() {
    if !selected("edit_loop_queued") {
        return;
    }
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").expect("set MKLIGHT_BENCH_PROJECT");
    let runs = env_usize("MKLIGHT_BENCH_RUNS", 10).max(10);
    quiet("queued edit loop / start");
    emit(
        "queued_host",
        json!({
            "git":command("git",&["rev-parse","HEAD"]),
            "dirty":command("git",&["status","--short"]),
            "os":command("sw_vers",&[]),"rust":command("rustc",&["-Vv"]),
            "rayon_workers":rayon::current_num_threads(),
            "engine_workers":std::thread::available_parallelism().unwrap().get().min(4)
        }),
    );
    for (variant, path) in std::iter::once(("fixed", path)).chain(
        std::env::var("MKLIGHT_BENCH_AUTOSD_PROJECT")
            .ok()
            .map(|p| ("autosd", p)),
    ) {
        emit("variant", json!({"name":variant}));
        let mut sessions: Vec<_> = MODES.iter().map(|m| open(m, Path::new(&path))).collect();
        let bases: Vec<_> = sessions.iter().map(|s| s.system().clone()).collect();
        let all = cases(&sessions[0]);
        quiet(&format!("{variant}/queued edit loop"));
        for round in 0..=runs {
            for offset in 0..MODES.len() {
                let mode = (round + offset) % MODES.len();
                if !mode_selected(mode) {
                    continue;
                }
                let session = &mut sessions[mode];
                let engine = session.engine().clone();
                let before = mklight_metal::device_seconds();
                let start = Instant::now();
                edit(session, &bases[mode], 900000 + round);
                let mut jobs = Vec::new();
                let mut errors = Vec::new();
                for name in EDIT_GRAPHS {
                    let case = all.iter().find(|c| c.name == name).unwrap();
                    let req = request(case, MODES[mode]).view_key(name);
                    match session.submit_analysis(&req) {
                        Ok(id) => jobs.push((name, id)),
                        Err(e) => errors.push(json!({"analysis":name,"error":e.to_string()})),
                    }
                }
                let mut outputs = Vec::new();
                for (name, id) in jobs {
                    match engine
                        .wait_job(id, None)
                        .and_then(|_| engine.take_result(id))
                    {
                        Ok(r) => outputs.push((name, r)),
                        Err(e) => errors.push(json!({"analysis":name,"error":e.to_string()})),
                    }
                }
                let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                let device_ms = (mklight_metal::device_seconds() - before) * 1e3;
                let results: Vec<_> = outputs.into_iter().map(|(name, r)| {
                    json!({"analysis":name,"run_info":r.run_info,"backend":r.backend})
                }).collect();
                emit(
                    "edit_loop_queued",
                    json!({
                        "mode":MODES[mode],"round":round,"wall_ms":wall_ms,"device_ms":device_ms,
                        "snapshot":session.snapshot_hash().to_hex(),"library_hash":session.library_hash(),"results":results,"errors":errors,
                        "load":load()
                    }),
                );
            }
        }
    }
}

/// Focused strategy experiments. Settings and private inputs stay outside the repo.
/// MKLIGHT_ASTRA_CASES='[{"name":"spot","analysis":"spot_diagram","settings":{}}]'
/// MKLIGHT_ASTRA_MODES=cpu,df MKLIGHT_ASTRA_RUNS=10
#[test]
#[ignore = "serial release strategy measurements on an idle host"]
fn astra_strategy_probe() {
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").expect("local project path");
    let spec: Vec<Value> =
        serde_json::from_str(&std::env::var("MKLIGHT_ASTRA_CASES").expect("JSON cases")).unwrap();
    let modes = std::env::var("MKLIGHT_ASTRA_MODES").unwrap_or("cpu,df".into());
    let modes: Vec<_> = modes.split(',').collect();
    let runs = env_usize("MKLIGHT_ASTRA_RUNS", 10);
    quiet("astra strategy");
    emit(
        "astra_setup",
        json!({"revision":command("git", &["rev-parse", "HEAD"]),
        "host":command("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "threads":rayon::current_num_threads(),"runs":runs,"modes":modes}),
    );
    let sessions: Vec<_> = modes.iter().map(|m| open(m, Path::new(&path))).collect();
    for v in spec {
        let case = Case {
            name: v["name"].as_str().unwrap().into(),
            analysis: v["analysis"].as_str().unwrap().into(),
            settings: v["settings"].clone(),
            geometric: false,
        };
        let mut first = Vec::new();
        for (s, mode) in sessions.iter().zip(&modes) {
            measured(s, &case, mode, "astra_warmup", 0, true).unwrap();
        }
        for round in 0..runs {
            let order: Vec<_> = if round % 2 == 0 {
                (0..modes.len()).collect()
            } else {
                (0..modes.len()).rev().collect()
            };
            for i in order {
                let r = measured(&sessions[i], &case, modes[i], "astra_warm", round, true).unwrap();
                if round == 0 {
                    first.push((i, r));
                }
            }
        }
        if let Some((_, reference)) = first.iter().find(|(i, _)| modes[*i] == "cpu") {
            for (i, r) in &first {
                if modes[*i] != "cpu" {
                    emit(
                        "astra_accuracy",
                        json!({"case":case.name,"mode":modes[*i],
                        "comparison":accuracy(reference, r, &case.analysis)}),
                    );
                }
            }
        }
    }
}
