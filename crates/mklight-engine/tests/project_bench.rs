//! Local-project release measurements; no project or catalogue is embedded.
//! MKLIGHT_BENCH_PROJECT=/path/project.mklight cargo test --release -p
//! mklight-engine --test project_bench -- --ignored --nocapture --test-threads=1
//! Optional: MKLIGHT_BENCH_AUTOSD_PROJECT, MKLIGHT_BENCH_FILTER (case names),
//! MKLIGHT_BENCH_RUNS (>=10), MKLIGHT_BENCH_COLD_RUNS (>=3),
//! MKLIGHT_BENCH_QUIET_SECONDS (default 30). Cold calls use this test binary
//! in fresh processes; the private child test does nothing without its marker.
//! Select `edit_loop_queued` to measure the six graphs on the worker queue.
//! Focused runs: MKLIGHT_BENCH_MODES (subset of cpu,df,auto; cpu is always
//! kept as the accuracy reference), MKLIGHT_BENCH_PHASES
//! (warm,edit), MKLIGHT_BENCH_SKIP_COLD=1, MKLIGHT_BENCH_GRAPH_BYTES (analysis
//! graph budget; 0 disables it), MKLIGHT_BENCH_EDIT_GRAPHS (comma list
//! replacing the six edit-loop graphs), MKLIGHT_BENCH_ALLOW_BUSY=1 to
//! measure next to other builds (every gate record is then provisional),
//! MKLIGHT_BENCH_AIM_STOP (requested|conditioned) and MKLIGHT_BENCH_TIER
//! (design|preview) for the accuracy policies. MKLIGHT_BENCH_MIN_RUNS lowers
//! the run minimums for quick provisional passes.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mklight_core::{JobContext, OpticalSystem, param::ParamPath};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, Session,
    analysis::{AnalysisOutput, AnalysisResult},
    backend::Backend,
    gpu::GpuProvider,
};
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;

const MODES: [&str; 3] = ["cpu", "df", "auto"];
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

/// The minimum of a run count: `floor`, or a quick provisional pass with
/// `MKLIGHT_BENCH_MIN_RUNS` (at least 1).
fn min_runs(floor: usize) -> usize {
    floor.min(env_usize("MKLIGHT_BENCH_MIN_RUNS", floor).max(1))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|s| s.parse().unwrap())
        .unwrap_or(default)
}

fn engine() -> Engine {
    let mut config = EngineConfig::default();
    if let Ok(b) = std::env::var("MKLIGHT_BENCH_GRAPH_BYTES") {
        config.analysis_graph_bytes = b.parse().unwrap();
    }
    // MKLIGHT_BENCH_AIM_STOP=requested|conditioned (CONVENTIONS §7).
    if let Ok(s) = std::env::var("MKLIGHT_BENCH_AIM_STOP") {
        config.aim_stop = match s.as_str() {
            "conditioned" => mklight_core::AimStopKind::Conditioned,
            _ => mklight_core::AimStopKind::Requested,
        };
    }
    let e = Engine::new(config);
    e.register_backend(Arc::new(GpuProvider::new(
        "metal",
        "project measurement",
        || {
            MetalBackend::with_options(&MetalOptions {
                library: LibraryChoice::Precompiled,
            })
            .map_err(|e| e.to_string())?
            .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>)
            .ok_or_else(|| "no Metal device".into())
        },
    )));
    e
}

fn open(path: &Path) -> Session {
    let s = engine().open_session(path).unwrap();
    assert!(s.open_warnings().is_empty(), "{:?}", s.open_warnings());
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
    // MKLIGHT_BENCH_TIER=preview runs the preview tier (GPU §9.4).
    if std::env::var("MKLIGHT_BENCH_TIER").as_deref() == Ok("preview") {
        r.tier = mklight_core::Tier::Preview;
    }
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
            geometric: info.category == "geometric",
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

fn accuracy(a: &AnalysisResult, b: &AnalysisResult, name: &str) -> Value {
    let (mut a, mut b) = (output(a), output(b));
    // Solver execution telemetry and returned designs legitimately differ.
    // Compare their physical score reports, not timing/backend identity.
    if SOLVERS.contains(&name) {
        a.json = comparison::solver_scores(name, &a.json);
        b.json = comparison::solver_scores(name, &b.json);
        a.buffers.clear();
        b.buffers.clear();
    }
    // Per-quantity detail: the contract's classes (support/comparison.rs,
    // GPU §9.4), each with its worst error as a fraction of its gate.
    let c = comparison::compare(&a, &b, name, &JobContext::new()).unwrap();
    json!({"pass_sampled_margin":c.reason.is_none(),"gate_failure":c.reason,"gate_failures":c.failures,
        "compared":c.compared,"max_abs":c.maximum,"max_path":c.maximum_path,"image_l1_relative":c.image_l1_relative,
        "image_l1_budget_fraction":c.image_l1_relative/2e-3,"image_energy_relative":c.energy_relative,
        "image_energy_budget_fraction":c.energy_relative/1e-3,"classes":c.classes,
        "resampled":c.resampled,"sampling_differences":c.sampling_differences})
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
                        "memory":s.get("memory"),
                        "max_command_seconds":s["stages"].as_array().map(|st| st
                            .iter()
                            .filter_map(|v| v["max_command_seconds"].as_f64())
                            .fold(0., f64::max))}))
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

/// The edit-loop graphs (`MKLIGHT_BENCH_EDIT_GRAPHS` or the six defaults).
fn edit_graphs() -> Vec<String> {
    std::env::var("MKLIGHT_BENCH_EDIT_GRAPHS").map_or_else(
        |_| EDIT_GRAPHS.iter().map(|s| s.to_string()).collect(),
        |s| s.split(',').map(String::from).collect(),
    )
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
    let runs = env_usize("MKLIGHT_BENCH_RUNS", 10).max(min_runs(10));
    let cold_runs = env_usize("MKLIGHT_BENCH_COLD_RUNS", 3).max(min_runs(3));
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
        let mut sessions: Vec<_> = MODES.iter().map(|_| open(path)).collect();
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
                // The comparison is outside measured calls.
                for mode in (0..MODES.len()).filter(|m| mode_selected(*m)) {
                    if mode > 0
                        && let (Some(a), Some(b)) = (first[0].as_ref(), first[mode].as_ref())
                    {
                        emit(
                            "accuracy",
                            json!({"case":case.name,"mode":MODES[mode],"phase":phase,
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
                    for name in edit_graphs() {
                        let name = name.as_str();
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
                        "wall_ms":start.elapsed().as_secs_f64()*1e3,"results":results,"errors":errors,"load":load(),
                        "graph":s.engine().analysis_graph_stats()}),
                    );
                    // The same six graphs again on the unchanged snapshot.
                    let start = Instant::now();
                    let mut used = Vec::new();
                    for name in edit_graphs() {
                        let name = name.as_str();
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
    let s = open(Path::new(&path));
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
    let runs = env_usize("MKLIGHT_BENCH_RUNS", 10).max(min_runs(10));
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
        let mut sessions: Vec<_> = MODES.iter().map(|_| open(Path::new(&path))).collect();
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
                for name in edit_graphs() {
                    let case = all.iter().find(|c| c.name == name).unwrap();
                    let req = request(case, MODES[mode]).view_key(&name);
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
                        "load":load(),"graph":engine.analysis_graph_stats()
                    }),
                );
            }
        }
    }
}

/// Cancellation latency of explicit-Metal jobs: each case runs once warm,
/// then is cancelled at spread fractions of that wall time; the record is the
/// time from the cancel to the call's return (it must return `Cancelled`).
/// Select `cancel_latency`; MKLIGHT_BENCH_CANCEL_CASES (default the wave and
/// solver cases), MKLIGHT_BENCH_RUNS cancels per case.
#[test]
#[ignore = "local-project cancellation latency"]
fn project_bench_cancel_latency() {
    if std::env::var("MKLIGHT_BENCH_FILTER").as_deref() != Ok("cancel_latency") {
        return;
    }
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").expect("set MKLIGHT_BENCH_PROJECT");
    let runs = env_usize("MKLIGHT_BENCH_RUNS", 10).max(1);
    let s = open(Path::new(&path));
    let all = cases(&s);
    let names = std::env::var("MKLIGHT_BENCH_CANCEL_CASES")
        .unwrap_or_else(|_| "mtf,psf,glass_substitution,optimize,tolerance".into());
    for name in names.split(',') {
        let case = all.iter().find(|c| c.name == name).unwrap();
        let req = request(case, "df");
        let start = Instant::now();
        s.run_analysis(&req, &JobContext::new()).unwrap();
        let full = start.elapsed();
        for k in 0..runs {
            let ctx = JobContext::new();
            let at = full.mul_f64(0.1 + 0.7 * k as f64 / runs as f64);
            let cancel = ctx.cancel.clone();
            let canceller = std::thread::spawn(move || {
                std::thread::sleep(at);
                cancel.cancel();
                Instant::now()
            });
            let result = s.run_analysis(&req, &ctx);
            let done = Instant::now();
            let cancelled_at = canceller.join().unwrap();
            let latency_ms = done.saturating_duration_since(cancelled_at).as_secs_f64() * 1e3;
            emit(
                "cancel",
                json!({"case":name,"full_ms":full.as_secs_f64()*1e3,"cancel_after_ms":at.as_secs_f64()*1e3,
                "latency_ms":latency_ms,"cancelled":result.as_ref().is_err_and(|e| e.is_cancelled()),
                "finished_first":done < cancelled_at,"load":load()}),
            );
        }
    }
}
