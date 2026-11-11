//! Calibration of the `auto` cost model (`analysis::cost`, GPU.md §9.2) and
//! validation of its routes. Real Metal, release build, one command:
//!
//! MKLIGHT_COST_CALIBRATION_OUT=$PWD/crates/mklight-engine/src/analysis/cost/apple-m2-pro.json \
//! cargo test --release -p mklight-bench --test tests -- --ignored \
//! --exact cost_calibration::calibrate --nocapture --test-threads=1
//!
//! It times every calibrated analysis on nine designs of different shape
//! (samples, plus an optional local project, `MKLIGHT_BENCH_PROJECT`;
//! nothing of it is embedded) at a ladder of workloads: the CPU reference
//! with cold intermediates (graph off) and Metal, each repeated on one
//! snapshot and after an edit, interleaved so host load hits all four
//! alike (load average and device time are recorded with every
//! rung; online refinement absorbs the host's current contention). It then
//! fits the stage coefficients, each mode with the features the model
//! applies its coefficients to (`cost::Mode::features`), and writes the
//! table (without `_OUT`, to the target tmp dir). Options:
//! `MKLIGHT_COST_REPS` (default 2), `MKLIGHT_COST_FILTER` (analyses),
//! `MKLIGHT_COST_MODES` (of `cpu`, `cpu_edit`, `metal`, `metal_edit`; a run
//! restricted by either fits a partial table and refuses `_OUT`),
//! `MKLIGHT_COST_DESIGNS` (sample names and `local`; `all` for the 16),
//! `MKLIGHT_COST_RECORDS` (measured records, JSON lines; written while
//! measuring), `MKLIGHT_COST_REFIT=1` (fit those records without measuring;
//! several comma-separated files may be given: a rung measured twice keeps
//! its least loaded record, and load discounts every record's weight) with
//! `MKLIGHT_COST_DEVICE_OPEN_MS` (the measured device-open time).
//!
//! `validate` compares `auto` with both backends (default engines, graph on)
//! on fresh edits and warm repeats and prints the routing table and regret
//! (the same command with `--exact validate`). `../README.md` lists every
//! harness with its command and variables.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use mklight_bench::{edit, host, solver};
use mklight_core::{JobContext, OpticalSystem, samples};
use mklight_engine::analysis::cost::{self, Calibration, Features, Mode, SystemShape};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, Session, backend::Backend, project::Project,
};
use serde_json::{Value, json};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn wanted(list: &str, name: &str) -> bool {
    env(list).is_none_or(|s| s.split(',').any(|n| n == name))
}

/// Designs calibrated by default: refracting, reflecting, finite,
/// non-rotational, freeform and diffractive shapes, and the local project.
const DESIGNS: &str = "singlet,cooke_triplet,double_gauss,petzval_asphere,newtonian,relay,anamorphic,zernike_sag,hybrid_doe,local";

/// Measured modes (`MKLIGHT_COST_MODES`, default all four): the CPU
/// reference and Metal, each on a repeated and on a freshly edited snapshot.
fn mode_wanted(mode: &str) -> bool {
    wanted("MKLIGHT_COST_MODES", mode)
}

fn design_wanted(name: &str) -> bool {
    match env("MKLIGHT_COST_DESIGNS").as_deref() {
        Some("all") => true,
        Some(list) => list.split(',').any(|n| n == name),
        None => DESIGNS.split(',').any(|n| n == name),
    }
}

/// Calibrated analyses: every analysis with a GPU path.
fn analyses() -> Vec<&'static str> {
    mklight_engine::analysis::gpu::GPU_ANALYSES
        .iter()
        .copied()
        .filter(|a| wanted("MKLIGHT_COST_FILTER", a))
        .collect()
}

/// Sessions of the designs in `engine`: the samples and the local project.
fn designs(engine: &Engine) -> Vec<(String, Session)> {
    let mut out: Vec<(String, Session)> = samples::all()
        .into_iter()
        .filter(|(n, _)| design_wanted(n))
        .map(|(n, s)| {
            (
                n.to_string(),
                engine.session_from_project(Project::new(s)).unwrap(),
            )
        })
        .collect();
    if let Some(p) = env("MKLIGHT_BENCH_PROJECT")
        && design_wanted("local")
    {
        out.push((
            "local".into(),
            engine.open_session(std::path::Path::new(&p)).unwrap(),
        ));
    }
    out
}

/// Settings ladder of one analysis: default and larger workloads.
fn ladder(analysis: &str, system: &OpticalSystem) -> Vec<Value> {
    let one = |k: &str, vs: &[Value]| vs.iter().map(|v| json!({k: v})).collect::<Vec<_>>();
    let wave = || {
        vec![
            Value::Null,
            json!({"pupil":{"rays":129}}),
            json!({"pupil":{"grid":512}}),
        ]
    };
    match analysis {
        "spot_diagram" | "ray_grid" => one("rings", &[json!(6), json!(24), json!(96)]),
        "best_focus" | "footprint" => one("rings", &[json!(8), json!(32), json!(96)]),
        "geometric_mtf" => one("rings", &[json!(32), json!(64), json!(160)]),
        "ray_fan" | "opd_fan" => one("samples", &[json!(41), json!(401), json!(4001)]),
        "layout" => one("rays_per_field", &[json!(7), json!(101), json!(1001)]),
        "field_curvature_distortion" | "lateral_colour" => {
            one("field_samples", &[json!(21), json!(201), json!(2001)])
        }
        "longitudinal_aberration" => one("pupil_samples", &[json!(21), json!(201), json!(2001)]),
        "grid_distortion" => one("grid_size", &[json!(11), json!(51), json!(201)]),
        "relative_illumination" => one("field_samples", &[json!(11), json!(51), json!(201)]),
        "pupil_aberration" | "relative_illumination_map" | "chief_ray_angle" => {
            one("grid", &[json!([5, 5]), json!([15, 15]), json!([41, 41])])
        }
        "system_transmission" => vec![
            Value::Null,
            json!({"pupil":{"type":"grid","n":96}}),
            json!({"pupil":{"type":"grid","n":256}}),
        ],
        "psf"
        | "mtf"
        | "line_spread"
        | "diffraction_encircled_energy"
        | "through_focus_mtf"
        | "mtf_vs_field" => wave(),
        "image_simulation" => vec![Value::Null, json!({"optics":{"field_grid":5}})],
        "ghost_irradiance" => one("pupil_samples", &[json!(48), json!(96)]),
        // Only runs whose transport gives the device sources are routed
        // by cost: from 163,840 rays on, on designs whose sags the device
        // intersects in closed form. Every other run is the CPU reference
        // by rule, on either backend.
        "stray_light" => [200_000u64, 400_000, 800_000]
            .into_iter()
            .filter(|&rays| mklight_stray::gpu::hybrid::planned_share(system, rays, None) > 0.)
            .map(|rays| json!({"rays": rays}))
            .collect(),
        "merit_evaluation" => vec![Value::Null],
        // The project-bench solver jobs (`solver`), `optimize` also with
        // more iterations.
        "optimize" => {
            let vars = solver::variables(system);
            [3, 10]
                .iter()
                .map(|&n| solver::optimize(&vars, n))
                .collect()
        }
        // Timed without recorded trials, unlike `solver::tolerance`.
        "tolerance" => [16, 64]
            .iter()
            .map(|n| json!({"trials":n,"seed":71}))
            .collect(),
        "glass_substitution" => vec![solver::glass_substitution(
            &solver::variables(system),
            &solver::glass(system),
        )],
        _ => vec![Value::Null],
    }
}

/// One fitted call: stage features, measured ms and weight.
type Point = (Features, f64, f64);

struct Call {
    grid: Option<f64>,
    ms: f64,
    device_ms: f64,
    used: String,
    decision: Value,
}

fn call(s: &Session, analysis: &str, settings: &Value, backend: Backend) -> Option<Call> {
    let mut req = AnalysisRequest::new(analysis, settings.clone());
    req.backend = Some(backend);
    req.allow_backend_fallback = true;
    let device = mklight_metal::device_seconds();
    let start = Instant::now();
    let r = s.run_analysis(&req, &JobContext::new());
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let device_ms = (mklight_metal::device_seconds() - device) * 1e3;
    match r {
        Ok(r) => Some(Call {
            grid: cost::decided_grid(&r.json),
            ms,
            device_ms,
            used: r.backend.used.clone(),
            decision: r.backend.cost_decision.clone().unwrap_or(Value::Null),
        }),
        Err(e) => {
            eprintln!("{analysis} {settings}: {e}");
            None
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => 0.5 * (v[n / 2 - 1] + v[n / 2]),
    }
}

fn emit(kind: &str, v: Value) {
    println!("{kind} {v}");
}

fn metal_engine(graph: usize) -> Engine {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        analysis_graph_bytes: graph,
        ..Default::default()
    });
    mklight_bench::require_metal(&engine);
    engine
}

fn records_path() -> String {
    env("MKLIGHT_COST_RECORDS").unwrap_or_else(|| {
        format!(
            "{}/cost-calibration-records.jsonl",
            env!("CARGO_TARGET_TMPDIR")
        )
    })
}

/// Measure every design × analysis × ladder rung; returns the records.
fn measure(device_open_ms: &mut f64) -> Vec<Value> {
    let reps = env("MKLIGHT_COST_REPS").map_or(2, |s| s.parse().unwrap());
    let open = Instant::now();
    let engine = metal_engine(0);
    *device_open_ms = open.elapsed().as_secs_f64() * 1e3;
    let mut file = std::fs::File::create(records_path()).unwrap();
    let mut records = Vec::new();
    let mut first_native: BTreeMap<String, f64> = BTreeMap::new();
    let mut serial = 0usize;
    for (design, mut session) in designs(&engine) {
        let base = session.system().clone();
        for analysis in analyses() {
            for settings in ladder(analysis, &base) {
                session.set_system(base.clone()).unwrap();
                let mut ms: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
                let mut dev: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
                let mut grid = None;
                let mut used: BTreeMap<&str, String> = BTreeMap::new();
                let mut first = None;
                let mut slow = false;
                for rep in 0..=reps {
                    let mut order = Mode::ALL.map(Mode::name);
                    if rep % 2 == 1 {
                        order.reverse();
                    }
                    for mode in order.into_iter().filter(|m| mode_wanted(m)) {
                        let backend = if mode.starts_with("cpu") {
                            Backend::CpuReference
                        } else {
                            Backend::Gpu("metal".into())
                        };
                        if mode.ends_with("_edit") {
                            serial += 1;
                            // A distinct tiny edit (new content hash every
                            // call); a design without a curved surface
                            // stays as it is.
                            edit::radius_ppm(&mut session, &base, serial);
                        } else {
                            session.set_system(base.clone()).unwrap();
                        }
                        let Some(c) = call(&session, analysis, &settings, backend) else {
                            continue;
                        };
                        if !mode.starts_with("cpu") && !first_native.contains_key(analysis) {
                            first_native.insert(analysis.into(), c.ms);
                            first = Some(c.ms);
                        }
                        // Rep 0 warms the resident snapshot and the process.
                        if rep > 0 {
                            ms.entry(mode).or_default().push(c.ms);
                            dev.entry(mode).or_default().push(c.device_ms);
                        }
                        slow |= c.ms > 8000.;
                        grid = grid.or(c.grid);
                        used.insert(mode, c.used);
                    }
                    if slow && rep >= 1 {
                        break;
                    }
                }
                let normalised = mklight_engine::analysis::prepare(
                    &*engine.analysis(analysis).unwrap(),
                    &settings,
                )
                .unwrap();
                // This rung's CPU time (0 when its CPU calls failed).
                let mut cpu = 0.;
                for (mode, mut v) in ms {
                    let device_ms = median(dev.get_mut(mode).unwrap());
                    let median_ms = median(&mut v);
                    if mode == "cpu" {
                        cpu = median_ms;
                    }
                    let r = json!({"design":design,"analysis":analysis,"settings":settings,
                        "normalised":normalised,"mode":mode,"ms":median_ms,"runs":v,
                        "device_ms":device_ms,"grid":grid,
                        "used":used.get(mode),"first_ms":first,"load":host::load()});
                    emit("COST_CAL", r.clone());
                    writeln!(file, "{r}").unwrap();
                    records.push(r);
                }
                // Larger rungs of a slow analysis add little evidence.
                if cpu > 4000. || slow {
                    break;
                }
            }
        }
    }
    records
}

/// Fit the table from records (systems from the same designs).
fn fit(records: &[Value], device_open_ms: f64) -> Calibration {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    let systems: BTreeMap<String, OpticalSystem> = designs(&engine)
        .into_iter()
        .map(|(n, s)| (n, s.system().clone()))
        .collect();
    let mut points: BTreeMap<(String, Mode), Vec<Point>> = BTreeMap::new();
    let mut labels: BTreeMap<(String, Mode), Vec<Value>> = BTreeMap::new();
    let mut first: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    // A rung measured more than once (a re-measurement on a quieter host)
    // keeps its least loaded record; load discounts the rest.
    let cores = std::thread::available_parallelism().map_or(12, |n| n.get()) as f64;
    let weight = |r: &Value| {
        let x = r["load"].as_f64().unwrap_or(0.) / cores;
        1. / (x / 2.).max(1.).powi(2)
    };
    let mut best: BTreeMap<String, &Value> = BTreeMap::new();
    for r in records {
        let key = format!(
            "{}|{}|{}|{}",
            r["design"], r["analysis"], r["settings"], r["mode"]
        );
        let load = |r: &Value| r["load"].as_f64().unwrap_or(f64::INFINITY);
        if best.get(&key).is_none_or(|b| load(r) < load(b)) {
            best.insert(key, r);
        }
    }
    let records: Vec<&Value> = best.into_values().collect();
    for r in records.iter().copied() {
        let (Some(design), Some(analysis), Some(mode), Some(ms)) = (
            r["design"].as_str(),
            r["analysis"].as_str(),
            r["mode"].as_str().and_then(Mode::parse),
            r["ms"].as_f64(),
        ) else {
            continue;
        };
        let Some(system) = systems.get(design) else {
            continue;
        };
        // Only calls that ran where they were asked to.
        let native = r["used"].as_str().is_some_and(|u| u.starts_with("gpu:"));
        if native != matches!(mode, Mode::Metal | Mode::MetalEdit) || !ms.is_finite() {
            continue;
        }
        let Some(p) =
            cost::profile::profile_with(analysis, &r["normalised"], system, r["grid"].as_f64())
        else {
            continue;
        };
        // The features this mode's coefficients are applied to.
        let features = mode.features(p.features(&SystemShape::of(system)));
        points
            .entry((analysis.into(), mode))
            .or_default()
            .push((features, ms, weight(r)));
        labels
            .entry((analysis.into(), mode))
            .or_default()
            .push(json!({"design":design,"settings":r["settings"],"load":r["load"]}));
        if mode == Mode::MetalEdit
            && let Some(f) = r["first_ms"].as_f64()
        {
            first.entry(analysis.into()).or_default().push(f - ms);
        }
    }
    let mut analyses_out = BTreeMap::new();
    for analysis in analyses() {
        let get = |mode: Mode| {
            points
                .get(&(analysis.to_string(), mode))
                .filter(|p| !p.is_empty())
                .map(|p| cost::model::fit_weighted(p))
        };
        let c = cost::AnalysisCalibration {
            cpu: get(Mode::Cpu),
            cpu_edit: get(Mode::CpuEdit),
            metal: get(Mode::Metal),
            metal_edit: get(Mode::MetalEdit),
            first_use_ms: first
                .get(analysis)
                .map_or(0., |v| v.iter().copied().fold(0., f64::max).max(0.)),
        };
        emit("COST_FIT", json!({"analysis":analysis,"fit":c}));
        // Residuals, for inspecting the fit.
        for mode in Mode::ALL {
            let (Some(fit), Some(p), Some(l)) = (
                c.fit(mode),
                points.get(&(analysis.to_string(), mode)),
                labels.get(&(analysis.to_string(), mode)),
            ) else {
                continue;
            };
            for ((f, ms, _), l) in p.iter().zip(l) {
                let predicted: f64 = fit.stages(f).iter().sum();
                emit(
                    "COST_RESID",
                    json!({"analysis":analysis,"mode":mode.name(),"at":l,"ms":ms,"predicted":predicted,"features":f}),
                );
            }
        }
        analyses_out.insert(analysis.to_string(), c);
    }
    let sysctl = |args: &[&str]| {
        std::process::Command::new("sysctl")
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let device = engine
        .backends()
        .into_iter()
        .find(|b| b.id == "gpu:metal")
        .and_then(|b| b.capabilities.device_name)
        .unwrap_or_default();
    let loads: Vec<f64> = records.iter().filter_map(|r| r["load"].as_f64()).collect();
    Calibration {
        format: cost::model::FORMAT,
        model_version: cost::VERSION,
        device_class: device,
        measured: json!({
            "cpu": sysctl(&["-n", "machdep.cpu.brand_string"]),
            "records": records.len(),
            "designs": systems.keys().map(|k| if k == "local" { "local project" } else { k.as_str() }).collect::<Vec<_>>(),
            "load1": {"min": loads.iter().copied().fold(f64::INFINITY, f64::min),
                      "max": loads.iter().copied().fold(0., f64::max)},
            "command": "cargo test --release -p mklight-bench --test tests -- --ignored --exact cost_calibration::calibrate",
        }),
        cpu_threads: rayon::current_num_threads(),
        device_open_ms,
        analyses: analyses_out,
    }
}

#[test]
#[ignore = "release cost-model calibration; real Metal required"]
fn calibrate() {
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    // A restricted run fits a partial table, and an analysis or mode missing
    // from the table routes without calibration: it never replaces a named
    // table.
    let restricted = env("MKLIGHT_COST_FILTER").is_some()
        || (env("MKLIGHT_COST_REFIT").is_none() && env("MKLIGHT_COST_MODES").is_some());
    assert!(
        !(restricted && env("MKLIGHT_COST_CALIBRATION_OUT").is_some()),
        "MKLIGHT_COST_FILTER and MKLIGHT_COST_MODES fit a partial table: unset \
         MKLIGHT_COST_CALIBRATION_OUT (the table then goes to the target tmp dir)"
    );
    let mut device_open_ms = 40.;
    let records: Vec<Value> = if env("MKLIGHT_COST_REFIT").is_some() {
        // One or more record files (comma-separated), e.g. a re-measurement
        // of some designs on a quieter host next to the full run.
        records_path()
            .split(',')
            .flat_map(|p| {
                std::fs::read_to_string(p)
                    .unwrap()
                    .lines()
                    .filter_map(|l| serde_json::from_str(l).ok())
                    .collect::<Vec<Value>>()
            })
            .collect()
    } else {
        measure(&mut device_open_ms)
    };
    if env("MKLIGHT_COST_REFIT").is_some() {
        device_open_ms =
            env("MKLIGHT_COST_DEVICE_OPEN_MS").map_or(device_open_ms, |s| s.parse().unwrap());
    }
    let table = fit(&records, device_open_ms);
    let out = env("MKLIGHT_COST_CALIBRATION_OUT")
        .unwrap_or_else(|| format!("{}/cost-calibration.json", env!("CARGO_TARGET_TMPDIR")));
    let mut text = serde_json::to_string_pretty(&table).unwrap();
    text.push('\n');
    std::fs::write(&out, text).unwrap();
    emit(
        "COST_TABLE",
        json!({"path":out,"device_open_ms":device_open_ms}),
    );
}

/// `auto` against both backends: routing table and regret.
#[test]
#[ignore = "release cost-model validation; real Metal required"]
fn validate() {
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    let reps = env("MKLIGHT_COST_REPS").map_or(3, |s| s.parse().unwrap());
    let modes = [
        ("cpu", Backend::CpuReference),
        ("metal", Backend::Gpu("metal".into())),
        ("auto", Backend::Auto),
    ];
    let engines: Vec<Engine> = modes.iter().map(|_| metal_engine(256 << 20)).collect();
    let mut sessions: Vec<Vec<(String, Session)>> = engines.iter().map(designs).collect();
    let mut serial = 0usize;
    for d in 0..sessions[0].len() {
        let base = sessions[0][d].1.system().clone();
        let design = sessions[0][d].0.clone();
        for analysis in analyses() {
            let rungs = ladder(analysis, &base);
            for settings in rungs.iter().take(2) {
                for phase in ["edit", "warm"] {
                    let mut ms: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
                    let mut used: Vec<String> = vec![String::new(); modes.len()];
                    let mut decision = Value::Null;
                    let mut slow = false;
                    for rep in 0..=reps {
                        if phase == "edit" {
                            serial += 1;
                        }
                        for k in 0..modes.len() {
                            let m = (rep + k) % modes.len();
                            let s = &mut sessions[m][d].1;
                            if phase == "edit" {
                                edit::radius_ppm(s, &base, serial);
                            }
                            let Some(c) = call(s, analysis, settings, modes[m].1.clone()) else {
                                continue;
                            };
                            // Warm: rep 0 is the first call on the snapshot.
                            if phase == "edit" || rep > 0 {
                                ms[m].push(c.ms);
                            }
                            slow |= c.ms > 8000.;
                            used[m] = c.used;
                            if m == 2 {
                                decision = c.decision;
                            }
                        }
                        if slow && rep >= 1 {
                            break;
                        }
                    }
                    let med: Vec<f64> = ms.iter_mut().map(|v| median(v)).collect();
                    let best = med[0].min(med[1]);
                    emit(
                        "COST_VAL",
                        json!({"design":design,"analysis":analysis,"settings":settings,"phase":phase,
                            "cpu_ms":med[0],"metal_ms":med[1],"auto_ms":med[2],
                            "metal_used":used[1],"auto_used":used[2],
                            "regret":med[2] / best.max(1e-3),"excess_ms":med[2] - best,
                            "decision":decision,"load":host::load()}),
                    );
                }
            }
        }
    }
    for e in &engines {
        emit("COST_MODEL", e.cost_model_stats());
    }
}
