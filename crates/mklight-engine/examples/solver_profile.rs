//! Stage profile of one solver analysis (optimize, tolerance,
//! merit_evaluation, glass_substitution) on one backend and work-split
//! variant.
//!
//! ```text
//! cargo run --release -p mklight-engine --example solver_profile -- LENS.mklight ANALYSIS \
//!     [--backend cpu|metal|auto] [--variant NAME] [--settings JSON] [--reps N] [--json]
//! ```
//!
//! Every curved surface's curvature is a variable (as the benchmark
//! harness's `curvature_variables` setup). One untimed warm-up call precedes
//! `--reps` timed calls, each after a tiny unique edit. Prints the wall time,
//! the result's `backend` block summary and the resident plan stages
//! (wall/device seconds and command buffers per stage).

use std::path::Path;
use std::time::Instant;

use mklight_core::{param::ParamPath, split};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, Session, backend::Backend};
use serde_json::{Value, json};

fn curvature_variables(s: &mut Session) {
    let vars: Vec<Value> = s.system().surfaces[1..s.system().surfaces.len() - 1]
        .iter()
        .filter(|x| x.shape.curvature() != 0.)
        .map(|x| json!({"param": {"param": "curvature", "surface": x.id}, "lower": null, "upper": null}))
        .collect();
    s.set_project_section("variables", Value::Array(vars))
        .unwrap();
}

fn edit(s: &mut Session, base: &mklight_core::OpticalSystem, serial: u64) {
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
    param
        .set(&mut system, c * (1. + (1. + serial as f64 * 1e-3) * 1e-9))
        .unwrap();
    s.set_system(system).unwrap();
}

fn stages(v: &Value, out: &mut Vec<(String, f64, f64, u64)>) {
    if let Some(st) = v.get("stages").and_then(Value::as_array)
        && v.get("command_buffers").is_some()
    {
        for s in st {
            let name = s["name"].as_str().unwrap_or("").to_string();
            let wall = s["wall_seconds"].as_f64().unwrap_or(0.);
            let dev = s["device_seconds"].as_f64().unwrap_or(0.);
            let cb = s["command_buffers"].as_u64().unwrap_or(0);
            if let Some(e) = out.iter_mut().find(|e| e.0 == name) {
                e.1 += wall;
                e.2 += dev;
                e.3 += cb;
            } else {
                out.push((name, wall, dev, cb));
            }
        }
        return;
    }
    match v {
        Value::Array(a) => a.iter().for_each(|x| stages(x, out)),
        Value::Object(o) => o.values().for_each(|x| stages(x, out)),
        _ => {}
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let lens = args.next().expect("LENS.mklight");
    let analysis = args.next().expect("ANALYSIS");
    let mut backend = "metal".to_string();
    let mut variant: Option<String> = None;
    let mut settings = json!({});
    let mut reps = 1usize;
    let mut dump = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next().unwrap(),
            "--variant" => variant = args.next(),
            "--settings" => settings = serde_json::from_str(&args.next().unwrap()).unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--json" => dump = true,
            _ => panic!("unexpected argument {a}"),
        }
    }
    split::set("solvers", variant.as_deref());
    let engine = Engine::new(EngineConfig::default());
    let mut s = engine.open_session(Path::new(&lens)).unwrap();
    curvature_variables(&mut s);
    let base = s.system().clone();
    let b = match backend.as_str() {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        g => Backend::Gpu(g.into()),
    };
    for rep in 0..=reps {
        edit(&mut s, &base, rep as u64 + 1);
        let mut req = AnalysisRequest::new(&analysis, settings.clone());
        req.backend = Some(b.clone());
        let t = Instant::now();
        let id = s.submit_analysis(&req).unwrap();
        engine.wait_job(id, None).unwrap();
        let r = engine.take_result(id).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let j = &r.json;
        let mut st = Vec::new();
        stages(
            &serde_json::to_value(&r.backend.execution).unwrap(),
            &mut st,
        );
        println!(
            "{} {analysis} {backend}/{} {:.1} ms; used {}; iterations {} evaluations {} merit {}",
            if rep == 0 { "warm" } else { "timed" },
            variant.as_deref().unwrap_or("default"),
            ms,
            r.backend.used,
            j["iterations"],
            j["evaluations"],
            j["final"]["merit"]
                .as_f64()
                .or(j["merit"].as_f64())
                .map_or("-".into(), |m| format!("{m:.6e}")),
        );
        if analysis == "optimize" || analysis == "glass_substitution" {
            println!(
                "  status {} termination {} improved {} start {} search {} warnings {:?}",
                j["status"],
                j["termination"],
                j["improved"],
                j["start"],
                j["search"],
                r.run_info.warnings
            );
        }
        if let Some(b) = j.get("backend") {
            println!(
                "  gpu_stats {}  resident {}",
                b["gpu_stats"], b["resident_evaluation"]
            );
            if let Some(v) = b.get("split") {
                println!("  split {v}");
            }
        }
        for (name, wall, dev, cb) in &st {
            println!(
                "  stage {name:<40} wall {:>9.1} ms  device {:>9.1} ms  commands {cb}",
                wall * 1e3,
                dev * 1e3
            );
        }
        if dump && rep == reps {
            println!(
                "{}",
                serde_json::to_string_pretty(&r.json["backend"]).unwrap()
            );
        }
    }
}
