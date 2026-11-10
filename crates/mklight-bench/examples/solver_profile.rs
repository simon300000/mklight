//! Stage profile of one solver analysis (optimize, tolerance,
//! merit_evaluation, glass_substitution) on one backend, optionally with
//! route options (`--route solver_aiming=device`;
//! `mklight_engine::RouteOptions`).
//!
//! ```text
//! cargo run --release -p mklight-bench --example solver_profile -- LENS.mklight ANALYSIS \
//!     [--backend cpu|metal|auto] [--route KEY=VALUE]... [--settings JSON] [--reps N] [--json]
//!     [--out FILE]
//! ```
//!
//! Every curved surface's curvature is a variable (as `backend_bench`'s
//! `curvature_variables` setup). One untimed warm-up call precedes
//! `--reps` timed calls, each after a tiny unique edit. Prints the wall time,
//! the result's `backend` block summary and the resident plan stages
//! (wall/device seconds and command buffers per stage). `--out` writes the
//! last call's result JSON (to compare outcomes between builds).

use std::path::Path;
use std::time::Instant;

use mklight_bench::{edit, solver};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, RouteOptions, backend::Backend};
use serde_json::{Value, json};

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
    let mut selections: Vec<String> = Vec::new();
    let mut settings = json!({});
    let mut reps = 1usize;
    let mut dump = false;
    let mut out: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next().unwrap(),
            "--route" => selections.push(args.next().unwrap()),
            "--settings" => settings = serde_json::from_str(&args.next().unwrap()).unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--json" => dump = true,
            "--out" => out = args.next(),
            _ => panic!("unexpected argument {a}"),
        }
    }
    let route = RouteOptions::from_pairs(
        selections
            .iter()
            .map(|s| s.split_once('=').expect("--route KEY=VALUE")),
    )
    .unwrap();
    let engine = Engine::new(EngineConfig::default());
    let mut s = engine.open_session(Path::new(&lens)).unwrap();
    solver::set_curvature_variables(&mut s);
    let base = s.system().clone();
    let b = match backend.as_str() {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        g => Backend::Gpu(g.into()),
    };
    for rep in 0..=reps {
        assert!(
            edit::curvature_ppb_growing(&mut s, &base, rep as u64 + 1),
            "a curved surface"
        );
        let mut req = AnalysisRequest::new(&analysis, settings.clone());
        req.backend = Some(b.clone());
        req.route = Some(route);
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
            if selections.is_empty() {
                "default".to_string()
            } else {
                selections.join(",")
            },
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
            if let Some(v) = b.get("aiming") {
                println!("  aiming {v}");
            }
        }
        for (name, wall, dev, cb) in &st {
            println!(
                "  stage {name:<40} wall {:>9.1} ms  device {:>9.1} ms  commands {cb}",
                wall * 1e3,
                dev * 1e3
            );
        }
        if let (Some(path), true) = (&out, rep == reps) {
            std::fs::write(path, serde_json::to_string_pretty(&r.json).unwrap()).unwrap();
        }
        if dump && rep == reps {
            println!(
                "{}",
                serde_json::to_string_pretty(&r.json["backend"]).unwrap()
            );
        }
    }
}
