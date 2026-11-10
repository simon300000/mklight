//! Stage profile of one analysis (stray light and ghosts): runs it on one
//! backend after a unique edit and prints wall time and the stage timings the
//! run records.
//!
//! ```text
//! stray_ghost_profile LENS.mklight ANALYSIS BACKEND [SETTINGS_JSON] [REPS] [KEY=VARIANT ...]
//! ```

use std::path::Path;
use std::time::Instant;

use mklight_core::{param::ParamPath, split};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend};
use serde_json::Value;

fn timings(v: &Value, path: &str, out: &mut Vec<(String, f64)>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let p = format!("{path}.{k}");
                if let Some(f) = x.as_f64()
                    && (k.ends_with("seconds") || k.ends_with("_ms") || k == "command_buffers")
                {
                    out.push((p, f));
                } else {
                    timings(x, &p, out);
                }
            }
        }
        Value::Array(a) if a.len() <= 64 => {
            for (i, x) in a.iter().enumerate() {
                timings(x, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let lens = &a[0];
    let analysis = &a[1];
    let backend = match a[2].as_str() {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        g => Backend::Gpu(g.into()),
    };
    let settings: Value = a
        .get(3)
        .map(|s| serde_json::from_str(s).unwrap())
        .unwrap_or(serde_json::json!({}));
    let reps: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);
    for kv in a.iter().skip(5) {
        let (k, v) = kv.split_once('=').unwrap();
        split::set(k, Some(v));
    }
    let engine = Engine::new(EngineConfig::default());
    let mut s = engine.open_session(Path::new(lens)).unwrap();
    let base = s.system().clone();
    for rep in 0..=reps {
        let mut system = base.clone();
        let surface = system
            .surfaces
            .iter()
            .skip(1)
            .find(|s| s.shape.curvature() != 0.)
            .unwrap();
        let p = ParamPath::Curvature {
            surface: surface.id,
        };
        let c = surface.shape.curvature();
        p.set(&mut system, c * (1. + (rep as f64 + 1.) * 1e-10))
            .unwrap();
        s.set_system(system).unwrap();
        let mut req = AnalysisRequest::new(analysis, settings.clone());
        req.backend = Some(backend.clone());
        let t = Instant::now();
        let id = s.submit_analysis(&req).unwrap();
        engine.wait_job(id, None).unwrap();
        let r = engine.take_result(id).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut t = Vec::new();
        timings(&r.run_info.settings, "", &mut t);
        println!(
            "{} rep {rep}: {ms:.1} ms wall, {:.1} ms reported, backend {}",
            if rep == 0 { "warm" } else { "timed" },
            r.run_info.duration_ms,
            r.run_info.backend
        );
        if rep == reps {
            for (k, v) in t {
                println!("  {k} = {v:.4}");
            }
            for w in &r.run_info.warnings {
                println!("  warning: {w}");
            }
            println!(
                "  backend report: {}",
                serde_json::to_string(&r.backend).unwrap()
            );
        }
    }
}
