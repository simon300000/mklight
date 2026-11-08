//! Stage profile of one analysis on one lens: wall time per call after an
//! edit, and the backend's execution evidence (resident plan stages).
//!
//! ```text
//! cargo run --release -p mklight-engine --example fieldmaps_profile -- LENS.mklight \
//!     ANALYSIS '{"settings": "json"}' [--backend cpu|metal] [--variant KEY=NAME] \
//!     [--reps N] [--evidence]
//! ```
//!
//! The first call is untimed (shader compilation, host caches).

use std::path::Path;
use std::time::Instant;

use mklight_core::{OpticalSystem, param::ParamPath, split};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, Session, backend::Backend};
use serde_json::Value;

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
    let sign = if serial.is_multiple_of(2) { 1. } else { -1. };
    param
        .set(
            &mut system,
            c * (1. + sign * (1. + serial as f64 * 1e-3) * 1e-9),
        )
        .unwrap();
    s.set_system(system).unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let lens = args.next().expect("lens");
    let analysis = args.next().expect("analysis");
    let settings: Value = serde_json::from_str(&args.next().unwrap_or("{}".into())).unwrap();
    let mut backend = "metal".to_string();
    let mut reps = 3;
    let mut evidence = false;
    let mut variants: Vec<(String, String)> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next().unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--evidence" => evidence = true,
            "--variant" => {
                let v = args.next().unwrap();
                let (k, n) = v.split_once('=').unwrap();
                variants.push((k.into(), n.into()));
            }
            _ => panic!("unexpected {a}"),
        }
    }
    for (k, v) in &variants {
        split::set(k, Some(v));
    }
    let engine = Engine::new(EngineConfig::default());
    let mut s = engine.open_session(Path::new(&lens)).unwrap();
    let base = s.system().clone();
    let mut serial = (std::process::id() as u64) << 20;
    let mut times = Vec::new();
    for rep in 0..=reps {
        serial += 1;
        edit(&mut s, &base, serial);
        let mut req = AnalysisRequest::new(&analysis, settings.clone());
        req.backend = Some(match backend.as_str() {
            "cpu" => Backend::CpuReference,
            "auto" => Backend::Auto,
            g => Backend::Gpu(g.into()),
        });
        let t = Instant::now();
        let id = s.submit_analysis(&req).unwrap();
        engine.wait_job(id, None).unwrap();
        let r = engine.take_result(id).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if rep > 0 {
            times.push(ms);
        }
        println!(
            "rep {rep}: {ms:.1} ms (reported {:.1}) backend {}",
            r.run_info.duration_ms, r.run_info.backend
        );
        if evidence && rep == reps {
            let mut settings = r.run_info.settings.clone();
            strip(&mut settings);
            println!("{}", serde_json::to_string_pretty(&settings).unwrap());
            println!(
                "{}",
                serde_json::to_string_pretty(&r.backend).unwrap_or_default()
            );
            for w in r.run_info.warnings.iter().take(5) {
                println!("warning: {w}");
            }
        }
    }
    times.sort_by(f64::total_cmp);
    println!("median {:.1} ms", times[times.len() / 2]);
}

/// Drops bulky arrays from the evidence.
fn strip(v: &mut Value) {
    match v {
        Value::Array(a) if a.len() > 24 => *v = Value::String(format!("[{} items]", a.len())),
        Value::Array(a) => a.iter_mut().for_each(strip),
        Value::Object(m) => m.values_mut().for_each(strip),
        _ => {}
    }
}
