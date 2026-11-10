//! Stage times of `ghost_irradiance` outside the engine, on the CPU
//! reference and on Metal, alternated within each repetition, with the
//! engine's default settings (48² pupil grid).
//!
//! ```text
//! cargo run --release -p mklight-bench --example ghost_stages -- LENS.mklight \
//!     [REPS] [COLUMNS] [SETTINGS_JSON]
//! ```
//!
//! `COLUMNS` is a comma-separated list of `cpu` (the CPU reference) and
//! `metal`. Metal is the measured device: other targets build a stub.
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::path::Path;
use std::time::Instant;

use mklight_bench::host::cpu_ms;
use mklight_core::JobContext;
use mklight_engine::{Engine, EngineConfig};
use mklight_ghost::{GhostModel, GhostSettings, IrradianceSettings};

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ghost_stages measures Metal and runs on macOS only");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let columns: Vec<String> = a
        .get(2)
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or_else(|| vec!["cpu".into(), "metal".into()]);
    let mut settings = IrradianceSettings {
        pupil_samples: 48,
        ..Default::default()
    };
    if let Some(json) = a.get(3) {
        let patch: serde_json::Value = serde_json::from_str(json).unwrap();
        let mut v = serde_json::to_value(&settings).unwrap();
        for (k, x) in patch.as_object().unwrap() {
            v[k] = x.clone();
        }
        settings = serde_json::from_value(v).unwrap();
    }
    let engine = Engine::new(EngineConfig::default());
    let s = engine.open_session(Path::new(&a[0])).unwrap();
    let rs = s.resolved().unwrap();
    let model = GhostModel::from_system(rs, GhostSettings::default()).unwrap();
    let metal = mklight_metal::MetalBackend::new().unwrap().unwrap();
    let ctx = JobContext::default();
    let mut reference = None;
    for rep in 0..=reps {
        for column in &columns {
            let (t, c0) = (Instant::now(), cpu_ms());
            let r = if column == "cpu" {
                mklight_ghost::ghost_irradiance(&model, &settings, &ctx).unwrap()
            } else {
                mklight_ghost::gpu::ghost_irradiance(&metal, &model, &settings, &ctx).unwrap()
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let cores = (cpu_ms() - c0) / ms;
            let gpu = &r.run.settings["gpu"];
            let peak = r.total_ghost_map.peak();
            let ratio = r.total_flux_ratio.expect("the primary reaches the map");
            let diff = reference.as_ref().map_or(0., |p: &(f64, f64)| {
                ((ratio - p.0) / p.0).abs().max(((peak - p.1) / p.1).abs())
            });
            if column == "cpu" && reference.is_none() {
                reference = Some((ratio, peak));
            }
            println!(
                "rep {rep} {column:>10}: {ms:9.1} ms ({cores:.1} cores), {} ghosts, flux ratio {:.9e} \
                 (rel. diff {diff:.1e} vs cpu), backend {}\n    timings [wall, cpu] {}\n    host wall {}\n    host cpu {}\n    probes {} reference {}",
                r.ghosts.len(),
                ratio,
                r.run.backend,
                r.run.settings["timings"],
                gpu["host_seconds"],
                gpu["host_cpu_seconds"],
                gpu["distinct_probes"],
                gpu["reference_probes"],
            );
        }
    }
}
