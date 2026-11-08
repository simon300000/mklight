//! Stage times of `ghost_irradiance` outside the engine, on the CPU
//! reference and on Metal, alternated within each repetition, with the
//! engine's default settings (48² pupil grid).
//!
//! ```text
//! ghost_stages LENS.mklight [REPS] [DESIGNS] [SETTINGS_JSON] [KEY=VARIANT ...]
//! ```
//!
//! `DESIGNS` is a comma-separated list: `cpu` and `ghost` variants
//! (`default`, `level-sync`, `pipelined`, `shared`).

use std::path::Path;
use std::time::Instant;

use mklight_core::{JobContext, split};
use mklight_engine::{Engine, EngineConfig};
use mklight_ghost::{GhostModel, GhostSettings, IrradianceSettings};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let designs: Vec<String> = a
        .get(2)
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or_else(|| vec!["cpu".into(), "default".into()]);
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
    let extra: Vec<(String, String)> = a
        .iter()
        .skip(4)
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    let engine = Engine::new(EngineConfig::default());
    let s = engine.open_session(Path::new(&a[0])).unwrap();
    let rs = s.resolved().unwrap();
    let model = GhostModel::from_system(rs, GhostSettings::default()).unwrap();
    let metal = mklight_metal::MetalBackend::new().unwrap().unwrap();
    let ctx = JobContext::default();
    let mut reference = None;
    for rep in 0..=reps {
        for d in &designs {
            split::reset();
            for (k, v) in &extra {
                split::set(k, Some(v));
            }
            if d != "cpu" && d != "default" {
                split::set("ghost", Some(d));
            }
            let t = Instant::now();
            let r = if d == "cpu" {
                mklight_ghost::ghost_irradiance(&model, &settings, &ctx).unwrap()
            } else {
                mklight_ghost::gpu::ghost_irradiance(&metal, &model, &settings, &ctx).unwrap()
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let gpu = &r.run.settings["gpu"];
            let peak = r.total_ghost_map.peak();
            let diff = reference.as_ref().map_or(0., |p: &(f64, f64)| {
                ((r.total_flux_ratio - p.0) / p.0)
                    .abs()
                    .max(((peak - p.1) / p.1).abs())
            });
            if d == "cpu" && reference.is_none() {
                reference = Some((r.total_flux_ratio, peak));
            }
            println!(
                "rep {rep} {d:>10}: {ms:9.1} ms, {} ghosts, flux ratio {:.9e} (rel. diff {diff:.1e} \
                 vs cpu), backend {}, timings {} host {} probes {} reference {}",
                r.ghosts.len(),
                r.total_flux_ratio,
                r.run.backend,
                r.run.settings["timings"],
                gpu["host_seconds"],
                gpu["distinct_probes"],
                gpu["reference_probes"],
            );
        }
    }
}
