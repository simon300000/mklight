//! Stage times of `ghost_irradiance` outside the engine, on the CPU
//! reference and on Metal, alternated within each repetition, with the
//! engine's default settings (48² pupil grid).
//!
//! ```text
//! ghost_stages LENS.mklight [REPS] [DESIGNS] [SETTINGS_JSON] [KEY=VARIANT ...]
//! ```
//!
//! `DESIGNS` is a comma-separated list: `cpu` and `ghost` variants
//! (`default`, `level-sync`, `pipelined`, `shared`), each optionally with
//! `+KEY=VARIANT` selections of its own.

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
            // A column `DESIGN+KEY=VARIANT+...` adds its own selections.
            let mut parts = d.split('+');
            let design = parts.next().unwrap();
            for kv in parts {
                let (k, v) = kv.split_once('=').unwrap();
                split::set(k, Some(v));
            }
            if design != "cpu" && design != "default" {
                split::set("ghost", Some(design));
            }
            let (t, c0) = (Instant::now(), cpu_ms());
            let r = if design == "cpu" {
                mklight_ghost::ghost_irradiance(&model, &settings, &ctx).unwrap()
            } else {
                mklight_ghost::gpu::ghost_irradiance(&metal, &model, &settings, &ctx).unwrap()
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let cores = (cpu_ms() - c0) / ms;
            let gpu = &r.run.settings["gpu"];
            let peak = r.total_ghost_map.peak();
            let diff = reference.as_ref().map_or(0., |p: &(f64, f64)| {
                ((r.total_flux_ratio - p.0) / p.0)
                    .abs()
                    .max(((peak - p.1) / p.1).abs())
            });
            if design == "cpu" && reference.is_none() {
                reference = Some((r.total_flux_ratio, peak));
            }
            println!(
                "rep {rep} {d:>10}: {ms:9.1} ms ({cores:.1} cores), {} ghosts, flux ratio {:.9e} \
                 (rel. diff {diff:.1e} vs cpu), backend {}\n    timings [wall, cpu] {}\n    host wall {}\n    host cpu {}\n    probes {} reference {}",
                r.ghosts.len(),
                r.total_flux_ratio,
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

/// Process CPU time (user + system) in ms.
fn cpu_ms() -> f64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes one rusage record.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) };
    // SAFETY: zero-initialised and written by getrusage.
    let u = unsafe { u.assume_init() };
    let t = |v: libc::timeval| v.tv_sec as f64 * 1e3 + v.tv_usec as f64 * 1e-3;
    t(u.ru_utime) + t(u.ru_stime)
}
