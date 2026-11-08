//! Stage times of `stray_light` outside the engine: system build, primary
//! reference, CPU transport and the Metal transport designs, each timed
//! separately and alternated within each repetition.
//!
//! ```text
//! stray_stages LENS.mklight [RAYS] [REPS] [SETTINGS_JSON] [DESIGNS] [KEY=VARIANT ...]
//! ```
//!
//! `DESIGNS` is a comma-separated list of `stray` variants (`default`,
//! `gpu-tally`, `cpu-tally`, `gpu-head`, `cosplit`, `wavefront`; `cpu` for
//! the CPU reference transport). `share=X` and `head=N` set the transport
//! options; other `KEY=VARIANT` pairs select work-split variants.

use std::path::Path;
use std::time::Instant;

use mklight_core::{JobContext, split};
use mklight_engine::{Engine, EngineConfig};
use mklight_stray::{StrayAnalysisSettings, StraySystem, primary_reference};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rays: u64 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(20_000);
    let reps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let mut settings: StrayAnalysisSettings = a
        .get(3)
        .map(|s| serde_json::from_str(s).unwrap())
        .unwrap_or_default();
    settings.rays = rays;
    let designs: Vec<String> = a
        .get(4)
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or_else(|| vec!["cpu".into(), "default".into()]);
    let extra: Vec<(String, String)> = a
        .iter()
        .skip(5)
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    // `share=X` and `head=N` set TransportOptions; other pairs are variants.
    let mut options = mklight_stray::gpu::TransportOptions::default();
    let extra: Vec<(String, String)> = extra
        .into_iter()
        .filter(|(k, v)| match k.as_str() {
            "share" => {
                options.gpu_share = v.parse().unwrap();
                false
            }
            "head" => {
                options.head_events = v.parse().unwrap();
                false
            }
            _ => true,
        })
        .collect();
    let engine = Engine::new(EngineConfig::default());
    let s = engine.open_session(Path::new(&a[0])).unwrap();
    let rs = s.resolved().unwrap();
    let ctx = JobContext::default();
    let metal = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions::from_env())
        .unwrap()
        .unwrap();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let sys = StraySystem::build(rs.clone(), &settings.system_options(), &ctx).unwrap();
    let build = ms(t);
    let beam = settings.source.resolve(&sys).unwrap();
    let t = Instant::now();
    let reference = primary_reference(&sys, &beam.spec, 64, &ctx).unwrap();
    println!(
        "{} primitives, build {build:.1} ms, reference {:.1} ms (obstruction {:.2e})",
        sys.scene.primitives().len(),
        ms(t),
        reference.obstruction()
    );
    let ss = settings.stray_settings();
    let cpu = mklight_stray::run(&sys.scene, &beam.spec, &ss, &ctx).unwrap();
    let mut times = vec![Vec::new(); designs.len()];
    for rep in 0..=reps {
        for (k, d) in designs.iter().enumerate() {
            split::reset();
            for (key, v) in &extra {
                split::set(key, Some(v));
            }
            if d != "default" && d != "cpu" {
                split::set("stray", Some(d));
            }
            let t = Instant::now();
            let r = if d == "cpu" {
                mklight_stray::run(&sys.scene, &beam.spec, &ss, &ctx).unwrap()
            } else {
                mklight_stray::gpu::run(
                    &metal,
                    rs.snapshot.clone(),
                    &sys.scene,
                    &beam.spec,
                    &ss,
                    options,
                    &ctx,
                )
                .unwrap()
            };
            let elapsed = ms(t);
            if rep > 0 {
                times[k].push(elapsed);
            }
            let e = &r.energy;
            let z = (e.detected_owned - cpu.energy.detected_owned)
                / (e.detected_owned_std_error
                    .hypot(cpu.energy.detected_owned_std_error));
            let info = r.run_info.as_ref().map(|i| &i.settings);
            let detail = info
                .map(|i| format!("{} {} {}", i["design"], i["counters"], i["timings"]))
                .unwrap_or_default();
            println!(
                "rep {rep} {d:>10}: {elapsed:8.1} ms, {} events (cpu {}), owned {:.6e} ± {:.1e} \
                 (z {z:+.2} vs cpu), balance {:.1e}, backend {} {detail} {:?}",
                e.events,
                cpu.energy.events,
                e.detected_owned,
                e.detected_owned_std_error,
                e.relative_residual(),
                r.run_info.as_ref().map_or("?", |i| i.backend.as_str()),
                r.warnings,
            );
        }
    }
    split::reset();
    for (d, t) in designs.iter().zip(&mut times) {
        t.sort_by(f64::total_cmp);
        if !t.is_empty() {
            println!(
                "median {d:>10}: {:.1} ms over {} reps",
                t[t.len() / 2],
                t.len()
            );
        }
    }
}
