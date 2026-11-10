//! Stage times of `stray_light` outside the engine: system build, primary
//! reference, CPU transport and the Metal transport, each timed separately
//! and alternated within each repetition.
//!
//! ```text
//! cargo run --release -p mklight-bench --example stray_stages -- LENS.mklight \
//!     [RAYS] [REPS] [SETTINGS_JSON] [COLUMNS] [KEY=VALUE ...]
//! ```
//!
//! `COLUMNS` is a comma-separated list of transports: `cpu` (the CPU
//! reference) or `device`, a device column optionally with `+KEY=VALUE`
//! transport options of its own. The options are `share=X` (the device's
//! share of the source chunks; without it the rule of the scene), `head=N`
//! (events per source on the device) and `sag=f32-scan|f32-grazing`
//! (`mklight_stray::gpu::TransportOptions`); trailing `KEY=VALUE` pairs set
//! them for every device column. Metal is the measured device: other
//! targets build a stub.
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::path::Path;
use std::time::Instant;

use mklight_bench::host::cpu_ms;
use mklight_core::JobContext;
use mklight_engine::{Engine, EngineConfig};
use mklight_stray::gpu::{SagMode, TransportOptions};
use mklight_stray::{StrayAnalysisSettings, StraySystem, primary_reference};

/// Sets the transport option `key` (`share`, `head`, `sag`) to `value`.
fn set(options: &mut TransportOptions, key: &str, value: &str) {
    match key {
        "share" => options.gpu_share = Some(value.parse().unwrap()),
        "head" => options.head_events = value.parse().unwrap(),
        "sag" => {
            options.sag = match value {
                "f32-scan" => SagMode::F32Scan,
                "f32-grazing" => SagMode::F32Grazing,
                _ => panic!("unknown sag mode {value}"),
            }
        }
        _ => panic!("unknown transport option {key}"),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("stray_stages measures Metal and runs on macOS only");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
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
        .unwrap_or_else(|| vec!["cpu".into(), "device".into()]);
    // The options of every device column.
    let mut options = TransportOptions::default();
    for kv in a.iter().skip(5) {
        let (key, value) = kv.split_once('=').expect("KEY=VALUE");
        set(&mut options, key, value);
    }
    let engine = Engine::new(EngineConfig::default());
    let s = engine.open_session(Path::new(&a[0])).unwrap();
    let rs = s.resolved().unwrap();
    let ctx = JobContext::default();
    let metal = mklight_metal::MetalBackend::new().unwrap().unwrap();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    // Wall and process CPU time (user + system) of a stage, ms.
    let stage = |name: &str, f: &mut dyn FnMut()| {
        let (t, c) = (Instant::now(), cpu_ms());
        f();
        let (wall, cpu) = (ms(t), cpu_ms() - c);
        println!(
            "  stage {name:<28} wall {wall:8.1} ms  cpu {cpu:8.1} ms  cores {:.1}",
            cpu / wall
        );
    };
    let ss = settings.stray_settings();
    let mut sys = None;
    for rep in 0..2 {
        println!("CPU stage profile, rep {rep}:");
        stage("sizing traces", &mut || {
            let tracer = mklight_trace::SequentialTracer::new(rs.clone()).unwrap();
            tracer
                .semi_diameters_with(&settings.system_options().semi_diameters, &ctx)
                .unwrap();
        });
        stage("StraySystem::build (all)", &mut || {
            sys = Some(StraySystem::build(rs.clone(), &settings.system_options(), &ctx).unwrap());
        });
        let sys = sys.as_ref().unwrap();
        stage("system-field checks (48²)", &mut || {
            for k in 0..rs.system().fields.fields.len() {
                if let Ok(b) = sys.system_field_source(k, 0.1, 1.) {
                    primary_reference(sys, &b.spec, 48, &ctx).unwrap();
                }
            }
        });
        let beam = settings.source.resolve(sys).unwrap();
        stage("source reference (64²)", &mut || {
            primary_reference(sys, &beam.spec, 64, &ctx).unwrap();
        });
        stage("CPU transport + tally", &mut || {
            mklight_stray::run(&sys.scene, &beam.spec, &ss, &ctx).unwrap();
        });
    }
    let sys = sys.unwrap();
    let beam = settings.source.resolve(&sys).unwrap();
    println!("{} primitives", sys.scene.primitives().len());
    let cpu = mklight_stray::run(&sys.scene, &beam.spec, &ss, &ctx).unwrap();
    let mut times = vec![Vec::new(); designs.len()];
    for rep in 0..=reps {
        for (k, d) in designs.iter().enumerate() {
            // `device+KEY=VALUE+…` sets options of one column.
            let mut parts = d.split('+');
            let transport = parts.next().unwrap();
            let mut options = options;
            for kv in parts {
                let (key, value) = kv.split_once('=').expect("KEY=VALUE");
                set(&mut options, key, value);
            }
            let (t, c0) = (Instant::now(), cpu_ms());
            let r = if transport == "cpu" {
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
            let cores = (cpu_ms() - c0) / elapsed;
            if rep > 0 {
                times[k].push(elapsed);
            }
            let e = &r.energy;
            let error = |e: &mklight_stray::EnergyReport| {
                e.detected_owned_std_error.expect("at least two rays")
            };
            let z =
                (e.detected_owned - cpu.energy.detected_owned) / error(e).hypot(error(&cpu.energy));
            let info = r.run_info.as_ref().map(|i| &i.settings);
            let detail = info
                .map(|i| {
                    format!(
                        "share {} {} {}",
                        i["gpu_share"], i["counters"], i["timings"]
                    )
                })
                .unwrap_or_default();
            println!(
                "rep {rep} {d:>10}: {elapsed:8.1} ms ({cores:.1} cores), {} events (cpu {}), owned {:.6e} ± {:.1e} \
                 (z {z:+.2} vs cpu), balance {:.1e}, backend {} {detail} {:?}",
                e.events,
                cpu.energy.events,
                e.detected_owned,
                error(e),
                e.relative_residual(),
                r.run_info.as_ref().map_or("?", |i| i.backend.as_str()),
                r.warnings,
            );
        }
    }
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
