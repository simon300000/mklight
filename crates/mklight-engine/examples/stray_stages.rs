//! Stage times of `stray_light` outside the engine: system build, primary
//! reference, CPU transport and the Metal transport designs, each timed
//! separately and alternated within each repetition.
//!
//! ```text
//! stray_stages LENS.mklight [RAYS] [REPS] [SETTINGS_JSON] [DESIGNS] [KEY=VARIANT ...]
//! ```
//!
//! `DESIGNS` is a comma-separated list of `stray` variants (`default`,
//! `cosplit`, `wavefront`; `cpu` for the CPU reference transport), each
//! optionally with `+KEY=VARIANT` selections of its own. `share=X` and
//! `head=N` set the transport options; other `KEY=VARIANT` pairs select
//! work-split variants for every column.

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
                options.gpu_share = Some(v.parse().unwrap());
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
            split::reset();
            for (key, v) in &extra {
                split::set(key, Some(v));
            }
            // `DESIGN+KEY=VARIANT+…` adds selections to one column.
            let mut parts = d.split('+');
            let design = parts.next().unwrap();
            for kv in parts {
                let (key, v) = kv.split_once('=').unwrap();
                split::set(key, Some(v));
            }
            if design != "default" && design != "cpu" {
                split::set("stray", Some(design));
            }
            let (t, c0) = (Instant::now(), cpu_ms());
            let r = if design == "cpu" {
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
            let z = (e.detected_owned - cpu.energy.detected_owned)
                / (e.detected_owned_std_error
                    .hypot(cpu.energy.detected_owned_std_error));
            let info = r.run_info.as_ref().map(|i| &i.settings);
            let detail = info
                .map(|i| format!("{} {} {}", i["design"], i["counters"], i["timings"]))
                .unwrap_or_default();
            println!(
                "rep {rep} {d:>10}: {elapsed:8.1} ms ({cores:.1} cores), {} events (cpu {}), owned {:.6e} ± {:.1e} \
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
