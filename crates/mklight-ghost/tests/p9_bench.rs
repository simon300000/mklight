#![cfg(target_os = "macos")]
mod common;
use common::*;
use mklight_core::{JobContext, samples};
use mklight_ghost::{GhostSettings, IrradianceSettings, ghost_irradiance, gpu};
use mklight_gpu::ComputeBackend;
use std::time::Instant;
fn counters(r: &mklight_ghost::GhostIrradiance) -> serde_json::Value {
    let g = &r.run.settings["gpu"];
    let plans = g["plans"]
        .as_array()
        .unwrap()
        .iter()
        .chain(g["map_plans"].as_array().unwrap());
    let (mut device, mut commands, mut copied, mut peak, mut max_command, mut allocations) =
        (0., 0u64, 0u64, 0u64, 0.0f64, 0u64);
    for p in plans {
        device += p["device_seconds"].as_f64().unwrap();
        commands += p["command_buffers"].as_u64().unwrap();
        copied += p["copied_bytes"].as_u64().unwrap();
        allocations += p["allocations"].as_u64().unwrap();
        peak = peak.max(p["peak_bytes"].as_u64().unwrap());
        for s in p["stages"].as_array().unwrap() {
            max_command = max_command.max(s["max_command_seconds"].as_f64().unwrap());
        }
    }
    serde_json::json!({"device_seconds":device,"commands":commands,"copied_bytes":copied,"peak_plan_bytes":peak,"retained_resident_bytes":g["resident_bytes"],"allocations":allocations,"max_command_seconds":max_command,"transport_probes":g["distinct_probes"],"reference_probes":g["reference_probes"]})
}
#[test]
#[ignore]
fn matched_double_gauss() {
    let init = Instant::now();
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    eprintln!(
        "{}",
        serde_json::json!({"device":b.info(),"precision":b.precision(),"cold_backend_seconds":init.elapsed().as_secs_f64(),"provisional":true,"threads":rayon::current_num_threads()})
    );
    let m = model(samples::double_gauss(), GhostSettings::default());
    for n in [20, 48, 128] {
        let settings = IrradianceSettings {
            fields: vec![0, 2],
            wavelengths: vec![1],
            pupil_samples: n,
            keep_path_maps: true,
            ..Default::default()
        };
        let (mut cpu, mut metal) = (vec![], vec![]);
        for run in 0..11 {
            let start = Instant::now();
            let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
            let cs = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let g = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
            let gs = start.elapsed().as_secs_f64();
            let l1 = g
                .total_ghost_map
                .values
                .iter()
                .zip(&c.total_ghost_map.values)
                .map(|(a, c)| (a - c).abs())
                .sum::<f64>()
                * c.total_ghost_map.bin_area()
                / c.total_ghost_map.flux();
            assert!(l1 < 1e-3, "map L1 {l1}");
            eprintln!(
                "{}",
                serde_json::json!({"n":n,"run":run,"settings":settings,"cpu_seconds":cs,"metal_seconds":gs,"map_l1_relative":l1,"counters":counters(&g)})
            );
            if run > 0 {
                cpu.push(cs);
                metal.push(gs);
            }
        }
        cpu.sort_by(f64::total_cmp);
        metal.sort_by(f64::total_cmp);
        eprintln!(
            "{}",
            serde_json::json!({"n":n,"cpu_median":cpu[5],"cpu_p95":cpu[9],"metal_median":metal[5],"metal_p95":metal[9]})
        );
    }
}
#[test]
#[ignore]
fn all_sample_default_maps() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    for (name, s) in samples::all() {
        let m = model(s, GhostSettings::default());
        let settings = IrradianceSettings::default();
        let start = Instant::now();
        let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
        let cs = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let g = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
        let gs = start.elapsed().as_secs_f64();
        let flux = c.total_ghost_map.flux();
        let l1 = g
            .total_ghost_map
            .values
            .iter()
            .zip(&c.total_ghost_map.values)
            .map(|(a, c)| (a - c).abs())
            .sum::<f64>()
            * c.total_ghost_map.bin_area();
        let primary = (g.primary.flux - c.primary.flux).abs() / c.primary.flux.max(1e-30);
        eprintln!(
            "{}",
            serde_json::json!({"sample":name,"settings":settings,"cpu_seconds":cs,"metal_seconds":gs,"map_l1_relative":l1/flux.max(1e-30),"primary_flux_error":primary,"counters":counters(&g)})
        );
        assert!(
            l1 <= 1e-3 * flux + 1e-12,
            "{name} map L1 {}",
            l1 / flux.max(1e-30)
        );
        assert!(primary <= 1e-3);
    }
}
