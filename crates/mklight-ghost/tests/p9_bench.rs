#![cfg(target_os = "macos")]
mod common;
use common::*;
use mklight_core::{JobContext, samples};
use mklight_ghost::{GhostSettings, IrradianceSettings, ghost_irradiance, gpu};
use mklight_gpu::ComputeBackend;
use std::time::Instant;
fn counters(r: &mklight_ghost::GhostIrradiance, system_copy_bytes: u64) -> serde_json::Value {
    let g = &r.run.settings["gpu"];
    let plans = g["plans"]
        .as_array()
        .unwrap()
        .iter()
        .chain(g["map_plans"].as_array().unwrap());
    let mut polished = 0u64;
    let (mut device, mut commands, mut copied, mut peak, mut max_command, mut allocations) =
        (0., 0u64, 0u64, 0u64, 0.0f64, 0u64);
    for p in plans {
        device += p["device_seconds"].as_f64().unwrap();
        commands += p["command_buffers"].as_u64().unwrap();
        copied += p["copied_bytes"].as_u64().unwrap();
        polished += p["polished_rays"].as_u64().unwrap();
        allocations += p["allocations"].as_u64().unwrap();
        peak = peak.max(p["peak_bytes"].as_u64().unwrap());
        for s in p["stages"].as_array().unwrap() {
            max_command = max_command.max(s["max_command_seconds"].as_f64().unwrap());
        }
    }
    serde_json::json!({"device_seconds":device,"commands":commands,"copied_bytes":copied+system_copy_bytes,"plan_copied_bytes":copied,"system_copy_bytes":system_copy_bytes,"polished_rays":polished,"reference_seconds":g["reference_seconds"],"peak_plan_bytes":peak,"retained_resident_bytes":g["resident_bytes"],"allocations":allocations,"max_command_seconds":max_command,"transport_probes":g["distinct_probes"],"lattice_keys":g["lattice_keys"],"reference_probes":g["reference_probes"],"boundary_probes":g["boundary_probes"],"pixel_migrations":g["pixel_migrations"]})
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
            let uploaded_before = b.resident_cache_stats().uploaded_bytes;
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
                serde_json::json!({"n":n,"run":run,"settings":settings,"cpu_seconds":cs,"metal_seconds":gs,"map_l1_relative":l1,"counters":counters(&g,b.resident_cache_stats().uploaded_bytes-uploaded_before)})
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
        if std::env::var("P9_SAMPLE").is_ok_and(|selected| selected != name) {
            continue;
        }
        let m = model(s, GhostSettings::default());
        let settings = IrradianceSettings::default();
        let start = Instant::now();
        let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
        let cs = start.elapsed().as_secs_f64();
        let uploaded_before = b.resident_cache_stats().uploaded_bytes;
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
        let pm = c.primary.map.as_ref().unwrap();
        let gm = g.primary.map.as_ref().unwrap();
        let primary_l1 = pm
            .values
            .iter()
            .zip(&gm.values)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            * pm.bin_area()
            / pm.flux();
        for ghost in &c.ghosts {
            let actual = g.ghosts.iter().find(|g| g.path == ghost.path).unwrap();
            for (a, c) in actual.fields.iter().zip(&ghost.fields) {
                assert_eq!(a.rays_traced, c.rays_traced, "{name} leaf count");
                assert_eq!(
                    a.rays_detected, c.rays_detected,
                    "{name} {:?} {:?} detected count, flux {} vs {}",
                    ghost.path, c.field, a.flux, c.flux
                );
                assert!((a.rms_radius - c.rms_radius).abs() <= 0.005 * c.rms_radius + 2e-5);
                for k in 0..2 {
                    assert!((a.centroid[k] - c.centroid[k]).abs() <= 2e-5);
                }
            }
        }
        assert!(primary_l1 <= 1e-3, "{name} primary map L1 {primary_l1}");
        eprintln!(
            "{}",
            serde_json::json!({"sample":name,"settings":settings,"cpu_seconds":cs,"metal_seconds":gs,"map_l1_relative":l1/flux.max(1e-30),"primary_flux_error":primary,"primary_map_l1_relative":primary_l1,"counters":counters(&g,b.resident_cache_stats().uploaded_bytes-uploaded_before)})
        );
        assert!(
            l1 <= 1e-3 * flux + 1e-12,
            "{name} map L1 {}",
            l1 / flux.max(1e-30)
        );
        assert!(primary <= 1e-3);
    }
}

#[test]
#[ignore]
fn paraxial_batches() {
    use mklight_core::{CoatingDef, Layer, MaterialRef};
    use mklight_ghost::{EnumerationSettings, enumerate};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    for coated in [false, true] {
        let mut system = samples::double_gauss();
        if coated {
            system.coatings.insert(
                "batch-film".into(),
                CoatingDef::ThinFilm {
                    reference_wavelength_um: D_LINE,
                    layers: (0..4)
                        .map(|i| Layer {
                            material: MaterialRef::Constant {
                                n: if i % 2 == 0 { 1.38 } else { 2.05 },
                            },
                            thickness_um: 0.,
                            quarter_waves: Some(1.),
                            extinction: 0.,
                        })
                        .collect(),
                    mirror_substrate: None,
                },
            );
            let image = system.surfaces.len() - 1;
            for s in &mut system.surfaces[1..image] {
                if s.interaction == mklight_core::Interaction::Refract {
                    s.coating = Some("batch-film".into());
                }
            }
        }
        let m = model(system, GhostSettings::default());
        let paths = enumerate(
            &m,
            &EnumerationSettings {
                max_reflections: 4,
                max_higher_order_paths: 1024,
                ..Default::default()
            },
        )
        .unwrap()
        .paths
        .into_iter()
        .map(|p| p.path)
        .collect::<Vec<_>>();
        assert!(paths.len() >= 1024);
        let wavelengths = (0..m.system().wavelengths_um.len()).collect::<Vec<_>>();
        for count in [1, 16, 256, 1024] {
            let paths = &paths[..count];
            let (mut cpu_times, mut metal_times) = (Vec::new(), Vec::new());
            for run in 0..11 {
                let start = Instant::now();
                let mut reference = Vec::new();
                for path in paths {
                    let steps = m.check_path(path).unwrap();
                    for &w in &wavelengths {
                        let fo = m.tracer().paraxial().at(w);
                        reference.push((
                            [
                                mklight_ghost::paraxial::trace_steps(
                                    &m,
                                    &steps,
                                    w,
                                    fo.marginal.y[1],
                                    fo.marginal.nu[0],
                                ),
                                mklight_ghost::paraxial::trace_steps(
                                    &m,
                                    &steps,
                                    w,
                                    fo.chief.y[1],
                                    fo.chief.nu[0],
                                ),
                            ],
                            mklight_ghost::paraxial::axial_throughput(&m, &steps, w),
                        ));
                    }
                }
                let cs = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let result =
                    gpu::paraxial_batch(&b, &m, paths, &wavelengths, &JobContext::new()).unwrap();
                let gs = start.elapsed().as_secs_f64();
                for (i, (rays, (r, t))) in reference.iter().enumerate() {
                    for j in 0..2 {
                        for (a, c) in result.rays[i][j].y.iter().zip(&rays[j].y) {
                            assert!((a - c).abs() <= 1e-6 * c.abs() + 2e-5);
                        }
                    }
                    for (a, c) in result.throughput[i].into_iter().zip([*r, *t]) {
                        assert!((a - c).abs() <= 1e-5 * c.abs() + 1e-10);
                    }
                }
                eprintln!(
                    "{}",
                    serde_json::json!({"coated":coated,"paths":count,"wavelengths":wavelengths,"run":run,"cpu_seconds":cs,"metal_seconds":gs,"stats":result.stats,"reference_rows":result.run.settings["reference_rows"]})
                );
                if run > 0 {
                    cpu_times.push(cs);
                    metal_times.push(gs);
                }
            }
            cpu_times.sort_by(f64::total_cmp);
            metal_times.sort_by(f64::total_cmp);
            eprintln!(
                "{}",
                serde_json::json!({"coated":coated,"paths":count,"cpu_median":cpu_times[5],"cpu_p95":cpu_times[9],"metal_median":metal_times[5],"metal_p95":metal_times[9]})
            );
        }
    }
}
