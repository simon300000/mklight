//! Matched source/first-hit stage benchmark, not a stray-transport benchmark.
#![cfg(target_os = "macos")]
#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, rng::Philox};
use mklight_gpu::{ComputeBackend, resident::*, stray_ops as ops};
use mklight_metal::MetalBackend;
use mklight_stray::{StraySystem, SystemOptions, geom::RayCaster, gpu};
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
#[test]
#[ignore]
fn first_hit_benchmark() {
    let b = MetalBackend::new().unwrap().expect("host Metal required");
    eprintln!(
        "P10 hardware={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    let mut cases = mklight_core::samples::all()
        .into_iter()
        .map(|(n, s)| (n.to_string(), s, SystemOptions::default(), 20000, 0.))
        .collect::<Vec<_>>();
    for n in [100, 10000, 400000, 1000000] {
        for angle in [0., 20.] {
            let (s, o) = common::double_gauss_with_barrels();
            cases.push((format!("barrel-{n}-{angle}"), s, o, n, angle));
        }
    }
    for (name, s, options, n, angle) in cases {
        let sys = StraySystem::new(Arc::new(common::resolve(s)), &options).unwrap();
        let ctx = JobContext::new();
        let t = Instant::now();
        let geometry = match gpu::prepare_geometry(&b, sys.scene.primitives(), &ctx) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "P10_BENCH {}",
                    serde_json::json!({"name":name,"unsupported":e.to_string(),"rays":n,"fallback_count":0,"unsupported_count":1})
                );
                continue;
            }
        };
        let prepare = t.elapsed().as_secs_f64();
        let beam = if angle == 0. {
            sys.system_field_source(0, 0.1, 1.).unwrap()
        } else {
            sys.out_of_field_source(0., angle, 0.1, 1.).unwrap()
        };
        let mut p = ResidentPlan::new(sys.resolved().snapshot.clone());
        p.seed = Some(1);
        let rays = gpu::generate(
            &mut p,
            &beam.spec,
            &sys.scene.wavelength_weights,
            1,
            0,
            n,
            65536,
        )
        .unwrap();
        let input = p.view(rays).unwrap();
        let hits = geometry.append(&mut p, input, 65536);
        p.outputs = vec![hits];
        let mut cpu = vec![];
        let mut gpu = vec![];
        let mut devices = vec![];
        let mut cold = 0.;
        let mut stats = PlanStats::default();
        for rep in 0..12 {
            // The immutable reference uses the same per-source stream and
            // wavelength draw before source sampling; no transport is run.
            let t = Instant::now();
            let count = (0..n)
                .into_par_iter()
                .filter(|&i| {
                    let mut rng = Philox::new(1, i as u64);
                    if sys.scene.num_wavelengths() > 1 {
                        let _ = rng.uniform();
                    }
                    let ray = beam.spec.sample(&mut rng);
                    std::hint::black_box(sys.scene.bvh().closest_hit(&ray, f64::INFINITY)).is_some()
                })
                .count();
            let c = t.elapsed().as_secs_f64();
            let before = b.resident_cache_stats();
            let t = Instant::now();
            let r = b
                .submit_plan(p.clone(), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let view = b.read_view(r.outputs[0].view()).unwrap();
            let actual = ops::read_records::<ops::Hit>(&view)
                .unwrap()
                .iter()
                .filter(|h| h.primitive != u32::MAX)
                .count();
            assert_eq!(actual, count);
            let g = t.elapsed().as_secs_f64();
            assert_eq!(r.stats.copied_bytes, 0);
            assert_eq!(
                b.resident_cache_stats().copied_bytes - before.copied_bytes,
                0
            );
            if rep == 0 {
                cold = g;
            }
            if rep >= 2 {
                cpu.push(c);
                gpu.push(g);
                devices.push(r.stats.stages.iter().map(|s| s.device_seconds).sum::<f64>());
            }
            stats = r.stats;
        }
        let percentile = |v: &[f64], k: usize| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[k] * 1000.
        };
        eprintln!(
            "P10_BENCH {}",
            serde_json::json!({"name":name,"rays":n,"seed":1,"source":beam.spec,"primitives":sys.scene.primitives().len(),"prepare_ms":prepare*1000.,"cold_ms":cold*1000.,"cpu_raw_s":cpu,"gpu_raw_s":gpu,"device_raw_s":devices,"cpu_median_ms":percentile(&cpu,5),"cpu_p95_ms":percentile(&cpu,9),"gpu_median_ms":percentile(&gpu,5),"gpu_p95_ms":percentile(&gpu,9),"stats":stats,"provisional":true,"scope":"source and first hit only"})
        );
    }
}
