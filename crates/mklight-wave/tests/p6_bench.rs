//! Provisional matched end-to-end timings. No idle-host wait or quiet-run retry.
#![cfg(target_os = "macos")]
use mklight_core::{JobContext, samples};
use mklight_gpu::{ComputeBackend, resident::PlanStats};
use mklight_wave::{analysis as cpu, gpu, image_simulation as image};
use std::{collections::BTreeMap, sync::Arc, time::Instant};
fn summary(v: &[f64]) -> [f64; 2] {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    [
        (v[(v.len() - 1) / 2] + v[v.len() / 2]) * 0.5,
        v[((v.len() as f64 * 0.95).ceil() as usize - 1).min(v.len() - 1)],
    ]
}
fn stats(info: &mklight_core::RunInfo) -> serde_json::Value {
    let s: PlanStats =
        serde_json::from_value(info.settings["resident"]["plan_stats"].clone()).unwrap();
    let mut stages = BTreeMap::<String, [f64; 4]>::new();
    for stage in &s.stages {
        let a = stages.entry(stage.name.clone()).or_default();
        a[0] += stage.wall_seconds;
        a[1] += stage.device_seconds;
        a[2] += stage.command_buffers as f64;
        a[3] = a[3].max(stage.max_command_seconds);
    }
    serde_json::json!({"plan_wall_ms":s.wall_seconds*1000.,"device_ms":s.device_seconds*1000.,"commands":s.command_buffers,"allocations":s.allocations,"reused_allocations":s.reused_allocations,"peak_plan_bytes":s.peak_bytes,"copied_bytes":s.copied_bytes,"fallback_count":s.fallback_count,"polished_rays":s.polished_rays,"stages_seconds_commands":stages})
}
fn backend() -> mklight_metal::MetalBackend {
    let start = Instant::now();
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required for P6 benchmark");
    eprintln!(
        "P6_BENCH_ENV {}",
        serde_json::json!({"device":b.info(),"precision":b.precision(),"device_init_ms":start.elapsed().as_secs_f64()*1000.,"provisional":true,"seed":1,"build":"release","host_policy":"normal shared host; no idle wait or retries"})
    );
    b
}
fn psf_case(
    b: &dyn ComputeBackend,
    name: &str,
    t: &mklight_trace::SequentialTracer,
    settings: &cpu::PsfAnalysisSettings,
) {
    let mut ctimes = vec![];
    let mut gtimes = vec![];
    for run in -2i32..10 {
        let ctx = JobContext::new();
        let before = b.resident_cache_stats();
        let cpu = || {
            let start = Instant::now();
            let r = cpu::psf_analysis_job(t, settings, &ctx).unwrap();
            (r, start.elapsed().as_secs_f64() * 1000.)
        };
        let gpu = || {
            let start = Instant::now();
            let r = gpu::analysis::psf_analysis_job(b, t, settings, &ctx).unwrap();
            (r, start.elapsed().as_secs_f64() * 1000.)
        };
        let ((a, ct), (z, gt)) = if run % 2 == 0 {
            let a = cpu();
            let z = gpu();
            (a, z)
        } else {
            let z = gpu();
            let a = cpu();
            (a, z)
        };
        if run == -2 {
            assert_eq!(a.psfs.len(), z.psfs.len());
            for (a, z) in a.psfs.iter().zip(&z.psfs) {
                assert_eq!(a.psf.size, z.psf.size);
                let norm = a.psf.data.iter().sum::<f64>();
                assert!(
                    a.psf
                        .data
                        .iter()
                        .zip(&z.psf.data)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        / norm
                        < 2e-3
                );
            }
        }
        let after = b.resident_cache_stats();
        let st = stats(&z.run_info);
        assert_eq!(st["copied_bytes"], 0);
        eprintln!(
            "P6_BENCH_ROW {}",
            serde_json::json!({"case":name,"run":run,"phase":if run==-2{"workload_cold"}else if run==-1{"warmup"}else{"measured"},"cpu_ms":ct,"gpu_ms":gt,"settings":settings,"stats":st,"boundary_copied_bytes":after.copied_bytes-before.copied_bytes,"system_upload_bytes":after.uploaded_bytes-before.uploaded_bytes,"cache":after})
        );
        if run >= 0 {
            ctimes.push(ct);
            gtimes.push(gt);
        }
    }
    eprintln!(
        "P6_BENCH_SUMMARY {}",
        serde_json::json!({"case":name,"cpu_median_p95_ms":summary(&ctimes),"gpu_median_p95_ms":summary(&gtimes),"provisional":true})
    );
}
#[test]
#[ignore = "provisional matched native CPU/GPU benchmarks, actual host Metal"]
fn native_pipeline_benchmark() {
    let b = backend();
    let image_only = std::env::var("P6_BENCH_IMAGE_ONLY").is_ok();
    if !image_only {
        for (name, model) in samples::all() {
            if std::env::var("FUSION_BENCH").is_ok()
                && !matches!(name, "newtonian" | "cooke_triplet" | "singlet")
            {
                continue;
            }
            let start = Instant::now();
            let t = mklight_analysis::tracer_for(model).unwrap();
            eprintln!(
                "P6_BENCH_SETUP {}",
                serde_json::json!({"case":name,"shared_resolution_ms":start.elapsed().as_secs_f64()*1000.})
            );
            psf_case(&b, name, &t, &Default::default());
        }
        if std::env::var("FUSION_BENCH").is_ok() {
            return;
        }
        let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
        // One primary-wavelength field at explicit pupil/image sampling.
        let dense = |rays: usize, n: usize, m: usize| cpu::PsfAnalysisSettings {
            fields: mklight_analysis::util::FieldSelection::Index { index: 2 },
            wavelengths: mklight_analysis::util::WavelengthSelection::Primary,
            pupil: mklight_wave::PupilSettings {
                rays,
                grid: n,
                ..Default::default()
            },
            grid_size: m,
            ..Default::default()
        };
        for (n, m) in [(64, 128), (256, 512), (1024, 2048)] {
            psf_case(&b, &format!("complete_dense_{n}_{m}"), &t, &dense(65, n, m));
        }
        for (name, model) in samples::all().into_iter().filter(|(name, _)| {
            matches!(
                *name,
                "singlet" | "double_gauss" | "newtonian" | "relay" | "hybrid_doe" | "anamorphic"
            )
        }) {
            let t = mklight_analysis::tracer_for(model).unwrap();
            psf_case(
                &b,
                &format!("dense_{name}_256_512"),
                &t,
                &dense(65, 256, 512),
            );
        }
        for (rays, n, m) in [(11, 64, 128), (101, 256, 512), (1001, 1024, 2048)] {
            psf_case(
                &b,
                &format!("lattice_{rays}_{n}_{m}"),
                &t,
                &dense(rays, n, m),
            );
        }
    }
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    let s = image::ImageSimulationSettings::default();
    let start = Instant::now();
    let cg = Arc::new(image::build_psf_grid(&t, &s.optics, &ctx).unwrap());
    let cpu_grid_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    let r = gpu::image_simulation::build_psf_grid(&b, &t, &s.optics, &ctx).unwrap();
    let gpu_grid_ms = start.elapsed().as_secs_f64() * 1000.;
    eprintln!(
        "P6_BENCH_IMAGE_GRID {}",
        serde_json::json!({"cpu_ms":cpu_grid_ms,"gpu_ms":gpu_grid_ms,"stats":r.stats,"settings":s.optics})
    );
    let gg = Arc::new(r.value);
    for cached in [false, true] {
        let name = if cached {
            "image_cached_grid"
        } else {
            "image_complete"
        };
        let mut ctimes = vec![];
        let mut gtimes = vec![];
        for run in -2i32..10 {
            let cpu = || {
                let start = Instant::now();
                let r = image::image_simulation_job(
                    &t,
                    &s,
                    cached.then(|| cg.clone()),
                    &JobContext::new(),
                )
                .unwrap();
                (r, start.elapsed().as_secs_f64() * 1000.)
            };
            let gpu = || {
                let start = Instant::now();
                let r = gpu::image_simulation::image_simulation_job(
                    &b,
                    &t,
                    &s,
                    cached.then(|| gg.clone()),
                    &JobContext::new(),
                )
                .unwrap();
                (r, start.elapsed().as_secs_f64() * 1000.)
            };
            let ((a, ct), (z, gt)) = if run % 2 == 0 {
                let a = cpu();
                let z = gpu();
                (a, z)
            } else {
                let z = gpu();
                let a = cpu();
                (a, z)
            };
            let norm = a.image.iter().sum::<f64>();
            assert_eq!(a.image.len(), z.image.len());
            assert!(
                a.image
                    .iter()
                    .zip(&z.image)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
                    / norm
                    < 2e-3
            );
            eprintln!(
                "P6_BENCH_ROW {}",
                serde_json::json!({"case":name,"run":run,"cpu_ms":ct,"gpu_ms":gt,"settings":s,"stats":stats(&z.run_info),"effective_tile_size":z.run_info.settings["effective_tile_size"]})
            );
            if run >= 0 {
                ctimes.push(ct);
                gtimes.push(gt);
            }
        }
        eprintln!(
            "P6_BENCH_SUMMARY {}",
            serde_json::json!({"case":name,"cpu_median_p95_ms":summary(&ctimes),"gpu_median_p95_ms":summary(&gtimes),"provisional":true})
        );
    }
}

fn analysis_case<S: serde::Serialize, T>(
    b: &dyn ComputeBackend,
    name: &str,
    settings: &S,
    cpu: impl Fn() -> mklight_core::MkResult<T>,
    gpu: impl Fn() -> mklight_core::MkResult<T>,
    info: impl Fn(&T) -> &mklight_core::RunInfo,
) {
    let mut ctimes = vec![];
    let mut gtimes = vec![];
    for run in -2i32..10 {
        let before = b.resident_cache_stats();
        let c = || {
            let start = Instant::now();
            let r = cpu().unwrap();
            std::hint::black_box(&r);
            (r, start.elapsed().as_secs_f64() * 1000.)
        };
        let g = || {
            let start = Instant::now();
            let r = gpu().unwrap();
            std::hint::black_box(&r);
            (r, start.elapsed().as_secs_f64() * 1000.)
        };
        let ((_a, ct), (z, gt)) = if run % 2 == 0 {
            let a = c();
            let z = g();
            (a, z)
        } else {
            let z = g();
            let a = c();
            (a, z)
        };
        let after = b.resident_cache_stats();
        let st = stats(info(&z));
        assert_eq!(st["copied_bytes"], 0);
        eprintln!(
            "P6_BENCH_ROW {}",
            serde_json::json!({"case":name,"run":run,"phase":if run==-2{"workload_cold"}else if run==-1{"warmup"}else{"measured"},"cpu_ms":ct,"gpu_ms":gt,"settings":settings,"stats":st,"boundary_copied_bytes":after.copied_bytes-before.copied_bytes,"cache":after})
        );
        if run >= 0 {
            ctimes.push(ct);
            gtimes.push(gt);
        }
    }
    eprintln!(
        "P6_BENCH_SUMMARY {}",
        serde_json::json!({"case":name,"cpu_median_p95_ms":summary(&ctimes),"gpu_median_p95_ms":summary(&gtimes),"provisional":true})
    );
}
#[test]
#[ignore = "matched complete wave analysis timings on host Metal"]
fn native_analysis_benchmark() {
    let b = backend();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    macro_rules! case {
        ($name:expr,$settings:expr,$job:ident) => {{
            let s = $settings;
            analysis_case(
                &b,
                $name,
                &s,
                || cpu::$job(&t, &s, &JobContext::new()),
                || gpu::analysis::$job(&b, &t, &s, &JobContext::new()),
                |r| &r.run_info,
            );
        }};
    }
    case!(
        "wavefront_cooke",
        cpu::WavefrontSettings::default(),
        wavefront_map_job
    );
    case!(
        "mtf_direct_cooke",
        cpu::MtfSettings::default(),
        mtf_analysis_job
    );
    case!(
        "mtf_fft_cooke",
        cpu::MtfSettings {
            method: cpu::MtfAnalysisMethod::Fft,
            ..Default::default()
        },
        mtf_analysis_job
    );
    case!(
        "mtf_field_cooke",
        cpu::MtfVsFieldSettings::default(),
        mtf_vs_field_job
    );
    case!(
        "focus_direct_cooke",
        cpu::ThroughFocusSettings::default(),
        through_focus_mtf_analysis_job
    );
    case!(
        "energy_cooke",
        cpu::EncircledEnergySettings::default(),
        diffraction_encircled_energy_job
    );
    case!(
        "line_spread_cooke",
        cpu::LineSpreadSettings::default(),
        line_spread_job
    );
    let pupil = mklight_wave::PupilSettings {
        grid: 16,
        rays: 17,
        ..Default::default()
    };
    let field = mklight_analysis::util::FieldSelection::Index { index: 2 };
    let huygens = cpu::HuygensGrid {
        grid_size: 32,
        padding: 2.,
    };
    case!(
        "mtf_huygens_cooke",
        cpu::MtfSettings {
            fields: field.clone(),
            pupil,
            method: cpu::MtfAnalysisMethod::Huygens,
            huygens,
            frequency_samples: 9,
            max_frequency: Some(500.),
            ..Default::default()
        },
        mtf_analysis_job
    );
    for (method, name) in [
        (cpu::MtfAnalysisMethod::Fft, "focus_fft_cooke"),
        (cpu::MtfAnalysisMethod::Huygens, "focus_huygens_cooke"),
    ] {
        case!(
            name,
            cpu::ThroughFocusSettings {
                fields: field.clone(),
                pupil,
                method,
                huygens,
                frequencies: vec![0., 40.],
                focus_samples: 5,
                focus_range_mm: Some(2.),
                ..Default::default()
            },
            through_focus_mtf_analysis_job
        );
    }
}
