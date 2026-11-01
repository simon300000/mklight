#[path = "../common/mod.rs"]
mod common;
use mklight_core::JobContext;
use mklight_gpu::ComputeBackend;
use mklight_metal::MetalBackend;
use mklight_stray::{StraySettings, StraySystem, gpu};
use std::{sync::Arc, time::Instant};
fn measure(
    b: &dyn ComputeBackend,
    name: &str,
    model: mklight_core::model::OpticalSystem,
    opts: mklight_stray::SystemOptions,
    rays: u64,
    angle: Option<f64>,
) {
    if let Ok(filter) = std::env::var("MKLIGHT_P10_BENCH_FILTER")
        && !format!("{name}:{rays}:{angle:?}").contains(&filter)
    {
        return;
    }
    let prep = Instant::now();
    let system = StraySystem::new(Arc::new(common::resolve(model)), &opts).unwrap();
    let scene_seconds = prep.elapsed().as_secs_f64();
    let source = if let Some(angle) = angle.filter(|a| *a != 0.) {
        system.out_of_field_source(0., angle, 0.1, 1.).unwrap().spec
    } else {
        system.system_field_source(0, 0.1, 1.).unwrap().spec
    };
    let set = StraySettings {
        rays,
        seed: 1,
        owned_classes: mklight_stray::ClassMask::ALL_STRAY,
        mode: if name == "split-heavy" {
            mklight_stray::SplitMode::Split {
                max_splits: 6,
                rare_floor: 0.02,
            }
        } else {
            Default::default()
        },
        ..Default::default()
    };
    let native = mklight_stray::StrayAnalysisSettings {
        rays,
        seed: 1,
        classes: set.owned_classes,
        scene: opts.scene.clone(),
        semi_diameters: opts.semi_diameters,
        detector_margin: opts.detector_margin,
        source: if let Some(angle) = angle.filter(|a| *a != 0.) {
            mklight_stray::StraySource::OutOfField {
                angle_x_deg: 0.,
                angle_y_deg: angle,
                margin: 0.1,
                irradiance: 1.,
            }
        } else {
            mklight_stray::StraySource::SystemField {
                index: 0,
                margin: 0.1,
                strength: 1.,
            }
        },
        transport: mklight_stray::analysis::TransportOptions {
            mode: set.mode,
            ..Default::default()
        },
        ..Default::default()
    };
    let options = gpu::TransportOptions {
        source_tile: 4096,
        detections_per_source: 8,
        ..Default::default()
    };
    for iteration in 0..12 {
        let now = Instant::now();
        let cpu = mklight_stray::analyze_system(&system, &native, &JobContext::new()).unwrap();
        let cpu_seconds = now.elapsed().as_secs_f64();
        let now = Instant::now();
        let gpu = gpu::analyze_system(b, &system, &native, options, &JobContext::new()).unwrap();
        let wall = now.elapsed().as_secs_f64();
        let info = &gpu.run_info;
        let execution = &info.settings["resident"]["settings"];
        let mut device = 0.;
        let mut commands = 0u64;
        let mut peak = 0u64;
        let mut copies = 0u64;
        let mut max_command = 0f64;
        let mut allocations = 0u64;
        if let Some(plans) = execution["plans"].as_array() {
            for p in plans {
                device += p["device_seconds"].as_f64().unwrap_or(0.);
                commands += p["command_buffers"].as_u64().unwrap_or(0);
                copies += p["copied_bytes"].as_u64().unwrap_or(0);
                peak = peak.max(p["peak_bytes"].as_u64().unwrap_or(0));
                allocations += p["allocations"].as_u64().unwrap_or(0);
                if let Some(stages) = p["stages"].as_array() {
                    for s in stages {
                        max_command =
                            max_command.max(s["max_command_seconds"].as_f64().unwrap_or(0.));
                    }
                }
            }
        }
        assert!(gpu.energy.relative_residual() < 1e-5);
        assert_eq!(
            execution["fallback_count"], 0,
            "unexpected reference replacement {name}"
        );
        let se = cpu
            .energy
            .detected_std_error
            .hypot(gpu.energy.detected_std_error);
        assert!(
            (cpu.energy.detected - gpu.energy.detected).abs() < 5. * se + 1e-8 * cpu.energy.emitted
        );
        println!(
            "P10_TRANSPORT {}",
            serde_json::json!({"name":name,"rays":rays,"angle":angle,"iteration":iteration,"phase":if iteration==0{"cold"}else if iteration==1{"warmup"}else{"measured"},"scene_seconds":scene_seconds,"cpu_seconds":cpu_seconds,"metal_seconds":wall,"device_seconds":device,"commands":commands,"peak_bytes":peak,"copied_bytes":copies,"allocations":allocations,"max_command_seconds":max_command,"fallback_count":0,"events":gpu.energy.events,"cpu_events_per_second":cpu.energy.events as f64/cpu_seconds,"gpu_events_per_second":gpu.energy.events as f64/wall,"cpu_variance_time":cpu.energy.detected_owned_std_error.powi(2)*cpu_seconds,"gpu_variance_time":gpu.energy.detected_owned_std_error.powi(2)*wall,"settings":native,"source":source,"stages":execution["timings"],"resident_options":execution["resident_options"],"backend":info.backend,"build":b.precision().kernel_build})
        );
    }
}
#[test]
#[ignore]
fn full_transport_benchmark() {
    let setup = Instant::now();
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let backend_setup_seconds = setup.elapsed().as_secs_f64();
    assert!(b.precision().probe.error_free_transforms());
    eprintln!(
        "P10_TRANSPORT_ENV {}",
        serde_json::json!({"device":b.info(),"precision":b.precision(),"backend_setup_seconds":backend_setup_seconds})
    );
    for (name, s) in mklight_core::samples::all() {
        measure(&b, name, s, Default::default(), 20000, None);
    }
    let mut plate = common::window(1.5, 2., 10.);
    plate.coatings.insert(
        "split".into(),
        mklight_core::CoatingDef::Ideal {
            reflectance: 0.5,
            transmittance: 0.5,
        },
    );
    for s in &mut plate.surfaces[1..3] {
        s.coating = Some("split".into());
    }
    for rays in [10000, 100000] {
        measure(
            &b,
            "split-heavy",
            plate.clone(),
            common::no_scatter_options(),
            rays,
            None,
        );
    }
    let (base, opts) = common::double_gauss_with_barrels();
    for rays in [100, 10000, 400000, 1000000] {
        for angle in [0., 20.] {
            measure(&b, "barrel", base.clone(), opts.clone(), rays, Some(angle));
        }
    }
    for (name, s) in mklight_core::samples::all() {
        if [
            "singlet",
            "cooke_triplet",
            "double_gauss",
            "newtonian",
            "relay",
            "anamorphic",
            "hybrid_doe",
        ]
        .contains(&name)
        {
            measure(&b, name, s, Default::default(), 1000000, None);
        }
    }
}
