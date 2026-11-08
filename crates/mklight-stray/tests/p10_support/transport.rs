//! Actual-Metal transport, estimator and analysis-adapter qualification.
#![cfg(target_os = "macos")]
#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{ComputeBackend, stray_ops as ops};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_stray::{SplitMode, StraySettings, StraySystem, gpu};
use std::sync::Arc;

/// The all-device design these qualification tests were written for
/// (`gpu::Design::GpuTally`); the hybrid designs have their own tests below.
#[allow(clippy::too_many_arguments)]
fn legacy_run(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &mklight_stray::Scene,
    source: &mklight_stray::SourceSpec,
    settings: &StraySettings,
    options: gpu::TransportOptions,
    ctx: &JobContext,
) -> mklight_core::MkResult<mklight_stray::StrayResult> {
    gpu::run_design(
        b,
        snapshot,
        scene,
        source,
        settings,
        options,
        gpu::Design::GpuTally,
        ctx,
    )
}
#[test]
fn controlled_transport() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual device required");
        assert!(b.precision().probe.error_free_transforms());
        for mode in [
            SplitMode::Analog,
            SplitMode::Weighted { rare_floor: 0.02 },
            SplitMode::Split {
                max_splits: 3,
                rare_floor: 0.02,
            },
        ] {
            let sys = common::window(1.5, 2., 10.);
            let snapshot = Snapshot::new(sys.clone()).unwrap();
            let system = StraySystem::new(
                Arc::new(common::resolve(sys)),
                &common::no_scatter_options(),
            )
            .unwrap();
            let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
            let settings = StraySettings {
                rays: 1024,
                mode,
                ..Default::default()
            };
            let cpu =
                mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
            let got = legacy_run(
                &b,
                snapshot,
                &system.scene,
                &source,
                &settings,
                gpu::TransportOptions::default(),
                &JobContext::new(),
            )
            .unwrap();
            eprintln!(
                "{origin:?} {mode:?}: CPU {} GPU {}",
                cpu.energy.summary(),
                got.energy.summary()
            );
            assert_resident(&got);
            assert!((got.energy.emitted - cpu.energy.emitted).abs() < 1e-8);
            assert!(got.energy.is_balanced(), "{}", got.energy.summary());
            for (a, c) in got
                .energy
                .detected_by_class
                .iter()
                .map(|v| v.1)
                .zip(cpu.energy.detected_by_class.iter().map(|v| v.1))
            {
                // Same-draw paths: measured at most 8e-15 of emitted power.
                assert!((a - c).abs() < 1e-12 * cpu.energy.emitted, "{a} != {c}");
            }
        }
    }
}

#[test]
fn transport_specification_and_tiles() {
    let metal = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let cpu = mklight_gpu::CpuF32Backend::matching(&metal.precision().probe);
    let sys = common::window(1.5, 2., 10.);
    let snap = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(
        Arc::new(common::resolve(sys)),
        &common::no_scatter_options(),
    )
    .unwrap();
    let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
    let settings = StraySettings {
        rays: 65,
        mode: SplitMode::Split {
            max_splits: 4,
            rare_floor: 0.02,
        },
        ..Default::default()
    };
    let raw = |b: &dyn ComputeBackend, budget| {
        let ctx = JobContext::new();
        let prep = gpu::TransportScene::new(b, &system.scene, &ctx).unwrap();
        let mut p = mklight_gpu::resident::ResidentPlan::new(snap.clone());
        let (a, c) = prep
            .append(
                &mut p,
                &system.scene,
                &source,
                &settings,
                0,
                65,
                gpu::TransportOptions {
                    event_budget: budget,
                    stack_capacity: 64,
                    ..Default::default()
                },
            )
            .unwrap();
        p.outputs = vec![a, c];
        let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
        assert_eq!(r.stats.copied_bytes, 0);
        let ledger_view = b.read_view(r.outputs[1].view()).unwrap();
        let ledgers = ops::read_records::<ops::Ledger>(&ledger_view).unwrap();
        assert_eq!(ledgers.len(), 65);
        // Detection rows past each source's count are unspecified.
        let cap = gpu::TransportOptions::default().detections_per_source;
        let width = std::mem::size_of::<ops::Detection>() / 4;
        let hits = b.read_view(r.outputs[0].view()).unwrap().words().to_vec();
        let valid = ledgers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| hits[i * cap * width..][..l.hits as usize * width].to_vec())
            .collect::<Vec<_>>();
        (valid, ledger_view.words().to_vec())
    };
    let a = raw(&cpu, 7);
    let b = raw(&metal, 7);
    assert!(!a.0.is_empty());
    assert_eq!(a, b, "CPU-f32 specification and Metal records differ");
    assert_eq!(b, raw(&metal, 64), "event budget changed the records");
}

/// Identity comparisons are meaningful only between actual GPU transports.
fn assert_resident(r: &mklight_stray::StrayResult) {
    assert_eq!(
        r.run_info.as_ref().unwrap().settings["fallback_count"],
        0,
        "comparison must use actual GPU transport"
    );
}
fn compare_results(a: &mklight_stray::StrayResult, b: &mklight_stray::StrayResult) {
    assert_resident(b);
    let floor = 1e-8 * a.energy.emitted;
    assert!(
        b.energy.relative_residual() < 1e-5,
        "{}",
        b.energy.summary()
    );
    for c in 0..5 {
        let (av, bv) = (
            a.energy.detected_by_class[c].1,
            b.energy.detected_by_class[c].1,
        );
        let se = a.energy.detected_std_error_by_class[c]
            .1
            .hypot(b.energy.detected_std_error_by_class[c].1);
        assert!(
            (av - bv).abs() <= 5. * se + floor,
            "class {c}: {av} vs {bv} SE={se}"
        );
        for i in 0..a.detector.flux[c].len() {
            let se = a.detector.flux_std_error[c][i].hypot(b.detector.flux_std_error[c][i]);
            assert!(
                (a.detector.flux[c][i] - b.detector.flux[c][i]).abs() <= 5. * se + floor,
                "map {c}/{i}"
            );
        }
        let (am, bm) = (&a.detector.moments[c], &b.detector.moments[c]);
        if av > floor && bv > floor {
            for j in 0..2 {
                assert!(
                    (am.centroid[j] - bm.centroid[j]).abs()
                        <= 5. * am.centroid_std_error[j].hypot(bm.centroid_std_error[j]) + 2e-5,
                    "centroid class {c}"
                );
            }
            assert!(
                (am.rms_radius - bm.rms_radius).abs()
                    <= 5. * am.rms_radius_std_error.hypot(bm.rms_radius_std_error) + 2e-5,
                "RMS class {c}"
            );
        }
    }
}
#[test]
fn samples_all_fields_and_spectra() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual device required");
        for (name, sys) in mklight_core::samples::all() {
            let snapshot = Snapshot::new(sys.clone()).unwrap();
            let fields = sys.fields.fields.len();
            let system =
                StraySystem::new(Arc::new(common::resolve(sys)), &Default::default()).unwrap();
            for field in 0..fields {
                let source = system.system_field_source(field, 0.1, 1.).unwrap().spec;
                let settings = StraySettings {
                    rays: 2048,
                    seed: 93 + field as u64,
                    ..Default::default()
                };
                let cpu = mklight_stray::run(&system.scene, &source, &settings, &JobContext::new())
                    .unwrap();
                let got = legacy_run(
                    &b,
                    snapshot.clone(),
                    &system.scene,
                    &source,
                    &settings,
                    gpu::TransportOptions {
                        detections_per_source: 8,
                        ..Default::default()
                    },
                    &JobContext::new(),
                )
                .unwrap_or_else(|e| panic!("{name} field {field}: {e}"));
                compare_results(&cpu, &got);
                eprintln!(
                    "P10 full origin={origin:?} {name} field={field} events={} residual={:.3e}",
                    got.energy.events,
                    got.energy.relative_residual()
                );
            }
        }
    }
}
#[test]
fn scatter_ensembles_and_doe() {
    use mklight_core::materials::diffraction::DiffractionEfficiency;
    use mklight_core::model::ScatterModel;
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual device required");
        for model in [
            ScatterModel::Lambertian { tis: 0.2 },
            ScatterModel::Gaussian {
                tis: 0.2,
                sigma_slope: 0.1,
            },
            ScatterModel::Abg {
                a: 0.01,
                b: 0.001,
                g: 2.,
            },
        ] {
            let mut sys = common::window(1.5, 2., 10.);
            sys.surfaces[1].scatter = Some(model.clone());
            let snapshot = Snapshot::new(sys.clone()).unwrap();
            let system = StraySystem::new(
                Arc::new(common::resolve(sys)),
                &common::no_scatter_options(),
            )
            .unwrap();
            let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
            for seed in 13..17 {
                let settings = StraySettings {
                    rays: 2048,
                    seed,
                    mode: SplitMode::Split {
                        max_splits: 2,
                        rare_floor: 0.1,
                    },
                    ..Default::default()
                };
                let cpu = mklight_stray::run(&system.scene, &source, &settings, &JobContext::new())
                    .unwrap();
                let got = legacy_run(
                    &b,
                    snapshot.clone(),
                    &system.scene,
                    &source,
                    &settings,
                    gpu::TransportOptions {
                        detections_per_source: 32,
                        ..Default::default()
                    },
                    &JobContext::new(),
                )
                .unwrap();
                compare_results(&cpu, &got);
                eprintln!(
                    "P10 ensemble {model:?} seed={seed} CPU={} GPU={}",
                    cpu.energy.detected, got.energy.detected
                );
            }
        }
        for efficiency in [
            DiffractionEfficiency::Ideal,
            DiffractionEfficiency::Kinoform,
            DiffractionEfficiency::Multilevel {
                levels: 4.try_into().unwrap(),
            },
        ] {
            let mut sys = mklight_core::samples::hybrid_diffractive_achromat();
            sys.surfaces[2].diffraction_efficiency = efficiency;
            let snap = Snapshot::new(sys.clone()).unwrap();
            let system = StraySystem::new(
                Arc::new(common::resolve(sys)),
                &common::no_scatter_options(),
            )
            .unwrap();
            let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
            let settings = StraySettings {
                rays: 4096,
                seed: 341,
                ..Default::default()
            };
            let cpu =
                mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
            let got = legacy_run(
                &b,
                snap,
                &system.scene,
                &source,
                &settings,
                gpu::TransportOptions {
                    detections_per_source: 8,
                    ..Default::default()
                },
                &JobContext::new(),
            )
            .unwrap();
            compare_results(&cpu, &got);
            assert!(got.signatures.iter().any(|s| !s.diffraction.is_empty()));
            eprintln!(
                "P10 DOE {efficiency:?} CPU={} GPU={}",
                cpu.energy.detected, got.energy.detected
            );
        }
    }
}

#[test]
fn coverage_partition_and_repeated_passes() {
    use mklight_core::Vec3;
    use mklight_core::model::*;
    use mklight_stray::{ClassMask, SourceSpec};
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let mut sys = common::window(1.5, 2., 10.);
    sys.coatings.insert(
        "split".into(),
        CoatingDef::Ideal {
            reflectance: 0.3,
            transmittance: 0.7,
        },
    );
    for s in &mut sys.surfaces[1..3] {
        s.coating = Some("split".into());
    }
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(
        Arc::new(common::resolve(sys)),
        &common::no_scatter_options(),
    )
    .unwrap();
    let source = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 2.,
        start_distance: 5.,
        power: 1.,
    };
    let settings = StraySettings {
        rays: 257,
        mode: SplitMode::Split {
            max_splits: 30,
            rare_floor: 0.,
        },
        max_events: 64,
        roulette_threshold: 0.,
        owned_classes: ClassMask::STRAY_WITH_GHOST,
        covered_ghosts: Some([mklight_ghost::GhostPath::new(vec![2, 1])].into()),
        ..Default::default()
    };
    let options = gpu::TransportOptions {
        detections_per_source: 64,
        stack_capacity: 4,
        event_budget: 8,
        ..Default::default()
    };
    let got = legacy_run(
        &b,
        snapshot.clone(),
        &system.scene,
        &source,
        &settings,
        options,
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
    assert!((got.energy.detected_by_class[0].1 - 0.49).abs() < 1e-10);
    assert!((got.energy.detected_covered_ghost - 0.49 * 0.09).abs() < 1e-10);
    assert!((got.energy.detected_uncovered_ghost - 0.49 * 0.09 * 0.09 / (1. - 0.09)).abs() < 1e-9);
    let cpu = mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
    compare_results(&cpu, &got);
    // Tiles and event budgets never change a source's records. In the
    // all-device design neither does the stack capacity; in the hybrid
    // designs it decides which sources the CPU continues (f64 instead of
    // double-float), which changes only rounding.
    let run = |design, options| {
        gpu::run_design(
            &b,
            snapshot.clone(),
            &system.scene,
            &source,
            &settings,
            options,
            design,
            &JobContext::new(),
        )
        .unwrap()
    };
    let retiled = gpu::TransportOptions {
        source_tile: 512,
        event_budget: 32,
        ..options
    };
    let deeper = gpu::TransportOptions {
        stack_capacity: 256,
        ..retiled
    };
    assert!(got.signatures.iter().any(|s| s.overflow));
    let legacy_tiled = run(gpu::Design::GpuTally, deeper);
    assert_eq!(got.detector, legacy_tiled.detector);
    assert_eq!(got.energy, legacy_tiled.energy);
    assert_eq!(got.signatures, legacy_tiled.signatures);
    let head = run(gpu::Design::GpuHead, options);
    let tiled = run(gpu::Design::GpuHead, retiled);
    assert_resident(&head);
    compare_results(&cpu, &head);
    assert!(
        head.run_info.as_ref().unwrap().settings["counters"]["stack_full"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(head.detector, tiled.detector);
    assert_eq!(head.energy, tiled.energy);
    assert_eq!(head.signatures, tiled.signatures);
    let deep = run(gpu::Design::GpuHead, deeper);
    for c in 0..5 {
        let (a, b) = (
            head.energy.detected_by_class[c].1,
            deep.energy.detected_by_class[c].1,
        );
        assert!((a - b).abs() <= 1e-12 * head.energy.emitted, "class {c}");
    }
    // A shared front interface is crossed in both designed directions.
    let mut sys = common::window(1.5, 5., -50.);
    sys.surfaces[2].material = MaterialRef::Constant { n: 1.5 };
    sys.surfaces[2].thickness = -5.;
    sys.surfaces[2].interaction = Interaction::Reflect;
    let mut return_face = Surface::new(4, SurfaceShape::Plane, -50., MaterialRef::Air);
    common::fixed(&mut return_face, 10.);
    sys.surfaces.insert(3, return_face);
    let snap = Snapshot::new(sys.clone()).unwrap();
    let scene = common::scene(sys);
    let settings = StraySettings {
        rays: 1024,
        mode: SplitMode::Split {
            max_splits: 6,
            rare_floor: 0.02,
        },
        ..Default::default()
    };
    let cpu = mklight_stray::run(&scene, &source, &settings, &JobContext::new()).unwrap();
    let got = legacy_run(
        &b,
        snap,
        &scene,
        &source,
        &settings,
        options,
        &JobContext::new(),
    )
    .unwrap();
    compare_results(&cpu, &got);
    assert!(got.energy.detected_by_class[0].1 > 0.9);
}
#[test]
fn integrating_sphere_energy() {
    use mklight_core::math::{Frame, Mat3};
    use mklight_core::model::ScatterModel;
    use mklight_core::{Snapshot, SurfaceShape, Vec3};
    use mklight_stray::geom::{SagPatch, Shape};
    use mklight_stray::scene::{Accept, Behavior, DetectorSpec, Opaque, OpaqueKind};
    use mklight_stray::{Bsdf, SceneBuilder, SourceSpec};
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let mut builder = SceneBuilder::new(vec![0.55], vec![1.]);
    builder.add_medium(vec![1.], vec![0.]);
    let wall = || {
        Behavior::Opaque(Opaque {
            kind: OpaqueKind::Generic,
            surface: None,
            finish: Some(Arc::new(
                Bsdf::new(&ScatterModel::Lambertian { tis: 0.9 }).unwrap(),
            )),
            specular_reflectance: 0.,
        })
    };
    let cap = |r0, r1| Shape::Sag(SagPatch::new(SurfaceShape::sphere(10.), r0, r1, None).unwrap());
    builder.add(Frame::IDENTITY, cap(0., 3.), Behavior::Detector, "port");
    builder.add(Frame::IDENTITY, cap(3., 10.), wall(), "south");
    builder.add(
        Frame {
            origin: Vec3::new(0., 0., 20.),
            rot: Mat3::rot_x(std::f64::consts::PI),
        },
        cap(0., 10.),
        wall(),
        "north",
    );
    builder.set_detector(DetectorSpec {
        surface: usize::MAX,
        accept: Accept::Plus,
        half_x: 3.,
        half_y: 3.,
        nx: 4,
        ny: 4,
    });
    let scene = builder.build().unwrap();
    let source = SourceSpec::Point {
        position: Vec3::new(0., 0., 10.),
        axis: Vec3::Z,
        half_angle_deg: 180.,
        intensity: 1. / (4. * std::f64::consts::PI),
    };
    let frac = (10. - 91f64.sqrt()) / 20.;
    let want = frac / (1. - 0.9 * (1. - frac));
    for seed in 0..4 {
        let set = StraySettings {
            rays: 4096,
            seed,
            ..Default::default()
        };
        let snap = Snapshot::new(mklight_core::samples::singlet()).unwrap();
        let got = legacy_run(
            &b,
            snap,
            &scene,
            &source,
            &set,
            gpu::TransportOptions {
                detections_per_source: 8,
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
        assert!((got.energy.detected - want).abs() < 5. * got.energy.detected_std_error);
        assert!(got.energy.escaped < 1e-10);
        assert!(got.energy.relative_residual() < 1e-5);
        eprintln!(
            "P10 sphere seed={seed} detected={} SE={} analytic={want}",
            got.energy.detected, got.energy.detected_std_error
        );
    }
}
#[test]
fn split_tree_cancellation_and_release() {
    use mklight_core::model::CoatingDef;
    let b = Arc::new(
        MetalBackend::new()
            .unwrap()
            .expect("actual device required"),
    );
    let mut sys = common::window(1.5, 2., 10.);
    sys.coatings.insert(
        "split".into(),
        CoatingDef::Ideal {
            reflectance: 0.5,
            transmittance: 0.5,
        },
    );
    for s in &mut sys.surfaces[1..3] {
        s.coating = Some("split".into());
    }
    let snap = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(
        Arc::new(common::resolve(sys)),
        &common::no_scatter_options(),
    )
    .unwrap();
    let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
    let ctx = JobContext::new();
    let prepared = gpu::TransportScene::new(&*b, &system.scene, &ctx).unwrap();
    let set = StraySettings {
        rays: 256,
        mode: SplitMode::Split {
            max_splits: 10000,
            rare_floor: 0.,
        },
        max_events: 10000,
        roulette_threshold: 0.,
        ..Default::default()
    };
    let plan = || {
        let mut p = mklight_gpu::resident::ResidentPlan::new(snap.clone());
        let (a, c) = prepared
            .append(
                &mut p,
                &system.scene,
                &source,
                &set,
                0,
                256,
                gpu::TransportOptions {
                    detections_per_source: 1024,
                    event_budget: 1,
                    stack_capacity: 64,
                    ..Default::default()
                },
            )
            .unwrap();
        p.outputs = vec![a, c];
        p
    };
    let ctx = JobContext::new();
    let pending = b.submit_plan(plan(), &ctx).unwrap();
    let started = std::time::Instant::now();
    let mut previous = 0.;
    while !pending.is_ready() {
        let now = ctx.progress.fraction();
        assert!(now >= previous);
        previous = now;
        if now > 0. {
            ctx.cancel.cancel();
            break;
        }
        assert!(started.elapsed().as_secs() < 30);
        std::thread::yield_now();
    }
    let cancelled = std::time::Instant::now();
    assert!(matches!(
        pending.wait(),
        Err(mklight_core::MkError::Cancelled)
    ));
    eprintln!(
        "P10 split cancellation {:.3} ms",
        cancelled.elapsed().as_secs_f64() * 1000.
    );
    let set = StraySettings {
        rays: 257,
        ..Default::default()
    };
    let got = legacy_run(
        &*b,
        snap.clone(),
        &system.scene,
        &source,
        &set,
        gpu::TransportOptions::default(),
        &JobContext::new(),
    )
    .unwrap();
    assert_resident(&got);
    assert!(got.energy.is_balanced());
    // A genuinely too-small hit journal: the all-device design gives an
    // explicitly labelled f64 result, the hybrid designs continue the
    // sources whose journal is full on the CPU.
    let split = StraySettings {
        mode: SplitMode::Split {
            max_splits: 4,
            rare_floor: 0.,
        },
        ..set
    };
    let small = gpu::TransportOptions {
        detections_per_source: 1,
        ..Default::default()
    };
    let run = |design| {
        gpu::run_design(
            &*b,
            snap.clone(),
            &system.scene,
            &source,
            &split,
            small,
            design,
            &JobContext::new(),
        )
        .unwrap()
    };
    let fallback = run(gpu::Design::GpuTally);
    assert_eq!(fallback.run_info.as_ref().unwrap().backend, "cpu-f64");
    assert_eq!(
        fallback.run_info.as_ref().unwrap().settings["fallback_count"],
        1
    );
    let cpu = mklight_stray::run(&system.scene, &source, &split, &JobContext::new()).unwrap();
    for design in [gpu::Design::CpuTally, gpu::Design::GpuHead] {
        let continued = run(design);
        assert_resident(&continued);
        let info = &continued.run_info.as_ref().unwrap().settings;
        assert!(info["counters"]["journal_full"].as_u64().unwrap() > 0);
        compare_results(&cpu, &continued);
    }
}

#[test]
fn full_pipeline_concurrency_and_chunk_identity() {
    let b = Arc::new(
        MetalBackend::new()
            .unwrap()
            .expect("actual device required"),
    );
    let sys = mklight_core::samples::cooke_triplet();
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(Arc::new(common::resolve(sys)), &Default::default()).unwrap();
    let source = system.system_field_source(1, 0.1, 1.).unwrap().spec;
    let set = StraySettings {
        rays: 1537,
        seed: 79,
        ..Default::default()
    };
    let (a, c) = std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            legacy_run(
                &*b,
                snapshot.clone(),
                &system.scene,
                &source,
                &set,
                gpu::TransportOptions {
                    source_tile: 256,
                    event_budget: 7,
                    detections_per_source: 8,
                    ..Default::default()
                },
                &JobContext::new(),
            )
            .unwrap()
        });
        let c = legacy_run(
            &*b,
            snapshot.clone(),
            &system.scene,
            &source,
            &set,
            gpu::TransportOptions {
                source_tile: 4096,
                event_budget: 32,
                detections_per_source: 8,
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        (a.join().unwrap(), c)
    });
    assert_resident(&a);
    assert_resident(&c);
    assert_eq!(a.energy, c.energy);
    assert_eq!(a.detector, c.detector);
    assert_eq!(a.signatures, c.signatures);
}
#[test]
fn controlled_film_bulk_weights() {
    use mklight_core::{CoatingDef, Layer, MaterialRef, Vec3};
    let mut sys = common::window(1.5, 2., 10.);
    sys.coatings.insert(
        "film".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.38 },
                extinction: 0.02,
                thickness_um: 0.07,
                quarter_waves: None,
            }],
            mirror_substrate: None,
        },
    );
    for s in &mut sys.surfaces[1..3] {
        s.coating = Some("film".into());
    }
    let snap = Snapshot::new(sys.clone()).unwrap();
    let mut scene = common::scene(sys);
    scene.media[1].alpha[0] = 0.02;
    let source = mklight_stray::SourceSpec::Collimated {
        direction: Vec3::new(0.1, 0.02, 1.),
        center: Vec3::ZERO,
        radius: 2.,
        start_distance: 1000.,
        power: 1.,
    };
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual GPU required");
        for seed in 0..8 {
            let set = StraySettings {
                rays: 1,
                seed,
                roulette_threshold: 0.,
                mode: SplitMode::Split {
                    max_splits: 4,
                    rare_floor: 0.02,
                },
                ..Default::default()
            };
            let cpu = mklight_stray::run(&scene, &source, &set, &JobContext::new()).unwrap();
            let got = legacy_run(
                &b,
                snap.clone(),
                &scene,
                &source,
                &set,
                gpu::TransportOptions::default(),
                &JobContext::new(),
            )
            .unwrap();
            assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
            assert_eq!(got.signatures.len(), cpu.signatures.len());
            for g in &got.signatures {
                let c = cpu
                    .signatures
                    .iter()
                    .find(|c| c.class == g.class && c.signature == g.signature)
                    .expect("same controlled path signature");
                assert_eq!(g.hits, c.hits);
                // Same-draw paths: measured at most 7.3e-14 relative.
                assert!(
                    (g.flux - c.flux).abs() <= 1e-11 * c.flux.abs(),
                    "weight {} vs {}",
                    g.flux,
                    c.flux
                );
            }
            assert!(got.energy.relative_residual() < 1e-5);
            assert!(
                got.energy
                    .absorbed_by_sink
                    .iter()
                    .any(|(s, w)| *s == mklight_stray::Sink::Bulk && *w > 0.)
            );
            let a = &cpu.detector.moments[0];
            let c = &got.detector.moments[0];
            for i in 0..2 {
                assert!((a.centroid[i] - c.centroid[i]).abs() < 2e-5);
            }
        }
    }
}
#[test]
fn explicit_ghost_partition() {
    use mklight_core::{CoatingDef, Vec3};
    use mklight_ghost::{
        ClipMode, EnumerationSettings, GhostModel, GhostSettings, IrradianceSettings,
        ghost_irradiance,
    };
    use mklight_stray::{ClassMask, SourceSpec};
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    let mut sys = common::window(1.5, 2., 10.);
    sys.coatings.insert(
        "split".into(),
        CoatingDef::Ideal {
            reflectance: 0.3,
            transmittance: 0.7,
        },
    );
    for s in &mut sys.surfaces[1..3] {
        s.coating = Some("split".into());
    }
    let snap = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(
        Arc::new(common::resolve(sys)),
        &common::no_scatter_options(),
    )
    .unwrap();
    let model = GhostModel::new(
        Arc::new(system.tracer().clone()),
        GhostSettings {
            clip: ClipMode::ClearApertures { margin_mm: 0. },
            ..Default::default()
        },
    )
    .unwrap();
    let mut values = Vec::new();
    let mut coverage = None;
    for grid in [64, 128, 256] {
        let ghost = ghost_irradiance(
            &model,
            &IrradianceSettings {
                enumeration: EnumerationSettings {
                    max_reflections: 2,
                    ..Default::default()
                },
                fields: vec![0],
                pupil_samples: grid,
                overfill: 1.,
                map_bins: [8, 8],
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        system.check_ghost_coverage(&ghost.coverage).unwrap();
        values.push((
            ghost.primary.fields[0].flux_on_map,
            ghost
                .ghosts
                .iter()
                .map(|g| g.fields[0].flux_on_map)
                .sum::<f64>(),
        ));
        coverage = Some(ghost.coverage.paths);
    }
    let source = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 5.,
        start_distance: 5.,
        power: 1.,
    };
    let settings = StraySettings {
        rays: 1024,
        mode: SplitMode::Split {
            max_splits: 20,
            rare_floor: 0.,
        },
        roulette_threshold: 0.,
        owned_classes: ClassMask::STRAY_WITH_GHOST,
        covered_ghosts: coverage,
        ..Default::default()
    };
    let gpu = legacy_run(
        &b,
        snap,
        &system.scene,
        &source,
        &settings,
        gpu::TransportOptions {
            detections_per_source: 64,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    assert_resident(&gpu);
    let (p, g) = values[2];
    let quadrature = values[..2]
        .iter()
        .map(|v| (v.0 - p).abs() + (v.1 - g).abs())
        .fold(0f64, f64::max);
    assert!(
        (p - 0.49).abs() < quadrature + 1e-10,
        "primary={p} ghosts={g} quadrature={quadrature} series={values:?}"
    );
    assert!((g - gpu.energy.detected_covered_ghost).abs() < quadrature + 1e-10);
    assert!((p + g + gpu.energy.detected_owned - gpu.energy.detected).abs() < quadrature + 1e-8);
    eprintln!(
        "P10 explicit partition primary={p} ghost={g} owned={} quadrature={quadrature}",
        gpu.energy.detected_owned
    );
}

#[test]
fn bsdf_disk_integrals() {
    use mklight_core::math::Frame;
    use mklight_core::model::ScatterModel;
    use mklight_core::{SurfaceShape, Vec3};
    use mklight_stray::geom::{SagPatch, Shape};
    use mklight_stray::scene::{Accept, Behavior, DetectorSpec, Opaque, OpaqueKind};
    use mklight_stray::{Bsdf, SceneBuilder, SourceSpec};
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    for model in [
        ScatterModel::Lambertian { tis: 0.4 },
        ScatterModel::Gaussian {
            tis: 0.4,
            sigma_slope: 0.2,
        },
        ScatterModel::Abg {
            a: 0.01,
            b: 0.01,
            g: 2.,
        },
    ] {
        let expected = match model {
            ScatterModel::Lambertian { tis } => tis * 0.2,
            ScatterModel::Gaussian { tis, sigma_slope } => {
                let s = 2. * sigma_slope;
                tis * (-(-0.2 / (2. * s * s)).exp_m1()) / (-(-1. / (2. * s * s)).exp_m1())
            }
            ScatterModel::Abg { a, b, .. } => std::f64::consts::PI * a * (1. + 0.2 / b).ln(),
        };
        let mut builder = SceneBuilder::new(vec![0.55], vec![1.]);
        builder.add_medium(vec![1.], vec![0.]);
        builder.add(
            Frame::IDENTITY,
            Shape::Sag(SagPatch::new(SurfaceShape::Plane, 0., 10., None).unwrap()),
            Behavior::Opaque(Opaque {
                kind: OpaqueKind::Generic,
                surface: None,
                finish: Some(Arc::new(Bsdf::new(&model).unwrap())),
                specular_reflectance: 0.,
            }),
            "scatter",
        );
        builder.add(
            Frame {
                origin: Vec3::new(0., 0., -1.),
                ..Frame::IDENTITY
            },
            Shape::Sag(SagPatch::new(SurfaceShape::Plane, 0., 0.5, None).unwrap()),
            Behavior::Detector,
            "disk",
        );
        builder.set_detector(DetectorSpec {
            surface: usize::MAX,
            accept: Accept::Plus,
            half_x: 0.5,
            half_y: 0.5,
            nx: 4,
            ny: 4,
        });
        let scene = builder.build().unwrap();
        let source = SourceSpec::Collimated {
            direction: Vec3::Z,
            center: Vec3::ZERO,
            radius: 1e-8,
            start_distance: 0.5,
            power: 1.,
        };
        let set = StraySettings {
            rays: 16384,
            seed: 251,
            ..Default::default()
        };
        let result = legacy_run(
            &b,
            Snapshot::new(mklight_core::samples::singlet()).unwrap(),
            &scene,
            &source,
            &set,
            gpu::TransportOptions {
                detections_per_source: 4,
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(
            result.run_info.as_ref().unwrap().settings["fallback_count"],
            0
        );
        assert!(
            (result.energy.detected - expected).abs()
                < 5. * result.energy.detected_std_error + 1e-8
        );
        eprintln!(
            "P10 BSDF integral {model:?}: {} +/- {} expected {expected}",
            result.energy.detected, result.energy.detected_std_error
        );
    }
}
#[test]
fn doe_visit_overflow_and_coverage_boundary() {
    use mklight_core::Vec3;
    use mklight_core::model::*;
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    let phase = mklight_core::model::surface::DiffractivePhase {
        coefficients: vec![0.],
        norm_radius: 10.,
        design_wavelength_um: 0.55,
        design_order: 1,
        trace_order: 0,
    };
    let shape = SurfaceShape::Diffractive {
        base: Box::new(SurfaceShape::Plane),
        phase,
    };
    let mut sys = common::window(1.5, 2., 10.);
    sys.coatings.insert(
        "split".into(),
        CoatingDef::Ideal {
            reflectance: 0.7,
            transmittance: 0.3,
        },
    );
    for s in &mut sys.surfaces[1..3] {
        s.shape = shape.clone();
        s.coating = Some("split".into());
    }
    let snap = Snapshot::new(sys.clone()).unwrap();
    let scene = common::scene(sys);
    let source = mklight_stray::SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 1.,
        start_distance: 5.,
        power: 1.,
    };
    let set = StraySettings {
        rays: 8,
        mode: SplitMode::Split {
            max_splits: 40,
            rare_floor: 0.,
        },
        max_events: 80,
        roulette_threshold: 0.,
        ..Default::default()
    };
    let got = legacy_run(
        &b,
        snap,
        &scene,
        &source,
        &set,
        gpu::TransportOptions {
            detections_per_source: 64,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
    assert!(
        got.signatures
            .iter()
            .any(|s| s.overflow && s.diffraction.len() == 32)
    );
    let mut sys = OpticalSystem::new_empty();
    sys.surfaces = vec![Surface::new(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    for j in 1..=9 {
        let mut s = Surface::new(j, shape.clone(), 1., MaterialRef::Air);
        common::fixed(&mut s, 10.);
        sys.surfaces.push(s);
    }
    let mut image = Surface::new(10, SurfaceShape::Plane, 0., MaterialRef::Air);
    common::fixed(&mut image, 10.);
    sys.surfaces.push(image);
    sys.stop = 1;
    let snap = Snapshot::new(sys.clone()).unwrap();
    let scene = common::scene(sys);
    let bad = StraySettings {
        covered_ghosts: Some([mklight_ghost::GhostPath::new(vec![9, 1, 9, 1, 9, 1])].into()),
        rays: 1,
        ..Default::default()
    };
    let result = legacy_run(
        &b,
        snap,
        &scene,
        &source,
        &bad,
        gpu::TransportOptions::default(),
        &JobContext::new(),
    );
    assert!(
        matches!(result,Err(mklight_core::MkError::InvalidModel(ref s)) if s.contains("DOE visits"))
    );
}

#[test]
fn doe_near_integer_preserves_draw_consumption() {
    use mklight_core::materials::diffraction::DiffractionEfficiency;
    use mklight_core::model::SurfaceShape;
    use mklight_core::model::surface::DiffractivePhase;
    let mut sys = common::window(1.5, 2., 10.);
    for s in &mut sys.surfaces[1..3] {
        s.shape = SurfaceShape::Diffractive {
            base: Box::new(SurfaceShape::Plane),
            phase: DiffractivePhase {
                coefficients: vec![0.],
                norm_radius: 10.,
                design_wavelength_um: 0.55 * (1. + 1e-9),
                design_order: 1,
                trace_order: 1,
            },
        };
        s.diffraction_efficiency = DiffractionEfficiency::Kinoform;
    }
    let snap = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(
        Arc::new(common::resolve(sys)),
        &common::no_scatter_options(),
    )
    .unwrap();
    let source = system.system_field_source(0, 0.1, 1.).unwrap().spec;
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual GPU required");
        let set = StraySettings {
            rays: 1024,
            seed: 71,
            mode: SplitMode::Analog,
            ..Default::default()
        };
        let cpu = mklight_stray::run(&system.scene, &source, &set, &JobContext::new()).unwrap();
        let got = legacy_run(
            &b,
            snap.clone(),
            &system.scene,
            &source,
            &set,
            gpu::TransportOptions::default(),
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
        assert_eq!(got.energy.events, cpu.energy.events);
        assert_eq!(
            got.energy.detected_hits_by_class,
            cpu.energy.detected_hits_by_class
        );
        assert!((got.energy.detected - cpu.energy.detected).abs() < 1e-9);
    }
}

#[test]
fn wide_selected_order_keeps_integer_bits() {
    use mklight_core::Vec3;
    use mklight_core::model::surface::DiffractivePhase;
    use mklight_core::model::{SemiDiameter, SurfaceShape};
    let mut sys = common::window(1.5, 2., 998.);
    sys.surfaces[1].shape = SurfaceShape::Diffractive {
        base: Box::new(SurfaceShape::Plane),
        phase: DiffractivePhase {
            coefficients: vec![2e-4],
            norm_radius: 10.,
            design_wavelength_um: 0.55,
            design_order: 1,
            trace_order: (1 << 30) + 63,
        },
    };
    sys.surfaces[3].semi_diameter = SemiDiameter::Fixed { value: 1000. };
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let scene = common::scene(sys);
    let source = mklight_stray::SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::new(1.5, 0., 0.),
        radius: 1e-8,
        start_distance: 5.,
        power: 1.,
    };
    let set = StraySettings {
        rays: 64,
        mode: SplitMode::Weighted { rare_floor: 0. },
        ..Default::default()
    };
    let reference = mklight_stray::run(&scene, &source, &set, &JobContext::new()).unwrap();
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual GPU required");
        let got = legacy_run(
            &b,
            snapshot.clone(),
            &scene,
            &source,
            &set,
            gpu::TransportOptions::default(),
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
        let a = reference.detector.moments[0].centroid;
        let g = got.detector.moments[0].centroid;
        assert!(
            (a[0] - g[0]).abs() < 2e-5,
            "wide order position {a:?} vs {g:?}"
        );
        assert!(
            got.signatures
                .iter()
                .flat_map(|s| &s.diffraction)
                .any(|d| d.order == (1 << 30) + 63)
        );
    }
}

#[test]
fn absolute_power_precision_fallback() {
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    let sys = common::window(1.5, 2., 10.);
    let snap = Snapshot::new(sys.clone()).unwrap();
    let scene = common::scene(sys);
    for power in [1e-100, 1e100] {
        let source = mklight_stray::SourceSpec::Collimated {
            direction: mklight_core::Vec3::Z,
            center: mklight_core::Vec3::ZERO,
            radius: 1.,
            start_distance: 5.,
            power,
        };
        let set = StraySettings {
            rays: 16,
            ..Default::default()
        };
        let got = legacy_run(
            &b,
            snap.clone(),
            &scene,
            &source,
            &set,
            gpu::TransportOptions::default(),
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(got.run_info.as_ref().unwrap().backend, "cpu-f64");
        assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 1);
        assert!((got.energy.emitted / power - 1.).abs() < 1e-12);
    }
}

#[test]
fn grazing_tis_weights() {
    use mklight_core::math::Frame;
    use mklight_core::model::ScatterModel;
    use mklight_core::{SurfaceShape, Vec3};
    use mklight_stray::geom::{SagPatch, Shape};
    use mklight_stray::scene::{Accept, Behavior, DetectorSpec, Opaque, OpaqueKind};
    use mklight_stray::{Bsdf, SceneBuilder, SourceSpec};
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library: origin,
            ..Default::default()
        })
        .unwrap()
        .expect("actual device required");
        for model in [
            ScatterModel::Gaussian {
                tis: 0.4,
                sigma_slope: 1e-5,
            },
            ScatterModel::Abg {
                a: 0.0001,
                b: 1e-8,
                g: 2.,
            },
            ScatterModel::Abg {
                a: 0.01,
                b: 0.,
                g: 1.9,
            },
        ] {
            let mut builder = SceneBuilder::new(vec![0.55], vec![1.]);
            builder.add_medium(vec![1.], vec![0.]);
            let bsdf = Arc::new(Bsdf::new(&model).unwrap());
            builder.add(
                Frame::IDENTITY,
                Shape::Sag(SagPatch::new(SurfaceShape::Plane, 0., 10., None).unwrap()),
                Behavior::Opaque(Opaque {
                    kind: OpaqueKind::Generic,
                    surface: None,
                    finish: Some(bsdf.clone()),
                    specular_reflectance: 0.,
                }),
                "grazing",
            );
            builder.add(
                Frame {
                    origin: Vec3::new(0., 0., -1.),
                    ..Frame::IDENTITY
                },
                Shape::Sag(SagPatch::new(SurfaceShape::Plane, 0., 10., None).unwrap()),
                Behavior::Detector,
                "detector",
            );
            builder.set_detector(DetectorSpec {
                surface: usize::MAX,
                accept: Accept::Plus,
                half_x: 10.,
                half_y: 10.,
                nx: 4,
                ny: 4,
            });
            let scene = builder.build().unwrap();
            for z in [0.0005, 0.001, 0.01] {
                let direction = Vec3::new(1., 0., z).normalized();
                let expected = 1. - bsdf.tis(((1. - direction.z) * (1. + direction.z)).sqrt());
                let source = SourceSpec::Collimated {
                    direction,
                    center: Vec3::ZERO,
                    radius: 1e-7,
                    start_distance: 0.5,
                    power: 1.,
                };
                let set = StraySettings {
                    rays: 64,
                    seed: 19,
                    ..Default::default()
                };
                let result = legacy_run(
                    &b,
                    Snapshot::new(mklight_core::samples::singlet()).unwrap(),
                    &scene,
                    &source,
                    &set,
                    gpu::TransportOptions {
                        detections_per_source: 4,
                        ..Default::default()
                    },
                    &JobContext::new(),
                )
                .unwrap();
                assert_eq!(
                    result.run_info.as_ref().unwrap().settings["fallback_count"],
                    0
                );
                let absorbed = result
                    .energy
                    .absorbed_by_sink
                    .iter()
                    .find(|(k, _)| *k == mklight_stray::Sink::Generic)
                    .unwrap()
                    .1;
                assert!(
                    (absorbed - expected).abs() <= 1e-5 * expected.abs() + 1e-10,
                    "{model:?} z={z} {absorbed} vs {expected}"
                );
                eprintln!(
                    "P10 grazing TIS {model:?} cosine={} error={:.3e}",
                    direction.z,
                    (absorbed - expected).abs()
                );
            }
        }
    }
}

#[test]
fn native_analysis_result_adapter() {
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    let rs = Arc::new(common::resolve(mklight_core::samples::cooke_triplet()));
    let settings = mklight_stray::StrayAnalysisSettings {
        rays: 2048,
        ..Default::default()
    };
    let ctx = JobContext::new();
    let (got, signatures) = gpu::analyze_with_signatures(
        &b,
        rs.clone(),
        &settings,
        gpu::TransportOptions::default(),
        &ctx,
    )
    .unwrap();
    let reference = mklight_stray::analyze(rs, &settings, &JobContext::new()).unwrap();
    assert!(got.run_info.backend.starts_with("metal:"));
    assert_eq!(got.run_info.precision, mklight_core::job::Precision::F32Gpu);
    assert_eq!(
        got.run_info.settings["resident"]["settings"]["fallback_count"],
        0
    );
    assert_eq!(ctx.progress.fraction(), 1.);
    assert_eq!(got.primary_obstruction, reference.primary_obstruction);
    assert_eq!(got.maps.len(), reference.maps.len());
    assert!(!signatures.is_empty());
    assert!(
        (got.stray_flux - reference.stray_flux).abs()
            <= 5. * got.stray_std_error.hypot(reference.stray_std_error) + 1e-8 * got.emitted
    );
    for i in 0..got.total_irradiance.len() {
        let sum = got.maps.iter().map(|m| m.irradiance[i]).sum::<f64>();
        assert!((sum - got.total_irradiance[i]).abs() < 1e-10 * got.emitted);
    }
}

/// Every division of the transport (`gpu::Design`) against the CPU
/// reference on a scattering, ghosting sample; designs that share device
/// records against each other; reproducibility and tile independence.
#[test]
fn hybrid_designs_agree() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let (sys, opts) = common::double_gauss_with_barrels();
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(Arc::new(common::resolve(sys)), &opts).unwrap();
    let source = system.system_field_source(1, 0.1, 1.).unwrap().spec;
    let settings = StraySettings {
        rays: 6000,
        seed: 17,
        chunk_rays: 700,
        ..Default::default()
    };
    let cpu = mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
    let run = |design, options| {
        gpu::run_design(
            &b,
            snapshot.clone(),
            &system.scene,
            &source,
            &settings,
            options,
            design,
            &JobContext::new(),
        )
        .unwrap()
    };
    let options = gpu::TransportOptions {
        head_events: 12,
        gpu_share: Some(0.5),
        source_tile: 2048,
        ..Default::default()
    };
    let mut results = Vec::new();
    for design in [
        gpu::Design::GpuTally,
        gpu::Design::CpuTally,
        gpu::Design::GpuHead,
        gpu::Design::Wavefront,
        gpu::Design::CoSplit,
    ] {
        let got = run(design, options);
        compare_results(&cpu, &got);
        let info = &got.run_info.as_ref().unwrap().settings;
        if design != gpu::Design::GpuTally {
            assert_eq!(info["design"], design.name());
            let n = &info["counters"];
            let sources = [
                "device_sources",
                "head_budget",
                "stack_full",
                "journal_full",
                "retraced",
                "cpu_sources",
            ]
            .iter()
            .map(|k| n[k].as_u64().unwrap())
            .sum::<u64>();
            assert_eq!(sources, settings.rays, "{design:?}: {n}");
            assert_eq!(
                got.energy.events,
                n["device_events"].as_u64().unwrap() + n["cpu_events"].as_u64().unwrap()
            );
        }
        if matches!(design, gpu::Design::GpuHead | gpu::Design::Wavefront) {
            assert!(info["counters"]["head_budget"].as_u64().unwrap() > 0);
        }
        eprintln!("{design:?}: {}", got.energy.summary());
        results.push(got);
    }
    // The device tallies and the CPU tally of the same device records.
    for c in 0..5 {
        let (a, b) = (
            results[0].energy.detected_by_class[c].1,
            results[1].energy.detected_by_class[c].1,
        );
        assert!(
            (a - b).abs() <= 1e-10 * cpu.energy.emitted,
            "class {c}: {a} {b}"
        );
    }
    // Persistent lanes and the wavefront produce identical records.
    assert_eq!(results[2].energy, results[3].energy);
    assert_eq!(results[2].detector, results[3].detector);
    assert_eq!(results[2].signatures, results[3].signatures);
    // Reproducible, and independent of the plan tile.
    let again = run(gpu::Design::CoSplit, options);
    assert_eq!(again.energy, results[4].energy);
    assert_eq!(again.detector, results[4].detector);
    let retiled = run(
        gpu::Design::GpuHead,
        gpu::TransportOptions {
            source_tile: 700,
            event_budget: 5,
            ..options
        },
    );
    assert_eq!(retiled.energy, results[2].energy);
    assert_eq!(retiled.detector, results[2].detector);
    assert_eq!(retiled.signatures, results[2].signatures);
    // A CPU share of everything is the CPU reference, bit for bit; a device
    // share of everything is GpuHead.
    let all_cpu = run(
        gpu::Design::CoSplit,
        gpu::TransportOptions {
            gpu_share: Some(0.),
            ..options
        },
    );
    assert_eq!(all_cpu.energy, cpu.energy);
    assert_eq!(all_cpu.detector, cpu.detector);
    assert_eq!(all_cpu.signatures, cpu.signatures);
    let all_gpu = run(
        gpu::Design::CoSplit,
        gpu::TransportOptions {
            gpu_share: Some(1.),
            ..options
        },
    );
    assert_eq!(all_gpu.energy, results[2].energy);
    assert_eq!(all_gpu.detector, results[2].detector);
}

/// The CPU continues sources from the device stacks after any head budget:
/// the event count and the physics are those of the full transport.
#[test]
fn head_budget_continuation() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let sys = mklight_core::samples::cooke_triplet();
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(Arc::new(common::resolve(sys)), &Default::default()).unwrap();
    let source = system.system_field_source(2, 0.1, 1.).unwrap().spec;
    for mode in [
        SplitMode::Analog,
        SplitMode::Weighted { rare_floor: 0.02 },
        SplitMode::Split {
            max_splits: 3,
            rare_floor: 0.02,
        },
    ] {
        let settings = StraySettings {
            rays: 3000,
            seed: 5,
            mode,
            ..Default::default()
        };
        let cpu =
            mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
        let full = gpu::run_design(
            &b,
            snapshot.clone(),
            &system.scene,
            &source,
            &settings,
            gpu::TransportOptions::default(),
            gpu::Design::CpuTally,
            &JobContext::new(),
        )
        .unwrap();
        for head in [1, 3, 9] {
            let got = gpu::run_design(
                &b,
                snapshot.clone(),
                &system.scene,
                &source,
                &settings,
                gpu::TransportOptions {
                    head_events: head,
                    ..Default::default()
                },
                gpu::Design::Wavefront,
                &JobContext::new(),
            )
            .unwrap();
            compare_results(&cpu, &got);
            // Same draws on both sides of the handoff: the same paths up to
            // the rounding of a branch decision.
            let events = |r: &mklight_stray::StrayResult| r.energy.events as f64;
            assert!(
                (events(&got) - events(&full)).abs() <= 1e-3 * events(&full),
                "{mode:?} head {head}: {} vs {}",
                events(&got),
                events(&full)
            );
            // Without splitting a rounding-level branch flip moves one
            // path; a split tree can move a heavy subtree (compare_results
            // covers that case statistically).
            if matches!(mode, SplitMode::Split { .. }) {
                continue;
            }
            for c in 0..5 {
                let (a, b) = (
                    got.energy.detected_by_class[c].1,
                    full.energy.detected_by_class[c].1,
                );
                let se = full.energy.detected_std_error_by_class[c].1;
                assert!(
                    (a - b).abs() <= 0.1 * se + 1e-12 * cpu.energy.emitted,
                    "{mode:?} head {head} class {c}: {a} {b}"
                );
            }
        }
    }
}

/// The hybrid analysis checks the system fields for obstruction beside the
/// device transport: the same obstruction, warnings and reference as the
/// all-device design, which checks them first.
#[test]
fn deferred_checks_match() {
    let b = MetalBackend::new().unwrap().expect("actual GPU required");
    let (sys, opts) = common::double_gauss_with_barrels();
    let rs = Arc::new(common::resolve(sys));
    let settings = mklight_stray::StrayAnalysisSettings {
        rays: 3000,
        scene: opts.scene.clone(),
        source: mklight_stray::StraySource::SystemField {
            index: 1,
            margin: 0.1,
            strength: 1.,
        },
        ..Default::default()
    };
    let analyze = |design| {
        let ctx = JobContext::new();
        let r = gpu::analyze_design(
            &b,
            rs.clone(),
            &settings,
            gpu::TransportOptions::default(),
            design,
            &ctx,
        )
        .unwrap();
        assert_eq!(ctx.progress.fraction(), 1.);
        r
    };
    let legacy = analyze(gpu::Design::GpuTally);
    let hybrid = analyze(gpu::Design::CoSplit);
    assert_eq!(hybrid.primary_obstruction, legacy.primary_obstruction);
    assert_eq!(hybrid.warnings, legacy.warnings);
    assert_eq!(hybrid.clear_semi_diameters, legacy.clear_semi_diameters);
}
