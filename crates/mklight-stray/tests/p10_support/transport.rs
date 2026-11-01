#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_stray::{SplitMode, StraySettings, StraySystem, gpu};
use std::sync::Arc;
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
            let got = gpu::run(
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
            assert!((got.energy.emitted - cpu.energy.emitted).abs() < 1e-8);
            assert!(got.energy.is_balanced(), "{}", got.energy.summary());
            for (a, c) in got
                .energy
                .detected_by_class
                .iter()
                .map(|v| v.1)
                .zip(cpu.energy.detected_by_class.iter().map(|v| v.1))
            {
                assert!((a - c).abs() < 1e-7, "{a} != {c}");
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
        r.outputs
            .iter()
            .map(|h| b.read_view(h.view()).unwrap().words().to_vec())
            .collect::<Vec<_>>()
    };
    let a = raw(&cpu, 7);
    let b = raw(&metal, 7);
    for (k, (a, b)) in a.iter().zip(&b).enumerate() {
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a, b, "output={k} word={i}");
        }
    }
    let c = raw(&metal, 64);
    for (k, (b, c)) in b.iter().zip(&c).enumerate() {
        for (i, (b, c)) in b.iter().zip(c).enumerate() {
            assert_eq!(b, c, "tile output={k} word={i}");
        }
    }
}
