//! The transport kernel designs (work-split study key `stray_kernel`) give
//! identical device records; so does a CPU-share thread budget
//! (`stray_threads`); the extended-sag intersection modes (`stray_sag`)
//! agree with the reference. The selections are process-wide, so these
//! checks run in one test, in a process of their own.
#![cfg(target_os = "macos")]
#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, Snapshot, model::ScatterModel, split};
use mklight_metal::MetalBackend;
use mklight_stray::{SplitMode, StraySettings, StraySystem, gpu};
use std::sync::Arc;

#[test]
fn device_variants() {
    kernel_designs_give_identical_records();
    sag_modes_and_thread_budgets();
}

fn sag_modes_and_thread_budgets() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let mut handoffs = 0;
    for sys in [
        mklight_core::samples::petzval_asphere(),
        mklight_core::samples::odd_asphere_element(),
        mklight_core::samples::q_asphere_lens(),
        mklight_core::samples::zernike_sag(),
    ] {
        let name = sys.name.clone();
        let snapshot = Snapshot::new(sys.clone()).unwrap();
        let system = StraySystem::new(Arc::new(common::resolve(sys)), &Default::default()).unwrap();
        let field = system.resolved().system().fields.fields.len() - 1;
        let source = system.system_field_source(field, 0.1, 1.).unwrap().spec;
        let settings = StraySettings {
            rays: 3000,
            seed: 23,
            ..Default::default()
        };
        let cpu =
            mklight_stray::run(&system.scene, &source, &settings, &JobContext::new()).unwrap();
        let run = |sets: &[(&str, &str)], design| {
            for (k, v) in sets {
                split::set(k, Some(v));
            }
            let r = gpu::run_design(
                &b,
                snapshot.clone(),
                &system.scene,
                &source,
                &settings,
                gpu::TransportOptions {
                    head_events: 0,
                    gpu_share: Some(0.5),
                    ..Default::default()
                },
                design,
                &JobContext::new(),
            )
            .unwrap();
            split::reset();
            assert_eq!(r.run_info.as_ref().unwrap().settings["fallback_count"], 0);
            r
        };
        let scan = run(&[], gpu::Design::Wavefront);
        // The f32 pre-scan only skips cells the double-float scan finds
        // empty: the same roots, so the same records.
        let fast = run(&[("stray_sag", "f32-scan")], gpu::Design::Wavefront);
        assert_eq!(scan.energy, fast.energy, "{name}");
        assert_eq!(scan.detector, fast.detector, "{name}");
        let persistent = run(&[("stray_sag", "f32-scan")], gpu::Design::CpuTally);
        assert_eq!(scan.energy, persistent.energy, "{name}");
        // Long scans continue on the CPU: the reference's physics.
        for design in [gpu::Design::Wavefront, gpu::Design::CpuTally] {
            let handed = run(&[("stray_sag", "cpu-grazing")], design);
            let n = &handed.run_info.as_ref().unwrap().settings["counters"];
            handoffs += n["sag_handoff"].as_u64().unwrap();
            compare(&cpu, &handed);
        }
        // A thread budget for the CPU share changes no record.
        let shared = run(&[], gpu::Design::CoSplit);
        let budget = run(&[("stray_threads", "3")], gpu::Design::CoSplit);
        assert_eq!(shared.energy, budget.energy, "{name}");
        assert_eq!(shared.detector, budget.detector, "{name}");
    }
    // Short sample runs rarely scan longer than the limit; the continuation
    // itself is covered by the handoff tests of p10_transport.
    eprintln!("sag handoffs: {handoffs}");
}

/// Agreement within Monte-Carlo error of the class totals.
fn compare(a: &mklight_stray::StrayResult, b: &mklight_stray::StrayResult) {
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
            (av - bv).abs() <= 5. * se + 1e-8 * a.energy.emitted,
            "class {c}: {av} vs {bv} SE={se}"
        );
    }
}

fn kernel_designs_give_identical_records() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("actual device required");
    let (sys, mut opts) = common::double_gauss_with_barrels();
    opts.scene.default_edge_finish = ScatterModel::Abg {
        a: 0.01,
        b: 0.001,
        g: 1.5,
    };
    let snapshot = Snapshot::new(sys.clone()).unwrap();
    let system = StraySystem::new(Arc::new(common::resolve(sys)), &opts).unwrap();
    let source = system.system_field_source(1, 0.1, 1.).unwrap().spec;
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
            seed: 11,
            mode,
            ..Default::default()
        };
        let mut reference: Option<mklight_stray::StrayResult> = None;
        for kernel in ["specialized", "partial", "uber"] {
            for design in [gpu::Design::CpuTally, gpu::Design::Wavefront] {
                split::set("stray_kernel", Some(kernel));
                let got = gpu::run_design(
                    &b,
                    snapshot.clone(),
                    &system.scene,
                    &source,
                    &settings,
                    gpu::TransportOptions {
                        head_events: 0,
                        ..Default::default()
                    },
                    design,
                    &JobContext::new(),
                )
                .unwrap();
                split::reset();
                assert_eq!(got.run_info.as_ref().unwrap().settings["fallback_count"], 0);
                match &reference {
                    None => reference = Some(got),
                    Some(r) => {
                        assert_eq!(r.energy, got.energy, "{mode:?} {kernel} {design:?}");
                        assert_eq!(r.detector, got.detector, "{mode:?} {kernel} {design:?}");
                        assert_eq!(r.signatures, got.signatures, "{mode:?} {kernel}");
                    }
                }
            }
        }
    }
}
