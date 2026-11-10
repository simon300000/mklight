//! The device's extended-sag intersections (`gpu::SagMode`) agree with the
//! reference on the aspheric samples.
#![cfg(target_os = "macos")]
#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, Snapshot};
use mklight_metal::MetalBackend;
use mklight_stray::{StraySettings, StraySystem, gpu};
use std::sync::Arc;

#[test]
fn sag_variants_agree_with_the_reference() {
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
        let run = |sag: gpu::SagMode| {
            let r = gpu::run(
                &b,
                snapshot.clone(),
                &system.scene,
                &source,
                &settings,
                gpu::TransportOptions {
                    head_events: 0,
                    gpu_share: Some(1.),
                    sag,
                    ..Default::default()
                },
                &JobContext::new(),
            )
            .unwrap();
            let info = &r.run_info.as_ref().unwrap().settings;
            assert_eq!(info["fallback_count"], 0);
            assert_eq!(info["sag_mode"], sag as u32);
            r
        };
        // Even aspheres in the radial double-float evaluation with Newton
        // polish, every other sag by its jets.
        compare(&cpu, &run(gpu::SagMode::F32Scan), &name);
        // Long scans continue on the CPU: the reference's physics.
        let handed = run(gpu::SagMode::F32Grazing);
        let n = &handed.run_info.as_ref().unwrap().settings["counters"];
        handoffs += n["sag_handoff"].as_u64().unwrap();
        compare(&cpu, &handed, &name);
    }
    // Short sample runs rarely scan longer than the limit; the continuation
    // itself is covered by the handoff tests of p10_transport.
    eprintln!("sag handoffs: {handoffs}");
}

/// Agreement within Monte-Carlo error of the class totals.
fn compare(a: &mklight_stray::StrayResult, b: &mklight_stray::StrayResult, name: &str) {
    assert!(
        b.energy.relative_residual() < 1e-5,
        "{name}: {}",
        b.energy.summary()
    );
    for c in 0..5 {
        let (av, bv) = (
            a.energy.detected_by_class[c].1,
            b.energy.detected_by_class[c].1,
        );
        let se = a.energy.detected_std_error_by_class[c]
            .1
            .unwrap()
            .hypot(b.energy.detected_std_error_by_class[c].1.unwrap());
        assert!(
            (av - bv).abs() <= 5. * se + 1e-8 * a.energy.emitted,
            "{name} class {c}: {av} vs {bv} SE={se}"
        );
    }
}
