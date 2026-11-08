//! The transport kernel designs (work-split study key `stray_kernel`) give
//! identical device records. The selection is process-wide, so this test
//! has a process of its own.
#![cfg(target_os = "macos")]
#[path = "../common/mod.rs"]
mod common;
use mklight_core::{JobContext, Snapshot, model::ScatterModel, split};
use mklight_metal::MetalBackend;
use mklight_stray::{SplitMode, StraySettings, StraySystem, gpu};
use std::sync::Arc;

#[test]
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
