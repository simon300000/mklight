//! The ghost trace kernel designs (work-split study key `ghost_kernel`)
//! give identical results, with and without thin films. The selection is
//! process-wide, so this test has a process of its own.
mod common;
use common::*;
use mklight_core::{JobContext, samples, split};
use mklight_ghost::*;

#[cfg(target_os = "macos")]
#[test]
fn kernel_designs_give_identical_results() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let mut filmed = samples::cooke_triplet();
    filmed
        .coatings
        .insert("film".into(), quarter_wave(1.38, D_LINE));
    for s in &mut filmed.surfaces[1..] {
        s.coating = Some("film".into());
    }
    for s in [samples::double_gauss(), filmed] {
        let m = model(s, GhostSettings::default());
        let settings = IrradianceSettings {
            pupil_samples: 12,
            map_bins: [16, 16],
            keep_path_maps: true,
            ..Default::default()
        };
        let mut reference: Option<GhostIrradiance> = None;
        // The kernel designs, and a thread budget for the CPU share of the
        // shared design, change no result.
        for (key, kernel) in [
            ("ghost_kernel", "specialized"),
            ("ghost_kernel", "partial"),
            ("ghost_kernel", "uber"),
            ("ghost_threads", "3"),
        ] {
            split::set(key, Some(kernel));
            let got = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
            split::reset();
            match &reference {
                None => reference = Some(got),
                Some(r) => {
                    assert_eq!(r.total_ghost_map, got.total_ghost_map, "{kernel}");
                    assert_eq!(r.primary, got.primary, "{kernel}");
                    assert_eq!(r.ghosts, got.ghosts, "{kernel}");
                }
            }
        }
    }
}
