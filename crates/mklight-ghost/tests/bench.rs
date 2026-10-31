//! Run with `RAYON_NUM_THREADS=5 cargo test --release -p mklight-ghost
//! --test bench -- --ignored --nocapture`.

mod common;

use common::*;
use mklight_core::{JobContext, samples};
use mklight_ghost::{GhostSettings, IrradianceSettings, ghost_irradiance};

#[test]
#[ignore]
fn double_gauss_irradiance() {
    let model = model(samples::double_gauss(), GhostSettings::default());
    println!("{} threads", rayon::current_num_threads());
    for n in [48, 128] {
        let settings = IrradianceSettings {
            fields: vec![0, 2],
            wavelengths: vec![1],
            pupil_samples: n,
            keep_path_maps: true,
            ..Default::default()
        };
        let mut times = Vec::new();
        for iteration in 0..6 {
            let result = ghost_irradiance(&model, &settings, &JobContext::new()).unwrap();
            assert_eq!(result.ghosts.len(), 45);
            if iteration > 0 {
                times.push(result.run.duration_ms);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{n}², 45 ghosts + primary, fields 0/2, wavelength 1: median {:.3} ms; {times:?}",
            times[times.len() / 2]
        );
    }
}
