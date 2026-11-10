//! Throughput benchmark (ignored by default). Run with
//! `cargo test --release -p mklight-stray --test bench -- --ignored --nocapture`.

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::*;
use mklight_core::JobContext;
use mklight_stray::{StrayAnalysisSettings, StraySource, StraySystem, analyze_system};

#[test]
#[ignore]
fn bench_double_gauss_with_barrels() {
    let (s, opts) = double_gauss_with_barrels();
    let sys = StraySystem::new(Arc::new(resolve(s)), &opts).unwrap();
    println!(
        "scene: {} primitives, {} threads",
        sys.scene.primitives().len(),
        rayon::current_num_threads()
    );
    for (name, source) in [
        (
            "on-axis field",
            StraySource::SystemField {
                index: 0,
                margin: 0.1,
                strength: 1.0,
            },
        ),
        (
            "sun at 20°",
            StraySource::OutOfField {
                angle_x_deg: 0.0,
                angle_y_deg: 20.0,
                margin: 0.1,
                irradiance: 1.0,
            },
        ),
    ] {
        let settings = StrayAnalysisSettings {
            source,
            rays: 400_000,
            seed: 1,
            scene: opts.scene.clone(),
            ..Default::default()
        };
        let t = Instant::now();
        let r = analyze_system(&sys, &settings, &JobContext::new()).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "{name}: {} rays, {} events in {dt:.3} s: {:.3e} rays/s, {:.3e} events/s; stray {:.4e} ± {:.1e} of the reference flux",
            r.energy.rays,
            r.energy.events,
            r.energy.rays as f64 / dt,
            r.energy.events as f64 / dt,
            r.stray_fraction.unwrap_or(f64::NAN),
            r.stray_std_error.unwrap() / r.source.reference_flux,
        );
    }
}
