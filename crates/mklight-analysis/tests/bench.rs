//! Throughput benchmark (ignored by default):
//! `cargo test --release -p mklight-analysis --test tests spot_diagram_throughput -- --ignored --nocapture`.

use crate::common;

use std::time::Instant;

use mklight_analysis::spot::{SpotDiagramSettings, spot_diagram};
use mklight_analysis::util::PupilPattern;

#[test]
#[ignore]
fn spot_diagram_throughput() {
    let t = common::tracer(mklight_core::samples::double_gauss());
    let set = SpotDiagramSettings {
        pattern: PupilPattern::Hexapolar,
        rings: 60,
        include_points: false,
        ..Default::default()
    };
    // Warm-up, then timed runs.
    let s = spot_diagram(&t, &set).unwrap();
    let rays: usize = s.fields.iter().map(|f| f.polychromatic.rays_launched).sum();
    let runs = 5;
    let t0 = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(spot_diagram(&t, &set).unwrap());
    }
    let dt = t0.elapsed().as_secs_f64() / runs as f64;
    println!(
        "spot diagram: {rays} rays in {:.2} ms → {:.2} Mrays/s ({} threads)",
        dt * 1e3,
        rays as f64 / dt / 1e6,
        rayon::current_num_threads()
    );
    assert_eq!(rays, 9 * (1 + 3 * 60 * 61));
}
