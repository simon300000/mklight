//! Release wall-time probe using the built-in camera designs.
use mklight_core::{JobContext, samples};
use mklight_wave::{analysis, image_simulation};
use std::time::Instant;

#[test]
#[ignore = "release performance measurement"]
fn default_camera_analyses() {
    for (name, system) in [
        ("singlet", samples::singlet()),
        ("double_gauss", samples::double_gauss()),
        ("cooke_triplet", samples::cooke_triplet()),
        ("petzval_asphere", samples::petzval_asphere()),
        ("anamorphic", samples::anamorphic_adapter()),
    ] {
        let t = mklight_analysis::tracer_for(system).unwrap();
        for a in analysis::ANALYSES {
            let start = Instant::now();
            let r = (a.run)(&t, &serde_json::json!({}), &JobContext::new()).unwrap();
            assert_eq!(r["run_info"]["analysis"], a.name);
            eprintln!("{name}/{}: {:.3} s", a.name, start.elapsed().as_secs_f64());
        }
        let start = Instant::now();
        image_simulation::image_simulation_job(&t, &Default::default(), None, &JobContext::new())
            .unwrap();
        eprintln!(
            "{name}/image_simulation: {:.3} s",
            start.elapsed().as_secs_f64()
        );
    }
}
