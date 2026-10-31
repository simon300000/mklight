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

#[test]
fn default_pupil_quadrature_matches_finer_camera_reference() {
    for system in [
        samples::double_gauss(),
        samples::cooke_triplet(),
        samples::singlet(),
    ] {
        let t = mklight_analysis::tracer_for(system).unwrap();
        let psf_settings = analysis::PsfAnalysisSettings {
            grid_size: 96,
            spacing_mm: Some(0.001),
            ..Default::default()
        };
        let psf = analysis::psf_analysis(&t, &psf_settings).unwrap();
        let mut fine_psf = psf_settings.clone();
        fine_psf.pupil.supersample = 8;
        let reference_psf = analysis::psf_analysis(&t, &fine_psf).unwrap();
        for (a, b) in psf.psfs.iter().zip(&reference_psf.psfs) {
            for (x, y) in a.psf.data.iter().zip(&b.psf.data) {
                assert!((x - y).abs() < 2e-3, "PSF {x} vs {y}");
            }
        }
        let energy_settings = analysis::EncircledEnergySettings {
            grid_size: 128,
            max_radius_mm: Some(0.03),
            ..Default::default()
        };
        let energy = analysis::diffraction_encircled_energy(&t, &energy_settings).unwrap();
        let mut fine_energy = energy_settings.clone();
        fine_energy.pupil.supersample = 8;
        let reference_energy = analysis::diffraction_encircled_energy(&t, &fine_energy).unwrap();
        for (a, b) in energy.fields.iter().zip(&reference_energy.fields) {
            for (x, y) in a
                .encircled
                .iter()
                .chain(&a.ensquared)
                .zip(b.encircled.iter().chain(&b.ensquared))
            {
                assert!((x - y).abs() < 2e-3, "energy {x} vs {y}");
            }
        }
        let settings = analysis::ThroughFocusSettings {
            frequencies: vec![10.0, 20.0, 40.0],
            focus_range_mm: Some(0.15),
            focus_samples: 7,
            ..Default::default()
        };
        let coarse = analysis::through_focus_mtf_analysis(&t, &settings).unwrap();
        let mut fine = settings.clone();
        fine.pupil.supersample = 8;
        let reference = analysis::through_focus_mtf_analysis(&t, &fine).unwrap();
        for (a, b) in coarse.fields.iter().zip(&reference.fields) {
            for (a, b) in a.curves.iter().zip(&b.curves) {
                for (x, y) in a
                    .tangential
                    .iter()
                    .chain(&a.sagittal)
                    .zip(b.tangential.iter().chain(&b.sagittal))
                {
                    assert!((x - y).abs() < 2e-3, "{x} vs {y}");
                }
            }
        }
    }
}
