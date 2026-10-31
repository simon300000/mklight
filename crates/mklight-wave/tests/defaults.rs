//! Defaults must be useful without hiding the diagnostics for explicit settings.
use mklight_core::{JobContext, samples};
use mklight_wave::{analysis, image_simulation};

fn accuracy_warning(w: &str) -> bool {
    [
        "between pupil samples",
        "PSF is aliased",
        "wraps around",
        "PSF is truncated",
        "wavefront there is extrapolated",
        "ill-conditioned",
        "beyond the sampled range",
        "only ",
        "have no ray through the image point",
    ]
    .iter()
    .any(|s| w.contains(s))
}

#[test]
fn every_analysis_default_on_every_sample() {
    let mut failures = Vec::new();
    for (name, system) in samples::all() {
        let t = mklight_analysis::tracer_for(system).unwrap();
        for a in analysis::ANALYSES {
            let r = (a.run)(&t, &serde_json::json!({}), &JobContext::new()).unwrap();
            for w in r["run_info"]["warnings"].as_array().unwrap() {
                let w = w.as_str().unwrap();
                if accuracy_warning(w) {
                    failures.push(format!("{name}/{}: {w}", a.name));
                }
            }
        }
        let r = mklight_analysis::field_maps::field_map_job(
            &t,
            mklight_analysis::field_maps::FieldAnalysis::RelativeIllumination,
            &Default::default(),
            &JobContext::new(),
        )
        .unwrap();
        failures.extend(
            r.run_info
                .warnings
                .iter()
                .filter(|w| accuracy_warning(w))
                .map(|w| format!("{name}/relative_illumination_map: {w}")),
        );
        let r = image_simulation::image_simulation_job(
            &t,
            &Default::default(),
            None,
            &JobContext::new(),
        )
        .unwrap();
        failures.extend(
            r.run_info
                .warnings
                .iter()
                .filter(|w| accuracy_warning(w))
                .map(|w| format!("{name}/image_simulation: {w}")),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn ideal_cone_sets_frequency_and_depth_and_explicit_settings_win() {
    use mklight_core::{
        ApertureSpec, Field, MaterialRef, OpticalSystem, Surface, SurfaceShape, Wavelength,
    };
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.fields.fields = vec![Field::new(0.0, 0.0)];
    s.wavelengths.wavelengths = vec![Wavelength {
        um: 0.55,
        weight: 1.0,
    }];
    s.wavelengths.primary = 0;
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial {
                focal_length: 100.0,
            },
            100.0,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    let t = mklight_analysis::tracer_for(s).unwrap();
    let n = (10000.0f64 + 25.0).sqrt() / 10.0;
    let mtf = analysis::mtf_analysis(&t, &Default::default()).unwrap();
    assert!((mtf.frequencies.last().unwrap() * 0.55e-3 * n - 0.25).abs() < 1e-8);
    let focus = analysis::through_focus_mtf_analysis(&t, &Default::default()).unwrap();
    assert!((focus.focus_mm.last().unwrap() / (8.0 * 0.55e-3 * n * n) - 1.0).abs() < 1e-8);
    assert_eq!(focus.frequencies, [5.0, 10.0, 20.0]);
    let explicit = analysis::ThroughFocusSettings {
        frequencies: vec![7.0, 19.0],
        focus_range_mm: Some(0.07),
        ..Default::default()
    };
    let focus = analysis::through_focus_mtf_analysis(&t, &explicit).unwrap();
    assert_eq!(focus.frequencies, [7.0, 19.0]);
    assert!((focus.focus_mm.last().unwrap() - 0.07).abs() < 1e-14);
    let psf = analysis::psf_analysis(
        &t,
        &analysis::PsfAnalysisSettings {
            grid_size: 31,
            spacing_mm: Some(0.001),
            pupil: mklight_wave::PupilSettings {
                grid: 32,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(psf.psfs[0].psf.size, 31);
    assert_eq!(psf.psfs[0].psf.spacing_mm, 0.001);
}
