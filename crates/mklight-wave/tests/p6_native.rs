//! Complete native-result qualification at unchanged default sampling.
#![cfg(target_os = "macos")]
use mklight_core::{JobContext, samples};
use mklight_gpu::ComputeBackend;
use mklight_wave::{analysis as cpu, gpu, image_simulation as image};
fn backend() -> mklight_metal::MetalBackend {
    let library = if std::env::var("P6_LIBRARY").as_deref() == Ok("runtime") {
        mklight_metal::LibraryChoice::RuntimeSource
    } else {
        mklight_metal::LibraryChoice::Precompiled
    };
    let b = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
        library,
        ..Default::default()
    })
    .unwrap()
    .expect("P6 qualification requires host Metal");
    assert!(b.precision().probe.error_free_transforms());
    eprintln!(
        "P6_NATIVE_DEVICE {} {:?}",
        b.info().backend_id(),
        b.precision()
    );
    b
}
fn compare(a: &[f64], b: &[f64], limit: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f64;
    for (x, y) in a.iter().zip(b) {
        if x.is_nan() {
            assert!(y.is_nan());
            continue;
        }
        assert!(y.is_finite());
        max = max.max((x - y).abs());
    }
    assert!(max <= limit, "max error {max:e} > {limit:e}");
    max
}
fn mtf(a: &cpu::WaveMtfResult, b: &cpu::WaveMtfResult) -> f64 {
    compare(&a.frequencies, &b.frequencies, 1e-9);
    assert_eq!(a.fields.len(), b.fields.len());
    let mut max = 0f64;
    for (a, b) in a.fields.iter().zip(&b.fields) {
        assert_eq!(a.pupil_sampling, b.pupil_sampling);
        max = max
            .max(compare(&a.tangential, &b.tangential, 2e-3))
            .max(compare(&a.sagittal, &b.sagittal, 2e-3))
            .max(compare(&a.diffraction_limit, &b.diffraction_limit, 2e-3));
    }
    max
}
#[test]
#[ignore = "complete host qualification across all 16 native default analyses"]
fn all_default_native_analyses() {
    let b = backend();
    let ctx = JobContext::new();
    let mut maxima = [0f64; 8];
    let mut cases = 0;
    for (name, model) in samples::all() {
        if std::env::var("P6_SAMPLE").is_ok_and(|v| v != name) {
            continue;
        }
        let t = mklight_analysis::tracer_for(model).unwrap();
        let s = cpu::WavefrontSettings::default();
        let a = cpu::wavefront_map_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::wavefront_map_job(&b, &t, &s, &ctx).unwrap();
        assert_eq!(a.maps.len(), z.maps.len());
        for (a, z) in a.maps.iter().zip(&z.maps) {
            assert_eq!(a.grid, z.grid);
            maxima[0] = maxima[0].max(compare(&a.opd_waves, &z.opd_waves, 5e-3));
            compare(
                &[
                    a.rms_waves,
                    a.rms_waves_tilt_removed,
                    a.zernike_residual_rms,
                ],
                &[
                    z.rms_waves,
                    z.rms_waves_tilt_removed,
                    z.zernike_residual_rms,
                ],
                5e-3,
            );
            maxima[1] = maxima[1].max((a.strehl_at_reference - z.strehl_at_reference).abs());
            assert!((a.strehl_at_reference - z.strehl_at_reference).abs() < 2e-3);
        }
        eprintln!("P6_NATIVE {name} wavefront passed");
        let s = cpu::PsfAnalysisSettings::default();
        let a = cpu::psf_analysis_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::psf_analysis_job(&b, &t, &s, &ctx).unwrap();
        assert_eq!(a.psfs.len(), z.psfs.len());
        for (a, z) in a.psfs.iter().zip(&z.psfs) {
            assert_eq!(a.psf.size, z.psf.size);
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            compare(&[a.psf.spacing_mm], &[z.psf.spacing_mm], 1e-9);
            let sum = a.psf.data.iter().sum::<f64>();
            let l1 = a
                .psf
                .data
                .iter()
                .zip(&z.psf.data)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / sum;
            assert!(l1 < 2e-3, "PSF {name} L1 {l1}");
            maxima[2] = maxima[2].max(l1);
            assert!(
                (a.psf.energy_in_grid() - z.psf.energy_in_grid()).abs() < 1e-3 * a.psf.total_energy
            );
            compare(&a.centroid_mm, &z.centroid_mm, 2e-5);
        }
        eprintln!("P6_NATIVE {name} PSF passed");
        for method in [cpu::MtfAnalysisMethod::Direct, cpu::MtfAnalysisMethod::Fft] {
            let s = cpu::MtfSettings {
                method,
                ..Default::default()
            };
            let a = cpu::mtf_analysis_job(&t, &s, &ctx).unwrap();
            let z = gpu::analysis::mtf_analysis_job(&b, &t, &s, &ctx).unwrap();
            maxima[3] = maxima[3].max(mtf(&a, &z));
        }
        eprintln!("P6_NATIVE {name} direct/FFT MTF passed");
        let s = cpu::MtfVsFieldSettings::default();
        let a = cpu::mtf_vs_field_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::mtf_vs_field_job(&b, &t, &s, &ctx).unwrap();
        assert_eq!(a.pupil_sampling, z.pupil_sampling);
        compare(&a.frequencies, &z.frequencies, 1e-9);
        for (a, z) in a.curves.iter().zip(&z.curves) {
            maxima[4] = maxima[4]
                .max(compare(&a.tangential, &z.tangential, 2e-3))
                .max(compare(&a.sagittal, &z.sagittal, 2e-3));
        }
        eprintln!("P6_NATIVE {name} field MTF passed");
        let s = cpu::ThroughFocusSettings::default();
        let a = cpu::through_focus_mtf_analysis_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::through_focus_mtf_analysis_job(&b, &t, &s, &ctx).unwrap();
        compare(&a.frequencies, &z.frequencies, 1e-9);
        compare(&a.focus_mm, &z.focus_mm, 1e-5);
        for (a, z) in a.fields.iter().zip(&z.fields) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            for (a, z) in a.curves.iter().zip(&z.curves) {
                maxima[5] = maxima[5]
                    .max(compare(&a.tangential, &z.tangential, 2e-3))
                    .max(compare(&a.sagittal, &z.sagittal, 2e-3));
            }
        }
        eprintln!("P6_NATIVE {name} focus MTF passed");
        let s = cpu::EncircledEnergySettings::default();
        let a = cpu::diffraction_encircled_energy_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::diffraction_encircled_energy_job(&b, &t, &s, &ctx).unwrap();
        compare(&a.radius_mm, &z.radius_mm, 1e-8);
        for (a, z) in a.fields.iter().zip(&z.fields) {
            assert_eq!(a.grid_size, z.grid_size);
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            maxima[6] = maxima[6]
                .max(compare(&a.encircled, &z.encircled, 2e-3))
                .max(compare(&a.ensquared, &z.ensquared, 2e-3))
                .max(compare(&a.diffraction_limit, &z.diffraction_limit, 2e-3));
            compare(
                &[a.r50_mm, a.r80_mm, a.r90_mm],
                &[z.r50_mm, z.r80_mm, z.r90_mm],
                s.max_radius_mm
                    .unwrap_or(a.grid_size as f64 * a.spacing_mm * 0.5)
                    / (s.samples * 8 - 1) as f64,
            );
        }
        eprintln!("P6_NATIVE {name} energy passed");
        let s = cpu::LineSpreadSettings::default();
        let a = cpu::line_spread_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::line_spread_job(&b, &t, &s, &ctx).unwrap();
        for (a, z) in a.fields.iter().zip(&z.fields) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            compare(&a.positions_mm, &z.positions_mm, 1e-8);
            let peak = a.x.iter().chain(&a.y).copied().fold(0f64, f64::max);
            compare(&a.x, &z.x, 2e-3 * peak);
            compare(&a.y, &z.y, 2e-3 * peak);
        }
        eprintln!("P6_NATIVE {name} LSF passed");
        let s = image::ImageSimulationSettings::default();
        let a = image::image_simulation_job(&t, &s, None, &ctx).unwrap();
        let z = gpu::image_simulation::image_simulation_job(&b, &t, &s, None, &ctx).unwrap();
        assert_eq!(a.psf_size, z.psf_size);
        assert_eq!(a.field_mapping.len(), z.field_mapping.len());
        for (a, z) in a.field_mapping.iter().zip(&z.field_mapping) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
        let sum = a.image.iter().sum::<f64>();
        let l1 = a
            .image
            .iter()
            .zip(&z.image)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / sum;
        maxima[7] = maxima[7].max(l1);
        assert!(l1 < 2e-3, "image {name}: {l1}");
        assert!((sum - z.image.iter().sum::<f64>()).abs() / sum < 1e-3);
        cases += 1;
        eprintln!("P6_NATIVE {name} image passed maxima={maxima:?}");
    }
    eprintln!("P6_NATIVE_SUMMARY samples={cases} maxima={maxima:?}");
}

#[test]
fn diffraction_reference_preserves_lateral_colour() {
    let b = backend();
    let ctx = JobContext::new();
    let t = mklight_analysis::tracer_for(samples::achromatic_doublet()).unwrap();
    let s = cpu::EncircledEnergySettings::default();
    let a = cpu::diffraction_encircled_energy_job(&t, &s, &ctx).unwrap();
    let z = gpu::analysis::diffraction_encircled_energy_job(&b, &t, &s, &ctx).unwrap();
    for (a, z) in a.fields.iter().zip(z.fields) {
        let ee = compare(&a.encircled, &z.encircled, 2e-3);
        let sq = compare(&a.ensquared, &z.ensquared, 2e-3);
        let dl = compare(&a.diffraction_limit, &z.diffraction_limit, 2e-3);
        eprintln!(
            "P6_ENERGY_COLOUR field={} EE={ee:e} square={sq:e} DL={dl:e}",
            a.field
        );
    }
}

#[test]
fn native_huygens_and_explicit_focus_diagnostics() {
    let b = backend();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    let pupil = mklight_wave::PupilSettings {
        grid: 16,
        rays: 17,
        ..Default::default()
    };
    let fields = mklight_analysis::util::FieldSelection::Index { index: 2 };
    let s = cpu::MtfSettings {
        fields: fields.clone(),
        pupil,
        method: cpu::MtfAnalysisMethod::Huygens,
        huygens: cpu::HuygensGrid {
            grid_size: 32,
            padding: 2.,
        },
        frequency_samples: 9,
        max_frequency: Some(500.),
        ..Default::default()
    };
    let a = cpu::mtf_analysis_job(&t, &s, &ctx).unwrap();
    let z = gpu::analysis::mtf_analysis_job(&b, &t, &s, &ctx).unwrap();
    mtf(&a, &z);
    assert!(z.run_info.warnings.iter().any(|w| w.contains("PSF")));
    assert!(z.run_info.warnings.iter().any(|w| w.contains("frequency")));
    for method in [cpu::MtfAnalysisMethod::Fft, cpu::MtfAnalysisMethod::Huygens] {
        let s = cpu::ThroughFocusSettings {
            fields: fields.clone(),
            pupil,
            method,
            huygens: cpu::HuygensGrid {
                grid_size: 32,
                padding: 2.,
            },
            frequencies: vec![0., 40.],
            focus_samples: 5,
            focus_range_mm: Some(2.),
            ..Default::default()
        };
        let a = cpu::through_focus_mtf_analysis_job(&t, &s, &ctx).unwrap();
        let z = gpu::analysis::through_focus_mtf_analysis_job(&b, &t, &s, &ctx).unwrap();
        for (a, z) in a.fields.iter().zip(&z.fields) {
            for (a, z) in a.curves.iter().zip(&z.curves) {
                compare(&a.tangential, &z.tangential, 2e-3);
                compare(&a.sagittal, &z.sagittal, 2e-3);
            }
        }
        assert!(
            z.run_info
                .warnings
                .iter()
                .any(|w| w.contains("between pupil samples"))
        );
    }
    let s = cpu::LineSpreadSettings {
        fields,
        pupil,
        method: cpu::PsfMethod::Huygens,
        grid_size: 32,
        padding: 2.,
        ..Default::default()
    };
    let a = cpu::line_spread_job(&t, &s, &ctx).unwrap();
    let z = gpu::analysis::line_spread_job(&b, &t, &s, &ctx).unwrap();
    for (a, z) in a.fields.iter().zip(&z.fields) {
        let peak = a.x.iter().chain(&a.y).copied().fold(0f64, f64::max);
        compare(&a.x, &z.x, 2e-3 * peak);
        compare(&a.y, &z.y, 2e-3 * peak);
    }
    assert!(z.run_info.warnings.iter().any(|w| w.contains("PSF")));
}
