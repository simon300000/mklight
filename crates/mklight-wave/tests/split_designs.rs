//! Every work-split design of the native wave analyses (`gpu::split`)
//! against the CPU f64 reference on built-in samples. Designs whose host
//! traces and resamples the pupils make the reference's own sampling
//! decisions, so their pupil grids, image grids and frequencies match it
//! exactly; designs whose host traces fit the reference's Zernike set.
#![cfg(target_os = "macos")]
use mklight_core::{JobContext, samples, split};
use mklight_wave::{analysis as cpu, gpu, gpu::Design, image_simulation as image};

fn backend() -> mklight_metal::MetalBackend {
    mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("the work-split designs need host Metal")
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

/// Run `f` with the design selected (the selection is process-wide, so
/// one test walks every design in turn).
fn with<R>(design: Design, f: impl FnOnce() -> R) -> R {
    split::set(gpu::split::KEY, Some(design.name()));
    let r = f();
    split::reset();
    r
}

fn check_sample(b: &mklight_metal::MetalBackend, name: &str, design: Design) {
    let model = samples::all()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1;
    let t = mklight_analysis::tracer_for(model).unwrap();
    let ctx = JobContext::new();
    let exact = design.host_pupil();
    let evidence = |info: &mklight_core::RunInfo| {
        assert_eq!(info.settings["work_split"]["design"], design.name());
    };

    let s = cpu::PsfAnalysisSettings::default();
    let a = cpu::psf_analysis_job(&t, &s, &ctx).unwrap();
    let z = with(design, || gpu::analysis::psf_analysis_job(b, &t, &s, &ctx)).unwrap();
    evidence(&z.run_info);
    assert_eq!(a.psfs.len(), z.psfs.len());
    for (a, z) in a.psfs.iter().zip(&z.psfs) {
        if exact {
            assert_eq!(a.psf.size, z.psf.size);
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
        if a.psf.size == z.psf.size {
            let sum = a.psf.data.iter().sum::<f64>();
            let l1 = a
                .psf
                .data
                .iter()
                .zip(&z.psf.data)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / sum;
            assert!(l1 < 2e-3, "{design:?} PSF {name} L1 {l1}");
        }
        compare(&a.centroid_mm, &z.centroid_mm, 2e-5);
        compare(&[a.strehl], &[z.strehl], 2e-3);
    }

    let s = cpu::MtfSettings::default();
    let a = cpu::mtf_analysis_job(&t, &s, &ctx).unwrap();
    let z = with(design, || gpu::analysis::mtf_analysis_job(b, &t, &s, &ctx)).unwrap();
    evidence(&z.run_info);
    compare(
        &a.frequencies,
        &z.frequencies,
        if exact { 0. } else { 1e-9 },
    );
    for (a, z) in a.fields.iter().zip(&z.fields) {
        if exact {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
        compare(&a.tangential, &z.tangential, 2e-3);
        compare(&a.sagittal, &z.sagittal, 2e-3);
        compare(&a.diffraction_limit, &z.diffraction_limit, 2e-3);
    }

    let s = cpu::MtfVsFieldSettings::default();
    let a = cpu::mtf_vs_field_job(&t, &s, &ctx).unwrap();
    let z = with(design, || gpu::analysis::mtf_vs_field_job(b, &t, &s, &ctx)).unwrap();
    evidence(&z.run_info);
    if exact {
        assert_eq!(a.pupil_sampling, z.pupil_sampling);
        assert_eq!(a.frequencies, z.frequencies);
    }
    for (a, z) in a.curves.iter().zip(&z.curves) {
        compare(&a.tangential, &z.tangential, 2e-3);
        compare(&a.sagittal, &z.sagittal, 2e-3);
    }

    let s = cpu::ThroughFocusSettings::default();
    let a = cpu::through_focus_mtf_analysis_job(&t, &s, &ctx).unwrap();
    let z = with(design, || {
        gpu::analysis::through_focus_mtf_analysis_job(b, &t, &s, &ctx)
    })
    .unwrap();
    evidence(&z.run_info);
    compare(&a.focus_mm, &z.focus_mm, if exact { 0. } else { 1e-5 });
    for (a, z) in a.fields.iter().zip(&z.fields) {
        if exact {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
        for (a, z) in a.curves.iter().zip(&z.curves) {
            compare(&a.tangential, &z.tangential, 2e-3);
            compare(&a.sagittal, &z.sagittal, 2e-3);
        }
    }

    let s = cpu::EncircledEnergySettings::default();
    let a = cpu::diffraction_encircled_energy_job(&t, &s, &ctx).unwrap();
    let z = with(design, || {
        gpu::analysis::diffraction_encircled_energy_job(b, &t, &s, &ctx)
    })
    .unwrap();
    evidence(&z.run_info);
    for (a, z) in a.fields.iter().zip(&z.fields) {
        if exact {
            assert_eq!(a.grid_size, z.grid_size);
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
        compare(&a.encircled, &z.encircled, 2e-3);
        compare(&a.ensquared, &z.ensquared, 2e-3);
        compare(&a.diffraction_limit, &z.diffraction_limit, 2e-3);
    }

    let s = cpu::LineSpreadSettings::default();
    let a = cpu::line_spread_job(&t, &s, &ctx).unwrap();
    let z = with(design, || gpu::analysis::line_spread_job(b, &t, &s, &ctx)).unwrap();
    evidence(&z.run_info);
    for (a, z) in a.fields.iter().zip(&z.fields) {
        if exact {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            compare(&a.positions_mm, &z.positions_mm, 1e-12);
        }
        if a.x.len() == z.x.len() {
            let peak = a.x.iter().chain(&a.y).copied().fold(0f64, f64::max);
            compare(&a.x, &z.x, 2e-3 * peak);
            compare(&a.y, &z.y, 2e-3 * peak);
        }
    }

    let s = cpu::WavefrontSettings::default();
    let a = cpu::wavefront_map_job(&t, &s, &ctx).unwrap();
    let z = with(design, || gpu::analysis::wavefront_map_job(b, &t, &s, &ctx)).unwrap();
    evidence(&z.run_info);
    assert_eq!(a.maps.len(), z.maps.len());
    for (a, z) in a.maps.iter().zip(&z.maps) {
        assert_eq!(a.grid, z.grid);
        compare(&a.opd_waves, &z.opd_waves, 5e-3);
        compare(&[a.rms_waves], &[z.rms_waves], 5e-3);
        if design.host_lattice() {
            // The reference's own fit of the same lattice.
            assert_eq!(a.zernike, z.zernike);
            assert_eq!(a.seidel, z.seidel);
        }
        assert!((a.strehl_at_reference - z.strehl_at_reference).abs() < 2e-3);
    }

    let s = image::ImageSimulationSettings::default();
    let a = image::image_simulation_job(&t, &s, None, &ctx).unwrap();
    let z = with(design, || {
        gpu::image_simulation::image_simulation_job(b, &t, &s, None, &ctx)
    })
    .unwrap();
    if exact {
        assert_eq!(a.psf_size, z.psf_size);
        for (a, z) in a.field_mapping.iter().zip(&z.field_mapping) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
    }
    if a.psf_size == z.psf_size {
        let sum = a.image.iter().sum::<f64>();
        let l1 = a
            .image
            .iter()
            .zip(&z.image)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / sum;
        assert!(l1 < 2e-3, "{design:?} image {name}: {l1}");
    }
    eprintln!("SPLIT {name} {} passed", design.name());
}

#[test]
#[ignore = "host Metal: every work-split design against the CPU reference"]
fn every_design_matches_the_reference() {
    let b = backend();
    for name in [
        "cooke_triplet",
        "double_gauss",
        "petzval_asphere",
        "singlet",
    ] {
        if std::env::var("SPLIT_SAMPLE").is_ok_and(|v| v != name) {
            continue;
        }
        for design in Design::ALL {
            check_sample(&b, name, design);
        }
    }
}

/// A focus sweep whose defocus phases exceed the device phase cache runs in
/// plane chunks (`mklight-metal` wave ops) with the same results as the
/// reference, on device-resampled and host-resampled pupils alike.
#[test]
#[ignore = "host Metal: chunked defocus phase cache"]
fn dense_focus_sweeps_match_the_reference() {
    use mklight_analysis::util::{FieldSelection, WavelengthSelection};
    let b = backend();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    // 1024² phases × 41 planes = 328 MiB: two chunks of 32 planes.
    let s = cpu::ThroughFocusSettings {
        fields: FieldSelection::Index { index: 2 },
        wavelengths: WavelengthSelection::Primary,
        focus_samples: 41,
        pupil: mklight_wave::PupilSettings {
            grid: 1024,
            ..Default::default()
        },
        ..Default::default()
    };
    let a = cpu::through_focus_mtf_analysis_job(&t, &s, &ctx).unwrap();
    for design in [Design::Resident, Design::Pipelined] {
        let z = with(design, || {
            gpu::analysis::through_focus_mtf_analysis_job(&b, &t, &s, &ctx)
        })
        .unwrap();
        compare(&a.focus_mm, &z.focus_mm, 1e-5);
        for (a, z) in a.fields.iter().zip(&z.fields) {
            for (a, z) in a.curves.iter().zip(&z.curves) {
                compare(&a.tangential, &z.tangential, 2e-3);
                compare(&a.sagittal, &z.sagittal, 2e-3);
            }
        }
    }
}

/// Host-traced designs share their lattices (and host-resampled pupils)
/// with the CPU analyses through the memo; results do not change.
#[test]
#[ignore = "host Metal: memo sharing of host lattices"]
fn host_lattices_are_shared_through_the_memo() {
    use mklight_core::memo::{Built, Memo, NodeKey, NodeValue};
    use std::collections::HashMap;
    use std::sync::Mutex;
    #[derive(Default)]
    struct Store(
        Mutex<HashMap<(&'static str, NodeKey), NodeValue>>,
        Mutex<usize>,
    );
    impl Memo for Store {
        fn nodes(
            &self,
            kind: &'static str,
            keys: &[NodeKey],
            _ctx: &JobContext,
            build: &(dyn Fn(usize) -> mklight_core::MkResult<Built> + Sync),
        ) -> Vec<mklight_core::MkResult<NodeValue>> {
            keys.iter()
                .enumerate()
                .map(|(i, k)| {
                    if let Some(v) = self.0.lock().unwrap().get(&(kind, *k)) {
                        return Ok(v.clone());
                    }
                    *self.1.lock().unwrap() += 1;
                    let v = build(i)?.value;
                    self.0.lock().unwrap().insert((kind, *k), v.clone());
                    Ok(v)
                })
                .collect()
        }
    }
    let b = backend();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    let s = cpu::PsfAnalysisSettings::default();
    let memo = Store::default();
    let a = cpu::psf_analysis_shared(&t, &s, Some(&memo), &ctx).unwrap();
    let built = *memo.1.lock().unwrap();
    for design in [Design::CpuPupil, Design::Pipelined, Design::CpuLattice] {
        let z = with(design, || {
            gpu::analysis::psf_analysis_shared(&b, &t, &s, Some(&memo), &ctx)
        })
        .unwrap();
        // Every lattice (and every host-resampled pupil) came from the memo.
        assert_eq!(*memo.1.lock().unwrap(), built, "{design:?}");
        for (a, z) in a.psfs.iter().zip(&z.psfs) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
        }
    }
}
