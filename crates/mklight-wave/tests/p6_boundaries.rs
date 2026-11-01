//! Physical edge cases and shared-interface boundaries, not default-grid only.
#![cfg(target_os = "macos")]
use mklight_core::{
    ApertureShape, ApertureSpec, Field, FieldKind, FieldSet, JobContext, MaterialRef,
    OpticalSystem, Placement, Surface, SurfaceShape, Wavelength, WavelengthSet, samples,
};
use mklight_gpu::{
    ComputeBackend, CpuF32Backend, Df32,
    resident::*,
    wave_ops::{self as w, simulation::Simulation},
};
use mklight_wave::*;
fn metal(origin: mklight_metal::LibraryChoice) -> mklight_metal::MetalBackend {
    mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
        library: origin,
        ..Default::default()
    })
    .unwrap()
    .expect("P6 boundary qualification requires host Metal")
}
fn lens(f: f64, diameter: f64, stop_distance: Option<f64>) -> OpticalSystem {
    let surf = |id, shape, t| Surface::new(id, shape, t, MaterialRef::Air);
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: diameter };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0., 0.), Field::new(0., 3.)],
    };
    s.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.55,
            weight: 1.,
        }],
        primary: 0,
    };
    s.surfaces = vec![surf(0, SurfaceShape::Plane, f64::INFINITY)];
    if let Some(d) = stop_distance {
        s.surfaces.push(surf(1, SurfaceShape::Plane, d));
    }
    let id = s.surfaces.len() as u64;
    s.surfaces.push(surf(
        id,
        SurfaceShape::Paraxial { focal_length: f },
        f + 0.1,
    ));
    s.surfaces.push(surf(id + 1, SurfaceShape::Plane, 0.));
    s.stop = 1;
    s
}
fn compare_pupil(
    b: &dyn ComputeBackend,
    t: &mklight_trace::SequentialTracer,
    fi: usize,
    wl: usize,
    settings: &PupilSettings,
) {
    let ctx = JobContext::new();
    let expected = trace_pupil(t, t.field(fi), wl, settings, &ctx).unwrap();
    let service = gpu::PupilTracer::new(b, t, &ctx).unwrap();
    let r = service.trace(t.field(fi), wl, settings, &ctx).unwrap();
    assert_eq!(r.value.pupil.n, expected.pupil.n());
    let p = &r.value.pupil;
    let cov = b.read_view(p.coverage.view()).unwrap();
    let cov = cov.as_slice::<f32>().unwrap();
    let opd = b.read_view(p.opd.clone()).unwrap();
    let opd = opd.as_slice::<Df32>().unwrap();
    let amp = b.read_view(p.amplitude.view()).unwrap();
    let amp = amp.as_slice::<f32>().unwrap();
    let mut maximum = 0f64;
    for k in 0..p.n * p.n {
        assert_eq!(
            cov[k] as f64,
            expected.pupil.coverage()[k],
            "coverage {fi}/{wl}/{k}"
        );
        if cov[k] > 0. {
            maximum = maximum.max((opd[k].to_f64() - expected.pupil.opd_waves()[k]).abs());
            assert!((amp[k] as f64 - expected.pupil.amplitude()[k]).abs() < 1e-3);
        }
    }
    assert!(maximum < 5e-3, "OPD {maximum}");
    assert!((r.value.info.clear_area - expected.info.clear_area).abs() < 1e-5);
    eprintln!(
        "P6_BOUNDARY pupil OPD={maximum:e}, rays={}, rim={}, telecentric={}",
        settings.rays, r.value.info.rim_points, r.value.info.telecentric
    );
}
#[test]
fn physical_pupil_boundaries_and_wavelength_chunks() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(origin);
        let mut tilted = lens(50., 10., None);
        tilted.surfaces.last_mut().unwrap().placement = Some(Placement {
            tilt_x: 20.,
            tilt_y: 12.,
            ..Default::default()
        });
        let mut annular = lens(50., 10., None);
        annular.surfaces[1].aperture = Some(ApertureShape::Annular {
            inner_radius: 3.75,
            outer_radius: 5.,
        });
        for (name, model, rays) in [
            ("telecentric", lens(50., 10., Some(50.)), 17),
            ("far_sheet", lens(50., 10., Some(75.)), 17),
            ("long_track", lens(399., 20., Some(598.5)), 17),
            ("high_na", lens(50., 150., None), 33),
            ("tilted", tilted, 17),
            ("annular_saddles", annular, 9),
        ] {
            let t = mklight_analysis::tracer_for(model).unwrap();
            for reference in [ReferencePoint::Chief, ReferencePoint::Centroid] {
                eprintln!("P6_BOUNDARY {name} {reference:?}");
                compare_pupil(
                    &b,
                    &t,
                    1,
                    0,
                    &PupilSettings {
                        rays,
                        grid: 32,
                        amplitude: AmplitudeModel::Radiometric,
                        reference,
                        ..Default::default()
                    },
                );
            }
        }
        let mut model = samples::cooke_triplet();
        model.wavelengths = WavelengthSet {
            wavelengths: (0..20)
                .map(|i| Wavelength {
                    um: 0.43 + i as f64 * 0.014,
                    weight: 1.,
                })
                .collect(),
            primary: 17,
        };
        model.fields = FieldSet {
            kind: FieldKind::RealImageHeight,
            fields: vec![Field::new(0., 0.), Field::new(0., 2.)],
        };
        let t = mklight_analysis::tracer_for(model).unwrap();
        let settings = PupilSettings {
            rays: 17,
            grid: 32,
            ..Default::default()
        };
        for wl in [0, 15, 19] {
            compare_pupil(&b, &t, 1, wl, &settings);
        }
        let s = analysis::ThroughFocusSettings {
            fields: mklight_analysis::util::FieldSelection::Index { index: 1 },
            pupil: settings,
            frequencies: vec![10., 20.],
            focus_range_mm: Some(0.05),
            focus_samples: 3,
            ..Default::default()
        };
        let a = analysis::through_focus_mtf_analysis_job(&t, &s, &JobContext::new()).unwrap();
        let z =
            gpu::analysis::through_focus_mtf_analysis_job(&b, &t, &s, &JobContext::new()).unwrap();
        for (a, z) in a.fields[0].curves.iter().zip(&z.fields[0].curves) {
            for (x, y) in a
                .tangential
                .iter()
                .chain(&a.sagittal)
                .zip(z.tangential.iter().chain(&z.sagittal))
            {
                assert!((x - y).abs() < 2e-3);
            }
        }
    }
}
fn pattern(b: &dyn ComputeBackend, kind: u32) -> Vec<u32> {
    let mut p = ResidentPlan::new(mklight_core::Snapshot::new(samples::singlet()).unwrap());
    let output = p.scratch(BufferDesc::new(ElementType::F32, [37, 43, 1]));
    p.operations.push(ResidentOp::Wave(w::Request::Simulation(
        Simulation::Pattern {
            size: [43, 37],
            kind,
            output,
        },
    )));
    p.outputs = vec![output];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    b.read_view(r.outputs[0].view()).unwrap().words().to_vec()
}
fn noise(b: &dyn ComputeBackend, well: f32, sigma: f32) -> mklight_core::MkResult<Vec<f32>> {
    let input = b
        .fill_buffer(
            BufferDesc::new(ElementType::F32, [256, 256, 1]),
            &mut |mut a| {
                a.as_mut_slice::<f32>()?.fill(0.5);
                Ok(())
            },
        )
        .unwrap();
    let mut p = ResidentPlan::new(mklight_core::Snapshot::new(samples::singlet()).unwrap());
    let input = p.import(input.view());
    let output = p.scratch(BufferDesc::new(ElementType::F32, [256, 256, 1]));
    p.operations.push(ResidentOp::Wave(w::Request::Simulation(
        Simulation::Noise {
            input: p.view(input).unwrap(),
            seed: 0x1827364501020304,
            well: Some(well),
            sigma,
            output,
        },
    )));
    p.outputs = vec![output];
    let r = b.submit_plan(p, &JobContext::new())?.wait()?;
    Ok(b.read_view(r.outputs[0].view())
        .unwrap()
        .as_slice::<f32>()
        .unwrap()
        .to_vec())
}
#[test]
fn image_algebra_emulation_and_noise_moments() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(origin);
        let c = CpuF32Backend::matching(&b.precision().probe);
        for kind in 0..4 {
            assert_eq!(
                pattern(&c, kind),
                pattern(&b, kind),
                "pattern branch/word contract {kind}"
            );
        }
        assert!(matches!(
            noise(&b, 1e14, 0.),
            Err(mklight_core::MkError::Unsupported(_))
        ));
        for (well, sigma) in [
            (2., 0.01),
            (40., 0.01),
            (200., 0.01),
            (1000000., 0.01),
            (1e12, 0.),
        ] {
            let a = noise(&c, well, sigma).unwrap();
            let z = noise(&b, well, sigma).unwrap();
            let again = noise(&b, well, sigma).unwrap();
            assert_eq!(z, again);
            if sigma == 0. {
                assert_eq!(a, z, "shot count branch contract");
            }
            let max = a
                .iter()
                .zip(&z)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0f32, f32::max);
            assert!(
                max < 2e-5,
                "noise post-transcendental branches: {well}, max {max}"
            );
            let mean = z.iter().map(|v| *v as f64).sum::<f64>() / z.len() as f64;
            let var =
                z.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (z.len() - 1) as f64;
            let expected = 0.5 / well as f64 + (sigma as f64).powi(2);
            assert!((mean - 0.5).abs() < 5. * (expected / z.len() as f64).sqrt());
            assert!((var - expected).abs() < 5. * expected * (2. / (z.len() - 1) as f64).sqrt());
            eprintln!(
                "P6_NOISE well={well} mean={mean} variance={var} expected={expected} emulation_max={max}"
            );
        }
    }
}
#[test]
fn sensor_horizon_and_huygens_limits() {
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let snapshot = mklight_core::Snapshot::new(samples::singlet()).unwrap();
    let g = PupilGeometry::from_working_f_number(0.7, 100., 1.)
        .unwrap()
        .with_chief_angle(60., 25.)
        .unwrap();
    let p = PupilFunction::diffraction_limited(32, 0.55, g, Aperture::Circular, 4).unwrap();
    let rp = gpu::ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    assert!(mtf_direct(&p, &[10.], &MtfOrientation::default(), &ctx).is_err());
    assert!(rp.otf(&b, snapshot.clone(), &[[0., 10.]], &ctx).is_err());
    assert!(
        rp.fft_otf(&b, snapshot.clone(), &[[0., 10.]], &ctx)
            .is_err()
    );
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let p = PupilFunction::diffraction_limited(128, 0.55, g, Aperture::Circular, 4).unwrap();
    let rp = gpu::ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    assert!(
        gpu::polychromatic_psf(
            &b,
            snapshot.clone(),
            &[(&rp, 1.), (&rp, 1.), (&rp, 1.)],
            &PolyPsfSettings::new(1024, 0.001),
            true,
            &ctx
        )
        .is_err()
    );
    let p = PupilFunction::diffraction_limited(8, 0.55, g, Aperture::Circular, 4).unwrap();
    let rp = gpu::ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    let settings = HuygensSettings {
        grid_size: 1024,
        spacing_mm: p.lambda_f_mm() / 2.5,
        plane: ImagePlane::default(),
        normalization: PsfNormalization::UnitEnergy,
    };
    let z = rp.huygens(&b, snapshot, &settings, &ctx).unwrap();
    let a = huygens_psf(
        &p,
        &HuygensSettings {
            grid_size: 17,
            ..settings
        },
        &ctx,
    )
    .unwrap();
    let peak = a.peak().0;
    for y in 0..17 {
        for x in 0..17 {
            assert!(
                (a.data[y * 17 + x] - z.value.data[(y + 512 - 8) * 1024 + x + 512 - 8]).abs()
                    < 2e-4 * peak
            );
        }
    }
    eprintln!(
        "P6_HUYGENS_CAP commands={} gpu_ms={} copies={} peak_bytes={}",
        z.stats.command_buffers,
        z.stats.device_seconds * 1000.,
        z.stats.copied_bytes,
        z.stats.peak_bytes
    );
}

#[test]
fn full_focus_range_and_transform_cancellation() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(origin);
        let ctx = JobContext::new();
        let snapshot = mklight_core::Snapshot::new(samples::singlet()).unwrap();
        let g = PupilGeometry::from_working_f_number(1.4, 100., 1.).unwrap();
        let p = PupilFunction::diffraction_limited(32, 0.55, g, Aperture::Circular, 4)
            .unwrap()
            .with_added_opd(|u, v| 0.3 * u * u * u + 0.2 * v * v);
        let rp = gpu::ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
        let shifts: Vec<_> = (0..1001).map(|i| -200. + 205. * i as f64 / 1000.).collect();
        let fs = [0., 40.];
        let r = rp
            .through_focus_otf(&b, snapshot.clone(), &[[0., 0.], [0., 40.]], &shifts, &ctx)
            .unwrap();
        let mut max = 0f64;
        for (i, &z) in shifts.iter().enumerate() {
            let a = mtf_direct(
                &p.with_focus_shift(z),
                &fs,
                &MtfOrientation::default(),
                &JobContext::new(),
            )
            .unwrap();
            for (j, a) in a.tangential.iter().enumerate() {
                let z = r.value[i * 2 + j];
                max = max.max((a - z[0].hypot(z[1])).abs());
            }
        }
        assert!(max < 5e-5, "full focus range: {max}");
        assert!(
            r.run_info
                .warnings
                .iter()
                .any(|w| w.contains("between pupil samples"))
        );
        let batch = rp
            .focus_batch(&b, snapshot.clone(), &shifts, &JobContext::new())
            .unwrap();
        for i in [0, 400, 500, 997, 1000] {
            let expected = p.with_focus_shift(shifts[i]).max_phase_step();
            let actual = batch.value[i].warnings.iter().find_map(|w| {
                if let WaveWarning::PupilUndersampled { max_step_waves } = w {
                    Some(*max_step_waves)
                } else {
                    None
                }
            });
            if expected > 0.5 {
                assert!((actual.unwrap() - expected).abs() < 2e-3);
            }
        }
        eprintln!(
            "P6_LARGE_FOCUS planes=1001 max_mtf={max:e} commands={} copies={}",
            r.stats.command_buffers, r.stats.copied_bytes
        );
        let g = PupilGeometry::from_working_f_number(4., 100., 1.)
            .unwrap()
            .with_chief_angle(20., 15.)
            .unwrap();
        let large =
            PupilFunction::diffraction_limited(1024, 0.55, g, Aperture::Circular, 1).unwrap();
        let large = gpu::ResidentPupil::from_pupil(&b, &large, &JobContext::new()).unwrap();
        let mut plan = ResidentPlan::new(snapshot.clone());
        let out = large
            .append_psf(
                &mut plan,
                &PsfSettings {
                    grid_size: 2048,
                    sampling: ImageSampling::Padding { factor: 2. },
                    normalization: PsfNormalization::UnitEnergy,
                    transform: PsfTransform::Matrix,
                },
                [0.; 2],
            )
            .unwrap();
        plan.outputs = vec![out.image];
        let ctx = JobContext::new();
        let pending = b.submit_plan(plan, &ctx).unwrap();
        let start = std::time::Instant::now();
        let mut progress = 0.;
        while ctx.progress.stage() != "transform" {
            let f = ctx.progress.fraction();
            assert!(f >= progress);
            progress = f;
            assert!(
                !pending.is_ready(),
                "transform completed before cancellation"
            );
            assert!(start.elapsed().as_secs() < 30);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let cancellation = std::time::Instant::now();
        pending.cancel();
        assert!(matches!(
            pending.wait(),
            Err(mklight_core::MkError::Cancelled)
        ));
        eprintln!(
            "P6_TRANSFORM_CANCEL latency_ms={}",
            cancellation.elapsed().as_secs_f64() * 1000.
        );
        let recovered = rp
            .otf(&b, snapshot, &[[0., 0.]], &JobContext::new())
            .unwrap();
        assert!((recovered.value[0][0] - 1.).abs() < 1e-6);
    }
}
