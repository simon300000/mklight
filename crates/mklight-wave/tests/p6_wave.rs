//! Matched-method tests of resident wave operations against independent f64 optics.
use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_wave::{gpu::ResidentPupil, *};
fn snapshot() -> Snapshot {
    Snapshot::new(samples::singlet()).unwrap()
}
/// Equal lengths and every |x - y| below `limit`.
fn close(a: &[f64], b: &[f64], limit: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < limit, "{x} {y}");
    }
}
fn error(a: &Psf, b: &Psf) -> f64 {
    let peak = a.peak().0;
    a.data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| (a - b).abs() / peak)
        .fold(0., f64::max)
}
fn resample_suite(b: &dyn ComputeBackend) -> Vec<Vec<u32>> {
    let mut masks_bits = vec![];
    let ctx = JobContext::new();
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let side = 17;
    let rays = StructuredSamples {
        nx: side,
        ny: side,
        samples: (0..side * side)
            .map(|k| {
                let u = 2. * (k % side) as f64 / (side - 1) as f64 - 1.;
                let v = 2. * (k / side) as f64 / (side - 1) as f64 - 1.;
                (u * u + v * v < 0.9 && u * u + v * v > 0.08).then_some(RaySample {
                    u: u + 0.03 * v * v,
                    v,
                    opd_waves: 0.3 * u - 0.2 * v + 0.1 * u * u + 0.01 * (17. * u + 3. * v).sin(),
                    amplitude: 1. + 0.2 * u,
                })
            })
            .collect(),
    };
    let masks = [
        None,
        Some(PupilMask::from(Aperture::Annular { obscuration: 0.3 })),
        Some(PupilMask::Polygons {
            loops: vec![
                vec![[-0.9, -0.9], [0.9, -0.9], [0.9, 0.9], [-0.9, 0.9]],
                vec![[-0.2, -0.2], [-0.2, 0.2], [0.2, 0.2], [0.2, -0.2]],
            ],
        }),
    ];
    for mask in &masks {
        let (reference, _) =
            grid_from_structured(&rays, 32, 0.55, g, mask.as_ref(), 4, &ctx).unwrap();
        let result =
            gpu::from_structured(b, snapshot(), &rays, 32, 0.55, g, mask.as_ref(), 4, &ctx)
                .unwrap();
        let rp = result.value;
        let cov = b.read_view(rp.coverage.view()).unwrap();
        let cov = cov.as_slice::<f32>().unwrap();
        masks_bits.push(cov.iter().map(|v| v.to_bits()).collect());
        let amp = b.read_view(rp.amplitude.view()).unwrap();
        let amp = amp.as_slice::<f32>().unwrap();
        let opd = b.read_view(rp.opd.clone()).unwrap();
        let opd = opd.as_slice::<mklight_gpu::Df32>().unwrap();
        for k in 0..32 * 32 {
            assert_eq!(
                cov[k] as f64,
                reference.coverage()[k],
                "mask {mask:?} cell {k}"
            );
            assert!(
                (amp[k] as f64 - reference.amplitude()[k]).abs() < 2e-6,
                "amplitude {k}"
            );
            assert!(
                (opd[k].to_f64() - reference.opd_waves()[k]).abs() < 1e-7,
                "OPD {k}: {} {}",
                opd[k].to_f64(),
                reference.opd_waves()[k]
            );
        }
        assert_eq!(result.stats.copied_bytes, 0);
    }
    masks_bits
}
fn focus_and_lsf(b: &dyn ComputeBackend) {
    use mklight_gpu::{Df32, resident::*, wave_ops as w};
    let ctx = JobContext::new();
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let pupil = PupilFunction::diffraction_limited(9, 0.55, g, Aperture::Circular, 4).unwrap();
    let rp = ResidentPupil::from_pupil(b, &pupil, &ctx).unwrap();
    let shifts = [-200., -100., -99., 0., 0.1];
    let z = b
        .fill_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [shifts.len()]),
            &mut |mut a| {
                for (v, z) in a.as_mut_slice::<Df32>()?.iter_mut().zip(shifts) {
                    *v = Df32::from_f64(z);
                }
                Ok(())
            },
        )
        .unwrap();
    let mut plan = ResidentPlan::new(snapshot());
    let v = rp.import(&mut plan);
    let z = plan.import(z.view());
    let out = plan.scratch(BufferDesc::new(
        ElementType::DoubleFloat,
        [shifts.len(), 9, 9],
    ));
    plan.operations.push(ResidentOp::Wave(w::Request::Focus {
        phase_steps: None,
        pupil: v,
        geometry: rp.wave_geometry(),
        shifts: plan.view(z).unwrap(),
        output: out,
    }));
    plan.outputs = vec![out];
    let r = b.submit_plan(plan, &ctx).unwrap().wait().unwrap();
    let data = b.read_view(r.outputs[0].view()).unwrap();
    let data = data.as_slice::<Df32>().unwrap();
    for (j, dz) in shifts.into_iter().enumerate() {
        for k in 0..81 {
            let reference = pupil.with_focus_shift(dz).opd_waves()[k];
            assert!(
                (data[j * 81 + k].to_f64() - reference).abs() < 1e-7,
                "far-side focus {dz}/{k}"
            );
        }
    }
    let settings = PsfSettings {
        grid_size: 31,
        sampling: ImageSampling::Padding { factor: 2.5 },
        normalization: PsfNormalization::UnitEnergy,
        transform: PsfTransform::Matrix,
    };
    let reference = fft_psf(&pupil, &settings, &ctx).unwrap();
    let mut plan = ResidentPlan::new(snapshot());
    let psf = rp.append_psf(&mut plan, &settings, [0.; 2]).unwrap();
    let out = plan.scratch(BufferDesc::new(ElementType::F32, [2, 31]));
    plan.operations
        .push(ResidentOp::Wave(w::Request::LineSpread {
            image: plan.view(psf.image).unwrap(),
            spacing: psf.spacing as f32,
            output: out,
        }));
    plan.outputs = vec![out];
    let r = b.submit_plan(plan, &ctx).unwrap().wait().unwrap();
    let data = b.read_view(r.outputs[0].view()).unwrap();
    let data = data.as_slice::<f32>().unwrap();
    for (j, axis) in [LsfAxis::X, LsfAxis::Y].into_iter().enumerate() {
        let expected = line_spread_function(&reference, axis);
        let peak = expected.values.iter().copied().fold(0f64, f64::max);
        for (k, v) in expected.values.iter().enumerate() {
            assert!((data[j * 31 + k] as f64 - v).abs() / peak < 2e-5);
        }
    }
}
fn suite(b: &dyn ComputeBackend) {
    resample_suite(b);
    focus_and_lsf(b);
    let detector = DetectorSettings::default();
    let response = gpu::detector_mtf(
        b,
        snapshot(),
        &detector,
        &[[0., 0.], [100., 0.], [200., 0.]],
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(response.value[0], 1.);
    assert!((response.value[1] - 2. / std::f64::consts::PI).abs() < 1e-6);
    assert!(response.value[2] < 1e-6);

    let ctx = JobContext::new();
    for g in [
        PupilGeometry::from_working_f_number(4., 100., 1.).unwrap(),
        PupilGeometry::telecentric(0.2, 1.).unwrap(),
        PupilGeometry::telecentric(0.85, 1.).unwrap(),
        PupilGeometry::from_working_f_number(3., 50., 1.)
            .unwrap()
            .with_chief_angle(17., 33.)
            .unwrap(),
    ] {
        let pupil = PupilFunction::from_fn(
            24,
            0.55,
            g,
            4,
            |u, v| {
                let r = u * u + v * v;
                r < 1. && r > 0.12
            },
            |u, v| (1. + 0.1 * u, 0.2 * u + 0.13 * v + 0.3 * (u * u + v * v)),
        )
        .unwrap();
        let rp = ResidentPupil::from_pupil(b, &pupil, &ctx).unwrap();
        for transform in [PsfTransform::Fft, PsfTransform::Matrix] {
            for normalization in [PsfNormalization::Strehl, PsfNormalization::UnitEnergy] {
                let settings = PsfSettings {
                    grid_size: 31,
                    sampling: ImageSampling::Padding { factor: 2.7 },
                    normalization,
                    transform,
                };
                let reference = fft_psf(&pupil, &settings, &ctx).unwrap();
                let result = rp.psf(b, snapshot(), &settings, &ctx).unwrap();
                let err = error(&reference, &result.value);
                assert!(err < 2e-5, "PSF {g:?}/{transform:?}: {err}");
                assert_eq!(reference.spacing_mm, result.value.spacing_mm);
                assert_eq!(reference.warnings, result.value.warnings);
                assert!((reference.total_energy / result.value.total_energy - 1.).abs() < 2e-6);
                assert_eq!(result.stats.copied_bytes, 0);
            }
        }
        for plane in [
            ImagePlane::image_surface(&g),
            ImagePlane::tilted(8., -4., 0.003),
        ] {
            let settings = HuygensSettings {
                grid_size: 9,
                spacing_mm: pupil.lambda_f_mm() / 2.,
                plane,
                normalization: PsfNormalization::UnitEnergy,
            };
            let reference = huygens_psf(&pupil, &settings, &ctx).unwrap();
            let result = rp.huygens(b, snapshot(), &settings, &ctx).unwrap();
            let err = error(&reference, &result.value);
            assert!(err < 2e-5, "Huygens {g:?}: {err}");
            assert!((reference.diffraction_peak / result.value.diffraction_peak - 1.).abs() < 2e-6);
            assert_eq!(result.stats.copied_bytes, 0);
        }
        let fs = [0., 5., 20., 50., 100.];
        let frequencies: Vec<_> = fs.iter().map(|f| [0., *f]).collect();
        let shifts = [-0.01, 0., 0.02];
        let batch = rp
            .through_focus_otf(b, snapshot(), &frequencies, &shifts, &ctx)
            .unwrap();
        for (j, &z) in shifts.iter().enumerate() {
            let focused = pupil.with_focus_shift(z);
            let reference = mtf_direct(&focused, &fs, &MtfOrientation::default(), &ctx).unwrap();
            for (a, z) in reference
                .tangential
                .iter()
                .zip(&batch.value[j * fs.len()..(j + 1) * fs.len()])
            {
                assert!((a - z[0].hypot(z[1])).abs() < 3e-5, "focus {j}: {a} {z:?}");
            }
        }
        let reference = mtf_direct(&pupil, &fs, &MtfOrientation::default(), &ctx).unwrap();
        let result = rp.otf(b, snapshot(), &frequencies, &ctx).unwrap();
        for (a, z) in reference.tangential.iter().zip(result.value) {
            assert!((a - z[0].hypot(z[1])).abs() < 3e-5);
        }
    }
}
#[test]
fn emulation_f64() {
    suite(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
fn metal(library: mklight_metal::LibraryChoice) -> mklight_metal::MetalBackend {
    let b = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
        library,
        ..Default::default()
    })
    .unwrap()
    .expect("P6 requires actual host Metal");
    assert!(b.precision().probe.error_free_transforms());
    eprintln!(
        "P6 device={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    b
}
#[cfg(target_os = "macos")]
#[test]
fn metal_f64_both_libraries() {
    for library in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        suite(&metal(library));
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "host qualification across all default traced pupils"]
fn all_sample_pupils() {
    for library in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(library);
        let ctx = JobContext::new();
        let (mut cases, mut psf_max, mut mtf_max, mut opd_max) = (0, 0f64, 0f64, 0f64);
        for (name, system) in samples::all() {
            let snap = Snapshot::new(system.clone()).unwrap();
            let tracer = mklight_analysis::tracer_for(system).unwrap();
            for (fi, field) in tracer.fields().iter().enumerate() {
                for wl in 0..tracer.num_wavelengths() {
                    let traced =
                        trace_pupil(&tracer, field, wl, &PupilSettings::default(), &ctx).unwrap();
                    let p = &traced.pupil;
                    let resident = gpu::from_structured(
                        &b,
                        snap.clone(),
                        &traced.samples,
                        p.n(),
                        p.wavelength_um(),
                        *p.geometry(),
                        traced.mask.as_ref(),
                        4,
                        &ctx,
                    )
                    .unwrap();
                    let mut rp = resident.value;
                    rp.image_offset_mm = p.image_offset_mm();
                    let cov = b.read_view(rp.coverage.view()).unwrap();
                    let cov = cov.as_slice::<f32>().unwrap();
                    let opd = b.read_view(rp.opd.clone()).unwrap();
                    let opd = opd.as_slice::<mklight_gpu::Df32>().unwrap();
                    let mut mask_diff = 0;
                    let mut opd_error = 0f64;
                    for k in 0..p.n() * p.n() {
                        mask_diff += usize::from(cov[k] as f64 != p.coverage()[k]);
                        if cov[k] > 0. {
                            opd_error = opd_error.max((opd[k].to_f64() - p.opd_waves()[k]).abs());
                        }
                    }
                    assert_eq!(mask_diff, 0, "{name}/{fi}/{wl} masks");
                    assert!(opd_error < 0.005, "{name}/{fi}/{wl} OPD {opd_error}");
                    let settings = PsfSettings {
                        grid_size: 127,
                        sampling: ImageSampling::Padding { factor: 2.5 },
                        normalization: PsfNormalization::UnitEnergy,
                        transform: PsfTransform::Matrix,
                    };
                    let reference = fft_psf(p, &settings, &ctx).unwrap();
                    let result = rp.psf(&b, snap.clone(), &settings, &ctx).unwrap();
                    let pe = error(&reference, &result.value);
                    assert!(pe < 1e-4, "{name}/{fi}/{wl} PSF {pe}");
                    let fs = [0., 10., 20., 40.];
                    let freq: Vec<_> = fs.iter().map(|f| [0., *f]).collect();
                    let reference = mtf_direct(p, &fs, &MtfOrientation::default(), &ctx).unwrap();
                    let result = rp.otf(&b, snap.clone(), &freq, &ctx).unwrap();
                    let me = reference
                        .tangential
                        .iter()
                        .zip(&result.value)
                        .map(|(a, z)| (a - z[0].hypot(z[1])).abs())
                        .fold(0f64, f64::max);
                    assert!(me < 0.005, "{name}/{fi}/{wl} MTF {me}");
                    psf_max = psf_max.max(pe);
                    mtf_max = mtf_max.max(me);
                    opd_max = opd_max.max(opd_error);
                    cases += 1;
                    eprintln!(
                        "{}",
                        serde_json::json!({"sample":name,"field":fi,"wavelength":wl,"pupil_grid":p.n(),"psf_peak_error":pe,"mtf_error":me,"opd_error":opd_error,"mask_differences":mask_diff,"resample_stats":resident.stats})
                    );
                }
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({"summary":"p6_all_pupils","cases":cases,"psf_peak_error":psf_max,"mtf_error":mtf_max,"opd_error":opd_max})
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn determinism_concurrency_and_cancel() {
    use mklight_gpu::resident::*;
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let p = PupilFunction::diffraction_limited(128, 0.55, g, Aperture::Circular, 4).unwrap();
    let rp = ResidentPupil::from_pupil(&b, &p, &JobContext::new()).unwrap();
    let settings = PsfSettings {
        grid_size: 127,
        sampling: ImageSampling::Padding { factor: 2.5 },
        normalization: PsfNormalization::UnitEnergy,
        transform: PsfTransform::Matrix,
    };
    let tilted = p.with_added_opd(|u, _| 0.3 * u);
    let rt = ResidentPupil::from_pupil(&b, &tilted, &JobContext::new()).unwrap();
    let make = |pupil: &ResidentPupil| {
        let mut plan = ResidentPlan::new(snapshot());
        let out = pupil.append_psf(&mut plan, &settings, [0.; 2]).unwrap();
        plan.outputs = vec![out.image];
        plan
    };
    let first = b.submit_plan(make(&rp), &JobContext::new()).unwrap();
    let second = b.submit_plan(make(&rt), &JobContext::new()).unwrap();
    let a = first.wait().unwrap();
    let c = second.wait().unwrap();
    assert_ne!(
        b.read_view(a.outputs[0].view()).unwrap().words(),
        b.read_view(c.outputs[0].view()).unwrap().words()
    );
    let expected = fft_psf(&tilted, &settings, &JobContext::new()).unwrap();
    let actual = b.read_view(c.outputs[0].view()).unwrap();
    let peak = expected.peak().0;
    for (a, v) in actual.as_slice::<f32>().unwrap().iter().zip(&expected.data) {
        assert!((*a as f64 - v).abs() / peak < 2e-5);
    }

    let ctx = JobContext::new();
    let mut plan = ResidentPlan::new(snapshot());
    let out = rp
        .append_huygens(
            &mut plan,
            &HuygensSettings {
                grid_size: 256,
                spacing_mm: p.lambda_f_mm() / 2.,
                plane: ImagePlane::default(),
                normalization: PsfNormalization::Strehl,
            },
        )
        .unwrap();
    plan.outputs = vec![out.image];
    let pending = b.submit_plan(plan, &ctx).unwrap();
    let start = std::time::Instant::now();
    let mut progress = 0.;
    while ctx.progress.fraction() <= 0.667 {
        let f = ctx.progress.fraction();
        assert!(f >= progress);
        progress = f;
        assert!(
            !pending.is_ready(),
            "long device job ended before cancellation"
        );
        assert!(start.elapsed().as_secs() < 30);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    pending.cancel();
    assert!(matches!(
        pending.wait(),
        Err(mklight_core::MkError::Cancelled)
    ));
    let again = b
        .submit_plan(make(&rp), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        b.read_view(a.outputs[0].view()).unwrap().words(),
        b.read_view(again.outputs[0].view()).unwrap().words()
    );
    assert!(again.stats.reused_allocations > 0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "provisional matched CPU/resident timing on host GPU"]
fn benchmark_prepared_pupils() {
    use std::time::Instant;
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let mut cases = Vec::new();
    for (name, system) in samples::all() {
        let snap = Snapshot::new(system.clone()).unwrap();
        let t = mklight_analysis::tracer_for(system).unwrap();
        let field = t.fields().last().unwrap();
        let p = trace_pupil(
            &t,
            field,
            t.system().primary_wavelength,
            &PupilSettings::default(),
            &ctx,
        )
        .unwrap()
        .pupil;
        cases.push((name.to_owned(), snap, p, 127));
    }
    for n in [64, 256, 1024] {
        let g = PupilGeometry::from_working_f_number(4., 100., 1.)
            .unwrap()
            .with_chief_angle(12., 31.)
            .unwrap();
        let p = PupilFunction::from_fn(
            n,
            0.55,
            g,
            4,
            |u, v| u * u + v * v < 1.,
            |u, v| (1., 0.2 * u + 0.1 * u * v),
        )
        .unwrap();
        cases.push((
            format!("dense_{n}"),
            snapshot(),
            p,
            if n == 1024 { 2048 } else { 2 * n },
        ));
    }
    for (name, snap, p, size) in cases {
        let settings = PsfSettings {
            grid_size: size,
            sampling: ImageSampling::Padding { factor: 2.5 },
            normalization: PsfNormalization::UnitEnergy,
            transform: PsfTransform::Matrix,
        };
        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        let before = b.resident_cache_stats().copied_bytes;
        let start = Instant::now();
        let rp = ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
        let prepare = start.elapsed().as_secs_f64();
        for run in 0..=11 {
            let t = Instant::now();
            let a = fft_psf(&p, &settings, &ctx).unwrap();
            let cpu_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let v = rp.psf(&b, snap.clone(), &settings, &ctx).unwrap();
            let gpu_seconds = t.elapsed().as_secs_f64();
            let err = error(&a, &v.value);
            assert!(err < 1e-4, "{name}: {err}");
            if run >= 2 {
                cpu.push(cpu_seconds);
                gpu.push(gpu_seconds);
            }
            eprintln!(
                "{}",
                serde_json::json!({"case":name,"run":run,"cold":run==0,"warmup":run==1,"pupil_side":p.n(),"settings":settings,"host_pupil_fill_seconds":prepare,"cpu_seconds":cpu_seconds,"gpu_seconds":gpu_seconds,"stats":v.stats,"boundary_copied_bytes":b.resident_cache_stats().copied_bytes-before,"peak_error":err})
            );
        }
        cpu.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        eprintln!(
            "{}",
            serde_json::json!({"summary":name,"cpu_median":(cpu[4]+cpu[5])*0.5,"cpu_p95":cpu[9],"gpu_median":(gpu[4]+gpu[5])*0.5,"gpu_p95":gpu[9]})
        );
    }
}

fn statistics_suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    let geom = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    for aperture in [
        Aperture::Circular,
        Aperture::Annular { obscuration: 0.45 },
        Aperture::Rectangular {
            half_u: 0.9,
            half_v: 0.4,
        },
    ] {
        let p = PupilFunction::from_aperture(
            65,
            0.55,
            geom,
            aperture,
            Apodization::Uniform,
            4,
            |u, v| 0.7 * u - 0.3 * v + 0.4 * (u * u + v * v),
        )
        .unwrap();
        let resident = ResidentPupil::from_pupil(b, &p, &ctx).unwrap();
        let a = resident.statistics(b, snapshot(), &ctx).unwrap();
        let s = p.opd_stats(false);
        let t = p.opd_stats(true);
        for (got, want) in [
            (a.value.rms, s.rms),
            (a.value.pv, s.pv),
            (a.value.rms_tilt_removed, t.rms),
            (a.value.pv_tilt_removed, t.pv),
            (a.value.strehl, p.strehl_at_reference()),
        ] {
            assert!((got - want).abs() < 2e-6, "statistics {got} != {want}");
        }
        assert_eq!(a.stats.copied_bytes, 0);
    }
}
#[test]
fn statistics_f64() {
    statistics_suite(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn statistics_metal() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        statistics_suite(&metal(origin));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn resident_traced_pupil_matches_reference() {
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let service = gpu::PupilTracer::new(&b, &t, &ctx).unwrap();
    for amplitude in [
        AmplitudeModel::Uniform,
        AmplitudeModel::Transmission,
        AmplitudeModel::Radiometric,
    ] {
        for reference in [ReferencePoint::Chief, ReferencePoint::Centroid] {
            let settings = PupilSettings {
                amplitude,
                reference,
                ..Default::default()
            };
            let field = t.field(t.fields().len() - 1);
            let wl = t.system().primary_wavelength;
            let expected = trace_pupil(&t, field, wl, &settings, &ctx).unwrap();
            let result = service.trace(field, wl, &settings, &ctx).unwrap();
            let p = &result.value.pupil;
            assert_eq!(p.n, expected.pupil.n());
            assert!(
                (result.value.info.sin_max - expected.info.sin_max).abs() < 2e-6,
                "sine {} {}",
                result.value.info.sin_max,
                expected.info.sin_max
            );
            let opd = b.read_view(p.opd.clone()).unwrap();
            let opd = opd.as_slice::<mklight_gpu::Df32>().unwrap();
            let cov = b.read_view(p.coverage.view()).unwrap();
            let cov = cov.as_slice::<f32>().unwrap();
            let amp = b.read_view(p.amplitude.view()).unwrap();
            let amp = amp.as_slice::<f32>().unwrap();
            let mut maximum = 0f64;
            let mut amp_err = 0f64;
            let mut mask_diff = 0;
            for k in 0..p.n * p.n {
                if cov[k] as f64 != expected.pupil.coverage()[k] {
                    mask_diff += 1;
                }
                if cov[k] > 0. && expected.pupil.coverage()[k] > 0. {
                    maximum = maximum.max((opd[k].to_f64() - expected.pupil.opd_waves()[k]).abs());
                    amp_err = amp_err.max((amp[k] as f64 - expected.pupil.amplitude()[k]).abs());
                }
            }
            eprintln!(
                "P6 trace {amplitude:?} {reference:?}: OPD={maximum:e}, amplitude={amp_err:e}, mask={mask_diff}, commands={}, device_ms={}",
                result.stats.command_buffers,
                result.stats.device_seconds * 1000.
            );
            // Gates are 5e-3 waves and 1e-3; measured maxima are ~2e-7 and
            // ~6e-8, so regressions far inside the gates still fail here.
            assert!(maximum < 1e-5, "OPD {maximum:e}");
            assert!(amp_err < 1e-5, "amplitude {amp_err:e}");
            assert_eq!(mask_diff, 0);
            assert_eq!(result.stats.copied_bytes, 0);
        }
    }
}

fn spectral_suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let mut a = PupilFunction::from_aperture(
        32,
        0.45,
        g,
        Aperture::Circular,
        Apodization::Uniform,
        4,
        |u, v| 0.12 * (u * u + v * v),
    )
    .unwrap();
    a.set_image_offset_mm([-0.001, 0.002]);
    let mut z = PupilFunction::from_aperture(
        32,
        0.65,
        g,
        Aperture::Circular,
        Apodization::Uniform,
        4,
        |u, v| 0.08 * u - 0.14 * v,
    )
    .unwrap();
    z.set_image_offset_mm([0.001, -0.002]);
    let ar = ResidentPupil::from_pupil(b, &a, &ctx).unwrap();
    let zr = ResidentPupil::from_pupil(b, &z, &ctx).unwrap();
    let settings = PolyPsfSettings::new(63, 0.001);
    let want = polychromatic_psf(&[(&a, 0.3), (&z, 0.7)], &settings, &ctx).unwrap();
    let got = gpu::polychromatic_psf(
        b,
        snapshot(),
        &[(&ar, 0.3), (&zr, 0.7)],
        &settings,
        false,
        &ctx,
    )
    .unwrap();
    let got = got.value.finish(b, &ctx).unwrap();
    assert!(error(&want, &got) < 2e-5);
    assert!((want.energy_in_grid() - got.energy_in_grid()).abs() < 1e-6 * want.total_energy);
    let fs = [[0., 0.], [10., 0.], [0., 40.], [40., 20.]];
    let got = gpu::polychromatic_otf(b, snapshot(), &[(&ar, 0.3), (&zr, 0.7)], &fs, &ctx).unwrap();
    let aa = ar.otf(b, snapshot(), &fs, &ctx).unwrap();
    let zz = zr.otf(b, snapshot(), &fs, &ctx).unwrap();
    for k in 0..fs.len() {
        for j in 0..2 {
            assert!((got.value[k][j] - (0.3 * aa.value[k][j] + 0.7 * zz.value[k][j])).abs() < 2e-6);
        }
    }
}
#[test]
fn spectral_f64() {
    spectral_suite(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn spectral_metal() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        spectral_suite(&metal(origin));
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "host qualification of GPU-generated default pupils"]
fn all_gpu_traced_pupils() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(origin);
        let ctx = JobContext::new();
        let mut count = 0;
        let mut max_opd = 0f64;
        let mut max_amp = 0f64;
        let mut masks = 0usize;
        for (name, model) in samples::all() {
            let t = mklight_analysis::tracer_for(model).unwrap();
            let service = gpu::PupilTracer::new(&b, &t, &ctx).unwrap();
            for field in 0..t.fields().len() {
                for wl in 0..t.num_wavelengths() {
                    let settings = PupilSettings::default();
                    let reference = trace_pupil(&t, t.field(field), wl, &settings, &ctx).unwrap();
                    let r = service.trace(t.field(field), wl, &settings, &ctx).unwrap();
                    let p = &r.value.pupil;
                    assert_eq!(
                        p.n,
                        reference.pupil.n(),
                        "{name} field {field} wavelength {wl}"
                    );
                    let w = b.read_view(p.opd.clone()).unwrap();
                    let w = w.as_slice::<mklight_gpu::Df32>().unwrap();
                    let c = b.read_view(p.coverage.view()).unwrap();
                    let c = c.as_slice::<f32>().unwrap();
                    let a = b.read_view(p.amplitude.view()).unwrap();
                    let a = a.as_slice::<f32>().unwrap();
                    let mut error = 0f64;
                    let mut amp = 0f64;
                    let mut differences = 0;
                    for k in 0..p.n * p.n {
                        if c[k] as f64 != reference.pupil.coverage()[k] {
                            differences += 1;
                        }
                        if c[k] > 0. && reference.pupil.coverage()[k] > 0. {
                            error =
                                error.max((w[k].to_f64() - reference.pupil.opd_waves()[k]).abs());
                            amp = amp.max((a[k] as f64 - reference.pupil.amplitude()[k]).abs());
                        }
                    }
                    eprintln!(
                        "P6_GENERATED {name} f={field} wl={wl} n={} opd={error:e} amp={amp:e} mask={differences} commands={} copies={} fallbacks={}",
                        p.n, r.stats.command_buffers, r.stats.copied_bytes, r.stats.fallback_count
                    );
                    assert!(error < 5e-3);
                    assert!(amp < 1e-3);
                    assert_eq!(differences, 0);
                    max_opd = max_opd.max(error);
                    max_amp = max_amp.max(amp);
                    masks += differences;
                    count += 1;
                }
            }
        }
        eprintln!(
            "P6_GENERATED_SUMMARY count={count} max_opd={max_opd:e} max_amplitude={max_amp:e} mask_differences={masks}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_wavefront_and_psf() {
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let settings = analysis::WavefrontSettings::default();
    let reference = analysis::wavefront_map_job(&t, &settings, &ctx).unwrap();
    let got = gpu::analysis::wavefront_map_job(&b, &t, &settings, &ctx).unwrap();
    assert_eq!(got.maps.len(), reference.maps.len());
    for (a, b) in got.maps.iter().zip(&reference.maps) {
        assert_eq!(a.grid, b.grid);
        assert!((a.rms_waves - b.rms_waves).abs() < 1e-5);
        assert!((a.rms_waves_tilt_removed - b.rms_waves_tilt_removed).abs() < 1e-5);
        assert!((a.strehl_at_reference - b.strehl_at_reference).abs() < 1e-5);
        assert!((a.zernike_residual_rms - b.zernike_residual_rms).abs() < 1e-5);
        close(a.zernike.coefficients(), b.zernike.coefficients(), 5e-3);
    }
    let settings = analysis::PsfAnalysisSettings::default();
    let reference = analysis::psf_analysis_job(&t, &settings, &ctx).unwrap();
    let got = gpu::analysis::psf_analysis_job(&b, &t, &settings, &ctx).unwrap();
    assert_eq!(got.psfs.len(), reference.psfs.len());
    for (a, b) in got.psfs.iter().zip(&reference.psfs) {
        assert_eq!(a.psf.size, b.psf.size);
        assert_eq!(a.pupil_sampling, b.pupil_sampling);
        assert!((a.psf.spacing_mm - b.psf.spacing_mm).abs() < 1e-9);
        assert!(error(&a.psf, &b.psf) < 2e-3);
        assert!((a.strehl - b.strehl).abs() < 2e-3);
    }
}

fn energy_and_fft_suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    for angle in [0f64, 25.] {
        let g = g
            .with_chief_ray([angle.to_radians().sin(), 0., angle.to_radians().cos()])
            .unwrap();
        let p = PupilFunction::from_aperture(
            32,
            0.55,
            g,
            Aperture::Annular { obscuration: 0.2 },
            Apodization::Uniform,
            4,
            |u, v| 0.4 * (u * u + v * v) + 0.1 * u,
        )
        .unwrap();
        let rp = ResidentPupil::from_pupil(b, &p, &ctx).unwrap();
        let fs = [[0., 0.], [10., 0.], [0., 40.], [30., 20.]];
        let reference = otf_fft(&p, &ctx).unwrap();
        let got = rp.fft_otf(b, snapshot(), &fs, &ctx).unwrap();
        for (f, z) in fs.iter().zip(got.value) {
            let want = reference.at(f[0], f[1]);
            assert!(
                (z[0] - want.re).hypot(z[1] - want.im) < 2e-3,
                "FFT {angle} {f:?} {z:?} {want:?}"
            );
        }
        let settings = PolyPsfSettings::new(65, 0.001);
        let psf = polychromatic_psf(&[(&p, 1.)], &settings, &ctx).unwrap();
        let resident = gpu::polychromatic_psf(b, snapshot(), &[(&rp, 1.)], &settings, false, &ctx)
            .unwrap()
            .value;
        let radii = [0., 0.001, 0.003, 0.008, 0.015];
        for center in [
            EnergyCenter::Reference,
            EnergyCenter::Centroid,
            EnergyCenter::Peak,
            EnergyCenter::Point {
                x: 0.001,
                y: -0.001,
            },
        ] {
            for square in [false, true] {
                let want = if square {
                    ensquared_energy(&psf, &radii, center, &ctx)
                } else {
                    encircled_energy(&psf, &radii, center, &ctx)
                }
                .unwrap();
                let got = resident
                    .energy(b, snapshot(), &radii, center, square, &ctx)
                    .unwrap();
                for (a, b) in got.value.iter().zip(want) {
                    assert!(
                        (a - b).abs() < 2e-3,
                        "energy {angle} {center:?} {square}: {a} {b}"
                    );
                }
            }
        }
        let got = resident
            .otf(b, snapshot(), &[[0., 0.], [0., 40.]], &ctx)
            .unwrap();
        let want = mtf_from_psf(&psf, &[0., 40.], &MtfOrientation::default(), &ctx).unwrap();
        assert!((got.value[1][0].hypot(got.value[1][1]) - want.tangential[1]).abs() < 2e-3);
    }
}
#[test]
fn energy_fft_f64() {
    energy_and_fft_suite(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn energy_fft_metal() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        energy_and_fft_suite(&metal(origin));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_mtf_and_focus() {
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    for method in [
        analysis::MtfAnalysisMethod::Direct,
        analysis::MtfAnalysisMethod::Fft,
    ] {
        let settings = analysis::MtfSettings {
            method,
            ..Default::default()
        };
        let a = analysis::mtf_analysis_job(&t, &settings, &ctx).unwrap();
        let z = gpu::analysis::mtf_analysis_job(&b, &t, &settings, &ctx).unwrap();
        assert_eq!(a.frequencies, z.frequencies);
        assert_eq!(a.fields.len(), z.fields.len());
        for (a, z) in a.fields.iter().zip(z.fields) {
            assert_eq!(a.pupil_sampling, z.pupil_sampling);
            close(&a.tangential, &z.tangential, 2e-3);
            close(&a.sagittal, &z.sagittal, 2e-3);
            close(&a.diffraction_limit, &z.diffraction_limit, 2e-3);
        }
    }
    let settings = analysis::ThroughFocusSettings::default();
    let a = analysis::through_focus_mtf_analysis_job(&t, &settings, &ctx).unwrap();
    let z = gpu::analysis::through_focus_mtf_analysis_job(&b, &t, &settings, &ctx).unwrap();
    assert_eq!(a.frequencies, z.frequencies);
    close(&a.focus_mm, &z.focus_mm, 1e-7);
    assert_eq!(a.fields.len(), z.fields.len());
    for (a, z) in a.fields.iter().zip(z.fields) {
        assert_eq!(a.pupil_sampling, z.pupil_sampling);
        assert_eq!(a.curves.len(), z.curves.len());
        for (a, z) in a.curves.iter().zip(z.curves) {
            close(&a.tangential, &z.tangential, 2e-3);
            close(&a.sagittal, &z.sagittal, 2e-3);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn full_image_simulation() {
    use mklight_wave::image_simulation as reference;
    use std::sync::Arc;
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let ctx = JobContext::new();
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let mut settings = reference::ImageSimulationSettings::default();
    let old = Arc::new(reference::build_psf_grid(&t, &settings.optics, &ctx).unwrap());
    let new = Arc::new(
        gpu::image_simulation::build_psf_grid(&b, &t, &settings.optics, &ctx)
            .unwrap()
            .value,
    );
    assert_eq!(old.psf_size, new.psf_size());
    assert!((old.spacing_mm - new.spacing_mm()).abs() < 1e-9);
    for (a, z) in old.mapping.iter().zip(new.mapping()) {
        assert_eq!(a.pupil_sampling, z.pupil_sampling);
        assert!((a.relative_illumination - z.relative_illumination).abs() < 1e-3);
    }
    for pattern in [
        reference::TestPattern::Grid,
        reference::TestPattern::SiemensStar,
        reference::TestPattern::SlantedEdge,
        reference::TestPattern::TextChart,
    ] {
        settings.source = reference::ImageSource::Pattern { pattern };
        let a = reference::image_simulation_job(&t, &settings, Some(old.clone()), &ctx).unwrap();
        let z =
            gpu::image_simulation::image_simulation_job(&b, &t, &settings, Some(new.clone()), &ctx)
                .unwrap();
        let norm = a.image.iter().sum::<f64>();
        let l1 = a
            .image
            .iter()
            .zip(&z.image)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / norm;
        let flux = (a.image.iter().sum::<f64>() - z.image.iter().sum::<f64>()).abs() / norm;
        eprintln!(
            "P6_IMAGE {pattern:?} L1={l1:e} flux={flux:e} stats={}",
            z.run_info.settings["resident"]["plan_stats"]["command_buffers"]
        );
        assert!(l1 < 2e-3);
        assert!(flux < 1e-3);
    }
    settings.source = reference::ImageSource::Buffer {
        width: 64,
        height: 64,
        channels: 3,
        data: (0..64 * 64 * 3)
            .map(|i| {
                if i / 3 == 32 * 64 + 32 {
                    (i % 3 + 1) as f64
                } else {
                    0.
                }
            })
            .collect(),
    };
    settings.monochrome = false;
    settings.detector_aperture_mm = Some([0.003, 0.004]);
    let a = reference::image_simulation_job(&t, &settings, Some(old.clone()), &ctx).unwrap();
    let z = gpu::image_simulation::image_simulation_job(&b, &t, &settings, Some(new.clone()), &ctx)
        .unwrap();
    let norm = a.image.iter().sum::<f64>();
    assert!(
        a.image
            .iter()
            .zip(&z.image)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / norm
            < 2e-3
    );
    settings.tile_size = 23;
    let tiled =
        gpu::image_simulation::image_simulation_job(&b, &t, &settings, Some(new.clone()), &ctx)
            .unwrap();
    assert!(
        z.image
            .iter()
            .zip(&tiled.image)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / norm
            < 2e-3
    );
    settings.source = reference::ImageSource::Buffer {
        width: 2,
        height: 2,
        channels: 1,
        data: vec![0.5; 4],
    };
    settings.monochrome = true;
    settings.full_well_electrons = Some(100.);
    settings.read_noise_sigma = 0.01;
    settings.seed = 938;
    let a = gpu::image_simulation::image_simulation_job(&b, &t, &settings, Some(new.clone()), &ctx)
        .unwrap();
    let z =
        gpu::image_simulation::image_simulation_job(&b, &t, &settings, Some(new), &ctx).unwrap();
    assert_eq!(a.image, z.image);
}

#[cfg(target_os = "macos")]
#[test]
fn resampling_masks_match_device_emulation() {
    for origin in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = metal(origin);
        let cpu = CpuF32Backend::matching(&b.precision().probe);
        assert_eq!(
            resample_suite(&b),
            resample_suite(&cpu),
            "exact resampling coverage/branch masks"
        );
    }
}

/// Bits of every resident channel of a traced pupil, plus its rim loops.
fn traced_bits(b: &dyn ComputeBackend, r: &gpu::ResidentTracedPupil) -> (Vec<Vec<u32>>, String) {
    let p = &r.pupil;
    let words = |v: mklight_gpu::resident::BufferView| b.read_view(v).unwrap().words().to_vec();
    (
        vec![
            words(r.nodes.as_ref().unwrap().view()),
            words(r.valid.as_ref().unwrap().view()),
            words(p.amplitude.view()),
            words(p.coverage.view()),
            words(p.opd.clone()),
        ],
        format!("{:?} {:?} {:?}", r.mask, r.entrance_rim, r.info),
    )
}

/// Batched fields × wavelengths reproduce single-pupil tracing bit for bit
/// (row indices, split ranges, rim offsets), and the CPU-f32 twins of the
/// lattice, rim, split, wavelength and placement operations match Metal.
#[cfg(target_os = "macos")]
#[test]
fn batched_pupils_match_single_and_emulation() {
    let ctx = JobContext::new();
    let b = metal(mklight_metal::LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let t = mklight_analysis::tracer_for(samples::cooke_triplet()).unwrap();
    let settings = PupilSettings {
        rays: 17,
        amplitude: AmplitudeModel::Radiometric,
        ..Default::default()
    };
    let fields: Vec<_> = (0..t.fields().len()).map(|f| t.field(f)).collect();
    let pupils: Vec<_> = fields
        .iter()
        .flat_map(|&f| (0..t.num_wavelengths()).map(move |wl| (f, wl)))
        .collect();
    assert!(pupils.len() > 3);
    let metal_tracer = gpu::PupilTracer::new(&b, &t, &ctx).unwrap();
    let (batched, shared) = metal_tracer.trace_many(&pupils, &settings, &ctx).unwrap();
    assert_eq!(batched.len(), pupils.len());
    let cpu_tracer = gpu::PupilTracer::new(&cpu, &t, &ctx).unwrap();
    let (emulated, _) = cpu_tracer.trace_many(&pupils, &settings, &ctx).unwrap();
    assert_eq!(emulated.len(), pupils.len());
    for (k, (&(f, wl), (r, e))) in pupils.iter().zip(batched.iter().zip(&emulated)).enumerate() {
        let r = r.as_ref().unwrap();
        let single = metal_tracer.trace(f, wl, &settings, &ctx).unwrap();
        assert!(single.value.mask.is_some(), "pupil {k} has a traced rim");
        assert_eq!(
            traced_bits(&b, &single.value),
            traced_bits(&b, &r.value),
            "pupil {k}"
        );
        assert_eq!(
            traced_bits(&b, &r.value),
            traced_bits(&cpu, &e.as_ref().unwrap().value),
            "pupil {k} CPU-f32"
        );
    }
    // The chief/lattice/saddle/rim commands are shared by the whole batch.
    let single = metal_tracer
        .trace(pupils[0].0, pupils[0].1, &settings, &ctx)
        .unwrap();
    let total = shared.command_buffers
        + batched
            .iter()
            .map(|r| r.as_ref().unwrap().stats.command_buffers)
            .sum::<u32>();
    eprintln!(
        "P6_BATCH pupils={} commands={total} single={}",
        pupils.len(),
        single.stats.command_buffers
    );
    // Fusion also removes most single-pupil commands, reducing the ratio
    // attributable to batching alone. The batch must still use fewer commands
    // than independent pupils, and retain its bit-identical outputs above.
    assert!(total < pupils.len() as u32 * single.stats.command_buffers);
    assert_eq!(shared.copied_bytes, 0);
}
