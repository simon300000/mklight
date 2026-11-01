//! Matched-method tests of resident wave operations against independent f64 optics.
use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_wave::{gpu::ResidentPupil, *};
fn snapshot() -> Snapshot {
    Snapshot::new(samples::singlet()).unwrap()
}
fn error(a: &Psf, b: &Psf) -> f64 {
    let peak = a.peak().0;
    a.data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| (a - b).abs() / peak)
        .fold(0., f64::max)
}
fn suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    for g in [
        PupilGeometry::from_working_f_number(4., 100., 1.).unwrap(),
        PupilGeometry::telecentric(0.2, 1.).unwrap(),
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
