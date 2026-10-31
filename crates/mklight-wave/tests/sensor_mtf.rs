//! Image-surface diffraction against Gaussian Fourier integrals.

use std::f64::consts::PI;

use mklight_core::JobContext;
use mklight_wave::{MtfMethod, PupilFunction, PupilGeometry, mtf::otf_at};

/// A Gaussian angular spectrum with a quadratic phase has a closed-form
/// autocorrelation. Build it on an oblique reference sphere, where neither
/// its amplitude nor its phase is quadratic in the stored pupil coordinates.
fn gaussian(chief_angle: f64, azimuth: f64) -> PupilFunction {
    let g = PupilGeometry::telecentric(0.4, 1.0)
        .unwrap()
        .with_chief_angle(chief_angle, azimuth)
        .unwrap();
    let [eu, ev, ew] = g.chief_frame();
    PupilFunction::from_fn(
        384,
        0.55,
        g,
        4,
        |u, v| u * u + v * v <= 1.0,
        |u, v| {
            let (a, b) = (g.sin_max() * u, g.sin_max() * v);
            let gamma = (1.0 - a * a - b * b).sqrt();
            let q = eu * a + ev * b - ew * gamma;
            let (x, y) = (q.x + ew.x, q.y + ew.y);
            let spectrum = (-0.25 * ((x / 0.045).powi(2) + (y / 0.035).powi(2))).exp();
            (-q.z * spectrum, 80.0 * x * x - 120.0 * y * y)
        },
    )
    .unwrap()
}

fn gaussian_mtf(fx: f64, fy: f64) -> f64 {
    let (dx, dy) = (0.00055 * fx, 0.00055 * fy);
    let exponent = |d: f64, sigma: f64, phase: f64| {
        d * d / (8.0 * sigma * sigma) + 8.0 * PI * PI * phase * phase * sigma * sigma * d * d
    };
    (-exponent(dx, 0.045, 80.0) - exponent(dy, 0.035, -120.0)).exp()
}

#[test]
fn oblique_gaussian_mtf_uses_sensor_direction_cosines() {
    let frequencies = [[0.0, 0.0], [20.0, 0.0], [0.0, 50.0], [15.0, -25.0]];
    for (angle, azimuth) in [(0.0, 0.0), (55.0, 90.0), (55.0, 35.0)] {
        let p = gaussian(angle, azimuth);
        for method in [MtfMethod::Direct, MtfMethod::Fft] {
            let values = otf_at(&p, &frequencies, method, &JobContext::new()).unwrap();
            for (nu, value) in frequencies.iter().zip(values) {
                let expected = gaussian_mtf(nu[0], nu[1]);
                assert!(
                    (value.norm() - expected).abs() < 1.5e-3,
                    "{method:?}, chief {angle}/{azimuth}, frequency {nu:?}: {} != {expected}",
                    value.norm()
                );
            }
        }
    }
}

#[test]
fn sensor_translation_and_reference_offset_cancel_in_complex_otf() {
    let p = gaussian(55.0, 35.0);
    let g = p.geometry();
    let [eu, ev, ew] = g.chief_frame();
    let offset = [0.017, -0.009];
    let mut shifted = p.with_added_opd(|u, v| {
        let (a, b) = (g.sin_max() * u, g.sin_max() * v);
        let q = eu * a + ev * b - ew * (1.0 - a * a - b * b).sqrt();
        (q.x * offset[0] + q.y * offset[1]) / 0.00055
    });
    shifted.set_image_offset_mm(offset);
    let frequencies = [[20.0, 0.0], [0.0, 50.0], [15.0, -25.0]];
    for method in [MtfMethod::Direct, MtfMethod::Fft] {
        let values = otf_at(&shifted, &frequencies, method, &JobContext::new()).unwrap();
        for (nu, value) in frequencies.iter().zip(values) {
            assert!((value.re - gaussian_mtf(nu[0], nu[1])).abs() < 1.5e-3);
            assert!(value.im.abs() < 1.5e-3, "{method:?}: {value}");
        }
    }
}

#[test]
fn wide_field_ideal_lens_matches_spherical_wave_integration() {
    use mklight_core::{
        ApertureSpec, Field, FieldKind, FieldSet, MaterialRef, OpticalSystem, Surface,
        SurfaceShape, Vec3, Wavelength, WavelengthSet,
    };
    use mklight_wave::{AmplitudeModel, MtfOrientation, PupilSettings, mtf_direct, trace_pupil};
    use rayon::prelude::*;

    let (f, radius, dz, theta) = (25.0, 2.0, 0.4, 55.0_f64.to_radians());
    let mut system = OpticalSystem::new_empty();
    system.aperture = ApertureSpec::EntrancePupilDiameter {
        value: 2.0 * radius,
    };
    system.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, theta.to_degrees())],
    };
    system.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.55,
            weight: 1.0,
        }],
        primary: 0,
    };
    system.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial { focal_length: f },
            f + dz,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    system.stop = 1;
    let tracer = mklight_analysis::tracer_for(system).unwrap();
    let pupil = trace_pupil(
        &tracer,
        tracer.field(0),
        0,
        &PupilSettings {
            grid: 256,
            rays: 129,
            amplitude: AmplitudeModel::Radiometric,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    let frequencies = [10.0, 20.0, 40.0];
    let actual = mtf_direct(
        &pupil.pupil,
        &frequencies,
        &MtfOrientation::default(),
        &JobContext::new(),
    )
    .unwrap();

    // An ideal lens supplies the spherical phase −|F−s| on its physical
    // plane. Integrate there, without a traced OPD, exit pupil, resampling
    // mask or Fourier-plane projection. Flux conservation supplies the
    // outgoing amplitude sqrt(cos(theta)/d_z).
    let focus = Vec3::new(0.0, f * theta.tan(), f);
    let image = Vec3::new(0.0, (f + dz) * theta.tan(), f + dz);
    let mut sources = Vec::new();
    let mut source_energy = 0.0;
    for ring in 0..48 {
        let r = radius * (ring as f64 + 0.5) / 48.0;
        let area = r * (radius / 48.0) * (2.0 * PI / 240.0);
        for angle in 0..240 {
            let phi = 2.0 * PI * (angle as f64 + 0.5) / 240.0;
            let source = Vec3::new(r * phi.cos(), r * phi.sin(), 0.0);
            let distance = (focus - source).norm();
            let cos_in = f / distance;
            let amplitude = (theta.cos() / cos_in).sqrt();
            source_energy += amplitude * amplitude * area;
            sources.push((source, distance, cos_in, amplitude * area));
        }
    }
    let size = 192;
    let spacing = 0.00125;
    let coord = |i: usize| (i as f64 - (size / 2) as f64) * spacing;
    let intensity: Vec<f64> = (0..size * size)
        .into_par_iter()
        .map(|k| {
            let point = image + Vec3::new(coord(k % size), coord(k / size), 0.0);
            let mut amplitude = num_complex::Complex64::default();
            for &(source, path, cos_in, weight) in &sources {
                let distance = (point - source).norm();
                let difference =
                    (point - focus).dot(point + focus - source * 2.0) / (distance + path);
                let obliquity = 0.5 * (cos_in + point.z / distance);
                amplitude += num_complex::Complex64::from_polar(
                    weight * obliquity / distance,
                    -2.0 * PI * difference / 0.00055,
                );
            }
            amplitude.norm_sqr()
        })
        .collect();
    let energy = intensity.iter().sum::<f64>();
    let fraction = energy * spacing * spacing / (0.00055_f64.powi(2) * source_energy);
    assert!(fraction > 0.97 && fraction < 1.01, "PSF energy {fraction}");
    for (axis, curve) in [(0, &actual.sagittal), (1, &actual.tangential)] {
        for (&frequency, &value) in frequencies.iter().zip(curve) {
            let otf: num_complex::Complex64 = intensity
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    let x = coord(if axis == 0 { k % size } else { k / size });
                    num_complex::Complex64::from_polar(v, -2.0 * PI * frequency * x)
                })
                .sum();
            // Flux conservation gives the whole-plane normalization. Using
            // only the captured PSF energy would overestimate its contrast.
            let expected = otf.norm() * spacing * spacing / (0.00055_f64.powi(2) * source_energy);
            assert!(
                (value - expected).abs() < 0.004,
                "axis {axis}, {frequency}: {value} != {expected}"
            );
        }
    }
}
