use mklight_analysis::{
    energy::{self, EncircledEnergySettings, GeometricMtfSettings},
    gpu,
    spot::{self, SpotDiagramSettings},
    tracer_for,
};
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
fn check(b: &dyn ComputeBackend) {
    let mut max_rms = 0.0f64;
    let mut max_point = 0.0f64;
    for (name, mut model) in samples::all() {
        for aim in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            model.ray_aiming = aim;
            let t = tracer_for(model.clone()).unwrap();
            for reference in [spot::SpotReference::Chief, spot::SpotReference::Centroid] {
                let s = SpotDiagramSettings {
                    rings: 3,
                    reference,
                    defocus: 0.02,
                    through_focus: Some(spot::ThroughFocus {
                        min: -0.1,
                        max: 0.1,
                        steps: 3,
                    }),
                    ..Default::default()
                };
                let a = spot::spot_diagram(&t, &s).unwrap();
                let c = JobContext::new();
                let g = gpu::spot_diagram_job(b, &t, &s, &c).unwrap();
                assert_eq!(c.progress.fraction(), 1.0);
                assert_eq!(a.fields.len(), g.fields.len(), "{name}");
                for (a, g) in a.fields.iter().zip(&g.fields) {
                    assert_eq!(a.points.len(), g.points.len(), "{name}");
                    assert_eq!(
                        a.polychromatic.rays_arrived, g.polychromatic.rays_arrived,
                        "{name}"
                    );
                    let e = (a.rms_radius - g.rms_radius).abs();
                    max_rms = max_rms.max(e);
                    assert!(
                        e <= 2e-5 + 0.005 * a.rms_radius.abs(),
                        "{name} RMS {} {}",
                        a.rms_radius,
                        g.rms_radius
                    );
                    for (a, g) in a.points.iter().zip(&g.points) {
                        let e = (a.x - g.x).abs().max((a.y - g.y).abs());
                        max_point = max_point.max(e);
                        assert!(e < 2e-5, "{name} point {e}");
                    }
                    for (a, g) in a.per_wavelength.iter().zip(&g.per_wavelength) {
                        assert_eq!(a.stats.rays_arrived, g.stats.rays_arrived);
                        assert!(
                            (a.stats.rms_radius_centroid - g.stats.rms_radius_centroid).abs()
                                < 2e-5
                        );
                    }
                    for (a, g) in a.through_focus.iter().zip(&g.through_focus) {
                        assert!((a.rms_radius_centroid - g.rms_radius_centroid).abs() < 2e-5);
                    }
                }
                let mut no = s.clone();
                no.include_points = false;
                let empty = gpu::spot_diagram_job(b, &t, &no, &JobContext::new()).unwrap();
                assert!(empty.fields.iter().all(|f| f.points.is_empty()));
                for (f, g) in empty.fields.iter().zip(&g.fields) {
                    assert_eq!(f.polychromatic, g.polychromatic);
                }
            }
            let s = EncircledEnergySettings {
                sampling: energy::RaySampling {
                    rings: 4,
                    ..Default::default()
                },
                samples: 17,
                defocus: 0.13,
                ..Default::default()
            };
            let a = energy::encircled_energy(&t, &s).unwrap();
            let g = gpu::encircled_energy_job(b, &t, &s, &JobContext::new()).unwrap();
            for (a, g) in a.fields.iter().zip(&g.fields) {
                assert_eq!(a.rays_arrived, g.rays_arrived);
                for (a, g) in a.encircled.iter().zip(&g.encircled) {
                    assert!((a - g).abs() < 0.025, "{name} EE {a} {g}");
                }
                assert!(
                    (a.r80 - g.r80).abs() < 2e-5,
                    "{name} r80 {} {}",
                    a.r80,
                    g.r80
                );
                assert!(g.encircled.windows(2).all(|w| w[0] <= w[1]));
            }
            let s = GeometricMtfSettings {
                sampling: s.sampling,
                frequency_samples: 5,
                max_frequency: Some(100.0),
                defocus: 0.13,
            };
            let a = energy::geometric_mtf(&t, &s).unwrap();
            let g = gpu::geometric_mtf_job(b, &t, &s, &JobContext::new()).unwrap();
            for (a, g) in a.curves.iter().zip(&g.curves) {
                for (a, g) in a.modulation.iter().zip(&g.modulation) {
                    assert!((a - g).abs() < 2e-3, "{name} MTF {a} {g}");
                }
            }
        }
    }
    eprintln!("P5_MAX rms_mm={max_rms:e} point_mm={max_point:e}");
}
#[test]
fn cpu_f32_complete_results() {
    check(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_complete_results_both_libraries() {
    for library in [
        mklight_metal::LibraryChoice::Precompiled,
        mklight_metal::LibraryChoice::RuntimeSource,
    ] {
        let b = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P5 requires host Metal");
        eprintln!("P5_DEVICE {:?} {:?}", b.info(), b.precision());
        assert!(b.precision().probe.error_free_transforms());
        check(&b);
        let cpu = CpuF32Backend::matching(&b.precision().probe);
        let t = tracer_for(samples::cooke_triplet()).unwrap();
        let s = SpotDiagramSettings {
            rings: 8,
            ..Default::default()
        };
        let a = gpu::spot_diagram_job(&b, &t, &s, &JobContext::new()).unwrap();
        let c = gpu::spot_diagram_job(&cpu, &t, &s, &JobContext::new()).unwrap();
        let r = gpu::spot_diagram_job(&b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.fields, c.fields);
        assert_eq!(a.fields, r.fields);
    }
}
#[test]
fn cancellation_is_not_a_reference_replacement() {
    let c = JobContext::new();
    c.cancel.cancel();
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    assert!(matches!(
        gpu::spot_diagram_job(&CpuF32Backend::new(), &t, &Default::default(), &c),
        Err(mklight_core::MkError::Cancelled)
    ));
}
