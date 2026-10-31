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
                    assert!((a - g).abs() < 2e-3, "{name} EE {a} {g}");
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

fn maps_and_records(b: &dyn ComputeBackend) {
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = mklight_analysis::field_maps::FieldMapSettings {
            grid: [3, 3],
            ..Default::default()
        };
        for kind in [
            mklight_analysis::field_maps::FieldAnalysis::PupilAberration,
            mklight_analysis::field_maps::FieldAnalysis::ChiefRayAngle,
        ] {
            let a = mklight_analysis::field_maps::field_map_job(&t, kind, &s, &JobContext::new())
                .unwrap();
            let g = gpu::field_map_job(b, &t, kind, &s, &JobContext::new()).unwrap();
            for (a, g) in a.image_mm.iter().flatten().zip(g.image_mm.iter().flatten()) {
                assert!(
                    (a - g).abs() < 2e-5 || a.is_nan() && g.is_nan(),
                    "{name} image {a} {g}"
                );
            }
            for (a, g) in a.values.iter().zip(&g.values) {
                assert!(
                    (a - g).abs()
                        < if matches!(
                            kind,
                            mklight_analysis::field_maps::FieldAnalysis::ChiefRayAngle
                        ) {
                            2e-4
                        } else {
                            (1e-3 * a.abs()).max(1e-5)
                        }
                        || a.is_nan() && g.is_nan(),
                    "{name} {} map {a} {g}",
                    kind.name()
                );
            }
        }
        let s = mklight_analysis::fans::FanSettings {
            samples: 11,
            ..Default::default()
        };
        let a = mklight_analysis::fans::ray_fan(&t, &s).unwrap();
        let g = gpu::ray_fan_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.fields.iter().zip(&g.fields) {
            for (a, g) in a
                .tangential
                .iter()
                .chain(&a.sagittal)
                .zip(g.tangential.iter().chain(&g.sagittal))
            {
                for (a, g) in a.values.iter().zip(&g.values) {
                    assert!(
                        (a - g).abs() < 2e-5 || a.is_nan() && g.is_nan(),
                        "{name} fan {a} {g}"
                    );
                }
            }
        }
        let s = mklight_trace::SemiDiameterSettings {
            azimuths: 16,
            ..Default::default()
        };
        let a = t.semi_diameters(&s);
        let (g, _) = gpu::semi_diameters_job(b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.sources, g.sources);
        assert_eq!(a.failed_rays, g.failed_rays, "{name}");
        for (a, g) in a.traced.iter().zip(&g.traced) {
            assert!((a - g).abs() < 2e-5, "{name} sizing {a} {g}");
        }
        let s = mklight_analysis::illumination::FootprintSettings {
            rings: 2,
            ..Default::default()
        };
        let a =
            mklight_analysis::footprint_overlay::footprint_overlay_job(&t, &s, &JobContext::new())
                .unwrap();
        let g = gpu::footprint_overlay_job(b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.sets.len(), g.sets.len());
        for (a, g) in a.sets.iter().zip(&g.sets) {
            assert_eq!(a.points, g.points, "{name}");
        }
        for (a, g) in a.points.iter().flatten().zip(g.points.iter().flatten()) {
            assert!((a - g).abs() < 2e-5, "{name} global {a} {g}");
        }
    }
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let s = energy::BestFocusSettings {
        sampling: energy::RaySampling {
            rings: 3,
            ..Default::default()
        },
        through_focus_samples: 3,
        ..Default::default()
    };
    let a = energy::best_focus(&t, &s).unwrap();
    let g = gpu::best_focus_job(b, &t, &s, &JobContext::new()).unwrap();
    assert!(
        (a.defocus - g.defocus).abs() < 2e-5,
        "focus {} {}",
        a.defocus,
        g.defocus
    );
    assert!((a.merit - g.merit).abs() < 2e-5);
}
#[test]
fn cpu_maps_and_records() {
    maps_and_records(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_maps_and_records() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P5 requires host Metal");
    maps_and_records(&b);
}
fn pencils(b: &dyn ComputeBackend) {
    let mut max = 0.0f64;
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = mklight_analysis::curvature::FieldCurvatureSettings {
            field_samples: 3,
            ..Default::default()
        };
        let a = mklight_analysis::curvature::field_curvature(&t, &s).unwrap();
        let g = gpu::field_curvature_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.curves.iter().zip(&g.curves) {
            for (a, g) in a
                .tangential
                .iter()
                .chain(&a.sagittal)
                .zip(g.tangential.iter().chain(&g.sagittal))
            {
                let e = (a - g).abs();
                max = max.max(e);
                assert!(
                    e < 1e-3 * (1.0 + a.abs())
                        || a.is_nan() && g.is_nan()
                        || a.is_infinite() && a == g,
                    "{name} Coddington {a} {g}"
                );
            }
            for (a, g) in a.distortion_percent.iter().zip(&g.distortion_percent) {
                assert!(
                    (a - g).abs() < 1e-4 || a.is_nan() && g.is_nan() || a.is_infinite() && a == g,
                    "{name} distortion {a} {g}"
                );
            }
        }
        let s = mklight_analysis::curvature::GridDistortionSettings {
            grid_size: 3,
            ..Default::default()
        };
        let a = mklight_analysis::curvature::grid_distortion(&t, &s).unwrap();
        let g = gpu::grid_distortion_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.points.iter().zip(&g.points) {
            for (a, g) in a
                .real
                .iter()
                .chain(&a.paraxial)
                .zip(g.real.iter().chain(&g.paraxial))
            {
                assert!(
                    (a - g).abs() < 2e-5 || a.is_nan() && g.is_nan(),
                    "{name} grid {a} {g}"
                );
            }
        }
    }
    eprintln!("P5_CODDINGTON_MAX mm={max:e}");
}
#[test]
fn cpu_pencils() {
    pencils(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_pencils() {
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
        pencils(&b);
    }
}
fn ri(b: &dyn ComputeBackend) {
    let mut max = 0.0f64;
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = mklight_analysis::illumination::RelativeIlluminationSettings {
            field_samples: 3,
            radial_samples: if b.info().kind == mklight_gpu::BackendKind::Metal {
                12
            } else {
                3
            },
            azimuth_samples: if b.info().kind == mklight_gpu::BackendKind::Metal {
                48
            } else {
                12
            },
            ..Default::default()
        };
        let a = mklight_analysis::illumination::relative_illumination(&t, &s).unwrap();
        let g = gpu::relative_illumination_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.relative_illumination.iter().zip(&g.relative_illumination) {
            max = max.max((a - g).abs());
            assert!(
                (a - g).abs() < 1e-3 || a.is_nan() && g.is_nan(),
                "{name} RI {a} {g}"
            );
        }
        for (a, g) in a.projected_solid_angle.iter().zip(&g.projected_solid_angle) {
            assert!(
                (a - g).abs() < 1e-3 * a.abs() + 1e-10 || a.is_nan() && g.is_nan(),
                "{name} omega {a} {g}"
            );
        }
        assert!(
            g.run_info
                .warnings
                .iter()
                .all(|s| !s.contains("pupil nodes have no ray")),
            "{name} {:?}",
            g.run_info.warnings
        );
    }
    eprintln!("P5_RI_MAX absolute={max:e}");
}
#[test]
fn cpu_ri() {
    ri(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_ri() {
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
        ri(&b);
    }
}
fn layouts(b: &dyn ComputeBackend) {
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        for section in [
            mklight_analysis::layout::LayoutSection::Yz,
            mklight_analysis::layout::LayoutSection::Xz,
        ] {
            let s = mklight_analysis::layout::LayoutSettings {
                profile_samples: 9,
                ring_samples: 12,
                mesh_rings: 2,
                section,
                rays_per_field: 3,
                ..Default::default()
            };
            let a = mklight_analysis::layout::layout(&t, &s).unwrap();
            let g = gpu::layout_job(b, &t, &s, &JobContext::new()).unwrap();
            assert_eq!(a.polylines.len(), g.polylines.len(), "{name}");
            for (a, g) in a.polylines.iter().zip(&g.polylines) {
                assert_eq!(a.kind, g.kind);
                assert_eq!(a.vignetted, g.vignetted);
                assert_eq!(a.in_section, g.in_section, "{name}");
                assert_eq!(a.points.len(), g.points.len(), "{name} {:?}", a.kind);
                for (a, g) in a.points.iter().flatten().zip(g.points.iter().flatten()) {
                    assert!(
                        (a - g).abs() < 2e-5 || a.is_nan() && g.is_nan(),
                        "{name} layout {a} {g}"
                    );
                }
            }
            assert_eq!(a.meshes.len(), g.meshes.len());
            for (a, g) in a.meshes.iter().zip(&g.meshes) {
                assert_eq!(a.triangles.len(), g.triangles.len());
                for (a, g) in a
                    .triangles
                    .iter()
                    .flatten()
                    .flatten()
                    .zip(g.triangles.iter().flatten().flatten())
                {
                    assert!((a - g).abs() < 2e-5, "{name} mesh {a} {g}");
                }
            }
            let s = mklight_analysis::fans::FanSettings {
                samples: 11,
                ..Default::default()
            };
            let a = mklight_analysis::fans::opd_fan(&t, &s).unwrap();
            let g = gpu::opd_fan_job(b, &t, &s, &JobContext::new()).unwrap();
            for (a, g) in a.fields.iter().zip(&g.fields) {
                for (a, g) in a
                    .tangential
                    .iter()
                    .chain(&a.sagittal)
                    .zip(g.tangential.iter().chain(&g.sagittal))
                {
                    for (a, g) in a.values.iter().zip(&g.values) {
                        assert!(
                            (a - g).abs() < 5e-3 || a.is_nan() && g.is_nan(),
                            "{name} OPD {a} {g}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn cpu_layout_and_opd() {
    layouts(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_layout_and_opd() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P5 requires host Metal");
    layouts(&b);
}

mod common;
fn known(b: &dyn ComputeBackend) {
    let t = tracer_for(common::ideal_lens(100.0, 10.0, &[0.0, 20.0])).unwrap();
    let settings = mklight_analysis::field_maps::FieldMapSettings {
        grid: [3, 3],
        ..Default::default()
    };
    let result = gpu::field_map_job(
        b,
        &t,
        mklight_analysis::field_maps::FieldAnalysis::RelativeIllumination,
        &settings,
        &JobContext::new(),
    )
    .unwrap();
    let disk = |h: f64| {
        let q = 10000.0 + h * h;
        std::f64::consts::PI
            * 0.5
            * (1.0 - (q - 25.0) / ((q + 25.0).powi(2) - 100.0 * h * h).sqrt())
    };
    for (i, row) in result.values.chunks_exact(3).enumerate() {
        let p = result.image_mm[i];
        if p[0].is_nan() {
            assert!(row.iter().all(|v| v.is_nan()));
            continue;
        }
        let omega = disk(p[0].hypot(p[1]));
        assert!((row[1] / omega - 1.0).abs() < 1e-6, "disk {row:?} {omega}");
        assert!((row[0] - omega / disk(0.0)).abs() < 1e-6);
    }
    let r = gpu::field_map_job(
        b,
        &t,
        mklight_analysis::field_maps::FieldAnalysis::PupilAberration,
        &settings,
        &JobContext::new(),
    )
    .unwrap();
    for row in r.values.chunks_exact(6) {
        assert!(row[0].abs() < 1e-8 && row[1].abs() < 1e-8);
        assert!((row[2] - 1.0).abs() < 1e-5 && (row[3] - 1.0).abs() < 1e-5);
    }
    let r = gpu::field_map_job(
        b,
        &t,
        mklight_analysis::field_maps::FieldAnalysis::ChiefRayAngle,
        &settings,
        &JobContext::new(),
    )
    .unwrap();
    for (i, row) in r.values.chunks_exact(5).enumerate() {
        let p = r.image_mm[i];
        assert!((row[0] - (p[0].hypot(p[1]) / 100.0).atan().to_degrees()).abs() < 2e-4);
    }
    let energy = gpu::encircled_energy_job(
        b,
        &t,
        &EncircledEnergySettings {
            sampling: energy::RaySampling {
                rings: 4,
                ..Default::default()
            },
            samples: 5,
            defocus: 5.0,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(energy.run_info.backend, "cpu-f64");
    assert_eq!(
        energy.run_info.settings["adapter_fallbacks"]["stage"],
        "energy_bin_classification"
    );
    assert!((energy.fields[0].encircled[0] - 1.0 / 49.0).abs() < 1e-12);
    let mut model = common::ideal_lens(100.0, 10.0, &[0.0]);
    model.surfaces[1].thickness = 105.0;
    let t = tracer_for(model).unwrap();
    let s = energy::BestFocusSettings {
        sampling: energy::RaySampling {
            rings: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let focus = gpu::best_focus_job(b, &t, &s, &JobContext::new()).unwrap();
    assert!(
        (focus.defocus + 5.0).abs() < 1e-6,
        "analytic focus {}",
        focus.defocus
    );
    assert!(focus.merit < 1e-7);
    let afocal = tracer_for(common::infinite_system(
        10.0,
        vec![common::surf(
            1,
            mklight_core::SurfaceShape::Plane,
            10.0,
            mklight_core::MaterialRef::Air,
        )],
        1,
        &[0.0],
    ))
    .unwrap();
    let settings = mklight_analysis::seidel::LongitudinalSettings {
        pupil_samples: 3,
        ..Default::default()
    };
    let reference = mklight_analysis::seidel::longitudinal_aberration(&afocal, &settings).unwrap();
    let result =
        gpu::longitudinal_aberration_job(b, &afocal, &settings, &JobContext::new()).unwrap();
    assert_eq!(reference.curves[0].focus, result.curves[0].focus);
}
fn remaining(b: &dyn ComputeBackend) {
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = mklight_analysis::seidel::LateralColourSettings {
            field_samples: 3,
            ..Default::default()
        };
        let a = mklight_analysis::seidel::lateral_colour(&t, &s).unwrap();
        let g = gpu::lateral_colour_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.lateral_colour.iter().zip(&g.lateral_colour) {
            assert!(
                (a - g).abs() < 2e-5 || a.is_nan() && g.is_nan(),
                "{name} lateral {a} {g}"
            );
        }
        let s = mklight_analysis::seidel::LongitudinalSettings {
            pupil_samples: 9,
            ..Default::default()
        };
        let a = mklight_analysis::seidel::longitudinal_aberration(&t, &s).unwrap();
        let g = gpu::longitudinal_aberration_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.curves.iter().zip(&g.curves) {
            for (a, g) in a.focus.iter().zip(&g.focus) {
                assert!(
                    (a - g).abs() < 1e-3 * (1.0 + a.abs())
                        || a.is_nan() && g.is_nan()
                        || a.is_infinite() && a == g,
                    "{name} longitudinal {a} {g}"
                );
            }
        }
        let s = mklight_analysis::illumination::FootprintSettings {
            surface: 1,
            rings: 3,
            ..Default::default()
        };
        let a = mklight_analysis::illumination::footprint(&t, &s).unwrap();
        let g = gpu::footprint_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.rays.iter().zip(&g.rays) {
            assert_eq!(a.lost_before, g.lost_before);
            assert_eq!(a.clipped.len(), g.clipped.len());
            assert_eq!(a.points.len(), g.points.len());
            for (a, g) in a.points.iter().flatten().zip(g.points.iter().flatten()) {
                assert!((a - g).abs() < 2e-5, "{name} footprint {a} {g}");
            }
        }
    }
}
#[test]
fn cpu_known_and_chromatic() {
    known(&CpuF32Backend::new());
    remaining(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_known_and_chromatic() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    known(&b);
    remaining(&b);
}
#[cfg(target_os = "macos")]
#[test]
fn metal_concurrent_cancel_and_release() {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };
    let b = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("host Metal required"),
    );
    let t = Arc::new(tracer_for(samples::cooke_triplet()).unwrap());
    let settings = Arc::new(SpotDiagramSettings {
        rings: 24,
        include_points: false,
        ..Default::default()
    });
    let mut threads = Vec::new();
    for _ in 0..3 {
        let (b, t, s) = (b.clone(), t.clone(), settings.clone());
        threads.push(std::thread::spawn(move || {
            gpu::spot_diagram_job(&*b, &t, &s, &JobContext::new())
                .unwrap()
                .fields
        }));
    }
    let result = threads.remove(0).join().unwrap();
    for thread in threads {
        assert_eq!(result, thread.join().unwrap());
    }
    let ctx = JobContext::new();
    let sub = ctx.clone();
    let bb = b.clone();
    let tt = t.clone();
    let thread = std::thread::spawn(move || {
        gpu::field_map_job(
            &*bb,
            &tt,
            mklight_analysis::field_maps::FieldAnalysis::RelativeIllumination,
            &mklight_analysis::field_maps::FieldMapSettings {
                grid: [33, 33],
                ..Default::default()
            },
            &sub,
        )
    });
    let start = Instant::now();
    let mut previous = 0.0;
    while ctx.progress.stage() != "geometry" {
        let p = ctx.progress.fraction();
        assert!(p >= previous);
        previous = p;
        if start.elapsed() > Duration::from_secs(60) {
            ctx.cancel.cancel();
            panic!("device work did not start within 60 seconds");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let start = Instant::now();
    ctx.cancel.cancel();
    assert!(matches!(
        thread.join().unwrap(),
        Err(mklight_core::MkError::Cancelled)
    ));
    eprintln!(
        "P5_CANCEL seconds={} progress={}",
        start.elapsed().as_secs_f64(),
        ctx.progress.fraction()
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(ctx.progress.fraction() < 1.0);
    // Successful work after cancellation verifies independent jobs and released
    // plan resources, with no CPU replacement produced by the cancelled job.
    let r = gpu::spot_diagram_job(&*b, &t, &settings, &JobContext::new()).unwrap();
    assert_eq!(result, r.fields);
}
fn prepared_system(b: &dyn ComputeBackend) {
    use mklight_gpu::{
        resident::{ResidentOp, ResidentPlan},
        systems::{Candidate, CompiledSystem},
    };
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    let compiled = CompiledSystem::compile(
        t.system().snapshot.clone(),
        t.system().library.clone(),
        vec![],
        None,
        &ctx,
    )
    .unwrap();
    let uploaded = compiled.upload(b, &ctx).unwrap();
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    let prepared = uploaded
        .append(
            b,
            &mut p,
            &[Candidate {
                id: 42,
                values: vec![],
            }],
            &ctx,
        )
        .unwrap();
    let geometry = gpu::Geometry::new(b, &t, &ctx).unwrap();
    let points: Vec<_> = t.fields().iter().map(|f| f.point).collect();
    let wavelengths: Vec<_> = (0..t.num_wavelengths()).collect();
    let grid = geometry
        .append_grid(
            &mut p,
            &points,
            &wavelengths,
            &[[0.0; 2]],
            t.aiming(),
            t.system().image_index(),
            false,
            true,
            &ctx,
        )
        .unwrap();
    let output = geometry.coddington(&mut p, grid.hits).unwrap();
    for op in &mut p.operations {
        match op {
            ResidentOp::Aiming(mklight_gpu::aiming::Request::GenerateAim(r)) => {
                r.systems = prepared.system_set()
            }
            ResidentOp::Aiming(mklight_gpu::aiming::Request::TraceResident(r)) => {
                r.systems = prepared.system_set()
            }
            ResidentOp::Geometry(mklight_gpu::geometry::Request::Coddington(r)) => {
                r.systems = prepared.system_set()
            }
            _ => {}
        }
    }
    p.outputs = vec![output];
    let result = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let view = b.read_view(result.outputs[0].view()).unwrap();
    let v = view.as_slice::<mklight_gpu::Df32>().unwrap();
    for (fi, f) in t.fields().iter().enumerate() {
        for (wi, &w) in wavelengths.iter().enumerate() {
            let reference = mklight_analysis::curvature::coddington_focus(&t, f, w).unwrap();
            let i = (fi * wavelengths.len() + wi) * 3;
            assert!(
                (v[i].to_f64() - reference.tangential).abs()
                    < 1e-3 * (1.0 + reference.tangential.abs())
            );
            assert!(
                (v[i + 1].to_f64() - reference.sagittal).abs()
                    < 1e-3 * (1.0 + reference.sagittal.abs())
            );
        }
    }
    uploaded.release(b).unwrap();
}
#[test]
fn cpu_p4_composition() {
    prepared_system(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_p4_composition() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    prepared_system(&b);
}
#[cfg(target_os = "macos")]
#[test]
fn metal_resident_integrals_match_ftz() {
    use mklight_gpu::resident::ResidentPlan;
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let run = |b: &dyn ComputeBackend| {
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let g = gpu::Geometry::new(b, &t, &JobContext::new()).unwrap();
        let buffers = g
            .illumination(
                &mut p,
                &[[0.0, 0.0], [0.0, 1.0]],
                t.system().primary_wavelength,
                3,
                12,
                &JobContext::new(),
            )
            .unwrap();
        p.outputs = vec![buffers.integrals];
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(r.stats.copied_bytes, 0);
        b.read_view(r.outputs[0].view()).unwrap().words().to_vec()
    };
    let a = run(&b);
    let c = run(&cpu);
    assert_eq!(a, c);
    assert_eq!(a, run(&b));
}
#[cfg(target_os = "macos")]
#[test]
fn metal_ri_rim_regression() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let t = tracer_for(samples::petzval_asphere()).unwrap();
    let s = mklight_analysis::illumination::RelativeIlluminationSettings {
        field_samples: 3,
        ..Default::default()
    };
    let a = mklight_analysis::illumination::relative_illumination(&t, &s).unwrap();
    let g = gpu::relative_illumination_job(&b, &t, &s, &JobContext::new()).unwrap();
    eprintln!(
        "P5_RI_RIM reference={:?} resident={:?} warnings={:?}",
        a.relative_illumination, g.relative_illumination, g.run_info.warnings
    );
    for (a, g) in a.relative_illumination.iter().zip(&g.relative_illumination) {
        assert!((a - g).abs() < 1e-3);
    }
}
fn qualified_energy(b: &dyn ComputeBackend) {
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let settings = EncircledEnergySettings {
        sampling: energy::RaySampling {
            rings: 32,
            fields: mklight_analysis::util::FieldSelection::Index { index: 1 },
            wavelengths: mklight_analysis::util::WavelengthSelection::Primary,
            ..Default::default()
        },
        ..Default::default()
    };
    let grid = gpu::ray_grid_job(b, &t, &settings.sampling, &JobContext::new()).unwrap();
    let fields = vec![t.field(1).clone()];
    let reference = mklight_trace::trace_grid_job(
        &t,
        &mklight_trace::Grid {
            fields: &fields,
            wavelengths: &grid.wavelengths,
            pupil: &grid.pupil,
        },
        &t.options(),
        &JobContext::new(),
    )
    .unwrap();
    let mut error = 0.0f64;
    for (a, g) in reference.iter().zip(&grid.rays) {
        assert_eq!(a.status.is_ok(), g.is_some());
        if let Some(g) = g {
            error = error
                .max((a.point.x - g.point[0]).abs())
                .max((a.point.y - g.point[1]).abs());
        }
    }
    let bound = (3.0 * error).max(1e-10);
    let result =
        gpu::encircled_energy_with_uncertainty_job(b, &t, &settings, bound, &JobContext::new())
            .unwrap();
    assert_eq!(result.run_info.backend, b.info().backend_id());
    let reference = energy::encircled_energy(&t, &settings).unwrap();
    for (a, g) in reference.fields.iter().zip(&result.fields) {
        for (a, g) in a
            .encircled
            .iter()
            .chain(&a.ensquared)
            .zip(g.encircled.iter().chain(&g.ensquared))
        {
            assert!((a - g).abs() < 2e-3);
        }
        assert!((a.r80 - g.r80).abs() < 2e-5);
    }
}
#[test]
fn cpu_qualified_energy() {
    qualified_energy(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_qualified_energy() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    qualified_energy(&b);
}
fn scoped_record_views(b: &dyn ComputeBackend) {
    use mklight_gpu::{
        aiming::{ResidentHits, ResidentRays},
        geometry::{self, FieldData},
        resident::*,
    };
    let t = tracer_for(samples::singlet()).unwrap();
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    let f = b
        .fill_buffer(BufferDesc::new(ElementType::F32, [15, 1]), &mut |mut w| {
            let f = w.as_mut_slice::<f32>()?;
            f.fill(0.0);
            f[0] = 2.0;
            f[1] = 3.0;
            f[3] = 0.5;
            let d = mklight_gpu::Df32::from_f64(0.75f64.sqrt());
            f[5] = d.hi;
            f[14] = d.lo;
            Ok(())
        })
        .unwrap();
    let floats = p.import(f.view());
    let hi = b
        .fill_buffer(
            BufferDesc::new(ElementType::F32, [4, 1, 13]),
            &mut |mut w| {
                let f = w.as_mut_slice::<f32>()?;
                f.fill(0.0);
                f[13] = 7.0;
                f[14] = 11.0;
                f[18] = 1.0;
                f[2 * 13 + 10] = 1.0;
                f[3 * 13 + 8] = 1.0;
                Ok(())
            },
        )
        .unwrap();
    let lo = b
        .fill_buffer(
            BufferDesc::new(ElementType::F32, [4, 1, 13]),
            &mut |mut w| {
                w.as_mut_slice::<f32>()?.fill(0.0);
                Ok(())
            },
        )
        .unwrap();
    let window = |h: &BufferHandle| BufferView {
        buffer: h.clone(),
        offset: 13,
        desc: BufferDesc::new(ElementType::F32, [2, 1, 13]),
        stride: 1,
    };
    let records = p.import(window(&hi));
    let low = p.import(window(&lo));
    let status = b
        .fill_buffer(BufferDesc::new(ElementType::U32, [3, 1]), &mut |mut w| {
            w.as_mut_slice::<u32>()?.fill(0);
            Ok(())
        })
        .unwrap();
    let status = p.import(status.view());
    let failed = b
        .fill_buffer(BufferDesc::new(ElementType::U32, [3, 1]), &mut |mut w| {
            let v = w.as_mut_slice::<u32>()?;
            v.fill(0);
            v[0] = mklight_gpu::consts::MKL_RAY_MISS;
            Ok(())
        })
        .unwrap();
    let failed = p.import(failed.view());
    let rays = ResidentRays {
        floats,
        indices: status,
        logical_ids: status,
        report: status,
        count: 1,
    };
    let hits = ResidentHits {
        rays,
        status,
        records: Some(records),
        record_low: Some(low),
    };
    let angle = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 5]));
    p.operations
        .push(ResidentOp::Geometry(geometry::Request::FieldData(
            FieldData {
                record_slots: 2,
                hits: hits.clone(),
                kind: 0,
                pupil_step: mklight_gpu::Df32::ZERO,
                output: angle,
            },
        )));
    let source = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 5]));
    let mut hits = hits;
    hits.status = failed;
    p.operations
        .push(ResidentOp::Geometry(geometry::Request::FieldData(
            FieldData {
                record_slots: 2,
                hits,
                kind: 2,
                pupil_step: mklight_gpu::Df32::from_f32(1.0),
                output: source,
            },
        )));
    p.outputs = vec![angle, source];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let v = b.read_view(r.outputs[0].view()).unwrap();
    let v = v.as_slice::<mklight_gpu::Df32>().unwrap();
    assert!((v[2].to_f64() - 30.0).abs() < 2e-4);
    let v = b.read_view(r.outputs[1].view()).unwrap();
    let v = v.as_slice::<mklight_gpu::Df32>().unwrap();
    assert!(v[0].hi.is_nan() && v[1].hi.is_nan());
    assert_eq!([v[2].to_f64(), v[3].to_f64()], [7.0, 11.0]);
}
#[test]
fn cpu_scoped_record_views() {
    scoped_record_views(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_scoped_record_views() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    scoped_record_views(&b);
}
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod p1_fixtures;
fn telecentric(b: &dyn ComputeBackend) {
    for (name, model) in p1_fixtures::special_systems()
        .into_iter()
        .filter(|(n, _)| n.contains("telecentric"))
    {
        let t = tracer_for(model).unwrap();
        let s = mklight_analysis::field_maps::FieldMapSettings {
            grid: [3, 3],
            ..Default::default()
        };
        for kind in [
            mklight_analysis::field_maps::FieldAnalysis::PupilAberration,
            mklight_analysis::field_maps::FieldAnalysis::ChiefRayAngle,
        ] {
            let a = mklight_analysis::field_maps::field_map_job(&t, kind, &s, &JobContext::new())
                .unwrap();
            let g = gpu::field_map_job(b, &t, kind, &s, &JobContext::new()).unwrap();
            for (a, g) in a.values.iter().zip(&g.values) {
                assert!(
                    (a - g).abs() < 2e-4 || a.is_nan() && g.is_nan(),
                    "{name} map {a} {g}"
                );
            }
        }
        let s = mklight_analysis::fans::FanSettings {
            samples: 9,
            ..Default::default()
        };
        let a = mklight_analysis::fans::opd_fan(&t, &s).unwrap();
        let g = gpu::opd_fan_job(b, &t, &s, &JobContext::new()).unwrap();
        for (a, g) in a.fields.iter().zip(&g.fields) {
            for (a, g) in a
                .tangential
                .iter()
                .chain(&a.sagittal)
                .zip(g.tangential.iter().chain(&g.sagittal))
            {
                for (a, g) in a.values.iter().zip(&g.values) {
                    assert!(
                        (a - g).abs() < 5e-3 || a.is_nan() && g.is_nan(),
                        "{name} OPD {a} {g}"
                    );
                }
            }
        }
    }
}
#[test]
fn cpu_telecentric_limits() {
    telecentric(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_telecentric_limits() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    telecentric(&b);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tiled_footprints() {
    use mklight_analysis::{
        footprint_overlay, illumination::FootprintSettings, util::PupilPattern,
    };
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let s = FootprintSettings {
        pattern: PupilPattern::Square,
        rings: 80,
        ..Default::default()
    };
    let a = footprint_overlay::footprint_overlay_job(&t, &s, &JobContext::new()).unwrap();
    let g = gpu::footprint_overlay_job(&b, &t, &s, &JobContext::new()).unwrap();
    for (a, g) in a.sets.iter().zip(&g.sets) {
        assert_eq!(a.points, g.points);
    }
    assert_eq!(a.points.len(), g.points.len());
    for (a, g) in a.points.iter().flatten().zip(g.points.iter().flatten()) {
        assert!((a - g).abs() < 2e-5);
    }
    let execution = &g.run_info.settings["resident_execution"];
    assert!(!execution["extra_plans"].as_array().unwrap().is_empty());
    assert!(execution["plan_stats"]["peak_bytes"].as_u64().unwrap() < 256 * 1024 * 1024);
}
