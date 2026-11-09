use mklight_analysis::{
    energy::{self, EncircledEnergySettings, GeometricMtfSettings},
    gpu,
    spot::{self, SpotDiagramSettings},
    tracer_for,
};
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
/// Equal-length element-wise comparison with the f64 reference. Non-finite
/// reference values (undefined or infinite) must be reproduced exactly.
fn close(what: &str, a: &[f64], g: &[f64], tol: impl Fn(f64) -> f64) {
    assert_eq!(a.len(), g.len(), "{what}: length");
    for (&a, &g) in a.iter().zip(g) {
        let ok = if a.is_finite() {
            (a - g).abs() <= tol(a)
        } else {
            a == g || a.is_nan() && g.is_nan()
        };
        assert!(ok, "{what}: reference {a}, resident {g}");
    }
}
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
                    through_focus: Some(spot::ThroughFocus::symmetric(0.1, 3)),
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
                    let spot = |v: f64| 2e-5 + 0.005 * v.abs();
                    assert_eq!(a.per_wavelength.len(), g.per_wavelength.len());
                    for (a, g) in a.per_wavelength.iter().zip(&g.per_wavelength) {
                        assert_eq!(a.stats.rays_arrived, g.stats.rays_arrived);
                        close(
                            &format!("{name} monochromatic RMS"),
                            &[a.stats.rms_radius_centroid, a.stats.rms_radius_chief],
                            &[g.stats.rms_radius_centroid, g.stats.rms_radius_chief],
                            spot,
                        );
                    }
                    let focus = |f: &spot::FieldSpot| -> Vec<f64> {
                        f.through_focus
                            .iter()
                            .flat_map(|s| [s.defocus, s.rms_radius_centroid, s.rms_radius_chief])
                            .collect()
                    };
                    close(&format!("{name} through focus"), &focus(a), &focus(g), spot);
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
            // Few heavy rays near bin edges under the full position budget:
            // usually the counted reference replacement (qualified_energy
            // covers the resident curve).
            if g.run_info.backend != b.info().backend_id() {
                let fallback = &g.run_info.settings["adapter_fallbacks"];
                assert_eq!(fallback["stage"], "energy_bin_classification", "{name}");
            }
            close(&format!("{name} EE radii"), &a.radius, &g.radius, |_| 2e-5);
            assert_eq!(a.fields.len(), g.fields.len());
            for (a, g) in a.fields.iter().zip(&g.fields) {
                assert_eq!(a.rays_arrived, g.rays_arrived);
                close(&format!("{name} EE"), &a.encircled, &g.encircled, |_| 2e-3);
                close(
                    &format!("{name} ensquared"),
                    &a.ensquared,
                    &g.ensquared,
                    |_| 2e-3,
                );
                close(
                    &format!("{name} EE radii 50/80/90"),
                    &[a.r50, a.r80, a.r90, a.half_width80],
                    &[g.r50, g.r80, g.r90, g.half_width80],
                    |_| 2e-5,
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
            assert_eq!(a.curves.len(), g.curves.len());
            for (a, g) in a.curves.iter().zip(&g.curves) {
                assert_eq!((a.field, a.orientation), (g.field, g.orientation));
                close(&format!("{name} MTF"), &a.modulation, &g.modulation, |_| {
                    2e-3
                });
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
            let flat = |v: &[[f64; 2]]| v.iter().flatten().copied().collect::<Vec<_>>();
            close(
                &format!("{name} map image"),
                &flat(&a.image_mm),
                &flat(&g.image_mm),
                |_| 2e-5,
            );
            close(
                &format!("{name} {}", kind.name()),
                &a.values,
                &g.values,
                |a| match kind {
                    // Degrees, from the direction-cosine budget.
                    mklight_analysis::field_maps::FieldAnalysis::ChiefRayAngle => 2e-4,
                    _ => 1e-5 * (1.0 + a.abs()),
                },
            );
        }
        let s = mklight_analysis::fans::FanSettings {
            samples: 11,
            ..Default::default()
        };
        let a = mklight_analysis::fans::ray_fan(&t, &s).unwrap();
        let g = gpu::ray_fan_job(b, &t, &s, &JobContext::new()).unwrap();
        close(&format!("{name} ray fan"), &fans(&a), &fans(&g), |_| 2e-5);
        let s = mklight_trace::SemiDiameterSettings {
            azimuths: 16,
            ..Default::default()
        };
        let a = t.semi_diameters(&s);
        let (g, _) = gpu::semi_diameters_job(b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.sources, g.sources);
        assert_eq!(a.failed_rays, g.failed_rays, "{name}");
        close(&format!("{name} sizing"), &a.traced, &g.traced, |_| 2e-5);
        close(&format!("{name} sizing"), &a.values, &g.values, |_| 2e-5);
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
        let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
        close(
            &format!("{name} global footprint"),
            &flat(&a.points),
            &flat(&g.points),
            |_| 2e-5,
        );
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
    assert_eq!(a.fields, g.fields);
    let spot = |v: f64| 2e-5 + 0.005 * v.abs();
    close(
        "best focus",
        &[a.defocus, a.mean_square_defocus],
        &[g.defocus, g.mean_square_defocus],
        |_| 2e-5,
    );
    close("best focus range", &a.range, &g.range, |_| 2e-5);
    close(
        "best focus merit",
        &[a.merit, a.merit_at_zero],
        &[g.merit, g.merit_at_zero],
        spot,
    );
    close("best focus field RMS", &a.field_rms, &g.field_rms, spot);
    close(
        "best focus field RMS",
        &a.field_rms_at_zero,
        &g.field_rms_at_zero,
        spot,
    );
    close(
        "best focus through focus",
        &a.through_focus_merit,
        &g.through_focus_merit,
        spot,
    );
}
/// Tangential then sagittal fan values and cross components of every curve.
fn fans(r: &mklight_analysis::fans::FanResult) -> Vec<f64> {
    r.fields
        .iter()
        .flat_map(|f| f.tangential.iter().chain(&f.sagittal))
        .flat_map(|c| c.values.iter().chain(&c.cross))
        .copied()
        .collect()
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
        assert_eq!(a.curves.len(), g.curves.len());
        for (a, g) in a.curves.iter().zip(&g.curves) {
            for (a, g) in a
                .tangential
                .iter()
                .chain(&a.sagittal)
                .zip(g.tangential.iter().chain(&g.sagittal))
            {
                if a.is_finite() {
                    max = max.max((a - g).abs());
                }
            }
            let focal = |v: f64| 1e-6 * (1.0 + v.abs());
            close(
                &format!("{name} Coddington"),
                &a.tangential,
                &g.tangential,
                focal,
            );
            close(
                &format!("{name} Coddington"),
                &a.sagittal,
                &g.sagittal,
                focal,
            );
            // Finite-difference pencils divide position errors by the close-ray step.
            let pencil = |v: f64| 1e-4 * (1.0 + v.abs());
            close(
                &format!("{name} close pencil"),
                &a.tangential_close_rays,
                &g.tangential_close_rays,
                pencil,
            );
            close(
                &format!("{name} close pencil"),
                &a.sagittal_close_rays,
                &g.sagittal_close_rays,
                pencil,
            );
            close(
                &format!("{name} real height"),
                &a.real_height,
                &g.real_height,
                |_| 2e-5,
            );
            for (what, a, g) in [
                ("distortion", &a.distortion_percent, &g.distortion_percent),
                (
                    "f-tan distortion",
                    &a.f_tan_theta_distortion_percent,
                    &g.f_tan_theta_distortion_percent,
                ),
                (
                    "f-theta distortion",
                    &a.f_theta_distortion_percent,
                    &g.f_theta_distortion_percent,
                ),
            ] {
                close(&format!("{name} {what}"), a, g, |_| 1e-4);
            }
        }
        let s = mklight_analysis::curvature::GridDistortionSettings {
            grid_size: 3,
            ..Default::default()
        };
        let a = mklight_analysis::curvature::grid_distortion(&t, &s).unwrap();
        let g = gpu::grid_distortion_job(b, &t, &s, &JobContext::new()).unwrap();
        let grid = |r: &mklight_analysis::curvature::GridDistortionResult| -> Vec<f64> {
            r.points
                .iter()
                .flat_map(|p| p.real.into_iter().chain(p.paraxial))
                .collect()
        };
        close(&format!("{name} grid"), &grid(&a), &grid(&g), |_| 2e-5);
        let percent = |r: &mklight_analysis::curvature::GridDistortionResult| -> Vec<f64> {
            r.points.iter().map(|p| p.distortion_percent).collect()
        };
        close(
            &format!("{name} grid distortion"),
            &percent(&a),
            &percent(&g),
            |_| 1e-4,
        );
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
            if a.is_finite() {
                max = max.max((a - g).abs());
            }
        }
        close(
            &format!("{name} RI"),
            &a.relative_illumination,
            &g.relative_illumination,
            |_| 1e-5,
        );
        close(
            &format!("{name} RI solid angle"),
            &a.projected_solid_angle,
            &g.projected_solid_angle,
            |a| 1e-5 * a.abs(),
        );
        close(
            &format!("{name} RI vignetting"),
            &a.vignetting_fraction,
            &g.vignetting_fraction,
            |_| 1e-5,
        );
        close(
            &format!("{name} RI CRA"),
            &a.chief_ray_angle_deg,
            &g.chief_ray_angle_deg,
            |_| 2e-4,
        );
        close(
            &format!("{name} RI cos4"),
            &a.cos4_object,
            &g.cos4_object,
            |_| 1e-6,
        );
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
                assert_eq!(
                    (a.surface, a.field, a.wavelength),
                    (g.surface, g.field, g.wavelength)
                );
                let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
                close(
                    &format!("{name} layout {:?}", a.kind),
                    &flat(&a.points),
                    &flat(&g.points),
                    |_| 2e-5,
                );
            }
            assert_eq!(a.meshes.len(), g.meshes.len());
            for (a, g) in a.meshes.iter().zip(&g.meshes) {
                let flat =
                    |v: &[[[f64; 3]; 3]]| v.iter().flatten().flatten().copied().collect::<Vec<_>>();
                close(
                    &format!("{name} mesh"),
                    &flat(&a.triangles),
                    &flat(&g.triangles),
                    |_| 2e-5,
                );
            }
            assert_eq!(a.elements.len(), g.elements.len());
            for (a, g) in a.elements.iter().zip(&g.elements) {
                let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
                close(
                    &format!("{name} element"),
                    &flat(&a.outline),
                    &flat(&g.outline),
                    |_| 2e-5,
                );
            }
            close(
                &format!("{name} bounds"),
                &[a.bounds.min, a.bounds.max].concat(),
                &[g.bounds.min, g.bounds.max].concat(),
                |_| 2e-5,
            );
            let s = mklight_analysis::fans::FanSettings {
                samples: 11,
                ..Default::default()
            };
            let a = mklight_analysis::fans::opd_fan(&t, &s).unwrap();
            let g = gpu::opd_fan_job(b, &t, &s, &JobContext::new()).unwrap();
            close(&format!("{name} OPD"), &fans(&a), &fans(&g), |_| 5e-3);
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
    for (i, row) in result.values.as_chunks::<3>().0.iter().enumerate() {
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
    for row in r.values.as_chunks::<6>().0 {
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
    for (i, row) in r.values.as_chunks::<5>().0.iter().enumerate() {
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
        close(
            &format!("{name} lateral colour"),
            &a.lateral_colour,
            &g.lateral_colour,
            |_| 2e-5,
        );
        assert_eq!(a.curves.len(), g.curves.len());
        for (a, g) in a.curves.iter().zip(&g.curves) {
            close(
                &format!("{name} lateral colour"),
                &a.offset,
                &g.offset,
                |_| 2e-5,
            );
        }
        let s = mklight_analysis::seidel::LongitudinalSettings {
            pupil_samples: 9,
            ..Default::default()
        };
        let a = mklight_analysis::seidel::longitudinal_aberration(&t, &s).unwrap();
        let g = gpu::longitudinal_aberration_job(b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.curves.len(), g.curves.len());
        for (a, g) in a.curves.iter().zip(&g.curves) {
            close(&format!("{name} longitudinal"), &a.focus, &g.focus, |v| {
                1e-6 * (1.0 + v.abs())
            });
        }
        let s = mklight_analysis::illumination::FootprintSettings {
            surface: 1,
            rings: 3,
            ..Default::default()
        };
        let a = mklight_analysis::illumination::footprint(&t, &s).unwrap();
        let g = gpu::footprint_job(b, &t, &s, &JobContext::new()).unwrap();
        assert_eq!(a.rays.len(), g.rays.len());
        let flat = |v: &[[f64; 2]]| v.iter().flatten().copied().collect::<Vec<_>>();
        for (a, g) in a.rays.iter().zip(&g.rays) {
            assert_eq!(a.lost_before, g.lost_before);
            close(
                &format!("{name} footprint"),
                &flat(&a.points),
                &flat(&g.points),
                |_| 2e-5,
            );
            close(
                &format!("{name} footprint clipped"),
                &flat(&a.clipped),
                &flat(&g.clipped),
                |_| 2e-5,
            );
        }
        close(
            &format!("{name} footprint extent"),
            &a.extent,
            &g.extent,
            |_| 2e-5,
        );
        close(
            &format!("{name} footprint semi-diameter"),
            &[a.semi_diameter, a.max_radius],
            &[g.semi_diameter, g.max_radius],
            |_| 2e-5,
        );
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
            close(
                "P4 Coddington",
                &[reference.tangential, reference.sagittal],
                &[v[i].to_f64(), v[i + 1].to_f64()],
                |v| 1e-6 * (1.0 + v.abs()),
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
    close(
        "rim RI",
        &a.relative_illumination,
        &g.relative_illumination,
        |_| 1e-5,
    );
    let polish = &g.run_info.settings["resident_execution"]["ri_polish"];
    assert_eq!(polish["remaining_failed_nodes"], 0, "{polish}");
}
/// The resident EE curve itself (no reference replacement) on every sample,
/// with the bin-classification bound taken from the measured position error.
fn qualified_energy(b: &dyn ComputeBackend) {
    let mut replaced = Vec::new();
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let settings = EncircledEnergySettings {
            sampling: energy::RaySampling {
                rings: 16,
                wavelengths: mklight_analysis::util::WavelengthSelection::Primary,
                ..Default::default()
            },
            ..Default::default()
        };
        let grid = gpu::ray_grid_job(b, &t, &settings.sampling, &JobContext::new()).unwrap();
        // Field indices include the analysis-local symmetry fields.
        let augmented =
            mklight_analysis::util::default_fields(&t, &settings.sampling.fields).unwrap();
        let fields: Vec<_> = grid
            .fields
            .iter()
            .map(|&f| augmented.field(f).clone())
            .collect();
        let reference = mklight_trace::trace_grid_job(
            &augmented,
            &mklight_trace::Grid {
                fields: &fields,
                wavelengths: &grid.wavelengths,
                pupil: &grid.pupil,
            },
            &t.options(),
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(reference.len(), grid.len());
        let mut error = 0.0f64;
        for (a, g) in reference.iter().zip((0..grid.len()).map(|i| grid.hit(i))) {
            assert_eq!(a.status.is_ok(), g.is_some(), "{name}");
            if let Some(g) = g {
                error = error
                    .max((a.point.x - g.point[0]).abs())
                    .max((a.point.y - g.point[1]).abs());
            }
        }
        assert!(error < 2e-5, "{name} grid error {error}");
        let bound = (3.0 * error).max(1e-10);
        let result =
            gpu::encircled_energy_with_uncertainty_job(b, &t, &settings, bound, &JobContext::new())
                .unwrap();
        let reference = energy::encircled_energy(&t, &settings).unwrap();
        if result.run_info.backend != b.info().backend_id() {
            // A perfect focus puts every radius within the bound of a bin edge.
            let fallback = &result.run_info.settings["adapter_fallbacks"];
            assert_eq!(fallback["stage"], "energy_bin_classification", "{name}");
            assert_eq!(result.fields, reference.fields, "{name}");
            replaced.push(name);
            continue;
        }
        close(
            &format!("{name} qualified EE radii"),
            &reference.radius,
            &result.radius,
            |_| 2e-5,
        );
        assert_eq!(reference.fields.len(), result.fields.len());
        for (a, g) in reference.fields.iter().zip(&result.fields) {
            assert_eq!(a.rays_arrived, g.rays_arrived);
            close(
                &format!("{name} qualified EE"),
                &a.encircled,
                &g.encircled,
                |_| 2e-3,
            );
            close(
                &format!("{name} qualified ensquared"),
                &a.ensquared,
                &g.ensquared,
                |_| 2e-3,
            );
            close(
                &format!("{name} qualified EE radii 50/80/90"),
                &[a.r50, a.r80, a.r90, a.half_width80],
                &[g.r50, g.r80, g.r90, g.half_width80],
                |_| 2e-5,
            );
        }
    }
    eprintln!("P5_EE_REPLACED {replaced:?}");
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
            close(
                &format!("{name} {}", kind.name()),
                &a.values,
                &g.values,
                |_| 2e-4,
            );
        }
        let s = mklight_analysis::fans::FanSettings {
            samples: 9,
            ..Default::default()
        };
        let a = mklight_analysis::fans::opd_fan(&t, &s).unwrap();
        let g = gpu::opd_fan_job(b, &t, &s, &JobContext::new()).unwrap();
        close(
            &format!("{name} telecentric OPD"),
            &fans(&a),
            &fans(&g),
            |_| 5e-3,
        );
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
    assert_eq!(a.sets.len(), g.sets.len());
    for (a, g) in a.sets.iter().zip(&g.sets) {
        assert_eq!(a.points, g.points);
    }
    let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
    close(
        "tiled footprint",
        &flat(&a.points),
        &flat(&g.points),
        |_| 2e-5,
    );
    let execution = &g.run_info.settings["resident_execution"];
    assert!(!execution["extra_plans"].as_array().unwrap().is_empty());
    assert!(execution["plan_stats"]["peak_bytes"].as_u64().unwrap() < 256 * 1024 * 1024);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tilted_mirror_focal_lines() {
    use mklight_core::{Interaction, MaterialRef, Placement, SurfaceShape};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let radius = 200.0;
    for (tx, ty) in [(0.0, 10.0), (10.0, 0.0)] {
        let cb = |id, t| {
            let mut s = common::surf(id, SurfaceShape::CoordinateBreak, t, MaterialRef::Air);
            s.placement = Some(Placement {
                tilt_x: tx,
                tilt_y: ty,
                ..Default::default()
            });
            s
        };
        let mut mirror = common::surf(2, SurfaceShape::sphere(-radius), 0.0, MaterialRef::Air);
        mirror.interaction = Interaction::Reflect;
        let t = tracer_for(common::infinite_system(
            20.0,
            vec![cb(1, 0.0), mirror, cb(3, -0.5 * radius)],
            2,
            &[0.0, 3.0],
        ))
        .unwrap();
        let field = t
            .prepare_field(mklight_trace::FieldPoint::normalized(0.0, 0.0))
            .unwrap();
        let mut records = Vec::new();
        let chief = t.trace_with(
            &field,
            0,
            0.0,
            0.0,
            &mklight_trace::TraceOptions::unclipped(),
            Some(&mut records),
        );
        assert!(chief.status.is_ok());
        let global = |i: usize| {
            t.system().surfaces[i]
                .frame
                .to_global_point(records[i].point)
        };
        let distance = (global(4) - global(2)).norm();
        let cos_i = records[2].cos_incidence;
        let inplane = 0.5 * radius * cos_i - distance;
        let perpendicular = 0.5 * radius / cos_i - distance;
        let (et, es) = if ty != 0.0 {
            (perpendicular, inplane)
        } else {
            (inplane, perpendicular)
        };
        let g = gpu::field_curvature_job(
            &b,
            &t,
            &mklight_analysis::curvature::FieldCurvatureSettings {
                field_samples: 3,
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        let c = &g.curves[0];
        let i = g
            .field_coordinates
            .iter()
            .position(|p| *p == [0.0, 0.0])
            .unwrap();
        for (value, expected) in [(c.tangential[i], et), (c.sagittal[i], es)] {
            assert!(
                (value - expected * chief.ray.dir.z).abs() < 1e-6,
                "tilts {tx}/{ty}: {value} versus {}",
                expected * chief.ray.dir.z
            );
        }
        assert!((c.tangential[i] - c.tangential_close_rays[i]).abs() < 1e-3);
        assert!((c.sagittal[i] - c.sagittal_close_rays[i]).abs() < 1e-3);
    }
}
#[cfg(target_os = "macos")]
#[test]
fn metal_long_track_geometry_and_opd() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let t = tracer_for(common::ideal_lens(1200.0, 24.0, &[0.0, 2.0, 5.0])).unwrap();
    let s = SpotDiagramSettings {
        rings: 8,
        ..Default::default()
    };
    let a = spot::spot_diagram(&t, &s).unwrap();
    let g = gpu::spot_diagram_job(&b, &t, &s, &JobContext::new()).unwrap();
    assert_eq!(a.fields.len(), g.fields.len());
    for (a, g) in a.fields.iter().zip(&g.fields) {
        assert_eq!(a.polychromatic.rays_arrived, g.polychromatic.rays_arrived);
        let xy = |f: &spot::FieldSpot| f.points.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>();
        close("long-track spot", &xy(a), &xy(g), |_| 2e-5);
        close("long-track chief", &a.chief, &g.chief, |_| 2e-5);
    }
    let s = mklight_analysis::fans::FanSettings {
        samples: 17,
        ..Default::default()
    };
    let a = mklight_analysis::fans::opd_fan(&t, &s).unwrap();
    let g = gpu::opd_fan_job(&b, &t, &s, &JobContext::new()).unwrap();
    close("long-track OPD", &fans(&a), &fans(&g), |_| 5e-3);
}
