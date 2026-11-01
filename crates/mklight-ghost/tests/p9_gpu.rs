mod common;
use common::*;
use mklight_core::{JobContext, samples};
use mklight_ghost::{gpu::GhostTransport, *};
use mklight_gpu::{ComputeBackend, CpuF32Backend, Df32, consts::*, ghost_ops::Probe};

fn compare(b: &dyn ComputeBackend, m: &GhostModel, paths: &[GhostPath]) {
    let g = GhostTransport::new(b, m, paths, &JobContext::new()).unwrap();
    let mut probes = Vec::new();
    let mut refs = Vec::new();
    for (pi, p) in paths.iter().enumerate() {
        let steps = m.check_path(p).unwrap();
        for field in m.tracer().fields() {
            for w in 0..m.system().wavelengths_um.len() {
                for pupil in [
                    [0., 0.],
                    [0.13, 0.17],
                    [-0.41, 0.33],
                    [0.7, -0.2],
                    [-0.9, 0.],
                ] {
                    let geometry = m.tracer().pupil();
                    let scale = if geometry.telecentric {
                        geometry.u_marginal.abs()
                    } else {
                        geometry.r_ep
                    };
                    let c = field.aims[w].chief;
                    let a = [c[0] + pupil[0] * scale, c[1] + pupil[1] * scale];
                    if let Some(ray) = m.tracer().start_ray(&field.source, a) {
                        probes.push(Probe {
                            position: [ray.pos.x, ray.pos.y, ray.pos.z].map(Df32::from_f64),
                            direction: [ray.dir.x, ray.dir.y, ray.dir.z].map(Df32::from_f64),
                            path: pi as u32,
                            wavelength: w as u32,
                        });
                        refs.push(m.trace_steps(&ray, w, &steps, None));
                    }
                }
            }
        }
    }
    assert!(!probes.is_empty());
    let result = g.trace(&probes, &JobContext::new()).unwrap();
    let again = g.trace(&probes, &JobContext::new()).unwrap();
    if b.info().kind == mklight_gpu::BackendKind::Metal {
        let emu = CpuF32Backend::matching(&b.precision().probe);
        let eg = GhostTransport::new(&emu, m, paths, &JobContext::new()).unwrap();
        let e = eg.trace(&probes, &JobContext::new()).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&result.values),
            bytemuck::cast_slice::<_, u32>(&e.values),
            "matching f32 emulation"
        );
    }
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&result.values),
        bytemuck::cast_slice::<_, u32>(&again.values)
    );
    let (mut maxp, mut maxd, mut maxw) = (0.0f64, 0.0f64, 0.0f64);
    let mut reference_probes = 0u64;
    for (i, (v, r)) in result.values.iter().zip(refs).enumerate() {
        let code = v.status & MKL_RAY_CODE_MASK;
        if code == MKL_GHOST_REFERENCE {
            reference_probes += 1;
            continue;
        }
        let expected = match r.status {
            SequenceStatus::Ok => 0,
            SequenceStatus::Miss { .. } => MKL_RAY_MISS,
            SequenceStatus::Clipped { .. } => MKL_RAY_VIGNETTED,
            SequenceStatus::TotalInternalReflection { .. } => MKL_RAY_TIR,
            SequenceStatus::Virtual { .. } => MKL_GHOST_VIRTUAL,
            SequenceStatus::Failed {
                surface, reason, ..
            } => match reason {
                SequenceFail::NoConvergence => MKL_RAY_NOT_CONVERGED,
                SequenceFail::OutsideDomain => MKL_RAY_OUTSIDE_DOMAIN,
                SequenceFail::NonFinite => MKL_RAY_NONFINITE,
                SequenceFail::Degenerate
                    if matches!(
                        m.system().surfaces[surface as usize].shape,
                        mklight_core::SurfaceShape::Diffractive { .. }
                    ) =>
                {
                    MKL_RAY_EVANESCENT
                }
                SequenceFail::Degenerate => MKL_RAY_DEGENERATE,
                SequenceFail::InvalidStep => panic!("validated path produced an invalid step"),
            },
        };
        assert_eq!(code, expected, "probe {i} {:?}", r.status);
        if code != 0 {
            continue;
        }
        for (k, x) in [r.ray.pos.x, r.ray.pos.y, r.ray.pos.z].iter().enumerate() {
            let e = (v.position[k].to_f64() - x).abs();
            maxp = maxp.max(e);
            assert!(e <= 2e-5, "position {i}: {e}");
        }
        for (k, x) in [r.ray.dir.x, r.ray.dir.y, r.ray.dir.z].iter().enumerate() {
            let e = (v.direction[k].to_f64() - x).abs();
            maxd = maxd.max(e);
            assert!(e <= 2e-6, "direction {i}: {e}");
        }
        let w = 0.5 * (v.weight_s.to_f64() + v.weight_p.to_f64()) * v.internal.to_f64();
        let e = (w - r.weight()).abs();
        maxw = maxw.max(e / r.weight().max(1e-10));
        assert!(
            e <= 1e-5 * r.weight() + 1e-10,
            "weight {i}: {w} != {}",
            r.weight()
        );
    }
    assert!(result.stats.fallback_count >= reference_probes);
    eprintln!(
        "{} reference={reference_probes} probes={} max_position={maxp:e} max_direction={maxd:e} max_weight_relative={maxw:e}",
        m.system().system().name,
        probes.len()
    );
}
fn samples_check(b: &dyn ComputeBackend) {
    for (_, s) in samples::all() {
        let m = model(
            s,
            GhostSettings {
                sensor_reflectance: Some(0.3),
                ..Default::default()
            },
        );
        let e = enumerate(&m, &EnumerationSettings::default()).unwrap();
        let paths = std::iter::once(GhostPath::primary())
            .chain(e.paths.iter().map(|p| p.path.clone()))
            .collect::<Vec<_>>();
        compare(b, &m, &paths);
    }
}
#[test]
fn cpu_f32_transport_against_f64() {
    samples_check(&CpuF32Backend::new());
}
#[cfg(target_os = "macos")]
#[test]
fn metal_transport_both_libraries() {
    use mklight_metal::*;
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P9 requires host Metal");
        assert!(b.precision().probe.error_free_transforms());
        eprintln!("device={:?} precision={:?}", b.info(), b.precision());
        samples_check(&b);
    }
}
#[cfg(target_os = "macos")]
#[test]
fn maps_and_progress() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    for (_, s) in samples::all() {
        let m = model(s, GhostSettings::default());
        let settings = IrradianceSettings {
            pupil_samples: 12,
            map_bins: [16, 16],
            fields: vec![0],
            wavelengths: vec![0],
            keep_path_maps: true,
            ..Default::default()
        };
        let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
        let a = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
        let flux = c.total_ghost_map.flux();
        let l1 = a
            .total_ghost_map
            .values
            .iter()
            .zip(&c.total_ghost_map.values)
            .map(|(a, c)| (a - c).abs())
            .sum::<f64>()
            * c.total_ghost_map.bin_area();
        eprintln!(
            "{} flux={flux:e} map_l1_relative={} primary_error={}",
            m.system().system().name,
            l1 / flux.max(1e-20),
            (a.primary.flux - c.primary.flux).abs() / c.primary.flux
        );
        assert!(l1 <= 1e-3 * flux + 1e-12);
        let pm = c.primary.map.as_ref().unwrap();
        let gm = a.primary.map.as_ref().unwrap();
        let pl1 = pm
            .values
            .iter()
            .zip(&gm.values)
            .map(|(c, g)| (c - g).abs())
            .sum::<f64>()
            * pm.bin_area();
        assert!(
            pl1 <= 1e-3 * pm.flux(),
            "{} primary L1 {}",
            m.system().system().name,
            pl1 / pm.flux()
        );
        for (g, c) in a.ghosts.iter().zip(&c.ghosts) {
            assert_eq!(g.path, c.path);
            for (g, c) in g.fields.iter().zip(&c.fields) {
                assert_eq!(g.rays_traced, c.rays_traced);
                assert_eq!(g.rays_detected, c.rays_detected);
                assert!((g.flux - c.flux).abs() < 1e-3 * c.flux + 1e-12);
                assert!((g.rms_radius - c.rms_radius).abs() < 1e-3 * c.rms_radius + 2e-5);
            }
        }

        assert!((a.primary.flux - c.primary.flux).abs() <= 1e-3 * c.primary.flux);
        assert_eq!(
            serde_json::to_value(&a.coverage).unwrap(),
            serde_json::to_value(&c.coverage).unwrap()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn physical_fixtures_and_focus() {
    use mklight_core::{ApertureShape, CoatingDef, MaterialRef, Placement, Sensor, SurfaceShape};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let paths = [
        GhostPath::primary(),
        GhostPath::new(vec![3, 2]),
        GhostPath::new(vec![3, 2, 3, 2]),
    ];
    for reflectance in [0., 0.3, 1.] {
        let m = model(
            plate(
                1.5,
                10.,
                20.,
                &[0., 5.],
                Some(CoatingDef::Ideal {
                    reflectance,
                    transmittance: 1. - reflectance,
                }),
            ),
            unclipped(),
        );
        compare(&b, &m, &paths);
        let g = GhostTransport::new(&b, &m, &paths, &JobContext::new()).unwrap();
        let probes = (0..paths.len())
            .map(|i| Probe {
                position: [Df32::ZERO; 3],
                direction: [Df32::ZERO, Df32::ZERO, Df32::from_f32(1.)],
                path: i as u32,
                wavelength: 0,
            })
            .collect::<Vec<_>>();
        let r = g.trace(&probes, &JobContext::new()).unwrap();
        for (i, v) in r.values.iter().enumerate() {
            let expected = (1. - reflectance).powi(2) * reflectance.powi(2 * i as i32);
            let actual = 0.5 * (v.weight_s.to_f64() + v.weight_p.to_f64()) * v.internal.to_f64();
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    let mut tilted = plate(1.6, 5., 15., &[0., 5.], Some(quarter_wave(1.38, D_LINE)));
    tilted.surfaces[2].placement = Some(Placement {
        tilt_x: 4.,
        decenter_y: 0.5,
        ..Default::default()
    });
    tilted.surfaces[3].placement = Some(Placement {
        tilt_y: -3.,
        decenter_x: 0.2,
        ..Default::default()
    });
    compare(&b, &model(tilted, unclipped()), &paths);
    for aperture in [
        ApertureShape::Circular { radius: 4. },
        ApertureShape::Annular {
            inner_radius: 1.,
            outer_radius: 4.,
        },
        ApertureShape::Rectangular {
            half_x: 2.,
            half_y: 4.,
        },
        ApertureShape::Elliptical {
            semi_x: 2.,
            semi_y: 4.,
        },
    ] {
        let mut s = plate(1.5, 5., 20., &[0., 10.], None);
        s.surfaces[3].aperture = Some(aperture);
        compare(&b, &model(s, GhostSettings::default()), &paths);
    }
    let mut sensor = plate(1.5, 10., 20., &[0., 5.], None);
    sensor.sensor = Some(Sensor {
        pixels_x: 40,
        pixels_y: 20,
        pitch_x: 0.1,
        pitch_y: 0.1,
        max_cra_deg: None,
        spectral_response: vec![],
    });
    let m = model(
        sensor,
        GhostSettings {
            sensor_reflectance: Some(0.3),
            ..unclipped()
        },
    );
    compare(&b, &m, &[GhostPath::primary(), GhostPath::new(vec![4, 2])]);
    let newtonian = model(
        samples::newtonian_telescope(),
        GhostSettings {
            sensor_reflectance: Some(0.3),
            ..unclipped()
        },
    );
    compare(
        &b,
        &newtonian,
        &[
            GhostPath::primary(),
            GhostPath::new(vec![newtonian.system().image_index() as u32]),
        ],
    );
    let mut converging = plate(1.5, 10., 0., &[0.], None);
    converging.surfaces[1] = surf(
        1,
        SurfaceShape::Paraxial { focal_length: 100. },
        20.,
        MaterialRef::Air,
    );
    converging.surfaces[3].thickness = 100. - 20. - 10. / 1.5;
    let m = model(converging, unclipped());
    let r = gpu::real_ghost_focus(&b, &m, &paths, 0.001, &JobContext::new()).unwrap();
    assert!((r.focus[1] + 20. / 1.5).abs() < 1e-5);
    // Every wavelength and expanded order visit shares the reference path checks.
    let doe = model(
        samples::hybrid_diffractive_achromat(),
        GhostSettings::default(),
    );
    let p = enumerate(
        &doe,
        &EnumerationSettings {
            diffraction_orders: vec![-1, 0, 2],
            ..Default::default()
        },
    )
    .unwrap()
    .paths
    .into_iter()
    .map(|p| p.path)
    .collect::<Vec<_>>();
    compare(&b, &doe, &p);
    for (_, s) in samples::all() {
        let m = model(s, GhostSettings::default());
        let p = std::iter::once(GhostPath::primary())
            .chain(
                enumerate(&m, &EnumerationSettings::default())
                    .unwrap()
                    .paths
                    .into_iter()
                    .map(|p| p.path),
            )
            .collect::<Vec<_>>();
        if p.is_empty() {
            continue;
        }
        let w = (0..m.system().wavelengths_um.len()).collect::<Vec<_>>();
        let r = gpu::paraxial_batch(&b, &m, &p, &w, &JobContext::new()).unwrap();
        for (i, rays) in r.rays.iter().enumerate() {
            let wl = w[i % w.len()];
            let fo = m.tracer().paraxial().at(wl);
            let steps = m.check_path(&p[i / w.len()]).unwrap();
            let (rp, tp) = mklight_ghost::paraxial::axial_throughput(&m, &steps, wl);
            for (a, c) in r.throughput[i].into_iter().zip([rp, tp]) {
                assert!((a - c).abs() <= 1e-5 * c.abs() + 1e-10);
            }

            for (j, (y, nu)) in [
                (fo.marginal.y[1], fo.marginal.nu[0]),
                (fo.chief.y[1], fo.chief.nu[0]),
            ]
            .into_iter()
            .enumerate()
            {
                let reference = mklight_ghost::paraxial::trace_steps(&m, &steps, wl, y, nu);
                for (a, c) in rays[j].y.iter().zip(&reference.y) {
                    assert!((a - c).abs() < 1e-6 * c.abs() + 2e-5);
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn concurrent_repeatability_and_cancellation() {
    use std::time::{Duration, Instant};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let m = model(samples::double_gauss(), GhostSettings::default());
    let settings = IrradianceSettings {
        pupil_samples: 16,
        fields: vec![0],
        wavelengths: vec![1],
        map_bins: [16, 16],
        ..Default::default()
    };
    std::thread::scope(|scope| {
        let a =
            scope.spawn(|| gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap());
        b.set_chunk_items(64);
        let c = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
        let a = a.join().unwrap();
        assert_eq!(a.total_ghost_map, c.total_ghost_map);
        assert_eq!(a.primary.map, c.primary.map);
    });
    let ctx = JobContext::new();
    let dense = IrradianceSettings {
        pupil_samples: 128,
        ..settings
    };
    std::thread::scope(|scope| {
        let task = scope.spawn(|| gpu::ghost_irradiance(&b, &m, &dense, &ctx));
        let begin = Instant::now();
        let mut previous = 0.;
        let mut intermediate = false;
        while !task.is_finished() {
            let p = ctx.progress.fraction();
            assert!(p >= previous);
            previous = p;
            if p > 0.04 && p < 1. {
                intermediate = true;
                ctx.cancel.cancel();
                break;
            }
            assert!(begin.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(intermediate);
        let cancel = Instant::now();
        assert!(matches!(
            task.join().unwrap(),
            Err(mklight_core::MkError::Cancelled)
        ));
        assert!(ctx.progress.fraction() < 1.);
        eprintln!("cancellation_ms={}", cancel.elapsed().as_secs_f64() * 1000.);
    });
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        gpu::ghost_irradiance(&b, &m, &dense, &cancelled),
        Err(mklight_core::MkError::Cancelled)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn virtual_backward_lens_and_reported_reference() {
    use mklight_core::{CoatingDef, MaterialRef, SurfaceShape};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let virtual_plate = model(plate(1.5, -1., 20., &[0., 5.], None), unclipped());
    compare(
        &b,
        &virtual_plate,
        &[GhostPath::primary(), GhostPath::new(vec![3, 2])],
    );
    let mut lens = plate(1.5, 2., 20., &[0., 5.], None);
    lens.surfaces.insert(
        3,
        surf(
            99,
            SurfaceShape::Paraxial { focal_length: 75. },
            3.,
            MaterialRef::Constant { n: 1.6 },
        ),
    );
    compare(
        &b,
        &model(lens, unclipped()),
        &[GhostPath::primary(), GhostPath::new(vec![4, 2])],
    );
    let mut film = quarter_wave(1.38, D_LINE);
    if let CoatingDef::ThinFilm { layers, .. } = &mut film {
        layers[0].quarter_waves = None;
        layers[0].thickness_um = 1e7;
    }
    let m = model(plate(1.5, 2., 20., &[0.], Some(film)), unclipped());
    let settings = IrradianceSettings {
        pupil_samples: 8,
        fields: vec![0],
        wavelengths: vec![0],
        map_bins: [8, 8],
        map_half_size: Some([12.; 2]),
        ..Default::default()
    };
    let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
    let g = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
    assert!(g.run.settings["gpu"]["reference_probes"].as_u64().unwrap() > 0);
    for (a, c) in g
        .total_ghost_map
        .values
        .iter()
        .zip(&c.total_ghost_map.values)
    {
        assert!((a - c).abs() < 1e-10 * c.abs() + 1e-14);
    }
    let ranking = gpu::paraxial_ghosts(
        &b,
        &m,
        &ParaxialGhostSettings::default(),
        &JobContext::new(),
    )
    .unwrap();
    let reference = paraxial_ghosts(&m, &ParaxialGhostSettings::default()).unwrap();
    assert_eq!(ranking.ghosts.len(), reference.ghosts.len());
    for (a, c) in ranking.ghosts.iter().zip(reference.ghosts) {
        assert_eq!(a.path, c.path);
        assert!(
            (a.relative_irradiance - c.relative_irradiance).abs()
                < 1e-10 * c.relative_irradiance + 1e-14
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn absorption_long_track_and_telecentric_launches() {
    use mklight_core::{
        ApertureSpec, Field, FieldKind, FieldSet, MaterialRef, OpticalSystem, SurfaceShape,
        materials::{GlassCatalog, MaterialLibrary, TransmittancePoint},
    };
    use std::sync::Arc;
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let mut lib = MaterialLibrary::with_builtin();
    let mut glass = (*lib.find_glass(None, "N-BK7").unwrap()).clone();
    glass.name = "ABS".into();
    glass.transmittance = vec![
        TransmittancePoint {
            um: 0.3,
            tau: 0.8,
            thickness_mm: 10.,
        },
        TransmittancePoint {
            um: 2.,
            tau: 0.8,
            thickness_mm: 10.,
        },
    ];
    let mut catalog = GlassCatalog {
        name: "TEST".into(),
        ..Default::default()
    };
    catalog.glasses.insert("ABS".into(), glass);
    lib.add_catalog(Arc::new(catalog));
    let mut s = plate(1.5, 10., 20., &[0., 20.], Some(quarter_wave(1.38, D_LINE)));
    s.surfaces[2].material = MaterialRef::Catalog {
        catalog: Some("TEST".into()),
        name: "ABS".into(),
        offset: Default::default(),
    };
    let m = GhostModel::from_system(resolve_with(s, lib), unclipped()).unwrap();
    compare(&b, &m, &[GhostPath::primary(), GhostPath::new(vec![3, 2])]);
    compare(
        &b,
        &model(plate(1.5, 10., 900., &[0., 5.], None), unclipped()),
        &[GhostPath::primary(), GhostPath::new(vec![3, 2])],
    );
    let mut s = OpticalSystem::new_empty();
    s.wavelengths = mono();
    s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![Field::new(0., 0.), Field::new(0., 5.)],
    };
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, 100., MaterialRef::Air),
        surf(
            1,
            SurfaceShape::Paraxial { focal_length: 50. },
            50.,
            MaterialRef::Air,
        ),
        surf(2, SurfaceShape::Plane, 50., MaterialRef::Air),
        surf(3, SurfaceShape::Plane, 0., MaterialRef::Air),
    ];
    s.stop = 2;
    let m = model(s, unclipped());
    assert!(m.tracer().pupil().telecentric);
    compare(&b, &m, &[GhostPath::primary()]);
    let settings = IrradianceSettings {
        pupil_samples: 8,
        map_bins: [8, 8],
        ..Default::default()
    };
    let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
    let a = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
    let cm = c.primary.map.unwrap();
    let am = a.primary.map.unwrap();
    let l1 = cm
        .values
        .iter()
        .zip(am.values)
        .map(|(c, a)| (c - a).abs())
        .sum::<f64>()
        * cm.bin_area();
    assert!(l1 < 1e-3 * cm.flux());
}

#[cfg(target_os = "macos")]
#[test]
fn passive_and_sensor_surfaces_ignore_doe_phase_and_efficiency() {
    use mklight_core::{Interaction, SurfaceShape};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let mut dummy = samples::hybrid_diffractive_achromat();
    let index = dummy
        .surfaces
        .iter()
        .position(|s| matches!(s.shape, SurfaceShape::Diffractive { .. }))
        .unwrap();
    let image_shape = dummy.surfaces[index].shape.clone();
    dummy.surfaces[index].interaction = Interaction::Dummy;
    let m = model(dummy, unclipped());
    let override_path = GhostPath {
        reflections: vec![],
        diffraction: vec![mklight_ghost::path::DiffractionOrder {
            surface: index as u32,
            visit: 0,
            order: 0,
        }],
    };
    compare(&b, &m, &[GhostPath::primary(), override_path.clone()]);
    let settings = IrradianceSettings {
        paths: Some(vec![override_path]),
        pupil_samples: 8,
        fields: vec![0],
        wavelengths: vec![0],
        map_bins: [8, 8],
        map_half_size: Some([12.; 2]),
        ..Default::default()
    };
    let c = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
    let g = gpu::ghost_irradiance(&b, &m, &settings, &JobContext::new()).unwrap();
    assert!((g.ghosts[0].flux - c.ghosts[0].flux).abs() < 1e-10 * c.ghosts[0].flux + 1e-14);
    for shape in [image_shape, SurfaceShape::Paraxial { focal_length: 80. }] {
        let mut s = samples::hybrid_diffractive_achromat();
        let image = s.surfaces.len() - 1;
        s.surfaces[image].shape = shape;
        let m = model(
            s,
            GhostSettings {
                sensor_reflectance: Some(0.3),
                ..unclipped()
            },
        );
        compare(
            &b,
            &m,
            &[
                GhostPath::primary(),
                GhostPath::new(vec![image as u32, (image - 1) as u32]),
            ],
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn paraxial_sensor_powers_and_afocal_completion() {
    use mklight_core::{CoatingDef, MaterialRef, SurfaceShape};
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let mut s = plate(
        1.5,
        10.,
        0.,
        &[0.],
        Some(CoatingDef::Ideal {
            reflectance: 0.3,
            transmittance: 0.7,
        }),
    );
    s.surfaces[1] = surf(
        1,
        SurfaceShape::Paraxial { focal_length: 100. },
        20.,
        MaterialRef::Air,
    );
    s.surfaces[3].thickness = 80. - 10. / 1.5;
    let m = model(
        s,
        GhostSettings {
            sensor_reflectance: Some(0.3),
            ..unclipped()
        },
    );
    let result = gpu::paraxial_batch(
        &b,
        &m,
        &[GhostPath::primary(), GhostPath::new(vec![4, 2])],
        &[0],
        &JobContext::new(),
    )
    .unwrap();
    for (a, c) in result
        .throughput
        .iter()
        .flatten()
        .zip([1., 0.7f64.powi(3), 0.09, 0.7f64.powi(5)])
    {
        assert!((a - c).abs() < 1e-12);
    }
    let m = model(plate(1.5, 10., 20., &[0.], None), unclipped());
    let r = gpu::paraxial_batch(&b, &m, &[GhostPath::primary()], &[0], &JobContext::new()).unwrap();
    assert_eq!(r.run.settings["reference_rows"], serde_json::json!([0]));
    assert_eq!(r.rays[0][0].nu.last().copied(), Some(0.));
}
