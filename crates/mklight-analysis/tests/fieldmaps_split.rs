//! The device illumination rows and maps, the chief-ray map designs
//! (`gpu::Route::chief`) and the device system transmission against the f64
//! reference, on the CPU-f32 emulation of the device kernels and (ignored,
//! run through the bench lock) on Metal, where the illumination kernels must
//! also reproduce the emulation bit for bit.
use mklight_analysis::{curvature, field_maps, gpu, illumination, tracer_for};
use mklight_core::{JobContext, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};

fn close(what: &str, a: &[f64], g: &[f64], tol: impl Fn(f64) -> f64) {
    assert_eq!(a.len(), g.len(), "{what}: length");
    for (&a, &g) in a.iter().zip(g) {
        let ok = if a.is_finite() {
            (a - g).abs() <= tol(a)
        } else {
            a == g || a.is_nan() && g.is_nan()
        };
        assert!(ok, "{what}: reference {a}, design {g}");
    }
}

/// The device RI on every sample system against the reference, within the
/// resident RI contract (`p5_geometry::ri`).
fn ri(b: &dyn ComputeBackend, radial: usize, azimuth: usize) {
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = illumination::RelativeIlluminationSettings {
            field_samples: 3,
            radial_samples: radial,
            azimuth_samples: azimuth,
            ..Default::default()
        };
        let a = illumination::relative_illumination(&t, &s).unwrap();
        let g = gpu::relative_illumination_job(b, &t, &s, &JobContext::new())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        close(
            &format!("{name} RI"),
            &a.relative_illumination,
            &g.relative_illumination,
            |_| 1e-5,
        );
        close(
            &format!("{name} solid angle"),
            &a.projected_solid_angle,
            &g.projected_solid_angle,
            |a| 1e-5 * a.abs(),
        );
        close(
            &format!("{name} vignetting"),
            &a.vignetting_fraction,
            &g.vignetting_fraction,
            |_| 1e-5,
        );
        close(
            &format!("{name} CRA"),
            &a.chief_ray_angle_deg,
            &g.chief_ray_angle_deg,
            |_| 2e-4,
        );
        close(
            &format!("{name} cos4"),
            &a.cos4_object,
            &g.cos4_object,
            |_| 1e-6,
        );
    }
}

#[test]
fn cpu_ri() {
    ri(&CpuF32Backend::new(), 3, 12);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    ri(&b, 12, 48);
}

/// The illumination kernel on Metal reproduces its CPU-f32 emulation bit
/// for bit, so the rows do too.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri_kernels_match_emulation() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let coords = [[0.0, 0.0], [0.0, 0.7], [0.5, 0.5]];
        let wl = t.system().primary_wavelength;
        let m = gpu::illumination_rows(&b, &t, &coords, wl, 4, 16, &JobContext::new());
        let c = gpu::illumination_rows(&cpu, &t, &coords, wl, 4, 16, &JobContext::new());
        let (m, c) = (m.unwrap(), c.unwrap());
        assert_eq!(m.0, c.0, "{name} rows");
        assert_eq!(m.1, c.1, "{name} on-axis");
    }
}

/// The device relative-illumination map matches the reference map.
fn ri_map(b: &dyn ComputeBackend) {
    let t = tracer_for(samples::double_gauss()).unwrap();
    let s = field_maps::FieldMapSettings {
        grid: [3, 3],
        radial_samples: 3,
        azimuth_samples: 12,
        ..Default::default()
    };
    let kind = field_maps::FieldAnalysis::RelativeIllumination;
    let a = field_maps::field_map_job(&t, kind, &s, &JobContext::new()).unwrap();
    let g = gpu::field_map_job(b, &t, kind, &s, &Default::default(), &JobContext::new()).unwrap();
    close("map", &a.values, &g.values, |a| 1e-5 * (1.0 + a.abs()));
    let ai: Vec<f64> = a.image_mm.iter().flatten().copied().collect();
    let gi: Vec<f64> = g.image_mm.iter().flatten().copied().collect();
    close("map images", &ai, &gi, |_| 1e-9);
}

#[test]
fn cpu_ri_map() {
    ri_map(&CpuF32Backend::new());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri_map() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    ri_map(&b);
}

/// The chief-ray maps (CRA, pupil aberration) and grid distortion by every
/// chief design, each named in the run evidence.
fn chief_designs(b: &dyn ComputeBackend) {
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = field_maps::FieldMapSettings {
            grid: [9, 9],
            ..Default::default()
        };
        let grid = curvature::GridDistortionSettings {
            grid_size: 9,
            ..Default::default()
        };
        let kinds = [
            field_maps::FieldAnalysis::ChiefRayAngle,
            field_maps::FieldAnalysis::PupilAberration,
        ];
        let refs: Vec<_> = kinds
            .iter()
            .map(|&k| field_maps::field_map_job(&t, k, &s, &JobContext::new()).unwrap())
            .collect();
        let grid_ref = curvature::grid_distortion_job(&t, &grid, &JobContext::new()).unwrap();
        for chief in [gpu::ChiefDesign::HostChief, gpu::ChiefDesign::DeviceChief] {
            let design = chief.name();
            let route = gpu::Route {
                chief,
                ..Default::default()
            };
            // The design a run names: its own, also when the reference
            // completed a device map.
            let named = |run: &mklight_core::RunInfo| {
                let own = &run.settings["fieldmaps_variant"];
                let device = &run.settings["resident_execution"]["fieldmaps_variant"];
                own.as_str().or(device.as_str()).map(String::from)
            };
            for (k, a) in kinds.iter().zip(&refs) {
                let g = gpu::field_map_job(b, &t, *k, &s, &route, &JobContext::new()).unwrap();
                assert_eq!(named(&g.run_info).as_deref(), Some(design), "{name}");
                let tol = match k {
                    // Degrees.
                    field_maps::FieldAnalysis::ChiefRayAngle => 2e-4,
                    // Stop millimetres and magnifications.
                    _ => 1e-5,
                };
                close(
                    &format!("{name} {design} {}", k.name()),
                    &a.values,
                    &g.values,
                    |a| tol * (1.0 + a.abs()),
                );
            }
            let g = gpu::grid_distortion_job(b, &t, &grid, &route, &JobContext::new()).unwrap();
            assert_eq!(named(&g.run_info).as_deref(), Some(design), "{name}");
            let real = |r: &curvature::GridDistortionResult| -> Vec<f64> {
                r.points.iter().flat_map(|p| p.real).collect()
            };
            close(
                &format!("{name} {design} grid"),
                &real(&grid_ref),
                &real(&g),
                |_| 2e-5,
            );
        }
    }
}

#[test]
fn cpu_chief_designs() {
    chief_designs(&CpuF32Backend::new());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_chief_designs() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    chief_designs(&b);
}

/// System transmission, unpolarised and polarised.
fn transmission(b: &dyn ComputeBackend) {
    use mklight_trace::{PolarizationState, PupilSamples, RadiometryModel, TransmissionSettings};
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        for model in [
            RadiometryModel::Unpolarized,
            RadiometryModel::Polarized {
                state: PolarizationState::Unpolarized,
            },
        ] {
            let s = TransmissionSettings {
                pupil: PupilSamples::Grid { n: 9 },
                model,
                ..Default::default()
            };
            let a = mklight_trace::system_transmission(&t, &s, &JobContext::new()).unwrap();
            let g = mklight_gpu::radiometry::system_transmission(b, &t, &s, &JobContext::new())
                .unwrap();
            assert_eq!(a.points.len(), g.points.len());
            for (p, q) in a.points.iter().zip(&g.points) {
                let w = format!("{name} {model:?} {p:?} {q:?}");
                assert_eq!(p.samples, q.samples, "{w}");
                assert_eq!(p.failed, q.failed, "{w}");
                assert!((p.transmission - q.transmission).abs() < 1e-6, "{w}");
                assert!(
                    (p.unvignetted_fraction - q.unvignetted_fraction).abs() < 1e-9,
                    "{w}"
                );
            }
        }
    }
}

#[test]
fn cpu_transmission() {
    transmission(&CpuF32Backend::new());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_transmission() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    transmission(&b);
}

/// Results do not depend on the CPU threads: the reference rows, the
/// device RI rows and the transmission in pools of one and four threads.
#[test]
fn cpu_thread_counts_do_not_change_results() {
    use mklight_trace::{PupilSamples, TransmissionSettings};
    let b = CpuF32Backend::new();
    let in_pool = |threads: usize, f: &(dyn Fn() -> String + Sync)| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    };
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let wl = t.system().primary_wavelength;
        let coords = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.6, 0.6]];
        let reference = || {
            let mut warnings = Vec::new();
            let r = illumination::illumination_rows(
                &t,
                &coords,
                wl,
                4,
                16,
                &mut warnings,
                &JobContext::new(),
            )
            .unwrap();
            format!("{:?} {warnings:?}", r)
        };
        assert_eq!(in_pool(1, &reference), in_pool(4, &reference), "{name} RI");
        let rows = || {
            let r = gpu::illumination_rows(&b, &t, &coords, wl, 4, 16, &JobContext::new());
            let (rows, omega0, images, _) = r.unwrap();
            format!("{rows:?} {omega0:?} {images:?}")
        };
        assert_eq!(in_pool(1, &rows), in_pool(4, &rows), "{name} device RI");
        let s = TransmissionSettings {
            pupil: PupilSamples::Grid { n: 9 },
            ..Default::default()
        };
        let st = || {
            let r = mklight_gpu::radiometry::system_transmission(&b, &t, &s, &JobContext::new());
            format!("{:?}", r.unwrap().points)
        };
        assert_eq!(in_pool(1, &st), in_pool(4, &st), "{name} transmission");
    }
}
