//! The work-split designs of the field maps, illumination and radiometry
//! (study key `fieldmaps`) against the f64 reference, on the CPU-f32
//! emulation of the device kernels and (ignored, run through the bench
//! lock) on Metal, where the restructured illumination kernels must also
//! reproduce the emulation bit for bit.
use mklight_analysis::{curvature, field_maps, gpu, illumination, tracer_for};
use mklight_core::{JobContext, samples, split};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use std::sync::Mutex;

/// The variant selection is process-wide: tests that set it run one at a
/// time.
static SELECTION: Mutex<()> = Mutex::new(());

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

const RI_DESIGNS: [&str; 6] = ["legacy", "nodes", "split", "hybrid", "pipelined", "shared"];

/// Every RI design on every sample system against the reference, within
/// the resident RI contract (`p5_geometry::ri`).
fn ri_designs(b: &dyn ComputeBackend, radial: usize, azimuth: usize, designs: &[&str]) {
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let s = illumination::RelativeIlluminationSettings {
            field_samples: 3,
            radial_samples: radial,
            azimuth_samples: azimuth,
            ..Default::default()
        };
        let a = illumination::relative_illumination(&t, &s).unwrap();
        for design in designs {
            split::set(gpu::SPLIT_KEY, Some(design));
            let g = gpu::relative_illumination_job(b, &t, &s, &JobContext::new());
            split::reset();
            let g = g.unwrap_or_else(|e| panic!("{name} {design}: {e}"));
            let w = format!("{name} {design}");
            close(
                &format!("{w} RI"),
                &a.relative_illumination,
                &g.relative_illumination,
                |_| 1e-5,
            );
            close(
                &format!("{w} solid angle"),
                &a.projected_solid_angle,
                &g.projected_solid_angle,
                |a| 1e-5 * a.abs(),
            );
            close(
                &format!("{w} vignetting"),
                &a.vignetting_fraction,
                &g.vignetting_fraction,
                |_| 1e-5,
            );
            close(
                &format!("{w} CRA"),
                &a.chief_ray_angle_deg,
                &g.chief_ray_angle_deg,
                |_| 2e-4,
            );
            close(&format!("{w} cos4"), &a.cos4_object, &g.cos4_object, |_| {
                1e-6
            });
            let evidence = g.run_info.settings.to_string();
            assert!(
                evidence.contains(&format!("\"fieldmaps_variant\":\"{design}\"")),
                "{w}: evidence names the design"
            );
        }
    }
}

#[test]
fn cpu_ri_designs() {
    ri_designs(
        &CpuF32Backend::new(),
        3,
        12,
        &["nodes", "split", "hybrid", "pipelined", "shared"],
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri_designs() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    ri_designs(&b, 12, 48, &RI_DESIGNS);
}

/// The restructured kernels on Metal reproduce their CPU-f32 emulation bit
/// for bit, so every design's rows do too.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri_kernels_match_emulation() {
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        let coords = [[0.0, 0.0], [0.0, 0.7], [0.5, 0.5]];
        let wl = t.system().primary_wavelength;
        for design in ["nodes", "split", "hybrid"] {
            split::set(gpu::SPLIT_KEY, Some(design));
            let m = gpu::illumination_rows(&b, &t, &coords, wl, 4, 16, &JobContext::new());
            let c = gpu::illumination_rows(&cpu, &t, &coords, wl, 4, 16, &JobContext::new());
            split::reset();
            let (m, c) = (m.unwrap(), c.unwrap());
            assert_eq!(m.0, c.0, "{name} {design} rows");
            assert_eq!(m.1, c.1, "{name} {design} on-axis");
        }
    }
}

/// The relative-illumination map of every design matches the reference map.
fn ri_map_designs(b: &dyn ComputeBackend, designs: &[&str]) {
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let t = tracer_for(samples::double_gauss()).unwrap();
    let s = field_maps::FieldMapSettings {
        grid: [3, 3],
        radial_samples: 3,
        azimuth_samples: 12,
        ..Default::default()
    };
    let kind = field_maps::FieldAnalysis::RelativeIllumination;
    let a = field_maps::field_map_job(&t, kind, &s, &JobContext::new()).unwrap();
    for design in designs {
        split::set(gpu::SPLIT_KEY, Some(design));
        let g = gpu::field_map_job(b, &t, kind, &s, &JobContext::new());
        split::reset();
        let g = g.unwrap();
        close(&format!("map {design}"), &a.values, &g.values, |a| {
            1e-5 * (1.0 + a.abs())
        });
        let ai: Vec<f64> = a.image_mm.iter().flatten().copied().collect();
        let gi: Vec<f64> = g.image_mm.iter().flatten().copied().collect();
        close(&format!("map {design} images"), &ai, &gi, |_| 1e-9);
    }
}

#[test]
fn cpu_ri_map_designs() {
    ri_map_designs(&CpuF32Backend::new(), &["nodes", "split", "shared"]);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_ri_map_designs() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    ri_map_designs(&b, &RI_DESIGNS);
}

/// The chief-ray maps (CRA, pupil aberration) and grid distortion by every
/// chief design.
fn chief_designs(b: &dyn ComputeBackend) {
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
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
        for design in ["host-chief", "device-chief", "cpu", "sized"] {
            split::set(gpu::SPLIT_KEY, Some(design));
            for (k, a) in kinds.iter().zip(&refs) {
                let g = gpu::field_map_job(b, &t, *k, &s, &JobContext::new()).unwrap();
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
            let g = gpu::grid_distortion_job(b, &t, &grid, &JobContext::new()).unwrap();
            let real = |r: &curvature::GridDistortionResult| -> Vec<f64> {
                r.points.iter().flat_map(|p| p.real).collect()
            };
            close(
                &format!("{name} {design} grid"),
                &real(&grid_ref),
                &real(&g),
                |_| 2e-5,
            );
            split::reset();
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

/// System transmission by every design, unpolarised and polarised.
fn transmission_designs(b: &dyn ComputeBackend) {
    use mklight_trace::{PolarizationState, PupilSamples, RadiometryModel, TransmissionSettings};
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
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
            for design in ["legacy", "batched", "shared", "cpu", "cpu-aim", "pipelined"] {
                split::set(gpu::SPLIT_KEY, Some(design));
                let g = mklight_gpu::radiometry::system_transmission(b, &t, &s, &JobContext::new());
                split::reset();
                let g = g.unwrap();
                assert_eq!(a.points.len(), g.points.len());
                for (p, q) in a.points.iter().zip(&g.points) {
                    let w = format!("{name} {design} {model:?} {p:?} {q:?}");
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
}

#[test]
fn cpu_transmission_designs() {
    transmission_designs(&CpuF32Backend::new());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal: run through the bench lock"]
fn metal_transmission_designs() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    transmission_designs(&b);
}

/// Results do not depend on the CPU threads: the reference rows in pools of
/// one and four threads, and the hybrids' CPU share
/// (`mklight_gpu::cpu_share`) with one thread and with the global pool.
#[test]
fn cpu_thread_budgets_do_not_change_results() {
    use mklight_trace::{PupilSamples, TransmissionSettings};
    let _guard = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let b = CpuF32Backend::new();
    let in_pool = |threads: usize, f: &(dyn Fn() -> String + Sync)| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    };
    let with_budget = |budget: Option<&str>, f: &dyn Fn() -> String| {
        split::set(mklight_gpu::cpu_share::KEY, budget);
        let r = f();
        split::reset();
        r
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
        for design in ["hybrid", "pipelined", "shared"] {
            let rows = || {
                split::set(gpu::SPLIT_KEY, Some(design));
                let r = gpu::illumination_rows(&b, &t, &coords, wl, 4, 16, &JobContext::new());
                let (rows, omega0, images, _) = r.unwrap();
                format!("{rows:?} {omega0:?} {images:?}")
            };
            if design != "shared" {
                assert_eq!(
                    with_budget(Some("1"), &rows),
                    with_budget(None, &rows),
                    "{name} {design}"
                );
            } else {
                // `shared` divides the fields by timing: only its success.
                with_budget(Some("1"), &rows);
            }
        }
        let s = TransmissionSettings {
            pupil: PupilSamples::Grid { n: 9 },
            ..Default::default()
        };
        for design in ["cpu-aim", "pipelined"] {
            let st = || {
                split::set(gpu::SPLIT_KEY, Some(design));
                let r =
                    mklight_gpu::radiometry::system_transmission(&b, &t, &s, &JobContext::new());
                format!("{:?}", r.unwrap().points)
            };
            assert_eq!(
                with_budget(Some("1"), &st),
                with_budget(None, &st),
                "{name} {design}"
            );
        }
    }
}
