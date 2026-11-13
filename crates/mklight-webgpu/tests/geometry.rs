//! The geometric analyses end to end on the device (`mklight-analysis`'s
//! resident spot diagram, ray grid and geometric MTF on `gpu:webgpu`)
//! against the CPU-f32 specification in the device's environment and, on
//! macOS, against Metal: spot diagrams and ray grids agree bit for bit
//! (aiming, trace, image coordinates, focus terms and segmented moments all
//! run on the device), MTFs within the characteristic-function tolerance
//! (cos/sin are the only non-bitwise step).

use std::sync::atomic::AtomicBool;

use mklight_analysis::energy::{GeometricMtfResult, GeometricMtfSettings, RaySampling};
use mklight_analysis::gpu::{self, AimDesign, Route};
use mklight_analysis::spot::{self, SpotDiagram, SpotDiagramSettings};
use mklight_analysis::tracer_for;
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};

/// Every kernel module compiles and every pipeline is created.
#[test]
fn all_pipelines_compile() {
    let g = crate::gpu_or_skip!();
    let p = g
        .prepare_kernels(None, &|_, _| {}, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(p.kernels, mklight_webgpu::ops::pipelines().len());
}

fn same_bits(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "{what}[{i}]: {x:e} != {y:e}"
        );
    }
}

fn spot_values(s: &SpotDiagram) -> Vec<f64> {
    let mut v = vec![];
    for f in &s.fields {
        v.extend([f.rms_radius, f.polychromatic.rays_arrived as f64]);
        v.extend(f.points.iter().flat_map(|p| [p.x, p.y]));
        for w in &f.per_wavelength {
            v.extend([w.stats.rms_radius_centroid, w.stats.rms_radius_chief]);
        }
        for t in &f.through_focus {
            v.extend([t.defocus, t.rms_radius_centroid, t.rms_radius_chief]);
        }
    }
    v
}

fn mtf_close(what: &str, a: &GeometricMtfResult, b: &GeometricMtfResult) -> f64 {
    assert_eq!(a.curves.len(), b.curves.len(), "{what}");
    let mut worst = 0f64;
    for (a, b) in a.curves.iter().zip(&b.curves) {
        assert_eq!((a.field, a.orientation), (b.field, b.orientation), "{what}");
        for (x, y) in a.modulation.iter().zip(&b.modulation) {
            let e = (x - y).abs();
            assert!(
                e <= 1e-6 || (x.is_nan() && y.is_nan()),
                "{what} MTF {x} {y}"
            );
            if e.is_finite() {
                worst = worst.max(e);
            }
        }
    }
    worst
}

/// Spot diagram (with through focus), ray grid and geometric MTF of every
/// sample system, aiming off and real, on `b` against `reference`.
fn compare(b: &dyn ComputeBackend, reference: &dyn ComputeBackend, routes: &[Route]) {
    let mut worst = 0f64;
    for (name, mut model) in samples::all() {
        for aim in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            model.ray_aiming = aim;
            let t = tracer_for(model.clone()).unwrap();
            for route in routes {
                let what = format!("{name} {aim:?} {:?}", route.aiming);
                let s = SpotDiagramSettings {
                    rings: 6,
                    reference: spot::SpotReference::Centroid,
                    defocus: 0.02,
                    through_focus: Some(spot::ThroughFocus::symmetric(0.1, 3)),
                    ..Default::default()
                };
                let ctx = JobContext::new();
                let a = gpu::spot_diagram_job(b, &t, &s, route, &ctx).unwrap();
                assert_eq!(ctx.progress.fraction(), 1.0);
                let c =
                    gpu::spot_diagram_job(reference, &t, &s, route, &JobContext::new()).unwrap();
                same_bits(&format!("{what} spot"), &spot_values(&a), &spot_values(&c));
                let sampling = RaySampling {
                    rings: 5,
                    ..Default::default()
                };
                let a = gpu::ray_grid_job(b, &t, &sampling, route, &JobContext::new()).unwrap();
                let c =
                    gpu::ray_grid_job(reference, &t, &sampling, route, &JobContext::new()).unwrap();
                same_bits(&format!("{what} grid points"), &a.points, &c.points);
                same_bits(
                    &format!("{what} grid directions"),
                    &a.directions,
                    &c.directions,
                );
                let m = GeometricMtfSettings {
                    sampling: RaySampling {
                        rings: 4,
                        ..Default::default()
                    },
                    frequency_samples: 5,
                    max_frequency: Some(100.0),
                    defocus: 0.13,
                };
                let a = gpu::geometric_mtf_job(b, &t, &m, route, &JobContext::new()).unwrap();
                let c =
                    gpu::geometric_mtf_job(reference, &t, &m, route, &JobContext::new()).unwrap();
                worst = worst.max(mtf_close(&what, &a, &c));
            }
        }
    }
    eprintln!("largest MTF difference {worst:e}");
}

fn routes() -> Vec<Route> {
    vec![
        Route::default(),
        Route {
            aiming: AimDesign::Table,
            ..Default::default()
        },
    ]
}

#[test]
fn analyses_match_cpu_f32() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    compare(g, &c, &routes());
}

#[cfg(target_os = "macos")]
#[test]
fn analyses_match_metal() {
    let g = crate::gpu_or_skip!();
    let Ok(Some(m)) = mklight_metal::MetalBackend::new() else {
        eprintln!("SKIP: no Metal device");
        return;
    };
    compare(g, &m, &[Route::default()]);
}
