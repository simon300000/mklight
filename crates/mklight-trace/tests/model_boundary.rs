//! Where this crate meets the model's validation and the macOS app's
//! fixtures: a shape validation accepts traces as the equivalent shape it
//! already accepted, and the committed solve-report fixture is what
//! `examples/dump_solve_fixtures` writes.

mod common;
use common::*;
use mklight_core::model::surface::DiffractivePhase;
use mklight_core::{MaterialRef, OpticalSystem, SemiDiameter, SurfaceShape, Vec3};
use mklight_trace::{Ray, TraceOptions, trace_ray_recorded};

// `main` writes the file; the test reads what it would write.
#[allow(dead_code)]
#[path = "../examples/dump_solve_fixtures.rs"]
mod generator;

fn overlay(base: SurfaceShape, astigmatism: f64, spherical: f64) -> SurfaceShape {
    SurfaceShape::SagOverlay {
        base: Box::new(base),
        norm_radius: 4.0,
        test_wavelength_um: 0.6328,
        astigmatism,
        astigmatism_angle_deg: 30.0,
        spherical,
    }
}

/// A plano-convex N-BK7 element whose curved, aspheric front carries a
/// diffractive phase on `base`.
fn doe_element(base: SurfaceShape) -> OpticalSystem {
    let shape = SurfaceShape::Diffractive {
        base: Box::new(base),
        phase: DiffractivePhase {
            norm_radius: 4.0,
            coefficients: vec![-12.0, 0.4],
            design_order: 1,
            design_wavelength_um: D_LINE,
            trace_order: 1,
        },
    };
    let mut front = surf(1, shape, 3.0, MaterialRef::glass("N-BK7"));
    front.semi_diameter = SemiDiameter::Fixed { value: 4.0 };
    let back = surf(2, SurfaceShape::Plane, 40.0, MaterialRef::Air);
    infinite_system(6.0, vec![front, back], 3)
}

/// Overlays stacked under a diffractive phase add like overlays stacked on
/// a bare surface: the stack traces, and has the first-order model, of the
/// one overlay with the same departure.
#[test]
fn stacked_overlays_under_a_diffractive_phase_trace_as_their_sum() {
    let asphere = || SurfaceShape::EvenAsphere {
        curvature: 0.03,
        conic: -0.6,
        coefficients: vec![2e-5, -1e-7],
    };
    // Astigmatism in one layer and spherical in the other, and both split
    // over two layers (halves and their sum are exact).
    let single = resolve(doe_element(overlay(asphere(), 3.0, -2.0)));
    let bare = resolve(doe_element(asphere()));
    let stacks = [
        overlay(overlay(asphere(), 3.0, 0.0), 0.0, -2.0),
        overlay(overlay(asphere(), 1.5, -1.0), 1.5, -1.0),
        overlay(overlay(overlay(asphere(), 3.0, -2.0), 0.0, 0.0), 0.0, 0.0),
    ];
    let efl = |rs: &mklight_core::ResolvedSystem| {
        mklight_trace::paraxial::first_order(rs, rs.primary_wavelength)
            .unwrap()
            .efl
    };
    let trace = |rs: &mklight_core::ResolvedSystem, x: f64, y: f64| {
        let start = Ray::new(
            Vec3::new(x, y, -5.0),
            Vec3::new(0.01, -0.02, 1.0).normalized(),
        );
        let (r, rec) = trace_ray_recorded(rs, &start, 0, &TraceOptions::default());
        assert!(r.status.is_ok(), "({x}, {y}): {:?}", r.status);
        (r.ray, rec[rec.len() - 1].opl, rec[1].normal)
    };
    let close = |a: Vec3, b: Vec3, tol: f64, what: &str| {
        assert!((a - b).norm() < tol, "{what}: {a:?} vs {b:?}");
    };
    // The overlay matters: it moves the focus and the rays.
    assert!((efl(&single) - efl(&bare)).abs() > 1e-4);
    let (with, _, _) = trace(&single, 2.0, -1.5);
    let (without, _, _) = trace(&bare, 2.0, -1.5);
    assert!((with.pos - without.pos).norm() > 1e-5);

    for (k, stack) in stacks.into_iter().enumerate() {
        let stacked = resolve(doe_element(stack));
        let (a, b) = (efl(&stacked), efl(&single));
        assert!((a - b).abs() < 1e-12 * b.abs(), "stack {k}: EFL {a} vs {b}");
        for i in -3..=3 {
            for j in -3..=3 {
                let (x, y) = (0.9 * i as f64, 0.9 * j as f64);
                let (ray, opl, normal) = trace(&stacked, x, y);
                let (ray0, opl0, normal0) = trace(&single, x, y);
                let what = format!("stack {k} at ({x}, {y})");
                close(ray.pos, ray0.pos, 1e-12, &what);
                close(ray.dir, ray0.dir, 1e-13, &what);
                close(normal, normal0, 1e-13, &what);
                assert!((opl - opl0).abs() < 1e-12, "{what}: OPL {opl} vs {opl0}");
            }
        }
    }
    // The real-ray tracer aims through the stack.
    tracer(doe_element(overlay(
        overlay(asphere(), 3.0, 0.0),
        0.0,
        -2.0,
    )));
}

/// `solve_reports.json` with the two values of each report's `run_info`
/// that depend on the run removed.
fn without_run_times(text: &str) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(text).expect("solve reports are JSON");
    let entries = v.as_array_mut().expect("an array of entries");
    assert!(!entries.is_empty());
    for entry in entries {
        let info = entry["report"]["run_info"]
            .as_object_mut()
            .expect("run_info");
        for key in ["started_unix_ms", "duration_ms"] {
            assert!(info.remove(key).is_some(), "run_info.{key}");
        }
    }
    v
}

#[test]
fn committed_solve_reports_are_what_the_generator_writes() {
    let path = generator::directory().join(generator::FILE);
    let committed =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        without_run_times(&committed),
        without_run_times(&generator::solve_reports()),
        "{}: regenerate with `cargo run -p mklight-trace --example dump_solve_fixtures`",
        path.display()
    );
}
