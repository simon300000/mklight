//! Closed-enclosure energy conservation: an integrating sphere.
//!
//! A sphere of radius `R` with Lambertian walls of reflectance `ρ` has a
//! detector cap covering the area fraction `f`. An isotropic source at the
//! centre lights the wall uniformly, and a Lambertian reflection inside a
//! sphere lights the whole sphere uniformly again. The detected fraction is
//! therefore the sphere-multiplier series
//!
//! ```text
//! f · (1 + ρ(1−f) + (ρ(1−f))² + …) = f / (1 − ρ(1 − f)),
//! ```
//!
//! and every other photon is absorbed by the walls: nothing escapes.

use std::sync::Arc;

use mklight_core::JobContext;
use mklight_core::materials::coating::ResolvedCoating;
use mklight_core::math::{Frame, Mat3, Vec3};
use mklight_core::model::{ScatterModel, SurfaceShape};
use mklight_stray::geom::{SagPatch, Shape};
use mklight_stray::scene::{
    Accept, Behavior, DetectorSpec, Interface, InterfaceKind, Opaque, OpaqueKind, Side,
};
use mklight_stray::{
    Bsdf, PathClass, Scene, SceneBuilder, Sink, SourceSpec, SplitMode, StraySettings,
};

const R: f64 = 10.0;

fn sphere(rho: f64, port: f64) -> Scene {
    let mut b = SceneBuilder::new(vec![0.55], vec![1.0]);
    b.add_medium(vec![1.0], vec![0.0]);
    let wall = || {
        Behavior::Opaque(Opaque {
            kind: OpaqueKind::Generic,
            surface: None,
            finish: Some(Arc::new(
                Bsdf::new(&ScatterModel::Lambertian { tis: rho }).unwrap(),
            )),
            specular_reflectance: 0.0,
        })
    };
    let cap = |r0: f64, r1: f64| {
        Shape::Sag(SagPatch::new(SurfaceShape::sphere(R), r0, r1, None).unwrap())
    };
    // South cap (vertex at the origin): detector port plus wall.
    b.add(Frame::IDENTITY, cap(0.0, port), Behavior::Detector, "port");
    b.add(Frame::IDENTITY, cap(port, R), wall(), "south");
    // North cap: vertex at z = 2R, flipped.
    let north = Frame {
        origin: Vec3::new(0.0, 0.0, 2.0 * R),
        rot: Mat3::rot_x(std::f64::consts::PI),
    };
    b.add(north, cap(0.0, R), wall(), "north");
    b.set_detector(DetectorSpec {
        surface: usize::MAX,
        accept: Accept::Plus,
        half_x: port,
        half_y: port,
        nx: 4,
        ny: 4,
    });
    b.build().unwrap()
}

fn source() -> SourceSpec {
    SourceSpec::Point {
        position: Vec3::new(0.0, 0.0, R),
        axis: Vec3::Z,
        half_angle_deg: 180.0,
        intensity: 1.0 / (4.0 * std::f64::consts::PI),
    }
}

#[test]
fn integrating_sphere_multiplier_and_closure() {
    let (rho, port) = (0.9, 3.0);
    let h = R - (R * R - port * port).sqrt();
    let f = h / (2.0 * R);
    let want = f / (1.0 - rho * (1.0 - f));
    let sc = sphere(rho, port);
    let mut vals = Vec::new();
    for seed in 0..6 {
        let set = StraySettings {
            rays: 20_000,
            seed,
            ..Default::default()
        };
        let r = mklight_stray::run(&sc, &source(), &set, &JobContext::new()).unwrap();
        let e = &r.energy;
        assert!(e.is_balanced(), "{}", e.summary());
        // Closed: nothing escapes except through the measure-zero equator seam.
        assert!(e.escaped < 1e-6, "{}", e.summary());
        assert_eq!(e.truncated, 0.0);
        // The walls absorb the rest; MC adjustment only from roulette.
        assert!(
            (e.detected + e.absorbed_in(Sink::Generic) - (1.0 + e.mc_adjustment)).abs() < 1e-12
        );
        assert_eq!(
            e.detected,
            e.detected_in(PathClass::Scatter) + e.detected_in(PathClass::Primary)
        );
        vals.push(e.detected);
    }
    let k = vals.len() as f64;
    let m = vals.iter().sum::<f64>() / k;
    let se = (vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (k - 1.0) / k).sqrt();
    assert!((m - want).abs() < 5.0 * se, "{m} ± {se} vs {want}");
    assert!(se / want < 0.02, "{se}");
}

/// Analog transport in the same sphere: each path ends in exactly one sink,
/// and the MC adjustment (from roulette only) averages out.
#[test]
fn analog_enclosure_books_every_photon() {
    let sc = sphere(0.7, 3.0);
    let set = StraySettings {
        rays: 50_000,
        seed: 9,
        mode: SplitMode::Analog,
        roulette_threshold: 0.0,
        ..Default::default()
    };
    let r = mklight_stray::run(&sc, &source(), &set, &JobContext::new()).unwrap();
    let e = &r.energy;
    assert_eq!(e.mc_adjustment, 0.0);
    assert!(e.is_balanced());
    let h = R - (R * R - 9.0f64).sqrt();
    let f = h / (2.0 * R);
    let want = f / (1.0 - 0.7 * (1.0 - f));
    let sigma = (want * (1.0 - want) / 50_000.0).sqrt();
    assert!(
        (e.detected - want).abs() < 5.0 * sigma,
        "{} vs {want}",
        e.detected
    );
}

/// A glass sphere in air: a ray started inside the glass reaches the
/// surface from inside and, with an index-matched (T = 1) ideal coating,
/// leaves without loss. With bare Fresnel, total internal reflection traps
/// light that is outside the escape cone until it scatters or is absorbed.
#[test]
fn rays_from_inside_glass_and_tir() {
    let build = |coating: ResolvedCoating, alpha: f64| {
        let mut b = SceneBuilder::new(vec![0.55], vec![1.0]);
        let air = b.add_medium(vec![1.0], vec![0.0]);
        let glass = b.add_medium(vec![1.5], vec![alpha]);
        let iface = || {
            Behavior::Interface(Interface::single(
                1,
                InterfaceKind::Refract,
                air,
                glass,
                Side::Minus,
                Arc::new(coating.clone()),
            ))
        };
        let cap = || Shape::Sag(SagPatch::new(SurfaceShape::sphere(R), 0.0, R, None).unwrap());
        b.add(Frame::IDENTITY, cap(), iface(), "south");
        let north = Frame {
            origin: Vec3::new(0.0, 0.0, 2.0 * R),
            rot: Mat3::rot_x(std::f64::consts::PI),
        };
        b.add(north, cap(), iface(), "north");
        // A detector far away so the scene is valid; nothing need reach it.
        let far = Frame {
            origin: Vec3::new(0.0, 0.0, 1000.0),
            rot: Mat3::IDENTITY,
        };
        b.add(
            far,
            Shape::Sag(SagPatch::new(SurfaceShape::Plane, 0.0, 1.0, None).unwrap()),
            Behavior::Detector,
            "det",
        );
        b.set_detector(DetectorSpec {
            surface: usize::MAX,
            accept: Accept::Both,
            half_x: 1.0,
            half_y: 1.0,
            nx: 1,
            ny: 1,
        });
        b.build().unwrap()
    };
    // Index-matched coating (T = 1), source 6 mm off centre: no ray exceeds
    // the critical angle (R/(n·d) > 1), so everything leaves the ball.
    let src = |d: f64| SourceSpec::Point {
        position: Vec3::new(0.0, d, R),
        axis: Vec3::Z,
        half_angle_deg: 180.0,
        intensity: 1.0 / (4.0 * std::f64::consts::PI),
    };
    let set = StraySettings {
        rays: 20_000,
        start_medium: Some(1),
        ..Default::default()
    };
    let ideal = ResolvedCoating::Ideal {
        reflectance: 0.0,
        transmittance: 1.0,
    };
    let r = mklight_stray::run(&build(ideal, 0.0), &src(6.0), &set, &JobContext::new()).unwrap();
    assert!(
        (r.energy.escaped - 1.0).abs() < 1e-12,
        "{}",
        r.energy.summary()
    );

    // Bare Fresnel, source 9 mm off centre. In a sphere every chord keeps its
    // angle of incidence, so the directions with sin ψ > R/(n·d) (ψ from the
    // radius) are trapped by total internal reflection until the glass
    // absorbs them. For isotropic emission that fraction is √(1 − (R/(n d))²).
    let (n, d) = (1.5, 9.0);
    let s0 = R / (n * d);
    let trapped = (1.0 - s0 * s0).sqrt();
    let r = mklight_stray::run(
        &build(ResolvedCoating::Bare, 0.05),
        &src(d),
        &set,
        &JobContext::new(),
    )
    .unwrap();
    let e = &r.energy;
    assert!(e.is_balanced());
    let sigma = (0.25 / 20_000.0f64).sqrt();
    assert!(
        e.absorbed_in(Sink::Bulk) > trapped - 5.0 * sigma,
        "{} vs trapped {trapped}",
        e.summary()
    );
    assert!(e.escaped < 1.0 - trapped + 5.0 * sigma);
    assert!(e.escaped > 0.1);
}
