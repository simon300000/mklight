//! The designed (Primary) path through real and ideal lenses.

use crate::common;

use common::*;
use mklight_core::math::Vec3;
use mklight_core::model::{MaterialRef, OpticalSystem, ScatterModel, Surface, SurfaceShape};
use mklight_stray::{PathClass, SourceSpec, StrayResult, StraySettings};

/// Flux inside `|x|, |y| ≤ half` of a detector map.
fn central_flux(r: &StrayResult, class: PathClass, half: f64) -> f64 {
    let d = &r.detector.spec;
    let map = r.detector.flux_of(class);
    let mut s = 0.0;
    for iy in 0..d.ny {
        for ix in 0..d.nx {
            let x = -d.half_x + (ix as f64 + 0.5) * 2.0 * d.half_x / d.nx as f64;
            let y = -d.half_y + (iy as f64 + 0.5) * 2.0 * d.half_y / d.ny as f64;
            if x.abs() <= half && y.abs() <= half {
                s += map[(iy * d.nx + ix) as usize];
            }
        }
    }
    s
}

fn axial_beam(radius: f64) -> SourceSpec {
    SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius,
        start_distance: 30.0,
        power: 1.0,
    }
}

#[test]
fn singlet_primary_image() {
    // The image surface is 4.3 mm inside the paraxial focus (f ≈ 40.7 mm),
    // so the beam forms a blur of ≈ 0.6 mm radius around the axis.
    let sc = scene(barrel_singlet(ScatterModel::Lambertian { tis: 0.02 }));
    let set = StraySettings {
        rays: 20_000,
        ..Default::default()
    };
    let r = run(&sc, &axial_beam(5.0), &set);
    let p = r.energy.detected_in(PathClass::Primary);
    // Two bare n = 1.5 faces: about (1 − 0.04)² with a little more loss off axis.
    assert!(p > 0.90 && p < 0.925, "{}", r.energy.summary());
    assert!(central_flux(&r, PathClass::Primary, 1.0) > 0.999 * p);
    assert_eq!(r.energy.detected_in(PathClass::Sneak), 0.0);
    // Only back-reflected light scattered by the mount reaches the detector as scatter.
    assert!(r.energy.detected_in(PathClass::Scatter) < 1e-4);
}

#[test]
fn ideal_lens_images_to_a_point() {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut lens = Surface::new(
        1,
        SurfaceShape::Paraxial { focal_length: 50.0 },
        50.0,
        MaterialRef::Air,
    );
    fixed(&mut lens, 8.0);
    let mut img = Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut img, 2.0);
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        lens,
        img,
    ];
    let set = StraySettings {
        rays: 5000,
        ..Default::default()
    };
    let r = run(&scene(s), &axial_beam(5.0), &set);
    assert!((r.energy.detected_in(PathClass::Primary) - 1.0).abs() < 1e-12);
    // All flux in the four pixels around the axis.
    assert!((central_flux(&r, PathClass::Primary, 0.02) - 1.0).abs() < 1e-12);
}
