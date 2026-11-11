//! Apertures as geometry: obscured mirrors, aperture stops, non-clear zones
//! and surfaces visited twice by the sequential model.

use crate::common;

use common::*;
use mklight_core::materials::coating::fresnel;
use mklight_core::math::Vec3;
use mklight_core::model::*;
use mklight_stray::geom::{Mask, Shape};
use mklight_stray::scene::{Behavior, OpaqueKind, Side};
use mklight_stray::{ClassMask, PathClass, Scene, SceneOptions, Sink, SourceSpec, SplitMode};
use mklight_stray::{StraySettings, scene::InterfaceKind};

fn axial_beam(radius: f64, start: f64) -> SourceSpec {
    SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius,
        start_distance: start,
        power: 1.0,
    }
}

fn count(sc: &Scene, kind: OpaqueKind) -> usize {
    (0..sc.primitives().len())
        .filter(|&i| matches!(sc.behavior(i), Behavior::Opaque(o) if o.kind == kind))
        .count()
}

/// Binomial standard error of a detected fraction `p` over `n` analog rays.
fn binomial(p: f64, n: f64) -> f64 {
    (p * (1.0 - p) / n).sqrt()
}

/// Cassegrain: a spherical primary (R = −200) with a central hole
/// (annular aperture 12…30) and a secondary (R = −82.5) 70 mm in front of it,
/// which relays the focus through the hole to the image 40 mm behind the
/// primary. Both mirrors are perfect, so exactly the annulus of the beam
/// reaches the image: `(30² − 12²)/30² = 0.84`. Light through the hole must
/// not be reflected by a solid centre.
#[test]
fn cassegrain_with_a_holed_primary_images_the_annulus() {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut p = Surface::new(1, SurfaceShape::sphere(-200.0), -70.0, MaterialRef::Air);
    p.interaction = Interaction::Reflect;
    fixed(&mut p, 30.0);
    p.aperture = Some(ApertureShape::Annular {
        inner_radius: 12.0,
        outer_radius: 30.0,
    });
    let mut sec = Surface::new(2, SurfaceShape::sphere(-82.5), 110.0, MaterialRef::Air);
    sec.interaction = Interaction::Reflect;
    fixed(&mut sec, 10.0);
    let mut img = Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut img, 8.0);
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        p,
        sec,
        img,
    ];
    s.stop = 1;
    let sc = scene(s);
    // The primary is a patch with a hole; its non-clear ring is a stop plate
    // (it is the stop) and there is no obscuration in the hole.
    let primary = (0..sc.primitives().len())
        .find(|&i| matches!(sc.behavior(i), Behavior::Interface(f) if f.surface() == 1))
        .unwrap();
    match &sc.primitives()[primary].shape {
        Shape::Sag(p) => assert_eq!((p.r_inner(), p.r_outer()), (12.0, 30.0)),
        s => panic!("{s:?}"),
    }
    assert_eq!(count(&sc, OpaqueKind::StopPlate), 1);
    let n = 40_000;
    let r = run(
        &sc,
        &axial_beam(30.0, 300.0),
        &StraySettings {
            rays: n,
            mode: SplitMode::Analog,
            ..Default::default()
        },
    );
    let e = &r.energy;
    let want = (900.0 - 144.0) / 900.0;
    let got = e.detected_in(PathClass::Primary);
    assert!(
        (got - want).abs() < 5.0 * binomial(want, n as f64),
        "{got} vs {want}\n{}",
        e.summary()
    );
    assert_eq!(e.detected_in(PathClass::Ghost), 0.0);
    assert_eq!(e.detected_in(PathClass::Sneak), 0.0);
    // The secondary's back (r < 10) blocks the centre of the incoming beam.
    let back = e.absorbed_in(Sink::MirrorBack);
    assert!(
        (back - 0.1111).abs() < 5.0 * binomial(0.1111, n as f64),
        "{back}"
    );
}

/// A rectangular fold mirror ends at its aperture: its patch is masked to the
/// rectangle and it has no mount zone unless a mechanical semi-diameter is
/// given.
#[test]
fn rectangular_mirror_is_masked() {
    let mut s = window(1.5, 5.0, 20.0);
    s.surfaces[2].thickness = 20.0;
    let mut m = Surface::new(9, SurfaceShape::Plane, -10.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    m.aperture = Some(ApertureShape::Rectangular {
        half_x: 4.0,
        half_y: 12.0,
    });
    s.surfaces.insert(3, m);
    let rect = ApertureShape::Rectangular {
        half_x: 4.0,
        half_y: 12.0,
    };
    let sc = scene(s.clone());
    let mirror = (0..sc.primitives().len())
        .find(|&i| matches!(sc.behavior(i), Behavior::Interface(f) if f.kind == InterfaceKind::Mirror))
        .unwrap();
    match &sc.primitives()[mirror].shape {
        Shape::Sag(p) => assert_eq!(p.mask(), Some(&Mask::Inside(rect.clone()))),
        s => panic!("{s:?}"),
    }
    assert!(
        !sc.labels_contain("mount zone 3"),
        "no substrate outside the rectangle"
    );
    s.surfaces[3].edge.mechanical_semi_diameter = Some(14.0);
    let sc = scene(s);
    let zone = (0..sc.primitives().len())
        .find(|&i| sc.label(i) == "mount zone 3")
        .expect("explicit substrate");
    match &sc.primitives()[zone].shape {
        Shape::Sag(p) => {
            assert_eq!(p.mask(), Some(&Mask::Outside(rect)));
            assert_eq!(p.r_outer(), 14.0);
        }
        s => panic!("{s:?}"),
    }
}

trait Labels {
    fn labels_contain(&self, l: &str) -> bool;
}
impl Labels for Scene {
    fn labels_contain(&self, l: &str) -> bool {
        (0..self.primitives().len()).any(|i| self.label(i) == l)
    }
}

/// Window behind an air-spaced aperture stop with an automatic
/// semi-diameter, as in an entrance-pupil-specified system.
fn window_behind_auto_stop() -> OpticalSystem {
    let mut s = window(1.5, 5.0, 20.0);
    s.surfaces.insert(
        1,
        Surface::new(9, SurfaceShape::Plane, 5.0, MaterialRef::Air),
    );
    s.stop = 1;
    s
}

/// The stop radius of an automatic stop comes from the supplied clear
/// semi-diameter: only `3²/8²` of a beam of radius 8 passes.
#[test]
fn automatic_stop_uses_the_supplied_semi_diameter() {
    let rs = resolve(window_behind_auto_stop());
    let opts = SceneOptions {
        clear_semi_diameters: vec![None, Some(3.0)],
        ..Default::default()
    };
    let sc = Scene::from_system(&rs, &opts).unwrap();
    assert_eq!(count(&sc, OpaqueKind::StopPlate), 1);
    assert!(sc.warnings.is_empty(), "{:?}", sc.warnings);
    let n = 40_000;
    let r = run(
        &sc,
        &axial_beam(8.0, 30.0),
        &StraySettings {
            rays: n,
            mode: SplitMode::Analog,
            ..Default::default()
        },
    );
    let t = fresnel(1.0, 1.5, 1.0).t_unpolarized();
    let want = t * t * 9.0 / 64.0;
    let got = r.energy.detected_in(PathClass::Primary);
    assert!(
        (got - want).abs() < 5.0 * binomial(want, n as f64),
        "{got} vs {want}\n{}",
        r.energy.summary()
    );
    assert_eq!(r.energy.detected_in(PathClass::Sneak), 0.0);

    // Without a supplied value the stop cannot block, and the scene says so.
    let sc = Scene::from_system(&rs, &SceneOptions::default()).unwrap();
    assert_eq!(count(&sc, OpaqueKind::StopPlate), 0);
    assert!(
        sc.warnings.iter().any(|w| w.contains("aperture stop")),
        "{:?}",
        sc.warnings
    );
}

/// The stop on a lens surface (clear 8, mechanical 9) blocks the non-clear
/// zone and, with the stop ring, the space around the element. Nothing
/// specular passes outside the stop, so no stray-owned specular (`Sneak`)
/// light appears.
#[test]
fn stop_on_a_refracting_surface_blocks_outside_its_aperture() {
    let mut s = window(1.5, 5.0, 20.0);
    fixed(&mut s.surfaces[1], 8.0);
    fixed(&mut s.surfaces[2], 8.0);
    s.surfaces[1].edge.mechanical_semi_diameter = Some(9.0);
    s.surfaces[2].edge.mechanical_semi_diameter = Some(9.0);
    s.stop = 1;
    let t = fresnel(1.0, 1.5, 1.0).t_unpolarized();
    let n = 40_000;
    for polished in [false, true] {
        let opts = SceneOptions {
            polished_outer_zone: polished,
            ..Default::default()
        };
        let sc = Scene::from_system(&resolve(s.clone()), &opts).unwrap();
        // Beam radius 9.5: covers the non-clear zone and part of the ring.
        let r = run(
            &sc,
            &axial_beam(9.5, 30.0),
            &StraySettings {
                rays: n,
                mode: SplitMode::Analog,
                owned_classes: ClassMask::STRAY_WITH_GHOST,
                ..Default::default()
            },
        );
        let e = &r.energy;
        let want = t * t * 64.0 / 90.25;
        let got = e.detected_in(PathClass::Primary);
        assert!(
            (got - want).abs() < 5.0 * binomial(want, n as f64),
            "polished {polished}: {got} vs {want}\n{}",
            e.summary()
        );
        assert_eq!(e.detected_in(PathClass::Sneak), 0.0, "{}", e.summary());
        // Owned light is only what the plates scatter.
        assert!(e.detected_owned < 1e-3, "{}", e.summary());
        // The plates (finish: 5 % Lambertian) absorb 95 % of what they meet.
        let blocked = (90.25 - 64.0) / 90.25;
        let a = e.absorbed_in(Sink::StopPlate);
        assert!(
            (a - 0.95 * blocked).abs() < 5.0 * binomial(blocked, n as f64),
            "{a} vs {}",
            0.95 * blocked
        );
    }
}

/// A surface that is not the stop: by default its non-clear zone is an
/// opaque mount zone. As polished glass (opt-in) it transmits unclipped
/// light, which the classification reports as `Sneak`.
#[test]
fn non_clear_zone_is_opaque_unless_polished() {
    let mut s = window(1.5, 5.0, 20.0);
    fixed(&mut s.surfaces[2], 8.0);
    s.surfaces[2].edge.mechanical_semi_diameter = Some(9.0);
    let t = fresnel(1.0, 1.5, 1.0).t_unpolarized();
    let n = 40_000;
    let set = StraySettings {
        rays: n,
        mode: SplitMode::Analog,
        ..Default::default()
    };
    let src = axial_beam(9.0, 30.0);
    let outer = (81.0 - 64.0) / 81.0;

    let sc = scene(s.clone());
    assert_eq!(count(&sc, OpaqueKind::Mount), 1);
    let r = run(&sc, &src, &set);
    assert_eq!(r.energy.detected_in(PathClass::Sneak), 0.0);
    let a = r.energy.absorbed_in(Sink::Mount);
    let want = t * 0.95 * outer;
    assert!(
        (a - want).abs() < 5.0 * binomial(want, n as f64),
        "{a} vs {want}"
    );

    let opts = SceneOptions {
        polished_outer_zone: true,
        ..Default::default()
    };
    let sc = Scene::from_system(&resolve(s), &opts).unwrap();
    assert_eq!(count(&sc, OpaqueKind::Mount), 0);
    let r = run(&sc, &src, &set);
    let sneak = r.energy.detected_in(PathClass::Sneak);
    let want = t * t * outer;
    assert!(
        (sneak - want).abs() < 5.0 * binomial(want, n as f64),
        "{sneak} vs {want}"
    );
}

/// A central obscuration on a refracting surface (annular aperture) is
/// opaque, unlike the hole of a mirror.
#[test]
fn annular_lens_aperture_is_obscured() {
    let mut s = window(1.5, 5.0, 20.0);
    s.surfaces[1].aperture = Some(ApertureShape::Annular {
        inner_radius: 3.0,
        outer_radius: 10.0,
    });
    let sc = scene(s);
    assert!(
        (0..sc.primitives().len()).any(|i| sc.label(i) == "obscuration 1"
            && matches!(sc.behavior(i), Behavior::Opaque(o) if o.kind == OpaqueKind::StopPlate))
    );
    let n = 40_000;
    let r = run(
        &sc,
        &axial_beam(6.0, 30.0),
        &StraySettings {
            rays: n,
            mode: SplitMode::Analog,
            ..Default::default()
        },
    );
    let t = fresnel(1.0, 1.5, 1.0).t_unpolarized();
    let want = t * t * (36.0 - 9.0) / 36.0;
    let got = r.energy.detected_in(PathClass::Primary);
    assert!(
        (got - want).abs() < 5.0 * binomial(want, n as f64),
        "{got} vs {want}"
    );
}

/// Double-pass element: a plane glass plate (n = 1.5) silvered on its back,
/// with the front face repeated in the sequence for the return pass, and the
/// image 50 mm in front. The repeated face is one primitive with two passes
/// and the solid has one edge. With splitting, every ray delivers exactly
/// `T²` (perfect mirror) to the designed (`Primary`) class.
#[test]
fn double_pass_surface_is_one_primitive_and_primary() {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut s1 = Surface::new(
        1,
        SurfaceShape::Plane,
        5.0,
        MaterialRef::Constant { n: 1.5 },
    );
    let mut s2 = Surface::new(
        2,
        SurfaceShape::Plane,
        -5.0,
        MaterialRef::Constant { n: 1.5 },
    );
    s2.interaction = Interaction::Reflect;
    let mut s3 = Surface::new(3, SurfaceShape::Plane, -50.0, MaterialRef::Air);
    let mut img = Surface::new(4, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    for (x, r) in [
        (&mut s1, 10.0),
        (&mut s2, 10.0),
        (&mut s3, 10.0),
        (&mut img, 15.0),
    ] {
        fixed(x, r);
    }
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        s1,
        s2,
        s3,
        img,
    ];
    s.stop = 1;
    let sc = scene(s);
    let mut passes = Vec::new();
    for i in 0..sc.primitives().len() {
        if let Behavior::Interface(f) = sc.behavior(i) {
            passes.push(
                f.passes
                    .iter()
                    .map(|p| (p.surface, p.incident_side))
                    .collect::<Vec<_>>(),
            );
        }
    }
    passes.sort_by_key(|p| p[0].0);
    assert_eq!(
        passes,
        [
            vec![(1, Side::Minus), (3, Side::Plus)],
            vec![(2, Side::Minus)]
        ]
    );
    assert_eq!(count(&sc, OpaqueKind::LensEdge), 1);
    assert_eq!(sc.sequence, vec![1, 2, 3, 4]);
    assert!(sc.warnings.is_empty(), "{:?}", sc.warnings);

    let r = run(
        &sc,
        &axial_beam(5.0, 20.0),
        &StraySettings {
            rays: 2000,
            mode: SplitMode::Split {
                max_splits: 64,
                rare_floor: 0.1,
            },
            roulette_threshold: 1e-6,
            ..Default::default()
        },
    );
    let f = fresnel(1.0, 1.5, 1.0);
    let (rr, t) = (f.r_unpolarized(), f.t_unpolarized());
    let e = &r.energy;
    assert!(
        (e.detected_in(PathClass::Primary) / (t * t) - 1.0).abs() < 1e-10,
        "{}",
        e.summary()
    );
    // Exits after k internal reflections at the front face deliver T²Rᵏ:
    // Even k ≥ 2 follow canonical ghost paths; odd k leave the sequential
    // train toward object space but meet this folded detector (Sneak, as
    // does the external front reflection R). Nothing designed is misclassified.
    let sneak = e.detected_in(PathClass::Sneak);
    let want = rr + t * t * rr / (1.0 - rr * rr);
    assert!((sneak / want - 1.0).abs() < 1e-6, "{sneak} vs {want}");
    let ghost = e.detected_in(PathClass::Ghost);
    let want = t * t * rr * rr / (1.0 - rr * rr);
    assert!((ghost / want - 1.0).abs() < 1e-4, "{ghost} vs {want}");
}
