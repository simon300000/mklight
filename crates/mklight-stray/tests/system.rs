//! Scenes built from optical systems: structure, mechanics, determinism and
//! job control.

use crate::common;

use common::*;
use mklight_core::math::Vec3;
use mklight_core::model::{
    ApertureShape, Interaction, MaterialRef, ScatterModel, SemiDiameter, Surface, SurfaceShape,
};
use mklight_core::{CancelToken, JobContext, MkError, Progress};
use mklight_stray::geom::{BruteForce, Shape};
use mklight_stray::scene::{Behavior, OpaqueKind, Side};
use mklight_stray::{ClassMask, PathClass, SourceSpec, SplitMode, StraySettings};

#[test]
fn scene_structure_of_a_barrelled_singlet() {
    let sc = scene(barrel_singlet(ScatterModel::Lambertian { tis: 0.5 }));
    let mut kinds = Vec::new();
    for i in 0..sc.primitives().len() {
        kinds.push(match sc.behavior(i) {
            Behavior::Interface(f) => format!("if{}", f.surface()),
            Behavior::Opaque(o) => format!("{:?}", o.kind),
            Behavior::Detector => "det".into(),
        });
    }
    kinds.sort();
    // Surface 1 is the stop: its non-clear zone and the ring outside the
    // element are stop plates. Surface 2's non-clear zone is a mount zone.
    assert_eq!(
        kinds,
        [
            "Baffle",
            "Barrel",
            "LensEdge",
            "Mount",
            "StopPlate",
            "StopPlate",
            "det",
            "if1",
            "if2"
        ]
    );
    assert_eq!(sc.sequence, vec![1, 2, 3]);
    // Edge cylinder joins the rims: sag(9) of R = ±40 at z = 0 and z = 4.
    let sag = 40.0 - (1600.0f64 - 81.0).sqrt();
    let edge = (0..sc.primitives().len())
        .find(|&i| matches!(sc.behavior(i), Behavior::Opaque(o) if o.kind == OpaqueKind::LensEdge))
        .unwrap();
    match &sc.primitives()[edge].shape {
        Shape::Cylinder(c) => {
            assert_eq!(c.radius, 9.0);
            assert!((c.z0 - sag).abs() < 1e-12 && (c.z1 - (4.0 - sag)).abs() < 1e-12);
        }
        s => panic!("{s:?}"),
    }
    assert!(sc.warnings.is_empty(), "{:?}", sc.warnings);
}

#[test]
fn stop_plate_flats_and_mirror_sides() {
    // Stop (air plane) with a fixed SD in front of a meniscus whose surfaces
    // have different mechanical semi-diameters, followed by a fold mirror.
    let mut s = mklight_core::model::OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut stop = Surface::new(1, SurfaceShape::Plane, 5.0, MaterialRef::Air);
    stop.semi_diameter = SemiDiameter::Fixed { value: 4.0 };
    let mut a = Surface::new(
        2,
        SurfaceShape::sphere(30.0),
        3.0,
        MaterialRef::Constant { n: 1.6 },
    );
    let mut b = Surface::new(3, SurfaceShape::sphere(50.0), 20.0, MaterialRef::Air);
    fixed(&mut a, 6.0);
    fixed(&mut b, 5.0);
    a.edge.mechanical_semi_diameter = Some(7.0);
    b.edge.mechanical_semi_diameter = Some(6.0);
    let mut m = Surface::new(4, SurfaceShape::Plane, -20.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    fixed(&mut m, 10.0);
    let mut img = Surface::new(5, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut img, 5.0);
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        stop,
        a,
        b,
        m,
        img,
    ];
    s.stop = 1;
    let sc = scene(s);
    let mut flats = 0;
    let mut plate = false;
    for i in 0..sc.primitives().len() {
        match sc.behavior(i) {
            Behavior::Opaque(o) if o.kind == OpaqueKind::LensFlat => {
                flats += 1;
                assert_eq!(o.surface, Some(3), "the smaller back rim gets the flat");
            }
            Behavior::Opaque(o) if o.kind == OpaqueKind::StopPlate => {
                plate = true;
                if let Shape::Sag(p) = &sc.primitives()[i].shape {
                    assert_eq!(
                        p.mask(),
                        Some(&mklight_stray::geom::Mask::Outside(
                            ApertureShape::Circular { radius: 4.0 }
                        ))
                    );
                    assert_eq!(
                        p.r_outer(),
                        12.0,
                        "largest interface MSD (mirror, 11) + margin"
                    );
                }
            }
            Behavior::Interface(f) if f.surface() == 4 => {
                assert_eq!(f.passes[0].incident_side, Side::Minus);
                assert_eq!(
                    sc.media[f.medium_minus as usize],
                    sc.media[f.medium_plus as usize]
                );
            }
            _ => {}
        }
    }
    assert_eq!(flats, 1);
    assert!(plate);
    assert_eq!(sc.sequence, vec![2, 3, 4, 5]);
    // After one mirror, the image is approached from its +z side.
    assert_eq!(sc.detector.accept, mklight_stray::scene::Accept::Plus);
}

#[test]
fn automatic_semi_diameters_need_tracer_values() {
    let mut s = window(1.5, 5.0, 20.0);
    s.surfaces[1].semi_diameter = SemiDiameter::Auto;
    let rs = resolve(s);
    let err = mklight_stray::Scene::from_system(&rs, &Default::default()).unwrap_err();
    assert!(matches!(err, MkError::InvalidModel(_)));
    let opts = mklight_stray::SceneOptions {
        clear_semi_diameters: vec![None, Some(9.0)],
        ..Default::default()
    };
    let sc = mklight_stray::Scene::from_system(&rs, &opts).unwrap();
    assert_eq!(sc.surfaces[1].as_ref().unwrap().clear_semi_diameter, 9.0);
}

fn barrel_source(sc: &mklight_stray::Scene) -> SourceSpec {
    SourceSpec::collimated_filling_entrance(sc, 0.0, 25.0, 1.4, 1.0).unwrap()
}

#[test]
fn blackened_barrel_reduces_detected_scatter() {
    let set = StraySettings {
        rays: 60_000,
        seed: 5,
        ..Default::default()
    };
    let bright = scene(barrel_singlet(ScatterModel::Lambertian { tis: 0.6 }));
    let black = scene(barrel_singlet(ScatterModel::Lambertian { tis: 0.02 }));
    let rb = run(&bright, &barrel_source(&bright), &set);
    let rk = run(&black, &barrel_source(&black), &set);
    let (sb, sk) = (
        rb.energy.detected_in(PathClass::Scatter),
        rk.energy.detected_in(PathClass::Scatter),
    );
    assert!(sb > 0.0 && sk > 0.0);
    assert!(sk < 0.15 * sb, "black {sk} vs bright {sb}");
    // At 25° the image lies outside the detector. Specular light that never
    // touches the barrel (ghosts, sneak paths) follows identical random
    // streams in both runs and is unchanged.
    for c in [PathClass::Ghost, PathClass::Sneak] {
        let (a, b) = (rb.energy.detected_in(c), rk.energy.detected_in(c));
        assert!((a - b).abs() <= 1e-12 * a.max(b), "{c:?}: {a} {b}");
    }
    assert!(
        rk.energy.absorbed_in(mklight_stray::Sink::Barrel)
            > rb.energy.absorbed_in(mklight_stray::Sink::Barrel)
    );
    assert!(rb.run_info.is_some());
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let sc = scene(barrel_singlet(ScatterModel::Abg {
        a: 1e-3,
        b: 1e-3,
        g: 1.5,
    }));
    let src = barrel_source(&sc);
    for mode in [
        SplitMode::default(),
        SplitMode::Split {
            max_splits: 4,
            rare_floor: 0.1,
        },
    ] {
        let set = StraySettings {
            rays: 6000,
            seed: 42,
            chunk_rays: 256,
            mode,
            ..Default::default()
        };
        let go = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| mklight_stray::run(&sc, &src, &set, &JobContext::new()).unwrap())
        };
        let a = go(1);
        let b = go(4);
        assert_eq!(a.energy, b.energy);
        assert_eq!(a.detector, b.detector);
        assert!(a.energy.detected > 0.0);
        // The brute-force caster gives the same answer as the BVH.
        let c = mklight_stray::run_with(
            &sc,
            &BruteForce(sc.primitives()),
            &src,
            &set,
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(a.energy, c.energy);
    }
}

#[test]
fn cancellation_and_progress() {
    let sc = scene(window(1.5, 5.0, 20.0));
    let src = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 2.0,
        start_distance: 30.0,
        power: 1.0,
    };
    let set = StraySettings {
        rays: 5000,
        ..Default::default()
    };
    let ctx = JobContext::with(CancelToken::new(), Progress::default());
    ctx.cancel.cancel();
    assert_eq!(
        mklight_stray::run(&sc, &src, &set, &ctx).unwrap_err(),
        MkError::Cancelled
    );
    let ctx = JobContext::new();
    let r = mklight_stray::run(&sc, &src, &set, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    assert_eq!(r.energy.rays, 5000);
    assert!(r.energy.is_balanced());
}

#[test]
fn owned_classes_exclude_ghost_solver_paths() {
    let sc = scene(window(1.5, 5.0, 20.0));
    let src = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 2.0,
        start_distance: 30.0,
        power: 1.0,
    };
    let set = StraySettings {
        rays: 2000,
        owned_classes: ClassMask::STRAY_WITH_GHOST,
        ..Default::default()
    };
    let r = run(&sc, &src, &set);
    // A clean window produces only primary and ghost light: nothing is owned.
    assert!(r.energy.detected > 0.9);
    assert_eq!(r.energy.detected_owned, 0.0);
    let ghost_map: f64 = r.detector.flux_of(PathClass::Ghost).iter().sum();
    assert!((ghost_map - r.energy.detected_in(PathClass::Ghost)).abs() < 1e-15);
}

/// The barrelled singlet behind a 40 mm lens hood: a passive dummy surface
/// 40 mm in front of the lens, and the black barrel (Lambertian, TIS 0.5)
/// running from the hood opening to the image.
fn hooded_singlet() -> mklight_core::model::OpticalSystem {
    use mklight_core::model::{MechanicalPart, SurfaceId};
    let mut s = barrel_singlet(ScatterModel::Lambertian { tis: 0.5 });
    let mut hood = Surface::new(10, SurfaceShape::Plane, 40.0, MaterialRef::Air);
    fixed(&mut hood, 12.0);
    s.surfaces.insert(1, hood);
    s.stop = 2;
    for m in &mut s.mechanics {
        if let MechanicalPart::Barrel { from, .. } = m {
            *from = SurfaceId(10);
        }
    }
    s
}

/// Sun light that enters a lens hood far in front of the first vertex
/// reaches the detector by scattering off the hood: the out-of-field source
/// must cover the hood opening. Its detected flux matches a collimated beam
/// of the same irradiance three times wider, while a disk sized from the
/// front lens alone (the old construction) misses most of it.
#[test]
fn out_of_field_source_covers_a_lens_hood() {
    let sys = stray_system(hooded_singlet(), &mklight_stray::SystemOptions::default());
    assert_eq!(sys.scene.sequence, [2, 3, 4]);
    let set = |rays, seed| StraySettings {
        rays,
        seed,
        ..Default::default()
    };
    let ctx = JobContext::new();
    let detected = |spec: &SourceSpec, rays, seed| {
        let r = mklight_stray::run(&sys.scene, spec, &set(rays, seed), &ctx).unwrap();
        assert!(r.energy.is_balanced(), "{}", r.energy.summary());
        let e = &r.energy;
        (e.detected, e.detected_std_error.unwrap())
    };
    let theta = 30.0;
    let sun = sys.out_of_field_source(0.0, theta, 0.1, 1.0).unwrap();
    let SourceSpec::Collimated {
        direction,
        center,
        radius,
        start_distance,
        ..
    } = sun.spec
    else {
        panic!("{:?}", sun.spec)
    };
    let wide = SourceSpec::Collimated {
        direction,
        center,
        radius: 3.0 * radius,
        start_distance: start_distance + 2.0 * radius,
        power: std::f64::consts::PI * 9.0 * radius * radius,
    };
    // Old construction: centred on the lens vertex, 1.1 × the largest
    // mechanical radius (the 12 mm barrel).
    let lens = sys.scene.surfaces[2].as_ref().unwrap().frame.origin;
    let narrow = SourceSpec::Collimated {
        direction,
        center: lens,
        radius: 13.2,
        start_distance,
        power: std::f64::consts::PI * 13.2 * 13.2,
    };
    let (a, sa) = detected(&sun.spec, 600_000, 81);
    let (b, sb) = detected(&wide, 3_000_000, 82);
    let (c, sc) = detected(&narrow, 600_000, 83);
    let zw = (a - b) / sa.hypot(sb);
    let zn = (c - b) / sc.hypot(sb);
    println!(
        "hood at {theta}°: sun source {a:.4e} ± {sa:.1e} W, 3× wider beam {b:.4e} ± {sb:.1e} W (z = {zw:+.2}); lens-sized disk {c:.4e} ± {sc:.1e} W (z = {zn:+.2})"
    );
    assert!(a > 0.0 && sa < 0.1 * a, "{a} ± {sa}");
    assert!(zw.abs() < 3.0, "z = {zw}");
    assert!(
        zn < -5.0,
        "the test cannot tell a lens-sized disk apart: z = {zn}"
    );
}
