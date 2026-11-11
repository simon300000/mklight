//! DOE visit capacity of path signatures.
use crate::common;

use common::*;
use mklight_core::model::*;
use mklight_core::{JobContext, Vec3};
use mklight_ghost::{ClipMode, GhostModel, GhostPath, GhostSettings};
use mklight_stray::path::MAX_DIFFRACTION_VISITS;
use mklight_stray::{ClassMask, PathClass, Signature, SourceSpec, SplitMode, StraySettings};

fn doe() -> SurfaceShape {
    SurfaceShape::Diffractive {
        base: Box::new(SurfaceShape::Plane),
        phase: surface::DiffractivePhase {
            norm_radius: 10.0,
            coefficients: vec![],
            design_order: 1,
            design_wavelength_um: 0.55,
            trace_order: 1,
        },
    }
}

#[test]
fn seven_pass_doe_ghost_matches_explicit_weight_and_exclusion() {
    // Three round trips inside a coated window: the internal phase plate
    // is crossed seven times, while there are only six extra reflections.
    let mut s = window(1.5, 5.0, 10.0);
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 2.0 };
    s.coatings.insert(
        "half".into(),
        CoatingDef::Ideal {
            reflectance: 0.5,
            transmittance: 0.5,
        },
    );
    s.surfaces[1].coating = Some("half".into());
    s.surfaces[2].coating = Some("half".into());
    s.surfaces[2].id = SurfaceId(3);
    s.surfaces[3].id = SurfaceId(4);
    let mut plate = Surface::new(2, doe(), 5.0, MaterialRef::Constant { n: 1.5 });
    fixed(&mut plate, 10.0);
    s.surfaces.insert(2, plate);
    let resolved = resolve(s);
    let tracer =
        std::sync::Arc::new(mklight_trace::SequentialTracer::new(resolved.clone().into()).unwrap());
    let model = GhostModel::new(
        tracer,
        GhostSettings {
            clip: ClipMode::ClearApertures { margin_mm: 0.0 },
            ..Default::default()
        },
    )
    .unwrap();
    let path = GhostPath::new(vec![3, 1, 3, 1, 3, 1]);
    let steps = model.check_path(&path).unwrap();
    assert_eq!(steps.iter().filter(|s| s.surface == 2).count(), 7);
    let generated = model
        .tracer()
        .generate(model.tracer().field(0), 0, 0.0, 0.0)
        .unwrap();
    let explicit = model.trace_steps(&generated.ray, 0, &steps, None);
    assert!(explicit.status.is_ok(), "{explicit:?}");
    // Two transmissions and six reflections, each with weight 1/2.
    let expected = 0.5_f64.powi(8);
    assert!((explicit.weight() - expected).abs() < 1e-15);

    let scene = mklight_stray::Scene::from_system(&resolved, &Default::default()).unwrap();
    let source = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 0.5,
        start_distance: 1.0,
        power: 1.0,
    };
    let set = StraySettings {
        rays: 16,
        max_events: 32,
        roulette_threshold: 0.0,
        mode: SplitMode::Split {
            max_splits: 8,
            rare_floor: 0.1,
        },
        owned_classes: ClassMask::STRAY_WITH_GHOST,
        covered_ghosts: Some([path].into()),
        ..Default::default()
    };
    let result = run(&scene, &source, &set);
    let seven = result
        .signatures
        .iter()
        .find(|s| s.class == PathClass::Ghost && s.signature.diffraction().count() == 7)
        .unwrap();
    assert!(!seven.signature.overflow());
    for (visit, event) in seven.signature.diffraction().enumerate() {
        assert_eq!(
            (event.surface, event.visit, event.order),
            (2, visit as u32, 1)
        );
        assert!(!event.reflected && event.selected);
    }
    assert!((seven.flux - expected).abs() < 1e-15);
    assert!((result.energy.detected_covered_ghost - expected).abs() < 1e-15);
    assert!(result.energy.detected_uncovered_ghost > 0.0);
}

#[test]
fn doe_capacity_preserves_late_order_overrides_and_refuses_overflow_coverage() {
    let mut signature = Signature::default();
    for _ in 0..MAX_DIFFRACTION_VISITS - 1 {
        signature.diffract(2, 1, 1, false);
    }
    signature.diffract(2, -1, 1, false);
    let mut path = GhostPath::primary();
    path.diffraction
        .push(mklight_ghost::path::DiffractionOrder {
            surface: 2,
            visit: (MAX_DIFFRACTION_VISITS - 1) as u32,
            order: -1,
        });
    let covered = [path].into();
    assert!(mklight_stray::tracer::covered(&covered, &signature));
    signature.diffract(2, 1, 1, true);
    assert!(signature.overflow());
    assert!(!mklight_stray::tracer::covered(&covered, &signature));

    // The transport limit counts visits over all DOE surfaces.
    let mut s = window(1.0, 1.0, 1.0);
    s.surfaces.truncate(1);
    for i in 1..=MAX_DIFFRACTION_VISITS + 1 {
        let mut surface = Surface::new(i as u64, doe(), 1.0, MaterialRef::Air);
        fixed(&mut surface, 10.0);
        s.surfaces.push(surface);
    }
    let mut image = Surface::new(
        (MAX_DIFFRACTION_VISITS + 2) as u64,
        SurfaceShape::Plane,
        0.0,
        MaterialRef::Air,
    );
    fixed(&mut image, 10.0);
    s.surfaces.push(image);
    let scene = scene(s);
    let source = SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius: 0.5,
        start_distance: 1.0,
        power: 1.0,
    };
    let error = mklight_stray::run(
        &scene,
        &source,
        &StraySettings {
            covered_ghosts: Some([GhostPath::primary()].into()),
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("33 DOE visits; signatures keep at most 32"),
        "{error}"
    );
}
