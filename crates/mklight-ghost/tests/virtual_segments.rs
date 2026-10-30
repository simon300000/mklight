//! The virtual-segment rule of the sequence tracer: passive planes may sit
//! on either side of the ray, a physically impossible backward segment
//! between two physical interactions loses the ray, and primary rays lost
//! that way are reported.

mod common;

use common::*;
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, JobContext, MaterialRef, OpticalSystem, SurfaceShape,
};
use mklight_ghost::{
    GhostModel, GhostPath, GhostSettings, IrradianceSettings, SequenceStatus, ghost_irradiance,
};

fn system(surfaces: Vec<mklight_core::Surface>, stop: usize) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.wavelengths = mono();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 5.0)],
    };
    s.surfaces = surfaces;
    s.stop = stop;
    s
}

fn glass() -> MaterialRef {
    MaterialRef::Constant { n: 1.5 }
}

/// Primary rays of the ghost model match the reference tracer, and the
/// given ghost paths trace without loss.
fn check_accepted(m: &GhostModel, ghosts: &[GhostPath]) {
    let tracer = m.tracer();
    let prim = m.check_path(&GhostPath::primary()).unwrap();
    for f in 0..2 {
        let field = tracer.field(f);
        for (px, py) in [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0)] {
            let g = tracer.generate(field, 0, px, py).unwrap();
            let reference = tracer.trace(field, 0, px, py, &tracer.options());
            assert!(reference.status.is_ok());
            let r = m.trace_steps(&g.ray, 0, &prim, None);
            assert_eq!(r.status, SequenceStatus::Ok, "field {f} ({px}, {py})");
            assert!(r.ray.pos.max_abs_diff(reference.ray.pos) < 1e-12);
            for p in ghosts {
                let steps = m.check_path(p).unwrap();
                let q = m.trace_steps(&g.ray, 0, &steps, None);
                assert_eq!(q.status, SequenceStatus::Ok, "{p:?} field {f} ({px}, {py})");
            }
        }
    }
}

#[test]
fn concave_first_surface_behind_a_stop_plane_is_accepted() {
    // Object at infinity, stop plane in air, then a concave surface at the
    // same vertex: its edge (sag ≈ −0.42 mm at 5 mm) lies *before* the stop
    // plane, so the stop → lens segment is negative. Light comes from
    // z = −∞, so nothing before the first physical interaction counts.
    let s = system(
        vec![
            surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            surf(1, SurfaceShape::Plane, 0.0, MaterialRef::Air),
            surf(2, SurfaceShape::sphere(-30.0), 5.0, glass()),
            surf(3, SurfaceShape::Plane, 40.0, MaterialRef::Air),
            surf(4, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ],
        1,
    );
    let m = model(s, unclipped());
    assert!(m.settings().reject_virtual);
    let tracer = m.tracer();
    let g = tracer.generate(tracer.field(0), 0, 0.0, 1.0).unwrap();
    let mut rec = Vec::new();
    let prim = m.check_path(&GhostPath::primary()).unwrap();
    m.trace_steps(&g.ray, 0, &prim, Some(&mut rec));
    assert!(
        rec[1].distance < -0.1,
        "stop → lens segment {}",
        rec[1].distance
    );
    check_accepted(&m, &[GhostPath::new(vec![3, 2])]);
}

#[test]
fn passive_plane_behind_the_ray_is_accepted() {
    // A stop plane listed after a lens but placed 2 mm before its rear
    // vertex: lens → stop is negative, stop → image positive; the distance
    // between the two physical interactions is positive.
    let s = system(
        vec![
            surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            surf(1, SurfaceShape::sphere(40.0), 4.0, glass()),
            surf(2, SurfaceShape::Plane, -2.0, MaterialRef::Air),
            surf(3, SurfaceShape::Plane, 60.0, MaterialRef::Air),
            surf(4, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ],
        3,
    );
    let m = model(s, unclipped());
    check_accepted(&m, &[GhostPath::new(vec![2, 1])]);
}

/// Two plates whose elements overlap by 1 mm (negative air gap).
fn overlapping_plates() -> OpticalSystem {
    system(
        vec![
            surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            surf(1, SurfaceShape::Plane, 5.0, MaterialRef::Air),
            surf(2, SurfaceShape::Plane, 5.0, glass()),
            surf(3, SurfaceShape::Plane, -1.0, MaterialRef::Air),
            surf(4, SurfaceShape::Plane, 5.0, glass()),
            surf(5, SurfaceShape::Plane, 20.0, MaterialRef::Air),
            surf(6, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ],
        1,
    )
}

#[test]
fn impossible_backward_segment_is_rejected() {
    let m = model(overlapping_plates(), unclipped());
    let tracer = m.tracer();
    let field = tracer.field(1);
    let g = tracer.generate(field, 0, 0.0, 0.7).unwrap();
    let reference = tracer.trace(field, 0, 0.0, 0.7, &tracer.options());
    assert!(reference.status.is_ok(), "the sequential model passes it");
    let prim = m.check_path(&GhostPath::primary()).unwrap();
    let r = m.trace_steps(&g.ray, 0, &prim, None);
    assert_eq!(
        r.status,
        SequenceStatus::Virtual {
            step: 3,
            surface: 4
        }
    );
    // A ghost whose backward leg crosses the overlap is lost as well.
    let gs = m.check_path(&GhostPath::new(vec![5, 2])).unwrap();
    assert!(matches!(
        m.trace_steps(&g.ray, 0, &gs, None).status,
        SequenceStatus::Virtual { .. }
    ));
    // Without the rule, the sequence tracer reproduces the reference.
    let loose = model(
        overlapping_plates(),
        GhostSettings {
            reject_virtual: false,
            ..unclipped()
        },
    );
    let r = loose.trace_steps(&g.ray, 0, &prim, None);
    assert!(r.status.is_ok());
    assert!(r.ray.pos.max_abs_diff(reference.ray.pos) < 1e-12);
}

#[test]
fn primary_virtual_losses_are_reported() {
    let settings = IrradianceSettings {
        paths: Some(Vec::new()),
        pupil_samples: 8,
        overfill: 1.0,
        map_bins: [4, 4],
        map_half_size: Some([10.0, 10.0]),
        ..Default::default()
    };
    let warns = |sys| {
        ghost_irradiance(&model(sys, unclipped()), &settings, &JobContext::new())
            .unwrap()
            .run
            .warnings
    };
    let w = warns(overlapping_plates());
    assert!(
        w.iter().any(|w| w.contains("physically impossible")),
        "{w:?}"
    );
    assert!(warns(plate(1.5, 5.0, 20.0, &[0.0, 5.0], None)).is_empty());
}
