//! Solves and pickups (`docs/SOLVES.md`): known answers from first-order
//! optics and geometry, conservation of constrained sums, coupling,
//! configurations, serde and validation.

mod common;

use std::sync::Arc;

use common::{angle_fields, infinite_system, mono, resolve, surf};
use mklight_core::materials::MaterialLibrary;
use mklight_core::model::Configuration;
use mklight_core::param::ParamPath;
use mklight_core::{
    ApertureSpec, CancelToken, Field, FieldKind, FieldSet, Interaction, JobContext, MaterialRef,
    MkError, OpticalSystem, Progress, SemiDiameter, Snapshot, Solve, SolveKind, SurfaceId,
    SurfaceShape, samples,
};
use mklight_trace::{
    FirstOrder, Paraxial, SemiDiameterSettings, SequentialTracer, SolveMethod, SolveSettings,
    apply_solves, apply_solves_all, apply_solves_configuration, apply_solves_with, solved_params,
};

fn lib() -> MaterialLibrary {
    MaterialLibrary::with_builtin()
}

fn first_order(sys: &OpticalSystem) -> FirstOrder {
    Paraxial::new(&resolve(sys.clone()))
        .expect("paraxial")
        .primary()
        .clone()
}

fn id(i: u64) -> SurfaceId {
    SurfaceId(i)
}
fn c(i: u64) -> ParamPath {
    ParamPath::Curvature { surface: id(i) }
}
fn t(i: u64) -> ParamPath {
    ParamPath::Thickness { surface: id(i) }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

fn solve(sys: &OpticalSystem) -> (OpticalSystem, mklight_trace::SolveReport) {
    apply_solves(sys, &lib()).unwrap_or_else(|e| panic!("{}: {e}", sys.name))
}

fn root_finding_only() -> SolveSettings {
    SolveSettings {
        closed_form: false,
        ..SolveSettings::default()
    }
}

fn last_thickness_id(sys: &OpticalSystem) -> SurfaceId {
    sys.surfaces[sys.image_index() - 1].id
}

/// Object-at-infinity singlet, N-BK7, stop on the front surface.
fn singlet() -> OpticalSystem {
    let mut s = infinite_system(
        20.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(60.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(2, SurfaceShape::sphere(-350.0), 90.0, MaterialRef::Air),
        ],
        3,
    );
    s.fields = angle_fields(&[0.0, 5.0]);
    s.name = "singlet".into();
    s
}

// ---------------------------------------------------------------------------
// Marginal-ray height.

#[test]
fn marginal_height_zero_on_the_last_thickness_is_the_paraxial_focus_for_every_sample() {
    for (name, mut sys) in samples::all() {
        let j = sys.image_index() - 1;
        let reference = first_order(&sys);
        // Defocus, then let the solve find the focus again.
        sys.surfaces[j].thickness += 3.7 * sys.surfaces[j].thickness.signum();
        sys.solves = vec![Solve::marginal_ray_height(sys.surfaces[j].id, 0.0)];
        let (out, report) = solve(&sys);
        let solved = out.surfaces[j].thickness;
        assert!(
            rel(solved, reference.image_distance) < 1e-12,
            "{name}: {solved} vs {}",
            reference.image_distance
        );
        if sys.object_at_infinity() {
            assert!(rel(solved, reference.bfl) < 1e-12, "{name}: BFL");
        }
        let fo = first_order(&out);
        let img = out.image_index();
        assert!(
            fo.marginal.y[img].abs() < 1e-12 * fo.marginal.y[1].abs(),
            "{name}: marginal height on the image {}",
            fo.marginal.y[img]
        );
        let o = &report.outcomes[0];
        assert_eq!(o.method, SolveMethod::ClosedForm, "{name}");
        assert!(o.residual.abs() < 1e-12 * fo.marginal.y[1].abs(), "{name}");
        assert_eq!(report.solved_params, vec![t(sys.surfaces[j].id.0)]);
        // The input is untouched.
        assert_ne!(sys.surfaces[j].thickness, solved);
    }
}

#[test]
fn marginal_height_nonzero_and_on_an_inner_thickness() {
    // Height 2 mm on the image: a defocus with a known transverse size.
    let mut sys = singlet();
    sys.solves = vec![Solve::marginal_ray_height(id(2), 2.0)];
    let (out, _) = solve(&sys);
    let fo = first_order(&out);
    assert!((fo.marginal.y[3] - 2.0).abs() < 1e-12);
    // On the lens thickness: y₂ = y₁ + t₁ u₁ = 9.5 mm.
    let mut sys = singlet();
    sys.solves = vec![Solve::marginal_ray_height(id(1), 9.5)];
    let (out, _) = solve(&sys);
    let fo = first_order(&out);
    assert!((fo.marginal.y[2] - 9.5).abs() < 1e-12);
    let expect = (9.5 - 10.0) / fo.marginal.u[1];
    assert!(rel(out.surfaces[1].thickness, expect) < 1e-12);
}

#[test]
fn marginal_height_for_every_aperture_specification() {
    // A stop 10 mm in front of an ideal f = 100 lens: each spec gives some
    // marginal ray, and the focus must not depend on it.
    let build = |aperture: ApertureSpec, finite: bool| {
        let mut sys = OpticalSystem::new_empty();
        sys.wavelengths = mono();
        sys.aperture = aperture;
        let mut stop = surf(1, SurfaceShape::Plane, 10.0, MaterialRef::Air);
        stop.semi_diameter = SemiDiameter::Fixed { value: 6.0 };
        let t0 = if finite { 300.0 } else { f64::INFINITY };
        sys.surfaces = vec![
            surf(0, SurfaceShape::Plane, t0, MaterialRef::Air),
            stop,
            surf(
                2,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                80.0,
                MaterialRef::Air,
            ),
            surf(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        sys.stop = 1;
        sys.solves = vec![Solve::marginal_ray_height(id(2), 0.0)];
        sys
    };
    let specs = [
        (ApertureSpec::EntrancePupilDiameter { value: 12.0 }, false),
        (ApertureSpec::ImageSpaceFNumber { value: 8.0 }, false),
        (ApertureSpec::ParaxialWorkingFNumber { value: 8.0 }, false),
        (ApertureSpec::FloatByStopSize, false),
        (ApertureSpec::EntrancePupilDiameter { value: 12.0 }, true),
        (ApertureSpec::ObjectSpaceNa { value: 0.02 }, true),
        (ApertureSpec::ParaxialWorkingFNumber { value: 8.0 }, true),
        (ApertureSpec::FloatByStopSize, true),
    ];
    for (spec, finite) in specs {
        let (out, _) = solve(&build(spec, finite));
        // Thin-lens equation: 1/b = 1/f − 1/a.
        let expect = if finite {
            1.0 / (1.0 / 100.0 - 1.0 / 310.0)
        } else {
            100.0
        };
        assert!(
            rel(out.surfaces[2].thickness, expect) < 1e-12,
            "{spec:?} finite={finite}: {}",
            out.surfaces[2].thickness
        );
    }
}

#[test]
fn concave_mirror_focus_is_negative() {
    // R = −200 mirror: focus 100 mm in front (−z after the reflection).
    let mut m = surf(1, SurfaceShape::sphere(-200.0), -60.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    let mut sys = infinite_system(20.0, vec![m], 2);
    sys.solves = vec![Solve::marginal_ray_height(id(1), 0.0)];
    let (out, _) = solve(&sys);
    assert!(rel(out.surfaces[1].thickness, -100.0) < 1e-14);
}

// ---------------------------------------------------------------------------
// F/#.

#[test]
fn f_number_solve_reaches_the_requested_f_number() {
    // Singlet: the rear curvature sets EFL = F·EPD.
    let mut sys = singlet();
    sys.solves = vec![Solve::f_number(id(2), 4.0)];
    let (out, report) = solve(&sys);
    let fo = first_order(&out);
    assert!(rel(fo.image_f_number, 4.0) < 1e-12, "{}", fo.image_f_number);
    assert!(rel(fo.efl, 80.0) < 1e-12);
    assert!(report.outcomes[0].residual.abs() < 1e-12 * 4.0);
    // The pupil is fixed, so ω'_A is affine in φ₂ and one update is exact.
    assert_eq!(report.outcomes[0].method, SolveMethod::ClosedForm);
    // Root finding alone agrees with the closed form.
    let (rf, rep) = apply_solves_with(&sys, &lib(), &root_finding_only(), &JobContext::new())
        .expect("root finding");
    assert_eq!(rep.outcomes[0].method, SolveMethod::RootFinding);
    assert!(rel(first_order(&rf).image_f_number, 4.0) < 1e-12);
    assert!(rel(c(2).get(&rf).unwrap(), c(2).get(&out).unwrap()) < 1e-10);

    // Plus a focus solve: F/4 and in focus.
    let mut sys = singlet();
    sys.solves = vec![
        Solve::marginal_ray_height(id(2), 0.0),
        Solve::f_number(id(2), 4.0),
    ];
    let (out, _) = solve(&sys);
    let fo = first_order(&out);
    assert!(rel(fo.image_f_number, 4.0) < 1e-12);
    assert!(fo.marginal.y[3].abs() < 1e-12 * 10.0);
}

#[test]
fn f_number_solve_on_samples_mirror_and_finite_object() {
    // Cooke triplet (stop inside), last surface; relay (finite object);
    // Newtonian primary mirror (sign of the reduced index flips).
    let cases: Vec<(OpticalSystem, u64, f64)> = vec![
        (samples::cooke_triplet(), 6, 3.0),
        (samples::finite_relay(), 7, 6.0),
        (samples::newtonian_telescope(), 1, 4.0),
        (samples::singlet(), 1, 6.0),
    ];
    for (mut sys, sid, f) in cases {
        sys.solves = vec![Solve::f_number(id(sid), f)];
        let (out, _) = solve(&sys);
        let fo = first_order(&out);
        assert!(
            rel(fo.image_f_number, f) < 1e-12,
            "{}: {}",
            sys.name,
            fo.image_f_number
        );
    }
}

#[test]
fn f_number_solve_with_a_pupil_that_depends_on_the_target() {
    // Stop floats by its size and sits behind the first element, so the
    // entrance pupil (and the marginal ray) change with c₁: the closed form
    // has to be iterated.
    let mut sys = samples::cooke_triplet();
    let stop = sys.stop;
    let fo0 = first_order(&sys);
    sys.surfaces[stop].semi_diameter = SemiDiameter::Fixed {
        value: fo0.stop_semi_diameter,
    };
    sys.aperture = ApertureSpec::FloatByStopSize;
    sys.solves = vec![Solve::f_number(id(1), 3.5)];
    let (out, report) = solve(&sys);
    let fo = first_order(&out);
    assert!(rel(fo.image_f_number, 3.5) < 1e-12, "{}", fo.image_f_number);
    assert!(report.outcomes[0].method >= SolveMethod::FixedPoint);
    let (rf, _) =
        apply_solves_with(&sys, &lib(), &root_finding_only(), &JobContext::new()).unwrap();
    assert!(rel(first_order(&rf).image_f_number, 3.5) < 1e-12);
}

#[test]
fn f_number_solve_on_an_ideal_lens_focal_length() {
    let mut sys = infinite_system(
        12.5,
        vec![surf(
            1,
            SurfaceShape::Paraxial { focal_length: 80.0 },
            100.0,
            MaterialRef::Air,
        )],
        2,
    );
    sys.solves = vec![Solve::new(
        ParamPath::ParaxialFocalLength { surface: id(1) },
        SolveKind::FNumber { value: 8.0 },
    )];
    let (out, _) = solve(&sys);
    let f = ParamPath::ParaxialFocalLength { surface: id(1) }
        .get(&out)
        .unwrap();
    assert!(rel(f, 100.0) < 1e-14);
}

// ---------------------------------------------------------------------------
// Ray angles.

#[test]
fn marginal_ray_angle_sets_the_slope_after_the_surface() {
    let mut sys = singlet();
    sys.solves = vec![Solve::new(
        c(2),
        SolveKind::MarginalRayAngle { slope: -0.125 },
    )];
    let (out, report) = solve(&sys);
    assert_eq!(report.outcomes[0].method, SolveMethod::ClosedForm);
    let fo = first_order(&out);
    assert!(rel(fo.marginal.u[2], -0.125) < 1e-12);
    // u' = −y₁/f' for the singlet at infinity: f' = 80.
    assert!(rel(fo.efl, 80.0) < 1e-12);

    // A mirror: slope after the reflection (the reduced index is negative).
    let mut m = surf(1, SurfaceShape::sphere(-200.0), -100.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    let mut sys = infinite_system(20.0, vec![m], 2);
    sys.solves = vec![Solve::new(
        c(1),
        SolveKind::MarginalRayAngle { slope: 0.05 },
    )];
    let (out, _) = solve(&sys);
    // Heights 10 → 0 over −200 mm: u = 0.05 ⇒ focus 200 mm in front, R = −400.
    assert!(rel(c(1).get(&out).unwrap(), -1.0 / 400.0) < 1e-14);
    assert!(rel(first_order(&out).marginal.u[1], 0.05) < 1e-14);
}

#[test]
fn chief_ray_angle_zero_makes_image_space_telecentric() {
    // Stop 40 mm in front of a singlet: telecentric when the stop is at the
    // front focal point.
    let mut sys = infinite_system(
        10.0,
        vec![
            surf(1, SurfaceShape::Plane, 40.0, MaterialRef::Air),
            surf(
                2,
                SurfaceShape::sphere(60.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(3, SurfaceShape::sphere(-350.0), 50.0, MaterialRef::Air),
        ],
        4,
    );
    sys.fields = angle_fields(&[0.0, 5.0]);
    sys.solves = vec![Solve::new(c(3), SolveKind::ChiefRayAngle { slope: 0.0 })];
    let (out, _) = solve(&sys);
    let fo = first_order(&out);
    // Residual within the tolerance on the condition's scale (the largest
    // chief-ray slope, tan 5°).
    assert!(fo.chief.u[3].abs() < 1e-13, "{}", fo.chief.u[3]);
    assert!(fo.image_space_telecentric());
    // The front focal point is on the stop (surface 1, from which FFL is
    // measured).
    assert!(fo.ffl.abs() < 1e-9, "{}", fo.ffl);
}

#[test]
fn chief_ray_height_zero_finds_the_exit_pupil() {
    // A dummy surface after the singlet, placed by the solve on the exit
    // pupil (a virtual one, inside the lens): its distance to the image is
    // then minus the exit-pupil position.
    let mut dummy = surf(3, SurfaceShape::Plane, 30.0, MaterialRef::Air);
    dummy.interaction = Interaction::Dummy;
    let mut sys = infinite_system(
        20.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(60.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(2, SurfaceShape::sphere(-350.0), 50.0, MaterialRef::Air),
            dummy,
        ],
        4,
    );
    sys.fields = angle_fields(&[0.0, 5.0]);
    sys.solves = vec![Solve::new(t(2), SolveKind::ChiefRayHeight { height: 0.0 })];
    let (out, _) = solve(&sys);
    let fo = first_order(&out);
    assert!(fo.chief.y[3].abs() < 1e-12);
    assert!(out.surfaces[2].thickness < 0.0);
    assert!(
        rel(fo.exit_pupil_position, -30.0) < 1e-10,
        "{}",
        fo.exit_pupil_position
    );
}

// ---------------------------------------------------------------------------
// Pickups.

#[test]
fn curvature_pickup_with_scale_minus_one_makes_a_symmetric_lens() {
    let mut sys = singlet();
    sys.solves = vec![Solve::pickup(c(2), c(1), -1.0, 0.0)];
    let (out, report) = solve(&sys);
    assert_eq!(c(2).get(&out).unwrap(), -1.0 / 60.0);
    assert_eq!(report.outcomes[0].method, SolveMethod::ClosedForm);
    // Symmetric: the reversed lens has the same EFL.
    let mut rev = out.clone();
    rev.surfaces[1].shape = SurfaceShape::Conic {
        curvature: -c(2).get(&out).unwrap(),
        conic: 0.0,
    };
    rev.surfaces[2].shape = SurfaceShape::Conic {
        curvature: -c(1).get(&out).unwrap(),
        conic: 0.0,
    };
    assert!(rel(first_order(&rev).efl, first_order(&out).efl) < 1e-15);

    // Coupled: F/# by c₁ with c₂ = −c₁ (an equi-convex lens of given F/#).
    // The pickup reads the F/# target, so the two are solved as one problem.
    let mut sys = singlet();
    sys.solves = vec![
        Solve::pickup(c(2), c(1), -1.0, 0.0),
        Solve::f_number(id(1), 5.0),
    ];
    for settings in [SolveSettings::default(), root_finding_only()] {
        let (out, _) = apply_solves_with(&sys, &lib(), &settings, &JobContext::new()).unwrap();
        let fo = first_order(&out);
        assert!(rel(fo.image_f_number, 5.0) < 1e-12, "{}", fo.image_f_number);
        assert_eq!(c(2).get(&out).unwrap(), -c(1).get(&out).unwrap());
    }
}

#[test]
fn chained_pickups_resolve_in_dependency_order() {
    // Doublet surfaces 1..3: c₃ ← c₂ ← c₁ and t₂ ← t₁, declared backwards,
    // with offsets; plus the image thickness ← a focus solve read by a pickup.
    let mut sys = infinite_system(
        20.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(
                2,
                SurfaceShape::sphere(-40.0),
                2.0,
                MaterialRef::glass("F2"),
            ),
            surf(3, SurfaceShape::sphere(-300.0), 90.0, MaterialRef::Air),
            surf(4, SurfaceShape::Plane, 7.0, MaterialRef::Air),
        ],
        5,
    );
    sys.solves = vec![
        Solve::pickup(t(4), t(3), 0.5, 1.0),
        Solve::pickup(c(3), c(2), 0.25, -0.001),
        Solve::pickup(c(2), c(1), -1.5, 0.0),
        Solve::pickup(t(2), t(1), 0.5, 0.0),
        Solve::marginal_ray_height(id(3), 0.0),
    ];
    let (out, report) = solve(&sys);
    let c1 = 1.0 / 50.0;
    let c2 = -1.5 * c1;
    let c3 = 0.25f64.mul_add(c2, -0.001);
    assert_eq!(c(2).get(&out).unwrap(), c2);
    assert_eq!(c(3).get(&out).unwrap(), c3);
    assert_eq!(t(2).get(&out).unwrap(), 2.5);
    // The focus solve sees the picked-up shape, and its pickup sees it.
    let fo = first_order(&out);
    assert!(fo.marginal.y[4].abs() < 1e-12 * 10.0);
    let t3 = t(3).get(&out).unwrap();
    assert_eq!(t(4).get(&out).unwrap(), 0.5f64.mul_add(t3, 1.0));
    // Pickups ran after their sources.
    let pos = |k: usize| report.order.iter().position(|&x| x == k).unwrap();
    assert!(pos(2) < pos(1));
    assert!(pos(4) < pos(0));
    assert!(pos(3) < pos(4)); // t₂ is upstream of the focus thickness
    assert_eq!(
        solved_params(&sys),
        vec![c(2), c(3), t(2), t(3), t(4)],
        "sorted solve-controlled parameters"
    );
}

#[test]
fn pickup_cycles_are_rejected() {
    let mut sys = singlet();
    sys.solves = vec![
        Solve::pickup(c(1), c(2), 1.0, 0.0),
        Solve::pickup(t(1), c(1), 1.0, 0.0),
        Solve::pickup(c(2), t(1), 1.0, 0.0),
    ];
    let e = sys.validate().unwrap_err().to_string();
    assert!(e.contains("cyclic"), "{e}");
    assert!(matches!(
        apply_solves(&sys, &lib()),
        Err(MkError::InvalidModel(_))
    ));
    // Breaking the cycle makes it valid.
    sys.solves.pop();
    sys.validate().unwrap();
}

#[test]
fn pickups_of_other_parameter_kinds() {
    let mut sys = singlet();
    sys.surfaces[1].shape = SurfaceShape::EvenAsphere {
        curvature: 1.0 / 60.0,
        conic: -0.5,
        coefficients: vec![1e-6],
    };
    sys.surfaces[2].shape = SurfaceShape::EvenAsphere {
        curvature: -1.0 / 350.0,
        conic: 0.0,
        coefficients: vec![],
    };
    sys.solves = vec![
        Solve::pickup(
            ParamPath::Conic { surface: id(2) },
            ParamPath::Conic { surface: id(1) },
            1.0,
            0.0,
        ),
        Solve::pickup(
            ParamPath::Asphere {
                surface: id(2),
                index: 1,
            },
            ParamPath::Asphere {
                surface: id(1),
                index: 0,
            },
            -2.0,
            0.0,
        ),
        Solve::pickup(
            ParamPath::FieldY { index: 1 },
            ParamPath::ApertureValue,
            0.25,
            0.0,
        ),
    ];
    let (out, _) = solve(&sys);
    assert_eq!(out.surfaces[2].shape.conic(), -0.5);
    match &out.surfaces[2].shape {
        SurfaceShape::EvenAsphere { coefficients, .. } => assert_eq!(coefficients, &[0.0, -2e-6]),
        s => panic!("{s:?}"),
    }
    assert_eq!(out.fields.fields[1].y, 5.0);
}

// ---------------------------------------------------------------------------
// Geometric solves.

/// Sag of a sphere of radius `r_c` at height `h`, from the circle equation.
fn sphere_sag(r_c: f64, h: f64) -> f64 {
    r_c - r_c.signum() * (r_c * r_c - h * h).sqrt()
}

#[test]
fn edge_thickness_from_the_sags() {
    let mut sys = singlet();
    sys.solves = vec![Solve::new(
        t(1),
        SolveKind::EdgeThickness {
            value: 1.5,
            radius: Some(11.0),
        },
    )];
    let (out, _) = solve(&sys);
    let t1 = out.surfaces[1].thickness;
    let edge = t1 + sphere_sag(-350.0, 11.0) - sphere_sag(60.0, 11.0);
    assert!((edge - 1.5).abs() < 1e-13, "{edge}");

    // Default radius: the larger fixed semi-diameter.
    let mut sys = singlet();
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 12.0 };
    sys.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 11.0 };
    sys.solves = vec![Solve::new(
        t(1),
        SolveKind::EdgeThickness {
            value: 1.0,
            radius: None,
        },
    )];
    let (out, _) = solve(&sys);
    let edge = out.surfaces[1].thickness + sphere_sag(-350.0, 12.0) - sphere_sag(60.0, 12.0);
    assert!((edge - 1.0).abs() < 1e-13, "{edge}");

    // Automatic semi-diameters depend on the thickness itself (the rear
    // surface's beam size): the result must be self-consistent.
    let mut sys = singlet();
    sys.solves = vec![
        Solve::new(
            t(1),
            SolveKind::EdgeThickness {
                value: 1.0,
                radius: None,
            },
        ),
        Solve::marginal_ray_height(id(2), 0.0),
    ];
    let (out, _) = solve(&sys);
    let tracer = SequentialTracer::new(Arc::new(resolve(out.clone()))).unwrap();
    let sd = tracer
        .semi_diameters(&SemiDiameterSettings::default())
        .values;
    let r = sd[1].max(sd[2]);
    let edge = out.surfaces[1].thickness + sphere_sag(-350.0, r) - sphere_sag(60.0, r);
    assert!((edge - 1.0).abs() < 1e-11, "{edge} at r = {r}");
    assert!(first_order(&out).marginal.y[3].abs() < 1e-11);
}

#[test]
fn position_solve_keeps_the_total_track() {
    let base = samples::cooke_triplet();
    let img = base.image_index();
    let track0 = first_order(&base).total_track;
    let last = last_thickness_id(&base);
    let mut sys = base.clone();
    sys.solves = vec![Solve::new(
        ParamPath::Thickness { surface: last },
        SolveKind::Position {
            from: base.surfaces[1].id,
            to: base.surfaces[img].id,
            distance: track0,
        },
    )];
    // A middle air space grows by 1.3 mm; the back focal space gives it back.
    sys.surfaces[3].thickness += 1.3;
    let (out, _) = solve(&sys);
    let track = first_order(&out).total_track;
    assert!(
        (track - track0).abs() < 1e-12 * track0,
        "{track} vs {track0}"
    );
    assert!(
        (out.surfaces[img - 1].thickness - (base.surfaces[img - 1].thickness - 1.3)).abs() < 1e-12
    );

    // Distance from a reference surface (3) to the image, and a target
    // that is not the last thickness.
    let mut sys = base.clone();
    let d = 30.0;
    sys.solves = vec![Solve::new(
        t(4),
        SolveKind::Position {
            from: id(3),
            to: base.surfaces[img].id,
            distance: d,
        },
    )];
    let (out, _) = solve(&sys);
    let sum: f64 = (3..img).map(|i| out.surfaces[i].thickness).sum();
    assert!((sum - d).abs() < 1e-12);
}

#[test]
fn centre_of_curvature_on_another_vertex() {
    // Doublet: surface 1's centre on vertex 3 (behind it), surface 3's on
    // vertex 1 (in front of it): concentric outer surfaces.
    let mut sys = infinite_system(
        20.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(
                2,
                SurfaceShape::sphere(-40.0),
                2.5,
                MaterialRef::glass("F2"),
            ),
            surf(3, SurfaceShape::sphere(-300.0), 90.0, MaterialRef::Air),
        ],
        4,
    );
    sys.solves = vec![
        Solve::new(c(1), SolveKind::CenterOfCurvature { surface: id(3) }),
        Solve::new(c(3), SolveKind::CenterOfCurvature { surface: id(1) }),
    ];
    let (out, _) = solve(&sys);
    assert!(rel(c(1).get(&out).unwrap(), 1.0 / 7.5) < 1e-15);
    assert!(rel(c(3).get(&out).unwrap(), -1.0 / 7.5) < 1e-15);
}

#[test]
fn mirror_at_its_centre_of_curvature_images_the_object_onto_itself() {
    // A finite object 150 mm in front of a mirror whose centre of
    // curvature is solved onto the object: the paraxial image (solved by a
    // marginal-height solve) is back on the object, at magnification −1.
    let mut sys = OpticalSystem::new_empty();
    sys.wavelengths = mono();
    sys.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 };
    sys.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 2.0)],
    };
    let mut m = surf(1, SurfaceShape::sphere(-120.0), -40.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    sys.surfaces = vec![
        surf(0, SurfaceShape::Plane, 150.0, MaterialRef::Air),
        m,
        surf(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    sys.stop = 1;
    sys.solves = vec![
        Solve::marginal_ray_height(id(1), 0.0),
        Solve::new(c(1), SolveKind::CenterOfCurvature { surface: id(0) }),
    ];
    let (out, report) = solve(&sys);
    assert!(rel(c(1).get(&out).unwrap(), -1.0 / 150.0) < 1e-15);
    assert!(rel(out.surfaces[1].thickness, -150.0) < 1e-12);
    assert!(rel(first_order(&out).magnification, -1.0) < 1e-12);
    // The curvature (upstream) is solved before the focus.
    assert_eq!(report.order, vec![1, 0]);
}

// ---------------------------------------------------------------------------
// Configurations.

#[test]
fn solves_apply_per_configuration_after_overrides() {
    // Focus solve; configuration 1 refocuses for an object at 1 m, and
    // configuration 2 tries to override the solved thickness.
    let mut sys = singlet();
    sys.solves = vec![Solve::marginal_ray_height(id(2), 0.0)];
    sys.configurations = vec![
        Configuration::default(),
        Configuration {
            name: "1 m".into(),
            overrides: vec![(t(0), 1000.0)],
        },
        Configuration {
            name: "override".into(),
            overrides: vec![(t(2), 50.0), (c(1), 1.0 / 55.0)],
        },
    ];
    let all = apply_solves_all(&sys, &lib(), &SolveSettings::default(), &JobContext::new())
        .expect("all configurations");
    assert_eq!(all.len(), 3);
    for (k, (out, report)) in all.iter().enumerate() {
        let cfg = sys.with_configuration(k).unwrap();
        let fo = first_order(&cfg);
        assert!(
            rel(out.surfaces[2].thickness, fo.image_distance) < 1e-12,
            "configuration {k}"
        );
        assert_eq!(report.warnings.is_empty(), k != 2, "configuration {k}");
    }
    assert!(all[1].0.surfaces[2].thickness > all[0].0.surfaces[2].thickness);
    assert!(all[2].1.warnings[0].contains("solve wins"));
    // One configuration on its own gives the same result.
    let (one, _) = apply_solves_configuration(
        &sys,
        &lib(),
        1,
        &SolveSettings::default(),
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(one, all[1].0);
}

// ---------------------------------------------------------------------------
// Serde, determinism, jobs.

fn every_kind() -> Vec<Solve> {
    vec![
        Solve::marginal_ray_height(id(2), 0.0),
        Solve::new(c(2), SolveKind::MarginalRayAngle { slope: -0.1 }),
        Solve::new(t(1), SolveKind::ChiefRayHeight { height: 1.0 }),
        Solve::new(c(1), SolveKind::ChiefRayAngle { slope: 0.0 }),
        Solve::new(
            ParamPath::Conic { surface: id(1) },
            SolveKind::Pickup {
                source: ParamPath::Conic { surface: id(2) },
                scale: 2.0,
                offset: -1.0,
            },
        ),
        Solve::new(
            t(0),
            SolveKind::Position {
                from: id(1),
                to: id(3),
                distance: 100.0,
            },
        ),
        Solve::new(
            ParamPath::DecenterX { surface: id(1) },
            SolveKind::CenterOfCurvature { surface: id(3) },
        ),
        Solve::new(
            t(1),
            SolveKind::EdgeThickness {
                value: 2.0,
                radius: Some(8.0),
            },
        ),
        Solve::f_number(id(2), 4.0),
    ]
}

#[test]
fn serde_round_trip_and_documented_json() {
    // Every kind round-trips (validity is not needed for serde).
    let mut sys = singlet();
    sys.solves = every_kind();
    let json = serde_json::to_string_pretty(&sys).unwrap();
    let back: OpticalSystem = serde_json::from_str(&json).unwrap();
    assert_eq!(back, sys);
    // A valid solved system has the same snapshot hash after a round trip.
    let mut sys = singlet();
    sys.solves = vec![
        Solve::marginal_ray_height(id(2), 0.0),
        Solve::pickup(c(2), c(1), -1.0, 0.0),
    ];
    let back: OpticalSystem = serde_json::from_str(&serde_json::to_string(&sys).unwrap()).unwrap();
    assert_eq!(
        Snapshot::new(back).unwrap().hash(),
        Snapshot::new(sys.clone()).unwrap().hash()
    );
    // Systems written before solves existed still load.
    let mut v = serde_json::to_value(OpticalSystem::new_empty()).unwrap();
    v.as_object_mut().unwrap().remove("solves");
    let old: OpticalSystem = serde_json::from_value(v).unwrap();
    assert!(old.solves.is_empty());

    // The JSON forms of docs/SOLVES.md.
    let doc = r#"[
      {"target": {"param": "thickness", "surface": 2},
       "kind": {"type": "marginal_ray_height", "height": 0.0}},
      {"target": {"param": "curvature", "surface": 2},
       "kind": {"type": "marginal_ray_angle", "slope": -0.1}},
      {"target": {"param": "thickness", "surface": 1},
       "kind": {"type": "chief_ray_height", "height": 1.0}},
      {"target": {"param": "curvature", "surface": 1},
       "kind": {"type": "chief_ray_angle", "slope": 0.0}},
      {"target": {"param": "conic", "surface": 1},
       "kind": {"type": "pickup", "source": {"param": "conic", "surface": 2},
                "scale": 2.0, "offset": -1.0}},
      {"target": {"param": "thickness", "surface": 0},
       "kind": {"type": "position", "from": 1, "to": 3, "distance": 100.0}},
      {"target": {"param": "decenter_x", "surface": 1},
       "kind": {"type": "center_of_curvature", "surface": 3}},
      {"target": {"param": "thickness", "surface": 1},
       "kind": {"type": "edge_thickness", "value": 2.0, "radius": 8.0}},
      {"target": {"param": "curvature", "surface": 2},
       "kind": {"type": "f_number", "value": 4.0}}
    ]"#;
    let parsed: Vec<Solve> = serde_json::from_str(doc).unwrap();
    assert_eq!(parsed, every_kind());
    // Defaults: scale 1, offset 0, radius from the semi-diameters.
    let p: Solve = serde_json::from_str(
        r#"{"target": {"param": "curvature", "surface": 2},
            "kind": {"type": "pickup", "source": {"param": "curvature", "surface": 1}}}"#,
    )
    .unwrap();
    assert_eq!(p, Solve::pickup(c(2), c(1), 1.0, 0.0));
    let e: Solve = serde_json::from_str(
        r#"{"target": {"param": "thickness", "surface": 1},
            "kind": {"type": "edge_thickness", "value": 2.0}}"#,
    )
    .unwrap();
    assert_eq!(
        e.kind,
        SolveKind::EdgeThickness {
            value: 2.0,
            radius: None
        }
    );
}

#[test]
fn deterministic_cancellable_with_monotone_progress_and_run_info() {
    let mut sys = samples::double_gauss();
    sys.solves = vec![
        Solve::marginal_ray_height(last_thickness_id(&sys), 0.0),
        Solve::pickup(c(10), c(1), 1.0, 0.0),
        Solve::new(c(11), SolveKind::MarginalRayAngle { slope: -0.24 }),
    ];
    let (a, ra) = solve(&sys);
    let (b, rb) = solve(&sys);
    assert_eq!(a, b);
    assert_eq!(ra.outcomes, rb.outcomes);
    assert_eq!(
        ra.run_info.snapshot_hash,
        Snapshot::new(sys.clone()).unwrap().hash().to_hex()
    );
    assert_eq!(ra.run_info.analysis, "solves");

    let progress = Progress::default();
    let ctx = JobContext::with(CancelToken::new(), progress.clone());
    apply_solves_with(&sys, &lib(), &SolveSettings::default(), &ctx).unwrap();
    assert_eq!(progress.fraction(), 1.0);

    let cancel = CancelToken::new();
    cancel.cancel();
    let ctx = JobContext::with(cancel, Progress::default());
    assert!(matches!(
        apply_solves_with(&sys, &lib(), &SolveSettings::default(), &ctx),
        Err(MkError::Cancelled)
    ));
}

// ---------------------------------------------------------------------------
// Validation and failures.

fn invalid(sys: &OpticalSystem, needle: &str) {
    let e = sys.validate().expect_err(needle).to_string();
    assert!(e.contains(needle), "expected {needle:?} in {e:?}");
    assert!(apply_solves(sys, &lib()).is_err());
}

#[test]
fn nonsensical_targets_are_rejected() {
    let with = |s: Solve| {
        let mut sys = singlet();
        sys.solves = vec![s];
        sys
    };
    invalid(
        &with(Solve::new(
            c(1),
            SolveKind::MarginalRayHeight { height: 0.0 },
        )),
        "must be a thickness",
    );
    invalid(
        &with(Solve::marginal_ray_height(id(3), 0.0)),
        "between the object and the image",
    );
    invalid(
        &with(Solve::marginal_ray_height(id(9), 0.0)),
        "no surface with id 9",
    );
    invalid(
        &with(Solve::new(t(1), SolveKind::FNumber { value: 4.0 })),
        "curvature",
    );
    invalid(&with(Solve::f_number(id(1), -2.0)), "> 0");
    invalid(&with(Solve::marginal_ray_height(id(2), f64::NAN)), "finite");
    invalid(&with(Solve::f_number(id(3), 4.0)), "no power");
    invalid(
        &with(Solve::pickup(
            c(1),
            ParamPath::ModelNd { surface: id(1) },
            1.0,
            0.0,
        )),
        "source",
    );
    invalid(
        &with(Solve::new(
            t(2),
            SolveKind::Position {
                from: id(1),
                to: id(2),
                distance: 5.0,
            },
        )),
        "between `from` and `to`",
    );
    invalid(
        &with(Solve::new(
            t(1),
            SolveKind::Position {
                from: id(0),
                to: id(3),
                distance: 5.0,
            },
        )),
        "infinity",
    );
    invalid(
        &with(Solve::new(
            c(2),
            SolveKind::CenterOfCurvature { surface: id(2) },
        )),
        "own vertex",
    );
    invalid(
        &with(Solve::new(
            t(1),
            SolveKind::EdgeThickness {
                value: 1.0,
                radius: Some(-1.0),
            },
        )),
        "radius",
    );
    // An F/# solve while the aperture is an F/#.
    let mut sys = with(Solve::f_number(id(2), 4.0));
    sys.aperture = ApertureSpec::ImageSpaceFNumber { value: 5.0 };
    invalid(&sys, "F/#");
    // Two solves on one parameter.
    let mut sys = with(Solve::f_number(id(2), 4.0));
    sys.solves.push(Solve::pickup(c(2), c(1), -1.0, 0.0));
    invalid(&sys, "already controlled");
    // Curvature of an ideal lens, and of a dummy.
    let mut sys = with(Solve::f_number(id(2), 4.0));
    sys.surfaces[2].shape = SurfaceShape::Paraxial { focal_length: 90.0 };
    invalid(&sys, "no curvature");
    let mut sys = with(Solve::f_number(id(2), 4.0));
    sys.surfaces[2].interaction = Interaction::Dummy;
    invalid(&sys, "dummy");
}

#[test]
fn curvature_solve_on_a_plane_turns_it_into_a_sphere() {
    // Documented behaviour: a plane has curvature 0, and solving it makes
    // it a sphere (conic 0).
    let mut sys = singlet();
    sys.surfaces[2].shape = SurfaceShape::Plane;
    sys.solves = vec![Solve::f_number(id(2), 4.0)];
    sys.validate().unwrap();
    let (out, _) = solve(&sys);
    assert!(matches!(
        out.surfaces[2].shape,
        SurfaceShape::Conic { conic, .. } if conic == 0.0
    ));
    assert!(rel(first_order(&out).image_f_number, 4.0) < 1e-12);
}

#[test]
fn impossible_conditions_fail_with_a_clear_error() {
    // Behind a plane stop the axial beam of an object at infinity is
    // parallel: no thickness changes its height.
    let mut sys = infinite_system(
        10.0,
        vec![
            surf(1, SurfaceShape::Plane, 10.0, MaterialRef::Air),
            surf(
                2,
                SurfaceShape::sphere(60.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(3, SurfaceShape::sphere(-350.0), 90.0, MaterialRef::Air),
        ],
        4,
    );
    sys.solves = vec![Solve::marginal_ray_height(id(1), 1.0)];
    let e = apply_solves(&sys, &lib()).unwrap_err().to_string();
    assert!(e.contains("solve 0 (marginal_ray_height"), "{e}");
    assert!(e.contains("parallel"), "{e}");

    // With an image-space F/# aperture the image-space marginal slope is
    // fixed at −1/(2F): a marginal-angle solve can only fail.
    let mut sys = singlet();
    sys.aperture = ApertureSpec::ImageSpaceFNumber { value: 5.0 };
    sys.solves = vec![Solve::new(
        c(2),
        SolveKind::MarginalRayAngle { slope: -0.2 },
    )];
    let e = apply_solves(&sys, &lib()).unwrap_err().to_string();
    assert!(e.contains("marginal_ray_angle"), "{e}");
    assert!(e.contains("no sign change"), "{e}");
}

#[test]
fn solve_report_lists_methods_previous_values_and_converges_coupled_sets() {
    // Coupled through implicit dependencies: a curvature upstream of a
    // focus thickness picks up that thickness (a cycle through the paraxial
    // trace, which is allowed and iterated to a fixed point).
    let mut sys = singlet();
    sys.solves = vec![
        Solve::marginal_ray_height(id(2), 0.0),
        Solve::pickup(c(1), t(2), 1.0 / 5000.0, 0.0),
    ];
    let (out, report) = solve(&sys);
    let t2 = out.surfaces[2].thickness;
    assert!((c(1).get(&out).unwrap() - t2 / 5000.0).abs() < 1e-14);
    assert!(first_order(&out).marginal.y[3].abs() < 1e-11);
    assert!(report.sweeps > 1);
    assert!(report.max_change <= SolveSettings::default().tolerance);
    assert_eq!(report.outcomes[0].previous, Some(90.0));
    assert_eq!(report.outcomes[1].previous, Some(1.0 / 60.0));
    assert_eq!(report.outcomes[1].value, c(1).get(&out).unwrap());
    // The solved system is consistent: solving it again changes nothing.
    let (again, rep2) = solve(&out);
    assert!(rel(again.surfaces[2].thickness, t2) < 1e-12);
    assert!(
        rep2.outcomes
            .iter()
            .all(|o| o.method == SolveMethod::Unchanged)
    );
}
