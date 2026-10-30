//! Regression tests of the tracer-backed evaluator: colour-operand ranges,
//! half pupils on asymmetric candidates, relative illumination by energy
//! conservation (real, vignetted and finite-conjugate lenses), the Petzval
//! sum of ideal lenses, edge thicknesses with automatic semi-diameters and
//! the `RunInfo` of runs.

use mklight_core::model::{
    ApertureSpec, Field, FieldKind, FieldSet, MaterialRef, OpticalSystem, SemiDiameter, Surface,
    SurfaceShape,
};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, SurfaceId, samples};
use mklight_optim::Monitor;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultMeritSpec, ThicknessBoundaries};
use mklight_optim::merit::operand::*;
use mklight_optim::merit::{
    FieldSelector, MeritFunctionSpec, Operand, OperandError, OperandEvaluator, OperandRegistry,
    OperandValue, OpticalProblem, PupilSampling, PupilSymmetry, TraceEvalSettings, TraceEvaluator,
    VariableSpec,
};
use mklight_optim::problem::{EvalStatus, Problem, evaluate_one, merit};
use mklight_optim::tolerance::{
    OpticalToleranceEvaluator, ToleranceOperand, ToleranceSpec, sensitivity,
};

fn eval_with(ev: &TraceEvaluator, sys: &OpticalSystem, ops: &[Operand]) -> Vec<OperandValue> {
    let refs: Vec<&Operand> = ops.iter().collect();
    ev.evaluate_batch(std::slice::from_ref(sys), &refs, &JobContext::new())
        .unwrap()
        .remove(0)
}

fn eval(sys: &OpticalSystem, ops: &[Operand]) -> Vec<OperandValue> {
    eval_with(&TraceEvaluator::builtin(), sys, ops)
}

fn ok(v: &OperandValue) -> f64 {
    v.clone()
        .unwrap_or_else(|e| panic!("operand failed: {e:?}"))
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

// ---------------------------------------------------------------------------
// Colour operands
// ---------------------------------------------------------------------------

#[test]
fn colour_wavelengths_out_of_range_are_invalid_not_a_panic() {
    let sys = samples::cooke_triplet();
    let lat = OperandKind::LateralColour { short: 0, long: 9 };
    let sens = OperandKind::ToleranceSensitivity {
        param: ParamPath::Thickness {
            surface: SurfaceId(2),
        },
        delta: 0.1,
        criterion: Box::new(lat.clone()),
    };
    let v = eval(
        &sys,
        &[
            Operand::new(lat.clone()).at_field(2),
            Operand::new(sens.clone()).at_field(2),
            Operand::new(OperandKind::LateralColour { short: 0, long: 2 }).at_field(2),
        ],
    );
    assert!(matches!(v[0], Err(OperandError::Invalid(_))), "{:?}", v[0]);
    assert!(matches!(v[1], Err(OperandError::Invalid(_))), "{:?}", v[1]);
    assert!(
        ok(&v[2]).abs() > 1e-6,
        "valid lateral colour still computed"
    );
    // Validation catches it too, also inside a sensitivity criterion.
    let spec = MeritFunctionSpec::new(vec![Operand::new(sens).at_field(2)]);
    assert!(spec.validate_for(&sys).is_err());
    let p = OpticalProblem::new(
        sys,
        vec![VariableSpec::new(ParamPath::Curvature {
            surface: SurfaceId(1),
        })],
        spec,
        TraceEvaluator::builtin(),
        &OperandRegistry::builtin(),
    );
    assert!(p.is_err());
    // The registry checks the criterion kind of a sensitivity.
    let reg = OperandRegistry::builtin();
    let unknown = OperandKind::ToleranceSensitivity {
        param: ParamPath::Thickness {
            surface: SurfaceId(2),
        },
        delta: 0.1,
        criterion: Box::new(OperandKind::Custom {
            name: "not_registered".into(),
            params: Default::default(),
        }),
    };
    assert!(!reg.knows(&unknown));
    assert!(reg.knows(&OperandKind::ToleranceSensitivity {
        param: ParamPath::Thickness {
            surface: SurfaceId(2),
        },
        delta: 0.1,
        criterion: Box::new(OperandKind::Efl),
    }));
}

// ---------------------------------------------------------------------------
// Half pupil on asymmetric candidates
// ---------------------------------------------------------------------------

/// Full-pupil sampling whose nodes are exactly the mirrored half pupil.
fn mirrored(half: &PupilSampling) -> PupilSampling {
    PupilSampling {
        rings: half.rings,
        arms: 2 * half.arms,
        symmetry: PupilSymmetry::None,
        obscuration: half.obscuration,
        arm_offset_deg: -90.0 + 90.0 / half.arms as f64,
    }
}

/// `Σ w (v − target)²` of a spec on one system.
fn phi(ev: &TraceEvaluator, spec: &MeritFunctionSpec, s: &OpticalSystem) -> f64 {
    let refs: Vec<&Operand> = spec.operands.iter().collect();
    let v = ev.evaluate_system(s, &refs, &JobContext::new()).unwrap();
    spec.operands
        .iter()
        .zip(&v)
        .map(|(o, r)| o.weight * (ok(r) - o.target).powi(2))
        .sum()
}

fn perturbed(sys: &OpticalSystem, p: &ParamPath, d: f64) -> OpticalSystem {
    let mut s = sys.clone();
    let v = p.get(&s).unwrap();
    p.set(&mut s, v + d).unwrap();
    s
}

#[test]
fn half_pupil_merit_on_asymmetric_candidates_equals_the_full_pupil() {
    let sys = samples::cooke_triplet();
    let ev = TraceEvaluator::builtin();
    let half = PupilSampling::bilateral(3, 6);
    for criterion in [
        mklight_optim::merit::default::DefaultCriterion::default(),
        mklight_optim::merit::default::DefaultCriterion::RmsSpot {
            reference: SpotReference::Chief,
            scope: CentroidScope::Monochromatic,
        },
        mklight_optim::merit::default::DefaultCriterion::RmsSpot {
            reference: SpotReference::Centroid,
            scope: CentroidScope::Polychromatic,
        },
        mklight_optim::merit::default::DefaultCriterion::Wavefront {
            reference: WavefrontReference::Centroid,
        },
    ] {
        let gen_half = DefaultMeritSpec {
            criterion,
            sampling: half,
            thickness: None,
            ..Default::default()
        };
        let spec_half = gen_half.generate(&sys).unwrap();
        let spec_full = DefaultMeritSpec {
            sampling: mirrored(&half),
            auto_symmetry: false,
            ..gen_half.clone()
        }
        .generate(&sys)
        .unwrap();
        assert!(matches!(
            spec_half.operands[0].kind.ray_group(),
            Some(g) if g.sampling.symmetry == PupilSymmetry::Bilateral
        ));
        let s3 = SurfaceId(3);
        for (p, d) in [
            (ParamPath::DecenterX { surface: s3 }, 0.05),
            (ParamPath::TiltY { surface: s3 }, 0.2),
            (ParamPath::TiltZ { surface: s3 }, 1.0),
            (ParamPath::DecenterY { surface: s3 }, 0.05),
        ] {
            let s = perturbed(&sys, &p, d);
            let (a, b) = (phi(&ev, &spec_half, &s), phi(&ev, &spec_full, &s));
            assert!(rel(a, b) < 1e-10, "{criterion:?} {p:?}: half {a} full {b}");
        }
        // Nominal: the symmetric half pupil equals the full one as before.
        let (a, b) = (phi(&ev, &spec_half, &sys), phi(&ev, &spec_full, &sys));
        assert!(rel(a, b) < 1e-10, "{criterion:?} nominal: {a} {b}");
    }
    // The aggregate operands follow the same rule (field off the y axis).
    let spot = |s: PupilSampling, field: FieldSelector| {
        let mut o = Operand::new(OperandKind::RmsSpot {
            reference: SpotReference::Centroid,
            component: Some(Axis::X),
            sampling: s,
        });
        o.field = field;
        o
    };
    let off = FieldSelector::Normalized { hx: 0.6, hy: 0.5 };
    let v = eval(&sys, &[spot(half, off), spot(mirrored(&half), off)]);
    assert!(rel(ok(&v[0]), ok(&v[1])) < 1e-10, "{v:?}");
    let dec = perturbed(
        &sys,
        &ParamPath::DecenterX {
            surface: SurfaceId(3),
        },
        0.05,
    );
    let v = eval(
        &dec,
        &[
            spot(half, FieldSelector::Index { index: 2 }),
            spot(mirrored(&half), FieldSelector::Index { index: 2 }),
        ],
    );
    assert!(rel(ok(&v[0]), ok(&v[1])) < 1e-10, "{v:?}");
}

#[test]
fn tolerancing_with_asymmetric_tolerances() {
    let sys = samples::cooke_triplet();
    let dx = ParamPath::DecenterX {
        surface: SurfaceId(3),
    };
    let ty = ParamPath::TiltY {
        surface: SurfaceId(3),
    };
    let spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(dx.clone(), 0.05),
        ToleranceOperand::symmetric(ty.clone(), 0.2),
        ToleranceOperand::symmetric(
            ParamPath::DecenterY {
                surface: SurfaceId(3),
            },
            0.05,
        ),
    ]);
    let gen_ = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    };
    // The tolerancing generator sees that x decenters / y tilts break the
    // meridional symmetry and uses the full pupil.
    let crit = gen_.generate_for_tolerancing(&sys, &spec).unwrap();
    assert!(
        crit.operands
            .iter()
            .all(|o| o.kind.ray_group().unwrap().sampling.symmetry == PupilSymmetry::None)
    );
    let only_y = ToleranceSpec::new(vec![spec.operands[2].clone()]);
    let crit_y = gen_.generate_for_tolerancing(&sys, &only_y).unwrap();
    assert!(
        crit_y
            .operands
            .iter()
            .all(|o| o.kind.ray_group().unwrap().sampling.symmetry == PupilSymmetry::Bilateral)
    );
    // A half-pupil criterion still gives the same sensitivities as its
    // mirrored full pupil (it was 18–39× too large for x decenter / y tilt).
    let half = gen_.generate(&sys).unwrap();
    let full = DefaultMeritSpec {
        sampling: mirrored(&PupilSampling {
            symmetry: PupilSymmetry::Bilateral,
            ..gen_.sampling
        }),
        auto_symmetry: false,
        ..gen_.clone()
    }
    .generate(&sys)
    .unwrap();
    let reg = OperandRegistry::builtin();
    let ev = TraceEvaluator::builtin();
    let run = |criterion: &MeritFunctionSpec| {
        let te = OpticalToleranceEvaluator {
            nominal: &sys,
            spec: &spec,
            criterion,
            evaluator: &ev,
            registry: &reg,
            lsq: LsqSettings::default(),
        };
        sensitivity(&spec, &te, &Monitor::default()).unwrap()
    };
    let (a, b) = (run(&half), run(&full));
    for (x, y) in a.operands.iter().zip(&b.operands) {
        assert!(
            rel(x.plus.criterion, y.plus.criterion) < 1e-9,
            "{x:?} {y:?}"
        );
        assert!(rel(x.minus.criterion, y.minus.criterion) < 1e-9);
    }
    // Decentering is a second-order effect on the RMS spot here (a few %),
    // not the 18× of a half pupil forced to x = 0.
    let n = a.nominal.criterion;
    assert!(a.operands[0].plus.criterion < 1.2 * n, "{a:?}");
}

// ---------------------------------------------------------------------------
// Relative illumination
// ---------------------------------------------------------------------------

fn ri_at(ev: &TraceEvaluator, sys: &OpticalSystem, field: FieldSelector) -> OperandValue {
    let mut o = Operand::new(OperandKind::RelativeIllumination);
    o.field = field;
    eval_with(ev, sys, &[o]).remove(0)
}

/// Singlet with its stop (and entrance pupil) at the first surface and no
/// vignetting: the transmitted beam is the entrance pupil seen at angle θ,
/// `G = A cos θ`, and the image mapping of a rotationally symmetric lens
/// gives `dΩ/dA' = sin θ / (y' dy'/dθ)` — so `RI = cos θ · y'_θ(0)² ·
/// sin θ / (y' y'_θ)`, from real chief-ray heights. This lens is not
/// aplanatic: the old image-space solid-angle ratio gave 1.005 at 5°.
#[test]
fn relative_illumination_conserves_energy_on_a_real_lens() {
    let sys = samples::singlet();
    let fmax = sys.fields.max_radial();
    let ev = TraceEvaluator::builtin();
    let y = |deg: f64| {
        let mut o = Operand::new(OperandKind::RealRay {
            quantity: RayQuantity::Y,
            surface: None,
            px: 0.0,
            py: 0.0,
            global: false,
        });
        o.field = FieldSelector::Normalized {
            hx: 0.0,
            hy: deg / fmax,
        };
        ok(&eval(&sys, &[o])[0])
    };
    let d = 1e-3f64;
    let slope = |deg: f64| (y(deg + d) - y(deg - d)) / (2.0 * d.to_radians());
    let s0 = slope(0.0);
    for deg in [2.5f64, 5.0] {
        let th = deg.to_radians();
        let want = th.cos() * s0 * s0 * th.sin() / (y(deg) * slope(deg));
        let got = ok(&ri_at(
            &ev,
            &sys,
            FieldSelector::Normalized {
                hx: 0.0,
                hy: deg / fmax,
            },
        ));
        assert!(rel(got, want) < 1e-7, "{deg}°: {got} vs {want}");
        assert!(got < 1.0);
        // Not cos⁴ (the lens distorts): the check can tell them apart.
        assert!(rel(got, th.cos().powi(4)) > 1e-4);
    }
}

/// Ideal lens `f` at the stop (radius `R`) behind a circular aperture of
/// radius `b` at distance `d` in front: at field angle θ the beam footprint
/// on the lens is the disc `R` intersected with the aperture's shadow, a disc
/// `b` shifted by `d tan θ`, so `RI = cos⁴θ · A∩(d tan θ) / A∩(0)`.
#[test]
fn relative_illumination_of_a_vignetted_ideal_lens() {
    let (f, r, b, d) = (100.0, 10.0, 9.0, 20.0);
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 2.0 * r };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: [0.0, 10.0, 20.0, 35.0]
            .iter()
            .map(|&a| Field::new(0.0, a))
            .collect(),
    };
    let mut ap = Surface::new(1, SurfaceShape::Plane, d, MaterialRef::Air);
    ap.semi_diameter = SemiDiameter::Fixed { value: b };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        ap,
        Surface::new(
            2,
            SurfaceShape::Paraxial { focal_length: f },
            f,
            MaterialRef::Air,
        ),
        Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 2;
    s.validate().unwrap();
    // Area of the intersection of discs of radii r1, r2 at centre distance c.
    let lens_area = |r1: f64, r2: f64, c: f64| -> f64 {
        if c >= r1 + r2 {
            return 0.0;
        }
        if c <= (r1 - r2).abs() {
            return std::f64::consts::PI * r1.min(r2).powi(2);
        }
        let a1 = ((c * c + r1 * r1 - r2 * r2) / (2.0 * c * r1)).acos();
        let a2 = ((c * c + r2 * r2 - r1 * r1) / (2.0 * c * r2)).acos();
        let k = ((-c + r1 + r2) * (c + r1 - r2) * (c - r1 + r2) * (c + r1 + r2)).sqrt();
        r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * k
    };
    let ev = TraceEvaluator::builtin();
    let fine = TraceEvaluator::with_settings(
        ev.library().clone(),
        TraceEvalSettings {
            illumination_arms: 128,
            ..Default::default()
        },
    );
    for (i, deg) in [(1, 10.0f64), (2, 20.0), (3, 35.0)] {
        let th = deg.to_radians();
        let want = th.cos().powi(4) * lens_area(r, b, d * th.tan()) / lens_area(r, b, 0.0);
        let got = ok(&ri_at(&ev, &s, FieldSelector::Index { index: i }));
        let got_fine = ok(&ri_at(&fine, &s, FieldSelector::Index { index: i }));
        // Strongly vignetted (to 35 % of the unvignetted cos⁴ at 35°).
        assert!(rel(got, want) < 2e-6, "{deg}°: {got} vs {want}");
        assert!(rel(got_fine, want) < 2e-6, "{deg}°: {got_fine} vs {want}");
    }
}

/// Finite object, ideal lens imaging plane to plane (constant image
/// Jacobian `m²`): `RI = F(h)/F(0)` with the view factor of a disc of radius
/// `a` at distance `d` from a point offset by `h`,
/// `F = ½(1 − (d² + h² − a²)/√((d² + h² + a²)² − 4a²h²))`.
#[test]
fn relative_illumination_of_a_finite_object() {
    let (f, d, a) = (100.0, 200.0, 20.0);
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 2.0 * a };
    s.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 80.0)],
    };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, d, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial { focal_length: f },
            200.0,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s.validate().unwrap();
    let view = |h: f64| {
        let (d2, h2, a2) = (d * d, h * h, a * a);
        0.5 * (1.0 - (d2 + h2 - a2) / ((d2 + h2 + a2).powi(2) - 4.0 * a2 * h2).sqrt())
    };
    let ev = TraceEvaluator::builtin();
    for (field, h) in [
        (FieldSelector::Index { index: 1 }, 80.0),
        (
            FieldSelector::Normalized { hx: 0.6, hy: 0.3 },
            80.0 * 0.45f64.sqrt(),
        ),
    ] {
        let want = view(h) / view(0.0);
        let got = ok(&ri_at(&ev, &s, field));
        assert!(rel(got, want) < 1e-8, "{h}: {got} vs {want}");
    }
}

/// A vignetted Cooke triplet: the boundary-locating integration has
/// converged at the default settings (the former dropped-cell grid still
/// moved by 10 % between 12 × 24 and 96 × 192 cells).
#[test]
fn relative_illumination_converges_on_a_vignetted_triplet() {
    let ev = TraceEvaluator::builtin();
    let mut sys = samples::cooke_triplet();
    let open = ok(&ri_at(&ev, &sys, FieldSelector::Index { index: 2 }));
    let tr = ev.tracer(&sys).unwrap();
    let sd = tr.semi_diameters(&mklight_trace::SemiDiameterSettings {
        azimuths: 16,
        clip: true,
    });
    sys = mklight_trace::fix_semi_diameters(&sys, &sd, 0.0);
    for k in [1, 6] {
        if let SemiDiameter::Fixed { value } = sys.surfaces[k].semi_diameter {
            sys.surfaces[k].semi_diameter = SemiDiameter::Fixed {
                value: 0.85 * value,
            };
        }
    }
    let with = |rings, arms| {
        let e = TraceEvaluator::with_settings(
            ev.library().clone(),
            TraceEvalSettings {
                illumination_rings: rings,
                illumination_arms: arms,
                ..Default::default()
            },
        );
        ok(&ri_at(&e, &sys, FieldSelector::Index { index: 2 }))
    };
    let def = ok(&ri_at(&ev, &sys, FieldSelector::Index { index: 2 }));
    let fine = with(48, 256);
    assert!(rel(def, fine) < 1e-6, "{def} vs {fine}");
    assert!(rel(with(6, 16), fine) < 1e-6);
    assert!(def < 0.8 * open, "vignetted {def} vs open {open}");
}

// ---------------------------------------------------------------------------
// Petzval sum, edge thickness, run info
// ---------------------------------------------------------------------------

#[test]
fn ideal_lenses_add_no_petzval_curvature() {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 20.0)],
    };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial {
                focal_length: 100.0,
            },
            100.0,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s.validate().unwrap();
    let spot = Operand::new(OperandKind::RmsSpot {
        reference: SpotReference::Centroid,
        component: None,
        sampling: PupilSampling::default(),
    })
    .at_field(1);
    let v = eval(&s, &[Operand::new(OperandKind::PetzvalSum), spot]);
    // A flat image is sharp at 20°: no field curvature.
    assert_eq!(ok(&v[0]), 0.0);
    assert!(ok(&v[1]) < 1e-12);
    // An ideal lens in front of a real singlet leaves the singlet's sum.
    let single = samples::singlet();
    let p0 = ok(&eval(&single, &[Operand::new(OperandKind::PetzvalSum)])[0]);
    let mut both = single.clone();
    both.surfaces.insert(
        1,
        Surface::new(
            99,
            SurfaceShape::Paraxial {
                focal_length: 500.0,
            },
            1.0,
            MaterialRef::Air,
        ),
    );
    both.stop += 1;
    let p1 = ok(&eval(&both, &[Operand::new(OperandKind::PetzvalSum)])[0]);
    assert!(rel(p1, p0) < 1e-14, "{p1} vs {p0}");
}

#[test]
fn auto_edge_thickness_on_the_double_gauss() {
    let sys = samples::double_gauss();
    let vars: Vec<VariableSpec> = (1..=4)
        .map(|i| {
            VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(i),
            })
        })
        .collect();
    let spec = DefaultMeritSpec {
        thickness: Some(ThicknessBoundaries::with_auto_semi_diameters()),
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    let edges: Vec<Operand> = spec
        .operands
        .iter()
        .filter(|o| matches!(o.kind, OperandKind::EdgeThickness { .. }))
        .cloned()
        .collect();
    // Ten gaps between active surfaces, less the back focus.
    assert_eq!(edges.len(), 9);
    let ev = TraceEvaluator::builtin();
    let v = eval(&sys, &edges);
    // Every edge is measurable (the cemented L2/L3 interface, steeper than
    // the larger L2 front aperture, used to be out of its sag domain).
    assert!(v.iter().all(|x| x.is_ok()), "{v:?}");
    // Each surface's sag at its own clear semi-diameter.
    let tr = ev.tracer(&sys).unwrap();
    let sd = tr.semi_diameters(&mklight_trace::SemiDiameterSettings {
        azimuths: ev.settings().semi_diameter_azimuths,
        clip: true,
    });
    let fixed = mklight_trace::fix_semi_diameters(&sys, &sd, 0.0);
    let h = |k: usize| match fixed.surfaces[k].semi_diameter {
        SemiDiameter::Fixed { value } => value,
        SemiDiameter::Auto => unreachable!(),
    };
    let sag = |k: usize| fixed.surfaces[k].shape.sag(h(k).powi(2)).unwrap();
    // L3 (surfaces 4 → 5, cemented flint).
    let i = edges
        .iter()
        .position(|o| {
            matches!(o.kind, OperandKind::EdgeThickness { surface, .. } if surface == SurfaceId(4))
        })
        .unwrap();
    let want = sys.surfaces[4].thickness - sag(4) + sag(5);
    assert!((ok(&v[i]) - want).abs() < 1e-12, "{:?} vs {want}", v[i]);
    assert!(h(4) > h(5), "the smaller rim continues flat");
    // The start point evaluates (violated edge bounds are constraints, not
    // failures).
    let p = OpticalProblem::new(sys, vars, spec, ev, &OperandRegistry::builtin()).unwrap();
    let x = p.start_point().unwrap();
    let e = evaluate_one(&p, &x, &JobContext::new()).unwrap();
    assert_eq!(e.status, EvalStatus::Ok);
}

#[test]
fn runs_carry_their_run_info() {
    let sys = samples::cooke_triplet();
    let vars = vec![VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(6),
    })];
    let spec = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    let ev = TraceEvaluator::builtin();
    let p = OpticalProblem::new(
        sys.clone(),
        vars,
        spec,
        ev.clone(),
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let x0 = p.start_point().unwrap();
    let lsq = LsqSettings {
        max_iterations: 5,
        ..Default::default()
    };
    let run = ev
        .run(&sys, "optimise/least_squares", &lsq, None, || {
            least_squares(&p, &x0, &lsq, &Monitor::default())
        })
        .unwrap();
    let e = evaluate_one(&p, &x0, &JobContext::new()).unwrap();
    assert!(run.result.merit <= merit(p.weights(), &e.residuals, e.status));
    assert_eq!(run.info.analysis, "optimise/least_squares");
    assert_eq!(run.info.backend, TraceEvaluator::BACKEND);
    assert_eq!(run.info.seed, None);
    assert_eq!(
        run.info.settings["max_iterations"],
        serde_json::json!(5),
        "{:?}",
        run.info.settings
    );
    assert_eq!(
        run.info.snapshot_hash,
        mklight_core::Snapshot::new(sys).unwrap().hash().to_hex()
    );
    assert!(run.info.duration_ms > 0.0);
    // Errors of the driver pass through.
    let err = ev.run(
        p.base(),
        "x",
        &lsq,
        None,
        || -> mklight_core::MkResult<()> { Err(mklight_core::MkError::Cancelled) },
    );
    assert!(err.is_err());
}
