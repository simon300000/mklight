//! Extended surfaces: physical optimisation targets, manufacturing metrics,
//! independent derivative checks and reproducible tolerance trials.
use mklight_core::model::{ApertureShape, SemiDiameter, SurfaceShape};
use mklight_core::param::{CoefficientSet, ParamPath, ShapeParameter};
use mklight_core::{JobContext, MkResult, OpticalSystem, SurfaceId, samples};
use mklight_optim::Monitor;
use mklight_optim::deriv::{
    CheckSettings, FdScheme, JacobianSettings, check_jacobian, fd_jacobian,
};
use mklight_optim::linalg::Matrix;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::operand::{OperandKind, RayQuantity};
use mklight_optim::merit::{
    FieldSelector, MeritFunctionSpec, Operand, OperandRegistry, OpticalProblem, PupilSampling,
    PupilSymmetry, TraceEvaluator, VariableSpec,
};
use mklight_optim::problem::{
    BatchEvaluation, CandidateBatch, Constraint, EvalStatus, Problem, Variable, evaluate_one, merit,
};
use mklight_optim::tolerance::{
    MonteCarloSettings, OpticalToleranceEvaluator, ToleranceOperand, ToleranceSpec, monte_carlo,
    sensitivity,
};

fn coeff(surface: u64, set: CoefficientSet, index: usize) -> ParamPath {
    ParamPath::ShapeCoefficient {
        surface: SurfaceId(surface),
        set,
        index,
    }
}
fn scalar(surface: u64, parameter: ShapeParameter) -> ParamPath {
    ParamPath::ShapeParam {
        surface: SurfaceId(surface),
        parameter,
    }
}
fn value(s: &OpticalSystem, kind: OperandKind) -> f64 {
    TraceEvaluator::builtin()
        .evaluate_system(s, &[&Operand::new(kind)], &JobContext::new())
        .unwrap()
        .remove(0)
        .unwrap()
}
fn problem(
    s: OpticalSystem,
    paths: Vec<ParamPath>,
    spec: MeritFunctionSpec,
) -> OpticalProblem<TraceEvaluator> {
    OpticalProblem::new(
        s,
        paths.into_iter().map(VariableSpec::new).collect(),
        spec,
        TraceEvaluator::builtin(),
        &OperandRegistry::builtin(),
    )
    .unwrap()
}
fn score(p: &impl Problem, x: &[f64]) -> f64 {
    let e = evaluate_one(p, x, &JobContext::new()).unwrap();
    assert_eq!(e.status, EvalStatus::Ok);
    merit(p.weights(), &e.residuals, e.status)
}
fn optimise(p: &OpticalProblem<TraceEvaluator>) -> mklight_optim::monitor::OptimResult {
    let settings = LsqSettings {
        max_iterations: 40,
        ..Default::default()
    };
    let run = TraceEvaluator::builtin()
        .run(p.base(), "extended-surface-optim", &settings, None, || {
            least_squares(p, &p.start_point()?, &settings, &Monitor::default())
        })
        .unwrap();
    assert_eq!(run.info.backend, TraceEvaluator::BACKEND);
    assert_eq!(
        run.info.snapshot_hash,
        mklight_core::Snapshot::new(p.base().clone())
            .unwrap()
            .hash()
            .to_hex()
    );
    run.result
}

#[test]
fn q_and_odd_singlets_improve_deterministically() {
    for (s, path) in [
        (
            samples::odd_asphere_element(),
            coeff(1, CoefficientSet::Odd, 2),
        ),
        (samples::q_asphere_lens(), coeff(1, CoefficientSet::Q, 0)),
        (samples::qbfs_lens(), coeff(1, CoefficientSet::Q, 0)),
    ] {
        let paths = vec![
            path,
            ParamPath::Thickness {
                surface: SurfaceId(2),
            },
        ];
        let vars: Vec<_> = paths.iter().cloned().map(VariableSpec::new).collect();
        let spec = DefaultMeritSpec {
            fields: Some(vec![0]),
            wavelengths: Some(vec![1]),
            thickness: None,
            ..Default::default()
        }
        .generate_for(&s, &vars)
        .unwrap();
        let p = problem(s, paths, spec);
        let m0 = score(&p, &p.start_point().unwrap());
        let a = optimise(&p);
        let b = optimise(&p);
        assert!(a.merit < 0.8 * m0, "{}: {m0} -> {}", p.base().name, a.merit);
        assert_eq!(a.x, b.x);
        assert_eq!(a.merit.to_bits(), b.merit.to_bits());
        assert_eq!(score(&p, &a.x), a.merit);
    }
}

#[test]
fn hybrid_is_achromatised_with_a1() {
    let mut s = samples::hybrid_diffractive_achromat();
    let a1 = coeff(2, CoefficientSet::Phase, 0);
    let corrected = a1.get(&s).unwrap();
    a1.set(&mut s, 0.0).unwrap();
    let colour = OperandKind::AxialColour { short: 0, long: 2 };
    assert!(value(&s, colour.clone()).abs() > 0.5);
    let p = problem(
        s,
        vec![a1],
        MeritFunctionSpec::new(vec![Operand::new(colour.clone())]),
    );
    let r = optimise(&p);
    let best = p.apply(&r.x).unwrap();
    assert!(value(&best, colour).abs() < 1e-6, "{r:?}");
    assert!((r.x[0] - corrected).abs() < 1e-4);
}

#[test]
fn anamorphic_reaches_squeeze_and_holds_y_focal_length() {
    let s = samples::anamorphic_adapter();
    let efl_y = value(&s, OperandKind::EflY);
    let spec = MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::AnamorphicSqueeze).target(1.7),
        Operand::new(OperandKind::EflY)
            .target(efl_y)
            .as_constraint(),
    ]);
    let p = problem(s, vec![scalar(10, ShapeParameter::CurvatureX)], spec);
    let r = optimise(&p);
    let best = p.apply(&r.x).unwrap();
    assert!(
        (value(&best, OperandKind::AnamorphicSqueeze) - 1.7).abs() < 1e-6,
        "{r:?}"
    );
    assert!((value(&best, OperandKind::EflY) - efl_y).abs() < 1e-8);
    assert!(r.violation < 1e-8);
}

#[test]
fn freeform_corrector_improves_over_both_field_axes() {
    let s = samples::xy_freeform();
    let paths = vec![
        coeff(1, CoefficientSet::Xy, 2),
        coeff(1, CoefficientSet::Xy, 4),
        coeff(1, CoefficientSet::Xy, 5),
        coeff(1, CoefficientSet::Xy, 7),
        ParamPath::Thickness {
            surface: SurfaceId(2),
        },
    ];
    let vars: Vec<_> = paths.iter().cloned().map(VariableSpec::new).collect();
    let spec = DefaultMeritSpec {
        thickness: None,
        wavelengths: Some(vec![1]),
        ..Default::default()
    }
    .generate_for(&s, &vars)
    .unwrap();
    assert!(spec.operands.iter().any(
        |o| matches!(o.field, FieldSelector::Normalized { hx, hy } if hx != 0.0 && hy == 0.0)
    ));
    assert!(spec.operands.iter().any(
        |o| matches!(o.field, FieldSelector::Normalized { hx, hy } if hx == 0.0 && hy != 0.0)
    ));
    assert!(
        spec.operands
            .iter()
            .all(|o| o.kind.ray_group().unwrap().sampling.symmetry == PupilSymmetry::None)
    );
    let p = problem(s, paths, spec);
    let m0 = score(&p, &p.start_point().unwrap());
    let r = optimise(&p);
    assert!(r.merit < 0.7 * m0, "{m0} -> {}", r.merit);
    assert_eq!(score(&p, &r.x), r.merit);
}

#[test]
fn q_coefficient_and_toroid_curvature_tolerances() {
    for (s, path, delta) in [
        (samples::qbfs_lens(), coeff(1, CoefficientSet::Q, 1), 0.001),
        (
            samples::toroidal_element(),
            scalar(1, ShapeParameter::CurvatureX),
            1e-4,
        ),
    ] {
        let tol = ToleranceSpec::new(vec![ToleranceOperand::symmetric(path, delta)]);
        let criterion = DefaultMeritSpec {
            thickness: None,
            ..Default::default()
        }
        .generate_for_tolerancing(&s, &tol)
        .unwrap();
        let ev = TraceEvaluator::builtin();
        let reg = OperandRegistry::builtin();
        let eval = OpticalToleranceEvaluator {
            nominal: &s,
            spec: &tol,
            criterion: &criterion,
            evaluator: &ev,
            registry: &reg,
            lsq: LsqSettings::default(),
        };
        let run = ev
            .run(&s, "extended-surface-tolerance", &tol, None, || {
                sensitivity(&tol, &eval, &Monitor::default())
            })
            .unwrap();
        let r = run.result;
        assert!(r.nominal.ok && r.operands[0].minus.ok && r.operands[0].plus.ok);
        assert!(r.operands[0].worst > 1e-12);
        let settings = MonteCarloSettings {
            trials: 12,
            seed: 1024,
            batch: 4,
            threshold: None,
        };
        let a = monte_carlo(&tol, &eval, &settings, &Monitor::default()).unwrap();
        let b = monte_carlo(&tol, &eval, &settings, &Monitor::default()).unwrap();
        assert_eq!(a.failures, 0);
        assert_eq!(a.criteria, b.criteria);
        assert!(a.std_dev > 0.0);
    }
}

/// Expose the production FD stencil through the analytic hook, then compare
/// it with the independent Richardson stencil in check_jacobian.
struct Fd<'a>(&'a OpticalProblem<TraceEvaluator>);
impl Problem for Fd<'_> {
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_residuals(&self) -> usize {
        self.0.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.0.weights()
    }
    fn constraints(&self) -> &[Constraint] {
        self.0.constraints()
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        self.0.evaluate_batch(x, ctx)
    }
    fn jacobian(&self, x: &[f64], ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        Some((|| {
            let f0 = evaluate_one(self.0, x, ctx)?;
            let j = fd_jacobian(
                self.0,
                x,
                &f0,
                &JacobianSettings {
                    scheme: FdScheme::Central,
                    ..Default::default()
                },
                ctx,
            )?;
            assert!(j.failed_columns.is_empty());
            Ok((j.residuals, j.constraints))
        })())
    }
}

#[test]
fn surface_sample_jacobians() {
    use CoefficientSet::*;
    use ShapeParameter::*;
    let cases = vec![
        (
            samples::odd_asphere_element(),
            vec![coeff(1, Odd, 1), coeff(1, Odd, 2), coeff(1, Odd, 5)],
        ),
        (
            samples::q_asphere_lens(),
            vec![
                coeff(1, Q, 0),
                coeff(1, Q, 2),
                coeff(1, Q, 19),
                scalar(1, NormRadius),
            ],
        ),
        (
            samples::qbfs_lens(),
            vec![
                coeff(1, Q, 0),
                coeff(1, Q, 2),
                coeff(1, Q, 19),
                scalar(1, NormRadius),
            ],
        ),
        (
            samples::hybrid_diffractive_achromat(),
            vec![
                coeff(2, Phase, 0),
                coeff(2, Phase, 2),
                scalar(2, NormRadius),
            ],
        ),
        (
            samples::anamorphic_adapter(),
            vec![scalar(10, CurvatureX), scalar(10, ConicX)],
        ),
        (
            samples::biconic_element(),
            vec![
                scalar(1, CurvatureX),
                scalar(1, CurvatureY),
                scalar(1, ConicX),
                scalar(1, ConicY),
                coeff(1, XAsphere, 0),
                coeff(1, YAsphere, 0),
            ],
        ),
        (
            samples::toroidal_element(),
            vec![
                scalar(1, CurvatureX),
                scalar(1, CurvatureY),
                scalar(1, ConicY),
                coeff(1, YAsphere, 0),
            ],
        ),
        (
            samples::xy_freeform(),
            vec![
                coeff(1, Xy, 2),
                coeff(1, Xy, 5),
                coeff(1, Xy, 8),
                scalar(1, NormRadius),
            ],
        ),
        (
            samples::zernike_sag(),
            vec![
                coeff(1, Zernike, 4),
                coeff(1, Zernike, 9),
                scalar(1, NormRadius),
            ],
        ),
    ];
    for (mut s, paths) in cases {
        // The clear radius differs from the normalisation radius, including
        // high Q orders whose low powers rule out monomial step estimates.
        s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 2.1 };
        let mut ops = Vec::new();
        for (hx, hy) in [(0.7, 0.0), (0.0, -0.7)] {
            for quantity in [RayQuantity::X, RayQuantity::Y] {
                for (px, py) in [(0.4, 0.6), (-0.6, -0.4)] {
                    let mut op = Operand::new(OperandKind::RealRay {
                        quantity,
                        surface: None,
                        px,
                        py,
                        global: false,
                    });
                    op.field = FieldSelector::Normalized { hx, hy };
                    ops.push(op);
                }
            }
        }
        let p = problem(s, paths, MeritFunctionSpec::new(ops));
        let x = p.start_point().unwrap();
        score(&p, &x);
        let r = check_jacobian(
            &Fd(&p),
            &x,
            &CheckSettings {
                rel_tol: 1e-4,
                abs_floor: 0.1,
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert!(r.passed, "{}: {r:?}", p.base().name);
    }
}

#[test]
fn manufacturing_known_answers_and_aperture_edges() {
    let mut s = samples::hybrid_diffractive_achromat();
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 3.0 };
    let a1 = coeff(2, CoefficientSet::Phase, 0);
    a1.set(&mut s, -40.0).unwrap();
    let slope = 2.0 * 40.0 * 3.0 / 25.0;
    assert!(
        (value(
            &s,
            OperandKind::DoeMaxPhaseSlope {
                surface: SurfaceId(2)
            }
        ) - slope)
            .abs()
            < 1e-12
    );
    assert!(
        (value(
            &s,
            OperandKind::DoeMinZonePeriod {
                surface: SurfaceId(2)
            }
        ) - std::f64::consts::TAU / slope)
            .abs()
            < 1e-12
    );
    assert!(
        (value(
            &s,
            OperandKind::DoeTotalPhase {
                surface: SurfaceId(2)
            }
        ) - 40.0 * 9.0 / 25.0)
            .abs()
            < 1e-12
    );
    // Obscuration removes the inner phase range; metadata and traced order
    // cannot change the manufactured pattern.
    s.surfaces[2].aperture = Some(ApertureShape::Annular {
        inner_radius: 1.0,
        outer_radius: 3.0,
    });
    scalar(2, ShapeParameter::TraceOrder)
        .set(&mut s, -2.0)
        .unwrap();
    assert!(
        (value(
            &s,
            OperandKind::DoeTotalPhase {
                surface: SurfaceId(2)
            }
        ) - 40.0 * 8.0 / 25.0)
            .abs()
            < 1e-12
    );
    // A sphere encoded as either Q family has zero fitted departure.
    for mut s in [samples::q_asphere_lens(), samples::qbfs_lens()] {
        s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 4.0 };
        s.surfaces[1].shape = match s.surfaces[1].shape {
            SurfaceShape::QCon {
                curvature,
                norm_radius,
                ..
            } => SurfaceShape::QCon {
                curvature,
                norm_radius,
                conic: 0.0,
                coefficients: vec![],
            },
            SurfaceShape::QBfs {
                curvature,
                norm_radius,
                ..
            } => SurfaceShape::QBfs {
                curvature,
                norm_radius,
                coefficients: vec![],
            },
            _ => unreachable!(),
        };
        assert!(
            value(
                &s,
                OperandKind::QMaxDeparture {
                    surface: SurfaceId(1)
                }
            ) < 1e-12
        );
        assert!(
            value(
                &s,
                OperandKind::QMaxDepartureSlope {
                    surface: SurfaceId(1)
                }
            ) < 1e-12
        );
        coeff(1, CoefficientSet::Q, 0).set(&mut s, 0.02).unwrap();
        assert!(
            value(
                &s,
                OperandKind::QMaxDeparture {
                    surface: SurfaceId(1)
                }
            ) > 1e-4
        );
        assert!(
            value(
                &s,
                OperandKind::QMaxDepartureSlope {
                    surface: SurfaceId(1)
                }
            ) > 1e-4
        );
    }
    let mut s = samples::xy_freeform();
    s.surfaces[1].shape = SurfaceShape::XYPolynomial {
        curvature: 0.02,
        conic: -0.5,
        norm_radius: 5.0,
        coefficients: vec![0.0, 0.0, 0.01],
    };
    s.surfaces[1].aperture = Some(ApertureShape::Rectangular {
        half_x: 3.0,
        half_y: 2.0,
    });
    assert!(
        (value(
            &s,
            OperandKind::FreeformMaxDeparture {
                surface: SurfaceId(1)
            }
        ) - 0.01 * 9.0 / 25.0)
            .abs()
            < 1e-12
    );
    coeff(1, CoefficientSet::Xy, 2).set(&mut s, 0.0).unwrap();
    coeff(1, CoefficientSet::Xy, 3).set(&mut s, 0.01).unwrap();
    assert!(
        (value(
            &s,
            OperandKind::FreeformMaxDeparture {
                surface: SurfaceId(1)
            }
        ) - 0.01 * 3.0 * 2.0 / 25.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn default_grid_symmetry_and_squeeze_option() {
    let s = samples::anamorphic_adapter();
    let spec = DefaultMeritSpec {
        thickness: None,
        squeeze_target: Some(2.0),
        sampling: PupilSampling {
            symmetry: PupilSymmetry::Bilateral,
            ..Default::default()
        },
        ..Default::default()
    }
    .generate(&s)
    .unwrap();
    assert_eq!(
        spec.operands
            .iter()
            .filter(|o| o.kind == OperandKind::AnamorphicSqueeze)
            .count(),
        1
    );
    let ray_ops: Vec<_> = spec
        .operands
        .iter()
        .filter(|o| o.kind.ray_group().is_some())
        .collect();
    assert!(
        ray_ops
            .iter()
            .all(|o| o.kind.ray_group().unwrap().sampling.symmetry == PupilSymmetry::None)
    );
    let mut fields = Vec::new();
    for op in ray_ops {
        if !fields.contains(&op.field) {
            fields.push(op.field);
        }
    }
    assert_eq!(fields.len(), 9);
    let s = samples::xy_freeform();
    assert!(!mklight_optim::merit::default::is_meridionally_symmetric(
        &s
    ));
    // A currently symmetric XY shape can become asymmetric through a zero
    // coefficient, so its variable also forces full-pupil sampling.
    let mut s = samples::singlet();
    s.surfaces[1].shape = SurfaceShape::XYPolynomial {
        curvature: 1.0 / 60.0,
        conic: 0.0,
        norm_radius: 5.0,
        coefficients: vec![],
    };
    let spec = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate_for(&s, &[VariableSpec::new(coeff(1, CoefficientSet::Xy, 5))])
    .unwrap();
    assert!(
        spec.operands
            .iter()
            .all(|o| o.kind.ray_group().unwrap().sampling.symmetry == PupilSymmetry::None)
    );
}

#[test]
fn first_order_axis_operands_match_thick_lens_powers() {
    let s = samples::biconic_element();
    let ev = TraceEvaluator::builtin();
    let tr = ev.tracer(&s).unwrap();
    let n = tr.system().surfaces[1].n_after[1];
    let t = s.surfaces[1].thickness;
    let p2 = (1.0 - n) * s.surfaces[2].shape.curvature();
    let mut efl = Vec::new();
    for (c, e, b) in [
        (0.02, OperandKind::EflX, OperandKind::BflX),
        (1.0 / 60.0, OperandKind::EflY, OperandKind::BflY),
    ] {
        let p1 = (n - 1.0) * c;
        let a = 1.0 - t * p1 / n;
        let power = p1 + a * p2;
        let f = value(&s, e);
        assert!((f - 1.0 / power).abs() < 1e-10);
        assert!((value(&s, b) - a / power).abs() < 1e-10);
        efl.push(f);
    }
    assert!((value(&s, OperandKind::AnamorphicSqueeze) - efl[1] / efl[0]).abs() < 1e-12);
    assert_eq!(value(&s, OperandKind::Efl), efl[1]);
}

#[test]
fn doe_metric_uses_the_curved_surface_and_q_fit_has_known_departure() {
    let mut s = samples::hybrid_diffractive_achromat();
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 3.0 };
    let SurfaceShape::Diffractive { base, phase } = &mut s.surfaces[2].shape else {
        unreachable!()
    };
    *base = Box::new(SurfaceShape::sphere(10.0));
    phase.coefficients = vec![-40.0];
    let expected = 2.0 * 40.0 * 3.0 / 25.0 * (1.0_f64 - 0.09).sqrt();
    assert!(
        (value(
            &s,
            OperandKind::DoeMaxPhaseSlope {
                surface: SurfaceId(2)
            }
        ) - expected)
            .abs()
            < 1e-12
    );
    let mut s = samples::q_asphere_lens();
    // For z = a (r/R)^4 on a flat base, the area least-squares sphere
    // tends to a quadratic: c = 2a/R², piston = -a/6, max departure = a/6,
    // max slope departure = 2a/R. The sphere's higher terms are O(a³).
    let a = 1e-5;
    s.surfaces[1].shape = SurfaceShape::QCon {
        curvature: 0.0,
        conic: 0.0,
        norm_radius: 5.0,
        coefficients: vec![a],
    };
    s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 5.0 };
    let departure = value(
        &s,
        OperandKind::QMaxDeparture {
            surface: SurfaceId(1),
        },
    );
    let slope = value(
        &s,
        OperandKind::QMaxDepartureSlope {
            surface: SurfaceId(1),
        },
    );
    assert!((departure / (a / 6.0) - 1.0).abs() < 0.001);
    assert!((slope / (2.0 * a / 5.0) - 1.0).abs() < 0.001);
}

#[test]
fn integer_diffraction_order_search_and_tolerancing() {
    use mklight_optim::discrete::{DiscreteSettings, FnDiscrete, SingleChange, tabu_search};
    use mklight_optim::problem::ScalarEval;
    use mklight_optim::tolerance::Distribution;
    let mut s = samples::hybrid_diffractive_achromat();
    let order = scalar(2, ShapeParameter::TraceOrder);
    order.set(&mut s, 2.0).unwrap();
    let target = value(&s, OperandKind::EflY);
    order.set(&mut s, 0.0).unwrap();
    let p = problem(
        s.clone(),
        vec![order.clone()],
        MeritFunctionSpec::new(vec![Operand::new(OperandKind::EflY).target(target)]),
    );
    let search = FnDiscrete::new(vec![3], move |assignment| ScalarEval {
        merit: score(&p, &[assignment[0] as f64]),
        violation: 0.0,
        status: EvalStatus::Ok,
    });
    let r = tabu_search(
        &search,
        &SingleChange::default(),
        &[0],
        &DiscreteSettings::default(),
        &Monitor::default(),
    )
    .unwrap();
    assert_eq!(r.assignment, vec![2]);
    assert!(r.merit < 1e-16);
    order.set(&mut s, 1.0).unwrap();
    let tol = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(order, 1.0).with_distribution(Distribution::EndPoint),
    ]);
    let criterion = MeritFunctionSpec::new(vec![Operand::new(OperandKind::EflY).target(target)]);
    let ev = TraceEvaluator::builtin();
    let reg = OperandRegistry::builtin();
    let eval = OpticalToleranceEvaluator {
        nominal: &s,
        spec: &tol,
        criterion: &criterion,
        evaluator: &ev,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let report = sensitivity(&tol, &eval, &Monitor::default()).unwrap();
    assert!(report.operands[0].minus.ok && report.operands[0].plus.ok);
    assert!(report.operands[0].plus.criterion < 1e-16);
    assert!(report.operands[0].minus.criterion > 1.0);
    let mc = monte_carlo(
        &tol,
        &eval,
        &MonteCarloSettings {
            trials: 8,
            seed: 5,
            batch: 4,
            threshold: None,
        },
        &Monitor::default(),
    )
    .unwrap();
    assert_eq!(mc.failures, 0);
}
