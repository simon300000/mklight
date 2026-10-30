//! Regression tests: pinned and failing variables, derivative checks near
//! narrow bounds, run isolation of a reused monitor, hard evaluation budgets,
//! thickness gaps across stops/dummies/mirrors, half-pupil validity, ray
//! groups, selector validation and the parametric evaluation path.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use mklight_core::model::{
    Field, FieldKind, Interaction, MaterialRef, Placement, SemiDiameter, Surface, SurfaceShape,
};
use mklight_core::param::ParamPath;
use mklight_core::{Configuration, JobContext, MkResult, OpticalSystem, SurfaceId};
use mklight_optim::deriv::{CheckSettings, check_jacobian, fd_jacobian};
use mklight_optim::discrete::{DiscreteSettings, FnDiscrete, SingleChange, tabu_search};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::global::nelder_mead::{NelderMeadSettings, nelder_mead};
use mklight_optim::linalg::Matrix;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::geometric::ModelEvaluator;
use mklight_optim::merit::operand::{
    Axis, MediumFilter, SpotReference, ThicknessMeasure, WavefrontReference,
};
use mklight_optim::merit::{
    BuildError, CompositeEvaluator, FieldSelector, MeritFunctionSpec, Operand, OperandEvaluator,
    OperandKind, OperandRegistry, OperandRole, OperandValue, OpticalProblem, ParametricBatch,
    PupilSampling, PupilSymmetry, RayGroup, VariableSpec,
};
use mklight_optim::problem::{ScalarEval, evaluate_one};
use mklight_optim::tolerance::{
    Compensator, Distribution, OpticalToleranceEvaluator, ToleranceOperand, ToleranceSpec,
    sensitivity,
};
use mklight_optim::{
    EvalStatus, FdScheme, FnProblem, JacobianSettings, Monitor, Termination, Variable,
};

fn fd_only(scheme: FdScheme) -> LsqSettings {
    LsqSettings {
        jacobian: JacobianSettings {
            scheme,
            use_analytic: false,
            relative_step: None,
        },
        ..Default::default()
    }
}

// ---------- pinned and failing variables ----------

#[test]
fn pinned_variable_gets_a_zero_column_and_lm_still_converges() {
    // r = (a − 3, a·b − 2) with b pinned to 1: minimum a = 2.5, Φ = 0.5.
    let p = FnProblem::residuals(
        vec![Variable::new("a"), Variable::bounded("b", 1.0, 1.0)],
        2,
        |x, r| {
            r[0] = x[0] - 3.0;
            r[1] = x[0] * x[1] - 2.0;
        },
    );
    let ctx = JobContext::new();
    let f0 = evaluate_one(&p, &[0.0, 1.0], &ctx).unwrap();
    for scheme in [FdScheme::Forward, FdScheme::Central] {
        let s = JacobianSettings {
            scheme,
            ..Default::default()
        };
        let j = fd_jacobian(&p, &[0.0, 1.0], &f0, &s, &ctx).unwrap();
        assert!(j.residuals.data.iter().all(|v| v.is_finite()), "{j:?}");
        assert_eq!((j.residuals[(0, 1)], j.residuals[(1, 1)]), (0.0, 0.0));
        assert!(j.failed_columns.is_empty());
        // Only `a` is probed.
        assert_eq!(
            j.evaluations,
            if scheme == FdScheme::Forward { 1 } else { 2 }
        );
    }
    for s in [LsqSettings::default(), LsqSettings::dogleg()] {
        let r = least_squares(&p, &[0.0, 1.0], &s, &Monitor::default()).unwrap();
        assert!(r.termination.is_converged(), "{:?}", r.termination);
        assert!((r.x[0] - 2.5).abs() < 1e-8 && r.x[1] == 1.0, "{:?}", r.x);
        assert!((r.merit - 0.5).abs() < 1e-12);
    }
}

#[test]
fn failed_probe_is_retried_on_the_other_side() {
    // r = x, evaluation fails for x > 1, start on the failure edge.
    let p = FnProblem::new(vec![Variable::new("x")], 1, |x, r, _| {
        r[0] = x[0];
        if x[0] > 1.0 {
            EvalStatus::Failed
        } else {
            EvalStatus::Ok
        }
    });
    for scheme in [FdScheme::Forward, FdScheme::Central] {
        let r = least_squares(&p, &[1.0], &fd_only(scheme), &Monitor::default()).unwrap();
        assert!(
            r.termination.is_converged(),
            "{scheme:?}: {:?}",
            r.termination
        );
        assert!(
            r.x[0].abs() < 1e-8 && r.failed_variables.is_empty(),
            "{r:?}"
        );
    }
    // 2-D: r = (a, b − a/2), failing for a > 1: `a` must not stay frozen.
    let p = FnProblem::new(
        vec![Variable::new("a"), Variable::new("b")],
        2,
        |x, r, _| {
            r[0] = x[0];
            r[1] = x[1] - 0.5 * x[0];
            if x[0] > 1.0 {
                EvalStatus::Failed
            } else {
                EvalStatus::Ok
            }
        },
    );
    let r = least_squares(
        &p,
        &[1.0, 3.0],
        &fd_only(FdScheme::Forward),
        &Monitor::default(),
    )
    .unwrap();
    assert!(r.termination.is_converged(), "{:?}", r.termination);
    assert!(r.x[0].abs() < 1e-8 && r.x[1].abs() < 1e-8, "{:?}", r.x);
}

#[test]
fn derivative_failing_on_both_sides_is_not_reported_as_convergence() {
    // `b` cannot move at all (any change fails), `a` is free: a → 2 but the
    // run must say that b's stationarity is not certified.
    let p = FnProblem::new(
        vec![Variable::new("a"), Variable::new("b")],
        2,
        |x, r, _| {
            r[0] = x[0] - 2.0;
            r[1] = x[1] - 3.0;
            if x[1] != 0.0 {
                EvalStatus::Failed
            } else {
                EvalStatus::Ok
            }
        },
    );
    for scheme in [FdScheme::Forward, FdScheme::Central] {
        let r = least_squares(&p, &[0.0, 0.0], &fd_only(scheme), &Monitor::default()).unwrap();
        assert_eq!(r.termination, Termination::DerivativeFailure, "{scheme:?}");
        assert!(!r.termination.is_converged());
        assert_eq!(r.failed_variables, vec![1]);
        assert!((r.x[0] - 2.0).abs() < 1e-8 && r.x[1] == 0.0);
        assert!(r.history.iter().all(|h| h.failed_variables == vec![1]));
    }
}

#[test]
fn zero_range_compensator_and_nominal_centred_window() {
    let mut s = OpticalSystem::new_empty();
    let g = MaterialRef::Constant { n: 1.5 };
    s.surfaces
        .insert(1, Surface::new(10, SurfaceShape::sphere(50.0), 5.0, g));
    s.surfaces.insert(
        2,
        Surface::new(11, SurfaceShape::sphere(-50.0), 3.0, MaterialRef::Air),
    );
    s.stop = 3;
    s.surfaces[3].thickness = 92.0;
    let t = |id| ParamPath::Thickness {
        surface: SurfaceId(id),
    };
    let crit = MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::TotalTrack {
            from: None,
            to: None,
        })
        .target(100.0),
    ]);
    let reg = OperandRegistry::builtin();
    let ev = ModelEvaluator;
    // A pinned compensator next to a free one must not stop the solve.
    let mut spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(t(10), 0.2)]);
    spec.compensators = vec![
        Compensator {
            param: t(11),
            range: Some(0.0),
            scale: None,
        },
        Compensator {
            param: t(1),
            range: Some(1.0),
            scale: None,
        },
    ];
    let te = OpticalToleranceEvaluator {
        nominal: &s,
        spec: &spec,
        criterion: &crit,
        evaluator: &ev,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let r = sensitivity(&spec, &te, &Monitor::default()).unwrap();
    for o in &r.operands {
        assert!(o.minus.ok && o.plus.ok);
        assert!(o.worst < 1e-20, "{o:?}");
        assert_eq!(o.plus.compensators[0], 3.0);
    }
    // The compensator is itself toleranced (+0.3 end point): its window is
    // nominal ± 0.1 = [91.9, 92.1], which contains the optimum 92.
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(1), 0.3).with_distribution(Distribution::EndPoint),
    ]);
    spec.compensators = vec![Compensator {
        param: t(1),
        range: Some(0.1),
        scale: None,
    }];
    let te = OpticalToleranceEvaluator {
        nominal: &s,
        spec: &spec,
        criterion: &crit,
        evaluator: &ev,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let r = sensitivity(&spec, &te, &Monitor::default()).unwrap();
    let o = &r.operands[0];
    assert!(o.worst < 1e-20, "{o:?}");
    assert!((o.plus.compensators[0] - 92.0).abs() < 1e-9);
}

// ---------- derivative check ----------

#[test]
fn derivative_check_inside_a_narrow_box() {
    // Box [0, 1.5e-3] narrower than 2h (h = 1e-3): the probes must stay
    // inside it, or a correct Jacobian is reported as wrong.
    let mk = |sign: f64| {
        FnProblem::residuals(vec![Variable::bounded("x", 0.0, 1.5e-3)], 1, |x, r| {
            r[0] = (3.0 * x[0]).exp();
        })
        .with_jacobian(move |x| {
            (
                Matrix::from_row_major(1, 1, vec![sign * 3.0 * (3.0 * x[0]).exp()]),
                Matrix::zeros(0, 1),
            )
        })
    };
    let ctx = JobContext::new();
    for x in [1e-3, 0.0, 1.5e-3, 0.7e-3] {
        let rep = check_jacobian(&mk(1.0), &[x], &CheckSettings::default(), &ctx).unwrap();
        assert!(rep.passed, "x = {x}: {rep:?}");
        let rep = check_jacobian(&mk(-1.0), &[x], &CheckSettings::default(), &ctx).unwrap();
        assert!(!rep.passed);
    }
}

// ---------- monitor reuse ----------

#[test]
fn reused_monitor_never_returns_another_runs_point() {
    let mon = Monitor::default();
    // DE on a 2-variable problem with minimum 0 …
    let p2 = FnProblem::residuals(
        vec![
            Variable::bounded("a", -1.0, 1.0),
            Variable::bounded("b", -1.0, 1.0),
        ],
        2,
        |x, r| {
            r[0] = x[0];
            r[1] = x[1];
        },
    );
    let d = differential_evolution(
        &p2,
        None,
        &DeSettings {
            max_generations: 50,
            ..Default::default()
        },
        &mon,
    )
    .unwrap();
    assert!(d.merit < 1e-6);
    // … then Nelder–Mead on a 3-variable problem whose minimum merit is 1.
    let p3 = FnProblem::residuals(vars(3), 2, |x, r| {
        r[0] = 1.0;
        r[1] = x[0] + x[1] + x[2] - 1.0;
    });
    let n = nelder_mead(&p3, &[0.0; 3], &NelderMeadSettings::default(), &mon).unwrap();
    assert_eq!(n.x.len(), 3);
    assert!(n.merit >= 1.0 - 1e-12, "{}", n.merit);
    assert_eq!(mon.best.get().unwrap().x.len(), 3);
    // A discrete run reports through its own slot, never the continuous one.
    let dp = FnDiscrete::new(vec![3, 3], |a| ScalarEval {
        merit: (a[0] + a[1]) as f64,
        violation: 0.0,
        status: EvalStatus::Ok,
    });
    let t = tabu_search(
        &dp,
        &SingleChange::default(),
        &[2, 2],
        &Default::default(),
        &mon,
    )
    .unwrap();
    assert_eq!(t.assignment, vec![0, 0]);
    assert_eq!(mon.best_assignment.get().unwrap().assignment, vec![0, 0]);
    assert!(mon.best.get().is_none());
}

// ---------- evaluation budgets ----------

/// Counts every candidate evaluation.
fn counted<F>(n: usize, m: usize, f: F) -> (FnProblem, Arc<AtomicUsize>)
where
    F: Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let v = (0..n)
        .map(|i| Variable::bounded(format!("x{i}"), -10.0, 10.0))
        .collect();
    let p = FnProblem::residuals(v, m, move |x, r| {
        c.fetch_add(1, Ordering::SeqCst);
        f(x, r)
    });
    (p, calls)
}

#[test]
fn evaluation_budgets_are_hard_caps() {
    let fr = |x: &[f64], r: &mut [f64]| {
        r[0] = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1];
        r[1] = -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1];
    };
    for budget in [3, 5, 17, 60, 400] {
        let (p, calls) = counted(2, 2, fr);
        let s = HammerSettings {
            max_evaluations: budget,
            local: fd_only(FdScheme::Forward),
            ..Default::default()
        };
        let r = hammer(&p, &[0.5, -2.0], &s, &Monitor::default()).unwrap();
        let used = calls.load(Ordering::SeqCst);
        assert!(
            used <= budget && r.evaluations == used,
            "hammer {budget}: {used} {r:?}"
        );

        let (p, calls) = counted(2, 2, fr);
        let r = least_squares(
            &p,
            &[0.5, -2.0],
            &LsqSettings {
                max_evaluations: budget,
                ..fd_only(FdScheme::Central)
            },
            &Monitor::default(),
        )
        .unwrap();
        let used = calls.load(Ordering::SeqCst);
        assert!(
            used <= budget && r.evaluations == used,
            "lm {budget}: {used}"
        );

        let (p, calls) = counted(2, 2, fr);
        let r = differential_evolution(
            &p,
            None,
            &DeSettings {
                max_evaluations: budget,
                ..Default::default()
            },
            &Monitor::default(),
        )
        .unwrap();
        let used = calls.load(Ordering::SeqCst);
        assert!(
            used <= budget && r.evaluations == used,
            "de {budget}: {used}"
        );

        let (p, calls) = counted(4, 2, |x, r| {
            r[0] = x[0] * x[0] + x[1] - 1.0;
            r[1] = x[2] - x[3];
        });
        let r = nelder_mead(
            &p,
            &[3.0; 4],
            &NelderMeadSettings {
                max_evaluations: budget,
                ..Default::default()
            },
            &Monitor::default(),
        )
        .unwrap();
        let used = calls.load(Ordering::SeqCst);
        assert!(
            used <= budget && r.evaluations == used,
            "nm {budget}: {used}"
        );

        let dp = FnDiscrete::new(vec![7; 6], |a| ScalarEval {
            merit: a.iter().map(|&v| (v as f64 - 3.0).powi(2)).sum(),
            violation: 0.0,
            status: EvalStatus::Ok,
        });
        let r = tabu_search(
            &dp,
            &SingleChange::default(),
            &[0; 6],
            &DiscreteSettings {
                max_evaluations: budget,
                ..Default::default()
            },
            &Monitor::default(),
        )
        .unwrap();
        assert!(r.evaluations <= budget, "tabu {budget}: {}", r.evaluations);
    }
}

// ---------- thickness gaps ----------

fn glass() -> MaterialRef {
    MaterialRef::Model {
        nd: 1.5168,
        vd: 64.2,
        dpgf: 0.0,
    }
}

/// OBJ | 10 (glass 4) | 11 (air 3) | STOP (air 2) | 12 (glass 5) | 13 (air 50) | IMG,
/// all with fixed semi-diameters 8.
fn two_singlets() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    let mk = |id, r: f64, t, m| {
        let mut sf = Surface::new(id, SurfaceShape::sphere(r), t, m);
        sf.semi_diameter = SemiDiameter::Fixed { value: 8.0 };
        sf
    };
    s.surfaces.insert(1, mk(10, 40.0, 4.0, glass()));
    s.surfaces.insert(2, mk(11, -60.0, 3.0, MaterialRef::Air));
    s.surfaces[3].thickness = 2.0; // stop
    s.surfaces.insert(4, mk(12, 50.0, 5.0, glass()));
    s.surfaces.insert(5, mk(13, -80.0, 50.0, MaterialRef::Air));
    s.stop = 3;
    s.validate().unwrap();
    s
}

fn values(s: &OpticalSystem, kinds: Vec<OperandKind>) -> Vec<OperandValue> {
    let ops: Vec<Operand> = kinds.into_iter().map(Operand::new).collect();
    let refs: Vec<&Operand> = ops.iter().collect();
    ModelEvaluator
        .evaluate_batch(std::slice::from_ref(s), &refs, &JobContext::new())
        .unwrap()
        .remove(0)
}

#[test]
fn air_gap_runs_through_the_stop_and_back_focus_is_excluded() {
    let s = two_singlets();
    let sag = |r: f64, h: f64| r.signum() * (r.abs() - (r * r - h * h).sqrt());
    let mm = |measure, medium, max: bool| {
        if max {
            OperandKind::MaxThickness {
                measure,
                medium,
                from: None,
                to: None,
            }
        } else {
            OperandKind::MinThickness {
                measure,
                medium,
                from: None,
                to: None,
            }
        }
    };
    let v = values(
        &s,
        vec![
            mm(ThicknessMeasure::Centre, MediumFilter::Air, true),
            mm(ThicknessMeasure::Centre, MediumFilter::Air, false),
            mm(ThicknessMeasure::Centre, MediumFilter::Any, true),
            mm(ThicknessMeasure::Edge, MediumFilter::Air, false),
            OperandKind::CentreThickness {
                surface: SurfaceId(11),
                to: Some(SurfaceId(12)),
            },
        ],
    );
    // One air gap of 3 + 2 = 5 mm (not 3 and 2; not the 50 mm back focus).
    assert_eq!(v[0], Ok(5.0));
    assert_eq!(v[1], Ok(5.0));
    assert_eq!(v[2], Ok(5.0));
    let edge = 5.0 - sag(-60.0, 8.0) + sag(50.0, 8.0);
    assert!((v[3].clone().unwrap() - edge).abs() < 1e-12);
    assert_eq!(v[4], Ok(5.0));

    // Default merit: glass gaps after 10 and 12, the merged air gap 11 → 12.
    let spec = DefaultMeritSpec::default().generate(&s).unwrap();
    let bounds: Vec<&OperandKind> = spec
        .operands
        .iter()
        .filter(|o| o.role == OperandRole::Constraint)
        .map(|o| &o.kind)
        .collect();
    assert!(bounds.contains(&&OperandKind::CentreThickness {
        surface: SurfaceId(11),
        to: Some(SurfaceId(12)),
    }));
    let starts: Vec<u64> = bounds
        .iter()
        .map(|k| match k {
            OperandKind::CentreThickness { surface, .. }
            | OperandKind::EdgeThickness { surface, .. } => surface.0,
            _ => unreachable!(),
        })
        .collect();
    assert!(starts.iter().all(|&id| id == 10 || id == 11 || id == 12));
    // glass ×2 (min, max centre, min edge) + air (min centre, min edge).
    assert_eq!(bounds.len(), 3 + 3 + 2);
}

#[test]
fn stop_at_a_lens_vertex_keeps_the_default_merit_feasible() {
    // OBJ | STOP (t = 0) | L1 (glass 6) | L2 (air 50) | IMG.
    let mut s = OpticalSystem::new_empty();
    s.surfaces[1].thickness = 0.0;
    s.surfaces.insert(
        2,
        Surface::new(10, SurfaceShape::sphere(50.0), 6.0, glass()),
    );
    s.surfaces.insert(
        3,
        Surface::new(11, SurfaceShape::sphere(-50.0), 50.0, MaterialRef::Air),
    );
    s.validate().unwrap();
    assert_eq!(s.surfaces.len(), 5);
    let spec = DefaultMeritSpec::default().generate(&s).unwrap();
    let cons: Vec<&Operand> = spec
        .operands
        .iter()
        .filter(|o| o.role == OperandRole::Constraint)
        .collect();
    assert_eq!(cons.len(), 2, "glass min + max centre only: {cons:?}");
    assert!(cons.iter().all(|o| matches!(
        o.kind,
        OperandKind::CentreThickness {
            surface: SurfaceId(10),
            to: None
        }
    )));
    // Optimise the lens thickness with just the boundaries + a pull to 0.5:
    // feasible, ends on the 1 mm glass minimum.
    let mut ops: Vec<Operand> = cons.into_iter().cloned().collect();
    ops.push(
        Operand::new(OperandKind::CentreThickness {
            surface: SurfaceId(10),
            to: None,
        })
        .target(0.5),
    );
    let p = OpticalProblem::new(
        s,
        vec![VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(10),
        })],
        MeritFunctionSpec::new(ops),
        ModelEvaluator,
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let r = least_squares(&p, &[6.0], &LsqSettings::default(), &Monitor::default()).unwrap();
    assert!(r.termination.is_converged(), "{:?}", r.termination);
    assert!(r.violation < 1e-8 && (r.x[0] - 1.0).abs() < 1e-6, "{r:?}");
}

#[test]
fn gaps_after_a_mirror_are_positive() {
    // OBJ | M1 (mirror, t = −30) | M2 (mirror, t = +25) | IMG.
    let mut s = OpticalSystem::new_empty();
    let mut m1 = Surface::new(10, SurfaceShape::sphere(-200.0), -30.0, MaterialRef::Air);
    m1.interaction = Interaction::Reflect;
    m1.semi_diameter = SemiDiameter::Fixed { value: 20.0 };
    let mut m2 = Surface::new(11, SurfaceShape::sphere(-80.0), 25.0, MaterialRef::Air);
    m2.interaction = Interaction::Reflect;
    m2.semi_diameter = SemiDiameter::Fixed { value: 10.0 };
    s.surfaces[1] = m1;
    s.surfaces.insert(2, m2);
    s.surfaces[3].id = SurfaceId(12);
    s.stop = 1;
    let sag = |r: f64, h: f64| r.signum() * (r.abs() - (r * r - h * h).sqrt());
    let v = values(
        &s,
        vec![
            OperandKind::CentreThickness {
                surface: SurfaceId(10),
                to: None,
            },
            OperandKind::EdgeThickness {
                surface: SurfaceId(10),
                to: None,
                height: None,
            },
        ],
    );
    assert_eq!(v[0], Ok(30.0));
    let edge = -(-30.0 - sag(-200.0, 20.0) + sag(-80.0, 20.0));
    assert!((v[1].clone().unwrap() - edge).abs() < 1e-12, "{v:?}");
    assert!(edge > 0.0);
}

// ---------- half pupil, ray groups and validation ----------

fn axial_system() -> OpticalSystem {
    let mut s = two_singlets();
    s.fields.kind = FieldKind::ObjectAngle;
    s.fields.fields = vec![Field::new(0.0, 0.0), Field::new(0.0, 10.0)];
    s
}

fn ray_groups(spec: &MeritFunctionSpec) -> Vec<RayGroup> {
    spec.operands
        .iter()
        .filter_map(|o| o.kind.ray_group().copied())
        .collect()
}

#[test]
fn half_pupil_only_when_every_configuration_and_variable_allows_it() {
    let s = axial_system();
    let gen_ = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    };
    let spec = gen_.generate(&s).unwrap();
    let groups = ray_groups(&spec);
    assert_eq!(groups.len(), spec.operands.len());
    assert!(
        groups
            .iter()
            .all(|g| g.sampling.symmetry == PupilSymmetry::Bilateral)
    );
    // Every ray operand carries the same, complete group.
    assert!(groups.windows(2).all(|w| w[0] == w[1]));
    if let OperandKind::RayAberration { px, .. } = spec.operands[0].kind {
        assert!(px >= 0.0);
    }

    // A configuration with an x decenter → full pupil for every config.
    let mut s2 = s.clone();
    s2.configurations = vec![
        Configuration::default(),
        Configuration {
            name: "decentered".into(),
            overrides: vec![(
                ParamPath::DecenterX {
                    surface: SurfaceId(12),
                },
                0.05,
            )],
        },
    ];
    let both = DefaultMeritSpec {
        configs: vec![0, 1],
        thickness: None,
        ..Default::default()
    };
    assert!(
        ray_groups(&both.generate(&s2).unwrap())
            .iter()
            .all(|g| g.sampling.symmetry == PupilSymmetry::None)
    );
    // Config 0 alone is still symmetric.
    assert!(!both.half_pupil_allowed(&s2, &[]).unwrap());
    let only0 = DefaultMeritSpec {
        configs: vec![0],
        ..both.clone()
    };
    assert!(only0.half_pupil_allowed(&s2, &[]).unwrap());

    // A variable that can tilt about y → full pupil.
    let tilt = VariableSpec::new(ParamPath::TiltY {
        surface: SurfaceId(12),
    });
    let spec = gen_.generate_for(&s, &[tilt]).unwrap();
    assert!(
        ray_groups(&spec)
            .iter()
            .all(|g| g.sampling.symmetry == PupilSymmetry::None)
    );
    // A curvature variable keeps the half pupil.
    let curv = VariableSpec::new(ParamPath::Curvature {
        surface: SurfaceId(12),
    });
    assert!(gen_.half_pupil_allowed(&s, &[curv]).unwrap());
    // An identity placement does not break symmetry, a real tilt does.
    let mut s3 = s.clone();
    s3.surfaces[4].placement = Some(Placement::default());
    assert!(gen_.half_pupil_allowed(&s3, &[]).unwrap());
    s3.surfaces[4].placement = Some(Placement {
        tilt_y: 0.1,
        ..Default::default()
    });
    assert!(!gen_.half_pupil_allowed(&s3, &[]).unwrap());
}

#[test]
fn chief_reference_needs_no_group_and_centroid_requires_one() {
    let reg = OperandRegistry::builtin();
    let ray = |reference, group| OperandKind::RayAberration {
        axis: Axis::Y,
        px: 0.0,
        py: 1.0,
        reference,
        group,
    };
    let ok = MeritFunctionSpec::new(vec![
        Operand::new(ray(SpotReference::Chief, None)),
        Operand::new(ray(
            SpotReference::Centroid,
            Some(RayGroup::new(PupilSampling::default())),
        )),
    ]);
    ok.validate(&reg).unwrap();
    let missing = MeritFunctionSpec::new(vec![Operand::new(ray(SpotReference::Centroid, None))]);
    assert!(missing.validate(&reg).is_err());
    let wf = MeritFunctionSpec::new(vec![Operand::new(OperandKind::WavefrontAberration {
        px: 0.0,
        py: 1.0,
        reference: WavefrontReference::Centroid,
        group: None,
    })]);
    assert!(wf.validate(&reg).is_err());
    // Chief-referenced default merit emits no groups.
    let spec = DefaultMeritSpec {
        criterion: mklight_optim::merit::default::DefaultCriterion::RmsSpot {
            reference: SpotReference::Chief,
            scope: Default::default(),
        },
        thickness: None,
        ..Default::default()
    }
    .generate(&axial_system())
    .unwrap();
    assert!(ray_groups(&spec).is_empty());
    spec.validate(&reg).unwrap();
}

#[test]
fn selectors_and_custom_names_are_validated() {
    let reg = OperandRegistry::builtin();
    let ray = OperandKind::RealRay {
        quantity: mklight_optim::merit::operand::RayQuantity::Y,
        surface: None,
        px: 0.0,
        py: 1.0,
        global: false,
    };
    // Per-ray kinds default to field 0, and `All` is rejected.
    let op = Operand::new(ray.clone());
    assert_eq!(op.field, FieldSelector::Index { index: 0 });
    MeritFunctionSpec::new(vec![op.clone()])
        .validate(&reg)
        .unwrap();
    let mut all = op.clone();
    all.field = FieldSelector::All;
    assert!(MeritFunctionSpec::new(vec![all]).validate(&reg).is_err());
    let mut poly = op;
    poly.wavelength = mklight_optim::merit::WavelengthSelector::All;
    assert!(MeritFunctionSpec::new(vec![poly]).validate(&reg).is_err());
    // A field choice on a field-independent kind is a mistake.
    let efl = Operand::new(OperandKind::Efl).at_field(1);
    assert!(MeritFunctionSpec::new(vec![efl]).validate(&reg).is_err());
    // `Custom` must name a plug-in, never a built-in.
    let fake = MeritFunctionSpec::new(vec![Operand::new(OperandKind::Custom {
        name: "efl".into(),
        params: serde_json::Value::Null,
    })]);
    assert!(fake.validate(&reg).is_err());
    // Indices are checked against the system.
    let s = axial_system();
    let far = MeritFunctionSpec::new(vec![Operand::new(ray).at_field(5)]);
    far.validate(&reg).unwrap();
    assert!(far.validate_for(&s).is_err());
    let cfg = MeritFunctionSpec::new(vec![Operand::new(OperandKind::Efl).in_config(2)]);
    assert!(cfg.validate_for(&s).is_err());
}

// ---------- parametric evaluation ----------

/// Supports `Efl` = 10·(thickness of surface 10) and evaluates it straight
/// from the parameter vectors, never building systems.
struct Packed {
    calls: AtomicUsize,
}

impl OperandEvaluator for Packed {
    fn supports(&self, k: &OperandKind) -> bool {
        matches!(k, OperandKind::Efl)
    }
    fn evaluate_batch(
        &self,
        _: &[OpticalSystem],
        _: &[&Operand],
        _: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        panic!("the parametric path must be used");
    }
    fn evaluate_parametric(
        &self,
        b: &ParametricBatch<'_>,
        ops: &[&Operand],
        _: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, BuildError>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(b.params.len(), 1);
        Ok((0..b.len())
            .map(|k| {
                let t = b.candidates.point(k)[0];
                if t < 0.0 {
                    Err(BuildError::Parameters("negative".into()))
                } else {
                    Ok(ops.iter().map(|_| Ok(10.0 * t)).collect())
                }
            })
            .collect())
    }
}

#[test]
fn optical_problem_uses_one_parametric_call_per_configuration() {
    let s = two_singlets();
    let ev = CompositeEvaluator::new().with(ModelEvaluator).with(Packed {
        calls: AtomicUsize::new(0),
    });
    let p = OpticalProblem::new(
        s,
        vec![VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(10),
        })],
        MeritFunctionSpec::new(vec![
            Operand::new(OperandKind::Efl).target(50.0),
            Operand::new(OperandKind::CentreThickness {
                surface: SurfaceId(10),
                to: None,
            }),
        ]),
        ev,
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let r = least_squares(&p, &[4.0], &LsqSettings::default(), &Monitor::default()).unwrap();
    assert!(r.termination.is_converged(), "{:?}", r.termination);
    // min (10t − 50)² + t²  →  t = 500/101.
    assert!((r.x[0] - 500.0 / 101.0).abs() < 1e-8, "{:?}", r.x);
    let ctx = JobContext::new();
    let e = evaluate_one(&p, &[-1.0], &ctx).unwrap();
    assert_eq!(e.status, EvalStatus::Failed);
}
