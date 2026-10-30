//! End-to-end optimisation and tolerancing on real lenses with the
//! tracer-backed evaluator.

use mklight_core::job::RunTimer;
use mklight_core::model::{ApertureSpec, OpticalSystem, SurfaceShape};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, SurfaceId, samples};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultMeritSpec, ThicknessBoundaries};
use mklight_optim::merit::operand::OperandKind;
use mklight_optim::merit::{
    Operand, OperandRegistry, OpticalProblem, TraceEvaluator, VariableSpec,
};
use mklight_optim::problem::{EvalStatus, Problem, evaluate_one, merit, total_violation};
use mklight_optim::tolerance::{
    Compensator, MonteCarloSettings, OpticalToleranceEvaluator, ToleranceOperand, ToleranceSpec,
    monte_carlo, sensitivity,
};
use mklight_optim::{Monitor, Termination};

fn curv(id: u64) -> VariableSpec {
    VariableSpec::new(ParamPath::Curvature {
        surface: SurfaceId(id),
    })
}
fn thick(id: u64) -> VariableSpec {
    VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(id),
    })
}

/// Merit and violation of `x`.
fn score<P: Problem>(p: &P, x: &[f64]) -> (f64, f64) {
    let e = evaluate_one(p, x, &JobContext::new()).unwrap();
    (
        merit(p.weights(), &e.residuals, e.status),
        total_violation(p.constraints(), &e.constraints, e.status),
    )
}

fn value(sys: &OpticalSystem, kind: OperandKind) -> f64 {
    let op = Operand::new(kind);
    TraceEvaluator::builtin()
        .evaluate_system(sys, &[&op], &JobContext::new())
        .unwrap()
        .remove(0)
        .unwrap()
}

/// (a) Singlet bending for minimum on-axis RMS spot at fixed EFL converges to
/// the best-form shape factor of thin-lens theory,
/// `q = (c₁ + c₂)/(c₁ − c₂) = 2(n² − 1)/(n + 2)` (object at infinity).
#[test]
fn singlet_converges_to_best_form_bending() {
    let mut sys = samples::singlet();
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    // Start from an equi-convex lens.
    sys.surfaces[1].shape = SurfaceShape::sphere(100.0);
    sys.surfaces[2].shape = SurfaceShape::sphere(-100.0);
    sys.surfaces[1].thickness = 3.0;
    let ev = TraceEvaluator::builtin();
    let n = ev.tracer(&sys).unwrap().system().surfaces[1].n_after[1];
    let vars = vec![curv(1), curv(2), thick(2)];
    let mut spec = DefaultMeritSpec {
        fields: Some(vec![0]),
        wavelengths: Some(vec![1]),
        thickness: None,
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    spec.operands
        .push(Operand::new(OperandKind::Efl).target(100.0).as_constraint());
    let p = OpticalProblem::new(sys, vars, spec, ev, &OperandRegistry::builtin()).unwrap();
    let x0 = p.start_point().unwrap();
    let (m0, _) = score(&p, &x0);
    let r = least_squares(&p, &x0, &LsqSettings::default(), &Monitor::default()).unwrap();
    assert!(r.termination.is_converged(), "{:?}", r.termination);
    assert!(r.violation < 1e-8, "{r:?}");
    let (c1, c2) = (r.x[0], r.x[1]);
    let q = (c1 + c2) / (c1 - c2);
    let q_thin = 2.0 * (n * n - 1.0) / (n + 2.0);
    eprintln!("singlet: q = {q}, thin-lens {q_thin}, merit {m0} -> {}", r.merit);
    assert!(
        (q - q_thin).abs() < 0.03,
        "q = {q}, thin-lens best form {q_thin}"
    );
    assert!(r.merit < 0.2 * m0, "{} → {}", m0, r.merit);
    let best = p.apply(&r.x).unwrap();
    assert!((value(&best, OperandKind::Efl) - 100.0).abs() < 1e-6);
    // Bending away from the optimum at the same EFL is worse.
    for dq in [-0.3, 0.3] {
        let q2 = q + dq;
        // Keep (n − 1)(c₁ − c₂) (thin-lens power) and change the shape.
        let d = c1 - c2;
        let (a1, a2) = (0.5 * d * (q2 + 1.0), 0.5 * d * (q2 - 1.0));
        let (m, _) = score(&p, &[a1, a2, r.x[2]]);
        assert!(m > r.merit, "q = {q2}: {m} vs {}", r.merit);
    }
}

/// (b) Achromatise a cemented doublet: equal paraxial focus in F and C at a
/// held EFL.
#[test]
fn doublet_is_achromatised_at_fixed_efl() {
    let mut sys = samples::achromatic_doublet();
    sys.surfaces[2].shape = SurfaceShape::sphere(-60.0);
    sys.surfaces[3].shape = SurfaceShape::sphere(-300.0);
    let colour = OperandKind::AxialColour { short: 0, long: 2 };
    let shift0 = value(&sys, colour.clone());
    assert!(shift0.abs() > 0.5, "the start is not achromatic: {shift0}");
    let vars = vec![curv(2), curv(3), thick(3)];
    let mut spec = DefaultMeritSpec {
        fields: Some(vec![0]),
        thickness: None,
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    spec.operands.extend([
        Operand::new(OperandKind::Efl).target(100.0).as_constraint(),
        Operand::new(colour.clone()).target(0.0).as_constraint(),
    ]);
    let p = OpticalProblem::new(
        sys,
        vars,
        spec,
        TraceEvaluator::builtin(),
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let x0 = p.start_point().unwrap();
    let r = least_squares(&p, &x0, &LsqSettings::default(), &Monitor::default()).unwrap();
    assert!(r.violation < 1e-8, "{:?} {}", r.termination, r.violation);
    let best = p.apply(&r.x).unwrap();
    let shift = value(&best, colour);
    let efl = value(&best, OperandKind::Efl);
    assert!(shift.abs() < 1e-6, "chromatic focal shift {shift}");
    assert!((efl - 100.0).abs() < 1e-6, "EFL {efl}");
    // The solution is the classic N-BK7/F2 achromat (R₂ ≈ −42.5 mm).
    assert!((1.0 / r.x[0] + 42.5).abs() < 3.0, "R2 = {}", 1.0 / r.x[0]);
    assert!(r.multipliers.as_ref().is_some_and(|m| m.len() == 2));
}

fn triplet_problem() -> OpticalProblem<TraceEvaluator> {
    triplet_problem_with(true)
}

fn triplet_problem_with(constrained: bool) -> OpticalProblem<TraceEvaluator> {
    let sys = samples::cooke_triplet();
    let efl = value(&sys, OperandKind::Efl);
    let mut vars: Vec<VariableSpec> = (1..=6).map(curv).collect();
    vars.push(thick(6));
    // The textbook triplet has a 1.0 mm flint and thin positive edges
    // (≈ 0.6 mm on L3, sized by the automatic semi-diameters): bounds the
    // start satisfies.
    let bounds = ThicknessBoundaries {
        glass_min_centre: Some(0.9),
        glass_min_edge: Some(0.5),
        ..ThicknessBoundaries::with_auto_semi_diameters()
    };
    let mut spec = DefaultMeritSpec {
        thickness: constrained.then_some(bounds),
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    if constrained {
        spec.operands
            .push(Operand::new(OperandKind::Efl).target(efl).as_constraint());
    }
    OpticalProblem::new(
        sys,
        vars,
        spec,
        TraceEvaluator::builtin(),
        &OperandRegistry::builtin(),
    )
    .unwrap()
}

/// (c) Cooke triplet: LM improves the default RMS-spot merit over all fields
/// and wavelengths, then a short seeded basin-hopping run is reproducible.
#[test]
fn cooke_triplet_local_then_global() {
    let p = triplet_problem();
    // Glass/air centre and edge bounds (edges from automatic semi-diameters)
    // plus the EFL constraint.
    assert!(p.num_constraints() > 10, "{}", p.num_constraints());
    let x0 = p.start_point().unwrap();
    let (m0, v0) = score(&p, &x0);
    assert!(v0 < 1e-12, "the sample starts feasible: {v0}");
    let lm = LsqSettings {
        max_iterations: 40,
        ..Default::default()
    };
    let r = least_squares(&p, &x0, &lm, &Monitor::default()).unwrap();
    eprintln!("triplet LM: {m0} -> {} ({:?}, {} evals)", r.merit, r.termination, r.evaluations);
    assert!(r.merit < 0.7 * m0, "LM: {m0} → {}", r.merit);
    assert!(r.violation < 1e-7, "{}", r.violation);
    let (m1, v1) = score(&p, &r.x);
    assert_eq!(m1, r.merit);
    assert!(v1 < 1e-7);

    let hs = HammerSettings {
        rounds: 2,
        candidates_per_round: 6,
        local_per_round: 1,
        seed: 1234,
        max_evaluations: 600,
        local: LsqSettings {
            max_iterations: 6,
            ..Default::default()
        },
        ..Default::default()
    };
    let timer = RunTimer::start();
    let g1 = hammer(&p, &r.x, &hs, &Monitor::default()).unwrap();
    let info = TraceEvaluator::builtin()
        .run_info(timer, p.base(), "optimise/hammer", &hs, Some(hs.seed))
        .unwrap();
    let g2 = hammer(&p, &r.x, &hs, &Monitor::default()).unwrap();
    eprintln!("triplet hammer: {} ({:?}, {} evals)", g1.merit, g1.termination, g1.evaluations);
    assert_eq!(g1.x, g2.x, "seeded global run is reproducible");
    assert_eq!(g1.merit.to_bits(), g2.merit.to_bits());
    assert_eq!(g1.evaluations, g2.evaluations);
    assert!(g1.merit <= r.merit * (1.0 + 1e-12), "{} vs {}", g1.merit, r.merit);
    assert!(g1.violation < 1e-7, "{}", g1.violation);
    assert_eq!(info.seed, Some(1234));
    assert_eq!(info.backend, TraceEvaluator::BACKEND);
    assert_eq!(
        info.snapshot_hash,
        mklight_core::Snapshot::new(p.base().clone())
            .unwrap()
            .hash()
            .to_hex()
    );
    // The optimised design still holds its focal length.
    let best = p.apply(&g1.x).unwrap();
    let efl0 = value(p.base(), OperandKind::Efl);
    assert!((value(&best, OperandKind::Efl) - efl0).abs() < 1e-5);
}

/// (d) Sensitivity with a back-focus compensator gives a ranked list, and
/// Monte Carlo is deterministic for a fixed seed.
#[test]
fn cooke_triplet_tolerancing() {
    let sys = samples::cooke_triplet();
    let criterion = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate(&sys)
    .unwrap();
    let t = |id| ParamPath::Thickness {
        surface: SurfaceId(id),
    };
    let c = |id| ParamPath::Curvature {
        surface: SurfaceId(id),
    };
    let curv_tol = |id: u64| {
        let cv = c(id).get(&sys).unwrap();
        ToleranceOperand::symmetric(c(id), 0.005 * cv.abs())
    };
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(1), 0.05),
        ToleranceOperand::symmetric(t(2), 0.05),
        ToleranceOperand::symmetric(t(3), 0.05),
        ToleranceOperand::symmetric(t(4), 0.05),
        ToleranceOperand::symmetric(t(5), 0.05),
        curv_tol(1),
        curv_tol(3),
        curv_tol(6),
        ToleranceOperand::symmetric(
            ParamPath::DecenterY {
                surface: SurfaceId(3),
            },
            0.02,
        ),
    ]);
    spec.compensators = vec![Compensator {
        param: t(6),
        range: Some(1.0),
        scale: None,
    }];
    let reg = OperandRegistry::builtin();
    let ev = TraceEvaluator::builtin();
    let lsq = LsqSettings {
        max_iterations: 20,
        ..Default::default()
    };
    let with = OpticalToleranceEvaluator {
        nominal: &sys,
        spec: &spec,
        criterion: &criterion,
        evaluator: &ev,
        registry: &reg,
        lsq,
    };
    let rep = sensitivity(&spec, &with, &Monitor::default()).unwrap();
    assert!(rep.nominal.ok);
    assert_eq!(rep.ranking.len(), spec.operands.len());
    for w in rep.ranking.windows(2) {
        assert!(rep.operands[w[0]].worst >= rep.operands[w[1]].worst);
    }
    assert!(rep.operands.iter().all(|o| o.minus.ok && o.plus.ok));
    assert!(rep.rss_change > 0.0);
    // Without the compensator every perturbation is at least as bad.
    let mut fixed = spec.clone();
    fixed.compensators.clear();
    let without = OpticalToleranceEvaluator {
        spec: &fixed,
        ..with
    };
    let rep0 = sensitivity(&fixed, &without, &Monitor::default()).unwrap();
    for (a, b) in rep.operands.iter().zip(&rep0.operands) {
        assert!(a.plus.criterion <= b.plus.criterion * (1.0 + 1e-9));
        assert!(a.minus.criterion <= b.minus.criterion * (1.0 + 1e-9));
    }
    // A thickness error ahead of focus is largely taken out by refocusing.
    let k = 3; // air space before the last element
    assert!(
        rep.operands[k].worst < 0.5 * rep0.operands[k].worst,
        "{} vs {}",
        rep.operands[k].worst,
        rep0.operands[k].worst
    );
    assert!(
        rep.operands[k].plus.compensators[0] != sys.surfaces[6].thickness,
        "the compensator moved"
    );

    let mc = MonteCarloSettings {
        trials: 12,
        seed: 99,
        batch: 5,
        threshold: Some(rep.nominal.criterion * 4.0),
    };
    let a = monte_carlo(&spec, &with, &mc, &Monitor::default()).unwrap();
    let b = monte_carlo(&spec, &with, &mc, &Monitor::default()).unwrap();
    assert_eq!(a.criteria, b.criteria);
    assert_eq!(a.mean.to_bits(), b.mean.to_bits());
    assert_eq!(a.compensator_mean, b.compensator_mean);
    assert_eq!(a.trials, 12);
    assert_eq!(a.failures, 0);
    assert!(a.mean >= rep.nominal.criterion * (1.0 - 1e-9));
    assert!(a.std_dev > 0.0 && a.min <= a.mean && a.mean <= a.max);
    assert!(a.yield_fraction.is_some());
    let other = monte_carlo(
        &spec,
        &with,
        &MonteCarloSettings { seed: 100, ..mc },
        &Monitor::default(),
    )
    .unwrap();
    assert_ne!(a.criteria, other.criteria);
}

/// (e) Cancelling mid-optimisation returns the best point so far.
#[test]
fn cancellation_mid_optimisation_returns_best_so_far() {
    let cancel_after = |n: usize| {
        let job = JobContext::new();
        let token = job.cancel.clone();
        Monitor::new(job).with_callback(move |rec| {
            if rec.iteration >= n {
                token.cancel();
            }
        })
    };
    // Unconstrained: every accepted step improves the best point.
    let p = triplet_problem_with(false);
    let x0 = p.start_point().unwrap();
    let (m0, _) = score(&p, &x0);
    let mon = cancel_after(2);
    let r = least_squares(&p, &x0, &LsqSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    assert!(r.iterations <= 4, "{}", r.iterations);
    assert!(r.merit.is_finite() && r.merit < m0, "{} vs {m0}", r.merit);
    let best = mon.best.get().expect("best-so-far point");
    assert!(best.merit <= r.merit && best.merit < m0, "{best:?} {}", r.merit);
    let (m, _) = score(&p, &best.x);
    assert_eq!(m, best.merit, "the best point re-evaluates to its merit");

    // Constrained: the best-so-far point is the best *feasible* one (here
    // possibly the start, while the augmented Lagrangian iterates are still
    // slightly infeasible).
    let p = triplet_problem();
    let x0 = p.start_point().unwrap();
    let (m0, _) = score(&p, &x0);
    let mon = cancel_after(2);
    let r = least_squares(&p, &x0, &LsqSettings::default(), &mon).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
    let best = mon.best.get().expect("best-so-far point");
    assert!(best.violation <= mon.feasibility_tol && best.merit <= m0);
    let (m, v) = score(&p, &best.x);
    assert!(m == best.merit && v <= mon.feasibility_tol);
    assert!(r.merit.is_finite());

    // A global run.
    let mon = cancel_after(1);
    let hs = HammerSettings {
        rounds: 10,
        candidates_per_round: 4,
        local_per_round: 1,
        local: LsqSettings {
            max_iterations: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let g = hammer(&p, &x0, &hs, &mon).unwrap();
    assert_eq!(g.termination, Termination::Cancelled);
    assert!(g.merit.is_finite() && g.merit <= m0);
    let e = evaluate_one(&p, &g.x, &JobContext::new()).unwrap();
    assert_ne!(e.status, EvalStatus::Failed);
}
