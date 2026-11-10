//! `WithSolves`: the solves of a model applied to every candidate an
//! evaluator sees, and through it to every case of a tolerance run.

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, OpticalSystem, SurfaceId, samples};
use mklight_optim::Monitor;
use mklight_optim::lsq::LsqSettings;
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::{
    MeritFunctionSpec, Operand, OperandError, OperandEvaluator, OperandKind, OperandRegistry,
    OperandValue, TraceEvaluator, WithSolves,
};
use mklight_optim::tolerance::{
    Compensator, MonteCarloSettings, OpticalToleranceEvaluator, ToleranceEvaluator,
    ToleranceOperand, ToleranceSpec, monte_carlo, sensitivity,
};
use mklight_trace::apply_solves;

fn library() -> Arc<MaterialLibrary> {
    Arc::new(MaterialLibrary::with_builtin())
}

fn thickness(id: u64) -> ParamPath {
    ParamPath::Thickness {
        surface: SurfaceId(id),
    }
}

/// The Cooke triplet with its image at the paraxial focus by a solve; the
/// stored back focus is the sample's best focus.
fn focused_triplet() -> OpticalSystem {
    let mut sys = samples::cooke_triplet();
    sys.solves = serde_json::from_value(serde_json::json!([
        {"target": thickness(6), "kind": {"type": "marginal_ray_height", "height": 0.0}},
    ]))
    .unwrap();
    sys
}

fn solved(sys: &OpticalSystem) -> OpticalSystem {
    apply_solves(sys, &library()).unwrap().0
}

fn spot_merit(sys: &OpticalSystem) -> MeritFunctionSpec {
    DefaultMeritSpec {
        thickness: None,
        fields: Some(vec![0, 2]),
        wavelengths: Some(vec![1]),
        ..Default::default()
    }
    .generate(sys)
    .unwrap()
}

fn bits(values: &[Vec<OperandValue>]) -> Vec<Result<u64, OperandError>> {
    values
        .iter()
        .flatten()
        .map(|v| v.clone().map(f64::to_bits))
        .collect()
}

/// Every candidate is evaluated with its solves applied; one whose solves
/// fail has no values; a system without solves passes through; a cancelled
/// run is an error.
#[test]
fn candidates_are_solved_before_they_are_evaluated() {
    let inner = TraceEvaluator::builtin();
    let ev = WithSolves::new(&inner, library());
    let ctx = JobContext::new();
    let sys = focused_triplet();
    let mut moved = sys.clone();
    thickness(2).set(&mut moved, 6.1).unwrap();
    // A collimated beam stays collimated behind a plane: no thickness
    // there puts its marginal ray on the axis.
    let mut unsolvable = samples::singlet();
    unsolvable.surfaces[1].shape = mklight_core::SurfaceShape::Plane;
    unsolvable.solves = serde_json::from_value(serde_json::json!([
        {"target": thickness(1), "kind": {"type": "marginal_ray_height", "height": 0.0}},
    ]))
    .unwrap();
    assert!(apply_solves(&unsolvable, &library()).is_err());
    let spec = spot_merit(&sys);
    let mut ops: Vec<&Operand> = spec.operands.iter().collect();
    let bfl = Operand::new(OperandKind::Bfl);
    ops.push(&bfl);
    let candidates = [sys.clone(), unsolvable, moved.clone()];
    let values = ev.evaluate_batch(&candidates, &ops, &ctx).unwrap();
    let direct = inner
        .evaluate_batch(&[solved(&sys), solved(&moved)], &ops, &ctx)
        .unwrap();
    assert_eq!(bits(&values[..1]), bits(&direct[..1]));
    assert_eq!(bits(&values[2..]), bits(&direct[1..]));
    assert!(
        values[1]
            .iter()
            .all(|v| matches!(v, Err(OperandError::Invalid(_)))),
        "{:?}",
        values[1]
    );
    // The stored (unsolved) systems give other values.
    let stored = inner
        .evaluate_batch(std::slice::from_ref(&sys), &ops, &ctx)
        .unwrap();
    assert_ne!(bits(&stored), bits(&values[..1]));
    // The candidate is what `solved` returns.
    assert_eq!(ev.solved(&moved, &ctx).unwrap().unwrap(), solved(&moved));
    // Without solves, the inner evaluator's values.
    let plain = samples::cooke_triplet();
    let plain = std::slice::from_ref(&plain);
    assert_eq!(
        bits(&ev.evaluate_batch(plain, &ops, &ctx).unwrap()),
        bits(&inner.evaluate_batch(plain, &ops, &ctx).unwrap())
    );
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        ev.evaluate_batch(&candidates, &ops, &cancelled),
        Err(MkError::Cancelled)
    ));
}

/// A tolerance run on such an evaluator re-solves every case: the nominal,
/// the sensitivity limits, the Monte Carlo trials and, with a compensator,
/// the candidates of its solve. Each criterion is the merit of the solved
/// case.
#[test]
fn tolerance_cases_are_re_solved() {
    let inner = TraceEvaluator::builtin();
    let ev = WithSolves::new(&inner, library());
    let reg = OperandRegistry::builtin();
    let nominal = solved(&focused_triplet());
    let criterion = spot_merit(&nominal);
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(thickness(1), 0.05),
        ToleranceOperand::symmetric(thickness(4), 0.05),
    ]);
    let ctx = JobContext::new();
    // The merit of a system as it stands (its solves applied by hand).
    let merit = |sys: &OpticalSystem| {
        let te = OpticalToleranceEvaluator {
            nominal: sys,
            spec: &ToleranceSpec::new(Vec::new()),
            criterion: &criterion,
            evaluator: &inner,
            registry: &reg,
            lsq: LsqSettings::default(),
        };
        te.evaluate_batch(&[vec![]], &ctx).unwrap()[0].criterion
    };
    let mc = MonteCarloSettings {
        trials: 3,
        seed: 9,
        ..Default::default()
    };
    for compensated in [false, true] {
        if compensated {
            spec.compensators = vec![Compensator {
                param: thickness(2),
                range: Some(0.5),
                scale: None,
            }];
        }
        let te = OpticalToleranceEvaluator {
            nominal: &nominal,
            spec: &spec,
            criterion: &criterion,
            evaluator: &ev,
            registry: &reg,
            lsq: LsqSettings::default(),
        };
        let unsolved = OpticalToleranceEvaluator {
            nominal: &nominal,
            spec: &spec,
            criterion: &criterion,
            evaluator: &inner,
            registry: &reg,
            lsq: LsqSettings::default(),
        };
        let s = sensitivity(&spec, &te, &Monitor::default()).unwrap();
        let frozen = sensitivity(&spec, &unsolved, &Monitor::default()).unwrap();
        let limit = &s.operands[0].plus;
        let case = spec
            .case_system(&nominal, &[0.05, 0.0], &limit.compensators)
            .unwrap();
        let refocused = solved(&case);
        let shift = thickness(6).get(&refocused).unwrap() - thickness(6).get(&case).unwrap();
        assert!(shift.abs() > 1e-4, "the focus follows: {shift}");
        assert_eq!(limit.criterion.to_bits(), merit(&refocused).to_bits());
        // Without the solves the image stays where the nominal has it.
        assert_ne!(limit.criterion, frozen.operands[0].plus.criterion);
        let r = monte_carlo(&spec, &te, &mc, &Monitor::default()).unwrap();
        assert_eq!(r.failures, 0);
        assert_eq!(r.nominal.criterion.to_bits(), s.nominal.criterion.to_bits());
        for trial in 0..mc.trials {
            let deltas = spec.sample_trial(mc.seed, trial as u64);
            let outcome = te
                .evaluate_trials(std::slice::from_ref(&deltas), &mc, trial as u64, &ctx)
                .unwrap()
                .remove(0);
            assert_eq!(outcome.criterion.to_bits(), r.criteria[trial].to_bits());
            let case = spec
                .trial_case_system(&nominal, &deltas, &outcome.compensators, &mc, trial as u64)
                .unwrap();
            assert_eq!(outcome.criterion.to_bits(), merit(&solved(&case)).to_bits());
        }
    }
}
