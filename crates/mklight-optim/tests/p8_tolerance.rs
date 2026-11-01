#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, param::ParamPath, samples};
use mklight_optim::{
    Monitor, gpu::*, lsq::LsqSettings, merit::default::DefaultMeritSpec, merit::*, tolerance::*,
};
use std::sync::Arc;
#[test]
fn trial_replay_compensation_and_refinements() {
    let b = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    let system = samples::singlet();
    let ctx = JobContext::new();
    let reference = BuiltinEvaluator::builtin();
    let gpu = ResidentEvaluator::new(
        GpuContext::new(b, Snapshot::new(system.clone()).unwrap()),
        reference.clone(),
    );
    let registry = OperandRegistry::builtin();
    let criterion = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate(&system)
    .unwrap();
    let curvature = ParamPath::Curvature {
        surface: system.surfaces[1].id,
    };
    let focus = ParamPath::Thickness {
        surface: system.surfaces[2].id,
    };
    let mut spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(curvature, 1e-5)]);
    let settings = MonteCarloSettings {
        trials: 8,
        batch: 4,
        seed: 73,
        record_trials: 8,
        ..Default::default()
    };
    let lsq = LsqSettings {
        max_iterations: 8,
        ..Default::default()
    };
    for compensated in [false, true] {
        spec.compensators = if compensated {
            vec![Compensator {
                param: focus.clone(),
                range: Some(1.),
                scale: Some(0.1),
            }]
        } else {
            vec![]
        };
        let evaluator = OpticalToleranceEvaluator {
            nominal: &system,
            spec: &spec,
            criterion: &criterion,
            evaluator: &gpu,
            registry: &registry,
            lsq: lsq.clone(),
        };
        let cpu = OpticalToleranceEvaluator {
            nominal: &system,
            spec: &spec,
            criterion: &criterion,
            evaluator: &reference,
            registry: &registry,
            lsq: lsq.clone(),
        };
        let got = monte_carlo(&spec, &evaluator, &settings, &Monitor::new(ctx.clone())).unwrap();
        let expected = monte_carlo(&spec, &cpu, &settings, &Monitor::new(ctx.clone())).unwrap();
        let replay = monte_carlo(
            &spec,
            &evaluator,
            &MonteCarloSettings {
                batch: 3,
                ..settings.clone()
            },
            &Monitor::new(ctx.clone()),
        )
        .unwrap();
        assert_eq!(got.criteria, replay.criteria);
        assert_eq!(got.failures, expected.failures);
        for (a, b) in got.criteria.iter().zip(&expected.criteria) {
            assert!(
                (a - b).abs() < 1e-8 + 0.01 * b.abs(),
                "{compensated}: {a} vs {b}"
            );
        }
        for (a, b) in got.records.iter().zip(&replay.records) {
            assert_eq!(a.perturbations, b.perturbations);
        }
        eprintln!(
            "P8 tolerance compensated={compensated} stats={:?}",
            gpu.stats()
        );
    }
    let mut alternative = system.clone();
    alternative.surfaces[1].material = mklight_core::MaterialRef::Model {
        nd: 1.6,
        vd: 60.,
        dpgf: 0.,
    };
    let systems = vec![system.clone(), alternative];
    let vars = vec![VariableSpec::bounded(focus, 40., 60.)];
    let results = gpu
        .refine_candidates(&systems, &vars, &criterion, &lsq, &ctx)
        .unwrap();
    assert_eq!(results.len(), 2);
    for (s, result) in systems.iter().zip(&results) {
        let problem = OpticalProblem::new(
            s.clone(),
            vars.clone(),
            criterion.clone(),
            reference.clone(),
            &registry,
        )
        .unwrap();
        let e = mklight_optim::problem::evaluate_one(&problem, &result.x, &ctx).unwrap();
        let merit = mklight_optim::problem::merit(
            mklight_optim::problem::Problem::weights(&problem),
            &e.residuals,
            e.status,
        );
        assert_eq!(result.merit, merit);
    }
    assert!(gpu.stats().device_operands > 0);
}
