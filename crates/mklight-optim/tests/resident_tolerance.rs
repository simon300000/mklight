#![cfg(target_os = "macos")]
//! Monte Carlo tolerancing on the resident evaluator (host Metal device)
//! against the reference, with and without compensation, and the host's
//! trial statistics.

mod common;

use mklight_core::{JobContext, Snapshot, param::ParamPath, samples};
use mklight_optim::{
    Monitor, gpu::*, lsq::LsqSettings, merit::default::DefaultMeritSpec, merit::*, tolerance::*,
};
#[test]
fn trial_replay_and_compensation() {
    let Some(b) = common::metal() else {
        return;
    };
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
            lsq,
        };
        let cpu = OpticalToleranceEvaluator {
            nominal: &system,
            spec: &spec,
            criterion: &criterion,
            evaluator: &reference,
            registry: &registry,
            lsq,
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
        assert_eq!(got.criteria.len(), expected.criteria.len());
        let mut relative = 0_f64;
        for (a, b) in got.criteria.iter().zip(&expected.criteria) {
            relative = relative.max((a - b).abs() / b.abs().max(1e-12));
        }
        eprintln!("tolerance compensated={compensated} criterion_relative={relative:e}");
        // Measured 2e-13 (compensated trials end on the f64 reference).
        assert!(relative < 1e-9, "{compensated}: {relative}");
        assert_eq!(got.records.len(), replay.records.len());
        for (a, b) in got.records.iter().zip(&replay.records) {
            assert_eq!(a.perturbations, b.perturbations);
        }
        eprintln!(
            "tolerance compensated={compensated} stats={:?}",
            gpu.stats()
        );
    }
    assert!(gpu.stats().device_operands > 0);
}

/// Trial summaries stay on the host (a device summary measured 5–50× slower
/// than the f64 reference it must agree with); large offsets must not cost
/// the spread its precision.
struct KnownTrials;
impl ToleranceEvaluator for KnownTrials {
    fn evaluate_batch(
        &self,
        p: &[Vec<f64>],
        _: &JobContext,
    ) -> mklight_core::MkResult<Vec<TrialOutcome>> {
        Ok(p.iter()
            .map(|_| TrialOutcome {
                criterion: 1e10,
                compensators: vec![0.],
                ok: true,
            })
            .collect())
    }
    fn evaluate_trials(
        &self,
        p: &[Vec<f64>],
        _: &MonteCarloSettings,
        first: u64,
        _: &JobContext,
    ) -> mklight_core::MkResult<Vec<TrialOutcome>> {
        Ok(p.iter()
            .enumerate()
            .map(|(i, _)| {
                let d = (2. * ((first + i as u64) % 4) as f64 - 3.) / 1024.;
                TrialOutcome {
                    criterion: 1e10 + d,
                    compensators: vec![d],
                    ok: true,
                }
            })
            .collect())
    }
}
#[test]
fn large_trial_statistics_have_known_variance_and_percentiles() {
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(
        ParamPath::Thickness {
            surface: mklight_core::SurfaceId(1),
        },
        0.1,
    )]);
    for n in [1000, 100000] {
        let ctx = JobContext::new();
        let settings = MonteCarloSettings {
            trials: n,
            batch: 256,
            percentiles: vec![0., 50., 100.],
            ..Default::default()
        };
        let r = monte_carlo(&spec, &KnownTrials, &settings, &Monitor::new(ctx.clone())).unwrap();
        let sd = (5. / 1048576. * n as f64 / (n - 1) as f64).sqrt();
        assert_eq!(r.mean, 1e10);
        assert!((r.std_dev - sd).abs() < 1e-12);
        assert!((r.compensator_std[0] - sd).abs() < 1e-12);
        assert_eq!(
            r.percentiles,
            vec![
                (0., 1e10 - 3. / 1024.),
                (50., 1e10),
                (100., 1e10 + 3. / 1024.)
            ]
        );
        assert_eq!(ctx.progress.fraction(), 1.);
    }
}
