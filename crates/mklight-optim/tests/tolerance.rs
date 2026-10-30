//! Tolerance sampling, sensitivity and Monte Carlo.

use mklight_core::model::{MaterialRef, Surface, SurfaceShape};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, OpticalSystem, SurfaceId};
use mklight_optim::Monitor;
use mklight_optim::lsq::LsqSettings;
use mklight_optim::merit::geometric::ModelEvaluator;
use mklight_optim::merit::{MeritFunctionSpec, Operand, OperandKind, OperandRegistry};
use mklight_optim::tolerance::{
    Compensator, Distribution, FnToleranceEvaluator, MonteCarloSettings, OpticalToleranceEvaluator,
    ToleranceOperand, ToleranceSpec, TrialOutcome, monte_carlo, sensitivity,
};

fn sys() -> OpticalSystem {
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
    s.validate().unwrap();
    s
}

fn t(id: u64) -> ParamPath {
    ParamPath::Thickness {
        surface: SurfaceId(id),
    }
}

fn track_spec() -> MeritFunctionSpec {
    MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::TotalTrack {
            from: None,
            to: None,
        })
        .target(100.0),
    ])
}

#[test]
fn distributions_have_the_right_moments() {
    let a = 0.2;
    let n = 40_000;
    let mk = |d| {
        ToleranceSpec::new(vec![
            ToleranceOperand::symmetric(t(10), a).with_distribution(d),
        ])
    };
    let stats = |spec: &ToleranceSpec| {
        let xs: Vec<f64> = (0..n).map(|i| spec.sample_trial(5, i)[0]).collect();
        assert!(xs.iter().all(|x| x.abs() <= a + 1e-15));
        let m = xs.iter().sum::<f64>() / n as f64;
        let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        (m, v, xs)
    };
    let (m, v, _) = stats(&mk(Distribution::Uniform));
    assert!(
        m.abs() < 3e-3 && (v / (a * a / 3.0) - 1.0).abs() < 0.03,
        "{m} {v}"
    );
    // Truncated at ±2σ: variance (1 − 2kφ(k)/(2Φ(k) − 1)) σ², σ = a/2.
    let k: f64 = 2.0;
    let phi = (-0.5 * k * k).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi_2m1 = 0.954_499_736_103_642; // 2Φ(2) − 1
    let tv = (1.0 - 2.0 * k * phi / big_phi_2m1) * (a / k).powi(2);
    let (m, v, _) = stats(&mk(Distribution::TruncatedNormal { sigmas: 2.0 }));
    assert!(m.abs() < 3e-3 && (v / tv - 1.0).abs() < 0.03, "{v} vs {tv}");
    let (_, v, xs) = stats(&mk(Distribution::EndPoint));
    assert!(xs.iter().all(|x| x.abs() == a));
    assert!((v / (a * a) - 1.0).abs() < 0.01);
    // Each trial is reproducible on its own and independent of trial order.
    let s = mk(Distribution::Uniform);
    assert_eq!(s.sample_trial(5, 123), s.sample_trial(5, 123));
    assert_ne!(s.sample_trial(5, 123), s.sample_trial(6, 123));
}

#[test]
fn sensitivity_with_and_without_compensator() {
    let nominal = sys();
    let reg = OperandRegistry::builtin();
    let crit = track_spec();
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(10), 0.1),
        ToleranceOperand {
            param: t(11),
            minus: 0.05,
            plus: 0.3,
            distribution: Distribution::Uniform,
            label: "air".into(),
        },
    ]);
    let ev = ModelEvaluator;
    let te = OpticalToleranceEvaluator {
        nominal: &nominal,
        spec: &spec,
        criterion: &crit,
        evaluator: &ev,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let r = sensitivity(&spec, &te, &Monitor::default()).unwrap();
    assert_eq!(r.evaluations, 5);
    assert!(r.nominal.criterion.abs() < 1e-20);
    // Criterion = (Δtrack)²: 0.01 for ±0.1, 0.0025 / 0.09 for −0.05 / +0.3.
    assert!((r.operands[0].change_minus - 0.01).abs() < 1e-12);
    assert!((r.operands[0].change_plus - 0.01).abs() < 1e-12);
    assert!((r.operands[1].change_minus - 0.0025).abs() < 1e-12);
    assert!((r.operands[1].change_plus - 0.09).abs() < 1e-12);
    assert_eq!(r.ranking, vec![1, 0]);
    assert!((r.rss_change - (0.01f64.powi(2) + 0.09f64.powi(2)).sqrt()).abs() < 1e-12);

    // With the stop→image distance as a compensator the track is restored.
    spec.compensators = vec![Compensator {
        param: ParamPath::Thickness {
            surface: SurfaceId(1),
        },
        range: Some(1.0),
        scale: None,
    }];
    let te = OpticalToleranceEvaluator {
        nominal: &nominal,
        spec: &spec,
        criterion: &crit,
        evaluator: &ev,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let r = sensitivity(&spec, &te, &Monitor::default()).unwrap();
    for o in &r.operands {
        assert!(o.worst < 1e-20, "{o:?}");
    }
    assert!((r.operands[1].plus.compensators[0] - (92.0 - 0.3)).abs() < 1e-9);
}

#[test]
fn monte_carlo_statistics_and_determinism() {
    // Criterion (δ₁ + δ₂)² with δ uniform on ±a: E = 2a²/3.
    let a = 0.1;
    let spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(10), a),
        ToleranceOperand::symmetric(t(11), a),
    ]);
    let ev = FnToleranceEvaluator::new(|d| TrialOutcome {
        criterion: (d[0] + d[1]).powi(2),
        compensators: vec![d[0] - d[1]],
        ok: true,
    });
    let s = MonteCarloSettings {
        trials: 20_000,
        seed: 17,
        batch: 1000,
        threshold: Some(a * a),
    };
    let r = monte_carlo(&spec, &ev, &s, &Monitor::default()).unwrap();
    assert_eq!(r.trials, 20_000);
    assert_eq!(r.failures, 0);
    let e = 2.0 * a * a / 3.0;
    assert!((r.mean / e - 1.0).abs() < 0.02, "{} vs {e}", r.mean);
    // P((δ₁+δ₂)² ≤ a²) = P(|U₁+U₂| ≤ 1) for U uniform on ±1 = 3/4.
    assert!((r.yield_fraction.unwrap() - 0.75).abs() < 0.01);
    assert!(r.percentiles[0].1 <= r.percentiles[4].1);
    assert!(r.compensator_mean[0].abs() < 3e-3);
    let r2 = monte_carlo(&spec, &ev, &s, &Monitor::default()).unwrap();
    assert_eq!(r.criteria, r2.criteria);
    // Batch size does not change the trials.
    let r3 = monte_carlo(
        &spec,
        &ev,
        &MonteCarloSettings { batch: 7, ..s },
        &Monitor::default(),
    )
    .unwrap();
    assert_eq!(r.criteria, r3.criteria);
}

#[test]
fn monte_carlo_cancellation_returns_partial() {
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(t(10), 0.1)]);
    let ctx = JobContext::new();
    let token = ctx.cancel.clone();
    let ev = FnToleranceEvaluator::new(move |d| {
        if d[0] > 0.09 {
            token.cancel();
        }
        TrialOutcome {
            criterion: d[0] * d[0],
            compensators: vec![],
            ok: true,
        }
    });
    let s = MonteCarloSettings {
        trials: 100_000,
        batch: 10,
        ..Default::default()
    };
    let r = monte_carlo(&spec, &ev, &s, &Monitor::new(ctx)).unwrap();
    assert!(r.cancelled);
    assert!(r.trials < 100_000 && r.trials > 0);
}

#[test]
fn spec_serialises() {
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(10), 0.02)
            .with_distribution(Distribution::TruncatedNormal { sigmas: 3.0 }),
    ]);
    spec.compensators.push(Compensator {
        param: t(1),
        range: None,
        scale: Some(0.1),
    });
    spec.criterion = Some(track_spec());
    let js = serde_json::to_string(&spec).unwrap();
    let back: ToleranceSpec = serde_json::from_str(&js).unwrap();
    assert_eq!(back, spec);
    let applied = spec.apply(&sys(), &[0.02]).unwrap();
    assert!((applied.surfaces[1].thickness - 5.02).abs() < 1e-15);
}
