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

#[test]
fn index_and_abbe_tolerances_add_to_catalogue_offsets() {
    use mklight_core::materials::{MaterialLibrary, MaterialOffset, lines};
    let mut nominal = mklight_core::samples::singlet();
    nominal.surfaces[1].material = MaterialRef::Catalog {
        catalog: None,
        name: "N-BK7".into(),
        offset: MaterialOffset {
            dn_d: 0.001,
            dv_d: -1.0,
        },
    };
    let dn = ParamPath::MaterialDnD {
        surface: nominal.surfaces[1].id,
    };
    let dv = ParamPath::MaterialDvD {
        surface: nominal.surfaces[1].id,
    };
    let spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(dn.clone(), 0.002),
        ToleranceOperand::symmetric(dn.clone(), 0.001),
        ToleranceOperand::symmetric(dv.clone(), 2.0),
    ]);
    let s = spec.apply(&nominal, &[0.002, -0.001, 2.0]).unwrap();
    assert_eq!(dn.get(&s).unwrap(), 0.002);
    assert_eq!(dv.get(&s).unwrap(), 1.0);
    let lib = MaterialLibrary::with_builtin();
    let a = lib
        .resolve(&nominal.surfaces[1].material, &nominal.environment)
        .unwrap();
    let b = lib
        .resolve(&s.surfaces[1].material, &s.environment)
        .unwrap();
    assert!((b.index(lines::D) - a.index(lines::D) - 0.001).abs() < 1e-15);
    let vd = |m: &mklight_core::materials::ResolvedMaterial| {
        (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C))
    };
    assert!((vd(&b) - vd(&a) - 2.0).abs() < 4e-12);
    let json = serde_json::to_string(&spec).unwrap();
    let restored: ToleranceSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.apply(&nominal, &[0.002, -0.001, 2.0]).unwrap(), s);
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
        ..Default::default()
    };
    let r = monte_carlo(&spec, &ev, &s, &Monitor::default()).unwrap();
    assert_eq!(r.trials, 20_000);
    assert_eq!(r.failures, 0);
    let e = 2.0 * a * a / 3.0;
    assert!((r.mean / e - 1.0).abs() < 0.02, "{} vs {e}", r.mean);
    // P((δ₁+δ₂)² ≤ a²) = P(|U₁+U₂| ≤ 1) for U uniform on ±1 = 3/4.
    assert!((r.yield_fraction.unwrap() - 0.75).abs() < 0.01);
    assert!(r.percentiles[0].1 <= r.percentiles[2].1);
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

/// A sensitivity analysis cancelled after its first batch keeps the
/// operands whose both limits were evaluated, with the values of an
/// uncancelled run; cancelled before the nominal, it is an error.
#[test]
fn sensitivity_cancellation_keeps_completed_operands() {
    let n = 40;
    let spec = ToleranceSpec::new(
        (0..n)
            .map(|i| ToleranceOperand::symmetric(t(i as u64), 0.01 * (i + 1) as f64))
            .collect(),
    );
    let criterion = |d: &[f64]| TrialOutcome {
        criterion: d
            .iter()
            .enumerate()
            .map(|(i, x)| (i + 1) as f64 * x * x)
            .sum(),
        compensators: vec![],
        ok: true,
    };
    let full = sensitivity(
        &spec,
        &FnToleranceEvaluator::new(criterion),
        &Monitor::default(),
    )
    .unwrap();
    assert!(!full.cancelled);
    assert_eq!(full.operands.len(), n);
    assert_eq!(full.evaluations, 2 * n + 1);
    // Cancel while the second batch (points 32..64, with operand 20's
    // limits) is evaluated: the evaluator stops, the first batch (the
    // nominal and operands 0..15, the minus limit of 15) is kept.
    let ctx = JobContext::new();
    let token = ctx.cancel.clone();
    let ev = FnToleranceEvaluator::new(move |d| {
        if d[20] != 0.0 {
            token.cancel();
        }
        criterion(d)
    });
    let r = sensitivity(&spec, &ev, &Monitor::new(ctx)).unwrap();
    assert!(r.cancelled);
    assert_eq!(r.evaluations, 32);
    assert_eq!(r.operands.len(), 15);
    assert_eq!(r.ranking.len(), 15);
    assert_eq!(r.nominal, full.nominal);
    for (a, b) in r.operands.iter().zip(&full.operands) {
        assert_eq!(a, b);
    }
    // Ranked worst first among the completed operands.
    assert_eq!(r.ranking[0], 14);
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(
        sensitivity(
            &spec,
            &FnToleranceEvaluator::new(criterion),
            &Monitor::new(ctx)
        )
        .is_err()
    );
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

#[test]
fn reflected_irregularity_has_known_pv_and_rms() {
    use mklight_core::param::IrregularityMode;
    use mklight_core::surface_math::evaluate;
    let mut system = sys();
    system.surfaces[1].shape = SurfaceShape::Plane;
    system.surfaces[1].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 5.0 };
    let wavelength_mm = system.wavelengths.primary_um() * 1e-3;
    for (mode, rms) in [
        (IrregularityMode::Astigmatism, 1.0 / (2.0 * 6f64.sqrt())),
        (IrregularityMode::Spherical, 2.0 / (3.0 * 5f64.sqrt())),
    ] {
        let path = ParamPath::Irregularity {
            surface: SurfaceId(10),
            mode,
        };
        let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(path, 1.0)]);
        let perturbed = spec.apply(&system, &[1.0]).unwrap();
        let opd =
            |x, y| 2.0 * evaluate(&perturbed.surfaces[1].shape, x, y).unwrap().sag / wavelength_mm;
        let pv = match mode {
            IrregularityMode::Astigmatism => opd(5.0, 0.0) - opd(0.0, 5.0),
            IrregularityMode::Spherical => opd(0.0, 0.0) - opd(5.0 / 2f64.sqrt(), 0.0),
        };
        assert!((pv - 1.0).abs() < 1e-14);
        // Independent area quadrature, uniform in radius squared and azimuth.
        let mut sum = 0.0;
        let mut square = 0.0;
        for k in 0..1000 {
            let radius = 5.0 * ((k as f64 + 0.5) / 1000.0).sqrt();
            for j in 0..32 {
                let a = std::f64::consts::TAU * j as f64 / 32.0;
                let w = opd(radius * a.cos(), radius * a.sin());
                sum += w;
                square += w * w;
            }
        }
        let measured = (square / 32000.0 - (sum / 32000.0).powi(2)).sqrt();
        assert!(
            (measured - rms).abs() < 2e-6,
            "{mode:?}: {measured} vs {rms}"
        );
    }
}

#[test]
fn returning_surface_and_rigid_group_frames() {
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::param::PlacementComponent;
    use mklight_core::{ResolvedSystem, Snapshot, Vec3};
    use std::sync::Arc;
    let nominal = mklight_core::samples::double_gauss();
    let resolve = |s: OpticalSystem| {
        ResolvedSystem::new(
            &Snapshot::new(s).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
        )
        .unwrap()
    };
    let baseline = resolve(nominal.clone());
    for path in [
        ParamPath::DecenterX {
            surface: SurfaceId(3),
        },
        ParamPath::TiltY {
            surface: SurfaceId(3),
        },
    ] {
        let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(path, 0.2)]);
        for d in [0.0, 0.2] {
            let moved = resolve(spec.apply(&nominal, &[d]).unwrap());
            for i in 4..nominal.surfaces.len() {
                assert_eq!(moved.surfaces[i].frame, baseline.surfaces[i].frame);
            }
        }
    }
    let decenter = ParamPath::ElementPlacement {
        surface: SurfaceId(3),
        last: SurfaceId(5),
        pivot: None,
        component: PlacementComponent::DecenterX,
    };
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(decenter, 0.2)]);
    for d in [0.0, 0.2] {
        let moved = resolve(spec.apply(&nominal, &[d]).unwrap());
        for i in 1..nominal.surfaces.len() {
            let shift = if (3..=5).contains(&i) {
                Vec3::new(d, 0.0, 0.0)
            } else {
                Vec3::ZERO
            };
            assert!(
                (moved.surfaces[i].frame.origin - baseline.surfaces[i].frame.origin - shift).norm()
                    < 1e-14
            );
            assert_eq!(moved.surfaces[i].frame.rot, baseline.surfaces[i].frame.rot);
        }
    }
    // Rear-vertex pivot: first vertex follows x'=sin(theta)*(z-z_pivot).
    let tilt = ParamPath::ElementPlacement {
        surface: SurfaceId(3),
        last: SurfaceId(5),
        pivot: Some(SurfaceId(5)),
        component: PlacementComponent::TiltY,
    };
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(tilt, 2.0)]);
    let moved = resolve(spec.apply(&nominal, &[2.0]).unwrap());
    for i in 3..=5 {
        let z = baseline.surfaces[i].frame.origin.z - baseline.surfaces[5].frame.origin.z;
        assert!((moved.surfaces[i].frame.origin.x - z * 2f64.to_radians().sin()).abs() < 1e-14);
    }
    for i in 6..nominal.surfaces.len() {
        assert_eq!(moved.surfaces[i].frame, baseline.surfaces[i].frame);
    }
    // One rigid motion per first surface: a second pivot is rejected up front.
    let pivot = |pivot| ParamPath::ElementPlacement {
        surface: SurfaceId(3),
        last: SurfaceId(5),
        pivot,
        component: PlacementComponent::TiltX,
    };
    let conflicting = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(pivot(None), 1.0),
        ToleranceOperand::symmetric(pivot(Some(SurfaceId(4))), 1.0),
    ]);
    assert!(conflicting.validate().is_err());
}

#[test]
fn requested_percentiles_records_and_case_reconstruction() {
    let nominal = sys();
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(t(10), 0.1)]);
    let reg = OperandRegistry::builtin();
    let criterion = track_spec();
    let evaluator = OpticalToleranceEvaluator {
        nominal: &nominal,
        spec: &spec,
        criterion: &criterion,
        evaluator: &ModelEvaluator,
        registry: &reg,
        lsq: LsqSettings::default(),
    };
    let settings = MonteCarloSettings {
        trials: 9,
        record_trials: 3,
        percentiles: vec![0.0, 50.0, 97.7, 100.0],
        ..Default::default()
    };
    let r = monte_carlo(&spec, &evaluator, &settings, &Monitor::default()).unwrap();
    assert!(r.nominal.criterion < 1e-20);
    assert_eq!(r.percentiles[0].1, r.min);
    assert_eq!(r.percentiles[3].1, r.max);
    let mut sorted = r.criteria.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(r.percentiles[1].1, sorted[4]);
    assert!((r.percentiles[2].1 - (sorted[7] + 0.816 * (sorted[8] - sorted[7]))).abs() < 1e-15);
    assert_eq!(r.records.len(), 3);
    for record in &r.records {
        let system = spec
            .case_system(
                &nominal,
                &record.perturbations,
                &record.outcome.compensators,
            )
            .unwrap();
        let track: f64 = system.surfaces[1..system.surfaces.len() - 1]
            .iter()
            .map(|sf| sf.thickness)
            .sum();
        assert!(((track - 100.0).powi(2) - record.outcome.criterion).abs() < 1e-16);
    }
    let r = sensitivity(&spec, &evaluator, &Monitor::default()).unwrap();
    assert_eq!(r.operands[0].label, "Surface 10: Thickness");
    assert!(!r.operands[0].label.contains('{'));
}

#[test]
fn double_gauss_mtf_compensation_stays_on_main_peak() {
    use mklight_optim::merit::operand::{MtfMethod, MtfOrientation};
    use mklight_optim::merit::{BuiltinEvaluator, FieldSelector, OperandEvaluator};
    use mklight_optim::tolerance::ToleranceEvaluator;
    let mut nominal = mklight_core::samples::double_gauss();
    // A saved focus offset puts nominal-start local refinement near a side
    // lobe; the allowed range still contains the principal peak.
    nominal.surfaces[11].thickness += 0.3;
    let mut op = Operand::new(OperandKind::Mtf {
        frequency: 30.0,
        orientation: MtfOrientation::Average,
        method: MtfMethod::Diffraction,
        defocus_mm: 0.0,
        detector: None,
        wavelengths: vec![],
    })
    .target(1.0);
    op.field = FieldSelector::Index { index: 0 };
    let criterion = MeritFunctionSpec::new(vec![op.clone()]);
    let mut spec = ToleranceSpec::new(vec![
        ToleranceOperand::symmetric(t(1), 0.02),
        ToleranceOperand::symmetric(t(3), 0.02),
        ToleranceOperand::symmetric(
            ParamPath::Curvature {
                surface: SurfaceId(11),
            },
            1e-5,
        ),
    ]);
    spec.compensators.push(Compensator {
        param: t(11),
        range: Some(0.8),
        scale: Some(0.1),
    });
    let mut evaluator = BuiltinEvaluator::builtin();
    evaluator.mtf.pupil.grid = 32;
    evaluator.mtf.pupil.rays = 17;
    evaluator.mtf.pupil.supersample = 2;
    let registry = OperandRegistry::builtin();
    let optical = OpticalToleranceEvaluator {
        nominal: &nominal,
        spec: &spec,
        criterion: &criterion,
        evaluator: &evaluator,
        registry: &registry,
        lsq: LsqSettings {
            max_iterations: 16,
            ..Default::default()
        },
    };
    let settings = MonteCarloSettings {
        trials: 18,
        seed: 71,
        batch: 6,
        record_trials: 1,
        ..Default::default()
    };
    let report = monte_carlo(&spec, &optical, &settings, &Monitor::default()).unwrap();
    assert_eq!(report.failures, 0);
    // The secondary peak (MTF about .12) gives a much larger squared
    // residual than the main peak. Every small-error trial stays near nominal.
    for value in &report.criteria {
        assert!(
            *value < 1.10 * report.nominal.criterion,
            "{} vs nominal {}",
            value,
            report.nominal.criterion
        );
    }
    // Independent dense scan of the first perturbed prescription.
    let delta = spec.sample_trial(settings.seed, 0);
    let mut system = spec.apply(&nominal, &delta).unwrap();
    let focus = t(11).get(&nominal).unwrap();
    let mut systems = Vec::new();
    for k in 0..=128 {
        t(11)
            .set(&mut system, focus - 0.8 + 1.6 * k as f64 / 128.0)
            .unwrap();
        systems.push(system.clone());
    }
    let scan = evaluator
        .evaluate_batch(&systems, &[&op], &JobContext::new())
        .unwrap();
    let best = scan
        .iter()
        .map(|row| (row[0].as_ref().unwrap() - 1.0).powi(2))
        .fold(f64::INFINITY, f64::min);
    assert!(
        report.criteria[0] <= best + 1e-6,
        "{} vs scan {best}",
        report.criteria[0]
    );
    let repeated = optical
        .evaluate_batch(&[delta], &JobContext::new())
        .unwrap();
    assert_eq!(repeated[0].criterion, report.criteria[0]);
    assert_eq!(
        repeated[0].compensators,
        report.records[0].outcome.compensators
    );
}

#[test]
fn rigid_decentre_ray_matches_translated_lens_analytic_solution() {
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::param::PlacementComponent;
    use mklight_core::{ResolvedSystem, Snapshot, Vec3};
    use mklight_trace::trace::{Ray, TraceOptions, trace_ray};
    use std::sync::Arc;
    let system = mklight_core::samples::singlet();
    let path = ParamPath::ElementPlacement {
        surface: SurfaceId(1),
        last: SurfaceId(2),
        pivot: None,
        component: PlacementComponent::DecenterX,
    };
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(path, 0.2)]);
    let moved = spec.apply(&system, &[0.2]).unwrap();
    let resolve = |s| {
        ResolvedSystem::new(
            &Snapshot::new(s).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
        )
        .unwrap()
    };
    let a = resolve(system);
    let b = resolve(moved);
    let trace = |s, x| {
        trace_ray(
            s,
            &Ray::new(Vec3::new(x, 0.0, -1.0), Vec3::Z),
            1,
            &TraceOptions::unclipped(),
            None,
        )
    };
    let nominal_ray = trace(&a, 0.7);
    let displaced_ray = trace(&b, 0.9);
    assert!(nominal_ray.status.is_ok() && displaced_ray.status.is_ok());
    assert!(
        (displaced_ray.ray.pos - nominal_ray.ray.pos - Vec3::new(0.2, 0.0, 0.0)).norm() < 1e-13
    );
    assert!((displaced_ray.ray.dir - nominal_ray.ray.dir).norm() < 1e-15);
}

#[test]
fn zero_irregularity_traces_exactly_as_its_base() {
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::param::IrregularityMode;
    use mklight_core::{ResolvedSystem, Snapshot, Vec3};
    use mklight_trace::trace::{Ray, TraceOptions, trace_ray};
    use std::sync::Arc;
    let mut system = mklight_core::samples::singlet();
    system.surfaces[2].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 5.0 };
    let path = ParamPath::Irregularity {
        surface: system.surfaces[2].id,
        mode: IrregularityMode::Spherical,
    };
    let spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(path, 1.0)]);
    let trace = |s: OpticalSystem| {
        let r = ResolvedSystem::new(
            &Snapshot::new(s).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
        )
        .unwrap();
        let ray = Ray::new(Vec3::new(3.0, 1.0, -1.0), Vec3::Z);
        trace_ray(&r, &ray, 1, &TraceOptions::unclipped(), None)
    };
    let zero = spec.apply(&system, &[0.0]).unwrap();
    assert!(matches!(
        zero.surfaces[2].shape,
        SurfaceShape::SagOverlay { .. }
    ));
    let (a, b) = (trace(system.clone()), trace(zero));
    assert_eq!((a.ray.pos, a.ray.dir, a.opl), (b.ray.pos, b.ray.dir, b.opl));
    let c = trace(spec.apply(&system, &[1.0]).unwrap());
    assert_ne!(a.ray.dir, c.ray.dir);
}

#[test]
fn rotated_irregularity_has_known_sag_gradient_and_hessian() {
    use mklight_core::param::{IrregularityMode, ShapeParameter};
    use mklight_core::surface_math::{Symmetry, evaluate, shape_symmetry};
    let mut s = sys();
    s.surfaces[1].shape = SurfaceShape::Plane;
    s.surfaces[1].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 5.0 };
    ParamPath::Irregularity {
        surface: SurfaceId(10),
        mode: IrregularityMode::Astigmatism,
    }
    .set(&mut s, 1.0)
    .unwrap();
    let angle = ParamPath::ShapeParam {
        surface: SurfaceId(10),
        parameter: ShapeParameter::AstigmatismAngleDeg,
    };
    angle.set(&mut s, 45.0).unwrap();
    assert_eq!(angle.get(&s).unwrap(), 45.0);
    let shape = &s.surfaces[1].shape;
    let a = s.wavelengths.primary_um() * 1e-3 / (4.0 * 25.0);
    // At 45 degrees the departure is 2*a*x*y, with pure mixed power.
    let v = evaluate(shape, 2.0, 3.0).unwrap();
    for (got, want) in [
        (v.sag, 12.0 * a),
        (v.gradient[0], 6.0 * a),
        (v.gradient[1], 4.0 * a),
        (v.hessian[0], 0.0),
        (v.hessian[1], 2.0 * a),
        (v.hessian[2], 0.0),
    ] {
        assert!((got - want).abs() < 1e-18, "{got} vs {want}");
    }
    assert_eq!(shape_symmetry(shape), Symmetry::None);
    let before = v;
    angle.set(&mut s, 225.0).unwrap();
    assert!((evaluate(&s.surfaces[1].shape, 2.0, 3.0).unwrap().sag - before.sag).abs() < 1e-18);
    angle.set(&mut s, 90.0).unwrap();
    assert_eq!(shape_symmetry(&s.surfaces[1].shape), Symmetry::Meridional);
    // Older JSON has a zero-degree axis; non-finite angles fail at the model boundary.
    let mut json = serde_json::to_value(&s).unwrap();
    json["surfaces"][1]["shape"]
        .as_object_mut()
        .unwrap()
        .remove("astigmatism_angle_deg");
    let back: OpticalSystem = serde_json::from_value(json).unwrap();
    assert_eq!(angle.get(&back).unwrap(), 0.0);
    if let SurfaceShape::SagOverlay {
        astigmatism_angle_deg,
        ..
    } = &mut s.surfaces[1].shape
    {
        *astigmatism_angle_deg = f64::NAN;
    }
    assert!(
        s.validation_issues()
            .iter()
            .any(|i| i.path.ends_with("astigmatism_angle_deg"))
    );
}

#[test]
fn irregularity_axes_are_uniform_independent_and_replayable() {
    use mklight_core::param::IrregularityMode;
    let mut nominal = sys();
    for s in &mut nominal.surfaces[1..=2] {
        s.semi_diameter = mklight_core::SemiDiameter::Fixed { value: 5.0 };
    }
    let spec = ToleranceSpec::new(
        [10, 11]
            .into_iter()
            .map(|id| {
                ToleranceOperand::symmetric(
                    ParamPath::Irregularity {
                        surface: SurfaceId(id),
                        mode: IrregularityMode::Astigmatism,
                    },
                    0.5,
                )
            })
            .collect(),
    );
    let settings = MonteCarloSettings {
        seed: 42,
        ..Default::default()
    };
    let axis = |s: &OpticalSystem, j: usize| match s.surfaces[j].shape {
        SurfaceShape::SagOverlay {
            astigmatism_angle_deg,
            ..
        } => astigmatism_angle_deg,
        _ => panic!(),
    };
    let mut bins = [[0usize; 12]; 2];
    for trial in 0..2400 {
        let deltas = spec.sample_trial(settings.seed, trial);
        let s = spec
            .trial_case_system(&nominal, &deltas, &[], &settings, trial)
            .unwrap();
        assert_eq!(
            s,
            spec.trial_case_system(&nominal, &deltas, &[], &settings, trial)
                .unwrap()
        );
        for (j, bins) in bins.iter_mut().enumerate() {
            let a = axis(&s, j + 1);
            assert!((0.0..180.0).contains(&a));
            bins[(a / 15.0) as usize] += 1;
        }
        assert_ne!(axis(&s, 1), axis(&s, 2));
    }
    assert!(
        bins.iter().flatten().all(|n| (140..260).contains(n)),
        "{bins:?}"
    );
    let d = spec.sample_trial(42, 3);
    let a = spec
        .trial_case_system(&nominal, &d, &[], &settings, 3)
        .unwrap();
    let b = spec
        .trial_case_system(
            &nominal,
            &d,
            &[],
            &MonteCarloSettings {
                seed: 43,
                ..settings.clone()
            },
            3,
        )
        .unwrap();
    assert_ne!(axis(&a, 1), axis(&b, 1));
    let fixed = spec
        .trial_case_system(
            &nominal,
            &d,
            &[],
            &MonteCarloSettings {
                random_irregularity_orientation: false,
                ..settings
            },
            3,
        )
        .unwrap();
    assert_eq!(axis(&fixed, 1), 0.0);
}

#[test]
fn element_tilt_z_rotates_freeform_axes_and_returns_after_the_range() {
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::param::PlacementComponent;
    use mklight_core::{ResolvedSystem, Snapshot, Vec3, samples};
    use std::sync::Arc;
    let nominal = samples::biconic_element();
    let mut moved = nominal.clone();
    let path = ParamPath::ElementPlacement {
        surface: moved.surfaces[1].id,
        last: moved.surfaces[2].id,
        pivot: None,
        component: PlacementComponent::TiltZ,
    };
    path.set(&mut moved, 90.0).unwrap();
    assert_eq!(path.get(&moved).unwrap(), 90.0);
    let resolve = |s| {
        ResolvedSystem::new(
            &Snapshot::new(s).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
        )
        .unwrap()
    };
    let (a, b) = (resolve(nominal), resolve(moved));
    for j in 1..=2 {
        let f = b.surfaces[j].frame;
        assert!((f.to_global_dir(Vec3::X) - Vec3::Y).norm() < 1e-15);
        assert!((f.to_global_dir(Vec3::Y) + Vec3::X).norm() < 1e-15);
        assert_eq!(f.origin, a.surfaces[j].frame.origin);
    }
    assert_eq!(a.surfaces[3].frame, b.surfaces[3].frame);
}
