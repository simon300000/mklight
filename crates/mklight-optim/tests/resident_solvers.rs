#![cfg(target_os = "macos")]
//! Solves and solver operands on the host's Metal device: least squares on
//! both sides of the tall-QR crossover, MTF, ghost and stray-light operands
//! through the resident evaluator, finite differences of a first-order
//! operand, and relative illumination with device visibility.

mod common;

use common::OnDevice;
use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::ComputeBackend;
use mklight_optim::{Monitor, deriv::*, gpu::*, lsq::*, merit::operand::*, merit::*, problem::*};
use std::sync::Arc;
fn context(b: Arc<dyn ComputeBackend>) -> GpuContext {
    GpuContext::new(b, Snapshot::new(samples::singlet()).unwrap())
}
/// LM solves keep host QR below the measured crossover and use the resident
/// tall QR (with retained Q projections) above it; both reach the optimum.
#[test]
fn tall_qr_crossover_and_resident_lm() {
    let Some(b) = common::metal() else {
        return;
    };
    let ctx = JobContext::new();
    for (n, m) in [(8, 32), (100, 400), (64, 4096)] {
        let p = FnProblem::new(
            (0..n)
                .map(|i| Variable::bounded(format!("x{i}"), -2., 2.))
                .collect(),
            m,
            move |x, r, _| {
                for (i, r) in r.iter_mut().enumerate() {
                    let j = i % n;
                    *r = (1. + i as f64 / m as f64) * (x[j].exp() - (0.1 * (j % 3) as f64).exp());
                }
                EvalStatus::Ok
            },
        );
        let p = OnDevice {
            problem: p,
            context: context(b.clone()),
        };
        let settings = LsqSettings {
            max_iterations: 15,
            jacobian: JacobianSettings {
                scheme: FdScheme::Central,
                relative_step: Some(1e-4),
                use_analytic: false,
            },
            ..Default::default()
        };
        let x = vec![0.3; n];
        let gpu = least_squares(&p, &x, &settings, &Monitor::new(ctx.clone())).unwrap();
        let cpu = least_squares(&p.problem, &x, &settings, &Monitor::new(ctx.clone())).unwrap();
        assert!(gpu.merit < 1e-12 && cpu.merit < 1e-12, "V={n}: {gpu:?}");
        assert_eq!(gpu.x.len(), cpu.x.len());
        let dx = gpu
            .x
            .iter()
            .zip(&cpu.x)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(dx < 1e-6, "V={n}: {dx}");
        let stats = p.context.stats();
        assert_eq!(stats.command_buffers > 0, m * n >= 1 << 16, "V={n} M={m}");
        assert_eq!(stats.fallback_count, 0);
        eprintln!(
            "V={n} M={m} merit={:e} dx={dx:e} stats={}",
            gpu.merit,
            serde_json::to_string(&stats).unwrap()
        );
    }
}
#[test]
fn host_wave_and_solver_operands() {
    let system = samples::singlet();
    let Some(b) = common::metal() else {
        return;
    };
    let ctx = JobContext::new();
    let mut reference = BuiltinEvaluator::builtin();
    reference.solvers.ghost.pupil_samples = 8;
    let ops = vec![
        Operand::new(OperandKind::WavefrontRms {
            reference: WavefrontReference::Centroid,
            sampling: Default::default(),
        }),
        Operand::new(OperandKind::Mtf {
            frequency: 10.,
            orientation: MtfOrientation::Average,
            method: MtfMethod::Diffraction,
            defocus_mm: 0.,
            detector: None,
            wavelengths: vec![],
        }),
        Operand::new(OperandKind::GhostFocus {
            pair: None,
            path: None,
            worst: 1,
            metric: GhostFocusMetric::Distance,
            max_reflections: 2,
        }),
        Operand::new(OperandKind::GhostIrradiance {
            pair: None,
            path: None,
            metric: GhostMetric::Power,
            orders: vec![],
            max_reflections: 2,
        }),
        // Enough sources for the stray solver to give the device its share
        // (a fifth, from 32768 device sources); a smaller run is the CPU
        // reference and counts as completed in f64.
        Operand::new(OperandKind::StrayLightPower {
            class: StrayPathClass::AllNonPrimary,
            rays: 5 * 32768,
            seed: 17,
        }),
    ];
    for op in &ops {
        let evaluator = ResidentEvaluator::new(context(b.clone()), reference.clone());
        let values = evaluator
            .evaluate_batch(std::slice::from_ref(&system), &[op], &ctx)
            .unwrap();
        let stats = evaluator.stats();
        eprintln!("{} value={values:?} stats={stats:?}", op.kind.name());
        assert!(values[0][0].as_ref().is_ok_and(|v| v.is_finite()));
        assert!(
            stats.device_operands > 0,
            "{}: fallback-only comparison: {stats:?}",
            op.kind.name()
        );
    }
}

#[test]
fn one_sided_retry_and_thick_lens_derivative() {
    use mklight_core::param::ParamPath;
    let Some(b) = common::metal() else {
        return;
    };
    let ctx = JobContext::new();
    let failed = FnProblem::new(vec![Variable::bounded("x", -1., 1.)], 1, |x, r, _| {
        r[0] = x[0] * x[0] + 2. * x[0];
        if x[0] > 0. {
            EvalStatus::Failed
        } else {
            EvalStatus::Ok
        }
    });
    let f = evaluate_one(&failed, &[0.], &ctx).unwrap();
    let j = fd_jacobian(
        &failed,
        &[0.],
        &f,
        &JacobianSettings {
            scheme: FdScheme::Central,
            relative_step: Some(1e-4),
            use_analytic: false,
        },
        &ctx,
    )
    .unwrap();
    assert!(j.failed_columns.is_empty());
    assert_eq!(j.evaluations, 4);
    assert!((j.residuals[(0, 0)] - 2.).abs() < 1e-10);
    let system = samples::singlet();
    let cpu = BuiltinEvaluator::builtin();
    let tr = cpu.trace.tracer(&system).unwrap();
    let w = tr.system().primary_wavelength;
    let n = tr.system().n_before(2, w);
    let c1 = system.surfaces[1].shape.curvature();
    let c2 = system.surfaces[2].shape.curvature();
    let t = system.surfaces[1].thickness;
    let power = (n - 1.) * (c1 - c2) + (n - 1.).powi(2) * t * c1 * c2 / n;
    let analytic = -(n - 1. + (n - 1.).powi(2) * t * c2 / n) / (power * power);
    let gpu = ResidentEvaluator::new(context(b), cpu);
    let p = OpticalProblem::new(
        system.clone(),
        vec![VariableSpec::bounded(
            ParamPath::Curvature {
                surface: system.surfaces[1].id,
            },
            c1 * 0.5,
            c1,
        )],
        MeritFunctionSpec::new(vec![Operand::new(OperandKind::Efl)]),
        gpu.clone(),
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let f = evaluate_one(&p, &[c1], &ctx).unwrap();
    for step in [1e-2, 1e-3, 1e-4, 1e-5] {
        let d = fd_jacobian(
            &p,
            &[c1],
            &f,
            &JacobianSettings {
                scheme: FdScheme::Central,
                relative_step: Some(step),
                use_analytic: false,
            },
            &ctx,
        )
        .unwrap()
        .residuals[(0, 0)];
        assert!(
            (d - analytic).abs() / analytic.abs() < 1e-3,
            "step {step}: {d} vs {analytic}"
        );
    }
    // First-order operands are host stages of the ray pass, not fallbacks.
    let stats = gpu.stats();
    assert!(
        stats.host_operands > 0 && stats.fallback_operands == 0,
        "{stats:?}"
    );
}

#[test]
fn host_ri_preserves_adaptive_etendue_model() {
    let Some(b) = common::metal() else {
        return;
    };
    let mut clipped = samples::cooke_triplet();
    clipped.surfaces[1].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 4.5 };
    for (name, system) in [
        ("singlet", samples::singlet()),
        ("cooke", samples::cooke_triplet()),
        ("clipped", clipped),
        ("relay", samples::finite_relay()),
    ] {
        let cpu = BuiltinEvaluator::builtin();
        let gpu = ResidentEvaluator::new(
            GpuContext::new(b.clone(), Snapshot::new(system.clone()).unwrap()),
            cpu.clone(),
        );
        let ops: Vec<_> = (0..system.fields.fields.len())
            .map(|f| Operand::new(OperandKind::RelativeIllumination).at_field(f))
            .collect();
        let refs: Vec<_> = ops.iter().collect();
        let ctx = JobContext::new();
        let expected = cpu
            .evaluate_batch(std::slice::from_ref(&system), &refs, &ctx)
            .unwrap();
        let got = gpu
            .evaluate_batch(std::slice::from_ref(&system), &refs, &ctx)
            .unwrap();
        let mut error = 0_f64;
        assert_eq!(got[0].len(), expected[0].len());
        for (a, b) in got[0].iter().zip(&expected[0]) {
            error = error.max((a.as_ref().unwrap() - b.as_ref().unwrap()).abs());
        }
        assert!(error < 1e-3 / 3.);
        assert!(gpu.stats().device_operands > 0, "{name}: {:?}", gpu.stats());
        assert_eq!(gpu.stats().fallback_operands, 0);
        let info = gpu
            .run(&system, "ri", &serde_json::json!({}), None, || Ok(()))
            .unwrap()
            .info;
        let visibility = &info.settings["optim_native_stages"]["ri_visibility"];
        assert!(visibility["lookups"].as_u64().unwrap() > 100);
        if name == "clipped" {
            assert!(visibility["blocked"].as_u64().unwrap() > 0);
        }
        eprintln!("RI visibility sample={name} counters={visibility}");
        let stats = gpu.context.stats();
        assert!(stats.fallback_count > 0);
        assert!(
            stats
                .stages
                .iter()
                .any(|s| s.name == "RI adaptive reference completion")
        );
        eprintln!(
            "RI sample={name} error={error:e} values={got:?} stats={:?}",
            gpu.stats()
        );
    }
}
