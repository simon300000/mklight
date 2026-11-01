#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::ComputeBackend;
use mklight_optim::{Monitor, deriv::*, gpu::*, lsq::*, merit::operand::*, merit::*, problem::*};
use std::sync::Arc;
fn backend() -> Arc<dyn ComputeBackend> {
    Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    )
}
fn context(b: Arc<dyn ComputeBackend>) -> GpuContext {
    GpuContext::new(b, Snapshot::new(samples::singlet()).unwrap())
}
#[test]
fn host_derivatives_and_tall_qr() {
    let b = backend();
    let ctx = JobContext::new();
    for n in [8, 32, 100] {
        let p = FnProblem::new(
            (0..n)
                .map(|i| Variable::bounded(format!("x{i}"), -2., 2.))
                .collect(),
            4 * n,
            move |x, r, _| {
                for (i, r) in r.iter_mut().enumerate() {
                    let j = i % n;
                    *r = (1. + i as f64 / (4 * n) as f64)
                        * (x[j].exp() - (0.1 * (j % 3) as f64).exp());
                }
                EvalStatus::Ok
            },
        );
        let p = ResidentProblem {
            problem: p,
            context: context(b.clone()),
        };
        let x = vec![0.3; n];
        let f = evaluate_one(&p, &x, &ctx).unwrap();
        let settings = JacobianSettings {
            scheme: FdScheme::Central,
            relative_step: Some(1e-4),
            use_analytic: false,
        };
        let gpu = fd_jacobian(&p, &x, &f, &settings, &ctx).unwrap();
        let cpu = fd_jacobian(&p.problem, &x, &f, &settings, &ctx).unwrap();
        let max = gpu
            .residuals
            .data
            .iter()
            .zip(&cpu.residuals.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(max < 1e-8, "V={n}: {max}");
        let result = least_squares(
            &p,
            &x,
            &LsqSettings {
                max_iterations: 15,
                jacobian: settings,
                ..Default::default()
            },
            &Monitor::new(ctx.clone()),
        )
        .unwrap();
        assert!(result.merit < 1e-12, "V={n}: {result:?}");
        assert!(p.context.stats().command_buffers > 0);
        eprintln!(
            "P8 V={n} derivative_max={max:e} merit={:e} stats={}",
            result.merit,
            serde_json::to_string(&p.context.stats()).unwrap()
        );
    }
}
#[test]
fn host_wave_and_solver_operands() {
    let system = samples::singlet();
    let b = backend();
    let ctx = JobContext::new();
    let mut reference = BuiltinEvaluator::builtin();
    reference.solvers.ghost.pupil_samples = 8;
    let ops = vec![
        Operand::new(OperandKind::RelativeIllumination).at_field(1),
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
        Operand::new(OperandKind::StrayLightPower {
            class: StrayPathClass::AllNonPrimary,
            rays: 100,
            seed: 17,
        }),
    ];
    for op in &ops {
        let evaluator = ResidentEvaluator::new(context(b.clone()), reference.clone());
        let values = evaluator
            .evaluate_batch(std::slice::from_ref(&system), &[op], &ctx)
            .unwrap();
        let stats = evaluator.stats();
        eprintln!("P8 {} value={values:?} stats={stats:?}", op.kind.name());
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
    let b = backend();
    let ctx = JobContext::new();
    let failed = FnProblem::new(vec![Variable::bounded("x", -1., 1.)], 1, |x, r, _| {
        r[0] = x[0] * x[0] + 2. * x[0];
        if x[0] > 0. {
            EvalStatus::Failed
        } else {
            EvalStatus::Ok
        }
    });
    let failed = ResidentProblem {
        problem: failed,
        context: context(b.clone()),
    };
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
    assert!(gpu.stats().device_operands > 0);
}
