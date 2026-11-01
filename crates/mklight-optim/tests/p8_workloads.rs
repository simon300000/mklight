#![cfg(target_os = "macos")]
//! Complete-workload timings include every candidate's qualification cost.
use mklight_core::{JobContext, MaterialRef, Snapshot, param::ParamPath, samples};
use mklight_gpu::ComputeBackend;
use mklight_optim::{
    Monitor, deriv::*, gpu::*, lsq::*, merit::default::DefaultMeritSpec, merit::*, problem::*,
    tolerance::*,
};
use std::{sync::Arc, time::Instant};
fn backend() -> Arc<dyn ComputeBackend> {
    Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    )
}
fn phase(i: usize) -> &'static str {
    if i == 0 {
        "cold"
    } else if i == 1 {
        "warmup"
    } else {
        "measured"
    }
}
#[test]
#[ignore = "provisional complete synthetic LM solve timings around the tall-QR crossover"]
fn jacobians_and_solves() {
    let b = backend();
    let ctx = JobContext::new();
    for (n, m) in [(8, 32), (32, 128), (100, 400), (32, 4000), (100, 4000)] {
        let problem = FnProblem::residuals(
            (0..n)
                .map(|i| Variable::bounded(format!("x{i}"), -2., 2.))
                .collect(),
            m,
            move |x, r| {
                for (i, v) in r.iter_mut().enumerate() {
                    *v = (1. + i as f64 / m as f64)
                        * (x[i % n].exp() - (0.1 * ((i % n) % 3) as f64).exp());
                }
            },
        );
        let p = ResidentProblem {
            problem,
            context: GpuContext::new(b.clone(), Snapshot::new(samples::singlet()).unwrap()),
        };
        let x = vec![0.3; n];
        let settings = LsqSettings {
            max_iterations: 15,
            jacobian: JacobianSettings {
                scheme: FdScheme::Central,
                relative_step: Some(1e-4),
                use_analytic: false,
            },
            ..Default::default()
        };
        for run in 0..12 {
            let t = Instant::now();
            let cpu = least_squares(&p.problem, &x, &settings, &Monitor::new(ctx.clone())).unwrap();
            let cpu_s = t.elapsed().as_secs_f64();
            let before = p.context.stats();
            let t = Instant::now();
            let gpu = least_squares(&p, &x, &settings, &Monitor::new(ctx.clone())).unwrap();
            let gpu_s = t.elapsed().as_secs_f64();
            assert!(gpu.merit < 1e-12 && cpu.merit < 1e-12);
            let after = p.context.stats();
            assert_eq!(
                after.command_buffers > before.command_buffers,
                m * n * n >= 1 << 24
            );
            eprintln!(
                "P8_WORKLOAD {}",
                serde_json::json!({"case":"synthetic_jacobian_and_solve","variables":n,"residuals":m,"run":run,"phase":phase(run),"cpu_seconds":cpu_s,"resident_seconds":gpu_s,"cpu_merit":cpu.merit,"resident_merit":gpu.merit,"commands":after.command_buffers-before.command_buffers,"device_seconds":after.device_seconds-before.device_seconds,"copied_bytes":after.copied_bytes-before.copied_bytes,"peak_plan_bytes":after.peak_bytes,"fallback_count":after.fallback_count-before.fallback_count})
            );
        }
    }
}
#[test]
#[ignore = "provisional complete compensated/uncompensated trial and glass refinement timings"]
fn trials_and_glass() {
    let b = backend();
    let system = samples::cooke_triplet();
    let ctx = JobContext::new();
    let reference = BuiltinEvaluator::builtin();
    let registry = OperandRegistry::builtin();
    let criterion = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate(&system)
    .unwrap();
    let focus = ParamPath::Thickness {
        surface: system.surfaces[system.surfaces.len() - 2].id,
    };
    let lsq = LsqSettings {
        max_iterations: 8,
        ..Default::default()
    };
    for compensated in [false, true] {
        let mut spec = ToleranceSpec::new(vec![ToleranceOperand::symmetric(
            ParamPath::Curvature {
                surface: system.surfaces[1].id,
            },
            1e-5,
        )]);
        if compensated {
            spec.compensators.push(Compensator {
                param: focus.clone(),
                range: Some(1.),
                scale: Some(0.1),
            });
        }
        let settings = MonteCarloSettings {
            trials: 32,
            batch: 16,
            seed: 73,
            record_trials: 32,
            ..Default::default()
        };
        let gpu = ResidentEvaluator::new(
            GpuContext::new(b.clone(), Snapshot::new(system.clone()).unwrap()),
            reference.clone(),
        );
        let cpu = OpticalToleranceEvaluator {
            nominal: &system,
            spec: &spec,
            criterion: &criterion,
            evaluator: &reference,
            registry: &registry,
            lsq,
        };
        let new = OpticalToleranceEvaluator {
            nominal: &system,
            spec: &spec,
            criterion: &criterion,
            evaluator: &gpu,
            registry: &registry,
            lsq,
        };
        for run in 0..12 {
            let t = Instant::now();
            let expected = monte_carlo(&spec, &cpu, &settings, &Monitor::new(ctx.clone())).unwrap();
            let cpu_s = t.elapsed().as_secs_f64();
            let before = gpu.context.stats();
            let q0 = gpu.stats();
            let t = Instant::now();
            let got = monte_carlo(&spec, &new, &settings, &Monitor::new(ctx.clone())).unwrap();
            let gpu_s = t.elapsed().as_secs_f64();
            assert_eq!(expected.failures, got.failures);
            assert_eq!(expected.criteria.len(), got.criteria.len());
            for (a, b) in expected.criteria.iter().zip(&got.criteria) {
                assert!((a - b).abs() < 1e-8 + 0.01 * a.abs());
            }
            let after = gpu.context.stats();
            let q = gpu.stats();
            assert!(q.device_operands > q0.device_operands);
            eprintln!(
                "P8_WORKLOAD {}",
                serde_json::json!({"case":"cooke_tolerance","compensated":compensated,"trials":32,"batch":16,"seed":73,"run":run,"phase":phase(run),"cpu_seconds":cpu_s,"resident_seconds":gpu_s,"cpu_mean":expected.mean,"resident_mean":got.mean,"commands":after.command_buffers-before.command_buffers,"device_seconds":after.device_seconds-before.device_seconds,"copied_bytes":after.copied_bytes-before.copied_bytes,"peak_plan_bytes":after.peak_bytes,"fallback_operands":q.fallback_operands-q0.fallback_operands,"device_operands":q.device_operands-q0.device_operands,"reference_seconds":q.reference_seconds-q0.reference_seconds,"reference_checks":q.reference_checks-q0.reference_checks})
            );
        }
    }
    let systems: Vec<_> = ["N-SK16", "N-BAK1", "N-BK7"]
        .iter()
        .map(|name| {
            let mut s = system.clone();
            s.surfaces[1].material = MaterialRef::Catalog {
                catalog: None,
                name: (*name).into(),
                offset: Default::default(),
            };
            s
        })
        .collect();
    let nominal = focus.get(&system).unwrap();
    let vars = vec![VariableSpec::bounded(focus, nominal - 1., nominal + 1.)];
    let gpu = ResidentEvaluator::new(
        GpuContext::new(b, Snapshot::new(system.clone()).unwrap()),
        reference.clone(),
    );
    for run in 0..12 {
        let t = Instant::now();
        let cpu = scheduler::run(&reference, systems.len(), &ctx, |i, e| {
            let p = OpticalProblem::new(
                systems[i].clone(),
                vars.clone(),
                criterion.clone(),
                e,
                &registry,
            )?;
            least_squares(&p, &p.start_point()?, &lsq, &Monitor::new(ctx.clone()))
        })
        .unwrap();
        let cpu_s = t.elapsed().as_secs_f64();
        let before = gpu.context.stats();
        let q0 = gpu.stats();
        let t = Instant::now();
        let got = gpu
            .refine_candidates(&systems, &vars, &criterion, &lsq, &ctx)
            .unwrap();
        let gpu_s = t.elapsed().as_secs_f64();
        assert_eq!(got.len(), systems.len());
        for (i, r) in got.iter().enumerate() {
            let p = OpticalProblem::new(
                systems[i].clone(),
                vars.clone(),
                criterion.clone(),
                reference.clone(),
                &registry,
            )
            .unwrap();
            let e = evaluate_one(&p, &r.x, &ctx).unwrap();
            assert_eq!(r.merit, merit(p.weights(), &e.residuals, e.status));
            assert!((r.merit - cpu[i].merit).abs() < 1e-8 + 0.01 * cpu[i].merit.abs());
        }
        let after = gpu.context.stats();
        let q = gpu.stats();
        assert!(q.device_operands > q0.device_operands);
        eprintln!(
            "P8_WORKLOAD {}",
            serde_json::json!({"case":"cooke_catalogue_refinements","choices":["N-SK16","N-BAK1","N-BK7"],"variables":1,"run":run,"phase":phase(run),"cpu_seconds":cpu_s,"resident_seconds":gpu_s,"cpu_merits":cpu.iter().map(|r|r.merit).collect::<Vec<_>>(),"resident_merits":got.iter().map(|r|r.merit).collect::<Vec<_>>(),"commands":after.command_buffers-before.command_buffers,"device_seconds":after.device_seconds-before.device_seconds,"copied_bytes":after.copied_bytes-before.copied_bytes,"peak_plan_bytes":after.peak_bytes,"fallback_operands":q.fallback_operands-q0.fallback_operands,"device_operands":q.device_operands-q0.device_operands,"reference_seconds":q.reference_seconds-q0.reference_seconds,"reference_checks":q.reference_checks-q0.reference_checks})
        );
    }
}

#[test]
#[ignore = "provisional matched-method hybrid RI timings"]
fn ri_visibility_batches() {
    let b = backend();
    let system = samples::cooke_triplet();
    let ctx = JobContext::new();
    let reference = BuiltinEvaluator::builtin();
    let registry = OperandRegistry::builtin();
    let criterion = MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::RelativeIllumination).at_field(2),
    ]);
    let param = ParamPath::Thickness {
        surface: system.surfaces[system.surfaces.len() - 2].id,
    };
    let nominal = param.get(&system).unwrap();
    let vars = vec![VariableSpec::new(param)];
    for batch in [1, 16] {
        let evaluator = ResidentEvaluator::new(
            GpuContext::new(b.clone(), Snapshot::new(system.clone()).unwrap()),
            reference.clone(),
        );
        let cpu = OpticalProblem::new(
            system.clone(),
            vars.clone(),
            criterion.clone(),
            reference.clone(),
            &registry,
        )
        .unwrap();
        let new = OpticalProblem::new(
            system.clone(),
            vars.clone(),
            criterion.clone(),
            evaluator.clone(),
            &registry,
        )
        .unwrap();
        let points: Vec<_> = (0..batch).map(|i| [nominal + 0.001 * i as f64]).collect();
        let points = CandidateBatch::from_points(1, points.iter().map(|v| &v[..]));
        for run in 0..12 {
            let t = Instant::now();
            let expected = cpu.evaluate_batch(&points, &ctx).unwrap();
            let cpu_s = t.elapsed().as_secs_f64();
            let before = evaluator.context.stats();
            let q0 = evaluator.stats();
            let t = Instant::now();
            let got = new.evaluate_batch(&points, &ctx).unwrap();
            let new_s = t.elapsed().as_secs_f64();
            let error = got
                .residuals
                .iter()
                .zip(&expected.residuals)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(error < 1e-12);
            let q = evaluator.stats();
            let after = evaluator.context.stats();
            assert!(q.device_operands > q0.device_operands);
            eprintln!(
                "P8_WORKLOAD {}",
                serde_json::json!({"case":"cooke_ri_visibility","batch":batch,"run":run,"phase":phase(run),"cpu_seconds":cpu_s,"resident_seconds":new_s,"maximum_error":error,"commands":after.command_buffers-before.command_buffers,"device_seconds":after.device_seconds-before.device_seconds,"copied_bytes":after.copied_bytes-before.copied_bytes,"peak_plan_bytes":after.peak_bytes,"fallback_operands":q.fallback_operands-q0.fallback_operands,"host_completions":after.fallback_count-before.fallback_count,"device_operands":q.device_operands-q0.device_operands,"reference_seconds":q.reference_seconds-q0.reference_seconds,"reference_checks":q.reference_checks-q0.reference_checks})
            );
        }
    }
}
