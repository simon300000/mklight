#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, param::ParamPath};
use mklight_gpu::ComputeBackend;
use mklight_optim::{gpu::*, merit::default::DefaultMeritSpec, merit::*, problem::*};
use std::{sync::Arc, time::Instant};
/// Per-stage wall/device/command deltas between two cumulative snapshots.
fn stage_deltas(
    before: &mklight_gpu::resident::PlanStats,
    after: &mklight_gpu::resident::PlanStats,
) -> serde_json::Value {
    after
        .stages
        .iter()
        .map(|s| {
            let b = before.stages.iter().find(|b| b.name == s.name);
            let d = |f: fn(&mklight_gpu::resident::StageStats) -> f64| f(s) - b.map_or(0., f);
            (
                s.name.clone(),
                serde_json::json!([
                    d(|s| s.wall_seconds),
                    d(|s| s.device_seconds),
                    s.command_buffers - b.map_or(0, |b| b.command_buffers)
                ]),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}
#[test]
#[ignore = "provisional host benchmark; ten measured runs after cold and warm-up"]
fn candidate_sweep() {
    let setup = Instant::now();
    let backend: Arc<dyn ComputeBackend> = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    eprintln!(
        "P8_HARDWARE {}",
        serde_json::json!({"device":backend.info(),"precision":backend.precision(),"timings":"provisional","seed":0,"backend_setup_seconds":setup.elapsed().as_secs_f64()})
    );
    for (name, system) in mklight_core::samples::all() {
        if std::env::var("P8_SAMPLE").is_ok_and(|v| v != name) {
            continue;
        }
        let variable = ParamPath::Thickness {
            surface: system.surfaces[system.surfaces.len() - 2].id,
        };
        let nominal = variable.get(&system).unwrap();
        for b in [1, 16, 256, 1024] {
            if std::env::var("P8_BATCH").is_ok_and(|v| v != b.to_string()) {
                continue;
            }
            let spec = DefaultMeritSpec {
                thickness: None,
                criterion: if std::env::var("P8_MERIT").is_ok_and(|v| v == "wave") {
                    mklight_optim::merit::default::DefaultCriterion::Wavefront {
                        reference: mklight_optim::merit::operand::WavefrontReference::Centroid,
                    }
                } else {
                    Default::default()
                },
                ..Default::default()
            }
            .generate(&system)
            .unwrap();
            let vars = vec![VariableSpec::new(variable.clone())];
            let registry = OperandRegistry::builtin();
            let ctx = JobContext::new();
            let reference = BuiltinEvaluator::builtin();
            let old_setup = Instant::now();
            let legacy = GpuTraceEvaluator::new(
                backend.clone(),
                reference.trace.clone(),
                Default::default(),
            );
            let old_setup_seconds = old_setup.elapsed().as_secs_f64();
            let new_setup = Instant::now();
            let resident = ResidentEvaluator::new(
                GpuContext::new(backend.clone(), Snapshot::new(system.clone()).unwrap()),
                reference.clone(),
            );
            let new_setup_seconds = new_setup.elapsed().as_secs_f64();
            let cpu = OpticalProblem::new(
                system.clone(),
                vars.clone(),
                spec.clone(),
                reference,
                &registry,
            )
            .unwrap();
            let old = OpticalProblem::new(
                system.clone(),
                vars.clone(),
                spec.clone(),
                legacy.clone(),
                &registry,
            )
            .unwrap();
            let new = OpticalProblem::new(system.clone(), vars, spec, resident.clone(), &registry)
                .unwrap();
            let points: Vec<_> = (0..b)
                .map(|i| [nominal + 0.001 * (i as f64 / (b as f64) - 0.5)])
                .collect();
            let points = CandidateBatch::from_points(1, points.iter().map(|p| &p[..]));
            for run in 0..12 {
                let t = Instant::now();
                let reference = cpu.evaluate_batch(&points, &ctx).unwrap();
                let cpu_s = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let old_values = old.evaluate_batch(&points, &ctx).unwrap();
                let old_s = t.elapsed().as_secs_f64();
                let cache = backend.resident_cache_stats();
                let stats = resident.context.stats();
                let qual = resident.stats();
                let t = Instant::now();
                let values = new.evaluate_batch(&points, &ctx).unwrap();
                let new_s = t.elapsed().as_secs_f64();
                let after = resident.context.stats();
                let q = resident.stats();
                let mem = backend.resident_cache_stats();
                assert_eq!(reference.status, values.status);
                assert_eq!(reference.status, old_values.status);
                assert!(
                    q.device_operands > qual.device_operands,
                    "{name} B={b}: {q:?}"
                );
                eprintln!(
                    "P8_BENCH {}",
                    serde_json::json!({"sample":name,"merit":std::env::var("P8_MERIT").unwrap_or_else(|_|"spot".into()),"batch":b,"run":run,"phase":if run==0{"cold"}else if run==1{"warmup"}else{"measured"},"cpu_seconds":cpu_s,"old_seconds":old_s,"resident_seconds":new_s,"device_seconds":after.device_seconds-stats.device_seconds,"commands":after.command_buffers-stats.command_buffers,"copied_bytes":mem.copied_bytes-cache.copied_bytes,"context_copied_bytes":after.copied_bytes-stats.copied_bytes,"allocations":after.allocations-stats.allocations,"reused":after.reused_allocations-stats.reused_allocations,"peak_plan_bytes":after.peak_bytes,"pool_bytes":mem.pooled_bytes,"fallback_operands":q.fallback_operands-qual.fallback_operands,"legacy_stats":legacy.stats(),"old_setup_seconds":old_setup_seconds,"new_setup_seconds":new_setup_seconds,"device_operands":q.device_operands-qual.device_operands,"stages":stage_deltas(&stats,&after)})
                );
            }
        }
    }
}

#[test]
#[ignore = "provisional dense complete-merit benchmark"]
fn dense_rays() {
    use mklight_optim::merit::operand::SpotReference;
    let backend: Arc<dyn ComputeBackend> = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    for (name, system) in mklight_core::samples::all()
        .into_iter()
        .filter(|(name, _)| {
            [
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "anamorphic",
                "hybrid_doe",
            ]
            .contains(name)
        })
    {
        let param = ParamPath::Thickness {
            surface: system.surfaces[system.surfaces.len() - 2].id,
        };
        let nominal = param.get(&system).unwrap();
        for (rings, arms, b) in [(5, 20, 1), (100, 100, 1), (100, 100, 100)] {
            let sampling = PupilSampling {
                rings,
                arms,
                ..Default::default()
            };
            let spec = MeritFunctionSpec::new(vec![
                Operand::new(if std::env::var("P8_MERIT").is_ok_and(|v| v == "wave") {
                    OperandKind::WavefrontRms {
                        reference: mklight_optim::merit::operand::WavefrontReference::Centroid,
                        sampling,
                    }
                } else {
                    OperandKind::RmsSpot {
                        reference: SpotReference::Centroid,
                        component: None,
                        sampling,
                    }
                })
                .at_field(system.fields.fields.len() - 1),
            ]);
            let vars = vec![VariableSpec::new(param.clone())];
            let registry = OperandRegistry::builtin();
            let ctx = JobContext::new();
            let reference = BuiltinEvaluator::builtin();
            let old = GpuTraceEvaluator::new(
                backend.clone(),
                reference.trace.clone(),
                Default::default(),
            );
            let resident = ResidentEvaluator::new(
                GpuContext::new(backend.clone(), Snapshot::new(system.clone()).unwrap()),
                reference.clone(),
            );
            let cpu = OpticalProblem::new(
                system.clone(),
                vars.clone(),
                spec.clone(),
                reference,
                &registry,
            )
            .unwrap();
            let old_problem = OpticalProblem::new(
                system.clone(),
                vars.clone(),
                spec.clone(),
                old.clone(),
                &registry,
            )
            .unwrap();
            let new = OpticalProblem::new(system.clone(), vars, spec, resident.clone(), &registry)
                .unwrap();
            let points: Vec<_> = (0..b)
                .map(|i| [nominal + 0.001 * (i as f64 / b as f64 - 0.5)])
                .collect();
            let points = CandidateBatch::from_points(1, points.iter().map(|p| &p[..]));
            for run in 0..12 {
                let t = Instant::now();
                let expected = cpu.evaluate_batch(&points, &ctx).unwrap();
                let cpu_s = t.elapsed().as_secs_f64();
                let t = Instant::now();
                old_problem.evaluate_batch(&points, &ctx).unwrap();
                let old_s = t.elapsed().as_secs_f64();
                let before = resident.context.stats();
                let q0 = resident.stats();
                let cache = backend.resident_cache_stats();
                let t = Instant::now();
                let got = new.evaluate_batch(&points, &ctx).unwrap();
                let new_s = t.elapsed().as_secs_f64();
                let stats = resident.context.stats();
                let q = resident.stats();
                let mem = backend.resident_cache_stats();
                assert_eq!(expected.status, got.status);
                assert!(q.device_operands > q0.device_operands, "{name}: {q:?}");
                eprintln!(
                    "P8_DENSE {}",
                    serde_json::json!({"sample":name,"merit":std::env::var("P8_MERIT").unwrap_or_else(|_|"spot".into()),"rings":rings,"arms":arms,"field":system.fields.fields.len()-1,"wavelength":"primary","batch":b,"rays":rings*arms*b,"run":run,"phase":if run==0{"cold"}else if run==1{"warmup"}else{"measured"},"cpu_seconds":cpu_s,"old_seconds":old_s,"resident_seconds":new_s,"device_seconds":stats.device_seconds-before.device_seconds,"commands":stats.command_buffers-before.command_buffers,"copied_bytes":mem.copied_bytes-cache.copied_bytes,"context_copied_bytes":stats.copied_bytes-before.copied_bytes,"allocations":stats.allocations-before.allocations,"reused":stats.reused_allocations-before.reused_allocations,"peak_plan_bytes":stats.peak_bytes,"pool_bytes":mem.pooled_bytes,"fallback_operands":q.fallback_operands-q0.fallback_operands,"device_operands":q.device_operands-q0.device_operands,"legacy_stats":old.stats()})
                );
            }
        }
    }
}
