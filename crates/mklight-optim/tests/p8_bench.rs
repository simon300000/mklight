#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, param::ParamPath};
use mklight_gpu::ComputeBackend;
use mklight_optim::{gpu::*, merit::default::DefaultMeritSpec, merit::*, problem::*};
use std::{sync::Arc, time::Instant};
#[test]
#[ignore = "provisional host benchmark; ten measured runs after cold and warm-up"]
fn candidate_sweep() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    eprintln!(
        "P8_HARDWARE {}",
        serde_json::json!({"device":backend.info(),"precision":backend.precision(),"timings":"provisional","seed":0})
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
                ..Default::default()
            }
            .generate(&system)
            .unwrap();
            let vars = vec![VariableSpec::new(variable.clone())];
            let registry = OperandRegistry::builtin();
            let ctx = JobContext::new();
            let reference = BuiltinEvaluator::builtin();
            let legacy = GpuTraceEvaluator::new(
                backend.clone(),
                reference.trace.clone(),
                &system,
                Default::default(),
                &ctx,
            )
            .unwrap();
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
                    serde_json::json!({"sample":name,"batch":b,"run":run,"phase":if run==0{"cold"}else if run==1{"warmup"}else{"measured"},"cpu_seconds":cpu_s,"old_seconds":old_s,"resident_seconds":new_s,"device_seconds":after.device_seconds-stats.device_seconds,"commands":after.command_buffers-stats.command_buffers,"copied_bytes":after.copied_bytes-stats.copied_bytes+mem.copied_bytes-cache.copied_bytes,"allocations":after.allocations-stats.allocations,"reused":after.reused_allocations-stats.reused_allocations,"peak_plan_bytes":after.peak_bytes,"pool_bytes":mem.pooled_bytes,"qualification_seconds":q.reference_seconds-qual.reference_seconds,"fallback_operands":q.fallback_operands-qual.fallback_operands,"legacy_stats":legacy.stats()})
                );
            }
        }
    }
}
