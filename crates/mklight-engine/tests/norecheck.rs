//! Provisional normal-load before/after benchmark. Explicit backend requests
//! preserve the P11 default fixtures independently of Auto cost policy.
use mklight_core::{JobContext, samples};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project};
use serde_json::{Value, json};
use std::time::Instant;

fn settings(analysis: &str) -> Value {
    if analysis == "glass_substitution" {
        json!({"material_variables":[
            {"surface":1,"candidates":{"nearest":3}},
            {"surface":3,"candidates":{"nearest":3}},
            {"surface":5,"candidates":{"nearest":3}}],
            "max_combinations":12,"local_iterations":8,
            "default_merit":{"fields":[0,2],"wavelengths":[1]},
            "backend":"auto","seed":17})
    } else {
        Value::Null
    }
}

#[test]
#[ignore = "normal-load real Metal measurement; ten cold and ten warm pairs"]
fn before_after() {
    use mklight_engine::backend::BackendProvider;
    let filter = std::env::var("NORECHECK_ANALYSES").ok();
    let sample = std::env::var("NORECHECK_SAMPLE").ok();
    let catalog = Engine::default().analyses();
    for (name, system) in samples::all() {
        if sample.as_ref().is_some_and(|s| s != name) {
            continue;
        }
        for info in &catalog {
            if !info.gpu_path
                || (info.name == "glass_substitution" && name != "cooke_triplet")
                || filter
                    .as_ref()
                    .is_some_and(|f| !f.split(',').any(|n| n == info.name))
            {
                continue;
            }
            let create = |backend: Backend| {
                let start = Instant::now();
                let engine = Engine::new(EngineConfig {
                    default_backend: Backend::CpuReference,
                    ..Default::default()
                });
                if matches!(backend, Backend::Gpu(_)) {
                    let provider = mklight_engine::gpu::GpuProvider::metal();
                    assert!(
                        provider.device().is_some(),
                        "real Metal required: {:?}",
                        provider.info().reason
                    );
                    engine.register_backend(std::sync::Arc::new(provider));
                }
                let session = engine
                    .session_from_project(Project::new(system.clone()))
                    .unwrap();
                let mut request = AnalysisRequest::new(&info.name, settings(&info.name));
                request.backend = Some(backend);
                (engine, session, request, start)
            };
            // Independent cold engines prevent CPU calls from warming GPU caches.
            for run in 0..12 {
                let execute = |backend| {
                    let (_engine, session, request, start) = create(backend);
                    let result = session.run_analysis(&request, &JobContext::new()).unwrap();
                    (start.elapsed().as_secs_f64(), result)
                };
                let (cpu, metal) = if run % 2 == 0 {
                    let c = execute(Backend::CpuReference);
                    (c, execute(Backend::Gpu("metal".into())))
                } else {
                    let g = execute(Backend::Gpu("metal".into()));
                    (execute(Backend::CpuReference), g)
                };
                println!(
                    "NORECHECK {}",
                    json!({"sample":name,"analysis":info.name,"temperature":"cold","run":run,"measured":run>=2,"cpu_seconds":cpu.0,"metal_seconds":metal.0,"used":metal.1.backend.used,"backend":metal.1.backend,"settings":cpu.1.run_info.settings})
                );
            }
            let (_cpu_engine, cpu_session, cpu_request, _) = create(Backend::CpuReference);
            let (_gpu_engine, gpu_session, gpu_request, _) = create(Backend::Gpu("metal".into()));
            for run in 0..12 {
                let cpu = || {
                    let t = Instant::now();
                    let r = cpu_session
                        .run_analysis(&cpu_request, &JobContext::new())
                        .unwrap();
                    (t.elapsed().as_secs_f64(), r)
                };
                let metal = || {
                    let t = Instant::now();
                    let r = gpu_session
                        .run_analysis(&gpu_request, &JobContext::new())
                        .unwrap();
                    (t.elapsed().as_secs_f64(), r)
                };
                let (c, g) = if run % 2 == 0 {
                    let c = cpu();
                    (c, metal())
                } else {
                    let g = metal();
                    (cpu(), g)
                };
                println!(
                    "NORECHECK {}",
                    json!({"sample":name,"analysis":info.name,"temperature":"warm","run":run,"measured":run>=2,"cpu_seconds":c.0,"metal_seconds":g.0,"used":g.1.backend.used,"backend":g.1.backend,"settings":c.1.run_info.settings})
                );
            }
        }
    }
}

#[path = "support/comparison.rs"]
mod comparison;

#[test]
#[ignore = "offline default-grid f64 accuracy evidence; real Metal required"]
fn unchecked_accuracy() {
    use mklight_engine::analysis::AnalysisOutput;
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:metal" && b.available),
        "real Metal required"
    );
    for (name, system) in samples::all() {
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for info in engine.analyses().into_iter().filter(|a| {
            a.gpu_path
                && !matches!(
                    a.name.as_str(),
                    "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
                )
        }) {
            let mut request = AnalysisRequest::new(&info.name, Value::Null);
            request.backend = Some(Backend::CpuReference);
            let cpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            request.backend = Some(Backend::Gpu("metal".into()));
            let gpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            let published = gpu.backend.used == "gpu:metal";
            let report = comparison::compare(
                &AnalysisOutput {
                    json: cpu.json,
                    buffers: cpu.buffers,
                    ..Default::default()
                },
                &AnalysisOutput {
                    json: gpu.json,
                    buffers: gpu.buffers,
                    ..Default::default()
                },
                &info.name,
                &JobContext::new(),
            )
            .unwrap();
            println!(
                "NORECHECK_ACCURACY {}",
                json!({"sample":name,"analysis":info.name,"published":published,"backend":gpu.backend,
                "compared":report.compared,"maximum_absolute_error":report.maximum,"maximum_image_l1_relative":report.image_l1_relative,
                "maximum_image_energy_relative":report.energy_relative,"previous_gate_failure":report.reason})
            );
            assert!(report.maximum.is_finite());
            if !published {
                assert!(gpu.backend.fallback_reason.is_some());
            }
        }
    }
}
