//! Offline accuracy evidence for the unchecked Metal routes: every GPU-capable
//! analysis on every sample at default settings against the CPU f64 reference.
use mklight_core::{JobContext, samples};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;

#[test]
#[ignore = "offline default-grid f64 accuracy evidence; real Metal required"]
fn default_sample_accuracy() {
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
                "METAL_ACCURACY {}",
                json!({"sample":name,"analysis":info.name,"published":published,"backend":gpu.backend,
                "compared":report.compared,"maximum_absolute_error":report.maximum,"maximum_error_path":report.maximum_path,"maximum_image_l1_relative":report.image_l1_relative,
                "maximum_image_energy_relative":report.energy_relative,"previous_gate_failure":report.reason})
            );
            assert!(report.maximum.is_finite());
            if !published {
                assert!(gpu.backend.fallback_reason.is_some());
            }
        }
    }
}
