//! Offline accuracy evidence for the unchecked Metal routes: every GPU-capable
//! analysis on every sample at default settings against the CPU f64 reference,
//! under the per-quantity contract of `support/comparison.rs` (GPU §9.4),
//! and the solvers' physical outcomes. A native result outside its contract
//! fails, and so does a native attempt a reference fallback completed
//! (`RunInfo::failed_attempts`).
use mklight_core::{JobContext, param::ParamPath, samples};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;
#[path = "support/solver_spread.rs"]
mod solver_spread;

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
    let mut unmapped = Vec::new();
    let mut failing = 0;
    // Native attempts a fallback hid count as failures too.
    let mut attempts = Vec::new();
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
            let native_failures = comparison::native_failures(&gpu.run_info);
            attempts.extend(
                native_failures
                    .iter()
                    .map(|f| format!("{name}/{}: {f}", info.name)),
            );
            let cpu = AnalysisOutput {
                json: cpu.json,
                buffers: cpu.buffers,
                ..Default::default()
            };
            let gpu_out = AnalysisOutput {
                json: gpu.json,
                buffers: gpu.buffers,
                ..Default::default()
            };
            let report =
                comparison::compare(&cpu, &gpu_out, &info.name, &JobContext::new()).unwrap();
            unmapped.extend(
                report
                    .failures
                    .iter()
                    .filter(|f| f.contains("unmapped"))
                    .cloned(),
            );
            failing += usize::from(published && report.reason.is_some());
            println!(
                "METAL_ACCURACY {}",
                json!({"sample":name,"analysis":info.name,"published":published,"backend":gpu.backend,
                "compared":report.compared,"maximum_absolute_error":report.maximum,"maximum_error_path":report.maximum_path,"maximum_image_l1_relative":report.image_l1_relative,
                "maximum_image_energy_relative":report.energy_relative,"gate_failure":report.reason,
                "gate_failures":report.failures,"classes":report.classes,"resampled":report.resampled,
                "sampling_differences":report.sampling_differences,"undefined":report.undefined,
                "native_failures":native_failures})
            );
            assert!(report.maximum.is_finite());
            if !published {
                assert!(gpu.backend.fallback_reason.is_some());
            }
        }
    }
    println!(
        "METAL_ACCURACY_SUMMARY {}",
        json!({"native_gate_failures":failing,"native_attempt_failures":attempts})
    );
    // A field the contract does not class is a harness defect, never a pass.
    assert!(
        unmapped.is_empty(),
        "unmapped result fields:\n{}",
        unmapped.join("\n")
    );
    assert_eq!(failing, 0, "native results outside their contract");
    assert!(
        attempts.is_empty(),
        "native attempts hidden by a fallback:\n{}",
        attempts.join("\n")
    );
}

/// The solvers' physical outcomes (merit, violation, feasibility, finalists,
/// tolerance figures; `comparison::SOLVER_SCORES`) on Metal against the CPU
/// reference on every sample at default settings. The truncated solves'
/// outcomes (the final design's or a finalist's merit or violation) outside
/// their gates pass within twice the reference's own movement
/// (`support/solver_spread.rs`).
#[test]
#[ignore = "offline solver-outcome evidence; real Metal required"]
fn solver_outcomes() {
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
    let mut failures = Vec::new();
    for (name, system) in samples::all() {
        // The benchmark's solver settings (project_bench): curvatures and
        // bounded thicknesses not controlled by a solve, one glass.
        let mut vars = Vec::new();
        for s in &system.surfaces[1..system.surfaces.len() - 1] {
            if s.shape.curvature() != 0. {
                vars.push(json!({"param":ParamPath::Curvature{surface:s.id}}));
            }
            if s.thickness.is_finite() && s.thickness > 0. {
                vars.push(json!({"param":ParamPath::Thickness{surface:s.id},
                    "lower":s.thickness * 0.8,"upper":s.thickness * 1.2}));
            }
        }
        vars.retain(|v| !system.solves.iter().any(|s| json!(s.target) == v["param"]));
        let glass = system.surfaces[1..]
            .iter()
            .find(|s| matches!(s.material, mklight_core::MaterialRef::Catalog { .. }))
            .map(|g| vec![json!({"surface":g.id,"candidates":{"nearest":3}})])
            .unwrap_or_default();
        let nudged = engine
            .session_from_project(Project::new(solver_spread::nudged(&system)))
            .unwrap();
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for (analysis, settings) in [
            ("merit_evaluation", Value::Null),
            (
                "optimize",
                json!({"variables":vars,"max_iterations":3,"options":{"ftol":0.,"xtol":0.,"gtol":0.},"seed":71}),
            ),
            (
                "tolerance",
                json!({"trials":16,"seed":71,"record_trials":16}),
            ),
            (
                "glass_substitution",
                json!({"variables":vars,"material_variables":glass,"max_combinations":3,
                    "local_iterations":2,"top_n":3,"seed":71}),
            ),
        ] {
            let mut request = AnalysisRequest::new(analysis, settings.clone());
            request.backend = Some(Backend::CpuReference);
            let cpu = match session.run_analysis(&request, &JobContext::new()) {
                Ok(r) => r,
                Err(e) => {
                    println!(
                        "SOLVER_OUTCOME {}",
                        json!({"sample":name,"analysis":analysis,"skipped":e.to_string()})
                    );
                    continue;
                }
            };
            request.backend = Some(Backend::Gpu("metal".into()));
            let gpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            let native_failures = comparison::native_failures(&gpu.run_info);
            let view = |r: &mklight_engine::AnalysisResult| AnalysisOutput {
                json: comparison::solver_scores(analysis, &r.json),
                ..Default::default()
            };
            let report =
                comparison::compare(&view(&cpu), &view(&gpu), analysis, &JobContext::new())
                    .unwrap();
            let published = gpu.backend.used == "gpu:metal";
            // Truncated solves: an outcome's merit or violation outside its
            // gate passes within twice the reference's own movement.
            let deviation = solver_spread::deviation(analysis, &cpu, &gpu);
            let movement =
                solver_spread::reference_movement(&session, &nudged, analysis, &settings, &cpu);
            let within =
                solver_spread::within_movement(analysis, &report.failures, deviation, movement);
            if (published && !within) || !native_failures.is_empty() {
                failures.push(format!(
                    "{name}/{analysis}: {:?} {:?}",
                    report.reason, native_failures
                ));
            }
            let finalists = |r: &mklight_engine::AnalysisResult| -> Vec<Value> {
                r.json["results"]
                    .as_array()
                    .map(|v| v.iter().map(|x| x["merit"].clone()).collect())
                    .unwrap_or_default()
            };
            println!(
                "SOLVER_OUTCOME {}",
                json!({"sample":name,"analysis":analysis,"published":published,"used":gpu.backend.used,
                "compared":report.compared,"gate_failures":report.failures,"classes":report.classes,
                "native_failures":native_failures,"cpu_ms":cpu.run_info.duration_ms,"metal_ms":gpu.run_info.duration_ms,
                "merits":[&cpu.json["final"]["merit"], &gpu.json["final"]["merit"]],
                "deviation":[deviation.0, deviation.1],"reference_movement":[movement.0, movement.1],
                "finalist_merits":[finalists(&cpu), finalists(&gpu)]})
            );
        }
    }
    println!("SOLVER_OUTCOME_SUMMARY {}", json!({"failures":failures}));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
