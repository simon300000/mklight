//! Offline accuracy evidence for the unchecked Metal routes: every GPU-capable
//! analysis on every sample at default settings against the CPU f64 reference,
//! under the per-quantity contract of `mklight_bench::comparison` (GPU §9.4),
//! and the solvers' physical outcomes. A native result outside its contract
//! fails, and so does a native attempt a reference fallback completed
//! (`RunInfo::failed_attempts`).
//!
//! cargo test --release -p mklight-bench --test tests metal_accuracy:: -- --ignored
//! --nocapture --test-threads=1
//! (`../README.md` lists every harness with its command.)
use mklight_bench::{comparison, solver, solver_spread};
use mklight_core::{JobContext, samples};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project};
use serde_json::{Value, json};

#[test]
#[ignore = "offline default-grid f64 accuracy evidence; real Metal required"]
fn default_sample_accuracy() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    mklight_bench::require_metal(&engine);
    sample_accuracy(&engine, "metal", None);
}

/// The same evidence for the WebGPU backend (GPU.md §11): every analysis it
/// publishes (`WEBGPU_ANALYSES`; the others run on the reference with that
/// reason) under the same contracts. Lines are tagged `WEBGPU_ACCURACY`.
#[test]
#[ignore = "offline default-grid f64 accuracy evidence; a WebGPU adapter required"]
fn webgpu_sample_accuracy() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:webgpu" && b.available),
        "a WebGPU adapter required"
    );
    sample_accuracy(&engine, "webgpu", None);
}

/// The same evidence for the CUDA backend (study; the WebGPU host
/// operations over CUDA kernels): every analysis it publishes under the same
/// contracts. Lines are tagged `CUDA_ACCURACY`.
#[test]
#[ignore = "offline default-grid f64 accuracy evidence; a CUDA device required"]
fn cuda_sample_accuracy() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:cuda" && b.available),
        "a CUDA device required"
    );
    sample_accuracy(&engine, "cuda", None);
}

/// `default_sample_accuracy` on the CPU-f32 emulation of the GPU kernels
/// (`gpu:cpu_f32`): every GPU analysis on every sample, same settings and
/// contracts. It guards the shared f32 specification that every GPU backend
/// follows, on any machine (release gate). Lines are tagged
/// `CPU_F32_ACCURACY`.
#[test]
#[ignore = "offline accuracy guard of the shared CPU-f32 specification (release gate)"]
fn emulation_sample_accuracy() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        register_emulation: true,
        ..Default::default()
    });
    sample_accuracy(&engine, "cpu_f32", None);
}

/// Every GPU-capable analysis on every sample on `gpu:<gpu>` against the
/// CPU reference (`default_sample_accuracy`); `only` restricts the analyses.
fn sample_accuracy(engine: &Engine, device: &str, only: Option<&[&str]>) {
    use mklight_engine::analysis::AnalysisOutput;
    let tag = device.to_uppercase();
    let id = format!("gpu:{device}");
    // The analyses this backend implements (explicit requests for others
    // are refused).
    let supported: Vec<String> = engine
        .backends()
        .into_iter()
        .find(|b| b.id == id)
        .map(|b| b.capabilities.analyses)
        .unwrap_or_default();
    let mut unmapped = Vec::new();
    let mut failing = 0;
    // Native attempts a fallback hid count as failures too.
    let mut attempts = Vec::new();
    for (name, system) in samples::all() {
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for info in engine.analyses().into_iter().filter(|a| {
            a.gpu_path
                && supported.contains(&a.name)
                && !solver::JOBS.contains(&a.name.as_str())
                && only.is_none_or(|o| o.contains(&a.name.as_str()))
        }) {
            let mut request = AnalysisRequest::new(&info.name, Value::Null);
            request.backend = Some(Backend::CpuReference);
            let cpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            request.backend = Some(Backend::Gpu(device.into()));
            let gpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            let published = gpu.backend.used == id;
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
                "{tag}_ACCURACY {}",
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
        "{tag}_ACCURACY_SUMMARY {}",
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
/// (`mklight_bench::solver_spread`); glass finalists the reference reports
/// infeasible are compared by structure only, each excused failure listed
/// under `excused`.
#[test]
#[ignore = "offline solver-outcome evidence; real Metal required"]
fn solver_outcomes() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    mklight_bench::require_metal(&engine);
    solver_outcomes_on(&engine, "metal", None);
}

/// `solver_outcomes` on the CPU-f32 emulation (`gpu:cpu_f32`): every solver
/// job on every sample, same settings and contracts (release gate).
#[test]
#[ignore = "offline solver-outcome guard of the shared CPU-f32 specification (release gate)"]
fn emulation_solver_outcomes() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        register_emulation: true,
        ..Default::default()
    });
    solver_outcomes_on(&engine, "cpu_f32", None);
}

/// `solver_outcomes` on `gpu:<device>`; `only` restricts the solver jobs.
fn solver_outcomes_on(engine: &Engine, device: &str, only: Option<&[&str]>) {
    use mklight_engine::analysis::AnalysisOutput;
    let id = format!("gpu:{device}");
    let mut failures = Vec::new();
    for (name, system) in samples::all() {
        // The benchmark's solver settings (project_bench): curvatures and
        // bounded thicknesses not controlled by a solve, one glass.
        let jobs = solver::jobs(&solver::variables(&system), &solver::glass(&system));
        let nudged = engine
            .session_from_project(Project::new(solver_spread::nudged(&system)))
            .unwrap();
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for (analysis, settings) in jobs {
            if only.is_some_and(|o| !o.contains(&analysis)) {
                continue;
            }
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
            request.backend = Some(Backend::Gpu(device.into()));
            let gpu = session.run_analysis(&request, &JobContext::new()).unwrap();
            let native_failures = comparison::native_failures(&gpu.run_info);
            let view = |r: &mklight_engine::AnalysisResult| AnalysisOutput {
                json: comparison::solver_scores(analysis, &r.json),
                ..Default::default()
            };
            let report =
                comparison::compare(&view(&cpu), &view(&gpu), analysis, &JobContext::new())
                    .unwrap();
            let published = gpu.backend.used == id;
            // Truncated solves: an outcome's merit or violation outside its
            // gate passes within twice the reference's own movement.
            // Infeasible glass finalists: structure only (`excuse_finalists`).
            let deviation = solver_spread::deviation(analysis, &cpu, &gpu);
            let runs = solver_spread::reference_runs(&session, &nudged, analysis, &settings);
            let movement = solver_spread::movement(analysis, &cpu, &runs);
            let (gated, excused) =
                solver_spread::excuse_finalists(analysis, &cpu, &runs, &report.failures);
            let within = solver_spread::within_movement(analysis, &gated, deviation, movement);
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
                "compared":report.compared,"gate_failures":gated,"excused":excused,"classes":report.classes,
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
