//! The solver analyses on a GPU device through the engine, with each of
//! their aiming options: the batch rule (the default), device aiming
//! (`aiming` = `device`) and host aiming (`aiming` = `host-table`) pinned. Every job (`merit_evaluation`, `optimize`,
//! `tolerance`, `glass_substitution`) names its aiming in the backend report
//! and agrees with the CPU f64 reference under the solver-outcome contract
//! of `support/comparison.rs`; for the truncated solves (`optimize` with 3
//! iterations, `glass_substitution` with 2 local iterations) a merit or
//! violation outside its gate passes within twice the reference's own
//! movement (`support/solver_spread.rs`). The CPU-f32 emulation runs
//! everywhere; the Metal test is ignored (real Metal required, run through
//! the benchmark lock). One test per process section: the aiming selection
//! is process-global.

use std::sync::Arc;

use mklight_core::{JobContext, param::ParamPath, samples, split};
use mklight_engine::analysis::AnalysisOutput;
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;
#[path = "support/solver_spread.rs"]
mod solver_spread;

fn settings(system: &mklight_core::OpticalSystem) -> Vec<(&'static str, Value)> {
    let vars: Vec<Value> = system.surfaces[1..system.surfaces.len() - 1]
        .iter()
        .filter(|s| s.shape.curvature() != 0.)
        .map(|s| json!({"param": ParamPath::Curvature { surface: s.id }}))
        .collect();
    let glass = system.surfaces[1..]
        .iter()
        .find(|s| matches!(s.material, mklight_core::MaterialRef::Catalog { .. }))
        .map(|g| vec![json!({"surface": g.id, "candidates": {"nearest": 3}})])
        .unwrap_or_default();
    vec![
        ("merit_evaluation", Value::Null),
        (
            "optimize",
            json!({"variables": vars, "max_iterations": 3,
                   "options": {"ftol": 0., "xtol": 0., "gtol": 0.}, "seed": 71}),
        ),
        (
            "tolerance",
            json!({"trials": 16, "seed": 71, "record_trials": 16}),
        ),
        (
            "glass_substitution",
            json!({"variables": vars, "material_variables": glass, "max_combinations": 3,
                   "local_iterations": 2, "top_n": 3, "seed": 71}),
        ),
    ]
}

fn run(s: &Session, analysis: &str, settings: &Value, backend: Backend) -> Option<AnalysisResult> {
    let mut request = AnalysisRequest::new(analysis, settings.clone());
    request.backend = Some(backend);
    s.run_analysis(&request, &JobContext::new()).ok()
}

/// The aiming options on `systems` and `device` against the CPU reference.
fn check(engine: &Engine, device: &str, systems: Vec<(&str, mklight_core::OpticalSystem)>) {
    // The aiming selection is process-wide: one test selects at a time.
    static SELECTION: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _selection = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let mut failures = Vec::new();
    for (name, system) in systems {
        let session = engine
            .session_from_project(Project::new(system.clone()))
            .unwrap();
        let nudged = engine
            .session_from_project(Project::new(solver_spread::nudged(&system)))
            .unwrap();
        for (analysis, settings) in settings(&system) {
            split::reset();
            let Some(cpu) = run(&session, analysis, &settings, Backend::CpuReference) else {
                continue;
            };
            let spread =
                solver_spread::reference_movement(&session, &nudged, analysis, &settings, &cpu);
            for (aiming, expected) in [
                (None, "by-batch"),
                (Some("device"), "device"),
                (Some("host-table"), "host"),
            ] {
                split::set("aiming", aiming);
                let gpu = run(&session, analysis, &settings, Backend::Gpu(device.into()))
                    .unwrap_or_else(|| panic!("{name}/{analysis}/{expected} failed"));
                split::reset();
                assert_eq!(
                    gpu.json["backend"]["aiming"],
                    json!(expected),
                    "{name}/{analysis}"
                );
                let view = |r: &AnalysisResult| AnalysisOutput {
                    json: comparison::solver_scores(analysis, &r.json),
                    ..Default::default()
                };
                let report =
                    comparison::compare(&view(&cpu), &view(&gpu), analysis, &JobContext::new())
                        .unwrap();
                let native = comparison::native_failures(&gpu.run_info);
                let dev = solver_spread::deviation(analysis, &cpu, &gpu);
                eprintln!(
                    "{name}/{analysis}/{expected} aiming: used {} reason {:?} merit, violation \
                     deviation {:.2e}, {:.2e} (reference moves {:.2e}, {:.2e}) cpu {:.1} ms \
                     device {:.1} ms",
                    gpu.backend.used,
                    report.reason,
                    dev.0,
                    dev.1,
                    spread.0,
                    spread.1,
                    cpu.run_info.duration_ms,
                    gpu.run_info.duration_ms
                );
                if !solver_spread::within_movement(&report.failures, dev, spread)
                    || !native.is_empty()
                {
                    failures.push(format!(
                        "{name}/{analysis}/{expected}: {:?} {:?} deviation {dev:?} \
                         (reference moves {spread:?})",
                        report.failures, native
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn engine_with(p: Option<GpuProvider>) -> Engine {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    if let Some(p) = p {
        e.register_backend(Arc::new(p));
    }
    e
}

#[test]
fn aiming_options_on_emulation() {
    let engine = engine_with(Some(GpuProvider::emulation()));
    let mut real = samples::cooke_triplet();
    real.ray_aiming = mklight_core::RayAiming::Real { tolerance: 1e-10 };
    check(
        &engine,
        "cpu_f32",
        vec![
            ("cooke_triplet", samples::cooke_triplet()),
            ("cooke_triplet real aiming", real),
        ],
    );
}

#[test]
#[ignore = "solver outcomes of the aiming options; real Metal required"]
fn aiming_options_on_metal() {
    let engine = engine_with(None);
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:metal" && b.available),
        "real Metal required"
    );
    check(&engine, "metal", samples::all());
}
