//! The solver analyses on a GPU device through the engine, with each of
//! their aiming options (route option `solver_aiming`): the batch rule
//! (`by-batch`, the default), and device aiming (`device`) and host aiming
//! (`host`) pinned.
//! Every job (`merit_evaluation`, `optimize`, `tolerance`,
//! `glass_substitution`) names its aiming in the backend report
//! and agrees with the CPU f64 reference under the solver-outcome contract
//! of `support/comparison.rs`; for the truncated solves (`optimize` with 3
//! iterations, `glass_substitution` with 2 local iterations) the final
//! design's or a finalist's merit or violation outside its gate passes
//! within twice the reference's own movement (`support/solver_spread.rs`).
//! With device aiming, the solves a job runs in parallel give the same
//! result for any number of workers.
//! The CPU-f32 emulation runs everywhere; the Metal test is ignored (real
//! Metal required).

use std::sync::Arc;

use mklight_core::{JobContext, param::ParamPath, samples};
use mklight_engine::analysis::AnalysisOutput;
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::route::{RouteOptions, SolverAiming};
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

fn run(
    s: &Session,
    analysis: &str,
    settings: &Value,
    backend: Backend,
    solver_aiming: SolverAiming,
) -> Option<AnalysisResult> {
    let mut request = AnalysisRequest::new(analysis, settings.clone());
    request.backend = Some(backend);
    request.route = Some(RouteOptions {
        solver_aiming,
        ..Default::default()
    });
    s.run_analysis(&request, &JobContext::new()).ok()
}

/// The aiming options on `systems` and `device` against the CPU reference.
fn check(engine: &Engine, device: &str, systems: Vec<(&str, mklight_core::OpticalSystem)>) {
    let mut failures = Vec::new();
    for (name, system) in systems {
        let session = engine
            .session_from_project(Project::new(system.clone()))
            .unwrap();
        let nudged = engine
            .session_from_project(Project::new(solver_spread::nudged(&system)))
            .unwrap();
        for (analysis, settings) in settings(&system) {
            let reference = Backend::CpuReference;
            let Some(cpu) = run(
                &session,
                analysis,
                &settings,
                reference,
                SolverAiming::ByBatch,
            ) else {
                continue;
            };
            let spread =
                solver_spread::reference_movement(&session, &nudged, analysis, &settings, &cpu);
            for (aiming, expected) in [
                (SolverAiming::ByBatch, "by-batch"),
                (SolverAiming::Device, "device"),
                (SolverAiming::Host, "host"),
            ] {
                let gpu = Backend::Gpu(device.into());
                let gpu = run(&session, analysis, &settings, gpu, aiming)
                    .unwrap_or_else(|| panic!("{name}/{analysis}/{expected} failed"));
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
                if !solver_spread::within_movement(analysis, &report.failures, dev, spread)
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

/// Device aiming learns, solve by solve, which rays to route to the host.
/// Basin hopping runs its local solves in parallel and glass substitution
/// its re-optimisations: each is a solve with routing state of its own, so
/// the result (every value and counter) does not depend on how many
/// workers run them, or when. The real-aimed triplet at one and a half
/// times its field, and basin hopping's wide steps, give the device rays
/// it leaves to the host and so something to learn.
#[test]
fn device_aimed_parallel_solves_do_not_depend_on_the_workers() {
    let engine = engine_with(Some(GpuProvider::emulation()));
    let mut system = samples::cooke_triplet();
    system.ray_aiming = mklight_core::RayAiming::Real { tolerance: 1e-10 };
    for field in &mut system.fields.fields {
        field.y *= 1.5;
    }
    let session = engine
        .session_from_project(Project::new(system.clone()))
        .unwrap();
    let all = settings(&system);
    let glass = all.iter().find(|s| s.0 == "glass_substitution").unwrap();
    let mut hammer = all.iter().find(|s| s.0 == "optimize").unwrap().1.clone();
    hammer["algorithm"] = json!("hammer");
    hammer["max_iterations"] = json!(2);
    hammer["options"] = json!({"candidates_per_round": 8, "local_per_round": 4,
        "step": 2.0, "local": {"max_iterations": 2}});
    for (analysis, settings) in [("optimize", &hammer), ("glass_substitution", &glass.1)] {
        let run_in = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let gpu = Backend::Gpu("cpu_f32".into());
            let r = pool
                .install(|| run(&session, analysis, settings, gpu, SolverAiming::Device))
                .unwrap_or_else(|| panic!("{analysis} with {threads} workers failed"));
            let mut json = r.json;
            let stats = &json["backend"]["resident_evaluation"];
            assert!(stats["device_aimed_batches"].as_u64() > Some(0), "{stats}");
            // The one timing in the result.
            json["backend"]["gpu_stats"]["device_seconds"] = Value::Null;
            json
        };
        let one = run_in(1);
        for threads in [2, 8] {
            assert_eq!(one, run_in(threads), "{analysis} with {threads} workers");
        }
    }
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
