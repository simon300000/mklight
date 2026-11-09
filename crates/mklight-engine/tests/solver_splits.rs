//! The solver analyses on a GPU device through the engine, with each of
//! their aiming options: host aiming (the default) and device aiming
//! (`aiming` = `device`). Every job (`merit_evaluation`, `optimize`,
//! `tolerance`, `glass_substitution`) names its aiming in the backend report
//! and agrees with the CPU f64 reference under the solver-outcome contract
//! of `support/comparison.rs`. For the truncated solves (`optimize` with 3
//! iterations, `glass_substitution` with 2 local iterations) a merit or
//! violation outside its gate passes when it deviates from the reference by
//! at most twice as much as the reference deviates from itself after a
//! 1e-9 relative change of one curvature: their outcomes follow the
//! least-squares trajectory, which perturbations far below the contract's
//! gates move by more than the gates on several samples. The CPU-f32 emulation runs everywhere;
//! the Metal test is ignored (real Metal required, run through the
//! benchmark lock). One test per process section: the aiming selection is
//! process-global.

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

/// The values (`merit` or `violation`) a solver outcome is judged by: the
/// final design's of an optimisation, the finalists' of a glass
/// substitution.
fn scores(analysis: &str, r: &AnalysisResult, key: &str) -> Vec<f64> {
    match analysis {
        "optimize" => vec![r.json["final"][key].as_f64().unwrap_or(f64::NAN)],
        "glass_substitution" => r.json["results"]
            .as_array()
            .map(|v| {
                v.iter()
                    .map(|x| x[key].as_f64().unwrap_or(f64::NAN))
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// Largest difference of `r` from the reference `cpu`: (merits, relative;
/// violations, absolute).
fn deviation(analysis: &str, cpu: &AnalysisResult, r: &AnalysisResult) -> (f64, f64) {
    let largest = |key: &str, relative: bool| {
        let (a, b) = (scores(analysis, cpu, key), scores(analysis, r, key));
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if relative { d / x.abs().max(1e-300) } else { d }
            })
            .fold(0.0, f64::max)
    };
    (largest("merit", true), largest("violation", false))
}

/// `system` with its first curvature changed by 1e-9 (relative).
fn nudged(system: &mklight_core::OpticalSystem) -> mklight_core::OpticalSystem {
    let mut s = system.clone();
    if let Some(id) = s.surfaces[1..]
        .iter()
        .find(|x| x.shape.curvature() != 0.)
        .map(|x| x.id)
    {
        let p = ParamPath::Curvature { surface: id };
        let c = p.get(&s).unwrap();
        p.set(&mut s, c * (1. + 1e-9)).unwrap();
    }
    s
}

/// Both aiming options on `systems` and `device` against the CPU reference.
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
            .session_from_project(Project::new(nudged(&system)))
            .unwrap();
        for (analysis, settings) in settings(&system) {
            split::reset();
            let Some(cpu) = run(&session, analysis, &settings, Backend::CpuReference) else {
                continue;
            };
            // The reference's own sensitivity of a truncated solve.
            let spread = run(&nudged, analysis, &settings, Backend::CpuReference)
                .map_or((0., 0.), |r| deviation(analysis, &cpu, &r));
            for (aiming, expected) in [(None, "host"), (Some("device"), "device")] {
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
                let dev = deviation(analysis, &cpu, &gpu);
                eprintln!(
                    "{name}/{analysis}/{expected} aiming: used {} reason {:?} merit, violation \
                     deviation {:.2e}, {:.2e} (reference nudged {:.2e}, {:.2e}) cpu {:.1} ms \
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
                // Every gate failure must be a merit or violation within
                // twice the reference's own deviation.
                let within_spread = |failure: &String| {
                    (failure.contains(".merit:") && dev.0 <= 2. * spread.0 + 1e-12)
                        || (failure.contains(".violation:") && dev.1 <= 2. * spread.1 + 1e-12)
                };
                if !report.failures.iter().all(within_spread) || !native.is_empty() {
                    failures.push(format!(
                        "{name}/{analysis}/{expected}: {:?} {:?} deviation {dev:?} \
                         (reference nudged {spread:?})",
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
#[ignore = "solver outcomes of both aiming options; real Metal required"]
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
