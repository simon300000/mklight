//! The `solvers` work-split variants (`mklight_optim::gpu::Split`) through
//! the engine: every variant runs `merit_evaluation`, `optimize`,
//! `tolerance` and `glass_substitution`, names itself in the backend report,
//! and agrees with the CPU f64 reference under the solver-outcome contract
//! of `support/comparison.rs`. The CPU-f32 emulation runs everywhere; the
//! Metal test is ignored (real Metal required, run through the benchmark
//! lock). One test per process section: the variant selection is
//! process-global.

use std::sync::Arc;

use mklight_core::{JobContext, param::ParamPath, samples, split};
use mklight_engine::analysis::AnalysisOutput;
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use mklight_optim::gpu::{SPLIT_STUDY, Split};
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

/// The merits a solver outcome is judged by: the final merit of an
/// optimisation, the finalists' merits of a glass substitution.
fn merits(analysis: &str, r: &AnalysisResult) -> Vec<f64> {
    match analysis {
        "optimize" => vec![r.json["final"]["merit"].as_f64().unwrap_or(f64::NAN)],
        "glass_substitution" => r.json["results"]
            .as_array()
            .map(|v| {
                v.iter()
                    .map(|x| x["merit"].as_f64().unwrap_or(f64::NAN))
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// Largest relative merit difference of `r` from the reference `cpu`.
fn deviation(analysis: &str, cpu: &AnalysisResult, r: &AnalysisResult) -> f64 {
    let (a, b) = (merits(analysis, cpu), merits(analysis, r));
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1e-300))
        .fold(0.0, f64::max)
}

/// Every variant on `systems` and `device` against the CPU reference: each
/// meets the solver-outcome contract, or, where the resident design (the
/// division before this study) misses it too, deviates from the reference
/// by at most twice as much as the resident design.
fn check(engine: &Engine, device: &str, systems: Vec<(&str, mklight_core::OpticalSystem)>) {
    let mut failures = Vec::new();
    for (name, system) in systems {
        let session = engine
            .session_from_project(Project::new(system.clone()))
            .unwrap();
        for (analysis, settings) in settings(&system) {
            split::reset();
            let Some(cpu) = run(&session, analysis, &settings, Backend::CpuReference) else {
                continue;
            };
            let mut resident: Option<(bool, f64)> = None;
            for variant in Split::ALL.map(Some).into_iter().chain([None]) {
                split::set(SPLIT_STUDY, variant.map(Split::name));
                let gpu = run(&session, analysis, &settings, Backend::Gpu(device.into()))
                    .unwrap_or_else(|| panic!("{name}/{analysis}/{variant:?} failed"));
                split::reset();
                let expected = variant.unwrap_or(Split::DEFAULT);
                assert_eq!(
                    gpu.json["backend"]["split"],
                    json!(expected.name()),
                    "{name}/{analysis}"
                );
                if expected == Split::CpuPar {
                    assert_eq!(gpu.backend.used, "cpu_reference", "{name}/{analysis}");
                }
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
                    "{name}/{analysis}/{}: used {} reason {:?} merit deviation {dev:.2e} \
                     cpu {:.1} ms device {:.1} ms",
                    expected.name(),
                    gpu.backend.used,
                    report.reason,
                    cpu.run_info.duration_ms,
                    gpu.run_info.duration_ms
                );
                if expected == Split::Resident {
                    resident = Some((report.reason.is_none(), dev));
                }
                let within_resident = matches!(resident, Some((false, r)) if dev <= 2. * r + 1e-12);
                if (report.reason.is_some() && !within_resident) || !native.is_empty() {
                    failures.push(format!(
                        "{name}/{analysis}/{}: {:?} {:?}",
                        expected.name(),
                        report.reason,
                        native
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
fn split_variants_on_emulation() {
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
#[ignore = "solver outcomes of every work-split variant; real Metal required"]
fn split_variants_on_metal() {
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
