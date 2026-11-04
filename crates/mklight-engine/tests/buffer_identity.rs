//! Result-identity and output-cost probe for the typed result paths.
//!
//! Prints one `BUFFER_IDENTITY` JSON line per analysis call: BLAKE3 of the
//! canonical result JSON and of every buffer's name, dtype, shape and raw
//! bytes, plus wall and Metal device time. Builds of two revisions are
//! compared line by line; equal hashes prove byte-identical results. When
//! `cpu` precedes another mode, a `BUFFER_CONTRACT` line also judges that
//! mode's result against the CPU f64 one under the per-quantity contract.
//!
//! cargo test --release -p mklight-engine --test buffer_identity -- --ignored
//! --nocapture --test-threads=1
//! Optional: MKLIGHT_IDENTITY_PROJECT (a local project), MKLIGHT_IDENTITY_SAMPLES=0
//! (skip the sample designs), MKLIGHT_IDENTITY_MODES (default cpu,metal),
//! MKLIGHT_IDENTITY_CASES (JSON [{name, analysis, settings}] replacing the
//! default list), MKLIGHT_IDENTITY_RUNS (timed calls per case, default 1),
//! MKLIGHT_IDENTITY_GRAPH_BYTES (analysis graph budget; 0 disables it).
#![cfg(target_os = "macos")]

use std::path::Path;
use std::time::Instant;

use mklight_core::{JobContext, samples};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, Session,
    analysis::{AnalysisOutput, AnalysisResult},
    backend::Backend,
    buffer::BufferData,
    project::Project,
};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;

fn bytes(d: &BufferData) -> &[u8] {
    fn raw<T>(v: &[T]) -> &[u8] {
        // SAFETY: plain-old-data element types, read-only view.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
    }
    match d {
        BufferData::F64(v) => raw(v),
        BufferData::F32(v) => raw(v),
        BufferData::I32(v) => raw(v),
        BufferData::U32(v) => raw(v),
        BufferData::I64(v) => raw(v),
        BufferData::U64(v) => raw(v),
        BufferData::U8(v) => raw(v),
    }
}

fn hashes(r: &AnalysisResult) -> Value {
    let buffers: Vec<Value> = r
        .buffers
        .iter()
        .map(|b| {
            let mut h = blake3::Hasher::new();
            h.update(bytes(&b.data));
            json!({"name":b.name,"dtype":b.dtype(),"shape":b.shape,
                "bytes":bytes(&b.data).len(),"hash":h.finalize().to_hex().to_string()})
        })
        .collect();
    json!({
        "json": blake3::hash(serde_json::to_string(&r.json).unwrap().as_bytes()).to_hex().to_string(),
        "warnings": blake3::hash(serde_json::to_string(&r.run_info.warnings).unwrap().as_bytes()).to_hex().to_string(),
        "buffers": buffers,
    })
}

fn default_cases(engine: &Engine, project: bool) -> Vec<Value> {
    let mut cases: Vec<Value> = engine
        .analyses()
        .into_iter()
        .filter(|a| a.available)
        .filter(|a| {
            !matches!(
                a.name.as_str(),
                "optimize" | "tolerance" | "glass_substitution" | "rng_stream"
            )
        })
        .map(|a| json!({"name":a.name,"analysis":a.name,"settings":null}))
        .collect();
    let dense = if project { 512 } else { 128 };
    cases.extend([
        json!({"name":"spot_128","analysis":"spot_diagram","settings":{"rings":128}}),
        json!({"name":"spot_dense","analysis":"spot_diagram","settings":{"rings":dense}}),
        json!({"name":"spot_dense_stats","analysis":"spot_diagram","settings":{"rings":dense,"include_points":false}}),
        json!({"name":"spot_centroid_tf","analysis":"spot_diagram","settings":{"rings":16,"reference":"centroid","defocus":0.05,"through_focus":{"min":-0.1,"max":0.1,"steps":5}}}),
        json!({"name":"ray_grid_dense","analysis":"ray_grid","settings":{"rings":64}}),
        json!({"name":"footprint_dense","analysis":"footprint","settings":{"rings":64}}),
        json!({"name":"layout_3d","analysis":"layout","settings":{"include_3d":true,"mesh_rings":8,"rays_per_field":21}}),
        json!({"name":"grid_distortion_dense","analysis":"grid_distortion","settings":{"grid_size":101}}),
    ]);
    cases
}

/// One `BUFFER_CONTRACT` line: a non-reference result against the CPU f64
/// result of the same case under the per-quantity contract (GPU §9.4), with
/// the native attempts a fallback hid. Hashes cannot compare builds whose
/// arithmetic legitimately changed; the contract can.
fn contract(label: &str, case: &Value, mode: &str, cpu: &AnalysisResult, other: &AnalysisResult) {
    let analysis = case["analysis"].as_str().unwrap();
    if !comparison::FIELDS
        .iter()
        .chain(comparison::SOLVER_SCORES)
        .any(|(a, _)| *a == analysis)
    {
        return; // CPU-only analysis: no native result to judge.
    }
    let solver = comparison::SOLVER_SCORES
        .iter()
        .any(|(a, _)| *a == analysis);
    let output = |r: &AnalysisResult| AnalysisOutput {
        json: if solver {
            comparison::solver_scores(analysis, &r.json)
        } else {
            r.json.clone()
        },
        buffers: if solver {
            Vec::new()
        } else {
            r.buffers.clone()
        },
        ..Default::default()
    };
    let c =
        comparison::compare(&output(cpu), &output(other), analysis, &JobContext::new()).unwrap();
    let line = json!({"design":label,"case":case["name"],"mode":mode,"used":other.backend.used,
        "compared":c.compared,"gate_failures":c.failures,"classes":c.classes,
        "native_failures":comparison::native_failures(&other.run_info),
        "sampling_differences":c.sampling_differences});
    println!("BUFFER_CONTRACT {line}");
}

fn run(
    session: &Session,
    label: &str,
    case: &Value,
    mode: &str,
    runs: usize,
) -> Option<AnalysisResult> {
    let mut last = None;
    let mut request =
        AnalysisRequest::new(case["analysis"].as_str().unwrap(), case["settings"].clone());
    request.backend = Some(match mode {
        "cpu" => Backend::CpuReference,
        "auto" => Backend::Auto,
        _ => Backend::Gpu("metal".into()),
    });
    request.allow_backend_fallback = true;
    for round in 0..runs.max(1) {
        let device = mklight_metal::device_seconds();
        let start = Instant::now();
        let result = session.run_analysis(&request, &JobContext::new());
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let device_ms = (mklight_metal::device_seconds() - device) * 1e3;
        let mut line = json!({"design":label,"case":case["name"],"mode":mode,"round":round,
            "wall_ms":wall_ms,"device_ms":device_ms});
        match result {
            Ok(r) => {
                line["used"] = json!(r.backend.used);
                line["fallback_reason"] = json!(r.backend.fallback_reason);
                // Every round: repeated requests may be answered by the
                // engine's analysis graph and must stay identical.
                line["identity"] = hashes(&r);
                last = Some(r);
            }
            Err(e) => line["error"] = json!(e.to_string()),
        }
        println!("BUFFER_IDENTITY {line}");
    }
    last
}

#[test]
#[ignore = "release identity/cost probe; compare two builds' output"]
fn buffer_identity_probe() {
    let mut config = EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    };
    if let Ok(b) = std::env::var("MKLIGHT_IDENTITY_GRAPH_BYTES") {
        config.analysis_graph_bytes = b.parse().unwrap();
    }
    let engine = Engine::new(config);
    let modes = std::env::var("MKLIGHT_IDENTITY_MODES").unwrap_or("cpu,metal".into());
    let runs = std::env::var("MKLIGHT_IDENTITY_RUNS").map_or(1, |s| s.parse().unwrap());
    let custom: Option<Vec<Value>> = std::env::var("MKLIGHT_IDENTITY_CASES")
        .ok()
        .map(|s| serde_json::from_str(&s).unwrap());
    let mut designs: Vec<(String, Session, bool)> = Vec::new();
    if std::env::var("MKLIGHT_IDENTITY_SAMPLES").as_deref() != Ok("0") {
        for (name, system) in samples::all() {
            designs.push((
                name.into(),
                engine.session_from_project(Project::new(system)).unwrap(),
                false,
            ));
        }
    }
    if let Ok(path) = std::env::var("MKLIGHT_IDENTITY_PROJECT") {
        designs.push((
            "project".into(),
            engine.open_session(Path::new(&path)).unwrap(),
            true,
        ));
    }
    for (label, session, project) in &designs {
        let cases = custom
            .clone()
            .unwrap_or_else(|| default_cases(&engine, *project));
        for case in &cases {
            let mut reference = None;
            for mode in modes.split(',') {
                let result = run(session, label, case, mode, runs);
                match (mode, &reference, &result) {
                    ("cpu", _, _) => reference = result,
                    (_, Some(cpu), Some(other)) => contract(label, case, mode, cpu, other),
                    _ => {}
                }
            }
        }
    }
    // Retained bytes stay within the configured graph budget.
    println!(
        "BUFFER_GRAPH {}",
        json!({"budget":engine.config().analysis_graph_bytes,"stats":engine.analysis_graph_stats()})
    );
}
