//! Engine analyses on Metal double-float versus pure f32 (docs/GPU.md §3.6),
//! against CPU f64. Measurement only, never registered for engine Auto:
//! `cargo test --release -p mklight-engine --test f32_study -- --ignored --nocapture`
//! (filters: `F32_STUDY_SAMPLE=name`, `F32_STUDY_ANALYSES=a,b`).
#![cfg(target_os = "macos")]
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use mklight_core::{JobContext, Snapshot, materials::MaterialLibrary, samples};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig,
    analysis::{AnalysisInput, AnalysisOutput, AnalysisRegistry, prepare},
    backend::{Backend, BackendInfo, BackendProvider},
    buffer::BufferData,
    gpu::GpuProvider,
    project::Project,
};
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions, MetalPrecision};
use serde_json::{Value, json};

const MODES: [&str; 3] = ["cpu_f64", "df", "f32"];

/// Maximum |a − b| per numeric JSON leaf (array indices collapsed) and per
/// numeric buffer, plus `<buffer>.l1` = Σ|a − b| / Σ|a|. Infinity (JSON null)
/// marks a structural or finiteness mismatch.
fn errors(a: &AnalysisOutput, b: &AnalysisOutput) -> BTreeMap<String, f64> {
    fn put(m: &mut BTreeMap<String, f64>, path: String, e: f64) {
        let v = m.entry(path).or_default();
        *v = v.max(e);
    }
    fn walk(m: &mut BTreeMap<String, f64>, a: &Value, b: &Value, path: String) {
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for (k, v) in a {
                    if !["run_info", "run", "warnings", "duration_ms"].contains(&k.as_str()) {
                        walk(
                            m,
                            v,
                            b.get(k).unwrap_or(&Value::Null),
                            format!("{path}.{k}"),
                        );
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
                for (a, b) in a.iter().zip(b) {
                    walk(m, a, b, format!("{path}[]"));
                }
            }
            (Value::Number(a), Value::Number(b)) => {
                put(m, path, (a.as_f64().unwrap() - b.as_f64().unwrap()).abs())
            }
            _ if a == b => {}
            _ => put(m, path, f64::INFINITY),
        }
    }
    let mut m = BTreeMap::new();
    walk(&mut m, &a.json, &b.json, "result".into());
    let floats = |d: &BufferData| match d {
        BufferData::F64(x) => Some(x.clone()),
        BufferData::F32(x) => Some(x.iter().map(|v| *v as f64).collect::<Vec<_>>()),
        _ => None,
    };
    for x in &a.buffers {
        let path = format!("buffer.{}", x.name);
        let y = b
            .buffers
            .iter()
            .find(|y| y.name == x.name && y.shape == x.shape);
        let (Some(xs), Some(ys)) = (floats(&x.data), y.and_then(|y| floats(&y.data))) else {
            if y.is_none_or(|y| y.data != x.data) {
                put(&mut m, path, f64::INFINITY);
            }
            continue;
        };
        let (mut max, mut l1, mut norm) = (0f64, 0., 0.);
        for (p, q) in xs.iter().zip(&ys) {
            if p.is_finite() && q.is_finite() {
                max = max.max((p - q).abs());
                l1 += (p - q).abs();
                norm += p.abs();
            } else if p.is_nan() != q.is_nan() || (!p.is_nan() && p != q) {
                max = f64::INFINITY;
            }
        }
        put(
            &mut m,
            format!("{path}.l1"),
            if norm > 0. { l1 / norm } else { 0. },
        );
        put(&mut m, path, max);
    }
    m
}

/// Wave sampling frozen so that precision cannot change the pupil or image
/// arrays; geometric analyses keep their defaults.
fn frozen_settings(name: &str, system: &mklight_core::OpticalSystem) -> Value {
    let pupil = json!({"grid":64,"rays":65});
    match name {
        "psf" => {
            let t = mklight_analysis::tracer_for(system.clone()).unwrap();
            let s = t.system();
            let first = mklight_trace::paraxial::first_order(s, s.primary_wavelength).unwrap();
            let shortest = s.wavelengths_um.iter().copied().reduce(f64::min).unwrap();
            let spacing = shortest * 1e-3 * first.working_f_number.abs() / 2.5;
            json!({"pupil":pupil,"grid_size":256,"spacing_mm":spacing})
        }
        "diffraction_encircled_energy" | "line_spread" => json!({"pupil":pupil,"grid_size":256}),
        "mtf" => json!({"pupil":pupil,"max_frequency":50.}),
        "through_focus_mtf" => {
            json!({"pupil":pupil,"frequencies":[10.,20.,40.],"focus_range_mm":0.1})
        }
        "mtf_vs_field" => json!({"pupil":pupil,"frequencies":[10.,20.,40.]}),
        "image_simulation" => json!({"optics":{"pupil":{"grid":64,"rays":65,
            "amplitude":"radiometric"},"psf_size":129,"psf_spacing_mm":0.001}}),
        _ => json!({"pupil":pupil}),
    }
}

/// Per sample and GPU analysis: the engine's route of an explicit Metal
/// request for each precision (`F32_ROUTE`), then the raw analysis adapters
/// (CPU f64, DF, f32) in one warm-up and ten alternating rounds: median
/// wall/device ms and the errors of the first measured DF/f32 outputs
/// against CPU f64.
#[test]
#[ignore = "real Metal: engine precision measurement"]
fn engine_precision_study() {
    let sample_filter = std::env::var("F32_STUDY_SAMPLE").ok();
    let analysis_filter = std::env::var("F32_STUDY_ANALYSES").ok();
    let registry = AnalysisRegistry::with_builtin();
    let library = Arc::new(MaterialLibrary::with_builtin());
    let backends = [MetalPrecision::DoubleFloat, MetalPrecision::PureF32].map(|precision| {
        let options = MetalOptions {
            precision,
            library: LibraryChoice::Precompiled,
        };
        Arc::new(
            MetalBackend::with_options(&options)
                .unwrap()
                .expect("real Metal required"),
        )
    });
    for (name, system) in samples::all() {
        if sample_filter.as_ref().is_some_and(|f| f != name) {
            continue;
        }
        let snapshot = Snapshot::new(system.clone()).unwrap();
        // One engine per precision, for the route it gives an explicit Metal
        // request.
        let engines = backends.each_ref().map(|b| {
            let engine = Engine::new(EngineConfig {
                default_backend: Backend::CpuReference,
                ..Default::default()
            });
            let b = b.clone();
            let provider = GpuProvider::new("metal", "precision study", move || {
                Ok(b.clone() as Arc<dyn ComputeBackend>)
            });
            let input = AnalysisInput::new(snapshot.clone(), library.clone(), provider.info())
                .with_device(Some(provider.device().unwrap()))
                .with_selection("gpu:metal", None);
            engine.register_backend(Arc::new(provider));
            let session = engine
                .session_from_project(Project::new(system.clone()))
                .unwrap();
            (session, input)
        });
        let cpu = AnalysisInput::new(
            snapshot.clone(),
            library.clone(),
            BackendInfo::cpu_reference(),
        );
        let excluded = [
            "optimize",
            "merit_evaluation",
            "tolerance",
            "glass_substitution",
        ];
        for info in registry.list() {
            if !info.gpu_path
                || excluded.contains(&info.name.as_str())
                || analysis_filter
                    .as_ref()
                    .is_some_and(|f| !f.split(',').any(|n| n == info.name))
            {
                continue;
            }
            let analysis = registry.get(&info.name).unwrap();
            let requested = if info.category == "wave" {
                frozen_settings(&info.name, &system)
            } else {
                Value::Null
            };
            let mut settings = prepare(analysis.as_ref(), &requested).unwrap();
            settings.as_object_mut().unwrap().remove("spectrum");
            for ((session, _), mode) in engines.iter().zip(&MODES[1..]) {
                let mut request = AnalysisRequest::new(&info.name, settings.clone());
                request.backend = Some(Backend::Gpu("metal".into()));
                let verdict = match session.run_analysis(&request, &JobContext::new()) {
                    Ok(r) => json!({"used":r.backend.used,"reason":r.backend.fallback_reason}),
                    Err(e) => json!({"error":e.to_string()}),
                };
                println!(
                    "F32_ROUTE {}",
                    json!({"sample":name,"analysis":info.name,"mode":mode,"verdict":verdict})
                );
            }
            let mut wall = [vec![], vec![], vec![]];
            let mut device = [vec![], vec![], vec![]];
            let mut first: [Option<AnalysisOutput>; 3] = Default::default();
            for round in 0..11 {
                let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                for mode in order {
                    let input = if mode == 0 {
                        &cpu
                    } else {
                        &engines[mode - 1].1
                    };
                    let (start, device_start) = (Instant::now(), mklight_metal::device_seconds());
                    let result = analysis.run(input, &settings, &JobContext::new());
                    let elapsed = start.elapsed().as_secs_f64();
                    let gpu_seconds = mklight_metal::device_seconds() - device_start;
                    match result {
                        Ok(out) if round > 0 => {
                            wall[mode].push(elapsed * 1e3);
                            device[mode].push(gpu_seconds * 1e3);
                            first[mode].get_or_insert(out);
                        }
                        Ok(_) => {}
                        Err(e) if round == 0 => println!(
                            "F32_ERROR {}",
                            json!({"sample":name,"analysis":info.name,"mode":MODES[mode],
                                "error":e.to_string()})
                        ),
                        Err(_) => {}
                    }
                }
            }
            let median = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied()
            };
            for mode in 0..3 {
                let accuracy = match (&first[0], &first[mode]) {
                    (Some(r), Some(o)) if mode > 0 => {
                        json!({"errors":errors(r, o),"warnings":o.warnings})
                    }
                    _ => Value::Null,
                };
                println!(
                    "F32_ANALYSIS {}",
                    json!({"sample":name,"analysis":info.name,"mode":MODES[mode],
                        "wall_ms":median(&mut wall[mode]),"device_ms":median(&mut device[mode]),
                        "accuracy":accuracy})
                );
            }
        }
    }
}
