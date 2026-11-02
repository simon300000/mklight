//! Measurement-only access to raw native adapters, alongside normal engine gates.
//! Never registers experimental arithmetic in the engine's automatic provider.
#![cfg(target_os = "macos")]
use mklight_core::{
    JobContext, Snapshot,
    job::{Precision, RunTimer},
    materials::MaterialLibrary,
    samples,
};
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
use std::{collections::BTreeMap, sync::Arc, time::Instant};

#[derive(Default)]
struct Metric {
    maximum: f64,
    fraction: f64,
    compared: usize,
    mismatches: usize,
    budgeted: bool,
}
type Metrics = BTreeMap<String, Metric>;

// Whole §9.4 budgets, never the sampled one-third publication margin.
fn budget(path: &str, reference: f64, analysis: &str) -> Option<f64> {
    if analysis == "stray_light"
        || ["r50", "r80", "r90", "half_width80"]
            .iter()
            .any(|v| path.contains(v))
    {
        return None;
    }
    Some(if path.contains("direction") {
        2e-6
    } else if path.contains("opd")
        || path.contains("wavefront")
        || path.contains("zernike")
        || (analysis == "wavefront_map" && (path.contains("rms") || path.contains("pv")))
    {
        5e-3
    } else if path.contains("modulation")
        || path.contains("mtf")
        || path.contains("strehl")
        || path.contains("encircled")
        || path.contains("ensquared")
        || path.contains("tangential")
        || path.contains("sagittal")
        || (path.contains("fraction") && !path.contains("illumination"))
    {
        2e-3
    } else if path.contains("relative_illumination") {
        1e-3
    } else if path.contains("angle") || path.contains("incidence") {
        2e-4
    } else if path.contains("magnification") {
        (1e-3 * reference.abs()).max(1e-5)
    } else if path.contains("rms") || path.contains("geo_radius") {
        (0.005 * reference.abs()).max(2e-5)
    } else if path.contains("retardance") {
        1e-3
    } else if path.contains("transmission") || path.contains("weight") {
        1e-5 * reference.abs() + 1e-10
    } else if path.contains("centroid")
        || path.contains("position")
        || path.contains("focus")
        || path.contains("shift")
        || path.contains("coordinates")
        || path.contains("ray_grid")
        || path.contains("point_mm")
        || path.contains("reference_mm")
        || path.contains("center_mm")
    {
        2e-5
    } else {
        return None;
    })
}
fn number(m: &mut Metrics, path: &str, a: f64, b: f64, analysis: &str) {
    let metric = m.entry(path.into()).or_default();
    metric.compared += 1;
    metric.budgeted |= budget(path, a, analysis).is_some();
    if a.is_nan() && b.is_nan() || a == b {
        return;
    }
    if !a.is_finite() || !b.is_finite() {
        metric.mismatches += 1;
        return;
    }
    let error = (a - b).abs();
    metric.maximum = metric.maximum.max(error);
    if let Some(budget) = budget(path, a, analysis) {
        metric.budgeted = true;
        metric.fraction = metric.fraction.max(error / budget);
    }
}
fn compare_json(m: &mut Metrics, a: &Value, b: &Value, path: &str, analysis: &str) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in a {
                if [
                    "run_info",
                    "run",
                    "warnings",
                    "duration_ms",
                    "iterations",
                    "evaluations",
                ]
                .contains(&k.as_str())
                {
                    continue;
                }
                compare_json(
                    m,
                    v,
                    b.get(k).unwrap_or(&Value::Null),
                    &format!("{path}.{k}"),
                    analysis,
                );
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (a, b) in a.iter().zip(b) {
                compare_json(m, a, b, &format!("{path}[]"), analysis);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            number(m, path, a.as_f64().unwrap(), b.as_f64().unwrap(), analysis)
        }
        _ if a == b => (),
        _ => m.entry(path.into()).or_default().mismatches += 1,
    }
}
fn compare(a: &AnalysisOutput, b: &AnalysisOutput, analysis: &str) -> Value {
    let mut m = Metrics::new();
    compare_json(&mut m, &a.json, &b.json, "result", analysis);
    let radius = a
        .buffers
        .iter()
        .find(|v| v.name == "radius_mm" || v.name == "radius")
        .and_then(|v| {
            if let BufferData::F64(x) = &v.data {
                Some(x.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            a.json["radius"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        });
    if let Some(step) = radius
        .windows(2)
        .map(|v| (v[1] - v[0]).abs())
        .filter(|v| *v > 0.)
        .reduce(f64::min)
    {
        for (path, value) in &mut m {
            if ["r50", "r80", "r90", "half_width80"]
                .iter()
                .any(|v| path.contains(v))
            {
                value.fraction = value.maximum / step;
                value.budgeted = true;
            }
        }
    }
    if analysis == "stray_light" {
        let emitted = a.json["emitted"].as_f64().unwrap();
        let residual = b.json["energy"]["balance_residual"].as_f64().unwrap();
        m.insert(
            "result.energy.ledger_relative".into(),
            Metric {
                maximum: residual.abs() / emitted,
                fraction: residual.abs() / (emitted * 1e-5),
                budgeted: true,
                compared: 1,
                ..Default::default()
            },
        );
        for (a, b) in a.json["totals"]
            .as_array()
            .unwrap()
            .iter()
            .zip(b.json["totals"].as_array().unwrap())
        {
            let error = (a["flux"].as_f64().unwrap() - b["flux"].as_f64().unwrap()).abs();
            let limit = 5.
                * a["std_error"]
                    .as_f64()
                    .unwrap()
                    .hypot(b["std_error"].as_f64().unwrap());
            for (value, se) in [
                ("centroid", "centroid_std_error"),
                ("rms_radius", "rms_radius_std_error"),
            ] {
                let av = &a["moments"][value];
                let bv = &b["moments"][value];
                let ae = &a["moments"][se];
                let be = &b["moments"][se];
                let arrays = |v: &Value| v.as_array().cloned().unwrap_or_else(|| vec![v.clone()]);
                let mut metric = Metric {
                    budgeted: true,
                    ..Default::default()
                };
                for (((a, b), ea), eb) in arrays(av)
                    .iter()
                    .zip(arrays(bv))
                    .zip(arrays(ae))
                    .zip(arrays(be))
                {
                    if let (Some(a), Some(b), Some(ea), Some(eb)) =
                        (a.as_f64(), b.as_f64(), ea.as_f64(), eb.as_f64())
                    {
                        let error = (a - b).abs();
                        let limit = 5. * ea.hypot(eb);
                        metric.maximum = metric.maximum.max(error);
                        metric.compared += 1;
                        if limit > 0. {
                            metric.fraction = metric.fraction.max(error / limit);
                        } else if error > 0. {
                            metric.mismatches += 1;
                        }
                    }
                }
                m.insert(
                    format!("result.class_{}.{value}_5se", a["class"].as_str().unwrap()),
                    metric,
                );
            }
            let key = format!("result.class_{}.flux_5se", a["class"].as_str().unwrap());
            m.insert(
                key,
                Metric {
                    maximum: error,
                    fraction: if limit > 0. { error / limit } else { 0. },
                    mismatches: usize::from(limit == 0. && error > 0.),
                    budgeted: true,
                    compared: 1,
                },
            );
        }
    }
    let a_output = a;
    let b_output = b;
    let columns = a.json["columns"].as_array();
    for a in &a.buffers {
        let path = format!("buffer.{}", a.name);
        let Some(b) = b
            .buffers
            .iter()
            .find(|b| b.name == a.name && b.shape == a.shape)
        else {
            m.entry(path).or_default().mismatches += 1;
            continue;
        };
        let vectors = match (&a.data, &b.data) {
            (BufferData::F64(a), BufferData::F64(b)) => Some((a.clone(), b.clone())),
            (BufferData::F32(a), BufferData::F32(b)) => Some((
                a.iter().map(|x| *x as f64).collect(),
                b.iter().map(|x| *x as f64).collect(),
            )),
            _ => None,
        };
        if let Some((a, b)) = vectors {
            let image = (analysis == "psf" && path.contains("data_"))
                || path.contains("intensity")
                || path.contains("irradiance")
                || path == "buffer.image"
                || path.contains("line_spread")
                || (analysis == "line_spread"
                    && (path.starts_with("buffer.x_") || path.starts_with("buffer.y_")));
            if image && analysis == "stray_light" && !path.contains("std_error") {
                let error_name = format!("{}_std_error", path.trim_start_matches("buffer."));
                let find = |out: &AnalysisOutput| {
                    out.buffers
                        .iter()
                        .find(|v| v.name == error_name)
                        .and_then(|v| {
                            if let BufferData::F64(x) = &v.data {
                                Some(x.clone())
                            } else {
                                None
                            }
                        })
                };
                if let (Some(ea), Some(eb)) = (find(a_output), find(b_output)) {
                    let mut metric = Metric {
                        budgeted: true,
                        ..Default::default()
                    };
                    for ((&a, &b), (&ea, &eb)) in a.iter().zip(&b).zip(ea.iter().zip(eb.iter())) {
                        let error = (a - b).abs();
                        let limit = 5. * ea.hypot(eb);
                        metric.maximum = metric.maximum.max(error);
                        metric.compared += 1;
                        if limit > 0. {
                            metric.fraction = metric.fraction.max(error / limit);
                        } else if error > 0. {
                            metric.mismatches += 1;
                        }
                    }
                    m.insert(format!("{path}.pixel_5se"), metric);
                }
            } else if image && analysis != "stray_light" {
                let norm = a
                    .iter()
                    .filter(|x| x.is_finite())
                    .map(|x| x.abs())
                    .sum::<f64>();
                let mut delta = 0.;
                let mut energy = 0.;
                for (&a, &b) in a.iter().zip(&b) {
                    if a.is_finite() && b.is_finite() {
                        delta += (a - b).abs();
                        energy += a - b;
                    } else if a.is_nan() != b.is_nan() {
                        m.entry(path.clone()).or_default().mismatches += 1;
                    }
                }
                if norm > 0. {
                    for (suffix, value, limit) in [
                        ("l1_relative", delta / norm, 2e-3),
                        ("energy_relative", energy.abs() / norm, 1e-3),
                    ] {
                        m.insert(
                            format!("{path}.{suffix}"),
                            Metric {
                                maximum: value,
                                fraction: value / limit,
                                compared: a.len(),
                                budgeted: true,
                                ..Default::default()
                            },
                        );
                    }
                }
            } else {
                for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                    let key = if path == "buffer.values" {
                        columns
                            .and_then(|c| c[i % c.len()]["name"].as_str())
                            .map(|name| format!("{path}.{name}"))
                    } else {
                        None
                    };
                    number(&mut m, key.as_deref().unwrap_or(&path), a, b, analysis);
                }
            }
        } else if a.data != b.data {
            m.entry(path).or_default().mismatches += 1;
        }
    }
    json!(m.iter().map(|(k,v)|(k,json!({"max_error":v.maximum,"budget_fraction":if v.budgeted{Some(v.fraction)}else{None},"compared":v.compared,"mismatches":v.mismatches}))).collect::<BTreeMap<_,_>>())
}

#[test]
#[ignore = "real Metal; all-sample engine precision and provisional alternating timing study"]
fn engine_precision_study() {
    let filter = std::env::var("F32_STUDY_ANALYSES").ok();
    let sample_filter = std::env::var("F32_STUDY_SAMPLE").ok();
    let registry = AnalysisRegistry::with_builtin();
    let library = Arc::new(MaterialLibrary::with_builtin());
    let backends: Vec<_> = [MetalPrecision::DoubleFloat, MetalPrecision::PureF32]
        .into_iter()
        .map(|precision| {
            let options = MetalOptions {
                precision,
                library: LibraryChoice::Precompiled,
                ..MetalOptions::from_env().unwrap()
            };
            Arc::new(
                MetalBackend::with_options(&options)
                    .unwrap()
                    .expect("real Metal required"),
            )
        })
        .collect();
    for b in &backends {
        println!(
            "F32_ENGINE_ENV {}",
            json!({"device":b.info(),"precision":b.precision(),"rayon_threads":rayon::current_num_threads()})
        );
    }
    for (name, system) in samples::all() {
        if sample_filter.as_ref().is_some_and(|f| f != name) {
            continue;
        }
        let snapshot = Snapshot::new(system.clone()).unwrap();
        // Engines exercise the unchanged production qualification gates.
        let engines: Vec<_> = backends
            .iter()
            .map(|b| {
                let engine = Engine::new(EngineConfig {
                    default_backend: Backend::CpuReference,
                    ..Default::default()
                });
                let b = b.clone();
                let provider = GpuProvider::new("metal", "explicit precision study", move || {
                    Ok(b.clone() as Arc<dyn ComputeBackend>)
                });
                let info = provider.info();
                let device = provider.device().unwrap();
                engine.register_backend(Arc::new(provider));
                (
                    engine
                        .session_from_project(Project::new(system.clone()))
                        .unwrap(),
                    info,
                    device,
                )
            })
            .collect();
        for info in registry.list() {
            if !info.gpu_path
                || [
                    "optimize",
                    "merit_evaluation",
                    "tolerance",
                    "glass_substitution",
                ]
                .contains(&info.name.as_str())
            {
                continue;
            }
            if filter
                .as_ref()
                .is_some_and(|f| !f.split(',').any(|n| n == info.name))
            {
                continue;
            }
            let analysis = registry.get(&info.name).unwrap();
            // Freeze wave sampling so precision cannot choose a different
            // pupil or image array. Geometric jobs retain their defaults.
            let mut requested = Value::Null;
            if info.category == "wave" {
                let pupil = json!({"grid":64,"rays":65});
                requested = json!({"pupil":pupil});
                match info.name.as_str() {
                    "psf" => {
                        let t = mklight_analysis::tracer_for(system.clone()).unwrap();
                        let first = mklight_trace::paraxial::first_order(
                            t.system(),
                            t.system().primary_wavelength,
                        )
                        .unwrap();
                        let spacing = t
                            .system()
                            .wavelengths_um
                            .iter()
                            .copied()
                            .reduce(f64::min)
                            .unwrap()
                            * 1e-3
                            * first.working_f_number.abs()
                            / 2.5;
                        requested["grid_size"] = json!(256);
                        requested["spacing_mm"] = json!(spacing);
                    }
                    "diffraction_encircled_energy" | "line_spread" => {
                        requested["grid_size"] = json!(256)
                    }
                    "mtf" => requested["max_frequency"] = json!(50.),
                    "through_focus_mtf" => {
                        requested["frequencies"] = json!([10., 20., 40.]);
                        requested["focus_range_mm"] = json!(0.1);
                    }
                    "mtf_vs_field" => requested["frequencies"] = json!([10., 20., 40.]),
                    "image_simulation" => {
                        requested =
                            json!({"optics":{"pupil":pupil,"psf_size":128,"psf_spacing_mm":0.001}})
                    }
                    _ => (),
                }
            }
            let mut settings = prepare(analysis.as_ref(), &requested).unwrap();
            settings.as_object_mut().unwrap().remove("spectrum");
            let cpu = AnalysisInput::new(
                snapshot.clone(),
                library.clone(),
                BackendInfo::cpu_reference(),
            );
            let inputs: Vec<_> = engines
                .iter()
                .map(|(_, info, device)| {
                    AnalysisInput::new(snapshot.clone(), library.clone(), info.clone())
                        .with_device(Some(device.clone()))
                        .with_selection("gpu:metal", None)
                })
                .collect();
            for (index, (session, _, _)) in engines.iter().enumerate() {
                let mut request = AnalysisRequest::new(&info.name, settings.clone());
                request.backend = Some(Backend::Gpu("metal".into()));
                match session.run_analysis(&request, &JobContext::new()) {
                    Ok(r) => println!(
                        "F32_ENGINE_GATE {}",
                        json!({"sample":name,"analysis":info.name,"mode":index,"used":r.backend.used,"reason":r.backend.fallback_reason,"qualification":r.backend.qualification,"run_info":r.run_info})
                    ),
                    Err(e) => println!(
                        "F32_ENGINE_GATE {}",
                        json!({"sample":name,"analysis":info.name,"mode":index,"error":e.to_string()})
                    ),
                }
            }
            // Raw adapter calls are confined to this ignored measurement test.
            // Rejected f32 outputs remain observable and never become a public result.
            for round in 0..11 {
                let mut outputs = Vec::new();
                let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                for index in order {
                    let input = if index == 0 { &cpu } else { &inputs[index - 1] };
                    let timer = RunTimer::start();
                    let start = Instant::now();
                    let device_start = if index == 0 {
                        0.
                    } else {
                        backends[index - 1].context().device_seconds()
                    };
                    let result = analysis.run(input, &settings, &JobContext::new());
                    let seconds = start.elapsed().as_secs_f64();
                    let device_seconds = if index == 0 {
                        0.
                    } else {
                        backends[index - 1].context().device_seconds() - device_start
                    };
                    let run=timer.finish(&snapshot,&info.name,&json!({"settings":settings,"precision":if index==0{Value::Null}else{serde_json::to_value(backends[index-1].precision()).unwrap()}}),if index==0{"cpu-f64"}else{"metal:experimental"},if index==0{Precision::F64Reference}else{Precision::F32Gpu},analysis.seed(&settings),vec![]);
                    match result {
                        Ok(out) => {
                            println!(
                                "F32_ENGINE_TIME {}",
                                json!({"sample":name,"analysis":info.name,"mode":index,"round":round,"phase":if round==0{"warmup"}else{"measured"},"wall_seconds":seconds,"device_seconds":device_seconds,"run_info":run,"warnings":out.warnings})
                            );
                            outputs.push((index, out));
                        }
                        Err(e) => println!(
                            "F32_ENGINE_ERROR {}",
                            json!({"sample":name,"analysis":info.name,"mode":index,"round":round,"error":e.to_string(),"wall_seconds":seconds,"device_seconds":device_seconds})
                        ),
                    }
                }
                // Session wall time includes selection, cached qualification,
                // final packing and any explicitly reported CPU fallback.
                for index in order {
                    let session = &engines[if index == 2 { 1 } else { 0 }].0;
                    let mut request = AnalysisRequest::new(&info.name, settings.clone());
                    request.backend = Some(if index == 0 {
                        Backend::CpuReference
                    } else {
                        Backend::Gpu("metal".into())
                    });
                    let device_start = if index == 0 {
                        0.
                    } else {
                        backends[index - 1].context().device_seconds()
                    };
                    let start = Instant::now();
                    match session.run_analysis(&request, &JobContext::new()) {
                        Ok(r) => println!(
                            "F32_ENGINE_SESSION_TIME {}",
                            json!({"sample":name,"analysis":info.name,"mode":index,"round":round,"phase":if round==0{"warmup"}else{"measured"},"wall_seconds":start.elapsed().as_secs_f64(),"device_seconds":if index==0{0.}else{backends[index-1].context().device_seconds()-device_start},"used":r.backend.used,"reason":r.backend.fallback_reason,"run_info":r.run_info})
                        ),
                        Err(e) => println!(
                            "F32_ENGINE_SESSION_ERROR {}",
                            json!({"sample":name,"analysis":info.name,"mode":index,"round":round,"error":e.to_string()})
                        ),
                    }
                }
                if round == 1
                    && let Some((_, reference)) = outputs.iter().find(|(index, _)| *index == 0)
                {
                    for (index, out) in &outputs {
                        if *index == 0 {
                            continue;
                        }
                        println!(
                            "F32_ENGINE_ERROR_STUDY {}",
                            json!({"sample":name,"analysis":info.name,"mode":index,"quantities":compare(reference,out,&info.name)})
                        );
                    }
                }
            }
        }
    }
}
