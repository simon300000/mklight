//! Offline qualification of the conditioned aim stop and the preview tier
//! (CONVENTIONS §7): every GPU-capable analysis (solvers excluded) on every
//! sample with real ray aiming, plus perturbed and adversarial variants,
//! compared under the per-quantity contract (`mklight_bench::comparison`, GPU
//! §9.4: one-third margins of the §7 budgets, survival counts and masks exact)
//! against the CPU f64 reference at the requested 1e-10 tolerance.
//!
//! cargo test --release -p mklight-bench --test aim_stop_qualify -- --ignored --nocapture
//! MKLIGHT_QUALIFY_METAL=1 also runs the conditioned stop on Metal;
//! MKLIGHT_QUALIFY_PREVIEW=1 records the preview tier's errors (not gated);
//! MKLIGHT_QUALIFY_FILTER=name,name limits the systems,
//! MKLIGHT_QUALIFY_ANALYSES=name,name the analyses,
//! MKLIGHT_QUALIFY_PROJECT=/path/project.mklight adds a local project (as
//! saved, with its catalogues; nothing is embedded) under the name `project`.
//! `../README.md` lists every harness with its command and variables.
use mklight_bench::{comparison, solver};
use mklight_core::model::Placement;
use mklight_core::{
    ApertureSpec, Field, JobContext, OpticalSystem, RayAiming, SemiDiameter, SurfaceShape, samples,
};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, analysis::AnalysisOutput, backend::Backend,
    buffer::BufferData, project::Project,
};
use serde_json::{Value, json};

fn real(mut sys: OpticalSystem) -> OpticalSystem {
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    sys
}

/// Perturbed and adversarial variants of the samples.
fn variants() -> Vec<(String, OpticalSystem)> {
    let mut out = Vec::new();
    // Decentred and tilted double Gauss (element 2 and the rear group).
    let mut dg = samples::double_gauss();
    dg.surfaces[3].placement = Some(Placement {
        decenter_x: 0.05,
        decenter_y: -0.03,
        tilt_x: 0.3,
        tilt_y: -0.2,
        ..Default::default()
    });
    if dg.surfaces.len() > 9 {
        dg.surfaces[9].placement = Some(Placement {
            decenter_x: -0.02,
            decenter_y: 0.04,
            tilt_x: -0.15,
            tilt_y: 0.1,
            ..Default::default()
        });
    }
    out.push(("double_gauss_decentred_tilted".into(), dg));
    // Near-vignetting: tight clear apertures on the first and last elements
    // (A2's rim case), and a field beyond the designed one.
    let mut nv = samples::double_gauss();
    let n = nv.surfaces.len();
    nv.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 11.0 };
    nv.surfaces[n - 2].semi_diameter = SemiDiameter::Fixed { value: 11.0 };
    let m = nv.fields.max_radial();
    nv.fields.fields.push(Field::new(0.0, 1.15 * m));
    out.push(("double_gauss_near_vignetting".into(), nv));
    // Strongly aspheric: the Petzval asphere with amplified departures.
    let mut pa = samples::petzval_asphere();
    for s in &mut pa.surfaces {
        if let SurfaceShape::EvenAsphere { coefficients, .. } = &mut s.shape {
            for c in coefficients.iter_mut() {
                *c *= 4.0;
            }
        }
    }
    out.push(("petzval_strong_asphere".into(), pa));
    // High NA: the Petzval asphere at F/1.4 instead of F/4.
    let mut hn = samples::petzval_asphere();
    hn.aperture = ApertureSpec::ImageSpaceFNumber { value: 1.4 };
    out.push(("petzval_high_na".into(), hn));
    out
}

/// Preview-tier metrics against the design tier (GPU §9.4 preview
/// contract): MTF curves at the same frequencies, PSF Strehl and centroid,
/// wavefront RMS, spot centroid. Empty where the analysis has none.
fn preview_metrics(
    a: &mklight_engine::AnalysisResult,
    b: &mklight_engine::AnalysisResult,
) -> Value {
    let mut m = serde_json::Map::new();
    let f = |v: &Value| v.as_f64();
    let mut worst = |key: &str, x: Option<f64>| {
        if let Some(x) = x {
            let e = m.entry(key.to_string()).or_insert(Value::from(0.0));
            if x > e.as_f64().unwrap_or(0.0) {
                *e = Value::from(x);
            }
        }
    };
    // Curves in buffers of equal name and shape (MTF tangential/sagittal, EE).
    for (x, y) in a.buffers.iter().zip(&b.buffers) {
        if x.name == y.name
            && x.shape == y.shape
            && let (BufferData::F64(p), BufferData::F64(q)) = (x.data(), y.data())
        {
            let kind = if x.name.starts_with("tangential") || x.name.starts_with("sagittal") {
                "mtf_abs"
            } else if x.name.starts_with("encircled") || x.name.starts_with("ensquared") {
                "energy_fraction_abs"
            } else {
                continue;
            };
            let d = p
                .iter()
                .zip(q)
                .filter(|(u, v)| u.is_finite() && v.is_finite())
                .map(|(u, v)| (u - v).abs())
                .fold(0.0, f64::max);
            worst(kind, Some(d));
        }
    }
    let arrays = |v: &Value, key: &str| v[key].as_array().cloned().unwrap_or_default();
    for (x, y) in arrays(&a.json, "psfs").iter().zip(arrays(&b.json, "psfs")) {
        worst(
            "strehl_abs",
            f(&x["strehl"])
                .zip(f(&y["strehl"]))
                .map(|(u, v)| (u - v).abs()),
        );
        let c = |v: &Value| Some([f(&v["centroid_mm"][0])?, f(&v["centroid_mm"][1])?]);
        worst(
            "centroid_mm",
            c(x).zip(c(&y))
                .map(|(u, v)| (u[0] - v[0]).hypot(u[1] - v[1])),
        );
    }
    for (x, y) in arrays(&a.json, "maps").iter().zip(arrays(&b.json, "maps")) {
        worst(
            "rms_waves_abs",
            f(&x["rms_waves"])
                .zip(f(&y["rms_waves"]))
                .map(|(u, v)| (u - v).abs()),
        );
    }
    for (x, y) in arrays(&a.json, "fields")
        .iter()
        .zip(arrays(&b.json, "fields"))
    {
        for (p, q) in arrays(x, "per_wavelength")
            .iter()
            .zip(arrays(&y, "per_wavelength"))
        {
            let c = |v: &Value| {
                Some([
                    f(&v["stats"]["centroid"][0])?,
                    f(&v["stats"]["centroid"][1])?,
                ])
            };
            worst(
                "centroid_mm",
                c(p).zip(c(&q))
                    .map(|(u, v)| (u[0] - v[0]).hypot(u[1] - v[1])),
            );
        }
        for (p, q) in arrays(x, "tangential").iter().zip(arrays(&y, "tangential")) {
            let d = arrays(p, "values")
                .iter()
                .zip(arrays(&q, "values"))
                .filter_map(|(u, v)| Some((f(u)? - f(&v)?).abs()))
                .fold(0.0, f64::max);
            worst("fan_mm", Some(d));
        }
    }
    Value::Object(m)
}

fn compare(
    a: &mklight_engine::AnalysisResult,
    b: &mklight_engine::AnalysisResult,
    name: &str,
) -> comparison::Comparison {
    let a = AnalysisOutput {
        json: a.json.clone(),
        buffers: a.buffers.clone(),
        ..Default::default()
    };
    let b = AnalysisOutput {
        json: b.json.clone(),
        buffers: b.buffers.clone(),
        ..Default::default()
    };
    comparison::compare(&a, &b, name, &JobContext::new()).unwrap()
}

/// The path of a gate failure line.
fn failure_path(f: &str) -> &str {
    f.split(": ").next().unwrap_or(f)
}

#[test]
#[ignore = "offline qualification of the aim-stop and preview policies"]
fn aim_stop_and_preview_qualification() {
    let metal = std::env::var("MKLIGHT_QUALIFY_METAL").is_ok();
    let preview = std::env::var("MKLIGHT_QUALIFY_PREVIEW").is_ok();
    let filter: Vec<String> = std::env::var("MKLIGHT_QUALIFY_FILTER")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let only: Vec<String> = std::env::var("MKLIGHT_QUALIFY_ANALYSES")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        analysis_graph_bytes: 0,
        ..Default::default()
    });
    if metal {
        mklight_bench::require_metal(&engine);
    }
    let mut systems: Vec<(String, OpticalSystem)> = samples::all()
        .into_iter()
        .map(|(n, s)| (n.to_string(), real(s)))
        .collect();
    systems.extend(variants().into_iter().map(|(n, s)| (n, real(s))));
    let project = std::env::var("MKLIGHT_QUALIFY_PROJECT").ok();
    if project.is_some() {
        systems.push(("project".into(), samples::singlet()));
    }
    let analyses: Vec<_> = engine
        .analyses()
        .into_iter()
        .filter(|a| {
            (only.is_empty() || only.contains(&a.name))
                && a.gpu_path
                && !solver::JOBS.contains(&a.name.as_str())
        })
        .collect();
    let mut failures = Vec::new();
    let mut unmapped = Vec::new();
    let mut rows = 0;
    for (name, system) in systems {
        if !filter.is_empty() && !filter.contains(&name) {
            continue;
        }
        let session = match (&project, name.as_str()) {
            (Some(path), "project") => engine.open_session(std::path::Path::new(path)).unwrap(),
            _ => engine.session_from_project(Project::new(system)).unwrap(),
        };
        for info in &analyses {
            // Fixed sampling axes (same physical frequencies, image grids
            // and focus ranges at both tiers), so a preview difference is an
            // error and not an axis shift; design-tier defaults otherwise.
            let settings = if preview {
                match info.name.as_str() {
                    "mtf" => json!({"max_frequency": 60.0, "frequency_samples": 13}),
                    "mtf_vs_field" | "through_focus_mtf" => json!({"frequencies": [10.0, 30.0]}),
                    "psf" | "line_spread" | "diffraction_encircled_energy" => {
                        json!({"grid_size": 128})
                    }
                    _ => Value::Null,
                }
            } else {
                Value::Null
            };
            let mut request = AnalysisRequest::new(&info.name, settings);
            request.backend = Some(Backend::CpuReference);
            request.aim_stop = Some(mklight_core::AimStopKind::Requested);
            let reference = match session.run_analysis(&request, &JobContext::new()) {
                Ok(r) => r,
                Err(e) => {
                    println!(
                        "AIM_STOP_QUALIFY {}",
                        json!({"system":name,"analysis":info.name,"skipped":e.to_string()})
                    );
                    continue;
                }
            };
            assert_eq!(
                reference.run_info.aim_stop,
                mklight_core::AimStopKind::Requested
            );
            use mklight_core::{AimStopKind::*, Tier::*};
            let mut cases = vec![(
                "cpu_conditioned",
                Backend::CpuReference,
                Design,
                Conditioned,
            )];
            if metal {
                // The control tells pre-existing native differences from policy ones.
                cases.push((
                    "metal_requested",
                    Backend::Gpu("metal".into()),
                    Design,
                    Requested,
                ));
                cases.push((
                    "metal_conditioned",
                    Backend::Gpu("metal".into()),
                    Design,
                    Conditioned,
                ));
            }
            if preview {
                cases.push(("cpu_preview", Backend::CpuReference, Preview, Conditioned));
            }
            let mut control: Vec<String> = Vec::new();
            for (label, backend, tier, stop) in cases {
                request.backend = Some(backend);
                request.allow_backend_fallback = true;
                request.aim_stop = Some(stop);
                request.tier = tier;
                let started = std::time::Instant::now();
                let run = session.run_analysis(&request, &JobContext::new()).unwrap();
                let wall = started.elapsed().as_secs_f64() * 1e3;
                assert_eq!(run.run_info.aim_stop, stop);
                assert_eq!(run.run_info.tier, tier);
                let report = compare(&reference, &run, &info.name);
                if label == "metal_requested" {
                    control = report.failures.clone();
                }
                unmapped.extend(
                    report
                        .failures
                        .iter()
                        .filter(|f| f.contains("unmapped"))
                        .cloned(),
                );
                // Every design-tier row is gated, the requested-stop control
                // included. A Metal failure that the control shows at the
                // same path is labelled pre-existing (not the policy's), but
                // it still counts, and so does a native attempt that a
                // reference fallback completed.
                let pre_existing = label == "metal_conditioned"
                    && !report.failures.is_empty()
                    && report
                        .failures
                        .iter()
                        .all(|f| control.iter().any(|c| failure_path(c) == failure_path(f)));
                let native_failures = comparison::native_failures(&run.run_info);
                let gated = tier == Design;
                let failed = gated && (report.reason.is_some() || !native_failures.is_empty());
                rows += 1;
                println!(
                    "AIM_STOP_QUALIFY {}",
                    json!({"system":name,"analysis":info.name,"case":label,"backend":run.backend.used,
                    "compared":report.compared,"maximum_absolute_error":report.maximum,"maximum_error_path":report.maximum_path,
                    "image_l1_relative":report.image_l1_relative,"energy_relative":report.energy_relative,
                    "gate_failure":report.reason,"gate_failures":report.failures,
                    "classes":report.classes,"resampled":report.resampled,"undefined":report.undefined,
                    "sampling_differences":report.sampling_differences,"failed":failed,"pre_existing":pre_existing,
                    "native_failures":native_failures,"reference_ms":reference.run_info.duration_ms,"ms":wall,
                    "preview_metrics":(tier == Preview).then(|| preview_metrics(&reference, &run)),
                    "settings":run.run_info.settings})
                );
                if failed {
                    failures.push(format!(
                        "{name}/{}/{label}: {:?} {:?}",
                        info.name, report.reason, native_failures
                    ));
                }
            }
        }
    }
    println!(
        "AIM_STOP_QUALIFY_SUMMARY {}",
        json!({"rows":rows,"failures":failures})
    );
    assert!(
        unmapped.is_empty(),
        "unmapped result fields:\n{}",
        unmapped.join("\n")
    );
    assert!(
        failures.is_empty(),
        "{} gate failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
