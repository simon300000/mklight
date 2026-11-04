//! Offline qualification of the conditioned aim stop and the preview tier
//! (CONVENTIONS §7): every GPU-capable analysis (solvers excluded) on every
//! sample with real ray aiming, plus perturbed and adversarial variants,
//! compared with the one-third-margin gates against the CPU f64 reference at
//! the requested 1e-10 tolerance.
//!
//! cargo test --release -p mklight-engine --test aim_stop_qualify -- --ignored --nocapture
//! MKLIGHT_QUALIFY_METAL=1 also runs the conditioned stop on Metal;
//! MKLIGHT_QUALIFY_PREVIEW=1 records the preview tier's errors (not gated);
//! MKLIGHT_QUALIFY_FILTER=name,name limits the systems,
//! MKLIGHT_QUALIFY_ANALYSES=name,name the analyses.
use mklight_core::model::Placement;
use mklight_core::{
    ApertureSpec, Field, JobContext, OpticalSystem, RayAiming, SemiDiameter, SurfaceShape, samples,
};
use mklight_engine::{
    AnalysisRequest, Engine, EngineConfig, analysis::AnalysisOutput, backend::Backend,
    buffer::BufferData, project::Project,
};
use serde_json::{Value, json};

#[path = "support/comparison.rs"]
mod comparison;

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

/// Undefined axes are not compared (GPU §9.4): the orientation angle of a
/// Seidel term (tilt, astigmatism, coma) whose reference magnitude is below
/// 1e-3 waves (the Zernike coefficients themselves are compared with the
/// OPD budget), and the fast axis of a polarisation sample whose retarder
/// is undefined or whose retardance is below the 1e-3 rad retardance
/// budget (the Jones matrices themselves are compared). Both are set to
/// NaN on both sides.
fn exempt_undefined(a: &mut AnalysisOutput, b: &mut AnalysisOutput) {
    fn seidel(reference: &Value, other: &mut Value) {
        match (reference, other) {
            (Value::Object(r), Value::Object(o)) => {
                for (term, period) in [("tilt", 360.0), ("astigmatism", 180.0), ("coma", 360.0)] {
                    let key = format!("{term}_angle_deg");
                    if let (Some(mag), Some(ra), Some(angle)) = (
                        r.get(term).and_then(Value::as_f64),
                        r.get(&key).and_then(Value::as_f64),
                        o.get_mut(&key),
                    ) {
                        if mag.abs() < 1e-3 {
                            *angle = Value::from(f64::NAN);
                        } else if let Some(oa) = angle.as_f64() {
                            // Orientation angles are periodic (astigmatism
                            // modulo 180°): compare the wrapped difference.
                            let d = (oa - ra).rem_euclid(period);
                            let d = if d > 0.5 * period { d - period } else { d };
                            *angle = Value::from(ra + d);
                        }
                    }
                }
                for (k, v) in o.iter_mut() {
                    if let Some(rv) = r.get(k) {
                        seidel(rv, v);
                    }
                }
            }
            (Value::Array(r), Value::Array(o)) => {
                for (rv, ov) in r.iter().zip(o.iter_mut()) {
                    seidel(rv, ov);
                }
            }
            _ => {}
        }
    }
    let reference = a.json.clone();
    seidel(&reference, &mut b.json);
    seidel(&reference, &mut a.json);
    let defined: Option<Vec<bool>> = a
        .buffers
        .iter()
        .find(|b| b.name == "retarder_defined")
        .and_then(|b| match &b.data {
            BufferData::U8(v) => Some(v.iter().map(|d| *d != 0).collect()),
            _ => None,
        });
    let retardance: Option<Vec<f64>> =
        a.buffers
            .iter()
            .find(|b| b.name == "retardance")
            .and_then(|b| match &b.data {
                BufferData::F64(v) => Some(v.clone()),
                _ => None,
            });
    if let (Some(defined), Some(retardance)) = (defined, retardance) {
        for out in [&mut *a, &mut *b] {
            if let Some(fa) = out.buffers.iter_mut().find(|b| b.name == "fast_axis")
                && let BufferData::F64(v) = &mut fa.data
            {
                for (i, axis) in v.chunks_mut(3).enumerate() {
                    if !defined[i] || retardance[i] < 1e-3 {
                        axis.fill(f64::NAN);
                    }
                }
            }
        }
    }
}

fn compare(
    a: &mklight_engine::AnalysisResult,
    b: &mklight_engine::AnalysisResult,
    name: &str,
) -> comparison::Comparison {
    let mut a = AnalysisOutput {
        json: a.json.clone(),
        buffers: a.buffers.clone(),
        ..Default::default()
    };
    let mut b = AnalysisOutput {
        json: b.json.clone(),
        buffers: b.buffers.clone(),
        ..Default::default()
    };
    exempt_undefined(&mut a, &mut b);
    comparison::compare(&a, &b, name, &JobContext::new()).unwrap()
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
        assert!(
            engine
                .backends()
                .iter()
                .any(|b| b.id == "gpu:metal" && b.available),
            "real Metal required"
        );
    }
    let mut systems: Vec<(String, OpticalSystem)> = samples::all()
        .into_iter()
        .map(|(n, s)| (n.to_string(), s))
        .collect();
    systems.extend(variants());
    let analyses: Vec<_> = engine
        .analyses()
        .into_iter()
        .filter(|a| {
            (only.is_empty() || only.contains(&a.name))
                && a.gpu_path
                && !matches!(
                    a.name.as_str(),
                    "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
                )
        })
        .collect();
    let mut failures = Vec::new();
    let mut rows = 0;
    for (name, system) in systems {
        if !filter.is_empty() && !filter.contains(&name) {
            continue;
        }
        let session = engine
            .session_from_project(Project::new(real(system)))
            .unwrap();
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
            let mut control: Option<String> = None;
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
                    control = report.reason.clone();
                }
                // Gated at the design tier; a Metal failure that the control
                // (requested stop) also shows is pre-existing, not the policy's.
                let pre_existing = label == "metal_conditioned"
                    && report.reason.is_some()
                    && control.as_deref().map(|c| c.split(':').next())
                        == report.reason.as_deref().map(|r| r.split(':').next());
                let gated = tier == Design && stop == Conditioned && !pre_existing;
                let failed = gated && report.reason.is_some();
                rows += 1;
                println!(
                    "AIM_STOP_QUALIFY {}",
                    json!({"system":name,"analysis":info.name,"case":label,"backend":run.backend.used,
                    "compared":report.compared,"maximum_absolute_error":report.maximum,"maximum_error_path":report.maximum_path,
                    "image_l1_relative":report.image_l1_relative,"energy_relative":report.energy_relative,
                    "gate_failure":report.reason,"failed":failed,"pre_existing":pre_existing,"reference_ms":reference.run_info.duration_ms,"ms":wall,
                    "settings":run.run_info.settings})
                );
                if failed {
                    failures.push(format!(
                        "{name}/{}/{label}: {}",
                        info.name,
                        report.reason.unwrap()
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
        failures.is_empty(),
        "{} gate failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
