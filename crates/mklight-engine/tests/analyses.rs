//! Golden tests of the registered analyses: every analysis runs through the
//! engine on sample systems, its result validates (strictly) against its
//! published result schema, its buffers are declared, its default settings
//! are the solver crate's defaults, and its key numbers equal the solver
//! crate called directly.

use serde_json::{Value, json};
use std::sync::Arc;

use mklight_analysis as ma;
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, OpticalSystem, ResolvedSystem, Snapshot, samples};
use mklight_engine::analysis::schema::{check_buffers, validate};
use mklight_engine::buffer::BufferData;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use mklight_trace::SequentialTracer;

fn engine() -> Engine {
    Engine::new(EngineConfig {
        worker_threads: 2,
        ..Default::default()
    })
}

fn session(e: &Engine, sys: OpticalSystem) -> Session {
    e.session_from_project(Project::new(sys)).unwrap()
}

fn run(s: &Session, name: &str, settings: Value) -> AnalysisResult {
    s.run_analysis(&AnalysisRequest::new(name, settings), &JobContext::new())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn tracer(sys: OpticalSystem) -> SequentialTracer {
    ma::tracer_for(sys).unwrap()
}

fn resolved(sys: OpticalSystem) -> Arc<ResolvedSystem> {
    let snap = Snapshot::new(sys).unwrap();
    Arc::new(ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap())
}

/// A solver result as JSON without its `run_info` / `run`.
fn direct<R: serde::Serialize>(r: &R) -> Value {
    let mut v = serde_json::to_value(r).unwrap();
    if let Value::Object(m) = &mut v {
        m.remove("run_info");
        m.remove("run");
    }
    v
}

fn f64s(r: &AnalysisResult, name: &str) -> Vec<f64> {
    let b = r
        .buffers
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no buffer {name}"));
    match &b.data {
        BufferData::F64(v) => v.clone(),
        BufferData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        _ => panic!("{name} is not a float buffer"),
    }
}

/// Validate result + buffers against the analysis' published schemas.
fn check_schema(e: &Engine, r: &AnalysisResult) {
    let name = &r.run_info.analysis;
    let a = e.analysis(name).unwrap();
    let schema = a.result_schema();
    if let Err(m) = validate(&schema, &r.json, true) {
        panic!("{name}: result does not match its schema: {m}");
    }
    let descs: Vec<(String, String, usize)> = r
        .buffers
        .iter()
        .map(|b| {
            let dt = serde_json::to_value(b.dtype()).unwrap();
            (b.name.clone(), dt.as_str().unwrap().to_string(), b.shape.len())
        })
        .collect();
    if let Err(m) = check_buffers(&schema, &descs) {
        panic!("{name}: {m}");
    }
    assert_eq!(r.result_version, a.result_version());
    let env = r.envelope();
    assert_eq!(env["backend"]["used"], r.run_info.backend.as_str());
    assert_eq!(env["result_version"], json!(r.result_version));
}

/// Every available analysis, with its defaults, on the Cooke triplet:
/// schema-valid results, declared buffers, recorded settings = defaults.
#[test]
fn every_analysis_matches_its_schemas() {
    let e = engine();
    let s = session(&e, samples::cooke_triplet());
    let mut ran = 0;
    for info in e.analyses() {
        // Settings schemas are themselves valid for their defaults.
        if !info.available {
            assert!(info.unavailable_reason.is_some());
            continue;
        }
        assert!(
            info.result_schema["x-buffers"].is_array(),
            "{}: result schema without x-buffers",
            info.name
        );
        let settings = match info.name.as_str() {
            "glass_info" => json!({"name": "N-BK7", "curve_samples": 5}),
            "rng_stream" => json!({"count": 100}),
            // Keep the debug-build test fast.
            "stray_light" => json!({"rays": 300}),
            "ghost_irradiance" => json!({"pupil_samples": 8, "map_bins": [8, 8]}),
            _ => {
                assert!(
                    !info.default_settings.is_null(),
                    "{} has no complete defaults",
                    info.name
                );
                Value::Null
            }
        };
        let r = run(&s, &info.name, settings.clone());
        check_schema(&e, &r);
        if settings.is_null() {
            assert_eq!(r.run_info.settings, info.default_settings, "{}", info.name);
        }
        ran += 1;
    }
    assert!(ran >= 25, "only {ran} analyses ran");
}

/// The geometric analyses on more systems (mirrors, finite conjugates,
/// aspheres): schema-valid everywhere.
#[test]
fn geometric_analyses_on_every_sample() {
    let e = engine();
    let names = [
        "first_order",
        "layout",
        "spot_diagram",
        "ray_fan",
        "opd_fan",
        "field_curvature_distortion",
        "seidel",
        "longitudinal_aberration",
        "chromatic_focal_shift",
        "lateral_colour",
        "encircled_energy",
        "geometric_mtf",
        "ray_grid",
        "footprint",
        "system_transmission",
        "validate",
    ];
    for (sample, sys) in samples::all() {
        let s = session(&e, sys);
        for n in names {
            let r = s
                .run_analysis(&AnalysisRequest::new(n, Value::Null), &JobContext::new())
                .unwrap_or_else(|err| panic!("{sample}/{n}: {err}"));
            check_schema(&e, &r);
        }
        let v = run(&s, "validate", Value::Null);
        assert_eq!(v.json["ok"], true, "{sample}: {}", v.json);
    }
}

#[test]
fn geometric_numbers_equal_the_crate() {
    let e = engine();
    for sys in [samples::cooke_triplet(), samples::newtonian_telescope()] {
        let s = session(&e, sys.clone());
        let t = tracer(sys);

        let r = run(&s, "first_order", Value::Null);
        let d = ma::system_data::system_data(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));

        let r = run(&s, "ray_fan", Value::Null);
        assert_eq!(r.json, direct(&ma::fans::ray_fan(&t, &Default::default()).unwrap()));
        let r = run(&s, "opd_fan", json!({"samples": 11}));
        let st = ma::fans::FanSettings {
            samples: 11,
            ..Default::default()
        };
        assert_eq!(r.json, direct(&ma::fans::opd_fan(&t, &st).unwrap()));

        let r = run(&s, "field_curvature_distortion", Value::Null);
        let d = ma::curvature::field_curvature(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));

        let r = run(&s, "seidel", Value::Null);
        let d = ma::seidel::seidel(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));
        let r = run(&s, "longitudinal_aberration", Value::Null);
        let d = ma::seidel::longitudinal_aberration(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));
        let r = run(&s, "chromatic_focal_shift", Value::Null);
        let d = ma::seidel::chromatic_focal_shift(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));
        let r = run(&s, "lateral_colour", Value::Null);
        let d = ma::seidel::lateral_colour(&t, &Default::default()).unwrap();
        assert_eq!(r.json, direct(&d));
        let r = run(&s, "relative_illumination", json!({"field_samples": 3}));
        let st = ma::illumination::RelativeIlluminationSettings {
            field_samples: 3,
            ..Default::default()
        };
        let d = ma::illumination::relative_illumination(&t, &st).unwrap();
        assert_eq!(r.json, direct(&d));

        // Spot diagram: statistics equal, the points moved into buffers.
        let r = run(&s, "spot_diagram", Value::Null);
        let d = ma::spot::spot_diagram(&t, &Default::default()).unwrap();
        let mut dj = direct(&d);
        let pts = f64s(&r, "points");
        let mut k = 0;
        for (fi, f) in dj["fields"].as_array_mut().unwrap().iter_mut().enumerate() {
            let p = f.as_object_mut().unwrap().remove("points").unwrap();
            let p = p.as_array().unwrap();
            assert_eq!(r.json["fields"][fi]["point_range"], json!([k, p.len()]));
            for q in p {
                assert_eq!(pts[2 * k], q["x"].as_f64().unwrap());
                assert_eq!(pts[2 * k + 1], q["y"].as_f64().unwrap());
                k += 1;
            }
        }
        assert_eq!(pts.len(), 2 * k);
        let mut ej = r.json.clone();
        for f in ej["fields"].as_array_mut().unwrap() {
            f.as_object_mut().unwrap().remove("point_range");
        }
        assert_eq!(ej, dj);

        let r = run(&s, "encircled_energy", json!({"rings": 8}));
        let st: ma::energy::EncircledEnergySettings =
            serde_json::from_value(json!({"rings": 8})).unwrap();
        let d = ma::energy::encircled_energy(&t, &st).unwrap();
        assert_eq!(r.json, direct(&d));
        let r = run(&s, "geometric_mtf", json!({"rings": 8}));
        let st: ma::energy::GeometricMtfSettings =
            serde_json::from_value(json!({"rings": 8})).unwrap();
        assert_eq!(r.json, direct(&ma::energy::geometric_mtf(&t, &st).unwrap()));
        let r = run(&s, "best_focus", json!({"rings": 4}));
        let mut st = ma::energy::BestFocusSettings::default();
        st.sampling.rings = 4;
        assert_eq!(r.json, direct(&ma::energy::best_focus(&t, &st).unwrap()));

        // Layout: the buffers are the crate's compact buffers.
        let r = run(&s, "layout", Value::Null);
        let scene = ma::layout::layout(&t, &Default::default()).unwrap();
        let b = scene.to_buffers();
        assert_eq!(f64s(&r, "vertices_f64"), b.vertices_f64);
        assert_eq!(r.json["polyline_count"], json!(b.polyline_count()));
        assert_eq!(r.json["origin"], json!(b.origin));

        // Grid distortion and footprints: points in buffers.
        let r = run(&s, "grid_distortion", json!({"grid_size": 5}));
        let st = ma::curvature::GridDistortionSettings {
            grid_size: 5,
            ..Default::default()
        };
        let d = ma::curvature::grid_distortion(&t, &st).unwrap();
        let real = f64s(&r, "real");
        for (i, p) in d.points.iter().enumerate() {
            assert!(real[2 * i] == p.real[0] || (real[2 * i].is_nan() && p.real[0].is_nan()));
        }
        assert_eq!(
            r.json["max_abs_distortion_percent"],
            direct(&d)["max_abs_distortion_percent"]
        );
        let r = run(&s, "footprint", json!({"surface": 2}));
        let st = ma::illumination::FootprintSettings {
            surface: 2,
            ..Default::default()
        };
        let d = ma::illumination::footprint(&t, &st).unwrap();
        let all: Vec<f64> = d
            .rays
            .iter()
            .flat_map(|r| r.points.iter().flat_map(|p| [p[0], p[1]]))
            .collect();
        assert_eq!(f64s(&r, "points"), all);
        assert_eq!(r.json["max_radius"], direct(&d)["max_radius"]);
    }
}

#[test]
fn ray_grid_matches_the_reference_trace() {
    let e = engine();
    let sys = samples::double_gauss();
    let s = session(&e, sys.clone());
    let r = run(&s, "ray_grid", json!({"rings": 3}));
    let t = tracer(sys);
    let pupil = ma::util::pupil_points(ma::util::PupilPattern::Hexapolar, 3, 1);
    let nw = t.num_wavelengths();
    let pts = f64s(&r, "points");
    let mut arrived = 0;
    for f in 0..t.fields().len() {
        for w in 0..nw {
            for (p, pp) in pupil.iter().enumerate() {
                let tr = t.trace(t.field(f), w, pp[0], pp[1], &t.options());
                let i = (f * nw + w) * pupil.len() + p;
                if tr.status.is_ok() {
                    arrived += 1;
                    assert_eq!(pts[2 * i], tr.ray.pos.x);
                    assert_eq!(pts[2 * i + 1], tr.ray.pos.y);
                } else {
                    assert!(pts[2 * i].is_nan());
                }
            }
        }
    }
    assert_eq!(r.json["arrived"], json!(arrived));
    assert_eq!(r.run_info.backend, "cpu_reference");
}

#[test]
fn ghost_stray_radiometry_and_solves_equal_the_crates() {
    let e = engine();
    let sys = samples::cooke_triplet();
    let s = session(&e, sys.clone());
    let rs = resolved(sys.clone());

    // Ghost ranking.
    let r = run(&s, "ghost_paths", Value::Null);
    let model = mklight_ghost::GhostModel::from_system(rs.clone(), Default::default()).unwrap();
    let d = mklight_ghost::paraxial_ghosts(&model, &Default::default()).unwrap();
    assert_eq!(r.json, direct(&d));

    // Ghost irradiance: maps in buffers.
    let set = json!({"pupil_samples": 12, "map_bins": [16, 12], "keep_path_maps": true});
    let r = run(&s, "ghost_irradiance", set);
    let is = mklight_ghost::IrradianceSettings {
        pupil_samples: 12,
        map_bins: [16, 12],
        keep_path_maps: true,
        ..Default::default()
    };
    let d = mklight_ghost::ghost_irradiance(&model, &is, &JobContext::new()).unwrap();
    assert_eq!(f64s(&r, "total_ghost_map"), d.total_ghost_map.values);
    assert_eq!(r.json["total_flux_ratio"], json!(d.total_flux_ratio));
    let pm = r.buffers.iter().find(|b| b.name == "path_maps").unwrap();
    assert_eq!(pm.shape, vec![d.ghosts.len(), 12, 16]);
    assert_eq!(
        f64s(&r, "path_maps")[..16 * 12],
        d.ghosts[0].map.as_ref().unwrap().values[..]
    );
    let ghost_apertures = r.json["coverage"]["apertures"].clone();

    // Stray light alone.
    let r = run(&s, "stray_light", json!({"rays": 400, "seed": 3}));
    let ss = mklight_stray::analysis::StrayAnalysisSettings {
        rays: 400,
        seed: 3,
        ..Default::default()
    };
    let d = mklight_stray::analysis::analyze(rs.clone(), &ss, &JobContext::new()).unwrap();
    assert_eq!(r.json["stray_flux"], json!(d.stray_flux));
    assert_eq!(f64s(&r, "total_irradiance"), d.total_irradiance);
    assert_eq!(r.json["partition"], Value::Null);
    assert_eq!(r.run_info.seed, Some(3));

    // Partitioned against the ghost solver: ghost class excluded, the same
    // clipping apertures as the ghost run.
    let r = run(
        &s,
        "stray_light",
        json!({"rays": 400, "seed": 3, "ghost_partition": {}}),
    );
    let p = &r.json["partition"];
    assert_eq!(p["same_apertures"], true);
    assert_eq!(p["apertures"], ghost_apertures);
    let ghost_total = r.json["totals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["class"] == "ghost")
        .unwrap();
    assert_eq!(ghost_total["included"], false);
    assert_eq!(r.json["classes"], json!(30 & !2));
    // Unsupported ghost clipping is refused rather than mis-partitioned.
    let bad = s.run_analysis(
        &AnalysisRequest::new(
            "stray_light",
            json!({"rays": 10, "ghost_partition": {"clip": {"mode": "none"}}}),
        ),
        &JobContext::new(),
    );
    assert!(bad.is_err());

    // Radiometry.
    let r = run(&s, "system_transmission", Value::Null);
    let t = SequentialTracer::new(rs.clone()).unwrap();
    let d = mklight_trace::system_transmission(&t, &Default::default(), &JobContext::new())
        .unwrap();
    assert_eq!(r.json, direct(&d));
    let r = run(&s, "polarization_pupil_map", json!({"grid": 9, "field": 1}));
    let ps = mklight_trace::PolarizationMapSettings {
        field: mklight_trace::FieldPoint::system(&sys, 1),
        wavelength: sys.wavelengths.primary,
        grid: 9,
        state: mklight_trace::PolarizationState::Unpolarized,
        clip: true,
    };
    let d = mklight_trace::polarization_pupil_map(&t, &ps, &JobContext::new()).unwrap();
    assert_eq!(r.json["mean_intensity"], json!(d.mean_intensity));
    assert_eq!(r.json["sample_count"], json!(d.samples.len()));
    let inten: Vec<f64> = d.samples.iter().map(|x| x.intensity).collect();
    assert_eq!(f64s(&r, "intensity"), inten);
    // Unpolarised intensity is the mean of t_max and t_min (sanity).
    let (tmax, tmin, i) = (f64s(&r, "t_max"), f64s(&r, "t_min"), f64s(&r, "intensity"));
    for k in 0..i.len() {
        assert!((i[k] - 0.5 * (tmax[k] + tmin[k])).abs() < 1e-12);
    }

    // Solves dry run.
    let r = run(&s, "solves", Value::Null);
    let (_, d) = mklight_trace::apply_solves(&sys, &MaterialLibrary::with_builtin()).unwrap();
    assert_eq!(r.json, direct(&d));
}
