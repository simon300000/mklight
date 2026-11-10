//! Route options (`mklight_engine::route`) decide how a device analysis is
//! divided. A request's own options replace the engine's and reach the
//! solver that divides the work, whose evidence names the division; a
//! stored result answers a repeated request only under the options it was
//! computed with; options a solver could not run fail when the request is
//! submitted.
use std::sync::Arc;

use mklight_core::{JobContext, samples};
use mklight_engine::backend::Backend;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::route::{Aiming, FieldMaps, RouteOptions, SolverAiming, StraySag};
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, EngineError, Session};
use serde_json::{Value, json};

fn engine(route: RouteOptions) -> Engine {
    let e = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        route,
        ..Default::default()
    });
    e.register_backend(Arc::new(GpuProvider::emulation()));
    e
}

fn session(engine: &Engine) -> Session {
    engine
        .session_from_project(Project::new(samples::cooke_triplet()))
        .unwrap()
}

fn request(analysis: &str, settings: Value, route: Option<RouteOptions>) -> AnalysisRequest {
    AnalysisRequest {
        route,
        ..AnalysisRequest::new(analysis, settings)
    }
}

/// Every value stored under `key` anywhere in `v`.
fn values<'a>(v: &'a Value, key: &str, out: &mut Vec<&'a Value>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                if k == key {
                    out.push(x);
                }
                values(x, key, out);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| values(x, key, out)),
        _ => {}
    }
}

/// The values of `key` in the device evidence of a result.
fn evidence(r: &AnalysisResult, key: &str) -> Vec<Value> {
    let execution = Value::Array(r.backend.execution.clone());
    let mut out = Vec::new();
    values(&execution, key, &mut out);
    out.into_iter().cloned().collect()
}

#[test]
fn a_stored_result_belongs_to_its_route_options() {
    let engine = engine(RouteOptions::default());
    let session = session(&engine);
    let run = |route: Option<RouteOptions>| {
        session
            .run_analysis(
                &request("chief_ray_angle", Value::Null, route),
                &JobContext::new(),
            )
            .unwrap();
        let stats = engine.analysis_graph_stats().unwrap();
        let results = &stats.kinds["result"];
        (results.builds, results.hits)
    };
    assert_eq!(run(None), (1, 0));
    assert_eq!(
        run(None),
        (1, 1),
        "the repeated request is answered from the graph"
    );
    assert_eq!(
        run(Some(RouteOptions::default())),
        (1, 2),
        "the engine's options spelled out are the same options"
    );
    // Every option is part of the key.
    let others = [
        RouteOptions {
            aiming: Aiming::Device,
            ..Default::default()
        },
        RouteOptions {
            aiming_share: Some(0.5),
            ..Default::default()
        },
        RouteOptions {
            fieldmaps: FieldMaps::DeviceChief,
            ..Default::default()
        },
        RouteOptions {
            stray_share: Some(1.0),
            ..Default::default()
        },
        RouteOptions {
            stray_sag: StraySag::F32Grazing,
            ..Default::default()
        },
        RouteOptions {
            solver_aiming: SolverAiming::Host,
            ..Default::default()
        },
    ];
    for (k, other) in others.into_iter().enumerate() {
        let k = k as u64;
        assert_eq!(
            run(Some(other)),
            (2 + k, 2 + k),
            "other options compute their own result: {other:?}"
        );
        assert_eq!(run(Some(other)), (2 + k, 3 + k), "{other:?}");
    }
    assert_eq!(run(None), (7, 9), "the first result is still stored");
}

#[test]
fn request_options_replace_the_engines_and_reach_the_solvers() {
    let device_chief = RouteOptions {
        fieldmaps: FieldMaps::DeviceChief,
        ..Default::default()
    };
    let engine = engine(device_chief);
    let session = session(&engine);
    let run = |analysis: &str, settings: Value, route: Option<RouteOptions>| {
        let mut req = request(analysis, settings, route);
        req.backend = Some(Backend::Gpu("cpu_f32".into()));
        session.run_analysis(&req, &JobContext::new()).unwrap()
    };
    // The chief-ray maps: the engine's default, then a request's own.
    let map = |route| run("chief_ray_angle", json!({"grid": [5, 5]}), route);
    assert_eq!(
        evidence(&map(None), "fieldmaps_variant"),
        [json!("device-chief")]
    );
    assert_eq!(
        evidence(&map(Some(RouteOptions::default())), "fieldmaps_variant"),
        [json!("host-chief")]
    );
    // Pupil aiming of the geometric analyses, and the device share of
    // large generations.
    let spot = |route| run("spot_diagram", json!({"rings": 30}), Some(route));
    let aiming = |r: &AnalysisResult| {
        let all = evidence(r, "aiming");
        let first = all.first().expect("aiming evidence");
        (first["design"].clone(), first["device_share"].clone())
    };
    assert_eq!(
        aiming(&spot(RouteOptions::default())),
        (json!("host-table"), json!(0.35))
    );
    assert_eq!(
        aiming(&spot(RouteOptions {
            aiming_share: Some(0.75),
            ..Default::default()
        })),
        (json!("host-table"), json!(0.75))
    );
    assert_eq!(
        aiming(&spot(RouteOptions {
            aiming: Aiming::Table,
            ..Default::default()
        })),
        (json!("table"), json!(0.0))
    );
    // The stray transport: the scene's rule gives the device none of so
    // few sources; a share of 1 gives it all of them, in either sag mode.
    let stray = |route| run("stray_light", json!({"rays": 2000}), Some(route));
    let by_rule = stray(RouteOptions::default());
    assert_eq!(by_rule.run_info.backend, "cpu_reference");
    // The share that ran and the share asked for (none).
    assert_eq!(evidence(&by_rule, "gpu_share"), [json!(0.0), Value::Null]);
    let all_device = stray(RouteOptions {
        stray_share: Some(1.0),
        stray_sag: StraySag::F32Grazing,
        ..Default::default()
    });
    assert_eq!(all_device.run_info.backend, "gpu:cpu_f32");
    assert_eq!(evidence(&all_device, "gpu_share"), [json!(1.0), json!(1.0)]);
    assert_eq!(evidence(&all_device, "sag_mode"), [json!(1)]);
    assert_eq!(evidence(&all_device, "sag"), [json!("f32-grazing")]);
    // Pupil aiming of the solver jobs.
    for (solver_aiming, name) in [
        (SolverAiming::ByBatch, "by-batch"),
        (SolverAiming::Host, "host"),
        (SolverAiming::Device, "device"),
    ] {
        let r = run(
            "merit_evaluation",
            Value::Null,
            Some(RouteOptions {
                solver_aiming,
                // The analyses' aiming does not aim the solvers.
                aiming: Aiming::Device,
                ..Default::default()
            }),
        );
        assert_eq!(r.json["backend"]["aiming"], json!(name));
    }
}

#[test]
fn invalid_options_fail_the_request() {
    let engine = engine(RouteOptions::default());
    let session = session(&engine);
    for (key, share) in [("aiming_share", 1.5), ("stray_share", -0.25)] {
        let route: RouteOptions = serde_json::from_value(json!({key: share})).unwrap();
        let req = request("spot_diagram", Value::Null, Some(route));
        for e in [
            session.submit_analysis(&req).unwrap_err(),
            session.run_analysis(&req, &JobContext::new()).unwrap_err(),
        ] {
            assert!(
                matches!(&e, EngineError::InvalidSettings(m) if m.contains(key)),
                "{key}: {e}"
            );
        }
    }
    // The engine's own options are checked with each request.
    let invalid = self::engine(RouteOptions {
        stray_share: Some(2.0),
        ..Default::default()
    });
    let s = self::session(&invalid);
    let req = request("spot_diagram", Value::Null, None);
    assert!(matches!(
        s.submit_analysis(&req).unwrap_err(),
        EngineError::InvalidSettings(_)
    ));
    let valid = request("spot_diagram", Value::Null, Some(RouteOptions::default()));
    s.run_analysis(&valid, &JobContext::new()).unwrap();
    // A request and a configuration as front ends send them (JSON): an
    // unknown key or name does not parse.
    let req: AnalysisRequest = serde_json::from_value(json!({
        "analysis": "spot_diagram",
        "route": {"aiming": "table", "aiming_share": null, "solver_aiming": "device"}}))
    .unwrap();
    assert_eq!(
        req.route,
        Some(RouteOptions {
            aiming: Aiming::Table,
            solver_aiming: SolverAiming::Device,
            ..Default::default()
        })
    );
    let config: EngineConfig =
        serde_json::from_value(json!({"route": {"stray_sag": "f32-grazing"}})).unwrap();
    assert_eq!(config.route.stray_sag, StraySag::F32Grazing);
    for bad in [
        json!({"analysis": "spot_diagram", "route": {"stray": "wavefront"}}),
        json!({"analysis": "spot_diagram", "route": {"aiming": "split:0.5"}}),
    ] {
        assert!(
            serde_json::from_value::<AnalysisRequest>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}
