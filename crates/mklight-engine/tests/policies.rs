//! Tier and aim-stop policies (CONVENTIONS §7, GPU §9.4): every result is
//! labelled, the preview tier only changes settings the request leaves out,
//! an explicitly requested tolerance is honoured, and the policy is part of
//! the result identity.

use mklight_core::{AimStopKind, JobContext, RayAiming, Tier, samples};
use mklight_engine::backend::Backend;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use serde_json::{Value, json};

fn engine(aim_stop: AimStopKind) -> Engine {
    Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: Backend::CpuReference,
        aim_stop,
        ..Default::default()
    })
}

fn session(e: &Engine) -> Session {
    let mut sys = samples::double_gauss();
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    e.session_from_project(Project::new(sys)).unwrap()
}

fn run(s: &Session, req: AnalysisRequest) -> AnalysisResult {
    s.run_analysis(&req, &JobContext::new()).unwrap()
}

#[test]
fn results_are_labelled_with_tier_and_aim_stop() {
    let e = engine(AimStopKind::Requested);
    let s = session(&e);
    let r = run(&s, AnalysisRequest::new("spot_diagram", Value::Null));
    assert_eq!(r.run_info.tier, Tier::Design);
    assert_eq!(r.run_info.aim_stop, AimStopKind::Requested);
    let env = r.envelope();
    assert_eq!(env["run_info"]["tier"], json!("design"));
    assert_eq!(env["run_info"]["aim_stop"], json!("requested"));

    let mut req = AnalysisRequest::new("spot_diagram", Value::Null);
    req.aim_stop = Some(AimStopKind::Conditioned);
    let c = run(&s, req);
    assert_eq!(c.run_info.aim_stop, AimStopKind::Conditioned);
    assert_eq!(c.run_info.tier, Tier::Design);
    // Same sampling, results within the one-third margin of the position
    // budget (the derived stop allocates 1/30 of it).
    assert_eq!(c.run_info.settings, r.run_info.settings);
    let rms =
        |r: &AnalysisResult| r.json["fields"][0]["polychromatic"]["rms_radius_centroid"].as_f64();
    if let (Some(a), Some(b)) = (rms(&r), rms(&c)) {
        assert!((a - b).abs() < 2e-5 / 3.0, "{a} vs {b}");
    }

    // The engine default applies when the request names no policy.
    let e2 = engine(AimStopKind::Conditioned);
    let s2 = session(&e2);
    let d = run(&s2, AnalysisRequest::new("spot_diagram", Value::Null));
    assert_eq!(d.run_info.aim_stop, AimStopKind::Conditioned);
    let mut req = AnalysisRequest::new("spot_diagram", Value::Null);
    req.aim_stop = Some(AimStopKind::Requested);
    assert_eq!(run(&s2, req).run_info.aim_stop, AimStopKind::Requested);
}

#[test]
fn preview_tier_takes_preview_defaults_only_for_omitted_settings() {
    let e = engine(AimStopKind::Requested);
    let s = session(&e);
    let mut req = AnalysisRequest::new("field_curvature_distortion", Value::Null);
    req.tier = Tier::Preview;
    let p = run(&s, req.clone());
    assert_eq!(p.run_info.tier, Tier::Preview);
    assert_eq!(p.run_info.settings["field_samples"], json!(11));
    let d = run(
        &s,
        AnalysisRequest::new("field_curvature_distortion", Value::Null),
    );
    assert_eq!(d.run_info.settings["field_samples"], json!(21));
    // A given value is never changed by the tier.
    req.settings = json!({"field_samples": 15});
    let given = run(&s, req);
    assert_eq!(given.run_info.settings["field_samples"], json!(15));
    assert_eq!(given.run_info.tier, Tier::Preview);

    // Wave: the preview lattice is half the rays per axis; an explicit
    // pupil object is kept whole.
    let mut req = AnalysisRequest::new("mtf", Value::Null);
    req.tier = Tier::Preview;
    let p = run(&s, req.clone());
    assert_eq!(p.run_info.settings["pupil"]["rays"], json!(33));
    req.settings = json!({"pupil": {"rays": 49}});
    let given = run(&s, req);
    assert_eq!(given.run_info.settings["pupil"]["rays"], json!(49));
    let d = run(&s, AnalysisRequest::new("mtf", Value::Null));
    assert_eq!(d.run_info.settings["pupil"]["rays"], json!(65));
}

#[test]
fn policy_is_part_of_the_result_identity() {
    // The graph memoises identical requests; a different policy is a
    // different result, never a restamped one.
    let e = engine(AimStopKind::Requested);
    let s = session(&e);
    let a = run(&s, AnalysisRequest::new("mtf", Value::Null));
    let mut req = AnalysisRequest::new("mtf", Value::Null);
    req.tier = Tier::Preview;
    let b = run(&s, req);
    assert_ne!(a.run_info.settings, b.run_info.settings);
    assert_eq!(b.run_info.tier, Tier::Preview);
    let mut req = AnalysisRequest::new("mtf", Value::Null);
    req.aim_stop = Some(AimStopKind::Conditioned);
    let c = run(&s, req);
    assert_eq!(c.run_info.aim_stop, AimStopKind::Conditioned);
    assert_eq!(c.run_info.settings, a.run_info.settings);
}
