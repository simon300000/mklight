//! Route options (`mklight_core::split`) are process-wide and decide how a
//! device analysis is divided, so a stored result answers a repeated
//! request only under the selection it was computed with.
//!
//! One test: the selection is process state.
use mklight_core::{JobContext, samples, split};
use mklight_engine::backend::Backend;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig};
use serde_json::Value;

#[test]
fn a_stored_result_belongs_to_its_route_options() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    let session = engine
        .session_from_project(Project::new(samples::cooke_triplet()))
        .unwrap();
    let run = || {
        session
            .run_analysis(
                &AnalysisRequest::new("chief_ray_angle", Value::Null),
                &JobContext::new(),
            )
            .unwrap();
        let stats = engine.analysis_graph_stats().unwrap();
        let results = &stats.kinds["result"];
        (results.builds, results.hits)
    };
    assert_eq!(run(), (1, 0));
    assert_eq!(
        run(),
        (1, 1),
        "the repeated request is answered from the graph"
    );
    // The default design by name: nothing runs differently, but the
    // selection is no longer the one the stored result was computed under.
    split::set("fieldmaps", Some("host-chief"));
    assert_eq!(run(), (2, 1), "another selection computes its own result");
    assert_eq!(run(), (2, 2));
    split::reset();
    assert_eq!(run(), (2, 3), "the first result is still stored");
}
