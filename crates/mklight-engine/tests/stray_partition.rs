//! The ghost / stray partition (CONVENTIONS §10): with `ghost_partition`,
//! every detected Monte-Carlo path is counted exactly once, by the
//! sequential solver (`Primary`), the ghost run (`Ghost` paths in its
//! coverage) or the stray total, whatever the ghost run's coverage is
//! (all enumerated paths, an explicit subset, one order).

use serde_json::{Value, json};

use mklight_core::{JobContext, samples};
use mklight_engine::analysis::schema::validate;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};

fn session() -> Session {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        ..Default::default()
    });
    e.session_from_project(Project::new(samples::cooke_triplet()))
        .unwrap()
}

fn run(s: &Session, name: &str, settings: Value) -> AnalysisResult {
    let r = s
        .run_analysis(&AnalysisRequest::new(name, settings), &JobContext::new())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let info = s
        .engine()
        .analyses()
        .into_iter()
        .find(|a| a.name == name)
        .unwrap();
    validate(&info.result_schema, &r.json, true).unwrap_or_else(|e| panic!("{name}: {e}"));
    r
}

fn class_flux(r: &AnalysisResult, class: &str) -> f64 {
    r.json["totals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["class"] == class)
        .unwrap()["flux"]
        .as_f64()
        .unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1e-300),
        "{what}: {a} vs {b}"
    );
}

/// `detected = Primary + covered Ghost + stray_flux`, and the stray total
/// is the mapped classes plus the uncovered ghosts.
fn check_books(r: &AnalysisResult) {
    let p = &r.json["partition"];
    let detected = f(&r.json["energy"]["detected"]);
    let primary = class_flux(r, "primary");
    close(
        primary + f(&p["covered_ghost_flux_mc"]) + f(&r.json["stray_flux"]),
        detected,
        "detected = primary + covered ghosts + stray",
    );
    close(
        f(&p["covered_ghost_flux_mc"]) + f(&p["uncovered_ghost_flux"]),
        f(&p["ghost_class_flux_mc"]),
        "ghost class = covered + uncovered",
    );
    close(
        f(&p["mapped_stray_flux"]) + f(&p["uncovered_ghost_flux"]),
        f(&r.json["stray_flux"]),
        "stray = mapped + uncovered",
    );
    let mapped: f64 = ["scatter", "mechanical", "sneak"]
        .iter()
        .map(|c| class_flux(r, c))
        .sum();
    close(mapped, f(&p["mapped_stray_flux"]), "mapped classes");
    assert!(
        f(&r.json["stray_std_error"])
            >= f(&p["mapped_stray_std_error"]) + f(&p["uncovered_ghost_std_error"]) - 1e-18
    );
}

#[test]
fn uncovered_ghost_paths_are_handed_back_to_the_stray_total() {
    let s = session();
    let base = json!({"rays": 2000, "seed": 1});
    let with = |gp: Value| {
        let mut v = base.clone();
        v["ghost_partition"] = gp;
        run(&s, "stray_light", v)
    };

    // The ghost run's paths (all 2-reflection paths).
    let ghost = run(&s, "ghost_paths", Value::Null);
    let paths: Vec<Value> = ghost.json["coverage"]["paths"]
        .as_array()
        .expect("coverage paths")
        .clone();
    assert!(paths.len() > 2, "the test needs several ghost paths");

    let full = with(json!({}));
    check_books(&full);
    assert_eq!(full.json["partition"]["ghost_paths"], json!(paths.len()));
    // Full 2-reflection coverage owns most of the ghost class.
    let fp = &full.json["partition"];
    assert!(f(&fp["covered_ghost_flux_mc"]) > 0.5 * f(&fp["ghost_class_flux_mc"]));

    // One explicit path: the other paths' flux moves to the stray total.
    let one = with(json!({"paths": [paths[0].clone()]}));
    check_books(&one);
    let op = &one.json["partition"];
    assert_eq!(op["ghost_paths"], json!(1));
    assert!(f(&op["covered_ghost_flux_mc"]) < f(&fp["covered_ghost_flux_mc"]));
    close(
        f(&one.json["stray_flux"]) - f(&full.json["stray_flux"]),
        f(&fp["covered_ghost_flux_mc"]) - f(&op["covered_ghost_flux_mc"]),
        "flux moved from the ghost run to the stray total",
    );
    assert!(f(&one.json["stray_flux"]) > 2.0 * f(&full.json["stray_flux"]));
    // The same random streams: the maps (mapped classes) do not change.
    assert_eq!(
        one.json["partition"]["mapped_stray_flux"],
        full.json["partition"]["mapped_stray_flux"]
    );
    assert!(
        one.run_info
            .warnings
            .iter()
            .any(|w| w.contains("outside the ghost run's coverage")),
        "{:?}",
        one.run_info.warnings
    );
    // Handed-back ghosts appear among the reported signatures.
    assert!(
        one.json["top_signatures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["class"] == "ghost")
    );

    // Empty coverage: the whole ghost class is the stray solver's, exactly
    // as a run without exclusion.
    let none = with(json!({"paths": []}));
    check_books(&none);
    let mut alone = base.clone();
    alone["classes"] = json!(30);
    let alone = run(&s, "stray_light", alone);
    close(
        f(&none.json["stray_flux"]),
        f(&alone.json["stray_flux"]),
        "empty coverage = no exclusion",
    );
}
