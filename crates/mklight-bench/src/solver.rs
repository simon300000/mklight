//! The solver jobs' variables and settings, shared so that the calibration,
//! the project measurements and the accuracy evidence run the same solves.

use mklight_core::{MaterialRef, OpticalSystem, Surface, param::ParamPath};
use mklight_engine::Session;
use serde_json::{Value, json};

/// The solver jobs, in the order [`jobs`] returns them.
pub const JOBS: [&str; 4] = [
    "merit_evaluation",
    "optimize",
    "tolerance",
    "glass_substitution",
];

/// The surfaces between the object and the image.
fn inner(system: &OpticalSystem) -> &[Surface] {
    &system.surfaces[1..system.surfaces.len() - 1]
}

/// Solver variables (`variables` of a job's settings): the curvature of
/// every curved surface, unbounded.
pub fn curvatures(system: &OpticalSystem) -> Vec<Value> {
    inner(system)
        .iter()
        .filter(|s| s.shape.curvature() != 0.)
        .map(|s| json!({"param": ParamPath::Curvature { surface: s.id }}))
        .collect()
}

/// Solver variables (`variables` of a job's settings): the curvature of
/// every curved surface and every positive finite thickness within ±20 %,
/// without the parameters a solve controls.
pub fn variables(system: &OpticalSystem) -> Vec<Value> {
    let mut vars = Vec::new();
    for s in inner(system) {
        if s.shape.curvature() != 0. {
            vars.push(json!({"param":ParamPath::Curvature{surface:s.id}}));
        }
        if s.thickness.is_finite() && s.thickness > 0. {
            vars.push(json!({"param":ParamPath::Thickness{surface:s.id},
                "lower":s.thickness * 0.8,"upper":s.thickness * 1.2}));
        }
    }
    // Solves control their targets before merit generation.
    vars.retain(|v| !system.solves.iter().any(|s| json!(s.target) == v["param"]));
    vars
}

/// The first surface whose material is a catalogue glass.
pub fn first_glass(system: &OpticalSystem) -> Option<&Surface> {
    system.surfaces[1..]
        .iter()
        .find(|s| matches!(s.material, MaterialRef::Catalog { .. }))
}

/// Material variables (`material_variables` of `glass_substitution`):
/// [`first_glass`] with its three nearest catalogue glasses, or none.
pub fn glass(system: &OpticalSystem) -> Vec<Value> {
    first_glass(system)
        .map(|g| vec![json!({"surface":g.id,"candidates":{"nearest":3}})])
        .unwrap_or_default()
}

/// `optimize` truncated after `iterations` (no convergence tolerances).
pub fn optimize(variables: &[Value], iterations: u64) -> Value {
    json!({"variables":variables,"max_iterations":iterations,
        "options":{"ftol":0.,"xtol":0.,"gtol":0.},"seed":71})
}

/// `tolerance` with 16 Monte Carlo trials, all recorded.
pub fn tolerance() -> Value {
    json!({"trials":16,"seed":71,"record_trials":16})
}

/// `glass_substitution` over three combinations of `glass`, each
/// re-optimised for two iterations.
pub fn glass_substitution(variables: &[Value], glass: &[Value]) -> Value {
    json!({"variables":variables,"material_variables":glass,"max_combinations":3,
        "local_iterations":2,"top_n":3,"seed":71})
}

/// Every solver job ([`JOBS`]) with its settings: `merit_evaluation` at its
/// defaults, [`optimize`] with three iterations, [`tolerance`] and
/// [`glass_substitution`].
pub fn jobs(variables: &[Value], glass: &[Value]) -> Vec<(&'static str, Value)> {
    vec![
        ("merit_evaluation", Value::Null),
        ("optimize", optimize(variables, 3)),
        ("tolerance", tolerance()),
        ("glass_substitution", glass_substitution(variables, glass)),
    ]
}

/// Makes the curvature of every curved surface a variable of the session's
/// project (its `variables` section), unbounded.
pub fn set_curvature_variables(session: &mut Session) {
    let vars: Vec<Value> = inner(session.system())
        .iter()
        .filter(|x| x.shape.curvature() != 0.)
        .map(|x| json!({"param": {"param": "curvature", "surface": x.id}, "lower": null, "upper": null}))
        .collect();
    session
        .set_project_section("variables", Value::Array(vars))
        .unwrap();
}
