//! The reference's own movement of a truncated solve (`optimize` with a
//! few iterations, `glass_substitution` with a few local iterations), the
//! allowance of the solver-outcome tests for a device run that misses a
//! merit or violation gate.
//!
//! Such outcomes follow the least-squares trajectory, which perturbations
//! far below the contract's gates move by more than the gates. A device run
//! differs from the reference in two ways: value noise and its
//! forward-difference step. The reference's movement is the larger of its
//! deviations from itself on a copy with one curvature changed by 1e-9
//! (relative) and with the device's step; a gate failure of the solve's
//! outcome (the final design's merit or violation, a finalist's) passes
//! when the device deviates at most twice as much.

use mklight_core::{JobContext, OpticalSystem, param::ParamPath};
use mklight_engine::backend::Backend;
use mklight_engine::{AnalysisRequest, AnalysisResult, Session};
use serde_json::{Value, json};

/// `ResidentEvaluator`'s forward step for ray operands
/// (`DEVICE_FD_RELATIVE_STEP_FORWARD`; the default merits of the tests are
/// spot merits).
pub const DEVICE_STEP: f64 = 1e-6;

/// The values (`merit` or `violation`) a solver outcome is judged by: the
/// final design's of an optimisation, the finalists' of a glass
/// substitution.
fn scores(analysis: &str, r: &AnalysisResult, key: &str) -> Vec<f64> {
    match analysis {
        "optimize" => vec![r.json["final"][key].as_f64().unwrap_or(f64::NAN)],
        "glass_substitution" => r.json["results"]
            .as_array()
            .map(|v| {
                v.iter()
                    .map(|x| x[key].as_f64().unwrap_or(f64::NAN))
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// Largest difference of `r` from the reference `cpu`: (merits, relative;
/// violations, absolute).
pub fn deviation(analysis: &str, cpu: &AnalysisResult, r: &AnalysisResult) -> (f64, f64) {
    let largest = |key: &str, relative: bool| {
        let (a, b) = (scores(analysis, cpu, key), scores(analysis, r, key));
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if relative { d / x.abs().max(1e-300) } else { d }
            })
            .fold(0.0, f64::max)
    };
    (largest("merit", true), largest("violation", false))
}

/// `system` with its first curvature changed by 1e-9 (relative).
pub fn nudged(system: &OpticalSystem) -> OpticalSystem {
    let mut s = system.clone();
    if let Some(id) = s.surfaces[1..]
        .iter()
        .find(|x| x.shape.curvature() != 0.)
        .map(|x| x.id)
    {
        let p = ParamPath::Curvature { surface: id };
        let c = p.get(&s).unwrap();
        p.set(&mut s, c * (1. + 1e-9)).unwrap();
    }
    s
}

/// The reference's movement (merits, violations) of `analysis` with
/// `settings`, whose reference result on `session` is `cpu`; `nudged` holds
/// [`nudged`] of the session's system. Zero for the analyses that are not
/// truncated solves.
pub fn reference_movement(
    session: &Session,
    nudged: &Session,
    analysis: &str,
    settings: &Value,
    cpu: &AnalysisResult,
) -> (f64, f64) {
    if !matches!(analysis, "optimize" | "glass_substitution") {
        return (0., 0.);
    }
    let moved = |s: &Session, settings: &Value| {
        let mut request = AnalysisRequest::new(analysis, settings.clone());
        request.backend = Some(Backend::CpuReference);
        s.run_analysis(&request, &JobContext::new())
            .map_or((0., 0.), |r| deviation(analysis, cpu, &r))
    };
    let mut stepped = settings.clone();
    stepped["fd_relative_step"] = json!(DEVICE_STEP);
    let (a, b) = (moved(nudged, settings), moved(session, &stepped));
    (a.0.max(b.0), a.1.max(b.1))
}

/// Does the device run of `analysis` pass with its gate `failures`? Only a
/// truncated solve's outcome is excused: the final design's merit or
/// violation of an optimisation, a finalist's of a glass substitution,
/// within twice the reference's `movement` (`deviation` is the device
/// run's). Every other failure (a start merit, a merit evaluation, a
/// tolerance figure, a status, a missing value) fails.
pub fn within_movement(
    analysis: &str,
    failures: &[String],
    deviation: (f64, f64),
    movement: (f64, f64),
) -> bool {
    let outcome = match analysis {
        "optimize" => "result.final.",
        "glass_substitution" => "result.finalists[",
        _ => return failures.is_empty(),
    };
    failures.iter().all(|f| {
        let path = super::comparison::failure_path(f);
        path.starts_with(outcome)
            && ((path.ends_with(".merit") && deviation.0 <= 2. * movement.0 + 1e-12)
                || (path.ends_with(".violation") && deviation.1 <= 2. * movement.1 + 1e-12))
    })
}
