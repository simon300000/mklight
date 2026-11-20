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
//!
//! A glass substitution finalist the reference reports infeasible (its
//! violation beyond the generic budget) is an unconverged re-optimisation
//! of a rejected combination: its merit and violation are not gated, only
//! the finalists' count and glasses, and a `feasible` label that the
//! reference's own perturbed runs also flip is excused
//! ([`excuse_finalists`]).

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

/// The reference's finalist `i` of a glass substitution: whether its merit
/// and violation are gated (reported feasible, or its violation within the
/// generic budget), with the reference's violation and label.
fn gated_finalist(cpu: &AnalysisResult, i: usize) -> (bool, f64, bool) {
    let r = &cpu.json["results"][i];
    let violation = r["violation"].as_f64().unwrap_or(f64::NAN);
    let feasible = r["feasible"].as_bool().unwrap_or(false);
    let band = crate::comparison::class_budget(crate::comparison::Class::Generic, violation);
    (feasible || violation <= band, violation, feasible)
}

/// Largest difference of `r` from the reference `cpu`: (merits, relative;
/// violations, absolute). A glass substitution counts only the finalists
/// whose outcome is gated ([`excuse_finalists`]).
pub fn deviation(analysis: &str, cpu: &AnalysisResult, r: &AnalysisResult) -> (f64, f64) {
    let largest = |key: &str, relative: bool| {
        let (a, b) = (scores(analysis, cpu, key), scores(analysis, r, key));
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(&b)
            .enumerate()
            .filter(|(i, _)| analysis != "glass_substitution" || gated_finalist(cpu, *i).0)
            .map(|(_, (x, y))| {
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

/// The reference's perturbed runs of a truncated solve (the [`nudged`]
/// system, the device's step); none for other analyses or failed runs.
pub fn reference_runs(
    session: &Session,
    nudged: &Session,
    analysis: &str,
    settings: &Value,
) -> Vec<AnalysisResult> {
    if !matches!(analysis, "optimize" | "glass_substitution") {
        return vec![];
    }
    let run = |s: &Session, settings: &Value| {
        let mut request = AnalysisRequest::new(analysis, settings.clone());
        request.backend = Some(Backend::CpuReference);
        s.run_analysis(&request, &JobContext::new()).ok()
    };
    let mut stepped = settings.clone();
    stepped["fd_relative_step"] = json!(DEVICE_STEP);
    [run(nudged, settings), run(session, &stepped)]
        .into_iter()
        .flatten()
        .collect()
}

/// The reference's movement (merits, violations): the largest
/// [`deviation`] of its perturbed `runs` ([`reference_runs`]) from `cpu`;
/// zero for the analyses that are not truncated solves.
pub fn movement(analysis: &str, cpu: &AnalysisResult, runs: &[AnalysisResult]) -> (f64, f64) {
    runs.iter()
        .map(|r| deviation(analysis, cpu, r))
        .fold((0., 0.), |m, d| (m.0.max(d.0), m.1.max(d.1)))
}

/// The gate `failures` of a glass substitution that its infeasible
/// finalists excuse: the merit and violation of a finalist the reference
/// reports infeasible beyond the generic budget, and a `feasible` label
/// that one of the reference's perturbed `runs` also flips. Returns the
/// failures that remain and a note per excused one.
pub fn excuse_finalists(
    analysis: &str,
    cpu: &AnalysisResult,
    runs: &[AnalysisResult],
    failures: &[String],
) -> (Vec<String>, Vec<String>) {
    if analysis != "glass_substitution" {
        return (failures.to_vec(), vec![]);
    }
    let (mut kept, mut notes) = (Vec::new(), Vec::new());
    for f in failures {
        let path = crate::comparison::failure_path(f);
        let finalist = path
            .strip_prefix("result.finalists[")
            .and_then(|p| p.split_once("]."))
            .and_then(|(i, key)| Some((i.parse::<usize>().ok()?, key)));
        let note = match finalist {
            Some((i, "merit" | "violation")) => {
                let (gated, violation, _) = gated_finalist(cpu, i);
                (!gated).then(|| {
                    format!("{path} not gated: the reference's finalist is infeasible (violation {violation:e})")
                })
            }
            Some((i, "feasible")) => {
                let (_, _, feasible) = gated_finalist(cpu, i);
                runs.iter()
                    .any(|r| r.json["results"][i]["feasible"].as_bool() != Some(feasible))
                    .then(|| format!("{path} excused: the reference's perturbed runs also flip it"))
            }
            _ => None,
        };
        match note {
            Some(n) => notes.push(n),
            None => kept.push(f.clone()),
        }
    }
    (kept, notes)
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
        let path = crate::comparison::failure_path(f);
        path.starts_with(outcome)
            && ((path.ends_with(".merit") && deviation.0 <= 2. * movement.0 + 1e-12)
                || (path.ends_with(".violation") && deviation.1 <= 2. * movement.1 + 1e-12))
    })
}
