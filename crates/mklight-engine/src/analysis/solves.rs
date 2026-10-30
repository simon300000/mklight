//! Solves (`mklight_trace::solves`, docs/SOLVES.md).
//!
//! Applying solves changes the model, so it is a session operation
//! (`Session::apply_solves`: a new snapshot plus this report). The `solves`
//! analysis is its dry run: the same report, and the solved values, without
//! touching the session; with `configuration` set it solves that
//! configuration (which `apply_solves` does not store back, because a
//! solved configuration has no configuration list).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::JobContext;
use mklight_trace::{SolveSettings, apply_solves_configuration, apply_solves_with};

use super::geometric::{crate_json, output, parse};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::error::EngineResult;

pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(Solves));
}

/// Settings of the `solves` analysis and of `Session::apply_solves`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolveOptions {
    #[serde(flatten)]
    pub solver: SolveSettings,
    /// Solve this configuration instead of the base system (dry run only).
    pub configuration: Option<usize>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        SolveOptions {
            solver: SolveSettings::default(),
            configuration: None,
        }
    }
}

pub fn solve_settings_schema() -> Value {
    settings_from_default::<SolveOptions>(json!({
        "tolerance": {"minimum": 0.0, "maximum": 1.0, "description": "Relative tolerance on each condition and on the change of every solved parameter between sweeps, in (0, 1)."},
        "max_sweeps": {"minimum": 1, "maximum": 100_000, "description": "Upper bound on sweeps over all solves."},
        "max_iterations": {"minimum": 1, "maximum": 1_000_000, "description": "Upper bound on evaluations per solve and sweep."},
        "closed_form": {"description": "Use closed-form updates (false: bracketed root finding only, a cross-check)."},
        "configuration": {"type": ["integer", "null"], "minimum": 0, "description": "Solve this configuration (dry run only); null = the base system."}
    }))
}

pub fn solve_report_schema() -> Value {
    let outcome = obj(&[
        ("index", int("Index into OpticalSystem::solves.")),
        ("kind", string("Solve kind.")),
        ("target", opaque("The ParamPath the solve sets.")),
        ("previous", nullable(num("", "Value in the input system (null: none, e.g. an automatic semi-diameter)."))),
        ("value", num("", "Solved value (units of the target parameter).")),
        ("residual", num("", "measured − goal of the condition in the solved system (mm, slope, F/# or c·D − 1).")),
        ("method", enumeration(&["unchanged", "closed_form", "fixed_point", "root_finding"], "")),
        ("evaluations", int("")),
    ]);
    result(
        &[
            ("outcomes", arr(outcome, "One per solve, in declaration order.")),
            ("order", arr(int(""), "Evaluation order (indices into solves).")),
            ("sweeps", int("")),
            ("max_change", num("", "Largest relative parameter change in the last sweep.")),
            ("solved_params", arr(opaque("ParamPath"), "Parameters controlled by solves (exclude them from optimisation).")),
            ("warnings", arr(string(""), "")),
            ("configuration", nullable(int("The configuration solved, or null for the base system."))),
        ],
        vec![],
    )
}

pub struct Solves;

impl Analysis for Solves {
    fn name(&self) -> &str {
        "solves"
    }
    fn title(&self) -> &str {
        "Solves (preview)"
    }
    fn category(&self) -> &str {
        "model"
    }
    fn description(&self) -> &str {
        "Evaluates the system's solves (marginal/chief ray height and angle, F/#, \
         pickups, …) without changing the session: solved values, residuals and methods. \
         `apply_solves` on the session stores them as a new snapshot."
    }
    fn settings_schema(&self) -> Value {
        solve_settings_schema()
    }
    fn result_schema(&self) -> Value {
        solve_report_schema()
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let o: SolveOptions = parse(settings)?;
        let sys = input.snapshot.system();
        let (_, report) = match o.configuration {
            Some(k) => apply_solves_configuration(sys, &input.library, k, &o.solver, ctx)?,
            None => apply_solves_with(sys, &input.library, &o.solver, ctx)?,
        };
        let (v, w) = crate_json(&report)?;
        Ok(output(v, vec![], w))
    }
}
