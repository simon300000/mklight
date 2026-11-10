//! Merit-function data model.
//!
//! A [`MeritFunctionSpec`] is a serialisable, versioned list of [`Operand`]s.
//! Each operand measures one quantity ([`OperandKind`]) at selected field /
//! wavelength / configuration, and contributes either
//!
//! * a weighted residual `w · ρ(v − t)²` to the merit, where `ρ` is the
//!   identity for `Equal` and the one-sided `max(0, ·)` / `min(0, ·)` for
//!   `LessThan` / `GreaterThan` ("boundary operands"), or
//! * an explicit constraint `v = t`, `v ≤ t` or `v ≥ t` handled by the
//!   optimiser (augmented Lagrangian / feasibility rules).
//!
//! Values are produced by an [`OperandEvaluator`]; [`OpticalProblem`] turns a
//! spec + variables + evaluator into a batched [`Problem`](crate::Problem).
//! [`trace_eval::TraceEvaluator`] computes paraxial, real-ray, wavefront and
//! model operands with the reference tracer; [`geometric::ModelEvaluator`]
//! covers model-only operands without tracing. MTF, ghost, stray-light and
//! plug-in operands come from further evaluators combined with
//! [`CompositeEvaluator`]. [`WithSolves`] applies a candidate's solves before
//! another evaluator sees it.

pub mod builtin;
pub mod default;
pub mod geometric;
pub mod gpu_eval;
mod mtf_default;
pub mod mtf_eval;
pub mod operand;
pub mod problem;
pub mod quadrature;
pub mod reference;
pub mod solved;
pub mod solver_eval;
mod surface;
pub mod trace_eval;

use serde::{Deserialize, Serialize};

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};

use crate::deriv::FdScheme;
use crate::problem::CandidateBatch;

pub use builtin::BuiltinEvaluator;

pub use operand::{Capability, OperandDescriptor, OperandKind, OperandRegistry};
pub use operand::{CentroidScope, RayGroup};
pub use problem::{OpticalProblem, VariableSpec};
pub use quadrature::{PupilNode, PupilSampling, PupilSymmetry};
pub use solved::WithSolves;
pub use trace_eval::{RunResult, TraceEvalSettings, TraceEvaluator};

/// Merit-file schema version. Bump when an existing serialised layout
/// changes; a spec at any other version is refused.
pub const MERIT_SCHEMA_VERSION: u32 = 1;

fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}
fn current_version() -> u32 {
    MERIT_SCHEMA_VERSION
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparison {
    /// Residual `v − t`.
    #[default]
    Equal,
    /// Residual `max(0, v − t)` (only penalised above the target).
    LessThan,
    /// Residual `min(0, v − t)` (only penalised below the target).
    GreaterThan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperandRole {
    /// Weighted least-squares residual.
    #[default]
    Residual,
    /// Explicit constraint (weight ignored).
    Constraint,
}

/// Field selection.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "field", rename_all = "snake_case")]
pub enum FieldSelector {
    /// Field-weighted average over all system fields (for aggregate
    /// operands such as RMS spot; per-ray operands need one field).
    #[default]
    All,
    /// One of the system's fields.
    Index { index: usize },
    /// An arbitrary normalised field point `(Hx, Hy)` (CONVENTIONS §3).
    Normalized { hx: f64, hy: f64 },
}

/// Wavelength selection.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "wavelength", rename_all = "snake_case")]
pub enum WavelengthSelector {
    #[default]
    Primary,
    /// Weighted polychromatic over all system wavelengths.
    All,
    Index {
        index: usize,
    },
}

/// One merit-function line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Operand {
    pub kind: OperandKind,
    #[serde(default)]
    pub target: f64,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default)]
    pub comparison: Comparison,
    #[serde(default)]
    pub role: OperandRole,
    #[serde(default)]
    pub field: FieldSelector,
    #[serde(default)]
    pub wavelength: WavelengthSelector,
    /// Configuration index (`OpticalSystem::with_configuration`).
    #[serde(default)]
    pub config: usize,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub label: String,
}

impl Operand {
    /// A residual operand with target 0 and weight 1, at the primary
    /// wavelength and — for kinds defined at one field point — at field 0
    /// (otherwise the field-averaged `All`).
    pub fn new(kind: OperandKind) -> Self {
        let field = if kind.needs_single_field() {
            FieldSelector::Index { index: 0 }
        } else {
            FieldSelector::All
        };
        Operand {
            kind,
            target: 0.0,
            weight: 1.0,
            comparison: Comparison::Equal,
            role: OperandRole::Residual,
            field,
            wavelength: WavelengthSelector::Primary,
            config: 0,
            enabled: true,
            label: String::new(),
        }
    }
    pub fn target(mut self, t: f64) -> Self {
        self.target = t;
        self
    }
    pub fn weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }
    pub fn less_than(mut self, t: f64) -> Self {
        self.comparison = Comparison::LessThan;
        self.target = t;
        self
    }
    pub fn greater_than(mut self, t: f64) -> Self {
        self.comparison = Comparison::GreaterThan;
        self.target = t;
        self
    }
    pub fn as_constraint(mut self) -> Self {
        self.role = OperandRole::Constraint;
        self
    }
    pub fn at_field(mut self, index: usize) -> Self {
        self.field = FieldSelector::Index { index };
        self
    }
    pub fn at_wavelength(mut self, index: usize) -> Self {
        self.wavelength = WavelengthSelector::Index { index };
        self
    }
    pub fn in_config(mut self, config: usize) -> Self {
        self.config = config;
        self
    }

    /// Residual of value `v` (before weighting).
    pub fn residual(&self, v: f64) -> f64 {
        let d = v - self.target;
        match self.comparison {
            Comparison::Equal => d,
            Comparison::LessThan => d.max(0.0),
            Comparison::GreaterThan => d.min(0.0),
        }
    }
    /// Bounds `(lower, upper)` when used as an explicit constraint.
    pub fn constraint_bounds(&self) -> (Option<f64>, Option<f64>) {
        match self.comparison {
            Comparison::Equal => (Some(self.target), Some(self.target)),
            Comparison::LessThan => (None, Some(self.target)),
            Comparison::GreaterThan => (Some(self.target), None),
        }
    }
}

/// What to do when an operand cannot be evaluated (ray failure, total
/// internal reflection, missed surface…).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Use `value` as the operand's *residual* (before weighting) and mark
    /// the candidate `Degraded`.
    Penalty { value: f64 },
    /// Mark the whole candidate `Failed` (merit = ∞, infeasible).
    FailCandidate,
}

impl Default for FailurePolicy {
    fn default() -> Self {
        FailurePolicy::Penalty { value: 1e3 }
    }
}

/// A complete, serialisable merit function.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeritFunctionSpec {
    #[serde(default = "current_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub name: String,
    pub operands: Vec<Operand>,
    #[serde(default)]
    pub on_failure: FailurePolicy,
}

impl Default for MeritFunctionSpec {
    fn default() -> Self {
        MeritFunctionSpec {
            schema_version: MERIT_SCHEMA_VERSION,
            name: String::new(),
            operands: Vec::new(),
            on_failure: FailurePolicy::default(),
        }
    }
}

impl MeritFunctionSpec {
    pub fn new(operands: Vec<Operand>) -> Self {
        MeritFunctionSpec {
            operands,
            ..Default::default()
        }
    }

    pub fn to_json(&self) -> MkResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse a spec at `MERIT_SCHEMA_VERSION` (see `from_value`).
    pub fn from_json(s: &str) -> MkResult<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        Self::from_value(v)
    }

    /// Parse a spec at `MERIT_SCHEMA_VERSION`; a missing `schema_version`
    /// means the current one. The version is checked before the layout, so
    /// a spec at another version gets a version error, not a field error.
    pub fn from_value(v: serde_json::Value) -> MkResult<Self> {
        let ver = match v.get("schema_version") {
            None | Some(serde_json::Value::Null) => u64::from(MERIT_SCHEMA_VERSION),
            Some(x) => x.as_u64().ok_or_else(|| {
                MkError::Parse(format!(
                    "merit function schema_version must be a non-negative integer, got {x}"
                ))
            })?,
        };
        if ver != u64::from(MERIT_SCHEMA_VERSION) {
            return Err(MkError::Parse(format!(
                "merit function schema {ver} is not supported (this engine reads \
                 {MERIT_SCHEMA_VERSION})"
            )));
        }
        Ok(serde_json::from_value(v)?)
    }

    /// Structural checks: known kinds (a `Custom` kind must name a
    /// registered plug-in), valid `Combine` references (to earlier operands
    /// only, so evaluation order is well defined), finite targets,
    /// non-negative weights, selectors compatible with the kind (one field /
    /// wavelength for per-ray kinds; no field or wavelength choice for kinds
    /// that ignore it) and a [`RayGroup`] for every
    /// `Centroid`-referenced per-ray operand.
    pub fn validate(&self, registry: &OperandRegistry) -> MkResult<()> {
        for (i, op) in self.operands.iter().enumerate() {
            if !registry.knows(&op.kind) {
                return Err(MkError::InvalidParam(format!(
                    "operand {i}: unknown kind '{}'",
                    op.kind.name()
                )));
            }
            let bad = |why: &str| {
                Err(MkError::InvalidParam(format!(
                    "operand {i} ('{}'): {why}",
                    op.kind.name()
                )))
            };
            let k = &op.kind;
            if k.needs_single_field() && op.field == FieldSelector::All {
                return bad("needs a single field, not `all`");
            }
            if !k.uses_field() && op.field != FieldSelector::All {
                return bad("does not depend on the field; leave the field selector at `all`");
            }
            if k.needs_single_wavelength() && op.wavelength == WavelengthSelector::All {
                return bad("needs a single wavelength, not `all`");
            }
            if !k.uses_wavelength() && op.wavelength != WavelengthSelector::Primary {
                return bad("does not depend on the wavelength; leave it at `primary`");
            }
            if k.missing_ray_group() {
                return bad("a centroid reference needs its ray group (pupil sampling)");
            }
            if !(op.weight >= 0.0 && op.weight.is_finite()) || !op.target.is_finite() {
                return Err(MkError::InvalidParam(format!(
                    "operand {i}: weight must be ≥ 0 and target finite"
                )));
            }
            if let OperandKind::Combine { op: c, args } = &op.kind {
                use operand::CombineOp::*;
                let arity_ok = match c {
                    Difference | Quotient => args.len() == 2,
                    Abs => args.len() == 1,
                    Sum | Product | Min | Max => !args.is_empty(),
                };
                if !arity_ok || args.iter().any(|&a| a >= i) {
                    return Err(MkError::InvalidParam(format!(
                        "operand {i}: combine needs valid arguments referring to earlier operands"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Checks against a concrete system: field, wavelength and configuration
    /// indices exist (also the colour wavelengths of a tolerance-sensitivity
    /// criterion).
    pub fn validate_for(&self, s: &OpticalSystem) -> MkResult<()> {
        let nf = s.fields.fields.len();
        let nw = s.wavelengths.wavelengths.len();
        let nc = s.configurations.len().max(1);
        for (i, op) in self.enabled() {
            let bad = |why: String| Err(MkError::InvalidParam(format!("operand {i}: {why}")));
            if let FieldSelector::Index { index } = op.field
                && index >= nf
            {
                return bad(format!("field {index} does not exist ({nf} fields)"));
            }
            if let WavelengthSelector::Index { index } = op.wavelength
                && index >= nw
            {
                return bad(format!(
                    "wavelength {index} does not exist ({nw} wavelengths)"
                ));
            }
            if op.config >= nc {
                return bad(format!("configuration {} does not exist", op.config));
            }
            // A tolerance sensitivity evaluates its criterion on this system.
            let mut kind = &op.kind;
            while let OperandKind::ToleranceSensitivity { criterion, .. } = kind {
                kind = criterion;
            }
            if let OperandKind::LateralColour { short, long }
            | OperandKind::AxialColour { short, long } = *kind
                && (short >= nw || long >= nw)
            {
                return bad("colour operand wavelength index out of range".into());
            }
        }
        Ok(())
    }

    pub fn enabled(&self) -> impl Iterator<Item = (usize, &Operand)> {
        self.operands.iter().enumerate().filter(|(_, o)| o.enabled)
    }
}

/// Why an operand could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub enum OperandError {
    /// A ray failed (missed a surface, TIR, vignetted chief ray…).
    RayFailure(String),
    /// The operand is not meaningful for this system (e.g. edge thickness
    /// with automatic semi-diameters).
    Invalid(String),
    /// The evaluator does not implement this kind.
    Unsupported,
}

pub type OperandValue = Result<f64, OperandError>;

/// Candidates given as parameter vectors on one shared base system — the
/// form a GPU backend wants: pack `base` once, then patch only the `params`
/// slots per candidate instead of re-resolving a full system each time.
#[derive(Clone, Copy, Debug)]
pub struct ParametricBatch<'a> {
    pub base: &'a OpticalSystem,
    /// Parameter of each candidate coordinate.
    pub params: &'a [ParamPath],
    pub candidates: &'a CandidateBatch,
    /// Configuration applied *after* the parameters (its overrides win).
    pub config: usize,
}

/// Why a candidate of a [`ParametricBatch`] could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildError {
    /// A parameter value was rejected (the candidate is invalid).
    Parameters(String),
    /// The configuration could not be applied (its operands are invalid).
    Configuration(String),
}

impl ParametricBatch<'_> {
    pub fn len(&self) -> usize {
        self.candidates.len()
    }
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
    /// The base with candidate `k`'s parameters applied (no configuration).
    pub fn apply(&self, k: usize) -> Result<OpticalSystem, BuildError> {
        self.apply_point(self.candidates.point(k))
    }
    fn apply_point(&self, x: &[f64]) -> Result<OpticalSystem, BuildError> {
        let mut s = self.base.clone();
        for (p, &v) in self.params.iter().zip(x) {
            p.set(&mut s, v)
                .map_err(|e| BuildError::Parameters(e.to_string()))?;
        }
        Ok(s)
    }
    /// The full system of candidate `k` (parameters, then configuration).
    pub fn build(&self, k: usize) -> Result<OpticalSystem, BuildError> {
        self.configure(self.apply(k)?)
    }
    fn configure(&self, s: OpticalSystem) -> Result<OpticalSystem, BuildError> {
        s.with_configuration(self.config)
            .map_err(|e| BuildError::Configuration(e.to_string()))
    }
    /// The full system at the batch's anchor point, if it has one and it
    /// can be built.
    pub fn anchor(&self) -> Option<OpticalSystem> {
        let x = self.candidates.anchor.as_deref()?;
        self.configure(self.apply_point(x).ok()?).ok()
    }
}

/// Computes operand values. Implementations are batched over candidate
/// systems so a GPU backend can trace all candidates in one dispatch.
///
/// `systems[k]` already has the configuration `config` applied; the
/// evaluator handles the field and wavelength selectors. Per-ray operands
/// with a `Centroid` reference carry their [`RayGroup`],
/// which says which rays define the reference.
pub trait OperandEvaluator: Sync {
    /// Explicit resident execution context, absent for the reference route.
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        None
    }

    /// Can this evaluator compute operands of `kind`?
    fn supports(&self, kind: &OperandKind) -> bool;

    /// Values `[system][operand]` for every operand in `operands` (all of
    /// which are supported). `Err` only for cancellation / infrastructure
    /// errors; per-operand failures go in the values.
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>>;

    /// [`evaluate_batch`](Self::evaluate_batch) for systems that are all
    /// small perturbations of `anchor` (the finite-difference neighbours of
    /// a Jacobian, tolerance trials). An evaluator may start its iterative
    /// solves from the anchor's (the CPU reference seeds its aims, see
    /// `mklight_trace::AimSeeds`); values then agree with `evaluate_batch`
    /// to within the solve tolerances. The default ignores the anchor.
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        _anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate_batch(systems, operands, ctx)
    }

    /// Do values near `base` depend on the batch they are evaluated in
    /// (anchor-seeded solves, device precision with reference completion)?
    /// Finite differences then re-evaluate their base point with the probes
    /// ([`Problem::anchored_differences`](crate::Problem::anchored_differences)).
    fn anchored_differences(&self, _base: &OpticalSystem) -> bool {
        false
    }

    /// Values for parameter-vector candidates (one row per candidate, or the
    /// reason it could not be built). [`OpticalProblem`] always calls this.
    /// The default builds every system and calls
    /// [`evaluate_batch_near`](Self::evaluate_batch_near) with the batch's
    /// anchor ([`CandidateBatch::anchor`](crate::CandidateBatch::anchor));
    /// a backend that keeps a packed copy of `batch.base` overrides it to
    /// upload only the parameter vectors.
    fn evaluate_parametric(
        &self,
        batch: &ParametricBatch<'_>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, BuildError>>> {
        let built: Vec<Result<OpticalSystem, BuildError>> =
            (0..batch.len()).map(|k| batch.build(k)).collect();
        let systems: Vec<OpticalSystem> = built.iter().filter_map(|b| b.clone().ok()).collect();
        let mut vals = if systems.is_empty() {
            Vec::new()
        } else {
            self.evaluate_batch_near(&systems, operands, batch.anchor().as_ref(), ctx)?
        }
        .into_iter();
        if vals.len() != systems.len() {
            return Err(MkError::Numerical(
                "operand evaluator returned the wrong shape".into(),
            ));
        }
        Ok(built
            .into_iter()
            .map(|b| b.map(|_| vals.next().expect("one row per built system")))
            .collect())
    }

    /// Relative finite-difference step `η` (see `deriv`) this evaluator's
    /// values of `kind` need for `scheme`, when the f64 defaults (`√ε`,
    /// `∛ε`) would difference rounding noise. `None` = the defaults.
    fn fd_relative_step(&self, _kind: &OperandKind, _scheme: FdScheme) -> Option<f64> {
        None
    }

    /// This evaluator for another solve, when it keeps state between the
    /// batches of one (the resident evaluator's learned ray routing): the
    /// same values, with that state starting afresh, so that solves running
    /// in parallel cannot influence each other. `None` (the default): the
    /// evaluator keeps no such state and serves every solve.
    fn for_solve(&self) -> Option<Box<dyn OperandEvaluator + '_>> {
        None
    }
}

/// Routes each operand to the first evaluator that supports it.
#[derive(Default)]
pub struct CompositeEvaluator {
    pub evaluators: Vec<Box<dyn OperandEvaluator + Send>>,
}

impl CompositeEvaluator {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, e: impl OperandEvaluator + Send + 'static) -> Self {
        self.evaluators.push(Box::new(e));
        self
    }
    fn route(&self, kind: &OperandKind) -> Option<usize> {
        self.evaluators.iter().position(|e| e.supports(kind))
    }
    /// Operand indices handled by each evaluator.
    fn routes(&self, operands: &[&Operand]) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.evaluators.len()];
        for (k, op) in operands.iter().enumerate() {
            if let Some(e) = self.route(&op.kind) {
                out[e].push(k);
            }
        }
        out
    }
}

impl OperandEvaluator for CompositeEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        self.route(kind).is_some()
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate_batch_near(systems, operands, None, ctx)
    }
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let mut out = vec![vec![Err(OperandError::Unsupported); operands.len()]; systems.len()];
        for (e, idx) in self.evaluators.iter().zip(self.routes(operands)) {
            if idx.is_empty() {
                continue;
            }
            let ops: Vec<&Operand> = idx.iter().map(|&k| operands[k]).collect();
            let vals = e.evaluate_batch_near(systems, &ops, anchor, ctx)?;
            for (s, row) in vals.into_iter().enumerate() {
                for (j, v) in row.into_iter().enumerate() {
                    out[s][idx[j]] = v;
                }
            }
        }
        Ok(out)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        self.route(kind)
            .and_then(|e| self.evaluators[e].fd_relative_step(kind, scheme))
    }
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        self.evaluators.iter().any(|e| e.anchored_differences(base))
    }
    /// Routes the parametric batch to each sub-evaluator's own
    /// `evaluate_parametric`, so a packed-system backend keeps its fast path.
    fn evaluate_parametric(
        &self,
        batch: &ParametricBatch<'_>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, BuildError>>> {
        let mut out: Vec<Result<Vec<OperandValue>, BuildError>> =
            vec![Ok(vec![Err(OperandError::Unsupported); operands.len()]); batch.len()];
        for (e, idx) in self.evaluators.iter().zip(self.routes(operands)) {
            if idx.is_empty() {
                continue;
            }
            let ops: Vec<&Operand> = idx.iter().map(|&k| operands[k]).collect();
            let vals = e.evaluate_parametric(batch, &ops, ctx)?;
            if vals.len() != batch.len() {
                return Err(MkError::Numerical(
                    "operand evaluator returned the wrong shape".into(),
                ));
            }
            for (row, v) in out.iter_mut().zip(vals) {
                match (row.as_mut(), v) {
                    (Ok(r), Ok(v)) => {
                        for (j, val) in v.into_iter().enumerate() {
                            r[idx[j]] = val;
                        }
                    }
                    (Ok(_), Err(err)) => *row = Err(err),
                    (Err(_), _) => {}
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::operand::*;
    use super::*;
    use mklight_core::SurfaceId;

    #[test]
    fn json_roundtrip_and_versioning() {
        let spec = MeritFunctionSpec::new(vec![
            Operand::new(OperandKind::Efl).target(50.0).weight(10.0),
            Operand::new(OperandKind::Mtf {
                frequency: 40.0,
                orientation: MtfOrientation::Sagittal,
                method: MtfMethod::Diffraction,
                defocus_mm: 0.0,
                detector: None,
                wavelengths: Vec::new(),
            })
            .greater_than(0.5)
            .at_field(2)
            .at_wavelength(1),
            Operand::new(OperandKind::CentreThickness {
                surface: SurfaceId(3),
                to: None,
            })
            .greater_than(1.0)
            .as_constraint(),
            Operand::new(OperandKind::Combine {
                op: CombineOp::Difference,
                args: vec![0, 1],
            }),
            Operand::new(OperandKind::Custom {
                name: "my_metric".into(),
                params: serde_json::json!({"k": 3}),
            }),
        ]);
        let js = spec.to_json().unwrap();
        let back = MeritFunctionSpec::from_json(&js).unwrap();
        assert_eq!(back, spec);
        // A minimal file without version and defaults parses as version 1.
        let minimal = r#"{"operands":[{"kind":{"type":"efl"},"target":35}]}"#;
        let m = MeritFunctionSpec::from_json(minimal).unwrap();
        assert_eq!(m.schema_version, MERIT_SCHEMA_VERSION);
        assert_eq!(m.operands[0].weight, 1.0);
        assert!(m.operands[0].enabled);
        // Every other version is rejected.
        for other in [0, 2, 99] {
            let js = format!(r#"{{"schema_version":{other},"operands":[]}}"#);
            let e = MeritFunctionSpec::from_json(&js).unwrap_err();
            assert!(e.to_string().contains("not supported"), "{e}");
        }
        // Registry knows the built-ins but not the custom kind until registered.
        let mut reg = OperandRegistry::builtin();
        assert!(spec.validate(&reg).is_err());
        reg.register_custom("my_metric", "test", "", true, true, serde_json::Value::Null)
            .unwrap();
        spec.validate(&reg).unwrap();
        assert!(
            reg.register_custom("efl", "dup", "", false, false, serde_json::Value::Null)
                .is_err()
        );
    }

    #[test]
    fn residual_comparisons() {
        let eq = Operand::new(OperandKind::Efl).target(10.0);
        let lt = Operand::new(OperandKind::Efl).less_than(10.0);
        let gt = Operand::new(OperandKind::Efl).greater_than(10.0);
        assert_eq!(eq.residual(12.0), 2.0);
        assert_eq!(lt.residual(12.0), 2.0);
        assert_eq!(lt.residual(8.0), 0.0);
        assert_eq!(gt.residual(8.0), -2.0);
        assert_eq!(gt.residual(12.0), 0.0);
        assert_eq!(gt.constraint_bounds(), (Some(10.0), None));
    }

    #[test]
    fn registry_covers_every_builtin_name_once() {
        let reg = OperandRegistry::builtin();
        let mut names: Vec<&str> = reg.kinds.iter().map(|d| d.name.as_str()).collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n);
        assert!(n >= 29);
        for d in &reg.kinds {
            // Templates serialise with their own name as the tag.
            let v = serde_json::to_value(&d.template).unwrap();
            assert_eq!(v["type"], d.name);
        }
        let bad = MeritFunctionSpec::new(vec![Operand::new(OperandKind::Combine {
            op: CombineOp::Sum,
            args: vec![0],
        })]);
        assert!(bad.validate(&reg).is_err(), "self reference");
    }
}
