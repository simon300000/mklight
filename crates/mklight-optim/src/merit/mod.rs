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
//! The tracer-backed evaluators are wired in a later phase; the
//! [`geometric::ModelEvaluator`] already covers model-only operands.

pub mod default;
pub mod geometric;
pub mod operand;
pub mod problem;
pub mod quadrature;

use serde::{Deserialize, Serialize};

use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};

pub use operand::{Capability, OperandDescriptor, OperandKind, OperandRegistry};
pub use problem::{OpticalProblem, VariableSpec};
pub use quadrature::{PupilNode, PupilSampling, PupilSymmetry};

/// Current merit-file schema version. Bump when an existing serialised
/// layout changes and add a migration step to `MeritFunctionSpec::migrate`.
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
    pub fn new(kind: OperandKind) -> Self {
        Operand {
            kind,
            target: 0.0,
            weight: 1.0,
            comparison: Comparison::Equal,
            role: OperandRole::Residual,
            field: FieldSelector::All,
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

    /// Parse any supported schema version, migrating to the current one.
    pub fn from_json(s: &str) -> MkResult<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        Self::from_value(v)
    }

    pub fn from_value(mut v: serde_json::Value) -> MkResult<Self> {
        let ver = v
            .get("schema_version")
            .and_then(|x| x.as_u64())
            .unwrap_or(1) as u32;
        if ver > MERIT_SCHEMA_VERSION {
            return Err(MkError::Parse(format!(
                "merit function schema {ver} is newer than supported {MERIT_SCHEMA_VERSION}"
            )));
        }
        Self::migrate(&mut v, ver)?;
        let mut spec: MeritFunctionSpec = serde_json::from_value(v)?;
        spec.schema_version = MERIT_SCHEMA_VERSION;
        Ok(spec)
    }

    /// Upgrade a serialised spec in place from version `from`. Each schema
    /// change adds one `from == k` step here.
    fn migrate(_v: &mut serde_json::Value, from: u32) -> MkResult<()> {
        match from {
            0 => Err(MkError::Parse("schema version 0 is invalid".into())),
            _ => Ok(()), // 1 = current
        }
    }

    /// Structural checks: known kinds, valid `Combine` references (to
    /// earlier operands only, so evaluation order is well defined), finite
    /// targets and non-negative weights.
    pub fn validate(&self, registry: &OperandRegistry) -> MkResult<()> {
        for (i, op) in self.operands.iter().enumerate() {
            if !registry.knows(&op.kind) {
                return Err(MkError::InvalidParam(format!(
                    "operand {i}: unknown kind '{}'",
                    op.kind.name()
                )));
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

/// Computes operand values. Implementations are batched over candidate
/// systems so a GPU backend can trace all candidates in one dispatch.
///
/// `systems[k]` already has the configuration `config` applied; the
/// evaluator handles the field and wavelength selectors.
pub trait OperandEvaluator: Sync {
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
        let mut out = vec![vec![Err(OperandError::Unsupported); operands.len()]; systems.len()];
        for (ei, e) in self.evaluators.iter().enumerate() {
            let idx: Vec<usize> = (0..operands.len())
                .filter(|&k| self.route(&operands[k].kind) == Some(ei))
                .collect();
            if idx.is_empty() {
                continue;
            }
            let ops: Vec<&Operand> = idx.iter().map(|&k| operands[k]).collect();
            let vals = e.evaluate_batch(systems, &ops, ctx)?;
            for (s, row) in vals.into_iter().enumerate() {
                for (j, v) in row.into_iter().enumerate() {
                    out[s][idx[j]] = v;
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
            })
            .greater_than(0.5)
            .at_field(2)
            .at_wavelength(1),
            Operand::new(OperandKind::CentreThickness {
                surface: SurfaceId(3),
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
        // Future versions are rejected.
        let future = r#"{"schema_version":99,"operands":[]}"#;
        assert!(MeritFunctionSpec::from_json(future).is_err());
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
