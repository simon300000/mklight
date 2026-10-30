//! Optical variables and the bridge from a merit spec to the generic
//! [`Problem`] trait.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};

use super::operand::{CombineOp, OperandKind};
use super::{
    FailurePolicy, MeritFunctionSpec, OperandError, OperandEvaluator, OperandRegistry, OperandRole,
    OperandValue,
};
use crate::problem::{BatchEvaluation, CandidateBatch, Constraint, EvalStatus, Problem, Variable};

/// An optimisation variable on a model parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableSpec {
    pub param: ParamPath,
    #[serde(default)]
    pub lower: Option<f64>,
    #[serde(default)]
    pub upper: Option<f64>,
    /// Typical change; `None` = [`default_scale`].
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub fd_step: Option<f64>,
    #[serde(default)]
    pub label: String,
}

impl VariableSpec {
    pub fn new(param: ParamPath) -> Self {
        VariableSpec {
            param,
            lower: None,
            upper: None,
            scale: None,
            fd_step: None,
            label: String::new(),
        }
    }
    pub fn bounded(param: ParamPath, lower: f64, upper: f64) -> Self {
        VariableSpec {
            lower: Some(lower),
            upper: Some(upper),
            ..VariableSpec::new(param)
        }
    }
    /// The generic [`Variable`] for this spec, using `system`'s current value
    /// to pick a default scale.
    pub fn to_variable(&self, system: &OpticalSystem) -> MkResult<Variable> {
        let v = self.param.get(system)?;
        Ok(Variable {
            name: if self.label.is_empty() {
                format!("{:?}", self.param)
            } else {
                self.label.clone()
            },
            lower: self.lower,
            upper: self.upper,
            scale: self.scale.unwrap_or_else(|| default_scale(&self.param, v)),
            fd_step: self.fd_step,
        })
    }
}

/// Typical magnitude of a useful change in a parameter, used for finite
/// differences, simplex/population spreads and trust regions when the user
/// gives none. Heuristic, in the parameter's own units (mm, mm⁻¹, degrees).
pub fn default_scale(p: &ParamPath, value: f64) -> f64 {
    use ParamPath::*;
    let a = value.abs();
    match p {
        Curvature { .. } => (0.1 * a).max(1e-4),
        Thickness { .. } => (0.1 * a).max(0.1),
        Conic { .. } => (0.1 * a).max(0.1),
        // Coefficient of r^(2i+4): sag contributions ~1 µm at r ≈ 10 mm.
        Asphere { index, .. } => (0.1 * a).max(1e-3 * 10f64.powi(-(2 * *index as i32 + 4))),
        ParaxialFocalLength { .. } => (0.1 * a).max(1.0),
        ModelNd { .. } => 0.01,
        ModelVd { .. } => 1.0,
        ModelDpgf { .. } => 1e-3,
        DecenterX { .. } | DecenterY { .. } => 0.01,
        TiltX { .. } | TiltY { .. } | TiltZ { .. } => 0.01,
        SemiDiameter { .. } => (0.05 * a).max(0.1),
        _ => (0.1 * a).max(1e-3),
    }
}

/// A merit function + variables + evaluator, as a batched [`Problem`].
///
/// Each candidate `x` is applied to a copy of the base system through the
/// variables' `ParamPath`s (an invalid value marks the candidate `Failed`),
/// then operands are evaluated per configuration in one evaluator batch.
/// `Combine` operands are resolved afterwards in spec order. Failed
/// residual operands follow the spec's [`FailurePolicy`]; a failed
/// *constraint* operand always fails the candidate, since feasibility cannot
/// be certified without it.
pub struct OpticalProblem<E: OperandEvaluator> {
    base: OpticalSystem,
    var_specs: Vec<VariableSpec>,
    vars: Vec<Variable>,
    spec: MeritFunctionSpec,
    evaluator: E,
    residual_ops: Vec<usize>,
    constraint_ops: Vec<usize>,
    weights: Vec<f64>,
    constraints: Vec<Constraint>,
}

impl<E: OperandEvaluator> OpticalProblem<E> {
    pub fn new(
        base: OpticalSystem,
        variables: Vec<VariableSpec>,
        spec: MeritFunctionSpec,
        evaluator: E,
        registry: &OperandRegistry,
    ) -> MkResult<Self> {
        spec.validate(registry)?;
        for (i, op) in spec.enabled() {
            if !matches!(op.kind, OperandKind::Combine { .. }) && !evaluator.supports(&op.kind) {
                return Err(MkError::Unsupported(format!(
                    "operand {i} ('{}') has no evaluator",
                    op.kind.name()
                )));
            }
            if let OperandKind::Combine { args, .. } = &op.kind
                && args.iter().any(|&a| !spec.operands[a].enabled)
            {
                return Err(MkError::InvalidParam(format!(
                    "operand {i}: combine refers to a disabled operand"
                )));
            }
        }
        let vars = variables
            .iter()
            .map(|v| v.to_variable(&base))
            .collect::<MkResult<Vec<_>>>()?;
        let mut residual_ops = Vec::new();
        let mut constraint_ops = Vec::new();
        for (i, op) in spec.enabled() {
            match op.role {
                OperandRole::Residual => residual_ops.push(i),
                OperandRole::Constraint => constraint_ops.push(i),
            }
        }
        let weights = residual_ops
            .iter()
            .map(|&i| spec.operands[i].weight)
            .collect();
        let constraints = constraint_ops
            .iter()
            .map(|&i| {
                let op = &spec.operands[i];
                let (lower, upper) = op.constraint_bounds();
                Constraint {
                    name: if op.label.is_empty() {
                        op.kind.name().to_string()
                    } else {
                        op.label.clone()
                    },
                    lower,
                    upper,
                    scale: 1.0,
                }
            })
            .collect();
        Ok(OpticalProblem {
            base,
            var_specs: variables,
            vars,
            spec,
            evaluator,
            residual_ops,
            constraint_ops,
            weights,
            constraints,
        })
    }

    pub fn spec(&self) -> &MeritFunctionSpec {
        &self.spec
    }
    pub fn base(&self) -> &OpticalSystem {
        &self.base
    }
    pub fn variable_specs(&self) -> &[VariableSpec] {
        &self.var_specs
    }
    /// Spec indices of the residual operands, in residual order.
    pub fn residual_operands(&self) -> &[usize] {
        &self.residual_ops
    }
    pub fn constraint_operands(&self) -> &[usize] {
        &self.constraint_ops
    }

    /// Current parameter values of the base system.
    pub fn start_point(&self) -> MkResult<Vec<f64>> {
        self.var_specs
            .iter()
            .map(|v| v.param.get(&self.base))
            .collect()
    }

    /// The base system with `x` applied.
    pub fn apply(&self, x: &[f64]) -> MkResult<OpticalSystem> {
        let mut s = self.base.clone();
        for (v, &xi) in self.var_specs.iter().zip(x) {
            v.param.set(&mut s, xi)?;
        }
        Ok(s)
    }

    /// Raw values of every operand (disabled ones are `Err(Invalid)`) for
    /// each candidate; `None` for candidates that could not be built.
    pub fn operand_values(
        &self,
        x: &CandidateBatch,
        ctx: &JobContext,
    ) -> MkResult<Vec<Option<Vec<OperandValue>>>> {
        let systems: Vec<Option<OpticalSystem>> =
            (0..x.len()).map(|k| self.apply(x.point(k)).ok()).collect();
        self.system_values(&systems, ctx)
    }

    /// Operand values for already-built systems (`None` = could not be
    /// built). Used directly when there are no variables (e.g. tolerance
    /// trials without compensators).
    pub fn system_values(
        &self,
        systems: &[Option<OpticalSystem>],
        ctx: &JobContext,
    ) -> MkResult<Vec<Option<Vec<OperandValue>>>> {
        let n_ops = self.spec.operands.len();
        let mut out: Vec<Option<Vec<OperandValue>>> = systems
            .iter()
            .map(|s| {
                s.as_ref()
                    .map(|_| vec![Err(OperandError::Invalid("disabled".into())); n_ops])
            })
            .collect();
        let configs: BTreeSet<usize> = self
            .spec
            .enabled()
            .filter(|(_, o)| !matches!(o.kind, OperandKind::Combine { .. }))
            .map(|(_, o)| o.config)
            .collect();
        for c in configs {
            ctx.check()?;
            let ops: Vec<usize> = self
                .spec
                .enabled()
                .filter(|(_, o)| o.config == c && !matches!(o.kind, OperandKind::Combine { .. }))
                .map(|(i, _)| i)
                .collect();
            let mut cand = Vec::new();
            let mut sys_c = Vec::new();
            for (k, s) in systems.iter().enumerate() {
                if let Some(s) = s {
                    match s.with_configuration(c) {
                        Ok(sc) => {
                            cand.push(k);
                            sys_c.push(sc);
                        }
                        Err(e) => {
                            if let Some(v) = out[k].as_mut() {
                                for &i in &ops {
                                    v[i] = Err(OperandError::Invalid(e.to_string()));
                                }
                            }
                        }
                    }
                }
            }
            if sys_c.is_empty() {
                continue;
            }
            let refs: Vec<&super::Operand> = ops.iter().map(|&i| &self.spec.operands[i]).collect();
            let vals = self.evaluator.evaluate_batch(&sys_c, &refs, ctx)?;
            if vals.len() != sys_c.len() || vals.iter().any(|r| r.len() != ops.len()) {
                return Err(MkError::Numerical(
                    "operand evaluator returned the wrong shape".into(),
                ));
            }
            for (row, &k) in vals.into_iter().zip(&cand) {
                if let Some(v) = out[k].as_mut() {
                    for (val, &i) in row.into_iter().zip(&ops) {
                        v[i] = val;
                    }
                }
            }
        }
        // Combine operands, in order.
        for v in out.iter_mut().flatten() {
            for (i, op) in self.spec.enabled() {
                if let OperandKind::Combine { op: c, args } = &op.kind {
                    v[i] = combine(*c, args.iter().map(|&a| v[a].clone()).collect());
                }
            }
        }
        Ok(out)
    }
}

fn combine(op: CombineOp, args: Vec<OperandValue>) -> OperandValue {
    let vals: Vec<f64> = args.into_iter().collect::<Result<_, _>>()?;
    Ok(match op {
        CombineOp::Sum => vals.iter().sum(),
        CombineOp::Difference => vals[0] - vals[1],
        CombineOp::Product => vals.iter().product(),
        CombineOp::Quotient => {
            if vals[1] == 0.0 {
                return Err(OperandError::Invalid("division by zero".into()));
            }
            vals[0] / vals[1]
        }
        CombineOp::Abs => vals[0].abs(),
        CombineOp::Min => vals.iter().cloned().fold(f64::INFINITY, f64::min),
        CombineOp::Max => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    })
}

impl<E: OperandEvaluator> Problem for OpticalProblem<E> {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_residuals(&self) -> usize {
        self.residual_ops.len()
    }
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        let vals = self.operand_values(x, ctx)?;
        Ok(self.to_evaluation(vals))
    }
}

impl<E: OperandEvaluator> OpticalProblem<E> {
    /// Residuals, constraints and status from operand values, applying the
    /// failure policy.
    pub fn to_evaluation(&self, vals: Vec<Option<Vec<OperandValue>>>) -> BatchEvaluation {
        let (m, k) = (self.residual_ops.len(), self.constraint_ops.len());
        let mut out = BatchEvaluation::zeros(vals.len(), m, k);
        for (c, v) in vals.into_iter().enumerate() {
            let Some(v) = v else {
                out.status[c] = EvalStatus::Failed;
                continue;
            };
            let mut failures = 0u32;
            let mut failed = false;
            for (j, &i) in self.residual_ops.iter().enumerate() {
                let op = &self.spec.operands[i];
                out.residuals[c * m + j] = match &v[i] {
                    Ok(val) if val.is_finite() => op.residual(*val),
                    _ => match self.spec.on_failure {
                        FailurePolicy::Penalty { value } => {
                            failures += 1;
                            value
                        }
                        FailurePolicy::FailCandidate => {
                            failed = true;
                            0.0
                        }
                    },
                };
            }
            for (j, &i) in self.constraint_ops.iter().enumerate() {
                match &v[i] {
                    Ok(val) if val.is_finite() => out.constraints[c * k + j] = *val,
                    _ => failed = true,
                }
            }
            out.status[c] = if failed {
                EvalStatus::Failed
            } else if failures > 0 {
                EvalStatus::Degraded { failures }
            } else {
                EvalStatus::Ok
            };
        }
        out
    }
}
