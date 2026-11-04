//! Optical variables and the bridge from a merit spec to the generic
//! [`Problem`] trait.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};

use super::operand::{CombineOp, OperandKind};
use super::{
    BuildError, FailurePolicy, MeritFunctionSpec, OperandError, OperandEvaluator, OperandRegistry,
    OperandRole, OperandValue, ParametricBatch,
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
    /// Typical change; `None` = [`default_scale_for`].
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
            scale: self
                .scale
                .unwrap_or_else(|| default_scale_for(system, &self.param, v)),
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
        ShapeCoefficient { set, index, .. } => {
            use mklight_core::param::CoefficientSet::*;
            let floor = match set {
                Odd => 1e-3 * 10f64.powi(-(*index as i32 + 1)),
                XAsphere | YAsphere => 1e-3 * 10f64.powi(-(2 * *index as i32 + 4)),
                Phase => 0.1,             // radians
                Q | Xy | Zernike => 1e-3, // normalized sag amplitudes in mm
            };
            (0.1 * a).max(floor)
        }
        ShapeParam { parameter, .. } => {
            use mklight_core::param::ShapeParameter::*;
            match parameter {
                CurvatureX | CurvatureY => (0.1 * a).max(1e-4),
                ConicX | ConicY | NormRadius => (0.1 * a).max(0.1),
                AstigmatismAngleDeg => 1.0,
                DesignWavelengthUm => (0.01 * a).max(1e-3),
                // Discrete: ParamPath rejects non-integral orders.
                DesignOrder | TraceOrder => 1.0,
            }
        }
        Curvature { .. } => (0.1 * a).max(1e-4),
        Thickness { .. } => (0.1 * a).max(0.1),
        Conic { .. } => (0.1 * a).max(0.1),
        // Coefficient of r^(2i+4): sag contributions ~1 µm at r ≈ 10 mm.
        Asphere { index, .. } => (0.1 * a).max(1e-3 * 10f64.powi(-(2 * *index as i32 + 4))),
        ParaxialFocalLength { .. } => (0.1 * a).max(1.0),
        ModelNd { .. } | MaterialDnD { .. } => 0.01,
        ModelVd { .. } | MaterialDvD { .. } => 1.0,
        ModelDpgf { .. } => 1e-3,
        DecenterX { .. } | DecenterY { .. } => 0.01,
        TiltX { .. } | TiltY { .. } | TiltZ { .. } => 0.01,
        SemiDiameter { .. } => (0.05 * a).max(0.1),
        _ => (0.1 * a).max(1e-3),
    }
}

/// [`default_scale`], except for shape coefficients: their floor is the
/// change that moves the sag by about 1 µm (or the phase by 0.1 rad) at the
/// clear radius, divided by the term's degree so that slopes change by a
/// similar amount. `0.1 |value|` remains a lower bound.
pub fn default_scale_for(s: &OpticalSystem, p: &ParamPath, value: f64) -> f64 {
    use mklight_core::model::{ApertureSpec, SemiDiameter, SurfaceShape};
    use mklight_core::param::CoefficientSet::*;
    use mklight_core::surface_math::xy_powers;
    let ParamPath::ShapeCoefficient {
        surface,
        set,
        index,
    } = *p
    else {
        return default_scale(p, value);
    };
    let sf = &s.surfaces[s.surface_index(surface).expect("parameter already read")];
    // Normalised bases; the odd and axis series use physical coordinates.
    let norm = match &sf.shape {
        SurfaceShape::QCon { norm_radius, .. }
        | SurfaceShape::QBfs { norm_radius, .. }
        | SurfaceShape::XYPolynomial { norm_radius, .. }
        | SurfaceShape::ZernikeSag { norm_radius, .. } => Some(*norm_radius),
        SurfaceShape::Diffractive { phase, .. } => Some(phase.norm_radius),
        _ => None,
    };
    let radius = match (&sf.aperture, sf.semi_diameter) {
        (Some(a), _) => a.bounding_radius(),
        (None, SemiDiameter::Fixed { value }) => value,
        (None, SemiDiameter::Auto) => norm.unwrap_or(match s.aperture {
            ApertureSpec::EntrancePupilDiameter { value } => value / 2.0,
            _ => 10.0,
        }),
    };
    let floor = match set {
        Q | Zernike => 1e-3 / unit_basis_gain(&sf.shape, index, radius),
        _ => {
            let (degree, amplitude) = match set {
                Odd => (index + 1, 1e-3),
                Phase => (2 * index + 2, 0.1),
                Xy => {
                    let (m, n) = xy_powers(index);
                    (m + n, 1e-3)
                }
                _ => (2 * index + 4, 1e-3),
            };
            let r = radius / norm.unwrap_or(1.0);
            amplitude / (degree as f64 * r.powi(degree as i32))
        }
    };
    (0.1 * value.abs()).max(floor)
}

/// Q and Zernike bases contain lower powers too, so a monomial estimate of
/// their highest degree gives enormous steps inside the normalising disk.
/// Measure the unit basis term instead: the maximum of |sag| and
/// `radius·|gradient|` over the clear disk (65 radii × 32 azimuths).
fn unit_basis_gain(shape: &mklight_core::SurfaceShape, index: usize, radius: f64) -> f64 {
    use mklight_core::SurfaceShape::*;
    let mut coefficients = vec![0.0; index + 1];
    coefficients[index] = 1.0;
    let (basis, norm_radius) = match *shape {
        QCon { norm_radius, .. } => (
            QCon {
                curvature: 0.0,
                conic: 0.0,
                norm_radius,
                coefficients,
            },
            norm_radius,
        ),
        QBfs { norm_radius, .. } => (
            QBfs {
                curvature: 0.0,
                norm_radius,
                coefficients,
            },
            norm_radius,
        ),
        ZernikeSag {
            norm_radius,
            ordering,
            ..
        } => (
            ZernikeSag {
                curvature: 0.0,
                conic: 0.0,
                norm_radius,
                coefficients,
                ordering,
            },
            norm_radius,
        ),
        _ => unreachable!("Q and Zernike coefficients belong to their shapes"),
    };
    let radius = radius.min(norm_radius);
    let mut gain = 0.0_f64;
    for j in 0..=64 {
        let r = radius * j as f64 / 64.0;
        for a in 0..32 {
            let (sin, cos) = (std::f64::consts::TAU * a as f64 / 32.0).sin_cos();
            let z = mklight_core::surface_math::evaluate(&basis, r * cos, r * sin)
                .expect("unit basis inside its normalisation disk");
            gain = gain
                .max(z.sag.abs())
                .max(radius * z.gradient[0].hypot(z.gradient[1]));
        }
    }
    gain
}

/// A merit function + variables + evaluator, as a batched [`Problem`].
///
/// Candidates go to the evaluator as parameter vectors on the shared base
/// system ([`ParametricBatch`], one call per configuration), so a GPU backend
/// can keep the base packed and patch only the variables. An invalid
/// parameter value marks the candidate `Failed`; a configuration that cannot
/// be applied makes that configuration's operands `Invalid`.
/// `Combine` operands are resolved afterwards in spec order. Failed
/// residual operands follow the spec's [`FailurePolicy`]; a failed
/// *constraint* operand always fails the candidate, since feasibility cannot
/// be certified without it.
pub struct OpticalProblem<E: OperandEvaluator> {
    base: OpticalSystem,
    var_specs: Vec<VariableSpec>,
    params: Vec<ParamPath>,
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
        spec.validate_for(&base)?;
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
            params: variables.iter().map(|v| v.param.clone()).collect(),
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
        let n_ops = self.spec.operands.len();
        let mut out: Vec<Option<Vec<OperandValue>>> =
            vec![Some(vec![Err(OperandError::Invalid("disabled".into())); n_ops]); x.len()];
        let configs = self.configs();
        if configs.is_empty() {
            // Nothing to evaluate; still reject invalid parameter values.
            let b = self.parametric(x, 0);
            for (k, o) in out.iter_mut().enumerate() {
                if b.apply(k).is_err() {
                    *o = None;
                }
            }
        }
        for c in configs {
            ctx.check()?;
            let ops = self.config_operands(c);
            let refs: Vec<&super::Operand> = ops.iter().map(|&i| &self.spec.operands[i]).collect();
            let rows = self
                .evaluator
                .evaluate_parametric(&self.parametric(x, c), &refs, ctx)?;
            if rows.len() != x.len() {
                return Err(MkError::Numerical(
                    "operand evaluator returned the wrong shape".into(),
                ));
            }
            for (o, row) in out.iter_mut().zip(rows) {
                let Some(v) = o.as_mut() else { continue };
                match row {
                    Ok(row) if row.len() == ops.len() => {
                        for (val, &i) in row.into_iter().zip(&ops) {
                            v[i] = val;
                        }
                    }
                    Ok(_) => {
                        return Err(MkError::Numerical(
                            "operand evaluator returned the wrong shape".into(),
                        ));
                    }
                    Err(BuildError::Parameters(_)) => *o = None,
                    Err(BuildError::Configuration(e)) => {
                        for &i in &ops {
                            v[i] = Err(OperandError::Invalid(e.clone()));
                        }
                    }
                }
            }
        }
        self.resolve_combines(&mut out);
        Ok(out)
    }

    fn parametric<'a>(&'a self, x: &'a CandidateBatch, config: usize) -> ParametricBatch<'a> {
        ParametricBatch {
            base: &self.base,
            params: &self.params,
            candidates: x,
            config,
        }
    }

    /// Configurations used by enabled, directly evaluated operands.
    fn configs(&self) -> BTreeSet<usize> {
        self.spec
            .enabled()
            .filter(|(_, o)| !matches!(o.kind, OperandKind::Combine { .. }))
            .map(|(_, o)| o.config)
            .collect()
    }

    /// Enabled, directly evaluated operands of configuration `c`.
    fn config_operands(&self, c: usize) -> Vec<usize> {
        self.spec
            .enabled()
            .filter(|(_, o)| o.config == c && !matches!(o.kind, OperandKind::Combine { .. }))
            .map(|(i, _)| i)
            .collect()
    }

    /// Resolve `Combine` operands, in spec order.
    fn resolve_combines(&self, out: &mut [Option<Vec<OperandValue>>]) {
        for v in out.iter_mut().flatten() {
            for (i, op) in self.spec.enabled() {
                if let OperandKind::Combine { op: c, args } = &op.kind {
                    v[i] = combine(*c, args.iter().map(|&a| v[a].clone()).collect());
                }
            }
        }
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
        for c in self.configs() {
            ctx.check()?;
            let ops = self.config_operands(c);
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
        self.resolve_combines(&mut out);
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
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        self.evaluator.gpu_context()
    }

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
        Ok(self.to_evaluation(self.operand_values(x, ctx)?))
    }
    fn anchored_differences(&self) -> bool {
        self.evaluator.anchored_differences(&self.base)
    }
    /// The largest step any enabled operand's evaluator asks for (a step
    /// large enough for the noisiest operand costs the others only `O(h)`
    /// truncation, far below their noise floor at these sizes).
    fn fd_relative_step(&self, scheme: crate::deriv::FdScheme) -> Option<f64> {
        self.spec
            .enabled()
            .filter(|(_, o)| !matches!(o.kind, OperandKind::Combine { .. }))
            .filter_map(|(_, o)| self.evaluator.fd_relative_step(&o.kind, scheme))
            .reduce(f64::max)
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
