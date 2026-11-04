//! The complete CPU evaluator set used by headless optimisation jobs.
use super::mtf_eval::MtfEvaluator;
use super::solver_eval::SolverEvaluator;
use super::{
    CompositeEvaluator, Operand, OperandEvaluator, OperandKind, OperandValue, TraceEvaluator,
};
use crate::deriv::FdScheme;
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkResult, OpticalSystem};
use std::sync::Arc;

/// Ray, MTF, ghost and stray-light operands on the CPU.
#[derive(Clone)]
pub struct BuiltinEvaluator {
    pub trace: TraceEvaluator,
    pub mtf: MtfEvaluator,
    pub solvers: SolverEvaluator,
}

impl BuiltinEvaluator {
    pub fn new(library: Arc<MaterialLibrary>) -> Self {
        Self {
            trace: TraceEvaluator::new(library.clone()),
            mtf: MtfEvaluator::new(library.clone()),
            solvers: SolverEvaluator::new(library),
        }
    }
    pub fn builtin() -> Self {
        Self::new(Arc::new(MaterialLibrary::with_builtin()))
    }
    pub fn evaluate_system(
        &self,
        s: &OpticalSystem,
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<OperandValue>> {
        Ok(self
            .evaluate_batch(std::slice::from_ref(s), ops, ctx)?
            .remove(0))
    }
}

impl OperandEvaluator for BuiltinEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        self.trace.supports(kind) || self.mtf.supports(kind) || self.solvers.supports(kind)
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate_batch_near(systems, ops, None, ctx)
    }
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        CompositeEvaluator::new()
            .with(self.trace.clone())
            .with(self.mtf.clone())
            .with(self.solvers.clone())
            .evaluate_batch_near(systems, ops, anchor, ctx)
    }
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        self.trace.anchored_differences(base)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        if self.mtf.supports(kind) {
            self.mtf.fd_relative_step(kind, scheme)
        } else {
            self.trace.fd_relative_step(kind, scheme)
        }
    }
}
