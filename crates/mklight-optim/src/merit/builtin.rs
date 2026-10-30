//! The complete CPU evaluator set used by headless optimisation jobs.
use super::mtf_eval::MtfEvaluator;
use super::solver_eval::SolverEvaluator;
use super::{
    CompositeEvaluator, Operand, OperandEvaluator, OperandKind, OperandValue, TraceEvaluator,
};
use crate::deriv::FdScheme;
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkResult, OpticalSystem};
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone)]
pub struct BuiltinEvaluator {
    trace: TraceEvaluator,
    pub mtf: MtfEvaluator,
    pub solvers: SolverEvaluator,
}

impl BuiltinEvaluator {
    pub const BACKEND: &'static str = TraceEvaluator::BACKEND;
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
    pub fn trace(&self) -> &TraceEvaluator {
        &self.trace
    }
    pub fn composite(&self) -> CompositeEvaluator {
        CompositeEvaluator::new()
            .with(self.trace.clone())
            .with(self.mtf.clone())
            .with(self.solvers.clone())
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
impl Deref for BuiltinEvaluator {
    type Target = TraceEvaluator;
    fn deref(&self) -> &Self::Target {
        &self.trace
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
        self.composite().evaluate_batch(systems, ops, ctx)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        self.composite().fd_relative_step(kind, scheme)
    }
}
