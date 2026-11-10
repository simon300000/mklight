//! Solves kept in effect while candidates are evaluated.

use std::sync::Arc;

use rayon::prelude::*;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use mklight_trace::{SolveSettings, apply_solves_with};

use super::{Operand, OperandError, OperandEvaluator, OperandKind, OperandValue};
use crate::deriv::FdScheme;

/// An evaluator that applies the solves of every candidate system
/// (`mklight_trace::apply_solves`, `docs/SOLVES.md`) before `inner`
/// evaluates it. The solves of a model then hold in every system a job
/// evaluates on it, whatever the job changed: the perturbed system of a
/// tolerance trial and each candidate of its compensator solve.
///
/// A system arrives with its configuration applied, so its solves run after
/// the configuration's overrides. A candidate whose solves cannot be applied
/// has no operand values (`OperandError::Invalid`, with the solver's
/// reason). A system without solves goes to `inner` as it is.
pub struct WithSolves<E> {
    inner: E,
    library: Arc<MaterialLibrary>,
    settings: SolveSettings,
}

impl<E: OperandEvaluator> WithSolves<E> {
    /// `inner` on solved candidates; `library` resolves their materials.
    pub fn new(inner: E, library: Arc<MaterialLibrary>) -> Self {
        WithSolves {
            inner,
            library,
            settings: SolveSettings::default(),
        }
    }

    /// `system` as `inner` evaluates it: with its solves applied. `Err` for
    /// cancellation; the inner error is why the solves failed.
    pub fn solved(
        &self,
        system: &OpticalSystem,
        ctx: &JobContext,
    ) -> MkResult<Result<OpticalSystem, MkError>> {
        if system.solves.is_empty() {
            return Ok(Ok(system.clone()));
        }
        // The solver's progress is not the caller's.
        let own = JobContext::with(ctx.cancel.clone(), Default::default());
        match apply_solves_with(system, &self.library, &self.settings, &own) {
            Ok((solved, _)) => Ok(Ok(solved)),
            Err(MkError::Cancelled) => Err(MkError::Cancelled),
            Err(e) => Ok(Err(e)),
        }
    }
}

impl<E: OperandEvaluator> OperandEvaluator for WithSolves<E> {
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        self.inner.gpu_context()
    }
    fn supports(&self, kind: &OperandKind) -> bool {
        self.inner.supports(kind)
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate_batch_near(systems, operands, None, ctx)
    }
    /// The candidates and the anchor are solved, in parallel and each on
    /// its own, so a candidate's solved system does not depend on the batch.
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let solved = systems
            .par_iter()
            .map(|s| self.solved(s, ctx))
            .collect::<MkResult<Vec<_>>>()?;
        // An anchor whose solves fail still marks the batch as near it.
        let anchor = match anchor {
            Some(a) => Some(self.solved(a, ctx)?.unwrap_or_else(|_| a.clone())),
            None => None,
        };
        let mut failed = Vec::with_capacity(solved.len());
        let mut candidates = Vec::with_capacity(solved.len());
        for s in solved {
            match s {
                Ok(s) => {
                    failed.push(None);
                    candidates.push(s);
                }
                Err(e) => failed.push(Some(e.to_string())),
            }
        }
        let mut values = if candidates.is_empty() {
            Vec::new()
        } else {
            self.inner
                .evaluate_batch_near(&candidates, operands, anchor.as_ref(), ctx)?
        }
        .into_iter();
        if values.len() != candidates.len() {
            return Err(MkError::Numerical(
                "operand evaluator returned the wrong shape".into(),
            ));
        }
        Ok(failed
            .into_iter()
            .map(|f| match f {
                None => values.next().expect("one row per solved candidate"),
                Some(why) => vec![Err(OperandError::Invalid(why)); operands.len()],
            })
            .collect())
    }
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        self.inner.anchored_differences(base)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        self.inner.fd_relative_step(kind, scheme)
    }
    fn for_solve(&self) -> Option<Box<dyn OperandEvaluator + '_>> {
        let inner = self.inner.for_solve()?;
        Some(Box::new(WithSolves {
            inner,
            library: self.library.clone(),
            settings: self.settings,
        }))
    }
}

impl<T: OperandEvaluator + ?Sized> OperandEvaluator for Box<T> {
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        (**self).gpu_context()
    }
    fn supports(&self, kind: &OperandKind) -> bool {
        (**self).supports(kind)
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        (**self).evaluate_batch(systems, operands, ctx)
    }
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        (**self).evaluate_batch_near(systems, operands, anchor, ctx)
    }
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        (**self).anchored_differences(base)
    }
    fn evaluate_parametric(
        &self,
        batch: &super::ParametricBatch<'_>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, super::BuildError>>> {
        (**self).evaluate_parametric(batch, operands, ctx)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        (**self).fd_relative_step(kind, scheme)
    }
    fn for_solve(&self) -> Option<Box<dyn OperandEvaluator + '_>> {
        (**self).for_solve()
    }
}
