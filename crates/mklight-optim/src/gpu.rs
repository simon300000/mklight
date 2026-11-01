//! Backend-neutral optimisation algebra and batched optical evaluation.
//! These explicit adapters do not change engine selection. The context retains
//! base identity and cumulative execution evidence for the outer RunInfo.
use crate::linalg::Matrix;
use mklight_core::{JobContext, MkResult, RunInfo, Snapshot};
use mklight_gpu::{ComputeBackend, Df32, optim_ops::Request, resident::*};
use std::sync::{Arc, Mutex};
pub mod scheduler;

#[derive(Clone)]
pub struct GpuContext {
    pub backend: Arc<dyn ComputeBackend>,
    pub snapshot: Snapshot,
    stats: Arc<Mutex<Vec<PlanStats>>>,
}
impl GpuContext {
    pub fn new(backend: Arc<dyn ComputeBackend>, snapshot: Snapshot) -> Self {
        Self {
            backend,
            snapshot,
            stats: Arc::default(),
        }
    }
    pub fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.snapshot.clone());
        p.settings = serde_json::json!({"optim_plan_version":mklight_gpu::optim_ops::VERSION});
        p
    }
    pub fn fill<T: BufferElement>(
        &self,
        p: &mut ResidentPlan,
        shape: impl Into<Vec<usize>>,
        mut fill: impl FnMut(&mut [T]),
    ) -> MkResult<PlanView> {
        let h = self
            .backend
            .fill_buffer(BufferDesc::new(T::ELEMENT, shape), &mut |mut f| {
                fill(f.as_mut_slice::<T>()?);
                Ok(())
            })?;
        let id = p.import(h.view());
        p.view(id)
    }
    pub fn df(
        &self,
        p: &mut ResidentPlan,
        shape: impl Into<Vec<usize>>,
        v: &[f64],
    ) -> MkResult<PlanView> {
        self.fill(p, shape, |out: &mut [Df32]| {
            for (o, &x) in out.iter_mut().zip(v) {
                *o = Df32::from_f64(x);
            }
        })
    }
    pub fn submit(&self, p: ResidentPlan, ctx: &JobContext) -> MkResult<PlanResult> {
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let result = self.backend.submit_plan(p, &inner)?.wait()?;
        self.stats.lock().unwrap().push(result.stats.clone());
        Ok(result)
    }
    pub fn values(&self, r: &PlanResult, index: usize) -> MkResult<Vec<f64>> {
        let view = self.backend.read_view(r.outputs[index].view())?;
        Ok(view
            .as_slice::<Df32>()?
            .iter()
            .map(|v| v.to_f64())
            .collect())
    }
    pub fn record(&self, info: &mut RunInfo) {
        let stats = self.stats.lock().unwrap();
        info.settings["optim_resident"] = serde_json::json!({"version":mklight_gpu::optim_ops::VERSION,
            "device":self.backend.info(),"precision":self.backend.precision(),"plans":*stats});
    }
    /// Difference a complete set of successful stencils in a single command.
    pub fn differences(
        &self,
        values: &[f64],
        b: usize,
        o: usize,
        rows: &[[u32; 3]],
        steps: &[[f64; 2]],
        ctx: &JobContext,
    ) -> MkResult<Matrix> {
        let mut p = self.plan();
        let v = self.df(&mut p, [b, o], values)?;
        let indices = self.fill(&mut p, [rows.len(), 3], |out: &mut [u32]| {
            for (a, b) in out.chunks_exact_mut(3).zip(rows) {
                a.copy_from_slice(b);
            }
        })?;
        let coeff = self.fill(&mut p, [rows.len(), 2], |out: &mut [Df32]| {
            for (a, b) in out.chunks_exact_mut(2).zip(steps) {
                a[0] = Df32::from_f64(b[0]);
                a[1] = Df32::from_f64(b[1]);
            }
        })?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [o, rows.len()]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [rows.len()]));
        p.operations.push(ResidentOp::Optim(Request::Difference {
            values: v,
            rows: indices,
            steps: coeff,
            output,
            status,
        }));
        p.outputs = vec![output, status];
        let result = self.submit(p, ctx)?;
        Ok(Matrix::from_row_major(
            o,
            rows.len(),
            self.values(&result, 0)?,
        ))
    }
    /// A batch of weighted damped QR solves, retaining factors for callers
    /// that want device products. Damping augments the matrix, never J^T J.
    pub fn least_squares(
        &self,
        j: &[f64],
        r: &[f64],
        weights: &[f64],
        diagonal: &[f64],
        b: usize,
        m: usize,
        n: usize,
        damping: f64,
        ctx: &JobContext,
    ) -> MkResult<PlanResult> {
        let mut p = self.plan();
        let jacobian = self.df(&mut p, [b, m, n], j)?;
        let residuals = self.df(&mut p, [b, m], r)?;
        let weights = self.df(&mut p, [m], weights)?;
        let diagonal = self.df(&mut p, [b, n], diagonal)?;
        let matrix = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, m + n, n]));
        let rhs = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, m + n]));
        p.operations.push(ResidentOp::Optim(Request::LmRows {
            jacobian,
            residuals,
            weights,
            diagonal,
            damping: Df32::from_f64(damping),
            matrix,
            rhs,
        }));
        let factors = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [b, m + 2 * n + 2, n],
        ));
        let solution = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, n]));
        let rank_condition = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, 4]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::QrFit(mklight_gpu::collectives::QrFit {
                matrix: p.view(matrix)?,
                rhs: p.view(rhs)?,
                rank_tolerance: 1e-12,
                factors,
                solution,
                rank_condition,
            }),
        ));
        p.outputs = vec![solution, rank_condition, factors];
        self.submit(p, ctx)
    }
}
