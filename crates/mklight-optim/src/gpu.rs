//! Backend-neutral optimisation algebra and batched optical evaluation.
//! These explicit adapters do not change engine selection. The context retains
//! base identity and cumulative execution evidence for the outer RunInfo.
use crate::linalg::Matrix;
use mklight_core::{JobContext, MkResult, RunInfo, Snapshot};
use mklight_gpu::{ComputeBackend, Df32, optim_ops::Request, resident::*};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
mod evaluator;
mod first_order;
pub mod scheduler;
pub use evaluator::{EvaluationStats, ResidentEvaluator};

type SamplingCache = Vec<(
    crate::merit::PupilSampling,
    Arc<Vec<crate::merit::PupilNode>>,
)>;
#[derive(Clone)]
pub struct GpuContext {
    pub backend: Arc<dyn ComputeBackend>,
    pub snapshot: Snapshot,
    pub identity: PlanIdentity,
    stats: Arc<Mutex<PlanStats>>,
    numeric_outputs_read: Arc<AtomicBool>,
    traced_rays: Arc<AtomicU64>,
    native_device_calls: Arc<AtomicU64>,
    samplings: Arc<Mutex<SamplingCache>>,
    native: Arc<Mutex<std::collections::BTreeMap<String, serde_json::Value>>>,
}
impl GpuContext {
    pub fn new(backend: Arc<dyn ComputeBackend>, snapshot: Snapshot) -> Self {
        Self {
            backend,
            snapshot,
            identity: PlanIdentity::default(),
            stats: Arc::default(),
            numeric_outputs_read: Arc::default(),
            traced_rays: Arc::default(),
            native_device_calls: Arc::default(),
            samplings: Arc::default(),
            native: Arc::default(),
        }
    }
    pub(crate) fn native_device_calls(&self) -> u64 {
        self.native_device_calls.load(Ordering::Relaxed)
    }
    pub(crate) fn record_traced_rays(&self, count: usize) {
        self.traced_rays.fetch_add(count as u64, Ordering::Relaxed);
    }
    pub fn traced_rays(&self) -> u64 {
        self.traced_rays.load(Ordering::Relaxed)
    }
    pub(crate) fn pupil_nodes(
        &self,
        sampling: &crate::merit::PupilSampling,
    ) -> Vec<crate::merit::PupilNode> {
        let mut cache = self.samplings.lock().unwrap();
        if let Some((_, nodes)) = cache.iter().find(|(s, _)| s == sampling) {
            return nodes.as_ref().clone();
        }
        let nodes = sampling.nodes();
        let bytes = nodes.len() * size_of::<crate::merit::PupilNode>();
        if bytes <= 64 * 1024 * 1024 {
            while cache
                .iter()
                .map(|(_, n)| n.len() * size_of::<crate::merit::PupilNode>())
                .sum::<usize>()
                + bytes
                > 64 * 1024 * 1024
                || cache.len() >= 16
            {
                cache.remove(0);
            }
            cache.push((*sampling, Arc::new(nodes.clone())));
        }
        nodes
    }
    pub fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.snapshot.clone());
        p.identity = self.identity.clone();
        p.identity.requested_backend = Some(self.backend.info().backend_id());
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
        let shape = shape.into();
        if BufferDesc::new(ElementType::DoubleFloat, shape.clone()).len()? != v.len() {
            return Err(mklight_core::MkError::InvalidParam(
                "optim numeric input length".into(),
            ));
        }
        self.fill(p, shape, |out: &mut [Df32]| {
            for (o, &x) in out.iter_mut().zip(v) {
                *o = Df32::from_f64(x);
            }
        })
    }
    pub fn submit(&self, p: ResidentPlan, ctx: &JobContext) -> MkResult<PlanResult> {
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let result = self.backend.submit_plan(p, &inner)?.wait()?;
        self.observe(result.stats.clone());
        Ok(result)
    }
    pub fn values(&self, r: &PlanResult, index: usize) -> MkResult<Vec<f64>> {
        let view = self.backend.read_view(r.outputs[index].view())?;
        let values: Vec<_> = view
            .as_slice::<Df32>()?
            .iter()
            .map(|v| v.to_f64())
            .collect();
        if values.iter().any(|v| v.is_finite()) {
            self.numeric_outputs_read.store(true, Ordering::Relaxed);
        }
        Ok(values)
    }
    pub fn stats(&self) -> PlanStats {
        self.stats.lock().unwrap().clone()
    }
    pub fn observe(&self, stats: PlanStats) {
        let mut total = self.stats.lock().unwrap();
        total.copied_bytes += stats.copied_bytes;
        total.wall_seconds += stats.wall_seconds;
        total.device_seconds += stats.device_seconds;
        total.command_buffers += stats.command_buffers;
        total.allocations += stats.allocations;
        total.reused_allocations += stats.reused_allocations;
        total.uploaded_bytes += stats.uploaded_bytes;
        total.parameter_bytes += stats.parameter_bytes;
        total.readback_bytes += stats.readback_bytes;
        total.peak_bytes = total.peak_bytes.max(stats.peak_bytes);
        total.fallback_count += stats.fallback_count;
        total.polished_rays += stats.polished_rays;
        for s in stats.stages {
            if let Some(t) = total.stages.iter_mut().find(|t| t.name == s.name) {
                t.wall_seconds += s.wall_seconds;
                t.device_seconds += s.device_seconds;
                t.command_buffers += s.command_buffers;
                t.max_command_seconds = t.max_command_seconds.max(s.max_command_seconds);
            } else {
                total.stages.push(s);
            }
        }
    }
    pub fn observe_run(&self, info: &RunInfo) {
        if info.precision != mklight_core::job::Precision::F64Reference {
            self.native_device_calls.fetch_add(1, Ordering::Relaxed);
        }
        fn plans(g: &GpuContext, v: &serde_json::Value) {
            if v.get("command_buffers").is_some()
                && v.get("stages").is_some()
                && let Ok(stats) = serde_json::from_value::<PlanStats>(v.clone())
            {
                g.observe(stats);
                return;
            }
            match v {
                serde_json::Value::Array(a) => {
                    for x in a {
                        plans(g, x)
                    }
                }
                serde_json::Value::Object(o) => {
                    for x in o.values() {
                        plans(g, x)
                    }
                }
                _ => {}
            }
        }
        plans(self, &info.settings);
        let mut native = self.native.lock().unwrap();
        let entry = native.entry(info.analysis.clone()).or_insert_with(
            || serde_json::json!({"calls":0,"reference_probes":0,"reference_rows":0}),
        );
        entry["calls"] = serde_json::json!(entry["calls"].as_u64().unwrap() + 1);
        entry["reference_probes"] = serde_json::json!(
            entry["reference_probes"].as_u64().unwrap()
                + info.settings["gpu"]["reference_probes"]
                    .as_u64()
                    .unwrap_or(0)
        );
        entry["reference_rows"] = serde_json::json!(
            entry["reference_rows"].as_u64().unwrap()
                + info.settings["reference_rows"]
                    .as_array()
                    .map_or(0, |a| a.len() as u64)
        );
        entry["last_run"] = serde_json::to_value(info).unwrap();
    }
    pub(crate) fn observe_ri_visibility(&self, lookups: u64, blocked: u64) {
        let mut native = self.native.lock().unwrap();
        let entry = native
            .entry("ri_visibility".into())
            .or_insert_with(|| serde_json::json!({"lookups":0,"blocked":0}));
        entry["lookups"] = serde_json::json!(entry["lookups"].as_u64().unwrap() + lookups);
        entry["blocked"] = serde_json::json!(entry["blocked"].as_u64().unwrap() + blocked);
    }
    pub fn record(&self, info: &mut RunInfo) {
        if self.numeric_outputs_read.load(Ordering::Relaxed) {
            info.backend = self.backend.info().backend_id();
            info.precision = self.backend.precision().model;
        }
        info.settings["optim_native_stages"] =
            serde_json::to_value(&*self.native.lock().unwrap()).unwrap();
        let stats = self.stats.lock().unwrap();
        info.settings["optim_resident"] = serde_json::json!({"version":mklight_gpu::optim_ops::VERSION,
            "device":self.backend.info(),"precision":self.backend.precision(),"identity":self.identity,"plans":*stats});
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
        ctx.check()?;
        if o == 0 {
            return Ok(Matrix::zeros(0, rows.len()));
        }
        let mut p = self.plan();
        let v = self.df(&mut p, [b, o], values)?;
        let indices = self.fill(&mut p, [rows.len(), 3], |out: &mut [u32]| {
            for (a, b) in out.as_chunks_mut::<3>().0.iter_mut().zip(rows) {
                a.copy_from_slice(b);
            }
        })?;
        let coeff = self.fill(&mut p, [rows.len(), 2], |out: &mut [Df32]| {
            for (a, b) in out.as_chunks_mut::<2>().0.iter_mut().zip(steps) {
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
        if self
            .backend
            .read_view(result.outputs[1].view())?
            .as_slice::<u32>()?
            .iter()
            .any(|&v| v != 0)
        {
            return Err(mklight_core::MkError::InvalidParam(
                "FD stencil row outside the batch".into(),
            ));
        }
        let mut data = self.values(&result, 0)?;
        if data.iter().any(|v| !v.is_finite()) {
            self.observe(PlanStats {
                fallback_count: rows.len() as u64,
                stages: vec![StageStats {
                    name: "FD output range reference".into(),
                    ..Default::default()
                }],
                ..Default::default()
            });
            for i in 0..o {
                for (j, (row, step)) in rows.iter().zip(steps).enumerate() {
                    let base = values[row[0] as usize * o + i];
                    data[i * rows.len() + j] = step[0] * (values[row[1] as usize * o + i] - base)
                        + step[1] * (values[row[2] as usize * o + i] - base);
                }
            }
        }
        Ok(Matrix::from_row_major(o, rows.len(), data))
    }
    /// A batch of weighted damped QR solves, retaining factors for callers
    /// that want device products. Damping augments the matrix, never J^T J.
    #[allow(clippy::too_many_arguments)]
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

/// Thin Q retained on device after a tall QR. Only the n×n unpermuted factor
/// crosses to the host for the serial trust-region secular equation.
pub struct ResidentQr {
    context: GpuContext,
    q: BufferView,
    rows: usize,
    cols: usize,
}
impl ResidentQr {
    pub fn project(&self, v: &[f64], ctx: &JobContext) -> MkResult<Vec<f64>> {
        let mut p = self.context.plan();
        let q = p.import(self.q.clone());
        let rhs = self.context.df(&mut p, [1, self.rows, 1], v)?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, self.cols, 1]));
        p.operations.push(ResidentOp::Optim(Request::Product {
            left: p.view(q)?,
            right: rhs,
            transpose_left: true,
            output,
        }));
        p.outputs = vec![output];
        let result = self.context.submit(p, ctx)?;
        self.context.values(&result, 0)
    }
}
impl GpuContext {
    /// Compress a tall system to its original-column-order factor and Qᵀr.
    /// Thin Q remains resident; a deficient factor returns `None` for f64 QR.
    pub fn reduce_qr(
        &self,
        j: &Matrix,
        r: &[f64],
        ctx: &JobContext,
    ) -> MkResult<Option<(ResidentQr, Matrix, Vec<f64>)>> {
        let (m, n) = (j.rows, j.cols);
        let mut p = self.plan();
        let matrix = self.df(&mut p, [1, m, n], &j.data)?;
        let rhs = self.df(&mut p, [1, m], r)?;
        let factors = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, m + n + 2, n]));
        let solution = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, n]));
        let rank_condition = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 4]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::QrFit(mklight_gpu::collectives::QrFit {
                matrix,
                rhs: rhs.clone(),
                rank_tolerance: 1e-12,
                factors,
                solution,
                rank_condition,
            }),
        ));
        let mut q = p.view(factors)?;
        q.desc.shape = vec![1, m, n];
        let mut rhs = rhs;
        rhs.desc.shape = vec![1, m, 1];
        let qtr = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, n, 1]));
        p.operations.push(ResidentOp::Optim(Request::Product {
            left: q,
            right: rhs,
            transpose_left: true,
            output: qtr,
        }));
        p.outputs = vec![factors, rank_condition, qtr];
        let result = self.submit(p, ctx)?;
        let rank = self.values(&result, 1)?;
        if rank[3] != 0. {
            self.observe(PlanStats {
                fallback_count: 1,
                stages: vec![StageStats {
                    name: "rank-deficient tall QR reference".into(),
                    ..Default::default()
                }],
                ..Default::default()
            });
            return Ok(None);
        }
        let mut factor_view = result.outputs[0].view();
        factor_view.offset = m * n;
        factor_view.desc.shape = vec![n + 2, n];
        let view = self.backend.read_view(factor_view)?;
        let v = view.as_slice::<Df32>()?;
        let mut unpermuted = Matrix::zeros(n, n);
        for col in 0..n {
            let original = v[(n + 1) * n + col].to_f64() as usize;
            let scale = v[n * n + original].to_f64();
            for row in 0..n {
                unpermuted[(row, original)] = v[row * n + col].to_f64() * scale;
            }
        }
        let mut q = result.outputs[0].view();
        q.desc.shape = vec![1, m, n];
        Ok(Some((
            ResidentQr {
                context: self.clone(),
                q,
                rows: m,
                cols: n,
            },
            unpermuted,
            self.values(&result, 2)?,
        )))
    }
}

/// Explicit algebra acceleration for an arbitrary batched problem. Optical
/// callers normally obtain this hook from `ResidentEvaluator` instead.
pub struct ResidentProblem<P> {
    pub problem: P,
    pub context: GpuContext,
}
impl<P: crate::problem::Problem> crate::problem::Problem for ResidentProblem<P> {
    fn gpu_context(&self) -> Option<&GpuContext> {
        Some(&self.context)
    }
    fn variables(&self) -> &[crate::problem::Variable] {
        self.problem.variables()
    }
    fn num_residuals(&self) -> usize {
        self.problem.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.problem.weights()
    }
    fn constraints(&self) -> &[crate::problem::Constraint] {
        self.problem.constraints()
    }
    fn evaluate_batch(
        &self,
        x: &crate::problem::CandidateBatch,
        ctx: &JobContext,
    ) -> MkResult<crate::problem::BatchEvaluation> {
        self.problem.evaluate_batch(x, ctx)
    }
    fn fd_relative_step(&self, s: crate::deriv::FdScheme) -> Option<f64> {
        self.problem.fd_relative_step(s)
    }
    fn jacobian(&self, x: &[f64], ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        self.problem.jacobian(x, ctx)
    }
    fn reference_evaluation(
        &self,
        x: &[f64],
        ctx: &JobContext,
    ) -> Option<MkResult<crate::problem::Evaluation>> {
        Some(crate::problem::evaluate_one(&self.problem, x, ctx))
    }
}
