//! Resident candidate evaluation. Accuracy is checked by offline f64 regressions;
//! only unsupported work and device failures invoke the reference evaluator.
use super::GpuContext;
use crate::deriv::FdScheme;
use crate::merit::{
    gpu_eval::{gpu_fd_relative_step, gpu_supports},
    mtf_eval::MtfEvaluator,
    solver_eval::SolverEvaluator,
    *,
};
use crate::problem::Problem;
use mklight_core::{
    JobContext, MkError, MkResult, OpticalSystem, RunInfo, Snapshot, param::ParamPath,
};
use mklight_gpu::systems::{Candidate, CompiledSystem, Parameter, UploadedSystem};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvaluationStats {
    pub candidates: u64,
    pub device_candidates: u64,
    pub cpu_candidates: u64,
    pub device_operands: u64,
    pub host_operands: u64,
    pub fallback_operands: u64,
    pub fallbacks: BTreeMap<String, u64>,
}
struct Template {
    base: OpticalSystem,
    snapshot: Snapshot,
    params: Vec<ParamPath>,
    config: usize,
    uploaded: UploadedSystem,
    backend: Arc<dyn mklight_gpu::ComputeBackend>,
}
impl Drop for Template {
    fn drop(&mut self) {
        let _ = self.uploaded.release(self.backend.as_ref());
    }
}
#[derive(Clone)]
pub struct ResidentEvaluator {
    pub context: GpuContext,
    pub(crate) reference: BuiltinEvaluator,
    mtf: MtfEvaluator,
    solvers: SolverEvaluator,
    template: Arc<Mutex<Option<Arc<Template>>>>,
    stats: Arc<Mutex<EvaluationStats>>,
}
impl ResidentEvaluator {
    pub fn new(mut context: GpuContext, reference: BuiltinEvaluator) -> Self {
        // Same length-prefixed catalogue identity as the engine; computed once.
        let mut hash = blake3::Hasher::new();
        for catalogue in reference.trace.library().catalogs() {
            let bytes =
                serde_json::to_vec(catalogue.as_ref()).expect("serializable material catalogue");
            for part in [catalogue.name.as_bytes(), &bytes] {
                hash.update(&(part.len() as u64).to_le_bytes());
                hash.update(part);
            }
        }
        context.identity.library_hash = hash.finalize().to_hex().to_string();
        Self {
            mtf: reference.mtf.clone().with_gpu(context.clone()),
            solvers: reference.solvers.clone().with_gpu(context.clone()),
            context,
            reference,
            template: Arc::default(),
            stats: Arc::default(),
        }
    }
    pub fn run<T>(
        &self,
        start: &OpticalSystem,
        analysis: &str,
        settings: &impl Serialize,
        seed: Option<u64>,
        f: impl FnOnce() -> MkResult<T>,
    ) -> MkResult<RunResult<T>> {
        let mut run = self
            .reference
            .trace
            .run(start, analysis, settings, seed, f)?;
        self.record(&mut run.info);
        Ok(run)
    }
    pub fn stats(&self) -> EvaluationStats {
        self.stats.lock().unwrap().clone()
    }
    pub fn record(&self, info: &mut RunInfo) {
        self.context.record(info);
        info.settings["candidate_evaluation"] = serde_json::to_value(self.stats()).unwrap();
        if self.stats().device_operands > 0 {
            info.backend = self.context.backend.info().backend_id();
            info.precision = mklight_core::job::Precision::F32Gpu;
        }
    }
    fn template(&self, batch: &ParametricBatch<'_>, ctx: &JobContext) -> MkResult<Arc<Template>> {
        let mut cache = self.template.lock().unwrap();
        if let Some(t) = cache.as_ref().filter(|t| {
            t.base == *batch.base
                && t.params == batch.params
                && t.config == batch.config
                && Arc::ptr_eq(&t.backend, &self.context.backend)
        }) {
            return Ok(t.clone());
        }
        let compiled = CompiledSystem::compile(
            Snapshot::new(batch.base.clone())?,
            self.reference.trace.library().clone(),
            batch
                .params
                .iter()
                .cloned()
                .map(|path| Parameter {
                    path,
                    lower: -1e30,
                    upper: 1e30,
                })
                .collect(),
            (!batch.base.configurations.is_empty()).then_some(batch.config),
            ctx,
        )?;
        let uploaded = compiled.upload(self.context.backend.as_ref(), ctx)?;
        let t = Arc::new(Template {
            base: batch.base.clone(),
            snapshot: compiled.snapshot().clone(),
            params: batch.params.to_vec(),
            config: batch.config,
            uploaded,
            backend: self.context.backend.clone(),
        });
        *cache = Some(t.clone());
        Ok(t)
    }
    fn evaluate(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        template: Option<(&Template, &[Candidate])>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        ctx.check()?;
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let tracers: Vec<_> = systems
            .par_iter()
            .map(|system| self.reference.trace.tracer(system).ok().map(Arc::new))
            .collect();
        let mut result = vec![vec![Err(OperandError::Unsupported); ops.len()]; systems.len()];
        // Per value: from the device, or completed on the reference after a
        // device fallback (host stages of a route are neither).
        let mut on_device = vec![vec![false; ops.len()]; systems.len()];
        let mut completed = vec![vec![false; ops.len()]; systems.len()];
        let mut device_rows = vec![false; systems.len()];
        let mut fallback_rows = vec![false; systems.len()];
        let mut stats = EvaluationStats {
            candidates: systems.len() as u64,
            ..Default::default()
        };
        for route in 0..5 {
            ctx.report(route as f64 / 5., "resident candidate operands");
            let indices: Vec<_> = ops
                .iter()
                .enumerate()
                .filter(|(_, o)| route_for(&o.kind) == route)
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                continue;
            }
            let refs: Vec<_> = indices.iter().map(|&i| ops[i]).collect();
            let active: Vec<_> = tracers
                .iter()
                .enumerate()
                .filter(|(_, t)| t.is_some())
                .map(|(i, _)| i)
                .collect();
            let selected = || -> Vec<_> { active.iter().map(|&i| systems[i].clone()).collect() };
            let selected_tracers: Vec<_> = active
                .iter()
                .map(|&i| tracers[i].as_ref().unwrap().clone())
                .collect();
            let selected_rows: Vec<_> = template.map_or_else(Vec::new, |(_, rows)| {
                active.iter().map(|&i| rows[i].clone()).collect()
            });
            let selected_template = template.map(|(t, _)| (&t.uploaded, selected_rows.as_slice()));
            let mut context = self.context.clone();
            if let Some((template, _)) = template {
                context.snapshot = template.snapshot.clone();
                context.identity.derived_snapshot_hash = Some(template.snapshot.hash().to_hex());
            }
            let native_before = self.context.native_device_calls();
            let attempt = if active.is_empty() {
                Ok(vec![])
            } else {
                match route {
                    0 => crate::merit::gpu_eval::resident::evaluate(
                        &context,
                        &self.reference.trace,
                        &selected_tracers,
                        &refs,
                        selected_template,
                        &inner,
                    ),
                    1 => self.mtf.evaluate_batch(&selected(), &refs, &inner),
                    2 => self.solvers.evaluate_batch(&selected(), &refs, &inner),
                    4 => {
                        let mut evaluator = self.clone();
                        evaluator.context = context;
                        evaluator.first_order(&selected(), &refs, selected_template, &inner)
                    }
                    _ => Ok(vec![]),
                }
            };
            ctx.check()?;
            let mut missing = vec![vec![true; indices.len()]; systems.len()];
            match attempt {
                Err(MkError::Cancelled) => return Err(MkError::Cancelled),
                Err(e) => {
                    *stats.fallbacks.entry(e.to_string()).or_default() +=
                        (active.len() * indices.len()) as u64;
                }
                Ok(values) => {
                    for (row, values) in active.iter().copied().zip(values) {
                        for (col, value) in values.into_iter().enumerate() {
                            let i = indices[col];
                            // Device status and finite output guards require no f64 rerun.
                            match &value {
                                Err(OperandError::RayFailure(reason))
                                    if reason.contains("requires reference") =>
                                {
                                    *stats.fallbacks.entry(reason.clone()).or_default() += 1;
                                }
                                Ok(v) if !v.is_finite() => {
                                    *stats
                                        .fallbacks
                                        .entry("non-finite device operand".into())
                                        .or_default() += 1;
                                }
                                _ => {
                                    result[row][i] = value;
                                    missing[row][col] = false;
                                    if route == 2
                                        && self.context.native_device_calls() == native_before
                                    {
                                        stats.host_operands += 1;
                                        stats.fallback_operands += 1;
                                        fallback_rows[row] = true;
                                        *stats
                                            .fallbacks
                                            .entry("native solver completed with f64".into())
                                            .or_default() += 1;
                                    } else {
                                        stats.device_operands += 1;
                                        device_rows[row] = true;
                                        on_device[row][i] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Complete the remaining operands on the reference, batched per
            // set of missing operands. Model/paraxial host work is a stage of
            // the route; anything else is a counted fallback.
            let mut groups = BTreeMap::<Vec<usize>, Vec<usize>>::new();
            for (row, missing) in missing.iter().enumerate() {
                let cols: Vec<_> = (0..indices.len())
                    .filter(|&c| missing[c])
                    .map(|c| indices[c])
                    .collect();
                if !cols.is_empty() {
                    groups.entry(cols).or_default().push(row);
                }
            }
            for (cols, rows) in groups {
                ctx.check()?;
                let refs: Vec<_> = cols.iter().map(|&i| ops[i]).collect();
                let batch: Vec<_> = rows.iter().map(|&r| systems[r].clone()).collect();
                let values = self.reference.evaluate_batch(&batch, &refs, &inner)?;
                for (&row, values) in rows.iter().zip(values) {
                    for (&i, value) in cols.iter().zip(values) {
                        result[row][i] = value;
                        stats.host_operands += 1;
                        if route == 3
                            && matches!(
                                ops[i].kind.capability(),
                                Capability::Model | Capability::Paraxial
                            )
                        {
                            continue;
                        }
                        fallback_rows[row] = true;
                        stats.fallback_operands += 1;
                        completed[row][i] = true;
                        let reason = if route == 3 {
                            format!("{}: no resident operand adapter", ops[i].kind.name())
                        } else if tracers[row].is_none() {
                            "candidate cannot be traced".into()
                        } else {
                            continue;
                        };
                        *stats.fallbacks.entry(reason).or_default() += 1;
                    }
                }
            }
        }
        // One evaluator and precision per operand within the batch: where
        // the reference completed an operand for some candidates, it also
        // evaluates it for the others. A finite difference across the batch
        // (its probes and re-evaluated base point) then never mixes device
        // and f64 values, whose difference it would divide by the step.
        let mixed: Vec<usize> = (0..ops.len())
            .filter(|&i| {
                (0..systems.len()).any(|r| on_device[r][i])
                    && (0..systems.len()).any(|r| completed[r][i])
            })
            .collect();
        let mut groups = BTreeMap::<Vec<usize>, Vec<usize>>::new();
        for row in 0..systems.len() {
            let cols: Vec<_> = mixed
                .iter()
                .copied()
                .filter(|&i| on_device[row][i])
                .collect();
            if !cols.is_empty() {
                groups.entry(cols).or_default().push(row);
            }
        }
        for (cols, rows) in groups {
            ctx.check()?;
            let refs: Vec<_> = cols.iter().map(|&i| ops[i]).collect();
            let batch: Vec<_> = rows.iter().map(|&r| systems[r].clone()).collect();
            let values = self.reference.evaluate_batch(&batch, &refs, &inner)?;
            for (&row, values) in rows.iter().zip(values) {
                for (&i, value) in cols.iter().zip(values) {
                    result[row][i] = value;
                    stats.device_operands -= 1;
                    stats.host_operands += 1;
                    stats.fallback_operands += 1;
                    fallback_rows[row] = true;
                    *stats
                        .fallbacks
                        .entry("uniform reference operand in a mixed batch".into())
                        .or_default() += 1;
                }
            }
        }
        stats.device_candidates = device_rows.iter().filter(|&&v| v).count() as u64;
        stats.cpu_candidates = device_rows
            .iter()
            .zip(&fallback_rows)
            .filter(|(d, f)| !**d || **f)
            .count() as u64;
        let mut total = self.stats.lock().unwrap();
        total.device_candidates += stats.device_candidates;
        total.cpu_candidates += stats.cpu_candidates;
        total.candidates += stats.candidates;
        total.device_operands += stats.device_operands;
        total.host_operands += stats.host_operands;
        total.fallback_operands += stats.fallback_operands;
        for (k, v) in stats.fallbacks {
            *total.fallbacks.entry(k).or_default() += v;
        }
        ctx.check()?;
        ctx.report(1., "candidate batch complete");
        Ok(result)
    }
}
/// Candidates per resident plan, bounded so plans stay inside the 256 MiB
/// buffer pool. Measured peaks: 60 MiB for 1024 default-spot Cooke candidates
/// at the spot budget; dense wavefront batches (OPD and fit channels) reach
/// 127 MiB at their budget and 376 MiB at the spot budget.
fn batch_limit(system: &OpticalSystem, ops: &[&Operand], ri_nodes: usize) -> usize {
    let mut groups = Vec::new();
    for op in ops {
        let sampling = match &op.kind {
            OperandKind::RmsSpot { sampling, .. } | OperandKind::WavefrontRms { sampling, .. } => {
                Some((*sampling, op.wavelength))
            }
            OperandKind::RayAberration { group: Some(g), .. }
            | OperandKind::WavefrontAberration { group: Some(g), .. } => Some((
                g.sampling,
                if g.scope == CentroidScope::Polychromatic {
                    WavelengthSelector::All
                } else {
                    op.wavelength
                },
            )),
            OperandKind::SpotCentroid { .. } => Some((PupilSampling::default(), op.wavelength)),
            _ => None,
        };
        if let Some((sampling, wavelength)) = sampling {
            let key = (sampling, op.field, wavelength);
            if !groups.contains(&key) {
                groups.push(key);
            }
        }
    }
    let rays = ops.len()
        + ops
            .iter()
            .filter(|o| matches!(o.kind, OperandKind::RelativeIllumination))
            .count()
            * 2
            * system.fields.fields.len()
            * system.wavelengths.wavelengths.len()
            * ri_nodes
        + groups
            .iter()
            .map(|(s, f, w)| {
                let fields = if *f == FieldSelector::All {
                    system.fields.fields.len()
                } else {
                    1
                };
                let wavelengths = if *w == WavelengthSelector::All {
                    system.wavelengths.wavelengths.len()
                } else {
                    1
                };
                let mirrors = if s.symmetry == PupilSymmetry::Bilateral {
                    2
                } else {
                    1
                };
                mirrors * s.rings * s.arms * fields * wavelengths
            })
            .sum::<usize>();
    let wavefront = ops.iter().any(|o| {
        matches!(
            o.kind,
            OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
        )
    });
    let budget = if wavefront { 196_608 } else { 786_432 };
    (budget / rays.max(1)).clamp(1, 1024)
}

fn route_for(kind: &OperandKind) -> usize {
    if matches!(
        kind,
        OperandKind::Efl
            | OperandKind::Bfl
            | OperandKind::EflX
            | OperandKind::EflY
            | OperandKind::BflX
            | OperandKind::BflY
    ) {
        4
    } else if matches!(kind, OperandKind::RelativeIllumination) || gpu_supports(kind, true) {
        0
    } else if matches!(kind, OperandKind::Mtf { .. }) {
        1
    } else if SolverEvaluator::CAPABILITIES.contains(&kind.capability()) {
        2
    } else {
        3
    }
}
impl OperandEvaluator for ResidentEvaluator {
    fn gpu_context(&self) -> Option<&GpuContext> {
        Some(&self.context)
    }
    fn supports(&self, kind: &OperandKind) -> bool {
        self.reference.supports(kind)
    }
    /// The base point of a finite difference is evaluated with its probes,
    /// so all of them share device precision or, operand by operand, the
    /// reference (`evaluate`).
    fn anchored_differences(&self, _base: &OpticalSystem) -> bool {
        true
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        if matches!(kind, OperandKind::RelativeIllumination) {
            return Some(match scheme {
                FdScheme::Forward => 1e-6,
                FdScheme::Central => 1e-5,
            });
        }
        if gpu_supports(kind, true) {
            Some(gpu_fd_relative_step(kind, scheme))
        } else {
            self.reference.fd_relative_step(kind, scheme)
        }
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let mut values = Vec::with_capacity(systems.len());
        let limit = systems.first().map_or(256, |s| {
            batch_limit(
                s,
                ops,
                1 + self.reference.trace.settings().illumination_rings.max(2)
                    * self.reference.trace.settings().illumination_arms.max(8),
            )
        });
        let count = systems.len().div_ceil(limit);
        for (i, chunk) in systems.chunks(limit).enumerate() {
            values.extend(self.evaluate(
                chunk,
                ops,
                None,
                &ctx.subrange(i as f64 / count as f64, (i + 1) as f64 / count as f64),
            )?);
        }
        ctx.check()?;
        ctx.report(1., "candidate batch complete");
        Ok(values)
    }
    fn evaluate_parametric(
        &self,
        batch: &ParametricBatch<'_>,
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, BuildError>>> {
        ctx.check()?;
        let built: Vec<_> = (0..batch.len())
            .into_par_iter()
            .map(|i| batch.build(i))
            .collect();
        let template = match self.template(batch, &ctx.subrange(0., 0.05)) {
            Ok(t) => Some(t),
            Err(MkError::Cancelled) => return Err(MkError::Cancelled),
            Err(e) => {
                *self
                    .stats
                    .lock()
                    .unwrap()
                    .fallbacks
                    .entry(format!("P4: {e}"))
                    .or_default() += batch.len() as u64;
                None
            }
        };
        let mut rows = Vec::new();
        let mut systems = Vec::new();
        for (i, s) in built.iter().enumerate() {
            if let Ok(s) = s {
                systems.push(s.clone());
                rows.push(Candidate {
                    id: i as u64,
                    values: batch.candidates.point(i).to_vec(),
                });
            }
        }
        let mut values = Vec::new();
        let limit = batch_limit(
            batch.base,
            ops,
            1 + self.reference.trace.settings().illumination_rings.max(2)
                * self.reference.trace.settings().illumination_arms.max(8),
        );
        let count = systems.len().div_ceil(limit);
        for (i, (systems, rows)) in systems.chunks(limit).zip(rows.chunks(limit)).enumerate() {
            values.extend(self.evaluate(
                systems,
                ops,
                template.as_ref().map(|t| (t.as_ref(), rows)),
                &ctx.subrange(
                    0.05 + 0.95 * i as f64 / count as f64,
                    0.05 + 0.95 * (i + 1) as f64 / count as f64,
                ),
            )?);
        }
        ctx.check()?;
        ctx.report(1., "parametric candidate batch complete");
        let mut values = values.into_iter();
        Ok(built
            .into_iter()
            .map(|s| s.map(|_| values.next().unwrap()))
            .collect())
    }
}

impl ResidentEvaluator {
    /// Independent glass or multistart refinements. Every supplied choice is
    /// refined; stable input order is retained and final scores come from f64.
    pub fn refine_candidates(
        &self,
        systems: &[OpticalSystem],
        variables: &[VariableSpec],
        spec: &MeritFunctionSpec,
        settings: &crate::lsq::LsqSettings,
        ctx: &JobContext,
    ) -> MkResult<Vec<crate::OptimResult>> {
        let mut results = Vec::with_capacity(systems.len());
        for chunk in systems.chunks(32) {
            results.extend(super::scheduler::run(
                self,
                chunk.len(),
                ctx,
                |i, worker| {
                    let problem = OpticalProblem::new(
                        chunk[i].clone(),
                        variables.to_vec(),
                        spec.clone(),
                        worker,
                        &OperandRegistry::builtin(),
                    )?;
                    let start = problem.start_point()?;
                    crate::discrete::refine_one(
                        &problem,
                        &start,
                        settings,
                        ctx,
                        crate::monitor::DEFAULT_FEASIBILITY_TOL,
                    )
                },
            )?);
        }
        // This API publishes independent refinements, so score their final
        // designs once after every device solve has completed.
        for (system, result) in systems.iter().zip(&mut results) {
            ctx.check()?;
            let problem = OpticalProblem::new(
                system.clone(),
                variables.to_vec(),
                spec.clone(),
                &self.reference,
                &OperandRegistry::builtin(),
            )?;
            let e = crate::problem::evaluate_one(&problem, &result.x, ctx)?;
            result.merit = crate::problem::merit(problem.weights(), &e.residuals, e.status);
            result.violation =
                crate::problem::total_violation(problem.constraints(), &e.constraints, e.status);
        }
        ctx.check()?;
        Ok(results)
    }
}
