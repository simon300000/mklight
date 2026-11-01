//! Explicit resident evaluator with per-candidate f64 qualification. No nominal
//! envelope is assumed: every output is compared with the unchanged reference.
use super::GpuContext;
use crate::deriv::FdScheme;
use crate::merit::{
    gpu_eval::{gpu_fd_relative_step, gpu_supports},
    mtf_eval::MtfEvaluator,
    solver_eval::SolverEvaluator,
    *,
};
use mklight_core::{
    JobContext, MkError, MkResult, OpticalSystem, RunInfo, Snapshot, param::ParamPath,
};
use mklight_gpu::systems::{Candidate, CompiledSystem, Parameter, UploadedSystem};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvaluationStats {
    pub candidates: u64,
    pub device_candidates: u64,
    pub cpu_candidates: u64,
    pub device_operands: u64,
    pub host_operands: u64,
    pub reference_checks: u64,
    pub reference_seconds: f64,
    pub reference_cache_hits: u64,
    pub geometry_qualifications: u64,
    pub geometry_rejections: u64,
    pub geometry_report: Option<mklight_gpu::xval::XvalReport>,
    pub fallback_operands: u64,
    pub fallbacks: BTreeMap<String, u64>,
    pub maximum_absolute_error: f64,
}
struct ReferenceEntry {
    system: OpticalSystem,
    values: Vec<OperandValue>,
    geometry: bool,
    tracer: Option<Arc<mklight_trace::SequentialTracer>>,
}
#[derive(Default)]
struct ReferenceCache {
    ops: Vec<Operand>,
    backend: Option<Arc<dyn mklight_gpu::ComputeBackend>>,
    entries: BTreeMap<u64, ReferenceEntry>,
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
struct CheckedReference {
    evaluator: BuiltinEvaluator,
    cache: Arc<Mutex<ReferenceCache>>,
}
impl OperandEvaluator for CheckedReference {
    fn supports(&self, kind: &OperandKind) -> bool {
        self.evaluator.supports(kind)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        self.evaluator.fd_relative_step(kind, scheme)
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        ctx.check()?;
        let cached: Option<Vec<_>> = {
            let cache = self.cache.lock().unwrap();
            if cache.ops.iter().eq(ops.iter().copied()) {
                systems
                    .iter()
                    .map(|system| {
                        cache
                            .entries
                            .values()
                            .find(|entry| entry.system == *system)
                            .map(|entry| entry.values.clone())
                    })
                    .collect()
            } else {
                None
            }
        };
        if let Some(values) = cached {
            return Ok(values);
        }
        self.evaluator.evaluate_batch(systems, ops, ctx)
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
    references: Arc<Mutex<ReferenceCache>>,
    checked_reference: CheckedReference,
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
        context.identity.qualification = Some(
            serde_json::json!({"mode":"per_candidate_geometry_and_full_operand_f64","publication":"after_reference_check"}),
        );
        let references = Arc::default();
        let checked_reference = CheckedReference {
            evaluator: reference.clone(),
            cache: Arc::clone(&references),
        };
        Self {
            mtf: reference.mtf.clone().with_gpu(context.clone()),
            solvers: reference.solvers.clone().with_gpu(context.clone()),
            context,
            reference,
            template: Arc::default(),
            stats: Arc::default(),
            references,
            checked_reference,
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
        info.settings["candidate_qualification"] = serde_json::json!({"version":1,"mode":"every_candidate_geometry_and_full_operand_f64","envelope":null,"stats":self.stats()});
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
        first: usize,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        ctx.check()?;
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let t = Instant::now();
        let keys: Vec<_> = (0..systems.len())
            .map(|i| template.map_or((first + i) as u64, |(_, rows)| rows[i].id))
            .collect();
        let mut reference = vec![None; systems.len()];
        let mut geometry = vec![false; systems.len()];
        let mut tracers = vec![None; systems.len()];
        let mut geometry_report = None;
        {
            let mut cache = self.references.lock().unwrap();
            if cache.ops.iter().ne(ops.iter().copied())
                || cache
                    .backend
                    .as_ref()
                    .is_none_or(|b| !Arc::ptr_eq(b, &self.context.backend))
            {
                cache.ops = ops.iter().map(|o| (*o).clone()).collect();
                cache.backend = Some(self.context.backend.clone());
                cache.entries.clear();
            }
            for (i, system) in systems.iter().enumerate() {
                if let Some(entry) = cache.entries.get(&keys[i]).filter(|e| e.system == *system) {
                    reference[i] = Some(entry.values.clone());
                    geometry[i] = entry.geometry;
                    tracers[i] = entry.tracer.clone();
                }
            }
        }
        let missing: Vec<_> = reference
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_none())
            .map(|(i, _)| i)
            .collect();
        if !missing.is_empty() {
            let pending: Vec<_> = missing.iter().map(|&i| systems[i].clone()).collect();
            let values = self.reference.evaluate_batch(&pending, ops, &inner)?;
            let prepared: Vec<_> = pending
                .par_iter()
                .map(|system| self.reference.trace.tracer(system).ok().map(Arc::new))
                .collect();
            let settings = mklight_gpu::xval::XvalSettings {
                clip: self.reference.trace.settings().clip,
                radiometry: mklight_gpu::Radiometry::Off,
                ..mklight_gpu::xval::XvalSettings::qualification()
            };
            let backend = super::qualification::QualificationBackend::new(&self.context);
            let validation_tracers: Vec<_> = prepared.iter().flatten().map(AsRef::as_ref).collect();
            let qualifications =
                mklight_gpu::xval::qualify_many(&backend, &validation_tracers, &settings, &inner)?;
            let mut qualifications = qualifications.into_iter();
            let qualified: Vec<_> = prepared
                .into_iter()
                .map(|tracer| match tracer {
                    Some(t) => {
                        let q = qualifications.next().unwrap();
                        (q.passed, Some(t), q.report)
                    }
                    None => (false, None, None),
                })
                .collect();
            let mut cache = self.references.lock().unwrap();
            // Another concurrent job may have changed the operand specification.
            let same = cache.ops.iter().eq(ops.iter().copied())
                && cache
                    .backend
                    .as_ref()
                    .is_some_and(|b| Arc::ptr_eq(b, &self.context.backend));
            for ((&i, value), (qualified, tracer, report)) in
                missing.iter().zip(values).zip(qualified)
            {
                if let Some(report) = report {
                    merge_geometry(&mut geometry_report, &report);
                }
                if same {
                    cache.entries.insert(
                        keys[i],
                        ReferenceEntry {
                            system: systems[i].clone(),
                            values: value.clone(),
                            geometry: qualified,
                            tracer: tracer.clone(),
                        },
                    );
                }
                reference[i] = Some(value);
                geometry[i] = qualified;
                tracers[i] = tracer;
            }
            let limit = (1_048_576 / ops.len().max(1)).clamp(1, 1024);
            while cache.entries.len() > limit {
                cache.entries.pop_first();
            }
        }
        let reference: Vec<_> = reference.into_iter().map(Option::unwrap).collect();
        let ref_seconds = t.elapsed().as_secs_f64();
        ctx.report(0.5, "candidate reference qualification");
        let mut result = reference.clone();
        let mut device_rows = vec![false; systems.len()];
        let mut fallback_rows: Vec<_> = geometry.iter().map(|q| !q).collect();
        let mut stats = EvaluationStats {
            candidates: systems.len() as u64,
            reference_checks: missing.len() as u64,
            geometry_qualifications: missing.len() as u64,
            geometry_report,
            geometry_rejections: missing.iter().filter(|&&i| !geometry[i]).count() as u64,
            reference_cache_hits: (systems.len() - missing.len()) as u64,
            reference_seconds: ref_seconds,
            ..Default::default()
        };
        let routes: Vec<Vec<usize>> = (0..5)
            .map(|route| {
                ops.iter()
                    .enumerate()
                    .filter(|(_, o)| route_for(&o.kind) == route)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        for (route, indices) in routes.into_iter().enumerate() {
            ctx.report(0.5 + 0.1 * route as f64, "resident candidate operands");
            if indices.is_empty() {
                continue;
            }
            let refs: Vec<_> = indices.iter().map(|&i| ops[i]).collect();
            let active: Vec<_> = geometry
                .iter()
                .enumerate()
                .filter(|(_, q)| **q)
                .map(|(i, _)| i)
                .collect();
            // Only the native MTF/solver adapters take owned systems.
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
            let attempt = if active.is_empty() && route != 3 {
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
                    _ => {
                        stats.host_operands += (systems.len() * indices.len()) as u64;
                        for &i in &indices {
                            if !matches!(
                                ops[i].kind.capability(),
                                Capability::Model | Capability::Paraxial
                            ) {
                                fallback_rows.fill(true);
                                stats.fallback_operands += systems.len() as u64;
                                *stats
                                    .fallbacks
                                    .entry(format!(
                                        "{}: no resident operand adapter",
                                        ops[i].kind.name()
                                    ))
                                    .or_default() += systems.len() as u64;
                            }
                        }
                        continue;
                    }
                }
            };
            ctx.check()?;
            match attempt {
                Err(MkError::Cancelled) => return Err(MkError::Cancelled),
                Err(e) => {
                    fallback_rows.fill(true);
                    let n = (systems.len() * indices.len()) as u64;
                    stats.fallback_operands += n;
                    *stats.fallbacks.entry(e.to_string()).or_default() += n;
                }
                Ok(values) => {
                    stats.fallback_operands +=
                        ((systems.len() - active.len()) * indices.len()) as u64;
                    if active.len() != systems.len() {
                        *stats
                            .fallbacks
                            .entry("candidate geometry qualification rejected".into())
                            .or_default() +=
                            ((systems.len() - active.len()) * indices.len()) as u64;
                    }
                    for (row, values) in active.iter().copied().zip(values) {
                        for (col, value) in values.into_iter().enumerate() {
                            let i = indices[col];
                            let (pass, error) = qualifies(
                                ops[i],
                                &value,
                                &reference[row][i],
                                ops.len() * systems[row].configurations.len().max(1),
                            );
                            stats.maximum_absolute_error = stats.maximum_absolute_error.max(error);
                            if pass && geometry[row] {
                                result[row][i] = value;
                                if route == 2 && self.context.native_device_calls() == native_before
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
                                }
                            } else {
                                fallback_rows[row] = true;
                                stats.fallback_operands += 1;
                                *stats
                                    .fallbacks
                                    .entry(format!(
                                        "{}: candidate output or constraint uncertainty",
                                        ops[i].kind.name()
                                    ))
                                    .or_default() += 1;
                            }
                        }
                    }
                }
            }
        }
        stats.device_candidates = device_rows.iter().filter(|&&v| v).count() as u64;
        stats.cpu_candidates = device_rows
            .iter()
            .zip(&fallback_rows)
            .filter(|(device, fallback)| !**device || **fallback)
            .count() as u64;
        let mut total = self.stats.lock().unwrap();
        total.device_candidates += stats.device_candidates;
        total.cpu_candidates += stats.cpu_candidates;
        total.candidates += stats.candidates;
        total.device_operands += stats.device_operands;
        total.host_operands += stats.host_operands;
        total.reference_checks += stats.reference_checks;
        total.reference_cache_hits += stats.reference_cache_hits;
        total.geometry_qualifications += stats.geometry_qualifications;
        total.geometry_rejections += stats.geometry_rejections;
        if let Some(report) = &stats.geometry_report {
            merge_geometry(&mut total.geometry_report, report);
        }
        total.reference_seconds += stats.reference_seconds;
        total.fallback_operands += stats.fallback_operands;
        total.maximum_absolute_error = total
            .maximum_absolute_error
            .max(stats.maximum_absolute_error);
        for (k, v) in stats.fallbacks {
            *total.fallbacks.entry(k).or_default() += v;
        }
        ctx.check()?;
        ctx.report(1., "candidate batch qualified");
        Ok(result)
    }
}
fn merge_geometry(
    total: &mut Option<mklight_gpu::xval::XvalReport>,
    r: &mklight_gpu::xval::XvalReport,
) {
    if let Some(t) = total {
        t.rays += r.rays;
        t.compared += r.compared;
        t.opd_compared += r.opd_compared;
        t.max_position_mm = t.max_position_mm.max(r.max_position_mm);
        t.max_direction = t.max_direction.max(r.max_direction);
        t.max_opd_waves = t.max_opd_waves.max(r.max_opd_waves);
        t.max_transmittance_abs = t.max_transmittance_abs.max(r.max_transmittance_abs);
        t.survival_mismatches += r.survival_mismatches;
        t.band_mismatches += r.band_mismatches;
        t.chief_mismatches += r.chief_mismatches;
        t.notes.extend(r.notes.iter().cloned());
        t.notes.truncate(8);
    } else {
        *total = Some(r.clone());
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
fn qualifies(
    op: &Operand,
    value: &OperandValue,
    reference: &OperandValue,
    terms: usize,
) -> (bool, f64) {
    let (Ok(a), Ok(b)) = (value, reference) else {
        return (false, 0.);
    };
    if !a.is_finite() || !b.is_finite() {
        return (false, 0.);
    }
    let error = (a - b).abs();
    let budget = match op.kind {
        OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. } => 0.005 / 3.,
        OperandKind::Mtf { .. } => 0.002 / 3.,
        OperandKind::RelativeIllumination => 1e-3 / 3.,
        OperandKind::RmsSpot { .. } => (0.005 * b.abs()).max(2e-5) / 3.,
        OperandKind::GhostFocus { .. }
        | OperandKind::GhostIrradiance { .. }
        | OperandKind::StrayLightPower { .. } => 1e-5 * b.abs() + 1e-10,
        _ => 2e-5 / 3.,
    };
    // Every merit residual must also pass the final-merit gate. Ambiguous
    // inequality decisions use the reference value, even if optics passed.
    let near = (op.comparison != Comparison::Equal || op.role == OperandRole::Constraint)
        && (b - op.target).abs() <= budget;
    let ar = op.residual(*a);
    let br = op.residual(*b);
    (
        error <= budget
            // RI's numeric completion is f64. A changed visibility decision
            // must not introduce an unchecked finite-difference discontinuity.
            && (!matches!(op.kind,OperandKind::RelativeIllumination) || error==0.)
            && !near
            && (op.weight * (ar * ar - br * br)).abs()
                <= (0.005 * op.weight * br * br).max(0.5e-8 / terms.max(1) as f64),
        error,
    )
}
impl OperandEvaluator for ResidentEvaluator {
    fn reference_evaluator(&self) -> Option<&dyn OperandEvaluator> {
        Some(&self.checked_reference)
    }
    fn gpu_context(&self) -> Option<&GpuContext> {
        Some(&self.context)
    }
    fn supports(&self, kind: &OperandKind) -> bool {
        self.reference.supports(kind)
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
                i * limit,
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
                0,
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
        ctx.check()?;
        Ok(results)
    }
}
