//! Resident candidate evaluation. Accuracy is checked by offline f64 regressions;
//! only unsupported work and device failures invoke the reference evaluator.
use super::{GpuContext, Lanes};
use crate::deriv::FdScheme;
use crate::merit::gpu_eval::resident::{self, RayDesign};
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
use mklight_trace::SequentialTracer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
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
/// How a [`ResidentEvaluator`] divides a candidate batch between the host
/// and the device (work-split study `solvers`, see [`super::Split`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalDesign {
    /// The fully resident P1/P4 adapter: candidate systems prepared on the
    /// device from a compiled template, first-order operands in their own
    /// device plan, every other host operand completed by the reference,
    /// and a candidate with any unresolved device ray reported whole to the
    /// reference. The library default; the engine selects its design.
    #[default]
    Resident,
    /// Host candidates (f64 tracers, chief seeds, model, paraxial and
    /// host-only ray operands) with one device plan per batch for the
    /// pupil-ray aiming, tracing and reductions; rays the device leaves
    /// unresolved are completed one by one on the host in f64.
    Hybrid,
    /// [`Self::Hybrid`] in sub-batches, pipelined: the host prepares
    /// sub-batch k + 1 and finishes k − 1 while the device traces k.
    Pipelined,
}
impl EvalDesign {
    fn ray_design(self, cap: Option<u32>, host_aim: bool) -> RayDesign {
        match (self, cap) {
            (EvalDesign::Resident, _) => RayDesign::RESIDENT,
            (_, Some(cap)) => RayDesign {
                host_aim,
                ..RayDesign::hybrid_capped(cap)
            },
            (_, None) => RayDesign {
                host_aim,
                ..RayDesign::HYBRID
            },
        }
    }
    fn hybrid(self) -> bool {
        self != EvalDesign::Resident
    }
}
/// Smallest pipelined sub-batch. A plan's device time hardly grows with its
/// size up to some ten thousand rays (it is the latency of the slowest
/// aiming lanes), so small sub-batches would only add plans.
const PIPELINE_MIN_CHUNK: usize = 8;
/// Candidates per pipelined sub-batch of `n`: three stages, of at least
/// [`PIPELINE_MIN_CHUNK`] candidates.
fn pipeline_chunk(n: usize) -> usize {
    n.div_ceil(3).max(PIPELINE_MIN_CHUNK)
}
/// The dedicated (host lane, feed lane) rayon pools of a
/// [`Lanes::Budget`]: `quarters` quarters of the global pool's threads (at
/// least one) for the host lane, the rest (at least one) for the feed
/// lane. Built once per budget.
fn lane_pools(quarters: u8) -> (&'static rayon::ThreadPool, &'static rayon::ThreadPool) {
    static POOLS: [std::sync::OnceLock<(rayon::ThreadPool, rayon::ThreadPool)>; 3] =
        [const { std::sync::OnceLock::new() }; 3];
    let q = quarters.clamp(1, 3);
    let (h, f) = POOLS[q as usize - 1].get_or_init(|| {
        let n = rayon::current_num_threads().max(2);
        let host = (n * q as usize / 4).clamp(1, n - 1);
        let pool = |threads: usize, name: &'static str| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |i| format!("{name}-{i}"))
                .build()
                .expect("lane pool")
        };
        (pool(host, "mkl-host-lane"), pool(n - host, "mkl-feed-lane"))
    });
    (h, f)
}
/// Smallest batch for which a hybrid evaluator that has not seen a batch
/// yet first traces one candidate alone to learn the slow rays: a large
/// cold batch would hold every aiming tile of its plan up for the slow
/// lanes (tiles run one after another, each as long as its slowest lane).
const PROBE_MIN_BATCH: usize = 4;

#[derive(Clone)]
pub struct ResidentEvaluator {
    pub context: GpuContext,
    pub(crate) reference: BuiltinEvaluator,
    mtf: MtfEvaluator,
    solvers: SolverEvaluator,
    template: Arc<Mutex<Option<Arc<Template>>>>,
    stats: Arc<Mutex<EvaluationStats>>,
    design: EvalDesign,
    /// Hybrid designs: rays the device recently left unresolved in a
    /// quarter or more of a batch's candidates. Later batches route them to
    /// the host before the plan, where they are traced while the device
    /// works, so the device's commands are not held up by their slow lanes.
    host_rays: Arc<Mutex<HostRays>>,
    /// Device aiming schedule cap of the hybrid designs (`None`: theirs).
    aim_cap: Option<u32>,
    /// CPU thread layout of the hybrid designs' sub-batches.
    lanes: Lanes,
    /// Hybrid designs: aim every pupil ray on the host, trace on the device.
    host_aim: bool,
}
/// Rays the hybrid designs route to the host, learned from the device's
/// aiming reports (unresolved or slow), and the batches seen so far. A ray
/// stays with the host for the evaluator's lifetime: the learned rays are
/// the merit's unaimable and slow rim rays, and one that has become easy
/// costs the host a few microseconds.
#[derive(Default)]
struct HostRays {
    batches: u64,
    rays: BTreeSet<resident::RayKey>,
    /// The device left so many rays to the host (more than
    /// [`HOST_ONLY_FRACTION`] of a candidate's) that the rest of the run is
    /// evaluated on the host alone, by the reference.
    host_only: bool,
}
impl HostRays {
    /// Learn `keys`; `rays` is a candidate's traced ray count.
    fn record(&mut self, keys: impl IntoIterator<Item = resident::RayKey>, rays: f64) {
        self.rays.extend(keys);
        self.batches += 1;
        if self.rays.len() as f64 > HOST_ONLY_FRACTION * rays {
            self.host_only = true;
        }
    }
}
/// Fraction of a candidate's rays the device may leave to the host before
/// a hybrid evaluator gives the rest of its run to the reference (on the
/// 35-surface ultra-wide benchmark lens the device aimed only 166 of 570).
const HOST_ONLY_FRACTION: f64 = 0.5;
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
            design: EvalDesign::default(),
            host_rays: Arc::default(),
            aim_cap: None,
            lanes: Lanes::DEFAULT,
            host_aim: false,
        }
    }
    /// The hybrid designs with every pupil ray aimed on the host (the
    /// candidates' f64 tracers, the reference's aims) and only traced and
    /// reduced on the device, or (`false`, the default) aimed on the device
    /// with host completion.
    pub fn with_host_aim(mut self, host_aim: bool) -> Self {
        self.host_aim = host_aim;
        self
    }
    /// Are the hybrid designs' pupil rays aimed on the host?
    pub fn host_aim(&self) -> bool {
        self.host_aim
    }
    /// The same evaluator with another CPU thread layout of the hybrid
    /// designs' sub-batches.
    pub fn with_lanes(mut self, lanes: Lanes) -> Self {
        self.lanes = lanes;
        self
    }
    /// The same evaluator with another host/device division.
    pub fn with_design(mut self, design: EvalDesign) -> Self {
        self.design = design;
        self
    }
    pub fn design(&self) -> EvalDesign {
        self.design
    }
    /// The hybrid designs with another device aiming schedule cap (Newton
    /// iterations per ray before the host completes it): a test and
    /// measurement knob.
    pub fn with_device_aim_cap(mut self, cap: u32) -> Self {
        self.aim_cap = Some(cap);
        self
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
        info.settings["evaluation_design"] = serde_json::to_value(self.design).unwrap();
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
    /// One batch through every route. `anchored`: the batch holds the probes
    /// and re-evaluated base point of finite differences (hybrid designs
    /// then complete a ray on the host for every candidate when any needed
    /// it).
    fn evaluate(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        template: Option<(&Template, &[Candidate])>,
        anchored: bool,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        ctx.check()?;
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let hybrid = self.design.hybrid();
        let started = std::time::Instant::now();
        // A finite-difference batch with host aiming: like the reference's
        // `evaluate_batch_near`, the base point (the batch's first
        // candidate) is evaluated on the host first and its solves seed
        // every candidate's tracer, so the probes' host solves (fields,
        // chief rays, automatic semi-diameters, pupil rays) start from the
        // base's and share its solve path, as the reference's do. (Device
        // aiming keeps unseeded host tracers: seeding them measured no host
        // saving on the 29-surface lens and adds the base evaluation.)
        let seeds = match (hybrid && anchored && self.host_aim, systems.first()) {
            (true, Some(base)) => {
                self.reference
                    .trace
                    .anchor_seeds(base, systems.len(), ops, &inner)?
            }
            _ => None,
        };
        let tracers: Vec<_> = systems
            .par_iter()
            .map(|system| {
                match &seeds {
                    Some(seeds) => self.reference.trace.tracer_near(system, seeds),
                    None => self.reference.trace.tracer(system),
                }
                .ok()
                .map(Arc::new)
            })
            .collect();
        if hybrid {
            let seconds = started.elapsed().as_secs_f64();
            self.context.observe(mklight_gpu::resident::PlanStats {
                wall_seconds: seconds,
                stages: vec![mklight_gpu::resident::StageStats {
                    name: "host tracers".into(),
                    wall_seconds: seconds,
                    ..Default::default()
                }],
                ..Default::default()
            });
        }
        // Hybrid designs evaluate the first-order, model, paraxial and
        // host-only ray operands from the same host candidates as the
        // device rays (one pass, route 0).
        let route_of = |kind: &OperandKind| match route_for(kind) {
            3 | 4 if hybrid => 0,
            r => r,
        };
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
                .filter(|(_, o)| route_of(&o.kind) == route)
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
                    0 if hybrid => {
                        self.split_rays(&context, &selected_tracers, &refs, anchored, &inner)
                    }
                    0 => resident::evaluate(
                        &context,
                        &self.reference.trace,
                        &selected_tracers,
                        &refs,
                        selected_template,
                        &RayDesign::RESIDENT,
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
                                _ if hybrid
                                    && route == 0
                                    && !gpu_supports(&ops[i].kind, true)
                                    && !matches!(
                                        ops[i].kind,
                                        OperandKind::RelativeIllumination
                                    ) =>
                                {
                                    // A host stage of the hybrid pass.
                                    result[row][i] = value;
                                    missing[row][col] = false;
                                    stats.host_operands += 1;
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
                        // Only usable values take part in a difference.
                        let usable = value.as_ref().is_ok_and(|v| v.is_finite());
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
                        completed[row][i] = usable;
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
    let (rays, wavefront) = estimated_rays(system, ops, ri_nodes);
    let budget = if wavefront { 196_608 } else { 786_432 };
    (budget / rays.max(1)).clamp(1, 1024)
}
/// Estimated device rays per candidate of `ops` (an upper bound of the
/// plan's probes), and whether the operands need OPD and fit channels.
fn estimated_rays(system: &OpticalSystem, ops: &[&Operand], ri_nodes: usize) -> (usize, bool) {
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
    (rays, wavefront)
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
        if self.design.hybrid() {
            // The ray stage splits the batch into plans itself.
            return self.evaluate(systems, ops, None, false, ctx);
        }
        let count = systems.len().div_ceil(limit);
        for (i, chunk) in systems.chunks(limit).enumerate() {
            values.extend(self.evaluate(
                chunk,
                ops,
                None,
                false,
                &ctx.subrange(i as f64 / count as f64, (i + 1) as f64 / count as f64),
            )?);
        }
        ctx.check()?;
        ctx.report(1., "candidate batch complete");
        Ok(values)
    }
    /// Hybrid designs evaluate a batch near an anchor (joined finite
    /// differences of the rendezvous scheduler) with one evaluator per ray
    /// across it; the resident design ignores the anchor.
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        if self.design.hybrid() {
            return self.evaluate(systems, ops, None, anchor.is_some(), ctx);
        }
        self.evaluate_batch(systems, ops, ctx)
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
        if self.design.hybrid() {
            // Host candidates: no device template; one pass over the batch.
            let systems: Vec<OpticalSystem> = built
                .iter()
                .filter_map(|s| s.as_ref().ok().cloned())
                .collect();
            let anchored = batch.candidates.anchor.is_some();
            let mut values = self
                .evaluate(&systems, ops, None, anchored, ctx)?
                .into_iter();
            ctx.check()?;
            ctx.report(1., "parametric candidate batch complete");
            return Ok(built
                .into_iter()
                .map(|s| s.map(|_| values.next().unwrap()))
                .collect());
        }
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
                false,
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

type Landing<'a> = (usize, resident::Traced<'a>);
/// What the stages of one hybrid batch share.
struct Stage<'s> {
    context: &'s GpuContext,
    cpu: &'s TraceEvaluator,
    ops: &'s [&'s Operand],
    design: &'s RayDesign,
    anchored: bool,
    ctx: &'s JobContext,
    /// Unresolved rays of the sub-batches finished on landing.
    unresolved: Mutex<Vec<Vec<resident::RayKey>>>,
}
/// A prepared sub-batch, its submitted plan and its rays (device and
/// host-routed).
type Fed<'a> = (
    resident::Prepared<'a>,
    mklight_gpu::Pending<mklight_gpu::resident::PlanResult>,
    usize,
);
/// `f` on `pool`, or on the calling thread.
fn on<R: Send>(pool: Option<&rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}
impl<'s> Stage<'s> {
    /// Prepare sub-batch `k` and submit its device plan.
    fn feed<'a>(
        &self,
        k: usize,
        chunk: &'a [Arc<SequentialTracer>],
        host_rays: &BTreeSet<resident::RayKey>,
    ) -> MkResult<Fed<'a>>
    where
        's: 'a,
    {
        self.ctx.check()?;
        let mut prepared = resident::prepare(
            self.context,
            self.cpu,
            chunk,
            self.ops,
            None,
            self.design,
            host_rays,
            self.ctx,
        )?;
        let plan = prepared.plan.take().expect("prepared plan");
        self.context.record_traced_rays(prepared.traced_rays());
        let rays = prepared.traced_rays() + prepared.host_rays();
        if resident::trace_routing() {
            eprintln!(
                "MKL_SOLVERS chunk {k}: {} candidates, {} device rays, {} host-routed rays (set {})",
                chunk.len(),
                prepared.traced_rays(),
                prepared.host_rays(),
                host_rays.len()
            );
        }
        let pending = self.context.submit_async(plan, self.ctx)?;
        Ok((prepared, pending, rays))
    }
    /// The host's share of a sub-batch while the device traces the rest:
    /// the rays routed to it and the operands that need no ray.
    fn host_share(
        &self,
        chunk: &[Arc<SequentialTracer>],
        prepared: &mut resident::Prepared<'_>,
    ) -> MkResult<()> {
        let started = std::time::Instant::now();
        resident::host_stage(prepared, self.cpu, chunk, self.ops, self.design, self.ctx)?;
        self.context.observe(mklight_gpu::resident::PlanStats {
            polished_rays: prepared.host_rays() as u64,
            stages: vec![mklight_gpu::resident::StageStats {
                name: "host stage (routed rays, rayless operands; overlapped)".into(),
                wall_seconds: started.elapsed().as_secs_f64(),
                ..Default::default()
            }],
            ..Default::default()
        });
        Ok(())
    }
    /// The sub-batches `chunks` in two concurrent lanes ([`Lanes::Shared`]
    /// or [`Lanes::Budget`]): this thread feeds the device (preparation
    /// and submission), a second one takes each submitted sub-batch's host
    /// share and lands it; at most one fed sub-batch waits between them.
    /// Returns the batch's rays. Called outside the rayon pool only, so
    /// that the lanes' blocking hand-over holds no pool thread.
    fn lanes<'a>(
        &self,
        lanes: Lanes,
        chunks: &[&'a [Arc<SequentialTracer>]],
        host_rays: &BTreeSet<resident::RayKey>,
        values: &mut [Option<Vec<Vec<OperandValue>>>],
        deferred: &mut Vec<Landing<'a>>,
    ) -> MkResult<usize>
    where
        's: 'a,
    {
        let (host_pool, feed_pool) = match lanes {
            Lanes::Budget { quarters } => {
                let (h, f) = lane_pools(quarters);
                (Some(h), Some(f))
            }
            _ => (None, None),
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Fed<'a>)>(1);
        std::thread::scope(|scope| {
            let host = scope.spawn(|| -> MkResult<()> {
                for (k, (mut prepared, pending, _)) in rx {
                    on(host_pool, || {
                        self.host_share(chunks[k], &mut prepared)?;
                        self.land(k, chunks[k], prepared, pending, values, deferred)
                    })?;
                }
                Ok(())
            });
            let mut rays = 0;
            let mut fed = Ok(());
            for (k, chunk) in chunks.iter().enumerate() {
                match on(feed_pool, || self.feed(k, chunk, host_rays)) {
                    Ok(item) => {
                        rays += item.2;
                        if tx.send((k, item)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        fed = Err(e);
                        break;
                    }
                }
            }
            drop(tx);
            let hosted = host.join().expect("host lane");
            fed.and(hosted).map(|()| rays)
        })
    }
    /// Wait for sub-batch `k`'s plan, read it and finish it (or, in an
    /// anchored batch, keep it for the uniform completion).
    fn land<'a>(
        &self,
        k: usize,
        chunk: &[Arc<SequentialTracer>],
        prepared: resident::Prepared<'a>,
        pending: mklight_gpu::Pending<mklight_gpu::resident::PlanResult>,
        values: &mut [Option<Vec<Vec<OperandValue>>>],
        deferred: &mut Vec<Landing<'a>>,
    ) -> MkResult<()> {
        let wait = std::time::Instant::now();
        let result = self.context.complete(pending)?;
        let waited = wait.elapsed().as_secs_f64();
        self.context.observe(mklight_gpu::resident::PlanStats {
            stages: vec![mklight_gpu::resident::StageStats {
                name: "host wait for device".into(),
                wall_seconds: waited,
                ..Default::default()
            }],
            ..Default::default()
        });
        let traced = resident::collect(self.context, prepared, result, self.design)?;
        if self.anchored {
            deferred.push((k, traced));
        } else {
            self.unresolved.lock().unwrap().extend(
                traced
                    .unresolved
                    .iter()
                    .zip(&traced.slow)
                    .map(|(u, s)| u.iter().chain(s).copied().collect()),
            );
            values[k] = Some(resident::finish(
                self.context,
                self.cpu,
                chunk,
                self.ops,
                traced,
                None,
                self.design,
                self.ctx,
            )?);
        }
        Ok(())
    }
}
impl ResidentEvaluator {
    /// The ray stage of the hybrid designs: host preparation, one device
    /// plan per sub-batch (aim, trace, reductions), host completion of the
    /// unresolved rays and every operand value from the host candidates.
    /// [`EvalDesign::Pipelined`] overlaps the host preparation of the next
    /// sub-batch and the finishing of the previous one with the device's
    /// work on the current one. In an `anchored` batch every sub-batch is
    /// finished after all are traced, so that a ray completed on the host
    /// for one candidate is completed for all.
    fn split_rays(
        &self,
        context: &GpuContext,
        tracers: &[Arc<SequentialTracer>],
        ops: &[&Operand],
        anchored: bool,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let n = tracers.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n >= PROBE_MIN_BATCH && !self.host_aim && self.host_rays.lock().unwrap().batches == 0 {
            // The probe's values are not used: a finite difference routes
            // all its candidates alike.
            self.split_rays(context, &tracers[..1], ops, false, ctx)?;
        }
        let cpu = &self.reference.trace;
        if self.host_rays.lock().unwrap().host_only {
            // The reference, seeded from the batch's base point when it holds
            // finite differences (its first candidate).
            let systems: Vec<OpticalSystem> = tracers
                .iter()
                .map(|t| t.system().system().clone())
                .collect();
            let anchor = anchored.then(|| systems[0].clone());
            let values = cpu.evaluate_batch_near(&systems, ops, anchor.as_ref(), ctx)?;
            let mut stats = self.stats.lock().unwrap();
            *stats
                .fallbacks
                .entry("device aiming left most rays to the host: reference evaluation".into())
                .or_default() += (n * ops.len()) as u64;
            return Ok(values);
        }
        let design = self.design.ray_design(self.aim_cap, self.host_aim);
        let limit = batch_limit(
            tracers[0].system().system(),
            ops,
            1 + cpu.settings().illumination_rings.max(2) * cpu.settings().illumination_arms.max(8),
        );
        let overlap = self.design == EvalDesign::Pipelined;
        let size = if overlap {
            pipeline_chunk(n).min(limit)
        } else {
            limit
        };
        let chunks: Vec<&[Arc<SequentialTracer>]> = tracers.chunks(size).collect();
        let mut values: Vec<Option<Vec<Vec<OperandValue>>>> = chunks.iter().map(|_| None).collect();
        let mut deferred: Vec<Landing<'_>> = Vec::new();
        let stage = Stage {
            context,
            cpu,
            ops,
            design: &design,
            anchored,
            ctx,
            unresolved: Mutex::default(),
        };
        // One routing for the whole batch, so that a ray is traced by the
        // same side for every candidate of a finite difference.
        let host_rays = self.host_rays.lock().unwrap().rays.clone();
        let mut unresolved: BTreeMap<resident::RayKey, usize> = BTreeMap::new();
        let mut rays_total = 0usize;
        // Two lanes only outside the rayon pool (a nested evaluation, as in
        // parallel glass refinements, keeps one).
        let lanes = match self.lanes {
            Lanes::Single => None,
            l if chunks.len() > 1 && rayon::current_thread_index().is_none() => Some(l),
            _ => None,
        };
        if let Some(lanes) = lanes {
            rays_total = stage.lanes(lanes, &chunks, &host_rays, &mut values, &mut deferred)?;
        } else {
            let mut in_flight = None;
            for (k, chunk) in chunks.iter().enumerate() {
                let (mut prepared, pending, rays) = stage.feed(k, chunk, &host_rays)?;
                rays_total += rays;
                stage.host_share(chunk, &mut prepared)?;
                if !overlap {
                    stage.land(k, chunk, prepared, pending, &mut values, &mut deferred)?;
                    continue;
                }
                if let Some((j, prev, p)) = in_flight.replace((k, prepared, pending)) {
                    stage.land(j, chunks[j], prev, p, &mut values, &mut deferred)?;
                }
            }
            if let Some((j, prev, p)) = in_flight.take() {
                stage.land(j, chunks[j], prev, p, &mut values, &mut deferred)?;
            }
        }
        for (_, t) in &deferred {
            for (u, slow) in t.unresolved.iter().zip(&t.slow) {
                for &key in u.iter().chain(slow) {
                    *unresolved.entry(key).or_default() += 1;
                }
            }
        }
        for keys in stage.unresolved.lock().unwrap().drain(..) {
            for key in keys {
                *unresolved.entry(key).or_default() += 1;
            }
        }
        // A ray slow or unresolved in an eighth of the batch's candidates
        // (any, in small batches) goes to the host for the next batches.
        let threshold = n.div_ceil(8).max(1);
        self.host_rays.lock().unwrap().record(
            unresolved
                .iter()
                .filter(|&(_, &count)| count >= threshold)
                .map(|(&key, _)| key),
            rays_total as f64 / n as f64,
        );
        if anchored {
            let uniform: BTreeSet<resident::RayKey> = deferred
                .iter()
                .flat_map(|(_, t)| t.unresolved.iter().flatten().copied())
                .collect();
            for (k, traced) in deferred {
                values[k] = Some(resident::finish(
                    context,
                    cpu,
                    chunks[k],
                    ops,
                    traced,
                    Some(&uniform),
                    &design,
                    ctx,
                )?);
            }
        }
        Ok(values
            .into_iter()
            .flat_map(|v| v.expect("finished"))
            .collect())
    }

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
