//! Resident candidate evaluation. Accuracy is checked by offline f64 regressions;
//! only unsupported work and device failures invoke the reference evaluator.
use super::GpuContext;
use crate::deriv::FdScheme;
use crate::merit::gpu_eval::resident::{self, RayDesign, SCHEDULE_CAP};
use crate::merit::{
    gpu_eval::gpu_supports, mtf_eval::MtfEvaluator, solver_eval::SolverEvaluator, *,
};
use mklight_core::{
    JobContext, MkError, MkResult, OpticalSystem, RayAiming, RunInfo, SurfaceShape,
};
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
    /// Batches whose pupil rays the device aimed.
    pub device_aimed_batches: u64,
    pub fallbacks: BTreeMap<String, u64>,
}
/// Batch [`aiming_work`] from which the device aims the pupil rays (with
/// host completion) instead of the host. Device aiming pays only where host
/// aiming is costly: many rays through surfaces without a closed-form hit.
/// Measured crossover (Monte Carlo batches of 128): host aiming faster at
/// 1.5e6 (2 aspheres, 5 × 10 pupil sampling, 0.67× the device's time),
/// device aiming faster from 2.2e6 (the same lens at 6 × 12: 0.84× the
/// host's; 8 aspheres at default sampling: 0.90×); spherical lenses never
/// gained (host aiming 0.96× the device's on 1000 trials at 6 × 12), nor
/// finite-difference Jacobians (6e5 on the 8-asphere lens: host 1.2× faster).
pub const DEVICE_AIM_FROM: usize = 2_000_000;
/// Smallest batch the evaluator pipelines in sub-batches (the host
/// prepares sub-batch k + 1 and finishes k − 1 while the device traces k):
/// Monte Carlo and sensitivity batches gain (1000 dense trials on the 29-
/// and 35-surface benchmark lenses 1.92 → 1.72 s and 1.98 → 1.75 s), a
/// 23-candidate finite-difference Jacobian of the Double Gauss loses (792
/// vs 582 ms), so Jacobians stay one plan.
pub const PIPELINE_FROM: usize = 64;
/// Smallest pipelined sub-batch. A plan's device time hardly grows with its
/// size up to some ten thousand rays (it is the latency of the slowest
/// aiming lanes), so small sub-batches would only add plans.
const PIPELINE_MIN_CHUNK: usize = 8;
/// Candidates per pipelined sub-batch of `n`: three stages, of at least
/// [`PIPELINE_MIN_CHUNK`] candidates.
fn pipeline_chunk(n: usize) -> usize {
    n.div_ceil(3).max(PIPELINE_MIN_CHUNK)
}
/// Relative forward-difference step of the device's ray operands. Device
/// values are double-float traces (rounding noise ~1e-12 relative, against
/// ~1e-16 on the f64 reference): with the reference's `√ε` the noise term
/// gave Jacobian column errors of 2e-6–5e-5 on the samples, at `1e-6`
/// 3e-8–2e-6 (as close to f64 central differences as the reference's own
/// forward differences). A step of `1e-3` costs up to 1.5e-3 of
/// truncation and, on a vignetted lens, lets a probe cross the vignetting
/// edge of a ray that fails at the base point: a penalty jump divided by
/// the step, one spurious column, and a least-squares run that stalls
/// (`docs/OPTIMIZATION.md`).
pub const DEVICE_FD_RELATIVE_STEP_FORWARD: f64 = 1e-6;
/// [`DEVICE_FD_RELATIVE_STEP_FORWARD`] of the wavefront operands (OPD
/// adds the path-length noise; the reference takes `1e-6`): column errors
/// 5e-7–2e-5 on the samples against 4e-6–1e-4 at `1e-6` and up to 2e-4 at
/// `1e-4`.
pub const DEVICE_WAVEFRONT_FD_RELATIVE_STEP_FORWARD: f64 = 1e-5;

/// Smallest batch for which a solve that has not yet aimed a batch on
/// the device first traces its first `PROBE_MIN_BATCH − 1` candidates
/// alone to learn the slow rays: a large cold batch would hold every aiming
/// tile of its plan up for the slow lanes (tiles run one after another,
/// each as long as its slowest lane). Under the aiming rule the probe
/// candidates are perturbed designs (Monte Carlo trials), so one learned
/// 18 of the 34 rays the batches route on the 35-surface zoom (+0.45 s of
/// device time for 1000 dense trials); three learn all 34.
const PROBE_MIN_BATCH: usize = 4;

/// Evaluates candidate batches on a device. An evaluator serves one solve:
/// a sequence of batches evaluated one after another (a least-squares run,
/// a population search, the batches of a tolerance run). Its clones serve
/// the same solve; [`ResidentEvaluator::for_solve`] gives the evaluator of
/// another one.
#[derive(Clone)]
pub struct ResidentEvaluator {
    pub context: GpuContext,
    pub(crate) reference: BuiltinEvaluator,
    mtf: MtfEvaluator,
    solvers: SolverEvaluator,
    stats: Arc<Mutex<EvaluationStats>>,
    /// What the solve has learned from its device-aimed batches. Shared by
    /// clones (the same solve), never by [`Self::for_solve`].
    routing: Arc<Mutex<Routing>>,
    /// Device aiming schedule cap (`None`: [`SCHEDULE_CAP`]).
    aim_cap: Option<u32>,
    /// Aim a batch's pupil rays on the device (with host completion) when
    /// its [`aiming_work`] reaches this (`Some(0)`: every batch), else on
    /// the host, which the device then only traces (`None`: every batch).
    device_aim_from: Option<usize>,
}
/// The routing state of one solve, learned from the aiming reports of its
/// device-aimed batches. Batches of one solve are evaluated one after
/// another, so the state a batch sees depends only on the batches before it.
#[derive(Default)]
struct Routing {
    /// Has a device-aimed batch reported its rays?
    learned: bool,
    /// Pupil rays the device left unresolved, or resolved slowly, in an
    /// eighth or more of a batch's candidates. Later device-aimed batches
    /// route them to the host before the plan, where they are traced while
    /// the device works, so the device's commands are not held up by their
    /// slow lanes. A ray stays with the host for the rest of the solve: the
    /// learned rays are the merit's unaimable and slow rim rays, and one
    /// that has become easy costs the host a few microseconds.
    rays: BTreeSet<resident::RayKey>,
    /// A device-aimed batch left so many of its rays to the host (more than
    /// [`HOST_ONLY_FRACTION`] of a candidate's) that the rest of the solve
    /// is evaluated on the host alone, by the reference.
    host_only: bool,
}
impl Routing {
    /// Learn the rays a device-aimed batch reported (`keys`: unresolved or
    /// slow in enough of its candidates). Of the `rays` of one of its
    /// candidates, `routed` were already routed to the host; with the ones
    /// learned now, that is what the batch leaves to the host.
    fn learn(&mut self, keys: impl IntoIterator<Item = resident::RayKey>, routed: f64, rays: f64) {
        let new = keys.into_iter().filter(|&k| self.rays.insert(k)).count();
        self.learned = true;
        if routed + new as f64 > HOST_ONLY_FRACTION * rays {
            self.host_only = true;
        }
    }
}
/// Fraction of a candidate's rays a device-aimed batch may leave to the
/// host before the evaluator gives the rest of its solve to the reference
/// (on the 35-surface ultra-wide benchmark lens the device aimed only 166
/// of 570).
const HOST_ONLY_FRACTION: f64 = 0.5;
/// The fallback counted for every value of a batch evaluated by the
/// reference after [`Routing::host_only`].
const HOST_ONLY_FALLBACK: &str = "device aiming left most rays to the host: reference evaluation";
/// The ray pass's values of a batch and who computed them.
enum RayPass {
    /// The host/device division.
    Split(Vec<Vec<OperandValue>>),
    /// The reference alone ([`Routing::host_only`]).
    Reference(Vec<Vec<OperandValue>>),
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
            stats: Arc::default(),
            routing: Arc::default(),
            aim_cap: None,
            device_aim_from: Some(DEVICE_AIM_FROM),
        }
    }
    /// This evaluator for another solve (another problem, or a solve that
    /// runs beside this one): the same device context, reference, settings
    /// and statistics, with a routing state of its own. What one solve
    /// learns from its batches then never routes another's rays, and solves
    /// that run in parallel do not depend on each other's timing.
    pub fn for_solve(&self) -> Self {
        Self {
            routing: Arc::default(),
            ..self.clone()
        }
    }
    /// The evaluator with every pupil ray aimed on the host (the
    /// candidates' f64 tracers, the reference's aims) and only traced and
    /// reduced on the device, or (`false`) aimed on the device with host
    /// completion, instead of the default rule ([`DEVICE_AIM_FROM`]).
    pub fn with_host_aim(mut self, host_aim: bool) -> Self {
        self.device_aim_from = (!host_aim).then_some(0);
        self
    }
    /// The aiming rule: device aiming from this [`aiming_work`] per batch.
    pub fn device_aim_from(&self) -> Option<usize> {
        self.device_aim_from
    }
    /// Does the device aim the pupil rays of a batch of `n` candidates like
    /// `system` for `ops`?
    pub fn aims_on_device(&self, system: &OpticalSystem, n: usize, ops: &[&Operand]) -> bool {
        let ri_nodes = ri_nodes(&self.reference.trace);
        self.device_aim_from
            .is_some_and(|from| aiming_work(system, ops, ri_nodes).saturating_mul(n) >= from)
    }
    /// The evaluator with another device aiming schedule cap (device aiming
    /// only; Newton iterations per ray before the host completes it): a
    /// test and measurement knob.
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
        if self.stats().device_operands > 0 {
            info.backend = self.context.backend.info().backend_id();
            info.precision = mklight_core::job::Precision::F32Gpu;
        }
    }
    /// The ray stage's division: host aiming, or device aiming with host
    /// completion (capped at `aim_cap` iterations per ray when set).
    fn ray_design(&self, host_aim: bool) -> RayDesign {
        RayDesign {
            host_aim,
            schedule_cap: self.aim_cap.unwrap_or(SCHEDULE_CAP).max(1),
        }
    }
    /// One batch through every route. `anchored`: the batch holds the probes
    /// and re-evaluated base point of finite differences (a ray is then
    /// completed on the host for every candidate when any needed it), near
    /// `anchor` (`None` when the base point cannot be built).
    fn evaluate(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        anchored: bool,
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        ctx.check()?;
        let inner = JobContext::with(ctx.cancel.clone(), Default::default());
        let started = std::time::Instant::now();
        // A finite-difference batch with host aiming: like the reference's
        // `evaluate_batch_near`, the anchor (the base point) is evaluated on
        // the host first and its solves seed every candidate's tracer, so
        // the probes' host solves (fields, chief rays, automatic
        // semi-diameters, pupil rays) start from the base's and share its
        // solve path, as the reference's do. (Device aiming keeps unseeded
        // host tracers: seeding them measured no host saving on the
        // 29-surface lens and adds the base evaluation.)
        let host_aim = systems
            .first()
            .is_none_or(|s| !self.aims_on_device(s, systems.len(), ops));
        let seeds = match anchor.filter(|_| anchored && host_aim) {
            Some(base) => self
                .reference
                .trace
                .anchor_seeds(base, systems.len(), ops, &inner)?,
            None => None,
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
        for route in 0..3 {
            ctx.report(route as f64 / 3., "resident candidate operands");
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
            let native_before = self.context.native_device_calls();
            // The ray pass handed to the reference: none of its values is
            // the device's.
            let mut from_reference = false;
            let attempt = if active.is_empty() {
                Ok(vec![])
            } else {
                match route {
                    0 => self
                        .split_rays(&selected_tracers, &refs, anchored, anchor, host_aim, &inner)
                        .map(|pass| match pass {
                            RayPass::Split(values) => values,
                            RayPass::Reference(values) => {
                                from_reference = true;
                                values
                            }
                        }),
                    1 => self.mtf.evaluate_batch(&selected(), &refs, &inner),
                    _ => self.solvers.evaluate_batch(&selected(), &refs, &inner),
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
                                _ if from_reference => {
                                    let usable = value.as_ref().is_ok_and(|v| v.is_finite());
                                    result[row][i] = value;
                                    missing[row][col] = false;
                                    completed[row][i] = usable;
                                    stats.host_operands += 1;
                                    stats.fallback_operands += 1;
                                    fallback_rows[row] = true;
                                    *stats
                                        .fallbacks
                                        .entry(HOST_ONLY_FALLBACK.into())
                                        .or_default() += 1;
                                }
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
                                _ if route == 0
                                    && !gpu_supports(&ops[i].kind)
                                    && !matches!(
                                        ops[i].kind,
                                        OperandKind::RelativeIllumination
                                    ) =>
                                {
                                    // A host stage of the ray pass.
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
            // set of missing operands: a counted fallback.
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
                        fallback_rows[row] = true;
                        stats.fallback_operands += 1;
                        completed[row][i] = usable;
                        if tracers[row].is_none() {
                            *stats
                                .fallbacks
                                .entry("candidate cannot be traced".into())
                                .or_default() += 1;
                        }
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
        stats.device_aimed_batches = u64::from(!host_aim);
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
        total.device_aimed_batches += stats.device_aimed_batches;
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
/// Pupil nodes of one relative-illumination evaluation with the trace
/// settings of `reference`.
fn ri_nodes(reference: &TraceEvaluator) -> usize {
    let s = reference.settings();
    1 + s.illumination_rings.max(2) * s.illumination_arms.max(8)
}
/// Estimated device rays per candidate of `ops` on `system` with the trace
/// settings of `reference`: an upper bound of the resident plan's probes,
/// reported as the run's workload.
pub fn device_rays_per_candidate(
    reference: &TraceEvaluator,
    system: &OpticalSystem,
    ops: &[&Operand],
) -> usize {
    estimated_rays(system, ops, ri_nodes(reference)).0
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

/// Host aiming work of one candidate of `system` for `ops`: its estimated
/// device rays times its surfaces without a closed-form intersection
/// (aspheres, polynomial and freeform sags), whose iterative f64 hits make
/// host aiming costly. Paraxial aiming solves nothing: no work.
fn aiming_work(system: &OpticalSystem, ops: &[&Operand], ri_nodes: usize) -> usize {
    fn closed_form(shape: &SurfaceShape) -> bool {
        match shape {
            SurfaceShape::Plane | SurfaceShape::Conic { .. } => true,
            SurfaceShape::Diffractive { base, .. } => closed_form(base),
            _ => false,
        }
    }
    if matches!(system.ray_aiming, RayAiming::Off) {
        return 0;
    }
    let iterative = system
        .surfaces
        .iter()
        .filter(|s| !closed_form(&s.shape))
        .count();
    estimated_rays(system, ops, ri_nodes).0 * iterative
}

/// The pass that evaluates an operand: 0 the ray pass (device rays, and the
/// first-order, model, paraxial and host-only ray operands from the same host
/// candidates), 1 MTF, 2 the native solvers.
fn route_for(kind: &OperandKind) -> usize {
    if matches!(kind, OperandKind::Mtf { .. }) {
        1
    } else if SolverEvaluator::CAPABILITIES.contains(&kind.capability()) {
        2
    } else {
        0
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
    /// Forward differences of device ray operands take
    /// [`DEVICE_FD_RELATIVE_STEP_FORWARD`] (wavefront:
    /// [`DEVICE_WAVEFRONT_FD_RELATIVE_STEP_FORWARD`]); central differences
    /// and everything the host evaluates, the reference's steps.
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        if matches!(kind, OperandKind::RelativeIllumination) {
            return Some(match scheme {
                FdScheme::Forward => 1e-6,
                FdScheme::Central => 1e-5,
            });
        }
        match scheme {
            FdScheme::Forward if gpu_supports(kind) => Some(
                if matches!(
                    kind,
                    OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
                ) {
                    DEVICE_WAVEFRONT_FD_RELATIVE_STEP_FORWARD
                } else {
                    DEVICE_FD_RELATIVE_STEP_FORWARD
                },
            ),
            _ => self.reference.fd_relative_step(kind, scheme),
        }
    }
    /// The ray stage splits the batch into plans itself.
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate(systems, ops, false, None, ctx)
    }
    /// A batch near an anchor (joined finite differences of the rendezvous
    /// scheduler) with one evaluator per ray across it, seeded from the
    /// anchor.
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.evaluate(systems, ops, anchor.is_some(), anchor, ctx)
    }
    fn for_solve(&self) -> Option<Box<dyn OperandEvaluator + '_>> {
        Some(Box::new(ResidentEvaluator::for_solve(self)))
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
        // Host candidates: one pass over the batch.
        let systems: Vec<OpticalSystem> = built
            .iter()
            .filter_map(|s| s.as_ref().ok().cloned())
            .collect();
        let anchored = batch.candidates.anchor.is_some();
        let anchor = batch.anchor();
        let mut values = self
            .evaluate(&systems, ops, anchored, anchor.as_ref(), ctx)?
            .into_iter();
        ctx.check()?;
        ctx.report(1., "parametric candidate batch complete");
        Ok(built
            .into_iter()
            .map(|s| s.map(|_| values.next().unwrap()))
            .collect())
    }
}

type Landing<'a> = (usize, resident::Traced<'a>);
/// What the stages of one batch share.
struct Stage<'s> {
    context: &'s GpuContext,
    cpu: &'s TraceEvaluator,
    ops: &'s [&'s Operand],
    design: &'s RayDesign,
    anchored: bool,
    ctx: &'s JobContext,
    /// Per pupil ray, the candidates of the landed sub-batches in which the
    /// device left it unresolved or resolved it slowly.
    unresolved: BTreeMap<resident::RayKey, usize>,
}
/// A prepared sub-batch and its submitted plan.
type Fed<'a> = (
    resident::Prepared<'a>,
    mklight_gpu::Pending<mklight_gpu::resident::PlanResult>,
);
impl<'s> Stage<'s> {
    /// Prepare a sub-batch, with the rays in `routed` kept off the device,
    /// and submit its device plan.
    fn feed<'a>(
        &self,
        chunk: &'a [Arc<SequentialTracer>],
        routed: &BTreeSet<resident::RayKey>,
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
            self.design,
            routed,
            self.ctx,
        )?;
        let plan = prepared.plan.take().expect("prepared plan");
        self.context.record_traced_rays(prepared.traced_rays());
        let pending = self.context.submit_async(plan, self.ctx)?;
        Ok((prepared, pending))
    }
    /// The host's share of a sub-batch while the device traces the rest:
    /// the rays routed to it and the operands that need no ray.
    fn host_share(
        &self,
        chunk: &[Arc<SequentialTracer>],
        prepared: &mut resident::Prepared<'_>,
    ) -> MkResult<()> {
        let started = std::time::Instant::now();
        resident::host_stage(prepared, self.cpu, chunk, self.ops, self.ctx)?;
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
    /// Wait for sub-batch `k`'s plan, read it and finish it (or, in an
    /// anchored batch, keep it for the uniform completion).
    fn land<'a>(
        &mut self,
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
        let traced = resident::collect(self.context, prepared, result)?;
        for (unresolved, slow) in traced.unresolved.iter().zip(&traced.slow) {
            for &key in unresolved.iter().chain(slow) {
                *self.unresolved.entry(key).or_default() += 1;
            }
        }
        if self.anchored {
            deferred.push((k, traced));
        } else {
            values[k] = Some(resident::finish(
                self.context,
                self.cpu,
                chunk,
                self.ops,
                traced,
                None,
                self.ctx,
            )?);
        }
        Ok(())
    }
}
impl ResidentEvaluator {
    /// The ray stage of the host/device division: host preparation, one
    /// device plan per sub-batch (aim, trace, reductions), host completion
    /// of the unresolved rays and every operand value from the host
    /// candidates. A batch of at least [`PIPELINE_FROM`] candidates overlaps
    /// the host preparation of the next sub-batch and the finishing of the
    /// previous one with the device's work on the current one. In an
    /// `anchored` batch every sub-batch is finished after all are traced,
    /// so that a ray completed on the host for one candidate is completed
    /// for all. Once a device-aimed batch of the solve has left most rays
    /// to the host ([`Routing::host_only`]) the reference evaluates the
    /// batch (seeded from `anchor` when it holds finite differences).
    fn split_rays(
        &self,
        tracers: &[Arc<SequentialTracer>],
        ops: &[&Operand],
        anchored: bool,
        anchor: Option<&OpticalSystem>,
        host_aim: bool,
        ctx: &JobContext,
    ) -> MkResult<RayPass> {
        let n = tracers.len();
        if n == 0 {
            return Ok(RayPass::Split(vec![]));
        }
        if n >= PROBE_MIN_BATCH && !host_aim && !self.routing.lock().unwrap().learned {
            // The probe's values are not used: a finite difference routes
            // all its candidates alike.
            self.split_rays(
                &tracers[..PROBE_MIN_BATCH - 1],
                ops,
                false,
                None,
                false,
                ctx,
            )?;
        }
        let cpu = &self.reference.trace;
        if self.routing.lock().unwrap().host_only {
            let systems: Vec<OpticalSystem> = tracers
                .iter()
                .map(|t| t.system().system().clone())
                .collect();
            return Ok(RayPass::Reference(cpu.evaluate_batch_near(
                &systems,
                ops,
                anchor.filter(|_| anchored),
                ctx,
            )?));
        }
        let context = &self.context;
        let design = self.ray_design(host_aim);
        let limit = batch_limit(tracers[0].system().system(), ops, ri_nodes(cpu));
        let overlap = n >= PIPELINE_FROM;
        let size = if overlap {
            pipeline_chunk(n).min(limit)
        } else {
            limit
        };
        let chunks: Vec<&[Arc<SequentialTracer>]> = tracers.chunks(size).collect();
        let mut values: Vec<Option<Vec<Vec<OperandValue>>>> = chunks.iter().map(|_| None).collect();
        let mut deferred: Vec<Landing<'_>> = Vec::new();
        let mut stage = Stage {
            context,
            cpu,
            ops,
            design: &design,
            anchored,
            ctx,
            unresolved: BTreeMap::new(),
        };
        // One routing for the whole batch, so that a ray is traced by the
        // same side for every candidate of a finite difference. Host aiming
        // routes nothing: the device only traces the host's rays.
        let routed = if host_aim {
            BTreeSet::new()
        } else {
            self.routing.lock().unwrap().rays.clone()
        };
        // The batch's rays, and those of them routed to the host.
        let (mut rays_total, mut routed_total) = (0usize, 0usize);
        let mut in_flight = None;
        for (k, chunk) in chunks.iter().enumerate() {
            let (mut prepared, pending) = stage.feed(chunk, &routed)?;
            rays_total += prepared.traced_rays() + prepared.routed_rays();
            routed_total += prepared.routed_rays();
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
        // A ray slow or unresolved in an eighth of the batch's candidates
        // (any, in small batches) goes to the host for the solve's next
        // device-aimed batches. Only device aiming learns, so that the
        // first device-aimed batch of a solve is preceded by its probe
        // candidates.
        let threshold = n.div_ceil(8).max(1);
        if !host_aim {
            self.routing.lock().unwrap().learn(
                stage
                    .unresolved
                    .iter()
                    .filter(|&(_, &count)| count >= threshold)
                    .map(|(&key, _)| key),
                routed_total as f64 / n as f64,
                rays_total as f64 / n as f64,
            );
        }
        if anchored {
            let uniform: BTreeSet<resident::RayKey> = deferred
                .iter()
                .flat_map(|(_, t)| t.host_completed())
                .collect();
            for (k, traced) in deferred {
                values[k] = Some(resident::finish(
                    context,
                    cpu,
                    chunks[k],
                    ops,
                    traced,
                    Some(&uniform),
                    ctx,
                )?);
            }
        }
        Ok(RayPass::Split(
            values
                .into_iter()
                .flat_map(|v| v.expect("finished"))
                .collect(),
        ))
    }
}
