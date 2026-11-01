//! Tolerancing: tolerance operands on model parameters, compensators,
//! reproducible Monte-Carlo sampling and a sensitivity-analysis driver.
//!
//! A [`ToleranceSpec`] perturbs parameters (`ParamPath`) by `δ ∈ [−minus,
//! +plus]` around nominal with a chosen distribution. A
//! [`ToleranceEvaluator`] turns perturbation vectors into a scalar
//! *criterion* (e.g. RMS spot or merit after re-optimising the compensators)
//! — in batches, so a GPU backend can trace many perturbed systems at once.
//!
//! * [`sensitivity`] evaluates nominal plus each operand at `−minus` and
//!   `+plus` individually (one batch) and reports the changes, worst first,
//!   with the root-sum-square prediction of the combined change.
//! * [`monte_carlo`] draws `trials` random perturbation vectors — trial `i`
//!   uses Philox stream `TOLERANCE | i`, so any trial can be reproduced on
//!   its own — and reports statistics and yield.
//!
//! [`OpticalToleranceEvaluator`] is the model-level evaluator: it applies
//! perturbations to a base `OpticalSystem` and, when compensators are given,
//! re-optimises them by least squares on the criterion merit function before
//! reporting it. It is generic over any [`OperandEvaluator`], so it works
//! today with model-only operands and later with the tracer.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use mklight_core::param::{IrregularityMode, ParamPath, PlacementComponent};
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};

use crate::lsq::{LsqSettings, least_squares};
use crate::merit::{
    MeritFunctionSpec, Operand, OperandEvaluator, OperandKind, OperandRegistry, OperandValue,
    OpticalProblem, VariableSpec,
};
use crate::monitor::{Monitor, Termination};
use crate::problem::{CandidateBatch, Problem, merit, total_violation};
use crate::rand::{Rng, streams};

/// Schema version of serialised tolerance specs.
pub const TOLERANCE_SCHEMA_VERSION: u32 = 1;

fn current_version() -> u32 {
    TOLERANCE_SCHEMA_VERSION
}

/// Distribution of a perturbation within `[−minus, +plus]`.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Distribution {
    #[default]
    Uniform,
    /// Normal truncated at `±sigmas·σ`, where the truncation points are the
    /// tolerance limits (each side scaled to its own limit).
    TruncatedNormal { sigmas: f64 },
    /// Exactly `−minus` or `+plus`, each with probability ½ (worst-case
    /// style Monte Carlo).
    EndPoint,
}

/// One toleranced parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToleranceOperand {
    pub param: ParamPath,
    /// Magnitude of the negative limit (≥ 0), parameter units.
    pub minus: f64,
    /// Positive limit (≥ 0).
    pub plus: f64,
    #[serde(default)]
    pub distribution: Distribution,
    #[serde(default)]
    pub label: String,
}

impl ToleranceOperand {
    pub fn symmetric(param: ParamPath, tol: f64) -> Self {
        ToleranceOperand {
            param,
            minus: tol,
            plus: tol,
            distribution: Distribution::Uniform,
            label: String::new(),
        }
    }
    pub fn with_distribution(mut self, d: Distribution) -> Self {
        self.distribution = d;
        self
    }
    /// `label`, or a readable name for the parameter.
    pub fn display_label(&self) -> String {
        if !self.label.trim().is_empty() {
            return self.label.clone();
        }
        if let ParamPath::ElementPlacement {
            surface,
            last,
            component,
            ..
        } = self.param
        {
            let component = match component {
                PlacementComponent::DecenterX => "Decenter X",
                PlacementComponent::DecenterY => "Decenter Y",
                PlacementComponent::TiltX => "Tilt X",
                PlacementComponent::TiltY => "Tilt Y",
                PlacementComponent::TiltZ => "Tilt Z",
            };
            return format!("Surfaces {}–{}: {component}", surface.0, last.0);
        }
        if let ParamPath::Irregularity { surface, mode } = self.param {
            let kind = match mode {
                IrregularityMode::Astigmatism => "Astigmatic",
                IrregularityMode::Spherical => "Spherical",
            };
            return format!("Surface {}: {kind} irregularity", surface.0);
        }
        let value = serde_json::to_value(&self.param).expect("parameter serializes");
        let name = value["param"].as_str().unwrap().replace('_', " ");
        let name = format!("{}{}", name[..1].to_uppercase(), &name[1..]);
        match self.param.surface() {
            Some(id) => format!("Surface {}: {name}", id.0),
            None => name,
        }
    }
    /// Draw one perturbation.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self.distribution {
            Distribution::Uniform => rng.range(-self.minus, self.plus),
            Distribution::TruncatedNormal { sigmas } => {
                let k = if sigmas > 0.0 { sigmas } else { 2.0 };
                let z = rng.truncated_normal(k) / k;
                if z >= 0.0 {
                    z * self.plus
                } else {
                    z * self.minus
                }
            }
            Distribution::EndPoint => {
                if rng.uniform() < 0.5 {
                    -self.minus
                } else {
                    self.plus
                }
            }
        }
    }
}

/// A parameter re-optimised after perturbation (back focus, an air space…).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Compensator {
    pub param: ParamPath,
    /// Allowed adjustment `[−range, +range]` around the parameter's
    /// **nominal** value (also when the parameter is itself toleranced, so the
    /// window does not move with the perturbation); `None` = unbounded.
    /// `Some(0.0)` pins the compensator.
    #[serde(default)]
    pub range: Option<f64>,
    #[serde(default)]
    pub scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloSettings {
    pub trials: usize,
    pub seed: u64,
    /// Trials per evaluator batch (cancellation is checked between batches).
    pub batch: usize,
    /// Criterion value for the yield statistic (`criterion ≤ threshold`).
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Reported percentiles of the successful criteria, in [0, 100].
    #[serde(default = "default_percentiles")]
    pub percentiles: Vec<f64>,
    /// Keep a `TrialRecord` for the first this many trials.
    #[serde(default)]
    pub record_trials: usize,
    /// Draw a uniform astigmatism axis in [0, 180)° per astigmatism-
    /// toleranced surface and trial; false keeps the model's angle.
    #[serde(default = "yes")]
    pub random_irregularity_orientation: bool,
}
fn yes() -> bool {
    true
}

pub fn default_percentiles() -> Vec<f64> {
    vec![50.0, 90.0, 97.7]
}

impl Default for MonteCarloSettings {
    fn default() -> Self {
        MonteCarloSettings {
            trials: 1000,
            seed: 0,
            batch: 64,
            threshold: None,
            percentiles: default_percentiles(),
            record_trials: 0,
            random_irregularity_orientation: true,
        }
    }
}

/// A complete tolerance set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToleranceSpec {
    #[serde(default = "current_version")]
    pub schema_version: u32,
    pub operands: Vec<ToleranceOperand>,
    #[serde(default)]
    pub compensators: Vec<Compensator>,
    /// Merit function whose value is the criterion (`None`: the evaluator
    /// defines it).
    #[serde(default)]
    pub criterion: Option<MeritFunctionSpec>,
    #[serde(default)]
    pub monte_carlo: MonteCarloSettings,
}

impl ToleranceSpec {
    pub fn new(operands: Vec<ToleranceOperand>) -> Self {
        ToleranceSpec {
            schema_version: TOLERANCE_SCHEMA_VERSION,
            operands,
            compensators: Vec::new(),
            criterion: None,
            monte_carlo: MonteCarloSettings::default(),
        }
    }
    pub fn validate(&self) -> MkResult<()> {
        if self.schema_version > TOLERANCE_SCHEMA_VERSION {
            return Err(MkError::Parse(format!(
                "tolerance schema {} is newer than supported {TOLERANCE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        let mut ranges = BTreeMap::new();
        for (i, o) in self.operands.iter().enumerate() {
            if !(o.minus >= 0.0 && o.plus >= 0.0 && o.minus.is_finite() && o.plus.is_finite()) {
                return Err(MkError::InvalidParam(format!(
                    "tolerance {i}: limits must be finite and ≥ 0"
                )));
            }
            // One rigid motion per first surface: every trial would fail.
            if let ParamPath::ElementPlacement {
                surface,
                last,
                pivot,
                ..
            } = o.param
                && *ranges.entry(surface).or_insert((last, pivot)) != (last, pivot)
            {
                return Err(MkError::InvalidParam(format!(
                    "tolerance {i}: another range or pivot starts at surface {}",
                    surface.0
                )));
            }
        }
        Ok(())
    }
    /// Perturbation vector of Monte-Carlo trial `trial` (reproducible on its
    /// own).
    pub fn sample_trial(&self, seed: u64, trial: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed, streams::TOLERANCE | trial);
        self.operands.iter().map(|o| o.sample(&mut rng)).collect()
    }
    /// The perturbed system of Monte-Carlo trial `trial`. Random astigmatism
    /// axes come from their own Philox stream (seed, trial), one draw per
    /// surface in surface-id order, so the amplitude draws and batching are
    /// unaffected.
    pub fn trial_system(
        &self,
        nominal: &OpticalSystem,
        deltas: &[f64],
        settings: &MonteCarloSettings,
        trial: u64,
    ) -> MkResult<OpticalSystem> {
        let mut system = self.apply(nominal, deltas)?;
        if !settings.random_irregularity_orientation {
            return Ok(system);
        }
        let surfaces: BTreeSet<_> = (self.operands.iter())
            .filter_map(|o| match o.param {
                ParamPath::Irregularity {
                    surface,
                    mode: IrregularityMode::Astigmatism,
                } => Some(surface),
                _ => None,
            })
            .collect();
        let mut rng = Rng::new(settings.seed, streams::TOLERANCE_ORIENTATION | trial);
        for id in surfaces {
            let angle = rng.range(0.0, 180.0);
            let Some(i) = system.surface_index(id) else {
                continue;
            };
            if let mklight_core::SurfaceShape::SagOverlay {
                astigmatism_angle_deg,
                ..
            } = system.surfaces[i].shape.sag_shape_mut()
            {
                *astigmatism_angle_deg = angle;
            }
        }
        Ok(system)
    }
    /// The system of a reported Monte-Carlo case, with its final compensators.
    pub fn trial_case_system(
        &self,
        nominal: &OpticalSystem,
        deltas: &[f64],
        compensators: &[f64],
        settings: &MonteCarloSettings,
        trial: u64,
    ) -> MkResult<OpticalSystem> {
        let system = self.trial_system(nominal, deltas, settings, trial)?;
        self.with_compensators(system, compensators)
    }
    /// The system of a reported case: perturbed, with its final compensators.
    pub fn case_system(
        &self,
        nominal: &OpticalSystem,
        deltas: &[f64],
        compensators: &[f64],
    ) -> MkResult<OpticalSystem> {
        self.with_compensators(self.apply(nominal, deltas)?, compensators)
    }
    fn with_compensators(
        &self,
        mut system: OpticalSystem,
        compensators: &[f64],
    ) -> MkResult<OpticalSystem> {
        if compensators.len() != self.compensators.len() {
            return Err(MkError::InvalidParam("compensator count differs".into()));
        }
        for (c, value) in self.compensators.iter().zip(compensators) {
            c.param.set(&mut system, *value)?;
        }
        Ok(system)
    }
    /// Apply perturbations to a copy of `nominal`. Several operands on the
    /// same parameter add up.
    pub fn apply(&self, nominal: &OpticalSystem, deltas: &[f64]) -> MkResult<OpticalSystem> {
        if deltas.len() != self.operands.len() {
            return Err(MkError::InvalidParam(
                "perturbation length differs from tolerance count".into(),
            ));
        }
        let mut sum: BTreeMap<&ParamPath, f64> = BTreeMap::new();
        for (o, d) in self.operands.iter().zip(deltas) {
            *sum.entry(&o.param).or_insert(0.0) += d;
        }
        let mut s = nominal.clone();
        for (p, d) in sum {
            let v = p.get(nominal)?;
            p.set(&mut s, v + d)?;
        }
        Ok(s)
    }
}

/// Outcome of one perturbed (and possibly compensated) system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialOutcome {
    /// `+∞` when the evaluation failed.
    pub criterion: f64,
    #[serde(default)]
    pub compensators: Vec<f64>,
    pub ok: bool,
}

impl TrialOutcome {
    pub fn failed() -> Self {
        TrialOutcome {
            criterion: f64::INFINITY,
            compensators: Vec::new(),
            ok: false,
        }
    }
}

/// Batched criterion evaluation for perturbation vectors.
pub trait ToleranceEvaluator: Sync {
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        None
    }
    fn evaluate_batch(
        &self,
        perturbations: &[Vec<f64>],
        ctx: &JobContext,
    ) -> MkResult<Vec<TrialOutcome>>;
    /// Monte-Carlo trials `first_trial..`: as `evaluate_batch`, plus the
    /// trials' random irregularity axes (`ToleranceSpec::trial_system`).
    fn evaluate_trials(
        &self,
        perturbations: &[Vec<f64>],
        _settings: &MonteCarloSettings,
        _first_trial: u64,
        ctx: &JobContext,
    ) -> MkResult<Vec<TrialOutcome>> {
        self.evaluate_batch(perturbations, ctx)
    }
}

type TolFn = dyn Fn(&[f64]) -> TrialOutcome + Send + Sync;

/// A [`ToleranceEvaluator`] from a closure (evaluated in parallel).
pub struct FnToleranceEvaluator(Box<TolFn>);

impl FnToleranceEvaluator {
    pub fn new(f: impl Fn(&[f64]) -> TrialOutcome + Send + Sync + 'static) -> Self {
        FnToleranceEvaluator(Box::new(f))
    }
}

impl ToleranceEvaluator for FnToleranceEvaluator {
    fn evaluate_batch(&self, p: &[Vec<f64>], ctx: &JobContext) -> MkResult<Vec<TrialOutcome>> {
        p.par_iter()
            .map(|d| {
                ctx.check()?;
                Ok((self.0)(d))
            })
            .collect()
    }
}

/// Per-operand sensitivity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperandSensitivity {
    pub index: usize,
    pub label: String,
    pub minus: TrialOutcome,
    pub plus: TrialOutcome,
    /// `criterion(−minus) − nominal`, `criterion(+plus) − nominal`.
    pub change_minus: f64,
    pub change_plus: f64,
    /// `max(|change_minus|, |change_plus|)`.
    pub worst: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityReport {
    pub nominal: TrialOutcome,
    /// In operand order.
    pub operands: Vec<OperandSensitivity>,
    /// Operand indices, worst first.
    pub ranking: Vec<usize>,
    /// Root-sum-square of the worst changes: predicted change of the
    /// criterion with every tolerance at its limit (statistical estimate).
    pub rss_change: f64,
    pub evaluations: usize,
    /// True when cancelled: `operands`, `ranking` and `rss_change` cover the
    /// operands whose both limits were evaluated.
    #[serde(default)]
    pub cancelled: bool,
}

/// Points per evaluation batch of [`sensitivity`] (the nominal first, then
/// each operand's minus and plus limits). Cancelling between batches keeps
/// the operands completed so far; the values do not depend on the batching.
pub const SENSITIVITY_BATCH: usize = 32;

/// Single-operand sensitivity analysis: `2n + 1` systems, evaluated in
/// batches of [`SENSITIVITY_BATCH`]. Cancelled after the nominal, it returns
/// the operands completed so far (`cancelled`); before, `Err(Cancelled)`.
pub fn sensitivity<E: ToleranceEvaluator + ?Sized>(
    spec: &ToleranceSpec,
    evaluator: &E,
    mon: &Monitor,
) -> MkResult<SensitivityReport> {
    spec.validate()?;
    let n = spec.operands.len();
    let mut pts = vec![vec![0.0; n]];
    for (i, o) in spec.operands.iter().enumerate() {
        let mut m = vec![0.0; n];
        m[i] = -o.minus;
        let mut p = vec![0.0; n];
        p[i] = o.plus;
        pts.push(m);
        pts.push(p);
    }
    mon.progress(0.0, "tolerance sensitivity");
    let mut out: Vec<TrialOutcome> = Vec::with_capacity(pts.len());
    let mut cancelled = false;
    for chunk in pts.chunks(SENSITIVITY_BATCH) {
        if mon.cancelled() && !out.is_empty() {
            cancelled = true;
            break;
        }
        match evaluator.evaluate_batch(chunk, &mon.job) {
            Ok(r) if r.len() == chunk.len() => out.extend(r),
            Ok(_) => {
                return Err(MkError::Numerical(
                    "tolerance evaluator returned the wrong count".into(),
                ));
            }
            Err(MkError::Cancelled) if !out.is_empty() => {
                cancelled = true;
                break;
            }
            Err(e) => return Err(e),
        }
        mon.progress(out.len() as f64 / pts.len() as f64, "tolerance sensitivity");
    }
    let nominal = out[0].clone();
    let done = (out.len() - 1) / 2;
    let mut ops = Vec::with_capacity(done);
    let mut rss = 0.0;
    for (i, o) in spec.operands.iter().enumerate().take(done) {
        let (m, p) = (out[1 + 2 * i].clone(), out[2 + 2 * i].clone());
        let cm = m.criterion - nominal.criterion;
        let cp = p.criterion - nominal.criterion;
        let worst = cm.abs().max(cp.abs());
        rss += worst * worst;
        ops.push(OperandSensitivity {
            index: i,
            label: o.display_label(),
            minus: m,
            plus: p,
            change_minus: cm,
            change_plus: cp,
            worst,
        });
    }
    let mut ranking: Vec<usize> = (0..done).collect();
    ranking.sort_by(|&a, &b| ops[b].worst.total_cmp(&ops[a].worst).then(a.cmp(&b)));
    if !cancelled {
        mon.progress(1.0, "tolerance sensitivity");
    }
    Ok(SensitivityReport {
        nominal,
        operands: ops,
        ranking,
        rss_change: rss.sqrt(),
        evaluations: out.len(),
        cancelled,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialRecord {
    pub trial: usize,
    pub outcome: TrialOutcome,
    pub perturbations: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloReport {
    /// The unperturbed system, compensated like a trial.
    pub nominal: TrialOutcome,
    pub records: Vec<TrialRecord>,
    /// Wall time over completed trials (parallel trials overlap).
    pub seconds_per_trial: f64,
    pub trials: usize,
    pub failures: usize,
    pub criteria: Vec<f64>,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    /// Requested `(percentile, value)` pairs, successful trials only.
    pub percentiles: Vec<(f64, f64)>,
    /// Fraction of all trials with `criterion ≤ threshold`.
    #[serde(default)]
    pub yield_fraction: Option<f64>,
    /// Mean and standard deviation of each compensator value.
    #[serde(default)]
    pub compensator_mean: Vec<f64>,
    #[serde(default)]
    pub compensator_std: Vec<f64>,
    /// True when cancelled; statistics cover the completed trials.
    pub cancelled: bool,
}

/// Monte-Carlo tolerance analysis.
pub fn monte_carlo<E: ToleranceEvaluator + ?Sized>(
    spec: &ToleranceSpec,
    evaluator: &E,
    s: &MonteCarloSettings,
    mon: &Monitor,
) -> MkResult<MonteCarloReport> {
    spec.validate()?;
    if !s.percentiles.iter().all(|p| (0.0..=100.0).contains(p)) {
        return Err(MkError::InvalidParam(
            "percentiles must be in [0, 100]".into(),
        ));
    }
    let nominal = evaluator
        .evaluate_batch(&[vec![0.0; spec.operands.len()]], &mon.job)?
        .remove(0);
    let start = Instant::now();
    let mut records = Vec::new();
    let mut outcomes: Vec<TrialOutcome> = Vec::with_capacity(s.trials);
    let batch = s.batch.max(1);
    let mut cancelled = false;
    let mut t = 0usize;
    while t < s.trials {
        if mon.cancelled() {
            cancelled = true;
            break;
        }
        let end = (t + batch).min(s.trials);
        let pts: Vec<Vec<f64>> = (t..end)
            .map(|i| spec.sample_trial(s.seed, i as u64))
            .collect();
        match evaluator.evaluate_trials(&pts, s, t as u64, &mon.job) {
            Ok(o) => {
                if o.len() != pts.len() {
                    return Err(MkError::Numerical(
                        "tolerance evaluator returned the wrong count".into(),
                    ));
                }
                for (outcome, perturbations) in o.into_iter().zip(pts) {
                    if outcomes.len() < s.record_trials {
                        records.push(TrialRecord {
                            trial: outcomes.len(),
                            outcome: outcome.clone(),
                            perturbations,
                        });
                    }
                    outcomes.push(outcome);
                }
            }
            Err(MkError::Cancelled) => {
                cancelled = true;
                break;
            }
            Err(e) => return Err(e),
        }
        t = end;
        mon.progress(0.9 * t as f64 / s.trials as f64, "tolerance monte carlo");
    }
    let ok: Vec<&TrialOutcome> = outcomes
        .iter()
        .filter(|o| o.ok && o.criterion.is_finite())
        .collect();
    // A completed reference checkpoint preserves Stop semantics if a device
    // summary is cancelled, without starting replacement work after cancellation.
    let clock = Instant::now();
    let reference = reference_statistics(&ok, &s.percentiles);
    if let Some(gpu) = evaluator.gpu_context() {
        gpu.observe(mklight_gpu::resident::PlanStats {
            stages: vec![mklight_gpu::resident::StageStats {
                name: "tolerance statistics f64 qualification".into(),
                wall_seconds: clock.elapsed().as_secs_f64(),
                ..Default::default()
            }],
            ..Default::default()
        });
    }
    let statistics = if let Some(gpu) = evaluator
        .gpu_context()
        .filter(|_| !cancelled && !ok.is_empty())
    {
        match gpu.tolerance_statistics(&ok, &s.percentiles, &mon.job.subrange(0.9, 0.99)) {
            Ok(v) if v.agrees(&reference) => v,
            Ok(_) => {
                gpu.observe(mklight_gpu::resident::PlanStats {
                    fallback_count: 1,
                    stages: vec![mklight_gpu::resident::StageStats {
                        name: "tolerance statistics f64 gate".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                });
                reference
            }
            Err(MkError::Cancelled) => {
                cancelled = true;
                reference
            }
            Err(e) => {
                gpu.observe(mklight_gpu::resident::PlanStats {
                    fallback_count: 1,
                    stages: vec![mklight_gpu::resident::StageStats {
                        name: format!("tolerance statistics reference: {e}"),
                        ..Default::default()
                    }],
                    ..Default::default()
                });
                reference
            }
        }
    } else {
        reference
    };
    let yield_fraction = s.threshold.map(|th| {
        outcomes
            .iter()
            .filter(|o| o.ok && o.criterion <= th)
            .count() as f64
            / outcomes.len().max(1) as f64
    });
    let report = MonteCarloReport {
        nominal,
        records,
        seconds_per_trial: start.elapsed().as_secs_f64() / outcomes.len().max(1) as f64,
        trials: outcomes.len(),
        failures: outcomes.len() - ok.len(),
        criteria: outcomes.iter().map(|o| o.criterion).collect(),
        mean: statistics.mean,
        std_dev: statistics.std_dev,
        min: statistics.min,
        max: statistics.max,
        percentiles: statistics.percentiles,
        yield_fraction,
        compensator_mean: statistics.compensator_mean,
        compensator_std: statistics.compensator_std,
        cancelled,
    };
    if !cancelled {
        mon.progress(1., "tolerance complete");
    }
    Ok(report)
}

pub(crate) struct Statistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: Vec<(f64, f64)>,
    pub compensator_mean: Vec<f64>,
    pub compensator_std: Vec<f64>,
}
impl Statistics {
    fn agrees(&self, r: &Self) -> bool {
        let close = |a: f64, b: f64| {
            a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-8_f64.max(1e-6 * b.abs())
        };
        [self.mean, self.std_dev, self.min, self.max]
            .into_iter()
            .zip([r.mean, r.std_dev, r.min, r.max])
            .all(|(a, b)| close(a, b))
            && self
                .percentiles
                .iter()
                .zip(&r.percentiles)
                .all(|(a, b)| close(a.1, b.1))
            && self
                .compensator_mean
                .iter()
                .chain(&self.compensator_std)
                .zip(r.compensator_mean.iter().chain(&r.compensator_std))
                .all(|(&a, &b)| close(a, b))
    }
}
fn reference_statistics(ok: &[&TrialOutcome], percentiles: &[f64]) -> Statistics {
    let mut vals: Vec<f64> = ok.iter().map(|o| o.criterion).collect();
    let nok = vals.len() as f64;
    let mean: f64 = vals
        .iter()
        .copied()
        .collect::<mklight_core::math::KahanSum>()
        .value()
        / nok.max(1.0);
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nok - 1.0).max(1.0);
    vals.sort_by(f64::total_cmp);
    let pct = |p: f64| -> f64 {
        if vals.is_empty() {
            return f64::NAN;
        }
        // Linear interpolation between order statistics.
        let r = p / 100.0 * (vals.len() - 1) as f64;
        let i = r.floor() as usize;
        let f = r - i as f64;
        if i + 1 < vals.len() {
            vals[i] + f * (vals[i + 1] - vals[i])
        } else {
            vals[i]
        }
    };
    let ncomp = ok.first().map_or(0, |o| o.compensators.len());
    let mut cm = vec![0.0; ncomp];
    let mut cs = vec![0.0; ncomp];
    for j in 0..ncomp {
        let xs: Vec<f64> = ok
            .iter()
            .filter_map(|o| o.compensators.get(j).copied())
            .collect();
        let m = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        cm[j] = m;
        cs[j] = (xs.iter().map(|v| (v - m).powi(2)).sum::<f64>()
            / (xs.len() as f64 - 1.0).max(1.0))
        .sqrt();
    }
    Statistics {
        mean: if vals.is_empty() { f64::NAN } else { mean },
        std_dev: var.sqrt(),
        min: vals.first().copied().unwrap_or(f64::NAN),
        max: vals.last().copied().unwrap_or(f64::NAN),
        percentiles: percentiles.iter().copied().map(|p| (p, pct(p))).collect(),
        compensator_mean: cm,
        compensator_std: cs,
    }
}

impl<T: OperandEvaluator + ?Sized> OperandEvaluator for &T {
    fn record_reference_stage(&self, name: &str, count: usize) {
        (**self).record_reference_stage(name, count)
    }
    fn reference_evaluator(&self) -> Option<&dyn OperandEvaluator> {
        (**self).reference_evaluator()
    }
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
    fn evaluate_parametric(
        &self,
        batch: &crate::merit::ParametricBatch<'_>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Result<Vec<OperandValue>, crate::merit::BuildError>>> {
        (**self).evaluate_parametric(batch, operands, ctx)
    }
    fn fd_relative_step(&self, kind: &OperandKind, scheme: crate::deriv::FdScheme) -> Option<f64> {
        (**self).fd_relative_step(kind, scheme)
    }
}

/// Model-level tolerance evaluator: perturb `nominal` by the spec, then
/// (with compensators) minimise the criterion merit over them by least
/// squares; the criterion is the resulting merit `Σ w r²`. A trial is `ok`
/// when its merit is finite and its constraint violation is within the
/// default monitor feasibility tolerance, with or without compensators.
pub struct OpticalToleranceEvaluator<'a, E: OperandEvaluator> {
    pub nominal: &'a OpticalSystem,
    pub spec: &'a ToleranceSpec,
    pub criterion: &'a MeritFunctionSpec,
    pub evaluator: &'a E,
    pub registry: &'a OperandRegistry,
    pub lsq: LsqSettings,
}

impl<E: OperandEvaluator> OpticalToleranceEvaluator<'_, E> {
    fn one(&self, sys: OpticalSystem, ctx: &JobContext) -> MkResult<TrialOutcome> {
        // The compensator solve shares the cancellation but not the
        // progress bar: trials run in parallel, and their LM progress would
        // move the caller's bar back and forth.
        let mon = Monitor::new(JobContext::with(
            ctx.cancel.clone(),
            mklight_core::job::Progress::default(),
        ));
        let ctx = &mon.job;
        let vars: Vec<VariableSpec> = self
            .spec
            .compensators
            .iter()
            .map(|c| {
                let v = c.param.get(self.nominal)?;
                Ok(VariableSpec {
                    param: c.param.clone(),
                    lower: c.range.map(|r| v - r),
                    upper: c.range.map(|r| v + r),
                    scale: c.scale,
                    fd_step: None,
                    label: String::new(),
                })
            })
            .collect::<MkResult<_>>()?;
        let p = OpticalProblem::new(
            sys,
            vars,
            self.criterion.clone(),
            self.evaluator,
            self.registry,
        )?;
        if p.num_variables() == 0 {
            let vals = p.system_values(&[Some(p.base().clone())], ctx)?;
            let e = p.to_evaluation(vals);
            let sc = merit(p.weights(), e.residuals(0), e.status[0]);
            let viol = total_violation(p.constraints(), e.constraints(0), e.status[0]);
            return Ok(if sc.is_finite() && viol <= mon.feasibility_tol {
                TrialOutcome {
                    criterion: sc,
                    compensators: Vec::new(),
                    ok: true,
                }
            } else {
                TrialOutcome::failed()
            });
        }
        let mut x0 = p.start_point()?;
        // MTF is not unimodal in focus, so LM from the start can settle on a
        // side lobe. Screen each bounded compensator on a uniform grid first.
        let multimodal =
            (self.criterion.operands.iter()).any(|o| matches!(o.kind, OperandKind::Mtf { .. }));
        for j in (0..x0.len()).filter(|_| multimodal) {
            let (Some(lo), Some(hi)) = (p.variables()[j].lower, p.variables()[j].upper) else {
                continue;
            };
            const STEPS: usize = 32;
            let points: Vec<Vec<f64>> = (0..=STEPS)
                .map(|k| {
                    let mut x = x0.clone();
                    x[j] = lo + (hi - lo) * k as f64 / STEPS as f64;
                    x
                })
                .chain([x0.clone()])
                .collect();
            let e = p.evaluate_batch(
                &CandidateBatch::from_points(x0.len(), points.iter().map(Vec::as_slice)),
                ctx,
            )?;
            let score = |k: usize| {
                if total_violation(p.constraints(), e.constraints(k), e.status[k])
                    <= mon.feasibility_tol
                {
                    merit(p.weights(), e.residuals(k), e.status[k])
                } else {
                    f64::INFINITY
                }
            };
            let best = (0..points.len())
                .min_by(|&a, &b| score(a).total_cmp(&score(b)))
                .expect("non-empty scan");
            x0[j] = points[best][j];
        }
        let r = least_squares(&p, &x0, &self.lsq, &mon)?;
        if r.termination == Termination::Cancelled {
            return Err(MkError::Cancelled);
        }
        Ok(TrialOutcome {
            ok: r.merit.is_finite() && r.violation <= mon.feasibility_tol,
            criterion: r.merit,
            compensators: r.x,
        })
    }
}

impl<E: OperandEvaluator> ToleranceEvaluator for OpticalToleranceEvaluator<'_, E> {
    fn gpu_context(&self) -> Option<&crate::gpu::GpuContext> {
        self.evaluator.gpu_context()
    }
    fn evaluate_batch(
        &self,
        perturbations: &[Vec<f64>],
        ctx: &JobContext,
    ) -> MkResult<Vec<TrialOutcome>> {
        let systems = perturbations
            .iter()
            .map(|d| self.spec.apply(self.nominal, d).ok())
            .collect();
        self.evaluate_systems(systems, ctx)
    }
    fn evaluate_trials(
        &self,
        perturbations: &[Vec<f64>],
        settings: &MonteCarloSettings,
        first_trial: u64,
        ctx: &JobContext,
    ) -> MkResult<Vec<TrialOutcome>> {
        let systems = (perturbations.iter().enumerate())
            .map(|(i, d)| {
                (self.spec)
                    .trial_system(self.nominal, d, settings, first_trial + i as u64)
                    .ok()
            })
            .collect();
        self.evaluate_systems(systems, ctx)
    }
}

impl<E: OperandEvaluator> OpticalToleranceEvaluator<'_, E> {
    /// Ordinary parameter trials retain the P4 path. Random irregularity
    /// orientation can change extra model fields; those keep their complete
    /// system representation and the reported host preparation route.
    fn parametric_trials(
        &self,
        systems: &[Option<OpticalSystem>],
        ctx: &JobContext,
    ) -> MkResult<Option<crate::problem::BatchEvaluation>> {
        let Some(gpu) = self.evaluator.gpu_context() else {
            return Ok(None);
        };
        ctx.check()?;
        let params: Vec<_> = self
            .spec
            .operands
            .iter()
            .map(|o| o.param.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if params.is_empty() {
            return Ok(None);
        }
        let mut points = CandidateBatch::new(params.len());
        let mut slots = Vec::new();
        for (i, system) in systems.iter().enumerate() {
            if let Some(system) = system {
                let values = params
                    .iter()
                    .map(|p| p.get(system))
                    .collect::<MkResult<Vec<_>>>()?;
                let mut represented = self.nominal.clone();
                for (p, &v) in params.iter().zip(&values) {
                    p.set(&mut represented, v)?;
                }
                if represented != *system {
                    gpu.observe(mklight_gpu::resident::PlanStats {
                        fallback_count: systems.len() as u64,
                        stages: vec![mklight_gpu::resident::StageStats {
                            name: "tolerance orientation host preparation".into(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    return Ok(None);
                }
                points.push(&values);
                slots.push(i);
            }
        }
        let variables = params
            .into_iter()
            .map(|param| VariableSpec {
                scale: Some(1.),
                ..VariableSpec::new(param)
            })
            .collect();
        let p = OpticalProblem::new(
            self.nominal.clone(),
            variables,
            self.criterion.clone(),
            self.evaluator,
            self.registry,
        )?;
        let evaluated = p.evaluate_batch(&points, ctx)?;
        let (m, k) = (p.num_residuals(), p.num_constraints());
        let mut out = crate::problem::BatchEvaluation::zeros(systems.len(), m, k);
        out.status.fill(crate::problem::EvalStatus::Failed);
        for (row, &slot) in slots.iter().enumerate() {
            out.status[slot] = evaluated.status[row];
            out.residuals[slot * m..(slot + 1) * m].copy_from_slice(evaluated.residuals(row));
            out.constraints[slot * k..(slot + 1) * k].copy_from_slice(evaluated.constraints(row));
        }
        Ok(Some(out))
    }
    /// Without compensators every trial is one evaluation: all perturbed
    /// systems go to the operand evaluator as one batch (one GPU dispatch
    /// for a batched backend). Same outcomes as trial by trial.
    fn evaluate_systems(
        &self,
        systems: Vec<Option<OpticalSystem>>,
        ctx: &JobContext,
    ) -> MkResult<Vec<TrialOutcome>> {
        if !self.spec.compensators.is_empty() && self.evaluator.gpu_context().is_some() {
            let mut outcomes = Vec::with_capacity(systems.len());
            for chunk in systems.chunks(32) {
                outcomes.extend(crate::gpu::scheduler::run(
                    self.evaluator,
                    chunk.len(),
                    ctx,
                    |i, worker| {
                        let local = OpticalToleranceEvaluator {
                            nominal: self.nominal,
                            spec: self.spec,
                            criterion: self.criterion,
                            evaluator: worker,
                            registry: self.registry,
                            lsq: self.lsq,
                        };
                        match &chunk[i] {
                            Some(s) => local.one(s.clone(), ctx),
                            None => Ok(TrialOutcome::failed()),
                        }
                    },
                )?);
            }
            return Ok(outcomes);
        }
        if !self.spec.compensators.is_empty() {
            return systems
                .into_par_iter()
                .map(|s| {
                    ctx.check()?;
                    match s {
                        Some(s) => self.one(s, ctx),
                        None => Ok(TrialOutcome::failed()),
                    }
                })
                .collect();
        }
        let p = OpticalProblem::new(
            self.nominal.clone(),
            Vec::new(),
            self.criterion.clone(),
            self.evaluator,
            self.registry,
        )?;
        let e = match self.parametric_trials(&systems, ctx)? {
            Some(e) => e,
            None => p.to_evaluation(p.system_values(&systems, ctx)?),
        };
        let tol = Monitor::new(ctx.clone()).feasibility_tol;
        Ok((0..systems.len())
            .map(|k| {
                let m = merit(p.weights(), e.residuals(k), e.status[k]);
                let v = total_violation(p.constraints(), e.constraints(k), e.status[k]);
                if systems[k].is_some() && m.is_finite() && v <= tol {
                    TrialOutcome {
                        criterion: m,
                        compensators: Vec::new(),
                        ok: true,
                    }
                } else {
                    TrialOutcome::failed()
                }
            })
            .collect())
    }
}
