//! Optimisation and tolerancing as engine analyses: `optimize` and
//! `tolerance` (`mklight-optim`), with merit evaluation on the CPU f64
//! reference or GPU-batched on a `mklight-gpu` backend
//! (`GpuTraceEvaluator`). Settings and result shapes: `docs/OPTIMIZATION.md`
//! (Engine jobs).
//!
//! * **Progress** is reported per optimiser iteration (stage text
//!   `"<method> <iteration>: merit <best>"`) and per Monte-Carlo batch.
//! * **Stopping and cancelling.** Cancelling through the job context stops
//!   the optimiser, and `Analysis::run` returns the best point so far
//!   (feasibility-ranked) with `"status": "cancelled"`. The scheduler,
//!   however, discards the results of cancelled jobs (a superseded result is
//!   stale). To stop a job *and keep* its best point, call
//!   `Engine::stop_job` (or give it a `stop_key` and call [`request_stop`]):
//!   the job then ends normally with `"status": "stopped"`.
//! * **Reported numbers** (start and final merit, operand table) are always
//!   computed with the CPU f64 reference, whatever evaluated the search; the
//!   result says which backend evaluated it, with its precision
//!   qualification, and `RunInfo.warnings` repeats it.
//! * **Never worse than the start**: the result is the better of the start
//!   and the optimiser's answer on the reference (feasibility first), with
//!   a warning when the optimiser found nothing better.
//! * **Backend `auto`** uses the engine's GPU device only past the measured
//!   GPU/CPU crossover for the run's rays per candidate and batch size.
//! * The result carries the final system as model JSON (`"system"`), so a
//!   front end can apply it as a new (undoable) snapshot.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use mklight_core::job::{CancelToken, Precision, Progress};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, OpticalSystem, Snapshot};
use mklight_gpu::xval::Qualification;
use mklight_optim::deriv::{FdScheme, JacobianSettings};
use mklight_optim::global::cmaes::{CmaEsSettings, cma_es};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::global::nelder_mead::{NelderMeadSettings, nelder_mead};
use mklight_optim::lsq::{AugLagSettings, LsqMethod, LsqSettings, least_squares};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::geometric::optical_gaps;
use mklight_optim::merit::gpu_eval::{gpu_beats_cpu, gpu_rays_per_candidate};
use mklight_optim::merit::operand::OperandKind;
use mklight_optim::merit::{
    Comparison, GpuEvalSettings, GpuTraceEvaluator, MeritFunctionSpec, Operand, OperandEvaluator,
    OperandRegistry, OperandRole, OperandValue, OpticalProblem, TraceEvaluator, VariableSpec,
};
use mklight_optim::problem::{
    CandidateBatch, EvalStatus, Problem, ScalarEval, merit, total_violation,
};
use mklight_optim::tolerance::{
    Compensator, MonteCarloSettings, OpticalToleranceEvaluator, SENSITIVITY_BATCH,
    ToleranceOperand, ToleranceSpec, monte_carlo, sensitivity,
};
use mklight_optim::{IterationRecord, Monitor, OptimResult, Termination};
use mklight_trace::{SemiDiameterSettings, SequentialTracer};

use super::BackendUse;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};
use crate::gpu::GpuDevice;

/// Register `optimize` and `tolerance`.
pub fn register(r: &mut AnalysisRegistry) {
    r.register(Arc::new(Optimize));
    r.register(Arc::new(Tolerance));
}

// The `backend` record shared by both result schemas.
fn evaluation_backend_schema() -> Value {
    use super::schema::*;
    obj_opt(
        &[
            (
                "requested",
                enumeration(
                    &["auto", "cpu", "gpu", "gpu_emulation"],
                    "Merit evaluator preference.",
                ),
            ),
            (
                "evaluated_on",
                string("Search evaluator backend; envelope uses engine backend ids."),
            ),
            (
                "precision",
                enumeration(&["f64_reference", "f32_gpu"], "Search precision."),
            ),
            (
                "reported_on",
                string("CPU reference for optimize; search evaluator for tolerance."),
            ),
            (
                "workload",
                obj(&[
                    ("rays_per_candidate", nullable(int(""))),
                    ("batch", int("")),
                    ("gpu_faster", boolean("")),
                ]),
            ),
            (
                "qualification",
                opaque("Cached GPU precision qualification for the start snapshot."),
            ),
            (
                "final_qualification",
                opaque("GPU qualification at the optimised system."),
            ),
            (
                "gpu_stats",
                obj(&[
                    ("gpu_candidates", int("")),
                    ("cpu_candidates", int("")),
                    ("rays", int("")),
                    ("dispatches", int("")),
                    ("device_seconds", num("s", "")),
                ]),
            ),
        ],
        &["qualification", "final_qualification", "gpu_stats"],
    )
}

// ---------------------------------------------------------------------------
// Stop requests

struct StopEntry {
    id: u64,
    token: CancelToken,
}

fn stops() -> &'static Mutex<HashMap<String, StopEntry>> {
    static S: OnceLock<Mutex<HashMap<String, StopEntry>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Ask the running `optimize` or `tolerance` job started with
/// `"stop_key": key` to stop and return its best result so far (the job
/// ends normally, `"status": "stopped"`). Returns whether such a job was
/// running.
pub fn request_stop(key: &str) -> bool {
    let g = stops().lock().unwrap_or_else(|e| e.into_inner());
    match g.get(key) {
        Some(e) => {
            e.token.cancel();
            true
        }
        None => false,
    }
}

/// Registration of a running job's stop token; removed on drop.
struct StopGuard {
    key: String,
    id: u64,
}

impl StopGuard {
    /// Register `token` under `key`; a key already held by a running job is
    /// refused (the first job could otherwise no longer be stopped).
    fn register(key: &str, token: CancelToken) -> EngineResult<StopGuard> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut g = stops().lock().unwrap_or_else(|e| e.into_inner());
        if g.contains_key(key) {
            return Err(invalid(format!(
                "'stop_key': \"{key}\" is already used by a running job"
            )));
        }
        g.insert(key.to_string(), StopEntry { id, token });
        Ok(StopGuard {
            key: key.to_string(),
            id,
        })
    }
}

/// A job's stop request: the job's stop token (`AnalysisInput::stop`),
/// also registered under its `stop_key` for the lifetime of the value.
struct StopRequest {
    token: CancelToken,
    _guard: Option<StopGuard>,
}

impl StopRequest {
    fn register(key: Option<&str>, token: CancelToken) -> EngineResult<StopRequest> {
        let _guard = key
            .map(|k| StopGuard::register(k, token.clone()))
            .transpose()?;
        Ok(StopRequest { token, _guard })
    }
}

impl Drop for StopGuard {
    fn drop(&mut self) {
        let mut g = stops().lock().unwrap_or_else(|e| e.into_inner());
        if g.get(&self.key).is_some_and(|e| e.id == self.id) {
            g.remove(&self.key);
        }
    }
}

/// How a run ended early.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Interrupt {
    None,
    Stopped,
    Cancelled,
}

/// Run `f` with a context that is cancelled when `ctx` is or when `stop` is
/// requested (a watcher thread joins the two tokens), and say which
/// happened.
fn interruptible<T>(
    ctx: &JobContext,
    stop: &StopRequest,
    f: impl FnOnce(&JobContext) -> T,
) -> (T, Interrupt) {
    let stop = &stop.token;
    let mut inner = ctx.subrange(0.0, 1.0);
    inner.cancel = CancelToken::new();
    let done = AtomicBool::new(false);
    let out = std::thread::scope(|s| {
        s.spawn(|| {
            while !done.load(Ordering::Acquire) {
                if ctx.cancel.is_cancelled() || stop.is_cancelled() {
                    inner.cancel.cancel();
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        let r = f(&inner);
        done.store(true, Ordering::Release);
        r
    });
    let why = if ctx.cancel.is_cancelled() {
        Interrupt::Cancelled
    } else if stop.is_cancelled() {
        Interrupt::Stopped
    } else {
        Interrupt::None
    };
    (out, why)
}

// ---------------------------------------------------------------------------
// Shared helpers

/// Finite numbers as numbers, NaN as null, ±∞ as `"inf"`/`"-inf"`
/// (CONVENTIONS §11).
fn num(x: f64) -> Value {
    if x.is_nan() {
        Value::Null
    } else if x.is_infinite() {
        json!(if x > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(x)
    }
}

fn invalid(e: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidSettings(e.to_string())
}

fn parse<T: DeserializeOwned>(v: &Value, what: &str) -> EngineResult<T> {
    serde_json::from_value(v.clone()).map_err(|e| invalid(format!("{what}: {e}")))
}

/// Recursively overlay `patch` on `base` (objects merge key by key, other
/// values replace).
fn overlay(base: &mut Value, patch: &Value) {
    match (base, patch) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                overlay(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => *a = b.clone(),
    }
}

/// `defaults` with the (partial) JSON `patch` laid over it.
fn patched<T: Serialize + DeserializeOwned>(
    defaults: &T,
    patch: &Value,
    what: &str,
) -> EngineResult<T> {
    let mut v = serde_json::to_value(defaults)?;
    if !patch.is_null() {
        overlay(&mut v, patch);
    }
    parse(&v, what)
}

/// Which evaluator computes the merit during the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BackendChoice {
    /// The engine's GPU device when the run's work is past the measured
    /// GPU/CPU crossover ([`gpu_beats_cpu`]: rays per candidate and batch
    /// size), else the CPU reference.
    Auto,
    Cpu,
    /// The engine's GPU device for this analysis; the CPU reference (with
    /// a warning) when there is none or it fails qualification.
    Gpu,
    /// The CPU-f32 emulation of the GPU kernels (validation and tests).
    GpuEmulation,
}

impl BackendChoice {
    fn name(self) -> &'static str {
        match self {
            BackendChoice::Auto => "auto",
            BackendChoice::Cpu => "cpu",
            BackendChoice::Gpu => "gpu",
            BackendChoice::GpuEmulation => "gpu_emulation",
        }
    }
}

/// The work of a run, for `backend: "auto"`: rays the GPU evaluator would
/// trace per candidate and the typical batch of candidates per evaluation.
#[derive(Clone, Copy, Debug)]
struct Workload {
    rays_per_candidate: Option<usize>,
    batch: usize,
}

impl Workload {
    /// `rays` estimated on the start system (configuration 0) for the
    /// operands of `spec`.
    fn new(
        cpu: &TraceEvaluator,
        system: &OpticalSystem,
        spec: &MeritFunctionSpec,
        batch: usize,
    ) -> Workload {
        let ops: Vec<&Operand> = spec
            .enabled()
            .map(|(_, o)| o)
            .filter(|o| o.config == 0)
            .collect();
        Workload {
            rays_per_candidate: gpu_rays_per_candidate(cpu, system, &ops, true),
            batch,
        }
    }
    fn gpu_faster(&self) -> bool {
        self.rays_per_candidate
            .is_some_and(|r| gpu_beats_cpu(r, self.batch))
    }
    fn report(&self) -> Value {
        json!({
            "rays_per_candidate": self.rays_per_candidate,
            "batch": self.batch,
            "gpu_faster": self.gpu_faster(),
        })
    }
}

/// The GPU backend for a request, or why there is none.
fn gpu_backend(
    choice: BackendChoice,
    input: &AnalysisInput<'_>,
    work: &Workload,
) -> Result<Option<Arc<GpuDevice>>, String> {
    let engine = super::gpu::device(input);
    match choice {
        BackendChoice::Cpu => Ok(None),
        BackendChoice::Auto => Ok(engine.filter(|_| {
            !crate::backend::Backend::is_auto_request(&input.requested_backend) || work.gpu_faster()
        })),
        BackendChoice::GpuEmulation => engine
            .filter(|d| d.id == "gpu:cpu_f32")
            .map(Some)
            .ok_or_else(|| "select the engine's gpu:cpu_f32 provider for gpu_emulation".into()),
        BackendChoice::Gpu => engine
            .map(Some)
            .ok_or_else(|| "the engine selected no GPU device for this analysis".to_string()),
    }
}

/// The merit evaluator of a run: the CPU reference alone, or a qualified
/// GPU evaluator in front of it.
struct Evaluation {
    cpu: TraceEvaluator,
    /// The GPU evaluator and the engine device behind it.
    gpu: Option<(GpuTraceEvaluator, Arc<GpuDevice>)>,
    work: Workload,
    fallback_reason: Option<String>,
    /// `"backend"` block of the result.
    report: Value,
    warnings: Vec<String>,
}

/// The device's qualification of `snapshot`, cached per snapshot and
/// library, and the reference tracer it was qualified with.
fn qualify(
    device: &GpuDevice,
    cpu: &TraceEvaluator,
    snapshot: &Snapshot,
    input: &AnalysisInput<'_>,
    ctx: &JobContext,
) -> EngineResult<(SequentialTracer, Arc<Qualification>)> {
    let tracer = cpu
        .tracer(snapshot.system())
        .map_err(|e| invalid(format!("tracer: {e:?}")))?;
    let key = format!("{}:{}", snapshot.hash().to_hex(), input.library_hash());
    let q = device.qualify(&key, &tracer, ctx)?;
    Ok((tracer, q))
}

impl Evaluation {
    fn new(
        choice: BackendChoice,
        input: &AnalysisInput<'_>,
        batch: usize,
        work: Workload,
        ctx: &JobContext,
    ) -> EngineResult<Self> {
        let cpu = TraceEvaluator::new(input.library.clone());
        let mut warnings = Vec::new();
        let mut fallback_reason = input.selection_note.clone();
        let gpu = match gpu_backend(choice, input, &work) {
            Ok(Some(device)) => {
                let settings = GpuEvalSettings {
                    batch: batch.max(1),
                    ..Default::default()
                };
                let (tracer, q) = qualify(&device, &cpu, &input.snapshot, input, ctx)?;
                let hints = tracer
                    .semi_diameters(&SemiDiameterSettings::default())
                    .values;
                let g = GpuTraceEvaluator::with_qualification(
                    device.backend.clone(),
                    cpu.clone(),
                    settings,
                    q.as_ref().clone(),
                    hints,
                );
                Some((g, device))
            }
            Ok(None) => {
                if super::gpu::device(input).is_some() {
                    fallback_reason = Some(if choice == BackendChoice::Cpu {
                        "merit evaluation requested the CPU reference".into()
                    } else {
                        "auto: merit workload below the GPU crossover; using the CPU reference"
                            .into()
                    });
                }
                None
            }
            Err(why) => {
                fallback_reason = Some(why.clone());
                warnings.push(format!(
                    "GPU not used ({why}); computed with the CPU f64 reference"
                ));
                None
            }
        };
        Ok(Evaluation {
            cpu,
            gpu,
            work,
            fallback_reason,
            report: Value::Null,
            warnings,
        })
    }

    fn active_gpu(&self) -> Option<&GpuTraceEvaluator> {
        self.gpu.as_ref().map(|(g, _)| g).filter(|g| g.is_active())
    }

    /// The `"backend"` block: who evaluated the search, its qualification
    /// (at the start and, after a GPU run, at the final design) and the GPU
    /// work counters. Reports the backend used to the engine.
    fn finish(
        &mut self,
        input: &AnalysisInput<'_>,
        requested: BackendChoice,
        final_snapshot: Option<&Snapshot>,
        ctx: &JobContext,
    ) -> EngineResult<()> {
        let mut b = json!({
            "requested": requested.name(),
            "evaluated_on": TraceEvaluator::BACKEND,
            "precision": Precision::F64Reference,
            "reported_on": TraceEvaluator::BACKEND,
            "workload": self.work.report(),
        });
        // The GPU evaluated the search only if it traced candidates.
        let active = self
            .active_gpu()
            .is_some_and(|g| g.stats().gpu_candidates > 0);
        if let Some((g, device)) = &self.gpu {
            if active {
                self.warnings.push(format!(
                    "ray operands evaluated on {} (f32, precision-qualified)",
                    g.backend_id()
                ));
                b["evaluated_on"] = json!(g.backend_id());
                b["precision"] = json!(Precision::F32Gpu);
                // A qualification of the final design (the search moved
                // away from the qualified start).
                if let Some(snapshot) = final_snapshot {
                    let (_, q) = qualify(device, &self.cpu, snapshot, input, ctx)?;
                    if !q.passed {
                        self.warnings.push(format!(
                            "the final design fails the GPU precision qualification ({}); its \
                             reported values come from the CPU f64 reference",
                            q.reason.clone().unwrap_or_default()
                        ));
                    }
                    b["final_qualification"] =
                        serde_json::to_value(q.as_ref()).unwrap_or(Value::Null);
                } else {
                    b["reported_on"] = json!(g.backend_id());
                }
            }
            b["qualification"] = serde_json::to_value(g.qualification()).unwrap_or(Value::Null);
            b["gpu_stats"] = serde_json::to_value(g.stats()).unwrap_or(Value::Null);
        }
        let used = self.gpu.as_ref().filter(|_| active);
        let fallback_reason = match &self.gpu {
            _ if active => None,
            Some((g, _)) => g
                .qualification()
                .reason
                .or_else(|| Some("no candidates traced on the GPU".into())),
            None => self.fallback_reason.clone(),
        };
        input.report_backend(BackendUse {
            id: used.map_or("cpu_reference".into(), |(_, d)| d.id.clone()),
            precision: if active {
                Precision::F32Gpu
            } else {
                Precision::F64Reference
            },
            device: used.map(|(_, d)| d.device_name()),
            fallback_reason,
            qualification: b.get("qualification").cloned(),
            deliberate: self.gpu.is_none(),
        });
        self.report = b;
        Ok(())
    }
}

/// The merit function of a run: an explicit spec, or the default merit
/// generated for the variables.
fn build_merit(
    system: &OpticalSystem,
    merit_spec: &Value,
    default_merit: &DefaultMeritSpec,
    variables: &[VariableSpec],
) -> EngineResult<MeritFunctionSpec> {
    Ok(if merit_spec.is_null() {
        default_merit.generate_for(system, variables)?
    } else {
        MeritFunctionSpec::from_value(merit_spec.clone())?
    })
}

/// Relative EFL error at which the penalty that holds the focal length in
/// the derivative-free methods costs as much as the start merit.
const EFL_PENALTY_RELATIVE: f64 = 1e-4;

/// The merit of `spec` at the start (CPU reference).
fn start_merit(
    system: &OpticalSystem,
    variables: &[VariableSpec],
    spec: &MeritFunctionSpec,
    cpu: &TraceEvaluator,
    reg: &OperandRegistry,
) -> EngineResult<f64> {
    let p = OpticalProblem::new(
        system.clone(),
        variables.to_vec(),
        spec.clone(),
        cpu.clone(),
        reg,
    )?;
    let x0 = p.start_point()?;
    Ok(reference_values(&p, &[&x0])?.merit[0])
}

/// Thickness boundaries of the generated default merit that the start
/// violates and that no variable changes (for example the default 1 mm
/// glass minimum on a fixed 0.99997 mm element) are relaxed to their start
/// value: no search could meet them, and the run would be infeasible from
/// the outset. A boundary counts as changed when moving any one variable by
/// its scale changes its value. Returns a warning per relaxed boundary.
fn relax_fixed_boundaries(
    system: &OpticalSystem,
    spec: &mut MeritFunctionSpec,
    variables: &[VariableSpec],
    cpu: &TraceEvaluator,
    reg: &OperandRegistry,
    ctx: &JobContext,
) -> EngineResult<Vec<String>> {
    let idx: Vec<usize> = spec
        .enabled()
        .filter(|(_, o)| {
            o.comparison != Comparison::Equal
                && matches!(
                    o.kind,
                    OperandKind::CentreThickness { .. } | OperandKind::EdgeThickness { .. }
                )
        })
        .map(|(i, _)| i)
        .collect();
    if idx.is_empty() {
        return Ok(Vec::new());
    }
    let sub = MeritFunctionSpec {
        operands: idx
            .iter()
            .map(|&i| {
                let mut o = spec.operands[i].clone();
                o.role = OperandRole::Constraint;
                o
            })
            .collect(),
        ..spec.clone()
    };
    let p = OpticalProblem::new(system.clone(), variables.to_vec(), sub, cpu.clone(), reg)?;
    let x0 = p.start_point()?;
    let mut points = vec![x0.clone()];
    for (j, v) in p.variables().iter().enumerate() {
        let mut x = x0.clone();
        x[j] += v.safe_scale();
        points.push(x);
    }
    let batch = CandidateBatch::from_points(x0.len(), points.iter().map(Vec::as_slice));
    let vals = p.operand_values(&batch, ctx)?;
    let Some(start) = &vals[0] else {
        return Ok(Vec::new());
    };
    let mut warnings = Vec::new();
    for (k, &i) in idx.iter().enumerate() {
        let Ok(v0) = start[k] else { continue };
        let op = &mut spec.operands[i];
        if op.residual(v0) == 0.0 {
            continue;
        }
        let fixed = vals[1..]
            .iter()
            .all(|v| v.as_ref().is_some_and(|v| matches!(v[k], Ok(x) if x == v0)));
        if fixed {
            warnings.push(format!(
                "{} {} {}: the start has {v0} and no variable changes it; relaxed to {v0}",
                op.kind.name(),
                match op.comparison {
                    Comparison::LessThan => "≤",
                    _ => "≥",
                },
                op.target
            ));
            op.target = v0;
            op.label = format!("{} (relaxed to the start)", op.kind.name());
        }
    }
    Ok(warnings)
}

/// The default variables: the curvature of every curved, optically active
/// surface and the back focus (the thickness before the image).
fn auto_variables(sys: &OpticalSystem) -> Vec<VariableSpec> {
    let n = sys.surfaces.len();
    let gaps = optical_gaps(sys);
    let mut v: Vec<VariableSpec> = gaps
        .iter()
        .map(|g| ParamPath::Curvature {
            surface: sys.surfaces[g.from].id,
        })
        .filter(|p| p.get(sys).is_ok_and(|c| c != 0.0))
        .map(VariableSpec::new)
        .collect();
    if n >= 3 {
        v.push(VariableSpec::new(ParamPath::Thickness {
            surface: sys.surfaces[n - 2].id,
        }));
    }
    v
}

/// The default tolerances: ±0.05 mm on every glass and air thickness but
/// the back focus, and ±0.5 % on every curvature of a curved optically
/// active surface.
fn auto_tolerances(sys: &OpticalSystem) -> Vec<ToleranceOperand> {
    let gaps = optical_gaps(sys);
    let mut t = Vec::new();
    for g in &gaps[..gaps.len().saturating_sub(1)] {
        let mut o = ToleranceOperand::symmetric(
            ParamPath::Thickness {
                surface: sys.surfaces[g.from].id,
            },
            0.05,
        );
        o.label = format!("thickness {}", sys.surfaces[g.from].id.0);
        t.push(o);
    }
    for g in &gaps {
        let p = ParamPath::Curvature {
            surface: sys.surfaces[g.from].id,
        };
        if let Ok(c) = p.get(sys)
            && c != 0.0
        {
            let mut o = ToleranceOperand::symmetric(p, 0.005 * c.abs());
            o.label = format!("curvature {}", sys.surfaces[g.from].id.0);
            t.push(o);
        }
    }
    t
}

/// Does the spec contain an enabled `efl` operand?
fn has_efl(spec: &MeritFunctionSpec) -> bool {
    spec.enabled()
        .any(|(_, o)| matches!(o.kind, OperandKind::Efl))
}

fn value_json(v: &OperandValue) -> (Value, Value) {
    match v {
        Ok(x) => (num(*x), Value::Null),
        Err(e) => (Value::Null, json!(format!("{e:?}"))),
    }
}

/// The operand table: every operand of `spec` with its value at the start
/// and at the end (CPU f64 reference), residual and share of the merit.
fn operand_table(
    spec: &MeritFunctionSpec,
    start: &[OperandValue],
    end: &[OperandValue],
    end_merit: f64,
) -> Vec<Value> {
    spec.operands
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let (v0, _) = value_json(&start[i]);
            let (v1, err) = value_json(&end[i]);
            let (res, contrib) = match (&end[i], op.role) {
                (Ok(v), OperandRole::Residual) if op.enabled => {
                    let r = op.residual(*v);
                    (num(r), op.weight * r * r)
                }
                _ => (Value::Null, f64::NAN),
            };
            json!({
                "index": i,
                "kind": op.kind.name(),
                "label": op.label,
                "enabled": op.enabled,
                "role": op.role,
                "comparison": op.comparison,
                "target": num(op.target),
                "weight": num(op.weight),
                "config": op.config,
                "field": op.field,
                "wavelength": op.wavelength,
                "start": v0,
                "value": v1,
                "residual": res,
                "contribution": num(contrib),
                "contribution_fraction": num(if end_merit > 0.0 { contrib / end_merit } else { f64::NAN }),
                "error": err,
            })
        })
        .collect()
}

/// Merit, violation and operand values of `points` on the CPU reference.
struct Reference {
    merit: Vec<f64>,
    violation: Vec<f64>,
    status: Vec<EvalStatus>,
    values: Vec<Vec<OperandValue>>,
}

fn reference_values(
    p: &OpticalProblem<TraceEvaluator>,
    points: &[&[f64]],
) -> EngineResult<Reference> {
    let n = p.num_variables();
    let batch = CandidateBatch::from_points(n, points.iter().copied());
    let ctx = JobContext::new();
    let vals = p.operand_values(&batch, &ctx)?;
    let e = p.to_evaluation(vals.clone());
    let n_ops = p.spec().operands.len();
    Ok(Reference {
        merit: (0..points.len())
            .map(|k| merit(p.weights(), e.residuals(k), e.status[k]))
            .collect(),
        violation: (0..points.len())
            .map(|k| total_violation(p.constraints(), e.constraints(k), e.status[k]))
            .collect(),
        status: e.status.clone(),
        values: vals
            .into_iter()
            .map(|v| {
                v.unwrap_or_else(|| {
                    vec![
                        Err(mklight_optim::merit::OperandError::Invalid(
                            "invalid parameters".into()
                        ));
                        n_ops
                    ]
                })
            })
            .collect(),
    })
}

fn interrupt_status(why: Interrupt, termination: Termination) -> &'static str {
    match why {
        Interrupt::Cancelled => "cancelled",
        Interrupt::Stopped => "stopped",
        Interrupt::None if termination.is_converged() => "converged",
        Interrupt::None => "finished",
    }
}

// ---------------------------------------------------------------------------
// optimize

/// Iterations an augmented-Lagrangian inner solve may take at an infeasible
/// point before the multipliers or the penalty are updated (the job's
/// default for `AugLagSettings::max_inner_iterations`).
const INNER_ITERATIONS: usize = 10;

/// The job's initial spreads of the population methods around the start,
/// in variable scales (`VariableSpec::scale`; 10 % of a curvature by
/// default). The library defaults (DE 1.0, CMA-ES 0.3) suit a search of a
/// box; around an existing design they sample candidates so much worse that
/// the search ends at or near the start. Measured on the seven samples with
/// the default merit and variables, 100 DE / 200 CMA-ES generations: DE
/// improved 2 of 7 at 1.0 and 7 of 7 at 0.03, CMA-ES 6 of 7 at 0.3 (with the
/// EFL penalty) and 7 of 7 at 0.05 (`docs/OPTIMIZATION.md`).
const DE_INIT_SPREAD: f64 = 0.03;
const CMA_ES_SIGMA0: f64 = 0.05;

/// Optimiser of an `optimize` job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Algorithm {
    Lm,
    Dogleg,
    De,
    CmaEs,
    NelderMead,
    Hammer,
}

impl Algorithm {
    /// Derivative-free methods ranking random or simplex trial points: an
    /// equality hold is a penalty for them.
    fn derivative_free(self) -> bool {
        matches!(
            self,
            Algorithm::De | Algorithm::CmaEs | Algorithm::NelderMead
        )
    }
    fn name(self) -> &'static str {
        match self {
            Algorithm::Lm => "lm",
            Algorithm::Dogleg => "dogleg",
            Algorithm::De => "de",
            Algorithm::CmaEs => "cma_es",
            Algorithm::NelderMead => "nelder_mead",
            Algorithm::Hammer => "hammer",
        }
    }
    /// The `max_iterations` default (iterations, generations or rounds).
    fn default_iterations(self) -> usize {
        match self {
            Algorithm::Lm | Algorithm::Dogleg => 200,
            Algorithm::De => 100,
            Algorithm::CmaEs => 200,
            Algorithm::NelderMead => 2000,
            Algorithm::Hammer => 20,
        }
    }
}

#[derive(Deserialize)]
struct OptimizeSettings {
    variables: Vec<VariableSpec>,
    hold_efl: bool,
    merit: Value,
    default_merit: Value,
    operands: Vec<Operand>,
    constraints: Vec<Operand>,
    algorithm: Algorithm,
    options: Value,
    max_iterations: Option<usize>,
    fd_scheme: FdScheme,
    fd_relative_step: Option<f64>,
    seed: u64,
    backend: BackendChoice,
    gpu_batch: usize,
    feasibility_tolerance: f64,
    stop_key: Option<String>,
}

/// An optimiser with its settings.
enum Algo {
    Lsq(LsqSettings),
    De(DeSettings),
    CmaEs(CmaEsSettings),
    NelderMead(NelderMeadSettings),
    Hammer(HammerSettings),
}

impl Algo {
    fn new(s: &OptimizeSettings) -> EngineResult<(Algo, usize)> {
        let iters = s.max_iterations.unwrap_or(s.algorithm.default_iterations());
        let jacobian = JacobianSettings {
            scheme: s.fd_scheme,
            relative_step: s.fd_relative_step,
            use_analytic: true,
        };
        let opts = &s.options;
        // Augmented-Lagrangian rounds update the multipliers at least every
        // `INNER_ITERATIONS` iterations while infeasible, so a run that ends
        // on its iteration budget ends close to feasible.
        let constraints = AugLagSettings {
            max_inner_iterations: Some(INNER_ITERATIONS),
            ..Default::default()
        };
        Ok(match s.algorithm {
            Algorithm::Lm | Algorithm::Dogleg => {
                let method = if s.algorithm == Algorithm::Lm {
                    LsqMethod::LevenbergMarquardt
                } else {
                    LsqMethod::Dogleg
                };
                let mut l = patched(
                    &LsqSettings {
                        method,
                        jacobian,
                        max_iterations: iters,
                        constraints,
                        ..Default::default()
                    },
                    opts,
                    "options",
                )?;
                l.method = method;
                (Algo::Lsq(l), l.max_iterations)
            }
            Algorithm::De => {
                let mut d = patched(
                    &DeSettings {
                        max_generations: iters,
                        init_spread: DE_INIT_SPREAD,
                        ..Default::default()
                    },
                    opts,
                    "options",
                )?;
                d.seed = s.seed;
                (Algo::De(d), d.max_generations)
            }
            Algorithm::CmaEs => {
                let mut c = patched(
                    &CmaEsSettings {
                        max_generations: iters,
                        sigma0: CMA_ES_SIGMA0,
                        ..Default::default()
                    },
                    opts,
                    "options",
                )?;
                c.seed = s.seed;
                (Algo::CmaEs(c), c.max_generations)
            }
            Algorithm::NelderMead => {
                let n = patched(
                    &NelderMeadSettings {
                        max_iterations: iters,
                        ..Default::default()
                    },
                    opts,
                    "options",
                )?;
                (Algo::NelderMead(n), n.max_iterations)
            }
            Algorithm::Hammer => {
                let mut d = HammerSettings {
                    rounds: iters,
                    ..Default::default()
                };
                d.local.constraints.max_inner_iterations = Some(INNER_ITERATIONS);
                let mut h = patched(&d, opts, "options")?;
                h.seed = s.seed;
                h.local.jacobian = jacobian;
                (Algo::Hammer(h), h.rounds)
            }
        })
    }

    /// Typical candidates per evaluation batch with `n` variables: the FD
    /// Jacobian, the population, the basin-hopping screen.
    fn batch(&self, n: usize) -> usize {
        match self {
            Algo::Lsq(l) => match l.jacobian.scheme {
                FdScheme::Forward => n,
                FdScheme::Central => 2 * n,
            },
            Algo::De(d) if d.population > 0 => d.population,
            Algo::De(_) => (10 * n).max(15),
            Algo::CmaEs(c) if c.population > 0 => c.population.max(4),
            Algo::CmaEs(_) => 4 + (3.0 * (n.max(1) as f64).ln()).floor() as usize,
            Algo::NelderMead(_) => 1,
            Algo::Hammer(h) => h.candidates_per_round.max(1),
        }
    }

    fn run<P: Problem>(
        &self,
        p: &P,
        x0: &[f64],
        mon: &Monitor,
    ) -> mklight_core::MkResult<OptimResult> {
        match self {
            Algo::Lsq(s) => least_squares(p, x0, s, mon),
            Algo::De(s) => differential_evolution(p, Some(x0), s, mon),
            Algo::CmaEs(s) => cma_es(p, Some(x0), s, mon),
            Algo::NelderMead(s) => nelder_mead(p, x0, s, mon),
            Algo::Hammer(s) => hammer(p, x0, s, mon),
        }
    }
}

/// One line of the merit history.
#[derive(Clone, Serialize)]
struct HistoryRow {
    iteration: usize,
    evaluations: usize,
    stage: String,
    #[serde(serialize_with = "ser_num")]
    merit: f64,
    #[serde(serialize_with = "ser_num")]
    best_merit: f64,
    #[serde(serialize_with = "ser_num")]
    violation: f64,
}

fn ser_num<S: serde::Serializer>(x: &f64, s: S) -> Result<S::Ok, S::Error> {
    num(*x).serialize(s)
}

/// `optimize`: local or global optimisation of model parameters against a
/// merit function, on the CPU reference or GPU-batched.
pub struct Optimize;

impl Analysis for Optimize {
    fn name(&self) -> &str {
        "optimize"
    }
    fn title(&self) -> &str {
        "Optimize"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Optimises the variables against a merit function (explicit, or the default RMS spot / \
         wavefront merit with thickness boundaries) with explicit constraints, by \
         Levenberg–Marquardt, dogleg, differential evolution, CMA-ES, Nelder–Mead or basin \
         hopping. Returns the optimised system, the merit history and the operand table."
    }
    fn settings_schema(&self) -> Value {
        let default_merit =
            serde_json::to_value(DefaultMeritSpec::default()).unwrap_or(Value::Null);
        json!({
            "type": "object",
            "properties": {
                "variables": {"type": "array", "default": [], "title": "Variables",
                    "description": "Variable specs: {\"param\": ParamPath, \"lower\", \"upper\", \"scale\", \"fd_step\", \"label\"}. Empty: the curvature of every curved optical surface and the back focus."},
                "hold_efl": {"type": "boolean", "default": true, "title": "Hold focal length",
                    "description": "Hold the EFL at its start value unless the merit already has an efl operand: a constraint for lm, dogleg and hammer, a penalty residual for de, cma_es and nelder_mead."},
                "merit": {"type": ["object", "null"], "default": null, "title": "Merit function",
                    "description": "A merit-function spec; null generates the default merit from `default_merit` for the variables."},
                "default_merit": {"type": "object", "default": default_merit, "title": "Default merit",
                    "description": "Default-merit generator settings, laid over the defaults (criterion, pupil sampling, fields, wavelengths, thickness boundaries)."},
                "operands": {"type": "array", "default": [], "title": "Extra operands",
                    "description": "Operands appended to the merit (e.g. an EFL target)."},
                "constraints": {"type": "array", "default": [], "title": "Constraints",
                    "description": "Operands appended as explicit constraints (role forced to constraint)."},
                "algorithm": {"type": "string", "enum": ["lm", "dogleg", "de", "cma_es", "nelder_mead", "hammer"],
                    "default": "lm", "title": "Algorithm"},
                "options": {"type": "object", "default": {}, "title": "Algorithm options",
                    "description": "Settings of the algorithm, laid over its defaults."},
                "max_iterations": {"type": ["integer", "null"], "minimum": 1, "default": null,
                    "title": "Iterations",
                    "description": "Iterations (lm, dogleg, nelder_mead), generations (de, cma_es) or rounds (hammer); null: 200, 200, 2000, 100, 200, 20."},
                "fd_scheme": {"type": "string", "enum": ["forward", "central"], "default": "forward",
                    "title": "Finite differences"},
                "fd_relative_step": {"type": ["number", "null"], "exclusiveMinimum": 0, "default": null,
                    "title": "FD relative step",
                    "description": "null chooses per operand and backend (wavefront 1e-6/1e-4 on the CPU, GPU f32 steps; docs/OPTIMIZATION.md)."},
                "seed": {"type": "integer", "minimum": 0, "default": 0, "title": "Seed"},
                "backend": {"type": "string", "enum": ["auto", "cpu", "gpu", "gpu_emulation"],
                    "default": "auto", "title": "Merit evaluation",
                    "description": "auto: the engine's GPU device when the run is past the measured GPU/CPU crossover (rays per candidate and batch size, see backend.workload), else the CPU reference."},
                "gpu_batch": {"type": "integer", "minimum": 1, "default": 512,
                    "title": "GPU sub-batch", "description": "Candidates per GPU dispatch."},
                "feasibility_tolerance": {"type": "number", "minimum": 0, "default": 1e-6,
                    "title": "Feasibility tolerance",
                    "description": "Total normalised constraint violation regarded as feasible (best-so-far ranking, `final.feasible`)."},
                "stop_key": {"type": ["string", "null"], "default": null, "title": "Stop key",
                    "description": "Key for `request_stop`: stop and keep the best point. One running job per key."}
            }
        })
    }

    fn gpu_path(&self) -> bool {
        true
    }

    fn result_schema(&self) -> Value {
        use super::schema::*;
        let evaluation = [
            ("merit", nf()),
            ("violation", nf()),
            ("status", opaque("EvalStatus.")),
        ];
        let mut final_value = evaluation.to_vec();
        final_value.push((
            "feasible",
            boolean("Violation within feasibility_tolerance."),
        ));
        result(
            &[
                (
                    "status",
                    enumeration(&["converged", "finished", "stopped", "cancelled"], ""),
                ),
                (
                    "stopped",
                    boolean("Stop requested; best-so-far result retained."),
                ),
                ("termination", string("Optimiser termination reason.")),
                ("algorithm", string("")),
                ("iterations", int("")),
                ("evaluations", int("")),
                (
                    "improved",
                    boolean("Better than the start, feasibility first."),
                ),
                ("start", obj(&evaluation)),
                ("final", obj(&final_value)),
                (
                    "search",
                    obj(&[
                        ("merit", nf()),
                        ("violation", nf()),
                        ("reference_merit", nf()),
                        ("reference_violation", nf()),
                    ]),
                ),
                (
                    "improvement",
                    nf_u("", "1 - final/start merit; null for zero start."),
                ),
                ("multipliers", nullable(arr(nf(), "Per constraint."))),
                ("failed_variables", arr(int("Variable index."), "")),
                (
                    "variables",
                    arr(
                        obj(&[
                            ("label", string("")),
                            ("param", opaque("ParamPath; stable surface id.")),
                            ("start", nf()),
                            ("value", nf()),
                            ("change", nf_u("parameter", "value - start.")),
                            ("lower", nf()),
                            ("upper", nf()),
                            ("at_bound", boolean("")),
                        ]),
                        "Values in each parameter's model units.",
                    ),
                ),
                (
                    "operands",
                    arr(
                        obj(&[
                            ("index", int("")),
                            ("kind", string("Operand name.")),
                            ("label", string("")),
                            ("enabled", boolean("")),
                            ("role", enumeration(&["residual", "constraint"], "")),
                            (
                                "comparison",
                                enumeration(&["equal", "less_than", "greater_than"], ""),
                            ),
                            ("target", nf()),
                            ("weight", nf()),
                            ("config", int("")),
                            ("field", opaque("FieldSelector.")),
                            ("wavelength", opaque("WavelengthSelector.")),
                            ("start", nf()),
                            ("value", nf()),
                            (
                                "residual",
                                nf_u(
                                    "operand",
                                    "Signed value - target for equal; one-sided violation otherwise.",
                                ),
                            ),
                            (
                                "contribution",
                                nf_u(
                                    "merit",
                                    "weight * residual squared; null for constraints/disabled operands.",
                                ),
                            ),
                            ("contribution_fraction", nf()),
                            ("error", nullable(string("Operand failure."))),
                        ]),
                        "Start/final values on the CPU f64 reference; operand units in OPTIMIZATION.md.",
                    ),
                ),
                (
                    "history",
                    arr(
                        obj(&[
                            ("stage", string("")),
                            ("iteration", int("")),
                            ("evaluations", int("")),
                            ("merit", nf()),
                            ("violation", nf()),
                            ("best_merit", nf()),
                        ]),
                        "One record per reported iteration.",
                    ),
                ),
                ("backend", evaluation_backend_schema()),
                (
                    "merit_spec",
                    opaque("Effective MeritFunctionSpec, including generated operands."),
                ),
                (
                    "system",
                    opaque(
                        "Optimised OpticalSystem. Apply with Session::set_system / mk_session_set_system_json.",
                    ),
                ),
                (
                    "system_hash",
                    string("Hash of the optimised model, not the input snapshot."),
                ),
            ],
            vec![buffer(
                "merit_history",
                "f64",
                &["records", "4"],
                "mixed",
                "Columns: iteration, evaluations, merit, best merit (search evaluator).",
            )],
        )
    }

    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let mut s: OptimizeSettings = parse(settings, "settings")?;
        let stop = StopRequest::register(s.stop_key.as_deref(), input.stop.clone())?;
        let system = input.snapshot.system().clone();
        let cpu = TraceEvaluator::new(input.library.clone());
        let reg = OperandRegistry::builtin();
        let mut warnings = Vec::new();
        if s.variables.is_empty() {
            s.variables = auto_variables(&system);
            if s.variables.is_empty() {
                return Err(invalid(
                    "'variables': none given, and the system has nothing to vary",
                ));
            }
        }
        let default_merit = patched(
            &DefaultMeritSpec::default(),
            &s.default_merit,
            "default_merit",
        )?;
        let mut spec = build_merit(&system, &s.merit, &default_merit, &s.variables)?;
        if s.merit.is_null() {
            warnings.extend(relax_fixed_boundaries(
                &system,
                &mut spec,
                &s.variables,
                &cpu,
                &reg,
                ctx,
            )?);
        }
        spec.operands.extend(s.operands.iter().cloned());
        spec.operands
            .extend(s.constraints.iter().cloned().map(|mut o| {
                o.role = OperandRole::Constraint;
                o
            }));
        let global = s.algorithm.derivative_free();
        if global
            && spec.enabled().any(|(_, o)| {
                o.role == OperandRole::Constraint && o.comparison == Comparison::Equal
            })
        {
            warnings.push(format!(
                "{} rarely samples a point that meets an equality constraint exactly; such \
                 constraints are better held by lm, dogleg or hammer, or by a weighted operand",
                s.algorithm.name()
            ));
        }
        if s.hold_efl && !has_efl(&spec) {
            match cpu
                .evaluate_system(&system, &[&Operand::new(OperandKind::Efl)], ctx)?
                .remove(0)
            {
                Ok(f) if f.is_finite() && f != 0.0 => {
                    let o = if global {
                        // Penalty: the derivative-free methods' random trial
                        // points never meet an equality exactly, so Deb's
                        // rule would rank them by EFL error alone.
                        let phi0 = start_merit(&system, &s.variables, &spec, &cpu, &reg)?;
                        let w = if phi0 > 0.0 && phi0.is_finite() {
                            phi0
                        } else {
                            1.0
                        } / (EFL_PENALTY_RELATIVE * f).powi(2);
                        let mut o = Operand::new(OperandKind::Efl).target(f).weight(w);
                        o.label = "hold EFL (penalty)".into();
                        o
                    } else {
                        let mut o = Operand::new(OperandKind::Efl).target(f).as_constraint();
                        o.label = "hold EFL".into();
                        o
                    };
                    spec.operands.push(o);
                }
                other => warnings.push(format!(
                    "the focal length is not held: the start has no finite EFL ({other:?})"
                )),
            }
        }
        let (algo, max_iter) = Algo::new(&s)?;
        let reference = OpticalProblem::new(
            system.clone(),
            s.variables.clone(),
            spec.clone(),
            cpu.clone(),
            &reg,
        )?;
        let x0 = reference.start_point()?;
        let work = Workload::new(&cpu, &system, &spec, algo.batch(x0.len()));
        ctx.report(0.0, "qualifying the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, s.gpu_batch, work, ctx)?;
        ctx.check()?;

        // The search.
        let history: Arc<Mutex<Vec<HistoryRow>>> = Arc::default();
        let method = s.algorithm.name();
        let (run, why) = interruptible(ctx, &stop, |inner| {
            let h = history.clone();
            let job = inner.clone();
            // The optimisers' own progress calls go to a detached bar; the
            // job reports iteration and merit from the records.
            let mon_ctx = JobContext::with(inner.cancel.clone(), Progress::default());
            let mut mon = Monitor::new(mon_ctx).with_callback(move |r: &IterationRecord| {
                if let Ok(mut h) = h.lock() {
                    h.push(HistoryRow {
                        iteration: r.iteration,
                        evaluations: r.evaluations,
                        stage: r.stage.clone(),
                        merit: r.merit,
                        best_merit: r.best_merit,
                        violation: r.violation,
                    });
                }
                job.report(
                    (r.iteration + 1) as f64 / (max_iter + 1) as f64,
                    &format!("{method} {}: merit {:.6e}", r.iteration, r.best_merit),
                );
            });
            mon.keep_history = false;
            mon.feasibility_tol = s.feasibility_tolerance;
            let r = match ev.active_gpu() {
                Some(g) => OpticalProblem::new(
                    system.clone(),
                    s.variables.clone(),
                    spec.clone(),
                    g.clone().composite(),
                    &reg,
                )
                .and_then(|p| algo.run(&p, &x0, &mon)),
                None => algo.run(&reference, &x0, &mon),
            };
            (r, mon.best.get())
        });
        let (run, best) = run;
        let result = match run {
            Ok(r) => r,
            // Cancellation surfaced as an error: the best point seen.
            Err(MkError::Cancelled) => match &best {
                Some(b) => OptimResult {
                    x: b.x.clone(),
                    merit: b.merit,
                    violation: b.violation,
                    termination: Termination::Cancelled,
                    iterations: history.lock().map(|h| h.len()).unwrap_or(0),
                    evaluations: history
                        .lock()
                        .ok()
                        .and_then(|h| h.last().map(|r| r.evaluations))
                        .unwrap_or(0),
                    multipliers: None,
                    history: Vec::new(),
                    failed_variables: Vec::new(),
                },
                None => return Err(MkError::Cancelled.into()),
            },
            Err(e) => return Err(e.into()),
        };
        let feas_tol = s.feasibility_tolerance;
        // The optimiser's answer; when interrupted (or without a usable
        // answer), the better of it and the best point seen.
        let answer = ScalarEval {
            merit: result.merit,
            violation: result.violation,
            status: EvalStatus::Ok,
        };
        let (x_search, search) = match &best {
            Some(b)
                if !result.merit.is_finite()
                    || (why != Interrupt::None
                        && ScalarEval {
                            merit: b.merit,
                            violation: b.violation,
                            status: EvalStatus::Ok,
                        }
                        .better_than(&answer, feas_tol)) =>
            {
                (b.x.clone(), (b.merit, b.violation))
            }
            _ => (result.x.clone(), (result.merit, result.violation)),
        };

        // Reported numbers: CPU f64 reference. The result is never worse
        // than the start (feasibility first, then merit): the global
        // methods do not evaluate the start itself, and a GPU-driven search
        // ranks on f32 values.
        ctx.report(1.0, "reference evaluation");
        let refs = reference_values(&reference, &[&x0, &x_search])?;
        let scalar = |k: usize| ScalarEval {
            merit: refs.merit[k],
            violation: refs.violation[k],
            status: refs.status[k],
        };
        let improved = scalar(1).better_than(&scalar(0), feas_tol);
        let k = if improved { 1 } else { 0 };
        let x = if improved { x_search } else { x0.clone() };
        if !improved {
            warnings.push(format!(
                "{method} found no design better than the start (start merit {:.6e}, violation \
                 {:.3e}; best found {:.6e}, violation {:.3e}); the result is the start design",
                refs.merit[0], refs.violation[0], refs.merit[1], refs.violation[1]
            ));
        }
        let final_system = Snapshot::new(reference.apply(&x)?)?;
        ev.finish(input, s.backend, Some(&final_system), ctx)?;
        let hash = final_system.hash().to_hex();
        let history = history.lock().map(|h| h.clone()).unwrap_or_default();
        let variables: Vec<Value> = s
            .variables
            .iter()
            .zip(reference.variables())
            .enumerate()
            .map(|(j, (v, var))| {
                json!({
                    "label": var.name,
                    "param": v.param,
                    "start": num(x0[j]),
                    "value": num(x[j]),
                    "change": num(x[j] - x0[j]),
                    "lower": v.lower.map(num),
                    "upper": v.upper.map(num),
                    "at_bound": v.lower == Some(x[j]) || v.upper == Some(x[j]),
                })
            })
            .collect();
        let feasible = refs.violation[k] <= feas_tol;
        let json = json!({
            "status": interrupt_status(why, result.termination),
            "stopped": why == Interrupt::Stopped,
            "termination": result.termination,
            "algorithm": method,
            "iterations": result.iterations,
            "evaluations": result.evaluations,
            "improved": improved,
            "start": {"merit": num(refs.merit[0]), "violation": num(refs.violation[0]),
                "status": refs.status[0]},
            "final": {"merit": num(refs.merit[k]), "violation": num(refs.violation[k]),
                "status": refs.status[k], "feasible": feasible},
            "search": {"merit": num(search.0), "violation": num(search.1),
                "reference_merit": num(refs.merit[1]), "reference_violation": num(refs.violation[1])},
            "improvement": num(if refs.merit[0] > 0.0 { 1.0 - refs.merit[k] / refs.merit[0] } else { f64::NAN }),
            "multipliers": result.multipliers.as_ref().map(|m| m.iter().map(|v| num(*v)).collect::<Vec<_>>()),
            "failed_variables": result.failed_variables,
            "variables": variables,
            "operands": operand_table(&spec, &refs.values[0], &refs.values[k], refs.merit[k]),
            "history": history,
            "backend": ev.report,
            "merit_spec": spec,
            "system": final_system.system(),
            "system_hash": hash,
        });
        let mut flat = Vec::with_capacity(history.len() * 4);
        for h in &history {
            flat.extend([
                h.iteration as f64,
                h.evaluations as f64,
                h.merit,
                h.best_merit,
            ]);
        }
        let mut all_warnings = ev.warnings;
        all_warnings.extend(warnings);
        if !feasible {
            all_warnings.push(format!(
                "the final design violates its constraints (total normalised violation {:.3e})",
                refs.violation[k]
            ));
        }
        Ok(AnalysisOutput {
            json,
            buffers: vec![Buffer::new(
                "merit_history",
                vec![history.len(), 4],
                BufferData::F64(flat),
            )],
            warnings: all_warnings,
        })
    }
}

// ---------------------------------------------------------------------------
// tolerance

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ToleranceMode {
    Sensitivity,
    MonteCarlo,
    Both,
}

#[derive(Deserialize)]
struct ToleranceSettings {
    tolerances: Vec<ToleranceOperand>,
    compensators: Vec<Compensator>,
    criterion: Value,
    default_merit: Value,
    mode: ToleranceMode,
    trials: usize,
    seed: u64,
    threshold: Option<f64>,
    batch: usize,
    compensator_iterations: usize,
    backend: BackendChoice,
    gpu_batch: usize,
    stop_key: Option<String>,
}

/// The default criterion generator of `tolerance`: the default merit
/// without thickness boundaries.
fn tolerance_default_merit() -> DefaultMeritSpec {
    DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
}

/// `tolerance`: sensitivity analysis and Monte Carlo with compensators.
pub struct Tolerance;

impl Analysis for Tolerance {
    fn name(&self) -> &str {
        "tolerance"
    }
    fn title(&self) -> &str {
        "Tolerance analysis"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Sensitivity of a criterion merit (default: the RMS spot merit) to each tolerance at its \
         limits, ranked worst first with the RSS prediction, and a reproducible Monte-Carlo \
         analysis, with compensators re-optimised for every trial."
    }
    fn settings_schema(&self) -> Value {
        let default_merit = serde_json::to_value(tolerance_default_merit()).unwrap_or(Value::Null);
        json!({
            "type": "object",
            "properties": {
                "tolerances": {"type": "array", "default": [], "title": "Tolerances",
                    "description": "Tolerance operands: {\"param\": ParamPath, \"minus\", \"plus\", \"distribution\", \"label\"}. Empty: ±0.05 mm on every thickness but the back focus and ±0.5 % on every curvature."},
                "compensators": {"type": "array", "default": [], "title": "Compensators",
                    "description": "{\"param\": ParamPath, \"range\", \"scale\"}: re-optimised for every trial."},
                "criterion": {"type": ["object", "null"], "default": null, "title": "Criterion",
                    "description": "A merit-function spec; null generates the default merit for tolerancing from `default_merit`."},
                "default_merit": {"type": "object", "default": default_merit, "title": "Default criterion"},
                "mode": {"type": "string", "enum": ["sensitivity", "monte_carlo", "both"],
                    "default": "both", "title": "Analysis"},
                "trials": {"type": "integer", "minimum": 1, "default": 100, "title": "Monte-Carlo trials"},
                "seed": {"type": "integer", "minimum": 0, "default": 0, "title": "Seed"},
                "threshold": {"type": ["number", "null"], "default": null, "title": "Yield threshold",
                    "description": "Criterion value for the yield (criterion ≤ threshold)."},
                "batch": {"type": "integer", "minimum": 1, "default": 16, "title": "Trials per batch"},
                "compensator_iterations": {"type": "integer", "minimum": 1, "default": 20,
                    "title": "Compensator iterations"},
                "backend": {"type": "string", "enum": ["auto", "cpu", "gpu", "gpu_emulation"],
                    "default": "auto", "title": "Merit evaluation"},
                "gpu_batch": {"type": "integer", "minimum": 1, "default": 512, "title": "GPU sub-batch"},
                "stop_key": {"type": ["string", "null"], "default": null, "title": "Stop key"}
            }
        })
    }

    fn gpu_path(&self) -> bool {
        true
    }

    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                (
                    "status",
                    enumeration(&["complete", "stopped", "cancelled"], ""),
                ),
                (
                    "stopped",
                    boolean("Stop requested; completed sensitivity operands/trials retained."),
                ),
                (
                    "sensitivity",
                    nullable(obj(&[
                        (
                            "nominal",
                            obj(&[
                                ("criterion", nf()),
                                ("ok", boolean("")),
                                ("compensators", arr(nf(), "Parameter values.")),
                            ]),
                        ),
                        (
                            "operands",
                            arr(
                                obj(&[
                                    ("index", int("Tolerance index.")),
                                    ("label", string("")),
                                    ("param", opaque("ParamPath.")),
                                    ("minus", nf()),
                                    ("plus", nf()),
                                    ("criterion_minus", nf()),
                                    ("criterion_plus", nf()),
                                    (
                                        "change_minus",
                                        nf_u("criterion", "criterion_minus - nominal."),
                                    ),
                                    (
                                        "change_plus",
                                        nf_u("criterion", "criterion_plus - nominal."),
                                    ),
                                    (
                                        "worst",
                                        nf_u(
                                            "criterion",
                                            "Largest absolute criterion change at either limit.",
                                        ),
                                    ),
                                    ("ok", boolean("Both limits evaluated successfully.")),
                                    ("compensators_minus", arr(nf(), "")),
                                    ("compensators_plus", arr(nf(), "")),
                                ]),
                                "Completed operands in declaration order.",
                            ),
                        ),
                        ("ranking", arr(int("Tolerance index."), "Worst first.")),
                        (
                            "rss_change",
                            nf_u("criterion", "Root sum of squared worst changes."),
                        ),
                        (
                            "predicted_criterion",
                            nf_u("criterion", "Nominal + rss_change."),
                        ),
                        ("evaluations", int("")),
                        (
                            "cancelled",
                            boolean("Partial sensitivity after interruption."),
                        ),
                    ])),
                ),
                (
                    "monte_carlo",
                    nullable(obj(&[
                        ("trials", int("Completed trials.")),
                        ("failures", int("")),
                        ("seed", int("Philox seed.")),
                        ("mean", nf()),
                        ("std_dev", nf()),
                        ("min", nf()),
                        ("max", nf()),
                        (
                            "percentiles",
                            arr(obj(&[("percentile", num("%", "")), ("value", nf())]), ""),
                        ),
                        (
                            "yield_fraction",
                            nf_u(
                                "",
                                "Fraction at or below threshold; null without threshold.",
                            ),
                        ),
                        ("threshold", nf()),
                        ("compensator_mean", arr(nf(), "")),
                        ("compensator_std", arr(nf(), "")),
                        (
                            "cancelled",
                            boolean("Statistics of completed trials after interruption."),
                        ),
                        (
                            "criteria",
                            arr(nf(), "Criterion per trial; inf for failed trials."),
                        ),
                    ])),
                ),
                (
                    "compensators",
                    arr(
                        opaque("Compensator: param, range, scale."),
                        "Model parameter units.",
                    ),
                ),
                ("criterion_spec", opaque("Effective MeritFunctionSpec.")),
                ("backend", evaluation_backend_schema()),
            ],
            vec![buffer(
                "monte_carlo_criteria",
                "f64",
                &["trials"],
                "criterion",
                "Per completed Monte-Carlo trial; inf for failures. Absent without Monte Carlo.",
            )],
        )
    }

    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let mut s: ToleranceSettings = parse(settings, "settings")?;
        let stop = StopRequest::register(s.stop_key.as_deref(), input.stop.clone())?;
        let system = input.snapshot.system().clone();
        // Every parameter must exist on the nominal system (a missing one
        // would fail every trial).
        let params = s
            .tolerances
            .iter()
            .map(|t| ("tolerances", &t.param))
            .chain(s.compensators.iter().map(|c| ("compensators", &c.param)));
        for (k, (what, p)) in params.enumerate() {
            if let Err(e) = p.get(&system) {
                let i = if what == "tolerances" {
                    k
                } else {
                    k - s.tolerances.len()
                };
                return Err(invalid(format!("'{what}'[{i}]: {e}")));
            }
        }
        if s.tolerances.is_empty() {
            s.tolerances = auto_tolerances(&system);
            if s.tolerances.is_empty() {
                return Err(invalid(
                    "'tolerances': none given, and the system has nothing to tolerance",
                ));
            }
        }
        let mc = MonteCarloSettings {
            trials: s.trials,
            seed: s.seed,
            batch: s.batch.max(1),
            threshold: s.threshold,
        };
        let mut spec = ToleranceSpec::new(s.tolerances.clone());
        spec.compensators = s.compensators.clone();
        spec.monte_carlo = mc;
        spec.validate()?;
        let criterion = if s.criterion.is_null() {
            patched(
                &tolerance_default_merit(),
                &s.default_merit,
                "default_merit",
            )?
            .generate_for_tolerancing(&system, &spec)?
        } else {
            MeritFunctionSpec::from_value(s.criterion.clone())?
        };
        let reg = OperandRegistry::builtin();
        let lsq = LsqSettings {
            max_iterations: s.compensator_iterations,
            ..Default::default()
        };
        let (do_sens, do_mc) = match s.mode {
            ToleranceMode::Sensitivity => (true, false),
            ToleranceMode::MonteCarlo => (false, true),
            ToleranceMode::Both => (true, true),
        };
        // Candidates per evaluation: compensated trials run LM over the
        // compensators (FD Jacobian batches); uncompensated ones go to the
        // evaluator a Monte-Carlo batch (or sensitivity batch) at a time.
        let batch = if !spec.compensators.is_empty() {
            spec.compensators.len()
        } else if do_mc {
            mc.batch
        } else {
            SENSITIVITY_BATCH.min(2 * spec.operands.len() + 1)
        };
        let work = Workload::new(
            &TraceEvaluator::new(input.library.clone()),
            &system,
            &criterion,
            batch,
        );
        ctx.report(0.0, "qualifying the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, s.gpu_batch, work, ctx)?;
        ctx.check()?;
        let split = if do_sens && do_mc {
            0.2
        } else if do_sens {
            1.0
        } else {
            0.0
        };

        let ((sens, mcr), why) = interruptible(ctx, &stop, |inner| {
            let run = |e: &dyn ToleranceRunner| -> (Option<Result<_, MkError>>, Option<Result<_, MkError>>) {
                let sens = do_sens.then(|| {
                    let mon = Monitor::new(inner.subrange(0.0, split));
                    mon.progress(0.0, "tolerance sensitivity");
                    e.sensitivity(&spec, &mon)
                });
                let cancelled = matches!(sens, Some(Err(MkError::Cancelled)))
                    || matches!(&sens, Some(Ok(r)) if r.cancelled);
                let mcr = (do_mc && !cancelled).then(|| {
                    let mon = Monitor::new(inner.subrange(split, 1.0));
                    e.monte_carlo(&spec, &mc, &mon)
                });
                (sens, mcr)
            };
            match ev.active_gpu() {
                Some(g) => {
                    let comp = g.clone().composite();
                    run(&Runner {
                        nominal: &system,
                        criterion: &criterion,
                        evaluator: &comp,
                        registry: &reg,
                        lsq,
                        spec: &spec,
                    })
                }
                None => run(&Runner {
                    nominal: &system,
                    criterion: &criterion,
                    evaluator: &ev.cpu,
                    registry: &reg,
                    lsq,
                    spec: &spec,
                }),
            }
        });
        let sens = match sens {
            Some(Ok(r)) => Some(r),
            Some(Err(MkError::Cancelled)) | None => None,
            Some(Err(e)) => return Err(e.into()),
        };
        let mcr = match mcr {
            Some(Ok(r)) => Some(r),
            Some(Err(MkError::Cancelled)) | None => None,
            Some(Err(e)) => return Err(e.into()),
        };
        ev.finish(input, s.backend, None, ctx)?;
        let status = match why {
            Interrupt::None => "complete",
            Interrupt::Stopped => "stopped",
            Interrupt::Cancelled => "cancelled",
        };
        let label = |i: usize| -> String {
            let o = &spec.operands[i];
            if o.label.is_empty() {
                serde_json::to_string(&o.param).unwrap_or_default()
            } else {
                o.label.clone()
            }
        };
        let sens_json = sens.as_ref().map(|r| {
            json!({
                "nominal": {"criterion": num(r.nominal.criterion), "ok": r.nominal.ok,
                    "compensators": r.nominal.compensators.iter().map(|v| num(*v)).collect::<Vec<_>>()},
                "operands": r.operands.iter().map(|o| json!({
                    "index": o.index,
                    "label": label(o.index),
                    "param": spec.operands[o.index].param,
                    "minus": num(spec.operands[o.index].minus),
                    "plus": num(spec.operands[o.index].plus),
                    "criterion_minus": num(o.minus.criterion),
                    "criterion_plus": num(o.plus.criterion),
                    "change_minus": num(o.change_minus),
                    "change_plus": num(o.change_plus),
                    "worst": num(o.worst),
                    "ok": o.minus.ok && o.plus.ok,
                    "compensators_minus": o.minus.compensators.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                    "compensators_plus": o.plus.compensators.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "ranking": r.ranking,
                "rss_change": num(r.rss_change),
                "predicted_criterion": num(r.nominal.criterion + r.rss_change),
                "evaluations": r.evaluations,
                "cancelled": r.cancelled,
            })
        });
        let mut buffers = Vec::new();
        let mc_json = mcr.as_ref().map(|r| {
            buffers.push(Buffer::f64_1d("monte_carlo_criteria", r.criteria.clone()));
            json!({
                "trials": r.trials,
                "failures": r.failures,
                "seed": s.seed,
                "mean": num(r.mean),
                "std_dev": num(r.std_dev),
                "min": num(r.min),
                "max": num(r.max),
                "percentiles": r.percentiles.iter().map(|(p, v)| json!({"percentile": p, "value": num(*v)})).collect::<Vec<_>>(),
                "yield_fraction": r.yield_fraction.map(num),
                "threshold": s.threshold.map(num),
                "compensator_mean": r.compensator_mean.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                "compensator_std": r.compensator_std.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                "cancelled": r.cancelled,
                "criteria": r.criteria.iter().map(|v| num(*v)).collect::<Vec<_>>(),
            })
        });
        let json = json!({
            "status": status,
            "stopped": why == Interrupt::Stopped,
            "sensitivity": sens_json,
            "monte_carlo": mc_json,
            "compensators": spec.compensators,
            "criterion_spec": criterion,
            "backend": ev.report,
        });
        Ok(AnalysisOutput {
            json,
            buffers,
            warnings: ev.warnings,
        })
    }
}

/// Sensitivity and Monte Carlo on one evaluator type, behind a trait object
/// so the run closure is written once.
trait ToleranceRunner {
    fn sensitivity(
        &self,
        spec: &ToleranceSpec,
        mon: &Monitor,
    ) -> mklight_core::MkResult<mklight_optim::tolerance::SensitivityReport>;
    fn monte_carlo(
        &self,
        spec: &ToleranceSpec,
        mc: &MonteCarloSettings,
        mon: &Monitor,
    ) -> mklight_core::MkResult<mklight_optim::tolerance::MonteCarloReport>;
}

struct Runner<'a, E: OperandEvaluator> {
    nominal: &'a OpticalSystem,
    criterion: &'a MeritFunctionSpec,
    evaluator: &'a E,
    registry: &'a OperandRegistry,
    lsq: LsqSettings,
    spec: &'a ToleranceSpec,
}

impl<E: OperandEvaluator> Runner<'_, E> {
    fn evaluator(&self) -> OpticalToleranceEvaluator<'_, E> {
        OpticalToleranceEvaluator {
            nominal: self.nominal,
            spec: self.spec,
            criterion: self.criterion,
            evaluator: self.evaluator,
            registry: self.registry,
            lsq: self.lsq,
        }
    }
}

impl<E: OperandEvaluator> ToleranceRunner for Runner<'_, E> {
    fn sensitivity(
        &self,
        spec: &ToleranceSpec,
        mon: &Monitor,
    ) -> mklight_core::MkResult<mklight_optim::tolerance::SensitivityReport> {
        sensitivity(spec, &self.evaluator(), mon)
    }
    fn monte_carlo(
        &self,
        spec: &ToleranceSpec,
        mc: &MonteCarloSettings,
        mon: &Monitor,
    ) -> mklight_core::MkResult<mklight_optim::tolerance::MonteCarloReport> {
        monte_carlo(spec, &self.evaluator(), mc, mon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisResult, prepare, run_analysis};
    use crate::backend::{BackendInfo, DeviceHandle};
    use crate::engine::{Engine, EngineConfig};
    use crate::scheduler::JobState;
    use crate::session::AnalysisRequest;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::model::{ApertureSpec, SurfaceShape};
    use mklight_core::param::ParamPath;
    use mklight_core::{SurfaceId, samples};
    use mklight_gpu::{ComputeBackend, CpuF32Backend};
    use mklight_optim::merit::operand::OperandKind;
    use std::time::Instant;

    const LONG: Option<Duration> = Some(Duration::from_secs(600));

    fn var(p: ParamPath) -> Value {
        serde_json::to_value(VariableSpec::new(p)).unwrap()
    }
    fn curv(id: u64) -> Value {
        var(ParamPath::Curvature {
            surface: SurfaceId(id),
        })
    }
    fn thick(id: u64) -> Value {
        var(ParamPath::Thickness {
            surface: SurfaceId(id),
        })
    }
    fn efl_constraint(target: f64) -> Value {
        serde_json::to_value(Operand::new(OperandKind::Efl).target(target)).unwrap()
    }

    fn input(sys: OpticalSystem) -> AnalysisInput<'static> {
        AnalysisInput::new(
            Snapshot::new(sys).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
            BackendInfo::cpu_reference(),
        )
    }

    fn run(sys: OpticalSystem, name: &str, settings: Value) -> EngineResult<AnalysisResult> {
        let reg = AnalysisRegistry::with_builtin();
        let a = reg.get(name)?;
        run_analysis(
            a.as_ref(),
            &input(sys),
            &settings,
            vec![],
            &JobContext::new(),
        )
    }

    fn f(v: &Value) -> f64 {
        v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
    }

    fn efl(sys: &OpticalSystem) -> f64 {
        TraceEvaluator::builtin()
            .evaluate_system(sys, &[&Operand::new(OperandKind::Efl)], &JobContext::new())
            .unwrap()
            .remove(0)
            .unwrap()
    }

    /// Every constraint row of the operand table holds.
    fn constraints_hold(r: &Value, tol: f64) {
        for row in r["operands"].as_array().unwrap() {
            if row["role"] != json!("constraint") || row["enabled"] != json!(true) {
                continue;
            }
            let (v, t) = (f(&row["value"]), f(&row["target"]));
            let ok = match row["comparison"].as_str().unwrap() {
                "equal" => (v - t).abs() <= tol,
                "less_than" => v <= t + tol,
                _ => v >= t - tol,
            };
            assert!(ok, "constraint violated: {row}");
        }
    }

    /// Cooke triplet: 6 curvatures and the back focus, the default spot
    /// merit at the axis and the edge of the field in d light (kept small
    /// for unoptimised test builds), thickness boundaries (the sample's
    /// 0.99997 mm flint needs a 0.9 mm glass minimum) and an EFL constraint.
    fn triplet_settings() -> Value {
        let sys = samples::cooke_triplet();
        json!({
            "variables": [curv(1), curv(2), curv(3), curv(4), curv(5), curv(6), thick(6)],
            "default_merit": {"fields": [0, 2], "wavelengths": [1],
                "thickness": {"glass_min_centre": 0.9}},
            "constraints": [efl_constraint(efl(&sys))],
        })
    }

    /// The triplet without constraints: every accepted step improves.
    fn unconstrained_triplet() -> Value {
        json!({
            "variables": [curv(1), curv(2), curv(3), curv(4), curv(5), curv(6), thick(6)],
            "default_merit": {"fields": [0, 2], "wavelengths": [1], "thickness": null},
            "hold_efl": false,
        })
    }

    #[test]
    fn registered_with_schemas() {
        let reg = AnalysisRegistry::with_builtin();
        for name in ["optimize", "tolerance"] {
            let info = reg.list().into_iter().find(|i| i.name == name).unwrap();
            assert!(info.available, "{name}");
            assert_eq!(info.category, "optimization");
            assert!(info.default_settings.is_object());
        }
        let sys = samples::singlet();
        for bad in [
            json!({"variables": [curv(1)], "algorithm": "simplex"}),
            json!({"variables": [curv(1)], "options": {"max_iterations": "many"}}),
            json!({"variables": [{"param": {"param": "nope"}}]}),
        ] {
            assert!(
                matches!(
                    run(sys.clone(), "optimize", bad.clone()),
                    Err(EngineError::InvalidSettings(_))
                ),
                "accepted {bad}"
            );
        }
    }

    /// Singlet bending at fixed EFL through the scheduler: the merit drops,
    /// the EFL constraint holds, and the returned system applies as a new
    /// snapshot.
    #[test]
    fn optimize_singlet_through_the_scheduler() {
        let engine = Engine::new(EngineConfig {
            worker_threads: 2,
            ..Default::default()
        });
        let mut session = engine.new_session();
        let mut sys = samples::singlet();
        sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
        sys.surfaces[1].shape = SurfaceShape::sphere(100.0);
        sys.surfaces[2].shape = SurfaceShape::sphere(-100.0);
        sys.surfaces[1].thickness = 3.0;
        session.set_system(sys).unwrap();
        let settings = json!({
            "variables": [curv(1), curv(2), thick(2)],
            "default_merit": {"fields": [0], "wavelengths": [1], "thickness": null},
            "constraints": [efl_constraint(100.0)],
        });
        let job = session
            .submit_analysis(&AnalysisRequest::new("optimize", settings))
            .unwrap();
        assert_eq!(engine.wait_job(job, LONG).unwrap(), JobState::Done);
        let r = engine.take_result(job).unwrap();
        let j = &r.json;
        assert_eq!(j["status"], json!("converged"), "{}", j["termination"]);
        let (m0, m1) = (f(&j["start"]["merit"]), f(&j["final"]["merit"]));
        assert!(m1 < 0.2 * m0, "{m0} -> {m1}");
        assert!(f(&j["final"]["violation"]) < 1e-8);
        assert_eq!(j["final"]["feasible"], json!(true));
        constraints_hold(j, 1e-6);
        // The merit history, also as a buffer.
        let h = j["history"].as_array().unwrap();
        assert!(!h.is_empty());
        assert!(h.iter().all(|r| f(&r["merit"]).is_finite()));
        assert_eq!(r.buffers[0].name, "merit_history");
        assert_eq!(r.buffers[0].shape, vec![h.len(), 4]);
        assert_eq!(r.run_info.analysis, "optimize");
        assert_eq!(r.run_info.seed, Some(0));
        assert_eq!(j["backend"]["evaluated_on"], json!(TraceEvaluator::BACKEND));
        // Apply the result as a new snapshot.
        let best: OpticalSystem = serde_json::from_value(j["system"].clone()).unwrap();
        let hash = session.set_system(best.clone()).unwrap();
        assert_eq!(hash.to_hex(), j["system_hash"].as_str().unwrap());
        assert!((efl(&best) - 100.0).abs() < 1e-6);
        // The variables moved (a bent lens, not the equi-convex start).
        let v = j["variables"].as_array().unwrap();
        assert!(f(&v[0]["change"]).abs() > 1e-4, "{}", v[0]);
    }

    /// Cooke triplet, default merit with thickness boundaries and an EFL
    /// constraint: LM lowers the merit, every constraint holds, and the run
    /// is deterministic; a seeded DE run is reproducible and seed-dependent.
    #[test]
    fn optimize_cooke_triplet_deterministic() {
        let mut s = triplet_settings();
        s["max_iterations"] = json!(40);
        // The full default merit (every field and wavelength).
        s["default_merit"] = json!({"thickness": {"glass_min_centre": 0.9}});
        let a = run(samples::cooke_triplet(), "optimize", s.clone()).unwrap();
        let j = &a.json;
        let (m0, m1) = (f(&j["start"]["merit"]), f(&j["final"]["merit"]));
        assert!(m1 < 0.8 * m0, "{m0} -> {m1}");
        assert!(f(&j["final"]["violation"]) < 1e-6, "{}", j["final"]);
        constraints_hold(j, 1e-6);
        // Thickness boundaries are among the constraints.
        let n_cons = j["operands"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["role"] == json!("constraint"))
            .count();
        assert!(n_cons > 5, "{n_cons}");
        let b = run(samples::cooke_triplet(), "optimize", s).unwrap();
        assert_eq!(a.json, b.json, "same settings, same result");

        let de = |seed: u64| {
            // Unconstrained: an explicit EFL equality constraint would leave
            // every DE trial point infeasible (the job's own EFL hold is a
            // penalty for DE).
            let mut s = unconstrained_triplet();
            s["algorithm"] = json!("de");
            s["max_iterations"] = json!(3);
            s["seed"] = json!(seed);
            s["options"] = json!({"population": 10, "init_spread": 0.05});
            run(samples::cooke_triplet(), "optimize", s).unwrap()
        };
        let (x, y, z) = (de(11), de(11), de(12));
        assert_eq!(x.json, y.json);
        assert_eq!(x.run_info.seed, Some(11));
        assert_ne!(x.json["system"], z.json["system"]);
        assert!(f(&x.json["final"]["merit"]) < f(&x.json["start"]["merit"]));
    }

    /// Wait until the job context reports iteration `n`, then cancel.
    fn cancel_at(ctx: &JobContext, n: usize) -> std::thread::JoinHandle<()> {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(600) {
                let st = ctx.progress.stage();
                if let Some(it) = st.strip_prefix("lm ").and_then(|r| r.split(':').next())
                    && it.parse::<usize>().is_ok_and(|i| i >= n)
                {
                    ctx.cancel.cancel();
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    }

    /// Cancelling mid-run returns the best point so far (`Analysis::run`);
    /// the scheduler discards cancelled jobs, so a `stop_key` stops a job
    /// and keeps its result.
    #[test]
    fn cancel_and_stop_return_best_so_far() {
        let mut s = unconstrained_triplet();
        s["max_iterations"] = json!(200);
        // Direct run, cancelled after two iterations.
        let settings = prepare(&Optimize, &s).unwrap();
        let ctx = JobContext::new();
        let w = cancel_at(&ctx, 2);
        let out = Optimize
            .run(&input(samples::cooke_triplet()), &settings, &ctx)
            .unwrap();
        w.join().unwrap();
        let j = &out.json;
        assert_eq!(j["status"], json!("cancelled"), "{}", j["termination"]);
        assert!(j["iterations"].as_u64().unwrap() < 200);
        let (m0, m1) = (f(&j["start"]["merit"]), f(&j["final"]["merit"]));
        assert!(m1 < m0, "{m0} -> {m1}");
        serde_json::from_value::<OpticalSystem>(j["system"].clone()).unwrap();

        // The legacy key and the job-level stop both retain the best point.
        for keyed in [true, false] {
            assert!(!request_stop("no-such-job"));
            let engine = Engine::new(EngineConfig {
                worker_threads: 1,
                ..Default::default()
            });
            let mut session = engine.new_session();
            session.set_system(samples::cooke_triplet()).unwrap();
            s["stop_key"] = if keyed {
                json!("test-stop-triplet")
            } else {
                Value::Null
            };
            let job = session
                .submit_analysis(&AnalysisRequest::new("optimize", s.clone()))
                .unwrap();
            let t = Instant::now();
            loop {
                let st = engine.job_status(job).unwrap();
                if st.stage.starts_with("lm 2") || st.state.is_terminal() {
                    break;
                }
                assert!(t.elapsed() < Duration::from_secs(600));
                std::thread::sleep(Duration::from_millis(1));
            }
            if keyed {
                assert!(request_stop("test-stop-triplet"));
            } else {
                engine.stop_job(job).unwrap();
            }
            assert_eq!(engine.wait_job(job, LONG).unwrap(), JobState::Done);
            let r = engine.take_result(job).unwrap();
            assert_eq!(r.json["status"], json!("stopped"));
            assert_eq!(r.json["stopped"], true);
            assert!(r.json["iterations"].as_u64().unwrap() < 200);
            assert!(f(&r.json["final"]["merit"]) < f(&r.json["start"]["merit"]));
            assert!(
                !request_stop("test-stop-triplet"),
                "unregistered after the run"
            );
        }
    }

    /// Sensitivity and Monte Carlo on the Cooke triplet with a back-focus
    /// compensator: ranked sensitivities, reproducible Monte Carlo.
    #[test]
    fn tolerance_cooke_triplet() {
        let sys = samples::cooke_triplet();
        let t = |id: u64, tol: f64| json!({"param": ParamPath::Thickness { surface: SurfaceId(id) }, "minus": tol, "plus": tol});
        let c = |id: u64| {
            let p = ParamPath::Curvature {
                surface: SurfaceId(id),
            };
            let tol = 0.005 * p.get(&sys).unwrap().abs();
            json!({"param": p, "minus": tol, "plus": tol})
        };
        let settings = |seed: u64| {
            json!({
                "tolerances": [t(1, 0.05), t(2, 0.05), t(4, 0.05), c(1), c(6),
                    {"param": ParamPath::DecenterY { surface: SurfaceId(3) }, "minus": 0.02, "plus": 0.02}],
                "compensators": [{"param": ParamPath::Thickness { surface: SurfaceId(6) }, "range": 1.0}],
                "trials": 6,
                "batch": 3,
                "seed": seed,
                "compensator_iterations": 8,
            })
        };
        let a = run(sys.clone(), "tolerance", settings(3)).unwrap();
        let j = &a.json;
        assert_eq!(j["status"], json!("complete"));
        let sens = &j["sensitivity"];
        assert_eq!(sens["nominal"]["ok"], json!(true));
        let ops = sens["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 6);
        assert!(ops.iter().all(|o| o["ok"] == json!(true)));
        let rank: Vec<usize> = serde_json::from_value(sens["ranking"].clone()).unwrap();
        for w in rank.windows(2) {
            assert!(f(&ops[w[0]]["worst"]) >= f(&ops[w[1]]["worst"]));
        }
        assert!(f(&sens["rss_change"]) > 0.0);
        // The air space before the last element is refocused.
        let comp = f(&ops[2]["compensators_plus"][0]);
        assert!((comp - sys.surfaces[6].thickness).abs() > 1e-6);
        let mc = &j["monte_carlo"];
        assert_eq!(mc["trials"], json!(6));
        assert_eq!(mc["failures"], json!(0));
        assert!(f(&mc["mean"]) >= f(&sens["nominal"]["criterion"]) * (1.0 - 1e-9));
        assert_eq!(a.buffers[0].name, "monte_carlo_criteria");
        assert_eq!(a.run_info.seed, Some(3));
        let b = run(sys.clone(), "tolerance", settings(3)).unwrap();
        assert_eq!(mc["criteria"], b.json["monte_carlo"]["criteria"]);
        let other = run(sys.clone(), "tolerance", settings(4)).unwrap();
        assert_ne!(mc["criteria"], other.json["monte_carlo"]["criteria"]);
    }

    /// The GPU path in the engine: the emulation backend everywhere, the
    /// Metal device when present. Search on the GPU, reported numbers from
    /// the CPU reference, qualification in the result.
    #[test]
    fn optimize_on_gpu_backends() {
        let handle = |b: Arc<dyn ComputeBackend>| -> DeviceHandle {
            let id = if b.info().backend_id().starts_with("metal:") {
                "gpu:metal"
            } else {
                "gpu:cpu_f32"
            };
            Arc::new(GpuDevice::new(id, b))
        };
        let emulation = handle(Arc::new(CpuF32Backend::new()));
        let metal = mklight_metal::MetalBackend::new()
            .ok()
            .flatten()
            .map(|m| handle(Arc::new(m)));
        if metal.is_none() {
            eprintln!("SKIP (Metal part): no GPU device");
        }
        let mut cases = vec![
            ("gpu_emulation", Some(emulation.clone())),
            ("gpu", Some(emulation.clone())),
        ];
        if let Some(m) = metal {
            cases.push(("gpu", Some(m)));
        }
        for (backend, device) in cases {
            let mut s = unconstrained_triplet();
            s["max_iterations"] = json!(4);
            s["backend"] = json!(backend);
            let reg = AnalysisRegistry::with_builtin();
            let a = reg.get("optimize").unwrap();
            let is_metal = device.as_ref().is_some_and(|d| {
                d.downcast_ref::<GpuDevice>()
                    .is_some_and(|g| g.id == "gpu:metal")
            });
            let inp = input(samples::cooke_triplet()).with_device(device);
            let r = run_analysis(a.as_ref(), &inp, &s, vec![], &JobContext::new()).unwrap();
            let b = &r.json["backend"];
            assert_eq!(b["precision"], json!("f32_gpu"), "{backend}: {b}");
            assert_eq!(b["qualification"]["passed"], json!(true));
            assert_eq!(b["final_qualification"]["passed"], json!(true));
            assert!(b["gpu_stats"]["gpu_candidates"].as_u64().unwrap() > 0);
            assert_eq!(b["reported_on"], json!(TraceEvaluator::BACKEND));
            assert_eq!(b["requested"], json!(backend));
            assert_eq!(
                b["evaluated_on"].as_str().unwrap().starts_with("metal:"),
                is_metal
            );
            assert!(
                r.run_info
                    .warnings
                    .iter()
                    .any(|w| w.contains("evaluated on")),
                "{:?}",
                r.run_info.warnings
            );
            let (m0, m1) = (f(&r.json["start"]["merit"]), f(&r.json["final"]["merit"]));
            assert!(m1 < m0, "{backend}: {m0} -> {m1}");
        }
        // `auto` with a device: the CPU reference below the measured
        // crossover (the default merit's 38 rays per candidate, LM batches
        // of 7), the GPU above it (12 × 24 pupil, every field and
        // wavelength, DE populations of 16).
        let auto = |s: Value| {
            let reg = AnalysisRegistry::with_builtin();
            let a = reg.get("optimize").unwrap();
            let inp = input(samples::cooke_triplet())
                .with_device(Some(emulation.clone()))
                .with_selection("auto", None);
            run_analysis(a.as_ref(), &inp, &s, vec![], &JobContext::new()).unwrap()
        };
        let mut s = unconstrained_triplet();
        s["max_iterations"] = json!(2);
        s["backend"] = json!("auto");
        let r = auto(s.clone());
        let b = &r.json["backend"];
        assert_eq!(b["precision"], json!("f64_reference"), "{b}");
        assert_eq!(b["workload"]["gpu_faster"], json!(false));
        assert_eq!(b["workload"]["batch"], json!(7));
        s["algorithm"] = json!("de");
        s["max_iterations"] = json!(1);
        s["options"] = json!({"population": 16});
        s["default_merit"] = json!({"sampling": {"rings": 12, "arms": 24}, "thickness": null});
        let r = auto(s);
        let b = &r.json["backend"];
        assert_eq!(b["precision"], json!("f32_gpu"), "{b}");
        assert_eq!(b["workload"]["batch"], json!(16));
        assert!(b["workload"]["rays_per_candidate"].as_u64().unwrap() >= 2600);
        assert!(b["gpu_stats"]["gpu_candidates"].as_u64().unwrap() >= 16);

        // `gpu` without a device: the CPU reference, and a warning that says
        // why.
        let mut s = unconstrained_triplet();
        s["max_iterations"] = json!(1);
        s["backend"] = json!("gpu");
        let r = run(samples::cooke_triplet(), "optimize", s).unwrap();
        assert_eq!(r.json["backend"]["precision"], json!("f64_reference"));
        assert!(
            r.run_info
                .warnings
                .iter()
                .any(|w| w.contains("GPU not used")),
            "{:?}",
            r.run_info.warnings
        );
    }

    /// The EFL row of an operand table: (target, value).
    fn efl_row(j: &Value) -> (f64, f64) {
        let row = j["operands"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["kind"] == json!("efl"))
            .unwrap();
        (f(&row["target"]), f(&row["value"]))
    }

    /// The global methods with the job's defaults (automatic variables, EFL
    /// held, thickness boundaries) never return a design worse than the
    /// start, and here improve it: the EFL hold is a penalty for them (their
    /// trial points never meet an equality), and their spreads around the
    /// start are small. Before, CMA-ES returned the singlet 16× worse and DE
    /// spent 7070 evaluations on the triplet without moving.
    #[test]
    fn global_methods_never_return_a_worse_design() {
        for (name, sys) in [
            ("singlet", samples::singlet()),
            ("cooke_triplet", samples::cooke_triplet()),
        ] {
            for algo in ["cma_es", "de", "nelder_mead"] {
                let r = run(sys.clone(), "optimize", json!({ "algorithm": algo })).unwrap();
                let j = &r.json;
                let (m0, m1) = (f(&j["start"]["merit"]), f(&j["final"]["merit"]));
                assert!(m1 < m0, "{name} {algo}: {m0} -> {m1}");
                assert_eq!(j["improved"], json!(true));
                assert_eq!(j["final"]["feasible"], json!(true), "{name} {algo}");
                let (t, v) = efl_row(j);
                assert!(((v - t) / t).abs() < 1e-4, "{name} {algo}: EFL {t} -> {v}");
                let held = j["operands"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|o| o["kind"] == json!("efl"))
                    .unwrap();
                assert_eq!(held["role"], json!("residual"));
                assert_eq!(held["label"], json!("hold EFL (penalty)"));
            }
        }
        // A search that cannot improve (one generation of a wide population
        // around a good design) returns the start, and says so.
        let r = run(
            samples::double_gauss(),
            "optimize",
            json!({"algorithm": "de", "max_iterations": 1,
                "options": {"population": 6, "init_spread": 1.0}}),
        )
        .unwrap();
        let j = &r.json;
        assert_eq!(j["improved"], json!(false));
        assert_eq!(j["final"]["merit"], j["start"]["merit"]);
        assert_eq!(j["final"]["violation"], j["start"]["violation"]);
        assert!(
            j["variables"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["change"] == json!(0.0))
        );
        assert!(
            r.run_info
                .warnings
                .iter()
                .any(|w| w.contains("no design better than the start")),
            "{:?}",
            r.run_info.warnings
        );
        let back: OpticalSystem = serde_json::from_value(j["system"].clone()).unwrap();
        assert_eq!(
            Snapshot::new(back).unwrap().hash(),
            Snapshot::new(samples::double_gauss()).unwrap().hash()
        );
    }

    /// `optimize` with no settings ends feasible on the samples that used to
    /// end infeasible: the triplet's fixed 0.99997 mm flint violated the
    /// default 1 mm glass minimum (now relaxed to its start value, with a
    /// warning), and the Petzval and relay EFL holds ran out of iterations
    /// in the first augmented-Lagrangian round.
    #[test]
    fn defaults_end_feasible() {
        let r = run(samples::cooke_triplet(), "optimize", json!({})).unwrap();
        let j = &r.json;
        assert_eq!(j["final"]["feasible"], json!(true), "{}", j["final"]);
        assert!(f(&j["final"]["merit"]) < f(&j["start"]["merit"]));
        let relaxed: Vec<&Value> = j["operands"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["label"].as_str().unwrap().contains("relaxed"))
            .collect();
        assert_eq!(relaxed.len(), 1, "{relaxed:?}");
        assert!((f(&relaxed[0]["target"]) - 0.99997).abs() < 1e-12);
        assert!(
            r.run_info.warnings.iter().any(|w| w.contains("relaxed")),
            "{:?}",
            r.run_info.warnings
        );
        constraints_hold(j, 1e-6);
        for sys in [samples::petzval_asphere(), samples::finite_relay()] {
            let r = run(sys, "optimize", json!({})).unwrap();
            let j = &r.json;
            assert_eq!(j["final"]["feasible"], json!(true), "{}", j["final"]);
            assert!(f(&j["final"]["merit"]) < f(&j["start"]["merit"]));
            let (t, v) = efl_row(j);
            assert!((v - t).abs() < 1e-6, "EFL {t} -> {v}");
        }
    }

    /// Bad input that used to pass silently: a zero FD step, a tolerance or
    /// compensator on a surface that does not exist, a stop key in use.
    #[test]
    fn rejects_bad_input() {
        let sys = samples::cooke_triplet();
        for step in [0.0, -1e-3] {
            assert!(matches!(
                run(sys.clone(), "optimize", json!({"fd_relative_step": step})),
                Err(EngineError::InvalidSettings(_))
            ));
        }
        let missing = ParamPath::Thickness {
            surface: SurfaceId(999),
        };
        for s in [
            json!({"tolerances": [{"param": missing, "minus": 0.1, "plus": 0.1}], "trials": 2}),
            json!({"compensators": [{"param": missing, "range": 1.0}], "trials": 2}),
        ] {
            match run(sys.clone(), "tolerance", s.clone()) {
                Err(EngineError::InvalidSettings(m)) => assert!(m.contains("[0]"), "{m}"),
                other => panic!(
                    "accepted {s}: {:?}",
                    other.map(|r| r.json["status"].clone())
                ),
            }
        }
        // A second job may not take a running job's stop key.
        let held = StopRequest::register(Some("test-key-in-use"), CancelToken::new()).unwrap();
        for (name, s) in [
            (
                "optimize",
                json!({"stop_key": "test-key-in-use", "max_iterations": 1}),
            ),
            (
                "tolerance",
                json!({"stop_key": "test-key-in-use", "trials": 1}),
            ),
        ] {
            assert!(matches!(
                run(sys.clone(), name, s),
                Err(EngineError::InvalidSettings(_))
            ));
        }
        assert!(request_stop("test-key-in-use"));
        assert!(held.token.is_cancelled());
        drop(held);
        assert!(!request_stop("test-key-in-use"));
    }

    /// Stopping the tolerance job in the sensitivity phase keeps the
    /// operands evaluated so far.
    #[test]
    fn stopped_sensitivity_keeps_completed_operands() {
        let sys = samples::cooke_triplet();
        let settings = prepare(
            &Tolerance,
            &json!({"mode": "sensitivity", "compensators": [{"param": ParamPath::Thickness { surface: SurfaceId(6) }, "range": 1.0}],
                "compensator_iterations": 4, "stop_key": "test-stop-sensitivity"}),
        )
        .unwrap();
        // Stop once the first sensitivity batch has been reported.
        let ctx = JobContext::new();
        let watch = {
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let t = Instant::now();
                while t.elapsed() < Duration::from_secs(600) {
                    if ctx.progress.fraction() > 0.0 && request_stop("test-stop-sensitivity") {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        // 30 tolerances: 61 trials, two sensitivity batches of 32 and 29.
        let mut tol = Vec::new();
        for id in [1u64, 2, 3, 4, 5, 6] {
            for p in [
                ParamPath::DecenterX {
                    surface: SurfaceId(id),
                },
                ParamPath::DecenterY {
                    surface: SurfaceId(id),
                },
                ParamPath::TiltX {
                    surface: SurfaceId(id),
                },
                ParamPath::TiltY {
                    surface: SurfaceId(id),
                },
                ParamPath::Curvature {
                    surface: SurfaceId(id),
                },
            ] {
                tol.push(json!({"param": p, "minus": 1e-3, "plus": 1e-3}));
            }
        }
        let mut settings = settings;
        settings["tolerances"] = json!(tol);
        let out = Tolerance.run(&input(sys), &settings, &ctx).unwrap();
        watch.join().unwrap();
        let j = &out.json;
        assert_eq!(j["status"], json!("stopped"));
        let sens = &j["sensitivity"];
        assert_eq!(sens["cancelled"], json!(true), "{sens}");
        let n = sens["operands"].as_array().unwrap().len();
        assert!(n > 0 && n < 30, "{n} of 30 operands");
        assert_eq!(sens["ranking"].as_array().unwrap().len(), n);
    }

    /// With no variables / tolerances the jobs choose them: the curvatures of
    /// the curved optical surfaces and the back focus with the EFL held, and
    /// ±0.05 mm / ±0.5 % tolerances.
    #[test]
    fn defaults_choose_variables_and_tolerances() {
        let sys = samples::singlet();
        let r = run(sys.clone(), "optimize", json!({"max_iterations": 3})).unwrap();
        let params: Vec<Value> = r.json["variables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["param"].clone())
            .collect();
        assert_eq!(
            params,
            vec![
                json!(ParamPath::Curvature {
                    surface: SurfaceId(1)
                }),
                json!(ParamPath::Curvature {
                    surface: SurfaceId(2)
                }),
                json!(ParamPath::Thickness {
                    surface: SurfaceId(2)
                }),
            ]
        );
        let held: Vec<&Value> = r.json["operands"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["kind"] == json!("efl"))
            .collect();
        assert_eq!(held.len(), 1);
        assert_eq!(held[0]["role"], json!("constraint"));
        assert!((f(&held[0]["target"]) - efl(&sys)).abs() < 1e-12);
        // An explicit EFL operand replaces the hold.
        let r = run(
            sys.clone(),
            "optimize",
            json!({"max_iterations": 1, "operands": [efl_constraint(100.0)]}),
        )
        .unwrap();
        let efls = r.json["operands"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["kind"] == json!("efl"))
            .count();
        assert_eq!(efls, 1);

        let t = run(sys.clone(), "tolerance", json!({"mode": "sensitivity"})).unwrap();
        let ops = t.json["sensitivity"]["operands"].as_array().unwrap();
        // The lens thickness (not the back focus) and both curvatures.
        assert_eq!(ops.len(), 3, "{ops:?}");
        assert_eq!(f(&ops[0]["plus"]), 0.05);
        let c1 = ParamPath::Curvature {
            surface: SurfaceId(1),
        }
        .get(&sys)
        .unwrap();
        assert!((f(&ops[1]["plus"]) - 0.005 * c1.abs()).abs() < 1e-15);
        assert!(t.json["monte_carlo"].is_null());
    }
}
