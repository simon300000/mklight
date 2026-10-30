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
//!   stale). To stop a job *and keep* its best point, give it a `stop_key`
//!   and call [`request_stop`]: the job then ends normally with
//!   `"status": "stopped"`.
//! * **Reported numbers** (start and final merit, operand table) are always
//!   computed with the CPU f64 reference, whatever evaluated the search; the
//!   result says which backend evaluated it, with its precision
//!   qualification, and `RunInfo.warnings` repeats it.
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
use mklight_core::{JobContext, MkError, OpticalSystem, Snapshot};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_optim::deriv::{FdScheme, JacobianSettings};
use mklight_optim::global::cmaes::{CmaEsSettings, cma_es};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::global::nelder_mead::{NelderMeadSettings, nelder_mead};
use mklight_optim::lsq::{LsqMethod, LsqSettings, least_squares};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::{
    GpuEvalSettings, GpuTraceEvaluator, MeritFunctionSpec, Operand, OperandEvaluator,
    OperandRegistry, OperandRole, OperandValue, OpticalProblem, TraceEvaluator, VariableSpec,
};
use mklight_optim::problem::{
    CandidateBatch, EvalStatus, Problem, ScalarEval, merit, total_violation,
};
use mklight_optim::tolerance::{
    Compensator, MonteCarloSettings, OpticalToleranceEvaluator, ToleranceOperand, ToleranceSpec,
    monte_carlo, sensitivity,
};
use mklight_optim::{IterationRecord, Monitor, OptimResult, Termination};

use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::backend::DeviceHandle;
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};

/// Register `optimize` and `tolerance`.
pub fn register(r: &mut AnalysisRegistry) {
    r.register(Arc::new(Optimize));
    r.register(Arc::new(Tolerance));
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
    fn register(key: &str, token: CancelToken) -> StopGuard {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut g = stops().lock().unwrap_or_else(|e| e.into_inner());
        g.insert(key.to_string(), StopEntry { id, token });
        StopGuard {
            key: key.to_string(),
            id,
        }
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

/// Run `f` with a context that is cancelled when `ctx` is or when a stop is
/// requested for `stop_key` (a watcher thread joins the two tokens), and
/// say which happened.
fn interruptible<T>(
    ctx: &JobContext,
    stop_key: Option<&str>,
    f: impl FnOnce(&JobContext) -> T,
) -> (T, Interrupt) {
    let stop = CancelToken::new();
    let _guard = stop_key.map(|k| StopGuard::register(k, stop.clone()));
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
    /// The GPU when the engine handed this analysis a GPU device, else the
    /// CPU reference.
    Auto,
    Cpu,
    /// The engine's GPU device, else the platform's default GPU (Metal);
    /// the CPU reference (with a warning) when there is none or it fails
    /// qualification.
    Gpu,
    /// The CPU-f32 emulation of the GPU kernels (validation and tests).
    GpuEmulation,
}

/// The compute interface behind an engine device handle.
fn device_backend(dev: &DeviceHandle) -> Option<Arc<dyn ComputeBackend>> {
    if let Some(b) = dev.downcast_ref::<Arc<dyn ComputeBackend>>() {
        return Some(b.clone());
    }
    #[cfg(target_os = "macos")]
    if let Ok(b) = dev.clone().downcast::<mklight_metal::MetalBackend>() {
        return Some(b);
    }
    if let Ok(b) = dev.clone().downcast::<CpuF32Backend>() {
        return Some(b);
    }
    None
}

/// The platform's default GPU backend, created once.
fn default_gpu() -> Option<Arc<dyn ComputeBackend>> {
    static B: OnceLock<Option<Arc<dyn ComputeBackend>>> = OnceLock::new();
    B.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            mklight_metal::MetalBackend::new()
                .ok()
                .flatten()
                .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>)
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    })
    .clone()
}

/// The GPU backend for a request, or why there is none.
fn gpu_backend(
    choice: BackendChoice,
    input: &AnalysisInput<'_>,
) -> Result<Option<Arc<dyn ComputeBackend>>, String> {
    let engine = input.device.as_ref().and_then(device_backend);
    match choice {
        BackendChoice::Cpu => Ok(None),
        BackendChoice::Auto => Ok(engine),
        BackendChoice::GpuEmulation => Ok(Some(Arc::new(CpuF32Backend::new()))),
        BackendChoice::Gpu => engine
            .or_else(default_gpu)
            .map(Some)
            .ok_or_else(|| "no GPU device on this machine".to_string()),
    }
}

/// The merit evaluator of a run: the CPU reference alone, or a qualified
/// GPU evaluator in front of it.
struct Evaluation {
    cpu: TraceEvaluator,
    gpu: Option<GpuTraceEvaluator>,
    /// `"backend"` block of the result.
    report: Value,
    warnings: Vec<String>,
}

impl Evaluation {
    fn new(
        choice: BackendChoice,
        input: &AnalysisInput<'_>,
        system: &OpticalSystem,
        batch: usize,
        ctx: &JobContext,
    ) -> EngineResult<Self> {
        let cpu = TraceEvaluator::new(input.library.clone());
        let mut warnings = Vec::new();
        let gpu = match gpu_backend(choice, input) {
            Ok(Some(b)) => {
                let settings = GpuEvalSettings {
                    batch: batch.max(1),
                    ..Default::default()
                };
                let g = GpuTraceEvaluator::new(b, cpu.clone(), system, settings, ctx)?;
                if g.is_active() {
                    warnings.push(format!(
                        "ray operands evaluated on {} (f32, precision-qualified); reported values \
                         from the CPU f64 reference",
                        g.backend_id()
                    ));
                } else {
                    warnings.push(format!(
                        "GPU backend {} not used ({}); computed with the CPU f64 reference",
                        g.backend_id(),
                        g.qualification().reason.unwrap_or_default()
                    ));
                }
                Some(g)
            }
            Ok(None) => None,
            Err(why) => {
                warnings.push(format!(
                    "GPU not used ({why}); computed with the CPU f64 reference"
                ));
                None
            }
        };
        Ok(Evaluation {
            cpu,
            gpu,
            report: Value::Null,
            warnings,
        })
    }

    fn active_gpu(&self) -> Option<&GpuTraceEvaluator> {
        self.gpu.as_ref().filter(|g| g.is_active())
    }

    /// The `"backend"` block: who evaluated the search, its qualification
    /// (at the start and, after a GPU run, at the final design) and the GPU
    /// work counters.
    fn finish(&mut self, requested: BackendChoice, final_system: &OpticalSystem) {
        let mut b = json!({
            "requested": format!("{requested:?}").to_lowercase(),
            "evaluated_on": TraceEvaluator::BACKEND,
            "precision": Precision::F64Reference,
            "reported_on": TraceEvaluator::BACKEND,
        });
        if let Some(g) = &self.gpu {
            let start = g.qualification();
            if g.is_active() {
                b["evaluated_on"] = json!(g.backend_id());
                b["precision"] = json!(Precision::F32Gpu);
                // A qualification of the final design (the search moved
                // away from the qualified start).
                if let Ok(q) = g.requalify(final_system, &JobContext::new()) {
                    if !q.passed {
                        self.warnings.push(format!(
                            "the final design fails the GPU precision qualification ({}); its \
                             reported values come from the CPU f64 reference",
                            q.reason.clone().unwrap_or_default()
                        ));
                    }
                    b["final_qualification"] = serde_json::to_value(&q).unwrap_or(Value::Null);
                }
            }
            b["qualification"] = serde_json::to_value(&start).unwrap_or(Value::Null);
            b["gpu_stats"] = serde_json::to_value(g.stats()).unwrap_or(Value::Null);
        }
        self.report = b;
    }
}

/// The merit function of a run: an explicit spec, or the default merit
/// generated for the variables, plus extra operands and constraints.
fn build_merit(
    system: &OpticalSystem,
    merit_spec: &Value,
    default_merit: &DefaultMeritSpec,
    variables: &[VariableSpec],
    operands: &[Operand],
    constraints: &[Operand],
) -> EngineResult<MeritFunctionSpec> {
    let mut spec = if merit_spec.is_null() {
        default_merit.generate_for(system, variables)?
    } else {
        MeritFunctionSpec::from_value(merit_spec.clone())?
    };
    spec.operands.extend(operands.iter().cloned());
    spec.operands
        .extend(constraints.iter().cloned().map(|mut o| {
            o.role = OperandRole::Constraint;
            o
        }));
    Ok(spec)
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
                let mut h = patched(
                    &HammerSettings {
                        rounds: iters,
                        ..Default::default()
                    },
                    opts,
                    "options",
                )?;
                h.seed = s.seed;
                h.local.jacobian = jacobian;
                (Algo::Hammer(h), h.rounds)
            }
        })
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
                "variables": {"type": "array", "title": "Variables",
                    "description": "Variable specs: {\"param\": ParamPath, \"lower\", \"upper\", \"scale\", \"fd_step\", \"label\"}. At least one."},
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
                "fd_relative_step": {"type": ["number", "null"], "minimum": 0, "default": null,
                    "title": "FD relative step",
                    "description": "null chooses per operand and backend (wavefront 1e-6/1e-4 on the CPU, GPU f32 steps; docs/OPTIMIZATION.md)."},
                "seed": {"type": "integer", "minimum": 0, "default": 0, "title": "Seed"},
                "backend": {"type": "string", "enum": ["auto", "cpu", "gpu", "gpu_emulation"],
                    "default": "auto", "title": "Merit evaluation",
                    "description": "auto: the GPU when the engine selected a GPU backend, else the CPU reference."},
                "gpu_batch": {"type": "integer", "minimum": 1, "default": 512,
                    "title": "GPU sub-batch", "description": "Candidates per GPU dispatch."},
                "feasibility_tolerance": {"type": "number", "minimum": 0, "default": 1e-6,
                    "title": "Feasibility tolerance",
                    "description": "Total normalised constraint violation regarded as feasible (best-so-far ranking, `final.feasible`)."},
                "stop_key": {"type": ["string", "null"], "default": null, "title": "Stop key",
                    "description": "Key for `request_stop`: stop and keep the best point."}
            },
            "required": ["variables"]
        })
    }

    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let s: OptimizeSettings = parse(settings, "settings")?;
        if s.variables.is_empty() {
            return Err(invalid("'variables': at least one variable is needed"));
        }
        let system = input.snapshot.system().clone();
        let default_merit = patched(
            &DefaultMeritSpec::default(),
            &s.default_merit,
            "default_merit",
        )?;
        let spec = build_merit(
            &system,
            &s.merit,
            &default_merit,
            &s.variables,
            &s.operands,
            &s.constraints,
        )?;
        let (algo, max_iter) = Algo::new(&s)?;
        let reg = OperandRegistry::builtin();
        let reference = OpticalProblem::new(
            system.clone(),
            s.variables.clone(),
            spec.clone(),
            TraceEvaluator::new(input.library.clone()),
            &reg,
        )?;
        let x0 = reference.start_point()?;
        ctx.report(0.0, "qualifying the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, &system, s.gpu_batch, ctx)?;
        ctx.check()?;

        // The search.
        let history: Arc<Mutex<Vec<HistoryRow>>> = Arc::default();
        let method = s.algorithm.name();
        let (run, why) = interruptible(ctx, s.stop_key.as_deref(), |inner| {
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
                Some(g) => {
                    let p = OpticalProblem::new(
                        system.clone(),
                        s.variables.clone(),
                        spec.clone(),
                        g.clone().composite(),
                        &reg,
                    )?;
                    algo.run(&p, &x0, &mon)
                }
                None => algo.run(&reference, &x0, &mon),
            };
            r.map(|r| (r, mon.best.get()))
        });
        let (result, best) = match run {
            Ok(r) => r,
            Err(MkError::Cancelled) => return Err(MkError::Cancelled.into()),
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
        let x = match &best {
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
                b.x.clone()
            }
            _ => result.x.clone(),
        };

        // Reported numbers: CPU f64 reference.
        ctx.report(1.0, "reference evaluation");
        let refs = reference_values(&reference, &[&x0, &x])?;
        let final_system = reference.apply(&x)?;
        ev.finish(s.backend, &final_system);
        let hash = Snapshot::new(final_system.clone())?.hash().to_hex();
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
        let feasible = refs.violation[1] <= feas_tol;
        let json = json!({
            "status": interrupt_status(why, result.termination),
            "termination": result.termination,
            "algorithm": method,
            "iterations": result.iterations,
            "evaluations": result.evaluations,
            "start": {"merit": num(refs.merit[0]), "violation": num(refs.violation[0]),
                "status": refs.status[0]},
            "final": {"merit": num(refs.merit[1]), "violation": num(refs.violation[1]),
                "status": refs.status[1], "feasible": feasible},
            "search": {"merit": num(result.merit), "violation": num(result.violation)},
            "improvement": num(if refs.merit[0] > 0.0 { 1.0 - refs.merit[1] / refs.merit[0] } else { f64::NAN }),
            "multipliers": result.multipliers.as_ref().map(|m| m.iter().map(|v| num(*v)).collect::<Vec<_>>()),
            "failed_variables": result.failed_variables,
            "variables": variables,
            "operands": operand_table(&spec, &refs.values[0], &refs.values[1], refs.merit[1]),
            "history": history,
            "backend": ev.report,
            "merit_spec": spec,
            "system": final_system,
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
        let mut warnings = ev.warnings;
        if !feasible {
            warnings.push(format!(
                "the final design violates its constraints (total normalised violation {:.3e})",
                refs.violation[1]
            ));
        }
        Ok(AnalysisOutput {
            json,
            buffers: vec![Buffer::new(
                "merit_history",
                vec![history.len(), 4],
                BufferData::F64(flat),
            )],
            warnings,
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
                "tolerances": {"type": "array", "title": "Tolerances",
                    "description": "Tolerance operands: {\"param\": ParamPath, \"minus\", \"plus\", \"distribution\", \"label\"}. At least one."},
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
            },
            "required": ["tolerances"]
        })
    }

    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let s: ToleranceSettings = parse(settings, "settings")?;
        if s.tolerances.is_empty() {
            return Err(invalid("'tolerances': at least one tolerance is needed"));
        }
        let system = input.snapshot.system().clone();
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
        ctx.report(0.0, "qualifying the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, &system, s.gpu_batch, ctx)?;
        ctx.check()?;
        let (do_sens, do_mc) = match s.mode {
            ToleranceMode::Sensitivity => (true, false),
            ToleranceMode::MonteCarlo => (false, true),
            ToleranceMode::Both => (true, true),
        };
        let split = if do_sens && do_mc {
            0.2
        } else if do_sens {
            1.0
        } else {
            0.0
        };

        let ((sens, mcr), why) = interruptible(ctx, s.stop_key.as_deref(), |inner| {
            let run = |e: &dyn ToleranceRunner| -> (Option<Result<_, MkError>>, Option<Result<_, MkError>>) {
                let sens = do_sens.then(|| {
                    let mon = Monitor::new(inner.subrange(0.0, split));
                    mon.progress(0.0, "tolerance sensitivity");
                    e.sensitivity(&spec, &mon)
                });
                let cancelled = matches!(sens, Some(Err(MkError::Cancelled)));
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
        ev.finish(s.backend, &system);
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
    use crate::backend::BackendInfo;
    use crate::engine::{Engine, EngineConfig};
    use crate::scheduler::JobState;
    use crate::session::AnalysisRequest;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::model::{ApertureSpec, SurfaceShape};
    use mklight_core::param::ParamPath;
    use mklight_core::{SurfaceId, samples};
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
        })
    }

    #[test]
    fn registered_with_schemas() {
        let reg = AnalysisRegistry::with_builtin();
        for name in ["optimize", "tolerance"] {
            let info = reg.list().into_iter().find(|i| i.name == name).unwrap();
            assert!(info.available, "{name}");
            assert_eq!(info.category, "optimization");
            // Variables / tolerances are required: no complete defaults.
            assert_eq!(info.default_settings, Value::Null);
        }
        let sys = samples::singlet();
        for bad in [
            json!({}),
            json!({"variables": []}),
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
            // Unconstrained: with the EFL equality every trial point would be
            // infeasible and both seeds would keep the start.
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

        // Through the scheduler with a stop key.
        assert!(!request_stop("no-such-job"));
        let engine = Engine::new(EngineConfig {
            worker_threads: 1,
            ..Default::default()
        });
        let mut session = engine.new_session();
        session.set_system(samples::cooke_triplet()).unwrap();
        s["stop_key"] = json!("test-stop-triplet");
        let job = session
            .submit_analysis(&AnalysisRequest::new("optimize", s))
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
        assert!(request_stop("test-stop-triplet"));
        assert_eq!(engine.wait_job(job, LONG).unwrap(), JobState::Done);
        let r = engine.take_result(job).unwrap();
        assert_eq!(r.json["status"], json!("stopped"));
        assert!(r.json["iterations"].as_u64().unwrap() < 200);
        assert!(f(&r.json["final"]["merit"]) < f(&r.json["start"]["merit"]));
        assert!(
            !request_stop("test-stop-triplet"),
            "unregistered after the run"
        );
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
        let mut backends = vec!["gpu_emulation"];
        if default_gpu().is_some() {
            backends.push("gpu");
        } else {
            eprintln!("SKIP (Metal part): no GPU device");
        }
        for backend in backends {
            let mut s = unconstrained_triplet();
            s["max_iterations"] = json!(4);
            s["backend"] = json!(backend);
            let r = run(samples::cooke_triplet(), "optimize", s).unwrap();
            let b = &r.json["backend"];
            assert_eq!(b["precision"], json!("f32_gpu"), "{backend}: {b}");
            assert_eq!(b["qualification"]["passed"], json!(true));
            assert_eq!(b["final_qualification"]["passed"], json!(true));
            assert!(b["gpu_stats"]["gpu_candidates"].as_u64().unwrap() > 0);
            assert_eq!(b["reported_on"], json!(TraceEvaluator::BACKEND));
            if backend == "gpu" {
                assert!(b["evaluated_on"].as_str().unwrap().starts_with("metal:"));
            }
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
    }
}
