//! Optimisation and tolerancing as engine analyses: `optimize` and
//! `tolerance` (`mklight-optim`), with merit evaluation on the CPU f64
//! reference or GPU-batched on a `mklight-gpu` backend
//! (`ResidentEvaluator`). Settings and result shapes: `docs/OPTIMIZATION.md`
//! (Engine jobs).
//!
//! * **Progress** is reported per optimiser iteration (stage text
//!   `"<method> <iteration>: merit <best>"`) and per Monte-Carlo batch.
//! * **Stopping and cancelling.** Cancelling through the job context stops
//!   the search of any of the jobs and returns `Cancelled`, without a
//!   result. To stop a job *and keep* what it has (the best point, the
//!   completed trials), call `Engine::stop_job`: the job then ends normally
//!   with `"status": "stopped"` (`job_status`).
//! * **One model.** Every job works on the snapshot's system with its
//!   solves applied and builds its merit the same way (`SolvedStart`,
//!   `PreparedMerit`); none changes a solve-controlled parameter itself.
//! * **Reported numbers** (start and final merit, operand table) are always
//!   computed with the CPU f64 reference, whatever evaluated the search; the
//!   result says which backend evaluated it, with its actual precision;
//!   `RunInfo.warnings` repeats it.
//! * **Never worse than the start**: the result is the better of the start
//!   and the optimiser's answer on the reference (feasibility first), with
//!   a warning when the optimiser found nothing better.
//! * **Backend `auto`** evaluates on the device the engine selected for the
//!   job (its cost model under engine `auto`, GPU.md §9.2), else on the CPU
//!   reference.
//! * The result carries the final system as model JSON (`"system"`), so a
//!   front end can apply it as a new (undoable) snapshot.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mklight_core::job::{CancelToken, Precision, Progress};
use mklight_core::materials::MaterialLibrary;
use mklight_core::param::ParamPath;
use mklight_core::platform;
use mklight_core::{JobContext, MkError, OpticalSystem, SemiDiameter, Snapshot, SurfaceShape};
use mklight_optim::deriv::{FdScheme, JacobianSettings};
use mklight_optim::global::cmaes::{CmaEsSettings, cma_es};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::global::hammer::{HammerSettings, hammer};
use mklight_optim::global::nelder_mead::{NelderMeadSettings, nelder_mead};
use mklight_optim::gpu::device_rays_per_candidate;
use mklight_optim::gpu::{GpuContext, ResidentEvaluator};
use mklight_optim::lsq::{AugLagSettings, LsqMethod, LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultMeritSpec, ThicknessBoundaries};
use mklight_optim::merit::geometric::optical_gaps;
use mklight_optim::merit::operand::OperandKind;
use mklight_optim::merit::{
    BuiltinEvaluator, Comparison, MeritFunctionSpec, Operand, OperandEvaluator, OperandRegistry,
    OperandRole, OperandValue, OpticalProblem, PupilSampling, TraceEvaluator, VariableSpec,
    WithSolves,
};
use mklight_optim::problem::{
    CandidateBatch, EvalStatus, Problem, ScalarEval, merit, total_violation,
};
use mklight_optim::tolerance::{
    Compensator, MonteCarloSettings, OpticalToleranceEvaluator, ToleranceEvaluator,
    ToleranceOperand, ToleranceSpec, monte_carlo, sensitivity_batched,
};
use mklight_optim::{IterationRecord, Monitor, OptimResult, Termination};

use super::BackendUse;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::{EngineError, EngineResult};
use crate::gpu::GpuDevice;
use crate::route::SolverAiming;

#[path = "glass.rs"]
mod glass;

/// Register continuous optimisation, glass substitution and tolerancing.
pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(Optimize));
    r.register(Arc::new(Tolerance));
    r.register(Arc::new(glass::GlassSubstitution));
}

// The `backend` record shared by both result schemas.
pub(super) fn evaluation_backend_schema() -> Value {
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
                string(
                    "CPU reference for final optimize/glass reports; active evaluator for tolerance and merit_evaluation.",
                ),
            ),
            (
                "workload",
                obj(&[("rays_per_candidate", int("")), ("batch", int(""))]),
            ),
            (
                "resident_evaluation",
                opaque("Device evaluation, host-stage and fallback accounting."),
            ),
            (
                "aiming",
                enumeration(
                    &["by-batch", "device", "host"],
                    "Where the pupil rays of a run handed a GPU device are aimed (route option `solver_aiming`): by batch (default; `resident_evaluation.device_aimed_batches` counts the device's), every batch on the device with host completion, or every batch on the host with device tracing.",
                ),
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
        &["gpu_stats", "resident_evaluation", "aiming"],
    )
}

/// How a run ended early.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Interrupt {
    None,
    Stopped,
    Cancelled,
    TimeLimit,
}

/// Run `f` with a context that is cancelled when `ctx` is or when `stop` (the
/// job's stop token, `AnalysisInput::stop`) is (a watcher thread joins the
/// two tokens), and say which happened.
fn interruptible<T>(
    ctx: &JobContext,
    stop: &CancelToken,
    f: impl FnOnce(&JobContext) -> T,
) -> (T, Interrupt) {
    interruptible_for(ctx, stop, None, f)
}

fn interruptible_for<T>(
    ctx: &JobContext,
    stop: &CancelToken,
    limit: Option<Duration>,
    f: impl FnOnce(&JobContext) -> T,
) -> (T, Interrupt) {
    /// Ends the watcher and joins it when the run returns or unwinds.
    struct Finished(Arc<AtomicBool>, Option<platform::JoinHandle<()>>);
    impl Drop for Finished {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
            if let Some(w) = self.1.take() {
                let _ = w.join();
            }
        }
    }
    let mut inner = ctx.subrange(0.0, 1.0);
    inner.cancel = CancelToken::new();
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let started = platform::Instant::now();
    // A plain thread rather than a scoped one (`platform::spawn` is the one
    // spawn that also exists on wasm32), so it owns clones of the tokens.
    let watcher = {
        let (done, timed_out) = (done.clone(), timed_out.clone());
        let (outer, stop, inner) = (ctx.cancel.clone(), stop.clone(), inner.cancel.clone());
        platform::spawn("mklight-optim-watch", move || {
            while !done.load(Ordering::Acquire) {
                if outer.is_cancelled() || stop.is_cancelled() {
                    inner.cancel();
                    break;
                }
                if limit.is_some_and(|d| started.elapsed() >= d) {
                    timed_out.store(true, Ordering::Release);
                    inner.cancel();
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        })
        .expect("spawn optimisation watcher")
    };
    let out = {
        // Dropped on a panic too: the watcher is joined before the panic
        // propagates.
        let _finished = Finished(done.clone(), Some(watcher));
        f(&inner)
    };
    let why = if ctx.cancel.is_cancelled() {
        Interrupt::Cancelled
    } else if stop.is_cancelled() {
        Interrupt::Stopped
    } else if timed_out.load(Ordering::Acquire) {
        Interrupt::TimeLimit
    } else {
        Interrupt::None
    };
    (out, why)
}

// ---------------------------------------------------------------------------
// Shared helpers

/// Finite numbers as numbers, NaN as null, ±∞ as `"inf"`/`"-inf"`
/// (CONVENTIONS §11).
fn json_nf(x: f64) -> Value {
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

/// The first key of the overlay `patch` that `fields` (a serialised
/// settings struct) does not have, through nested objects: an error with
/// the keys that exist there.
fn check_overlay_keys(fields: &Value, patch: &Value, path: &str) -> EngineResult<()> {
    let (Value::Object(fields), Value::Object(patch)) = (fields, patch) else {
        return Ok(());
    };
    for (key, value) in patch {
        match fields.get(key) {
            Some(field) => check_overlay_keys(field, value, &format!("{path}.{key}"))?,
            None => {
                let valid: Vec<&str> = fields.keys().map(String::as_str).collect();
                return Err(invalid(format!(
                    "'{path}': unknown key '{key}' (valid: {})",
                    valid.join(", ")
                )));
            }
        }
    }
    Ok(())
}

/// The recursive check of a job's `settings` against its schema: beyond the
/// top-level normalisation, unknown keys and wrong types inside nested
/// objects (`default_merit` and below) are errors. The criterion of the
/// default merit is a tagged union, checked against the alternative of its
/// `type`.
fn validate_settings(schema: &Value, settings: &Value) -> EngineResult<()> {
    super::schema::validate(schema, settings, true).map_err(invalid)?;
    let criterion = &settings["default_merit"]["criterion"];
    let alternatives = &schema["properties"]["default_merit"]["properties"]["criterion"]["oneOf"];
    if let (Some(tag), Some(alternatives)) = (criterion.get("type"), alternatives.as_array()) {
        const AT: &str = "$.default_merit.criterion";
        let tag_of = |a: &Value| a["properties"]["type"]["const"].clone();
        let variant = alternatives
            .iter()
            .find(|a| tag_of(a) == *tag)
            .ok_or_else(|| {
                let tags: Vec<Value> = alternatives.iter().map(tag_of).collect();
                invalid(format!("{AT}.type: {tag} is not one of {}", json!(tags)))
            })?;
        super::schema::validate(variant, criterion, true)
            .map_err(|e| invalid(e.replacen('$', AT, 1)))?;
    }
    Ok(())
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

/// Shared schema of the default-merit generator (every `DefaultMeritSpec`
/// key) accepted by `optimize`, `tolerance` and `glass_substitution`.
fn default_merit_schema(default: Value) -> Value {
    use super::schema::settings_from_default;
    let reference = json!({"type": "string", "enum": ["centroid", "chief"], "default": "centroid"});
    let sampling = settings_from_default::<PupilSampling>(json!({
        "symmetry": {"enum": ["none", "bilateral"]},
        "arm_offset_deg": {"unit": "deg"}
    }));
    let boundary = json!({"type": ["number", "null"], "unit": "mm"});
    let thickness = settings_from_default::<ThicknessBoundaries>(json!({
        "glass_min_centre": boundary, "glass_max_centre": boundary, "glass_min_edge": boundary,
        "air_min_centre": boundary, "air_max_centre": boundary, "air_min_edge": boundary
    }));
    let mut schema = settings_from_default::<DefaultMeritSpec>(json!({
        "criterion": {"type": "object", "oneOf": [
            {"type": "object", "required": ["type"], "properties": {
                "type": {"const": "rms_spot"},
                "reference": reference,
                "scope": {"type": "string", "enum": ["monochromatic", "polychromatic"], "default": "polychromatic",
                    "description": "Spot centroid of the white-light spot (polychromatic; penalises lateral colour) or of each wavelength's own spot (monochromatic)."}
            }},
            {"type": "object", "required": ["type"], "properties": {
                "type": {"const": "wavefront"},
                "reference": reference
            }},
            {"type": "object", "required": ["type", "frequencies", "target"], "properties": {
                "type": {"const": "mtf"},
                "frequencies": {"type": "array", "minItems": 1, "items": {"type": "number", "minimum": 0}, "description": "Fixed image frequencies in cycles/mm."},
                "target": {"type": "number", "minimum": 0, "maximum": 1},
                "method": {"type": "string", "enum": ["diffraction", "geometric"], "default": "diffraction"},
                "worst_field": {"type": "boolean", "default": false, "description": "One combine:min over every field and both T/S directions per frequency."}
            }}
        ]},
        "sampling": {"properties": sampling["properties"],
            "description": "Rings and arms of the pupil sampling."},
        "fields": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 0},
            "description": "Field indices; null: every field (or the x/y grid for anamorphic and freeform systems)."},
        "wavelengths": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 0},
            "description": "Wavelength indices; null: every wavelength."},
        "thickness": {"type": ["object", "null"], "properties": thickness["properties"],
            "description": "Centre and edge thickness boundaries; null: none."}
    }));
    // Omitted from the serialised default when unset.
    schema["properties"]["squeeze_target"] = json!({"type": ["number", "null"], "exclusiveMinimum": 0,
        "default": null, "description": "Anamorphic squeeze EFL_y/EFL_x to hold; null: none."});
    let o = schema.as_object_mut().unwrap();
    o.insert("default".into(), default);
    o.insert("title".into(), json!("Default merit"));
    o.insert("description".into(), json!("Default-merit generator: RMS spot, wavefront or polychromatic MTF; fields, wavelengths, sampling and thickness boundaries."));
    schema
}

/// Which evaluator computes the merit during the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BackendChoice {
    /// The device the engine selected for the job, else the CPU reference.
    Auto,
    Cpu,
    /// The engine's GPU device for this analysis; the CPU reference (with
    /// a warning) when there is none.
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

/// Whose values a result reports (`backend.reported_on`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reported {
    /// The CPU f64 reference (`optimize`, `glass_substitution`).
    Reference,
    /// The evaluator of the run (`tolerance`, `merit_evaluation`).
    Evaluator,
}

/// The work of a run, reported in `backend.workload`: the estimated device
/// rays per candidate and the typical batch of candidates.
#[derive(Clone, Copy, Debug)]
struct Workload {
    rays_per_candidate: usize,
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
            rays_per_candidate: device_rays_per_candidate(cpu, system, &ops),
            batch,
        }
    }
    fn report(&self) -> Value {
        json!({
            "rays_per_candidate": self.rays_per_candidate,
            "batch": self.batch,
        })
    }
}

/// The GPU backend for a request, or why there is none.
fn gpu_backend(
    choice: BackendChoice,
    input: &AnalysisInput<'_>,
) -> Result<Option<Arc<GpuDevice>>, String> {
    let engine = super::gpu::device(input);
    match choice {
        BackendChoice::Cpu => Ok(None),
        BackendChoice::Auto => Ok(engine),
        BackendChoice::GpuEmulation => engine
            .filter(|d| d.id == "gpu:cpu_f32")
            .map(Some)
            .ok_or_else(|| "select the engine's gpu:cpu_f32 provider for gpu_emulation".into()),
        BackendChoice::Gpu => engine
            .map(Some)
            .ok_or_else(|| "the engine selected no GPU device for this analysis".to_string()),
    }
}

/// The merit evaluator of a run: the CPU reference alone, or a GPU
/// evaluator in front of it.
struct Evaluation {
    cpu: BuiltinEvaluator,
    /// The GPU evaluator and the engine device behind it.
    gpu: Option<(ResidentEvaluator, Arc<GpuDevice>)>,
    /// A run handed a GPU device: where the pupil rays are aimed
    /// (`backend.aiming`).
    aiming: Option<&'static str>,
    work: Workload,
    fallback_reason: Option<String>,
    /// `"backend"` block of the result.
    report: Value,
    warnings: Vec<String>,
}

impl Evaluation {
    /// The evaluator for `snapshot`, the start of the run.
    fn new(
        choice: BackendChoice,
        input: &AnalysisInput<'_>,
        snapshot: &Snapshot,
        work: Workload,
        ctx: &JobContext,
    ) -> EngineResult<Self> {
        let cpu = BuiltinEvaluator::new(input.library.clone());
        let mut warnings = Vec::new();
        let mut fallback_reason = input.selection_note.clone();
        let mut aiming = None;
        let gpu = match gpu_backend(choice, input) {
            Ok(Some(device)) => {
                ctx.check()?;
                // Where the pupil rays are aimed (route option
                // `solver_aiming`): pinned to the host or the device, or
                // left to the evaluator's rule per batch
                // (`gpu::DEVICE_AIM_FROM`, docs/OPTIMIZATION.md "Aiming").
                let route = input.route.solver_aiming;
                aiming = Some(route.name());
                let context = GpuContext::new(device.backend.clone(), snapshot.clone());
                let mut evaluator = ResidentEvaluator::new(context, cpu.clone());
                match route {
                    SolverAiming::ByBatch => {}
                    SolverAiming::Host => evaluator = evaluator.with_host_aim(true),
                    SolverAiming::Device => evaluator = evaluator.with_host_aim(false),
                }
                Some((evaluator, device))
            }
            Ok(None) => {
                if super::gpu::device(input).is_some() && choice == BackendChoice::Cpu {
                    fallback_reason = Some("merit evaluation requested the CPU reference".into());
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
            aiming,
            work,
            fallback_reason,
            report: Value::Null,
            warnings,
        })
    }

    fn active_gpu(&self) -> Option<&ResidentEvaluator> {
        self.gpu.as_ref().map(|(g, _)| g)
    }

    /// The evaluator of the search: the GPU evaluator of a run that has
    /// one, else the CPU reference.
    fn evaluator(&self) -> &dyn OperandEvaluator {
        match &self.gpu {
            Some((g, _)) => g,
            None => &self.cpu,
        }
    }

    /// Run `f` on the evaluator of the search; with `solves` (the library
    /// they resolve materials in) on one that applies the solves of every
    /// candidate first (`WithSolves`).
    fn with_evaluator<T>(
        &self,
        solves: Option<&Arc<MaterialLibrary>>,
        f: impl FnOnce(&dyn OperandEvaluator) -> T,
    ) -> T {
        match solves {
            Some(library) => f(&WithSolves::new(self.evaluator(), library.clone())),
            None => f(self.evaluator()),
        }
    }

    /// Least-squares solves with device aiming pinned evaluate their trial points
    /// with the next Jacobian's probes (one device round trip per accepted
    /// iteration instead of two). With host aiming every probe is a host
    /// evaluation, as on the CPU reference (which evaluates a trial point
    /// alone: in a batch its anchored base row would change the values),
    /// and a rejected trial point's probes, at a design whose rim rays fail,
    /// cost as much (`docs/OPTIMIZATION.md`, "Speculative Jacobians").
    fn speculative(&self) -> bool {
        self.active_gpu()
            .is_some_and(|g| g.device_aim_from() == Some(0))
    }

    /// The `"backend"` block: who evaluated the search, whose values the
    /// result reports and the GPU work counters. Reports the backend used
    /// to the engine.
    fn finish(
        &mut self,
        input: &AnalysisInput<'_>,
        requested: BackendChoice,
        reported: Reported,
    ) -> EngineResult<()> {
        let mut b = json!({
            "requested": requested.name(),
            "evaluated_on": TraceEvaluator::BACKEND,
            "precision": Precision::F64Reference,
            "reported_on": TraceEvaluator::BACKEND,
            "workload": self.work.report(),
        });
        if let Some(aiming) = self.aiming {
            b["aiming"] = json!(aiming);
        }
        // The GPU evaluated the search only if it traced candidates.
        let active = self
            .active_gpu()
            .is_some_and(|g| g.stats().device_operands > 0);
        if let Some((g, device)) = &self.gpu {
            if active {
                self.warnings.push(format!(
                    "ray operands evaluated on {} (double-float)",
                    device.backend.info().backend_id()
                ));
                b["evaluated_on"] = json!(device.backend.info().backend_id());
                b["precision"] = json!(Precision::F32Gpu);
                if reported == Reported::Evaluator {
                    b["reported_on"] = b["evaluated_on"].clone();
                }
            }
            let stats = g.stats();
            let plans = g.context.stats();
            b["gpu_stats"] = json!({
                "gpu_candidates": stats.device_candidates,
                "cpu_candidates": stats.cpu_candidates,
                "rays": g.context.traced_rays(), "dispatches": plans.command_buffers, "device_seconds": plans.device_seconds,
            });
            let mut run = mklight_core::job::RunTimer::start().finish(
                &input.snapshot,
                "resident_merit",
                &json!({}),
                "cpu_reference",
                Precision::F64Reference,
                None,
                vec![],
            );
            g.record(&mut run);
            if let Some(aiming) = self.aiming {
                run.settings["solvers_aiming"] = json!(aiming);
            }
            input.executions.borrow_mut().push(run.settings.clone());
            b["resident_evaluation"] = serde_json::to_value(stats)?;
        }
        let used = self.gpu.as_ref().filter(|_| active);
        // A CPU-reference request never had a GPU to fall back from.
        let fallback_reason = if active {
            None
        } else {
            self.fallback_reason.clone().or_else(|| {
                self.gpu
                    .is_some()
                    .then(|| "no candidates evaluated on the GPU".into())
            })
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
    cpu: &BuiltinEvaluator,
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

/// The curvature parameters of a surface, with labels: x and y for toroids
/// and biconics, whose legacy curvature path is only the y curvature.
fn curvature_params(sys: &OpticalSystem, i: usize) -> Vec<(ParamPath, String)> {
    use mklight_core::SurfaceShape::{Biconic, Toroid};
    use mklight_core::param::ShapeParameter::{CurvatureX, CurvatureY};
    let surface = sys.surfaces[i].id;
    if matches!(sys.surfaces[i].shape, Toroid { .. } | Biconic { .. }) {
        [(CurvatureX, "x"), (CurvatureY, "y")]
            .map(|(parameter, axis)| {
                (
                    ParamPath::ShapeParam { surface, parameter },
                    format!("curvature {axis} {}", surface.0),
                )
            })
            .into()
    } else {
        vec![(
            ParamPath::Curvature { surface },
            format!("curvature {}", surface.0),
        )]
    }
}

/// The default variables: the curvature of every curved, optically active
/// surface and the back focus (the thickness before the image).
fn auto_variables(sys: &OpticalSystem) -> Vec<VariableSpec> {
    let n = sys.surfaces.len();
    let gaps = optical_gaps(sys);
    let mut v: Vec<VariableSpec> = gaps
        .iter()
        .flat_map(|g| curvature_params(sys, g.from))
        .filter(|(p, _)| p.get(sys).is_ok_and(|c| c != 0.0))
        .map(|(p, _)| VariableSpec::new(p))
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
    for (p, label) in gaps.iter().flat_map(|g| curvature_params(sys, g.from)) {
        if let Ok(c) = p.get(sys)
            && c != 0.0
        {
            let mut o = ToleranceOperand::symmetric(p, 0.005 * c.abs());
            o.label = label;
            t.push(o);
        }
    }
    t
}

/// Does the spec already hold the legacy y EFL (or its explicit y alias)?
fn has_efl(spec: &MeritFunctionSpec) -> bool {
    spec.enabled()
        .any(|(_, o)| matches!(o.kind, OperandKind::Efl | OperandKind::EflY))
}

fn value_json(v: &OperandValue) -> (Value, Value) {
    match v {
        Ok(x) => (json_nf(*x), Value::Null),
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
                    (json_nf(r), op.weight * r * r)
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
                "target": json_nf(op.target),
                "weight": json_nf(op.weight),
                "config": op.config,
                "field": op.field,
                "wavelength": op.wavelength,
                "start": v0,
                "value": v1,
                "residual": res,
                "contribution": json_nf(contrib),
                "contribution_fraction": json_nf(if end_merit > 0.0 { contrib / end_merit } else { f64::NAN }),
                "error": err,
            })
        })
        .collect()
}

pub(super) fn operand_schema() -> Value {
    use super::schema::*;
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
    ])
}

/// Merit, violation and operand values of candidate points, on the
/// evaluator that computed the operand values.
struct PointValues {
    merit: Vec<f64>,
    violation: Vec<f64>,
    status: Vec<EvalStatus>,
    values: Vec<Vec<OperandValue>>,
}

/// The values of `points` on the CPU reference.
fn reference_values(
    p: &OpticalProblem<BuiltinEvaluator>,
    points: &[&[f64]],
    ctx: &JobContext,
) -> EngineResult<PointValues> {
    ctx.check()?;
    let batch = CandidateBatch::from_points(p.num_variables(), points.iter().copied());
    let inner = JobContext::with(ctx.cancel.clone(), Progress::default());
    Ok(values_report(p, p.operand_values(&batch, &inner)?))
}

/// The report of operand values `vals` (one entry per candidate) of `p`.
fn values_report<E: OperandEvaluator>(
    p: &OpticalProblem<E>,
    vals: Vec<Option<Vec<OperandValue>>>,
) -> PointValues {
    let count = vals.len();
    let e = p.to_evaluation(vals.clone());
    let n_ops = p.spec().operands.len();
    PointValues {
        merit: (0..count)
            .map(|k| merit(p.weights(), e.residuals(k), e.status[k]))
            .collect(),
        violation: (0..count)
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
    }
}

/// `status` of a solver job's result, from how its search ended (a
/// cancelled job returns no result): `stopped` when a Stop ended it and the
/// result holds what it had, `converged` when its optimiser met a
/// convergence test, else `finished` (a budget or a time limit; a search
/// without a convergence test, `termination` `None`, that ran to its end).
fn job_status(stopped: bool, termination: Option<Termination>) -> &'static str {
    if stopped {
        "stopped"
    } else if termination.is_some_and(Termination::is_converged) {
        "converged"
    } else {
        "finished"
    }
}

// ---------------------------------------------------------------------------
// optimize

/// Iterations an augmented-Lagrangian inner solve may take at an infeasible
/// point before the multipliers or the penalty are updated (the job's
/// default for `AugLagSettings::max_inner_iterations`).
const INNER_ITERATIONS: usize = 10;

/// Candidates per tolerance evaluation batch: sensitivity points go to the
/// evaluator this many at a time, Monte Carlo trials at least this many (the
/// `batch` setting only raises it; on the CPU reference and with host
/// aiming trial values do not depend on the batching, with device aiming
/// they do within the device's precision). The hundred trials of a default
/// Monte Carlo run go to the evaluator as one batch, which the CPU spreads
/// over its pool without a barrier every 16 trials (29-surface F/1.4
/// benchmark lens, 12 threads: 155 → 120 ms for the default tolerance run,
/// 2.4 → 1.9 s for 1000 dense trials, identical results) and a pipelined
/// evaluator splits into overlapped device plans.
const TOLERANCE_BATCH: usize = 128;

/// The job's initial spreads of the population methods around the start,
/// in variable scales (`VariableSpec::scale`; 10 % of a curvature by
/// default). The library defaults (DE 1.0, CMA-ES 0.3) suit a search of a
/// box; around an existing design they sample candidates so much worse that
/// the search ends at or near the start. Measured on the seven samples with
/// the default merit and variables, 100 DE / 200 CMA-ES generations: DE
/// improved 2 of 7 at 1.0 and 7 of 7 at 0.03, CMA-ES 6 of 7 at 0.3 (with the
/// EFL penalty) and 7 of 7 at 0.05.
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
    feasibility_tolerance: f64,
}

impl OptimizeSettings {
    /// The checked settings of an `optimize` or `merit_evaluation` request
    /// with its algorithm (and iteration budget) and its merit on the
    /// solved start.
    fn prepare(
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<(OptimizeSettings, (Algo, usize), PreparedMerit)> {
        validate_settings(&Optimize.settings_schema(), settings)?;
        let s: OptimizeSettings = parse(settings, "settings")?;
        let algo = Algo::new(&s)?;
        let start = SolvedStart::new(input, ctx)?;
        let variables =
            start.parameters("variables", s.variables.clone(), auto_variables, |v| {
                &v.param
            })?;
        let merit = MeritSettings {
            variables,
            merit: s.merit.clone(),
            default_merit: patched(
                &DefaultMeritSpec::default(),
                &s.default_merit,
                "default_merit",
            )?,
            relax_boundaries: true,
            operands: s.operands.clone(),
            constraints: s.constraints.clone(),
            hold_efl: s.hold_efl,
            derivative_free: s.algorithm.derivative_free().then(|| s.algorithm.name()),
        };
        let merit = PreparedMerit::new(input, start, merit, ctx)?;
        Ok((s, algo, merit))
    }
}

/// The start of a solver job: the snapshot's system with its solves applied
/// (`docs/SOLVES.md`). Every job builds its merit on it, and none changes a
/// solve-controlled parameter itself: such a parameter is never a variable,
/// a tolerance or a compensator.
struct SolvedStart {
    system: OpticalSystem,
    /// The solve-controlled parameters.
    solved: Vec<ParamPath>,
    warnings: Vec<String>,
}

impl SolvedStart {
    fn new(input: &AnalysisInput<'_>, ctx: &JobContext) -> EngineResult<Self> {
        let (system, solves) = mklight_trace::apply_solves_with(
            input.snapshot.system(),
            &input.library,
            &Default::default(),
            ctx,
        )?;
        Ok(SolvedStart {
            solved: system.solved_params(),
            system,
            warnings: solves.warnings,
        })
    }

    /// The parameters a job changes on the start (its setting `key`): the
    /// `given` ones, none of which a solve may control, or with none given
    /// the `automatic` ones that no solve controls.
    fn parameters<T>(
        &self,
        key: &str,
        given: Vec<T>,
        automatic: impl FnOnce(&OpticalSystem) -> Vec<T>,
        param: impl Fn(&T) -> &ParamPath,
    ) -> EngineResult<Vec<T>> {
        if given.is_empty() {
            let mut all = automatic(&self.system);
            all.retain(|p| !self.solved.contains(param(p)));
            return Ok(all);
        }
        match given.iter().position(|p| self.solved.contains(param(p))) {
            Some(i) => Err(invalid(format!(
                "'{key}'[{i}]: a solve controls this parameter; a job cannot change it"
            ))),
            None => Ok(given),
        }
    }
}

/// What a solver job asks of the shared merit preparation.
struct MeritSettings {
    /// The continuous parameters the job changes on the start: its
    /// variables, for `tolerance` its tolerances and compensators. They
    /// decide the pupil symmetry of a generated merit and which of its
    /// boundaries are fixed.
    variables: Vec<VariableSpec>,
    /// An explicit merit-function spec; null generates the default merit.
    merit: Value,
    /// The default-merit generator: the job's `default_merit` laid over its
    /// defaults.
    default_merit: DefaultMeritSpec,
    /// Relax the generated thickness boundaries that the start violates and
    /// no variable changes (`relax_fixed_boundaries`).
    relax_boundaries: bool,
    /// Operands appended to the merit, and operands appended as constraints.
    operands: Vec<Operand>,
    constraints: Vec<Operand>,
    /// Hold the focal length of the start, unless the merit has an EFL
    /// operand.
    hold_efl: bool,
    /// The search method, when it is derivative-free: the EFL hold is then
    /// a penalty residual, and equality constraints get a warning.
    derivative_free: Option<&'static str>,
}

/// The merit of a solver job on its solved start, built the same way for
/// `optimize`, `merit_evaluation`, `glass_substitution` and `tolerance`.
struct PreparedMerit {
    /// The start: the snapshot's system with its solves applied.
    snapshot: Snapshot,
    /// The parameters the job changes (`MeritSettings::variables`).
    variables: Vec<VariableSpec>,
    cpu: BuiltinEvaluator,
    reg: OperandRegistry,
    spec: MeritFunctionSpec,
    warnings: Vec<String>,
    /// The derivative-free EFL penalty, still of weight 0 (`weight_efl`).
    efl_penalty: Option<usize>,
}

impl PreparedMerit {
    fn new(
        input: &AnalysisInput<'_>,
        start: SolvedStart,
        s: MeritSettings,
        ctx: &JobContext,
    ) -> EngineResult<Self> {
        let SolvedStart {
            system,
            mut warnings,
            ..
        } = start;
        let cpu = BuiltinEvaluator::new(input.library.clone());
        let reg = OperandRegistry::builtin();
        let mut spec = build_merit(&system, &s.merit, &s.default_merit, &s.variables)?;
        if s.merit.is_null() && s.relax_boundaries {
            warnings.extend(relax_fixed_boundaries(
                &system,
                &mut spec,
                &s.variables,
                &cpu,
                &reg,
                ctx,
            )?);
        }
        spec.operands.extend(s.operands);
        spec.operands.extend(s.constraints.into_iter().map(|mut o| {
            o.role = OperandRole::Constraint;
            o
        }));
        if let Some(algorithm) = s.derivative_free
            && spec.enabled().any(|(_, o)| {
                o.role == OperandRole::Constraint && o.comparison == Comparison::Equal
            })
        {
            warnings.push(format!(
                "{algorithm} rarely samples a point that meets an equality constraint exactly; \
                 such constraints are better held by lm, dogleg or hammer, or by a weighted operand"
            ));
        }
        let mut efl_penalty = None;
        if s.hold_efl && !has_efl(&spec) {
            match cpu
                .evaluate_system(&system, &[&Operand::new(OperandKind::Efl)], ctx)?
                .remove(0)
            {
                Ok(f) if f.is_finite() && f != 0.0 => {
                    let o = if s.derivative_free.is_some() {
                        // Penalty: the derivative-free methods' random trial
                        // points never meet an equality exactly, so Deb's
                        // rule would rank them by EFL error alone.
                        efl_penalty = Some(spec.operands.len());
                        let mut o = Operand::new(OperandKind::Efl).target(f).weight(0.0);
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
        Ok(PreparedMerit {
            snapshot: Snapshot::new(system)?,
            variables: s.variables,
            cpu,
            reg,
            spec,
            warnings,
            efl_penalty,
        })
    }

    fn problem<E: OperandEvaluator>(&self, evaluator: E) -> EngineResult<OpticalProblem<E>> {
        Ok(OpticalProblem::new(
            self.snapshot.system().clone(),
            self.variables.clone(),
            self.spec.clone(),
            evaluator,
            &self.reg,
        )?)
    }

    /// Weight the EFL penalty by the start merit, which its zero weight
    /// leaves unchanged.
    fn weight_efl(&mut self, start_merit: f64) {
        if let Some(i) = self.efl_penalty {
            let op = &mut self.spec.operands[i];
            let phi = if start_merit > 0.0 && start_merit.is_finite() {
                start_merit
            } else {
                1.0
            };
            op.weight = phi / (EFL_PENALTY_RELATIVE * op.target).powi(2);
        }
    }
}

/// An optimiser with its settings.
enum Algo {
    Lsq(LsqSettings),
    De(DeSettings),
    CmaEs(CmaEsSettings),
    NelderMead(NelderMeadSettings),
    Hammer(HammerSettings),
}

/// The settings of an algorithm: `options` laid over `defaults`, the
/// job's defaults for it with the job's own `max_iterations`, `fd_scheme`,
/// `fd_relative_step` and `seed` in their fields, so a key of `options`
/// wins over the setting that fills the same field. A key must be a field of
/// the algorithm's settings (through nested objects); a field in `owned` is
/// the job's alone (it names the setting to use) and is refused.
fn algorithm_settings<T: Serialize + DeserializeOwned>(
    defaults: &T,
    options: &Value,
    owned: &[(&str, &str)],
) -> EngineResult<T> {
    let mut fields = serde_json::to_value(defaults)?;
    for (field, setting) in owned {
        if options.get(field).is_some() {
            return Err(invalid(format!(
                "'options.{field}': set by the job's '{setting}' setting"
            )));
        }
        if let Some(fields) = fields.as_object_mut() {
            fields.remove(*field);
        }
    }
    check_overlay_keys(&fields, options, "options")?;
    patched(defaults, options, "options")
}

impl Algo {
    /// The algorithm of `s` with its settings, and its iteration budget.
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
        const SEED: [(&str, &str); 1] = [("seed", "seed")];
        Ok(match s.algorithm {
            Algorithm::Lm | Algorithm::Dogleg => {
                let l = algorithm_settings(
                    &LsqSettings {
                        method: if s.algorithm == Algorithm::Lm {
                            LsqMethod::LevenbergMarquardt
                        } else {
                            LsqMethod::Dogleg
                        },
                        jacobian,
                        max_iterations: iters,
                        constraints,
                        ..Default::default()
                    },
                    opts,
                    &[("method", "algorithm")],
                )?;
                (Algo::Lsq(l), l.max_iterations)
            }
            Algorithm::De => {
                let d = algorithm_settings(
                    &DeSettings {
                        max_generations: iters,
                        init_spread: DE_INIT_SPREAD,
                        seed: s.seed,
                        ..Default::default()
                    },
                    opts,
                    &SEED,
                )?;
                (Algo::De(d), d.max_generations)
            }
            Algorithm::CmaEs => {
                let c = algorithm_settings(
                    &CmaEsSettings {
                        max_generations: iters,
                        sigma0: CMA_ES_SIGMA0,
                        seed: s.seed,
                        ..Default::default()
                    },
                    opts,
                    &SEED,
                )?;
                (Algo::CmaEs(c), c.max_generations)
            }
            Algorithm::NelderMead => {
                let n = algorithm_settings(
                    &NelderMeadSettings {
                        max_iterations: iters,
                        ..Default::default()
                    },
                    opts,
                    &[],
                )?;
                (Algo::NelderMead(n), n.max_iterations)
            }
            Algorithm::Hammer => {
                let mut d = HammerSettings {
                    rounds: iters,
                    seed: s.seed,
                    ..Default::default()
                };
                d.local.jacobian = jacobian;
                d.local.constraints.max_inner_iterations = Some(INNER_ITERATIONS);
                let h = algorithm_settings(&d, opts, &SEED)?;
                (Algo::Hammer(h), h.rounds)
            }
        })
    }

    /// Least-squares solves (LM, dogleg, basin hopping's local runs)
    /// evaluate each trial point with the next Jacobian's probes.
    fn with_speculation(mut self, on: bool) -> Self {
        match &mut self {
            Algo::Lsq(l) => l.speculative_jacobian = on,
            Algo::Hammer(h) => h.local.speculative_jacobian = on,
            _ => {}
        }
        self
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
    json_nf(*x).serialize(s)
}

/// `optimize`: local or global optimisation of model parameters against a
/// merit function, on the CPU reference or GPU-batched.
pub struct Optimize;

impl Analysis for Optimize {
    fn result_version(&self) -> u32 {
        2
    }
    fn name(&self) -> &str {
        "optimize"
    }
    fn supports_stop(&self) -> bool {
        true
    }
    fn title(&self) -> &str {
        "Optimize"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Optimises the variables against a merit function (explicit, or the default RMS spot / \
         wavefront / polychromatic MTF merit with thickness boundaries) with explicit constraints, by \
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
                    "description": "Variable specs: {\"param\": ParamPath, \"lower\", \"upper\", \"scale\", \"fd_step\", \"label\"}. Empty: the curvature of every curved optical surface and the back focus, without the solve-controlled ones. A solve-controlled parameter is rejected."},
                "hold_efl": {"type": "boolean", "default": true, "title": "Hold focal length",
                    "description": "Hold the EFL at its start value unless the merit already has an efl operand: a constraint for lm, dogleg and hammer, a penalty residual for de, cma_es and nelder_mead."},
                "merit": {"type": ["object", "null"], "default": null, "title": "Merit function",
                    "description": "A merit-function spec; null generates the default merit from `default_merit` for the variables."},
                "default_merit": default_merit_schema(default_merit),
                "operands": {"type": "array", "default": [], "title": "Extra operands",
                    "description": "Operands appended to the merit, including MTF (frequency, orientation, method, defocus_mm, detector), ghost and stray power operands. Stray power uses a fixed seed and is unsuitable for LM."},
                "constraints": {"type": "array", "default": [], "title": "Constraints",
                    "description": "Operands appended as explicit constraints (role forced to constraint)."},
                "algorithm": {"type": "string", "enum": ["lm", "dogleg", "de", "cma_es", "nelder_mead", "hammer"],
                    "default": "lm", "title": "Algorithm"},
                "options": {"type": "object", "default": {}, "title": "Algorithm options",
                    "description": "Settings of the algorithm, laid over its defaults; a key wins over max_iterations, fd_scheme and fd_relative_step. Keys must be fields of the selected algorithm's settings; method and seed are the job's own."},
                "max_iterations": {"type": ["integer", "null"], "minimum": 1, "default": null,
                    "title": "Iterations",
                    "description": "Iterations (lm, dogleg, nelder_mead), generations (de, cma_es) or rounds (hammer); null: 200, 200, 2000, 100, 200, 20."},
                "fd_scheme": {"type": "string", "enum": ["forward", "central"], "default": "forward",
                    "title": "Finite differences"},
                "fd_relative_step": {"type": ["number", "null"], "exclusiveMinimum": 0, "default": null,
                    "title": "FD relative step",
                    "description": "null chooses per operand and backend (wavefront 1e-6/1e-4, MTF 1e-6/1e-5 on the CPU; device ray operands 1e-6, wavefront 1e-5 forward; docs/OPTIMIZATION.md)."},
                "seed": {"type": "integer", "minimum": 0, "default": 0, "title": "Seed"},
                "backend": {"type": "string", "enum": ["auto", "cpu", "gpu", "gpu_emulation"],
                    "default": "auto", "title": "Merit evaluation",
                    "description": "auto: the device the engine selected for this job (under engine auto, its cost model; docs/GPU.md §9.2), else the CPU reference."},
                "feasibility_tolerance": {"type": "number", "minimum": 0, "default": 1e-6,
                    "title": "Feasibility tolerance",
                    "description": "Total normalised constraint violation regarded as feasible (best-so-far ranking, `final.feasible`)."}
            }
        })
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
                    enumeration(&["converged", "finished", "stopped"], ""),
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
                        operand_schema(),
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
        // Before the merit is prepared, so that a stop meanwhile is kept.
        let stop = input.stop.clone();
        let (s, (algo, max_iter), mut m) =
            OptimizeSettings::prepare(input, settings, &ctx.subrange(0.0, 0.0))?;
        if m.variables.is_empty() {
            return Err(invalid(
                "'variables': none given, and the system has nothing to vary",
            ));
        }
        if m.efl_penalty.is_some() {
            let p = m.problem(m.cpu.clone())?;
            let phi = reference_values(&p, &[&p.start_point()?], ctx)?.merit[0];
            m.weight_efl(phi);
        }
        let reference = m.problem(m.cpu.clone())?;
        let PreparedMerit {
            snapshot: start,
            variables,
            cpu,
            reg,
            spec,
            mut warnings,
            ..
        } = m;
        let system = start.system();
        let x0 = reference.start_point()?;
        let work = Workload::new(&cpu.trace, system, &spec, algo.batch(x0.len()));
        ctx.report(0.0, "preparing the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, &start, work, ctx)?;
        let algo = algo.with_speculation(ev.speculative());
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
                    0.9 * (r.iteration + 1) as f64 / (max_iter + 1) as f64,
                    &format!("{method} {}: merit {:.6e}", r.iteration, r.best_merit),
                );
            });
            mon.keep_history = false;
            mon.feasibility_tol = s.feasibility_tolerance;
            let r = OpticalProblem::new(
                system.clone(),
                variables.clone(),
                spec.clone(),
                ev.evaluator(),
                &reg,
            )
            .and_then(|p| algo.run(&p, &x0, &mon));
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
        ctx.report(0.9, "reference evaluation");
        let refs = reference_values(&reference, &[&x0, &x_search], ctx)?;
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
        ev.finish(input, s.backend, Reported::Reference)?;
        let hash = final_system.hash().to_hex();
        let history = history.lock().map(|h| h.clone()).unwrap_or_default();
        let variables: Vec<Value> = variables
            .iter()
            .zip(reference.variables())
            .enumerate()
            .map(|(j, (v, var))| {
                json!({
                    "label": var.name,
                    "param": v.param,
                    "start": json_nf(x0[j]),
                    "value": json_nf(x[j]),
                    "change": json_nf(x[j] - x0[j]),
                    "lower": v.lower.map(json_nf),
                    "upper": v.upper.map(json_nf),
                    "at_bound": v.lower == Some(x[j]) || v.upper == Some(x[j]),
                })
            })
            .collect();
        let feasible = refs.violation[k] <= feas_tol;
        let json = json!({
            "status": job_status(why == Interrupt::Stopped, Some(result.termination)),
            "stopped": why == Interrupt::Stopped,
            "termination": result.termination,
            "algorithm": method,
            "iterations": result.iterations,
            "evaluations": result.evaluations,
            "improved": improved,
            "start": {"merit": json_nf(refs.merit[0]), "violation": json_nf(refs.violation[0]),
                "status": refs.status[0]},
            "final": {"merit": json_nf(refs.merit[k]), "violation": json_nf(refs.violation[k]),
                "status": refs.status[k], "feasible": feasible},
            "search": {"merit": json_nf(search.0), "violation": json_nf(search.1),
                "reference_merit": json_nf(refs.merit[1]), "reference_violation": json_nf(refs.violation[1])},
            "improvement": json_nf(if refs.merit[0] > 0.0 { 1.0 - refs.merit[k] / refs.merit[0] } else { f64::NAN }),
            "multipliers": result.multipliers.as_ref().map(|m| m.iter().map(|v| json_nf(*v)).collect::<Vec<_>>()),
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

/// `merit_evaluation`: the merit of the start of an `optimize` request,
/// evaluated once.
pub(super) fn evaluate_merit(
    input: &AnalysisInput<'_>,
    settings: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let (s, _, mut m) = OptimizeSettings::prepare(input, settings, &ctx.subrange(0.0, 0.0))?;
    let work = Workload::new(&m.cpu.trace, m.snapshot.system(), &m.spec, 1);
    ctx.report(0.0, "preparing the merit evaluator");
    let mut ev = Evaluation::new(s.backend, input, &m.snapshot, work, &ctx.subrange(0.0, 0.2))?;
    ctx.report(0.2, "evaluating merit");
    let reference = m.problem(m.cpu.clone())?;
    let x0 = reference.start_point()?;
    let batch = CandidateBatch::from_points(x0.len(), [&x0[..]]);
    let values = m
        .problem(ev.evaluator())?
        .operand_values(&batch, &ctx.subrange(0.2, 1.0))?;
    if m.efl_penalty.is_some() {
        // The penalty's weight comes from the reference's start merit, as
        // in `optimize`, whichever evaluator computed the values.
        let phi = match ev.active_gpu() {
            Some(_) => reference_values(&reference, &[&x0], ctx)?.merit[0],
            None => values_report(&reference, values.clone()).merit[0],
        };
        m.weight_efl(phi);
    }
    let r = values_report(&m.problem(m.cpu.clone())?, values);
    ev.finish(input, s.backend, Reported::Evaluator)?;
    ev.warnings.extend(m.warnings);
    Ok(AnalysisOutput {
        json: json!({
            "merit": json_nf(r.merit[0]),
            "violation": json_nf(r.violation[0]),
            "status": r.status[0],
            "operands": operand_table(&m.spec, &r.values[0], &r.values[0], r.merit[0]),
            "merit_spec": m.spec,
            "backend": ev.report,
        }),
        warnings: ev.warnings,
        ..Default::default()
    })
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
    percentiles: Vec<f64>,
    record_trials: usize,
    preview_case: Option<ToleranceCase>,
    irregularity_wavelength_um: Option<f64>,
    random_irregularity_orientation: bool,
    trials: usize,
    seed: u64,
    threshold: Option<f64>,
    batch: usize,
    compensator_iterations: usize,
    backend: BackendChoice,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ToleranceCase {
    Trial { trial: u64 },
    Sensitivity { operand: usize, positive: bool },
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
    /// 3: `predicted_criterion` is the expected (Monte-Carlo mean)
    /// criterion, nominal + `expected_change`, no longer nominal + RSS.
    fn result_version(&self) -> u32 {
        3
    }
    fn name(&self) -> &str {
        "tolerance"
    }
    fn supports_stop(&self) -> bool {
        true
    }
    fn title(&self) -> &str {
        "Tolerance analysis"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Sensitivity of a criterion merit (default: the RMS spot merit) to each tolerance at its \
         limits, ranked worst first with the predicted mean criterion, and a reproducible Monte-Carlo \
         analysis, with compensators re-optimised and the model's solves re-applied for every \
         trial."
    }
    fn settings_schema(&self) -> Value {
        let default_merit = serde_json::to_value(tolerance_default_merit()).unwrap_or(Value::Null);
        json!({
            "type": "object",
            "properties": {
                "tolerances": {"type": "array", "default": [], "title": "Tolerances",
                    "description": "Tolerance operands: {\"param\": ParamPath, \"minus\", \"plus\", \"distribution\", \"label\"}. material_dn_d / material_dv_d perturb catalogue index / Abbe offsets around the saved melt values. Empty: ±0.05 mm on every thickness but the back focus and ±0.5 % on every curvature, without the solve-controlled ones. A solve-controlled parameter is rejected."},
                "compensators": {"type": "array", "default": [], "title": "Compensators",
                    "description": "{\"param\": ParamPath, \"range\", \"scale\"}: re-optimised for every trial. A solve-controlled parameter is rejected."},
                "criterion": {"type": ["object", "null"], "default": null, "title": "Criterion",
                    "description": "A merit-function spec; null generates the default merit for tolerancing from `default_merit`."},
                "default_merit": default_merit_schema(default_merit),
                "mode": {"type": "string", "enum": ["sensitivity", "monte_carlo", "both"],
                    "default": "both", "title": "Analysis"},
                "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 100},
                    "default": [50, 90, 97.7], "title": "Percentiles",
                    "description": "Monte Carlo percentiles of the successful criteria, linearly interpolated."},
                "record_trials": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 0,
                    "title": "Recorded trials", "description": "Keep perturbations and compensators of the first this many trials."},
                "preview_case": {"type": ["object", "null"], "default": null, "title": "Preview case",
                    "description": "Evaluate only this case and return its compensated system: {type:trial,trial} or {type:sensitivity,operand,positive}; zero-based."},
                "irregularity_wavelength_um": {"type": ["number", "null"], "exclusiveMinimum": 0, "default": null, "unit": "µm",
                    "title": "Irregularity test wavelength", "description": "Fringe test wavelength; null keeps an existing overlay's, else the primary wavelength."},
                "random_irregularity_orientation": {"type":"boolean", "default":true,
                    "title":"Random irregularity orientation", "description":"Monte Carlo: independent uniform astigmatism axes in [0,180) degrees per surface and trial, reproducible from the seed. False keeps the model angle; sensitivity always keeps it."},
                "trials": {"type": "integer", "minimum": 1, "default": 100, "title": "Monte-Carlo trials"},
                "seed": {"type": "integer", "minimum": 0, "default": 0, "title": "Seed"},
                "threshold": {"type": ["number", "null"], "default": null, "title": "Yield threshold",
                    "description": "Criterion value for the yield (criterion ≤ threshold)."},
                "batch": {"type": "integer", "minimum": 1, "default": 16, "title": "Trials per batch",
                    "description": "Monte-Carlo trials per evaluation batch, raised to at least 128 (the criteria do not depend on the batching on the CPU reference; on a device, within its precision)."},
                "compensator_iterations": {"type": "integer", "minimum": 1, "default": 20,
                    "title": "Compensator iterations"},
                "backend": {"type": "string", "enum": ["auto", "cpu", "gpu", "gpu_emulation"],
                    "default": "auto", "title": "Merit evaluation"},
            }
        })
    }

    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                (
                    "status",
                    enumeration(&["finished", "stopped"], "Cancel discards the result."),
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
                                    (
                                        "expected_change",
                                        nf_u(
                                            "criterion",
                                            "Expected change over the tolerance's distribution: a·E[δ] + b·E[δ²] of the quadratic through both limits.",
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
                            nf_u(
                                "criterion",
                                "Root sum of squared worst changes (for a criterion linear in the perturbations; not the prediction of a merit).",
                            ),
                        ),
                        (
                            "expected_change",
                            nf_u("criterion", "Sum of the operands' expected changes."),
                        ),
                        (
                            "predicted_criterion",
                            nf_u(
                                "criterion",
                                "Predicted Monte-Carlo mean: nominal + expected_change (independent perturbations add their expected merit changes).",
                            ),
                        ),
                        ("evaluations", int("")),
                        ("cancelled", boolean("Partial sensitivity after a stop.")),
                    ])),
                ),
                (
                    "monte_carlo",
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
                            "seconds_per_trial",
                            num("s", "Wall time over completed trials."),
                        ),
                        (
                            "records",
                            arr(
                                obj(&[
                                    ("trial", int("Zero-based index.")),
                                    ("criterion", nf()),
                                    ("ok", boolean("")),
                                    ("compensators", arr(nf(), "")),
                                    ("perturbations", arr(nf(), "In tolerance order.")),
                                ]),
                                "The first record_trials trials.",
                            ),
                        ),
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
                            boolean("Statistics of completed trials after a stop."),
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
                (
                    "preview",
                    nullable(obj(&[
                        (
                            "system",
                            opaque("Compensated OpticalSystem; apply as a new session snapshot."),
                        ),
                        ("system_hash", string("Content hash of the preview model.")),
                        ("criterion", nf()),
                        ("compensators", arr(nf(), "Final compensator values.")),
                        (
                            "perturbations",
                            arr(nf(), "Selected case in tolerance declaration order."),
                        ),
                    ])),
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
        validate_settings(&self.settings_schema(), settings)?;
        let s: ToleranceSettings = parse(settings, "settings")?;
        let stop = input.stop.clone();
        let mut start = SolvedStart::new(input, &ctx.subrange(0.0, 0.0))?;
        // Every parameter must exist on the nominal system (a missing one
        // would fail every trial).
        let params = s
            .tolerances
            .iter()
            .map(|t| ("tolerances", &t.param))
            .chain(s.compensators.iter().map(|c| ("compensators", &c.param)));
        for (k, (what, p)) in params.enumerate() {
            if let Err(e) = p.get(&start.system) {
                let i = if what == "tolerances" {
                    k
                } else {
                    k - s.tolerances.len()
                };
                return Err(invalid(format!("'{what}'[{i}]: {e}")));
            }
        }
        // Every trial re-solves the solve-controlled parameters (below), so
        // none of them is a tolerance or a compensator.
        let tolerances =
            start.parameters("tolerances", s.tolerances.clone(), auto_tolerances, |t| {
                &t.param
            })?;
        if tolerances.is_empty() {
            return Err(invalid(
                "'tolerances': none given, and the system has nothing to tolerance",
            ));
        }
        let compensators = start.parameters(
            "compensators",
            s.compensators.clone(),
            |_| Vec::new(),
            |c| &c.param,
        )?;
        let mc = MonteCarloSettings {
            trials: s.trials,
            seed: s.seed,
            batch: s.batch.max(TOLERANCE_BATCH),
            threshold: s.threshold,
            percentiles: s.percentiles.clone(),
            record_trials: s.record_trials,
            random_irregularity_orientation: s.random_irregularity_orientation,
        };
        let mut spec = ToleranceSpec::new(tolerances);
        spec.compensators = compensators;
        spec.monte_carlo = mc.clone();
        spec.validate()?;
        // Freeze each irregular surface's nominal clear radius into a zero
        // overlay: every trial then measures fringes over the same disk.
        let irregular: Vec<&ParamPath> = (spec.operands.iter())
            .map(|o| &o.param)
            .filter(|p| matches!(p, ParamPath::Irregularity { .. }))
            .collect();
        if !irregular.is_empty() {
            let system = &mut start.system;
            let radii = TraceEvaluator::new(input.library.clone())
                .tracer(system)
                .map_err(|e| invalid(format!("{e:?}")))?
                .semi_diameters_with(&Default::default(), &ctx.subrange(0.0, 0.0))?;
            for p in irregular {
                let i = system.surface_index(p.surface().unwrap()).unwrap();
                let semi = system.surfaces[i].semi_diameter;
                if semi == SemiDiameter::Auto {
                    system.surfaces[i].semi_diameter = SemiDiameter::Fixed {
                        value: radii.values[i],
                    };
                }
                let fringes = p.get(system)?;
                p.set(system, fringes)?;
                let sf = &mut system.surfaces[i];
                sf.semi_diameter = semi;
                if let (
                    Some(w),
                    SurfaceShape::SagOverlay {
                        test_wavelength_um, ..
                    },
                ) = (s.irregularity_wavelength_um, sf.shape.sag_shape_mut())
                {
                    *test_wavelength_um = w;
                }
            }
        }
        // The criterion, generated for the toleranced parameters and the
        // compensators (`DefaultMeritSpec::generate_for_tolerancing`).
        let changed = (spec.operands.iter().map(|o| &o.param))
            .chain(spec.compensators.iter().map(|c| &c.param))
            .map(|p| VariableSpec::new(p.clone()))
            .collect();
        let m = PreparedMerit::new(
            input,
            start,
            MeritSettings {
                variables: changed,
                merit: s.criterion.clone(),
                default_merit: patched(
                    &tolerance_default_merit(),
                    &s.default_merit,
                    "default_merit",
                )?,
                relax_boundaries: false,
                operands: Vec::new(),
                constraints: Vec::new(),
                hold_efl: false,
                derivative_free: None,
            },
            &ctx.subrange(0.0, 0.0),
        )?;
        let (system, criterion, reg) = (m.snapshot.system(), &m.spec, &m.reg);
        // The model's solves stay in effect in everything the job evaluates:
        // the perturbed system of a trial and each candidate of its
        // compensator solve are re-solved, as a compensator would be.
        let solves = (!system.solves.is_empty()).then_some(&input.library);
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
            mc.batch.min(mc.trials)
        } else {
            TOLERANCE_BATCH.min(2 * spec.operands.len() + 1)
        };
        let work = Workload::new(&m.cpu.trace, system, criterion, batch);
        ctx.report(0.0, "preparing the merit evaluator");
        let mut ev = Evaluation::new(s.backend, input, &m.snapshot, work, ctx)?;
        let lsq = LsqSettings {
            speculative_jacobian: ev.speculative(),
            ..lsq
        };
        if let Some(case) = &s.preview_case {
            let deltas = match case {
                ToleranceCase::Trial { trial } => spec.sample_trial(s.seed, *trial),
                ToleranceCase::Sensitivity { operand, positive } => {
                    let o = spec
                        .operands
                        .get(*operand)
                        .ok_or_else(|| invalid("preview operand out of range"))?;
                    let mut d = vec![0.0; spec.operands.len()];
                    d[*operand] = if *positive { o.plus } else { -o.minus };
                    d
                }
            };
            let (outcome, why) = interruptible(ctx, &stop, |inner| {
                ev.with_evaluator(solves, |e| {
                    let evaluator = OpticalToleranceEvaluator {
                        nominal: system,
                        spec: &spec,
                        criterion,
                        evaluator: &e,
                        registry: reg,
                        lsq,
                    };
                    match case {
                        ToleranceCase::Trial { trial } => evaluator.evaluate_trials(
                            std::slice::from_ref(&deltas),
                            &mc,
                            *trial,
                            inner,
                        ),
                        ToleranceCase::Sensitivity { .. } => {
                            evaluator.evaluate_batch(std::slice::from_ref(&deltas), inner)
                        }
                    }
                })
            });
            ctx.check()?; // Cancel discards; Stop cancelled only the evaluation.
            let preview = match outcome.map(|mut o| o.remove(0)) {
                Ok(outcome) if outcome.ok => {
                    let model = match case {
                        ToleranceCase::Trial { trial } => spec.trial_case_system(
                            system,
                            &deltas,
                            &outcome.compensators,
                            &mc,
                            *trial,
                        )?,
                        ToleranceCase::Sensitivity { .. } => {
                            spec.case_system(system, &deltas, &outcome.compensators)?
                        }
                    };
                    // The case as it was evaluated: with its solves applied.
                    let model = match solves {
                        Some(library) => {
                            mklight_trace::apply_solves_with(
                                &model,
                                library,
                                &Default::default(),
                                &ctx.subrange(0.0, 0.0),
                            )?
                            .0
                        }
                        None => model,
                    };
                    let snapshot = Snapshot::new(model)?;
                    json!({"system": snapshot.system(), "system_hash": snapshot.hash().to_hex(),
                        "criterion": json_nf(outcome.criterion), "compensators": outcome.compensators,
                        "perturbations": deltas})
                }
                Ok(_) | Err(MkError::Cancelled) => Value::Null,
                Err(e) => return Err(e.into()),
            };
            let stopped = why == Interrupt::Stopped;
            ev.finish(input, s.backend, Reported::Evaluator)?;
            ev.warnings.extend(m.warnings);
            return Ok(AnalysisOutput {
                json: json!({"status": job_status(stopped, None), "stopped": stopped,
                    "sensitivity": null, "monte_carlo": null, "preview": preview,
                    "compensators": spec.compensators, "criterion_spec": criterion,
                    "backend": ev.report}),
                warnings: ev.warnings,
                ..Default::default()
            });
        }
        ctx.check()?;
        let split = if do_sens && do_mc {
            0.2
        } else if do_sens {
            1.0
        } else {
            0.0
        };

        let ((sens, mcr), why) = interruptible(ctx, &stop, |inner| {
            ev.with_evaluator(solves, |e| {
                let evaluator = OpticalToleranceEvaluator {
                    nominal: system,
                    spec: &spec,
                    criterion,
                    evaluator: &e,
                    registry: reg,
                    lsq,
                };
                let sens = do_sens.then(|| {
                    let mon = Monitor::new(inner.subrange(0.0, split));
                    mon.progress(0.0, "tolerance sensitivity");
                    sensitivity_batched(&spec, &evaluator, &mon, TOLERANCE_BATCH)
                });
                let cancelled = matches!(sens, Some(Err(MkError::Cancelled)))
                    || matches!(&sens, Some(Ok(r)) if r.cancelled);
                let mcr = (do_mc && !cancelled).then(|| {
                    let mon = Monitor::new(inner.subrange(split, 1.0));
                    monte_carlo(&spec, &evaluator, &mc, &mon)
                });
                (sens, mcr)
            })
        });
        ctx.check()?; // Cancel discards; Stop cancelled only the search.
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
        ev.finish(input, s.backend, Reported::Evaluator)?;
        ev.warnings.extend(m.warnings.iter().cloned());
        let stopped = why == Interrupt::Stopped;
        let label = |i: usize| spec.operands[i].display_label();
        let sens_json = sens.as_ref().map(|r| {
            json!({
                "nominal": {"criterion": json_nf(r.nominal.criterion), "ok": r.nominal.ok,
                    "compensators": r.nominal.compensators.iter().map(|v| json_nf(*v)).collect::<Vec<_>>()},
                "operands": r.operands.iter().map(|o| json!({
                    "index": o.index,
                    "label": label(o.index),
                    "param": spec.operands[o.index].param,
                    "minus": json_nf(spec.operands[o.index].minus),
                    "plus": json_nf(spec.operands[o.index].plus),
                    "criterion_minus": json_nf(o.minus.criterion),
                    "criterion_plus": json_nf(o.plus.criterion),
                    "change_minus": json_nf(o.change_minus),
                    "change_plus": json_nf(o.change_plus),
                    "worst": json_nf(o.worst),
                    "expected_change": json_nf(o.expected_change),
                    "ok": o.minus.ok && o.plus.ok,
                    "compensators_minus": o.minus.compensators.iter().map(|v| json_nf(*v)).collect::<Vec<_>>(),
                    "compensators_plus": o.plus.compensators.iter().map(|v| json_nf(*v)).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "ranking": r.ranking,
                "rss_change": json_nf(r.rss_change),
                "expected_change": json_nf(r.expected_change),
                "predicted_criterion": json_nf(r.predicted_criterion()),
                "evaluations": r.evaluations,
                "cancelled": r.cancelled,
            })
        });
        let mut buffers = Vec::new();
        let mc_json = mcr.as_ref().map(|r| {
            buffers.push(Buffer::f64_1d("monte_carlo_criteria", r.criteria.clone()));
            json!({
                "nominal": {"criterion": json_nf(r.nominal.criterion), "ok": r.nominal.ok,
                    "compensators": r.nominal.compensators},
                "seconds_per_trial": r.seconds_per_trial,
                "records": r.records.iter().map(|t| json!({"trial": t.trial,
                    "criterion": json_nf(t.outcome.criterion), "ok": t.outcome.ok,
                    "compensators": t.outcome.compensators, "perturbations": t.perturbations}))
                    .collect::<Vec<_>>(),
                "trials": r.trials,
                "failures": r.failures,
                "seed": s.seed,
                "mean": json_nf(r.mean),
                "std_dev": json_nf(r.std_dev),
                "min": json_nf(r.min),
                "max": json_nf(r.max),
                "percentiles": r.percentiles.iter().map(|(p, v)| json!({"percentile": p, "value": json_nf(*v)})).collect::<Vec<_>>(),
                "yield_fraction": r.yield_fraction.map(json_nf),
                "threshold": s.threshold.map(json_nf),
                "compensator_mean": r.compensator_mean.iter().map(|v| json_nf(*v)).collect::<Vec<_>>(),
                "compensator_std": r.compensator_std.iter().map(|v| json_nf(*v)).collect::<Vec<_>>(),
                "cancelled": r.cancelled,
                "criteria": r.criteria.iter().map(|v| json_nf(*v)).collect::<Vec<_>>(),
            })
        });
        let json = json!({
            "status": job_status(stopped, None),
            "stopped": stopped,
            "preview": null,
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
    use mklight_core::platform::Instant;
    use mklight_core::{SurfaceId, samples};
    use mklight_gpu::{ComputeBackend, CpuF32Backend};
    use mklight_optim::merit::operand::OperandKind;

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
            default_backend: crate::backend::Backend::CpuReference,
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

    /// A panic in the run reaches the caller: the watcher thread ends with
    /// it instead of keeping the job running.
    #[test]
    fn a_panicking_run_does_not_hang_its_watcher() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let run = || {
                interruptible(&JobContext::new(), &CancelToken::new(), |_| -> () {
                    panic!("boom")
                })
            };
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
            let _ = tx.send(caught.map(|((), why)| why == Interrupt::None));
        });
        let caught = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the watcher never ended");
        let message = caught.unwrap_err();
        assert_eq!(message.downcast_ref::<&str>(), Some(&"boom"));
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

    /// Cancel discards without reference scoring; Stop retains the best point.
    #[test]
    fn cancel_discards_and_stop_returns_best_so_far() {
        let mut s = unconstrained_triplet();
        s["max_iterations"] = json!(200);
        // Direct run, cancelled after two iterations.
        let settings = prepare(&Optimize, &s).unwrap();
        let ctx = JobContext::new();
        let w = cancel_at(&ctx, 2);
        let out = Optimize.run(&input(samples::cooke_triplet()), &settings, &ctx);
        w.join().unwrap();
        assert!(out.unwrap_err().is_cancelled());

        // The job-level stop retains the best point.
        let engine = Engine::new(EngineConfig {
            worker_threads: 1,
            ..Default::default()
        });
        let mut session = engine.new_session();
        session.set_system(samples::cooke_triplet()).unwrap();
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
        engine.stop_job(job).unwrap();
        assert_eq!(engine.wait_job(job, LONG).unwrap(), JobState::Done);
        let r = engine.take_result(job).unwrap();
        assert_eq!(r.json["status"], json!("stopped"));
        assert_eq!(r.json["stopped"], true);
        assert!(r.json["iterations"].as_u64().unwrap() < 200);
        assert!(f(&r.json["final"]["merit"]) < f(&r.json["start"]["merit"]));
    }

    /// Index/Abbe offsets, records and percentiles; a trial preview
    /// reproduces the recorded trial and its glass offsets.
    #[test]
    fn tolerance_catalogue_monte_carlo_and_preview() {
        use mklight_core::materials::lines;
        let system = samples::singlet();
        let settings = json!({"backend":"cpu", "mode":"monte_carlo", "trials":8, "seed":23,
            "record_trials":3, "percentiles":[0,50,97.7,100],
            "tolerances":[
                {"param":ParamPath::MaterialDnD {surface:SurfaceId(1)}, "minus":0.001,"plus":0.001},
                {"param":ParamPath::MaterialDvD {surface:SurfaceId(1)}, "minus":0.5,"plus":0.5}],
            "compensators":[{"param":ParamPath::Thickness {surface:SurfaceId(2)},"range":0.5}]});
        let result = run(system.clone(), "tolerance", settings.clone()).unwrap();
        let mc = &result.json["monte_carlo"];
        assert_eq!(mc["failures"], json!(0));
        assert_eq!(mc["records"].as_array().unwrap().len(), 3);
        assert!(f(&mc["nominal"]["criterion"]).is_finite());
        assert_ne!(mc["criteria"][0], mc["criteria"][1]);
        assert_eq!(mc["percentiles"][0]["value"], mc["min"]);
        assert_eq!(mc["percentiles"][3]["value"], mc["max"]);
        let mut preview_settings = settings;
        preview_settings["preview_case"] = json!({"type":"trial","trial":1});
        let preview = run(system.clone(), "tolerance", preview_settings).unwrap();
        let case = &preview.json["preview"];
        assert_eq!(case["criterion"], mc["criteria"][1]);
        assert_eq!(case["compensators"], mc["records"][1]["compensators"]);
        let model: OpticalSystem = serde_json::from_value(case["system"].clone()).unwrap();
        let snapshot = Snapshot::new(model.clone()).unwrap();
        assert_eq!(
            snapshot.hash().to_hex(),
            case["system_hash"].as_str().unwrap()
        );
        let lib = MaterialLibrary::with_builtin();
        let material = |s: &OpticalSystem| {
            lib.resolve(&s.surfaces[1].material, &s.environment)
                .unwrap()
        };
        let (a, b) = (material(&system), material(&model));
        let vd = |m: &mklight_core::materials::ResolvedMaterial| {
            (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C))
        };
        assert!(
            (b.index(lines::D) - a.index(lines::D) - f(&case["perturbations"][0])).abs() < 3e-16
        );
        assert!((vd(&b) - vd(&a) - f(&case["perturbations"][1])).abs() < 4e-12);
        assert_eq!(preview.run_info.seed, Some(23));
        let mut outside_cap = result.run_info.settings.clone();
        outside_cap["preview_case"] = json!({"type":"trial","trial":5});
        let preview = run(system, "tolerance", outside_cap).unwrap();
        assert_eq!(preview.json["preview"]["criterion"], mc["criteria"][5]);
    }

    #[test]
    fn tolerance_irregularity_and_sensitivity_preview() {
        use mklight_core::param::IrregularityMode;
        for system in [
            samples::singlet(),
            samples::biconic_element(),
            samples::hybrid_diffractive_achromat(),
        ] {
            let id = system.surfaces[2].id;
            let settings = json!({"backend":"auto", "trials":3,
                "irregularity_wavelength_um":0.55,
                "tolerances":[{"param":ParamPath::Irregularity {surface:id,mode:IrregularityMode::Astigmatism},
                    "minus":0.25, "plus":0.25}]});
            let result = run(system.clone(), "tolerance", settings.clone()).unwrap();
            assert_eq!(result.json["monte_carlo"]["failures"], json!(0));
            let sensitivity = &result.json["sensitivity"]["operands"][0];
            assert_eq!(sensitivity["ok"], json!(true));
            assert!(f(&sensitivity["worst"]) > 0.0);
            assert_eq!(
                sensitivity["label"],
                json!(format!("Surface {}: Astigmatic irregularity", id.0))
            );
            let mut settings = settings;
            settings["preview_case"] = json!({"type":"sensitivity","operand":0,"positive":true});
            let preview = run(system.clone(), "tolerance", settings).unwrap();
            assert_eq!(
                preview.json["preview"]["criterion"],
                sensitivity["criterion_plus"]
            );
            let model: OpticalSystem =
                serde_json::from_value(preview.json["preview"]["system"].clone()).unwrap();
            let shape = match &model.surfaces[2].shape {
                SurfaceShape::Diffractive { base, .. } => &**base,
                shape => shape,
            };
            let SurfaceShape::SagOverlay {
                test_wavelength_um,
                astigmatism,
                ..
            } = shape
            else {
                panic!("missing overlay")
            };
            assert_eq!(*test_wavelength_um, 0.55);
            assert_eq!(*astigmatism, 0.25);
        }
    }

    #[test]
    fn tolerance_irregularity_gpu_and_trial_replay() {
        use mklight_core::param::IrregularityMode;
        let mut backends: Vec<(&str, DeviceHandle)> = vec![(
            "gpu_emulation",
            Arc::new(GpuDevice::new(
                "gpu:cpu_f32",
                Arc::new(CpuF32Backend::new()),
            )),
        )];
        if let Some(metal) = mklight_metal::MetalBackend::new().unwrap() {
            backends.push((
                "gpu",
                Arc::new(GpuDevice::new("gpu:metal", Arc::new(metal))),
            ));
        }
        for (backend, device) in backends {
            for system in [
                samples::cooke_triplet(),
                samples::biconic_element(),
                samples::hybrid_diffractive_achromat(),
            ] {
                let id = system.surfaces[2].id;
                let settings = json!({"backend":backend,"mode":"monte_carlo","trials":4,"seed":71,"batch":3,
                    "record_trials":2,"default_merit":{"fields":[0,2],"wavelengths":[1]},
                    "tolerances":[{"param":ParamPath::Irregularity {surface:id,mode:IrregularityMode::Astigmatism},"minus":0.5,"plus":0.5},
                        {"param":ParamPath::Irregularity {surface:id,mode:IrregularityMode::Spherical},"minus":0.25,"plus":0.25}]});
                let registry = AnalysisRegistry::with_builtin();
                let analysis = registry.get("tolerance").unwrap();
                let inp = input(system.clone()).with_device(Some(device.clone()));
                let execute = |settings: &Value| {
                    run_analysis(
                        analysis.as_ref(),
                        &inp,
                        settings,
                        vec![],
                        &JobContext::new(),
                    )
                    .unwrap()
                };
                let result = execute(&settings);
                assert_eq!(
                    result.json["backend"]["precision"], "f32_gpu",
                    "{}",
                    result.json["backend"]
                );
                assert_eq!(result.json["monte_carlo"]["failures"], 0);
                assert!(
                    result.json["backend"]["gpu_stats"]["gpu_candidates"]
                        .as_u64()
                        .unwrap()
                        > 0
                );
                // The four trials are one evaluation batch, reported on the
                // device that evaluated them; only the requested records
                // are kept.
                let b = &result.json["backend"];
                assert_eq!(b["workload"]["batch"], 4);
                assert_eq!(b["reported_on"], b["evaluated_on"]);
                assert_eq!(
                    result.json["monte_carlo"]["records"]
                        .as_array()
                        .unwrap()
                        .len(),
                    2
                );
                let mut preview_settings = settings.clone();
                preview_settings["preview_case"] = json!({"type":"trial","trial":2});
                let preview = execute(&preview_settings);
                assert_eq!(
                    preview.json["preview"]["criterion"],
                    result.json["monte_carlo"]["criteria"][2]
                );
                let model: OpticalSystem =
                    serde_json::from_value(preview.json["preview"]["system"].clone()).unwrap();
                let SurfaceShape::SagOverlay {
                    astigmatism_angle_deg,
                    ..
                } = model.surfaces[2].shape.sag_shape()
                else {
                    panic!("missing overlay")
                };
                assert!(
                    (0.0..180.0).contains(astigmatism_angle_deg) && *astigmatism_angle_deg != 0.0
                );
                // The reconstructed system's f64 criterion agrees.
                let reference = run(
                    model,
                    "merit_evaluation",
                    json!({"backend":"cpu","merit":result.json["criterion_spec"],"hold_efl":false}),
                )
                .unwrap();
                let (got, want) = (
                    f(&preview.json["preview"]["criterion"]),
                    f(&reference.json["merit"]),
                );
                assert!(
                    (got - want).abs() <= 1e-3 * want.max(1e-12),
                    "{backend}: {got} vs {want}"
                );
                preview_settings["random_irregularity_orientation"] = json!(false);
                let fixed = execute(&preview_settings);
                let model: OpticalSystem =
                    serde_json::from_value(fixed.json["preview"]["system"].clone()).unwrap();
                let SurfaceShape::SagOverlay {
                    astigmatism_angle_deg,
                    ..
                } = model.surfaces[2].shape.sag_shape()
                else {
                    panic!("missing overlay")
                };
                assert_eq!(*astigmatism_angle_deg, 0.0);
            }
        }
    }

    /// The predicted criterion is the Monte-Carlo mean of the merit (a mean
    /// square: independent perturbations add their expected changes), not
    /// nominal + RSS. Cooke triplet, ten end-point tolerances, no
    /// compensator.
    #[test]
    fn tolerance_prediction_matches_monte_carlo_mean() {
        let sys = samples::cooke_triplet();
        let n = sys.surfaces.len();
        let mut tols = Vec::new();
        for sf in &sys.surfaces[1..n - 2] {
            let id = sf.id;
            tols.push(json!({"param": ParamPath::Curvature { surface: id },
                "minus": 2e-4, "plus": 2e-4, "distribution": {"type": "end_point"}}));
            tols.push(json!({"param": ParamPath::Thickness { surface: id },
                "minus": 0.05, "plus": 0.05, "distribution": {"type": "end_point"}}));
        }
        assert_eq!(tols.len(), 10);
        let r = run(
            sys,
            "tolerance",
            json!({"tolerances": tols, "mode": "both",
            "trials": 2000, "seed": 3, "backend": "cpu"}),
        )
        .unwrap();
        let sens = &r.json["sensitivity"];
        let mc = &r.json["monte_carlo"];
        assert_eq!(mc["failures"], json!(0));
        let nominal = f(&sens["nominal"]["criterion"]);
        let predicted = f(&sens["predicted_criterion"]);
        let expected: f64 = (sens["operands"].as_array().unwrap().iter())
            .map(|o| f(&o["expected_change"]))
            .sum();
        assert!((f(&sens["expected_change"]) - expected).abs() < 1e-12 * expected);
        assert_eq!(predicted, nominal + f(&sens["expected_change"]));
        // End points: each expected change is the mean of the two limits.
        for o in sens["operands"].as_array().unwrap() {
            let mean = (f(&o["change_minus"]) + f(&o["change_plus"])) / 2.0;
            assert!((f(&o["expected_change"]) - mean).abs() <= 1e-12 * mean.abs());
        }
        let mean = f(&mc["mean"]);
        let se = f(&mc["std_dev"]) / 2000f64.sqrt();
        eprintln!(
            "nominal {nominal:e} predicted {predicted:e} rss {:e} mc mean {mean:e} ± {se:e}",
            nominal + f(&sens["rss_change"])
        );
        assert!(
            (predicted - mean).abs() < 0.01 * mean && (predicted - mean).abs() < 4.0 * se,
            "predicted {predicted:e}, Monte-Carlo mean {mean:e} ± {se:e}"
        );
        // Nominal + RSS (the former prediction) is far too optimistic.
        let rss = nominal + f(&sens["rss_change"]);
        assert!(rss < 0.75 * mean, "{rss:e} vs {mean:e}");
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
        assert_eq!(j["status"], json!("finished"));
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
        assert_eq!(mc["nominal"]["criterion"], sens["nominal"]["criterion"]);
        assert_eq!(a.buffers[0].name, "monte_carlo_criteria");
        assert_eq!(a.run_info.seed, Some(3));
        let b = run(sys.clone(), "tolerance", settings(3)).unwrap();
        assert_eq!(mc["criteria"], b.json["monte_carlo"]["criteria"]);
        let other = run(sys.clone(), "tolerance", settings(4)).unwrap();
        assert_ne!(mc["criteria"], other.json["monte_carlo"]["criteria"]);
    }

    /// The GPU path in the engine: the emulation backend everywhere, the
    /// Metal device when present. Search on the GPU, reported numbers from
    /// the CPU reference, with actual device execution metadata in the result.
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
        // Backend `auto` evaluates on the device handed over, at a small and
        // at a larger candidate batch.
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
        assert_eq!(b["precision"], json!("f32_gpu"), "{b}");
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

    /// Bad input is rejected: a zero or negative FD step, a tolerance or
    /// compensator on a surface that does not exist.
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
    }

    /// Stopping the tolerance job in the sensitivity phase keeps the
    /// operands evaluated so far.
    #[test]
    fn stopped_sensitivity_keeps_completed_operands() {
        let sys = samples::cooke_triplet();
        let settings = prepare(
            &Tolerance,
            &json!({"mode": "sensitivity", "compensators": [{"param": ParamPath::Thickness { surface: SurfaceId(6) }, "range": 1.0}],
                "compensator_iterations": 4}),
        )
        .unwrap();
        // Stop (the job's stop token) once the first sensitivity batch has
        // been reported.
        let ctx = JobContext::new();
        let stop = CancelToken::new();
        let watch = {
            let (ctx, stop) = (ctx.clone(), stop.clone());
            std::thread::spawn(move || {
                let t = Instant::now();
                while t.elapsed() < Duration::from_secs(600) {
                    if ctx.progress.fraction() > 0.0 {
                        stop.cancel();
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        // 90 tolerances: 181 points, two sensitivity batches of 128 and 53
        // (`TOLERANCE_BATCH`), so that the stop lands between them.
        let mut tol = Vec::new();
        for (id, size) in [1u64, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(|id| [1e-3, 2e-3, 3e-3].map(|size| (id, size)))
        {
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
                tol.push(json!({"param": p, "minus": size, "plus": size}));
            }
        }
        let mut settings = settings;
        settings["tolerances"] = json!(tol);
        let out = Tolerance
            .run(&input(sys).with_stop_token(stop), &settings, &ctx)
            .unwrap();
        watch.join().unwrap();
        let j = &out.json;
        assert_eq!(j["status"], json!("stopped"));
        let sens = &j["sensitivity"];
        let n = sens["operands"].as_array().unwrap().len();
        // The last batch may finish before the observing thread runs. A stop
        // keeps every completed operand, including a just-completed full set.
        assert!(n > 0 && n <= 90, "{n} of 90 operands");
        assert_eq!(sens["cancelled"], json!(n < 90), "{sens}");
        assert_eq!(sens["ranking"].as_array().unwrap().len(), n);
    }

    /// Cancelling ends every solver job with `Cancelled` and no result; the
    /// statuses of the results are those of `job_status`.
    #[test]
    fn cancel_is_an_error_in_every_job() {
        assert_eq!(job_status(true, Some(Termination::SmallStep)), "stopped");
        assert_eq!(job_status(false, Some(Termination::SmallStep)), "converged");
        assert_eq!(
            job_status(false, Some(Termination::MaxIterations)),
            "finished"
        );
        assert_eq!(job_status(false, Some(Termination::Cancelled)), "finished");
        assert_eq!(job_status(false, None), "finished");
        assert_eq!(job_status(true, None), "stopped");
        let reg = AnalysisRegistry::with_builtin();
        for (name, statuses) in [
            ("optimize", json!(["converged", "finished", "stopped"])),
            ("tolerance", json!(["finished", "stopped"])),
            ("glass_substitution", json!(["finished", "stopped"])),
        ] {
            let schema = reg.get(name).unwrap().result_schema();
            assert_eq!(schema["properties"]["status"]["enum"], statuses, "{name}");
        }
        // A compensated sensitivity and Monte Carlo, cancelled once the
        // first batch has been reported.
        let c = |id: u64| json!({"param": ParamPath::Curvature { surface: SurfaceId(id) }, "minus": 1e-4, "plus": 1e-4});
        let settings = prepare(
            &Tolerance,
            &json!({"tolerances": [c(1), c(2), c(3), c(4), c(5), c(6)], "trials": 100_000,
                "compensators": [{"param": ParamPath::Thickness { surface: SurfaceId(6) }, "range": 1.0}]}),
        )
        .unwrap();
        for preview in [false, true] {
            let mut settings = settings.clone();
            if preview {
                settings["preview_case"] = json!({"type": "trial", "trial": 0});
            }
            let ctx = JobContext::new();
            let watch = {
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    let t = Instant::now();
                    while t.elapsed() < Duration::from_secs(600) {
                        if preview || ctx.progress.fraction() > 0.0 {
                            ctx.cancel.cancel();
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                })
            };
            let out = Tolerance.run(&input(samples::cooke_triplet()), &settings, &ctx);
            watch.join().unwrap();
            assert!(out.unwrap_err().is_cancelled(), "preview {preview}");
        }
    }

    /// A triplet whose last curvature holds F/4.2 and whose image sits at
    /// the paraxial focus. The stored values are the sample's, which meets
    /// neither condition.
    fn solved_triplet() -> OpticalSystem {
        let mut sys = samples::cooke_triplet();
        sys.solves = serde_json::from_value(json!([
            {"target": {"param": "curvature", "surface": 6},
                "kind": {"type": "f_number", "value": 4.2}},
            {"target": {"param": "thickness", "surface": 6},
                "kind": {"type": "marginal_ray_height", "height": 0.0}},
        ]))
        .unwrap();
        sys
    }

    /// The system with its solves applied.
    fn solved(sys: &OpticalSystem) -> OpticalSystem {
        mklight_trace::apply_solves(sys, &MaterialLibrary::with_builtin())
            .unwrap()
            .0
    }

    /// The image's distance from the paraxial focus (object at infinity)
    /// and the image-space F/# at the triplet's 12.5 mm entrance pupil.
    fn focus_and_f_number(sys: &OpticalSystem) -> (f64, f64) {
        let ops = [
            Operand::new(OperandKind::Bfl),
            Operand::new(OperandKind::Efl),
        ];
        let v = TraceEvaluator::builtin()
            .evaluate_system(sys, &[&ops[0], &ops[1]], &JobContext::new())
            .unwrap();
        let back_focus = sys.surfaces[sys.surfaces.len() - 2].thickness;
        (
            v[0].clone().unwrap() - back_focus,
            v[1].clone().unwrap() / 12.5,
        )
    }

    /// On a design with solves the four jobs work on the same model: the
    /// solved start, with the same start merit. No job changes a
    /// solve-controlled parameter itself, and each system a job derives is
    /// solved where its meaning asks for it: every tolerance case and every
    /// substituted glass combination.
    #[test]
    fn jobs_work_on_the_same_solved_model() {
        let sys = solved_triplet();
        let start = solved(&sys);
        let (c6, t6) = (curv(6)["param"].clone(), thick(6)["param"].clone());
        let value = |sys: &OpticalSystem, p: &Value| {
            serde_json::from_value::<ParamPath>(p.clone())
                .unwrap()
                .get(sys)
                .unwrap()
        };
        // The solves move both targets off the stored values.
        assert!((value(&start, &c6) - value(&sys, &c6)).abs() > 1e-4);
        assert!((value(&start, &t6) - value(&sys, &t6)).abs() > 1e-2);
        let (focus, f_number) = focus_and_f_number(&start);
        assert!(focus.abs() < 1e-9 && (f_number - 4.2).abs() < 1e-9);

        // One explicit merit for every job: the start merit is that of the
        // solved system, in all four.
        let merit = json!({"hold_efl": false, "backend": "cpu",
            "default_merit": {"fields": [0, 2], "wavelengths": [1], "thickness": null}});
        let e = run(sys.clone(), "merit_evaluation", merit.clone()).unwrap();
        let spec = e.json["merit_spec"].clone();
        let m0 = e.json["merit"].clone();
        let mut unsolved = start.clone();
        unsolved.solves.clear();
        let on_solved = run(unsolved, "merit_evaluation", merit.clone()).unwrap();
        assert_eq!(m0, on_solved.json["merit"]);
        let mut stored = sys.clone();
        stored.solves.clear();
        let on_stored = run(stored, "merit_evaluation", merit).unwrap();
        assert!((f(&m0) / f(&on_stored.json["merit"]) - 1.0).abs() > 0.1);

        let o = run(
            sys.clone(),
            "optimize",
            json!({"merit": spec, "hold_efl": false, "backend": "cpu", "max_iterations": 2}),
        )
        .unwrap();
        assert_eq!(o.json["start"]["merit"], m0);
        let thickness = |id: u64, tol: f64| json!({"param": ParamPath::Thickness { surface: SurfaceId(id) }, "minus": tol, "plus": tol});
        let tolerance = json!({"criterion": spec, "backend": "cpu", "trials": 4, "seed": 5,
            "record_trials": 4, "tolerances": [thickness(1, 0.05), thickness(4, 0.05)]});
        let t = run(sys.clone(), "tolerance", tolerance.clone()).unwrap();
        assert_eq!(t.json["sensitivity"]["nominal"]["criterion"], m0);
        assert_eq!(t.json["monte_carlo"]["nominal"]["criterion"], m0);
        let glass = json!({"merit": spec, "hold_efl": false, "backend": "cpu",
            "material_variables": [{"surface": 1, "candidates": {"nearest": 3}}],
            "max_combinations": 3, "local_iterations": 2, "top_n": 3});
        let g = run(sys.clone(), "glass_substitution", glass.clone()).unwrap();
        assert_eq!(g.json["starting_merit"], m0);

        // Automatic variables and tolerances skip the solve-controlled
        // parameters; explicit ones are rejected, in every job.
        let params = |rows: &Value| -> Vec<Value> {
            (rows.as_array().unwrap().iter())
                .map(|v| v["param"].clone())
                .collect()
        };
        let auto = run(sys.clone(), "optimize", json!({"max_iterations": 1})).unwrap();
        let varied = params(&auto.json["variables"]);
        assert_eq!(varied.len(), 5, "{varied:?}");
        assert!(!varied.contains(&c6) && !varied.contains(&t6));
        let mut auto = glass.clone();
        auto["max_combinations"] = json!(1);
        let auto = run(sys.clone(), "glass_substitution", auto).unwrap();
        assert_eq!(params(&auto.json["results"][0]["variables"]), varied);
        let auto = run(
            sys.clone(),
            "tolerance",
            json!({"mode": "sensitivity", "backend": "cpu"}),
        )
        .unwrap();
        let toleranced = params(&auto.json["sensitivity"]["operands"]);
        assert_eq!(toleranced.len(), 10, "{toleranced:?}");
        assert!(!toleranced.contains(&c6) && !toleranced.contains(&t6));
        for (name, settings) in [
            ("optimize", json!({"variables": [curv(1), thick(6)]})),
            ("merit_evaluation", json!({"variables": [curv(6)]})),
            (
                "glass_substitution",
                json!({"variables": [curv(6)],
                "material_variables": [{"surface": 1}]}),
            ),
            (
                "tolerance",
                json!({"tolerances": [thickness(1, 0.05), thickness(6, 0.05)]}),
            ),
            (
                "tolerance",
                json!({"tolerances": [thickness(1, 0.05)],
                "compensators": [{"param": t6, "range": 1.0}]}),
            ),
        ] {
            match run(sys.clone(), name, settings.clone()) {
                Err(EngineError::InvalidSettings(m)) => {
                    assert!(m.contains("a solve controls"), "{name}: {m}")
                }
                other => panic!("{name} accepted {settings}: {:?}", other.map(|r| r.json)),
            }
        }

        // Tolerance: every case is re-solved. A sensitivity limit, and a
        // trial with a compensator, move both solved parameters; the
        // previewed system meets its solves, applying them changes nothing,
        // and its merit is the reported criterion.
        let mut compensated = tolerance.clone();
        compensated["compensators"] =
            json!([{"param": ParamPath::Thickness { surface: SurfaceId(2) }, "range": 0.5}]);
        for (settings, case) in [
            (
                &tolerance,
                json!({"type": "sensitivity", "operand": 0, "positive": true}),
            ),
            (&tolerance, json!({"type": "trial", "trial": 2})),
            (&compensated, json!({"type": "trial", "trial": 2})),
        ] {
            let mut preview = settings.clone();
            preview["preview_case"] = case.clone();
            let p = run(sys.clone(), "tolerance", preview).unwrap();
            let p = &p.json["preview"];
            let model: OpticalSystem = serde_json::from_value(p["system"].clone()).unwrap();
            assert!(
                (value(&model, &c6) - value(&start, &c6)).abs() > 1e-7,
                "{case}"
            );
            assert!(
                (value(&model, &t6) - value(&start, &t6)).abs() > 1e-4,
                "{case}"
            );
            let (focus, f_number) = focus_and_f_number(&model);
            assert!(
                focus.abs() < 1e-9 && (f_number - 4.2).abs() < 1e-9,
                "{case}"
            );
            assert_eq!(solved(&model), model, "{case}");
            let again = run(
                model,
                "merit_evaluation",
                json!({"merit": spec, "hold_efl": false, "backend": "cpu"}),
            )
            .unwrap();
            assert_eq!(again.json["merit"], p["criterion"], "{case}");
            if case["type"] == "trial" {
                // The preview is the trial of the run.
                let all = run(sys.clone(), "tolerance", settings.clone()).unwrap();
                assert_eq!(all.json["monte_carlo"]["criteria"][2], p["criterion"]);
                assert_eq!(all.json["monte_carlo"]["failures"], 0);
            }
        }

        // On a device too: the cases reach it re-solved, plain and through
        // the joined batches of the compensator solves.
        let emulation: DeviceHandle = Arc::new(GpuDevice::new(
            "gpu:cpu_f32",
            Arc::new(CpuF32Backend::new()),
        ));
        let reg = AnalysisRegistry::with_builtin();
        let on_device = |settings: &Value| {
            let mut settings = settings.clone();
            settings["backend"] = json!("gpu_emulation");
            let inp = input(sys.clone()).with_device(Some(emulation.clone()));
            let analysis = reg.get("tolerance").unwrap();
            let r = run_analysis(
                analysis.as_ref(),
                &inp,
                &settings,
                vec![],
                &JobContext::new(),
            );
            let r = r.unwrap();
            assert_eq!(r.json["backend"]["precision"], "f32_gpu");
            assert_eq!(r.json["monte_carlo"]["failures"], 0);
            r.json
        };
        let d = on_device(&tolerance);
        let criteria = |j: &Value| -> Vec<f64> {
            let mut all = vec![f(&j["sensitivity"]["nominal"]["criterion"])];
            all.extend(
                j["monte_carlo"]["criteria"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(f),
            );
            all
        };
        for (a, b) in criteria(&d).iter().zip(criteria(&t.json)) {
            assert!((a / b - 1.0).abs() < 1e-6, "{a} vs {b}");
        }
        on_device(&compensated);

        // Glass substitution: each combination starts from the substituted
        // design with its solves applied, which its re-optimisation (as
        // `optimize`'s search) leaves alone.
        let rows = g.json["results"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let mut moved = 0;
        for row in rows {
            let model: OpticalSystem = serde_json::from_value(row["system"].clone()).unwrap();
            let mut substituted = start.clone();
            substituted.surfaces[1].material = model.surfaces[1].material.clone();
            let expected = solved(&substituted);
            let (focus, f_number) = focus_and_f_number(&expected);
            assert!(focus.abs() < 1e-9 && (f_number - 4.2).abs() < 1e-9);
            assert_eq!(
                value(&model, &c6),
                value(&expected, &c6),
                "{}",
                row["glasses"]
            );
            assert_eq!(
                value(&model, &t6),
                value(&expected, &t6),
                "{}",
                row["glasses"]
            );
            moved += usize::from(value(&model, &c6) != value(&start, &c6));
        }
        assert_eq!(moved, 2, "the two other glasses are re-solved");
    }

    /// A misspelt key inside a nested settings object is an error in every
    /// job, as at the top level; unknown `options` name the valid keys of
    /// the selected algorithm.
    #[test]
    fn misspelt_nested_settings_are_rejected() {
        let sys = samples::cooke_triplet();
        let rejected = |name: &str, settings: Value| -> String {
            match run(sys.clone(), name, settings.clone()) {
                Err(EngineError::InvalidSettings(m)) => m,
                other => panic!("{name} accepted {settings}: {:?}", other.map(|r| r.json)),
            }
        };
        let mtf = |key: &str| {
            json!({"default_merit": {"criterion":
                {"type": "mtf", "frequencies": [30.0], "target": 0.5, key: true}}})
        };
        for name in [
            "optimize",
            "merit_evaluation",
            "tolerance",
            "glass_substitution",
        ] {
            for (settings, word) in [
                (json!({"default_merit": {"feilds": [0]}}), "feilds"),
                (json!({"default_merit": {"sampling": {"rngs": 4}}}), "rngs"),
                (
                    json!({"default_merit": {"sampling": {"rings": "four"}}}),
                    "rings",
                ),
                (
                    json!({"default_merit": {"thickness": {"glass_min_center": 0.9}}}),
                    "glass_min_center",
                ),
                (mtf("worst_feild"), "worst_feild"),
                (
                    json!({"default_merit": {"criterion": {"type": "rms_spot", "frequencies": [30.0]}}}),
                    "frequencies",
                ),
                (
                    json!({"default_merit": {"criterion": {"type": "strehl"}}}),
                    "strehl",
                ),
            ] {
                let m = rejected(name, settings);
                assert!(m.contains(word), "{name}: {m}");
            }
        }
        // The same objects, spelt right, are accepted.
        let good = json!({"max_iterations": 1, "default_merit": {"fields": [0],
            "sampling": {"rings": 4, "arms": 6}, "thickness": {"glass_min_centre": 0.9},
            "criterion": {"type": "rms_spot", "reference": "chief", "scope": "polychromatic"}}});
        run(sys.clone(), "optimize", good).unwrap();

        // `options`: the fields of the selected algorithm's settings.
        for name in ["optimize", "merit_evaluation"] {
            for (settings, word) in [
                (
                    json!({"options": {"max_iteratons": 5}}),
                    "valid: constraints, ftol",
                ),
                (
                    json!({"options": {"jacobian": {"shceme": "central"}}}),
                    "relative_step, scheme",
                ),
                (
                    json!({"options": {"constraints": {"mu": 1.0}}}),
                    "max_inner_iterations",
                ),
                (
                    json!({"algorithm": "de", "options": {"gtol": 0.0}}),
                    "valid: cr, f_max",
                ),
                (
                    json!({"algorithm": "hammer", "options": {"local": {"rounds": 3}}}),
                    "'options.local'",
                ),
                (
                    json!({"options": {"speculative_jacobian": true}}),
                    "unknown key",
                ),
                // Fields the job sets from settings of its own.
                (
                    json!({"options": {"method": "dogleg"}}),
                    "'algorithm' setting",
                ),
                (
                    json!({"algorithm": "cma_es", "options": {"seed": 3}}),
                    "'seed' setting",
                ),
            ] {
                let m = rejected(name, settings);
                assert!(m.contains(word), "{name}: {m}");
            }
        }
        // Every field of an algorithm's settings is a valid key, and one
        // given in `options` wins over the job's own setting for it.
        let lsq = serde_json::to_value(LsqSettings::default()).unwrap();
        let mut all = lsq.clone();
        all.as_object_mut().unwrap().remove("method");
        all["max_iterations"] = json!(2);
        let mut hammer = serde_json::to_value(HammerSettings::default()).unwrap();
        hammer.as_object_mut().unwrap().remove("seed");
        hammer["rounds"] = json!(1);
        hammer["local"] = lsq;
        hammer["local"]["max_iterations"] = json!(2);
        hammer["local"]["jacobian"]["scheme"] = json!("central");
        for (algorithm, options, iterations) in [("lm", all, 2), ("hammer", hammer, 1)] {
            let r = run(
                sys.clone(),
                "optimize",
                json!({"algorithm": algorithm, "options": options, "max_iterations": 7,
                    "default_merit": {"fields": [0], "wavelengths": [1], "thickness": null},
                    "hold_efl": false}),
            )
            .unwrap();
            assert!(
                r.json["iterations"].as_u64().unwrap() <= iterations,
                "{algorithm}: {}",
                r.json["iterations"]
            );
        }
        let s: OptimizeSettings = parse(
            &prepare(
                &Optimize,
                &json!({"algorithm": "hammer", "fd_scheme": "forward", "seed": 9,
                    "options": {"local": {"jacobian": {"scheme": "central"}}}}),
            )
            .unwrap(),
            "settings",
        )
        .unwrap();
        let (Algo::Hammer(h), _) = Algo::new(&s).unwrap() else {
            panic!("hammer")
        };
        assert_eq!(h.local.jacobian.scheme, FdScheme::Central);
        assert_eq!(h.seed, 9);
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

    #[test]
    fn image_quality_merits_run_in_optimize_and_tolerance() {
        let system = samples::cooke_triplet();
        let variable = VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(6),
        });
        let merit = json!({"operands": [{"kind": {"type": "mtf", "frequency": 30},
            "field": {"field": "index", "index": 1}, "wavelength": {"wavelength": "all"},
            "target": 0.8, "comparison": "greater_than"}]});
        let default = json!({"criterion": {"type": "mtf", "frequencies": [10,30],
            "target": 0.8, "worst_field": true}, "fields": [0,1], "thickness": null});
        for settings in [
            json!({"merit": merit, "variables": [variable.clone()], "hold_efl": false, "backend": "cpu", "max_iterations": 1}),
            json!({"default_merit": default, "variables": [variable], "hold_efl": false, "backend": "gpu_emulation", "max_iterations": 1}),
        ] {
            let r = run(system.clone(), "optimize", settings).unwrap();
            assert_eq!(r.json["start"]["status"]["status"], "ok");
            assert_eq!(r.json["final"]["status"]["status"], "ok");
            let rows: Vec<_> = r.json["operands"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|o| o["kind"] == "mtf")
                .collect();
            assert!(!rows.is_empty());
            assert!(
                rows.iter()
                    .all(|o| o["start"].as_f64().is_some_and(|v| v > 0.0 && v < 1.0))
            );
        }
        let r = run(system, "tolerance", json!({"criterion": merit, "mode": "sensitivity", "backend": "cpu",
            "tolerances": [{"param": ParamPath::Thickness { surface: SurfaceId(6) }, "minus": 0.01, "plus": 0.01}]})).unwrap();
        assert_eq!(r.json["sensitivity"]["nominal"]["ok"], true);
        assert_eq!(r.json["sensitivity"]["operands"][0]["ok"], true);
    }

    #[test]
    fn axis_defaults_optimise_squeeze_and_tolerance_both_curvatures() {
        let sys = samples::biconic_element();
        let fy = efl(&sys);
        let merit = MeritFunctionSpec::new(vec![
            Operand::new(OperandKind::AnamorphicSqueeze).target(1.4),
            Operand::new(OperandKind::EflY).target(fy).as_constraint(),
        ]);
        let r = run(
            sys.clone(),
            "optimize",
            json!({"merit": merit, "backend": "cpu"}),
        )
        .unwrap();
        let best: OpticalSystem = serde_json::from_value(r.json["system"].clone()).unwrap();
        let values = TraceEvaluator::builtin()
            .evaluate_system(
                &best,
                &[
                    &Operand::new(OperandKind::AnamorphicSqueeze),
                    &Operand::new(OperandKind::EflY),
                ],
                &JobContext::new(),
            )
            .unwrap();
        assert!((values[0].as_ref().unwrap() - 1.4).abs() < 1e-6);
        assert!((values[1].as_ref().unwrap() - fy).abs() < 1e-6);
        assert!(
            !r.json["operands"]
                .as_array()
                .unwrap()
                .iter()
                .any(|o| o["kind"] == "efl")
        );
        let r = run(
            sys,
            "tolerance",
            json!({"mode": "sensitivity", "backend": "cpu"}),
        )
        .unwrap();
        let rows = r.json["sensitivity"]["operands"].as_array().unwrap();
        assert!(rows.iter().all(|o| o["ok"] == true));
        for parameter in ["curvature_x", "curvature_y"] {
            assert!(
                rows.iter()
                    .any(|o| o["param"]["parameter"] == parameter && f(&o["worst"]) > 0.0)
            );
        }
    }

    /// 20 LM iterations on the Double Gauss with the worst-field MTF merit
    /// end 3e-3 off the EFL hold; before the end-point restoration the run
    /// returned the start design.
    #[test]
    #[ignore = "diffraction MTF merit; run in release mode"]
    fn double_gauss_mtf_end_point_is_restored() {
        let r = run(
            samples::double_gauss(),
            "optimize",
            json!({
                "backend": "cpu", "max_iterations": 20,
                "default_merit": {"criterion": {"type": "mtf", "frequencies": [30.0],
                    "target": 0.8, "worst_field": true}}
            }),
        )
        .unwrap()
        .json;
        assert_eq!(r["improved"], true);
        assert_eq!(r["final"]["feasible"], true);
        assert!(f(&r["final"]["merit"]) < 0.9 * f(&r["start"]["merit"]));
    }
}
