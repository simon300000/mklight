//! Static capability, the cost model's `auto` route and native-attempt
//! accounting; no runtime f64 verification.
//!
//! * `auto_route` settles every `auto` request from provider hints and the
//!   cost model (`cost`), opening the device only when the predicted route
//!   is native or the device family cannot be known without it.
//! * `run` executes the selected route, skips native paths that already
//!   failed deterministically on the same input and device, and records
//!   every attempt that did not produce the result
//!   (`RunInfo::failed_attempts`).
use std::time::Instant;

use super::cost::{self, CostModel, Device, Route, State};
use super::{Analysis, AnalysisInput, AnalysisOutput, BackendUse, gpu};
use crate::backend::{AutoCandidate, Backend, BackendInfo, BackendRegistry, SelectedBackend};
use crate::error::{EngineError, EngineResult};
use crate::gpu::{GpuDevice, KnownFailure};
use mklight_core::job::{BackendAttempt, Precision};
use mklight_core::{JobContext, MkError, OpticalSystem};
use serde_json::{Value, json};

/// Analyses whose Metal output failed the f64 comparison on most P11 default
/// samples (GPU §9.2) always run on the reference. Sporadic disagreements are
/// documented residual gaps, not runtime gates.
pub(crate) fn reference_reason(analysis: &str) -> Option<&'static str> {
    match analysis {
        "encircled_energy" => Some(
            "static capability: P11 encircled_energy failed f64 comparison on 16/16 default samples (geometry-uncertainty bin completion)",
        ),
        "wavefront_map" => Some(
            "static capability: P11 wavefront_map failed f64 comparison on 16/16 default samples (undefined on-axis Seidel angles)",
        ),
        "polarization_pupil_map" => Some(
            "static capability: P11 polarization_pupil_map failed f64 comparison on 13/16 default samples (undefined polarization axes)",
        ),
        _ => None,
    }
}

fn static_capability(reason: &str) -> Value {
    json!({"decision":"cpu_reference","rule":"static capability","reason":reason})
}

/// One `auto` request as the router sees it.
pub(crate) struct AutoRequest<'a> {
    pub analysis: &'a str,
    pub settings: &'a Value,
    pub system: &'a OpticalSystem,
    pub snapshot: &'a [u8; 32],
    pub library: &'a str,
    /// The graph holds the snapshot's reference tracer.
    pub tracer_warm: bool,
}

/// The `auto` selection: the reference, decided before any device opens
/// when the hints suffice, or the device with the cost decision. Never
/// fails (an unavailable device is the reference with its reason).
pub(crate) fn auto_route(
    model: &CostModel,
    registry: &BackendRegistry,
    r: &AutoRequest<'_>,
) -> SelectedBackend {
    let cpu = |note: &str, mut decision: Value, before_device: bool| {
        let mut s = SelectedBackend::cpu_reference();
        s.requested = Backend::Auto.id();
        s.note = Some(note.into());
        if !decision.is_null() {
            decision["before_device"] = json!(before_device);
            s.cost_decision = Some(decision);
        }
        s
    };
    let state = |device_open, known_failure| State {
        snapshot: Some(r.snapshot),
        library: r.library,
        device_open,
        tracer_warm: r.tracer_warm,
        known_failure,
    };
    let (apple, open) = match registry.auto_candidate(r.analysis) {
        Some(AutoCandidate::NoGpu(note)) => return cpu(&note, Value::Null, true),
        Some(AutoCandidate::Gpu { apple_gpu, open }) => (apple_gpu, open),
        None => (None, false),
    };
    if let Some(reason) = reference_reason(r.analysis) {
        return cpu(reason, static_capability(reason), !open);
    }
    let hinted = model.decide(
        r.analysis,
        r.settings,
        r.system,
        Device::hinted(apple),
        &state(open, false),
    );
    if hinted.route == Route::Cpu {
        return cpu(hinted.note(), hinted.json, !open);
    }
    // Native, or the family needs the device: open it.
    let selected = registry.select_auto(Some(r.analysis));
    let Some(device) = selected
        .device
        .clone()
        .and_then(|d| d.downcast::<GpuDevice>().ok())
    else {
        return selected;
    };
    let known = device
        .known_failure(&request_key(r.snapshot, r.library, r.analysis, r.settings))
        .is_some();
    let d = if hinted.route == Route::Metal && !known {
        hinted
    } else {
        // The device is open now; its setup is sunk.
        model.decide(
            r.analysis,
            r.settings,
            r.system,
            Device::of(device.backend.info()),
            &state(true, known),
        )
    };
    if d.route == Route::Cpu {
        return cpu(d.note(), d.json, false);
    }
    let mut selected = selected;
    let mut json = d.json;
    json["before_device"] = json!(false);
    selected.cost_decision = Some(json);
    selected
}

/// A failure that recurs for the same input on the same device: model,
/// support and numerical outcomes of the deterministic kernels. Device,
/// driver and I/O errors can be transient and are retried.
fn deterministic(e: &EngineError) -> bool {
    match e {
        EngineError::Core(MkError::Backend(_) | MkError::Io(_) | MkError::Cancelled) => false,
        EngineError::Core(_) | EngineError::InvalidSettings(_) => true,
        _ => false,
    }
}

/// Identity of a native request on one device (the cache lives in the
/// device): the snapshot and glass library contents, the analysis and its
/// normalised settings. An edit changes the snapshot hash, so entries for
/// the previous design can no longer match.
pub(crate) fn request_key(
    snapshot: &[u8; 32],
    library: &str,
    analysis: &str,
    settings: &Value,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let snapshot = blake3::Hash::from(*snapshot).to_hex();
    let settings = settings.to_string();
    for part in [
        snapshot.as_bytes(),
        library.as_bytes(),
        analysis.as_bytes(),
        settings.as_bytes(),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

pub(crate) fn run(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let Some(device) = gpu::device(input) else {
        return analysis.run(input, settings, ctx);
    };
    let reference = AnalysisInput::with_resolver(
        input.snapshot.clone(),
        input.library.clone(),
        BackendInfo::cpu_reference(),
        Box::new(|| input.resolved()),
    )
    .with_stop_token(input.stop.clone())
    .with_library_hash(input.library_hash().into());
    let auto = crate::backend::Backend::is_auto_request(&input.requested_backend);
    let capability = reference_reason(analysis.name());
    let system = input.snapshot.system();
    let (cost, note) = if let Some(reason) = capability {
        (static_capability(reason), reason)
    } else if auto {
        let decided = input.cost_decision.borrow().clone();
        match decided {
            // Decided by `auto_route` (the cost model).
            Some(d) => (d, ""),
            // An input routed elsewhere: only the accuracy exclusions apply.
            None => match cost::accuracy_exclusion(analysis.name(), settings, system) {
                Some(evidence) => (
                    json!({"decision":"cpu_reference","rule":"accuracy exclusion","evidence":evidence,"version":cost::VERSION}),
                    "auto: the native path failed its f64 comparison on the measured evidence for this analysis",
                ),
                None => (
                    json!({"decision":"qualified_metal","reason":"no cost decision","version":cost::VERSION}),
                    "",
                ),
            },
        }
    } else {
        (
            json!({"decision":"explicit_backend","reason":"cost rules bypassed"}),
            "",
        )
    };
    let cpu = cost["decision"] == "cpu_reference";
    *input.cost_decision.borrow_mut() = Some(cost);
    let attempt = |outcome: &str, reason: String, elapsed_ms: f64| {
        input.record_attempt(BackendAttempt {
            backend: device.id.clone(),
            outcome: outcome.into(),
            reason,
            elapsed_ms,
        });
    };
    let fallback = |reason: String, deliberate| {
        input.report_backend(BackendUse {
            id: "cpu_reference".into(),
            precision: Precision::F64Reference,
            device: None,
            fallback_reason: Some(reason),
            deliberate,
        });
    };
    if cpu {
        if !auto {
            // An explicit request sent to the reference by policy.
            attempt("downgraded", note.into(), 0.);
        }
        fallback(note.into(), true);
        return analysis.run(&reference, settings, ctx);
    }
    let key = request_key(
        &input.snapshot.hash().0,
        input.library_hash(),
        analysis.name(),
        settings,
    );
    if let Some(known) = device.known_failure(&key) {
        let reason = format!(
            "resident {} skipped: the same input failed on this device after {:.1} ms ({})",
            analysis.name(),
            known.elapsed_ms,
            known.reason
        );
        attempt("skipped", reason.clone(), 0.);
        fallback(reason, false);
        return analysis.run(&reference, settings, ctx);
    }
    if let Err(why) = analysis.native_preflight(settings, system, device.backend.info()) {
        let reason = format!(
            "resident {} rejected before preparation: {why}",
            analysis.name()
        );
        attempt("infeasible", reason.clone(), 0.);
        fallback(reason, auto);
        return analysis.run(&reference, settings, ctx);
    }
    let start = Instant::now();
    match analysis.run(input, settings, ctx) {
        Ok(out) => Ok(out),
        Err(e) if e.is_cancelled() => Err(e),
        Err(e) => {
            ctx.check()?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
            if deterministic(&e) {
                device.remember_failure(
                    key,
                    KnownFailure {
                        reason: e.to_string(),
                        elapsed_ms,
                    },
                );
            }
            let reason = format!(
                "resident {} failed after {elapsed_ms:.1} ms: {e}",
                analysis.name()
            );
            attempt("failed", reason.clone(), elapsed_ms);
            fallback(reason, false);
            analysis.run(
                &reference,
                settings,
                &ctx.subrange(ctx.progress.fraction(), 1.),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use crate::backend::Backend;
    use crate::gpu::GpuProvider;
    use crate::project::Project;
    use crate::{AnalysisRequest, Engine, EngineConfig};
    use mklight_core::samples;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A GPU-capable analysis whose native path fails with `error`; the
    /// reference returns the snapshot's surface count.
    struct Flaky {
        native: Arc<AtomicUsize>,
        error: fn() -> MkError,
    }

    impl Analysis for Flaky {
        fn name(&self) -> &str {
            "ray_grid"
        }
        fn title(&self) -> &str {
            "flaky"
        }
        fn category(&self) -> &str {
            "diagnostic"
        }
        fn gpu_path(&self) -> bool {
            true
        }
        fn settings_schema(&self) -> Value {
            json!({"type":"object","properties":{"rings":{"type":"integer","default":2}}})
        }
        fn run(
            &self,
            input: &AnalysisInput<'_>,
            _: &Value,
            _: &JobContext,
        ) -> EngineResult<AnalysisOutput> {
            if gpu::device(input).is_some() {
                self.native.fetch_add(1, Ordering::SeqCst);
                return Err((self.error)().into());
            }
            Ok(AnalysisOutput {
                json: json!({"n": input.snapshot.system().surfaces.len()}),
                ..Default::default()
            })
        }
    }

    fn engine(opened: &Arc<AtomicUsize>) -> Engine {
        let e = Engine::new(EngineConfig {
            default_backend: Backend::CpuReference,
            ..Default::default()
        });
        // Only the counting provider: no default Metal device.
        *e.inner.backends.write().unwrap() = crate::backend::BackendRegistry::cpu_only();
        let opened = opened.clone();
        e.register_backend(Arc::new(GpuProvider::new(
            "probe",
            "counting emulation",
            move || {
                opened.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(mklight_gpu::CpuF32Backend::new())
                    as Arc<dyn mklight_gpu::ComputeBackend>)
            },
        )));
        e
    }

    #[test]
    fn deterministic_native_failures_are_not_attempted_again() {
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        let native = Arc::new(AtomicUsize::new(0));
        e.register_analysis(Arc::new(Flaky {
            native: native.clone(),
            error: || MkError::Unsupported("unresolved aims".into()),
        }));
        let mut s = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let mut req = AnalysisRequest::new("ray_grid", Value::Null);
        req.backend = Some(Backend::Gpu("probe".into()));
        let run = |s: &crate::Session| s.run_analysis(&req, &JobContext::new()).unwrap();
        let first = run(&s);
        assert_eq!(first.run_info.backend, "cpu_reference");
        let a = &first.run_info.failed_attempts;
        assert_eq!(a.len(), 1, "{a:?}");
        assert_eq!(
            (a[0].backend.as_str(), a[0].outcome.as_str()),
            ("gpu:probe", "failed")
        );
        assert!(a[0].reason.contains("unresolved aims") && a[0].elapsed_ms >= 0.);
        assert!(
            first
                .run_info
                .warnings
                .iter()
                .any(|w| w.contains("failed after"))
        );
        let again = run(&s);
        assert_eq!(
            native.load(Ordering::SeqCst),
            1,
            "re-attempted a known failure"
        );
        assert_eq!(again.run_info.failed_attempts[0].outcome, "skipped");
        assert_eq!(again.json, first.json);
        // Other settings and an edited design are new inputs.
        let mut other = req.clone();
        other.settings = json!({"rings": 3});
        s.run_analysis(&other, &JobContext::new()).unwrap();
        assert_eq!(native.load(Ordering::SeqCst), 2);
        let mut edited = s.system().clone();
        edited.surfaces[1].thickness *= 1.01;
        s.set_system(edited).unwrap();
        assert_eq!(run(&s).run_info.failed_attempts[0].outcome, "failed");
        assert_eq!(native.load(Ordering::SeqCst), 3);
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn transient_device_failures_are_retried() {
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        let native = Arc::new(AtomicUsize::new(0));
        e.register_analysis(Arc::new(Flaky {
            native: native.clone(),
            error: || MkError::Backend("command buffer: impacting interactivity".into()),
        }));
        let s = e
            .session_from_project(Project::new(samples::singlet()))
            .unwrap();
        let mut req = AnalysisRequest::new("ray_grid", Value::Null);
        req.backend = Some(Backend::Gpu("probe".into()));
        for n in 1..=2 {
            let r = s.run_analysis(&req, &JobContext::new()).unwrap();
            assert_eq!(r.run_info.failed_attempts[0].outcome, "failed");
            assert_eq!(native.load(Ordering::SeqCst), n);
        }
    }

    /// `auto` routes that are CPU by policy never open the device; the
    /// decision says so.
    #[test]
    fn auto_cpu_routes_are_decided_before_the_device_opens() {
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        let s = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let run = |name: &str, settings: Value| {
            let mut req = AnalysisRequest::new(name, settings);
            req.backend = Some(Backend::Auto);
            s.run_analysis(&req, &JobContext::new()).unwrap()
        };
        for (name, rule) in [
            ("ghost_irradiance", "accuracy exclusion"),
            ("wavefront_map", "static capability"),
        ] {
            let r = run(name, Value::Null);
            assert_eq!(r.run_info.backend, "cpu_reference", "{name}");
            let cost = r.backend.cost_decision.as_ref().unwrap();
            assert_eq!(
                (&cost["rule"], &cost["before_device"]),
                (&json!(rule), &json!(true))
            );
            assert!(r.backend.fallback_reason.is_some());
            assert!(r.run_info.warnings.is_empty() && r.run_info.failed_attempts.is_empty());
        }
        // No GPU path: known from the provider's catalogue.
        let r = run("seidel", Value::Null);
        assert!(
            r.backend
                .fallback_reason
                .unwrap()
                .contains("no GPU path for 'seidel'")
        );
        assert_eq!(opened.load(Ordering::SeqCst), 0);
        // The cost model needs the device family: this provider opens and
        // has no calibration, so it runs natively.
        let r = run("spot_diagram", Value::Null);
        assert_eq!(r.run_info.backend, "gpu:probe");
        assert_eq!(
            r.backend.cost_decision.unwrap()["reason"],
            "no cost calibration for this device"
        );
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    /// Without GPU providers every `auto` route is the reference, decided
    /// before any device; with the calibrated family the hint decides
    /// without opening anything.
    #[test]
    fn auto_routes_need_no_device_when_hints_suffice() {
        let reg = crate::backend::BackendRegistry::cpu_only();
        let model = CostModel::default();
        let system = samples::double_gauss();
        let analyses = AnalysisRegistry::with_builtin();
        for name in gpu::GPU_ANALYSES {
            let a = analyses.get(name).unwrap();
            let settings = super::super::prepare(a.as_ref(), &Value::Null).unwrap();
            let r = AutoRequest {
                analysis: name,
                settings: &settings,
                system: &system,
                snapshot: &[0; 32],
                library: "",
                tracer_warm: false,
            };
            let s = auto_route(&model, &reg, &r);
            assert!(s.device.is_none() && s.info.id == "cpu_reference", "{name}");
            let d = model.decide(name, &settings, &system, Device::Apple, &State::default());
            assert_ne!(d.route, Route::Undecided, "{name}");
        }
    }
}
