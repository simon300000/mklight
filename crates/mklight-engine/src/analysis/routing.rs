//! The cost model's `auto` route and native-attempt accounting; no runtime
//! f64 verification.
//!
//! * `auto_route` settles every `auto` request from provider hints, the
//!   division rule of the analysis (`reference_by_division`) and the cost
//!   model (`cost`), opening the device only when the predicted route is
//!   native or the device family cannot be known without it.
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
use crate::route::RouteOptions;
use mklight_core::job::{BackendAttempt, Precision};
use mklight_core::{JobContext, MkError, OpticalSystem};
use serde_json::{Value, json};

/// Why the native division of this request gives the device no part of the
/// run, when the request alone tells: such a run is the CPU reference, and
/// `auto` settles it without a prediction or a device.
///
/// `stray_light`: the transport's share of the source chunks
/// (`mklight_stray::gpu::hybrid::planned_share`) is zero below the device's
/// smallest worthwhile share and for sags it does not intersect in closed
/// form, or as `route.stray_share` says. The analysis reports the same for
/// a run that reaches the device and gets no share
/// (`gpu::record_reference_by_rule`).
fn reference_by_division(r: &AutoRequest<'_>) -> Option<&'static str> {
    match r.analysis {
        "stray_light" => {
            let rays = r.settings["rays"].as_u64()?;
            let share =
                mklight_stray::gpu::hybrid::planned_share(r.system, rays, r.route.stray_share);
            (share == 0.).then_some(if r.route.stray_share.is_some() {
                "stray_light: the route option stray_share gives the device no source of this run"
            } else {
                "stray_light: the transport gives the device no source of this run \
                 (too few sources, or surfaces it does not intersect in closed form)"
            })
        }
        _ => None,
    }
}

fn no_device_share(reason: &str) -> Value {
    json!({"decision":"cpu_reference","rule":"no device share","reason":reason,"version":cost::VERSION})
}

/// One `auto` request as the router sees it.
pub(crate) struct AutoRequest<'a> {
    pub analysis: &'a str,
    pub settings: &'a Value,
    pub system: &'a OpticalSystem,
    pub snapshot: &'a [u8; 32],
    pub library: &'a str,
    /// The run's route options.
    pub route: &'a RouteOptions,
    /// The graph holds the snapshot's reference tracer.
    pub tracer_warm: bool,
    /// The device's kernels were created ahead (`Engine::prepare_gpu`).
    pub kernels_prepared: bool,
}

/// The `auto` selection: the reference, decided before any device opens
/// when the hints suffice, or the device with the cost decision. Never
/// fails (an unavailable device is the reference with its reason). An
/// identical earlier request's route is taken again (`cost` module docs),
/// unless its device has since failed on it deterministically.
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
        kernels_prepared: r.kernels_prepared,
    };
    let (apple, open) = match registry.auto_candidate(r.analysis) {
        Some(AutoCandidate::NoGpu(note)) => return cpu(&note, Value::Null, true),
        Some(AutoCandidate::Gpu {
            apple_gpu,
            webgpu,
            open,
        }) => (Device::hinted(apple_gpu, webgpu), open),
        None => (Device::Unknown, false),
    };
    if let Some(reason) = reference_by_division(r) {
        return cpu(reason, no_device_share(reason), !open);
    }
    let route = request_key(r.snapshot, r.library, r.analysis, r.settings);
    let repeated = model.repeated(&route);
    let hinted = repeated.clone().unwrap_or_else(|| {
        model.decide(r.analysis, r.settings, r.system, apple, &state(open, false))
    });
    if hinted.route == Route::Cpu {
        if repeated.is_none() {
            model.settle(route, &hinted);
        }
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
    let known = device.known_failure(&route).is_some();
    let mut d = if hinted.route == Route::Metal && !known {
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
    model.note_device(&mut d, device.backend.info());
    if repeated.is_none() {
        model.settle(route, &d);
    }
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
    // The same run on the reference: policy, graph and stop carry over.
    let reference = input.derive(
        input.snapshot.clone(),
        BackendInfo::cpu_reference(),
        Box::new(|| input.resolved()),
    );
    let auto = crate::backend::Backend::is_auto_request(&input.requested_backend);
    let (cost, note) = if auto {
        let decided = input.cost_decision.borrow().clone();
        match decided {
            // Decided by `auto_route` (the cost model).
            Some(d) => (d, ""),
            // An input routed elsewhere: only the accuracy exclusions apply.
            None => match cost::accuracy_exclusion(analysis.name()) {
                Some(evidence) => (
                    json!({"decision":"cpu_reference","rule":"accuracy exclusion","evidence":evidence,"version":cost::VERSION}),
                    cost::EXCLUDED_NOTE,
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
            "native {} skipped: the same input failed on this device after {:.1} ms ({})",
            analysis.name(),
            known.elapsed_ms,
            known.reason
        );
        attempt("skipped", reason.clone(), 0.);
        fallback(reason, false);
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
                "native {} failed after {elapsed_ms:.1} ms: {e}",
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
    use crate::analysis::cost::Calibration;
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

    /// The reference run that completes a failed or skipped native attempt
    /// is the same run: it sees the request's policy and the engine's
    /// analysis graph.
    #[test]
    fn fallbacks_keep_the_run_policy_and_the_graph() {
        use mklight_core::job::{AimStopKind, Tier};
        struct Echo;
        impl Analysis for Echo {
            fn name(&self) -> &str {
                "ray_grid"
            }
            fn title(&self) -> &str {
                "echo"
            }
            fn category(&self) -> &str {
                "diagnostic"
            }
            fn run(
                &self,
                input: &AnalysisInput<'_>,
                _: &Value,
                _: &JobContext,
            ) -> EngineResult<AnalysisOutput> {
                if gpu::device(input).is_some() {
                    return Err(MkError::Unsupported("no native path".into()).into());
                }
                Ok(AnalysisOutput {
                    json: json!({"policy": input.policy, "graph": input.graph().is_some()}),
                    ..Default::default()
                })
            }
        }
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        assert!(e.analysis_graph().is_some());
        e.register_analysis(Arc::new(Echo));
        let s = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let mut req = AnalysisRequest::new("ray_grid", Value::Null);
        req.backend = Some(Backend::Gpu("probe".into()));
        req.tier = Tier::Preview;
        // Neither is the default policy (design tier, requested aim stop).
        let policy = crate::analysis::RunPolicy {
            tier: Tier::Preview,
            aim_stop: AimStopKind::Conditioned,
        };
        assert_eq!(e.config().aim_stop, policy.aim_stop);
        for outcome in ["failed", "skipped"] {
            let r = s.run_analysis(&req, &JobContext::new()).unwrap();
            assert_eq!(r.run_info.failed_attempts[0].outcome, outcome);
            assert_eq!(
                r.json,
                json!({"policy": policy, "graph": true}),
                "{outcome}"
            );
            assert_eq!(
                (r.run_info.tier, r.run_info.aim_stop),
                (policy.tier, policy.aim_stop)
            );
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
        let glass = json!({
            "material_variables": [{"surface": 1, "candidates": {"nearest": 2}}],
            "max_combinations": 2, "local_iterations": 1
        });
        for (name, settings, rule) in [
            ("glass_substitution", glass, "accuracy exclusion"),
            ("stray_light", json!({"rays": 2000}), "no device share"),
        ] {
            let r = run(name, settings);
            assert_eq!(r.run_info.backend, "cpu_reference", "{name}");
            let cost = r.backend.cost_decision.as_ref().unwrap();
            assert_eq!(
                (&cost["rule"], &cost["before_device"]),
                (&json!(rule), &json!(true))
            );
            assert!(r.backend.fallback_reason.is_some());
            // No native attempt (the analysis's own warnings may remain).
            assert!(r.run_info.failed_attempts.is_empty(), "{name}");
        }
        // No GPU path: known from the provider's catalogue.
        for name in [
            "seidel",
            "encircled_energy",
            "polarization_pupil_map",
            "wavefront_map",
        ] {
            let r = run(name, Value::Null);
            assert_eq!(r.run_info.backend, "cpu_reference", "{name}");
            assert!(
                r.backend
                    .fallback_reason
                    .unwrap()
                    .contains(&format!("no GPU path for '{name}'"))
            );
            assert!(r.backend.cost_decision.is_none(), "{name}");
            assert!(r.run_info.warnings.is_empty() && r.run_info.failed_attempts.is_empty());
        }
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

    /// A stray run whose division gives the device no source is the CPU
    /// reference by rule: decided from the request alone, before any device
    /// opens and without a prediction. A share that gives the device
    /// sources leaves the decision to the cost model.
    #[test]
    fn stray_runs_without_a_device_share_are_decided_by_rule() {
        let settings = |rays: u64| json!({"rays": rays});
        let verdict = |system: &OpticalSystem, rays: u64, route: &RouteOptions| {
            reference_by_division(&AutoRequest {
                analysis: "stray_light",
                settings: &settings(rays),
                system,
                snapshot: &[0; 32],
                library: "",
                route,
                tracer_warm: false,
                kernels_prepared: false,
            })
        };
        let default = RouteOptions::default();
        let share = |s: f64| RouteOptions {
            stray_share: Some(s),
            ..Default::default()
        };
        let triplet = samples::cooke_triplet();
        // A fifth of 163,840 sources is the smallest device share.
        assert!(verdict(&triplet, 20_000, &default).is_some());
        assert!(verdict(&triplet, 163_839, &default).is_some());
        assert!(verdict(&triplet, 163_840, &default).is_none());
        // Aspheres are intersected iteratively: no share at any size.
        let asphere = samples::petzval_asphere();
        assert!(verdict(&asphere, 1_000_000, &default).is_some());
        // The route option replaces the rule.
        assert!(verdict(&triplet, 2000, &share(1.)).is_none());
        assert!(verdict(&asphere, 2000, &share(0.5)).is_none());
        let none = verdict(&triplet, 1_000_000, &share(0.)).unwrap();
        assert!(none.contains("stray_share"), "{none}");
        // The verdict is the transport's own.
        for (system, rays, route) in [
            (&triplet, 20_000, &default),
            (&triplet, 163_840, &default),
            (&asphere, 1_000_000, &default),
        ] {
            let planned =
                mklight_stray::gpu::hybrid::planned_share(system, rays, route.stray_share);
            assert_eq!(verdict(system, rays, route).is_some(), planned == 0.);
        }
        // Through the engine: no device opens, and the decision names the
        // rule.
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        let s = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let mut req = AnalysisRequest::new("stray_light", settings(2000));
        req.backend = Some(Backend::Auto);
        let r = s.run_analysis(&req, &JobContext::new()).unwrap();
        assert_eq!(r.run_info.backend, "cpu_reference");
        let cost = r.backend.cost_decision.as_ref().unwrap();
        assert_eq!(
            (&cost["decision"], &cost["rule"], &cost["before_device"]),
            (
                &json!("cpu_reference"),
                &json!("no device share"),
                &json!(true)
            )
        );
        let reason = r.backend.fallback_reason.clone().unwrap();
        assert!(reason.contains("gives the device no source"), "{reason}");
        assert!(r.run_info.warnings.is_empty(), "{:?}", r.run_info.warnings);
        assert!(r.run_info.failed_attempts.is_empty());
        assert_eq!(opened.load(Ordering::SeqCst), 0);
        // With every source on the device the same request is routed (this
        // provider has no calibration, so natively).
        req.route = Some(share(1.));
        let r = s.run_analysis(&req, &JobContext::new()).unwrap();
        assert_eq!(r.run_info.backend, "gpu:probe", "{:?}", r.backend);
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    /// Kernels prepared ahead (`Engine::prepare_gpu`) leave no first-use
    /// pipeline creation to charge to a native route; without a finished
    /// preparation the first native use of an analysis pays it. (Real
    /// Metal: only an Apple GPU is predicted.)
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn prepared_kernels_are_not_charged_as_first_use() {
        use crate::gpu::GpuPreparation;
        let engine = || {
            Engine::new(EngineConfig {
                default_backend: Backend::Auto,
                ..Default::default()
            })
        };
        let decision = |e: &Engine| {
            let s = e
                .session_from_project(Project::new(samples::double_gauss()))
                .unwrap();
            let r = s
                .run_analysis(
                    &AnalysisRequest::new("ray_grid", json!({"rings": 8})),
                    &JobContext::new(),
                )
                .unwrap();
            r.backend.cost_decision.unwrap()
        };
        let e = engine();
        assert!(e.prepare_gpu(None));
        let start = Instant::now();
        while matches!(e.gpu_preparation(), GpuPreparation::Preparing { .. })
            && start.elapsed().as_secs() < 300
        {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        match e.gpu_preparation() {
            GpuPreparation::Ready { .. } => (),
            GpuPreparation::Unavailable { reason } => {
                return eprintln!("skipping: Metal unavailable ({reason})");
            }
            s => panic!("{s:?}"),
        }
        // Prepared: the device is open and its kernels exist.
        let prepared = decision(&e);
        assert_eq!(prepared["rule"], "cost model", "{prepared}");
        assert_eq!(prepared["state"]["kernels_prepared"], true);
        assert_eq!(prepared["state"]["device_open"], true);
        assert_eq!(prepared["stages_ms"]["metal_setup"], 0.);
        // Not prepared: the first native use is charged, with the device
        // open here (the other engine opened the process's Metal device)
        // or not.
        let table = Calibration::builtin();
        let first_use = table.analyses["ray_grid"].first_use_ms;
        assert!(first_use > 0.);
        let cold = decision(&engine());
        assert_eq!(cold["state"]["kernels_prepared"], false, "{cold}");
        let open = if cold["state"]["device_open"] == true {
            0.
        } else {
            table.device_open_ms
        };
        let setup = cold["stages_ms"]["metal_setup"].as_f64().unwrap();
        assert!((setup - (open + first_use)).abs() < 1e-3 * setup, "{setup}");
    }

    /// An explicit GPU request for an analysis without a GPU path falls back
    /// to the reference from the provider's catalogue, without device setup.
    #[test]
    fn explicit_requests_without_a_gpu_path_open_no_device() {
        let opened = Arc::new(AtomicUsize::new(0));
        let e = engine(&opened);
        let s = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        // The analyses the reference alone computes behave alike, whatever
        // their category.
        for name in [
            "seidel",
            "encircled_energy",
            "polarization_pupil_map",
            "wavefront_map",
        ] {
            let mut req = AnalysisRequest::new(name, Value::Null);
            req.backend = Some(Backend::Gpu("probe".into()));
            req.allow_backend_fallback = true;
            let r = s.run_analysis(&req, &JobContext::new()).unwrap();
            assert_eq!(r.run_info.backend, "cpu_reference");
            assert!(
                r.run_info
                    .warnings
                    .iter()
                    .any(|w| w.contains(&format!("it does not implement '{name}'")))
            );
            assert!(r.run_info.failed_attempts.is_empty(), "{name}");
            req.allow_backend_fallback = false;
            let e = s.run_analysis(&req, &JobContext::new()).unwrap_err();
            assert!(
                matches!(&e, EngineError::BackendUnavailable(m) if m.contains(name)),
                "{e}"
            );
        }
        assert_eq!(opened.load(Ordering::SeqCst), 0);
        // A GPU path opens the device.
        let mut req = AnalysisRequest::new("spot_diagram", Value::Null);
        req.backend = Some(Backend::Gpu("probe".into()));
        let r = s.run_analysis(&req, &JobContext::new()).unwrap();
        assert_eq!(r.run_info.backend, "gpu:probe");
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    /// Without GPU providers every `auto` route is the reference, decided
    /// before any device.
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
                route: &RouteOptions::default(),
                tracer_warm: false,
                kernels_prepared: false,
            };
            let s = auto_route(&model, &reg, &r);
            assert!(s.device.is_none() && s.info.id == "cpu_reference", "{name}");
        }
    }

    /// An identical request keeps its first route when refinement (or the
    /// load, or warm caches) would now predict the other backend: the
    /// backends agree only within the accuracy contracts, so a new route
    /// could change the same seeded run's result. Other settings decide
    /// afresh.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn identical_auto_requests_keep_their_first_route() {
        use crate::analysis::cost::Observation;
        let mut reg = crate::backend::BackendRegistry::cpu_only();
        // Named `metal`: the provider hints the calibrated Apple family.
        reg.register(Arc::new(GpuProvider::new("metal", "emulation", || {
            Ok(Arc::new(mklight_gpu::CpuF32Backend::new()) as Arc<dyn mklight_gpu::ComputeBackend>)
        })));
        let model = CostModel::default();
        let system = samples::singlet();
        let analyses = AnalysisRegistry::with_builtin();
        let a = analyses.get("ray_grid").unwrap();
        let settings = super::super::prepare(a.as_ref(), &json!({"rings": 64})).unwrap();
        let snapshot = [7u8; 32];
        let route = RouteOptions::default();
        let request = |settings| AutoRequest {
            analysis: "ray_grid",
            settings,
            system: &system,
            snapshot: &snapshot,
            library: "",
            route: &route,
            tracer_warm: false,
            kernels_prepared: false,
        };
        let first = auto_route(&model, &reg, &request(&settings));
        assert_eq!(first.info.id, "cpu_reference", "{:?}", first.cost_decision);
        let decision = first.cost_decision.clone().unwrap();
        assert_eq!(decision["rule"], "cost model");
        // The CPU keeps running far slower than predicted.
        let predicted = decision["predicted_ms"]["cpu"].as_f64().unwrap();
        let slow = Observation {
            used: "cpu_reference",
            failed: false,
            any_attempt: false,
            duration_ms: predicted * 100.,
            decision: Some(&decision),
            json: &Value::Null,
        };
        for _ in 0..20 {
            model.observe("ray_grid", &settings, &system, &snapshot, "", &slow);
        }
        let state = State {
            snapshot: Some(&snapshot),
            ..Default::default()
        };
        let fresh = model.decide("ray_grid", &settings, &system, Device::Apple, &state);
        assert_eq!(fresh.route, Route::Metal, "{}", fresh.json);
        let again = auto_route(&model, &reg, &request(&settings));
        assert_eq!(again.info.id, "cpu_reference");
        let replayed = again.cost_decision.unwrap();
        assert_eq!(replayed["repeated"], true);
        assert_eq!(replayed["predicted_ms"], decision["predicted_ms"]);
        // Other settings are a new request.
        let mut other = settings.clone();
        other["rings"] = json!(settings["rings"].as_u64().unwrap() + 1);
        let s = auto_route(&model, &reg, &request(&other));
        assert!(s.device.is_some() && s.cost_decision.unwrap().get("repeated").is_none());
        assert_eq!(model.stats()["repeated_routes"], 1);
    }
}
