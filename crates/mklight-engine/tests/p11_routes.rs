//! Backend routes through the engine: the advertised GPU analyses against
//! the registry, `auto` against its own cost decision, the reference
//! fallback after a device resource failure and batched candidate geometry
//! on the CPU-f32 emulation. The ignored tests need real Metal: every
//! sample's resident routes (`all_sample_resident_routes`) and the batched
//! candidate geometry on both kernel libraries.
use mklight_core::{JobContext, samples};
use mklight_engine::{AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project};
use serde_json::{Value, json};

#[test]
fn capabilities_match_registry_routes() {
    let engine = Engine::default();
    let advertised = mklight_engine::analysis::gpu::GPU_ANALYSES;
    for a in engine.analyses() {
        assert_eq!(
            a.gpu_path,
            advertised.contains(&a.name.as_str()),
            "{}",
            a.name
        );
    }
    for name in advertised {
        assert!(engine.analysis(name).is_ok(), "{name}");
    }
    assert!(advertised.contains(&"glass_substitution"));
    // Computed by the CPU reference alone.
    for name in [
        "encircled_energy",
        "polarization_pupil_map",
        "wavefront_map",
    ] {
        assert!(
            !advertised.contains(&name) && engine.analysis(name).is_ok(),
            "{name}"
        );
    }
}

/// Auto routes by the cost model, whose predictions depend on the machine
/// (its calibration, thread count and load): a small and a dense workload
/// each run on the backend their recorded decision names, without a
/// warning, and Metal is chosen only where the model predicts it faster. An
/// explicit backend bypasses the model.
#[test]
fn auto_follows_its_cost_decision() {
    let engine = Engine::default();
    if !engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:metal" && b.available)
    {
        eprintln!("skipping: Metal unavailable");
        return;
    }
    let run = |system, analysis: &str, settings: Value, backend| {
        let session = engine.session_from_project(Project::new(system)).unwrap();
        let mut request = AnalysisRequest::new(analysis, settings);
        request.backend = backend;
        let r = session.run_analysis(&request, &JobContext::new()).unwrap();
        println!(
            "P11_AUTO {} {}",
            r.backend.used,
            r.backend.cost_decision.as_ref().unwrap()
        );
        r
    };
    for (system, analysis, settings) in [
        (samples::singlet(), "spot_diagram", Value::Null),
        (
            samples::double_gauss(),
            "geometric_mtf",
            json!({"rings":160}),
        ),
    ] {
        let r = run(system, analysis, settings, None);
        let cost = r.backend.cost_decision.as_ref().unwrap();
        let reference = cost["decision"] == "cpu_reference";
        assert_eq!(r.backend.used == "cpu_reference", reference, "{cost}");
        assert_eq!(r.backend.used == "gpu:metal", !reference, "{cost}");
        assert_eq!(r.backend.fallback_reason.is_some(), reference, "{cost}");
        assert!(r.run_info.warnings.is_empty(), "{:?}", r.run_info.warnings);
        if !reference && cost["rule"] == "cost model" {
            let predicted = &cost["predicted_ms"];
            assert!(
                predicted["metal"].as_f64() < predicted["cpu"].as_f64(),
                "{cost}"
            );
        }
    }
    let explicit = run(
        samples::singlet(),
        "spot_diagram",
        Value::Null,
        Some(Backend::Gpu("metal".into())),
    );
    assert_eq!(
        explicit.backend.cost_decision.as_ref().unwrap()["decision"],
        "explicit_backend"
    );
    assert_eq!(explicit.backend.used, "gpu:metal");
}

/// Every GPU analysis except the solvers on every sample, explicitly on
/// Metal: the result validates against its schema and buffer declarations
/// and reports its backend consistently. `P11_SAMPLE` and `P11_ANALYSES`
/// (comma list) restrict the matrix; `P11_RUNTIME_SOURCE` runs the kernels
/// compiled from source at run time.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real Metal; complete-chain release matrix"]
fn all_sample_resident_routes() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    if std::env::var_os("P11_RUNTIME_SOURCE").is_some() {
        engine.register_backend(std::sync::Arc::new(mklight_engine::gpu::GpuProvider::new(
            "metal",
            "Metal runtime source",
            || {
                mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
                    library: mklight_metal::LibraryChoice::RuntimeSource,
                })
                .map_err(|e| e.to_string())
                .and_then(|b| {
                    b.map(|b| {
                        std::sync::Arc::new(b) as std::sync::Arc<dyn mklight_gpu::ComputeBackend>
                    })
                    .ok_or_else(|| "real Metal required".into())
                })
            },
        )));
    }
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:metal" && b.available),
        "real Metal required"
    );
    let names = std::env::var("P11_ANALYSES").ok();
    let sample = std::env::var("P11_SAMPLE").ok();
    for (name, system) in samples::all() {
        if sample.as_ref().is_some_and(|s| s != name) {
            continue;
        }
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for a in engine.analyses() {
            if !a.gpu_path
                || matches!(
                    a.name.as_str(),
                    "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
                )
            {
                continue;
            }
            if names
                .as_ref()
                .is_some_and(|n| !n.split(',').any(|n| n == a.name))
            {
                continue;
            }
            let mut request = AnalysisRequest::new(&a.name, Value::Null);
            request.backend = Some(Backend::Gpu("metal".into()));
            let ctx = JobContext::new();
            let result = session
                .run_analysis(&request, &ctx)
                .unwrap_or_else(|e| panic!("{name}/{}: {e}", a.name));
            assert!(
                !result
                    .backend
                    .fallback_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("panicked")),
                "{name}/{}: {:?}",
                a.name,
                result.backend.fallback_reason
            );
            mklight_engine::analysis::schema::validate(&a.result_schema, &result.json, true)
                .unwrap_or_else(|e| panic!("{name}/{}: {e}", a.name));
            let buffers: Vec<_> = result
                .buffers
                .iter()
                .map(|b| {
                    (
                        b.name.clone(),
                        serde_json::to_value(b.dtype())
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string(),
                        b.shape.len(),
                    )
                })
                .collect();
            mklight_engine::analysis::schema::check_buffers(&a.result_schema, &buffers).unwrap();
            assert_eq!(result.backend.used, result.run_info.backend);
            assert_eq!(result.backend.precision, result.run_info.precision);
            assert_eq!(ctx.progress.fraction(), 1.);
            println!(
                "P11_ROUTE {}",
                json!({"sample":name,"analysis":a.name,"used":result.backend.used,
                "reason":result.backend.fallback_reason,
                "execution":result.backend.execution,"duration_ms":result.run_info.duration_ms})
            );
        }
    }
}

#[test]
fn batched_candidate_geometry_preserves_individual_verdicts() {
    check_candidate_geometry(&mklight_gpu::CpuF32Backend::new());
}
fn check_candidate_geometry(backend: &dyn mklight_gpu::QualificationBackend) {
    use mklight_gpu::{
        Radiometry,
        xval::{XvalSettings, qualify, qualify_many},
    };
    let mut systems = vec![
        samples::singlet(),
        samples::cooke_triplet(),
        samples::double_gauss(),
    ];
    let mut changed = systems[1].clone();
    changed.surfaces[1].thickness += 0.25;
    systems.push(changed);
    systems.extend((0..65).map(|_| samples::singlet()));
    let tracers: Vec<_> = systems
        .into_iter()
        .map(|s| mklight_analysis::tracer_for(s).unwrap())
        .collect();
    let refs: Vec<_> = tracers.iter().collect();
    let settings = XvalSettings {
        radiometry: Radiometry::Off,
        pupil: mklight_trace::PupilSampling::Hexapolar { rings: 2 },
        pupil_extra: vec![],
        ..XvalSettings::qualification()
    };
    let batch = qualify_many(backend, &refs, &settings, &JobContext::new()).unwrap();
    assert_eq!(batch.len(), refs.len());
    for (tracer, got) in refs.iter().zip(&batch) {
        let want = qualify(backend, tracer, Some(&settings), &JobContext::new()).unwrap();
        assert_eq!(*got, want);
        assert!(got.passed, "{:?}", got.reason);
    }
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        qualify_many(backend, &refs, &settings, &cancelled),
        Err(mklight_core::MkError::Cancelled)
    ));
}

struct FailedResident {
    inner: mklight_gpu::CpuF32Backend,
    cancelled: bool,
}
impl mklight_gpu::ComputeBackend for FailedResident {
    fn info(&self) -> &mklight_gpu::DeviceInfo {
        self.inner.info()
    }
    fn precision(&self) -> &mklight_gpu::PrecisionDescriptor {
        self.inner.precision()
    }
    fn supports(&self, k: mklight_gpu::JobKind) -> bool {
        self.inner.supports(k)
    }
    fn upload_systems(
        &self,
        s: &[mklight_gpu::PackedSystem],
    ) -> mklight_core::MkResult<mklight_gpu::SystemHandle> {
        self.inner.upload_systems(s)
    }
    fn upload_systems_cached(
        &self,
        key: &mklight_gpu::resident::SystemCacheKey,
        systems: &[mklight_gpu::PackedSystem],
    ) -> mklight_core::MkResult<mklight_gpu::SystemHandle> {
        self.inner.upload_systems_cached(key, systems)
    }
    fn release_systems(&self, h: mklight_gpu::SystemHandle) -> mklight_core::MkResult<()> {
        self.inner.release_systems(h)
    }
    fn read_view(
        &self,
        v: mklight_gpu::resident::BufferView,
    ) -> mklight_core::MkResult<mklight_gpu::resident::ReadView> {
        self.inner.read_view(v)
    }
    fn submit_plan(
        &self,
        p: mklight_gpu::resident::ResidentPlan,
        c: &JobContext,
    ) -> mklight_core::MkResult<mklight_gpu::Pending<mklight_gpu::resident::PlanResult>> {
        self.inner.submit_plan(p, c)
    }
    fn resident_cache_stats(&self) -> mklight_gpu::resident::ResidentCacheStats {
        self.inner.resident_cache_stats()
    }
    fn fill_buffer(
        &self,
        _: mklight_gpu::resident::BufferDesc,
        _: &mut dyn FnMut(mklight_gpu::resident::BufferFill<'_>) -> mklight_core::MkResult<()>,
    ) -> mklight_core::MkResult<mklight_gpu::resident::BufferHandle> {
        Err(if self.cancelled {
            mklight_core::MkError::Cancelled
        } else {
            mklight_core::MkError::Backend("injected resource allocation failure".into())
        })
    }
}

#[test]
fn resource_fallback_reports_reference_and_cancellation_discards_output() {
    use mklight_gpu::ComputeBackend;
    use std::sync::Arc;
    for cancelled in [false, true] {
        let engine = Engine::new(EngineConfig {
            default_backend: Backend::CpuReference,
            ..Default::default()
        });
        let provider = mklight_engine::gpu::GpuProvider::new(
            "failed_resident",
            "injected failure",
            move || {
                Ok(Arc::new(FailedResident {
                    inner: mklight_gpu::CpuF32Backend::new(),
                    cancelled,
                }) as Arc<dyn ComputeBackend>)
            },
        );
        engine.register_backend(Arc::new(provider));
        let session = engine
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let mut request = AnalysisRequest::new("spot_diagram", json!({"rings":3}));
        request.backend = Some(Backend::Gpu("failed_resident".into()));
        let result = session.run_analysis(&request, &JobContext::new());
        if cancelled {
            assert!(result.unwrap_err().is_cancelled());
            continue;
        }
        let result = result.unwrap();
        assert_eq!(result.run_info.backend, "cpu_reference");
        assert!(
            result
                .backend
                .fallback_reason
                .unwrap()
                .contains("resource allocation failure")
        );
        assert!(!result.run_info.warnings.is_empty());
        request.backend = Some(Backend::CpuReference);
        let reference = session.run_analysis(&request, &JobContext::new()).unwrap();
        assert_eq!(result.json, reference.json);
        assert_eq!(result.buffers, reference.buffers);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real Metal; batched per-candidate geometry verdicts"]
fn host_batched_candidate_geometry() {
    use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions { library })
            .unwrap()
            .expect("real Metal required");
        check_candidate_geometry(&b);
    }
}
