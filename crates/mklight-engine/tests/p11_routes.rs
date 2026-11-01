//! Engine-chain qualification on real devices. These tests require Metal;
//! sandbox/no-device runs cannot count as release evidence.
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
}

#[test]
#[ignore = "requires real Metal; complete-chain release matrix"]
fn all_sample_resident_routes() {
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
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
            assert_eq!(result.backend.used, result.run_info.backend);
            assert_eq!(result.backend.precision, result.run_info.precision);
            assert!(result.backend.qualification.is_some(), "{name}/{}", a.name);
            assert_eq!(ctx.progress.fraction(), 1.);
            println!(
                "P11_ROUTE {}",
                json!({"sample":name,"analysis":a.name,"used":result.backend.used,
                "reason":result.backend.fallback_reason,"qualification":result.backend.qualification,
                "execution":result.backend.execution,"duration_ms":result.run_info.duration_ms})
            );
        }
    }
}

#[test]
fn batched_candidate_geometry_preserves_individual_verdicts() {
    use mklight_gpu::{
        CpuF32Backend, Radiometry,
        xval::{XvalSettings, qualify, qualify_many},
    };
    let backend = CpuF32Backend::new();
    let mut systems = vec![
        samples::singlet(),
        samples::cooke_triplet(),
        samples::double_gauss(),
    ];
    let mut changed = systems[1].clone();
    changed.surfaces[1].thickness += 0.25;
    systems.push(changed);
    let tracers: Vec<_> = systems
        .into_iter()
        .map(|s| mklight_analysis::tracer_for(s).unwrap())
        .collect();
    let refs: Vec<_> = tracers.iter().collect();
    let settings = XvalSettings {
        radiometry: Radiometry::Off,
        ..XvalSettings::qualification()
    };
    let batch = qualify_many(&backend, &refs, &settings, &JobContext::new()).unwrap();
    assert_eq!(batch.len(), refs.len());
    for (tracer, got) in refs.iter().zip(&batch) {
        let want = qualify(&backend, tracer, Some(&settings), &JobContext::new()).unwrap();
        assert_eq!(*got, want);
        assert!(got.passed, "{:?}", got.reason);
    }
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        qualify_many(&backend, &refs, &settings, &cancelled),
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
    fn release_systems(&self, h: mklight_gpu::SystemHandle) -> mklight_core::MkResult<()> {
        self.inner.release_systems(h)
    }
    fn submit(
        &self,
        j: mklight_gpu::Job,
        c: &JobContext,
    ) -> mklight_core::MkResult<mklight_gpu::Pending<mklight_gpu::JobResult>> {
        self.inner.submit(j, c)
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
