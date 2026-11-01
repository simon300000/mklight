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
#[ignore = "requires calibrated release evidence and actual Metal"]
fn recorded_auto_cost_explicit_override_and_unknown_input() {
    let data: Value =
        serde_json::from_str(include_str!("../src/analysis/cost-evidence.json")).unwrap();
    assert_eq!(
        data["source_build"],
        mklight_engine::analysis::gpu::cost_build()
    );
    let row = data["observations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["rings"] == 0 && row["decision"] == "cpu_reference")
        .expect("a measured default CPU cost exception");
    let system = samples::all()
        .into_iter()
        .find(|(name, _)| *name == row["sample"].as_str().unwrap())
        .unwrap()
        .1;
    let engine = Engine::default();
    let session = engine
        .session_from_project(Project::new(system.clone()))
        .unwrap();
    let mut request = AnalysisRequest::new(row["analysis"].as_str().unwrap(), Value::Null);
    let automatic = session.run_analysis(&request, &JobContext::new()).unwrap();
    assert_eq!(automatic.backend.used, "cpu_reference");
    assert_eq!(
        automatic.backend.cost_decision.as_ref().unwrap()["decision"],
        "cpu_reference"
    );
    assert!(automatic.backend.qualification.is_none());
    request.backend = Some(Backend::Gpu("metal".into()));
    let explicit = session.run_analysis(&request, &JobContext::new()).unwrap();
    assert_eq!(explicit.backend.used, "gpu:metal");
    assert_eq!(
        explicit.backend.cost_decision.as_ref().unwrap()["decision"],
        "explicit_backend"
    );
    let mut changed = system;
    changed.surfaces[1].thickness += 1e-9;
    let session = engine.session_from_project(Project::new(changed)).unwrap();
    request.backend = None;
    let unknown = session.run_analysis(&request, &JobContext::new()).unwrap();
    assert_eq!(
        unknown.backend.cost_decision.as_ref().unwrap()["decision"],
        "qualified_metal"
    );
    assert_eq!(unknown.backend.used, "gpu:metal");
}

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
                    ..Default::default()
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
            assert!(result.backend.qualification.is_some(), "{name}/{}", a.name);
            if result.backend.used == "cpu_reference" {
                assert_eq!(
                    result.backend.qualification.as_ref().unwrap()["passed"],
                    false,
                    "a reference-only result cannot qualify a resident chain"
                );
            }
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
    check_candidate_geometry(&mklight_gpu::CpuF32Backend::new());
}
fn check_candidate_geometry(backend: &dyn mklight_gpu::ComputeBackend) {
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

#[test]
#[ignore = "consolidated quiet-host release benchmark; real Metal required"]
fn consolidated_engine_benchmark() {
    use std::time::Instant;
    let filter = std::env::var("P11_BENCH_ANALYSES").ok();
    let sample = std::env::var("P11_BENCH_SAMPLE").ok();
    let dense = std::env::var_os("P11_BENCH_DENSE").is_some();
    let candidates = std::env::var_os("P11_BENCH_CANDIDATES").is_some();
    for (name, system) in samples::all() {
        if sample.as_ref().is_some_and(|s| s != name) {
            continue;
        }
        if dense
            && ![
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
                "anamorphic",
            ]
            .contains(&name)
        {
            continue;
        }
        let engine = Engine::new(EngineConfig {
            default_backend: Backend::CpuReference,
            ..Default::default()
        });
        use mklight_engine::backend::BackendProvider;
        let opened = Instant::now();
        let provider = mklight_engine::gpu::GpuProvider::metal();
        let device = provider.device().expect("real Metal required");
        let device = device.downcast::<mklight_engine::gpu::GpuDevice>().unwrap();
        engine.register_backend(std::sync::Arc::new(provider));
        println!(
            "P11_ENGINE_ENV {}",
            json!({"device":device.backend.info(),"precision":device.backend.precision(),"source_build":mklight_engine::analysis::gpu::cost_build(),"rayon_threads":rayon::current_num_threads(),"setup_seconds":opened.elapsed().as_secs_f64(),"initial_copied_bytes":device.backend.resident_cache_stats().copied_bytes})
        );
        let session = engine.session_from_project(Project::new(system)).unwrap();
        for info in engine.analyses() {
            if !info.gpu_path
                || candidates
                    != matches!(
                        info.name.as_str(),
                        "optimize" | "tolerance" | "glass_substitution"
                    )
                || (info.name == "glass_substitution" && name != "cooke_triplet")
            {
                continue;
            }
            if filter
                .as_ref()
                .is_some_and(|v| !v.split(',').any(|v| v == info.name))
            {
                continue;
            }
            if dense
                && ![
                    "spot_diagram",
                    "ray_grid",
                    "geometric_mtf",
                    "best_focus",
                    "encircled_energy",
                ]
                .contains(&info.name.as_str())
            {
                continue;
            }
            let cases = if dense { vec![5, 56, 512] } else { vec![0] };
            for rings in cases {
                let settings = if candidates && info.name == "glass_substitution" {
                    json!({"material_variables":[
                        {"surface":1,"candidates":{"nearest":3}},
                        {"surface":3,"candidates":{"nearest":3}},
                        {"surface":5,"candidates":{"nearest":3}}],
                        "max_combinations":12,"local_iterations":8,
                        "default_merit":{"fields":[0,2],"wavelengths":[1]},
                        "backend":"auto","seed":17})
                } else if dense {
                    json!({"pattern":"square","rings":rings,"fields":{"type":"index","index":2},"wavelengths":{"type":"primary"}})
                } else {
                    Value::Null
                };
                // One cold pair, one warm-up pair and ten alternating measured pairs.
                for run in 0..12 {
                    let execute = |backend| {
                        let mut request = AnalysisRequest::new(&info.name, settings.clone());
                        request.backend = Some(backend);
                        let copied = device.backend.resident_cache_stats().copied_bytes;
                        let start = Instant::now();
                        let result = session.run_analysis(&request, &JobContext::new()).unwrap();
                        (
                            start.elapsed().as_secs_f64(),
                            result,
                            device.backend.resident_cache_stats().copied_bytes - copied,
                        )
                    };
                    let ((cpu_seconds, cpu, _), (metal_seconds, metal, copied_bytes)) =
                        if run % 2 == 0 {
                            let cpu = execute(Backend::CpuReference);
                            let gpu = execute(Backend::Gpu("metal".into()));
                            (cpu, gpu)
                        } else {
                            let gpu = execute(Backend::Gpu("metal".into()));
                            let cpu = execute(Backend::CpuReference);
                            (cpu, gpu)
                        };
                    let mut effective = cpu.run_info.settings.clone();
                    effective.as_object_mut().unwrap().remove("spectrum");
                    let key = mklight_engine::analysis::gpu::cost_key(
                        &info.name,
                        &cpu.run_info.snapshot_hash,
                        &cpu.library_hash,
                        &effective,
                    );
                    println!(
                        "P11_ENGINE_BENCH {}",
                        json!({"sample":name,"analysis":info.name,"rings":rings,"run":run,
                        "phase":if run==0{"cold"}else if run==1{"warmup"}else{"measured"},"key":key,
                        "snapshot":cpu.run_info.snapshot_hash,"library":cpu.library_hash,"settings":effective,
                        "cpu_seconds":cpu_seconds,"metal_seconds":metal_seconds,"copied_bytes":copied_bytes,"used":metal.backend.used,
                        "qualification":metal.backend.qualification,"reason":metal.backend.fallback_reason,
                        "execution":metal.backend.execution})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires real Metal; batched per-candidate geometry verdicts"]
fn host_batched_candidate_geometry() {
    use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("real Metal required");
        check_candidate_geometry(&b);
    }
}

#[test]
#[ignore = "quiet-host independent engine/provider/cache cold starts"]
fn isolated_engine_cold_starts() {
    use mklight_engine::backend::BackendProvider;
    use std::time::Instant;
    let catalogue = Engine::default().analyses();
    for (name, system) in samples::all() {
        for info in &catalogue {
            if !info.gpu_path
                || matches!(
                    info.name.as_str(),
                    "optimize" | "tolerance" | "glass_substitution"
                )
            {
                continue;
            }
            let start = Instant::now();
            let cpu_engine = Engine::new(EngineConfig {
                default_backend: Backend::CpuReference,
                ..Default::default()
            });
            let cpu_session = cpu_engine
                .session_from_project(Project::new(system.clone()))
                .unwrap();
            let mut request = AnalysisRequest::new(&info.name, Value::Null);
            request.backend = Some(Backend::CpuReference);
            let cpu = cpu_session
                .run_analysis(&request, &JobContext::new())
                .unwrap();
            let cpu_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let engine = Engine::new(EngineConfig {
                default_backend: Backend::CpuReference,
                ..Default::default()
            });
            let opened = Instant::now();
            let provider = mklight_engine::gpu::GpuProvider::metal();
            let device = provider.device().expect("actual Metal required");
            let device = device.downcast::<mklight_engine::gpu::GpuDevice>().unwrap();
            let setup_seconds = opened.elapsed().as_secs_f64();
            engine.register_backend(std::sync::Arc::new(provider));
            let session = engine
                .session_from_project(Project::new(system.clone()))
                .unwrap();
            request.backend = Some(Backend::Gpu("metal".into()));
            let metal = session.run_analysis(&request, &JobContext::new()).unwrap();
            let metal_seconds = start.elapsed().as_secs_f64();
            let mut effective = cpu.run_info.settings.clone();
            effective.as_object_mut().unwrap().remove("spectrum");
            println!(
                "P11_COLD {}",
                json!({"sample":name,"analysis":info.name,
                "key":mklight_engine::analysis::gpu::cost_key(&info.name,&cpu.run_info.snapshot_hash,&cpu.library_hash,&effective),
                "cpu_seconds":cpu_seconds,"metal_seconds":metal_seconds,"metal_backend_setup_seconds":setup_seconds,
                "source_build":mklight_engine::analysis::gpu::cost_build(),"used":metal.backend.used,
                "qualification":metal.backend.qualification,"execution":metal.backend.execution,
                "copied_bytes":device.backend.resident_cache_stats().copied_bytes,
                "scope":"independent engines/sessions; CPU does not warm the Metal engine reference caches"})
            );
        }
    }
}

#[test]
#[ignore = "quiet-host explicit resident QR compression calibration"]
fn qr_crossover_probe() {
    use mklight_optim::{
        gpu::GpuContext,
        linalg::{Matrix, Qr},
    };
    use std::{sync::Arc, time::Instant};
    let backend = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("real Metal required"),
    );
    let g = GpuContext::new(
        backend.clone(),
        mklight_core::Snapshot::new(samples::singlet()).unwrap(),
    );
    for (m, n) in [
        (400, 100),
        (1000, 100),
        (1600, 100),
        (2000, 100),
        (4000, 100),
        (2000, 60),
        (4225, 37),
        (32, 8),
        (128, 32),
        (4000, 32),
        (65536, 37),
    ] {
        let extra = matches!((m, n), (32, 8) | (128, 32) | (4000, 32) | (65536, 37));
        if extra != std::env::var_os("P11_QR_EXTRA_ONLY").is_some() {
            continue;
        }
        let a: Vec<_> = (0..m * n)
            .map(|i| ((i * 37 % 1009) as f64 / 1009. - 0.5) + if i / n == i % n { 2. } else { 0. })
            .collect();
        let wanted: Vec<_> = (0..n).map(|i| (i + 1) as f64 / 100.).collect();
        let rhs: Vec<_> = (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * wanted[j]).sum::<f64>())
            .collect();
        for run in 0..12 {
            let cpu = || {
                let t = Instant::now();
                let x = Qr::new(Matrix::from_row_major(m, n, a.clone()))
                    .solve_ls(&rhs)
                    .unwrap();
                (t.elapsed().as_secs_f64(), x)
            };
            let metal = || {
                let before = g.stats();
                let t = Instant::now();
                let (_q, factor, qtr) = g
                    .reduce_qr(
                        &Matrix::from_row_major(m, n, a.clone()),
                        &rhs,
                        &JobContext::new(),
                    )
                    .unwrap()
                    .unwrap();
                let x = Qr::new(factor).solve_ls(&qtr).unwrap();
                let wall = t.elapsed().as_secs_f64();
                let after = g.stats();
                let stats = json!({"command_buffers":after.command_buffers-before.command_buffers,
                    "device_seconds":after.device_seconds-before.device_seconds,
                    "copied_bytes":after.copied_bytes-before.copied_bytes,
                    "allocations":after.allocations-before.allocations,
                    "reused_allocations":after.reused_allocations-before.reused_allocations,
                    "peak_bytes":after.peak_bytes,"fallback_count":after.fallback_count-before.fallback_count,
                    "stages_cumulative":after.stages});
                (wall, x, stats)
            };
            let ((cpu_seconds, c), (metal_seconds, x, stats)) = if run % 2 == 0 {
                let c = cpu();
                (c, metal())
            } else {
                let m = metal();
                (cpu(), m)
            };
            let error = c
                .iter()
                .zip(&x)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(error < 1e-8, "{m}x{n}: {error}");
            println!(
                "P11_QR {}",
                json!({"rows":m,"columns":n,"work":m*n*n,"run":run,"phase":if run==0{"cold"}else if run==1{"warmup"}else{"measured"},"cpu_seconds":cpu_seconds,"metal_seconds":metal_seconds,"stats":stats,"maximum_solution_error":error,"forced_resident":true,"operation":"trust_region_qr_compression"})
            );
        }
    }
}
