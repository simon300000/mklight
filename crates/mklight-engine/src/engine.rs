//! `Engine`: the process-wide facade. Owns the analysis registry, backend
//! registry and job scheduler, and creates `Session`s. Cheap to clone (an
//! `Arc`); sessions hold a clone, so the worker pool lives until the engine
//! and every session are dropped.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use mklight_core::OpticalSystem;

use crate::analysis::{Analysis, AnalysisInfo, AnalysisRegistry, AnalysisResult};
use crate::backend::{Backend, BackendInfo, BackendProvider, BackendRegistry, SelectedBackend};
use crate::error::EngineResult;
use crate::graph::{AnalysisGraph, GraphStats};
use crate::project::Project;
use crate::scheduler::{
    EventCallback, JobId, JobSpec, JobState, JobStatus, Scheduler, SchedulerConfig,
};
use crate::session::Session;

/// Engine configuration. Every field has a default, so `{}` is valid JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    /// Worker threads; 0 picks `min(available cores, 4)`.
    pub worker_threads: usize,
    /// Finished jobs kept for status and result retrieval.
    pub max_retained_jobs: usize,
    /// Resolved systems cached per session (LRU by snapshot hash).
    pub resolved_cache_capacity: usize,
    /// Byte budget of the analysis graph (`crate::graph`): shared
    /// tracers, traced pupils and results. 0 disables it, so every
    /// analysis builds all of its intermediates.
    pub analysis_graph_bytes: usize,
    /// Progress polling interval for the event callback.
    pub progress_interval_ms: u64,
    /// Stamped into project metadata on save.
    pub app_name: String,
    pub app_version: String,
    pub default_backend: Backend,
    /// Aim-stop policy of requests that name none (CONVENTIONS §7):
    /// `conditioned`, qualified under the per-quantity contract of GPU §9.4
    /// on CPU and Metal. An explicit aiming tolerance in the system is
    /// honoured exactly under either policy.
    pub aim_stop: mklight_core::AimStopKind,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let s = SchedulerConfig::default();
        EngineConfig {
            worker_threads: 0,
            max_retained_jobs: s.max_retained,
            resolved_cache_capacity: 8,
            analysis_graph_bytes: 256 << 20,
            progress_interval_ms: s.progress_interval.as_millis() as u64,
            app_name: String::new(),
            app_version: String::new(),
            default_backend: Backend::Auto,
            aim_stop: mklight_core::AimStopKind::Conditioned,
        }
    }
}

pub(crate) struct EngineInner {
    pub(crate) config: EngineConfig,
    pub(crate) registry: RwLock<AnalysisRegistry>,
    pub(crate) backends: RwLock<BackendRegistry>,
    pub(crate) default_backend: RwLock<Backend>,
    pub(crate) scheduler: Scheduler<AnalysisResult>,
    pub(crate) graph: Option<Arc<AnalysisGraph>>,
    /// `auto` routing (`analysis::cost`): calibration and online state.
    pub(crate) cost: Arc<crate::analysis::cost::CostModel>,
    /// `prepare_gpu`'s progress, shared with its thread.
    gpu_preparation: Arc<crate::gpu::GpuPreparationState>,
    next_session: AtomicU64,
}

impl Drop for EngineInner {
    /// A running GPU preparation stops at its next kernel.
    fn drop(&mut self) {
        self.gpu_preparation.cancel.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new(EngineConfig::default())
    }
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let mut sc = SchedulerConfig::default();
        if config.worker_threads > 0 {
            sc.workers = config.worker_threads;
        }
        sc.max_retained = config.max_retained_jobs.max(1);
        sc.progress_interval = Duration::from_millis(config.progress_interval_ms.max(1));
        Engine {
            inner: Arc::new(EngineInner {
                registry: RwLock::new(AnalysisRegistry::with_builtin()),
                backends: RwLock::new(BackendRegistry::with_defaults()),
                default_backend: RwLock::new(config.default_backend.clone()),
                scheduler: Scheduler::new(sc),
                graph: (config.analysis_graph_bytes > 0)
                    .then(|| Arc::new(AnalysisGraph::new(config.analysis_graph_bytes))),
                cost: Arc::new(crate::analysis::cost::CostModel::default()),
                gpu_preparation: Arc::default(),
                next_session: AtomicU64::new(1),
                config,
            }),
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    pub(crate) fn next_session_id(&self) -> u64 {
        self.inner.next_session.fetch_add(1, Ordering::Relaxed)
    }

    // --- sessions

    /// A session with a new untitled project (`OpticalSystem::new_empty`).
    pub fn new_session(&self) -> Session {
        Session::from_project(self, Project::new(OpticalSystem::new_empty()), None)
            .expect("the empty system is valid")
    }

    /// A session around an in-memory project.
    pub fn session_from_project(&self, project: Project) -> EngineResult<Session> {
        Session::from_project(self, project, None)
    }

    /// Open a `.mklight` file.
    pub fn open_session(&self, path: &Path) -> EngineResult<Session> {
        Session::open(self, path)
    }

    // --- analysis graph

    /// The shared analysis graph (`None` when disabled).
    pub fn analysis_graph(&self) -> Option<&Arc<AnalysisGraph>> {
        self.inner.graph.as_ref()
    }

    /// Activity and occupancy of the analysis graph per node kind.
    pub fn analysis_graph_stats(&self) -> Option<GraphStats> {
        self.inner.graph.as_ref().map(|g| g.stats())
    }

    /// Drop every stored graph node (statistics are kept).
    pub fn clear_analysis_graph(&self) {
        if let Some(g) = &self.inner.graph {
            g.clear();
        }
    }

    // --- analyses

    pub fn analyses(&self) -> Vec<AnalysisInfo> {
        read(&self.inner.registry).list()
    }

    pub fn analysis(&self, name: &str) -> EngineResult<Arc<dyn Analysis>> {
        read(&self.inner.registry).get(name)
    }

    /// Add or replace an analysis (how solver crates plug in).
    pub fn register_analysis(&self, a: Arc<dyn Analysis>) {
        if let Ok(mut r) = self.inner.registry.write() {
            r.register(a);
        }
    }

    // --- backends

    pub fn backends(&self) -> Vec<BackendInfo> {
        read(&self.inner.backends).list()
    }

    pub fn register_backend(&self, p: Arc<dyn BackendProvider>) {
        if let Ok(mut r) = self.inner.backends.write() {
            r.register(p);
        }
    }

    /// Make `b` the default for new jobs. Fails if it is unavailable.
    /// Create the GPU's compute kernels in the background, so the first GPU
    /// analysis does not wait for shader compilation (several seconds on a
    /// new install, update or system-cache eviction). Opens the first
    /// registered GPU. With `cache_dir` the compiled kernels are saved there
    /// and later launches load them. Progress: [`Self::gpu_preparation`].
    /// Returns `false` when a preparation already started.
    pub fn prepare_gpu(&self, cache_dir: Option<std::path::PathBuf>) -> bool {
        let state = self.inner.gpu_preparation.clone();
        {
            let mut s = state.state.lock().unwrap_or_else(|e| e.into_inner());
            if *s != crate::gpu::GpuPreparation::Idle {
                return false;
            }
            *s = crate::gpu::GpuPreparation::Preparing {
                device: String::new(),
                done: 0,
                total: 0,
            };
        }
        let provider = self.inner.backends.read().unwrap().first_gpu();
        let spawned = std::thread::Builder::new()
            .name("mklight-gpu-prepare".into())
            .spawn(move || {
                use crate::gpu::{GpuDevice, GpuPreparation};
                let Some(provider) = provider else {
                    return state.set(GpuPreparation::Unavailable {
                        reason: "no GPU backend is registered".into(),
                    });
                };
                // Opening the device runs its numerics probe.
                match provider
                    .device()
                    .and_then(|d| d.downcast::<GpuDevice>().ok())
                {
                    Some(device) => crate::gpu::prepare(device, cache_dir, &state),
                    None => state.set(GpuPreparation::Unavailable {
                        reason: provider
                            .info()
                            .reason
                            .unwrap_or_else(|| "the GPU is not available".into()),
                    }),
                }
            });
        if let Err(e) = spawned {
            self.inner
                .gpu_preparation
                .set(crate::gpu::GpuPreparation::Failed {
                    device: String::new(),
                    message: e.to_string(),
                });
        }
        true
    }

    /// Progress of [`Self::prepare_gpu`].
    pub fn gpu_preparation(&self) -> crate::gpu::GpuPreparation {
        self.inner.gpu_preparation.get()
    }

    pub fn set_default_backend(&self, b: Backend) -> EngineResult<()> {
        read(&self.inner.backends).select(&b, None, false)?;
        if let Ok(mut d) = self.inner.default_backend.write() {
            *d = b;
        }
        Ok(())
    }

    pub fn default_backend(&self) -> Backend {
        read(&self.inner.default_backend).clone()
    }

    /// Pick the backend for one run of `analysis`. An explicit request is
    /// honoured or fails (unless `allow_fallback`); `auto` picks a GPU with
    /// a path for the analysis, else the CPU reference; the engine default
    /// is a preference, so analyses it does not implement run on the CPU
    /// reference (with a `RunInfo` warning for an explicit GPU default).
    pub fn select_backend(
        &self,
        requested: Option<&Backend>,
        analysis: &str,
        allow_fallback: bool,
    ) -> EngineResult<SelectedBackend> {
        let reg = read(&self.inner.backends);
        match requested {
            Some(b) => reg.select(b, Some(analysis), allow_fallback),
            None => {
                let d = self.default_backend();
                let mut s = reg.select(&d, Some(analysis), true)?;
                s.requested = format!("default:{}", d.id());
                Ok(s)
            }
        }
    }

    /// `select_backend` for one run with normalised `settings` on the
    /// `snapshot`. An `auto` run (requested, or the engine default) is routed
    /// by the cost model (`analysis::cost`): routes that end on the CPU
    /// reference are decided before any GPU device is opened, and the
    /// decision travels in `SelectedBackend::cost_decision`.
    pub fn route_backend(
        &self,
        requested: Option<&Backend>,
        analysis: &str,
        settings: &serde_json::Value,
        snapshot: &mklight_core::Snapshot,
        library_hash: &str,
        allow_fallback: bool,
    ) -> EngineResult<SelectedBackend> {
        let Some(id) = self.auto_request_id(requested) else {
            return self.select_backend(requested, analysis, allow_fallback);
        };
        let hash = snapshot.hash().0;
        let tracer_warm = self
            .inner
            .graph
            .as_ref()
            .is_some_and(|g| g.contains(&crate::analysis::tracer_key(&hash, library_hash)));
        let mut s = crate::analysis::auto_route(
            &self.inner.cost,
            &read(&self.inner.backends),
            &crate::analysis::AutoRequest {
                analysis,
                settings,
                system: snapshot.system(),
                snapshot: &hash,
                library: library_hash,
                tracer_warm,
            },
        );
        s.requested = id;
        Ok(s)
    }

    /// The recorded `requested` id when `requested` (or the engine default)
    /// is `auto`.
    pub(crate) fn auto_request_id(&self, requested: Option<&Backend>) -> Option<String> {
        match requested {
            Some(b) => (*b == Backend::Auto).then(|| b.id()),
            None => (self.default_backend() == Backend::Auto)
                .then(|| format!("default:{}", Backend::Auto.id())),
        }
    }

    /// The `auto` cost model (calibration table and online state).
    pub fn cost_model(&self) -> &Arc<crate::analysis::cost::CostModel> {
        &self.inner.cost
    }

    /// The cost model's calibration identity and online state.
    pub fn cost_model_stats(&self) -> serde_json::Value {
        self.inner.cost.stats()
    }

    // --- jobs

    /// Queue a prepared job (see `Session::analysis_job`).
    pub fn submit_job(&self, spec: JobSpec<AnalysisResult>) -> EngineResult<JobId> {
        self.inner.scheduler.submit(spec)
    }

    pub fn job_status(&self, id: JobId) -> EngineResult<JobStatus> {
        self.inner.scheduler.status(id)
    }

    pub fn job_statuses(&self) -> Vec<JobStatus> {
        self.inner.scheduler.statuses()
    }

    /// Stop an optimisation/tolerance job, retaining its partial result.
    pub fn stop_job(&self, id: JobId) -> EngineResult<()> {
        self.inner.scheduler.stop(id)
    }

    pub fn cancel_job(&self, id: JobId) -> EngineResult<()> {
        self.inner.scheduler.cancel(id)
    }

    pub fn cancel_all_jobs(&self) {
        self.inner.scheduler.cancel_all()
    }

    /// Wait for a job to finish (`None` = forever); `Timeout` otherwise.
    pub fn wait_job(&self, id: JobId, timeout: Option<Duration>) -> EngineResult<JobState> {
        self.inner.scheduler.wait(id, timeout)
    }

    /// Take a finished job's result and forget the job.
    pub fn take_result(&self, id: JobId) -> EngineResult<AnalysisResult> {
        self.inner.scheduler.take_result(id)
    }

    /// Cancel if needed and forget a job without taking its result.
    pub fn release_job(&self, id: JobId) -> EngineResult<()> {
        self.inner.scheduler.release(id)
    }

    /// Install or clear the job event callback. It runs on the engine's
    /// event thread, one event at a time, with no engine lock held (see
    /// `Scheduler::set_callback`).
    pub fn set_event_callback(&self, cb: Option<EventCallback>) {
        self.inner.scheduler.set_callback(cb)
    }

    /// Wait until every job event posted so far has reached the callback
    /// (see `Scheduler::flush_events`).
    pub fn flush_events(&self, timeout: Duration) -> bool {
        self.inner.scheduler.flush_events(timeout)
    }
}

/// Poison-tolerant read lock (registries are replaced wholesale per insert).
fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}
