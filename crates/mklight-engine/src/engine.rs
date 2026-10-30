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
use crate::backend::{Backend, BackendInfo, BackendProvider, BackendRegistry};
use crate::error::EngineResult;
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
    /// Progress polling interval for the event callback.
    pub progress_interval_ms: u64,
    /// Stamped into project metadata on save.
    pub app_name: String,
    pub app_version: String,
    pub default_backend: Backend,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let s = SchedulerConfig::default();
        EngineConfig {
            worker_threads: 0,
            max_retained_jobs: s.max_retained,
            resolved_cache_capacity: 8,
            progress_interval_ms: s.progress_interval.as_millis() as u64,
            app_name: String::new(),
            app_version: String::new(),
            default_backend: Backend::CpuReference,
        }
    }
}

pub(crate) struct EngineInner {
    pub(crate) config: EngineConfig,
    pub(crate) registry: RwLock<AnalysisRegistry>,
    pub(crate) backends: RwLock<BackendRegistry>,
    pub(crate) default_backend: RwLock<Backend>,
    pub(crate) scheduler: Scheduler<AnalysisResult>,
    next_session: AtomicU64,
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

    /// Open a `.mklight` file (migrating older schema versions).
    pub fn open_session(&self, path: &Path) -> EngineResult<Session> {
        Session::open(self, path)
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
    pub fn set_default_backend(&self, b: Backend) -> EngineResult<()> {
        read(&self.inner.backends).select(&b, false)?;
        if let Ok(mut d) = self.inner.default_backend.write() {
            *d = b;
        }
        Ok(())
    }

    pub fn default_backend(&self) -> Backend {
        read(&self.inner.default_backend).clone()
    }

    pub(crate) fn select_backend(
        &self,
        requested: Option<&Backend>,
        allow_fallback: bool,
    ) -> EngineResult<(BackendInfo, Option<String>)> {
        let b = requested.cloned().unwrap_or_else(|| self.default_backend());
        read(&self.inner.backends).select(&b, allow_fallback)
    }

    // --- jobs

    pub(crate) fn submit(&self, spec: JobSpec<AnalysisResult>) -> EngineResult<JobId> {
        self.inner.scheduler.submit(spec)
    }

    pub fn job_status(&self, id: JobId) -> EngineResult<JobStatus> {
        self.inner.scheduler.status(id)
    }

    pub fn job_statuses(&self) -> Vec<JobStatus> {
        self.inner.scheduler.statuses()
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

    /// Install or clear the job event callback (see `Scheduler::set_callback`
    /// for the threading rules).
    pub fn set_event_callback(&self, cb: Option<EventCallback>) {
        self.inner.scheduler.set_callback(cb)
    }
}

/// Poison-tolerant read lock (registries are replaced wholesale per insert).
fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}
