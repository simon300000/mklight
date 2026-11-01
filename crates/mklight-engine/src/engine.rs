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
use crate::project::Project;
use crate::scheduler::{
    EventCallback, JobId, JobSpec, JobState, JobStatus, Scheduler, SchedulerConfig,
};
use crate::session::Session;

/// Result of `Engine::migrate_file`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MigrationOutcome {
    pub path: String,
    pub from_version: u32,
    pub to_version: u32,
    pub notes: Vec<String>,
    /// The file was rewritten at the current version.
    pub migrated: bool,
    /// Where the original was kept.
    pub backup: Option<String>,
    pub dry_run: bool,
}

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
            default_backend: Backend::Auto,
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

    /// Migrate a project file to the current schema version in place.
    /// Unless `backup` is false, the original bytes are kept as
    /// `project::backup_path`; with `dry_run` nothing is written. A file
    /// already at the current version is left untouched.
    pub fn migrate_file(
        &self,
        path: &Path,
        backup: bool,
        dry_run: bool,
    ) -> EngineResult<MigrationOutcome> {
        let bytes = std::fs::read(path).map_err(|e| {
            crate::EngineError::Core(mklight_core::MkError::Io(format!(
                "cannot read {}: {e}",
                path.display()
            )))
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            crate::EngineError::Core(mklight_core::MkError::Parse(format!(
                "{}: not UTF-8 text: {e}",
                path.display()
            )))
        })?;
        let (_, report) = Project::from_json_str(text)?;
        let mut out = MigrationOutcome {
            path: path.display().to_string(),
            from_version: report.from_version,
            to_version: report.to_version,
            notes: report.notes.clone(),
            migrated: false,
            backup: None,
            dry_run,
        };
        if !report.migrated() || dry_run {
            return Ok(out);
        }
        // Open through a session (catalogs resolve relative to the file)
        // before writing anything, so a file that cannot be opened is left
        // untouched.
        let mut session = self.open_session(path)?;
        if backup {
            let b = crate::project::backup_path(path, report.from_version);
            crate::fsutil::atomic_write(&b, &bytes).map_err(|e| {
                crate::EngineError::Core(mklight_core::MkError::Io(format!(
                    "cannot write backup {}: {e}",
                    b.display()
                )))
            })?;
            out.backup = Some(b.display().to_string());
        }
        session.save(None)?;
        out.migrated = true;
        Ok(out)
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
