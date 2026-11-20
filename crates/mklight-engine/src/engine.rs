//! `Engine`: the process-wide facade. Owns the analysis registry, backend
//! registry and job scheduler, and creates `Session`s. Cheap to clone (an
//! `Arc`); sessions hold a clone, so the worker pool lives until the engine
//! and every session are dropped.

use serde::{Deserialize, Serialize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use mklight_core::OpticalSystem;

use crate::analysis::cost::measured::{self, DeviceCalibration};
use crate::analysis::{Analysis, AnalysisInfo, AnalysisRegistry, AnalysisResult};
use crate::backend::{Backend, BackendInfo, BackendProvider, BackendRegistry, SelectedBackend};
use crate::error::EngineResult;
use crate::graph::{AnalysisGraph, GraphStats};
use crate::project::Project;
use crate::scheduler::{
    EventCallback, JobId, JobSpec, JobState, JobStatus, Scheduler, SchedulerConfig, panic_message,
};
use crate::session::Session;

/// Engine configuration. Every field has a default, so `{}` is valid JSON;
/// a key that is not a field is an error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    /// Route options of requests that carry none (`crate::route`). They
    /// are checked with each request.
    pub route: crate::route::RouteOptions,
    /// Register the CPU-f32 emulation of the GPU kernels (`gpu:cpu_f32`)
    /// after the default providers, so a front end can exercise its GPU
    /// paths without a GPU. `auto` never selects it; only an explicit
    /// `{"kind": "gpu", "name": "cpu_f32"}` request runs on it.
    pub register_emulation: bool,
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
            route: Default::default(),
            register_emulation: false,
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
    /// The measured backends' calibration states by backend name
    /// (`Engine::start_calibration`).
    calibrations: std::sync::Mutex<std::collections::BTreeMap<&'static str, GpuCalibrationState>>,
    /// Where the CUDA calibration is cached (`<cache_dir>/cost/cuda.json`),
    /// once `prepare_gpu` named a cache directory.
    cuda_calibration_cache: std::sync::Mutex<Option<std::path::PathBuf>>,
    /// Requests routed outside the calibration threads (`route_backend`),
    /// so a calibration knows whether foreground work ran meanwhile.
    foreground_requests: AtomicU64,
    /// The CUDA calibration was started lazily (once per engine:
    /// `Engine::calibrate_cuda_lazily`).
    lazy_cuda_calibration: std::sync::atomic::AtomicBool,
    next_session: AtomicU64,
}

thread_local! {
    /// The current thread runs a cost calibration (`calibrate`).
    static CALIBRATING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Progress of a measured backend's calibration
/// ([`Engine::start_calibration`]).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GpuCalibrationState {
    #[default]
    Idle,
    Running,
    /// Measured (or loaded from the cache) and installed in the cost model.
    Ready {
        calibration: DeviceCalibration,
    },
    Failed {
        message: String,
    },
}

/// The WebGPU calibration's state, by its earlier name.
pub type WebGpuCalibrationState = GpuCalibrationState;

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
        #[cfg(feature = "windows-worker-priority")]
        crate::priority::raise_workers();
        let mut sc = SchedulerConfig::default();
        if config.worker_threads > 0 {
            sc.workers = config.worker_threads;
        }
        sc.max_retained = config.max_retained_jobs.max(1);
        sc.progress_interval = Duration::from_millis(config.progress_interval_ms.max(1));
        let mut backends = BackendRegistry::with_defaults();
        if config.register_emulation {
            backends.register(Arc::new(crate::gpu::GpuProvider::emulation()));
        }
        Engine {
            inner: Arc::new(EngineInner {
                registry: RwLock::new(AnalysisRegistry::with_builtin()),
                backends: RwLock::new(backends),
                default_backend: RwLock::new(config.default_backend.clone()),
                scheduler: Scheduler::new(sc),
                graph: (config.analysis_graph_bytes > 0)
                    .then(|| Arc::new(AnalysisGraph::new(config.analysis_graph_bytes))),
                cost: Arc::new(crate::analysis::cost::CostModel::default()),
                gpu_preparation: Arc::default(),
                calibrations: Default::default(),
                cuda_calibration_cache: Default::default(),
                foreground_requests: AtomicU64::new(0),
                lazy_cuda_calibration: Default::default(),
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
        write(&self.inner.registry).register(a);
    }

    // --- backends

    pub fn backends(&self) -> Vec<BackendInfo> {
        read(&self.inner.backends).list()
    }

    pub fn register_backend(&self, p: Arc<dyn BackendProvider>) {
        write(&self.inner.backends).register(p);
    }

    /// A backend by name (`BackendRegistry::parse`): `cpu`, `cpu_reference`,
    /// `auto`, `gpu:<name>` or the bare name of a registered GPU backend.
    pub fn parse_backend(&self, name: &str) -> EngineResult<Backend> {
        read(&self.inner.backends).parse(name)
    }

    /// Measure the WebGPU backend against the CPU reference on this machine
    /// ([`Self::start_calibration`] of `measured::WEBGPU`).
    pub fn start_webgpu_calibration(&self) -> bool {
        self.start_calibration(&measured::WEBGPU)
    }

    /// The state of the WebGPU calibration.
    pub fn webgpu_calibration_state(&self) -> GpuCalibrationState {
        self.calibration_state(&measured::WEBGPU)
    }

    /// Install a calibration measured earlier (the web host's cache), or a
    /// forced one; `None` returns WebGPU to the prior.
    pub fn set_webgpu_calibration(&self, c: Option<DeviceCalibration>) {
        self.set_calibration(&measured::WEBGPU, c)
    }

    /// The state of the CUDA calibration (measured or loaded by
    /// [`Self::prepare_gpu`]).
    pub fn cuda_calibration_state(&self) -> GpuCalibrationState {
        self.calibration_state(&measured::CUDA)
    }

    /// Install a CUDA calibration, or with `None` return CUDA to the prior.
    pub fn set_cuda_calibration(&self, c: Option<DeviceCalibration>) {
        self.set_calibration(&measured::CUDA, c)
    }

    /// Measure `rules`' backend against the CPU reference on this machine
    /// (`analysis::cost::measured`) on an engine thread and install the
    /// result in the cost model; `auto` routes that backend by the prior
    /// until then, and requests never wait for it. Returns false when a
    /// calibration already ran or runs (see [`Self::calibration_state`]).
    /// Dropping the engine stops it after the current analysis.
    pub fn start_calibration(&self, rules: &'static measured::Rules) -> bool {
        if !self.begin_calibration(rules, true) {
            return false;
        }
        self.spawn_calibration(rules);
        true
    }

    /// Run `rules`' calibration, marked running, on its own thread.
    fn spawn_calibration(&self, rules: &'static measured::Rules) {
        let weak = Arc::downgrade(&self.inner);
        let name = format!("mklight-{}-calibration", rules.name);
        if let Err(e) = mklight_core::platform::spawn(name, move || calibrate_into(&weak, rules)) {
            self.set_calibration_state(
                rules,
                GpuCalibrationState::Failed {
                    message: e.to_string(),
                },
            );
        }
    }

    /// After an `auto` request decided by the CUDA prior on an open CUDA
    /// device: start the CUDA calibration in the background, once per
    /// engine, if none ever started (library and command-line users that
    /// never call [`Self::prepare_gpu`]). It is cached only once
    /// `prepare_gpu` names a cache directory; the request does not wait
    /// for it. A calibration removed with [`Self::set_cuda_calibration`]
    /// is not measured again.
    fn calibrate_cuda_lazily(&self, analysis: &str, decision: Option<&serde_json::Value>) {
        if CALIBRATING.get()
            || !decision.is_some_and(|d| d["rule"] == "cuda break-even")
            || self.inner.lazy_cuda_calibration.load(Ordering::Relaxed)
            || self.calibration_state(&measured::CUDA) != GpuCalibrationState::Idle
        {
            return;
        }
        let open_cuda = matches!(
            read(&self.inner.backends).auto_candidate(analysis),
            Some(crate::backend::AutoCandidate::Gpu {
                cuda: true,
                open: true,
                ..
            })
        );
        if open_cuda
            && !self
                .inner
                .lazy_cuda_calibration
                .swap(true, Ordering::Relaxed)
            && self.begin_calibration(&measured::CUDA, false)
        {
            self.spawn_calibration(&measured::CUDA);
        }
    }

    /// The state of `rules`' calibration.
    pub fn calibration_state(&self, rules: &measured::Rules) -> GpuCalibrationState {
        self.calibrations()
            .get(rules.name)
            .cloned()
            .unwrap_or_default()
    }

    /// Install a calibration of `rules`' backend measured earlier, or a
    /// forced one; `None` returns it to the prior.
    pub fn set_calibration(&self, rules: &'static measured::Rules, c: Option<DeviceCalibration>) {
        self.set_calibration_state(
            rules,
            match &c {
                Some(c) => GpuCalibrationState::Ready {
                    calibration: c.clone(),
                },
                None => GpuCalibrationState::Idle,
            },
        );
        self.inner.cost.set_calibration(rules, c);
    }

    fn calibrations(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::BTreeMap<&'static str, GpuCalibrationState>>
    {
        self.inner
            .calibrations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn set_calibration_state(&self, rules: &'static measured::Rules, s: GpuCalibrationState) {
        self.calibrations().insert(rules.name, s);
    }

    /// Mark `rules`' calibration running, unless it runs or ran (or, unless
    /// `retry_failed`, failed).
    fn begin_calibration(&self, rules: &'static measured::Rules, retry_failed: bool) -> bool {
        let mut m = self.calibrations();
        let s = m.entry(rules.name).or_default();
        let ran = match s {
            GpuCalibrationState::Idle => false,
            GpuCalibrationState::Failed { .. } => !retry_failed,
            GpuCalibrationState::Running | GpuCalibrationState::Ready { .. } => true,
        };
        if ran {
            return false;
        }
        *s = GpuCalibrationState::Running;
        true
    }

    /// Where the CUDA calibration is cached, once known.
    fn cuda_calibration_path(&self) -> Option<std::path::PathBuf> {
        lock(&self.inner.cuda_calibration_cache).clone()
    }

    /// On the CUDA device `device` that `prepare_gpu` opened, with the
    /// kernel build `kernel_build`: remember the cache path under
    /// `cache_dir`, and install the calibration cached there when it still
    /// applies and return false; else mark the calibration running and
    /// return true (measure it). False too when one runs or ran: a running
    /// one is cached when it ends, one measured earlier in this process
    /// (lazily, [`Self::calibrate_cuda_lazily`]) is cached now.
    fn cached_cuda_calibration(
        &self,
        device: &str,
        kernel_build: &str,
        cache_dir: Option<&Path>,
    ) -> bool {
        let rules = &measured::CUDA;
        let path = cache_dir.map(|d| d.join("cost").join(format!("{}.json", rules.name)));
        if let Some(p) = &path {
            *lock(&self.inner.cuda_calibration_cache) = Some(p.clone());
        }
        if !self.begin_calibration(rules, true) {
            if let (Some(p), GpuCalibrationState::Ready { calibration }) =
                (&path, self.calibration_state(rules))
            {
                save_calibration(rules, &calibration, p);
            }
            return false;
        }
        let cached = path.as_deref().and_then(|p| {
            let threads = rayon::current_num_threads();
            let print = measured::fingerprint(rules, kernel_build);
            DeviceCalibration::load(
                p,
                rules,
                device,
                &crate::gpu::cuda_driver(),
                threads,
                &print,
            )
        });
        match cached {
            Some(c) => {
                self.set_calibration(rules, Some(c));
                false
            }
            None => true,
        }
    }

    /// Create the GPU's compute kernels in the background, so the first GPU
    /// analysis does not wait for shader compilation (several seconds on a
    /// new install, update or system-cache eviction). Opens the first
    /// registered GPU. With `cache_dir` the compiled kernels are saved there
    /// and later launches load them. Progress: [`Self::gpu_preparation`].
    /// Returns `false` when a preparation already started.
    ///
    /// On a CUDA device it then installs the CUDA cost calibration: the one
    /// cached in `<cache_dir>/cost/` when it was measured on this device,
    /// driver, engine version, thread count, kernel build and calibration
    /// runs (installed as soon as the device opens), else one measured on
    /// the same thread after the kernels and cached there unless the
    /// measurement looks disturbed ([`Self::cuda_calibration_state`]).
    /// Until the kernels are ready, `auto` routes CUDA requests to the CPU
    /// (`rule`: `cuda preparing`).
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
        let cost = self.inner.cost.clone();
        cost.set_kernels_preparing(true);
        let provider = read(&self.inner.backends).first_gpu();
        let weak = Arc::downgrade(&self.inner);
        let spawned = mklight_core::platform::spawn("mklight-gpu-prepare", move || {
            use crate::gpu::{GpuDevice, GpuPreparation};
            let prepare = || {
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
                    Some(device) => {
                        let cuda = device.backend.info().kind == mklight_gpu::BackendKind::Cuda;
                        let calibrate =
                            cuda.then(|| weak.upgrade()).flatten().is_some_and(|inner| {
                                Engine { inner }.cached_cuda_calibration(
                                    &device.device_name(),
                                    &device.backend.precision().kernel_build,
                                    cache_dir.as_deref(),
                                )
                            });
                        crate::gpu::prepare(device, cache_dir, &state);
                        cost.set_kernels_preparing(false);
                        if calibrate {
                            calibrate_into(&weak, &measured::CUDA);
                        }
                    }
                    None => state.set(GpuPreparation::Unavailable {
                        reason: provider
                            .info()
                            .reason
                            .unwrap_or_else(|| "the GPU is not available".into()),
                    }),
                }
            };
            // A panic while opening the device or creating its kernels
            // must not leave the state at `Preparing`.
            let r = catch_unwind(AssertUnwindSafe(prepare));
            cost.set_kernels_preparing(false);
            if let Err(panic) = r {
                let device = match state.get() {
                    GpuPreparation::Preparing { device, .. } => device,
                    _ => String::new(),
                };
                state.set(GpuPreparation::Failed {
                    device,
                    message: format!("internal error (panic): {}", panic_message(panic.as_ref())),
                });
            }
        });
        if let Err(e) = spawned {
            self.inner.cost.set_kernels_preparing(false);
            self.inner
                .gpu_preparation
                .set(crate::gpu::GpuPreparation::Failed {
                    device: String::new(),
                    message: e.to_string(),
                });
        }
        true
    }

    /// The kernel build of `rules`' device (opening it), for the
    /// calibration's fingerprint.
    fn kernel_build(&self, rules: &measured::Rules) -> Option<String> {
        let s = read(&self.inner.backends)
            .select(&Backend::Gpu(rules.name.into()), None, false)
            .ok()?;
        let d = s.device?.downcast::<crate::gpu::GpuDevice>().ok()?;
        Some(d.backend.precision().kernel_build.clone())
    }

    /// Progress of [`Self::prepare_gpu`].
    pub fn gpu_preparation(&self) -> crate::gpu::GpuPreparation {
        self.inner.gpu_preparation.get()
    }

    /// Make `b` the default for new jobs. Fails if it is unavailable.
    pub fn set_default_backend(&self, b: Backend) -> EngineResult<()> {
        read(&self.inner.backends).select(&b, None, false)?;
        *write(&self.inner.default_backend) = b;
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
    fn select_backend(
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
    /// decision travels in `SelectedBackend::cost_decision`. `policy` and
    /// `route` are the run's (its reference tracer is a graph node per
    /// policy; a route option can leave the device no part of the run).
    #[allow(clippy::too_many_arguments)]
    pub fn route_backend(
        &self,
        requested: Option<&Backend>,
        analysis: &str,
        settings: &serde_json::Value,
        snapshot: &mklight_core::Snapshot,
        library_hash: &str,
        policy: &crate::analysis::RunPolicy,
        route: &crate::route::RouteOptions,
        allow_fallback: bool,
    ) -> EngineResult<SelectedBackend> {
        if !CALIBRATING.get() {
            self.inner
                .foreground_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        let Some(id) = self.auto_request_id(requested) else {
            return self.select_backend(requested, analysis, allow_fallback);
        };
        let hash = snapshot.hash().0;
        let tracer_warm =
            self.inner.graph.as_ref().is_some_and(|g| {
                g.contains(&crate::analysis::tracer_key(&hash, library_hash, policy))
            });
        let mut s = crate::analysis::auto_route(
            &self.inner.cost,
            &read(&self.inner.backends),
            &crate::analysis::AutoRequest {
                analysis,
                settings,
                system: snapshot.system(),
                snapshot: &hash,
                library: library_hash,
                route,
                tracer_warm,
                // The prepared device is the one `auto` selects (the first
                // registered GPU).
                kernels_prepared: matches!(
                    self.gpu_preparation(),
                    crate::gpu::GpuPreparation::Ready { .. }
                ),
            },
        );
        s.requested = id;
        self.calibrate_cuda_lazily(analysis, s.cost_decision.as_ref());
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
}

/// Poison-tolerant locks: every write is one registration or one assignment,
/// so a panic elsewhere leaves the data consistent.
fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Cache a calibration measured here at `path`, unless it is marked not
/// to be cached (`DeviceCalibration::not_cached`).
fn save_calibration(rules: &measured::Rules, c: &DeviceCalibration, path: &Path) {
    if c.source != "measured" || !c.not_cached.is_empty() {
        return;
    }
    if let Err(e) = c.save(path) {
        eprintln!(
            "mklight: the {} cost calibration could not be cached at {}: {e}",
            rules.label,
            path.display()
        );
    }
}

/// Run `rules`' calibration ([`calibrate`]) and record the outcome: on
/// success install it, and cache it (CUDA, once `prepare_gpu` named a cache
/// directory) unless its repeats disagree or requests ran meanwhile; on
/// failure keep the prior. Nothing when the engine was dropped meanwhile.
fn calibrate_into(weak: &std::sync::Weak<EngineInner>, rules: &'static measured::Rules) {
    let requests = |inner: &EngineInner| inner.foreground_requests.load(Ordering::Relaxed);
    let before = weak.upgrade().map(|inner| requests(&inner));
    CALIBRATING.set(true);
    let r = catch_unwind(AssertUnwindSafe(|| calibrate(weak, rules)));
    CALIBRATING.set(false);
    let Some(inner) = weak.upgrade() else {
        return;
    };
    let ran = requests(&inner) - before.unwrap_or(0);
    let engine = Engine { inner };
    let failed = |message| GpuCalibrationState::Failed { message };
    match r {
        // The engine caches only CUDA's (the web host caches WebGPU's, in
        // its unchanged format).
        Ok(Ok(mut c)) if rules.name == measured::CUDA.name => {
            c.not_cached = match c.disturbed() {
                Some(why) => format!("its repeats disagree: {why}"),
                None if ran > 0 => format!("{ran} requests ran while it measured"),
                None => String::new(),
            };
            if let Some(path) = engine.cuda_calibration_path() {
                save_calibration(rules, &c, &path);
            }
            engine.set_calibration(rules, Some(c));
        }
        Ok(Ok(c)) => engine.set_calibration(rules, Some(c)),
        Ok(Err(e)) => engine.set_calibration_state(rules, failed(e.to_string())),
        Err(panic) => engine.set_calibration_state(
            rules,
            failed(format!(
                "internal error (panic): {}",
                panic_message(panic.as_ref())
            )),
        ),
    }
}

/// The calibration runs of `rules` (`analysis::cost::measured`) on its
/// sample: per backend and analysis an untimed warm-up (kernel compilation,
/// first allocations), then three fresh runs of each size. Each analysis
/// holds the engine only while it runs, so dropping the engine cancels.
fn calibrate(
    weak: &std::sync::Weak<EngineInner>,
    rules: &measured::Rules,
) -> EngineResult<DeviceCalibration> {
    use measured::Point;
    let start = mklight_core::platform::Instant::now();
    let engine = || {
        weak.upgrade()
            .map(|inner| Engine { inner })
            .ok_or(crate::EngineError::Core(mklight_core::MkError::Cancelled))
    };
    let id = rules.backend_id();
    let info = engine()?
        .backends()
        .into_iter()
        .find(|b| b.id == id && b.available)
        .ok_or_else(|| {
            crate::EngineError::InvalidArgument(format!(
                "the {} backend is not available",
                rules.label
            ))
        })?;
    let system = (rules.sample)();
    let ctx = mklight_core::JobContext::new();
    let runs = rules.runs();
    let mut points = Vec::new();
    for (name, backend) in [
        ("cpu", Backend::CpuReference),
        (rules.name, Backend::Gpu(rules.name.into())),
    ] {
        for key in rules.calibrated {
            let analysis = measured::analysis_of(key);
            let session = engine()?.session_from_project(Project::new(system.clone()))?;
            let mut warmed = false;
            for (_, settings) in runs.iter().filter(|(a, _)| a == key) {
                let mut req = crate::session::AnalysisRequest::new(analysis, settings.clone());
                req.backend = Some(backend.clone());
                req.fresh = true;
                if !std::mem::replace(&mut warmed, true) {
                    session.run_analysis(&req, &ctx)?;
                }
                let mwork = measured::mwork(analysis, settings, &system).unwrap_or(0.);
                for _ in 0..3 {
                    let t = mklight_core::platform::Instant::now();
                    let r = session.run_analysis(&req, &ctx)?;
                    if r.backend.used != info.id && name != "cpu" {
                        return Err(crate::EngineError::InvalidArgument(format!(
                            "calibration: {analysis} did not run on {} ({:?})",
                            rules.label, r.backend.fallback_reason
                        )));
                    }
                    points.push(Point {
                        analysis: (*key).into(),
                        backend: name.into(),
                        mwork,
                        ms: t.elapsed().as_secs_f64() * 1e3,
                    });
                }
            }
        }
    }
    Ok(DeviceCalibration {
        format: measured::FORMAT,
        engine_version: mklight_core::ENGINE_VERSION.into(),
        adapter: info
            .capabilities
            .device_name
            .clone()
            .unwrap_or_else(|| info.id.clone()),
        driver: if rules.name == measured::CUDA.name {
            crate::gpu::cuda_driver()
        } else {
            String::new()
        },
        threads: rayon::current_num_threads(),
        analyses: measured::fit(rules, &points),
        points,
        seconds: start.elapsed().as_secs_f64(),
        source: "measured".into(),
        fingerprint: if rules.fingerprinted {
            measured::fingerprint(rules, &engine()?.kernel_build(rules).unwrap_or_default())
        } else {
            String::new()
        },
        not_cached: String::new(),
    })
}

fn write<T>(l: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuPreparation;

    /// A GPU provider whose device cannot be opened without a panic.
    struct Panicking;

    impl BackendProvider for Panicking {
        fn id(&self) -> String {
            "gpu:panicking".into()
        }
        fn info(&self) -> BackendInfo {
            BackendInfo::cpu_reference()
        }
        fn device(&self) -> Option<crate::backend::DeviceHandle> {
            panic!("the probe blew up")
        }
    }

    #[test]
    fn a_panicking_gpu_preparation_ends_failed() {
        let e = Engine::default();
        let mut r = BackendRegistry::cpu_only();
        r.register(Arc::new(Panicking));
        *write(&e.inner.backends) = r;
        assert!(e.prepare_gpu(None));
        let start = mklight_core::platform::Instant::now();
        let state = loop {
            match e.gpu_preparation() {
                GpuPreparation::Preparing { .. } if start.elapsed() < Duration::from_secs(60) => {
                    std::thread::sleep(Duration::from_millis(2))
                }
                s => break s,
            }
        };
        match state {
            GpuPreparation::Failed { message, .. } => {
                assert!(message.contains("the probe blew up"), "{message}")
            }
            s => panic!("{s:?}"),
        }
        assert!(!e.prepare_gpu(None), "a preparation runs once per engine");
    }

    #[test]
    fn configuration_rejects_unknown_keys() {
        // What the engine writes, it reads.
        let text = serde_json::to_string(&EngineConfig::default()).unwrap();
        let back: EngineConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, EngineConfig::default());
        assert_eq!(
            serde_json::from_str::<EngineConfig>("{}").unwrap(),
            EngineConfig::default()
        );
        let flag = |text: &str| serde_json::from_str::<EngineConfig>(text);
        assert!(!EngineConfig::default().register_emulation);
        assert!(
            flag(r#"{"register_emulation": true}"#)
                .unwrap()
                .register_emulation
        );
        assert!(
            !flag(r#"{"register_emulation": false}"#)
                .unwrap()
                .register_emulation
        );
        assert!(flag(r#"{"register_emulation": "yes"}"#).is_err());
        for (bad, named) in [
            (r#"{"worker_thread": 2}"#, "`worker_thread`"),
            (r#"{"route": {"fieldmap": "device-chief"}}"#, "`fieldmap`"),
            (
                r#"{"default_backend": {"kind": "auto", "nmae": "x"}}"#,
                "\"nmae\"",
            ),
        ] {
            let e = serde_json::from_str::<EngineConfig>(bad)
                .unwrap_err()
                .to_string();
            assert!(e.contains(named), "{bad}: {e}");
        }
    }

    #[test]
    fn backend_names_resolve_against_the_registered_backends() {
        let e = Engine::default();
        assert_eq!(e.parse_backend("cpu"), Ok(Backend::CpuReference));
        assert_eq!(
            e.parse_backend("gpu:metal"),
            Ok(Backend::Gpu("metal".into()))
        );
        assert!(matches!(
            e.parse_backend("cpu_f32"),
            Err(crate::error::EngineError::InvalidArgument(_))
        ));
        e.register_backend(Arc::new(crate::gpu::GpuProvider::emulation()));
        assert_eq!(
            e.parse_backend("cpu_f32"),
            Ok(Backend::Gpu("cpu_f32".into()))
        );
    }
}
