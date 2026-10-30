//! Job scheduler: a fixed pool of worker threads running jobs against
//! immutable snapshots (ARCHITECTURE §State model).
//!
//! * `submit` returns a `JobId` at once; the job is queued FIFO.
//! * Each job gets its own `JobContext` (cancel token + progress). Jobs check
//!   the token at bounded intervals; `cancel` sets it. A queued job that is
//!   cancelled never starts.
//! * **Supersede**: a job submitted with a view key cancels the previous job
//!   with the same key, so only the newest request for one analysis window
//!   keeps running.
//! * A job whose token was cancelled ends `Cancelled` even if its closure
//!   returned `Ok`: a superseded result is stale by definition.
//! * Panics in a job are caught; the job ends `Failed` and the worker lives on.
//! * Finished jobs keep their outcome until `take_result` or `release`. At most
//!   `max_retained` finished jobs are kept; the oldest are dropped beyond that.
//!
//! Events: every state change (and, while a job runs, every progress change
//! seen by the progress thread) is posted to a FIFO and delivered to the
//! callback by one dedicated event thread, never by the thread that caused
//! it. The callback therefore runs with no scheduler (or caller) lock held
//! and may call back into the scheduler, even `wait`. A job's events are
//! posted under its lock together with the state change they report, so
//! they arrive in lifecycle order (`Queued` → `Running`* → terminal, or
//! `Queued` → `Cancelled`) whichever threads submit, cancel or run it.
//!
//! Locking order: table → job → event queue. Callback slot is taken alone.

use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use mklight_core::{JobContext, MkError};

use crate::error::{EngineError, EngineResult};

pub type JobId = u64;

/// Job lifecycle. The numeric codes are part of the C ABI (`MKJobState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(i32)]
pub enum JobState {
    Queued = 0,
    Running = 1,
    Done = 2,
    Failed = 3,
    Cancelled = 4,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Done | JobState::Failed | JobState::Cancelled
        )
    }
}

/// Snapshot of a job's state for display.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JobStatus {
    pub id: JobId,
    pub state: JobState,
    /// 0…1, monotone.
    pub progress: f64,
    pub stage: String,
    pub label: String,
    pub view_key: Option<String>,
    /// Error message when `Failed`.
    pub error: Option<String>,
    /// The job that superseded this one, if any.
    pub superseded_by: Option<JobId>,
    /// Wall time spent running so far (or in total once finished).
    pub elapsed_ms: f64,
}

/// Delivered to the event callback on every state change and, while a job
/// runs, whenever its progress or stage changed (polled every
/// `progress_interval`).
#[derive(Clone, Debug, PartialEq)]
pub struct JobEvent {
    pub id: JobId,
    pub state: JobState,
    pub progress: f64,
    pub stage: String,
}

pub type EventCallback = Arc<dyn Fn(&JobEvent) + Send + Sync>;

type Work<T> = Box<dyn FnOnce(&JobContext) -> EngineResult<T> + Send>;

/// A unit of work for the scheduler.
pub struct JobSpec<T> {
    pub label: String,
    /// Jobs with the same key supersede each other. Callers namespace keys
    /// (the engine prefixes the session id).
    pub view_key: Option<String>,
    pub work: Work<T>,
}

impl<T> JobSpec<T> {
    pub fn new(
        label: impl Into<String>,
        work: impl FnOnce(&JobContext) -> EngineResult<T> + Send + 'static,
    ) -> Self {
        JobSpec {
            label: label.into(),
            view_key: None,
            work: Box::new(work),
        }
    }
    pub fn with_view_key(mut self, key: impl Into<String>) -> Self {
        self.view_key = Some(key.into());
        self
    }
}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Worker threads (at least 1).
    pub workers: usize,
    /// Finished jobs kept for status queries and result retrieval.
    pub max_retained: usize,
    /// How often running jobs' progress is polled for the event callback.
    pub progress_interval: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let n = thread::available_parallelism().map_or(2, |n| n.get());
        SchedulerConfig {
            workers: n.clamp(1, 4),
            max_retained: 256,
            progress_interval: Duration::from_millis(50),
        }
    }
}

struct JobInner<T> {
    state: JobState,
    work: Option<Work<T>>,
    outcome: Option<EngineResult<T>>,
    error: Option<String>,
    superseded_by: Option<JobId>,
    started: Option<Instant>,
    finished: Option<Instant>,
    /// Last progress/stage delivered to the callback.
    reported: (f64, String),
}

struct Job<T> {
    id: JobId,
    label: String,
    view_key: Option<String>,
    ctx: JobContext,
    inner: Mutex<JobInner<T>>,
    done: Condvar,
}

impl<T> Job<T> {
    fn lock(&self) -> MutexGuard<'_, JobInner<T>> {
        lock(&self.inner)
    }
    fn status(&self) -> JobStatus {
        let g = self.lock();
        let elapsed = match (g.started, g.finished) {
            (Some(s), Some(f)) => f.duration_since(s),
            (Some(s), None) => s.elapsed(),
            _ => Duration::ZERO,
        };
        JobStatus {
            id: self.id,
            state: g.state,
            progress: if g.state == JobState::Done {
                1.0
            } else {
                self.ctx.progress.fraction()
            },
            stage: self.ctx.progress.stage(),
            label: self.label.clone(),
            view_key: self.view_key.clone(),
            error: g.error.clone(),
            superseded_by: g.superseded_by,
            elapsed_ms: elapsed.as_secs_f64() * 1e3,
        }
    }
    fn event(&self, state: JobState) -> JobEvent {
        JobEvent {
            id: self.id,
            state,
            progress: if state == JobState::Done {
                1.0
            } else {
                self.ctx.progress.fraction()
            },
            stage: self.ctx.progress.stage(),
        }
    }
}

/// Poison-tolerant lock: every critical section leaves the data consistent
/// (state changes are single assignments), so a panic elsewhere must not
/// wedge the scheduler.
fn lock<X>(m: &Mutex<X>) -> MutexGuard<'_, X> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct Table<T> {
    jobs: HashMap<JobId, Arc<Job<T>>>,
    queue: VecDeque<Arc<Job<T>>>,
    by_view: HashMap<String, JobId>,
    finished: VecDeque<JobId>,
    shutdown: bool,
}

#[derive(Default)]
struct CallbackSlot {
    cb: Option<EventCallback>,
    /// Threads currently inside the callback (one entry per nesting level).
    in_flight: Vec<ThreadId>,
}

/// Events waiting for the event thread, in posting order.
#[derive(Default)]
struct EventQueue {
    queue: VecDeque<JobEvent>,
    /// Events posted / handed to the callback (or dropped) so far.
    posted: u64,
    delivered: u64,
    shutdown: bool,
}

struct Shared<T> {
    table: Mutex<Table<T>>,
    next_id: AtomicU64,
    work_cv: Condvar,
    callback: Mutex<CallbackSlot>,
    callback_cv: Condvar,
    /// Cheap check that spares posting events nobody listens to.
    has_callback: AtomicBool,
    events: Mutex<EventQueue>,
    /// Signalled when an event is posted and when one has been delivered.
    events_cv: Condvar,
    event_thread: OnceLock<ThreadId>,
    monitor: (Mutex<bool>, Condvar),
    config: SchedulerConfig,
}

impl<T: Send + 'static> Shared<T> {
    /// Queue an event for the event thread. Callers post while holding the
    /// job's lock across the state change the event reports, so the events
    /// of one job are queued (and therefore delivered) in lifecycle order.
    fn post(&self, ev: JobEvent) {
        if !self.has_callback.load(Ordering::Acquire) {
            return;
        }
        let mut q = lock(&self.events);
        q.queue.push_back(ev);
        q.posted += 1;
        self.events_cv.notify_all();
    }

    /// Run the callback (if any) for one event, with no scheduler lock held.
    fn deliver(&self, ev: &JobEvent) {
        let me = thread::current().id();
        let cb = {
            let mut g = lock(&self.callback);
            let Some(cb) = g.cb.clone() else { return };
            g.in_flight.push(me);
            cb
        };
        // A panicking callback must not take the event thread down with it.
        let _ = catch_unwind(AssertUnwindSafe(|| cb(ev)));
        let mut g = lock(&self.callback);
        if let Some(i) = g.in_flight.iter().position(|t| *t == me) {
            g.in_flight.swap_remove(i);
        }
        self.callback_cv.notify_all();
    }

    fn event_loop(self: Arc<Self>) {
        loop {
            let ev = {
                let mut q = lock(&self.events);
                loop {
                    if let Some(ev) = q.queue.pop_front() {
                        break ev;
                    }
                    if q.shutdown {
                        return;
                    }
                    q = self.events_cv.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            self.deliver(&ev);
            lock(&self.events).delivered += 1;
            self.events_cv.notify_all();
        }
    }

    /// Book-keeping after a job reached a terminal state.
    fn record_finished(&self, job: &Job<T>) {
        let mut t = lock(&self.table);
        if !t.jobs.contains_key(&job.id) {
            return; // released while running
        }
        if let Some(k) = &job.view_key
            && t.by_view.get(k) == Some(&job.id)
        {
            t.by_view.remove(k);
        }
        t.finished.push_back(job.id);
        while t.finished.len() > self.config.max_retained {
            if let Some(old) = t.finished.pop_front() {
                t.jobs.remove(&old);
            }
        }
    }

    /// Cancel `job`; a queued job becomes `Cancelled` immediately.
    fn cancel_job(&self, job: &Arc<Job<T>>) {
        job.ctx.cancel.cancel();
        let became_cancelled = {
            let mut g = job.lock();
            if g.state == JobState::Queued {
                g.state = JobState::Cancelled;
                g.work = None;
                g.outcome = Some(Err(EngineError::Core(MkError::Cancelled)));
                g.finished = Some(Instant::now());
                self.post(job.event(JobState::Cancelled));
                job.done.notify_all();
                true
            } else {
                false
            }
        };
        if became_cancelled {
            self.record_finished(job);
        }
    }

    fn worker_loop(self: Arc<Self>) {
        loop {
            let job = {
                let mut t = lock(&self.table);
                loop {
                    if let Some(j) = t.queue.pop_front() {
                        break j;
                    }
                    if t.shutdown {
                        return;
                    }
                    t = self.work_cv.wait(t).unwrap_or_else(|e| e.into_inner());
                }
            };
            let work = {
                let mut g = job.lock();
                if g.state != JobState::Queued {
                    continue; // cancelled while queued
                }
                g.state = JobState::Running;
                g.started = Some(Instant::now());
                self.post(job.event(JobState::Running));
                g.work.take()
            };
            let Some(work) = work else { continue };

            let ctx = job.ctx.clone();
            let outcome = match catch_unwind(AssertUnwindSafe(|| work(&ctx))) {
                Ok(r) => r,
                Err(p) => {
                    let msg = p
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".into());
                    Err(EngineError::JobFailed(format!(
                        "internal error (panic): {msg}"
                    )))
                }
            };
            let state = match &outcome {
                _ if ctx.cancel.is_cancelled() => JobState::Cancelled,
                Ok(_) => JobState::Done,
                Err(e) if e.is_cancelled() => JobState::Cancelled,
                Err(_) => JobState::Failed,
            };
            {
                let mut g = job.lock();
                g.state = state;
                g.finished = Some(Instant::now());
                g.error = match (&outcome, state) {
                    (Err(e), JobState::Failed) => Some(e.to_string()),
                    _ => None,
                };
                g.outcome = Some(if state == JobState::Cancelled {
                    Err(EngineError::Core(MkError::Cancelled))
                } else {
                    outcome
                });
                self.post(job.event(state));
                job.done.notify_all();
            }
            self.record_finished(&job);
        }
    }

    fn monitor_loop(self: Arc<Self>) {
        let (m, cv) = &self.monitor;
        loop {
            {
                let g = lock(m);
                let (g, _) = cv
                    .wait_timeout(g, self.config.progress_interval)
                    .unwrap_or_else(|e| e.into_inner());
                if *g {
                    return;
                }
            }
            if !self.has_callback.load(Ordering::Acquire) {
                continue;
            }
            let jobs: Vec<Arc<Job<T>>> = {
                let t = lock(&self.table);
                t.jobs.values().cloned().collect()
            };
            for job in jobs {
                // Checking the state and posting under the job lock means no
                // progress event can follow the job's terminal event.
                let mut g = job.lock();
                if g.state != JobState::Running {
                    continue;
                }
                let f = job.ctx.progress.fraction();
                let s = job.ctx.progress.stage();
                if (f - g.reported.0).abs() < 1e-4 && s == g.reported.1 {
                    continue;
                }
                g.reported = (f, s.clone());
                self.post(JobEvent {
                    id: job.id,
                    state: JobState::Running,
                    progress: f,
                    stage: s,
                });
            }
        }
    }
}

/// The worker pool. Dropping it cancels all jobs and joins the threads.
pub struct Scheduler<T: Send + 'static> {
    shared: Arc<Shared<T>>,
    /// Workers and the progress thread.
    threads: Vec<JoinHandle<()>>,
    event_thread: Option<JoinHandle<()>>,
}

impl<T: Send + 'static> Scheduler<T> {
    pub fn new(config: SchedulerConfig) -> Self {
        let workers = config.workers.max(1);
        let shared = Arc::new(Shared {
            table: Mutex::new(Table {
                jobs: HashMap::new(),
                queue: VecDeque::new(),
                by_view: HashMap::new(),
                finished: VecDeque::new(),
                shutdown: false,
            }),
            next_id: AtomicU64::new(1),
            work_cv: Condvar::new(),
            callback: Mutex::new(CallbackSlot::default()),
            callback_cv: Condvar::new(),
            has_callback: AtomicBool::new(false),
            events: Mutex::new(EventQueue::default()),
            events_cv: Condvar::new(),
            event_thread: OnceLock::new(),
            monitor: (Mutex::new(false), Condvar::new()),
            config,
        });
        let mut threads = Vec::with_capacity(workers + 1);
        for i in 0..workers {
            let s = shared.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("mklight-worker-{i}"))
                    .spawn(move || s.worker_loop())
                    .expect("spawn worker thread"),
            );
        }
        let s = shared.clone();
        threads.push(
            thread::Builder::new()
                .name("mklight-progress".into())
                .spawn(move || s.monitor_loop())
                .expect("spawn progress thread"),
        );
        let s = shared.clone();
        let event_thread = thread::Builder::new()
            .name("mklight-events".into())
            .spawn(move || s.event_loop())
            .expect("spawn event thread");
        let _ = shared.event_thread.set(event_thread.thread().id());
        Scheduler {
            shared,
            threads,
            event_thread: Some(event_thread),
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.shared.config
    }

    /// Install (or clear) the event callback. It runs on the scheduler's
    /// single event thread, one event at a time, with no scheduler lock
    /// held, so it may call any scheduler (or engine) method, including
    /// `wait`. Events posted while no callback is installed are not kept.
    ///
    /// When this returns, no other thread is still running the previous
    /// callback, so whatever it captured may be freed. Calling it from
    /// inside the callback is allowed.
    pub fn set_callback(&self, cb: Option<EventCallback>) {
        let me = thread::current().id();
        let mut g = lock(&self.shared.callback);
        self.shared
            .has_callback
            .store(cb.is_some(), Ordering::Release);
        g.cb = cb;
        while g.in_flight.iter().any(|t| *t != me) {
            g = self
                .shared
                .callback_cv
                .wait(g)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Block until every event posted before this call has been handed to
    /// the callback, or until `timeout`. Returns whether that happened.
    /// From the event thread itself (inside the callback) it returns `false`
    /// at once, since waiting there could never finish.
    pub fn flush_events(&self, timeout: Duration) -> bool {
        if self.shared.event_thread.get() == Some(&thread::current().id()) {
            return false;
        }
        let deadline = Instant::now().checked_add(timeout);
        let mut q = lock(&self.shared.events);
        let target = q.posted;
        while q.delivered < target {
            let wait = match deadline {
                None => Duration::from_secs(3600),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return false;
                    }
                    d - now
                }
            };
            q = self
                .shared
                .events_cv
                .wait_timeout(q, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }

    /// Queue a job. A view key cancels the previous job with the same key.
    pub fn submit(&self, spec: JobSpec<T>) -> EngineResult<JobId> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let job = Arc::new(Job {
            id,
            label: spec.label,
            view_key: spec.view_key.clone(),
            ctx: JobContext::new(),
            inner: Mutex::new(JobInner {
                state: JobState::Queued,
                work: Some(spec.work),
                outcome: None,
                error: None,
                superseded_by: None,
                started: None,
                finished: None,
                reported: (0.0, String::new()),
            }),
            done: Condvar::new(),
        });
        let superseded = {
            let mut t = lock(&self.shared.table);
            if t.shutdown {
                return Err(EngineError::ShutDown);
            }
            // Posted before the job becomes visible to workers or `cancel`,
            // so `Queued` precedes every other event of this job.
            self.shared.post(job.event(JobState::Queued));
            let superseded = spec
                .view_key
                .and_then(|k| t.by_view.insert(k, id))
                .and_then(|old| t.jobs.get(&old).cloned());
            t.jobs.insert(id, job.clone());
            t.queue.push_back(job);
            self.shared.work_cv.notify_one();
            superseded
        };
        if let Some(old) = superseded {
            old.lock().superseded_by = Some(id);
            self.shared.cancel_job(&old);
        }
        Ok(id)
    }

    fn get(&self, id: JobId) -> EngineResult<Arc<Job<T>>> {
        lock(&self.shared.table)
            .jobs
            .get(&id)
            .cloned()
            .ok_or(EngineError::UnknownJob(id))
    }

    pub fn status(&self, id: JobId) -> EngineResult<JobStatus> {
        Ok(self.get(id)?.status())
    }

    /// Status of every job still known to the scheduler, by id.
    pub fn statuses(&self) -> Vec<JobStatus> {
        let jobs: Vec<Arc<Job<T>>> = lock(&self.shared.table).jobs.values().cloned().collect();
        let mut v: Vec<JobStatus> = jobs.iter().map(|j| j.status()).collect();
        v.sort_by_key(|s| s.id);
        v
    }

    /// Request cancellation. Cancelling a finished job is a no-op.
    pub fn cancel(&self, id: JobId) -> EngineResult<()> {
        let job = self.get(id)?;
        self.shared.cancel_job(&job);
        Ok(())
    }

    /// Block until the job finishes, or until `timeout` (`None` = forever;
    /// a timeout too large to represent also waits forever). Returns the
    /// terminal state, or `Timeout`.
    pub fn wait(&self, id: JobId, timeout: Option<Duration>) -> EngineResult<JobState> {
        let job = self.get(id)?;
        let deadline = timeout.and_then(|t| Instant::now().checked_add(t));
        let mut g = job.lock();
        while !g.state.is_terminal() {
            match deadline {
                None => g = job.done.wait(g).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Err(EngineError::Timeout(id));
                    }
                    g = job
                        .done
                        .wait_timeout(g, d - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
        Ok(g.state)
    }

    /// Take a finished job's outcome and forget the job. `Done` yields the
    /// value; `Failed` yields the job's error; `Cancelled` yields
    /// `Core(Cancelled)`. A job that has not finished stays untouched.
    pub fn take_result(&self, id: JobId) -> EngineResult<T> {
        let job = self.get(id)?;
        let outcome = {
            let mut g = job.lock();
            if !g.state.is_terminal() {
                return Err(EngineError::NotFinished(id));
            }
            g.outcome.take()
        };
        self.forget(id);
        outcome.unwrap_or(Err(EngineError::UnknownJob(id)))
    }

    /// Cancel (if needed) and forget a job.
    pub fn release(&self, id: JobId) -> EngineResult<()> {
        let job = self.get(id)?;
        self.shared.cancel_job(&job);
        self.forget(id);
        Ok(())
    }

    fn forget(&self, id: JobId) {
        let mut t = lock(&self.shared.table);
        if let Some(job) = t.jobs.remove(&id) {
            t.finished.retain(|&j| j != id);
            if let Some(k) = &job.view_key
                && t.by_view.get(k) == Some(&id)
            {
                t.by_view.remove(k);
            }
        }
    }

    /// Cancel every queued and running job.
    pub fn cancel_all(&self) {
        let jobs: Vec<Arc<Job<T>>> = lock(&self.shared.table).jobs.values().cloned().collect();
        for j in jobs {
            self.shared.cancel_job(&j);
        }
    }
}

impl<T: Send + 'static> Drop for Scheduler<T> {
    fn drop(&mut self) {
        lock(&self.shared.table).shutdown = true;
        self.cancel_all();
        self.shared.work_cv.notify_all();
        {
            let (m, cv) = &self.shared.monitor;
            *lock(m) = true;
            cv.notify_all();
        }
        // A job closure may own the last handle; never join the own thread.
        let me = thread::current().id();
        for t in self.threads.drain(..) {
            if t.thread().id() != me {
                let _ = t.join();
            }
        }
        // Every event has been posted now; the event thread delivers what is
        // left and exits. If the last owner is dropped from inside the
        // callback (on the event thread), joining it would deadlock: detach.
        lock(&self.shared.events).shutdown = true;
        self.shared.events_cv.notify_all();
        if let Some(t) = self.event_thread.take()
            && t.thread().id() != me
        {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    fn sched(workers: usize) -> Scheduler<u64> {
        Scheduler::new(SchedulerConfig {
            workers,
            max_retained: 64,
            progress_interval: Duration::from_millis(5),
        })
    }

    const LONG: Duration = Duration::from_secs(20);

    /// A job that spins (checking cancellation) until cancelled, after
    /// signalling that it started.
    fn spinner(started: mpsc::Sender<()>) -> JobSpec<u64> {
        JobSpec::new("spin", move |ctx: &JobContext| {
            let _ = started.send(());
            let mut i = 0u64;
            loop {
                ctx.check()?;
                ctx.report(((i % 1000) as f64) / 1000.0, "spinning");
                i += 1;
                thread::sleep(Duration::from_micros(200));
            }
        })
    }

    #[test]
    fn runs_job_and_returns_result() {
        let s = sched(2);
        let id = s
            .submit(JobSpec::new("sum", |ctx: &JobContext| {
                let mut acc = 0u64;
                for i in 0..=100u64 {
                    ctx.check()?;
                    acc += i;
                    ctx.report(i as f64 / 100.0, "summing");
                }
                Ok(acc)
            }))
            .unwrap();
        assert_eq!(s.wait(id, Some(LONG)).unwrap(), JobState::Done);
        let st = s.status(id).unwrap();
        assert_eq!(st.progress, 1.0);
        assert_eq!(st.stage, "summing");
        assert_eq!(s.take_result(id).unwrap(), 5050);
        assert_eq!(s.status(id), Err(EngineError::UnknownJob(id)));
    }

    #[test]
    fn cancel_running_job() {
        let s = sched(1);
        let (tx, rx) = mpsc::channel();
        let id = s.submit(spinner(tx)).unwrap();
        rx.recv_timeout(LONG).unwrap();
        assert_eq!(s.status(id).unwrap().state, JobState::Running);
        assert_eq!(
            s.wait(id, Some(Duration::from_millis(20))),
            Err(EngineError::Timeout(id))
        );
        s.cancel(id).unwrap();
        assert_eq!(s.wait(id, Some(LONG)).unwrap(), JobState::Cancelled);
        assert!(s.take_result(id).unwrap_err().is_cancelled());
    }

    #[test]
    fn cancel_queued_job_never_runs() {
        let s = sched(1);
        let (tx, rx) = mpsc::channel();
        let blocker = s.submit(spinner(tx)).unwrap();
        rx.recv_timeout(LONG).unwrap();
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        let queued = s
            .submit(JobSpec::new("never", move |_: &JobContext| {
                r.fetch_add(1, Ordering::SeqCst);
                Ok(1)
            }))
            .unwrap();
        assert_eq!(s.status(queued).unwrap().state, JobState::Queued);
        s.cancel(queued).unwrap();
        assert_eq!(s.status(queued).unwrap().state, JobState::Cancelled);
        s.cancel(blocker).unwrap();
        s.wait(blocker, Some(LONG)).unwrap();
        // Let the worker drain the queue, then confirm the job never ran.
        let probe = s
            .submit(JobSpec::new("probe", |_: &JobContext| Ok(7)))
            .unwrap();
        assert_eq!(s.wait(probe, Some(LONG)).unwrap(), JobState::Done);
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn supersede_cancels_previous_view_job() {
        let s = sched(2);
        let (tx, rx) = mpsc::channel();
        let a = s.submit(spinner(tx).with_view_key("spot")).unwrap();
        rx.recv_timeout(LONG).unwrap();
        // Different key: unaffected.
        let (tx2, rx2) = mpsc::channel();
        let other = s.submit(spinner(tx2).with_view_key("mtf")).unwrap();
        rx2.recv_timeout(LONG).unwrap();
        let b = s
            .submit(JobSpec::new("b", |_: &JobContext| Ok(42)).with_view_key("spot"))
            .unwrap();
        assert_eq!(s.wait(a, Some(LONG)).unwrap(), JobState::Cancelled);
        assert_eq!(s.status(a).unwrap().superseded_by, Some(b));
        assert_eq!(s.wait(b, Some(LONG)).unwrap(), JobState::Done);
        assert_eq!(s.take_result(b).unwrap(), 42);
        assert_eq!(s.status(other).unwrap().state, JobState::Running);
        s.cancel(other).unwrap();
        assert_eq!(s.wait(other, Some(LONG)).unwrap(), JobState::Cancelled);
    }

    #[test]
    fn ok_after_cancel_is_still_cancelled() {
        let s = sched(1);
        let (tx, rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let id = s
            .submit(JobSpec::new("ignores-cancel", move |_: &JobContext| {
                let _ = tx.send(());
                let _ = go_rx.recv_timeout(LONG);
                Ok(1)
            }))
            .unwrap();
        rx.recv_timeout(LONG).unwrap();
        s.cancel(id).unwrap();
        go_tx.send(()).unwrap();
        assert_eq!(s.wait(id, Some(LONG)).unwrap(), JobState::Cancelled);
    }

    #[test]
    fn failure_and_panic_are_reported() {
        let s = sched(1);
        let f = s
            .submit(JobSpec::new("fail", |_: &JobContext| {
                Err(EngineError::Core(MkError::Numerical("diverged".into())))
            }))
            .unwrap();
        let p = s
            .submit(JobSpec::new(
                "panic",
                |_: &JobContext| -> EngineResult<u64> { panic!("kaboom") },
            ))
            .unwrap();
        assert_eq!(s.wait(f, Some(LONG)).unwrap(), JobState::Failed);
        assert!(s.status(f).unwrap().error.unwrap().contains("diverged"));
        assert_eq!(
            s.take_result(f),
            Err(EngineError::Core(MkError::Numerical("diverged".into())))
        );
        assert_eq!(s.wait(p, Some(LONG)).unwrap(), JobState::Failed);
        assert!(s.status(p).unwrap().error.unwrap().contains("kaboom"));
        // The worker survived the panic.
        let ok = s
            .submit(JobSpec::new("ok", |_: &JobContext| Ok(3)))
            .unwrap();
        assert_eq!(s.wait(ok, Some(LONG)).unwrap(), JobState::Done);
    }

    #[test]
    fn events_cover_lifecycle_and_progress() {
        let s = sched(1);
        let events = Arc::new(Mutex::new(Vec::<JobEvent>::new()));
        let e = events.clone();
        s.set_callback(Some(Arc::new(move |ev: &JobEvent| {
            lock(&e).push(ev.clone());
        })));
        let id = s
            .submit(JobSpec::new("slow", |ctx: &JobContext| {
                for i in 0..10 {
                    ctx.check()?;
                    ctx.report(i as f64 / 10.0, "stage-x");
                    thread::sleep(Duration::from_millis(10));
                }
                Ok(1)
            }))
            .unwrap();
        s.wait(id, Some(LONG)).unwrap();
        // `wait` may return just before the terminal event is delivered.
        assert!(s.flush_events(LONG));
        s.set_callback(None);
        let ev = lock(&events).clone();
        let states: Vec<JobState> = ev.iter().map(|e| e.state).collect();
        assert_eq!(states.first(), Some(&JobState::Queued));
        assert_eq!(states.last(), Some(&JobState::Done));
        assert!(
            ev.iter()
                .any(|e| e.state == JobState::Running && e.stage == "stage-x" && e.progress > 0.0),
            "no progress event: {ev:?}"
        );
        // Progress delivered in order never decreases.
        let p: Vec<f64> = ev.iter().map(|e| e.progress).collect();
        assert!(p.windows(2).all(|w| w[1] >= w[0]), "{p:?}");
    }

    #[test]
    fn clearing_callback_waits_for_in_flight_calls() {
        let s = sched(1);
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let done = Arc::new(AtomicUsize::new(0));
        let d = done.clone();
        let entered_tx = Mutex::new(entered_tx);
        s.set_callback(Some(Arc::new(move |ev: &JobEvent| {
            if ev.state == JobState::Running {
                let _ = lock(&entered_tx).send(());
                thread::sleep(Duration::from_millis(100));
                d.fetch_add(1, Ordering::SeqCst);
            }
        })));
        let id = s.submit(JobSpec::new("x", |_: &JobContext| Ok(1))).unwrap();
        entered_rx.recv_timeout(LONG).unwrap();
        // The worker is inside the callback; clearing must wait for it.
        s.set_callback(None);
        assert_eq!(done.load(Ordering::SeqCst), 1);
        s.wait(id, Some(LONG)).unwrap();
    }

    #[test]
    fn retention_is_bounded() {
        let s = Scheduler::<u64>::new(SchedulerConfig {
            workers: 2,
            max_retained: 3,
            progress_interval: Duration::from_millis(50),
        });
        let ids: Vec<JobId> = (0..10)
            .map(|i| {
                s.submit(JobSpec::new("n", move |_: &JobContext| Ok(i)))
                    .unwrap()
            })
            .collect();
        for &id in &ids {
            let _ = s.wait(id, Some(LONG));
        }
        // Allow the last record_finished calls to land.
        let t0 = Instant::now();
        while s.statuses().len() > 3 && t0.elapsed() < LONG {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(s.statuses().len(), 3);
    }

    #[test]
    fn many_threads_hammering_is_consistent() {
        let s = Arc::new(sched(4));
        let mut hs = Vec::new();
        for t in 0..8u64 {
            let s = s.clone();
            hs.push(thread::spawn(move || {
                let mut sum = 0u64;
                for i in 0..50u64 {
                    let key = format!("view-{}", i % 3);
                    let id = s
                        .submit(
                            JobSpec::new("h", move |ctx: &JobContext| {
                                ctx.check()?;
                                Ok(t * 1000 + i)
                            })
                            .with_view_key(key),
                        )
                        .unwrap();
                    if i % 7 == 0 {
                        let _ = s.cancel(id);
                    }
                    match s.wait(id, Some(LONG)) {
                        Ok(JobState::Done) => {
                            if let Ok(v) = s.take_result(id) {
                                sum += v;
                            }
                        }
                        Ok(st) => assert!(st.is_terminal()),
                        Err(EngineError::UnknownJob(_)) => {} // pruned
                        Err(e) => panic!("{e}"),
                    }
                }
                sum
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        // Nothing left running.
        assert!(s.statuses().iter().all(|st| st.state.is_terminal()));
    }

    /// Per job: `Queued`, then `Cancelled` directly, or `Running`* and one
    /// terminal state last.
    fn check_lifecycle(id: JobId, ev: &[JobState]) -> Result<(), String> {
        let bad = || Err(format!("job {id}: {ev:?}"));
        match ev {
            [JobState::Queued, JobState::Cancelled] => Ok(()),
            [JobState::Queued, JobState::Running, rest @ ..] => match rest.split_last() {
                Some((last, mid))
                    if last.is_terminal() && mid.iter().all(|s| *s == JobState::Running) =>
                {
                    Ok(())
                }
                _ => bad(),
            },
            _ => bad(),
        }
    }

    #[test]
    fn event_order_holds_under_concurrent_supersede_and_cancel() {
        let s = Arc::new(sched(4));
        let log = Arc::new(Mutex::new(HashMap::<JobId, Vec<JobState>>::new()));
        let l = log.clone();
        s.set_callback(Some(Arc::new(move |ev: &JobEvent| {
            lock(&l).entry(ev.id).or_default().push(ev.state);
        })));
        let mut hs = Vec::new();
        for t in 0..8u64 {
            let s = s.clone();
            hs.push(thread::spawn(move || {
                let mut ids = Vec::new();
                for i in 0..200u64 {
                    let id = s
                        .submit(
                            JobSpec::new("h", move |ctx: &JobContext| {
                                ctx.report(0.5, "half");
                                ctx.check()?;
                                Ok(t * 1000 + i)
                            })
                            .with_view_key("same"),
                        )
                        .unwrap();
                    if i % 5 == 0 {
                        let _ = s.cancel(id);
                    }
                    ids.push(id);
                }
                ids
            }));
        }
        let ids: Vec<JobId> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
        for &id in &ids {
            match s.wait(id, Some(LONG)) {
                Ok(st) => assert!(st.is_terminal()),
                Err(EngineError::UnknownJob(_)) => {} // pruned after finishing
                Err(e) => panic!("{e}"),
            }
        }
        assert!(s.flush_events(LONG));
        s.set_callback(None);
        let log = lock(&log);
        assert_eq!(log.len(), ids.len());
        let bad: Vec<String> = ids
            .iter()
            .filter_map(|id| check_lifecycle(*id, &log[id]).err())
            .collect();
        assert!(
            bad.is_empty(),
            "{} bad, e.g. {:?}",
            bad.len(),
            &bad[..bad.len().min(5)]
        );
    }

    #[test]
    fn callback_may_wait_on_the_job_it_is_told_about() {
        let s = Arc::new(sched(1));
        let weak = Arc::downgrade(&s);
        let (tx, rx) = mpsc::channel::<(JobId, Result<JobState, EngineError>)>();
        let tx = Mutex::new(tx);
        s.set_callback(Some(Arc::new(move |ev: &JobEvent| {
            if ev.state == JobState::Queued
                && let Some(s) = weak.upgrade()
            {
                // Re-entering the scheduler from the callback, and blocking
                // on the very job just queued, must not deadlock.
                let r = s.wait(ev.id, Some(LONG));
                let _ = s.status(ev.id);
                let _ = lock(&tx).send((ev.id, r));
            }
        })));
        let t0 = Instant::now();
        let id = s.submit(JobSpec::new("x", |_: &JobContext| Ok(5))).unwrap();
        assert!(t0.elapsed() < Duration::from_secs(5), "submit blocked");
        let (got, r) = rx.recv_timeout(LONG).expect("callback never finished");
        assert_eq!((got, r), (id, Ok(JobState::Done)));
        assert_eq!(s.take_result(id).unwrap(), 5);
        s.set_callback(None);
    }

    #[test]
    fn huge_timeouts_mean_forever() {
        let s = sched(1);
        let id = s.submit(JobSpec::new("x", |_: &JobContext| Ok(1))).unwrap();
        assert_eq!(s.wait(id, Some(Duration::MAX)).unwrap(), JobState::Done);
        assert!(s.flush_events(Duration::MAX));
    }

    #[test]
    fn drop_cancels_running_jobs() {
        let s = sched(1);
        let (tx, rx) = mpsc::channel();
        let _id = s.submit(spinner(tx)).unwrap();
        rx.recv_timeout(LONG).unwrap();
        let t0 = Instant::now();
        drop(s); // must not hang
        assert!(t0.elapsed() < LONG);
    }
}
