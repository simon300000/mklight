//! MKLight engine for the web (docs/WEB.md): the wasm-bindgen API that the
//! engine worker (`apps/web/engine/src/worker.ts`) calls. It is the web's
//! counterpart of `mklight-ffi` and wraps the same `mklight-engine` calls;
//! no optics logic lives here.
//!
//! * One engine per wasm instance, created by `engine_create`. Sessions are
//!   numbered handles (`session_new`, `session_free`).
//! * Structured data crosses as JSON text, exactly the shapes of
//!   docs/FFI.md. Bulk result buffers cross as typed arrays copied once out
//!   of the shared wasm memory (`job_take_result`), which the worker then
//!   transfers to the page without another copy.
//! * Errors are thrown as JS `Error`s with a `kind` property (the
//!   `EngineErrorKind` names of `types.ts`).
//! * Threads: every thread (the rayon pool's, and the engine's scheduler
//!   workers, progress and event threads) runs on a host worker started by
//!   the glue (`thread_host`, `wanted_host_threads`; see
//!   `mklight_core::platform`). The rayon pool is plain rayon, started
//!   through that mechanism by `init_thread_pool` (docs/WEB.md says why not
//!   wasm-bindgen-rayon's own worker glue).
//! * The engine is only ever called from the engine worker and the threads
//!   it starts, never from the page's main thread (which cannot block).

#![cfg(target_arch = "wasm32")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use js_sys::{Array, Object, Reflect};
use serde_json::Value;
use wasm_bindgen::prelude::*;

use mklight_core::import::{PrescriptionFormat, import_prescription};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, OpticalSystem, platform};
use mklight_engine::buffer::BufferData;
use mklight_engine::project::Project;
use mklight_engine::session::{AnalysisRequest, Session};
use mklight_engine::{Engine, EngineConfig, EngineError, ErrorKind};

// ---------------------------------------------------------------------------
// Errors

fn kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::InvalidModel => "invalidModel",
        ErrorKind::UnknownMaterial | ErrorKind::UnknownAnalysis | ErrorKind::UnknownJob => {
            "notFound"
        }
        ErrorKind::WavelengthOutOfRange | ErrorKind::InvalidArgument => "invalidArgument",
        ErrorKind::InvalidSettings => "invalidSettings",
        ErrorKind::Parse | ErrorKind::ProjectFile => "parse",
        ErrorKind::Io => "io",
        ErrorKind::Unsupported => "unsupported",
        ErrorKind::Compute => "compute",
        ErrorKind::NotFinished => "notReady",
        ErrorKind::Timeout => "timeout",
        ErrorKind::JobFailed => "jobFailed",
        ErrorKind::ShutDown => "internal",
    }
}

fn js_error(kind: &str, message: &str) -> JsValue {
    let e = js_sys::Error::new(message);
    e.set_name("EngineError");
    let _ = Reflect::set(&e, &"kind".into(), &kind.into());
    e.into()
}

impl From<EngineErrorJs> for JsValue {
    fn from(e: EngineErrorJs) -> JsValue {
        js_error(e.0, &e.1)
    }
}

/// An error on its way to JS: (kind, message).
struct EngineErrorJs(&'static str, String);

impl From<EngineError> for EngineErrorJs {
    fn from(e: EngineError) -> Self {
        EngineErrorJs(kind_name(e.kind()), e.to_string())
    }
}

impl From<MkError> for EngineErrorJs {
    fn from(e: MkError) -> Self {
        EngineError::from(e).into()
    }
}

fn arg(message: impl Into<String>) -> EngineErrorJs {
    EngineErrorJs("invalidArgument", message.into())
}

type R<T> = Result<T, EngineErrorJs>;

/// JSON text that must be an object of type `T` (docs/FFI.md's rule: not
/// JSON → `parse`, another shape → `invalidArgument`).
fn object<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> R<T> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| EngineErrorJs("parse", format!("{what}: invalid JSON: {e}")))?;
    if !v.is_object() {
        return Err(arg(format!("{what} must be a JSON object")));
    }
    serde_json::from_value(v).map_err(|e| arg(format!("{what}: {e}")))
}

/// Optional JSON: empty text is `null`.
fn optional(text: &str, what: &str) -> R<Value> {
    if text.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(text)
        .map_err(|e| EngineErrorJs("parse", format!("{what}: invalid JSON: {e}")))
}

fn to_json(v: &impl serde::Serialize) -> R<String> {
    serde_json::to_string(v).map_err(|e| EngineErrorJs("internal", format!("JSON: {e}")))
}

// ---------------------------------------------------------------------------
// State

struct State {
    engine: Engine,
    sessions: Mutex<HashMap<u32, Arc<Mutex<Session>>>>,
    next_session: AtomicU32,
}

static STATE: OnceLock<State> = OnceLock::new();

fn state() -> R<&'static State> {
    STATE
        .get()
        .ok_or_else(|| EngineErrorJs("notReady", "engine_create has not been called".into()))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Session edits are all-or-nothing, so a poisoned lock's state is sound.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn session(id: u32) -> R<Arc<Mutex<Session>>> {
    lock(&state()?.sessions)
        .get(&id)
        .cloned()
        .ok_or_else(|| arg(format!("no session {id}")))
}

// ---------------------------------------------------------------------------
// Engine

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// Engine version (`run_info.engine_version`).
#[wasm_bindgen]
pub fn engine_version() -> String {
    mklight_core::ENGINE_VERSION.to_string()
}

/// Create the engine. `config_json` is an `EngineConfig` object (docs/FFI.md)
/// or empty for the web defaults; `cores` is the browser's core count (the
/// cost model's view of the host). A second call fails.
#[wasm_bindgen]
pub fn engine_create(config_json: &str, cores: u32) -> Result<(), JsValue> {
    platform::set_available_parallelism(cores as usize);
    let cfg: EngineConfig = if config_json.trim().is_empty() {
        EngineConfig::default()
    } else {
        object(config_json, "config_json")?
    };
    let engine = Engine::new(cfg);
    STATE
        .set(State {
            engine,
            sessions: Mutex::new(HashMap::new()),
            next_session: AtomicU32::new(1),
        })
        .map_err(|_| js_error("invalidArgument", "the engine exists already"))?;
    Ok(())
}

/// Build rayon's global pool with `threads` threads, each on a host worker.
/// Call once, before `engine_create`; then start the host workers
/// (`wanted_host_threads`). The pool's threads start once this worker
/// returns to its event loop.
#[wasm_bindgen]
pub fn init_thread_pool(threads: u32) -> Result<(), JsValue> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1) as usize)
        .thread_name(|i| format!("mklight-rayon-{i}"))
        .spawn_handler(|t| {
            let name = t.name().unwrap_or("mklight-rayon").to_string();
            platform::spawn(name, move || t.run()).map(|_| ())
        })
        .build_global()
        .map_err(|e| js_error("internal", &format!("rayon pool: {e}")))
}

/// Body of the GPU worker (docs/GPU.md §11): opens `navigator.gpu`'s
/// adapter, publishes the request channel in the shared memory and serves
/// the engine's threads forever; it returns only when the browser has no
/// usable adapter (the reason then reaches the engine when it opens
/// `gpu:webgpu`). Never blocks: it sleeps in `Atomics.waitAsync`.
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn gpu_thread() {
    mklight_webgpu::web::run_gpu_thread().await;
}

/// Register the WebGPU backend `gpu:webgpu` (after `engine_create`, once
/// the GPU worker is started). Its device opens on first use, through the
/// GPU worker's proxy.
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub fn register_webgpu() -> Result<(), JsValue> {
    state()?
        .engine
        .register_backend(Arc::new(mklight_engine::gpu::GpuProvider::webgpu_browser()));
    Ok(())
}

/// Host workers the glue should start now for engine threads that wait for
/// one (`mklight_core::platform`). Each started worker calls `thread_host`.
#[wasm_bindgen]
pub fn wanted_host_threads() -> u32 {
    platform::wanted_host_threads() as u32
}

/// The glue could not start a host worker `wanted_host_threads` asked for;
/// the next `wanted_host_threads` asks for it again.
#[wasm_bindgen]
pub fn host_start_failed() {
    platform::host_start_failed();
}

/// The compiled module, for starting host workers on the same code.
#[wasm_bindgen]
pub fn wasm_module() -> JsValue {
    wasm_bindgen::module()
}

/// The shared memory, for starting host workers on the same heap.
#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Body of a host worker: runs engine threads, never returns.
#[wasm_bindgen]
pub fn thread_host() {
    platform::run_thread_host();
}

/// `mk_engine_list_analyses`.
#[wasm_bindgen]
pub fn list_analyses() -> Result<String, JsValue> {
    Ok(to_json(&state()?.engine.analyses())?)
}

/// `mk_engine_list_backends`.
#[wasm_bindgen]
pub fn list_backends() -> Result<String, JsValue> {
    Ok(to_json(&state()?.engine.backends())?)
}

// ---------------------------------------------------------------------------
// Sessions

fn add_session(s: Session) -> R<u32> {
    let st = state()?;
    let id = st.next_session.fetch_add(1, Ordering::Relaxed);
    lock(&st.sessions).insert(id, Arc::new(Mutex::new(s)));
    Ok(id)
}

/// A session with a new untitled project; returns its handle.
#[wasm_bindgen]
pub fn session_new() -> Result<u32, JsValue> {
    let s = state()?.engine.new_session();
    Ok(add_session(s)?)
}

/// Release a session (its running jobs keep running).
#[wasm_bindgen]
pub fn session_free(id: u32) -> Result<(), JsValue> {
    lock(&state()?.sessions).remove(&id);
    Ok(())
}

/// `mk_session_info_json`.
#[wasm_bindgen]
pub fn session_info(id: u32) -> Result<String, JsValue> {
    Ok(to_json(&lock(&*session(id)?).info())?)
}

/// Replace the system; returns `{"hash", "library_hash"}`. On failure the
/// session is unchanged.
#[wasm_bindgen]
pub fn session_set_system(id: u32, system_json: &str) -> Result<String, JsValue> {
    let s = session(id)?;
    let sys: OpticalSystem = object(system_json, "system_json")?;
    let mut g = lock(&s);
    let h = g.set_system(sys).map_err(EngineErrorJs::from)?;
    Ok(to_json(
        &serde_json::json!({"hash": h.to_hex(), "library_hash": g.library_hash()}),
    )?)
}

/// `mk_session_get_system_json`.
#[wasm_bindgen]
pub fn session_get_system(id: u32) -> Result<String, JsValue> {
    Ok(to_json(lock(&*session(id)?).system())?)
}

/// `{"hash", "library_hash"}` of the session now.
#[wasm_bindgen]
pub fn session_hashes(id: u32) -> Result<String, JsValue> {
    let s = session(id)?;
    let g = lock(&s);
    Ok(to_json(&serde_json::json!({
        "hash": g.snapshot_hash().to_hex(), "library_hash": g.library_hash()
    }))?)
}

/// `mk_session_list_catalogs`.
#[wasm_bindgen]
pub fn session_list_catalogs(id: u32) -> Result<String, JsValue> {
    Ok(to_json(&lock(&*session(id)?).catalogs())?)
}

/// `mk_session_list_glasses`; empty `catalog` means every catalog.
#[wasm_bindgen]
pub fn session_list_glasses(id: u32, catalog: &str) -> Result<String, JsValue> {
    let s = session(id)?;
    let g = lock(&s)
        .glasses((!catalog.is_empty()).then_some(catalog))
        .map_err(EngineErrorJs::from)?;
    Ok(to_json(&g)?)
}

/// Load a user catalog from the contents of a file named `file_name` (AGF,
/// or GlassCatalog JSON for `.json`).
#[wasm_bindgen]
pub fn session_load_catalog(id: u32, file_name: &str, bytes: &[u8]) -> Result<String, JsValue> {
    let s = session(id)?;
    let summary = lock(&s)
        .load_catalog_bytes(file_name, bytes)
        .map_err(EngineErrorJs::from)?;
    Ok(to_json(&summary)?)
}

/// `mk_session_apply_solves`: returns `{"report", "hash"}`. Runs in the
/// calling thread without holding the session lock.
#[wasm_bindgen]
pub fn session_apply_solves(id: u32, settings_json: &str) -> Result<String, JsValue> {
    let s = session(id)?;
    let settings = optional(settings_json, "settings_json")?;
    let pending = lock(&s)
        .prepare_solves(&settings)
        .map_err(EngineErrorJs::from)?;
    let (sys, report) = pending
        .run(&JobContext::new())
        .map_err(EngineErrorJs::from)?;
    let hash = lock(&s)
        .commit_solves(&pending, sys)
        .map_err(EngineErrorJs::from)?;
    Ok(to_json(
        &serde_json::json!({"report": report, "hash": hash.to_hex()}),
    )?)
}

/// Replace the session with a `.mklight` project given as text. Catalog
/// references by path cannot be read in a browser: they are listed in
/// `open_warnings`. Returns `mk_session_info_json`.
#[wasm_bindgen]
pub fn session_open_project(id: u32, text: &str) -> Result<String, JsValue> {
    let st = state()?;
    let project = Project::from_json_str(text).map_err(EngineErrorJs::from)?;
    let new = st
        .engine
        .session_from_project(project)
        .map_err(EngineErrorJs::from)?;
    let info = new.info();
    lock(&st.sessions).insert(id, Arc::new(Mutex::new(new)));
    Ok(to_json(&info)?)
}

/// The session as `.mklight` project text.
#[wasm_bindgen]
pub fn session_save_project(id: u32) -> Result<String, JsValue> {
    let s = session(id)?;
    let text = lock(&s).save_to_string().map_err(EngineErrorJs::from)?;
    Ok(text)
}

/// `mk_import_prescription`: `.zmx` / `.seq` bytes, glasses resolved against
/// the built-in catalog plus this session's catalogs when `session` is not 0.
#[wasm_bindgen]
pub fn import_prescription_bytes(
    session_id: u32,
    bytes: &[u8],
    format: &str,
) -> Result<String, JsValue> {
    let format: PrescriptionFormat = format.parse().map_err(|e: MkError| arg(e.to_string()))?;
    let library = if session_id == 0 {
        Arc::new(MaterialLibrary::with_builtin())
    } else {
        lock(&*session(session_id)?).library().clone()
    };
    let imported = import_prescription(bytes, format, &library).map_err(EngineErrorJs::from)?;
    Ok(to_json(&imported)?)
}

// ---------------------------------------------------------------------------
// Jobs

/// Submit an analysis request (`mk_session_submit_analysis_json`, docs/FFI.md
/// "Analysis request") against the session's current system; returns the
/// engine job id.
#[wasm_bindgen]
pub fn session_submit(id: u32, request_json: &str) -> Result<f64, JsValue> {
    let s = session(id)?;
    let req: AnalysisRequest = object(request_json, "request_json")?;
    let (engine, spec) = {
        let g = lock(&s);
        (
            g.engine().clone(),
            g.analysis_job(&req).map_err(EngineErrorJs::from)?,
        )
    };
    let job = engine.submit_job(spec).map_err(EngineErrorJs::from)?;
    Ok(job as f64)
}

/// Set the session to `system_json` unless it already holds snapshot
/// `hash`, then submit (one locked step, like the Swift client).
#[wasm_bindgen]
pub fn session_submit_on(
    id: u32,
    hash: &str,
    system_json: &str,
    request_json: &str,
) -> Result<f64, JsValue> {
    let s = session(id)?;
    let req: AnalysisRequest = object(request_json, "request_json")?;
    let (engine, spec) = {
        let mut g = lock(&s);
        if g.snapshot_hash().to_hex() != hash {
            let sys: OpticalSystem = object(system_json, "system_json")?;
            g.set_system(sys).map_err(EngineErrorJs::from)?;
        }
        (
            g.engine().clone(),
            g.analysis_job(&req).map_err(EngineErrorJs::from)?,
        )
    };
    let job = engine.submit_job(spec).map_err(EngineErrorJs::from)?;
    Ok(job as f64)
}

/// `mk_job_status_json`.
#[wasm_bindgen]
pub fn job_status(job: f64) -> Result<String, JsValue> {
    let st = state()?
        .engine
        .job_status(job as u64)
        .map_err(EngineErrorJs::from)?;
    Ok(to_json(&st)?)
}

/// Status of several jobs at once: a JSON array in the order of `jobs`
/// (`null` for an unknown id). The worker's progress poll.
#[wasm_bindgen]
pub fn job_statuses(jobs: &[f64]) -> Result<String, JsValue> {
    let e = &state()?.engine;
    let v: Vec<Value> = jobs
        .iter()
        .map(|&j| {
            e.job_status(j as u64)
                .ok()
                .and_then(|s| serde_json::to_value(s).ok())
                .unwrap_or(Value::Null)
        })
        .collect();
    Ok(to_json(&v)?)
}

#[wasm_bindgen]
pub fn job_cancel(job: f64) -> Result<(), JsValue> {
    state()?
        .engine
        .cancel_job(job as u64)
        .map_err(EngineErrorJs::from)?;
    Ok(())
}

#[wasm_bindgen]
pub fn job_stop(job: f64) -> Result<(), JsValue> {
    state()?
        .engine
        .stop_job(job as u64)
        .map_err(EngineErrorJs::from)?;
    Ok(())
}

#[wasm_bindgen]
pub fn job_release(job: f64) -> Result<(), JsValue> {
    state()?
        .engine
        .release_job(job as u64)
        .map_err(EngineErrorJs::from)?;
    Ok(())
}

/// Take a finished job's result: `{envelope: string, buffers: TypedArray[]}`
/// with the buffers in envelope order. Each array is a fresh (unshared)
/// copy of the engine's buffer, ready to be transferred.
#[wasm_bindgen]
pub fn job_take_result(job: f64) -> Result<JsValue, JsValue> {
    let r = state()?
        .engine
        .take_result(job as u64)
        .map_err(EngineErrorJs::from)?;
    let envelope = to_json(&r.envelope())?;
    let buffers = Array::new();
    for b in &r.buffers {
        buffers.push(&typed_array(b.data()));
    }
    let out = Object::new();
    Reflect::set(&out, &"envelope".into(), &envelope.into())?;
    Reflect::set(&out, &"buffers".into(), &buffers)?;
    Ok(out.into())
}

/// One copy out of the shared wasm memory into a new JS typed array.
fn typed_array(d: &BufferData) -> JsValue {
    match d {
        BufferData::F64(v) => js_sys::Float64Array::from(v.as_slice()).into(),
        BufferData::F32(v) => js_sys::Float32Array::from(v.as_slice()).into(),
        BufferData::I32(v) => js_sys::Int32Array::from(v.as_slice()).into(),
        BufferData::U32(v) => js_sys::Uint32Array::from(v.as_slice()).into(),
        BufferData::I64(v) => js_sys::BigInt64Array::from(v.as_slice()).into(),
        BufferData::U64(v) => js_sys::BigUint64Array::from(v.as_slice()).into(),
        BufferData::U8(v) => js_sys::Uint8Array::from(v.as_slice()).into(),
    }
}

/// Run a request in the calling thread (`mk_session_run_analysis_json`):
/// for tests and quick checks; blocks the engine worker.
#[wasm_bindgen]
pub fn session_run(id: u32, request_json: &str) -> Result<JsValue, JsValue> {
    let s = session(id)?;
    let req: AnalysisRequest = object(request_json, "request_json")?;
    let spec = lock(&s).analysis_job(&req).map_err(EngineErrorJs::from)?;
    let r = (spec.work)(&JobContext::new()).map_err(EngineErrorJs::from)?;
    let envelope = to_json(&r.envelope())?;
    let buffers = Array::new();
    for b in &r.buffers {
        buffers.push(&typed_array(b.data()));
    }
    let out = Object::new();
    Reflect::set(&out, &"envelope".into(), &envelope.into())?;
    Reflect::set(&out, &"buffers".into(), &buffers)?;
    Ok(out.into())
}
