//! MKLight C ABI for native applications (see `include/mklight.h` and
//! `docs/FFI.md`, which are the normative description).
//!
//! Conventions:
//! * Every fallible function returns an `MKStatus` (0 = `MK_OK`). On failure
//!   a message is available from `mk_last_error()` on the same thread.
//! * Structured data crosses as UTF-8 JSON. Strings returned through `char**`
//!   are owned by the caller and released with `mk_string_free`.
//! * Bulk results cross as `MKBuffer` arrays released with `mk_buffer_free`.
//! * No panic ever unwinds into C: each entry point catches it and returns
//!   `MK_ERR_PANIC`.
//!
//! Safety: every function trusts the caller to pass pointers that are NULL
//! or valid for the documented access, handles obtained from this library
//! and not yet destroyed, and NUL-terminated UTF-8 strings.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::Value;

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkError, OpticalSystem};
use mklight_engine::backend::Backend;
use mklight_engine::buffer::BufferData;
use mklight_engine::session::{AnalysisRequest, ParamEdit, Session};
use mklight_engine::{
    AnalysisResult, Buffer, Engine, EngineConfig, EngineError, JobEvent, JobState,
};

// ---------------------------------------------------------------------------
// Versioning and status codes (mirrored in include/mklight.h)

pub const MK_ABI_VERSION_MAJOR: u32 = 1;
pub const MK_ABI_VERSION_MINOR: u32 = 2;
/// `(major << 16) | minor`. Major changes break compatibility; minor
/// versions only add functions or JSON fields.
pub const MK_ABI_VERSION: u32 = (MK_ABI_VERSION_MAJOR << 16) | MK_ABI_VERSION_MINOR;

pub type MKStatus = i32;

pub const MK_OK: MKStatus = 0;
pub const MK_ERR_INVALID_ARGUMENT: MKStatus = 1;
pub const MK_ERR_PARSE: MKStatus = 2;
pub const MK_ERR_INVALID_MODEL: MKStatus = 3;
pub const MK_ERR_NOT_FOUND: MKStatus = 4;
pub const MK_ERR_IO: MKStatus = 5;
pub const MK_ERR_UNSUPPORTED: MKStatus = 6;
pub const MK_ERR_CANCELLED: MKStatus = 7;
pub const MK_ERR_TIMEOUT: MKStatus = 8;
pub const MK_ERR_NOT_READY: MKStatus = 9;
pub const MK_ERR_JOB_FAILED: MKStatus = 10;
pub const MK_ERR_COMPUTE: MKStatus = 11;
pub const MK_ERR_PANIC: MKStatus = 12;
pub const MK_ERR_INTERNAL: MKStatus = 13;

/// Every status code with its C name (used by the header consistency test).
pub const STATUS_CODES: &[(&str, MKStatus)] = &[
    ("MK_OK", MK_OK),
    ("MK_ERR_INVALID_ARGUMENT", MK_ERR_INVALID_ARGUMENT),
    ("MK_ERR_PARSE", MK_ERR_PARSE),
    ("MK_ERR_INVALID_MODEL", MK_ERR_INVALID_MODEL),
    ("MK_ERR_NOT_FOUND", MK_ERR_NOT_FOUND),
    ("MK_ERR_IO", MK_ERR_IO),
    ("MK_ERR_UNSUPPORTED", MK_ERR_UNSUPPORTED),
    ("MK_ERR_CANCELLED", MK_ERR_CANCELLED),
    ("MK_ERR_TIMEOUT", MK_ERR_TIMEOUT),
    ("MK_ERR_NOT_READY", MK_ERR_NOT_READY),
    ("MK_ERR_JOB_FAILED", MK_ERR_JOB_FAILED),
    ("MK_ERR_COMPUTE", MK_ERR_COMPUTE),
    ("MK_ERR_PANIC", MK_ERR_PANIC),
    ("MK_ERR_INTERNAL", MK_ERR_INTERNAL),
];

// ---------------------------------------------------------------------------
// Errors

struct FfiError {
    status: MKStatus,
    message: String,
}

impl FfiError {
    fn new(status: MKStatus, message: impl Into<String>) -> Self {
        FfiError {
            status,
            message: message.into(),
        }
    }
    fn arg(message: impl Into<String>) -> Self {
        Self::new(MK_ERR_INVALID_ARGUMENT, message)
    }
}

fn core_status(e: &MkError) -> MKStatus {
    match e {
        MkError::Cancelled => MK_ERR_CANCELLED,
        MkError::InvalidModel(_) => MK_ERR_INVALID_MODEL,
        MkError::UnknownMaterial(_) | MkError::UnknownCoating(_) => MK_ERR_NOT_FOUND,
        MkError::WavelengthOutOfRange { .. } | MkError::InvalidParam(_) => MK_ERR_INVALID_ARGUMENT,
        MkError::Parse(_) => MK_ERR_PARSE,
        MkError::Io(_) => MK_ERR_IO,
        MkError::Unsupported(_) => MK_ERR_UNSUPPORTED,
        MkError::Trace(_) | MkError::Numerical(_) | MkError::Backend(_) => MK_ERR_COMPUTE,
    }
}

impl From<EngineError> for FfiError {
    fn from(e: EngineError) -> Self {
        let status = match &e {
            EngineError::Core(c) => core_status(c),
            EngineError::UnknownAnalysis(_) | EngineError::UnknownJob(_) => MK_ERR_NOT_FOUND,
            EngineError::AnalysisUnavailable { .. } | EngineError::BackendUnavailable(_) => {
                MK_ERR_UNSUPPORTED
            }
            EngineError::InvalidSettings(_) => MK_ERR_INVALID_ARGUMENT,
            EngineError::NotFinished(_) => MK_ERR_NOT_READY,
            EngineError::Timeout(_) => MK_ERR_TIMEOUT,
            EngineError::JobFailed(_) => MK_ERR_JOB_FAILED,
            EngineError::Project(_) => MK_ERR_PARSE,
            EngineError::ShutDown => MK_ERR_INTERNAL,
        };
        FfiError::new(status, e.to_string())
    }
}

impl From<serde_json::Error> for FfiError {
    fn from(e: serde_json::Error) -> Self {
        FfiError::new(MK_ERR_PARSE, format!("JSON: {e}"))
    }
}

type FfiResult<T> = Result<T, FfiError>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn to_cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "\u{FFFD}")).unwrap_or_default()
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = to_cstring(msg));
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Run an entry point: clear the last error, catch panics, map errors.
fn guard(f: impl FnOnce() -> FfiResult<()>) -> MKStatus {
    set_last_error("");
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => MK_OK,
        Ok(Err(e)) => {
            set_last_error(&e.message);
            e.status
        }
        Err(p) => {
            set_last_error(&format!("internal error (panic): {}", panic_message(&*p)));
            MK_ERR_PANIC
        }
    }
}

/// For `void` entry points: swallow panics.
fn guard_void(f: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

// ---------------------------------------------------------------------------
// Argument helpers

unsafe fn in_str<'a>(p: *const c_char, what: &str) -> FfiResult<&'a str> {
    if p.is_null() {
        return Err(FfiError::arg(format!("{what} must not be NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| FfiError::arg(format!("{what} is not valid UTF-8")))
}

unsafe fn in_opt_str<'a>(p: *const c_char, what: &str) -> FfiResult<Option<&'a str>> {
    if p.is_null() {
        Ok(None)
    } else {
        unsafe { in_str(p, what) }.map(Some)
    }
}

/// Parse optional JSON: NULL or an empty string mean `null`.
unsafe fn in_json(p: *const c_char, what: &str) -> FfiResult<Value> {
    match unsafe { in_opt_str(p, what) }? {
        None => Ok(Value::Null),
        Some(s) if s.trim().is_empty() => Ok(Value::Null),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| FfiError::new(MK_ERR_PARSE, format!("{what}: invalid JSON: {e}"))),
    }
}

unsafe fn out_ptr<'a, T>(p: *mut T, what: &str) -> FfiResult<&'a mut T> {
    unsafe { p.as_mut() }.ok_or_else(|| FfiError::arg(format!("{what} must not be NULL")))
}

/// Store a newly allocated C string in `*out` (if `out` is not NULL).
unsafe fn put_string(out: *mut *mut c_char, s: &str) {
    if let Some(o) = unsafe { out.as_mut() } {
        *o = to_cstring(s).into_raw();
    }
}

unsafe fn put_json(out: *mut *mut c_char, v: &impl serde::Serialize) -> FfiResult<()> {
    let s = serde_json::to_string(v)?;
    unsafe { put_string(out, &s) };
    Ok(())
}

/// Clear a `char**` out-parameter so it is NULL on failure.
unsafe fn clear_out<T>(out: *mut *mut T) {
    if let Some(o) = unsafe { out.as_mut() } {
        *o = ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Handles

/// Opaque engine handle (`MKEngine*`).
pub struct MKEngine {
    engine: Engine,
}

/// Opaque session handle (`MKSession*`). Calls on one session are
/// serialised by an internal lock.
pub struct MKSession {
    session: Mutex<Session>,
}

impl MKSession {
    fn lock(&self) -> MutexGuard<'_, Session> {
        // Session edits are all-or-nothing, so the state behind a poisoned
        // lock is still consistent.
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }
}

unsafe fn engine_ref<'a>(p: *const MKEngine) -> FfiResult<&'a MKEngine> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::arg("engine must not be NULL"))
}

unsafe fn session_ref<'a>(p: *const MKSession) -> FfiResult<&'a MKSession> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::arg("session must not be NULL"))
}

// ---------------------------------------------------------------------------
// Library-level functions

#[unsafe(no_mangle)]
pub extern "C" fn mk_abi_version() -> u32 {
    MK_ABI_VERSION
}

/// Static NUL-terminated engine version string (do not free).
#[unsafe(no_mangle)]
pub extern "C" fn mk_engine_version() -> *const c_char {
    static V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr().cast()
}

/// Message of the last failed call on this thread ("" if it succeeded).
/// Valid until the next `mk_*` call on the same thread. Never NULL.
#[unsafe(no_mangle)]
pub extern "C" fn mk_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_string_free(s: *mut c_char) {
    if !s.is_null() {
        guard_void(|| drop(unsafe { CString::from_raw(s) }));
    }
}

// ---------------------------------------------------------------------------
// Buffers

pub const MK_DTYPE_F64: i32 = 1;
pub const MK_DTYPE_F32: i32 = 2;
pub const MK_DTYPE_I32: i32 = 3;
pub const MK_DTYPE_U32: i32 = 4;
pub const MK_DTYPE_I64: i32 = 5;
pub const MK_DTYPE_U64: i32 = 6;
pub const MK_DTYPE_U8: i32 = 7;

/// A typed, row-major array (`MKBuffer`). All pointers are owned by the
/// array returned from `mk_job_take_result` and freed by `mk_buffer_free`.
#[repr(C)]
pub struct MKBuffer {
    pub name: *const c_char,
    /// `len` elements of `dtype`, suitably aligned. Non-NULL (dangling but
    /// aligned when `len == 0`).
    pub data: *const c_void,
    /// Number of elements (not bytes).
    pub len: usize,
    pub dtype: i32,
    pub ndim: u32,
    /// `ndim` extents; their product is `len`.
    pub shape: *const usize,
}

fn leak_slice<T>(v: Vec<T>) -> *const c_void {
    Box::into_raw(v.into_boxed_slice()) as *mut T as *const c_void
}

unsafe fn free_slice<T>(p: *const c_void, len: usize) {
    drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(p as *mut T, len)) });
}

fn export_buffer(b: Buffer) -> MKBuffer {
    let dtype = b.dtype() as i32;
    let len = b.data.len();
    let data = match b.data {
        BufferData::F64(v) => leak_slice(v),
        BufferData::F32(v) => leak_slice(v),
        BufferData::I32(v) => leak_slice(v),
        BufferData::U32(v) => leak_slice(v),
        BufferData::I64(v) => leak_slice(v),
        BufferData::U64(v) => leak_slice(v),
        BufferData::U8(v) => leak_slice(v),
    };
    let ndim = b.shape.len() as u32;
    MKBuffer {
        name: to_cstring(&b.name).into_raw(),
        data,
        len,
        dtype,
        ndim,
        shape: leak_slice(b.shape) as *const usize,
    }
}

unsafe fn free_buffer(b: &MKBuffer) {
    unsafe {
        if !b.name.is_null() {
            drop(CString::from_raw(b.name as *mut c_char));
        }
        if !b.shape.is_null() {
            free_slice::<usize>(b.shape as *const c_void, b.ndim as usize);
        }
        if !b.data.is_null() {
            match b.dtype {
                MK_DTYPE_F64 => free_slice::<f64>(b.data, b.len),
                MK_DTYPE_F32 => free_slice::<f32>(b.data, b.len),
                MK_DTYPE_I32 => free_slice::<i32>(b.data, b.len),
                MK_DTYPE_U32 => free_slice::<u32>(b.data, b.len),
                MK_DTYPE_I64 => free_slice::<i64>(b.data, b.len),
                MK_DTYPE_U64 => free_slice::<u64>(b.data, b.len),
                MK_DTYPE_U8 => free_slice::<u8>(b.data, b.len),
                _ => {} // corrupted: leak rather than free with the wrong layout
            }
        }
    }
}

/// Free an array of `count` buffers returned by `mk_job_take_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_buffer_free(buffers: *mut MKBuffer, count: usize) {
    if buffers.is_null() {
        return;
    }
    guard_void(|| unsafe {
        let all = Box::from_raw(ptr::slice_from_raw_parts_mut(buffers, count));
        for b in all.iter() {
            free_buffer(b);
        }
    });
}

// ---------------------------------------------------------------------------
// Engine

/// Create an engine. `config_json` may be NULL (defaults) or an
/// `EngineConfig` object (see docs/FFI.md).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_create(
    config_json: *const c_char,
    out_engine: *mut *mut MKEngine,
) -> MKStatus {
    unsafe { clear_out(out_engine) };
    guard(|| {
        let out = unsafe { out_ptr(out_engine, "out_engine") }?;
        let cfg = match unsafe { in_json(config_json, "config_json") }? {
            Value::Null => EngineConfig::default(),
            v => serde_json::from_value::<EngineConfig>(v)?,
        };
        *out = Box::into_raw(Box::new(MKEngine {
            engine: Engine::new(cfg),
        }));
        Ok(())
    })
}

/// Destroy an engine handle. Clears the job callback first (waiting for any
/// call in progress on another thread). Worker threads stop once every
/// session created from this engine is destroyed as well.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_destroy(engine: *mut MKEngine) {
    if engine.is_null() {
        return;
    }
    guard_void(|| {
        let e = unsafe { Box::from_raw(engine) };
        e.engine.set_event_callback(None);
        drop(e);
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_list_analyses(
    engine: *const MKEngine,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, &e.engine.analyses()) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_list_backends(
    engine: *const MKEngine,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, &e.engine.backends()) }
    })
}

/// C job-event callback. `stage` is valid only during the call.
pub type MKJobCallback = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        job_id: u64,
        state: i32,
        progress: f64,
        stage: *const c_char,
    ),
>;

struct UserData(*mut c_void);
// The caller promises (docs/FFI.md) that `user_data` may be used from any thread.
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// Install (`cb != NULL`) or clear (`cb == NULL`) the job-event callback.
/// It is invoked on the engine's single event thread, one event at a time,
/// never on the thread that submitted or cancelled the job, and with no
/// engine or session lock held; it may call any `mk_*` function except
/// destroying this engine. When this returns, no other thread is still
/// inside the previous callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_set_job_callback(
    engine: *mut MKEngine,
    cb: MKJobCallback,
    user_data: *mut c_void,
) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        match cb {
            None => e.engine.set_event_callback(None),
            Some(f) => {
                let ud = UserData(user_data);
                e.engine
                    .set_event_callback(Some(Arc::new(move |ev: &JobEvent| {
                        let stage = to_cstring(&ev.stage);
                        let ud = &ud;
                        unsafe { f(ud.0, ev.id, ev.state as i32, ev.progress, stage.as_ptr()) };
                    })));
            }
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Sessions

fn new_session_handle(s: Session) -> *mut MKSession {
    Box::into_raw(Box::new(MKSession {
        session: Mutex::new(s),
    }))
}

/// A session holding a new untitled project.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_new(
    engine: *const MKEngine,
    out_session: *mut *mut MKSession,
) -> MKStatus {
    unsafe { clear_out(out_session) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let out = unsafe { out_ptr(out_session, "out_session") }?;
        *out = new_session_handle(e.engine.new_session());
        Ok(())
    })
}

/// Open a `.mklight` file. Non-fatal problems (missing user catalogs,
/// migration notes) are listed in `mk_session_info_json` → `open_warnings`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_open(
    engine: *const MKEngine,
    path: *const c_char,
    out_session: *mut *mut MKSession,
) -> MKStatus {
    unsafe { clear_out(out_session) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let path = unsafe { in_str(path, "path") }?;
        let out = unsafe { out_ptr(out_session, "out_session") }?;
        let s = e.engine.open_session(Path::new(path))?;
        *out = new_session_handle(s);
        Ok(())
    })
}

/// Destroy a session. Its running jobs keep running to completion (their
/// results stay retrievable through the engine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_destroy(session: *mut MKSession) {
    if !session.is_null() {
        guard_void(|| drop(unsafe { Box::from_raw(session) }));
    }
}

/// Save atomically to `path`, or to the session's current path if NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_save(session: *mut MKSession, path: *const c_char) -> MKStatus {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let path = unsafe { in_opt_str(path, "path") }?;
        s.lock().save(path.map(Path::new))?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_info_json(
    session: *const MKSession,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, &s.lock().info()) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_get_project_json(
    session: *const MKSession,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, s.lock().project()) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_get_system_json(
    session: *const MKSession,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, s.lock().system()) }
    })
}

/// Replace the whole optical system. On success `*out_hash` (if not NULL)
/// receives the new snapshot hash; on failure the session is unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_set_system_json(
    session: *mut MKSession,
    system_json: *const c_char,
    out_hash: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_hash) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let text = unsafe { in_str(system_json, "system_json") }?;
        let sys: OpticalSystem = serde_json::from_str(text)?;
        let h = s.lock().set_system(sys)?;
        unsafe { put_string(out_hash, &h.to_hex()) };
        Ok(())
    })
}

/// Apply one `{"path": ParamPath, "value": number}` edit or an array of
/// them, atomically.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_apply_param_edit(
    session: *mut MKSession,
    edits_json: *const c_char,
    out_hash: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_hash) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let v = unsafe { in_json(edits_json, "edits_json") }?;
        let edits: Vec<ParamEdit> = match v {
            Value::Array(_) => serde_json::from_value(v)?,
            Value::Object(_) => vec![serde_json::from_value(v)?],
            _ => return Err(FfiError::arg("edits_json must be an object or an array")),
        };
        let h = s.lock().apply_edits(&edits)?;
        unsafe { put_string(out_hash, &h.to_hex()) };
        Ok(())
    })
}

/// Replace a project section: "views", "merit", "variables", "tolerances"
/// or "metadata".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_set_project_section_json(
    session: *mut MKSession,
    section: *const c_char,
    json: *const c_char,
) -> MKStatus {
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let section = unsafe { in_str(section, "section") }?;
        let v = unsafe { in_json(json, "json") }?;
        s.lock().set_project_section(section, v)?;
        Ok(())
    })
}

/// 64 hex characters of the current snapshot's BLAKE3 hash.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_current_snapshot_hash(
    session: *const MKSession,
    out_hash: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_hash) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_hash, "out_hash") }?;
        let h = s.lock().snapshot_hash().to_hex();
        unsafe { put_string(out_hash, &h) };
        Ok(())
    })
}

/// 64 hex characters of the BLAKE3 content hash of the session's glass
/// library. A result is current while both its `run_info.snapshot_hash` and
/// its `library_hash` equal the session's.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_current_library_hash(
    session: *const MKSession,
    out_hash: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_hash) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_hash, "out_hash") }?;
        let h = s.lock().library_hash().to_string();
        unsafe { put_string(out_hash, &h) };
        Ok(())
    })
}

/// Load a user glass catalog (AGF, or JSON when the extension is `.json`).
/// `*out_json` (if not NULL) receives the catalog summary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_load_catalog(
    session: *mut MKSession,
    path: *const c_char,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let path = unsafe { in_str(path, "path") }?;
        let summary = s.lock().load_catalog_file(Path::new(path))?;
        unsafe { put_json(out_json, &summary) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_list_catalogs(
    session: *const MKSession,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        unsafe { put_json(out_json, &s.lock().catalogs()) }
    })
}

/// Glasses of `catalog` (case-insensitive), or of every catalog if NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_list_glasses(
    session: *const MKSession,
    catalog: *const c_char,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let catalog = unsafe { in_opt_str(catalog, "catalog") }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let g = s.lock().glasses(catalog)?;
        unsafe { put_json(out_json, &g) }
    })
}

fn submit(s: &MKSession, req: &AnalysisRequest, out_job: &mut u64) -> FfiResult<()> {
    // Build the job under the session lock, queue it after releasing it.
    let (engine, spec) = {
        let g = s.lock();
        (g.engine().clone(), g.analysis_job(req)?)
    };
    *out_job = engine.submit_job(spec)?;
    Ok(())
}

/// Queue an analysis on the current snapshot. `settings_json` and
/// `view_key` may be NULL. A view key supersedes (cancels) the previous job
/// with the same key in this session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_submit_analysis(
    session: *const MKSession,
    analysis: *const c_char,
    settings_json: *const c_char,
    view_key: *const c_char,
    out_job: *mut u64,
) -> MKStatus {
    if let Some(o) = unsafe { out_job.as_mut() } {
        *o = 0;
    }
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let out = unsafe { out_ptr(out_job, "out_job") }?;
        let req = AnalysisRequest {
            analysis: unsafe { in_str(analysis, "analysis") }?.to_string(),
            settings: unsafe { in_json(settings_json, "settings_json") }?,
            view_key: unsafe { in_opt_str(view_key, "view_key") }?.map(str::to_string),
            ..Default::default()
        };
        submit(s, &req, out)
    })
}

/// Queue an analysis from a full request object
/// `{"analysis", "settings", "view_key", "backend", "allow_backend_fallback"}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_submit_analysis_json(
    session: *const MKSession,
    request_json: *const c_char,
    out_job: *mut u64,
) -> MKStatus {
    if let Some(o) = unsafe { out_job.as_mut() } {
        *o = 0;
    }
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let out = unsafe { out_ptr(out_job, "out_job") }?;
        let req: AnalysisRequest =
            serde_json::from_str(unsafe { in_str(request_json, "request_json") }?)?;
        submit(s, &req, out)
    })
}

// ---------------------------------------------------------------------------
// Jobs

/// State (`MKJobState`) and progress (0…1) of a job.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_status(
    engine: *const MKEngine,
    job: u64,
    out_state: *mut i32,
    out_progress: *mut f64,
) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let st = e.engine.job_status(job)?;
        if let Some(o) = unsafe { out_state.as_mut() } {
            *o = st.state as i32;
        }
        if let Some(o) = unsafe { out_progress.as_mut() } {
            *o = st.progress;
        }
        Ok(())
    })
}

/// Current stage text of a job.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_stage(
    engine: *const MKEngine,
    job: u64,
    out_stage: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_stage) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        unsafe { out_ptr(out_stage, "out_stage") }?;
        let st = e.engine.job_status(job)?;
        unsafe { put_string(out_stage, &st.stage) };
        Ok(())
    })
}

/// Full job status as JSON (state, progress, stage, error, superseded_by, …).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_status_json(
    engine: *const MKEngine,
    job: u64,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let st = e.engine.job_status(job)?;
        unsafe { put_json(out_json, &st) }
    })
}

/// Stop optimisation/tolerancing and retain the best-so-far result.
/// Unsupported for other active jobs; no-op for finished jobs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_stop(engine: *const MKEngine, job: u64) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        e.engine.stop_job(job)?;
        Ok(())
    })
}

/// Request cancellation (no-op for finished jobs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_cancel(engine: *const MKEngine, job: u64) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        e.engine.cancel_job(job)?;
        Ok(())
    })
}

/// Wait for a job to finish. `timeout_ms < 0` waits forever. Returns
/// `MK_OK` with the terminal state, or `MK_ERR_TIMEOUT` with the current
/// state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_wait(
    engine: *const MKEngine,
    job: u64,
    timeout_ms: i64,
    out_state: *mut i32,
) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let timeout = (timeout_ms >= 0).then(|| Duration::from_millis(timeout_ms as u64));
        let r = e.engine.wait_job(job, timeout);
        let state = match &r {
            Ok(s) => *s,
            Err(_) => e
                .engine
                .job_status(job)
                .map(|s| s.state)
                .unwrap_or(JobState::Queued),
        };
        if let Some(o) = unsafe { out_state.as_mut() } {
            *o = state as i32;
        }
        r.map(|_| ()).map_err(FfiError::from)
    })
}

/// Take a finished job's result and forget the job.
///
/// * Done: `MK_OK`; `*out_json` receives `{"run_info", "result", "buffers"}`
///   and `*out_buffers`/`*out_count` the buffers (NULL/0 when there are none).
/// * Failed: `MK_ERR_JOB_FAILED` (message in `mk_last_error`).
/// * Cancelled: `MK_ERR_CANCELLED`.
/// * Still queued/running: `MK_ERR_NOT_READY` (the job is kept).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_take_result(
    engine: *const MKEngine,
    job: u64,
    out_json: *mut *mut c_char,
    out_buffers: *mut *mut MKBuffer,
    out_count: *mut usize,
) -> MKStatus {
    unsafe {
        clear_out(out_json);
        clear_out(out_buffers);
        if let Some(c) = out_count.as_mut() {
            *c = 0;
        }
    }
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let out_b = unsafe { out_ptr(out_buffers, "out_buffers") }?;
        let out_c = unsafe { out_ptr(out_count, "out_count") }?;
        let state = e.engine.job_status(job)?.state;
        let r = match e.engine.take_result(job) {
            Ok(r) => r,
            Err(err @ (EngineError::NotFinished(_) | EngineError::UnknownJob(_))) => {
                return Err(err.into());
            }
            Err(err) if state == JobState::Cancelled || err.is_cancelled() => {
                return Err(FfiError::new(MK_ERR_CANCELLED, err.to_string()));
            }
            Err(err) => return Err(FfiError::new(MK_ERR_JOB_FAILED, err.to_string())),
        };
        unsafe { export_result(r, out_json, out_b, out_c) }
    })
}

/// Hand a result to C: the envelope JSON and the buffer array.
unsafe fn export_result(
    r: AnalysisResult,
    out_json: *mut *mut c_char,
    out_b: &mut *mut MKBuffer,
    out_c: &mut usize,
) -> FfiResult<()> {
    let json = serde_json::to_string(&r.envelope())?;
    let bufs: Vec<MKBuffer> = r.buffers.into_iter().map(export_buffer).collect();
    *out_c = bufs.len();
    *out_b = if bufs.is_empty() {
        ptr::null_mut()
    } else {
        Box::into_raw(bufs.into_boxed_slice()) as *mut MKBuffer
    };
    unsafe { put_string(out_json, &json) };
    Ok(())
}

// ---------------------------------------------------------------------------
// ABI 1.1: analysis descriptions, synchronous runs, backends, validation,
// solves, autosave, migration, parameter reads

/// Description of one analysis (as one element of `mk_engine_list_analyses`:
/// settings and result schemas included).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_analysis_info(
    engine: *const MKEngine,
    name: *const c_char,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let name = unsafe { in_str(name, "name") }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let a = e.engine.analysis(name)?;
        unsafe { put_json(out_json, &a.info()) }
    })
}

/// Make `backend_json` (`{"kind": "cpu_reference"}`, `{"kind": "auto"}`,
/// `{"kind": "gpu", "name": "metal"}`) the default for requests that name
/// no backend. An unavailable explicit backend is refused.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_set_default_backend(
    engine: *const MKEngine,
    backend_json: *const c_char,
) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let v = unsafe { in_json(backend_json, "backend_json") }?;
        let b: Backend = serde_json::from_value(v)
            .map_err(|err| FfiError::arg(format!("backend_json: {err}")))?;
        e.engine.set_default_backend(b)?;
        Ok(())
    })
}

/// Migrate a project file in place (`backup != 0` keeps the original as
/// `<file>.v<N>.bak`; `dry_run != 0` writes nothing). `*out_json`
/// (nullable) receives `{path, from_version, to_version, notes, migrated,
/// backup, dry_run}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_engine_migrate_project(
    engine: *const MKEngine,
    path: *const c_char,
    backup: i32,
    dry_run: i32,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        let path = unsafe { in_str(path, "path") }?;
        let r = e
            .engine
            .migrate_file(Path::new(path), backup != 0, dry_run != 0)?;
        unsafe { put_json(out_json, &r) }
    })
}

/// Run `req` in the calling thread on the session's current snapshot,
/// holding the session lock only while the job is built (as `submit`), so
/// other calls on the session proceed meanwhile.
fn run_unlocked(s: &MKSession, req: &AnalysisRequest) -> FfiResult<AnalysisResult> {
    let spec = s.lock().analysis_job(req)?;
    Ok((spec.work)(&JobContext::new())?)
}

/// Run an analysis on the current snapshot in the calling thread (blocking;
/// no job, no callback) and return the result like `mk_job_take_result`.
/// `request_json` is an analysis request as for
/// `mk_session_submit_analysis_json` (`view_key` is ignored). Errors of the
/// run are returned directly (e.g. `MK_ERR_NOT_FOUND` for an unknown glass,
/// `MK_ERR_COMPUTE` for a trace failure).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_run_analysis_json(
    session: *const MKSession,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
    out_buffers: *mut *mut MKBuffer,
    out_count: *mut usize,
) -> MKStatus {
    unsafe {
        clear_out(out_json);
        clear_out(out_buffers);
        if let Some(c) = out_count.as_mut() {
            *c = 0;
        }
    }
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let text = unsafe { in_str(request_json, "request_json") }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let out_b = unsafe { out_ptr(out_buffers, "out_buffers") }?;
        let out_c = unsafe { out_ptr(out_count, "out_count") }?;
        let req: AnalysisRequest = serde_json::from_str(text)?;
        let r = run_unlocked(s, &req)?;
        unsafe { export_result(r, out_json, out_b, out_c) }
    })
}

/// The `validate` report of the current snapshot (`{ok, errors, warnings,
/// issues}`, docs/RESULTS.md), computed in the calling thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_validate_json(
    session: *const MKSession,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let r = run_unlocked(s, &AnalysisRequest::new("validate", Value::Null))?;
        unsafe { put_json(out_json, &r.json) }
    })
}

/// Apply the system's solves and commit the solved system as a new
/// snapshot (docs/SOLVES.md). `settings_json` (nullable): the `solves`
/// settings (`configuration` must be null). `*out_report` (nullable): the
/// solve report (the `solves` result shape); `*out_hash` (nullable): the
/// new snapshot hash. On failure the session is unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_apply_solves(
    session: *mut MKSession,
    settings_json: *const c_char,
    out_report: *mut *mut c_char,
    out_hash: *mut *mut c_char,
) -> MKStatus {
    unsafe {
        clear_out(out_report);
        clear_out(out_hash);
    }
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let settings = unsafe { in_json(settings_json, "settings_json") }?;
        // Solve without holding the session lock; commit only if the
        // session did not change meanwhile.
        let pending = s.lock().prepare_solves(&settings)?;
        let (sys, report) = pending.run(&JobContext::new())?;
        let hash = s.lock().commit_solves(&pending, sys)?;
        unsafe {
            put_json(out_report, &report)?;
            put_string(out_hash, &hash.to_hex());
        }
        Ok(())
    })
}

/// Write the current project to `path`, or to `<project file>.autosave`
/// when NULL, without changing the session (path, dirty flag). An untitled
/// session needs a path. `*out_path` (nullable) receives the path written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_autosave(
    session: *const MKSession,
    path: *const c_char,
    out_path: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_path) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let path = unsafe { in_opt_str(path, "path") }?;
        let written = s.lock().autosave(path.map(Path::new))?;
        unsafe { put_string(out_path, &written.display().to_string()) };
        Ok(())
    })
}

/// Current value of a parameter: `{"value": number | "inf" | "-inf" |
/// null}` (null: the parameter has no value, e.g. an automatic
/// semi-diameter). `path_json` is a ParamPath as in
/// `mk_session_apply_param_edit`; a path this system does not have is
/// `MK_ERR_INVALID_ARGUMENT`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_session_get_param(
    session: *const MKSession,
    path_json: *const c_char,
    out_json: *mut *mut c_char,
) -> MKStatus {
    unsafe { clear_out(out_json) };
    guard(|| {
        let s = unsafe { session_ref(session) }?;
        let text = unsafe { in_str(path_json, "path_json") }?;
        unsafe { out_ptr(out_json, "out_json") }?;
        let p: ParamPath =
            serde_json::from_str(text).map_err(|e| FfiError::arg(format!("path_json: {e}")))?;
        let v = s.lock().param_value(&p)?;
        let value = match v {
            None => Value::Null,
            Some(x) if x.is_infinite() => Value::from(if x > 0.0 { "inf" } else { "-inf" }),
            Some(x) => serde_json::json!(x),
        };
        unsafe { put_json(out_json, &serde_json::json!({ "value": value })) }
    })
}

/// Cancel (if needed) and forget a job without taking its result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mk_job_release(engine: *const MKEngine, job: u64) -> MKStatus {
    guard(|| {
        let e = unsafe { engine_ref(engine) }?;
        e.engine.release_job(job)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        unsafe { CStr::from_ptr(mk_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn panics_are_caught() {
        let st = guard(|| panic!("boom"));
        assert_eq!(st, MK_ERR_PANIC);
        assert!(last_error().contains("boom"));
        assert_eq!(guard(|| Ok(())), MK_OK);
        assert_eq!(last_error(), "");
    }

    #[test]
    fn header_matches_rust_constants() {
        let h = include_str!("../include/mklight.h");
        let define = |name: &str| -> i64 {
            let line = h
                .lines()
                .find(|l| {
                    let mut it = l.split_whitespace();
                    it.next() == Some("#define") && it.next() == Some(name)
                })
                .unwrap_or_else(|| panic!("{name} missing from header"));
            let v = line.split_whitespace().nth(2).unwrap();
            v.trim_matches(|c| c == '(' || c == ')').parse().unwrap()
        };
        assert_eq!(define("MK_ABI_VERSION_MAJOR"), MK_ABI_VERSION_MAJOR as i64);
        assert_eq!(define("MK_ABI_VERSION_MINOR"), MK_ABI_VERSION_MINOR as i64);
        for (name, code) in STATUS_CODES {
            assert_eq!(define(name), *code as i64, "{name}");
        }
        for (name, code) in [
            ("MK_DTYPE_F64", MK_DTYPE_F64),
            ("MK_DTYPE_F32", MK_DTYPE_F32),
            ("MK_DTYPE_I32", MK_DTYPE_I32),
            ("MK_DTYPE_U32", MK_DTYPE_U32),
            ("MK_DTYPE_I64", MK_DTYPE_I64),
            ("MK_DTYPE_U64", MK_DTYPE_U64),
            ("MK_DTYPE_U8", MK_DTYPE_U8),
        ] {
            assert_eq!(define(name), code as i64, "{name}");
        }
        for (name, st) in [
            ("MK_JOB_QUEUED", JobState::Queued),
            ("MK_JOB_RUNNING", JobState::Running),
            ("MK_JOB_DONE", JobState::Done),
            ("MK_JOB_FAILED", JobState::Failed),
            ("MK_JOB_CANCELLED", JobState::Cancelled),
        ] {
            assert_eq!(define(name), st as i64, "{name}");
        }
        // Rust dtypes agree with the C codes.
        use mklight_engine::DType;
        assert_eq!(DType::F64 as i32, MK_DTYPE_F64);
        assert_eq!(DType::U8 as i32, MK_DTYPE_U8);
        // Every exported function is declared in the header.
        let src = include_str!("lib.rs");
        for l in src.lines() {
            if let Some(rest) = l
                .trim_start()
                .strip_prefix("pub unsafe extern \"C\" fn ")
                .or_else(|| l.trim_start().strip_prefix("pub extern \"C\" fn "))
            {
                let name = rest.split('(').next().unwrap();
                assert!(
                    h.contains(&format!("{name}(")),
                    "{name} not declared in mklight.h"
                );
            }
        }
    }

    #[test]
    fn buffers_roundtrip_and_free() {
        let b = export_buffer(Buffer::new(
            "x",
            vec![2, 2],
            BufferData::U32(vec![1, 2, 3, 4]),
        ));
        assert_eq!((b.len, b.dtype, b.ndim), (4, MK_DTYPE_U32, 2));
        let v = unsafe { std::slice::from_raw_parts(b.data as *const u32, b.len) };
        assert_eq!(v, &[1, 2, 3, 4]);
        let arr =
            Box::into_raw(vec![b, export_buffer(Buffer::f64_1d("e", vec![]))].into_boxed_slice())
                as *mut MKBuffer;
        unsafe { mk_buffer_free(arr, 2) };
    }

    #[test]
    fn null_arguments_are_rejected() {
        unsafe {
            assert_eq!(
                mk_engine_create(ptr::null(), ptr::null_mut()),
                MK_ERR_INVALID_ARGUMENT
            );
            let mut out: *mut c_char = ptr::null_mut();
            assert_eq!(
                mk_engine_list_analyses(ptr::null(), &mut out),
                MK_ERR_INVALID_ARGUMENT
            );
            assert!(out.is_null());
            assert!(last_error().contains("engine"));
            mk_string_free(ptr::null_mut());
            mk_buffer_free(ptr::null_mut(), 3);
            mk_engine_destroy(ptr::null_mut());
            mk_session_destroy(ptr::null_mut());
        }
    }
}
