//! `"register_emulation"` in the engine configuration: the CPU-f32
//! emulation is listed (titled as an emulation on the CPU), an explicit
//! request runs on it and `auto` never does.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use serde_json::{Value, json};

use mklight_core::samples;
use mklight_ffi::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn take(p: *mut c_char) -> String {
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { mk_string_free(p) };
    s
}

fn create(config: &str) -> (MKStatus, *mut MKEngine) {
    let config = c(config);
    let mut engine = ptr::null_mut();
    let st = unsafe { mk_engine_create(config.as_ptr(), &mut engine) };
    (st, engine)
}

fn backends(engine: *mut MKEngine) -> Vec<Value> {
    let mut out = ptr::null_mut();
    unsafe {
        assert_eq!(mk_engine_list_backends(engine, &mut out), MK_OK);
        serde_json::from_str::<Value>(&take(out))
            .unwrap()
            .as_array()
            .unwrap()
            .clone()
    }
}

/// The backend a synchronous run of `backend` (or the default) ran on.
fn ran_on(session: *mut MKSession, backend: Value) -> String {
    let mut request = json!({"analysis": "spot_diagram", "settings": {"rings": 3}});
    if !backend.is_null() {
        request["backend"] = backend;
    }
    let request = c(&request.to_string());
    let mut json = ptr::null_mut();
    let mut bufs: *mut MKBuffer = ptr::null_mut();
    let mut n = 0usize;
    unsafe {
        let st =
            mk_session_run_analysis_json(session, request.as_ptr(), &mut json, &mut bufs, &mut n);
        assert_eq!(
            st,
            MK_OK,
            "{}",
            CStr::from_ptr(mk_last_error()).to_string_lossy()
        );
        let result: Value = serde_json::from_str(&take(json)).unwrap();
        mk_buffer_free(bufs, n);
        result["run_info"]["backend"].as_str().unwrap().to_string()
    }
}

#[test]
fn register_emulation_lists_it_and_only_explicit_requests_run_on_it() {
    // A wrong type fails the creation.
    let (st, engine) = create(r#"{"register_emulation": "yes"}"#);
    assert_eq!(st, MK_ERR_INVALID_ARGUMENT);
    assert!(engine.is_null());

    // Off by default.
    let (st, engine) = create(r#"{"register_emulation": false}"#);
    assert_eq!(st, MK_OK);
    assert!(backends(engine).iter().all(|b| b["id"] != "gpu:cpu_f32"));
    unsafe { mk_engine_destroy(engine) };

    let (st, engine) = create(r#"{"register_emulation": true}"#);
    assert_eq!(st, MK_OK);
    let listed = backends(engine);
    let emu = listed.iter().find(|b| b["id"] == "gpu:cpu_f32").unwrap();
    assert_eq!(emu["available"], true);
    let title = emu["title"].as_str().unwrap();
    assert!(
        title.contains("CPU") && title.contains("not a real GPU"),
        "{title}"
    );

    unsafe {
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let system = c(&serde_json::to_string(&samples::double_gauss()).unwrap());
        assert_eq!(
            mk_session_set_system_json(session, system.as_ptr(), ptr::null_mut()),
            MK_OK
        );
        assert_eq!(
            ran_on(session, json!({"kind": "gpu", "name": "cpu_f32"})),
            "gpu:cpu_f32"
        );
        assert_ne!(ran_on(session, json!({"kind": "auto"})), "gpu:cpu_f32");
        assert_ne!(ran_on(session, Value::Null), "gpu:cpu_f32");
        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
}
