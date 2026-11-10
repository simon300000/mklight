//! Errors over the C ABI: a failed job returns the status of its own
//! error, JSON input follows one rule (not JSON: `MK_ERR_PARSE`; another
//! shape: `MK_ERR_INVALID_ARGUMENT`), unknown keys are errors naming the
//! key, and the stop-capable analyses are those the listing marks.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use serde_json::{Value, json};

use mklight_core::{MaterialRef, OpticalSystem, samples};
use mklight_ffi::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Take ownership of a returned string.
unsafe fn take(p: *mut c_char) -> String {
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { mk_string_free(p) };
    s
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(mk_last_error()) }
        .to_string_lossy()
        .into_owned()
}

struct Handles {
    engine: *mut MKEngine,
    session: *mut MKSession,
}

impl Drop for Handles {
    fn drop(&mut self) {
        unsafe {
            mk_session_destroy(self.session);
            mk_engine_destroy(self.engine);
        }
    }
}

fn open(sys: &OpticalSystem) -> Handles {
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let text = c(&serde_json::to_string(sys).unwrap());
        assert_eq!(
            mk_session_set_system_json(session, text.as_ptr(), ptr::null_mut()),
            MK_OK,
            "{}",
            last_error()
        );
        Handles { engine, session }
    }
}

/// Status and message of a synchronous run of `request`.
fn run_status(h: &Handles, request: &str) -> (MKStatus, String) {
    unsafe {
        let req = c(request);
        let mut json = ptr::null_mut();
        let mut bufs: *mut MKBuffer = ptr::null_mut();
        let mut n = 0usize;
        let st =
            mk_session_run_analysis_json(h.session, req.as_ptr(), &mut json, &mut bufs, &mut n);
        let message = last_error();
        if st == MK_OK {
            mk_string_free(json);
            mk_buffer_free(bufs, n);
        } else {
            assert!(json.is_null() && bufs.is_null() && n == 0);
        }
        (st, message)
    }
}

/// Status and message of submitting `request`; a job that was queued is
/// released.
fn submit_status(h: &Handles, request: &str) -> (MKStatus, String) {
    unsafe {
        let req = c(request);
        let mut job = 0u64;
        let st = mk_session_submit_analysis_json(h.session, req.as_ptr(), &mut job);
        let message = last_error();
        if st == MK_OK {
            assert_eq!(mk_job_release(h.engine, job), MK_OK);
        } else {
            assert_eq!(job, 0);
        }
        (st, message)
    }
}

/// Submit `request`, wait, and take the result: the job's terminal state
/// with the status and message of `mk_job_take_result`.
fn job_outcome(h: &Handles, request: &str) -> (i32, MKStatus, String) {
    unsafe {
        let req = c(request);
        let mut job = 0u64;
        let st = mk_session_submit_analysis_json(h.session, req.as_ptr(), &mut job);
        assert_eq!(st, MK_OK, "{request}: {}", last_error());
        let mut state = -1;
        assert_eq!(mk_job_wait(h.engine, job, 60_000, &mut state), MK_OK);
        let mut json = ptr::null_mut();
        let mut bufs: *mut MKBuffer = ptr::null_mut();
        let mut n = 0usize;
        let st = mk_job_take_result(h.engine, job, &mut json, &mut bufs, &mut n);
        let message = last_error();
        if st == MK_OK {
            mk_string_free(json);
            mk_buffer_free(bufs, n);
        } else {
            assert!(json.is_null() && bufs.is_null() && n == 0);
        }
        // Taken or failed, the job is forgotten.
        assert_eq!(
            mk_job_status(h.engine, job, ptr::null_mut(), ptr::null_mut()),
            MK_ERR_NOT_FOUND
        );
        (state, st, message)
    }
}

#[test]
fn a_failed_job_returns_the_status_of_its_error() {
    // An unknown glass is found when the job resolves the system.
    let mut unknown_glass = samples::singlet();
    unknown_glass.surfaces[1].material = MaterialRef::Catalog {
        catalog: None,
        name: "NOPE-42".into(),
        offset: Default::default(),
    };
    // A wavelength at which the glass has no physical index.
    let mut far_infrared = samples::singlet();
    far_infrared.wavelengths.wavelengths[0].um = 10.0;
    for (sys, status, text) in [
        (unknown_glass, MK_ERR_NOT_FOUND, "NOPE-42"),
        (far_infrared, MK_ERR_INVALID_MODEL, "refractive index"),
    ] {
        let h = open(&sys);
        let request = r#"{"analysis": "system_summary"}"#;
        let (state, st, message) = job_outcome(&h, request);
        assert_eq!(state, 3, "MK_JOB_FAILED");
        assert_eq!(st, status, "{message}");
        assert!(message.contains(text), "{message}");
        // The synchronous run of the same request fails the same way.
        assert_eq!(run_status(&h, request), (status, message));
    }
    // A job that succeeds, and one that is cancelled.
    let h = open(&samples::singlet());
    let (state, st, _) = job_outcome(&h, r#"{"analysis": "system_summary"}"#);
    assert_eq!((state, st), (2, MK_OK));
    unsafe {
        let req = c(
            r#"{"analysis": "rng_stream", "settings": {"count": 100000, "chunk": 100, "delay_ms": 20}}"#,
        );
        let mut job = 0u64;
        assert_eq!(
            mk_session_submit_analysis_json(h.session, req.as_ptr(), &mut job),
            MK_OK
        );
        assert_eq!(mk_job_cancel(h.engine, job), MK_OK);
        let mut state = -1;
        assert_eq!(mk_job_wait(h.engine, job, 60_000, &mut state), MK_OK);
        let mut json = ptr::null_mut();
        let mut bufs: *mut MKBuffer = ptr::null_mut();
        let mut n = 0usize;
        assert_eq!(
            mk_job_take_result(h.engine, job, &mut json, &mut bufs, &mut n),
            MK_ERR_CANCELLED
        );
        assert_eq!(state, 4, "MK_JOB_CANCELLED");
    }
}

#[test]
fn json_input_is_parse_or_invalid_argument_by_one_rule() {
    let h = open(&samples::singlet());
    // Each entry point with JSON input, called with `text`.
    type Call = Box<dyn Fn(&Handles, &CString) -> MKStatus>;
    let calls: Vec<(&str, Call, &str)> = vec![
        (
            "mk_engine_create",
            Box::new(|_, t| unsafe {
                let mut e = ptr::null_mut();
                let st = mk_engine_create(t.as_ptr(), &mut e);
                assert!(e.is_null());
                st
            }),
            r#"{"worker_threads": "many"}"#,
        ),
        (
            "mk_session_set_system_json",
            Box::new(|h, t| unsafe {
                mk_session_set_system_json(h.session, t.as_ptr(), ptr::null_mut())
            }),
            r#"{"name": 3}"#,
        ),
        (
            "mk_session_apply_param_edit",
            Box::new(|h, t| unsafe {
                mk_session_apply_param_edit(h.session, t.as_ptr(), ptr::null_mut())
            }),
            r#"{"path": 3, "value": 1}"#,
        ),
        (
            "mk_session_set_project_section_json",
            Box::new(|h, t| unsafe {
                mk_session_set_project_section_json(h.session, c("views").as_ptr(), t.as_ptr())
            }),
            "3",
        ),
        (
            "mk_session_submit_analysis (settings_json)",
            Box::new(|h, t| unsafe {
                let mut job = 0u64;
                mk_session_submit_analysis(
                    h.session,
                    c("rng_stream").as_ptr(),
                    t.as_ptr(),
                    ptr::null(),
                    &mut job,
                )
            }),
            r#"{"count": "many"}"#,
        ),
        (
            "mk_session_submit_analysis_json",
            Box::new(|h, t| unsafe {
                let mut job = 0u64;
                mk_session_submit_analysis_json(h.session, t.as_ptr(), &mut job)
            }),
            r#"{"analysis": 3}"#,
        ),
        (
            "mk_session_run_analysis_json",
            Box::new(|h, t| unsafe {
                let mut json = ptr::null_mut();
                let mut bufs: *mut MKBuffer = ptr::null_mut();
                let mut n = 0usize;
                mk_session_run_analysis_json(h.session, t.as_ptr(), &mut json, &mut bufs, &mut n)
            }),
            r#"["validate"]"#,
        ),
        (
            "mk_engine_set_default_backend",
            Box::new(|h, t| unsafe { mk_engine_set_default_backend(h.engine, t.as_ptr()) }),
            r#"{"kind": 3}"#,
        ),
        (
            "mk_session_apply_solves",
            Box::new(|h, t| unsafe {
                mk_session_apply_solves(h.session, t.as_ptr(), ptr::null_mut(), ptr::null_mut())
            }),
            r#"{"max_iterations": "many"}"#,
        ),
        (
            "mk_session_get_param",
            Box::new(|h, t| unsafe {
                let mut out = ptr::null_mut();
                mk_session_get_param(h.session, t.as_ptr(), &mut out)
            }),
            r#"{"param": "no_such_parameter"}"#,
        ),
    ];
    for (name, call, wrong_shape) in &calls {
        assert_eq!(
            call(&h, &c("{not json")),
            MK_ERR_PARSE,
            "{name}: {}",
            last_error()
        );
        assert_eq!(
            call(&h, &c(wrong_shape)),
            MK_ERR_INVALID_ARGUMENT,
            "{name}: {}",
            last_error()
        );
        assert!(!last_error().is_empty(), "{name}");
    }
    // An array is not read as an object's fields in order.
    let create = &calls[0].1;
    assert_eq!(create(&h, &c("[2]")), MK_ERR_INVALID_ARGUMENT);
    assert!(last_error().contains("object"), "{}", last_error());
}

#[test]
fn misspelt_keys_are_errors_that_name_the_key() {
    // Configuration keys.
    for (config, key) in [
        (r#"{"worker_thread": 2}"#, "worker_thread"),
        (r#"{"route": {"fieldmap": "device-chief"}}"#, "fieldmap"),
        (
            r#"{"default_backend": {"kind": "auto", "nmae": "metal"}}"#,
            "nmae",
        ),
    ] {
        unsafe {
            let mut e = ptr::null_mut();
            assert_eq!(
                mk_engine_create(c(config).as_ptr(), &mut e),
                MK_ERR_INVALID_ARGUMENT,
                "{config}"
            );
            assert!(e.is_null());
        }
        let message = last_error();
        assert!(
            message.contains(&format!("`{key}`")) || message.contains(&format!("\"{key}\"")),
            "{config}: {message}"
        );
    }
    // Request keys, queued and synchronous alike.
    let h = open(&samples::singlet());
    for (request, key) in [
        (r#"{"analysis": "validate", "setings": {}}"#, "setings"),
        (
            r#"{"analysis": "validate", "backened": {"kind": "cpu_reference"}}"#,
            "backened",
        ),
        (
            r#"{"analysis": "validate", "backend": {"kind": "auto", "nmae": "metal"}}"#,
            "nmae",
        ),
        (
            r#"{"analysis": "validate", "route": {"fieldmap": "device-chief"}}"#,
            "fieldmap",
        ),
    ] {
        for (st, message) in [submit_status(&h, request), run_status(&h, request)] {
            assert_eq!(st, MK_ERR_INVALID_ARGUMENT, "{request}: {message}");
            assert!(
                message.contains(&format!("`{key}`")) || message.contains(&format!("\"{key}\"")),
                "{request}: {message}"
            );
        }
    }
    // Every documented key is accepted.
    let full = r#"{"analysis": "validate", "settings": {}, "view_key": "w",
                   "backend": {"kind": "cpu_reference"}, "allow_backend_fallback": true,
                   "tier": "design", "aim_stop": null, "route": {"fieldmaps": "host-chief"}}"#;
    assert_eq!(submit_status(&h, full).0, MK_OK, "{}", last_error());
    assert_eq!(run_status(&h, full).0, MK_OK, "{}", last_error());
}

/// Every value stored under `key` anywhere in `v`.
fn values<'a>(v: &'a Value, key: &str, out: &mut Vec<&'a Value>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                if k == key {
                    out.push(x);
                }
                values(x, key, out);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| values(x, key, out)),
        _ => {}
    }
}

#[test]
fn a_request_route_reaches_the_run() {
    let h = open(&samples::cooke_triplet());
    // A route value the run cannot take fails the request, naming the option.
    for (route, named) in [
        (json!({"stray_share": 2}), "stray_share"),
        (json!({"fieldmaps": "nonsense"}), "nonsense"),
    ] {
        let request = json!({"analysis": "chief_ray_angle", "route": route}).to_string();
        for (st, message) in [submit_status(&h, &request), run_status(&h, &request)] {
            assert_eq!(st, MK_ERR_INVALID_ARGUMENT, "{request}: {message}");
            assert!(message.contains(named), "{request}: {message}");
        }
    }
    // On a machine with a GPU the device evidence of the run names the
    // variant the request's route selected.
    let backends: Value = unsafe {
        let mut out = ptr::null_mut();
        assert_eq!(mk_engine_list_backends(h.engine, &mut out), MK_OK);
        serde_json::from_str(&take(out)).unwrap()
    };
    let metal = backends
        .as_array()
        .unwrap()
        .iter()
        .any(|b| b["id"] == "gpu:metal" && b["available"] == true);
    if !metal {
        eprintln!("no Metal device: the route evidence is not checked");
        return;
    }
    let variant = |route: Value| -> Vec<Value> {
        let request = json!({
            "analysis": "chief_ray_angle", "settings": {"grid": [5, 5]},
            "backend": {"kind": "gpu", "name": "metal"}, "route": route,
        })
        .to_string();
        unsafe {
            let req = c(&request);
            let mut json = ptr::null_mut();
            let mut bufs: *mut MKBuffer = ptr::null_mut();
            let mut n = 0usize;
            let st =
                mk_session_run_analysis_json(h.session, req.as_ptr(), &mut json, &mut bufs, &mut n);
            assert_eq!(st, MK_OK, "{request}: {}", last_error());
            mk_buffer_free(bufs, n);
            let envelope: Value = serde_json::from_str(&take(json)).unwrap();
            let mut out = Vec::new();
            values(
                &envelope["backend"]["execution"],
                "fieldmaps_variant",
                &mut out,
            );
            out.into_iter().cloned().collect()
        }
    };
    assert_eq!(
        variant(json!({"fieldmaps": "device-chief"})),
        [json!("device-chief")]
    );
    assert_eq!(variant(json!({})), [json!("host-chief")]);
    assert_eq!(variant(Value::Null), [json!("host-chief")]);
}

#[test]
fn the_listing_says_which_analyses_can_be_stopped() {
    let h = open(&samples::singlet());
    let list: Value = unsafe {
        let mut out = ptr::null_mut();
        assert_eq!(mk_engine_list_analyses(h.engine, &mut out), MK_OK);
        serde_json::from_str(&take(out)).unwrap()
    };
    let mut stoppable = Vec::new();
    for a in list.as_array().unwrap() {
        assert!(a["stoppable"].is_boolean(), "{}", a["name"]);
        if a["stoppable"] == true {
            stoppable.push(a["name"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(stoppable, ["glass_substitution", "optimize", "tolerance"]);
    // `mk_job_stop` accepts exactly the jobs of those analyses.
    for (request, status) in [
        (
            r#"{"analysis": "optimize", "settings": {"algorithm": "de", "max_iterations": 100000, "hold_efl": false}}"#,
            MK_OK,
        ),
        (
            r#"{"analysis": "rng_stream", "settings": {"count": 100000, "chunk": 100, "delay_ms": 20}}"#,
            MK_ERR_UNSUPPORTED,
        ),
    ] {
        unsafe {
            let mut job = 0u64;
            assert_eq!(
                mk_session_submit_analysis_json(h.session, c(request).as_ptr(), &mut job),
                MK_OK,
                "{request}: {}",
                last_error()
            );
            assert_eq!(mk_job_stop(h.engine, job), status, "{request}");
            assert_eq!(mk_job_release(h.engine, job), MK_OK);
        }
    }
}
