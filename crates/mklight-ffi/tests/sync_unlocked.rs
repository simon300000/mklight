//! The synchronous FFI calls do not hold the session lock while they
//! compute: other calls on the same session proceed during a long
//! `mk_session_run_analysis_json`, and the result belongs to the snapshot
//! captured when the call started.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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

fn last_error() -> String {
    unsafe { CStr::from_ptr(mk_last_error()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_long_synchronous_run_does_not_block_the_session() {
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let text = c(&serde_json::to_string(&samples::cooke_triplet()).unwrap());
        assert_eq!(
            mk_session_set_system_json(session, text.as_ptr(), ptr::null_mut()),
            MK_OK
        );
        let mut h0 = ptr::null_mut();
        assert_eq!(mk_session_current_snapshot_hash(session, &mut h0), MK_OK);
        let h0 = take(h0);

        // A stray-light run of a second or more on another thread.
        let done = Arc::new(AtomicBool::new(false));
        let addr = session as usize;
        let worker = {
            let done = done.clone();
            std::thread::spawn(move || {
                let t0 = Instant::now();
                let s = addr as *const MKSession;
                let req = c(&json!({"analysis": "stray_light",
                                    "settings": {"rays": 400_000, "seed": 1}})
                .to_string());
                let (mut json, mut bufs, mut n) = (ptr::null_mut(), ptr::null_mut(), 0usize);
                let st =
                    mk_session_run_analysis_json(s, req.as_ptr(), &mut json, &mut bufs, &mut n);
                done.store(true, Ordering::SeqCst);
                assert_eq!(st, MK_OK, "{}", last_error());
                let env: Value = serde_json::from_str(&take(json)).unwrap();
                mk_buffer_free(bufs, n);
                (env, t0.elapsed())
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        // Meanwhile: read and edit the session.
        let t = Instant::now();
        let mut h = ptr::null_mut();
        assert_eq!(mk_session_current_snapshot_hash(session, &mut h), MK_OK);
        assert_eq!(take(h), h0);
        let id = samples::cooke_triplet().surfaces[2].id.0;
        let edit =
            c(&json!({"path": {"param": "thickness", "surface": id}, "value": 3.0}).to_string());
        let mut h1 = ptr::null_mut();
        let st = mk_session_apply_param_edit(session, edit.as_ptr(), &mut h1);
        assert_eq!(st, MK_OK, "{}", last_error());
        let h1 = take(h1);
        let mut v = ptr::null_mut();
        assert_eq!(mk_session_validate_json(session, &mut v), MK_OK);
        take(v);
        let busy = !done.load(Ordering::SeqCst);
        let waited = t.elapsed();
        let (env, run) = worker.join().unwrap();
        // Blocked calls would have waited for most of the run.
        assert!(
            busy && waited * 3 < run,
            "the session calls took {waited:?} during a {run:?} synchronous run"
        );
        // The run used the snapshot it started on.
        assert_eq!(env["run_info"]["snapshot_hash"], h0.as_str());
        assert_ne!(h0, h1);
        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
}
