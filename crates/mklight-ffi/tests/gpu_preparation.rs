//! GPU kernel preparation through the C ABI: argument checks and the state
//! report, on an engine whose GPU backends are disabled, so the test neither
//! needs a GPU nor compiles kernels on a machine that has one.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

use serde_json::Value;

use mklight_ffi::*;

fn last_error() -> String {
    unsafe { CStr::from_ptr(mk_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The engine's preparation report.
unsafe fn preparation(engine: *const MKEngine) -> Value {
    let mut out: *mut c_char = ptr::null_mut();
    let st = unsafe { mk_engine_gpu_preparation_json(engine, &mut out) };
    assert_eq!(st, MK_OK, "{}", last_error());
    assert!(!out.is_null());
    let text = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    unsafe { mk_string_free(out) };
    serde_json::from_str(&text).unwrap()
}

#[test]
fn preparation_checks_arguments_and_reports_its_state() {
    // SAFETY: this file's only test, so no other thread of the process
    // reads or writes the environment.
    unsafe { std::env::set_var("MKLIGHT_DISABLE_GPU", "1") };
    unsafe {
        let mut engine: *mut MKEngine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);

        // NULL arguments.
        assert_eq!(
            mk_engine_prepare_gpu(ptr::null(), ptr::null()),
            MK_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("engine"), "{}", last_error());
        let mut out: *mut c_char = c"stale".as_ptr().cast_mut();
        assert_eq!(
            mk_engine_gpu_preparation_json(ptr::null(), &mut out),
            MK_ERR_INVALID_ARGUMENT
        );
        assert!(out.is_null(), "the out-parameter is cleared on failure");
        assert_eq!(
            mk_engine_gpu_preparation_json(engine, ptr::null_mut()),
            MK_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("out_json"), "{}", last_error());

        // A cache directory that is not UTF-8 is refused and starts nothing.
        let bad = CString::new(vec![0xff, 0xfe]).unwrap();
        assert_eq!(
            mk_engine_prepare_gpu(engine, bad.as_ptr()),
            MK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(preparation(engine), serde_json::json!({"state": "idle"}));

        // Starting it (cache_dir is nullable) leaves `idle` at once.
        assert_eq!(
            mk_engine_prepare_gpu(engine, ptr::null()),
            MK_OK,
            "{}",
            last_error()
        );
        assert_ne!(preparation(engine)["state"], "idle");

        // With every GPU disabled it settles as unavailable, with the reason.
        let start = Instant::now();
        let settled = loop {
            let p = preparation(engine);
            if p["state"] != "preparing" {
                break p;
            }
            assert!(start.elapsed() < Duration::from_secs(60), "{p}");
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(settled["state"], "unavailable", "{settled}");
        assert!(
            settled["reason"].as_str().is_some_and(|r| !r.is_empty()),
            "{settled}"
        );

        // A second call succeeds and starts nothing.
        let dir = CString::new("/nonexistent/mklight-kernels").unwrap();
        assert_eq!(mk_engine_prepare_gpu(engine, dir.as_ptr()), MK_OK);
        assert_eq!(preparation(engine), settled);

        mk_engine_destroy(engine);
    }
}
