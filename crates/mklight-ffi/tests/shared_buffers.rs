//! A result's buffers cross the C ABI as the allocations the solver filled:
//! every take of a result the analysis graph holds exports the same `data`
//! pointers, and each export keeps the arrays alive on its own.

use std::ffi::{CStr, CString};
use std::ptr;

use serde_json::{Value, json};

use mklight_core::samples;
use mklight_ffi::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
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

fn open() -> Handles {
    unsafe {
        let mut engine = ptr::null_mut();
        // The CPU reference: its results are stored in the analysis graph
        // whichever way they are requested.
        let config = c(r#"{"default_backend":{"kind":"cpu_reference"}}"#);
        assert_eq!(mk_engine_create(config.as_ptr(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let text = c(&serde_json::to_string(&samples::double_gauss()).unwrap());
        assert_eq!(
            mk_session_set_system_json(session, text.as_ptr(), ptr::null_mut()),
            MK_OK,
            "{}",
            last_error()
        );
        Handles { engine, session }
    }
}

/// One take of a result: the buffer array as C sees it.
struct Take {
    buffers: *mut MKBuffer,
    count: usize,
}

impl Take {
    fn all(&self) -> &[MKBuffer] {
        unsafe { std::slice::from_raw_parts(self.buffers, self.count) }
    }
    /// The bytes of every buffer, read through `data`.
    fn bytes(&self) -> Vec<Vec<u8>> {
        self.all()
            .iter()
            .map(|b| {
                let size = match b.dtype {
                    MK_DTYPE_F64 | MK_DTYPE_I64 | MK_DTYPE_U64 => 8,
                    MK_DTYPE_F32 | MK_DTYPE_I32 | MK_DTYPE_U32 => 4,
                    MK_DTYPE_U8 => 1,
                    other => panic!("dtype {other}"),
                };
                unsafe { std::slice::from_raw_parts(b.data as *const u8, b.len * size) }.to_vec()
            })
            .collect()
    }
    fn pointers(&self) -> Vec<usize> {
        self.all().iter().map(|b| b.data as usize).collect()
    }
    fn free(self) {
        unsafe { mk_buffer_free(self.buffers, self.count) };
    }
}

/// `mk_session_run_analysis_json`.
fn run(h: &Handles, request: &Value) -> Take {
    unsafe {
        let req = c(&request.to_string());
        let mut json = ptr::null_mut();
        let mut buffers: *mut MKBuffer = ptr::null_mut();
        let mut count = 0usize;
        let st = mk_session_run_analysis_json(
            h.session,
            req.as_ptr(),
            &mut json,
            &mut buffers,
            &mut count,
        );
        assert_eq!(st, MK_OK, "{request}: {}", last_error());
        mk_string_free(json);
        Take { buffers, count }
    }
}

/// Submit, wait, `mk_job_take_result`.
fn run_job(h: &Handles, request: &Value) -> Take {
    unsafe {
        let req = c(&request.to_string());
        let mut job = 0u64;
        assert_eq!(
            mk_session_submit_analysis_json(h.session, req.as_ptr(), &mut job),
            MK_OK,
            "{}",
            last_error()
        );
        let mut state = -1;
        assert_eq!(mk_job_wait(h.engine, job, 60_000, &mut state), MK_OK);
        let mut json = ptr::null_mut();
        let mut buffers: *mut MKBuffer = ptr::null_mut();
        let mut count = 0usize;
        assert_eq!(
            mk_job_take_result(h.engine, job, &mut json, &mut buffers, &mut count),
            MK_OK,
            "{}",
            last_error()
        );
        mk_string_free(json);
        Take { buffers, count }
    }
}

#[test]
fn takes_of_a_stored_result_export_the_same_allocations() {
    let h = open();
    for request in [
        json!({"analysis": "ray_grid", "settings": {"pattern": "square", "rings": 40}}),
        json!({"analysis": "spot_diagram", "settings": {"include_points": true}}),
        json!({"analysis": "psf", "settings": {"pupil": {"rays": 17}, "grid_size": 32}}),
    ] {
        // The run that builds the result, and two answered from the graph:
        // three exports alive at once, one set of arrays.
        let built = run(&h, &request);
        let hit = run_job(&h, &request);
        let again = run(&h, &request);
        assert!(built.count > 0, "{request}");
        assert!(built.all().iter().any(|b| b.len > 0), "{request}");
        assert_eq!(built.pointers(), hit.pointers(), "{request}");
        assert_eq!(built.pointers(), again.pointers(), "{request}");
        for b in built.all() {
            assert!(!b.data.is_null() && !b.owner.is_null());
        }
        let expected = built.bytes();
        let pointers = built.pointers();

        // Freeing one export leaves the others readable, in any order.
        built.free();
        assert_eq!(hit.bytes(), expected, "{request}");
        assert_eq!(again.bytes(), expected, "{request}");
        hit.free();
        assert_eq!(again.bytes(), expected, "{request}");
        again.free();

        // With every export freed the graph still holds the arrays.
        let later = run(&h, &request);
        assert_eq!(later.pointers(), pointers, "{request}");
        assert_eq!(later.bytes(), expected, "{request}");
        later.free();
    }
}

#[test]
fn an_export_outlives_the_engine() {
    let h = open();
    let request = json!({"analysis": "ray_grid", "settings": {"rings": 12}});
    let take = run(&h, &request);
    let second = run(&h, &request);
    let expected = take.bytes();
    // The graph's reference goes with the engine: the exports hold theirs.
    drop(h);
    assert_eq!(take.bytes(), expected);
    take.free();
    assert_eq!(second.bytes(), expected);
    second.free();
}
