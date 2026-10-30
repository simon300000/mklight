//! The job callback may call back into the engine and the session that
//! submitted the job (docs/FFI.md, "Job callback"): it runs on the engine's
//! event thread with no lock held, so neither a session call nor waiting on
//! the job it is told about can deadlock.

use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::Mutex;
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use mklight_ffi::*;

/// (job, state, snapshot-hash status, wait status, waited state)
type Report = (u64, i32, MKStatus, MKStatus, i32);

struct Ctx {
    engine: *mut MKEngine,
    session: *mut MKSession,
    tx: Mutex<Sender<Report>>,
    submitter: std::thread::ThreadId,
    on_submitter: Mutex<bool>,
}

unsafe extern "C" fn on_event(
    user: *mut c_void,
    job: u64,
    state: i32,
    _progress: f64,
    _stage: *const c_char,
) {
    let ctx = unsafe { &*(user as *const Ctx) };
    if std::thread::current().id() == ctx.submitter {
        *ctx.on_submitter.lock().unwrap() = true;
    }
    if state != 0 {
        return; // only QUEUED re-enters
    }
    unsafe {
        // Re-enter the very session that is submitting.
        let mut hash: *mut c_char = ptr::null_mut();
        let st_hash = mk_session_current_snapshot_hash(ctx.session, &mut hash);
        mk_string_free(hash);
        // Block on the job just queued.
        let mut done = -1;
        let st_wait = mk_job_wait(ctx.engine, job, 20_000, &mut done);
        let _ = ctx
            .tx
            .lock()
            .unwrap()
            .send((job, state, st_hash, st_wait, done));
    }
}

#[test]
fn callback_reentering_session_and_waiting_does_not_deadlock() {
    unsafe {
        let mut engine: *mut MKEngine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session: *mut MKSession = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let (tx, rx) = channel();
        let ctx = Box::new(Ctx {
            engine,
            session,
            tx: Mutex::new(tx),
            submitter: std::thread::current().id(),
            on_submitter: Mutex::new(false),
        });
        let user = &*ctx as *const Ctx as *mut c_void;
        assert_eq!(
            mk_engine_set_job_callback(engine, Some(on_event), user),
            MK_OK
        );

        let mut ids = Vec::new();
        for i in 0..5 {
            let t0 = Instant::now();
            let mut job = 0u64;
            let st = mk_session_submit_analysis(
                session,
                c"system_summary".as_ptr(),
                ptr::null(),
                if i % 2 == 0 {
                    c"pane".as_ptr()
                } else {
                    ptr::null()
                },
                &mut job,
            );
            assert_eq!(st, MK_OK, "{:?}", CStr::from_ptr(mk_last_error()));
            assert!(t0.elapsed() < Duration::from_secs(5), "submit blocked");
            ids.push(job);
        }
        for _ in &ids {
            let (job, state, st_hash, st_wait, done) = rx
                .recv_timeout(Duration::from_secs(20))
                .expect("callback deadlocked");
            assert!(ids.contains(&job));
            assert_eq!((state, st_hash, st_wait), (0, MK_OK, MK_OK));
            // Superseded jobs end cancelled, the others done.
            assert!(done == 2 || done == 4, "state {done}");
        }
        assert!(!*ctx.on_submitter.lock().unwrap());

        assert_eq!(
            mk_engine_set_job_callback(engine, None, ptr::null_mut()),
            MK_OK
        );
        drop(ctx);
        for id in ids {
            let _ = mk_job_release(engine, id);
        }
        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
}

#[test]
fn library_hash_is_exposed() {
    unsafe {
        let mut engine: *mut MKEngine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session: *mut MKSession = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let mut h: *mut c_char = ptr::null_mut();
        assert_eq!(mk_session_current_library_hash(session, &mut h), MK_OK);
        let hash = CStr::from_ptr(h).to_str().unwrap().to_string();
        mk_string_free(h);
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(
            mk_session_current_library_hash(session, ptr::null_mut()),
            MK_ERR_INVALID_ARGUMENT
        );

        let mut job = 0u64;
        assert_eq!(
            mk_session_submit_analysis(
                session,
                c"validate".as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut job
            ),
            MK_OK
        );
        let mut state = 0;
        assert_eq!(mk_job_wait(engine, job, -1, &mut state), MK_OK);
        let (mut json, mut bufs, mut n) = (ptr::null_mut(), ptr::null_mut(), 0usize);
        assert_eq!(
            mk_job_take_result(engine, job, &mut json, &mut bufs, &mut n),
            MK_OK
        );
        let v: serde_json::Value =
            serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
        mk_string_free(json);
        mk_buffer_free(bufs, n);
        assert_eq!(v["library_hash"], hash.as_str());
        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
}
