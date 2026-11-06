//! The ABI 1.3 entry points: lens prescription import, with and without a
//! session.

use std::ffi::{CStr, CString, c_char};
use std::path::PathBuf;
use std::ptr;

use serde_json::Value;

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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mklight-core/tests/fixtures/prescriptions")
        .join(name)
}

#[test]
fn import_bytes_without_a_session() {
    let bytes = std::fs::read(fixture("cooke_triplet.zmx")).unwrap();
    unsafe {
        let mut out = ptr::null_mut();
        let st = mk_import_prescription(bytes.as_ptr(), bytes.len(), c("zmx").as_ptr(), &mut out);
        assert_eq!(st, MK_OK, "{}", last_error());
        let v: Value = serde_json::from_str(&take(out)).unwrap();
        assert_eq!(v["format"], "zmx");
        assert_eq!(v["encoding"], "utf16_le");
        assert_eq!(v["system"]["stop"], 4);
        assert_eq!(v["system"]["surfaces"].as_array().unwrap().len(), 8);
        // The system is the engine's model JSON.
        let _: mklight_core::OpticalSystem = serde_json::from_value(v["system"].clone()).unwrap();
        let w = v["warnings"].as_array().unwrap();
        assert!(
            w.iter()
                .all(|w| w["kind"].is_string() && w["message"].is_string())
        );

        // Text without surfaces is a parse error; an unknown format an argument error.
        let mut out = ptr::null_mut();
        let junk = b"hello";
        assert_eq!(
            mk_import_prescription(junk.as_ptr(), junk.len(), c("seq").as_ptr(), &mut out),
            MK_ERR_PARSE
        );
        assert!(out.is_null());
        assert_eq!(
            mk_import_prescription(junk.as_ptr(), junk.len(), c("lens").as_ptr(), &mut out),
            MK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            mk_import_prescription(ptr::null(), 3, c("zmx").as_ptr(), &mut out),
            MK_ERR_INVALID_ARGUMENT
        );
    }
}

#[test]
fn import_into_a_session() {
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        let path = c(fixture("petzval_asphere.seq").to_str().unwrap());
        let mut out = ptr::null_mut();
        let st = mk_session_import_prescription(session, path.as_ptr(), &mut out);
        assert_eq!(st, MK_OK, "{}", last_error());
        let v: Value = serde_json::from_str(&take(out)).unwrap();
        assert_eq!(v["format"], "seq");
        let mut hash = ptr::null_mut();
        assert_eq!(mk_session_current_snapshot_hash(session, &mut hash), MK_OK);
        assert_eq!(v["snapshot_hash"], take(hash));
        let mut sys = ptr::null_mut();
        assert_eq!(mk_session_get_system_json(session, &mut sys), MK_OK);
        let sys: Value = serde_json::from_str(&take(sys)).unwrap();
        assert_eq!(sys["name"], "Petzval asphere f69 F/4");

        // A missing file fails and leaves the session as it was.
        let missing = c(fixture("missing.zmx").to_str().unwrap());
        assert_eq!(
            mk_session_import_prescription(session, missing.as_ptr(), ptr::null_mut()),
            MK_ERR_IO
        );
        let mut again = ptr::null_mut();
        assert_eq!(mk_session_get_system_json(session, &mut again), MK_OK);
        let again: Value = serde_json::from_str(&take(again)).unwrap();
        assert_eq!(again, sys);
        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
}
