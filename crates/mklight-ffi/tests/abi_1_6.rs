//! The ABI 1.6 entry point: `mk_session_set_project_json`, so that a front
//! end autosaves and recovers its whole project through the engine.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use serde_json::{Value, json};

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

unsafe fn project(session: *const MKSession) -> Value {
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { mk_session_get_project_json(session, &mut out) },
        MK_OK
    );
    serde_json::from_str(&unsafe { take(out) }).unwrap()
}

unsafe fn info(session: *const MKSession) -> Value {
    let mut out = ptr::null_mut();
    assert_eq!(unsafe { mk_session_info_json(session, &mut out) }, MK_OK);
    serde_json::from_str(&unsafe { take(out) }).unwrap()
}

/// The project without the fields every save stamps.
fn unstamped(mut p: Value) -> Value {
    p["metadata"]["modified"] = Value::Null;
    p["metadata"]["engine_version"] = Value::Null;
    p
}

#[test]
fn whole_project_survives_autosave_and_recovery() {
    assert_eq!(mk_abi_version() & 0xffff, 6);
    let dir = std::env::temp_dir().join(format!("mklight-ffi-abi16-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("lens.mklight");
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);
        assert_eq!(
            mk_session_save(session, c(file.to_str().unwrap()).as_ptr()),
            MK_OK
        );

        // The front end's whole document: every section plus a key of its own.
        let mut doc = project(session);
        doc["system"]["name"] = json!("Recovered");
        doc["metadata"]["author"] = json!("Optics team");
        doc["views"] = json!([{"key": "spot-1", "analysis": "spot_diagram",
            "settings": {"field": 1}, "ui": {"split": 0.4}}]);
        doc["merit"] = json!({"operands": [{"kind": "effl", "target": 50.0}]});
        doc["variables"] = json!([{"param": "thickness", "surface": 1}]);
        doc["tolerances"] = json!({"operands": [{"kind": "tthi", "min": -0.1, "max": 0.1}]});
        doc["layout"] = json!({"window": [0, 0, 1280, 800], "panes": ["lde", "spot-1"]});

        let text = c(&doc.to_string());
        let mut hash = ptr::null_mut();
        let st = mk_session_set_project_json(session, text.as_ptr(), &mut hash);
        assert_eq!(st, MK_OK, "{}", last_error());
        let hash = take(hash);
        let i = info(session);
        assert_eq!(i["snapshot_hash"], hash.as_str());
        assert_eq!(i["dirty"], true);
        assert_eq!(unstamped(project(session)), unstamped(doc.clone()));

        // Autosave next to the project, then recover from the autosave.
        let mut written = ptr::null_mut();
        assert_eq!(
            mk_session_autosave(session, ptr::null(), &mut written),
            MK_OK
        );
        let written = take(written);
        assert!(written.ends_with("lens.mklight.autosave"), "{written}");
        let mut recovered = ptr::null_mut();
        assert_eq!(
            mk_session_open(engine, c(&written).as_ptr(), &mut recovered),
            MK_OK
        );
        assert_eq!(unstamped(project(recovered)), unstamped(doc.clone()));
        assert_eq!(info(recovered)["snapshot_hash"], hash.as_str());
        mk_session_destroy(recovered);

        // Errors leave the session as it was.
        let before = project(session);
        let mut bad_system = doc.clone();
        bad_system["merit"] = json!("changed");
        bad_system["system"]["stop"] = json!(999);
        let mut wrong_version = doc.clone();
        wrong_version["schema_version"] = json!(99);
        let mut no_format = doc.clone();
        no_format["format"] = json!("something-else");
        for (text, status) in [
            ("{not json".to_string(), MK_ERR_PARSE),
            ("[]".to_string(), MK_ERR_INVALID_ARGUMENT),
            (wrong_version.to_string(), MK_ERR_INVALID_ARGUMENT),
            (no_format.to_string(), MK_ERR_INVALID_ARGUMENT),
            (bad_system.to_string(), MK_ERR_INVALID_MODEL),
        ] {
            let mut h = ptr::null_mut();
            let st = mk_session_set_project_json(session, c(&text).as_ptr(), &mut h);
            assert_eq!(st, status, "{text}: {}", last_error());
            assert!(h.is_null());
            assert_eq!(project(session), before);
        }
        assert_eq!(
            mk_session_set_project_json(session, ptr::null(), ptr::null_mut()),
            MK_ERR_INVALID_ARGUMENT
        );

        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
    std::fs::remove_dir_all(&dir).ok();
}
