//! ABI 1.7: `mk_session_set_project_json` takes the document's `catalogs`
//! as the project's catalog references, so that a front end that resolves
//! catalog files itself autosaves and recovers every reference it holds.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr;

use serde_json::{Value, json};

use mklight_ffi::*;

/// A synthetic catalog with two made-up glasses.
const AGF: &str = "NM SYNCROWN 2 517642 1.5168 64.17 0 0 0\r\n\
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\n\
LD 0.3 2.5\r\n\
NM SYNFLINT 2 620364 1.62004 36.37 0 0 0\r\n\
CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r\n\
LD 0.32 2.5\r\n";

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn cp(p: &Path) -> CString {
    c(p.to_str().unwrap())
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

unsafe fn json_of(
    f: unsafe extern "C" fn(*const MKSession, *mut *mut c_char) -> MKStatus,
    s: *const MKSession,
) -> Value {
    let mut out = ptr::null_mut();
    assert_eq!(unsafe { f(s, &mut out) }, MK_OK, "{}", last_error());
    serde_json::from_str(&unsafe { take(out) }).unwrap()
}

unsafe fn project(s: *const MKSession) -> Value {
    unsafe { json_of(mk_session_get_project_json, s) }
}

unsafe fn info(s: *const MKSession) -> Value {
    unsafe { json_of(mk_session_info_json, s) }
}

/// Names of the catalogs in the session's library.
unsafe fn loaded(s: *const MKSession) -> Vec<String> {
    let v = unsafe { json_of(mk_session_list_catalogs, s) };
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_string())
        .collect()
}

unsafe fn missing(s: *const MKSession, name: &str) -> bool {
    let i = unsafe { info(s) };
    i["open_warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains(&format!("'{name}'")))
}

unsafe fn set(s: *mut MKSession, doc: &Value) -> MKStatus {
    unsafe { mk_session_set_project_json(s, c(&doc.to_string()).as_ptr(), ptr::null_mut()) }
}

#[test]
fn catalog_references_survive_autosave_recovery_and_save() {
    assert_eq!(mk_abi_version() & 0xffff, 7);
    let dir = std::env::temp_dir().join(format!("mklight-ffi-abi17-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Canonical, as `mk_session_open` records the path it found a file at.
    let dir = std::fs::canonicalize(&dir).unwrap();
    let proj = dir.join("proj");
    let optics = dir.join("optics");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::create_dir_all(&optics).unwrap();
    let local = proj.join("LOCAL.agf");
    let shared = optics.join("SHARED.agf");
    std::fs::write(&local, AGF).unwrap();
    std::fs::write(&shared, AGF).unwrap();
    let offline = dir.join("offline-share").join("NETCAT.agf");
    let file = proj.join("lens.mklight");
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut s = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut s), MK_OK);
        assert_eq!(mk_session_save(s, cp(&file).as_ptr()), MK_OK);
        // The front end found two of its three files and loaded them.
        for f in [&local, &shared] {
            let st = mk_session_load_catalog(s, cp(f).as_ptr(), ptr::null_mut());
            assert_eq!(st, MK_OK, "{}", last_error());
        }

        // Without `catalogs`, nothing changes (the ABI 1.6 behaviour).
        let mut doc = project(s);
        let engine_refs = doc["catalogs"].clone();
        doc.as_object_mut().unwrap().remove("catalogs");
        doc["layout"] = json!({"panes": 2});
        assert_eq!(set(s, &doc), MK_OK, "{}", last_error());
        assert_eq!(project(s)["catalogs"], engine_refs);
        assert_eq!(project(s)["layout"], json!({"panes": 2}));

        // Every reference as written, the one not found included.
        let refs = json!([
            {"name": "NETCAT", "format": "agf", "path": offline.to_str().unwrap(),
             "relative_path": "../net/NETCAT.agf"},
            {"name": "LOCAL", "format": "agf", "path": local.to_str().unwrap(),
             "relative_path": "LOCAL.agf"},
            {"name": "SHARED", "format": "agf", "path": shared.to_str().unwrap(),
             "relative_path": "../optics/SHARED.agf"},
        ]);
        doc["catalogs"] = refs.clone();
        assert_eq!(set(s, &doc), MK_OK, "{}", last_error());
        assert_eq!(project(s)["catalogs"], refs);
        assert!(missing(s, "NETCAT"));
        assert!(!missing(s, "LOCAL") && !missing(s, "SHARED"));
        let names = loaded(s);
        assert!(names.contains(&"LOCAL".into()) && names.contains(&"SHARED".into()));

        // Autosave, recover, save: all three as written.
        let mut written = ptr::null_mut();
        assert_eq!(mk_session_autosave(s, ptr::null(), &mut written), MK_OK);
        let written = take(written);
        let mut r = ptr::null_mut();
        assert_eq!(mk_session_open(engine, c(&written).as_ptr(), &mut r), MK_OK);
        assert_eq!(project(r)["catalogs"], refs);
        assert!(missing(r, "NETCAT"));
        assert!(loaded(r).contains(&"SHARED".into()));
        assert_eq!(mk_session_save(r, cp(&file).as_ptr()), MK_OK);
        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(saved["catalogs"][0], refs[0]);
        mk_session_destroy(r);

        // The session's own save keeps the given relative paths too.
        assert_eq!(mk_session_save(s, ptr::null()), MK_OK);
        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(saved["catalogs"], refs);

        // A recorded catalog left out is removed; restore brings it back.
        doc["catalogs"] = json!([refs[0], refs[1]]);
        assert_eq!(set(s, &doc), MK_OK, "{}", last_error());
        assert!(!loaded(s).contains(&"SHARED".into()));
        assert_eq!(project(s)["catalogs"].as_array().unwrap().len(), 2);
        assert_eq!(
            mk_session_restore_catalog(s, c("SHARED").as_ptr(), ptr::null_mut()),
            MK_OK
        );
        assert!(loaded(s).contains(&"SHARED".into()));

        // Duplicate names are refused and change nothing.
        let before = project(s);
        doc["catalogs"] = json!([refs[1], {"name": "local"}]);
        assert_eq!(set(s, &doc), MK_ERR_INVALID_ARGUMENT);
        assert_eq!(project(s), before);

        mk_session_destroy(s);
        mk_engine_destroy(engine);
    }
    std::fs::remove_dir_all(&dir).ok();
}
