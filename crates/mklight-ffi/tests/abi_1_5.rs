//! The ABI 1.5 entry points: glass catalog management (library catalogs,
//! removal and reload, availability overrides, glass details, user glasses
//! and AGF export).

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

unsafe fn take_json(p: *mut c_char) -> Value {
    serde_json::from_str(&unsafe { take(p) }).unwrap()
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(mk_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// A synthetic catalog with two made-up glasses.
const AGF: &str = "NM SYNCROWN 2 517642 1.5168 64.17 0 0 0\r\n\
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\n\
LD 0.3 2.5\r\n\
NM SYNFLINT 2 620364 1.62004 36.37 0 0 0\r\n\
CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r\n\
LD 0.32 2.5\r\n";

#[test]
fn library_overrides_details_and_reload() {
    let dir = std::env::temp_dir().join(format!("mklight-ffi-abi15-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("SYN.agf");
    std::fs::write(&path, AGF).unwrap();
    unsafe {
        let mut engine = ptr::null_mut();
        assert_eq!(mk_engine_create(ptr::null(), &mut engine), MK_OK);
        let mut session = ptr::null_mut();
        assert_eq!(mk_session_new(engine, &mut session), MK_OK);

        let paths = c(&json!([path]).to_string());
        let mut out = ptr::null_mut();
        let st = mk_session_set_library_catalogs(session, paths.as_ptr(), &mut out);
        assert_eq!(st, MK_OK, "{}", last_error());
        let v = take_json(out);
        assert_eq!(v["changed"], true);
        assert_eq!(v["catalogs"][0]["name"], "SYN");
        assert_eq!(v["catalogs"][0]["library"], true);
        assert_eq!(v["catalogs"][0]["in_project"], false);

        let o = c(r#"[{"catalog":"SYN","name":"SYNFLINT","status":"preferred"}]"#);
        assert_eq!(mk_session_set_glass_overrides(session, o.as_ptr()), MK_OK);
        let mut out = ptr::null_mut();
        let st = mk_session_glass_details(session, ptr::null(), c("synflint").as_ptr(), &mut out);
        assert_eq!(st, MK_OK, "{}", last_error());
        let d = take_json(out);
        assert_eq!(d["catalog"], "SYN");
        assert_eq!(d["status"], "preferred");
        assert_eq!(d["catalog_status"], "standard");
        assert_eq!(d["formula"], 2);
        assert!(d["index_curve"].as_array().unwrap().len() > 10);
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_session_list_glasses(session, c("SYN").as_ptr(), &mut out),
            MK_OK
        );
        let rows = take_json(out);
        assert_eq!(rows.as_array().unwrap().len(), 2);
        assert!(rows[0]["dpgf"].is_number());
        let bad = c(r#"[{"catalog":"SYN"}]"#);
        assert_eq!(
            mk_session_set_glass_overrides(session, bad.as_ptr()),
            MK_ERR_INVALID_ARGUMENT
        );

        std::fs::write(&path, AGF.replace("1.62004 36.37", "1.62010 36.37")).unwrap();
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_session_reload_catalog(session, c("SYN").as_ptr(), &mut out),
            MK_OK
        );
        let ch = take_json(out);
        assert_eq!(ch["changed"][0]["name"], "SYNFLINT");
        assert_eq!(ch["changed"][0]["nd_after"], 1.6201);

        let mut out = ptr::null_mut();
        assert_eq!(mk_session_library_catalogs_in_use(session, &mut out), MK_OK);
        assert_eq!(take_json(out), json!([]));

        // Project catalogs: load, remove, restore.
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_session_load_catalog(session, c(path.to_str().unwrap()).as_ptr(), &mut out),
            MK_OK
        );
        assert_eq!(take_json(out)["in_project"], true);
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_session_remove_catalog(session, c("SYN").as_ptr(), &mut out),
            MK_OK
        );
        assert_eq!(take_json(out)["name"], "SYN");
        assert_eq!(
            mk_session_remove_catalog(session, c("SYN").as_ptr(), ptr::null_mut()),
            MK_ERR_NOT_FOUND
        );
        assert_eq!(
            mk_session_restore_catalog(session, c("SYN").as_ptr(), ptr::null_mut()),
            MK_OK
        );
        assert_eq!(
            mk_session_remove_library_catalog(session, c("SYN").as_ptr()),
            MK_OK
        );
        assert_eq!(
            mk_session_remove_library_catalog(session, c("SYN").as_ptr()),
            MK_ERR_NOT_FOUND
        );
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_session_add_library_catalog(session, c(path.to_str().unwrap()).as_ptr(), &mut out),
            MK_OK
        );
        assert_eq!(take_json(out)["library"], true);

        mk_session_destroy(session);
        mk_engine_destroy(engine);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn user_glasses_from_a_spec_and_agf_export() {
    unsafe {
        let spec = c(r#"{"name":"MYGLASS","model":{"nd":1.6,"vd":40,"dpgf":0.001},"density":3.2}"#);
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_glass_from_spec(spec.as_ptr(), &mut out),
            MK_OK,
            "{}",
            last_error()
        );
        let g = take_json(out);
        assert!((g["nd"].as_f64().unwrap() - 1.6).abs() < 1e-12);
        assert_eq!(g["density"], 3.2);
        let bad = c(r#"{"name":"","model":{"nd":1.6,"vd":40}}"#);
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_glass_from_spec(bad.as_ptr(), &mut out),
            MK_ERR_INVALID_ARGUMENT
        );
        assert!(out.is_null());

        let cat = c(&json!({"name": "USER", "glasses": {"MYGLASS": g}}).to_string());
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_catalog_to_agf(cat.as_ptr(), &mut out),
            MK_OK,
            "{}",
            last_error()
        );
        let v = take_json(out);
        assert!(v["text"].as_str().unwrap().starts_with("NM MYGLASS 1 "));
        assert_eq!(v["warnings"], json!([]));
        let mut out = ptr::null_mut();
        assert_eq!(
            mk_catalog_to_agf(c("[]").as_ptr(), &mut out),
            MK_ERR_INVALID_ARGUMENT
        );
    }
}
