//! The ABI 1.1 entry points on real systems: analysis descriptions with
//! result schemas, synchronous runs with buffers (checked against the
//! published schemas), solves as a new snapshot, parameter reads, backend
//! defaults and validation.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use serde_json::{Value, json};

use mklight_core::{Solve, samples};
use mklight_engine::analysis::schema::{check_buffers, validate};
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

fn open(sys: &mklight_core::OpticalSystem) -> Handles {
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

/// (name, dtype, ndim, data as f64).
type BufferCopy = (String, String, usize, Vec<f64>);

/// Run synchronously; returns the envelope and copies of the buffers.
fn run(h: &Handles, request: Value) -> (Value, Vec<BufferCopy>) {
    unsafe {
        let req = c(&request.to_string());
        let mut json = ptr::null_mut();
        let mut bufs: *mut MKBuffer = ptr::null_mut();
        let mut n = 0usize;
        let st =
            mk_session_run_analysis_json(h.session, req.as_ptr(), &mut json, &mut bufs, &mut n);
        assert_eq!(st, MK_OK, "{request}: {}", last_error());
        let env: Value = serde_json::from_str(&take(json)).unwrap();
        let mut out = Vec::new();
        for i in 0..n {
            let b = &*bufs.add(i);
            let name = CStr::from_ptr(b.name).to_string_lossy().into_owned();
            let dtype = env["buffers"][i]["dtype"].as_str().unwrap().to_string();
            let data: Vec<f64> = match b.dtype {
                MK_DTYPE_F64 => std::slice::from_raw_parts(b.data as *const f64, b.len).to_vec(),
                MK_DTYPE_F32 => std::slice::from_raw_parts(b.data as *const f32, b.len)
                    .iter()
                    .map(|&x| x as f64)
                    .collect(),
                _ => Vec::new(),
            };
            let shape = std::slice::from_raw_parts(b.shape, b.ndim as usize);
            assert_eq!(shape.iter().product::<usize>(), b.len, "{name}");
            assert_eq!(env["buffers"][i]["name"], name.as_str());
            out.push((name, dtype, b.ndim as usize, data));
        }
        mk_buffer_free(bufs, n);
        (env, out)
    }
}

#[test]
fn every_listed_analysis_has_schemas_and_results_match_them() {
    let h = open(&samples::double_gauss());
    let list: Value = unsafe {
        let mut out = ptr::null_mut();
        assert_eq!(mk_engine_list_analyses(h.engine, &mut out), MK_OK);
        serde_json::from_str(&take(out)).unwrap()
    };
    for a in list.as_array().unwrap() {
        assert!(a["result_schema"].is_object(), "{}", a["name"]);
        assert!(a["result_version"].as_u64().unwrap() >= 1);
        assert!(a["gpu_path"].is_boolean());
    }
    for name in [
        "spot_diagram",
        "layout",
        "ghost_paths",
        "polarization_pupil_map",
        "first_order",
    ] {
        let info: Value = unsafe {
            let mut out = ptr::null_mut();
            let n = c(name);
            assert_eq!(
                mk_engine_analysis_info(h.engine, n.as_ptr(), &mut out),
                MK_OK
            );
            serde_json::from_str(&take(out)).unwrap()
        };
        let (env, bufs) = run(&h, json!({"analysis": name}));
        assert_eq!(env["run_info"]["analysis"], name);
        validate(&info["result_schema"], &env["result"], true)
            .unwrap_or_else(|m| panic!("{name}: {m}"));
        let descs: Vec<(String, String, usize)> = bufs
            .iter()
            .map(|(n, d, k, _)| (n.clone(), d.clone(), *k))
            .collect();
        check_buffers(&info["result_schema"], &descs).unwrap();
    }
    // Spot points: one row per (field, ray) with the field ranges in JSON.
    let (env, bufs) = run(
        &h,
        json!({"analysis": "spot_diagram", "settings": {"rings": 3}}),
    );
    let pts = &bufs.iter().find(|b| b.0 == "points").unwrap().3;
    let total: u64 = env["result"]["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["point_range"][1].as_u64().unwrap())
        .sum();
    assert_eq!(pts.len() as u64, 2 * total);
    assert!(total > 0);
}

#[test]
fn solves_params_backends_and_validation() {
    let mut sys = samples::cooke_triplet();
    let j = sys.surfaces.len() - 2;
    sys.surfaces[j].thickness += 1.5;
    let id = sys.surfaces[j].id.0;
    sys.solves = vec![Solve::marginal_ray_height(sys.surfaces[j].id, 0.0)];
    let h = open(&sys);
    unsafe {
        let mut h0 = ptr::null_mut();
        assert_eq!(mk_session_current_snapshot_hash(h.session, &mut h0), MK_OK);
        let h0 = take(h0);
        let (mut report, mut hash) = (ptr::null_mut(), ptr::null_mut());
        assert_eq!(
            mk_session_apply_solves(h.session, ptr::null(), &mut report, &mut hash),
            MK_OK,
            "{}",
            last_error()
        );
        let report: Value = serde_json::from_str(&take(report)).unwrap();
        let h1 = take(hash);
        assert_ne!(h0, h1);
        let solved = report["outcomes"][0]["value"].as_f64().unwrap();
        // The new value is readable through a ParamPath.
        let p = c(&json!({"param": "thickness", "surface": id}).to_string());
        let mut out = ptr::null_mut();
        assert_eq!(mk_session_get_param(h.session, p.as_ptr(), &mut out), MK_OK);
        let v: Value = serde_json::from_str(&take(out)).unwrap();
        assert_eq!(v["value"].as_f64().unwrap(), solved);
        // The object distance is infinite.
        let p = c(r#"{"param": "thickness", "surface": 0}"#);
        assert_eq!(mk_session_get_param(h.session, p.as_ptr(), &mut out), MK_OK);
        assert_eq!(take(out), r#"{"value":"inf"}"#);

        // Validation through the engine.
        assert_eq!(mk_session_validate_json(h.session, &mut out), MK_OK);
        let v: Value = serde_json::from_str(&take(out)).unwrap();
        assert_eq!(v["ok"], true, "{v}");

        // Backends: auto as the default, recorded in the result.
        let b = c(r#"{"kind": "auto"}"#);
        assert_eq!(mk_engine_set_default_backend(h.engine, b.as_ptr()), MK_OK);
    }
    let (env, _) = run(&h, json!({"analysis": "seidel"}));
    assert_eq!(env["backend"]["requested"], "default:auto");
    assert_eq!(env["run_info"]["backend"], "cpu_reference");
    let (env, _) = run(
        &h,
        json!({"analysis": "ray_grid", "settings": {"rings": 2}}),
    );
    // A GPU when this machine has one that qualifies, else the reference;
    // either way the report and RunInfo agree.
    assert_eq!(env["backend"]["used"], env["run_info"]["backend"]);
}
