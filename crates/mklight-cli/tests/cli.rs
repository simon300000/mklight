//! Integration tests: run the built `mklight` binary on the sample systems
//! and check its JSON output, exit codes, cancellation and determinism.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_mklight");

/// A temporary directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "mklight-cli-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn mklight<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(BIN).args(args).output().expect("run mklight")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exited normally")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

/// Run with `--json`, expect `exit`, parse stdout.
fn json_run(args: &[&str], exit: i32) -> Value {
    let mut a: Vec<&str> = args.to_vec();
    a.push("--json");
    let o = mklight(&a);
    assert_eq!(
        code(&o),
        exit,
        "mklight {args:?}\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    serde_json::from_str(&stdout(&o)).unwrap_or_else(|e| panic!("bad JSON ({e}): {}", stdout(&o)))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Create a project from a sample.
fn sample(dir: &TempDir, name: &str) -> PathBuf {
    let p = dir.join(&format!("{name}.mklight"));
    let o = mklight(["new", s(&p), "--sample", name]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    p
}

fn f(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected a number, got {v}"))
}

const SAMPLES: &[&str] = &[
    "singlet",
    "achromat",
    "cooke_triplet",
    "double_gauss",
    "newtonian",
    "petzval_asphere",
    "relay",
];

// ---------------------------------------------------------------------------
// usage and info

#[test]
fn help_version_and_usage_errors() {
    assert_eq!(code(&mklight(["--help"])), 0);
    assert_eq!(code(&mklight(["--version"])), 0);
    assert!(stdout(&mklight(["--version"])).contains(env!("CARGO_PKG_VERSION")));
    // No command, unknown command, unknown flag: usage (1), not clap's 2.
    assert_eq!(code(&mklight::<[&str; 0], &str>([])), 1);
    assert_eq!(code(&mklight(["frobnicate"])), 1);
    assert_eq!(code(&mklight(["info", "--bogus"])), 1);
    assert_eq!(code(&mklight(["run", "--backend", "quantum", "x", "y"])), 1);
}

#[test]
fn info_reports_versions_and_backends() {
    let v = json_run(&["info"], 0);
    assert_eq!(v["engine_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["abi_version"]["major"], 1);
    assert_eq!(
        f(&v["abi_version"]["packed"]) as u64,
        (v["abi_version"]["major"].as_u64().unwrap() << 16)
            | v["abi_version"]["minor"].as_u64().unwrap()
    );
    assert_eq!(v["project_schema_version"], 1);
    let backends = v["backends"].as_array().unwrap();
    let cpu = backends
        .iter()
        .find(|b| b["id"] == "cpu_reference")
        .expect("cpu reference listed");
    assert_eq!(cpu["available"], true);
    assert_eq!(cpu["precision"], "f64_reference");
    assert_eq!(cpu["capabilities"]["fp64"], true);
    assert!(stdout(&mklight(["info"])).contains("cpu_reference"));
}

// ---------------------------------------------------------------------------
// projects

#[test]
fn every_sample_is_created_validated_and_shown() {
    let d = TempDir::new("samples");
    for name in SAMPLES {
        let p = sample(&d, name);
        let v = json_run(&["validate", s(&p)], 0);
        assert_eq!(v["ok"], true, "{name}: {v}");
        assert_eq!(v["errors"], 0);
        let show = json_run(&["show", s(&p)], 0);
        let surfaces = show["surfaces"].as_array().unwrap();
        assert!(surfaces.len() >= 3, "{name}");
        // Object at infinity is written as "inf"; every surface has a type.
        assert!(surfaces.iter().all(|r| r["type"].is_string()));
        assert_eq!(
            surfaces.iter().filter(|r| r["is_stop"] == true).count(),
            1,
            "{name}: one stop"
        );
        // Auto semi-diameters were traced: every non-object, non-CB surface has one.
        for r in surfaces.iter().skip(1) {
            if r["type"] != "coordinate_break" {
                assert!(f(&r["semi_diameter"]) > 0.0, "{name}: {r}");
            }
        }
        // Human output contains the table header and the stop.
        let text = stdout(&mklight(["show", s(&p)]));
        assert!(
            text.contains("thickness") && text.contains("STOP"),
            "{text}"
        );
    }
}

#[test]
fn show_reports_lens_data_of_the_achromat() {
    let d = TempDir::new("show");
    let p = sample(&d, "achromat");
    let v = json_run(&["show", s(&p)], 0);
    assert_eq!(v["name"], "Achromat f100 F/5");
    let r = &v["surfaces"];
    assert_eq!(r[0]["thickness"], "inf");
    assert_eq!(r[0]["radius"], "inf");
    assert!((f(&r[1]["radius"]) - 47.40).abs() < 1e-9);
    assert!((f(&r[2]["radius"]) + 42.54).abs() < 1e-9);
    assert_eq!(r[1]["material"], "N-BK7");
    assert_eq!(r[2]["material"], "F2");
    assert!((f(&r[1]["nd"]) - 1.5168).abs() < 1e-4);
    assert_eq!(r[1]["semi_diameter_mode"], "stop");
    // F/5 at f = 100: stop semi-diameter ≈ 10 mm.
    assert!((f(&r[1]["semi_diameter"]) - 10.0).abs() < 0.01);
    assert_eq!(v["system"]["aperture"]["type"], "image_space_f_number");
    assert_eq!(v["system"]["wavelengths"]["primary"], 1);
}

#[test]
fn show_lists_parameters_of_the_extended_samples() {
    let d = TempDir::new("extended");
    for (name, ty, expected) in [
        (
            "hybrid_doe",
            "diffractive",
            "A1 ρ^2 -76.957542; normalisation radius 5 mm; design order 1; design wavelength 0.5875618 µm; trace order 1",
        ),
        ("anamorphic", "biconic", "cx 0.01 mm⁻¹"),
        ("biconic", "biconic", "A4 x^4 1e-6; A4 y^4 -1e-6"),
        ("toroid", "toroid", "ky -0.5; A4 y^4 1e-6"),
        ("q_con", "q_con", "Q1 -0.0004"),
        ("q_bfs", "q_bfs", "normalisation radius 5 mm"),
        ("odd_asphere", "odd_asphere", "β1 r 0, β2 r^2 1e-5"),
        ("xy_freeform", "xy_polynomial", "C(x^2 y) -0.0001"),
        ("zernike_sag", "zernike_sag", "Z5 0.001"),
    ] {
        let p = sample(&d, name);
        assert_eq!(json_run(&["validate", s(&p)], 0)["ok"], true, "{name}");
        let text = stdout(&mklight(["show", s(&p)]));
        assert!(text.contains(&format!(" {ty}: ")), "{name}: {text}");
        assert!(text.contains(expected), "{name}: {text}");
        let v = json_run(&["show", s(&p)], 0);
        assert!(
            v["surfaces"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["type"] == ty && r["shape"]["type"] == ty),
            "{name}: {v}"
        );
    }
}

#[test]
fn paraxial_reports_x_and_y_first_order_of_anamorphic_systems() {
    let d = TempDir::new("anamorphic");
    let p = sample(&d, "anamorphic");
    let v = json_run(&["paraxial", s(&p)], 0);
    let primary = v["primary"].as_u64().unwrap() as usize;
    let xy = &v["paraxial_xy"];
    assert!((f(&xy["squeeze"][primary]) - 2.0).abs() < 1e-9, "{xy}");
    assert!((f(&xy["y"][primary]["efl"]) - 2.0 * f(&xy["x"][primary]["efl"])).abs() < 1e-9);
    let text = stdout(&mklight(["paraxial", s(&p)]));
    assert!(
        text.contains("anamorphic first order") && text.contains("squeeze EFL_y/EFL_x: 2\n"),
        "{text}"
    );
    let rotational = stdout(&mklight(["paraxial", s(&sample(&d, "achromat"))]));
    assert!(!rotational.contains("squeeze"), "{rotational}");
}

#[test]
fn new_refuses_to_overwrite_and_unknown_samples() {
    let d = TempDir::new("new");
    let p = sample(&d, "singlet");
    let before = std::fs::read(&p).unwrap();
    assert_eq!(code(&mklight(["new", s(&p), "--sample", "singlet"])), 1);
    assert_eq!(std::fs::read(&p).unwrap(), before);
    assert_eq!(
        code(&mklight([
            "new",
            s(&p),
            "--sample",
            "cooke_triplet",
            "--force"
        ])),
        0
    );
    let e = json_run(&["new", s(&d.join("x.mklight")), "--sample", "nope"], 1);
    assert_eq!(e["error"]["kind"], "usage");
    assert!(e["error"]["message"].as_str().unwrap().contains("achromat"));
    // Without a sample: the empty system, which validates.
    let empty = d.join("empty.mklight");
    assert_eq!(code(&mklight(["new", s(&empty)])), 0);
    assert_eq!(json_run(&["validate", s(&empty)], 0)["ok"], true);
}

#[test]
fn invalid_and_missing_projects_have_distinct_codes() {
    let d = TempDir::new("invalid");
    // Missing file: I/O.
    let missing = d.join("missing.mklight");
    let e = json_run(&["validate", s(&missing)], 4);
    assert_eq!(e["error"]["kind"], "io");
    // Not JSON: invalid model.
    let garbage = d.join("garbage.mklight");
    std::fs::write(&garbage, "{ not json").unwrap();
    assert_eq!(code(&mklight(["show", s(&garbage)])), 2);
    // A newer schema version is refused.
    let p = sample(&d, "singlet");
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    doc["schema_version"] = json!(99);
    let newer = d.join("newer.mklight");
    std::fs::write(&newer, doc.to_string()).unwrap();
    assert_eq!(code(&mklight(["validate", s(&newer)])), 2);
    // An unknown glass resolves to a validation error (exit 2, report printed),
    // and analyses that need indices fail with "invalid model" too.
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    doc["system"]["surfaces"][1]["material"] = json!({"type": "catalog", "name": "NO-SUCH"});
    let unknown = d.join("unknown.mklight");
    std::fs::write(&unknown, doc.to_string()).unwrap();
    let v = json_run(&["validate", s(&unknown)], 2);
    assert_eq!(v["ok"], false);
    assert!(
        v["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i["code"] == "material")
    );
    assert_eq!(code(&mklight(["run", s(&unknown), "system_summary"])), 2);
    assert_eq!(code(&mklight(["paraxial", s(&unknown)])), 2);
    // show still prints the model data, with a warning.
    let show = json_run(&["show", s(&unknown)], 0);
    assert!(!show["warnings"].as_array().unwrap().is_empty());
}

#[test]
fn migrate_upgrades_and_keeps_a_backup() {
    let d = TempDir::new("migrate");
    let p = sample(&d, "achromat");
    let doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    // A version-0 draft: the bare optical system.
    let old = d.join("old.mklight");
    let original = serde_json::to_vec(&doc["system"]).unwrap();
    std::fs::write(&old, &original).unwrap();

    let dry = json_run(&["migrate", s(&old), "--dry-run"], 0);
    assert_eq!(dry["from_version"], 0);
    assert_eq!(dry["to_version"], 1);
    assert_eq!(dry["migrated"], false);
    assert_eq!(
        std::fs::read(&old).unwrap(),
        original,
        "dry run writes nothing"
    );

    let v = json_run(&["migrate", s(&old)], 0);
    assert_eq!(v["migrated"], true);
    assert!(!v["notes"].as_array().unwrap().is_empty());
    let backup = PathBuf::from(v["backup"].as_str().unwrap());
    assert_eq!(
        std::fs::read(&backup).unwrap(),
        original,
        "backup is byte-exact"
    );
    let new: Value = serde_json::from_slice(&std::fs::read(&old).unwrap()).unwrap();
    assert_eq!(new["schema_version"], 1);
    assert_eq!(new["format"], "mklight-project");
    assert_eq!(new["system"], doc["system"]);

    // Already current: nothing to do, no second backup.
    let again = json_run(&["migrate", s(&old)], 0);
    assert_eq!(again["migrated"], false);
    assert_eq!(again["backup"], Value::Null);
    // The migrated project behaves like the original.
    let a = json_run(&["paraxial", s(&p), "--no-timing"], 0);
    let b = json_run(&["paraxial", s(&old), "--no-timing"], 0);
    assert_eq!(a["wavelengths"], b["wavelengths"]);
}

#[test]
fn set_edits_validates_and_saves_atomically() {
    let d = TempDir::new("set");
    let p = sample(&d, "achromat");
    // Surface ids equal indices in the samples; id 3 is the rear surface.
    let path = r#"{"param":"thickness","surface":3}"#;
    let v = json_run(&["set", s(&p), path, "90"], 0);
    assert!((f(&v["old"]) - 95.0853479).abs() < 1e-6);
    assert_eq!(f(&v["new"]), 90.0);
    assert_eq!(v["changed"], true);
    assert_eq!(v["saved"], true);
    assert_ne!(v["snapshot_hash"], v["snapshot_hash_before"]);
    let show = json_run(&["show", s(&p)], 0);
    assert_eq!(f(&show["surfaces"][3]["thickness"]), 90.0);

    // Dry run and --out leave the project alone.
    let before = std::fs::read(&p).unwrap();
    let v = json_run(&["set", s(&p), path, "80", "--dry-run"], 0);
    assert_eq!(v["saved"], false);
    let out = d.join("copy.mklight");
    json_run(&["set", s(&p), path, "70", "--out", s(&out)], 0);
    assert_eq!(std::fs::read(&p).unwrap(), before);
    let copy = json_run(&["show", s(&out)], 0);
    assert_eq!(f(&copy["surfaces"][3]["thickness"]), 70.0);

    // An edit that makes the model invalid is rejected (2) and not saved.
    let e = json_run(
        &["set", s(&p), r#"{"param":"wavelength_um","index":0}"#, "-1"],
        2,
    );
    assert_eq!(e["error"]["kind"], "invalid_model");
    let e = json_run(&["set", s(&p), r#"{"param":"field_y","index":2}"#, "95"], 2);
    assert!(e["error"]["message"].as_str().unwrap().contains("90"));
    assert_eq!(std::fs::read(&p).unwrap(), before);
    // A value the parameter cannot take: invalid model (2), not a bad path.
    let e = json_run(
        &[
            "set",
            s(&p),
            r#"{"param":"semi_diameter","surface":1}"#,
            "0",
        ],
        2,
    );
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid value"),
        "{e}"
    );
    // Values out of range (an F/# of 0 or -3, a negative weight) are
    // rejected by the engine's structural validation, with its issue code,
    // and not saved.
    for (path, value, code) in [
        (
            r#"{"param":"aperture_value"}"#,
            "0",
            "invalid_aperture_value",
        ),
        (
            r#"{"param":"aperture_value"}"#,
            "-3",
            "invalid_aperture_value",
        ),
        (
            r#"{"param":"field_weight","index":0}"#,
            "-1",
            "negative_weight",
        ),
    ] {
        let e = json_run(&["set", s(&p), path, value], 2);
        assert_eq!(e["error"]["kind"], "invalid_model", "{path} = {value}");
        let msg = e["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("edit rejected") && msg.contains(&format!("[{code}]")),
            "{e}"
        );
    }
    assert_eq!(std::fs::read(&p).unwrap(), before);
    // An edit that keeps the model valid but stops real rays is saved, with
    // the new warning reported.
    let v = json_run(
        &[
            "set",
            s(&p),
            r#"{"param":"tilt_x","surface":1}"#,
            "89",
            "--dry-run",
        ],
        0,
    );
    let w = v["new_warnings"].as_array().unwrap();
    assert!(
        w.iter()
            .any(|w| w.as_str().unwrap().starts_with("marginal_ray")),
        "{v}"
    );
    let v = json_run(&["set", s(&p), path, "85", "--dry-run"], 0);
    assert_eq!(v["new_warnings"], json!([]));
    // A path that does not address a parameter of this system: usage.
    assert_eq!(
        code(&mklight([
            "set",
            s(&p),
            r#"{"param":"conic","surface":4}"#,
            "1"
        ])),
        1
    );
    // Unknown surface, malformed path or value: usage.
    assert_eq!(
        code(&mklight([
            "set",
            s(&p),
            r#"{"param":"thickness","surface":99}"#,
            "1"
        ])),
        1
    );
    assert_eq!(code(&mklight(["set", s(&p), "thickness", "1"])), 1);
    assert_eq!(code(&mklight(["set", s(&p), path, "abc"])), 1);
    // "inf" is accepted where the model allows it: the image-side gap
    // cannot be infinite (only the object may).
    assert_eq!(code(&mklight(["set", s(&p), path, "inf"])), 2);
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

// ---------------------------------------------------------------------------
// catalogs

/// A synthetic catalog: one usable glass (published N-BK7 Sellmeier
/// coefficients under a made-up name) and one with an unsupported formula.
const TEST_AGF: &str = "CC Synthetic MKLight CLI test catalog\n\
NM TESTCROWN 2 517642 1.5168 64.17 0 1 0\n\
ED 7.1 8.3 2.51 -0.0009 0\n\
CD 1.03961212E+000 6.00069867E-003 2.31792344E-001 2.00179144E-002 1.01046945E+000 1.03560653E+002 0 0 0 0\n\
LD 3.0E-001 2.5E+000\n\
NM BADGLASS 99 - 1.5 50 0 0 0\n\
CD 1 2 3\n\
LD 0.3 2.5\n";

#[test]
fn catalogs_list_show_and_import() {
    let d = TempDir::new("catalogs");
    let v = json_run(&["catalogs", "list"], 0);
    let cats = v["catalogs"].as_array().unwrap();
    assert_eq!(cats.len(), 1);
    assert_eq!(cats[0]["name"], "BUILTIN");
    assert_eq!(cats[0]["source"], "builtin");
    let b = json_run(&["catalogs", "show", "builtin"], 0);
    let glasses = b["glasses"].as_array().unwrap();
    assert_eq!(glasses.len(), f(&cats[0]["glass_count"]) as usize);
    let bk7 = glasses.iter().find(|g| g["name"] == "N-BK7").unwrap();
    assert!((f(&bk7["nd"]) - 1.5168).abs() < 1e-4);
    let names: Vec<&str> = glasses
        .iter()
        .map(|g| g["name"].as_str().unwrap())
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "deterministic order");
    assert_eq!(code(&mklight(["catalogs", "show", "nope"])), 1);

    let p = sample(&d, "singlet");
    let agf = d.join("mycat.agf");
    std::fs::write(&agf, TEST_AGF).unwrap();
    let v = json_run(&["catalogs", "import", s(&agf), "--project", s(&p)], 0);
    assert_eq!(v["catalog"]["name"], "mycat");
    assert_eq!(v["catalog"]["glass_count"], 1);
    assert!(
        v["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("BADGLASS"))
    );
    // The project records a reference, not the data.
    let doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    assert_eq!(doc["catalogs"][0]["name"], "mycat");
    assert_eq!(doc["catalogs"][0]["relative_path"], "mycat.agf");
    assert!(
        !String::from_utf8(std::fs::read(&p).unwrap())
            .unwrap()
            .contains("1.03961212")
    );

    let l = json_run(&["catalogs", "list", "--project", s(&p)], 0);
    assert_eq!(l["catalogs"].as_array().unwrap().len(), 2);
    let m = json_run(&["catalogs", "show", "MYCAT", "--project", s(&p)], 0);
    assert_eq!(m["glasses"][0]["name"], "TESTCROWN");

    // Cross-check: the user glass has N-BK7's coefficients, so a singlet
    // made of it has the same first-order properties.
    let reference = json_run(&["paraxial", s(&p)], 0);
    let mut doc = doc;
    doc["system"]["surfaces"][1]["material"] =
        json!({"type": "catalog", "catalog": "mycat", "name": "TESTCROWN"});
    std::fs::write(&p, doc.to_string()).unwrap();
    assert_eq!(json_run(&["validate", s(&p)], 0)["ok"], true);
    let user = json_run(&["paraxial", s(&p)], 0);
    for k in 0..3 {
        let (a, b) = (
            f(&reference["wavelengths"][k]["efl"]),
            f(&user["wavelengths"][k]["efl"]),
        );
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
    // Import of a missing file: I/O; of a file without usable glasses: usage.
    let e = json_run(
        &[
            "catalogs",
            "import",
            s(&d.join("none.agf")),
            "--project",
            s(&p),
        ],
        4,
    );
    assert_eq!(e["error"]["kind"], "io");
    let bad = d.join("bad.agf");
    std::fs::write(&bad, "NM X 99 - 1 1\n").unwrap();
    assert_eq!(
        code(&mklight([
            "catalogs",
            "import",
            s(&bad),
            "--project",
            s(&p)
        ])),
        1
    );
}

// ---------------------------------------------------------------------------
// paraxial and trace

#[test]
fn paraxial_efl_of_the_achromat_is_100mm() {
    let d = TempDir::new("paraxial");
    let p = sample(&d, "achromat");
    let v = json_run(&["paraxial", s(&p)], 0);
    let primary = v["primary"].as_u64().unwrap() as usize;
    let fo = &v["wavelengths"][primary];
    assert!((f(&fo["wavelength_um"]) - 0.5875618).abs() < 1e-9);
    assert!((f(&fo["efl"]) - 100.0).abs() < 0.05, "{}", fo["efl"]);
    assert!((f(&fo["image_f_number"]) - 5.0).abs() < 1e-9);
    // The image surface sits at the paraxial focus of the primary wavelength.
    assert!((f(&fo["image_distance"]) - 95.0853479).abs() < 1e-6);
    // Achromat: F and C focal lengths agree to better than 0.1 %.
    let (ef, ec) = (
        f(&v["wavelengths"][0]["efl"]),
        f(&v["wavelengths"][2]["efl"]),
    );
    assert!((ef - ec).abs() < 0.1);
    // Paraxial marginal ray ends at the axis on the image surface.
    let y = v["marginal_ray"]["y"].as_array().unwrap();
    assert!(f(y.last().unwrap()).abs() < 1e-6);
    // The object distance is infinite: the front focal length is finite,
    // the entrance pupil is at the first surface (stop = surface 1).
    assert!(f(&fo["entrance_pupil_position"]).abs() < 1e-12);
    let text = stdout(&mklight(["paraxial", s(&p), "--rays"]));
    assert!(text.contains("effective focal length") && text.contains("marginal y"));
}

#[test]
fn axial_marginal_ray_lands_near_focus() {
    let d = TempDir::new("trace");
    let p = sample(&d, "achromat");
    let par = json_run(&["paraxial", s(&p)], 0);
    let epd = f(&par["wavelengths"][1]["entrance_pupil_diameter"]);
    let v = json_run(
        &[
            "trace",
            s(&p),
            "--field",
            "0",
            "--wavelength",
            "1",
            "--pupil",
            "0,1",
            "--surfaces",
        ],
        0,
    );
    assert_eq!(v["status"], "ok");
    let img = &v["image"]["point"];
    // Corrected doublet at F/5: transverse spherical aberration of a few µm.
    assert!(f(&img[1]).abs() < 0.01, "image y = {}", img[1]);
    assert!(f(&img[0]).abs() < 1e-12);
    assert!(f(&img[2]).abs() < 1e-9, "on the image surface");
    // The ray converges toward the axis with slope ≈ −1/(2 F/#) = −0.1.
    assert!((f(&v["image"]["dir"][1]) + 0.1).abs() < 0.005);
    let surfaces = v["surfaces"].as_array().unwrap();
    assert_eq!(surfaces.len(), 5);
    // It starts at the edge of the entrance pupil (at the first surface).
    assert!((f(&surfaces[0]["point"][1]) - epd / 2.0).abs() < 1e-9);
    // OPL is cumulative and ends at the reported total.
    let opl: Vec<f64> = surfaces.iter().map(|r| f(&r["opl_mm"])).collect();
    assert!(opl.windows(2).all(|w| w[1] > w[0]));
    assert!((opl[4] - f(&v["opl_mm"])).abs() < 1e-9);
    // Transverse aberration against the chief ray (on axis: the axis).
    assert!((f(&v["transverse_aberration"][1]) - f(&img[1])).abs() < 1e-12);

    // Default wavelength is the primary; same result.
    let w = json_run(&["trace", s(&p), "--pupil", "0,1"], 0);
    assert_eq!(w["image"], v["image"]);
    assert!(w.get("surfaces").is_none());
}

#[test]
fn chief_ray_matches_paraxial_image_height() {
    let d = TempDir::new("chief");
    let p = sample(&d, "achromat");
    let par = json_run(&["paraxial", s(&p)], 0);
    let h = f(&par["wavelengths"][1]["paraxial_image_height"]);
    let v = json_run(&["trace", s(&p), "--field", "2", "--pupil", "0,0"], 0);
    assert_eq!(v["field"]["hy"], 1.0);
    let y = f(&v["image"]["point"][1]);
    // 2° field of a thin doublet: distortion well below 0.1 %.
    assert!((y - h).abs() / h < 1e-3, "{y} vs {h}");
    // Off-axis marginal ray: its transverse aberration is measured from the
    // chief ray of a separate trace (coma and field curvature at full aperture: tens of µm).
    let m = json_run(&["trace", s(&p), "--field", "2", "--pupil", "0,1"], 0);
    let ey = f(&m["transverse_aberration"][1]);
    assert!(
        (ey - (f(&m["image"]["point"][1]) - y)).abs() < 1e-12,
        "ray - chief of an independent chief trace"
    );
    assert!(ey.abs() > 1e-3 && ey.abs() < 0.2, "{ey}");
    // A negative pupil coordinate is accepted.
    let n = json_run(&["trace", s(&p), "--field", "2", "--pupil", "0,-1"], 0);
    assert_eq!(n["status"], "ok");
}

#[test]
fn trace_reports_clipping_and_usage_errors() {
    let d = TempDir::new("clip");
    let p = sample(&d, "achromat");
    // Fix the rear semi-diameter below the beam: the marginal ray is clipped.
    json_run(
        &[
            "set",
            s(&p),
            r#"{"param":"semi_diameter","surface":3}"#,
            "5",
        ],
        0,
    );
    let v = json_run(&["trace", s(&p), "--pupil", "0,1"], 0);
    assert_eq!(v["status"], "clipped");
    assert_eq!(v["stopped_at_surface"], 3);
    assert_eq!(v["image"], Value::Null);
    let u = json_run(&["trace", s(&p), "--pupil", "0,1", "--no-clip"], 0);
    assert_eq!(u["status"], "ok");

    assert_eq!(code(&mklight(["trace", s(&p), "--field", "9"])), 1);
    assert_eq!(code(&mklight(["trace", s(&p), "--wavelength", "7"])), 1);
    assert_eq!(code(&mklight(["trace", s(&p), "--pupil", "1"])), 1);
    assert_eq!(code(&mklight(["trace", s(&p), "--pupil", "a,b"])), 1);
    // The normalised pupil is the unit disk.
    assert_eq!(code(&mklight(["trace", s(&p), "--pupil", "3,0"])), 1);
    assert_eq!(code(&mklight(["trace", s(&p), "--pupil", "0.8,0.8"])), 1);
}

/// Known first-order answers of the samples (primary wavelength): EFL with
/// tolerance, and the bound on the axial marginal ray's image height
/// (spherical aberration at full aperture).
const FIRST_ORDER: &[(&str, f64, f64, f64)] = &[
    ("singlet", 100.0, 1.0, 0.2),
    ("achromat", 100.0, 0.05, 0.01),
    ("cooke_triplet", 50.0, 0.05, 0.05),
    ("double_gauss", 50.0, 0.05, 0.05),
    ("newtonian", 500.0, 1e-9, 1e-9),
    ("petzval_asphere", 69.0, 0.5, 0.01),
    ("relay", 55.5, 0.1, 0.01),
];

#[test]
fn every_sample_has_first_order_data_and_traces() {
    let d = TempDir::new("all-samples");
    assert_eq!(FIRST_ORDER.len(), SAMPLES.len());
    for &(name, efl, tol, sa) in FIRST_ORDER {
        let p = sample(&d, name);
        let par = json_run(&["paraxial", s(&p)], 0);
        let fo = &par["wavelengths"][par["primary"].as_u64().unwrap() as usize];
        assert!(
            (f(&fo["efl"]) - efl).abs() < tol,
            "{name}: EFL {}",
            fo["efl"]
        );
        // The image surface is at the paraxial focus: the paraxial marginal
        // ray ends near the axis.
        let y = par["marginal_ray"]["y"].as_array().unwrap();
        assert!(
            f(y.last().unwrap()).abs() < 0.1,
            "{name}: {}",
            y.last().unwrap()
        );
        // Axial marginal ray (field 0 is on axis in every sample).
        let v = json_run(&["trace", s(&p), "--pupil", "0,1", "--surfaces"], 0);
        assert_eq!(v["status"], "ok", "{name}");
        assert_eq!(v["field"]["hy"], 0.0);
        let img = &v["image"]["point"];
        assert!(f(&img[1]).abs() < sa, "{name}: marginal y {}", img[1]);
        assert!(
            f(&img[0]).abs() < 1e-12 && f(&img[2]).abs() < 1e-9,
            "{name}"
        );
        // Sine condition (well corrected at the primary wavelength): the
        // real image-space slope matches 1/(2 working F/#) within 2 %.
        let m = -f(&v["image"]["dir"][1]);
        let na = 1.0 / (2.0 * f(&fo["working_f_number"]));
        assert!((m - na).abs() / na < 0.02, "{name}: sin u' {m} vs {na}");
        assert_eq!(
            v["surfaces"].as_array().unwrap().len(),
            show_len(&p),
            "{name}: one record per surface"
        );
        // Every other ray-based entry point runs on it too.
        json_run(&["run", s(&p), "system_summary"], 0);
        let edge = json_run(&["trace", s(&p), "--field", "2", "--pupil", "0,0"], 0);
        assert_eq!(edge["status"], "ok", "{name}: edge chief ray");
    }

    // Newtonian: an f/5 paraboloid is free of spherical aberration on axis,
    // and the 45° fold sends the image to global (0, -100, -400): 400 mm
    // back along the axis to the flat, then 100 mm sideways.
    let p = d.join("newtonian.mklight");
    for pupil in ["0,0", "0,1", "-0.6,0.8"] {
        let v = json_run(&["trace", s(&p), "--pupil", pupil, "--surfaces"], 0);
        let g = &v["surfaces"].as_array().unwrap().last().unwrap()["global_point"];
        for (k, want) in [0.0, -100.0, -400.0].iter().enumerate() {
            assert!((f(&g[k]) - want).abs() < 1e-9, "pupil {pupil}: {g}");
        }
    }
    // Relay: 1:1 imaging (m = -1, working F/# = 1/(2 NA)), the edge field
    // (object height 5 mm) is imaged at -5 mm.
    let p = d.join("relay.mklight");
    let par = json_run(&["paraxial", s(&p)], 0);
    let fo = &par["wavelengths"][1];
    assert!((f(&fo["magnification"]) + 1.0).abs() < 1e-6);
    assert!((f(&fo["object_na"]) - 0.08).abs() < 1e-12);
    assert!((f(&fo["working_f_number"]) - 1.0 / (2.0 * 0.08)).abs() < 0.05);
    let v = json_run(&["trace", s(&p), "--field", "2", "--pupil", "0,0"], 0);
    assert!(
        (f(&v["image"]["point"][1]) + 5.0).abs() < 1e-4,
        "{}",
        v["image"]
    );
}

/// Number of surfaces of a project (from `show`).
fn show_len(p: &Path) -> usize {
    json_run(&["show", s(p), "--no-semi-diameters"], 0)["surfaces"]
        .as_array()
        .unwrap()
        .len()
}

#[test]
fn coordinate_breaks_are_not_reported_as_hits() {
    let d = TempDir::new("cb");
    let p = sample(&d, "newtonian");
    let v = json_run(&["trace", s(&p), "--pupil", "0,0", "--surfaces"], 0);
    let r = v["surfaces"].as_array().unwrap();
    let cbs: Vec<usize> = r
        .iter()
        .filter(|x| x["is_coordinate_break"] == true)
        .map(|x| x["index"].as_u64().unwrap() as usize)
        .collect();
    assert_eq!(cbs, [2, 4]);
    for x in r {
        let i = x["index"].as_u64().unwrap();
        if i == 0 || x["is_coordinate_break"] == true {
            assert_eq!(x["aoi_deg"], Value::Null, "{x}");
            assert_eq!(x["normal"], Value::Null, "{x}");
        } else {
            assert!(x["aoi_deg"].is_number() && x["normal"].is_array(), "{x}");
        }
    }
    assert_eq!(r[2]["type"], "coordinate_break");
    assert_eq!(r[3]["type"], "plane");
    // The chief ray meets the fold flat at 45°, the primary at normal
    // incidence.
    assert!((f(&r[3]["aoi_deg"]) - 45.0).abs() < 1e-9);
    assert!(f(&r[1]["aoi_deg"]).abs() < 1e-9);
    let text = stdout(&mklight(["trace", s(&p), "--pupil", "0,1", "--surfaces"]));
    let row = text
        .lines()
        .find(|l| l.trim_start().starts_with("2 ") && l.contains("fold in"))
        .unwrap();
    assert!(row.split_whitespace().nth(9) == Some("-"), "{row}");
    assert!(text.contains("coordinate breaks"), "{text}");
}

#[test]
fn validate_agrees_with_the_ray_commands() {
    let d = TempDir::new("validate-rays");
    let p = sample(&d, "achromat");
    let doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    // A file edited by hand to F/# 0 and to a negative field weight: the
    // engine's structural validation refuses to open it, so paraxial and
    // trace fail, and validate lists every structural issue (code, JSON
    // path, surface) instead of a single load error.
    let mut bad = doc.clone();
    bad["system"]["aperture"]["value"] = json!(0.0);
    bad["system"]["fields"]["fields"][1]["weight"] = json!(-1.0);
    let b = d.join("bad.mklight");
    std::fs::write(&b, bad.to_string()).unwrap();
    let v = json_run(&["validate", s(&b)], 2);
    assert_eq!(v["ok"], false, "{v}");
    assert_eq!(v["structural_only"], true);
    assert_eq!(f(&v["errors"]), 2.0);
    assert_eq!(v["snapshot_hash"], Value::Null);
    let issues: Vec<(&str, &str)> = v["issues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| (i["code"].as_str().unwrap(), i["path"].as_str().unwrap()))
        .collect();
    assert_eq!(
        issues,
        [
            ("invalid_aperture_value", "aperture.value"),
            ("negative_weight", "fields.fields[1].weight"),
        ]
    );
    let text = stdout(&mklight(["validate", s(&b)]));
    assert!(
        text.contains("aperture.value") && text.contains("fields.fields[1].weight"),
        "{text}"
    );
    assert_eq!(code(&mklight(["paraxial", s(&b)])), 2);
    assert_eq!(code(&mklight(["trace", s(&b), "--pupil", "0,1"])), 2);
    // The broken file can be repaired step by step with set: an edit that
    // adds no structural issue is saved (also one that does not repair
    // anything), one that adds an issue is refused and not written.
    let before = std::fs::read(&b).unwrap();
    let e = json_run(
        &[
            "set",
            s(&b),
            r#"{"param":"wavelength_weight","index":0}"#,
            "-2",
        ],
        2,
    );
    let msg = e["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("[negative_weight] wavelengths.wavelengths[0].weight"),
        "{msg}"
    );
    assert_eq!(std::fs::read(&b).unwrap(), before);
    let r = json_run(
        &["set", s(&b), r#"{"param":"thickness","surface":3}"#, "90"],
        0,
    );
    assert_eq!(
        (r["saved"].clone(), r["valid"].clone()),
        (json!(true), json!(false))
    );
    assert_eq!(r["remaining_issues"].as_array().unwrap().len(), 2);
    let r = json_run(&["set", s(&b), r#"{"param":"aperture_value"}"#, "-1"], 0);
    assert_eq!(f(&r["old"]), 0.0);
    assert_eq!(r["remaining_issues"].as_array().unwrap().len(), 2);
    let r = json_run(&["set", s(&b), r#"{"param":"aperture_value"}"#, "5"], 0);
    assert_eq!(r["valid"], false);
    assert_eq!(r["remaining_issues"][0]["code"], "negative_weight");
    // A dry run of the last repair reports a valid model and writes nothing.
    let before = std::fs::read(&b).unwrap();
    let path_w = r#"{"param":"field_weight","index":1}"#;
    let r = json_run(&["set", s(&b), path_w, "1", "--dry-run"], 0);
    assert_eq!(
        (r["valid"].clone(), r["saved"].clone()),
        (json!(true), json!(false))
    );
    assert!(r["snapshot_hash"].is_string());
    assert_eq!(std::fs::read(&b).unwrap(), before);
    let r = json_run(&["set", s(&b), path_w, "1"], 0);
    assert_eq!(r["valid"], true);
    assert!(r["snapshot_hash"].is_string());
    let v = json_run(&["validate", s(&b)], 0);
    assert_eq!(
        (v["ok"].clone(), v["structural_only"].clone()),
        (json!(true), json!(false))
    );
    assert_eq!(code(&mklight(["paraxial", s(&b)])), 0);
    let efl = |p: &Path| f(&json_run(&["paraxial", s(p)], 0)["wavelengths"][0]["efl"]);
    assert_eq!(efl(&b), efl(&p));
    // A file that is not a project at all keeps its load error.
    std::fs::write(&b, "{\"format\": \"x\"}").unwrap();
    let e = json_run(&["validate", s(&b)], 2);
    assert!(e["error"]["message"].is_string(), "{e}");
    assert_eq!(json_run(&["validate", s(&p)], 0)["ok"], true);
    assert_eq!(code(&mklight(["paraxial", s(&p)])), 0);
}

#[test]
fn non_utf8_projects_are_invalid_models_for_every_command() {
    let d = TempDir::new("utf8");
    let b = d.join("bin.mklight");
    std::fs::write(&b, [0xff, 0xfe, 0x00]).unwrap();
    let cases: &[&[&str]] = &[
        &["validate", s(&b)],
        &["show", s(&b)],
        &["paraxial", s(&b)],
        &["trace", s(&b)],
        &["migrate", s(&b)],
        &["run", s(&b), "validate"],
        &["set", s(&b), r#"{"param":"aperture_value"}"#, "4"],
        &["catalogs", "list", "--project", s(&b)],
    ];
    for args in cases {
        let e = json_run(args, 2);
        assert_eq!(e["error"]["kind"], "invalid_model", "{args:?}");
        assert!(
            e["error"]["message"].as_str().unwrap().contains("UTF-8"),
            "{args:?}: {e}"
        );
    }
    assert_eq!(std::fs::read(&b).unwrap(), [0xff, 0xfe, 0x00]);
}

// ---------------------------------------------------------------------------
// analyses and run

#[test]
fn analyses_lists_the_registry() {
    let v = json_run(&["analyses"], 0);
    let list = v["analyses"].as_array().unwrap();
    let names: Vec<&str> = list.iter().map(|a| a["name"].as_str().unwrap()).collect();
    for n in ["validate", "system_summary", "glass_info", "rng_stream"] {
        assert!(names.contains(&n), "{n} missing from {names:?}");
    }
    let rng = list.iter().find(|a| a["name"] == "rng_stream").unwrap();
    assert_eq!(rng["available"], true);
    assert_eq!(rng["default_settings"]["count"], 100000);
    assert!(rng["settings_schema"]["properties"]["seed"].is_object());
    // Reserved names are listed as unavailable with a reason.
    for a in list.iter().filter(|a| a["available"] == false) {
        assert!(a["unavailable_reason"].is_string());
    }
    let one = json_run(&["analyses", "glass_info"], 0);
    assert_eq!(one["name"], "glass_info");
    assert_eq!(one["default_settings"], Value::Null, "name is required");
    assert_eq!(code(&mklight(["analyses", "nope"])), 1);
    let text = stdout(&mklight(["analyses"]));
    assert!(text.contains("rng_stream") && text.contains("count=100000"));
}

#[test]
fn run_produces_a_result_envelope_for_every_available_analysis() {
    let d = TempDir::new("run");
    let p = sample(&d, "cooke_triplet");
    let list = json_run(&["analyses"], 0);
    for a in list["analyses"].as_array().unwrap() {
        let name = a["name"].as_str().unwrap();
        let settings = if name == "glass_info" {
            r#"{"name":"N-SK16"}"#
        } else if name == "glass_substitution" {
            r#"{"material_variables":[{"surface":1,"candidates":{"nearest":2}}],"max_combinations":2,"local_iterations":2}"#
        } else {
            "{}"
        };
        let exit = if a["available"] == true { 0 } else { 3 };
        let v = json_run(&["run", s(&p), name, "--settings", settings], exit);
        if exit == 0 {
            assert_eq!(v["run_info"]["analysis"], name);
            // `--backend auto` (the default): a GPU only for analyses with a
            // GPU path, when this machine has one that qualifies.
            let b = v["run_info"]["backend"].as_str().unwrap();
            if a["gpu_path"] == true {
                assert!(b == "cpu_reference" || b.starts_with("gpu:"), "{name}: {b}");
            } else {
                assert_eq!(b, "cpu_reference", "{name}");
            }
            assert_eq!(v["backend"]["used"], b);
            assert_eq!(v["run_info"]["snapshot_hash"].as_str().unwrap().len(), 64);
            assert_eq!(v["library_hash"].as_str().unwrap().len(), 64);
            assert!(v["run_info"]["duration_ms"].is_number());
            assert!(v["result"].is_object());
        } else {
            assert_eq!(v["error"]["kind"], "analysis_failed");
        }
    }
    let v = json_run(
        &[
            "run",
            s(&p),
            "glass_info",
            "--settings",
            r#"{"name":"N-SK16"}"#,
        ],
        0,
    );
    assert!((f(&v["result"]["nd"]) - 1.62041).abs() < 1e-4);
}

#[test]
fn run_settings_seed_buffers_and_out() {
    let d = TempDir::new("runopts");
    let p = sample(&d, "singlet");
    let v = json_run(
        &[
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            r#"{"count": 16}"#,
            "--seed",
            "7",
            "--buffers",
        ],
        0,
    );
    assert_eq!(v["run_info"]["seed"], 7);
    assert_eq!(v["run_info"]["settings"]["seed"], 7);
    assert_eq!(v["run_info"]["settings"]["count"], 16);
    let data = v["buffers"][0]["data"].as_array().unwrap();
    assert_eq!(data.len(), 16);
    let first: Vec<f64> = v["result"]["first"]
        .as_array()
        .unwrap()
        .iter()
        .map(f)
        .collect();
    assert_eq!(first, data[..4].iter().map(f).collect::<Vec<_>>());
    let mean = data.iter().map(f).sum::<f64>() / 16.0;
    assert!((mean - f(&v["result"]["mean"])).abs() < 1e-12);
    // Another seed, another stream.
    let w = json_run(
        &[
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            r#"{"count": 16}"#,
            "--seed",
            "8",
        ],
        0,
    );
    assert_ne!(w["result"]["first"], v["result"]["first"]);
    // Settings from a file, result to a file.
    let sf = d.join("settings.json");
    std::fs::write(&sf, r#"{"count": 5, "stream": 2}"#).unwrap();
    let out = d.join("result.json");
    let at = format!("@{}", s(&sf));
    let r = json_run(
        &[
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            &at,
            "--out",
            s(&out),
        ],
        0,
    );
    assert_eq!(r["out"], s(&out));
    let file: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(file["result"]["count"], 5);
    assert_eq!(file["run_info"]["settings"]["stream"], 2);
    // Errors.
    let e = json_run(&["run", s(&p), "nope"], 1);
    assert_eq!(e["error"]["kind"], "usage");
    json_run(
        &["run", s(&p), "rng_stream", "--settings", r#"{"bogus":1}"#],
        1,
    );
    json_run(&["run", s(&p), "rng_stream", "--settings", "{"], 1);
    json_run(
        &[
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            "@/no/such/file.json",
        ],
        4,
    );
    json_run(&["run", s(&p), "system_summary", "--seed", "1"], 1);
    json_run(&["run", s(&p), "glass_info"], 1);
    json_run(
        &[
            "run",
            s(&p),
            "glass_info",
            "--settings",
            r#"{"name":"NOPE"}"#,
        ],
        2,
    );
    // Explicit Metal request: fails unless a Metal backend that implements
    // the analysis is registered; `auto` and `cpu` always run.
    let metal = mklight(["run", s(&p), "rng_stream", "--backend", "metal"]);
    assert!(code(&metal) == 0 || code(&metal) == 3);
    for b in ["cpu", "auto"] {
        json_run(&["run", s(&p), "rng_stream", "--backend", b], 0);
    }
}

#[test]
fn json_output_is_deterministic_without_timing() {
    let d = TempDir::new("determinism");
    let p = sample(&d, "double_gauss");
    let cases: &[&[&str]] = &[
        &["run", s(&p), "rng_stream", "--seed", "42", "--buffers"],
        &["run", s(&p), "system_summary"],
        &["run", s(&p), "validate"],
        &["show", s(&p)],
        &["paraxial", s(&p)],
        &[
            "trace",
            s(&p),
            "--field",
            "2",
            "--pupil",
            "0.3,-0.7",
            "--surfaces",
        ],
        &["analyses"],
        &["info"],
    ];
    for args in cases {
        let mut a = args.to_vec();
        a.extend(["--json", "--no-timing"]);
        let x = mklight(&a);
        let y = mklight(&a);
        assert_eq!(code(&x), 0, "{args:?}: {}", stderr(&x));
        assert_eq!(x.stdout, y.stdout, "{args:?} is not deterministic");
        for k in ["started_unix_ms", "duration_ms", "elapsed_ms"] {
            assert!(!stdout(&x).contains(k), "{args:?} still has {k}");
        }
    }
    // Timing fields are present by default.
    let t = json_run(&["run", s(&p), "validate"], 0);
    assert!(t["run_info"]["started_unix_ms"].is_number());
    // --out honours --no-timing too.
    let out = d.join("r.json");
    let o = mklight(["run", s(&p), "validate", "--out", s(&out), "--no-timing"]);
    assert_eq!(code(&o), 0);
    let file: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert!(file["run_info"].get("duration_ms").is_none());
}

// ---------------------------------------------------------------------------
// cancellation

/// rng_stream settings for a job that takes ~4 s: 400 chunks of 20 ms.
const SLOW: &str = r#"{"count": 4000, "chunk": 10, "delay_ms": 10}"#;

#[test]
fn timeout_cancels_the_job_and_reports_partial_status() {
    let d = TempDir::new("timeout");
    let p = sample(&d, "singlet");
    let t0 = Instant::now();
    let v = json_run(
        &[
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            SLOW,
            "--timeout",
            "0.3",
        ],
        130,
    );
    assert!(t0.elapsed() < Duration::from_secs(3), "stopped early");
    assert_eq!(v["cancelled"], true);
    assert_eq!(v["reason"], "timeout");
    assert_eq!(v["status"]["state"], "cancelled");
    let progress = f(&v["status"]["progress"]);
    assert!(
        progress > 0.0 && progress < 1.0,
        "partial progress {progress}"
    );
    // Human mode: the partial status goes to stderr, stdout stays empty.
    let o = mklight([
        "run",
        s(&p),
        "rng_stream",
        "--settings",
        SLOW,
        "--timeout",
        "0.2",
    ]);
    assert_eq!(code(&o), 130);
    assert!(stdout(&o).is_empty());
    assert!(stderr(&o).contains("cancelled (timeout)"), "{}", stderr(&o));
    assert_eq!(
        code(&mklight(["run", s(&p), "rng_stream", "--timeout", "-1"])),
        1
    );
}

#[cfg(unix)]
#[test]
fn ctrl_c_cancels_the_job_cleanly() {
    use std::io::{BufRead, BufReader};
    let d = TempDir::new("sigint");
    let p = sample(&d, "singlet");
    let mut child = Command::new(BIN)
        .args([
            "run",
            s(&p),
            "rng_stream",
            "--settings",
            SLOW,
            "--progress",
            "--json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    // Wait for the first progress report, so the job is running.
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    err.read_line(&mut line).unwrap();
    let ev: Value = serde_json::from_str(line.trim()).expect("JSON progress line");
    assert_eq!(ev["state"], "running");
    assert!(f(&ev["progress"]) > 0.0);
    let pid = child.id().to_string();
    assert!(
        Command::new("kill")
            .args(["-INT", &pid])
            .status()
            .unwrap()
            .success()
    );
    let out = child.wait_with_output().unwrap();
    assert!(t0.elapsed() < Duration::from_secs(3), "cancelled promptly");
    assert_eq!(out.status.code(), Some(130));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["cancelled"], true);
    assert_eq!(v["reason"], "interrupt");
    assert_eq!(v["status"]["state"], "cancelled");
    let progress = f(&v["status"]["progress"]);
    assert!(progress > 0.0 && progress < 1.0);
    // Progress reported on stderr is monotone.
    let mut last = f(&ev["progress"]);
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut err, &mut rest).unwrap();
    for l in rest.lines().filter(|l| l.starts_with('{')) {
        let x = f(&serde_json::from_str::<Value>(l).unwrap()["progress"]);
        assert!(x >= last);
        last = x;
    }
}

#[test]
fn progress_is_reported_on_stderr() {
    let d = TempDir::new("progress");
    let p = sample(&d, "singlet");
    let o = mklight([
        "run",
        s(&p),
        "rng_stream",
        "--settings",
        r#"{"count": 200, "chunk": 10, "delay_ms": 10}"#,
        "--progress",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let err = stderr(&o);
    let lines: Vec<&str> = err.lines().filter(|l| l.starts_with("progress")).collect();
    assert!(lines.len() >= 2, "{err}");
    assert!(stdout(&o).contains("result:"));
}

#[cfg(unix)]
#[test]
fn ctrl_c_stops_optimization_and_second_signal_cancels() {
    for signals in [1, 2] {
        use std::io::{BufRead, BufReader};
        let dir = TempDir::new("stop-optim");
        let path = sample(&dir, "singlet");
        let mut child = Command::new(BIN)
            .args([
                "run",
                s(&path),
                "optimize",
                "--backend",
                "cpu",
                "--settings",
                r#"{"algorithm":"de","max_iterations":100000,"options":{"population":32}}"#,
                "--progress",
                "--json",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stderr = BufReader::new(child.stderr.take().unwrap());
        let mut line = String::new();
        loop {
            assert!(stderr.read_line(&mut line).unwrap() > 0);
            if serde_json::from_str::<Value>(line.trim()).is_ok_and(|v| v["state"] == "running") {
                break;
            }
            line.clear();
        }
        unsafe {
            assert_eq!(libc::kill(child.id() as libc::pid_t, libc::SIGINT), 0);
            if signals == 2 {
                assert_eq!(libc::kill(child.id() as libc::pid_t, libc::SIGTERM), 0);
            }
        }
        let out = child.wait_with_output().unwrap();
        assert_eq!(
            out.status.code(),
            Some(if signals == 1 { 0 } else { 130 }),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result: Value = serde_json::from_slice(&out.stdout).unwrap();
        if signals == 2 {
            assert_eq!(result["cancelled"], true);
            assert_eq!(result["status"]["state"], "cancelled");
            continue;
        }
        assert_eq!(result["result"]["stopped"], true);
        assert_eq!(result["result"]["status"], "stopped");
        assert!(f(&result["result"]["final"]["merit"]) <= f(&result["result"]["start"]["merit"]));
        assert!(result["result"]["system_hash"].is_string());
    }
}
