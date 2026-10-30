//! Frozen ("golden") results for every sample system: first-order data, RMS
//! spot radii per field and wavelength, Seidel sums and field curvature /
//! distortion. A change in any of them beyond a relative 10⁻⁹ fails here.
//!
//! Regenerate after an intended change with
//!
//! ```text
//! MKLIGHT_UPDATE_GOLDEN=1 cargo test -p mklight-analysis --test golden
//! ```
//!
//! and review the diff of `tests/golden/*.json`.

mod common;

use std::path::PathBuf;

use mklight_analysis::curvature::{FieldCurvatureSettings, field_curvature};
use mklight_analysis::seidel::{SeidelSettings, seidel};
use mklight_analysis::spot::{SpotDiagramSettings, spot_diagram};
use mklight_analysis::system_data::{SystemDataSettings, system_data};
use serde_json::{Value, json};

const REL: f64 = 1e-9;
const ABS: f64 = 1e-12;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// The JSON compared for one sample (no timings or hashes).
fn compute(t: &mklight_trace::SequentialTracer) -> Value {
    let fo = system_data(t, &SystemDataSettings::default()).unwrap();
    let first_order: serde_json::Map<String, Value> = fo
        .entries
        .iter()
        .map(|e| (e.key.clone(), serde_json::to_value(mklight_analysis::util::nf::Num(e.value)).unwrap()))
        .collect();
    let spot = spot_diagram(t, &SpotDiagramSettings {
        include_points: false,
        ..Default::default()
    })
    .unwrap();
    let spots: Vec<Value> = spot
        .fields
        .iter()
        .map(|f| {
            json!({
                "field": f.field,
                "rms_radius_centroid": f.polychromatic.rms_radius_centroid,
                "rms_radius_chief": f.polychromatic.rms_radius_chief,
                "geo_radius_centroid": f.polychromatic.geo_radius_centroid,
                "chief": f.chief,
                "per_wavelength_rms_centroid": f.per_wavelength.iter().map(|w| w.stats.rms_radius_centroid).collect::<Vec<_>>(),
            })
        })
        .collect();
    let s = seidel(t, &SeidelSettings::default()).unwrap();
    let fc = field_curvature(t, &FieldCurvatureSettings {
        field_samples: 6,
        ..Default::default()
    })
    .unwrap();
    let c = fc.curve(t.system().primary_wavelength).unwrap();
    json!({
        "first_order": first_order,
        "spot": spots,
        "seidel": {
            "sums": s.sums,
            "petzval_sum": s.petzval_sum,
            "lagrange_invariant": s.lagrange_invariant,
        },
        "field_curvature": {
            "normalized_field": fc.normalized_field,
            "tangential": serde_json::to_value(&c.tangential.iter().map(|v| mklight_analysis::util::nf::Num(*v)).collect::<Vec<_>>()).unwrap(),
            "sagittal": serde_json::to_value(&c.sagittal.iter().map(|v| mklight_analysis::util::nf::Num(*v)).collect::<Vec<_>>()).unwrap(),
            "distortion_percent": serde_json::to_value(&c.distortion_percent.iter().map(|v| mklight_analysis::util::nf::Num(*v)).collect::<Vec<_>>()).unwrap(),
        },
    })
}

fn compare(path: &str, a: &Value, b: &Value, errs: &mut Vec<String>) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > REL * x.abs().max(y.abs()) + ABS {
                errs.push(format!("{path}: {x:e} (now) vs {y:e} (golden)"));
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                compare(&format!("{path}[{i}]"), p, q, errs);
            }
        }
        (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
            for (k, p) in x {
                match y.get(k) {
                    Some(q) => compare(&format!("{path}.{k}"), p, q, errs),
                    None => errs.push(format!("{path}.{k}: missing in golden")),
                }
            }
        }
        _ if a == b => {}
        _ => errs.push(format!("{path}: {a} (now) vs {b} (golden)")),
    }
}

#[test]
fn golden_results_of_every_sample() {
    let update = std::env::var_os("MKLIGHT_UPDATE_GOLDEN").is_some();
    let mut errs = Vec::new();
    for (name, t) in common::samples() {
        let now = compute(&t);
        let file = golden_dir().join(format!("{name}.json"));
        if update {
            std::fs::create_dir_all(golden_dir()).unwrap();
            std::fs::write(&file, serde_json::to_string_pretty(&now).unwrap() + "\n").unwrap();
            continue;
        }
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("{}: {e} (set MKLIGHT_UPDATE_GOLDEN=1 to create)", file.display()));
        let golden: Value = serde_json::from_str(&text).unwrap();
        compare(name, &now, &golden, &mut errs);
    }
    assert!(errs.is_empty(), "golden mismatches:\n{}", errs.join("\n"));
}
