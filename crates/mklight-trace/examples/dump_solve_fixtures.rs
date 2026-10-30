//! Writes solve-report fixtures for the macOS app's Codable mirror
//! (`apps/macos/Tests/MKLightModelTests/Fixtures/solve_reports.json`).
//!
//! Run with `cargo run -p mklight-trace --example dump_solve_fixtures [out_dir]`
//! (the model fixtures come from `mklight-core`'s `dump_fixtures`).
//!
//! The file is an array of `{"system", "report"}`: the singlet with a focus
//! solve and an F/# solve, solved as given (`configuration: null`) and per
//! configuration (`configuration: k`), the second configuration overriding a
//! solve-controlled thickness (a warning in its report).

use std::path::PathBuf;

use mklight_core::materials::MaterialLibrary;
use mklight_core::param::ParamPath;
use mklight_core::{Configuration, JobContext, OpticalSystem, Solve, SurfaceId, samples};
use mklight_trace::solves::{SolveReport, SolveSettings, apply_solves_all, apply_solves_with};
use serde::Serialize;

#[derive(Serialize)]
struct Entry {
    system: OpticalSystem,
    report: SolveReport,
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../apps/macos/Tests/MKLightModelTests/Fixtures")
        });
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    let mut sys = samples::singlet();
    sys.solves = vec![
        Solve::marginal_ray_height(SurfaceId(2), 0.0),
        Solve::f_number(SurfaceId(2), 5.5),
    ];
    sys.configurations = vec![
        Configuration::default(),
        Configuration {
            name: "override".into(),
            overrides: vec![(
                ParamPath::Thickness {
                    surface: SurfaceId(2),
                },
                50.0,
            )],
        },
    ];
    let lib = MaterialLibrary::with_builtin();
    let settings = SolveSettings::default();
    let ctx = JobContext::new();
    let mut out = Vec::new();
    let (system, report) = apply_solves_with(&sys, &lib, &settings, &ctx).expect("solve");
    out.push(Entry { system, report });
    for (system, report) in apply_solves_all(&sys, &lib, &settings, &ctx).expect("solve all") {
        out.push(Entry { system, report });
    }
    let mut json = serde_json::to_string_pretty(&out).expect("serialise");
    json.push('\n');
    let path = dir.join("solve_reports.json");
    std::fs::write(&path, json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {}", path.display());
}
