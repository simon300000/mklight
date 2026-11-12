//! `convert_spec`: fields converted to another kind keep their chief rays,
//! and an aperture converted to another type keeps the paraxial marginal
//! ray; converting back returns the original numbers.

use serde_json::{Value, json};

use mklight_analysis as ma;
use mklight_core::{
    ApertureSpec, FieldKind, JobContext, OpticalSystem, RayAiming, SemiDiameter, samples,
};
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, Engine, EngineConfig};

const KINDS: [FieldKind; 4] = [
    FieldKind::ObjectAngle,
    FieldKind::ObjectHeight,
    FieldKind::ParaxialImageHeight,
    FieldKind::RealImageHeight,
];

fn convert(sys: &OpticalSystem, settings: Value) -> Value {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: mklight_engine::backend::Backend::CpuReference,
        ..Default::default()
    });
    let s = e.session_from_project(Project::new(sys.clone())).unwrap();
    s.run_analysis(
        &AnalysisRequest::new("convert_spec", settings),
        &JobContext::new(),
    )
    .unwrap()
    .json
}

fn with_fields(sys: &OpticalSystem, out: &Value) -> OpticalSystem {
    let mut s = sys.clone();
    s.fields.kind = serde_json::from_value(out["fields"]["kind"].clone()).unwrap();
    let values: Vec<[f64; 2]> = serde_json::from_value(out["fields"]["values"].clone()).unwrap();
    assert_eq!(values.len(), s.fields.fields.len());
    for (f, v) in s.fields.fields.iter_mut().zip(values) {
        (f.x, f.y) = (v[0], v[1]);
    }
    s
}

/// Real chief-ray image heights of every field (primary wavelength).
fn chief_heights(sys: &OpticalSystem) -> Vec<[f64; 2]> {
    let t = ma::tracer_for(sys.clone()).unwrap();
    t.fields()
        .iter()
        .map(|f| {
            assert!(f.is_valid(), "{:?}", f.error);
            t.field_value(&f.source, FieldKind::RealImageHeight)
                .unwrap()
        })
        .collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn systems() -> Vec<(&'static str, OpticalSystem)> {
    let mut aimed = samples::double_gauss();
    aimed.ray_aiming = RayAiming::Real {
        tolerance: RayAiming::DEFAULT_TOLERANCE,
    };
    vec![
        ("cooke", samples::cooke_triplet()),
        ("double gauss, real aiming", aimed),
        ("finite relay", samples::finite_relay()),
    ]
}

#[test]
fn fields_keep_their_chief_rays() {
    for (name, sys) in systems() {
        let reference = chief_heights(&sys);
        let at_inf = ma::tracer_for(sys.clone())
            .unwrap()
            .system()
            .object_at_infinity;
        for kind in KINDS {
            let out = convert(&sys, json!({"field_kind": kind}));
            if at_inf && kind == FieldKind::ObjectHeight {
                assert!(out["fields"].is_null(), "{name}");
                assert!(out["fields_error"].is_string(), "{name}");
                continue;
            }
            assert!(out["fields_error"].is_null(), "{name} {kind:?}: {out}");
            let converted = with_fields(&sys, &out);
            for (a, b) in chief_heights(&converted).iter().zip(&reference) {
                assert!(
                    close(a[0], b[0], 1e-9) && close(a[1], b[1], 1e-9),
                    "{name} {kind:?}: chief ray at {a:?}, was {b:?}"
                );
            }
            // And back to the original kind.
            let back = with_fields(
                &converted,
                &convert(&converted, json!({"field_kind": sys.fields.kind})),
            );
            for (a, b) in back.fields.fields.iter().zip(&sys.fields.fields) {
                assert!(
                    close(a.x, b.x, 1e-9) && close(a.y, b.y, 1e-9),
                    "{name} {kind:?}: round trip ({}, {}) vs ({}, {})",
                    a.x,
                    a.y,
                    b.x,
                    b.y
                );
                assert_eq!(a.weight, b.weight);
            }
        }
    }
}

#[test]
fn aperture_keeps_the_marginal_ray() {
    for (name, sys) in systems() {
        let t = ma::tracer_for(sys.clone()).unwrap();
        let fo = t.paraxial().primary().clone();
        let at_inf = t.system().object_at_infinity;
        for kind in [
            "entrance_pupil_diameter",
            "image_space_f_number",
            "object_space_na",
            "paraxial_working_f_number",
            "float_by_stop_size",
        ] {
            let out = convert(&sys, json!({"aperture": kind}));
            if at_inf && kind == "object_space_na" {
                assert!(out["aperture"].is_null() && out["aperture_error"].is_string());
                continue;
            }
            assert!(out["aperture_error"].is_null(), "{name} {kind}: {out}");
            let mut s = sys.clone();
            s.aperture = serde_json::from_value(out["aperture"].clone()).unwrap();
            if s.aperture == ApertureSpec::FloatByStopSize {
                s.surfaces[s.stop].semi_diameter = SemiDiameter::Fixed {
                    value: out["stop_semi_diameter"].as_f64().unwrap(),
                };
            }
            let f = ma::tracer_for(s).unwrap().paraxial().primary().clone();
            assert!(
                close(f.entrance_pupil_diameter, fo.entrance_pupil_diameter, 1e-9),
                "{name} {kind}: EPD {} vs {}",
                f.entrance_pupil_diameter,
                fo.entrance_pupil_diameter
            );
            assert!(close(f.working_f_number, fo.working_f_number, 1e-9));
        }
    }
}
