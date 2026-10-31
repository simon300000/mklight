use super::*;
use crate::{OpticalSystem, samples};

fn reference(offset: MaterialOffset) -> MaterialRef {
    MaterialRef::Catalog {
        catalog: None,
        name: "N-BK7".into(),
        offset,
    }
}

#[test]
fn offset_has_known_spectral_indices_and_preserves_partial_dispersion() {
    let lib = MaterialLibrary::with_builtin();
    let env = Environment::default();
    let base = lib.resolve(&MaterialRef::glass("N-BK7"), &env).unwrap();
    let m = lib
        .resolve(
            &reference(MaterialOffset {
                dn_d: 0.002,
                dv_d: -3.0,
            }),
            &env,
        )
        .unwrap();
    // Independent 60-digit evaluation of the published Sellmeier coefficients.
    for (l, expected) in [
        (lines::D, 1.5188000345005885),
        (lines::F, 1.5246724198828),
        (lines::C, 1.5161907682953219),
    ] {
        assert!((m.index(l) - expected).abs() < 2e-15, "{l}: {}", m.index(l));
    }
    let vd =
        |m: &ResolvedMaterial| (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C));
    assert!((vd(&m) - (vd(&base) - 3.0)).abs() < 3e-12);
    let pgf = |m: &ResolvedMaterial| {
        (m.index(lines::G) - m.index(lines::F)) / (m.index(lines::F) - m.index(lines::C))
    };
    assert!((pgf(&m) - pgf(&base)).abs() < 1e-13);
}

#[test]
fn offset_is_anchored_at_each_environment_with_thermal_data() {
    let mut cat = builtin::catalog();
    let g = cat.glasses.get_mut("N-BK7").unwrap();
    g.thermal = Some(ThermalCoefficients {
        d0: 1.2e-6,
        d1: 2e-9,
        d2: 3e-12,
        e0: 2e-8,
        e1: 1e-10,
        lambda_tk_um: 0.17,
        t_ref_c: 25.0,
    });
    let mut lib = MaterialLibrary::new();
    lib.add_catalog(Arc::new(cat));
    for env in [
        Environment {
            temperature_c: 25.0,
            pressure_atm: 1.0,
        },
        Environment {
            temperature_c: -10.0,
            pressure_atm: 0.6,
        },
        Environment {
            temperature_c: 70.0,
            pressure_atm: 1.3,
        },
    ] {
        let base = lib.resolve(&MaterialRef::glass("N-BK7"), &env).unwrap();
        for offset in [
            MaterialOffset {
                dn_d: 0.002,
                dv_d: -3.0,
            },
            MaterialOffset {
                dn_d: -0.001,
                dv_d: 2.0,
            },
        ] {
            let m = lib.resolve(&reference(offset), &env).unwrap();
            assert_eq!(m.index(lines::D), base.index(lines::D) + offset.dn_d);
            let vd = |m: &ResolvedMaterial| {
                (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C))
            };
            assert!((vd(&m) - vd(&base) - offset.dv_d).abs() < 4e-12);
        }
        let zero: MaterialRef =
            serde_json::from_str(r#"{"type":"catalog","name":"N-BK7","offset":{}}"#).unwrap();
        let m = lib.resolve(&zero, &env).unwrap();
        for l in [0.31, lines::F, lines::D, lines::C, 0.9, 2.3] {
            assert_eq!(m.index(l).to_bits(), base.index(l).to_bits());
        }
        assert_eq!(
            serde_json::to_value(&zero).unwrap(),
            serde_json::to_value(MaterialRef::glass("N-BK7")).unwrap()
        );
    }
}

#[test]
fn offsets_round_trip_default_and_reject_unknown_fields() {
    let r: MaterialRef =
        serde_json::from_str(r#"{"type":"catalog","name":"N-BK7","offset":{"dn_d":0.002}}"#)
            .unwrap();
    assert_eq!(
        r,
        reference(MaterialOffset {
            dn_d: 0.002,
            dv_d: 0.0
        })
    );
    let json = serde_json::to_string(&r).unwrap();
    assert_eq!(serde_json::from_str::<MaterialRef>(&json).unwrap(), r);
    for offset in [r#"{"dn":0.002}"#, r#"{"dv_d":null}"#, r#"{"dn_d":"inf"}"#] {
        assert!(
            serde_json::from_str::<MaterialRef>(&format!(
                r#"{{"type":"catalog","name":"N-BK7","offset":{offset}}}"#
            ))
            .is_err()
        );
    }
}

#[test]
fn offset_validation_codes_and_paths() {
    let mut s = samples::singlet();
    for (offset, field, code) in [
        (
            MaterialOffset {
                dn_d: 0.050001,
                dv_d: 0.0,
            },
            "dn_d",
            "invalid_material",
        ),
        (
            MaterialOffset {
                dn_d: -0.050001,
                dv_d: 0.0,
            },
            "dn_d",
            "invalid_material",
        ),
        (
            MaterialOffset {
                dn_d: 0.0,
                dv_d: 10.0001,
            },
            "dv_d",
            "invalid_material",
        ),
        (
            MaterialOffset {
                dn_d: 0.0,
                dv_d: -10.0001,
            },
            "dv_d",
            "invalid_material",
        ),
        (
            MaterialOffset {
                dn_d: f64::NAN,
                dv_d: 0.0,
            },
            "dn_d",
            "non_finite",
        ),
        (
            MaterialOffset {
                dn_d: 0.0,
                dv_d: f64::INFINITY,
            },
            "dv_d",
            "non_finite",
        ),
    ] {
        s.surfaces[1].material = reference(offset);
        let issues = s.validation_issues();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, code);
        assert_eq!(
            issues[0].path,
            format!("surfaces[1].material.offset.{field}")
        );
        assert!(
            MaterialLibrary::with_builtin()
                .resolve(&s.surfaces[1].material, &Default::default())
                .is_err()
        );
    }
    s.surfaces[1].material = reference(MaterialOffset {
        dn_d: -0.05,
        dv_d: 10.0,
    });
    s.validate().unwrap();
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(serde_json::from_str::<OpticalSystem>(&json).unwrap(), s);
}
