use super::*;
use crate::{OpticalSystem, samples};

fn bk7(dn_d: f64, dv_d: f64) -> MaterialRef {
    MaterialRef::Catalog {
        catalog: None,
        name: "N-BK7".into(),
        offset: MaterialOffset { dn_d, dv_d },
    }
}

fn vd(m: &ResolvedMaterial) -> f64 {
    (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C))
}

fn pgf(m: &ResolvedMaterial) -> f64 {
    (m.index(lines::G) - m.index(lines::F)) / (m.index(lines::F) - m.index(lines::C))
}

#[test]
fn offset_has_known_spectral_indices_and_preserves_partial_dispersion() {
    let lib = MaterialLibrary::with_builtin();
    let env = Environment::default();
    let base = lib.resolve(&MaterialRef::glass("N-BK7"), &env).unwrap();
    let m = lib.resolve(&bk7(0.002, -3.0), &env).unwrap();
    // Independent high-precision evaluation of the published Sellmeier coefficients.
    for (l, expected) in [
        (lines::D, 1.5188000345005885),
        (lines::F, 1.5246724198828),
        (lines::C, 1.516190768295322),
    ] {
        assert!((m.index(l) - expected).abs() < 2e-15, "{l}: {}", m.index(l));
    }
    assert!((vd(&m) - (vd(&base) - 3.0)).abs() < 3e-12);
    assert!((pgf(&m) - pgf(&base)).abs() < 1e-13);
}

#[test]
fn offset_is_anchored_at_each_environment_and_zero_is_bit_identical() {
    let mut cat = builtin::catalog();
    cat.glasses.get_mut("N-BK7").unwrap().thermal = Some(ThermalCoefficients {
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
    for (temperature_c, pressure_atm) in [(25.0, 1.0), (-10.0, 0.6), (70.0, 1.3)] {
        let env = Environment {
            temperature_c,
            pressure_atm,
        };
        let base = lib.resolve(&MaterialRef::glass("N-BK7"), &env).unwrap();
        for (dn, dv) in [(0.002, -3.0), (-0.001, 2.0)] {
            let m = lib.resolve(&bk7(dn, dv), &env).unwrap();
            assert_eq!(m.index(lines::D), base.index(lines::D) + dn);
            assert!((vd(&m) - vd(&base) - dv).abs() < 4e-12);
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
fn offsets_round_trip_and_reject_unknown_fields() {
    let r: MaterialRef =
        serde_json::from_str(r#"{"type":"catalog","name":"N-BK7","offset":{"dn_d":0.002}}"#)
            .unwrap();
    assert_eq!(r, bk7(0.002, 0.0));
    let json = serde_json::to_string(&r).unwrap();
    assert_eq!(serde_json::from_str::<MaterialRef>(&json).unwrap(), r);
    for offset in [r#"{"dn":0.002}"#, r#"{"dv_d":null}"#] {
        let json = format!(r#"{{"type":"catalog","name":"N-BK7","offset":{offset}}}"#);
        assert!(serde_json::from_str::<MaterialRef>(&json).is_err());
    }
}

#[test]
fn offset_validation_codes_and_paths() {
    let mut s = samples::singlet();
    for (dn, dv, field, code) in [
        (0.050001, 0.0, "dn_d", "invalid_material"),
        (-0.050001, 0.0, "dn_d", "invalid_material"),
        (0.0, 10.0001, "dv_d", "invalid_material"),
        (0.0, -10.0001, "dv_d", "invalid_material"),
        (f64::NAN, 0.0, "dn_d", "non_finite"),
        (0.0, f64::INFINITY, "dv_d", "non_finite"),
    ] {
        s.surfaces[1].material = bk7(dn, dv);
        let issues = s.validation_issues();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, code);
        assert_eq!(
            issues[0].path,
            format!("surfaces[1].material.offset.{field}")
        );
    }
    s.surfaces[1].material = bk7(-0.05, 10.0);
    s.validate().unwrap();
    let json = serde_json::to_string(&s).unwrap();
    assert_eq!(serde_json::from_str::<OpticalSystem>(&json).unwrap(), s);
    // Resolution still refuses a non-finite offset outside a validated system.
    assert!(
        MaterialLibrary::with_builtin()
            .resolve(&bk7(f64::NAN, 0.0), &Environment::default())
            .is_err()
    );
}
