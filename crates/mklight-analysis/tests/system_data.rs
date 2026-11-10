//! First-order report.

mod common;

use common::*;
use mklight_analysis::system_data::*;
use mklight_analysis::util::WavelengthSelection;
use mklight_core::{MaterialRef, SurfaceShape, samples};

#[test]
fn report_matches_paraxial_data() {
    let t = tracer(samples::cooke_triplet());
    let r = system_data(
        &t,
        &SystemDataSettings {
            wavelength: WavelengthSelection::All,
        },
    )
    .unwrap();
    assert_eq!(r.tables.len(), 3);
    for tb in &r.tables {
        let fo = t.paraxial().at(tb.wavelength);
        assert_eq!(tb.get("efl"), Some(fo.efl));
        assert_eq!(tb.get("exit_pupil_position"), Some(fo.exit_pupil_position));
        assert_eq!(tb.get("max_field"), Some(20.0));
    }
    // The flattened entries are the first selected wavelength's.
    assert_eq!(r.entries, r.tables[0].entries);
    let keys: std::collections::BTreeSet<_> = r.entries.iter().map(|e| e.key.clone()).collect();
    assert_eq!(keys.len(), r.entries.len(), "unique keys");
    // Petzval radius from the report equals the Seidel module's.
    let s = mklight_analysis::seidel::seidel(&t, &Default::default()).unwrap();
    let p = r.tables[1].get("petzval_radius").unwrap();
    assert!((p - s.petzval_radius).abs() < 1e-9 * p.abs());
}

#[test]
fn afocal_values_serialise_as_infinity() {
    // Keplerian telescope of two ideal lenses: EFL and working F/# infinite.
    let sys = infinite_system(
        10.0,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                150.0,
                MaterialRef::Air,
            ),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: 50.0 },
                20.0,
                MaterialRef::Air,
            ),
        ],
        1,
        &[0.0, 1.0],
    );
    let t = tracer(sys);
    let r = system_data(&t, &SystemDataSettings::default()).unwrap();
    let e = r.entries.iter().find(|e| e.key == "efl").unwrap();
    assert!(e.value.is_infinite());
    let j = serde_json::to_value(&r).unwrap();
    let efl = j["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["key"] == "efl")
        .unwrap();
    assert!(efl["value"] == "inf" || efl["value"] == "-inf");
    let back: SystemDataResult = serde_json::from_value(j).unwrap();
    assert!(
        back.entries
            .iter()
            .find(|e| e.key == "efl")
            .unwrap()
            .value
            .is_infinite()
    );
}

#[test]
fn ideal_lens_adds_no_petzval_curvature() {
    // A glass singlet followed by an ideal lens: the report's Petzval radius
    // is the singlet's alone, K/n, and equals the Seidel module's.
    let (r1, r2) = (60.0, -90.0);
    let sys = infinite_system(
        10.0,
        vec![
            surf(
                1,
                SurfaceShape::sphere(r1),
                4.0,
                MaterialRef::glass("N-BK7"),
            ),
            surf(2, SurfaceShape::sphere(r2), 5.0, MaterialRef::Air),
            surf(
                3,
                SurfaceShape::Paraxial { focal_length: 80.0 },
                40.0,
                MaterialRef::Air,
            ),
        ],
        1,
        &[0.0, 2.0],
    );
    let t = tracer(sys);
    let n = t.system().surfaces[1].n_after[0];
    let p = (n - 1.0) * (1.0 / r1 - 1.0 / r2) / n;
    let r = system_data(&t, &SystemDataSettings::default()).unwrap();
    let radius = r
        .entries
        .iter()
        .find(|e| e.key == "petzval_radius")
        .unwrap();
    assert!(
        (radius.value + 1.0 / p).abs() < 1e-9 / p,
        "{} vs {}",
        radius.value,
        -1.0 / p
    );
    let s = mklight_analysis::seidel::seidel(&t, &Default::default()).unwrap();
    assert_eq!(radius.value, s.petzval_radius);
    // Ideal lenses alone: a flat Petzval surface in both.
    let t = tracer(ideal_lens(100.0, 10.0, &[0.0, 5.0]));
    let r = system_data(&t, &SystemDataSettings::default()).unwrap();
    let s = mklight_analysis::seidel::seidel(&t, &Default::default()).unwrap();
    let radius = r
        .entries
        .iter()
        .find(|e| e.key == "petzval_radius")
        .unwrap();
    assert_eq!(radius.value, f64::INFINITY);
    assert_eq!(s.petzval_radius, f64::INFINITY);
}
