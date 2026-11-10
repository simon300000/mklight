//! `.zmx` and `.seq` import: structure, encodings and warnings. The
//! first-order round trip against the native samples is in
//! `mklight-trace/tests/prescription_round_trip.rs`.

use std::path::PathBuf;

use mklight_core::import::{
    ImportWarningKind as K, PrescriptionFormat, PrescriptionImport, import_prescription,
    read_prescription_file,
};
use mklight_core::materials::MaterialLibrary;
use mklight_core::materials::agf::{TextEncoding, encode_text};
use mklight_core::model::{
    ApertureSpec, FieldKind, Interaction, MaterialRef, SemiDiameter, SolveKind, SurfaceShape,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prescriptions")
        .join(name)
}

fn import(name: &str) -> PrescriptionImport {
    read_prescription_file(fixture(name), &MaterialLibrary::with_builtin())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn kinds(imp: &PrescriptionImport) -> Vec<K> {
    imp.warnings.iter().map(|w| w.kind).collect()
}

fn builtin(name: &str) -> MaterialRef {
    MaterialRef::Catalog {
        catalog: Some("BUILTIN".into()),
        name: name.into(),
        offset: Default::default(),
    }
}

#[test]
fn zmx_singlet_maps_every_section() {
    let imp = import("singlet.zmx");
    let s = &imp.system;
    assert_eq!(imp.format, PrescriptionFormat::Zmx);
    assert_eq!(imp.encoding, TextEncoding::Utf8);
    assert_eq!(s.name, "Singlet f100 F/5");
    assert_eq!(s.notes, "N-BK7 near-best-form singlet");
    assert_eq!(
        s.aperture,
        ApertureSpec::EntrancePupilDiameter { value: 20.0 }
    );
    assert_eq!(s.surfaces.len(), 4);
    assert_eq!(s.stop, 1);
    assert!(s.surfaces[0].thickness.is_infinite());
    assert_eq!(s.surfaces[1].shape, SurfaceShape::sphere(60.0));
    assert_eq!(s.surfaces[1].material, builtin("N-BK7"));
    assert_eq!(s.surfaces[1].label, "front");
    assert_eq!(s.surfaces[2].thickness, 96.6967287);
    assert_eq!(s.surfaces[2].material, MaterialRef::Air);
    // Clear apertures are computed by MKLight, not fixed.
    assert_eq!(s.surfaces[1].semi_diameter, SemiDiameter::Auto);
    // FTYP says three wavelengths although five WAVM records follow.
    let wls: Vec<f64> = s.wavelengths.wavelengths.iter().map(|w| w.um).collect();
    assert_eq!(wls, [0.4861327, 0.5875618, 0.6562725]);
    assert_eq!(s.wavelengths.primary, 1);
    assert_eq!(s.fields.kind, FieldKind::ObjectAngle);
    let ys: Vec<f64> = s.fields.fields.iter().map(|f| f.y).collect();
    assert_eq!(ys, [0.0, 3.5, 5.0]);
    assert_eq!(s.environment.temperature_c, 20.0);
    // Only the ray-aiming setting does not map.
    assert_eq!(kinds(&imp), [K::Approximated], "{:?}", imp.warnings);
    s.validate().unwrap();
}

#[test]
fn zmx_utf16_with_bom_is_read() {
    let imp = import("cooke_triplet.zmx");
    assert_eq!(imp.encoding, TextEncoding::Utf16Le);
    let s = &imp.system;
    assert_eq!(s.name, "Cooke triplet f50 F/4");
    assert_eq!(s.surfaces.len(), 8);
    assert_eq!(s.stop, 4);
    assert_eq!(s.surfaces[3].material, builtin("F2"));
    assert_eq!(s.surfaces[5].material, builtin("N-SK16"));
    assert_eq!(s.surfaces[6].thickness, 42.252162);
}

#[test]
fn every_encoding_gives_the_same_system() {
    let utf8 = std::fs::read_to_string(fixture("singlet.zmx")).unwrap();
    let lib = MaterialLibrary::with_builtin();
    let reference = import_prescription(utf8.as_bytes(), PrescriptionFormat::Zmx, &lib).unwrap();
    let mut bom8 = vec![0xEF, 0xBB, 0xBF];
    bom8.extend_from_slice(utf8.as_bytes());
    let le = encode_text(&utf8, TextEncoding::Utf16Le);
    let be = encode_text(&utf8, TextEncoding::Utf16Be);
    assert_eq!(&le[..2], [0xFF, 0xFE]);
    assert_eq!(&be[..2], [0xFE, 0xFF]);
    let cases: [(&[u8], TextEncoding); 5] = [
        (&bom8, TextEncoding::Utf8Bom),
        (&le, TextEncoding::Utf16Le),
        (&be, TextEncoding::Utf16Be),
        (&le[2..], TextEncoding::Utf16Le), // no BOM
        (&be[2..], TextEncoding::Utf16Be), // no BOM
    ];
    for (bytes, encoding) in cases {
        let imp = import_prescription(bytes, PrescriptionFormat::Zmx, &lib).unwrap();
        assert_eq!(imp.encoding, encoding);
        assert_eq!(imp.system, reference.system, "{encoding:?}");
        assert_eq!(imp.warnings, reference.warnings, "{encoding:?}");
    }
}

#[test]
fn zmx_even_asphere_and_image_height_fields() {
    let imp = import("petzval_asphere.zmx");
    let s = &imp.system;
    assert_eq!(s.aperture, ApertureSpec::ImageSpaceFNumber { value: 4.0 });
    assert_eq!(s.fields.kind, FieldKind::RealImageHeight);
    match &s.surfaces[6].shape {
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => {
            assert_eq!(*curvature, 1.0 / -410.38);
            assert_eq!(*conic, 0.0);
            assert_eq!(coefficients, &[3.2e-6, 8.0e-10]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn zmx_content_that_does_not_map_becomes_warnings() {
    let imp = import("awkward.zmx");
    let s = &imp.system;
    let k = kinds(&imp);
    // Centimetres: EPD 2 cm, a 6 cm radius, a 0.5 cm thickness, a 0.1 cm decentre.
    assert_eq!(
        s.aperture,
        ApertureSpec::EntrancePupilDiameter { value: 20.0 }
    );
    assert!((s.surfaces[1].shape.curvature() - 1.0 / 60.0).abs() < 1e-15);
    assert_eq!(s.surfaces[1].thickness, 5.0);
    // Unknown glass with n_d/V_d: a model glass.
    assert_eq!(
        s.surfaces[1].material,
        MaterialRef::Model {
            nd: 1.6,
            vd: 40.0,
            dpgf: 0.0
        }
    );
    assert!(imp.warnings.iter().any(|w| w.kind == K::UnknownGlass
        && w.surface == Some(1)
        && w.message.contains("UNOBTAINIUM")));
    // Unknown surface type: base sphere kept.
    assert_eq!(s.surfaces[2].shape, SurfaceShape::sphere(-350.0));
    assert!(k.contains(&K::UnsupportedSurface));
    // Coordinate break.
    assert_eq!(s.surfaces[3].shape, SurfaceShape::CoordinateBreak);
    let p = s.surfaces[3].placement.unwrap();
    assert_eq!((p.decenter_x, p.tilt_x), (1.0, 2.5));
    // Non-sequential component, multi-configuration, tolerances, a pickup.
    assert_eq!(s.surfaces[4].shape, SurfaceShape::Plane);
    assert!(k.contains(&K::NonSequential));
    let text: Vec<String> = imp.warnings.iter().map(|w| w.to_string()).collect();
    for needle in ["multi-configuration", "tolerance", "PZUP"] {
        assert!(
            text.iter().any(|t| t.contains(needle)),
            "{needle}: {text:#?}"
        );
    }
    // MOFF's lines are reported once.
    assert_eq!(
        text.iter()
            .filter(|t| t.contains("multi-configuration"))
            .count(),
        1
    );
    assert!(!k.contains(&K::Validation), "{text:#?}");
}

#[test]
fn seq_singlet_with_comments_qualifiers_and_a_paraxial_image_solve() {
    let imp = import("singlet.seq");
    let s = &imp.system;
    assert_eq!(imp.format, PrescriptionFormat::Seq);
    assert_eq!(s.name, "Singlet f100 F/5");
    assert_eq!(s.surfaces.len(), 4);
    assert_eq!(s.stop, 1);
    assert!(s.surfaces[0].thickness.is_infinite());
    assert_eq!(s.surfaces[1].material, builtin("N-BK7"));
    // WL lists nm, longest first; REF 2 is d.
    assert_eq!(s.wavelengths.primary_um(), 0.5875618);
    assert_eq!(s.wavelengths.wavelengths[0].um, 0.6562725);
    // PIM is a marginal-ray-height solve on the last thickness.
    assert_eq!(s.solves.len(), 1);
    assert_eq!(
        s.solves[0].kind,
        SolveKind::MarginalRayHeight { height: 0.0 }
    );
    // The variable flags are reported once.
    assert_eq!(kinds(&imp), [K::SolveDropped], "{:?}", imp.warnings);
    s.validate().unwrap();
}

#[test]
fn seq_quoted_semicolons_and_stop_size() {
    let imp = import("cooke_triplet.seq");
    let s = &imp.system;
    assert_eq!(s.name, "Cooke triplet f50 F/4; SK16/F2/SK16");
    assert_eq!(s.stop, 4);
    assert_eq!(s.surfaces[3].material, builtin("F2"));
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
}

#[test]
fn seq_asphere_letters() {
    let imp = import("petzval_asphere.seq");
    let s = &imp.system;
    assert_eq!(s.aperture, ApertureSpec::ImageSpaceFNumber { value: 4.0 });
    assert_eq!(s.fields.kind, FieldKind::RealImageHeight);
    assert_eq!(
        s.surfaces[6].shape,
        SurfaceShape::EvenAsphere {
            curvature: 1.0 / -410.38,
            conic: 0.0,
            coefficients: vec![3.2e-6, 8.0e-10],
        }
    );
}

#[test]
fn seq_unknown_glasses_and_commands() {
    let text = "TIT 'x'; EPD 10; DIM C; WL 550\nSO 0 1e11\nS 5 0.4 NOSUCH_ACME; STO\nS -5 1 1.6:40\n\
                S 0 2 REFL\nS 0 -3 620603.ACME\nFOO 1 2; FOO 3\nADE S2 5\nPUY S4 0.1\nSI";
    let lib = MaterialLibrary::with_builtin();
    let imp = import_prescription(text.as_bytes(), PrescriptionFormat::Seq, &lib).unwrap();
    let s = &imp.system;
    // Not in any catalog and no index data: kept by name.
    assert_eq!(s.surfaces[1].material, MaterialRef::glass("NOSUCH"));
    assert!(
        imp.warnings
            .iter()
            .any(|w| w.kind == K::UnknownGlass && w.surface == Some(1))
    );
    assert_eq!(s.surfaces[1].thickness, 4.0); // DIM C
    assert_eq!(
        s.surfaces[2].material,
        MaterialRef::Model {
            nd: 1.6,
            vd: 40.0,
            dpgf: 0.0
        }
    );
    assert_eq!(s.surfaces[3].interaction, Interaction::Reflect);
    assert_eq!(s.surfaces[3].material, s.surfaces[2].material);
    match &s.surfaces[4].material {
        MaterialRef::Model { nd, vd, .. } => {
            assert!((nd - 1.62).abs() < 1e-12 && (vd - 60.3).abs() < 1e-12)
        }
        m => panic!("{m:?}"),
    }
    let text: Vec<String> = imp.warnings.iter().map(|w| w.to_string()).collect();
    assert!(
        text.iter()
            .any(|t| t.contains("FOO") && t.contains("(2 times)")),
        "{text:#?}"
    );
    assert!(text.iter().any(|t| t.contains("ADE")), "{text:#?}");
    assert!(imp.warnings.iter().any(|w| w.kind == K::SolveDropped));
}

#[test]
fn defaults_for_a_bare_prescription() {
    let lib = MaterialLibrary::with_builtin();
    let imp =
        import_prescription(b"S 50 5 N-BK7\nS -50 48\n", PrescriptionFormat::Seq, &lib).unwrap();
    let s = &imp.system;
    assert_eq!(s.surfaces.len(), 3);
    assert_eq!(s.stop, 1);
    assert_eq!(s.wavelengths.wavelengths.len(), 3);
    assert!(
        kinds(&imp).iter().all(|k| *k == K::Defaulted),
        "{:?}",
        imp.warnings
    );
}

#[test]
fn zmx_counts_beyond_the_data_present_are_clamped() {
    let lib = MaterialLibrary::with_builtin();
    let text = "MODE SEQ\nENPD 10\nFTYP 0 0 99999999999 99999999999 0 0 0\n\
                YFLN 0 2\nWAVM 1 0.55 1\nWAVM 3 0.65 1\n\
                SURF 0\n  DISZ INFINITY\nSURF 1\n  STOP\n  CURV 0.02\n  DISZ 5\n  GLAS N-BK7\n\
                SURF 2\n  DISZ 90\nSURF 3\n";
    let imp = import_prescription(text.as_bytes(), PrescriptionFormat::Zmx, &lib).unwrap();
    let s = &imp.system;
    let ys: Vec<f64> = s.fields.fields.iter().map(|f| f.y).collect();
    assert_eq!(ys, [0.0, 2.0]);
    let wls: Vec<f64> = s.wavelengths.wavelengths.iter().map(|w| w.um).collect();
    assert_eq!(wls, [0.55, 0.65]);
    assert!(
        imp.warnings
            .iter()
            .any(|w| w.kind == K::Defaulted && w.message.contains("FTYP states")),
        "{:?}",
        imp.warnings
    );
}

#[test]
fn text_without_surfaces_is_an_error() {
    let lib = MaterialLibrary::with_builtin();
    for (bytes, format) in [
        (&b""[..], PrescriptionFormat::Zmx),
        (b"MODE NSC\nNSOB 1 2 3\n", PrescriptionFormat::Zmx),
        (b"hello world", PrescriptionFormat::Seq),
    ] {
        assert!(import_prescription(bytes, format, &lib).is_err());
    }
    assert!(read_prescription_file("lens.txt", &lib).is_err());
}

#[test]
fn glasses_resolve_against_loaded_user_catalogs() {
    use mklight_core::materials::GlassCatalog;
    use std::sync::Arc;
    let mut lib = MaterialLibrary::with_builtin();
    let builtin = lib.catalogs()[0].clone();
    let mut user = GlassCatalog {
        name: "schott".into(),
        glasses: Default::default(),
    };
    let mut g = builtin.glasses["N-SK16"].clone();
    g.name = "SK16".into();
    user.glasses.insert("SK16".into(), g);
    lib.add_catalog(Arc::new(user));
    let text = "EPD 10; WL 587.5618\nSO 0 1e11\nS 20 3 SK16_SCHOTT; STO\nS -20 40\nSI";
    let imp = import_prescription(text.as_bytes(), PrescriptionFormat::Seq, &lib).unwrap();
    assert_eq!(
        imp.system.surfaces[1].material,
        MaterialRef::Catalog {
            catalog: Some("schott".into()),
            name: "SK16".into(),
            offset: Default::default()
        }
    );
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
}
