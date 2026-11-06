//! Imported `.zmx` and `.seq` fixtures against the native samples they
//! describe: the first-order properties must agree.

use std::path::PathBuf;
use std::sync::Arc;

use mklight_core::import::read_prescription_file;
use mklight_core::materials::MaterialLibrary;
use mklight_core::{OpticalSystem, ResolvedSystem, Snapshot, samples};
use mklight_trace::paraxial::{FirstOrder, first_order};

fn first_order_of(sys: &OpticalSystem) -> FirstOrder {
    let lib = Arc::new(MaterialLibrary::with_builtin());
    let r = ResolvedSystem::new(&Snapshot::new(sys.clone()).unwrap(), lib).unwrap();
    first_order(&r, r.primary_wavelength).unwrap()
}

fn close(name: &str, what: &str, a: f64, b: f64) {
    let tol = 1e-9 * b.abs().max(1.0);
    assert!((a - b).abs() <= tol, "{name}: {what} {a} vs native {b}");
}

#[test]
fn imported_fixtures_match_the_native_first_order() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mklight-core/tests/fixtures/prescriptions");
    let lib = MaterialLibrary::with_builtin();
    let cases = [
        ("singlet", samples::singlet()),
        ("cooke_triplet", samples::cooke_triplet()),
        ("petzval_asphere", samples::petzval_asphere()),
    ];
    for (stem, native) in cases {
        let reference = first_order_of(&native);
        for ext in ["zmx", "seq"] {
            let name = format!("{stem}.{ext}");
            let imported = read_prescription_file(dir.join(&name), &lib)
                .unwrap()
                .system;
            assert_eq!(imported.surfaces.len(), native.surfaces.len(), "{name}");
            assert_eq!(imported.stop, native.stop, "{name}");
            assert_eq!(imported.aperture, native.aperture, "{name}");
            assert_eq!(imported.fields.kind, native.fields.kind, "{name}");
            assert_eq!(
                imported.wavelengths.primary_um(),
                native.wavelengths.primary_um(),
                "{name}"
            );
            let fo = first_order_of(&imported);
            close(&name, "EFL", fo.efl, reference.efl);
            close(&name, "BFL", fo.bfl, reference.bfl);
            close(
                &name,
                "image distance",
                fo.image_distance,
                reference.image_distance,
            );
            close(
                &name,
                "EPD",
                fo.entrance_pupil_diameter,
                reference.entrance_pupil_diameter,
            );
            close(
                &name,
                "entrance pupil",
                fo.entrance_pupil_position,
                reference.entrance_pupil_position,
            );
            close(
                &name,
                "exit pupil",
                fo.exit_pupil_position,
                reference.exit_pupil_position,
            );
            close(&name, "F/#", fo.image_f_number, reference.image_f_number);
            close(&name, "total track", fo.total_track, reference.total_track);
            close(
                &name,
                "image height",
                fo.paraxial_image_height,
                reference.paraxial_image_height,
            );
        }
    }
}
