//! Writes JSON fixtures of the model's serde representation for the macOS
//! app's Codable mirror (`apps/macos/Tests/MKLightModelTests/Fixtures`) and
//! the built-in samples the app ships
//! (`apps/macos/Sources/MKLightModel/Resources/samples.json`).
//!
//! Run with `cargo run -p mklight-core --example dump_fixtures [out_dir]`;
//! with `out_dir`, every file goes there instead.
//! The fixtures cover every enum variant of the model, `ParamPath` and
//! `RunInfo`, so a Swift decode → encode round trip that compares equal to
//! these files proves the two representations agree; `validation_cases.json`
//! holds the engine's verdicts for the app's mirror of the validation rules.
//!
//! [`fixtures`] is everything [`main`] writes: the test `tests/app_fixtures.rs`
//! includes this file and compares it with the committed files.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mklight_core::job::Precision;
use mklight_core::model::surface::{DiffractivePhase, ElementPlacement, ZernikeOrdering};
use mklight_core::param::{
    CoefficientSet, IrregularityMode, ParamPath, PlacementComponent, ShapeParameter,
};
use mklight_core::*;
use serde::Serialize;

fn cooke_triplet() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "Cooke triplet".into();
    s.notes =
        "Classic three-element anastigmat, EFL ≈ 47.1 mm, F/4.7, image at the paraxial focus."
            .into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![
            Field::new(0.0, 0.0),
            Field::new(0.0, 14.0),
            Field::new(0.0, 20.0),
        ],
    };
    let glass = |r: f64, t: f64, g: &str| (SurfaceShape::sphere(r), t, MaterialRef::glass(g));
    let air = |r: f64, t: f64| (SurfaceShape::sphere(r), t, MaterialRef::Air);
    let rows = [
        (SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        glass(22.01359, 3.25896, "N-SK16"),
        air(-435.76044, 6.00755),
        glass(-22.21328, 0.99997, "F2"),
        air(20.29192, 4.75041),
        (SurfaceShape::Plane, 2.0, MaterialRef::Air),
        glass(79.68360, 2.95208, "N-SK16"),
        air(-18.39533, 41.36144),
        (SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.surfaces = rows
        .into_iter()
        .enumerate()
        .map(|(i, (shape, t, m))| Surface::new(i as u64, shape, t, m))
        .collect();
    s.stop = 5;
    s.surfaces[5].label = "STOP".into();
    s.coatings.insert(
        "AR".into(),
        CoatingDef::Ideal {
            reflectance: 0.005,
            transmittance: 0.995,
        },
    );
    for i in 1..=7 {
        if i != 5 {
            s.surfaces[i].coating = Some("AR".into());
        }
    }
    s
}

/// Every enum variant of the model in one (valid) system.
fn kitchen_sink() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "Kitchen sink".into();
    s.notes = "Covers every model variant; not a sensible lens.".into();
    s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![
            Field::new(0.0, 0.0),
            Field {
                x: 1.5,
                y: -2.25,
                weight: 0.5,
                vignetting: Vignetting {
                    vdx: 0.0,
                    vdy: 0.1,
                    vcx: 0.05,
                    vcy: 0.2,
                },
            },
        ],
    };
    s.wavelengths = WavelengthSet {
        wavelengths: vec![
            Wavelength {
                um: 0.5460740,
                weight: 2.0,
            },
            Wavelength {
                um: 1.064,
                weight: 0.25,
            },
        ],
        primary: 0,
    };
    s.ray_aiming = RayAiming::Real { tolerance: 1e-9 };
    s.environment = Environment {
        temperature_c: 35.5,
        pressure_atm: 0.75,
    };

    let mut obj = Surface::new(0, SurfaceShape::Plane, 120.0, MaterialRef::Air);
    obj.label = "OBJECT".into();

    let mut s1 = Surface::new(
        10,
        SurfaceShape::Conic {
            curvature: 0.02,
            conic: -0.5,
        },
        4.0,
        MaterialRef::Catalog {
            catalog: Some("BUILTIN".into()),
            name: "N-BK7".into(),
            offset: materials::MaterialOffset {
                dn_d: 0.0012,
                dv_d: -0.8,
            },
        },
    );
    s1.semi_diameter = SemiDiameter::Fixed { value: 12.5 };
    s1.coating = Some("V-coat".into());
    s1.scatter = Some(ScatterModel::Lambertian { tis: 1e-4 });
    s1.edge = EdgeSpec {
        mechanical_semi_diameter: Some(14.0),
        edge_scatter: Some(ScatterModel::Gaussian {
            tis: 0.02,
            sigma_slope: 0.1,
        }),
    };

    let mut s2 = Surface::new(
        11,
        SurfaceShape::EvenAsphere {
            curvature: -0.01,
            conic: 0.0,
            coefficients: vec![1.5e-5, -2.0e-8, 3.25e-11],
        },
        2.5,
        MaterialRef::Model {
            nd: 1.6,
            vd: 40.0,
            dpgf: 0.002,
        },
    );
    s2.aperture = Some(ApertureShape::Annular {
        inner_radius: 1.0,
        outer_radius: 11.0,
    });
    s2.coating = Some("Ideal".into());
    s2.scatter = Some(ScatterModel::Abg {
        a: 1e-5,
        b: 1e-3,
        g: 2.0,
    });

    let mut s3 = Surface::new(
        12,
        SurfaceShape::Plane,
        3.0,
        MaterialRef::Constant { n: 1.5 },
    );
    s3.label = "STOP".into();
    s3.aperture = Some(ApertureShape::Circular { radius: 5.0 });

    let mut s4 = Surface::new(
        13,
        SurfaceShape::Paraxial { focal_length: 80.0 },
        5.0,
        MaterialRef::Vacuum,
    );
    s4.aperture = Some(ApertureShape::Rectangular {
        half_x: 6.0,
        half_y: 4.0,
    });

    let mut cb = Surface::new(14, SurfaceShape::CoordinateBreak, 0.0, MaterialRef::Air);
    cb.placement = Some(Placement {
        decenter_x: 0.1,
        decenter_y: -0.2,
        tilt_x: 1.5,
        tilt_y: 0.0,
        tilt_z: -45.0,
        order: TransformOrder::TiltThenDecenter,
        restore_after: false,
    });

    let mut m = Surface::new(
        15,
        SurfaceShape::sphere(-200.0),
        -30.0,
        MaterialRef::glass("MIRROR-IGNORED"),
    );
    m.interaction = Interaction::Reflect;
    m.coating = Some("Silver".into());
    m.aperture = Some(ApertureShape::Elliptical {
        semi_x: 15.0,
        semi_y: 10.0,
    });
    m.placement = Some(Placement {
        tilt_x: 2.0,
        restore_after: true,
        ..Default::default()
    });

    let mut d = Surface::new(16, SurfaceShape::Plane, -10.0, MaterialRef::Air);
    d.interaction = Interaction::Dummy;
    d.label = "reference".into();

    let mut img = Surface::new(17, SurfaceShape::sphere(-500.0), 0.0, MaterialRef::Air);
    img.label = "IMAGE".into();

    s.surfaces = vec![obj, s1, s2, s3, s4, cb, m, d, img];
    s.stop = 3;

    s.coatings.insert(
        "V-coat".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![
                Layer {
                    material: MaterialRef::Constant { n: 2.1 },
                    thickness_um: 0.0,
                    quarter_waves: Some(1.0),
                    extinction: 0.0,
                },
                Layer {
                    material: MaterialRef::glass("MGF2"),
                    thickness_um: 0.0996,
                    quarter_waves: None,
                    extinction: 1e-4,
                },
            ],
            mirror_substrate: None,
        },
    );
    s.coatings.insert(
        "Silver".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.38 },
                thickness_um: 0.1,
                quarter_waves: None,
                extinction: 0.0,
            }],
            mirror_substrate: Some((0.12, 3.4)),
        },
    );
    s.coatings.insert(
        "Ideal".into(),
        CoatingDef::Ideal {
            reflectance: 0.01,
            transmittance: 0.98,
        },
    );
    s.coatings.insert(
        "Measured".into(),
        CoatingDef::Tabulated {
            reflectance: vec![(0.4, 0.012), (0.55, 0.004), (0.7, 0.009)],
        },
    );

    s.mechanics = vec![
        MechanicalPart::Barrel {
            from: SurfaceId(10),
            to: SurfaceId(13),
            inner_radius: 15.0,
            finish: ScatterModel::Lambertian { tis: 0.05 },
        },
        MechanicalPart::AnnularBaffle {
            surface: SurfaceId(12),
            z: -0.5,
            inner_radius: 5.5,
            outer_radius: 16.0,
            finish: ScatterModel::Abg {
                a: 0.01,
                b: 0.001,
                g: 1.5,
            },
        },
    ];
    s.sensor = Some(Sensor {
        pixels_x: 4000,
        pixels_y: 3000,
        pitch_x: 0.00376,
        pitch_y: 0.00376,
        max_cra_deg: Some(30.0),
        spectral_response: vec![(0.45, 0.6), (0.55, 1.0), (0.65, 0.7)],
    });
    s.configurations = vec![
        Configuration {
            name: "near focus".into(),
            overrides: vec![
                (
                    ParamPath::Thickness {
                        surface: SurfaceId(0),
                    },
                    80.0,
                ),
                (ParamPath::ApertureValue, 0.04),
            ],
        },
        Configuration::default(),
    ];
    s
}

/// Each aperture spec, field kind and ray-aiming mode, `±∞` thicknesses and
/// the tolerancing sag overlay and element placement.
fn variants() -> Vec<OpticalSystem> {
    let base = OpticalSystem::new_empty;
    let mut out = Vec::new();

    let mut a = base();
    a.name = "image F/#".into();
    a.aperture = ApertureSpec::ImageSpaceFNumber { value: 2.8 };
    a.fields.kind = FieldKind::ParaxialImageHeight;
    a.fields.fields.push(Field::new(0.0, 21.6));
    out.push(a);

    let mut b = base();
    b.name = "working F/#".into();
    b.aperture = ApertureSpec::ParaxialWorkingFNumber { value: 4.0 };
    b.fields.kind = FieldKind::RealImageHeight;
    b.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    out.push(b);

    let mut c = base();
    c.name = "float by stop".into();
    c.aperture = ApertureSpec::FloatByStopSize;
    c.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 3.0 };
    out.push(c);

    let mut d = base();
    d.name = "negative infinite object".into();
    d.surfaces[0].thickness = f64::NEG_INFINITY;
    d.surfaces[2].thickness = f64::INFINITY;
    out.push(d);

    // Tolerancing perturbations: an irregular surface in a moved range.
    let mut e = base();
    e.name = "irregular moved surface".into();
    e.surfaces[1].shape = SurfaceShape::SagOverlay {
        base: Box::new(SurfaceShape::sphere(50.0)),
        norm_radius: 5.0,
        test_wavelength_um: 0.6328,
        astigmatism: 0.5,
        astigmatism_angle_deg: 30.0,
        spherical: -0.25,
    };
    e.surfaces[1].element_placement = Some(ElementPlacement {
        last: e.surfaces[1].id,
        pivot: None,
        motion: Placement {
            decenter_y: 0.02,
            tilt_x: 0.1,
            ..Default::default()
        },
    });
    out.push(e);
    out
}

fn all_param_paths() -> Vec<ParamPath> {
    let surface = SurfaceId(7);
    vec![
        ParamPath::Curvature { surface },
        ParamPath::Thickness { surface },
        ParamPath::Conic { surface },
        ParamPath::Asphere { surface, index: 2 },
        ParamPath::ParaxialFocalLength { surface },
        ParamPath::ModelNd { surface },
        ParamPath::ModelVd { surface },
        ParamPath::ModelDpgf { surface },
        ParamPath::MaterialDnD { surface },
        ParamPath::MaterialDvD { surface },
        ParamPath::DecenterX { surface },
        ParamPath::DecenterY { surface },
        ParamPath::TiltX { surface },
        ParamPath::TiltY { surface },
        ParamPath::TiltZ { surface },
        ParamPath::SemiDiameter { surface },
        ParamPath::ApertureValue,
        ParamPath::FieldX { index: 1 },
        ParamPath::FieldY { index: 1 },
        ParamPath::FieldWeight { index: 1 },
        ParamPath::WavelengthUm { index: 0 },
        ParamPath::WavelengthWeight { index: 0 },
        ParamPath::TemperatureC,
        ParamPath::PressureAtm,
        ParamPath::Irregularity {
            surface,
            mode: IrregularityMode::Spherical,
        },
        ParamPath::ElementPlacement {
            surface,
            last: SurfaceId(9),
            pivot: Some(SurfaceId(8)),
            component: PlacementComponent::TiltY,
        },
        ParamPath::ElementPlacement {
            surface,
            last: SurfaceId(9),
            pivot: None,
            component: PlacementComponent::DecenterX,
        },
    ]
    .into_iter()
    .chain(
        [
            CoefficientSet::Odd,
            CoefficientSet::Q,
            CoefficientSet::Phase,
            CoefficientSet::XAsphere,
            CoefficientSet::YAsphere,
            CoefficientSet::Xy,
            CoefficientSet::Zernike,
        ]
        .map(|set| ParamPath::ShapeCoefficient {
            surface,
            set,
            index: 2,
        }),
    )
    .chain(
        [
            ShapeParameter::CurvatureX,
            ShapeParameter::ConicX,
            ShapeParameter::CurvatureY,
            ShapeParameter::ConicY,
            ShapeParameter::NormRadius,
            ShapeParameter::AstigmatismAngleDeg,
            ShapeParameter::DesignOrder,
            ShapeParameter::DesignWavelengthUm,
            ShapeParameter::TraceOrder,
        ]
        .map(|parameter| ParamPath::ShapeParam { surface, parameter }),
    )
    .collect()
}

/// Every extended surface shape (SURFACES.md) in one valid system: odd
/// asphere, both Q types, diffractive phases on plane, conic and
/// even-asphere bases with signed orders, toroid, biconic, a cylinder
/// preset, an XY polynomial and Zernike sags in both orderings. Surface ids
/// equal their indices. Not a sensible lens.
fn extended_shapes() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "Extended shapes".into();
    s.notes = "Covers every extended surface shape; not a sensible lens.".into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 8.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0), Field::new(1.0, 2.0)],
    };
    let glass = || MaterialRef::glass("N-BK7");
    let phase = |norm_radius: f64,
                 coefficients: Vec<f64>,
                 design_order: i32,
                 design_wavelength_um: f64,
                 trace_order: i32| DiffractivePhase {
        norm_radius,
        coefficients,
        design_order,
        design_wavelength_um,
        trace_order,
    };
    let shapes: Vec<(SurfaceShape, MaterialRef)> = vec![
        (SurfaceShape::Plane, MaterialRef::Air),
        (
            SurfaceShape::OddAsphere {
                curvature: 0.01,
                conic: -0.5,
                coefficients: vec![0.0, 1e-5, -1e-7],
            },
            glass(),
        ),
        (
            SurfaceShape::QCon {
                curvature: -0.01,
                conic: 0.2,
                norm_radius: 8.0,
                coefficients: vec![0.001, -0.0002],
            },
            MaterialRef::Air,
        ),
        (
            SurfaceShape::QBfs {
                curvature: 0.02,
                norm_radius: 8.0,
                coefficients: vec![0.0005],
            },
            glass(),
        ),
        (
            SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::Plane),
                phase: phase(6.0, vec![-20.0, 0.5], 1, 0.5875618, 1),
            },
            MaterialRef::Air,
        ),
        (
            SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::Conic {
                    curvature: 0.005,
                    conic: -1.0,
                }),
                phase: phase(7.0, vec![-10.0], 2, 0.55, -1),
            },
            glass(),
        ),
        (
            SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::EvenAsphere {
                    curvature: -0.004,
                    conic: 0.0,
                    coefficients: vec![1e-6, -2e-9],
                }),
                phase: phase(9.0, vec![5.0, -0.1, 0.01], -1, 1.064, 0),
            },
            MaterialRef::Air,
        ),
        (
            SurfaceShape::Toroid {
                curvature_x: 0.02,
                curvature_y: 0.01,
                conic_y: -0.3,
                y_asphere: vec![1e-6],
            },
            glass(),
        ),
        (
            SurfaceShape::Biconic {
                curvature_x: 0.015,
                curvature_y: -0.01,
                conic_x: -0.5,
                conic_y: 0.1,
                x_asphere: vec![1e-6],
                y_asphere: vec![-2e-7, 1e-9],
            },
            MaterialRef::Air,
        ),
        // Cylinder preset: power in x only.
        (
            SurfaceShape::Biconic {
                curvature_x: 0.025,
                curvature_y: 0.0,
                conic_x: 0.0,
                conic_y: 0.0,
                x_asphere: vec![],
                y_asphere: vec![],
            },
            MaterialRef::Constant { n: 1.5 },
        ),
        (SurfaceShape::Plane, MaterialRef::Air),
        (
            SurfaceShape::XYPolynomial {
                curvature: 0.005,
                conic: 0.0,
                norm_radius: 10.0,
                coefficients: vec![0.0, 0.0, 0.001, 0.0002, -0.001, 0.0, -0.0001, 0.0, 0.0001],
            },
            glass(),
        ),
        (
            SurfaceShape::ZernikeSag {
                curvature: -0.005,
                conic: 0.0,
                norm_radius: 10.0,
                ordering: ZernikeOrdering::Fringe,
                coefficients: vec![0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0002],
            },
            MaterialRef::Air,
        ),
        (
            SurfaceShape::ZernikeSag {
                curvature: 0.0,
                conic: 0.0,
                norm_radius: 10.0,
                ordering: ZernikeOrdering::Standard,
                coefficients: vec![0.0, 0.0, 0.0, 0.0005, -0.0003, 0.0002],
            },
            glass(),
        ),
        (SurfaceShape::Plane, MaterialRef::Air),
        (SurfaceShape::Plane, MaterialRef::Air),
    ];
    let last = shapes.len() - 1;
    s.surfaces = shapes
        .into_iter()
        .enumerate()
        .map(|(i, (shape, m))| {
            let t = if i == 0 {
                f64::INFINITY
            } else if i == last {
                0.0
            } else {
                4.0
            };
            Surface::new(i as u64, shape, t, m)
        })
        .collect();
    s.stop = 4;
    s.surfaces[4].label = "STOP".into();
    s.surfaces[9].label = "cylinder".into();
    s.surfaces[12].semi_diameter = SemiDiameter::Fixed { value: 9.0 };
    // Every order-efficiency model (surface 4 keeps the default, ideal).
    s.surfaces[5].diffraction_efficiency =
        mklight_core::materials::diffraction::DiffractionEfficiency::Kinoform;
    s.surfaces[6].diffraction_efficiency =
        mklight_core::materials::diffraction::DiffractionEfficiency::Multilevel {
            levels: std::num::NonZeroU16::new(8).unwrap(),
        };
    s
}

/// One model and the engine's verdict on it, for front-end validators that
/// mirror `OpticalSystem::validate`.
#[derive(Serialize)]
struct ValidationCase {
    name: String,
    system: OpticalSystem,
    /// `None` when `validate` accepts the system, else its error message.
    error: Option<String>,
}

/// The Cooke triplet with one configuration override each: some that
/// `ParamPath::get` accepts and some that it rejects for the surface's shape,
/// material or semi-diameter mode, the aperture spec, or an index range.
/// Then [`extended_validation_cases`] and, without configurations,
/// [`structural_validation_cases`].
fn validation_cases() -> Vec<ValidationCase> {
    let id = SurfaceId;
    let mut base = cooke_triplet();
    base.surfaces[2].shape = SurfaceShape::EvenAsphere {
        curvature: -1.0 / 435.76044,
        conic: 0.0,
        coefficients: vec![1e-6],
    };
    base.surfaces[3].material = MaterialRef::Model {
        nd: 1.62004,
        vd: 36.37,
        dpgf: 0.0,
    };
    base.surfaces[4].shape = SurfaceShape::Paraxial { focal_length: 50.0 };
    base.surfaces[6].semi_diameter = SemiDiameter::Fixed { value: 7.0 };
    base.surfaces[5].shape = SurfaceShape::CoordinateBreak;
    base.surfaces[5].placement = Some(Placement::default());
    base.stop = 1;
    let with_value = |name: &str, p: ParamPath, v: f64, edit: &dyn Fn(&mut OpticalSystem)| {
        let mut s = base.clone();
        s.name = name.into();
        edit(&mut s);
        s.configurations = vec![Configuration {
            name: name.into(),
            overrides: vec![(p, v)],
        }];
        let error = s.validate().err().map(|e| e.to_string());
        ValidationCase {
            name: name.into(),
            system: s,
            error,
        }
    };
    let with = |name: &str, p: ParamPath, edit: &dyn Fn(&mut OpticalSystem)| {
        with_value(name, p, 1.0, edit)
    };
    let none = |_: &mut OpticalSystem| {};
    vec![
        with(
            "curvature on conic",
            ParamPath::Curvature { surface: id(1) },
            &none,
        ),
        with(
            "curvature on plane",
            ParamPath::Curvature { surface: id(8) },
            &none,
        ),
        with(
            "curvature on paraxial",
            ParamPath::Curvature { surface: id(4) },
            &none,
        ),
        with(
            "curvature on coordinate break",
            ParamPath::Curvature { surface: id(5) },
            &none,
        ),
        with(
            "conic on asphere",
            ParamPath::Conic { surface: id(2) },
            &none,
        ),
        with("conic on plane", ParamPath::Conic { surface: id(8) }, &none),
        with(
            "asphere term beyond list",
            ParamPath::Asphere {
                surface: id(2),
                index: 3,
            },
            &none,
        ),
        with(
            "asphere term at last index",
            ParamPath::Asphere {
                surface: id(2),
                index: 19,
            },
            &none,
        ),
        with(
            "asphere term at limit",
            ParamPath::Asphere {
                surface: id(2),
                index: 20,
            },
            &none,
        ),
        with(
            "asphere on conic",
            ParamPath::Asphere {
                surface: id(1),
                index: 0,
            },
            &none,
        ),
        with(
            "focal length on paraxial",
            ParamPath::ParaxialFocalLength { surface: id(4) },
            &none,
        ),
        with(
            "focal length on conic",
            ParamPath::ParaxialFocalLength { surface: id(1) },
            &none,
        ),
        // A model glass index is above 1 (the structural cases hold the rule).
        with_value(
            "nd on model glass",
            ParamPath::ModelNd { surface: id(3) },
            1.7,
            &none,
        ),
        with(
            "vd on catalog glass",
            ParamPath::ModelVd { surface: id(1) },
            &none,
        ),
        with(
            "dpgf on air",
            ParamPath::ModelDpgf { surface: id(2) },
            &none,
        ),
        with(
            "dv_d on catalog glass",
            ParamPath::MaterialDvD { surface: id(1) },
            &none,
        ),
        with(
            "dn_d on model glass",
            ParamPath::MaterialDnD { surface: id(3) },
            &none,
        ),
        with(
            "tilt on any surface",
            ParamPath::TiltX { surface: id(7) },
            &none,
        ),
        with(
            "thickness of missing surface",
            ParamPath::Thickness { surface: id(99) },
            &none,
        ),
        with(
            "fixed semi-diameter",
            ParamPath::SemiDiameter { surface: id(6) },
            &none,
        ),
        with(
            "auto semi-diameter",
            ParamPath::SemiDiameter { surface: id(1) },
            &none,
        ),
        with("aperture value", ParamPath::ApertureValue, &none),
        with(
            "aperture value floating by stop",
            ParamPath::ApertureValue,
            &|s: &mut OpticalSystem| {
                s.aperture = ApertureSpec::FloatByStopSize;
                s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 5.0 };
            },
        ),
        with("last field", ParamPath::FieldY { index: 2 }, &none),
        with(
            "field out of range",
            ParamPath::FieldWeight { index: 3 },
            &none,
        ),
        with("wavelength", ParamPath::WavelengthUm { index: 2 }, &none),
        with(
            "wavelength out of range",
            ParamPath::WavelengthWeight { index: 3 },
            &none,
        ),
        with("temperature", ParamPath::TemperatureC, &none),
    ]
    .into_iter()
    .chain(extended_validation_cases())
    .chain(structural_validation_cases())
    .collect()
}

/// `extended_shapes()` (surface id = index) with one override each, for the
/// extended parameter paths: coefficient sets and scalars that do or do not
/// belong to the shape, indices at the shape's limits, a missing in-range
/// coefficient, diffraction orders (integral or not), legacy curvature,
/// conic and asphere paths on a diffractive base, toroid and QBfs.
fn extended_validation_cases() -> Vec<ValidationCase> {
    use CoefficientSet::*;
    use ShapeParameter::*;
    let id = SurfaceId;
    let base = extended_shapes();
    let with = |name: &str, p: ParamPath, v: f64| {
        let mut s = base.clone();
        s.name = name.into();
        s.configurations = vec![Configuration {
            name: name.into(),
            overrides: vec![(p, v)],
        }];
        let error = s.validate().err().map(|e| e.to_string());
        ValidationCase {
            name: name.into(),
            system: s,
            error,
        }
    };
    let coef = |surface: u64, set, index| ParamPath::ShapeCoefficient {
        surface: id(surface),
        set,
        index,
    };
    let param = |surface: u64, parameter| ParamPath::ShapeParam {
        surface: id(surface),
        parameter,
    };
    vec![
        with("odd coefficient", coef(1, Odd, 2), 1e-6),
        with("odd coefficient missing in range", coef(1, Odd, 15), 1e-9),
        with("odd coefficient at limit", coef(1, Odd, 20), 1.0),
        with("q coefficient on q_con", coef(2, Q, 0), 0.001),
        with("q coefficient on q_bfs at last term", coef(3, Q, 19), 0.001),
        with("q coefficient on odd asphere", coef(1, Q, 0), 0.001),
        with("phase coefficient at last term", coef(4, Phase, 9), 0.1),
        with("phase coefficient at limit", coef(4, Phase, 10), 0.1),
        with("phase coefficient on q_con", coef(2, Phase, 0), 0.1),
        with("x asphere on toroid", coef(7, XAsphere, 0), 1e-6),
        with("y asphere on toroid", coef(7, YAsphere, 3), 1e-6),
        with(
            "x asphere on biconic at last term",
            coef(8, XAsphere, 9),
            1e-9,
        ),
        with("y asphere on cylinder", coef(9, YAsphere, 0), 1e-9),
        with("xy at last term", coef(11, Xy, 64), 1e-5),
        with("xy at limit", coef(11, Xy, 65), 1e-5),
        with("zernike on xy polynomial", coef(11, Zernike, 0), 1e-5),
        with("fringe zernike at last term", coef(12, Zernike, 36), 1e-5),
        with("fringe zernike at limit", coef(12, Zernike, 37), 1e-5),
        with("standard zernike at last term", coef(13, Zernike, 44), 1e-5),
        with("standard zernike at limit", coef(13, Zernike, 45), 1e-5),
        with("zernike on diffractive", coef(5, Zernike, 0), 1e-5),
        with("curvature_x on toroid", param(7, CurvatureX), 0.03),
        with("conic_x on toroid", param(7, ConicX), -1.0),
        with("conic_x on biconic", param(8, ConicX), -1.0),
        with("conic_y on toroid", param(7, ConicY), -1.0),
        with("curvature_y on q_con", param(2, CurvatureY), 0.01),
        with("norm radius on q_bfs", param(3, NormRadius), 9.0),
        with("norm radius on diffractive", param(4, NormRadius), 5.0),
        with("norm radius on odd asphere", param(1, NormRadius), 5.0),
        with("norm radius on zernike", param(13, NormRadius), 11.0),
        with("design wavelength", param(5, DesignWavelengthUm), 0.65),
        with(
            "design wavelength on xy",
            param(11, DesignWavelengthUm),
            0.65,
        ),
        with("design order", param(6, DesignOrder), 2.0),
        with("trace order zero", param(4, TraceOrder), 0.0),
        with("trace order negative", param(5, TraceOrder), -3.0),
        with("trace order not integral", param(5, TraceOrder), 0.5),
        with("design order beyond i32", param(6, DesignOrder), 3e9),
        with("trace order on q_con", param(2, TraceOrder), 1.0),
        with(
            "curvature on diffractive plane base",
            ParamPath::Curvature { surface: id(4) },
            0.01,
        ),
        with(
            "conic on diffractive plane base",
            ParamPath::Conic { surface: id(4) },
            -1.0,
        ),
        with(
            "conic on diffractive conic base",
            ParamPath::Conic { surface: id(5) },
            -0.5,
        ),
        with(
            "asphere on diffractive even-asphere base",
            ParamPath::Asphere {
                surface: id(6),
                index: 3,
            },
            1e-12,
        ),
        with(
            "asphere on diffractive conic base",
            ParamPath::Asphere {
                surface: id(5),
                index: 0,
            },
            1e-6,
        ),
        with(
            "curvature on toroid",
            ParamPath::Curvature { surface: id(7) },
            0.01,
        ),
        with(
            "curvature on biconic",
            ParamPath::Curvature { surface: id(8) },
            0.01,
        ),
        with("conic on toroid", ParamPath::Conic { surface: id(7) }, -1.0),
        with(
            "curvature on q_bfs",
            ParamPath::Curvature { surface: id(3) },
            0.01,
        ),
        with("conic on q_bfs", ParamPath::Conic { surface: id(3) }, -1.0),
        with(
            "conic on zernike",
            ParamPath::Conic { surface: id(12) },
            -1.0,
        ),
        with(
            "asphere on odd asphere",
            ParamPath::Asphere {
                surface: id(1),
                index: 0,
            },
            1e-6,
        ),
    ]
}

/// The Cooke triplet (no configurations) with one edit each, for the
/// structural rules of `OpticalSystem::validation_issues`: values at and
/// beyond a limit, tables in and out of order, references to surfaces that do
/// or do not exist. A rejected case has exactly one issue, so a front end
/// can compare its own with the `[code] path` of `error`.
fn structural_validation_cases() -> Vec<ValidationCase> {
    let id = SurfaceId;
    let base = cooke_triplet();
    let case = |name: &str, edit: &dyn Fn(&mut OpticalSystem)| {
        let mut s = base.clone();
        s.name = name.into();
        edit(&mut s);
        let issues = s.validation_issues();
        assert!(
            issues.len() <= 1,
            "{name}: one problem per case: {issues:?}"
        );
        let error = s.validate().err().map(|e| e.to_string());
        ValidationCase {
            name: name.into(),
            system: s,
            error,
        }
    };
    let even_asphere = |terms: usize| {
        move |s: &mut OpticalSystem| {
            let mut coefficients = vec![0.0; terms];
            coefficients[0] = 1e-6;
            s.surfaces[2].shape = SurfaceShape::EvenAsphere {
                curvature: -1.0 / 435.76044,
                conic: 0.0,
                coefficients,
            }
        }
    };
    let material = |surface: usize, m: MaterialRef| {
        move |s: &mut OpticalSystem| s.surfaces[surface].material = m.clone()
    };
    let model = |nd: f64, vd: f64| MaterialRef::Model { nd, vd, dpgf: 0.0 };
    let offset = |dn_d: f64, dv_d: f64| MaterialRef::Catalog {
        catalog: None,
        name: "N-SK16".into(),
        offset: materials::MaterialOffset { dn_d, dv_d },
    };
    let coating = |name: &'static str, c: CoatingDef| {
        move |s: &mut OpticalSystem| {
            s.coatings.insert(name.into(), c.clone());
        }
    };
    let tabulated = |reflectance: &[(f64, f64)]| {
        coating(
            "T",
            CoatingDef::Tabulated {
                reflectance: reflectance.to_vec(),
            },
        )
    };
    // A quarter-wave MgF2 layer on a metal.
    struct Film {
        reference_wavelength_um: f64,
        layer: Layer,
        mirror_substrate: (f64, f64),
    }
    let film = |edit: &dyn Fn(&mut Film)| {
        let mut film = Film {
            reference_wavelength_um: 0.55,
            layer: Layer {
                material: MaterialRef::glass("MGF2"),
                thickness_um: 0.0,
                quarter_waves: Some(1.0),
                extinction: 0.0,
            },
            mirror_substrate: (0.12, 3.4),
        };
        edit(&mut film);
        coating(
            "T",
            CoatingDef::ThinFilm {
                reference_wavelength_um: film.reference_wavelength_um,
                layers: vec![film.layer],
                mirror_substrate: Some(film.mirror_substrate),
            },
        )
    };
    let sensor = |edit: &dyn Fn(&mut Sensor)| {
        let mut sensor = Sensor {
            pixels_x: 4000,
            pixels_y: 3000,
            pitch_x: 0.00376,
            pitch_y: 0.00376,
            max_cra_deg: Some(30.0),
            spectral_response: vec![(0.45, 0.6), (0.55, 1.0), (0.65, 0.7)],
        };
        edit(&mut sensor);
        move |s: &mut OpticalSystem| s.sensor = Some(sensor.clone())
    };
    let response = |table: &'static [(f64, f64)]| {
        sensor(&|sn: &mut Sensor| sn.spectral_response = table.to_vec())
    };
    let barrel = |from: u64, to: u64, inner_radius: f64, tis: f64| {
        move |s: &mut OpticalSystem| {
            s.mechanics = vec![MechanicalPart::Barrel {
                from: id(from),
                to: id(to),
                inner_radius,
                finish: ScatterModel::Lambertian { tis },
            }]
        }
    };
    let baffle = |surface: u64, inner_radius: f64, outer_radius: f64| {
        move |s: &mut OpticalSystem| {
            s.mechanics = vec![MechanicalPart::AnnularBaffle {
                surface: id(surface),
                z: -0.5,
                inner_radius,
                outer_radius,
                finish: ScatterModel::Lambertian { tis: 0.05 },
            }]
        }
    };
    let aperture =
        |a: ApertureShape| move |s: &mut OpticalSystem| s.surfaces[3].aperture = Some(a.clone());
    let group = |first: usize, last: u64, pivot: Option<u64>| {
        move |s: &mut OpticalSystem| {
            s.surfaces[first].element_placement = Some(ElementPlacement {
                last: id(last),
                pivot: pivot.map(id),
                motion: Placement {
                    tilt_x: 0.1,
                    ..Default::default()
                },
            })
        }
    };
    let overlay = |base: SurfaceShape| SurfaceShape::SagOverlay {
        base: Box::new(base),
        norm_radius: 8.0,
        test_wavelength_um: 0.6328,
        astigmatism: 0.5,
        astigmatism_angle_deg: 30.0,
        spherical: -0.25,
    };
    let doe = |base: SurfaceShape| SurfaceShape::Diffractive {
        base: Box::new(base),
        phase: DiffractivePhase {
            norm_radius: 8.0,
            coefficients: vec![-20.0],
            design_order: 1,
            design_wavelength_um: 0.5875618,
            trace_order: 1,
        },
    };
    let shape = |sh: SurfaceShape| move |s: &mut OpticalSystem| s.surfaces[2].shape = sh.clone();
    let sphere = || SurfaceShape::sphere(-435.76044);
    vec![
        // Structure of the system.
        case("stop on the object", &|s| s.stop = 0),
        case("object and image only", &|s| {
            s.surfaces.drain(1..8);
        }),
        case("repeated surface id", &|s| s.surfaces[2].id = id(1)),
        case("infinite thickness inside", &|s| {
            s.surfaces[3].thickness = f64::INFINITY
        }),
        case("unknown coating", &|s| {
            s.surfaces[2].coating = Some("nope".into())
        }),
        case("fixed semi-diameter of zero", &|s| {
            s.surfaces[3].semi_diameter = SemiDiameter::Fixed { value: 0.0 }
        }),
        case("coordinate break as image", &|s| {
            s.surfaces[8].shape = SurfaceShape::CoordinateBreak
        }),
        case("mirror as image", &|s| {
            s.surfaces[8].interaction = Interaction::Reflect
        }),
        // Aperture, fields, wavelengths, environment and aiming.
        case("aperture value of zero", &|s| {
            s.aperture = ApertureSpec::EntrancePupilDiameter { value: 0.0 }
        }),
        case("object-space NA at infinity", &|s| {
            s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 }
        }),
        case("object heights at infinity", &|s| {
            s.fields.kind = FieldKind::ObjectHeight
        }),
        case("no fields", &|s| s.fields.fields.clear()),
        case("field angle of 90 degrees", &|s| {
            s.fields.fields[2] = Field::new(0.0, 90.0)
        }),
        case("field weight of zero", &|s| s.fields.fields[1].weight = 0.0),
        case("negative field weight", &|s| {
            s.fields.fields[1].weight = -1.0
        }),
        case("pupil compression below 1", &|s| {
            s.fields.fields[2].vignetting.vcy = 0.99
        }),
        case("pupil compression of 1", &|s| {
            s.fields.fields[2].vignetting.vcy = 1.0
        }),
        case("negative pupil compression", &|s| {
            s.fields.fields[1].vignetting.vcx = -0.1
        }),
        case("wavelength of zero", &|s| {
            s.wavelengths.wavelengths[0].um = 0.0
        }),
        case("negative wavelength weight", &|s| {
            s.wavelengths.wavelengths[1].weight = -0.5
        }),
        case("primary wavelength out of range", &|s| {
            s.wavelengths.primary = 3
        }),
        case("temperature at absolute zero", &|s| {
            s.environment.temperature_c = -273.15
        }),
        case("vacuum pressure", &|s| s.environment.pressure_atm = 0.0),
        case("negative pressure", &|s| s.environment.pressure_atm = -1.0),
        case("real ray aiming", &|s| {
            s.ray_aiming = RayAiming::Real { tolerance: 1e-9 }
        }),
        case("ray aiming tolerance of zero", &|s| {
            s.ray_aiming = RayAiming::Real { tolerance: 0.0 }
        }),
        // Shapes.
        case("even asphere with the most coefficients", &even_asphere(20)),
        case("even asphere with too many coefficients", &even_asphere(21)),
        case("stacked overlays", &shape(overlay(overlay(sphere())))),
        case("overlay on an ideal lens", &|s| {
            s.surfaces[2].shape = SurfaceShape::SagOverlay {
                base: Box::new(SurfaceShape::Paraxial { focal_length: 50.0 }),
                norm_radius: 8.0,
                test_wavelength_um: 0.6328,
                astigmatism: 0.5,
                astigmatism_angle_deg: 0.0,
                spherical: 0.0,
            }
        }),
        case(
            "overlay on a diffractive surface",
            &shape(overlay(doe(sphere()))),
        ),
        case(
            "diffractive phase on an overlay",
            &shape(doe(overlay(sphere()))),
        ),
        case(
            "diffractive phase on stacked overlays",
            &shape(doe(overlay(overlay(sphere())))),
        ),
        case(
            "diffractive phase on an overlaid toroid",
            &shape(doe(overlay(overlay(SurfaceShape::Toroid {
                curvature_x: 0.01,
                curvature_y: 0.02,
                conic_y: 0.0,
                y_asphere: vec![],
            })))),
        ),
        // Materials.
        case(
            "model glass index above 1",
            &material(3, model(1.0001, 36.37)),
        ),
        case("model glass index of 1", &material(3, model(1.0, 36.37))),
        case("model glass index below 1", &material(3, model(0.9, 36.37))),
        case(
            "model glass with an Abbe number of zero",
            &material(3, model(1.62004, 0.0)),
        ),
        case(
            "constant index of zero",
            &material(3, MaterialRef::Constant { n: 0.0 }),
        ),
        case(
            "catalog offsets at their bounds",
            &material(1, offset(0.05, -10.0)),
        ),
        case(
            "catalog index offset beyond its bound",
            &material(1, offset(0.06, 0.0)),
        ),
        case(
            "catalog Abbe offset beyond its bound",
            &material(1, offset(0.0, -10.5)),
        ),
        // Surface apertures, scatter and edges.
        case(
            "annular aperture from the axis",
            &aperture(ApertureShape::Annular {
                inner_radius: 0.0,
                outer_radius: 6.0,
            }),
        ),
        case(
            "annular aperture inside out",
            &aperture(ApertureShape::Annular {
                inner_radius: 3.0,
                outer_radius: 2.0,
            }),
        ),
        case(
            "circular aperture of zero radius",
            &aperture(ApertureShape::Circular { radius: 0.0 }),
        ),
        case(
            "rectangular aperture without height",
            &aperture(ApertureShape::Rectangular {
                half_x: 3.0,
                half_y: 0.0,
            }),
        ),
        case(
            "elliptical aperture with a negative axis",
            &aperture(ApertureShape::Elliptical {
                semi_x: -3.0,
                semi_y: 2.0,
            }),
        ),
        case("surface scattering everything", &|s| {
            s.surfaces[5].scatter = Some(ScatterModel::Lambertian { tis: 1.0 })
        }),
        case("surface scattering more than it receives", &|s| {
            s.surfaces[5].scatter = Some(ScatterModel::Lambertian { tis: 1.5 })
        }),
        case("ABg scatter with a negative A", &|s| {
            s.surfaces[5].scatter = Some(ScatterModel::Abg {
                a: -1e-5,
                b: 1e-3,
                g: 2.0,
            })
        }),
        case("Gaussian edge scatter without a slope", &|s| {
            s.surfaces[2].edge.edge_scatter = Some(ScatterModel::Gaussian {
                tis: 0.1,
                sigma_slope: 0.0,
            })
        }),
        case("mechanical semi-diameter of zero", &|s| {
            s.surfaces[2].edge.mechanical_semi_diameter = Some(0.0)
        }),
        // Rigid groups.
        case("rigid group about its first vertex", &group(1, 2, Some(1))),
        case("rigid group ending before it starts", &group(3, 1, None)),
        case("rigid group ending at the image", &group(6, 8, None)),
        case("rigid group to a missing surface", &group(1, 99, None)),
        case("rigid group about a missing vertex", &group(1, 2, Some(99))),
        // Coatings.
        case(
            "tabulated coating",
            &tabulated(&[(0.4, 0.012), (0.55, 0.004), (0.7, 0.009)]),
        ),
        case("tabulated coating without samples", &tabulated(&[])),
        case(
            "tabulated coating out of order",
            &tabulated(&[(0.4, 0.012), (0.7, 0.009), (0.55, 0.004)]),
        ),
        case(
            "tabulated coating with a repeated wavelength",
            &tabulated(&[(0.4, 0.012), (0.4, 0.004), (0.7, 0.009)]),
        ),
        case(
            "tabulated coating with a wavelength of zero",
            &tabulated(&[(0.0, 0.012), (0.55, 0.004)]),
        ),
        case(
            "tabulated coating reflecting more than it receives",
            &tabulated(&[(0.4, 0.012), (0.55, 1.5)]),
        ),
        case(
            "ideal coating gaining energy",
            &coating(
                "T",
                CoatingDef::Ideal {
                    reflectance: 0.7,
                    transmittance: 0.5,
                },
            ),
        ),
        case(
            "coating name with quotes",
            &coating(
                r#"the "best" AR\V"#,
                CoatingDef::Ideal {
                    reflectance: -0.1,
                    transmittance: 0.5,
                },
            ),
        ),
        case("thin film", &film(&|_| {})),
        case(
            "thin film with a reference wavelength of zero",
            &film(&|f| f.reference_wavelength_um = 0.0),
        ),
        case(
            "thin film with a negative thickness",
            &film(&|f| f.layer.thickness_um = -0.1),
        ),
        case(
            "thin film with a negative quarter-wave count",
            &film(&|f| f.layer.quarter_waves = Some(-1.0)),
        ),
        case(
            "thin film with a negative extinction",
            &film(&|f| f.layer.extinction = -1e-4),
        ),
        case(
            "thin film of a model glass index of 1",
            &film(&|f| f.layer.material = model(1.0, 60.0)),
        ),
        case(
            "thin film on a substrate of index zero",
            &film(&|f| f.mirror_substrate.0 = 0.0),
        ),
        case(
            "thin film on a substrate with gain",
            &film(&|f| f.mirror_substrate.1 = -3.4),
        ),
        // Mechanics.
        case("barrel", &barrel(1, 7, 15.0, 0.05)),
        case("barrel from a missing surface", &barrel(42, 7, 15.0, 0.05)),
        case("barrel to a missing surface", &barrel(1, 42, 15.0, 0.05)),
        case("barrel without a bore", &barrel(1, 7, 0.0, 0.05)),
        case(
            "barrel finish scattering more than it receives",
            &barrel(1, 7, 15.0, 1.5),
        ),
        case("baffle", &baffle(5, 0.0, 16.0)),
        case("baffle on a missing surface", &baffle(42, 5.5, 16.0)),
        case("baffle inside out", &baffle(5, 16.0, 5.5)),
        case(
            "baffle with a negative inner radius",
            &baffle(5, -1.0, 16.0),
        ),
        // Sensor.
        case("sensor", &sensor(&|_| {})),
        case(
            "sensor with a flat response",
            &sensor(&|sn| sn.spectral_response.clear()),
        ),
        case("sensor without pixels", &sensor(&|sn| sn.pixels_x = 0)),
        case(
            "sensor with a pitch of zero",
            &sensor(&|sn| sn.pitch_y = 0.0),
        ),
        case(
            "sensor response out of order",
            &response(&[(0.55, 1.0), (0.45, 0.6), (0.65, 0.7)]),
        ),
        case(
            "sensor response with a repeated wavelength",
            &response(&[(0.45, 0.6), (0.55, 1.0), (0.55, 0.7)]),
        ),
        case(
            "sensor response at a wavelength of zero",
            &response(&[(0.0, 0.6), (0.55, 1.0)]),
        ),
        case(
            "sensor response below zero",
            &response(&[(0.45, -0.1), (0.55, 1.0)]),
        ),
    ]
}

fn run_info(snapshot: &Snapshot) -> RunInfo {
    RunInfo {
        engine_version: ENGINE_VERSION.to_string(),
        snapshot_hash: snapshot.hash().to_hex(),
        analysis: "spot_diagram".into(),
        settings: serde_json::json!({
            "pattern": "hexapolar",
            "rings": 6,
            "seed": 42,
            "fields": {"type": "all"},
        }),
        backend: "metal".into(),
        precision: Precision::F32Gpu,
        seed: Some(42),
        started_unix_ms: 1_767_225_600_000,
        duration_ms: 12.5,
        tier: Default::default(),
        aim_stop: Default::default(),
        warnings: vec!["example warning".into()],
        failed_attempts: vec![],
    }
}

/// The syntax of every solve kind (`docs/SOLVES.md` §6). Not one valid
/// system: only for the front ends' serde mirrors.
fn solve_kinds() -> Vec<Solve> {
    let id = SurfaceId;
    let thickness = |s| ParamPath::Thickness { surface: id(s) };
    let curvature = |s| ParamPath::Curvature { surface: id(s) };
    vec![
        Solve::new(thickness(2), SolveKind::MarginalRayHeight { height: 0.0 }),
        Solve::new(curvature(2), SolveKind::MarginalRayAngle { slope: -0.1 }),
        Solve::new(thickness(1), SolveKind::ChiefRayHeight { height: 1.0 }),
        Solve::new(curvature(1), SolveKind::ChiefRayAngle { slope: 0.0 }),
        Solve::pickup(
            ParamPath::Conic { surface: id(1) },
            ParamPath::Conic { surface: id(2) },
            2.0,
            -1.0,
        ),
        Solve::new(
            thickness(0),
            SolveKind::Position {
                from: id(1),
                to: id(3),
                distance: 100.0,
            },
        ),
        Solve::new(
            ParamPath::DecenterX { surface: id(1) },
            SolveKind::CenterOfCurvature { surface: id(3) },
        ),
        Solve::new(
            thickness(1),
            SolveKind::EdgeThickness {
                value: 2.0,
                radius: Some(8.0),
            },
        ),
        Solve::new(
            thickness(2),
            SolveKind::EdgeThickness {
                value: 1.5,
                radius: None,
            },
        ),
        Solve::f_number(id(2), 4.0),
    ]
}

/// A valid system with solves and configurations: the Cooke triplet with a
/// focus solve, an F/# solve on the last curvature, a symmetric pickup, a
/// total-track position solve and a chief-ray angle (slope) solve, plus a
/// near-focus configuration.
fn solved_triplet() -> OpticalSystem {
    let id = SurfaceId;
    let mut s = cooke_triplet();
    s.name = "Cooke triplet with solves".into();
    s.solves = vec![
        Solve::marginal_ray_height(id(7), 0.0),
        Solve::f_number(id(6), 4.5),
        Solve::pickup(
            ParamPath::Thickness { surface: id(4) },
            ParamPath::Thickness { surface: id(2) },
            0.75,
            0.25,
        ),
        Solve::new(
            ParamPath::Thickness { surface: id(5) },
            SolveKind::Position {
                from: id(1),
                to: id(7),
                distance: 21.0,
            },
        ),
        Solve::new(
            ParamPath::Curvature { surface: id(3) },
            SolveKind::ChiefRayAngle { slope: 0.25 },
        ),
        Solve::new(
            ParamPath::Thickness { surface: id(1) },
            SolveKind::EdgeThickness {
                value: 1.0,
                radius: None,
            },
        ),
    ];
    s.configurations = vec![
        Configuration {
            name: "infinity".into(),
            overrides: vec![],
        },
        Configuration {
            name: "2 m".into(),
            overrides: vec![(
                ParamPath::Thickness {
                    surface: SurfaceId(0),
                },
                2000.0,
            )],
        },
    ];
    s
}

/// Every built-in sample, as `{"id", "system"}` (the app's "New from
/// Sample" menu ships this file).
#[derive(Serialize)]
struct SampleEntry {
    id: &'static str,
    system: OpticalSystem,
}

/// Where a generated file lives in the repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// `apps/macos/Tests/MKLightModelTests/Fixtures`.
    TestFixture,
    /// `apps/macos/Sources/MKLightModel/Resources`: shipped in the app, and
    /// read by its tests through the model's own loader.
    AppResource,
}

impl Place {
    pub fn directory(self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(match self {
            Place::TestFixture => "../../apps/macos/Tests/MKLightModelTests/Fixtures",
            Place::AppResource => "../../apps/macos/Sources/MKLightModel/Resources",
        })
    }
}

/// One generated file: its place, name and exact text.
pub struct FixtureFile {
    pub place: Place,
    pub name: &'static str,
    pub json: String,
}

fn file<T: Serialize>(place: Place, name: &'static str, value: &T) -> FixtureFile {
    let mut json = serde_json::to_string_pretty(value).expect("serialise fixture");
    json.push('\n');
    FixtureFile { place, name, json }
}

/// Every file this generator writes. Nothing in them depends on the run.
pub fn fixtures() -> Vec<FixtureFile> {
    let fixture = Place::TestFixture;
    // Fixtures meant to be valid models must pass validation.
    let cooke = Snapshot::new(cooke_triplet()).expect("cooke triplet is valid");
    let sink = Snapshot::new(kitchen_sink()).expect("kitchen sink is valid");
    let empty = Snapshot::new(OpticalSystem::new_empty()).expect("empty system is valid");
    let extended = Snapshot::new(extended_shapes()).expect("extended shapes are valid");
    // Solves and pickups (docs/SOLVES.md §6).
    let solved = Snapshot::new(solved_triplet()).expect("solved triplet is valid");

    // Built-in samples, for the app bundle and its tests.
    let samples: Vec<SampleEntry> = samples::all()
        .into_iter()
        .map(|(id, system)| {
            Snapshot::new(system.clone()).expect("samples are valid");
            SampleEntry { id, system }
        })
        .collect();

    // Snapshot hashes of the valid fixtures, so a front end can check that its
    // engine binding reproduces the engine's content hashes.
    let hashes: BTreeMap<&str, String> = [
        ("cooke_triplet.json", &cooke),
        ("empty.json", &empty),
        ("kitchen_sink.json", &sink),
        ("solved_triplet.json", &solved),
        ("extended_shapes.json", &extended),
    ]
    .into_iter()
    .map(|(name, snapshot)| (name, snapshot.hash().to_hex()))
    .collect();

    vec![
        file(fixture, "empty.json", empty.system()),
        file(fixture, "cooke_triplet.json", cooke.system()),
        file(fixture, "kitchen_sink.json", sink.system()),
        file(fixture, "extended_shapes.json", extended.system()),
        file(fixture, "variants.json", &variants()),
        file(fixture, "param_paths.json", &all_param_paths()),
        file(fixture, "run_info.json", &run_info(&cooke)),
        file(fixture, "validation_cases.json", &validation_cases()),
        file(fixture, "solve_kinds.json", &solve_kinds()),
        file(fixture, "solved_triplet.json", solved.system()),
        file(Place::AppResource, "samples.json", &samples),
        file(fixture, "snapshot_hashes.json", &hashes),
    ]
}

pub fn main() {
    let out_dir = std::env::args().nth(1).map(PathBuf::from);
    for f in fixtures() {
        let dir = out_dir.clone().unwrap_or_else(|| f.place.directory());
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        let path = dir.join(f.name);
        std::fs::write(&path, f.json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
