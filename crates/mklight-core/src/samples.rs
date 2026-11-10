//! Sample optical systems: shared test fixtures for every crate and the app.
//!
//! Catalog glasses come from the built-in library; the cylindrical adapter
//! also uses constant-index model glass. All samples validate and trace every
//! field at every wavelength without failures. Image distances were set
//! with the reference tracer's paraxial solve (and, where noted, refined to
//! the best RMS focus), so each sample is well focused as delivered. The tests
//! in `mklight-trace` check this.

use crate::model::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, MaterialRef, OpticalSystem, Placement,
    Surface, SurfaceShape, WavelengthSet,
};

fn s(id: u64, label: &str, shape: SurfaceShape, thickness: f64, material: MaterialRef) -> Surface {
    let mut s = Surface::new(id, shape, thickness, material);
    s.label = label.to_string();
    s
}

fn sph(r: f64) -> SurfaceShape {
    SurfaceShape::sphere(r)
}

fn air() -> MaterialRef {
    MaterialRef::Air
}

fn glass(name: &str) -> MaterialRef {
    MaterialRef::glass(name)
}

fn object_at_infinity() -> Surface {
    s(0, "OBJ", SurfaceShape::Plane, f64::INFINITY, air())
}

fn image(id: u64) -> Surface {
    s(id, "IMA", SurfaceShape::Plane, 0.0, air())
}

fn fields(kind: FieldKind, ys: &[f64]) -> FieldSet {
    FieldSet {
        kind,
        fields: ys.iter().map(|&y| Field::new(0.0, y)).collect(),
    }
}

fn system(name: &str, notes: &str) -> OpticalSystem {
    let mut sys = OpticalSystem::new_empty();
    sys.name = name.to_string();
    sys.notes = notes.to_string();
    sys.wavelengths = WavelengthSet::default();
    sys
}

/// Every sample, with a short identifier.
pub fn all() -> Vec<(&'static str, OpticalSystem)> {
    vec![
        ("singlet", singlet()),
        ("achromat", achromatic_doublet()),
        ("cooke_triplet", cooke_triplet()),
        ("double_gauss", double_gauss()),
        ("newtonian", newtonian_telescope()),
        ("petzval_asphere", petzval_asphere()),
        ("relay", finite_relay()),
        ("hybrid_doe", hybrid_diffractive_achromat()),
        ("anamorphic", anamorphic_adapter()),
        ("biconic", biconic_element()),
        ("toroid", toroidal_element()),
        ("q_con", q_asphere_lens()),
        ("q_bfs", qbfs_lens()),
        ("odd_asphere", odd_asphere_element()),
        ("xy_freeform", xy_freeform()),
        ("zernike_sag", zernike_sag()),
    ]
}

/// A thin-ish N-BK7 singlet, f ≈ 100 mm, F/5, ±5° field.
///
/// Purpose: the simplest refracting fixture (spherical aberration, coma,
/// astigmatism and colour all present). Provenance: a near "best-form"
/// shape (R₁/R₂ ≈ −1/6) designed here from the thin-lens bending formulas;
/// the image distance is the paraxial back focal distance at d-light.
pub fn singlet() -> OpticalSystem {
    let mut sys = system("Singlet f100 F/5", "N-BK7 near-best-form singlet");
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 20.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 3.5, 5.0]);
    sys.surfaces = vec![
        object_at_infinity(),
        s(1, "front", sph(60.0), 5.0, glass("N-BK7")),
        s(2, "rear", sph(-350.0), SINGLET_BFL, air()),
        image(3),
    ];
    sys.stop = 1;
    sys
}
const SINGLET_BFL: f64 = 96.696_728_7;

/// Cemented achromatic doublet, f ≈ 100 mm, F/5 (image-space F/#), ±2°.
///
/// Purpose: colour-corrected cemented interface (the low index contrast at
/// the cemented surface exercises the stable Snell form). Provenance:
/// designed here as a crown-first N-BK7 / F2 Fraunhofer-type doublet:
/// thin-lens achromatisation (element powers in proportion to the Abbe
/// numbers), bending chosen by a ray-traced scan for minimum on-axis
/// spherical aberration, centre thicknesses 5.0 / 2.5 mm, curvatures scaled
/// for f = 100 mm. The image is at the paraxial focus.
pub fn achromatic_doublet() -> OpticalSystem {
    let mut sys = system("Achromat f100 F/5", "cemented N-BK7 / F2 doublet");
    sys.aperture = ApertureSpec::ImageSpaceFNumber { value: 5.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 1.4, 2.0]);
    sys.surfaces = vec![object_at_infinity()];
    sys.surfaces.extend(achromat(1, false, ACHROMAT_BFL));
    sys.surfaces.push(image(4));
    sys.stop = 1;
    sys
}
const ACHROMAT_BFL: f64 = 95.085_347_9;

/// The three surfaces of the f = 100 mm achromat, ids `id0..id0+3`, either
/// crown-first (`reversed = false`) or flint-first. `t_after` follows the
/// last surface.
fn achromat(id0: u64, reversed: bool, t_after: f64) -> Vec<Surface> {
    const R: [f64; 3] = [47.40, -42.54, -410.38];
    if !reversed {
        vec![
            s(id0, "crown front", sph(R[0]), 5.0, glass("N-BK7")),
            s(id0 + 1, "cemented", sph(R[1]), 2.5, glass("F2")),
            s(id0 + 2, "flint rear", sph(R[2]), t_after, air()),
        ]
    } else {
        vec![
            s(id0, "flint front", sph(-R[2]), 2.5, glass("F2")),
            s(id0 + 1, "cemented", sph(-R[1]), 5.0, glass("N-BK7")),
            s(id0 + 2, "crown rear", sph(-R[0]), t_after, air()),
        ]
    }
}

/// Cooke triplet, f ≈ 50 mm, F/4, ±20°.
///
/// Purpose: the classic three-element anastigmat; a mid-complexity fixture
/// with the stop inside the lens. Provenance: the well-known textbook
/// N-SK16 / F2 / N-SK16 triplet of 50 mm focal length and 40° full field
/// (originally F/5) with N-SK16 for SK16, opened here to F/4. The image is at
/// the RMS best focus.
///
/// L3 is edged to a Ø17.2 mm mount ([`COOKE_L3_MECHANICAL_SD`]), 0.4–0.6 mm
/// beyond its clear apertures (≈ 8.0 and 8.2 mm): its edge is 0.61 mm thick
/// at the clear aperture, 0.35 mm at the mount, and would be negative at the
/// default 1 mm mechanical margin of the stray-light scene.
pub fn cooke_triplet() -> OpticalSystem {
    let mut sys = system("Cooke triplet f50 F/4", "SK16/F2/SK16 triplet, 40° field");
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 12.5 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 14.0, 20.0]);
    sys.surfaces = vec![
        object_at_infinity(),
        s(1, "L1 front", sph(22.01359), 3.25896, glass("N-SK16")),
        s(2, "L1 rear", sph(-435.7604), 6.00755, air()),
        s(3, "L2 front", sph(-22.21328), 0.99997, glass("F2")),
        s(4, "L2 rear / stop", sph(20.29192), 4.750409, air()),
        s(5, "L3 front", sph(79.68360), 2.952076, glass("N-SK16")),
        s(6, "L3 rear", sph(-18.39533), COOKE_BFL, air()),
        image(7),
    ];
    for j in [5, 6] {
        sys.surfaces[j].edge.mechanical_semi_diameter = Some(COOKE_L3_MECHANICAL_SD);
    }
    sys.stop = 4;
    sys
}
const COOKE_BFL: f64 = 42.252_162;
/// Mechanical semi-diameter of the Cooke triplet's L3 (mm).
pub const COOKE_L3_MECHANICAL_SD: f64 = 8.6;

/// Six-element Double-Gauss camera lens, f ≈ 50 mm, F/2, ±14°.
///
/// Purpose: a realistic camera lens with two cemented doublets, a central
/// stop and strongly curved surfaces. Provenance: derived from the classic
/// F/3, 100 mm, 28° full-field Double-Gauss example of the lens-design
/// literature: its crown and flint were replaced by N-SK16 and F2 from the
/// built-in catalog, the lens was opened to F/2, the eight curved surfaces
/// were re-optimised here (coordinate descent on the polychromatic RMS spot
/// over all fields) and the result was scaled to f ≈ 50 mm.
///
/// Like real fast Double-Gauss lenses it vignettes off axis: the 10° and 14°
/// fields carry the meridional pupil compression `vcy` = 0.15 and 0.30
/// (CONVENTIONS §3), so the automatic clear apertures of the outer elements
/// (L1 ≈ 15.6 mm, L6 ≈ 11.5 mm) leave every element a positive edge
/// thickness (≥ 0.8 mm at the clear aperture, ≥ 0.3 mm at the default 1 mm
/// mechanical margin) and no surfaces cross inside a clear aperture. Without
/// vignetting the 14° beam needs clear apertures where L1 and L6 have
/// negative edge thickness. The image is at the RMS best focus of the
/// vignetted fields (≈ 20 µm polychromatic RMS spot).
pub fn double_gauss() -> OpticalSystem {
    let mut sys = system(
        "Double Gauss f50 F/2",
        "six-element Double-Gauss, 28° field",
    );
    sys.aperture = ApertureSpec::ImageSpaceFNumber { value: 2.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 10.0, 14.0]);
    sys.fields.fields[1].vignetting.vcy = DOUBLE_GAUSS_VCY[0];
    sys.fields.fields[2].vignetting.vcy = DOUBLE_GAUSS_VCY[1];
    let sk = || glass("N-SK16");
    let fl = || glass("F2");
    sys.surfaces = vec![
        object_at_infinity(),
        s(1, "L1 front", sph(28.074), 4.62, sk()),
        s(2, "L1 rear", sph(86.538), 0.26, air()),
        s(3, "L2 front", sph(19.301), 7.40, sk()),
        s(4, "L2/L3 cemented", SurfaceShape::Plane, 2.00, fl()),
        s(5, "L3 rear", sph(11.540), 7.53, air()),
        s(6, "stop", SurfaceShape::Plane, 6.57, air()),
        s(7, "L4 front", sph(-13.853), 2.00, fl()),
        s(8, "L4/L5 cemented", SurfaceShape::Plane, 5.72, sk()),
        s(9, "L5 rear", sph(-19.260), 0.26, air()),
        s(10, "L6 front", sph(88.609), 3.62, sk()),
        s(11, "L6 rear", sph(-33.445), DOUBLE_GAUSS_BFL, air()),
        image(12),
    ];
    sys.stop = 6;
    sys
}
const DOUBLE_GAUSS_BFL: f64 = 28.955_554;
/// Meridional pupil compression `vcy` of the 10° and 14° fields.
const DOUBLE_GAUSS_VCY: [f64; 2] = [0.15, 0.30];

/// Newtonian-style reflecting telescope: F/5 paraboloidal primary
/// (f = 500 mm, D = 100 mm) and a 45° flat fold mirror placed with
/// coordinate breaks, ±0.35° field.
///
/// Purpose: mirrors (signed reduced indices, negative thicknesses after an
/// odd number of reflections) and coordinate-break folds. Provenance: the
/// textbook Newtonian layout; a paraboloid images an axial point at
/// infinity perfectly. The fold flat sits 400 mm in front of the primary and
/// sends the beam along global −y to a focus 100 mm from the flat. The
/// central obstruction is not modelled.
pub fn newtonian_telescope() -> OpticalSystem {
    let mut sys = system(
        "Newtonian telescope f500 F/5",
        "paraboloidal primary with 45° fold flat",
    );
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 100.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 0.25, 0.35]);
    let mut primary = s(
        1,
        "primary",
        SurfaceShape::Conic {
            curvature: -1.0 / 1000.0,
            conic: -1.0,
        },
        -400.0,
        air(),
    );
    primary.interaction = Interaction::Reflect;
    let fold_cb = |id: u64, label: &str, t: f64| {
        let mut c = s(id, label, SurfaceShape::CoordinateBreak, t, air());
        c.placement = Some(Placement {
            tilt_x: 45.0,
            ..Default::default()
        });
        c
    };
    let mut flat = s(3, "fold flat", SurfaceShape::Plane, 0.0, air());
    flat.interaction = Interaction::Reflect;
    sys.surfaces = vec![
        object_at_infinity(),
        primary,
        fold_cb(2, "fold in", 0.0),
        flat,
        fold_cb(4, "fold out", 100.0),
        image(5),
    ];
    sys.stop = 1;
    sys
}

/// Petzval-type lens with an even asphere: two cemented achromats
/// (f ≈ 100 mm each) 50 mm apart, f ≈ 69 mm, F/4, real-image-height fields up
/// to 6 mm.
///
/// Purpose: an even-asphere fixture (Newton intersection), a stop at the
/// front group, and real-image-height fields (solved by real-ray iteration).
/// Provenance: the classic Petzval arrangement of two separated positive
/// achromats, built here from the doublet of [`achromatic_doublet`]; the
/// A4 and A6 terms on the last surface were chosen by a ray-traced scan to
/// reduce the spherical aberration, and the image is at the RMS best focus.
pub fn petzval_asphere() -> OpticalSystem {
    let mut sys = system(
        "Petzval asphere f69 F/4",
        "two achromats with an aspheric last surface",
    );
    sys.aperture = ApertureSpec::ImageSpaceFNumber { value: 4.0 };
    sys.fields = fields(FieldKind::RealImageHeight, &[0.0, 4.0, 6.0]);
    sys.surfaces = vec![object_at_infinity()];
    sys.surfaces.extend(achromat(1, false, 50.0));
    let mut rear = achromat(4, false, PETZVAL_BFL);
    rear[2].label = "flint rear (asphere)".into();
    rear[2].shape = SurfaceShape::EvenAsphere {
        curvature: rear[2].shape.curvature(),
        conic: 0.0,
        coefficients: vec![PETZVAL_A4, PETZVAL_A6],
    };
    sys.surfaces.extend(rear);
    sys.surfaces.push(image(7));
    sys.stop = 1;
    sys
}
const PETZVAL_A4: f64 = 3.2e-6;
const PETZVAL_A6: f64 = 8.0e-10;
const PETZVAL_BFL: f64 = 26.185_686;

/// Finite-conjugate 1:1 relay: two achromats (f ≈ 100 mm) facing each
/// other around a central stop, object NA 0.08, object heights up to 5 mm.
///
/// Purpose: finite object distance, object-space NA aperture, object-height
/// fields, magnification ≈ −1. Provenance: the standard "two achromats,
/// infinity space in between" relay; the first achromat is reversed so both
/// face their collimated side to the stop. The object sits at the front
/// focal point of the first lens and the image distance is the paraxial one.
pub fn finite_relay() -> OpticalSystem {
    let mut sys = system("1:1 achromat relay NA 0.08", "two back-to-back achromats");
    sys.aperture = ApertureSpec::ObjectSpaceNa { value: 0.08 };
    sys.fields = fields(FieldKind::ObjectHeight, &[0.0, 3.0, 5.0]);
    sys.surfaces = vec![s(
        0,
        "OBJ",
        SurfaceShape::Plane,
        RELAY_OBJECT_DISTANCE,
        air(),
    )];
    sys.surfaces.extend(achromat(1, true, 10.0));
    sys.surfaces
        .push(s(4, "stop", SurfaceShape::Plane, 10.0, air()));
    sys.surfaces
        .extend(achromat(5, false, RELAY_IMAGE_DISTANCE));
    sys.surfaces.push(image(8));
    sys.stop = 4;
    sys
}
const RELAY_OBJECT_DISTANCE: f64 = 95.085_347_9;
const RELAY_IMAGE_DISTANCE: f64 = 95.085_347_9;

/// Whether `system` has a surface outside the basic shapes
/// ([`SurfaceShape::is_legacy`]). Test suites select the extended-surface
/// samples, or the samples without them, by it.
pub fn has_extended_surfaces(system: &OpticalSystem) -> bool {
    system.surfaces.iter().any(|s| !s.shape.is_legacy())
}

fn bk7_index(wavelength_um: f64) -> f64 {
    crate::materials::MaterialLibrary::with_builtin()
        .resolve(&glass("N-BK7"), &Default::default())
        .expect("built-in glass")
        .index(wavelength_um)
}

/// N-BK7 plano-convex singlet with a rear DOE: exactly equal paraxial F/C
/// back focal lengths including the 3 mm glass transfer, not a thin-lens fit.
pub fn hybrid_diffractive_achromat() -> OpticalSystem {
    use crate::model::surface::DiffractivePhase;
    let mut sys = system(
        "Hybrid refractive-diffractive achromat",
        "N-BK7 singlet, fixed-period DOE; F/C BFL corrected",
    );
    let (lf, lc, ld) = (0.4861327, 0.6562725, 0.5875618);
    let c = 0.02;
    let t = 3.0;
    let r = 5.0;
    let incident_vergence = |l: f64| {
        let n = bk7_index(l);
        let p = (n - 1.0) * c;
        p / (1.0 - t * p / n)
    };
    let k = (incident_vergence(lc) - incident_vergence(lf)) / ((lf - lc) * 1e-3);
    let bfl = 1.0 / (incident_vergence(ld) + k * ld * 1e-3);
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 4.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 0.5, 1.0]);
    sys.surfaces = vec![
        object_at_infinity(),
        s(1, "singlet front", sph(1.0 / c), t, glass("N-BK7")),
        s(
            2,
            "rear DOE",
            SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::Plane),
                phase: DiffractivePhase {
                    norm_radius: r,
                    coefficients: vec![-std::f64::consts::PI * r * r * k],
                    design_order: 1,
                    design_wavelength_um: ld,
                    trace_order: 1,
                },
            },
            bfl,
            air(),
        ),
        image(3),
    ];
    sys.stop = 1;
    sys
}

fn cylinder(curvature_x: f64) -> SurfaceShape {
    SurfaceShape::Biconic {
        curvature_x,
        curvature_y: 0.0,
        conic_x: 0.0,
        conic_y: 0.0,
        x_asphere: vec![],
        y_asphere: vec![],
    }
}

/// Galilean cylindrical adapter, 2× beam expansion in x, in front of the
/// singlet. Constant n=1.5 adapter glass makes its 2× squeeze achromatic;
/// the imaging singlet retains its real N-BK7 dispersion.
pub fn anamorphic_adapter() -> OpticalSystem {
    let mut sys = singlet();
    sys.name = "2x cylindrical adapter and singlet".into();
    sys.notes = "Two finite-thickness cylindrical elements; EFL_y/EFL_x=2".into();
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 4.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 0.2, 0.4]);
    let mut body = vec![
        object_at_infinity(),
        s(
            10,
            "negative cylinder",
            cylinder(-0.02),
            2.0,
            MaterialRef::Constant { n: 1.5 },
        ),
        s(
            11,
            "adapter first rear",
            SurfaceShape::Plane,
            100.0 - 2.0 / 1.5,
            air(),
        ),
        s(
            12,
            "positive cylinder",
            cylinder(0.01),
            2.0,
            MaterialRef::Constant { n: 1.5 },
        ),
        s(13, "adapter second rear", SurfaceShape::Plane, 5.0, air()),
    ];
    body.extend(sys.surfaces.into_iter().skip(1));
    sys.surfaces = body;
    sys.stop = 1;
    sys
}

fn extended_singlet(name: &str, shape: SurfaceShape) -> OpticalSystem {
    let mut sys = singlet();
    sys.name = name.into();
    sys.notes = "CPU reference surface fixture, positive edge thickness".into();
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 4.0 };
    sys.fields = fields(FieldKind::ObjectAngle, &[0.0, 0.5, 1.0]);
    sys.surfaces[1].shape = shape;
    // The y-axis d-line back focus from the two finite-thickness powers.
    let n = bk7_index(0.5875618);
    let p1 = (n - 1.0) * crate::surface_math::vertex_curvatures(&sys.surfaces[1].shape)[2];
    let y = 1.0 - sys.surfaces[1].thickness * p1 / n;
    let p2 = (1.0 - n) * sys.surfaces[2].shape.curvature();
    sys.surfaces[2].thickness = y / (p1 + p2 * y);
    sys
}

pub fn biconic_element() -> OpticalSystem {
    extended_singlet(
        "Biconic element",
        SurfaceShape::Biconic {
            curvature_x: 0.02,
            curvature_y: 1.0 / 60.0,
            conic_x: -0.5,
            conic_y: 0.0,
            x_asphere: vec![1e-6],
            y_asphere: vec![-1e-6],
        },
    )
}
pub fn toroidal_element() -> OpticalSystem {
    extended_singlet(
        "Toroidal element",
        SurfaceShape::Toroid {
            curvature_x: 0.02,
            curvature_y: 1.0 / 60.0,
            conic_y: -0.5,
            y_asphere: vec![1e-6],
        },
    )
}
pub fn q_asphere_lens() -> OpticalSystem {
    extended_singlet(
        "Q-conic asphere lens",
        SurfaceShape::QCon {
            curvature: 1.0 / 60.0,
            conic: -0.4,
            norm_radius: 5.0,
            coefficients: vec![0.002, -0.0004, 0.0001],
        },
    )
}
pub fn qbfs_lens() -> OpticalSystem {
    extended_singlet(
        "Q-BFS asphere lens",
        SurfaceShape::QBfs {
            curvature: 1.0 / 60.0,
            norm_radius: 5.0,
            coefficients: vec![0.002, -0.0004, 0.0001],
        },
    )
}
pub fn odd_asphere_element() -> OpticalSystem {
    // r¹=0 makes a regular first-order focus; the nonzero r³ is genuinely odd.
    extended_singlet(
        "Odd asphere element",
        SurfaceShape::OddAsphere {
            curvature: 1.0 / 60.0,
            conic: 0.0,
            coefficients: vec![0.0, 1e-5, -1e-6, 1e-7],
        },
    )
}
pub fn xy_freeform() -> OpticalSystem {
    extended_singlet(
        "XY freeform corrector",
        SurfaceShape::XYPolynomial {
            curvature: 1.0 / 60.0,
            conic: 0.0,
            norm_radius: 5.0,
            coefficients: vec![
                0.0, 0.0, 0.001, 0.0, -0.001, 0.0002, -0.0001, 0.0003, 0.0001,
            ],
        },
    )
}
pub fn zernike_sag() -> OpticalSystem {
    extended_singlet(
        "Zernike sag element",
        SurfaceShape::ZernikeSag {
            curvature: 1.0 / 60.0,
            conic: 0.0,
            norm_radius: 5.0,
            ordering: crate::model::surface::ZernikeOrdering::Fringe,
            coefficients: vec![0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0002],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_validate_and_use_builtin_glasses() {
        let lib = crate::materials::MaterialLibrary::with_builtin();
        for (name, sys) in all() {
            sys.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            for surf in &sys.surfaces {
                if let MaterialRef::Catalog { name: g, .. } = &surf.material {
                    lib.find_glass(Some(crate::materials::builtin::BUILTIN_CATALOG_NAME), g)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                }
            }
        }
    }
}
