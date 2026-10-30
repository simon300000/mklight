//! Shared fixtures for tests in this crate and in backend crates: the packing
//! singlet and the trace cross-validation systems (every sample of
//! `mklight_core::samples`, vignetting variants, and systems that exercise
//! what the samples do not: tilts and decentres, ideal lenses, every aperture
//! shape, TIR, high-order aspheres, a 1000 mm track, an absorbing glass).

use std::sync::Arc;

use mklight_core::materials::{GlassCatalog, MaterialLibrary, TransmittancePoint};
use mklight_core::model::{
    ApertureShape, ApertureSpec, Field, FieldKind, FieldSet, Interaction, MaterialRef,
    OpticalSystem, Placement, SemiDiameter, Surface,
};
use mklight_core::{ResolvedSystem, Snapshot, SurfaceShape, samples};
use mklight_trace::{SemiDiameterSettings, SequentialTracer};

/// A biconvex N-BK7 singlet with an even-asphere rear surface, a fixed
/// semi-diameter and a rectangular aperture: exercises every packed field.
pub fn singlet() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::sphere(50.0),
            5.0,
            MaterialRef::glass("N-BK7"),
        ),
        Surface::new(
            2,
            SurfaceShape::EvenAsphere {
                curvature: -1.0 / 50.0,
                conic: -0.8,
                coefficients: vec![1.2e-5, -3.4e-8, 0.0, 5.0e-14, 0.0],
            },
            45.123456789,
            MaterialRef::Air,
        ),
        Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 12.5 };
    s.surfaces[2].aperture = Some(ApertureShape::Rectangular {
        half_x: 10.0,
        half_y: 8.0,
    });
    s.stop = 1;
    s
}

/// The built-in glasses plus catalog `TEST` with `ABSORB`: N-BK7's
/// dispersion with an internal transmittance of 0.80 … 0.95 per 10 mm
/// across the visible.
pub fn library() -> Arc<MaterialLibrary> {
    let mut lib = MaterialLibrary::with_builtin();
    let bk7 = lib
        .find_glass(
            Some(mklight_core::materials::builtin::BUILTIN_CATALOG_NAME),
            "N-BK7",
        )
        .expect("N-BK7")
        .clone();
    let mut g = (*bk7).clone();
    g.name = "ABSORB".into();
    g.transmittance = [(0.40, 0.80), (0.50, 0.90), (0.70, 0.95)]
        .iter()
        .map(|&(um, tau)| TransmittancePoint {
            um,
            tau,
            thickness_mm: 10.0,
        })
        .collect();
    let mut cat = GlassCatalog {
        name: "TEST".into(),
        glasses: Default::default(),
    };
    cat.glasses.insert(g.name.clone(), g);
    lib.add_catalog(Arc::new(cat));
    Arc::new(lib)
}

pub fn resolve(s: OpticalSystem) -> ResolvedSystem {
    let snap = Snapshot::new(s).expect("valid system");
    ResolvedSystem::new(&snap, library()).expect("resolves")
}

pub fn singlet_resolved() -> ResolvedSystem {
    resolve(singlet())
}

/// The reference tracer of a system (test glass library).
pub fn tracer(s: OpticalSystem) -> SequentialTracer {
    SequentialTracer::new(Arc::new(resolve(s))).expect("tracer")
}

fn surf(id: u64, shape: SurfaceShape, t: f64, m: MaterialRef) -> Surface {
    Surface::new(id, shape, t, m)
}

fn angles(ys: &[f64]) -> FieldSet {
    FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: ys.iter().map(|&y| Field::new(0.0, y)).collect(),
    }
}

/// `system` with every automatic semi-diameter except the stop's frozen at
/// `scale` × its traced value, so off-axis pupil edges are vignetted.
pub fn vignetted(system: &OpticalSystem, scale: f64) -> OpticalSystem {
    let sd = tracer(system.clone()).semi_diameters(&SemiDiameterSettings::default());
    let mut v = system.clone();
    let n = v.surfaces.len();
    for (j, s) in v.surfaces.iter_mut().enumerate() {
        if j > 0
            && j + 1 < n
            && j != system.stop
            && sd.values[j] > 0.0
            && matches!(s.semi_diameter, SemiDiameter::Auto)
            && !s.is_coordinate_break()
        {
            s.semi_diameter = SemiDiameter::Fixed {
                value: scale * sd.values[j],
            };
        }
    }
    v
}

/// Cooke triplet with a decentred and tilted middle element (restored
/// after), a tilted rear element, and an image plane tilted by coordinate
/// breaks: every transform is non-axial somewhere.
pub fn tilted_triplet() -> OpticalSystem {
    let mut s = samples::cooke_triplet();
    s.name = "tilted triplet".into();
    s.surfaces[3].placement = Some(Placement {
        decenter_x: 0.05,
        decenter_y: -0.1,
        tilt_x: 0.8,
        tilt_y: -0.5,
        restore_after: true,
        ..Default::default()
    });
    s.surfaces[5].placement = Some(Placement {
        tilt_x: -0.3,
        tilt_z: 15.0,
        ..Default::default()
    });
    let img = s.surfaces.len() - 1;
    let bfl = s.surfaces[img - 1].thickness;
    s.surfaces[img - 1].thickness = bfl - 2.0;
    let mut cb = surf(100, SurfaceShape::CoordinateBreak, 2.0, MaterialRef::Air);
    cb.placement = Some(Placement {
        tilt_x: 2.0,
        tilt_y: 1.0,
        decenter_y: 0.2,
        ..Default::default()
    });
    s.surfaces.insert(img, cb);
    s
}

/// Two ideal lenses at finite conjugates: object 150 mm in front of an
/// f = 100 lens (the stop), a second f = 60 lens 40 mm behind it, image at
/// the paraxial image; object heights up to 4 mm.
pub fn ideal_lenses() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "ideal lenses".into();
    s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: [0.0, 2.5, 4.0]
            .iter()
            .map(|&y| Field::new(0.0, y))
            .collect(),
    };
    // Lens 1 images the object at 300 mm; lens 2 (40 mm on) sees a virtual
    // object 260 mm behind it: 1/b = 1/60 + 1/260.
    let b = 1.0 / (1.0 / 60.0 + 1.0 / 260.0);
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, 150.0, MaterialRef::Air),
        surf(
            1,
            SurfaceShape::Paraxial {
                focal_length: 100.0,
            },
            40.0,
            MaterialRef::Air,
        ),
        surf(
            2,
            SurfaceShape::Paraxial { focal_length: 60.0 },
            b,
            MaterialRef::Air,
        ),
        surf(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

/// A singlet followed by dummy planes carrying an annular (central
/// obstruction), a rectangular and an elliptical aperture, plus a fixed
/// semi-diameter: every aperture shape clips part of the beam.
pub fn apertures() -> OpticalSystem {
    let mut s = samples::singlet();
    s.name = "apertures".into();
    let img = s.surfaces.len() - 1;
    let bfl = s.surfaces[img - 1].thickness;
    s.surfaces[img - 1].thickness = 10.0;
    let mut a = surf(10, SurfaceShape::Plane, 10.0, MaterialRef::Air);
    a.interaction = Interaction::Dummy;
    a.aperture = Some(ApertureShape::Annular {
        inner_radius: 2.0,
        outer_radius: 9.5,
    });
    let mut r = surf(11, SurfaceShape::Plane, 10.0, MaterialRef::Air);
    r.interaction = Interaction::Dummy;
    r.aperture = Some(ApertureShape::Rectangular {
        half_x: 7.5,
        half_y: 8.0,
    });
    let mut e = surf(12, SurfaceShape::Plane, bfl - 30.0, MaterialRef::Air);
    e.interaction = Interaction::Dummy;
    e.aperture = Some(ApertureShape::Elliptical {
        semi_x: 6.0,
        semi_y: 7.0,
    });
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 9.3 };
    s.surfaces.splice(img..img, [a, r, e]);
    s
}

/// A plano-convex block whose rear sphere (R = −10 mm, n ≈ 1.52) totally
/// reflects rays above ≈ 6.6 mm: a large part of the beam fails with TIR.
pub fn tir_block() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "TIR block".into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 18.0 };
    s.fields = angles(&[0.0, 3.0]);
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        surf(1, SurfaceShape::Plane, 6.0, MaterialRef::glass("N-BK7")),
        surf(2, SurfaceShape::sphere(-10.0), 15.0, MaterialRef::Air),
        surf(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

/// A fast molded-asphere singlet (model glass 1.53/56): conic plus A4 … A12
/// on both faces, F/2, ±25°.
pub fn asphere_heavy() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "asphere singlet".into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 2.4 };
    s.fields = angles(&[0.0, 15.0, 25.0]);
    let glass = MaterialRef::Model {
        nd: 1.53,
        vd: 56.0,
        dpgf: 0.0,
    };
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        surf(
            1,
            SurfaceShape::EvenAsphere {
                curvature: 1.0 / 3.0,
                conic: -0.5,
                coefficients: vec![-2e-3, 1e-3, -3e-4, 4e-5],
            },
            1.4,
            glass,
        ),
        surf(
            2,
            SurfaceShape::EvenAsphere {
                curvature: -1.0 / 8.0,
                conic: 3.0,
                coefficients: vec![1e-2, -3e-3, 6e-4, -5e-5, 3e-6],
            },
            4.2,
            MaterialRef::Air,
        ),
        surf(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

/// The sample achromat scaled to f = 1000 mm (F/10, ±0.5°): a ≈ 1000 mm
/// track, the limit the CONVENTIONS §7 budget covers.
pub fn long_track() -> OpticalSystem {
    let mut s = samples::achromatic_doublet();
    s.name = "f1000 achromat".into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 100.0 };
    s.fields = angles(&[0.0, 0.35, 0.5]);
    for surf in &mut s.surfaces {
        if let SurfaceShape::Conic { curvature, .. } = &mut surf.shape {
            *curvature /= 10.0;
        }
        if surf.thickness.is_finite() {
            surf.thickness *= 10.0;
        }
    }
    s
}

/// The sample Cooke triplet with its crown elements in the absorbing test
/// glass: non-trivial bulk transmittance.
pub fn absorbing_triplet() -> OpticalSystem {
    let mut s = samples::cooke_triplet();
    s.name = "absorbing triplet".into();
    for surf in &mut s.surfaces {
        if surf.material == MaterialRef::glass("N-SK16") {
            surf.material = MaterialRef::Catalog {
                catalog: Some("TEST".into()),
                name: "ABSORB".into(),
            };
        }
    }
    s
}

/// The cross-validation set: every sample, every sample with vignetting,
/// and the fixtures above.
pub fn xval_systems() -> Vec<(String, OpticalSystem)> {
    let mut v = Vec::new();
    for (name, sys) in samples::all() {
        let vig = vignetted(&sys, 0.93);
        v.push((name.to_string(), sys));
        v.push((format!("{name} (vignetted)"), vig));
    }
    v.push(("tilted triplet".into(), tilted_triplet()));
    v.push(("ideal lenses".into(), ideal_lenses()));
    v.push(("apertures".into(), apertures()));
    v.push(("TIR block".into(), tir_block()));
    v.push(("asphere singlet".into(), asphere_heavy()));
    v.push(("f1000 achromat".into(), long_track()));
    v.push(("absorbing triplet".into(), absorbing_triplet()));
    v
}

/// Result of [`f64_path`].
#[derive(Clone, Copy, Debug)]
pub struct F64Path {
    /// Final point and direction in the last surface's frame.
    pub pos: mklight_core::Vec3,
    pub dir: mklight_core::Vec3,
    pub opl: f64,
    /// Unpolarised interface factors (bare Fresnel, mirrors 1) times bulk
    /// transmittance.
    pub weight: f64,
}

/// An independent f64 implementation of a path program, for tests: the
/// frames of `ResolvedSystem` (forward `to_next`, backward its inverse; for
/// an object at infinity surface 0 → 1 is surface 1's frame), the reference
/// intersection and interaction primitives of `mklight_trace` (ideal lenses
/// with the reference's object vergence going forward, zero going
/// backward), and core Fresnel coefficients. Apertures always clip. `None`
/// when the ray fails or a step transmits through a mirror.
pub fn f64_path(
    sys: &ResolvedSystem,
    start: u32,
    pos: mklight_core::Vec3,
    dir: mklight_core::Vec3,
    wl: usize,
    steps: &[crate::batch::PathStep],
) -> Option<F64Path> {
    use crate::batch::StepMode;
    use mklight_core::materials::coating::fresnel;
    use mklight_trace::interact::{ideal_lens, reflect, refract};
    use mklight_trace::surface::{IntersectSettings, intersect};
    let frame = |j: usize| -> mklight_core::math::Frame {
        match sys.surfaces[j].to_next {
            Some(f) => f,
            None => sys.surfaces[1].frame,
        }
    };
    let n_of = |gap: usize| sys.surfaces[gap].n_after[wl];
    let (mut p, mut d) = (pos, dir);
    let (mut opl, mut weight) = (0.0f64, 1.0f64);
    let mut cur = start as usize;
    for st in steps {
        let s = st.surface as usize;
        if st.backward {
            let f = frame(s);
            p = f.to_global_point(p);
            d = f.to_global_dir(d);
        } else {
            let f = frame(cur);
            p = f.to_local_point(p);
            d = f.to_local_dir(d);
        }
        let gap = if st.backward { s } else { s - 1 };
        let n1 = n_of(gap);
        let n2 = if st.backward {
            n_of(s.saturating_sub(1))
        } else {
            n_of(s)
        };
        let surf = &sys.surfaces[s];
        let hit = intersect(&surf.shape, p, d, &IntersectSettings::default()).ok()?;
        let t = hit.t_transfer + hit.t_surface;
        opl += n1 * t;
        let alpha = if gap == 0 && sys.object_at_infinity {
            0.0
        } else {
            surf_alpha(sys, gap, wl)
        };
        weight *= (-alpha * t).exp();
        p = hit.point;
        if !surf.passes_aperture(p.x, p.y) && !matches!(surf.shape, SurfaceShape::CoordinateBreak) {
            return None;
        }
        let image = s == sys.image_index();
        let cb = matches!(surf.shape, SurfaceShape::CoordinateBreak);
        let mirror = surf.is_mirror();
        let reflect_here = match st.mode {
            StepMode::Reflect => true,
            StepMode::Sequential => mirror,
            StepMode::Transmit => {
                if mirror {
                    return None; // no medium behind a mirror
                }
                false
            }
        };
        let passive = image || surf.interaction == Interaction::Dummy;
        if cb || (!reflect_here && passive) {
        } else if let SurfaceShape::Paraxial { focal_length } = surf.shape
            && (mirror || !reflect_here)
        {
            // Ideal lens: transmitted, or reflected by a mirror-type lens.
            let n_out = if reflect_here {
                d = reflect(d, hit.normal).0;
                n1
            } else {
                n2
            };
            let verg = if st.backward {
                0.0
            } else {
                mklight_trace::paraxial::ideal_lens_object_vergence(sys, s, wl)
            };
            let (k, delta) = ideal_lens(p, d, focal_length, n1, n_out, verg)?;
            d = k;
            opl += delta;
        } else if reflect_here {
            let (k, c) = reflect(d, hit.normal);
            d = k;
            weight *= if mirror {
                1.0
            } else {
                fresnel(n1, n2, c).r_unpolarized()
            };
        } else {
            let (k, c) = refract(d, hit.normal, n1, n2)?;
            d = k;
            weight *= fresnel(n1, n2, c).t_unpolarized();
        }
        cur = s;
    }
    Some(F64Path {
        pos: p,
        dir: d,
        opl,
        weight,
    })
}

fn surf_alpha(sys: &ResolvedSystem, gap: usize, wl: usize) -> f64 {
    sys.surfaces[gap].alpha_after[wl]
}

/// A tilted, decentred N-BK7 plate (10 mm) in air with an image plane
/// 20 mm behind it, for ghost-path tests.
pub fn plate(tilted: bool) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.name = "plate".into();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        surf(1, SurfaceShape::Plane, 10.0, MaterialRef::glass("N-BK7")),
        surf(2, SurfaceShape::sphere(-200.0), 20.0, MaterialRef::Air),
        surf(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    if tilted {
        s.surfaces[1].placement = Some(Placement {
            decenter_y: 0.4,
            tilt_x: 7.0,
            tilt_y: -3.0,
            ..Default::default()
        });
    }
    s.stop = 1;
    s
}
