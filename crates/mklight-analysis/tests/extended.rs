//! Extended geometry pinned against surface definitions and independent rays.
use crate::common;

use common::*;
use mklight_analysis::{curvature::*, fans::*, layout::*, seidel::*, spot::*, system_data::*};
use mklight_core::{ApertureSpec, Field, SurfaceShape, samples, surface_math};
use mklight_trace::{FieldPoint, TraceOptions};

#[test]
fn hessian_coddington_matches_close_rays_for_every_extended_sample() {
    for (name, sys) in samples::all()
        .into_iter()
        .filter(|(_, s)| samples::has_extended_surfaces(s))
    {
        let t = tracer(sys);
        for (hx, hy) in [(0.0, 0.0), (0.7, 0.0), (-0.4, 0.6), (0.0, -1.0)] {
            let f = t.prepare_field(FieldPoint::normalized(hx, hy)).unwrap();
            for w in 0..t.system().wavelengths_um.len() {
                let a = coddington_focus(&t, &f, w).unwrap();
                let b = close_ray_focus(&t, &f, w, 1e-5).unwrap();
                for (a, b) in [(a.tangential, b.tangential), (a.sagittal, b.sagittal)] {
                    assert!(
                        (a - b).abs() < 2e-6 * (1.0 + a.abs()),
                        "{name} ({hx},{hy}) wl {w}: {a} != {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn curved_doe_phase_tensor_matches_close_rays() {
    let mut sys = samples::hybrid_diffractive_achromat();
    let SurfaceShape::Diffractive { base, phase } = &mut sys.surfaces[2].shape else {
        unreachable!()
    };
    **base = SurfaceShape::sphere(-120.0);
    phase.coefficients.push(0.03);
    let t = tracer(sys);
    for xy in [[0.4, 0.6], [-0.7, 0.3]] {
        let f = t
            .prepare_field(FieldPoint::normalized(xy[0], xy[1]))
            .unwrap();
        let a = coddington_focus(&t, &f, 1).unwrap();
        let b = close_ray_focus(&t, &f, 1, 1e-4).unwrap();
        assert!((a.tangential - b.tangential).abs() < 1e-6);
        assert!((a.sagittal - b.sagittal).abs() < 1e-6);
    }
}

#[test]
fn layout_sections_rims_and_mesh_follow_cartesian_sag() {
    for sys in [
        samples::xy_freeform(),
        samples::toroidal_element(),
        samples::biconic_element(),
    ] {
        let t = tracer(sys);
        for section in [LayoutSection::Yz, LayoutSection::Xz] {
            let s = layout(
                &t,
                &LayoutSettings {
                    section,
                    mesh_rings: 3,
                    ..Default::default()
                },
            )
            .unwrap();
            let shape = &t.system().surfaces[1].shape;
            let frame = t.system().surfaces[1].frame;
            let radius = s
                .surfaces
                .iter()
                .find(|s| s.index == 1)
                .unwrap()
                .semi_diameter;
            for poly in s.polylines.iter().filter(|p| {
                p.surface == Some(1)
                    && matches!(
                        p.kind,
                        PolylineKind::SurfaceProfile | PolylineKind::SurfaceRim
                    )
            }) {
                for p in &poly.points {
                    let p = frame.to_local_point(mklight_core::Vec3::new(p[0], p[1], p[2]));
                    let z = surface_math::evaluate(shape, p.x, p.y).unwrap().sag;
                    assert!((p.z - z).abs() < 1e-12);
                }
            }
            let element = &s.elements[0];
            for p in &element.front {
                let p = frame.to_local_point(mklight_core::Vec3::new(p[0], p[1], p[2]));
                let (x, y) = (p.x.clamp(-radius, radius), p.y.clamp(-radius, radius));
                assert!((p.z - surface_math::evaluate(shape, x, y).unwrap().sag).abs() < 1e-12);
                assert!(if section == LayoutSection::Yz {
                    p.x == 0.0
                } else {
                    p.y == 0.0
                });
            }
            let mesh = s.meshes.iter().find(|m| m.surface == 1).unwrap();
            assert!(!mesh.triangles.is_empty());
            for tri in &mesh.triangles {
                for p in tri {
                    let p = frame.to_local_point(mklight_core::Vec3::new(p[0], p[1], p[2]));
                    assert!(
                        (p.z - surface_math::evaluate(shape, p.x, p.y).unwrap().sag).abs() < 1e-12
                    );
                }
            }
        }
    }
    let t = tracer(samples::hybrid_diffractive_achromat());
    let scene = layout(&t, &Default::default()).unwrap();
    assert!(
        scene
            .surfaces
            .iter()
            .find(|s| s.index == 2)
            .unwrap()
            .is_diffractive
    );
    let buffers = scene.to_buffers();
    assert!(
        buffers
            .surfaces
            .iter()
            .zip(&buffers.flags)
            .any(|(&s, &f)| s == 2 && f & 4 != 0)
    );
}

#[test]
fn q_fourth_order_matches_independent_even_asphere_equivalents() {
    // Independent low-order Q polynomials, including their t coefficient.
    for bfs in [false, true] {
        let (c, r, a) = (0.02_f64, 5.0_f64, [0.001, -0.0003, 0.0002]);
        let (cv, fourth, shape) = if bfs {
            let s0 = a[0] + 13.0 / 19.0_f64.sqrt() * a[1] + 58.0 / 190.0_f64.sqrt() * a[2];
            let s1 = -16.0 / 19.0_f64.sqrt() * a[1] - 200.0 / 190.0_f64.sqrt() * a[2];
            (
                c + 2.0 * s0 / r.powi(2),
                c.powi(3) / 8.0 + (s1 - s0) / r.powi(4) + 0.5 * c * c * s0 / r.powi(2),
                SurfaceShape::QBfs {
                    curvature: c,
                    norm_radius: r,
                    coefficients: a.to_vec(),
                },
            )
        } else {
            (
                c,
                c.powi(3) / 8.0 + (a[0] - 5.0 * a[1] + 15.0 * a[2]) / r.powi(4),
                SurfaceShape::QCon {
                    curvature: c,
                    conic: 0.0,
                    norm_radius: r,
                    coefficients: a.to_vec(),
                },
            )
        };
        let mut sys = samples::singlet();
        sys.surfaces[1].shape = shape;
        let t = tracer(sys.clone());
        sys.surfaces[1].shape = SurfaceShape::EvenAsphere {
            curvature: cv,
            conic: 0.0,
            coefficients: vec![fourth - cv.powi(3) / 8.0],
        };
        let u = tracer(sys);
        let (a, b) = (
            seidel(&t, &Default::default()).unwrap(),
            seidel(&u, &Default::default()).unwrap(),
        );
        for (key, a) in serde_json::to_value(a.sums).unwrap().as_object().unwrap() {
            let b = serde_json::to_value(b.sums).unwrap();
            assert!(
                (a.as_f64().unwrap() - b[key].as_f64().unwrap()).abs() < 1e-12,
                "{bfs} {key}"
            );
        }
    }
}

#[test]
fn hybrid_doe_seidel_predicts_real_aberrations_and_colour() {
    let mut sys = samples::hybrid_diffractive_achromat();
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 0.8 };
    sys.fields = angle_fields(&[0.0, 0.2]);
    let t = tracer(sys.clone());
    let img = sys.surfaces.len() - 2;
    sys.surfaces[img].thickness = t.paraxial().primary().image_distance;
    let t = tracer(sys);
    let w = t.system().primary_wavelength;
    let s = seidel(&t, &Default::default()).unwrap();
    let opts = TraceOptions::unclipped();
    for h in [0.0, 0.7, 1.0] {
        let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
        let chief = t.chief(&f, w);
        for (px, py) in [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.6, -0.5)] {
            let r = t.trace(&f, w, px, py, &opts);
            assert!(r.status.is_ok());
            let real = [r.ray.pos.x - chief.ray.pos.x, r.ray.pos.y - chief.ray.pos.y];
            let pred = s.transverse_aberration(h, px, py, false);
            for i in 0..2 {
                assert!(
                    (real[i] - pred[i]).abs() < 0.003 * pred[i].abs() + 1e-10,
                    "h={h} p={px},{py}: {real:?} vs {pred:?}"
                );
            }
        }
    }
    // The achromat's F/C paraxial back-focus correction is independent of
    // its finite spherochromatism. Lateral colour tests the phase dispersion.
    let f = |w| {
        let a = &t.paraxial().at(w).marginal;
        -a.y[a.y.len() - 1] / a.u[a.u.len() - 2]
    };
    assert!((f(s.short_wavelength) - f(s.long_wavelength)).abs() < 1e-9);
    assert!(
        (s.sums.cl / s.image_marginal_slope.powi(2)).abs() < 1e-4,
        "C_L focus {}",
        s.sums.cl / s.image_marginal_slope.powi(2)
    );
    let field = t.prepare_field(FieldPoint::normalized(0.0, 1.0)).unwrap();
    let colour = t.chief(&field, s.short_wavelength).ray.pos.y
        - t.chief(&field, s.long_wavelength).ray.pos.y;
    let pred = s.sums.ct / s.image_marginal_slope;
    assert!(
        (colour - pred).abs() < 0.01 * pred.abs() + 1e-9,
        "{colour} vs {pred}"
    );
}

#[test]
fn non_symmetric_spots_fans_grid_and_first_order_cover_both_axes() {
    for sys in [samples::xy_freeform(), samples::anamorphic_adapter()] {
        let t = tracer(sys);
        let spot = spot_diagram(
            &t,
            &SpotDiagramSettings {
                rings: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(spot.fields.iter().any(|f| f.field_value[0] != 0.0));
        let fans = ray_fan(
            &t,
            &FanSettings {
                samples: 7,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fans.fields.iter().any(|f| f.field_value[0] != 0.0));
        // Explicit x/y fields cross-check against unmodified direct traces.
        let mut sys = t.system().system().clone();
        sys.fields.fields = vec![Field::new(0.5, 0.0), Field::new(0.0, 0.5)];
        let u = tracer(sys);
        let fans = ray_fan(
            &u,
            &FanSettings {
                samples: 7,
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..2 {
            let row = fans.field(i).unwrap();
            let curve = &row.sagittal[0];
            let w = curve.wavelength;
            let c = u.chief(u.field(i), u.system().primary_wavelength);
            for (&p, &value) in curve.pupil.iter().zip(&curve.values) {
                let r = u.trace(u.field(i), w, p, 0.0, &u.options());
                if r.status.is_ok() {
                    assert!((r.ray.pos.x - c.ray.pos.x - value).abs() < 1e-12);
                }
            }
        }
        let grid = grid_distortion(
            &t,
            &GridDistortionSettings {
                grid_size: 3,
                half_width: 0.2,
                ..Default::default()
            },
        )
        .unwrap();
        let xy = mklight_trace::ParaxialXY::new(t.system()).unwrap();
        let w = grid.wavelength;
        for p in &grid.points {
            let f = t
                .prepare_field(FieldPoint::normalized(p.field[0], p.field[1]))
                .unwrap();
            let mklight_trace::ObjectSource::Direction { dir } = f.source else {
                unreachable!()
            };
            assert!(
                (p.paraxial[0] - xy.x[w].image_height_per_tan.unwrap() * dir.x / dir.z).abs()
                    < 1e-12
            );
            assert!(
                (p.paraxial[1] - xy.y[w].image_height_per_tan.unwrap() * dir.y / dir.z).abs()
                    < 1e-12
            );
        }
        let s = seidel(&t, &Default::default()).unwrap();
        assert_eq!(s.status, SeidelStatus::NotApplicable);
        assert!(s.sums.s1.is_nan() && s.surfaces.is_empty());
        let report = system_data(&t, &Default::default()).unwrap();
        assert_eq!(report.paraxial_xy.symmetry, xy.symmetry);
        assert!(serde_json::to_value(&s).unwrap()["sums"]["s1"].is_null());
        let fc = field_curvature(
            &t,
            &FieldCurvatureSettings {
                field_samples: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fc.field_coordinates.iter().any(|f| f[0] != 0.0));
    }
}

#[test]
fn odd_nonanalytic_terms_warn_and_keep_the_regular_fourth_order_part() {
    let mut sys = samples::singlet();
    let c = 0.02_f64;
    sys.surfaces[1].shape = SurfaceShape::OddAsphere {
        curvature: c,
        conic: -0.5,
        coefficients: vec![0.001, 0.002, 0.0001, 1e-6],
    };
    let t = tracer(sys.clone());
    let s = seidel(&t, &Default::default()).unwrap();
    assert_eq!(s.status, SeidelStatus::Applicable);
    assert!(
        s.run_info
            .warnings
            .iter()
            .any(|w| w.contains("non-analytic"))
    );
    let cv = c + 0.004;
    let a4 = 0.5 * c.powi(3) / 8.0 + 1e-6;
    sys.surfaces[1].shape = SurfaceShape::EvenAsphere {
        curvature: cv,
        conic: 0.0,
        coefficients: vec![a4 - cv.powi(3) / 8.0],
    };
    let b = seidel(&tracer(sys), &Default::default()).unwrap();
    assert!((s.sums.s1 - b.sums.s1).abs() < 1e-12);
    assert!((s.sums.s2 - b.sums.s2).abs() < 1e-12);
    let f = t.prepare_field(FieldPoint::normalized(0.0, 0.0)).unwrap();
    assert!(coddington_focus(&t, &f, t.system().primary_wavelength).is_none());
}

#[test]
fn asymmetric_longitudinal_and_colour_cover_both_signed_axes() {
    let t = tracer(samples::xy_freeform());
    let r = longitudinal_aberration(&t, &Default::default()).unwrap();
    assert!(r.pupil[0] < 0.0);
    assert!(
        r.curves
            .iter()
            .any(|c| c.axis == mklight_trace::ParaxialAxis::X)
    );
    let r = chromatic_focal_shift(&t, &Default::default()).unwrap();
    assert_eq!(r.paraxial_focus_x.len(), r.paraxial_focus.len());
}

#[test]
fn layout_stays_inside_the_x_domain_of_steep_toroids_and_biconics() {
    // Regression: the clear radius used only the y conic, so a fixed
    // semi-diameter beyond 1/c_x put NaN sags into profiles and outlines.
    for shape in [
        SurfaceShape::Biconic {
            curvature_x: 0.25,
            curvature_y: 1.0 / 60.0,
            conic_x: 0.0,
            conic_y: 0.0,
            x_asphere: vec![],
            y_asphere: vec![],
        },
        SurfaceShape::Toroid {
            curvature_x: 0.25,
            curvature_y: 1.0 / 60.0,
            conic_y: 0.0,
            y_asphere: vec![],
        },
    ] {
        let mut sys = samples::biconic_element();
        sys.surfaces[1].shape = shape;
        sys.surfaces[1].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 4.5 };
        let t = tracer(sys);
        for section in [LayoutSection::Yz, LayoutSection::Xz] {
            let s = layout(
                &t,
                &LayoutSettings {
                    section,
                    mesh_rings: 2,
                    ..Default::default()
                },
            )
            .unwrap();
            let finite = |p: &[[f64; 3]]| p.iter().flatten().all(|v| v.is_finite());
            assert!(s.polylines.iter().all(|p| finite(&p.points)));
            assert!(s.elements.iter().all(|e| finite(&e.outline)));
            let sd = s.surfaces.iter().find(|x| x.index == 1).unwrap();
            assert!(sd.semi_diameter <= 4.0);
        }
    }
}
