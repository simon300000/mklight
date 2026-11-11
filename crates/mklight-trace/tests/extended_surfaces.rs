use crate::common;
use common::*;
use mklight_core::model::surface::{DiffractivePhase, ZernikeOrdering};
use mklight_core::surface_math::{Symmetry, evaluate, system_symmetry};
use mklight_core::{JobContext, MaterialRef, MkError, SurfaceShape, Vec3, samples};
use mklight_trace::surface::{IntersectError, IntersectSettings, intersect};
use mklight_trace::{
    FailReason, Paraxial, ParaxialXY, Ray, TraceOptions, TraceStatus, trace_ray_recorded,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} != {b} (tol {tol})");
}
fn doe(a: Vec<f64>, r: f64, order: i32) -> SurfaceShape {
    SurfaceShape::Diffractive {
        base: Box::new(SurfaceShape::Plane),
        phase: DiffractivePhase {
            norm_radius: r,
            coefficients: a,
            design_order: 1,
            design_wavelength_um: D_LINE,
            trace_order: order,
        },
    }
}
fn focal_ray(sys: &mklight_core::ResolvedSystem, wl: usize, x: f64, y: f64) -> (f64, f64) {
    let last = sys.num_surfaces() - 2;
    let (r, rec) = trace_ray_recorded(
        sys,
        &Ray::new(Vec3::new(x, y, 0.0), Vec3::Z),
        wl,
        &TraceOptions::default().to_surface(last),
    );
    assert!(r.status.is_ok(), "{:?}", r.status);
    let v = &rec[last];
    let (h, slope) = if x != 0.0 {
        (v.point.x, v.dir.x / v.dir.z)
    } else {
        (v.point.y, v.dir.y / v.dir.z)
    };
    // BFL measured from last vertex; EFL from unit-index reduced slope.
    (
        v.point.z - h / slope,
        -if x != 0.0 { x } else { y } / (sys.n_before(sys.image_index(), wl) * slope),
    )
}

#[test]
fn coincident_shapes_have_identical_reference_traces() {
    let c = 0.02;
    let k = -0.4;
    let shapes = vec![
        SurfaceShape::OddAsphere {
            curvature: c,
            conic: k,
            coefficients: vec![],
        },
        SurfaceShape::QCon {
            curvature: c,
            conic: k,
            norm_radius: 10.0,
            coefficients: vec![0.0; 20],
        },
        SurfaceShape::Biconic {
            curvature_x: c,
            curvature_y: c,
            conic_x: k,
            conic_y: k,
            x_asphere: vec![],
            y_asphere: vec![],
        },
        SurfaceShape::XYPolynomial {
            curvature: c,
            conic: k,
            norm_radius: 10.0,
            coefficients: vec![0.0; 65],
        },
        SurfaceShape::ZernikeSag {
            curvature: c,
            conic: k,
            norm_radius: 10.0,
            ordering: ZernikeOrdering::Fringe,
            coefficients: vec![0.0; 37],
        },
        SurfaceShape::Diffractive {
            base: Box::new(SurfaceShape::Conic {
                curvature: c,
                conic: k,
            }),
            phase: DiffractivePhase {
                norm_radius: 10.0,
                coefficients: vec![0.0],
                design_order: 1,
                design_wavelength_um: D_LINE,
                trace_order: 0,
            },
        },
    ];
    let reference = SurfaceShape::Conic {
        curvature: c,
        conic: k,
    };
    for shape in shapes {
        let make = |sh| {
            resolve(infinite_system(
                4.0,
                vec![surf(1, sh, 20.0, MaterialRef::Constant { n: 1.5 })],
                2,
            ))
        };
        let a = make(reference.clone());
        let b = make(shape);
        for (x, y) in [(0.0, 0.0), (1.0, -2.0), (3.0, 0.3)] {
            let ray = Ray::new(
                Vec3::new(x, y, -4.0),
                Vec3::new(0.01, -0.02, 1.0).normalized(),
            );
            let aa = trace_ray_recorded(&a, &ray, 0, &Default::default());
            let bb = trace_ray_recorded(&b, &ray, 0, &Default::default());
            assert_eq!(aa, bb);
        }
    }
    for sh in [
        SurfaceShape::QBfs {
            curvature: c,
            norm_radius: 10.0,
            coefficients: vec![],
        },
        SurfaceShape::Toroid {
            curvature_x: c,
            curvature_y: c,
            conic_y: 0.0,
            y_asphere: vec![],
        },
    ] {
        let p = Vec3::new(2.0, 1.0, -4.0);
        assert_eq!(
            intersect(&sh, p, Vec3::Z, &Default::default()),
            intersect(
                &SurfaceShape::sphere(1.0 / c),
                p,
                Vec3::Z,
                &Default::default()
            )
        );
    }
}

#[test]
fn polynomial_reductions_match_even_asphere() {
    let even = SurfaceShape::EvenAsphere {
        curvature: 0.02,
        conic: -0.5,
        coefficients: vec![1e-5],
    };
    let equivalent = [
        SurfaceShape::OddAsphere {
            curvature: 0.02,
            conic: -0.5,
            coefficients: vec![0.0, 0.0, 0.0, 1e-5],
        },
        SurfaceShape::QCon {
            curvature: 0.02,
            conic: -0.5,
            norm_radius: 5.0,
            coefficients: vec![1e-5 * 625.0],
        },
        SurfaceShape::XYPolynomial {
            curvature: 0.02,
            conic: -0.5,
            norm_radius: 5.0,
            coefficients: {
                let mut c = vec![0.0; 14];
                c[9] = 0.00625;
                c[11] = 0.0125;
                c[13] = 0.00625;
                c
            },
        },
    ];
    for sh in equivalent {
        for i in 1..10 {
            let p = Vec3::new(0.2 * i as f64, -0.1 * i as f64, -3.0);
            let d = Vec3::new(0.03, 0.05, 1.0).normalized();
            let a = intersect(&even, p, d, &Default::default()).unwrap();
            let b = intersect(&sh, p, d, &Default::default()).unwrap();
            assert!(a.point.max_abs_diff(b.point) < 1e-13);
            assert!(a.normal.max_abs_diff(b.normal) < 1e-13);
        }
    }
}

#[test]
fn cylinders_form_exact_line_focus() {
    let c = 0.06;
    let k = -4.0 / 9.0; // exact collimated air -> n=1.5 focus at 50 mm
    let shapes = [
        SurfaceShape::Toroid {
            curvature_x: 0.0,
            curvature_y: c,
            conic_y: k,
            y_asphere: vec![],
        },
        SurfaceShape::Biconic {
            curvature_x: 0.0,
            curvature_y: c,
            conic_x: 0.0,
            conic_y: k,
            x_asphere: vec![],
            y_asphere: vec![],
        },
    ];
    for shape in shapes {
        let rs = resolve(infinite_system(
            10.0,
            vec![surf(1, shape, 50.0, MaterialRef::Constant { n: 1.5 })],
            2,
        ));
        for x in [-2.0, 0.0, 2.0] {
            for y in [-4.0, -1.0, 0.0, 1.0, 4.0] {
                let (r, _) = trace_ray_recorded(
                    &rs,
                    &Ray::new(Vec3::new(x, y, 0.0), Vec3::Z),
                    0,
                    &Default::default(),
                );
                assert!(r.status.is_ok());
                close(r.ray.pos.x, x, 1e-13);
                close(r.ray.pos.y, 0.0, 1e-13);
            }
        }
        let xy = ParaxialXY::new(&rs).unwrap();
        assert!(xy.x[0].efl.is_infinite());
        close(xy.y[0].bfl, 50.0, 1e-12);
    }
}

#[test]
fn anamorphic_focal_lengths_equal_narrow_fans() {
    for sys in [
        samples::anamorphic_adapter(),
        samples::biconic_element(),
        samples::toroidal_element(),
        samples::xy_freeform(),
        samples::zernike_sag(),
    ] {
        let rs = resolve(sys);
        let xy = ParaxialXY::new(&rs).unwrap();
        for wl in 0..rs.wavelengths_um.len() {
            for (x, y, fo) in [(1e-4, 0.0, &xy.x[wl]), (0.0, 1e-4, &xy.y[wl])] {
                let (bfl, efl) = focal_ray(&rs, wl, x, y);
                close(bfl, fo.bfl, 1e-5);
                close(efl, fo.efl, 1e-5);
            }
        }
        assert_eq!(xy.run_info.snapshot_hash, rs.snapshot.hash().to_hex());
    }
    let rs = resolve(samples::anamorphic_adapter());
    let xy = ParaxialXY::new(&rs).unwrap();
    close(xy.squeeze[rs.primary_wavelength], 2.0, 1e-13);
    assert_eq!(xy.symmetry, Symmetry::Meridional);
}

#[test]
fn doe_obeys_grating_equation_and_surface_metric() {
    let lambda = 0.00055;
    let period = 0.005;
    let order = 2;
    let sin_i = 0.1;
    let k = Vec3::new(sin_i, 0.0, (1.0_f64 - sin_i * sin_i).sqrt());
    // g = mλ∇Φ/(2π) with ∇Φ = 2π/Λ.
    let g = [order as f64 * lambda / period, 0.0];
    let d = mklight_trace::interact::diffract(k, Vec3::Z, 1.0, 1.0, g, false).unwrap();
    close(d.x, sin_i + order as f64 * lambda / period, 1e-15);
    // Radial phase has exactly the same local grating gradient at x=1 mm.
    let mut sys = infinite_system(
        4.0,
        vec![surf(
            1,
            doe(vec![std::f64::consts::PI / period], 1.0, order),
            1.0,
            MaterialRef::Air,
        )],
        2,
    );
    sys.wavelengths.wavelengths[0].um = lambda * 1e3;
    let rs = resolve(sys);
    let (r, _) = trace_ray_recorded(
        &rs,
        &Ray::new(Vec3::new(1.0, 0.0, 0.0), k),
        0,
        &Default::default(),
    );
    assert!(r.status.is_ok());
    close(r.ray.dir.x, d.x, 1e-15);
    let n = Vec3::new(-0.3, 0.2, 1.0).normalized();
    let ex = Vec3::new(1.0, 0.0, 0.3);
    let ey = Vec3::new(0.0, 1.0, -0.2);
    let g = [
        -lambda * 12.0 / std::f64::consts::TAU,
        lambda * 8.0 / std::f64::consts::TAU,
    ];
    let out = mklight_trace::interact::diffract(k, n, 1.2, 1.6, g, false).unwrap();
    let delta = out * 1.6 - k * 1.2;
    close(delta.dot(ex), -lambda * 12.0 / std::f64::consts::TAU, 1e-15);
    close(delta.dot(ey), lambda * 8.0 / std::f64::consts::TAU, 1e-15);
}

#[test]
fn doe_power_dispersion_and_hybrid_colour_correction() {
    let f = 80.0;
    let r = 5.0;
    let a = -std::f64::consts::PI * r * r / (D_LINE * 1e-3 * f);
    let mut sys = infinite_system(
        4.0,
        vec![surf(1, doe(vec![a], r, 1), f, MaterialRef::Air)],
        2,
    );
    sys.wavelengths = Default::default();
    let rs = resolve(sys);
    let parax = Paraxial::new(&rs).unwrap();
    for (wl, l) in rs.wavelengths_um.iter().enumerate() {
        let expected = f * D_LINE / l;
        close(parax.at(wl).efl, expected, 2e-13);
        close(focal_ray(&rs, wl, 0.0, 1e-4).0, expected, 1e-9);
    }
    close(D_LINE / (0.4861327 - 0.6562725), -3.453, 0.001);
    let rs = resolve(samples::hybrid_diffractive_achromat());
    let p = Paraxial::new(&rs).unwrap();
    let f = rs
        .wavelengths_um
        .iter()
        .position(|l| (*l - 0.4861327).abs() < 1e-7)
        .unwrap();
    let c = rs
        .wavelengths_um
        .iter()
        .position(|l| (*l - 0.6562725).abs() < 1e-7)
        .unwrap();
    close(p.at(f).bfl, p.at(c).bfl, 1e-11);
    close(
        focal_ray(&rs, f, 0.0, 1e-3).0,
        focal_ray(&rs, c, 0.0, 1e-3).0,
        1e-7,
    );
    let mut plain = rs.system().clone();
    plain.surfaces[2].shape = SurfaceShape::Plane;
    let p = Paraxial::new(&resolve(plain)).unwrap();
    assert!((p.at(f).bfl - p.at(c).bfl).abs() > 1.0);
}

#[test]
fn perfect_doe_phase_has_flat_wavefront_at_design() {
    let f: f64 = 50.0;
    let radius: f64 = 5.0;
    let lambda = D_LINE * 1e-3;
    let order = 2;
    let mut binomial = 1.0;
    let coefficients = (1..=10)
        .map(|i| {
            binomial *= (1.5 - i as f64) / i as f64;
            -std::f64::consts::TAU * f / lambda / order as f64 * binomial * (radius / f).powi(2 * i)
        })
        .collect();
    let rs = resolve(infinite_system(
        10.0,
        vec![surf(
            1,
            doe(coefficients, radius, order),
            f,
            MaterialRef::Air,
        )],
        2,
    ));
    let chief = trace_ray_recorded(&rs, &Ray::new(Vec3::ZERO, Vec3::Z), 0, &Default::default()).0;
    for i in 1..=20 {
        let h = radius * i as f64 / 20.0;
        let (ray, rec) = trace_ray_recorded(
            &rs,
            &Ray::new(Vec3::new(h, 0.0, 0.0), Vec3::Z),
            0,
            &Default::default(),
        );
        assert!(ray.status.is_ok());
        close(ray.ray.pos.x, 0.0, 2e-14);
        close(chief.opl.diff(&ray.opl) / lambda, 0.0, 1e-10);
        close(rec[1].opl, -h * h / ((f * f + h * h).sqrt() + f), 2e-15);
    }
}

#[test]
fn evanescent_orders_and_zero_order_are_distinct() {
    let rs = resolve(infinite_system(
        4.0,
        vec![surf(1, doe(vec![1e6], 2.0, 1), 10.0, MaterialRef::Air)],
        2,
    ));
    let ray = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::Z);
    let (r, _) = trace_ray_recorded(&rs, &ray, 0, &Default::default());
    assert_eq!(
        r.status,
        TraceStatus::Failed {
            surface: 1,
            reason: FailReason::EvanescentOrder
        }
    );
    let mut sys = rs.system().clone();
    if let SurfaceShape::Diffractive { phase, .. } = &mut sys.surfaces[1].shape {
        phase.trace_order = 0;
    }
    let (r, _) = trace_ray_recorded(&resolve(sys), &ray, 0, &Default::default());
    assert!(r.status.is_ok());
    assert_eq!(r.ray.dir, Vec3::Z);
    close(r.opl.value(), 10.0, 1e-15);
}

#[test]
fn steep_q_and_freeform_intersections_converge() {
    for shape in [
        SurfaceShape::QCon {
            curvature: 0.1,
            conic: -0.5,
            norm_radius: 5.0,
            coefficients: vec![3.0, 0.3, -0.02],
        },
        SurfaceShape::QBfs {
            curvature: 0.1,
            norm_radius: 5.0,
            coefficients: vec![1.0, -0.1, 0.02],
        },
        SurfaceShape::XYPolynomial {
            curvature: 0.1,
            conic: -0.5,
            norm_radius: 5.0,
            coefficients: vec![0.0, 0.0, 1.0, 0.2, -0.5, 0.3, 0.1, 0.0, -0.1],
        },
    ] {
        for i in 1..=12 {
            let x = 0.3 * i as f64;
            let y = 0.1 * i as f64;
            let z = evaluate(&shape, x, y).unwrap().sag;
            let target = Vec3::new(x, y, z);
            let d = Vec3::new(0.03, 0.04, 1.0).normalized();
            let hit =
                intersect(&shape, target - d * 8.0, d, &IntersectSettings::default()).unwrap();
            assert!(hit.point.max_abs_diff(target) < 2e-12, "{shape:?} {hit:?}");
        }
    }
}

#[test]
fn extended_samples_have_positive_thickness_around_entire_rim() {
    for (name, sys) in samples::all()
        .into_iter()
        .filter(|(_, s)| samples::has_extended_surfaces(s))
    {
        sys.validate().unwrap();
        let t = tracer(sys.clone());
        let sd = t.semi_diameters(&Default::default());
        for i in 1..sys.surfaces.len() - 2 {
            if matches!(sys.surfaces[i].material, MaterialRef::Air) {
                continue;
            }
            let radius = sd.values[i].max(sd.values[i + 1]) + 1.0;
            for a in 0..72 {
                for r in 0..=10 {
                    let theta = a as f64 * std::f64::consts::TAU / 72.0;
                    let h = radius * r as f64 / 10.0;
                    let h1 = h.min(sd.values[i]);
                    let h2 = h.min(sd.values[i + 1]);
                    let z1 = evaluate(&sys.surfaces[i].shape, h1 * theta.cos(), h1 * theta.sin())
                        .unwrap()
                        .sag;
                    let z2 = evaluate(
                        &sys.surfaces[i + 1].shape,
                        h2 * theta.cos(),
                        h2 * theta.sin(),
                    )
                    .unwrap()
                    .sag;
                    assert!(
                        sys.surfaces[i].thickness + z2 - z1 > 0.0,
                        "{name} surface {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn first_order_job_cancellation_and_symmetry() {
    let sys = samples::q_asphere_lens();
    assert_eq!(system_symmetry(&sys), Symmetry::Rotational);
    let rs = resolve(sys);
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        ParaxialXY::new_job(&rs, &ctx),
        Err(MkError::Cancelled)
    ));
    let ctx = JobContext::new();
    ParaxialXY::new_job(&rs, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    assert_eq!(system_symmetry(&samples::xy_freeform()), Symmetry::None);
}

#[test]
fn irregularity_on_a_cone_term_odd_asphere_keeps_the_first_order_model() {
    use mklight_core::param::{IrregularityMode, ParamPath};
    let mut sys = samples::odd_asphere_element();
    let SurfaceShape::OddAsphere { coefficients, .. } = &mut sys.surfaces[1].shape else {
        panic!("odd asphere sample");
    };
    coefficients[0] = 1e-4;
    sys.surfaces[1].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 5.0 };
    let efl = |sys: &mklight_core::OpticalSystem| {
        let rs = resolve(sys.clone());
        mklight_trace::paraxial::first_order(&rs, rs.primary_wavelength)
            .unwrap()
            .efl
    };
    let bare = efl(&sys);
    assert!(bare.is_finite());

    // One fringe of spherical irregularity has the vertex curvature
    // −4λ/R², the same as −2λ/R² on the r² coefficient.
    let mut equivalent = sys.clone();
    let wave = equivalent.wavelengths.primary_um() * 1e-3;
    let SurfaceShape::OddAsphere { coefficients, .. } = &mut equivalent.surfaces[1].shape else {
        panic!("odd asphere sample");
    };
    coefficients[1] -= 2.0 * wave / 25.0;
    let expected = efl(&equivalent);
    assert!((expected - bare).abs() > 1e-3, "{expected} vs {bare}");

    let spherical = ParamPath::Irregularity {
        surface: sys.surfaces[1].id,
        mode: IrregularityMode::Spherical,
    };
    spherical.set(&mut sys, 0.0).unwrap();
    assert_eq!(efl(&sys), bare);
    spherical.set(&mut sys, 1.0).unwrap();
    close(efl(&sys), expected, 1e-9);
    // The real-ray tracer builds on the same first-order model.
    tracer(sys);
}

#[test]
fn reflective_doe_has_signed_focus_and_the_same_phase_lag() {
    let radius = 2.0;
    let f = 40.0;
    let a = -std::f64::consts::PI * radius * radius / (D_LINE * 1e-3 * f);
    let mut mirror = surf(1, doe(vec![a], radius, 1), -f, MaterialRef::Air);
    mirror.interaction = mklight_core::Interaction::Reflect;
    let rs = resolve(infinite_system(2.0, vec![mirror], 2));
    let p = Paraxial::new(&rs).unwrap();
    close(p.primary().bfl, -f, 1e-12);
    let (ray, rec) = trace_ray_recorded(
        &rs,
        &Ray::new(Vec3::new(1e-3, 0.0, 0.0), Vec3::Z),
        0,
        &Default::default(),
    );
    assert!(ray.status.is_ok());
    assert!(ray.ray.dir.z < 0.0);
    close(rec[1].opl, -1e-6 / (2.0 * f), 1e-22);
    close(ray.ray.pos.x, 0.0, 1e-12);
}

/// Regression: a ray that misses the base conic and then runs out of the
/// sag domain without any sign change was reported as `OutsideDomain` when
/// backtracking hit the domain edge. Like the even asphere, it is a miss.
#[test]
fn missed_seed_leaving_the_domain_is_a_miss() {
    let shape = SurfaceShape::QCon {
        curvature: 0.18,
        conic: 0.0,
        norm_radius: 5.2,
        coefficients: vec![0.05, -0.01, 0.003],
    };
    let p = Vec3::new(1.416408, -1.029432, -10.0);
    let d = Vec3::new(0.256578, -0.291829, 0.921414).normalized();
    assert_eq!(
        intersect(&shape, p, d, &Default::default()),
        Err(IntersectError::Miss)
    );
}
