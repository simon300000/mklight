//! Field curvature and distortion: analytic cases and the Coddington ↔
//! close-ray cross-check on every sample.

use crate::common;

use common::*;
use mklight_analysis::curvature::{
    FieldCurvatureSettings, GridDistortionSettings, close_ray_focus, coddington_focus,
    differential_focus, field_curvature, grid_distortion,
};
use mklight_trace::{FieldPoint, TraceOptions};

#[test]
fn coddington_matches_close_rays_on_every_sample() {
    for (name, t) in samples() {
        let r = field_curvature(
            &t,
            &FieldCurvatureSettings {
                field_samples: 7,
                ..Default::default()
            },
        )
        .unwrap();
        for c in &r.curves {
            for i in 0..r.field_values.len() {
                let (a, b) = (c.tangential[i], c.tangential_close_rays[i]);
                let (p, q) = (c.sagittal[i], c.sagittal_close_rays[i]);
                assert!(a.is_finite() && p.is_finite(), "{name}");
                assert!((a - b).abs() < 1e-5 * (1.0 + a.abs()), "{name} T {a} {b}");
                assert!((p - q).abs() < 1e-5 * (1.0 + p.abs()), "{name} S {p} {q}");
            }
        }
    }
}

/// The differential pencil (the analytic tangent of the chief ray) agrees
/// with the Hessian Coddington trace to rounding, four orders tighter than
/// the close rays, on every field and wavelength of every sample; and it is
/// the limit of the close rays: their error against it falls as h² and then
/// plateaus on the cancellation floor.
#[test]
fn differential_pencil_matches_coddington_to_rounding_on_every_sample() {
    let mut compared = 0;
    let mut undefined = 0;
    for (name, t) in samples() {
        for f in t.fields() {
            for w in 0..t.num_wavelengths() {
                let Some(a) = coddington_focus(&t, f, w) else {
                    continue;
                };
                let Some(b) = differential_focus(&t, f, w) else {
                    undefined += 1;
                    continue;
                };
                for (x, y) in [(a.tangential, b.tangential), (a.sagittal, b.sagittal)] {
                    if x.is_finite() && y.is_finite() {
                        assert!((x - y).abs() < 1e-9 * (1.0 + x.abs()), "{name}: {x} vs {y}");
                    }
                }
                compared += 1;
            }
        }
    }
    assert!(
        compared > 40 && undefined == 0,
        "{compared} compared, {undefined} undefined"
    );
    // Error plateau of the close rays against the differential pencil.
    let (name, t) = samples()
        .into_iter()
        .find(|(n, _)| *n == "double_gauss")
        .unwrap();
    let f = &t.fields()[t.fields().len() - 1];
    let d = differential_focus(&t, f, 0).unwrap();
    let errors: Vec<f64> = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        .iter()
        .map(|&h| {
            let c = close_ray_focus(&t, f, 0, h).unwrap();
            (c.tangential - d.tangential)
                .abs()
                .max((c.sagittal - d.sagittal).abs())
        })
        .collect();
    assert!(
        errors[1] < errors[0] && errors[2] < errors[1] / 10.0,
        "{name}: {errors:?}"
    );
    let floor = errors.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(floor < 1e-8 && errors[6] > floor, "{name}: {errors:?}");
}

#[test]
fn concentric_mirror_focal_surface_is_a_sphere() {
    // Stop at the centre of curvature: every chief ray is an axis of
    // symmetry, so the focal surface is the sphere of radius R/2 about C:
    // Δz = −(R/2)(1 − cos θ), with no astigmatism.
    let r = 200.0;
    let t = tracer(concentric_mirror(r, 10.0, &[0.0, 5.0, 10.0]));
    let res = field_curvature(
        &t,
        &FieldCurvatureSettings {
            field_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let c = &res.curves[0];
    for (i, &h) in res.normalized_field.iter().enumerate() {
        let th = (10.0 * h).to_radians();
        let expect = -0.5 * r * (1.0 - th.cos());
        assert!(
            (c.tangential[i] - expect).abs() < 1e-9,
            "{} {}",
            c.tangential[i],
            expect
        );
        assert!((c.sagittal[i] - expect).abs() < 1e-9);
        assert!((c.tangential_close_rays[i] - expect).abs() < 1e-5);
    }
}

#[test]
fn ideal_lens_is_flat_and_distortion_free() {
    let f = 100.0;
    let t = tracer(ideal_lens(f, 10.0, &[0.0, 10.0, 20.0]));
    let res = field_curvature(
        &t,
        &FieldCurvatureSettings {
            field_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let c = &res.curves[0];
    for (i, &h) in res.normalized_field.iter().enumerate() {
        let th = (20.0 * h).to_radians();
        assert!(c.tangential[i].abs() < 1e-9 && c.sagittal[i].abs() < 1e-9);
        assert!(
            c.distortion_percent[i].abs() < 1e-10,
            "{}",
            c.distortion_percent[i]
        );
        assert!(c.f_tan_theta_distortion_percent[i].abs() < 1e-10);
        assert!((c.real_height[i] - f * th.tan()).abs() < 1e-11);
        let ftheta = if h == 0.0 {
            0.0
        } else {
            100.0 * (th.tan() / th - 1.0)
        };
        assert!((c.f_theta_distortion_percent[i] - ftheta).abs() < 1e-9);
    }
    let g = grid_distortion(&t, &GridDistortionSettings::default()).unwrap();
    assert_eq!(g.points.len(), 121);
    assert!(g.max_abs_distortion_percent < 1e-10);
}

/// An ideal lens with the image plane defocused by `dz`: the chief ray
/// passes the lens (the stop) undeviated, so its height is `(f + dz) tan θ`
/// exactly. The `f·tan θ` reference must use that image-surface scale (not
/// `n₀·EFL = f`, which would give a constant `100 dz/f` % offset that jumps to
/// the forced 0 on axis): every distortion is 0 and `f·θ` is
/// `100(tan θ/θ − 1)` as at focus.
#[test]
fn defocused_ideal_lens_is_distortion_free() {
    let (f, dz) = (100.0, 2.0);
    let mut sys = ideal_lens(f, 10.0, &[0.0, 10.0, 20.0]);
    sys.surfaces[1].thickness = f + dz;
    let res = field_curvature(
        &tracer(sys),
        &FieldCurvatureSettings {
            field_samples: 9,
            ..Default::default()
        },
    )
    .unwrap();
    let c = &res.curves[0];
    for (i, &h) in res.normalized_field.iter().enumerate() {
        let th = (20.0 * h).to_radians();
        assert!((c.real_height[i] - (f + dz) * th.tan()).abs() < 1e-11);
        assert!(c.distortion_percent[i].abs() < 1e-10);
        assert!(
            c.f_tan_theta_distortion_percent[i].abs() < 1e-10,
            "H = {h}: {}",
            c.f_tan_theta_distortion_percent[i]
        );
        let ftheta = if h == 0.0 {
            0.0
        } else {
            100.0 * (th.tan() / th - 1.0)
        };
        assert!((c.f_theta_distortion_percent[i] - ftheta).abs() < 1e-9);
    }
}

/// Every rotational sample with an object at infinity, its image plane moved
/// 0.5 mm off focus: the `f·tan θ` and `f·θ` curves tend to 0 continuously
/// as `H → 0` (the first sample off axis, H = 1/40, is within `H²`-scaled
/// reach of the axis value), and `f·tan θ` equals the paraxial distortion.
#[test]
fn f_tan_theta_distortion_is_continuous_on_axis_off_focus() {
    let mut checked = 0;
    for (name, mut sys) in mklight_core::samples::all()
        .into_iter()
        .filter(|(_, s)| !mklight_core::samples::has_extended_surfaces(s))
    {
        let n = sys.surfaces.len();
        sys.surfaces[n - 2].thickness += 0.5;
        let rotational = mklight_core::surface_math::system_symmetry(&sys)
            == mklight_core::surface_math::Symmetry::Rotational;
        let t = tracer(sys);
        if !(rotational && t.system().object_at_infinity) {
            continue;
        }
        let res = field_curvature(
            &t,
            &FieldCurvatureSettings {
                field_samples: 41,
                ..Default::default()
            },
        )
        .unwrap();
        for c in &res.curves {
            let (ft, fa) = (
                &c.f_tan_theta_distortion_percent,
                &c.f_theta_distortion_percent,
            );
            let edge = ft[40].abs().max(fa[40].abs());
            assert_eq!((ft[0], fa[0]), (0.0, 0.0), "{name}");
            // Near-axis distortion grows as H²: (1/40)² of the edge value,
            // with a generous factor, plus a floor for rounding.
            for v in [ft[1], fa[1]] {
                assert!(
                    v.abs() < 1e-6 + 4.0 * edge / 1600.0,
                    "{name}: {v} (edge {edge})"
                );
            }
            for (i, (&f, &d)) in ft.iter().zip(&c.distortion_percent).enumerate() {
                assert!((f - d).abs() < 1e-9 * (1.0 + d.abs()), "{name} {i}");
            }
        }
        checked += 1;
    }
    assert!(checked >= 2, "{checked}");
}

/// Spherical mirror of radius `r` between two coordinate breaks with the
/// same tilts (the second re-aims the axis along the reflected chief ray),
/// stop at the mirror, image `r/2` behind it.
fn tilted_mirror(r: f64, tilt_x: f64, tilt_y: f64) -> mklight_trace::SequentialTracer {
    use mklight_core::{Interaction, MaterialRef, Placement, SurfaceShape};
    let cb = |id: u64, t: f64| {
        let mut c = surf(id, SurfaceShape::CoordinateBreak, t, MaterialRef::Air);
        c.placement = Some(Placement {
            tilt_x,
            tilt_y,
            ..Default::default()
        });
        c
    };
    let mut m = surf(2, SurfaceShape::sphere(-r), 0.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    tracer(infinite_system(
        20.0,
        vec![cb(1, 0.0), m, cb(3, -0.5 * r)],
        2,
        &[0.0, 3.0],
    ))
}

/// Mirror incidence cosine and the chief path from the mirror to the image.
fn mirror_geometry(t: &mklight_trace::SequentialTracer, h: f64) -> (f64, f64) {
    use mklight_trace::trace::to_global;
    let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
    let mut rec = Vec::new();
    let r = t.trace_with(&f, 0, 0.0, 0.0, &TraceOptions::unclipped(), Some(&mut rec));
    assert!(r.status.is_ok());
    let sys = t.system();
    let g = |i: usize| to_global(sys, i, &mklight_trace::Ray::new(rec[i].point, rec[i].dir)).pos;
    (rec[2].cos_incidence, (g(4) - g(2)).norm())
}

#[test]
fn mirror_tilted_about_y_labels_foci_by_the_image_frame() {
    // Tilt about y: the plane of incidence is x–z, so the mirror's in-plane
    // focus (R/2) cos I images the image-frame x (sagittal) direction and
    // R/(2 cos I) the y (tangential) one. The plane-of-incidence labelling
    // swapped them silently.
    let r = 200.0;
    for (tx, ty) in [(0.0, 10.0), (10.0, 0.0)] {
        let t = tilted_mirror(r, tx, ty);
        for h in [0.0, 1.0] {
            let (ci, d) = mirror_geometry(&t, h);
            let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
            let a = coddington_focus(&t, &f, 0).unwrap();
            let b = close_ray_focus(&t, &f, 0, 1e-3).unwrap();
            if h == 0.0 {
                let (inplane, perp) = (0.5 * r * ci - d, 0.5 * r / ci - d);
                let (et, es) = if ty != 0.0 {
                    (perp, inplane)
                } else {
                    (inplane, perp)
                };
                assert!((a.tangential - et).abs() < 1e-9, "{tx} {ty}: {a:?} vs {et}");
                assert!((a.sagittal - es).abs() < 1e-9, "{tx} {ty}: {a:?} vs {es}");
                assert!((a.tangential - a.sagittal).abs() > 3.0);
            }
            assert!((a.dir_z + 1.0).abs() < 0.01);
            assert!(
                (a.tangential - b.tangential).abs() < 1e-5 * (1.0 + a.tangential.abs()),
                "{tx} {ty} H {h}: {a:?} vs close rays {b:?}"
            );
            assert!(
                (a.sagittal - b.sagittal).abs() < 1e-5 * (1.0 + a.sagittal.abs()),
                "{tx} {ty} H {h}: {a:?} vs close rays {b:?}"
            );
        }
    }
}

#[test]
fn compound_tilted_mirror_matches_close_rays_and_focal_lines() {
    // Tilts about x and y: the astigmatic focal lines are rotated with
    // respect to the image frame and the chief ray is skew. Both methods
    // must find the two focal lines, (R/2) cos I and R/(2 cos I) from the
    // mirror, and label them identically.
    let r = 200.0;
    let t = tilted_mirror(r, 7.0, 7.0);
    for h in [0.0, 0.5, 1.0] {
        let (ci, d) = mirror_geometry(&t, h);
        let f = t.prepare_field(FieldPoint::normalized(0.0, h)).unwrap();
        let a = coddington_focus(&t, &f, 0).unwrap();
        let b = close_ray_focus(&t, &f, 0, 1e-3).unwrap();
        let mut got = [a.tangential, a.sagittal];
        got.sort_by(f64::total_cmp);
        let expect = [0.5 * r * ci - d, 0.5 * r / ci - d];
        assert!((got[0] - expect[0]).abs() < 1e-9 && (got[1] - expect[1]).abs() < 1e-9);
        assert!(
            (a.tangential - b.tangential).abs() < 1e-5 * (1.0 + a.tangential.abs()),
            "H {h}: {a:?} vs close rays {b:?}"
        );
        assert!(
            (a.sagittal - b.sagittal).abs() < 1e-5 * (1.0 + a.sagittal.abs()),
            "H {h}: {a:?} vs close rays {b:?}"
        );
    }
}
