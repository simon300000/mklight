//! Field curvature and distortion: analytic cases and the Coddington ↔
//! close-ray cross-check on every sample.

mod common;

use common::*;
use mklight_analysis::curvature::{
    FieldCurvatureSettings, GridDistortionSettings, field_curvature, grid_distortion,
};

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
