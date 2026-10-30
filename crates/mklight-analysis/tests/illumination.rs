//! Relative illumination, vignetting and footprints against closed forms.

mod common;

use common::*;
use mklight_analysis::illumination::*;
use mklight_core::{MaterialRef, SemiDiameter, SurfaceShape};

/// Projected solid angle of a disk of radius `a` seen from a point at axial
/// distance `f` and lateral offset `h`, from a surface element parallel to
/// the disk: `π/2 · (1 − (f² + h² − a²)/√((f² + h² + a²)² − 4a²h²))`.
fn disk_psa(a: f64, f: f64, h: f64) -> f64 {
    let s = f * f + h * h;
    0.5 * std::f64::consts::PI
        * (1.0 - (s - a * a) / ((s + a * a).powi(2) - 4.0 * a * a * h * h).sqrt())
}

#[test]
fn ideal_lens_relative_illumination_is_the_disk_view_factor() {
    let (f, epd, th) = (100.0, 10.0, 30.0f64);
    let t = tracer(ideal_lens(f, epd, &[0.0, th]));
    let r = relative_illumination(
        &t,
        &RelativeIlluminationSettings {
            field_samples: 4,
            ..Default::default()
        },
    )
    .unwrap();
    for (i, &h) in r.normalized_field.iter().enumerate() {
        let a = (th * h).to_radians();
        let exact = disk_psa(0.5 * epd, f, f * a.tan());
        let ri = exact / disk_psa(0.5 * epd, f, 0.0);
        assert!(
            (r.projected_solid_angle[i] - exact).abs() < 1e-8 * exact,
            "{} {}",
            r.projected_solid_angle[i],
            exact
        );
        assert!((r.relative_illumination[i] - ri).abs() < 1e-8);
        // Small aperture: the cos⁴ law.
        assert!((r.relative_illumination[i] - a.cos().powi(4)).abs() < 2e-3);
        assert!((r.cos4_object[i] - a.cos().powi(4)).abs() < 1e-12);
        assert!((r.chief_ray_angle_deg[i] - th * h).abs() < 1e-9);
        assert_eq!(r.vignetting_fraction[i], 0.0);
    }
}

/// Area of the intersection of two disks (radii `r1`, `r2`, centres `d` apart).
fn lens_area(r1: f64, r2: f64, d: f64) -> f64 {
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return std::f64::consts::PI * r1.min(r2).powi(2);
    }
    let a1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos();
    let a2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos();
    r1 * r1 * (a1 - a1.sin() * a1.cos()) + r2 * r2 * (a2 - a2.sin() * a2.cos())
}

#[test]
fn vignetting_by_a_field_aperture_matches_disk_overlap() {
    // Ideal lens (stop), then a clipping aperture 40 mm behind it: the beam
    // there is a disk of radius a(1 − d/f) centred at d·tan θ.
    let (f, epd, th, d, ra) = (100.0, 10.0, 6.0f64, 40.0, 4.0);
    let mut clip = surf(2, SurfaceShape::Plane, f - d, MaterialRef::Air);
    clip.semi_diameter = SemiDiameter::Fixed { value: ra };
    let sys = infinite_system(
        epd,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial { focal_length: f },
                d,
                MaterialRef::Air,
            ),
            clip,
        ],
        1,
        &[0.0, th],
    );
    let t = tracer(sys);
    let r = relative_illumination(
        &t,
        &RelativeIlluminationSettings {
            field_samples: 3,
            radial_samples: 32,
            azimuth_samples: 128,
            ..Default::default()
        },
    )
    .unwrap();
    let rb = 0.5 * epd * (1.0 - d / f);
    for (i, &h) in r.normalized_field.iter().enumerate() {
        let e = d * (th * h).to_radians().tan();
        let pass = lens_area(rb, ra, e) / (std::f64::consts::PI * rb * rb);
        assert!(
            ((1.0 - r.vignetting_fraction[i]) - pass).abs() < 0.01,
            "H {h}: {} vs {pass}",
            1.0 - r.vignetting_fraction[i]
        );
    }
    assert_eq!(r.vignetting_fraction[0], 0.0);
    assert!(r.vignetting_fraction[2] > 0.2);
    assert!(r.relative_illumination[2] < 0.8);
}

#[test]
fn footprint_on_an_ideal_lens() {
    let mut sys = ideal_lens(100.0, 10.0, &[0.0, 5.0]);
    let t = tracer(sys.clone());
    let fp = footprint(&t, &FootprintSettings::default()).unwrap();
    assert_eq!(fp.surface, 1);
    assert!((fp.max_radius - 5.0).abs() < 1e-12);
    assert!((fp.semi_diameter - 5.0).abs() < 1e-12);
    for r in &fp.rays {
        assert!(r.clipped.is_empty() && r.lost_before == 0);
        assert_eq!(r.points.len(), r.launched);
    }
    // A smaller fixed aperture clips the rim.
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 4.0 };
    let t = tracer(sys);
    let fp = footprint(&t, &FootprintSettings::default()).unwrap();
    let r = &fp.rays[0];
    assert!(!r.clipped.is_empty());
    assert!(r.points.iter().all(|p| p[0].hypot(p[1]) <= 4.0));
    assert!(r.clipped.iter().all(|p| p[0].hypot(p[1]) > 4.0));
    assert_eq!(r.points.len() + r.clipped.len(), r.launched);
    assert!(fp.max_radius <= 4.0);
}
