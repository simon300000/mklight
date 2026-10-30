//! Plane-parallel plate: exact two-reflection ghost geometry, Fresnel
//! intensities, coated vs bare, and the paraxial ghost focus shift.

mod common;

use common::*;
use mklight_core::{ApertureSpec, JobContext, MaterialRef, OpticalSystem, SurfaceShape};
use mklight_ghost::{
    GhostPath, IrradianceSettings, ParaxialGhostSettings, ghost_irradiance, paraxial_ghosts,
    real_ghost_focus,
};

/// Hand-written Fresnel amplitude coefficients (air → glass), independent
/// of the shared coating code: `(R_s, R_p)` at incidence `theta`.
fn fresnel_by_hand(n: f64, theta: f64) -> (f64, f64) {
    let ci = theta.cos();
    let st = theta.sin() / n;
    let ct = (1.0 - st * st).sqrt();
    let rs = (ci - n * ct) / (ci + n * ct);
    let rp = (n * ci - ct) / (n * ci + ct);
    (rs * rs, rp * rp)
}

#[test]
fn two_reflection_geometry_and_intensity_oblique() {
    let (n, d) = (1.5, 10.0);
    let theta_deg = 10.0;
    let m = model(plate(n, d, 20.0, &[0.0, theta_deg], None), unclipped());
    let tracer = m.tracer();
    let g = tracer.generate(tracer.field(1), 0, 0.0, 0.0).unwrap();
    let prim = m.check_path(&GhostPath::primary()).unwrap();
    let ghost = m.check_path(&GhostPath::new(vec![3, 2])).unwrap();
    let p = m.trace_steps(&g.ray, 0, &prim, None);
    let q = m.trace_steps(&g.ray, 0, &ghost, None);
    assert!(p.status.is_ok() && q.status.is_ok(), "{p:?} {q:?}");

    let theta = theta_deg.to_radians();
    let tp = (theta.sin() / n).asin();
    // Lateral displacement 2 d tan θ' along +y, same direction, no x shift.
    let dy = q.ray.pos.y - p.ray.pos.y;
    assert!((dy - 2.0 * d * tp.tan()).abs() < 1e-12, "dy = {dy}");
    assert!((q.ray.pos.x - p.ray.pos.x).abs() < 1e-14);
    assert!(q.ray.dir.max_abs_diff(p.ray.dir) < 1e-14);
    // Extra optical path: one glass round trip, 2 n d / cos θ'.
    let dopl = q.opl.diff(&p.opl);
    assert!(
        (dopl - 2.0 * n * d / tp.cos()).abs() < 1e-11,
        "ΔOPL = {dopl}"
    );

    // Intensity: T² R² per polarisation (reciprocity: the inside
    // reflectance at θ' equals the outside one at θ).
    let (rs, rp) = fresnel_by_hand(n, theta);
    let (ts, tpp) = (1.0 - rs, 1.0 - rp);
    assert!((p.weight_s - ts * ts).abs() < 1e-14);
    assert!((p.weight_p - tpp * tpp).abs() < 1e-14);
    assert!((q.weight_s - ts * ts * rs * rs).abs() < 1e-15);
    assert!((q.weight_p - tpp * tpp * rp * rp).abs() < 1e-15);
    // Unpolarised weight: s and p averaged at the end, not per surface.
    let w = 0.5 * (ts * ts * rs * rs + tpp * tpp * rp * rp);
    assert!((q.weight() - w).abs() < 1e-16);
    let r_avg = 0.5 * (rs + rp);
    let t_avg = 1.0 - r_avg;
    assert!((q.weight() - t_avg * t_avg * r_avg * r_avg).abs() > 1e-9);
}

#[test]
fn normal_incidence_ghost_ratio_is_r_squared() {
    let n = 1.5168;
    let m = model(plate(n, 8.0, 20.0, &[0.0], None), unclipped());
    let settings = IrradianceSettings {
        paths: Some(vec![GhostPath::new(vec![3, 2])]),
        pupil_samples: 24,
        overfill: 1.0,
        map_bins: [16, 16],
        map_half_size: Some([6.0, 6.0]),
        keep_path_maps: true,
        ..Default::default()
    };
    let res = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
    let r = r_normal(1.0, n);
    let g = &res.ghosts[0];
    // Every ray meets the plate at normal incidence: ghost / primary = R²
    // ray by ray, so for flux and for every bin.
    assert!(
        (g.flux_ratio - r * r).abs() < 1e-14 * r * r + 1e-16,
        "{}",
        g.flux_ratio
    );
    assert!((g.peak_ratio - r * r).abs() < 1e-12 * r * r);
    let gm = g.map.as_ref().unwrap();
    let pm = res.primary.map.as_ref().unwrap();
    for (a, b) in gm.values.iter().zip(&pm.values) {
        assert!((a - r * r * b).abs() <= 1e-12 * b.abs());
    }
    // Unit pupil area: boundary quadrature must conserve incident flux.
    let t2 = (1.0 - r) * (1.0 - r);
    assert!((res.primary.flux / t2 - 1.0).abs() < 3e-4);
    assert!((res.total_flux_ratio - r * r).abs() < 1e-14);
}

#[test]
fn quarter_wave_ar_reduces_ghost_by_r_ratio() {
    let n = 1.5168;
    let nf = 1.38;
    let run = |coating| {
        let m = model(plate(n, 8.0, 20.0, &[0.0], coating), unclipped());
        let settings = IrradianceSettings {
            paths: Some(vec![GhostPath::new(vec![3, 2])]),
            pupil_samples: 12,
            overfill: 1.0,
            map_bins: [8, 8],
            map_half_size: Some([6.0, 6.0]),
            ..Default::default()
        };
        let r = ghost_irradiance(&m, &settings, &JobContext::new()).unwrap();
        (r.ghosts[0].flux, r.primary.flux, r.ghosts[0].flux_ratio)
    };
    let (gb, pb, rb_ratio) = run(None);
    let (gc, pc, rc_ratio) = run(Some(quarter_wave(nf, D_LINE)));
    // Single quarter-wave layer at its design wavelength, normal incidence:
    // R = ((n₀ n_s − n_f²)/(n₀ n_s + n_f²))², lossless, same from both sides.
    let rc = ((n - nf * nf) / (n + nf * nf)).powi(2);
    let rb = r_normal(1.0, n);
    let expect = (rc * rc * (1.0 - rc).powi(2)) / (rb * rb * (1.0 - rb).powi(2));
    assert!(
        ((gc / gb) / expect - 1.0).abs() < 1e-10,
        "{} vs {expect}",
        gc / gb
    );
    assert!(((rc_ratio / rb_ratio) / (rc / rb).powi(2) - 1.0).abs() < 1e-10);
    // The AR coating transmits more signal.
    assert!(pc > pb);
}

/// Ideal lens (f = 100, stop) → plate → sensor at the primary focus.
fn converging_plate(n: f64, d: f64) -> OpticalSystem {
    let mut s = plate(n, d, 0.0, &[0.0, 2.0], None);
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.surfaces[1] = surf(
        1,
        SurfaceShape::Paraxial {
            focal_length: 100.0,
        },
        20.0,
        MaterialRef::Air,
    );
    s.surfaces[3].thickness = 100.0 - 20.0 - d / n;
    s
}

#[test]
fn ghost_focus_shift_in_converging_beam() {
    // Unfolded, the ghost crosses 3d of glass instead of d, so its paraxial
    // focus is 2d/n in front of the primary focus (−z on the sensor).
    let (n, d) = (1.5, 10.0);
    let m = model(converging_plate(n, d), unclipped());
    let rep = paraxial_ghosts(
        &m,
        &ParaxialGhostSettings {
            paths: Some(vec![GhostPath::new(vec![3, 2])]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rep.primary_focus_from_sensor.abs() < 1e-12);
    let g = &rep.ghosts[0];
    assert!(
        (g.focus_from_sensor + 2.0 * d / n).abs() < 1e-11,
        "{}",
        g.focus_from_sensor
    );
    // Blur: behind the plate the ghost marginal ray has the primary's air
    // slope −5/100 and has passed its focus 2d/n before the sensor.
    let y_m = 5.0 * (2.0 * d / n) / 100.0;
    assert!(
        (g.spot_diameter - 2.0 * y_m).abs() < 1e-12,
        "{}",
        g.spot_diameter
    );
    let r = r_normal(1.0, n);
    assert!((g.reflectance_product - r * r).abs() < 1e-15);
    // Real ghost focus at a small aperture agrees with the paraxial one.
    let z = real_ghost_focus(&m, &GhostPath::new(vec![3, 2]), 0, 0.01).unwrap();
    assert!(
        (z - g.focus_from_sensor).abs() < 1e-5,
        "{z} vs {}",
        g.focus_from_sensor
    );
    // ... and departs from it at full aperture (plate spherical aberration).
    let zf = real_ghost_focus(&m, &GhostPath::new(vec![3, 2]), 0, 1.0).unwrap();
    assert!((zf - g.focus_from_sensor).abs() > 1e-3);
}

#[test]
fn curved_sensor_reflection_paraxial_matches_real() {
    // The sensor reflects about its own curved normal; the paraxial model
    // must use the reflection power −2cN there too. The sensor sits δ in
    // front of the primary focus so that the beam has a finite height on it.
    let (n, d, delta, c) = (1.5, 10.0, 5.0, -0.02);
    let mut sys = converging_plate(n, d);
    sys.surfaces[3].thickness -= delta;
    sys.surfaces[4].shape = SurfaceShape::sphere(1.0 / c);
    let a = sys.surfaces[3].thickness;
    let settings = mklight_ghost::GhostSettings {
        sensor_reflectance: Some(0.1),
        ..unclipped()
    };
    let m = model(sys, settings);
    let rep = paraxial_ghosts(
        &m,
        &ParaxialGhostSettings {
            paths: Some(vec![GhostPath::new(vec![4, 3]), GhostPath::new(vec![4, 2])]),
            ..Default::default()
        },
    )
    .unwrap();
    // [4, 3] by vergences (air, unfolded): the beam arrives converging to
    // δ behind the sensor (V = 1/δ); the concave sensor (radius 1/c < 0,
    // seen from the plate) adds −2c: the reflected beam focuses
    // 1/(1/δ − 2c) in front of the sensor, then travels a to the flat rear
    // face and a back. The ghost diverges from that point, 2a − 1/(1/δ − 2c)
    // in front of the sensor. (A flat sensor would give δ − 2a.)
    let expect = 1.0 / (1.0 / delta - 2.0 * c) - 2.0 * a;
    let g43 = rep
        .ghosts
        .iter()
        .find(|g| g.path == GhostPath::new(vec![4, 3]))
        .unwrap();
    assert!(
        (g43.focus_from_sensor - expect).abs() < 1e-9,
        "{} vs {expect}",
        g43.focus_from_sensor
    );
    assert!((g43.focus_from_sensor - (delta - 2.0 * a)).abs() > 0.5);
    for g in &rep.ghosts {
        let zp = g.focus_from_sensor;
        let z1 = real_ghost_focus(&m, &g.path, 0, 0.02).unwrap();
        let z2 = real_ghost_focus(&m, &g.path, 0, 0.01).unwrap();
        // Converges quadratically in the aperture to the paraxial value.
        assert!((z2 - zp).abs() < 1e-3, "{:?}: {z2} vs {zp}", g.path);
        let ratio = (z1 - zp) / (z2 - zp);
        assert!((ratio - 4.0).abs() < 0.1, "{:?}: {ratio}", g.path);
    }
}

#[test]
fn arbitrary_field_points() {
    use mklight_ghost::FieldRef;
    use mklight_trace::{FieldPoint, ObjectSource};
    let (n, d) = (1.5, 10.0);
    let sys = plate(n, d, 20.0, &[0.0, 5.0], None);
    let m = model(sys.clone(), unclipped());
    let base = IrradianceSettings {
        paths: Some(vec![GhostPath::new(vec![3, 2])]),
        pupil_samples: 16,
        overfill: 1.0,
        map_bins: [16, 16],
        map_half_size: Some([20.0, 20.0]),
        ..Default::default()
    };
    let ctx = JobContext::new();
    // A point equal to system field 1 gives the same result.
    let by_index = ghost_irradiance(
        &m,
        &IrradianceSettings {
            fields: vec![1],
            ..base.clone()
        },
        &ctx,
    )
    .unwrap();
    let point = FieldPoint::system(&sys, 1);
    let by_point = ghost_irradiance(
        &m,
        &IrradianceSettings {
            points: vec![point],
            ..base.clone()
        },
        &ctx,
    )
    .unwrap();
    assert_eq!(
        by_index.primary.fields[0].field,
        FieldRef::System { index: 1 }
    );
    assert_eq!(by_point.primary.fields.len(), 1, "points only");
    assert_eq!(by_point.primary.fields[0].field, FieldRef::Point { point });
    assert_eq!(by_index.ghosts[0].flux, by_point.ghosts[0].flux);
    assert_eq!(by_index.total_ghost_map, by_point.total_ghost_map);

    // A source three times outside the field: the ghost is displaced
    // laterally by exactly 2 d tan θ' for every ray, so the centroids are.
    let far = FieldPoint::normalized(0.0, 3.0);
    let res = ghost_irradiance(
        &m,
        &IrradianceSettings {
            fields: vec![0],
            points: vec![far],
            ..base
        },
        &ctx,
    )
    .unwrap();
    assert_eq!(res.primary.fields.len(), 2);
    let ObjectSource::Direction { dir } = m.tracer().prepare_field(far).unwrap().source else {
        panic!("object at infinity");
    };
    let sin_t = dir.x.hypot(dir.y) / dir.norm();
    assert!(sin_t > 0.2, "≈ 15°: {sin_t}");
    let st = sin_t / n;
    let shift = 2.0 * d * st / (1.0 - st * st).sqrt();
    let (p, g) = (&res.primary.fields[1], &res.ghosts[0].fields[1]);
    assert_eq!(p.field, FieldRef::Point { point: far });
    let dy = (g.centroid[1] - p.centroid[1]).abs();
    assert!((dy - shift).abs() < 1e-10, "{dy} vs {shift}");
}
