//! Clipping semantics of `ClipMode`: passive surfaces never clip, sensor
//! reflections only inside the sensor rectangle, `FixedApertures` bounds an
//! `Auto` stop, and the coverage record carries the apertures.

use crate::common;

use common::*;
use mklight_core::{
    ApertureSpec, Interaction, JobContext, MaterialRef, OpticalSystem, Sensor, SurfaceShape,
};
use mklight_ghost::{
    Clip, ClipMode, GhostModel, GhostPath, GhostSettings, IrradianceSettings,
    ParaxialGhostSettings, ghost_irradiance, paraxial_ghosts,
};

/// Ideal lens (f = 100, stop) → plate (n, d) → sensor at the primary focus.
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

/// Insert a passive plane `dz` behind surface `after` (splitting its gap).
fn insert_plane(sys: &OpticalSystem, after: usize, dz: f64, dummy: bool) -> OpticalSystem {
    let mut s = sys.clone();
    let t = s.surfaces[after].thickness;
    s.surfaces[after].thickness = dz;
    let mut p = surf(
        100,
        SurfaceShape::Plane,
        t - dz,
        s.surfaces[after].material.clone(),
    );
    if dummy {
        p.interaction = Interaction::Dummy;
    }
    s.surfaces.insert(after + 1, p);
    if s.stop > after {
        s.stop += 1;
    }
    s
}

/// The path of `base` in a system with a plane inserted after `after`.
fn shift(path: &GhostPath, after: usize) -> GhostPath {
    GhostPath::new(
        path.reflections
            .iter()
            .map(|&r| if r as usize > after { r + 1 } else { r })
            .collect(),
    )
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}

#[test]
fn passive_planes_leave_ghost_results_unchanged() {
    let base = converging_plate(1.5, 10.0);
    let paths = vec![GhostPath::new(vec![3, 2])];
    let irr = |m: &GhostModel, paths: Vec<GhostPath>| {
        let s = IrradianceSettings {
            paths: Some(paths),
            pupil_samples: 32,
            map_bins: [16, 16],
            map_half_size: Some([2.0, 2.0]),
            ..Default::default()
        };
        ghost_irradiance(m, &s, &JobContext::new()).unwrap()
    };
    let par = |m: &GhostModel, paths: Vec<GhostPath>| {
        paraxial_ghosts(
            m,
            &ParaxialGhostSettings {
                paths: Some(paths),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let m0 = model(base.clone(), GhostSettings::default());
    let (i0, p0) = (irr(&m0, paths.clone()), par(&m0, paths.clone()));
    assert!(i0.ghosts[0].flux > 1e-4, "{}", i0.ghosts[0].flux);
    // A dummy in the air gap, one inside the glass, one 2 mm before the
    // sensor (where the defocused ghost is far wider than the primary
    // beam), and a bare air-to-air plane that is not the stop.
    let variants = [
        (1, 5.0, true),
        (2, 4.0, true),
        (3, base.surfaces[3].thickness - 2.0, true),
        (3, base.surfaces[3].thickness - 2.0, false),
    ];
    for (after, dz, dummy) in variants {
        let sys = insert_plane(&base, after, dz, dummy);
        let m = model(sys, GhostSettings::default());
        assert_eq!(m.clip(after + 1), Clip::None);
        let sp: Vec<GhostPath> = paths.iter().map(|p| shift(p, after)).collect();
        let (i1, p1) = (irr(&m, sp.clone()), par(&m, sp));
        let what = format!("plane after {after} (dummy: {dummy})");
        assert!(rel(i1.primary.flux, i0.primary.flux) < 1e-12, "{what}");
        assert!(
            rel(i1.ghosts[0].flux, i0.ghosts[0].flux) < 1e-9,
            "{what}: {} vs {}",
            i1.ghosts[0].flux,
            i0.ghosts[0].flux
        );
        for (a, b) in i1.ghosts[0].fields.iter().zip(&i0.ghosts[0].fields) {
            assert_eq!(a.rays_detected, b.rays_detected, "{what}");
        }
        let (g1, g0) = (&p1.ghosts[0], &p0.ghosts[0]);
        assert!(rel(g1.focus_from_sensor, g0.focus_from_sensor) < 1e-12);
        assert!(rel(g1.spot_diameter, g0.spot_diameter) < 1e-12);
        assert_eq!(g1.vignetting, g0.vignetting, "{what}");
        assert!(
            rel(g1.relative_irradiance, g0.relative_irradiance) < 1e-12,
            "{what}"
        );
    }
}

#[test]
fn passive_stop_and_explicit_apertures_still_clip() {
    // The stop of the plate system is a bare plane in air (passive), yet it
    // bounds the beam; a dummy with a fixed semi-diameter clips too.
    let mut sys = plate(1.5, 10.0, 20.0, &[0.0], None);
    sys = insert_plane(&sys, 3, 10.0, true);
    sys.surfaces[4].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 3.0 };
    let m = model(sys, GhostSettings::default());
    assert!(matches!(m.clip(1), Clip::Radius { radius } if (radius - 5.0).abs() < 1e-9));
    assert_eq!(m.clip(4), Clip::Surface);
    assert_eq!(m.clip_radius(4), Some(3.0));
}

/// Plate with a sensor of half-size 2 × 1 mm and a reflecting sensor.
fn sensor_plate(rs: f64, clip: ClipMode) -> GhostModel {
    let mut sys = plate(1.5, 10.0, 20.0, &[0.0], None);
    sys.sensor = Some(Sensor {
        pixels_x: 40,
        pixels_y: 20,
        pitch_x: 0.1,
        pitch_y: 0.1,
        max_cra_deg: None,
        spectral_response: Vec::new(),
    });
    model(
        sys,
        GhostSettings {
            sensor_reflectance: Some(rs),
            clip,
            ..Default::default()
        },
    )
}

#[test]
fn sensor_reflections_only_inside_the_sensor_rectangle() {
    let (rs, n_grid) = (0.5, 20usize);
    let m = sensor_plate(rs, ClipMode::ClearApertures { margin_mm: 0.0 });
    let s = IrradianceSettings {
        paths: Some(vec![GhostPath::new(vec![4, 3])]),
        pupil_samples: n_grid,
        overfill: 1.0,
        map_bins: [8, 8],
        ..Default::default()
    };
    let res = ghost_irradiance(&m, &s, &JobContext::new()).unwrap();
    // Normal incidence: the ghost is limited to the 4 × 2 mm sensor,
    // while the primary covers the radius-5 pupil (including off-map flux).
    let (p, g) = (&res.primary.fields[0], &res.ghosts[0].fields[0]);
    let r = r_normal(1.0, 1.5);
    let expect = rs * r * 8.0 / (25.0 * std::f64::consts::PI);
    assert!(rel(g.flux / p.flux, expect) < 2e-3, "{}", g.flux / p.flux);
}

#[test]
fn fixed_apertures_bound_an_auto_stop() {
    // An overfilled grid must not let light around an unbounded stop: the
    // primary flux is T² × (grid fraction inside the entrance pupil).
    let n_grid = 30usize;
    let k = 1.5;
    let m = sensor_plate(0.0, ClipMode::FixedApertures);
    assert!(matches!(m.clip(1), Clip::Radius { .. }));
    assert_eq!(m.clip(2), Clip::None);
    let s = IrradianceSettings {
        paths: Some(vec![GhostPath::new(vec![3, 2])]),
        pupil_samples: n_grid,
        overfill: k,
        map_bins: [8, 8],
        ..Default::default()
    };
    let res = ghost_irradiance(&m, &s, &JobContext::new()).unwrap();
    let r = r_normal(1.0, 1.5);
    let t2 = (1.0 - r) * (1.0 - r);
    let p = &res.primary.fields[0];
    assert!(rel(p.flux, t2) < 5e-4, "{} vs {t2}", p.flux);
    assert!(p.flux < 1.0);
}

#[test]
fn coverage_records_the_apertures() {
    let m = sensor_plate(0.1, ClipMode::ClearApertures { margin_mm: 0.5 });
    let rep = paraxial_ghosts(&m, &ParaxialGhostSettings::default()).unwrap();
    assert_eq!(rep.coverage.apertures, m.clips());
    assert!(rep.coverage.same_apertures(m.clips()));
    let other = sensor_plate(0.1, ClipMode::None);
    assert!(!rep.coverage.same_apertures(other.clips()));
    // The margin widens the clear apertures of the plate faces only.
    let tight = sensor_plate(0.1, ClipMode::ClearApertures { margin_mm: 0.0 });
    for i in [2, 3] {
        let (Clip::Radius { radius: a }, Clip::Radius { radius: b }) = (m.clip(i), tight.clip(i))
        else {
            panic!("surface {i} should clip at a radius");
        };
        assert!((a - b - 0.5).abs() < 1e-12);
    }
    assert_eq!(m.clip(1), tight.clip(1));
}
