//! The packet paths (`trace_rays`, `generate_many`, `trace_many`, the
//! chunked batches) are bitwise identical to the scalar reference
//! (`trace_ray`, `generate`, `trace`) on every sample, with paraxial and
//! real aiming, clipped and unclipped, including rays that miss, clip or
//! cannot be aimed.

use crate::common;

use common::*;
use mklight_core::model::ApertureShape;
use mklight_core::{OpticalSystem, RayAiming, samples};
use mklight_trace::{
    Grid, PupilSampling, RayRequest, SequentialTracer, TraceOptions, opd_grid, trace_grid,
    trace_ray, trace_ray_recorded, trace_rays, trace_rays_recorded,
};

/// Debug output is exact for f64 (shortest round-trip form), so equal
/// strings mean bitwise-equal values.
fn same<T: std::fmt::Debug>(a: &T, b: &T, what: &str) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}");
}

/// Pupil points inside, on and outside the unit disk.
fn points() -> Vec<[f64; 2]> {
    let mut p = PupilSampling::Hexapolar { rings: 5 }.points();
    p.extend([[1.0, 0.0], [0.0, -1.0], [0.9, 0.9], [-1.4, 0.2], [0.0, 2.5]]);
    p
}

fn variants() -> Vec<(String, OpticalSystem)> {
    let mut out = Vec::new();
    for (name, sys) in samples::all() {
        out.push((name.to_string(), sys.clone()));
        let mut real = sys.clone();
        real.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        out.push((format!("{name} (real aiming)"), real));
    }
    // Clear apertures that clip oblique beams, with real aiming.
    let mut dg = samples::double_gauss();
    dg.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    for (surface, radius) in [(1, 11.), (11, 11.)] {
        dg.surfaces[surface].aperture = Some(ApertureShape::Circular { radius });
    }
    out.push(("double gauss (clipped, real aiming)".into(), dg));
    out
}

fn check(name: &str, t: &SequentialTracer) {
    let pts = points();
    let opts = [
        t.options(),
        TraceOptions::unclipped(),
        t.options().to_surface(t.system().stop),
    ];
    let mut requests: Vec<RayRequest> = Vec::new();
    for f in t.fields() {
        for w in 0..t.num_wavelengths() {
            for &p in &pts {
                requests.push((f, w, p));
            }
        }
    }
    let generated: Vec<_> = requests
        .iter()
        .map(|&(f, w, p)| t.generate(f, w, p[0], p[1]))
        .collect();
    for chunk in [1, 7, 64, requests.len()] {
        let many: Vec<_> = requests
            .chunks(chunk)
            .flat_map(|c| t.generate_many(c))
            .collect();
        same(&generated, &many, &format!("{name}: generate_many/{chunk}"));
    }
    for o in &opts {
        let single: Vec<_> = requests
            .iter()
            .map(|&(f, w, p)| t.trace(f, w, p[0], p[1], o))
            .collect();
        for chunk in [5, 64] {
            let many: Vec<_> = requests
                .chunks(chunk)
                .flat_map(|c| t.trace_many(c, o))
                .collect();
            same(&single, &many, &format!("{name}: trace_many/{chunk} {o:?}"));
        }
        // The packet trace of generated starts, one wavelength per ray.
        let (starts, wls): (Vec<_>, Vec<_>) = generated
            .iter()
            .zip(&requests)
            .filter_map(|(g, r)| g.as_ref().ok().map(|g| (g.ray, r.1)))
            .unzip();
        let scalar: Vec<_> = starts
            .iter()
            .zip(&wls)
            .map(|(s, &w)| trace_ray(t.system(), s, w, o, None))
            .collect();
        same(
            &scalar,
            &trace_rays(t.system(), &starts, &wls, o),
            &format!("{name}: trace_rays {o:?}"),
        );
        // With per-surface records (the vectors start dirty on purpose).
        let recorded: Vec<_> = starts
            .iter()
            .zip(&wls)
            .map(|(s, &w)| trace_ray_recorded(t.system(), s, w, o))
            .collect();
        let mut records = vec![vec![]; starts.len()];
        records
            .iter_mut()
            .for_each(|r| r.extend(recorded[0].1.iter().take(2)));
        let traced = trace_rays_recorded(t.system(), &starts, &wls, o, &mut records);
        let packed: Vec<_> = traced.into_iter().zip(records).collect();
        same(
            &recorded,
            &packed,
            &format!("{name}: trace_rays_recorded {o:?}"),
        );
    }
    // Batches: chunked packet traces against the per-ray results.
    let wls: Vec<usize> = (0..t.num_wavelengths()).collect();
    let grid = Grid {
        fields: t.fields(),
        wavelengths: &wls,
        pupil: &pts,
    };
    let o = t.options();
    let rays = trace_grid(t, &grid, &o);
    let opd = opd_grid(t, &grid, &o);
    for (i, (r, d)) in rays.iter().zip(&opd).enumerate() {
        let (f, w, p) = grid.split(i);
        let s = t.trace(&t.fields()[f], w, pts[p][0], pts[p][1], &o);
        same(
            &(s.status, s.ray.pos, s.opl),
            &(r.status, r.point, r.opl),
            name,
        );
        let reference = t.opd_reference(&t.fields()[f], w).ok();
        same(&reference.and_then(|c| c.opd_waves(&s)), d, name);
    }
}

#[test]
fn packet_paths_match_the_scalar_reference() {
    let mut failures = 0;
    for (name, sys) in variants() {
        let t = tracer(sys);
        check(&name, &t);
        failures += t
            .fields()
            .iter()
            .flat_map(|f| points().into_iter().map(move |p| (f, p)))
            .filter(|(f, p)| !t.trace(f, 0, p[0], p[1], &t.options()).status.is_ok())
            .count();
    }
    // Rays outside the pupil, clipped or unaimable are part of the check.
    assert!(failures > 100, "{failures}");
}

/// A steep ray crossing a surface that bulges toward it twice (first
/// farther from the axis): with a fixed semi-diameter containing both
/// crossings the trace takes the first along the ray, with an automatic one
/// the crossing nearer the vertex (SURFACES.md §8, conic root choice).
/// Scalar and packet traces agree bitwise, next to ordinary rays.
#[test]
fn steep_double_crossing_matches_in_packets() {
    use mklight_core::{MaterialRef, SemiDiameter, SurfaceShape, Vec3};
    use mklight_trace::Ray;
    let steep = Vec3::new(0.0, 0.999, 0.05).normalized();
    // Exact crossings of the sphere |x − C| = 20, C = (0, 0, −18).
    let start = Vec3::new(0.0, -30.0, 0.0);
    let m = start - Vec3::new(0.0, 0.0, -18.0);
    let (b, c) = (m.dot(steep), m.dot(m) - 400.0);
    let (t1, t2) = (-b - (b * b - c).sqrt(), -b + (b * b - c).sqrt());
    assert!(t1 > 0.0 && t2 > t1);
    // Records hold local points; surface 2's vertex is at z = 2.
    let vertex = Vec3::new(0.0, 0.0, 2.0);
    let (first, second) = (start + steep * t1 - vertex, start + steep * t2 - vertex);
    // The first crossing lies farther from the axis than the second.
    assert!(first.y < -5.0 && second.y > 3.0, "{first:?} {second:?}");

    for (sd, expected) in [
        (SemiDiameter::Auto, second),
        (SemiDiameter::Fixed { value: 8.0 }, first),
        (SemiDiameter::Fixed { value: 4.5 }, second),
    ] {
        let mut sphere = surf(
            2,
            SurfaceShape::sphere(-20.0),
            10.0,
            MaterialRef::glass("N-BK7"),
        );
        sphere.semi_diameter = sd;
        let sys = infinite_system(
            10.0,
            vec![surf(1, SurfaceShape::Plane, 2.0, MaterialRef::Air), sphere],
            3,
        );
        let t = tracer(sys);
        let o = TraceOptions::unclipped();
        let starts = vec![
            Ray::new(start, steep),
            Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::Z),
            Ray::new(
                Vec3::new(1.0, -3.0, 0.0),
                Vec3::new(0.0, 0.1, 1.0).normalized(),
            ),
        ];
        let wls = vec![0; starts.len()];
        let scalar: Vec<_> = starts
            .iter()
            .map(|s| trace_ray_recorded(t.system(), s, 0, &o))
            .collect();
        assert!(
            (scalar[0].1[2].point - expected).norm() < 1e-12,
            "{sd:?}: {:?}",
            scalar[0].1[2]
        );
        let mut records = vec![vec![]; starts.len()];
        let traced = trace_rays_recorded(t.system(), &starts, &wls, &o, &mut records);
        let packed: Vec<_> = traced.into_iter().zip(records).collect();
        same(
            &scalar,
            &packed,
            &format!("steep rays, {sd:?}: trace_rays_recorded"),
        );
        // Clipping does not change the choice: the hit inside the extent
        // passes the aperture.
        if matches!(sd, SemiDiameter::Fixed { .. }) {
            let clipped = trace_ray_recorded(t.system(), &starts[0], 0, &TraceOptions::default());
            assert_eq!(clipped.1[2].point, scalar[0].1[2].point, "{sd:?}");
        }
    }
}

/// A deep negative meniscus in front of the stop, a field out to 89° and
/// real aiming (a fisheye-like front group), with fixed or automatic
/// semi-diameters.
fn steep_meniscus(fixed: bool, fields: &[f64]) -> OpticalSystem {
    use mklight_core::{MaterialRef, SemiDiameter, SurfaceShape};
    let glass = |n| MaterialRef::Constant { n };
    let rows = [
        (SurfaceShape::sphere(78.0), 2.5, glass(1.6), 31.7),
        (SurfaceShape::sphere(15.9), 11.8, MaterialRef::Air, 15.85),
        (SurfaceShape::sphere(22.2), 2.5, glass(1.6), 13.9),
        (SurfaceShape::sphere(13.3), 7.5, MaterialRef::Air, 10.7),
        (SurfaceShape::Plane, 3.0, MaterialRef::Air, 3.0),
        (SurfaceShape::sphere(12.0), 4.0, glass(1.7), 6.0),
        (SurfaceShape::sphere(-12.0), 15.0, MaterialRef::Air, 6.0),
    ];
    let body = rows
        .into_iter()
        .enumerate()
        .map(|(i, (shape, t, m, sd))| {
            let mut s = surf(i as u64 + 1, shape, t, m);
            if fixed {
                s.semi_diameter = SemiDiameter::Fixed { value: sd };
            }
            s
        })
        .collect();
    let mut s = infinite_system(1.0, body, 8);
    s.stop = 5;
    s.fields = angle_fields(fields);
    s.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    s
}

/// Light from an object at infinity starts on a reference plane through the
/// first vertex, which a steep ray can cross between its two crossings of a
/// deep front surface. Both lie ahead of the light, which arrives from −∞,
/// so the first along the line is taken (here also the nearer one), not the
/// one ahead of the reference point (SURFACES.md §8). Scalar and packets
/// agree bitwise.
#[test]
fn steep_ray_from_infinity_takes_the_first_crossing() {
    use mklight_core::Vec3;
    use mklight_trace::Ray;
    let theta = 89f64.to_radians();
    let d = Vec3::new(0.0, theta.sin(), theta.cos());
    let start = Vec3::new(0.0, -theta.cos(), theta.sin()) * 6.16;
    // Crossings of the front sphere |x − (0, 0, 78)| = 78.
    let m = start - Vec3::new(0.0, 0.0, 78.0);
    let (b, c) = (m.dot(d), m.dot(m) - 78.0 * 78.0);
    let (t1, t2) = (-b - (b * b - c).sqrt(), -b + (b * b - c).sqrt());
    let (first, ahead) = (start + d * t1, start + d * t2);
    let r = |p: Vec3| p.x.hypot(p.y);
    // The reference point lies between them, both inside the 31.7 mm lens
    // and on the vertex sheet (z below the centre).
    assert!(t1 < 0.0 && t2 > 0.0, "{t1} {t2}");
    assert!(r(first) < r(ahead) && r(ahead) < 31.7 && ahead.z < 78.0);
    for fixed in [false, true] {
        let t = tracer(steep_meniscus(fixed, &[0.0]));
        for o in [TraceOptions::default(), TraceOptions::unclipped()] {
            let rays = vec![Ray::new(start, d), Ray::new(Vec3::ZERO, Vec3::Z)];
            let o = o.to_surface(1);
            let (res, rec) = trace_ray_recorded(t.system(), &rays[0], 0, &o);
            assert!(res.status.is_ok(), "{fixed}: {res:?}");
            assert!(
                (rec[1].point - first).norm() < 1e-9,
                "{fixed}: {:?}",
                rec[1]
            );
            let mut records = vec![vec![]; 2];
            let packed = trace_rays_recorded(t.system(), &rays, &[0], &o, &mut records);
            let scalar: Vec<_> = rays
                .iter()
                .map(|s| trace_ray_recorded(t.system(), s, 0, &o))
                .collect();
            let packed: Vec<_> = packed.into_iter().zip(records).collect();
            same(&scalar, &packed, "steep ray from infinity");
        }
    }
}

/// With fixed semi-diameters the steep meniscus aims and traces every grid
/// ray of an 89° field exactly as with automatic ones: the root choice
/// keeps the aim map of the nearer-vertex rule there.
#[test]
fn fixed_extents_keep_the_steep_field_aim_map() {
    let fields = [0.0, 45.0, 72.0, 89.0];
    let pupil = PupilSampling::Grid { n: 21 }.points();
    let run = |fixed: bool| {
        let t = tracer(steep_meniscus(fixed, &fields));
        let f = t.fields().to_vec();
        let grid = Grid {
            fields: &f,
            wavelengths: &[0],
            pupil: &pupil,
        };
        trace_grid(&t, &grid, &TraceOptions::unclipped())
    };
    let (auto, fixed) = (run(false), run(true));
    let ok = auto.iter().filter(|r| r.status.is_ok()).count();
    let steep = auto[3 * pupil.len()..]
        .iter()
        .filter(|r| r.status.is_ok())
        .count();
    eprintln!("ok {ok} of {}, 89°: {steep}", auto.len());
    assert!(ok > auto.len() / 2 && steep > 0, "{ok} {steep}");
    same(&auto, &fixed, "steep meniscus grid");
}
