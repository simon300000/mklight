//! Frames, placements, reversibility, aiming, clipping and TIR.

mod common;

use common::*;
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, MaterialRef, Placement, RayAiming,
    ResolvedSystem, SemiDiameter, SurfaceShape, Vec3, samples,
};
use mklight_trace::interact::{reflect, refract};
use mklight_trace::surface::intersect;
use mklight_trace::trace::to_global;
use mklight_trace::{
    FailReason, FieldPoint, IntersectSettings, PupilSampling, Ray, SemiDiameterSettings,
    TraceOptions, TraceStatus, TracerSettings, fix_semi_diameters, trace_ray, trace_ray_recorded,
};

/// Independent backwards trace from the image to the starting frame, built
/// only from the public intersection and interaction primitives and the
/// inverse relative transforms.
fn reverse_trace(sys: &ResolvedSystem, wl: usize, at_image: Ray) -> Ray {
    let settings = IntersectSettings::default();
    let mut ray = Ray::new(at_image.pos, -at_image.dir);
    let n = sys.num_surfaces();
    for i in (1..n - 1).rev() {
        let tn = sys.surfaces[i].to_next.expect("relative transform");
        ray = Ray::new(tn.to_global_point(ray.pos), tn.to_global_dir(ray.dir));
        let s = &sys.surfaces[i];
        let h = intersect(&s.shape, ray.pos, ray.dir, &settings).expect("reverse hit");
        ray.pos = h.point;
        if matches!(s.shape, SurfaceShape::CoordinateBreak) {
            continue;
        }
        ray.dir = match s.interaction {
            Interaction::Refract => {
                refract(ray.dir, h.normal, s.n_after[wl], sys.n_before(i, wl))
                    .expect("no TIR")
                    .0
            }
            Interaction::Reflect => reflect(ray.dir, h.normal).0,
            Interaction::Dummy => ray.dir,
        };
    }
    // Into the starting frame and onto the object plane.
    let tn = sys.surfaces[0].to_next.expect("finite object");
    ray = Ray::new(tn.to_global_point(ray.pos), tn.to_global_dir(ray.dir));
    let h = intersect(&sys.surfaces[0].shape, ray.pos, ray.dir, &settings).unwrap();
    Ray::new(h.point, -ray.dir)
}

/// The relay with a tilted, decentered element and a tilted fold-free
/// coordinate break, so non-axial transfers are exercised.
fn tilted_relay() -> mklight_core::OpticalSystem {
    let mut s = samples::finite_relay();
    s.surfaces[5].placement = Some(Placement {
        tilt_x: 3.0,
        decenter_y: 0.4,
        tilt_y: -1.5,
        restore_after: true,
        ..Default::default()
    });
    s.surfaces[4].placement = Some(Placement {
        tilt_x: -2.0,
        ..Default::default()
    });
    s
}

#[test]
fn trace_is_reversible() {
    for sys in [samples::finite_relay(), tilted_relay()] {
        let t = tracer(sys);
        for w in 0..t.num_wavelengths() {
            for (fi, f) in t.fields().iter().enumerate() {
                for p in (PupilSampling::Hexapolar { rings: 3 }).points() {
                    let g = t.generate(f, w, p[0], p[1]).unwrap();
                    let r = trace_ray(t.system(), &g.ray, w, &TraceOptions::default(), None);
                    assert!(r.status.is_ok(), "field {fi}: {:?}", r.status);
                    let back = reverse_trace(t.system(), w, r.ray);
                    assert!(
                        back.pos.max_abs_diff(g.ray.pos) < 1e-10,
                        "{:?} vs {:?}",
                        back.pos,
                        g.ray.pos
                    );
                    assert!(back.dir.max_abs_diff(g.ray.dir) < 1e-12);
                }
            }
        }
    }
}

#[test]
fn fold_mirror_sends_axis_along_y() {
    let t = tracer(samples::newtonian_telescope());
    let sys = t.system();
    let w = sys.primary_wavelength;
    let mut rec = Vec::new();
    let r = t.trace_with(
        t.field(0),
        w,
        0.0,
        0.0,
        &TraceOptions::default(),
        Some(&mut rec),
    );
    assert!(r.status.is_ok());
    // After the primary the axial ray travels −z; after the fold it travels
    // along global −y, which is local +z of the image frame.
    let after_primary = to_global(sys, 1, &Ray::new(rec[1].point, rec[1].dir));
    assert!(after_primary.dir.max_abs_diff(-Vec3::Z) < 1e-15);
    let after_fold = to_global(sys, 3, &Ray::new(rec[3].point, rec[3].dir));
    assert!(
        after_fold.dir.max_abs_diff(-Vec3::Y) < 1e-15,
        "{:?}",
        after_fold.dir
    );
    assert!(r.ray.dir.max_abs_diff(Vec3::Z) < 1e-15);
    assert!(r.ray.pos.x.abs() < 1e-12 && r.ray.pos.y.abs() < 1e-12);
    // The fold sits 400 mm in front of the primary, the focus 100 mm along −y.
    let image_global = to_global(sys, 5, &r.ray).pos;
    assert!(image_global.max_abs_diff(Vec3::new(0.0, -100.0, -400.0)) < 1e-12);
    // Paraxially the fold is invisible.
    assert!((t.paraxial().primary().efl - 500.0).abs() < 1e-9);
    assert!(!t.paraxial().warnings.is_empty());
}

#[test]
fn restore_after_keeps_downstream_axis() {
    // A tilted, decentered dummy plane with restore_after changes nothing for
    // the surfaces after it.
    let base = samples::cooke_triplet();
    let mut tilted = base.clone();
    let id = tilted.next_surface_id();
    let mut dummy = surf(id.0, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    dummy.interaction = Interaction::Dummy;
    dummy.placement = Some(Placement {
        tilt_x: 25.0,
        tilt_y: -10.0,
        decenter_x: 1.0,
        restore_after: true,
        ..Default::default()
    });
    // Insert after L1 rear: split its 6.00755 gap as 0 + 6.00755.
    tilted.surfaces.insert(3, dummy);
    tilted.surfaces[3].thickness = 6.00755;
    tilted.surfaces[2].thickness = 0.0;
    tilted.stop = 5;
    let (a, b) = (tracer(base), tracer(tilted));
    for (fa, fb) in a.fields().iter().zip(b.fields()) {
        for p in [[0.0, 0.0], [0.3, -0.9], [-1.0, 0.0]] {
            let ra = a.trace(fa, 1, p[0], p[1], &TraceOptions::default());
            let rb = b.trace(fb, 1, p[0], p[1], &TraceOptions::default());
            assert!(ra.ray.pos.max_abs_diff(rb.ray.pos) < 1e-12);
            assert!(ra.ray.dir.max_abs_diff(rb.ray.dir) < 1e-14);
            assert!((ra.opl.diff(&rb.opl)).abs() < 1e-11);
        }
    }
}

#[test]
fn decentered_lens_equals_shifted_ray() {
    // Decentering both surfaces of a lens by dy (each with restore_after) is
    // the same as shifting the ray by −dy through the centred lens and the
    // result back by +dy.
    let dy = 1.3;
    let body = |dec: bool| {
        let mut a = surf(
            1,
            SurfaceShape::sphere(40.0),
            6.0,
            MaterialRef::glass("N-BK7"),
        );
        let mut b = surf(2, SurfaceShape::sphere(-60.0), 60.0, MaterialRef::Air);
        if dec {
            for s in [&mut a, &mut b] {
                s.placement = Some(Placement {
                    decenter_y: dy,
                    restore_after: true,
                    ..Default::default()
                });
            }
        }
        vec![a, b]
    };
    let rd = resolve(infinite_system(10.0, body(true), 3));
    let rc = resolve(infinite_system(10.0, body(false), 3));
    let dir = Vec3::new(0.02, -0.05, 1.0).normalized();
    for y in [-4.0, 0.0, 2.5] {
        let s1 = Ray::new(Vec3::new(0.5, y, -3.0), dir);
        let s2 = Ray::new(Vec3::new(0.5, y - dy, -3.0), dir);
        let r1 = trace_ray(&rd, &s1, 0, &TraceOptions::default(), None);
        let r2 = trace_ray(&rc, &s2, 0, &TraceOptions::default(), None);
        assert!((r1.ray.pos.y - (r2.ray.pos.y + dy)).abs() < 1e-12);
        assert!((r1.ray.pos.x - r2.ray.pos.x).abs() < 1e-12);
        assert!(r1.ray.dir.max_abs_diff(r2.ray.dir) < 1e-15);
    }
}

#[test]
fn total_internal_reflection_is_reported() {
    // Light enters a glass block normally and meets an internal face tilted
    // 45°: beyond the critical angle (41.1° for n = 1.52).
    let mut hyp = surf(2, SurfaceShape::Plane, 10.0, MaterialRef::Air);
    hyp.placement = Some(Placement {
        tilt_x: 45.0,
        ..Default::default()
    });
    let body = vec![
        surf(1, SurfaceShape::Plane, 10.0, MaterialRef::glass("N-BK7")),
        hyp,
    ];
    let t = tracer(infinite_system(4.0, body, 3));
    let r = t.trace(t.field(0), 0, 0.0, 0.5, &TraceOptions::default());
    assert_eq!(
        r.status,
        TraceStatus::TotalInternalReflection { surface: 2 }
    );
}

#[test]
fn fixed_apertures_clip_and_can_be_disabled() {
    let mut s = samples::singlet();
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 8.0 };
    let t = tracer(s);
    let f = t.field(0);
    let on = TraceOptions::default();
    assert!(t.trace(f, 1, 0.0, 0.7, &on).status.is_ok());
    assert_eq!(
        t.trace(f, 1, 0.0, 1.0, &on).status,
        TraceStatus::Clipped { surface: 2 }
    );
    assert!(
        t.trace(f, 1, 0.0, 1.0, &TraceOptions::unclipped())
            .status
            .is_ok()
    );
    // Explicit apertures override the semi-diameter.
    let mut s = samples::singlet();
    s.surfaces[1].aperture = Some(mklight_core::ApertureShape::Annular {
        inner_radius: 3.0,
        outer_radius: 20.0,
    });
    let t = tracer(s);
    assert_eq!(
        t.trace(t.field(0), 1, 0.0, 0.0, &on).status,
        TraceStatus::Clipped { surface: 1 }
    );
}

#[test]
fn real_aiming_converges_with_strongly_tilted_stop() {
    // Double Gauss with its stop tilted 20° and decentered: real aiming must
    // put every ray on its target in the stop's own frame.
    let mut s = samples::double_gauss();
    s.surfaces[6].placement = Some(Placement {
        tilt_x: 20.0,
        tilt_y: 8.0,
        decenter_y: 0.8,
        restore_after: true,
        ..Default::default()
    });
    s.ray_aiming = RayAiming::Real { tolerance: 1e-12 };
    let t = tracer(s);
    let stop = t.system().stop;
    let r_stop = t.pupil().r_stop;
    for f in t.fields() {
        for w in 0..t.num_wavelengths() {
            assert!(f.aims[w].converged);
            for p in (PupilSampling::Hexapolar { rings: 4 }).points() {
                let g = t.generate(f, w, p[0], p[1]).expect("aim converges");
                assert!(g.report.converged && g.report.residual <= 1e-12);
                let r = trace_ray(
                    t.system(),
                    &g.ray,
                    w,
                    &TraceOptions::unclipped().to_surface(stop),
                    None,
                );
                assert!((r.ray.pos.x - p[0] * r_stop).abs() < 1e-10 * r_stop);
                assert!((r.ray.pos.y - p[1] * r_stop).abs() < 1e-10 * r_stop);
            }
        }
    }
    // With aiming off the paraxial pupil misses the tilted stop's targets.
    let off = tracer_with(
        {
            let mut s = samples::double_gauss();
            s.surfaces[6].placement = Some(Placement {
                tilt_x: 20.0,
                tilt_y: 8.0,
                decenter_y: 0.8,
                restore_after: true,
                ..Default::default()
            });
            s
        },
        TracerSettings::default(),
    );
    let g = off.generate(off.field(2), 1, 0.0, 1.0).unwrap();
    let r = trace_ray(
        off.system(),
        &g.ray,
        1,
        &TraceOptions::unclipped().to_surface(stop),
        None,
    );
    assert!((r.ray.pos.y - r_stop).abs() > 1e-3);
}

#[test]
fn real_image_height_fields_land_on_target() {
    let t = tracer(samples::petzval_asphere());
    let w = t.system().primary_wavelength;
    for (i, h) in [0.0, 4.0, 6.0].into_iter().enumerate() {
        let c = t.chief(t.field(i), w);
        assert!(c.status.is_ok());
        assert!((c.ray.pos.y - h).abs() < 1e-9, "field {i}: {}", c.ray.pos.y);
        assert!(c.ray.pos.x.abs() < 1e-12);
    }
}

#[test]
fn paraxial_image_height_fields() {
    let mut s = samples::cooke_triplet();
    s.fields = FieldSet {
        kind: FieldKind::ParaxialImageHeight,
        fields: vec![
            Field::new(0.0, 0.0),
            Field::new(0.0, 10.0),
            Field::new(-3.0, 15.0),
        ],
    };
    let t = tracer(s);
    let fo = t.paraxial().primary();
    assert!((fo.paraxial_image_height - 15.0f64.hypot(3.0)).abs() < 1e-9);
    // A tiny version of field 1 lands at its paraxial height (real ≈ paraxial).
    let small = t.prepare_field(FieldPoint::normalized(0.0, 1e-6)).unwrap();
    let c = t.chief(&small, t.system().primary_wavelength);
    let hmax = 15.0f64.hypot(3.0);
    assert!((c.ray.pos.y - 1e-6 * hmax).abs() < 1e-12);
    // Distortion at full field is small but non-zero.
    let c = t.chief(t.field(1), t.system().primary_wavelength);
    assert!((c.ray.pos.y - 10.0).abs() < 0.2);
}

#[test]
fn finite_object_angle_and_na() {
    // Relay with angle fields at a finite object: the object point lies on the
    // chief line through the entrance-pupil centre. NA aperture: the real
    // axial marginal ray has exactly the requested NA.
    let t = tracer(samples::finite_relay());
    let w = t.system().primary_wavelength;
    let g = t.generate(t.field(0), w, 0.0, 1.0).unwrap();
    assert!((g.ray.dir.y.hypot(g.ray.dir.x) - 0.08).abs() < 1e-14);
    let mut s = samples::finite_relay();
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 2.0)],
    };
    let t = tracer(s);
    let c = t.generate(t.field(1), w, 0.0, 0.0).unwrap();
    assert!((c.ray.dir.y / c.ray.dir.z - 2f64.to_radians().tan()).abs() < 1e-14);
    let z_ep = t.pupil().z_ep;
    let obj_dist = t.system().surfaces[0].thickness;
    assert!((c.ray.pos.y + 2f64.to_radians().tan() * (z_ep + obj_dist)).abs() < 1e-12);
}

#[test]
fn automatic_semi_diameters() {
    // On-axis singlet: the semi-diameters are the marginal-ray heights.
    let mut s = samples::singlet();
    s.fields = FieldSet::default();
    let t = tracer(s.clone());
    let sd = t.semi_diameters(&SemiDiameterSettings::default());
    assert!((sd.values[1] - 10.0).abs() < 1e-12); // stop = EPD / 2
    let w_max = (0..3)
        .map(|w| {
            let (_, rec) = trace_ray_recorded(
                t.system(),
                &t.generate(t.field(0), w, 0.0, 1.0).unwrap().ray,
                w,
                &TraceOptions::default(),
            );
            rec[2].point.y.abs()
        })
        .fold(0.0, f64::max);
    assert!((sd.values[2] - w_max).abs() < 1e-9);
    assert_eq!(sd.failed_rays, 0);
    // Freezing them (with a margin) clips nothing.
    let fixed = fix_semi_diameters(&s, &sd, 0.01);
    let tf = tracer(fixed);
    for p in (PupilSampling::Ring { n: 32, radius: 1.0 }).points() {
        assert!(
            tf.trace(tf.field(0), 1, p[0], p[1], &TraceOptions::default())
                .status
                .is_ok()
        );
    }
    // A sample with several fields: every edge ray fits inside.
    let t = tracer(samples::double_gauss());
    let sd = t.semi_diameters(&SemiDiameterSettings::default());
    let tf = tracer(fix_semi_diameters(&samples::double_gauss(), &sd, 1e-9));
    for f in tf.fields() {
        for p in (PupilSampling::Ring { n: 64, radius: 1.0 }).points() {
            assert!(
                tf.trace(f, 0, p[0], p[1], &TraceOptions::default())
                    .status
                    .is_ok()
            );
        }
    }
}

#[test]
fn aim_failure_is_reported_not_silent() {
    // A pupil point far outside the stop cannot be reached by aiming at a stop
    // that every ray misses: report AimNotConverged.
    let mut s = samples::singlet();
    s.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    s.surfaces[2].shape = SurfaceShape::sphere(-12.0);
    s.stop = 2;
    s.aperture = ApertureSpec::FloatByStopSize;
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 11.9 };
    let t = tracer(s);
    let r = t.trace(t.field(0), 1, 0.0, 3.0, &TraceOptions::unclipped());
    assert_eq!(
        r.status,
        TraceStatus::Failed {
            surface: 2,
            reason: FailReason::AimNotConverged
        }
    );
}
