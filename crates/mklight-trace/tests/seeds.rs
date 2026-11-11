//! Anchor-seeded aiming (`AimSeeds`): a tracer of a slightly perturbed
//! system, seeded from an anchor tracer's recorded solves, agrees with the
//! unseeded tracer far inside the aiming tolerance, needs fewer Newton
//! iterations, and still generates rays one by one bitwise as in batches.

use crate::common;

use std::sync::Arc;

use common::resolve;
use mklight_core::model::RayAiming;
use mklight_core::{FieldKind, OpticalSystem, SurfaceShape, samples};
use mklight_trace::{ObjectSource, RayRequest, SequentialTracer, TracerSettings};

/// Double Gauss with real aiming and real-image-height fields (both
/// iterative solves the seeds start).
fn system() -> OpticalSystem {
    let mut sys = samples::double_gauss();
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    sys.fields.kind = FieldKind::RealImageHeight;
    for (f, h) in sys.fields.fields.iter_mut().zip([0.0, 8.0, 12.0]) {
        f.y = h;
    }
    sys
}

/// `sys` with a finite-difference-sized change before and after the stop.
fn perturbed(mut sys: OpticalSystem) -> OpticalSystem {
    let SurfaceShape::Conic { curvature, .. } = &mut sys.surfaces[3].shape else {
        panic!("spherical surface")
    };
    *curvature *= 1.0 + 3e-8;
    sys.surfaces[8].thickness += 2e-7;
    sys
}

fn pupil() -> Vec<[f64; 2]> {
    let mut p = Vec::new();
    for r in [0.3, 0.7, 1.0] {
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::TAU / 8.0;
            p.push([r * a.cos(), r * a.sin()]);
        }
    }
    p
}

fn requests<'a>(t: &'a SequentialTracer, pupil: &[[f64; 2]]) -> Vec<RayRequest<'a>> {
    let mut r = Vec::new();
    for f in t.fields() {
        for w in 0..t.num_wavelengths() {
            for &p in pupil {
                r.push((f, w, p));
            }
        }
    }
    r
}

#[test]
fn seeded_aims_agree_with_unseeded_and_take_fewer_iterations() {
    let settings = TracerSettings::default();
    let mut anchor =
        SequentialTracer::with_settings(Arc::new(resolve(system())), settings).unwrap();
    let seeds = anchor.record_seeds();
    let pupil = pupil();
    let traced = anchor.generate_many(&requests(&anchor, &pupil));
    assert!(traced.iter().all(|g| g.is_ok()));
    assert_eq!(seeds.rays(), traced.len());

    let near = Arc::new(resolve(perturbed(system())));
    let cold = SequentialTracer::with_settings(near.clone(), settings).unwrap();
    let seeded = SequentialTracer::with_seeds(near, settings, seeds).unwrap();
    // Field sources and chief aims: the same solutions.
    for (c, s) in cold.fields().iter().zip(seeded.fields()) {
        assert!(c.is_valid() && s.is_valid());
        let (ObjectSource::Direction { dir: a }, ObjectSource::Direction { dir: b }) =
            (c.source, s.source)
        else {
            panic!("object at infinity")
        };
        assert!((a - b).norm() <= 1e-12, "source {a:?} vs {b:?}");
        for (a, b) in c.aims.iter().zip(&s.aims) {
            assert!(a.converged && b.converged);
            for k in 0..2 {
                assert!((a.chief[k] - b.chief[k]).abs() <= 1e-11, "chief aim");
            }
        }
    }
    // Pupil rays: both meet the aiming tolerance; the seeded solve resolves
    // the change far more tightly, with fewer iterations.
    let rc = cold.generate_many(&requests(&cold, &pupil));
    let rs = seeded.generate_many(&requests(&seeded, &pupil));
    let (mut it_c, mut it_s) = (0, 0);
    let tol = 1e-10 * cold.pupil().r_stop;
    for (c, s) in rc.iter().zip(&rs) {
        let (c, s) = (c.as_ref().unwrap(), s.as_ref().unwrap());
        assert!(c.report.residual * cold.pupil().r_stop <= tol);
        assert!(s.report.residual * cold.pupil().r_stop <= tol);
        // Aim parameters agree to the cold solve's tolerance (mm of aim
        // per mm of stop is of order one here).
        for k in 0..2 {
            assert!(
                (c.aim[k] - s.aim[k]).abs() <= 10.0 * tol,
                "aim {:?} vs {:?}",
                c.aim,
                s.aim
            );
        }
        it_c += c.report.iterations;
        it_s += s.report.iterations;
    }
    assert!(
        it_s < it_c,
        "seeded {it_s} vs unseeded {it_c} Newton iterations"
    );
    // One by one equals the batch, bit for bit, on the seeded tracer too.
    for (&(f, w, [px, py]), b) in requests(&seeded, &pupil).iter().zip(&rs) {
        let one = seeded.generate(f, w, px, py).unwrap();
        let b = b.as_ref().unwrap();
        assert_eq!(one.aim.map(f64::to_bits), b.aim.map(f64::to_bits));
        assert_eq!(one.report.iterations, b.report.iterations);
    }
}

#[test]
fn unrelated_seeds_fall_back_to_unseeded_solves() {
    // Seeds of a very different system (curvature changed by 10 %): every
    // solve still converges to the unseeded answer.
    let settings = TracerSettings::default();
    let mut far = system();
    let SurfaceShape::Conic { curvature, .. } = &mut far.surfaces[3].shape else {
        panic!("spherical surface")
    };
    *curvature *= 1.1;
    let mut anchor = SequentialTracer::with_settings(Arc::new(resolve(far)), settings).unwrap();
    let seeds = anchor.record_seeds();
    let pupil = pupil();
    anchor.generate_many(&requests(&anchor, &pupil));
    let sys = Arc::new(resolve(system()));
    let cold = SequentialTracer::with_settings(sys.clone(), settings).unwrap();
    let seeded = SequentialTracer::with_seeds(sys, settings, seeds).unwrap();
    let tol = 1e-10 * cold.pupil().r_stop;
    let rc = cold.generate_many(&requests(&cold, &pupil));
    let rs = seeded.generate_many(&requests(&seeded, &pupil));
    for (c, s) in rc.iter().zip(&rs) {
        let (c, s) = (c.as_ref().unwrap(), s.as_ref().unwrap());
        for k in 0..2 {
            assert!((c.aim[k] - s.aim[k]).abs() <= 10.0 * tol);
        }
    }
}
