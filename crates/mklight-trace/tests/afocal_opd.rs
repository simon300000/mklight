//! OPD in an afocal image space: the plane reference normal to the chief
//! ray through the exit-pupil point (`mklight_trace::opd`, "Afocal image
//! space").

use crate::common;

use common::*;
use mklight_core::{Interaction, MaterialRef, SurfaceShape};
use mklight_trace::{SequentialTracer, TraceOptions, trace_ray};

/// Two ideal lenses `f1`, `f2` (stop on the first) separated by
/// `f1 + f2 + delta`; image 100 mm behind the second. Angle fields `ys`.
fn telescope(f1: f64, f2: f64, delta: f64, epd: f64, ys: &[f64]) -> SequentialTracer {
    let mut s = infinite_system(
        epd,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial { focal_length: f1 },
                f1 + f2 + delta,
                MaterialRef::Air,
            ),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: f2 },
                100.0,
                MaterialRef::Air,
            ),
        ],
        3,
    );
    s.fields = angle_fields(ys);
    tracer(s)
}

/// The confocal paraboloid beam compressor (5×, EPD 50 mm).
fn compressor() -> SequentialTracer {
    let mirror = |id, c: f64, t| {
        let mut m = surf(
            id,
            SurfaceShape::Conic {
                curvature: c,
                conic: -1.0,
            },
            t,
            MaterialRef::Air,
        );
        m.interaction = Interaction::Reflect;
        m
    };
    tracer(infinite_system(
        50.0,
        vec![mirror(1, -1.0 / 1000.0, -400.0), mirror(2, -1.0 / 200.0, 200.0)],
        3,
    ))
}

/// The afocal systems are recognised and their collimated beams have no
/// OPD on the plane reference (the sphere about the chief ray's image
/// point gave tens of waves of defocus).
#[test]
fn collimated_beams_have_zero_opd() {
    let opts = TraceOptions::default();
    for (label, t) in [
        ("compressor", compressor()),
        ("keplerian", telescope(20.0, 200.0, 0.0, 5.0, &[0.0, 1.0])),
        ("galilean", telescope(-20.0, 100.0, 0.0, 4.0, &[0.0, 1.0])),
    ] {
        assert!(t.afocal_image_space(), "{label}");
        for (fi, field) in t.fields().iter().enumerate() {
            let reference = t.opd_reference(field, 0).unwrap();
            assert!(reference.afocal);
            for (px, py) in [(0.0, 1.0), (0.0, -1.0), (0.6, 0.3), (-1.0, 0.0)] {
                let o = reference.opd_waves(&t.trace(field, 0, px, py, &opts)).unwrap();
                assert!(o.abs() < 1e-7, "{label} field {fi} ({px}, {py}): {o}");
            }
        }
    }
    // Focal systems keep the reference sphere.
    let focal = telescope(20.0, 200.0, 2.0, 5.0, &[0.0]);
    assert!(!focal.afocal_image_space());
    assert!(!focal.opd_reference(focal.field(0), 0).unwrap().afocal);
}

/// A slightly convergent output beam (second lens moved by `delta`) is an
/// exact spherical wave focusing `L` beyond the exit-pupil plane; on the
/// plane the OPD is `√(L² + h²) − L` at height `h` (positive: converging).
/// The system is not afocal itself, so the plane reference is selected
/// explicitly; the paraxial exit pupil fixes the plane.
#[test]
fn plane_reference_of_a_converging_beam_is_the_closed_form() {
    let (f1, f2, delta, epd) = (20.0, 200.0, 2.0, 5.0);
    let t = telescope(f1, f2, delta, epd, &[0.0]);
    let field = t.field(0);
    let mut reference = t.opd_reference(field, 0).unwrap();
    reference.afocal = true;
    // Exit pupil: the first lens imaged by the second.
    let d = f1 + f2 + delta;
    let z_xp = 1.0 / (1.0 / f2 - 1.0 / d);
    assert!((100.0 + reference.s_exit_pupil - z_xp).abs() < 1e-9);
    // Focus of the output beam behind the second lens.
    let s_img = 1.0 / (1.0 / f2 - 1.0 / (d - f1));
    let l = s_img - z_xp;
    let lam = D_LINE * 1e-3;
    let opts = TraceOptions::default();
    for (px, py) in [(0.0, 1.0), (0.0, -0.5), (0.6, 0.6), (-1.0, 0.0)] {
        let r = t.trace(field, 0, px, py, &opts);
        let rho = px.hypot(py);
        // Height at the second lens, then along the converging cone.
        let h2 = 0.5 * epd * rho * (d - f1) / f1;
        let h = h2 * (1.0 - z_xp / s_img);
        let exact = ((l * l + h * h).sqrt() - l) / lam;
        let o = reference.opd_waves(&r).unwrap();
        assert!(exact > 1.0 || rho < 0.9);
        assert!(
            (o - exact).abs() < 1e-6 * exact.max(1.0),
            "({px}, {py}): {o} vs {exact}"
        );
    }
}

/// The OPD tangent of the plane reference against central differences.
#[test]
fn plane_reference_tangent_matches_central_differences() {
    let t = telescope(20.0, 200.0, 2.0, 5.0, &[0.0, 1.0]);
    let plan = t.tangent_plan();
    let image = t.system().image_index();
    for field in t.fields() {
        let mut reference = t.opd_reference(field, 0).unwrap();
        reference.afocal = true;
        let g = t.generate(field, 0, 0.4, -0.55).unwrap();
        let (ray, tangent) = t.start_tangent(&field.source, g.aim).unwrap();
        let mut rec = Vec::new();
        let r = trace_ray(
            t.system(),
            &ray,
            0,
            &TraceOptions::unclipped(),
            Some(&mut rec),
        );
        let ti = plan
            .differentiate(t.system(), 0, &tangent, &rec, image)
            .unwrap();
        let analytic = reference.opd_tangent(&r, &ti).unwrap();
        let opd = |a: [f64; 2]| {
            let ray = t.start_ray(&field.source, a).unwrap();
            let r = trace_ray(t.system(), &ray, 0, &TraceOptions::unclipped(), None);
            reference.opd_waves(&r).unwrap()
        };
        let h = 1e-4 * t.pupil().r_ep;
        for k in 0..2 {
            let (mut ap, mut am) = (g.aim, g.aim);
            ap[k] += h;
            am[k] -= h;
            let fd = (opd(ap) - opd(am)) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-5 * analytic[k].abs().max(1.0),
                "component {k}: {fd} vs {}",
                analytic[k]
            );
        }
    }
}
