//! OPD of an image-space telecentric system (exit pupil at infinity): the
//! `R → ∞` limit of the reference-sphere formula (CONVENTIONS §4).

use crate::common;

use common::*;
use mklight_core::{MaterialRef, SurfaceShape};
use mklight_trace::TraceOptions;

/// Stop in the front focal plane of an ideal lens of focal length `f`;
/// image `dz` beyond focus. Angle fields `ys` (degrees).
fn telecentric(f: f64, epd: f64, dz: f64, ys: &[f64]) -> mklight_trace::SequentialTracer {
    let mut s = infinite_system(
        epd,
        vec![
            surf(1, SurfaceShape::Plane, f, MaterialRef::Air),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: f },
                f + dz,
                MaterialRef::Air,
            ),
        ],
        3,
    );
    s.fields = angle_fields(ys);
    tracer(s)
}

/// Every ray of a perfect beam meets at the focus with equal path; on a
/// plane `dz` beyond it the `R → ∞` reference gives
/// `OPD = n·dz·(1 − cosθ)` (positive: image beyond focus, as for finite
/// `R`), with `θ` the ray's angle to the chief ray. The plane-at-the-image
/// reference used before gave `n·dz·(1 − 1/cosθ)`, the opposite sign.
#[test]
fn defocused_telecentric_opd_is_the_plane_wave_limit() {
    let (f, epd, dz) = (50.0, 10.0, 0.02);
    let t = telecentric(f, epd, dz, &[0.0, 5.0]);
    assert!(t.paraxial().primary().image_space_telecentric());
    let opts = TraceOptions::default();
    let lam = D_LINE * 1e-3;
    for fi in 0..2 {
        let field = t.field(fi);
        let reference = t.opd_reference(field, 0).unwrap();
        assert!(reference.s_exit_pupil.is_infinite());
        let dc = reference.chief_dir;
        for (px, py) in [(0.0, 1.0), (0.0, -0.7), (0.6, 0.3), (-1.0, 0.0)] {
            let r = t.trace(field, 0, px, py, &opts);
            let cos = r.ray.dir.dot(dc);
            let exact = dz * (1.0 - cos) / lam;
            let o = reference.opd_waves(&r).unwrap();
            assert!(exact > 0.01);
            assert!(
                (o - exact).abs() < 1e-8,
                "field {fi} ({px}, {py}): {o} vs {exact}"
            );
        }
    }
    // In focus the beam is perfect.
    let t0 = telecentric(f, epd, 0.0, &[0.0, 5.0]);
    for fi in 0..2 {
        let o = t0.opd(t0.field(fi), 0, 0.0, 1.0, &opts).unwrap();
        assert!(o.abs() < 1e-9, "field {fi}: {o}");
    }
}
