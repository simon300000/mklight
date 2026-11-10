//! The even-asphere slope of the double-float trace: the hit refinement and
//! the normal of an even asphere take the polynomial slope in double-float,
//! so the double-float trace follows the f64 reference to its own precision.
//! Measured on the CPU-f32 specification of the kernel, through two strong
//! even aspheres at steep incidence.
use mklight_core::SurfaceShape;
use mklight_gpu::{
    Df32, Ieee, PackOptions, PackedSystem, testutil,
    trace_kernel::{self as tr, KRayIn},
};
use mklight_trace::{PupilSampling, TraceOptions, trace_ray};

fn ray_in(r: &mklight_trace::Ray) -> KRayIn {
    let d = |x: f64| Df32::from_f64(x);
    let p = [r.pos.x, r.pos.y, r.pos.z].map(d);
    let q = [r.dir.x, r.dir.y, r.dir.z].map(d);
    KRayIn {
        pos: p.map(|x| x.hi),
        pos_lo: p.map(|x| x.lo),
        dir: q.map(|x| x.hi),
        dir_lo: q.map(|x| x.lo),
        weight: 1.0,
        wavelength: 0,
        ..Default::default()
    }
}

/// The largest image position (mm) and direction errors of the
/// double-float trace against the f64 trace, over every field and a pupil
/// grid, and the number of rays compared.
fn errors() -> (f64, f64, usize) {
    let t = testutil::tracer(testutil::asphere_heavy());
    let wl = t.system().primary_wavelength;
    let packed = PackedSystem::pack(
        t.system(),
        &PackOptions {
            wavelengths: Some(vec![wl]),
            radius_hints: None,
        },
    )
    .unwrap();
    let image = t.system().image_index();
    let steps: Vec<u32> = (1..=image as u32).collect();
    let params = mklight_gpu::layout::TraceParams {
        num_steps: steps.len() as u32,
        ..mklight_gpu::aiming::kernel::newton_params()
    };
    let (mut pos, mut dir, mut n) = (0f64, 0f64, 0);
    for f in t.fields() {
        for p in (PupilSampling::Hexapolar { rings: 6 }).points() {
            let Ok(g) = t.generate(f, wl, p[0], p[1]) else {
                continue;
            };
            let r = trace_ray(t.system(), &g.ray, wl, &TraceOptions::unclipped(), None);
            let s = tr::trace_resident::<Ieee>(
                &packed.header,
                &packed.surfaces,
                &packed.coefficients,
                &steps,
                &params,
                &ray_in(&g.ray),
                |_, _| {},
            );
            if !r.status.is_ok() || s.status != 0 {
                continue;
            }
            let v = |hi: [f32; 3], lo: [f32; 3], k: usize| hi[k] as f64 + lo[k] as f64;
            let dp = [r.ray.pos.x, r.ray.pos.y, r.ray.pos.z];
            let dd = [r.ray.dir.x, r.ray.dir.y, r.ray.dir.z];
            for k in 0..3 {
                pos = pos.max((v(s.ray.pos, s.ray.pos_lo, k) - dp[k]).abs());
                dir = dir.max((v(s.ray.dir, s.ray.dir_lo, k) - dd[k]).abs());
            }
            n += 1;
        }
    }
    (pos, dir, n)
}

#[test]
fn double_float_slope_follows_the_f64_trace() {
    let t = testutil::tracer(testutil::asphere_heavy());
    assert!(matches!(
        t.system().surfaces[1].shape,
        SurfaceShape::EvenAsphere { .. }
    ));
    let (pos, dir, n) = errors();
    assert!(n > 100, "{n} rays");
    // An f32 slope leaves ~2e-8 mm and ~4e-9 here.
    assert!(pos < 1e-11 && dir < 1e-12, "{pos:e} mm, {dir:e}");
    println!("{pos:e} mm {dir:e}");
}
