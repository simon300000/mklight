//! The even-asphere slope designs of the double-float trace (work-split key
//! `trace_slope`): with `df` (the default) the hit refinement and the normal
//! of an even asphere take the polynomial slope in double-float, and the
//! double-float trace follows the f64 reference to its own precision; with
//! `f32` the slope is the f32 polynomial's and the normal carries its
//! rounding. Measured on the CPU-f32 specification of the kernel, through
//! two strong even aspheres at steep incidence.
use mklight_core::{SurfaceShape, split};
use mklight_gpu::{
    Df32, Ieee, PackOptions, PackedSystem,
    consts::*,
    layout::df_slope,
    testutil,
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
/// grid, with the system packed under the current `trace_slope` selection.
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
    // The fixture's surfaces are even aspheres: `df` marks them.
    assert!(df_slope().unwrap());
    let t = testutil::tracer(testutil::asphere_heavy());
    let p = PackedSystem::pack_all(t.system()).unwrap();
    assert!(
        p.surfaces
            .iter()
            .filter(|s| s.shape == MKL_SHAPE_EVEN_ASPHERE)
            .all(|s| s.flags & MKL_SURF_DF_SLOPE != 0)
    );
    assert!(matches!(
        t.system().surfaces[1].shape,
        SurfaceShape::EvenAsphere { .. }
    ));
    let (df_pos, df_dir, n) = errors();
    split::set("trace_slope", Some("f32"));
    let flags = PackedSystem::pack_all(t.system()).unwrap().surfaces[1].flags;
    let (f32_pos, f32_dir, m) = errors();
    split::set("trace_slope", Some("nonsense"));
    assert!(PackedSystem::pack_all(t.system()).is_err());
    split::reset();
    assert_eq!(flags & MKL_SURF_DF_SLOPE, 0, "f32 leaves the flag off");
    assert_eq!(n, m);
    assert!(n > 100, "{n} rays");
    // Double-float slope: the trace keeps double-float precision.
    assert!(
        df_pos < 1e-11 && df_dir < 1e-12,
        "df: {df_pos:e} mm, {df_dir:e}"
    );
    // The f32 slope's normal error is orders of magnitude larger.
    assert!(
        f32_dir > 100.0 * df_dir.max(1e-15),
        "f32 {f32_dir:e} vs df {df_dir:e}"
    );
    println!("df {df_pos:e} mm {df_dir:e}; f32 {f32_pos:e} mm {f32_dir:e}");
}
