//! Cancellation latency of device aiming: with bounded commands a cancel
//! issued during pupil aiming reaches the terminal state within a few
//! commands.
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;
use mklight_core::{JobContext, MkError, RayAiming};
use mklight_gpu::*;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::PupilSampling;
use std::time::Instant;
use support::*;

#[test]
fn cancel_during_aiming_is_prompt() {
    let gpu = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
    })
    .unwrap()
    .expect("requires host Metal hardware");
    let mut model = mklight_core::samples::double_gauss();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let field = t.fields().len() - 1;
    // Enough rays for many time-calibrated tiles of pupil aiming (a plan of
    // a few hundred milliseconds).
    let points = (PupilSampling::Grid { n: 1400 }).points();
    let (p, h) = make_plan(&gpu, &t, field, 0, &points, false, 60, false);
    let full = Instant::now();
    gpu.submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let full = full.elapsed();
    assert!(full.as_secs_f64() > 0.1, "fixture too light: {full:?}");
    let mut worst = 0.;
    for _ in 0..3 {
        let ctx = JobContext::new();
        let pending = gpu.submit_plan(p.clone(), &ctx).unwrap();
        let start = Instant::now();
        while ctx.progress.fraction() < 0.3 && !pending.is_ready() {
            std::thread::yield_now();
            assert!(start.elapsed().as_secs() < 60);
        }
        assert!(
            !pending.is_ready(),
            "fixture must cancel during device work"
        );
        let cancel = Instant::now();
        pending.cancel();
        assert!(matches!(pending.wait(), Err(MkError::Cancelled)));
        let latency = cancel.elapsed().as_secs_f64();
        worst = f64::max(worst, latency);
        eprintln!(
            "full run {:.1} ms, cancel to terminal {:.1} ms",
            full.as_secs_f64() * 1e3,
            latency * 1e3
        );
    }
    assert!(worst < 0.2, "cancellation took {worst:.3} s");
    gpu.release_systems(h).unwrap();
}
