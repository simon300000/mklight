#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p4_common.rs"]
mod common;
use common::*;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{systems::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
fn backend(origin: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions {
        library: origin,
        ..Default::default()
    })
    .unwrap()
    .expect("P4 requires a host Metal device; a skipped test is not evidence");
    eprintln!("device={:?} precision={:?}", b.info(), b.precision());
    assert!(b.precision().probe.error_free_transforms());
    b
}
#[test]
fn host_metal_both_origins_all_samples() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let gpu = backend(origin);
        let cpu = CpuF32Backend::matching(&gpu.precision().probe);
        for (name, system) in mklight_core::samples::all() {
            let variables = basic_variables(&system);
            let rows = basic_rows(&system);
            let c = CompiledSystem::compile(
                Snapshot::new(system).unwrap(),
                testutil::library(),
                variables,
                None,
                &JobContext::new(),
            )
            .unwrap();
            let (want, _) = run(&cpu, &c, &rows, 64);
            let (got, stats) = run(&gpu, &c, &rows, 1);
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(a.words(), b.words(), "{name} buffer {i}");
            }
            let (again, _) = run(&gpu, &c, &rows, 64);
            for (a, b) in got.iter().zip(&again) {
                assert_eq!(a.words(), b.words());
            }
            for (i, row) in rows.iter().enumerate() {
                let rs = c.reference(row, &JobContext::new()).unwrap();
                let (ni, fo) = compare(&got, i, &rs);
                let trace = trace_check(&gpu, &got, i, rs, 64);
                eprintln!(
                    "P4 {name} candidate={i} index_error={ni:e} first_order_error={fo:e} trace={} stats={}",
                    serde_json::to_string(&trace).unwrap(),
                    serde_json::to_string(&stats).unwrap()
                );
            }
        }
    }
}
#[test]
fn host_metal_extended_and_concurrent() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let gpu = backend(origin);
        let cpu = CpuF32Backend::matching(&gpu.precision().probe);
        for (name, c, rows) in extended_cases() {
            let (want, _) = run(&cpu, &c, &rows, 64);
            let (got, stats) = run(&gpu, &c, &rows, 1);
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(a.words(), b.words(), "{name} buffer {i}");
            }
            for (i, row) in rows.iter().enumerate() {
                let rs = c.reference(row, &JobContext::new()).unwrap();
                let (ni, fo) = compare(&got, i, &rs);
                let trace = trace_check(&gpu, &got, i, rs, 32);
                eprintln!(
                    "P4-extended {name} candidate={i} index_error={ni:e} first_order_error={fo:e} trace={} stats={}",
                    serde_json::to_string(&trace).unwrap(),
                    serde_json::to_string(&stats).unwrap()
                );
            }
            std::thread::scope(|scope| {
                let jobs: Vec<_> = (0..3)
                    .map(|_| scope.spawn(|| run(&gpu, &c, &rows, 64).0))
                    .collect();
                for job in jobs {
                    for (a, b) in got.iter().zip(job.join().unwrap()) {
                        assert_eq!(a.words(), b.words());
                    }
                }
            });
        }
    }
}
#[test]
fn host_metal_cancel_progress_release() {
    use mklight_gpu::resident::*;
    use std::time::{Duration, Instant};
    let gpu = backend(LibraryChoice::Precompiled);
    let system = mklight_core::samples::double_gauss();
    let vars = basic_variables(&system);
    let small = basic_rows(&system);
    let c = CompiledSystem::compile(
        Snapshot::new(system).unwrap(),
        testutil::library(),
        vars,
        None,
        &JobContext::new(),
    )
    .unwrap();
    let uploaded = c.upload(&gpu, &JobContext::new()).unwrap();
    let rows: Vec<_> = (0..1024)
        .map(|i| Candidate {
            id: i,
            values: small[i as usize % 3].values.clone(),
        })
        .collect();
    let build = || {
        let mut plan = ResidentPlan::new(c.snapshot().clone());
        let r = uploaded
            .append(&gpu, &mut plan, &rows, &JobContext::new())
            .unwrap();
        plan.outputs = r.outputs();
        let ResidentOp::Systems(Request::Prepare(r)) = plan.operations.last_mut().unwrap() else {
            unreachable!()
        };
        r.tile_candidates = 1;
        plan
    };
    let ctx = JobContext::new();
    let pending = gpu.submit_plan(build(), &ctx).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut previous = 0.0;
    while ctx.progress.fraction() < 0.01 {
        let f = ctx.progress.fraction();
        assert!(f >= previous);
        previous = f;
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(ctx.progress.fraction() < 1.0);
    let cancelled_at = Instant::now();
    ctx.cancel.cancel();
    assert!(matches!(
        pending.wait(),
        Err(mklight_core::MkError::Cancelled)
    ));
    eprintln!(
        "P4 cancellation_ms={:.3} completed_fraction={}",
        cancelled_at.elapsed().as_secs_f64() * 1e3,
        ctx.progress.fraction()
    );
    assert!(cancelled_at.elapsed() < Duration::from_millis(100));
    // Reuse the pool after cancellation while the released system remains
    // pinned by a submitted job; all outputs still match the small reference.
    let ctx = JobContext::new();
    let pending = gpu.submit_plan(build(), &ctx).unwrap();
    uploaded.release(&gpu).unwrap();
    let mut last = 0.0;
    let mut intermediate = 0;
    while !pending.is_ready() {
        let f = ctx.progress.fraction();
        assert!(f >= last);
        if f > 0.0 && f < 1.0 {
            intermediate += 1;
        }
        last = f;
        std::thread::sleep(Duration::from_millis(1));
    }
    let result = pending.wait().unwrap();
    assert!(intermediate > 0);
    assert_eq!(ctx.progress.fraction(), 1.0);
    assert_eq!(result.stats.fallback_count, 0);
    assert_eq!(result.stats.readback_bytes, 0);
    assert!(result.stats.reused_allocations > 0);
    let max = result
        .stats
        .stages
        .iter()
        .map(|s| s.max_command_seconds)
        .fold(0f64, f64::max);
    eprintln!("P4 max_command_seconds={max:.6}");
    assert!(max < 0.05);
}
#[test]
fn host_metal_resident_generated_rows() {
    resident_input_check(&backend(LibraryChoice::Precompiled));
}
#[test]
fn host_metal_traced_fd() {
    ray_derivative_check(&backend(LibraryChoice::Precompiled));
}
#[test]
fn host_metal_vignetted_tilted_long_track_and_telecentric() {
    domain_check(&backend(LibraryChoice::Precompiled), 64);
}
