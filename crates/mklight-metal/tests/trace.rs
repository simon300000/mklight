//! The Metal trace kernel on the real GPU: bit-identical to the CPU-f32
//! emulation (every output and record), the cross-validation suite against
//! the f64 reference, path programs against an independent f64
//! computation, precision qualification, job mechanics, and a throughput
//! benchmark (ignored by default). Every test skips without a Metal device.
#![cfg(target_os = "macos")]

use std::sync::OnceLock;
use std::time::Instant;

use mklight_core::job::{Precision, RunInfo};
use mklight_core::{JobContext, MkError, Vec3, samples};
use mklight_gpu::testutil::{self, f64_path};
use mklight_gpu::xval::{XvalSettings, cross_validate, qualify};
use mklight_gpu::{
    ComputeBackend, CpuF32Backend, Job, JobKind, JobOutput, PackedSystem, PathStep, Radiometry,
    RayBatch, RecordMode, StepMode, TraceRequest, TraceResult,
};
use mklight_metal::MetalBackend;
use mklight_trace::{PupilSampling, Ray, SequentialTracer};
use rayon::prelude::*;

fn gpu() -> Option<&'static MetalBackend> {
    static B: OnceLock<Option<MetalBackend>> = OnceLock::new();
    let b = B
        .get_or_init(|| MetalBackend::new().expect("Metal backend initialisation failed"))
        .as_ref();
    if b.is_none() {
        eprintln!("SKIP: no Metal device");
    }
    b
}

macro_rules! gpu_or_skip {
    () => {
        match gpu() {
            Some(b) => b,
            None => return,
        }
    };
}

fn trace(b: &dyn ComputeBackend, req: TraceRequest) -> TraceResult {
    match b
        .run(Job::Trace(Box::new(req)), &JobContext::new())
        .unwrap()
        .output
    {
        JobOutput::Trace(t) => *t,
        o => panic!("{o:?}"),
    }
}

/// Pupil-grid rays of every field and wavelength, as an f64 batch.
fn grid_rays(t: &SequentialTracer, n: usize) -> RayBatch {
    let mut b = RayBatch::with_capacity(0);
    for f in 0..t.fields().len() {
        for w in 0..t.num_wavelengths() {
            for p in (PupilSampling::Grid { n }).points() {
                if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                    let (a, d) = (g.ray.pos, g.ray.dir);
                    b.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                }
            }
        }
    }
    b
}

fn assert_bitwise(a: &TraceResult, b: &TraceResult, what: &str) {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let planes: [(&[f32], &[f32], &str); 11] = [
        (&a.x, &b.x, "x"),
        (&a.y, &b.y, "y"),
        (&a.z, &b.z, "z"),
        (&a.l, &b.l, "l"),
        (&a.m, &b.m, "m"),
        (&a.n, &b.n, "n"),
        (&a.opl_hi, &b.opl_hi, "opl_hi"),
        (&a.opl_lo, &b.opl_lo, "opl_lo"),
        (&a.weight, &b.weight, "weight"),
        (&a.transmittance, &b.transmittance, "transmittance"),
        (&a.recorded, &b.recorded, "records"),
    ];
    for (x, y, name) in planes {
        let (bx, by) = (bits(x), bits(y));
        assert_eq!(bx.len(), by.len(), "{what} {name}");
        if let Some(i) = (0..bx.len()).find(|&i| bx[i] != by[i]) {
            panic!("{what}: {name}[{i}] GPU {} vs emulation {}", x[i], y[i]);
        }
    }
    assert_eq!(a.status, b.status, "{what}");
    assert_eq!(a.last_surface, b.last_surface, "{what}");
    assert_eq!(a.steps, b.steps, "{what}");
    assert_eq!(a.recorded_status, b.recorded_status, "{what}");
}

/// Every output and every per-surface record of the GPU is bit-identical to
/// the CPU-f32 emulation in the device's f32 environment, for every sample
/// and fixture (all fields, wavelengths, a 32 × 32 grid, radiometry on).
#[test]
fn gpu_matches_emulation_bitwise() {
    let g = gpu_or_skip!();
    assert!(g.supports(JobKind::Trace));
    let c = CpuF32Backend::matching(&g.precision().probe);
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let p = PackedSystem::pack_all(t.system()).unwrap();
        let (hg, hc) = (g.upload_system(&p).unwrap(), c.upload_system(&p).unwrap());
        let rays = grid_rays(&t, 32);
        let end = t.system().image_index() as u32;
        let req = |h| {
            let mut r = TraceRequest::new(h, rays.clone(), 0, end);
            r.record = RecordMode::AllSurfaces;
            r
        };
        let a = trace(g, req(hg));
        let b = trace(&c, req(hc));
        assert_bitwise(&a, &b, &name);
        g.release_systems(hg).unwrap();
    }
}

/// The cross-validation suite on the GPU: every sample (plain and
/// vignetted) and fixture, all fields and wavelengths, a 64 × 64 pupil
/// grid, against the f64 reference, within the CONVENTIONS §7 budget. The
/// printed table is the one in docs/GPU.md §4.
#[test]
fn cross_validation_suite_gpu() {
    let g = gpu_or_skip!();
    let s = XvalSettings::default();
    eprintln!("| system | rays | compared | position (mm) | direction | OPD (waves) | band |");
    eprintln!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let r = cross_validate(g, &t, &s, &JobContext::new()).unwrap();
        eprintln!(
            "| {name} | {} | {} | {:.1e} | {:.1e} | {:.1e} | {} |",
            r.rays,
            r.compared,
            r.max_position_mm,
            r.max_direction,
            r.max_opd_waves,
            r.band_mismatches
        );
        r.check(&s.budget)
            .unwrap_or_else(|e| panic!("{name}: {e} {:?}", r.notes));
        assert!(r.backend.starts_with("metal:"));
        assert!(r.max_transmittance_abs < 1e-6, "{name}");
    }
}

/// The plate's two-reflection ghost path (forward and backward steps,
/// straight and tilted) on the GPU against the independent f64 path.
#[test]
fn plate_two_reflection_path_gpu() {
    let g = gpu_or_skip!();
    use StepMode::*;
    let prog = vec![
        PathStep::forward(1, Transmit),
        PathStep::forward(2, Reflect),
        PathStep::backward(1, Reflect),
        PathStep::forward(2, Transmit),
        PathStep::forward(3, Transmit),
    ];
    for tilted in [false, true] {
        let t = testutil::tracer(testutil::plate(tilted));
        let sys = t.system();
        let h = g
            .upload_system(&PackedSystem::pack_all(sys).unwrap())
            .unwrap();
        let mut rays = RayBatch::with_capacity(0);
        let mut input = Vec::new();
        for k in 0..64 {
            let a = k as f64 * 0.1;
            let p = Vec3::new(4.0 * a.cos(), 3.0 * a.sin(), -5.0);
            let d = Vec3::new(0.15 * a.sin(), -0.2 * a.cos(), 1.0).normalized();
            let w = (k % 3) as u32;
            rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], w);
            input.push((Ray::new(p, d), w));
        }
        let r = trace(g, TraceRequest::path(h, rays, 0, prog.clone()));
        for (i, (ray, w)) in input.iter().enumerate() {
            let want = f64_path(sys, 0, ray.pos, ray.dir, *w as usize, &prog).unwrap();
            assert!(r.ray_status(i).is_ok());
            let p = Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64);
            let d = Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64);
            assert!((p - want.pos).norm() < 1e-6, "tilted {tilted} ray {i}");
            assert!((d - want.dir).norm() < 1e-7, "tilted {tilted} ray {i}");
            let want_opl = want.opl + (p - want.pos).dot(want.dir);
            assert!(
                (r.opl_f64(i) - want_opl).abs() < 1e-9,
                "tilted {tilted} ray {i}"
            );
            assert!((r.weight[i] as f64 / want.weight - 1.0).abs() < 1e-5);
        }
        g.release_systems(h).unwrap();
    }
}

/// Qualification on the GPU passes for the samples and is recorded in
/// `RunInfo`; an unmeetable budget makes the caller fall back, recorded.
#[test]
fn qualification_on_gpu() {
    let g = gpu_or_skip!();
    let ctx = JobContext::new();
    for (name, sys) in samples::all() {
        let q = qualify(g, &testutil::tracer(sys), None, &ctx);
        assert!(q.passed, "{name}: {q:?}");
        assert!(q.kernel_build.contains("mklight-metal"));
    }
    let t = testutil::tracer(samples::double_gauss());
    let mut info = RunInfo {
        engine_version: "test".into(),
        snapshot_hash: String::new(),
        analysis: "spot".into(),
        settings: serde_json::Value::Null,
        backend: String::new(),
        precision: Precision::F64Reference,
        seed: None,
        started_unix_ms: 0,
        duration_ms: 0.0,
        warnings: vec![],
    };
    qualify(g, &t, None, &ctx).record(&mut info);
    assert!(info.backend.starts_with("metal:"));
    assert_eq!(info.precision, Precision::F32Gpu);
    let mut strict = XvalSettings::qualification();
    strict.budget.opd_waves = 1e-9;
    let q = qualify(g, &t, Some(&strict), &ctx);
    assert!(!q.passed);
    let mut info = info.clone();
    q.record(&mut info);
    assert_eq!(info.backend, "cpu-f64");
    assert!(
        info.warnings
            .iter()
            .any(|w| w.contains("metal:") && w.contains("OPD"))
    );
}

/// Command-buffer chunking does not change a bit; progress reaches 1;
/// cancellation stops the job; empty batches complete.
#[test]
fn trace_chunks_progress_and_cancellation() {
    let g = gpu_or_skip!();
    let t = testutil::tracer(samples::cooke_triplet());
    let h = g
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let rays = grid_rays(&t, 24);
    let end = t.system().image_index() as u32;
    let mut req = TraceRequest::new(h, rays.clone(), 0, end);
    req.record = RecordMode::AllSurfaces;
    let whole = trace(g, req.clone());
    g.set_chunk_items(997);
    let ctx = JobContext::new();
    let r = g.run(Job::Trace(Box::new(req.clone())), &ctx).unwrap();
    g.set_chunk_items(1 << 20);
    assert!(r.stats.command_buffers > 3);
    assert!(r.stats.device_seconds > 0.0);
    assert_eq!(ctx.progress.fraction(), 1.0);
    let JobOutput::Trace(chunked) = r.output else {
        panic!()
    };
    assert_bitwise(&chunked, &whole, "chunked");
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        g.run(Job::Trace(Box::new(req)), &ctx),
        Err(MkError::Cancelled)
    );
    let empty = TraceRequest::new(h, RayBatch::default(), 0, end);
    assert!(trace(g, empty).is_empty());
    let mut off = TraceRequest::new(h, rays, 0, end);
    off.radiometry = Radiometry::Off;
    off.apply_apertures = false;
    assert_eq!(trace(g, off).x.len(), whole.x.len());
    g.release_systems(h).unwrap();
}

/// Throughput: about 1e6 Double-Gauss rays (12 surfaces each) on the GPU,
/// including packing the planes, upload and download, against the
/// `mklight-trace` f64 reference on all CPU cores (rayon). Run with
/// `cargo test --release -p mklight-metal --test trace -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn benchmark_double_gauss_1e6() {
    let g = gpu_or_skip!();
    let t = testutil::tracer(samples::double_gauss());
    let sys = t.system();
    let mut rays = Vec::new();
    for f in 0..3 {
        for p in (PupilSampling::Grid { n: 650 }).points() {
            rays.push(t.generate(t.field(f), 1, p[0], p[1]).unwrap().ray);
        }
    }
    let n = rays.len();
    let steps = sys.image_index();
    let mut batch = RayBatch::with_capacity(n);
    for r in &rays {
        batch.push_f64([r.pos.x, r.pos.y, r.pos.z], [r.dir.x, r.dir.y, r.dir.z], 1);
    }
    let h = g
        .upload_system(&PackedSystem::pack_all(sys).unwrap())
        .unwrap();
    let opts = t.options();
    let mut cpu_best = f64::INFINITY;
    let mut gpu_best = f64::INFINITY;
    let mut dev_best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let out: Vec<_> = rays
            .par_iter()
            .map(|r| mklight_trace::trace_ray(sys, r, 1, &opts, None))
            .collect();
        cpu_best = cpu_best.min(t0.elapsed().as_secs_f64());
        assert_eq!(out.len(), n);
        let req = TraceRequest::new(h, batch.clone(), 0, steps as u32);
        let t0 = Instant::now();
        let r = g
            .run(Job::Trace(Box::new(req)), &JobContext::new())
            .unwrap();
        gpu_best = gpu_best.min(t0.elapsed().as_secs_f64());
        dev_best = dev_best.min(r.stats.device_seconds);
    }
    let rs = (n * steps) as f64;
    eprintln!(
        "{n} rays x {steps} surfaces on {}: GPU {:.1} ms wall ({:.2e} ray-surfaces/s), \
         {:.1} ms device ({:.2e}/s); CPU f64 rayon ({} threads) {:.1} ms ({:.2e}/s); \
         speed-up {:.1}x wall",
        g.info().name,
        gpu_best * 1e3,
        rs / gpu_best,
        dev_best * 1e3,
        rs / dev_best,
        rayon::current_num_threads(),
        cpu_best * 1e3,
        rs / cpu_best,
        cpu_best / gpu_best
    );
}
