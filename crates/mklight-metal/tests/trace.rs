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
use mklight_metal::{MetalBackend, MetalOptions};
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

/// Bit identity of every plane. NaNs only need to be NaN on both sides:
/// IEEE 754 leaves the sign and payload of a generated NaN open, and the
/// GPU's differ from the CPU's (invalid inputs such as a zero direction
/// produce them).
fn assert_bitwise(a: &TraceResult, b: &TraceResult, what: &str) {
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
        assert_eq!(x.len(), y.len(), "{what} {name}");
        let same = |i: usize| x[i].to_bits() == y[i].to_bits() || (x[i].is_nan() && y[i].is_nan());
        if let Some(i) = (0..x.len()).find(|&i| !same(i)) {
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

/// `rays` with the double-float residual planes dropped (plain f32 input).
fn f32_only(rays: &RayBatch) -> RayBatch {
    let mut r = rays.clone();
    for v in [
        &mut r.x_lo,
        &mut r.y_lo,
        &mut r.z_lo,
        &mut r.l_lo,
        &mut r.m_lo,
        &mut r.n_lo,
    ] {
        v.clear();
    }
    r
}

/// `rays` plus invalid inputs: NaN and infinite positions and directions,
/// a zero direction, a huge offset and a ray travelling backward.
fn with_invalid(rays: &RayBatch) -> RayBatch {
    let mut r = rays.clone();
    let bad: [([f64; 3], [f64; 3]); 6] = [
        ([f64::NAN, 0.0, -1.0], [0.0, 0.0, 1.0]),
        ([0.0, f64::INFINITY, -1.0], [0.0, 0.0, 1.0]),
        ([0.0, 1.0, -1.0], [0.0, f64::NEG_INFINITY, 1.0]),
        ([0.0, 1.0, -1.0], [0.0, 0.0, 0.0]),
        ([1e30, -1e30, 0.0], [0.6, 0.0, 0.8]),
        ([0.0, 2.0, -1.0], [0.0, 0.1, -0.99498743710662]),
    ];
    for (k, (p, d)) in bad.into_iter().enumerate() {
        r.push_f64(p, d, (k % 3) as u32);
    }
    r
}

/// Bit identity beyond sequential grids: ghost path programs on every
/// sample (two-reflection ghosts; the Newtonian's sensor ghost through
/// both mirrors), plain-f32 inputs, invalid inputs, and the strong-asphere
/// sweeps with clipping on and off at both Newton caps, all with records.
#[test]
fn gpu_matches_emulation_bitwise_paths_and_edge_cases() {
    let g = gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let run_both =
        |p: &PackedSystem, req: &dyn Fn(mklight_gpu::SystemHandle) -> TraceRequest, what: &str| {
            let (hg, hc) = (g.upload_system(p).unwrap(), c.upload_system(p).unwrap());
            let a = trace(g, req(hg));
            let b = trace(&c, req(hc));
            assert_bitwise(&a, &b, what);
            g.release_systems(hg).unwrap();
            a
        };
    let mut ghosts = 0;
    for (name, sys) in samples::all()
        .into_iter()
        .filter(|(_, s)| !samples::has_extended_surfaces(s))
    {
        let t = testutil::tracer(sys);
        let rs = t.system();
        let p = PackedSystem::pack_all(rs).unwrap();
        let rays = grid_rays(&t, 12);
        let end = rs.image_index() as u32;
        let mut programs = testutil::ghost_programs(rs, &p.surfaces);
        if name == "newtonian" {
            programs.push(testutil::newtonian_sensor_ghost());
        }
        for (k, prog) in programs.into_iter().enumerate() {
            for clip in [true, false] {
                let r = run_both(
                    &p,
                    &|h| {
                        let mut r = TraceRequest::path(h, rays.clone(), 0, prog.clone());
                        r.record = RecordMode::AllSurfaces;
                        r.apply_apertures = clip;
                        r
                    },
                    &format!("{name} ghost {k} clip {clip}"),
                );
                ghosts += r.status.iter().filter(|&&s| s == 0).count();
            }
        }
        for (what, batch) in [
            ("f32 input", f32_only(&rays)),
            ("invalid input", with_invalid(&rays)),
        ] {
            for inputs in [batch.clone(), f32_only(&batch)] {
                run_both(
                    &p,
                    &|h| {
                        let mut r = TraceRequest::new(h, inputs.clone(), 0, end);
                        r.record = RecordMode::AllSurfaces;
                        r
                    },
                    &format!("{name} {what}"),
                );
            }
        }
    }
    assert!(ghosts > 1000, "{ghosts} ghost rays completed");
    let sweep = testutil::sweep_rays();
    let mut batch = RayBatch::with_capacity(sweep.len());
    for r in &sweep {
        batch.push_f64([r.pos.x, r.pos.y, r.pos.z], [r.dir.x, r.dir.y, r.dir.z], 1);
    }
    for (k, shape) in testutil::strong_aspheres().into_iter().enumerate() {
        let rs = testutil::resolve(testutil::one_surface(shape));
        let p = PackedSystem::pack_all(&rs).unwrap();
        for cap in [8, 40] {
            for clip in [true, false] {
                for inputs in [batch.clone(), f32_only(&batch)] {
                    run_both(
                        &p,
                        &|h| {
                            let mut r = TraceRequest::new(h, inputs.clone(), 0, 2);
                            r.record = RecordMode::AllSurfaces;
                            r.apply_apertures = clip;
                            r.newton.max_iterations = cap;
                            r
                        },
                        &format!("asphere {k} cap {cap} clip {clip}"),
                    );
                }
            }
        }
    }
}

/// The Newtonian's sensor ghost on the GPU: back through the fold flat and
/// the primary by their reflections, leaving antiparallel to the incoming
/// beam at the mirrored pupil point (the paraboloid images the axial point
/// perfectly), and matching the independent f64 path.
#[test]
fn catadioptric_ghost_gpu() {
    let g = gpu_or_skip!();
    let t = testutil::tracer(samples::newtonian_telescope());
    let sys = t.system();
    let h = g
        .upload_system(&PackedSystem::pack_all(sys).unwrap())
        .unwrap();
    let prog = testutil::newtonian_sensor_ghost();
    let mut rays = RayBatch::with_capacity(0);
    let mut input = Vec::new();
    for p in (PupilSampling::Grid { n: 15 }).points() {
        let r = t.generate(t.field(0), 1, p[0], p[1]).unwrap().ray;
        rays.push_f64([r.pos.x, r.pos.y, r.pos.z], [r.dir.x, r.dir.y, r.dir.z], 1);
        input.push(r);
    }
    let r = trace(g, TraceRequest::path(h, rays, 0, prog.clone()));
    for (i, ray) in input.iter().enumerate() {
        assert!(r.ray_status(i).is_ok());
        let p = Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64);
        let d = Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64);
        let (_, rec) = mklight_trace::trace_ray_recorded(sys, ray, 1, &t.options());
        let mirrored = Vec3::new(-rec[1].point.x, -rec[1].point.y, rec[1].point.z);
        assert!((p - mirrored).norm() < 2e-5, "ray {i}");
        assert!((d - Vec3::new(0.0, 0.0, -1.0)).norm() < 2e-6, "ray {i}");
        let want = f64_path(sys, 0, ray.pos, ray.dir, 1, &prog).unwrap();
        assert!((p - want.pos).norm() < 4e-6, "ray {i}");
        let want_opl = want.opl + (p - want.pos).dot(want.dir);
        assert!((r.opl_f64(i) - want_opl).abs() < 1e-8, "ray {i}");
    }
    g.release_systems(h).unwrap();
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
    for (name, sys) in samples::all().into_iter() {
        let q = qualify(g, &testutil::tracer(sys), None, &ctx).unwrap();
        assert!(q.passed, "{name}: {q:?}");
        assert!(q.budget_fraction.unwrap() < q.margin);
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
        failed_attempts: vec![],
    };
    qualify(g, &t, None, &ctx).unwrap().record(&mut info);
    assert!(info.backend.starts_with("metal:"));
    assert_eq!(info.precision, Precision::F32Gpu);
    let mut strict = XvalSettings::qualification();
    strict.budget.opd_waves = 1e-9;
    let q = qualify(g, &t, Some(&strict), &ctx).unwrap();
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
/// `cargo test --release -p mklight-metal --test trace -- --ignored --nocapture`;
/// `MKLIGHT_METAL_PRECISION=f32` times the pure-f32 kernels (docs/GPU.md §3.6).
#[test]
#[ignore = "benchmark"]
fn benchmark_double_gauss_1e6() {
    let options = MetalOptions::from_env();
    let Some(g) = MetalBackend::with_options(&options).unwrap() else {
        return eprintln!("SKIP: no Metal device");
    };
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
        "{n} rays x {steps} surfaces on {} ({}): GPU {:.1} ms wall ({:.2e} ray-surfaces/s), \
         {:.1} ms device ({:.2e}/s); CPU f64 rayon ({} threads) {:.1} ms ({:.2e}/s); \
         speed-up {:.1}x wall",
        g.info().name,
        options.precision.name(),
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

#[test]
fn extended_full_capacity_and_steep_rays_bitwise() {
    use mklight_core::model::Interaction;
    use mklight_gpu::consts::MKL_RAY_EVANESCENT;
    let g = gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let rays = testutil::steep_random_rays();
    let mut b = RayBatch::with_capacity(rays.len());
    for (i, r) in rays.iter().enumerate() {
        b.push_f64(
            [r.pos.x, r.pos.y, r.pos.z],
            [r.dir.x, r.dir.y, r.dir.z],
            (i % 3) as u32,
        );
    }
    for x in [-1e-30, 0.0, 1e-30, 1e-10] {
        b.push_f64([x, 0.0, -5.0], [0.0, 0.0, 1.0], 1);
    }
    let mut systems: Vec<_> = testutil::extended_shapes()
        .into_iter()
        .map(|(n, s)| (n.to_string(), testutil::one_surface(s)))
        .collect();
    systems.push((
        "odd cone".into(),
        testutil::one_surface(mklight_core::SurfaceShape::OddAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![0.01],
        }),
    ));
    let doe = systems
        .iter()
        .find(|(n, _)| n == "doe10")
        .unwrap()
        .1
        .clone();
    for interaction in [
        Interaction::Refract,
        Interaction::Reflect,
        Interaction::Dummy,
    ] {
        for order in [-2, 0, 3, 10000] {
            let mut sys = doe.clone();
            sys.surfaces[1].interaction = interaction;
            if let mklight_core::SurfaceShape::Diffractive { phase, .. } =
                &mut sys.surfaces[1].shape
            {
                phase.trace_order = order;
            }
            systems.push((format!("DOE {interaction:?} order {order}"), sys));
        }
    }
    let packed: Vec<_> = systems
        .iter()
        .map(|(_, s)| PackedSystem::pack_all(&testutil::resolve(s.clone())).unwrap())
        .collect();
    // One upload exercises rebasing of every surface's pool offset.
    let (hg, hc) = (
        g.upload_systems(&packed).unwrap(),
        c.upload_systems(&packed).unwrap(),
    );
    let mut evanescent = 0;
    for (k, (name, _)) in systems.iter().enumerate() {
        let mut rays = b.clone();
        rays.system = vec![k as u32; rays.len()];
        let req = |h| {
            let mut r = TraceRequest::new(h, rays.clone(), 0, 1);
            r.record = RecordMode::AllSurfaces;
            r.apply_apertures = false;
            r
        };
        let a = trace(g, req(hg));
        let b = trace(&c, req(hc));
        assert_bitwise(&a, &b, name);
        evanescent += a
            .status
            .iter()
            .filter(|s| **s & 255 == MKL_RAY_EVANESCENT)
            .count();
        if systems[k].1.surfaces[1].interaction == Interaction::Dummy {
            let forced = |h| {
                let mut r = TraceRequest::path(
                    h,
                    rays.clone(),
                    0,
                    vec![PathStep::forward(1, StepMode::Reflect)],
                );
                r.record = RecordMode::AllSurfaces;
                r.apply_apertures = false;
                r
            };
            let a = trace(g, forced(hg));
            let b = trace(&c, forced(hc));
            assert_bitwise(&a, &b, &format!("{name}, forced reflection"));
            assert!(a.status.iter().all(|s| s & 255 != MKL_RAY_EVANESCENT));
        }
        let r2: Vec<f32> = (0..=64).map(|j| j as f32 * 160.0 / 64.0).collect();
        let probe = |backend: &dyn ComputeBackend, h| match backend
            .run(
                Job::SagProbe {
                    systems: h,
                    system: k as u32,
                    surface: 1,
                    r2: r2.clone(),
                },
                &JobContext::new(),
            )
            .unwrap()
            .output
        {
            JobOutput::F32(v) => v,
            _ => panic!(),
        };
        let (a, b) = (probe(g, hg), probe(&c, hc));
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{name} sag {i}: {a} {b}"
            );
        }
    }
    assert!(evanescent > 1000);
    g.release_systems(hg).unwrap();
    c.release_systems(hc).unwrap();
}

/// Same rays, one refracting surface and full coefficient capacities; best
/// of ten interleaved dispatches. Device time excludes upload/readback; wall
/// includes the host ray planes and output unpacking (GPU.md §5.3).
#[test]
#[ignore = "benchmark"]
fn benchmark_extended_shapes() {
    let g = gpu_or_skip!();
    eprintln!(
        "device {} build {}",
        g.info().name,
        g.precision().kernel_build
    );
    let n = 1 << 18;
    let mut rays = RayBatch::with_capacity(n);
    for i in 0..n {
        let a = i as f64 * 2.399963229728653;
        let r = 8.0 * ((i % 4096) as f64 / 4096.0).sqrt();
        rays.push_f64([r * a.cos(), r * a.sin(), -5.0], [0.0, 0.0, 1.0], 1);
    }
    let mut shapes = vec![(
        "even10",
        mklight_core::SurfaceShape::EvenAsphere {
            curvature: 0.04,
            conic: -0.4,
            coefficients: (0..10).map(|j| 0.01 / 16f64.powi(2 * j + 4)).collect(),
        },
    )];
    shapes.extend(testutil::extended_shapes());
    let reqs: Vec<_> = shapes
        .iter()
        .map(|(_, shape)| {
            let rs = testutil::resolve(testutil::one_surface(shape.clone()));
            let h = g
                .upload_system(&PackedSystem::pack_all(&rs).unwrap())
                .unwrap();
            let mut req = TraceRequest::new(h, rays.clone(), 0, 1);
            req.apply_apertures = false;
            req.radiometry = Radiometry::Off;
            req
        })
        .collect();
    // Rounds interleave the shapes, so GPU clock ramps and machine load
    // affect them alike; the first round is a warm-up.
    let mut best = vec![(f64::INFINITY, f64::INFINITY); shapes.len()];
    for round in 0..11 {
        for ((name, _), (req, best)) in shapes.iter().zip(reqs.iter().zip(&mut best)) {
            let r = g
                .run(Job::Trace(Box::new(req.clone())), &JobContext::new())
                .unwrap();
            let JobOutput::Trace(t) = r.output else {
                panic!()
            };
            assert!(t.status.iter().all(|s| *s == 0), "{name}");
            if round > 0 {
                best.0 = best.0.min(r.stats.device_seconds);
                best.1 = best.1.min(r.stats.wall_seconds);
            }
        }
    }
    for ((name, _), (device, wall)) in shapes.iter().zip(&best) {
        eprintln!(
            "| {name} | {:.1} | {:.1} | {:.3} |",
            n as f64 / device / 1e6,
            n as f64 / wall / 1e6,
            best[0].0 / device
        );
    }
    for req in reqs {
        g.release_systems(req.systems).unwrap();
    }
}
