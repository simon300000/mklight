//! The trace kernel on the device against the CPU-f32 emulation and the f64
//! reference (the WebGPU twin of `mklight-metal/tests/trace.rs` and of the
//! resident trace tests of `p0_resident.rs` / `p1_aiming.rs`).
//!
//! Against the CPU-f32 emulation in the device's environment, statuses,
//! surface indices, step counts and record statuses agree exactly. Floats
//! agree bit for bit except where a hardware division or square root
//! (not correctly rounded on every WebGPU compiler) enters; there the tests
//! bound the difference far inside the CONVENTIONS §7 budgets and print the
//! largest differences. The cross-validation suite against the f64
//! reference passes the budgets themselves. Every test skips without a
//! WebGPU adapter.

use std::time::Instant;

use mklight_core::job::{Precision, RunInfo};
use mklight_core::{JobContext, MkError, Snapshot, Vec3, samples};
use mklight_gpu::resident::*;
use mklight_gpu::testutil::{self, f64_path};
use mklight_gpu::xval::{XvalSettings, cross_validate, qualify};
use mklight_gpu::{
    ComputeBackend, CpuF32Backend, Job, JobKind, JobOutput, PackedSystem, PathStep,
    QualificationBackend, Radiometry, RayBatch, RecordMode, ReduceMode, StepMode, SystemHandle,
    TraceRequest, TraceResult,
};
use mklight_trace::{PupilSampling, Ray, SequentialTracer};
use mklight_webgpu::ops::trace::set_chunk_rays;
use rayon::prelude::*;

#[path = "../../mklight-gpu/tests/p1_support.rs"]
pub(crate) mod p1_support;

fn trace(b: &dyn QualificationBackend, req: TraceRequest) -> TraceResult {
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

/// Bit identity of every plane and record. NaNs only need to be NaN on both
/// sides: IEEE 754 leaves the sign and payload of a generated NaN open
/// (invalid inputs such as a zero direction produce them).
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
            panic!("{what}: {name}[{i}] device {} vs emulation {}", x[i], y[i]);
        }
    }
    assert_eq!(a.status, b.status, "{what}");
    assert_eq!(a.last_surface, b.last_surface, "{what}");
    assert_eq!(a.steps, b.steps, "{what}");
    assert_eq!(a.recorded_status, b.recorded_status, "{what}");
}

fn run_both(
    g: &dyn QualificationBackend,
    c: &CpuF32Backend,
    p: &PackedSystem,
    req: &dyn Fn(SystemHandle) -> TraceRequest,
    what: &str,
) -> TraceResult {
    let (hg, hc) = (g.upload_system(p).unwrap(), c.upload_system(p).unwrap());
    let a = trace(g, req(hg));
    let b = trace(c, req(hc));
    assert_bitwise(&a, &b, what);
    g.release_systems(hg).unwrap();
    c.release_systems(hc).unwrap();
    a
}

/// Every output and every per-surface record of the device against the
/// CPU-f32 emulation in the device's f32 environment, for every sample and
/// fixture (all fields, wavelengths, a 32 × 32 grid, radiometry on).
#[test]
fn device_matches_emulation() {
    let g = crate::gpu_or_skip!();
    assert!(g.supports(JobKind::Trace));
    let c = CpuF32Backend::matching(&g.precision().probe);
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let p = PackedSystem::pack_all(t.system()).unwrap();
        let rays = grid_rays(&t, 32);
        let end = t.system().image_index() as u32;
        let req = |h| {
            let mut r = TraceRequest::new(h, rays.clone(), 0, end);
            r.record = RecordMode::AllSurfaces;
            r
        };
        run_both(g, &c, &p, &req, &name);
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

/// Beyond sequential grids: ghost path programs on every sample
/// (two-reflection ghosts; the Newtonian's sensor ghost through both
/// mirrors), plain-f32 inputs, invalid inputs, and the strong-asphere sweeps
/// with clipping on and off at both Newton caps, all with records.
#[test]
fn device_matches_emulation_paths_and_edge_cases() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
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
                    g,
                    &c,
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
                    g,
                    &c,
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
                        g,
                        &c,
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

/// Extended shapes (odd asphere, Q-con, Q-bfs, DOE, toroid, biconic, XY,
/// Zernike, sag overlays) at full coefficient capacity with steep random
/// rays, the DOE in every interaction and order (including evanescent
/// orders and a forced reflection at a dummy surface), in one upload.
#[test]
fn extended_full_capacity_and_steep_rays() {
    use mklight_core::model::Interaction;
    use mklight_gpu::consts::MKL_RAY_EVANESCENT;
    let g = crate::gpu_or_skip!();
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
    let overlays = packed.iter().any(|p| {
        p.surfaces
            .iter()
            .any(|s| s.flags & mklight_gpu::consts::MKL_SURF_HAS_SAG_OVERLAY != 0)
    });
    // The overlay module (trace_overlays) runs too.
    assert!(overlays);
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
        assert_bitwise(&a, &trace(&c, req(hc)), name);
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
            assert_bitwise(
                &a,
                &trace(&c, forced(hc)),
                &format!("{name}, forced reflection"),
            );
            assert!(a.status.iter().all(|s| s & 255 != MKL_RAY_EVANESCENT));
        }
    }
    assert!(evanescent > 1000);
    g.release_systems(hg).unwrap();
    c.release_systems(hc).unwrap();
}

/// The Newtonian's sensor ghost on the device: back through the fold flat
/// and the primary by their reflections, leaving antiparallel to the
/// incoming beam at the mirrored pupil point, and matching the independent
/// f64 path.
#[test]
fn catadioptric_ghost() {
    let g = crate::gpu_or_skip!();
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

/// The cross-validation suite on the device: every sample (plain and
/// vignetted) and fixture, all fields and wavelengths, a 64 × 64 pupil
/// grid, against the f64 reference, within the CONVENTIONS §7 budget.
#[test]
fn cross_validation_suite() {
    let g = crate::gpu_or_skip!();
    let s = XvalSettings::default();
    eprintln!(
        "| system | rays | compared | position (mm) | direction | OPD (waves) | band | budget |"
    );
    eprintln!("|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut worst = 0f64;
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let r = cross_validate(g, &t, &s, &JobContext::new()).unwrap();
        let fraction = r.budget_fraction(&s.budget);
        worst = worst.max(fraction);
        eprintln!(
            "| {name} | {} | {} | {:.1e} | {:.1e} | {:.1e} | {} | {fraction:.3} |",
            r.rays,
            r.compared,
            r.max_position_mm,
            r.max_direction,
            r.max_opd_waves,
            r.band_mismatches
        );
        r.check(&s.budget)
            .unwrap_or_else(|e| panic!("{name}: {e} {:?}", r.notes));
        assert!(r.backend.starts_with("webgpu:"), "{}", r.backend);
        assert!(r.max_transmittance_abs < 1e-6, "{name}");
    }
    eprintln!("largest budget fraction {worst:.3}");
}

/// The plate's two-reflection ghost path (forward and backward steps,
/// straight and tilted) on the device against the independent f64 path.
#[test]
fn plate_two_reflection_path() {
    let g = crate::gpu_or_skip!();
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

/// Qualification on the device passes for the samples and is recorded in
/// `RunInfo`; an unmeetable budget makes the caller fall back, recorded.
#[test]
fn qualification_on_device() {
    let g = crate::gpu_or_skip!();
    let ctx = JobContext::new();
    for (name, sys) in samples::all().into_iter() {
        let q = qualify(g, &testutil::tracer(sys), None, &ctx).unwrap();
        assert!(q.passed, "{name}: {q:?}");
        assert!(q.budget_fraction.unwrap() < q.margin);
        assert!(q.kernel_build.contains("WGSL"));
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
        tier: Default::default(),
        aim_stop: Default::default(),
        warnings: vec![],
        failed_attempts: vec![],
    };
    qualify(g, &t, None, &ctx).unwrap().record(&mut info);
    assert!(info.backend.starts_with("webgpu:"));
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
            .any(|w| w.contains("webgpu:") && w.contains("OPD"))
    );
}

/// Launch chunking does not change a bit; progress reaches 1; cancellation
/// stops the job; empty batches complete.
#[test]
fn trace_chunks_progress_and_cancellation() {
    let g = crate::gpu_or_skip!();
    let t = testutil::tracer(samples::cooke_triplet());
    let h = g
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let rays = grid_rays(&t, 24);
    let end = t.system().image_index() as u32;
    let mut req = TraceRequest::new(h, rays.clone(), 0, end);
    req.record = RecordMode::AllSurfaces;
    let whole = trace(g, req.clone());
    let ctx = JobContext::new();
    let previous = set_chunk_rays(997);
    let r = g.run(Job::Trace(Box::new(req.clone())), &ctx);
    set_chunk_rays(previous);
    let r = r.unwrap();
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

fn read(b: &dyn QualificationBackend, h: &BufferHandle) -> BufferData {
    b.readback(h.view(), &JobContext::new()).unwrap().data
}

/// A resident plan tracing a host request (`ResidentOp::TraceUpload`) and
/// summing the positions: the trace on the device (the stage says so), its
/// outputs and records as the CPU-f32 job's, chunked or not; the sum is the
/// specification's of the device's positions.
#[test]
fn resident_trace_upload() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    for (name, sys) in testutil::xval_systems().into_iter().step_by(3) {
        let snap = Snapshot::new(sys.clone()).unwrap();
        let t = testutil::tracer(sys);
        let packed = PackedSystem::pack_all(t.system()).unwrap();
        let (hg, hc) = (
            g.upload_system(&packed).unwrap(),
            c.upload_system(&packed).unwrap(),
        );
        let rays = grid_rays(&t, 9);
        let n = rays.len();
        let request = |h| {
            let mut q = TraceRequest::new(h, rays.clone(), 0, t.system().image_index() as u32);
            q.radiometry = Radiometry::Off;
            q.record = RecordMode::AllSurfaces;
            q
        };
        let cpu = trace(&c, request(hc));
        let mut results = vec![];
        for chunk in [103, 1 << 18] {
            let mut p = ResidentPlan::new(snap.clone());
            let o = p.trace(request(hg));
            let input = PlanView {
                buffer: o.floats,
                offset: 0,
                desc: BufferDesc::new(ElementType::F32, vec![3, n]),
                stride: 1,
            };
            let sum = p.sum(input, ReduceMode::DoubleFloat);
            p.outputs = vec![sum, o.floats, o.indices, o.records, o.record_status];
            let previous = set_chunk_rays(chunk);
            let ctx = JobContext::new();
            let out = g.submit_plan(p, &ctx).unwrap().wait();
            set_chunk_rays(previous);
            let out = out.unwrap();
            assert_eq!(ctx.progress.fraction(), 1.);
            assert!(out.stats.stages.iter().any(|s| s.name == "webgpu:trace"));
            let words = |k: usize| read(g, &out.outputs[k]).words().to_vec();
            let gpu = TraceResult::from_planes(
                bytemuck::cast_slice(&words(1)),
                &words(2),
                n,
                n,
                bytemuck::cast_slice(&words(3)).to_vec(),
                words(4),
            );
            assert_bitwise(&gpu, &cpu, &name);
            let BufferData::DoubleFloat(sums) = read(g, &out.outputs[0]) else {
                panic!()
            };
            for (ch, values) in [&gpu.x, &gpu.y, &gpu.z].iter().enumerate() {
                let want = mklight_gpu::reduce::reduce_channel_in::<mklight_gpu::Ftz>(
                    ReduceMode::DoubleFloat,
                    values,
                );
                assert!(
                    sums[ch] == want || (want.hi.is_nan() && sums[ch].hi.is_nan()),
                    "{name}"
                );
            }
            results.push(gpu);
        }
        assert_bitwise(&results[0], &results[1], &format!("{name} chunked plan"));
        g.release_systems(hg).unwrap();
        c.release_systems(hc).unwrap();
    }
}

/// Aimed resident rays (`GenerateAim`, the specification on the host)
/// traced on the device (`TraceResident` with and without records, apertures
/// on) and projected to the pupil: the trace matches the CPU-f32 plan.
#[test]
fn resident_trace_matches_cpu_plan() {
    use mklight_core::RayAiming;
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 7 }).points());
    for (name, model) in testutil::xval_systems().into_iter().step_by(2) {
        for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            let mut model = model.clone();
            model.ray_aiming = aiming;
            let t = testutil::tracer(model);
            for records in [true, false] {
                let field = t.fields().len() - 1;
                let wl = t.num_wavelengths() / 2;
                let (p, h) = p1_support::make_plan(g, &t, field, wl, &points, records, 40, true);
                let (pc, hc) = p1_support::make_plan(&c, &t, field, wl, &points, records, 40, true);
                let ctx = JobContext::new();
                let r = g.submit_plan(p, &ctx).unwrap().wait().unwrap();
                assert_eq!(ctx.progress.fraction(), 1.0);
                assert!(
                    r.stats
                        .stages
                        .iter()
                        .any(|s| s.name == "webgpu:trace_resident"),
                    "{:?}",
                    r.stats.stages
                );
                let rc = c
                    .submit_plan(pc, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let (a, b) = (p1_support::read_all(g, &r), p1_support::read_all(&c, &rc));
                eprintln!("{name} {aiming:?} records {records}: bitwise");
                p1_support::bits(&a, &b);
                g.release_systems(h).unwrap();
                c.release_systems(hc).unwrap();
            }
        }
    }
}

/// A resident path program that starts backwards from the image (the
/// singlet traced back to the object), with records.
#[test]
fn resident_path_can_start_backwards() {
    use mklight_gpu::aiming::*;
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let t = testutil::tracer(samples::singlet());
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    let run = |b: &dyn QualificationBackend| {
        let h = b.upload_system(&packed).unwrap();
        let mut p = ResidentPlan::new(Snapshot::new(t.system().system().clone()).unwrap());
        let input = rays(&mut p, 1);
        for (id, data) in [
            (
                input.floats,
                BufferData::F32(vec![
                    0.5, 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                ]),
            ),
            (input.indices, BufferData::U32(vec![0, 1])),
            (input.logical_ids, BufferData::U32(vec![7, 0])),
            (input.report, BufferData::U32(vec![0; 4])),
        ] {
            let buffer = b.upload_buffer(p.desc(id).unwrap().clone(), data).unwrap();
            p.buffers[id.0] = PlanBuffer::Import(buffer.view());
        }
        let start = t.system().image_index() as u32;
        let steps: Vec<_> = (0..start)
            .rev()
            .map(|surface| {
                PathStep {
                    surface,
                    mode: StepMode::Transmit,
                    backward: true,
                }
                .encode()
            })
            .collect();
        let buffer = b
            .upload_buffer(
                BufferDesc::new(ElementType::U32, [steps.len()]),
                BufferData::U32(steps),
            )
            .unwrap();
        let si = p.import(buffer.view());
        let hit = hits(&mut p, 1, start as usize + 1);
        p.operations
            .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
                systems: h,
                input,
                steps: p.view(si).unwrap(),
                start_surface: start,
                apply_apertures: false,
                radiometry: 0,
                newton: Default::default(),
                output: hit.clone(),
            })));
        p.outputs = vec![
            hit.rays.floats,
            hit.status,
            hit.records.unwrap(),
            hit.record_low.unwrap(),
        ];
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let data = p1_support::read_all(b, &r);
        assert_eq!(p1_support::uints(&data[1]), &[0, 0, start]);
        b.release_systems(h).unwrap();
        data
    };
    p1_support::bits(&run(g), &run(&c));
}

/// Best device and wall seconds of `runs` Double-Gauss traces on `b`.
fn time_trace(b: &dyn QualificationBackend, req: &TraceRequest, runs: usize) -> (f64, f64) {
    let (mut wall, mut dev) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..runs {
        let t0 = Instant::now();
        let r = b
            .run(Job::Trace(Box::new(req.clone())), &JobContext::new())
            .unwrap();
        wall = wall.min(t0.elapsed().as_secs_f64());
        dev = dev.min(r.stats.device_seconds);
    }
    (dev, wall)
}

#[cfg(target_os = "macos")]
fn metal() -> Option<mklight_metal::MetalBackend> {
    mklight_metal::MetalBackend::new().ok().flatten()
}

/// Throughput: about 1e6 Double-Gauss rays (12 surfaces each) on WebGPU,
/// on Metal (macOS) and with the `mklight-trace` f64 reference on all CPU
/// cores (rayon), including packing the planes, upload and download. Run
/// with `cargo test --release -p mklight-webgpu --test tests trace:: --
/// --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn benchmark_double_gauss_1e6() {
    let g = crate::gpu_or_skip!();
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
    let packed = PackedSystem::pack_all(sys).unwrap();
    let rs = (n * steps) as f64;
    let line = |what: &str, dev: f64, wall: f64| {
        eprintln!(
            "{what}: {:.1} ms wall ({:.2e} ray-surfaces/s), {:.1} ms device ({:.2e}/s)",
            wall * 1e3,
            rs / wall,
            dev * 1e3,
            rs / dev
        );
    };
    eprintln!("{n} rays x {steps} surfaces");
    let h = g.upload_system(&packed).unwrap();
    let req = TraceRequest::new(h, batch.clone(), 0, steps as u32);
    let (dev, wall) = time_trace(g, &req, 3);
    line(&format!("WebGPU {}", g.info().name), dev, wall);
    g.release_systems(h).unwrap();
    #[cfg(target_os = "macos")]
    if let Some(m) = metal() {
        let h = m.upload_system(&packed).unwrap();
        let req = TraceRequest::new(h, batch.clone(), 0, steps as u32);
        let (dev, wall) = time_trace(&m, &req, 3);
        line(&format!("Metal {}", m.info().name), dev, wall);
        m.release_systems(h).unwrap();
    }
    let opts = t.options();
    let mut cpu = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let out: Vec<_> = rays
            .par_iter()
            .map(|r| mklight_trace::trace_ray(sys, r, 1, &opts, None))
            .collect();
        cpu = cpu.min(t0.elapsed().as_secs_f64());
        assert_eq!(out.len(), n);
    }
    eprintln!(
        "CPU f64 rayon ({} threads): {:.1} ms ({:.2e} ray-surfaces/s)",
        rayon::current_num_threads(),
        cpu * 1e3,
        rs / cpu
    );
}

/// Same rays, one refracting surface and full coefficient capacities; best
/// of ten interleaved runs per backend. Mrays/s on device and wall time
/// (GPU.md §5.3) for WebGPU, Metal (macOS) and CPU f64 (rayon).
#[test]
#[ignore = "benchmark"]
fn benchmark_extended_shapes() {
    let g = crate::gpu_or_skip!();
    eprintln!(
        "device {} build {}",
        g.info().name,
        g.precision().kernel_build
    );
    let n = 1 << 18;
    let mut rays = RayBatch::with_capacity(n);
    let mut f64_rays = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64 * 2.399963229728653;
        let r = 8.0 * ((i % 4096) as f64 / 4096.0).sqrt();
        rays.push_f64([r * a.cos(), r * a.sin(), -5.0], [0.0, 0.0, 1.0], 1);
        f64_rays.push(Ray::new(
            Vec3::new(r * a.cos(), r * a.sin(), -5.0),
            Vec3::new(0.0, 0.0, 1.0),
        ));
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
    let systems: Vec<_> = shapes
        .iter()
        .map(|(_, shape)| testutil::resolve(testutil::one_surface(shape.clone())))
        .collect();
    let backends: Vec<(String, Box<dyn QualificationBackend>)> = {
        let mut v: Vec<(String, Box<dyn QualificationBackend>)> =
            vec![("WebGPU".into(), Box::new(g.clone()))];
        #[cfg(target_os = "macos")]
        if let Some(m) = metal() {
            v.push(("Metal".into(), Box::new(m)));
        }
        v
    };
    // best[backend][shape] = (device, wall)
    let mut best = vec![vec![(f64::INFINITY, f64::INFINITY); shapes.len()]; backends.len()];
    let reqs: Vec<Vec<_>> = backends
        .iter()
        .map(|(_, b)| {
            systems
                .iter()
                .map(|rs| {
                    let h = b
                        .upload_system(&PackedSystem::pack_all(rs).unwrap())
                        .unwrap();
                    let mut req = TraceRequest::new(h, rays.clone(), 0, 1);
                    req.apply_apertures = false;
                    req.radiometry = Radiometry::Off;
                    req
                })
                .collect()
        })
        .collect();
    // Rounds interleave shapes and backends; the first round is a warm-up.
    for round in 0..11 {
        for (k, (name, _)) in shapes.iter().enumerate() {
            for (j, (_, b)) in backends.iter().enumerate() {
                let r = b
                    .run(Job::Trace(Box::new(reqs[j][k].clone())), &JobContext::new())
                    .unwrap();
                let JobOutput::Trace(t) = r.output else {
                    panic!()
                };
                assert!(t.status.iter().all(|s| *s == 0), "{name}");
                if round > 0 {
                    best[j][k].0 = best[j][k].0.min(r.stats.device_seconds);
                    best[j][k].1 = best[j][k].1.min(r.stats.wall_seconds);
                }
            }
        }
    }
    let cpu: Vec<f64> = systems
        .iter()
        .map(|rs| {
            let opts = mklight_trace::TraceOptions::default();
            let mut b = f64::INFINITY;
            for _ in 0..3 {
                let t0 = Instant::now();
                let out: Vec<_> = f64_rays
                    .par_iter()
                    .map(|r| mklight_trace::trace_ray(rs, r, 1, &opts, None))
                    .collect();
                b = b.min(t0.elapsed().as_secs_f64());
                assert_eq!(out.len(), n);
            }
            b
        })
        .collect();
    let mut header = String::from("| shape |");
    let mut rule = String::from("|---|");
    for (name, _) in &backends {
        header += &format!(" {name} device Mrays/s | {name} wall Mrays/s |");
        rule += "---:|---:|";
    }
    eprintln!("{header} CPU f64 rayon Mrays/s |\n{rule}---:|");
    for (k, (name, _)) in shapes.iter().enumerate() {
        let mut row = format!("| {name} |");
        for b in &best {
            row += &format!(
                " {:.1} | {:.1} |",
                n as f64 / b[k].0 / 1e6,
                n as f64 / b[k].1 / 1e6
            );
        }
        eprintln!("{row} {:.1} |", n as f64 / cpu[k] / 1e6);
    }
    for ((_, b), reqs) in backends.iter().zip(reqs) {
        for req in reqs {
            b.release_systems(req.systems).unwrap();
        }
    }
}

/// A steep ray crossing a surface that bulges toward it twice (the first
/// crossing lies farther from the axis), next to ordinary rays: the device
/// matches the emulation bitwise on the conic, even-asphere and general-sag
/// paths for every kind of extent, and takes the root the extent calls for
/// (docs/SURFACES.md §8; the twin of the Metal test).
#[test]
fn steep_double_crossing_bitwise() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let mut rays = RayBatch::with_capacity(3);
    let d = Vec3::new(0.0, 0.999, 0.050).normalized();
    rays.push_f64([0.0, -30.0, -2.0], [d.x, d.y, d.z], 1);
    rays.push_f64([0.0, 2.0, -2.0], [0.0, 0.0, 1.0], 1);
    let d = Vec3::new(0.0, 0.1, 1.0).normalized();
    rays.push_f64([1.0, -3.0, -2.0], [d.x, d.y, d.z], 1);
    for shape in testutil::double_crossing_shapes() {
        for (extent, first) in testutil::double_crossing_extents() {
            let mut sys = testutil::one_surface(shape.clone());
            (sys.surfaces[1].semi_diameter, sys.surfaces[1].aperture) = extent.clone();
            let t = testutil::tracer(sys);
            let p = PackedSystem::pack_all(t.system()).unwrap();
            let req = |h| {
                let mut r = TraceRequest::new(h, rays.clone(), 0, 1);
                r.record = RecordMode::AllSurfaces;
                r.apply_apertures = false;
                r
            };
            let what = format!("{shape:?} {extent:?}");
            let a = run_both(g, &c, &p, &req, &what);
            let y = a.y[0];
            assert!(if first { y < -5.0 } else { y > 3.0 }, "{what}: {y}");
        }
    }
}

/// Light from an object at infinity meeting a deep front surface twice
/// around its start: the device matches the emulation bitwise and takes the
/// first crossing along the line (`mklight-gpu` trace tests).
#[test]
fn steep_ray_from_infinity_bitwise() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let (sys, ray) = testutil::steep_meniscus_from_infinity();
    let t = testutil::tracer(sys);
    let p = PackedSystem::pack_all(t.system()).unwrap();
    let mut rays = RayBatch::with_capacity(2);
    let (a, d) = (ray.pos, ray.dir);
    rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], 1);
    rays.push_f64([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1);
    let end = t.system().image_index() as u32;
    for to in [1, end] {
        let req = |h| {
            let mut r = TraceRequest::new(h, rays.clone(), 0, to);
            r.record = RecordMode::AllSurfaces;
            r
        };
        let (hg, hc) = (g.upload_system(&p).unwrap(), c.upload_system(&p).unwrap());
        let (a, b) = (trace(g, req(hg)), trace(&c, req(hc)));
        assert_bitwise(&a, &b, "steep ray from infinity");
        // Ray 0 at surface 1 (record slots are surface-major).
        let k = 2 * mklight_gpu::batch::RECORD_FLOATS;
        assert!(
            (b.recorded[k + 1] + 29.15).abs() < 0.01,
            "{}",
            b.recorded[k + 1]
        );
        g.release_systems(hg).unwrap();
        c.release_systems(hc).unwrap();
    }
}
