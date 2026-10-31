#![cfg(target_os = "macos")]
use mklight_core::{JobContext, MkError, Snapshot};
use mklight_gpu::{resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{PupilSampling, SequentialTracer};
use std::{sync::Arc, time::Instant};

fn backend(library: LibraryChoice) -> MetalBackend {
    let init = Instant::now();
    let b = MetalBackend::with_options(&MetalOptions {
        library,
        ..Default::default()
    })
    .unwrap()
    .expect("P0 qualification requires a host Metal device");
    eprintln!("startup_probe_seconds={:.6}", init.elapsed().as_secs_f64());
    eprintln!(
        "device={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    assert!(b.precision().probe.error_free_transforms());
    b
}
fn rays(t: &SequentialTracer, n: usize) -> RayBatch {
    let mut rays = RayBatch::default();
    for f in 0..t.fields().len() {
        for w in 0..t.num_wavelengths() {
            for p in (PupilSampling::Grid { n }).points() {
                if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                    let (a, d) = (g.ray.pos, g.ray.dir);
                    rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                }
            }
        }
    }
    rays
}
fn key(b: &dyn ComputeBackend, s: &Snapshot) -> SystemCacheKey {
    SystemCacheKey {
        snapshot: s.hash().to_hex(),
        library: "test-library-v1".into(),
        configuration: "base".into(),
        spectrum: "all".into(),
        layout: consts::MKL_LAYOUT_VERSION,
        kernel_build: b.precision().kernel_build.clone(),
        operation_settings: "radiometry-off/full-grid".into(),
    }
}
fn plan(s: Snapshot, req: TraceRequest, keep_trace: bool) -> ResidentPlan {
    let n = req.rays.len();
    let mut p = ResidentPlan::new(s);
    let o = p.trace(req);
    // Only geometry planes here: unnormalised sums, no implicit survival mask.
    let input = PlanView {
        buffer: o.floats,
        offset: 0,
        desc: BufferDesc::new(ElementType::F32, vec![3, n]),
        stride: 1,
    };
    let sum = p.sum(input, ReduceMode::DoubleFloat);
    p.outputs = vec![sum];
    if keep_trace {
        p.outputs
            .extend([o.floats, o.indices, o.records, o.record_status]);
    }
    p
}
fn read(b: &dyn ComputeBackend, h: &BufferHandle) -> BufferData {
    b.readback(h.view(), &JobContext::new()).unwrap().data
}
fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
            "element {i}: {a} != {b}"
        );
    }
}
#[test]
fn both_origins_resident_trace_sum_all_samples() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let g = backend(origin);
        let c = CpuF32Backend::matching(&g.precision().probe);
        let mut max_sum_error = 0f64;
        for (name, sys) in testutil::xval_systems() {
            let snap = Snapshot::new(sys.clone()).unwrap();
            let t = testutil::tracer(sys);
            let packed = PackedSystem::pack_all(t.system()).unwrap();
            let hg = g
                .upload_systems_cached(&key(&g, &snap), std::slice::from_ref(&packed))
                .unwrap();
            let hc = c.upload_system(&packed).unwrap();
            let r = rays(&t, 9);
            let n = r.len();
            if n == 0 {
                panic!("empty {name}");
            }
            let request = |h| {
                let mut q = TraceRequest::new(h, r.clone(), 0, t.system().image_index() as u32);
                q.radiometry = Radiometry::Off;
                q.record = RecordMode::AllSurfaces;
                q
            };
            let ref_req = request(hc);
            let JobOutput::Trace(cpu) = c
                .run(Job::Trace(Box::new(ref_req)), &JobContext::new())
                .unwrap()
                .output
            else {
                panic!()
            };
            let p = plan(snap.clone(), request(hg), true);
            g.set_chunk_items(103);
            let ctx = JobContext::new();
            let pending = g.submit_plan(p.clone(), &ctx).unwrap();
            g.release_systems(hg).unwrap();
            let out = pending.wait().unwrap();
            assert_eq!(ctx.progress.fraction(), 1.);
            assert_eq!(out.stats.readback_bytes, 0);
            assert_eq!(out.stats.fallback_count, 0);
            let BufferData::F32(f) = read(&g, &out.outputs[1]) else {
                panic!()
            };
            let BufferData::U32(u) = read(&g, &out.outputs[2]) else {
                panic!()
            };
            let BufferData::F32(rf) = read(&g, &out.outputs[3]) else {
                panic!()
            };
            let BufferData::U32(ru) = read(&g, &out.outputs[4]) else {
                panic!()
            };
            let gpu = TraceResult::from_planes(&f, &u, n, n, rf, ru);
            for (a, b) in [
                (&gpu.x, &cpu.x),
                (&gpu.y, &cpu.y),
                (&gpu.z, &cpu.z),
                (&gpu.l, &cpu.l),
                (&gpu.m, &cpu.m),
                (&gpu.n, &cpu.n),
                (&gpu.opl_hi, &cpu.opl_hi),
                (&gpu.opl_lo, &cpu.opl_lo),
                (&gpu.weight, &cpu.weight),
                (&gpu.transmittance, &cpu.transmittance),
                (&gpu.recorded, &cpu.recorded),
            ] {
                bits(a, b);
            }
            assert_eq!(gpu.status, cpu.status, "{name}");
            assert_eq!(gpu.last_surface, cpu.last_surface);
            assert_eq!(gpu.steps, cpu.steps);
            assert_eq!(gpu.recorded_status, cpu.recorded_status);
            let BufferData::DoubleFloat(sums) = read(&g, &out.outputs[0]) else {
                panic!()
            };
            for (ch, values) in [&gpu.x, &gpu.y, &gpu.z].iter().enumerate() {
                let expected = reduce::reduce_channel_in::<Ftz>(ReduceMode::DoubleFloat, values);
                if expected.hi.is_nan() {
                    assert!(sums[ch].hi.is_nan());
                } else {
                    assert_eq!(sums[ch], expected, "{name}");
                }
                if values.iter().all(|x| x.is_finite()) {
                    let f64_sum: f64 = values.iter().map(|&x| x as f64).sum();
                    let error = (sums[ch].hi as f64 + sums[ch].lo as f64 - f64_sum).abs();
                    max_sum_error = max_sum_error.max(error);
                    assert!(error < 1e-5, "{name} {error}");
                }
            }
            // Same mathematical tree with different command partition, warm cache.
            let hg2 = g
                .upload_systems_cached(&key(&g, &snap), std::slice::from_ref(&packed))
                .unwrap();
            g.set_chunk_items(65536);
            let warm = g
                .submit_plan(plan(snap, request(hg2), true), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let BufferData::F32(wf) = read(&g, &warm.outputs[1]) else {
                panic!()
            };
            bits(&f, &wf);
            let BufferData::DoubleFloat(ws) = read(&g, &warm.outputs[0]) else {
                panic!()
            };
            for (a, b) in sums.iter().zip(ws) {
                bits(&[a.hi, a.lo], &[b.hi, b.lo]);
            }
            g.release_systems(hg2).unwrap();
            c.release_systems(hc).unwrap();
            eprintln!(
                "sample={name} rays={n} commands={} max_command_ms={:.3}",
                out.stats.command_buffers,
                out.stats
                    .stages
                    .iter()
                    .map(|s| s.max_command_seconds)
                    .fold(0., f64::max)
                    * 1e3
            );
        }
        eprintln!(
            "resident sum max absolute error vs f64 sum of rounded positions={max_sum_error:e} cache={:?}",
            g.resident_cache_stats()
        );
    }
}
#[test]
fn concurrent_cancel_reuse_and_foreign_handles() {
    let g = Arc::new(backend(LibraryChoice::Precompiled));
    let other = backend(LibraryChoice::RuntimeSource);
    let t = testutil::tracer(testutil::singlet());
    let snap = t.system().snapshot.clone();
    let pk = PackedSystem::pack_all(t.system()).unwrap();
    let h = g.upload_system(&pk).unwrap();
    let base = rays(&t, 100);
    let q = TraceRequest::new(h, base, 0, t.system().image_index() as u32);
    let p = plan(snap, q, true);
    let a = g.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let b = g.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let a = a.wait().unwrap();
    let b = b.wait().unwrap();
    let BufferData::F32(af) = read(g.as_ref(), &a.outputs[1]) else {
        panic!()
    };
    let BufferData::F32(bf) = read(g.as_ref(), &b.outputs[1]) else {
        panic!()
    };
    bits(&af, &bf);
    assert!(
        other
            .readback(a.outputs[0].view(), &JobContext::new())
            .is_err()
    );
    let mut foreign = ResidentPlan::new(p.snapshot.clone());
    let id = foreign.import(a.outputs[1].view());
    let sum = foreign.sum(
        PlanView {
            buffer: id,
            offset: 0,
            desc: BufferDesc::new(ElementType::F32, vec![1, af.len()]),
            stride: 1,
        },
        ReduceMode::DoubleFloat,
    );
    foreign.outputs = vec![sum];
    assert!(other.submit_plan(foreign, &JobContext::new()).is_err());
    g.set_chunk_items(64);
    let ctx = JobContext::new();
    let pending = g.submit_plan(p.clone(), &ctx).unwrap();
    let start = Instant::now();
    let mut last = 0.;
    while ctx.progress.fraction() == 0. && !pending.is_ready() {
        std::thread::yield_now();
        assert!(start.elapsed().as_secs() < 30);
    }
    assert!(
        !pending.is_ready(),
        "fixture must cancel during device work"
    );
    for _ in 0..20 {
        let f = ctx.progress.fraction();
        assert!(f >= last);
        last = f;
        std::thread::yield_now();
    }
    pending.cancel();
    assert!(matches!(pending.wait(), Err(MkError::Cancelled)));
    assert!(!ctx.cancel.is_cancelled());
    g.set_chunk_items(65536);
    let r = g
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::F32(f) = read(g.as_ref(), &r.outputs[1]) else {
        panic!()
    };
    bits(&af, &f);
    assert!(r.stats.reused_allocations > 0);
    g.release_systems(h).unwrap();
    assert!(g.submit_plan(p, &JobContext::new()).is_err());
}
#[test]
fn ill_conditioned_resident_reduction() {
    let g = backend(LibraryChoice::Precompiled);
    let v: Vec<f32> = (0..100003)
        .map(|i| match i % 3 {
            0 => 1e8,
            1 => 1.,
            _ => -1e8,
        })
        .collect();
    let exact = v.iter().map(|&x| x as f64).sum::<f64>();
    let h = g
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![1, v.len()]),
            BufferData::F32(v.clone()),
        )
        .unwrap();
    let mut p = ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap());
    let id = p.import(h.view());
    let sum = p.sum(p.view(id).unwrap(), ReduceMode::DoubleFloat);
    p.outputs = vec![sum];
    for chunk in [17, 4096, 1000000] {
        g.set_chunk_items(chunk);
        let r = g
            .submit_plan(p.clone(), &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let BufferData::DoubleFloat(s) = read(&g, &r.outputs[0]) else {
            panic!()
        };
        assert_eq!(s[0].hi as f64 + s[0].lo as f64, exact);
        assert_eq!(
            s[0],
            reduce::reduce_channel_in::<Ftz>(ReduceMode::DoubleFloat, &v)
        );
    }
}
#[test]
fn bounded_snapshot_cache_and_pinned_outputs() {
    let g = backend(LibraryChoice::Precompiled);
    let mut max_bytes = 0u64;
    for i in 0..160 {
        let mut sys = testutil::singlet();
        sys.surfaces[1].shape = mklight_core::SurfaceShape::sphere(50. + i as f64 * 0.01);
        let snap = Snapshot::new(sys.clone()).unwrap();
        let t = testutil::tracer(sys);
        let pk = PackedSystem::pack_all(t.system()).unwrap();
        max_bytes = max_bytes.max(
            (size_of::<PackedHeader>()
                + pk.surfaces.len() * size_of::<PackedSurface>()
                + pk.coefficients.len() * size_of::<Df32>()) as u64,
        );
        let h = g
            .upload_systems_cached(&key(&g, &snap), std::slice::from_ref(&pk))
            .unwrap();
        g.release_systems(h).unwrap();
        if i % 16 == 0 {
            let changed = g
                .upload_systems_cached(&key(&g, &snap), std::slice::from_ref(&pk))
                .unwrap();
            g.release_systems(changed).unwrap();
        }
    }
    let stats = g.resident_cache_stats();
    assert_eq!(stats.system_uploads, 160);
    assert_eq!(stats.system_hits, 10);
    assert!(stats.system_bytes <= 64 * max_bytes);
    eprintln!("snapshot edits: {stats:?}");
}

#[test]
#[ignore = "host benchmark, run alone"]
fn benchmark_resident_chains() {
    use rayon::prelude::*;
    let mut cases: Vec<_> = mklight_core::samples::all()
        .into_iter()
        .map(|(name, sys)| (name, sys, 10_000))
        .collect();
    for (name, sys) in mklight_core::samples::all()
        .into_iter()
        .filter(|(name, _)| {
            [
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
                "xy_freeform",
            ]
            .contains(name)
        })
    {
        for n in [100, 1_000_000] {
            cases.push((name, sys.clone(), n));
        }
    }
    for (name, sys, n) in cases {
        let setup = Instant::now();
        let g = backend(LibraryChoice::Precompiled);
        let setup_s = setup.elapsed().as_secs_f64();
        let snap = Snapshot::new(sys.clone()).unwrap();
        let prep = Instant::now();
        let t = testutil::tracer(sys);
        let grid_side = ((n as f64 * 4.
            / std::f64::consts::PI
            / (t.fields().len() * t.num_wavelengths()) as f64)
            .sqrt()
            .ceil() as usize
            + 2)
        .max(3);
        let base = rays(&t, grid_side);
        let mut batch = RayBatch::with_capacity(n);
        let mut reference = Vec::with_capacity(n);
        for i in 0..n {
            let j = i % base.len();
            let pos = mklight_core::Vec3::new(
                base.x[j] as f64 + base.x_lo[j] as f64,
                base.y[j] as f64 + base.y_lo[j] as f64,
                base.z[j] as f64 + base.z_lo[j] as f64,
            );
            let dir = mklight_core::Vec3::new(
                base.l[j] as f64 + base.l_lo[j] as f64,
                base.m[j] as f64 + base.m_lo[j] as f64,
                base.n[j] as f64 + base.n_lo[j] as f64,
            );
            batch.push_f64(
                [pos.x, pos.y, pos.z],
                [dir.x, dir.y, dir.z],
                base.wavelength[j],
            );
            reference.push((
                mklight_trace::Ray::new(pos, dir),
                base.wavelength[j] as usize,
            ));
        }
        let packed = PackedSystem::pack_all(t.system()).unwrap();
        let prepare = prep.elapsed().as_secs_f64();
        let qualification_timer = Instant::now();
        let mut qualification_settings = mklight_gpu::xval::XvalSettings::qualification();
        qualification_settings.radiometry = Radiometry::Off;
        let qualification =
            mklight_gpu::xval::qualify(&g, &t, Some(&qualification_settings), &JobContext::new())
                .unwrap();
        assert!(qualification.passed, "{name}: {:?}", qualification.reason);
        let qualification_s = qualification_timer.elapsed().as_secs_f64();

        let upload = Instant::now();
        let h = g.upload_systems_cached(&key(&g, &snap), &[packed]).unwrap();
        let upload_s = upload.elapsed().as_secs_f64();
        let req = || {
            let mut q = TraceRequest::new(h, batch.clone(), 0, t.system().image_index() as u32);
            q.radiometry = Radiometry::Off;
            q
        };
        let plan_timer = Instant::now();
        let mut p = plan(snap, req(), false);
        p.identity.library_hash = "test-library-v1".into();
        p.identity.qualification = Some(serde_json::to_value(&qualification).unwrap());
        let plan_prepare_s = plan_timer.elapsed().as_secs_f64();
        let mut records = vec![];
        for round in 0..12 {
            let wall = Instant::now();
            let mut result = g
                .submit_plan(p.clone(), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let rb = g
                .readback(result.outputs[0].view(), &JobContext::new())
                .unwrap();
            result.stats.readback_bytes = rb.copied_bytes;
            let new_wall = wall.elapsed().as_secs_f64();
            let wall = Instant::now();
            let old = g
                .run(Job::Trace(Box::new(req())), &JobContext::new())
                .unwrap();
            let JobOutput::Trace(out) = old.output else {
                panic!()
            };
            let reduce_wall = Instant::now();
            let values = [out.x.as_slice(), out.y.as_slice(), out.z.as_slice()].concat();
            let old_sum = g
                .run(
                    Job::Reduce {
                        mode: ReduceMode::DoubleFloat,
                        count: n,
                        values,
                    },
                    &JobContext::new(),
                )
                .unwrap();
            std::hint::black_box(&old_sum.output);
            let reduction = reduce_wall.elapsed().as_secs_f64();
            let old_wall = wall.elapsed().as_secs_f64();
            let wall = Instant::now();
            let opts = t.options();
            let out: Vec<_> = reference
                .par_iter()
                .map(|(r, w)| mklight_trace::trace_ray(t.system(), r, *w, &opts, None))
                .collect();
            let cpu_sums = [
                out.iter().map(|r| r.ray.pos.x).sum::<f64>(),
                out.iter().map(|r| r.ray.pos.y).sum::<f64>(),
                out.iter().map(|r| r.ray.pos.z).sum::<f64>(),
            ];
            std::hint::black_box(cpu_sums);
            let cpu = wall.elapsed().as_secs_f64();
            let record = serde_json::json!({"sample":name,"rays":n,"pupil_grid_side":grid_side,"round":round,"cold":round==0,"warmup":round==1,"seed":null,"setup_probe_s":setup_s,"prepare_s":prepare,"plan_prepare_s":plan_prepare_s,"qualification_s":qualification_s,"qualification":qualification,"system_upload_s":upload_s,"new_wall_s":new_wall,"new":result.stats,"readback_s":rb.wall_seconds,"old_wall_s":old_wall,"old":old.stats,"old_sum":old_sum.stats,"old_allocations":10,"old_uploaded_bytes":n*80+(t.system().image_index()*4),"old_readback_bytes":n*52+24,"old_host_reduce_s":reduction,"cpu_chain_wall_s":cpu,"cache":g.resident_cache_stats()});
            eprintln!("RAW {record}");
            if round >= 2 {
                records.push((new_wall, old_wall, cpu));
            }
        }
        for (which, ix) in [("resident", 0), ("legacy", 1), ("cpu_chain", 2)] {
            let mut vals: Vec<_> = records.iter().map(|r| [r.0, r.1, r.2][ix]).collect();
            vals.sort_by(f64::total_cmp);
            let median = (vals[4] + vals[5]) / 2.;
            let p95 = vals[9];
            eprintln!(
                "SUMMARY sample={name} rays={n} kind={which} median_ms={:.6} p95_ms={:.6}",
                median * 1e3,
                p95 * 1e3
            );
        }
        g.release_systems(h).unwrap();
    }
}
#[test]
#[ignore = "host benchmark, run alone"]
fn benchmark_snapshot_churn() {
    let g = backend(LibraryChoice::Precompiled);
    let mut rows = vec![];
    for round in 0..12 {
        let before = g.resident_cache_stats();
        let start = Instant::now();
        let mut prepare = 0.;
        let mut upload = 0.;
        for i in 0..160 {
            let timer = Instant::now();
            let mut sys = testutil::singlet();
            sys.surfaces[1].shape = mklight_core::SurfaceShape::sphere(50. + i as f64 * 0.01);
            let snapshot = Snapshot::new(sys.clone()).unwrap();
            let t = testutil::tracer(sys);
            let packed = PackedSystem::pack_all(t.system()).unwrap();
            prepare += timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let k = key(&g, &snapshot);
            let h = g
                .upload_systems_cached(&k, std::slice::from_ref(&packed))
                .unwrap();
            g.release_systems(h).unwrap();
            let warm = g
                .upload_systems_cached(&k, std::slice::from_ref(&packed))
                .unwrap();
            g.release_systems(warm).unwrap();
            upload += timer.elapsed().as_secs_f64();
        }
        let stats = g.resident_cache_stats();
        let wall = start.elapsed().as_secs_f64();
        assert_eq!(stats.system_uploads - before.system_uploads, 160);
        assert_eq!(stats.system_hits - before.system_hits, 160);
        eprintln!(
            "CHURN {}",
            serde_json::json!({"round":round,"changes":160,"prepare_s":prepare,"cache_upload_s":upload,"wall_s":wall,"stats":stats})
        );
        if round >= 2 {
            rows.push(wall);
        }
    }
    rows.sort_by(f64::total_cmp);
    eprintln!(
        "CHURN_SUMMARY changes=160 median_ms={} p95_ms={}",
        (rows[4] + rows[5]) * 500.,
        rows[9] * 1e3
    );
}
