#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p3_cases.rs"]
mod cases;
use mklight_core::{JobContext, MkError};
use mklight_gpu::{collectives::*, resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{PupilSampling, SequentialTracer, TraceOptions};
use rayon::prelude::*;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
fn backend(origin: LibraryChoice) -> MetalBackend {
    let started = Instant::now();
    let b = MetalBackend::with_options(&MetalOptions { library: origin })
        .unwrap()
        .expect("actual host Metal required");
    eprintln!(
        "backend_setup_probe_ms={:.6} build={}",
        started.elapsed().as_secs_f64() * 1e3,
        b.precision().kernel_build
    );
    b
}
fn grid(t: &SequentialTracer, side: usize) -> (RayBatch, Vec<u32>, Vec<Option<[f64; 2]>>, usize) {
    let mut rays = RayBatch::default();
    let mut offsets = vec![0];
    let mut reference = vec![];
    let mut aim_failures = 0;
    for f in 0..t.fields().len() {
        for w in 0..t.num_wavelengths() {
            for p in (PupilSampling::Grid { n: side }).points() {
                if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                    let (a, d) = (g.ray.pos, g.ray.dir);
                    rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                    let hit = t.trace(t.field(f), w, p[0], p[1], &TraceOptions::default());
                    reference.push(hit.status.is_ok().then_some([hit.ray.pos.x, hit.ray.pos.y]));
                } else {
                    aim_failures += 1;
                }
            }
            offsets.push(rays.len() as u32);
        }
    }
    (rays, offsets, reference, aim_failures)
}
fn trace_plan(
    b: &dyn QualificationBackend,
    t: &SequentialTracer,
    h: SystemHandle,
    rays: RayBatch,
    offsets: Vec<u32>,
    keep: bool,
    bound: usize,
) -> ResidentPlan {
    let n = rays.len();
    let s = offsets.len() - 1;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.seed = Some(0);
    p.settings =
        serde_json::json!({"package":"P3","sampling":"supplied host rays","radiometry":"off"});
    let mut req = TraceRequest::new(h, rays, 0, t.system().image_index() as u32);
    req.radiometry = Radiometry::Off;
    let trace = p.trace(req);
    let values = PlanView {
        buffer: trace.floats,
        offset: 0,
        stride: 1,
        desc: BufferDesc::new(ElementType::F32, [2, n]),
    };
    let mask = PlanView {
        buffer: trace.indices,
        offset: 0,
        stride: 1,
        desc: BufferDesc::new(ElementType::U32, [n]),
    };
    let weights = PlanView {
        buffer: trace.floats,
        offset: 8 * n,
        stride: 1,
        desc: BufferDesc::new(ElementType::F32, [n]),
    };
    let offsets = cases::import(b, &mut p, [s + 1], BufferData::U32(offsets));
    let output = cases::scratch(&mut p, ElementType::DoubleFloat, [s, 10]);
    let status = cases::scratch(&mut p, ElementType::U32, [s]);
    p.operations
        .push(ResidentOp::Collectives(Request::SegmentReduce(
            SegmentReduce {
                values,
                offsets,
                mask: Some(mask),
                mask_value: consts::MKL_RAY_OK,
                weights: Some(weights),
                max_segment_len: bound,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: SegmentKind::Moments2,
                output,
                status,
            },
        )));
    p.outputs = vec![output, status];
    if keep {
        p.outputs.extend([trace.floats, trace.indices]);
    }
    p
}
#[test]
fn trace_moments_all_samples_and_fixtures() {
    let g = backend(LibraryChoice::Precompiled);
    let c = CpuF32Backend::matching(&g.precision().probe);
    let (mut max_centroid, mut max_rms) = (0f64, 0f64);
    let (mut status_mismatches, mut aim_failures) = (0, 0);
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let packed = PackedSystem::pack_all(t.system()).unwrap();
        let h = g.upload_system(&packed).unwrap();
        let hc = c.upload_system(&packed).unwrap();
        let (rays, offsets, reference, af) = grid(&t, 9);
        aim_failures += af;
        let n = rays.len();
        let (cpu, _) = cases::read(
            &c,
            trace_plan(&c, &t, hc, rays.clone(), offsets.clone(), false, n),
        );
        let (gpu, _) = cases::read(
            &g,
            trace_plan(&g, &t, h, rays.clone(), offsets.clone(), true, n),
        );
        cases::assert_bits(&cpu[0], &gpu[0]);
        cases::assert_bits(&cpu[1], &gpu[1]);
        let moments = cases::df(&gpu[0]);
        let statuses = cases::uint(&gpu[3]);
        for (i, r) in reference.iter().enumerate() {
            status_mismatches += usize::from(r.is_some() != (statuses[i] == consts::MKL_RAY_OK));
        }
        for (s, range) in offsets.windows(2).enumerate() {
            let pts: Vec<_> = reference[range[0] as usize..range[1] as usize]
                .iter()
                .flatten()
                .collect();
            let m = &moments[s * 10..s * 10 + 10];
            assert_eq!(m[0], pts.len() as f64, "{name}");
            if pts.is_empty() {
                assert!(m[2].is_nan());
                continue;
            }
            let mean = [
                pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64,
                pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64,
            ];
            let rms = (pts
                .iter()
                .map(|p| (p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2))
                .sum::<f64>()
                / pts.len() as f64)
                .sqrt();
            let grms = ((m[5] + m[7]) / m[1]).sqrt();
            let ec = (m[2] - mean[0]).abs().max((m[3] - mean[1]).abs());
            let er = (grms - rms).abs();
            max_centroid = max_centroid.max(ec);
            max_rms = max_rms.max(er);
            assert!(ec < 2e-5 / 3., "{name} centroid {ec:e}");
            assert!(er < (0.005 * rms).max(2e-5) / 3., "{name} RMS {er:e}");
        }
        // Changing the legal segment bound and trace command size must leave
        // every moment bit unchanged even though the dispatch partitions vary.
        let bound = offsets
            .windows(2)
            .map(|p| (p[1] - p[0]) as usize)
            .max()
            .unwrap_or(0);
        g.set_chunk_items(103);
        let (split, _) = cases::read(&g, trace_plan(&g, &t, h, rays, offsets, false, bound));
        cases::assert_bits(&gpu[0], &split[0]);
        g.release_systems(h).unwrap();
        c.release_systems(hc).unwrap();
        eprintln!("sample={name} rays={n} aim_failures={af}");
    }
    eprintln!(
        "f64 max centroid_mm={max_centroid:e} rms_mm={max_rms:e} status_mismatches={status_mismatches} aim_failures={aim_failures}"
    );
    assert_eq!(status_mismatches, 0);
}
/// Masked subnormal cancellation sums, every mode and kind, through the
/// tree-group path (long bound) and the SIMD path (many short segments).
#[test]
fn masked_subnormals_modes_and_views() {
    let g = backend(LibraryChoice::Precompiled);
    let c = CpuF32Backend::matching(&g.precision().probe);
    let n = 15007;
    let long = (vec![0, 0, 257, 4097, n as u32], n);
    let short = (
        (0..n as u32)
            .step_by(37)
            .chain([n as u32])
            .collect::<Vec<_>>(),
        37,
    );
    for (offsets, bound) in [long, short] {
        for mode in [
            ReduceMode::Plain,
            ReduceMode::Kahan,
            ReduceMode::DoubleFloat,
        ] {
            for kind in [
                SegmentKind::Sum,
                SegmentKind::Summary,
                SegmentKind::SourceProducts,
            ] {
                let build = |b: &dyn QualificationBackend| {
                    let values = (0..2 * n).map(|i| [1e8, 1., -1e8, 1e-39][i % 4]).collect();
                    let mut p = cases::segment(
                        b,
                        BufferData::F32(values),
                        2,
                        offsets.clone(),
                        kind,
                        None,
                        bound,
                    );
                    let mask = (0..n).map(|i| u32::from(i % 7 != 0)).collect();
                    let mask = cases::import(b, &mut p, [n], BufferData::U32(mask));
                    let ResidentOp::Collectives(Request::SegmentReduce(r)) = &mut p.operations[0]
                    else {
                        unreachable!()
                    };
                    (r.mode, r.mask, r.mask_value) = (mode, Some(mask), 1);
                    p
                };
                let (a, _) = cases::read(&c, build(&c));
                let (b, _) = cases::read(&g, build(&g));
                for (a, b) in a.iter().zip(&b) {
                    cases::assert_bits(a, b);
                }
            }
        }
    }
}
/// A fit spanning several command buffers (20000 x 120).
fn long_fit(g: &dyn QualificationBackend) -> ResidentPlan {
    let (m, n) = (20000, 120);
    let a: Vec<_> = (0..m)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if j == 0 {
                    1.
                } else {
                    ((i as f64 + 0.5) * j as f64 * std::f64::consts::PI / m as f64).cos()
                }
            })
        })
        .collect();
    let rhs: Vec<_> = (0..m).map(|i| a[i * n] + 0.2 * a[i * n + 1]).collect();
    cases::qr_plan(g, m, n, 1, a, rhs, 1e-10)
}
#[test]
fn concurrent_cancellation_progress_and_reuse() {
    let g = Arc::new(backend(LibraryChoice::Precompiled));
    let p = long_fit(&*g);
    let ctx = JobContext::new();
    let pending = g.submit_plan(p.clone(), &ctx).unwrap();
    let start = Instant::now();
    let mut previous = 0.;
    while !pending.is_ready() {
        let f = ctx.progress.fraction();
        assert!(f >= previous);
        previous = f;
        if f > 0.0 && f < 1.0 {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_micros(200));
    }
    assert!(
        previous > 0. && previous < 1.,
        "must observe device-work progress before cancellation"
    );
    pending.cancel();
    assert!(matches!(pending.wait(), Err(MkError::Cancelled)));
    // Concurrent jobs reproduce every bit.
    let a = g.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let b = g.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let (a, b) = (a.wait().unwrap(), b.wait().unwrap());
    for (a, b) in a.outputs.iter().zip(&b.outputs) {
        let aa = g.readback(a.view(), &JobContext::new()).unwrap();
        let bb = g.readback(b.view(), &JobContext::new()).unwrap();
        cases::assert_bits(&aa.data, &bb.data);
    }
    let (v, warm) = cases::read(&*g, p);
    assert!(warm.stats.reused_allocations > 0);
    assert!(warm.stats.command_buffers > 1);
    assert!(
        warm.stats
            .stages
            .iter()
            .any(|s| s.max_command_seconds > 0. && s.max_command_seconds < 0.05)
    );
    assert_eq!(cases::df(&v[2])[0], 120.);
    assert!((cases::df(&v[1])[0] - 1.).abs() < 1e-10);
    let mut bad = cases::scan_case(&*g, 7, true, true);
    let ResidentOp::Collectives(Request::Scan(r)) = &mut bad.operations[0] else {
        unreachable!()
    };
    r.values.desc.element = ElementType::Complex32;
    assert!(g.submit_plan(bad, &JobContext::new()).is_err());
}
#[test]
fn small_batched_qr() {
    let g = backend(LibraryChoice::Precompiled);
    let mut first_factors = None;
    for b in [1, 16, 256, 1024] {
        let (m, n) = (9, 3);
        let a: Vec<_> = (0..b)
            .flat_map(|_| {
                (0..m).flat_map(|i| {
                    let x = i as f64 / 8.;
                    [1., x, x * x]
                })
            })
            .collect();
        let rhs: Vec<_> = (0..b)
            .flat_map(|z| {
                (0..m).map(move |i| {
                    1. + z as f64 * 0.01 + 2. * i as f64 / 8. - 0.3 * (i as f64 / 8.).powi(2)
                })
            })
            .collect();
        let (v, _) = cases::read(&g, cases::qr_plan(&g, m, n, b, a, rhs, 1e-10));
        let factor_words = v[0].words()[..(m + n + 2) * n * 2].to_vec();
        if let Some(ref first) = first_factors {
            assert_eq!(first, &factor_words);
        } else {
            first_factors = Some(factor_words);
        }
        let x = cases::df(&v[1]);
        for z in 0..b {
            assert!((x[z * n] - 1. - z as f64 * 0.01).abs() < 1e-10);
            assert!((x[z * n + 1] - 2.).abs() < 1e-10);
            assert!((x[z * n + 2] + 0.3).abs() < 1e-10);
        }
    }
}
fn bench_rays(t: &SequentialTracer, n: usize) -> (RayBatch, Vec<u32>) {
    let fw = t.fields().len() * t.num_wavelengths();
    let mut rays = RayBatch::with_capacity(n);
    let mut offsets = vec![0];
    for f in 0..t.fields().len() {
        for w in 0..t.num_wavelengths() {
            let j = f * t.num_wavelengths() + w;
            let count = n / fw + usize::from(j < n % fw);
            for i in 0..count {
                let r = 0.95 * ((i as f64 + 0.5) / count as f64).sqrt();
                let a = i as f64 * 2.399963229728653;
                let g = t.generate(t.field(f), w, r * a.cos(), r * a.sin()).unwrap();
                let (p, d) = (g.ray.pos, g.ray.dir);
                rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], w as u32);
            }
            offsets.push(rays.len() as u32);
        }
    }
    (rays, offsets)
}
fn host_moments(pts: &[[f64; 2]], offsets: &[u32]) -> Vec<[f64; 10]> {
    use mklight_core::math::KahanSum;
    offsets
        .windows(2)
        .map(|range| {
            let points = pts[range[0] as usize..range[1] as usize]
                .iter()
                .filter(|p| p[0].is_finite());
            let (mut sw, mut sx, mut sy) = (
                KahanSum::default(),
                KahanSum::default(),
                KahanSum::default(),
            );
            let mut count = 0;
            for point in points.clone() {
                sw.add(1.0);
                sx.add(point[0]);
                sy.add(point[1]);
                count += 1;
            }
            let weight = sw.value();
            let mx = sx.value() / weight;
            let my = sy.value() / weight;
            let (mut sr, mut xx, mut xy, mut yy) = (
                KahanSum::default(),
                KahanSum::default(),
                KahanSum::default(),
                KahanSum::default(),
            );
            let (mut geo_origin, mut geo_centroid) = (0f64, 0f64);
            for point in points {
                let (dx, dy) = (point[0] - mx, point[1] - my);
                sr.add(point[0].mul_add(point[0], point[1] * point[1]));
                xx.add(dx * dx);
                xy.add(dx * dy);
                yy.add(dy * dy);
                geo_origin = geo_origin.max(point[0].hypot(point[1]));
                geo_centroid = geo_centroid.max(dx.hypot(dy));
            }
            [
                count as f64,
                weight,
                mx,
                my,
                sr.value(),
                xx.value(),
                xy.value(),
                yy.value(),
                geo_origin * geo_origin,
                geo_centroid * geo_centroid,
            ]
        })
        .collect()
}
fn reference_moments(t: &SequentialTracer, rays: &RayBatch, offsets: &[u32]) -> Vec<[f64; 10]> {
    use mklight_core::Vec3;
    let pts: Vec<_> = (0..rays.len())
        .into_par_iter()
        .map(|i| {
            let v = |hi: &[f32], lo: &[f32]| hi[i] as f64 + lo[i] as f64;
            let r = mklight_trace::Ray::new(
                Vec3::new(
                    v(&rays.x, &rays.x_lo),
                    v(&rays.y, &rays.y_lo),
                    v(&rays.z, &rays.z_lo),
                ),
                Vec3::new(
                    v(&rays.l, &rays.l_lo),
                    v(&rays.m, &rays.m_lo),
                    v(&rays.n, &rays.n_lo),
                ),
            );
            let hit = mklight_trace::trace_ray(
                t.system(),
                &r,
                rays.wavelength[i] as usize,
                &TraceOptions::default(),
                None,
            );
            if hit.status.is_ok() {
                [hit.ray.pos.x, hit.ray.pos.y]
            } else {
                [f64::NAN; 2]
            }
        })
        .collect();
    host_moments(&pts, offsets)
}
fn report_bench(label: &str, runs: &[(f64, f64)], cold: f64) {
    let mut walls: Vec<_> = runs.iter().map(|r| r.0).collect();
    let mut device: Vec<_> = runs.iter().map(|r| r.1).collect();
    walls.sort_by(f64::total_cmp);
    device.sort_by(f64::total_cmp);
    eprintln!(
        "{}",
        serde_json::json!({"summary":label,"cold_ms":cold*1e3,"wall_median_ms":walls[walls.len()/2]*1e3,"wall_p95_ms":walls[(walls.len()*95).div_ceil(100)-1]*1e3,"device_median_ms":device[device.len()/2]*1e3,"device_p95_ms":device[(device.len()*95).div_ceil(100)-1]*1e3})
    );
}
#[test]
#[ignore = "host benchmark: cold + warm-up + ten matched runs"]
fn benchmark_trace_collectives() {
    let g = backend(LibraryChoice::Precompiled);
    eprintln!(
        "{}",
        serde_json::json!({"hardware":g.info(),"precision":g.precision(),"seed":0,"iterations":10,"warmup":1,"reference_workers":rayon::current_num_threads(),"build":"release","scope":"P3 resident trace + moments; CPU-generated launch rays, no engine dispatch"})
    );
    let mut workloads: Vec<_> = mklight_core::samples::all()
        .into_iter()
        .map(|(name, sys)| (format!("default/{name}"), sys, 0usize))
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
                "anamorphic",
            ]
            .contains(name)
        })
    {
        for n in [100, 10_000, 1_000_000] {
            workloads.push((format!("sweep/{name}"), sys.clone(), n));
        }
    }
    for (name, sys, n) in workloads {
        let setup = Instant::now();
        let t = testutil::tracer(sys);
        let packed = PackedSystem::pack_all(t.system()).unwrap();
        let h = g.upload_system(&packed).unwrap();
        let setup_seconds = setup.elapsed().as_secs_f64();
        let generation_at = Instant::now();
        let (rays, offsets) = if n == 0 {
            let mut rays = RayBatch::default();
            let mut offsets = vec![0];
            for f in 0..t.fields().len() {
                for w in 0..t.num_wavelengths() {
                    for point in (PupilSampling::Hexapolar { rings: 6 }).points() {
                        let launch = t.generate(t.field(f), w, point[0], point[1]).unwrap();
                        let (pos, dir) = (launch.ray.pos, launch.ray.dir);
                        rays.push_f64([pos.x, pos.y, pos.z], [dir.x, dir.y, dir.z], w as u32);
                    }
                    offsets.push(rays.len() as u32);
                }
            }
            (rays, offsets)
        } else {
            bench_rays(&t, n)
        };
        let n = rays.len();
        let generation = generation_at.elapsed().as_secs_f64();
        eprintln!(
            "{}",
            serde_json::json!({"workload":name,"rays":n,"setup_ms":setup_seconds*1e3,"generation_ms":generation*1e3,"offsets":offsets,"sampling":if name.starts_with("default/"){ "hexapolar rings=6; all fields/wavelengths"}else{"deterministic spiral, radius 0.95; all fields/wavelengths"},"packed_surfaces":t.system().surfaces.len()})
        );
        let mut cpu = vec![];
        let mut old = vec![];
        let mut new = vec![];
        let mut cold = [0.; 3];
        let mut ref_value = vec![];
        for run in 0..12 {
            let phase = if run == 0 {
                "cold"
            } else if run == 1 {
                "warmup"
            } else {
                "measured"
            };
            let wall = Instant::now();
            let ref_result = reference_moments(&t, &rays, &offsets);
            let cpu_wall = wall.elapsed().as_secs_f64();
            std::hint::black_box(&ref_result);
            if run == 0 {
                ref_value = ref_result;
            }
            let wall = Instant::now();
            let mut q = TraceRequest::new(h, rays.clone(), 0, t.system().image_index() as u32);
            q.radiometry = Radiometry::Off;
            let old_out = g.run(Job::Trace(Box::new(q)), &JobContext::new()).unwrap();
            let old_device = old_out.stats.device_seconds;
            let JobOutput::Trace(tr) = old_out.output else {
                unreachable!()
            };
            let reduce_at = Instant::now();
            let pts: Vec<_> = (0..n)
                .map(|i| {
                    if tr.status[i] == consts::MKL_RAY_OK {
                        [tr.x[i] as f64, tr.y[i] as f64]
                    } else {
                        [f64::NAN; 2]
                    }
                })
                .collect();
            let old_result = host_moments(&pts, &offsets);
            let reduction = reduce_at.elapsed().as_secs_f64();
            std::hint::black_box(&old_result);
            let old_wall = wall.elapsed().as_secs_f64();
            let wall = Instant::now();
            let bound = offsets
                .windows(2)
                .map(|r| (r[1] - r[0]) as usize)
                .max()
                .unwrap_or(0);
            let p = trace_plan(&g, &t, h, rays.clone(), offsets.clone(), false, bound);
            let prepare = wall.elapsed().as_secs_f64();
            let stats = g
                .submit_plan(p, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let moments: Vec<f64> = (g.read_view(stats.outputs[0].view()).unwrap().words())
                .chunks(2)
                .map(|d| f32::from_bits(d[0]) as f64 + f32::from_bits(d[1]) as f64)
                .collect();
            let new_wall = wall.elapsed().as_secs_f64();
            let mut error = 0f64;
            let mut survival_count_mismatched_segments = 0;
            for (j, want) in ref_value.iter().enumerate() {
                let a = &moments[j * 10..j * 10 + 10];
                for (got, want) in [a[2], a[3], ((a[5] + a[7]) / a[1]).sqrt()]
                    .into_iter()
                    .zip([want[2], want[3], ((want[5] + want[7]) / want[1]).sqrt()])
                {
                    if want.is_finite() {
                        error = error.max((got - want).abs());
                    }
                }
                survival_count_mismatched_segments += usize::from(a[0] != want[0]);
            }
            eprintln!(
                "{}",
                serde_json::json!({"workload":name,"rays":n,"run":run,"phase":phase,"cpu_f64_ms":cpu_wall*1e3,"old_wall_ms":old_wall*1e3,"old_device_ms":old_device*1e3,"old_reduce_pack_ms":reduction*1e3,"new_wall_ms":new_wall*1e3,"read_view_bytes":moments.len()*8,"run_info":if run==0{Some(&stats.run_info)}else{None},"new_prepare_ms":prepare*1e3,"new_device_ms":stats.stats.device_seconds*1e3,"new_stats":stats.stats,"old_readback_bytes":n*52,"new_readback_bytes":offsets.len().saturating_sub(1)*84,"max_aggregate_error_mm":error,"survival_count_mismatched_segments":survival_count_mismatched_segments})
            );
            if run == 0 {
                cold = [cpu_wall, old_wall, new_wall];
            } else if run >= 2 {
                cpu.push((cpu_wall, 0.));
                old.push((old_wall, old_device));
                new.push((new_wall, stats.stats.device_seconds));
            }
        }
        report_bench(&format!("{name}/{n}/cpu_f64"), &cpu, cold[0]);
        report_bench(&format!("{name}/{n}/old_metal_host_reduce"), &old, cold[1]);
        report_bench(&format!("{name}/{n}/resident"), &new, cold[2]);
        g.release_systems(h).unwrap();
    }
}
fn median_p95(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() * 95).div_ceil(100) - 1])
}
/// Cold run, one warm-up, then ten measured runs of `f` (seconds, device seconds).
// Complete matched pairs, with one cold pair, warm-up and ten measurements.
fn bench_pair(
    g: &MetalBackend,
    p: &ResidentPlan,
    mut cpu: impl FnMut(),
) -> (serde_json::Value, serde_json::Value) {
    let mut cpu_raw = vec![];
    let mut gpu_raw = vec![];
    let mut device = vec![];
    let mut cold = (0., 0.);
    let mut stats = None;
    for run in 0..12 {
        let mut host = || {
            let t = Instant::now();
            cpu();
            t.elapsed().as_secs_f64()
        };
        let mut metal = || {
            let t = Instant::now();
            let out = g
                .submit_plan(p.clone(), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let (mut sum, mut bytes) = (0u32, 0usize);
            for h in &out.outputs {
                let v = g.read_view(h.view()).unwrap();
                sum = v.words().iter().fold(sum, |a, &b| a.wrapping_add(b));
                bytes += v.words().len() * 4;
            }
            std::hint::black_box(sum);
            let wall = t.elapsed().as_secs_f64();
            stats = Some((out.stats.clone(), bytes));
            (wall, out.stats.device_seconds)
        };
        let (c, (w, d)) = if run % 2 == 0 {
            let c = host();
            (c, metal())
        } else {
            let m = metal();
            (host(), m)
        };
        if run == 0 {
            cold = (c, w);
        } else if run > 1 {
            cpu_raw.push(c);
            gpu_raw.push(w);
            device.push(d);
        }
    }
    let cpu = median_p95(cpu_raw.clone());
    let gpu = median_p95(gpu_raw.clone());
    let dev = median_p95(device.clone());
    let (s, bytes) = stats.unwrap();
    (
        serde_json::json!({"cpu_ms":cpu.0*1e3,"cpu_p95_ms":cpu.1*1e3,"cpu_cold_ms":cold.0*1e3,"cpu_raw_seconds":cpu_raw}),
        serde_json::json!({"gpu_ms":gpu.0*1e3,"gpu_p95_ms":gpu.1*1e3,"gpu_device_ms":dev.0*1e3,"gpu_cold_ms":cold.1*1e3,"gpu_raw_seconds":gpu_raw,"device_raw_seconds":device,"commands":s.command_buffers,"peak_bytes":s.peak_bytes,"read_view_bytes":bytes,"copied_bytes":s.copied_bytes}),
    )
}
fn report(workload: &str, size: &str, (cpu, gpu): (serde_json::Value, serde_json::Value)) {
    let mut v = serde_json::json!({"workload":workload,"size":size});
    for part in [cpu, gpu] {
        v.as_object_mut()
            .unwrap()
            .extend(part.as_object().unwrap().clone());
    }
    eprintln!("{v}");
}
fn uniform(n: usize, stream: u64) -> Vec<f64> {
    let mut r = mklight_core::rng::Philox::new(0, stream);
    (0..n).map(|_| r.uniform()).collect()
}
#[test]
#[ignore = "host benchmark: warm medians of resident collectives versus CPU f64"]
fn benchmark_collectives() {
    let g = backend(LibraryChoice::Precompiled);
    eprintln!(
        "{}",
        serde_json::json!({"hardware":g.info(),"precision":g.precision(),"seed":0,"runs":"cold, warm-up, 10 measured","cpu":"serial f64 host reference"})
    );
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let size = n.to_string();
        // Spot moments: x, y, weight in nine field/wavelength segments, 5% vignetted.
        let (x, y, w) = (uniform(n, 1), uniform(n, 2), uniform(n, 3));
        let keep: Vec<bool> = (0..n).map(|i| i % 20 != 7).collect();
        let offsets: Vec<u32> = (0..=9).map(|s| (s * n / 9) as u32).collect();
        let mut p = cases::segment(
            &g,
            BufferData::F32(
                x.iter()
                    .chain(&y)
                    .chain(&w)
                    .map(|&v| v as f32 * 10.)
                    .collect(),
            ),
            3,
            offsets.clone(),
            SegmentKind::Moments2,
            None,
            n.div_ceil(9),
        );
        let mask = cases::import(
            &g,
            &mut p,
            [n],
            BufferData::U32(keep.iter().map(|&k| u32::from(k)).collect()),
        );
        let ResidentOp::Collectives(Request::SegmentReduce(r)) = &mut p.operations[0] else {
            unreachable!()
        };
        r.mask = Some(mask);
        r.mask_value = 1;
        let cpu = || {
            let pts: Vec<_> = (0..n)
                .map(|i| {
                    if keep[i] {
                        [x[i] * 10., y[i] * 10.]
                    } else {
                        [f64::NAN; 2]
                    }
                })
                .collect();
            std::hint::black_box(host_moments(&pts, &offsets));
        };
        report("moments_9_segments", &size, bench_pair(&g, &p, cpu));

        // Many short segments (detector bins / split children): length 17.
        let values: Vec<f64> = uniform(n, 4);
        let offsets: Vec<u32> = (0..n)
            .step_by(17)
            .chain(std::iter::once(n))
            .map(|i| i as u32)
            .collect();
        let p = cases::segment(
            &g,
            BufferData::F32(values.iter().map(|&v| v as f32).collect()),
            1,
            offsets.clone(),
            SegmentKind::Summary,
            None,
            17,
        );
        let cpu = || {
            let out: Vec<_> = offsets
                .windows(2)
                .map(|r| {
                    let c = &values[r[0] as usize..r[1] as usize];
                    let sum = c.iter().copied().collect::<mklight_core::math::KahanSum>();
                    let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    [sum.value(), c.len() as f64, lo, hi]
                })
                .collect();
            std::hint::black_box(out);
        };
        report("summary_short_segments_17", &size, bench_pair(&g, &p, cpu));

        // Radius sort with logical-ID ties.
        let keys: Vec<f32> = uniform(n, 5).iter().map(|&v| v as f32).collect();
        let ids: Vec<u32> = (0..n as u32).map(|i| i % 97).collect();
        let mut p = cases::plan();
        let k = cases::import(&g, &mut p, [n], BufferData::F32(keys.clone()));
        let i = cases::import(&g, &mut p, [n], BufferData::U32(ids.clone()));
        let ok = cases::scratch(&mut p, ElementType::F32, [n]);
        let oi = cases::scratch(&mut p, ElementType::U32, [n]);
        p.operations
            .push(ResidentOp::Collectives(Request::StableSort(StableSort {
                keys: k,
                ids: i,
                values: None,
                output_keys: ok,
                output_ids: oi,
                output_values: None,
            })));
        p.outputs = vec![ok, oi];
        let cpu = || {
            let mut order: Vec<u32> = (0..n as u32).collect();
            order.sort_by(|&a, &b| {
                keys[a as usize]
                    .total_cmp(&keys[b as usize])
                    .then(ids[a as usize].cmp(&ids[b as usize]))
            });
            let out: (Vec<_>, Vec<_>) = order
                .iter()
                .map(|&j| (keys[j as usize], ids[j as usize]))
                .unzip();
            std::hint::black_box(out);
        };
        report("sort_f32_keys", &size, bench_pair(&g, &p, cpu));

        // Inclusive f32 scan (energy curves) published as double-float.
        let mut p = cases::plan();
        let v = cases::import(&g, &mut p, [n], BufferData::F32(keys.clone()));
        let o = cases::scratch(&mut p, ElementType::DoubleFloat, [n]);
        p.operations
            .push(ResidentOp::Collectives(Request::Scan(Scan {
                values: v,
                inclusive: true,
                output: o,
            })));
        p.outputs = vec![o];
        let cpu = || {
            let mut s = 0f64;
            let out: Vec<f64> = keys
                .iter()
                .map(|&v| {
                    s += v as f64;
                    s
                })
                .collect();
            std::hint::black_box(out);
        };
        report("scan_f32", &size, bench_pair(&g, &p, cpu));

        // Stable compaction of [n,2] double-float records, two thirds kept.
        let mut p = cases::plan();
        let records: Vec<Df32> = uniform(2 * n, 6).into_iter().map(Df32::from_f64).collect();
        let mask: Vec<u32> = (0..n as u32).map(|i| u32::from(i % 3 != 0)).collect();
        let v = cases::import(&g, &mut p, [n, 2], BufferData::DoubleFloat(records.clone()));
        let ids = cases::import(&g, &mut p, [n], BufferData::U32((0..n as u32).collect()));
        let m = cases::import(&g, &mut p, [n], BufferData::U32(mask.clone()));
        let out = cases::scratch(&mut p, ElementType::DoubleFloat, [n, 2]);
        let oi = cases::scratch(&mut p, ElementType::U32, [n]);
        let cnt = cases::scratch(&mut p, ElementType::U32, [1]);
        p.operations
            .push(ResidentOp::Collectives(Request::StableCompact(
                StableCompact {
                    values: v,
                    ids,
                    mask: m,
                    output: out,
                    output_ids: oi,
                    count: cnt,
                },
            )));
        p.outputs = vec![out, oi, cnt];
        let cpu = || {
            let mut out = Vec::with_capacity(2 * n);
            let mut oi = Vec::with_capacity(n);
            for i in 0..n {
                if mask[i] != 0 {
                    out.extend_from_slice(&records[2 * i..2 * i + 2]);
                    oi.push(i as u32);
                }
            }
            std::hint::black_box((out, oi));
        };
        report("compact_df_records", &size, bench_pair(&g, &p, cpu));
    }
    // Tall-skinny and batched least squares; known coefficients.
    for (m, n, batch) in [
        (200, 20, 1),
        (200, 20, 256),
        (2000, 60, 1),
        (4225, 37, 1),
        (20000, 120, 1),
        (65536, 37, 1),
    ] {
        let a: Vec<f64> = (0..batch)
            .flat_map(|z| {
                (0..m).flat_map(move |i| {
                    (0..n).map(move |j| {
                        let t = (i as f64 + 0.5) / m as f64;
                        if j == 0 {
                            1.
                        } else {
                            (t * j as f64 * std::f64::consts::PI + z as f64 * 1e-3).cos()
                        }
                    })
                })
            })
            .collect();
        let rhs: Vec<f64> = (0..batch * m)
            .map(|r| (0..n).map(|j| a[r * n + j] * (j as f64 + 1.) / 100.).sum())
            .collect();
        let mut p = cases::qr_plan(&g, m, n, batch, a.clone(), rhs.clone(), 1e-10);
        p.outputs.remove(0);
        let size = format!("{m}x{n}x{batch}");
        let cpu = || {
            for z in 0..batch {
                std::hint::black_box(cases::reference_qr(
                    &a[z * m * n..(z + 1) * m * n],
                    &rhs[z * m..(z + 1) * m],
                    m,
                    n,
                ));
            }
        };
        if g.submit_plan(p.clone(), &JobContext::new()).is_err() {
            eprintln!(
                "{}",
                serde_json::json!({"workload":"qr","size":size,"gpu":"unsupported"})
            );
            continue;
        }
        let (v, _) = cases::read(&g, p.clone());
        let x = cases::df(&v[0]);
        let err = (0..batch * n)
            .map(|i| (x[i] - ((i % n) as f64 + 1.) / 100.).abs())
            .fold(0f64, f64::max);
        let (cpu, mut gpu) = bench_pair(&g, &p, cpu);
        gpu["coefficient_max_error"] = err.into();
        report("qr", &size, (cpu, gpu));
    }
}
