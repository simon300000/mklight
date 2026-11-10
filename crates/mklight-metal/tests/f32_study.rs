//! Pure-f32 versus double-float Metal kernels (docs/GPU.md §3.6): one
//! invariant test, plus ignored measurements against independent CPU f64:
//! `cargo test --release -p mklight-metal --test f32_study -- --ignored --nocapture --test-threads=1`.
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;

use std::time::Instant;

use mklight_core::{JobContext, RayAiming, Vec3, samples};
use mklight_gpu::xval::{XvalSettings, cross_validate};
use mklight_gpu::{
    ComputeBackend, Df32, DfOp, Job, JobOutput, PackedSystem, RayBatch, RecordMode, TraceRequest,
    testutil,
};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions, MetalPrecision};
use mklight_trace::{PupilSampling, Ray, SequentialTracer, TraceOptions};
use rayon::prelude::*;
use serde_json::json;

const MODES: [&str; 3] = ["df", "f32", "cpu_f64"];

/// `[DoubleFloat, PureF32]` from the embedded metallibs; `None` without Metal.
fn backends() -> Option<[MetalBackend; 2]> {
    let make = |precision| {
        MetalBackend::with_options(&MetalOptions {
            precision,
            library: LibraryChoice::Precompiled,
        })
        .unwrap()
    };
    Some([
        make(MetalPrecision::DoubleFloat)?,
        make(MetalPrecision::PureF32)?,
    ])
}

/// All fields and wavelengths on an n×n pupil grid: device input and the
/// matching f64 rays.
fn grid(t: &SequentialTracer, n: usize) -> (RayBatch, Vec<(Ray, usize)>) {
    let mut rays = RayBatch::default();
    let mut reference = Vec::new();
    for f in 0..t.fields().len() {
        for w in 0..t.num_wavelengths() {
            for p in (PupilSampling::Grid { n }).points() {
                if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                    let (a, d) = (g.ray.pos, g.ray.dir);
                    rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                    reference.push((g.ray, w));
                }
            }
        }
    }
    (rays, reference)
}

fn trace_all(b: &MetalBackend, t: &SequentialTracer, rays: RayBatch) -> mklight_gpu::TraceResult {
    let h = b
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let mut req = TraceRequest::new(h, rays, 0, t.system().image_index() as u32);
    req.record = RecordMode::AllSurfaces;
    let r = b
        .run(Job::Trace(Box::new(req)), &JobContext::new())
        .unwrap();
    b.release_systems(h).unwrap();
    let JobOutput::Trace(out) = r.output else {
        panic!("not a trace result")
    };
    *out
}

/// One warm-up and ten measured rounds alternating DF, f32 and CPU f64
/// (order reversed every round); prints median wall and device ms per mode.
/// `run(mode)` returns that run's device seconds (0 for the CPU).
fn time_modes(label: serde_json::Value, mut run: impl FnMut(usize) -> f64) {
    let mut wall = [vec![], vec![], vec![]];
    let mut device = [vec![], vec![], vec![]];
    for round in 0..11 {
        let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for mode in order {
            let start = Instant::now();
            let d = run(mode);
            if round > 0 {
                wall[mode].push(start.elapsed().as_secs_f64() * 1e3);
                device[mode].push(d * 1e3);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for mode in 0..3 {
        println!(
            "F32_TIME {}",
            json!({"case":label,"mode":MODES[mode],"wall_ms":median(&mut wall[mode]),
                "device_ms":median(&mut device[mode])})
        );
    }
}

/// (2^24 + 1) − 2^24 on the device: 1 in double-float, 0 once the residual
/// is dropped.
fn cancellation(b: &MetalBackend) -> f64 {
    let job = Job::DfOps {
        op: DfOp::Add,
        a: vec![Df32::from_f64(16_777_217.)],
        b: vec![Df32::from_f64(-16_777_216.)],
    };
    let JobOutput::Df(v) = b.run(job, &JobContext::new()).unwrap().output else {
        panic!("not a double-float result")
    };
    v[0].to_f64()
}

#[test]
fn pure_f32_drops_residuals() {
    assert_eq!(
        MetalOptions::default().precision,
        MetalPrecision::DoubleFloat
    );
    let Some([df, f32]) = backends() else {
        return eprintln!("SKIP: no Metal device");
    };
    assert!(df.precision().opl_double_float && df.precision().compensated_reductions);
    let p = f32.precision();
    assert!(p.fast_math_disabled && !p.opl_double_float && !p.compensated_reductions);
    assert!(p.kernel_build.contains("pure_f32"), "{}", p.kernel_build);
    assert_eq!(cancellation(&df), 1.);
    assert_eq!(cancellation(&f32), 0.);
    // Trace inputs enter without their lo parts, and no lo part leaves.
    let t = testutil::tracer(samples::cooke_triplet());
    let (rays, _) = grid(&t, 4);
    let mut dropped = rays.clone();
    for plane in [
        &mut dropped.x_lo,
        &mut dropped.y_lo,
        &mut dropped.z_lo,
        &mut dropped.l_lo,
        &mut dropped.m_lo,
        &mut dropped.n_lo,
    ] {
        plane.fill(0.);
    }
    let out = trace_all(&f32, &t, rays);
    assert_eq!(out, trace_all(&f32, &t, dropped));
    assert!(out.opl_lo.iter().all(|v| *v == 0.));
    assert!(out.recorded.as_chunks::<9>().0.iter().all(|v| v[7] == 0.));
}

/// The runtime-source fallback honours the define as well.
#[test]
fn pure_f32_runtime_source() {
    let options = MetalOptions {
        precision: MetalPrecision::PureF32,
        library: LibraryChoice::RuntimeSource,
    };
    let Some(b) = MetalBackend::with_options(&options).unwrap() else {
        return eprintln!("SKIP: no Metal device");
    };
    let build = &b.precision().kernel_build;
    assert!(build.contains("pure_f32 runtime-compiled"), "{build}");
    assert_eq!(cancellation(&b), 0.);
}

/// Per sample and mode: the §5.1 cross-validation (positions, directions,
/// OPD, statuses), all-record OPL at the reported point (CONVENTIONS §12),
/// and real aiming at the default tolerance, retraced to the stop in f64
/// (`failed`: aims the device reports unconverged; residuals include them).
#[test]
#[ignore = "real Metal: accuracy measurement"]
fn trace_and_aiming_accuracy() {
    let backends = backends().expect("real Metal required");
    for (name, model) in samples::all() {
        let t = testutil::tracer(model.clone());
        for (b, mode) in backends.iter().zip(MODES) {
            let mut report =
                cross_validate(b, &t, &XvalSettings::default(), &JobContext::new()).unwrap();
            report.notes.clear();
            let (rays, reference) = grid(&t, 64);
            let out = trace_all(b, &t, rays);
            let mut max_opl = 0f64;
            for (i, (ray, w)) in reference.iter().enumerate() {
                let (_, rec) = mklight_trace::trace_ray_recorded(t.system(), ray, *w, &t.options());
                for (slot, record) in rec.iter().enumerate() {
                    let k = slot * out.len() + i;
                    if out.recorded_status[k] != 0 {
                        continue;
                    }
                    let v = &out.recorded[9 * k..9 * k + 9];
                    let a = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                    let n = t.system().surfaces[slot].n_after[*w];
                    let want = record.opl + n * (a - record.point).dot(record.dir);
                    max_opl = max_opl.max((v[6] as f64 + v[7] as f64 - want).abs());
                }
            }
            println!(
                "F32_TRACE {}",
                json!({"sample":name,"mode":mode,"xval":report,"max_opl_mm":max_opl})
            );
        }
        let mut model = model;
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let points = (PupilSampling::Grid { n: 7 }).points();
        let n = points.len();
        let radius = t.pupil().r_stop;
        for (b, mode) in backends.iter().zip(MODES) {
            let (mut worst, mut worst_accepted, mut failed, mut lost) = (0f64, 0f64, 0, 0);
            for f in 0..t.fields().len() {
                for w in 0..t.num_wavelengths() {
                    let (p, h) = support::make_plan(b, &t, f, w, &points, false, 40, false);
                    let r = b
                        .submit_plan(p, &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let d = support::read_all(b, &r);
                    let (ff, status) = (support::floats(&d[0]), support::uints(&d[1]));
                    let df = |k: usize, i: usize| ff[k * n + i] as f64 + ff[(k + 9) * n + i] as f64;
                    for (i, pt) in points.iter().enumerate() {
                        let ray = Ray::new(
                            Vec3::new(df(0, i), df(1, i), df(2, i)),
                            Vec3::new(df(3, i), df(4, i), df(5, i)),
                        );
                        let stop = mklight_trace::trace_ray(
                            t.system(),
                            &ray,
                            w,
                            &TraceOptions::unclipped().to_surface(t.system().stop),
                            None,
                        );
                        failed += usize::from(status[i] != 0);
                        if !stop.status.is_ok() {
                            lost += 1;
                            continue;
                        }
                        let (x, y) = t.field(f).point.vignetting.apply(pt[0], pt[1]);
                        let residual = (stop.ray.pos.x - x * radius)
                            .hypot(stop.ray.pos.y - y * radius)
                            / radius;
                        worst = worst.max(residual);
                        if status[i] == 0 {
                            worst_accepted = worst_accepted.max(residual);
                        }
                    }
                    b.release_systems(h).unwrap();
                }
            }
            println!(
                "F32_AIM {}",
                json!({"sample":name,"mode":mode,"tolerance":1e-10,"max_residual":worst,
                    "max_accepted_residual":worst_accepted,"failed":failed,"lost_at_stop":lost,
                    "aims":n * t.fields().len() * t.num_wavelengths()})
            );
        }
    }
}

/// Double Gauss trace anchor (10⁴ and 10⁶ rays), then per sample the 64×64
/// all-record trace grid and the 7×7 real-aiming workload.
#[test]
#[ignore = "real Metal: speed measurement"]
fn speed() {
    let backends = backends().expect("real Metal required");
    let t = testutil::tracer(samples::double_gauss());
    let points = (PupilSampling::Grid { n: 650 }).points();
    let opts = t.options();
    for n in [10_000, 1_000_000] {
        let rays: Vec<_> = (0..n)
            .map(|i| {
                let p = points[(i / 3) % points.len()];
                t.generate(t.field(i % 3), 1, p[0], p[1]).unwrap().ray
            })
            .collect();
        let mut batch = RayBatch::with_capacity(n);
        for r in &rays {
            batch.push_f64([r.pos.x, r.pos.y, r.pos.z], [r.dir.x, r.dir.y, r.dir.z], 1);
        }
        let pack = PackedSystem::pack_all(t.system()).unwrap();
        let handles = backends.each_ref().map(|b| b.upload_system(&pack).unwrap());
        let end = t.system().image_index() as u32;
        time_modes(json!({"anchor":"double_gauss","rays":n}), |mode| {
            if mode == 2 {
                let out: Vec<_> = rays
                    .par_iter()
                    .map(|r| mklight_trace::trace_ray(t.system(), r, 1, &opts, None))
                    .collect();
                std::hint::black_box(out);
                return 0.;
            }
            let req = TraceRequest::new(handles[mode], batch.clone(), 0, end);
            let r = backends[mode]
                .run(Job::Trace(Box::new(req)), &JobContext::new())
                .unwrap();
            r.stats.device_seconds
        });
        for (b, h) in backends.iter().zip(handles) {
            b.release_systems(h).unwrap();
        }
    }
    for (name, model) in samples::all() {
        let t = testutil::tracer(model.clone());
        let (rays, reference) = grid(&t, 64);
        let pack = PackedSystem::pack_all(t.system()).unwrap();
        let handles = backends.each_ref().map(|b| b.upload_system(&pack).unwrap());
        let end = t.system().image_index() as u32;
        time_modes(json!({"grid":name,"rays":reference.len()}), |mode| {
            if mode == 2 {
                let out: Vec<_> = reference
                    .par_iter()
                    .map(|(r, w)| {
                        mklight_trace::trace_ray_recorded(t.system(), r, *w, &t.options())
                    })
                    .collect();
                std::hint::black_box(out);
                return 0.;
            }
            let mut req = TraceRequest::new(handles[mode], rays.clone(), 0, end);
            req.record = RecordMode::AllSurfaces;
            let r = backends[mode]
                .run(Job::Trace(Box::new(req)), &JobContext::new())
                .unwrap();
            r.stats.device_seconds
        });
        for (b, h) in backends.iter().zip(handles) {
            b.release_systems(h).unwrap();
        }

        let mut model = model;
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let points = (PupilSampling::Grid { n: 7 }).points();
        let jobs: Vec<_> = (0..t.fields().len())
            .flat_map(|f| (0..t.num_wavelengths()).map(move |w| (f, w)))
            .collect();
        let plans = backends.each_ref().map(|b| {
            jobs.iter()
                .map(|&(f, w)| support::make_plan(b, &t, f, w, &points, false, 40, false))
                .collect::<Vec<_>>()
        });
        time_modes(
            json!({"aiming":name,"rays":jobs.len() * points.len()}),
            |mode| {
                if mode == 2 {
                    let out: Vec<_> = jobs
                        .par_iter()
                        .flat_map(|&(f, w)| {
                            let t = &t;
                            points
                                .par_iter()
                                .map(move |p| t.trace(t.field(f), w, p[0], p[1], &t.options()))
                        })
                        .collect();
                    std::hint::black_box(out);
                    return 0.;
                }
                let b = &backends[mode];
                plans[mode]
                    .iter()
                    .map(|(plan, _)| {
                        let r = b
                            .submit_plan(plan.clone(), &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        std::hint::black_box(support::read_all(b, &r));
                        r.stats.device_seconds
                    })
                    .sum()
            },
        );
        for (b, plans) in backends.iter().zip(plans) {
            for (_, h) in plans {
                b.release_systems(h).unwrap();
            }
        }
    }
}
