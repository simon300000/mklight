//! Explicit, real-device precision experiments. CPU emulation remains the DF spec.
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;

use mklight_core::{JobContext, RayAiming, Vec3, samples};
use mklight_gpu::{
    ComputeBackend, Df32, DfOp, Job, JobOutput, PackedSystem, RayBatch, RecordMode, TraceRequest,
    testutil,
    xval::{XvalSettings, cross_validate},
};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions, MetalPrecision};
use mklight_trace::{PupilSampling, Ray, TraceOptions};
use serde_json::json;

fn backend(precision: MetalPrecision, library: LibraryChoice) -> MetalBackend {
    MetalBackend::with_options(&MetalOptions {
        precision,
        library,
        ..Default::default()
    })
    .unwrap()
    .expect("precision study requires real Metal")
}

#[test]
fn pure_f32_drops_residuals_and_keeps_strict_flags() {
    if !mklight_metal::is_available() {
        return;
    }
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = backend(MetalPrecision::PureF32, library);
        assert!(b.precision().fast_math_disabled);
        assert!(!b.precision().opl_double_float);
        assert!(!b.precision().compensated_reductions);
        assert!(b.precision().kernel_build.contains("pure_f32"));
        // Cancellation leaves 1 in DF and 0 in f32, independently known.
        let x = Df32::from_f64(16_777_217.0);
        let r = b
            .run(
                Job::DfOps {
                    op: DfOp::Add,
                    a: vec![x],
                    b: vec![Df32::from_f64(-16_777_216.)],
                },
                &JobContext::new(),
            )
            .unwrap();
        let JobOutput::Df(v) = r.output else { panic!() };
        assert_eq!(v[0].to_f64(), 0.);
        let t = testutil::tracer(samples::cooke_triplet());
        let h = b
            .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
            .unwrap();
        let g = t.generate(t.field(1), 1, 0.3, 0.2).unwrap();
        let mut rays = RayBatch::default();
        rays.push_f64(
            [g.ray.pos.x, g.ray.pos.y, g.ray.pos.z],
            [g.ray.dir.x, g.ray.dir.y, g.ray.dir.z],
            1,
        );
        let mut req = TraceRequest::new(h, rays.clone(), 0, t.system().image_index() as u32);
        req.record = RecordMode::AllSurfaces;
        let first = b
            .run(Job::Trace(Box::new(req.clone())), &JobContext::new())
            .unwrap();
        let repeated = b
            .run(Job::Trace(Box::new(req.clone())), &JobContext::new())
            .unwrap();
        assert_eq!(first.output, repeated.output);
        for plane in [
            &mut rays.x_lo,
            &mut rays.y_lo,
            &mut rays.z_lo,
            &mut rays.l_lo,
            &mut rays.m_lo,
            &mut rays.n_lo,
        ] {
            plane.fill(0.);
        }
        req.rays = rays;
        let dropped = b
            .run(Job::Trace(Box::new(req)), &JobContext::new())
            .unwrap();
        assert_eq!(first.output, dropped.output);
        let JobOutput::Trace(out) = first.output else {
            panic!()
        };
        assert!(out.opl_lo.iter().all(|v| *v == 0.));
        assert!(out.recorded.as_chunks::<9>().0.iter().all(|v| v[7] == 0.));
        b.release_systems(h).unwrap();
    }
    assert_eq!(
        MetalOptions::default().precision,
        MetalPrecision::DoubleFloat
    );
}

#[test]
#[ignore = "real Metal: all-sample accuracy evidence"]
fn trace_and_aiming_accuracy() {
    let backends = [
        backend(MetalPrecision::DoubleFloat, LibraryChoice::Precompiled),
        backend(MetalPrecision::PureF32, LibraryChoice::Precompiled),
    ];
    for b in &backends {
        println!(
            "F32_ENV {}",
            json!({"device":b.info(),"precision":b.precision()})
        );
    }
    for (name, model) in samples::all() {
        let t = testutil::tracer(model.clone());
        for b in &backends {
            let report =
                cross_validate(b, &t, &XvalSettings::default(), &JobContext::new()).unwrap();
            if b.precision().opl_double_float {
                report.check(&b.precision().budget).unwrap();
            }
            println!(
                "F32_TRACE {}",
                json!({"sample":name,"precision":b.precision(),"report":report})
            );
            // OPL compared at the reported point (CONVENTIONS §12), all records.
            let h = b
                .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
                .unwrap();
            let mut rays = RayBatch::default();
            let mut reference = Vec::new();
            for f in 0..t.fields().len() {
                for w in 0..t.num_wavelengths() {
                    for p in (PupilSampling::Grid { n: 64 }).points() {
                        if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                            let a = g.ray.pos;
                            let d = g.ray.dir;
                            rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                            reference.push((g.ray, w));
                        }
                    }
                }
            }
            let mut req = TraceRequest::new(h, rays, 0, t.system().image_index() as u32);
            req.record = RecordMode::AllSurfaces;
            let r = b
                .run(Job::Trace(Box::new(req)), &JobContext::new())
                .unwrap();
            let JobOutput::Trace(out) = r.output else {
                panic!()
            };
            let mut max_opl = 0f64;
            let mut compared = 0;
            for (i, (ray, w)) in reference.iter().enumerate() {
                let (_, rec) = mklight_trace::trace_ray_recorded(t.system(), ray, *w, &t.options());
                for (slot, record) in rec.iter().enumerate() {
                    let offset = slot * out.len() + i;
                    if out.recorded_status[offset] != 0 {
                        continue;
                    }
                    let v = &out.recorded[offset * 9..offset * 9 + 9];
                    let a = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                    let index = t.system().surfaces[slot].n_after[*w];
                    let expected = record.opl + index * (a - record.point).dot(record.dir);
                    max_opl = max_opl.max((v[6] as f64 + v[7] as f64 - expected).abs());
                    compared += 1;
                }
            }
            println!(
                "F32_OPL {}",
                json!({"sample":name,"mode":if b.precision().opl_double_float{"df"}else{"f32"},"max_mm":max_opl,"fraction_internal_target":max_opl/1e-7,"compared":compared})
            );
            b.release_systems(h).unwrap();
        }
        let mut model = model;
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let points = (PupilSampling::Grid { n: 7 }).points();
        for b in &backends {
            let mut max_residual = 0f64;
            let mut max_reported = 0f64;
            let mut failed = 0;
            let mut compared = 0;
            let mut stop_failed = 0;
            for f in 0..t.fields().len() {
                for w in 0..t.num_wavelengths() {
                    let (p, h) = support::make_plan(b, &t, f, w, &points, false, 40, false);
                    let r = b
                        .submit_plan(p, &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let d = support::read_all(b, &r);
                    let ff = support::floats(&d[0]);
                    let report = support::uints(&d[1]);
                    let n = points.len();
                    for (i, pt) in points.iter().enumerate() {
                        max_reported = max_reported.max(
                            f32::from_bits(report[2 * n + i]) as f64
                                + f32::from_bits(report[3 * n + i]) as f64,
                        );
                        if report[i] != 0 {
                            failed += 1;
                        }
                        let p = Vec3::new(
                            ff[i] as f64 + ff[9 * n + i] as f64,
                            ff[n + i] as f64 + ff[10 * n + i] as f64,
                            ff[2 * n + i] as f64 + ff[11 * n + i] as f64,
                        );
                        let dir = Vec3::new(
                            ff[3 * n + i] as f64 + ff[12 * n + i] as f64,
                            ff[4 * n + i] as f64 + ff[13 * n + i] as f64,
                            ff[5 * n + i] as f64 + ff[14 * n + i] as f64,
                        );
                        let stop = mklight_trace::trace_ray(
                            t.system(),
                            &Ray::new(p, dir),
                            w,
                            &TraceOptions::unclipped().to_surface(t.system().stop),
                            None,
                        );
                        if !stop.status.is_ok() {
                            stop_failed += 1;
                            continue;
                        }
                        let (x, y) = t.field(f).point.vignetting.apply(pt[0], pt[1]);
                        let radius = t.pupil().r_stop;
                        max_residual = max_residual.max(
                            (stop.ray.pos.x - x * radius).hypot(stop.ray.pos.y - y * radius)
                                / radius,
                        );
                        compared += 1;
                    }
                    b.release_systems(h).unwrap();
                }
            }
            println!(
                "F32_AIM {}",
                json!({"sample":name,"mode":if b.precision().opl_double_float{"df"}else{"f32"},"max_relative_stop_residual":max_residual,"fraction_requested":max_residual/1e-10,"max_reported":max_reported,"failed":failed,"compared":compared,"stop_failed":stop_failed})
            );
        }
    }
}

#[test]
#[ignore = "real Metal; all-sample alternating trace-grid and aiming timings"]
fn trace_grid_and_aiming_speed() {
    use rayon::prelude::*;
    use std::time::Instant;
    let backends = [
        backend(MetalPrecision::DoubleFloat, LibraryChoice::Precompiled),
        backend(MetalPrecision::PureF32, LibraryChoice::Precompiled),
    ];
    for (name, model) in samples::all() {
        let t = testutil::tracer(model.clone());
        let mut rays = RayBatch::default();
        let mut reference = Vec::new();
        for f in 0..t.fields().len() {
            for w in 0..t.num_wavelengths() {
                for p in (PupilSampling::Grid { n: 64 }).points() {
                    if let Ok(g) = t.generate(t.field(f), w, p[0], p[1]) {
                        let a = g.ray.pos;
                        let d = g.ray.dir;
                        rays.push_f64([a.x, a.y, a.z], [d.x, d.y, d.z], w as u32);
                        reference.push((g.ray, w));
                    }
                }
            }
        }
        let handles: Vec<_> = backends
            .iter()
            .map(|b| {
                b.upload_system(&PackedSystem::pack_all(t.system()).unwrap())
                    .unwrap()
            })
            .collect();
        for round in 0..11 {
            for index in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                let start = Instant::now();
                let device_seconds = if index == 2 {
                    let out: Vec<_> = reference
                        .par_iter()
                        .map(|(r, w)| {
                            mklight_trace::trace_ray_recorded(t.system(), r, *w, &t.options())
                        })
                        .collect();
                    std::hint::black_box(out);
                    0.
                } else {
                    let mut req = TraceRequest::new(
                        handles[index],
                        rays.clone(),
                        0,
                        t.system().image_index() as u32,
                    );
                    req.record = RecordMode::AllSurfaces;
                    let out = backends[index]
                        .run(Job::Trace(Box::new(req)), &JobContext::new())
                        .unwrap();
                    std::hint::black_box(&out.output);
                    out.stats.device_seconds
                };
                println!(
                    "F32_GRID_TIME {}",
                    json!({"sample":name,"rays":rays.len(),"round":round,"mode":index,"phase":if round==0{"warmup"}else{"measured"},"wall_seconds":start.elapsed().as_secs_f64(),"device_seconds":device_seconds})
                );
            }
        }
        for (b, h) in backends.iter().zip(handles) {
            b.release_systems(h).unwrap();
        }
        let mut model = model;
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let points = (PupilSampling::Grid { n: 7 }).points();
        let plans: Vec<Vec<_>> = backends
            .iter()
            .map(|b| {
                let mut plans = Vec::new();
                for f in 0..t.fields().len() {
                    for w in 0..t.num_wavelengths() {
                        plans.push(support::make_plan(b, &t, f, w, &points, false, 40, false));
                    }
                }
                plans
            })
            .collect();
        let jobs: Vec<_> = (0..t.fields().len())
            .flat_map(|f| {
                (0..t.num_wavelengths()).flat_map({
                    let points = &points;
                    move |w| points.iter().map(move |p| (f, w, *p))
                })
            })
            .collect();
        for round in 0..11 {
            for index in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                let start = Instant::now();
                let device_seconds = if index == 2 {
                    let out: Vec<_> = jobs
                        .par_iter()
                        .map(|(f, w, p)| t.trace(t.field(*f), *w, p[0], p[1], &t.options()))
                        .collect();
                    std::hint::black_box(out);
                    0.
                } else {
                    let b = &backends[index];
                    let mut device_seconds = 0.;
                    for (plan, _) in &plans[index] {
                        let out = b
                            .submit_plan(plan.clone(), &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        device_seconds += out.stats.device_seconds;
                        std::hint::black_box(support::read_all(b, &out));
                    }
                    device_seconds
                };
                println!(
                    "F32_AIM_TIME {}",
                    json!({"sample":name,"rays":jobs.len(),"round":round,"mode":index,"phase":if round==0{"warmup"}else{"measured"},"wall_seconds":start.elapsed().as_secs_f64(),"device_seconds":device_seconds})
                );
            }
        }
        for (b, plans) in backends.iter().zip(plans) {
            for (_, h) in plans {
                b.release_systems(h).unwrap();
            }
        }
    }
}
