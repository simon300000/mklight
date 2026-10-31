#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;
use mklight_core::{JobContext, RayAiming, Vec3, samples};
use mklight_gpu::{resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{PupilSampling, SequentialTracer};
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};
use support::*;
fn legacy_opd(r: &TraceResult, i: usize, reference: &mklight_trace::OpdReference) -> f64 {
    if r.status[i] != 0 {
        return f64::NAN;
    }
    let w = Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64) - reference.center;
    let d = Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64);
    let b = d.dot(w);
    let p2 = w.cross(d).norm2();
    let radius = reference.s_exit_pupil.abs();
    let corr = if radius.is_finite() {
        if p2 > radius * radius {
            return f64::NAN;
        }
        reference.s_exit_pupil.signum() * p2 / (radius + (radius * radius - p2).sqrt())
    } else {
        0.
    };
    ((reference.chief_opl.hi - r.opl_hi[i] as f64)
        + (reference.chief_opl.lo - r.opl_lo[i] as f64)
        + reference.n_image * (b + corr))
        / (reference.wavelength_um * 1e-3)
}
fn old(
    gpu: &MetalBackend,
    t: &SequentialTracer,
    field: usize,
    wl: usize,
    points: &[[f64; 2]],
    h: SystemHandle,
) -> (f64, f64, f64, usize) {
    let total = Instant::now();
    let generated: Vec<_> = points
        .par_iter()
        .map(|p| t.generate(t.field(field), wl, p[0], p[1]))
        .collect();
    let mut rays = RayBatch::with_capacity(points.len());
    let mut failed = 0;
    for r in generated {
        match r {
            Ok(g) => {
                let (p, d) = (g.ray.pos, g.ray.dir);
                rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], wl as u32);
            }
            Err(_) => failed += 1,
        }
    }
    let prep = total.elapsed().as_secs_f64();
    let mut request = TraceRequest::new(h, rays, 0, t.system().image_index() as u32);
    request.radiometry = Radiometry::Off;
    let result = gpu
        .run(Job::Trace(Box::new(request)), &JobContext::new())
        .unwrap();
    let JobOutput::Trace(r) = result.output else {
        panic!()
    };
    let reference = t.opd_reference(t.field(field), wl).unwrap();
    let v: Vec<_> = (0..r.len())
        .map(|i| legacy_opd(&r, i, &reference))
        .collect();
    black_box(v);
    (
        total.elapsed().as_secs_f64(),
        result.stats.device_seconds,
        prep,
        failed,
    )
}
#[test]
#[ignore = "P1 matched end-to-end benchmark; host idle, release, one test thread"]
fn matched_generation_trace_opd() {
    let init = Instant::now();
    let gpu = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap()
    .expect("benchmark requires host Metal");
    eprintln!(
        "P1_SETUP {}",
        serde_json::json!({"seconds":init.elapsed().as_secs_f64(),"device":gpu.info(),"precision":gpu.precision(),"threads":rayon::current_num_threads(),"seed":42})
    );
    let dense = std::env::var("P1_BENCH_DENSE").is_ok();
    for (name, model) in samples::all() {
        let mut sides = if dense { vec![] } else { vec![12, 113] };
        if dense
            && [
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
            ]
            .contains(&name)
        {
            sides.push(1129);
        }
        for side in sides {
            for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
                let setup = Instant::now();
                let mut model = model.clone();
                model.ray_aiming = aiming;
                let t = testutil::tracer(model);
                let field = t.fields().len() - 1;
                let wl = t.system().primary_wavelength;
                let points = (PupilSampling::Grid { n: side }).points();
                let n = points.len();
                let reference = t.opd_reference(t.field(field), wl).unwrap();
                let reference_setup = setup.elapsed().as_secs_f64();
                let setup = Instant::now();
                let (mut p, h) = make_plan(&gpu, &t, field, wl, &points, false, 40, true);
                p.outputs = p.outputs[5..].to_vec();
                let prepare = setup.elapsed().as_secs_f64();
                for run in 0..11 {
                    let wall = Instant::now();
                    let cpu: Vec<_> = points
                        .par_iter()
                        .map(|p| {
                            let r = t.trace(t.field(field), wl, p[0], p[1], &t.options());
                            reference.opd_waves(&r)
                        })
                        .collect();
                    black_box(&cpu);
                    let cpu_values = cpu;
                    let cpu = wall.elapsed().as_secs_f64();
                    let (old_wall, old_device, old_prepare, old_failed) =
                        old(&gpu, &t, field, wl, &points, h);
                    let wall = Instant::now();
                    let result = gpu
                        .submit_plan(p.clone(), &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let read = Instant::now();
                    let mut v = result.outputs[0].view();
                    v.desc = BufferDesc::new(ElementType::DoubleFloat, [n]);
                    let opd = gpu.readback(v, &JobContext::new()).unwrap();
                    let status = gpu
                        .readback(result.outputs[1].view(), &JobContext::new())
                        .unwrap();
                    black_box(&opd.data);
                    black_box(&status.data);
                    let read_seconds = read.elapsed().as_secs_f64();
                    let new_wall = wall.elapsed().as_secs_f64();
                    let mut max_opd = 0.0_f64;
                    if run == 0 {
                        let values = doubles(&opd.data);
                        let status = uints(&status.data);
                        for i in 0..n {
                            assert_eq!(
                                status[i] == 0,
                                cpu_values[i].is_some(),
                                "{name} pupil {:?}",
                                points[i]
                            );
                            if let Some(reference) = cpu_values[i] {
                                max_opd = max_opd.max((values[i].to_f64() - reference).abs());
                            }
                        }
                        assert!(max_opd <= 0.005, "{name} dense OPD error {max_opd}");
                    }

                    eprintln!(
                        "P1_BENCH {}",
                        serde_json::json!({"sample":name,"side":side,"rays":n,"field":field,"wl":wl,"aiming":aiming,"run":run,"cold":run==0,"cpu_wall":cpu,"old_wall":old_wall,"old_device":old_device,"old_prepare":old_prepare,"old_failed":old_failed,"new_wall":new_wall,"new_stats":result.stats,"read_seconds":read_seconds,"read_bytes":opd.copied_bytes+status.copied_bytes,"reference_setup":reference_setup,"resident_prepare":prepare,"seed":42,"max_opd_waves":if run==0{Some(max_opd)}else{None}})
                    );
                }
                gpu.release_systems(h).unwrap();
            }
        }
    }
}
