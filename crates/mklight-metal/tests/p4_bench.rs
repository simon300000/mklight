#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p4_common.rs"]
mod common;
use common::*;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, systems::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use std::{hint::black_box, time::Instant};

fn reference(c: &CompiledSystem, rows: &[Candidate]) -> Vec<PackedSystem> {
    let mut out = vec![];
    for row in rows {
        let rs = c.reference(row, &JobContext::new()).unwrap();
        for w in 0..rs.wavelengths_um.len() {
            for axis in [
                mklight_trace::paraxial::ParaxialAxis::X,
                mklight_trace::paraxial::ParaxialAxis::Y,
            ] {
                let model = mklight_trace::paraxial::ParaxialModel::for_axis(&rs, w, axis);
                let a = model.trace(1.0, 0.0);
                let b = model.trace(0.0, 1.0);
                let last = rs.surfaces.len() - 2;
                let summary = [
                    -1.0 / a.nu[last],
                    -a.y[last] / a.u[last],
                    b.nu[last] / a.nu[last],
                    model.thickness[1..=last].iter().sum(),
                    a.y[last],
                    b.y[last],
                    a.nu[last],
                    b.nu[last],
                ];
                black_box((model, a, b, summary));
            }
        }
        out.extend(PackedSystem::pack_chunks(&rs, None).unwrap());
    }
    out
}
fn new_run(
    gpu: &MetalBackend,
    c: &CompiledSystem,
    u: &UploadedSystem,
    rows: &[Candidate],
) -> serde_json::Value {
    let started = Instant::now();
    let mut plan = ResidentPlan::new(c.snapshot().clone());
    let r = u.append(gpu, &mut plan, rows, &JobContext::new()).unwrap();
    plan.outputs = r.outputs();
    let prep = started.elapsed().as_secs_f64();
    let p = gpu
        .submit_plan(plan, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let read = Instant::now();
    let mut bytes = 0;
    for h in &p.outputs {
        let r = gpu.readback(h.view(), &JobContext::new()).unwrap();
        bytes += r.copied_bytes;
        black_box(r.data);
    }
    let read_seconds = read.elapsed().as_secs_f64();
    let wall = started.elapsed().as_secs_f64();
    serde_json::json!({"wall":wall,"prepare_upload":prep,"readback":read_seconds,"readback_bytes":bytes,"stats":p.stats,"candidate_upload_bytes":rows.len()*(c.parameter_count()*8+16)})
}
fn summary(values: &[serde_json::Value], key: &str) -> serde_json::Value {
    let mut xs: Vec<_> = values.iter().map(|x| x[key].as_f64().unwrap()).collect();
    xs.sort_by(f64::total_cmp);
    serde_json::json!({"median":(xs[4]+xs[5])*0.5,"p95":xs[9]})
}
#[test]
#[ignore = "host measurement: 10 runs after warm-up; all samples and B sweep"]
fn preparation_benchmark_all_samples() {
    let boot = Instant::now();
    let gpu = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap()
    .expect("host Metal required");
    eprintln!(
        "P4_BENCH_MACHINE {}",
        serde_json::json!({"startup_probe_seconds":boot.elapsed().as_secs_f64(),"device":gpu.info(),"precision":gpu.precision(),"seed":0,"runs":10,"warmup":1})
    );
    let mut cases = vec![];
    for (name, s) in mklight_core::samples::all() {
        let v = basic_variables(&s);
        let rows = basic_rows(&s);
        let compile = Instant::now();
        let c = CompiledSystem::compile(
            Snapshot::new(s).unwrap(),
            testutil::library(),
            v,
            None,
            &JobContext::new(),
        )
        .unwrap();
        cases.push((name.to_string(), c, rows, compile.elapsed().as_secs_f64()));
    }
    cases.extend(
        extended_cases()
            .into_iter()
            .filter(|(name, _, _)| {
                ["rigid-finite", "thermal-offset-20wl", "material-choices"].contains(&name.as_str())
            })
            .map(|(n, c, r)| (n, c, r, 0.0)),
    );
    for (name, c, small, compile_seconds) in cases {
        let setup = Instant::now();
        let u = c.upload(&gpu, &JobContext::new()).unwrap();
        let upload_seconds = setup.elapsed().as_secs_f64();
        for count in [1, 16, 256, 1024] {
            let rows: Vec<_> = (0..count)
                .map(|i| Candidate {
                    id: i as u64,
                    values: small[i % small.len()].values.clone(),
                })
                .collect();
            let cold = new_run(&gpu, &c, &u, &rows);
            black_box(reference(&c, &rows));
            black_box(new_run(&gpu, &c, &u, &rows));
            let mut cpu = vec![];
            let mut old = vec![];
            let mut new = vec![];
            for _ in 0..10 {
                let now = Instant::now();
                let packed = reference(&c, &rows);
                let host = now.elapsed().as_secs_f64();
                let bytes: usize = packed
                    .iter()
                    .map(|p| {
                        size_of::<PackedHeader>()
                            + p.surfaces.len() * size_of::<PackedSurface>()
                            + p.coefficients.len() * 8
                    })
                    .sum();
                cpu.push(serde_json::json!({"wall":host,"packed_bytes":bytes}));
                let h = gpu.upload_systems(&packed).unwrap();
                gpu.release_systems(h).unwrap();
                old.push(
                    serde_json::json!({"wall":now.elapsed().as_secs_f64(),"uploaded_bytes":bytes}),
                );
                new.push(new_run(&gpu, &c, &u, &rows));
            }
            eprintln!(
                "P4_BENCH {}",
                serde_json::json!({"name":name,"batch":count,"instructions":c.instruction_count(),"compile_seconds":compile_seconds,"template_upload_seconds":upload_seconds,"cold":cold,"cpu_summary":summary(&cpu,"wall"),"old_metal_upload_summary":summary(&old,"wall"),"new_summary":summary(&new,"wall"),"cpu_raw":cpu,"old_metal_raw":old,"new_raw":new})
            );
        }
        u.release(&gpu).unwrap();
    }
}

#[test]
#[ignore = "host end-to-end measurement with reference-generated rays until P1 integration"]
fn full_evaluation_benchmark() {
    use mklight_trace::{PupilSampling, SequentialTracer};
    use rayon::prelude::*;
    let gpu = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap()
    .expect("host Metal required");
    for (name, system) in mklight_core::samples::all().into_iter().filter(|(n, _)| {
        [
            "singlet",
            "cooke_triplet",
            "double_gauss",
            "newtonian",
            "relay",
            "hybrid_doe",
        ]
        .contains(n)
    }) {
        let vars = basic_variables(&system);
        let small = basic_rows(&system);
        let steps = system.surfaces.len() - 1;
        let c = CompiledSystem::compile(
            Snapshot::new(system).unwrap(),
            testutil::library(),
            vars,
            None,
            &JobContext::new(),
        )
        .unwrap();
        let u = c.upload(&gpu, &JobContext::new()).unwrap();
        for target in [100usize, 10_000, 1_000_000] {
            let batch_count = if target == 100 { 1 } else { 16 };
            let rows: Vec<_> = (0..batch_count)
                .map(|i| Candidate {
                    id: i as u64,
                    values: small[i % 3].values.clone(),
                })
                .collect();
            let mut cpu_raw = vec![];
            let mut old_raw = vec![];
            let mut new_raw = vec![];
            for run_index in 0..11 {
                let setup = Instant::now();
                let systems: Vec<_> = rows
                    .iter()
                    .map(|row| c.reference(row, &JobContext::new()).unwrap())
                    .collect();
                let tracers: Vec<_> = systems
                    .iter()
                    .cloned()
                    .map(|s| SequentialTracer::new(s.into()).unwrap())
                    .collect();
                let resolution = setup.elapsed().as_secs_f64();
                let mut rays = vec![];
                let mut batch = RayBatch::default();
                let nf = tracers[0].fields().len();
                let nw = tracers[0].num_wavelengths();
                let side = ((target as f64 / (batch_count * nf * nw) as f64 * 4.0
                    / std::f64::consts::PI)
                    .sqrt()
                    .ceil() as usize)
                    .max(3);
                let points = PupilSampling::Grid { n: side }.points();
                for (i, t) in tracers.iter().enumerate() {
                    for f in 0..nf {
                        for w in 0..nw {
                            for p in &points {
                                let ray = t.generate(t.field(f), w, p[0], p[1]).unwrap().ray;
                                batch.push_f64(
                                    [ray.pos.x, ray.pos.y, ray.pos.z],
                                    [ray.dir.x, ray.dir.y, ray.dir.z],
                                    w as u32,
                                );
                                batch.system.push(i as u32);
                                rays.push((i, w, ray));
                            }
                        }
                    }
                }
                let common = setup.elapsed().as_secs_f64();
                let n = rays.len();
                let now = Instant::now();
                let cpu: Vec<_> = rays
                    .par_iter()
                    .map(|(i, w, r)| {
                        mklight_trace::trace_ray(&systems[*i], r, *w, &tracers[*i].options(), None)
                    })
                    .collect();
                black_box(cpu);
                let cpu_seconds = common + now.elapsed().as_secs_f64();
                let now = Instant::now();
                let packed: Vec<_> = systems
                    .iter()
                    .map(|s| PackedSystem::pack_all(s).unwrap())
                    .collect();
                let pack_seconds = now.elapsed().as_secs_f64();
                let h = gpu.upload_systems(&packed).unwrap();
                let mut req = TraceRequest::new(h, batch.clone(), 0, steps as u32);
                req.radiometry = Radiometry::Off;
                let old = gpu
                    .run(Job::Trace(Box::new(req)), &JobContext::new())
                    .unwrap();
                gpu.release_systems(h).unwrap();
                let old_seconds = common + now.elapsed().as_secs_f64();
                let now = Instant::now();
                let (data, prep_stats) = run_uploaded(&gpu, &c, &u, &rows, 64);
                let prepared: Vec<_> = systems
                    .iter()
                    .enumerate()
                    .flat_map(|(i, s)| decode(&data, i, s))
                    .collect();
                let prepare_seconds = now.elapsed().as_secs_f64();
                let h = gpu.upload_systems(&prepared).unwrap();
                let mut req = TraceRequest::new(h, batch.clone(), 0, steps as u32);
                req.radiometry = Radiometry::Off;
                let new = gpu
                    .run(Job::Trace(Box::new(req)), &JobContext::new())
                    .unwrap();
                gpu.release_systems(h).unwrap();
                let new_seconds = common + now.elapsed().as_secs_f64();
                if run_index == 0 {
                    eprintln!(
                        "P4_FULL_COLD {}",
                        serde_json::json!({"name":name,"batch":batch_count,"target_rays":target,"actual_rays":n,"cpu_wall":cpu_seconds,"old_wall":old_seconds,"new_wall":new_seconds,"common_setup":common,"preparation":prep_stats,"trace":new.stats})
                    );
                }
                if run_index > 0 {
                    cpu_raw.push(serde_json::json!({"wall":cpu_seconds,"common_setup":common,"resolution":resolution}));
                    old_raw.push(serde_json::json!({"wall":old_seconds,"packing":pack_seconds,"stats":old.stats}));
                    new_raw.push(serde_json::json!({"wall":new_seconds,"preparation_and_bridge":prepare_seconds,"preparation":prep_stats,"trace":new.stats}));
                }
                if run_index == 10 {
                    eprintln!(
                        "P4_FULL_BENCH {}",
                        serde_json::json!({"name":name,"batch":batch_count,"target_rays":target,"actual_rays":n,"pupil_side":side,"fields":nf,"wavelengths":nw,"seed":0,"radiometry":"off","reference_generated_rays":true,"compatibility_readback_reupload":true,"cpu_summary":summary(&cpu_raw,"wall"),"old_summary":summary(&old_raw,"wall"),"new_summary":summary(&new_raw,"wall"),"cpu_raw":cpu_raw,"old_raw":old_raw,"new_raw":new_raw})
                    );
                }
            }
        }
        u.release(&gpu).unwrap();
    }
}
