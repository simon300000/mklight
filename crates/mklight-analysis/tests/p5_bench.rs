#![cfg(target_os = "macos")]
use mklight_analysis::{
    gpu,
    spot::{self, FieldRays, ImageHit, ImageRay, RayBatch as SpotBatch, SpotDiagramSettings},
    tracer_for,
    util::{self, BackendInfo, FieldSelection, PupilPattern, WavelengthSelection},
};
use mklight_core::{JobContext, Vec3, samples};
use mklight_gpu::{
    ComputeBackend, Job, JobOutput, PackedSystem, Radiometry, RayBatch, TraceRequest,
};
use mklight_metal::MetalBackend;
use rayon::prelude::*;
use std::time::Instant;
fn host() -> serde_json::Value {
    let output = std::process::Command::new("ps")
        .args(["-Ao", "pcpu,comm"])
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let mut total = 0.0f64;
    let mut maximum = 0.0f64;
    let mut jobs = Vec::new();
    for line in text.lines().skip(1) {
        let mut columns = line.split_whitespace();
        let cpu = columns
            .next()
            .and_then(|x| x.parse::<f64>().ok())
            .unwrap_or(0.0);
        let command = columns.collect::<Vec<_>>().join(" ");
        if command.contains("/p5_bench-") {
            continue;
        }
        total += cpu;
        maximum = maximum.max(cpu);
        if (command.contains("mklight")
            && command.contains("/target/")
            && !command.contains("/p5_bench-"))
            || command.ends_with("/rustc")
            || command.ends_with("/metal")
        {
            jobs.push(command);
        }
    }
    serde_json::json!({"total_cpu_percent":total,"maximum_cpu_percent":maximum,"other_compute_jobs":jobs})
}
fn wait_quiet() -> serde_json::Value {
    loop {
        let h = host();
        if h["other_compute_jobs"].as_array().unwrap().is_empty()
            && h["maximum_cpu_percent"].as_f64().unwrap() < 30.0
            && h["total_cpu_percent"].as_f64().unwrap() < 100.0
        {
            return h;
        }
        eprintln!("P5_HOST_WAIT {h}");
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
fn old(
    b: &dyn ComputeBackend,
    t: &mklight_trace::SequentialTracer,
    s: &SpotDiagramSettings,
) -> (spot::SpotDiagram, serde_json::Value) {
    let augmented = util::default_fields(t, &s.fields).unwrap();
    let t = augmented.as_ref();
    let fields = s.fields.resolve(t.fields().len()).unwrap();
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)
        .unwrap();
    let pupil = util::pupil_points(s.pattern, s.rings, s.seed);
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    let handle = b.upload_system(&packed).unwrap();
    let build = |pupil: &[[f64; 2]]| {
        let count = fields.len() * wls.len() * pupil.len();
        let generated: Vec<_> = (0..count)
            .into_par_iter()
            .map(|i| {
                let fi = i / (wls.len() * pupil.len());
                let wi = (i / pupil.len()) % wls.len();
                let p = pupil[i % pupil.len()];
                let g = t
                    .generate(t.field(fields[fi]), wls[wi], p[0], p[1])
                    .unwrap();
                (g, wi)
            })
            .collect();
        let mut rays = RayBatch::with_capacity(count);
        for (g, w) in generated {
            let p = g.ray.pos;
            let d = g.ray.dir;
            rays.push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], w as u32);
        }
        rays
    };
    let trace = |rays, clip| {
        let mut request = TraceRequest::new(handle, rays, 0, t.system().image_index() as u32);
        request.apply_apertures = clip;
        request.radiometry = Radiometry::Off;
        let result = b
            .run(Job::Trace(Box::new(request)), &JobContext::new())
            .unwrap();
        match result.output {
            JobOutput::Trace(v) => (*v, result.stats),
            _ => unreachable!(),
        }
    };
    let (result, rs) = trace(build(&pupil), true);
    let (chiefs, cs) = trace(build(&[[0.0; 2]]), false);
    let old_stats = serde_json::json!({"trace_jobs":[rs,cs],"result_copy_bytes":52*(result.len()+chiefs.len())});
    let hit = |r: &mklight_gpu::TraceResult, i| {
        if r.status[i] == 0 {
            Some(ImageHit {
                point: [r.x[i] as f64, r.y[i] as f64],
                dir: Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64),
            })
        } else {
            None
        }
    };
    let mut warnings = Vec::new();
    let mut out = Vec::new();
    for (fi, &field) in fields.iter().enumerate() {
        let c: Vec<_> = (0..wls.len())
            .map(|wi| hit(&chiefs, fi * wls.len() + wi))
            .collect();
        let primary = wls
            .iter()
            .position(|&w| w == t.system().primary_wavelength)
            .unwrap();
        let rays: Vec<_> = (0..wls.len() * pupil.len())
            .map(|i| ImageRay {
                wavelength: wls[i / pupil.len()],
                hit: hit(&result, fi * wls.len() * pupil.len() + i),
            })
            .collect();
        out.push(
            FieldRays::from_image_rays(
                field,
                &wls,
                &wls.iter()
                    .map(|&w| t.system().wavelength_weights[w])
                    .collect::<Vec<_>>(),
                c[primary],
                &c,
                &rays,
                &mut warnings,
            )
            .unwrap(),
        );
    }
    b.release_systems(handle).unwrap();
    let diagram = spot::spot_from_rays(
        t,
        s,
        &SpotBatch {
            wavelengths: wls,
            fields: out,
        },
        &BackendInfo {
            name: b.info().backend_id(),
            precision: mklight_core::job::Precision::F32Gpu,
        },
        warnings,
    )
    .unwrap();
    (diagram, old_stats)
}
fn stats(v: &serde_json::Value, out: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, v) in m {
                if k == "ri_polish" {
                    out["fallback_count"] = serde_json::json!(
                        out["fallback_count"].as_f64().unwrap_or(0.0)
                            + v["attempted_nodes"].as_f64().unwrap_or(0.0)
                    );
                    out["polished_nodes"] = serde_json::json!(
                        out["polished_nodes"].as_f64().unwrap_or(0.0)
                            + v["resolved_nodes"].as_f64().unwrap_or(0.0)
                    );
                    out["reference_polish_seconds"] = serde_json::json!(
                        out["reference_polish_seconds"].as_f64().unwrap_or(0.0)
                            + v["seconds"].as_f64().unwrap_or(0.0)
                    );
                }
                if k == "plan_stats" {
                    for key in [
                        "device_seconds",
                        "command_buffers",
                        "copied_bytes",
                        "uploaded_bytes",
                        "readback_bytes",
                        "allocations",
                        "reused_allocations",
                        "fallback_count",
                        "polished_rays",
                    ] {
                        out[key] = serde_json::json!(
                            out[key].as_f64().unwrap_or(0.0) + v[key].as_f64().unwrap_or(0.0)
                        );
                    }
                    out["peak_bytes"] = serde_json::json!(
                        out["peak_bytes"]
                            .as_u64()
                            .unwrap_or(0)
                            .max(v["peak_bytes"].as_u64().unwrap_or(0))
                    );
                    let max = v["stages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| s["max_command_seconds"].as_f64().unwrap_or(0.0))
                        .fold(0.0, f64::max);
                    out["max_command_seconds"] = serde_json::json!(
                        out["max_command_seconds"].as_f64().unwrap_or(0.0).max(max)
                    );
                } else {
                    stats(v, out);
                }
            }
        }
        serde_json::Value::Array(a) => {
            for v in a {
                stats(v, out)
            }
        }
        _ => {}
    }
}
#[test]
#[ignore = "quiet host matched end-to-end benchmark"]
fn matched_spots() {
    let b = MetalBackend::new().unwrap().expect("host Metal required");
    eprintln!("P5_HARDWARE {:?} {:?}", b.info(), b.precision());
    let dense = [
        "singlet",
        "cooke_triplet",
        "double_gauss",
        "newtonian",
        "relay",
        "hybrid_doe",
    ];
    for (name, model) in samples::all() {
        if std::env::var("MKLIGHT_P5_SAMPLE").is_ok_and(|v| v != name) {
            continue;
        }
        let t = tracer_for(model).unwrap();
        let mut cases = vec![(
            "default",
            SpotDiagramSettings {
                include_points: true,
                ..Default::default()
            },
        )];
        for (label, rings) in [("tiny", 6), ("grid10k", 56), ("dense", 512)] {
            if label == "dense" && !dense.contains(&name) {
                continue;
            }
            cases.push((
                label,
                SpotDiagramSettings {
                    pattern: PupilPattern::Square,
                    rings,
                    fields: FieldSelection::Index {
                        index: t.fields().len() - 1,
                    },
                    wavelengths: WavelengthSelection::Primary,
                    include_points: false,
                    ..Default::default()
                },
            ));
        }
        for (label, s) in cases {
            if std::env::var("MKLIGHT_P5_CASE").is_ok_and(|v| v != label) {
                continue;
            }
            let packed = PackedSystem::pack_all(t.system()).unwrap();
            let prepare_copied_bytes = size_of::<mklight_gpu::layout::PackedHeader>()
                + packed.surfaces.len() * size_of::<mklight_gpu::layout::PackedSurface>()
                + packed.coefficients.len() * size_of::<mklight_gpu::Df32>();
            let preparation = Instant::now();
            let prepared = gpu::Geometry::new(&b, &t, &JobContext::new()).unwrap();
            let prepare_seconds = preparation.elapsed().as_secs_f64();
            for run in 0..=10 {
                let host_before = wait_quiet();
                let start = Instant::now();
                let cpu = spot::spot_diagram(&t, &s).unwrap();
                let cpu_s = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let (old, old_counters) = old(&b, &t, &s);
                let old_s = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let new = prepared.spot_diagram_job(&s, &JobContext::new()).unwrap();
                let new_s = start.elapsed().as_secs_f64();
                for (a, g) in cpu.fields.iter().zip(&new.fields) {
                    assert_eq!(a.polychromatic.rays_arrived, g.polychromatic.rays_arrived);
                    assert!(
                        (a.rms_radius - g.rms_radius).abs() < 2e-5 + 0.005 * a.rms_radius.abs()
                    );
                }
                assert_eq!(old.fields.len(), new.fields.len());
                let mut counters = serde_json::json!({});
                stats(&new.run_info.settings, &mut counters);
                eprintln!(
                    "P5_BENCH {}",
                    serde_json::json!({"sample":name,"case":label,"run":run,"settings":s,"cpu_seconds":cpu_s,"old_metal_seconds":old_s,"old_counters":old_counters,"new_metal_seconds":new_s,"prepare_seconds":prepare_seconds,"prepare_copied_bytes":prepare_copied_bytes,"host_before":host_before,"host_after":host(),"rays":new.fields.iter().map(|f|f.polychromatic.rays_launched).sum::<usize>(),"counters":counters})
                );
            }
        }
    }
}
#[test]
#[ignore = "quiet host RI and interactive layout benchmark"]
fn maps_and_layout() {
    let b = MetalBackend::new().unwrap().expect("host Metal required");
    for (name, model) in samples::all() {
        let t = tracer_for(model).unwrap();
        for run in 0..=10 {
            let host_before = wait_quiet();
            let s = mklight_analysis::layout::LayoutSettings::default();
            let start = Instant::now();
            let a = mklight_analysis::layout::layout(&t, &s).unwrap();
            let cpu = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let g = gpu::layout_job(&b, &t, &s, &JobContext::new()).unwrap();
            let wall = start.elapsed().as_secs_f64();
            assert_eq!(a.polylines.len(), g.polylines.len());
            let mut c = serde_json::json!({});
            stats(&g.run_info.settings, &mut c);
            eprintln!(
                "P5_LAYOUT {}",
                serde_json::json!({"sample":name,"run":run,"cpu_seconds":cpu,"metal_seconds":wall,"counters":c,"host_before":host_before,"host_after":host()})
            );
        }
    }
    for grid in [5, 33] {
        let t = tracer_for(samples::cooke_triplet()).unwrap();
        let s = mklight_analysis::field_maps::FieldMapSettings {
            grid: [grid, grid],
            ..Default::default()
        };
        for run in 0..=10 {
            let host_before = wait_quiet();
            let start = Instant::now();
            let a = mklight_analysis::field_maps::field_map_job(
                &t,
                mklight_analysis::field_maps::FieldAnalysis::RelativeIllumination,
                &s,
                &JobContext::new(),
            )
            .unwrap();
            let cpu = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let g = gpu::field_map_job(
                &b,
                &t,
                mklight_analysis::field_maps::FieldAnalysis::RelativeIllumination,
                &s,
                &JobContext::new(),
            )
            .unwrap();
            let wall = start.elapsed().as_secs_f64();
            for (a, g) in a.values.iter().step_by(3).zip(g.values.iter().step_by(3)) {
                assert!((a - g).abs() < 1e-3 || a.is_nan() && g.is_nan());
            }
            let mut c = serde_json::json!({});
            stats(&g.run_info.settings, &mut c);
            eprintln!(
                "P5_RI_BENCH {}",
                serde_json::json!({"grid":grid,"run":run,"settings":s,"cpu_seconds":cpu,"metal_seconds":wall,"counters":c,"host_before":host_before,"host_after":host()})
            );
        }
    }
}

fn old_footprint(
    b: &dyn ComputeBackend,
    t: &mklight_trace::SequentialTracer,
    s: &mklight_analysis::illumination::FootprintSettings,
) -> (usize, serde_json::Value) {
    use mklight_analysis::footprint_overlay::convex_hull;
    let field = s.fields.resolve(t.fields().len()).unwrap()[0];
    let wl = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)
        .unwrap()[0];
    let pupil = util::pupil_points(s.pattern, s.rings, s.seed);
    let generated: Vec<_> = pupil
        .par_iter()
        .map(|p| t.generate(t.field(field), wl, p[0], p[1]).unwrap())
        .collect();
    let mut rays = RayBatch::with_capacity(pupil.len());
    for g in generated {
        rays.push_f64(
            [g.ray.pos.x, g.ray.pos.y, g.ray.pos.z],
            [g.ray.dir.x, g.ray.dir.y, g.ray.dir.z],
            wl as u32,
        );
    }
    let handle = b
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let slots = t.system().num_surfaces();
    let mut req = TraceRequest::new(handle, rays, 0, (slots - 1) as u32);
    req.record = mklight_gpu::RecordMode::AllSurfaces;
    req.radiometry = Radiometry::Off;
    let result = b
        .run(Job::Trace(Box::new(req)), &JobContext::new())
        .unwrap();
    let JobOutput::Trace(r) = result.output else {
        unreachable!()
    };
    let mut points = Vec::new();
    let mut outlines = Vec::new();
    for surface in 1..slots {
        let start = surface * pupil.len();
        let hit: Vec<_> = (start..start + pupil.len())
            .filter(|&i| r.recorded_status[i] == 0)
            .map(|i| {
                let j = i * mklight_gpu::consts::MKL_TRACE_RECORD_FLOATS as usize;
                Vec3::new(
                    r.recorded[j] as f64,
                    r.recorded[j + 1] as f64,
                    r.recorded[j + 2] as f64,
                )
            })
            .collect();
        let frame = t.system().surfaces[surface].frame;
        outlines.extend(
            convex_hull(&hit)
                .into_iter()
                .map(|p| frame.to_global_point(p)),
        );
        points.extend(hit.into_iter().map(|p| frame.to_global_point(p)));
    }
    std::hint::black_box(&outlines);
    b.release_systems(handle).unwrap();
    (
        points.len(),
        serde_json::json!({"trace_job":result.stats,"result_copy_bytes":52*r.len()+4*(r.recorded.len()+r.recorded_status.len())}),
    )
}
#[test]
#[ignore = "host complete all-surface/footprint sweep"]
fn footprints() {
    use mklight_analysis::{footprint_overlay, illumination::FootprintSettings};
    let b = MetalBackend::new().unwrap().expect("host Metal required");
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    for (label, rings) in [("tiny", 6), ("grid10k", 56), ("dense", 512)] {
        let s = FootprintSettings {
            pattern: PupilPattern::Square,
            rings,
            fields: FieldSelection::Index {
                index: t.fields().len() - 1,
            },
            wavelengths: WavelengthSelection::Primary,
            ..Default::default()
        };
        for run in 0..=10 {
            let before = wait_quiet();
            let start = Instant::now();
            let a = footprint_overlay::footprint_overlay_job(&t, &s, &JobContext::new()).unwrap();
            let cpu = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let (old_points, old_counters) = old_footprint(&b, &t, &s);
            let old = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let g = gpu::footprint_overlay_job(&b, &t, &s, &JobContext::new()).unwrap();
            let wall = start.elapsed().as_secs_f64();
            for (a, g) in a.sets.iter().zip(&g.sets) {
                assert_eq!(a.points, g.points);
            }
            let mut c = serde_json::json!({});
            stats(&g.run_info.settings, &mut c);
            eprintln!(
                "P5_FOOTPRINT {}",
                serde_json::json!({"sample":"cooke_triplet","case":label,"settings":s,"run":run,"rays":util::pupil_points(s.pattern,s.rings,s.seed).len(),"points":g.points.len(),"old_points":old_points,"cpu_seconds":cpu,"old_metal_seconds":old,"metal_seconds":wall,"old_counters":old_counters,"counters":c,"host_before":before,"host_after":host()})
            );
        }
    }
}
