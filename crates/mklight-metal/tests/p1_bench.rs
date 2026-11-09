//! P1 end-to-end benchmark: resident aiming + trace (+ OPD for pupil grids)
//! against the unchanged CPU f64 reference on all cores, with matched
//! systems, fields, wavelengths, pupil points and settings.
//! `cargo test --release -p mklight-metal --test p1_bench -- --ignored --nocapture --test-threads=1`
//! (`P1_BENCH_SAMPLES=cooke_triplet,...` and `P1_BENCH_RUNS` override defaults).
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;
use mklight_core::{JobContext, OpticalSystem, RayAiming, samples};
use mklight_gpu::{aiming::*, resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{PupilSampling, SequentialTracer};
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};
use support::*;

/// Cooke triplet pushed from 20° to a 25° half field (`wide_field`), and to
/// 28° (`wide_field_28`), where about a sixth of the edge-field pupil cannot
/// reach the stop: every such ray exhausts the reference Newton schedule.
fn wide_field(degrees: f64) -> OpticalSystem {
    let mut m = samples::cooke_triplet();
    m.fields.fields.last_mut().unwrap().y = degrees;
    m
}

struct Case {
    /// (field, wavelength) rows and their pupil points.
    rows: Vec<(usize, usize)>,
    points: Vec<[f64; 2]>,
    opd: bool,
}

/// A plan generating `points` for every row (row-major), tracing to the image
/// and, for one-row cases, projecting OPD. Returns the plan and the outputs to
/// read back.
fn plan(
    gpu: &MetalBackend,
    t: &SequentialTracer,
    h: SystemHandle,
    case: &Case,
    key: Option<String>,
) -> ResidentPlan {
    let mut p =
        ResidentPlan::new(mklight_core::Snapshot::new(t.system().system().clone()).unwrap());
    let specs: Vec<_> = case
        .rows
        .iter()
        .map(|&(f, w)| {
            FieldSpec::new(
                t.system(),
                t.paraxial(),
                t.field(f).point,
                0,
                w as u32,
                t.aiming(),
            )
            .unwrap()
        })
        .collect();
    let fw = upload(gpu, &mut p, specs.len(), &specs);
    let samples: Vec<_> = (0..case.rows.len())
        .flat_map(|r| case.points.iter().map(move |&pt| (r, pt)))
        .enumerate()
        .map(|(i, (r, pt))| Sample::new(pt, r as u32, i as u64))
        .collect();
    let n = samples.len();
    let sv = upload(gpu, &mut p, n, &samples);
    let generated = rays(&mut p, n);
    p.operations
        .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
            systems: SystemSet::Uploaded(h),
            samples: sv,
            field_wavelength: fw,
            field_key: key,
            seeds: None,
            tolerance: 1e-10,
            max_iterations: 40,
            output: generated.clone(),
            options: Default::default(),
        })));
    let steps: Vec<u32> = (1..=t.system().image_index() as u32).collect();
    let buf = gpu
        .upload_buffer(
            BufferDesc::new(ElementType::U32, [steps.len()]),
            BufferData::U32(steps),
        )
        .unwrap();
    let si = p.import(buf.view());
    let hit = hits(&mut p, n, 0);
    p.operations
        .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
            systems: SystemSet::Uploaded(h),
            input: generated.clone(),
            steps: p.view(si).unwrap(),
            start_surface: 0,
            apply_apertures: true,
            radiometry: 0,
            newton: Default::default(),
            output: hit.clone(),
        })));
    p.outputs = vec![generated.report, hit.status];
    if case.opd {
        let (f, w) = case.rows[0];
        let r = t.opd_reference(t.field(f), w).unwrap();
        let cv = [
            r.center.x,
            r.center.y,
            r.center.z,
            r.chief_dir.x,
            r.chief_dir.y,
            r.chief_dir.z,
            r.chief_opl.value(),
        ]
        .map(Df32::from_f64);
        let chief = df(gpu, &mut p, &cv);
        let centre = df(gpu, &mut p, &cv[..3]);
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, n]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Aiming(Request::ProjectPupil(ProjectPupil {
                hits: hit,
                reference: PupilReference {
                    chief,
                    centre,
                    radius: Df32::from_f64(r.s_exit_pupil.abs()),
                    image_index: Df32::from_f64(r.n_image),
                    wavelength_um: Df32::from_f64(r.wavelength_um),
                    sphere_sheet: if r.s_exit_pupil < 0.0 { -1 } else { 1 },
                    telecentric: !r.s_exit_pupil.is_finite(),
                },
                output,
                status,
            })));
        p.outputs = vec![generated.report, output, status];
    }
    p
}

/// CPU f64 reference: generate (aim) and trace every ray, plus OPD.
fn cpu_run(t: &SequentialTracer, case: &Case) -> Vec<Option<f64>> {
    let refs: Vec<_> = case
        .rows
        .iter()
        .map(|&(f, w)| case.opd.then(|| t.opd_reference(t.field(f), w).unwrap()))
        .collect();
    let opts = t.options();
    (0..case.rows.len() * case.points.len())
        .into_par_iter()
        .map(|i| {
            let r = i / case.points.len();
            let (f, w) = case.rows[r];
            let p = case.points[i % case.points.len()];
            let traced = t.trace(t.field(f), w, p[0], p[1], &opts);
            match &refs[r] {
                Some(reference) => reference.opd_waves(&traced),
                None => traced.status.is_ok().then_some(0.0),
            }
        })
        .collect()
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    let p95 = v[((v.len() as f64 * 0.95).ceil() as usize).clamp(1, v.len()) - 1];
    (v[v.len() / 2] * 1e3, p95 * 1e3)
}

#[test]
#[ignore = "P1 benchmark; idle host, release, one test thread"]
fn aim_trace_opd_matrix() {
    let gpu = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap()
    .expect("benchmark requires host Metal");
    let runs: usize = std::env::var("P1_BENCH_RUNS").map_or(10, |v| v.parse().unwrap());
    let names =
        std::env::var("P1_BENCH_SAMPLES").unwrap_or("double_gauss,cooke_triplet,wide_field".into());
    eprintln!(
        "P1_SETUP device={} build={} threads={} runs={runs}",
        gpu.info().name,
        gpu.precision().kernel_build,
        rayon::current_num_threads()
    );
    eprintln!(
        "| sample | case | aim | rays | CPU f64 med/p95 ms | GPU med/p95 ms | GPU device ms | commands | speed-up | mean iterations | unresolved | upload/readback copy bytes |"
    );
    for name in names.split(',') {
        let base = match name {
            "wide_field" => wide_field(25.0),
            "wide_field_28" => wide_field(28.0),
            n => samples::all().into_iter().find(|s| s.0 == n).unwrap().1,
        };
        for aiming in [RayAiming::Real { tolerance: 1e-10 }, RayAiming::Off] {
            let mut model = base.clone();
            model.ray_aiming = aiming;
            let t = testutil::tracer(model);
            let edge = t.fields().len() - 1;
            let primary = t.system().primary_wavelength;
            let all: Vec<_> = (0..t.fields().len())
                .flat_map(|f| (0..t.num_wavelengths()).map(move |w| (f, w)))
                .collect();
            let mut cases = vec![];
            if matches!(aiming, RayAiming::Real { .. }) {
                cases.push((
                    "chief".to_string(),
                    Case {
                        rows: all.clone(),
                        points: vec![[0.0, 0.0]],
                        opd: false,
                    },
                ));
                cases.push((
                    "marginal".to_string(),
                    Case {
                        rows: all.clone(),
                        points: vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                        opd: false,
                    },
                ));
            }
            for side in [64, 256, 1024] {
                cases.push((
                    format!("grid {side}²"),
                    Case {
                        rows: vec![(edge, primary)],
                        points: PupilSampling::Grid { n: side }.points(),
                        opd: true,
                    },
                ));
            }
            let h = gpu
                .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
                .unwrap();
            for (label, case) in &cases {
                let n = case.rows.len() * case.points.len();
                let warm = plan(
                    &gpu,
                    &t,
                    h,
                    case,
                    Some(format!("{name}:{label}:{aiming:?}")),
                );
                let (mut cpu, mut wall, mut device) = (vec![], vec![], vec![]);
                let (mut commands, mut copies) = (0, (0, 0));
                let (mut iterations, mut unresolved) = (0.0, 0);
                for run in 0..=runs {
                    let start = Instant::now();
                    let reference = cpu_run(&t, case);
                    let cpu_s = start.elapsed().as_secs_f64();
                    let start = Instant::now();
                    let r = gpu
                        .submit_plan(warm.clone(), &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    // Read only what a consumer needs: the OPD plane (or the
                    // trace status plane) and the status.
                    let mut copied = 0;
                    let data: Vec<_> = r.outputs[1..]
                        .iter()
                        .map(|o| {
                            let mut v = o.view();
                            v.desc = BufferDesc::new(v.desc.element, [n]);
                            let b = gpu.readback(v, &JobContext::new()).unwrap();
                            copied += b.copied_bytes;
                            b.data
                        })
                        .collect();
                    black_box(&data);
                    let gpu_s = start.elapsed().as_secs_f64();
                    if run == 0 {
                        let report = gpu
                            .readback(r.outputs[0].view(), &JobContext::new())
                            .unwrap();
                        let report = uints(&report.data);
                        iterations =
                            report[n..2 * n].iter().map(|&k| k as f64).sum::<f64>() / n as f64;
                        unresolved = r.stats.fallback_count;
                        let status = uints(data.last().unwrap());
                        let mut max_opd = 0.0_f64;
                        for i in 0..n {
                            assert_eq!(
                                status[i] == 0,
                                reference[i].is_some(),
                                "{name} {label} ray {i}"
                            );
                            if let (true, Some(v)) = (case.opd, reference[i]) {
                                max_opd = max_opd.max((doubles(&data[0])[i].to_f64() - v).abs());
                            }
                        }
                        assert!(max_opd <= 5e-3, "{name} {label} OPD error {max_opd}");
                        eprintln!(
                            "P1_ACCURACY {name} {label} {aiming:?} max_opd_waves={max_opd:e}"
                        );
                        continue;
                    }
                    if run == runs {
                        let stages: Vec<_> = r
                            .stats
                            .stages
                            .iter()
                            .filter(|s| s.command_buffers > 0)
                            .map(|s| {
                                format!(
                                    "{} {:.3}/{:.3} ms x{} max {:.3}",
                                    s.name,
                                    s.wall_seconds * 1e3,
                                    s.device_seconds * 1e3,
                                    s.command_buffers,
                                    s.max_command_seconds * 1e3
                                )
                            })
                            .collect();
                        eprintln!("P1_STAGES {name} {label} {aiming:?}: {}", stages.join("; "));
                    }
                    cpu.push(cpu_s);
                    wall.push(gpu_s);
                    device.push(r.stats.device_seconds);
                    commands = r.stats.command_buffers;
                    // Copies per warm run: plan uploads (none: inputs are
                    // imported leases) and host readback of the results.
                    copies = (r.stats.uploaded_bytes, copied);
                }
                // Cold: chief seeds prepared inside the plan (no cache key).
                let cold: Vec<_> = (0..3)
                    .map(|_| {
                        let p = plan(&gpu, &t, h, case, None);
                        let start = Instant::now();
                        gpu.submit_plan(p, &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        start.elapsed().as_secs_f64()
                    })
                    .collect();
                let cpu_cold: Vec<_> = (0..3)
                    .map(|_| {
                        let start = Instant::now();
                        let fields: Vec<_> = case
                            .rows
                            .iter()
                            .map(|&(f, _)| f)
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .map(|f| t.prepare_field(t.field(f).point).unwrap())
                            .collect();
                        black_box(fields);
                        start.elapsed().as_secs_f64() + stats(cpu.clone()).0 * 1e-3
                    })
                    .collect();
                eprintln!(
                    "P11_P1_RAW {}",
                    serde_json::json!({"sample":name,"case":label,"aiming":aiming,"rays":n,"cpu_seconds":cpu,"metal_seconds":wall,"device_seconds":device,"cold_cpu_seconds":cpu_cold,"cold_metal_seconds":cold,"commands":commands,"copy_counts":copies,"unresolved":unresolved,"mean_iterations":iterations})
                );
                let (c, cp) = stats(cpu.clone());
                let (g, gp) = stats(wall);
                let mode = if matches!(aiming, RayAiming::Off) {
                    "off"
                } else {
                    "real"
                };
                eprintln!(
                    "| {name} | {label} | {mode} | {n} | {c:.3}/{cp:.3} | {g:.3}/{gp:.3} | {:.3} | {commands} | {:.2}x | {iterations:.2} | {unresolved} | {}/{} |",
                    stats(device).0,
                    c / g,
                    copies.0,
                    copies.1
                );
                eprintln!(
                    "| {name} | {label} cold | {mode} | {n} | {:.3} | {:.3} | | | {:.2}x | | |",
                    stats(cpu_cold.clone()).0,
                    stats(cold.clone()).0,
                    stats(cpu_cold).0 / stats(cold).0
                );
            }
            gpu.release_systems(h).unwrap();
        }
    }
}
