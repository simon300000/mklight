//! Device aiming cost and convergence against the stop tolerance.
//!
//! Ignored, env-driven, nothing embedded:
//! `cargo test --release -p mklight-bench --test aim_precision_study -- --ignored --nocapture`
//! `MKLIGHT_STUDY_PROJECT=/path/project.mklight` studies a local project (with
//! its catalogues) instead of three samples; `MKLIGHT_STUDY_TOLERANCES`
//! (comma list, default `1e-10,1e-8,1e-6,1e-5,1e-4`), `MKLIGHT_STUDY_ROWS`
//! (`all` or the number of field rows), `MKLIGHT_STUDY_REPEATS` (default 3,
//! the minimum device time is reported). `../README.md` lists every harness
//! with its command and variables.
//!
//! Every generation aims a 65×65 lattice inside the unit disk from the host
//! f64 chief row, once with the wave path's 25×25 host seed table and once
//! with the chief prediction alone (the candidate evaluator's seeding; every
//! third lattice node, below the device-table threshold). The generated
//! start rays are retraced to the stop in f64, so the reported residual is
//! the true stop miss, not the kernel's own estimate.
#![cfg(target_os = "macos")]
use mklight_core::{JobContext, RayAiming, Vec3, samples};
use mklight_engine::{Engine, EngineConfig};
use mklight_gpu::{ComputeBackend, aiming::*, consts::*, resident::*, testutil};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{Ray, SequentialTracer, TraceOptions};
use serde_json::json;

#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn systems() -> Vec<(String, SequentialTracer)> {
    if let Ok(path) = std::env::var("MKLIGHT_STUDY_PROJECT") {
        let engine = Engine::new(EngineConfig::default());
        let session = engine.open_session(std::path::Path::new(&path)).unwrap();
        let resolved = session.resolved().unwrap();
        let t = SequentialTracer::new(resolved).unwrap();
        return vec![("project".into(), t)];
    }
    ["cooke_triplet", "double_gauss", "petzval_asphere"]
        .into_iter()
        .map(|name| {
            let mut model = samples::all()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap()
                .1;
            model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
            (name.to_string(), testutil::tracer(model))
        })
        .collect()
}

fn lattice() -> Vec<[f64; 2]> {
    let n = 65;
    let mut out = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let x = -1. + 2. * i as f64 / (n - 1) as f64;
            let y = -1. + 2. * j as f64 / (n - 1) as f64;
            if x * x + y * y <= 1. {
                out.push([x, y]);
            }
        }
    }
    out
}

#[test]
#[ignore = "real Metal: aiming cost and convergence study"]
fn aiming_cost_versus_tolerance() {
    let tolerances: Vec<f64> = env_list("MKLIGHT_STUDY_TOLERANCES", "1e-10,1e-8,1e-6,1e-5,1e-4")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let repeats: usize = std::env::var("MKLIGHT_STUDY_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let rows_limit = std::env::var("MKLIGHT_STUDY_ROWS").unwrap_or_else(|_| "all".into());
    let b = &MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
    })
    .unwrap()
    .expect("real Metal required");
    let full = lattice();
    // Below `TABLE_MIN_RAYS_PER_ROW`, so no device table is built.
    let sparse: Vec<[f64; 2]> = full.iter().copied().step_by(3).collect();
    assert!(sparse.len() < TABLE_MIN_RAYS_PER_ROW);
    for (name, t) in systems() {
        let radius = t.pupil().r_stop;
        let rows: Vec<(usize, usize)> = (0..t.fields().len())
            .flat_map(|f| (0..t.num_wavelengths()).map(move |w| (f, w)))
            .collect();
        let rows = match rows_limit.as_str() {
            "all" => rows,
            k => rows.into_iter().take(k.parse().unwrap()).collect(),
        };
        for (f, w) in rows {
            let table = host_tables(&t, &[(t.field(f), w)], 25);
            let seed = ChiefSeed::host(&t, t.field(f), w);
            for seeding in ["table25", "chief"] {
                let points: &[[f64; 2]] = if seeding == "chief" { &sparse } else { &full };
                let n = points.len();
                for &tol in &tolerances {
                    let (p, h) = support::make_plan(b, &t, f, w, points, false, 60, false);
                    let mut p = p;
                    let seeds = support::upload(b, &mut p, 1, &[seed]);
                    let tv = support::upload(b, &mut p, table.len(), &table);
                    for op in &mut p.operations {
                        if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
                            g.seeds = Some(seeds.clone());
                            g.tolerance = tol as f32;
                            g.field_key = None;
                            g.options.cull_outside = true;
                            if seeding == "table25" {
                                g.options.table = Some(tv.clone());
                                g.options.table_side = 25;
                            }
                        }
                    }
                    let mut best: Option<(PlanResult, Vec<_>)> = None;
                    for _ in 0..repeats {
                        let r = b
                            .submit_plan(p.clone(), &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        let d = support::read_all(b, &r);
                        if best
                            .as_ref()
                            .is_none_or(|(x, _)| r.stats.device_seconds < x.stats.device_seconds)
                        {
                            best = Some((r, d));
                        }
                    }
                    let (r, d) = best.unwrap();
                    let aim_stage = r
                        .stats
                        .stages
                        .iter()
                        .find(|s| s.name == "generate_aim")
                        .map_or(0., |s| s.device_seconds);
                    let trace_stage = r
                        .stats
                        .stages
                        .iter()
                        .find(|s| s.name == "trace_resident")
                        .map_or(0., |s| s.device_seconds);
                    let (ff, status) = (support::floats(&d[0]), support::uints(&d[1]));
                    let df = |k: usize, i: usize| ff[k * n + i] as f64 + ff[(k + 9) * n + i] as f64;
                    let (mut worst, mut worst_accepted, mut sum, mut failed, mut lost, mut count) =
                        (0f64, 0f64, 0f64, 0usize, 0usize, 0usize);
                    // Iterations of converged / failed rays by bin
                    // (≤2, ≤4, ≤8, ≤16, ≤32, ≤64, >64), and the failed
                    // rays' kernel-reported residuals.
                    let bin = |it: u32| {
                        [2, 4, 8, 16, 32, 64]
                            .iter()
                            .position(|&b| it <= b)
                            .unwrap_or(6)
                    };
                    let (mut ok_bins, mut fail_bins) = ([0usize; 7], [0usize; 7]);
                    let (mut fail_res_min, mut fail_res_max) = (f64::INFINITY, 0f64);
                    for (i, pt) in points.iter().enumerate() {
                        if status[i] == MKL_AIM_OUTSIDE {
                            continue;
                        }
                        count += 1;
                        failed += usize::from(status[i] != 0);
                        let it = status[n + i];
                        if status[i] == 0 {
                            ok_bins[bin(it)] += 1;
                        } else {
                            fail_bins[bin(it)] += 1;
                            let res = f32::from_bits(status[2 * n + i]) as f64;
                            fail_res_min = fail_res_min.min(res);
                            fail_res_max = fail_res_max.max(res);
                        }
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
                        if !stop.status.is_ok() {
                            lost += 1;
                            continue;
                        }
                        let (x, y) = t.field(f).point.vignetting.apply(pt[0], pt[1]);
                        let residual = (stop.ray.pos.x - x * radius)
                            .hypot(stop.ray.pos.y - y * radius)
                            / radius;
                        worst = worst.max(residual);
                        sum += residual;
                        if status[i] == 0 {
                            worst_accepted = worst_accepted.max(residual);
                        }
                    }
                    let a = &r.stats.aiming;
                    println!(
                        "AIM_STUDY {}",
                        json!({"system":name,"field":f,"wavelength":w,"seeding":seeding,
                                "tolerance":tol,"rays":count,"device_ms":r.stats.device_seconds*1e3,
                                "aim_ms":aim_stage*1e3,"trace_ms":trace_stage*1e3,
                                "wall_ms":r.stats.wall_seconds*1e3,"command_buffers":r.stats.command_buffers,
                                "iterations_per_ray":a.iterations as f64/a.rays.max(1) as f64,
                                "max_iterations":a.max_iterations,"histogram":a.histogram,
                                "reported_failed":failed,"lost_at_stop":lost,
                                "converged_by_iterations":ok_bins,"failed_by_iterations":fail_bins,
                                "failed_residual_min":fail_res_min,"failed_residual_max":fail_res_max,
                                "max_residual":worst,"max_accepted_residual":worst_accepted,
                                "mean_residual":sum/count.max(1) as f64})
                    );
                    b.release_systems(h).unwrap();
                }
            }
        }
    }
}
