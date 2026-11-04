//! Bounded, resumable device aiming: continuation rounds reproduce the
//! monolithic CPU-f32 specification bitwise, excluded samples are neither
//! aimed nor counted, and host seed tables of another side converge to the
//! same rays within tolerance.
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;
use mklight_core::{JobContext, RayAiming};
use mklight_gpu::{aiming::*, consts::*, resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::PupilSampling;
use support::*;

fn backend() -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap()
    .expect("P1 requires host Metal hardware");
    assert!(b.precision().probe.error_free_transforms());
    b
}
fn run(b: &dyn ComputeBackend, p: ResidentPlan) -> (PlanResult, Vec<BufferData>) {
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let d = read_all(b, &r);
    (r, d)
}
fn with_options(mut p: ResidentPlan, f: impl Fn(&mut GenerateAim)) -> ResidentPlan {
    for op in &mut p.operations {
        if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
            f(g);
        }
    }
    p
}
fn reports(d: &[BufferData], n: usize) -> Vec<[u32; 4]> {
    let r = uints(&d[1]);
    (0..n)
        .map(|i| std::array::from_fn(|k| r[k * n + i]))
        .collect()
}

/// With one iteration unit per command every real aim (and every device
/// table node) runs through several continuation rounds. Rays, reports,
/// hits and fallback counts must be bitwise the specification's.
#[test]
fn continuation_rounds_match_specification() {
    // Read once by the executor; this test binary aims nothing before it.
    unsafe { std::env::set_var("MKLIGHT_AIM_BUDGET", "1,1") };
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut small = vec![[0., 0.]];
    small.extend((PupilSampling::Grid { n: 7 }).points());
    let big = (PupilSampling::Grid { n: 64 }).points();
    assert!(big.len() >= TABLE_MIN_RAYS_PER_ROW);
    let (mut rounds, mut misses) = (0, 0);
    for (name, mut model) in testutil::xval_systems().into_iter().take(10) {
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let last = t.fields().len() - 1;
        for (field, points) in [(0, &small), (last, &small), (last, &big)] {
            let (p, h) = make_plan(&gpu, &t, field, 0, points, false, 40, false);
            let (pc, hc) = make_plan(&cpu, &t, field, 0, points, false, 40, false);
            let (r, data) = run(&gpu, p.clone());
            let (_, dc) = run(&cpu, pc);
            bits(&data, &dc);
            // Speculative rounds without readback: too few (one round of one
            // unit) must be detected and rerun, enough must need no rerun;
            // both give the specification's bits.
            for (spec, missed) in [(1, true), (400, false)] {
                let ps = with_options(p.clone(), |g| g.options.speculative_rounds = spec);
                let (rs, ds) = run(&gpu, ps);
                bits(&ds, &dc);
                if missed {
                    misses += rs.stats.speculation_misses;
                } else {
                    assert_eq!(rs.stats.speculation_misses, 0, "{name} f={field}");
                }
            }
            let e = check_f64(&t, field, 0, points, &data, false, false);
            assert_eq!(
                r.stats.fallback_count, e.fallback as u64,
                "{name} f={field}"
            );
            rounds += r.stats.command_buffers;
            eprintln!(
                "{name} f={field} n={} {e:?} buffers={}",
                points.len(),
                r.stats.command_buffers
            );
            gpu.release_systems(h).unwrap();
            cpu.release_systems(hc).unwrap();
        }
    }
    assert!(
        rounds > 30,
        "continuation rounds ran: {rounds} command buffers"
    );
    assert!(misses > 5, "speculation misses rerun: {misses}");
}

/// Culled (outside the disk) and skip-flagged samples carry
/// `MKL_AIM_OUTSIDE`, are not traced, and change nothing for the others.
#[test]
fn excluded_samples_are_neither_aimed_nor_counted() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut model = mklight_core::samples::double_gauss();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let field = t.fields().len() - 1;
    let mut points = (PupilSampling::Grid { n: 9 }).points();
    let inside = points.len();
    points.extend([[1.5, 0.], [0., -1.25], [0.9, 0.9], [-1.0000001, 0.]]);
    let n = points.len();
    let (plain, h) = make_plan(&gpu, &t, field, 0, &points, false, 40, false);
    let (r0, d0) = run(&gpu, plain.clone());
    let culled = with_options(plain.clone(), |g| g.options.cull_outside = true);
    let (r1, d1) = run(&gpu, culled.clone());
    let a = reports(&d0, n);
    let b = reports(&d1, n);
    for i in 0..inside {
        assert_eq!(a[i], b[i], "inside sample {i}");
    }
    let outside_failures = (inside..n).filter(|&i| a[i][0] != 0).count();
    let hit_status = uints(&d1[4]);
    for i in inside..n {
        assert_eq!(b[i], [MKL_AIM_OUTSIDE, 0, 0, 0], "outside sample {i}");
        assert_eq!(hit_status[i], MKL_AIM_OUTSIDE);
    }
    for (k, (x, y)) in floats(&d0[0]).iter().zip(floats(&d1[0])).enumerate() {
        if k % n < inside {
            assert_eq!(x.to_bits(), y.to_bits(), "ray plane word {k}");
        }
    }
    assert_eq!(
        r1.stats.fallback_count + outside_failures as u64,
        r0.stats.fallback_count
    );
    assert_eq!(r1.stats.aiming.rays, inside as u64);
    assert_eq!(r0.stats.aiming.rays, n as u64);
    // The CPU-f32 specification excludes the same samples.
    let (pc, hc) = make_plan(&cpu, &t, field, 0, &points, false, 40, false);
    let (_, dc) = run(&cpu, with_options(pc, |g| g.options.cull_outside = true));
    bits(&d1, &dc);
    // A skip-flagged sample is excluded wherever it lies.
    let mut p = plain;
    let samples: Vec<Sample> = points
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            let mut s = Sample::new(pt, 0, 0x100000000 + i as u64);
            if i == 3 {
                s.field |= MKL_AIM_SKIP;
            }
            s
        })
        .collect();
    let view = upload(&gpu, &mut p, n, &samples);
    let (r2, d2) = run(&gpu, with_options(p, |g| g.samples = view.clone()));
    let c = reports(&d2, n);
    assert_eq!(c[3], [MKL_AIM_OUTSIDE, 0, 0, 0]);
    for i in (0..n).filter(|&i| i != 3) {
        assert_eq!(a[i], c[i], "sample {i}");
    }
    assert_eq!(r2.stats.aiming.rays, n as u64 - 1);
    gpu.release_systems(h).unwrap();
    cpu.release_systems(hc).unwrap();
}

/// A host f64 seed table of 13 nodes per axis, a device table of that side
/// and the default device table all converge every ray to the requested
/// tolerance, with starts equal to within that tolerance; the device uses a
/// host table exactly as the specification does.
#[test]
fn host_and_small_tables_converge_within_tolerance() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut model = mklight_core::samples::cooke_triplet();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let field = 1;
    let points = (PupilSampling::Grid { n: 64 }).points();
    assert!(points.len() >= TABLE_MIN_RAYS_PER_ROW);
    let n = points.len();
    let (plain, h) = make_plan(&gpu, &t, field, 0, &points, false, 40, false);
    let (r0, d0) = run(&gpu, plain.clone());
    let small = with_options(plain.clone(), |g| g.options.table_side = 13);
    let (r1, d1) = run(&gpu, small);
    let table = host_tables(&t, &[(t.field(field), 0)], 13);
    assert_eq!(table.len(), 169);
    assert!(table.iter().filter(|n| n.status == 0).count() > 100);
    let mut hosted = plain.clone();
    let view = upload(&gpu, &mut hosted, table.len(), &table);
    let hosted = with_options(hosted, |g| {
        g.options.table = Some(view.clone());
        g.options.table_side = 13;
    });
    let (r2, d2) = run(&gpu, hosted.clone());
    let (pc, hc) = make_plan(&cpu, &t, field, 0, &points, false, 40, false);
    let mut pc = pc;
    let viewc = upload(&cpu, &mut pc, table.len(), &table);
    let (_, dc) = run(
        &cpu,
        with_options(pc, |g| {
            g.options.table = Some(viewc.clone());
            g.options.table_side = 13;
        }),
    );
    bits(&d2, &dc);
    for (label, r, d) in [
        ("device 37", &r0, &d0),
        ("device 13", &r1, &d1),
        ("host 13", &r2, &d2),
    ] {
        assert_eq!(r.stats.fallback_count, 0, "{label}");
        let e = check_f64(&t, field, 0, &points, d, false, false);
        assert_eq!(e.fallback, 0, "{label}");
        // Coarser tables need a Newton iteration or two more per ray (the
        // triplet: 1.1 at 37 nodes per axis, 2.6 at 13); none needs the
        // chief or continuation schedule.
        let mean = r.stats.aiming.iterations as f64 / r.stats.aiming.rays as f64;
        eprintln!("{label}: mean iterations {mean:.2} {e:?}");
        assert!(mean < 4., "{label}: mean iterations {mean}");
    }
    // Start positions and directions (hi + lo) of rays converged to 1e-10 of
    // the stop radius from different seeds agree to about that fraction of
    // the entrance pupil.
    let value = |f: &[f32], k: usize, i: usize| f[k * n + i] as f64 + f[(k + 9) * n + i] as f64;
    for (label, d) in [("device 13", &d1), ("host 13", &d2)] {
        let (a, b) = (floats(&d0[0]), floats(&d[0]));
        let worst = (0..6 * n)
            .map(|j| (value(a, j / n, j % n) - value(b, j / n, j % n)).abs())
            .fold(0., f64::max);
        assert!(worst <= 1e-8, "{label}: start rays differ by {worst}");
    }
    gpu.release_systems(h).unwrap();
    cpu.release_systems(hc).unwrap();
}

/// A schedule cap bounds every ray's Newton iterations over its whole
/// schedule (`AimOptions::schedule_cap`): no ray reports more, a ray the
/// uncapped schedule converges within the cap is bitwise the uncapped run's,
/// and the device is bitwise the CPU-f32 specification under every cap.
/// Samples far outside the pupil disk (not culled) are unaimable and would
/// otherwise run the whole schedule.
#[test]
fn schedule_cap_bounds_the_schedule_bitwise() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut points = (PupilSampling::Grid { n: 9 }).points();
    points.extend([[1.6, 0.], [0., -1.7], [1.3, 1.3], [-1.8, 0.4]]);
    let n = points.len();
    let (mut capped, mut kept) = (0, 0);
    for (name, mut model) in testutil::xval_systems().into_iter().take(10) {
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let field = t.fields().len() - 1;
        let (plain, h) = make_plan(&gpu, &t, field, 0, &points, false, 40, false);
        let (pc, hc) = make_plan(&cpu, &t, field, 0, &points, false, 40, false);
        let (_, d0) = run(&gpu, plain.clone());
        let a = reports(&d0, n);
        let f0 = floats(&d0[0]);
        for cap in [1u32, 6, 24] {
            let set = |g: &mut GenerateAim| g.options.schedule_cap = cap;
            let (_, d) = run(&gpu, with_options(plain.clone(), set));
            let (_, dc) = run(&cpu, with_options(pc.clone(), set));
            bits(&d, &dc);
            let b = reports(&d, n);
            let f = floats(&d[0]);
            for i in 0..n {
                assert!(
                    b[i][1] <= cap,
                    "{name}: ray {i} ran {} iterations under cap {cap}",
                    b[i][1]
                );
                if a[i][0] == 0 && a[i][1] <= cap {
                    assert_eq!(a[i], b[i], "{name}: converged ray {i} under cap {cap}");
                    for k in 0..15 {
                        assert_eq!(
                            f[k * n + i].to_bits(),
                            f0[k * n + i].to_bits(),
                            "{name}: ray {i} plane {k} under cap {cap}"
                        );
                    }
                    kept += 1;
                } else {
                    capped += 1;
                }
            }
        }
        gpu.release_systems(h).unwrap();
        cpu.release_systems(hc).unwrap();
    }
    assert!(capped > 100, "capped rays: {capped}");
    assert!(kept > 1000, "rays unchanged under a cap: {kept}");
}
