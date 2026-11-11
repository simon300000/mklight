//! CPU f64 tracer throughput on a local project; nothing is embedded.
//! MKLIGHT_BENCH_PROJECT=/path/project.mklight cargo test --release -p
//! mklight-bench --test tests cpu_trace_bench:: -- --ignored --nocapture
//! Optional: MKLIGHT_BENCH_RINGS (hexapolar rings, default 24),
//! MKLIGHT_BENCH_REPEATS (default 3; best time is reported),
//! MKLIGHT_BENCH_CHUNK (rays per `generate_many`/`trace_many` call, default
//! 256), MKLIGHT_BENCH_AIM_STOP (`conditioned` or `conditioned:preview`).
//! `../README.md` lists every harness with its command and variables.
//!
//! Stages, single thread unless noted: tracer preparation (fields and chief
//! aims), ray generation (aiming) alone, the clipped image trace of
//! pre-generated rays alone, generation plus trace, and `trace_grid` on the
//! global rayon pool. A checksum over every traced ray guards against a
//! faster path that changes results.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mklight_engine::{Engine, EngineConfig};
use mklight_trace::{
    Grid, PACKET, PupilSampling, RayRequest, SequentialTracer, trace_grid, trace_ray, trace_rays,
};

fn best<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = None;
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let t = Instant::now();
        let v = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(v);
    }
    (best, out.unwrap())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "local project measurement"]
fn cpu_trace_throughput() {
    let path = std::env::var("MKLIGHT_BENCH_PROJECT").expect("set MKLIGHT_BENCH_PROJECT");
    let engine = Engine::new(EngineConfig::default());
    let session = engine.open_session(Path::new(&path)).unwrap();
    let sys = session.resolved().unwrap();
    let rings = env_usize("MKLIGHT_BENCH_RINGS", 24);
    let repeats = env_usize("MKLIGHT_BENCH_REPEATS", 3);

    // MKLIGHT_BENCH_AIM_STOP=conditioned[:preview] derives the aim stop from
    // the propagated error (CONVENTIONS §7) instead of the requested
    // tolerance.
    let aim_stop = match std::env::var("MKLIGHT_BENCH_AIM_STOP").as_deref() {
        Ok("conditioned") => mklight_trace::AimStop::Conditioned(mklight_trace::AimBudget::DESIGN),
        Ok("conditioned:preview") => {
            mklight_trace::AimStop::Conditioned(mklight_trace::AimBudget::PREVIEW)
        }
        _ => mklight_trace::AimStop::Requested,
    };
    let settings = mklight_trace::TracerSettings {
        aim_stop,
        ..Default::default()
    };
    let (t_prep, tracer) = best(repeats, || {
        SequentialTracer::with_settings(Arc::clone(&sys), settings).unwrap()
    });
    for (i, f) in tracer.fields().iter().enumerate() {
        for (w, a) in f.aims.iter().enumerate() {
            if let Some(c) = a.conditioning {
                println!(
                    "  field {i} wl {w}: stop tolerance {:.3e} (requested {:.1e}); sensitivity \
                     position {:.3} direction {:.3e} /mm opd {:.3e} waves/mm ({} probes)",
                    tracer.aim_tolerance_of(f, w),
                    tracer.requested_tolerance(),
                    c.position,
                    c.direction,
                    c.opd,
                    c.probes
                );
            }
        }
    }
    let pupil = PupilSampling::Hexapolar { rings }.points();
    let fields = tracer.fields().to_vec();
    let wls: Vec<usize> = (0..tracer.num_wavelengths()).collect();
    let grid = Grid {
        fields: &fields,
        wavelengths: &wls,
        pupil: &pupil,
    };
    let n = grid.len();
    let opts = tracer.options();

    let chunk = env_usize("MKLIGHT_BENCH_CHUNK", 256);
    let requests: Vec<RayRequest> = (0..n)
        .map(|i| {
            let (f, w, p) = grid.split(i);
            (&fields[f], wls[w], pupil[p])
        })
        .collect();
    let (t_gen, full) = best(repeats, || {
        requests
            .iter()
            .map(|&(f, w, p)| tracer.generate(f, w, p[0], p[1]))
            .collect::<Vec<_>>()
    });
    let (t_gen_many, many) = best(repeats, || {
        requests
            .chunks(chunk)
            .flat_map(|c| tracer.generate_many(c))
            .collect::<Vec<_>>()
    });
    for (a, b) in full.iter().zip(&many) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "generate_many differs");
    }
    let mut aim_stats = mklight_trace::AimStats::default();
    for c in requests.chunks(chunk) {
        tracer.generate_many_stats(c, &mut aim_stats);
    }
    let starts: Vec<_> = full.iter().map(|g| g.ok()).collect();
    let iterations: u64 = starts
        .iter()
        .flatten()
        .map(|g| g.report.iterations as u64)
        .sum();
    let generated = starts.iter().flatten().count();
    let (t_trace, traced) = best(repeats, || {
        starts
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let (_, w, _) = grid.split(i);
                g.as_ref()
                    .map(|g| trace_ray(&sys, &g.ray, wls[w], &opts, None))
            })
            .collect::<Vec<_>>()
    });
    let ok = traced.iter().flatten().filter(|r| r.status.is_ok()).count();
    // Packet trace of the same starts (one wavelength per ray).
    let (gen_starts, gen_wls): (Vec<_>, Vec<_>) = starts
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (g.ray, wls[grid.split(i).1])))
        .unzip();
    let (t_packet, packed) = best(repeats, || trace_rays(&sys, &gen_starts, &gen_wls, &opts));
    // Jacobian trace cost breakdown (aiming): plain packet trace to the
    // stop, the same with records, and the differentiation pass alone.
    let stop_opts = mklight_trace::TraceOptions::unclipped().to_surface(sys.stop);
    let (t_stop, _) = best(repeats, || {
        trace_rays(&sys, &gen_starts, &gen_wls, &stop_opts)
    });
    let mut records: Vec<Vec<mklight_trace::SurfaceRecord>> =
        vec![Vec::with_capacity(sys.stop + 1); gen_starts.len()];
    let (t_stop_rec, stop_hits) = best(repeats, || {
        mklight_trace::trace_rays_recorded(&sys, &gen_starts, &gen_wls, &stop_opts, &mut records)
    });
    let tangent = mklight_trace::RayTangent {
        dpos: [mklight_core::Vec3::X, mklight_core::Vec3::Y],
        ddir: [mklight_core::Vec3::ZERO; 2],
    };
    let plan = tracer.tangent_plan();
    let (t_diff, n_diff) = best(repeats, || {
        stop_hits
            .iter()
            .zip(&records)
            .zip(&gen_wls)
            .filter(|((r, _), _)| r.status.is_ok())
            .filter(|((r, rec), w)| {
                plan.differentiate(&sys, **w, &tangent, rec, r.surface)
                    .is_ok()
            })
            .count()
    });
    let scalar: Vec<_> = traced.iter().flatten().copied().collect();
    assert_eq!(packed.len(), scalar.len());
    for (a, b) in packed.iter().zip(&scalar) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "packet trace differs");
    }
    let checksum: f64 = traced
        .iter()
        .flatten()
        .filter(|r| r.status.is_ok())
        .map(|r| r.ray.pos.x + r.ray.pos.y + r.opl.value())
        .sum();
    let (t_both, both) = best(repeats, || {
        requests
            .iter()
            .map(|&(f, w, p)| tracer.trace(f, w, p[0], p[1], &opts))
            .collect::<Vec<_>>()
    });
    let (t_both_many, both_many) = best(repeats, || {
        requests
            .chunks(chunk)
            .flat_map(|c| tracer.trace_many(c, &opts))
            .collect::<Vec<_>>()
    });
    for (a, b) in both.iter().zip(&both_many) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "trace_many differs");
    }
    let (t_par, par) = best(repeats, || trace_grid(&tracer, &grid, &opts));
    let par_checksum: f64 = par
        .iter()
        .filter(|r| r.status.is_ok())
        .map(|r| r.point.x + r.point.y + r.opl.value())
        .sum();
    assert_eq!(checksum.to_bits(), par_checksum.to_bits());

    let surfaces = sys.surfaces.len() - 1;
    let us = |t: f64, k: usize| 1e6 * t / k.max(1) as f64;
    println!(
        "CPU_TRACE_BENCH rays={n} generated={generated} ok={ok} surfaces={surfaces} stop={} \
         threads={}",
        sys.stop,
        rayon::current_num_threads()
    );
    println!("  prepare tracer          {:9.3} ms", 1e3 * t_prep);
    println!(
        "  generate (aim)          {:9.3} ms  {:7.3} us/ray  {:.2} Newton it/ray",
        1e3 * t_gen,
        us(t_gen, n),
        iterations as f64 / generated.max(1) as f64
    );
    println!(
        "  trace only              {:9.3} ms  {:7.3} us/ray  {:6.1} ns/surface",
        1e3 * t_trace,
        us(t_trace, generated),
        1e3 * us(t_trace, generated) / surfaces as f64
    );
    println!(
        "  generate_many ({chunk:4})    {:9.3} ms  {:7.3} us/ray",
        1e3 * t_gen_many,
        us(t_gen_many, n)
    );
    println!(
        "  aim work: {:.3} it/ray, {:.3} traces/ray ({} plain + {} with Jacobian), \
         {} fd fallbacks, {} continued, {} failed of {} rays",
        aim_stats.mean_iterations(),
        aim_stats.mean_traces(),
        aim_stats.traces,
        aim_stats.jacobian_traces,
        aim_stats.fd_fallbacks,
        aim_stats.continued,
        aim_stats.failed,
        aim_stats.rays
    );
    println!(
        "  trace packet ({PACKET:2})       {:9.3} ms  {:7.3} us/ray  {:6.1} ns/surface",
        1e3 * t_packet,
        us(t_packet, generated),
        1e3 * us(t_packet, generated) / surfaces as f64
    );
    println!(
        "  generate + trace        {:9.3} ms  {:7.3} us/ray  {:.3} Mray/s",
        1e3 * t_both,
        us(t_both, n),
        n as f64 / t_both / 1e6
    );
    println!(
        "  trace_many ({chunk:4})       {:9.3} ms  {:7.3} us/ray  {:.3} Mray/s",
        1e3 * t_both_many,
        us(t_both_many, n),
        n as f64 / t_both_many / 1e6
    );
    println!(
        "  trace_grid (rayon)      {:9.3} ms  {:7.3} us/ray  {:.3} Mray/s",
        1e3 * t_par,
        us(t_par, n),
        n as f64 / t_par / 1e6
    );
    println!(
        "  stop trace: packet {:6.1} ns/surface, with records {:6.1} ns/surface, \
         differentiate {:6.1} ns/surface ({n_diff} tangents)",
        1e3 * us(t_stop, generated) / sys.stop as f64,
        1e3 * us(t_stop_rec, generated) / sys.stop as f64,
        1e3 * us(t_diff, generated) / sys.stop as f64,
    );
    println!("  checksum {:.17e}", checksum);
}
