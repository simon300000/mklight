//! CPU f64 tracer throughput on a local project; nothing is embedded.
//! MKLIGHT_BENCH_PROJECT=/path/project.mklight cargo test --release -p
//! mklight-engine --test cpu_trace_bench -- --ignored --nocapture
//! Optional: MKLIGHT_BENCH_RINGS (hexapolar rings, default 24),
//! MKLIGHT_BENCH_REPEATS (default 3; best time is reported).
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

    let (t_prep, tracer) = best(repeats, || SequentialTracer::new(Arc::clone(&sys)).unwrap());
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
    println!("  checksum {:.17e}", checksum);
}
