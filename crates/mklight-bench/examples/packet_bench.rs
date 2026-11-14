//! Throughput of the CPU packet path (`trace_rays` lane kernel) inside the
//! analyses that use it: `trace_grid`, `opd_grid` and a traced wave pupil,
//! on the Cooke triplet, double Gauss and Petzval asphere samples, on one thread and on the
//! global rayon pool. Used to judge kernel changes (CONVENTIONS §7.7).
//!
//!     cargo run --release -p mklight-bench --example packet_bench -- \
//!         [RINGS] [REPEATS] [LENS.zmx|LENS.seq]
//!
//! RINGS: hexapolar pupil rings (default 48); REPEATS: runs per timing, the
//! best is reported (default 7); LENS: an imported prescription benched as a
//! third case (`zmx`). One line per case and thread count: times in ms of
//! `trace_grid`, `opd_grid`, `trace_pupil` (129 lattice rays, last field),
//! `trace_rays` of the generated starts alone (one thread in both rows),
//! and checksums of the grid points and OPDs. A second line per case times
//! the bare traces of those starts on the current pool (rayon over chunks of
//! 1024 rays): `trace_ray` one ray at a time (scalar) and `trace_rays`
//! (packets), in ns per ray, median and best of REPEATS, with a checksum of
//! the result bits.

use std::time::Instant;

use mklight_analysis::tracer_for;
use mklight_core::{JobContext, samples};
use mklight_trace::{Grid, PupilSampling, opd_grid, trace_grid, trace_rays};
use mklight_trace::{TraceResult, trace_ray};
use mklight_wave::{PupilSettings, traced::trace_pupil};
use rayon::prelude::*;

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

/// Median and best seconds of `repeats` runs (after one warm-up run).
fn median_best<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, f64, T) {
    let mut out = f();
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        out = f();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0], out)
}

/// Order-dependent hash of every bit of the results.
fn checksum(rs: &[TraceResult]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for r in rs {
        let (p, d) = (r.ray.pos, r.ray.dir);
        for v in [
            p.x,
            p.y,
            p.z,
            d.x,
            d.y,
            d.z,
            r.opl.hi,
            r.opl.lo,
            r.transmittance,
        ] {
            h = (h ^ v.to_bits()).wrapping_mul(0x100_0000_01b3);
        }
        h = (h ^ r.surface as u64 ^ (r.status.is_ok() as u64) << 32).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn arg_usize(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .map(|s| s.parse().expect("a number"))
        .unwrap_or(default)
}

fn run(name: &str, threads: &str) {
    let rings = arg_usize(1, 48);
    let repeats = arg_usize(2, 7);
    let system = match name {
        "cooke" => samples::cooke_triplet(),
        "dgauss" => samples::double_gauss(),
        "petzval" => samples::petzval_asphere(),
        _ => {
            let path = std::env::args().nth(3).unwrap();
            let lib = mklight_core::materials::MaterialLibrary::with_builtin();
            mklight_core::import::read_prescription_file(path, &lib)
                .unwrap()
                .system
        }
    };
    let tracer = tracer_for(system).unwrap();
    let sys = tracer.system();
    let pupil = PupilSampling::Hexapolar { rings }.points();
    let fields = tracer.fields().to_vec();
    let wls: Vec<usize> = (0..tracer.num_wavelengths()).collect();
    let grid = Grid {
        fields: &fields,
        wavelengths: &wls,
        pupil: &pupil,
    };
    let opts = tracer.options();
    let (t_grid, g) = best(repeats, || trace_grid(&tracer, &grid, &opts));
    let (t_opd, o) = best(repeats, || opd_grid(&tracer, &grid, &opts));
    // The packet trace alone, from the generated starts.
    let (starts, swl): (Vec<_>, Vec<_>) = (0..grid.len())
        .filter_map(|i| {
            let (f, w, p) = grid.split(i);
            tracer
                .generate(&fields[f], wls[w], p_of(&pupil, p).0, p_of(&pupil, p).1)
                .ok()
                .map(|g| (g.ray, wls[w]))
        })
        .unzip();
    let (t_pack, _) = best(repeats, || trace_rays(sys, &starts, &swl, &opts));
    const CHUNK: usize = 1024;
    let per_ray = 1e9 / starts.len() as f64;
    let (s_med, s_best, s_out) = median_best(repeats, || {
        starts
            .par_chunks(CHUNK)
            .zip(swl.par_chunks(CHUNK))
            .flat_map_iter(|(s, w)| {
                s.iter()
                    .zip(w)
                    .map(|(s, &w)| trace_ray(sys, s, w, &opts, None))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let (p_med, p_best, p_out) = median_best(repeats, || {
        starts
            .par_chunks(CHUNK)
            .zip(swl.par_chunks(CHUNK))
            .flat_map_iter(|(s, w)| trace_rays(sys, s, w, &opts))
            .collect::<Vec<_>>()
    });
    assert_eq!(
        checksum(&s_out),
        checksum(&p_out),
        "scalar and packets differ"
    );
    println!(
        "{name:7} {threads:4} rays={:6}  scalar {:7.1} ns/ray (best {:7.1})  \
         packets {:7.1} ns/ray (best {:7.1})  [{:016x}]",
        starts.len(),
        s_med * per_ray,
        s_best * per_ray,
        p_med * per_ray,
        p_best * per_ray,
        checksum(&s_out),
    );
    let settings = PupilSettings {
        rays: 129,
        ..Default::default()
    };
    let ctx = JobContext::new();
    let last = fields.len() - 1;
    let (t_wave, _) = best(repeats, || {
        trace_pupil(
            &tracer,
            &fields[last],
            sys.primary_wavelength,
            &settings,
            &ctx,
        )
        .unwrap()
    });
    let sum_g: f64 = g
        .iter()
        .filter(|r| r.status.is_ok())
        .map(|r| r.point.x + r.point.y)
        .sum();
    let sum_o: f64 = o.iter().flatten().sum();
    println!(
        "{name:6} {threads:4} rays={:6}  trace_grid {:8.3} ms  opd_grid {:8.3} ms  \
         wave_pupil {:8.3} ms  trace_rays {:8.3} ms  [{:.17e} {:.17e}]",
        grid.len(),
        1e3 * t_grid,
        1e3 * t_opd,
        1e3 * t_wave,
        1e3 * t_pack,
        sum_g,
        sum_o
    );
}

fn p_of(pupil: &[[f64; 2]], p: usize) -> (f64, f64) {
    (pupil[p][0], pupil[p][1])
}

fn main() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut names = vec!["cooke", "dgauss", "petzval"];
    if std::env::args().nth(3).is_some() {
        names.push("zmx");
    }
    for name in names {
        one.install(|| run(name, "1"));
        run(name, &format!("{}", rayon::current_num_threads()));
    }
}
