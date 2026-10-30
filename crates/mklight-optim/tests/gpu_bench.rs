//! Merit-evaluation throughput, CPU f64 reference against the GPU-batched
//! evaluator on the Metal device, for the Double Gauss default merit at
//! batch sizes 1 … 4096 (ignored by default; skips without a Metal
//! device). Run with
//!
//! ```text
//! cargo test -p mklight-optim --release --test gpu_bench -- --ignored --nocapture
//! ```
//!
//! Results are recorded in `docs/OPTIMIZATION.md` (GPU-batched evaluation).

use std::sync::Arc;
use std::time::Instant;

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, OpticalSystem, SurfaceId, samples};
use mklight_gpu::ComputeBackend;
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec};
use mklight_optim::merit::operand::WavefrontReference;
use mklight_optim::merit::{
    GpuEvalSettings, GpuTraceEvaluator, OperandEvaluator, OperandRegistry, OpticalProblem,
    PupilSampling, TraceEvaluator, VariableSpec,
};
use mklight_optim::problem::{CandidateBatch, Problem};

fn variables() -> Vec<VariableSpec> {
    let mut v: Vec<VariableSpec> = [1u64, 2, 3, 5, 7, 9, 10, 11]
        .iter()
        .map(|&id| {
            VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(id),
            })
        })
        .collect();
    v.push(VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(11),
    }));
    v
}

fn problem<E: OperandEvaluator>(
    sys: &OpticalSystem,
    gen_: &DefaultMeritSpec,
    ev: E,
) -> OpticalProblem<E> {
    let vars = variables();
    let spec = gen_.generate_for(sys, &vars).unwrap();
    OpticalProblem::new(sys.clone(), vars, spec, ev, &OperandRegistry::builtin()).unwrap()
}

/// `n` distinct candidates around `x0`.
fn batch(x0: &[f64], n: usize) -> CandidateBatch {
    let mut b = CandidateBatch::with_capacity(x0.len(), n);
    for k in 0..n {
        let mut x = x0.to_vec();
        x[k % x0.len()] *= 1.0 + 1e-5 * (k as f64 + 1.0);
        b.push(&x);
    }
    b
}

/// Evaluations per second of `p` on batches of `n` (best of repeated runs
/// over at least ~0.3 s).
fn rate<P: Problem>(p: &P, b: &CandidateBatch) -> f64 {
    let ctx = JobContext::new();
    p.evaluate_batch(b, &ctx).unwrap(); // warm-up
    let mut best = 0.0f64;
    let t_all = Instant::now();
    while t_all.elapsed().as_secs_f64() < 0.3 {
        let t = Instant::now();
        p.evaluate_batch(b, &ctx).unwrap();
        best = best.max(b.len() as f64 / t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn double_gauss_cpu_vs_gpu_throughput() {
    #[cfg(target_os = "macos")]
    let gpu: Option<Arc<dyn ComputeBackend>> = mklight_metal::MetalBackend::new()
        .ok()
        .flatten()
        .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>);
    #[cfg(not(target_os = "macos"))]
    let gpu: Option<Arc<dyn ComputeBackend>> = None;
    let Some(gpu) = gpu else {
        eprintln!("SKIP: no Metal device");
        return;
    };
    let sys = samples::double_gauss();
    println!(
        "Double Gauss f50 F/2, {} ({} rayon threads)",
        gpu.info().backend_id(),
        rayon::current_num_threads()
    );
    let spot = |rings, arms| DefaultMeritSpec {
        sampling: PupilSampling {
            rings,
            arms,
            ..Default::default()
        },
        ..Default::default()
    };
    for (name, gen_) in [
        ("spot, 3 rings x 6 arms (half pupil)", spot(3, 6)),
        (
            "wavefront, 3 rings x 6 arms (half pupil)",
            DefaultMeritSpec {
                criterion: DefaultCriterion::Wavefront {
                    reference: WavefrontReference::Centroid,
                },
                ..Default::default()
            },
        ),
        ("spot, 6 rings x 12 arms (half pupil)", spot(6, 12)),
        ("spot, 12 rings x 24 arms (half pupil)", spot(12, 24)),
    ] {
        let cpu = problem(&sys, &gen_, TraceEvaluator::builtin());
        let ev = GpuTraceEvaluator::new(
            gpu.clone(),
            TraceEvaluator::builtin(),
            &sys,
            GpuEvalSettings {
                // `MKLIGHT_BENCH_GPU_BATCH` overrides the sub-batch size.
                batch: std::env::var("MKLIGHT_BENCH_GPU_BATCH")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(GpuEvalSettings::default().batch),
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert!(ev.is_active(), "{:?}", ev.qualification().reason);
        let g = problem(&sys, &gen_, ev.clone().composite());
        let x0 = cpu.start_point().unwrap();
        println!("-- {name}: {} residuals", cpu.num_residuals());
        println!(
            "{:>6} {:>14} {:>14} {:>8}",
            "batch", "CPU evals/s", "GPU evals/s", "GPU/CPU"
        );
        for n in [1usize, 4, 16, 64, 256, 1024, 4096] {
            let b = batch(&x0, n);
            let (c, gr) = (rate(&cpu, &b), rate(&g, &b));
            println!("{n:>6} {c:>14.0} {gr:>14.0} {:>8.2}", gr / c);
        }
        let st = ev.stats();
        println!(
            "   GPU: {} candidates, {} rays, {} dispatches, {:.1} ms device time ({:.2} µs/candidate)",
            st.gpu_candidates,
            st.rays,
            st.dispatches,
            st.device_seconds * 1e3,
            st.device_seconds * 1e6 / st.gpu_candidates.max(1) as f64
        );
    }
}
