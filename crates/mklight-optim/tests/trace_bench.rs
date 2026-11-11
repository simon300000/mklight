//! Merit-evaluation throughput of the tracer-backed evaluator (ignored by
//! default). Run with
//!
//! ```text
//! cargo test -p mklight-optim --release --test tests trace_bench:: -- --ignored --nocapture
//! ```
//!
//! Results are recorded in `docs/OPTIMIZATION.md`.

use std::time::Instant;

use mklight_core::param::ParamPath;
use mklight_core::{JobContext, SurfaceId, samples};
use mklight_optim::deriv::{JacobianSettings, jacobian};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::{OperandRegistry, OpticalProblem, TraceEvaluator, VariableSpec};
use mklight_optim::problem::{CandidateBatch, Problem, evaluate_one};

fn problem(gen_: &DefaultMeritSpec) -> OpticalProblem<TraceEvaluator> {
    let sys = samples::double_gauss();
    let curved = [1u64, 2, 3, 5, 7, 9, 10, 11];
    let mut vars: Vec<VariableSpec> = curved
        .iter()
        .map(|&id| {
            VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(id),
            })
        })
        .collect();
    vars.push(VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(11),
    }));
    let spec = gen_.generate_for(&sys, &vars).unwrap();
    OpticalProblem::new(
        sys,
        vars,
        spec,
        TraceEvaluator::builtin(),
        &OperandRegistry::builtin(),
    )
    .unwrap()
}

fn rate(label: &str, evals: usize, rays: usize, f: impl FnOnce()) {
    let t = Instant::now();
    f();
    let s = t.elapsed().as_secs_f64();
    println!(
        "{label:<44} {:>9.0} merit evals/s  {:>6.3} ms/eval  {:>6.2} Mrays/s",
        evals as f64 / s,
        1e3 * s / evals as f64,
        (evals * rays) as f64 / s / 1e6
    );
}

#[test]
#[ignore]
fn double_gauss_default_merit_throughput() {
    let ctx = JobContext::new();
    println!(
        "Double Gauss f50 F/2, rayon threads: {}",
        rayon::current_num_threads()
    );
    for (name, gen_) in [
        (
            "spot, 3 rings x 6 arms (half pupil)",
            DefaultMeritSpec::default(),
        ),
        (
            "spot, 6 rings x 12 arms (half pupil)",
            DefaultMeritSpec {
                sampling: mklight_optim::merit::PupilSampling {
                    rings: 6,
                    arms: 12,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "wavefront, 3 rings x 6 arms (half pupil)",
            DefaultMeritSpec {
                criterion: mklight_optim::merit::default::DefaultCriterion::Wavefront {
                    reference: Default::default(),
                },
                ..Default::default()
            },
        ),
    ] {
        let p = problem(&gen_);
        let x0 = p.start_point().unwrap();
        let n = x0.len();
        // Rays per candidate: every pupil node of every (field, λ) + chiefs.
        let rays = 9 * (gen_.sampling.nodes().len() + 1);
        println!("-- {name}: {} residuals, {n} variables", p.num_residuals());
        // Warm-up.
        evaluate_one(&p, &x0, &ctx).unwrap();
        let reps = 200;
        rate("single candidate (parallel inside)", reps, rays, || {
            for _ in 0..reps {
                evaluate_one(&p, &x0, &ctx).unwrap();
            }
        });
        let batch_n = 256;
        let mut b = CandidateBatch::with_capacity(n, batch_n);
        for k in 0..batch_n {
            let mut x = x0.clone();
            x[k % n] *= 1.0 + 1e-4 * (k as f64 + 1.0);
            b.push(&x);
        }
        rate("batch of 256 candidates", batch_n, rays, || {
            p.evaluate_batch(&b, &ctx).unwrap();
        });
        let f0 = evaluate_one(&p, &x0, &ctx).unwrap();
        let reps = 20;
        rate(
            "forward FD Jacobian (n + 0 evals each)",
            reps * n,
            rays,
            || {
                for _ in 0..reps {
                    jacobian(&p, &x0, &f0, &JacobianSettings::default(), &ctx).unwrap();
                }
            },
        );
    }
}
