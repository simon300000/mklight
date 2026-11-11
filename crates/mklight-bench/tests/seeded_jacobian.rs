//! Finite-difference Jacobians of the default merit with aims seeded from
//! the anchor point (`mklight_trace::AimSeeds`, the production path) against
//! unseeded ones, both measured against a central-difference reference with a
//! larger step. Ignored by default; prints a table. Run with
//!
//! ```text
//! cargo test --release -p mklight-bench --test tests seeded_jacobian:: -- --ignored --nocapture
//! ```
//!
//! `MKLIGHT_SEED_PROJECT=/path/project.mklight` adds a project lens.
//! `../README.md` lists every harness with its command and variables.

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkResult, OpticalSystem, samples};
use mklight_optim::deriv::{FdScheme, JacobianSettings, fd_jacobian};
use mklight_optim::linalg::Matrix;
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::merit::{BuiltinEvaluator, OperandRegistry, OpticalProblem, VariableSpec};
use mklight_optim::problem::{
    BatchEvaluation, CandidateBatch, Constraint, Problem, Variable, evaluate_one,
};

/// The problem with the batch anchor removed (unseeded evaluation).
struct Unseeded<'a, P>(&'a P);

impl<P: Problem> Problem for Unseeded<'_, P> {
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_residuals(&self) -> usize {
        self.0.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.0.weights()
    }
    fn constraints(&self) -> &[Constraint] {
        self.0.constraints()
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        let mut x = x.clone();
        x.anchor = None;
        self.0.evaluate_batch(&x, ctx)
    }
    fn fd_relative_step(&self, s: FdScheme) -> Option<f64> {
        self.0.fd_relative_step(s)
    }
}

/// The project-bench solver variables (`mklight_bench::solver::variables`).
fn variables(system: &OpticalSystem) -> Vec<VariableSpec> {
    let vars = mklight_bench::solver::variables(system);
    serde_json::from_value(serde_json::Value::Array(vars)).unwrap()
}

fn column_errors(j: &Matrix, r: &Matrix) -> Vec<f64> {
    (0..r.cols)
        .filter_map(|c| {
            let (mut d, mut n) = (0.0f64, 0.0f64);
            for i in 0..r.rows {
                d += (j[(i, c)] - r[(i, c)]).powi(2);
                n += r[(i, c)].powi(2);
            }
            (n > 0.0).then(|| (d / n).sqrt())
        })
        .collect()
}

fn summary(e: &mut [f64]) -> (f64, f64) {
    e.sort_by(f64::total_cmp);
    if e.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    (e[e.len() / 2], e[e.len() - 1])
}

fn probe(name: &str, system: OpticalSystem, library: Arc<MaterialLibrary>) {
    let vars = variables(&system);
    if vars.is_empty() {
        return;
    }
    let spec = match DefaultMeritSpec::default().generate_for(&system, &vars) {
        Ok(s) => s,
        Err(e) => {
            println!("{name}: no default merit ({e})");
            return;
        }
    };
    let p = match OpticalProblem::new(
        system,
        vars,
        spec,
        BuiltinEvaluator::new(library),
        &OperandRegistry::builtin(),
    ) {
        Ok(p) => p,
        Err(e) => {
            println!("{name}: {e}");
            return;
        }
    };
    let ctx = JobContext::new();
    let x = p.start_point().unwrap();
    let f0 = evaluate_one(&p, &x, &ctx).unwrap();
    let forward = JacobianSettings {
        scheme: FdScheme::Forward,
        ..Default::default()
    };
    let reference = JacobianSettings {
        scheme: FdScheme::Central,
        relative_step: Some(1e-5),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let cold = fd_jacobian(&Unseeded(&p), &x, &f0, &forward, &ctx).unwrap();
    let t_cold = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let seeded = fd_jacobian(&p, &x, &f0, &forward, &ctx).unwrap();
    let t_seeded = t.elapsed().as_secs_f64() * 1e3;
    let r = fd_jacobian(&Unseeded(&p), &x, &f0, &reference, &ctx).unwrap();
    let (cm, cx) = summary(&mut column_errors(&cold.residuals, &r.residuals));
    let (sm, sx) = summary(&mut column_errors(&seeded.residuals, &r.residuals));
    let (dm, dx) = summary(&mut column_errors(&seeded.residuals, &cold.residuals));
    let (ccm, ccx) = summary(&mut column_errors(&cold.constraints, &r.constraints));
    let (scm, scx) = summary(&mut column_errors(&seeded.constraints, &r.constraints));
    println!(
        "SEEDED_JACOBIAN {name:>18} n={:3} m={:4} cold {cm:.2e}/{cx:.2e} seeded {sm:.2e}/{sx:.2e} \
         seeded-vs-cold {dm:.2e}/{dx:.2e} constraints cold {ccm:.2e}/{ccx:.2e} seeded {scm:.2e}/{scx:.2e} \
         ms cold {t_cold:.1} seeded {t_seeded:.1} failed {}/{}",
        x.len(),
        p.num_residuals(),
        cold.failed_columns.len(),
        seeded.failed_columns.len(),
    );
}

#[test]
#[ignore = "prints the seeded and unseeded Jacobian errors and times; asserts nothing"]
fn seeded_jacobian_errors() {
    for (name, system) in samples::all() {
        probe(name, system, Arc::new(MaterialLibrary::with_builtin()));
    }
    if let Ok(path) = std::env::var("MKLIGHT_SEED_PROJECT") {
        let engine = mklight_engine::Engine::new(Default::default());
        let s = engine.open_session(std::path::Path::new(&path)).unwrap();
        probe("project", s.system().clone(), s.library().clone());
    }
}
