//! Standard test problems (Moré, Garbow & Hillstrom 1981 and classics) and
//! the device helpers of the resident tests.
#![allow(dead_code)]

use std::sync::Arc;

use mklight_core::{JobContext, MkResult};
use mklight_gpu::ComputeBackend;
use mklight_optim::gpu::GpuContext;
use mklight_optim::linalg::Matrix;
use mklight_optim::problem::{BatchEvaluation, CandidateBatch, Problem};
use mklight_optim::{Constraint, EvalStatus, FnProblem, Variable};

/// A Metal backend on the host's device, or `None` (with a note on stderr)
/// where there is none: device tests then skip.
pub fn metal() -> Option<Arc<dyn ComputeBackend>> {
    #[cfg(target_os = "macos")]
    let backend = mklight_metal::MetalBackend::new()
        .expect("Metal backend")
        .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>);
    #[cfg(not(target_os = "macos"))]
    let backend: Option<Arc<dyn ComputeBackend>> = None;
    if backend.is_none() {
        eprintln!("SKIP: no Metal device");
    }
    backend
}

/// `problem` with a device context: least squares then factors its tall
/// Jacobians on the device.
pub struct OnDevice<P> {
    pub problem: P,
    pub context: GpuContext,
}
impl<P: Problem> Problem for OnDevice<P> {
    fn gpu_context(&self) -> Option<&GpuContext> {
        Some(&self.context)
    }
    fn variables(&self) -> &[Variable] {
        self.problem.variables()
    }
    fn num_residuals(&self) -> usize {
        self.problem.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.problem.weights()
    }
    fn constraints(&self) -> &[Constraint] {
        self.problem.constraints()
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        self.problem.evaluate_batch(x, ctx)
    }
}

pub fn vars(n: usize) -> Vec<Variable> {
    (0..n).map(|i| Variable::new(format!("x{i}"))).collect()
}

/// Rosenbrock: r = (10(x₂ − x₁²), 1 − x₁). Minimum 0 at (1, 1).
pub fn rosenbrock() -> FnProblem {
    FnProblem::residuals(vars(2), 2, |x, r| {
        r[0] = 10.0 * (x[1] - x[0] * x[0]);
        r[1] = 1.0 - x[0];
    })
    .with_jacobian(|x| {
        (
            Matrix::from_row_major(2, 2, vec![-20.0 * x[0], 10.0, -1.0, 0.0]),
            Matrix::zeros(0, 2),
        )
    })
}

/// Powell singular function. Minimum 0 at the origin, where the Jacobian is
/// singular (so convergence is only linear).
pub fn powell_singular() -> FnProblem {
    FnProblem::residuals(vars(4), 4, |x, r| {
        r[0] = x[0] + 10.0 * x[1];
        r[1] = 5f64.sqrt() * (x[2] - x[3]);
        r[2] = (x[1] - 2.0 * x[2]).powi(2);
        r[3] = 10f64.sqrt() * (x[0] - x[3]).powi(2);
    })
}

/// Freudenstein–Roth. Global minimum 0 at (5, 4); local minimum
/// 48.9842536792… at (11.41…, −0.8968…).
pub fn freudenstein_roth() -> FnProblem {
    FnProblem::residuals(vars(2), 2, |x, r| {
        r[0] = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1];
        r[1] = -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1];
    })
}
pub const FR_LOCAL_MIN: f64 = 48.984_253_679_240_88;

/// Rastrigin as least squares: rᵢ = √(xᵢ² + 10(1 − cos 2πxᵢ)). Minimum 0 at 0.
pub fn rastrigin(n: usize, half_width: f64) -> FnProblem {
    let v = (0..n)
        .map(|i| Variable::bounded(format!("x{i}"), -half_width, half_width))
        .collect();
    FnProblem::residuals(v, n, |x, r| {
        for (ri, xi) in r.iter_mut().zip(x) {
            *ri = (xi * xi + 10.0 * (1.0 - (std::f64::consts::TAU * xi).cos())).sqrt();
        }
    })
}

/// Ackley as a single residual r = √f. Minimum 0 at 0.
pub fn ackley(n: usize, half_width: f64) -> FnProblem {
    let v = (0..n)
        .map(|i| Variable::bounded(format!("x{i}"), -half_width, half_width))
        .collect();
    FnProblem::residuals(v, 1, move |x, r| {
        let nf = x.len() as f64;
        let s2 = x.iter().map(|v| v * v).sum::<f64>() / nf;
        let sc = x
            .iter()
            .map(|v| (std::f64::consts::TAU * v).cos())
            .sum::<f64>()
            / nf;
        let f = -20.0 * (-0.2 * s2.sqrt()).exp() - sc.exp() + 20.0 + std::f64::consts::E;
        r[0] = f.max(0.0).sqrt();
    })
}

/// min (x₁ − 2)² + (x₂ − 1)²  s.t.  x₁ − 2x₂ + 1 = 0,  x₁²/4 + x₂² ≤ 1.
/// KKT point x* = ((√7 − 1)/2, (√7 + 1)/4), both constraints active.
pub fn constrained_kkt() -> FnProblem {
    FnProblem::new(vars(2), 2, |x, r, c| {
        r[0] = x[0] - 2.0;
        r[1] = x[1] - 1.0;
        c[0] = x[0] - 2.0 * x[1] + 1.0;
        c[1] = 0.25 * x[0] * x[0] + x[1] * x[1];
        EvalStatus::Ok
    })
    .with_constraints(vec![
        Constraint::equal("line", 0.0),
        Constraint::at_most("ellipse", 1.0),
    ])
}

pub fn kkt_solution() -> [f64; 2] {
    let s7 = 7f64.sqrt();
    [(s7 - 1.0) / 2.0, (s7 + 1.0) / 4.0]
}
