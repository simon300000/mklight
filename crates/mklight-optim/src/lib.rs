//! MKLight optimisation numerics, merit-function data model and tolerancing.
//!
//! Everything here is generic over the [`problem::Problem`] trait, a batched
//! weighted least-squares problem with bounds and explicit constraints.
//! Optical evaluation (ray tracing, wavefronts, MTF…) is plugged in later
//! through [`merit::OperandEvaluator`] and [`merit::OpticalProblem`]; this crate
//! deliberately does not depend on the tracer.
//!
//! | module | contents |
//! |---|---|
//! | [`problem`] | variables, bounds, constraints, batched evaluation, failure status |
//! | [`deriv`] | batched finite-difference Jacobians, analytic hook, derivative check |
//! | [`lsq`] | Levenberg–Marquardt and dogleg trust-region least squares, bounds, augmented Lagrangian |
//! | [`global`] | differential evolution, Nelder–Mead, CMA-ES, basin-hopping "hammer" |
//! | [`discrete`] | finite-choice variables (glass substitution) with pluggable neighbourhoods |
//! | [`merit`] | serialisable merit-function spec, operand registry, pupil quadrature, default merit |
//! | [`tolerance`] | tolerance operands, Monte-Carlo sampling, sensitivity analysis |
//! | [`monitor`] | cancellation/progress, iteration history callback, best-so-far |
//!
//! All randomness goes through `mklight_core::rng::Philox` with explicit
//! seeds, so every run is reproducible bit for bit.

// `!(x > 0.0)` is used deliberately so that NaN fails the test; index loops
// mirror the matrix formulas they implement.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

pub mod deriv;
pub mod discrete;
pub mod global;
pub mod linalg;
pub mod lsq;
pub mod merit;
pub mod monitor;
pub mod problem;
pub mod rand;
pub mod tolerance;

pub use deriv::{FdScheme, JacobianSettings};
pub use lsq::{LsqMethod, LsqSettings, least_squares};
pub use monitor::{BestSoFar, IterationRecord, Monitor, OptimResult, Termination};
pub use problem::{
    BatchEvaluation, CandidateBatch, Constraint, EvalStatus, Evaluation, FnProblem, Problem,
    Variable,
};
