//! The integration tests of `mklight-optim` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-optim --test tests <file>::`.

mod common;

mod discrete;
mod glass;
mod global;
mod host_device_division;
mod least_squares;
mod merit;
mod mtf_ghost_stray_operands;
mod resident_algebra;
mod resident_bench;
mod resident_candidates;
mod resident_solvers;
mod resident_tolerance;
mod resident_workloads;
mod robustness;
mod scheduler;
mod surfaces;
mod tolerance;
mod trace_bench;
mod trace_eval;
mod trace_optim;
mod trace_regressions;
mod with_solves;
