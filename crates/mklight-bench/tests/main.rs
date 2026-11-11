//! The integration tests of `mklight-bench` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-bench --test tests <file>::`.

mod aim_precision_study;
mod aim_stop_qualify;
mod buffer_identity;
mod contract;
mod cost_calibration;
mod cpu_trace_bench;
mod metal_accuracy;
mod project_bench;
mod seeded_jacobian;
mod solver_aiming;
