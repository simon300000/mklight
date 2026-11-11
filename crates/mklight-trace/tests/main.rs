//! The integration tests of `mklight-trace` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-trace --test tests <file>::`.

mod common;

mod aim_stop;
mod analytic;
mod extended_surfaces;
mod generate_from;
mod geometry;
mod model_boundary;
mod packets;
mod polarization;
mod prescription_round_trip;
mod samples_and_batch;
mod seeds;
mod solves;
mod telecentric_opd;
