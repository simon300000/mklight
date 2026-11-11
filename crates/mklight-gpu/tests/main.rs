//! The integration tests of `mklight-gpu` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-gpu --test tests <file>::`.

// Helpers of several files here and of the Metal crate's tests.
mod p1_support;

mod c_layout;
mod p0_contracts;
mod p0_zerocopy;
mod p10_cases;
mod p1_aiming;
mod p2_transforms;
mod p3_collectives;
mod p5_layout;
mod p7_radiometry;
mod p9_ops;
mod trace;
mod trace_slope;
