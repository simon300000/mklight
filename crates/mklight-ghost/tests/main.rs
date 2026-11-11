//! The integration tests of `mklight-ghost` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-ghost --test tests <file>::`.

mod common;

mod bench;
mod clipping;
mod conservation;
mod p9_bench;
mod p9_gpu;
mod plate;
mod samples;
mod singlet;
mod undefined_ratios;
mod virtual_segments;
