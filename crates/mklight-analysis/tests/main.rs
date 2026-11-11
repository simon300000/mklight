//! The integration tests of `mklight-analysis` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-analysis --test tests <file>::`.

mod common;

mod aim_designs;
mod bench;
mod curvature;
mod extended;
mod fieldmaps_split;
mod golden;
mod illumination;
mod layout;
mod p5_bench;
mod p5_geometry;
mod seidel;
mod spot_energy;
mod system_data;
