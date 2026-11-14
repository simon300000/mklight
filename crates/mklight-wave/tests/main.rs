//! The integration tests of `mklight-wave` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-wave --test tests <file>::`.

mod afocal;
mod analytic;
mod defaults;
mod interactive;
mod p6_bench;
mod p6_boundaries;
mod p6_native;
mod p6_offaxis;
mod p6_wave;
mod regressions;
mod sensor_mtf;
mod split_designs;
mod traced;
