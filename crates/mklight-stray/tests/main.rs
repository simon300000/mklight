//! The integration tests of `mklight-stray` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-stray --test tests <file>::`.

mod common;

mod all_signatures;
mod analysis;
mod apertures;
mod bench;
mod doe_visits;
mod enclosure;
mod estimators;
mod imaging;
#[path = "p10_support/bench.rs"]
mod p10_bench;
#[path = "p10_support/device_variants.rs"]
mod p10_device_variants;
mod p10_geometry;
#[path = "p10_support/host.rs"]
mod p10_metal_geometry;
#[path = "p10_support/transport.rs"]
mod p10_transport;
#[path = "p10_support/transport_bench.rs"]
mod p10_transport_bench;
mod partition;
mod sequential;
mod system;
mod window;
