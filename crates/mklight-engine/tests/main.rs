//! The integration tests of `mklight-engine` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-engine --test tests <file>::`.

mod analyses;
mod gpu;
mod graph;
mod material_offsets;
mod p11_routes;
mod policies;
mod results_doc;
mod route_options;
mod session_ops;
mod stray_partition;
mod wave;
