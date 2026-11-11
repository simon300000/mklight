//! The integration tests of `mklight-metal` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-metal --test tests <file>::`.

// Helpers shared with the tests of `mklight-gpu`, loaded once.
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod p1_support;
#[path = "../../mklight-gpu/tests/p3_cases.rs"]
mod p3_cases;
#[path = "../../mklight-gpu/tests/p7_support.rs"]
mod p7_support;

mod aim_cancel;
mod aim_resume;
mod aim_reuse;
mod kernel_occupancy;
mod metal;
mod p0_resident;
mod p0_zerocopy;
mod p10_stray;
mod p1_aiming;
mod p1_bench;
mod p2_transforms;
mod p3_collectives;
mod p3_resident;
mod p7_bench;
mod p7_radiometry;
mod p9_ops;
mod trace;
