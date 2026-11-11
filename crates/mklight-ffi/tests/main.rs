//! The integration tests of `mklight-ffi` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-ffi --test tests <file>::`.

mod abi_1_1;
mod abi_1_3;
mod c_smoke;
mod callback_reentrancy;
mod errors_and_keys;
mod shared_buffers;
mod sync_unlocked;
