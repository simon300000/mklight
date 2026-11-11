//! The integration tests of `mklight-core` as one test program: every file of
//! this directory is a module of it, so the tests of one file are
//! `cargo test -p mklight-core --test tests <file>::`.

mod app_fixtures;
mod prescription_import;
