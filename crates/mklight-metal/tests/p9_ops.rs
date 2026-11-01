#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p9_ops.rs"]
mod support;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
#[test]
fn fixed_hits_both_libraries() {
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P9 requires host Metal");
        support::fixed_hits(&b);
    }
}
