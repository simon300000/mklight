#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p9_ops.rs"]
mod support;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
#[test]
fn fixed_hits_both_libraries() {
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions { library })
            .unwrap()
            .expect("P9 requires host Metal");
        support::fixed_hits(&b);
        let actual = support::first_order(&b);
        let cpu =
            mklight_gpu::CpuF32Backend::matching(&mklight_gpu::ComputeBackend::precision(&b).probe);
        assert_eq!(actual, support::first_order(&cpu));
    }
}
