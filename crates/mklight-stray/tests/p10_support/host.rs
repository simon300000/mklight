//! Actual-Metal geometry comparisons for the shared P10 cases.
#![cfg(target_os = "macos")]
#[path = "../p10_geometry.rs"]
mod cases;
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
#[test]
fn host_reference_comparisons() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions { library: origin })
            .unwrap()
            .expect("actual Metal required");
        eprintln!(
            "P10 {} {} {:?}",
            b.info().backend_id(),
            b.precision().kernel_build,
            b.precision().probe
        );
        assert!(b.precision().probe.error_free_transforms());
        cases::sources(&b);
        cases::geometry(&b);
        cases::tangents(&b);
        cases::samples(&b);
        cases::large_scene(&b);
    }
}
