#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p3_cases.rs"]
mod p3_cases;
use mklight_gpu::{resident::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
fn backend(origin: LibraryChoice) -> MetalBackend {
    let started = std::time::Instant::now();
    let b = MetalBackend::with_options(&MetalOptions {
        library: origin,
        ..Default::default()
    })
    .unwrap()
    .expect("P3 requires actual host Metal");
    eprintln!(
        "backend_setup_probe_ms={:.6}",
        started.elapsed().as_secs_f64() * 1e3
    );
    eprintln!(
        "device={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    assert!(b.precision().probe.error_free_transforms());
    b
}
#[test]
fn actual_metal_known_answers_both_libraries() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = backend(origin);
        p3_cases::scan_answers(&b);
        p3_cases::sort_answers(&b);
        p3_cases::compact_answers(&b);
        p3_cases::segment_answers(&b);
        p3_cases::qr_answers(&b);
        p3_cases::energy_chain(&b);
        p3_cases::conditioned_qr(&b);
        p3_cases::nonfinite_qr(&b);
    }
}
#[test]
fn bitwise_fixed_collectives() {
    let g = backend(LibraryChoice::Precompiled);
    let c = CpuF32Backend::matching(&g.precision().probe);
    for n in [0, 257, 10003] {
        for t in [ElementType::U32, ElementType::F32, ElementType::DoubleFloat] {
            let (cpu, _) = p3_cases::read(&c, p3_cases::sort_case(&c, n, t));
            let (gpu, _) = p3_cases::read(&g, p3_cases::sort_case(&g, n, t));
            for (a, b) in cpu.iter().zip(&gpu) {
                p3_cases::assert_bits(a, b);
            }
        }
        for integer in [true, false] {
            for inclusive in [true, false] {
                let (a, _) = p3_cases::read(&c, p3_cases::scan_case(&c, n, integer, inclusive));
                let (b, _) = p3_cases::read(&g, p3_cases::scan_case(&g, n, integer, inclusive));
                p3_cases::assert_bits(&a[0], &b[0]);
            }
        }
    }
}
