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
/// Matching-environment CPU specification and Metal agree bit for bit,
/// across both segment paths, every radix digit and QR group widths.
#[test]
fn bitwise_fixed_collectives() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        bitwise(&backend(origin));
    }
}
fn bitwise(g: &MetalBackend) {
    let c = CpuF32Backend::matching(&g.precision().probe);
    let same = |build: &dyn Fn(&dyn ComputeBackend) -> ResidentPlan| {
        let (a, _) = p3_cases::read(&c, build(&c));
        let (b, _) = p3_cases::read(g, build(g));
        for (a, b) in a.iter().zip(&b) {
            p3_cases::assert_bits(a, b);
        }
    };
    for n in [0, 257, 10003] {
        for t in [ElementType::U32, ElementType::F32, ElementType::DoubleFloat] {
            same(&|b| p3_cases::sort_case(b, n, t));
        }
        for integer in [true, false] {
            for inclusive in [true, false] {
                same(&|b| p3_cases::scan_case(b, n, integer, inclusive));
            }
        }
    }
    for t in [ElementType::U32, ElementType::F32, ElementType::DoubleFloat] {
        same(&|b| p3_cases::random_sort_case(b, 70001, t));
    }
    // Short segments (SIMD path) and the same records under a long bound
    // (tree-group path); one segment spans several tree groups.
    let short: Vec<u32> = (0..=600)
        .map(|s| s * 61 / 2)
        .chain([18_300, 30_001])
        .collect();
    same(&|b| p3_cases::moments_case(b, short[..601].to_vec(), 61));
    same(&|b| p3_cases::moments_case(b, short[..601].to_vec(), 18_300));
    same(&|b| p3_cases::moments_case(b, short.clone(), 30_001));
    for (m, n, batch) in [(41, 3, 1), (9, 3, 16), (9000, 5, 2)] {
        let a: Vec<f64> = (0..batch * m * n)
            .map(|i| ((i * 7919 % 1013) as f64 / 1013.).powi(1 + (i % n) as i32))
            .collect();
        let rhs: Vec<f64> = (0..batch * m).map(|i| (i as f64 * 0.37).sin()).collect();
        same(&|b| p3_cases::qr_plan(b, m, n, batch, a.clone(), rhs.clone(), 1e-10));
    }
}
