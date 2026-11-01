#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p10_cases.rs"]
mod cases;
use mklight_gpu::*;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
fn backend(origin: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions {
        library: origin,
        ..Default::default()
    })
    .unwrap()
    .expect("P10 requires actual host Metal");
    assert!(b.precision().probe.error_free_transforms());
    eprintln!(
        "P10 device={} build={} probe={:?}",
        b.info().backend_id(),
        b.precision().kernel_build,
        b.precision().probe
    );
    b
}
#[test]
fn device_sources_and_bvh() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = backend(origin);
        cases::answers(&b);
        cases::split_child_estimator(&b);
        cases::close_hits(&b);
        let cpu = CpuF32Backend::matching(&b.precision().probe);
        for kind in 0..4 {
            let (reference, _) = cases::run(&cpu, cases::plan(&cpu, kind, 4099, 65536));
            for tile in [127, 65536] {
                let (actual, _) = cases::run(&b, cases::plan(&b, kind, 4099, tile));
                for (k, (a, e)) in actual.iter().zip(&reference).enumerate() {
                    for (j, (a, e)) in a.iter().zip(e).enumerate() {
                        assert_eq!(a, e, "kind={kind} tile={tile} output={k} word={j}");
                    }
                }
            }
        }
    }
}
#[test]
fn concurrent_and_cancelled() {
    use mklight_core::{JobContext, MkError};
    use std::sync::Arc;
    let b = Arc::new(backend(LibraryChoice::Precompiled));
    let a = {
        let b = b.clone();
        std::thread::spawn(move || cases::run(&*b, cases::plan(&*b, 0, 65537, 2048)).0)
    };
    let expected = cases::run(&*b, cases::plan(&*b, 0, 65537, 65536)).0;
    assert_eq!(a.join().unwrap(), expected);
    let ctx = JobContext::new();
    let p = cases::plan(&*b, 0, 1_000_000, 2048);
    let pending = b.submit_plan(p, &ctx).unwrap();
    let mut values = Vec::new();
    while !pending.is_ready() {
        let f = ctx.progress.fraction();
        values.push(f);
        if f > 0. {
            ctx.cancel.cancel();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(matches!(pending.wait(), Err(MkError::Cancelled)));
    assert!(values.iter().any(|v| *v > 0.));
    assert!(values.windows(2).all(|v| v[0] <= v[1]));
    cases::answers(&*b);
}
