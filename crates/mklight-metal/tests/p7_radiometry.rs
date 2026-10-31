#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p7_support.rs"]
mod support;
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
#[test]
fn coatings_both_libraries() {
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P7 requires host Metal");
        assert!(b.precision().probe.error_free_transforms());
        eprintln!("device={:?} precision={:?}", b.info(), b.precision());
        support::check_coatings(&b);
    }
}
#[test]
fn samples_both_libraries() {
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("P7 requires host Metal");
        support::check_samples(&b);
    }
}
#[test]
fn coated_and_fallback_jobs() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::fixtures(&b);
}
#[test]
fn analytic_coating_answers() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::known_answers(&b);
}

#[test]
fn determinism_concurrency_and_cancellation() {
    use mklight_core::{JobContext, MkError, Snapshot, samples};
    use mklight_gpu::{radiometry::*, *};
    use std::time::{Duration, Instant};
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let mut ls = vec![];
    let specs: Vec<_> = support::coatings()
        .iter()
        .map(|c| CoatingSpec::resolve(c, 0.55, 1., 1.52, false, &mut ls).unwrap())
        .collect();
    let q: Vec<_> = (0..257)
        .map(|i| CoatingQuery::new((i % specs.len()) as u32, 0.1 + 0.9 * i as f64 / 257.).unwrap())
        .collect();
    let p = support::plan(&b, &specs, &ls, &q, 64);
    let a = b.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let concurrent = b.submit_plan(p, &JobContext::new()).unwrap();
    let a = a.wait().unwrap();
    let c = concurrent.wait().unwrap();
    let ar = b.read_view(a.outputs[0].view()).unwrap();
    let cr = b.read_view(c.outputs[0].view()).unwrap();
    assert_eq!(ar.words(), cr.words());
    let (chunk, _) = support::run(&b, support::plan(&b, &specs, &ls, &q, 256));
    assert_eq!(ar.words(), bytemuck::cast_slice::<_, u32>(&chunk));
    let (emulated, _) = support::run(&cpu, support::plan(&cpu, &specs, &ls, &q, 64));
    for (i, (a, c)) in chunk.iter().zip(emulated).enumerate() {
        assert_eq!(a.status, c.status);
        if q[i].coating < 3 {
            assert_eq!(
                bytemuck::bytes_of(a),
                bytemuck::bytes_of(&c),
                "specified bare/ideal/table algebra {i}"
            );
        } else {
            for (x, y) in a
                .amplitudes
                .iter()
                .chain(&a.powers)
                .zip(c.amplitudes.iter().chain(&c.powers))
            {
                assert!((x.to_f64() - y.to_f64()).abs() < 1e-10);
            }
        }
    }
    let huge: Vec<_> = (0..65536)
        .map(|i| CoatingQuery::new(4, 0.1 + 0.9 * i as f64 / 65536.).unwrap())
        .collect();
    let plan = support::plan(&b, &specs, &ls, &huge, 64);
    let ctx = JobContext::new();
    let pending = b.submit_plan(plan, &ctx).unwrap();
    let start = Instant::now();
    let mut previous = 0.;
    while ctx.progress.fraction() == 0. {
        assert!(!pending.is_ready());
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::yield_now();
    }
    for _ in 0..20 {
        let p = ctx.progress.fraction();
        assert!(p >= previous);
        previous = p;
        std::thread::yield_now();
    }
    let start = Instant::now();
    ctx.cancel.cancel();
    assert!(matches!(pending.wait(), Err(MkError::Cancelled)));
    assert!(ctx.progress.fraction() < 1.);
    eprintln!(
        "P7 cancellation latency_ms={}",
        start.elapsed().as_secs_f64() * 1e3
    );
    assert!(start.elapsed() < Duration::from_millis(50));
    let prepared = PreparedCoatings::new(
        &b,
        &[CoatingDomain {
            coating: support::coatings()[3].clone(),
            wavelength: 0.55,
            n_in: 1.,
            n_out: 1.69,
            mirror: false,
        }],
        &JobContext::new(),
    )
    .unwrap();
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        prepared.evaluate(
            &b,
            Snapshot::new(samples::singlet()).unwrap(),
            &[CoatingQuery::new(0, 1e-7).unwrap()],
            &ctx
        ),
        Err(MkError::Cancelled)
    ));
}
#[test]
fn malus_and_vector_channels() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::malus_and_channels(&b);
}
