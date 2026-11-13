#![cfg(target_os = "macos")]
use crate::p7_support as support;
use mklight_gpu::ComputeBackend;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
#[test]
fn coatings_both_libraries() {
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions { library })
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
        let b = MetalBackend::with_options(&MetalOptions { library })
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
        .map(|c| {
            CoatingSpec::resolve(
                c,
                0.55,
                1.,
                1.52,
                false,
                mklight_core::materials::coating::Arrival::Before,
                &mut ls,
            )
            .unwrap()
        })
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
    // Bare, ideal, tabulated and thin-film algebra (including the reduced
    // sin/cos/exp polynomials) is specified double-float: identical bits.
    for (i, (a, c)) in chunk.iter().zip(emulated).enumerate() {
        assert_eq!(bytemuck::bytes_of(a), bytemuck::bytes_of(&c), "query {i}");
    }
    let t = testutil::tracer(samples::cooke_triplet());
    let points = (mklight_trace::PupilSampling::Grid { n: 65 }).points();
    let (small, hs) = support::transport_plan(&b, &t, 2, 1, &points, 64);
    let (large, hl) = support::transport_plan(&b, &t, 2, 1, &points, 4096);
    let pa = b.submit_plan(small.clone(), &JobContext::new()).unwrap();
    let pb = b.submit_plan(large, &JobContext::new()).unwrap();
    let a = pa.wait().unwrap();
    let z = pb.wait().unwrap();
    assert_eq!(
        b.read_view(a.outputs[0].view()).unwrap().words(),
        b.read_view(z.outputs[0].view()).unwrap().words()
    );
    let repeated = b
        .submit_plan(small, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        b.read_view(a.outputs[0].view()).unwrap().words(),
        b.read_view(repeated.outputs[0].view()).unwrap().words()
    );
    b.release_systems(hs).unwrap();
    b.release_systems(hl).unwrap();
    // Enough small commands to be at work when the cancel arrives, and
    // bounds that hold on a busy machine.
    let huge: Vec<_> = (0..262144)
        .map(|i| CoatingQuery::new(4, 0.1 + 0.9 * i as f64 / 262144.).unwrap())
        .collect();
    let plan = support::plan(&b, &specs, &ls, &huge, 64);
    let ctx = JobContext::new();
    let pending = b.submit_plan(plan, &ctx).unwrap();
    let start = Instant::now();
    let mut previous = 0.;
    while ctx.progress.fraction() == 0. {
        assert!(!pending.is_ready());
        assert!(start.elapsed() < Duration::from_secs(60));
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
            arrival: mklight_core::materials::coating::Arrival::Before,
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
#[test]
fn geometric_and_dim_fixtures() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::geometric_and_dim_fixtures(&b);
}
#[test]
fn singular_retarder() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::singular_retarder(&b);
}
#[test]
fn scalar_reference() {
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    support::scalar_reference(&b);
}
#[test]
fn transport_matches_cpu_emulation() {
    use mklight_core::{JobContext, samples};
    use mklight_gpu::{radiometry::*, *};
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 requires host Metal");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let mut model = samples::cooke_triplet();
    model.coatings.insert(
        "film".into(),
        mklight_core::CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![mklight_core::Layer {
                material: mklight_core::MaterialRef::Constant { n: 1.38 },
                extinction: 0.02,
                thickness_um: 0.07,
                quarter_waves: None,
            }],
            mirror_substrate: None,
            substrate: Default::default(),
        },
    );
    for s in &mut model.surfaces[1..] {
        s.coating = Some("film".into());
    }
    let points = (mklight_trace::PupilSampling::Grid { n: 17 }).points();
    for (name, model) in [("cooke", samples::cooke_triplet()), ("film", model)] {
        let t = testutil::tracer(model);
        for scalar in [false, true] {
            let run = |b: &dyn QualificationBackend| {
                let (mut p, h) = support::transport_plan(b, &t, 2, 1, &points, 256);
                for op in &mut p.operations {
                    if let resident::ResidentOp::Radiometry(Request::Polarization(r)) = op {
                        r.scalar = scalar;
                    }
                }
                let r = b
                    .submit_plan(p, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                b.release_systems(h).unwrap();
                let v = b.read_view(r.outputs[0].view()).unwrap();
                bytemuck::cast_slice::<u32, PolarizationValue>(v.words()).to_vec()
            };
            for (i, (mut m, mut c)) in run(&b).into_iter().zip(run(&cpu)).enumerate() {
                // Everything but the retardance angles (an f32 library atan2)
                // is the specified double-float algebra: identical bits.
                for k in [1, 8] {
                    let (a, z) = (m.properties[k].to_f64(), c.properties[k].to_f64());
                    assert!((a - z).abs() <= 1e-6 * z.abs(), "{name} {i} angle {a} {z}");
                    m.properties[k] = Df32::ZERO;
                    c.properties[k] = Df32::ZERO;
                }
                assert_eq!(
                    bytemuck::bytes_of(&m),
                    bytemuck::bytes_of(&c),
                    "{name} scalar={scalar} row {i}"
                );
            }
        }
    }
}
