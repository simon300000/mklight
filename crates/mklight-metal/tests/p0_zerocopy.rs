//! Matched host benchmark and lease-lifetime checks for unified storage.
#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p0_zerocopy.rs"]
mod contracts;
#[path = "../../mklight-gpu/tests/p1_support.rs"]
mod support;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, transforms, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use std::{hint::black_box, time::Instant};

fn backend(origin: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions {
        library: origin,
        ..Default::default()
    })
    .unwrap()
    .expect("host Metal required");
    assert!(b.precision().probe.error_free_transforms());
    eprintln!("DEVICE {:?} BUILD {}", b.info(), b.precision().kernel_build);
    b
}
fn fft(b: &dyn ComputeBackend) -> ResidentPlan {
    let n = 1024;
    let h = b
        .fill_buffer(
            BufferDesc::new(ElementType::Complex32, [n, n]),
            &mut |mut f| {
                for (i, v) in f.as_mut_slice::<[f32; 2]>()?.iter_mut().enumerate() {
                    *v = [((i % 251) as f32 - 125.) / 128., 0.];
                }
                Ok(())
            },
        )
        .unwrap();
    let mut p = ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap());
    let mut prev = p.import(h.view());
    for sign in [-1, 1] {
        let out = p.scratch(BufferDesc::new(ElementType::Complex32, [n, n]));
        p.operations
            .push(ResidentOp::Transforms(transforms::Request::Transform(
                transforms::TransformPlan {
                    kind: transforms::TransformKind::Fft,
                    input: p.view(prev).unwrap(),
                    output: out,
                    dimensions: vec![n, n],
                    output_dimensions: vec![n, n],
                    sign,
                    scale: if sign == 1 { 1. / (n * n) as f32 } else { 1. },
                    affine: [Df32::ZERO; 6],
                },
            )));
        prev = out;
    }
    p.outputs = vec![prev];
    p
}
fn trace(b: &dyn ComputeBackend) -> (ResidentPlan, SystemHandle) {
    let t = testutil::tracer(testutil::singlet());
    let h = b
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let mut rays = RayBatch::with_capacity(1_000_000);
    for i in 0..1_000_000 {
        rays.push_f64(
            [((i % 997) as f64 / 997. - 0.5) * 4., 0., 0.],
            [0., 0., 1.],
            0,
        );
    }
    let mut q = TraceRequest::new(h, rays, 0, t.system().image_index() as u32);
    q.radiometry = Radiometry::Off;
    let mut p = ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap());
    let out = p.trace(q);
    let mut v = p.view(out.floats).unwrap();
    v.desc.shape = vec![3, 1_000_000];
    let sum = p.sum(v, ReduceMode::DoubleFloat);
    p.outputs = vec![sum];
    (p, h)
}
fn opd(b: &dyn ComputeBackend) -> (ResidentPlan, SystemHandle) {
    let t = testutil::tracer(testutil::singlet());
    let points: Vec<_> = (0..1024 * 1024)
        .map(|i| {
            [
                2. * (i % 1024) as f64 / 1023. - 1.,
                2. * (i / 1024) as f64 / 1023. - 1.,
            ]
        })
        .filter(|p| p[0] * p[0] + p[1] * p[1] <= 1.)
        .collect();
    let (mut p, h) = support::make_plan(b, &t, 0, 0, &points, false, 40, true);
    p.outputs = p.outputs[p.outputs.len() - 2..].to_vec();
    (p, h)
}
fn digest(words: &[u32]) -> u64 {
    words.iter().fold(0xcbf29ce484222325, |h, &w| {
        (h ^ w as u64).wrapping_mul(0x100000001b3)
    })
}
#[test]
#[ignore = "matched 1M trace, 1024 pupil and FFT host benchmark"]
fn benchmark_zero_copy() {
    let b = backend(LibraryChoice::Precompiled);
    for name in ["trace_1m", "opd_1024", "fft_1024_roundtrip"] {
        if std::env::var("FUSION_BENCH").is_ok() && name != "trace_1m" {
            continue;
        }
        let setup_copied = b.resident_cache_stats().copied_bytes;
        let setup = Instant::now();
        let (p, h) = match name {
            "trace_1m" => {
                let (p, h) = trace(&b);
                (p, Some(h))
            }
            "opd_1024" => {
                let (p, h) = opd(&b);
                (p, Some(h))
            }
            _ => (fft(&b), None),
        };
        let setup_bytes = b.resident_cache_stats().copied_bytes - setup_copied;
        eprintln!(
            "ZERO_COPY_SETUP {}",
            serde_json::json!({"case":name,"seconds":setup.elapsed().as_secs_f64(),"copied_bytes":setup_bytes})
        );
        for round in 0..11 {
            let wall = Instant::now();
            let before = b.resident_cache_stats().copied_bytes;
            let r = b
                .submit_plan(p.clone(), &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let hashes: Vec<_> = (r.outputs.iter())
                .map(|h| digest(black_box(b.read_view(h.view()).unwrap().words())))
                .collect();
            let expected: &[u64] = match name {
                "trace_1m" => &[16663909210024701490],
                "opd_1024" => &[5738852189248301980, 17415920747824051813],
                _ => &[13232921648690179073],
            };
            assert_eq!(hashes, expected);
            assert_eq!(r.stats.copied_bytes, 0);
            assert_eq!(b.resident_cache_stats().copied_bytes, before);
            eprintln!(
                "ZERO_COPY_RAW {}",
                serde_json::json!({"case":name,"round":round,"warmup":round==0,"wall_s":wall.elapsed().as_secs_f64(),"hashes":hashes,"stats":r.stats})
            );
        }
        if let Some(h) = h {
            b.release_systems(h).unwrap();
        }
    }
}

#[test]
fn host_views_both_origins() {
    for origin in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = backend(origin);
        contracts::suite(&b);
        let (opd_plan, opd_system) = opd(&b);
        for (plan, expected) in [
            // The singlet's rear surface is an even asphere: its OPD bits
            // follow the trace-slope design (`trace_slope`).
            (
                opd_plan,
                if mklight_gpu::layout::df_slope().unwrap() {
                    vec![7341078886716340591, 17415920747824051813]
                } else {
                    vec![5738852189248301980, 17415920747824051813]
                },
            ),
            (fft(&b), vec![13232921648690179073]),
        ] {
            let r = b
                .submit_plan(plan, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(r.stats.copied_bytes, 0);
            let hashes: Vec<_> = r
                .outputs
                .iter()
                .map(|h| digest(b.read_view(h.view()).unwrap().words()))
                .collect();
            assert_eq!(hashes, expected);
        }
        b.release_systems(opd_system).unwrap();
        let (p, h) = trace(&b);
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let g = b.read_view(r.outputs[0].view()).unwrap();
        // The traced singlet's rear surface is an even asphere (`trace_slope`).
        let traced = if mklight_gpu::layout::df_slope().unwrap() {
            10683428124538957362
        } else {
            16663909210024701490
        };
        assert_eq!(digest(g.words()), traced);
        assert_eq!(r.stats.copied_bytes, 0);
        b.release_systems(h).unwrap();
        drop(r);
        drop(b);
        assert_eq!(digest(g.words()), traced);
    }
}
