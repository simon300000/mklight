//! Metal backend validation on the real GPU, against the CPU references:
//! `mklight_core` (f64 / exact integer reference) and `mklight_gpu`'s CPU-f32
//! emulation (bit-identical f32 algorithms). Every test skips with a message
//! when no Metal device is available.
#![cfg(target_os = "macos")]

use std::sync::OnceLock;

use mklight_core::math::KahanSum;
use mklight_core::rng::{Philox, philox4x32_10};
use mklight_core::{JobContext, MkError};
use mklight_gpu::cpu::run_output;
use mklight_gpu::df::{two_prod, two_sum};
use mklight_gpu::fmath::u32_to_unit_f32;
use mklight_gpu::layout::{header_offsets, surface_offsets};
use mklight_gpu::{
    ComputeBackend, CpuF32Backend, Df32, DfOp, Job, JobKind, JobOutput, PackedHeader,
    PackedSurface, PackedSystem, ReduceMode, TraceRequest, testutil,
};
use mklight_metal::{LibraryChoice, LibraryOrigin, MetalBackend, MetalOptions};

fn gpu() -> Option<&'static MetalBackend> {
    static B: OnceLock<Option<MetalBackend>> = OnceLock::new();
    let b = B
        .get_or_init(|| MetalBackend::new().expect("Metal backend initialisation failed"))
        .as_ref();
    if b.is_none() {
        eprintln!("SKIP: no Metal device");
    }
    b
}

macro_rules! gpu_or_skip {
    () => {
        match gpu() {
            Some(b) => b,
            None => return,
        }
    };
}

fn rnd(r: &mut Philox, emin: i32, emax: i32) -> f32 {
    let m = r.uniform() * 2.0 - 1.0;
    let e = emin + ((emax - emin) as f64 * r.uniform()) as i32;
    (m * 2f64.powi(e)) as f32
}

fn rnd_df(r: &mut Philox, emin: i32, emax: i32) -> Df32 {
    let x = rnd(r, emin, emax) as f64;
    Df32::from_f64(x * (1.0 + (r.uniform() - 0.5) * 2f64.powi(-23)))
}

fn bits(v: &[Df32]) -> Vec<(u32, u32)> {
    v.iter().map(|d| (d.hi.to_bits(), d.lo.to_bits())).collect()
}

#[test]
fn device_info_and_numerics_probe() {
    let b = gpu_or_skip!();
    let info = b.info();
    let p = b.precision();
    eprintln!("device: {info:#?}");
    eprintln!("probe: {:#?}", p.probe);
    eprintln!("library: {:?}", b.context().library_origin());
    assert!(info.max_buffer_bytes > 0 && info.simd_width > 0);
    assert!(!info.native_fp64);
    // Error-free transforms need IEEE add/mul, a true fma, and no
    // contraction or reassociation by the compiler: i.e. fast-math is off.
    assert!(p.probe.ieee_add_mul, "{:?}", p.probe);
    assert!(p.probe.true_fma, "{:?}", p.probe);
    assert!(
        p.probe.no_contraction,
        "FP contraction is on: {:?}",
        p.probe
    );
    assert!(
        p.probe.no_reassociation,
        "fast-math reassociation: {:?}",
        p.probe
    );
    assert!(p.fast_math_disabled);
    assert!(b.supports(JobKind::DfOps) && b.supports(JobKind::Reduce));
    assert!(!b.supports(JobKind::Trace));
    assert!(info.backend_id().starts_with("metal:"));
}

#[test]
fn layout_matches_metal_compiler() {
    let b = gpu_or_skip!();
    let got = b.layout_probe().unwrap();
    let mut want = vec![
        size_of::<PackedHeader>() as u32,
        size_of::<PackedSurface>() as u32,
        align_of::<PackedHeader>() as u32,
        align_of::<PackedSurface>() as u32,
    ];
    want.extend(header_offsets().iter().map(|(_, o)| *o as u32));
    want.extend(surface_offsets().iter().map(|(_, o)| *o as u32));
    assert_eq!(got, want);
}

#[test]
fn philox_blocks_match_core() {
    let b = gpu_or_skip!();
    let mut r = Philox::new(2024, 3);
    let mut counters = vec![
        [0, 0, 0, 0],
        [u32::MAX; 4],
        [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
    ];
    let mut keys = vec![[0, 0], [u32::MAX; 2], [0xa4093822, 0x299f31d0]];
    for _ in 0..(1 << 16) {
        counters.push([r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()]);
        keys.push([r.next_u32(), r.next_u32()]);
    }
    let out = run_output(
        b,
        Job::PhiloxBlocks {
            counters: counters.clone(),
            keys: keys.clone(),
        },
    )
    .unwrap();
    let JobOutput::Philox { raw, uniform } = out else {
        panic!("unexpected output")
    };
    // Random123 known answers.
    assert_eq!(raw[0..4], [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
    assert_eq!(raw[4..8], [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
    assert_eq!(raw[8..12], [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]);
    for (i, (c, k)) in counters.iter().zip(&keys).enumerate() {
        assert_eq!(raw[4 * i..4 * i + 4], philox4x32_10(*c, *k), "block {i}");
    }
    for (w, u) in raw.iter().zip(&uniform) {
        assert_eq!(u.to_bits(), u32_to_unit_f32(*w).to_bits());
        assert!(*u > 0.0 && *u < 1.0);
    }
}

#[test]
fn philox_stream_matches_core() {
    let b = gpu_or_skip!();
    // Sequential stream from draw 0, against the core's stateful generator.
    let seed = 0xDEAD_BEEF_0123_4567u64;
    let out = run_output(
        b,
        Job::PhiloxStream {
            seed,
            stream: 5,
            first_block: 0,
            num_blocks: 1 << 18,
        },
    )
    .unwrap();
    let JobOutput::Philox { raw, uniform } = out else {
        panic!()
    };
    let mut r = Philox::new(seed, 5);
    for (i, w) in raw.iter().enumerate() {
        assert_eq!(*w, r.next_u32(), "draw {i}");
    }
    assert_eq!(uniform[17], u32_to_unit_f32(raw[17]));

    // 64-bit block counter carry and a 64-bit stream id.
    let first = 0xFFFF_FFF0u64;
    let stream = 0x1234_5678_9ABC_DEF0u64;
    let JobOutput::Philox { raw, .. } = run_output(
        b,
        Job::PhiloxStream {
            seed: 7,
            stream,
            first_block: first,
            num_blocks: 64,
        },
    )
    .unwrap() else {
        panic!()
    };
    for t in 0..64u64 {
        let blk = first + t;
        let ctr = [
            blk as u32,
            (blk >> 32) as u32,
            stream as u32,
            (stream >> 32) as u32,
        ];
        assert_eq!(
            raw[4 * t as usize..4 * t as usize + 4],
            philox4x32_10(ctr, [7, 0])
        );
    }
}

/// The core claim of the double-float design: two_sum and two_prod are
/// *exact* on the GPU (fast-math would make the error terms zero).
#[test]
fn error_free_transforms_survive_on_gpu() {
    let b = gpu_or_skip!();
    let mut r = Philox::new(11, 0);
    let n = 1 << 16;
    let a: Vec<Df32> = (0..n)
        .map(|_| Df32::from_f32(rnd(&mut r, -30, 30)))
        .collect();
    let c: Vec<Df32> = (0..n)
        .map(|_| Df32::from_f32(rnd(&mut r, -30, 30)))
        .collect();
    for op in [DfOp::TwoSum, DfOp::TwoProd] {
        let JobOutput::Df(out) = run_output(
            b,
            Job::DfOps {
                op,
                a: a.clone(),
                b: c.clone(),
            },
        )
        .unwrap() else {
            panic!()
        };
        let mut nonzero_err = 0;
        for i in 0..n {
            let (x, y) = (a[i].hi as f64, c[i].hi as f64);
            let exact = if op == DfOp::TwoSum { x + y } else { x * y };
            assert_eq!(out[i].hi as f64 + out[i].lo as f64, exact, "{op:?} {x} {y}");
            nonzero_err += (out[i].lo != 0.0) as usize;
        }
        // Most random pairs have a non-zero rounding error; fast-math would
        // report zero for all of them.
        assert!(
            nonzero_err > n / 2,
            "{op:?}: only {nonzero_err} non-zero errors"
        );
    }
    // Also directly against the CPU primitives.
    let (s, t) = two_sum(1.0, 1e-8);
    assert_eq!((s, t), (1.0, 1e-8));
    let (p, e) = two_prod(1.0 + f32::EPSILON, 1.0 - f32::EPSILON);
    assert!(
        e != 0.0
            && p as f64 + e as f64 == (1.0 + f32::EPSILON as f64) * (1.0 - f32::EPSILON as f64)
    );
}

/// GPU double-float operations are bit-identical to the CPU-f32 emulation,
/// and accurate against f64.
#[test]
fn df_ops_match_emulation_and_f64() {
    let b = gpu_or_skip!();
    let cpu = CpuF32Backend::new();
    let probe = b.precision().probe;
    let mut r = Philox::new(12, 0);
    let n = 1 << 15;
    let a: Vec<Df32> = (0..n).map(|_| rnd_df(&mut r, -20, 20)).collect();
    let c: Vec<Df32> = (0..n).map(|_| rnd_df(&mut r, -20, 20)).collect();
    let u2 = 2f64.powi(-48);
    for op in DfOp::ALL {
        let (a, c) = if op == DfOp::Sqrt {
            let pos = a
                .iter()
                .map(|d| Df32::new(d.hi.abs(), d.lo * d.hi.signum()))
                .collect();
            (pos, c.clone())
        } else if op == DfOp::FastTwoSum {
            // Precondition |a| >= |b|.
            let (big, small) = a
                .iter()
                .zip(&c)
                .map(|(x, y)| {
                    if x.hi.abs() >= y.hi.abs() {
                        (*x, *y)
                    } else {
                        (*y, *x)
                    }
                })
                .unzip();
            (big, small)
        } else {
            (a.clone(), c.clone())
        };
        let job = || Job::DfOps {
            op,
            a: a.clone(),
            b: c.clone(),
        };
        let JobOutput::Df(g) = run_output(b, job()).unwrap() else {
            panic!()
        };
        let JobOutput::Df(e) = run_output(&cpu, job()).unwrap() else {
            panic!()
        };
        let need_cr = op.needs_correct_div_sqrt();
        if !need_cr || (probe.correctly_rounded_div && probe.correctly_rounded_sqrt) {
            assert_eq!(
                bits(&g),
                bits(&e),
                "{op:?}: GPU differs from CPU-f32 emulation"
            );
        }
        // Accuracy against f64 (bounds from df.rs, with margin for the
        // f32 first approximation when div/sqrt are not correctly rounded).
        for i in 0..n {
            let (x, y) = (a[i].to_f64(), c[i].to_f64());
            let (exact, bound) = match op {
                DfOp::Add => (x + y, 3.0 * u2 * (x.abs() + y.abs())),
                DfOp::Mul => (x * y, 4.0 * u2 * (x * y).abs()),
                DfOp::Div => (x / y, 16.0 * u2 * (x / y).abs()),
                DfOp::Sqrt => (x.sqrt(), 4.0 * u2 * x.sqrt()),
                DfOp::AddF32 => (
                    x + c[i].hi as f64,
                    2.0 * u2 * (x.abs() + c[i].hi.abs() as f64),
                ),
                DfOp::MulF32 => (x * c[i].hi as f64, 2.0 * u2 * (x * c[i].hi as f64).abs()),
                _ => continue,
            };
            let err = (g[i].to_f64() - exact).abs();
            assert!(err <= bound, "{op:?} #{i}: err {err:e} > {bound:e}");
        }
    }
}

/// Summing 10⁶ values: double-float on the GPU matches the f64 sum to
/// ~1e-14 relative; plain f32 does not. All modes are bit-identical to the
/// CPU-f32 emulation (deterministic order).
#[test]
fn compensated_reduction_1e6() {
    let b = gpu_or_skip!();
    let cpu = CpuF32Backend::new();
    let mut r = Philox::new(13, 0);
    let n = 1_000_000;
    let x: Vec<f32> = (0..n).map(|_| r.uniform() as f32).collect();
    let exact = x.iter().map(|&v| v as f64).collect::<KahanSum>().value();
    let rel = |d: Df32| ((d.to_f64() - exact) / exact).abs();
    let mut errs = Vec::new();
    for mode in [
        ReduceMode::Plain,
        ReduceMode::Kahan,
        ReduceMode::DoubleFloat,
    ] {
        let job = || Job::Reduce {
            mode,
            count: n,
            values: x.clone(),
        };
        let JobOutput::Sums(g) = run_output(b, job()).unwrap() else {
            panic!()
        };
        let JobOutput::Sums(e) = run_output(&cpu, job()).unwrap() else {
            panic!()
        };
        assert_eq!(bits(&g), bits(&e), "{mode:?}: GPU differs from emulation");
        errs.push(rel(g[0]));
    }
    eprintln!("relative errors plain/kahan/df: {errs:?}");
    assert!(errs[2] <= 1e-14, "double-float sum error {:e}", errs[2]);
    assert!(errs[1] <= 1e-12, "Kahan sum error {:e}", errs[1]);
    assert!(
        errs[0] > 1e-10,
        "plain f32 unexpectedly accurate: {:e}",
        errs[0]
    );
}

/// Ill-conditioned, multi-channel: alternating ±1000 plus small values.
#[test]
fn compensated_reduction_cancellation_multichannel() {
    let b = gpu_or_skip!();
    let mut r = Philox::new(14, 0);
    let count = 300_001;
    let channels = 3;
    let mut values = Vec::with_capacity(count * channels);
    for c in 0..channels {
        for i in 0..count {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            values.push((s * 1000.0 * (c + 1) as f64 + r.uniform()) as f32);
        }
    }
    let JobOutput::Sums(g) = run_output(
        b,
        Job::Reduce {
            mode: ReduceMode::DoubleFloat,
            count,
            values: values.clone(),
        },
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(g.len(), channels);
    for c in 0..channels {
        let ch = &values[c * count..(c + 1) * count];
        let exact = ch.iter().map(|&v| v as f64).collect::<KahanSum>().value();
        let l1: f64 = ch.iter().map(|v| v.abs() as f64).sum();
        let err = (g[c].to_f64() - exact).abs();
        assert!(err <= 4e-14 * l1, "channel {c}: {err:e} vs l1 {l1:e}");
    }
}

#[test]
fn sag_probe_reads_uploaded_layout() {
    let b = gpu_or_skip!();
    let cpu = CpuF32Backend::new();
    let rs = testutil::singlet_resolved();
    let packed = PackedSystem::pack_all(&rs).unwrap();
    // Two systems in one set; probe the second to exercise surface_offset.
    let hg = b.upload_systems(&[packed.clone(), packed.clone()]).unwrap();
    let hc = cpu.upload_systems(&[packed.clone(), packed]).unwrap();
    let r2: Vec<f32> = (0..=1000)
        .map(|i| (12.0 * i as f32 / 1000.0).powi(2))
        .collect();
    for surface in [1u32, 2] {
        let job = |h| Job::SagProbe {
            systems: h,
            system: 1,
            surface,
            r2: r2.clone(),
        };
        let JobOutput::F32(g) = run_output(b, job(hg)).unwrap() else {
            panic!()
        };
        let JobOutput::F32(e) = run_output(&cpu, job(hc)).unwrap() else {
            panic!()
        };
        let p = b.precision().probe;
        if p.correctly_rounded_div && p.correctly_rounded_sqrt {
            let gb: Vec<u32> = g.iter().map(|v| v.to_bits()).collect();
            let eb: Vec<u32> = e.iter().map(|v| v.to_bits()).collect();
            assert_eq!(gb, eb, "surface {surface}");
        }
        for (k, &q) in r2.iter().enumerate() {
            let exact = rs.surfaces[surface as usize].shape.sag(q as f64).unwrap();
            assert!(
                (g[k] as f64 - exact).abs() <= 2e-6 * exact.abs().max(1e-3),
                "surface {surface} r2 {q}: {} vs {exact}",
                g[k]
            );
        }
    }
    b.release_systems(hg).unwrap();
    let again = b.submit(
        Job::SagProbe {
            systems: hg,
            system: 0,
            surface: 1,
            r2: vec![1.0],
        },
        &JobContext::new(),
    );
    assert!(again.is_err());
}

/// The runtime-compiled library (fallback path) has the same numerics
/// contract and produces the same bits as the precompiled metallib.
#[test]
fn runtime_source_fallback_matches() {
    if gpu().is_none() {
        return;
    }
    let rt = MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::RuntimeSource,
        ..Default::default()
    })
    .unwrap()
    .unwrap();
    assert_eq!(rt.context().library_origin(), LibraryOrigin::RuntimeSource);
    assert!(
        rt.precision().probe.error_free_transforms(),
        "{:?}",
        rt.precision().probe
    );
    let mut r = Philox::new(15, 0);
    let a: Vec<Df32> = (0..4096).map(|_| rnd_df(&mut r, -10, 10)).collect();
    let c: Vec<Df32> = (0..4096).map(|_| rnd_df(&mut r, -10, 10)).collect();
    let job = || Job::DfOps {
        op: DfOp::Mul,
        a: a.clone(),
        b: c.clone(),
    };
    let JobOutput::Df(x) = run_output(&rt, job()).unwrap() else {
        panic!()
    };
    let JobOutput::Df(y) = run_output(gpu().unwrap(), job()).unwrap() else {
        panic!()
    };
    assert_eq!(bits(&x), bits(&y));
}

#[test]
fn chunking_progress_cancellation_and_timing() {
    let b = gpu_or_skip!();
    // A private backend so the chunk size does not affect other tests.
    let b2 = MetalBackend::new().unwrap().unwrap();
    b2.set_chunk_items(10_000);
    let ctx = JobContext::new();
    let res = b2
        .run(
            Job::PhiloxStream {
                seed: 3,
                stream: 0,
                first_block: 0,
                num_blocks: 95_000,
            },
            &ctx,
        )
        .unwrap();
    assert_eq!(res.stats.command_buffers, 10);
    assert!(res.stats.device_seconds > 0.0);
    assert_eq!(ctx.progress.fraction(), 1.0);
    // Chunked output equals the single-command-buffer output.
    let JobOutput::Philox { raw, .. } = res.output else {
        panic!()
    };
    let JobOutput::Philox { raw: whole, .. } = run_output(
        b,
        Job::PhiloxStream {
            seed: 3,
            stream: 0,
            first_block: 0,
            num_blocks: 95_000,
        },
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(raw, whole);

    // Cancelled before the first command buffer.
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    let job = Job::PhiloxStream {
        seed: 3,
        stream: 0,
        first_block: 0,
        num_blocks: 95_000,
    };
    assert_eq!(b2.run(job.clone(), &ctx), Err(MkError::Cancelled));

    // Cancelled through the Pending handle while running: either it finished
    // first or it reports Cancelled, never a partial result.
    b2.set_chunk_items(1_000);
    let pending = b2.submit(job, &JobContext::new()).unwrap();
    pending.cancel();
    match pending.wait() {
        Ok(r) => assert_eq!(r.stats.command_buffers, 95),
        Err(e) => assert_eq!(e, MkError::Cancelled),
    }
}

#[test]
fn async_submission_and_unsupported_jobs() {
    let b = gpu_or_skip!();
    let pendings: Vec<_> = (0..4u64)
        .map(|s| {
            b.submit(
                Job::PhiloxStream {
                    seed: s,
                    stream: 1,
                    first_block: 0,
                    num_blocks: 1024,
                },
                &JobContext::new(),
            )
            .unwrap()
        })
        .collect();
    for (s, p) in pendings.into_iter().enumerate() {
        let JobOutput::Philox { raw, .. } = p.wait().unwrap().output else {
            panic!()
        };
        let mut r = Philox::new(s as u64, 1);
        assert!(raw.iter().all(|w| *w == r.next_u32()));
    }
    let rs = testutil::singlet_resolved();
    let h = b
        .upload_system(&PackedSystem::pack_all(&rs).unwrap())
        .unwrap();
    let trace = Job::Trace(Box::new(TraceRequest {
        systems: h,
        rays: Default::default(),
        start_surface: 0,
        end_surface: 3,
        record: Default::default(),
        apply_apertures: true,
    }));
    assert!(matches!(
        b.submit(trace, &JobContext::new()),
        Err(MkError::Unsupported(_))
    ));
    let bad = Job::DfOps {
        op: DfOp::Add,
        a: vec![Df32::ZERO; 3],
        b: vec![Df32::ZERO; 2],
    };
    assert!(b.submit(bad, &JobContext::new()).is_err());
    b.release_systems(h).unwrap();
}
