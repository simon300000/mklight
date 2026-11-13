//! Transforms on the device against the CPU-f32 specification and f64
//! under the GPU.md §9.4 "T" gates: relative L2 ≤1e-4, max abs ≤1e-4 × the
//! scaled L1 input norm, round trip and Parseval ≤1e-4. The shared
//! qualification suites of `mklight-gpu/tests/p2_transforms.rs` (known
//! answers, small DFT/MFT, chirp, correlation) run unchanged on the device.

#[path = "../../mklight-gpu/tests/p2_transforms.rs"]
mod common;

use common::*;
use mklight_core::JobContext;
use mklight_gpu::{resident::*, transforms::*, *};
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64 as C64;
use std::time::Instant;

type C = [f32; 2];

fn spec(b: &mklight_webgpu::WebGpuBackend) -> CpuF32Backend {
    CpuF32Backend::matching(&b.precision().probe)
}

/// (relative L2, max abs / l1) of `got` against `want`, asserted against
/// the T gates.
fn gate(what: &str, got: &[C], want: &[[f64; 2]], l1: f64) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{what}");
    let (mut err, mut norm, mut max) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        for i in 0..2 {
            let e = a[i] as f64 - b[i];
            assert!(e.is_finite(), "{what}: non-finite");
            err += e * e;
            norm += b[i] * b[i];
            max = max.max(e.abs());
        }
    }
    let rel = (err / norm.max(1e-300)).sqrt();
    let scaled = max / l1;
    eprintln!("{what}: relative L2 {rel:.2e}, max abs/L1 {scaled:.2e}");
    assert!(rel <= 1e-4, "{what}: relative L2 {rel:e}");
    assert!(scaled <= 1e-4, "{what}: max abs / L1 {scaled:e}");
    (rel, scaled)
}

fn l1(data: &[C], scale: f32) -> f64 {
    data.iter()
        .map(|z| (z[0] as f64).hypot(z[1] as f64))
        .sum::<f64>()
        * scale.abs() as f64
}

fn f64s(v: &[C]) -> Vec<[f64; 2]> {
    v.iter().map(|z| z.map(f64::from)).collect()
}

/// Unnormalised f64 FFT of `h` rows of `w` (2-D if h > 1), times `scale`.
fn fft_f64(data: &[C], w: usize, h: usize, sign: i32, scale: f64) -> Vec<[f64; 2]> {
    let mut planner = FftPlanner::<f64>::new();
    let mut plan = |n| {
        if sign < 0 {
            planner.plan_fft_forward(n)
        } else {
            planner.plan_fft_inverse(n)
        }
    };
    let (row, col) = (plan(w), plan(h));
    let mut a: Vec<C64> = data
        .iter()
        .map(|z| C64::new(z[0] as f64, z[1] as f64))
        .collect();
    a.par_chunks_mut(w).for_each(|r| row.process(r));
    if h > 1 {
        let mut t = vec![C64::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                t[x * h + y] = a[y * w + x];
            }
        }
        t.par_chunks_mut(h).for_each(|c| col.process(c));
        for y in 0..h {
            for x in 0..w {
                a[y * w + x] = t[x * h + y];
            }
        }
    }
    a.iter().map(|z| [z.re * scale, z.im * scale]).collect()
}

/// f64 separable MFT of a diagonal map:
/// out[l,k] = scale Σ in[y,x] exp(sign 2πi (x(a0 k + a2) + y(a4 l + a5))).
fn mft_f64(
    data: &[C],
    (w, h): (usize, usize),
    (ow, oh): (usize, usize),
    sign: i32,
    scale: f64,
    a: [f64; 6],
) -> Vec<[f64; 2]> {
    let cis = |t: f64| {
        let (s, c) = (sign as f64 * std::f64::consts::TAU * (t - t.round())).sin_cos();
        C64::new(c, s)
    };
    // Along x: t[y][k] = Σ_x in[y][x] kx[k][x].
    let kx: Vec<C64> = (0..ow * w)
        .map(|q| cis((q % w) as f64 * (a[0] * (q / w) as f64 + a[2])))
        .collect();
    let mut t = vec![C64::default(); h * ow];
    t.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        let src: Vec<C64> = data[y * w..(y + 1) * w]
            .iter()
            .map(|z| C64::new(z[0] as f64, z[1] as f64))
            .collect();
        for (k, o) in row.iter_mut().enumerate() {
            *o = src
                .iter()
                .zip(&kx[k * w..(k + 1) * w])
                .map(|(s, e)| s * e)
                .sum();
        }
    });
    let ky: Vec<C64> = (0..oh * h)
        .map(|q| cis((q % h) as f64 * (a[4] * (q / h) as f64 + a[5])))
        .collect();
    let mut out = vec![[0f64; 2]; oh * ow];
    out.par_chunks_mut(ow).enumerate().for_each(|(l, row)| {
        for (k, o) in row.iter_mut().enumerate() {
            let z: C64 = (0..h).map(|y| t[y * ow + k] * ky[l * h + y]).sum();
            *o = [z.re * scale, z.im * scale];
        }
    });
    out
}

/// The operations' stage names (without the plan's closing "metadata").
fn stage_names(r: &PlanResult) -> Vec<String> {
    r.stats
        .stages
        .iter()
        .map(|s| s.name.clone())
        .filter(|n| n != "metadata")
        .collect()
}

/// Every operation of the plan ran on the device.
fn on_device(r: &PlanResult) {
    let names = stage_names(r);
    assert!(
        names.iter().all(|n| n.starts_with("webgpu:")),
        "stages {names:?}"
    );
}

fn fft(
    b: &dyn QualificationBackend,
    data: &[C],
    dims: &[usize],
    sign: i32,
    scale: f32,
) -> (Vec<C>, PlanResult) {
    run(
        b,
        data,
        dims,
        dims,
        TransformKind::Fft,
        sign,
        scale,
        [Df32::ZERO; 6],
    )
}

fn bits(v: &[C]) -> Vec<u32> {
    bytemuck::cast_slice(v).to_vec()
}

#[test]
fn shared_qualification_suites() {
    let b = crate::gpu_or_skip!();
    suite(b);
    correlation_suite(b);
    primitives_suite(b);
    phase_and_views(b);
    complex_correlation_suite(b);
    physical_grid_mft(b);
}

/// Power-of-two and mixed-radix FFTs, shared-memory rows and global stages,
/// 1-D and 2-D up to 2048², against f64 and the CPU-f32 specification.
#[test]
fn fft_against_f64_and_specification() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let mut identical = vec![];
    for dims in [
        vec![8],
        vec![64],
        vec![1000],
        vec![2048],
        vec![3000],
        vec![4096],
        vec![4200],
        vec![8192],
        vec![16, 32],
        vec![7, 1536],
        vec![256, 256],
        vec![512, 512],
        vec![1000, 1000],
        vec![1024, 1024],
        vec![2048, 2048],
    ] {
        let (w, h) = recipe::shape(&dims);
        let a = input(w * h);
        for (sign, scale) in [(-1, 1.0), (1, 0.37)] {
            let (g, r) = fft(b, &a, &dims, sign, scale);
            on_device(&r);
            assert_eq!(stage_names(&r), ["webgpu:transform"]);
            let want = fft_f64(&a, w, h, sign, scale as f64);
            gate(
                &format!("fft {dims:?} sign {sign}"),
                &g,
                &want,
                l1(&a, scale),
            );
            let (c, _) = fft(&cpu, &a, &dims, sign, scale);
            gate(
                &format!("fft {dims:?} sign {sign} vs CPU-f32"),
                &g,
                &f64s(&c),
                l1(&a, scale),
            );
            identical.push((dims.clone(), bits(&g) == bits(&c)));
        }
    }
    eprintln!("bitwise equal to the CPU-f32 specification: {identical:?}");
    // The butterflies are the specification's operations, protected from
    // fast math: every FFT reproduces its bits.
    assert!(identical.iter().all(|(_, same)| *same), "{identical:?}");
}

/// Bluestein (non-smooth) FFTs: fused chirp rows and, past 2048 padded
/// points, the unfused global path.
#[test]
fn non_power_of_two_chirp_fft() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    for dims in [
        vec![11],
        vec![1031],
        vec![2039],
        vec![101, 211],
        vec![61, 2039],
        vec![1021, 1021],
    ] {
        let (w, h) = recipe::shape(&dims);
        let a = input(w * h);
        for sign in [-1, 1] {
            let (g, r) = fft(b, &a, &dims, sign, 0.5);
            on_device(&r);
            gate(
                &format!("bluestein {dims:?} sign {sign}"),
                &g,
                &fft_f64(&a, w, h, sign, 0.5),
                l1(&a, 0.5),
            );
            let (c, _) = fft(&cpu, &a, &dims, sign, 0.5);
            gate(
                &format!("bluestein {dims:?} vs CPU-f32"),
                &g,
                &f64s(&c),
                l1(&a, 0.5),
            );
        }
    }
}

/// Chirp-z (separable MFT) on square and rectangular grids against an f64
/// MFT, and at 2048² against the CPU-f32 specification; skew affine MFT
/// against the direct sum.
#[test]
fn chirp_and_affine_mft() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    for (dims, od) in [
        (vec![64, 64], vec![64, 64]),
        (vec![200, 300], vec![384, 256]),
        (vec![256, 256], vec![256, 256]),
        (vec![512, 512], vec![512, 512]),
        (vec![1024, 1024], vec![1024, 1024]),
        (vec![2048, 2048], vec![2048, 2048]),
    ] {
        let (w, h) = recipe::shape(&dims);
        let (ow, oh) = recipe::shape(&od);
        let a = input(w * h);
        let affine = [0.31 / w as f64, 0., -0.15, 0., 0.27 / h as f64, 0.11];
        let df = affine.map(Df32::from_f64);
        let (g, r) = run(b, &a, &dims, &od, TransformKind::ChirpZ, 1, 0.25, df);
        on_device(&r);
        let (c, _) = run(&cpu, &a, &dims, &od, TransformKind::ChirpZ, 1, 0.25, df);
        gate(
            &format!("chirp {dims:?}->{od:?} vs CPU-f32"),
            &g,
            &f64s(&c),
            l1(&a, 0.25),
        );
        if w <= 1024 {
            let want = mft_f64(&a, (w, h), (ow, oh), 1, 0.25, affine);
            gate(&format!("chirp {dims:?}->{od:?}"), &g, &want, l1(&a, 0.25));
        }
    }
    let (dims, od) = (vec![40, 48], vec![56, 64]);
    let a = input(40 * 48);
    let affine = [0.011, -0.004, 0.13, -0.004, 0.009, -0.21].map(Df32::from_f64);
    let (g, r) = run(b, &a, &dims, &od, TransformKind::AffineMft, -1, 0.5, affine);
    on_device(&r);
    let want = dft(&a, &dims, &od, -1, 0.5, Some(affine));
    gate("affine MFT 48x40 -> 64x56", &g, &want, l1(&a, 0.5));
}

/// Forward then inverse at 1024² and 2048² restores the input; Parseval.
#[test]
fn round_trip_and_parseval() {
    let b = crate::gpu_or_skip!();
    for n in [1024, 2048] {
        let a = input(n * n);
        let (f, _) = fft(b, &a, &[n, n], -1, 1.0);
        let (back, _) = fft(b, &f, &[n, n], 1, 1.0 / (n * n) as f32);
        gate(&format!("round trip {n}²"), &back, &f64s(&a), l1(&a, 1.0));
        let energy = |v: &[C]| {
            v.iter()
                .map(|z| z[0] as f64 * z[0] as f64 + z[1] as f64 * z[1] as f64)
                .sum::<f64>()
        };
        let parseval = (energy(&f) / ((n * n) as f64 * energy(&a)) - 1.).abs();
        eprintln!("Parseval {n}²: {parseval:.2e}");
        assert!(parseval <= 1e-4);
    }
}

/// Repeated runs are bitwise identical, including the first run that builds
/// (and caches) the chirp spectra and later runs that reuse them.
#[test]
fn repeated_runs_bitwise_identical() {
    let b = crate::gpu_or_skip!();
    let a = input(300 * 200);
    let affine = [0.002, 0., 0.1, 0., 0.003, -0.2].map(Df32::from_f64);
    let runs: Vec<_> = (0..3)
        .map(|_| {
            let (g, r) = run(
                b,
                &a,
                &[200, 300],
                &[250, 270],
                TransformKind::ChirpZ,
                -1,
                1.,
                affine,
            );
            on_device(&r);
            bits(&g)
        })
        .collect();
    assert!(runs.iter().all(|r| *r == runs[0]));
    let a = input(512 * 512);
    let (f1, _) = fft(b, &a, &[512, 512], -1, 1.);
    let (f2, _) = fft(b, &a, &[512, 512], -1, 1.);
    assert_eq!(bits(&f1), bits(&f2));
}

/// 4096² fits one 128 MiB binding and runs on the device; a grid past the
/// binding limit runs its CPU-f32 specification (stage "domain") instead of
/// failing the plan.
#[test]
fn binding_limit_4096_and_fallback() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let n = 4096;
    let a = input(n * n);
    let (g, r) = fft(b, &a, &[n, n], -1, 1. / n as f32);
    on_device(&r);
    let (c, _) = fft(&cpu, &a, &[n, n], -1, 1. / n as f32);
    gate("fft 4096² vs CPU-f32", &g, &f64s(&c), l1(&a, 1. / n as f32));
    assert_eq!(bits(&g), bits(&c));
    // 4096 × 4116 complex values exceed 128 MiB.
    let dims = [4116, 4096];
    let a = input(4116 * 4096);
    let (g, r) = fft(b, &a, &dims, 1, 1.);
    assert_eq!(stage_names(&r), ["domain"]);
    let (c, _) = fft(&cpu, &a, &dims, 1, 1.);
    assert_eq!(bits(&g), bits(&c));
}

/// An 8192² transform exceeds the 1 GiB working-set limit of
/// `transforms::validate`: the plan is rejected exactly as on the CPU-f32
/// specification (no backend runs it).
#[test]
#[ignore = "allocates 512 MiB of host input"]
fn side_8192_rejected_like_specification() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let n = 8192;
    for be in [b as &dyn QualificationBackend, &cpu] {
        let h = upload(be, vec![[0.; 2]; n * n], vec![n, n]);
        let mut p = ResidentPlan::new(snapshot());
        let i = p.import(h.view());
        let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![n, n]));
        p.operations
            .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
                kind: TransformKind::Fft,
                input: p.view(i).unwrap(),
                output: o,
                dimensions: vec![n, n],
                output_dimensions: vec![n, n],
                sign: -1,
                scale: 1.,
                affine: [Df32::ZERO; 6],
            })));
        p.outputs = vec![o];
        let e = be.submit_plan(p, &JobContext::new()).and_then(|q| q.wait());
        assert!(
            matches!(e, Err(mklight_core::MkError::Unsupported(_))),
            "{e:?}"
        );
    }
}

/// Device correlations against the CPU-f32 specification on a larger grid
/// with many frequency tiles.
#[test]
fn correlation_against_specification() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let n = 128;
    let mut amplitude = vec![0f32; n * n];
    let mut coverage = vec![0f32; n * n];
    let mut opd = vec![Df32::ZERO; n * n];
    for k in 0..n * n {
        let (u, v) = (
            (2 * (k % n) + 1) as f64 / n as f64 - 1.,
            (2 * (k / n) + 1) as f64 / n as f64 - 1.,
        );
        if u * u + v * v < 1. {
            let edge = if u * u + v * v > 0.9 { 0.6 } else { 1. };
            (amplitude[k], coverage[k], opd[k]) = (
                (0.9 + 0.1 * u) as f32,
                edge,
                Df32::from_f64(3. * u * v + 0.7 * u),
            );
        }
    }
    let frequencies: Vec<f32> = (0..300)
        .flat_map(|i| [i as f32 * 1.7, -(i as f32) * 0.4])
        .collect();
    let run = |be: &dyn QualificationBackend| {
        let mut p = ResidentPlan::new(snapshot());
        let import = |p: &mut ResidentPlan, ty, shape, data| {
            let i = p.import(
                be.upload_buffer(BufferDesc::new(ty, shape), data)
                    .unwrap()
                    .view(),
            );
            p.view(i).unwrap()
        };
        let amplitude = import(
            &mut p,
            ElementType::F32,
            vec![n, n],
            BufferData::F32(amplitude.clone()),
        );
        let coverage = import(
            &mut p,
            ElementType::F32,
            vec![n, n],
            BufferData::F32(coverage.clone()),
        );
        let opd = import(
            &mut p,
            ElementType::DoubleFloat,
            vec![n, n],
            BufferData::DoubleFloat(opd.clone()),
        );
        let frequencies = import(
            &mut p,
            ElementType::F32,
            vec![300, 2],
            BufferData::F32(frequencies.clone()),
        );
        let st = p.scratch(BufferDesc::new(ElementType::F32, vec![300, n * n, 5]));
        let (s, c) = 0.2f64.sin_cos();
        let mut req = OpticalCorrelation {
            stencil: None,
            amplitude,
            opd,
            coverage,
            frequencies,
            frame: [[c, 0., -s], [0., 1., 0.], [s, 0., c]].map(|v| v.map(Df32::from_f64)),
            sin_max: Df32::from_f64(0.1),
            wavelength_over_index: Df32::from_f64(0.00055),
            output: st,
        };
        p.operations
            .push(ResidentOp::Transforms(Request::PrepareOpticalStencil(
                req.clone(),
            )));
        let fresh = p.scratch(BufferDesc::new(ElementType::Complex32, vec![300]));
        req.output = fresh;
        p.operations
            .push(ResidentOp::Transforms(Request::OpticalCorrelation(
                req.clone(),
            )));
        let prepared = p.scratch(BufferDesc::new(ElementType::Complex32, vec![300]));
        req.output = prepared;
        req.stencil = Some(p.view(st).unwrap());
        p.operations
            .push(ResidentOp::Transforms(Request::OpticalCorrelation(req)));
        p.outputs = vec![fresh, prepared];
        let r = be
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let read = |i: usize| {
            let BufferData::Complex32(v) = be
                .readback(r.outputs[i].view(), &JobContext::new())
                .unwrap()
                .data
            else {
                panic!()
            };
            v
        };
        (read(0), read(1), stage_names(&r))
    };
    let (g_fresh, g_prepared, names) = run(b);
    assert_eq!(
        names,
        [
            "webgpu:optical-stencil",
            "webgpu:optical-correlation",
            "webgpu:optical-correlation"
        ]
    );
    assert_eq!(bits(&g_fresh), bits(&g_prepared));
    let (c_fresh, _, _) = run(&cpu);
    let max = g_fresh
        .iter()
        .zip(&c_fresh)
        .flat_map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()])
        .fold(0f32, f32::max);
    eprintln!("optical correlation 128² × 300: max abs vs CPU-f32 {max:.2e}");
    assert!(max < 1e-5, "{max}");
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// A plan of `batch` transforms of distinct N² inputs (the PSF/MTF adapters
/// run one per pupil in a plan).
fn bench_plan(
    b: &dyn QualificationBackend,
    n: usize,
    batch: usize,
    kind: TransformKind,
) -> (ResidentPlan, Vec<BufferHandle>) {
    let mut p = ResidentPlan::new(snapshot());
    let mut handles = vec![];
    let affine = [0.31 / n as f64, 0., -0.15, 0., 0.27 / n as f64, 0.11].map(Df32::from_f64);
    for k in 0..batch {
        let data: Vec<C> = input(n * n)
            .into_iter()
            .map(|z| [z[0] + k as f32, z[1]])
            .collect();
        let h = upload(b, data, vec![n, n]);
        let i = p.import(h.view());
        handles.push(h);
        let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![n, n]));
        p.operations
            .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
                kind,
                input: p.view(i).unwrap(),
                output: o,
                dimensions: vec![n, n],
                output_dimensions: vec![n, n],
                sign: 1,
                scale: 1.,
                affine: if matches!(kind, TransformKind::Fft) {
                    [Df32::ZERO; 6]
                } else {
                    affine
                },
            })));
        p.outputs.push(o);
    }
    (p, handles)
}

/// Median (device ms, wall ms) of `reps` runs after a warm-up.
fn time_plan(b: &dyn QualificationBackend, p: &ResidentPlan, reps: usize) -> (f64, f64) {
    let mut runs = vec![];
    for i in 0..=reps {
        let t = Instant::now();
        let r = b
            .submit_plan(p.clone(), &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        if i > 0 {
            runs.push((r.stats.device_seconds * 1e3, wall));
        }
    }
    runs.sort_by(|a, b| a.1.total_cmp(&b.1));
    runs[runs.len() / 2]
}

/// The CPU path of the reference analyses: f64 rustfft rows in parallel
/// with transposes (`mklight-wave` `fft::fft2`), and for chirps the f64
/// chirp convolution of `fft::chirp_mft` (pre-chirp, padded FFT, spectrum,
/// inverse, post-chirp per row, both axes).
fn cpu_reference_ms(n: usize, batch: usize, kind: TransformKind, reps: usize) -> f64 {
    let mut planner = FftPlanner::<f64>::new();
    let data: Vec<C64> = input(n * n)
        .iter()
        .map(|z| C64::new(z[0] as f64, z[1] as f64))
        .collect();
    let chirp = !matches!(kind, TransformKind::Fft);
    let pw = if chirp {
        (2 * n - 1).next_power_of_two()
    } else {
        n
    };
    let fwd = planner.plan_fft_forward(pw);
    let inv = planner.plan_fft_inverse(pw);
    let a = 0.31 / n as f64;
    let pre: Vec<C64> = (0..n)
        .map(|x| C64::from_polar(1., std::f64::consts::PI * a * (x * x) as f64))
        .collect();
    let mut kernel: Vec<C64> = (0..pw)
        .map(|i| {
            let l = if i < n {
                i as f64
            } else if i > pw - n {
                i as f64 - pw as f64
            } else {
                return C64::default();
            };
            C64::from_polar(1., -std::f64::consts::PI * a * l * l)
        })
        .collect();
    fwd.process(&mut kernel);
    let axis = |grid: &mut Vec<C64>| {
        grid.par_chunks_mut(n).for_each(|row| {
            if chirp {
                let mut buf = vec![C64::default(); pw];
                for x in 0..n {
                    buf[x] = row[x] * pre[x];
                }
                fwd.process(&mut buf);
                for (v, k) in buf.iter_mut().zip(&kernel) {
                    *v *= k;
                }
                inv.process(&mut buf);
                for x in 0..n {
                    row[x] = buf[x] * pre[x] / pw as f64;
                }
            } else {
                fwd.process(row);
            }
        });
        let mut t = vec![C64::default(); n * n];
        for y in 0..n {
            for x in 0..n {
                t[x * n + y] = grid[y * n + x];
            }
        }
        *grid = t;
    };
    let mut times = vec![];
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        for _ in 0..batch {
            let mut g = data.clone();
            axis(&mut g);
            axis(&mut g);
            std::hint::black_box(&g);
        }
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

#[test]
#[ignore = "benchmark: cargo test --release -p mklight-webgpu --test tests transforms::bench -- --ignored --nocapture"]
fn bench_transforms() {
    let b = crate::gpu_or_skip!();
    #[cfg(target_os = "macos")]
    let metal = mklight_metal::MetalBackend::new().ok().flatten();
    eprintln!("adapter {:?}", b.adapter().name);
    eprintln!(
        "kind  N     batch | WebGPU dev/wall ms | Metal dev/wall ms | CPU-f32 spec ms | CPU f64 ref ms"
    );
    for kind in [TransformKind::Fft, TransformKind::ChirpZ] {
        for n in [256, 512, 1024, 2048] {
            for batch in [1, 8] {
                let reps = if n >= 2048 { 3 } else { 5 };
                let (p, _h) = bench_plan(b, n, batch, kind);
                let (wd, ww) = time_plan(b, &p, reps);
                #[cfg(target_os = "macos")]
                let (md, mw) = match &metal {
                    Some(m) => {
                        let (p, _h) = bench_plan(m, n, batch, kind);
                        time_plan(m, &p, reps)
                    }
                    None => (f64::NAN, f64::NAN),
                };
                #[cfg(not(target_os = "macos"))]
                let (md, mw) = (f64::NAN, f64::NAN);
                let cpu_spec = if n <= 1024 || batch == 1 {
                    let c = spec(b);
                    let (p, _h) = bench_plan(&c, n, batch, kind);
                    time_plan(&c, &p, 1).1
                } else {
                    f64::NAN
                };
                let reference = cpu_reference_ms(n, batch, kind, reps);
                eprintln!(
                    "{:<5} {n:<5} {batch:<5} | {wd:8.2} / {ww:8.2} | {md:8.2} / {mw:8.2} | {cpu_spec:10.1} | {reference:9.2}",
                    format!("{kind:?}")
                );
            }
        }
    }
}
