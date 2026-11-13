//! The numerics probe, double-float operations and the fixed-tree reduction
//! on the device against the CPU-f32 emulation and f64.

use mklight_core::JobContext;
use mklight_core::math::KahanSum;
use mklight_core::rng::Philox;
use mklight_gpu::cpu::run_output;
use mklight_gpu::resident::{BufferDesc, ElementType, ResidentPlan};
use mklight_gpu::{ComputeBackend, CpuF32Backend, Df32, DfOp, Job, JobKind, JobOutput, ReduceMode};

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

/// The protected primitives are IEEE whatever the compiler does with plain
/// expressions; the raw probe documents the compiler.
#[test]
fn numerics_probe_protected_operations_are_ieee() {
    let b = crate::gpu_or_skip!();
    let p = b.precision().probe;
    eprintln!("adapter {:?}", b.adapter());
    eprintln!("raw probe {:?}", b.raw_probe());
    eprintln!("protected probe {p:?}");
    eprintln!("kernel build: {}", b.precision().kernel_build);
    assert!(p.error_free_transforms(), "{p:?}");
    assert!(b.supports(JobKind::DfOps) && b.supports(JobKind::Reduce));
    assert_eq!(
        b.info().backend_id(),
        format!("webgpu:{}", b.adapter().name)
    );
}

#[test]
fn df_ops_match_emulation_and_f64() {
    let b = crate::gpu_or_skip!();
    let cpu = CpuF32Backend::matching(&b.precision().probe);
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
            a.iter()
                .zip(&c)
                .map(|(x, y)| {
                    if x.hi.abs() >= y.hi.abs() {
                        (*x, *y)
                    } else {
                        (*y, *x)
                    }
                })
                .unzip()
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
        for i in 0..n {
            let (x, y) = (a[i].to_f64(), c[i].to_f64());
            let (exact, bound) = match op {
                DfOp::Add => (x + y, 3.0 * u2 * (x.abs() + y.abs())),
                DfOp::Mul => (x * y, 4.0 * u2 * (x * y).abs()),
                DfOp::Div => (x / y, 16.0 * u2 * (x / y).abs()),
                // One Karp correction of an approximate sqrt (fast-math
                // compilers, probe.correctly_rounded_sqrt false) lands within
                // ~5u² instead of ~4u².
                DfOp::Sqrt => (
                    x.sqrt(),
                    if probe.correctly_rounded_sqrt {
                        4.0
                    } else {
                        8.0
                    } * u2
                        * x.sqrt(),
                ),
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

/// Summing 10⁶ values: every mode bit-identical to the emulation (fixed
/// order, protected additions only); double-float matches f64 to ~1e-14.
#[test]
fn compensated_reduction_1e6() {
    let b = crate::gpu_or_skip!();
    let cpu = CpuF32Backend::matching(&b.precision().probe);
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
}

/// A resident plan's `Sum` runs on the device (stage `webgpu:sum`) and
/// equals the CPU-f32 backend's plan bit for bit, many channels included.
#[test]
fn resident_sum_matches_emulation() {
    let b = crate::gpu_or_skip!();
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let mut r = Philox::new(15, 0);
    for (channels, count) in [(1, 1), (3, 300_001), (700, 513)] {
        let values: Vec<f32> = (0..channels * count)
            .map(|i| (if i % 2 == 0 { 1000.0 } else { -1000.0 }) + r.uniform() as f32)
            .collect();
        let run = |be: &dyn ComputeBackend| {
            let input = be
                .fill_buffer(
                    BufferDesc::new(ElementType::F32, [channels, count]),
                    &mut |mut f| {
                        f.as_mut_slice::<f32>()?.copy_from_slice(&values);
                        Ok(())
                    },
                )
                .unwrap();
            let snap = mklight_core::Snapshot::new(mklight_gpu::testutil::singlet()).unwrap();
            let mut p = ResidentPlan::new(snap);
            let id = p.import(input.view());
            let v = p.view(id).unwrap();
            let s = p.sum(v, ReduceMode::DoubleFloat);
            p.outputs = vec![s];
            let res = be
                .submit_plan(p, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let names: Vec<String> = res.stats.stages.iter().map(|s| s.name.clone()).collect();
            let sums = be.read_view(res.outputs[0].view()).unwrap();
            (sums.as_slice::<Df32>().unwrap().to_vec(), names)
        };
        let (g, names) = run(b);
        let (e, _) = run(&cpu);
        assert!(names.iter().any(|n| n == "webgpu:sum"), "{names:?}");
        assert_eq!(bits(&g), bits(&e), "{channels}x{count}");
    }
}
