//! The GPU-thread protocol of the browser build, natively: a backend whose
//! device is reached through `GpuProxy` (a thread owns the wgpu device)
//! gives bit-identical results to direct calls, from many threads at once.

use std::sync::Arc;

use mklight_core::JobContext;
use mklight_core::rng::Philox;
use mklight_gpu::cpu::run_output;
use mklight_gpu::{ComputeBackend, Df32, DfOp, Job, JobOutput, ReduceMode};
use mklight_webgpu::{WebGpuBackend, WgpuDevice, proxy};
use rayon::prelude::*;

#[test]
fn proxy_backend_matches_direct_backend() {
    let direct = crate::gpu_or_skip!();
    let device = WgpuDevice::open_native().unwrap().unwrap();
    let (gpu, channel) = proxy::spawn_native(device).unwrap();
    let b = WebGpuBackend::with_gpu(Arc::new(gpu)).unwrap();
    assert_eq!(b.precision().probe, direct.precision().probe);
    let mut r = Philox::new(21, 0);
    let a: Vec<Df32> = (0..10_000)
        .map(|_| Df32::from_f64(r.uniform() * 100.0 - 50.0))
        .collect();
    let c: Vec<Df32> = (0..10_000)
        .map(|_| Df32::from_f64(r.uniform() * 100.0 + 1.0))
        .collect();
    let values: Vec<f32> = (0..300_000).map(|_| r.uniform() as f32).collect();
    // Engine threads call concurrently; the GPU thread serialises them.
    (0..16).into_par_iter().for_each(|k| {
        let job = if k % 2 == 0 {
            Job::DfOps {
                op: DfOp::Div,
                a: a.clone(),
                b: c.clone(),
            }
        } else {
            Job::Reduce {
                mode: ReduceMode::DoubleFloat,
                count: values.len() / 3,
                values: values.clone(),
            }
        };
        let via_proxy = run_output(&b, job.clone()).unwrap();
        let direct = run_output(direct, job).unwrap();
        match (via_proxy, direct) {
            (JobOutput::Df(x), JobOutput::Df(y)) | (JobOutput::Sums(x), JobOutput::Sums(y)) => {
                assert!(
                    x.iter()
                        .zip(&y)
                        .all(|(p, q)| p.hi.to_bits() == q.hi.to_bits()
                            && p.lo.to_bits() == q.lo.to_bits())
                );
            }
            _ => panic!("unexpected outputs"),
        }
    });
    // A stopped GPU thread fails calls instead of hanging them.
    channel.close("stopped by the test");
    let e = run_output(
        &b,
        Job::Reduce {
            mode: ReduceMode::DoubleFloat,
            count: 4,
            values: vec![1.0; 4],
        },
    );
    assert!(e.is_err(), "{e:?}");
    let _ = JobContext::new();
}
