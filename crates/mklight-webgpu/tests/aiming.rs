//! Pupil aiming on the device (`GenerateAim` from host chief seeds, with a
//! host seed table or none) against the CPU-f32 specification in the
//! device's environment, bit for bit: generated rays, reports and the
//! resident trace of them (the WebGPU twin of the host-seeded cases of
//! `mklight-metal/tests/p1_aiming.rs`).

use mklight_core::{JobContext, RayAiming};
use mklight_gpu::aiming::*;
use mklight_gpu::resident::*;
use mklight_gpu::testutil;
use mklight_gpu::{ComputeBackend, CpuF32Backend, QualificationBackend};
use mklight_trace::{PupilSampling, SequentialTracer};

use crate::trace::p1_support;

fn run(b: &dyn QualificationBackend, p: ResidentPlan) -> (Vec<BufferData>, PlanResult) {
    let ctx = JobContext::new();
    let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    (p1_support::read_all(b, &r), r)
}

/// The host-seeded plan of `make_plan` on backend `b`, with a host seed
/// table of `side` nodes per axis when given.
fn plan(
    b: &dyn QualificationBackend,
    t: &SequentialTracer,
    field: usize,
    wl: usize,
    points: &[[f64; 2]],
    table: Option<u32>,
) -> (ResidentPlan, mklight_gpu::SystemHandle) {
    let (p, h) = p1_support::make_plan(b, t, field, wl, points, false, 40, false);
    let mut p = p1_support::host_seeded(b, p, t, field, wl);
    if let Some(side) = table {
        let nodes = host_tables(t, &[(t.field(field), wl)], side);
        let v = p1_support::upload(b, &mut p, nodes.len(), &nodes);
        for op in &mut p.operations {
            if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
                g.options.table = Some(v.clone());
                g.options.table_side = side;
                g.options.schedule_cap = 24;
                g.options.cull_outside = true;
            }
        }
    }
    (p, h)
}

#[test]
fn host_seeded_generation_matches_cpu_bitwise() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let mut points = vec![[0., 0.], [1.2, 0.3]];
    points.extend((PupilSampling::Grid { n: 9 }).points());
    for (name, model) in testutil::xval_systems() {
        for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            let mut model = model.clone();
            model.ray_aiming = aiming;
            let t = testutil::tracer(model);
            let field = t.fields().len() - 1;
            let wl = t.num_wavelengths() / 2;
            for table in [None, Some(mklight_gpu::consts::MKL_AIM_TABLE_SIDE)] {
                let (pg, hg) = plan(g, &t, field, wl, &points, table);
                let (pc, hc) = plan(&c, &t, field, wl, &points, table);
                let (a, r) = run(g, pg);
                let (b, rc) = run(&c, pc);
                assert!(
                    r.stats
                        .stages
                        .iter()
                        .any(|s| s.name == "webgpu:generate_aim"),
                    "{name}: {:?}",
                    r.stats.stages
                );
                eprintln!(
                    "{name} {aiming:?} table {table:?}: bitwise ({} iterations, {} failed)",
                    r.stats.aiming.iterations, r.stats.aiming.failed
                );
                p1_support::bits(&a, &b);
                assert_eq!(r.stats.aiming, rc.stats.aiming, "{name}");
                assert_eq!(r.stats.fallback_count, rc.stats.fallback_count, "{name}");
                g.release_systems(hg).unwrap();
                c.release_systems(hc).unwrap();
            }
        }
    }
}

/// A paraxial generation larger than one launch, with skip-flagged
/// samples: launch boundaries change no bit.
#[test]
fn launches_and_skipped_samples() {
    let g = crate::gpu_or_skip!();
    let c = CpuF32Backend::matching(&g.precision().probe);
    let t = testutil::tracer(testutil::xval_systems()[0].1.clone());
    let n = mklight_webgpu::ops::aiming::AIM_LAUNCH + 1000;
    let points: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = i as f64 * 0.618;
            let r = (i as f64 / n as f64).sqrt() * 1.1;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    let build = |b: &dyn QualificationBackend| {
        let (mut p, h) = plan(b, &t, 0, 0, &points, None);
        for op in &mut p.operations {
            if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
                g.options.cull_outside = true;
            }
        }
        (p, h)
    };
    let ((a, r), (b, _)) = (run(g, build(g).0), run(&c, build(&c).0));
    assert!(
        r.stats
            .stages
            .iter()
            .any(|s| s.name == "webgpu:generate_aim")
    );
    p1_support::bits(&a, &b);
}
