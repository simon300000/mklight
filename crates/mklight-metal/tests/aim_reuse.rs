#![cfg(target_os = "macos")]
//! Host f64 chief rows and content-addressed aiming reuse on Metal: host
//! rows replace the device chief solve without leaving the f64 gates, and
//! chief rows/seed tables are reused exactly when every input is bitwise
//! identical (new leases, released handles, concurrent plans), never after
//! geometry, catalogue or environment changes, and never from a cancelled
//! build.
use crate::p1_support as support;
use mklight_core::{JobContext, MkError, RayAiming, samples};
use mklight_gpu::{
    resident::{immutable::CacheCategory, *},
    *,
};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::{PupilSampling, SequentialTracer};
use std::time::{Duration, Instant};
use support::*;

fn backend() -> MetalBackend {
    MetalBackend::with_options(&MetalOptions {
        library: LibraryChoice::Precompiled,
    })
    .unwrap()
    .expect("aiming reuse requires host Metal hardware")
}
fn run(b: &dyn QualificationBackend, p: ResidentPlan) -> (Vec<BufferData>, PlanStats) {
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    (read_all(b, &r), r.stats)
}
fn cache_key(t: &SequentialTracer, b: &dyn ComputeBackend) -> SystemCacheKey {
    SystemCacheKey {
        snapshot: t.system().snapshot.hash().to_hex(),
        library: "aim-reuse".into(),
        configuration: "base".into(),
        spectrum: "all".into(),
        layout: consts::MKL_LAYOUT_VERSION,
        kernel_build: b.precision().kernel_build.clone(),
        operation_settings: "aim".into(),
    }
}
fn real(mut model: mklight_core::OpticalSystem) -> mklight_core::OpticalSystem {
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    model
}
/// A plan template whose system handle is rebound per call, uploaded under
/// one fixed external key (the worst case for handle- or key-based reuse).
struct Case {
    template: ResidentPlan,
    key: SystemCacheKey,
}
impl Case {
    fn new(b: &dyn QualificationBackend, t: &SequentialTracer, host: bool) -> Self {
        let field = t.fields().len() - 1;
        let points = PupilSampling::Grid { n: 64 }.points();
        assert!(points.len() >= aiming::TABLE_MIN_RAYS_PER_ROW);
        let (p, h) = make_plan(b, t, field, 1, &points, false, 40, false);
        b.release_systems(h).unwrap();
        let template = if host {
            host_seeded(b, p, t, field, 1)
        } else {
            p
        };
        Self {
            template,
            key: cache_key(t, b),
        }
    }
    fn run(
        &self,
        b: &dyn QualificationBackend,
        packed: PackedSystem,
        cached: bool,
    ) -> (Vec<BufferData>, PlanStats) {
        let h = b.upload_systems_cached(&self.key, &[packed]).unwrap();
        let p = rebind(self.template.clone(), h);
        let out = run(b, if cached { p } else { uncached(p) });
        b.release_systems(h).unwrap();
        out
    }
}
fn hits(s: &PlanStats, c: CacheCategory) -> (u64, u64) {
    s.cache.get(&c).map_or((0, 0), |s| (s.hits, s.misses))
}

/// Host rows (the reference tracer's prepared fields) replace the device
/// chief and real-image-height solve: Metal stays bitwise equal to the CPU-f32
/// specification, inside the f64 gates, and never resolves fewer rays than
/// the device-seeded generation.
#[test]
fn host_seeds_match_specification_and_reference() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 7 }).points());
    let mut rows = 0;
    for (name, model) in testutil::xval_systems().into_iter().take(32) {
        for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            let mut model = model.clone();
            model.ray_aiming = aiming;
            let t = testutil::tracer(model);
            for field in 0..t.fields().len() {
                for wl in 0..t.num_wavelengths() {
                    let (p, h) = make_plan(&gpu, &t, field, wl, &points, true, 40, true);
                    let (device, _) = run(&gpu, uncached(p.clone()));
                    let (host, stats) = run(&gpu, host_seeded(&gpu, p, &t, field, wl));
                    let (pc, hc) = make_plan(&cpu, &t, field, wl, &points, true, 40, true);
                    let (spec, _) = run(&cpu, host_seeded(&cpu, pc, &t, field, wl));
                    bits(&host, &spec);
                    let e = check_f64(&t, field, wl, &points, &host, true, true);
                    let d = check_f64(&t, field, wl, &points, &device, true, true);
                    assert!(e.fallback <= d.fallback, "{name} {aiming:?} {e:?} {d:?}");
                    assert!(e.position <= 2e-5 && e.direction <= 2e-6, "{name} {e:?}");
                    assert!(e.opd <= 5e-3, "{name} {e:?}");
                    assert_eq!(stats.cache.get(&CacheCategory::ChiefSources), None);
                    assert_eq!(stats.aiming.rays, points.len() as u64);
                    assert_eq!(stats.aiming.failed, e.fallback as u64);
                    rows += 1;
                    gpu.release_systems(h).unwrap();
                    cpu.release_systems(hc).unwrap();
                }
            }
        }
    }
    eprintln!("host seed rows checked: {rows}");
}

/// The seed-table path with host rows: bitwise equal to the specification,
/// inside the f64 gates and as cheap to converge as with device rows.
#[test]
fn host_seed_tables_match_specification_and_reference() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let points = (PupilSampling::Grid { n: 64 }).points();
    for model in [samples::cooke_triplet(), samples::double_gauss()] {
        let t = testutil::tracer(real(model));
        let field = t.fields().len() - 1;
        let (p, h) = make_plan(&gpu, &t, field, 1, &points, false, 40, true);
        let (pc, hc) = make_plan(&cpu, &t, field, 1, &points, false, 40, true);
        let (data, stats) = run(&gpu, uncached(host_seeded(&gpu, p, &t, field, 1)));
        let (spec, _) = run(&cpu, uncached(host_seeded(&cpu, pc, &t, field, 1)));
        bits(&data, &spec);
        let e = check_f64(&t, field, 1, &points, &data, false, true);
        assert_eq!(e.fallback, 0);
        let mean = stats.aiming.iterations as f64 / stats.aiming.rays as f64;
        assert!(mean < 1.5, "mean Newton iterations {mean}");
        let n = points.len();
        let direct = uints(&data[1])[n..2 * n]
            .iter()
            .map(|&v| v as u64)
            .sum::<u64>();
        assert_eq!(stats.aiming.iterations, direct);
        assert_eq!(stats.aiming.histogram.iter().sum::<u64>(), n as u64);
        gpu.release_systems(h).unwrap();
        cpu.release_systems(hc).unwrap();
    }
}

/// Reuse across new leases of equal content; recomputation after geometry,
/// catalogue (same snapshot and external key) and environment changes; and
/// no stale answer when an earlier snapshot returns.
#[test]
fn reuse_follows_content_not_handles_or_keys() {
    let gpu = backend();
    for host in [false, true] {
        let model = real(samples::cooke_triplet());
        let t = testutil::tracer(model.clone());
        let case = Case::new(&gpu, &t, host);
        let pack = |t: &SequentialTracer| PackedSystem::pack_all(t.system()).unwrap();
        let (first, s1) = case.run(&gpu, pack(&t), true);
        assert_eq!(hits(&s1, CacheCategory::SeedTables).0, 0);
        let (again, s2) = case.run(&gpu, pack(&t), true);
        bits(&first, &again);
        assert_eq!(hits(&s2, CacheCategory::SeedTables), (1, 0));
        if !host {
            assert_eq!(hits(&s2, CacheCategory::ChiefSources), (1, 0));
        }
        let (reference, _) = case.run(&gpu, pack(&t), false);
        bits(&first, &reference);
        // Geometry: a new snapshot under the same external key.
        let mut thick = model.clone();
        thick.surfaces[1].thickness *= 1.1;
        let thick = testutil::tracer(thick);
        let changed = |what: &str, t: &SequentialTracer, packed: PackedSystem| {
            let case = Case {
                template: if host {
                    let field = t.fields().len() - 1;
                    let p = case.template.clone();
                    host_seeded(&gpu, p, t, field, 1)
                } else {
                    case.template.clone()
                },
                key: case.key.clone(),
            };
            let (got, s) = case.run(&gpu, packed.clone(), true);
            assert_eq!(
                hits(&s, CacheCategory::SeedTables).0,
                0,
                "{what} host={host}"
            );
            let (want, _) = case.run(&gpu, packed, false);
            bits(&got, &want);
            assert_ne!(floats(&got[3]), floats(&first[3]));
        };
        changed("geometry", &thick, pack(&thick));
        // Environment: indices change with the temperature.
        let mut warm = model.clone();
        warm.environment.temperature_c = 45.;
        let warm = testutil::tracer(warm);
        changed("environment", &warm, pack(&warm));
        // Catalogue: same snapshot, same key, different glass indices only
        // (another catalogue's data for the same glass names).
        let mut glass = pack(&t);
        for s in &mut glass.surfaces {
            for n in &mut s.n_after {
                if *n > 1.01 {
                    *n *= 1. + 2e-5;
                }
            }
        }
        assert_ne!(
            mklight_gpu::resident::cache::content_digest(&[glass.clone()]),
            mklight_gpu::resident::cache::content_digest(&[pack(&t)])
        );
        if !host {
            changed("catalogue", &t, glass);
        }
        // The original snapshot returns: its own entries, still exact.
        let (back, s) = case.run(&gpu, pack(&t), true);
        bits(&first, &back);
        assert_eq!(hits(&s, CacheCategory::SeedTables), (1, 0));
    }
}

/// Released handles stay invalid, while equal content uploaded again (a new
/// handle and a new allocation) still finds its entries.
#[test]
fn released_handles_do_not_pin_or_poison_reuse() {
    let gpu = backend();
    let t = testutil::tracer(real(samples::double_gauss()));
    let case = Case::new(&gpu, &t, true);
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    let h = gpu
        .upload_systems_cached(&case.key, std::slice::from_ref(&packed))
        .unwrap();
    let (first, _) = run(&gpu, rebind(case.template.clone(), h));
    gpu.release_systems(h).unwrap();
    let stale = gpu.submit_plan(rebind(case.template.clone(), h), &JobContext::new());
    assert!(
        matches!(stale, Err(MkError::Backend(_))),
        "released handle accepted"
    );
    // An uncached upload of equal content is distinct storage, same identity.
    let h = gpu.upload_systems(&[packed]).unwrap();
    let (again, s) = run(&gpu, rebind(case.template.clone(), h));
    gpu.release_systems(h).unwrap();
    bits(&first, &again);
    assert_eq!(hits(&s, CacheCategory::SeedTables), (1, 0));
}

/// Concurrent leases of two systems: every plan equals its uncached result.
#[test]
fn concurrent_leases_share_only_equal_content() {
    let gpu = backend();
    let models = [
        real(samples::cooke_triplet()),
        real(samples::double_gauss()),
    ];
    let tracers: Vec<_> = models.into_iter().map(testutil::tracer).collect();
    let cases: Vec<_> = tracers.iter().map(|t| Case::new(&gpu, t, false)).collect();
    let references: Vec<_> = (0..2)
        .map(|i| {
            let packed = PackedSystem::pack_all(tracers[i].system()).unwrap();
            cases[i].run(&gpu, packed, false).0
        })
        .collect();
    std::thread::scope(|s| {
        let jobs: Vec<_> = (0..8)
            .map(|k| {
                let (gpu, cases, tracers) = (&gpu, &cases, &tracers);
                s.spawn(move || {
                    let i = k % 2;
                    let packed = PackedSystem::pack_all(tracers[i].system()).unwrap();
                    (i, cases[i].run(gpu, packed, true).0)
                })
            })
            .collect();
        for j in jobs {
            let (i, data) = j.join().unwrap();
            bits(&data, &references[i]);
        }
    });
}

/// Cancelling a plan while it builds its chief rows or seed table publishes
/// nothing (CPU specification, which checks cancellation inside the build),
/// and on Metal the next run is still bitwise the uncached result.
#[test]
fn cancelled_builds_are_never_reused() {
    let gpu = backend();
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let t = testutil::tracer(real(samples::double_gauss()));
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    for (b, stages) in [
        (
            &cpu as &dyn QualificationBackend,
            ["chief aiming", "seed table"],
        ),
        (
            &gpu as &dyn QualificationBackend,
            ["seed table", "pupil aiming"],
        ),
    ] {
        for stage in stages {
            let case = Case::new(b, &t, false);
            let mut case = case;
            // A unique field key per stage keeps earlier entries out of the way.
            for op in &mut case.template.operations {
                if let ResidentOp::Aiming(aiming::Request::GenerateAim(g)) = op {
                    g.field_key = Some(format!("cancel {stage}"));
                }
            }
            let before = b.resident_cache_stats().categories;
            let h = b
                .upload_systems_cached(&case.key, std::slice::from_ref(&packed))
                .unwrap();
            let ctx = JobContext::new();
            let pending = b
                .submit_plan(rebind(case.template.clone(), h), &ctx)
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(60);
            while ctx.progress.stage() != stage && ctx.progress.fraction() < 0.9 {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            ctx.cancel.cancel();
            let cancelled = matches!(pending.wait(), Err(MkError::Cancelled));
            b.release_systems(h).unwrap();
            let after = b.resident_cache_stats().categories;
            if cancelled && b.info().kind == BackendKind::CpuF32 {
                for c in [CacheCategory::ChiefSources, CacheCategory::SeedTables] {
                    assert_eq!(
                        after.get(&c).map(|s| s.inserts),
                        before.get(&c).map(|s| s.inserts),
                        "cancelled {stage} build published {c:?}"
                    );
                }
            }
            let (got, _) = case.run(b, packed.clone(), true);
            let (want, _) = case.run(b, packed.clone(), false);
            bits(&got, &want);
            eprintln!(
                "cancel at {stage} on {:?}: cancelled={cancelled}",
                b.info().kind
            );
        }
    }
}

/// Category budgets are independent and reported with their occupancy.
#[test]
fn categories_report_budgets_and_activity() {
    let gpu = backend();
    let t = testutil::tracer(real(samples::cooke_triplet()));
    let case = Case::new(&gpu, &t, false);
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    case.run(&gpu, packed.clone(), true);
    case.run(&gpu, packed, true);
    let stats = gpu.resident_cache_stats();
    for c in [
        CacheCategory::PreparedSystems,
        CacheCategory::ChiefSources,
        CacheCategory::SeedTables,
        CacheCategory::TracedPupils,
        CacheCategory::TransformPlans,
        CacheCategory::Metadata,
    ] {
        let s = &stats.categories[&c];
        assert!(s.limit_bytes > 0 && s.bytes <= s.limit_bytes, "{c:?} {s:?}");
    }
    let tables = &stats.categories[&CacheCategory::SeedTables];
    assert!(tables.hits >= 1 && tables.inserts >= 1 && tables.rebuild_seconds > 0.);
    assert!(stats.categories[&CacheCategory::PreparedSystems].hits >= 1);
}
