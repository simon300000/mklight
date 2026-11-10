//! The aiming designs (`gpu::Route::aiming`) of the geometric analyses
//! agree with the CPU reference on the built-in samples with real aiming,
//! in field groups of several plans, and name themselves in the run
//! evidence; large generations split between host and device give the same
//! rays however the work is timed. The design and the thresholds are each
//! adapter's own ([`gpu::Route`], [`gpu::Tuning`]).
use mklight_analysis::{
    energy::{self, BestFocusSettings, GeometricMtfResult, GeometricMtfSettings, RaySampling},
    gpu,
    illumination::{self, FootprintSettings},
    spot::{self, SpotDiagram, SpotDiagramSettings},
    tracer_for,
};
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_trace::{Grid, SequentialTracer};

const AIMS: [gpu::AimDesign; 3] = [
    gpu::AimDesign::HostTable,
    gpu::AimDesign::Table,
    gpu::AimDesign::Device,
];

/// The default route with the aiming design `aiming`.
fn aimed(aiming: gpu::AimDesign) -> gpu::Route {
    gpu::Route {
        aiming,
        ..Default::default()
    }
}

/// The default route with the device share `device_share` of split
/// generations.
fn shared(device_share: f64) -> gpu::Route {
    gpu::Route {
        device_share,
        ..Default::default()
    }
}

/// Small thresholds so that field groups form several plans.
fn small_groups() -> gpu::Tuning {
    gpu::Tuning {
        group_min_rays: 64,
        ..Default::default()
    }
}

/// Thresholds under which small generations split into several tiles.
fn small_tiles() -> gpu::Tuning {
    gpu::Tuning {
        split_min_work: 0,
        split_min_tile: 512,
        ..Default::default()
    }
}

/// The device adapter of `t` with thresholds `tuning`, under `route`.
fn adapter<'a>(
    b: &'a dyn ComputeBackend,
    t: &'a SequentialTracer,
    tuning: gpu::Tuning,
    route: &gpu::Route,
) -> gpu::Geometry<'a> {
    gpu::Geometry::new(b, t, &JobContext::new())
        .unwrap()
        .with_route(route)
        .unwrap()
        .with_tuning(tuning)
}

fn close(what: &str, a: &[f64], g: &[f64], tol: impl Fn(f64) -> f64) {
    assert_eq!(a.len(), g.len(), "{what}: length");
    for (&a, &g) in a.iter().zip(g) {
        let ok = if a.is_finite() {
            (a - g).abs() <= tol(a)
        } else {
            a.is_nan() && g.is_nan() || a == g
        };
        assert!(ok, "{what}: reference {a}, resident {g}");
    }
}

fn tracers() -> Vec<(&'static str, SequentialTracer)> {
    [
        ("cooke", samples::cooke_triplet()),
        ("double gauss", samples::double_gauss()),
        ("petzval asphere", samples::petzval_asphere()),
    ]
    .into_iter()
    .map(|(name, mut m)| {
        m.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        (name, tracer_for(m).unwrap())
    })
    .collect()
}

/// The aiming design named in a run's evidence.
fn evidence(run: &mklight_core::RunInfo) -> String {
    let e = &run.settings["resident_execution"];
    e["aiming"]["design"].as_str().unwrap_or("").to_string()
}

fn same_spots(what: &str, a: &SpotDiagram, g: &SpotDiagram) {
    assert_eq!(a.fields.len(), g.fields.len(), "{what}");
    for (a, g) in a.fields.iter().zip(&g.fields) {
        assert_eq!(
            a.polychromatic.rays_arrived, g.polychromatic.rays_arrived,
            "{what}"
        );
        assert_eq!(a.points.len(), g.points.len(), "{what}");
        for (a, g) in a.per_wavelength.iter().zip(&g.per_wavelength) {
            assert_eq!(a.stats.rays_arrived, g.stats.rays_arrived, "{what}");
        }
        close(what, &a.chief, &g.chief, |_| 1e-6);
        close(
            what,
            &[
                a.rms_radius,
                a.geo_radius,
                a.polychromatic.vignetted_fraction,
            ],
            &[
                g.rms_radius,
                g.geo_radius,
                g.polychromatic.vignetted_fraction,
            ],
            |v| 2e-5 + 0.005 * v.abs(),
        );
        for (a, g) in a.points.iter().zip(&g.points) {
            close(what, &[a.x, a.y], &[g.x, g.y], |_| 2e-5);
        }
    }
}

fn same_mtf(what: &str, a: &GeometricMtfResult, g: &GeometricMtfResult) {
    assert_eq!(a.curves.len(), g.curves.len(), "{what}");
    for (a, g) in a.curves.iter().zip(&g.curves) {
        assert_eq!((a.field, a.orientation), (g.field, g.orientation), "{what}");
        close(&format!("{what} MTF"), &a.modulation, &g.modulation, |_| {
            2e-3
        });
    }
}

fn spots(b: &dyn ComputeBackend, aims: &[gpu::AimDesign], rings: usize) {
    for (name, t) in tracers() {
        let s = SpotDiagramSettings {
            rings,
            ..Default::default()
        };
        let a = spot::spot_diagram(&t, &s).unwrap();
        for &aim in aims {
            let g = adapter(b, &t, small_groups(), &aimed(aim))
                .split_large_generations()
                .spot_diagram_job(&s, &JobContext::new())
                .unwrap();
            let what = format!("{name} {}", aim.name());
            assert_eq!(evidence(&g.run_info), aim.name(), "{what}");
            same_spots(&what, &a, &g);
        }
    }
}

fn energies(b: &dyn ComputeBackend, aims: &[gpu::AimDesign]) {
    for (name, t) in tracers() {
        let sampling = RaySampling {
            rings: 4,
            ..Default::default()
        };
        let m = GeometricMtfSettings {
            sampling: sampling.clone(),
            frequency_samples: 5,
            max_frequency: Some(100.0),
            defocus: 0.05,
        };
        let am = energy::geometric_mtf(&t, &m).unwrap();
        let f = BestFocusSettings {
            sampling: sampling.clone(),
            ..Default::default()
        };
        let af = energy::best_focus(&t, &f).unwrap();
        let fp = FootprintSettings {
            surface: 2,
            rings: 4,
            ..Default::default()
        };
        let ap = illumination::footprint(&t, &fp).unwrap();
        let fields: Vec<_> = t.fields().to_vec();
        let wls: Vec<_> = (0..t.num_wavelengths()).collect();
        let pupil = mklight_analysis::util::pupil_points(sampling.pattern, sampling.rings, 1);
        let ag = mklight_trace::trace_grid(
            &t,
            &Grid {
                fields: &fields,
                wavelengths: &wls,
                pupil: &pupil,
            },
            &t.options(),
        );
        for &aim in aims {
            let route = aimed(aim);
            let what = format!("{name} {}", aim.name());
            let gm = adapter(b, &t, small_groups(), &route)
                .split_large_generations()
                .geometric_mtf_job(&m, &JobContext::new())
                .unwrap();
            let gf = gpu::best_focus_job(b, &t, &f, &route, &JobContext::new()).unwrap();
            let geo = adapter(b, &t, small_groups(), &route);
            let gp = geo.footprint_job(&fp, &JobContext::new()).unwrap();
            let gg = geo.ray_grid_job(&sampling, &JobContext::new()).unwrap();
            drop(geo);
            assert_eq!(evidence(&gm.run_info), aim.name(), "{what}");
            assert_eq!(evidence(&gg.run_info), aim.name(), "{what}");
            same_mtf(&what, &am, &gm);
            close(
                &format!("{what} best focus"),
                &[af.defocus, af.merit],
                &[gf.defocus, gf.merit],
                |v| 1e-5 + 1e-3 * v.abs(),
            );
            assert_eq!(ap.rays.len(), gp.rays.len(), "{what}");
            for (a, g) in ap.rays.iter().zip(&gp.rays) {
                assert_eq!(
                    (a.points.len(), a.clipped.len(), a.lost_before),
                    (g.points.len(), g.clipped.len(), g.lost_before),
                    "{what} footprint"
                );
                for (a, g) in a.points.iter().zip(&g.points) {
                    close(&format!("{what} footprint"), a, g, |_| 2e-5);
                }
            }
            assert_eq!(ag.len(), gg.len(), "{what}");
            for (i, a) in ag.iter().enumerate() {
                match gg.hit(i) {
                    Some(h) => {
                        assert!(a.status.is_ok(), "{what} ray {i}");
                        close(&what, &[a.point.x, a.point.y], &h.point, |_| 2e-5);
                    }
                    None => assert!(!a.status.is_ok(), "{what} ray {i}"),
                }
            }
        }
    }
}

/// Large generations (at least 2,048 rays per field row, so with a seed
/// table) of the spot diagram and the geometric MTF split between host and
/// device: the same bits on a one-thread and a four-thread rayon pool, i.e.
/// whichever side is faster, and within the reference's tolerances; the
/// device share follows the route (0: host only) and is named in the run
/// evidence.
fn split_designs(b: &dyn ComputeBackend) {
    for (name, t) in tracers() {
        let s = SpotDiagramSettings {
            rings: 26,
            ..Default::default()
        };
        let m = GeometricMtfSettings {
            sampling: RaySampling {
                rings: 26,
                ..Default::default()
            },
            frequency_samples: 5,
            max_frequency: Some(100.0),
            defocus: 0.05,
        };
        let a = spot::spot_diagram(&t, &s).unwrap();
        let am = energy::geometric_mtf(&t, &m).unwrap();
        let run = |threads: usize, route: gpu::Route| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let geo = adapter(b, &t, small_tiles(), &route).split_large_generations();
                let g = geo.spot_diagram_job(&s, &JobContext::new()).unwrap();
                let gm = geo.geometric_mtf_job(&m, &JobContext::new()).unwrap();
                (g, gm)
            })
        };
        let device_rays = |g: &SpotDiagram| {
            let e = &g.run_info.settings["resident_execution"]["aiming"];
            (
                e["device_plans"].as_u64().unwrap(),
                e["device_rays"].as_u64().unwrap(),
            )
        };
        let share = |g: &SpotDiagram| {
            g.run_info.settings["resident_execution"]["aiming"]["device_share"]
                .as_f64()
                .unwrap()
        };
        let runs = [
            run(1, gpu::Route::default()),
            run(4, gpu::Route::default()),
            run(4, shared(0.75)),
            run(4, shared(0.0)),
        ];
        for (g, gm) in &runs {
            let what = format!("{name} split");
            same_spots(&what, &a, g);
            same_mtf(&what, &am, gm);
        }
        let (plans, rays) = device_rays(&runs[0].0);
        assert!(plans > 1 && rays > 0, "{name}: {plans} device plans");
        assert!(device_rays(&runs[2].0).1 > rays, "{name}: share 0.75");
        assert_eq!(device_rays(&runs[3].0), (0, 0), "{name}: share 0");
        assert_eq!(
            [&runs[0], &runs[2], &runs[3]].map(|r| share(&r.0)),
            [gpu::aim::SPLIT_DEVICE_SHARE, 0.75, 0.0],
            "{name}: shares in the evidence"
        );
        let bits = |(g, gm): &(SpotDiagram, GeometricMtfResult)| {
            serde_json::to_string(&(&g.fields, &gm.curves)).unwrap()
        };
        assert!(
            bits(&runs[0]) == bits(&runs[1]),
            "{name}: split results differ between runs"
        );
    }
}

#[test]
fn cpu_f32_designs_match_reference() {
    let b = CpuF32Backend::new();
    spots(&b, &AIMS, 3);
    energies(&b, &AIMS);
}

#[test]
fn cpu_f32_split_is_deterministic() {
    split_designs(&CpuF32Backend::new());
}

/// A device share outside [0, 1] is an error, not a clamped share.
#[test]
fn shares_outside_the_unit_interval_are_rejected() {
    let (_, t) = tracers().remove(0);
    let b = CpuF32Backend::new();
    for share in [1.5, -0.1, f64::NAN, f64::INFINITY] {
        let r = gpu::spot_diagram_job(
            &b,
            &t,
            &Default::default(),
            &shared(share),
            &JobContext::new(),
        );
        assert!(r.is_err(), "device share {share}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_designs_match_reference() {
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    spots(&b, &AIMS, 4);
    energies(&b, &AIMS);
    split_designs(&b);
}
