//! The aiming designs (work-split key `aiming`) of the geometric analyses
//! agree with the CPU reference on the built-in samples with real aiming,
//! in field groups of several plans, and name themselves in the run
//! evidence; large generations split between host and device give the same
//! rays however the work is timed. Selections are process-wide, so every
//! check holds one lock.
use mklight_analysis::{
    energy::{self, BestFocusSettings, GeometricMtfResult, GeometricMtfSettings, RaySampling},
    gpu,
    illumination::{self, FootprintSettings},
    spot::{self, SpotDiagram, SpotDiagramSettings},
    tracer_for,
};
use mklight_core::{JobContext, RayAiming, samples, split};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_trace::{Grid, SequentialTracer};
use std::sync::Mutex;

static SELECTION: Mutex<()> = Mutex::new(());

const AIMS: [&str; 3] = ["host-table", "table", "device"];

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

fn spots(b: &dyn ComputeBackend, aims: &[&str], rings: usize) {
    for (name, t) in tracers() {
        let s = SpotDiagramSettings {
            rings,
            ..Default::default()
        };
        let a = spot::spot_diagram(&t, &s).unwrap();
        for &aim in aims {
            split::set("aiming", Some(aim));
            let g = gpu::spot_diagram_job(b, &t, &s, &JobContext::new()).unwrap();
            split::reset();
            let what = format!("{name} {aim}");
            assert_eq!(evidence(&g.run_info), aim, "{what}");
            same_spots(&what, &a, &g);
        }
    }
}

fn energies(b: &dyn ComputeBackend, aims: &[&str]) {
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
            split::set("aiming", Some(aim));
            let what = format!("{name} {aim}");
            let gm = gpu::geometric_mtf_job(b, &t, &m, &JobContext::new()).unwrap();
            let gf = gpu::best_focus_job(b, &t, &f, &JobContext::new()).unwrap();
            let gp = gpu::footprint_job(b, &t, &fp, &JobContext::new()).unwrap();
            let gg = gpu::ray_grid_job(b, &t, &sampling, &JobContext::new()).unwrap();
            split::reset();
            assert_eq!(evidence(&gm.run_info), aim, "{what}");
            assert_eq!(evidence(&gg.run_info), aim, "{what}");
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
/// table) split between host and device: the same bits on a one-thread and
/// a four-thread rayon pool, i.e. whichever side is faster, and within the
/// reference's tolerances.
fn split_designs(b: &dyn ComputeBackend) {
    gpu::aim::set_split_thresholds(Some((0, 512)));
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
        let mut runs = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            runs.push(pool.install(|| {
                let g = gpu::spot_diagram_job(b, &t, &s, &JobContext::new()).unwrap();
                let gm = gpu::geometric_mtf_job(b, &t, &m, &JobContext::new()).unwrap();
                (g, gm)
            }));
        }
        for (g, gm) in &runs {
            let what = format!("{name} split");
            let e = &g.run_info.settings["resident_execution"]["aiming"];
            assert!(e["device_plans"].as_u64().unwrap() > 1, "{what}: {e}");
            assert!(e["host_rays"].as_u64().unwrap() > 0, "{what}: {e}");
            same_spots(&what, &a, g);
            same_mtf(&what, &am, gm);
        }
        let bits = |(g, gm): &(SpotDiagram, GeometricMtfResult)| {
            serde_json::to_string(&(&g.fields, &gm.curves)).unwrap()
        };
        assert!(
            bits(&runs[0]) == bits(&runs[1]),
            "{name}: split results differ between runs"
        );
    }
    gpu::aim::set_split_thresholds(None);
}

/// Small thresholds so that field groups form several plans.
fn small_thresholds() {
    gpu::plans::set_group_min_rays(Some(64));
}
fn default_thresholds() {
    gpu::plans::set_group_min_rays(None);
}

#[test]
fn cpu_f32_designs_match_reference() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let b = CpuF32Backend::new();
    small_thresholds();
    spots(&b, &AIMS, 3);
    energies(&b, &AIMS);
    default_thresholds();
}

#[test]
fn cpu_f32_split_is_deterministic() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    split_designs(&CpuF32Backend::new());
}

#[test]
fn unknown_designs_are_rejected() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let (_, t) = tracers().remove(0);
    let b = CpuF32Backend::new();
    for (key, name) in [("aiming", "nonsense"), ("aiming_share", "1.5")] {
        split::set(key, Some(name));
        let r = gpu::spot_diagram_job(&b, &t, &Default::default(), &JobContext::new());
        split::reset();
        assert!(r.is_err(), "{key}={name}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_designs_match_reference() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("host Metal required");
    small_thresholds();
    spots(&b, &AIMS, 4);
    energies(&b, &AIMS);
    default_thresholds();
    split_designs(&b);
}
