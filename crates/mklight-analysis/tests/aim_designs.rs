//! The aiming designs (work-split key `aiming`) and the divisions of the
//! geometric analyses (key `geometric`) agree with the CPU reference on the
//! built-in samples with real aiming, and name themselves in the run
//! evidence. Selections are process-wide, so every check holds one lock.
use mklight_analysis::{
    energy::{self, BestFocusSettings, GeometricMtfSettings, RaySampling},
    gpu,
    illumination::{self, FootprintSettings},
    spot::{self, SpotDiagramSettings},
    tracer_for,
};
use mklight_core::{JobContext, RayAiming, samples, split};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_trace::{Grid, SequentialTracer};
use std::sync::Mutex;

static SELECTION: Mutex<()> = Mutex::new(());

const AIMS: [&str; 5] = ["host", "host-table", "device", "table", "split"];
const DIVISIONS: [&str; 3] = ["fused", "pipelined", "host-reduce"];

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

/// The design and division named in a run's evidence.
fn evidence(run: &mklight_core::RunInfo) -> (String, String) {
    let e = &run.settings["resident_execution"];
    (
        e["aiming"]["design"].as_str().unwrap_or("").to_string(),
        e["geometric"]["division"]
            .as_str()
            .unwrap_or("")
            .to_string(),
    )
}

fn spots(b: &dyn ComputeBackend, aims: &[&str], rings: usize) {
    for (name, t) in tracers() {
        let s = SpotDiagramSettings {
            rings,
            ..Default::default()
        };
        let a = spot::spot_diagram(&t, &s).unwrap();
        for &aim in aims {
            for division in DIVISIONS {
                split::set("aiming", Some(aim));
                split::set("geometric", Some(division));
                let g = gpu::spot_diagram_job(b, &t, &s, &JobContext::new()).unwrap();
                split::reset();
                let what = format!("{name} {aim}/{division}");
                assert_eq!(
                    evidence(&g.run_info),
                    (aim.into(), division.into()),
                    "{what}"
                );
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
                    close(&what, &a.chief, &g.chief, |_| 1e-6);
                    close(
                        &what,
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
                        close(&what, &[a.x, a.y], &[g.x, g.y], |_| 2e-5);
                    }
                }
            }
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
            for division in DIVISIONS {
                split::set("aiming", Some(aim));
                split::set("geometric", Some(division));
                let what = format!("{name} {aim}/{division}");
                let gm = gpu::geometric_mtf_job(b, &t, &m, &JobContext::new()).unwrap();
                let gf = gpu::best_focus_job(b, &t, &f, &JobContext::new()).unwrap();
                let gp = gpu::footprint_job(b, &t, &fp, &JobContext::new()).unwrap();
                let gg = gpu::ray_grid_job(b, &t, &sampling, &JobContext::new()).unwrap();
                split::reset();
                assert_eq!(
                    evidence(&gm.run_info),
                    (aim.into(), division.into()),
                    "{what}"
                );
                assert_eq!(
                    evidence(&gg.run_info),
                    (aim.into(), division.into()),
                    "{what}"
                );
                for (a, g) in am.curves.iter().zip(&gm.curves) {
                    assert_eq!((a.field, a.orientation), (g.field, g.orientation), "{what}");
                    close(&format!("{what} MTF"), &a.modulation, &g.modulation, |_| {
                        2e-3
                    });
                }
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
}

/// Small thresholds so that the pipelined division forms several plans and
/// the split design hands the device tiles even on these grids.
fn small_thresholds() {
    gpu::division::set_pipeline_min_rays(Some(64));
    gpu::aim::set_split_thresholds(Some((256, 128)));
}
fn default_thresholds() {
    gpu::division::set_pipeline_min_rays(None);
    gpu::aim::set_split_thresholds(None);
}

#[test]
fn cpu_f32_designs_match_reference() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let b = CpuF32Backend::new();
    small_thresholds();
    spots(&b, &AIMS, 3);
    energies(&b, &["host", "device", "split"]);
    default_thresholds();
}

#[test]
fn unknown_designs_are_rejected() {
    let _g = SELECTION.lock().unwrap_or_else(|e| e.into_inner());
    let (_, t) = tracers().remove(0);
    let b = CpuF32Backend::new();
    for (key, name) in [("aiming", "nonsense"), ("geometric", "nonsense")] {
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
}
