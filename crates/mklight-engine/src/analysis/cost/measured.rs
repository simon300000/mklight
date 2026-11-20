//! The `auto` decision of a GPU backend measured on the running machine
//! (GPU.md §11.6): the WebGPU backend and the CUDA backend. A per-device
//! calibration measured in the running engine, with a fixed break-even rule
//! of one machine as the prior until it exists.
//!
//! No shipped table can predict these devices: the browser's CPU is the
//! wasm reference on however many threads it grants and its GPU whatever
//! adapter it exposes; CUDA runs on any NVIDIA GPU beside any CPU. So the
//! engine measures them ([`Rules::runs`], `Engine::calibrate`): a few
//! analyses of a built-in sample at two sizes on the CPU reference and on
//! the device (after an untimed warm-up that pays kernel compilation), and
//! fits each analysis and backend a line `fixed_ms + per_mwork_ms · work`,
//! with `work` ([`mwork`]) the cost profile's rays × the system's surfaces
//! (plus the transform work of wave analyses) in millions. `auto` then
//! predicts both backends and takes the device when the CPU is at least
//! `RATIO` slower and the saving at least `MIN_GAIN_MS`, like the Metal
//! cost model. The web host caches the WebGPU calibration (keyed by adapter,
//! threads and engine version) and installs it at start
//! ([`super::CostModel::set_calibration`]); natively the engine caches the
//! CUDA calibration itself (`Engine::prepare_gpu`, [`DeviceCalibration::matches`]).

use std::collections::BTreeMap;

use mklight_core::OpticalSystem;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::profile::{SystemShape, profile_with};
use super::{Decision, MIN_GAIN_MS, RATIO, Route, VERSION};

/// Calibration format.
pub const FORMAT: u32 = 1;

/// The rules of one measured backend.
#[derive(Debug)]
pub struct Rules {
    /// Engine backend name (`gpu:<name>`), and its name in decisions.
    pub name: &'static str,
    /// Its name in reasons.
    pub label: &'static str,
    /// The prior: the ray count from which the device measured at least
    /// `RATIO` faster than the CPU reference, per analysis. Analyses not
    /// listed stay on the CPU until a calibration shows a gain.
    pub min_rays: &'static [(&'static str, f64)],
    /// The analyses calibrated (and routed by the calibration), by [`key`]:
    /// `analysis` or `analysis/method`.
    pub calibrated: &'static [&'static str],
    /// The calibration's sample system.
    pub sample: fn() -> OpticalSystem,
    /// Each calibrated analysis's settings at its two sizes.
    pub sizes: fn(&str) -> [Value; 2],
}

/// WebGPU (GPU.md §11.6): the prior is the ray count from which WebGPU
/// measured at least `RATIO` faster in Chromium and WebKit on one machine
/// (an Apple M2 Pro).
pub const WEBGPU: Rules = Rules {
    name: "webgpu",
    label: "WebGPU",
    min_rays: &[
        ("ray_grid", 250_000.),
        ("spot_diagram", 400_000.),
        ("geometric_mtf", 400_000.),
    ],
    calibrated: &["spot_diagram", "ray_grid", "geometric_mtf"],
    sample: mklight_core::samples::cooke_triplet,
    sizes: |_| [json!({"rings": 12}), json!({"rings": 40})],
};

/// CUDA: the prior is the smallest ray count at which CUDA measured at
/// least `RATIO` faster than the CPU reference on 24 threads (RTX 4090,
/// i9-12900KS; the Double Gauss after an edit): 128 rings of every field
/// and wavelength (446k rays) for the spot diagram (2.1×), the ray grid
/// (2.1×) and the geometric MTF (1.5×); at 32 rings (28k rays) the CPU
/// was as fast or faster. The wave analyses measured no consistent gain
/// and stay on the CPU unless the calibration shows one. The PSF is not
/// calibrated: CUDA measured within 10 % of the CPU at every pupil size,
/// and the sample's line sent 256² and 512² pupils of the Double Gauss to
/// CUDA, which were 10 % slower there.
pub const CUDA: Rules = Rules {
    name: "cuda",
    label: "CUDA",
    min_rays: &[
        ("ray_grid", 400_000.),
        ("spot_diagram", 400_000.),
        ("geometric_mtf", 400_000.),
    ],
    calibrated: &[
        "spot_diagram",
        "ray_grid",
        "geometric_mtf",
        "mtf",
        "mtf_vs_field",
        "through_focus_mtf",
        "mtf/fft",
    ],
    sample: mklight_core::samples::cooke_triplet,
    sizes: |key| {
        if key == "mtf/fft" {
            // The FFT's cost is the pupil grid's, not the traced rays'.
            [
                json!({"method": "fft", "pupil": {"grid": 128}}),
                json!({"method": "fft", "pupil": {"grid": 512}}),
            ]
        } else if WAVE.contains(&key) {
            [
                json!({"pupil": {"rays": 33}}),
                json!({"pupil": {"rays": 65}}),
            ]
        } else {
            [json!({"rings": 24}), json!({"rings": 96})]
        }
    },
};

/// The measured backends.
pub const ALL: [&Rules; 2] = [&WEBGPU, &CUDA];

/// Wave analyses: their work includes the transforms, and a calibrated
/// line applies to the default method only ([`key`]).
const WAVE: &[&str] = &["psf", "mtf", "mtf_vs_field", "through_focus_mtf"];

/// The analysis of a calibration key (`mtf/fft` → `mtf`).
pub fn analysis_of(key: &str) -> &str {
    key.split_once('/').map_or(key, |(a, _)| a)
}

impl Rules {
    /// The calibration runs: `(analysis, settings)` at two sizes each.
    pub fn runs(&self) -> Vec<(&'static str, Value)> {
        self.calibrated
            .iter()
            .flat_map(|a| (self.sizes)(a).map(|s| (*a, s)))
            .collect()
    }

    /// The engine backend id (`gpu:cuda`).
    pub fn backend_id(&self) -> String {
        format!("gpu:{}", self.name)
    }
}

/// One backend's cost of one analysis (ms).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub fixed_ms: f64,
    /// Per million ray × surface.
    pub per_mwork_ms: f64,
}

impl Line {
    pub fn at(&self, mwork: f64) -> f64 {
        self.fixed_ms + self.per_mwork_ms * mwork
    }

    /// The line through two measurements (non-negative coefficients).
    pub fn fit(a: (f64, f64), b: (f64, f64)) -> Self {
        let ((w1, t1), (w2, t2)) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let per = if w2 > w1 {
            ((t2 - t1) / (w2 - w1)).max(0.)
        } else {
            0.
        };
        Line {
            fixed_ms: (t1 - per * w1).max(0.),
            per_mwork_ms: per,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub cpu: Line,
    /// The device's line (`webgpu` in the format, which WebGPU defined).
    #[serde(rename = "webgpu")]
    pub gpu: Line,
}

/// One timed run of the calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub analysis: String,
    pub backend: String,
    /// Rays × surfaces, millions.
    pub mwork: f64,
    pub ms: f64,
}

/// A device's calibration: what it was measured on and the fitted lines.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceCalibration {
    pub format: u32,
    pub engine_version: String,
    /// The device's name (WebGPU: the adapter).
    pub adapter: String,
    /// The device's driver, where the engine knows it (CUDA).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub driver: String,
    /// CPU threads of the reference (rayon's pool).
    pub threads: usize,
    pub analyses: BTreeMap<String, Pair>,
    pub points: Vec<Point>,
    /// Wall time of the calibration (s).
    pub seconds: f64,
    /// Where it came from: `measured` here, or `cached` / `forced` by the host.
    #[serde(default)]
    pub source: String,
}

/// The WebGPU calibration's name in the web host's interface.
pub type WebGpuCalibration = DeviceCalibration;

impl DeviceCalibration {
    /// A cached calibration still applies: same format, engine version,
    /// device, driver and thread count, and it covers the analyses `rules`
    /// calibrate now.
    pub fn matches(&self, rules: &Rules, adapter: &str, driver: &str, threads: usize) -> bool {
        self.format == FORMAT
            && self.engine_version == mklight_core::ENGINE_VERSION
            && self.adapter == adapter
            && self.driver == driver
            && self.threads == threads
            && self.analyses.keys().eq(rules
                .calibrated
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>())
    }

    /// The calibration cached at `path` if it still applies
    /// ([`Self::matches`]), marked `cached`.
    pub fn load(
        path: &std::path::Path,
        rules: &Rules,
        adapter: &str,
        driver: &str,
        threads: usize,
    ) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut c: Self = serde_json::from_str(&text).ok()?;
        c.matches(rules, adapter, driver, threads).then(|| {
            c.source = "cached".into();
            c
        })
    }

    /// Write it to `path` (creating the directory), atomically.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }
}

/// The calibration key of a request: the analysis, or `analysis/method`
/// for a wave analysis with a method other than its default (a line
/// measured with one transform says nothing about another).
pub fn key<'a>(analysis: &'a str, settings: &Value) -> std::borrow::Cow<'a, str> {
    let default = if analysis == "psf" { "fft" } else { "direct" };
    match settings["method"].as_str() {
        Some(m) if WAVE.contains(&analysis) && m != default => format!("{analysis}/{m}").into(),
        _ => analysis.into(),
    }
}

/// Work of a request in millions of ray × surface (plus the transform
/// work of a wave analysis), from the cost profile.
pub fn mwork(analysis: &str, settings: &Value, system: &OpticalSystem) -> Option<f64> {
    let p = profile_with(analysis, settings, system, None)?;
    let transform = if WAVE.contains(&analysis) {
        p.transform
    } else {
        0.
    };
    Some((p.rays * SystemShape::of(system).surfaces + transform) / 1e6)
}

/// The decision for `rules`' device: by the calibration when it covers the
/// analysis, else by the prior.
pub(crate) fn decide(
    rules: &Rules,
    analysis: &str,
    settings: &Value,
    system: &OpticalSystem,
    calibration: Option<&DeviceCalibration>,
) -> Decision {
    let name = rules.name;
    let qualified = format!("qualified_{name}");
    let work = mwork(analysis, settings, system);
    if let (Some(cal), Some(w)) = (calibration, work)
        && let Some(pair) = cal.analyses.get(key(analysis, settings).as_ref())
    {
        let (cpu, gpu) = (pair.cpu.at(w), pair.gpu.at(w));
        let native = cpu >= RATIO * gpu && cpu - gpu >= MIN_GAIN_MS;
        let mut c = json!({
            "source": cal.source,
            "adapter": cal.adapter,
            "threads": cal.threads,
            "engine_version": cal.engine_version,
            "cpu": pair.cpu,
            name: pair.gpu,
        });
        if !cal.driver.is_empty() {
            c["driver"] = json!(cal.driver);
        }
        return Decision {
            route: if native { Route::Metal } else { Route::Cpu },
            json: json!({
                "decision": if native { qualified.as_str() } else { "cpu_reference" },
                "rule": format!("{name} calibration"),
                "mwork": w,
                "predicted_ms": {"cpu": cpu, name: gpu},
                "required": {"ratio": RATIO, "gain_ms": MIN_GAIN_MS},
                "calibration": c,
                "version": VERSION,
            }),
        };
    }
    let rays = profile_with(analysis, settings, system, None).map(|p| p.rays);
    let min = rules
        .min_rays
        .iter()
        .find(|(a, _)| *a == analysis)
        .map(|(_, r)| *r);
    let native = matches!((rays, min), (Some(r), Some(m)) if r >= m);
    let label = rules.label;
    Decision {
        route: if native { Route::Metal } else { Route::Cpu },
        json: json!({
            "decision": if native { qualified.as_str() } else { "cpu_reference" },
            "rule": format!("{name} break-even"),
            "rays": rays,
            "min_rays": min,
            "calibration": calibration.map(|c| c.source.clone()).unwrap_or_else(|| "none yet (prior)".into()),
            "reason": if min.is_none() { format!("no measured {label} gain for this analysis") } else if native { "above the measured break-even".into() } else { "below the measured break-even".into() },
            "version": VERSION,
        }),
    }
}

/// Fit the calibration from its timed points (the median of each
/// analysis/backend/size).
pub fn fit(rules: &Rules, points: &[Point]) -> BTreeMap<String, Pair> {
    let mut out = BTreeMap::new();
    for a in rules.calibrated {
        let line = |backend: &str| {
            let mut by_size: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
            for p in points
                .iter()
                .filter(|p| p.analysis == *a && p.backend == backend)
            {
                by_size.entry(p.mwork.to_bits()).or_default().push(p.ms);
            }
            let pts: Vec<(f64, f64)> = by_size
                .into_iter()
                .map(|(w, mut t)| {
                    t.sort_by(f64::total_cmp);
                    (f64::from_bits(w), t[t.len() / 2])
                })
                .collect();
            (pts.len() >= 2).then(|| Line::fit(pts[0], pts[pts.len() - 1]))
        };
        if let (Some(cpu), Some(gpu)) = (line("cpu"), line(rules.name)) {
            out.insert(a.to_string(), Pair { cpu, gpu });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cal(analysis: &str, cpu: Line, gpu: Line) -> DeviceCalibration {
        DeviceCalibration {
            format: FORMAT,
            engine_version: "t".into(),
            adapter: "test".into(),
            driver: String::new(),
            threads: 4,
            analyses: [(analysis.to_string(), Pair { cpu, gpu })].into(),
            points: vec![],
            seconds: 0.,
            source: "forced".into(),
        }
    }

    fn line(fixed_ms: f64, per_mwork_ms: f64) -> Line {
        Line {
            fixed_ms,
            per_mwork_ms,
        }
    }

    #[test]
    fn lines_fit_and_decide() {
        let l = Line::fit((1., 12.), (3., 32.));
        assert_eq!((l.fixed_ms, l.per_mwork_ms), (2., 10.));
        // A falling pair has no slope.
        let f = Line::fit((1., 30.), (3., 20.));
        assert_eq!(f.per_mwork_ms, 0.);
        let dg = mklight_core::samples::double_gauss();
        let fast = cal("ray_grid", line(20., 100.), line(0.5, 1.));
        let slow = cal("ray_grid", line(1., 100.), line(50., 1000.));
        let small = json!({"rings": 8});
        let d =
            |c: Option<&DeviceCalibration>, s: &Value| decide(&WEBGPU, "ray_grid", s, &dg, c).route;
        // The prior keeps a small grid on the CPU; a fast GPU estimate moves
        // it, a slow one keeps even a large grid on the CPU.
        assert_eq!(d(None, &small), Route::Cpu);
        assert_eq!(d(Some(&fast), &small), Route::Metal);
        assert_eq!(d(Some(&slow), &json!({"rings": 250})), Route::Cpu);
        // Uncalibrated analyses keep the prior.
        assert_eq!(
            decide(
                &WEBGPU,
                "spot_diagram",
                &json!({"rings": 250}),
                &dg,
                Some(&fast)
            )
            .json["rule"],
            "webgpu break-even"
        );
    }

    /// The WebGPU decision's record is the one the web host has always
    /// read.
    #[test]
    fn webgpu_records_keep_their_shape() {
        let dg = mklight_core::samples::double_gauss();
        let fast = cal("ray_grid", line(20., 100.), line(0.5, 1.));
        let d = decide(&WEBGPU, "ray_grid", &json!({"rings": 8}), &dg, Some(&fast)).json;
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(d["decision"], "qualified_webgpu");
        assert_eq!(d["rule"], "webgpu calibration");
        assert_eq!(keys(&d["predicted_ms"]), ["cpu", "webgpu"]);
        assert_eq!(
            keys(&d["calibration"]),
            [
                "adapter",
                "cpu",
                "engine_version",
                "source",
                "threads",
                "webgpu"
            ]
        );
        let p = decide(&WEBGPU, "mtf", &json!({}), &dg, None).json;
        assert_eq!(p["reason"], "no measured WebGPU gain for this analysis");
        assert_eq!(p["calibration"], "none yet (prior)");
        // The format names the device's line `webgpu`, without a driver.
        let text = serde_json::to_value(&fast).unwrap();
        assert!(text.get("driver").is_none());
        assert_eq!(text["analyses"]["ray_grid"]["webgpu"]["fixed_ms"], 0.5);
    }

    /// The CUDA prior keeps small geometric requests and every wave
    /// analysis on the CPU.
    #[test]
    fn cuda_prior_keeps_small_and_wave_requests_on_the_cpu() {
        let dg = mklight_core::samples::double_gauss();
        let d = |a: &str, s: Value| decide(&CUDA, a, &s, &dg, None);
        for a in ["spot_diagram", "ray_grid", "geometric_mtf"] {
            for rings in [6, 32, 64] {
                let r = d(a, json!({"rings": rings}));
                assert_eq!(r.route, Route::Cpu, "{a} {rings}: {}", r.json);
                assert_eq!(r.json["reason"], "below the measured break-even");
            }
            for rings in [128, 250] {
                let r = d(a, json!({"rings": rings}));
                assert_eq!(r.route, Route::Metal, "{a} {rings}: {}", r.json);
                assert_eq!(r.json["decision"], "qualified_cuda");
                assert_eq!(r.json["rule"], "cuda break-even");
            }
        }
        for a in WAVE {
            let r = d(a, json!({}));
            assert_eq!(r.route, Route::Cpu);
            assert_eq!(r.json["reason"], "no measured CUDA gain for this analysis");
        }
        // The analyses the rules list are the backend's analyses.
        for rules in ALL {
            for (a, _) in rules.min_rays {
                assert!(rules.calibrated.contains(a));
            }
            for a in rules.calibrated {
                assert!(crate::analysis::gpu::WEBGPU_ANALYSES.contains(&analysis_of(a)));
            }
        }
    }

    /// A CUDA calibration routes by its lines, a wave line only for the
    /// method it was measured with, and records the driver.
    #[test]
    fn cuda_calibration_decides_by_its_lines() {
        let dg = mklight_core::samples::double_gauss();
        // A wave analysis the calibration measured faster on the device.
        let mut c = cal("through_focus_mtf", line(20., 5.), line(10., 2.));
        c.driver = "CUDA driver 13.0, sm_89".into();
        let mut spot = cal("spot_diagram", line(1., 50.), line(6., 20.));
        c.analyses.append(&mut spot.analyses);
        let d = |a: &str, s: Value| decide(&CUDA, a, &s, &dg, Some(&c));
        let tf = d("through_focus_mtf", json!({}));
        assert_eq!(tf.route, Route::Metal, "{}", tf.json);
        assert_eq!(tf.json["rule"], "cuda calibration");
        assert_eq!(tf.json["calibration"]["driver"], "CUDA driver 13.0, sm_89");
        assert!(tf.json["predicted_ms"]["cuda"].as_f64().unwrap() > 0.);
        // Another method is not what was measured: the prior.
        let fft = d("through_focus_mtf", json!({"method": "fft"}));
        assert_eq!(fft.route, Route::Cpu);
        assert_eq!(fft.json["rule"], "cuda break-even");
        assert_eq!(key("mtf", &json!({"method": "direct"})), "mtf");
        assert_eq!(key("psf", &json!({"method": "fft"})), "psf");
        assert_eq!(key("psf", &json!({"method": "huygens"})), "psf/huygens");
        assert_eq!(key("spot_diagram", &json!({"method": "x"})), "spot_diagram");
        // The geometric line: small on the CPU (fixed cost), large on CUDA.
        assert_eq!(d("spot_diagram", json!({"rings": 6})).route, Route::Cpu);
        assert_eq!(d("spot_diagram", json!({"rings": 250})).route, Route::Metal);
        // Uncalibrated analyses keep the prior.
        assert_eq!(d("mtf", json!({})).json["rule"], "cuda break-even");
    }

    #[test]
    fn cached_calibrations_round_trip_and_expire() {
        let dir = std::env::temp_dir().join(format!("mklight-cost-{}", std::process::id()));
        let path = dir.join("cost").join("cuda.json");
        let mut c = cal("spot_diagram", line(1., 50.), line(6., 20.));
        for a in CUDA.calibrated {
            c.analyses.insert(a.to_string(), c.analyses["spot_diagram"]);
        }
        c.engine_version = mklight_core::ENGINE_VERSION.into();
        c.adapter = "GPU".into();
        c.driver = "driver 1".into();
        c.source = "measured".into();
        c.save(&path).unwrap();
        let load = |adapter, driver, threads| {
            DeviceCalibration::load(&path, &CUDA, adapter, driver, threads)
        };
        let back = load("GPU", "driver 1", 4).expect("the same device loads");
        assert_eq!(back.source, "cached");
        assert_eq!(back.analyses, c.analyses);
        // Another device, driver or thread count measures again.
        assert!(load("other", "driver 1", 4).is_none());
        assert!(load("GPU", "driver 2", 4).is_none());
        assert!(load("GPU", "driver 1", 8).is_none());
        // So does another engine version, format or analysis set.
        for change in [
            |c: &mut DeviceCalibration| c.engine_version = "0.0.0-other".into(),
            |c: &mut DeviceCalibration| c.format = FORMAT + 1,
            |c: &mut DeviceCalibration| {
                c.analyses.remove("mtf");
            },
        ] {
            let mut stale = c.clone();
            change(&mut stale);
            stale.save(&path).unwrap();
            assert!(load("GPU", "driver 1", 4).is_none());
        }
        // An unreadable file is no calibration.
        std::fs::write(&path, "{not json").unwrap();
        assert!(load("GPU", "driver 1", 4).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn calibration_runs_fit_every_calibrated_analysis() {
        let sys = (CUDA.sample)();
        let mut points = Vec::new();
        for (a, s) in CUDA.runs() {
            let w = mwork(analysis_of(a), &s, &sys).unwrap();
            assert!(w > 0., "{a}");
            for (backend, fixed, per) in [("cpu", 1., 40.), ("cuda", 6., 10.)] {
                for jitter in [0.9, 1., 1.2] {
                    points.push(Point {
                        analysis: a.into(),
                        backend: backend.into(),
                        mwork: w,
                        ms: (fixed + per * w) * jitter,
                    });
                }
            }
        }
        let fitted = fit(&CUDA, &points);
        assert_eq!(fitted.len(), CUDA.calibrated.len());
        for p in fitted.values() {
            // The medians lie on the generating lines.
            assert!((p.cpu.per_mwork_ms - 40.).abs() < 1e-6, "{p:?}");
            assert!((p.gpu.fixed_ms - 6.).abs() < 1e-6, "{p:?}");
        }
    }
}
