//! The WebGPU `auto` decision (GPU.md §11.6): a per-device calibration
//! measured in the running engine, with the fixed break-even rule of one
//! machine as the prior until it exists.
//!
//! The browser is the WebGPU backend's main use, and there the CPU is the
//! wasm reference on however many threads the browser grants while the GPU
//! is whatever adapter it exposes: no shipped table can predict either. So
//! the engine measures them ([`calibrate`]): the spot diagram, the ray grid
//! and the geometric MTF of a built-in sample at two small sizes on the CPU reference and on
//! `gpu:webgpu` (after an untimed warm-up that pays pipeline compilation),
//! about 0.3–1 s in all, and fits each analysis and backend a line
//! `fixed_ms + per_mwork_ms · work`, with `work` the cost profile's rays ×
//! the system's surfaces in millions. `auto` then predicts both backends and
//! takes WebGPU when the CPU is at least `RATIO` slower and the saving at
//! least `MIN_GAIN_MS`, like the Metal cost model. The web host caches the
//! calibration (keyed by adapter, threads and engine version) and installs
//! it at start ([`super::CostModel::set_webgpu_calibration`]).

use std::collections::BTreeMap;

use mklight_core::OpticalSystem;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::profile::{SystemShape, profile_with};
use super::{Decision, MIN_GAIN_MS, RATIO, Route, VERSION};

/// The analyses calibrated (and routed by the calibration).
pub const CALIBRATED: &[&str] = &["spot_diagram", "ray_grid", "geometric_mtf"];

/// Calibration format.
pub const FORMAT: u32 = 1;

/// The prior (GPU.md §11.6): the ray count from which WebGPU measured at
/// least `RATIO` faster in Chromium and WebKit on one machine (an Apple M2
/// Pro), used until a calibration is installed.
pub(crate) const WEBGPU_MIN_RAYS: &[(&str, f64)] = &[
    ("ray_grid", 250_000.),
    ("spot_diagram", 400_000.),
    ("geometric_mtf", 400_000.),
];

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
    pub webgpu: Line,
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
pub struct WebGpuCalibration {
    pub format: u32,
    pub engine_version: String,
    /// WebGPU adapter (`DeviceInfo::backend_id`).
    pub adapter: String,
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

/// Work of a request in millions of ray × surface, from the cost profile.
pub fn mwork(analysis: &str, settings: &Value, system: &OpticalSystem) -> Option<f64> {
    let p = profile_with(analysis, settings, system, None)?;
    Some(p.rays * SystemShape::of(system).surfaces / 1e6)
}

/// The WebGPU decision: by the calibration when it covers the analysis,
/// else by the prior.
pub(crate) fn decide(
    analysis: &str,
    settings: &Value,
    system: &OpticalSystem,
    calibration: Option<&WebGpuCalibration>,
) -> Decision {
    let work = mwork(analysis, settings, system);
    if let (Some(cal), Some(w)) = (calibration, work)
        && let Some(pair) = cal.analyses.get(analysis)
    {
        let (cpu, gpu) = (pair.cpu.at(w), pair.webgpu.at(w));
        let native = cpu >= RATIO * gpu && cpu - gpu >= MIN_GAIN_MS;
        return Decision {
            route: if native { Route::Metal } else { Route::Cpu },
            json: json!({
                "decision": if native { "qualified_webgpu" } else { "cpu_reference" },
                "rule": "webgpu calibration",
                "mwork": w,
                "predicted_ms": {"cpu": cpu, "webgpu": gpu},
                "required": {"ratio": RATIO, "gain_ms": MIN_GAIN_MS},
                "calibration": {
                    "source": cal.source,
                    "adapter": cal.adapter,
                    "threads": cal.threads,
                    "engine_version": cal.engine_version,
                    "cpu": pair.cpu,
                    "webgpu": pair.webgpu,
                },
                "version": VERSION,
            }),
        };
    }
    let rays = profile_with(analysis, settings, system, None).map(|p| p.rays);
    let min = WEBGPU_MIN_RAYS
        .iter()
        .find(|(a, _)| *a == analysis)
        .map(|(_, r)| *r);
    let native = matches!((rays, min), (Some(r), Some(m)) if r >= m);
    Decision {
        route: if native { Route::Metal } else { Route::Cpu },
        json: json!({
            "decision": if native { "qualified_webgpu" } else { "cpu_reference" },
            "rule": "webgpu break-even",
            "rays": rays,
            "min_rays": min,
            "calibration": calibration.map(|c| c.source.clone()).unwrap_or_else(|| "none yet (prior)".into()),
            "reason": if min.is_none() { "no measured WebGPU gain for this analysis" } else if native { "above the measured break-even" } else { "below the measured break-even" },
            "version": VERSION,
        }),
    }
}

/// The calibration runs: `(analysis, settings)` at two sizes each.
pub fn runs() -> Vec<(&'static str, Value)> {
    let mut v = Vec::new();
    for a in CALIBRATED {
        for rings in [12, 40] {
            v.push((*a, json!({"rings": rings})));
        }
    }
    v
}

/// Fit the calibration from its timed points (the median of each
/// analysis/backend/size).
pub fn fit(points: &[Point]) -> BTreeMap<String, Pair> {
    let mut out = BTreeMap::new();
    for a in CALIBRATED {
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
        if let (Some(cpu), Some(webgpu)) = (line("cpu"), line("webgpu")) {
            out.insert(a.to_string(), Pair { cpu, webgpu });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_fit_and_decide() {
        let l = Line::fit((1., 12.), (3., 32.));
        assert_eq!((l.fixed_ms, l.per_mwork_ms), (2., 10.));
        // A falling pair has no slope.
        let f = Line::fit((1., 30.), (3., 20.));
        assert_eq!(f.per_mwork_ms, 0.);
        let dg = mklight_core::samples::double_gauss();
        let cal = |cpu: Line, webgpu: Line| WebGpuCalibration {
            format: FORMAT,
            engine_version: "t".into(),
            adapter: "test".into(),
            threads: 4,
            analyses: [("ray_grid".to_string(), Pair { cpu, webgpu })].into(),
            points: vec![],
            seconds: 0.,
            source: "forced".into(),
        };
        let fast = cal(
            Line {
                fixed_ms: 20.,
                per_mwork_ms: 100.,
            },
            Line {
                fixed_ms: 0.5,
                per_mwork_ms: 1.,
            },
        );
        let slow = cal(
            Line {
                fixed_ms: 1.,
                per_mwork_ms: 100.,
            },
            Line {
                fixed_ms: 50.,
                per_mwork_ms: 1000.,
            },
        );
        let small = json!({"rings": 8});
        let d = |c: Option<&WebGpuCalibration>, s: &Value| decide("ray_grid", s, &dg, c).route;
        // The prior keeps a small grid on the CPU; a fast GPU estimate moves
        // it, a slow one keeps even a large grid on the CPU.
        assert_eq!(d(None, &small), Route::Cpu);
        assert_eq!(d(Some(&fast), &small), Route::Metal);
        assert_eq!(d(Some(&slow), &json!({"rings": 250})), Route::Cpu);
        // Uncalibrated analyses keep the prior.
        assert_eq!(
            decide("spot_diagram", &json!({"rings": 250}), &dg, Some(&fast)).json["rule"],
            "webgpu break-even"
        );
    }
}
