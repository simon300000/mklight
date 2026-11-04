//! The calibration table and its fit (GPU.md §9.2 "Cost model").
//!
//! A backend's predicted time for one analysis is a sum of stage terms,
//! `fixed + prep·f_prep + ray·f_ray + transform·f_transform +
//! output·f_output` (`profile::Profile::features`), with one non-negative
//! coefficient set per analysis, backend and resident state. Coefficients
//! are fitted from measured complete calls (`tests/cost_calibration.rs`)
//! by relative least squares, so a 1 ms and a 10 s call weigh alike.
//! Nothing here is a universal GPU overhead: every analysis has its own
//! measured fixed cost on each backend.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stage features, in this order: constant (1), host preparation
/// (field–wavelength rows × surfaces), ray work (rays × (surfaces + aim
/// traces × stop surfaces)), transforms (samples × log₂), output bytes,
/// and the device seed-table build of large aimed generations (rows ×
/// table nodes × aim traces × stop surfaces; Metal only, cached while the
/// snapshot stays resident).
pub const FEATURES: [&str; 6] = ["fixed", "prep", "ray", "transform", "output", "table"];
pub type Features = [f64; 6];

/// Format of the stored table; bump when a field changes meaning.
pub const FORMAT: u32 = 2;

/// One fitted coefficient set (ms per feature unit).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub coef: Features,
    /// Measured calls the fit used.
    pub points: usize,
    /// Root-mean-square of ln(predicted / measured) over those calls.
    pub rms_log_error: f64,
}

impl Fit {
    /// Stage contributions in ms (same order as [`FEATURES`]).
    pub fn stages(&self, f: &Features) -> Features {
        std::array::from_fn(|k| self.coef[k] * f[k])
    }
}

/// Calibration of one analysis on one device class.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnalysisCalibration {
    /// CPU f64 reference with cold intermediates (graph disabled), on a
    /// snapshot the session already resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<Fit>,
    /// The same on a freshly edited snapshot (resolution, solves and host
    /// preparation of the new content).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_edit: Option<Fit>,
    /// Native, repeated on a resident snapshot (device caches warm).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metal: Option<Fit>,
    /// Native on a freshly edited snapshot (device caches cold).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metal_edit: Option<Fit>,
    /// Extra time of the first native call of this analysis in a process
    /// (pipeline creation, first allocations), above `metal_edit`.
    #[serde(default)]
    pub first_use_ms: f64,
}

/// A stored, versioned calibration table for one device class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub format: u32,
    /// `cost::VERSION` of the features the table was fitted with.
    pub model_version: u32,
    /// Device name the table was measured on (`DeviceInfo::name`).
    pub device_class: String,
    /// Date, source revision, host and load of the measurement.
    #[serde(default)]
    pub measured: Value,
    /// Rayon workers of the measuring process; CPU ray and transform
    /// stages scale with the ratio to the current pool.
    pub cpu_threads: usize,
    /// Opening the device (library load and numerics probe).
    pub device_open_ms: f64,
    pub analyses: BTreeMap<String, AnalysisCalibration>,
}

impl Calibration {
    /// The built-in table of the measured device class.
    /// (`builtin_table_matches_this_model` keeps it current; an unreadable
    /// table would leave every analysis uncalibrated, i.e. qualified Metal.)
    pub fn builtin() -> Self {
        serde_json::from_str(include_str!("apple-m2-pro.json")).unwrap_or_else(|e| {
            eprintln!("mklight: built-in cost calibration unreadable ({e})");
            Calibration {
                format: FORMAT,
                model_version: super::VERSION,
                device_class: String::new(),
                measured: Value::Null,
                cpu_threads: 1,
                device_open_ms: 0.,
                analyses: BTreeMap::new(),
            }
        })
    }

    /// The table to use: `MKLIGHT_COST_CALIBRATION` (a JSON file written by
    /// `tests/cost_calibration.rs`) when set and readable, else the built-in
    /// one. Returns the table and where it came from.
    pub fn load() -> (Self, String) {
        if let Ok(path) = std::env::var("MKLIGHT_COST_CALIBRATION") {
            match std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|s| serde_json::from_str::<Self>(&s).map_err(|e| e.to_string()))
            {
                Ok(c) if c.format == FORMAT && c.model_version == super::VERSION => {
                    return (c, path);
                }
                Ok(_) => eprintln!(
                    "mklight: {path}: cost calibration of another model version, using the built-in table"
                ),
                Err(e) => eprintln!("mklight: {path}: {e}; using the built-in cost calibration"),
            }
        }
        (Self::builtin(), "built-in".into())
    }
}

/// Non-negative relative least squares: minimise Σ ((f_i·c − t_i)/t_i)²
/// over c ≥ 0 by cyclic coordinate descent on column-normalised features.
/// Deterministic; columns that are zero everywhere get 0.
pub fn fit(points: &[(Features, f64)]) -> Fit {
    let weighted: Vec<_> = points.iter().map(|&(f, t)| (f, t, 1.)).collect();
    fit_weighted(&weighted)
}

/// [`fit`] with a weight per point (`(features, ms, weight)`), for
/// example to discount calls measured on a contended host.
pub fn fit_weighted(points: &[(Features, f64, f64)]) -> Fit {
    let n = FEATURES.len();
    let rows: Vec<(Features, f64)> = points
        .iter()
        .filter(|(_, t, w)| t.is_finite() && *t > 0. && *w > 0.)
        .map(|(f, t, w)| (std::array::from_fn(|k| w.sqrt() * f[k] / t), w.sqrt()))
        .collect();
    let mut fit = Fit {
        points: rows.len(),
        ..Default::default()
    };
    if rows.is_empty() {
        return fit;
    }
    let norm: Vec<f64> = (0..n)
        .map(|k| rows.iter().map(|(a, _)| a[k] * a[k]).sum::<f64>().sqrt())
        .collect();
    let a: Vec<Vec<f64>> = rows
        .iter()
        .map(|(r, _)| {
            (0..n)
                .map(|k| if norm[k] > 0. { r[k] / norm[k] } else { 0. })
                .collect()
        })
        .collect();
    let mut c = vec![0.; n];
    let mut residual: Vec<f64> = rows.iter().map(|(_, b)| *b).collect();
    for _ in 0..4000 {
        let mut moved = 0f64;
        for k in (0..n).filter(|&k| norm[k] > 0.) {
            let num: f64 = a.iter().zip(&residual).map(|(r, e)| r[k] * e).sum();
            let next = (c[k] + num).max(0.);
            let step = next - c[k];
            if step != 0. {
                for (r, e) in a.iter().zip(residual.iter_mut()) {
                    *e -= step * r[k];
                }
                c[k] = next;
                moved = moved.max(step.abs());
            }
        }
        if moved < 1e-12 {
            break;
        }
    }
    for k in 0..n {
        fit.coef[k] = if norm[k] > 0. { c[k] / norm[k] } else { 0. };
    }
    let logs: Vec<f64> = points
        .iter()
        .filter(|(_, t, w)| t.is_finite() && *t > 0. && *w > 0.)
        .map(|(f, t, _)| {
            let p: f64 = fit.stages(f).iter().sum();
            (p.max(1e-6) / t).ln()
        })
        .collect();
    fit.rms_log_error = (logs.iter().map(|l| l * l).sum::<f64>() / logs.len() as f64).sqrt();
    fit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_non_negative_stage_coefficients() {
        // t = 2 + 0.5·prep + 1e-3·ray, over sizes spanning three decades.
        let points: Vec<_> = [(1., 10.), (4., 1e3), (2., 1e4), (8., 1e5), (3., 1e6)]
            .iter()
            .map(|&(p, r)| ([1., p, r, 0., 0., 0.], 2. + 0.5 * p + 1e-3 * r))
            .collect();
        let f = fit(&points);
        assert!((f.coef[0] - 2.).abs() < 1e-6, "{f:?}");
        assert!((f.coef[1] - 0.5).abs() < 1e-6 && (f.coef[2] - 1e-3).abs() < 1e-9);
        assert_eq!((f.coef[3], f.coef[4], f.coef[5]), (0., 0., 0.));
        assert!(f.rms_log_error < 1e-9);
        // A negative slope in the data is clamped to zero.
        let falling = [([1., 1., 0., 0., 0., 0.], 10.), ([1., 9., 0., 0., 0., 0.], 5.)];
        let f = fit(&falling);
        assert!(f.coef.iter().all(|c| *c >= 0.) && f.coef[1] == 0.);
    }

    #[test]
    fn builtin_table_matches_this_model() {
        let c = Calibration::builtin();
        assert_eq!((c.format, c.model_version), (FORMAT, super::super::VERSION));
        for name in crate::analysis::gpu::GPU_ANALYSES {
            if !super::super::always_excluded(name) && !super::super::static_route(name)
            {
                let a = c.analyses.get(*name);
                assert!(
                    a.is_some_and(|a| a.cpu.is_some() && a.metal.is_some()),
                    "{name} is not calibrated"
                );
            }
        }
    }
}
