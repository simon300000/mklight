//! Route options: named alternative divisions of an analysis between the
//! CPU and the GPU. The default division of each analysis was measured on
//! one machine; an option selects a GPU-heavier alternative that a machine
//! with more GPU cores per CPU core may prefer (docs/GPU.md §9.3). They are
//! the hook for letting users test and choose routes on their own machine.
//!
//! The engine has default options (`EngineConfig::route`) and a request may
//! carry its own (`AnalysisRequest::route`), which replace the engine's as a
//! whole. The options of a run are checked when it is submitted, are part
//! of the key of a stored result, and reach the solver that divides the
//! work as its typed parameters; the run's backend evidence names the
//! division that ran.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{EngineError, EngineResult};

/// The route options of a run. Every key has a default, so `{}` is valid
/// JSON; an unknown key is an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteOptions {
    /// Pupil aiming of the geometric device analyses.
    pub aiming: Aiming,
    /// Under `aiming: host-table`, the device's share in [0, 1] of the
    /// generations large enough to be aimed by host and device together
    /// (spot diagram, geometric MTF); 0 aims them on the host alone. `None`:
    /// the default share (`mklight_analysis::gpu::aim::SPLIT_DEVICE_SHARE`).
    pub aiming_share: Option<f64>,
    /// Who solves the chief rays of `chief_ray_angle`, `pupil_aberration`
    /// and `grid_distortion` on a device.
    pub fieldmaps: FieldMaps,
    /// The device's share in [0, 1] of the source chunks of `stray_light`;
    /// the CPU reference traces the others. `None`: the rule of the scene
    /// (`mklight_stray::gpu::hybrid::default_share`).
    pub stray_share: Option<f64>,
    /// The device's intersection of extended sags in `stray_light`.
    pub stray_sag: StraySag,
    /// Pupil aiming of the solver jobs (`optimize`, `tolerance`,
    /// `glass_substitution`, `merit_evaluation`) on a device.
    pub solver_aiming: SolverAiming,
}

/// [`RouteOptions::aiming`] (`mklight_analysis::gpu::aim`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Aiming {
    /// The host aims every ray in f64 (large generations from a seed
    /// table, and with [`RouteOptions::aiming_share`] of them on the
    /// device); the device traces.
    #[default]
    HostTable,
    /// A coarse host aim map and device Newton corrections of every ray,
    /// completed by the host.
    Table,
    /// Device aiming from host chief rows, completed by the host.
    Device,
}

/// [`RouteOptions::fieldmaps`] (`mklight_analysis::gpu::ChiefDesign`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldMaps {
    /// The host solves every field's chief ray in f64; the device traces.
    #[default]
    HostChief,
    /// The device solves the chief rays too.
    DeviceChief,
}

/// [`RouteOptions::stray_sag`] (`mklight_stray::gpu::SagMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StraySag {
    /// The device scans the whole ray in f32.
    #[default]
    F32Scan,
    /// As `f32-scan`, with oblique and grazing rays handed to the CPU.
    F32Grazing,
}

/// [`RouteOptions::solver_aiming`] (docs/OPTIMIZATION.md, "Aiming").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverAiming {
    /// The evaluator's rule decides per batch
    /// (`mklight_optim::gpu::DEVICE_AIM_FROM`).
    #[default]
    ByBatch,
    /// Every batch is aimed on the host; the device traces.
    Host,
    /// Every batch is aimed on the device, completed by the host.
    Device,
}

impl SolverAiming {
    /// The name in a solver's backend report (`backend.aiming`).
    pub fn name(self) -> &'static str {
        match self {
            SolverAiming::ByBatch => "by-batch",
            SolverAiming::Host => "host",
            SolverAiming::Device => "device",
        }
    }
}

impl RouteOptions {
    /// The options spelled as `KEY=VALUE` pairs, as on a command line: a
    /// value that reads as a number or `null` is that, any other value is a
    /// name. Keys not given keep their defaults; of a repeated key the last
    /// value counts. The result is checked ([`Self::validate`]).
    pub fn from_pairs<'a>(
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> EngineResult<Self> {
        let mut object = Map::new();
        for (key, value) in pairs {
            let value = match serde_json::from_str::<Value>(value) {
                Ok(v) if v.is_number() || v.is_null() => v,
                _ => Value::String(value.to_string()),
            };
            object.insert(key.to_string(), value);
        }
        let route: Self = serde_json::from_value(Value::Object(object))
            .map_err(|e| EngineError::InvalidSettings(format!("route options: {e}")))?;
        route.validate()?;
        Ok(route)
    }

    /// Checks the values serde cannot: the shares lie in [0, 1].
    pub fn validate(&self) -> EngineResult<()> {
        for (key, share) in [
            ("aiming_share", self.aiming_share),
            ("stray_share", self.stray_share),
        ] {
            if share.is_some_and(|s| !(0.0..=1.0).contains(&s)) {
                return Err(EngineError::InvalidSettings(format!(
                    "route option {key}: {} is not in [0, 1]",
                    share.unwrap_or(f64::NAN)
                )));
            }
        }
        Ok(())
    }

    /// The division of the geometric device jobs.
    pub(crate) fn geometric(&self) -> mklight_analysis::gpu::Route {
        use mklight_analysis::gpu::{AimDesign, ChiefDesign, Route};
        let default = Route::default();
        Route {
            aiming: match self.aiming {
                Aiming::HostTable => AimDesign::HostTable,
                Aiming::Table => AimDesign::Table,
                Aiming::Device => AimDesign::Device,
            },
            device_share: self.aiming_share.unwrap_or(default.device_share),
            chief: match self.fieldmaps {
                FieldMaps::HostChief => ChiefDesign::HostChief,
                FieldMaps::DeviceChief => ChiefDesign::DeviceChief,
            },
        }
    }

    /// The transport options of the device stray-light analysis.
    pub(crate) fn transport(&self) -> mklight_stray::gpu::TransportOptions {
        use mklight_stray::gpu::{SagMode, TransportOptions};
        TransportOptions {
            gpu_share: self.stray_share,
            sag: match self.stray_sag {
                StraySag::F32Scan => SagMode::F32Scan,
                StraySag::F32Grazing => SagMode::F32Grazing,
            },
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_the_solvers_defaults() {
        let route = RouteOptions::default();
        assert_eq!(route.geometric(), mklight_analysis::gpu::Route::default());
        let transport = route.transport();
        let default = mklight_stray::gpu::TransportOptions::default();
        assert_eq!(
            (transport.gpu_share, transport.sag),
            (default.gpu_share, default.sag)
        );
        assert_eq!(route.solver_aiming, SolverAiming::ByBatch);
        assert_eq!(
            serde_json::to_value(route).unwrap(),
            json!({"aiming": "host-table", "aiming_share": null, "fieldmaps": "host-chief",
                   "stray_share": null, "stray_sag": "f32-scan", "solver_aiming": "by-batch"})
        );
        assert_eq!(
            serde_json::from_value::<RouteOptions>(json!({})).unwrap(),
            route
        );
    }

    #[test]
    fn every_value_reaches_its_solver_parameter() {
        use mklight_analysis::gpu::{AimDesign, ChiefDesign};
        let route: RouteOptions = serde_json::from_value(json!({
            "aiming": "device", "aiming_share": 0.75, "fieldmaps": "device-chief",
            "stray_share": 1, "stray_sag": "f32-grazing", "solver_aiming": "host"}))
        .unwrap();
        route.validate().unwrap();
        let g = route.geometric();
        assert_eq!(
            (g.aiming, g.device_share, g.chief),
            (AimDesign::Device, 0.75, ChiefDesign::DeviceChief)
        );
        let table: RouteOptions = serde_json::from_value(json!({"aiming": "table"})).unwrap();
        assert_eq!(table.geometric().aiming, AimDesign::Table);
        let t = route.transport();
        assert_eq!(
            (t.gpu_share, t.sag),
            (Some(1.0), mklight_stray::gpu::SagMode::F32Grazing)
        );
        assert_eq!(route.solver_aiming, SolverAiming::Host);
        for (name, aiming) in [
            ("by-batch", SolverAiming::ByBatch),
            ("host", SolverAiming::Host),
            ("device", SolverAiming::Device),
        ] {
            assert_eq!(aiming.name(), name);
            assert_eq!(serde_json::to_value(aiming).unwrap(), json!(name));
        }
    }

    #[test]
    fn unknown_keys_names_and_shares_are_rejected() {
        let parse = |v: Value| serde_json::from_value::<RouteOptions>(v);
        for bad in [
            json!({"stray": "wavefront"}),
            json!({"aiming": "split:0.5"}),
            json!({"aiming": "host"}),
            json!({"fieldmaps": "cpu"}),
            json!({"stray_sag": "f64"}),
            json!({"solver_aiming": "host-table"}),
            json!({"stray_share": "half"}),
        ] {
            assert!(parse(bad.clone()).is_err(), "{bad}");
        }
        for (key, share) in [
            ("aiming_share", 1.5),
            ("aiming_share", -0.1),
            ("stray_share", 2.0),
        ] {
            let route = parse(json!({key: share})).unwrap();
            let e = route.validate().unwrap_err();
            assert!(
                matches!(&e, EngineError::InvalidSettings(m) if m.contains(key)),
                "{e}"
            );
        }
        let nan = RouteOptions {
            stray_share: Some(f64::NAN),
            ..Default::default()
        };
        assert!(nan.validate().is_err());
    }

    #[test]
    fn pairs_spell_numbers_null_and_names() {
        let route = RouteOptions::from_pairs([
            ("aiming", "table"),
            ("stray_share", "0.5"),
            ("aiming_share", "1"),
            ("stray_share", "null"),
            ("solver_aiming", "device"),
        ])
        .unwrap();
        assert_eq!(
            route,
            RouteOptions {
                aiming: Aiming::Table,
                aiming_share: Some(1.0),
                solver_aiming: SolverAiming::Device,
                ..Default::default()
            }
        );
        assert_eq!(
            RouteOptions::from_pairs([]).unwrap(),
            RouteOptions::default()
        );
        for bad in [
            ("stray", "wavefront"),
            ("aiming", "0.5"),
            ("stray_share", "1.5"),
            ("fieldmaps", "null"),
        ] {
            let e = RouteOptions::from_pairs([bad]).unwrap_err();
            assert!(matches!(e, EngineError::InvalidSettings(_)), "{bad:?}: {e}");
        }
    }
}
