//! One merit evaluation on the snapshot, sharing optimisation's setup and report.

use std::sync::Arc;

use mklight_core::JobContext;
use serde_json::Value;

use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry, optim};
use crate::error::EngineResult;

pub fn register_all(r: &mut AnalysisRegistry) {
    r.register(Arc::new(MeritEvaluation));
}

pub struct MeritEvaluation;

impl Analysis for MeritEvaluation {
    /// 3: MTF operands use the radiometric pupil amplitude.
    /// Also: wavefront operands of afocal image spaces use the plane OPD
    /// reference.
    fn result_version(&self) -> u32 {
        3
    }
    fn name(&self) -> &str {
        "merit_evaluation"
    }
    fn title(&self) -> &str {
        "Evaluate merit"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Evaluates the snapshot's merit once, without optimisation or finite differences. \
         Uses optimize's merit settings, solves, default generator and operand report; returns \
         the expanded merit spec for conversion to custom operands."
    }
    fn settings_schema(&self) -> Value {
        // An optimize request; `algorithm` still selects the EFL hold.
        optim::Optimize.settings_schema()
    }
    fn result_schema(&self) -> Value {
        use super::schema::*;
        result(
            &[
                ("merit", nf()),
                (
                    "violation",
                    nf_u("", "Total normalised constraint violation."),
                ),
                ("status", opaque("EvalStatus.")),
                (
                    "operands",
                    arr(
                        optim::operand_schema(),
                        "Snapshot operand values, residuals and merit shares.",
                    ),
                ),
                (
                    "merit_spec",
                    opaque(
                        "Effective MeritFunctionSpec, including generated operands and the EFL hold.",
                    ),
                ),
                ("backend", optim::evaluation_backend_schema()),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        optim::evaluate_merit(input, settings, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisResult, run_analysis};
    use crate::backend::BackendInfo;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::param::ParamPath;
    use mklight_core::{OpticalSystem, Snapshot, SurfaceId, samples};
    use serde_json::json;

    fn run(system: OpticalSystem, name: &str, settings: Value) -> AnalysisResult {
        let input = AnalysisInput::new(
            Snapshot::new(system).unwrap(),
            Arc::new(MaterialLibrary::with_builtin()),
            BackendInfo::cpu_reference(),
        );
        let reg = AnalysisRegistry::with_builtin();
        run_analysis(
            reg.get(name).unwrap().as_ref(),
            &input,
            &settings,
            vec![],
            &JobContext::new(),
        )
        .unwrap()
    }

    #[test]
    fn one_evaluation_matches_optimize_start_on_both_lenses() {
        for system in [samples::cooke_triplet(), samples::double_gauss()] {
            for merit_settings in [
                json!({}),
                json!({"default_merit": {"criterion": {
                    "type": "mtf", "frequencies": [30.0], "target": 0.8, "worst_field": true
                }}}),
                json!({"merit": {"operands": [
                    {"kind": {"type": "efl"}, "target": 55.0, "weight": 0.01},
                    {"kind": {"type": "param_value", "param": {"param": "thickness", "surface": 1}},
                        "target": 2.0, "comparison": "greater_than", "weight": 4.0},
                    {"kind": {"type": "param_value", "param": {"param": "thickness", "surface": 1}},
                        "target": 5.0, "comparison": "less_than", "role": "constraint"}
                ]}}),
            ] {
                let mut settings = merit_settings;
                settings["backend"] = json!("cpu");
                settings["max_iterations"] = json!(1);
                let e = run(system.clone(), "merit_evaluation", settings.clone());
                super::super::schema::validate(&MeritEvaluation.result_schema(), &e.json, true)
                    .unwrap();
                let o = run(system.clone(), "optimize", settings.clone());
                let (e, o) = (&e.json, &o.json);
                for key in ["merit", "violation", "status"] {
                    assert_eq!(e[key], o["start"][key]);
                }
                assert_eq!(e["merit_spec"], o["merit_spec"]);
                let ops = e["operands"].as_array().unwrap();
                assert_eq!(ops.len(), o["operands"].as_array().unwrap().len());
                for (a, b) in ops.iter().zip(o["operands"].as_array().unwrap()) {
                    assert_eq!(a["value"], b["start"]);
                    for key in ["start", "target", "comparison", "weight", "role"] {
                        assert_eq!(a[key], b[key]);
                    }
                }
                let sum: f64 = ops.iter().filter_map(|a| a["contribution"].as_f64()).sum();
                assert!((sum / e["merit"].as_f64().unwrap() - 1.0).abs() < 1e-12);
                // Converting the expanded default to custom keeps every start value.
                settings["merit"] = e["merit_spec"].clone();
                let custom = run(system.clone(), "merit_evaluation", settings);
                assert_eq!(custom.json["operands"], e["operands"]);
                assert_eq!(custom.json["merit"], e["merit"]);
            }
        }
    }

    #[test]
    fn gpu_evaluation_traces_one_candidate_and_reports_its_backend() {
        use crate::{AnalysisRequest, Backend, Engine};
        let engine = Engine::default();
        engine.register_backend(Arc::new(crate::gpu::GpuProvider::emulation()));
        let mut session = engine.new_session();
        session.set_system(samples::cooke_triplet()).unwrap();
        let mut request =
            AnalysisRequest::new("merit_evaluation", json!({"backend": "gpu_emulation"}));
        request.backend = Some(Backend::Gpu("cpu_f32".into()));
        let a = session.run_analysis(&request, &JobContext::new()).unwrap();
        assert_eq!(a.run_info.backend, "gpu:cpu_f32");
        assert_eq!(a.json["backend"]["gpu_stats"]["gpu_candidates"], 1);
        assert_eq!(
            a.json["backend"]["reported_on"],
            a.json["backend"]["evaluated_on"]
        );
        // The GPU path stays inside the spot precision budget.
        let cpu = run(
            samples::cooke_triplet(),
            "merit_evaluation",
            json!({"backend": "cpu"}),
        );
        assert!(
            (a.json["merit"].as_f64().unwrap() / cpu.json["merit"].as_f64().unwrap() - 1.0).abs()
                < 0.01
        );
    }

    /// The EFL penalty of the derivative-free methods is weighted by the
    /// start merit on the CPU reference, in both jobs and on any backend.
    #[test]
    fn derivative_free_hold_uses_the_same_start_merit() {
        let settings = json!({"algorithm": "de", "max_iterations": 1,
            "options": {"population": 4}, "backend": "cpu"});
        let e = run(
            samples::cooke_triplet(),
            "merit_evaluation",
            settings.clone(),
        );
        let o = run(samples::cooke_triplet(), "optimize", settings);
        assert_eq!(e.json["merit_spec"], o.json["merit_spec"]);
        for key in ["merit", "violation"] {
            assert_eq!(e.json[key], o.json["start"][key]);
        }
        // On a device the penalty is still weighted by the reference's
        // start merit: both jobs report the same merit spec.
        use crate::{AnalysisRequest, Backend, Engine};
        let engine = Engine::default();
        engine.register_backend(Arc::new(crate::gpu::GpuProvider::emulation()));
        let mut session = engine.new_session();
        session.set_system(samples::cooke_triplet()).unwrap();
        let on_device = |analysis: &str| {
            let settings = json!({"algorithm": "de", "max_iterations": 1,
                "options": {"population": 4}, "backend": "gpu_emulation"});
            let mut request = AnalysisRequest::new(analysis, settings);
            request.backend = Some(Backend::Gpu("cpu_f32".into()));
            session.run_analysis(&request, &JobContext::new()).unwrap()
        };
        let (e, o) = (on_device("merit_evaluation"), on_device("optimize"));
        assert_eq!(e.json["backend"]["precision"], "f32_gpu");
        assert_ne!(e.json["merit"], o.json["start"]["merit"], "device values");
        assert_eq!(e.json["merit_spec"], o.json["merit_spec"]);
    }

    #[test]
    fn solves_precede_merit_generation_and_evaluation() {
        let mut system = samples::cooke_triplet();
        let target = ParamPath::Thickness {
            surface: SurfaceId(6),
        };
        system.solves = serde_json::from_value(json!([{
            "target": target, "kind": {"type": "pickup", "source": {"param": "thickness", "surface": 1},
                "scale": 1.0, "offset": 35.0}
        }])).unwrap();
        let settings = json!({"backend": "cpu", "max_iterations": 1, "merit": {"operands": [{
            "kind": {"type": "param_value", "param": target}, "target": 0.0
        }]}});
        let e = run(system.clone(), "merit_evaluation", settings.clone());
        assert_eq!(
            e.json["operands"][0]["value"],
            system.surfaces[1].thickness + 35.0
        );
        let o = run(system, "optimize", settings);
        assert_eq!(e.json["merit"], o.json["start"]["merit"]);
        assert!(
            o.json["variables"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["param"] != json!(target))
        );
    }

    #[test]
    fn operand_failures_are_reported_without_losing_other_values() {
        let mut system = samples::cooke_triplet();
        // A ray at the pupil edge is clipped by this physical aperture.
        system.surfaces[1].aperture =
            Some(mklight_core::model::ApertureShape::Circular { radius: 0.01 });
        let r = run(
            system,
            "merit_evaluation",
            json!({"hold_efl": false, "backend": "cpu",
            "merit": {"operands": [
                {"kind": {"type": "real_ray", "px": 0.0, "py": 1.0, "quantity": "y"}, "field": {"field": "index", "index": 0}},
                {"kind": {"type": "efl"}, "target": 100.0}
            ]}}),
        );
        assert!(r.json["operands"][0]["error"].is_string());
        assert!(r.json["operands"][0]["value"].is_null());
        assert!(r.json["operands"][1]["error"].is_null());
        assert!(r.json["operands"][1]["value"].is_number());
    }
}
