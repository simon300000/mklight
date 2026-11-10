//! Shared run-local spectrum preparation, before any solver/material cache.
use super::*;
use mklight_analysis::spectrum::{SpectralSampling, Spectrum};

/// `schema` with the optional `spectrum` setting when the analysis uses
/// wavelengths: the categories that trace rays, and by name the model and
/// material reports that evaluate the design's wavelengths.
pub(super) fn settings_schema(mut schema: Value, category: &str, name: &str) -> Value {
    if matches!(
        category,
        "geometric" | "wave" | "ghost" | "stray" | "radiometry" | "polarization" | "optimization"
    ) || matches!(
        name,
        "system_summary" | "validate" | "glass_info" | "solves"
    ) {
        let sampling = super::schema::settings_from_default::<SpectralSampling>(json!({
            "samples": {"minimum":1,"maximum":4096},
            "range_um": {"description":"Increasing [min,max] wavelength in µm, inside the response table."},
            "spacing": {"enum":["equal_weight","uniform"]}
        }));
        schema["properties"]["spectrum"] = json!({
            "type":["object","null"],
            "description":"Optional run-local spectrum; absent/null keeps the design. kind: explicit (wavelengths:[{um,weight}]), photopic, scotopic, blackbody (temperature_k), sensor (table:[[um,response]]). Presets/sensor take sampling. Primary is nearest the energy-weighted mean (first on ties).",
            "x-sampling":sampling,
            "x-kinds":["explicit","photopic","scotopic","blackbody","sensor"]
        });
    }
    schema
}

/// The input with the design's wavelengths replaced by `spectrum` (a
/// derived snapshot used by solver caches).
pub(super) fn derived<'a>(
    input: &AnalysisInput<'a>,
    spectrum: Value,
) -> EngineResult<AnalysisInput<'a>> {
    let spectrum: Spectrum = serde_json::from_value(spectrum)
        .map_err(|e| EngineError::InvalidSettings(format!("spectrum: {e}")))?;
    let mut system = input.snapshot.system().clone();
    system.wavelengths = spectrum.wavelengths()?;
    let snapshot = Snapshot::new(system)?;
    let (s, l) = (snapshot.clone(), input.library.clone());
    Ok(input
        .derive(
            snapshot,
            input.backend.clone(),
            Box::new(move || ResolvedSystem::new(&s, l.clone()).map(Arc::new)),
        )
        .with_device(input.device.clone())
        .with_selection(
            input.requested_backend.clone(),
            input.selection_note.clone(),
        )
        .with_cost_decision(input.cost_decision.borrow().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;
    use crate::{AnalysisRequest, Engine, EngineConfig};
    use mklight_core::{model::Wavelength, samples};
    #[test]
    fn custom_spot_matches_design_selection_and_keeps_design() {
        let engine = Engine::new(EngineConfig::default());
        let original = samples::cooke_triplet();
        let session = engine
            .session_from_project(Project::new(original.clone()))
            .unwrap();
        let custom=session.run_analysis(&AnalysisRequest::new("spot_diagram",json!({"spectrum":{"kind":"explicit","wavelengths":[{"um":0.55,"weight":1.0}]}})),&JobContext::new()).unwrap();
        let mut modified = original.clone();
        modified.wavelengths.wavelengths = vec![Wavelength {
            um: 0.55,
            weight: 1.0,
        }];
        modified.wavelengths.primary = 0;
        let expected_hash = Snapshot::new(modified.clone()).unwrap().hash().to_string();
        let other = engine.session_from_project(Project::new(modified)).unwrap();
        let plain = other
            .run_analysis(
                &AnalysisRequest::new("spot_diagram", Value::Null),
                &JobContext::new(),
            )
            .unwrap();
        assert_eq!(custom.json, plain.json);
        assert_eq!(custom.buffers, plain.buffers);
        assert_eq!(
            custom.run_info.snapshot_hash,
            session.snapshot_hash().to_string()
        );
        assert_eq!(
            custom.derived_snapshot_hash.as_deref(),
            Some(expected_hash.as_str())
        );
        assert_eq!(custom.envelope()["derived_snapshot_hash"], expected_hash);
        assert!(plain.derived_snapshot_hash.is_none());
        let mut appended = original.clone();
        appended.wavelengths.wavelengths.push(Wavelength {
            um: 0.55,
            weight: 1.0,
        });
        appended.wavelengths.primary = 3;
        let appended = engine.session_from_project(Project::new(appended)).unwrap();
        let selected = appended
            .run_analysis(
                &AnalysisRequest::new(
                    "spot_diagram",
                    json!({"wavelengths":{"type":"index","index":3}}),
                ),
                &JobContext::new(),
            )
            .unwrap();
        // Wavelength identity is an index; geometric coordinates and
        // wavelength/weight columns must match the single-sample run.
        for name in ["points", "weights"] {
            let a = custom.buffers.iter().find(|b| b.name == name);
            let b = selected.buffers.iter().find(|b| b.name == name);
            assert_eq!(a.map(|b| &b.data), b.map(|b| &b.data));
        }
        let base = session
            .run_analysis(
                &AnalysisRequest::new("spot_diagram", Value::Null),
                &JobContext::new(),
            )
            .unwrap();
        assert_eq!(
            base.run_info.snapshot_hash,
            Snapshot::new(original).unwrap().hash().to_string()
        );
    }
    /// A run-local spectrum runs under the request's policy: tier and aim
    /// stop reach the derived input and are the ones `RunInfo` reports.
    #[test]
    fn custom_spectrum_keeps_the_run_policy() {
        use mklight_core::job::{AimStopKind, Tier};
        struct Policy;
        impl Analysis for Policy {
            fn name(&self) -> &str {
                "policy_echo"
            }
            fn title(&self) -> &str {
                "policy"
            }
            fn category(&self) -> &str {
                "geometric"
            }
            fn run(
                &self,
                input: &AnalysisInput<'_>,
                _: &Value,
                ctx: &JobContext,
            ) -> EngineResult<AnalysisOutput> {
                let tracer = input.tracer(ctx)?;
                Ok(AnalysisOutput {
                    json: json!({
                        "policy": input.policy,
                        "aim_stop": tracer.settings().aim_stop,
                        "wavelengths": tracer.num_wavelengths(),
                    }),
                    ..Default::default()
                })
            }
        }
        let engine = Engine::new(EngineConfig::default());
        engine.register_analysis(Arc::new(Policy));
        let session = engine
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let mut req = AnalysisRequest::new(
            "policy_echo",
            json!({"spectrum":{"kind":"explicit","wavelengths":[{"um":0.55,"weight":1.0}]}}),
        );
        req.tier = Tier::Preview;
        // Neither is the default policy (design tier, requested aim stop).
        let policy = RunPolicy {
            tier: Tier::Preview,
            aim_stop: AimStopKind::Conditioned,
        };
        assert_eq!(engine.config().aim_stop, policy.aim_stop);
        let r = session.run_analysis(&req, &JobContext::new()).unwrap();
        assert_eq!(
            r.json,
            json!({
                "policy": policy,
                "aim_stop": policy.tracer_settings().aim_stop,
                "wavelengths": 1,
            })
        );
        assert_eq!(
            (r.run_info.tier, r.run_info.aim_stop),
            (policy.tier, policy.aim_stop)
        );
    }

    #[test]
    fn custom_spectrum_material_range_warning() {
        let engine = Engine::new(EngineConfig::default());
        let session = engine
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let r = session
            .run_analysis(
                &AnalysisRequest::new(
                    "spot_diagram",
                    json!({"spectrum":{"kind":"explicit","wavelengths":[{"um":2.6,"weight":1.0}]}}),
                ),
                &JobContext::new(),
            )
            .unwrap();
        assert!(
            r.run_info.warnings.iter().any(|w| w.contains("range")),
            "{:?}",
            r.run_info.warnings
        );
    }
}
