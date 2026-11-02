//! Static capability and measured cost policy; no runtime f64 verification.
use super::{Analysis, AnalysisInput, AnalysisOutput, BackendUse, gpu};
use crate::{backend::BackendInfo, error::EngineResult};
use mklight_core::{JobContext, job::Precision};
use serde_json::{Value, json};

/// Analyses whose Metal output failed the f64 comparison on most P11 default
/// samples (GPU §9.2) always run on the reference. Sporadic disagreements are
/// documented residual gaps, not runtime gates.
fn reference_reason(analysis: &str) -> Option<&'static str> {
    match analysis {
        "encircled_energy" => Some(
            "static capability: P11 encircled_energy failed f64 comparison on 16/16 default samples (geometry-uncertainty bin completion)",
        ),
        "wavefront_map" => Some(
            "static capability: P11 wavefront_map failed f64 comparison on 16/16 default samples (undefined on-axis Seidel angles)",
        ),
        "polarization_pupil_map" => Some(
            "static capability: P11 polarization_pupil_map failed f64 comparison on 13/16 default samples (undefined polarization axes)",
        ),
        _ => None,
    }
}

pub(crate) fn run(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let Some(device) = gpu::device(input) else {
        return analysis.run(input, settings, ctx);
    };
    let reference = AnalysisInput::with_resolver(
        input.snapshot.clone(),
        input.library.clone(),
        BackendInfo::cpu_reference(),
        Box::new(|| input.resolved()),
    )
    .with_stop_token(input.stop.clone())
    .with_library_hash(input.library_hash().into());
    let capability = reference_reason(analysis.name());
    let cost = if let Some(reason) = capability {
        json!({"decision":"cpu_reference","rule":"static capability","reason":reason})
    } else if crate::backend::Backend::is_auto_request(&input.requested_backend) {
        super::cost::decision(
            analysis.name(),
            settings,
            input.snapshot.system(),
            device.backend.info(),
        )
    } else {
        json!({"decision":"explicit_backend","reason":"cost rules bypassed"})
    };
    let cpu = cost["decision"] == "cpu_reference";
    *input.cost_decision.borrow_mut() = Some(cost);
    let fallback = |reason: String, deliberate| {
        input.report_backend(BackendUse {
            id: "cpu_reference".into(),
            precision: Precision::F64Reference,
            device: None,
            fallback_reason: Some(reason),
            deliberate,
        });
    };
    if cpu {
        fallback(
            capability
                .unwrap_or("auto: the CPU reference is faster for this workload")
                .into(),
            true,
        );
        return analysis.run(&reference, settings, ctx);
    }
    match analysis.run(input, settings, ctx) {
        Ok(out) => Ok(out),
        Err(e) if e.is_cancelled() => Err(e),
        Err(e) => {
            ctx.check()?;
            fallback(format!("resident {} failed: {e}", analysis.name()), false);
            analysis.run(
                &reference,
                settings,
                &ctx.subrange(ctx.progress.fraction(), 1.),
            )
        }
    }
}
