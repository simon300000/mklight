//! Engine-only validation: reference-tracer setup and chief/marginal ray
//! reachability. Structural model checks belong to mklight-core.

use mklight_core::ResolvedSystem;
use mklight_trace::{SequentialTracer, TraceStatus};
use std::sync::Arc;

/// One validation finding (also the `issues` entries of `validate`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Issue {
    /// `error`, `warning` or `info`.
    pub severity: &'static str,
    pub code: &'static str,
    /// Surface index, when the issue belongs to one surface.
    pub surface: Option<usize>,
    pub message: String,
}

impl Issue {
    pub fn new(
        severity: &'static str,
        code: &'static str,
        surface: Option<usize>,
        message: String,
    ) -> Self {
        Issue {
            severity,
            code,
            surface,
            message,
        }
    }
}

/// Checks that need the resolved system: the reference tracer must set the
/// system up, and the chief and marginal rays should reach the image.
pub fn tracer_checks(resolved: Arc<ResolvedSystem>) -> Vec<Issue> {
    let mut out = Vec::new();
    let tracer = match SequentialTracer::new(resolved) {
        Ok(t) => t,
        Err(e) => {
            out.push(Issue::new(
                "error",
                "paraxial",
                None,
                format!("the reference tracer cannot set up the system: {e}"),
            ));
            return out;
        }
    };
    let wl = tracer.system().primary_wavelength;
    for (k, field) in tracer.fields().iter().enumerate() {
        if let Some(e) = &field.error {
            out.push(Issue::new(
                "warning",
                "chief_ray",
                None,
                format!("field {k}: {e}"),
            ));
            continue;
        }
        let r = tracer.chief(field, wl);
        if let Some(s) = r.status.surface() {
            out.push(Issue::new(
                "warning",
                "chief_ray",
                Some(s),
                format!(
                    "field {k}: the chief ray is stopped ({}) at surface {s}",
                    status_name(&r.status)
                ),
            ));
        }
    }
    if let Some(field) = tracer.fields().first().filter(|f| f.is_valid()) {
        let r = tracer.trace(field, wl, 0.0, 1.0, &tracer.options());
        if let Some(s) = r.status.surface() {
            out.push(Issue::new(
                "warning",
                "marginal_ray",
                Some(s),
                format!(
                    "field 0: the marginal ray (pupil 0, 1) is stopped ({}) at surface {s}",
                    status_name(&r.status)
                ),
            ));
        }
    }
    out
}

/// Short name of a trace status (`miss`, `clipped`, …).
fn status_name(s: &TraceStatus) -> &'static str {
    match s {
        TraceStatus::Ok => "ok",
        TraceStatus::Miss { .. } => "miss",
        TraceStatus::Clipped { .. } => "clipped",
        TraceStatus::TotalInternalReflection { .. } => "total_internal_reflection",
        TraceStatus::Failed { .. } => "failed",
    }
}
