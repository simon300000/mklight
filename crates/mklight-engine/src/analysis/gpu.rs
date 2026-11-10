//! Complete resident analysis routes. Capability and cost policy apply to
//! the whole public result.
use std::sync::Arc;

use mklight_core::{RunInfo, job::Precision};
use serde_json::json;

use super::{AnalysisInput, BackendUse};
use crate::gpu::GpuDevice;

/// The analyses with a GPU path: the one catalogue behind
/// `Analysis::gpu_path` and the GPU providers' capabilities.
pub const GPU_ANALYSES: &[&str] = &[
    "layout",
    "spot_diagram",
    "ray_fan",
    "opd_fan",
    "field_curvature_distortion",
    "grid_distortion",
    "longitudinal_aberration",
    "lateral_colour",
    "relative_illumination",
    "footprint",
    "encircled_energy",
    "geometric_mtf",
    "best_focus",
    "ray_grid",
    "pupil_aberration",
    "relative_illumination_map",
    "chief_ray_angle",
    "system_transmission",
    "polarization_pupil_map",
    "wavefront_map",
    "psf",
    "mtf",
    "mtf_vs_field",
    "through_focus_mtf",
    "diffraction_encircled_energy",
    "line_spread",
    "image_simulation",
    "ghost_paths",
    "ghost_irradiance",
    "stray_light",
    "optimize",
    "merit_evaluation",
    "tolerance",
    "glass_substitution",
];

pub fn device(input: &AnalysisInput<'_>) -> Option<Arc<GpuDevice>> {
    input
        .device
        .clone()
        .and_then(|d| d.downcast::<GpuDevice>().ok())
}

/// Reported when a native adapter returns an f64 reference result.
const REFERENCE_COMPLETION: &str =
    "the native path completed the whole result on the CPU f64 reference";

pub(crate) fn record_native(input: &AnalysisInput<'_>, run: &RunInfo) {
    let Some(dev) = device(input) else { return };
    input.executions.borrow_mut().push(json!({
        "backend": run.backend, "precision": run.precision,
        "snapshot_hash": run.snapshot_hash, "settings": run.settings,
        "seed": run.seed, "duration_ms": run.duration_ms,
    }));
    let reference = run.precision == Precision::F64Reference;
    if reference {
        // The native adapter's whole time, including the f64 completion.
        input.record_attempt(mklight_core::job::BackendAttempt {
            backend: dev.id.clone(),
            outcome: "reference_completion".into(),
            reason: REFERENCE_COMPLETION.into(),
            elapsed_ms: run.duration_ms,
        });
    }
    input.report_backend(BackendUse {
        id: if reference {
            "cpu_reference".into()
        } else {
            dev.id.clone()
        },
        precision: run.precision,
        device: (!reference).then(|| dev.device_name()),
        fallback_reason: reference.then(|| REFERENCE_COMPLETION.into()),
        deliberate: false,
    });
}

/// A native adapter whose own division gave the device no part of this run
/// (`reason`): the reference is the run's expected backend, so nothing is
/// recorded as a failed attempt. An `auto` request reports the reason only;
/// an explicit device request also gets the warning.
pub(crate) fn record_reference_by_rule(input: &AnalysisInput<'_>, run: &RunInfo, reason: &str) {
    if device(input).is_none() {
        return;
    }
    input.executions.borrow_mut().push(json!({
        "backend": run.backend, "precision": run.precision,
        "snapshot_hash": run.snapshot_hash, "settings": run.settings,
        "seed": run.seed, "duration_ms": run.duration_ms,
    }));
    input.report_backend(BackendUse {
        id: "cpu_reference".into(),
        precision: run.precision,
        device: None,
        fallback_reason: Some(reason.into()),
        deliberate: crate::backend::Backend::is_auto_request(&input.requested_backend),
    });
}

/// Select a native adapter without changing its result type or its packing.
/// Errors reach the outer chain gate; cancellation never invokes the reference.
macro_rules! native_job {
    ($input:expr, |$backend:ident| $gpu:expr, $cpu:expr) => {{
        if let Some(device) = $crate::analysis::gpu::device($input) {
            let $backend = device.backend.as_ref();
            $gpu.map(|result| {
                $crate::analysis::gpu::record_native($input, &result.run_info);
                result
            })
        } else {
            $cpu
        }
    }};
}
pub(crate) use native_job;
