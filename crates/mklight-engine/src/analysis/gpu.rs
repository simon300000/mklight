//! Complete resident analysis routes. Qualification and cost policy apply to
//! the whole public result; the legacy host-generated trace route is retired.
use std::sync::Arc;

use mklight_core::{RunInfo, job::Precision};
use serde_json::json;

use super::{AnalysisInput, BackendUse};
use crate::gpu::GpuDevice;

/// The same operation catalogue drives registry flags and provider capabilities.
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

pub(crate) fn record_native(input: &AnalysisInput<'_>, run: &RunInfo) {
    let Some(dev) = device(input) else { return };
    input.executions.borrow_mut().push(json!({
        "backend": run.backend, "precision": run.precision,
        "snapshot_hash": run.snapshot_hash, "settings": run.settings,
        "seed": run.seed, "duration_ms": run.duration_ms,
    }));
    let reference = run.precision == Precision::F64Reference;
    input.report_backend(BackendUse {
        id: if reference {
            "cpu_reference".into()
        } else {
            dev.id.clone()
        },
        precision: run.precision,
        device: (!reference).then(|| dev.device_name()),
        fallback_reason: reference
            .then(|| "resident adapter required whole-result f64 completion".into()),
        qualification: None,
        deliberate: false,
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

/// Reproducible workload identity used by the consolidated benchmark/table.
pub fn cost_key(
    analysis: &str,
    snapshot: &str,
    library: &str,
    settings: &serde_json::Value,
) -> String {
    super::cost::key(analysis, snapshot, library, settings)
}

/// Host source/compiler/profile identity for full-chain cost calibration.
pub fn cost_build() -> &'static str {
    super::cost::BUILD
}
