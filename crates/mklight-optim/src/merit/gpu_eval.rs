//! Device ray operands: which operands the device traces ([`gpu_supports`])
//! and the shared host helpers of the resident candidate traces
//! ([`resident`], driven by [`crate::gpu::ResidentEvaluator`]).
//!
//! Reported points are f32 (`docs/GPU.md` §8), so values carry a rounding
//! noise of about half an f32 ulp of the image coordinates. Finite
//! differences therefore need larger steps than on the f64 reference
//! (`docs/OPTIMIZATION.md`).

use mklight_trace::{OpdReference, SequentialTracer, TraceResult};

use super::OperandError;
use super::operand::{OperandKind, RayQuantity};

/// Can the device trace operands of `kind` (with wavefront operands enabled
/// or not)? Everything else (paraxial, model, relative illumination, …) is
/// evaluated on the host.
pub fn gpu_supports(kind: &OperandKind, wavefront: bool) -> bool {
    use OperandKind::*;
    match kind {
        RayAberration { .. }
        | RmsSpot { .. }
        | SpotCentroid { .. }
        | Distortion { .. }
        | LateralColour { .. } => true,
        RealRay { quantity, .. } => *quantity != RayQuantity::IncidenceAngle,
        WavefrontAberration { .. } | WavefrontRms { .. } => wavefront,
        _ => false,
    }
}

/// The OPD reference sphere of a device chief ray (as
/// `SequentialTracer::opd_reference`, from the device's point, direction and
/// optical path).
fn gpu_reference(
    tr: &SequentialTracer,
    chief: &TraceResult,
    wl: usize,
) -> Result<OpdReference, OperandError> {
    let sys = tr.system();
    let img = sys.image_index();
    let z_xp = tr.paraxial().at(wl).exit_pupil_position;
    let s_e = if z_xp.is_finite() {
        if chief.ray.dir.z == 0.0 {
            return Err(OperandError::RayFailure(
                "chief ray: degenerate reference sphere".into(),
            ));
        }
        (z_xp - chief.ray.pos.z) / chief.ray.dir.z
    } else {
        f64::INFINITY
    };
    Ok(OpdReference {
        chief_opl: chief.opl,
        center: chief.ray.pos,
        chief_dir: chief.ray.dir,
        s_exit_pupil: s_e,
        n_image: sys.n_before(img, wl),
        wavelength_um: sys.wavelengths_um[wl],
    })
}

pub(crate) mod resident;
