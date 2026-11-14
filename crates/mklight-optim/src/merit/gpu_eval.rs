//! Device ray operands: which operands the device traces ([`gpu_supports`])
//! and the shared host helpers of the resident candidate traces
//! ([`resident`], driven by [`crate::gpu::ResidentEvaluator`]).
//!
//! Device rays are double-float traces (hi + lo planes), so values carry a
//! rounding noise of ~1e-12 relative, against ~1e-16 on the f64 reference.
//! Forward differences therefore take larger steps than on the reference
//! (`docs/OPTIMIZATION.md`).

use mklight_trace::{OpdReference, SequentialTracer, TraceResult};

use super::OperandError;
use super::operand::{OperandKind, RayQuantity};

/// Can the device trace operands of `kind`? Everything else (paraxial,
/// model, …) is evaluated on the host; relative illumination is integrated
/// there too, with its visibility scans traced on the device.
pub(crate) fn gpu_supports(kind: &OperandKind) -> bool {
    use OperandKind::*;
    match kind {
        RayAberration { .. }
        | RmsSpot { .. }
        | SpotCentroid { .. }
        | Distortion { .. }
        | LateralColour { .. }
        | WavefrontAberration { .. }
        | WavefrontRms { .. } => true,
        RealRay { quantity, .. } => *quantity != RayQuantity::IncidenceAngle,
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
        afocal: tr.afocal_image_space(),
    })
}

pub(crate) mod resident;
