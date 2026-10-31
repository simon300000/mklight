//! Numeric upload boundary for P1. Small first-order field metadata is host
//! control work; no chief, real-image solve, or pupil ray is traced here.
use super::*;
use bytemuck::{Pod, Zeroable};
use mklight_core::{FieldKind, RayAiming, ResolvedSystem};
use mklight_trace::{FieldPoint, Paraxial};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FieldSpec {
    pub system: u32,
    pub wavelength: u32,
    pub flags: u32,
    pub reserved: u32,
    pub source: [Df32; 2],
    pub target: [Df32; 2],
    /// Entrance pupil z, aim scale (radius or tangent), stop radius.
    pub pupil: [Df32; 3],
    /// Decentre x/y, compression x/y.
    pub vignette: [Df32; 4],
    pub origin: [Df32; 3],
    pub rotation: [Df32; 9],
}
impl FieldSpec {
    pub fn new(
        sys: &ResolvedSystem,
        paraxial: &Paraxial,
        point: FieldPoint,
        system: u32,
        wavelength: u32,
        aiming: mklight_core::RayAiming,
    ) -> MkResult<Self> {
        let fo = paraxial.primary();
        let model = sys.system();
        let target = [
            point.hx * model.fields.max_radial(),
            point.hy * model.fields.max_radial(),
        ];
        let finite = !sys.object_at_infinity;
        let tele = fo.object_space_telecentric();
        let mut flags = if finite { MKL_AIM_FINITE } else { 0 }
            | if tele { MKL_AIM_TELECENTRIC } else { 0 }
            | if matches!(aiming, RayAiming::Real { .. }) {
                MKL_AIM_REAL
            } else {
                0
            };
        let source = match model.fields.kind {
            FieldKind::ObjectAngle => {
                flags |= MKL_AIM_ANGLE;
                target.map(|x| x.to_radians().tan())
            }
            FieldKind::ObjectHeight => target,
            FieldKind::ParaxialImageHeight | FieldKind::RealImageHeight => {
                if model.fields.kind == FieldKind::RealImageHeight {
                    flags |= MKL_AIM_IMAGE_HEIGHT;
                }
                let h = if finite {
                    fo.image_height_per_object_height
                } else {
                    fo.image_height_per_tan
                };
                let h = h
                    .filter(|h| h.is_finite() && *h != 0.0)
                    .ok_or_else(|| bad("paraxial chief does not reach the image"))?;
                target.map(|x| x / h)
            }
        };
        let frame = mklight_trace::trace::start_frame(sys);
        let v = point.vignetting;
        let spec = Self {
            system,
            wavelength,
            flags,
            reserved: 0,
            source: source.map(Df32::from_f64),
            target: target.map(Df32::from_f64),
            pupil: [
                if tele {
                    0.0
                } else {
                    fo.entrance_pupil_position
                },
                if tele {
                    fo.marginal_object_slope.abs()
                } else {
                    fo.entrance_pupil_radius()
                },
                fo.stop_semi_diameter,
            ]
            .map(Df32::from_f64),
            vignette: [v.vdx, v.vdy, v.vcx, v.vcy].map(Df32::from_f64),
            origin: [frame.origin.x, frame.origin.y, frame.origin.z].map(Df32::from_f64),
            rotation: std::array::from_fn(|k| Df32::from_f64(frame.rot.m[k / 3][k % 3])),
        };
        if !bytemuck::cast_slice::<_, f32>(&[spec])[4..]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(bad("nonfinite aiming field metadata"));
        }
        Ok(spec)
    }
}

/// Explicit probes, lattice/rim nodes, or a caller's deterministic pupil
/// pattern. Their order and 64-bit IDs survive every operation and failure.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Sample {
    pub pupil: [Df32; 2],
    pub field: u32,
    pub id: [u32; 2],
    pub weight: f32,
}
impl Sample {
    pub fn new(pupil: [f64; 2], field: u32, id: u64) -> Self {
        Self {
            pupil: pupil.map(Df32::from_f64),
            field,
            id: [id as u32, (id >> 32) as u32],
            weight: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChiefSeed {
    pub source: [Df32; 3],
    pub chief: [Df32; 2],
    pub jacobian: [Df32; 4],
    pub residual: Df32,
    pub status: u32,
    pub iterations: u32,
    pub has_jacobian: u32,
    pub reserved: u32,
}
const _: () = assert!(size_of::<FieldSpec>() == MKL_AIM_FIELD_WORDS as usize * 4);
const _: () = assert!(size_of::<Sample>() == MKL_AIM_SAMPLE_WORDS as usize * 4);
const _: () = assert!(size_of::<ChiefSeed>() == MKL_AIM_CACHE_WORDS as usize * 4);

/// Stable sparse unresolved list for an adapter's optional f64 polish. A
/// failure is never relabelled converged; polishing is a separate decision.
pub fn unresolved(report: &[u32], logical_ids: &[u32]) -> Vec<(usize, u64, u32)> {
    let n = report.len() / MKL_AIM_REPORT_PLANES as usize;
    (0..n)
        .filter(|&i| report[i] != 0)
        .map(|i| {
            (
                i,
                logical_ids[i] as u64 | ((logical_ids[n + i] as u64) << 32),
                report[i],
            )
        })
        .collect()
}
