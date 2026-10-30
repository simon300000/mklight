//! Structure-of-arrays ray batches exchanged with trace kernels.
//!
//! Rays enter on a chosen surface, expressed in that surface's local frame
//! (CONVENTIONS §7.1): the host (the f64 reference code: field/pupil mapping,
//! ray aiming) produces them, rounds once to f32, and the kernel carries them
//! from surface to surface with relative transforms only. Optical path is
//! carried in double-float (hi/lo pairs).

use mklight_core::{MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::backend::SystemHandle;
use crate::consts::*;

/// Input rays (SoA; every vector has the same length). Construct with
/// `RayBatch::default()` and fill the fields; the struct may gain optional
/// per-ray fields (empty by default), so it is `non_exhaustive`.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct RayBatch {
    /// Position (mm) in the local frame of `TraceRequest::start_surface`.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    /// Unit direction cosines in the same frame.
    pub l: Vec<f32>,
    pub m: Vec<f32>,
    pub n: Vec<f32>,
    /// Optical path already accumulated (mm), double-float.
    pub opl_hi: Vec<f32>,
    pub opl_lo: Vec<f32>,
    /// Packed wavelength index (0..num_wavelengths) per ray.
    pub wavelength: Vec<u32>,
    /// System index within the uploaded set per ray; empty means all 0.
    pub system: Vec<u32>,
    /// Flux weight per ray (relative units or W, CONVENTIONS §1); empty
    /// means all 1. Carried through the trace and multiplied by the path
    /// transmittance (see [`Radiometry`]).
    pub weight: Vec<f32>,
}

impl RayBatch {
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    pub fn validate(&self) -> MkResult<()> {
        let n = self.len();
        let lens = [
            self.y.len(),
            self.z.len(),
            self.l.len(),
            self.m.len(),
            self.n.len(),
            self.opl_hi.len(),
            self.opl_lo.len(),
            self.wavelength.len(),
        ];
        let optional_ok = |k: usize| k == 0 || k == n;
        if lens.iter().any(|&k| k != n)
            || !optional_ok(self.system.len())
            || !optional_ok(self.weight.len())
        {
            return Err(MkError::Backend("ray batch arrays differ in length".into()));
        }
        if u32::try_from(n).is_err() {
            return Err(MkError::Backend("ray batch: more than 2^32 rays".into()));
        }
        Ok(())
    }
}

/// What the trace kernel records besides the final ray state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordMode {
    /// Only the state on `end_surface`.
    #[default]
    Final,
    /// The full ray state and status on every surface from start to end (for
    /// layout drawings, footprints, auto semi-diameters, per-surface
    /// throughput and ghost bookkeeping).
    AllSurfaces,
}

/// How the trace carries flux (CONVENTIONS §5). The enum is
/// `non_exhaustive`: a polarised mode (a Jones vector or Stokes state per ray,
/// with extra SoA fields in [`RayBatch`]/[`TraceResult`]) is added as a new
/// variant without changing the existing ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Radiometry {
    /// Geometry only: weights pass through unchanged.
    Off,
    /// Unpolarised flux: the weight is multiplied by the internal
    /// transmittance `exp(−α·L)` of every medium crossed (`alpha_after`) and
    /// by the transmittance (or reflectance, at a mirror) of every
    /// interface, with the same polarisation model as the CPU reference
    /// tracer, so `T + R + A = 1` bookkeeping matches it.
    #[default]
    Unpolarized,
}

/// Number of f32 values recorded per ray per surface with
/// [`RecordMode::AllSurfaces`], in this order: x, y, z, l, m, n, opl_hi,
/// opl_lo, weight.
pub const RECORD_FLOATS: usize = MKL_TRACE_RECORD_FLOATS as usize;

/// A batch of rays to trace through an uploaded system. Build with
/// [`TraceRequest::new`] and adjust the public fields; new fields get
/// defaults, so the struct is `non_exhaustive`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TraceRequest {
    pub systems: SystemHandle,
    pub rays: RayBatch,
    pub start_surface: u32,
    /// Last surface to reach (inclusive), usually the image.
    pub end_surface: u32,
    pub record: RecordMode,
    /// Clip rays at apertures (vignetting). Off for unvignetted analyses.
    pub apply_apertures: bool,
    pub radiometry: Radiometry,
}

impl TraceRequest {
    /// Trace `rays` from `start_surface` to `end_surface` with apertures
    /// applied, unpolarised radiometry and only the final state recorded.
    pub fn new(
        systems: SystemHandle,
        rays: RayBatch,
        start_surface: u32,
        end_surface: u32,
    ) -> Self {
        TraceRequest {
            systems,
            rays,
            start_surface,
            end_surface,
            record: RecordMode::Final,
            apply_apertures: true,
            radiometry: Radiometry::Unpolarized,
        }
    }

    /// Backend-independent checks (surface ranges against a specific system
    /// are checked by the backend against the stored set).
    pub fn validate(&self) -> MkResult<()> {
        self.rays.validate()?;
        if self.start_surface >= self.end_surface {
            return Err(MkError::Backend(
                "trace: start_surface must precede end_surface".into(),
            ));
        }
        if let Some(&s) = self
            .rays
            .system
            .iter()
            .find(|&&s| s >= self.systems.count())
        {
            return Err(MkError::Backend(format!(
                "trace: ray system index {s} out of range"
            )));
        }
        Ok(())
    }
}

/// Trace output (SoA, same order as the input).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct TraceResult {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub l: Vec<f32>,
    pub m: Vec<f32>,
    pub n: Vec<f32>,
    pub opl_hi: Vec<f32>,
    pub opl_lo: Vec<f32>,
    /// Input weight × path transmittance (see [`Radiometry`]); the lost
    /// fraction `1 − weight/input` is the path's reflection + absorption.
    pub weight: Vec<f32>,
    /// `MKL_RAY_*` code | surface << `MKL_RAY_SURFACE_SHIFT`.
    pub status: Vec<u32>,
    /// With [`RecordMode::AllSurfaces`]: `[surface − start][ray][k]` for
    /// `k < RECORD_FLOATS` (x, y, z, l, m, n, opl_hi, opl_lo, weight),
    /// flattened. Empty otherwise.
    pub recorded: Vec<f32>,
    /// With [`RecordMode::AllSurfaces`]: `[surface − start][ray]` status word
    /// after that surface (`MKL_RAY_OK` while the ray is alive). Empty
    /// otherwise.
    pub recorded_status: Vec<u32>,
}

/// Decoded per-ray status word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayStatus {
    Ok,
    Miss { surface: u32 },
    TotalInternalReflection { surface: u32 },
    Vignetted { surface: u32 },
    NonFinite { surface: u32 },
    NotConverged { surface: u32 },
    Unknown { code: u32, surface: u32 },
}

impl RayStatus {
    pub fn decode(word: u32) -> Self {
        let surface = word >> MKL_RAY_SURFACE_SHIFT;
        match word & MKL_RAY_CODE_MASK {
            MKL_RAY_OK => RayStatus::Ok,
            MKL_RAY_MISS => RayStatus::Miss { surface },
            MKL_RAY_TIR => RayStatus::TotalInternalReflection { surface },
            MKL_RAY_VIGNETTED => RayStatus::Vignetted { surface },
            MKL_RAY_NONFINITE => RayStatus::NonFinite { surface },
            MKL_RAY_NOT_CONVERGED => RayStatus::NotConverged { surface },
            code => RayStatus::Unknown { code, surface },
        }
    }
    pub fn encode(self) -> u32 {
        let (code, s) = match self {
            RayStatus::Ok => (MKL_RAY_OK, 0),
            RayStatus::Miss { surface } => (MKL_RAY_MISS, surface),
            RayStatus::TotalInternalReflection { surface } => (MKL_RAY_TIR, surface),
            RayStatus::Vignetted { surface } => (MKL_RAY_VIGNETTED, surface),
            RayStatus::NonFinite { surface } => (MKL_RAY_NONFINITE, surface),
            RayStatus::NotConverged { surface } => (MKL_RAY_NOT_CONVERGED, surface),
            RayStatus::Unknown { code, surface } => (code & MKL_RAY_CODE_MASK, surface),
        };
        code | (s << MKL_RAY_SURFACE_SHIFT)
    }
    pub fn is_ok(self) -> bool {
        self == RayStatus::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrip() {
        for s in [
            RayStatus::Ok,
            RayStatus::Miss { surface: 3 },
            RayStatus::TotalInternalReflection { surface: 17 },
            RayStatus::Vignetted { surface: 1 },
            RayStatus::NonFinite { surface: 0 },
            RayStatus::NotConverged { surface: 999 },
        ] {
            assert_eq!(RayStatus::decode(s.encode()), s);
        }
        assert_eq!(
            RayStatus::decode(MKL_RAY_VIGNETTED | (5 << 8)),
            RayStatus::Vignetted { surface: 5 }
        );
    }

    #[test]
    fn batch_validation() {
        let mut b = RayBatch {
            x: vec![0.0; 2],
            y: vec![0.0; 2],
            z: vec![0.0; 2],
            l: vec![0.0; 2],
            m: vec![0.0; 2],
            n: vec![1.0; 2],
            opl_hi: vec![0.0; 2],
            opl_lo: vec![0.0; 2],
            wavelength: vec![0; 2],
            system: vec![],
            weight: vec![],
        };
        assert!(b.validate().is_ok());
        b.weight = vec![1.0, 0.5];
        assert!(b.validate().is_ok());
        b.system = vec![0];
        assert!(b.validate().is_err());
        b.system = vec![0, 1];
        let req = TraceRequest::new(SystemHandle::new(1, 1, 1), b.clone(), 0, 3);
        assert!(req.validate().is_err(), "system 1 of a 1-system set");
        let mut req = TraceRequest::new(SystemHandle::new(1, 1, 2), b.clone(), 0, 3);
        assert!(req.validate().is_ok());
        assert_eq!(req.radiometry, Radiometry::Unpolarized);
        req.end_surface = 0;
        assert!(req.validate().is_err());
        b.weight = vec![1.0];
        assert!(b.validate().is_err());
    }
}
