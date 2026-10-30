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

/// Input rays (SoA; every vector has the same length).
#[derive(Clone, Debug, Default, PartialEq)]
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
        if lens.iter().any(|&k| k != n) || !(self.system.is_empty() || self.system.len() == n) {
            return Err(MkError::Backend("ray batch arrays differ in length".into()));
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
    /// Position and direction on every surface from start to end (for
    /// layout drawings, footprints and auto semi-diameters).
    AllSurfaces,
}

/// A batch of rays to trace through an uploaded system.
#[derive(Clone, Debug)]
pub struct TraceRequest {
    pub systems: SystemHandle,
    pub rays: RayBatch,
    pub start_surface: u32,
    /// Last surface to reach (inclusive), usually the image.
    pub end_surface: u32,
    pub record: RecordMode,
    /// Clip rays at apertures (vignetting). Off for unvignetted analyses.
    pub apply_apertures: bool,
}

/// Trace output (SoA, same order as the input).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceResult {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub l: Vec<f32>,
    pub m: Vec<f32>,
    pub n: Vec<f32>,
    pub opl_hi: Vec<f32>,
    pub opl_lo: Vec<f32>,
    /// `MKL_RAY_*` code | surface << `MKL_RAY_SURFACE_SHIFT`.
    pub status: Vec<u32>,
    /// With [`RecordMode::AllSurfaces`]: `[surface - start][ray]` positions
    /// and directions, flattened surface-major with 6 values per ray
    /// (x, y, z, l, m, n). Empty otherwise.
    pub recorded: Vec<f32>,
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
        };
        assert!(b.validate().is_ok());
        b.system = vec![0];
        assert!(b.validate().is_err());
    }
}
