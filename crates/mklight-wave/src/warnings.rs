//! Sampling diagnostics attached to wave-optics results.
//!
//! A warning never changes the numbers; it says which of them cannot be
//! trusted. Front ends show them, and `RunInfo::warnings` carries their
//! `Display` text.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A sampling or conditioning problem detected while computing a result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaveWarning {
    /// The image-plane spacing exceeds the intensity Nyquist limit
    /// `λ·F#/2` (padding factor < 2): the PSF is aliased and the MTF above
    /// `1/(2Δx)` is wrong.
    ImageUndersampled { spacing_mm: f64, nyquist_mm: f64 },
    /// The requested image grid is wider than the alias-free field
    /// `N·λ·F#` set by the pupil sampling, so the PSF repeats periodically
    /// inside the grid.
    FieldWraps { grid_mm: f64, alias_free_mm: f64 },
    /// The wavefront phase changes by more than half a wave between adjacent
    /// pupil samples, so light lands outside the alias-free field and wraps
    /// back in. Use more pupil samples.
    PupilUndersampled { max_step_waves: f64 },
    /// Too few pupil samples across the diameter for a meaningful result.
    FewPupilSamples { samples: usize, recommended: usize },
    /// The traced ray lattice is too coarse for the wavefront's curvature:
    /// linear interpolation between neighbouring rays misses the wavefront
    /// by up to `interpolation_error_waves` (limit
    /// [`crate::interp::LATTICE_INTERPOLATION_LIMIT_WAVES`]), so the
    /// resampled pupil, and the PSF, Strehl and MTF from it, are biased.
    /// Typical of fast aspheric systems with strong high-order
    /// aberration. Trace more rays (`pupil.rays`).
    LatticeUndersampled {
        interpolation_error_waves: f64,
        rays: usize,
        recommended_rays: usize,
    },
    /// The image space is afocal (paraxial image-space NA `image_na` ≈ 0):
    /// the OPD reference sphere about the chief ray's image point turns
    /// the collimated beam into a large spurious defocus, so OPD, Zernike,
    /// PSF, Strehl and MTF describe that, not the system. There is no
    /// plane-wave (afocal) reference yet; focus the beam with an ideal
    /// (paraxial) lens in front of the image surface.
    AfocalImageSpace { image_na: f64 },
    /// The FFT size was rounded, so the image spacing differs from the
    /// request.
    SpacingAdjusted { requested_mm: f64, actual_mm: f64 },
    /// A least-squares fit is poorly conditioned.
    IllConditioned { condition: f64, limit: f64 },
    /// The requested frequency exceeds what the sampling can represent; the
    /// value returned there is zero or extrapolated.
    FrequencyBeyondSampling { frequency: f64, limit: f64 },
    /// Part of the clear aperture lies far from every traced ray (distance
    /// `max_gap` against a typical ray spacing `typical_spacing`, both in
    /// normalised pupil units), so the wavefront there is extrapolated.
    SparseRays { max_gap: f64, typical_spacing: f64 },
    /// Only `energy_in_grid` of the PSF's energy falls inside the image
    /// grid: the PSF is truncated (spot larger than the grid), so
    /// centroids, encircled energies and an MTF taken from it are biased.
    /// Enlarge the grid or its spacing.
    PsfTruncated { energy_in_grid: f64 },
}

impl fmt::Display for WaveWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveWarning::ImageUndersampled {
                spacing_mm,
                nyquist_mm,
            } => write!(
                f,
                "image sampling {spacing_mm:.3e} mm is coarser than the Nyquist limit \
                 {nyquist_mm:.3e} mm; the PSF is aliased"
            ),
            WaveWarning::FieldWraps {
                grid_mm,
                alias_free_mm,
            } => write!(
                f,
                "image grid {grid_mm:.3e} mm exceeds the alias-free field {alias_free_mm:.3e} mm \
                 set by the pupil sampling; the PSF wraps around"
            ),
            WaveWarning::PupilUndersampled { max_step_waves } => write!(
                f,
                "wavefront changes by up to {max_step_waves:.2} waves between pupil samples \
                 (limit 0.5); increase the pupil sampling"
            ),
            WaveWarning::FewPupilSamples {
                samples,
                recommended,
            } => write!(
                f,
                "only {samples} pupil samples across the diameter (recommended ≥ {recommended})"
            ),
            WaveWarning::LatticeUndersampled {
                interpolation_error_waves,
                rays,
                recommended_rays,
            } => write!(
                f,
                "pupil.rays = {rays} under-resolves the wavefront (linear interpolation \
                 error up to {interpolation_error_waves:.3} waves); PSF, Strehl and MTF are \
                 biased; use pupil.rays ≥ {recommended_rays}"
            ),
            WaveWarning::AfocalImageSpace { image_na } => write!(
                f,
                "image space is afocal (paraxial image-space NA {image_na:.1e}): the reference \
                 sphere about the chief ray's image point makes the collimated beam a large \
                 defocus, so OPD, Zernike, PSF, Strehl and MTF are not meaningful; focus the \
                 beam with an ideal (paraxial) lens before the image surface"
            ),
            WaveWarning::SpacingAdjusted {
                requested_mm,
                actual_mm,
            } => write!(
                f,
                "image spacing adjusted from {requested_mm:.4e} mm to {actual_mm:.4e} mm \
                 to reach an efficient FFT size"
            ),
            WaveWarning::IllConditioned { condition, limit } => write!(
                f,
                "least-squares fit is ill-conditioned (condition {condition:.3e} > {limit:.1e})"
            ),
            WaveWarning::FrequencyBeyondSampling { frequency, limit } => write!(
                f,
                "frequency {frequency:.4} cycles/mm is beyond the sampled range {limit:.4} cycles/mm"
            ),
            WaveWarning::SparseRays {
                max_gap,
                typical_spacing,
            } => write!(
                f,
                "parts of the aperture are {max_gap:.3} (pupil units) from the nearest ray, \
                 against a ray spacing of {typical_spacing:.3}; the wavefront there is extrapolated"
            ),
            WaveWarning::PsfTruncated { energy_in_grid } => write!(
                f,
                "only {:.1} % of the PSF energy lies inside the image grid; the PSF is \
                 truncated (enlarge the grid or its spacing)",
                100.0 * energy_in_grid
            ),
        }
    }
}

/// Paraxial image-space NA below which the image space counts as afocal
/// (working F/# above 5·10⁸).
pub const AFOCAL_IMAGE_NA: f64 = 1e-9;

/// [`WaveWarning::AfocalImageSpace`] for a first-order solution whose
/// image-space marginal ray is (numerically) parallel to the axis.
pub fn afocal_warning(fo: &mklight_trace::paraxial::FirstOrder) -> Option<WaveWarning> {
    (!(fo.image_na > AFOCAL_IMAGE_NA)).then_some(WaveWarning::AfocalImageSpace {
        image_na: fo.image_na,
    })
}

/// Push `w` unless an equal-kind warning is already present.
pub(crate) fn push_unique(list: &mut Vec<WaveWarning>, w: WaveWarning) {
    if !list
        .iter()
        .any(|x| std::mem::discriminant(x) == std::mem::discriminant(&w))
    {
        list.push(w);
    }
}
