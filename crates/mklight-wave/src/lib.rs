//! MKLight wave optics: sampled exit-pupil functions, Zernike analysis,
//! PSF (FFT, exact matrix transform, Huygens), MTF (FFT, direct pupil
//! autocorrelation, from PSF), encircled/ensquared energy, line-spread
//! functions and Strehl ratios.
//!
//! Everything here starts from a [`PupilFunction`]: the complex amplitude
//! `A·exp(i2π·W)` on an `n × n` grid over the normalised exit pupil, plus
//! the image-space geometry (exit-pupil radius, reference-sphere radius,
//! image index) that scales image-plane coordinates. [`traced::trace_pupil`]
//! builds it from the reference tracer (`mklight-trace`): reference sphere,
//! OPD, amplitudes and the traced vignetting rim, resampled through
//! [`interp`]. The engine-facing analyses (wavefront map, PSF, MTF,
//! encircled energy, line spread; [`analysis::ANALYSES`]) build on it, and
//! [`detector`] adds the pixel MTF. Design notes: `docs/WAVE.md`.
//!
//! Conventions (see `docs/CONVENTIONS.md`): lengths in mm, wavelengths in
//! µm, OPD in waves with `OPD = OPL_chief − OPL_ray`, Fringe Zernikes by
//! default, frequencies in cycles/mm, tangential = image y on axis.
//! The sampling relations are documented in [`pupil`] and [`psf`]:
//! `Δx = λF#/Q`, alias-free field `n·λF#`, cutoff `1/(λF#)`.
//!
//! Every heavy routine takes a [`mklight_core::JobContext`] (cancellation
//! and progress), runs row- or frequency-parallel with rayon, and reduces
//! in a fixed order, so results are bit-reproducible regardless of thread
//! count.

// `!(x > 0.0)` is used on purpose throughout: unlike `x <= 0.0` it also
// rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod analysis;
pub mod detector;
pub mod energy;
mod fft;
pub mod image_simulation;
pub mod interp;
mod linalg;
pub mod mtf;
pub mod psf;
pub mod pupil;
mod sampling;
pub mod special;
pub mod traced;
pub mod warnings;
pub mod zernike;

pub use detector::{DetectorCurve, DetectorSettings};
pub use traced::{
    AmplitudeModel, ConfiguredTracer, ExitPupilInfo, PupilSettings, ReferencePoint, TracedPupil,
    configured_tracer, trace_pupil, tracer_for_configuration,
};

pub use energy::{
    EnergyCenter, LineSpread, LsfAxis, StrehlReport, encircled_energy, ensquared_energy,
    line_spread_function, strehl_marechal, strehl_marechal_classic, strehl_report,
};
pub use interp::{RaySample, StructuredSamples, grid_from_scattered_zernike, grid_from_structured};
pub use mtf::{
    Defocus, MtfCurve, MtfMethod, MtfOrientation, Otf, diffraction_limited_mtf, mtf, mtf_direct,
    mtf_fft, mtf_from_psf, otf_fft, polychromatic_mtf, through_focus_mtf,
};
pub use psf::{
    HuygensSettings, ImagePlane, ImageSampling, PolyPsfSettings, Psf, PsfNormalization,
    PsfSettings, PsfTransform, fft_psf, huygens_psf, polychromatic_psf,
};
pub use pupil::{Aperture, Apodization, OpdStats, PupilFunction, PupilGeometry, PupilMask};
pub use warnings::WaveWarning;
pub use zernike::{
    SeidelTerms, ZernikeBasis, ZernikeFit, ZernikeFitSettings, ZernikeOrdering, ZernikeSet,
    ZernikeTerm, fit_zernike,
};
