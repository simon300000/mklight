//! Detector (pixel) MTF, multiplied into the optical MTF.
//!
//! A pixel integrates the image over its photosensitive aperture, a
//! rectangle of widths `(w_x, w_y)`, and lateral charge diffusion blurs it
//! further. For an image-surface frequency `ν = (ν_x, ν_y)` (cycles/mm):
//!
//! `MTF_det(ν) = |sinc(w_x ν_x)·sinc(w_y ν_y)| · exp(−2π²σ²|ν|²)`,
//! `sinc(t) = sin(πt)/(πt)`,
//!
//! with the diffusion modelled as a Gaussian point spread of standard
//! deviation `σ` (mm). The aperture widths are `pitch·√FF` per axis for a
//! square-shaped fill factor `FF` (area fraction), or given explicitly, so a
//! fully filled pixel (`FF = 1`) has its first zero at `1/pitch`. The
//! sampling **Nyquist** frequency of the pixel lattice along a unit
//! direction `e` is the distance to the edge of the rectangle
//! `|ν_x| ≤ 1/(2p_x)`, `|ν_y| ≤ 1/(2p_y)`:
//! `ν_N(e) = min(1/(2p_x|e_x|), 1/(2p_y|e_y|))`. Above it the sampled image
//! aliases; the MTF is still reported there and the frequency is marked.
//!
//! The system MTF is the product `MTF_optical · MTF_det` (the detector is
//! linear and shift invariant in this model; the phase-dependent sampling
//! MTF is not included).

use mklight_core::{MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::mtf::MtfOrientation;

/// Pixel geometry and diffusion of a detector.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectorSettings {
    /// Pixel pitch `(p_x, p_y)` (mm).
    pub pitch_mm: [f64; 2],
    /// Photosensitive area fraction `0 < FF ≤ 1`; the aperture is a
    /// rectangle of widths `p·√FF` (ignored when `aperture_mm` is set).
    pub fill_factor: f64,
    /// Explicit aperture widths `(w_x, w_y)` (mm), at most the pitch.
    pub aperture_mm: Option<[f64; 2]>,
    /// Standard deviation of the Gaussian charge-diffusion spread (mm).
    pub diffusion_sigma_mm: f64,
}

impl Default for DetectorSettings {
    fn default() -> Self {
        DetectorSettings {
            pitch_mm: [0.005, 0.005],
            fill_factor: 1.0,
            aperture_mm: None,
            diffusion_sigma_mm: 0.0,
        }
    }
}

/// `sin(πt)/(πt)`, with the removable singularity at 0.
pub fn sinc(t: f64) -> f64 {
    let x = std::f64::consts::PI * t;
    if x.abs() < 1e-8 {
        1.0 - x * x / 6.0
    } else {
        x.sin() / x
    }
}

impl DetectorSettings {
    /// Square pixels of pitch `p` (mm), fully filled, without diffusion.
    pub fn square(pitch_mm: f64) -> Self {
        DetectorSettings {
            pitch_mm: [pitch_mm, pitch_mm],
            ..Default::default()
        }
    }

    pub fn validate(&self) -> MkResult<()> {
        let pos = |v: f64| v > 0.0 && v.is_finite();
        let mut ok = self.pitch_mm.iter().all(|&p| pos(p))
            && self.diffusion_sigma_mm >= 0.0
            && self.diffusion_sigma_mm.is_finite();
        match self.aperture_mm {
            Some(w) => {
                ok &= w.iter().all(|&v| pos(v))
                    && w[0] <= self.pitch_mm[0] * (1.0 + 1e-12)
                    && w[1] <= self.pitch_mm[1] * (1.0 + 1e-12);
            }
            None => ok &= self.fill_factor > 0.0 && self.fill_factor <= 1.0,
        }
        if ok {
            Ok(())
        } else {
            Err(MkError::InvalidParam(format!(
                "invalid detector settings {self:?}"
            )))
        }
    }

    /// Photosensitive aperture widths `(w_x, w_y)` (mm).
    pub fn aperture(&self) -> [f64; 2] {
        self.aperture_mm.unwrap_or_else(|| {
            let s = self.fill_factor.sqrt();
            [self.pitch_mm[0] * s, self.pitch_mm[1] * s]
        })
    }

    /// Signed detector OTF at the image-surface frequency `ν` (negative
    /// beyond the first aperture zero: contrast reversal).
    pub fn otf(&self, nu: [f64; 2]) -> f64 {
        let w = self.aperture();
        let s = self.diffusion_sigma_mm;
        let g =
            (-2.0 * std::f64::consts::PI.powi(2) * s * s * (nu[0] * nu[0] + nu[1] * nu[1])).exp();
        sinc(w[0] * nu[0]) * sinc(w[1] * nu[1]) * g
    }

    /// Detector MTF `|OTF|` at `ν`.
    pub fn mtf(&self, nu: [f64; 2]) -> f64 {
        self.otf(nu).abs()
    }

    /// Nyquist frequency `(1/(2p_x), 1/(2p_y))` along the pixel axes.
    pub fn nyquist(&self) -> [f64; 2] {
        [0.5 / self.pitch_mm[0], 0.5 / self.pitch_mm[1]]
    }

    /// Nyquist frequency along the unit direction `e` (module docs).
    pub fn nyquist_along(&self, e: [f64; 2]) -> f64 {
        let n = self.nyquist();
        let f = |nq: f64, c: f64| {
            if c.abs() < 1e-15 {
                f64::INFINITY
            } else {
                nq / c.abs()
            }
        };
        f(n[0], e[0]).min(f(n[1], e[1]))
    }

    /// Detector MTF along the tangential and sagittal directions of
    /// `orientation` at each frequency.
    pub fn curve(&self, freqs: &[f64], orientation: &MtfOrientation) -> DetectorCurve {
        let t = orientation.tangential;
        let s = orientation.sagittal();
        DetectorCurve {
            tangential: freqs
                .iter()
                .map(|&f| self.mtf([f * t[0], f * t[1]]))
                .collect(),
            sagittal: freqs
                .iter()
                .map(|&f| self.mtf([f * s[0], f * s[1]]))
                .collect(),
            nyquist_tangential: self.nyquist_along(t),
            nyquist_sagittal: self.nyquist_along(s),
        }
    }
}

/// Detector MTF slices along one field's tangential and sagittal
/// directions, with the Nyquist frequency of the pixel lattice along each.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorCurve {
    pub tangential: Vec<f64>,
    pub sagittal: Vec<f64>,
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub nyquist_tangential: f64,
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub nyquist_sagittal: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_zeros_and_values() {
        assert_eq!(sinc(0.0), 1.0);
        for k in 1..5 {
            assert!(sinc(k as f64).abs() < 1e-15);
        }
        assert!((sinc(0.5) - 2.0 / std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn nyquist_along_diagonal() {
        let d = DetectorSettings::square(0.004);
        assert!((d.nyquist()[0] - 125.0).abs() < 1e-12);
        let e = [0.5f64.sqrt(), 0.5f64.sqrt()];
        assert!((d.nyquist_along(e) - 125.0 * 2f64.sqrt()).abs() < 1e-9);
        assert!((d.nyquist_along([0.0, 1.0]) - 125.0).abs() < 1e-12);
        assert!(DetectorSettings::square(-1.0).validate().is_err());
        let bad = DetectorSettings {
            aperture_mm: Some([0.005, 0.001]),
            ..DetectorSettings::square(0.004)
        };
        assert!(bad.validate().is_err());
    }
}
