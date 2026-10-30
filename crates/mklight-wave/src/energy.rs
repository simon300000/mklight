//! Encircled and ensquared energy, line-spread functions and Strehl
//! ratios.
//!
//! # Energy integrals
//!
//! The intensity PSF is band-limited, so a PSF sampled at or above Nyquist
//! is represented exactly by the trigonometric interpolant of its samples.
//! Encircled energy is integrated *analytically* through that interpolant:
//! with `F_m` the DFT of the samples,
//! `∫_{|x−q|<r} PSF dA = (1/M²)·Σ_m F_m·e^{2πiν_m·(q−x₀)}·r·J₁(2π|ν_m|r)/|ν_m|`
//! (and `πr²` at `ν = 0`), and the square of half-width `h` uses
//! `Π_axes sin(2πν_a h)/(πν_a)`. This avoids the pixel-staircase error of
//! summing samples inside a circle. Fractions are relative to the PSF's
//! `total_energy` (the energy in the whole plane, from Parseval), so light
//! that falls outside the grid counts as outside every circle.

use mklight_core::{JobContext, MkError, MkResult};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fft::{Dir, fft2, signed_index};
use crate::psf::Psf;
use crate::pupil::PupilFunction;
use crate::special::bessel_j1;

/// Centre of the encircled/ensquared energy integration.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnergyCenter {
    /// The PSF's reference point (`center_mm`).
    #[default]
    Reference,
    /// The intensity centroid.
    Centroid,
    /// The brightest sample.
    Peak,
    /// An explicit image point (mm).
    Point { x: f64, y: f64 },
}

fn resolve_center(psf: &Psf, c: EnergyCenter) -> [f64; 2] {
    match c {
        EnergyCenter::Reference => psf.center_mm,
        EnergyCenter::Centroid => psf.centroid(),
        EnergyCenter::Peak => {
            let (_, i, j) = psf.peak();
            [psf.x(i), psf.y(j)]
        }
        EnergyCenter::Point { x, y } => [x, y],
    }
}

/// Real parts of the phase-shifted spectrum `Re(F_m·e^{2πiν_m·(q−x₀)})/M²`
/// and the bin frequencies `(νx, νy)`.
fn spectrum(psf: &Psf, q: [f64; 2], ctx: &JobContext) -> MkResult<(Vec<f64>, Vec<[f64; 2]>)> {
    let m = psf.size;
    let mut buf: Vec<C64> = psf.data.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft2(&mut buf, m, Dir::Forward, m, ctx)?;
    let dx = psf.spacing_mm;
    let x0 = [psf.x(0), psf.y(0)];
    let dnu = 1.0 / (m as f64 * dx);
    let tau = 2.0 * std::f64::consts::PI;
    let m2 = (m * m) as f64;
    let mut g = Vec::with_capacity(m * m);
    let mut nu = Vec::with_capacity(m * m);
    for j in 0..m {
        let ny = signed_index(j, m) as f64 * dnu;
        for i in 0..m {
            let nx = signed_index(i, m) as f64 * dnu;
            let ph = tau * (nx * (q[0] - x0[0]) + ny * (q[1] - x0[1]));
            g.push((buf[j * m + i] * C64::from_polar(1.0, ph)).re / m2);
            nu.push([nx, ny]);
        }
    }
    Ok((g, nu))
}

fn energy_integral(
    psf: &Psf,
    sizes: &[f64],
    center: EnergyCenter,
    kernel: impl Fn([f64; 2], f64) -> f64 + Sync,
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    if !(psf.total_energy > 0.0) {
        return Err(MkError::Numerical("PSF has no energy".into()));
    }
    let q = resolve_center(psf, center);
    let (g, nu) = spectrum(psf, q, &ctx.subrange(0.0, 0.2))?;
    // F_m holds Σ samples; the integral of the interpolant is in units of
    // samples·mm², i.e. already Σ·(area kernel).
    sizes
        .par_iter()
        .map(|&r| -> MkResult<f64> {
            ctx.check()?;
            if r <= 0.0 {
                return Ok(0.0);
            }
            let s: mklight_core::math::KahanSum = g
                .iter()
                .zip(&nu)
                .map(|(gv, v)| gv * kernel(*v, r))
                .collect();
            Ok(s.value() / psf.total_energy)
        })
        .collect()
}

/// Fraction of the total energy inside circles of the given radii (mm).
pub fn encircled_energy(
    psf: &Psf,
    radii_mm: &[f64],
    center: EnergyCenter,
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    let tau = 2.0 * std::f64::consts::PI;
    energy_integral(
        psf,
        radii_mm,
        center,
        |v, r| {
            let rho = v[0].hypot(v[1]);
            if rho * r < 1e-12 {
                std::f64::consts::PI * r * r
            } else {
                r * bessel_j1(tau * rho * r) / rho
            }
        },
        ctx,
    )
}

/// Fraction of the total energy inside squares of the given half-widths
/// (mm), aligned with the image axes.
pub fn ensquared_energy(
    psf: &Psf,
    half_widths_mm: &[f64],
    center: EnergyCenter,
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    let tau = 2.0 * std::f64::consts::PI;
    let side = |nu: f64, h: f64| {
        if (nu * h).abs() < 1e-12 {
            2.0 * h
        } else {
            (tau * nu * h).sin() / (std::f64::consts::PI * nu)
        }
    };
    energy_integral(
        psf,
        half_widths_mm,
        center,
        |v, h| side(v[0], h) * side(v[1], h),
        ctx,
    )
}

/// Image direction along which a line-spread function is profiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LsfAxis {
    /// Profile along x (a line source parallel to y); its transform is the
    /// MTF at frequencies along x.
    X,
    /// Profile along y.
    Y,
}

/// Line-spread function sampled on the PSF grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineSpread {
    pub positions_mm: Vec<f64>,
    /// Energy per unit length (fraction of total energy per mm).
    pub values: Vec<f64>,
}

/// `LSF(x) = ∫ PSF(x, y) dy / total_energy` (for [`LsfAxis::X`]).
pub fn line_spread_function(psf: &Psf, axis: LsfAxis) -> LineSpread {
    let m = psf.size;
    let dx = psf.spacing_mm;
    let (positions_mm, values) = (0..m)
        .map(|a| {
            let s: mklight_core::math::KahanSum = (0..m)
                .map(|b| match axis {
                    LsfAxis::X => psf.at(a, b),
                    LsfAxis::Y => psf.at(b, a),
                })
                .collect();
            let pos = match axis {
                LsfAxis::X => psf.x(a),
                LsfAxis::Y => psf.y(a),
            };
            (pos, s.value() * dx / psf.total_energy)
        })
        .unzip();
    LineSpread {
        positions_mm,
        values,
    }
}

/// Extended Maréchal approximation `S ≈ exp(−(2πσ)²)`, σ in waves.
pub fn strehl_marechal(rms_waves: f64) -> f64 {
    (-(2.0 * std::f64::consts::PI * rms_waves).powi(2)).exp()
}

/// Classic Maréchal approximation `S ≈ 1 − (2πσ)²`.
pub fn strehl_marechal_classic(rms_waves: f64) -> f64 {
    1.0 - (2.0 * std::f64::consts::PI * rms_waves).powi(2)
}

/// Strehl ratio by several routes, for comparison.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrehlReport {
    /// `|Σp|²/(Σ|p|)²`: exact intensity at the reference point.
    pub at_reference: f64,
    /// PSF sample maximum over the diffraction-limited peak (if a PSF was
    /// given).
    pub psf_peak: Option<f64>,
    /// Weighted RMS wavefront error (waves), piston and tilt removed.
    pub rms_waves: f64,
    /// `exp(−(2πσ)²)` with that RMS (approximates the peak Strehl).
    pub marechal: f64,
}

pub fn strehl_report(pupil: &PupilFunction, psf: Option<&Psf>) -> StrehlReport {
    let rms = pupil.opd_stats(true).rms;
    StrehlReport {
        at_reference: pupil.strehl_at_reference(),
        psf_peak: psf.map(|p| p.strehl()),
        rms_waves: rms,
        marechal: strehl_marechal(rms),
    }
}
