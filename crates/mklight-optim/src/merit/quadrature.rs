//! Gaussian pupil quadrature ("rings and arms") for RMS spot and wavefront
//! merit functions (Forbes, JOSA A 5, 1988).
//!
//! Over the unit (or annular) pupil, polar coordinates with `u = ρ²` turn the
//! area element into `dA = ½ du dθ`, so
//!
//! ```text
//! ∬ f dA = ½ ∫₀^{2π} ∫_{ε²}^{1} f(√u cos θ, √u sin θ) du dθ.
//! ```
//!
//! * **Rings:** `N`-point Gauss–Legendre in `u` on `[ε², 1]`, exact for
//!   polynomials of degree `≤ 2N − 1` in `u = ρ²`.
//! * **Arms:** `M` equally spaced angles with equal weights (the trapezoidal
//!   rule), exact for trigonometric polynomials of order `< M`.
//!
//! A polynomial `pxᵃ pyᵇ` of total degree `d = a + b` is `ρᵈ × (trig.
//! polynomial of order d)` and integrates to zero unless `a`, `b` are even, so
//! `N` rings and `M ≥ d + 1` arms integrate every polynomial of degree
//! `d ≤ 4N − 2` **exactly**. For systems symmetric about the meridional plane
//! (all fields on the y axis, no tilts in x) [`PupilSymmetry::Bilateral`]
//! samples only the half pupil `px ≥ 0` with doubled weights; for functions
//! even in `px` this is equivalent to `2M` full-pupil arms.
//!
//! Weights are normalised to sum to 1, so `Σ wₖ f(pₖ)` is the **pupil
//! average** of `f` — e.g. `Σ wₖ εₖ²` is the squared RMS spot radius.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Gauss–Legendre nodes and weights on `[−1, 1]` (Newton iteration on
/// `Pₙ` from the Tricomi initial guess; accurate to ~1 ulp for `n ≤ 100`).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "Gauss–Legendre needs n ≥ 1");
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        // Root i (descending) initial guess.
        let mut z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            // Three-term recurrence for P_n(z) and P_{n−1}(z).
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            let pn = if n == 1 { z } else { p1 };
            let pnm1 = if n == 1 { 1.0 } else { p0 };
            // P'_n(z) = n (z Pₙ − Pₙ₋₁) / (z² − 1)
            dp = nf * (z * pn - pnm1) / (z * z - 1.0);
            let dz = pn / dp;
            z -= dz;
            if dz.abs() <= 1e-16 * z.abs().max(1.0) {
                break;
            }
        }
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    if n % 2 == 1 {
        x[n / 2] = 0.0;
    }
    (x, w)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PupilSymmetry {
    /// Full pupil.
    #[default]
    None,
    /// Half pupil `px ≥ 0`, weights doubled (meridional-plane symmetry).
    Bilateral,
}

/// Rings-and-arms pupil sampling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilSampling {
    pub rings: usize,
    pub arms: usize,
    #[serde(default)]
    pub symmetry: PupilSymmetry,
    /// Central obscuration ratio `ε` (0 = clear pupil).
    #[serde(default)]
    pub obscuration: f64,
    /// Rotation of the arm pattern (degrees, full-pupil mode only).
    #[serde(default)]
    pub arm_offset_deg: f64,
}

impl Default for PupilSampling {
    fn default() -> Self {
        PupilSampling {
            rings: 3,
            arms: 6,
            symmetry: PupilSymmetry::None,
            obscuration: 0.0,
            arm_offset_deg: 0.0,
        }
    }
}

/// One pupil sample: normalised pupil coordinates and quadrature weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilNode {
    pub px: f64,
    pub py: f64,
    pub weight: f64,
}

impl PupilSampling {
    pub fn bilateral(rings: usize, arms: usize) -> Self {
        PupilSampling {
            rings,
            arms,
            symmetry: PupilSymmetry::Bilateral,
            ..Default::default()
        }
    }

    /// Highest total polynomial degree integrated exactly.
    pub fn exact_degree(&self) -> usize {
        let radial = 4 * self.rings - 2;
        let angular = match self.symmetry {
            PupilSymmetry::None => self.arms.saturating_sub(1),
            PupilSymmetry::Bilateral => (2 * self.arms).saturating_sub(1),
        };
        radial.min(angular)
    }

    /// Nodes and weights (weights sum to 1).
    pub fn nodes(&self) -> Vec<PupilNode> {
        let rings = self.rings.max(1);
        let arms = self.arms.max(1);
        let e2 = self.obscuration.clamp(0.0, 0.999_999).powi(2);
        let (t, wt) = gauss_legendre(rings);
        let mut out = Vec::with_capacity(rings * arms);
        // Radial: u ∈ [ε², 1]; Σ wᵤ = 1 − ε². Angular weights 1/M → the
        // product sums to (1 − ε²); divide by it for a normalised average.
        let half = 0.5 * (1.0 - e2);
        for (ti, wi) in t.iter().zip(&wt) {
            let u = e2 + half * (ti + 1.0);
            let rho = u.sqrt();
            let wr = half * wi / (1.0 - e2);
            for j in 0..arms {
                let theta = match self.symmetry {
                    PupilSymmetry::None => {
                        self.arm_offset_deg.to_radians() + 2.0 * PI * j as f64 / arms as f64
                    }
                    PupilSymmetry::Bilateral => -0.5 * PI + PI * (j as f64 + 0.5) / arms as f64,
                };
                let (s, c) = theta.sin_cos();
                out.push(PupilNode {
                    px: rho * c,
                    py: rho * s,
                    weight: wr / arms as f64,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_known_values() {
        let (x, w) = gauss_legendre(2);
        assert!((x[1] - 1.0 / 3f64.sqrt()).abs() < 1e-15 && (x[0] + x[1]).abs() < 1e-16);
        assert!((w[0] - 1.0).abs() < 1e-15);
        let (x, w) = gauss_legendre(3);
        assert!((x[2] - 0.6f64.sqrt()).abs() < 1e-15 && x[1] == 0.0);
        assert!((w[1] - 8.0 / 9.0).abs() < 1e-15 && (w[0] - 5.0 / 9.0).abs() < 1e-15);
        // Exactness for t^k, k ≤ 2n − 1, for n up to 40.
        for n in [1, 4, 7, 12, 40] {
            let (x, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
            for k in 0..2 * n {
                let q: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).sum();
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k as f64 + 1.0)
                };
                assert!((q - exact).abs() < 1e-13, "n={n} k={k}: {q} vs {exact}");
            }
        }
    }
}
