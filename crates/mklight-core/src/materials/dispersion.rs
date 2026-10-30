//! Dispersion formulas. λ is in µm and indices are relative to air at the
//! catalog reference conditions (20 °C, 1 atm), unless a formula says otherwise.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "formula", rename_all = "snake_case")]
pub enum Dispersion {
    /// `n² = a0 + a1 λ² + a2 λ⁻² + a3 λ⁻⁴ + a4 λ⁻⁶ + a5 λ⁻⁸`
    Schott { a: [f64; 6] },
    /// `n² − 1 = Σ B_i λ² / (λ² − C_i)` (any number of terms).
    Sellmeier { b: Vec<f64>, c: Vec<f64> },
    /// Herzberger: `n = A + B L + C L² + D λ² + E λ⁴ + F λ⁶`, `L = 1/(λ² − 0.028)`.
    Herzberger { a: [f64; 6] },
    /// Conrady: `n = n0 + A/λ + B/λ^3.5`.
    Conrady { n0: f64, a: f64, b: f64 },
    /// Extended Schott: `n² = a0 + a1 λ² + a2 λ⁻² + a3 λ⁻⁴ + a4 λ⁻⁶ + a5 λ⁻⁸ + a6 λ⁻¹⁰ + a7 λ⁻¹²`.
    Extended { a: [f64; 8] },
    /// Cauchy: `n = A + B/λ² + C/λ⁴`.
    Cauchy { a: f64, b: f64, c: f64 },
    /// Tabulated `(λ µm, n)` pairs, interpolated with monotone cubic Hermite.
    Tabulated { points: Vec<(f64, f64)> },
}

impl Dispersion {
    pub fn index(&self, lambda_um: f64) -> f64 {
        let l = lambda_um;
        let l2 = l * l;
        match self {
            Dispersion::Schott { a } => {
                let il2 = 1.0 / l2;
                // Horner in λ⁻².
                let neg = il2 * (a[2] + il2 * (a[3] + il2 * (a[4] + il2 * a[5])));
                (a[0] + a[1] * l2 + neg).sqrt()
            }
            Dispersion::Extended { a } => {
                let il2 = 1.0 / l2;
                let neg = il2
                    * (a[2]
                        + il2 * (a[3] + il2 * (a[4] + il2 * (a[5] + il2 * (a[6] + il2 * a[7])))));
                (a[0] + a[1] * l2 + neg).sqrt()
            }
            Dispersion::Sellmeier { b, c } => {
                let s: f64 = b.iter().zip(c).map(|(b, c)| b * l2 / (l2 - c)).sum();
                (1.0 + s).sqrt()
            }
            Dispersion::Herzberger { a } => {
                let lh = 1.0 / (l2 - 0.028);
                a[0] + a[1] * lh + a[2] * lh * lh + a[3] * l2 + a[4] * l2 * l2 + a[5] * l2 * l2 * l2
            }
            Dispersion::Conrady { n0, a, b } => n0 + a / l + b / l.powf(3.5),
            Dispersion::Cauchy { a, b, c } => a + b / l2 + c / (l2 * l2),
            Dispersion::Tabulated { points } => monotone_cubic(points, l),
        }
    }
}

/// Fritsch–Carlson monotone cubic interpolation, clamped at the ends.
pub fn monotone_cubic(p: &[(f64, f64)], x: f64) -> f64 {
    match p.len() {
        0 => f64::NAN,
        1 => p[0].1,
        _ => {
            if x <= p[0].0 {
                return p[0].1;
            }
            if x >= p[p.len() - 1].0 {
                return p[p.len() - 1].1;
            }
            let i = p.partition_point(|q| q.0 <= x) - 1;
            let slope = |j: usize| (p[j + 1].1 - p[j].1) / (p[j + 1].0 - p[j].0);
            let tangent = |j: usize| -> f64 {
                if j == 0 {
                    slope(0)
                } else if j == p.len() - 1 {
                    slope(j - 1)
                } else {
                    let (a, b) = (slope(j - 1), slope(j));
                    // Harmonic mean of the neighbouring secants keeps the interpolant monotone.
                    if a * b <= 0.0 {
                        0.0
                    } else {
                        2.0 * a * b / (a + b)
                    }
                }
            };
            let (x0, y0, x1, y1) = (p[i].0, p[i].1, p[i + 1].0, p[i + 1].1);
            let h = x1 - x0;
            let t = (x - x0) / h;
            let (m0, m1) = (tangent(i) * h, tangent(i + 1) * h);
            let t2 = t * t;
            let t3 = t2 * t;
            (2.0 * t3 - 3.0 * t2 + 1.0) * y0
                + (t3 - 2.0 * t2 + t) * m0
                + (-2.0 * t3 + 3.0 * t2) * y1
                + (t3 - t2) * m1
        }
    }
}

/// Refractive index of standard air relative to vacuum, at temperature `t_c`
/// (°C) and pressure `p_atm` (atm). Uses the Kohlrausch form of Edlén's
/// dispersion at 15 °C with the usual ideal-gas T/P scaling, which is the
/// convention optical design programs and glass catalogs share.
pub fn air_index(lambda_um: f64, t_c: f64, p_atm: f64) -> f64 {
    let l2 = lambda_um * lambda_um;
    let n_ref = 1.0
        + (6432.8 + 2_949_810.0 * l2 / (146.0 * l2 - 1.0) + 25_540.0 * l2 / (41.0 * l2 - 1.0))
            * 1e-8;
    1.0 + (n_ref - 1.0) * p_atm / (1.0 + (t_c - 15.0) * 3.4785e-3)
}

/// Schott thermo-optic model for the *absolute* index change from the 20 °C
/// reference: `Δn_abs = (n²−1)/(2n)·(D0ΔT + D1ΔT² + D2ΔT³ + (E0ΔT + E1ΔT²)/(λ² − λ_tk²))`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThermalCoefficients {
    pub d0: f64,
    pub d1: f64,
    pub d2: f64,
    pub e0: f64,
    pub e1: f64,
    pub lambda_tk_um: f64,
    /// Reference temperature of the catalog data (normally 20 °C).
    #[serde(default = "twenty")]
    pub t_ref_c: f64,
}

fn twenty() -> f64 {
    20.0
}

impl ThermalCoefficients {
    pub fn delta_n_abs(&self, n_abs_ref: f64, lambda_um: f64, t_c: f64) -> f64 {
        let dt = t_c - self.t_ref_c;
        let l2 = lambda_um * lambda_um;
        let lt2 = self.lambda_tk_um * self.lambda_tk_um;
        let denom = l2 - lt2;
        let e_term = if denom.abs() > 1e-12 {
            (self.e0 * dt + self.e1 * dt * dt) / denom
        } else {
            0.0
        };
        (n_abs_ref * n_abs_ref - 1.0) / (2.0 * n_abs_ref)
            * (self.d0 * dt + self.d1 * dt * dt + self.d2 * dt * dt * dt + e_term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_index_reasonable() {
        let n = air_index(0.5875618, 20.0, 1.0);
        assert!((n - 1.000_271_8).abs() < 2e-6, "n_air = {n}");
        assert!(air_index(0.5875618, 20.0, 0.0) == 1.0);
    }

    #[test]
    fn tabulated_monotone() {
        let pts = vec![(0.4, 1.53), (0.5, 1.52), (0.6, 1.515), (0.7, 1.513)];
        let d = Dispersion::Tabulated {
            points: pts.clone(),
        };
        for p in &pts {
            assert!((d.index(p.0) - p.1).abs() < 1e-14);
        }
        let mut prev = f64::INFINITY;
        for i in 0..=300 {
            let n = d.index(0.4 + 0.3 * i as f64 / 300.0);
            assert!(n <= prev + 1e-15);
            prev = n;
        }
    }
}
