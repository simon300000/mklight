//! Dispersion formulas. λ is in µm and indices are relative to air at the
//! catalog reference conditions (the glass's T_ref, normally 20 °C, and 1 atm),
//! unless a formula says otherwise.

use serde::{Deserialize, Serialize};

use crate::math::KahanSum;

/// A dispersion law `n(λ)`. Each variant's doc comment gives its exact
/// formula. The numbers in brackets are the standard AGF formula numbers that
/// `agf` maps onto each variant (see `docs/MATERIALS.md`).
///
/// Formulas written for `n²` are evaluated as a compensated sum of their terms
/// and then square-rooted. A negative `n²` (outside the physical range of the
/// fit) yields NaN rather than a silently wrong value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "formula", rename_all = "snake_case")]
pub enum Dispersion {
    /// \[1\] Schott: `n² = a0 + a1 λ² + a2 λ⁻² + a3 λ⁻⁴ + a4 λ⁻⁶ + a5 λ⁻⁸`
    Schott { a: [f64; 6] },
    /// `n² − 1 = Σ B_i λ² / (λ² − C_i)` (any number of terms, `C_i` in µm²).
    /// AGF \[2\] Sellmeier1 = 3 terms, \[6\] Sellmeier3 = 4 terms,
    /// \[11\] Sellmeier5 = 5 terms.
    Sellmeier { b: Vec<f64>, c: Vec<f64> },
    /// \[3\] Herzberger: `n = A + B L + C L² + D λ² + E λ⁴ + F λ⁶`, `L = 1/(λ² − 0.028)`.
    Herzberger { a: [f64; 6] },
    /// \[5\] Conrady: `n = n0 + A/λ + B/λ^3.5`.
    Conrady { n0: f64, a: f64, b: f64 },
    /// \[10\] Extended: `n² = a0 + a1 λ² + a2 λ⁻² + a3 λ⁻⁴ + a4 λ⁻⁶ + a5 λ⁻⁸ + a6 λ⁻¹⁰ + a7 λ⁻¹²`.
    Extended { a: [f64; 8] },
    /// Cauchy: `n = A + B/λ² + C/λ⁴`. It has no AGF number of its own and is
    /// exported *exactly* as a Schott \[1\] formula, because
    /// `n² = A² + 2AB λ⁻² + (B² + 2AC) λ⁻⁴ + 2BC λ⁻⁶ + C² λ⁻⁸`.
    Cauchy { a: f64, b: f64, c: f64 },
    /// Tabulated `(λ µm, n)` pairs, interpolated with monotone cubic Hermite.
    Tabulated { points: Vec<(f64, f64)> },
    /// \[4\] Sellmeier2: `n² − 1 = A + B1 λ²/(λ² − λ1²) + B2/(λ² − λ2²)`.
    /// The second term has **no** `λ²` in its numerator (`B2` is in µm²). The
    /// resonance *wavelengths* `λ1, λ2` (µm) are stored, not their squares.
    ///
    /// NOT VERIFIED AGAINST A PRIMARY SPECIFICATION: this form agrees with the
    /// majority of independent open-source AGF readers; at least one reader
    /// puts `λ²` in the second numerator as well. No manufacturer catalog
    /// checked uses formula 4 (`docs/MATERIALS.md`), so there is no published
    /// glass to settle it with a known answer.
    Sellmeier2 {
        a: f64,
        b: [f64; 2],
        lambda_um: [f64; 2],
    },
    /// \[7\] Handbook of Optics 1: `n² = A + B/(λ² − C) − D λ²`.
    Handbook1 { a: f64, b: f64, c: f64, d: f64 },
    /// \[8\] Handbook of Optics 2: `n² = A + B λ²/(λ² − C) − D λ²`.
    Handbook2 { a: f64, b: f64, c: f64, d: f64 },
    /// \[9\] Sellmeier4: `n² = A + B λ²/(λ² − C) + D λ²/(λ² − E)`.
    /// There is no implicit `1 +`, unlike the other Sellmeier forms.
    Sellmeier4 {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
    },
    /// \[12\] Extended2:
    /// `n² = a0 + a1 λ² + a2 λ⁻² + a3 λ⁻⁴ + a4 λ⁻⁶ + a5 λ⁻⁸ + a6 λ⁴ + a7 λ⁶`.
    ///
    /// Verified against a manufacturer catalog (`docs/MATERIALS.md`): every
    /// formula-12 glass reproduces its catalog `n_d`, `V_d` to 5·10⁻⁷ with
    /// `a6` on `λ⁴`. All of them have `a7 = 0`, so the `λ⁶` position of `a7`
    /// rests on secondary documentation.
    Extended2 { a: [f64; 8] },
    /// \[13\] Extended3:
    /// `n² = a0 + a1 λ² + a2 λ⁴ + a3 λ⁻² + a4 λ⁻⁴ + a5 λ⁻⁶ + a6 λ⁻⁸ + a7 λ⁻¹⁰ + a8 λ⁻¹²`.
    ///
    /// Verified against a manufacturer catalog (`docs/MATERIALS.md`): every
    /// formula-13 glass, including those with `a7, a8 ≠ 0`, reproduces its
    /// catalog `n_d`, `V_d` to 5·10⁻⁷.
    Extended3 { a: [f64; 9] },
}

/// Horner evaluation of `Σ_{k≥1} c[k−1] x^k` (no constant term).
#[inline]
fn horner_no_const(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ck| (acc + ck) * x)
}

/// One Sellmeier resonance term, `B λ² / (λ² − C)`.
#[inline]
fn resonance(b: f64, c: f64, l2: f64) -> f64 {
    b * l2 / (l2 - c)
}

/// `√(Σ terms)` with a compensated sum.
#[inline]
fn root_of_sum(terms: &[f64]) -> f64 {
    terms.iter().copied().collect::<KahanSum>().value().sqrt()
}

impl Dispersion {
    /// Index at `lambda_um`, relative to air at the catalog reference conditions.
    pub fn index(&self, lambda_um: f64) -> f64 {
        let l = lambda_um;
        let l2 = l * l;
        let il2 = 1.0 / l2;
        match self {
            Dispersion::Schott { a } => {
                root_of_sum(&[a[0], a[1] * l2, horner_no_const(&a[2..6], il2)])
            }
            Dispersion::Extended { a } => {
                root_of_sum(&[a[0], a[1] * l2, horner_no_const(&a[2..8], il2)])
            }
            Dispersion::Extended2 { a } => root_of_sum(&[
                a[0],
                a[1] * l2,
                horner_no_const(&a[2..6], il2),
                l2 * l2 * (a[6] + a[7] * l2),
            ]),
            Dispersion::Extended3 { a } => root_of_sum(&[
                a[0],
                horner_no_const(&a[1..3], l2),
                horner_no_const(&a[3..9], il2),
            ]),
            Dispersion::Sellmeier { b, c } => {
                let mut s = KahanSum::default();
                s.add(1.0);
                for (b, c) in b.iter().zip(c) {
                    s.add(resonance(*b, *c, l2));
                }
                s.value().sqrt()
            }
            Dispersion::Sellmeier2 { a, b, lambda_um } => {
                // λ² − λi² as (λ − λi)(λ + λi): no cancellation near a resonance.
                let gap = |li: f64| (l - li) * (l + li);
                root_of_sum(&[
                    1.0,
                    *a,
                    b[0] * l2 / gap(lambda_um[0]),
                    b[1] / gap(lambda_um[1]),
                ])
            }
            Dispersion::Sellmeier4 { a, b, c, d, e } => {
                root_of_sum(&[*a, resonance(*b, *c, l2), resonance(*d, *e, l2)])
            }
            Dispersion::Handbook1 { a, b, c, d } => root_of_sum(&[*a, b / (l2 - c), -d * l2]),
            Dispersion::Handbook2 { a, b, c, d } => {
                root_of_sum(&[*a, resonance(*b, *c, l2), -d * l2])
            }
            Dispersion::Herzberger { a } => {
                let lh = 1.0 / (l2 - 0.028);
                [
                    a[0],
                    horner_no_const(&a[1..3], lh),
                    horner_no_const(&a[3..6], l2),
                ]
                .into_iter()
                .collect::<KahanSum>()
                .value()
            }
            Dispersion::Conrady { n0, a, b } => n0 + a / l + b / (l2 * l * l.sqrt()),
            Dispersion::Cauchy { a, b, c } => a + il2 * (b + c * il2),
            Dispersion::Tabulated { points } => monotone_cubic(points, l),
        }
    }

    /// Short human-readable name of the formula.
    pub fn formula_name(&self) -> &'static str {
        match self {
            Dispersion::Schott { .. } => "Schott",
            Dispersion::Sellmeier { b, .. } => match b.len() {
                4 => "Sellmeier3",
                5 => "Sellmeier5",
                _ => "Sellmeier1",
            },
            Dispersion::Herzberger { .. } => "Herzberger",
            Dispersion::Conrady { .. } => "Conrady",
            Dispersion::Extended { .. } => "Extended",
            Dispersion::Cauchy { .. } => "Cauchy",
            Dispersion::Tabulated { .. } => "Tabulated",
            Dispersion::Sellmeier2 { .. } => "Sellmeier2",
            Dispersion::Handbook1 { .. } => "Handbook1",
            Dispersion::Handbook2 { .. } => "Handbook2",
            Dispersion::Sellmeier4 { .. } => "Sellmeier4",
            Dispersion::Extended2 { .. } => "Extended2",
            Dispersion::Extended3 { .. } => "Extended3",
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

/// Schott thermo-optic model for the *absolute* index change from the
/// catalog reference temperature `t_ref_c`: `Δn_abs = (n²−1)/(2n)·(D0ΔT + D1ΔT² + D2ΔT³ + (E0ΔT + E1ΔT²)/(λ² − λ_tk²))`.
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

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (Δ = {:e})", a - b);
    }

    /// Hand-computed values. At λ = 1 µm every power of λ is 1, so `n²` is the
    /// plain sum of the polynomial coefficients (which checks the set of terms,
    /// not their order). At λ = 2 µm the powers are exact binary fractions and
    /// the distinct coefficients pin down the order.
    #[test]
    fn polynomial_forms_known_values() {
        let a6 = [2.0, 0.01, 0.02, 0.003, 0.0004, 0.00005];
        let a8 = [2.0, 0.01, 0.02, 0.003, 0.0004, 0.00005, 0.000006, 0.0000007];
        let a9 = [
            2.0, 0.01, 0.001, 0.02, 0.003, 0.0004, 0.00005, 0.000006, 0.0000007,
        ];
        let one = |a: &[f64]| a.iter().sum::<f64>().sqrt();
        close(Dispersion::Schott { a: a6 }.index(1.0), one(&a6), 1e-15);
        close(Dispersion::Extended { a: a8 }.index(1.0), one(&a8), 1e-15);
        close(Dispersion::Extended2 { a: a8 }.index(1.0), one(&a8), 1e-15);
        close(Dispersion::Extended3 { a: a9 }.index(1.0), one(&a9), 1e-15);

        // λ = 2: λ² = 4, λ⁴ = 16, λ⁶ = 64, λ⁻² = 1/4 …
        let schott2 =
            a6[0] + 4.0 * a6[1] + a6[2] / 4.0 + a6[3] / 16.0 + a6[4] / 64.0 + a6[5] / 256.0;
        close(
            Dispersion::Schott { a: a6 }.index(2.0),
            schott2.sqrt(),
            1e-15,
        );
        let ext2 = a8[0]
            + 4.0 * a8[1]
            + a8[2] / 4.0
            + a8[3] / 16.0
            + a8[4] / 64.0
            + a8[5] / 256.0
            + 16.0 * a8[6]
            + 64.0 * a8[7];
        close(
            Dispersion::Extended2 { a: a8 }.index(2.0),
            ext2.sqrt(),
            1e-15,
        );
        let ext3 = a9[0]
            + 4.0 * a9[1]
            + 16.0 * a9[2]
            + a9[3] / 4.0
            + a9[4] / 16.0
            + a9[5] / 64.0
            + a9[6] / 256.0
            + a9[7] / 1024.0
            + a9[8] / 4096.0;
        close(
            Dispersion::Extended3 { a: a9 }.index(2.0),
            ext3.sqrt(),
            1e-15,
        );
        let ext = a8[0]
            + 4.0 * a8[1]
            + a8[2] / 4.0
            + a8[3] / 16.0
            + a8[4] / 64.0
            + a8[5] / 256.0
            + a8[6] / 1024.0
            + a8[7] / 4096.0;
        close(Dispersion::Extended { a: a8 }.index(2.0), ext.sqrt(), 1e-15);

        // Herzberger at λ = 2 (L = 1/3.972) and λ = 0.5 (L = 1/0.222), with
        // distinct coefficients so any permutation of B…F is detected.
        let h = [1.5, 0.004, 0.0002, -0.001, 0.0001, -0.00001];
        let herz = Dispersion::Herzberger { a: h };
        let lh = 1.0 / 3.972;
        let nh = h[0] + h[1] * lh + h[2] * lh * lh + 4.0 * h[3] + 16.0 * h[4] + 64.0 * h[5];
        close(herz.index(2.0), nh, 1e-15);
        let lh = 1.0 / 0.222;
        let nh = h[0] + h[1] * lh + h[2] * lh * lh + h[3] / 4.0 + h[4] / 16.0 + h[5] / 64.0;
        close(herz.index(0.5), nh, 1e-15);
        // Sellmeier2 at λ = 2: n² = 1 + A + B1·4/(4 − λ1²) + B2/(4 − λ2²).
        let s2 = Dispersion::Sellmeier2 {
            a: 0.1,
            b: [1.0, 0.04],
            lambda_um: [0.1, 1.0],
        };
        close(
            s2.index(2.0),
            (1.0 + 0.1 + 4.0 / 3.99 + 0.04 / 3.0_f64).sqrt(),
            1e-15,
        );
        // Conrady at λ = 4: 4^3.5 = 128.
        let c = Dispersion::Conrady {
            n0: 1.5,
            a: 0.01,
            b: 0.002,
        };
        close(c.index(4.0), 1.5 + 0.01 / 4.0 + 0.002 / 128.0, 1e-15);
        // Cauchy at λ = 0.5: λ⁻² = 4, λ⁻⁴ = 16.
        let cy = Dispersion::Cauchy {
            a: 1.5,
            b: 0.004,
            c: 0.0001,
        };
        close(cy.index(0.5), 1.5 + 0.016 + 0.0016, 1e-15);
    }

    /// The new resonance forms against algebraic identities with the plain
    /// Sellmeier sum, which is checked against catalog N-BK7 data elsewhere.
    #[test]
    fn resonance_forms_identities() {
        // N-BK7.
        let b = [1.03961212, 0.231792344, 1.01046945];
        let c = [0.00600069867, 0.0200179144, 103.560653];
        let s3 = Dispersion::Sellmeier {
            b: b.to_vec(),
            c: c.to_vec(),
        };
        let s2 = Dispersion::Sellmeier {
            b: b[..2].to_vec(),
            c: c[..2].to_vec(),
        };
        let s1 = Dispersion::Sellmeier {
            b: vec![b[0]],
            c: vec![c[0]],
        };
        for i in 0..=50 {
            let l = 0.35 + 2.0 * i as f64 / 50.0;
            let l2 = l * l;
            // Sellmeier2's second term has no λ² in its numerator. Because
            // b λ²/(λ² − c) = b + b c/(λ² − c), a two-term Sellmeier equals
            // Sellmeier2(A = b2, B1 = b1, λ1 = √c1, B2 = b2 c2, λ2 = √c2).
            let d4 = Dispersion::Sellmeier2 {
                a: b[1],
                b: [b[0], b[1] * c[1]],
                lambda_um: [c[0].sqrt(), c[1].sqrt()],
            };
            close(d4.index(l), s2.index(l), 4e-15);
            // With B1 = 0 it is Handbook1(1 + A, B2, λ2², 0); A shifts n² by A.
            let d4h = Dispersion::Sellmeier2 {
                a: 0.05,
                b: [0.0, 0.3],
                lambda_um: [0.0, 0.25],
            };
            let h1 = Dispersion::Handbook1 {
                a: 1.05,
                b: 0.3,
                c: 0.0625,
                d: 0.0,
            };
            close(d4h.index(l), h1.index(l), 2e-15);
            // Sellmeier4 with A = 1 is a two-term Sellmeier.
            let d9 = Dispersion::Sellmeier4 {
                a: 1.0,
                b: b[0],
                c: c[0],
                d: b[1],
                e: c[1],
            };
            close(d9.index(l), s2.index(l), 2e-15);
            // Handbook2 with A = 1, D = 0 is a one-term Sellmeier; D subtracts Dλ².
            let d8 = Dispersion::Handbook2 {
                a: 1.0,
                b: b[0],
                c: c[0],
                d: 0.01,
            };
            close(d8.index(l).powi(2), s1.index(l).powi(2) - 0.01 * l2, 1e-14);
            // B λ²/(λ² − C) = B + B C/(λ² − C), so
            // Handbook2(A, B, C, D) ≡ Handbook1(A + B, B·C, C, D).
            let d7 = Dispersion::Handbook1 {
                a: 1.0 + b[0],
                b: b[0] * c[0],
                c: c[0],
                d: 0.01,
            };
            close(d7.index(l), d8.index(l), 2e-15);
            // Sellmeier with a zero 4th/5th term equals the 3-term form.
            let s5 = Dispersion::Sellmeier {
                b: vec![b[0], b[1], b[2], 0.0, 0.0],
                c: vec![c[0], c[1], c[2], 0.0, 0.0],
            };
            close(s5.index(l), s3.index(l), 2e-15);
        }
    }

    #[test]
    fn negative_n_squared_is_nan() {
        let d = Dispersion::Schott {
            a: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        assert!(d.index(0.5).is_nan());
    }
}
