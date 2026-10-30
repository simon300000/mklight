//! Deterministic numerical quadrature used to normalise BSDFs and to build
//! their total-integrated-scatter tables.
//!
//! * [`gauss_kronrod`] — globally adaptive Gauss–Kronrod (7/15) on a finite
//!   interval: the interval with the largest error estimate is bisected until
//!   the summed estimate meets the tolerance.
//! * [`periodic_trapezoid`] — trapezoid rule for smooth periodic integrands,
//!   which converges geometrically. The point count doubles (reusing old
//!   points) until two successive estimates agree.

use mklight_core::math::KahanSum;

/// Gauss–Kronrod 15-point abscissae on [-1, 1] (non-negative half).
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
/// Kronrod weights for `XGK`.
const WGK: [f64; 8] = [
    0.022_935_322_010_529_225,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];
/// 7-point Gauss weights for the odd-indexed `XGK` nodes (1, 3, 5, 7).
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// One Gauss–Kronrod 7/15 panel on `[a, b]`: `(kronrod, |kronrod − gauss|)`.
pub fn gk15(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    let mut k = WGK[7] * fc;
    let mut g = WG[3] * fc;
    for j in 0..7 {
        let dx = h * XGK[j];
        let s = f(c - dx) + f(c + dx);
        k += WGK[j] * s;
        if j % 2 == 1 {
            g += WG[j / 2] * s;
        }
    }
    (k * h, ((k - g) * h).abs())
}

/// Result of an adaptive quadrature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadrature {
    pub value: f64,
    pub error: f64,
    pub converged: bool,
}

/// Globally adaptive Gauss–Kronrod on `[a, b]`. Stops when the summed error
/// estimate is below `max(abs_tol, rel_tol·|I|)` or after `max_panels` panels.
pub fn gauss_kronrod(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_panels: usize,
) -> Quadrature {
    if a == b {
        return Quadrature {
            value: 0.0,
            error: 0.0,
            converged: true,
        };
    }
    let (v, e) = gk15(&f, a, b);
    // (a, b, value, error)
    let mut panels = vec![(a, b, v, e)];
    loop {
        let value: KahanSum = panels.iter().map(|p| p.2).collect();
        let error: f64 = panels.iter().map(|p| p.3).sum();
        let value = value.value();
        let tol = abs_tol.max(rel_tol * value.abs());
        if error <= tol || panels.len() >= max_panels {
            return Quadrature {
                value,
                error,
                converged: error <= tol,
            };
        }
        // Bisect the worst panel (first maximum, so the order is deterministic).
        let (worst, _) =
            panels
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(bi, be), (i, p)| {
                    if p.3 > be { (i, p.3) } else { (bi, be) }
                });
        let (pa, pb, _, _) = panels.swap_remove(worst);
        let m = 0.5 * (pa + pb);
        let (v1, e1) = gk15(&f, pa, m);
        let (v2, e2) = gk15(&f, m, pb);
        panels.push((pa, m, v1, e1));
        panels.push((m, pb, v2, e2));
    }
}

/// Trapezoid rule for a `period`-periodic integrand over one period, with
/// point doubling from 8 points up to `2^max_level` points.
pub fn periodic_trapezoid(
    f: impl Fn(f64) -> f64,
    period: f64,
    rel_tol: f64,
    abs_tol: f64,
    max_level: u32,
) -> Quadrature {
    let mut n = 8usize;
    let mut sum: KahanSum = (0..n).map(|i| f(period * i as f64 / n as f64)).collect();
    let mut prev = sum.value() * period / n as f64;
    let mut agreed = 0;
    let mut level = 3;
    loop {
        // New midpoints.
        for i in 0..n {
            sum.add(f(period * (i as f64 + 0.5) / n as f64));
        }
        n *= 2;
        level += 1;
        let cur = sum.value() * period / n as f64;
        let err = (cur - prev).abs();
        if err <= abs_tol.max(rel_tol * cur.abs()) {
            agreed += 1;
            if agreed >= 2 {
                return Quadrature {
                    value: cur,
                    error: err,
                    converged: true,
                };
            }
        } else {
            agreed = 0;
        }
        if level >= max_level {
            return Quadrature {
                value: cur,
                error: err,
                converged: false,
            };
        }
        prev = cur;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn gk_polynomial_exact() {
        // GK15 integrates polynomials up to degree 22 exactly.
        let q = gauss_kronrod(
            |x| x.powi(9) - 3.0 * x * x + 1.0,
            -1.0,
            2.0,
            1e-14,
            1e-14,
            10,
        );
        let exact = (2f64.powi(10) - 1.0) / 10.0 - (8.0 + 1.0) + 3.0;
        assert!((q.value - exact).abs() < 1e-12, "{q:?} vs {exact}");
    }

    #[test]
    fn gk_peaked_integrand() {
        // ∫_0^1 1/(1e-6 + x²)… peaked at 0: arctan form.
        let b = 1e-6f64;
        let q = gauss_kronrod(|x| 1.0 / (b + x * x), 0.0, 1.0, 1e-12, 1e-11, 4000);
        let exact = (1.0 / b.sqrt()).atan() / b.sqrt();
        assert!(q.converged);
        assert!((q.value / exact - 1.0).abs() < 1e-9, "{q:?} {exact}");
    }

    #[test]
    fn trapezoid_periodic() {
        // ∫_0^{2π} exp(cos φ) dφ = 2π I0(1).
        let q = periodic_trapezoid(|p| p.cos().exp(), 2.0 * PI, 1e-14, 0.0, 20);
        let i0_1 = 1.266_065_877_752_008_4;
        assert!((q.value - 2.0 * PI * i0_1).abs() < 1e-13);
    }
}
