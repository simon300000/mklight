//! Special functions used by the analytic references and the energy
//! integrals: Bessel functions of the first kind, the Airy pattern and its
//! encircled energy, and Gauss–Legendre quadrature nodes.

use std::f64::consts::PI;

/// `J_n(x)` from Bessel's integral `J_n(x) = (1/π)∫₀^π cos(nτ − x sin τ) dτ`.
///
/// The integrand extends to a smooth 2π-periodic function, so the
/// trapezoidal rule converges geometrically once the node count exceeds
/// `~|x|/2 + n/2`. With the margin used here the result is accurate to a few
/// ulp for `|x| ≲ 10³`. Slow but exact; used as the reference.
pub fn bessel_jn(n: u32, x: f64) -> f64 {
    let k = 2 * ((x.abs() + n as f64) as usize / 2 + 24);
    // Trapezoid over the full period [0, 2π) with k nodes; the integrand is
    // even in τ around π, so this equals the [0, π] integral.
    let h = 2.0 * PI / k as f64;
    let mut s = 0.0;
    for i in 0..k {
        let t = i as f64 * h;
        s += (n as f64 * t - x * t.sin()).cos();
    }
    s / k as f64
}

/// Fast `J₁(x)` (Abramowitz & Stegun 9.4.4 / 9.4.6, |error| < 1·10⁻⁷).
/// Used in the inner loops of the energy integrals.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 3.0 {
        let t = (x / 3.0) * (x / 3.0);
        let p = 0.5
            + t * (-0.562_499_85
                + t * (0.210_935_73
                    + t * (-0.039_542_89
                        + t * (0.004_433_19 + t * (-0.000_317_61 + t * 0.000_011_09)))));
        x * p
    } else {
        let y = 3.0 / ax;
        let f1 = 0.797_884_56
            + y * (0.000_001_56
                + y * (0.016_596_67
                    + y * (0.000_171_05
                        + y * (-0.002_495_11 + y * (0.001_136_53 - y * 0.000_200_33)))));
        let th = ax - 2.356_194_49
            + y * (0.124_996_12
                + y * (0.000_056_50
                    + y * (-0.006_378_79
                        + y * (0.000_743_48 + y * (0.000_798_24 - y * 0.000_291_66)))));
        let r = f1 * th.cos() / ax.sqrt();
        if x < 0.0 { -r } else { r }
    }
}

/// `2·J₁(x)/x`, the Airy amplitude, with the removable singularity at 0.
pub fn jinc(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        1.0 - x * x / 8.0
    } else {
        2.0 * bessel_jn(1, x) / x
    }
}

/// Normalised intensity of an annular pupil's diffraction pattern (outer
/// radius 1, obscuration ratio `eps`) at the reduced radius
/// `v = π·r/(λ·F#)`; equal to 1 at `v = 0`. `eps = 0` is the Airy pattern
/// `[2J₁(v)/v]²`.
pub fn annular_airy(v: f64, eps: f64) -> f64 {
    let a = jinc(v) - eps * eps * jinc(eps * v);
    let n = 1.0 - eps * eps;
    (a / n) * (a / n)
}

/// Encircled energy of the Airy pattern within reduced radius `v`
/// (Rayleigh): `1 − J₀²(v) − J₁²(v)`.
pub fn airy_encircled_energy(v: f64) -> f64 {
    let j0 = bessel_jn(0, v);
    let j1 = bessel_jn(1, v);
    1.0 - j0 * j0 - j1 * j1
}

/// Gauss–Legendre nodes and weights on [−1, 1] (exact for polynomials of
/// degree ≤ 2n − 1). Newton iteration on `P_n` from the Chebyshev guess.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(z), p0 = P_{n−1}(z).
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_known_values() {
        // Tabulated values (A&S table 9.1) and the first zeros.
        assert!((bessel_jn(0, 1.0) - 0.765_197_686_557_966_6).abs() < 1e-14);
        assert!((bessel_jn(1, 1.0) - 0.440_050_585_744_933_5).abs() < 1e-14);
        assert!((bessel_jn(1, 10.0) - 0.043_472_746_168_861_44).abs() < 1e-14);
        assert!(bessel_jn(1, 3.831_705_970_207_512).abs() < 1e-14);
        assert!(bessel_jn(0, 2.404_825_557_695_773).abs() < 1e-14);
    }

    #[test]
    fn fast_j1_matches_integral() {
        let mut x = -40.0;
        while x < 200.0 {
            assert!((bessel_j1(x) - bessel_jn(1, x)).abs() < 2e-7, "x = {x}");
            x += 0.173;
        }
    }

    #[test]
    fn airy_encircled_energy_first_ring() {
        // 83.8 % inside the first dark ring.
        let ee = airy_encircled_energy(3.831_705_970_207_512);
        assert!((ee - 0.8378).abs() < 1e-4);
    }

    #[test]
    fn gauss_legendre_exact() {
        let (x, w) = gauss_legendre(7);
        let s: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(12)).sum();
        assert!((s - 2.0 / 13.0).abs() < 1e-14);
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
    }
}
