//! Random variates on top of the Philox4x32-10 stream (CONVENTIONS §6).
//!
//! Every stochastic method derives its generators from an explicit user seed
//! and a *stream id* that encodes what the numbers are for (generation,
//! individual, Monte-Carlo trial…), so results never depend on thread
//! scheduling or on how many numbers another component drew.

use mklight_core::rng::Philox;

/// Stream-id namespaces, so different uses of one seed never overlap.
pub mod streams {
    pub const DE: u64 = 0x01 << 56;
    pub const CMAES: u64 = 0x02 << 56;
    pub const HAMMER: u64 = 0x03 << 56;
    pub const DISCRETE: u64 = 0x04 << 56;
    pub const NELDER_MEAD: u64 = 0x05 << 56;
    pub const TOLERANCE: u64 = 0x06 << 56;
    pub const TOLERANCE_ORIENTATION: u64 = 0x07 << 56;
}

/// A Philox stream with convenience draws.
#[derive(Clone, Debug)]
pub struct Rng {
    inner: Philox,
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        Rng {
            inner: Philox::new(seed, stream),
            spare_normal: None,
        }
    }
    /// Uniform in the open interval (0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.inner.uniform()
    }
    /// Uniform in `[a, b)` (a < b).
    #[inline]
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }
    /// Uniform integer in `0..n` (n > 0). Uses the top 24 bits of a draw, so
    /// the bias is below `n / 2²⁴`.
    #[inline]
    pub fn index(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }
    /// Standard normal by the Box–Muller transform (both variates used).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let rad = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare_normal = Some(rad * s);
        rad * c
    }
    /// Standard normal truncated to `[−k, k]` by rejection (exact
    /// distribution; acceptance ≥ 68 % for `k ≥ 1`). For very small `k` it
    /// falls back to a uniform draw after 1000 rejections.
    pub fn truncated_normal(&mut self, k: f64) -> f64 {
        let k = k.abs();
        for _ in 0..1000 {
            let z = self.normal();
            if z.abs() <= k {
                return z;
            }
        }
        self.range(-k, k)
    }
    /// `k` distinct indices from `0..n`, excluding `exclude` (partial
    /// Fisher–Yates on a small scratch vector).
    pub fn distinct(&mut self, n: usize, k: usize, exclude: &[usize]) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).filter(|i| !exclude.contains(i)).collect();
        let k = k.min(pool.len());
        for i in 0..k {
            let j = i + self.index(pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_moments() {
        let mut r = Rng::new(7, 1);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let z = r.normal();
            s1 += z;
            s2 += z * z;
        }
        let mean = s1 / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
    }

    #[test]
    fn truncated_and_distinct() {
        let mut r = Rng::new(1, 2);
        for _ in 0..1000 {
            assert!(r.truncated_normal(1.5).abs() <= 1.5);
        }
        let d = r.distinct(10, 3, &[4]);
        assert_eq!(d.len(), 3);
        assert!(!d.contains(&4));
        assert!(d[0] != d[1] && d[1] != d[2] && d[0] != d[2]);
    }

    #[test]
    fn streams_are_reproducible() {
        let a: Vec<f64> = {
            let mut r = Rng::new(5, streams::DE | 3);
            (0..10).map(|_| r.normal()).collect()
        };
        let b: Vec<f64> = {
            let mut r = Rng::new(5, streams::DE | 3);
            (0..10).map(|_| r.normal()).collect()
        };
        assert_eq!(a, b);
    }
}
