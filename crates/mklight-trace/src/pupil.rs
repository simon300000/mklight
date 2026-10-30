//! Deterministic pupil sampling patterns on the unit disk. Every pattern
//! returns points in a fixed order so batch outputs are reproducible.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// A pupil sampling pattern.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PupilSampling {
    /// `n × n` square grid over `[−1, 1]²`, keeping points inside the unit disk
    /// (row-major, `Py` outer, ascending).
    Grid { n: usize },
    /// Centre plus `rings` rings with `6k` points on ring `k` (radius `k/rings`).
    Hexapolar { rings: usize },
    /// `n` points evenly spaced on a circle of radius `radius`, starting at +x.
    Ring { n: usize, radius: f64 },
    /// `n` points from −1 to +1 along `Py` (tangential fan).
    FanY { n: usize },
    /// `n` points from −1 to +1 along `Px` (sagittal fan).
    FanX { n: usize },
}

impl PupilSampling {
    pub fn points(&self) -> Vec<[f64; 2]> {
        match *self {
            PupilSampling::Grid { n } => grid(n),
            PupilSampling::Hexapolar { rings } => hexapolar(rings),
            PupilSampling::Ring { n, radius } => ring(n, radius),
            PupilSampling::FanY { n } => fan(n).into_iter().map(|t| [0.0, t]).collect(),
            PupilSampling::FanX { n } => fan(n).into_iter().map(|t| [t, 0.0]).collect(),
        }
    }
}

fn fan(n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![0.0],
        _ => (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

/// Square grid inside the unit disk.
pub fn grid(n: usize) -> Vec<[f64; 2]> {
    let t = fan(n);
    let mut out = Vec::with_capacity(n * n);
    for &y in &t {
        for &x in &t {
            if x * x + y * y <= 1.0 + 1e-12 {
                out.push([x, y]);
            }
        }
    }
    out
}

/// Hexapolar pattern: `1 + 3 rings (rings + 1)` points.
pub fn hexapolar(rings: usize) -> Vec<[f64; 2]> {
    let mut out = vec![[0.0, 0.0]];
    for k in 1..=rings {
        let r = k as f64 / rings as f64;
        let m = 6 * k;
        for j in 0..m {
            let a = 2.0 * PI * j as f64 / m as f64;
            out.push([r * a.cos(), r * a.sin()]);
        }
    }
    out
}

/// `n` points on a circle of radius `radius`.
pub fn ring(n: usize, radius: f64) -> Vec<[f64; 2]> {
    (0..n)
        .map(|j| {
            let a = 2.0 * PI * j as f64 / n as f64;
            [radius * a.cos(), radius * a.sin()]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_bounds() {
        assert_eq!(hexapolar(3).len(), 1 + 6 + 12 + 18);
        assert!(grid(11).iter().all(|p| p[0].hypot(p[1]) <= 1.0 + 1e-12));
        assert_eq!(grid(3).len(), 5);
        assert_eq!(PupilSampling::FanY { n: 5 }.points()[4], [0.0, 1.0]);
    }
}
