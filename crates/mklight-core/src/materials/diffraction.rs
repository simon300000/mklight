//! Thin-element scalar efficiencies; see SURFACES §3 for the Fourier derivation.
use crate::rng::Philox;
use crate::{ResolvedSystem, SurfaceShape};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DiffractionEfficiency {
    #[default]
    Ideal,
    Kinoform,
    Multilevel {
        levels: std::num::NonZeroU16,
    },
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        // Reduce before sin: integer orders must have exactly zero amplitude.
        let r = x - x.round();
        (PI * r).sin() / (PI * x)
    }
}

impl DiffractionEfficiency {
    /// Fraction of the coating's outgoing channel in order m. `blaze` is αm₀.
    pub fn efficiency(self, order: i32, selected: i32, blaze: f64) -> f64 {
        let m = order as f64;
        match self {
            Self::Ideal => f64::from(order == selected),
            Self::Kinoform => sinc(m - blaze).powi(2),
            Self::Multilevel { levels } => {
                let n = levels.get() as f64;
                // Finite geometric sum, reduced around its removable singularities.
                let r = (blaze - m) / n;
                let r = r - r.round();
                (sinc(m / n) * sinc(n * r) / sinc(r)).powi(2)
            }
        }
    }

    /// Sample the entire integer spectrum, without truncating or renormalising
    /// a finite order window. Rejection uses a rounded Cauchy envelope.
    pub fn sample(self, selected: i32, blaze: f64, rng: &mut Philox) -> i32 {
        if self == Self::Ideal {
            return selected;
        }
        if self == Self::Kinoform && blaze == blaze.round() {
            return blaze as i32;
        }
        if let Self::Multilevel { levels } = self {
            let n = levels.get() as i32;
            // The residue spectrum is the aliased continuous-ramp spectrum:
            // P(r)=Σ_l sinc²(r+lN−b)=|Dirichlet_N((b−r)/N)|².
            // Given r, P(k|r)=sinc²(k+r/N), m=r+kN. Two kinoform
            // draws therefore sample the staircase exactly, at O(1) cost.
            loop {
                let r = Self::Kinoform.sample(selected, blaze, rng).rem_euclid(n);
                let k = Self::Kinoform.sample(selected, -(r as f64) / n as f64, rng);
                if let Some(m) = k.checked_mul(n).and_then(|m| m.checked_add(r)) {
                    return m;
                }
            }
        }
        let center = blaze.round() as i32;
        loop {
            let k = ((PI * (rng.uniform() - 0.5)).tan()).round() as i32;
            let Some(m) = center.checked_add(k) else {
                continue;
            };
            let x = k as f64;
            let q = if k == 0 {
                2.0 * 0.5f64.atan() / PI
            } else {
                1.0f64.atan2((x - 0.5) * (x + 0.5) + 1.0) / PI
            };
            if rng.uniform() * 4.0 * q < self.efficiency(m, selected, blaze) {
                return m;
            }
        }
    }
}

/// Blaze αm₀ of an outgoing channel at λ (SURFACES §3.1): the optical path
/// the relief adds per unit depth on that channel, over the path it adds on
/// the designed channel at λ₀. A transmitting relief adds n_before − n_after
/// in either direction; its Fresnel reflection adds +2n from the medium
/// before and −2n from the medium after; a designed mirror adds 2n.
pub fn blaze(
    sys: &ResolvedSystem,
    surface: usize,
    lambda: f64,
    reflected: bool,
    backward: bool,
) -> f64 {
    let s = &sys.surfaces[surface];
    let SurfaceShape::Diffractive { phase, .. } = &s.shape else {
        return 0.0;
    };
    let before = &sys.surfaces[surface - 1].medium_after;
    let after = &s.medium_after;
    let l0 = phase.design_wavelength_um;
    let (path, design) = if s.is_mirror() {
        let n = if backward { after } else { before };
        (n.index(lambda), before.index(l0))
    } else {
        let contrast = |l| before.index(l) - after.index(l);
        let path = match (reflected, backward) {
            (false, _) => contrast(lambda),
            (true, false) => 2.0 * before.index(lambda),
            (true, true) => -2.0 * after.index(lambda),
        };
        (path, contrast(l0))
    };
    // A same-medium phase screen has no relief depth: use its
    // nondispersive limit rather than an undefined 0/0.
    let ratio = if design == 0.0 { 1.0 } else { path / design };
    phase.design_order as f64 * l0 / lambda * ratio
}

/// η of `order` (`None`: the trace order) on one channel of a surface; 1 for
/// a surface that is not diffractive.
pub fn order_efficiency(
    sys: &ResolvedSystem,
    surface: usize,
    wl: usize,
    order: Option<i32>,
    reflected: bool,
    backward: bool,
) -> f64 {
    let SurfaceShape::Diffractive { phase, .. } = &sys.surfaces[surface].shape else {
        return 1.0;
    };
    sys.system().surfaces[surface]
        .diffraction_efficiency
        .efficiency(
            order.unwrap_or(phase.trace_order),
            phase.trace_order,
            blaze(sys, surface, sys.wavelengths_um[wl], reflected, backward),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_known_answers_and_parseval() {
        let binary = DiffractionEfficiency::Multilevel {
            levels: 2.try_into().unwrap(),
        };
        let four = DiffractionEfficiency::Multilevel {
            levels: 4.try_into().unwrap(),
        };
        assert!((binary.efficiency(1, 1, 1.0) - 4.0 / PI.powi(2)).abs() < 1e-15);
        assert!((four.efficiency(1, 1, 1.0) - 8.0 / PI.powi(2)).abs() < 1e-15);
        assert_eq!(DiffractionEfficiency::Kinoform.efficiency(3, 3, 3.0), 1.0);
        for model in [DiffractionEfficiency::Kinoform, binary, four] {
            for b in [-2.3, 0.0, 1.0, 1.27, 3.0] {
                let sum: f64 = (-10000..=10000).map(|m| model.efficiency(m, 1, b)).sum();
                assert!(sum <= 1.0 + 2e-13 && sum > 0.9998, "{model:?} {b}: {sum}");
            }
        }
    }
    #[test]
    fn sampled_orders_follow_fourier_power() {
        for model in [
            DiffractionEfficiency::Kinoform,
            DiffractionEfficiency::Multilevel {
                levels: 4.try_into().unwrap(),
            },
        ] {
            let mut rng = Philox::new(73, 0);
            let mut counts = [0; 9];
            let n = 100000;
            for _ in 0..n {
                let m = model.sample(1, 1.27, &mut rng);
                if (-4..=4).contains(&m) {
                    counts[(m + 4) as usize] += 1;
                }
            }
            for m in -4..=4 {
                let p = model.efficiency(m, 1, 1.27);
                let se = (p * (1.0 - p) / n as f64).sqrt();
                assert!((counts[(m + 4) as usize] as f64 / n as f64 - p).abs() < 5.0 * se + 1e-5);
            }
        }
    }
    #[test]
    fn blaze_matches_facet_optics_and_orders_close_energy() {
        use crate::{Interaction, MaterialRef, Snapshot, Vec3};
        use std::sync::Arc;
        let resolve = |s| {
            let snap = Snapshot::new(s).unwrap();
            ResolvedSystem::new(
                &snap,
                Arc::new(super::super::MaterialLibrary::with_builtin()),
            )
            .unwrap()
        };
        let mut s = crate::samples::hybrid_diffractive_achromat();
        let SurfaceShape::Diffractive { phase, .. } = &s.surfaces[2].shape else {
            unreachable!()
        };
        let l0 = phase.design_wavelength_um;
        // Glass in air: α = (λ₀/λ)(n(λ)−1)/(n(λ₀)−1).
        let r = resolve(s.clone());
        let n = |l| r.surfaces[1].medium_after.index(l);
        for l in [0.4861327, 0.6562725] {
            let alpha = l0 / l * (n(l) - 1.0) / (n(l0) - 1.0);
            assert!((blaze(&r, 2, l, false, false) - alpha).abs() < 1e-14);
        }
        // Geometric limit: a facet z = εx between n_b (z < 0) and n_a blazes
        // each channel into the order whose grating momentum equals the
        // facet's specular kick; the transmitted channel defines order m₀ = 1.
        for (nb, na) in [(1.5, 1.0), (1.0, 1.5)] {
            let mut facet = s.clone();
            facet.surfaces[1].material = MaterialRef::Constant { n: nb };
            facet.surfaces[2].material = MaterialRef::Constant { n: na };
            let r = resolve(facet);
            let f = Vec3::new(-1e-7, 0.0, 1.0).normalized();
            let reflect = |k: Vec3| k - f * (2.0 * k.dot(f));
            let tangent = Vec3::Z - f * f.dot(Vec3::Z);
            let t =
                tangent * (nb / na) + f * (1.0 - (nb / na).powi(2) * tangent.dot(tangent)).sqrt();
            let kick = na * t.x;
            let forward = nb * reflect(Vec3::Z).x / kick;
            let backward = na * reflect(-Vec3::Z).x / kick;
            assert!((blaze(&r, 2, l0, false, false) - 1.0).abs() < 1e-15);
            assert!((blaze(&r, 2, l0, true, false) - forward).abs() < 1e-6);
            assert!((blaze(&r, 2, l0, true, true) - backward).abs() < 1e-6);
        }
        // Σ_m (T η_T + R η_R) + A = 1 up to the omitted sinc² tail.
        s.surfaces[2].diffraction_efficiency = DiffractionEfficiency::Kinoform;
        let r = resolve(s.clone());
        let model = DiffractionEfficiency::Kinoform;
        for lambda in [l0, 0.45, 0.7] {
            let bt = blaze(&r, 2, lambda, false, false);
            let br = blaze(&r, 2, lambda, true, true);
            let outgoing: f64 = (-10000..=10000)
                .map(|m| 0.8 * model.efficiency(m, 1, bt) + 0.1 * model.efficiency(m, 1, br))
                .sum();
            assert!(outgoing + 0.1 <= 1.0 + 1e-13 && outgoing + 0.1 > 0.99997);
        }
        // A cemented relief reverses its phase where the two dispersions cross.
        let mut crossing = s.clone();
        crossing.surfaces[1].material = MaterialRef::Constant { n: 1.52 };
        crossing.surfaces[2].material = MaterialRef::glass("N-BK7");
        assert!(blaze(&resolve(crossing), 2, 0.4861327, false, false) < 0.0);
        // A designed mirror relief scales with the incident medium's index.
        s.surfaces[2].interaction = Interaction::Reflect;
        let r = resolve(s);
        let n = |l| r.surfaces[1].medium_after.index(l);
        assert!((blaze(&r, 2, l0, true, false) - 1.0).abs() < 1e-15);
        let expected = 0.5 * n(2.0 * l0) / n(l0);
        assert!((blaze(&r, 2, l0 * 2.0, true, false) - expected).abs() < 1e-15);
    }
}
