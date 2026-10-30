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

/// Blaze depth from the actual media on both sides, evaluated at λ and λ₀.
/// A designed reflective relief uses 2n; a transmissive relief uses |n₂−n₁|.
/// Its Fresnel-reflected channel uses 2n_in over that same manufactured depth.
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
    let depth_index = if s.is_mirror() {
        2.0 * before.index(l0)
    } else {
        (after.index(l0) - before.index(l0)).abs()
    };
    let contrast = if reflected {
        2.0 * if backward {
            after.index(lambda)
        } else {
            before.index(lambda)
        }
    } else {
        (after.index(lambda) - before.index(lambda)) * (after.index(l0) - before.index(l0)).signum()
    };
    // A same-medium phase screen has no relief substrate: use the
    // nondispersive phase-screen limit rather than an undefined 0/0 depth.
    let ratio = if depth_index == 0.0 {
        1.0
    } else {
        contrast / depth_index
    };
    phase.design_order as f64 * l0 / lambda * ratio
}

pub fn order_efficiency(
    sys: &ResolvedSystem,
    surface: usize,
    wl: usize,
    order: i32,
    reflected: bool,
    backward: bool,
) -> f64 {
    let SurfaceShape::Diffractive { phase, .. } = &sys.surfaces[surface].shape else {
        return 1.0;
    };
    sys.system().surfaces[surface]
        .diffraction_efficiency
        .efficiency(
            order,
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
    fn substrate_contrast_reflection_and_energy_closure() {
        use crate::{Interaction, MaterialRef, Snapshot};
        use std::sync::Arc;
        let mut s = crate::samples::hybrid_diffractive_achromat();
        s.surfaces[1].material = MaterialRef::Constant { n: 1.5 };
        s.surfaces[2].diffraction_efficiency = DiffractionEfficiency::Kinoform;
        let resolve = |s| {
            let snap = Snapshot::new(s).unwrap();
            ResolvedSystem::new(
                &snap,
                Arc::new(super::super::MaterialLibrary::with_builtin()),
            )
            .unwrap()
        };
        let r = resolve(s.clone());
        let l0 = if let SurfaceShape::Diffractive { phase, .. } = &s.surfaces[2].shape {
            phase.design_wavelength_um
        } else {
            unreachable!()
        };
        assert!((blaze(&r, 2, l0, false, false) - 1.0).abs() < 1e-15);
        assert!((blaze(&r, 2, l0, true, false) - 6.0).abs() < 1e-15);
        assert!((blaze(&r, 2, l0, true, true) - 4.0).abs() < 1e-15);
        let model = DiffractionEfficiency::Kinoform;
        for lambda in [l0, 0.45, 0.7] {
            let bt = blaze(&r, 2, lambda, false, false);
            let br = blaze(&r, 2, lambda, true, false);
            let outgoing: f64 = (-10000..=10000)
                .map(|m| 0.8 * model.efficiency(m, 1, bt) + 0.1 * model.efficiency(m, 1, br))
                .sum();
            // The finite-window omission is bounded by the sinc² tail, while
            // coating absorption is independent of the phase partition.
            assert!(outgoing + 0.1 <= 1.0 + 1e-13 && outgoing + 0.1 > 0.99997);
        }
        // A cemented relief can reverse its phase when the two dispersions
        // cross; absolute index contrasts would incorrectly keep a positive blaze.
        let mut crossing = s.clone();
        crossing.surfaces[1].material = MaterialRef::Constant { n: 1.52 };
        crossing.surfaces[2].material = MaterialRef::glass("N-BK7");
        assert!(blaze(&resolve(crossing), 2, 0.4861327, false, false) < 0.0);
        s.surfaces[2].interaction = Interaction::Reflect;
        let r = resolve(s);
        assert!((blaze(&r, 2, l0, true, false) - 1.0).abs() < 1e-15);
        assert!((blaze(&r, 2, l0 * 2.0, true, false) - 0.5).abs() < 1e-15);
    }
}
