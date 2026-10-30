//! Ghost paths: the ordered list of extra reflections, its expansion into a
//! surface-by-surface step sequence, and path enumeration.
//!
//! # Paths and steps
//!
//! A [`GhostPath`] lists the surfaces where light is reflected *in addition
//! to* the designed sequential interactions, in the order they happen. Light
//! leaves the object travelling forward (increasing surface index). The first
//! reflection `r₀` sends it backward, the next `r₁ < r₀` forward again, and so
//! on. Between two reflections it passes every intermediate surface with that
//! surface's designed interaction (refraction, a designed mirror reflection,
//! or nothing for dummies and coordinate breaks). An even number of
//! reflections ends at the image surface; an odd number ends by leaving
//! through surface 1 into object space (not detected, used for energy
//! bookkeeping).
//!
//! [`GhostPath::steps`] expands a path into [`Step`]s: surface, direction of
//! travel when arriving, and [`Action`] (`Pass` = the designed interaction,
//! `Reflect` = the extra reflection). The empty path is the primary path
//! `1, 2, …, N−1`. The classic two-reflection ghost `[i, j]` (`j < i`) is
//! `1 … i` forward, reflect at `i`, `i−1 … j` backward, reflect at `j`,
//! `j+1 … N−1` forward.
//!
//! # Enumeration
//!
//! [`enumerate`] lists every sequence of `2, 4, …, max_reflections`
//! reflections over the ghost reflectors of a [`GhostModel`] (refracting
//! optical interfaces, plus the sensor when it has a reflectance), subject to
//! the alternation rule above, a surface range, exclusions and a minimum
//! normal-incidence reflectance product. Products only decrease as
//! reflections are added, so the search prunes on the product. All
//! two-reflection paths that pass the filters are kept; higher orders are
//! ranked by reflectance product and cut to a budget. The output order is
//! canonical (by order, then lexicographic) and deterministic.

use std::collections::BTreeSet;

use mklight_core::{MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::model::GhostModel;
use crate::signature::GhostCoverage;

/// Direction of travel along the sequential surface order when a step's
/// surface is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Arriving from surface `i − 1` (or the object).
    Forward,
    /// Arriving from surface `i + 1`.
    Backward,
}

impl Direction {
    pub fn reversed(self) -> Self {
        match self {
            Direction::Forward => Direction::Backward,
            Direction::Backward => Direction::Forward,
        }
    }
}

/// What happens at a step's surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// The surface's designed interaction: refraction (in either
    /// direction), the designed mirror reflection, nothing for dummies and
    /// coordinate breaks, detection at the image.
    Pass,
    /// An extra specular reflection at a refracting interface or the sensor,
    /// back into the incident medium.
    Reflect,
}

/// One surface visit of a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Step {
    pub surface: u32,
    pub direction: Direction,
    pub action: Action,
}

/// An ordered list of extra reflections (module docs). Empty = primary path.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GhostPath {
    pub reflections: Vec<u32>,
}

impl GhostPath {
    pub fn new(reflections: Vec<u32>) -> Self {
        GhostPath { reflections }
    }
    /// The primary (zero-reflection) path.
    pub fn primary() -> Self {
        GhostPath::default()
    }
    /// Number of extra reflections.
    pub fn order(&self) -> usize {
        self.reflections.len()
    }
    pub fn is_primary(&self) -> bool {
        self.reflections.is_empty()
    }
    /// An even number of reflections ends at the image.
    pub fn ends_at_image(&self) -> bool {
        self.reflections.len() % 2 == 0
    }

    /// A short label, e.g. `"R11·R3"`; the image surface is `IMG`.
    pub fn label(&self, image: usize) -> String {
        if self.reflections.is_empty() {
            return "primary".into();
        }
        self.reflections
            .iter()
            .map(|&r| {
                if r as usize == image {
                    "IMG".to_string()
                } else {
                    format!("R{r}")
                }
            })
            .collect::<Vec<_>>()
            .join("·")
    }

    /// Check the alternation rule for a system of `n_surfaces` surfaces
    /// (geometry only; which surfaces may reflect is checked by the model).
    pub fn validate(&self, n_surfaces: usize) -> MkResult<()> {
        if n_surfaces < 3 {
            return Err(MkError::InvalidModel("system too short".into()));
        }
        let image = n_surfaces - 1;
        let (mut pos, mut fwd) = (0usize, true);
        for (k, &r) in self.reflections.iter().enumerate() {
            let r = r as usize;
            let ok = if fwd {
                r > pos && r <= image
            } else {
                r >= 1 && r < pos
            };
            if !ok {
                return Err(MkError::InvalidModel(format!(
                    "ghost path {:?}: reflection {k} at surface {r} cannot be reached travelling {}",
                    self.reflections,
                    if fwd { "forward" } else { "backward" }
                )));
            }
            pos = r;
            fwd = !fwd;
        }
        Ok(())
    }

    /// Expand into steps (module docs).
    pub fn steps(&self, n_surfaces: usize) -> MkResult<Vec<Step>> {
        self.validate(n_surfaces)?;
        let image = n_surfaces - 1;
        let mut out = Vec::with_capacity(n_surfaces * (self.reflections.len() + 1));
        let push = |out: &mut Vec<Step>, s: usize, d: Direction, a: Action| {
            out.push(Step {
                surface: s as u32,
                direction: d,
                action: a,
            })
        };
        let (mut pos, mut fwd) = (0usize, true);
        for &r in &self.reflections {
            let r = r as usize;
            if fwd {
                for s in pos + 1..r {
                    push(&mut out, s, Direction::Forward, Action::Pass);
                }
                push(&mut out, r, Direction::Forward, Action::Reflect);
            } else {
                for s in (r + 1..pos).rev() {
                    push(&mut out, s, Direction::Backward, Action::Pass);
                }
                push(&mut out, r, Direction::Backward, Action::Reflect);
            }
            pos = r;
            fwd = !fwd;
        }
        if fwd {
            for s in pos + 1..=image {
                push(&mut out, s, Direction::Forward, Action::Pass);
            }
        } else {
            for s in (1..pos).rev() {
                push(&mut out, s, Direction::Backward, Action::Pass);
            }
        }
        Ok(out)
    }
}

/// Filters and budget of [`enumerate`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnumerationSettings {
    /// Highest order enumerated: 2, 4, … (even, at most 8).
    pub max_reflections: u32,
    /// Budget for paths of order > 2 (the strongest by reflectance product
    /// are kept).
    pub max_higher_order_paths: usize,
    /// Drop paths whose normal-incidence, unpolarised reflectance product at
    /// the primary wavelength is below this.
    pub min_reflectance_product: f64,
    /// Only surfaces in `[first, last]` may reflect.
    #[serde(default)]
    pub surface_range: Option<[u32; 2]>,
    /// Surfaces that may not reflect.
    #[serde(default)]
    pub exclude_surfaces: Vec<u32>,
}

impl Default for EnumerationSettings {
    fn default() -> Self {
        EnumerationSettings {
            max_reflections: 2,
            max_higher_order_paths: 2000,
            min_reflectance_product: 0.0,
            surface_range: None,
            exclude_surfaces: Vec::new(),
        }
    }
}

/// A path with its normal-incidence reflectance product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnumeratedPath {
    pub path: GhostPath,
    /// Π R over the reflections, unpolarised, normal incidence, primary λ.
    pub reflectance_product: f64,
}

/// Result of [`enumerate`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathEnumeration {
    pub paths: Vec<EnumeratedPath>,
    /// Higher-order paths were dropped by the budget.
    pub truncated: bool,
    /// The exact set of paths this enumeration hands to the ghost solver.
    pub coverage: GhostCoverage,
}

/// Enumerate ghost paths (module docs).
pub fn enumerate(model: &GhostModel, settings: &EnumerationSettings) -> MkResult<PathEnumeration> {
    let k_max = settings.max_reflections as usize;
    if k_max < 2 || k_max % 2 != 0 || k_max > 8 {
        return Err(MkError::InvalidModel(
            "max_reflections must be 2, 4, 6 or 8".into(),
        ));
    }
    let sys = model.system();
    let n = sys.num_surfaces();
    let wl = sys.primary_wavelength;
    let allowed = |s: usize| {
        model.is_reflector(s)
            && settings
                .surface_range
                .is_none_or(|[a, b]| (a as usize..=b as usize).contains(&s))
            && !settings.exclude_surfaces.contains(&(s as u32))
    };
    let reflectors: Vec<usize> = (1..n).filter(|&s| allowed(s)).collect();
    // Normal-incidence reflectance arriving forward / backward.
    let r_fwd: Vec<f64> = (0..n)
        .map(|s| model.normal_reflectance(s, Direction::Forward, wl))
        .collect();
    let r_bwd: Vec<f64> = (0..n)
        .map(|s| model.normal_reflectance(s, Direction::Backward, wl))
        .collect();

    let mut found: Vec<Vec<EnumeratedPath>> = vec![Vec::new(); k_max / 2];
    let mut seq = Vec::with_capacity(k_max);
    #[allow(clippy::too_many_arguments)]
    fn rec(
        seq: &mut Vec<u32>,
        pos: usize,
        fwd: bool,
        prod: f64,
        k_max: usize,
        reflectors: &[usize],
        r: (&[f64], &[f64]),
        min: f64,
        found: &mut [Vec<EnumeratedPath>],
    ) {
        if !seq.is_empty() && seq.len() % 2 == 0 {
            found[seq.len() / 2 - 1].push(EnumeratedPath {
                path: GhostPath::new(seq.clone()),
                reflectance_product: prod,
            });
        }
        if seq.len() == k_max {
            return;
        }
        for &s in reflectors {
            let reachable = if fwd { s > pos } else { s < pos };
            if !reachable {
                continue;
            }
            let p = prod * if fwd { r.0[s] } else { r.1[s] };
            if p < min || p == 0.0 {
                continue;
            }
            seq.push(s as u32);
            rec(seq, s, !fwd, p, k_max, reflectors, r, min, found);
            seq.pop();
        }
    }
    rec(
        &mut seq,
        0,
        true,
        1.0,
        k_max,
        &reflectors,
        (&r_fwd, &r_bwd),
        settings.min_reflectance_product,
        &mut found,
    );

    let mut truncated = false;
    let mut paths: Vec<EnumeratedPath> = std::mem::take(&mut found[0]);
    let mut higher: Vec<EnumeratedPath> = found.into_iter().skip(1).flatten().collect();
    if higher.len() > settings.max_higher_order_paths {
        truncated = true;
        higher.sort_by(|a, b| {
            b.reflectance_product
                .total_cmp(&a.reflectance_product)
                .then_with(|| a.path.cmp(&b.path))
        });
        higher.truncate(settings.max_higher_order_paths);
    }
    paths.extend(higher);
    paths.sort_by(|a, b| {
        a.path
            .order()
            .cmp(&b.path.order())
            .then_with(|| a.path.cmp(&b.path))
    });
    let coverage = GhostCoverage {
        interfaces: model.interface_kinds().to_vec(),
        paths: paths
            .iter()
            .map(|p| p.path.clone())
            .collect::<BTreeSet<_>>(),
    };
    Ok(PathEnumeration {
        paths,
        truncated,
        coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_reflection_expansion() {
        let p = GhostPath::new(vec![4, 2]);
        let s = p.steps(6).unwrap();
        let v: Vec<(u32, Direction, Action)> = s
            .iter()
            .map(|s| (s.surface, s.direction, s.action))
            .collect();
        use Action::*;
        use Direction::*;
        assert_eq!(
            v,
            vec![
                (1, Forward, Pass),
                (2, Forward, Pass),
                (3, Forward, Pass),
                (4, Forward, Reflect),
                (3, Backward, Pass),
                (2, Backward, Reflect),
                (3, Forward, Pass),
                (4, Forward, Pass),
                (5, Forward, Pass),
            ]
        );
        assert_eq!(GhostPath::primary().steps(4).unwrap().len(), 3);
        // Odd path leaves through surface 1.
        let s = GhostPath::new(vec![3]).steps(5).unwrap();
        assert_eq!(s.last().unwrap().surface, 1);
        assert_eq!(s.last().unwrap().direction, Backward);
        // Front-surface reflection: nothing after it.
        assert_eq!(GhostPath::new(vec![1]).steps(5).unwrap().len(), 1);
    }

    #[test]
    fn alternation_rule() {
        assert!(GhostPath::new(vec![2, 3]).validate(6).is_err());
        assert!(GhostPath::new(vec![3, 3]).validate(6).is_err());
        assert!(GhostPath::new(vec![6, 1]).validate(6).is_err());
        assert!(GhostPath::new(vec![5, 1]).validate(6).is_ok()); // sensor reflection
        assert!(GhostPath::new(vec![4, 1, 3, 2]).validate(6).is_ok());
        assert!(GhostPath::new(vec![4, 1, 3, 3]).validate(6).is_err());
        assert_eq!(GhostPath::new(vec![12, 3]).label(12), "IMG·R3");
    }
}
