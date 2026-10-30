//! Path signatures and the partition rule between the sequential, ghost and
//! stray-light solvers (ARCHITECTURE "Solvers and double counting").
//!
//! # Optical interfaces
//!
//! [`interface_kinds`] classifies every surface of a [`ResolvedSystem`]:
//!
//! | kind | surfaces |
//! |---|---|
//! | `Object` | surface 0 |
//! | `Passive` | dummies, coordinate breaks, and refracting surfaces with a bare coating and the *same* index on both sides at every system wavelength (e.g. a stop plane in air): physically absent |
//! | `Refractive` | every other refracting surface (conic, asphere, plane) |
//! | `IdealLens` | refracting ideal (paraxial) lenses: they bend rays but cannot reflect |
//! | `Mirror` | designed mirrors |
//! | `Image` | the image (sensor) surface |
//!
//! # Signature
//!
//! A [`PathSignature`] is the ordered list of physical events
//! `(surface, Transmit | Reflect)` of a path at the non-passive surfaces,
//! from leaving the object to the end of the path. The event names are
//! *physical*: a designed mirror's reflection is a `Reflect`; detection by
//! the sensor is a final `Transmit` at the image surface. Passive surfaces
//! never appear (a stray-light tracer that does not know about them produces
//! the same signature).
//!
//! The **ghost reflections** of a signature are its `Reflect` events at
//! non-mirror surfaces (`Refractive` or `Image`). Given them, the sequential
//! model fixes the rest: [`PathSignature::for_path`] expands a
//! [`GhostPath`] into the *canonical* signature, where light meets every
//! non-passive surface between two reflections in order.
//!
//! # Partition rule
//!
//! For a path reaching the detector, with flags `scattered` (any scatter
//! event, on any geometry) and `mechanical` (any interaction with
//! non-optical geometry), [`classify`] gives the physical [`PathClass`]:
//!
//! 1. `scattered` → `Scatter`;
//! 2. else `mechanical` → `Mechanical`;
//! 3. else the signature does not end with `(image, Transmit)` →
//!    `Undetected`;
//! 4. else, with `g` = its ghost reflections, if the signature equals the
//!    canonical expansion of `GhostPath(g)`: `Primary` when `g` is empty,
//!    otherwise `Ghost { reflections: |g| }`;
//! 5. else → `Irregular` (purely specular, but not sequentially consistent:
//!    it skipped a surface, e.g. passing beside an element).
//!
//! The solver that **owns** (counts) a path is given by
//! [`GhostCoverage::owner`]: `Primary` → `Sequential`; `Ghost` → `Ghost`
//! **iff its reflection sequence is in the coverage** of the ghost run
//! (the exact set of paths that run traced, after its order limit, filters
//! and budget), otherwise `Stray`; `Scatter`, `Mechanical`, `Irregular` →
//! `Stray`; `Undetected` → nobody. Every detected path therefore has exactly
//! one owner, and a ghost path the ghost run left out (higher order, below
//! the reflectance threshold, outside the surface range) is still counted,
//! by the stray solver.
//!
//! The rule assumes both solvers clip at identical apertures
//! ([`GhostCoverage::apertures`], [`GhostCoverage::same_apertures`]).

use std::collections::BTreeSet;

use mklight_core::materials::coating::ResolvedCoating;
use mklight_core::{Interaction, ResolvedSystem, SurfaceShape};
use serde::{Deserialize, Serialize};

use crate::model::Clip;
use crate::path::{Action, GhostPath};

/// What a surface is to the partition rule (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterfaceKind {
    Object,
    Passive,
    Refractive,
    IdealLens,
    Mirror,
    Image,
}

impl InterfaceKind {
    /// Does an event at this surface appear in a signature?
    pub fn in_signature(self) -> bool {
        !matches!(self, InterfaceKind::Object | InterfaceKind::Passive)
    }
}

/// Classify every surface (module docs).
pub fn interface_kinds(sys: &ResolvedSystem) -> Vec<InterfaceKind> {
    let n = sys.num_surfaces();
    (0..n)
        .map(|i| {
            let s = &sys.surfaces[i];
            if i == 0 {
                return InterfaceKind::Object;
            }
            if i == n - 1 {
                return InterfaceKind::Image;
            }
            if matches!(s.shape, SurfaceShape::CoordinateBreak)
                || s.interaction == Interaction::Dummy
            {
                return InterfaceKind::Passive;
            }
            if s.interaction == Interaction::Reflect {
                return InterfaceKind::Mirror;
            }
            if matches!(s.shape, SurfaceShape::Paraxial { .. }) {
                return InterfaceKind::IdealLens;
            }
            let same_media =
                (0..sys.wavelengths_um.len()).all(|w| sys.n_before(i, w) == s.n_after[w]);
            if same_media && matches!(s.coating, ResolvedCoating::Bare) {
                InterfaceKind::Passive
            } else {
                InterfaceKind::Refractive
            }
        })
        .collect()
}

/// A physical event at a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Transmit,
    Reflect,
}

/// `(surface, event)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PathEvent {
    pub surface: u32,
    pub event: Event,
}

/// Ordered physical events at the non-passive surfaces (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PathSignature {
    pub events: Vec<PathEvent>,
}

impl PathSignature {
    /// The canonical signature of a sequential ghost path (`None` if the
    /// path is invalid for these surfaces).
    pub fn for_path(path: &GhostPath, kinds: &[InterfaceKind]) -> Option<PathSignature> {
        let steps = path.steps(kinds.len()).ok()?;
        let mut events = Vec::with_capacity(steps.len());
        for st in steps {
            let kind = kinds[st.surface as usize];
            if !kind.in_signature() {
                if st.action == Action::Reflect {
                    return None;
                }
                continue;
            }
            let event = match (kind, st.action) {
                (InterfaceKind::Mirror, Action::Pass) => Event::Reflect,
                (InterfaceKind::Mirror, Action::Reflect)
                | (InterfaceKind::IdealLens, Action::Reflect) => {
                    return None;
                }
                (_, Action::Pass) => Event::Transmit,
                (_, Action::Reflect) => Event::Reflect,
            };
            events.push(PathEvent {
                surface: st.surface,
                event,
            });
        }
        Some(PathSignature { events })
    }

    /// The ghost reflections: `Reflect` events at non-mirror surfaces.
    pub fn ghost_reflections(&self, kinds: &[InterfaceKind]) -> Vec<u32> {
        self.events
            .iter()
            .filter(|e| {
                e.event == Event::Reflect
                    && kinds
                        .get(e.surface as usize)
                        .is_some_and(|k| *k != InterfaceKind::Mirror)
            })
            .map(|e| e.surface)
            .collect()
    }

    /// Ends with detection at the image surface.
    pub fn detected(&self, kinds: &[InterfaceKind]) -> bool {
        self.events.last().is_some_and(|e| {
            e.event == Event::Transmit
                && kinds.get(e.surface as usize) == Some(&InterfaceKind::Image)
        })
    }
}

/// Physical class of a path (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum PathClass {
    Primary,
    Ghost { reflections: u32 },
    Scatter,
    Mechanical,
    Irregular,
    Undetected,
}

/// The solver that counts a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Sequential,
    Ghost,
    Stray,
    /// Not detected: no solver counts it as image flux.
    Nobody,
}

/// Physical class of a path (partition rule steps 1–5).
pub fn classify(
    sig: &PathSignature,
    kinds: &[InterfaceKind],
    scattered: bool,
    mechanical: bool,
) -> PathClass {
    if scattered {
        return PathClass::Scatter;
    }
    if mechanical {
        return PathClass::Mechanical;
    }
    if !sig.detected(kinds) {
        return PathClass::Undetected;
    }
    let g = GhostPath::new(sig.ghost_reflections(kinds));
    match PathSignature::for_path(&g, kinds) {
        Some(c) if c == *sig => {
            if g.is_primary() {
                PathClass::Primary
            } else {
                PathClass::Ghost {
                    reflections: g.order() as u32,
                }
            }
        }
        _ => PathClass::Irregular,
    }
}

/// The paths one ghost run traced, for the stray solver's exclusion.
///
/// Ownership by signature alone is only disjoint and complete if both
/// solvers clip a ray at the same places: a ray the ghost run clipped must
/// not survive in the stray solver with a covered signature (nobody would
/// count it), and vice versa (both would). The record therefore carries the
/// clipping apertures of the ghost run, which the stray solver must
/// reproduce at the optical surfaces (check with
/// [`GhostCoverage::same_apertures`]).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct GhostCoverage {
    /// Interface classification of the system the run used.
    pub interfaces: Vec<InterfaceKind>,
    /// Clipping of every surface in the ghost run (`Clip::Surface`: the
    /// surface's own fixed aperture in the same snapshot).
    #[serde(default)]
    pub apertures: Vec<Clip>,
    /// Exactly the reflection sequences the ghost run counted.
    pub paths: BTreeSet<GhostPath>,
}

impl GhostCoverage {
    /// Does a stray solver that clips with `apertures` (one per surface,
    /// same snapshot) partition consistently with this ghost run?
    pub fn same_apertures(&self, apertures: &[Clip]) -> bool {
        self.apertures == apertures
    }

    pub fn covers(&self, path: &GhostPath) -> bool {
        self.paths.contains(path)
    }

    /// The solver that counts a path (module docs).
    pub fn owner(&self, sig: &PathSignature, scattered: bool, mechanical: bool) -> Owner {
        match classify(sig, &self.interfaces, scattered, mechanical) {
            PathClass::Primary => Owner::Sequential,
            PathClass::Ghost { .. } => {
                let g = GhostPath::new(sig.ghost_reflections(&self.interfaces));
                if self.covers(&g) {
                    Owner::Ghost
                } else {
                    Owner::Stray
                }
            }
            PathClass::Undetected => Owner::Nobody,
            PathClass::Scatter | PathClass::Mechanical | PathClass::Irregular => Owner::Stray,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use InterfaceKind::*;

    // Object, lens front, stop (passive), lens rear, mirror, image.
    const K: [InterfaceKind; 6] = [Object, Refractive, Passive, Refractive, Mirror, Image];

    fn ev(s: u32, e: Event) -> PathEvent {
        PathEvent {
            surface: s,
            event: e,
        }
    }

    #[test]
    fn signatures_skip_passive_and_name_physical_events() {
        let p = PathSignature::for_path(&GhostPath::primary(), &K).unwrap();
        assert_eq!(
            p.events,
            vec![
                ev(1, Event::Transmit),
                ev(3, Event::Transmit),
                ev(4, Event::Reflect),
                ev(5, Event::Transmit)
            ]
        );
        let g = PathSignature::for_path(&GhostPath::new(vec![3, 1]), &K).unwrap();
        assert_eq!(
            g.events,
            vec![
                ev(1, Event::Transmit),
                ev(3, Event::Reflect),
                ev(1, Event::Reflect),
                ev(3, Event::Transmit),
                ev(4, Event::Reflect),
                ev(5, Event::Transmit)
            ]
        );
        assert_eq!(g.ghost_reflections(&K), vec![3, 1]);
        // Reflecting at a passive surface or a mirror is not a ghost path.
        assert!(PathSignature::for_path(&GhostPath::new(vec![2, 1]), &K).is_none());
        assert!(PathSignature::for_path(&GhostPath::new(vec![4, 1]), &K).is_none());
    }

    #[test]
    fn partition_is_disjoint_and_complete() {
        let mut cov = GhostCoverage {
            interfaces: K.to_vec(),
            apertures: vec![Clip::None; K.len()],
            paths: BTreeSet::new(),
        };
        cov.paths.insert(GhostPath::new(vec![3, 1]));
        let prim = PathSignature::for_path(&GhostPath::primary(), &K).unwrap();
        let g31 = PathSignature::for_path(&GhostPath::new(vec![3, 1]), &K).unwrap();
        let g51 = PathSignature::for_path(&GhostPath::new(vec![5, 1]), &K).unwrap();
        let g4 = PathSignature::for_path(&GhostPath::new(vec![3, 1, 3, 1]), &K).unwrap();
        assert_eq!(classify(&prim, &K, false, false), PathClass::Primary);
        assert_eq!(cov.owner(&prim, false, false), Owner::Sequential);
        assert_eq!(cov.owner(&g31, false, false), Owner::Ghost);
        // Ghosts the run did not cover go to the stray solver.
        assert_eq!(
            classify(&g4, &K, false, false),
            PathClass::Ghost { reflections: 4 }
        );
        assert_eq!(cov.owner(&g4, false, false), Owner::Stray);
        assert_eq!(cov.owner(&g51, false, false), Owner::Stray);
        // Scatter and mechanical always belong to stray, even on a covered sequence.
        assert_eq!(cov.owner(&g31, true, false), Owner::Stray);
        assert_eq!(classify(&g31, &K, false, true), PathClass::Mechanical);
        // Skipping surface 3 on the way forward again (passing beside the
        // element) is irregular → stray.
        let skip = PathSignature {
            events: vec![
                ev(1, Event::Transmit),
                ev(3, Event::Reflect),
                ev(1, Event::Reflect),
                ev(4, Event::Reflect),
                ev(5, Event::Transmit),
            ],
        };
        assert_eq!(classify(&skip, &K, false, false), PathClass::Irregular);
        assert_eq!(cov.owner(&skip, false, false), Owner::Stray);
        // A path leaving through the front is not detected.
        let back = PathSignature::for_path(&GhostPath::new(vec![3]), &K).unwrap();
        assert_eq!(cov.owner(&back, false, false), Owner::Nobody);
        // JSON round trip of the coverage.
        let j = serde_json::to_string(&cov).unwrap();
        assert_eq!(serde_json::from_str::<GhostCoverage>(&j).unwrap(), cov);
    }
}
