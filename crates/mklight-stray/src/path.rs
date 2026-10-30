//! Path classification (ARCHITECTURE "Solvers and double counting").
//!
//! Every Monte-Carlo path keeps a compact [`History`] of its interactions.
//! When the path reaches the detector, the history gives one disjoint
//! [`PathClass`]:
//!
//! | class | rule (first match wins) | owner |
//! |---|---|---|
//! | `Scatter` | at least one scattering event (BSDF sample) anywhere | stray |
//! | `Mechanical` | touched non-optical geometry with a specular reflection, no scatter | stray |
//! | `Sneak` | specular only, but crossed an optical surface outside its sequential clipping aperture | stray |
//! | `Ghost` | specular only, inside the clipping apertures, ≥ 2 *stray reflections* | ghost |
//! | `Primary` | specular only, every interface met in designed order with the designed interaction, 0 stray reflections | sequential |
//! | `Sneak` | anything else specular (out-of-sequence paths with fewer than 2 stray reflections) | stray |
//!
//! A *stray reflection* is a specular reflection, including total internal
//! reflection, at a refracting (or ideal-lens) interface. Mirror reflections
//! are the designed interaction of a mirror and do not count. The `Sneak`
//! class is the stray solver's own: those paths are neither the designed path
//! nor ghosts that the ghost solver enumerates, so the partition stays
//! disjoint and complete.
//!
//! When the stray solver runs alongside the ghost solver, the engine tallies
//! only [`ClassMask::STRAY_WITH_GHOST`] (`Scatter | Mechanical | Sneak`).
//! Running alone, it can report `Ghost` and `Primary` too
//! ([`ClassMask::ALL`]). For cross-validation with the ghost solver, ghost
//! flux is also tallied by order (number of stray reflections) and by the
//! surface pair of the first two reflections.

use serde::{Deserialize, Serialize};

/// Disjoint path classes at the detector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathClass {
    Primary = 0,
    Ghost = 1,
    Scatter = 2,
    Mechanical = 3,
    Sneak = 4,
}

pub const NUM_CLASSES: usize = 5;

impl PathClass {
    pub const ALL: [PathClass; NUM_CLASSES] = [
        PathClass::Primary,
        PathClass::Ghost,
        PathClass::Scatter,
        PathClass::Mechanical,
        PathClass::Sneak,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of path classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClassMask(pub u8);

impl ClassMask {
    pub const ALL: ClassMask = ClassMask(0b1_1111);
    /// Classes the stray solver owns when the ghost solver also runs.
    pub const STRAY_WITH_GHOST: ClassMask = ClassMask(
        (1 << PathClass::Scatter as u8)
            | (1 << PathClass::Mechanical as u8)
            | (1 << PathClass::Sneak as u8),
    );
    /// Everything except the designed path.
    pub const ALL_STRAY: ClassMask = ClassMask(0b1_1110);
    pub fn contains(self, c: PathClass) -> bool {
        self.0 & c.bit() != 0
    }
}

impl Default for ClassMask {
    fn default() -> Self {
        ClassMask::ALL
    }
}

/// History flag bits.
pub mod flags {
    /// At least one BSDF scattering event.
    pub const SCATTERED: u8 = 1;
    /// Touched non-optical geometry (specular reflection or scatter).
    pub const MECHANICAL: u8 = 2;
    /// Left the designed sequence (wrong surface, side or interaction).
    pub const OUT_OF_SEQUENCE: u8 = 4;
    /// Crossed or reflected off an optical surface outside its clipping aperture.
    pub const OUTSIDE_CLIP: u8 = 8;
    /// At least one total internal reflection.
    pub const TIR: u8 = 16;
}

/// No surface recorded.
pub const NO_SURFACE: u16 = u16::MAX;

/// Event history of one path (12 bytes, cheap to copy on splitting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct History {
    pub flags: u8,
    /// Specular reflections at refracting interfaces (saturating).
    pub stray_reflections: u8,
    /// Position in `Scene::sequence` of the next designed surface.
    pub next_seq: u16,
    /// Surfaces of the first two stray reflections (`NO_SURFACE` if none).
    pub first_reflections: [u16; 2],
    /// Number of scattering events (saturating).
    pub scatter_events: u16,
}

impl Default for History {
    fn default() -> Self {
        History {
            flags: 0,
            stray_reflections: 0,
            next_seq: 0,
            first_reflections: [NO_SURFACE; 2],
            scatter_events: 0,
        }
    }
}

impl History {
    #[inline]
    pub fn has(&self, f: u8) -> bool {
        self.flags & f != 0
    }

    /// Records a specular interaction with interface `surface`. `designed`
    /// says whether the side and the interaction match the sequential model.
    /// `stray_reflection` marks a reflection at a refracting surface.
    pub fn specular(
        &mut self,
        sequence: &[usize],
        surface: usize,
        designed: bool,
        stray_reflection: bool,
    ) {
        let in_seq = !self.has(flags::OUT_OF_SEQUENCE)
            && designed
            && sequence.get(self.next_seq as usize) == Some(&surface);
        if in_seq {
            self.next_seq += 1;
        } else {
            self.flags |= flags::OUT_OF_SEQUENCE;
        }
        if stray_reflection {
            let k = self.stray_reflections as usize;
            if k < 2 {
                self.first_reflections[k] = surface.min(NO_SURFACE as usize - 1) as u16;
            }
            self.stray_reflections = self.stray_reflections.saturating_add(1);
        }
    }

    pub fn scattered(&mut self, mechanical: bool) {
        self.flags |= flags::SCATTERED | flags::OUT_OF_SEQUENCE;
        if mechanical {
            self.flags |= flags::MECHANICAL;
        }
        self.scatter_events = self.scatter_events.saturating_add(1);
    }

    pub fn mechanical_specular(&mut self) {
        self.flags |= flags::MECHANICAL | flags::OUT_OF_SEQUENCE;
    }

    /// Class of a path that reaches the detector. `sequence` is
    /// `Scene::sequence` (the detector's surface last).
    pub fn classify(&self, sequence: &[usize]) -> PathClass {
        if self.has(flags::SCATTERED) {
            PathClass::Scatter
        } else if self.has(flags::MECHANICAL) {
            PathClass::Mechanical
        } else if self.has(flags::OUTSIDE_CLIP) {
            PathClass::Sneak
        } else if self.stray_reflections >= 2 {
            PathClass::Ghost
        } else if self.stray_reflections == 0
            && !self.has(flags::OUT_OF_SEQUENCE)
            && (sequence.is_empty() || self.next_seq as usize + 1 == sequence.len())
        {
            PathClass::Primary
        } else {
            PathClass::Sneak
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: [usize; 4] = [1, 2, 3, 4]; // two-lens-surface window + image at 4

    #[test]
    fn designed_path_is_primary() {
        let mut h = History::default();
        for s in [1, 2, 3] {
            h.specular(&SEQ, s, true, false);
        }
        assert_eq!(h.classify(&SEQ), PathClass::Primary);
        // Skipping a surface is not the designed path.
        let mut h = History::default();
        h.specular(&SEQ, 1, true, false);
        h.specular(&SEQ, 3, true, false);
        assert_eq!(h.classify(&SEQ), PathClass::Sneak);
    }

    #[test]
    fn two_reflections_make_a_ghost() {
        let mut h = History::default();
        h.specular(&SEQ, 1, true, false);
        h.specular(&SEQ, 2, true, false);
        h.specular(&SEQ, 3, false, true); // reflect at 3
        h.specular(&SEQ, 2, false, false);
        h.specular(&SEQ, 1, false, true); // reflect at 1 (from inside)
        h.specular(&SEQ, 2, false, false);
        h.specular(&SEQ, 3, false, false);
        assert_eq!(h.classify(&SEQ), PathClass::Ghost);
        assert_eq!(h.first_reflections, [3, 1]);
        // One reflection only (for example back-reflected by a tilted plate): Sneak.
        let mut h = History::default();
        h.specular(&SEQ, 1, false, true);
        assert_eq!(h.classify(&SEQ), PathClass::Sneak);
    }

    #[test]
    fn scatter_and_mechanical_take_precedence() {
        let mut h = History::default();
        h.specular(&SEQ, 3, false, true);
        h.specular(&SEQ, 1, false, true);
        h.mechanical_specular();
        assert_eq!(h.classify(&SEQ), PathClass::Mechanical);
        h.scattered(false);
        assert_eq!(h.classify(&SEQ), PathClass::Scatter);
        let mut h = History::default();
        h.flags |= flags::OUTSIDE_CLIP;
        h.stray_reflections = 2;
        assert_eq!(h.classify(&SEQ), PathClass::Sneak);
    }

    #[test]
    fn masks() {
        assert!(ClassMask::STRAY_WITH_GHOST.contains(PathClass::Scatter));
        assert!(!ClassMask::STRAY_WITH_GHOST.contains(PathClass::Ghost));
        assert!(!ClassMask::STRAY_WITH_GHOST.contains(PathClass::Primary));
        assert!(ClassMask::ALL_STRAY.contains(PathClass::Ghost));
        assert!(PathClass::ALL.iter().all(|c| ClassMask::ALL.contains(*c)));
    }
}
