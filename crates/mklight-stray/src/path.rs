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
//! | `Ghost` | specular only, inside the clipping apertures, an even number ≥ 2 of *stray reflections* in their implied sequential order | ghost for covered paths, otherwise stray |
//! | `Primary` | specular only, every interface met in designed order with the designed interaction, 0 stray reflections | sequential |
//! | `Sneak` | anything else specular (including out-of-sequence paths at any reflection order) | stray |
//!
//! A *stray reflection* is a specular reflection, including total internal
//! reflection, at a refracting (or ideal-lens) interface. Mirror reflections
//! are the designed interaction of a mirror and do not count. The `Sneak`
//! class is the stray solver's own: those paths are neither the designed path
//! nor ghosts that the ghost solver enumerates, so the partition stays
//! disjoint and complete.
//!
//! `Sneak` is the stray solver's own class (ARCHITECTURE "Solvers and
//! double counting" lists all five); it holds the `Irregular` signatures of
//! CONVENTIONS §10 and specular paths outside a clipping aperture. With
//! apertures built as geometry, an opaque non-clear zone and stop
//! plates, specular light outside the clipping apertures is rare by
//! construction; `Sneak` mostly holds single stray reflections (for example
//! the front-surface reflection of a Mangin mirror or a tilted window) and
//! out-of-sequence specular paths.
//!
//! A surface the sequential model visits twice is one primitive with two
//! passes (`scene::Pass`); the tracer reports the pass that matches the
//! path's next designed surface, so the designed return pass stays `Primary`.
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
    /// Physical events departed from the sequence implied by reflections.
    pub const IRREGULAR_SPECULAR: u8 = 32;
    /// Walking backward through that sequence (designed mirrors do not flip it).
    pub const BACKWARD: u8 = 64;
    pub const DIFFRACTED: u8 = 128;
}

/// No surface recorded.
pub const NO_SURFACE: u16 = u16::MAX;

/// Events kept per path signature; later events set `Signature::overflow`.
pub const MAX_SIGNATURE_EVENTS: usize = 6;

/// DOE visits kept independently of non-designed events. Thirty-two supports
/// multi-pass trains without heap allocation in the copied path history.
/// Longer coverage is rejected at transport input; overflow stays uncovered.
pub const MAX_DIFFRACTION_VISITS: usize = 32;

/// Kind of a non-designed event in a [`Signature`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Specular (Fresnel or coating) reflection at a refracting interface.
    Reflect = 1,
    /// Total internal reflection at a refracting interface.
    Tir = 2,
    /// BSDF scatter into the reflected hemisphere of an optical surface.
    ScatterReflect = 3,
    /// BSDF scatter into the transmitted hemisphere of an optical surface.
    ScatterTransmit = 4,
    /// Specular reflection off non-optical geometry.
    MechanicalSpecular = 5,
    /// BSDF scatter off non-optical geometry.
    MechanicalScatter = 6,
    /// Crossed or reflected off an optical surface outside its sequential
    /// clipping aperture.
    OutsideClip = 7,
}

impl EventKind {
    fn from_code(c: u32) -> Option<EventKind> {
        Some(match c {
            1 => EventKind::Reflect,
            2 => EventKind::Tir,
            3 => EventKind::ScatterReflect,
            4 => EventKind::ScatterTransmit,
            5 => EventKind::MechanicalSpecular,
            6 => EventKind::MechanicalScatter,
            7 => EventKind::OutsideClip,
            _ => return None,
        })
    }
    /// True for events whose id is a primitive index (non-optical geometry)
    /// rather than a sequential surface index.
    pub fn is_mechanical(self) -> bool {
        matches!(
            self,
            EventKind::MechanicalSpecular | EventKind::MechanicalScatter
        )
    }
}

/// The ordered non-designed events of a path: every interaction that is not
/// the designed transmission (or designed mirror reflection) of the next
/// surface in sequence. Designed interactions between the recorded events
/// are implied, so for a `Ghost` path the signature is the ordered list of
/// its reflecting surfaces, which [`Signature::expand_ghost`] turns into the
/// full ordered `(surface, transmit | reflect)` signature used by the
/// explicit ghost solver. Only the first [`MAX_SIGNATURE_EVENTS`] events are
/// kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Signature {
    len: u8,
    /// More events happened than were kept.
    overflow: bool,
    /// `kind << 28 | id`.
    events: [u32; MAX_SIGNATURE_EVENTS],
    orders: [DiffractionEvent; MAX_DIFFRACTION_VISITS],
    order_len: u8,
}

/// Physical DOE events, including selected orders, in encounter order.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct DiffractionEvent {
    pub surface: u32,
    pub visit: u32,
    pub order: i32,
    pub reflected: bool,
    pub selected: bool,
}

/// Largest id a signature event can hold.
const SIG_ID_MASK: u32 = (1 << 28) - 1;

/// Designed interaction of a surface in a full ghost signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceOp {
    Transmit,
    Reflect,
}

impl Signature {
    pub fn diffraction(&self) -> &[DiffractionEvent] {
        &self.orders[..self.order_len as usize]
    }
    pub fn diffract(&mut self, surface: usize, order: i32, selected: i32, reflected: bool) {
        let visit = self
            .diffraction()
            .iter()
            .filter(|e| e.surface == surface as u32)
            .count() as u32;
        if (self.order_len as usize) < MAX_DIFFRACTION_VISITS {
            self.orders[self.order_len as usize] = DiffractionEvent {
                surface: surface as u32,
                visit,
                order,
                reflected,
                selected: order == selected,
            };
            self.order_len += 1;
        } else {
            self.overflow = true;
        }
    }
    /// Appends an event; `id` is the sequential surface index for optical
    /// events and the primitive index for mechanical ones.
    #[inline]
    pub fn push(&mut self, kind: EventKind, id: usize) {
        let k = self.len as usize;
        if k < MAX_SIGNATURE_EVENTS {
            self.events[k] = ((kind as u32) << 28) | (id.min(SIG_ID_MASK as usize) as u32);
            self.len += 1;
        } else {
            self.overflow = true;
        }
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn overflow(&self) -> bool {
        self.overflow
    }
    /// The kept events in order.
    pub fn events(&self) -> impl Iterator<Item = (EventKind, usize)> + '_ {
        self.events[..self.len as usize]
            .iter()
            .filter_map(|&e| EventKind::from_code(e >> 28).map(|k| (k, (e & SIG_ID_MASK) as usize)))
    }
    /// Compact text: `R5` (reflection at surface 5), `TIR5`, `Sr5` / `St5`
    /// (scatter at surface 5, reflected / transmitted side), `C5` (outside
    /// the clipping aperture of surface 5), `M[label]` / `Ms[label]`
    /// (specular / scatter off non-optical primitive), `…` on overflow, and
    /// `designed` for an empty signature. `label` names primitives.
    pub fn describe(&self, label: impl Fn(usize) -> String) -> String {
        let mut parts: Vec<String> = self
            .events()
            .map(|(k, id)| match k {
                EventKind::Reflect => format!("R{id}"),
                EventKind::Tir => format!("TIR{id}"),
                EventKind::ScatterReflect => format!("Sr{id}"),
                EventKind::ScatterTransmit => format!("St{id}"),
                EventKind::OutsideClip => format!("C{id}"),
                EventKind::MechanicalSpecular => format!("M[{}]", label(id)),
                EventKind::MechanicalScatter => format!("Ms[{}]", label(id)),
            })
            .collect();
        parts.extend(self.diffraction().iter().map(|e| {
            format!(
                "D{}:{}:{}{}",
                e.surface,
                e.visit,
                if e.reflected { "R" } else { "T" },
                e.order
            )
        }));
        if self.overflow {
            parts.push("…".into());
        }
        if parts.is_empty() {
            "designed".into()
        } else {
            parts.join(" ")
        }
    }
    /// The full ordered `(surface, op)` signature of a purely specular path
    /// whose recorded events are all reflections at refracting interfaces
    /// (a `Ghost`-class or designed path), assuming it meets the interfaces
    /// of `sequence` in order: transmit through every interface up to the
    /// next reflecting one, reverse, and so on, ending at the detector (the
    /// last entry of `sequence`, which is not included). Mirrors in the
    /// sequence keep their designed reflection. `None` when the signature
    /// holds other events, overflowed, or does not fit the sequence.
    pub fn expand_ghost(
        &self,
        sequence: &[usize],
        is_mirror: impl Fn(usize) -> bool,
    ) -> Option<Vec<(usize, SurfaceOp)>> {
        if self.overflow || sequence.is_empty() {
            return None;
        }
        let ifaces = &sequence[..sequence.len() - 1];
        let op = |s: usize| {
            if is_mirror(s) {
                SurfaceOp::Reflect
            } else {
                SurfaceOp::Transmit
            }
        };
        let mut out = Vec::new();
        // Position of the next interface to meet and the walking direction.
        let mut pos: isize = 0;
        let mut forward = true;
        for (kind, surf) in self.events() {
            if !matches!(kind, EventKind::Reflect | EventKind::Tir) {
                return None;
            }
            // Walk to the reflecting surface, transmitting on the way.
            loop {
                if pos < 0 || pos as usize >= ifaces.len() {
                    return None;
                }
                let s = ifaces[pos as usize];
                if s == surf {
                    out.push((s, SurfaceOp::Reflect));
                    forward = !forward;
                    pos += if forward { 1 } else { -1 };
                    break;
                }
                out.push((s, op(s)));
                pos += if forward { 1 } else { -1 };
            }
        }
        if !forward {
            return None;
        }
        while (pos as usize) < ifaces.len() {
            let s = ifaces[pos as usize];
            out.push((s, op(s)));
            pos += 1;
        }
        Some(out)
    }
}

/// Event history of one path (cheap to copy on splitting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct History {
    pub flags: u8,
    /// Specular reflections at refracting interfaces (saturating).
    pub stray_reflections: u8,
    /// Position in `Scene::sequence` of the next designed surface.
    pub next_seq: u16,
    /// Next surface in the specular walk, including ghost reversals. Signed
    /// so a path leaving through the front cannot wrap back into the train.
    next_specular: i32,
    /// Surfaces of the first two stray reflections (`NO_SURFACE` if none).
    pub first_reflections: [u16; 2],
    /// Number of scattering events (saturating).
    pub scatter_events: u16,
    /// Ordered non-designed events.
    pub signature: Signature,
}

impl Default for History {
    fn default() -> Self {
        History {
            flags: 0,
            stray_reflections: 0,
            next_seq: 0,
            next_specular: 0,
            first_reflections: [NO_SURFACE; 2],
            scatter_events: 0,
            signature: Signature::default(),
        }
    }
}

impl History {
    #[inline]
    pub fn has(&self, f: u8) -> bool {
        self.flags & f != 0
    }

    pub(crate) fn expected_specular(&self, sequence: &[usize]) -> Option<usize> {
        if self.has(flags::IRREGULAR_SPECULAR | flags::SCATTERED | flags::MECHANICAL) {
            None
        } else {
            sequence.get(self.next_specular as usize).copied()
        }
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
        if sequence.get(self.next_specular as usize) != Some(&surface) {
            self.flags |= flags::IRREGULAR_SPECULAR;
        }
        if stray_reflection {
            self.flags ^= flags::BACKWARD;
        }
        self.next_specular += if self.has(flags::BACKWARD) { -1 } else { 1 };
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
        } else if (self.stray_reflections >= 2 || self.has(flags::DIFFRACTED))
            && !self.has(flags::IRREGULAR_SPECULAR | flags::BACKWARD)
            && self.next_specular == sequence.len() as i32 - 1
        {
            PathClass::Ghost
        } else if self.stray_reflections == 0
            && !self.has(flags::DIFFRACTED)
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
    fn covered_reflections_with_a_skipped_interface_stay_stray() {
        let mut h = History::default();
        // Same [3,1] reflections as the canonical ghost above, but missing
        // the forward transmission through surface 2. Reflection lists
        // alone cannot determine ownership (CONVENTIONS §10).
        for (surface, designed, reflected) in [
            (1, true, false),
            (3, false, true),
            (2, false, false),
            (1, false, true),
            (2, false, false),
            (3, false, false),
        ] {
            h.specular(&SEQ, surface, designed, reflected);
        }
        assert_eq!(h.first_reflections, [3, 1]);
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
    fn signatures_record_and_expand() {
        let mut sig = Signature::default();
        assert_eq!(sig.describe(|_| String::new()), "designed");
        assert_eq!(
            sig.expand_ghost(&SEQ, |_| false).unwrap(),
            [
                (1, SurfaceOp::Transmit),
                (2, SurfaceOp::Transmit),
                (3, SurfaceOp::Transmit)
            ]
        );
        sig.push(EventKind::Reflect, 3);
        sig.push(EventKind::Reflect, 1);
        assert_eq!(sig.describe(|_| String::new()), "R3 R1");
        // T1 T2 R3 T2 R1 T2 T3: the ghost solver's ordered signature.
        use SurfaceOp::{Reflect as R, Transmit as T};
        assert_eq!(
            sig.expand_ghost(&SEQ, |_| false).unwrap(),
            [(1, T), (2, T), (3, R), (2, T), (1, R), (2, T), (3, T)]
        );
        // A path that ends travelling backwards never reaches the detector.
        let mut odd = Signature::default();
        odd.push(EventKind::Reflect, 2);
        assert!(odd.expand_ghost(&SEQ, |_| false).is_none());
        // A designed mirror keeps its reflection.
        assert_eq!(
            Signature::default().expand_ghost(&SEQ, |s| s == 2).unwrap()[1],
            (2, R)
        );
        // Mechanical events are not ghosts; labels name the primitive.
        let mut m = sig;
        m.push(EventKind::MechanicalScatter, 7);
        assert!(m.expand_ghost(&SEQ, |_| false).is_none());
        assert_eq!(m.describe(|p| format!("prim {p}")), "R3 R1 Ms[prim 7]");
        // Overflow keeps the first events.
        let mut long = Signature::default();
        for k in 0..MAX_SIGNATURE_EVENTS + 2 {
            long.push(EventKind::Tir, k);
        }
        assert_eq!(long.len(), MAX_SIGNATURE_EVENTS);
        assert!(long.overflow() && long.describe(|_| String::new()).ends_with('…'));
        assert!(long.expand_ghost(&SEQ, |_| false).is_none());
        // Ordering is total and distinguishes kinds.
        let mut a = Signature::default();
        a.push(EventKind::Reflect, 3);
        let mut b = Signature::default();
        b.push(EventKind::Tir, 3);
        assert_ne!(a, b);
        assert_eq!(a.events().next(), Some((EventKind::Reflect, 3)));
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
