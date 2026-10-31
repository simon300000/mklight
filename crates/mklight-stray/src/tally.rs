//! Detector tallies and energy bookkeeping.
//!
//! Each chunk of source rays fills a [`ChunkTally`] with plain f64 sums
//! (a chunk is at most a few thousand rays). Chunks are folded **in chunk
//! order** into a [`StrayTally`], whose sums are compensated (Neumaier,
//! CONVENTIONS §7.5). The result is therefore independent of the number of
//! threads.
//!
//! # Energy identity
//!
//! Every unit of emitted flux ends in exactly one sink: detected (by class),
//! absorbed (by [`Sink`]), escaped (left the scene from the ambient medium),
//! escaped from a solid (left the scene while inside a non-ambient medium:
//! a leak through a seam or an open solid, reported separately and warned
//! about), truncated (event limit) or a scatter sampling failure. Variance-reduction steps change weights
//! without changing the expectation: Russian roulette (survivors gain,
//! killed paths lose) and biased branch selection (`w·p/q` instead of
//! `w·Σp`). Both are booked as an explicit **MC adjustment**. So
//!
//! ```text
//! emitted + mc_adjustment = detected + absorbed + escaped
//!                            + escaped_from_solid + truncated + failed
//! ```
//!
//! holds exactly (to rounding) for every run, and `E[mc_adjustment] = 0`.
//! [`EnergyReport`] checks both.
//!
//! # Monte-Carlo uncertainty
//!
//! Source rays are independent, so each detected quantity is a sum
//! `F = Σᵢ xᵢ` of per-ray contributions (`xᵢ` is everything ray `i`'s whole
//! path tree, split children included, delivered to that class or pixel).
//! The tally accumulates `Σ xᵢ²` per class, for the owned classes, for the
//! total and per pixel, and reports the standard error
//!
//! ```text
//! SE(F) = √( N/(N−1) · (Σ xᵢ² − F²/N) )
//! ```
//!
//! over all `N` source rays (rays that deliver nothing count as `xᵢ = 0`).
//!
//! The same per-ray bookkeeping gives
//!
//! * the standard error of the **MC adjustment** (`xᵢ` = net roulette and
//!   reweighting change of ray `i`, zero in expectation), so energy closure
//!   can be tested against its own Monte-Carlo error;
//! * **image moments** per class: `F = Σ w`, `Sx = Σ w x`, `Sy = Σ w y`,
//!   `Sr = Σ w (x² + y²)` over detector hits in the detector's local frame,
//!   with the 4 × 4 matrix `Q` of per-ray cross products. The centroid
//!   `(Sx/F, Sy/F)` and the RMS radius about it,
//!   `√(Sr/F − (Sx/F)² − (Sy/F)²)`, are ratio estimators; their standard
//!   errors are the delta-method (linearised) errors
//!   `SE(g)² = N/(N−1) · (∇gᵀ Q ∇g − (∇gᵀ S)²/N)`, `S` being the four sums;
//! * flux, hits and standard error per class and **path signature**
//!   ([`crate::path::Signature`]).
//!
//! An extra *owned* slot (index `NUM_CLASSES`) accumulates the classes of
//! the owned mask together, so the owned sum map, total and moments carry
//! correct standard errors (one ray's contributions to several classes are
//! correlated).

use std::collections::BTreeMap;

use mklight_core::math::KahanSum;
use serde::{Deserialize, Serialize};

use crate::path::{ClassMask, EventKind, History, NUM_CLASSES, PathClass, Signature};
use crate::scene::{DetectorSpec, OpaqueKind};

/// Where absorbed flux went.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sink {
    /// Beer–Lambert absorption inside a medium.
    Bulk = 0,
    /// Absorption in a coating (`A` of `R + T + A = 1`).
    Coating = 1,
    /// The back of a mirror.
    MirrorBack = 2,
    /// The back of the detector.
    DetectorBack = 3,
    LensEdge = 4,
    LensFlat = 5,
    StopPlate = 6,
    Barrel = 7,
    Baffle = 8,
    Generic = 9,
    /// Non-clear (mount) zones of optical surfaces.
    Mount = 10,
    /// Scalar order power that has no propagating grating solution.
    Diffraction = 11,
}

pub const NUM_SINKS: usize = 12;

impl Sink {
    pub const ALL: [Sink; NUM_SINKS] = [
        Sink::Bulk,
        Sink::Coating,
        Sink::MirrorBack,
        Sink::DetectorBack,
        Sink::LensEdge,
        Sink::LensFlat,
        Sink::StopPlate,
        Sink::Barrel,
        Sink::Baffle,
        Sink::Generic,
        Sink::Mount,
        Sink::Diffraction,
    ];
    pub fn of(kind: OpaqueKind) -> Sink {
        match kind {
            OpaqueKind::LensEdge => Sink::LensEdge,
            OpaqueKind::LensFlat => Sink::LensFlat,
            OpaqueKind::StopPlate => Sink::StopPlate,
            OpaqueKind::Barrel => Sink::Barrel,
            OpaqueKind::Baffle => Sink::Baffle,
            OpaqueKind::Generic => Sink::Generic,
            OpaqueKind::Mount => Sink::Mount,
        }
    }
}

/// Highest ghost order tallied separately (higher orders are pooled here).
pub const MAX_GHOST_ORDER: usize = 8;

/// Class slots plus the owned and uncovered-ghost slots.
const SLOTS: usize = NUM_CLASSES + 2;
/// Index of the owned slot.
const OWNED: usize = NUM_CLASSES;
/// Ghost detections outside the explicit run's coverage.
const UNCOVERED: usize = NUM_CLASSES + 1;
/// Unique entries of a symmetric 4 × 4 matrix, row-major upper triangle.
const QN: usize = 10;

/// Upper-triangle index of `(a, b)` in a symmetric 4 × 4 matrix.
#[inline]
fn qi(a: usize, b: usize) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    // Rows start at 0, 4, 7, 9.
    [0, 4, 7, 9][a] + (b - a)
}

/// Adds the outer product of `v` to `q`.
#[inline]
fn add_outer(q: &mut [f64; QN], v: &[f64; 4]) {
    for a in 0..4 {
        for b in a..4 {
            q[qi(a, b)] += v[a] * v[b];
        }
    }
}

/// Per-signature accumulator of one chunk.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignatureAcc {
    pub flux: f64,
    pub flux_sq: f64,
    pub hits: u64,
}

/// Per-chunk sums (uncompensated; chunks are small).
#[derive(Clone, Debug)]
pub struct ChunkTally {
    pub emitted: f64,
    pub emitted_by_wavelength: Vec<f64>,
    pub detected: [f64; NUM_CLASSES],
    pub detected_hits: [u64; NUM_CLASSES],
    pub detected_by_wavelength: Vec<[f64; NUM_CLASSES]>,
    /// Detector hits `(pixel, class, flux)` in trace order (sparse, so a
    /// chunk stays small whatever the map size).
    pub hits: Vec<(u32, u8, f64)>,
    pub absorbed: [f64; NUM_SINKS],
    pub escaped: f64,
    pub escaped_from_solid: f64,
    pub truncated: f64,
    pub sampling_failed: f64,
    pub roulette_gain: f64,
    pub roulette_loss: f64,
    pub weighting_gain: f64,
    pub weighting_loss: f64,
    /// Ghost flux by number of stray reflections (index = order, ≤ MAX_GHOST_ORDER).
    pub ghost_by_order: [f64; MAX_GHOST_ORDER + 1],
    /// Ghost flux by the surfaces of the first two stray reflections.
    pub ghost_by_pair: BTreeMap<(u16, u16), f64>,
    pub rays: u64,
    pub events: u64,
    /// Classes whose summed per-ray contribution gets its own second moment.
    pub owned: ClassMask,
    /// Detections owned although their class is not in `owned` (`Ghost`
    /// paths outside a ghost run's coverage, `StraySettings::covered_ghosts`).
    pub detected_uncovered: f64,
    /// The other `Ghost`-class detections.
    pub detected_covered: f64,
    /// `Σ xᵢ²` per class, for the total and for the owned classes.
    pub detected_sq: [f64; NUM_CLASSES],
    pub total_sq: f64,
    pub owned_sq: f64,
    pub uncovered_sq: f64,
    pub covered_sq: f64,
    /// `(pixel, slot, xᵢ²)` per ray and pixel/slot it reached (slot
    /// `NUM_CLASSES` is the owned sum).
    pub hits_sq: Vec<(u32, u8, f64)>,
    /// `Σ xᵢ²` of the per-ray MC adjustment.
    pub adjustment_sq: f64,
    /// Image moment sums `[F, Sx, Sy, Sr]` per slot.
    pub moments: [[f64; 4]; SLOTS],
    /// Per-ray cross products of the moment sums per slot.
    pub moments_q: [[f64; QN]; SLOTS],
    /// Tally path signatures.
    pub track_signatures: bool,
    pub signatures: BTreeMap<(u8, Signature), SignatureAcc>,
    /// Contributions of the ray in flight.
    ray_class: [f64; NUM_CLASSES],
    ray_uncovered: f64,
    ray_covered: f64,
    ray_moments: [[f64; 4]; SLOTS],
    ray_adjustment: f64,
    ray_hits_start: usize,
    ray_sigs: Vec<(u8, Signature, f64)>,
    scratch: Vec<(u32, u8, f64)>,
}

impl ChunkTally {
    pub fn new(num_wavelengths: usize, owned: ClassMask) -> Self {
        ChunkTally {
            emitted: 0.0,
            emitted_by_wavelength: vec![0.0; num_wavelengths],
            detected: [0.0; NUM_CLASSES],
            detected_hits: [0; NUM_CLASSES],
            detected_by_wavelength: vec![[0.0; NUM_CLASSES]; num_wavelengths],
            hits: Vec::new(),
            absorbed: [0.0; NUM_SINKS],
            escaped: 0.0,
            escaped_from_solid: 0.0,
            truncated: 0.0,
            sampling_failed: 0.0,
            roulette_gain: 0.0,
            roulette_loss: 0.0,
            weighting_gain: 0.0,
            weighting_loss: 0.0,
            ghost_by_order: [0.0; MAX_GHOST_ORDER + 1],
            ghost_by_pair: BTreeMap::new(),
            rays: 0,
            events: 0,
            owned,
            detected_uncovered: 0.0,
            detected_covered: 0.0,
            detected_sq: [0.0; NUM_CLASSES],
            total_sq: 0.0,
            owned_sq: 0.0,
            uncovered_sq: 0.0,
            covered_sq: 0.0,
            hits_sq: Vec::new(),
            adjustment_sq: 0.0,
            moments: [[0.0; 4]; SLOTS],
            moments_q: [[0.0; QN]; SLOTS],
            track_signatures: false,
            signatures: BTreeMap::new(),
            ray_class: [0.0; NUM_CLASSES],
            ray_uncovered: 0.0,
            ray_covered: 0.0,
            ray_moments: [[0.0; 4]; SLOTS],
            ray_adjustment: 0.0,
            ray_hits_start: 0,
            ray_sigs: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Starts the per-ray accumulation of a new source ray.
    pub fn begin_ray(&mut self) {
        self.ray_class = [0.0; NUM_CLASSES];
        self.ray_uncovered = 0.0;
        self.ray_covered = 0.0;
        self.ray_moments = [[0.0; 4]; SLOTS];
        self.ray_adjustment = 0.0;
        self.ray_hits_start = self.hits.len();
        self.ray_sigs.clear();
    }

    /// Books the second moments of the finished source ray.
    pub fn end_ray(&mut self) {
        let x = self.ray_class;
        let mut total = 0.0;
        let mut owned = 0.0;
        for c in PathClass::ALL {
            let v = x[c.index()];
            self.detected_sq[c.index()] += v * v;
            total += v;
            if self.owned.contains(c) {
                owned += v;
            }
        }
        owned += self.ray_uncovered;
        self.total_sq += total * total;
        self.owned_sq += owned * owned;
        self.uncovered_sq += self.ray_uncovered * self.ray_uncovered;
        self.covered_sq += self.ray_covered * self.ray_covered;
        self.adjustment_sq += self.ray_adjustment * self.ray_adjustment;
        for k in 0..SLOTS {
            let m = self.ray_moments[k];
            if m != [0.0; 4] {
                add_outer(&mut self.moments_q[k], &m);
                for (a, b) in self.moments[k].iter_mut().zip(m) {
                    *a += b;
                }
            }
        }
        // Signatures: merge this ray's detections by key.
        if !self.ray_sigs.is_empty() {
            self.ray_sigs
                .sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
            let mut k = 0;
            while k < self.ray_sigs.len() {
                let (c, sig, _) = self.ray_sigs[k];
                let (mut v, mut n) = (0.0, 0u64);
                while k < self.ray_sigs.len()
                    && self.ray_sigs[k].0 == c
                    && self.ray_sigs[k].1 == sig
                {
                    v += self.ray_sigs[k].2;
                    n += 1;
                    k += 1;
                }
                let e = self.signatures.entry((c, sig)).or_default();
                e.flux += v;
                e.flux_sq += v * v;
                e.hits += n;
            }
        }
        // Per pixel and slot: merge this ray's hits by key.
        let hits = &self.hits[self.ray_hits_start..];
        if hits.is_empty() {
            return;
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(hits);
        let owned_mask = self.owned;
        self.scratch.extend(
            hits.iter()
                .filter(|h| {
                    (h.1 as usize) < NUM_CLASSES
                        && owned_mask.contains(PathClass::ALL[h.1 as usize])
                })
                .map(|&(p, _, w)| (p, OWNED as u8, w)),
        );
        self.scratch
            .sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
        let mut k = 0;
        while k < self.scratch.len() {
            let (p, c, _) = self.scratch[k];
            let mut v = 0.0;
            while k < self.scratch.len() && self.scratch[k].0 == p && self.scratch[k].1 == c {
                v += self.scratch[k].2;
                k += 1;
            }
            self.hits_sq.push((p, c, v * v));
        }
    }

    #[inline]
    pub fn absorb(&mut self, sink: Sink, w: f64) {
        self.absorbed[sink as usize] += w;
    }

    /// Books a weight change from biased branch selection.
    #[inline]
    pub fn reweight(&mut self, delta: f64) {
        if delta >= 0.0 {
            self.weighting_gain += delta;
        } else {
            self.weighting_loss -= delta;
        }
        self.ray_adjustment += delta;
    }

    /// Books a Russian-roulette weight change (`delta > 0`: a survivor's
    /// gain; `delta < 0`: a killed path's loss).
    #[inline]
    pub fn roulette(&mut self, delta: f64) {
        if delta >= 0.0 {
            self.roulette_gain += delta;
        } else {
            self.roulette_loss -= delta;
        }
        self.ray_adjustment += delta;
    }

    /// Books a detection of flux `w` at local detector point `(x, y)`.
    /// `owned`: the detection counts in the owned sum (always when its
    /// class is in the owned mask; also for an uncovered ghost path).
    #[allow(clippy::too_many_arguments)]
    pub fn detect(
        &mut self,
        class: PathClass,
        owned: bool,
        pixel: Option<usize>,
        w: f64,
        wavelength: usize,
        hist: &History,
        x: f64,
        y: f64,
    ) {
        let c = class.index();
        self.ray_class[c] += w;
        self.detected[c] += w;
        self.detected_hits[c] += 1;
        self.detected_by_wavelength[wavelength][c] += w;
        if let Some(p) = pixel {
            self.hits.push((p as u32, c as u8, w));
        }
        let owned = owned || self.owned.contains(class);
        if owned && !self.owned.contains(class) {
            // Owned outside the mask: booked to the owned slot directly
            // (the mask's classes are copied there when the ray ends).
            self.ray_uncovered += w;
            self.detected_uncovered += w;
            if let Some(p) = pixel {
                self.hits.push((p as u32, OWNED as u8, w));
                self.hits.push((p as u32, UNCOVERED as u8, w));
            }
        } else if class == PathClass::Ghost {
            self.ray_covered += w;
            self.detected_covered += w;
        }
        let m = [w, w * x, w * y, w * (x * x + y * y)];
        let slots = [c, OWNED];
        let n = if owned { 2 } else { 1 };
        for &k in &slots[..n] {
            for (a, b) in self.ray_moments[k].iter_mut().zip(m) {
                *a += b;
            }
        }
        if self.track_signatures {
            self.ray_sigs.push((c as u8, hist.signature, w));
        }
        if class == PathClass::Ghost {
            self.ghost_by_order[(hist.stray_reflections as usize).min(MAX_GHOST_ORDER)] += w;
            *self
                .ghost_by_pair
                .entry((hist.first_reflections[0], hist.first_reflections[1]))
                .or_insert(0.0) += w;
        }
    }
}

/// Run-level signature accumulator.
#[derive(Clone, Copy, Debug, Default)]
struct SignatureSum {
    flux: KahanSum,
    flux_sq: KahanSum,
    hits: u64,
}

/// Run-level accumulator with compensated sums.
#[derive(Clone, Debug)]
pub struct StrayTally {
    emitted: KahanSum,
    emitted_by_wavelength: Vec<KahanSum>,
    detected: [KahanSum; NUM_CLASSES],
    detected_hits: [u64; NUM_CLASSES],
    detected_by_wavelength: Vec<[KahanSum; NUM_CLASSES]>,
    /// Per slot (classes, then owned).
    maps: Vec<Vec<KahanSum>>,
    maps_sq: Vec<Vec<KahanSum>>,
    absorbed: [KahanSum; NUM_SINKS],
    escaped: KahanSum,
    escaped_from_solid: KahanSum,
    truncated: KahanSum,
    sampling_failed: KahanSum,
    detected_sq: [KahanSum; NUM_CLASSES],
    total_sq: KahanSum,
    owned_sq: KahanSum,
    uncovered_sq: KahanSum,
    covered_sq: KahanSum,
    detected_uncovered: KahanSum,
    detected_covered: KahanSum,
    adjustment_sq: KahanSum,
    moments: [[KahanSum; 4]; SLOTS],
    moments_q: [[KahanSum; QN]; SLOTS],
    roulette_gain: KahanSum,
    roulette_loss: KahanSum,
    weighting_gain: KahanSum,
    weighting_loss: KahanSum,
    ghost_by_order: [KahanSum; MAX_GHOST_ORDER + 1],
    ghost_by_pair: BTreeMap<(u16, u16), KahanSum>,
    signatures: BTreeMap<(u8, Signature), SignatureSum>,
    rays: u64,
    events: u64,
}

impl StrayTally {
    pub fn new(num_wavelengths: usize, pixels: usize) -> Self {
        let z = KahanSum::default();
        StrayTally {
            emitted: z,
            emitted_by_wavelength: vec![z; num_wavelengths],
            detected: [z; NUM_CLASSES],
            detected_hits: [0; NUM_CLASSES],
            detected_by_wavelength: vec![[z; NUM_CLASSES]; num_wavelengths],
            maps: vec![vec![z; pixels]; SLOTS],
            maps_sq: vec![vec![z; pixels]; SLOTS],
            absorbed: [z; NUM_SINKS],
            escaped: z,
            escaped_from_solid: z,
            truncated: z,
            sampling_failed: z,
            detected_sq: [z; NUM_CLASSES],
            total_sq: z,
            owned_sq: z,
            uncovered_sq: z,
            covered_sq: z,
            detected_uncovered: z,
            detected_covered: z,
            adjustment_sq: z,
            moments: [[z; 4]; SLOTS],
            moments_q: [[z; QN]; SLOTS],
            roulette_gain: z,
            roulette_loss: z,
            weighting_gain: z,
            weighting_loss: z,
            ghost_by_order: [z; MAX_GHOST_ORDER + 1],
            ghost_by_pair: BTreeMap::new(),
            signatures: BTreeMap::new(),
            rays: 0,
            events: 0,
        }
    }

    /// Folds one chunk in. Call in chunk order for reproducible sums.
    pub fn add(&mut self, c: &ChunkTally) {
        self.emitted.add(c.emitted);
        for (a, b) in self
            .emitted_by_wavelength
            .iter_mut()
            .zip(&c.emitted_by_wavelength)
        {
            a.add(*b);
        }
        for k in 0..NUM_CLASSES {
            self.detected[k].add(c.detected[k]);
            self.detected_hits[k] += c.detected_hits[k];
        }
        for &(p, k, w) in &c.hits {
            self.maps[k as usize][p as usize].add(w);
            if (k as usize) < NUM_CLASSES && c.owned.contains(PathClass::ALL[k as usize]) {
                self.maps[OWNED][p as usize].add(w);
            }
        }
        for &(p, k, w2) in &c.hits_sq {
            self.maps_sq[k as usize][p as usize].add(w2);
        }
        for k in 0..NUM_CLASSES {
            self.detected_sq[k].add(c.detected_sq[k]);
        }
        self.total_sq.add(c.total_sq);
        self.owned_sq.add(c.owned_sq);
        self.uncovered_sq.add(c.uncovered_sq);
        self.covered_sq.add(c.covered_sq);
        self.detected_uncovered.add(c.detected_uncovered);
        self.detected_covered.add(c.detected_covered);
        self.adjustment_sq.add(c.adjustment_sq);
        for k in 0..SLOTS {
            for (a, b) in self.moments[k].iter_mut().zip(&c.moments[k]) {
                a.add(*b);
            }
            for (a, b) in self.moments_q[k].iter_mut().zip(&c.moments_q[k]) {
                a.add(*b);
            }
        }
        for (a, b) in self
            .detected_by_wavelength
            .iter_mut()
            .zip(&c.detected_by_wavelength)
        {
            for k in 0..NUM_CLASSES {
                a[k].add(b[k]);
            }
        }
        for k in 0..NUM_SINKS {
            self.absorbed[k].add(c.absorbed[k]);
        }
        self.escaped.add(c.escaped);
        self.escaped_from_solid.add(c.escaped_from_solid);
        self.truncated.add(c.truncated);
        self.sampling_failed.add(c.sampling_failed);
        self.roulette_gain.add(c.roulette_gain);
        self.roulette_loss.add(c.roulette_loss);
        self.weighting_gain.add(c.weighting_gain);
        self.weighting_loss.add(c.weighting_loss);
        for k in 0..=MAX_GHOST_ORDER {
            self.ghost_by_order[k].add(c.ghost_by_order[k]);
        }
        for (key, v) in &c.ghost_by_pair {
            self.ghost_by_pair.entry(*key).or_default().add(*v);
        }
        for (key, v) in &c.signatures {
            let e = self.signatures.entry(*key).or_default();
            e.flux.add(v.flux);
            e.flux_sq.add(v.flux_sq);
            e.hits += v.hits;
        }
        self.rays += c.rays;
        self.events += c.events;
    }

    /// Standard error of a sum over the run's `N` independent rays, from the
    /// sum and the sum of per-ray squares (module docs).
    fn se(&self, sum: f64, sq: f64) -> f64 {
        if self.rays < 2 {
            return f64::INFINITY;
        }
        let n = self.rays as f64;
        (n / (n - 1.0) * (sq - sum * sum / n)).max(0.0).sqrt()
    }

    /// Linearised standard error of `g(S)` with gradient `grad` (module docs).
    fn se_linear(&self, slot: usize, grad: [f64; 4]) -> f64 {
        if self.rays < 2 {
            return f64::INFINITY;
        }
        let n = self.rays as f64;
        let s: Vec<f64> = self.moments[slot].iter().map(|k| k.value()).collect();
        let q = &self.moments_q[slot];
        let mut gqg = 0.0;
        for a in 0..4 {
            for b in 0..4 {
                gqg += grad[a] * grad[b] * q[qi(a, b)].value();
            }
        }
        let gs: f64 = (0..4).map(|a| grad[a] * s[a]).sum();
        (n / (n - 1.0) * (gqg - gs * gs / n)).max(0.0).sqrt()
    }

    fn image_moments(&self, slot: usize) -> ImageMoments {
        let [f, sx, sy, sr] = self.moments[slot].map(|k| k.value());
        if f <= 0.0 {
            return ImageMoments::default();
        }
        let (cx, cy) = (sx / f, sy / f);
        let m = sr / f;
        let var = (m - cx * cx - cy * cy).max(0.0);
        let rms = var.sqrt();
        let se_cx = self.se_linear(slot, [-cx / f, 1.0 / f, 0.0, 0.0]);
        let se_cy = self.se_linear(slot, [-cy / f, 0.0, 1.0 / f, 0.0]);
        let se_rms = if rms > 0.0 {
            let h = 0.5 / (rms * f);
            self.se_linear(
                slot,
                [
                    h * (-m + 2.0 * (cx * cx + cy * cy)),
                    -2.0 * cx * h,
                    -2.0 * cy * h,
                    h,
                ],
            )
        } else {
            f64::INFINITY
        };
        ImageMoments {
            flux: f,
            centroid: [cx, cy],
            centroid_std_error: [se_cx, se_cy],
            rms_radius: rms,
            rms_radius_std_error: se_rms,
        }
    }

    /// Final numbers.
    pub fn finish(
        &self,
        spec: &DetectorSpec,
        owned: ClassMask,
    ) -> (DetectorResult, EnergyReport, Vec<SignatureFlux>) {
        let v = |k: &KahanSum| k.value();
        let detected_by_class: Vec<f64> = self.detected.iter().map(v).collect();
        let absorbed_by_sink: Vec<(Sink, f64)> = Sink::ALL
            .iter()
            .map(|s| (*s, self.absorbed[*s as usize].value()))
            .collect();
        let sum = |xs: &mut dyn Iterator<Item = f64>| xs.collect::<KahanSum>().value();
        let detected = sum(&mut detected_by_class.iter().copied());
        let absorbed = sum(&mut absorbed_by_sink.iter().map(|x| x.1));
        let emitted = self.emitted.value();
        let escaped = self.escaped.value();
        let escaped_from_solid = self.escaped_from_solid.value();
        let truncated = self.truncated.value();
        let failed = self.sampling_failed.value();
        let mc_adjustment = sum(&mut [
            self.roulette_gain.value(),
            -self.roulette_loss.value(),
            self.weighting_gain.value(),
            -self.weighting_loss.value(),
        ]
        .into_iter());
        let residual = sum(&mut [
            emitted,
            mc_adjustment,
            -detected,
            -absorbed,
            -escaped,
            -escaped_from_solid,
            -truncated,
            -failed,
        ]
        .into_iter());
        let mut owned_flux = sum(&mut PathClass::ALL
            .iter()
            .filter(|c| owned.contains(**c))
            .map(|c| detected_by_class[c.index()]));
        let uncovered = self.detected_uncovered.value();
        if uncovered != 0.0 {
            owned_flux = sum(&mut [owned_flux, uncovered].into_iter());
        }
        let report = EnergyReport {
            emitted,
            emitted_by_wavelength: self.emitted_by_wavelength.iter().map(v).collect(),
            detected,
            detected_by_class: PathClass::ALL
                .iter()
                .map(|c| (*c, detected_by_class[c.index()]))
                .collect(),
            detected_std_error: self.se(detected, self.total_sq.value()),
            detected_std_error_by_class: PathClass::ALL
                .iter()
                .map(|c| {
                    (
                        *c,
                        self.se(
                            detected_by_class[c.index()],
                            self.detected_sq[c.index()].value(),
                        ),
                    )
                })
                .collect(),
            detected_hits_by_class: PathClass::ALL
                .iter()
                .map(|c| (*c, self.detected_hits[c.index()]))
                .collect(),
            detected_by_wavelength: self
                .detected_by_wavelength
                .iter()
                .map(|a| a.iter().map(v).collect())
                .collect(),
            owned_classes: owned,
            detected_owned: owned_flux,
            detected_owned_std_error: self.se(owned_flux, self.owned_sq.value()),
            detected_uncovered_ghost: uncovered,
            detected_uncovered_ghost_std_error: self.se(uncovered, self.uncovered_sq.value()),
            detected_covered_ghost: self.detected_covered.value(),
            detected_covered_ghost_std_error: self
                .se(self.detected_covered.value(), self.covered_sq.value()),
            absorbed,
            absorbed_by_sink,
            escaped,
            escaped_from_solid,
            truncated,
            sampling_failed: failed,
            roulette_gain: self.roulette_gain.value(),
            roulette_loss: self.roulette_loss.value(),
            weighting_gain: self.weighting_gain.value(),
            weighting_loss: self.weighting_loss.value(),
            mc_adjustment,
            mc_adjustment_std_error: self.se(mc_adjustment, self.adjustment_sq.value()),
            balance_residual: residual,
            ghost_by_order: self.ghost_by_order.iter().map(v).collect(),
            ghost_by_pair: self
                .ghost_by_pair
                .iter()
                .map(|(k, s)| (k.0, k.1, s.value()))
                .collect(),
            rays: self.rays,
            events: self.events,
        };
        let area = spec.pixel_area();
        let maps: Vec<Vec<f64>> = self
            .maps
            .iter()
            .map(|m| m.iter().map(v).collect())
            .collect();
        let mut errors: Vec<Vec<f64>> = maps
            .iter()
            .zip(&self.maps_sq)
            .map(|(m, q)| {
                m.iter()
                    .zip(q)
                    .map(|(f, s)| self.se(*f, s.value()))
                    .collect()
            })
            .collect();
        let mut flux = maps;
        let uncovered_map = flux.pop().expect("uncovered slot");
        let uncovered_err = errors.pop().expect("uncovered slot");
        let owned_map = flux.pop().expect("owned slot");
        let owned_err = errors.pop().expect("owned slot");
        let irr = |m: &[f64]| m.iter().map(|f| f / area).collect::<Vec<f64>>();
        let det = DetectorResult {
            spec: spec.clone(),
            irradiance: flux.iter().map(|m| irr(m)).collect(),
            flux,
            flux_std_error: errors,
            owned_classes: owned,
            uncovered_ghost_irradiance: irr(&uncovered_map),
            uncovered_ghost_flux_std_error: uncovered_err,
            owned_irradiance: irr(&owned_map),
            owned_flux: owned_map,
            owned_flux_std_error: owned_err,
            moments: PathClass::ALL
                .iter()
                .map(|c| self.image_moments(c.index()))
                .collect(),
            owned_moments: self.image_moments(OWNED),
        };
        let mut sigs: Vec<SignatureFlux> = self
            .signatures
            .iter()
            .map(|((c, sig), acc)| SignatureFlux {
                class: PathClass::ALL[*c as usize],
                events: sig.events().collect(),
                diffraction: sig.diffraction().collect(),
                overflow: sig.overflow(),
                label: String::new(),
                flux: acc.flux.value(),
                std_error: self.se(acc.flux.value(), acc.flux_sq.value()),
                hits: acc.hits,
                signature: *sig,
            })
            .collect();
        // Largest flux first; ties by class and signature (deterministic).
        sigs.sort_by(|a, b| {
            b.flux
                .total_cmp(&a.flux)
                .then(a.class.cmp(&b.class))
                .then(a.signature.cmp(&b.signature))
        });
        (det, report, sigs)
    }
}

/// Flux-weighted image moments of one class (or the owned sum) on the
/// detector, in its local frame (mm), with delta-method standard errors
/// (module docs). All zero when nothing was detected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageMoments {
    /// Detected flux (W).
    pub flux: f64,
    pub centroid: [f64; 2],
    pub centroid_std_error: [f64; 2],
    /// RMS radius about the centroid.
    pub rms_radius: f64,
    pub rms_radius_std_error: f64,
}

/// Detected flux of one (class, path signature) pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignatureFlux {
    pub class: PathClass,
    /// Ordered non-designed events (`path::Signature`).
    pub events: Vec<(EventKind, usize)>,
    pub diffraction: Vec<crate::path::DiffractionEvent>,
    /// More events happened than were kept.
    pub overflow: bool,
    /// Human-readable form (`Signature::describe`).
    pub label: String,
    /// W.
    pub flux: f64,
    /// Monte-Carlo standard error of `flux`.
    pub std_error: f64,
    /// Detector hits (split copies count separately).
    pub hits: u64,
    #[serde(skip)]
    pub signature: Signature,
}

/// Detector maps per class, row-major `ny × nx` in the detector's local
/// `(x, y)` (row 0 at `y = −half_y`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorResult {
    pub spec: DetectorSpec,
    /// `flux[class][pixel]` in W.
    pub flux: Vec<Vec<f64>>,
    /// `irradiance[class][pixel]` in W/mm².
    pub irradiance: Vec<Vec<f64>>,
    /// Monte-Carlo standard error of `flux[class][pixel]` in W (divide by
    /// `spec.pixel_area()` for irradiance).
    pub flux_std_error: Vec<Vec<f64>>,
    /// The classes summed in the `owned_*` maps.
    pub owned_classes: ClassMask,
    /// Sum of the owned classes' flux maps (W), with its own standard error
    /// (per-ray contributions to several classes are correlated, so it is
    /// not the root-sum-square of the class errors).
    pub owned_flux: Vec<f64>,
    pub owned_flux_std_error: Vec<f64>,
    /// W/mm².
    pub owned_irradiance: Vec<f64>,
    /// Stray-owned subset of the Ghost class (W/mm²).
    pub uncovered_ghost_irradiance: Vec<f64>,
    /// Per-pixel standard error of the uncovered ghost flux (W).
    pub uncovered_ghost_flux_std_error: Vec<f64>,
    /// Image moments per class (`PathClass::index`).
    pub moments: Vec<ImageMoments>,
    /// Image moments of the owned sum.
    pub owned_moments: ImageMoments,
}

impl DetectorResult {
    pub fn flux_of(&self, class: PathClass) -> &[f64] {
        &self.flux[class.index()]
    }
    pub fn moments_of(&self, class: PathClass) -> &ImageMoments {
        &self.moments[class.index()]
    }
    /// Sum of the irradiance maps of the classes in `mask` (no standard
    /// error; use `owned_*` for the owned sum).
    pub fn irradiance_sum(&self, mask: ClassMask) -> Vec<f64> {
        let n = self.irradiance[0].len();
        (0..n)
            .map(|p| {
                PathClass::ALL
                    .iter()
                    .filter(|c| mask.contains(**c))
                    .map(|c| self.irradiance[c.index()][p])
                    .sum()
            })
            .collect()
    }
}

/// Where the emitted flux went (all in W).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyReport {
    pub emitted: f64,
    pub emitted_by_wavelength: Vec<f64>,
    pub detected: f64,
    /// Monte-Carlo standard error of `detected`.
    pub detected_std_error: f64,
    pub detected_by_class: Vec<(PathClass, f64)>,
    /// Monte-Carlo standard error per class.
    pub detected_std_error_by_class: Vec<(PathClass, f64)>,
    pub detected_hits_by_class: Vec<(PathClass, u64)>,
    /// `[wavelength][class]`.
    pub detected_by_wavelength: Vec<Vec<f64>>,
    pub owned_classes: ClassMask,
    /// Detected flux in `owned_classes`, plus `detected_uncovered_ghost`.
    pub detected_owned: f64,
    pub detected_owned_std_error: f64,
    /// `Ghost`-class flux outside the coverage of the ghost run the stray
    /// run is combined with (`StraySettings::covered_ghosts`): owned by the
    /// stray solver even when `Ghost` is not in `owned_classes`.
    #[serde(default)]
    pub detected_uncovered_ghost: f64,
    #[serde(default)]
    pub detected_uncovered_ghost_std_error: f64,
    /// Remaining Ghost-class flux; the explicit solver owns it when the
    /// Ghost class is excluded. Errors are accumulated per source ray.
    #[serde(default)]
    pub detected_covered_ghost: f64,
    #[serde(default)]
    pub detected_covered_ghost_std_error: f64,
    pub absorbed: f64,
    pub absorbed_by_sink: Vec<(Sink, f64)>,
    /// Left the scene from the ambient medium.
    pub escaped: f64,
    /// Left the scene while inside a non-ambient medium: a leak out of a
    /// solid (seam, open element) or an open non-ambient image space.
    pub escaped_from_solid: f64,
    pub truncated: f64,
    pub sampling_failed: f64,
    pub roulette_gain: f64,
    pub roulette_loss: f64,
    pub weighting_gain: f64,
    pub weighting_loss: f64,
    /// Net weight added by Russian roulette and biased branch selection
    /// (zero in expectation).
    pub mc_adjustment: f64,
    /// Monte-Carlo standard error of `mc_adjustment` (its expectation is
    /// zero, so `|mc_adjustment|` should be a few of these at most).
    pub mc_adjustment_std_error: f64,
    /// `emitted + mc_adjustment − (detected + absorbed + escaped +
    /// escaped_from_solid + truncated + sampling_failed)`. Zero to rounding.
    pub balance_residual: f64,
    pub ghost_by_order: Vec<f64>,
    /// `(first surface, second surface, flux)`.
    pub ghost_by_pair: Vec<(u16, u16, f64)>,
    pub rays: u64,
    pub events: u64,
}

impl EnergyReport {
    pub fn detected_in(&self, class: PathClass) -> f64 {
        self.detected_by_class[class.index()].1
    }
    pub fn absorbed_in(&self, sink: Sink) -> f64 {
        self.absorbed_by_sink[sink as usize].1
    }
    /// Monte-Carlo standard error of `detected_in(class)`.
    pub fn std_error_in(&self, class: PathClass) -> f64 {
        self.detected_std_error_by_class[class.index()].1
    }
    /// Relative standard error of `detected_in(class)` (∞ when nothing was
    /// detected).
    pub fn relative_error_in(&self, class: PathClass) -> f64 {
        let f = self.detected_in(class);
        if f > 0.0 {
            self.std_error_in(class) / f
        } else {
            f64::INFINITY
        }
    }
    /// `|balance_residual| / emitted`.
    pub fn relative_residual(&self) -> f64 {
        if self.emitted > 0.0 {
            self.balance_residual.abs() / self.emitted
        } else {
            0.0
        }
    }
    /// True when the books balance to rounding (`≤ 1e-12` relative).
    pub fn is_balanced(&self) -> bool {
        self.relative_residual() <= 1e-12
    }
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let pct = |x: f64| 100.0 * x / self.emitted.max(f64::MIN_POSITIVE);
        let mut s = format!(
            "emitted {:.6e} W over {} rays ({} events)\n",
            self.emitted, self.rays, self.events
        );
        s += &format!(
            "  detected  {:.6e} ± {:.2e} W ({:.4} %)\n",
            self.detected,
            self.detected_std_error,
            pct(self.detected)
        );
        for ((c, f), (_, e)) in self
            .detected_by_class
            .iter()
            .zip(&self.detected_std_error_by_class)
        {
            s += &format!(
                "    {:<10} {:.6e} ± {:.2e} W ({:.4} %)\n",
                format!("{c:?}"),
                f,
                e,
                pct(*f)
            );
        }
        s += &format!(
            "    owned      {:.6e} ± {:.2e} W\n",
            self.detected_owned, self.detected_owned_std_error
        );
        s += &format!(
            "  absorbed  {:.6e} W ({:.4} %)\n",
            self.absorbed,
            pct(self.absorbed)
        );
        for (k, f) in &self.absorbed_by_sink {
            if *f != 0.0 {
                s += &format!("    {:<12} {:.6e} W\n", format!("{k:?}"), f);
            }
        }
        s += &format!(
            "  escaped   {:.6e} W ({:.4} %), from inside solids {:.6e} W\n",
            self.escaped,
            pct(self.escaped),
            self.escaped_from_solid
        );
        s += &format!(
            "  truncated {:.6e} W, sampling failures {:.6e} W\n",
            self.truncated, self.sampling_failed
        );
        s += &format!(
            "  MC adjustment {:+.6e} ± {:.2e} W ({:+.4} %), balance residual {:.3e} (relative {:.3e})\n",
            self.mc_adjustment,
            self.mc_adjustment_std_error,
            pct(self.mc_adjustment),
            self.balance_residual,
            self.relative_residual()
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Accept;

    #[test]
    fn partition_errors_keep_split_path_covariance() {
        let mask = ClassMask::STRAY_WITH_GHOST;
        let mut chunk = ChunkTally::new(1, mask);
        // Three independent source rays, each split among covered ghosts,
        // uncovered ghosts and sneak paths. The first ray has two hits in
        // the same uncovered signature/pixel: square their sum, not each hit.
        for (covered, uncovered, sneak) in [(1.0, 2.0, 1.0), (4.0, 1.0, 4.0), (2.0, 0.0, 0.0)] {
            chunk.begin_ray();
            chunk.rays += 1;
            let hist = History::default();
            chunk.detect(
                PathClass::Ghost,
                false,
                Some(0),
                covered,
                0,
                &hist,
                0.0,
                0.0,
            );
            for _ in 0..2 {
                chunk.detect(
                    PathClass::Ghost,
                    true,
                    Some(0),
                    uncovered * 0.5,
                    0,
                    &hist,
                    0.0,
                    0.0,
                );
            }
            chunk.detect(PathClass::Sneak, true, Some(0), sneak, 0, &hist, 0.0, 0.0);
            chunk.end_ray();
        }
        let spec = DetectorSpec {
            surface: 0,
            accept: Accept::Both,
            half_x: 0.5,
            half_y: 0.5,
            nx: 1,
            ny: 1,
        };
        let mut tally = StrayTally::new(1, 1);
        tally.add(&chunk);
        let (map, energy, _) = tally.finish(&spec, mask);
        // Owned contributions [3,5,0]: sum 8, SE sqrt(19).
        // Uncovered [2,1,0]: sum 3, SE sqrt(3).
        // Covered [1,4,2]: sum 7, SE sqrt(7).
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        close(energy.detected_owned, 8.0);
        close(map.owned_flux[0], 8.0);
        close(map.owned_moments.flux, 8.0);
        close(energy.detected_owned_std_error, 19.0f64.sqrt());
        close(map.owned_flux_std_error[0], 19.0f64.sqrt());
        close(energy.detected_uncovered_ghost, 3.0);
        close(map.uncovered_ghost_irradiance[0], 3.0);
        close(energy.detected_uncovered_ghost_std_error, 3.0f64.sqrt());
        close(map.uncovered_ghost_flux_std_error[0], 3.0f64.sqrt());
        close(energy.detected_covered_ghost, 7.0);
        close(energy.detected_covered_ghost_std_error, 7.0f64.sqrt());
    }
}
