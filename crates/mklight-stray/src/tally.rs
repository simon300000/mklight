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

use std::collections::BTreeMap;

use mklight_core::math::KahanSum;
use serde::{Deserialize, Serialize};

use crate::path::{ClassMask, NUM_CLASSES, PathClass};
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
}

pub const NUM_SINKS: usize = 11;

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
    /// `Σ xᵢ²` per class, for the total and for the owned classes.
    pub detected_sq: [f64; NUM_CLASSES],
    pub total_sq: f64,
    pub owned_sq: f64,
    /// `(pixel, class, xᵢ²)` per ray and pixel/class it reached.
    pub hits_sq: Vec<(u32, u8, f64)>,
    /// Contributions of the ray in flight.
    ray_class: [f64; NUM_CLASSES],
    ray_hits_start: usize,
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
            detected_sq: [0.0; NUM_CLASSES],
            total_sq: 0.0,
            owned_sq: 0.0,
            hits_sq: Vec::new(),
            ray_class: [0.0; NUM_CLASSES],
            ray_hits_start: 0,
            scratch: Vec::new(),
        }
    }

    /// Starts the per-ray accumulation of a new source ray.
    pub fn begin_ray(&mut self) {
        self.ray_class = [0.0; NUM_CLASSES];
        self.ray_hits_start = self.hits.len();
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
        self.total_sq += total * total;
        self.owned_sq += owned * owned;
        // Per pixel and class: merge this ray's hits by key.
        let hits = &self.hits[self.ray_hits_start..];
        if hits.is_empty() {
            return;
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(hits);
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
    }

    #[allow(clippy::too_many_arguments)]
    pub fn detect(
        &mut self,
        class: PathClass,
        pixel: Option<usize>,
        w: f64,
        wavelength: usize,
        ghost_order: u8,
        ghost_pair: [u16; 2],
    ) {
        let c = class.index();
        self.ray_class[c] += w;
        self.detected[c] += w;
        self.detected_hits[c] += 1;
        self.detected_by_wavelength[wavelength][c] += w;
        if let Some(p) = pixel {
            self.hits.push((p as u32, c as u8, w));
        }
        if class == PathClass::Ghost {
            self.ghost_by_order[(ghost_order as usize).min(MAX_GHOST_ORDER)] += w;
            *self
                .ghost_by_pair
                .entry((ghost_pair[0], ghost_pair[1]))
                .or_insert(0.0) += w;
        }
    }
}

/// Run-level accumulator with compensated sums.
#[derive(Clone, Debug)]
pub struct StrayTally {
    emitted: KahanSum,
    emitted_by_wavelength: Vec<KahanSum>,
    detected: [KahanSum; NUM_CLASSES],
    detected_hits: [u64; NUM_CLASSES],
    detected_by_wavelength: Vec<[KahanSum; NUM_CLASSES]>,
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
    roulette_gain: KahanSum,
    roulette_loss: KahanSum,
    weighting_gain: KahanSum,
    weighting_loss: KahanSum,
    ghost_by_order: [KahanSum; MAX_GHOST_ORDER + 1],
    ghost_by_pair: BTreeMap<(u16, u16), KahanSum>,
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
            maps: vec![vec![z; pixels]; NUM_CLASSES],
            maps_sq: vec![vec![z; pixels]; NUM_CLASSES],
            absorbed: [z; NUM_SINKS],
            escaped: z,
            escaped_from_solid: z,
            truncated: z,
            sampling_failed: z,
            detected_sq: [z; NUM_CLASSES],
            total_sq: z,
            owned_sq: z,
            roulette_gain: z,
            roulette_loss: z,
            weighting_gain: z,
            weighting_loss: z,
            ghost_by_order: [z; MAX_GHOST_ORDER + 1],
            ghost_by_pair: BTreeMap::new(),
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
        }
        for &(p, k, w2) in &c.hits_sq {
            self.maps_sq[k as usize][p as usize].add(w2);
        }
        for k in 0..NUM_CLASSES {
            self.detected_sq[k].add(c.detected_sq[k]);
        }
        self.total_sq.add(c.total_sq);
        self.owned_sq.add(c.owned_sq);
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
        self.rays += c.rays;
        self.events += c.events;
    }

    /// Final numbers.
    pub fn finish(&self, spec: &DetectorSpec, owned: ClassMask) -> (DetectorResult, EnergyReport) {
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
        let owned_flux = sum(&mut PathClass::ALL
            .iter()
            .filter(|c| owned.contains(**c))
            .map(|c| detected_by_class[c.index()]));
        let n = self.rays as f64;
        // Standard error of a sum over N independent rays (module docs).
        let se = |sum: f64, sq: f64| {
            if self.rays < 2 {
                return f64::INFINITY;
            }
            (n / (n - 1.0) * (sq - sum * sum / n)).max(0.0).sqrt()
        };
        let report = EnergyReport {
            emitted,
            emitted_by_wavelength: self.emitted_by_wavelength.iter().map(v).collect(),
            detected,
            detected_by_class: PathClass::ALL
                .iter()
                .map(|c| (*c, detected_by_class[c.index()]))
                .collect(),
            detected_std_error: se(detected, self.total_sq.value()),
            detected_std_error_by_class: PathClass::ALL
                .iter()
                .map(|c| {
                    (
                        *c,
                        se(
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
            detected_owned_std_error: se(owned_flux, self.owned_sq.value()),
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
        let flux_maps: Vec<Vec<f64>> = self
            .maps
            .iter()
            .map(|m| m.iter().map(v).collect())
            .collect();
        let flux_std_error = flux_maps
            .iter()
            .zip(&self.maps_sq)
            .map(|(m, q)| m.iter().zip(q).map(|(f, s)| se(*f, s.value())).collect())
            .collect();
        let det = DetectorResult {
            spec: spec.clone(),
            flux_std_error,
            irradiance: flux_maps
                .iter()
                .map(|m| m.iter().map(|f| f / area).collect())
                .collect(),
            flux: flux_maps,
        };
        (det, report)
    }
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
}

impl DetectorResult {
    pub fn flux_of(&self, class: PathClass) -> &[f64] {
        &self.flux[class.index()]
    }
    /// Sum of the irradiance maps of the classes in `mask`.
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
    /// Detected flux in `owned_classes`.
    pub detected_owned: f64,
    pub detected_owned_std_error: f64,
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
            "  MC adjustment {:+.6e} W ({:+.4} %), balance residual {:.3e} (relative {:.3e})\n",
            self.mc_adjustment,
            pct(self.mc_adjustment),
            self.balance_residual,
            self.relative_residual()
        );
        s
    }
}
