//! The Monte-Carlo path tracer.
//!
//! # Transport
//!
//! A ray carries a flux weight, a wavelength index (drawn ∝ the system
//! wavelength weights) and the id of the medium it is in. Polarisation is
//! averaged: every interface uses the unpolarised `R`, `T` (CONVENTIONS §5).
//! At each step:
//!
//! 1. Closest hit through the [`RayCaster`]. A miss means the ray escapes
//!    (or, inside an absorbing medium, is absorbed).
//! 2. **Beer–Lambert** along the segment: the weight is multiplied by
//!    `exp(−α L)` and the difference is booked as bulk absorption (implicit,
//!    noise-free absorption).
//! 3. The interaction splits the flux into fractions that sum to 1:
//!    * **interface**: `R` and `T` from `ResolvedCoating::rt` (the same code
//!      the sequential and ghost solvers use), each split again into specular
//!      `(1 − TIS)` and scattered `TIS` parts by the surface's BSDF, plus the
//!      coating absorption `A`. If refraction is impossible (total internal
//!      reflection), `T` is moved into `R`. A mirror's back side absorbs.
//!    * **opaque**: scattered `TIS`, specular `min(ρ_s, 1 − TIS)`, the rest absorbed.
//!    * **detector**: tallied by path class and pixel when hit from an
//!      accepted side, otherwise absorbed.
//! 4. A branch is chosen by [`SplitMode`]:
//!    * `Analog` — branch (or absorption) with its physical probability. The
//!      weight is unchanged.
//!    * `Weighted` — absorption is implicit (`w·A` is booked). A branch is
//!      drawn with the defensive mixture `q = (1−m)·p/Σp + m/#branches`, and
//!      the weight becomes `w·p/q`. Rare branches (ghost reflections, scatter)
//!      are thereby sampled far more often than their probability.
//!    * `Split` — all branches are followed with weights `w·p` (up to
//!      `max_splits` splitting levels per path), then `Weighted`.
//! 5. **Russian roulette** when the weight falls below `roulette_threshold`
//!    times the initial weight: survive with probability `w/w_t` at weight `w_t`.
//!
//! All estimators are unbiased. The weight changes of steps 4–5 are booked
//! as the MC adjustment (see `tally`), so the energy books balance exactly.
//!
//! # Reproducibility and parallelism
//!
//! Source ray `i` uses its own Philox stream `(seed, i)` for everything on its
//! path tree, including split children (processed depth-first in a fixed
//! order). Rays are grouped in fixed-size chunks that run in parallel (rayon).
//! Chunk tallies are folded in chunk order with compensated sums, so results
//! are **bit-identical for any thread count**. Cancellation is checked every
//! 64 rays in each chunk and between batches. Progress is reported per batch.

use mklight_core::error::{MkError, MkResult};
use mklight_core::job::{JobContext, Precision, RunInfo, RunTimer};
use mklight_core::math::Vec3;
use mklight_core::rng::Philox;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bsdf::Bsdf;
use crate::geom::{Ray, RayCaster, onb, reflect, refract};
use crate::path::{ClassMask, History, flags};
use crate::scene::{Behavior, InterfaceKind, MediumId, Scene, Side};
use crate::source::SourceSpec;
use crate::tally::{ChunkTally, DetectorResult, EnergyReport, Sink, StrayTally};

/// Branch selection strategy at interactions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SplitMode {
    Analog,
    Weighted { uniform_mix: f64 },
    Split { max_splits: u32, uniform_mix: f64 },
}

impl Default for SplitMode {
    fn default() -> Self {
        SplitMode::Weighted { uniform_mix: 0.5 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StraySettings {
    /// Number of source rays.
    pub rays: u64,
    pub seed: u64,
    /// Interactions per path before the remaining weight is booked as truncated.
    pub max_events: u32,
    /// Russian-roulette threshold relative to the initial ray weight.
    pub roulette_threshold: f64,
    pub mode: SplitMode,
    /// Apply surface BSDFs to the reflected channel of optical surfaces.
    pub scatter_reflection: bool,
    /// Apply surface BSDFs to the transmitted channel of optical surfaces.
    pub scatter_transmission: bool,
    /// Rays per chunk (part of the reproducibility key: the summation order
    /// depends on it, the per-ray results do not).
    pub chunk_rays: u32,
    /// Classes reported as `EnergyReport::detected_owned`.
    pub owned_classes: ClassMask,
    /// Medium the source rays start in (default: the scene's ambient medium).
    pub start_medium: Option<MediumId>,
}

impl Default for StraySettings {
    fn default() -> Self {
        StraySettings {
            rays: 100_000,
            seed: 1,
            max_events: 512,
            roulette_threshold: 1e-3,
            mode: SplitMode::default(),
            scatter_reflection: true,
            scatter_transmission: true,
            chunk_rays: 1024,
            owned_classes: ClassMask::ALL,
            start_medium: None,
        }
    }
}

impl StraySettings {
    pub fn validate(&self) -> MkResult<()> {
        let mix_ok = |m: f64| (0.0..=1.0).contains(&m);
        let ok = self.rays > 0
            && self.chunk_rays > 0
            && self.max_events > 0
            && (0.0..1.0).contains(&self.roulette_threshold)
            && match self.mode {
                SplitMode::Analog => true,
                SplitMode::Weighted { uniform_mix } | SplitMode::Split { uniform_mix, .. } => {
                    mix_ok(uniform_mix)
                }
            };
        if ok {
            Ok(())
        } else {
            Err(MkError::InvalidModel("invalid stray-light settings".into()))
        }
    }
}

/// Output of a stray-light run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrayResult {
    pub detector: DetectorResult,
    pub energy: EnergyReport,
    pub run_info: Option<RunInfo>,
    pub warnings: Vec<String>,
}

/// Branch indices.
const SPEC_R: usize = 0;
const SCAT_R: usize = 1;
const SPEC_T: usize = 2;
const SCAT_T: usize = 3;

#[derive(Clone, Copy, Debug)]
struct State {
    ray: Ray,
    w: f64,
    medium: MediumId,
    hist: History,
    events: u32,
    splits: u32,
}

/// Everything needed to turn a chosen branch into a new ray.
struct Event<'a> {
    p: Vec3,
    /// Unit normal pointing back into the incident half-space.
    n_inc: Vec3,
    refl: Vec3,
    trans: Option<Vec3>,
    m_in: MediumId,
    m_out: MediumId,
    /// `(surface, designed if transmitted, designed if reflected, R is a stray reflection)`.
    surface: Option<(usize, bool, bool, bool)>,
    bsdf: Option<&'a Bsdf>,
    mechanical: bool,
    tir: bool,
}

/// `|β₀|` of a specular direction about the outgoing-side normal.
#[inline]
fn beta_mag(spec: Vec3, n_out: Vec3) -> f64 {
    let c = spec.dot(n_out).clamp(-1.0, 1.0);
    ((1.0 - c) * (1.0 + c)).max(0.0).sqrt()
}

/// Samples a scattered direction about the specular direction `spec` in the
/// hemisphere of `n_out`.
fn scatter_dir(bsdf: &Bsdf, spec: Vec3, n_out: Vec3, rng: &mut Philox) -> Option<Vec3> {
    let (t1, t2) = onb(n_out);
    let b = bsdf.sample([spec.dot(t1), spec.dot(t2)], rng)?;
    let bz = (1.0 - b[0] * b[0] - b[1] * b[1]).max(0.0).sqrt();
    Some((t1 * b[0] + t2 * b[1] + n_out * bz).normalized())
}

struct Tracer<'a, C: RayCaster> {
    scene: &'a Scene,
    caster: &'a C,
    settings: &'a StraySettings,
    source: &'a SourceSpec,
    w0: f64,
    wl_cdf: Vec<f64>,
}

impl<C: RayCaster> Tracer<'_, C> {
    fn pick_wavelength(&self, rng: &mut Philox) -> usize {
        if self.wl_cdf.len() == 1 {
            return 0;
        }
        let u = rng.uniform();
        self.wl_cdf
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.wl_cdf.len() - 1)
    }

    fn trace_chunk(&self, chunk: u64, ctx: &JobContext) -> MkResult<ChunkTally> {
        let n = self.settings.chunk_rays as u64;
        let (lo, hi) = (chunk * n, ((chunk + 1) * n).min(self.settings.rays));
        let mut tally = ChunkTally::new(self.scene.num_wavelengths());
        let mut stack = Vec::with_capacity(32);
        for (k, i) in (lo..hi).enumerate() {
            if k % 64 == 0 {
                ctx.check()?;
            }
            self.trace_ray(i, &mut tally, &mut stack);
        }
        Ok(tally)
    }

    /// Applies branch `br` to a copy of `s` with weight `w`. `None` ends the
    /// path (scatter sampling failure, booked by the caller).
    fn apply(
        &self,
        mut s: State,
        w: f64,
        br: usize,
        ev: &Event,
        rng: &mut Philox,
    ) -> Option<State> {
        let eps = 1e-9 * (1.0 + ev.p.norm());
        let (dir, out_sign) = match br {
            SPEC_R => (ev.refl, 1.0),
            SPEC_T => (ev.trans?, -1.0),
            SCAT_R => (scatter_dir(ev.bsdf?, ev.refl, ev.n_inc, rng)?, 1.0),
            SCAT_T => (scatter_dir(ev.bsdf?, ev.trans?, -ev.n_inc, rng)?, -1.0),
            _ => return None,
        };
        match br {
            SPEC_R | SPEC_T => {
                if ev.mechanical {
                    s.hist.mechanical_specular();
                } else if let Some((surf, dt, dr, stray)) = ev.surface {
                    let refl = br == SPEC_R;
                    s.hist.specular(
                        &self.scene.sequence,
                        surf,
                        if refl { dr } else { dt },
                        refl && stray,
                    );
                    if refl && ev.tir {
                        s.hist.flags |= flags::TIR;
                    }
                }
            }
            _ => s.hist.scattered(ev.mechanical),
        }
        s.medium = if out_sign > 0.0 { ev.m_in } else { ev.m_out };
        s.w = w;
        s.ray = Ray {
            origin: ev.p + ev.n_inc * (eps * out_sign),
            dir,
        };
        Some(s)
    }

    fn trace_ray(&self, index: u64, tally: &mut ChunkTally, stack: &mut Vec<State>) {
        let scene = self.scene;
        let set = self.settings;
        let mut rng = Philox::new(set.seed, index);
        let wl = self.pick_wavelength(&mut rng);
        let lambda = scene.wavelengths_um[wl];
        let ray = self.source.sample(&mut rng);
        let w0 = self.w0;
        tally.rays += 1;
        tally.emitted += w0;
        tally.emitted_by_wavelength[wl] += w0;
        stack.clear();
        stack.push(State {
            ray,
            w: w0,
            medium: set.start_medium.unwrap_or(scene.ambient),
            hist: History::default(),
            events: 0,
            splits: 0,
        });
        let prims = scene.primitives();
        let w_rr = set.roulette_threshold * w0;
        while let Some(mut s) = stack.pop() {
            loop {
                if s.w <= 0.0 {
                    break;
                }
                if s.events >= set.max_events {
                    tally.truncated += s.w;
                    break;
                }
                s.events += 1;
                tally.events += 1;
                let alpha = scene.media[s.medium as usize].alpha[wl];
                let Some(hit) = self.caster.closest_hit(&s.ray, f64::INFINITY) else {
                    if alpha > 0.0 {
                        tally.absorb(Sink::Bulk, s.w);
                    } else {
                        tally.escaped += s.w;
                    }
                    break;
                };
                if alpha > 0.0 {
                    let tau = (-alpha * hit.local.t).exp();
                    tally.absorb(Sink::Bulk, s.w * (1.0 - tau));
                    s.w *= tau;
                }
                let prim = &prims[hit.prim as usize];
                let p = prim.frame.to_global_point(hit.local.p);
                let n = prim.frame.to_global_dir(hit.local.n);
                let d = s.ray.dir;
                let side = Side::arrival(d, n);
                let n_inc = if side == Side::Plus { n } else { -n };
                let refl = reflect(d, n);

                // Fractions [SpecR, ScatR, SpecT, ScatT], absorption, its sink.
                let (probs, absorbed, sink, ev) = match scene.behavior(hit.prim as usize) {
                    Behavior::Detector => {
                        if scene.detector.accept.accepts(side) {
                            let class = s.hist.classify(&scene.sequence);
                            let pix = scene.detector.pixel(hit.local.p.x, hit.local.p.y);
                            tally.detect(
                                class,
                                pix,
                                s.w,
                                wl,
                                s.hist.stray_reflections,
                                s.hist.first_reflections,
                            );
                        } else {
                            tally.absorb(Sink::DetectorBack, s.w);
                        }
                        break;
                    }
                    Behavior::Opaque(o) => {
                        let tis = o
                            .finish
                            .as_ref()
                            .map_or(0.0, |b| b.tis(beta_mag(refl, n_inc)));
                        let spec = o.specular_reflectance.clamp(0.0, 1.0).min(1.0 - tis);
                        (
                            [spec, tis, 0.0, 0.0],
                            (1.0 - spec - tis).max(0.0),
                            Sink::of(o.kind),
                            Event {
                                p,
                                n_inc,
                                refl,
                                trans: None,
                                m_in: s.medium,
                                m_out: s.medium,
                                surface: None,
                                bsdf: o.finish.as_deref(),
                                mechanical: true,
                                tir: false,
                            },
                        )
                    }
                    Behavior::Interface(i) => {
                        let mirror = i.kind == InterfaceKind::Mirror;
                        if mirror && side != i.incident_side {
                            tally.absorb(Sink::MirrorBack, s.w);
                            break;
                        }
                        if let Some(c) = &i.clip
                            && !c.passes(hit.local.p.x, hit.local.p.y)
                        {
                            s.hist.flags |= flags::OUTSIDE_CLIP;
                        }
                        let (m_in, m_out) = (i.medium(side), i.medium(side.other()));
                        let n1 = scene.media[m_in as usize].n[wl];
                        let n2 = scene.media[m_out as usize].n[wl];
                        let cos_i = (-d.dot(n_inc)).clamp(0.0, 1.0);
                        let rt = i.coating.rt(lambda, n1, n2, cos_i, mirror);
                        let mut r = rt.r_unpolarized();
                        let mut t = if mirror { 0.0 } else { rt.t_unpolarized() };
                        let mut trans = None;
                        let mut tir = false;
                        if !mirror {
                            match refract(d, -n_inc, n1 / n2) {
                                Some(td) => {
                                    trans = Some(match i.kind {
                                        InterfaceKind::IdealLens { focal_length } => ideal_lens(
                                            prim.frame.to_local_dir(td),
                                            hit.local.p,
                                            focal_length,
                                        )
                                        .map_or(td, |dl| prim.frame.to_global_dir(dl)),
                                        _ => td,
                                    })
                                }
                                None => {
                                    r += t;
                                    t = 0.0;
                                    tir = true;
                                }
                            }
                        }
                        if r + t > 1.0 {
                            let k = r + t;
                            r /= k;
                            t /= k;
                        }
                        let bsdf = i.scatter.as_deref();
                        let sr = match bsdf {
                            Some(b) if set.scatter_reflection && r > 0.0 => {
                                b.tis(beta_mag(refl, n_inc))
                            }
                            _ => 0.0,
                        };
                        let st = match (bsdf, trans) {
                            (Some(b), Some(td)) if set.scatter_transmission && t > 0.0 => {
                                b.tis(beta_mag(td, -n_inc))
                            }
                            _ => 0.0,
                        };
                        let designed = side == i.incident_side;
                        (
                            [r * (1.0 - sr), r * sr, t * (1.0 - st), t * st],
                            (1.0 - r - t).max(0.0),
                            Sink::Coating,
                            Event {
                                p,
                                n_inc,
                                refl,
                                trans,
                                m_in,
                                m_out,
                                surface: Some((
                                    i.surface,
                                    designed && !mirror,
                                    designed && mirror,
                                    !mirror,
                                )),
                                bsdf,
                                mechanical: false,
                                tir,
                            },
                        )
                    }
                };

                // Choose the branch(es).
                let m = probs.iter().filter(|x| **x > 0.0).count();
                if m == 0 {
                    tally.absorb(sink, s.w);
                    break;
                }
                let sum_p: f64 = probs.iter().sum();
                let weighted = |s: State, mix: f64, rng: &mut Philox, tally: &mut ChunkTally| {
                    tally.absorb(sink, s.w * absorbed);
                    let q = |j: usize| {
                        if probs[j] > 0.0 {
                            (1.0 - mix) * probs[j] / sum_p + mix / m as f64
                        } else {
                            0.0
                        }
                    };
                    let u = rng.uniform();
                    let mut acc = 0.0;
                    let mut pick = None;
                    for (j, pj) in probs.iter().enumerate() {
                        if *pj > 0.0 {
                            acc += q(j);
                            pick = Some(j);
                            if u < acc {
                                break;
                            }
                        }
                    }
                    let j = pick.expect("one branch is positive");
                    let w = s.w * probs[j] / q(j);
                    tally.reweight(w - s.w * sum_p);
                    (j, w)
                };
                let next = match set.mode {
                    SplitMode::Analog => {
                        let u = rng.uniform();
                        if u < absorbed {
                            tally.absorb(sink, s.w);
                            break;
                        }
                        let mut acc = absorbed;
                        let mut pick = 0;
                        for (j, pj) in probs.iter().enumerate() {
                            if *pj > 0.0 {
                                acc += pj;
                                pick = j;
                                if u < acc {
                                    break;
                                }
                            }
                        }
                        (pick, s.w)
                    }
                    SplitMode::Weighted { uniform_mix } => {
                        weighted(s, uniform_mix, &mut rng, tally)
                    }
                    SplitMode::Split {
                        max_splits,
                        uniform_mix,
                    } => {
                        if s.splits < max_splits && m >= 2 {
                            tally.absorb(sink, s.w * absorbed);
                            let mut first = None;
                            let mut children = [None; 4];
                            for (j, pj) in probs.iter().enumerate() {
                                if *pj > 0.0 {
                                    let w = s.w * pj;
                                    let mut c = s;
                                    c.splits += 1;
                                    match self.apply(c, w, j, &ev, &mut rng) {
                                        Some(c) if first.is_none() => first = Some(c),
                                        Some(c) => children[j] = Some(c),
                                        None => tally.sampling_failed += w,
                                    }
                                }
                            }
                            for c in children.into_iter().rev().flatten() {
                                stack.push(c);
                            }
                            match first {
                                Some(c) => {
                                    s = c;
                                    self.roulette(&mut s, w_rr, &mut rng, tally);
                                    continue;
                                }
                                None => break,
                            }
                        }
                        weighted(s, uniform_mix, &mut rng, tally)
                    }
                };
                let (j, w) = next;
                match self.apply(s, w, j, &ev, &mut rng) {
                    Some(ns) => s = ns,
                    None => {
                        tally.sampling_failed += w;
                        break;
                    }
                }
                if !self.roulette(&mut s, w_rr, &mut rng, tally) {
                    break;
                }
            }
        }
    }

    /// Russian roulette. Returns false when the path is killed.
    #[inline]
    fn roulette(&self, s: &mut State, w_rr: f64, rng: &mut Philox, tally: &mut ChunkTally) -> bool {
        if s.w >= w_rr || s.w <= 0.0 {
            return s.w > 0.0;
        }
        if rng.uniform() * w_rr < s.w {
            tally.roulette_gain += w_rr - s.w;
            s.w = w_rr;
            true
        } else {
            tally.roulette_loss += s.w;
            s.w = 0.0;
            false
        }
    }
}

/// Ideal thin lens in its local frame: direction tangents change by
/// `−(x, y)/f` along the direction of travel.
fn ideal_lens(dl: Vec3, p: Vec3, f: f64) -> Option<Vec3> {
    if dl.z == 0.0 || f == 0.0 || !f.is_finite() {
        return None;
    }
    let az = dl.z.abs();
    let (u, v) = (dl.x / az - p.x / f, dl.y / az - p.y / f);
    Some(Vec3::new(u, v, dl.z.signum()).normalized())
}

#[derive(Serialize)]
struct RunSettings<'a> {
    settings: &'a StraySettings,
    source: &'a SourceSpec,
}

/// Runs the stray-light Monte Carlo with the scene's own BVH.
pub fn run(
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    run_with(scene, scene.bvh(), source, settings, ctx)
}

/// Runs the stray-light Monte Carlo with any [`RayCaster`] over the scene's
/// primitives (for example a hardware ray-tracing backend).
pub fn run_with<C: RayCaster>(
    scene: &Scene,
    caster: &C,
    source: &SourceSpec,
    settings: &StraySettings,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    settings.validate()?;
    source.validate()?;
    if settings
        .start_medium
        .is_some_and(|m| m as usize >= scene.media.len())
    {
        return Err(MkError::InvalidModel("start medium does not exist".into()));
    }
    let timer = RunTimer::start();
    let total: f64 = scene.wavelength_weights.iter().sum();
    let mut acc = 0.0;
    let wl_cdf: Vec<f64> = scene
        .wavelength_weights
        .iter()
        .map(|w| {
            acc += w / total;
            acc
        })
        .collect();
    let tracer = Tracer {
        scene,
        caster,
        settings,
        source,
        w0: source.power() / settings.rays as f64,
        wl_cdf,
    };
    let pixels = (scene.detector.nx * scene.detector.ny) as usize;
    let mut tally = StrayTally::new(scene.num_wavelengths(), pixels);
    let n_chunks = settings.rays.div_ceil(settings.chunk_rays as u64);
    let batch = (rayon::current_num_threads() as u64 * 4).max(1);
    let stage = "stray light: tracing rays";
    ctx.report(0.0, stage);
    let mut c = 0;
    while c < n_chunks {
        ctx.check()?;
        let hi = (c + batch).min(n_chunks);
        let chunks: Vec<ChunkTally> = (c..hi)
            .into_par_iter()
            .map(|k| tracer.trace_chunk(k, ctx))
            .collect::<MkResult<_>>()?;
        for t in &chunks {
            tally.add(t);
        }
        c = hi;
        ctx.report(c as f64 / n_chunks as f64, stage);
    }
    let (detector, energy) = tally.finish(&scene.detector, settings.owned_classes);
    let mut warnings = scene.warnings.clone();
    if energy.sampling_failed > 0.0 {
        warnings.push(format!(
            "{:.3e} W lost to BSDF rejection-sampling failures",
            energy.sampling_failed
        ));
    }
    if energy.truncated > 1e-6 * energy.emitted {
        warnings.push(format!(
            "{:.3e} W truncated at {} events per path",
            energy.truncated, settings.max_events
        ));
    }
    let run_info = scene.snapshot.as_ref().map(|snap| {
        timer.finish(
            snap,
            "stray_light",
            &RunSettings { settings, source },
            "cpu-f64",
            Precision::F64Reference,
            Some(settings.seed),
            warnings.clone(),
        )
    });
    Ok(StrayResult {
        detector,
        energy,
        run_info,
        warnings,
    })
}
