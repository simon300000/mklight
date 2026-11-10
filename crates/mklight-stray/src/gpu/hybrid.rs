//! Divisions of stray-light transport between the GPU and the CPU (the
//! route key `stray`, docs/STRAY.md "Resident GPU transport").
//!
//! The device runs the wavefront kernels on a share of the source chunks
//! (all of them in [`Design::Wavefront`]). Every design keeps the
//! reference's per-source semantics: source ray `i`
//! uses Philox stream `(seed, i)` for its whole depth-first path tree, its
//! contributions form one per-source term of every estimator, and chunks of
//! `chunk_rays` sources are folded in source order. A source is transported
//! by the device for some events and finished by the CPU tracer from the
//! device's stack, ledger and draw cursor, so the split never changes which
//! random draws decide a path. Which part of a source runs where depends
//! only on the source (its event count, stack depth and detections) and on
//! fixed options, never on timing, so a seed gives bit-identical results.
//!
//! The device writes detections, ledgers and stacks into shared memory; the
//! CPU reads them in place and forms the estimators (no copies).
//! Two plans are in flight, so the GPU transports plan `k + 1` while the CPU
//! finishes and tallies plan `k`.

use super::transport::{TransportOptions, TransportScene, history};
use super::*;
use crate::{
    geom::Ray,
    path::PathClass,
    tally::{ChunkTally, NUM_SINKS, StrayTally},
    tracer::{State, StrayResult, StraySettings, Tracer},
};
use rayon::prelude::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

/// Route key of the transport divisions (`mklight_core::split`).
pub const DESIGN_KEY: &str = "stray";
/// Route key of the device share of [`Design::CoSplit`] (a number in
/// 0…1; unset: [`default_share`]).
pub const SHARE_KEY: &str = "stray_share";
/// Route key of the device's extended-sag intersection
/// (`ops::Transport::sag_mode`): `f32-scan` (default) or `f32-grazing`,
/// which hands oblique and grazing rays to the CPU.
pub const SAG_KEY: &str = "stray_sag";

/// The selected extended-sag intersection mode (`ops::Transport::sag_mode`).
pub fn sag_mode() -> MkResult<u32> {
    match mklight_core::split::variant(SAG_KEY).as_deref() {
        None | Some("f32-scan") => Ok(0),
        Some("f32-grazing") => Ok(1),
        Some(other) => Err(MkError::InvalidModel(format!(
            "unknown stray sag mode '{other}' (f32-scan, f32-grazing)"
        ))),
    }
}

/// Detections a source keeps on the device before the CPU continues it.
const HANDOFF_DETECTIONS: usize = 8;

/// Resident scratch per plan (two plans are in flight).
const PLAN_BYTES: usize = 160 << 20;

/// A division of the stray transport between the GPU and the CPU: the
/// share of the source chunks the device transports. The device transports
/// the first `TransportOptions::head_events` events of each of its sources
/// in wavefront kernels (each bounce advances every live source by one
/// event in separate pop, closest-hit and interaction kernels over
/// device-compacted lists); the CPU continues the longer path trees from
/// the device stacks and tallies, overlapped with the next plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Design {
    /// The device transports a share (`TransportOptions::gpu_share`, else
    /// the route key `stray_share`, else [`default_share`]) of the source
    /// chunks while the CPU reference traces the other chunks at the same
    /// time.
    CoSplit,
    /// The device transports every source chunk ([`Design::CoSplit`] with a
    /// share of 1, whatever the share options say).
    Wavefront,
}

impl Design {
    /// The design selected for this run (`None`: the default design).
    pub fn selected() -> MkResult<Design> {
        Self::named(mklight_core::split::variant(DESIGN_KEY).as_deref())
    }
    pub fn named(name: Option<&str>) -> MkResult<Design> {
        Ok(match name {
            None | Some("cosplit") => Design::CoSplit,
            Some("wavefront") => Design::Wavefront,
            Some(other) => {
                return Err(MkError::InvalidModel(format!(
                    "unknown stray transport design '{other}' (cosplit, wavefront)"
                )));
            }
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Design::CoSplit => "cosplit",
            Design::Wavefront => "wavefront",
        }
    }
}

/// Device share of [`Design::CoSplit`] when neither the options nor the
/// route key `stray_share` set it: a
/// fixed rule of the scene, so that a seed gives the same partition. The
/// 12-core reference traces analytic scenes about four times faster than
/// the device, so the device takes the share it finishes in the CPU's time
/// when that share is at least `MIN_DEVICE_SOURCES` sources; scenes with
/// iteratively intersected sags (aspheres, freeforms) are 7–50 times faster
/// on the CPU, and the device gets none of them (measured on an Apple M2
/// Pro).
pub fn default_share(scene: &Scene, rays: u64) -> f64 {
    let iterative = scene
        .primitives()
        .iter()
        .any(|p| matches!(&p.shape, Shape::Sag(s) if s.conic().is_none()));
    // Lenses with iterative sags: the device is 7–50 times slower per event
    // than the CPU (with the f32 pre-scan), so it gets nothing.
    let share = if iterative { 0. } else { 0.2 };
    // A device plan has a fixed latency (its longest path trees): a device
    // share of fewer than `MIN_DEVICE_SOURCES` sources cannot pay for it.
    if share * rays as f64 >= MIN_DEVICE_SOURCES {
        share
    } else {
        0.
    }
}

/// Sources the device must receive before [`default_share`] gives it any.
const MIN_DEVICE_SOURCES: f64 = 32768.;

/// Which source chunks the device transports: units of `block` consecutive
/// chunks, spread evenly so that the device holds `round(u·share)` of the
/// first `u` units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ChunkSplit {
    block: u64,
    share: f64,
}

impl ChunkSplit {
    /// About `share` of `chunks`, in units of at most `plan_chunks` and at
    /// least two device units when the run has the chunks, so the device has
    /// a plan in flight while the CPU finishes the other.
    fn new(share: f64, chunks: u64, plan_chunks: u64) -> ChunkSplit {
        let share = share.clamp(0.0, 1.0);
        let device = ((chunks as f64 * share).round() as u64).max(1);
        let block = plan_chunks.min(device.div_ceil(2)).max(1);
        ChunkSplit { block, share }
    }
    fn device(&self, c: u64) -> bool {
        let u = (c / self.block) as f64;
        ((u + 1.) * self.share).round() > (u * self.share).round()
    }
}

/// Source ranges of the device plans: runs of consecutive device chunks cut
/// into plans of at most `per_plan` sources, at chunk boundaries where a
/// chunk fits a plan.
fn device_plans(rays: u64, chunk: u64, per_plan: u64, split: ChunkSplit) -> Vec<(u64, usize)> {
    let chunks = rays.div_ceil(chunk);
    let step = if chunk <= per_plan {
        per_plan / chunk * chunk
    } else {
        per_plan
    };
    let mut plans = Vec::new();
    let mut c = 0;
    while c < chunks {
        if !split.device(c) {
            c += 1;
            continue;
        }
        let start = c * chunk;
        while c < chunks && split.device(c) {
            c += 1;
        }
        let end = (c * chunk).min(rays);
        let mut first = start;
        while first < end {
            let count = (end - first).min(step);
            plans.push((first, count as usize));
            first += count;
        }
    }
    plans
}

/// Counters of one run.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Counters {
    /// Sources the device finished.
    pub device_sources: u64,
    /// Sources the CPU continued from a device stack, by cause.
    pub head_budget: u64,
    pub stack_full: u64,
    pub journal_full: u64,
    /// Sources the CPU continued at a long extended-sag scan (`f32-grazing`).
    pub sag_handoff: u64,
    /// Sources retraced by the CPU reference (reference arithmetic needed).
    pub retraced: u64,
    /// Sources traced by the CPU reference from the start (`CoSplit`).
    pub cpu_sources: u64,
    /// Events on the device, and on the CPU (continuations, retraces and
    /// CPU-share sources).
    pub device_events: u64,
    pub cpu_events: u64,
}

impl Counters {
    fn add(&mut self, o: &Counters) {
        self.device_sources += o.device_sources;
        self.head_budget += o.head_budget;
        self.stack_full += o.stack_full;
        self.journal_full += o.journal_full;
        self.sag_handoff += o.sag_handoff;
        self.retraced += o.retraced;
        self.cpu_sources += o.cpu_sources;
        self.device_events += o.device_events;
        self.cpu_events += o.cpu_events;
    }
}

/// Completed chunks of a plan, the chunk it leaves open, and its counters.
type PlanTallies = (Vec<(u64, ChunkTally)>, Option<(u64, ChunkTally)>, Counters);

/// Device records of one plan, read in place.
struct Records<'a> {
    first: u64,
    detections: &'a [ops::Detection],
    ledgers: &'a [ops::Ledger],
    stacks: &'a [ops::State],
    hit_capacity: usize,
    stack_capacity: usize,
}

fn cpu_state(s: &ops::State) -> MkResult<State> {
    let v = |a: [Df32; 3]| Vec3::new(a[0].to_f64(), a[1].to_f64(), a[2].to_f64());
    Ok(State {
        ray: Ray {
            origin: v(s.ray.origin),
            dir: v(s.ray.direction),
        },
        w: s.weight.to_f64(),
        medium: s.medium as crate::scene::MediumId,
        hist: history(&s.history)?,
        events: s.events,
        splits: s.splits,
    })
}

/// A device record that no kernel of this build writes.
fn malformed(what: &str) -> MkError {
    MkError::Backend(format!("stray transport: malformed device {what}"))
}

struct Finisher<'a> {
    scene: &'a Scene,
    settings: &'a StraySettings,
    tracer: Tracer<'a, crate::geom::Bvh>,
}

impl Finisher<'_> {
    /// Books source `k` of `r` into `tally`: its device ledger and
    /// detections, then the CPU continuation of a stopped source, or the
    /// whole source on the CPU when the device needed reference arithmetic.
    /// Device records outside what the kernels write are a backend error.
    fn source(
        &self,
        r: &Records,
        k: usize,
        tally: &mut ChunkTally,
        stack: &mut Vec<State>,
        n: &mut Counters,
    ) -> MkResult<()> {
        let index = r.first + k as u64;
        let l = r.ledgers.get(k).ok_or_else(|| malformed("ledger"))?;
        tally.begin_ray();
        let events = tally.events;
        if l.status == 4 {
            self.tracer.trace_ray(index, tally, stack);
            n.retraced += 1;
            n.cpu_events += tally.events - events;
            tally.end_ray();
            return Ok(());
        }
        let w0 = self.tracer.source_weight();
        let wl = l.wavelength as usize;
        // The source's used part of a journal of `capacity` records.
        let used = |n: u32, capacity: usize| {
            (n as usize <= capacity).then(|| k * capacity..k * capacity + n as usize)
        };
        let detections = used(l.hits, r.hit_capacity)
            .and_then(|range| r.detections.get(range))
            .ok_or_else(|| malformed("detection count"))?;
        let stacked = used(l.top, r.stack_capacity)
            .and_then(|range| r.stacks.get(range))
            .ok_or_else(|| malformed("stack depth"))?;
        if wl >= tally.emitted_by_wavelength.len() {
            return Err(malformed("wavelength index"));
        }
        tally.rays += 1;
        tally.emitted += w0;
        tally.emitted_by_wavelength[wl] += w0;
        let s = l.sums.map(|v| v.to_f64());
        for (a, v) in tally.absorbed.iter_mut().zip(&s[..NUM_SINKS]) {
            *a += v;
        }
        tally.escaped += s[12];
        tally.escaped_from_solid += s[13];
        tally.truncated += s[14];
        tally.sampling_failed += s[15];
        tally.roulette(s[16]);
        tally.roulette(-s[17]);
        tally.reweight(s[18]);
        tally.reweight(-s[19]);
        tally.events += l.events as u64;
        n.device_events += l.events as u64;
        let detector = &self.scene.detector;
        for d in detections {
            let hist = history(&d.history)?;
            let class = *PathClass::ALL
                .get(d.class as usize)
                .ok_or_else(|| malformed("path class"))?;
            let (x, y) = (d.x.to_f64(), d.y.to_f64());
            tally.detect(
                class,
                self.settings.owns(class, &hist.signature),
                detector.pixel(x, y),
                d.weight.to_f64(),
                wl,
                &hist,
                x,
                y,
            );
        }
        match l.status {
            0 => n.device_sources += 1,
            status => {
                match status {
                    2 => n.stack_full += 1,
                    3 => n.journal_full += 1,
                    6 => n.sag_handoff += 1,
                    _ => n.head_budget += 1,
                }
                stack.clear();
                for s in stacked {
                    stack.push(cpu_state(s)?);
                }
                let before = tally.events;
                self.tracer.resume(index, l.draws, wl, tally, stack);
                n.cpu_events += tally.events - before;
            }
        }
        tally.end_ray();
        Ok(())
    }

    /// Finishes a plan: chunk segments in parallel, each in source order.
    /// `carry` is a chunk begun by the previous plan. Returns the completed
    /// chunks and the chunk left open at the end of the plan.
    fn plan(
        &self,
        r: &Records,
        count: usize,
        carry: Option<(u64, ChunkTally)>,
        ctx: &JobContext,
    ) -> MkResult<PlanTallies> {
        let chunk = self.settings.chunk_rays as u64;
        let mut segments = Vec::new();
        let mut lo = r.first;
        let end = r.first + count as u64;
        while lo < end {
            let c = lo / chunk;
            let hi = ((c + 1) * chunk).min(end);
            segments.push((c, lo, hi));
            lo = hi;
        }
        let mut carry = carry;
        let starts: Vec<Option<ChunkTally>> = segments
            .iter()
            .map(|&(c, ..)| match carry.take() {
                Some((cc, t)) if cc == c => Some(t),
                other => {
                    carry = other;
                    None
                }
            })
            .collect();
        debug_assert!(
            carry.is_none(),
            "a carried chunk continues in the next plan"
        );
        let done = segments
            .into_par_iter()
            .zip(starts)
            .map(|((c, lo, hi), start)| {
                ctx.check()?;
                let mut tally = start.unwrap_or_else(|| self.tracer.new_chunk());
                let mut stack = Vec::with_capacity(32);
                let mut n = Counters::default();
                for i in lo..hi {
                    self.source(r, (i - r.first) as usize, &mut tally, &mut stack, &mut n)?;
                }
                Ok((c, hi, tally, n))
            })
            .collect::<MkResult<Vec<_>>>()?;
        let mut out = Vec::new();
        let mut open = None;
        let mut counters = Counters::default();
        for (c, hi, tally, n) in done {
            counters.add(&n);
            if hi == ((c + 1) * chunk).min(self.settings.rays) {
                out.push((c, tally));
            } else {
                open = Some((c, tally));
            }
        }
        Ok((out, open, counters))
    }

    /// Chunk `c` traced by the CPU reference.
    fn cpu_chunk(&self, c: u64, ctx: &JobContext) -> MkResult<(ChunkTally, Counters)> {
        let tally = self.tracer.trace_chunk(c, ctx)?;
        let n = Counters {
            cpu_sources: tally.rays,
            cpu_events: tally.events,
            ..Default::default()
        };
        Ok((tally, n))
    }
}

/// Folds completed chunks into the run tally in chunk order.
struct Folder {
    tally: StrayTally,
    next: u64,
    waiting: BTreeMap<u64, ChunkTally>,
}

impl Folder {
    fn put(&mut self, c: u64, t: ChunkTally) {
        self.waiting.insert(c, t);
        while let Some(t) = self.waiting.remove(&self.next) {
            self.tally.add(&t);
            self.next += 1;
        }
    }
}

/// Runs `design`; `side` runs on the CPU once, while the device transports
/// the first plans. A run whose device share holds no source chunk is the
/// CPU reference run ([`crate::tracer::run`]) and is reported as one.
/// `fingerprint` is the scene's (`Scene::fingerprint`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    fingerprint: &str,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    design: Design,
    side: &mut (dyn FnMut() -> MkResult<()> + Send),
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    crate::tracer::validate_run(scene, source, settings)?;
    if options.source_tile == 0
        || options.source_tile > 65536
        || options.detections_per_source == 0
        || options.detections_per_source > 1024
        || options.stack_capacity == 0
        || options.stack_capacity > 8192
        || options.gpu_share.is_some_and(|s| !(0.0..=1.0).contains(&s))
    {
        return Err(MkError::InvalidModel("stray transport settings".into()));
    }
    let timer = mklight_core::job::RunTimer::start();
    let started = Instant::now();
    let hit_capacity = options.detections_per_source.min(HANDOFF_DETECTIONS);
    let stack_capacity = options.stack_capacity.max(4);
    let sag = sag_mode()?;
    let share = match design {
        // Options first, then the share route key, then the scene rule.
        Design::CoSplit => match options.gpu_share {
            Some(share) => share,
            None => match mklight_core::split::variant(SHARE_KEY) {
                Some(v) => v
                    .parse::<f64>()
                    .ok()
                    .filter(|s| (0.0..=1.0).contains(s))
                    .ok_or_else(|| MkError::InvalidModel(format!("stray share '{v}'")))?,
                None => default_share(scene, settings.rays),
            },
        },
        Design::Wavefront => 1.0,
    };
    let per_source = std::mem::size_of::<ops::Ray>()
        + std::mem::size_of::<ops::Ledger>()
        + 8
        + stack_capacity * std::mem::size_of::<ops::State>()
        + hit_capacity * std::mem::size_of::<ops::Detection>();
    let per_plan = (PLAN_BYTES / per_source)
        .min(options.source_tile)
        .min(mklight_gpu::consts::MKL_STRAY_MAX_TILE as usize)
        .max(1) as u64;
    let chunk = settings.chunk_rays as u64;
    let n_chunks = settings.rays.div_ceil(chunk);
    let split = ChunkSplit::new(share, n_chunks, (per_plan / chunk).max(1));
    let plans = device_plans(settings.rays, chunk, per_plan, split);
    if plans.is_empty() {
        // The device gets no source: the reference run, with `side` beside
        // it, labelled as the CPU f64 reference it is.
        let (sided, result) = rayon::join(&mut *side, || {
            crate::tracer::run_fingerprinted(scene, source, settings, fingerprint, ctx)
        });
        sided?;
        let mut result = result?;
        result.run_info = Some(timer.finish(
            &snapshot,
            "stray_light",
            &serde_json::json!({"settings": settings, "source": source, "resident_options": options,
                "design": design.name(), "gpu_share": share,
                "requested_backend": b.info().backend_id(), "fallback_count": 0,
                "scene_fingerprint": fingerprint}),
            "cpu-f64",
            mklight_core::job::Precision::F64Reference,
            Some(settings.seed),
            result.warnings.clone(),
        ));
        ctx.check()?;
        ctx.report(1., "stray complete");
        return Ok(result);
    }
    // Device weights are double-float with f32 exponents; the estimators
    // are formed on the CPU in f64.
    let w0 = source.power() / settings.rays as f64;
    if !(1e-30..=1e30).contains(&w0) {
        return Err(unsupported("source weight is outside the device range"));
    }
    let cpu_chunks: Vec<u64> = (0..n_chunks).filter(|&c| !split.device(c)).collect();
    let prepared = TransportScene::new(b, scene, &ctx.subrange(0., 0.02))?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let finisher = Finisher {
        scene,
        settings,
        tracer: Tracer::new(scene, scene.bvh(), source, settings),
    };
    let submit = |first: u64, count: usize| -> MkResult<_> {
        let mut plan = ResidentPlan::new(snapshot.clone());
        plan.seed = Some(settings.seed);
        plan.settings = serde_json::json!({"stray_version": mklight_gpu::consts::MKL_STRAY_VERSION,
            "scene": fingerprint, "first": first, "design": design.name()});
        let opts = TransportOptions {
            stack_capacity,
            detections_per_source: hit_capacity,
            ..options
        };
        let (hits, ledger, stack) =
            prepared.append(&mut plan, scene, source, settings, first, count, opts, sag)?;
        plan.outputs = vec![hits, ledger, stack];
        // Progress is reported per finished chunk, not by the device.
        b.submit_plan(
            plan,
            &JobContext::with(ctx.cancel.clone(), mklight_core::job::Progress::default()),
        )
    };
    let folder = Mutex::new(Folder {
        tally: StrayTally::new(
            scene.num_wavelengths(),
            (scene.detector.nx * scene.detector.ny) as usize,
        ),
        next: 0,
        waiting: BTreeMap::new(),
    });
    let done_chunks = std::sync::atomic::AtomicU64::new(0);
    let report = |k: u64| {
        let d = done_chunks.fetch_add(k, std::sync::atomic::Ordering::Relaxed) + k;
        ctx.report(
            0.02 + 0.97 * d as f64 / n_chunks.max(1) as f64,
            "stray light: transport",
        );
    };
    let mut counters = Counters::default();
    let mut plan_stats = Vec::new();
    let (mut wait_seconds, mut finish_seconds, mut side_seconds) = (0., 0., 0.);
    let cpu_counters = Mutex::new(Counters::default());
    let abort = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| -> MkResult<()> {
        // The CPU share (CoSplit) runs in the rayon pool beside the device.
        let cpu = (!cpu_chunks.is_empty()).then(|| {
            scope.spawn(|| -> MkResult<()> {
                cpu_chunks.par_iter().try_for_each(|&c| {
                    if abort.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err(MkError::Cancelled);
                    }
                    let (t, n) = finisher.cpu_chunk(c, ctx)?;
                    cpu_counters.lock().unwrap().add(&n);
                    folder.lock().unwrap().put(c, t);
                    report(1);
                    Ok(())
                })
            })
        });
        let result = (|| -> MkResult<()> {
            let mut queue = VecDeque::new();
            let mut next = 0;
            while next < plans.len() && queue.len() < 2 {
                let (first, count) = plans[next];
                queue.push_back((first, count, submit(first, count)?));
                next += 1;
            }
            let t = Instant::now();
            side()?;
            side_seconds = t.elapsed().as_secs_f64();
            let mut carry = None;
            while let Some((first, count, pending)) = queue.pop_front() {
                ctx.check()?;
                let t = Instant::now();
                let result = pending.wait()?;
                wait_seconds += t.elapsed().as_secs_f64();
                if next < plans.len() {
                    let (first, count) = plans[next];
                    queue.push_back((first, count, submit(first, count)?));
                    next += 1;
                }
                let t = Instant::now();
                let views: [ReadView; 3] = [0, 1, 2]
                    .map(|i| b.read_view(result.outputs[i].view()))
                    .into_iter()
                    .collect::<MkResult<Vec<_>>>()?
                    .try_into()
                    .ok()
                    .unwrap();
                let records = Records {
                    first,
                    detections: ops::read_records(&views[0])?,
                    ledgers: ops::read_records(&views[1])?,
                    stacks: ops::read_records(&views[2])?,
                    hit_capacity,
                    stack_capacity,
                };
                // A chunk left open by the previous plan continues here
                // (plans of consecutive sources; see `device_plans`).
                let (complete, open, n) = finisher.plan(&records, count, carry.take(), ctx)?;
                drop(views);
                counters.add(&n);
                let k = complete.len() as u64;
                {
                    let mut f = folder.lock().unwrap();
                    for (c, t) in complete {
                        f.put(c, t);
                    }
                }
                report(k);
                carry = open;
                plan_stats.push(result.stats);
                finish_seconds += t.elapsed().as_secs_f64();
            }
            if let Some((c, t)) = carry {
                // Only the last chunk of the run can stay open.
                folder.lock().unwrap().put(c, t);
                report(1);
            }
            Ok(())
        })();
        if result.is_err() {
            abort.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let cpu_result = cpu.map(|h| h.join().expect("CPU share panicked"));
        result?;
        cpu_result.transpose()?;
        Ok(())
    })?;
    counters.add(&cpu_counters.into_inner().unwrap());
    let folder = folder.into_inner().unwrap();
    if folder.next != n_chunks || !folder.waiting.is_empty() {
        return Err(MkError::Backend("stray transport: chunks missing".into()));
    }
    let pack_started = Instant::now();
    let mut result = crate::tracer::finish(scene, settings, &folder.tally);
    let packing_seconds = pack_started.elapsed().as_secs_f64();
    let info = timer.finish(
        &snapshot,
        "stray_light",
        &serde_json::json!({"settings": settings, "source": source, "resident_options": options,
            "design": design.name(), "gpu_share": share, "sag_mode": sag,
            "hit_capacity": hit_capacity, "stack_capacity": stack_capacity, "plan_sources": per_plan,
            "stray_version": mklight_gpu::consts::MKL_STRAY_VERSION, "plans": plan_stats,
            "device": b.info(), "precision": b.precision(), "fallback_count": 0,
            "counters": counters, "scene_fingerprint": fingerprint,
            "timings": {"preparation_seconds": preparation_seconds, "device_wait_seconds": wait_seconds,
                "finish_seconds": finish_seconds, "side_seconds": side_seconds,
                "packing_seconds": packing_seconds}}),
        &b.info().backend_id(),
        mklight_core::job::Precision::F32Gpu,
        Some(settings.seed),
        result.warnings.clone(),
    );
    result.run_info = Some(info);
    ctx.check()?;
    ctx.report(1., "stray complete");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_chunks_follow_the_share_and_plans_cover_them() {
        for (chunks, plan_chunks) in [(20u64, 17u64), (391, 17), (1954, 17), (3, 17), (100, 1)] {
            for share in [0.0, 0.25, 1. / 3., 0.5, 0.8, 1.0] {
                let split = ChunkSplit::new(share, chunks, plan_chunks);
                let device = (0..chunks).filter(|&c| split.device(c)).count() as f64;
                // Within one unit of the share (exact for 0 and 1).
                assert!(
                    (device - share * chunks as f64).abs() <= split.block as f64,
                    "{chunks} {share}: {device} ({split:?})"
                );
                if share == 1.0 {
                    assert_eq!(device as u64, chunks);
                }
                if share == 0.0 {
                    assert_eq!(device, 0.);
                }
                let rays = chunks * 1024 - 7;
                let plans = device_plans(rays, 1024, plan_chunks * 1024, split);
                let covered: u64 = plans.iter().map(|p| p.1 as u64).sum();
                let wanted: u64 = (0..chunks)
                    .filter(|&c| split.device(c))
                    .map(|c| ((c + 1) * 1024).min(rays) - c * 1024)
                    .sum();
                assert_eq!(covered, wanted);
                for &(first, count) in &plans {
                    assert!(count as u64 <= plan_chunks * 1024);
                    assert_eq!(first % 1024, 0);
                }
            }
        }
        // Chunks larger than a plan are cut.
        let all = ChunkSplit::new(1.0, 3, 1);
        let plans = device_plans(10_000, 4096, 1000, all);
        assert_eq!(plans.iter().map(|p| p.1).sum::<usize>(), 10_000);
        assert!(plans.iter().all(|p| p.1 <= 1000));
    }

    #[test]
    fn design_names_round_trip() {
        for d in [Design::CoSplit, Design::Wavefront] {
            assert_eq!(Design::named(Some(d.name())).unwrap(), d);
        }
        assert_eq!(Design::named(None).unwrap(), Design::CoSplit);
        assert!(Design::named(Some("nope")).is_err());
    }
}
