//! Divisions of stray-light transport between the GPU and the CPU (the
//! work-split study key `stray`, docs/STRAY.md "Resident GPU transport").
//!
//! Every design keeps the reference's per-source semantics: source ray `i`
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
//! CPU reads them in place (no estimator chain on the device, no copies).
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

/// Study key of the transport divisions (`mklight_core::split`).
pub const STUDY_KEY: &str = "stray";
/// Study key of the device share of [`Design::CoSplit`] (a number in
/// 0…1; unset: [`default_share`]).
pub const SHARE_KEY: &str = "stray_share";
/// Study key of the device's extended-sag intersection
/// (`ops::Transport::sag_mode`): `f32-scan` (default), `scan`,
/// `cpu-grazing` or `f32-grazing` (both).
pub const SAG_KEY: &str = "stray_sag";

/// Study key of the threads of the CPU share of [`Design::CoSplit`]: a
/// number of threads for a pool of its own (unset: the global pool, which
/// the device plans' CPU completion shares).
pub const THREADS_KEY: &str = "stray_threads";

/// A pool of the selected size for a CPU share, if one is selected.
pub(crate) fn share_pool(key: &str) -> MkResult<Option<rayon::ThreadPool>> {
    match mklight_core::split::variant(key) {
        None => Ok(None),
        Some(v) => {
            let n = v
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| MkError::InvalidModel(format!("{key} '{v}'")))?;
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map(Some)
                .map_err(|e| MkError::Backend(format!("{key} pool: {e}")))
        }
    }
}

/// The selected extended-sag intersection mode.
pub fn sag_mode() -> MkResult<u32> {
    Ok(match mklight_core::split::variant(SAG_KEY).as_deref() {
        Some("scan") => 0,
        None | Some("f32-scan") => 1,
        Some("cpu-grazing") => 2,
        Some("f32-grazing") => 3,
        Some(other) => {
            return Err(MkError::InvalidModel(format!(
                "unknown stray sag mode '{other}' (scan, f32-scan, cpu-grazing, f32-grazing)"
            )));
        }
    })
}

/// Detections a source keeps on the device before the CPU continues it.
const HANDOFF_DETECTIONS: usize = 8;

/// Resident scratch per plan (two plans are in flight).
const PLAN_BYTES: usize = 160 << 20;

/// A division of the stray transport between the GPU and the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Design {
    /// The device transports and tallies everything (detector estimators on
    /// the device, integer keys on the host); a capacity or precision limit
    /// sends the whole run to the CPU reference.
    GpuTally,
    /// The device transports every source to completion; the CPU tallies the
    /// detections in place, continues sources whose stack or detection
    /// journal is full, and retraces sources that need reference arithmetic.
    CpuTally,
    /// The device transports the first `TransportOptions::head_events`
    /// events of every source, the CPU continues the longer path trees from
    /// the device stacks and tallies, overlapped with the next plan.
    GpuHead,
    /// `Wavefront` on a fixed share (`TransportOptions::gpu_share`) of the
    /// source chunks, while the CPU reference traces the other chunks at the
    /// same time.
    CoSplit,
    /// `GpuHead` with wavefront device transport: each bounce advances every
    /// live source by one event in separate pop, closest-hit and interaction
    /// kernels, with device-compacted lists (identical device records).
    Wavefront,
}

impl Design {
    /// The design selected for this run (`None`: the default design).
    pub fn selected() -> MkResult<Design> {
        Self::named(mklight_core::split::variant(STUDY_KEY).as_deref())
    }
    pub fn named(name: Option<&str>) -> MkResult<Design> {
        Ok(match name {
            None => Design::CoSplit,
            Some("gpu-tally") => Design::GpuTally,
            Some("cpu-tally") => Design::CpuTally,
            Some("gpu-head") => Design::GpuHead,
            Some("cosplit") => Design::CoSplit,
            Some("wavefront") => Design::Wavefront,
            Some(other) => {
                return Err(MkError::InvalidModel(format!(
                    "unknown stray transport design '{other}' (gpu-tally, cpu-tally, gpu-head, cosplit, wavefront)"
                )));
            }
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Design::GpuTally => "gpu-tally",
            Design::CpuTally => "cpu-tally",
            Design::GpuHead => "gpu-head",
            Design::CoSplit => "cosplit",
            Design::Wavefront => "wavefront",
        }
    }
    /// Whether the device runs the wavefront kernels.
    fn wavefront(self) -> bool {
        matches!(self, Design::Wavefront | Design::CoSplit)
    }
}

/// Device share of [`Design::CoSplit`] when the options leave it open: a
/// fixed rule of the scene, so that a seed gives the same partition. The
/// 12-core reference traces analytic scenes about four times faster than
/// the device, so the device takes the share it finishes in the CPU's time
/// when that share is at least `MIN_DEVICE_SOURCES` sources; scenes with
/// iteratively intersected sags (aspheres, freeforms) are 7–50 times faster
/// on the CPU, and the device gets none of them (provisional M2 Pro
/// measurements).
pub fn default_share(scene: &Scene, rays: u64) -> f64 {
    let iterative = scene.primitives().iter().any(|p| match &p.shape {
        // The device's closed-form kinds (`prepare_geometry`).
        Shape::Sag(s) => !match s.shape() {
            SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::Conic { .. } => {
                true
            }
            SurfaceShape::EvenAsphere { coefficients, .. } => coefficients.iter().all(|v| *v == 0.),
            _ => false,
        },
        Shape::Cylinder(_) => false,
    });
    // Lenses with iterative sags: the device is 7–50 times slower per event
    // than the CPU (round 2, with the f32 pre-scan), so it gets nothing.
    let share = if iterative { 0. } else { 0.2 };
    // A device plan has a fixed latency (its longest path trees): below
    // about two plans of work the device cannot pay for it.
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
    /// Sources the CPU continued at a long extended-sag scan (`cpu-grazing`).
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

fn cpu_state(s: &ops::State) -> State {
    let v = |a: [Df32; 3]| Vec3::new(a[0].to_f64(), a[1].to_f64(), a[2].to_f64());
    State {
        ray: Ray {
            origin: v(s.ray.origin),
            dir: v(s.ray.direction),
        },
        w: s.weight.to_f64(),
        medium: s.medium as crate::scene::MediumId,
        hist: history(&s.history),
        events: s.events,
        splits: s.splits,
    }
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
    fn source(
        &self,
        r: &Records,
        k: usize,
        tally: &mut ChunkTally,
        stack: &mut Vec<State>,
        n: &mut Counters,
    ) {
        let index = r.first + k as u64;
        let l = &r.ledgers[k];
        tally.begin_ray();
        let events = tally.events;
        if l.status == 4 {
            self.tracer.trace_ray(index, tally, stack);
            n.retraced += 1;
            n.cpu_events += tally.events - events;
            tally.end_ray();
            return;
        }
        let w0 = self.tracer.source_weight();
        let wl = l.wavelength as usize;
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
        for d in &r.detections[k * r.hit_capacity..][..l.hits as usize] {
            let hist = history(&d.history);
            let class = PathClass::ALL[d.class as usize];
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
                stack.extend(
                    r.stacks[k * r.stack_capacity..][..l.top as usize]
                        .iter()
                        .map(cpu_state),
                );
                let before = tally.events;
                self.tracer.resume(index, l.draws, wl, tally, stack);
                n.cpu_events += tally.events - before;
            }
        }
        tally.end_ray();
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
                let mut tally = start.unwrap_or_else(|| self.new_chunk());
                let mut stack = Vec::with_capacity(32);
                let mut n = Counters::default();
                for i in lo..hi {
                    self.source(r, (i - r.first) as usize, &mut tally, &mut stack, &mut n);
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

    fn new_chunk(&self) -> ChunkTally {
        let mut t = ChunkTally::new(self.scene.num_wavelengths(), self.settings.owned_classes);
        t.track_signatures = self.settings.signatures;
        t
    }

    /// Chunk `c` traced by the CPU reference.
    fn cpu_chunk(&self, c: u64, ctx: &JobContext) -> MkResult<(ChunkTally, Counters)> {
        let chunk = self.settings.chunk_rays as u64;
        let (lo, hi) = (c * chunk, ((c + 1) * chunk).min(self.settings.rays));
        let mut tally = self.new_chunk();
        let mut stack = Vec::with_capacity(32);
        for (k, i) in (lo..hi).enumerate() {
            if k % 64 == 0 {
                ctx.check()?;
            }
            tally.begin_ray();
            self.tracer.trace_ray(i, &mut tally, &mut stack);
            tally.end_ray();
        }
        let n = Counters {
            cpu_sources: hi - lo,
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

/// Runs `design` (not [`Design::GpuTally`]). `side` runs on the CPU once,
/// while the device transports the first plans.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    design: Design,
    side: &mut dyn FnMut() -> MkResult<()>,
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
        || design == Design::GpuTally
    {
        return Err(MkError::InvalidModel("stray transport settings".into()));
    }
    // Device weights are double-float with f32 exponents; the estimators
    // are formed on the CPU in f64.
    let w0 = source.power() / settings.rays as f64;
    if !(1e-30..=1e30).contains(&w0) {
        return Err(unsupported("source weight is outside the device range"));
    }
    let timer = mklight_core::job::RunTimer::start();
    let started = Instant::now();
    let hit_capacity = options.detections_per_source.min(HANDOFF_DETECTIONS);
    let stack_capacity = options.stack_capacity.max(4);
    let sag = sag_mode()?;
    let head_events = match design {
        Design::CpuTally => 0,
        _ => options.head_events,
    };
    let share = match design {
        // Options first, then the share study key, then the scene rule.
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
        _ => 1.0,
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
    let cpu_chunks: Vec<u64> = (0..n_chunks).filter(|&c| !split.device(c)).collect();
    let prepared = if plans.is_empty() {
        None
    } else {
        Some(TransportScene::new(b, scene, &ctx.subrange(0., 0.02))?)
    };
    let preparation_seconds = started.elapsed().as_secs_f64();
    let finisher = Finisher {
        scene,
        settings,
        tracer: Tracer::new(scene, scene.bvh(), source, settings),
    };
    let fingerprint = scene.fingerprint();
    let submit = |first: u64, count: usize| -> MkResult<_> {
        let prepared = prepared.as_ref().unwrap();
        let mut plan = ResidentPlan::new(snapshot.clone());
        plan.seed = Some(settings.seed);
        plan.settings = serde_json::json!({"stray_version": mklight_gpu::consts::MKL_STRAY_VERSION,
            "scene": fingerprint, "first": first, "design": design.name()});
        let stack = plan.scratch(ops::record_desc::<ops::State>(count * stack_capacity));
        let opts = TransportOptions {
            stack_capacity,
            detections_per_source: hit_capacity,
            ..options
        };
        let (hits, ledger) = prepared.append_handoff(
            &mut plan,
            scene,
            source,
            settings,
            first,
            count,
            opts,
            stack,
            head_events,
            design.wavefront(),
            sag,
        )?;
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
    let pool = share_pool(THREADS_KEY)?;
    std::thread::scope(|scope| -> MkResult<()> {
        // The CPU share (CoSplit) runs in the rayon pool beside the device,
        // or in a pool of its own (`stray_threads`).
        let cpu = (!cpu_chunks.is_empty()).then(|| {
            scope.spawn(|| -> MkResult<()> {
                let share = || {
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
                };
                match &pool {
                    Some(p) => p.install(share),
                    None => share(),
                }
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
        return Err(MkError::Backend("stray hybrid: chunks missing".into()));
    }
    let pack_started = Instant::now();
    let (detector, energy, mut signatures) =
        folder.tally.finish(&scene.detector, settings.owned_classes);
    for s in &mut signatures {
        s.label = s.signature.describe(|p| scene.label(p).to_string());
    }
    let mut warnings = scene.warnings.clone();
    if energy.sampling_failed > 0. {
        warnings.push(format!(
            "{:.3e} W lost to BSDF rejection-sampling failures",
            energy.sampling_failed
        ));
    }
    if energy.escaped_from_solid > 1e-12 * energy.emitted {
        warnings.push(format!(
            "{:.3e} W left the scene from inside a solid",
            energy.escaped_from_solid
        ));
    }
    if energy.truncated > 1e-6 * energy.emitted {
        warnings.push(format!(
            "{:.3e} W truncated at {} events per path",
            energy.truncated, settings.max_events
        ));
    }
    let packing_seconds = pack_started.elapsed().as_secs_f64();
    let info = timer.finish(
        &snapshot,
        "stray_light",
        &serde_json::json!({"settings": settings, "source": source, "resident_options": options,
            "design": design.name(), "head_events": head_events, "gpu_share": share, "sag_mode": sag,
            "kernel": mklight_core::split::variant("stray_kernel"),
            "cpu_share_threads": pool.as_ref().map(|p| p.current_num_threads()),
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
        warnings.clone(),
    );
    ctx.check()?;
    ctx.report(1., "stray complete");
    Ok(StrayResult {
        detector,
        energy,
        signatures,
        warnings,
        run_info: Some(info),
    })
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
        for d in [
            Design::GpuTally,
            Design::CpuTally,
            Design::GpuHead,
            Design::CoSplit,
            Design::Wavefront,
        ] {
            assert_eq!(Design::named(Some(d.name())).unwrap(), d);
        }
        assert_eq!(Design::named(None).unwrap(), Design::CoSplit);
        assert!(Design::named(Some("nope")).is_err());
    }
}
