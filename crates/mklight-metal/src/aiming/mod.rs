//! Encoders for P1. Generation copies only a four-byte failure count to the
//! host; optical outputs, chief seeds and seed tables stay device-resident.
use crate::{
    SharedBuffer,
    backend::Uploaded,
    context::{Encoder, PipelineKey},
    resident::{Execution, Lease},
};
use bytemuck::{Pod, Zeroable};
use mklight_core::{MkError, MkResult};
use mklight_gpu::{
    aiming::*,
    consts::*,
    layout::TraceParams,
    pending::JobControl,
    resident::immutable::{BuildCost, CacheCategory, OperationCacheKey},
    resident::*,
};
use std::{ops::Deref, sync::Arc, time::Instant};
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AimParams {
    count: u32,
    offset: u32,
    stride: u32,
    fields: u32,
    systems: u32,
    cap: u32,
    tolerance: f32,
    table: u32,
    pattern: u32,
    side: u32,
    extent: [f32; 2],
    /// Iteration units per lane of this command.
    budget: u32,
    table_side: u32,
    flags: u32,
    reserved: u32,
}
const AIM_CULL: u32 = 1;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResidentTraceParams {
    trace: TraceParams,
    offset: u32,
    systems: u32,
    reserved: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProjectParams {
    radius: [f32; 2],
    index: [f32; 2],
    wavelength: [f32; 2],
    sheet: i32,
    telecentric: u32,
    count: u32,
    offset: u32,
    stride: u32,
    reserved: u32,
}
fn view(id: BufferId, element: ElementType) -> PlanView {
    PlanView {
        buffer: id,
        offset: 0,
        desc: BufferDesc::new(element, [0]),
        stride: 1,
    }
}
fn bind(e: &Encoder<'_>, index: usize, x: &Execution<'_>, v: &PlanView) {
    let (b, o) = x.buffer(v);
    e.set_buffer(index, b, o);
}
enum Systems {
    Uploaded(Arc<Uploaded>),
    Resident {
        headers: BufferId,
        surfaces: BufferId,
        coefficients: BufferId,
        count: usize,
    },
}
impl Systems {
    fn new(x: &Execution<'_>, s: &SystemSet) -> MkResult<Self> {
        Ok(match s {
            SystemSet::Uploaded(h) => Self::Uploaded(x.system(*h)?),
            SystemSet::Resident {
                headers,
                surfaces,
                coefficients,
                count,
                ..
            } => Self::Resident {
                headers: *headers,
                surfaces: *surfaces,
                coefficients: *coefficients,
                count: *count,
            },
        })
    }
    fn count(&self) -> u32 {
        match self {
            Self::Uploaded(u) => u.set.headers.len() as u32,
            Self::Resident { count, .. } => *count as u32,
        }
    }
    fn bind(&self, e: &Encoder<'_>, x: &Execution<'_>) {
        match self {
            Self::Uploaded(u) => {
                e.set_buffer(0, &u.headers, 0);
                e.set_buffer(1, &u.surfaces, 0);
                e.set_buffer(2, &u.coefficients, 0);
            }
            Self::Resident {
                headers,
                surfaces,
                coefficients,
                ..
            } => {
                bind(e, 0, x, &view(*headers, ElementType::U32));
                bind(e, 1, x, &view(*surfaces, ElementType::U32));
                bind(e, 2, x, &view(*coefficients, ElementType::DoubleFloat));
            }
        }
    }
    /// The kernel variant: the overlay code only for systems that have sag
    /// overlays, or device-built systems whose surfaces the host cannot see.
    fn kernel(&self, plain: &'static str, overlays: &'static str) -> PipelineKey {
        let any = match self {
            Self::Uploaded(u) => u
                .set
                .surfaces
                .iter()
                .any(|s| s.flags & MKL_SURF_HAS_SAG_OVERLAY != 0),
            Self::Resident { .. } => true,
        };
        PipelineKey::plain(if any { overlays } else { plain })
    }
}
fn take_rays(x: &mut Execution<'_>, r: &ResidentRays) -> [Lease; 4] {
    [
        x.take_output(r.floats),
        x.take_output(r.indices),
        x.take_output(r.logical_ids),
        x.take_output(r.report),
    ]
}
fn put_rays(x: &mut Execution<'_>, r: &ResidentRays, b: [Lease; 4]) {
    for (id, b) in [r.floats, r.indices, r.logical_ids, r.report]
        .into_iter()
        .zip(b)
    {
        x.put_output(id, b);
    }
}
/// Per-field-row data (chief seeds, seed tables): cached across plans when
/// the request names its immutable fields, otherwise plan-local.
enum Rows {
    Cached(Arc<SharedBuffer<u32>>),
    Local(Lease),
}
impl Deref for Rows {
    type Target = SharedBuffer<u32>;
    fn deref(&self) -> &SharedBuffer<u32> {
        match self {
            Self::Cached(b) => b,
            Self::Local(b) => b,
        }
    }
}
/// Chief rows: host-prepared rows bound from the plan, or device rows.
enum Seeds {
    Host(PlanView),
    Device(Rows),
}
/// Seed table: host-built rows bound from the plan, or device rows.
enum Table {
    Host(PlanView),
    Device(Rows),
}
impl Seeds {
    fn bind(&self, e: &Encoder<'_>, index: usize, x: &Execution<'_>) {
        match self {
            Self::Host(v) => bind(e, index, x, v),
            Self::Device(b) => e.set_buffer(index, b, 0),
        }
    }
}
fn rows(
    x: &mut Execution<'_>,
    key: Option<OperationCacheKey>,
    words: usize,
    build: impl FnOnce(&mut Execution<'_>, &mut SharedBuffer<u32>) -> MkResult<()>,
) -> MkResult<Rows> {
    if let Some(k) = &key
        && let Some(b) = x.cached::<SharedBuffer<u32>>(k)?
    {
        return Ok(Rows::Cached(b));
    }
    Ok(match key {
        Some(k) => {
            let start = Instant::now();
            x.stats.allocations += 1;
            let mut b = x.gpu.buffer_filled(words, |_| {})?;
            build(x, &mut b)?;
            let b = Arc::new(b);
            // Published when the command holding the build completes; a
            // cancelled or failed plan never publishes it.
            x.cache(k, words * 4, BuildCost::Since(start), b.clone());
            Rows::Cached(b)
        }
        None => {
            let mut b = x.allocate(words)?;
            build(x, &mut b)?;
            Rows::Local(b)
        }
    })
}
/// Device seconds aimed for per Newton command: commands stay well inside
/// the cancellation budget while amortising submission latency.
const AIM_COMMAND_SECONDS: f64 = 0.015;
/// Rays per trace/projection command (bounded work per ray).
const TRACE_CHUNK: usize = 1 << 18;
const PROJECT_CHUNK: usize = 1 << 20;
/// Iteration units per lane of a tile's first command (rays converging from
/// a table or chief seed need two to three) and of each continuation
/// command. A unit is at most 24 stop traces; a fully divergent tile of hard
/// candidate rays ran 87 ms per command at eight units, ~45 ms at four.
const AIM_FIRST_BUDGET: u32 = 4;
const AIM_CONTINUE_BUDGET: u32 = 4;
/// The budgets, or the qualification override `MKLIGHT_AIM_BUDGET=first,
/// continue` (tests force many continuation rounds with `1,1`).
fn budgets() -> (u32, u32) {
    static B: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("MKLIGHT_AIM_BUDGET")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .filter(|&(a, b)| a > 0 && b > 0)
            .unwrap_or((AIM_FIRST_BUDGET, AIM_CONTINUE_BUDGET))
    })
}
/// Continuation rounds of a tile's first group; each speculative group after
/// it doubles.
const AIM_FIRST_ROUNDS: usize = 2;
/// Rounds that complete any schedule: six solves of a phase entry plus `cap`
/// iterations each, at `budget` units per round.
fn max_rounds(cap: u32, budget: u32) -> usize {
    (6 * (cap as usize + 1)).div_ceil(budget as usize) + 1
}

/// Resumable-aim buffers of one tile: per-lane state, two pending lists
/// (alternating by round), the indirect threadgroup arguments of each
/// continuation round and the pending count after each round. The counts of
/// each round group live in their own lease, so the host can read a
/// completed group's last count while a later group, which only reads it,
/// is still running.
struct Resume {
    state: Lease,
    lists: [Lease; 2],
    /// (first round, counts of the group's rounds), in round order.
    counts: Vec<(usize, Lease)>,
    args: Lease,
    rounds: usize,
}
impl Resume {
    fn new(x: &mut Execution<'_>, lanes: usize, rounds: usize) -> MkResult<Self> {
        let mut r = Self {
            state: x.allocate(lanes * MKL_AIM_STATE_WORDS as usize)?,
            lists: [x.allocate(lanes.max(1))?, x.allocate(lanes.max(1))?],
            counts: vec![],
            args: x.allocate(3 * rounds)?,
            rounds,
        };
        r.group(x, 0, 1)?;
        Ok(r)
    }
    /// Zeroed count storage for rounds `first..first + len` (the first
    /// command appends to round 0's count; the indirect kernel resets the
    /// count of each continuation round before it runs).
    fn group(&mut self, x: &mut Execution<'_>, first: usize, len: usize) -> MkResult<()> {
        let mut c = x.allocate(len)?;
        c.as_mut_slice().fill(0);
        self.counts.push((first, c));
        Ok(())
    }
    fn count(&self, round: usize) -> (&Lease, usize) {
        let (first, c) = self.counts.iter().rev().find(|c| c.0 <= round).unwrap();
        (c, round - first)
    }
    fn count_mut(&mut self, round: usize) -> (&mut Lease, usize) {
        let (first, c) = self.counts.iter_mut().rev().find(|c| c.0 <= round).unwrap();
        (c, round - *first)
    }
    /// The pending count after `round`, once its command has completed.
    fn pending(&self, round: usize) -> u32 {
        let (c, i) = self.count(round);
        c.as_slice()[i]
    }
    /// Bind the resume buffers of `round` (0: the first command) at the
    /// kernel's resume slots.
    fn bind(&mut self, e: &Encoder<'_>, round: usize) {
        let [a, b] = &mut self.lists;
        let (out, inp) = if round.is_multiple_of(2) {
            (a, b)
        } else {
            (b, a)
        };
        e.set_buffer_mut(15, out, 0);
        e.set_buffer(17, inp, 0);
        e.set_buffer_mut(14, &mut self.state, 0);
        let (c, i) = self.count_mut(round);
        e.set_buffer_mut(16, c, i);
        let (c, i) = self.count(round.saturating_sub(1));
        e.set_buffer(18, c, i);
    }
    fn recycle(self, x: &mut Execution<'_>) {
        let [a, b] = self.lists;
        for b in [self.state, a, b, self.args] {
            x.recycle(b);
        }
        for (_, c) in self.counts {
            x.recycle(c);
        }
    }
}

/// Newton work per thread varies by orders of magnitude between systems and
/// pupil regions, so commands follow the measured device time: they start
/// small, then grow at most eightfold towards [`AIM_COMMAND_SECONDS`].
/// Threads are independent, so the split never changes a result. With a
/// continuation pipeline every command additionally runs a bounded number
/// of iteration units per lane: the tile's first command parks unfinished
/// rays, and continuation commands (sized on the device from the pending
/// count) resume them in rounds. Rounds go out in groups; the pending count
/// after a group decides whether more are needed, but the next group is
/// committed before the host reads it, so the device never idles on that
/// readback. Once a group leaves nothing pending, the speculative group
/// after it finds empty lists and dispatches no thread. With `speculative`
/// rounds (and the plan allowing speculation) a tile commits that many
/// rounds and never reads its count; the plan verifies it at completion.
#[allow(clippy::too_many_arguments)]
fn newton_commands(
    x: &mut Execution<'_>,
    stage: &mut StageStats,
    ctl: &JobControl,
    n: usize,
    cap: u32,
    progress: (f64, f64, &str),
    pipelines: (&PipelineKey, Option<&PipelineKey>),
    speculative: usize,
    mut encode: impl FnMut(&Encoder<'_>, &Execution<'_>, AimParams),
) -> MkResult<()> {
    const MIN: usize = 4096;
    const MAX: usize = 1 << 18;
    let first = x.gpu.pipeline(pipelines.0)?;
    let cont = pipelines.1.map(|k| x.gpu.pipeline(k)).transpose()?;
    let indirect = cont
        .as_ref()
        .map(|_| x.gpu.pipeline(&PipelineKey::plain("mkl_aim_indirect")))
        .transpose()?;
    let (first_budget, continue_budget) = budgets();
    let (mut offset, mut next) = (0, MIN);
    while offset < n {
        ctl.check()?;
        let count = (n - offset).min(next);
        let p = AimParams {
            count: count as u32,
            offset: offset as u32,
            budget: first_budget,
            ..Zeroable::zeroed()
        };
        let wall = Instant::now();
        let mut resume = cont
            .as_ref()
            .map(|_| Resume::new(x, count, max_rounds(cap, continue_budget)))
            .transpose()?;
        // SAFETY: callers bind plan-validated extents; kernels check every
        // device-provided field, system and wavelength index; the resume
        // buffers hold `count` lanes and `rounds` rounds.
        unsafe {
            x.run(progress.2, |e| {
                encode(e, x, p);
                if let Some(r) = &mut resume {
                    r.bind(e, 0);
                }
                e.dispatch_1d(&first, count);
                Ok(())
            })?
        };
        account(x, stage, wall, size_of::<AimParams>());
        let more = offset + count < n;
        let calibrate = n > 65536 || (offset == 0 && more);
        let mut seconds = None;
        if let (Some(r), Some(cont), Some(indirect)) = (&mut resume, &cont, &indirect) {
            let tile = x.ticket();
            let width = Encoder::width(cont) as u32;
            let p = AimParams {
                budget: continue_budget,
                ..p
            };
            let mut rounds =
                |x: &mut Execution<'_>, r: &mut Resume, from: usize, to: usize, fused: bool| {
                    r.group(x, from + 1, to - from)?;
                    for round in from + 1..=to {
                        // One command buffer per round: a round of hard rays
                        // runs up to the first command's time, and fused rounds
                        // would add up past the interactivity bound. A first
                        // speculative round (expected empty) stays with the first
                        // command.
                        if !fused || round > from + 1 {
                            x.submit()?;
                            // Committing may have waited on the in-flight
                            // window: stay cancellable between rounds.
                            ctl.check()?;
                        }
                        let wall = Instant::now();
                        // SAFETY: as above; the indirect arguments and counts of
                        // round `round` lie inside the resume buffers.
                        unsafe {
                            x.run("aim continuation", |e| {
                                let (c, i) = r.count(round - 1);
                                e.set_buffer(0, c, i);
                                e.set_buffer_mut(1, &mut r.args, 3 * (round - 1));
                                let (c, i) = r.count_mut(round);
                                e.set_buffer_mut(2, c, i);
                                e.set_bytes(3, &width);
                                e.dispatch_1d(indirect, 1);
                                encode(e, x, p);
                                r.bind(e, round);
                                e.dispatch_indirect(cont, &r.args, 3 * (round - 1));
                                Ok(())
                            })?
                        };
                        account(x, stage, wall, size_of::<AimParams>());
                    }
                    Ok::<_, MkError>(to)
                };
            if speculative > 0 && x.speculate {
                // Committed without reading the count back: verified when
                // the plan completes (the backend reruns it otherwise).
                let round = rounds(x, r, 0, speculative.min(r.rounds), true)?;
                x.submit()?;
                if calibrate {
                    seconds = Some(x.complete(tile)?.unwrap_or(0.));
                }
                let (first, count) = r.counts.pop().unwrap();
                x.verify(count, round - first);
            } else {
                let mut group = AIM_FIRST_ROUNDS;
                let mut round = rounds(x, r, 0, group.min(r.rounds), false)?;
                let mut decided = (x.submit()?, round);
                loop {
                    let ahead = if round < r.rounds {
                        round = rounds(x, r, round, (round + group).min(r.rounds), false)?;
                        group *= 2;
                        Some((x.submit()?, round))
                    } else {
                        None
                    };
                    if seconds.is_none() && calibrate {
                        seconds = Some(x.complete(tile)?.unwrap_or(0.));
                    }
                    x.complete(decided.0)?;
                    x.stats.readback_bytes += 4;
                    let pending = r.pending(decided.1);
                    if pending == 0 {
                        break;
                    }
                    let Some(a) = ahead else {
                        return Err(MkError::Numerical(format!(
                            "{pending} aims unfinished after {} continuation rounds",
                            decided.1
                        )));
                    };
                    ctl.check()?;
                    decided = a;
                }
            }
        }
        if let Some(r) = resume.take() {
            r.recycle(x);
        }
        // Single-tile batches without continuation stay fused with their
        // chief/table/pupil chain. Otherwise the first tile completes to
        // calibrate the next; later tiles of mid-sized batches keep that rate.
        // Every tile of a large batch (~15 ms each) completes, so progress
        // follows device completion.
        if calibrate {
            let seconds = match seconds {
                Some(s) => s,
                None => x.measure()?,
            };
            if seconds > 0. {
                let fit = (count as f64 * AIM_COMMAND_SECONDS / seconds) as usize;
                next = fit.clamp(MIN, (8 * count).clamp(MIN, MAX));
            }
        }
        offset += count;
        ctl.check()?;
        ctl.report(
            progress.0 + progress.1 * offset as f64 / n as f64,
            progress.2,
        );
    }
    Ok(())
}
fn account(x: &mut Execution<'_>, stage: &mut StageStats, start: Instant, bytes: usize) {
    stage.max_command_seconds = stage.max_command_seconds.max(start.elapsed().as_secs_f64());
    x.stats.parameter_bytes += bytes as u64;
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        ..Default::default()
    };
    match r {
        Request::GenerateAim(g) => {
            stage.name = "generate_aim".into();
            let systems = Systems::new(x, &g.systems)?;
            let nf = g.field_wavelength.desc.shape[0];
            let n = g.output.count;
            let (pattern, side, extent) = g.sampling.parameters();
            let table_side = g.options.table_side;
            let params = |p: AimParams| AimParams {
                stride: n as u32,
                fields: nf as u32,
                systems: systems.count(),
                cap: g.max_iterations,
                tolerance: g.tolerance,
                pattern,
                side,
                extent: [extent.hi, extent.lo],
                table_side,
                flags: if g.options.cull_outside { AIM_CULL } else { 0 },
                ..p
            };
            let newton = kernel::newton_params();
            // Reuse follows the content of every input, never handles: the
            // immutable upload, the imported field rows and host seed rows.
            let identity = aim_identity(
                g,
                match &systems {
                    Systems::Uploaded(up) => Some(up.content),
                    Systems::Resident { .. } => None,
                },
                x.import_words(&g.field_wavelength),
                g.seeds.as_ref().and_then(|v| x.import_words(v)),
            );
            let key = |x: &Execution<'_>, category, what: &str| {
                identity
                    .as_ref()
                    .map(|id| x.content_key(category, format!("{what}:{id}")))
            };
            // Host rows (the reference tracer's f64 preparation) replace the
            // device chief/image-height solve entirely.
            let seeds = match &g.seeds {
                Some(v) => Seeds::Host(v.clone()),
                None => Seeds::Device(rows(
                    x,
                    key(x, CacheCategory::ChiefSources, "aim chiefs"),
                    nf * MKL_AIM_CACHE_WORDS as usize,
                    |x, seeds| {
                        let pso = systems.kernel("mkl_aim_chiefs", "mkl_aim_chiefs_overlays");
                        newton_commands(
                            x,
                            &mut stage,
                            ctl,
                            nf,
                            g.max_iterations,
                            (0.0, 0.05, "chief aiming"),
                            (&pso, None),
                            0,
                            |e, x, p| {
                                systems.bind(e, x);
                                bind(e, 3, x, &g.field_wavelength);
                                e.set_buffer_mut(4, seeds, 0);
                                e.set_bytes(5, &params(p));
                                e.set_bytes(6, &newton);
                            },
                        )
                    },
                )?),
            };
            // A host table is bound as supplied; otherwise a device table
            // is built (and cached by content) for large generations.
            let nodes = table_nodes(table_side);
            let table = match &g.options.table {
                Some(v) => Some(Table::Host(v.clone())),
                None if uses_table(g) => {
                    let words = nf * nodes * MKL_AIM_NODE_WORDS as usize;
                    let table_key = key(x, CacheCategory::SeedTables, "aim table");
                    Some(Table::Device(rows(x, table_key, words, |x, table| {
                        let pso = systems.kernel("mkl_aim_table", "mkl_aim_table_overlays");
                        let cont = systems
                            .kernel("mkl_aim_table_continue", "mkl_aim_table_continue_overlays");
                        newton_commands(
                            x,
                            &mut stage,
                            ctl,
                            nf * nodes,
                            g.max_iterations,
                            (0.05, 0.05, "seed table"),
                            (&pso, Some(&cont)),
                            0,
                            |e, x, p| {
                                systems.bind(e, x);
                                bind(e, 3, x, &g.field_wavelength);
                                seeds.bind(e, 4, x);
                                e.set_bytes(5, &params(p));
                                e.set_bytes(6, &newton);
                                e.set_buffer_mut(7, table, 0);
                            },
                        )
                    })?))
                }
                None => None,
            };
            let mut out = take_rays(x, &g.output);
            let mut failures = x.allocate(1)?;
            failures.as_mut_slice()[0] = 0;
            x.stats.uploaded_bytes += 4;
            let pso = systems.kernel("mkl_generate_aim", "mkl_generate_aim_overlays");
            let cont = systems.kernel("mkl_aim_continue", "mkl_aim_continue_overlays");
            newton_commands(
                x,
                &mut stage,
                ctl,
                n,
                g.max_iterations,
                (0.1, 0.9, "pupil aiming"),
                (&pso, Some(&cont)),
                g.options.speculative_rounds as usize,
                |e, x, p| {
                    let [ff, uu, ids, reports] = &mut out;
                    systems.bind(e, x);
                    bind(e, 3, x, &g.field_wavelength);
                    seeds.bind(e, 4, x);
                    e.set_bytes(
                        5,
                        &AimParams {
                            table: u32::from(table.is_some()),
                            ..params(p)
                        },
                    );
                    bind(e, 6, x, &g.samples);
                    e.set_buffer_mut(7, ff, 0);
                    e.set_buffer_mut(8, uu, 0);
                    e.set_buffer_mut(9, ids, 0);
                    e.set_buffer_mut(10, reports, 0);
                    e.set_buffer_mut(11, &mut failures, 0);
                    e.set_bytes(12, &newton);
                    // Without a table the binding is never read.
                    match &table {
                        Some(Table::Host(v)) => bind(e, 13, x, v),
                        Some(Table::Device(t)) => e.set_buffer(13, t, 0),
                        None => seeds.bind(e, 13, x),
                    }
                },
            )?;
            aim_stats(x, &out[3], n, &mut stage)?;
            if let Seeds::Device(Rows::Local(b)) = seeds {
                x.recycle(b);
            }
            if let Some(Table::Device(Rows::Local(b))) = table {
                x.recycle(b);
            }
            x.count_fallbacks(failures);
            put_rays(x, &g.output, out);
            if let Some(id) = g.unresolved {
                compact(x, g.output.report, id, n, ctl, &mut stage)?;
            }
        }
        Request::TraceResident(t) => {
            stage.name = "trace_resident".into();
            let systems = Systems::new(x, &t.systems)?;
            if t.radiometry != 0 {
                match &systems {
                    Systems::Uploaded(up)
                        if up
                            .set
                            .surfaces
                            .iter()
                            .all(|s| s.coating != MKL_COATING_TABLE) => {}
                    _ => return unsupported("resident table-coating radiometry"),
                }
            }
            let n = t.input.count;
            let mut out = take_rays(x, &t.output.rays);
            let mut status = x.take_output(t.output.status);
            let mut hi = if let Some(id) = t.output.records {
                x.take_output(id)
            } else {
                x.allocate(0)?
            };
            let mut lo = if let Some(id) = t.output.record_low {
                x.take_output(id)
            } else {
                x.allocate(0)?
            };
            let pso = x.gpu.pipeline(&if t.output.records.is_some() {
                systems.kernel(
                    "mkl_trace_resident_records",
                    "mkl_trace_resident_overlays_records",
                )
            } else {
                systems.kernel("mkl_trace_resident", "mkl_trace_resident_overlays")
            })?;
            let recorded = t.output.records.is_some();
            let chunk = if recorded {
                TRACE_CHUNK / 2
            } else {
                TRACE_CHUNK
            };
            for offset in (0..n).step_by(chunk) {
                ctl.check()?;
                let count = (n - offset).min(chunk);
                let p = ResidentTraceParams {
                    trace: TraceParams {
                        num_rays: count as u32,
                        stride: n as u32,
                        num_steps: t.steps.desc.len()? as u32,
                        start_surface: t.start_surface,
                        max_iterations: t.newton.max_iterations,
                        tolerance_abs: t.newton.tolerance_abs,
                        tolerance_rel: t.newton.tolerance_rel,
                        flags: if t.apply_apertures { MKL_TRACE_CLIP } else { 0 }
                            | if t.radiometry != 0 {
                                MKL_TRACE_RADIOMETRY
                            } else {
                                0
                            }
                            | if t.output.records.is_some() {
                                MKL_TRACE_RECORD
                            } else {
                                0
                            },
                    },
                    offset: offset as u32,
                    systems: systems.count(),
                    reserved: [0; 2],
                };
                let [ff, uu, ids, reports] = &mut out;
                let wall = Instant::now();
                // SAFETY: extents are plan-validated; dynamic system/path and
                // wavelength indices are checked before the shared trace core.
                unsafe {
                    x.run("resident trace", |e| {
                        systems.bind(e, x);
                        bind(e, 3, x, &t.steps);
                        e.set_bytes(4, &p);
                        bind(e, 5, x, &view(t.input.floats, ElementType::F32));
                        bind(e, 6, x, &view(t.input.indices, ElementType::U32));
                        bind(e, 7, x, &view(t.input.logical_ids, ElementType::U32));
                        bind(e, 8, x, &view(t.input.report, ElementType::U32));
                        e.set_buffer_mut(9, ff, 0);
                        e.set_buffer_mut(10, uu, 0);
                        e.set_buffer_mut(11, ids, 0);
                        e.set_buffer_mut(12, reports, 0);
                        e.set_buffer_mut(13, &mut status, 0);
                        e.set_buffer_mut(14, &mut hi, 0);
                        e.set_buffer_mut(15, &mut lo, 0);
                        e.dispatch_1d(&pso, count);
                        Ok(())
                    })?
                };
                if recorded && n > chunk {
                    // Full hi/lo surface records cost more than final-ray
                    // output; complete each bounded record tile before reuse.
                    x.measure()?;
                }
                account(x, &mut stage, wall, size_of::<ResidentTraceParams>());
                ctl.check()?;
                ctl.report((offset + count) as f64 / n as f64, "resident trace");
            }
            put_rays(x, &t.output.rays, out);
            x.put_output(t.output.status, status);
            if let Some(id) = t.output.records {
                x.put_output(id, hi);
            } else {
                x.recycle(hi);
            }
            if let Some(id) = t.output.record_low {
                x.put_output(id, lo);
            } else {
                x.recycle(lo);
            }
        }
        Request::ProjectPupil(q) => {
            stage.name = "project_pupil".into();
            let n = q.hits.rays.count;
            let mut out = x.take_output(q.output);
            let mut status = x.take_output(q.status);
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_project_pupil"))?;
            for offset in (0..n).step_by(PROJECT_CHUNK) {
                ctl.check()?;
                let count = (n - offset).min(PROJECT_CHUNK);
                let r = &q.reference;
                let p = ProjectParams {
                    radius: [r.radius.hi, r.radius.lo],
                    index: [r.image_index.hi, r.image_index.lo],
                    wavelength: [r.wavelength_um.hi, r.wavelength_um.lo],
                    sheet: r.sphere_sheet,
                    telecentric: u32::from(r.telecentric),
                    count: count as u32,
                    offset: offset as u32,
                    stride: n as u32,
                    reserved: 0,
                };
                let wall = Instant::now();
                // SAFETY: plan validation fixed all plane and reference sizes.
                unsafe {
                    x.run("project pupil", |e| {
                        bind(e, 0, x, &view(q.hits.rays.floats, ElementType::F32));
                        bind(e, 1, x, &view(q.hits.status, ElementType::U32));
                        bind(e, 2, x, &r.chief);
                        bind(e, 3, x, &r.centre);
                        e.set_bytes(4, &p);
                        e.set_buffer_mut(5, &mut out, 0);
                        e.set_buffer_mut(6, &mut status, 0);
                        e.dispatch_1d(&pso, count);
                        Ok(())
                    })?
                };
                account(x, &mut stage, wall, size_of::<ProjectParams>());
                ctl.check()?;
                ctl.report((offset + count) as f64 / n as f64, "pupil projection");
            }
            x.put_output(q.output, out);
            x.put_output(q.status, status);
        }
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(stage);
    ctl.check()
}

/// Convergence statistics of a generation, reduced on the device and read
/// at completion: [failed, iterations (low word), max iterations,
/// histogram[8], iterations (high word)].
fn aim_stats(
    x: &mut Execution<'_>,
    report: &Lease,
    n: usize,
    stage: &mut StageStats,
) -> MkResult<()> {
    if n == 0 {
        return Ok(());
    }
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_aim_stats"))?;
    let mut words = x.allocate(AIM_STATS_WORDS)?;
    words.as_mut_slice().fill(0);
    let wall = Instant::now();
    // SAFETY: the report holds 4n words; each thread reads 256 disjoint rays
    // and updates the AIM_STATS_WORDS counters atomically.
    unsafe {
        x.run("aim statistics", |e| {
            e.set_buffer(0, report, 0);
            e.set_buffer_mut(1, &mut words, 0);
            e.set_bytes(2, &(n as u32));
            e.dispatch_1d(&pso, n.div_ceil(256));
            Ok(())
        })?
    };
    account(x, stage, wall, 4);
    x.stats.readback_bytes += AIM_STATS_WORDS as u64 * 4;
    x.check_buffer(words, |b, stats| {
        let w = b.as_slice();
        let histogram: [u64; 8] = std::array::from_fn(|k| w[3 + k] as u64);
        stats.aiming.add(&AimStats {
            generations: 1,
            rays: histogram.iter().sum(),
            failed: w[0] as u64,
            iterations: w[1] as u64 | (w[11] as u64) << 32,
            max_iterations: w[2],
            histogram,
        });
        Ok(())
    });
    Ok(())
}
const AIM_STATS_WORDS: usize = 12;

/// Stable unresolved-index compaction: count, prefix and scatter as three
/// serial dispatches of one command.
fn compact(
    x: &mut Execution<'_>,
    report: BufferId,
    output: BufferId,
    n: usize,
    ctl: &JobControl,
    stage: &mut StageStats,
) -> MkResult<()> {
    ctl.check()?;
    let groups = n.div_ceil(256);
    let mut counts = x.allocate(groups)?;
    let mut sparse = x.take_output(output);
    let [count, prefix, scatter] = ["mkl_aim_count", "mkl_aim_prefix", "mkl_aim_scatter"]
        .map(|k| x.gpu.pipeline(&PipelineKey::plain(k)));
    let (count, prefix, scatter) = (count?, prefix?, scatter?);
    let wall = Instant::now();
    // SAFETY: plan validation fixes the sparse size; each group owns a
    // disjoint source range and prefix-selected destination range, and the
    // serial dispatches of one encoder complete in order.
    unsafe {
        x.run("aim compaction", |e| {
            bind(e, 0, x, &view(report, ElementType::U32));
            e.set_buffer_mut(1, &mut counts, 0);
            e.set_buffer_mut(2, &mut sparse, 0);
            e.set_bytes(3, &(n as u32));
            e.dispatch_1d(&count, groups.max(1));
            e.set_buffer_mut(0, &mut counts, 0);
            e.set_buffer_mut(1, &mut sparse, 0);
            e.set_bytes(2, &(groups as u32));
            e.dispatch_1d(&prefix, 1);
            bind(e, 0, x, &view(report, ElementType::U32));
            e.set_buffer_mut(1, &mut counts, 0);
            e.set_buffer_mut(2, &mut sparse, 0);
            e.set_bytes(3, &(n as u32));
            e.dispatch_1d(&scatter, groups.max(1));
            Ok(())
        })?
    };
    account(x, stage, wall, 8);
    x.recycle(counts);
    x.put_output(output, sparse);
    Ok(())
}
