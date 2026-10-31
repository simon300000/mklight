//! Resident Metal collectives. Each operation builds its dispatch list and
//! encodes it serially into as few command buffers as the work bound allows;
//! command boundaries never change a reduction tree.
use crate::{
    context::PipelineKey,
    resident::{Execution, Lease},
};
use bytemuck::{Pod, Zeroable};
use mklight_core::MkResult;
use mklight_gpu::{
    collectives::*,
    consts::*,
    pending::JobControl,
    reduce,
    resident::{ElementType as Ty, PlanView, StageStats, unsupported},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;

/// Work units per command buffer. Units are calibrated per kernel to about
/// 0.1-0.2 ns on an M2 Pro, so a command stays near 10 ms, far below the
/// 50 ms cancellation target.
const COMMAND_WORK: usize = 1 << 26;
/// Partial-sum storage bound for long segments.
const PARTIAL_BYTES: usize = 16 << 20;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct SegmentParams {
    count: u32,
    channels: u32,
    segments: u32,
    first: u32,
    end: u32,
    max_len: u32,
    max_groups: u32,
    kind: u32,
    mode: u32,
    phase: u32,
    value_words: u32,
    reference_words: u32,
    weight_words: u32,
    masked: u32,
    mask_value: u32,
    terms: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct QrParams {
    rows: u32,
    cols: u32,
    batch: u32,
    step: u32,
    input_words: u32,
    tolerance_bits: u32,
    aux_stride: u32,
    threads: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct StepParams {
    n: u32,
    kind: u32,
    flag: u32,
    words: u32,
    shift: u32,
    blocks: u32,
    value_words: u32,
    reserved: u32,
}

#[derive(Clone)]
enum Buf {
    View(PlanView),
    Slot(usize),
}
struct Dispatch {
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buffers: Vec<Buf>,
    params: [u32; 16],
    groups: [usize; 3],
    threads: usize,
    work: usize,
}
/// Leases (outputs and scratch) and the dispatches that use them.
#[derive(Default)]
struct Plan {
    slots: Vec<Lease>,
    list: Vec<Dispatch>,
}
impl Plan {
    fn scratch(&mut self, x: &mut Execution<'_>, words: usize) -> MkResult<Buf> {
        self.slots.push(x.allocate(words.max(1))?);
        Ok(Buf::Slot(self.slots.len() - 1))
    }
    fn output(&mut self, x: &mut Execution<'_>, id: mklight_gpu::resident::BufferId) -> Buf {
        self.slots.push(x.take_output(id));
        Buf::Slot(self.slots.len() - 1)
    }
    fn push<P: Pod>(
        &mut self,
        x: &Execution<'_>,
        kernel: &'static str,
        buffers: Vec<Buf>,
        params: P,
        groups: [usize; 3],
        work: usize,
    ) -> MkResult<()> {
        self.push_threads(x, kernel, buffers, params, groups, 256, work)
    }
    #[allow(clippy::too_many_arguments)]
    fn push_threads<P: Pod>(
        &mut self,
        x: &Execution<'_>,
        kernel: &'static str,
        buffers: Vec<Buf>,
        params: P,
        groups: [usize; 3],
        threads: usize,
        work: usize,
    ) -> MkResult<()> {
        let pso = x.gpu.pipeline(&PipelineKey::plain(kernel))?;
        if pso.maxTotalThreadsPerThreadgroup() < threads {
            return unsupported("collective threadgroup size");
        }
        let mut words = [0u32; 16];
        let p: &[u32] = bytemuck::cast_slice(std::slice::from_ref(&params));
        words[..p.len()].copy_from_slice(p);
        self.list.push(Dispatch {
            pso,
            buffers,
            params: words,
            groups,
            threads,
            work: work.max(1),
        });
        Ok(())
    }
    /// One-dimensional dispatch of `n` threads in groups of 256.
    fn push_1d<P: Pod>(
        &mut self,
        x: &Execution<'_>,
        kernel: &'static str,
        buffers: Vec<Buf>,
        params: P,
        n: usize,
    ) -> MkResult<()> {
        self.push(x, kernel, buffers, params, [n.div_ceil(256).max(1), 1, 1], n)
    }
    /// Encodes the dispatches serially, cutting command buffers at the work
    /// bound; returns the leases for publication or recycling.
    fn run(
        mut self,
        x: &mut Execution<'_>,
        ctl: &JobControl,
        stats: &mut StageStats,
    ) -> MkResult<Vec<Lease>> {
        if let Some(d) = self.list.first()
            && d.pso.threadExecutionWidth() != 32
        {
            return unsupported("collectives require 32-wide SIMD groups");
        }
        let total: usize = self.list.iter().map(|d| d.work).sum();
        let (mut start, mut done) = (0, 0);
        while start < self.list.len() {
            let mut end = start + 1;
            let mut work = self.list[start].work;
            while end < self.list.len() && work + self.list[end].work <= COMMAND_WORK {
                work += self.list[end].work;
                end += 1;
            }
            ctl.check()?;
            let wall = Instant::now();
            let (xr, slots, list) = (&*x, &mut self.slots, &self.list[start..end]);
            // SAFETY: request validation fixed every view's type and extent;
            // scratch and outputs are sized for their dispatches, whose kernels
            // bound all indices by the parameter extents. The encoder is serial.
            let cs = unsafe {
                xr.gpu.run("collectives", |e| {
                    for d in list {
                        e.set_pipeline(&d.pso);
                        for (i, b) in d.buffers.iter().enumerate() {
                            match b {
                                Buf::View(v) => {
                                    let (b, o) = xr.buffer(v);
                                    e.set_buffer(i, b, o);
                                }
                                Buf::Slot(s) => e.set_buffer_mut(i, &mut slots[*s], 0),
                            }
                        }
                        e.set_bytes(d.buffers.len(), &d.params);
                        e.dispatch_groups(d.groups, d.threads);
                    }
                    Ok(())
                })?
            };
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
            x.stats.parameter_bytes += 64 * (end - start) as u64;
            done += work;
            ctl.check()?;
            ctl.report(done as f64 / total as f64, "collectives");
            start = end;
        }
        Ok(self.slots)
    }
}
fn finish(
    plan: Plan,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
    outputs: &[mklight_gpu::resident::BufferId],
) -> MkResult<()> {
    let mut slots = plan.run(x, ctl, stats)?.into_iter();
    for &id in outputs {
        x.put_output(id, slots.next().unwrap());
    }
    slots.for_each(|s| x.recycle(s));
    Ok(())
}
fn view_or(v: &Option<PlanView>, fallback: &PlanView) -> Buf {
    Buf::View(v.as_ref().unwrap_or(fallback).clone())
}
fn words(v: &Option<PlanView>) -> u32 {
    v.as_ref().map_or(0, |v| v.desc.element.words() as u32)
}

pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stats = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        name: "collectives".into(),
        ..Default::default()
    };
    match r {
        Request::SegmentReduce(r) => segments(r, x, ctl, &mut stats)?,
        Request::Scan(r) => {
            let n = r.values.desc.shape[0];
            let mut plan = Plan::default();
            let out = plan.output(x, r.output);
            if n > 0 {
                let kind = kind(r.values.desc.element);
                let levels = ScanLevels::new(&mut plan, x, n, kind)?;
                let input = Buf::View(r.values.clone());
                levels.level(&mut plan, x, 0, input, n, kind, false, out, !r.inclusive)?;
            }
            finish(plan, x, ctl, &mut stats, &[r.output])?;
        }
        Request::StableCompact(r) => compact(r, x, ctl, &mut stats)?,
        Request::StableSort(r) => sort(r, x, ctl, &mut stats)?,
        Request::QrFit(r) => qr(r, x, ctl, &mut stats)?,
    }
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}
fn kind(t: Ty) -> u32 {
    match t {
        Ty::U32 => 0,
        Ty::F32 => 1,
        _ => 2,
    }
}

fn segments(
    r: &SegmentReduce,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let [c, n] = r.values.desc.shape[..] else {
        unreachable!()
    };
    let s = r.offsets.desc.shape[0] - 1;
    let mut plan = Plan::default();
    let out = plan.output(x, r.output);
    let status = plan.output(x, r.status);
    let terms = |phase: usize| match r.kind {
        SegmentKind::Moments2 => [4, 6][phase],
        SegmentKind::Summary => 4 * c,
        _ => c,
    };
    let phases = if r.kind == SegmentKind::Moments2 { 2 } else { 1 };
    let all_terms: usize = (0..phases).map(terms).sum();
    let maxg = reduce::num_groups(r.max_segment_len);
    let values = Buf::View(r.values.clone());
    let inputs = vec![
        values.clone(),
        Buf::View(r.offsets.clone()),
        view_or(&r.reference, &r.values),
        view_or(&r.mask, &r.values),
        view_or(&r.weights, &r.values),
    ];
    let base = SegmentParams {
        count: n as u32,
        channels: c as u32,
        segments: s as u32,
        max_len: r.max_segment_len as u32,
        max_groups: maxg as u32,
        kind: r.kind as u32,
        mode: r.mode as u32,
        value_words: r.values.desc.element.words() as u32,
        reference_words: words(&r.reference),
        weight_words: words(&r.weights),
        masked: u32::from(r.mask.is_some()),
        mask_value: r.mask_value,
        ..Default::default()
    };
    // Work of a tile of segments: at most max_len records each, n in total.
    let work = |segments: usize, terms: usize| (segments * r.max_segment_len).min(n).max(segments) * terms;
    // Both paths reproduce the same trees; a SIMD group per segment wins
    // when segments are many or tiny, tree groups when few are long.
    if maxg == 1 && (s >= 512 || r.max_segment_len <= 64) {
        let tile = (COMMAND_WORK / (4 * r.max_segment_len.max(1) * all_terms)).max(8) / 8 * 8;
        for first in (0..s).step_by(tile) {
            let end = (first + tile).min(s);
            let p = SegmentParams {
                first: first as u32,
                end: end as u32,
                ..base
            };
            let mut buffers = inputs.clone();
            buffers.extend([out.clone(), status.clone()]);
            let units = 4 * work(end - first, all_terms);
            plan.push(x, "mkl_c_seg_short", buffers, p, [(end - first).div_ceil(8), 1, 1], units)?;
        }
    } else {
        let max_terms = (0..phases).map(terms).max().unwrap();
        let tile = (PARTIAL_BYTES / (8 * max_terms * maxg))
            .min(COMMAND_WORK / (r.max_segment_len * max_terms))
            .max(1);
        let partials = plan.scratch(x, 2 * tile.min(s) * max_terms * maxg)?;
        for first in (0..s).step_by(tile) {
            let end = (first + tile).min(s);
            for phase in 0..phases {
                let p = SegmentParams {
                    first: first as u32,
                    end: end as u32,
                    phase: phase as u32,
                    terms: terms(phase) as u32,
                    ..base
                };
                let mut buffers = inputs.clone();
                buffers.extend([out.clone(), partials.clone()]);
                plan.push(x, "mkl_c_seg_pass1", buffers, p, [maxg, terms(phase), end - first], work(end - first, terms(phase)))?;
                let buffers = vec![Buf::View(r.offsets.clone()), partials.clone(), out.clone(), status.clone()];
                plan.push(x, "mkl_c_seg_pass2", buffers, p, [(end - first).div_ceil(8), 1, 1], (end - first) * terms(phase) * maxg)?;
            }
        }
    }
    finish(plan, x, ctl, stats, &[r.output, r.status])
}

/// Scratch for a scan of n elements: per level, block-local prefixes, block
/// totals and (except at the last level) the scanned totals.
struct ScanLevels(Vec<[Buf; 3]>);
impl ScanLevels {
    fn new(plan: &mut Plan, x: &mut Execution<'_>, n: usize, kind: u32) -> MkResult<Self> {
        let words = if kind == 0 { 1 } else { 2 };
        let mut levels = vec![];
        let mut len = n;
        loop {
            let blocks = len.div_ceil(256);
            levels.push([
                plan.scratch(x, len * words)?,
                plan.scratch(x, blocks * words)?,
                plan.scratch(x, if blocks > 1 { blocks * words } else { 0 })?,
            ]);
            if blocks == 1 {
                return Ok(Self(levels));
            }
            len = blocks;
        }
    }
    /// Scan `n > 0` elements of `input` into `out`, starting at `level`.
    #[allow(clippy::too_many_arguments)]
    fn level(
        &self,
        plan: &mut Plan,
        x: &Execution<'_>,
        level: usize,
        input: Buf,
        n: usize,
        kind: u32,
        binary: bool,
        out: Buf,
        exclusive: bool,
    ) -> MkResult<()> {
        let [local, totals, prefix] = self.0[level].clone();
        let blocks = n.div_ceil(256);
        let p = StepParams {
            n: n as u32,
            kind,
            flag: u32::from(binary),
            ..Default::default()
        };
        plan.push(x, "mkl_c_scan_blocks", vec![input, local.clone(), totals.clone()], p, [blocks, 1, 1], n)?;
        if blocks > 1 {
            let k = if kind == 0 { 0 } else { 2 };
            self.level(plan, x, level + 1, totals, blocks, k, false, prefix.clone(), false)?;
        }
        let p = StepParams {
            n: n as u32,
            kind: if kind == 0 { 0 } else { 2 },
            flag: u32::from(exclusive),
            ..Default::default()
        };
        plan.push_1d(x, "mkl_c_scan_finish", vec![local, prefix, out], p, n)
    }
}

fn compact(
    r: &StableCompact,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let n = r.ids.desc.shape[0];
    let width = if n == 0 {
        0
    } else {
        r.values.desc.len()? * r.values.desc.element.words() / n
    };
    let mut plan = Plan::default();
    let out = plan.output(x, r.output);
    let ids = plan.output(x, r.output_ids);
    let count = plan.output(x, r.count);
    let prefix = plan.scratch(x, n)?;
    if n > 0 {
        let levels = ScanLevels::new(&mut plan, x, n, 0)?;
        levels.level(&mut plan, x, 0, Buf::View(r.mask.clone()), n, 0, true, prefix.clone(), false)?;
    }
    let p = StepParams {
        n: n as u32,
        words: width as u32,
        ..Default::default()
    };
    let buffers = vec![
        Buf::View(r.values.clone()),
        Buf::View(r.ids.clone()),
        Buf::View(r.mask.clone()),
        prefix,
        out,
        ids,
        count,
    ];
    plan.push_1d(x, "mkl_c_compact", buffers, p, n.max(1))?;
    finish(plan, x, ctl, stats, &[r.output, r.output_ids, r.count])
}

fn sort(
    r: &StableSort,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let n = r.ids.desc.shape[0];
    let mut plan = Plan::default();
    let keys = plan.output(x, r.output_keys);
    let ids = plan.output(x, r.output_ids);
    let values = match r.output_values {
        Some(o) => plan.output(x, o),
        None => plan.scratch(x, 0)?,
    };
    let mut outputs = vec![r.output_keys, r.output_ids];
    outputs.extend(r.output_values);
    if n == 0 {
        return finish(plan, x, ctl, stats, &outputs);
    }
    let kind = kind(r.keys.desc.element);
    let key_words = r.keys.desc.element.words() as u32;
    let mut a = plan.scratch(x, 4 * n)?;
    let mut b = plan.scratch(x, 4 * n)?;
    let blocks = n.div_ceil(MKL_COLLECT_SORT_BLOCK as usize);
    let hist = plan.scratch(x, 256 * blocks)?;
    let start = plan.scratch(x, 256 * blocks)?;
    let levels = ScanLevels::new(&mut plan, x, 256 * blocks, 0)?;
    let p = StepParams {
        n: n as u32,
        kind,
        words: key_words,
        ..Default::default()
    };
    let buffers = vec![Buf::View(r.keys.clone()), Buf::View(r.ids.clone()), a.clone()];
    plan.push_1d(x, "mkl_c_sort_init", buffers, p, n)?;
    // Least significant first: logical ID, then key words from low to high.
    let words: &[u32] = if kind == 2 { &[2, 1, 0] } else { &[2, 0] };
    for &word in words {
        for shift in [0, 8, 16, 24] {
            let p = StepParams {
                n: n as u32,
                flag: word,
                shift,
                blocks: blocks as u32,
                ..Default::default()
            };
            plan.push(x, "mkl_c_radix_count", vec![a.clone(), hist.clone()], p, [blocks, 1, 1], n)?;
            levels.level(&mut plan, x, 0, hist.clone(), 256 * blocks, 0, false, start.clone(), true)?;
            plan.push(x, "mkl_c_radix_scatter", vec![a.clone(), start.clone(), b.clone()], p, [blocks, 1, 1], 2 * n)?;
            std::mem::swap(&mut a, &mut b);
        }
    }
    let value_words = r.values.as_ref().map_or(0, |v| v.desc.len().unwrap() * v.desc.element.words() / n);
    let p = StepParams {
        n: n as u32,
        words: key_words,
        value_words: value_words as u32,
        ..Default::default()
    };
    let buffers = vec![
        Buf::View(r.keys.clone()),
        Buf::View(r.ids.clone()),
        view_or(&r.values, &r.keys),
        a,
        keys,
        ids,
        values,
    ];
    plan.push_1d(x, "mkl_c_sort_gather", buffers, p, n)?;
    finish(plan, x, ctl, stats, &outputs)
}

fn qr(r: &QrFit, x: &mut Execution<'_>, ctl: &JobControl, stats: &mut StageStats) -> MkResult<()> {
    let [b, m, n] = r.matrix.desc.shape[..] else {
        unreachable!()
    };
    let mut plan = Plan::default();
    let factors = plan.output(x, r.factors);
    let solution = plan.output(x, r.solution);
    let diag = plan.output(x, r.rank_condition);
    let outputs = [r.factors, r.solution, r.rank_condition];
    if b == 0 {
        return finish(plan, x, ctl, stats, &outputs);
    }
    let aux_stride = 5 * n + 2;
    let work = plan.scratch(x, 2 * b * (n + 1) * m)?;
    let qcol = plan.scratch(x, 2 * b * n * m)?;
    let aux = plan.scratch(x, 2 * b * aux_stride)?;
    let inverse = plan.scratch(x, 2 * b * n * n)?;
    let threads = spec::qr_threads(m);
    let p = QrParams {
        rows: m as u32,
        cols: n as u32,
        batch: b as u32,
        input_words: r.matrix.desc.element.words() as u32,
        tolerance_bits: r.rank_tolerance.to_bits(),
        aux_stride: aux_stride as u32,
        threads: threads as u32,
        step: 0,
    };
    let buffers = vec![
        Buf::View(r.matrix.clone()),
        Buf::View(r.rhs.clone()),
        work.clone(),
        factors.clone(),
        aux.clone(),
        diag.clone(),
    ];
    let cols = b * (n + 1) * m;
    plan.push_threads(x, "mkl_c_qr_prepare", buffers, p, [n + 1, b, 1], threads, cols)?;
    for k in 0..n {
        let p = QrParams { step: k as u32, ..p };
        let pivot = vec![work.clone(), factors.clone(), aux.clone(), diag.clone()];
        plan.push_threads(x, "mkl_c_qr_pivot", pivot, p, [b, 1, 1], threads, b * m)?;
        // One classical Gram-Schmidt pass of the (already projected) pivot
        // column against every earlier q restores orthogonality to rounding.
        if k > 0 {
            let dot = vec![work.clone(), aux.clone(), diag.clone(), qcol.clone()];
            plan.push_threads(x, "mkl_c_qr_reorth_dot", dot, p, [k, b, 1], threads, b * k * m)?;
            let update = vec![work.clone(), factors.clone(), aux.clone(), diag.clone(), qcol.clone()];
            plan.push(x, "mkl_c_qr_reorth_update", update, p, [m.div_ceil(256), b, 1], b * k * m)?;
        }
        let normalise = vec![work.clone(), factors.clone(), diag.clone(), qcol.clone()];
        plan.push_threads(x, "mkl_c_qr_normalise", normalise, p, [b, 1, 1], threads, b * m)?;
        let project = vec![work.clone(), factors.clone(), aux.clone(), diag.clone(), qcol.clone()];
        plan.push_threads(x, "mkl_c_qr_project", project, p, [n - k, b, 1], threads, b * (n - k) * m)?;
    }
    let buffers = vec![factors, aux, diag, solution, inverse];
    plan.push_threads(x, "mkl_c_qr_finish", buffers, p, [b, 1, 1], threads, b * n * n * n)?;
    finish(plan, x, ctl, stats, &outputs)
}
