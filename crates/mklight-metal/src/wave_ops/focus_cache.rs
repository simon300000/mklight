//! Reuse defocus phases across frequencies; keep the on-the-fly path when the
//! optional cache would exceed 256 MiB. Sampling summaries are independent of it.
use crate::{
    buffer::SharedBuffer,
    context::PipelineKey,
    resident::{Execution, Lease},
};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::{PlanView, StageStats},
    wave_ops::{
        Pupil, Request, accesses,
        spec::{Params, SUM_BLOCK, params},
    },
};
use std::time::Instant;
fn inputs(r: &Request) -> (&Pupil, Option<&PlanView>) {
    match r {
        Request::Focus { pupil, shifts, .. } => (pupil, Some(shifts)),
        Request::FocusCorrelation { pupil, shifts, .. } => (pupil, shifts.as_ref()),
        _ => unreachable!(),
    }
}
fn stage(x: &Execution<'_>, name: &str) -> StageStats {
    StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        name: name.into(),
        ..Default::default()
    }
}
fn record(x: &mut Execution<'_>, mut s: StageStats, start: Instant) {
    s.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(s);
}
fn bind(x: &Execution<'_>, e: &crate::context::Encoder<'_>, r: &Request) {
    let (v, z) = inputs(r);
    let (a, o) = x.buffer(&v.coverage);
    e.set_buffer(1, a, o);
    let (a, o) = x.buffer(&v.opd);
    e.set_buffer(2, a, o);
    let (a, o) = x.buffer(z.unwrap_or(&v.opd));
    e.set_buffer(9, a, o);
    e.set_buffer(3, a, o);
}
/// Phase-cache limit (bytes of double-float phases).
const CACHE_BYTES: usize = 256 * 1024 * 1024;
/// The focus shifts view from plane `first` (double-float: two words each).
fn shifted<'a>(
    x: &'a Execution<'_>,
    shifts: &PlanView,
    first: usize,
) -> (&'a SharedBuffer<u32>, usize) {
    let (b, o) = x.buffer(shifts);
    (b, o + 2 * first)
}
/// A focus sweep whose whole phase volume exceeds [`CACHE_BYTES`] runs in
/// plane chunks, each with its own cache of the chunk's phases: the cache
/// holds exactly the phases the on-the-fly path evaluates (five per
/// correlation sample and plane), so results are identical while each
/// phase is evaluated once per plane. Returns false (nothing done) when the
/// volume fits the single cache of [`prepare`], or one plane alone exceeds
/// the limit (the on-the-fly path).
pub fn chunked(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<bool> {
    let Request::FocusCorrelation {
        shifts: Some(shifts),
        ..
    } = r
    else {
        return Ok(false);
    };
    let base = params(r);
    let n2 = base.n as usize * base.n as usize;
    let planes = base.size as usize;
    let per = CACHE_BYTES / (n2 * 8);
    if n2 * planes * 8 <= CACHE_BYTES || per == 0 {
        return Ok(false);
    }
    let access = accesses(r);
    let freqs = base.field_side as usize;
    let blocks = n2.div_ceil(SUM_BLOCK);
    let tile = (super::BLOCK_THREADS / blocks).clamp(1, per * freqs);
    let mut output = x.take_output(access.writes[0]);
    let mut steps = access.writes.get(1).map(|&w| x.take_output(w));
    let mut cache = x.allocate(per * n2 * 2)?;
    let mut acc = x.allocate(tile * blocks * 6)?;
    let step_blocks = n2.div_ceil(256);
    let mut partial = x.allocate(per * step_blocks)?;
    let pipe = |name| x.gpu.pipeline(&PipelineKey::plain(name));
    let (focus, correlation, finish, step, step_finish) = (
        pipe("mkl_wave_focus")?,
        pipe("mkl_wave_focus_correlation")?,
        pipe("mkl_wave_focus_correlation_finish")?,
        pipe("mkl_wave_focus_step_blocks")?,
        pipe("mkl_wave_focus_step_finish")?,
    );
    let start = Instant::now();
    let mut s = stage(x, "focus_planes");
    let (amplitude, coverage, opd, stencil) = (
        &access.reads[0],
        &access.reads[1],
        &access.reads[2],
        &access.reads[3],
    );
    for z0 in (0..planes).step_by(per) {
        let nz = per.min(planes - z0);
        // 1. The chunk's phases, contiguous from its first plane.
        let count = nz * n2;
        for first in (0..count).step_by(1 << 20) {
            ctl.check()?;
            let mut p = base;
            p.first = first as u32;
            p.count = (count - first).min(1 << 20) as u32;
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: the chunk cache holds `per` planes of n² phases; tiles
            // are disjoint and the shifts view starts at the chunk's plane.
            unsafe {
                xr.run("focus phase chunk", |e| {
                    let (b, o) = xr.buffer(coverage);
                    e.set_buffer(1, b, o);
                    let (b, o) = xr.buffer(opd);
                    e.set_buffer(2, b, o);
                    let (b, o) = shifted(xr, shifts, z0);
                    e.set_buffer(3, b, o);
                    e.set_buffer_mut(4, &mut cache, 0);
                    e.set_bytes(7, &p);
                    e.dispatch_1d(&focus, p.count as usize);
                    Ok(())
                })?
            };
            s.max_command_seconds = s.max_command_seconds.max(wall.elapsed().as_secs_f64());
        }
        // 2. Correlations of the chunk's planes: outputs z0·F onward.
        let total = nz * freqs;
        let mut dispatches = vec![];
        for first in (0..total).step_by(tile) {
            let mut q = base;
            q.reserved[0] = 1;
            q.first = first as u32;
            q.count = (total - first).min(tile) as u32;
            dispatches.push((
                q,
                q.count as usize * n2 * super::COMMAND_UNITS / super::COMMAND_TERMS,
            ));
        }
        for group in super::commands(dispatches) {
            ctl.check()?;
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: validated views; the output binding starts at the
            // chunk's first output (complex: two words), the block sums are
            // exclusive scratch and dispatches serialize their reuse.
            unsafe {
                xr.run("focus correlation chunk", |e| {
                    for (index, v) in [(0, amplitude), (1, coverage), (2, opd), (8, stencil)] {
                        let (b, o) = xr.buffer(v);
                        e.set_buffer(index, b, o);
                    }
                    e.set_buffer(10, &cache, 0);
                    let (b, o) = shifted(xr, shifts, z0);
                    e.set_buffer(9, b, o);
                    e.set_buffer_mut(4, &mut output, 2 * z0 * freqs);
                    e.set_buffer_mut(6, &mut acc, 0);
                    for q in &group {
                        let mut b = *q;
                        b.sample_first = blocks as u32;
                        b.sample_count = SUM_BLOCK as u32;
                        b.count = q.count * blocks as u32;
                        e.set_bytes(7, &b);
                        e.dispatch_1d(&correlation, b.count as usize);
                        b.count = q.count;
                        e.set_bytes(7, &b);
                        e.dispatch_1d(&finish, q.count as usize);
                    }
                    Ok(())
                })?
            };
            s.max_command_seconds = s.max_command_seconds.max(wall.elapsed().as_secs_f64());
            x.stats.parameter_bytes += (2 * group.len() * size_of::<Params>()) as u64;
        }
        // 3. Phase steps of the chunk's planes from the same cache.
        if let Some(out) = steps.as_mut() {
            let mut p = base;
            p.reserved[0] = 1;
            p.first = 0;
            p.count = (nz * step_blocks) as u32;
            let mut f = p;
            f.size = nz as u32;
            let xr = &*x;
            // SAFETY: per-chunk block maxima, then the chunk's planes of the
            // step output (one word each).
            unsafe {
                xr.run("focus sampling chunk", |e| {
                    let (b, o) = xr.buffer(coverage);
                    e.set_buffer(1, b, o);
                    let (b, o) = xr.buffer(opd);
                    e.set_buffer(2, b, o);
                    let (b, o) = shifted(xr, shifts, z0);
                    e.set_buffer(9, b, o);
                    e.set_buffer(10, &cache, 0);
                    e.set_buffer_mut(11, &mut partial, 0);
                    e.set_buffer_mut(12, out, z0);
                    e.set_bytes(7, &p);
                    e.dispatch_1d(&step, p.count as usize);
                    e.set_bytes(7, &f);
                    e.dispatch_1d(&step_finish, nz);
                    Ok(())
                })?
            };
        }
        ctl.report((z0 + nz) as f64 / planes as f64, "focus planes");
    }
    ctl.check()?;
    x.recycle(cache);
    x.recycle(acc);
    x.recycle(partial);
    x.put_output(access.writes[0], output);
    if let (Some(&w), Some(out)) = (access.writes.get(1), steps) {
        x.put_output(w, out);
    }
    record(x, s, start);
    Ok(true)
}
pub fn prepare(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<Option<Lease>> {
    let Request::FocusCorrelation {
        shifts: Some(_), ..
    } = r
    else {
        return Ok(None);
    };
    let mut p = params(r);
    let count = p.n as usize * p.n as usize * p.size as usize;
    if count * 8 > 256 * 1024 * 1024 {
        return Ok(None);
    }
    let mut out = x.allocate(count * 2)?;
    let pipe = x.gpu.pipeline(&PipelineKey::plain("mkl_wave_focus"))?;
    let start = Instant::now();
    let mut s = stage(x, "focus_phase_cache");
    for first in (0..count).step_by(1 << 20) {
        ctl.check()?;
        p.first = first as u32;
        p.count = (count - first).min(1 << 20) as u32;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: one complete contiguous phase volume, with disjoint tiles.
        unsafe {
            xr.run("focus phase cache", |e| {
                bind(xr, e, r);
                e.set_buffer_mut(4, &mut out, 0);
                e.set_bytes(7, &p);
                e.dispatch_1d(&pipe, p.count as usize);
                Ok(())
            })?
        };
        s.max_command_seconds = s.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.report(
            0.08 * (first + p.count as usize) as f64 / count as f64,
            "focus phases",
        );
    }
    ctl.check()?;
    record(x, s, start);
    Ok(Some(out))
}
pub fn steps(
    r: &Request,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    cache: Option<&SharedBuffer<u32>>,
    output: &mut SharedBuffer<u32>,
) -> MkResult<()> {
    let mut p = params(r);
    p.reserved[0] = u32::from(cache.is_some());
    p.unit_energy = u32::from(inputs(r).1.is_some());
    let blocks = (p.n as usize * p.n as usize).div_ceil(256);
    let count = blocks * p.size as usize;
    let mut partial = x.allocate(count)?;
    let a = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_focus_step_blocks"))?;
    let b = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_focus_step_finish"))?;
    let start = Instant::now();
    let mut s = stage(x, "focus_sampling");
    let tile = if cache.is_some() { 16384 } else { 1024 };
    for first in (0..count).step_by(tile) {
        ctl.check()?;
        p.first = first as u32;
        p.count = (count - first).min(tile) as u32;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: block rows are disjoint and complete before the final maxima.
        unsafe {
            xr.run("focus sampling", |e| {
                bind(xr, e, r);
                if let Some(c) = cache {
                    e.set_buffer(10, c, 0);
                } else {
                    let (c, o) = xr.buffer(&inputs(r).0.opd);
                    e.set_buffer(10, c, o);
                }
                e.set_buffer_mut(11, &mut partial, 0);
                e.set_buffer_mut(12, output, 0);
                e.set_bytes(7, &p);
                e.dispatch_1d(&a, p.count as usize);
                if first + p.count as usize == count {
                    e.dispatch_1d(&b, p.size as usize);
                }
                Ok(())
            })?
        };
        s.max_command_seconds = s.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.report(
            0.95 + 0.05 * (first + p.count as usize) as f64 / count as f64,
            "focus sampling",
        );
    }
    ctl.check()?;
    x.recycle(partial);
    record(x, s, start);
    Ok(())
}
