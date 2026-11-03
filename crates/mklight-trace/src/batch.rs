//! Parallel batch traces over pupil samples × fields × wavelengths.
//!
//! Output ordering is deterministic and independent of thread scheduling:
//! sample `(f, w, p)` is at index `(f · W + w) · P + p` for `F` fields, `W`
//! wavelengths and `P` pupil points. The job variants check the
//! [`JobContext`]'s cancellation token between blocks of rays and report
//! progress in `[0, 1]`.

use std::ops::Range;

use mklight_core::{JobContext, MkResult, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::numeric::Compensated;
use crate::opd::OpdReference;
use crate::trace::{TraceOptions, TraceResult, TraceStatus};
use crate::tracer::{PreparedField, RayRequest, SequentialTracer};

/// Rays per cancellation / progress block in the job variants.
const BLOCK: usize = 8192;
/// Rays per parallel task: one [`SequentialTracer::trace_many`] call, whose
/// aims and traces run as packets.
pub(crate) const CHUNK: usize = 64;
/// OPD references (chief-ray aims and traces) per block in [`opd_grid_job`].
const REF_BLOCK: usize = 16;

/// One traced sample of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchRay {
    /// Index into the batch's field list.
    pub field: u32,
    /// System wavelength index.
    pub wavelength: u32,
    /// Requested (pre-vignetting) pupil coordinates.
    pub pupil: [f64; 2],
    pub status: TraceStatus,
    /// Final point (image local frame on success).
    pub point: Vec3,
    pub dir: Vec3,
    pub opl: Compensated,
    pub transmittance: f64,
}

/// A batch request: what to trace.
#[derive(Clone, Copy, Debug)]
pub struct Grid<'a> {
    pub fields: &'a [PreparedField],
    /// System wavelength indices.
    pub wavelengths: &'a [usize],
    pub pupil: &'a [[f64; 2]],
}

impl Grid<'_> {
    pub fn len(&self) -> usize {
        self.fields.len() * self.wavelengths.len() * self.pupil.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// `(field, wavelength-list position, pupil)` of flat index `i`.
    #[inline]
    pub fn split(&self, i: usize) -> (usize, usize, usize) {
        let p = self.pupil.len();
        let w = self.wavelengths.len();
        (i / (w * p), (i / p) % w, i % p)
    }
}

/// The requests of flat indices `range`.
fn requests<'a>(g: &Grid<'a>, range: Range<usize>) -> Vec<RayRequest<'a>> {
    range
        .map(|i| {
            let (f, wi, p) = g.split(i);
            (&g.fields[f], g.wavelengths[wi], g.pupil[p])
        })
        .collect()
}

/// Trace flat indices `range` together ([`SequentialTracer::trace_many`]).
fn traced(
    tracer: &SequentialTracer,
    g: &Grid,
    opts: &TraceOptions,
    range: Range<usize>,
) -> Vec<TraceResult> {
    tracer.trace_many(&requests(g, range), opts)
}

fn batch_rays(
    tracer: &SequentialTracer,
    g: &Grid,
    opts: &TraceOptions,
    range: Range<usize>,
) -> Vec<BatchRay> {
    let rs = traced(tracer, g, opts, range.clone());
    range
        .zip(rs)
        .map(|(i, r)| {
            let (f, wi, p) = g.split(i);
            BatchRay {
                field: f as u32,
                wavelength: g.wavelengths[wi] as u32,
                pupil: g.pupil[p],
                status: r.status,
                point: r.ray.pos,
                dir: r.ray.dir,
                opl: r.opl,
                transmittance: r.transmittance,
            }
        })
        .collect()
}

/// Trace a full grid in parallel.
pub fn trace_grid(tracer: &SequentialTracer, grid: &Grid, opts: &TraceOptions) -> Vec<BatchRay> {
    par_chunks(0..grid.len(), |r| batch_rays(tracer, grid, opts, r))
}

/// `f` over consecutive sub-ranges of [`CHUNK`] indices, in parallel; the
/// outputs are concatenated in index order.
pub(crate) fn par_chunks<T: Send>(
    range: Range<usize>,
    f: impl Fn(Range<usize>) -> Vec<T> + Sync,
) -> Vec<T> {
    let (start, end) = (range.start, range.end);
    let parts: Vec<Vec<T>> = (start..end)
        .step_by(CHUNK)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|a| f(a..(a + CHUNK).min(end)))
        .collect();
    let mut out = Vec::with_capacity(end - start);
    for mut p in parts {
        out.append(&mut p);
    }
    out
}

/// [`trace_grid`] with cancellation and progress.
pub fn trace_grid_job(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    ctx: &JobContext,
) -> MkResult<Vec<BatchRay>> {
    run_chunks(grid.len(), BLOCK, ctx, "tracing rays", |r| {
        batch_rays(tracer, grid, opts, r)
    })
}

/// Evaluate `f(0..total)` in parallel blocks of `block`, checking for
/// cancellation before each block and reporting progress after it (ending
/// at 1, also for an empty range).
pub(crate) fn run_blocks<T: Send>(
    total: usize,
    block: usize,
    ctx: &JobContext,
    stage: &str,
    f: impl Fn(usize) -> T + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    ctx.report(0.0, stage);
    while start < total {
        ctx.check()?;
        let end = (start + block).min(total);
        let mut block: Vec<T> = (start..end)
            .into_par_iter()
            .with_min_len(64)
            .map(&f)
            .collect();
        out.append(&mut block);
        start = end;
        ctx.report(start as f64 / total as f64, stage);
    }
    ctx.check()?;
    ctx.report(1.0, stage);
    Ok(out)
}

/// [`run_blocks`] where `f` maps a range of indices at a time (packet
/// traces); each block is split into [`CHUNK`]-index ranges in parallel.
pub(crate) fn run_chunks<T: Send>(
    total: usize,
    block: usize,
    ctx: &JobContext,
    stage: &str,
    f: impl Fn(Range<usize>) -> Vec<T> + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    ctx.report(0.0, stage);
    while start < total {
        ctx.check()?;
        let end = (start + block).min(total);
        out.append(&mut par_chunks(start..end, &f));
        start = end;
        ctx.report(start as f64 / total as f64, stage);
    }
    ctx.check()?;
    ctx.report(1.0, stage);
    Ok(out)
}

/// OPD (waves) over a grid, `None` where the ray or its chief ray failed.
/// Same ordering as [`trace_grid`].
pub fn opd_grid(tracer: &SequentialTracer, grid: &Grid, opts: &TraceOptions) -> Vec<Option<f64>> {
    let refs = references(tracer, grid);
    par_chunks(0..grid.len(), |r| opd_range(tracer, grid, opts, &refs, r))
}

/// [`opd_grid`] with cancellation and progress.
pub fn opd_grid_job(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    ctx: &JobContext,
) -> MkResult<Vec<Option<f64>>> {
    // Progress share of the references: each costs a chief-ray aim and
    // trace, weighted here as 8 rays.
    let n_refs = grid.fields.len() * grid.wavelengths.len();
    let rays = if grid.pupil.is_empty() { 0 } else { grid.len() };
    let split = if n_refs + rays == 0 {
        0.0
    } else {
        (8 * n_refs) as f64 / (8 * n_refs + rays) as f64
    };
    let refs = run_blocks(
        n_refs,
        REF_BLOCK,
        &ctx.subrange(0.0, split),
        "OPD references",
        |k| reference(tracer, grid, k),
    )?;
    run_chunks(
        grid.len(),
        BLOCK,
        &ctx.subrange(split, 1.0),
        "computing OPD",
        |r| opd_range(tracer, grid, opts, &refs, r),
    )
}

fn reference(tracer: &SequentialTracer, grid: &Grid, k: usize) -> Option<OpdReference> {
    let w = grid.wavelengths.len();
    tracer
        .opd_reference(&grid.fields[k / w], grid.wavelengths[k % w])
        .ok()
}

fn references(tracer: &SequentialTracer, grid: &Grid) -> Vec<Option<OpdReference>> {
    let w = grid.wavelengths.len();
    (0..grid.fields.len() * w)
        .into_par_iter()
        .map(|k| reference(tracer, grid, k))
        .collect()
}

fn opd_range(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    refs: &[Option<OpdReference>],
    range: Range<usize>,
) -> Vec<Option<f64>> {
    let w = grid.wavelengths.len();
    let reference = |i: usize| {
        let (f, wi, _) = grid.split(i);
        refs[f * w + wi].as_ref()
    };
    // Rays whose chief ray failed are not traced.
    let live: Vec<usize> = range.clone().filter(|&i| reference(i).is_some()).collect();
    let reqs: Vec<RayRequest> = live
        .iter()
        .map(|&i| {
            let (f, wi, p) = grid.split(i);
            (&grid.fields[f], grid.wavelengths[wi], grid.pupil[p])
        })
        .collect();
    let mut rs = live.iter().zip(tracer.trace_many(&reqs, opts));
    let mut next = rs.next();
    range
        .map(|i| match next {
            Some((&k, ref r)) if k == i => {
                let v = reference(i).and_then(|c| c.opd_waves(r));
                next = rs.next();
                v
            }
            _ => None,
        })
        .collect()
}
