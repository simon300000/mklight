//! Parallel batch traces over pupil samples × fields × wavelengths.
//!
//! Output ordering is deterministic and independent of thread scheduling:
//! sample `(f, w, p)` is at index `(f · W + w) · P + p` for `F` fields, `W`
//! wavelengths and `P` pupil points. The job variants check the
//! [`JobContext`]'s cancellation token between blocks of rays and report
//! progress in `[0, 1]`.

use mklight_core::{JobContext, MkResult, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::numeric::Compensated;
use crate::opd::OpdReference;
use crate::trace::{TraceOptions, TraceResult, TraceStatus};
use crate::tracer::{PreparedField, SequentialTracer};

/// Rays per cancellation / progress block in the job variants.
const BLOCK: usize = 8192;

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

fn one(tracer: &SequentialTracer, g: &Grid, opts: &TraceOptions, i: usize) -> BatchRay {
    let (f, wi, p) = g.split(i);
    let wl = g.wavelengths[wi];
    let pp = g.pupil[p];
    let r: TraceResult = tracer.trace(&g.fields[f], wl, pp[0], pp[1], opts);
    BatchRay {
        field: f as u32,
        wavelength: wl as u32,
        pupil: pp,
        status: r.status,
        point: r.ray.pos,
        dir: r.ray.dir,
        opl: r.opl,
        transmittance: r.transmittance,
    }
}

/// Trace a full grid in parallel.
pub fn trace_grid(tracer: &SequentialTracer, grid: &Grid, opts: &TraceOptions) -> Vec<BatchRay> {
    (0..grid.len())
        .into_par_iter()
        .with_min_len(64)
        .map(|i| one(tracer, grid, opts, i))
        .collect()
}

/// [`trace_grid`] with cancellation and progress.
pub fn trace_grid_job(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    ctx: &JobContext,
) -> MkResult<Vec<BatchRay>> {
    run_blocks(grid.len(), ctx, "tracing rays", |i| {
        one(tracer, grid, opts, i)
    })
}

fn run_blocks<T: Send>(
    total: usize,
    ctx: &JobContext,
    stage: &str,
    f: impl Fn(usize) -> T + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    ctx.report(0.0, stage);
    while start < total {
        ctx.check()?;
        let end = (start + BLOCK).min(total);
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
    Ok(out)
}

/// OPD (waves) over a grid, `None` where the ray or its chief ray failed.
/// Same ordering as [`trace_grid`].
pub fn opd_grid(tracer: &SequentialTracer, grid: &Grid, opts: &TraceOptions) -> Vec<Option<f64>> {
    let refs = references(tracer, grid);
    (0..grid.len())
        .into_par_iter()
        .with_min_len(64)
        .map(|i| opd_one(tracer, grid, opts, &refs, i))
        .collect()
}

/// [`opd_grid`] with cancellation and progress.
pub fn opd_grid_job(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    ctx: &JobContext,
) -> MkResult<Vec<Option<f64>>> {
    ctx.check()?;
    let refs = references(tracer, grid);
    run_blocks(grid.len(), ctx, "computing OPD", |i| {
        opd_one(tracer, grid, opts, &refs, i)
    })
}

fn references(tracer: &SequentialTracer, grid: &Grid) -> Vec<Option<OpdReference>> {
    let w = grid.wavelengths.len();
    (0..grid.fields.len() * w)
        .into_par_iter()
        .map(|k| {
            tracer
                .opd_reference(&grid.fields[k / w], grid.wavelengths[k % w])
                .ok()
        })
        .collect()
}

fn opd_one(
    tracer: &SequentialTracer,
    grid: &Grid,
    opts: &TraceOptions,
    refs: &[Option<OpdReference>],
    i: usize,
) -> Option<f64> {
    let (f, wi, p) = grid.split(i);
    let r = refs[f * grid.wavelengths.len() + wi].as_ref()?;
    let pp = grid.pupil[p];
    r.opd_waves(&tracer.trace(&grid.fields[f], grid.wavelengths[wi], pp[0], pp[1], opts))
}
