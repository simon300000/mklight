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
        Pupil, Request,
        spec::{Params, params},
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
    x.stats.command_buffers += s.command_buffers;
    x.stats.device_seconds += s.device_seconds;
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
        let cs = unsafe {
            xr.gpu.run("focus phase cache", |e| {
                bind(xr, e, r);
                e.set_buffer_mut(4, &mut out, 0);
                e.set_bytes(7, &p);
                e.dispatch_1d(&pipe, p.count as usize);
                Ok(())
            })?
        };
        s.command_buffers += 1;
        s.device_seconds += cs.gpu_seconds;
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
        let cs = unsafe {
            xr.gpu.run("focus sampling", |e| {
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
        s.command_buffers += 1;
        s.device_seconds += cs.gpu_seconds;
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
