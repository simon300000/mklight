//! Scalar-wave dispatches use bounded commands and retain intermediates in shared storage.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::StageStats,
    wave_ops::{spec::*, *},
};
use std::time::Instant;
mod diagnostics;
mod fft_mtf;
mod focus_cache;
mod image_integrals;
mod lattice;
mod placement;
mod resample;
mod shells;
mod simulation;
mod spectral;
mod statistics;
/// Work units (threads × relative per-thread cost) per command. Smaller
/// operations run as one command; larger ones keep bounded commands for
/// cancellation latency.
const COMMAND_UNITS: usize = 1 << 22;
/// Ordered dispatches grouped into commands of at most [`COMMAND_UNITS`]
/// (a single larger dispatch gets its own command).
fn commands<T>(dispatches: impl IntoIterator<Item = (T, usize)>) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = vec![];
    let mut units = usize::MAX;
    for (d, u) in dispatches {
        if units.saturating_add(u) > COMMAND_UNITS {
            out.push(vec![]);
            units = 0;
        }
        units += u;
        out.last_mut().unwrap().push(d);
    }
    out
}
/// Threads of one blocked Huygens/correlation dispatch (outputs × sample blocks).
const BLOCK_THREADS: usize = 1 << 16;
/// Pixel-sample (or frequency-sample) terms per blocked command.
const COMMAND_TERMS: usize = 1 << 24;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if let Request::Resample(r) = r {
        return resample::execute(r, x, ctl);
    }
    if let Request::Diagnostics(r) = r {
        return diagnostics::execute(r, x, ctl);
    }
    if let Request::Statistics(r) = r {
        return statistics::execute(r, x, ctl);
    }
    if let Request::Lattice(r) = r {
        return lattice::execute(r, x, ctl);
    }
    if let Request::Placement(r) = r {
        return placement::execute(r, x, ctl);
    }
    if let Request::Spectral(r) = r {
        return spectral::execute(r, x, ctl);
    }
    if let Request::ImageIntegral(r) = r {
        return image_integrals::execute(r, x, ctl);
    }
    if let Request::FftMtf(r) = r {
        return fft_mtf::execute(r, x, ctl);
    }
    if let Request::Simulation(r) = r {
        return simulation::execute(r, x, ctl);
    }
    if let Request::Shells(r) = r {
        return shells::execute(r, x, ctl);
    }
    if focus_cache::chunked(r, x, ctl)? {
        return Ok(());
    }
    let cache = focus_cache::prepare(r, x, ctl)?;
    let access = accesses(r);
    let mut p = params(r);
    p.reserved[0] = u32::from(cache.is_some());
    let mut output = x.take_output(access.writes[0]);
    let mut moments = if access.writes.len() > 1 {
        Some(x.take_output(access.writes[1]))
    } else {
        None
    };
    if matches!(r, Request::Prepare { .. }) {
        p.field_side = (output.len() / 2).isqrt() as u32;
    }
    let total = match p.mode {
        0 => (p.field_side * p.field_side) as usize,
        1 | 2 => (p.size * p.size) as usize,
        3 => (p.size * p.n * p.n) as usize,
        4 => p.size as usize,
        6 => (p.size * p.field_side) as usize,
        7 => p.n as usize,
        _ => 2 * p.n as usize,
    };
    let kernel = [
        "mkl_wave_prepare",
        "mkl_wave_intensity",
        "mkl_wave_huygens",
        "mkl_wave_focus",
        "mkl_wave_detector",
        "mkl_wave_lsf",
        "mkl_wave_focus_correlation",
        "mkl_wave_zernike",
    ][p.mode as usize];
    let pipeline = x.gpu.pipeline(&PipelineKey::plain(kernel))?;
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        name: "wave".into(),
        ..Default::default()
    };
    // Huygens and correlation outputs sum all n² pupil samples: each thread
    // sums one fixed block, a finish dispatch adds the blocks in order.
    let n2 = p.n as usize * p.n as usize;
    let blocked = matches!(p.mode, 2 | 6);
    let blocks = n2.div_ceil(spec::SUM_BLOCK);
    let tile = if blocked {
        (BLOCK_THREADS / blocks).clamp(1, total)
    } else {
        1 << 18
    };
    let mut acc = if blocked {
        Some(x.allocate(tile * blocks * if p.mode == 2 { 4 } else { 6 })?)
    } else {
        None
    };
    let finish = if blocked {
        Some(x.gpu.pipeline(&PipelineKey::plain(if p.mode == 2 {
            "mkl_wave_huygens_finish"
        } else {
            "mkl_wave_focus_correlation_finish"
        }))?)
    } else {
        None
    };
    let mut dispatches = vec![];
    for first in (0..total).step_by(tile) {
        let mut q = p;
        q.first = first as u32;
        q.count = (total - first).min(tile) as u32;
        let work = q.count as usize
            * match p.mode {
                2 | 6 => n2 * COMMAND_UNITS / COMMAND_TERMS,
                5 => p.n as usize,
                _ => 1,
            };
        dispatches.push((q, work));
    }
    let groups = commands(dispatches);
    for (done, group) in groups.iter().enumerate() {
        ctl.check()?;
        let wall = Instant::now();
        let xr = &*x;
        // SAFETY: request validation bounds each typed view; output leases are
        // exclusive. Dispatches serialize scratch reuse, including block sums.
        unsafe {
            xr.run("wave", |e| {
                let bind = |index, at| {
                    let (b, o) = xr.buffer(&access.reads[at]);
                    e.set_buffer(index, b, o);
                };
                match p.mode {
                    7 => {
                        bind(0, 0);
                        bind(1, 1);
                    }
                    0 | 2 | 3 | 6 => {
                        bind(0, 0);
                        bind(1, 1);
                        bind(2, 2);
                        if p.mode == 6 {
                            bind(8, 3);
                            if let Some(c) = &cache {
                                e.set_buffer(10, c, 0);
                            } else {
                                bind(10, 2);
                            }
                            bind(9, if access.reads.len() > 4 { 4 } else { 2 });
                        } else if p.mode != 0 {
                            bind(3, 3);
                        }
                    }
                    1 => {
                        bind(0, 0);
                        bind(3, 1);
                    }
                    _ => bind(0, 0),
                }
                e.set_buffer_mut(4, &mut output, 0);
                if let Some(m) = moments.as_mut() {
                    e.set_buffer_mut(5, m, 0);
                }
                if let Some(a) = acc.as_mut() {
                    e.set_buffer_mut(6, a, 0);
                }
                for q in group {
                    if let Some(finish) = &finish {
                        let mut b = *q;
                        b.sample_first = blocks as u32;
                        b.sample_count = spec::SUM_BLOCK as u32;
                        b.count = q.count * blocks as u32;
                        e.set_bytes(7, &b);
                        e.dispatch_1d(&pipeline, b.count as usize);
                        b.count = q.count;
                        e.set_bytes(7, &b);
                        e.dispatch_1d(finish, q.count as usize);
                    } else {
                        e.set_bytes(7, q);
                        e.dispatch_1d(&pipeline, q.count as usize);
                    }
                }
                Ok(())
            })?
        };
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes +=
            (group.len() * size_of::<Params>() * if blocked { 2 } else { 1 }) as u64;
        let begin = if cache.is_some() { 0.08 } else { 0. };
        let end = if moments.is_some() && matches!(p.mode, 3 | 6) {
            0.95
        } else {
            1.
        };
        ctl.report(
            begin + (end - begin) * (done + 1) as f64 / groups.len() as f64,
            "wave",
        );
    }
    ctl.check()?;
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(stage);
    if matches!(p.mode, 3 | 6)
        && let Some(m) = moments.as_mut()
    {
        focus_cache::steps(
            r,
            x,
            ctl,
            if p.mode == 3 {
                Some(&*output)
            } else {
                cache.as_deref()
            },
            m,
        )?;
    }
    if let Some(c) = cache {
        x.recycle(c);
    }
    x.put_output(access.writes[0], output);
    if let Some(m) = moments {
        x.put_output(access.writes[1], m);
    }
    if let Some(a) = acc {
        x.recycle(a);
    }
    Ok(())
}
