//! Scalar-wave dispatches use bounded commands and retain intermediates in shared storage.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::StageStats,
    wave_ops::{spec::*, *},
};
use std::time::Instant;
mod resample;
const PIXELS: usize = 256;
const SAMPLES: usize = 256;
const COMMAND_TERMS: usize = 1 << 20;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if let Request::Resample(r) = r {
        return resample::execute(r, x, ctl);
    }
    let access = accesses(r);
    let mut p = params(r);
    let mut output = x.take_output(access.writes[0]);
    let mut moments = if access.writes.len() > 1 {
        Some(x.take_output(access.writes[1]))
    } else {
        None
    };
    let mut acc = if matches!(
        r,
        Request::Huygens { .. } | Request::FocusCorrelation { .. }
    ) {
        Some(x.allocate(PIXELS * 6)?)
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
    let (mut first, mut sample) = (0, 0);
    let n2 = (p.n * p.n) as usize;
    while first < total {
        ctl.check()?;
        let mut dispatches = Vec::new();
        let mut work = 0;
        while first < total && work < COMMAND_TERMS {
            let count = (total - first).min(if matches!(p.mode, 2 | 6) {
                PIXELS
            } else {
                1 << 18
            });
            p.first = first as u32;
            p.count = count as u32;
            if matches!(p.mode, 2 | 6) {
                p.sample_first = sample as u32;
                p.sample_count = (n2 - sample).min(SAMPLES) as u32;
                sample += p.sample_count as usize;
                work += count * p.sample_count as usize;
                if sample == n2 {
                    sample = 0;
                    first += count;
                }
            } else {
                first += count;
                work += count * (if p.mode == 5 { p.n as usize } else { 1 });
            }
            dispatches.push(p);
        }
        let wall = Instant::now();
        let xr = &*x;
        // SAFETY: request validation bounds each typed view; output leases are
        // exclusive. Dispatches serialize scratch reuse, including sample tiles.
        let cs = unsafe {
            xr.gpu.run("wave", |e| {
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
                for q in &dispatches {
                    e.set_bytes(7, q);
                    e.dispatch_1d(&pipeline, q.count as usize);
                }
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += (dispatches.len() * size_of::<Params>()) as u64;
        ctl.report(
            (first as f64 + sample as f64 / n2.max(1) as f64 * p.count as f64) / total as f64,
            "wave",
        );
    }
    ctl.check()?;
    x.put_output(access.writes[0], output);
    if let Some(m) = moments {
        x.put_output(access.writes[1], m);
    }
    if let Some(a) = acc {
        x.recycle(a);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.stages.push(stage);
    Ok(())
}
