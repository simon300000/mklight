use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::StageStats,
    wave_ops::{spec::pair, statistics::*},
};
use std::time::Instant;
pub fn execute(r: &Statistics, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let access = access(r);
    let prepare = matches!(r, Statistics::Prepare { .. });
    let (n, sin) = match r {
        Statistics::Prepare {
            pupil, geometry, ..
        } => (pupil.amplitude.desc.shape[0], pair(geometry.sin_max)),
        Statistics::Residual { samples, .. } => (samples.desc.shape[0].isqrt(), [0.; 2]),
    };
    let blocks = (n * n).div_ceil(256);
    let mut partial = x.allocate(blocks * 12)?;
    let mut out = x.take_output(access.writes[0]);
    let mut sums = if prepare {
        Some(x.take_output(access.writes[1]))
    } else {
        None
    };
    let names = if prepare {
        ["mkl_wave_statistics_prepare", "mkl_wave_statistics_sum"]
    } else {
        ["mkl_wave_statistics_residual", "mkl_wave_statistics_finish"]
    };
    let a = x.gpu.pipeline(&PipelineKey::plain(names[0]))?;
    let b = x.gpu.pipeline(&PipelineKey::plain(names[1]))?;
    let start = Instant::now();
    ctl.check()?;
    let xr = &*x;
    // SAFETY: validation fixes all row extents; block scratch and outputs are exclusive.
    let cs = unsafe {
        xr.run("wave statistics", |e| {
            for (i, v) in access.reads.iter().enumerate() {
                let (buf, off) = xr.buffer(v);
                e.set_buffer(i, buf, off);
            }
            e.set_buffer_mut(3, &mut partial, 0);
            e.set_buffer_mut(4, &mut out, 0);
            if let Some(s) = sums.as_mut() {
                e.set_buffer_mut(5, s, 0);
            }
            e.set_bytes(
                6,
                &[n as u32, blocks as u32, sin[0].to_bits(), sin[1].to_bits()],
            );
            e.dispatch_1d(&a, blocks);
            e.dispatch_1d(&b, 1);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(access.writes[0], out);
    if let Some(s) = sums {
        x.put_output(access.writes[1], s);
    }
    x.recycle(partial);
    x.stats.command_buffers += cs.command_buffers;
    x.stats.device_seconds += cs.gpu_seconds;
    x.stats.parameter_bytes += 16;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "wave_statistics".into(),
        command_buffers: cs.command_buffers,
        device_seconds: cs.gpu_seconds,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    Ok(())
}
