use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::diagnostics::Diagnostics};
use std::time::Instant;
pub fn execute(r: &Diagnostics, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    ctl.check()?;
    let n = r.coverage.desc.shape[0];
    let blocks = (n * n).div_ceil(256);
    let mut partial = x.allocate(blocks * 10)?;
    let mut out = x.take_output(r.output);
    let a = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_diagnostics"))?;
    let b = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_diagnostics_finish"))?;
    let start = Instant::now();
    let xr = &*x;
    // SAFETY: validated contiguous grids and exclusive block/output storage.
    unsafe {
        xr.run("wave diagnostics", |e| {
            for (i, v) in [&r.coverage, &r.opd, &r.extrapolation].iter().enumerate() {
                let (buf, off) = xr.buffer(v);
                e.set_buffer(i, buf, off);
            }
            e.set_buffer_mut(3, &mut partial, 0);
            e.set_buffer_mut(4, &mut out, 0);
            e.set_bytes(5, &[n as u32, blocks as u32]);
            e.dispatch_1d(&a, blocks);
            e.dispatch_1d(&b, 1);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(r.output, out);
    x.recycle(partial);
    x.stats.parameter_bytes += 8;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "wave_diagnostics".into(),
        command_buffers: 0,
        device_seconds: 0.,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    Ok(())
}
