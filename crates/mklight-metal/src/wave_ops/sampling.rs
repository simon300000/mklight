use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::sampling::*};
use std::time::Instant;
pub fn execute(r: &Blur, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    ctl.check()?;
    let p = params(r);
    let pipe = x.gpu.pipeline(&PipelineKey::plain("mkl_wave_blur"))?;
    let mut out = x.take_output(r.points);
    let mut mask = x.take_output(r.mask);
    let start = Instant::now();
    let xr = &*x;
    // SAFETY: validated grids and exclusive output/mask leases.
    unsafe {
        xr.run("wave sampling", |e| {
            for (i, v) in [&r.pupil.amplitude, &r.pupil.coverage, &r.pupil.opd]
                .iter()
                .enumerate()
            {
                let (b, o) = xr.buffer(v);
                e.set_buffer(i, b, o);
            }
            e.set_buffer_mut(3, &mut out, 0);
            e.set_buffer_mut(4, &mut mask, 0);
            e.set_bytes(5, &p);
            e.dispatch_1d(&pipe, p.count as usize);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(r.points, out);
    x.put_output(r.mask, mask);
    x.stats.parameter_bytes += size_of::<Params>() as u64;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "wave_sampling".into(),
        command_buffers: 0,
        device_seconds: 0.,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    Ok(())
}
