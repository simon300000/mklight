use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::spectral::*};
use std::time::Instant;
pub fn execute(r: &Spectral, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    ctl.check()?;
    let p = params(r);
    let pipe = x.gpu.pipeline(&PipelineKey::plain("mkl_wave_spectral"))?;
    let mut out = x.take_output(r.output);
    let start = Instant::now();
    let xr = &*x;
    // SAFETY: matching typed inputs and exclusively leased output; unused static
    // bindings alias the first immutable input and are never loaded.
    let cs = unsafe {
        xr.gpu.run("spectral accumulation", |e| {
            for i in 0..16 {
                let (b, o) = xr.buffer(&r.inputs[i.min(r.inputs.len() - 1)]);
                e.set_buffer(i, b, o);
            }
            let (b, o) = xr.buffer(r.frequencies.as_ref().unwrap_or(&r.inputs[0]));
            e.set_buffer(17, b, o);
            e.set_buffer_mut(16, &mut out, 0);
            e.set_bytes(18, &p);
            e.dispatch_1d(&pipe, p.count as usize);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(r.output, out);
    x.stats.command_buffers += 1;
    x.stats.device_seconds += cs.gpu_seconds;
    x.stats.parameter_bytes += size_of::<Params>() as u64;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "spectral".into(),
        command_buffers: 1,
        device_seconds: cs.gpu_seconds,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
        ..Default::default()
    });
    Ok(())
}
