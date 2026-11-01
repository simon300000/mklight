//! Zero-copy encoder for the optimisation algebra specification.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{optim_ops::*, pending::JobControl, resident::*};
use std::time::Instant;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let a = mklight_gpu::optim_ops::accesses(r);
    let mut p = params(r);
    let count = p.count as usize;
    let mut outputs: Vec<_> = a.writes.iter().map(|&id| x.take_output(id)).collect();
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_optim_algebra"))?;
    let mut stage = StageStats {
        name: "optim algebra".into(),
        ..Default::default()
    };
    // Bound the product's work, not just its output size, for cancellation.
    let tile = (4_194_304usize / (p.k as usize).max(1)).clamp(256, 1 << 20);
    for first in (0..count).step_by(tile) {
        ctl.check()?;
        p.offset = first as u32;
        p.count = (count - first).min(tile) as u32;
        let t = Instant::now();
        // SAFETY: the submission boundary validates every typed extent;
        // device stencil indices are checked before dereference.
        let cs = unsafe {
            x.gpu.run("optim algebra", |e| {
                for i in 0..4 {
                    let v = &a.reads[i.min(a.reads.len() - 1)];
                    let (b, o) = x.buffer(v);
                    e.set_buffer(i, b, o);
                }
                if outputs.len() == 2 {
                    let (out, extra) = outputs.split_at_mut(1);
                    e.set_buffer_mut(4, &mut out[0], 0);
                    e.set_buffer_mut(5, &mut extra[0], 0);
                } else {
                    e.set_buffer_mut(4, &mut outputs[0], 0);
                    let (b, o) = x.buffer(&a.reads[0]);
                    e.set_buffer(5, b, o);
                }
                e.set_bytes(6, &p);
                e.dispatch_1d(&pso, p.count as usize);
                Ok(())
            })?
        };
        stage.device_seconds += cs.gpu_seconds;
        stage.command_buffers += 1;
        stage.max_command_seconds = stage.max_command_seconds.max(t.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.check()?;
        ctl.report(
            (first + p.count as usize) as f64 / count as f64,
            "optim algebra",
        );
    }
    for (id, out) in a.writes.into_iter().zip(outputs) {
        x.put_output(id, out);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stage.device_seconds;
    x.stats.command_buffers += stage.command_buffers;
    x.stats.stages.push(stage);
    Ok(())
}
