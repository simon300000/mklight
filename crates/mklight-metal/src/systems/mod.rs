//! Bounded resident system preparation. No host reads of intermediate arrays.
use crate::context::PipelineKey;
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, systems::Request};
use std::time::Instant;

pub(crate) fn execute(
    request: &Request,
    x: &mut crate::resident::Execution<'_>,
    ctl: &JobControl,
) -> MkResult<()> {
    let Request::Prepare(r) = request;
    let pipeline = x.gpu.pipeline(&PipelineKey::plain("mkl_systems_prepare"))?;
    let mut scratch = x.allocate(r.scratch_words())?;
    let mut outputs: Vec<_> = r
        .outputs()
        .into_iter()
        .map(|id| (id, x.take_output(id)))
        .collect();
    let inputs = [
        &r.parameters,
        &r.slots,
        &r.bounds,
        &r.dependencies,
        &r.material_tables,
    ]
    .map(|v| x.buffer(v));
    let mut stage = StageStats {
        name: "system preparation".into(),
        ..Default::default()
    };
    for first in (0..r.map.count).step_by(r.tile_candidates) {
        ctl.check()?;
        let count = r.tile_candidates.min(r.map.count - first);
        let params = r.dispatch_params(first, count);
        let wall = Instant::now();
        // SAFETY: the private host compiler owns the immutable instruction and
        // patch leases. Submission verifies their identity and every output
        // extent. Scratch has one disjoint instruction plane per tile lane.
        let cs = unsafe {
            x.gpu.run("prepare systems", |e| {
                e.set_pipeline(&pipeline);
                for (i, (b, off)) in inputs.iter().enumerate() {
                    e.set_buffer(i, *b, *off);
                }
                e.set_buffer_mut(5, &mut scratch, 0);
                for (i, (_, out)) in outputs.iter_mut().enumerate() {
                    e.set_buffer_mut(6 + i, out, 0);
                }
                e.set_bytes(14, &params);
                e.dispatch_1d(&pipeline, count);
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.check()?;
        ctl.report(
            (first + count) as f64 / r.map.count as f64,
            "prepare systems",
        );
    }
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.parameter_bytes +=
        stage.command_buffers as u64 * size_of::<mklight_gpu::systems::DispatchParams>() as u64;
    // Package timings are retained separately; the router also records the
    // outer domain wall time without counting our commands a second time.
    x.stats.stages.push(stage);
    for (id, out) in outputs {
        x.put_output(id, out);
    }
    x.recycle(scratch);
    Ok(())
}
