use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::StageStats,
    wave_ops::{image_integrals::*, spec},
};
use std::time::Instant;
pub fn execute(r: &ImageIntegral, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if let ImageIntegral::Metrics { input, output } = r {
        return metrics(input, *output, x, ctl);
    }
    let access = access(r);
    let p = params(r);
    let query = p.kind < 3;
    let pipe = x.gpu.pipeline(&PipelineKey::plain(if query {
        "mkl_wave_image_query"
    } else {
        "mkl_wave_image_complex"
    }))?;
    let finish = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_image_query_finish"))?;
    let mut out = x.take_output(access.writes[0]);
    let pixels = (p.n * p.n) as usize;
    // Queries sum all pixels: each thread sums one fixed block, a finish
    // dispatch adds the blocks in order.
    let blocks = pixels.div_ceil(spec::SUM_BLOCK);
    let (total, tile) = if query {
        let queries = p.queries as usize;
        (queries, ((1 << 16) / blocks).clamp(1, queries))
    } else {
        (pixels, 1 << 18)
    };
    let mut acc = if query {
        Some(x.allocate(tile * blocks * 4)?)
    } else {
        None
    };
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "wave_image_integral".into(),
        ..Default::default()
    };
    let mut dispatches = vec![];
    for first in (0..total).step_by(tile) {
        let mut q = p;
        q.first = first as u32;
        q.count = (total - first).min(tile) as u32;
        q.reserved = [blocks as u32, spec::SUM_BLOCK as u32];
        let work = q.count as usize * if query { pixels / 4 } else { 1 };
        dispatches.push((q, work));
    }
    let groups = super::commands(dispatches);
    for (done, group) in groups.iter().enumerate() {
        ctl.check()?;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: inputs and query extents are validated; each dispatch owns its
        // pixel range or its queries' block partials, finished in order.
        let cs = unsafe {
            xr.run("wave image integral", |e| {
                for (i, v) in access.reads.iter().enumerate() {
                    let (b, o) = xr.buffer(v);
                    e.set_buffer(i, b, o);
                }
                e.set_buffer_mut(2, &mut out, 0);
                if let Some(a) = acc.as_mut() {
                    e.set_buffer_mut(3, a, 0);
                }
                for q in group {
                    if query {
                        let mut b = *q;
                        b.count = q.count * blocks as u32;
                        e.set_bytes(4, &b);
                        e.dispatch_1d(&pipe, b.count as usize);
                        e.set_bytes(4, q);
                        e.dispatch_1d(&finish, q.count as usize);
                    } else {
                        e.set_bytes(4, q);
                        e.dispatch_1d(&pipe, q.count as usize);
                    }
                }
                Ok(())
            })?
        };
        stage.command_buffers += cs.command_buffers;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report((done + 1) as f64 / groups.len() as f64, "image integral");
    }
    ctl.check()?;
    x.put_output(access.writes[0], out);
    if let Some(a) = acc {
        x.recycle(a);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.stages.push(stage);
    Ok(())
}
fn metrics(
    input: &mklight_gpu::resident::PlanView,
    output: mklight_gpu::resident::BufferId,
    x: &mut Execution<'_>,
    ctl: &JobControl,
) -> MkResult<()> {
    ctl.check()?;
    let n = input.desc.shape[0];
    let blocks = (n * n).div_ceil(256);
    let mut part = x.allocate(blocks * 10)?;
    let mut out = x.take_output(output);
    let a = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_image_metrics"))?;
    let b = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_image_metrics_finish"))?;
    let start = Instant::now();
    let xr = &*x;
    // SAFETY: disjoint block rows, then a single writer for five summary scalars.
    let cs = unsafe {
        xr.run("image metrics", |e| {
            let (v, o) = xr.buffer(input);
            e.set_buffer(0, v, o);
            e.set_buffer_mut(1, &mut part, 0);
            e.set_buffer_mut(2, &mut out, 0);
            e.set_bytes(3, &[n as u32, blocks as u32]);
            e.dispatch_1d(&a, blocks);
            e.dispatch_1d(&b, 1);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(output, out);
    x.recycle(part);
    x.stats.command_buffers += cs.command_buffers;
    x.stats.device_seconds += cs.gpu_seconds;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "image_metrics".into(),
        command_buffers: cs.command_buffers,
        device_seconds: cs.gpu_seconds,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    Ok(())
}
