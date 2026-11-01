use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::resample::*};
use std::time::Instant;
pub fn execute(r: &Resample, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let p = params(r);
    let ns = (p.n * p.supersample) as usize;
    let cells = ((p.nx - 1) * (p.ny - 1)) as usize;
    let mut owner = x.allocate(ns * ns)?;
    let mut tris = x.allocate(cells * 6)?;
    let mut amp = x.take_output(r.amplitude);
    let mut cov = x.take_output(r.coverage);
    let mut opd = x.take_output(r.opd);
    let mut diag = x.take_output(r.diagnostics);
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        name: "wave_resample".into(),
        ..Default::default()
    };
    let names = [
        "mkl_wave_owner_clear",
        "mkl_wave_triangles",
        "mkl_wave_raster",
        "mkl_wave_resample",
    ];
    let sizes = [ns * ns, cells, 2 * cells, r.size * r.size];
    for step in 0..4 {
        let pipeline = x.gpu.pipeline(&PipelineKey::plain(names[step]))?;
        let tile = match step {
            2 => ((1 << 20) * cells / (ns * ns)).max(128),
            3 => (65536 / (r.supersample * r.supersample)).max(1),
            _ => 1 << 20,
        };
        for first in (0..sizes[step]).step_by(tile) {
            ctl.check()?;
            let mut q = p;
            q.first = first as u32;
            q.count = (sizes[step] - first).min(tile) as u32;
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: validated grid extents size the owner raster and triangle
            // table; all writes target exclusive plan scratch.
            let cs = unsafe {
                xr.gpu.run("wave resample", |e| {
                    let bind = |index, v| {
                        let (b, o) = xr.buffer(v);
                        e.set_buffer(index, b, o);
                    };
                    bind(0, &r.nodes);
                    bind(1, &r.valid);
                    bind(2, &r.edges);
                    e.set_buffer_mut(3, &mut tris, 0);
                    e.set_buffer_mut(4, &mut owner, 0);
                    e.set_buffer_mut(5, &mut amp, 0);
                    e.set_buffer_mut(6, &mut cov, 0);
                    e.set_buffer_mut(7, &mut opd, 0);
                    e.set_buffer_mut(8, &mut diag, 0);
                    e.set_bytes(9, &q);
                    e.dispatch_1d(&pipeline, q.count as usize);
                    Ok(())
                })?
            };
            stage.command_buffers += 1;
            stage.device_seconds += cs.gpu_seconds;
            stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
            x.stats.parameter_bytes += size_of::<Params>() as u64;
            ctl.report(
                (step as f64 + (first + q.count as usize) as f64 / sizes[step] as f64) / 4.,
                "pupil resampling",
            );
        }
    }
    ctl.check()?;
    x.put_output(r.amplitude, amp);
    x.put_output(r.coverage, cov);
    x.put_output(r.opd, opd);
    x.put_output(r.diagnostics, diag);
    x.recycle(owner);
    x.recycle(tris);
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.stages.push(stage);
    Ok(())
}
