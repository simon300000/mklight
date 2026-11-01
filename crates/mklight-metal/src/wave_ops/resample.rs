use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::resample::*};
use std::time::Instant;
pub fn execute(r: &Resample, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let mut p = params(r);
    let ns = (p.n * p.supersample) as usize;
    let cells = ((p.nx - 1) * (p.ny - 1)) as usize;
    let mut owner = x.allocate(ns * ns)?;
    let mut tris = x.allocate(cells * 6)?;
    let mut nearest = if p.mask != 0 {
        Some(x.allocate(ns * ns * 2)?)
    } else {
        None
    };
    let mut column_blocks = if p.mask != 0 {
        Some(x.allocate(ns * ns.div_ceil(256) * 2)?)
    } else {
        None
    };
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
    // Polygon crossings depend on the subrow, not the subpixel. Retain the
    // exact edge order/arithmetic and bound this optional cache at 256 MiB.
    let cache = p.mask == 4 && ns * p.edges as usize * 8 <= 256 * 1024 * 1024;
    p.reserved[0] = u32::from(cache);
    let mut crossings = if cache {
        Some(x.allocate(ns * p.edges as usize * 2)?)
    } else {
        None
    };
    let mut counts = if cache { Some(x.allocate(ns)?) } else { None };
    if cache {
        let pipeline = x.gpu.pipeline(&PipelineKey::plain("mkl_wave_mask_rows"))?;
        let tile = ((1usize << 20) / p.edges as usize).max(1);
        for first in (0..ns).step_by(tile) {
            ctl.check()?;
            let mut q = p;
            q.first = first as u32;
            q.count = (ns - first).min(tile) as u32;
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: each row owns its edge-count stride and count slot.
            let cs = unsafe {
                xr.gpu.run("pupil mask crossings", |e| {
                    let (b, o) = xr.buffer(&r.edges);
                    e.set_buffer(2, b, o);
                    e.set_buffer_mut(10, crossings.as_mut().unwrap(), 0);
                    e.set_buffer_mut(11, counts.as_mut().unwrap(), 0);
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
                0.05 * (first + q.count as usize) as f64 / ns as f64,
                "pupil mask",
            );
        }
    }
    let names = [
        "mkl_wave_owner_clear",
        "mkl_wave_triangles",
        "mkl_wave_raster",
        "mkl_wave_nearest_blocks",
        "mkl_wave_resample",
    ];
    let sizes = [
        ns * ns,
        cells,
        2 * cells,
        ns * ns.div_ceil(256),
        r.size * r.size,
    ];
    for step in 0..5 {
        if step == 3 && nearest.is_none() {
            continue;
        }
        let pipeline = x.gpu.pipeline(&PipelineKey::plain(names[step]))?;
        let tile = match step {
            2 => ((1 << 20) * cells / (ns * ns)).max(128),
            3 => (1 << 20) / 256,
            4 => ((1 << 18) / (r.supersample * r.supersample)).max(1),
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
                    if let (Some(c), Some(n)) = (&crossings, &counts) {
                        e.set_buffer(10, c, 0);
                        e.set_buffer(11, n, 0);
                    } else {
                        bind(10, &r.nodes);
                        bind(11, &r.valid);
                    }
                    if let Some(n) = nearest.as_mut() {
                        e.set_buffer_mut(12, n, 0);
                    } else {
                        bind(12, &r.valid);
                    }
                    if let Some(n) = column_blocks.as_mut() {
                        e.set_buffer_mut(13, n, 0);
                    } else {
                        bind(13, &r.valid);
                    }
                    e.set_bytes(9, &q);
                    if step == 3 {
                        e.set_pipeline(&pipeline);
                        e.dispatch_groups([q.count as usize, 1, 1], 256);
                    } else {
                        e.dispatch_1d(&pipeline, q.count as usize);
                    }
                    Ok(())
                })?
            };
            stage.command_buffers += 1;
            stage.device_seconds += cs.gpu_seconds;
            stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
            x.stats.parameter_bytes += size_of::<Params>() as u64;
            ctl.report(
                (step as f64 + (first + q.count as usize) as f64 / sizes[step] as f64) / 5.,
                "pupil resampling",
            );
        }
    }
    ctl.check()?;
    x.put_output(r.amplitude, amp);
    x.put_output(r.coverage, cov);
    x.put_output(r.opd, opd);
    x.put_output(r.diagnostics, diag);
    if let Some(c) = crossings {
        x.recycle(c);
    }
    if let Some(c) = counts {
        x.recycle(c);
    }
    if let Some(n) = column_blocks {
        x.recycle(n);
    }
    if let Some(n) = nearest {
        x.recycle(n);
    }
    x.recycle(owner);
    x.recycle(tris);
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.stages.push(stage);
    Ok(())
}
