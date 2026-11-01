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
    let names = [
        "mkl_wave_mask_rows",
        "mkl_wave_owner_clear",
        "mkl_wave_triangles",
        "mkl_wave_raster",
        "mkl_wave_nearest_blocks",
        "mkl_wave_resample",
    ];
    let pipelines = names
        .iter()
        .map(|n| x.gpu.pipeline(&PipelineKey::plain(n)))
        .collect::<MkResult<Vec<_>>>()?;
    let ss = r.supersample * r.supersample;
    // (threads, tile, work units per thread) of each step.
    let sizes = [
        (
            ns,
            ((1usize << 20) / p.edges.max(1) as usize).max(1),
            p.edges.max(1) as usize,
        ),
        (ns * ns, 1 << 20, 1),
        (cells, 1 << 20, 1),
        (
            2 * cells,
            ((1 << 20) * cells / (ns * ns)).max(128),
            (ns * ns).div_ceil(cells),
        ),
        (ns * ns.div_ceil(256), (1 << 20) / 256, 256),
        (r.size * r.size, ((1 << 18) / ss).max(1), 4 * ss),
    ];
    let mut dispatches = vec![];
    for (step, &(total, tile, cost)) in sizes.iter().enumerate() {
        if (step == 0 && !cache) || (step == 4 && nearest.is_none()) {
            continue;
        }
        for first in (0..total).step_by(tile) {
            let count = (total - first).min(tile);
            dispatches.push(((step, first as u32, count as u32), count * cost));
        }
    }
    let groups = super::commands(dispatches);
    for (done, group) in groups.iter().enumerate() {
        ctl.check()?;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: validated grid extents size the owner raster and triangle
        // table; all writes target exclusive plan scratch, and each mask row
        // owns its edge-count stride and count slot.
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
                if let (Some(c), Some(n)) = (crossings.as_mut(), counts.as_mut()) {
                    e.set_buffer_mut(10, c, 0);
                    e.set_buffer_mut(11, n, 0);
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
                for &(step, first, count) in group {
                    let mut q = p;
                    q.first = first;
                    q.count = count;
                    e.set_bytes(9, &q);
                    if step == 4 {
                        e.set_pipeline(&pipelines[step]);
                        e.dispatch_groups([count as usize, 1, 1], 256);
                    } else {
                        e.dispatch_1d(&pipelines[step], count as usize);
                    }
                }
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += (group.len() * size_of::<Params>()) as u64;
        ctl.report((done + 1) as f64 / groups.len() as f64, "pupil resampling");
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
