use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    resident::StageStats,
    wave_ops::{shells::*, spec},
};
use std::time::Instant;
pub fn execute(r: &Shells, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let mut p = params(r);
    let mut out = x.take_output(a.writes[0]);
    let mut aux = if a.writes.len() > 1 {
        Some(x.take_output(a.writes[1]))
    } else {
        None
    };
    // Queries sum all keys: each thread sums one fixed block for every
    // field, a finish dispatch adds the blocks in order.
    let blocks = (p.length as usize).div_ceil(spec::SUM_BLOCK);
    let tile = ((1usize << 16) / blocks).clamp(1, p.queries.max(1) as usize);
    let mut acc = if p.mode == 4 {
        Some(x.allocate(tile * blocks * p.fields as usize * 2)?)
    } else {
        None
    };
    let names = [
        "mkl_wave_shell_keys",
        "mkl_wave_shell_boundaries",
        "mkl_wave_shell_offsets",
        "mkl_wave_shell_gather",
        "mkl_wave_shell_query",
    ];
    let pipe = x
        .gpu
        .pipeline(&PipelineKey::plain(names[p.mode as usize]))?;
    let clear = match p.mode {
        2 => Some(
            x.gpu
                .pipeline(&PipelineKey::plain("mkl_wave_shell_offsets_clear"))?,
        ),
        4 => Some(
            x.gpu
                .pipeline(&PipelineKey::plain("mkl_wave_shell_query_finish"))?,
        ),
        _ => None,
    };
    let mut dispatches = vec![];
    if p.mode == 4 {
        p.reserved = [blocks as u32, spec::SUM_BLOCK as u32];
        for q in (0..p.queries as usize).step_by(tile) {
            p.query_first = q as u32;
            p.query_count = (p.queries as usize - q).min(tile) as u32;
            let work = p.query_count as usize * p.length as usize * p.fields as usize / 4;
            dispatches.push((p, work));
        }
    } else {
        let total = p.length + u32::from(p.mode == 2);
        for first in (0..total).step_by(1 << 18) {
            p.first = first;
            p.count = (total - first).min(1 << 18);
            dispatches.push((p, p.count as usize));
        }
    }
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "wave_radial_bins".into(),
        ..Default::default()
    };
    let count = dispatches.len();
    let groups = super::commands(dispatches);
    for (done, group) in groups.iter().enumerate() {
        ctl.check()?;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: fixed typed extents and immutable radial order; only exclusive
        // accumulator/offset storage is written, with integer atomic max only.
        let cs = unsafe {
            xr.run("radial energy", |e| {
                if let Shells::Query {
                    values,
                    keys,
                    radii,
                    ..
                } = r
                {
                    for i in 0..16 {
                        let (b, o) = xr.buffer(&values[i.min(values.len() - 1)]);
                        e.set_buffer(i, b, o);
                    }
                    let (b, o) = xr.buffer(keys);
                    e.set_buffer(16, b, o);
                    let (b, o) = xr.buffer(radii);
                    e.set_buffer(17, b, o);
                    e.set_buffer_mut(18, &mut out, 0);
                    e.set_buffer_mut(19, acc.as_mut().unwrap(), 0);
                } else {
                    for (i, v) in a.reads.iter().enumerate() {
                        let (b, o) = xr.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer_mut(8, &mut out, 0);
                    if let Some(o) = aux.as_mut() {
                        e.set_buffer_mut(9, o, 0);
                    }
                }
                if let (true, 2, Some(c)) = (done == 0, p.mode, &clear) {
                    e.set_bytes(20, &p);
                    e.dispatch_1d(c, 1);
                }
                for q in group {
                    if let (4, Some(finish)) = (p.mode, &clear) {
                        let mut b = *q;
                        b.query_count = q.query_count * blocks as u32;
                        e.set_bytes(20, &b);
                        e.dispatch_1d(&pipe, b.query_count as usize);
                        e.set_bytes(20, q);
                        e.dispatch_1d(finish, q.query_count as usize);
                    } else {
                        e.set_bytes(20, q);
                        e.dispatch_1d(&pipe, q.count as usize);
                    }
                }
                Ok(())
            })?
        };
        stage.command_buffers += cs.command_buffers;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report((done + 1) as f64 / groups.len() as f64, "radial energy");
    }
    ctl.check()?;
    x.put_output(a.writes[0], out);
    if let Some(o) = aux {
        x.put_output(a.writes[1], o);
    }
    if let Some(o) = acc {
        x.recycle(o);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.parameter_bytes += (count * size_of::<Params>()) as u64;
    x.stats.stages.push(stage);
    Ok(())
}
