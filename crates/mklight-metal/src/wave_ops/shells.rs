use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::shells::*};
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
    let mut acc = if p.mode == 4 {
        Some(x.allocate((p.fields * p.queries * 2) as usize)?)
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
    let clear = if p.mode == 2 {
        Some(
            x.gpu
                .pipeline(&PipelineKey::plain("mkl_wave_shell_offsets_clear"))?,
        )
    } else {
        None
    };
    let mut dispatches = vec![];
    if p.mode == 4 {
        for q in (0..p.queries).step_by(128) {
            p.query_first = q;
            p.query_count = (p.queries - q).min(128);
            for first in (0..p.length).step_by(512) {
                p.first = first;
                p.count = (p.length - first).min(512);
                p.finish = u32::from(first + p.count == p.length);
                dispatches.push((p, (p.count * p.query_count * p.fields) as usize));
            }
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
    let mut offset = 0;
    while offset < dispatches.len() {
        ctl.check()?;
        let begin = offset;
        let mut work = 0;
        while offset < dispatches.len() && work < 1 << 20 {
            work += dispatches[offset].1;
            offset += 1;
        }
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: fixed typed extents and immutable radial order; only exclusive
        // accumulator/offset storage is written, with integer atomic max only.
        let cs = unsafe {
            xr.gpu.run("radial energy", |e| {
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
                if begin == 0 {
                    if let Some(c) = &clear {
                        e.set_bytes(20, &p);
                        e.dispatch_1d(c, 1);
                    }
                }
                for (q, _) in &dispatches[begin..offset] {
                    e.set_bytes(20, q);
                    e.dispatch_1d(
                        &pipe,
                        if p.mode == 4 {
                            q.query_count as usize
                        } else {
                            q.count as usize
                        },
                    );
                }
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report(offset as f64 / dispatches.len() as f64, "radial energy");
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
    x.stats.parameter_bytes += (dispatches.len() * size_of::<Params>()) as u64;
    x.stats.stages.push(stage);
    Ok(())
}
