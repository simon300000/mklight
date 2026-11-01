use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::placement::*};
use std::time::Instant;
pub fn execute(r: &Placement, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let mut p = params(r);
    let nodes = matches!(r, Placement::Nodes { .. });
    let name = match r {
        Placement::Centroid { .. } => "mkl_wave_centroid",
        Placement::Project { .. } => "mkl_wave_place",
        Placement::Nodes { .. } => "mkl_wave_node_amplitude",
        Placement::Slopes { .. } => "mkl_wave_slopes",
    };
    let pipe = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let finish = if nodes {
        Some((
            x.gpu.pipeline(&PipelineKey::plain("mkl_wave_node_norm"))?,
            x.gpu.pipeline(&PipelineKey::plain("mkl_wave_nodes"))?,
        ))
    } else {
        None
    };
    let mut out = x.take_output(a.writes[0]);
    let mut second = if a.writes.len() > 1 {
        Some(x.take_output(a.writes[1]))
    } else {
        None
    };
    let mut amp = if nodes {
        Some(x.allocate(p.n as usize * 2)?)
    } else {
        None
    };
    let mut norm = if nodes { Some(x.allocate(4)?) } else { None };
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "wave_placement".into(),
        ..Default::default()
    };
    // The norm is small lattice control. Amplitude/placement kernels run across
    // all nodes and use a fixed logical reduction order independent of commands.
    let steps = if nodes { 3 } else { 1 };
    for step in 0..steps {
        let count = if step == 1 { 1 } else { p.n as usize };
        for first in (0..count).step_by(1 << 18) {
            ctl.check()?;
            p.first = first as u32;
            p.count = (count - first).min(1 << 18) as u32;
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: validated contiguous inputs; all mutable storage is plan-local.
            let cs = unsafe {
                xr.gpu.run("wave placement", |e| {
                    for (i, v) in a.reads.iter().enumerate() {
                        let (b, o) = xr.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    if matches!(r, Placement::Centroid { power: None, .. }) {
                        let (b, o) = xr.buffer(&a.reads[0]);
                        e.set_buffer(1, b, o);
                    }
                    if matches!(r, Placement::Project { power: None, .. }) {
                        let (b, o) = xr.buffer(&a.reads[0]);
                        e.set_buffer(4, b, o);
                    }
                    e.set_buffer_mut(5, &mut out, 0);
                    if let Some(o) = second.as_mut() {
                        e.set_buffer_mut(6, o, 0);
                    }
                    if let Some(o) = amp.as_mut() {
                        e.set_buffer_mut(6, o, 0);
                    }
                    if let Some(o) = norm.as_mut() {
                        e.set_buffer_mut(7, o, 0);
                    }
                    e.set_bytes(8, &p);
                    let kernel = match step {
                        0 => &pipe,
                        1 => &finish.as_ref().unwrap().0,
                        _ => &finish.as_ref().unwrap().1,
                    };
                    e.dispatch_1d(kernel, p.count as usize);
                    Ok(())
                })?
            };
            stage.command_buffers += 1;
            stage.device_seconds += cs.gpu_seconds;
            stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
            ctl.report(
                (step as f64 + (first + p.count as usize) as f64 / count as f64) / steps as f64,
                "wave placement",
            );
        }
    }
    ctl.check()?;
    x.put_output(a.writes[0], out);
    if let Some(o) = second {
        x.put_output(a.writes[1], o);
    }
    if let Some(o) = amp {
        x.recycle(o);
    }
    if let Some(o) = norm {
        x.recycle(o);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.parameter_bytes += stage.command_buffers as u64 * size_of::<Params>() as u64;
    x.stats.stages.push(stage);
    Ok(())
}
