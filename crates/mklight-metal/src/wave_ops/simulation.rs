use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::simulation::*};
use std::time::Instant;
pub fn execute(r: &Simulation, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if matches!(r, Simulation::Kernel { .. }) {
        return detector_kernel(r, x, ctl);
    }
    let a = access(r);
    let mut p = params(r);
    let mut out = x.take_output(a.writes[0]);
    let mut aux = if a.writes.len() > 1 {
        Some(x.take_output(a.writes[1]))
    } else {
        None
    };
    let mut invalid = if p.mode == 7 {
        let mut v = x.allocate(1)?;
        v.as_mut_slice()[0] = 0;
        Some(v)
    } else {
        None
    };
    let names = [
        "mkl_wave_pattern",
        "mkl_wave_warp",
        "mkl_wave_detector_kernel",
        "mkl_wave_active",
        "mkl_wave_weighted",
        "mkl_wave_crop",
        "mkl_wave_assemble",
        "mkl_wave_noise",
    ];
    let pipe = x
        .gpu
        .pipeline(&PipelineKey::plain(names[p.mode as usize]))?;
    let clear = if p.mode == 3 {
        Some(
            x.gpu
                .pipeline(&PipelineKey::plain("mkl_wave_active_clear"))?,
        )
    } else {
        None
    };
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "image_simulation".into(),
        ..Default::default()
    };
    let total = match p.mode {
        0 | 1 => p.width * p.height,
        3 => (p.tile[2] + 2 * p.radius) * (p.tile[3] + 2 * p.radius),
        4 => p.canvas * p.canvas,
        _ => p.width * p.height * p.channels,
    };
    let rounds = if p.mode == 6 { a.reads.len() } else { 1 };
    for round in 0..rounds {
        if p.mode == 6 {
            let nx = p.width.div_ceil(p.canvas);
            p.tile = [
                round as u32 % nx * p.canvas,
                round as u32 / nx * p.canvas,
                a.reads[round].desc.shape[1] as u32,
                a.reads[round].desc.shape[0] as u32,
            ];
        }
        let count = if p.mode == 6 {
            p.tile[2] * p.tile[3] * p.channels
        } else {
            total
        };
        // Bound actual arithmetic, rather than paying a command round trip
        // for every small slice of simple raster work.
        let tile = match p.mode {
            0 if p.pattern == 1 => 16384,
            1 => ((1usize << 18) / ((p.grid - 1) as usize).pow(2)).clamp(256, 65536),
            3 => 1 << 18,
            7 if p.well > 0. || p.sigma > 0. => 65536,
            _ => 1 << 20,
        };
        for first in (0..count).step_by(tile) {
            ctl.check()?;
            p.first = first;
            p.count = (count - first).min(tile as u32);
            let xr = &*x;
            let wall = Instant::now();
            // SAFETY: validated stage layouts, immutable inputs, exclusive outputs
            // and deterministic scratch reused only after command completion.
            let cs = unsafe {
                xr.gpu.run("image simulation", |e| {
                    if p.mode == 6 {
                        let (b, o) = xr.buffer(&a.reads[round]);
                        e.set_buffer(0, b, o);
                    } else if let Simulation::Crop {
                        planes, previous, ..
                    } = r
                    {
                        for i in 0..3 {
                            let (b, o) = xr.buffer(&planes[i.min(planes.len() - 1)]);
                            e.set_buffer(i, b, o);
                        }
                        let (b, o) = xr.buffer(previous.as_ref().unwrap_or(&planes[0]));
                        e.set_buffer(3, b, o);
                    } else {
                        for (i, v) in a.reads.iter().enumerate() {
                            let (b, o) = xr.buffer(v);
                            e.set_buffer(i, b, o);
                        }
                    }
                    e.set_buffer_mut(8, &mut out, 0);
                    if let Some(o) = aux.as_mut() {
                        e.set_buffer_mut(9, o, 0);
                    }
                    if let Some(v) = invalid.as_mut() {
                        e.set_buffer_mut(9, v, 0);
                    }
                    e.set_bytes(10, &p);
                    if first == 0 && round == 0 {
                        if let Some(clear) = &clear {
                            e.dispatch_1d(clear, (p.grid * p.grid) as usize);
                        }
                    }
                    e.dispatch_1d(&pipe, p.count as usize);
                    Ok(())
                })?
            };
            if let Some(v) = &invalid {
                let count = v.as_slice()[0];
                if count > 0 {
                    return Err(mklight_core::MkError::Unsupported(format!(
                        "{count} image samples exceed the qualified f32 noise domain (Poisson mean <= 1e12)"
                    )));
                }
            }
            stage.command_buffers += 1;
            stage.device_seconds += cs.gpu_seconds;
            stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
            x.stats.parameter_bytes += size_of::<Params>() as u64;
            ctl.report(
                (round as f64 + (first + p.count) as f64 / count as f64) / rounds as f64,
                "image simulation",
            );
        }
    }
    ctl.check()?;
    x.put_output(a.writes[0], out);
    if let Some(o) = aux {
        x.put_output(a.writes[1], o);
    }
    if let Some(v) = invalid {
        x.recycle(v);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.stages.push(stage);
    Ok(())
}
fn detector_kernel(r: &Simulation, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let mut p = params(r);
    let side = (2 * p.radius + 1) as usize;
    let mut horizontal = x.allocate(p.source_width as usize * side * 2)?;
    let mut out = x.take_output(a.writes[0]);
    let pipelines = [
        x.gpu
            .pipeline(&PipelineKey::plain("mkl_wave_kernel_horizontal"))?,
        x.gpu
            .pipeline(&PipelineKey::plain("mkl_wave_kernel_vertical"))?,
    ];
    let clear = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_kernel_clear"))?;
    let mut dispatches = vec![];
    for axis in 0..2 {
        let count = if axis == 0 {
            p.source_width as usize * side
        } else {
            side * side
        };
        let source_per_pixel =
            ((2. * p.pitch[axis].to_f64() + p.aperture[axis].to_f64()) / p.spacing.to_f64() + 4.)
                .min(p.source_width as f64)
                .ceil() as usize;
        let cost = source_per_pixel * p.quadrature as usize;
        let chunk = ((1 << 18) / cost.max(1)).clamp(1, 65536);
        for first in (0..count).step_by(chunk) {
            p.first = first as u32;
            p.count = (count - first).min(chunk) as u32;
            dispatches.push((axis, p, p.count as usize * cost));
        }
    }
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "image_detector_kernel".into(),
        ..Default::default()
    };
    let mut offset = 0;
    while offset < dispatches.len() {
        ctl.check()?;
        let begin = offset;
        let mut work = 0;
        while offset < dispatches.len() && work < 1 << 20 {
            work += dispatches[offset].2;
            offset += 1;
        }
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: horizontal rows are exclusive, complete before the vertical
        // pass, and all output padding is explicitly cleared once.
        let cs = unsafe {
            xr.gpu.run("detector kernel", |e| {
                let (b, o) = xr.buffer(&a.reads[0]);
                e.set_buffer(0, b, o);
                e.set_buffer_mut(8, &mut out, 0);
                e.set_buffer_mut(9, &mut horizontal, 0);
                if begin == 0 {
                    e.set_bytes(10, &p);
                    e.dispatch_1d(&clear, (p.canvas * p.canvas) as usize);
                }
                for (axis, q, _) in &dispatches[begin..offset] {
                    e.set_bytes(10, q);
                    e.dispatch_1d(&pipelines[*axis], q.count as usize);
                }
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report(offset as f64 / dispatches.len() as f64, "detector kernel");
    }
    ctl.check()?;
    x.put_output(a.writes[0], out);
    x.recycle(horizontal);
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.parameter_bytes += (dispatches.len() * size_of::<Params>()) as u64;
    x.stats.stages.push(stage);
    Ok(())
}
