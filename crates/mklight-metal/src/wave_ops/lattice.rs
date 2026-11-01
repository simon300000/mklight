use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::lattice::*};
use std::time::Instant;
pub fn execute(r: &Lattice, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let (name, count, mut p) = match r {
        Lattice::Wavelength {
            count, wavelength, ..
        } => (
            "mkl_wave_wavelength",
            *count,
            [*wavelength, 0, 0, 0, 0, 0, 0, 0],
        ),
        Lattice::Generate { side, centres, .. } => (
            "mkl_wave_lattice",
            side * side + if *centres { (side - 1) * (side - 1) } else { 0 },
            [*side as u32, 0, 0, 0, 0, 0, 0, 0],
        ),
        Lattice::Survival { samples, .. } => ("mkl_wave_survival", samples.desc.shape[0], [0; 8]),
        Lattice::Rim {
            edges,
            previous,
            tolerance,
            final_point,
            ..
        } => (
            "mkl_wave_rim",
            edges.desc.shape[0],
            [
                0,
                0,
                u32::from(previous.is_some()),
                u32::from(*final_point),
                tolerance.hi.to_bits(),
                tolerance.lo.to_bits(),
                0,
                0,
            ],
        ),
    };
    p[1] = count as u32;
    let pipe = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let mut out = x.take_output(a.writes[0]);
    let mut second = if a.writes.len() > 1 {
        Some(x.take_output(a.writes[1]))
    } else {
        None
    };
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "wave_lattice".into(),
        ..Default::default()
    };
    for first in (0..count).step_by(1 << 18) {
        ctl.check()?;
        p[6] = first as u32;
        p[7] = (count - first).min(1 << 18) as u32;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: all views and outputs have validated row counts and layouts.
        let cs = unsafe {
            xr.gpu.run("wave lattice", |e| {
                for (i, v) in a.reads.iter().enumerate() {
                    let (b, o) = xr.buffer(v);
                    e.set_buffer(i, b, o);
                }
                // Initial rim dispatch does not read its optional inputs, but Metal
                // still requires bindings for statically declared buffers.
                if a.reads.len() == 1 {
                    let (b, o) = xr.buffer(&a.reads[0]);
                    e.set_buffer(1, b, o);
                    e.set_buffer(2, b, o);
                }
                e.set_buffer_mut(3, &mut out, 0);
                if let Some(o) = second.as_mut() {
                    e.set_buffer_mut(4, o, 0);
                }
                e.set_bytes(5, &p);
                e.dispatch_1d(&pipe, p[7] as usize);
                Ok(())
            })?
        };
        stage.command_buffers += 1;
        stage.device_seconds += cs.gpu_seconds;
        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report(
            (first + p[7] as usize) as f64 / count as f64,
            "wave lattice",
        );
    }
    ctl.check()?;
    x.put_output(a.writes[0], out);
    if let Some(o) = second {
        x.put_output(a.writes[1], o);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.command_buffers += stage.command_buffers;
    x.stats.device_seconds += stage.device_seconds;
    x.stats.parameter_bytes += stage.command_buffers as u64 * 32;
    x.stats.stages.push(stage);
    Ok(())
}
