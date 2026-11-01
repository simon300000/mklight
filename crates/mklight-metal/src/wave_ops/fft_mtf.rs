use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::fft_mtf::*};
use std::time::Instant;
pub fn execute(r: &FftMtf, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if let FftMtf::Domain {
        coverage, status, ..
    } = r
    {
        return domain(r, coverage, *status, x, ctl);
    }
    let a = access(r);
    let mut p = params(r);
    let mut out = x.take_output(a.writes[0]);
    let mut second = if a.writes.len() > 1 {
        Some(x.take_output(a.writes[1]))
    } else {
        None
    };
    if p.kind == 0 {
        p.side = (out.len() / 2).isqrt() as u32;
        p.count = p.side * p.side;
    } else if p.kind == 1 {
        p.count = p.n * p.n;
    }
    let name = [
        "mkl_wave_sensor_field",
        "mkl_wave_otf_normalize",
        "mkl_wave_otf_sample",
    ][p.kind as usize];
    let pipe = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let start = Instant::now();
    ctl.check()?;
    let xr = &*x;
    // SAFETY: validated grids and exclusive output leases; padding is written.
    unsafe {
        xr.run("FFT MTF", |e| {
            for (i, v) in a.reads.iter().enumerate() {
                let (b, o) = xr.buffer(v);
                e.set_buffer(i, b, o);
            }
            e.set_buffer_mut(3, &mut out, 0);
            if let Some(o) = second.as_mut() {
                e.set_buffer_mut(4, o, 0);
            }
            e.set_bytes(5, &p);
            e.dispatch_1d(&pipe, p.count as usize);
            Ok(())
        })?
    };
    ctl.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(a.writes[0], out);
    if let Some(o) = second {
        x.put_output(a.writes[1], o);
    }

    x.stats.parameter_bytes += size_of::<Params>() as u64;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "fft_mtf".into(),
        command_buffers: 0,
        device_seconds: 0.,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    Ok(())
}
fn domain(
    r: &FftMtf,
    coverage: &mklight_gpu::resident::PlanView,
    status: mklight_gpu::resident::BufferId,
    x: &mut Execution<'_>,
    ctl: &JobControl,
) -> MkResult<()> {
    ctl.check()?;
    let p = params(r);
    let mut out = x.take_output(status);
    let clear = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_sensor_domain_clear"))?;
    let kernel = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_wave_sensor_domain"))?;
    let start = Instant::now();
    let xr = &*x;
    // SAFETY: the domain counter is an exclusive u32, reset before exact integer
    // increments; immutable coverage has the validated square extent.
    unsafe {
        xr.run("sensor domain", |e| {
            let (b, o) = xr.buffer(coverage);
            e.set_buffer(0, b, o);
            e.set_buffer_mut(6, &mut out, 0);
            e.set_bytes(5, &p);
            e.dispatch_1d(&clear, 1);
            e.dispatch_1d(&kernel, p.count as usize);
            Ok(())
        })?
    };
    ctl.check()?;
    x.flush()?;
    let count = out.as_slice()[0];
    let elapsed = start.elapsed().as_secs_f64();
    x.put_output(status, out);

    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "sensor_domain".into(),
        command_buffers: 0,
        device_seconds: 0.,
        wall_seconds: elapsed,
        max_command_seconds: elapsed,
    });
    if count > 0 {
        return Err(mklight_core::MkError::InvalidParam(format!(
            "pupil crosses image-surface horizon at {count} covered cells"
        )));
    }
    Ok(())
}
