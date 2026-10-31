use super::*;
use mklight_gpu::{
    consts::*,
    layout::ReduceParams,
    reduce,
    transforms::correlation::{Params, params},
};
pub(super) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<f64> {
    let mut max_command = 0f64;
    let mut p = params(r);
    let count = (p.n * p.height) as usize;
    let (freq, output) = match r {
        Request::Correlation(c) => (&c.frequencies, c.output),
        Request::OpticalCorrelation(c) | Request::PrepareOpticalStencil(c) => {
            (&c.frequencies, c.output)
        }
        _ => unreachable!(),
    };
    let nf = freq.desc.shape[0];
    let mut out = x.take_output(output);
    if matches!(r, Request::PrepareOpticalStencil(_)) {
        let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_optical_stencil"))?;
        for f in 0..nf {
            p.frequency = f as u32;
            for start in (0..count).step_by(65536) {
                ctl.check()?;
                p.start = start as u32;
                let (a, ao) = x.buffer(freq);
                let wall = Instant::now();
                let cs = unsafe {
                    x.gpu.run("optical stencil", |e| {
                        e.set_buffer(0, a, ao);
                        e.set_buffer_mut(1, &mut out, 0);
                        e.set_bytes(2, &p);
                        e.dispatch_1d(&pso, 65536.min(count - start));
                        Ok(())
                    })?
                };
                max_command = max_command.max(wall.elapsed().as_secs_f64());
                x.stats.command_buffers += 1;
                x.stats.device_seconds += cs.gpu_seconds;
                x.stats.parameter_bytes += size_of::<Params>() as u64;
                ctl.report(
                    (f * count + start + 65536.min(count - start)) as f64 / (nf * count) as f64,
                    "optical stencil",
                );
            }
        }
        x.put_output(output, out);
        return Ok(max_command);
    }
    let groups = reduce::num_groups(count);
    let mut terms = x.allocate(3 * count)?;
    let mut partials = x.allocate(groups * 6)?;
    let mut sums = x.allocate(6)?;
    let pso = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_correlation_terms"))?;
    let norm = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_correlation_normalize"))?;
    let p1 = x
        .gpu
        .pipeline(&PipelineKey::with_u32("mkl_reduce_pass1", 1, MKL_REDUCE_DF))?;
    let p2 = x
        .gpu
        .pipeline(&PipelineKey::with_u32("mkl_reduce_pass2", 1, MKL_REDUCE_DF))?;
    let rp = ReduceParams {
        count: count as u32,
        channel_stride: count as u32,
        num_groups: groups as u32,
        first_group: 0,
    };
    for f in 0..nf {
        p.frequency = f as u32;
        for start in (0..count).step_by(65536) {
            ctl.check()?;
            p.start = start as u32;
            let (field, opd, cov, stencil) = match r {
                Request::Correlation(c) => (
                    &c.pupil,
                    &c.pupil,
                    c.coverage.as_ref().unwrap_or(&c.pupil),
                    &c.pupil,
                ),
                Request::OpticalCorrelation(c) => (
                    &c.amplitude,
                    &c.opd,
                    &c.coverage,
                    c.stencil.as_ref().unwrap_or(&c.amplitude),
                ),
                _ => unreachable!(),
            };
            let (a, ao) = x.buffer(field);
            let (b, bo) = x.buffer(opd);
            let (c, co) = x.buffer(cov);
            let (d, doff) = x.buffer(freq);
            let (e, eo) = x.buffer(stencil);
            let wall = Instant::now();
            let cs = unsafe {
                x.gpu.run("correlation terms", |enc| {
                    enc.set_buffer(0, a, ao);
                    enc.set_buffer(1, b, bo);
                    enc.set_buffer(2, c, co);
                    enc.set_buffer(3, d, doff);
                    enc.set_buffer(4, e, eo);
                    enc.set_buffer_mut(5, &mut terms, 0);
                    enc.set_bytes(6, &p);
                    enc.dispatch_1d(&pso, 65536.min(count - start));
                    Ok(())
                })?
            };
            max_command = max_command.max(wall.elapsed().as_secs_f64());
            x.stats.command_buffers += 1;
            x.stats.device_seconds += cs.gpu_seconds;
            x.stats.parameter_bytes += size_of::<Params>() as u64;
            ctl.report(
                (f as f64 + 0.8 * (start + 65536.min(count - start)) as f64 / count as f64)
                    / nf as f64,
                "correlation terms",
            );
        }
        ctl.check()?;
        let wall = Instant::now();
        let cs = unsafe {
            x.gpu.run("correlation reduction", |e| {
                e.set_pipeline(&p1);
                e.set_buffer(0, &terms, 0);
                e.set_buffer_mut(1, &mut partials, 0);
                e.set_bytes(2, &rp);
                e.dispatch_groups([groups, 3, 1], MKL_REDUCE_THREADS as usize);
                e.set_pipeline(&p2);
                e.set_buffer(0, &partials, 0);
                e.set_buffer_mut(1, &mut sums, 0);
                e.dispatch_groups([1, 3, 1], MKL_REDUCE_THREADS as usize);
                e.set_buffer(0, &sums, 0);
                e.set_buffer_mut(1, &mut out, f * 2);
                e.dispatch_1d(&norm, 1);
                Ok(())
            })?
        };
        max_command = max_command.max(wall.elapsed().as_secs_f64());
        x.stats.command_buffers += 1;
        x.stats.device_seconds += cs.gpu_seconds;
        x.stats.parameter_bytes += size_of::<ReduceParams>() as u64;
        ctl.report((f + 1) as f64 / nf as f64, "correlation");
    }
    x.recycle(terms);
    x.recycle(partials);
    x.recycle(sums);
    x.put_output(output, out);
    Ok(max_command)
}
