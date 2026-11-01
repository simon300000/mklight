use super::*;
use mklight_gpu::{
    layout::ReduceParams,
    reduce,
    transforms::correlation::{Params, params},
};
/// Pupil samples × frequencies per command; also bounds the term planes.
const TILE: usize = 1 << 22;

pub(super) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<f64> {
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
    let tile = (TILE / count).clamp(1, nf);
    let mut out = x.take_output(output);
    let mut max_command = 0f64;
    let mut account = |x: &mut Execution<'_>, wall: Instant, done: usize| {
        max_command = max_command.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.report(done as f64 / nf as f64, "correlation");
    };
    if matches!(r, Request::PrepareOpticalStencil(_)) {
        let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_optical_stencil"))?;
        for first in (0..nf).step_by(tile) {
            ctl.check()?;
            (p.first, p.frequencies) = (first as u32, tile.min(nf - first) as u32);
            let (a, ao) = x.buffer(freq);
            let wall = Instant::now();
            // SAFETY: the validated stencil holds nf·count·5 floats; the grid
            // covers frequencies first..first+p.frequencies.
            unsafe {
                x.run("optical stencil", |e| {
                    e.set_pipeline(&pso);
                    e.set_buffer(0, a, ao);
                    e.set_buffer_mut(1, &mut out, 0);
                    e.set_bytes(2, &p);
                    e.dispatch_groups([count.div_ceil(256), p.frequencies as usize, 1], 256);
                    Ok(())
                })?
            };
            account(x, wall, first + p.frequencies as usize);
        }
        x.put_output(output, out);
        return Ok(max_command);
    }
    let groups = reduce::num_groups(count);
    let mut terms = x.allocate(3 * tile * count)?;
    let mut partials = x.allocate(3 * tile * groups * 2)?;
    let mut sums = x.allocate(3 * tile * 2)?;
    let pipeline = |name, mode| x.gpu.pipeline(&PipelineKey::with_u32(name, 1, mode));
    let (pass1, pass2) = (
        pipeline("mkl_reduce_pass1", MKL_REDUCE_DF)?,
        pipeline("mkl_reduce_pass2", MKL_REDUCE_DF)?,
    );
    let terms_pso = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_correlation_terms"))?;
    let norm = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_correlation_normalize"))?;
    let rp = ReduceParams {
        count: count as u32,
        channel_stride: count as u32,
        num_groups: groups as u32,
        first_group: 0,
    };
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
    for first in (0..nf).step_by(tile) {
        ctl.check()?;
        (p.first, p.frequencies) = (first as u32, tile.min(nf - first) as u32);
        let channels = 3 * p.frequencies as usize;
        let views = [field, opd, cov, freq, stencil].map(|v| x.buffer(v));
        let wall = Instant::now();
        // SAFETY: validated views cover count samples (stencil nf·count·5);
        // term/partial/sum planes are sized for a whole tile; the output
        // holds nf complex values.
        unsafe {
            x.run("correlation", |e| {
                e.set_pipeline(&terms_pso);
                for (i, (v, o)) in views.iter().enumerate() {
                    e.set_buffer(i, v, *o);
                }
                e.set_buffer_mut(5, &mut terms, 0);
                e.set_bytes(6, &p);
                e.dispatch_groups([count.div_ceil(256), p.frequencies as usize, 1], 256);
                e.set_pipeline(&pass1);
                e.set_buffer(0, &terms, 0);
                e.set_buffer_mut(1, &mut partials, 0);
                e.set_bytes(2, &rp);
                e.dispatch_groups([groups, channels, 1], MKL_REDUCE_THREADS as usize);
                e.set_pipeline(&pass2);
                e.set_buffer(0, &partials, 0);
                e.set_buffer_mut(1, &mut sums, 0);
                e.dispatch_groups([1, channels, 1], MKL_REDUCE_THREADS as usize);
                e.set_buffer(0, &sums, 0);
                e.set_buffer_mut(1, &mut out, 0);
                e.set_bytes(2, &p);
                e.dispatch_1d(&norm, p.frequencies as usize);
                Ok(())
            })?
        };
        x.stats.parameter_bytes += size_of::<ReduceParams>() as u64;
        account(x, wall, first + p.frequencies as usize);
    }
    x.recycle(terms);
    x.recycle(partials);
    x.recycle(sums);
    x.put_output(output, out);
    Ok(max_command)
}
