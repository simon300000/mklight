use super::*;
pub(super) fn execute(t: &Tally, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let (name, n, output) = match t {
        Tally::Gather {
            indices, output, ..
        } => ("mkl_stray_gather", indices.desc.shape[0], *output),
        Tally::Tags { params, output, .. } => ("mkl_stray_tags", params.count as usize, *output),
        Tally::Terms {
            params,
            output,
            products,
            ..
        } => (
            if *products {
                "mkl_stray_moments"
            } else {
                "mkl_stray_terms"
            },
            params.sources as usize * params.queries as usize,
            *output,
        ),
        Tally::Transpose {
            sources,
            queries,
            width,
            output,
            ..
        } => (
            "mkl_stray_tally_transpose",
            width * sources * queries,
            *output,
        ),
    };
    let pso = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let mut out = x.take_output(output);
    ctl.check()?;
    let start = Instant::now();
    let cs = unsafe {
        x.gpu.run(name, |e| {
            e.set_pipeline(&pso);
            match t {
                Tally::Gather { hits, indices, .. } => {
                    for (i, v) in [hits, indices].into_iter().enumerate() {
                        let (b, o) = x.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer_mut(2, &mut out, 0);
                    e.set_bytes(3, &(n as u32));
                }
                Tally::Tags {
                    hits,
                    ledger,
                    coverage,
                    params,
                    ..
                } => {
                    for (i, v) in [hits, ledger, coverage].into_iter().enumerate() {
                        let (b, o) = x.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer_mut(3, &mut out, 0);
                    e.set_bytes(4, params);
                    e.set_bytes(5, &(coverage.desc.shape[0] as u32));
                }
                Tally::Terms {
                    hits,
                    ledger,
                    tags,
                    queries,
                    params,
                    ..
                } => {
                    for (i, v) in [hits, ledger, tags, queries].into_iter().enumerate() {
                        let (b, o) = x.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer_mut(4, &mut out, 0);
                    e.set_bytes(5, params);
                }
                Tally::Transpose {
                    input,
                    sources,
                    queries,
                    width,
                    ..
                } => {
                    let (b, o) = x.buffer(input);
                    e.set_buffer(0, b, o);
                    e.set_buffer_mut(1, &mut out, 0);
                    e.set_bytes(2, &[*sources as u32, *queries as u32, *width as u32, 0]);
                }
            }
            e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
            Ok(())
        })?
    };
    ctl.check()?;
    x.put_output(output, out);
    let wall = start.elapsed().as_secs_f64();
    x.stats.command_buffers += 1;
    x.stats.device_seconds += cs.gpu_seconds;
    x.stats.stages.push(StageStats {
        name: name.into(),
        command_buffers: 1,
        device_seconds: cs.gpu_seconds,
        wall_seconds: wall,
        max_command_seconds: wall,
        ..Default::default()
    });
    Ok(())
}
