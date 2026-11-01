//! Batched source/BVH encoding; shared allocations and bounded commands.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, stray_ops::*};
use std::time::Instant;
mod tally;
mod transport;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    if let Request::Tally(t) = r {
        return tally::execute(t, x, ctl);
    }
    if let Request::Build(r) = r {
        return transport::build(r, x, ctl);
    }
    if let Request::Transport(r) = r {
        return transport::execute(r, x, ctl);
    }
    let (name, n, tile, output) = match r {
        Request::Generate(r) => ("mkl_stray_generate", r.ray_count, r.tile, r.output),
        Request::Cast(r) => (
            "mkl_stray_cast",
            r.rays().desc.shape[0],
            r.tile(),
            r.output(),
        ),
        Request::Tally(_) | Request::Build(_) => unreachable!(),
        Request::Transport(_) => {
            return mklight_gpu::resident::unsupported(
                "stray split-tree transport is not qualified",
            );
        }
    };
    let start = Instant::now();
    let pso = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let mut out = x.take_output(output);
    let cdf = if let Request::Generate(r) = r {
        let mut b = x.allocate(r.cdf.len() * 2)?;
        for (dst, src) in b.as_mut_slice().chunks_exact_mut(2).zip(&r.cdf) {
            dst[0] = src.hi.to_bits();
            dst[1] = src.lo.to_bits();
        }
        Some(b)
    } else {
        None
    };
    let mut stats = StageStats {
        name: name.into(),
        ..Default::default()
    };
    for offset in (0..n).step_by(tile) {
        ctl.check()?;
        let wall = Instant::now();
        let mut p = [0u32; 8];
        p[0] = (offset + tile).min(n) as u32;
        p[1] = offset as u32;
        // SAFETY: records, extents, aligned views and exclusive output are
        // validated before encoding; BVH topology is immutable after upload.
        let cs = unsafe {
            x.gpu.run(name, |e| {
                e.set_pipeline(&pso);
                match r {
                    Request::Generate(r) => {
                        p[2] = r.seed as u32;
                        p[3] = (r.seed >> 32) as u32;
                        p[4] = r.first_ray as u32;
                        p[5] = (r.first_ray >> 32) as u32;
                        e.set_bytes(0, &r.source);
                        e.set_buffer(1, cdf.as_ref().unwrap(), 0);
                        e.set_buffer_mut(2, &mut out, 0);
                        e.set_bytes(3, &p);
                    }
                    Request::Cast(r) => {
                        p[6] = r.nodes().desc.shape[0] as u32;
                        p[7] = r.primitives().desc.shape[0] as u32;
                        for (i, v) in [r.primitives(), r.nodes(), r.rays()]
                            .into_iter()
                            .enumerate()
                        {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(i, b, o);
                        }
                        e.set_buffer_mut(3, &mut out, 0);
                        e.set_bytes(4, &p);
                        let (b, o) = x.buffer(r.shapes());
                        e.set_buffer(5, b, o);
                    }
                    Request::Tally(_) | Request::Build(_) | Request::Transport(_) => unreachable!(),
                }
                e.dispatch_groups([((n - offset).min(tile)).div_ceil(64), 1, 1], 64);
                Ok(())
            })?
        };
        stats.device_seconds += cs.gpu_seconds;
        stats.command_buffers += 1;
        stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
        x.stats.parameter_bytes += 32
            + if matches!(r, Request::Generate(_)) {
                std::mem::size_of::<Source>() as u64
            } else {
                0
            };
        ctl.check()?;
        ctl.report(
            ((offset + tile).min(n)) as f64 / n as f64,
            "stray source/BVH",
        );
    }
    if let Some(cdf) = cdf {
        x.recycle(cdf);
    }
    x.put_output(output, out);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}
