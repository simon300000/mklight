use super::*;
use mklight_core::MkError;
use mklight_gpu::resident::unsupported;

pub(super) fn execute(r: &Transport, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let source_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_generate"))?;
    // Immutable topology/material features specialize away unused large local
    // split, thin-film, sag and scatter state without changing the algebra.
    let (mb, mo) = x.buffer(&r.materials);
    let materials: &[Material] =
        bytemuck::cast_slice(&mb.as_slice()[mo..mo + r.materials.desc.len()?]);
    let (pb, po) = x.buffer(&r.primitives);
    let prims: &[Primitive] =
        bytemuck::cast_slice(&pb.as_slice()[po..po + r.primitives.desc.len()?]);
    let features = r.settings.mode
        | (materials
            .iter()
            .any(|m| m.coatings.iter().any(|c| c.kind == 3)) as u32)
            << 2
        | (prims.iter().any(|p| p.kind == 3) as u32) << 3
        | (materials.iter().any(|m| m.doe != 0) as u32) << 4
        | (materials.iter().any(|m| m.bsdf_kind != 0) as u32) << 5
        | (materials.iter().any(|m| m.bsdf_kind >= 2) as u32) << 6
        | (materials.iter().any(|m| m.doe >= 2) as u32) << 7;
    let transport = x.gpu.pipeline(&PipelineKey::with_u32(
        "mkl_stray_transport",
        mklight_gpu::consts::MKL_STRAY_FEATURES_CONSTANT as u16,
        features,
    ))?;
    let spill = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_spill"))?;
    let n = r.ray_count;
    let tile_max = n.min(1024);
    let mut capacity = r.stack_capacity.max(4);
    let words = |size: usize, count: usize| size / 4 * count;
    let mut rays = x.allocate(words(std::mem::size_of::<Ray>(), n))?;
    let stack_words = words(std::mem::size_of::<State>(), tile_max * capacity);
    if stack_words * 4 > 256 * 1024 * 1024 {
        return unsupported("stray initial stack exceeds 256 MiB");
    }
    let mut stack = x.allocate(stack_words)?;
    let mut cdf = x.allocate(2 * r.cdf.len())?;
    cdf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(&r.cdf));
    let mut hits = x.take_output(r.output);
    let mut ledger = x.take_output(r.ledger);
    let mut generated = false;
    let mut stats = StageStats {
        name: "stray transport".into(),
        ..Default::default()
    };
    let mut iterations = 0u64;
    for offset in (0..n).step_by(tile_max) {
        let count = (n - offset).min(tile_max);
        let mut initial = true;
        loop {
            ctl.check()?;
            let wall = Instant::now();
            let p = [
                count as u32,
                r.nodes.desc.shape[0] as u32,
                capacity as u32,
                r.hit_capacity as u32,
                r.event_budget as u32,
                initial as u32,
                r.seed as u32,
                (r.seed >> 32) as u32,
            ];
            // SAFETY: all records are boundary-packed, source-local stack slices
            // are disjoint, and no lease is published until every source is done.
            let cs = unsafe {
                x.gpu.run("stray transport", |e| {
                    if !generated {
                        let gp = [
                            n as u32,
                            0,
                            r.seed as u32,
                            (r.seed >> 32) as u32,
                            r.first_ray as u32,
                            (r.first_ray >> 32) as u32,
                            0,
                            0,
                        ];
                        e.set_pipeline(&source_pso);
                        e.set_bytes(0, &r.source);
                        e.set_buffer(1, &cdf, 0);
                        e.set_buffer_mut(2, &mut rays, 0);
                        e.set_bytes(3, &gp);
                        e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
                    }
                    e.set_pipeline(&transport);
                    for (i, v) in [
                        &r.primitives,
                        &r.nodes,
                        &r.materials,
                        &r.layers,
                        &r.passes,
                        &r.media,
                        &r.tis,
                        &r.sequence,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let (b, o) = x.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer(8, &rays, words(std::mem::size_of::<Ray>(), offset));
                    e.set_buffer_mut(9, &mut stack, 0);
                    e.set_buffer_mut(
                        10,
                        &mut hits,
                        words(std::mem::size_of::<Detection>(), offset * r.hit_capacity),
                    );
                    e.set_buffer_mut(
                        11,
                        &mut ledger,
                        words(std::mem::size_of::<Ledger>(), offset),
                    );
                    e.set_bytes(12, &r.settings);
                    e.set_bytes(13, &p);
                    let (b, o) = x.buffer(&r.shapes);
                    e.set_buffer(14, b, o);
                    e.dispatch_groups([count.div_ceil(32), 1, 1], 32);
                    Ok(())
                })?
            };
            initial = false;
            generated = true;
            iterations += 1;
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
            ctl.check()?;
            let rows: &[Ledger] = bytemuck::cast_slice(ledger.as_slice());
            let rows = &rows[offset..offset + count];
            let completed = rows.iter().filter(|l| l.status == 0).count();
            if rows.iter().any(|l| l.status == 4) {
                return unsupported("stray coating/transport arithmetic requires reference");
            }
            if rows.iter().any(|l| l.status == 3) {
                return unsupported("stray detection capacity; no partial result published");
            }
            if completed == count {
                ctl.report((offset + count) as f64 / n as f64, "stray transport");
                break;
            }
            if rows.iter().any(|l| l.status == 2) {
                let next_capacity = capacity * 2;
                let size = words(std::mem::size_of::<State>(), tile_max * next_capacity);
                if size * 4 > 256 * 1024 * 1024 {
                    return Err(MkError::Unsupported(
                        "stray continuation stack exceeds 256 MiB; no paths discarded".into(),
                    ));
                }
                let mut next = x.allocate(size)?;
                let spill_wall = Instant::now();
                let cs = unsafe {
                    x.gpu.run("stray stack spill", |e| {
                        e.set_pipeline(&spill);
                        e.set_buffer(0, &stack, 0);
                        e.set_buffer_mut(1, &mut next, 0);
                        e.set_buffer(2, &ledger, words(std::mem::size_of::<Ledger>(), offset));
                        e.set_bytes(3, &[count as u32, capacity as u32, next_capacity as u32, 0]);
                        e.dispatch_groups([count.div_ceil(64), 1, 1], 64);
                        Ok(())
                    })?
                };
                let wall = spill_wall.elapsed().as_secs_f64();
                x.stats.device_seconds += cs.gpu_seconds;
                x.stats.command_buffers += 1;
                x.stats.parameter_bytes += 16;
                x.stats.stages.push(StageStats {
                    name: "stray stack spill".into(),
                    wall_seconds: wall,
                    device_seconds: cs.gpu_seconds,
                    command_buffers: 1,
                    max_command_seconds: wall,
                    ..Default::default()
                });
                x.recycle(stack);
                stack = next;
                capacity = next_capacity;
            }
            // Completed sources never become active again. Reserve a tail for
            // long split trees instead of guessing their final path count.
            ctl.report(
                (offset as f64
                    + 0.9 * completed as f64
                    + 0.09 * count as f64 * (1. - 1. / (iterations + 1) as f64))
                    / n as f64,
                "stray continuation",
            );
        }
    }
    x.recycle(rays);
    x.recycle(stack);
    x.recycle(cdf);
    x.put_output(r.output, hits);
    x.put_output(r.ledger, ledger);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.parameter_bytes += iterations * (std::mem::size_of::<Settings>() + 32) as u64;
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}

pub(super) fn build(r: &Build, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_build"))?;
    let n = r.primitives.desc.shape[0];
    let size = n.next_power_of_two();
    let depth = size.ilog2();
    let mut out = x.take_output(r.output);
    let start = Instant::now();
    let mut stats = StageStats {
        name: "stray BVH build".into(),
        ..Default::default()
    };
    for level in (0..=depth).rev() {
        ctl.check()?;
        let wall = Instant::now();
        let cs = unsafe {
            x.gpu.run("stray BVH level", |e| {
                e.set_pipeline(&pso);
                let (b, o) = x.buffer(&r.primitives);
                e.set_buffer(0, b, o);
                e.set_buffer_mut(1, &mut out, 0);
                e.set_bytes(2, &[n as u32, size as u32, level, 0]);
                e.dispatch_groups([(1usize << level).div_ceil(64), 1, 1], 64);
                Ok(())
            })?
        };
        stats.device_seconds += cs.gpu_seconds;
        stats.command_buffers += 1;
        stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.check()?;
        ctl.report(
            (depth - level + 1) as f64 / (depth + 1) as f64,
            "stray BVH build",
        );
    }
    x.put_output(r.output, out);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}
