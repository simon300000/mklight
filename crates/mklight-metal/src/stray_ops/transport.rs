use super::*;
use mklight_gpu::consts::MKL_STRAY_FEATURES_CONSTANT;

/// Device time per command: several bounces share a command while they are
/// short, and long bounces run in tiles.
const TARGET_COMMAND_SECONDS: f64 = 0.015;

/// Wavefront transport: per bounce, every live source pops its next path
/// (`mkl_stray_wave_pop`), casts it (`mkl_stray_wave_cast`) and interacts
/// (`mkl_stray_wave_interact`); live-source lists are compacted on the
/// device and each kernel is dispatched indirectly over them. Several
/// bounces share a command; only the live count is read back. Every source
/// keeps its own stack, ledger, detections and Philox cursor, so scheduling
/// never changes a source's depth-first draws or records.
pub(super) fn execute(r: &Transport, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
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
    let specialized =
        |function| PipelineKey::with_u32(function, MKL_STRAY_FEATURES_CONSTANT as u16, features);
    let source_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_generate"))?;
    let init = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_init"))?;
    let args_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_args"))?;
    let pop = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_pop"))?;
    let cast = x.gpu.pipeline(&specialized("mkl_stray_wave_cast"))?;
    let interact = x.gpu.pipeline(&specialized("mkl_stray_wave_interact"))?;
    let widths = [&pop, &cast, &interact].map(|p| crate::context::Encoder::width(p) as u32);
    let n = r.ray_count;
    let words = |size: usize, count: usize| size / 4 * count;
    let mut rays = x.allocate(words(std::mem::size_of::<Ray>(), n))?;
    let mut current = x.allocate(words(std::mem::size_of::<State>(), n))?;
    let mut found = x.allocate(words(std::mem::size_of::<Hit>(), n))?;
    let mut cdf = x.allocate(2 * r.cdf.len())?;
    cdf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(&r.cdf));
    let mut lists = x.allocate(2 * n)?;
    let mut counters = x.allocate(4)?;
    counters.as_mut_slice().fill(0);
    let mut args = x.allocate(12)?;
    args.as_mut_slice().fill(0);
    let mut stack = x.take_output(r.stack);
    let mut hits = x.take_output(r.output);
    let mut ledger = x.take_output(r.ledger);
    let mut stats = StageStats {
        name: "stray wavefront".into(),
        ..Default::default()
    };
    // `tile`: (first list entry, entries) of a tiled cast/interaction.
    let params = |parity: u32, tile: (u32, u32)| {
        [
            n as u32,
            r.nodes.desc.shape[0] as u32,
            r.stack_capacity as u32,
            r.hit_capacity as u32,
            tile.1,
            parity,
            r.seed as u32,
            (r.seed >> 32) as u32,
            tile.0,
            r.head_events,
            r.sag_mode,
            0,
        ]
    };
    // Encodes the sources' generation (`first`) and then `bounces` bounces
    // from list `parity`, each of the phases in `phases` (bits: pop, cast,
    // interact; a tiled bounce encodes the pop and the tiles separately).
    let gpu = x.gpu;
    let reads = |v: &mklight_gpu::resident::PlanView| x.buffer(v);
    let mut encode = |e: &crate::context::Encoder<'_>,
                      counters: &mut crate::resident::Lease,
                      first: bool,
                      bounces: usize,
                      parity: u32,
                      phases: u32,
                      tile: Option<(u32, u32)>| {
        if first {
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
            e.set_pipeline(&init);
            e.set_buffer(0, &rays, 0);
            e.set_buffer_mut(1, &mut stack, 0);
            e.set_buffer_mut(2, &mut ledger, 0);
            e.set_bytes(3, &r.settings);
            e.set_bytes(4, &params(0, (0, 0)));
            e.set_buffer_mut(5, &mut lists, 0);
            e.set_buffer_mut(6, &mut *counters, 0);
            e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
        }
        for b in 0..bounces {
            let par = parity ^ (b as u32 & 1);
            let p = params(par, (0, 0));
            // A tiled cast and interaction dispatch a fixed grid over a
            // range of the list instead of the whole list indirectly.
            let tp = tile.map(|t| params(par, t));
            if phases & 1 != 0 {
                // Pop: list `par` into list `1 - par` (cleared first).
                e.set_pipeline(&args_pso);
                e.set_buffer_mut(0, &mut *counters, 0);
                e.set_buffer_mut(1, &mut args, 0);
                e.set_bytes(2, &[par, widths[0], 0, 1 - par]);
                e.dispatch_groups([1, 1, 1], 1);
                e.set_pipeline(&pop);
                e.set_buffer_mut(0, &mut stack, 0);
                e.set_buffer_mut(1, &mut ledger, 0);
                e.set_buffer_mut(2, &mut current, 0);
                e.set_bytes(3, &r.settings);
                e.set_bytes(4, &p);
                e.set_buffer_mut(5, &mut lists, 0);
                e.set_buffer_mut(6, &mut *counters, 0);
                e.dispatch_indirect(&pop, &args, 0);
                // Arguments of the cast and interaction over list `1 - par`.
                e.set_pipeline(&args_pso);
                e.set_buffer_mut(0, &mut *counters, 0);
                e.set_buffer_mut(1, &mut args, 0);
                e.set_bytes(2, &[1 - par, widths[1], 3, 2]);
                e.dispatch_groups([1, 1, 1], 1);
                e.set_bytes(2, &[1 - par, widths[2], 6, 2]);
                e.dispatch_groups([1, 1, 1], 1);
            }
            if phases & 2 != 0 {
                e.set_pipeline(&cast);
                for (i, v) in [&r.primitives, &r.nodes, &r.shapes].into_iter().enumerate() {
                    let (b, o) = reads(v);
                    e.set_buffer(i, b, o);
                }
                e.set_buffer(3, &current, 0);
                e.set_buffer_mut(4, &mut found, 0);
                e.set_bytes(5, tp.as_ref().unwrap_or(&p));
                e.set_buffer(6, &lists, 0);
                e.set_buffer(7, &*counters, 0);
                match tile {
                    Some((_, len)) => e.dispatch_groups([(len as usize).div_ceil(64), 1, 1], 64),
                    None => e.dispatch_indirect(&cast, &args, 3),
                }
            }
            if phases & 4 != 0 {
                e.set_pipeline(&interact);
                for (i, v) in [
                    &r.primitives,
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
                    let (b, o) = reads(v);
                    e.set_buffer(i, b, o);
                }
                e.set_buffer_mut(7, &mut stack, 0);
                e.set_buffer_mut(8, &mut hits, 0);
                e.set_buffer_mut(9, &mut ledger, 0);
                e.set_bytes(10, &r.settings);
                e.set_bytes(11, tp.as_ref().unwrap_or(&p));
                e.set_buffer(12, &current, 0);
                e.set_buffer(13, &found, 0);
                e.set_buffer(14, &lists, 0);
                e.set_buffer(15, &*counters, 0);
                match tile {
                    Some((_, len)) => e.dispatch_groups([(len as usize).div_ceil(64), 1, 1], 64),
                    None => e.dispatch_indirect(&interact, &args, 6),
                }
            }
        }
        Ok(())
    };
    let (mut parity, mut bounces, mut commands, mut parameter_bytes) = (0u32, 1usize, 0u64, 0u64);
    // Bounded commands: a bounce whose casts may exceed the command time
    // (long iterative sag scans) runs tiled until one fits; the first is
    // tiled until the cost of a cast is known.
    let mut tiled: Option<usize> = Some(2048);
    let mut tile_commands = 0u32;
    loop {
        ctl.check()?;
        let wall = Instant::now();
        // SAFETY: records are boundary-packed; every kernel bounds its
        // thread index by the device count of its list, list entries are
        // distinct sources, and source-local stack and journal slices are
        // disjoint.
        let gpu_seconds = if let Some(size) = tiled {
            // One bounce: the pop, then the cast and interaction of the
            // popped list in tiles of bounded device time.
            let mut total = unsafe {
                gpu.run("stray wavefront pop", |e| {
                    encode(e, &mut counters, commands == 0, 1, parity, 1, None)
                })?
            }
            .gpu_seconds;
            let popped = counters.as_slice()[1 - parity as usize] as usize;
            let (mut first, mut size) = (0, size);
            while first < popped {
                ctl.check()?;
                let len = size.min(popped - first);
                let t = unsafe {
                    gpu.run("stray wavefront tile", |e| {
                        encode(
                            e,
                            &mut counters,
                            false,
                            1,
                            parity,
                            6,
                            Some((first as u32, len as u32)),
                        )
                    })?
                }
                .gpu_seconds;
                total += t;
                first += len;
                tile_commands += 1;
                size = ((size as f64 * TARGET_COMMAND_SECONDS / t.max(1e-5)) as usize)
                    .clamp(256, 1 << 20);
            }
            tiled = Some(size);
            bounces = 1;
            total
        } else {
            unsafe {
                gpu.run("stray wavefront", |e| {
                    encode(e, &mut counters, commands == 0, bounces, parity, 7, None)
                })?
            }
            .gpu_seconds
        };
        let seconds = wall.elapsed().as_secs_f64();
        parity ^= bounces as u32 & 1;
        commands += 1;
        parameter_bytes += (bounces * (2 * std::mem::size_of::<Settings>() + 180)) as u64;
        stats.device_seconds += gpu_seconds;
        stats.command_buffers += 1 + std::mem::take(&mut tile_commands);
        stats.max_command_seconds = stats.max_command_seconds.max(seconds);
        let live = counters.as_slice()[parity as usize] as usize;
        if live == 0 {
            break;
        }
        // About TARGET_COMMAND_SECONDS of device time per command: several
        // bounces per command while a bounce is short, tiles while it is
        // long.
        let per = gpu_seconds.max(1e-5) / bounces as f64;
        if per > 4. * TARGET_COMMAND_SECONDS {
            let size = (live as f64 * TARGET_COMMAND_SECONDS / per) as usize;
            tiled = Some(size.clamp(256, 1 << 20));
        } else if per < 0.5 * TARGET_COMMAND_SECONDS {
            tiled = None;
        }
        bounces = (TARGET_COMMAND_SECONDS / per).clamp(1., 256.) as usize;
        ctl.report(1. - live as f64 / n as f64, "stray wavefront");
    }
    drop(encode);
    x.stats.parameter_bytes += parameter_bytes;
    ctl.report(1., "stray transport");
    x.put_output(r.stack, stack);
    for b in [rays, current, found, cdf, lists, counters, args] {
        x.recycle(b);
    }
    x.put_output(r.output, hits);
    x.put_output(r.ledger, ledger);
    stats.wall_seconds = start.elapsed().as_secs_f64();
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
