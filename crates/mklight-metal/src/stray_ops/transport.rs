use super::*;
use mklight_core::MkError;
use mklight_gpu::resident::unsupported;

pub(super) fn execute(r: &Transport, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let source_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_generate"))?;
    let transport = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_transport"))?;
    let spill = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_spill"))?;
    let n = r.ray_count;
    let mut capacity = r.stack_capacity.max(64);
    let words = |size: usize, count: usize| size / 4 * count;
    let mut rays = x.allocate(words(std::mem::size_of::<Ray>(), n))?;
    let mut stack = x.allocate(words(std::mem::size_of::<State>(), n * capacity))?;
    let mut cdf = x.allocate(2 * r.cdf.len())?;
    cdf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(&r.cdf));
    let mut hits = x.take_output(r.output);
    let mut ledger = x.take_output(r.ledger);
    let mut initial = true;
    let mut stats = StageStats {
        name: "stray transport".into(),
        ..Default::default()
    };
    let mut iterations = 0u64;
    loop {
        ctl.check()?;
        let wall = Instant::now();
        let p = [
            n as u32,
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
                if initial {
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
                e.set_buffer(8, &rays, 0);
                e.set_buffer_mut(9, &mut stack, 0);
                e.set_buffer_mut(10, &mut hits, 0);
                e.set_buffer_mut(11, &mut ledger, 0);
                e.set_bytes(12, &r.settings);
                e.set_bytes(13, &p);
                let (b, o) = x.buffer(&r.shapes);
                e.set_buffer(14, b, o);
                e.dispatch_groups([n.div_ceil(32), 1, 1], 32);
                Ok(())
            })?
        };
        initial = false;
        iterations += 1;
        stats.device_seconds += cs.gpu_seconds;
        stats.command_buffers += 1;
        stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.check()?;
        let rows: &[Ledger] = bytemuck::cast_slice(ledger.as_slice());
        let completed = rows.iter().filter(|l| l.status == 0).count();
        if rows.iter().any(|l| l.status == 4) {
            return unsupported("stray coating/order requires reference");
        }
        if rows.iter().any(|l| l.status == 3) {
            return unsupported("stray detection capacity; no partial result published");
        }
        if completed == n {
            break;
        }
        if rows.iter().any(|l| l.status == 2) {
            let next_capacity = capacity * 2;
            let size = words(std::mem::size_of::<State>(), n * next_capacity);
            if size * 4 > 256 * 1024 * 1024 {
                return Err(MkError::Unsupported(
                    "stray continuation stack exceeds 256 MiB; no paths discarded".into(),
                ));
            }
            let mut next = x.allocate(size)?;
            let cs = unsafe {
                x.gpu.run("stray stack spill", |e| {
                    e.set_pipeline(&spill);
                    e.set_buffer(0, &stack, 0);
                    e.set_buffer_mut(1, &mut next, 0);
                    e.set_buffer(2, &ledger, 0);
                    e.set_bytes(3, &[n as u32, capacity as u32, next_capacity as u32, 0]);
                    e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
                    Ok(())
                })?
            };
            stats.device_seconds += cs.gpu_seconds;
            stats.command_buffers += 1;
            x.recycle(stack);
            stack = next;
            capacity = next_capacity;
        }
        // Completed sources never become active again. Reserve a tail for
        // long split trees instead of guessing their final path count.
        ctl.report(
            0.9 * completed as f64 / n as f64 + 0.09 * (1. - 1. / (iterations + 1) as f64),
            "stray continuation",
        );
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
