//! Bounded resident coating/PRT commands. Published input leases stay immutable.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, radiometry::*, resident::StageStats};
use std::time::Instant;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stats = StageStats {
        name: match r {
            Request::Coating(_) => "coating",
            Request::PupilChannels { .. } => "complex pupil channels",
            Request::Polarization(_) => "polarization",
            Request::Channels { .. } => "radiometry channels",
        }
        .into(),
        ..Default::default()
    };
    match r {
        Request::PupilChannels {
            values,
            vector,
            output,
            status,
        } => {
            let n = values.desc.shape[0];
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_radiometry_pupil_channels"))?;
            let mut out = x.take_output(*output);
            let mut flags = x.take_output(*status);
            for offset in (0..n).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: record extent, typed output and exclusive writers validated.
                let cs = unsafe {
                    x.gpu.run("complex pupil channels", |e| {
                        e.set_pipeline(&pso);
                        let (b, o) = x.buffer(values);
                        e.set_buffer(0, b, o);
                        e.set_buffer_mut(1, &mut out, 0);
                        e.set_buffer_mut(2, &mut flags, 0);
                        e.set_bytes(3, &[n as u32, offset as u32, *vector as u32, 0u32]);
                        e.dispatch_groups([(n - offset).min(262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 16;
                ctl.check()?;
            }
            x.put_output(*output, out);
            x.put_output(*status, flags);
        }
        Request::Coating(r) => {
            let n = r.queries.desc.shape[0];
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_radiometry_coating"))?;
            let mut out = x.take_output(r.output);
            let count_pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_radiometry_count"))?;
            let mut counter = x.allocate(1)?;
            counter.as_mut_slice()[0] = 0;
            let mut offset = 0;
            let mut tile = r.tile.min(4096);
            while offset < n {
                let end = (offset + tile).min(n);
                ctl.check()?;
                let wall = Instant::now();
                let params = [
                    end as u32,
                    offset as u32,
                    r.specs.desc.shape[0] as u32,
                    r.layers.desc.shape[0] as u32,
                ];
                // SAFETY: validated typed extents and single-writer output. Query
                // indices are checked in the kernel before dereferencing records.
                let cs = unsafe {
                    x.gpu.run("coating", |e| {
                        e.set_pipeline(&pso);
                        for (i, v) in [&r.specs, &r.layers, &r.queries].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(i, b, o);
                        }
                        e.set_buffer_mut(3, &mut out, 0);
                        e.set_bytes(4, &params);
                        e.dispatch_groups([(end - offset).div_ceil(64), 1, 1], 64);
                        e.set_pipeline(&count_pso);
                        e.set_buffer(0, &out, 0);
                        e.set_buffer_mut(1, &mut counter, 0);
                        e.set_bytes(
                            2,
                            &[
                                end as u32,
                                offset as u32,
                                (std::mem::size_of::<CoatingValue>() / 4) as u32,
                                (std::mem::offset_of!(CoatingValue, status) / 4) as u32,
                            ],
                        );
                        e.dispatch_groups([(end - offset).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 32;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "coating");
                tile = next_tile(tile, r.tile, cs.gpu_seconds);
            }
            x.stats.fallback_count += counter.as_slice()[0] as u64;
            x.stats.readback_bytes += 4;
            x.recycle(counter);
            x.put_output(r.output, out);
        }
        Request::Polarization(r) => {
            let n = r.records.desc.shape[1];
            let pso = x.gpu.pipeline(&PipelineKey::plain(if r.scalar {
                "mkl_radiometry_scalar"
            } else {
                "mkl_radiometry_polarization"
            }))?;
            let mut out = x.take_output(r.output);
            let count_pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_radiometry_count"))?;
            let mut counter = x.allocate(1)?;
            counter.as_mut_slice()[0] = 0;
            let mut offset = 0;
            let mut tile = r.tile.min(4096);
            while offset < n {
                let end = (offset + tile).min(n);
                ctl.check()?;
                let wall = Instant::now();
                let mut params = [0u32; 14];
                params[..4].copy_from_slice(&[
                    n as u32,
                    offset as u32,
                    r.records.desc.shape[0] as u32,
                    r.wavelengths as u32,
                ]);
                params[12] = r.layers.desc.shape[0] as u32;
                params[13] = end as u32;
                params[4..12].copy_from_slice(bytemuck::cast_slice(&r.stokes));
                // SAFETY: P1 record extents and output ownership were validated.
                let cs = unsafe {
                    x.gpu.run("polarization", |e| {
                        e.set_pipeline(&pso);
                        for (i, v) in [
                            &r.surfaces,
                            &r.layers,
                            &r.records,
                            &r.record_low,
                            &r.indices,
                            &r.trace_status,
                        ]
                        .iter()
                        .enumerate()
                        {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(i, b, o);
                        }
                        e.set_buffer_mut(6, &mut out, 0);
                        e.set_bytes(7, &params);
                        e.dispatch_groups([(end - offset).div_ceil(64), 1, 1], 64);
                        e.set_pipeline(&count_pso);
                        e.set_buffer(0, &out, 0);
                        e.set_buffer_mut(1, &mut counter, 0);
                        e.set_bytes(
                            2,
                            &[
                                end as u32,
                                offset as u32,
                                (std::mem::size_of::<PolarizationValue>() / 4) as u32,
                                (std::mem::offset_of!(PolarizationValue, status) / 4) as u32,
                            ],
                        );
                        e.dispatch_groups([(end - offset).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 72;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "polarization");
                tile = next_tile(tile, r.tile, cs.gpu_seconds);
            }
            x.stats.fallback_count += counter.as_slice()[0] as u64;
            x.stats.readback_bytes += 4;
            x.recycle(counter);
            x.put_output(r.output, out);
        }
        Request::Channels {
            values,
            trace_status,
            output,
        } => {
            let n = values.desc.shape[0];
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_radiometry_channels"))?;
            let mut out = x.take_output(*output);
            for offset in (0..n).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: validated channels and a single output writer.
                let cs = unsafe {
                    x.gpu.run("radiometry channels", |e| {
                        e.set_pipeline(&pso);
                        for (i, v) in [values, trace_status].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(i, b, o);
                        }
                        e.set_buffer_mut(2, &mut out, 0);
                        e.set_bytes(3, &[n as u32, offset as u32]);
                        e.dispatch_groups([(n - offset).min(262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 8;
                ctl.check()?;
            }
            x.put_output(*output, out);
        }
    }
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}

// Aim for 10 ms, growing at most eightfold; arithmetic order is per ray and
// independent of these command boundaries. A slow stack reduces the next tile.
fn next_tile(tile: usize, cap: usize, seconds: f64) -> usize {
    ((tile as f64 * (0.010 / seconds.max(1e-6)).clamp(0.125, 8.0)) as usize).clamp(1, cap)
}
