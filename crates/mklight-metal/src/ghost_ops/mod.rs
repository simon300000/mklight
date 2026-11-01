//! Bounded ghost commands; all path steps share one geometry dispatch.
use crate::{context::PipelineKey, resident::Execution};
use bytemuck::{Pod, Zeroable};
use mklight_core::MkResult;
use mklight_gpu::{ghost_ops::*, layout::TraceParams, pending::JobControl, resident::*};
use std::time::Instant;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceParameters {
    trace: TraceParams,
    end: u32,
    offset: u32,
    paths: u32,
    wavelengths: u32,
    reject_virtual: u32,
    reserved: u32,
    sensor: [mklight_gpu::Df32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WeightParameters {
    end: u32,
    offset: u32,
    paths: u32,
    max_steps: u32,
    sensor: mklight_gpu::Df32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelParameters {
    end: u32,
    offset: u32,
    nx: u32,
    ny: u32,
    maps: u32,
    reserved: u32,
    half: [mklight_gpu::Df32; 2],
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stats = StageStats {
        name: "ghost transport".into(),
        ..Default::default()
    };
    match r {
        Request::MomentTerms { hits, output } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_moment_terms"))?;
            let mut out = x.take_output(*output);
            let n = hits.desc.shape[0];
            for offset in (0..n).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: a pure transpose from validated leaf records.
                let cs = unsafe {
                    x.gpu.run("ghost field moments", |e| {
                        e.set_pipeline(&pso);
                        let (b, o) = x.buffer(hits);
                        e.set_buffer(0, b, o);
                        e.set_buffer_mut(1, &mut out, 0);
                        e.set_bytes(
                            2,
                            &[(offset + 262144).min(n) as u32, offset as u32, n as u32, 0],
                        );
                        e.dispatch_groups([(n - offset).min(262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*output, out);
        }

        Request::SegmentFlags { keys, bins, output } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_segment_flags"))?;
            let mut out = x.take_output(*output);
            let n = keys.desc.len()?;
            for offset in (0..n).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: one flag per sorted key in validated storage.
                let cs = unsafe {
                    x.gpu.run("ghost segment flags", |e| {
                        e.set_pipeline(&pso);
                        let (b, o) = x.buffer(keys);
                        e.set_buffer(0, b, o);
                        e.set_buffer_mut(1, &mut out, 0);
                        e.set_bytes(
                            2,
                            &[
                                (offset + 262144).min(n) as u32,
                                offset as u32,
                                *bins as u32,
                                0,
                            ],
                        );
                        e.dispatch_groups([(n - offset).min(262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*output, out);
        }
        Request::PixelSegments {
            keys,
            prefix,
            bins,
            offsets,
            pixels,
            count,
        } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_pixel_segments"))?;
            let (mut os, mut ps, mut cs) = (
                x.take_output(*offsets),
                x.take_output(*pixels),
                x.take_output(*count),
            );
            cs.as_mut_slice().fill(0);
            let n = keys.desc.len()?;
            for offset in (0..n.max(1)).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: prefix IDs give disjoint segment records; only
                // integer maxima share an atomic counter.
                let cmd = unsafe {
                    x.gpu.run("ghost pixel segments", |e| {
                        e.set_pipeline(&pso);
                        for (j, v) in [keys, prefix].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j, b, o);
                        }
                        e.set_buffer_mut(2, &mut os, 0);
                        e.set_buffer_mut(3, &mut ps, 0);
                        e.set_buffer_mut(4, &mut cs, 0);
                        e.set_bytes(5, &[n as u32, offset as u32, *bins as u32, 0]);
                        e.dispatch_groups([(n - offset).clamp(1, 262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cmd.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*offsets, os);
            x.put_output(*pixels, ps);
            x.put_output(*count, cs);
        }
        Request::MapScatter {
            values,
            pixels,
            output,
        } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_map_scatter"))?;
            let mut out = x.take_output(*output);
            let nb = out.as_slice().len() / 2;
            let n = pixels.desc.len()?;
            for (mode, total) in [(0, nb), (1, n)] {
                for offset in (0..total).step_by(262144) {
                    ctl.check()?;
                    let wall = Instant::now();
                    // SAFETY: zero pass precedes unique pixel scatter writes.
                    let cs = unsafe {
                        x.gpu.run("ghost map scatter", |e| {
                            e.set_pipeline(&pso);
                            for (j, v) in [values, pixels].iter().enumerate() {
                                let (b, o) = x.buffer(v);
                                e.set_buffer(j, b, o);
                            }
                            e.set_buffer_mut(2, &mut out, 0);
                            e.set_bytes(
                                3,
                                &[(offset + 262144).min(total) as u32, offset as u32, mode, 0],
                            );
                            e.dispatch_groups(
                                [(total - offset).min(262144).div_ceil(256), 1, 1],
                                256,
                            );
                            Ok(())
                        })?
                    };
                    stats.device_seconds += cs.gpu_seconds;
                    stats.command_buffers += 1;
                    stats.max_command_seconds =
                        stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                    ctl.check()?;
                }
            }
            x.put_output(*output, out);
        }

        Request::FirstOrder {
            steps,
            offsets,
            initial,
            output,
        } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_first_order"))?;
            let mut out = x.take_output(*output);
            let n = offsets.desc.shape[0] - 1;
            for offset in (0..n).step_by(2048) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: each validated path owns a disjoint record range.
                let cs = unsafe {
                    x.gpu.run("ghost first order", |e| {
                        e.set_pipeline(&pso);
                        for (j, v) in [steps, offsets, initial].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j, b, o);
                        }
                        e.set_buffer_mut(3, &mut out, 0);
                        e.set_bytes(4, &[(offset + 2048).min(n) as u32, offset as u32]);
                        e.dispatch_groups([(n - offset).min(2048).div_ceil(64), 1, 1], 64);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*output, out);
        }
        Request::Gather {
            sources,
            leaves,
            output,
        } => {
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_ghost_gather"))?;
            let mut out = x.take_output(*output);
            let (b, o) = x.buffer(leaves);
            let words = &b.as_slice()[o..o + leaves.desc.len()?];
            let rows = bytemuck::cast_slice::<_, Leaf>(words);
            let ranges = (0..sources.len())
                .map(|batch| {
                    let start = rows.partition_point(|l| l.batch < batch as u32);
                    let end = rows.partition_point(|l| l.batch <= batch as u32);
                    (start, end)
                })
                .collect::<Vec<_>>();
            for group in ranges.chunks(32).enumerate() {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: stable leaves have distinct destination IDs; all
                // sources remain pinned until this command completes.
                let cs = unsafe {
                    x.gpu.run("ghost resident leaves", |e| {
                        e.set_pipeline(&pso);
                        let (b, o) = x.buffer(leaves);
                        e.set_buffer(1, b, o);
                        e.set_buffer_mut(2, &mut out, 0);
                        for (j, &(start, end)) in group.1.iter().enumerate() {
                            if start == end {
                                continue;
                            }
                            let (b, o) = x.buffer(&sources[group.0 * 32 + j]);
                            e.set_buffer(0, b, o);
                            e.set_bytes(3, &[(end - start) as u32, start as u32]);
                            e.dispatch_groups([(end - start).div_ceil(128), 1, 1], 128);
                        }
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*output, out);
        }
        Request::PixelKeys {
            hits,
            bins,
            maps,
            half,
            keys,
            ids,
            weights,
        } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_pixel_keys"))?;
            let (mut ks, mut is, mut ws) = (
                x.take_output(*keys),
                x.take_output(*ids),
                x.take_output(*weights),
            );
            let n = hits.desc.shape[0];
            for offset in (0..n).step_by(262144) {
                ctl.check()?;
                let wall = Instant::now();
                let prm = PixelParameters {
                    end: (offset + 262144).min(n) as u32,
                    offset: offset as u32,
                    nx: bins[0],
                    ny: bins[1],
                    maps: *maps as u32,
                    reserved: 0,
                    half: *half,
                };
                // SAFETY: disjoint pixel-key and weight rows in validated outputs.
                let cs = unsafe {
                    x.gpu.run("ghost pixel keys", |e| {
                        e.set_pipeline(&pso);
                        let (b, o) = x.buffer(hits);
                        e.set_buffer(0, b, o);
                        e.set_buffer_mut(1, &mut ks, 0);
                        e.set_buffer_mut(2, &mut is, 0);
                        e.set_buffer_mut(3, &mut ws, 0);
                        e.set_bytes(4, &[prm]);
                        e.dispatch_groups([(n - offset).min(262144).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.put_output(*keys, ks);
            x.put_output(*ids, is);
            x.put_output(*weights, ws);
        }
        Request::Launch {
            chiefs,
            fields,
            samples,
            output,
        } => {
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_ghost_launch"))?;
            let mut out = x.take_output(*output);
            let n = samples.desc.shape[0];
            for offset in (0..n).step_by(65536) {
                ctl.check()?;
                let wall = Instant::now();
                // SAFETY: immutable metadata and one writer per validated probe record.
                let cs = unsafe {
                    x.gpu.run("ghost launch", |e| {
                        e.set_pipeline(&pso);
                        for (j, v) in [chiefs, fields, samples].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j, b, o);
                        }
                        e.set_buffer_mut(3, &mut out, 0);
                        e.set_bytes(
                            4,
                            &[
                                (offset + 65536).min(n) as u32,
                                offset as u32,
                                fields.desc.shape[0] as u32,
                                0,
                            ],
                        );
                        e.dispatch_groups([(n - offset).min(65536).div_ceil(128), 1, 1], 128);
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
        }
        Request::Trace {
            systems,
            probes,
            offsets,
            steps,
            wavelengths,
            reject_virtual,
            sensor,
            newton,
            output,
            queries,
            factors,
            max_steps,
        } => {
            let SystemSet::Uploaded(handle) = systems else {
                return unsupported("ghost resident candidates");
            };
            let up = x.system(*handle)?;
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_ghost_trace"))?;
            let count_pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_reference_count"))?;
            let mut counter = x.allocate(1)?;
            counter.as_mut_slice()[0] = 0;
            let (mut out, mut qs, mut fs) = (
                x.take_output(*output),
                x.take_output(*queries),
                x.take_output(*factors),
            );
            let n = probes.desc.shape[0];
            let mut offset = 0;
            let mut tile = 2048;
            while offset < n {
                ctl.check()?;
                let end = (offset + tile).min(n);
                let wall = Instant::now();
                let prm = TraceParameters {
                    trace: TraceParams {
                        num_steps: *max_steps as u32,
                        max_iterations: newton.max_iterations,
                        tolerance_abs: newton.tolerance_abs,
                        tolerance_rel: newton.tolerance_rel,
                        ..Default::default()
                    },
                    end: end as u32,
                    offset: offset as u32,
                    paths: (offsets.desc.shape[0] - 1) as u32,
                    wavelengths: *wavelengths as u32,
                    reject_virtual: *reject_virtual as u32,
                    reserved: 0,
                    sensor: *sensor,
                };
                // SAFETY: validated record extents, immutable imports, exclusive outputs.
                let cs = unsafe {
                    x.gpu.run("ghost probes", |e| {
                        e.set_pipeline(&pso);
                        e.set_buffer(0, &up.headers, 0);
                        e.set_buffer(1, &up.surfaces, 0);
                        e.set_buffer(2, &up.coefficients, 0);
                        for (j, v) in [probes, offsets, steps].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j + 3, b, o);
                        }
                        e.set_buffer_mut(6, &mut out, 0);
                        e.set_buffer_mut(7, &mut qs, 0);
                        e.set_buffer_mut(8, &mut fs, 0);
                        e.set_bytes(9, &[prm]);
                        e.dispatch_groups([(end - offset).div_ceil(64), 1, 1], 64);
                        e.set_pipeline(&count_pso);
                        e.set_buffer(0, &out, 0);
                        e.set_buffer_mut(1, &mut counter, 0);
                        e.set_bytes(2, &[end as u32, offset as u32]);
                        e.dispatch_groups([(end - offset).div_ceil(256), 1, 1], 256);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += size_of::<TraceParameters>() as u64;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "ghost probes");
                if cs.gpu_seconds < 0.005 {
                    tile = (tile * 2).min(16384);
                } else if cs.gpu_seconds > 0.02 {
                    tile = (tile / 2).max(64);
                }
            }
            x.stats.fallback_count += counter.as_slice()[0] as u64;
            x.stats.readback_bytes += 4;
            x.recycle(counter);
            x.put_output(*output, out);
            x.put_output(*queries, qs);
            x.put_output(*factors, fs);
        }
        Request::Weights {
            values,
            coatings,
            factors,
            offsets,
            steps,
            probes,
            wavelengths: _,
            max_steps,
            sensor_reflectance,
            output,
        } => {
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_ghost_weights"))?;
            let mut out = x.take_output(*output);
            let n = probes.desc.shape[0];
            for offset in (0..n).step_by(16384) {
                ctl.check()?;
                let end = (offset + 16384).min(n);
                let wall = Instant::now();
                let prm = WeightParameters {
                    end: end as u32,
                    offset: offset as u32,
                    paths: (offsets.desc.shape[0] - 1) as u32,
                    max_steps: *max_steps as u32,
                    sensor: *sensor_reflectance,
                };
                // SAFETY: one output per probe; typed inputs validated at submission.
                let cs = unsafe {
                    x.gpu.run("ghost weights", |e| {
                        e.set_pipeline(&pso);
                        for (j, v) in [values, coatings, factors, offsets, steps, probes]
                            .iter()
                            .enumerate()
                        {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j, b, o);
                        }
                        e.set_buffer_mut(6, &mut out, 0);
                        e.set_bytes(7, &[prm]);
                        e.dispatch_groups([(end - offset).div_ceil(64), 1, 1], 64);
                        Ok(())
                    })?
                };
                stats.device_seconds += cs.gpu_seconds;
                stats.command_buffers += 1;
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += size_of::<WeightParameters>() as u64;
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
