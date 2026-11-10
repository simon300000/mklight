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
    specs: u32,
    layers: u32,
    reserved: u32,
    sensor: [mklight_gpu::Df32; 2],
    sensor_reflectance: mklight_gpu::Df32,
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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FirstOrderParameters {
    end: u32,
    offset: u32,
    sensor: mklight_gpu::Df32,
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
                unsafe {
                    x.run("ghost field moments", |e| {
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
                unsafe {
                    x.run("ghost segment flags", |e| {
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
                unsafe {
                    x.run("ghost pixel segments", |e| {
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
            let nb = out.len() / 2;
            let n = pixels.desc.len()?;
            for (mode, total) in [(0, nb), (1, n)] {
                for offset in (0..total).step_by(262144) {
                    ctl.check()?;
                    let wall = Instant::now();
                    // SAFETY: zero pass precedes unique pixel scatter writes.
                    unsafe {
                        x.run("ghost map scatter", |e| {
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
            coatings,
            sensor_reflectance,
            output,
            throughput,
            status,
        } => {
            let pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_first_order"))?;
            let (mut out, mut rt, mut flags) = (
                x.take_output(*output),
                x.take_output(*throughput),
                x.take_output(*status),
            );
            let n = offsets.desc.shape[0] - 1;
            for offset in (0..n).step_by(2048) {
                ctl.check()?;
                let wall = Instant::now();
                let params = FirstOrderParameters {
                    end: (offset + 2048).min(n) as u32,
                    offset: offset as u32,
                    sensor: *sensor_reflectance,
                };
                // SAFETY: disjoint path records and independent radiometry folds.
                unsafe {
                    x.run("ghost first order", |e| {
                        e.set_pipeline(&pso);
                        for (j, v) in [steps, offsets, initial].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j, b, o);
                        }
                        e.set_buffer_mut(3, &mut out, 0);
                        let (b, o) = x.buffer(coatings);
                        e.set_buffer(4, b, o);
                        e.set_buffer_mut(5, &mut rt, 0);
                        e.set_buffer_mut(6, &mut flags, 0);
                        e.set_bytes(7, &[params]);
                        e.dispatch_groups([(n - offset).min(2048).div_ceil(64), 1, 1], 64);
                        Ok(())
                    })?
                };
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                ctl.check()?;
            }
            x.flush()?;
            for block in flags.as_slice()[..n].chunks(4096) {
                ctl.check()?;
                x.stats.fallback_count += block.iter().filter(|&&s| s != 0).count() as u64;
            }
            x.stats.readback_bytes += 4 * n as u64;
            x.put_output(*output, out);
            x.put_output(*throughput, rt);
            x.put_output(*status, flags);
        }
        Request::Gather {
            sources,
            leaves,
            output,
        } => {
            let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_ghost_gather"))?;
            let mut out = x.take_output(*output);
            x.flush()?;
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
                unsafe {
                    x.run("ghost resident leaves", |e| {
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
                unsafe {
                    x.run("ghost pixel keys", |e| {
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
                unsafe {
                    x.run("ghost launch", |e| {
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
            coatings,
            output,
        } => {
            use mklight_gpu::consts::*;
            let up = x.system(*systems)?;
            // Thin-film matrices are compiled in only when a film record exists.
            let features = coatings.as_ref().map_or(0, |c| {
                let (b, o) = x.buffer(&c.specs);
                let words = &b.as_slice()[o..o + c.specs.desc.len().unwrap_or(0)];
                let films = bytemuck::cast_slice::<_, mklight_gpu::radiometry::CoatingSpec>(words)
                    .iter()
                    .any(|s| s.kind == MKL_RAD_FILM);
                MKL_GHOST_FEATURE_COATINGS | if films { MKL_GHOST_FEATURE_FILMS } else { 0 }
            });
            let pso = x.gpu.pipeline(&PipelineKey::with_u32(
                "mkl_ghost_trace",
                MKL_GHOST_FEATURES_CONSTANT as u16,
                features,
            ))?;
            let count_pso = x
                .gpu
                .pipeline(&PipelineKey::plain("mkl_ghost_reference_count"))?;
            let mut counter = x.allocate(1)?;
            counter.as_mut_slice()[0] = 0;
            let mut out = x.take_output(*output);
            let n = probes.desc.shape[0];
            // Without coatings the kernel never reads its coating arguments.
            let (specs, layers) = match coatings {
                Some(c) => (&c.specs, &c.layers),
                None => (probes, probes),
            };
            let records = |v: &PlanView| v.desc.shape.first().copied().unwrap_or(0) as u32;
            let mut offset = 0;
            let mut tile = 8192;
            while offset < n {
                ctl.check()?;
                let end = (offset + tile).min(n);
                let wall = Instant::now();
                let prm = TraceParameters {
                    trace: TraceParams {
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
                    specs: coatings.as_ref().map_or(0, |c| records(&c.specs)),
                    layers: coatings.as_ref().map_or(0, |c| records(&c.layers)),
                    reserved: 0,
                    sensor: *sensor,
                    sensor_reflectance: coatings
                        .as_ref()
                        .map_or(mklight_gpu::Df32::ZERO, |c| c.sensor_reflectance),
                };
                // SAFETY: validated record extents, immutable imports, exclusive outputs.
                unsafe {
                    x.run("ghost probes", |e| {
                        e.set_pipeline(&pso);
                        e.set_buffer(0, &up.headers, 0);
                        e.set_buffer(1, &up.surfaces, 0);
                        e.set_buffer(2, &up.coefficients, 0);
                        for (j, v) in [probes, offsets, steps].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j + 3, b, o);
                        }
                        e.set_buffer_mut(6, &mut out, 0);
                        for (j, v) in [specs, layers].iter().enumerate() {
                            let (b, o) = x.buffer(v);
                            e.set_buffer(j + 7, b, o);
                        }
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
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += size_of::<TraceParameters>() as u64;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "ghost probes");
                if offset < n {
                    // Probes trace whole ghost paths (~0.1 µs each on the P9
                    // Double Gauss), so calibrate beyond 2^16 items.
                    tile = crate::radiometry::next_tile(x, tile, 1 << 20, n - offset, 1 << 16)?;
                }
            }
            x.count_fallbacks(counter);
            x.put_output(*output, out);
        }
    }
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(stats);
    Ok(())
}
