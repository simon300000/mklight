//! Bounded resident coating/PRT commands. Published input leases stay immutable.
use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, radiometry::*, resident::StageStats};
use std::time::Instant;
/// Bare coating queries are cheap; film stacks and PRT need an earlier timing
/// checkpoint before their per-item cost is known.
const CHEAP_COATING_UNMEASURED: usize = 1 << 20;
const RADIOMETRY_UNMEASURED: usize = 1 << 14;
const RADIOMETRY_TILE: usize = 1 << 15;
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stats = StageStats {
        name: match r {
            Request::Coating(_) => "coating",
            Request::PupilChannels { .. } => "complex pupil channels",
            Request::Polarization(r) => {
                if r.scalar {
                    "scalar transmission"
                } else {
                    "polarization"
                }
            }
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
                unsafe {
                    x.run("complex pupil channels", |e| {
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
            let (cap, unmeasured) = if r.layers.desc.shape[0] == 0 {
                (r.tile, CHEAP_COATING_UNMEASURED)
            } else {
                (r.tile.min(RADIOMETRY_TILE), RADIOMETRY_UNMEASURED)
            };
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
            let mut tile = cap.min(4096);
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
                unsafe {
                    x.run("coating", |e| {
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
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 32;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "coating");
                if offset < n {
                    tile = next_tile(x, tile, cap, n - offset, unmeasured)?;
                }
            }
            x.count_fallbacks(counter);
            x.put_output(r.output, out);
        }
        Request::Polarization(r) => {
            let n = r.records.desc.shape[1];
            let cap = r.tile.min(RADIOMETRY_TILE);
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
            let mut tile = cap.min(4096);
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
                unsafe {
                    x.run("polarization", |e| {
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
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 72;
                ctl.check()?;
                offset = end;
                ctl.report(offset as f64 / n as f64, "polarization");
                if offset < n {
                    tile = next_tile(x, tile, cap, n - offset, RADIOMETRY_UNMEASURED)?;
                }
            }
            x.count_fallbacks(counter);
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
                unsafe {
                    x.run("radiometry channels", |e| {
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
                stats.max_command_seconds =
                    stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
                x.stats.parameter_bytes += 8;
                ctl.check()?;
            }
            x.put_output(*output, out);
        }
    }
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(stats);
    Ok(())
}

/// The next tile when `remaining` items follow. If the work bound completed
/// the batch holding this tile, or the next tile would exceed `unmeasured`
/// items (forcing a completion), the tile grows (at most eightfold) or shrinks
/// towards 10 ms of measured device time; that time includes earlier fused
/// work, so it errs towards smaller tiles. Otherwise it grows eightfold
/// unmeasured, so small and mid-sized operations stay fused with their
/// neighbours. Arithmetic is per item and independent of tile boundaries.
pub(crate) fn next_tile(
    x: &mut Execution<'_>,
    tile: usize,
    cap: usize,
    remaining: usize,
    unmeasured: usize,
) -> MkResult<usize> {
    let cap = cap.min(remaining).max(1);
    let grown = (8 * tile).min(cap);
    let seconds = match x.observed() {
        Some(seconds) => seconds,
        None if grown <= unmeasured => return Ok(grown),
        None => x.measure()?,
    };
    Ok(((tile as f64 * (0.010 / seconds.max(1e-6)).clamp(0.125, 8.0)) as usize).clamp(1, cap))
}
