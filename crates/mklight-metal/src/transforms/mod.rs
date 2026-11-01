//! Metal resident transforms. Recipe steps are encoded serially into few
//! command buffers; scratch is reused within the plan and recycled at the end.
use crate::{
    buffer::SharedBuffer,
    context::PipelineKey,
    resident::{Execution, Lease},
};
use mklight_core::MkResult;
use mklight_gpu::{
    consts::*,
    pending::JobControl,
    resident::{StageStats, unsupported},
    transforms::{recipe::*, *},
};
use objc2_metal::MTLComputePipelineState;
use std::{collections::HashMap, sync::Arc, time::Instant};
mod correlation;

/// Complex elements touched per command buffer: a 2048² FFT is one command,
/// and every command stays far below the 50 ms cancellation target.
const COMMAND_WORK: usize = 1 << 25;

pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let (device, commands) = (x.stats.device_seconds, x.stats.command_buffers);
    let operation = x.stats.stages.last().map_or(0, |s| s.operation + 1);
    let max_command = match r {
        Request::Correlation(_)
        | Request::OpticalCorrelation(_)
        | Request::PrepareOpticalStencil(_) => correlation::execute(r, x, ctl)?,
        _ => transform(r, x, ctl)?,
    };
    x.stats.stages.push(StageStats {
        operation,
        name: "transforms".into(),
        wall_seconds: start.elapsed().as_secs_f64(),
        device_seconds: x.stats.device_seconds - device,
        command_buffers: x.stats.command_buffers - commands,
        max_command_seconds: max_command,
    });
    Ok(())
}

enum Buf {
    Input(usize),
    Owned(Lease),
    /// A new immutable result, cached once its command completes.
    Fresh(SharedBuffer<u32>),
    Cached(Arc<SharedBuffer<u32>>),
}
/// Local row kernels by [fused chirp][mixed radix][shared elements 1k/2k/4k].
const ROW_KERNELS: [&str; 12] = [
    "mkl_fft_rows_1k",
    "mkl_fft_rows_2k",
    "mkl_fft_rows_4k",
    "mkl_fft_rows_1k_mixed",
    "mkl_fft_rows_2k_mixed",
    "mkl_fft_rows_4k_mixed",
    "mkl_chirp_rows_1k",
    "mkl_chirp_rows_2k",
    "mkl_chirp_rows_4k",
    "mkl_chirp_rows_1k_mixed",
    "mkl_chirp_rows_2k_mixed",
    "mkl_chirp_rows_4k_mixed",
];
/// Shared-memory elements per group for rows of n: whole rows, ≥1024.
fn group_elements(n: usize) -> usize {
    n.next_power_of_two().max(1024)
}

fn transform(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<f64> {
    let access = accesses(r);
    let key = x.cache_key(format!("transform-recipe:{r:?}"));
    let recipe = match x.cached::<Recipe>(&key)? {
        Some(v) => v,
        None => {
            let v = Arc::new(request(r)?);
            x.cache(key, v.steps.len() * size_of::<Step>(), v.clone());
            v
        }
    };
    let steps = &recipe.steps;
    let names = recipe.cache_keys();
    let keys: Vec<_> = names.iter().map(|k| x.cache_key(k.clone())).collect();
    // bufs[slot_buf[slot]] holds each slot; dead owned buffers are reused.
    let mut bufs: Vec<Option<Buf>> = vec![
        Some(Buf::Input(0)),
        Some(Buf::Input(access.reads.len() - 1)),
    ];
    let mut slot_buf: Vec<usize> = vec![0, 1];
    slot_buf.resize(steps.len() + 2, usize::MAX);
    // Resolve cached spectra, then mark the steps the output still needs.
    let mut needed = vec![false; steps.len() + 2];
    needed[recipe.output] = true;
    for i in (0..steps.len()).rev() {
        if !needed[i + 2] {
            continue;
        }
        if steps[i].cache
            && let Some(v) = x.cached::<SharedBuffer<u32>>(&keys[i])?
        {
            slot_buf[i + 2] = bufs.len();
            bufs.push(Some(Buf::Cached(v)));
            continue;
        }
        for d in dependencies(&steps[i]) {
            needed[d] = true;
        }
    }
    let active: Vec<bool> = (0..steps.len())
        .map(|i| needed[i + 2] && slot_buf[i + 2] == usize::MAX)
        .collect();
    let mut uses = vec![0usize; steps.len() + 2];
    for (_, s) in steps.iter().enumerate().filter(|(i, _)| active[*i]) {
        for d in dependencies(s) {
            uses[d] += 1;
        }
    }
    // Immutable f64-derived tables: twiddles per row length, chirp factors per step.
    let table = |x: &mut Execution<'_>,
                 name: String,
                 len: usize,
                 fill: &dyn Fn(&mut [[f32; 2]])|
     -> MkResult<Arc<SharedBuffer<u32>>> {
        let key = x.cache_key(name);
        if let Some(v) = x.cached::<SharedBuffer<u32>>(&key)? {
            return Ok(v);
        }
        let v = Arc::new(
            x.gpu
                .buffer_filled::<u32>(len * 2, |out| fill(bytemuck::cast_slice_mut(out)))?,
        );
        x.stats.allocations += 1;
        x.stats.uploaded_bytes += len as u64 * 8;
        x.cache(key, (len * 8).max(16), v.clone());
        Ok(v)
    };
    let mut tables = HashMap::new();
    let mut chirps = HashMap::new();
    for (i, s) in steps.iter().enumerate().filter(|(i, _)| active[*i]) {
        let w = s.p.width as usize;
        if matches!(s.p.mode, MKL_TRANSFORM_FFT_ROWS | MKL_TRANSFORM_CHIRP_ROWS)
            && !tables.contains_key(&w)
        {
            tables.insert(
                w,
                table(x, format!("p2-twiddles:{w}"), w - 1, &|out| {
                    fill_twiddles(w, out)
                })?,
            );
        }
        if s.p.mode == MKL_TRANSFORM_CHIRP_ROWS {
            chirps.insert(
                i,
                table(
                    x,
                    format!("{}:chirp", names[i]),
                    (s.p.nx + s.p.ox) as usize,
                    &|out| fill_chirp_factors(&s.p, out),
                )?,
            );
        }
    }
    let pso = |name: &'static str| x.gpu.pipeline(&PipelineKey::plain(name));
    let (elementwise, transpose, stage) = (
        pso("mkl_transform")?,
        pso("mkl_transform_transpose")?,
        pso("mkl_fft_stage")?,
    );
    let rows = ROW_KERNELS.map(pso);
    let rows = rows.into_iter().collect::<MkResult<Vec<_>>>()?;
    // Row groups use at most 256 threads (SH/E).
    if rows.iter().any(|p| p.maxTotalThreadsPerThreadgroup() < 256) {
        return unsupported("transform threadgroup size");
    }
    let mut free: Vec<usize> = vec![];
    let mut temps: Vec<Option<usize>> = vec![None; steps.len()];
    let mut out_buf = None;
    let mut max_command = 0f64;
    let mut i = 0;
    while i < steps.len() {
        ctl.check()?;
        // Allocate a bounded batch, reusing buffers freed earlier in plan order.
        let (first, mut work) = (i, 0);
        while i < steps.len() && (i == first || work + steps[i].p.count as usize <= COMMAND_WORK) {
            if !active[i] {
                i += 1;
                continue;
            }
            let s = &steps[i];
            let words = s.p.count as usize * 2;
            let mut take =
                |x: &mut Execution<'_>, bufs: &mut Vec<Option<Buf>>| -> MkResult<usize> {
                    if let Some(at) = free
                        .iter()
                        .position(|&b| matches!(&bufs[b], Some(Buf::Owned(l)) if l.len() >= words))
                    {
                        return Ok(free.swap_remove(at));
                    }
                    bufs.push(Some(Buf::Owned(x.allocate(words)?)));
                    Ok(bufs.len() - 1)
                };
            slot_buf[i + 2] = if i + 2 == recipe.output {
                bufs.push(Some(Buf::Owned(x.take_output(access.writes[0]))));
                out_buf = Some(bufs.len() - 1);
                bufs.len() - 1
            } else if s.cache && words * 4 <= 64 << 20 {
                // P0's immutable cache holds ≤64 MiB; larger spectra stay pooled scratch.
                x.stats.allocations += 1;
                bufs.push(Some(Buf::Fresh(x.gpu.buffer_zeroed(words)?)));
                bufs.len() - 1
            } else {
                take(x, &mut bufs)?
            };
            if s.p.mode == MKL_TRANSFORM_FFT_ROWS && s.p.width as usize > LOCAL_MAX {
                temps[i] = Some(take(x, &mut bufs)?);
            }
            for d in dependencies(s) {
                uses[d] -= 1;
                if d >= 2 && uses[d] == 0 && matches!(bufs[slot_buf[d]], Some(Buf::Owned(_))) {
                    free.push(slot_buf[d]);
                }
            }
            free.extend(temps[i]);
            work += s.p.count as usize;
            i += 1;
        }
        let xr = &*x;
        let wall = Instant::now();
        let mut dispatches = 0u64;
        // SAFETY: plan validation bounds every view; recipe extents size every
        // allocation; serial dispatch orders all reuse of scratch buffers.
        let cs = unsafe {
            xr.run("transforms", |e| {
                let bind = |bufs: &[Option<Buf>], index: usize, b: usize| match bufs[b]
                    .as_ref()
                    .unwrap()
                {
                    Buf::Input(r) => {
                        let (v, o) = xr.buffer(&access.reads[*r]);
                        e.set_buffer(index, v, o)
                    }
                    Buf::Owned(v) => e.set_buffer(index, v, 0),
                    Buf::Fresh(v) => e.set_buffer(index, v, 0),
                    Buf::Cached(v) => e.set_buffer(index, v, 0),
                };
                let bind_mut = |bufs: &mut [Option<Buf>], index: usize, b: usize| match bufs[b]
                    .as_mut()
                    .unwrap()
                {
                    Buf::Owned(v) => e.set_buffer_mut(index, v, 0),
                    Buf::Fresh(v) => e.set_buffer_mut(index, v, 0),
                    _ => unreachable!("only scratch is written"),
                };
                for s in first..i {
                    if !active[s] {
                        continue;
                    }
                    let st = &steps[s];
                    let p = st.p;
                    let (n, h) = (p.width as usize, p.height as usize);
                    let (a, dst) = (slot_buf[st.a], slot_buf[s + 2]);
                    dispatches += 1;
                    match p.mode {
                        MKL_TRANSFORM_FFT_ROWS if n > LOCAL_MAX => {
                            let tmp = temps[s].unwrap();
                            let stages = p.stages as usize;
                            let mut src = a;
                            let mut span = 1;
                            for k in 0..stages {
                                let r = p.radix(k);
                                let to = if (stages - 1 - k).is_multiple_of(2) {
                                    dst
                                } else {
                                    tmp
                                };
                                let mut q = p;
                                q.radices = [r as u32, 0];
                                q.span = span as u32;
                                if k + 1 < stages {
                                    q.scale_bits = 1f32.to_bits();
                                }
                                e.set_pipeline(&stage);
                                bind(&bufs, 0, src);
                                e.set_buffer(1, &tables[&n], 0);
                                bind_mut(&mut bufs, 2, to);
                                e.set_bytes(3, &q);
                                e.dispatch_groups([(n / r).div_ceil(256), h, 1], 256);
                                (src, span) = (to, span * r);
                                dispatches += 1;
                            }
                        }
                        MKL_TRANSFORM_FFT_ROWS | MKL_TRANSFORM_CHIRP_ROWS => {
                            let sh = group_elements(n);
                            let chirped = p.mode == MKL_TRANSFORM_CHIRP_ROWS;
                            let mixed =
                                (0..p.stages as usize).any(|k| ![2, 4, 8].contains(&p.radix(k)));
                            let level = sh.trailing_zeros() as usize - 10;
                            e.set_pipeline(
                                &rows[6 * chirped as usize + 3 * mixed as usize + level],
                            );
                            bind(&bufs, 0, a);
                            if chirped {
                                bind(&bufs, 1, slot_buf[st.b]);
                                e.set_buffer(4, &tables[&n], 0);
                                e.set_buffer(5, &chirps[&s], 0);
                            } else {
                                e.set_buffer(1, &tables[&n], 0);
                            }
                            bind_mut(&mut bufs, 2, dst);
                            e.set_bytes(3, &p);
                            let per = if sh > 2048 { 16 } else { 8 };
                            let t = n.div_ceil(per);
                            let g = sh / (per * t);
                            e.dispatch_groups([h.div_ceil(g), 1, 1], g * t);
                        }
                        MKL_TRANSFORM_TRANSPOSE => {
                            e.set_pipeline(&transpose);
                            bind(&bufs, 0, a);
                            bind_mut(&mut bufs, 2, dst);
                            e.set_bytes(3, &p);
                            e.dispatch_groups([n.div_ceil(32), h.div_ceil(32), 1], 256);
                        }
                        _ => {
                            bind(&bufs, 0, a);
                            bind(&bufs, 1, slot_buf[st.b]);
                            bind_mut(&mut bufs, 2, dst);
                            e.set_bytes(3, &p);
                            e.dispatch_1d(&elementwise, p.count as usize);
                        }
                    }
                }
                Ok(())
            })?
        };
        max_command = max_command.max(wall.elapsed().as_secs_f64());
        x.stats.command_buffers += cs.command_buffers;
        x.stats.device_seconds += cs.gpu_seconds;
        x.stats.parameter_bytes += dispatches * size_of::<Params>() as u64;
        for s in first..i {
            let b = slot_buf[s + 2];
            if active[s] && matches!(bufs[b], Some(Buf::Fresh(_))) {
                let Some(Buf::Fresh(v)) = bufs[b].take() else {
                    unreachable!()
                };
                let v = Arc::new(v);
                x.cache(keys[s].clone(), v.len() * 4, v.clone());
                bufs[b] = Some(Buf::Cached(v));
            }
        }
        ctl.report(i as f64 / steps.len() as f64, "transform");
    }
    let Some(Buf::Owned(output)) = bufs[out_buf.unwrap()].take() else {
        unreachable!("the output is plan scratch")
    };
    x.put_output(access.writes[0], output);
    for b in bufs.into_iter().flatten() {
        if let Buf::Owned(v) = b {
            x.recycle(v);
        }
    }
    Ok(max_command)
}
