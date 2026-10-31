//! Metal resident FFT/chirp encoder. Each command owns bounded work; all
//! temporary leases survive completion and return to the resident pool.
use crate::{
    buffer::SharedBuffer,
    context::PipelineKey,
    resident::{Execution, Lease},
};
use mklight_core::MkResult;
use mklight_gpu::{
    pending::JobControl,
    transforms::{recipe::*, *},
};
use std::{sync::Arc, time::Instant};
mod correlation;
enum Node {
    Owned(Lease),
    Cached(Arc<SharedBuffer<u32>>),
}
impl Node {
    fn buffer(&self) -> &SharedBuffer<u32> {
        match self {
            Self::Owned(v) => v,
            Self::Cached(v) => v,
        }
    }
}

pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let device = x.stats.device_seconds;
    let commands = x.stats.command_buffers;
    let operation = x.stats.stages.last().map_or(0, |s| s.operation + 1);
    let max_command = if matches!(
        r,
        Request::Correlation(_)
            | Request::OpticalCorrelation(_)
            | Request::PrepareOpticalStencil(_)
    ) {
        correlation::execute(r, x, ctl)?
    } else {
        execute_transform(r, x, ctl)?
    };
    // P0 adds a coarse domain boundary after this handler. This detailed entry
    // supplies the operation's device time, commands and cancellation bound.
    x.stats.stages.push(mklight_gpu::resident::StageStats {
        operation,
        name: "transforms".into(),
        wall_seconds: start.elapsed().as_secs_f64(),
        device_seconds: x.stats.device_seconds - device,
        command_buffers: x.stats.command_buffers - commands,
        max_command_seconds: max_command,
    });
    Ok(())
}
fn execute_transform(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<f64> {
    let mut max_command = 0f64;
    let access = accesses(r);
    let key = x.cache_key(format!("transform-recipe:{r:?}"));
    let recipe = if let Some(v) = x.cached::<Recipe>(&key)? {
        v
    } else {
        let v = Arc::new(request(r)?);
        x.cache(key, v.steps.len() * size_of::<Step>(), v.clone());
        v
    };
    let mut slots: Vec<Option<Node>> = (0..recipe.steps.len() + 2).map(|_| None).collect();
    let mut needed = vec![false; slots.len()];
    needed[recipe.output] = true;
    let keys: Vec<_> = recipe
        .cache_keys()
        .into_iter()
        .map(|k| x.cache_key(k))
        .collect();
    for i in (0..recipe.steps.len()).rev() {
        if !needed[i + 2] {
            continue;
        }
        let s = &recipe.steps[i];
        if s.cache {
            if let Some(v) = x.cached::<SharedBuffer<u32>>(&keys[i])? {
                slots[i + 2] = Some(Node::Cached(v));
                continue;
            }
        }
        for d in dependencies(s) {
            needed[d] = true;
        }
    }
    let mut uses = vec![0usize; slots.len()];
    for (i, s) in recipe.steps.iter().enumerate() {
        if needed[i + 2] && slots[i + 2].is_none() {
            for d in dependencies(s) {
                uses[d] += 1;
            }
        }
    }
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_transform"))?;
    let transpose = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_transform_transpose"))?;
    let local_small = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_transform_rows_small"))?;
    let local_large = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_transform_rows_large"))?;
    let mut final_buffer = Some(x.take_output(access.writes[0]));
    for (i, s) in recipe.steps.iter().enumerate() {
        ctl.check()?;
        if needed[i + 2] && slots[i + 2].is_none() {
            let mut dst = if i + 2 == recipe.output {
                final_buffer.take().unwrap()
            } else {
                x.allocate(s.p.count as usize * 2)?
            };
            let binding = |id: usize| {
                if id < 2 {
                    x.buffer(&access.reads[id.min(access.reads.len() - 1)])
                } else {
                    (slots[id].as_ref().unwrap().buffer(), 0)
                }
            };
            let deps = dependencies(s);
            let (a, ao) = binding(if deps.is_empty() { 0 } else { s.a });
            let (b, bo) = binding(if deps.len() < 2 {
                if deps.is_empty() { 0 } else { s.a }
            } else {
                s.b
            });
            let mut commands = 0;
            let mut device = 0.;
            let tiled = s.p.mode == mklight_gpu::consts::MKL_TRANSFORM_TRANSPOSE;
            let local = s.p.mode == mklight_gpu::consts::MKL_TRANSFORM_LOCAL_FFT;
            let total = if local {
                s.p.height as usize
            } else if tiled {
                (s.p.width as usize).div_ceil(16) * (s.p.height as usize).div_ceil(16)
            } else {
                s.p.count as usize
            };
            let chunk = if local {
                (1048576 / s.p.width as usize).max(1)
            } else if tiled {
                4096
            } else {
                1048576
            };
            for start in (0..total).step_by(chunk) {
                ctl.check()?;
                let mut p = s.p;
                p.start = start as u32;
                let n = chunk.min(total - start);
                // Validated typed views and schedule extents bound every access;
                // run waits for completion before the exclusive lease is reused.
                let wall = Instant::now();
                let cs = unsafe {
                    x.gpu.run("transform", |e| {
                        e.set_buffer(0, a, ao);
                        e.set_buffer(1, b, bo);
                        e.set_buffer_mut(2, &mut dst, 0);
                        e.set_bytes(3, &p);
                        if local {
                            e.set_pipeline(if s.p.width <= 1024 {
                                &local_small
                            } else {
                                &local_large
                            });
                            e.dispatch_groups([n, 1, 1], (s.p.width as usize).min(256));
                        } else if tiled {
                            e.set_pipeline(&transpose);
                            e.dispatch_groups([n, 1, 1], 256);
                        } else {
                            e.dispatch_1d(&pso, n);
                        }
                        Ok(())
                    })?
                };
                max_command = max_command.max(wall.elapsed().as_secs_f64());
                device += cs.gpu_seconds;
                commands += 1;
            }
            x.stats.command_buffers += commands;
            x.stats.device_seconds += device;
            x.stats.parameter_bytes += commands as u64 * size_of::<Params>() as u64;
            if s.cache {
                // Cache storage is immutable and separately accounted. Copy on
                // device; reading an intermediate on the host is never needed.
                let mut cached = x.gpu.buffer_zeroed::<u32>(s.p.count as usize * 2)?;
                let mut p = Params::new(
                    mklight_gpu::consts::MKL_TRANSFORM_COPY,
                    s.p.count as usize,
                    1,
                );
                p.start = 0;
                let wall = Instant::now();
                let cs = unsafe {
                    x.gpu.run("cache transform", |e| {
                        e.set_buffer(0, &dst, 0);
                        e.set_buffer(1, &dst, 0);
                        e.set_buffer_mut(2, &mut cached, 0);
                        e.set_bytes(3, &p);
                        e.dispatch_1d(&pso, s.p.count as usize);
                        Ok(())
                    })?
                };
                max_command = max_command.max(wall.elapsed().as_secs_f64());
                x.stats.command_buffers += 1;
                x.stats.device_seconds += cs.gpu_seconds;
                x.stats.allocations += 1;
                x.stats.parameter_bytes += size_of::<Params>() as u64;
                let cached = Arc::new(cached);
                x.cache(keys[i].clone(), s.p.count as usize * 8, cached.clone());
                x.recycle(dst);
                slots[i + 2] = Some(Node::Cached(cached));
            } else {
                slots[i + 2] = Some(Node::Owned(dst));
            }
            for d in deps {
                uses[d] -= 1;
                if d >= 2 && uses[d] == 0 && d != recipe.output {
                    if let Some(Node::Owned(v)) = slots[d].take() {
                        x.recycle(v);
                    }
                }
            }
        }
        ctl.report((i + 1) as f64 / recipe.steps.len() as f64, "transform");
    }
    let Node::Owned(output) = slots[recipe.output].take().unwrap() else {
        unreachable!("final copy is not cached")
    };
    x.put_output(access.writes[0], output);
    for node in slots.into_iter().flatten() {
        if let Node::Owned(v) = node {
            x.recycle(v);
        }
    }
    Ok(max_command)
}
