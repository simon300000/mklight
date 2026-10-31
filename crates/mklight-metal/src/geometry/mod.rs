//! Metal encoders for resident geometric quantities. Inputs are bound in
//! place; only plan-owned outputs are writable.
use crate::{context::PipelineKey, resident::Execution};
use bytemuck::{Pod, Zeroable};
use mklight_core::MkResult;
use mklight_gpu::{geometry::*, pending::JobControl, resident::*};
use std::time::Instant;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    offset: u32,
    stride: u32,
    chiefs: u32,
    segment: u32,
    centroid: u32,
    systems: u32,
    reserved: u32,
    defocus: [f32; 2],
}
fn bind(e: &crate::context::Encoder<'_>, i: usize, x: &Execution<'_>, v: &PlanView) {
    let (b, o) = x.buffer(v);
    e.set_buffer(i, b, o);
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stage = StageStats {
        name: "geometry".into(),
        ..Default::default()
    };
    let (name, n, mut p) = match r {
        Request::Coordinates(r) => (
            "mkl_geometry_coordinates",
            r.hits.rays.count,
            Params {
                chiefs: r.chiefs.rays.count as u32,
                defocus: [r.defocus.hi, r.defocus.lo],
                ..Zeroable::zeroed()
            },
        ),
        Request::Radial(r) => (
            "mkl_geometry_radial",
            r.mask.desc.len()?,
            Params {
                segment: r.segment_len as u32,
                centroid: u32::from(r.centroid),
                ..Zeroable::zeroed()
            },
        ),
        Request::Characteristic(r) => (
            "mkl_geometry_characteristic",
            r.mask.desc.len()?,
            Params {
                segment: r.segment_len as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::Distribution(r) => (
            "mkl_geometry_distribution",
            r.radii.desc.len()? + 4,
            Params {
                segment: r.keys.desc.len()? as u32,
                chiefs: r.radii.desc.len()? as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::SurfaceSamples(r) => (
            "mkl_geometry_surfaces",
            r.samples.desc.shape[0],
            Zeroable::zeroed(),
        ),
    };
    p.stride = n as u32;
    let uploaded = match r {
        Request::SurfaceSamples(r) => match r.systems {
            SystemSet::Uploaded(h) => Some(x.system(h)?),
            _ => None,
        },
        _ => None,
    };
    if let Request::SurfaceSamples(r) = r {
        p.systems = uploaded.as_ref().map_or_else(
            || match r.systems {
                SystemSet::Resident { count, .. } => count as u32,
                _ => 0,
            },
            |u| u.set.headers.len() as u32,
        );
    }
    let a = mklight_gpu::geometry::accesses(r);
    let mut out: Vec<_> = a.writes.iter().map(|&id| x.take_output(id)).collect();
    let pso = x.gpu.pipeline(&PipelineKey::plain(name))?;
    for offset in (0..n).step_by(1 << 18) {
        ctl.check()?;
        p.offset = offset as u32;
        p.count = (n - offset).min(1 << 18) as u32;
        let t = Instant::now();
        // SAFETY: submission validates every view/output extent. The kernels
        // bound device-provided chief and system/surface indices before use.
        let cs = unsafe {
            x.gpu.run(name, |e| {
                match r {
                    Request::Coordinates(r) => {
                        for (i, v) in [
                            view(r.hits.rays.floats, ElementType::F32, [0]),
                            view(r.hits.status, ElementType::U32, [0]),
                            view(r.chiefs.rays.floats, ElementType::F32, [0]),
                            view(r.chiefs.status, ElementType::U32, [0]),
                            r.chief_index.clone(),
                        ]
                        .iter()
                        .enumerate()
                        {
                            bind(e, i, x, v);
                        }
                        let (a, b) = out.split_at_mut(1);
                        e.set_buffer_mut(5, &mut a[0], 0);
                        e.set_buffer_mut(6, &mut b[0], 0);
                    }
                    Request::Radial(r) => {
                        for (i, v) in [&r.points, &r.mask, &r.moments].iter().enumerate() {
                            bind(e, i, x, v);
                        }
                        for (i, b) in out.iter_mut().enumerate() {
                            e.set_buffer_mut(i + 3, b, 0);
                        }
                    }
                    Request::Characteristic(r) => {
                        for (i, v) in [&r.points, &r.mask, &r.frequency].iter().enumerate() {
                            bind(e, i, x, v);
                        }
                        e.set_buffer_mut(3, &mut out[0], 0);
                    }
                    Request::Distribution(r) => {
                        for (i, v) in [&r.keys, &r.cumulative, &r.radii].iter().enumerate() {
                            bind(e, i, x, v);
                        }
                        e.set_buffer_mut(3, &mut out[0], 0);
                    }
                    Request::SurfaceSamples(r) => {
                        if let Some(u) = &uploaded {
                            e.set_buffer(0, &u.headers, 0);
                            e.set_buffer(1, &u.surfaces, 0);
                            e.set_buffer(2, &u.coefficients, 0);
                        } else if let SystemSet::Resident {
                            headers,
                            surfaces,
                            coefficients,
                            ..
                        } = r.systems
                        {
                            for (i, (id, ty)) in [
                                (headers, ElementType::U32),
                                (surfaces, ElementType::U32),
                                (coefficients, ElementType::DoubleFloat),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                bind(e, i, x, &view(id, ty, [0]));
                            }
                        }
                        bind(e, 3, x, &r.samples);
                        for (i, b) in out.iter_mut().enumerate() {
                            e.set_buffer_mut(i + 4, b, 0);
                        }
                    }
                }
                e.set_bytes(7, &p);
                e.dispatch_1d(&pso, p.count as usize);
                Ok(())
            })?
        };
        stage.device_seconds += cs.gpu_seconds;
        stage.command_buffers += 1;
        stage.max_command_seconds = stage.max_command_seconds.max(t.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.check()?;
        ctl.report((offset + p.count as usize) as f64 / n as f64, "geometry");
    }
    for (id, b) in a.writes.into_iter().zip(out) {
        x.put_output(id, b);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stage.device_seconds;
    x.stats.command_buffers += stage.command_buffers;
    x.stats.stages.push(stage);
    ctl.check()
}
