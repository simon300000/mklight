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
    origin: [[f32; 2]; 3],
}
const _: () = assert!(size_of::<Params>() == 64, "MklGeometryParams");
fn bind(e: &crate::context::Encoder<'_>, i: usize, x: &Execution<'_>, v: &PlanView) {
    let (b, o) = x.buffer(v);
    e.set_buffer(i, b, o);
}
/// Packed headers, surfaces and coefficients at slots 0-2.
fn bind_systems(
    e: &crate::context::Encoder<'_>,
    x: &Execution<'_>,
    uploaded: Option<&crate::backend::Uploaded>,
    systems: &SystemSet,
) {
    if let Some(u) = uploaded {
        e.set_buffer(0, &u.headers, 0);
        e.set_buffer(1, &u.surfaces, 0);
        e.set_buffer(2, &u.coefficients, 0);
    } else if let SystemSet::Resident {
        headers,
        surfaces,
        coefficients,
        ..
    } = *systems
    {
        bind(e, 0, x, &view(headers, ElementType::U32, [0]));
        bind(e, 1, x, &view(surfaces, ElementType::U32, [0]));
        bind(e, 2, x, &view(coefficients, ElementType::DoubleFloat, [0]));
    }
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.last().map_or(0, |s| s.operation + 1),
        name: String::new(),
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
            r.mask.desc.len()? * r.frequency.desc.shape[0],
            Params {
                segment: r.segment_len as u32,
                chiefs: r.frequency.desc.shape[0] as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::FocusTerms(r) => (
            "mkl_geometry_focus_terms",
            r.mask.desc.len()?,
            Params {
                segment: r.segment_len as u32,
                centroid: u32::from(r.centroid),
                ..Zeroable::zeroed()
            },
        ),
        Request::Opd(r) => (
            "mkl_geometry_opd",
            r.hits.rays.count,
            Params {
                centroid: u32::from(r.axial),
                chiefs: r.chiefs.rays.count as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::Illumination(r) => (
            "mkl_geometry_illumination",
            r.nodes.desc.shape[0],
            Params {
                chiefs: r.chiefs.rays.count as u32,
                segment: r.slots as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::CloseFocus(r) => (
            "mkl_geometry_close",
            r.hits.rays.count / 5,
            Params {
                chiefs: r.hits.rays.count as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::Coddington(r) => (
            "mkl_geometry_coddington",
            r.hits.rays.count,
            Params {
                segment: r.slots as u32,
                ..Zeroable::zeroed()
            },
        ),
        Request::Records(r) => (
            "mkl_geometry_records",
            r.slots * r.hits.rays.count,
            Params {
                centroid: u32::from(r.terminal),
                chiefs: r.hits.rays.count as u32,
                origin: r.origin.map(|v| [v.hi, v.lo]),
                ..Zeroable::zeroed()
            },
        ),
        Request::FieldData(r) => (
            "mkl_geometry_fields",
            r.hits.rays.count / if r.kind == 1 { 5 } else { 1 },
            Params {
                chiefs: r.hits.rays.count as u32,
                centroid: r.kind,
                defocus: [r.pupil_step.hi, r.pupil_step.lo],
                segment: if r.kind != 1 {
                    (r.record_slots - 1) as u32
                } else {
                    0
                },
                ..Zeroable::zeroed()
            },
        ),
        Request::Distribution(r) => (
            "mkl_geometry_distribution",
            r.radii.desc.len()? + 4,
            Params {
                segment: r.keys.desc.len()? as u32,
                defocus: [r.uncertainty.hi, r.uncertainty.lo],
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
    stage.name = name.into();
    p.stride = n as u32;
    let uploaded = match r.systems() {
        Some(SystemSet::Uploaded(h)) => Some(x.system(*h)?),
        _ => None,
    };
    p.systems = match (&uploaded, r.systems()) {
        (Some(u), _) => u.set.headers.len() as u32,
        (None, Some(SystemSet::Resident { count, .. })) => *count as u32,
        _ => 0,
    };
    if let Request::SurfaceSamples(SurfaceSamples {
        projection: Some(v),
        ..
    }) = r
    {
        p.centroid = 1;
        p.segment = v.frames.desc.shape[0] as u32;
        p.origin = v.origin.map(|v| [v.hi, v.lo]);
    }
    if let Request::Characteristic(r) = r {
        p.stride = r.mask.desc.len()? as u32;
        p.systems = n as u32;
    }
    let a = mklight_gpu::geometry::accesses(r);
    let mut out: Vec<_> = a.writes.iter().map(|&id| x.take_output(id)).collect();
    let pso = x.gpu.pipeline(&PipelineKey::plain(name))?;
    // RI nodes run nested Newton solves: size those commands adaptively from
    // measured throughput, starting large enough to occupy the device. The
    // other kernels are closed-form per thread; fixed chunks only bound the
    // command time (and so the cancellation latency).
    let adaptive = matches!(r, Request::Illumination(_));
    let (minimum, maximum) = (4096, 1 << 18);
    let mut next = match r {
        Request::Illumination(_) => minimum,
        Request::Coddington(_) => 1 << 16,
        _ => 1 << 22,
    };
    let mut offset = 0;
    while offset < n {
        ctl.check()?;
        p.offset = offset as u32;
        p.count = (n - offset).min(next) as u32;
        let t = Instant::now();
        // SAFETY: submission validates every view/output extent. The kernels
        // bound device-provided chief and system/surface indices before use.
        unsafe {
            x.run(name, |e| {
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
                    Request::FocusTerms(r) => {
                        for (i, v) in [&r.points, &r.mask, &r.position_moments, &r.slope_moments]
                            .iter()
                            .enumerate()
                        {
                            bind(e, i, x, v);
                        }
                        e.set_buffer_mut(4, &mut out[0], 0);
                    }
                    Request::Opd(r) => {
                        for (i, v) in [
                            view(r.hits.rays.floats, ElementType::F32, [0]),
                            view(r.hits.status, ElementType::U32, [0]),
                            view(r.chiefs.rays.floats, ElementType::F32, [0]),
                            view(r.chiefs.status, ElementType::U32, [0]),
                            r.chief_index.clone(),
                            r.reference.clone(),
                        ]
                        .iter()
                        .enumerate()
                        {
                            bind(e, i, x, v);
                        }
                        e.set_buffer_mut(6, &mut out[0], 0);
                        e.set_buffer_mut(8, &mut out[1], 0);
                    }
                    Request::Illumination(r) => {
                        bind_systems(e, x, uploaded.as_deref(), &r.systems);
                        bind(
                            e,
                            3,
                            x,
                            &view(r.chiefs.records.unwrap(), ElementType::F32, [0]),
                        );
                        bind(
                            e,
                            4,
                            x,
                            &view(r.chiefs.record_low.unwrap(), ElementType::F32, [0]),
                        );
                        bind(e, 5, x, &r.fields);
                        bind(e, 6, x, &r.nodes);
                        e.set_buffer_mut(8, &mut out[0], 0);
                        e.set_buffer_mut(9, &mut out[1], 0);
                        e.set_buffer_mut(12, &mut out[2], 0);
                        let un = mklight_gpu::aiming::kernel::newton_params();
                        let clipped = mklight_gpu::layout::TraceParams {
                            flags: mklight_gpu::consts::MKL_TRACE_CLIP,
                            ..un
                        };
                        e.set_bytes(10, &[un, clipped]);
                        bind(e, 11, x, &view(r.chiefs.status, ElementType::U32, [0]));
                    }
                    Request::CloseFocus(r) => {
                        bind(e, 0, x, &view(r.hits.rays.floats, ElementType::F32, [0]));
                        bind(e, 1, x, &view(r.hits.status, ElementType::U32, [0]));
                        e.set_buffer_mut(2, &mut out[0], 0);
                        e.set_buffer_mut(3, &mut out[1], 0);
                    }
                    Request::Coddington(r) => {
                        bind_systems(e, x, uploaded.as_deref(), &r.systems);
                        for (i, (id, ty)) in [
                            (r.hits.records.unwrap(), ElementType::F32),
                            (r.hits.record_low.unwrap(), ElementType::F32),
                            (r.hits.rays.indices, ElementType::U32),
                            (r.hits.status, ElementType::U32),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            bind(e, i + 3, x, &view(id, ty, [0]));
                        }
                        e.set_buffer_mut(8, &mut out[0], 0);
                        e.set_buffer_mut(9, &mut out[1], 0);
                    }
                    Request::Records(r) => {
                        bind(
                            e,
                            0,
                            x,
                            &view(r.hits.records.unwrap(), ElementType::F32, [0]),
                        );
                        bind(
                            e,
                            1,
                            x,
                            &view(r.hits.record_low.unwrap(), ElementType::F32, [0]),
                        );
                        bind(e, 2, x, &r.frames);
                        bind(e, 8, x, &view(r.hits.status, ElementType::U32, [0]));
                        for (i, b) in out[..4].iter_mut().enumerate() {
                            e.set_buffer_mut(i + 3, b, 0);
                        }
                        e.set_buffer_mut(9, &mut out[4], 0);
                        bind(e, 10, x, &view(r.hits.rays.floats, ElementType::F32, [0]));
                    }
                    Request::FieldData(r) => {
                        bind(e, 0, x, &view(r.hits.rays.floats, ElementType::F32, [0]));
                        bind(e, 1, x, &view(r.hits.status, ElementType::U32, [0]));
                        if r.kind != 1 {
                            bind(
                                e,
                                2,
                                x,
                                &view(r.hits.records.unwrap(), ElementType::F32, [0]),
                            );
                            bind(
                                e,
                                3,
                                x,
                                &view(r.hits.record_low.unwrap(), ElementType::F32, [0]),
                            );
                        } else {
                            bind(e, 2, x, &view(r.hits.rays.floats, ElementType::F32, [0]));
                            bind(e, 3, x, &view(r.hits.rays.floats, ElementType::F32, [0]));
                        }
                        e.set_buffer_mut(4, &mut out[0], 0);
                    }
                    Request::Distribution(r) => {
                        for (i, v) in [&r.keys, &r.cumulative, &r.radii].iter().enumerate() {
                            bind(e, i, x, v);
                        }
                        e.set_buffer_mut(3, &mut out[0], 0);
                        e.set_buffer_mut(4, &mut out[1], 0);
                    }
                    Request::SurfaceSamples(r) => {
                        bind_systems(e, x, uploaded.as_deref(), &r.systems);
                        bind(e, 3, x, &r.samples);
                        if let Some(v) = &r.projection {
                            bind(e, 6, x, &v.frames);
                            bind(e, 8, x, &v.xy);
                        } else {
                            bind(e, 6, x, &r.samples);
                            bind(e, 8, x, &r.samples);
                        }

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
        let more = offset + (p.count as usize) < n;
        let adaptive_seconds = if adaptive && more { x.measure()? } else { 0. };
        stage.max_command_seconds = stage.max_command_seconds.max(t.elapsed().as_secs_f64());
        x.stats.parameter_bytes += size_of::<Params>() as u64;
        ctl.check()?;
        ctl.report((offset + p.count as usize) as f64 / n as f64, "geometry");
        offset += p.count as usize;
        if adaptive {
            let fit = (p.count as f64 * 0.010 / adaptive_seconds.max(1e-6)) as usize;
            next = fit.clamp(minimum, (p.count as usize * 8).clamp(minimum, maximum));
            x.stats.parameter_bytes += 2 * size_of::<mklight_gpu::layout::TraceParams>() as u64;
        }
    }
    for (id, b) in a.writes.into_iter().zip(out) {
        x.put_output(id, b);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.stages.push(stage);
    ctl.check()
}
