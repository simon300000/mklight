//! Resident coating and polarisation service. Direct angle evaluation avoids
//! interpolation at critical angles; immutable material records can be reused
//! across plans. Engine routing and optical qualification remain with P11.
mod jobs;
mod plan;
mod records;
pub use jobs::*;
pub use plan::*;
mod coating;
mod math;
mod transport;
use crate::{pending::JobControl, resident::*};
use mklight_core::MkResult;
pub use records::*;

#[derive(Clone, Debug)]
pub struct CoatingRequest {
    pub specs: PlanView,
    pub layers: PlanView,
    pub queries: PlanView,
    pub output: BufferId,
    /// Bounded dispatch size, independent of numerical operation order.
    pub tile: usize,
}
#[derive(Clone, Debug)]
pub enum Request {
    PupilChannels {
        values: PlanView,
        vector: bool,
        output: BufferId,
        status: BufferId,
    },
    Coating(CoatingRequest),
    Polarization(Box<PolarizationRequest>),
    Channels {
        values: PlanView,
        trace_status: PlanView,
        output: BufferId,
    },
}
/// Consume P1 all-surface hi/lo records without a host readback. One system
/// per request; wavelength IDs are the P1 [2,N] ray-index plane.
#[derive(Clone, Debug)]
pub struct PolarizationRequest {
    /// Scalar transmission skips PRT/properties; only power[0..2] are populated.
    pub scalar: bool,
    pub surfaces: PlanView,
    pub layers: PlanView,
    pub records: PlanView,
    pub record_low: PlanView,
    pub indices: PlanView,
    pub trace_status: PlanView,
    pub wavelengths: usize,
    pub stokes: [crate::Df32; 4],
    pub output: BufferId,
    pub tile: usize,
}

pub fn record_desc<T>(count: usize) -> BufferDesc {
    BufferDesc::new(ElementType::U32, [count, std::mem::size_of::<T>() / 4])
}
fn check<T>(v: &PlanView, p: &ResidentPlan) -> MkResult<usize> {
    let base = match p.buffers.get(v.buffer.0) {
        Some(PlanBuffer::Import(input)) => input.offset,
        _ => 0,
    };
    let n = v.desc.shape.first().copied().unwrap_or(0);
    if v.desc != record_desc::<T>(n)
        || v.stride != 1
        || !(base + v.offset).is_multiple_of(2)
        || v.desc.len()? > u32::MAX as usize
    {
        return Err(bad("radiometry record layout"));
    }
    Ok(n)
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    match r {
        Request::PupilChannels {
            values,
            vector,
            output,
            status,
        } => {
            let n = check::<PolarizationValue>(values, p)?;
            if p.desc(*output)?
                != &BufferDesc::new(ElementType::Complex32, [if *vector { 6 } else { 4 }, n])
                || p.desc(*status)? != &BufferDesc::new(ElementType::U32, [n])
            {
                return Err(bad("complex pupil channels"));
            }
        }
        Request::Coating(r) => {
            check::<CoatingSpec>(&r.specs, p)?;
            check::<CoatingLayer>(&r.layers, p)?;
            let n = check::<CoatingQuery>(&r.queries, p)?;
            if p.desc(r.output)? != &record_desc::<CoatingValue>(n) || r.tile == 0 || r.tile > 65536
            {
                return Err(bad("coating output or tile"));
            }
        }
        Request::Polarization(r) => {
            let ns = check::<SurfaceSpec>(&r.surfaces, p)?;
            check::<CoatingLayer>(&r.layers, p)?;
            let shape = &r.records.desc.shape;
            if r.wavelengths == 0
                || ns % r.wavelengths != 0
                || shape.len() != 3
                || shape[0] != ns / r.wavelengths
                || shape[0] < 2
                || shape[2] != 13
                || r.records.desc.element != ElementType::F32
                || r.record_low.desc != r.records.desc
                || r.records.stride != 1
                || r.record_low.stride != 1
            {
                return Err(bad("polarization P1 records"));
            }
            let n = shape[1];
            if r.indices.desc != BufferDesc::new(ElementType::U32, [2, n])
                || r.trace_status.desc != BufferDesc::new(ElementType::U32, [3, n])
                || r.indices.stride != 1
                || r.trace_status.stride != 1
                || p.desc(r.output)? != &record_desc::<PolarizationValue>(n)
                || r.tile == 0
                || r.tile > 65536
            {
                return Err(bad("polarization output, indices or tile"));
            }
            let st = r.stokes.map(|v| v.to_f64());
            mklight_trace::PolarizationState::Stokes {
                s: mklight_trace::Stokes {
                    s0: st[0],
                    s1: st[1],
                    s2: st[2],
                    s3: st[3],
                },
            }
            .validate()?;
        }
        Request::Channels {
            values,
            trace_status,
            output,
        } => {
            let n = check::<PolarizationValue>(values, p)?;
            if trace_status.desc != BufferDesc::new(ElementType::U32, [3, n])
                || trace_status.stride != 1
                || p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [10, n])
            {
                return Err(bad("radiometry summary channels"));
            }
        }
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    match r {
        Request::PupilChannels {
            values,
            output,
            status,
            ..
        } => Access {
            reads: vec![values.clone()],
            writes: vec![*output, *status],
            systems: vec![],
        },
        Request::Coating(r) => Access {
            reads: vec![r.specs.clone(), r.layers.clone(), r.queries.clone()],
            writes: vec![r.output],
            systems: vec![],
        },
        Request::Polarization(r) => Access {
            reads: vec![
                r.surfaces.clone(),
                r.layers.clone(),
                r.records.clone(),
                r.record_low.clone(),
                r.indices.clone(),
                r.trace_status.clone(),
            ],
            writes: vec![r.output],
            systems: vec![],
        },
        Request::Channels {
            values,
            trace_status,
            output,
        } => Access {
            reads: vec![values.clone(), trace_status.clone()],
            writes: vec![*output],
            systems: vec![],
        },
    }
}
fn rows<'a, T: bytemuck::Pod>(x: &'a cpu::Execution, v: &PlanView) -> &'a [T] {
    bytemuck::cast_slice(&x.words(v.buffer)[v.offset..v.offset + v.desc.len().unwrap()])
}
pub fn execute(r: &Request, x: &mut cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    match r {
        Request::PupilChannels {
            values,
            vector,
            output,
            status,
        } => {
            let n = values.desc.shape[0];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let v = rows::<PolarizationValue>(x, values)[i];
                let channels = if *vector { &v.fields[..] } else { &v.jones[..] };
                let dest: &mut [[f32; 2]] = bytemuck::cast_slice_mut(x.words_mut(*output));
                for (c, z) in channels.as_chunks::<2>().0.iter().enumerate() {
                    dest[c * n + i] = if v.status == 0 {
                        [z[0].hi + z[0].lo, z[1].hi + z[1].lo]
                    } else {
                        [0.; 2]
                    };
                }
                x.words_mut(*status)[i] = v.status;
            }
        }
        Request::Coating(r) => {
            let n = r.queries.desc.shape[0];
            for i in 0..n {
                if i % r.tile == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n as f64, "coating");
                }
                let q = rows::<CoatingQuery>(x, &r.queries)[i];
                let specs = rows::<CoatingSpec>(x, &r.specs);
                let layers = rows::<CoatingLayer>(x, &r.layers);
                let out = if let Some(s) = specs.get(q.coating as usize) {
                    if s.count <= crate::consts::MKL_RAD_MAX_LAYERS
                        && (s.offset as usize + s.count as usize) <= layers.len()
                    {
                        coating::evaluate(
                            *s,
                            &layers[s.offset as usize..][..s.count as usize],
                            q.cosine,
                        )
                    } else {
                        CoatingValue {
                            status: crate::consts::MKL_RAD_REFERENCE,
                            ..bytemuck::Zeroable::zeroed()
                        }
                    }
                } else {
                    CoatingValue {
                        status: crate::consts::MKL_RAD_REFERENCE,
                        ..bytemuck::Zeroable::zeroed()
                    }
                };
                x.stats.fallback_count += (out.status == crate::consts::MKL_RAD_REFERENCE) as u64;
                let output: &mut [CoatingValue] = bytemuck::cast_slice_mut(x.words_mut(r.output));
                output[i] = out;
            }
        }
        Request::Polarization(r) => {
            let n = r.records.desc.shape[1];
            for i in 0..n {
                if i % r.tile == 0 {
                    ctl.check()?;
                    ctl.report(i as f64 / n as f64, "polarization");
                }
                let status = x.words(r.trace_status.buffer)[r.trace_status.offset + i];
                let wl = x.words(r.indices.buffer)[r.indices.offset + n + i] as usize;
                let out = if status != 0 {
                    PolarizationValue {
                        status: crate::consts::MKL_RAD_BLOCKED,
                        ..bytemuck::Zeroable::zeroed()
                    }
                } else if wl >= r.wavelengths {
                    PolarizationValue {
                        status: crate::consts::MKL_RAD_REFERENCE,
                        ..bytemuck::Zeroable::zeroed()
                    }
                } else {
                    transport::evaluate(
                        r.scalar,
                        rows(x, &r.surfaces),
                        rows(x, &r.layers),
                        &bytemuck::cast_slice::<u32, f32>(x.words(r.records.buffer))
                            [r.records.offset..],
                        &bytemuck::cast_slice::<u32, f32>(x.words(r.record_low.buffer))
                            [r.record_low.offset..],
                        n,
                        i,
                        wl,
                        r.wavelengths,
                        r.stokes,
                    )
                };
                x.stats.fallback_count += (out.status == crate::consts::MKL_RAD_REFERENCE) as u64;
                let output: &mut [PolarizationValue] =
                    bytemuck::cast_slice_mut(x.words_mut(r.output));
                output[i] = out;
            }
        }
        Request::Channels {
            values,
            trace_status,
            output,
        } => {
            let n = values.desc.shape[0];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let v = rows::<PolarizationValue>(x, values)[i];
                let status = x.words(trace_status.buffer)[trace_status.offset + i];
                let channels = summary_channels(v, status);
                let dest: &mut [crate::Df32] = bytemuck::cast_slice_mut(x.words_mut(*output));
                for k in 0..10 {
                    dest[k * n + i] = channels[k];
                }
            }
        }
    }
    Ok(())
}
fn summary_channels(v: PolarizationValue, trace_status: u32) -> [crate::Df32; 10] {
    use crate::{Df32 as D, consts::*};
    let failed = trace_status & MKL_RAY_CODE_MASK >= MKL_RAY_NONFINITE;
    let fallback = v.status == MKL_RAD_REFERENCE;
    let ok = v.status == MKL_RAD_OK && !failed;
    let defined = ok && v.defined & 1 != 0;
    let flag = |b: bool| D::from_f32(b as u32 as f32);
    [
        if ok { v.power[0] } else { D::ZERO },
        if ok { v.power[2] } else { D::ZERO },
        if ok { v.power[3] } else { D::ZERO },
        if ok { v.properties[0] } else { D::ZERO },
        if defined {
            v.properties[1] * v.properties[1]
        } else {
            D::ZERO
        },
        flag(!failed && !fallback),
        flag(ok),
        flag(defined),
        flag(fallback),
        flag(failed),
    ]
}
