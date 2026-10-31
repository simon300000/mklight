//! Resident geometric quantities. Optical states stay in hi+lo form until
//! final result packing. Reductions use the shared deterministic collectives.
use crate::{Df32, aiming::ResidentHits, resident::*};
use mklight_core::MkResult;
mod cpu;
pub mod spec;

#[derive(Clone, Debug)]
pub struct Coordinates {
    pub hits: ResidentHits,
    pub chiefs: ResidentHits,
    /// U32 [rays]: chief index for each hit (device indices are checked).
    pub chief_index: PlanView,
    pub defocus: Df32,
    /// DF [5,rays]: chief-relative x/y, slope differences x/y, weight.
    pub output: BufferId,
    /// U32 [rays]: 1 arrived, 0 failed. Chief failure does not discard hits;
    /// the origin is zero in that case and chief-referenced results are undefined.
    pub mask: BufferId,
}
#[derive(Clone, Debug)]
pub struct Radial {
    /// DF [5,rays], from Coordinates.
    pub points: PlanView,
    pub mask: PlanView,
    /// DF [segments,10], from Moments2.
    pub moments: PlanView,
    /// Each segment has exactly this many rays, in stable input order.
    pub segment_len: usize,
    pub centroid: bool,
    /// DF [rays] radius, square half-width, and masked weight; U32 [rays] IDs.
    pub radius: BufferId,
    pub square: BufferId,
    pub weight: BufferId,
    pub ids: BufferId,
}
#[derive(Clone, Debug)]
pub struct Characteristic {
    pub points: PlanView,
    pub mask: PlanView,
    pub segment_len: usize,
    /// DF [segments,2], frequency vector in cycles/mm.
    pub frequency: PlanView,
    /// DF [2,rays], weighted cos and sin terms, ready for SegmentReduce.
    pub output: BufferId,
}
/// Requested local sag probes. The system and surface index are validated
/// on the device so resident candidate sidecars can be consumed directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceSample {
    pub system: u32,
    pub surface: u32,
    pub reserved: [u32; 2],
    pub xy: [Df32; 2],
}
#[derive(Clone, Debug)]
pub struct SurfaceSamples {
    pub systems: SystemSet,
    /// U32 [count,8], SurfaceSample records.
    pub samples: PlanView,
    /// DF [count,3]: sag, gradient x/y; status 0 valid, 1 outside domain,
    /// 2 invalid index. No clamping of the sag domain.
    pub output: BufferId,
    pub status: BufferId,
}
/// Queries a resident sorted distance distribution and its inclusive weight scan.
#[derive(Clone, Debug)]
pub struct Distribution {
    pub keys: PlanView,
    pub cumulative: PlanView,
    pub radii: PlanView,
    /// DF [radii+4]: fractions at each radius, then 50/80/90% and total weight.
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    Coordinates(Coordinates),
    Radial(Radial),
    Characteristic(Characteristic),
    SurfaceSamples(SurfaceSamples),
    Distribution(Distribution),
}
fn input(v: &PlanView, ty: ElementType, shape: &[usize]) -> MkResult<()> {
    if v.stride != 1
        || v.desc != BufferDesc::new(ty, shape)
        || v.desc.len()? > u32::MAX as usize / ty.words()
    {
        return Err(bad("geometry input layout"));
    }
    Ok(())
}
fn output(p: &ResidentPlan, id: BufferId, ty: ElementType, shape: &[usize]) -> MkResult<()> {
    input(&p.view(id)?, ty, shape)
}
fn hit_views(h: &ResidentHits) -> [PlanView; 2] {
    [
        view(h.rays.floats, ElementType::F32, [15, h.rays.count]),
        view(h.status, ElementType::U32, [3, h.rays.count]),
    ]
}
pub fn view(id: BufferId, ty: ElementType, shape: impl Into<Vec<usize>>) -> PlanView {
    PlanView {
        buffer: id,
        offset: 0,
        stride: 1,
        desc: BufferDesc::new(ty, shape),
    }
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::{DoubleFloat as D, U32};
    match r {
        Request::Coordinates(r) => {
            for h in [&r.hits, &r.chiefs] {
                for v in hit_views(h) {
                    output(p, v.buffer, v.desc.element, &v.desc.shape)?;
                }
            }
            let n = r.hits.rays.count;
            input(&r.chief_index, U32, &[n])?;
            if !r.defocus.to_f64().is_finite() {
                return Err(bad("nonfinite defocus"));
            }
            output(p, r.output, D, &[5, n])?;
            output(p, r.mask, U32, &[n])
        }
        Request::Radial(r) => {
            let n = points(&r.points, &r.mask, r.segment_len)?;
            input(&r.moments, D, &[n / r.segment_len, 10])?;
            for id in [r.radius, r.square, r.weight] {
                output(p, id, D, &[n])?;
            }
            output(p, r.ids, U32, &[n])
        }
        Request::Characteristic(r) => {
            let n = points(&r.points, &r.mask, r.segment_len)?;
            input(&r.frequency, D, &[n / r.segment_len, 2])?;
            output(p, r.output, D, &[2, n])
        }
        Request::Distribution(r) => {
            let n = r.keys.desc.len()?;
            let k = r.radii.desc.len()?;
            input(&r.keys, D, &[n])?;
            input(&r.cumulative, D, &[n])?;
            input(&r.radii, D, &[k])?;
            output(p, r.output, D, &[k + 4])
        }
        Request::SurfaceSamples(r) => {
            let n = r.samples.desc.shape.first().copied().unwrap_or(0);
            input(&r.samples, U32, &[n, 8])?;
            if let SystemSet::Resident {
                headers,
                surfaces,
                coefficients,
                count,
                max_surfaces,
                ..
            } = &r.systems
            {
                output(
                    p,
                    *headers,
                    U32,
                    &[count * size_of::<crate::layout::PackedHeader>() / 4],
                )?;
                output(
                    p,
                    *surfaces,
                    U32,
                    &[count * max_surfaces * size_of::<crate::layout::PackedSurface>() / 4],
                )?;
                let d = p.desc(*coefficients)?;
                if d.element != D {
                    return Err(bad("geometry coefficient layout"));
                }
            }
            output(p, r.output, D, &[n, 3])?;
            output(p, r.status, U32, &[n])
        }
    }
}
fn points(v: &PlanView, mask: &PlanView, segment: usize) -> MkResult<usize> {
    let n = v.desc.shape.get(1).copied().unwrap_or(0);
    input(v, ElementType::DoubleFloat, &[5, n])?;
    input(mask, ElementType::U32, &[n])?;
    if segment == 0 || n % segment != 0 {
        return Err(bad("geometry segment length"));
    }
    Ok(n)
}
pub fn accesses(r: &Request) -> Access {
    let mut a = Access::default();
    match r {
        Request::Coordinates(r) => {
            a.reads.extend(hit_views(&r.hits));
            a.reads.extend(hit_views(&r.chiefs));
            a.reads.push(r.chief_index.clone());
            a.writes = vec![r.output, r.mask];
        }
        Request::Radial(r) => {
            a.reads = vec![r.points.clone(), r.mask.clone(), r.moments.clone()];
            a.writes = vec![r.radius, r.square, r.weight, r.ids];
        }
        Request::Characteristic(r) => {
            a.reads = vec![r.points.clone(), r.mask.clone(), r.frequency.clone()];
            a.writes = vec![r.output];
        }
        Request::Distribution(r) => {
            a.reads = vec![r.keys.clone(), r.cumulative.clone(), r.radii.clone()];
            a.writes = vec![r.output];
        }
        Request::SurfaceSamples(r) => {
            a.reads.push(r.samples.clone());
            a.writes = vec![r.output, r.status];
            match &r.systems {
                SystemSet::Uploaded(h) => a.systems.push(*h),
                SystemSet::Resident {
                    headers,
                    surfaces,
                    coefficients,
                    ..
                } => a.reads.extend([
                    view(*headers, ElementType::U32, [0]),
                    view(*surfaces, ElementType::U32, [0]),
                    view(*coefficients, ElementType::DoubleFloat, [0]),
                ]),
            }
        }
    }
    a
}
pub fn execute(
    r: &Request,
    x: &mut cpu_execution::Execution,
    ctl: &crate::pending::JobControl,
) -> MkResult<()> {
    cpu::execute(r, x, ctl)
}
use crate::resident::cpu as cpu_execution;
