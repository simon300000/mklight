//! Resident geometric quantities. Optical states stay in hi+lo form until
//! final result packing. Reductions use the shared deterministic collectives.
use crate::{Df32, aiming::ResidentHits, resident::*};
use mklight_core::MkResult;
pub const VERSION: u32 = 1;
pub(crate) const RECORD_FLOATS: usize = crate::consts::MKL_AIM_RECORD_FLOATS as usize;
mod cpu;
mod curvature;
mod illumination;
mod sag;
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
    /// DF [frequencies,segments,2], frequency vectors in cycles/mm.
    pub frequency: PlanView,
    /// DF [2,frequencies*rays], weighted cos/sin terms for SegmentReduce.
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
pub struct VertexProjection {
    pub xy: PlanView,
    pub frames: PlanView,
    pub origin: [Df32; 3],
}
#[derive(Clone, Debug)]
pub struct SurfaceSamples {
    pub projection: Option<VertexProjection>,
    pub systems: SystemSet,
    /// U32 [count,8], SurfaceSample records.
    pub samples: PlanView,
    /// DF [count,3]: sag, gradient x/y; status 0 valid, 1 outside domain,
    /// 2 invalid index. No clamping of the sag domain.
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct FieldData {
    pub record_slots: usize,
    pub hits: ResidentHits,
    /// 0: chief incidence and signed x/y angles (one ray per field, records
    /// required); 1: pupil error/magnification (five rays per field: 0,+x,-x,+y,-y);
    /// 2: image x/y and launch source (one ray per field, records required).
    pub kind: u32,
    /// Kind 1: twice the physical entrance pupil difference step (mm). Kind 2:
    /// zero for a direction source (launch tangents), nonzero for a point
    /// source (launch x/y).
    pub pupil_step: Df32,
    /// DF [fields,5]: image x/y and three angles; stop x/y, two
    /// magnifications and zero; or image x/y, source x/y and zero. Invalid
    /// probes publish NaN.
    pub output: BufferId,
}
/// All-surface record projection, including hits whose rays later fail.
#[derive(Clone, Debug)]
pub struct Records {
    pub terminal: bool,
    pub launch: BufferId,
    pub hits: ResidentHits,
    pub slots: usize,
    /// DF [slots,12]: row-major global rotation followed by origin.
    pub frames: PlanView,
    pub origin: [Df32; 3],
    /// DF [slots,rays,3], global points relative to origin; NaN for lost records.
    pub output: BufferId,
    /// DF [1,slots*rays], local radii; U32 [slots*rays] validity.
    pub radii: BufferId,
    pub mask: BufferId,
    /// DF [rays], one for a failed ray, zero for an arrived ray.
    pub failures: BufferId,
}
#[derive(Clone, Debug)]
pub struct CloseFocus {
    pub hits: ResidentHits,
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct Coddington {
    pub systems: SystemSet,
    pub hits: ResidentHits,
    pub slots: usize,
    /// DF [rays,3]: signed tangential/sagittal focal distances and chief dz.
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct Illumination {
    /// Stable U32 [nodes] logical indices for sparse unresolved compaction.
    pub ids: BufferId,
    pub systems: SystemSet,
    /// FieldSpec rows aligned with chief hits, last row is the q-step scale field.
    pub fields: PlanView,
    pub chiefs: ResidentHits,
    pub slots: usize,
    /// Sample rows, field selects a chief, weight is the polar quadrature weight.
    pub nodes: PlanView,
    /// DF [4,nodes]: weighted solid-angle/area, failed count, weighted cos4.
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct Opd {
    pub axial: bool,
    pub hits: ResidentHits,
    pub chiefs: ResidentHits,
    pub chief_index: PlanView,
    /// DF [chiefs,3]: exit-pupil z, image index, wavelength um.
    pub reference: PlanView,
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct FocusTerms {
    pub points: PlanView,
    pub mask: PlanView,
    pub position_moments: PlanView,
    pub slope_moments: PlanView,
    pub segment_len: usize,
    pub centroid: bool,
    /// DF [3,rays]: weighted |q|^2, q.dot(dt), |dt|^2 after centring.
    pub output: BufferId,
}
/// Queries a resident sorted distance distribution and its inclusive weight scan.
#[derive(Clone, Debug)]
pub struct Distribution {
    /// Absolute distance uncertainty used to bound CDF classification error.
    pub uncertainty: Df32,
    pub boundary: BufferId,
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
    FieldData(FieldData),
    Records(Records),
    Coddington(Coddington),
    CloseFocus(CloseFocus),
    Illumination(Illumination),
    Opd(Opd),
    FocusTerms(FocusTerms),
}
impl Request {
    /// The packed systems a request evaluates surfaces of, if any.
    pub fn systems(&self) -> Option<&SystemSet> {
        match self {
            Request::SurfaceSamples(r) => Some(&r.systems),
            Request::Coddington(r) => Some(&r.systems),
            Request::Illumination(r) => Some(&r.systems),
            _ => None,
        }
    }
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
        view(
            h.rays.floats,
            ElementType::F32,
            [crate::consts::MKL_TRACE_IN_FLOATS as usize, h.rays.count],
        ),
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
    if let Some(SystemSet::Resident {
        headers,
        surfaces,
        coefficients,
        count,
        max_surfaces,
        max_wavelengths,
    }) = r.systems()
    {
        if *count == 0
            || *count > u32::MAX as usize
            || *max_surfaces < 2
            || *max_surfaces > u32::MAX as usize
            || *max_wavelengths > crate::layout::MAX_WAVELENGTHS
        {
            return Err(bad("geometry resident system dimensions"));
        }
        let hw = count
            .checked_mul(size_of::<crate::layout::PackedHeader>() / 4)
            .ok_or_else(|| bad("geometry header extent"))?;
        let sw = count
            .checked_mul(*max_surfaces)
            .and_then(|n| n.checked_mul(size_of::<crate::layout::PackedSurface>() / 4))
            .ok_or_else(|| bad("geometry surface extent"))?;
        for (id, len) in [(*headers, hw), (*surfaces, sw)] {
            let d = p.desc(id)?;
            if d.element != U32 || d.len()? != len {
                return Err(bad("geometry resident system sidecar layout"));
            }
        }
        if p.desc(*coefficients)?.element != D {
            return Err(bad("geometry coefficient layout"));
        }
    }
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
            let k = r.frequency.desc.shape.first().copied().unwrap_or(0);
            input(&r.frequency, D, &[k, n / r.segment_len, 2])?;
            output(p, r.output, D, &[2, k * n])
        }
        Request::FocusTerms(r) => {
            let n = points(&r.points, &r.mask, r.segment_len)?;
            let s = n / r.segment_len;
            input(&r.position_moments, D, &[s, 10])?;
            input(&r.slope_moments, D, &[s, 10])?;
            output(p, r.output, D, &[3, n])
        }
        Request::Opd(r) => {
            for h in [&r.hits, &r.chiefs] {
                for v in hit_views(h) {
                    output(p, v.buffer, v.desc.element, &v.desc.shape)?;
                }
            }
            let n = r.hits.rays.count;
            input(&r.chief_index, U32, &[n])?;
            input(&r.reference, D, &[r.chiefs.rays.count, 3])?;
            output(p, r.output, D, &[n])?;
            output(p, r.status, U32, &[n])
        }
        Request::Illumination(r) => {
            let nr = r.chiefs.rays.count;
            let n = r.nodes.desc.shape.first().copied().unwrap_or(0);
            input(
                &r.nodes,
                U32,
                &[n, crate::consts::MKL_AIM_SAMPLE_WORDS as usize],
            )?;
            input(
                &r.fields,
                U32,
                &[nr, crate::consts::MKL_AIM_FIELD_WORDS as usize],
            )?;
            for id in [r.chiefs.records, r.chiefs.record_low] {
                output(
                    p,
                    id.ok_or_else(|| bad("illumination chief records"))?,
                    ElementType::F32,
                    &[r.slots, nr, RECORD_FLOATS],
                )?;
            }
            output(p, r.chiefs.status, U32, &[3, nr])?;
            output(p, r.ids, U32, &[n])?;
            output(p, r.output, D, &[4, n])?;
            output(p, r.status, U32, &[n])
        }
        Request::CloseFocus(r) => {
            let n = r.hits.rays.count;
            if n % 5 != 0 {
                return Err(bad("close-ray pencil probes"));
            }
            for v in hit_views(&r.hits) {
                output(p, v.buffer, v.desc.element, &v.desc.shape)?;
            }
            output(p, r.output, D, &[n / 5, 3])?;
            output(p, r.status, U32, &[n / 5])
        }
        Request::Coddington(r) => {
            let n = r.hits.rays.count;
            for id in [r.hits.records, r.hits.record_low] {
                output(
                    p,
                    id.ok_or_else(|| bad("Coddington records"))?,
                    ElementType::F32,
                    &[r.slots, n, RECORD_FLOATS],
                )?;
            }
            output(p, r.hits.rays.indices, U32, &[2, n])?;
            output(p, r.hits.status, U32, &[3, n])?;
            output(p, r.output, D, &[n, 3])?;
            output(p, r.status, U32, &[n])
        }
        Request::Records(r) => {
            for v in hit_views(&r.hits) {
                output(p, v.buffer, v.desc.element, &v.desc.shape)?;
            }
            for id in [r.hits.records, r.hits.record_low] {
                output(
                    p,
                    id.ok_or_else(|| bad("geometry requires trace records"))?,
                    ElementType::F32,
                    &[r.slots, r.hits.rays.count, RECORD_FLOATS],
                )?;
            }
            input(&r.frames, D, &[r.slots, 12])?;
            if r.origin.iter().any(|v| !v.to_f64().is_finite()) {
                return Err(bad("geometry display origin"));
            }
            let n = r
                .slots
                .checked_mul(r.hits.rays.count)
                .ok_or_else(|| bad("record extent overflow"))?;
            output(p, r.output, D, &[r.slots, r.hits.rays.count, 3])?;
            output(p, r.radii, D, &[1, n])?;
            output(p, r.mask, U32, &[n])?;
            output(p, r.failures, D, &[r.hits.rays.count])?;
            output(p, r.launch, D, &[r.hits.rays.count, 3])
        }
        Request::FieldData(r) => {
            if r.hits.rays.count == 0 || r.kind > 2 || (r.kind == 1 && r.hits.rays.count % 5 != 0) {
                return Err(bad("geometry field probes"));
            }
            for v in hit_views(&r.hits) {
                output(p, v.buffer, v.desc.element, &v.desc.shape)?;
            }
            if r.kind != 1 {
                for id in [r.hits.records, r.hits.record_low] {
                    let id = id.ok_or_else(|| bad("chief incidence requires records"))?;
                    let d = p.desc(id)?;
                    if d.element != ElementType::F32
                        || d.shape.len() != 3
                        || d.shape[0] == 0
                        || d.shape[0] != r.record_slots
                        || d.shape[1..] != [r.hits.rays.count, RECORD_FLOATS]
                    {
                        return Err(bad("chief normal layout"));
                    }
                }
            }
            output(
                p,
                r.output,
                D,
                &[r.hits.rays.count / if r.kind == 1 { 5 } else { 1 }, 5],
            )
        }
        Request::Distribution(r) => {
            let n = r.keys.desc.len()?;
            let k = r.radii.desc.len()?;
            input(&r.keys, D, &[n])?;
            input(&r.cumulative, D, &[n])?;
            input(&r.radii, D, &[k])?;
            if !r.uncertainty.to_f64().is_finite() || r.uncertainty.to_f64() < 0.0 {
                return Err(bad("energy distance uncertainty"));
            }
            output(p, r.boundary, D, &[k])?;
            output(p, r.output, D, &[k + 4])
        }
        Request::SurfaceSamples(r) => {
            let n = r.samples.desc.shape.first().copied().unwrap_or(0);
            input(
                &r.samples,
                U32,
                &[n, crate::consts::MKL_GEOMETRY_SAMPLE_WORDS as usize],
            )?;
            if let Some(v) = &r.projection {
                input(&v.xy, D, &[n, 2])?;
                let rows = v.frames.desc.shape.first().copied().unwrap_or(0);
                input(&v.frames, D, &[rows, 12])?;
                if v.origin.iter().any(|v| !v.to_f64().is_finite()) {
                    return Err(bad("geometry vertex origin"));
                }
            }
            output(p, r.output, D, &[n, 3])?;
            output(p, r.status, U32, &[n])
        }
    }
}
fn points(v: &PlanView, mask: &PlanView, segment: usize) -> MkResult<usize> {
    let n = v.desc.shape.get(1).copied().unwrap_or(0);
    let channels = crate::consts::MKL_GEOMETRY_POINT_CHANNELS as usize;
    input(v, ElementType::DoubleFloat, &[channels, n])?;
    input(mask, ElementType::U32, &[n])?;
    if segment == 0 || n % segment != 0 {
        return Err(bad("geometry segment length"));
    }
    Ok(n)
}
fn system_reads(a: &mut Access, systems: &SystemSet) {
    match systems {
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
        Request::FocusTerms(r) => {
            a.reads = vec![
                r.points.clone(),
                r.mask.clone(),
                r.position_moments.clone(),
                r.slope_moments.clone(),
            ];
            a.writes = vec![r.output];
        }
        Request::Opd(r) => {
            a.reads.extend(hit_views(&r.hits));
            a.reads.extend(hit_views(&r.chiefs));
            a.reads.extend([r.chief_index.clone(), r.reference.clone()]);
            a.writes = vec![r.output, r.status];
        }
        Request::Illumination(r) => {
            a.reads = vec![
                r.fields.clone(),
                r.nodes.clone(),
                view(r.chiefs.status, ElementType::U32, [0]),
            ];
            for id in [r.chiefs.records, r.chiefs.record_low]
                .into_iter()
                .flatten()
            {
                a.reads.push(view(id, ElementType::F32, [0]));
            }
            system_reads(&mut a, &r.systems);
            a.writes = vec![r.output, r.status, r.ids];
        }
        Request::CloseFocus(r) => {
            a.reads.extend(hit_views(&r.hits));
            a.writes = vec![r.output, r.status];
        }
        Request::Coddington(r) => {
            for id in [r.hits.records, r.hits.record_low].into_iter().flatten() {
                a.reads.push(view(id, ElementType::F32, [0]));
            }
            for id in [r.hits.rays.indices, r.hits.status] {
                a.reads.push(view(id, ElementType::U32, [0]));
            }
            system_reads(&mut a, &r.systems);
            a.writes = vec![r.output, r.status];
        }
        Request::Records(r) => {
            a.reads.push(r.frames.clone());
            a.reads.extend(hit_views(&r.hits));
            for id in [r.hits.records, r.hits.record_low].into_iter().flatten() {
                a.reads.push(view(id, ElementType::F32, [0]));
            }
            a.writes = vec![r.output, r.radii, r.mask, r.failures, r.launch];
        }
        Request::FieldData(r) => {
            a.reads.extend(hit_views(&r.hits));
            if r.kind != 1 {
                for id in [r.hits.records, r.hits.record_low].into_iter().flatten() {
                    a.reads.push(view(id, ElementType::F32, [0]));
                }
            }
            a.writes = vec![r.output];
        }
        Request::Distribution(r) => {
            a.reads = vec![r.keys.clone(), r.cumulative.clone(), r.radii.clone()];
            a.writes = vec![r.output, r.boundary];
        }
        Request::SurfaceSamples(r) => {
            if let Some(v) = &r.projection {
                a.reads.extend([v.xy.clone(), v.frames.clone()]);
            }
            a.reads.push(r.samples.clone());
            a.writes = vec![r.output, r.status];
            system_reads(&mut a, &r.systems);
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
