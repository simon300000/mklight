//! Resident ray generation, double-float aiming, trace and pupil projection.
//! Numeric row layouts are in `mklight_aiming_types.h`; helpers pack only
//! field/first-order metadata and explicit pupil probes. No per-ray CPU
//! tracing is hidden in a Metal operation. Unresolved rays retain their IDs.
//!
//! `ResidentRays`: float [15,N] in legacy input-plane order, uint [2,N]
//! (system,wavelength), uint [2,N] logical IDs, uint [4,N] aiming reports.
//! Trace status is uint [3,N] (packed status,last surface,completed steps).
//! Optional paired record planes are float [slots,N,13]: position (3),
//! direction (3), uncorrected OPL, weight, unnormalised normal (3), medium
//! index, status bits. The low plane carries residuals; weight/status lows
//! are zero. No legacy point-rounding correction enters resident geometry.
//!
//! A pupil reference has DoubleFloat [7] chief (position,direction,OPL) and
//! DoubleFloat [3] centre, in the image frame. Projection publishes
//! DoubleFloat [4,N]: OPD in waves and the unit-radius sphere x/y/z, with z
//! mirrored for negative propagation. P6 constructs the pupil basis, scale,
//! mask and rim topology. Explicit probes support arbitrary/annular rims;
//! Lattice and Fan generate regular probes without a per-ray input upload.
//!
//! Failure codes extend the shared low-byte code space; aim failures name
//! the stop in the upper surface bits. Sparse output retains original order.
//! There is no automatic f64 polish: adapters may read only that sparse
//! subset, preserve its IDs, and report any later polish in their own run.
//! Generation copies only a four-byte failure counter back to PlanStats;
//! all optical intermediates remain resident. Cached seeds require an
//! immutable field content key and the full plan identity; use no key for
//! plan-local candidate fields. The uploaded system handle also keys seeds.
//! Table-coating radiometry remains Unsupported pending the radiometry
//! package; geometry-only tracing works with those systems. Real-image-height
//! wavelength chunks must include the primary wavelength (up to 15 other
//! wavelengths per chunk); a missing primary is an explicit unresolved reason,
//! never a solve at another wavelength.

use crate::{
    Df32,
    consts::*,
    resident::{self, Access, BufferDesc, ElementType, ResidentPlan, SystemSet, bad},
};
mod cpu;
pub mod kernel;
mod records;
use crate::resident::{BufferId, PlanView};
use mklight_core::MkResult;
pub use records::*;
#[derive(Clone, Debug)]
pub struct ResidentRays {
    pub floats: BufferId,
    pub indices: BufferId,
    pub count: usize,
    pub logical_ids: BufferId,
    /// u32 [4,count]: failure code, iterations, relative residual hi/lo bits.
    pub report: BufferId,
}
#[derive(Clone, Debug)]
pub struct ResidentHits {
    pub rays: ResidentRays,
    pub status: BufferId,
    pub records: Option<BufferId>,
    pub record_low: Option<BufferId>,
}
/// Device construction of regular probes; explicit tables preserve arbitrary
/// rings, annular boundaries and caller sampling. Lattices include the square
/// guard nodes outside the disk, for later rim topology and masking.
#[derive(Clone, Copy, Debug)]
pub enum Sampling {
    Explicit,
    Lattice { side: u32, extent: Df32 },
    Fan { axis: u32, extent: Df32 },
}
impl Sampling {
    pub fn parameters(self) -> (u32, u32, Df32) {
        match self {
            Self::Explicit => (0, 0, Df32::from_f32(1.0)),
            Self::Lattice { side, extent } => (1, side, extent),
            Self::Fan { axis, extent } => (2, axis, extent),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerateAim {
    pub systems: crate::resident::SystemSet,
    /// Explicit: one Sample per ray. Procedural: one base Sample, with field,
    /// weight and first logical ID; pupil coordinates are generated on device.
    pub samples: PlanView,
    pub sampling: Sampling,
    /// Optional stable sparse ray indices: u32 [count+1], count at element 0,
    /// ascending original indices next, unused tail = u32::MAX.
    pub unresolved: Option<BufferId>,
    pub field_wavelength: PlanView,
    pub seed: u64,
    /// Content identity of immutable field metadata, for chief-seed reuse.
    /// None for plan-local candidate fields (no cross-plan cache).
    pub field_key: Option<String>,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub output: ResidentRays,
}
#[derive(Clone, Debug)]
pub struct TraceResident {
    pub systems: crate::resident::SystemSet,
    pub input: ResidentRays,
    pub steps: PlanView,
    pub start_surface: u32,
    pub apply_apertures: bool,
    pub radiometry: u32,
    pub newton: crate::NewtonSettings,
    pub output: ResidentHits,
}
#[derive(Clone, Debug)]
pub struct PupilReference {
    pub chief: PlanView,
    pub centre: PlanView,
    pub radius: crate::Df32,
    pub image_index: crate::Df32,
    pub wavelength_um: crate::Df32,
    pub sphere_sheet: i32,
    pub telecentric: bool,
}
#[derive(Clone, Debug)]
pub struct ProjectPupil {
    pub hits: ResidentHits,
    pub reference: PupilReference,
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    GenerateAim(GenerateAim),
    TraceResident(TraceResident),
    ProjectPupil(ProjectPupil),
}

pub fn rays(plan: &mut ResidentPlan, count: usize) -> ResidentRays {
    ResidentRays {
        floats: plan.scratch(BufferDesc::new(ElementType::F32, [15, count])),
        indices: plan.scratch(BufferDesc::new(ElementType::U32, [2, count])),
        logical_ids: plan.scratch(BufferDesc::new(ElementType::U32, [2, count])),
        report: plan.scratch(BufferDesc::new(ElementType::U32, [4, count])),
        count,
    }
}
pub fn hits(plan: &mut ResidentPlan, count: usize, record_slots: usize) -> ResidentHits {
    ResidentHits {
        rays: rays(plan, count),
        status: plan.scratch(BufferDesc::new(ElementType::U32, [3, count])),
        records: (record_slots > 0).then(|| {
            plan.scratch(BufferDesc::new(
                ElementType::F32,
                [record_slots, count, MKL_AIM_RECORD_FLOATS as usize],
            ))
        }),
        record_low: (record_slots > 0).then(|| {
            plan.scratch(BufferDesc::new(
                ElementType::F32,
                [record_slots, count, MKL_AIM_RECORD_FLOATS as usize],
            ))
        }),
    }
}
fn view(id: BufferId, element: ElementType, shape: impl Into<Vec<usize>>) -> PlanView {
    PlanView {
        buffer: id,
        offset: 0,
        stride: 1,
        desc: BufferDesc::new(element, shape),
    }
}
fn ray_views(r: &ResidentRays) -> Vec<PlanView> {
    vec![
        view(r.floats, ElementType::F32, [15, r.count]),
        view(r.indices, ElementType::U32, [2, r.count]),
        view(r.logical_ids, ElementType::U32, [2, r.count]),
        view(r.report, ElementType::U32, [4, r.count]),
    ]
}
fn ray_ids(r: &ResidentRays) -> Vec<BufferId> {
    vec![r.floats, r.indices, r.logical_ids, r.report]
}
fn check_view(v: &PlanView, element: ElementType, shape: &[usize]) -> MkResult<()> {
    if v.stride != 1 || v.desc != BufferDesc::new(element, shape) {
        return Err(bad("aiming input layout"));
    }
    Ok(())
}
fn check_rays(r: &ResidentRays, p: &ResidentPlan) -> MkResult<()> {
    if r.count > u32::MAX as usize {
        return Err(bad("aiming ray count"));
    }
    for v in ray_views(r) {
        if p.desc(v.buffer)? != &v.desc {
            return Err(bad("resident ray layout"));
        }
    }
    Ok(())
}
fn check_systems(s: &SystemSet, p: &ResidentPlan) -> MkResult<()> {
    if let SystemSet::Resident {
        headers,
        surfaces,
        coefficients,
        count,
        max_surfaces,
        max_wavelengths,
    } = s
    {
        if *count == 0
            || *max_surfaces < 2
            || *max_surfaces > u32::MAX as usize
            || *max_wavelengths > crate::layout::MAX_WAVELENGTHS
        {
            return Err(bad("resident system dimensions"));
        }
        if p.desc(*headers)?.element != ElementType::U32
            || p.desc(*headers)?.len()? != count * size_of::<crate::layout::PackedHeader>() / 4
            || p.desc(*surfaces)?.element != ElementType::U32
            || p.desc(*surfaces)?.len()?
                != count * max_surfaces * size_of::<crate::layout::PackedSurface>() / 4
            || p.desc(*coefficients)?.element != ElementType::DoubleFloat
        {
            return Err(bad("resident system sidecars"));
        }
    }
    Ok(())
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    let output = |id, element, shape: &[usize]| {
        if p.desc(id)? != &BufferDesc::new(element, shape) {
            return Err(bad("aiming output layout"));
        }
        Ok(())
    };
    match r {
        Request::GenerateAim(a) => {
            check_systems(&a.systems, p)?;
            check_rays(&a.output, p)?;
            check_view(
                &a.samples,
                ElementType::U32,
                &[
                    if matches!(a.sampling, Sampling::Explicit) {
                        a.output.count
                    } else {
                        1
                    },
                    MKL_AIM_SAMPLE_WORDS as usize,
                ],
            )?;
            let (kind, side, extent) = a.sampling.parameters();
            if kind != 0
                && (!extent.to_f64().is_finite()
                    || extent.to_f64() <= 0.0
                    || (kind == 1
                        && (side == 0 || side as u64 * side as u64 != a.output.count as u64))
                    || (kind == 2 && side > 1))
            {
                return Err(bad("aiming sampling dimensions"));
            }
            if let Some(id) = a.unresolved {
                output(id, ElementType::U32, &[a.output.count + 1])?;
            }
            let shape = &a.field_wavelength.desc.shape;
            if shape.len() != 2 || shape[0] == 0 || shape[0] > u32::MAX as usize {
                return Err(bad("aiming field rows"));
            }
            check_view(
                &a.field_wavelength,
                ElementType::U32,
                &[shape[0], MKL_AIM_FIELD_WORDS as usize],
            )?;
            if !a.tolerance.is_finite() || a.tolerance <= 0.0 {
                return Err(bad("aim tolerance must be positive"));
            }
        }
        Request::TraceResident(t) => {
            check_systems(&t.systems, p)?;
            check_rays(&t.input, p)?;
            check_rays(&t.output.rays, p)?;
            if t.input.count != t.output.rays.count
                || t.radiometry > 1
                || t.steps.desc.shape.len() != 1
                || (t.steps.desc.len()? == 0 || t.steps.desc.len()? > MKL_MAX_STEPS as usize)
            {
                return Err(bad("resident trace settings"));
            }
            if !t.newton.tolerance_abs.is_finite()
                || t.newton.tolerance_abs < 0.0
                || !t.newton.tolerance_rel.is_finite()
                || t.newton.tolerance_rel < 0.0
            {
                return Err(bad("trace Newton tolerance"));
            }
            check_view(&t.steps, ElementType::U32, &t.steps.desc.shape)?;
            output(t.output.status, ElementType::U32, &[3, t.input.count])?;
            if t.output.records.is_some() != t.output.record_low.is_some() {
                return Err(bad("records require paired residuals"));
            }
            for id in [t.output.records, t.output.record_low]
                .into_iter()
                .flatten()
            {
                output(
                    id,
                    ElementType::F32,
                    &[
                        t.steps.desc.len()? + 1,
                        t.input.count,
                        MKL_AIM_RECORD_FLOATS as usize,
                    ],
                )?;
            }
        }
        Request::ProjectPupil(q) => {
            check_rays(&q.hits.rays, p)?;
            output(q.hits.status, ElementType::U32, &[3, q.hits.rays.count])?;
            check_view(&q.reference.chief, ElementType::DoubleFloat, &[7])?;
            check_view(&q.reference.centre, ElementType::DoubleFloat, &[3])?;
            let v = &q.reference;
            if !matches!(v.sphere_sheet, -1 | 1)
                || !v.image_index.to_f64().is_finite()
                || v.image_index.to_f64() <= 0.0
                || !v.wavelength_um.to_f64().is_finite()
                || v.wavelength_um.to_f64() <= 0.0
                || (!v.telecentric && (!v.radius.to_f64().is_finite() || v.radius.to_f64() <= 0.0))
            {
                return Err(bad("pupil reference"));
            }
            output(q.output, ElementType::DoubleFloat, &[4, q.hits.rays.count])?;
            output(q.status, ElementType::U32, &[q.hits.rays.count])?;
        }
    }
    Ok(())
}
fn system_access(s: &SystemSet, a: &mut Access) {
    match s {
        SystemSet::Uploaded(h) => a.systems.push(*h),
        SystemSet::Resident {
            headers,
            surfaces,
            coefficients,
            ..
        } => {
            a.reads.extend([
                view(*headers, ElementType::U32, [0]),
                view(*surfaces, ElementType::U32, [0]),
                view(*coefficients, ElementType::DoubleFloat, [0]),
            ]);
        }
    }
}
pub fn accesses(r: &Request) -> Access {
    let mut a = Access::default();
    match r {
        Request::GenerateAim(g) => {
            system_access(&g.systems, &mut a);
            a.reads
                .extend([g.samples.clone(), g.field_wavelength.clone()]);
            a.writes = ray_ids(&g.output);
            a.writes.extend(g.unresolved);
        }
        Request::TraceResident(t) => {
            system_access(&t.systems, &mut a);
            a.reads = ray_views(&t.input).into_iter().chain(a.reads).collect();
            a.reads.push(t.steps.clone());
            a.writes = ray_ids(&t.output.rays);
            a.writes.push(t.output.status);
            a.writes.extend(
                [t.output.records, t.output.record_low]
                    .into_iter()
                    .flatten(),
            );
        }
        Request::ProjectPupil(p) => {
            a.reads = ray_views(&p.hits.rays);
            a.reads.extend([
                view(p.hits.status, ElementType::U32, [3, p.hits.rays.count]),
                p.reference.chief.clone(),
                p.reference.centre.clone(),
            ]);
            a.writes = vec![p.output, p.status];
        }
    }
    a
}
pub fn execute(
    r: &Request,
    x: &mut resident::cpu::Execution,
    ctl: &crate::pending::JobControl,
) -> MkResult<()> {
    cpu::execute(r, x, ctl)
}
