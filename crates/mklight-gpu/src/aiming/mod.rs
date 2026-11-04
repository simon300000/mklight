//! Resident ray generation, double-float aiming, trace and pupil projection.
//! Row layouts are in `mklight_aiming_types.h`; `kernel` is the CPU-f32
//! specification of `aiming.metal`.
//!
//! * `ResidentRays`: f32 [15,N] in legacy input-plane order (hi planes, then
//!   position/direction lo), u32 [2,N] (system, wavelength), u32 [2,N]
//!   logical IDs and u32 [4,N] reports (status, Newton iterations, relative
//!   stop residual hi/lo bits). Unresolved rays keep their IDs; aim failures
//!   name the stop in the status surface bits.
//! * `ResidentHits`: trace status u32 [3,N] (status, last surface, steps) and
//!   optional paired f32 records [slots,N,13]: position, direction, OPL
//!   without the legacy rounding correction, weight, unnormalised normal,
//!   medium index, status bits; the low plane holds the residuals.
//! * `ProjectPupil`: chief DF [7] (position, direction, OPL) and centre DF
//!   [3] in the image frame give DF [4,N]: OPD in waves and the unit-radius
//!   sphere x/y/z (z mirrored for negative propagation).
//!
//! There is no automatic f64 polish: `GenerateAim.unresolved` lists the
//! unresolved indices for an adapter, and `PlanStats::fallback_count` counts
//! them. Real-image-height fields need the primary wavelength in the packed
//! chunk; a missing primary is reported, never replaced. Samples excluded by
//! [`AimOptions::cull_outside`] or the `MKL_AIM_SKIP` field flag carry
//! `MKL_AIM_OUTSIDE`: they are neither aimed, traced, counted nor listed.
//!
//! Device aiming is resumable: every command runs a bounded number of
//! iteration units per ray (a phase entry or one Newton iteration), parks
//! unfinished rays in a state buffer and continues them in further bounded
//! commands, so no command's duration depends on the hardest ray's whole
//! schedule. The arithmetic of a resumed solve is the specification's,
//! operation for operation.
//!
//! Chief rows normally come from the host: [`ChiefSeed::host`] converts the
//! reference tracer's f64 `PreparedField` (real-image-height source once per
//! field, chief and Jacobian once per field and wavelength). The device chief
//! solve remains for requests without `seeds`. Chief rows and seed tables are
//! reused across plans by [`aim_identity`], a digest of every input.

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
    /// Opts into cross-plan reuse of chief rows and seed tables: the caller's
    /// name for its immutable field metadata. The executor keys reuse by
    /// [`aim_identity`] of the actual inputs, never by this name or by handles.
    /// None keeps them plan-local (candidate fields).
    pub field_key: Option<String>,
    /// Host-prepared chief seeds, one [`ChiefSeed`] per field row (u32 [rows,
    /// MKL_AIM_CACHE_WORDS]), normally [`ChiefSeed::host`]. When present the
    /// device chief/image-height solve is skipped and these rows seed the
    /// seed table and pupil aiming directly.
    pub seeds: Option<PlanView>,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub output: ResidentRays,
    pub options: AimOptions,
}
/// Seed-table and sample-exclusion choices of a generation.
#[derive(Clone, Debug)]
pub struct AimOptions {
    /// Host-built seed table (u32 [rows · side², MKL_AIM_NODE_WORDS],
    /// normally [`host_tables`]) used for every sample of the request, in
    /// place of a device-built table. Converged rays differ from those of a
    /// device table or of chief seeds only within the tolerance.
    pub table: Option<PlanView>,
    /// Nodes per axis of the host or device table (odd, 5..=129).
    pub table_side: u32,
    /// Exclude samples outside the unit pupil disk before aiming
    /// (`MKL_AIM_OUTSIDE`): lattice guard nodes a disk mask discards anyway,
    /// which are the costliest aims and never needed.
    pub cull_outside: bool,
    /// Continuation rounds a backend may commit without reading the pending
    /// count back (for generations expected to converge within them, e.g.
    /// from seed tables). A backend that speculates verifies at completion
    /// that no ray was left pending and otherwise runs the plan again with
    /// read-back rounds, so results never depend on it. 0: no speculation.
    pub speculative_rounds: u32,
}
impl Default for AimOptions {
    fn default() -> Self {
        Self {
            table: None,
            table_side: MKL_AIM_TABLE_SIDE,
            cull_outside: false,
            speculative_rounds: 0,
        }
    }
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

/// Seed-table nodes per field row at the default side.
pub const TABLE_NODES: usize = (MKL_AIM_TABLE_SIDE * MKL_AIM_TABLE_SIDE) as usize;
pub fn table_nodes(side: u32) -> usize {
    (side * side) as usize
}
/// Generation aims from an interpolated seed table (built once per cached
/// field row) when the plan has at least this many rays per field row (twice
/// the node count; this is the default-side value); smaller plans aim from
/// the chief seeds alone. Both reach the requested tolerance; their converged
/// rays differ only within it. A host table ([`AimOptions::table`]) is used
/// whatever the size.
pub const TABLE_MIN_RAYS_PER_ROW: usize = 2 * TABLE_NODES;
pub fn uses_table(g: &GenerateAim) -> bool {
    g.options.table.is_some()
        || g.output.count
            >= 2 * table_nodes(g.options.table_side) * g.field_wavelength.desc.shape[0]
}
/// Content identity of a generation's chief rows and seed table: the packed
/// system content, the exact field rows, the host seed rows (or the device
/// solve), tolerance, iteration cap and table side. None when the caller
/// keeps them plan-local (`field_key` None) or an input is not immutable at
/// submission: device-built systems, or field/seed rows produced inside the
/// plan.
pub fn aim_identity(
    g: &GenerateAim,
    system: Option<blake3::Hash>,
    fields: Option<&[u32]>,
    seeds: Option<&[u32]>,
) -> Option<String> {
    let name = g.field_key.as_ref()?;
    let seeds = match (&g.seeds, seeds) {
        (None, _) => "device".to_string(),
        (Some(_), Some(words)) => blake3::hash(bytemuck::cast_slice(words))
            .to_hex()
            .to_string(),
        (Some(_), None) => return None,
    };
    Some(format!(
        "{name}:{}:{}:{seeds}:{}:{}:{}",
        system?.to_hex(),
        blake3::hash(bytemuck::cast_slice(fields?)).to_hex(),
        g.tolerance.to_bits(),
        g.max_iterations,
        g.options.table_side
    ))
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
            || *count > u32::MAX as usize
            || *max_surfaces < 2
            || *max_surfaces > u32::MAX as usize
            || *max_wavelengths > crate::layout::MAX_WAVELENGTHS
        {
            return Err(bad("resident system dimensions"));
        }
        let header_words = count
            .checked_mul(size_of::<crate::layout::PackedHeader>() / 4)
            .ok_or_else(|| bad("resident system extent overflow"))?;
        let surface_words = count
            .checked_mul(*max_surfaces)
            .and_then(|n| n.checked_mul(size_of::<crate::layout::PackedSurface>() / 4))
            .ok_or_else(|| bad("resident system extent overflow"))?;
        if p.desc(*headers)?.element != ElementType::U32
            || p.desc(*headers)?.len()? != header_words
            || p.desc(*surfaces)?.element != ElementType::U32
            || p.desc(*surfaces)?.len()? != surface_words
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
            if let Some(v) = &a.seeds {
                check_view(
                    v,
                    ElementType::U32,
                    &[shape[0], MKL_AIM_CACHE_WORDS as usize],
                )?;
            }
            let side = a.options.table_side;
            if !(5..=129).contains(&side) || side % 2 == 0 {
                return Err(bad("aim table side"));
            }
            if let Some(v) = &a.options.table {
                check_view(
                    v,
                    ElementType::U32,
                    &[shape[0] * table_nodes(side), MKL_AIM_NODE_WORDS as usize],
                )?;
            }
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
            a.reads.extend(g.seeds.clone());
            a.reads.extend(g.options.table.clone());
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
