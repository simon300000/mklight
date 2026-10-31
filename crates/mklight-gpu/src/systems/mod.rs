//! Resident parameter batches. Compile model addresses/dependencies once, upload
//! immutable topology once, and upload only ordered parameter rows thereafter.
//!
//! Numeric preparation is not optical qualification. Consumers must qualify each
//! candidate (or a tested envelope) before using its rays, and must honour the map
//! status and dirty flags. Unsupported compilation uses `reference_batch`, whose
//! report retains the reason and original candidate slots.
mod algebra;
mod compile;
mod frames;
mod materials;
mod shapes;

use crate::{ComputeBackend, Df32, PackedSystem, SystemHandle, resident::*};
use mklight_core::{
    JobContext, MkError, MkResult, ResolvedSystem, Snapshot,
    materials::MaterialLibrary,
    model::{MaterialRef, SurfaceId},
    param::ParamPath,
};
use std::sync::Arc;

pub const VERSION: u32 = 1;
pub const INSTRUCTIONS_PER_COMMAND: usize = 2048;
pub const READY: u32 = 0;
pub const OUTSIDE_BOUNDS: u32 = 1;
pub const INVALID_NUMERIC: u32 = 2;
pub const DIRTY_AIM: u32 = 1;
pub const DIRTY_APERTURES: u32 = 2;
pub const DIRTY_SPECTRUM: u32 = 4;
pub const DIRTY_ALL: u32 = DIRTY_AIM | DIRTY_APERTURES | DIRTY_SPECTRUM;

#[derive(Clone, Debug)]
pub struct Parameter {
    pub path: ParamPath,
    pub lower: f64,
    pub upper: f64,
}
/// A discrete material column follows the continuous columns in Candidate.values.
/// Values must be exact integers indexing choices. Catalogue filters stay host-side.
#[derive(Clone, Debug)]
pub struct MaterialSlot {
    pub surface: SurfaceId,
    pub choices: Vec<MaterialRef>,
}
#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: u64,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct CompiledSystem {
    snapshot: Snapshot,
    library: Arc<MaterialLibrary>,
    variables: Vec<Parameter>,
    material_slots: Vec<MaterialSlot>,
    configuration: Option<usize>,
    program: Arc<Program>,
    fingerprint: String,
    nominal: Vec<PackedSystem>,
}
#[derive(Clone, Debug)]
struct Program {
    code: Vec<[u32; 4]>,
    patches: Vec<[u32; 4]>,
    guards: Vec<u32>,
    headers: Vec<u32>,
    surfaces: Vec<u32>,
    coefficients: Vec<u32>,
    first_order_len: usize,
    model_len: usize,
    surface_count: usize,
    wavelength_count: usize,
    chunk_count: usize,
}
/// Immutable device template. Keep it alive across FD/population batches.
/// The base handle is explicitly released by `release` after the last append.
#[derive(Clone, Debug)]
pub struct UploadedSystem {
    compiled: CompiledSystem,
    base: SystemHandle,
    code: BufferHandle,
    template: BufferHandle,
    bounds: BufferHandle,
}
/// Candidate IDs are u64 split low/high, configurations are u32 (`MAX` = base).
/// Status is `[candidate,2]`: READY/OUTSIDE_BOUNDS/INVALID_NUMERIC, dirty flags.
/// Failure slots remain present; consumers must not trace them.
#[derive(Clone, Debug)]
pub struct CandidateMap {
    pub candidate_ids: BufferId,
    pub configuration_ids: BufferId,
    pub status: BufferId,
    pub count: usize,
}
/// Outputs use the existing packed layout, with global offsets rebased in
/// candidate-major / wavelength-chunk order. First order is DF
/// `[B,W,2,8+7*S]` (x then y): EFL,BFL,FFL,track,A,B,C,D followed by per-surface
/// `[signed_index,power,thickness,y_A,nu_A,y_B,nu_B]`. Basis heights at surface
/// zero are formal launch coordinates, not object-plane heights. Undefined
/// focal lengths retain IEEE infinities/NaNs; this is not a scaled pupil table.
#[derive(Clone, Debug)]
pub struct SystemBatchPlan {
    pub base: SystemHandle,
    pub parameters: PlanView,
    pub slots: PlanView,
    pub bounds: PlanView,
    pub dependencies: PlanView,
    pub material_tables: PlanView,
    pub headers: BufferId,
    pub surfaces: BufferId,
    pub coefficients: BufferId,
    pub first_order: BufferId,
    /// DF `[B,4+7*F+2*W+P+solves]`: aperture value (0 for stop-sized),
    /// temperature, pressure, object distance; field x/y/weight/vignetting;
    /// wavelength/weight; final parameter values; solve targets in declaration
    /// order. Discrete field/aperture kinds come from the immutable snapshot.
    pub model_data: BufferId,
    pub map: CandidateMap,
    program: Arc<Program>,
    imports: Vec<(PlanView, BufferHandle)>,
    parameter_count: usize,
    /// Fixed work partition; changes scheduling only, not arithmetic or IDs.
    pub tile_candidates: usize,
}
#[derive(Clone, Debug)]
pub enum Request {
    Prepare(SystemBatchPlan),
}

impl CompiledSystem {
    pub fn compile(
        snapshot: Snapshot,
        library: Arc<MaterialLibrary>,
        variables: Vec<Parameter>,
        configuration: Option<usize>,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        compile::compile(snapshot, library, variables, vec![], configuration, ctx)
    }
    pub fn compile_with_materials(
        snapshot: Snapshot,
        library: Arc<MaterialLibrary>,
        variables: Vec<Parameter>,
        materials: Vec<MaterialSlot>,
        configuration: Option<usize>,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        compile::compile(snapshot, library, variables, materials, configuration, ctx)
    }
    /// Stable transient identity, independent of batch order/chunking. A
    /// published candidate instead uses the normal Snapshot content hash from
    /// `reference` after reconstruction and solves.
    pub fn transient_hash(&self, candidate: &Candidate) -> String {
        let mut hash = blake3::Hasher::new();
        hash.update(self.snapshot.hash().to_hex().as_bytes());
        hash.update(self.fingerprint.as_bytes());
        for value in &candidate.values {
            hash.update(&value.to_bits().to_le_bytes());
        }
        hash.finalize().to_hex().to_string()
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }
    pub fn parameter_count(&self) -> usize {
        self.variables.len() + self.material_slots.len()
    }
    pub fn instruction_count(&self) -> usize {
        self.program.code.len()
    }
    pub fn upload(
        &self,
        backend: &dyn ComputeBackend,
        ctx: &JobContext,
    ) -> MkResult<UploadedSystem> {
        ctx.check()?;
        let key = SystemCacheKey {
            snapshot: self.snapshot.hash().to_hex(),
            library: self.fingerprint.clone(),
            configuration: format!("{:?}", self.configuration),
            spectrum: "all-chunks".into(),
            layout: crate::consts::MKL_LAYOUT_VERSION,
            kernel_build: backend.precision().kernel_build.clone(),
            operation_settings: format!("systems-v{VERSION}"),
        };
        let upload = |data: BufferData| {
            backend.upload_buffer(
                BufferDesc::new(
                    data.element(),
                    [data.words().len() / data.element().words()],
                ),
                data,
            )
        };
        let mut code: Vec<u32> = bytemuck::cast_slice(&self.program.code).to_vec();
        code.extend_from_slice(bytemuck::cast_slice(&self.program.patches));
        code.extend(&self.program.guards);
        let mut template = self.program.headers.clone();
        template.extend(&self.program.surfaces);
        template.extend(&self.program.coefficients);
        let code = upload(BufferData::U32(code))?;
        let template = upload(BufferData::U32(template))?;
        let bounds = upload(BufferData::DoubleFloat(
            self.variables
                .iter()
                .flat_map(|p| [Df32::from_f64(p.lower), Df32::from_f64(p.upper)])
                .chain(
                    self.material_slots
                        .iter()
                        .flat_map(|s| [Df32::ZERO, Df32::from_f64((s.choices.len() - 1) as f64)]),
                )
                .collect(),
        ))?;
        let base = backend.upload_systems_cached(&key, &self.nominal)?;
        Ok(UploadedSystem {
            compiled: self.clone(),
            base,
            code,
            template,
            bounds,
        })
    }
    /// Boundary reconstruction for publishing/independent qualification. This
    /// applies configuration overrides after the candidate, then reference solves.
    pub fn reference(&self, candidate: &Candidate, ctx: &JobContext) -> MkResult<ResolvedSystem> {
        reference_system(
            &self.snapshot,
            &self.library,
            &self.variables,
            &self.material_slots,
            candidate,
            self.configuration,
            ctx,
        )
    }
}
impl UploadedSystem {
    pub fn release(&self, backend: &dyn ComputeBackend) -> MkResult<()> {
        backend.release_systems(self.base)
    }
    pub fn append(
        &self,
        backend: &dyn ComputeBackend,
        plan: &mut ResidentPlan,
        candidates: &[Candidate],
        ctx: &JobContext,
    ) -> MkResult<SystemBatchPlan> {
        ctx.check()?;
        if plan.snapshot.hash() != self.compiled.snapshot.hash() {
            return Err(bad("system batch base snapshot mismatch"));
        }
        let count = candidates.len();
        let nparam = self.compiled.parameter_count();
        let mut parameters = Vec::with_capacity(count * nparam);
        let mut ids = Vec::with_capacity(count * 4);
        let mut hash = blake3::Hasher::new();
        hash.update(self.compiled.snapshot.hash().to_hex().as_bytes());
        for candidate in candidates {
            ctx.check()?;
            if candidate.values.len() != nparam {
                return Err(MkError::InvalidParam("candidate parameter width".into()));
            }
            let mut input_status = READY;
            for (column, &v) in candidate.values.iter().enumerate() {
                let (lower, upper) = if column < self.compiled.variables.len() {
                    let p = &self.compiled.variables[column];
                    (p.lower, p.upper)
                } else {
                    (
                        0.0,
                        (self.compiled.material_slots[column - self.compiled.variables.len()]
                            .choices
                            .len()
                            - 1) as f64,
                    )
                };
                if !v.is_finite() || v < lower || v > upper {
                    input_status = OUTSIDE_BOUNDS;
                }
                let d = if v.is_finite() {
                    Df32::from_f64(v)
                } else {
                    Df32::new(v as f32, 0.0)
                };
                if input_status == READY
                    && (!d.is_finite() || d.hi.is_subnormal() || (d.hi == 0.0 && v != 0.0))
                {
                    input_status = INVALID_NUMERIC;
                }
                hash.update(&v.to_bits().to_le_bytes());
                parameters.push(d);
            }
            ids.extend([
                candidate.id as u32,
                (candidate.id >> 32) as u32,
                self.compiled.configuration.map_or(u32::MAX, |k| k as u32),
                input_status,
            ]);
        }
        let parameters = backend.upload_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [count, nparam]),
            BufferData::DoubleFloat(parameters),
        )?;
        let slots = backend.upload_buffer(
            BufferDesc::new(ElementType::U32, [count, 4]),
            BufferData::U32(ids),
        )?;
        let parameters = plan.import(parameters.view());
        let slots = plan.import(slots.view());
        self.append_resident(
            plan,
            plan.view(parameters)?,
            plan.view(slots)?,
            &hash.finalize().to_hex().to_string(),
            ctx,
        )
    }
    /// Append rows produced by an earlier resident operation. Parameters are DF
    /// `[B,P]`, slots U32 `[B,4]` (ID low/high, configuration, input status).
    /// Arithmetic and bounds are checked on device. The supplied identity names
    /// the producer's seed/settings or parameter digest in the outer RunInfo.
    pub fn append_resident(
        &self,
        plan: &mut ResidentPlan,
        parameters: PlanView,
        slots: PlanView,
        candidate_identity: &str,
        ctx: &JobContext,
    ) -> MkResult<SystemBatchPlan> {
        ctx.check()?;
        if plan.snapshot.hash() != self.compiled.snapshot.hash() {
            return Err(bad("system batch base snapshot mismatch"));
        }
        let count = parameters
            .desc
            .shape
            .first()
            .copied()
            .ok_or_else(|| bad("candidate parameter shape"))?;
        let p = &self.compiled.program;
        let mut imports = vec![];
        let mut import = |h: &BufferHandle| {
            let id = plan.import(h.view());
            let view = plan.view(id).unwrap();
            imports.push((view.clone(), h.clone()));
            view
        };
        let bounds = import(&self.bounds);
        let dependencies = import(&self.code);
        let material_tables = import(&self.template);
        let headers = plan.scratch(BufferDesc::new(ElementType::U32, [count, p.headers.len()]));
        let surfaces = plan.scratch(BufferDesc::new(ElementType::U32, [count, p.surfaces.len()]));
        let coefficients = plan.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [count, p.coefficients.len() / 2],
        ));
        let first_order = plan.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [count, p.wavelength_count, 2, 8 + 7 * p.surface_count],
        ));
        let model_data = plan.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [count, p.model_len],
        ));
        let map = CandidateMap {
            candidate_ids: plan.scratch(BufferDesc::new(ElementType::U32, [count, 2])),
            configuration_ids: plan.scratch(BufferDesc::new(ElementType::U32, [count])),
            status: plan.scratch(BufferDesc::new(ElementType::U32, [count, 2])),
            count,
        };
        let request = SystemBatchPlan {
            base: self.base,
            parameters,
            slots,
            bounds,
            dependencies,
            material_tables,
            headers,
            surfaces,
            coefficients,
            first_order,
            model_data,
            map,
            program: p.clone(),
            imports,
            parameter_count: self.compiled.parameter_count(),
            tile_candidates: 64,
        };
        let entry = serde_json::json!({"version":VERSION,"program":self.compiled.fingerprint,"configuration":self.compiled.configuration,"candidate_vectors_hash":candidate_identity,"count":count,"qualification":"required_per_candidate_or_tested_envelope","dirty_flags":DIRTY_ALL});
        let settings = plan
            .settings
            .as_object_mut()
            .ok_or_else(|| bad("systems settings must be an object"))?;
        settings
            .entry("system_batches")
            .or_insert_with(|| serde_json::json!([]))
            .as_array_mut()
            .ok_or_else(|| bad("system_batches settings must be an array"))?
            .push(entry);
        plan.operations
            .push(ResidentOp::Systems(Request::Prepare(request.clone())));
        Ok(request)
    }
}
impl SystemBatchPlan {
    pub fn outputs(&self) -> Vec<BufferId> {
        vec![
            self.headers,
            self.surfaces,
            self.coefficients,
            self.first_order,
            self.map.candidate_ids,
            self.map.configuration_ids,
            self.map.status,
            self.model_data,
        ]
    }
    pub fn system_set(&self) -> SystemSet {
        SystemSet::Resident {
            headers: self.headers,
            surfaces: self.surfaces,
            coefficients: self.coefficients,
            count: self.map.count * self.program.chunk_count,
            max_surfaces: self.program.surface_count,
            max_wavelengths: self
                .program
                .wavelength_count
                .min(crate::layout::MAX_WAVELENGTHS),
        }
    }
    pub fn dispatch_params(&self, first: usize, count: usize) -> DispatchParams {
        let p = &self.program;
        DispatchParams {
            first: first as u32,
            count: count as u32,
            parameters: self.parameters.desc.shape[1] as u32,
            instructions: p.code.len() as u32,
            patches: p.patches.len() as u32,
            guards: p.guards.len() as u32,
            header_words: p.headers.len() as u32,
            surface_words: p.surfaces.len() as u32,
            coefficient_words: p.coefficients.len() as u32,
            first_order_words: (p.first_order_len * 2) as u32,
            chunks: p.chunk_count as u32,
            surfaces: p.surface_count as u32,
            model_words: (p.model_len * 2) as u32,
            first_instruction: 0,
            instruction_count: p.code.len() as u32,
            reserved: 0,
        }
    }
    /// Instruction scratch is bounded by a tile, independent of the batch size.
    pub fn scratch_words(&self) -> usize {
        self.program.code.len() * 2 * self.tile_candidates.min(self.map.count)
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchParams {
    pub first: u32,
    pub count: u32,
    pub parameters: u32,
    pub instructions: u32,
    pub patches: u32,
    pub guards: u32,
    pub header_words: u32,
    pub surface_words: u32,
    pub coefficient_words: u32,
    pub first_order_words: u32,
    pub chunks: u32,
    pub surfaces: u32,
    pub model_words: u32,
    pub first_instruction: u32,
    pub instruction_count: u32,
    pub reserved: u32,
}

pub fn validate(request: &Request, plan: &ResidentPlan) -> MkResult<()> {
    let Request::Prepare(r) = request;
    if r.tile_candidates == 0 || r.tile_candidates > 64 {
        return Err(bad("system preparation tile must be 1..=64"));
    }
    if r.parameters.desc
        != BufferDesc::new(ElementType::DoubleFloat, [r.map.count, r.parameter_count])
        || r.slots.desc != BufferDesc::new(ElementType::U32, [r.map.count, 4])
        || r.parameters.stride != 1
        || r.slots.stride != 1
    {
        return Err(bad("system preparation candidate shape"));
    }
    let views = [&r.bounds, &r.dependencies, &r.material_tables];
    for ((expected, handle), actual) in r.imports.iter().zip(views) {
        let Some(PlanBuffer::Import(v)) = plan.buffers.get(actual.buffer.0) else {
            return Err(bad(
                "system preparation requires its immutable uploaded inputs",
            ));
        };
        if actual.buffer != expected.buffer
            || actual.offset != 0
            || actual.stride != 1
            || actual.desc != expected.desc
            || v.offset != 0
            || v.stride != 1
            || v.desc != *handle.desc()
            || v.buffer.owner() != handle.owner()
            || v.buffer.generation() != handle.generation()
        {
            return Err(bad("system preparation input identity changed"));
        }
    }
    let p = &r.program;
    let n = r.map.count;
    let descs = [
        BufferDesc::new(ElementType::U32, [n, p.headers.len()]),
        BufferDesc::new(ElementType::U32, [n, p.surfaces.len()]),
        BufferDesc::new(ElementType::DoubleFloat, [n, p.coefficients.len() / 2]),
        BufferDesc::new(
            ElementType::DoubleFloat,
            [n, p.wavelength_count, 2, 8 + 7 * p.surface_count],
        ),
        BufferDesc::new(ElementType::U32, [n, 2]),
        BufferDesc::new(ElementType::U32, [n]),
        BufferDesc::new(ElementType::U32, [n, 2]),
        BufferDesc::new(ElementType::DoubleFloat, [n, p.model_len]),
    ];
    for (id, desc) in r.outputs().into_iter().zip(descs) {
        if plan.desc(id)? != &desc || desc.len()? > u32::MAX as usize {
            return Err(bad("system preparation output extent"));
        }
    }
    Ok(())
}
pub fn accesses(request: &Request) -> Access {
    let Request::Prepare(r) = request;
    Access {
        reads: vec![
            r.parameters.clone(),
            r.slots.clone(),
            r.bounds.clone(),
            r.dependencies.clone(),
            r.material_tables.clone(),
        ],
        writes: r.outputs(),
        systems: vec![r.base],
    }
}

pub fn execute(
    request: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &crate::pending::JobControl,
) -> MkResult<()> {
    let Request::Prepare(r) = request;
    let p = &r.program;
    let width = r.parameters.desc.shape[1];
    let mut values = vec![Df32::ZERO; p.code.len()];
    for row in 0..r.map.count {
        ctl.check()?;
        let parameters: Vec<Df32> =
            x.words(r.parameters.buffer)[(r.parameters.offset + row * width) * 2
                ..(r.parameters.offset + (row + 1) * width) * 2]
                .chunks_exact(2)
                .map(|v| Df32::new(f32::from_bits(v[0]), f32::from_bits(v[1])))
                .collect();
        let bounds: Vec<Df32> = x
            .words(r.bounds.buffer)
            .chunks_exact(2)
            .map(|v| Df32::new(f32::from_bits(v[0]), f32::from_bits(v[1])))
            .collect();
        let mut status = x.words(r.slots.buffer)[r.slots.offset + row * 4 + 3];
        for (i, &v) in parameters.iter().enumerate() {
            if !v.is_finite()
                || v.to_f64() < bounds[2 * i].to_f64()
                || v.to_f64() > bounds[2 * i + 1].to_f64()
            {
                status = OUTSIDE_BOUNDS;
            }
        }
        for first in (0..p.code.len()).step_by(INSTRUCTIONS_PER_COMMAND) {
            ctl.check()?;
            algebra::evaluate(
                &p.code,
                &parameters,
                &mut values,
                first..(first + INSTRUCTIONS_PER_COMMAND).min(p.code.len()),
            );
        }
        if status == READY
            && p.guards
                .iter()
                .any(|&i| !values[i as usize].is_finite() || values[i as usize].to_f64() <= 0.0)
        {
            status = INVALID_NUMERIC;
        }
        for (id, template) in [
            (r.headers, &p.headers),
            (r.surfaces, &p.surfaces),
            (r.coefficients, &p.coefficients),
        ] {
            x.words_mut(id)[row * template.len()..(row + 1) * template.len()]
                .copy_from_slice(template);
        }
        for &[dest, hi, lo, node] in &p.patches {
            let id = [
                r.headers,
                r.surfaces,
                r.coefficients,
                r.first_order,
                r.model_data,
            ][dest as usize];
            let stride = [
                p.headers.len(),
                p.surfaces.len(),
                p.coefficients.len(),
                p.first_order_len * 2,
                p.model_len * 2,
            ][dest as usize];
            let v = values[node as usize];
            if dest < 3 && !v.is_finite() && status == READY {
                status = INVALID_NUMERIC;
            }
            let out = x.words_mut(id);
            out[row * stride + hi as usize] = v.hi.to_bits();
            if lo != u32::MAX {
                out[row * stride + lo as usize] = v.lo.to_bits();
            }
        }
        for chunk in 0..p.chunk_count {
            let offset = row * p.headers.len()
                + chunk * size_of::<crate::PackedHeader>() / 4
                + std::mem::offset_of!(crate::PackedHeader, surface_offset) / 4;
            x.words_mut(r.headers)[offset] += (row * p.chunk_count * p.surface_count) as u32;
        }
        for j in 0..p.surface_count * p.chunk_count {
            let offset = row * p.surfaces.len()
                + j * size_of::<crate::PackedSurface>() / 4
                + std::mem::offset_of!(crate::PackedSurface, coefficient_offset) / 4;
            x.words_mut(r.surfaces)[offset] += (row * p.coefficients.len() / 2) as u32;
        }
        let ids = x.words(r.slots.buffer)[r.slots.offset + row * 4..r.slots.offset + row * 4 + 3]
            .to_vec();
        x.words_mut(r.map.candidate_ids)[row * 2..row * 2 + 2].copy_from_slice(&ids[..2]);
        x.words_mut(r.map.configuration_ids)[row] = ids[2];
        x.words_mut(r.map.status)[row * 2..row * 2 + 2].copy_from_slice(&[status, DIRTY_ALL]);
        ctl.report((row + 1) as f64 / r.map.count as f64, "prepare systems");
    }
    Ok(())
}

/// Explicit reference preparation; errors occupy their original slots. The
/// caller's reason is retained with a CPU-f64 RunInfo and a counted fallback.
#[derive(Debug)]
pub struct ReferenceBatch {
    pub candidates: Vec<(u64, MkResult<Vec<PackedSystem>>)>,
    pub run_info: mklight_core::job::RunInfo,
    pub fallback_count: usize,
}
fn reference_system(
    snapshot: &Snapshot,
    library: &Arc<MaterialLibrary>,
    variables: &[Parameter],
    materials: &[MaterialSlot],
    candidate: &Candidate,
    configuration: Option<usize>,
    ctx: &JobContext,
) -> MkResult<ResolvedSystem> {
    ctx.check()?;
    if candidate.values.len() != variables.len() + materials.len() {
        return Err(MkError::InvalidParam("candidate parameter width".into()));
    }
    let mut model = snapshot.system().clone();
    for (i, slot) in materials.iter().enumerate() {
        let choice = candidate.values[variables.len() + i];
        if !choice.is_finite()
            || choice.fract() != 0.0
            || choice < 0.0
            || choice >= slot.choices.len() as f64
        {
            return Err(MkError::InvalidParam("material choice index".into()));
        }
        let surface = model
            .surface_index(slot.surface)
            .ok_or_else(|| MkError::InvalidParam("material surface".into()))?;
        model.surfaces[surface].material = slot.choices[choice as usize].clone();
    }
    for (variable, &value) in variables.iter().zip(&candidate.values) {
        if !value.is_finite() || value < variable.lower || value > variable.upper {
            return Err(MkError::InvalidParam(
                "candidate outside parameter bounds".into(),
            ));
        }
        variable.path.set(&mut model, value)?;
    }
    let settings = mklight_trace::solves::SolveSettings::default();
    let solve_ctx = ctx.subrange(0.0, 0.9);
    let (model, _) = if let Some(k) = configuration {
        mklight_trace::solves::apply_solves_configuration(
            &model, library, k, &settings, &solve_ctx,
        )?
    } else {
        mklight_trace::solves::apply_solves_with(&model, library, &settings, &solve_ctx)?
    };
    let resolved = ResolvedSystem::new(&Snapshot::new(model)?, library.clone())?;
    ctx.check()?;
    ctx.report(1.0, "resolve candidate");
    Ok(resolved)
}
pub fn reference_batch(
    snapshot: &Snapshot,
    library: Arc<MaterialLibrary>,
    variables: &[Parameter],
    candidates: &[Candidate],
    configuration: Option<usize>,
    reason: &str,
    ctx: &JobContext,
) -> MkResult<ReferenceBatch> {
    let timer = mklight_core::job::RunTimer::start();
    let mut out = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        ctx.check()?;
        let sub = ctx.subrange(
            0.95 * i as f64 / candidates.len() as f64,
            0.95 * (i + 1) as f64 / candidates.len() as f64,
        );
        let result = reference_system(
            snapshot,
            &library,
            variables,
            &[],
            c,
            configuration,
            &sub.subrange(0.0, 0.9),
        )
        .and_then(|rs| PackedSystem::pack_chunks(&rs, None));
        if matches!(result, Err(MkError::Cancelled)) {
            return Err(MkError::Cancelled);
        }
        out.push((c.id, result));
        ctx.report(
            0.95 * (i + 1) as f64 / candidates.len() as f64,
            "reference system preparation",
        );
    }
    ctx.check()?;
    let settings = serde_json::json!({"version":VERSION,"fallback_reason":reason,"fallback_count":candidates.len(),"configuration":configuration,"candidate_ids":candidates.iter().map(|c|c.id).collect::<Vec<_>>()});
    let run_info = timer.finish(
        snapshot,
        "system_preparation",
        &settings,
        "cpu-f64-reference",
        mklight_core::job::Precision::F64Reference,
        None,
        vec![reason.into()],
    );
    ctx.check()?;
    ctx.report(1.0, "reference system preparation");
    Ok(ReferenceBatch {
        candidates: out,
        run_info,
        fallback_count: candidates.len(),
    })
}
