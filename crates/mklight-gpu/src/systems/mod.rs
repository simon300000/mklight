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
    /// Program: instructions, then patches, then guards.
    pub dependencies: PlanView,
    /// Packed header/surface/coefficient template words.
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
    /// Uploaded bounds, program and template behind the three input views.
    imports: [BufferHandle; 3],
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
            self.bounds()
                .into_iter()
                .flat_map(|(a, b)| [Df32::from_f64(a), Df32::from_f64(b)])
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
    /// Inclusive bounds of every candidate column; material columns are choice indices.
    fn bounds(&self) -> Vec<(f64, f64)> {
        let variables = self.variables.iter().map(|p| (p.lower, p.upper));
        let materials = (self.material_slots.iter()).map(|s| (0.0, (s.choices.len() - 1) as f64));
        variables.chain(materials).collect()
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
        let count = candidates.len();
        let nparam = self.compiled.parameter_count();
        let mut parameters = Vec::with_capacity(count * nparam);
        let mut ids = Vec::with_capacity(count * 4);
        let mut hash = blake3::Hasher::new();
        hash.update(self.compiled.snapshot.hash().to_hex().as_bytes());
        let bounds = self.compiled.bounds();
        for candidate in candidates {
            ctx.check()?;
            if candidate.values.len() != nparam {
                return Err(MkError::InvalidParam("candidate parameter width".into()));
            }
            let mut input_status = READY;
            for (&v, &(lower, upper)) in candidate.values.iter().zip(&bounds) {
                let d = Df32::from_f64(v);
                if !(lower..=upper).contains(&v) {
                    input_status = OUTSIDE_BOUNDS;
                } else if input_status == READY
                    && (!d.is_finite() || d.hi.is_subnormal() || (d.hi == 0.0 && v != 0.0))
                {
                    // Not representable as a normal double-float.
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
            &hash.finalize().to_hex(),
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
        let count = parameters.desc.shape.first().copied().unwrap_or(0);
        let p = &self.compiled.program;
        let imports = [&self.bounds, &self.code, &self.template].map(|h| h.clone());
        let [bounds, dependencies, material_tables] = imports.clone().map(|h| {
            let id = plan.import(h.view());
            plan.view(id).unwrap()
        });
        let [
            headers,
            surfaces,
            coefficients,
            first_order,
            ids,
            configurations,
            status,
            model_data,
        ] = output_descs(p, count).map(|d| plan.scratch(d));
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
            map: CandidateMap {
                candidate_ids: ids,
                configuration_ids: configurations,
                status,
                count,
            },
            program: p.clone(),
            imports,
            parameter_count: self.compiled.parameter_count(),
            tile_candidates: 1024,
        };
        let entry = serde_json::json!({"version":VERSION,"program":self.compiled.fingerprint,"configuration":self.compiled.configuration,"candidate_vectors_hash":candidate_identity,"count":count,"qualification":"required_per_candidate_or_tested_envelope","dirty_flags":DIRTY_ALL});
        let batches = &mut plan.settings["system_batches"];
        if batches.is_null() {
            *batches = serde_json::json!([]);
        }
        batches.as_array_mut().unwrap().push(entry);
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
    if r.tile_candidates == 0 || r.tile_candidates > 1024 {
        return Err(bad("system preparation tile must be 1..=1024"));
    }
    if r.parameters.desc
        != BufferDesc::new(ElementType::DoubleFloat, [r.map.count, r.parameter_count])
        || r.slots.desc != BufferDesc::new(ElementType::U32, [r.map.count, 4])
        || r.parameters.stride != 1
        || r.slots.stride != 1
    {
        return Err(bad("system preparation candidate shape"));
    }
    // The kernel trusts the program's indices: only the compiler's own
    // uploaded, unmodified buffers are accepted.
    let views = [&r.bounds, &r.dependencies, &r.material_tables];
    for (view, handle) in views.into_iter().zip(&r.imports) {
        let full =
            |offset, stride, desc: &BufferDesc| offset == 0 && stride == 1 && desc == handle.desc();
        let ok = matches!(plan.buffers.get(view.buffer.0), Some(PlanBuffer::Import(v))
            if full(v.offset, v.stride, &v.desc)
                && v.buffer.owner() == handle.owner()
                && v.buffer.generation() == handle.generation());
        if !ok || !full(view.offset, view.stride, &view.desc) {
            return Err(bad("system preparation input identity changed"));
        }
    }
    for (id, desc) in r
        .outputs()
        .into_iter()
        .zip(output_descs(&r.program, r.map.count))
    {
        if plan.desc(id)? != &desc || desc.len()? > u32::MAX as usize {
            return Err(bad("system preparation output extent"));
        }
    }
    Ok(())
}
fn output_descs(p: &Program, n: usize) -> [BufferDesc; 8] {
    use ElementType::{DoubleFloat, U32};
    let first_order = [n, p.wavelength_count, 2, 8 + 7 * p.surface_count];
    [
        BufferDesc::new(U32, [n, p.headers.len()]),
        BufferDesc::new(U32, [n, p.surfaces.len()]),
        BufferDesc::new(DoubleFloat, [n, p.coefficients.len() / 2]),
        BufferDesc::new(DoubleFloat, first_order),
        BufferDesc::new(U32, [n, 2]),
        BufferDesc::new(U32, [n]),
        BufferDesc::new(U32, [n, 2]),
        BufferDesc::new(DoubleFloat, [n, p.model_len]),
    ]
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
    let df = |w: &[u32]| -> Vec<Df32> {
        (w.as_chunks().0.iter())
            .map(|&[hi, lo]| Df32::new(f32::from_bits(hi), f32::from_bits(lo)))
            .collect()
    };
    let width = r.parameter_count;
    let bounds = df(x.words(r.bounds.buffer));
    let mut values = vec![Df32::ZERO; p.code.len()];
    for row in 0..r.map.count {
        ctl.check()?;
        let start = (r.parameters.offset + row * width) * 2;
        let parameters = df(&x.words(r.parameters.buffer)[start..start + width * 2]);
        let le = |a: Df32, b: Df32| a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
        let slot = r.slots.offset + row * 4;
        let mut status = x.words(r.slots.buffer)[slot + 3];
        for (i, &v) in parameters.iter().enumerate() {
            if !v.is_finite() || !le(bounds[2 * i], v) || !le(v, bounds[2 * i + 1]) {
                status = OUTSIDE_BOUNDS;
            }
        }
        algebra::evaluate(&p.code, &parameters, &mut values);
        let invalid = |i: u32| !values[i as usize].is_finite();
        if status == READY
            && (p
                .guards
                .iter()
                .any(|&i| invalid(i) || le(values[i as usize], Df32::ZERO))
                || p.patches
                    .iter()
                    .any(|&[dest, _, _, i]| dest < 3 && invalid(i)))
        {
            status = INVALID_NUMERIC;
        }
        let outputs = [
            r.headers,
            r.surfaces,
            r.coefficients,
            r.first_order,
            r.model_data,
        ];
        let strides = [
            p.headers.len(),
            p.surfaces.len(),
            p.coefficients.len(),
            p.first_order_len * 2,
            p.model_len * 2,
        ];
        for (k, template) in [&p.headers, &p.surfaces, &p.coefficients]
            .into_iter()
            .enumerate()
        {
            x.words_mut(outputs[k])[row * strides[k]..(row + 1) * strides[k]]
                .copy_from_slice(template);
        }
        for &[dest, hi, lo, node] in &p.patches {
            let v = values[node as usize];
            let out = &mut x.words_mut(outputs[dest as usize])[row * strides[dest as usize]..];
            out[hi as usize] = v.hi.to_bits();
            if lo != u32::MAX {
                out[lo as usize] = v.lo.to_bits();
            }
        }
        let header = size_of::<crate::PackedHeader>() / 4;
        let surface = size_of::<crate::PackedSurface>() / 4;
        let surface_offset = std::mem::offset_of!(crate::PackedHeader, surface_offset) / 4;
        let coefficient_offset = std::mem::offset_of!(crate::PackedSurface, coefficient_offset) / 4;
        for chunk in 0..p.chunk_count {
            x.words_mut(r.headers)[row * p.headers.len() + chunk * header + surface_offset] +=
                (row * p.chunk_count * p.surface_count) as u32;
        }
        for j in 0..p.surface_count * p.chunk_count {
            x.words_mut(r.surfaces)[row * p.surfaces.len() + j * surface + coefficient_offset] +=
                (row * p.coefficients.len() / 2) as u32;
        }
        let ids = x.words(r.slots.buffer)[slot..slot + 3].to_vec();
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
#[allow(clippy::too_many_arguments)]
pub fn reference_batch(
    snapshot: &Snapshot,
    library: Arc<MaterialLibrary>,
    variables: &[Parameter],
    materials: &[MaterialSlot],
    candidates: &[Candidate],
    configuration: Option<usize>,
    reason: &str,
    ctx: &JobContext,
) -> MkResult<ReferenceBatch> {
    let timer = mklight_core::job::RunTimer::start();
    let mut out = Vec::new();
    let n = candidates.len() as f64;
    for (i, c) in candidates.iter().enumerate() {
        let sub = ctx.subrange(i as f64 / n, (i + 1) as f64 / n);
        let result = reference_system(
            snapshot,
            &library,
            variables,
            materials,
            c,
            configuration,
            &sub,
        )
        .and_then(|rs| PackedSystem::pack_chunks(&rs, None));
        if matches!(result, Err(MkError::Cancelled)) {
            return Err(MkError::Cancelled);
        }
        out.push((c.id, result));
    }
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
    ctx.report(1.0, "reference system preparation");
    Ok(ReferenceBatch {
        candidates: out,
        run_info,
        fallback_count: candidates.len(),
    })
}
