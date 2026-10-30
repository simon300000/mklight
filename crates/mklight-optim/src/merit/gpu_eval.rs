//! [`GpuTraceEvaluator`]: ray-based operands of many candidate systems
//! traced on a GPU compute backend ([`mklight_gpu::ComputeBackend`]) in one
//! dispatch per batch.
//!
//! # Per batch
//!
//! 1. **Host preparation** (parallel over candidates, f64): each candidate is
//!    frozen, resolved and wrapped in a `SequentialTracer` exactly as in
//!    [`TraceEvaluator`], its operands are planned (chief rays, pupil
//!    samplings with the half-pupil mirroring rule, single rays, real rays
//!    to a surface), and every planned ray is generated with the reference
//!    field/pupil mapping and ray aiming. The candidate is packed into the
//!    shared f32 layout (`PackedSystem`, all wavelengths, asphere
//!    normalisation radii from the qualified base system).
//! 2. **Trace**: all packed candidates are uploaded as one system set and
//!    their rays (selected per ray by system index) are traced in one
//!    request per (clipping, last surface): the unclipped chief rays, the
//!    clipped pupil rays to the image, and real rays to other surfaces.
//!    Rays enter in double-float (`RayBatch::push_f64`), radiometry off.
//! 3. **Read**: the operand values come from the same code as the CPU
//!    evaluator (`Candidate::value`), reading the GPU rays. OPD is formed
//!    on the host from the GPU's double-float optical paths against the
//!    reference sphere of the GPU's own chief ray (CONVENTIONS §12).
//!
//! Every value is a pure function of the candidate and the operand: the
//! trace kernel's result for a ray does not depend on the batch around it
//! (chunking changes no bit, `docs/GPU.md` §6), and host work is assembled
//! in a fixed order. Results are therefore deterministic and independent of
//! the batch composition (tested).
//!
//! # Precision and fallback
//!
//! The evaluator is built for a system with a precision qualification
//! (`mklight_gpu::xval::qualify`, a third of the CONVENTIONS §7 budget,
//! radiometry off). If the backend fails it, every operand is computed by the
//! CPU f64 [`TraceEvaluator`] and [`GpuTraceEvaluator::record`] says so in
//! the `RunInfo`. Candidates the packer cannot represent (more than 16
//! wavelengths, asphere terms beyond A22, values outside the normal f32
//! range) and batches whose dispatch fails are evaluated on the CPU as well,
//! and counted in [`GpuEvalStats`].
//!
//! # Operands
//!
//! Accelerated ([`gpu_supports`]): `ray_aberration`, `rms_spot`,
//! `spot_centroid`, `real_ray` (every quantity but the incidence angle),
//! `distortion`, `lateral_colour`, and — when [`GpuEvalSettings::wavefront`]
//! is set — `wavefront_aberration` and `wavefront_rms`. Everything else
//! (paraxial, model, relative illumination, …) goes to the CPU evaluator:
//! [`GpuTraceEvaluator::composite`] combines both.
//!
//! Reported points are f32 (`docs/GPU.md` §8), so values carry a rounding
//! noise of about half an f32 ulp of the image coordinates. Finite
//! differences therefore need larger steps than on the f64 reference
//! ([`gpu_fd_relative_step`], `docs/OPTIMIZATION.md`).

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::job::{Precision, Progress, RunInfo};
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem, Vec3};
use mklight_gpu::layout::MAX_WAVELENGTHS;
use mklight_gpu::xval::{QUALIFICATION_MARGIN, Qualification, XvalSettings, qualify};
use mklight_gpu::{
    ComputeBackend, ErrorBudget, Job, JobKind, JobOutput, JobResult, PackOptions, PackedSystem,
    Pending, Radiometry, RayBatch, RecordMode, SystemHandle, TraceRequest, TraceResult as GpuTrace,
};
use mklight_trace::{
    Compensated, OpdReference, Ray, SemiDiameterSettings, SequentialTracer, TraceResult,
    TraceStatus,
};

use crate::deriv::FdScheme;

use super::operand::{OperandKind, RayQuantity};
use super::trace_eval::{
    Candidate, ChiefData, FieldKey, GroupKey, GroupMemo, NodeRay, RealHit, RealKey, SingleKey,
    TraceEvaluator, trace_supports,
};
use super::{CompositeEvaluator, Operand, OperandError, OperandEvaluator, OperandValue};

/// Settings of a [`GpuTraceEvaluator`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuEvalSettings {
    /// Trace wavefront operands on the GPU. OPD comes from the kernel's
    /// double-float optical path, which the qualification checks against
    /// the OPD budget; off sends them to the CPU evaluator.
    pub wavefront: bool,
    /// Candidates per upload and dispatch. Larger batches are split into
    /// sub-batches of this size and pipelined: the host prepares the next
    /// sub-batch and reads back the previous one while the GPU traces.
    pub batch: usize,
}

impl Default for GpuEvalSettings {
    fn default() -> Self {
        GpuEvalSettings {
            wavefront: true,
            batch: 512,
        }
    }
}

/// Work counters of a [`GpuTraceEvaluator`] (since construction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuEvalStats {
    /// Candidates traced on the GPU.
    pub gpu_candidates: u64,
    /// Candidates evaluated on the CPU reference instead (unqualified
    /// backend, unpackable system, failed dispatch).
    pub cpu_candidates: u64,
    /// Rays traced on the GPU.
    pub rays: u64,
    /// Trace submissions.
    pub dispatches: u64,
    /// Device time of the traces (s).
    pub device_seconds: f64,
}

#[derive(Default)]
struct Counters {
    gpu: AtomicU64,
    cpu: AtomicU64,
    rays: AtomicU64,
    dispatches: AtomicU64,
    device_ns: AtomicU64,
}

struct GpuState {
    qualification: Qualification,
    /// Per-surface radii for the asphere normalisation of the packer (the
    /// qualified system's automatic semi-diameters).
    radius_hints: Option<Vec<f64>>,
}

/// Can a [`GpuTraceEvaluator`] trace operands of `kind` (with wavefront
/// operands enabled or not)?
pub fn gpu_supports(kind: &OperandKind, wavefront: bool) -> bool {
    use OperandKind::*;
    match kind {
        RayAberration { .. }
        | RmsSpot { .. }
        | SpotCentroid { .. }
        | Distortion { .. }
        | LateralColour { .. } => true,
        RealRay { quantity, .. } => *quantity != RayQuantity::IncidenceAngle,
        WavefrontAberration { .. } | WavefrontRms { .. } => wavefront,
        _ => false,
    }
}

/// Relative finite-difference step `η` for an operand evaluated in f32 on a
/// GPU. Reported points are f32, so ray positions carry an absolute noise of
/// about half an f32 ulp of the coordinate (5e-7 mm at 8 mm image height)
/// and the `√ε` / `∛ε` defaults difference noise (Jacobian column errors
/// of 6 and 7 on the Cooke triplet). Measured column errors against the f64
/// reference (`tests/gpu_eval.rs`, `docs/OPTIMIZATION.md`):
///
/// | operands | forward | central |
/// |---|---|---|
/// | ray positions (spot, real rays, distortion) | `η = 1e-3`: 4e-4 | `3e-3`: 6e-5 |
/// | wavefront (OPD from the double-float path) | `1e-4`: 9e-5 | `1e-3`: 5e-6 |
pub fn gpu_fd_relative_step(kind: &OperandKind, scheme: FdScheme) -> f64 {
    let wavefront = matches!(
        kind,
        OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
    );
    match (scheme, wavefront) {
        (FdScheme::Forward, false) => 1e-3,
        (FdScheme::Central, false) => 3e-3,
        (FdScheme::Forward, true) => 1e-4,
        (FdScheme::Central, true) => 1e-3,
    }
}

/// Operand evaluator tracing ray-based operands on a GPU backend (module
/// docs). Cheap to clone (shared state).
#[derive(Clone)]
pub struct GpuTraceEvaluator {
    backend: Arc<dyn ComputeBackend>,
    cpu: TraceEvaluator,
    settings: GpuEvalSettings,
    state: Arc<RwLock<GpuState>>,
    counters: Arc<Counters>,
}

impl std::fmt::Debug for GpuTraceEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTraceEvaluator")
            .field("backend", &self.backend_id())
            .field("active", &self.is_active())
            .field("settings", &self.settings)
            .finish()
    }
}

/// A context sharing `ctx`'s cancellation but not its progress bar (inner
/// GPU jobs report their own progress, which must not move the caller's).
fn detached(ctx: &JobContext) -> JobContext {
    JobContext::with(ctx.cancel.clone(), Progress::default())
}

impl GpuTraceEvaluator {
    /// An evaluator on `backend` qualified for `system` (normally the start
    /// of the optimisation). `cpu` computes everything the GPU does not and
    /// is the fallback. Fails only when `ctx` is cancelled; a failed
    /// qualification gives an inactive evaluator (CPU only) that says why.
    pub fn new(
        backend: Arc<dyn ComputeBackend>,
        cpu: TraceEvaluator,
        system: &OpticalSystem,
        settings: GpuEvalSettings,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        let state = qualify_system(backend.as_ref(), &cpu, system, ctx)?;
        Ok(GpuTraceEvaluator {
            backend,
            cpu,
            settings,
            state: Arc::new(RwLock::new(state)),
            counters: Arc::default(),
        })
    }

    /// Qualify again, for `system` (e.g. the design an optimisation ended
    /// with). The evaluator (and its clones) use the GPU only while the
    /// latest qualification passed.
    pub fn requalify(&self, system: &OpticalSystem, ctx: &JobContext) -> MkResult<Qualification> {
        let st = qualify_system(self.backend.as_ref(), &self.cpu, system, ctx)?;
        let q = st.qualification.clone();
        if let Ok(mut g) = self.state.write() {
            *g = st;
        }
        Ok(q)
    }

    pub fn backend(&self) -> &Arc<dyn ComputeBackend> {
        &self.backend
    }
    /// `DeviceInfo::backend_id` of the GPU backend.
    pub fn backend_id(&self) -> String {
        self.backend.info().backend_id()
    }
    /// The CPU evaluator (fallback and non-ray operands).
    pub fn cpu(&self) -> &TraceEvaluator {
        &self.cpu
    }
    pub fn settings(&self) -> &GpuEvalSettings {
        &self.settings
    }
    /// The latest qualification.
    pub fn qualification(&self) -> Qualification {
        self.state
            .read()
            .map(|g| g.qualification.clone())
            .unwrap_or_else(|e| e.into_inner().qualification.clone())
    }
    /// Does the evaluator trace on the GPU (the latest qualification
    /// passed)?
    pub fn is_active(&self) -> bool {
        self.state
            .read()
            .map(|g| g.qualification.passed)
            .unwrap_or(false)
    }
    /// The precision of the values it returns.
    pub fn precision(&self) -> Precision {
        if self.is_active() {
            Precision::F32Gpu
        } else {
            Precision::F64Reference
        }
    }

    pub fn stats(&self) -> GpuEvalStats {
        let c = &self.counters;
        GpuEvalStats {
            gpu_candidates: c.gpu.load(Ordering::Relaxed),
            cpu_candidates: c.cpu.load(Ordering::Relaxed),
            rays: c.rays.load(Ordering::Relaxed),
            dispatches: c.dispatches.load(Ordering::Relaxed),
            device_seconds: c.device_ns.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    /// Record in a `RunInfo` which backend computed the ray operands: the
    /// qualification decision (`Qualification::record`: backend, precision,
    /// fallback warning, `settings["backend_qualification"]`) and the work
    /// counters under `settings["gpu_evaluation"]`.
    pub fn record(&self, info: &mut RunInfo) {
        self.qualification().record(info);
        let stats = serde_json::to_value(self.stats()).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Object(m) = &mut info.settings {
            m.insert("gpu_evaluation".into(), stats);
        }
    }

    /// GPU operands first, then the CPU reference for every other trace,
    /// paraxial and model operand.
    pub fn composite(self) -> CompositeEvaluator {
        let cpu = self.cpu.clone();
        CompositeEvaluator::new().with(self).with(cpu)
    }

    /// Relative finite-difference step for an operand this evaluator traces
    /// on the GPU ([`gpu_fd_relative_step`]); `None` when the evaluator is
    /// not active (the CPU reference's own rule applies).
    pub fn gpu_fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        self.is_active().then(|| gpu_fd_relative_step(kind, scheme))
    }

    /// Host stages of one sub-batch, then its upload and trace submissions
    /// (not awaited). `hosts` and `packed` are the candidates' tracers and
    /// packed systems.
    fn launch<'a>(
        &'a self,
        systems: &'a [OpticalSystem],
        hosts: &'a [Host],
        packed: Vec<Option<PackedSystem>>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<InFlight<'a>> {
        let clip = self.cpu.settings().clip;
        // Plans, cache skeletons and generated rays.
        let mut prepared: Vec<Option<Prepared<'a>>> = hosts
            .par_iter()
            .map(|h| match h {
                Host::Gpu(tr) => Some(prepare(&self.cpu, tr, operands, clip)),
                _ => None,
            })
            .collect();
        ctx.check()?;
        // One request per (clipping, last surface): the candidates'
        // fragments concatenated in candidate order.
        let packed: Vec<PackedSystem> = packed.into_iter().flatten().collect();
        for (s, p) in prepared.iter_mut().flatten().enumerate() {
            p.system = s as u32;
        }
        let mut keys: BTreeMap<RequestKey, Vec<(usize, usize, u32)>> = BTreeMap::new();
        for (k, p) in prepared.iter().enumerate() {
            let Some(p) = p else { continue };
            for (i, fr) in p.frags.iter().enumerate() {
                keys.entry(fr.key).or_default().push((k, i, p.system));
            }
        }
        // Where each fragment's rays start in its request's output.
        let mut starts = vec![Vec::new(); prepared.len()];
        for (r, parts) in keys.values().enumerate() {
            let mut at = 0;
            for &(k, i, _) in parts {
                let p = prepared[k].as_ref().expect("planned");
                starts[k].push((i, r, at));
                at += p.frags[i].rays.len();
            }
        }
        let requests: Vec<((bool, u32), RayBatch)> = keys
            .iter()
            .map(|(&key, parts)| {
                let frags: Vec<(&RayBatch, u32)> = parts
                    .iter()
                    .map(|&(k, i, s)| (&prepared[k].as_ref().expect("planned").frags[i].rays, s))
                    .collect();
                (key, concat(&frags))
            })
            .collect();
        let mut fl = InFlight {
            systems,
            hosts,
            prepared,
            starts,
            handle: None,
            pending: Vec::new(),
            error: None,
        };
        if requests.is_empty() {
            return Ok(fl);
        }
        let handle = match self.backend.upload_systems(&packed) {
            Ok(h) => h,
            Err(e) => {
                fl.error = Some(e);
                return Ok(fl);
            }
        };
        fl.handle = Some(handle);
        drop(packed);
        let inner = detached(ctx);
        for ((clip, end), rays) in requests {
            self.counters
                .rays
                .fetch_add(rays.len() as u64, Ordering::Relaxed);
            let mut r = TraceRequest::new(handle, rays, 0, end);
            r.apply_apertures = clip;
            r.radiometry = Radiometry::Off;
            r.record = RecordMode::Final;
            match self.backend.submit(Job::Trace(Box::new(r)), &inner) {
                Ok(p) => fl.pending.push(p),
                Err(e) => {
                    fl.error = Some(e);
                    break;
                }
            }
            self.counters.dispatches.fetch_add(1, Ordering::Relaxed);
        }
        Ok(fl)
    }

    /// Await a sub-batch's traces, fill its caches and read the operands.
    fn finish(
        &self,
        mut fl: InFlight<'_>,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let mut traced = Vec::with_capacity(fl.pending.len());
        let mut error = fl.error.take();
        for p in std::mem::take(&mut fl.pending) {
            if error.is_some() {
                p.cancel();
                continue;
            }
            match p.wait() {
                Ok(res) => {
                    self.counters
                        .device_ns
                        .fetch_add((res.stats.device_seconds * 1e9) as u64, Ordering::Relaxed);
                    match res.output {
                        JobOutput::Trace(t) => traced.push(*t),
                        other => {
                            error = Some(MkError::Backend(format!("trace job returned {other:?}")))
                        }
                    }
                }
                Err(e) => error = Some(e),
            }
        }
        if let Some(h) = fl.handle.take() {
            let _ = self.backend.release_systems(h);
        }
        ctx.check()?;
        let systems = fl.systems;
        match error {
            None => {}
            Some(MkError::Cancelled) => return Err(MkError::Cancelled),
            // A failed upload or dispatch: the sub-batch is computed by the
            // reference.
            Some(_) => {
                self.counters
                    .cpu
                    .fetch_add(systems.len() as u64, Ordering::Relaxed);
                return self.cpu.evaluate_batch(systems, operands, ctx);
            }
        }
        // Fill the caches.
        fl.prepared
            .par_iter_mut()
            .zip(&mut fl.starts)
            .for_each(|(p, st)| {
                if let Some(p) = p {
                    st.sort_unstable();
                    let outs: Vec<(&GpuTrace, usize)> =
                        st.iter().map(|&(_, r, at)| (&traced[r], at)).collect();
                    fill(p, &outs);
                }
            });
        ctx.check()?;
        // Values.
        let n_gpu = fl.prepared.iter().filter(|p| p.is_some()).count() as u64;
        self.counters.gpu.fetch_add(n_gpu, Ordering::Relaxed);
        self.counters
            .cpu
            .fetch_add(systems.len() as u64 - n_gpu, Ordering::Relaxed);
        let values = systems
            .par_iter()
            .zip(fl.hosts)
            .zip(&fl.prepared)
            .map(|((s, h), p)| match (h, p) {
                (_, Some(p)) => {
                    let mut memo = GroupMemo::default();
                    Ok(operands
                        .iter()
                        .map(|op| p.cand.value_with(op, &mut memo))
                        .collect())
                }
                (Host::Failed(e), _) => Ok(vec![Err(e.clone()); operands.len()]),
                _ => self.cpu.evaluate_one(s, operands, ctx, false),
            })
            .collect();
        // Many candidates' caches: free them in parallel.
        fl.prepared.into_par_iter().for_each(drop);
        values
    }
}

/// A sub-batch whose traces have been submitted.
struct InFlight<'a> {
    systems: &'a [OpticalSystem],
    hosts: &'a [Host],
    prepared: Vec<Option<Prepared<'a>>>,
    /// Per candidate: (fragment, request, first ray in the request output).
    starts: Vec<Vec<(usize, usize, usize)>>,
    handle: Option<SystemHandle>,
    pending: Vec<Pending<JobResult>>,
    /// Upload or submission failure (the sub-batch then runs on the CPU).
    error: Option<MkError>,
}

/// Concatenate ray fragments (each of one system, given by index) into one
/// batch, the planes copied in parallel.
fn concat(frags: &[(&RayBatch, u32)]) -> RayBatch {
    type Plane = fn(&RayBatch) -> &Vec<f32>;
    const PLANES: [Plane; 14] = [
        |b| &b.x,
        |b| &b.y,
        |b| &b.z,
        |b| &b.l,
        |b| &b.m,
        |b| &b.n,
        |b| &b.opl_hi,
        |b| &b.opl_lo,
        |b| &b.x_lo,
        |b| &b.y_lo,
        |b| &b.z_lo,
        |b| &b.l_lo,
        |b| &b.m_lo,
        |b| &b.n_lo,
    ];
    let n: usize = frags.iter().map(|(f, _)| f.len()).sum();
    let mut planes: Vec<Vec<f32>> = PLANES
        .par_iter()
        .map(|get| {
            let mut v = Vec::with_capacity(n);
            for (f, _) in frags {
                v.extend_from_slice(get(f));
            }
            v
        })
        .collect();
    let mut b = RayBatch::default();
    let mut wavelength = Vec::with_capacity(n);
    let mut system = Vec::with_capacity(n);
    for &(f, s) in frags {
        wavelength.extend_from_slice(&f.wavelength);
        system.extend(std::iter::repeat_n(s, f.len()));
    }
    b.wavelength = wavelength;
    b.system = system;
    for field in [
        &mut b.n_lo,
        &mut b.m_lo,
        &mut b.l_lo,
        &mut b.z_lo,
        &mut b.y_lo,
        &mut b.x_lo,
        &mut b.opl_lo,
        &mut b.opl_hi,
        &mut b.n,
        &mut b.m,
        &mut b.l,
        &mut b.z,
        &mut b.y,
        &mut b.x,
    ] {
        *field = planes.pop().expect("14 planes");
    }
    b
}

/// The qualification of `backend` for `system` (CPU tracer settings,
/// clipping as evaluated, radiometry off) and the packer's radius hints.
fn qualify_system(
    backend: &dyn ComputeBackend,
    cpu: &TraceEvaluator,
    system: &OpticalSystem,
    ctx: &JobContext,
) -> MkResult<GpuState> {
    let failed = |reason: String| GpuState {
        qualification: Qualification {
            backend: backend.info().backend_id(),
            kernel_build: backend.precision().kernel_build.clone(),
            passed: false,
            budget: ErrorBudget::CONVENTIONS,
            margin: QUALIFICATION_MARGIN,
            budget_fraction: None,
            report: None,
            reason: Some(reason),
        },
        radius_hints: None,
    };
    ctx.check()?;
    if !backend.supports(JobKind::Trace) {
        return Ok(failed("backend does not support the trace kernel".into()));
    }
    let tr = match cpu.tracer(system) {
        Ok(t) => t,
        Err(e) => return Ok(failed(format!("the system cannot be traced: {e:?}"))),
    };
    if tr.num_wavelengths() > MAX_WAVELENGTHS {
        return Ok(failed(format!(
            "{} wavelengths: the batched evaluator packs at most {MAX_WAVELENGTHS}",
            tr.num_wavelengths()
        )));
    }
    let xs = XvalSettings {
        clip: cpu.settings().clip,
        radiometry: Radiometry::Off,
        ..XvalSettings::qualification()
    };
    let qualification = qualify(backend, &tr, Some(&xs), &detached(ctx))?;
    let hints = tr.semi_diameters(&SemiDiameterSettings::default()).values;
    Ok(GpuState {
        qualification,
        radius_hints: Some(hints),
    })
}

/// One candidate's host-side state.
enum Host {
    /// Traced on the GPU.
    Gpu(Box<SequentialTracer>),
    /// The packer cannot represent it: evaluated on the CPU.
    Cpu,
    /// No tracer (invalid system, paraxial failure): every operand fails.
    Failed(OperandError),
}

/// The tracer of a candidate and, when the packer can represent it, its
/// packed system.
fn host(
    cpu: &TraceEvaluator,
    s: &OpticalSystem,
    hints: Option<&[f64]>,
) -> (Host, Option<PackedSystem>) {
    let tr = match cpu.tracer(s) {
        Ok(t) => t,
        Err(e) => return (Host::Failed(e), None),
    };
    if tr.num_wavelengths() > MAX_WAVELENGTHS {
        return (Host::Cpu, None);
    }
    let rs = tr.system();
    let opts = PackOptions {
        wavelengths: None,
        radius_hints: hints
            .filter(|h| h.len() == rs.num_surfaces())
            .map(<[f64]>::to_vec),
    };
    match PackedSystem::pack(rs, &opts).and_then(|p| p.validate().map(|_| p)) {
        Ok(p) => (Host::Gpu(Box::new(tr)), Some(p)),
        Err(_) => (Host::Cpu, None),
    }
}

/// Where a traced ray's result goes in its candidate's cache (indices into
/// [`Prepared`]'s key lists).
#[derive(Clone, Copy)]
enum Dest {
    Chief(u32),
    Node(u32, u32),
    Single(u32),
    Real(u32),
}

/// A trace request: (clipped by fixed apertures, last surface).
type RequestKey = (bool, u32);

/// A candidate's rays for one trace request: (clipped, last surface), the
/// rays in surface 0's frame (double-float) and their destinations.
struct Fragment {
    key: (bool, u32),
    rays: RayBatch,
    dests: Vec<Dest>,
}

/// A planned candidate: its cache skeleton, the cache keys its rays fill,
/// and its ray fragments.
struct Prepared<'a> {
    cand: Candidate<'a>,
    chiefs: Vec<(FieldKey, usize)>,
    groups: Vec<GroupKey>,
    singles: Vec<SingleKey>,
    reals: Vec<RealKey>,
    frags: Vec<Fragment>,
    /// Index of the candidate's packed system in the uploaded set.
    system: u32,
}

/// Append a generated ray to the fragment of request `key`.
fn push_ray(frags: &mut Vec<Fragment>, key: (bool, u32), ray: Ray, wavelength: usize, dest: Dest) {
    let i = match frags.iter().position(|f| f.key == key) {
        Some(i) => i,
        None => {
            frags.push(Fragment {
                key,
                rays: RayBatch::default(),
                dests: Vec::new(),
            });
            frags.len() - 1
        }
    };
    let f = &mut frags[i];
    let (p, d) = (ray.pos, ray.dir);
    f.rays
        .push_f64([p.x, p.y, p.z], [d.x, d.y, d.z], wavelength as u32);
    f.dests.push(dest);
}

fn aim_failure(what: &str) -> OperandError {
    OperandError::RayFailure(format!("{what}: ray aiming failed"))
}

/// Plan a candidate, lay out its cache with failure placeholders and
/// generate every planned ray (f64, the reference's field/pupil mapping and
/// aiming) into its fragments. Rays that cannot be generated keep their
/// placeholders. `clip`: whether the evaluator clips at fixed apertures.
fn prepare<'a>(
    ev: &'a TraceEvaluator,
    tr: &'a SequentialTracer,
    operands: &[&Operand],
    clip: bool,
) -> Prepared<'a> {
    let (mut c, plan) = Candidate::planned(ev, tr, operands);
    c.prepare_points(&plan, false);
    let image = tr.system().image_index() as u32;
    let gen_ = |c: &Candidate<'_>, f: FieldKey, w: usize, px: f64, py: f64| {
        c.field(f)
            .map_err(|e| e.clone())
            .and_then(|field| {
                tr.generate(field, w, px, py)
                    .map_err(|_| aim_failure("pupil ray"))
            })
            .map(|g| g.ray)
    };
    let mut frags = Vec::new();
    for (k, &(f, w)) in plan.chiefs.iter().enumerate() {
        let d = match gen_(&c, f, w, 0.0, 0.0) {
            Ok(ray) => {
                push_ray(&mut frags, (false, image), ray, w, Dest::Chief(k as u32));
                // Replaced once traced.
                let e = OperandError::RayFailure("chief ray not traced".into());
                ChiefData {
                    chief: Err(e.clone()),
                    reference: Err(e),
                }
            }
            Err(e) => {
                let e = match e {
                    OperandError::RayFailure(_) => aim_failure("chief ray"),
                    other => other,
                };
                ChiefData {
                    chief: Err(e.clone()),
                    reference: Err(e),
                }
            }
        };
        c.chiefs.insert((f, w), d);
    }
    for (g, (&key, sampling)) in plan.groups.iter().enumerate() {
        let (f, w, _) = key;
        let (nodes, symmetry) = c.group_nodes(f, sampling);
        for (i, n) in nodes.iter().enumerate() {
            if let Ok(ray) = gen_(&c, f, w, n.px, n.py) {
                push_ray(
                    &mut frags,
                    (clip, image),
                    ray,
                    w,
                    Dest::Node(g as u32, i as u32),
                );
            }
        }
        let failed = vec![NodeRay::FAILED; nodes.len()];
        c.groups
            .insert(key, Candidate::group_rays(nodes, symmetry, failed));
    }
    for (k, &key) in plan.singles.iter().enumerate() {
        let (f, w, px, py) = key;
        if let Ok(ray) = gen_(&c, f, w, f64::from_bits(px), f64::from_bits(py)) {
            push_ray(&mut frags, (clip, image), ray, w, Dest::Single(k as u32));
        }
        c.singles.insert(key, NodeRay::FAILED);
    }
    for (k, &key) in plan.real.iter().enumerate() {
        let (f, w, px, py, j) = key;
        // Surface 0 is where rays start: left to the CPU path.
        if j == 0 {
            continue;
        }
        match gen_(&c, f, w, f64::from_bits(px), f64::from_bits(py)) {
            Ok(ray) => {
                push_ray(&mut frags, (clip, j as u32), ray, w, Dest::Real(k as u32));
                c.real_rays.insert(
                    key,
                    Err(OperandError::RayFailure("real ray not traced".into())),
                );
            }
            Err(e) => {
                c.real_rays.insert(key, Err(e));
            }
        }
    }
    Prepared {
        cand: c,
        chiefs: plan.chiefs.iter().copied().collect(),
        groups: plan.groups.keys().copied().collect(),
        singles: plan.singles.iter().copied().collect(),
        reals: plan.real.iter().copied().collect(),
        frags,
        system: 0,
    }
}

/// GPU ray `i` of `r` as a reference-style result (an error unless it
/// reached its last surface).
fn gpu_ray(r: &GpuTrace, i: usize) -> Result<TraceResult, OperandError> {
    let st = r.ray_status(i);
    if !st.is_ok() {
        return Err(OperandError::RayFailure(format!("{st:?}")));
    }
    Ok(TraceResult {
        status: TraceStatus::Ok,
        ray: Ray::new(
            Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64),
            Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64),
        ),
        surface: r.last_surface[i] as usize,
        opl: Compensated {
            hi: r.opl_hi[i] as f64,
            lo: r.opl_lo[i] as f64,
        },
        transmittance: r.transmittance[i] as f64,
    })
}

/// The OPD reference sphere of a GPU chief ray (as
/// `SequentialTracer::opd_reference`, from the GPU's point, direction and
/// optical path).
fn gpu_reference(
    tr: &SequentialTracer,
    chief: &TraceResult,
    wl: usize,
) -> Result<OpdReference, OperandError> {
    let sys = tr.system();
    let img = sys.image_index();
    let z_xp = tr.paraxial().at(wl).exit_pupil_position;
    let s_e = if z_xp.is_finite() {
        if chief.ray.dir.z == 0.0 {
            return Err(OperandError::RayFailure(
                "chief ray: degenerate reference sphere".into(),
            ));
        }
        (z_xp - chief.ray.pos.z) / chief.ray.dir.z
    } else {
        f64::INFINITY
    };
    Ok(OpdReference {
        chief_opl: chief.opl,
        center: chief.ray.pos,
        chief_dir: chief.ray.dir,
        s_exit_pupil: s_e,
        n_image: sys.n_before(img, wl),
        wavelength_um: sys.wavelengths_um[wl],
    })
}

/// Write the traced rays into a candidate's cache: `outs[i]` is the request
/// output holding fragment `i`'s rays and their start in it. Chief rays
/// (and reference spheres) first, then the rays whose OPD refers to them.
fn fill(p: &mut Prepared<'_>, outs: &[(&GpuTrace, usize)]) {
    let tr = p.cand.tr;
    for (fr, &(out, at)) in p.frags.iter().zip(outs) {
        for (i, d) in fr.dests.iter().enumerate() {
            if let Dest::Chief(k) = *d {
                let (f, w) = p.chiefs[k as usize];
                let chief = gpu_ray(out, at + i).map_err(|e| match e {
                    OperandError::RayFailure(m) => {
                        OperandError::RayFailure(format!("chief ray: {m}"))
                    }
                    other => other,
                });
                let reference = chief
                    .as_ref()
                    .map_err(|e| e.clone())
                    .and_then(|ch| gpu_reference(tr, ch, w));
                p.cand.chiefs.insert((f, w), ChiefData { chief, reference });
            }
        }
    }
    let c = &mut p.cand;
    let node = |c: &Candidate<'_>, f: FieldKey, w: usize, out: &GpuTrace, i: usize| match gpu_ray(
        out, i,
    ) {
        Ok(r) => NodeRay {
            pos: Some([r.ray.pos.x, r.ray.pos.y]),
            opd: c
                .chiefs
                .get(&(f, w))
                .and_then(|d| d.reference.as_ref().ok())
                .and_then(|rf| rf.opd_waves(&r)),
        },
        Err(_) => NodeRay::FAILED,
    };
    for (fr, &(out, at)) in p.frags.iter().zip(outs) {
        for (i, d) in fr.dests.iter().enumerate() {
            match *d {
                Dest::Chief(_) => {}
                Dest::Node(g, k) => {
                    let key = p.groups[g as usize];
                    let v = node(c, key.0, key.1, out, at + i);
                    if let Some(gr) = c.groups.get_mut(&key) {
                        gr.rays[k as usize] = v;
                    }
                }
                Dest::Single(k) => {
                    let key = p.singles[k as usize];
                    let v = node(c, key.0, key.1, out, at + i);
                    c.singles.insert(key, v);
                }
                Dest::Real(k) => {
                    let key = p.reals[k as usize];
                    let hit = gpu_ray(out, at + i).map(|r| RealHit {
                        point: r.ray.pos,
                        dir: r.ray.dir,
                        opl: r.opl.hi + r.opl.lo,
                    });
                    c.real_rays.insert(key, hit);
                }
            }
        }
    }
}

impl OperandEvaluator for GpuTraceEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        gpu_supports(kind, self.settings.wavefront) && trace_supports(kind)
    }

    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        if !self.is_active() {
            self.counters
                .cpu
                .fetch_add(systems.len() as u64, Ordering::Relaxed);
            return self.cpu.evaluate_batch(systems, operands, ctx);
        }
        let hints = self.state.read().ok().and_then(|g| g.radius_hints.clone());
        let parts: Vec<&[OpticalSystem]> = systems.chunks(self.settings.batch.max(1)).collect();
        // The tracers of every sub-batch live until the end: in-flight
        // caches borrow them.
        let arena: Vec<OnceLock<Vec<Host>>> = parts.iter().map(|_| OnceLock::new()).collect();
        let mut queue: VecDeque<InFlight<'_>> = VecDeque::new();
        let mut out = Vec::with_capacity(systems.len());
        for (&part, slot) in parts.iter().zip(&arena) {
            ctx.check()?;
            // Host stages of this sub-batch run while the GPU traces the
            // previous one.
            let (hosts, packed): (Vec<Host>, Vec<Option<PackedSystem>>) = part
                .par_iter()
                .map(|s| host(&self.cpu, s, hints.as_deref()))
                .unzip();
            let hosts = slot.get_or_init(|| hosts);
            queue.push_back(self.launch(part, hosts, packed, operands, ctx)?);
            if queue.len() > 1
                && let Some(fl) = queue.pop_front()
            {
                out.extend(self.finish(fl, operands, ctx)?);
            }
        }
        while let Some(fl) = queue.pop_front() {
            out.extend(self.finish(fl, operands, ctx)?);
        }
        Ok(out)
    }

    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        if self.is_active() {
            self.gpu_fd_relative_step(kind, scheme)
        } else {
            self.cpu.fd_relative_step(kind, scheme)
        }
    }
}
