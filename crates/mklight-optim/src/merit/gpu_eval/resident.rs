//! Resident candidate traces of the hybrid host/device division. The legacy
//! evaluator above remains an independent before/after benchmark route.
//!
//! A batch goes through stages that callers may pipeline across
//! sub-batches: [`prepare`] (host: candidate plans, chief rays and seeds,
//! the resident plan), the device plan, [`collect`] (the outputs read in
//! place) and [`finish`] (host: caches, completion of the rays the device
//! left unresolved in f64, operand values). [`RayDesign`] selects where the
//! pupil rays are aimed and the device's aiming schedule.
mod fits;
use super::*;
use crate::gpu::GpuContext;
use crate::merit::Capability;
use crate::merit::trace_eval::Plan;
use mklight_gpu::{Df32, aiming as a, resident::*};
use std::collections::BTreeSet;
#[derive(Clone, Copy)]
enum Dest {
    Chief,
    Node(u32, u32),
    Single,
    /// Real ray ending at this surface.
    Real(u32),
    Illumination,
}
#[derive(Clone)]
struct Probe {
    candidate: usize,
    /// Row of this candidate's field/λ in the plan's aiming metadata.
    slot: u32,
    field: FieldKey,
    wavelength: usize,
    pupil: [f64; 2],
    dest: Dest,
}
pub(crate) struct Uploaded<'a>(&'a dyn ComputeBackend, SystemHandle);
impl Drop for Uploaded<'_> {
    fn drop(&mut self) {
        let _ = self.0.release_systems(self.1);
    }
}
fn records<T: bytemuck::Pod>(
    g: &GpuContext,
    p: &mut ResidentPlan,
    count: usize,
    mut value: impl FnMut(usize) -> MkResult<T>,
) -> MkResult<PlanView> {
    let buffer = g.backend.fill_buffer(
        BufferDesc::new(ElementType::U32, [count, size_of::<T>() / 4]),
        &mut |mut fill| {
            if count > 0 {
                for (i, out) in bytemuck::cast_slice_mut::<_, T>(fill.as_mut_slice::<u32>()?)
                    .iter_mut()
                    .enumerate()
                {
                    *out = value(i)?;
                }
            }
            Ok(())
        },
    )?;
    let id = p.import(buffer.view());
    p.view(id)
}
/// How the ray stage of a batch is divided between the device and the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RayDesign {
    /// Newton iterations a device ray may spend over its aiming schedule.
    pub schedule_cap: u32,
    /// Aiming continuation rounds the device commits without reading the
    /// pending count back (0: read back after each round group).
    pub speculative_rounds: u32,
    /// Aim every pupil ray on the host (the candidates' f64 tracers, the
    /// reference's aims) and only trace it on the device (the solvers'
    /// default; device aiming is the `aiming` key's `device` and `table`
    /// options).
    pub host_aim: bool,
}
impl RayDesign {
    /// Host candidates and per-ray host completion: the device aims each ray
    /// for at most [`HYBRID_SCHEDULE_CAP`] iterations and the host completes
    /// the rest, so the device rounds are bounded and committed
    /// speculatively (no read-back inside the plan).
    pub(crate) const HYBRID: Self = Self {
        schedule_cap: HYBRID_SCHEDULE_CAP,
        speculative_rounds: hybrid_rounds(HYBRID_SCHEDULE_CAP),
        host_aim: false,
    };
    /// [`Self::HYBRID`] with another device schedule cap.
    pub(crate) fn hybrid_capped(cap: u32) -> Self {
        Self {
            schedule_cap: cap.max(1),
            speculative_rounds: hybrid_rounds(cap.max(1)),
            ..Self::HYBRID
        }
    }
}
/// Device iterations per ray. Healthy rays converge in
/// 2–6 (chief-seeded); a ray still unconverged at this cap is aimed on the
/// host, whose schedule is the reference's.
pub(crate) const HYBRID_SCHEDULE_CAP: u32 = 8;
/// Continuation rounds that cover a capped schedule without read-back: the
/// cap's iterations plus up to six phase entries, less the first command's
/// four units, at four units per round (the Metal executor's budgets), plus
/// one spare round. A round with no pending ray dispatches no thread.
const fn hybrid_rounds(cap: u32) -> u32 {
    (cap + 2).div_ceil(4) + 1
}

/// `MKLIGHT_SOLVERS_TRACE_ROUTING=1` prints, per hybrid batch, the rays
/// routed to the host and the device's unresolved, failed and slow rays
/// (diagnostics).
pub(crate) fn trace_routing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MKLIGHT_SOLVERS_TRACE_ROUTING").is_ok_and(|v| v == "1"))
}

/// Identity of one traced ray, shared by the candidates of a batch:
/// (field, λ, bits of `px`, bits of `py`, clipped, last surface).
pub(crate) type RayKey = (FieldKey, usize, u64, u64, bool, usize);

#[allow(clippy::too_many_arguments)]
fn append(
    g: &GpuContext,
    p: &mut ResidentPlan,
    set: &SystemSet,
    metadata: &PlanView,
    seeds: &PlanView,
    probes: &[Probe],
    remap: Option<&[u32]>,
    last: usize,
    clip: bool,
    tolerance: f32,
    design: &RayDesign,
    ctx: &JobContext,
) -> MkResult<a::ResidentHits> {
    let samples = records(g, p, probes.len(), |i| {
        if i % 4096 == 0 {
            ctx.check()?;
        }
        let q = &probes[i];
        let slot = remap.map_or(q.slot, |r| r[q.slot as usize]);
        Ok(a::Sample::new(q.pupil, slot, i as u64))
    })?;
    let steps = g.fill(p, [last], |out: &mut [u32]| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (i + 1) as u32;
        }
    })?;
    let generated = a::rays(p, probes.len());
    let hits = a::hits(p, probes.len(), 0);
    p.operations
        .push(ResidentOp::Aiming(a::Request::GenerateAim(
            a::GenerateAim {
                systems: set.clone(),
                samples,
                field_wavelength: metadata.clone(),
                field_key: None,
                seeds: Some(seeds.clone()),
                tolerance,
                max_iterations: 60,
                output: generated.clone(),
                options: a::AimOptions {
                    schedule_cap: design.schedule_cap,
                    speculative_rounds: design.speculative_rounds,
                    ..Default::default()
                },
            },
        )));
    p.operations
        .push(ResidentOp::Aiming(a::Request::TraceResident(
            a::TraceResident {
                systems: set.clone(),
                input: generated,
                steps,
                start_surface: 0,
                apply_apertures: clip,
                radiometry: 0,
                newton: Default::default(),
                output: hits.clone(),
            },
        )));
    Ok(hits)
}
/// Chief rays generated (aimed) on the host by the candidates' f64 tracers
/// and traced on the device to the image (unclipped): the hybrid design's
/// chief rows, which no device aim can leave unresolved. A chief the host
/// cannot generate enters with a failure report and is not traced.
#[allow(clippy::too_many_arguments)]
fn host_chief_rays(
    g: &GpuContext,
    p: &mut ResidentPlan,
    set: &SystemSet,
    chiefs: &[Probe],
    candidates: &[(Candidate<'_>, Plan)],
    tracers: &[Arc<SequentialTracer>],
    last: usize,
    ctx: &JobContext,
) -> MkResult<a::ResidentHits> {
    let rays: Vec<Option<Ray>> = chiefs
        .par_iter()
        .map(|q| {
            let field = candidates[q.candidate].0.field(q.field).ok()?;
            let tr = &tracers[q.candidate];
            // The prepared field's converged chief aim is the pupil-centre
            // solve when the field has no vignetting decentre.
            let v = field.point.vignetting;
            if field.is_valid()
                && v.vdx == 0.
                && v.vdy == 0.
                && let Some(a) = field.aims.get(q.wavelength).filter(|a| a.converged)
            {
                return tr.start_ray(&field.source, a.chief);
            }
            tr.generate(field, q.wavelength, 0.0, 0.0)
                .ok()
                .map(|g| g.ray)
        })
        .collect();
    ctx.check()?;
    host_rays_traced(
        g,
        p,
        set,
        chiefs,
        &rays,
        last,
        false,
        mklight_gpu::consts::MKL_AIM_FAILED,
    )
}
/// Pupil probes per host aiming work item of [`host_aimed_rays`]: runs of
/// one (candidate, field, wavelength) are aimed together in trace packets.
const HOST_AIM_CHUNK: usize = 8;
/// Pupil probes aimed on the host by the candidates' f64 tracers
/// (`generate_many`, the reference's aims bit for bit) and traced on the
/// device ([`RayDesign::host_aim`]). A ray the host cannot aim fails as on
/// the reference: it enters with an aim-outside report, so that it is
/// neither traced nor completed again.
#[allow(clippy::too_many_arguments)]
fn host_aimed_rays(
    g: &GpuContext,
    p: &mut ResidentPlan,
    set: &SystemSet,
    probes: &[Probe],
    candidates: &[(Candidate<'_>, Plan)],
    tracers: &[Arc<SequentialTracer>],
    last: usize,
    clip: bool,
    ctx: &JobContext,
) -> MkResult<a::ResidentHits> {
    let started = std::time::Instant::now();
    let rays: Vec<Option<Ray>> = probes
        .par_chunks(HOST_AIM_CHUNK)
        .flat_map_iter(|chunk| {
            let mut out = Vec::with_capacity(chunk.len());
            for run in chunk.chunk_by(|a, b| {
                (a.candidate, a.field, a.wavelength) == (b.candidate, b.field, b.wavelength)
            }) {
                let q = &run[0];
                match candidates[q.candidate].0.field(q.field) {
                    Ok(field) => {
                        let requests: Vec<_> =
                            run.iter().map(|q| (field, q.wavelength, q.pupil)).collect();
                        out.extend(
                            tracers[q.candidate]
                                .generate_many(&requests)
                                .into_iter()
                                .map(|r| r.ok().map(|g| g.ray)),
                        );
                    }
                    Err(_) => out.extend(std::iter::repeat_n(None, run.len())),
                }
            }
            out
        })
        .collect();
    ctx.check()?;
    g.observe(PlanStats {
        stages: vec![StageStats {
            name: "host pupil-ray aiming".into(),
            wall_seconds: started.elapsed().as_secs_f64(),
            ..Default::default()
        }],
        ..Default::default()
    });
    host_rays_traced(
        g,
        p,
        set,
        probes,
        &rays,
        last,
        clip,
        mklight_gpu::consts::MKL_AIM_OUTSIDE,
    )
}
/// Host-generated starting rays of `probes` (`None`: not generated, entered
/// with the report `failed`) traced on the device to surface `last`.
#[allow(clippy::too_many_arguments)]
fn host_rays_traced(
    g: &GpuContext,
    p: &mut ResidentPlan,
    set: &SystemSet,
    probes: &[Probe],
    rays: &[Option<Ray>],
    last: usize,
    clip: bool,
    failed: u32,
) -> MkResult<a::ResidentHits> {
    let n = probes.len();
    let floats = g.fill(p, [15, n], |out: &mut [f32]| {
        for (i, r) in rays.iter().enumerate() {
            let Some(r) = r else { continue };
            let v = [r.pos.x, r.pos.y, r.pos.z, r.dir.x, r.dir.y, r.dir.z];
            for (k, &x) in v.iter().enumerate() {
                let d = Df32::from_f64(x);
                // Hi planes 0..6, then OPL hi/lo and weight, then lo planes.
                out[k * n + i] = d.hi;
                out[(9 + k) * n + i] = d.lo;
            }
            out[6 * n + i] = 0.;
            out[7 * n + i] = 0.;
            out[8 * n + i] = 1.;
        }
    })?;
    let indices = g.fill(p, [2, n], |out: &mut [u32]| {
        for (i, q) in probes.iter().enumerate() {
            out[i] = q.candidate as u32;
            out[n + i] = q.wavelength as u32;
        }
    })?;
    let ids = g.fill(p, [2, n], |out: &mut [u32]| {
        for (i, v) in out[..n].iter_mut().enumerate() {
            *v = i as u32;
        }
        out[n..].fill(0);
    })?;
    let report = g.fill(p, [4, n], |out: &mut [u32]| {
        out.fill(0);
        for (i, r) in rays.iter().enumerate() {
            if r.is_none() {
                out[i] = failed;
            }
        }
    })?;
    let steps = g.fill(p, [last], |out: &mut [u32]| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (i + 1) as u32;
        }
    })?;
    let hits = a::hits(p, n, 0);
    p.operations
        .push(ResidentOp::Aiming(a::Request::TraceResident(
            a::TraceResident {
                systems: set.clone(),
                input: a::ResidentRays {
                    floats: floats.buffer,
                    indices: indices.buffer,
                    count: n,
                    logical_ids: ids.buffer,
                    report: report.buffer,
                },
                steps,
                start_surface: 0,
                apply_apertures: clip,
                radiometry: 0,
                newton: Default::default(),
                output: hits.clone(),
            },
        )));
    Ok(hits)
}
fn trace_result(
    f: &[f32],
    status: &[u32],
    n: usize,
    i: usize,
) -> Result<TraceResult, OperandError> {
    if status[i] != 0 {
        return Err(OperandError::RayFailure(format!(
            "resident status {}",
            status[i]
        )));
    }
    let val = |k: usize, lo: usize| f[k * n + i] as f64 + f[lo * n + i] as f64;
    Ok(TraceResult {
        status: TraceStatus::Ok,
        ray: Ray::new(
            Vec3::new(val(0, 9), val(1, 10), val(2, 11)),
            Vec3::new(val(3, 12), val(4, 13), val(5, 14)),
        ),
        surface: status[n + i] as usize,
        opl: Compensated {
            hi: f[6 * n + i] as f64,
            lo: f[7 * n + i] as f64,
        },
        transmittance: 1.,
    })
}
fn ray_key(q: &Probe, clip: bool, last: usize) -> RayKey {
    (
        q.field,
        q.wavelength,
        q.pupil[0].to_bits(),
        q.pupil[1].to_bits(),
        clip,
        last,
    )
}
/// Whether an aiming report leaves the ray unresolved (excluded samples are
/// not).
fn unresolved_report(code: u32) -> bool {
    !matches!(code, 0 | mklight_gpu::consts::MKL_AIM_OUTSIDE)
}

/// Group nodes of one candidate that the host traces instead of the device
/// (rays routed to the host before the plan): group index, key, node
/// indices and, after [`host_stage`], their f64 traces.
struct HostGroup {
    gi: u32,
    key: GroupKey,
    nodes: Vec<usize>,
    traced: Option<Vec<Option<TraceResult>>>,
}

/// A batch's host preparation and its resident plan, ready to submit.
pub(crate) struct Prepared<'a> {
    pub(crate) plan: Option<ResidentPlan>,
    candidates: Vec<(Candidate<'a>, Plan)>,
    /// Per candidate, the group nodes routed to the host.
    host: Vec<Vec<HostGroup>>,
    /// Per candidate, the values of the operands that need no ray (model
    /// and paraxial), computed by [`host_stage`] while the device works.
    early: Option<Vec<Vec<OperandValue>>>,
    chiefs: Vec<Probe>,
    /// Traced probe partitions by (clipping, last surface), in plan output
    /// order.
    partitions: Vec<((bool, usize), Vec<Probe>)>,
    moment_segments: Vec<Vec<Option<(usize, u32)>>>,
    fits: Vec<fits::Fit>,
    uploaded: Uploaded<'a>,
    traced_rays: usize,
    aggregates: bool,
    /// Image surface index (the chief rays' last surface).
    image: usize,
}
impl Prepared<'_> {
    pub(crate) fn traced_rays(&self) -> usize {
        self.traced_rays
    }
    /// Rays of the batch routed to the host before the plan.
    pub(crate) fn host_rays(&self) -> usize {
        self.host.iter().flatten().map(|h| h.nodes.len()).sum()
    }
}

/// Is `op` evaluated without any ray (model and paraxial operands)?
fn rayless(op: &Operand) -> bool {
    matches!(
        op.kind.capability(),
        Capability::Model | Capability::Paraxial
    )
}

/// Rays per host work item of [`host_stage`]: a failing aim costs the host
/// ~0.1 ms, so a few per item balance a single candidate's rays over the
/// pool.
const HOST_RAYS_PER_ITEM: usize = 4;

/// The host's share of a batch, run while the device works on the plan:
/// the group nodes routed to the host (in parallel over candidates, groups
/// and runs of nodes), and every operand that needs no ray (model operands, with their automatic semi-diameters, and
/// paraxial operands). The OPD of the host rays is formed in [`finish`],
/// against the device chief's reference sphere as for the device rays.
pub(crate) fn host_stage(
    prepared: &mut Prepared<'_>,
    cpu: &TraceEvaluator,
    tracers: &[Arc<SequentialTracer>],
    operands: &[&Operand],
    ctx: &JobContext,
) -> MkResult<()> {
    let candidates = &prepared.candidates;
    let items: Vec<(usize, usize, usize, usize)> = prepared
        .host
        .iter()
        .enumerate()
        .flat_map(|(ci, groups)| {
            groups.iter().enumerate().flat_map(move |(hi, h)| {
                (0..h.nodes.len())
                    .step_by(HOST_RAYS_PER_ITEM)
                    .map(move |a| (ci, hi, a, (a + HOST_RAYS_PER_ITEM).min(h.nodes.len())))
            })
        })
        .filter(|&(ci, hi, _, _)| prepared.host[ci][hi].traced.is_none())
        .collect();
    let host = &prepared.host;
    let early_ops: Vec<&Operand> = operands.iter().copied().filter(|o| rayless(o)).collect();
    let want_early = prepared.early.is_none();
    // Both on the pool at once: the routed rays and the rayless operands.
    let (traced, early) = rayon::join(
        || {
            items
                .par_iter()
                .map(|&(ci, hi, a, b)| {
                    let h = &host[ci][hi];
                    candidates[ci].0.trace_nodes(h.key, &h.nodes[a..b])
                })
                .collect::<Vec<_>>()
        },
        || {
            want_early.then(|| {
                candidates
                    .par_iter()
                    .enumerate()
                    .map(|(ci, (c, _))| {
                        cpu.host_values(tracers[ci].system().system(), c, &early_ops, ctx)
                    })
                    .collect::<MkResult<Vec<_>>>()
            })
        },
    );
    for (&(ci, hi, _, _), t) in items.iter().zip(traced) {
        prepared.host[ci][hi]
            .traced
            .get_or_insert_with(Vec::new)
            .extend(t);
    }
    if let Some(early) = early {
        prepared.early = Some(early?);
    }
    Ok(())
}

/// Host preparation of a batch (parallel over candidates): candidate plans,
/// chief seeds from the f64 tracers, and the resident plan. Rays in
/// `host_rays` stay off the device: their group
/// nodes are traced by [`host_stage`], every other kind on demand.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare<'a>(
    g: &'a GpuContext,
    cpu: &'a TraceEvaluator,
    tracers: &'a [Arc<SequentialTracer>],
    operands: &[&Operand],
    design: &RayDesign,
    host_rays: &BTreeSet<RayKey>,
    ctx: &JobContext,
) -> MkResult<Prepared<'a>> {
    let on_host = |key: RayKey| host_rays.contains(&key);
    ctx.check()?;
    let preparation = std::time::Instant::now();
    let wavefront = operands.iter().any(|o| {
        matches!(
            o.kind,
            OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
        )
    });
    let aggregates = operands.iter().all(|o| {
        matches!(
            o.kind,
            OperandKind::RmsSpot { .. }
                | OperandKind::SpotCentroid { .. }
                | OperandKind::WavefrontRms { .. }
        )
    });
    let last = tracers[0].system().image_index();
    if tracers
        .iter()
        .any(|t| t.num_wavelengths() > MAX_WAVELENGTHS || t.system().image_index() != last)
    {
        return Err(MkError::Unsupported(
            "resident merit batch topology/wavelength domain".into(),
        ));
    }
    let mut p = g.plan();
    let (set, uploaded) = {
        let packed = tracers
            .par_iter()
            .map(|t| PackedSystem::pack(t.system(), &Default::default()))
            .collect::<MkResult<Vec<_>>>()?;
        let h = g.backend.upload_systems(&packed)?;
        // This legacy boundary owns packed arrays; its copies occur before
        // the resident plan's own copy counter starts.
        let bytes = packed
            .iter()
            .map(|s| {
                size_of_val(&s.header)
                    + size_of_val(s.surfaces.as_slice())
                    + size_of_val(s.coefficients.as_slice())
            })
            .sum::<usize>() as u64;
        g.observe(PlanStats {
            uploaded_bytes: bytes,
            copied_bytes: if g.backend.info().kind == mklight_gpu::BackendKind::Metal {
                bytes
            } else {
                0
            },
            allocations: 3,
            peak_bytes: bytes,
            stages: vec![StageStats {
                name: "complete-system upload".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
        (SystemSet::Uploaded(h), Uploaded(g.backend.as_ref(), h))
    };
    // Candidates prepare independently; merging in candidate order keeps every
    // probe list, field index and reference row identical to a serial pass.
    let clip = cpu.settings().clip;
    let prepared = tracers
        .par_iter()
        .map(|tr| {
            let (mut c, plan) = Candidate::planned(cpu, tr, operands);
            c.prepare_points(&plan, false);
            c.cancel = Some(ctx.cancel.clone());
            let mut ri = std::collections::BTreeSet::new();
            for op in operands
                .iter()
                .filter(|o| matches!(o.kind, OperandKind::RelativeIllumination))
            {
                for (f, _) in c.fields(op.field).unwrap_or_default() {
                    for (w, _) in c.wavelengths(op.wavelength).unwrap_or_default() {
                        ri.insert((f, w));
                        ri.insert((FieldKey::Point(0, 0), w));
                    }
                }
            }
            let mut all = plan.chiefs.clone();
            for &(f, w, _, _) in &plan.singles {
                all.insert((f, w));
            }
            for &(f, w, _, _, _) in &plan.real {
                all.insert((f, w));
            }
            all.extend(ri.iter().copied());
            let field_keys: Vec<_> = all.iter().map(|x| x.0).collect();
            for f in field_keys {
                all.insert((f, tr.system().primary_wavelength));
            }
            let slot = |f, w| all.iter().position(|&k| k == (f, w)).unwrap() as u32;
            let mut local: BTreeMap<(bool, usize), Vec<Probe>> = BTreeMap::new();
            let nr = cpu.settings().illumination_rings.max(2);
            let na = cpu.settings().illumination_arms.max(8);
            let dt = std::f64::consts::TAU / na as f64;
            for (f, w) in ri {
                ctx.check()?;
                let slot = slot(f, w);
                let probes = local.entry((clip, last)).or_default();
                let mut push = |pupil: [f64; 2]| {
                    let q = Probe {
                        candidate: 0,
                        slot,
                        field: f,
                        wavelength: w,
                        pupil,
                        dest: Dest::Illumination,
                    };
                    // A host ray has no visibility entry: the host traces it.
                    if !on_host(ray_key(&q, clip, last)) {
                        probes.push(q);
                    }
                };
                push([0., 0.]);
                for j in 0..na {
                    let (sn, cs) = (dt * j as f64).sin_cos();
                    for k in 1..=nr {
                        let r = k as f64 / nr as f64;
                        push([r * cs, r * sn]);
                    }
                }
            }
            let mut host = Vec::new();
            for (gi, (&key, sampling)) in plan.groups.iter().enumerate() {
                ctx.check()?;
                let (nodes, symmetry) =
                    c.group_nodes_from(key.0, sampling, g.pupil_nodes(sampling));
                let slot = slot(key.0, key.1);
                let mut host_nodes = Vec::new();
                let probes = local.entry((clip, last)).or_default();
                for (i, node) in nodes.iter().enumerate() {
                    let q = Probe {
                        candidate: 0,
                        slot,
                        field: key.0,
                        wavelength: key.1,
                        pupil: [node.px, node.py],
                        dest: Dest::Node(gi as u32, i as u32),
                    };
                    if on_host(ray_key(&q, clip, last)) {
                        host_nodes.push(i);
                    } else {
                        probes.push(q);
                    }
                }
                if !host_nodes.is_empty() {
                    host.push(HostGroup {
                        gi: gi as u32,
                        key,
                        nodes: host_nodes,
                        traced: None,
                    });
                }
                let failed = if aggregates {
                    Vec::new()
                } else {
                    vec![NodeRay::FAILED; nodes.len()]
                };
                c.groups
                    .insert(key, Candidate::group_rays(nodes, symmetry, failed));
            }
            for &(f, w, px, py) in &plan.singles {
                let q = Probe {
                    candidate: 0,
                    slot: slot(f, w),
                    field: f,
                    wavelength: w,
                    pupil: [f64::from_bits(px), f64::from_bits(py)],
                    dest: Dest::Single,
                };
                if on_host(ray_key(&q, clip, last)) {
                    // Traced by the host on demand (no cache entry).
                    continue;
                }
                local.entry((clip, last)).or_default().push(q);
                c.singles.insert((f, w, px, py), NodeRay::FAILED);
            }
            for &(f, w, px, py, j) in &plan.real {
                if j == 0 || tr.system().surfaces[j].is_passive() {
                    // Traced on the host on demand (no cache entry).
                    continue;
                }
                let q = Probe {
                    candidate: 0,
                    slot: slot(f, w),
                    field: f,
                    wavelength: w,
                    pupil: [f64::from_bits(px), f64::from_bits(py)],
                    dest: Dest::Real(j as u32),
                };
                if !on_host(ray_key(&q, clip, j)) {
                    local.entry((clip, j)).or_default().push(q);
                }
            }
            Ok(((c, plan), all, local, host))
        })
        .collect::<MkResult<Vec<_>>>()?;
    let mut candidates = Vec::with_capacity(tracers.len());
    let mut host = Vec::with_capacity(tracers.len());
    let mut chiefs = Vec::new();
    let mut batches: BTreeMap<(bool, usize), Vec<Probe>> = BTreeMap::new();
    // Rows in (candidate, field, λ) order, which is also the slot order.
    let mut field_keys = Vec::new();
    let mut refs = Vec::new();
    for (ci, (candidate, all, local, host_groups)) in prepared.into_iter().enumerate() {
        host.push(host_groups);
        let tr = &tracers[ci];
        let base = field_keys.len() as u32;
        for &(f, w) in &all {
            chiefs.push(Probe {
                candidate: ci,
                slot: field_keys.len() as u32,
                field: f,
                wavelength: w,
                pupil: [0.; 2],
                dest: Dest::Chief,
            });
            refs.extend([
                tr.paraxial().at(w).exit_pupil_position,
                tr.system().n_before(last, w),
                tr.system().wavelengths_um[w],
            ]);
            field_keys.push((ci, f, w));
        }
        for (key, probes) in local {
            batches
                .entry(key)
                .or_default()
                .extend(probes.into_iter().map(|q| Probe {
                    candidate: ci,
                    slot: q.slot + base,
                    ..q
                }));
        }
        candidates.push(candidate);
    }
    let field_spec = |(ci, f, w): (usize, FieldKey, usize)| {
        ctx.check()?;
        let tr = &tracers[ci];
        let point = candidates[ci]
            .0
            .field(f)
            .map_err(|e| MkError::Unsupported(format!("field: {e:?}")))?
            .point;
        a::FieldSpec::new(
            tr.system(),
            tr.paraxial(),
            point,
            ci as u32,
            w as u32,
            tr.aiming(),
        )
    };
    let metadata = records(g, &mut p, field_keys.len(), |i| field_spec(field_keys[i]))?;
    // Chief rows from each candidate's own f64 tracer (system fields are
    // prepared at its construction): the device solves no per-candidate chief
    // or real-image-height source, the work that once ran on a few threads
    // for seconds per command.
    let per_candidate = (0..tracers.len())
        .into_par_iter()
        .map(|ci| {
            let rows = (field_keys.iter())
                .filter(|k| k.0 == ci)
                .map(|&(_, f, w)| {
                    let field = candidates[ci]
                        .0
                        .field(f)
                        .map_err(|e| MkError::Unsupported(format!("field: {e:?}")))?;
                    Ok((field.point, w))
                })
                .collect::<MkResult<Vec<_>>>()?;
            Ok(a::host_seeds(&tracers[ci], &rows))
        })
        .collect::<MkResult<Vec<_>>>()?;
    let mut seed_rows = Vec::with_capacity(field_keys.len());
    for (rows, host) in per_candidate {
        seed_rows.extend(rows);
        p.host.add(&host);
    }
    // Rows are in (candidate, field, wavelength) order, as are field_keys.
    let seeds = records(g, &mut p, seed_rows.len(), |i| Ok(seed_rows[i]))?;
    // P1's seed-table choice is per candidate, not a side effect of which
    // other candidates share its command. Partition only at that boundary.
    let mut field_counts = vec![0usize; tracers.len()];
    for &(ci, _, _) in &field_keys {
        field_counts[ci] += 1;
    }
    let mut partitions = Vec::new();
    for (key, probes) in batches {
        let mut counts = vec![0usize; tracers.len()];
        for q in &probes {
            counts[q.candidate] += 1;
        }
        let (table, chief): (Vec<_>, Vec<_>) = probes.into_iter().partition(|q| {
            counts[q.candidate] >= a::TABLE_MIN_RAYS_PER_ROW * field_counts[q.candidate]
        });
        for probes in [chief, table] {
            if !probes.is_empty() {
                partitions.push((key, probes));
            }
        }
    }
    let traced_rays = chiefs.len()
        + partitions
            .iter()
            .map(|(_, probes)| probes.len())
            .sum::<usize>();
    let tolerance = tracers
        .iter()
        .map(|t| {
            match cpu
                .settings()
                .tracer
                .aiming
                .unwrap_or(t.system().system().ray_aiming)
            {
                mklight_core::RayAiming::Real { tolerance } => tolerance as f32,
                _ => 1e-10,
            }
        })
        .fold(f32::INFINITY, f32::min);
    // Without real aiming every ray is generated in the first command, and
    // chief rows are converged host seeds: those aims need no continuation
    // beyond the one round the executor commits.
    let real = tracers.iter().any(|t| {
        matches!(
            cpu.settings()
                .tracer
                .aiming
                .unwrap_or(t.system().system().ray_aiming),
            mklight_core::RayAiming::Real { .. }
        )
    });
    let design = &RayDesign {
        speculative_rounds: if real {
            design.speculative_rounds
        } else {
            design.speculative_rounds.min(1)
        },
        ..*design
    };
    let ch = host_chief_rays(g, &mut p, &set, &chiefs, &candidates, tracers, last, ctx)?;
    let reference = g.df(&mut p, [chiefs.len(), 3], &refs)?;
    p.outputs = vec![ch.rays.floats, ch.status, ch.rays.report];
    let mut moment_segments = Vec::new();
    let mut fit_inputs = Vec::new();
    for (key, probes) in &partitions {
        let (clip, last) = *key;
        let selected: std::collections::BTreeSet<_> = probes.iter().map(|q| q.candidate).collect();
        let (aim_metadata, aim_seeds, remap) = if selected.len() == tracers.len() {
            (metadata.clone(), seeds.clone(), None)
        } else {
            let mut remap = vec![u32::MAX; field_keys.len()];
            let mut subset = Vec::new();
            for (i, k) in field_keys.iter().enumerate() {
                if selected.contains(&k.0) {
                    remap[i] = subset.len() as u32;
                    subset.push(i);
                }
            }
            (
                records(g, &mut p, subset.len(), |i| {
                    field_spec(field_keys[subset[i]])
                })?,
                records(g, &mut p, subset.len(), |i| Ok(seed_rows[subset[i]]))?,
                Some(remap),
            )
        };
        let h = if design.host_aim {
            host_aimed_rays(
                g,
                &mut p,
                &set,
                probes,
                &candidates,
                tracers,
                last,
                clip,
                ctx,
            )?
        } else {
            append(
                g,
                &mut p,
                &set,
                &aim_metadata,
                &aim_seeds,
                probes,
                remap.as_deref(),
                last,
                clip,
                tolerance,
                design,
                ctx,
            )?
        };
        let index = g.fill(&mut p, [probes.len()], |out: &mut [u32]| {
            for (o, q) in out.iter_mut().zip(probes) {
                *o = q.slot;
            }
        })?;
        let (opd, status) = if wavefront {
            let opd = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [probes.len()]));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [probes.len()]));
            p.operations
                .push(ResidentOp::Geometry(mklight_gpu::geometry::Request::Opd(
                    mklight_gpu::geometry::Opd {
                        axial: false,
                        hits: h.clone(),
                        chiefs: ch.clone(),
                        chief_index: index.clone(),
                        reference: reference.clone(),
                        output: opd,
                        status,
                    },
                )));
            (opd, status)
        } else {
            (h.status, h.status)
        };
        p.outputs
            .extend([h.rays.floats, h.status, h.rays.report, opd, status]);
        let mut segments = Vec::new();
        let mut bounds = vec![0u32];
        for (i, q) in probes.iter().enumerate() {
            let key = match q.dest {
                Dest::Node(group, _) => Some((q.candidate, group)),
                _ => None,
            };
            if i == 0 {
                segments.push(key);
            } else if segments.last() != Some(&key) {
                bounds.push(i as u32);
                segments.push(key);
            }
        }
        bounds.push(probes.len() as u32);
        if segments.iter().all(Option::is_none) {
            p.outputs.extend([h.status, h.status]);
            moment_segments.push(Vec::new());
            fit_inputs.push((probes, opd, status));
            continue;
        }
        let offsets = g.fill(&mut p, [bounds.len()], |out: &mut [u32]| {
            out.copy_from_slice(&bounds)
        })?;
        let weights = g.fill(&mut p, [probes.len()], |out: &mut [Df32]| {
            // Group probes are contiguous: resolve each group once.
            let mut group = None;
            for (o, q) in out.iter_mut().zip(probes) {
                *o = if let Dest::Node(gi, j) = q.dest {
                    let nodes = match group {
                        Some((key, nodes)) if key == (q.candidate, gi) => nodes,
                        _ => {
                            let (c, plan) = &candidates[q.candidate];
                            let nodes =
                                &c.groups[plan.groups.keys().nth(gi as usize).unwrap()].nodes[..];
                            group = Some(((q.candidate, gi), nodes));
                            nodes
                        }
                    };
                    Df32::from_f64(nodes[j as usize].weight)
                } else {
                    Df32::ZERO
                };
            }
        })?;
        let points = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, probes.len()]));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [probes.len()]));
        p.operations.push(ResidentOp::Geometry(
            mklight_gpu::geometry::Request::Coordinates(mklight_gpu::geometry::Coordinates {
                hits: h.clone(),
                chiefs: ch.clone(),
                chief_index: index,
                defocus: Df32::ZERO,
                output: points,
                mask,
            }),
        ));
        let moments = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [segments.len(), 10],
        ));
        let moment_status = p.scratch(BufferDesc::new(ElementType::U32, [segments.len()]));
        let mut xy = p.view(points)?;
        xy.desc.shape = vec![2, probes.len()];
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::SegmentReduce(
                mklight_gpu::collectives::SegmentReduce {
                    values: xy,
                    offsets,
                    mask: Some(p.view(mask)?),
                    mask_value: 1,
                    weights: Some(weights),
                    max_segment_len: bounds
                        .windows(2)
                        .map(|w| (w[1] - w[0]) as usize)
                        .max()
                        .unwrap_or(0),
                    reference: None,
                    mode: mklight_gpu::ReduceMode::DoubleFloat,
                    kind: mklight_gpu::collectives::SegmentKind::Moments2,
                    output: moments,
                    status: moment_status,
                },
            ),
        ));
        p.outputs.extend([moments, moment_status]);
        moment_segments.push(segments);
        fit_inputs.push((probes, opd, status));
    }
    let mut fits = Vec::new();
    if wavefront {
        for (probes, opd, mask) in fit_inputs {
            fits.extend(fits::append(g, &mut p, &candidates, probes, opd, mask)?);
        }
    }
    let seconds = preparation.elapsed().as_secs_f64();
    g.observe(PlanStats {
        wall_seconds: seconds,
        stages: vec![StageStats {
            name: "candidate preparation".into(),
            wall_seconds: seconds,
            ..Default::default()
        }],
        ..Default::default()
    });
    Ok(Prepared {
        plan: Some(p),
        candidates,
        host,
        early: None,
        chiefs,
        partitions,
        moment_segments,
        fits,
        uploaded,
        traced_rays,
        aggregates,
        image: last,
    })
}

/// A completed plan's outputs, read in place, with what the device left
/// unresolved.
pub(crate) struct Traced<'a> {
    prepared: Prepared<'a>,
    views: Vec<ReadView>,
    /// Per candidate, the rays the device left unresolved.
    pub(crate) unresolved: Vec<BTreeSet<RayKey>>,
    /// Per candidate, the rays the device resolved only after more than
    /// [`SLOW_ITERATIONS`] Newton iterations (they hold up their commands).
    pub(crate) slow: Vec<BTreeSet<RayKey>>,
}
/// Device aiming iterations above which a resolved ray counts as slow.
pub(crate) const SLOW_ITERATIONS: u32 = 6;

/// Read a completed plan's outputs in place and find the unresolved rays.
pub(crate) fn collect<'a>(
    g: &GpuContext,
    prepared: Prepared<'a>,
    result: PlanResult,
) -> MkResult<Traced<'a>> {
    let n = prepared.candidates.len();
    let views = result
        .outputs
        .iter()
        .map(|o| g.backend.read_view(o.view()))
        .collect::<MkResult<Vec<_>>>()?;
    let chief_status = views[1].as_slice::<u32>()?;
    if trace_routing() {
        let k = prepared.chiefs.len();
        let aim = views[2].as_slice::<u32>()?[..k].iter().filter(|&&s| s != 0);
        eprintln!(
            "MKL_SOLVERS chiefs: {k} of {n} candidates, aim-failed {}, trace-failed {}",
            aim.count(),
            chief_status[..k].iter().filter(|&&s| s != 0).count()
        );
    }
    let mut chief_keys = vec![BTreeSet::new(); n];
    for (i, (q, &status)) in prepared
        .chiefs
        .iter()
        .zip(views[2].as_slice::<u32>()?)
        .enumerate()
    {
        // A chief ray the device could not trace: the f64 chief replaces
        // it, with every ray of its field and wavelength traced on the host
        // (they depend on the chief through the device's chief-relative
        // coordinates and OPD).
        if status != 0 || chief_status[i] != 0 {
            chief_keys[q.candidate].insert(ray_key(q, false, prepared.image));
        }
    }
    let mut unresolved = chief_keys;
    let mut slow = vec![BTreeSet::new(); n];
    for (bi, ((clip, last), probes)) in prepared.partitions.iter().enumerate() {
        let report = views[3 + 7 * bi + 2].as_slice::<u32>()?;
        let traced = views[3 + 7 * bi + 1].as_slice::<u32>()?;
        let count = probes.len();
        if trace_routing() {
            let aim = (0..count).filter(|&i| unresolved_report(report[i])).count();
            let clipped = (0..count)
                .filter(|&i| report[i] == 0 && traced[i] != 0)
                .count();
            let slow = (0..count)
                .filter(|&i| report[i] == 0 && report[count + i] > SLOW_ITERATIONS)
                .count();
            eprintln!(
                "MKL_SOLVERS partition {bi}: {count} probes of {n} candidates, aim-unresolved \
                 {aim}, trace-failed {clipped}, slow {slow}"
            );
        }
        for (i, q) in probes.iter().enumerate() {
            let code = report[i];
            // Unresolved aims are completed by the host in f64. A ray the
            // device aimed and then traced to a failure (clipped by an
            // aperture) stays a failure: an f32/f64 disagreement needs the
            // ray within ~1e-6 mm of the edge, and completing every clipped
            // ray of a vignetted lens would cost the host as much as the
            // reference.
            if unresolved_report(code) {
                unresolved[q.candidate].insert(ray_key(q, *clip, *last));
            } else if code == 0 && report[count + i] > SLOW_ITERATIONS {
                slow[q.candidate].insert(ray_key(q, *clip, *last));
            }
        }
    }
    Ok(Traced {
        prepared,
        views,
        unresolved,
        slow,
    })
}

/// Fill the candidates' caches from a traced batch, complete on the host the
/// rays the device left unresolved (and, for the probes and base point of
/// one finite difference, every ray in `uniform` in every candidate, so an
/// operand never differences a device ray against a host one), and compute
/// the operand values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish(
    g: &GpuContext,
    cpu: &TraceEvaluator,
    tracers: &[Arc<SequentialTracer>],
    operands: &[&Operand],
    traced: Traced<'_>,
    uniform: Option<&BTreeSet<RayKey>>,
    ctx: &JobContext,
) -> MkResult<Vec<Vec<OperandValue>>> {
    let assembly = std::time::Instant::now();
    let Traced {
        prepared,
        views,
        unresolved: rays,
        ..
    } = traced;
    let Prepared {
        mut candidates,
        host,
        early,
        chiefs,
        partitions,
        moment_segments,
        fits,
        uploaded,
        aggregates,
        image,
        ..
    } = prepared;
    // Chief rays the host takes over (failed on the device, here or, in a
    // finite difference, for another candidate): with every ray of their
    // field and wavelength.
    let host_chiefs: BTreeSet<(usize, FieldKey, usize)> = chiefs
        .iter()
        .filter(|q| {
            let key = ray_key(q, false, image);
            rays[q.candidate].contains(&key) || uniform.is_some_and(|u| u.contains(&key))
        })
        .map(|q| (q.candidate, q.field, q.wavelength))
        .collect();
    let wavefront = operands.iter().any(|o| {
        matches!(
            o.kind,
            OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
        )
    });
    let f = views[0].as_slice::<f32>()?;
    let status = views[1].as_slice::<u32>()?;
    for (i, q) in chiefs.iter().enumerate() {
        let data = if host_chiefs.contains(&(q.candidate, q.field, q.wavelength)) {
            candidates[q.candidate].0.host_chief(q.field, q.wavelength)
        } else {
            let chief = trace_result(f, status, chiefs.len(), i);
            let reference = chief
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|r| gpu_reference(&tracers[q.candidate], r, q.wavelength));
            ChiefData { chief, reference }
        };
        candidates[q.candidate]
            .0
            .chiefs
            .insert((q.field, q.wavelength), data);
    }
    // Which probes the host completes, and which groups (candidate, group
    // index) that touches.
    let complete = |q: &Probe, key: &RayKey| {
        rays[q.candidate].contains(key)
            || uniform.is_some_and(|u| u.contains(key))
            || host_chiefs.contains(&(q.candidate, q.field, q.wavelength))
    };
    let mut dirty: BTreeSet<(usize, u32)> = BTreeSet::new();
    let mut completions: BTreeMap<(usize, u32), Vec<usize>> = BTreeMap::new();
    let mut completed_rays = 0usize;
    // Groups with nodes routed to the host before the plan.
    for (ci, groups) in host.iter().enumerate() {
        for h in groups {
            dirty.insert((ci, h.gi));
        }
    }
    for ((clip, last), probes) in &partitions {
        for q in probes {
            if complete(q, &ray_key(q, *clip, *last)) {
                completed_rays += 1;
                if let Dest::Node(gi, j) = q.dest {
                    dirty.insert((q.candidate, gi));
                    completions
                        .entry((q.candidate, gi))
                        .or_default()
                        .push(j as usize);
                }
            }
        }
    }
    // Aggregate-only batches keep no host rays; a candidate with a completed
    // group gets all its groups' rays, so that field and spectral aggregates
    // over several groups are all formed from rays.
    let materialised: BTreeSet<usize> = if aggregates {
        dirty.iter().map(|&(ci, _)| ci).collect()
    } else {
        BTreeSet::new()
    };
    for &ci in &materialised {
        for g in candidates[ci].0.groups.values_mut() {
            if g.rays.len() != g.nodes.len() {
                let nodes = std::mem::take(&mut g.nodes);
                let n = nodes.len();
                *g = Candidate::group_rays(nodes, g.symmetry, vec![NodeRay::FAILED; n]);
            }
        }
    }
    let own_rays =
        |ci: usize, gi: u32| !aggregates || materialised.contains(&ci) || dirty.contains(&(ci, gi));
    for (bi, ((clip, last), probes)) in partitions.iter().enumerate() {
        if aggregates && materialised.is_empty() {
            continue;
        }
        let offset = 3 + 7 * bi;
        let f = views[offset].as_slice::<f32>()?;
        let status = views[offset + 1].as_slice::<u32>()?;
        let opd = if wavefront {
            Some(views[offset + 3].as_slice::<Df32>()?)
        } else {
            None
        };
        let valid = views[offset + 4].as_slice::<u32>()?;
        let mut group = None;
        for (i, q) in probes.iter().enumerate() {
            if complete(q, &ray_key(q, *clip, *last)) {
                // An absent cache entry makes the host trace the ray itself;
                // group nodes are completed below.
                if let Dest::Single = q.dest {
                    let [px, py] = q.pupil.map(f64::to_bits);
                    candidates[q.candidate]
                        .0
                        .singles
                        .remove(&(q.field, q.wavelength, px, py));
                }
                continue;
            }
            if matches!(q.dest, Dest::Illumination) {
                candidates[q.candidate].0.resident_visibility.insert(
                    (
                        q.field,
                        q.wavelength,
                        q.pupil[0].to_bits(),
                        q.pupil[1].to_bits(),
                    ),
                    (status[i] != 0).then_some(status[probes.len() + i] as usize),
                );
                continue;
            }
            if let Dest::Node(gi, _) = q.dest
                && !own_rays(q.candidate, gi)
            {
                continue;
            }
            let hit = trace_result(f, status, probes.len(), i);
            let node = NodeRay {
                pos: hit.as_ref().ok().map(|r| [r.ray.pos.x, r.ray.pos.y]),
                opd: opd.and_then(|v| (valid[i] == 0).then(|| v[i].to_f64())),
            };
            let (c, plan) = &mut candidates[q.candidate];
            let [px, py] = q.pupil.map(f64::to_bits);
            match q.dest {
                Dest::Node(gi, j) => {
                    let key = match group {
                        Some((at, key)) if at == (q.candidate, gi) => key,
                        _ => {
                            let key = *plan.groups.keys().nth(gi as usize).unwrap();
                            group = Some(((q.candidate, gi), key));
                            key
                        }
                    };
                    c.groups.get_mut(&key).unwrap().rays[j as usize] = node;
                }
                Dest::Single => {
                    c.singles.insert((q.field, q.wavelength, px, py), node);
                }
                Dest::Real(surface) => {
                    c.real_rays.insert(
                        (q.field, q.wavelength, px, py, surface as usize),
                        hit.map(|r| RealHit {
                            point: r.ray.pos,
                            dir: r.ray.dir,
                            opl: r.opl.hi + r.opl.lo,
                        }),
                    );
                }
                Dest::Chief | Dest::Illumination => unreachable!(),
            }
        }
    }
    // Host-routed nodes (traced by `host_stage` while the device ran, or
    // now), with their OPD against the chiefs just read.
    let routed = std::time::Instant::now();
    let mut host = host;
    candidates
        .par_iter_mut()
        .zip(host.par_iter_mut())
        .for_each(|((c, _), groups)| {
            for h in groups {
                let traced = match h.traced.take() {
                    Some(t) => t,
                    None => c.trace_nodes(h.key, &h.nodes),
                };
                c.set_nodes(h.key, &h.nodes, &traced);
            }
        });
    let routed_seconds = routed.elapsed().as_secs_f64();
    // Device aggregates stand for the groups the host left alone.
    let keep = |ci: usize, gi: u32| !materialised.contains(&ci) && !dirty.contains(&(ci, gi));
    for (bi, segments) in moment_segments.iter().enumerate() {
        if segments.is_empty() {
            continue;
        }
        let moments = views[3 + 7 * bi + 5].as_slice::<Df32>()?;
        for (si, key) in segments.iter().enumerate() {
            if let &Some((ci, group)) = key {
                if !keep(ci, group) {
                    continue;
                }
                let (c, plan) = &mut candidates[ci];
                let key = plan.groups.keys().nth(group as usize).unwrap();
                let mut m: [f64; 10] = std::array::from_fn(|i| moments[10 * si + i].to_f64());
                if let Ok(chief) = &c.chiefs[&(key.0, key.1)].chief {
                    m[2] += chief.ray.pos.x;
                    m[3] += chief.ray.pos.y;
                }
                if c.groups[key].symmetry == crate::merit::PupilSymmetry::Bilateral {
                    m[5] += m[1] * m[2] * m[2];
                    m[2] = 0.;
                }
                c.resident_moments.insert(*key, m);
            }
        }
    }
    fits::finish(&fits, &views, &mut candidates)?;
    if !dirty.is_empty() {
        for (ci, (c, plan)) in candidates.iter_mut().enumerate() {
            for (gi, key) in plan.groups.keys().enumerate() {
                if !keep(ci, gi as u32) {
                    c.resident_wavefits.remove(key);
                }
            }
        }
    }
    // Host f64 completion of the unresolved group nodes, in parallel over
    // candidates, groups and runs of nodes (a single candidate's failing
    // aims are the slowest host work of a batch).
    let completion = std::time::Instant::now();
    if !completions.is_empty() {
        let groups: Vec<(usize, GroupKey, Vec<usize>)> = completions
            .into_iter()
            .map(|((ci, gi), nodes)| {
                let key = *candidates[ci].1.groups.keys().nth(gi as usize).unwrap();
                (ci, key, nodes)
            })
            .collect();
        let items: Vec<(usize, usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(k, (_, _, nodes))| {
                (0..nodes.len())
                    .step_by(HOST_RAYS_PER_ITEM)
                    .map(move |a| (k, a, (a + HOST_RAYS_PER_ITEM).min(nodes.len())))
            })
            .collect();
        let traced: Vec<_> = items
            .par_iter()
            .map(|&(k, a, b)| {
                let (ci, key, nodes) = &groups[k];
                candidates[*ci].0.trace_nodes(*key, &nodes[a..b])
            })
            .collect();
        let mut per_group: Vec<Vec<Option<TraceResult>>> = vec![Vec::new(); groups.len()];
        for (&(k, _, _), t) in items.iter().zip(traced) {
            per_group[k].extend(t);
        }
        for ((ci, key, nodes), traced) in groups.iter().zip(per_group) {
            candidates[*ci].0.set_nodes(*key, nodes, &traced);
        }
    }
    let completion_seconds = completion.elapsed().as_secs_f64();
    let values = candidates
        .par_iter()
        .enumerate()
        .map(|(ci, (c, _))| {
            let system = tracers[ci].system().system();
            let Some(early) = &early else {
                return cpu.host_values(system, c, operands, ctx);
            };
            // The rayless values came from the host stage.
            let late: Vec<&Operand> = operands.iter().copied().filter(|o| !rayless(o)).collect();
            let mut late = cpu.host_values(system, c, &late, ctx)?.into_iter();
            let mut early = early[ci].iter().cloned();
            Ok(operands
                .iter()
                .map(|o| {
                    if rayless(o) {
                        early.next().unwrap()
                    } else {
                        late.next().unwrap()
                    }
                })
                .collect())
        })
        .collect::<MkResult<Vec<_>>>()?;
    drop(uploaded);
    let seconds = assembly.elapsed().as_secs_f64();
    let ri = operands
        .iter()
        .filter(|o| matches!(o.kind, OperandKind::RelativeIllumination))
        .count();
    if ri > 0 {
        let count = |i: usize| {
            candidates
                .iter()
                .map(|(c, _)| c.visibility_lookups[i].load(std::sync::atomic::Ordering::Relaxed))
                .sum()
        };
        g.observe_ri_visibility(count(0), count(1));
        g.observe(PlanStats {
            fallback_count: (ri * tracers.len()) as u64,
            stages: vec![StageStats {
                name: "RI adaptive reference completion".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
    }
    ctx.check()?;
    g.observe(PlanStats {
        polished_rays: completed_rays as u64,
        stages: vec![
            StageStats {
                name: "host f64 ray completion".into(),
                wall_seconds: completion_seconds,
                ..Default::default()
            },
            StageStats {
                name: "host-routed rays (install)".into(),
                wall_seconds: routed_seconds,
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    g.observe(PlanStats {
        wall_seconds: seconds,
        stages: vec![StageStats {
            name: "operand assembly".into(),
            wall_seconds: seconds,
            ..Default::default()
        }],
        ..Default::default()
    });
    Ok(values)
}
