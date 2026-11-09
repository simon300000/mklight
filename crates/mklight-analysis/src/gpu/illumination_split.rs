//! Image-point illumination on the device: the CPU solves, the GPU traces
//! the pencils.
//!
//! The host prepares every field exactly as the reference does
//! (`illumination::image_point`, in parallel over fields): the real-aimed
//! chief, image point and basis, chief ray angle and `cos⁴`, and the chord
//! inverse of `∂X/∂q`. The CPU then solves the image points of the pupil
//! nodes (the reference's Newton iterations, bitwise) in field blocks, and
//! the GPU traces each block's pencils (the clipped ray and the eight
//! differentiation rays, seeded with the converged aim;
//! `mkl_geometry_ri_rays`, one thread per ray) while the CPU solves the next
//! block.
//!
//! The host forms the pencils' determinants and the per-field sums in f64
//! (reading the shared result buffers in place), in the reference's node
//! order and weights. A pencil the device could not decide is finished by
//! the reference from the CPU's own solve; a node the CPU solve fails is
//! the reference's failure.
use super::*;
use crate::illumination::{self as host, ImagePoint};
use crate::util::m2;
use mklight_core::math::KahanSum;
use mklight_gpu::consts::*;
use mklight_gpu::geometry::{
    self, IlluminationRays,
    ri::{PENCIL, RiField, RiRay, RiTask},
};
use rayon::prelude::*;
use std::time::Instant;

/// Nodes per plan (bounds plan storage).
const PLAN_NODES: usize = 1 << 18;
/// A node's CPU image-point solve: the solved `q` and its converged aim, or
/// the reference's failure.
type Solved = Option<Converged>;
/// The solved `q` of a node and the aim of its converged ray.
type Converged = ([f64; 2], [f64; 2]);
/// A field block in the pipeline: its fields, CPU solves and pencil plan.
type Block<'a> = (&'a [(usize, u32)], Vec<Solved>, Submitted);
/// A submitted pencil plan and its traced node count.
type Submitted = Option<(mklight_gpu::pending::Pending<PlanResult>, usize)>;

/// What decided a node.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    /// Surviving, with its projected-solid-angle density.
    Survives(f64),
    Vignetted,
    /// The reference's own failure (no ray through the image point).
    Failed,
    /// The device did not decide; the reference completes it.
    Unresolved,
}

/// Host stage times (ms) and counts, recorded as evidence.
#[derive(Default, serde::Serialize)]
struct Stages {
    host_fields_ms: f64,
    cpu_solve_ms: f64,
    device_wait_ms: f64,
    assembly_ms: f64,
    polish_ms: f64,
    plans: usize,
    blocks: usize,
    nodes: usize,
    unresolved: usize,
    polished_failed: usize,
    /// Nodes the CPU finished from its own solve.
    cpu_finished: usize,
    /// Failure codes (low byte) of the traced pencil rays.
    pencil_failures: std::collections::BTreeMap<u32, usize>,
}

fn df2(v: [f64; 2]) -> [Df32; 2] {
    v.map(Df32::from_f64)
}
fn ray_value(r: &RiRay, i: usize) -> f64 {
    match i {
        0 | 1 => r.x[i].to_f64(),
        _ => r.lm[i - 2].to_f64(),
    }
}

/// The device row of a prepared image point (`MklRiField`).
fn field_row(
    t: &SequentialTracer,
    point: [f64; 2],
    ip: &ImagePoint,
    chunk: (u32, u32),
    wl: usize,
    real: mklight_core::RayAiming,
    aim_floor: f64,
) -> MkResult<RiField> {
    let aim = FieldSpec::new(
        t.system(),
        t.paraxial(),
        FieldPoint::normalized(point[0], point[1]),
        chunk.0,
        chunk.1,
        real,
    )?;
    let ad = &ip.field.aims[wl];
    let inverse = ad.jacobian.and_then(|j| m2::inv(&j));
    let [b0, b1] = ip.basis;
    Ok(RiField {
        aim,
        chief: df2(ad.chief),
        aim_inverse: inverse
            .map(|m| [m[0][0], m[0][1], m[1][0], m[1][1]].map(Df32::from_f64))
            .unwrap_or([Df32::ZERO; 4]),
        basis: [b0.x, b0.y, b0.z, b1.x, b1.y, b1.z].map(Df32::from_f64),
        tolerance: Df32::from_f64(ip.tolerance),
        status: u32::from(inverse.is_none()),
        aim_floor: aim_floor as f32,
    })
}

/// Stall acceptance of the device stop aims (`MklRiField::aim_floor`): the
/// reference's aim tolerance of the field, within `[1e-10, 1e-9]` of the
/// stop radius. The lower bound is the double-float trace's noise floor on
/// aspheric systems; the upper bound keeps the pencils' central differences
/// (steps of 1e-4 of the stop radius) within ~1e-5 even where the
/// reference's own derived stop is looser, a stall above it failing to the
/// reference.
fn aim_floor(rt: &SequentialTracer, ip: &ImagePoint, wl: usize) -> f64 {
    rt.aim_tolerance_of(&ip.field, wl).clamp(1e-10, 1e-9)
}

/// The pencil of a CPU-solved node (f64 twin of `ri::pencil`): the clipped
/// ray at the converged aim, then `q₀ ±, q₁ ±, P₀ ±, P₁ ±`, the pupil
/// columns seeded by the chief prediction of their target change.
fn host_pencil(
    f: &RiField,
    row: u32,
    p: [f64; 2],
    q: [f64; 2],
    aim: [f64; 2],
    hq: f64,
) -> [RiTask; PENCIL] {
    let v = f.aim.vignette.map(|x| x.to_f64());
    let r_stop = f.aim.pupil[2].to_f64();
    let target = |p: [f64; 2]| {
        [
            (v[0] + (1.0 - v[2]) * p[0]) * r_stop,
            (v[1] + (1.0 - v[3]) * p[1]) * r_stop,
        ]
    };
    let inv = [
        [f.aim_inverse[0].to_f64(), f.aim_inverse[1].to_f64()],
        [f.aim_inverse[2].to_f64(), f.aim_inverse[3].to_f64()],
    ];
    let hp = geometry::ri::pupil_step().to_f64();
    std::array::from_fn(|i| {
        if i == 0 {
            return RiTask {
                field: row,
                flags: MKL_RI_CLIP | MKL_RI_SEEDED | MKL_RI_AIMED,
                q: df2(q),
                pupil: df2(p),
                seed: df2(aim),
            };
        }
        let s = i - 1;
        let (k, sign) = (s / 2, if s % 2 == 0 { 1.0 } else { -1.0 });
        let (mut qq, mut pp, mut seed) = (q, p, aim);
        if k < 2 {
            qq[k] += sign * hq;
        } else {
            pp[k - 2] += sign * hp;
            let (t0, t1) = (target(p), target(pp));
            let d = m2::apply(&inv, [t1[0] - t0[0], t1[1] - t0[1]]);
            seed = [aim[0] + d[0], aim[1] + d[1]];
        }
        RiTask {
            field: row,
            flags: MKL_RI_SEEDED,
            q: df2(qq),
            pupil: df2(pp),
            seed: df2(seed),
        }
    })
}

/// Tally the failure codes of traced pencil rays (skipped tasks excluded).
fn count_failures(rays: &[RiRay], stages: &mut Stages) {
    for r in rays {
        let code = r.status & MKL_RAY_CODE_MASK;
        if code != 0 && code != MKL_AIM_OUTSIDE {
            *stages.pencil_failures.entry(code).or_default() += 1;
        }
    }
}

/// A node's outcome from its traced pencil (the reference's semantics: a
/// failed clipped ray is vignetting; the density of the central
/// differences in f64).
fn pencil_outcome(rays: &[RiRay], hq: f64) -> Outcome {
    if rays[0].status != 0 {
        return Outcome::Vignetted;
    }
    let hp = geometry::ri::pupil_step().to_f64();
    let mut j = [[0.0; 4]; 4];
    for col in 0..4 {
        let (a, b) = (&rays[1 + 2 * col], &rays[2 + 2 * col]);
        if a.status != 0 || b.status != 0 {
            return Outcome::Unresolved;
        }
        let h = if col < 2 { hq } else { hp };
        for (i, row) in j.iter_mut().enumerate() {
            row[col] = (ray_value(a, i) - ray_value(b, i)) / (2.0 * h);
        }
    }
    let xq = [[j[0][0], j[0][1]], [j[1][0], j[1][1]]];
    let xp = [[j[0][2], j[0][3]], [j[1][2], j[1][3]]];
    let lq = [[j[2][0], j[2][1]], [j[3][0], j[3][1]]];
    let lp = [[j[2][2], j[2][3]], [j[3][2], j[3][3]]];
    let Some(xqi) = m2::inv(&xq) else {
        return Outcome::Unresolved;
    };
    let corr = m2::mul(&lq, &m2::mul(&xqi, &xp));
    let d = m2::add(&lp, &m2::scale(&corr, -1.0));
    Outcome::Survives(m2::det(&d).abs())
}

/// Everything the solve needs about the fields and nodes.
struct Problem<'a> {
    t: &'a SequentialTracer,
    rt: &'a SequentialTracer,
    geo: &'a Geometry<'a>,
    wl: usize,
    /// Field point, prepared image point (or why not) and device row.
    points: Vec<[f64; 2]>,
    prepared: Vec<Result<ImagePoint, host::ImagePointError>>,
    rows: Vec<RiField>,
    /// Device row of each point (usable points only).
    row_of: Vec<Option<u32>>,
    /// The quadrature, `(Px, Py, weight)`.
    nodes: Vec<(f64, f64, f64)>,
}
impl Problem<'_> {
    fn per(&self) -> usize {
        self.nodes.len()
    }
    /// Usable points in order, as (point index, device row).
    fn usable(&self) -> Vec<(usize, u32)> {
        self.row_of
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .collect()
    }
    fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.t.system().snapshot.clone());
        p.settings = serde_json::json!({"operation": "relative_illumination_split"});
        p
    }
    fn import_fields(&self, p: &mut ResidentPlan) -> MkResult<BufferId> {
        self.geo
            .records::<RiField>(p, self.rows.len(), |i| Ok(self.rows[i]))
    }
    fn slots(&self) -> usize {
        self.t.system().num_surfaces()
    }
}

/// RI rows of `coordinates` (module docs).
pub fn rows(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    coordinates: &[[f64; 2]],
    wl: usize,
    radial: usize,
    azimuth: usize,
    ctx: &JobContext,
) -> MkResult<IlluminationRows> {
    let mut stages = Stages::default();
    let timer = mklight_core::job::RunTimer::start();
    let clock = Instant::now();
    let rt = host::real_aimed(t)?;
    let rt = rt.as_ref();
    let geo = Geometry::new(b, t, ctx)?;
    let mut points = coordinates.to_vec();
    let axis = match coordinates.iter().position(|p| *p == [0.0; 2]) {
        Some(k) => k,
        None => {
            points.push([0.0; 2]);
            points.len() - 1
        }
    };
    let q_scale = host::q_scale(rt);
    let prepared: Vec<_> = points
        .par_iter()
        .map(|&p| {
            ctx.check()?;
            Ok(host::image_point(rt, p, wl, q_scale))
        })
        .collect::<MkResult<_>>()?;
    let chunk = geo.chunk(wl)?;
    let real = mklight_core::RayAiming::Real {
        tolerance: match t.aiming() {
            mklight_core::RayAiming::Real { tolerance } => tolerance,
            _ => 1e-12,
        },
    };
    let fields: Vec<Option<RiField>> = points
        .par_iter()
        .zip(&prepared)
        .map(|(p, ip)| {
            ip.as_ref()
                .ok()
                .map(|ip| field_row(t, *p, ip, chunk, wl, real, aim_floor(rt, ip, wl)))
                .transpose()
        })
        .collect::<MkResult<_>>()?;
    let mut rows = Vec::new();
    let mut row_of = Vec::with_capacity(points.len());
    for field in fields {
        row_of.push(field.map(|f| {
            rows.push(f);
            rows.len() as u32 - 1
        }));
    }
    let problem = Problem {
        t,
        rt,
        geo: &geo,
        wl,
        points,
        prepared,
        rows,
        row_of,
        nodes: host::pupil_nodes(radial, azimuth),
    };
    stages.host_fields_ms = clock.elapsed().as_secs_f64() * 1e3;
    ctx.report(0.05, "relative illumination");
    // Outcome of every node of every usable point, point-major. The solves
    // report progress in 5…90 % (device plans as stage `geometry`).
    let usable = problem.usable();
    let mut runs: Vec<RunInfo> = Vec::new();
    let solve_ctx = ctx.subrange(0.05, 0.9);
    let mut outcomes = if usable.is_empty() {
        Vec::new()
    } else {
        host_solve(&problem, &usable, &mut stages, &mut runs, &solve_ctx)?
    };
    stages.nodes = outcomes.len();
    // Reference completion of undecided nodes.
    let clock = Instant::now();
    let per = problem.per();
    let unresolved: Vec<usize> = outcomes
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, Outcome::Unresolved))
        .map(|(i, _)| i)
        .collect();
    stages.unresolved = unresolved.len();
    if !unresolved.is_empty() {
        let selected: Vec<([f64; 2], usize)> = unresolved
            .iter()
            .map(|&i| (problem.points[usable[i / per].0], i % per))
            .collect();
        let decided = host::reference_nodes(rt, wl, q_scale, &problem.nodes, &selected, ctx)?;
        ctx.report(0.95, "relative illumination");
        for (&i, (density, survives, failed)) in unresolved.iter().zip(decided) {
            outcomes[i] = if survives {
                Outcome::Survives(density)
            } else if failed {
                stages.polished_failed += 1;
                Outcome::Failed
            } else {
                Outcome::Vignetted
            };
        }
    }
    stages.polish_ms = clock.elapsed().as_secs_f64() * 1e3;
    ctx.check()?;
    // Per-point sums in the reference's node order.
    let total: KahanSum = problem.nodes.iter().map(|n| n.2).collect();
    let mut out_rows = Vec::with_capacity(problem.points.len());
    let mut images = Vec::with_capacity(problem.points.len());
    let mut warnings = Vec::new();
    let mut u = 0;
    for (i, (p, ip)) in problem.points.iter().zip(&problem.prepared).enumerate() {
        match ip {
            Err(e) => {
                warnings.push(e.warning(p[0], p[1]));
                out_rows.push([f64::NAN; 4]);
                images.push([f64::NAN; 2]);
            }
            Ok(ip) => {
                let block = &outcomes[u * per..(u + 1) * per];
                u += 1;
                let mut omega = KahanSum::default();
                let mut area = KahanSum::default();
                let mut failed = 0;
                for (o, n) in block.iter().zip(&problem.nodes) {
                    match o {
                        Outcome::Survives(j) => {
                            omega.add(n.2 * j);
                            area.add(n.2);
                        }
                        Outcome::Failed | Outcome::Unresolved => failed += 1,
                        Outcome::Vignetted => {}
                    }
                }
                if failed > 0 && i < coordinates.len() {
                    warnings.push(format!(
                        "field ({}, {}): {failed} of {per} pupil nodes have no ray through the \
                         image point",
                        p[0], p[1]
                    ));
                }
                out_rows.push([omega.value(), area.value() / total.value(), ip.cra, ip.cos4]);
                images.push(ip.x0);
            }
        }
    }
    let omega0 = out_rows[axis][0];
    out_rows.truncate(coordinates.len());
    images.truncate(coordinates.len());
    if !(omega0 > 0.0) {
        return Err(mklight_core::MkError::Numerical(
            "on-axis beam has no projected solid angle".into(),
        ));
    }
    ctx.check()?;
    ctx.report(1.0, "relative illumination");
    let mut run = match runs.len() {
        // Nothing ran on the device.
        0 => timer.finish(
            &t.system().snapshot,
            "relative_illumination",
            &serde_json::json!({}),
            &b.info().name,
            b.precision().model,
            None,
            Vec::new(),
        ),
        _ => {
            let mut first = runs.remove(0);
            first.settings["extra_plans"] =
                serde_json::json!(runs.iter().map(|r| r.settings.clone()).collect::<Vec<_>>());
            first
        }
    };
    run.duration_ms = stages.host_fields_ms
        + stages.cpu_solve_ms
        + stages.device_wait_ms
        + stages.assembly_ms
        + stages.polish_ms;
    run.settings["ri_stages"] = serde_json::to_value(&stages).unwrap();
    // The reference-completion record.
    run.settings["ri_polish"] = serde_json::json!({
        "attempted_nodes": stages.unresolved,
        "resolved_nodes": stages.unresolved - stages.polished_failed,
        "remaining_failed_nodes": stages.polished_failed,
        "seconds": stages.polish_ms * 1e-3,
        "reference_backend": "cpu-f64",
    });
    run.warnings.extend(warnings);
    if stages.unresolved > 0 {
        run.warnings.push(format!(
            "RI: {} unresolved device nodes used the f64 reference; {} resolved",
            stages.unresolved,
            stages.unresolved - stages.polished_failed
        ));
    }
    Ok((out_rows, omega0, images, run))
}

/// The image-point solves on the CPU, the pencils on the device, in field
/// blocks.
fn host_solve(
    problem: &Problem,
    usable: &[(usize, u32)],
    stages: &mut Stages,
    runs: &mut Vec<RunInfo>,
    ctx: &JobContext,
) -> MkResult<Vec<Outcome>> {
    let per = problem.per();
    let pupil: Vec<[f64; 2]> = problem.nodes.iter().map(|n| [n.0, n.1]).collect();
    // Field blocks: about six (and at least 2048 nodes each) so the GPU's
    // last block is short.
    let fields_per_block = usable
        .len()
        .div_ceil(6)
        .max(2048usize.div_ceil(per))
        .min(PLAN_NODES / per)
        .max(1);
    let blocks: Vec<&[(usize, u32)]> = usable.chunks(fields_per_block).collect();
    stages.blocks = blocks.len();
    // Solve a block on the CPU: per node the solved (q, aim) or the
    // reference's failure. The block's fields run in parallel, and so do
    // the nodes of each field.
    let (rt, wl, prepared) = (problem.rt, problem.wl, &problem.prepared);
    let solve = |block: &[(usize, u32)], stages: &mut Stages| -> MkResult<Vec<Solved>> {
        let clock = Instant::now();
        let solved: Vec<Vec<Solved>> = block
            .par_iter()
            .map(|&(pi, _)| {
                ctx.check()?;
                let ip = prepared[pi].as_ref().unwrap();
                Ok(host::solve_image_point(rt, ip, wl, &pupil))
            })
            .collect::<MkResult<_>>()?;
        stages.cpu_solve_ms += clock.elapsed().as_secs_f64() * 1e3;
        Ok(solved.concat())
    };
    // The pencil plans share the job's cancellation but not its progress:
    // with CPU solves and device plans overlapping, the loop below reports
    // the progress, as stage `geometry` from the first submission on.
    let mut plan_ctx = JobContext::new();
    plan_ctx.cancel = ctx.cancel.clone();
    // Submit a block's pencils (converged nodes only).
    let submit = |block: &[(usize, u32)], solved: &[Solved]| -> MkResult<Submitted> {
        let traced: Vec<(usize, Converged)> = solved
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| (i, s)))
            .collect();
        if traced.is_empty() {
            return Ok(None);
        }
        let n = traced.len() * PENCIL;
        let mut p = problem.plan();
        let fields = problem.import_fields(&mut p)?;
        let tasks =
            problem
                .geo
                .import::<u32>(&mut p, [n, MKL_RI_TASK_WORDS as usize], |words| {
                    let tasks: &mut [RiTask] = bytemuck::cast_slice_mut(words);
                    tasks
                        .par_chunks_mut(PENCIL)
                        .zip(&traced)
                        .for_each(|(out, &(i, (q, aim)))| {
                            let (pi, row) = block[i / per];
                            let n = problem.nodes[i % per];
                            let hq = problem.prepared[pi].as_ref().unwrap().hq;
                            out.copy_from_slice(&host_pencil(
                                &problem.rows[row as usize],
                                row,
                                [n.0, n.1],
                                q,
                                aim,
                                hq,
                            ));
                        });
                    Ok(())
                })?;
        let rays = p.scratch(BufferDesc::new(
            ElementType::U32,
            [n, MKL_RI_RAY_WORDS as usize],
        ));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::IlluminationRays(
                IlluminationRays {
                    systems: SystemSet::Uploaded(problem.geo.systems),
                    fields: p.view(fields)?,
                    tasks: p.view(tasks)?,
                    slots: problem.slots(),
                    output: rays,
                },
            )));
        p.outputs = vec![rays];
        Ok(Some((
            problem.geo.backend.submit_plan(p, &plan_ctx)?,
            traced.len(),
        )))
    };
    // Outcomes of a block from its solves and traced pencils (`None`: the
    // block's pencils were not traced). A pencil the device could not decide
    // is finished by the reference from the CPU's own solve (the rest of the
    // reference's node, bitwise). Returns the outcomes and the fraction of
    // traced pencils the device could not decide.
    let assemble = |block: &[(usize, u32)],
                    solved: &[Solved],
                    result: Option<PlanResult>,
                    runs: &mut Vec<RunInfo>,
                    stages: &mut Stages|
     -> MkResult<(Vec<Outcome>, f64)> {
        let view = match &result {
            Some(r) => Some(problem.geo.backend.read_view(r.outputs[0].view())?),
            None => None,
        };
        let rays: &[RiRay] = view
            .as_ref()
            .map_or(&[][..], |v| bytemuck::cast_slice(v.words()));
        count_failures(rays, stages);
        // The pencils were traced for the converged nodes, in node order.
        let has_rays = view.is_some();
        let mut pencil = Vec::with_capacity(solved.len());
        let mut traced = 0;
        for s in solved {
            pencil.push(traced);
            traced += usize::from(s.is_some());
        }
        let out: Vec<Outcome> = solved
            .par_iter()
            .zip(&pencil)
            .enumerate()
            .map(|(i, (s, &k))| match s {
                None => Outcome::Failed,
                Some(_) if !has_rays => Outcome::Unresolved,
                Some(_) => {
                    let hq = problem.prepared[block[i / per].0].as_ref().unwrap().hq;
                    pencil_outcome(&rays[k * PENCIL..(k + 1) * PENCIL], hq)
                }
            })
            .collect();
        if !has_rays {
            traced = 0;
        }
        let undecided = if traced > 0 {
            out.iter()
                .zip(solved)
                .filter(|(o, s)| s.is_some() && matches!(o, Outcome::Unresolved))
                .count()
        } else {
            0
        };
        let mut out = out;
        // Finish the undecided nodes on the CPU: the fields in parallel, and
        // the nodes of each field too.
        let todo: Vec<(usize, Vec<usize>)> = block
            .iter()
            .enumerate()
            .map(|(b, &(pi, _))| {
                let range = b * per..(b + 1) * per;
                let todo: Vec<usize> = range
                    .filter(|&i| matches!(out[i], Outcome::Unresolved))
                    .collect();
                (pi, todo)
            })
            .filter(|(_, todo)| !todo.is_empty())
            .collect();
        if !todo.is_empty() {
            ctx.check()?;
            let decided: Vec<Vec<(f64, bool, bool)>> = todo
                .par_iter()
                .map(|(pi, todo)| {
                    let nodes: Vec<([f64; 2], [f64; 2])> = todo
                        .iter()
                        .map(|&i| (pupil[i % per], solved[i].unwrap().0))
                        .collect();
                    let ip = prepared[*pi].as_ref().unwrap();
                    host::finish_nodes(rt, ip, wl, &nodes)
                })
                .collect();
            for ((_, todo), decided) in todo.iter().zip(decided) {
                stages.cpu_finished += todo.len();
                for (&i, (d, survives, failed)) in todo.iter().zip(decided) {
                    out[i] = match (survives, failed) {
                        (true, _) => Outcome::Survives(d),
                        (_, true) => Outcome::Failed,
                        _ => Outcome::Vignetted,
                    };
                }
            }
        }
        if let Some(r) = result {
            runs.push(r.run_info);
        }
        Ok((out, undecided as f64 / traced.max(1) as f64))
    };
    let mut outcomes = Vec::with_capacity(usable.len() * per);
    // The pipeline: block k's pencils on the device while the CPU solves
    // block k+1. Where
    // the device leaves most pencils of a block undecided (its trace cannot
    // resolve the stop, e.g. wide-angle aspheric fields), the blocks not yet
    // submitted are finished on the CPU (a deterministic rule: it depends
    // only on the device's results).
    let mut pending: Option<Block> = None;
    let mut device = true;
    // Progress: a block's CPU solve and its assembly are half a block each.
    let (mut halves, mut stage) = (0, "relative illumination");
    let progress = |k: usize| k as f64 / (2 * blocks.len()) as f64;
    for block in blocks.iter().copied().chain(std::iter::once(&[][..])) {
        let next = if block.is_empty() {
            None
        } else {
            let solved = solve(block, stages)?;
            let submitted = if device {
                submit(block, &solved)?
            } else {
                None
            };
            stages.plans += usize::from(submitted.is_some());
            if submitted.is_some() {
                stage = "geometry";
            }
            halves += 1;
            ctx.report(progress(halves), stage);
            Some((block, solved, submitted))
        };
        if let Some((done, solved, submitted)) = pending.take() {
            let clock = Instant::now();
            let result = match submitted {
                Some((pending, _)) => Some(pending.wait()?),
                None => None,
            };
            stages.device_wait_ms += clock.elapsed().as_secs_f64() * 1e3;
            let clock = Instant::now();
            let traced = result.is_some();
            let (block_outcomes, undecided) = assemble(done, &solved, result, runs, stages)?;
            outcomes.extend(block_outcomes);
            if traced && undecided > DEVICE_GIVE_UP {
                device = false;
            }
            stages.assembly_ms += clock.elapsed().as_secs_f64() * 1e3;
            halves += 1;
            ctx.report(progress(halves), stage);
        }
        pending = next;
    }
    Ok(outcomes)
}

/// Fraction of a block's traced pencils left undecided by the
/// device above which the remaining blocks are finished on the CPU.
const DEVICE_GIVE_UP: f64 = 0.5;
