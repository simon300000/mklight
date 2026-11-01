//! P1/P4 resident candidate traces. The legacy evaluator above remains an
//! independent before/after benchmark route.
mod fits;
use super::*;
use crate::gpu::GpuContext;
use mklight_gpu::{Df32, aiming as a, resident::*, systems};
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
struct Uploaded<'a>(&'a dyn ComputeBackend, SystemHandle);
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
#[allow(clippy::too_many_arguments)]
fn append(
    g: &GpuContext,
    p: &mut ResidentPlan,
    set: &SystemSet,
    metadata: &PlanView,
    probes: &[Probe],
    remap: Option<&[u32]>,
    last: usize,
    clip: bool,
    tolerance: f32,
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
                sampling: a::Sampling::Explicit,
                unresolved: None,
                field_wavelength: metadata.clone(),
                field_key: None,
                tolerance,
                max_iterations: 60,
                output: generated.clone(),
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
/// Unqualified numeric output: the owning evaluator independently checks every
/// candidate against f64 before publishing it. P4 status is also consumed here.
pub(crate) fn evaluate(
    g: &GpuContext,
    cpu: &TraceEvaluator,
    tracers: &[Arc<SequentialTracer>],
    operands: &[&Operand],
    template: Option<(&systems::UploadedSystem, &[systems::Candidate])>,
    ctx: &JobContext,
) -> MkResult<Vec<Vec<OperandValue>>> {
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
    if tracers.is_empty() {
        return Ok(vec![]);
    }
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
    let mut uploaded = None;
    let (set, map) = if let Some((template, rows)) = template {
        let prep = template.append(g.backend.as_ref(), &mut p, rows, ctx)?;
        (prep.system_set(), Some(prep.map.status))
    } else {
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
        uploaded = Some(Uploaded(g.backend.as_ref(), h));
        (SystemSet::Uploaded(h), None)
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
                probes.push(Probe {
                    candidate: 0,
                    slot,
                    field: f,
                    wavelength: w,
                    pupil: [0., 0.],
                    dest: Dest::Illumination,
                });
                for j in 0..na {
                    let (sn, cs) = (dt * j as f64).sin_cos();
                    for k in 1..=nr {
                        let r = k as f64 / nr as f64;
                        probes.push(Probe {
                            candidate: 0,
                            slot,
                            field: f,
                            wavelength: w,
                            pupil: [r * cs, r * sn],
                            dest: Dest::Illumination,
                        });
                    }
                }
            }
            for (gi, (&key, sampling)) in plan.groups.iter().enumerate() {
                ctx.check()?;
                let (nodes, symmetry) =
                    c.group_nodes_from(key.0, sampling, g.pupil_nodes(sampling));
                let slot = slot(key.0, key.1);
                local
                    .entry((clip, last))
                    .or_default()
                    .extend(nodes.iter().enumerate().map(|(i, node)| Probe {
                        candidate: 0,
                        slot,
                        field: key.0,
                        wavelength: key.1,
                        pupil: [node.px, node.py],
                        dest: Dest::Node(gi as u32, i as u32),
                    }));
                let failed = if aggregates {
                    Vec::new()
                } else {
                    vec![NodeRay::FAILED; nodes.len()]
                };
                c.groups
                    .insert(key, Candidate::group_rays(nodes, symmetry, failed));
            }
            for &(f, w, px, py) in &plan.singles {
                local.entry((clip, last)).or_default().push(Probe {
                    candidate: 0,
                    slot: slot(f, w),
                    field: f,
                    wavelength: w,
                    pupil: [f64::from_bits(px), f64::from_bits(py)],
                    dest: Dest::Single,
                });
                c.singles.insert((f, w, px, py), NodeRay::FAILED);
            }
            for &(f, w, px, py, j) in &plan.real {
                if j == 0 || tr.system().surfaces[j].is_passive() {
                    return Err(MkError::Unsupported(
                        "real ray at passive/start surface".into(),
                    ));
                }
                local.entry((clip, j)).or_default().push(Probe {
                    candidate: 0,
                    slot: slot(f, w),
                    field: f,
                    wavelength: w,
                    pupil: [f64::from_bits(px), f64::from_bits(py)],
                    dest: Dest::Real(j as u32),
                });
            }
            Ok(((c, plan), all, local))
        })
        .collect::<MkResult<Vec<_>>>()?;
    let mut candidates = Vec::with_capacity(tracers.len());
    let mut chiefs = Vec::new();
    let mut batches: BTreeMap<(bool, usize), Vec<Probe>> = BTreeMap::new();
    // Rows in (candidate, field, λ) order, which is also the slot order.
    let mut field_keys = Vec::new();
    let mut refs = Vec::new();
    for (ci, (candidate, all, local)) in prepared.into_iter().enumerate() {
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
    let ch = append(
        g, &mut p, &set, &metadata, &chiefs, None, last, false, tolerance, ctx,
    )?;
    let reference = g.df(&mut p, [chiefs.len(), 3], &refs)?;
    p.outputs = vec![ch.rays.floats, ch.status, ch.rays.report];
    let mut outputs = Vec::new();
    let mut moment_segments = Vec::new();
    let mut fit_inputs = Vec::new();
    for (key, probes) in &partitions {
        let (clip, last) = *key;
        let selected: std::collections::BTreeSet<_> = probes.iter().map(|q| q.candidate).collect();
        let (aim_metadata, remap) = if selected.len() == tracers.len() {
            (metadata.clone(), None)
        } else {
            let mut remap = vec![u32::MAX; field_keys.len()];
            let mut subset = Vec::new();
            for (i, k) in field_keys.iter().enumerate() {
                if selected.contains(&k.0) {
                    remap[i] = subset.len() as u32;
                    subset.push(*k);
                }
            }
            (
                records(g, &mut p, subset.len(), |i| field_spec(subset[i]))?,
                Some(remap),
            )
        };
        let h = append(
            g,
            &mut p,
            &set,
            &aim_metadata,
            probes,
            remap.as_deref(),
            last,
            clip,
            tolerance,
            ctx,
        )?;
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
            outputs.push(probes);
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
        outputs.push(probes);
    }
    let mut fits = Vec::new();
    if operands.iter().any(|op| {
        matches!(
            op.kind,
            OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. }
        )
    }) {
        for (probes, opd, mask) in fit_inputs {
            fits.extend(fits::append(g, &mut p, &candidates, probes, opd, mask)?);
        }
    }
    if let Some(status) = map {
        p.outputs.push(status);
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
    let result = g.submit(p, ctx)?;
    g.record_traced_rays(traced_rays);
    let assembly = std::time::Instant::now();
    let mut unresolved = std::collections::BTreeSet::new();
    if map.is_some() {
        let v = g.backend.read_view(result.outputs.last().unwrap().view())?;
        for (ci, status) in v.as_slice::<u32>()?.as_chunks::<2>().0.iter().enumerate() {
            if status[0] != 0 {
                unresolved.insert(ci);
            }
        }
    }
    let views = result
        .outputs
        .iter()
        .map(|o| g.backend.read_view(o.view()))
        .collect::<MkResult<Vec<_>>>()?;
    let f = views[0].as_slice::<f32>()?;
    let status = views[1].as_slice::<u32>()?;
    for (q, &status) in chiefs.iter().zip(views[2].as_slice::<u32>()?) {
        if status != 0 {
            unresolved.insert(q.candidate);
        }
    }
    for (i, q) in chiefs.iter().enumerate() {
        let chief = trace_result(f, status, chiefs.len(), i);
        let reference = chief
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|r| gpu_reference(&tracers[q.candidate], r, q.wavelength));
        candidates[q.candidate]
            .0
            .chiefs
            .insert((q.field, q.wavelength), ChiefData { chief, reference });
    }
    for (bi, probes) in outputs.iter().enumerate() {
        let offset = 3 + 7 * bi;
        let f = views[offset].as_slice::<f32>()?;
        let status = views[offset + 1].as_slice::<u32>()?;
        for (q, &status) in probes.iter().zip(views[offset + 2].as_slice::<u32>()?) {
            if status != 0 {
                unresolved.insert(q.candidate);
            }
        }
        if aggregates {
            continue;
        }
        let opd = if wavefront {
            Some(views[offset + 3].as_slice::<Df32>()?)
        } else {
            None
        };
        let valid = views[offset + 4].as_slice::<u32>()?;
        let mut group = None;
        for (i, q) in probes.iter().enumerate() {
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
    for (bi, segments) in moment_segments.iter().enumerate() {
        if segments.is_empty() {
            continue;
        }
        let moments = views[3 + 7 * bi + 5].as_slice::<Df32>()?;
        for (si, key) in segments.iter().enumerate() {
            if let &Some((ci, group)) = key {
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
    let values = candidates
        .par_iter()
        .enumerate()
        .map(|(ci, (c, _))| {
            if unresolved.contains(&ci) {
                return vec![
                    Err(OperandError::RayFailure(
                        "resident candidate preparation or aiming requires reference".into()
                    ));
                    operands.len()
                ];
            }
            let mut memo = GroupMemo::default();
            operands
                .iter()
                .map(|op| c.value_with(op, &mut memo))
                .collect()
        })
        .collect();
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
