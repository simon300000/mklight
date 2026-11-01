use super::transport::history;
use super::*;
use crate::{
    path::Signature,
    tally::{ChunkTally, SignatureAcc},
    tracer::StraySettings,
};
use mklight_gpu::{
    ReduceMode,
    collectives::{self, SegmentKind, SegmentReduce},
};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};

pub(super) fn coverage(
    b: &dyn ComputeBackend,
    s: &StraySettings,
    ctx: &JobContext,
) -> MkResult<BufferHandle> {
    let paths = s.covered_ghosts.iter().flatten().collect::<Vec<_>>();
    if paths.iter().any(|p| p.diffraction.len() > 32) {
        return Err(MkError::InvalidModel(
            "coverage exceeds 32 DOE visits".into(),
        ));
    }
    ops::fill_records(b, paths.len(), ctx, |i| {
        let path = paths[i];
        let mut h = ops::Query::default().history;
        h.length = path.reflections.len() as u32;
        for (a, b) in h.events.iter_mut().zip(&path.reflections) {
            *a = *b;
        }
        h.order_count = path.diffraction.len() as u32;
        for (a, b) in h.orders.iter_mut().zip(&path.diffraction) {
            *a = [b.surface | (b.visit << 16), b.order as u32];
        }
        h
    })
}
pub(super) fn params(scene: &Scene, s: &StraySettings, n: usize, cap: usize) -> ops::TallyParams {
    ops::TallyParams {
        half: [scene.detector.half_x, scene.detector.half_y].map(Df32::from_f64),
        nx: scene.detector.nx,
        ny: scene.detector.ny,
        owned: s.owned_classes.0 as u32,
        covered: s.covered_ghosts.is_some() as u32,
        count: (n * cap) as u32,
        capacity: cap as u32,
        sources: n as u32,
        queries: 0,
    }
}
pub(super) fn tags(
    plan: &mut ResidentPlan,
    hits: BufferId,
    ledger: BufferId,
    coverage: &BufferHandle,
    p: ops::TallyParams,
) -> MkResult<BufferId> {
    let coverage = plan.import(coverage.view());
    let output = plan.scratch(BufferDesc::new(ElementType::U32, [p.count as usize, 4]));
    plan.operations
        .push(ResidentOp::Stray(ops::Request::Tally(ops::Tally::Tags {
            hits: plan.view(hits)?,
            ledger: plan.view(ledger)?,
            coverage: plan.view(coverage)?,
            params: p,
            output,
        })));
    Ok(output)
}
fn query(kind: u32, slot: u32, key: u32, other: u32) -> ops::Query {
    ops::Query {
        kind,
        slot,
        key,
        other,
        ..Default::default()
    }
}
/// Fixed 256-source logical leaves. Dispatch may batch any number of leaves;
/// publication folds these leaves in source-ID order, independently of tiling.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect(
    b: &dyn ComputeBackend,
    snapshot: &Snapshot,
    scene: &Scene,
    s: &StraySettings,
    views: [BufferView; 3],
    n: usize,
    cap: usize,
    ctx: &JobContext,
    stats: &mut Vec<PlanStats>,
) -> MkResult<Vec<ChunkTally>> {
    let hv = b.read_view(views[0].clone())?;
    let lv = b.read_view(views[1].clone())?;
    let tv = b.read_view(views[2].clone())?;
    let hits = ops::read_records::<ops::Detection>(&hv)?;
    let ledgers = ops::read_records::<ops::Ledger>(&lv)?;
    let tags = tv.as_slice::<u32>()?;
    // Enumerate only integer topology/labels. Flux, positions, and all estimator
    // arithmetic remain device-side. Keys carry complete signatures, never hashes.
    // Source blocks are scanned in parallel and merged in block order.
    type SigKey = (u32, Signature);
    let parts = ledgers
        .par_chunks(256)
        .enumerate()
        .map(|(block, ledgers)| {
            let mut maps = Vec::new();
            let mut local = HashMap::<SigKey, u32>::new();
            let mut sigs = Vec::new();
            let mut members = Vec::new();
            let mut pairs = Vec::new();
            for (i, l) in ledgers.iter().enumerate() {
                let i = block * 256 + i;
                for k in i * cap..i * cap + l.hits as usize {
                    let h = &hits[k];
                    let tag = &tags[4 * k..4 * k + 4];
                    let member = |slot| ((slot, tag[0]), i as u32, k as u32);
                    if tag[0] != u32::MAX {
                        maps.push(member(h.class));
                        if tag[2] != 0 {
                            maps.push(member(5));
                        }
                        if tag[3] != 0 {
                            maps.push(member(6));
                        }
                    }
                    if s.signatures {
                        let key = (h.class, history(&h.history).signature);
                        let id = *local.entry(key).or_insert_with(|| {
                            sigs.push((key, h.history));
                            sigs.len() as u32 - 1
                        });
                        members.push((id, i as u32, k as u32));
                    }
                    if h.class == 1 {
                        pairs.push((h.history.first[0], h.history.first[1]));
                    }
                }
            }
            (maps, sigs, members, pairs)
        })
        .collect::<Vec<_>>();
    let mut maps = Vec::new();
    let mut signatures = HashMap::<SigKey, (u32, ops::History)>::new();
    let mut sig_members = Vec::new();
    let mut pairs = BTreeSet::new();
    for (m, sigs, members, p) in parts {
        maps.extend(m);
        let ids = sigs
            .into_iter()
            .map(|(key, history)| {
                let next = signatures.len() as u32;
                signatures.entry(key).or_insert((next, history)).0
            })
            .collect::<Vec<_>>();
        sig_members.extend(
            members
                .into_iter()
                .map(|(id, i, k)| (ids[id as usize], i, k)),
        );
        pairs.extend(p);
    }
    let mut queries = (0..9).map(|slot| query(0, slot, 0, 0)).collect::<Vec<_>>();
    queries.extend((0..23).map(|key| query(3, 0, key, 0)));
    queries.extend((0..scene.num_wavelengths() as u32).map(|wl| query(3, 0, 23, wl)));
    queries.extend((0..5).map(|c| query(7, c, 0, 0)));
    for wl in 0..scene.num_wavelengths() as u32 {
        queries.extend((0..5).map(|c| query(6, c, wl, 0)));
    }
    queries.extend((0..9).map(|order| query(4, 0, order, 0)));
    queries.extend(pairs.into_iter().map(|(a, b)| query(5, 0, a, b)));
    // Bins: sorted map keys, then sorted signatures. Members are ordered by
    // (bin, 256-source block, source, record), which fixes every reduction.
    let mut keys = maps.iter().map(|m| m.0).collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();
    let mut bins = keys
        .iter()
        .map(|&(slot, pixel)| query(1, slot, pixel, 0))
        .collect::<Vec<_>>();
    let mut sigs = signatures.into_iter().collect::<Vec<_>>();
    sigs.sort_unstable_by_key(|s| s.0);
    let mut sig_ids = vec![0; sigs.len()];
    for (sorted, (_, (id, _))) in sigs.iter().enumerate() {
        sig_ids[*id as usize] = (bins.len() + sorted) as u32;
    }
    bins.extend(sigs.into_iter().map(|((c, _), (_, history))| ops::Query {
        history,
        ..query(2, c, 0, 0)
    }));
    let mut groups = maps
        .into_iter()
        .map(|(key, i, k)| (keys.binary_search(&key).unwrap() as u32, i / 256, i, k))
        .chain(
            sig_members
                .into_iter()
                .map(|(id, i, k)| (sig_ids[id as usize], i / 256, i, k)),
        )
        .collect::<Vec<_>>();
    groups.par_sort_unstable();
    drop(hv);
    drop(lv);
    drop(tv);
    let blocks = n.div_ceil(256);
    let mut chunks = (0..blocks)
        .map(|i| {
            let mut c = ChunkTally::new(scene.num_wavelengths(), s.owned_classes);
            c.rays = (n - i * 256).min(256) as u64;
            c.track_signatures = s.signatures;
            c
        })
        .collect::<Vec<_>>();
    // The 9 moment slots keep all 14 source products; every other query needs
    // only (sum, sum²). Every source's entire split family is combined in
    // depth-first order on device before P3 forms source products. One plan
    // normally carries every estimator; batches bound column scratch.
    let mut batches = vec![(0..9, 14)];
    let per = ((8 << 20) / (2 * n)).max(1);
    batches.extend(
        (9..queries.len())
            .step_by(per)
            .map(|a| (a..(a + per).min(queries.len()), 2)),
    );
    let offsets = b.fill_buffer(
        BufferDesc::new(ElementType::U32, [blocks + 1]),
        &mut |mut out| {
            for (i, v) in out.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                *v = (i * 256).min(n) as u32;
            }
            Ok(())
        },
    )?;
    let mut groups = (!bins.is_empty()).then_some(&groups[..]);
    let mut next = 0;
    while next < batches.len() {
        ctx.check()?;
        let mut plan = ResidentPlan::new(snapshot.clone());
        let imports = views.clone().map(|v| plan.import(v));
        let offsets = plan.import(offsets.view());
        let (first, mut columns) = (next, 0);
        let mut outputs = Vec::new();
        while next < batches.len()
            && (next == first || columns + batches[next].0.len() * batches[next].1 <= 2 * per)
        {
            let (range, width) = batches[next].clone();
            columns += range.len() * width;
            let qs = &queries[range];
            let qh = ops::fill_records(b, qs.len(), ctx, |i| qs[i])?;
            let qid = plan.import(qh.view());
            let mut p = params(scene, s, n, cap);
            p.queries = qs.len() as u32;
            let terms = plan.scratch(BufferDesc::new(
                ElementType::DoubleFloat,
                [width * qs.len(), n],
            ));
            plan.operations
                .push(ResidentOp::Stray(ops::Request::Tally(ops::Tally::Terms {
                    width: width as u32,
                    hits: plan.view(imports[0])?,
                    ledger: plan.view(imports[1])?,
                    tags: plan.view(imports[2])?,
                    queries: plan.view(qid)?,
                    params: p,
                    output: terms,
                })));
            let totals = plan.scratch(BufferDesc::new(
                ElementType::DoubleFloat,
                [width * qs.len(), blocks],
            ));
            let status = plan.scratch(BufferDesc::new(ElementType::U32, [blocks]));
            plan.operations.push(ResidentOp::Collectives(
                collectives::Request::SegmentReduce(SegmentReduce {
                    values: plan.view(terms)?,
                    offsets: plan.view(offsets)?,
                    mask: None,
                    mask_value: 0,
                    weights: None,
                    max_segment_len: n.min(256),
                    reference: None,
                    mode: ReduceMode::DoubleFloat,
                    kind: SegmentKind::Sum,
                    output: totals,
                    status,
                }),
            ));
            outputs.push(totals);
            next += 1;
        }
        let grouped = match groups.take() {
            Some(members) => Some((
                append_groups(
                    b,
                    &mut plan,
                    imports[0],
                    bins.len(),
                    blocks,
                    members,
                    cap,
                    ctx,
                )?,
                members,
            )),
            None => None,
        };
        plan.outputs = outputs.clone();
        plan.outputs.extend(grouped.as_ref().map(|g| g.0.0));
        let result = b.submit_plan(plan, ctx)?.wait()?;
        for (batch, output) in batches[first..next].iter().zip(&result.outputs) {
            let view = b.read_view(output.view())?;
            let values = view.as_slice::<Df32>()?;
            let width = batch.1;
            for (j, q) in queries[batch.0.clone()].iter().enumerate() {
                for (k, chunk) in chunks.iter_mut().enumerate() {
                    let v = std::array::from_fn::<_, 14, _>(|c| {
                        if c < width {
                            values[(j * width + c) * blocks + k].to_f64()
                        } else {
                            0.
                        }
                    });
                    if v.iter().any(|v| !v.is_finite()) {
                        return Err(unsupported("stray estimator overflow"));
                    }
                    apply(q, chunk, &v, if width == 14 { 4 } else { 1 });
                }
            }
        }
        if let Some(((_, starts), members)) = grouped {
            let view = b.read_view(result.outputs[outputs.len()].view())?;
            apply_groups(
                view.as_slice::<Df32>()?,
                &bins,
                members,
                &starts,
                &mut chunks,
            )?;
        }
        stats.push(result.stats);
    }
    Ok(chunks)
}

/// Folds one query's per-block (sum, sum²) or full moment row into `chunk`.
fn apply(q: &ops::Query, chunk: &mut ChunkTally, v: &[f64; 14], sq: usize) {
    let (sum, sq) = (v[0], v[sq]);
    match q.kind {
        0 => {
            let slot = q.slot as usize;
            if slot < 7 {
                chunk.moments[slot].copy_from_slice(&v[..4]);
                chunk.moments_q[slot].copy_from_slice(&v[4..]);
            }
            match slot {
                0..=4 => {
                    chunk.detected[slot] = sum;
                    chunk.detected_sq[slot] = sq;
                }
                5 => chunk.owned_sq = sq,
                6 => {
                    chunk.detected_uncovered = sum;
                    chunk.uncovered_sq = sq;
                }
                7 => chunk.total_sq = sq,
                8 => {
                    chunk.detected_covered = sum;
                    chunk.covered_sq = sq;
                }
                _ => unreachable!(),
            };
        }
        3 => match q.key {
            0..=11 => chunk.absorbed[q.key as usize] = sum,
            12 => chunk.escaped = sum,
            13 => chunk.escaped_from_solid = sum,
            14 => chunk.truncated = sum,
            15 => chunk.sampling_failed = sum,
            16 => chunk.roulette_gain = sum,
            17 => chunk.roulette_loss = sum,
            18 => chunk.weighting_gain = sum,
            19 => chunk.weighting_loss = sum,
            20 => chunk.emitted = sum,
            21 => chunk.adjustment_sq = sq,
            22 => chunk.events = sum as u64,
            23 => chunk.emitted_by_wavelength[q.other as usize] = sum,
            _ => unreachable!(),
        },
        4 => chunk.ghost_by_order[q.key as usize] = sum,
        5 => {
            chunk
                .ghost_by_pair
                .insert((q.key as u16, q.other as u16), sum);
        }
        6 => chunk.detected_by_wavelength[q.key as usize][q.slot as usize] = sum,
        7 => chunk.detected_hits[q.slot as usize] = sum as u64,
        _ => unreachable!(),
    }
}

/// Sparse grouping uses only integer bin/signature/source IDs. P3 sums the
/// original device weights within each source, forms products, then reduces
/// canonical source blocks. No per-bin scan of every ray, or float atomics.
/// `members` is sorted by (bin, block, source, record).
#[allow(clippy::too_many_arguments)]
fn append_groups(
    b: &dyn ComputeBackend,
    plan: &mut ResidentPlan,
    hits: BufferId,
    bins: usize,
    blocks: usize,
    members: &[(u32, u32, u32, u32)],
    cap: usize,
    ctx: &JobContext,
) -> MkResult<(BufferId, Vec<usize>)> {
    // One segment per (bin, block, source).
    let starts = (0..members.len())
        .filter(|&i| {
            let segment = |m: (u32, u32, u32, u32)| (m.0, m.1, m.2);
            i == 0 || segment(members[i]) != segment(members[i - 1])
        })
        .collect::<Vec<_>>();
    let count = members.len();
    let ns = starts.len();
    let nb = bins * blocks;
    let ids = ops::fill_records(b, count, ctx, |i| members[i].3)?;
    let ids = plan.import(ids.view());
    let values = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, count]));
    plan.operations
        .push(ResidentOp::Stray(ops::Request::Tally(ops::Tally::Gather {
            hits: plan.view(hits)?,
            indices: plan.view(ids)?,
            output: values,
        })));
    let source_offsets = b.fill_buffer(
        BufferDesc::new(ElementType::U32, [ns + 1]),
        &mut |mut out| {
            let a = out.as_mut_slice::<u32>()?;
            for (a, &start) in a.iter_mut().zip(&starts) {
                *a = start as u32;
            }
            a[ns] = count as u32;
            Ok(())
        },
    )?;
    let so = plan.import(source_offsets.view());
    let products = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [ns, 5]));
    let status = plan.scratch(BufferDesc::new(ElementType::U32, [ns]));
    plan.operations.push(ResidentOp::Collectives(
        collectives::Request::SegmentReduce(SegmentReduce {
            values: plan.view(values)?,
            offsets: plan.view(so)?,
            mask: None,
            mask_value: 0,
            weights: None,
            max_segment_len: cap.min(count),
            reference: None,
            mode: ReduceMode::DoubleFloat,
            kind: SegmentKind::SourceProducts,
            output: products,
            status,
        }),
    ));
    let columns = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, ns]));
    plan.operations.push(ResidentOp::Stray(ops::Request::Tally(
        ops::Tally::Transpose {
            input: plan.view(products)?,
            sources: ns,
            queries: 1,
            width: 5,
            output: columns,
        },
    )));
    let offsets = b.fill_buffer(
        BufferDesc::new(ElementType::U32, [nb + 1]),
        &mut |mut out| {
            let a = out.as_mut_slice::<u32>()?;
            let mut segments = starts
                .iter()
                .map(|&i| members[i].0 as usize * blocks + members[i].1 as usize)
                .peekable();
            let mut offset = 0;
            for (bucket, v) in a[..nb].iter_mut().enumerate() {
                *v = offset;
                while segments.next_if_eq(&bucket).is_some() {
                    offset += 1;
                }
            }
            a[nb] = offset;
            Ok(())
        },
    )?;
    let offsets = plan.import(offsets.view());
    let totals = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, nb]));
    let status = plan.scratch(BufferDesc::new(ElementType::U32, [nb]));
    plan.operations.push(ResidentOp::Collectives(
        collectives::Request::SegmentReduce(SegmentReduce {
            values: plan.view(columns)?,
            offsets: plan.view(offsets)?,
            mask: None,
            mask_value: 0,
            weights: None,
            max_segment_len: 256.min(ns),
            reference: None,
            mode: ReduceMode::DoubleFloat,
            kind: SegmentKind::Sum,
            output: totals,
            status,
        }),
    ));
    Ok((totals, starts))
}

fn apply_groups(
    values: &[Df32],
    bins: &[ops::Query],
    members: &[(u32, u32, u32, u32)],
    starts: &[usize],
    chunks: &mut [ChunkTally],
) -> MkResult<()> {
    let blocks = chunks.len();
    let nb = bins.len() * blocks;
    // Most (bin, block) buckets are empty; their zero sums change nothing.
    let mut occupied = vec![false; nb];
    for &i in starts {
        occupied[members[i].0 as usize * blocks + members[i].1 as usize] = true;
    }
    for (i, q) in bins.iter().enumerate() {
        for (k, chunk) in chunks.iter_mut().enumerate() {
            let index = i * blocks + k;
            if !occupied[index] {
                continue;
            }
            let (sum, sq) = (values[index].to_f64(), values[2 * nb + index].to_f64());
            if !(sum.is_finite() && sq.is_finite()) {
                return Err(unsupported("stray estimator overflow"));
            }
            if q.kind == 1 {
                let slot = q.slot as usize;
                // StrayTally copies selected class rows into the owned map itself.
                // The explicit owned row supplies only the uncovered contribution.
                if slot < 5 || slot == 6 {
                    chunk.hits.push((q.key, q.slot as u8, sum));
                }
                if slot == 6 {
                    chunk.hits.push((q.key, 5, sum));
                }
                chunk.hits_sq.push((q.key, q.slot as u8, sq));
            } else {
                chunk.signatures.insert(
                    (q.slot as u8, history(&q.history).signature),
                    SignatureAcc {
                        flux: sum,
                        flux_sq: sq,
                        hits: values[nb + index].to_f64() as u64,
                    },
                );
            }
        }
    }
    Ok(())
}
