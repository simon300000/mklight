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
use std::collections::{BTreeMap, BTreeSet};

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
    let mut maps = BTreeSet::new();
    let mut signatures = BTreeMap::<(u32, Signature), ops::History>::new();
    let mut pairs = BTreeSet::new();
    for (i, l) in ledgers.iter().enumerate() {
        for j in 0..l.hits as usize {
            let k = i * cap + j;
            let h = &hits[k];
            let tag = &tags[4 * k..4 * k + 4];
            if tag[0] != u32::MAX {
                maps.insert((h.class, tag[0]));
                if tag[2] != 0 {
                    maps.insert((5, tag[0]));
                }
                if tag[3] != 0 {
                    maps.insert((6, tag[0]));
                }
            }
            if s.signatures {
                signatures
                    .entry((h.class, history(&h.history).signature))
                    .or_insert(h.history);
            }
            if h.class == 1 {
                pairs.insert((h.history.first[0], h.history.first[1]));
            }
        }
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
    let mut bins = maps
        .into_iter()
        .map(|(slot, pixel)| query(1, slot, pixel, 0))
        .collect::<Vec<_>>();
    let map_ids = bins
        .iter()
        .enumerate()
        .map(|(i, q)| ((q.slot, q.key), i))
        .collect::<BTreeMap<_, _>>();
    let sig_ids = signatures
        .keys()
        .enumerate()
        .map(|(i, k)| (k.clone(), i + bins.len()))
        .collect::<BTreeMap<_, _>>();
    bins.extend(signatures.into_iter().map(|((c, _), history)| ops::Query {
        history,
        ..query(2, c, 0, 0)
    }));
    let mut groups = BTreeMap::<(usize, usize, usize), Vec<u32>>::new();
    for (i, l) in ledgers.iter().enumerate() {
        for j in 0..l.hits as usize {
            let k = i * cap + j;
            let h = &hits[k];
            let tag = &tags[4 * k..4 * k + 4];
            if tag[0] != u32::MAX {
                let mut slots = vec![h.class];
                if tag[2] != 0 {
                    slots.push(5);
                }
                if tag[3] != 0 {
                    slots.push(6);
                }
                for slot in slots {
                    let id = map_ids[&(slot, tag[0])];
                    groups.entry((id, i / 256, i)).or_default().push(k as u32);
                }
            }
            if s.signatures {
                let id = sig_ids[&(h.class, history(&h.history).signature)];
                groups.entry((id, i / 256, i)).or_default().push(k as u32);
            }
        }
    }
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
    // Query tiles bound resident scratch; every source's entire split family is
    // combined in depth-first order on device before P3 forms source products.
    for qs in queries.chunks(128) {
        ctx.check()?;
        let mut plan = ResidentPlan::new(snapshot.clone());
        let imports = views.clone().map(|v| plan.import(v));
        let qh = ops::fill_records(b, qs.len(), ctx, |i| qs[i])?;
        let qid = plan.import(qh.view());
        let qn = qs.len();
        let mut p = params(scene, s, n, cap);
        p.queries = qn as u32;
        // Fuse P3's one-record SourceProducts case into channel generation;
        // no reduction or transpose is needed after each source was combined.
        let columns = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [14 * qn, n]));
        plan.operations
            .push(ResidentOp::Stray(ops::Request::Tally(ops::Tally::Terms {
                products: true,
                hits: plan.view(imports[0])?,
                ledger: plan.view(imports[1])?,
                tags: plan.view(imports[2])?,
                queries: plan.view(qid)?,
                params: p,
                output: columns,
            })));
        let offsets = b.fill_buffer(
            BufferDesc::new(ElementType::U32, [blocks + 1]),
            &mut |mut out| {
                for (i, v) in out.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                    *v = (i * 256).min(n) as u32;
                }
                Ok(())
            },
        )?;
        let offsets = plan.import(offsets.view());
        let totals = plan.scratch(BufferDesc::new(ElementType::DoubleFloat, [14 * qn, blocks]));
        let status = plan.scratch(BufferDesc::new(ElementType::U32, [blocks]));
        plan.operations.push(ResidentOp::Collectives(
            collectives::Request::SegmentReduce(SegmentReduce {
                values: plan.view(columns)?,
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
        plan.outputs = vec![totals];
        let result = b.submit_plan(plan, ctx)?.wait()?;
        let view = b.read_view(result.outputs[0].view())?;
        let values = view.as_slice::<Df32>()?;
        for (j, q) in qs.iter().enumerate() {
            for (k, chunk) in chunks.iter_mut().enumerate() {
                let v =
                    std::array::from_fn::<_, 14, _>(|c| values[(j * 14 + c) * blocks + k].to_f64());
                if v.iter().any(|v| !v.is_finite()) {
                    return Err(unsupported("stray estimator overflow"));
                }
                let (sum, sq) = (v[0], v[4]);
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
                    1 => {
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
                    }
                    2 => {
                        chunk.signatures.insert(
                            (q.slot as u8, history(&q.history).signature),
                            SignatureAcc {
                                flux: sum,
                                flux_sq: sq,
                                hits: v[1] as u64,
                            },
                        );
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
        }
        stats.push(result.stats);
    }
    if !bins.is_empty() {
        collect_groups(
            b,
            snapshot,
            &views[0],
            &bins,
            &groups,
            &mut chunks,
            cap,
            ctx,
            stats,
        )?;
    }
    Ok(chunks)
}

/// Sparse grouping uses only integer bin/signature/source IDs. P3 sums the
/// original device weights within each source, forms products, then reduces
/// canonical source blocks. No per-bin scan of every ray, or float atomics.
fn collect_groups(
    b: &dyn ComputeBackend,
    snapshot: &Snapshot,
    hits: &BufferView,
    bins: &[ops::Query],
    groups: &BTreeMap<(usize, usize, usize), Vec<u32>>,
    chunks: &mut [ChunkTally],
    cap: usize,
    ctx: &JobContext,
    stats: &mut Vec<PlanStats>,
) -> MkResult<()> {
    let count = groups.values().map(Vec::len).sum::<usize>();
    let ns = groups.len();
    let nb = bins.len() * chunks.len();
    let mut plan = ResidentPlan::new(snapshot.clone());
    let hits = plan.import(hits.clone());
    let mut ids = groups.values().flatten();
    let ids = ops::fill_records(b, count, ctx, |_| *ids.next().unwrap())?;
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
            let mut offset = 0;
            for (i, g) in groups.values().enumerate() {
                a[i] = offset;
                offset += g.len() as u32;
            }
            a[ns] = offset;
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
            let mut keys = groups.keys().peekable();
            let mut offset = 0;
            for (bucket, v) in a[..nb].iter_mut().enumerate() {
                *v = offset;
                while keys
                    .peek()
                    .is_some_and(|(bin, block, _)| bin * chunks.len() + block == bucket)
                {
                    offset += 1;
                    keys.next();
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
    plan.outputs = vec![totals];
    let result = b.submit_plan(plan, ctx)?.wait()?;
    let view = b.read_view(result.outputs[0].view())?;
    let values = view.as_slice::<Df32>()?;
    let blocks = chunks.len();
    for (i, q) in bins.iter().enumerate() {
        for (k, chunk) in chunks.iter_mut().enumerate() {
            let index = i * blocks + k;
            let (sum, sq) = (values[index].to_f64(), values[2 * nb + index].to_f64());
            if !(sum.is_finite() && sq.is_finite()) {
                return Err(unsupported("stray estimator overflow"));
            }
            if q.kind == 1 {
                let slot = q.slot as usize;
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
    stats.push(result.stats);
    Ok(())
}
