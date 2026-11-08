use super::*;
use mklight_gpu::{collectives as col, optim_ops as op};
type GroupSlice = (usize, GroupKey, usize);
pub(super) struct Fit {
    groups: Vec<(usize, GroupKey)>,
    cols: usize,
    output: usize,
}
pub(super) fn append(
    g: &GpuContext,
    p: &mut ResidentPlan,
    candidates: &[(Candidate<'_>, super::super::super::trace_eval::Plan)],
    probes: &[Probe],
    opd: BufferId,
    mask: BufferId,
) -> MkResult<Vec<Fit>> {
    let mut batches: BTreeMap<(usize, usize), Vec<GroupSlice>> = BTreeMap::new();
    // A fit reads a group's nodes as contiguous probes: groups with nodes
    // routed to the host (fewer probes than nodes) are fitted on the host.
    let mut counts: BTreeMap<(usize, u32), usize> = BTreeMap::new();
    for q in probes {
        if let Dest::Node(group, _) = q.dest {
            *counts.entry((q.candidate, group)).or_default() += 1;
        }
    }
    for (i, q) in probes.iter().enumerate() {
        if let Dest::Node(group, 0) = q.dest {
            let key = *candidates[q.candidate]
                .1
                .groups
                .keys()
                .nth(group as usize)
                .unwrap();
            let nodes = &candidates[q.candidate].0.groups[&key];
            if counts[&(q.candidate, group)] != nodes.nodes.len() {
                continue;
            }
            let cols = if nodes.symmetry == crate::merit::PupilSymmetry::Bilateral {
                2
            } else {
                3
            };
            for cols in [1, cols] {
                batches
                    .entry((nodes.nodes.len(), cols))
                    .or_default()
                    .push((q.candidate, key, i));
            }
        }
    }
    let mut fits = Vec::new();
    for ((rows, cols), groups) in batches {
        if rows < cols {
            continue;
        }
        let b = groups.len();
        let indices = g.fill(p, [b, rows], |out: &mut [u32]| {
            for (batch, &(_, _, start)) in groups.iter().enumerate() {
                for row in 0..rows {
                    out[batch * rows + row] = (start + row) as u32;
                }
            }
        })?;
        let basis = g.fill(p, [b, rows, cols], |out: &mut [Df32]| {
            for (batch, &(ci, key, _)) in groups.iter().enumerate() {
                for (row, node) in candidates[ci].0.groups[&key].nodes.iter().enumerate() {
                    let w = node.weight.sqrt();
                    let start = (batch * rows + row) * cols;
                    out[start] = Df32::from_f64(w);
                    if cols == 2 {
                        out[start + 1] = Df32::from_f64(w * node.py);
                    } else if cols == 3 {
                        out[start + 1] = Df32::from_f64(w * node.px);
                        out[start + 2] = Df32::from_f64(w * node.py);
                    }
                }
            }
        })?;
        let matrix = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, rows, cols]));
        let rhs = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, rows]));
        p.operations.push(ResidentOp::Optim(op::Request::FitRows {
            values: p.view(opd)?,
            mask: p.view(mask)?,
            indices,
            basis,
            matrix,
            rhs,
        }));
        let factors = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [b, rows + cols + 2, cols],
        ));
        let solution = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, cols]));
        let rank = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b, 4]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::QrFit(col::QrFit {
                matrix: p.view(matrix)?,
                rhs: p.view(rhs)?,
                rank_tolerance: 1e-12,
                factors,
                solution,
                rank_condition: rank,
            })));
        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, b * rows]));
        p.operations.push(ResidentOp::Optim(op::Request::FitError {
            matrix: p.view(matrix)?,
            rhs: p.view(rhs)?,
            coefficients: p.view(solution)?,
            output: terms,
        }));
        let offsets = g.fill(p, [b + 1], |out: &mut [u32]| {
            for (i, o) in out.iter_mut().enumerate() {
                *o = (i * rows) as u32;
            }
        })?;
        let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, b]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [b]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::SegmentReduce(
                col::SegmentReduce {
                    values: p.view(terms)?,
                    offsets,
                    mask: None,
                    mask_value: 0,
                    weights: None,
                    max_segment_len: rows,
                    reference: None,
                    mode: mklight_gpu::ReduceMode::DoubleFloat,
                    kind: col::SegmentKind::Sum,
                    output: sums,
                    status,
                },
            )));
        let output = p.outputs.len();
        p.outputs.extend([solution, rank, sums]);
        fits.push(Fit {
            groups: groups.into_iter().map(|(ci, key, _)| (ci, key)).collect(),
            cols,
            output,
        });
    }
    Ok(fits)
}
pub(super) fn finish(
    fits: &[Fit],
    views: &[ReadView],
    candidates: &mut [(Candidate<'_>, super::super::super::trace_eval::Plan)],
) -> MkResult<()> {
    for fit in fits {
        let coefficients = views[fit.output].as_slice::<Df32>()?;
        let rank = views[fit.output + 1].as_slice::<Df32>()?;
        let sums = views[fit.output + 2].as_slice::<Df32>()?;
        for (i, &(ci, key)) in fit.groups.iter().enumerate() {
            let mut v = [f64::NAN; 4];
            if rank[4 * i + 3].to_f64() == 0. && sums[fit.groups.len() + i].to_f64() > 0. {
                v = [
                    coefficients[i * fit.cols].to_f64(),
                    0.,
                    0.,
                    sums[i].to_f64() / sums[fit.groups.len() + i].to_f64(),
                ];
                if fit.cols == 2 {
                    v[2] = coefficients[i * fit.cols + 1].to_f64();
                } else if fit.cols == 3 {
                    v[1] = coefficients[i * fit.cols + 1].to_f64();
                    v[2] = coefficients[i * fit.cols + 2].to_f64();
                }
            }
            candidates[ci]
                .0
                .resident_wavefits
                .entry(key)
                .or_insert([[f64::NAN; 4]; 2])[usize::from(fit.cols != 1)] = v;
        }
    }
    Ok(())
}
