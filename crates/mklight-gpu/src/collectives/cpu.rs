use super::*;
use crate::{
    Df32 as D,
    fenv::{FpEnv, Ftz, Ieee},
    pending::JobControl,
    reduce::{self, Accumulator as Acc},
    resident::cpu::Execution,
};

/// A typed input view borrowed in place.
#[derive(Clone, Copy)]
struct Values<'a> {
    words: &'a [u32],
    ty: Ty,
}
impl<'a> Values<'a> {
    fn new(x: &'a Execution, v: &PlanView) -> Self {
        Self {
            words: x.view(v),
            ty: v.desc.element,
        }
    }
    fn d(&self, i: usize) -> D {
        match self.ty {
            Ty::DoubleFloat => D::new(
                f32::from_bits(self.words[2 * i]),
                f32::from_bits(self.words[2 * i + 1]),
            ),
            _ => D::from_f32(f32::from_bits(self.words[i])),
        }
    }
}
fn put(x: &mut Execution, id: BufferId, offset: usize, v: &[D]) {
    for (dest, q) in x.words_mut(id)[2 * offset..]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(v)
    {
        *dest = [q.hi.to_bits(), q.lo.to_bits()];
    }
}
pub fn execute(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    if x.flush_to_zero {
        execute_in::<Ftz>(r, x, ctl)
    } else {
        execute_in::<Ieee>(r, x, ctl)
    }
}
pub fn execute_in<E: FpEnv>(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    match r {
        Request::SegmentReduce(r) => {
            let (out, status) = segments::<E>(r, x, ctl)?;
            put(x, r.output, 0, &out);
            x.words_mut(r.status).copy_from_slice(&status);
        }
        Request::Scan(r) => {
            let a = Values::new(x, &r.values);
            if a.ty == Ty::U32 {
                let mut v = spec::scan_u32(a.words, ctl)?;
                if !r.inclusive && !v.is_empty() {
                    v.rotate_right(1);
                    v[0] = 0;
                }
                x.words_mut(r.output).copy_from_slice(&v);
            } else {
                let vals: Vec<_> = (0..r.values.desc.len()?).map(|i| a.d(i)).collect();
                let mut v = spec::scan_df::<E>(&vals, ctl)?;
                if !r.inclusive && !v.is_empty() {
                    v.rotate_right(1);
                    v[0] = D::ZERO;
                }
                put(x, r.output, 0, &v);
            }
        }
        Request::StableCompact(r) => {
            let (a, ids, mask) = (
                Values::new(x, &r.values),
                Values::new(x, &r.ids),
                Values::new(x, &r.mask),
            );
            let n = ids.words.len();
            let width = a.words.len().checked_div(n).unwrap_or(0);
            let mut out = vec![0; a.words.len()];
            let mut out_ids = vec![u32::MAX; n];
            let mut kept = 0;
            for i in (0..n).filter(|&i| mask.words[i] != 0) {
                out[kept * width..(kept + 1) * width]
                    .copy_from_slice(&a.words[i * width..(i + 1) * width]);
                out_ids[kept] = ids.words[i];
                kept += 1;
            }
            x.words_mut(r.output).copy_from_slice(&out);
            x.words_mut(r.output_ids).copy_from_slice(&out_ids);
            x.words_mut(r.count)[0] = kept as u32;
        }
        Request::StableSort(r) => {
            let (k, ids) = (Values::new(x, &r.keys), Values::new(x, &r.ids));
            let vals = r.values.as_ref().map(|v| Values::new(x, v));
            let n = ids.words.len();
            let order = spec::permutation(
                n,
                |a, b| {
                    let cmp = if k.ty == Ty::U32 {
                        k.words[a].cmp(&k.words[b])
                    } else {
                        crate::df::df_cmp(k.d(a), k.d(b))
                    };
                    cmp.then(ids.words[a].cmp(&ids.words[b])).then(a.cmp(&b))
                },
                ctl,
            )?;
            let gather = |v: &Values| {
                let w = v.words.len() / n.max(1);
                order
                    .iter()
                    .flat_map(|&i| &v.words[i * w..(i + 1) * w])
                    .copied()
                    .collect::<Vec<_>>()
            };
            let (keys, out_ids, values) = (gather(&k), gather(&ids), vals.as_ref().map(gather));
            x.words_mut(r.output_keys).copy_from_slice(&keys);
            x.words_mut(r.output_ids).copy_from_slice(&out_ids);
            if let (Some(v), Some(o)) = (values, r.output_values) {
                x.words_mut(o).copy_from_slice(&v);
            }
        }
        Request::QrFit(r) => {
            let [batch, m, n] = r.matrix.desc.shape[..] else {
                unreachable!()
            };
            for z in 0..batch {
                let (a, b) = (Values::new(x, &r.matrix), Values::new(x, &r.rhs));
                let aa: Vec<_> = (0..m * n).map(|i| a.d(z * m * n + i)).collect();
                let bb: Vec<_> = (0..m).map(|i| b.d(z * m + i)).collect();
                let sub = ctl.subrange(z as f64 / batch as f64, (z + 1) as f64 / batch as f64);
                let fit = spec::qr::<E>(&aa, &bb, m, n, r.rank_tolerance, &sub)?;
                put(x, r.factors, z * (m + n + 2) * n, &fit.factors);
                put(x, r.solution, z * n, &fit.solution);
                put(x, r.rank_condition, z * 4, &fit.diagnostics);
            }
        }
    }
    ctl.check()?;
    ctl.report(1.0, "collectives");
    Ok(())
}

/// Terms per phase, in output order (see `SegmentKind`).
fn terms(r: &SegmentReduce, c: usize, phase: usize) -> usize {
    match r.kind {
        SegmentKind::Moments2 => [4, 6][phase],
        SegmentKind::Summary => 4 * c,
        _ => c,
    }
}
fn op(r: &SegmentReduce, phase: usize, term: usize) -> Acc {
    match (r.kind, phase, term) {
        (SegmentKind::Summary, _, t) if t % 4 == 1 => Acc::Sum(crate::ReduceMode::DoubleFloat),
        (SegmentKind::Summary, _, t) if t % 4 == 2 => Acc::Min,
        (SegmentKind::Summary, _, t) if t % 4 == 3 => Acc::Max,
        (SegmentKind::Moments2, 0, 0) => Acc::Sum(crate::ReduceMode::DoubleFloat),
        (SegmentKind::Moments2, 1, t) if t >= 4 => Acc::Max,
        _ => Acc::Sum(r.mode),
    }
}
fn identity(r: &SegmentReduce, op: Acc) -> D {
    match op {
        Acc::Min => D::from_f32(f32::INFINITY),
        Acc::Max if r.kind == SegmentKind::Summary => D::from_f32(f32::NEG_INFINITY),
        _ => D::ZERO,
    }
}
fn slot(r: &SegmentReduce, c: usize, s: usize, seg: usize, phase: usize, term: usize) -> usize {
    match r.kind {
        SegmentKind::Sum => term * s + seg,
        SegmentKind::Summary => ((term / 4) * s + seg) * 4 + term % 4,
        SegmentKind::Moments2 => seg * 10 + phase * 4 + term,
        SegmentKind::SourceProducts => seg * (c + c * (c + 1) / 2) + term,
    }
}
fn segments<E: FpEnv>(
    r: &SegmentReduce,
    x: &Execution,
    ctl: &JobControl,
) -> MkResult<(Vec<D>, Vec<u32>)> {
    let vals = Values::new(x, &r.values);
    let offsets = Values::new(x, &r.offsets).words;
    let mask = r.mask.as_ref().map(|v| Values::new(x, v));
    let weights = r.weights.as_ref().map(|v| Values::new(x, v));
    let refs = r.reference.as_ref().map(|v| Values::new(x, v));
    let [c, n] = r.values.desc.shape[..] else {
        unreachable!()
    };
    let s = offsets.len() - 1;
    let phases = if r.kind == SegmentKind::Moments2 {
        2
    } else {
        1
    };
    let mut out = vec![D::ZERO; segment_shape(r).iter().product()];
    let mut status = vec![0; s];
    for z in 0..s {
        ctl.check()?;
        let (a, b) = (offsets[z] as usize, offsets[z + 1] as usize);
        if !(a <= b && b <= n && b - a <= r.max_segment_len) {
            status[z] = 1;
            let products = if r.kind == SegmentKind::SourceProducts {
                c * (c + 1) / 2
            } else {
                0
            };
            for phase in 0..phases {
                for t in 0..terms(r, c, phase) + products {
                    out[slot(r, c, s, z, phase, t)] = D::from_f32(f32::NAN);
                }
            }
            continue;
        }
        let keep = |i: usize| mask.is_none_or(|m| m.words[i] == r.mask_value);
        let load = |ch: usize, i: usize| vals.d(ch * n + i);
        let weight = |i: usize| match weights {
            Some(w) => w.d(i),
            None if c == 3 => load(2, i),
            None => D::from_f32(1.0),
        };
        let coordinate = |ch: usize, i: usize| match refs {
            Some(v) => load(ch, i).add_in::<E>(-v.d(2 * z + ch)),
            None => load(ch, i),
        };
        let mut mean = [D::ZERO; 2];
        for phase in 0..phases {
            for term in 0..terms(r, c, phase) {
                let op = op(r, phase, term);
                let id = identity(r, op);
                // Term value of record i, mirroring mkl_c_seg_value.
                let value = |i: usize| -> D {
                    if !keep(i) {
                        return id;
                    }
                    match r.kind {
                        SegmentKind::Summary if term % 4 == 1 => D::from_f32(1.0),
                        SegmentKind::Summary => load(term / 4, i),
                        SegmentKind::Sum | SegmentKind::SourceProducts => load(term, i),
                        SegmentKind::Moments2 => {
                            let w = weight(i);
                            if phase == 0 && term < 2 {
                                return if term == 0 { D::from_f32(1.0) } else { w };
                            }
                            let (mut px, mut py) = (coordinate(0, i), coordinate(1, i));
                            if phase == 0 {
                                return w.mul_in::<E>(if term == 2 { px } else { py });
                            }
                            let r2 = |a: D, b: D| a.mul_in::<E>(a).add_in::<E>(b.mul_in::<E>(b));
                            let positive = |v: D| if w.hi > 0.0 { v } else { D::ZERO };
                            match term {
                                0 => return w.mul_in::<E>(r2(px, py)),
                                4 => return positive(r2(px, py)),
                                _ => (),
                            }
                            px = px.add_in::<E>(-mean[0]);
                            py = py.add_in::<E>(-mean[1]);
                            match term {
                                5 => positive(r2(px, py)),
                                1 => w.mul_in::<E>(px.mul_in::<E>(px)),
                                2 => w.mul_in::<E>(px.mul_in::<E>(py)),
                                _ => w.mul_in::<E>(py.mul_in::<E>(py)),
                            }
                        }
                    }
                };
                out[slot(r, c, s, z, phase, term)] =
                    reduce::tree_in::<E>(op, id, b - a, |i| value(a + i), ctl)?;
            }
            match r.kind {
                SegmentKind::Moments2 if phase == 0 => {
                    let w = out[z * 10 + 1];
                    for k in 0..2 {
                        mean[k] = if w.hi > 0.0 {
                            out[z * 10 + 2 + k].div_in::<E>(w)
                        } else {
                            D::from_f32(f32::NAN)
                        };
                        out[z * 10 + 2 + k] = mean[k];
                    }
                }
                SegmentKind::SourceProducts => {
                    let base = slot(r, c, s, z, 0, 0);
                    let mut dest = base + c;
                    for i in 0..c {
                        for j in i..c {
                            out[dest] = out[base + i].mul_in::<E>(out[base + j]);
                            dest += 1;
                        }
                    }
                }
                _ => (),
            }
        }
        ctl.report((z + 1) as f64 / s as f64, "segmented reduction");
    }
    Ok((out, status))
}
