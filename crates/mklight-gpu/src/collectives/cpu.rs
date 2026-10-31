use super::*;
use crate::{
    Df32 as D,
    fenv::{FpEnv, Ftz, Ieee},
    pending::JobControl,
    reduce,
    resident::cpu::Execution,
};
use std::cmp::Ordering;

struct Values {
    words: Vec<u32>,
    ty: Ty,
}
impl Values {
    fn read(x: &Execution, v: &PlanView) -> Self {
        let w = v.desc.element.words();
        Self {
            words: x.words(v.buffer)[v.offset * w..(v.offset + v.desc.len().unwrap()) * w].to_vec(),
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
fn put(x: &mut Execution, id: BufferId, v: &[D]) {
    for (dest, q) in x.words_mut(id).chunks_exact_mut(2).zip(v) {
        dest[0] = q.hi.to_bits();
        dest[1] = q.lo.to_bits();
    }
}
pub fn execute(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    // P0 exposes the emulation environment in its immutable build identity.
    // Keep the frozen router unchanged; both build strings are pinned in cpu.rs.
    if x.cache_key("collectives")
        .system
        .kernel_build
        .ends_with("flush-to-zero")
    {
        execute_in::<Ftz>(r, x, ctl)
    } else {
        execute_in::<Ieee>(r, x, ctl)
    }
}
pub fn execute_in<E: FpEnv>(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    match r {
        Request::SegmentReduce(r) => segments::<E>(r, x, ctl)?,
        Request::Scan(r) => {
            let a = Values::read(x, &r.values);
            if a.ty == Ty::U32 {
                let v = spec::scan_u32(&a.words, ctl)?;
                let dst = x.words_mut(r.output);
                if r.inclusive {
                    dst.copy_from_slice(&v);
                } else if !v.is_empty() {
                    dst[0] = 0;
                    dst[1..].copy_from_slice(&v[..v.len() - 1]);
                }
            } else {
                let vals: Vec<_> = (0..r.values.desc.len()?).map(|i| a.d(i)).collect();
                let mut v = spec::scan_df::<E>(&vals, ctl)?;
                if !r.inclusive && !v.is_empty() {
                    v.rotate_right(1);
                    v[0] = D::ZERO;
                }
                put(x, r.output, &v);
            }
        }
        Request::StableCompact(r) => {
            let a = Values::read(x, &r.values);
            let ids = Values::read(x, &r.ids);
            let mask = Values::read(x, &r.mask);
            let n = ids.words.len();
            let width = if n == 0 { 0 } else { a.words.len() / n };
            let keep: Vec<_> = mask
                .words
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| (m != 0).then_some(i))
                .collect();
            x.words_mut(r.output).fill(0);
            x.words_mut(r.output_ids).fill(u32::MAX);
            for (j, &i) in keep.iter().enumerate() {
                if j % 4096 == 0 {
                    ctl.check()?;
                    ctl.report(j as f64 / n.max(1) as f64, "stable compact");
                }
                x.words_mut(r.output)[j * width..(j + 1) * width]
                    .copy_from_slice(&a.words[i * width..(i + 1) * width]);
                x.words_mut(r.output_ids)[j] = ids.words[i];
            }
            x.words_mut(r.count)[0] = keep.len() as u32;
        }
        Request::StableSort(r) => {
            let k = Values::read(x, &r.keys);
            let ids = Values::read(x, &r.ids);
            let vals = r.values.as_ref().map(|v| Values::read(x, v));
            let n = ids.words.len();
            let order = spec::permutation(
                n,
                |a, b| {
                    let cmp = if k.ty == Ty::U32 {
                        k.words[a].cmp(&k.words[b])
                    } else {
                        spec::df_cmp(k.d(a), k.d(b))
                    };
                    cmp.then(ids.words[a].cmp(&ids.words[b])).then(a.cmp(&b))
                },
                ctl,
            )?;
            let kw = k.ty.words();
            let vw = vals
                .as_ref()
                .map(|v| if n == 0 { 0 } else { v.words.len() / n })
                .unwrap_or(0);
            for (j, &i) in order.iter().enumerate() {
                if j % 4096 == 0 {
                    ctl.check()?;
                }
                x.words_mut(r.output_keys)[j * kw..(j + 1) * kw]
                    .copy_from_slice(&k.words[i * kw..(i + 1) * kw]);
                x.words_mut(r.output_ids)[j] = ids.words[i];
                if let (Some(v), Some(o)) = (&vals, r.output_values) {
                    x.words_mut(o)[j * vw..(j + 1) * vw]
                        .copy_from_slice(&v.words[i * vw..(i + 1) * vw]);
                }
            }
        }
        Request::QrFit(r) => {
            let a = Values::read(x, &r.matrix);
            let b = Values::read(x, &r.rhs);
            let [batch, m, n] = r.matrix.desc.shape[..] else {
                unreachable!()
            };
            for z in 0..batch {
                let aa: Vec<_> = (0..m * n).map(|i| a.d(z * m * n + i)).collect();
                let bb: Vec<_> = (0..m).map(|i| b.d(z * m + i)).collect();
                let fit = spec::qr::<E>(
                    &aa,
                    &bb,
                    m,
                    n,
                    r.rank_tolerance,
                    &ctl.subrange(z as f64 / batch as f64, (z + 1) as f64 / batch as f64),
                )?;
                for (id, src, offset) in [
                    (r.factors, fit.factors.as_slice(), z * (m + n + 2) * n),
                    (r.solution, fit.solution.as_slice(), z * n),
                    (r.rank_condition, fit.diagnostics.as_slice(), z * 4),
                ] {
                    for (i, q) in src.iter().enumerate() {
                        let out = x.words_mut(id);
                        out[2 * (offset + i)] = q.hi.to_bits();
                        out[2 * (offset + i) + 1] = q.lo.to_bits();
                    }
                }
            }
        }
    }
    ctl.check()?;
    ctl.report(1.0, "collectives");
    Ok(())
}
fn segments<E: FpEnv>(r: &SegmentReduce, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    let vals = Values::read(x, &r.values);
    let offsets = Values::read(x, &r.offsets);
    let mask = r.mask.as_ref().map(|v| Values::read(x, v));
    let weights = r.weights.as_ref().map(|v| Values::read(x, v));
    let refs = r.reference.as_ref().map(|v| Values::read(x, v));
    let [c, n] = r.values.desc.shape[..] else {
        unreachable!()
    };
    let s = offsets.words.len() - 1;
    let mut out = vec![D::ZERO; segment_shape(r).iter().product()];
    for z in 0..s {
        ctl.check()?;
        let (a, b) = (offsets.words[z] as usize, offsets.words[z + 1] as usize);
        let valid = a <= b && b <= n && b - a <= r.max_segment_len;
        x.words_mut(r.status)[z] = u32::from(!valid);
        if !valid {
            match r.kind {
                SegmentKind::Sum => {
                    for ch in 0..c {
                        out[ch * s + z] = D::from_f32(f32::NAN);
                    }
                }
                SegmentKind::Summary => {
                    for ch in 0..c {
                        out[(ch * s + z) * 4..(ch * s + z + 1) * 4].fill(D::from_f32(f32::NAN));
                    }
                }
                _ => {
                    let w = out.len() / s;
                    out[z * w..(z + 1) * w].fill(D::from_f32(f32::NAN));
                }
            }
            continue;
        }
        let keep = |i: usize| {
            mask.as_ref()
                .map(|v| v.words[a + i] == r.mask_value)
                .unwrap_or(true)
        };
        let get = |ch: usize, i: usize| {
            if keep(i) {
                vals.d(ch * n + a + i)
            } else {
                D::ZERO
            }
        };
        let weight = |i: usize| {
            if keep(i) {
                weights
                    .as_ref()
                    .map(|v| v.d(a + i))
                    .unwrap_or_else(|| if c == 3 { get(2, i) } else { D::from_f32(1.0) })
            } else {
                D::ZERO
            }
        };
        let sum = |f: &dyn Fn(usize) -> D| {
            reduce::mapped_in::<E>(r.mode, b - a, |i| if keep(i) { f(i) } else { D::ZERO }, ctl)
        };
        let count = reduce::mapped_in::<E>(
            crate::ReduceMode::DoubleFloat,
            b - a,
            |i| D::from_f32(if keep(i) { 1.0 } else { 0.0 }),
            ctl,
        )?;
        match r.kind {
            SegmentKind::Sum | SegmentKind::Summary => {
                for ch in 0..c {
                    let v = sum(&|i| get(ch, i))?;
                    if r.kind == SegmentKind::Sum {
                        out[ch * s + z] = v;
                    } else {
                        let mut lo = D::from_f32(f32::INFINITY);
                        let mut hi = D::from_f32(f32::NEG_INFINITY);
                        for i in 0..b - a {
                            if !keep(i) {
                                continue;
                            }
                            let v = get(ch, i);
                            if v.hi.is_nan() {
                                lo = v;
                                hi = v;
                                break;
                            }
                            if spec::df_cmp(v, lo) == Ordering::Less {
                                lo = v;
                            }
                            if spec::df_cmp(v, hi) == Ordering::Greater {
                                hi = v;
                            }
                        }
                        out[(ch * s + z) * 4..(ch * s + z + 1) * 4]
                            .copy_from_slice(&[v, count, lo, hi]);
                    }
                }
            }
            SegmentKind::SourceProducts => {
                let w = c + c * (c + 1) / 2;
                let start = z * w;
                for ch in 0..c {
                    out[start + ch] = sum(&|i| get(ch, i))?;
                }
                let mut j = start + c;
                for a in 0..c {
                    for b in a..c {
                        out[j] = out[start + a].mul_in::<E>(out[start + b]);
                        j += 1;
                    }
                }
            }
            SegmentKind::Moments2 => {
                let origin = [
                    refs.as_ref().map(|v| v.d(2 * z)).unwrap_or(D::ZERO),
                    refs.as_ref().map(|v| v.d(2 * z + 1)).unwrap_or(D::ZERO),
                ];
                let q = |ch: usize, i: usize| get(ch, i).add_in::<E>(-(origin[ch]));
                let wt = sum(&|i| weight(i))?;
                let mx = sum(&|i| q(0, i).mul_in::<E>(weight(i)))?;
                let my = sum(&|i| q(1, i).mul_in::<E>(weight(i)))?;
                let mu = if wt.hi > 0.0 {
                    [mx.div_in::<E>(wt), my.div_in::<E>(wt)]
                } else {
                    [D::from_f32(f32::NAN); 2]
                };
                let central = |ch: usize, i: usize| q(ch, i).add_in::<E>(-(mu[ch]));
                let radius = |i: usize, cent: bool| {
                    let a = if cent { central(0, i) } else { q(0, i) };
                    let b = if cent { central(1, i) } else { q(1, i) };
                    a.mul_in::<E>(a).add_in::<E>(b.mul_in::<E>(b))
                };
                let sr = sum(&|i| weight(i).mul_in::<E>(radius(i, false)))?;
                let xx = sum(&|i| weight(i).mul_in::<E>(central(0, i).mul_in::<E>(central(0, i))))?;
                let xy = sum(&|i| weight(i).mul_in::<E>(central(0, i).mul_in::<E>(central(1, i))))?;
                let yy = sum(&|i| weight(i).mul_in::<E>(central(1, i).mul_in::<E>(central(1, i))))?;
                let mut maxima = [D::ZERO; 2];
                for i in 0..b - a {
                    if weight(i).hi > 0.0 {
                        for (j, max) in maxima.iter_mut().enumerate() {
                            let v = radius(i, j == 1);
                            if spec::df_cmp(v, *max) == Ordering::Greater {
                                *max = v;
                            }
                        }
                    }
                }
                out[z * 10..z * 10 + 10].copy_from_slice(&[
                    count, wt, mu[0], mu[1], sr, xx, xy, yy, maxima[0], maxima[1],
                ]);
            }
        }
        ctl.report((z + 1) as f64 / s as f64, "segmented reduction");
    }
    put(x, r.output, &out);
    Ok(())
}
