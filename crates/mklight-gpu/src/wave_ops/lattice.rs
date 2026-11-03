//! Device pupil probes, disk masking and batched boundary bisection.
use super::*;
use crate::aiming::Sample;
#[derive(Clone, Debug)]
pub enum Lattice {
    /// Radiometric [system 0, global wavelength] ray indices from chunk-local
    /// [system, slot] indices through a [chunks, MAX_WAVELENGTHS] table.
    Wavelength {
        indices: PlanView,
        table: PlanView,
        output: BufferId,
    },
    /// `copies` consecutive guard-ringed lattices; copy q aims field row q.
    Generate {
        side: usize,
        copies: usize,
        samples: BufferId,
    },
    Survival {
        samples: PlanView,
        status: PlanView,
        output: BufferId,
    },
    /// Edges are [inside x/y, outside x/y], probed at field row `fields[k]`.
    /// State stores a,b,lo,hi,trial,done. `source` (u32 [n]) names the
    /// previous round's probe (`16k + j`) that became the surviving end `lo`,
    /// or `u32::MAX` when it did not move, so its traced ray is kept rather
    /// than traced again (a re-aimed ray at the rim may clip).
    Rim {
        edges: PlanView,
        fields: PlanView,
        previous: Option<PlanView>,
        status: Option<PlanView>,
        tolerance: Df32,
        final_point: bool,
        state: BufferId,
        samples: BufferId,
        source: BufferId,
    },
    /// Per edge k, copy column `source[k]` of every probe channel into
    /// output column k; `u32::MAX` keeps column k of `previous` (zero
    /// without `previous`). Channels are [planes, N] (or [N, row] when
    /// `rows`); outputs have `source.len()` columns.
    Keep {
        channels: Vec<(PlanView, bool)>,
        previous: Option<Vec<PlanView>>,
        source: PlanView,
        outputs: Vec<BufferId>,
    },
    /// Copy ray ranges of batched word channels into per-range buffers.
    /// Channels are [planes, N] (or [N, row] when `rows`); outputs are
    /// channel-major: output `c * ranges.len() + r` holds range r of channel c.
    Split {
        channels: Vec<(PlanView, bool)>,
        ranges: Vec<[usize; 2]>,
        outputs: Vec<BufferId>,
    },
    /// Concatenate plane-major parts `[planes, m_i]` (or `[m_i]`) of one
    /// element type along their last axis into `[planes, Σ m_i]`.
    Concat {
        parts: Vec<PlanView>,
        output: BufferId,
    },
}
pub fn validate(r: &Lattice, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Lattice::Wavelength {
            indices,
            table,
            output,
        } => {
            let n = indices.desc.shape.get(1).copied().unwrap_or(0);
            view(indices, U32, &[2, n])?;
            let t = table.desc.shape.first().copied().unwrap_or(0);
            if t == 0 || t % crate::layout::MAX_WAVELENGTHS != 0 {
                return Err(bad("wavelength table"));
            }
            view(table, U32, &[t])?;
            out(p, *output, U32, &[2, n])
        }
        Lattice::Generate {
            side,
            copies,
            samples,
        } => {
            if !(10..=2050).contains(side) || *copies == 0 {
                return Err(bad("wave lattice side"));
            }
            out(p, *samples, U32, &[copies * side * side, 8])
        }
        Lattice::Survival {
            samples,
            status,
            output,
        } => {
            let n = samples.desc.shape[0];
            view(samples, U32, &[n, 8])?;
            view(status, U32, &[3, n])?;
            out(p, *output, U32, &[n])
        }
        Lattice::Rim {
            edges,
            fields,
            previous,
            status,
            tolerance,
            state,
            samples,
            source,
            final_point,
        } => {
            let n = edges.desc.shape[0];
            view(edges, DoubleFloat, &[n, 4])?;
            view(fields, U32, &[n])?;
            if previous.is_some() != status.is_some()
                || !tolerance.is_finite()
                || tolerance.hi <= 0.
            {
                return Err(bad("rim state/tolerance"));
            }
            if let Some(v) = previous {
                view(v, DoubleFloat, &[n, 8])?;
            }
            if let Some(v) = status {
                view(v, U32, &[3, 16 * n])?;
            }
            out(p, *state, DoubleFloat, &[n, 8])?;
            out(p, *source, U32, &[n])?;
            out(
                p,
                *samples,
                U32,
                &[if *final_point { n } else { 16 * n }, 8],
            )
        }
        Lattice::Keep {
            channels,
            previous,
            source,
            outputs,
        } => {
            let n = source.desc.shape.first().copied().unwrap_or(0);
            view(source, U32, &[n])?;
            if n == 0
                || channels.is_empty()
                || outputs.len() != channels.len()
                || previous.as_ref().is_some_and(|v| v.len() != channels.len())
            {
                return Err(bad("keep layout"));
            }
            for (c, (v, rows)) in channels.iter().enumerate() {
                let s = &v.desc.shape;
                if !(1..=2).contains(&s.len())
                    || (*rows && s.len() != 2)
                    || !matches!(v.desc.element, U32 | F32)
                {
                    return Err(bad("keep channel"));
                }
                view(v, v.desc.element, s)?;
                let mut shape = s.clone();
                shape[if *rows { 0 } else { s.len() - 1 }] = n;
                if let Some(prev) = previous {
                    view(&prev[c], v.desc.element, &shape)?;
                }
                out(p, outputs[c], v.desc.element, &shape)?;
            }
            Ok(())
        }
        Lattice::Split {
            channels,
            ranges,
            outputs,
        } => {
            if channels.is_empty() || outputs.len() != channels.len() * ranges.len() {
                return Err(bad("split layout"));
            }
            for (c, (v, rows)) in channels.iter().enumerate() {
                let s = &v.desc.shape;
                if !(1..=2).contains(&s.len())
                    || (*rows && s.len() != 2)
                    || !matches!(v.desc.element, U32 | F32)
                {
                    return Err(bad("split channel"));
                }
                view(v, v.desc.element, s)?;
                let n = if *rows { s[0] } else { s[s.len() - 1] };
                for (r, &[start, len]) in ranges.iter().enumerate() {
                    if len == 0 || start + len > n {
                        return Err(bad("split range"));
                    }
                    let mut shape = s.clone();
                    shape[if *rows { 0 } else { s.len() - 1 }] = len;
                    out(p, outputs[c * ranges.len() + r], v.desc.element, &shape)?;
                }
            }
            Ok(())
        }
        Lattice::Concat { parts, output } => {
            let Some(first) = parts.first() else {
                return Err(bad("concat parts"));
            };
            let (element, planes) = (first.desc.element, concat_part(first).0);
            let mut total = 0;
            for v in parts {
                let (p, m) = concat_part(v);
                if v.desc.element != element || p != planes || m == 0 || p == 0 {
                    return Err(bad("concat part layout"));
                }
                view(v, element, &v.desc.shape)?;
                total += m;
            }
            let mut shape = first.desc.shape.clone();
            *shape.last_mut().unwrap() = total;
            out(p, *output, element, &shape)
        }
    }
}
/// Planes and columns of a concat part.
pub fn concat_part(v: &PlanView) -> (usize, usize) {
    match v.desc.shape[..] {
        [planes, m] => (planes, m),
        [m] => (1, m),
        _ => (0, 0),
    }
}
pub fn access(r: &Lattice) -> Access {
    let (reads, writes) = match r {
        Lattice::Wavelength {
            indices,
            table,
            output,
        } => (vec![indices.clone(), table.clone()], vec![*output]),
        Lattice::Generate { samples, .. } => (vec![], vec![*samples]),
        Lattice::Survival {
            samples,
            status,
            output,
        } => (vec![samples.clone(), status.clone()], vec![*output]),
        Lattice::Rim {
            edges,
            fields,
            previous,
            status,
            state,
            samples,
            source,
            ..
        } => {
            let mut reads = vec![edges.clone(), fields.clone()];
            reads.extend(previous.clone());
            reads.extend(status.clone());
            (reads, vec![*state, *samples, *source])
        }
        Lattice::Keep {
            channels,
            previous,
            source,
            outputs,
        } => {
            let mut reads: Vec<_> = channels.iter().map(|(v, _)| v.clone()).collect();
            reads.extend(previous.iter().flatten().cloned());
            reads.push(source.clone());
            (reads, outputs.clone())
        }
        Lattice::Split {
            channels, outputs, ..
        } => (
            channels.iter().map(|(v, _)| v.clone()).collect(),
            outputs.clone(),
        ),
        Lattice::Concat { parts, output } => (parts.clone(), vec![*output]),
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
/// Source word offset, source plane stride, plane count and words per plane
/// of one split range.
pub fn split_part(
    v: &PlanView,
    rows: bool,
    [start, len]: [usize; 2],
) -> (usize, usize, usize, usize) {
    let s = &v.desc.shape;
    match (rows, &s[..]) {
        (true, &[_, w]) => (start * w, 0, 1, len * w),
        (_, &[planes, n]) => (start, n, planes, len),
        _ => (start, 0, 1, len),
    }
}
/// Columns, planes and words per row (0 for plane-major channels) of a
/// keep channel.
pub fn keep_part(v: &PlanView, rows: bool) -> (usize, usize, usize) {
    let s = &v.desc.shape;
    match (rows, &s[..]) {
        (true, &[n, w]) => (n, 1, w),
        (_, &[planes, n]) => (n, planes, 0),
        _ => (s[0], 1, 0),
    }
}
pub fn cpu(r: &Lattice, x: &mut crate::resident::cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    let one = Df32::from_f32(1.);
    match r {
        Lattice::Wavelength {
            indices,
            table,
            output,
        } => {
            let i = values::<u32>(x, indices).to_vec();
            let t = values::<u32>(x, table).to_vec();
            let n = i.len() / 2;
            let out = x.words_mut(*output);
            for k in 0..n {
                let slot = i[k] as usize * crate::layout::MAX_WAVELENGTHS + i[n + k] as usize;
                out[k] = 0;
                out[n + k] = t.get(slot).copied().unwrap_or(u32::MAX);
            }
        }
        Lattice::Generate {
            side,
            copies,
            samples,
        } => {
            let per = side * side;
            let mut a = vec![Sample::new([0.; 2], 0, 0); copies * per];
            let h = Df32::from_f32(2.) / Df32::from_f32((side - 3) as f32);
            let c = Df32::from_f32((side - 1) as f32 * 0.5);
            for (k, s) in a.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let (i, j) = (k % per % side, k % per / side);
                s.pupil = [
                    (Df32::from_f32(i as f32) - c) * h,
                    (Df32::from_f32(j as f32) - c) * h,
                ];
                s.field = (k / per) as u32;
                s.id = [k as u32, 0];
            }
            x.words_mut(*samples)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Lattice::Survival {
            samples,
            status,
            output,
        } => {
            let s: &[Sample] = bytemuck::cast_slice(values::<u32>(x, samples));
            let t = values::<u32>(x, status);
            let a: Vec<u32> = s
                .iter()
                .enumerate()
                .map(|(k, s)| {
                    u32::from(
                        t[k] == 0
                            && (s.pupil[0] * s.pupil[0] + s.pupil[1] * s.pupil[1]).to_f64()
                                <= 1. + 1e-12,
                    )
                })
                .collect();
            x.words_mut(*output).copy_from_slice(&a);
        }
        Lattice::Rim {
            edges,
            fields,
            previous,
            status,
            tolerance,
            final_point,
            state,
            samples,
            source,
        } => {
            let e = values::<Df32>(x, edges).as_chunks::<4>().0;
            let fields = values::<u32>(x, fields);
            let prev = previous
                .as_ref()
                .map(|v| values::<Df32>(x, v).as_chunks::<8>().0);
            let st = status.as_ref().map(|v| values::<u32>(x, v));
            let mut a = vec![[Df32::ZERO; 8]; e.len()];
            let mut from = vec![u32::MAX; e.len()];
            let width = if *final_point { 1 } else { 16 };
            let mut probes = vec![Sample::new([0.; 2], 0, 0); e.len() * width];
            for (k, (a, e)) in a.iter_mut().zip(e).enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                a[..4].copy_from_slice(e);
                let dx = e[2] - e[0];
                let dy = e[3] - e[1];
                let len = (dx * dx + dy * dy).sqrt();
                if let Some(prev) = prev {
                    *a = prev[k];
                    if a[7].hi == 0. {
                        if st.unwrap()[16 * k] == 0 {
                            a[4] = a[5];
                            from[k] = 16 * k as u32;
                        } else {
                            let (lo, hi) = (a[4], a[5]);
                            let span = hi - lo;
                            let (mut l, mut h) = (0, 16);
                            for _ in 0..4 {
                                let m = (l + h) / 2;
                                if st.unwrap()[16 * k + m] == 0 {
                                    l = m;
                                } else {
                                    h = m;
                                }
                            }
                            a[4] = lo + span.mul_f32(l as f32 / 16.);
                            a[5] = lo + span.mul_f32(h as f32 / 16.);
                            if l > 0 {
                                from[k] = (16 * k + l) as u32;
                            }
                        }
                        if ((a[5] - a[4]) * len).to_f64() <= tolerance.to_f64() {
                            a[7] = one;
                        }
                    }
                } else {
                    a[5] = one;
                    let outside = e[2] * e[2] + e[3] * e[3];
                    if outside.to_f64() > 1. {
                        let qa = dx * dx + dy * dy;
                        let qb = e[0] * dx + e[1] * dy;
                        let qc = e[0] * e[0] + e[1] * e[1] - one;
                        let disc = qb * qb - qa * qc;
                        let root = if disc.hi < 0. {
                            Df32::ZERO
                        } else {
                            disc.sqrt()
                        };
                        let t = if qb.hi <= 0. {
                            (-qb + root) / qa
                        } else {
                            -qc / (qb + root)
                        };
                        a[5] = if t.to_f64() < 0. {
                            Df32::ZERO
                        } else if t.to_f64() > 1. {
                            one
                        } else {
                            t
                        };
                    }
                }
                for j in 0..width {
                    let t = if *final_point || a[7].hi > 0. {
                        a[4]
                    } else if j == 0 {
                        a[5]
                    } else {
                        a[4] + (a[5] - a[4]).mul_f32(j as f32 / 16.)
                    };
                    let s = &mut probes[k * width + j];
                    s.pupil = [e[0] + dx * t, e[1] + dy * t];
                    s.field = fields[k];
                    s.id = [(k * width + j) as u32, 0];
                    if prev.is_none() {
                        let rad = (s.pupil[0] * s.pupil[0] + s.pupil[1] * s.pupil[1]).sqrt();
                        if rad.to_f64() > 1. {
                            s.pupil = s.pupil.map(|v| v / rad);
                        }
                    }
                }
            }
            x.words_mut(*state)
                .copy_from_slice(bytemuck::cast_slice(&a));
            x.words_mut(*samples)
                .copy_from_slice(bytemuck::cast_slice(&probes));
            x.words_mut(*source).copy_from_slice(&from);
        }
        Lattice::Keep {
            channels,
            previous,
            source,
            outputs,
        } => {
            let from = values::<u32>(x, source).to_vec();
            let n = from.len();
            for (c, (v, rows)) in channels.iter().enumerate() {
                ctl.check()?;
                let (big, planes, w) = keep_part(v, *rows);
                let src = x.words(v.buffer)[v.offset..].to_vec();
                let prev = previous
                    .as_ref()
                    .map(|p| x.words(p[c].buffer)[p[c].offset..].to_vec());
                let out = x.words_mut(outputs[c]);
                for (k, &s) in from.iter().enumerate() {
                    for j in 0..planes.max(w) {
                        let (o, i) = if w > 0 {
                            (k * w + j, s as usize * w + j)
                        } else {
                            (j * n + k, j * big + s as usize)
                        };
                        out[o] = if s == u32::MAX {
                            prev.as_ref().map_or(0, |p| p[o])
                        } else {
                            src[i]
                        };
                    }
                }
            }
        }
        Lattice::Split {
            channels,
            ranges,
            outputs,
        } => {
            for (c, (v, rows)) in channels.iter().enumerate() {
                for (r, &range) in ranges.iter().enumerate() {
                    ctl.check()?;
                    let (offset, stride, planes, len) = split_part(v, *rows, range);
                    let src = &x.words(v.buffer)[v.offset..];
                    let part: Vec<u32> = (0..planes)
                        .flat_map(|j| &src[offset + j * stride..offset + j * stride + len])
                        .copied()
                        .collect();
                    x.words_mut(outputs[c * ranges.len() + r])
                        .copy_from_slice(&part);
                }
            }
        }
        Lattice::Concat { parts, output } => {
            let words = parts[0].desc.element.words();
            let planes = concat_part(&parts[0]).0;
            let total: usize = parts.iter().map(|v| concat_part(v).1).sum();
            let mut out = vec![0u32; planes * total * words];
            let mut start = 0;
            for v in parts {
                ctl.check()?;
                let m = concat_part(v).1;
                let src = &x.words(v.buffer)[v.offset * words..];
                for j in 0..planes {
                    out[(j * total + start) * words..(j * total + start + m) * words]
                        .copy_from_slice(&src[j * m * words..(j + 1) * m * words]);
                }
                start += m;
            }
            x.words_mut(*output).copy_from_slice(&out);
        }
    }
    Ok(())
}
