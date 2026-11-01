//! Device bin/signature queries followed by P3 source products. Host code may
//! enumerate integer keys for result labels; it never sums optical records.
use super::*;
use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Query {
    pub history: History,
    pub kind: u32,
    pub slot: u32,
    pub key: u32,
    pub other: u32,
}
impl Default for Query {
    fn default() -> Self {
        Self::zeroed()
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TallyParams {
    pub half: [Df32; 2],
    pub nx: u32,
    pub ny: u32,
    pub owned: u32,
    pub covered: u32,
    pub count: u32,
    pub capacity: u32,
    pub sources: u32,
    pub queries: u32,
}
#[derive(Clone, Debug)]
pub enum Tally {
    Gather {
        hits: PlanView,
        indices: PlanView,
        output: BufferId,
    },
    Tags {
        hits: PlanView,
        ledger: PlanView,
        coverage: PlanView,
        params: TallyParams,
        output: BufferId,
    },
    /// `width` 0: raw sums [4, queries × sources]; 2: per-source (sum,
    /// sum²) [2 × queries, sources]; 14: sums and their upper-triangle
    /// products [14 × queries, sources].
    Terms {
        width: u32,
        hits: PlanView,
        ledger: PlanView,
        tags: PlanView,
        queries: PlanView,
        params: TallyParams,
        output: BufferId,
    },
    Transpose {
        input: PlanView,
        sources: usize,
        queries: usize,
        width: usize,
        output: BufferId,
    },
}
impl Tally {
    pub fn accesses(&self) -> Access {
        match self {
            Self::Gather {
                hits,
                indices,
                output,
            } => Access {
                reads: vec![hits.clone(), indices.clone()],
                writes: vec![*output],
                ..Default::default()
            },
            Self::Tags {
                hits,
                ledger,
                coverage,
                output,
                ..
            } => Access {
                reads: vec![hits.clone(), ledger.clone(), coverage.clone()],
                writes: vec![*output],
                ..Default::default()
            },
            Self::Terms {
                hits,
                ledger,
                tags,
                queries,
                output,
                ..
            } => Access {
                reads: vec![hits.clone(), ledger.clone(), tags.clone(), queries.clone()],
                writes: vec![*output],
                ..Default::default()
            },
            Self::Transpose { input, output, .. } => Access {
                reads: vec![input.clone()],
                writes: vec![*output],
                ..Default::default()
            },
        }
    }
    pub fn validate(&self, p: &ResidentPlan) -> MkResult<()> {
        match self {
            Self::Gather {
                hits,
                indices,
                output,
            } => {
                check::<Detection>(hits, p)?;
                let n = check::<u32>(indices, p)?;
                if p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [2, n]) {
                    return Err(bad("stray gathered terms"));
                }
            }
            Self::Tags {
                hits,
                ledger,
                coverage,
                params: t,
                output,
            } => {
                let n = check::<Detection>(hits, p)?;
                let s = check::<Ledger>(ledger, p)?;
                check::<History>(coverage, p)?;
                if n != t.count as usize
                    || s != t.sources as usize
                    || n != s * t.capacity as usize
                    || p.desc(*output)? != &BufferDesc::new(ElementType::U32, [n, 4])
                {
                    return Err(bad("stray tally tags"));
                }
            }
            Self::Terms {
                width,
                hits,
                ledger,
                tags,
                queries,
                params: t,
                output,
            } => {
                let n = check::<Detection>(hits, p)?;
                let s = check::<Ledger>(ledger, p)?;
                let q = check::<Query>(queries, p)?;
                if n != t.count as usize
                    || s != t.sources as usize
                    || q != t.queries as usize
                    || n != s * t.capacity as usize
                    || tags.desc != BufferDesc::new(ElementType::U32, [n, 4])
                    || !matches!(width, 0 | 2 | 14)
                    || p.desc(*output)?
                        != &BufferDesc::new(
                            ElementType::DoubleFloat,
                            if *width == 0 {
                                vec![4, q * s]
                            } else {
                                vec![*width as usize * q, s]
                            },
                        )
                {
                    return Err(bad("stray tally terms"));
                }
            }
            Self::Transpose {
                input,
                sources: s,
                queries: q,
                width: w,
                output,
            } => {
                if input.desc != BufferDesc::new(ElementType::DoubleFloat, [s * q, *w])
                    || p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [w * q, *s])
                {
                    return Err(bad("stray tally transpose"));
                }
            }
        }
        Ok(())
    }
}
fn covered(h: History, coverage: &[History]) -> bool {
    if h.overflow != 0 {
        return false;
    }
    let reflections: Vec<_> = h.events[..h.length as usize]
        .iter()
        .filter(|&&e| e >> 28 == 1 || e >> 28 == 2)
        .map(|e| e & 0xfffffff)
        .collect();
    let mut orders = Vec::new();
    for (j, &[tag, order]) in h.orders[..h.order_count as usize].iter().enumerate() {
        if tag & (1 << 17) != 0 {
            continue;
        }
        let surface = tag & 65535;
        let visit = h.orders[..j]
            .iter()
            .filter(|v| v[0] & 65535 == surface)
            .count() as u32;
        orders.push([surface | visit << 16, order]);
    }
    coverage.iter().any(|c| {
        c.length as usize == reflections.len()
            && c.events[..c.length as usize] == reflections
            && c.order_count as usize == orders.len()
            && orders
                .iter()
                .all(|o| c.orders[..c.order_count as usize].contains(o))
    })
}
pub fn same_signature(a: &History, b: &History) -> bool {
    a.length == b.length
        && a.overflow == b.overflow
        && a.order_count == b.order_count
        && a.events == b.events
        && a.orders == b.orders
}
fn pixel(h: Detection, p: TallyParams) -> u32 {
    let x = (h.x + p.half[0]) / (p.half[0] * Df32::from_f32(2.));
    let y = (h.y + p.half[1]) / (p.half[1] * Df32::from_f32(2.));
    if x.to_f64() < 0. || x.to_f64() > 1. || y.to_f64() < 0. || y.to_f64() > 1. {
        return u32::MAX;
    }
    let ix = ((x * Df32::from_f32(p.nx as f32)).to_f64() as u32).min(p.nx - 1);
    let iy = ((y * Df32::from_f32(p.ny as f32)).to_f64() as u32).min(p.ny - 1);
    iy * p.nx + ix
}
fn selected(slot: u32, c: u32, tag: [u32; 4]) -> bool {
    match slot {
        0..=4 => c == slot,
        5 => tag[2] != 0,
        6 => tag[3] != 0,
        7 => true,
        8 => c == 1 && tag[3] == 0,
        _ => false,
    }
}
pub(super) fn execute(t: &Tally, x: &mut cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    match t {
        Tally::Gather {
            hits,
            indices,
            output,
        } => {
            let n = indices.desc.shape[0];
            for i in 0..n {
                if i % 1024 == 0 {
                    ctl.check()?;
                }
                let id = rows::<u32>(x, indices)[i] as usize;
                let w = rows::<Detection>(x, hits)[id].weight;
                let out = bytemuck::cast_slice_mut::<u32, Df32>(x.words_mut(*output));
                out[i] = w;
                out[n + i] = Df32::from_f32(1.);
            }
        }
        Tally::Tags {
            hits,
            ledger,
            coverage,
            params: p,
            output,
        } => {
            for i in 0..p.count as usize {
                if i % 1024 == 0 {
                    ctl.check()?;
                }
                let l = rows::<Ledger>(x, ledger)[i / p.capacity as usize];
                let mut tag = [u32::MAX, 0, 0, 0];
                if i % (p.capacity as usize) < l.hits as usize {
                    let h = rows::<Detection>(x, hits)[i];
                    let owned = p.owned & (1 << h.class) != 0;
                    let uncovered = !owned
                        && h.class == 1
                        && p.covered != 0
                        && !covered(h.history, rows(x, coverage));
                    tag = [
                        pixel(h, *p),
                        h.class,
                        (owned || uncovered) as u32,
                        uncovered as u32,
                    ];
                }
                x.words_mut(*output)[4 * i..4 * i + 4].copy_from_slice(&tag);
            }
        }
        Tally::Terms {
            width,
            hits,
            ledger,
            tags,
            queries,
            params: p,
            output,
        } => {
            let n = p.sources as usize;
            let len = n * p.queries as usize;
            for i in 0..len {
                if i % 1024 == 0 {
                    ctl.check()?;
                }
                let q = rows::<Query>(x, queries)[i / n];
                let source = i % n;
                let l = rows::<Ledger>(x, ledger)[source];
                let mut sum = [Df32::ZERO; 4];
                for j in 0..if q.kind == 3 { 1 } else { l.hits as usize } {
                    let k = source * p.capacity as usize + j;
                    let mut values = [Df32::ZERO; 4];
                    if q.kind == 3 && j == 0 {
                        values[0] = match q.key {
                            0..=20 => l.sums[q.key as usize],
                            21 => l.sums[16] - l.sums[17] + l.sums[18] - l.sums[19],
                            22 => Df32::from_f32(l.events as f32),
                            23 => {
                                if l.wavelength == q.other {
                                    l.sums[20]
                                } else {
                                    Df32::ZERO
                                }
                            }
                            _ => unreachable!(),
                        };
                    } else if j < l.hits as usize {
                        let h = rows::<Detection>(x, hits)[k];
                        let tag: [u32; 4] =
                            rows::<u32>(x, tags)[4 * k..4 * k + 4].try_into().unwrap();
                        let keep = match q.kind {
                            0 => selected(q.slot, h.class, tag),
                            1 => tag[0] == q.key && selected(q.slot, h.class, tag),
                            2 => h.class == q.slot && same_signature(&h.history, &q.history),
                            4 => h.class == 1 && h.history.reflections.min(8) == q.key,
                            5 => h.class == 1 && h.history.first == [q.key, q.other],
                            6 => h.class == q.slot && h.wavelength == q.key,
                            7 => h.class == q.slot,
                            _ => false,
                        };
                        if keep {
                            values[0] = if q.kind == 7 {
                                Df32::from_f32(1.)
                            } else {
                                h.weight
                            };
                            if q.kind == 0 {
                                values[1] = h.weight * h.x;
                                values[2] = h.weight * h.y;
                                values[3] = h.weight * (h.x * h.x + h.y * h.y);
                            }
                            if q.kind == 2 {
                                values[1] = Df32::from_f32(1.);
                            }
                        }
                    }
                    for c in 0..4 {
                        sum[c] = sum[c] + values[c];
                    }
                }
                let ftz = x.flush_to_zero;
                let out: &mut [Df32] = bytemuck::cast_slice_mut(x.words_mut(*output));
                if *width != 0 {
                    let sum = sum.map(|v| {
                        if ftz {
                            v.add_in::<crate::Ftz>(Df32::ZERO)
                        } else {
                            v
                        }
                    });
                    let product = |a: usize, b: usize| {
                        if ftz {
                            sum[a].mul_in::<crate::Ftz>(sum[b])
                        } else {
                            sum[a] * sum[b]
                        }
                    };
                    let (w, query, source) = (*width as usize, i / n, i % n);
                    let mut channels = vec![sum[0]];
                    if w == 2 {
                        channels.push(product(0, 0));
                    } else {
                        channels.extend(&sum[1..]);
                        for a in 0..4 {
                            channels.extend((a..4).map(|b| product(a, b)));
                        }
                    }
                    for (c, v) in channels.into_iter().enumerate() {
                        out[(w * query + c) * n + source] = v;
                    }
                } else {
                    for c in 0..4 {
                        out[c * len + i] = sum[c];
                    }
                }
            }
        }
        Tally::Transpose {
            input,
            sources: n,
            queries: q,
            width: w,
            output,
        } => {
            for c in 0..w * q {
                for i in 0..*n {
                    let value = rows::<Df32>(x, input)[((c / w) * n + i) * w + c % w];
                    bytemuck::cast_slice_mut::<u32, Df32>(x.words_mut(*output))[c * n + i] = value;
                }
            }
        }
    }
    Ok(())
}
