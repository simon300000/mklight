//! Deterministic resident collectives. Layouts and fixed-order algorithms
//! are part of the portable kernel contract; `spec` and `cpu` are the CPU-f32
//! specification that Metal reproduces bit for bit (QR included).
//!
//! Active records are finite and double-float pairs normalised; upstream
//! producers own these invariants. Masked records are never loaded.
//! Segments are half-open ranges of a u32 offsets array, checked on the
//! device (status 1 and NaN outputs for an invalid range); empty segments
//! are legal. Every segment term uses the fixed tree of [`crate::reduce`].
//! Stable ordering is by (key, logical ID, original position); float keys
//! use numeric order, signed zeros tie and NaNs follow all numbers.
//! Compaction keeps input order and clears the tail (IDs u32::MAX). Scans
//! use 256-element doubling blocks, recursively scanned block totals and
//! wrapping u32 or double-float addition (f32 inputs publish double-float).
//!
//! QR is column-equilibrated, pivoted modified Gram-Schmidt with two
//! projections per step, in double-float. The largest residual norm wins,
//! ties take the lower original column; rank stops at norm <= tolerance *
//! first pivot norm. Factors [batch, rows+cols+2, cols] hold thin Q, upper R,
//! the column scales and pivot indices: A[:,pivot[j]] = scale[pivot[j]] (QR)[:,j].
//! The condition is ||R||_inf ||R^-1||_inf of the equilibrated matrix
//! (infinite when deficient), an estimate rather than a singular-value ratio.
//! Rank-deficient or non-finite fits publish NaN coefficients.
use crate::resident::{
    Access, BufferDesc, BufferId, ElementType as Ty, PlanView, ResidentPlan, bad,
};
use mklight_core::MkResult;

mod cpu;
pub mod spec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SegmentKind {
    Sum = crate::consts::MKL_COLLECT_SUM,
    /// Per channel: sum, count, minimum, maximum. Empty extrema are +/-inf.
    Summary = crate::consts::MKL_COLLECT_SUMMARY,
    /// Values [2,n] or [3,n]: x, y, optional nonnegative weight. Reference [segments,2],
    /// default zero. Output [segments,10]: count, weight, chief-relative mean
    /// x/y, sum(w*r_origin^2), sum(w*dx^2), sum(w*dx*dy), sum(w*dy^2),
    /// max(r_origin^2), max(r_centroid^2). Maxima are zero without positive
    /// weights; means are NaN at zero total weight. Central moments use a second pass.
    Moments2 = crate::consts::MKL_COLLECT_MOMENTS2,
    /// First sum all split-child channels within each source segment, then
    /// publish those sums followed by their upper-triangular cross products.
    /// No second moment is ever formed from individual split children.
    SourceProducts = crate::consts::MKL_COLLECT_SOURCE_PRODUCTS,
}
#[derive(Clone, Debug)]
pub struct SegmentReduce {
    /// Channel-major [channels,count], f32 or double-float.
    pub values: PlanView,
    pub offsets: PlanView,
    /// Keep records whose mask equals mask_value; omitted means all records.
    pub mask: Option<PlanView>,
    pub mask_value: u32,
    /// Optional separate moment weights [count]; otherwise third channel or 1.
    pub weights: Option<PlanView>,
    /// Upper bound on a segment length (<= count); checked on the device. It
    /// selects the dispatch path and tiles scratch, never the result bits.
    pub max_segment_len: usize,
    pub reference: Option<PlanView>,
    pub mode: crate::ReduceMode,
    pub kind: SegmentKind,
    pub output: BufferId,
    /// U32 [segments], 0 valid / 1 invalid offsets.
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub struct StableCompact {
    /// Record-major [count,width] (or [count]), any element type.
    pub values: PlanView,
    pub ids: PlanView,
    pub mask: PlanView,
    pub output: BufferId,
    pub output_ids: BufferId,
    pub count: BufferId,
}
#[derive(Clone, Debug)]
pub struct StableSort {
    /// U32, F32 or DoubleFloat [count].
    pub keys: PlanView,
    pub ids: PlanView,
    pub values: Option<PlanView>,
    pub output_keys: BufferId,
    pub output_ids: BufferId,
    pub output_values: Option<BufferId>,
}
#[derive(Clone, Debug)]
pub struct Scan {
    pub values: PlanView,
    pub inclusive: bool,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub struct QrFit {
    /// Row-major [batch,rows,cols], rows >= cols > 0; f32 or double-float.
    pub matrix: PlanView,
    /// [batch,rows], same input type as matrix.
    pub rhs: PlanView,
    pub rank_tolerance: f32,
    pub factors: BufferId,
    /// DoubleFloat [batch,cols], in original column order.
    pub solution: BufferId,
    /// DoubleFloat [batch,4]: rank, condition estimate, residual norm, status
    /// (0 full rank, 1 rank deficient, 2 non-finite input or arithmetic).
    pub rank_condition: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    SegmentReduce(SegmentReduce),
    StableCompact(StableCompact),
    StableSort(StableSort),
    Scan(Scan),
    QrFit(QrFit),
}

pub fn segment_shape(r: &SegmentReduce) -> Vec<usize> {
    let c = r.values.desc.shape[0];
    let s = r.offsets.desc.shape[0] - 1;
    match r.kind {
        SegmentKind::Sum => vec![c, s],
        SegmentKind::Summary => vec![c, s, 4],
        SegmentKind::Moments2 => vec![s, 10],
        SegmentKind::SourceProducts => vec![s, c + c * (c + 1) / 2],
    }
}
fn input(v: &PlanView, types: &[Ty], ndim: usize) -> MkResult<()> {
    if !types.contains(&v.desc.element)
        || v.desc.shape.len() != ndim
        || v.stride != 1
        || v.desc.shape.iter().any(|&d| d > u32::MAX as usize)
        || v.desc.len()? > u32::MAX as usize / v.desc.element.words()
    {
        return Err(bad("collective input type, shape or stride"));
    }
    Ok(())
}
fn output(p: &ResidentPlan, id: BufferId, t: Ty, shape: impl Into<Vec<usize>>) -> MkResult<()> {
    let expected = BufferDesc::new(t, shape);
    if expected.len()? > u32::MAX as usize / t.words() || p.desc(id)? != &expected {
        return Err(bad("collective output layout"));
    }
    Ok(())
}
fn records(v: &PlanView, n: usize) -> MkResult<()> {
    input(
        v,
        &[Ty::F32, Ty::U32, Ty::DoubleFloat, Ty::Complex32],
        v.desc.shape.len(),
    )?;
    if !(1..=2).contains(&v.desc.shape.len()) || v.desc.shape[0] != n {
        return Err(bad("collective record layout"));
    }
    Ok(())
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    let numeric = &[Ty::F32, Ty::DoubleFloat];
    match r {
        Request::SegmentReduce(r) => {
            input(&r.values, numeric, 2)?;
            input(&r.offsets, &[Ty::U32], 1)?;
            let [c, _] = r.values.desc.shape[..] else {
                unreachable!()
            };
            if c == 0 || r.max_segment_len > r.values.desc.shape[1] || r.offsets.desc.shape[0] == 0
            {
                return Err(bad("segment dimensions"));
            }
            let s = r.offsets.desc.shape[0] - 1;
            if let Some(v) = &r.reference {
                input(v, numeric, 2)?;
                if r.kind != SegmentKind::Moments2 || v.desc.shape != [s, 2] {
                    return Err(bad("segment reference layout"));
                }
            }
            for (v, types) in [
                (r.mask.as_ref(), &[Ty::U32][..]),
                (r.weights.as_ref(), numeric),
            ] {
                if let Some(v) = v {
                    input(v, types, 1)?;
                    if v.desc.shape != [r.values.desc.shape[1]] {
                        return Err(bad("segment mask/weight length"));
                    }
                }
            }
            if r.weights.is_some() && r.kind != SegmentKind::Moments2 {
                return Err(bad("weights require moments"));
            }
            if r.kind == SegmentKind::Moments2 && !(2..=3).contains(&c) {
                return Err(bad("moments need x,y,weight"));
            }
            output(p, r.output, Ty::DoubleFloat, segment_shape(r))?;
            output(p, r.status, Ty::U32, [s])?;
        }
        Request::StableCompact(r) => {
            input(&r.ids, &[Ty::U32], 1)?;
            let n = r.ids.desc.shape[0];
            records(&r.values, n)?;
            input(&r.mask, &[Ty::U32], 1)?;
            if r.mask.desc.shape != [n] {
                return Err(bad("compact mask length"));
            }
            output(
                p,
                r.output,
                r.values.desc.element,
                r.values.desc.shape.clone(),
            )?;
            output(p, r.output_ids, Ty::U32, [n])?;
            output(p, r.count, Ty::U32, [1])?;
        }
        Request::StableSort(r) => {
            input(&r.keys, &[Ty::U32, Ty::F32, Ty::DoubleFloat], 1)?;
            input(&r.ids, &[Ty::U32], 1)?;
            let n = r.keys.desc.shape[0];
            if r.ids.desc.shape != [n] {
                return Err(bad("sort ID length"));
            }
            output(p, r.output_keys, r.keys.desc.element, [n])?;
            output(p, r.output_ids, Ty::U32, [n])?;
            match (&r.values, r.output_values) {
                (Some(v), Some(o)) => {
                    records(v, n)?;
                    output(p, o, v.desc.element, v.desc.shape.clone())?;
                }
                (None, None) => (),
                _ => return Err(bad("sort value pairing")),
            }
        }
        Request::Scan(r) => {
            input(&r.values, &[Ty::U32, Ty::F32, Ty::DoubleFloat], 1)?;
            let t = if r.values.desc.element == Ty::U32 {
                Ty::U32
            } else {
                Ty::DoubleFloat
            };
            output(p, r.output, t, r.values.desc.shape.clone())?;
        }
        Request::QrFit(r) => {
            input(&r.matrix, numeric, 3)?;
            input(&r.rhs, &[r.matrix.desc.element], 2)?;
            let [b, m, n] = r.matrix.desc.shape[..] else {
                unreachable!()
            };
            if n == 0
                || m < n
                || r.rhs.desc.shape != [b, m]
                || !r.rank_tolerance.is_finite()
                || r.rank_tolerance <= 0.0
                || r.rank_tolerance >= 1.0
            {
                return Err(bad("QR dimensions or tolerance"));
            }
            if 2 * b * (n + 1) * m > u32::MAX as usize {
                return Err(bad("QR work extent"));
            }
            output(p, r.factors, Ty::DoubleFloat, [b, m + n + 2, n])?;
            output(p, r.solution, Ty::DoubleFloat, [b, n])?;
            output(p, r.rank_condition, Ty::DoubleFloat, [b, 4])?;
        }
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    let (reads, writes) = match r {
        Request::SegmentReduce(r) => {
            let mut v = vec![r.values.clone(), r.offsets.clone()];
            v.extend(r.reference.clone());
            v.extend(r.mask.clone());
            v.extend(r.weights.clone());
            (v, vec![r.output, r.status])
        }
        Request::StableCompact(r) => (
            vec![r.values.clone(), r.ids.clone(), r.mask.clone()],
            vec![r.output, r.output_ids, r.count],
        ),
        Request::StableSort(r) => {
            let mut v = vec![r.keys.clone(), r.ids.clone()];
            v.extend(r.values.clone());
            let mut o = vec![r.output_keys, r.output_ids];
            o.extend(r.output_values);
            (v, o)
        }
        Request::Scan(r) => (vec![r.values.clone()], vec![r.output]),
        Request::QrFit(r) => (
            vec![r.matrix.clone(), r.rhs.clone()],
            vec![r.factors, r.solution, r.rank_condition],
        ),
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
pub use cpu::execute;
